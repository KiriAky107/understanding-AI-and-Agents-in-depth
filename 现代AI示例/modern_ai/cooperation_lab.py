"""三个独立消息窗口的协作实验；模型选择由显式回放器提供，工具真实执行。"""
from copy import deepcopy
from hashlib import sha256
import json
from .agent import strict_json
from .retrieval import search, load_chunks


def encode(value):
    # 统一字节计数口径；本实验的跨角色报文都实际经过这次编码和解码。
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Stopped(RuntimeError):
    pass


class Allowance:
    def __init__(self, limit=12):
        if type(limit) is not int or limit < 1:
            raise ValueError("总调用额度必须为正整数")
        self.limit, self.used = limit, 0

    def take(self):
        # 父子循环共享额度；创建子窗口不会复制一份新的总预算。
        if self.used >= self.limit:
            raise Stopped("共享模型调用预算耗尽")
        self.used += 1


class RoleAgent:
    """相互独立的 messages、seen、trace、工具集合；模型回调可替换。"""
    def __init__(self, role, model, tools, allowance):
        self.role, self.model, self.tools, self.allowance = role, model, tools, allowance
        self.messages, self.trace, self.seen = [], [], set()

    def run(self, task, limit=6):
        self.messages = [
            {"role":"system", "content":f"{self.role}：仅使用本窗口证据与已授权工具。"},
            {"role":"user", "content":encode(task)},
        ]
        for step in range(limit):
            self.allowance.take()
            inputs = deepcopy(self.messages)
            output = self.model(deepcopy(inputs))
            self.trace.append({"step":step, "input":inputs, "output":deepcopy(output)})
            if type(output) is not dict:
                raise ValueError("提案必须为对象")
            if output.get("type") == "final":
                if type(output.get("value")) is not dict:
                    raise ValueError("结构化结果必须为对象")
                self.messages.append({"role":"assistant", "content":encode(output["value"])})
                return deepcopy(output["value"])
            if output.get("type") != "tool_call":
                raise ValueError("未知响应类型")
            call_id = output.get("id")
            if type(call_id) is not str or not call_id or call_id in self.seen:
                raise ValueError("调用编号缺失或重复")
            self.seen.add(call_id)
            self.messages.append({"role":"assistant", "tool_call":deepcopy(output)})
            try:
                name = output["name"]
                if name not in self.tools:
                    raise ValueError("当前角色无权调用此工具")
                arguments = strict_json(output["arguments"])
                if type(arguments) is not dict:
                    raise ValueError("参数必须为对象")
                value = self.tools[name](arguments)
                receipt = {"ok":True, "value":value}
            except Stopped:
                # 子循环停止必须向上穿透，不能伪造为一份正常证据。
                raise
            except (ValueError, KeyError, TypeError) as error:
                receipt = {"ok":False, "error":str(error)}
            self.messages.append({"role":"tool", "call_id":call_id, "content":encode(receipt)})
        raise Stopped(f"{self.role} 局部轮数耗尽")


def call(identifier, name, arguments):
    return {"type":"tool_call", "id":identifier, "name":name, "arguments":encode(arguments)}


def pack_evidence(question):
    # 权限身份来自宿主；跨角色包保存完整摘录和摘要，Checker 仍重读可信原文。
    result = []
    for item in search(question, owner="alice")["selected"]:
        row = {k:item[k] for k in ("id","source","owner","revision","start","end","text")}
        row["sha256"] = sha256(row["text"].encode("utf-8")).hexdigest()
        result.append(row)
    return result


def check_packet(packet):
    if set(packet) != {"question","claims","evidence"}:
        raise ValueError("核验包字段不符")
    visible = {c["id"]:c for c in load_chunks(owner="alice")}
    supplied, invalid = {}, []
    for item in packet["evidence"]:
        trusted = visible.get(item.get("id"))
        fields = ("source","owner","revision","start","end","text")
        if (trusted is None or any(item.get(k) != trusted[k] for k in fields)
                or item.get("sha256") != sha256(trusted["text"].encode("utf-8")).hexdigest()):
            invalid.append(item.get("id"))
        else:
            supplied[item["id"]] = trusted
    # 独立于检索排名的人工语义标注；这里只核验列出的两项命题。
    gold = {
        ("切片共享数据。","numpy:0"):True,
        ("切片会自动复制数据。","numpy:0"):False,
    }
    rows = []
    for claim in packet["claims"]:
        citation, text = claim["citation"], claim["text"]
        in_packet = citation in supplied
        supported = gold.get((text,citation))
        accepted = in_packet and supported is True
        rows.append(dict(claim, in_packet=in_packet, supported=supported, accepted=accepted))
    accepted = [r for r in rows if r["accepted"]]
    status = ("conflict" if invalid or any(r["supported"] is False for r in rows)
              else "complete" if accepted and len(accepted)==len(rows)
              else "insufficient")
    return {"status":status, "claims":rows, "invalid_evidence":invalid,
            "accepted_claims":[{"text":r["text"],"citation":r["citation"]} for r in accepted]}


class ReplayPolicy:
    """回放动作选择；数据来自本轮工具回执，不返回写死的检索或核验结果。"""
    def __init__(self, role, fault=None):
        self.role, self.fault = role, fault

    def __call__(self, messages):
        task = strict_json(messages[1]["content"])
        receipts = [m for m in messages if m["role"] == "tool"]
        if self.role == "Finder":
            if not receipts:
                return call("find-1", "search_evidence", {"question":task["question"]})
            receipt = strict_json(receipts[-1]["content"])
            if not receipt["ok"]:
                return {"type":"final","value":{"question":task["question"],"claims":[],"evidence":[]}}
            evidence = receipt["value"]
            claims = []
            if any(c["id"] == "numpy:0" for c in evidence):
                claims = [{"text":"切片共享数据。","citation":"numpy:0"}]
            if self.fault == "false_claim" and claims:
                claims[0]["text"] = "切片会自动复制数据。"
            if self.fault == "tamper" and evidence:
                evidence[0]["text"] += "额外内容"
            return {"type":"final","value":{"question":task["question"],"claims":claims,"evidence":evidence}}
        if self.role == "Checker":
            if not receipts:
                return call("check-1","verify_packet",task)
            receipt = strict_json(receipts[-1]["content"])
            return {"type":"final","value":receipt["value"] if receipt["ok"] else
                    {"status":"conflict","accepted_claims":[],"error":receipt["error"]}}
        if not receipts:
            return call("delegate-1","delegate_finder",{"question":task["question"]})
        last = strict_json(receipts[-1]["content"])
        if not last["ok"]:
            return {"type":"final","value":{"status":"conflict","answer":"","error":last["error"]}}
        if len(receipts) == 1:
            return call("delegate-2","delegate_checker",last["value"])
        checked = last["value"]
        answer = "\n".join(f'{c["text"]} [{c["citation"]}]' for c in checked.get("accepted_claims",[]))
        return {"type":"final","value":{"status":checked["status"],
                "answer":answer if checked["status"] == "complete" else "",
                "review":checked}}


def cooperate(question="切片共享", fault=None, budget=12, models=None):
    """真实执行三个循环和只读工具；默认模型接口使用 ReplayPolicy。"""
    allowance, agents, wire = Allowance(budget), {}, []
    reviewed = []
    def transfer(sender, receiver, kind, payload):
        message = {"sender":sender,"receiver":receiver,"kind":kind,"payload":payload}
        raw = encode(message).encode("utf-8")
        wire.append({"message":deepcopy(message),"utf8_bytes":len(raw)})
        return strict_json(raw.decode("utf-8"))["payload"]

    def find(args):
        if set(args) != {"question"} or type(args["question"]) is not str:
            raise ValueError("Finder 参数不符")
        return pack_evidence(args["question"])

    def delegate(role, payload):
        incoming = transfer("Coordinator", role, "task", payload)
        tools = {"search_evidence":find} if role == "Finder" else {"verify_packet":check_packet}
        policy = models[role] if models else ReplayPolicy(role, fault)
        agent = RoleAgent(role, policy, tools, allowance)
        agents[role] = agent
        value = agent.run(incoming)
        if role == "Checker":
            # 模型生成的 final 仍是提案；主管重新计算核验，不能照收一个 passed 标签。
            canonical = check_packet(incoming)
            if value != canonical:
                raise ValueError("Checker 最终结果与可信核验不一致")
            reviewed.append(deepcopy(canonical))
        return transfer(role,"Coordinator","result",value)

    coordinator = RoleAgent("Coordinator", models["Coordinator"] if models else ReplayPolicy("Coordinator"),
        {"delegate_finder":lambda a:delegate("Finder",a),
         "delegate_checker":lambda a:delegate("Checker",a)}, allowance)
    agents["Coordinator"] = coordinator
    try:
        result = coordinator.run({"question":question})
        if reviewed:
            checked = reviewed[-1]
            answer = "\n".join(f'{c["text"]} [{c["citation"]}]' for c in checked["accepted_claims"])
            expected = {"status":checked["status"], "answer":answer if checked["status"]=="complete" else "",
                        "review":checked}
            if result != expected:
                raise ValueError("主管最终结果与已核验成果不一致")
        elif result.get("status") == "complete":
            raise ValueError("未经核验不能完成")
    except Stopped as error:
        result = {"status":"stopped","answer":"","reason":str(error)}
    # 记录每次真正送给模型回调的快照长度；不是只算最后一份历史。
    counts = {role:len(a.trace) for role,a in agents.items()}
    input_bytes = {role:sum(len(encode(t["input"]).encode("utf-8")) for t in a.trace)
                   for role,a in agents.items()}
    roles = list(agents)
    assert all(agents[a].messages is not agents[b].messages for i,a in enumerate(roles) for b in roles[i+1:])
    return {"result":result,"model_mode":"deterministic_action_replay",
            "agents":{role:{"messages":deepcopy(a.messages),"trace":deepcopy(a.trace),
                            "allowed_tools":list(a.tools)} for role,a in agents.items()},
            "wire":wire,"wire_bytes":sum(x["utf8_bytes"] for x in wire),
            "model_calls":counts,"total_calls":allowance.used,"model_input_bytes":input_bytes,
            "isolated_message_lists":True}


def conflicts():
    # 两个子任务都读到版本 7；主管合并时必须检验它们各自的依据。
    state = {"version":7,"plan":"旧提案"}
    proposals = [{"worker":"A","expected_version":7,"plan":"方案 A"},
                 {"worker":"B","expected_version":7,"plan":"方案 B"}]
    outcomes = []
    for p in proposals:
        if p["expected_version"] != state["version"]:
            outcomes.append({"worker":p["worker"],"accepted":False,"current_version":state["version"]})
        else:
            state = {"version":state["version"]+1,"plan":p["plan"]}
            outcomes.append({"worker":p["worker"],"accepted":True,"current_version":state["version"]})
    return {"proposals":proposals,"outcomes":outcomes,"final":state,
            "scope":"单进程顺序合并；没有实现跨进程原子比较交换"}


def experiment():
    cases = {name:cooperate(question, fault, budget) for name,question,fault,budget in [
        ("complete","切片共享",None,12),("missing","火星天气",None,12),
        ("conflict","切片共享","false_claim",12),("tampered","切片共享","tamper",12),
        ("stopped","切片共享",None,4)]}
    assert [c["result"]["status"] for c in cases.values()] == [
        "complete","insufficient","conflict","conflict","stopped"]
    assert cases["complete"]["model_calls"] == {"Coordinator":3,"Finder":2,"Checker":2}
    assert cases["complete"]["total_calls"] == 7
    assert cases["complete"]["result"]["answer"] == "切片共享数据。 [numpy:0]"
    assert len(cases["complete"]["wire"]) == 4
    # 保存的输入快照不能因为后续消息字段修改而改变。
    sample = cases["complete"]["agents"]["Finder"]
    frozen = deepcopy(sample["trace"][0]["input"])
    sample["messages"][1]["content"] += "审阅后标记"
    assert sample["trace"][0]["input"] == frozen
    sample["messages"][1] = deepcopy(frozen[1])
    # 越权工具名和重复编号分别在循环边界拒绝。
    unauthorized = RoleAgent("Finder",lambda m:call("x","save",{}),{},Allowance(3))
    try:
        unauthorized.run({"question":"x"},limit=2)
    except ValueError as error:
        duplicate_error = str(error)
    assert "重复" in duplicate_error
    assert "无权" in strict_json(unauthorized.messages[-1]["content"])["error"]
    policies = {r:ReplayPolicy(r) for r in ["Coordinator","Finder","Checker"]}
    policies["Checker"] = lambda m:{"type":"final","value":{"status":"complete","accepted_claims":[]}}
    forged = cooperate(models=policies)
    assert forged["result"]["status"] == "conflict"
    return {"cases":cases,"snapshot_stable":True,
            "forged_checker_rejected":forged["result"],
            "unauthorized_receipt":strict_json(unauthorized.messages[-1]["content"]),
            "duplicate_rejected":duplicate_error,"version_merge":conflicts()}

