"""新第八篇：协议会话、按需加载、框架映射与独立协作循环。"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import yaml
from .agent import strict_json
from .mcp_stdio import VERSION, StdioClient
from .application_details import transport_comparison
from .cooperation_lab import cooperate, experiment as cooperation_experiment, encode
from .harness import communication_cost

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT/"extensions"/"reading-plugin"


def protocol_experiment():
    # 只保留 JSON 报文和退出事实；进程与管道由上下文管理器回收。
    with StdioClient() as client:
        early = client.send({"jsonrpc":"2.0","id":0,"method":"tools/list"})
        client.initialize()
        listing = client.send({"jsonrpc":"2.0","id":2,"method":"tools/list"})
        results = {}
        cases = [
            ("valid","tools/call",{"name":"reading_minutes","arguments":{"pages":3,"questions":4}}),
            ("bool","tools/call",{"name":"reading_minutes","arguments":{"pages":True,"questions":4}}),
            ("unknown_tool","tools/call",{"name":"missing","arguments":{}}),
            ("bad_params","tools/call",{"name":"reading_minutes","arguments":[]}),
            ("unknown_method","tools/missing",{}),
        ]
        for identifier,(name,method,params) in enumerate(cases,3):
            results[name] = client.send({"jsonrpc":"2.0","id":identifier,"method":method,"params":params})
        # 请求体 JSON 中的换行被转义，整个消息仍只占一条物理行。
        framed = encode({"jsonrpc":"2.0","id":8,"method":"echo",
                         "params":{"text":"第一行\n第二行"}})+"\n"
        assert framed.count("\n")==1 and "\\n" in framed
        client.process.stdin.write("{bad\n")
        client.process.stdin.flush()
        malformed = strict_json(client.incoming.get(timeout=5))
        assert malformed["error"]["code"] == -32700
        record = {"version":VERSION,"early":early,"listing":listing,"cases":results,
                  "malformed":malformed,"requests":deepcopy(client.requests),
                  "responses":deepcopy(client.responses),"physical_newlines":framed.count("\n")}
    record["process_closed"] = client.process.poll() == 0
    record["reader_threads_closed"] = all(not t.is_alive() for t in client.readers)
    record["stderr"] = "".join(client.errors)
    with StdioClient() as other:
        try:
            other.initialize("not-supported")
        except ValueError as error:
            record["version_rejected"] = str(error)
        else:
            raise AssertionError("版本失配未拒绝")
        record["mismatch_sent_methods"] = [r["method"] for r in other.requests]
    record["mismatch_process_closed"] = other.process.poll() == 0
    assert early["error"]["code"] == -32000
    assert results["valid"]["result"]["isError"] is False
    assert results["bool"]["result"]["isError"] is True
    assert results["unknown_tool"]["error"]["code"] == -32602
    assert results["bad_params"]["error"]["code"] == -32602
    assert results["unknown_method"]["error"]["code"] == -32601
    assert record["mismatch_sent_methods"] == ["initialize"]
    return record


class SkillReader:
    """三个读取入口分别记录读取了什么；不把文本内容解释成执行权限。"""
    def __init__(self):
        self.events = []
        self.allowed = {}

    def discover(self):
        discovered = []
        for path in sorted((PLUGIN/"skills").glob("*/SKILL.md")):
            # 逐行停在第二个分隔符，正文不进入解析结果。
            with path.open(encoding="utf-8") as stream:
                first = stream.readline()
                if first.strip() != "---":
                    raise ValueError("缺少 frontmatter")
                lines = []
                for line in stream:
                    if line.strip() == "---":
                        break
                    lines.append(line)
                else:
                    raise ValueError("frontmatter 未闭合")
            metadata = yaml.safe_load("".join(lines))
            if metadata["name"] != path.parent.name:
                raise ValueError("技能名与目录不匹配")
            entry = {k:metadata[k] for k in ("name","description")}
            discovered.append(entry)
            self.allowed[entry["name"]] = path.parent.resolve()
            self.events.append({"phase":"discover","file":str(path.relative_to(PLUGIN)),
                                "context_bytes":len(encode(entry).encode("utf-8"))})
        return discovered

    def body(self, name):
        if name not in self.allowed:
            raise ValueError("必须先发现合法技能名")
        text = (self.allowed[name]/"SKILL.md").read_text("utf-8").split("---",2)[2].strip()
        self.events.append({"phase":"body","file":f"skills/{name}/SKILL.md",
                            "context_bytes":len(text.encode("utf-8"))})
        return text

    def resource(self, name, relative):
        if name not in self.allowed or relative != "references/template.txt":
            raise ValueError("资源不在明确允许列表")
        base = self.allowed[name]
        path = (base/relative).resolve(strict=True)
        if not path.is_relative_to(base):
            raise ValueError("资源越出技能目录")
        text = path.read_text("utf-8")
        self.events.append({"phase":"resource","file":f"skills/{name}/{relative}",
                            "context_bytes":len(text.encode("utf-8"))})
        return text


def skill_experiment():
    reader = SkillReader()
    metadata = reader.discover()
    after_discover = deepcopy(reader.events)
    body = reader.body("organize-notes")
    after_body = deepcopy(reader.events)
    template = reader.resource("organize-notes","references/template.txt")
    denied = []
    for name,relative in [("organize-notes","../SKILL.md"),("missing","references/template.txt")]:
        try:
            reader.resource(name,relative)
        except ValueError as error:
            denied.append(str(error))
    manifest = json.loads((PLUGIN/".claude-plugin/plugin.json").read_text("utf-8"))
    assert manifest["name"]=="reading-notes" and manifest["skills"]=="./skills"
    assert [e["phase"] for e in reader.events]==["discover","body","resource"]
    return {"metadata":metadata,"after_discover":after_discover,"after_body":after_body,
            "events":reader.events,"body":body,"template":template,"denied":denied,
            "manifest":manifest,"installed_in_product":False,
            "tool_permissions_granted_by_reading":[],
            "context_bytes_total":sum(e["context_bytes"] for e in reader.events)}


def framework_experiment():
    # 复用既有的参数模型、编码器与原生检索。此入口不创建数据库或检查点。
    from langchain_core.runnables import RunnableLambda
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Annotated
    import operator
    from .frameworks import CountEmbeddings, Minutes, retrieval_graph, reading_minutes
    from .retrieval import load_chunks, search, embed
    chunks = load_chunks(owner="alice")
    documents = [Document(id=c["id"],page_content=c["text"],
                          metadata={k:c[k] for k in ("id","source","revision","owner")}) for c in chunks]
    index = InMemoryVectorStore.from_documents(documents, CountEmbeddings())
    pairs = index.similarity_search_with_score("切片共享",k=3)
    assert CountEmbeddings().embed_query("切片共享")==embed("切片共享").tolist()
    pipe = RunnableLambda(strict_json) | RunnableLambda(Minutes.model_validate)
    parsed = pipe.invoke('{"pages":3,"questions":4}')
    rejected = []
    for raw in ['{"pages":true,"questions":4}','{"pages":"3","questions":4}',
                '{"pages":3.0,"questions":4}','{"pages":3,"pages":30,"questions":4}',
                '{"pages":3,"questions":4,"owner":"bob"}']:
        try:
            pipe.invoke(raw)
        except (ValueError,TypeError) as error:
            rejected.append(type(error).__name__)
    prompt = ChatPromptTemplate.from_messages([
        ("system","依据资料作答；缺证据则说明缺失。"),
        ("human","问题：{question}\n资料：{evidence}")])
    rendered = prompt.invoke({"question":"切片共享","evidence":f'[{pairs[0][0].id}] {pairs[0][0].page_content}'})
    graph = retrieval_graph()
    paths = {}
    for question in ["切片共享","火星天气"]:
        steps = list(graph.stream({"question":question},stream_mode="updates"))
        paths[question] = {"steps":steps,"nodes":[next(iter(s)) for s in steps],
                           "answer":graph.invoke({"question":question})["answer"]}
    # 默认覆盖和显式 reducer 是两种字段更新语义；用相同节点实跑对照。
    class Replace(TypedDict):
        notes: list[str]
    class Append(TypedDict):
        notes: Annotated[list[str],operator.add]
    def tiny(schema):
        b=StateGraph(schema)
        b.add_node("first",lambda s:{"notes":["检索"]})
        b.add_node("second",lambda s:{"notes":["核验"]})
        b.add_edge(START,"first");b.add_edge("first","second");b.add_edge("second",END)
        return b.compile().invoke({"notes":[]})["notes"]
    replacement, accumulation = tiny(Replace), tiny(Append)
    assert replacement==["核验"] and accumulation==["检索","核验"]
    assert paths["切片共享"]["nodes"]==["retrieve","reply"]
    assert paths["火星天气"]["nodes"]==["retrieve","abstain"]
    return {"versions":{p:importlib.metadata.version(p) for p in [
                "langchain","langchain-core","langgraph","pydantic"]},
            "vector_results":[{"id":d.id,"cosine":float(score)} for d,score in pairs],
            "native_hybrid_ids":[c["id"] for c in search("切片共享")["selected"]],
            "prompt":[{"role":m.type,"content":m.content} for m in rendered.to_messages()],
            "parsed":parsed.model_dump(),"tool_result":reading_minutes.invoke(parsed.model_dump()),
            "rejected":rejected,"graph_paths":paths,"replace":replacement,"append":accumulation,
            "persistence_enabled":False}


def cli_experiment():
    # 对 CLI 真正启动新进程，检查 stdout 是完整 JSON、stderr 是进度文本。
    result = subprocess.run([sys.executable,"-B","-m","modern_ai.ecosystem_lab",
                             "cli","--question","切片共享"],
                            cwd=ROOT,text=True,encoding="utf-8",capture_output=True,
                            timeout=30,check=True)
    parsed = strict_json(result.stdout)
    assert parsed["status"]=="complete"
    return {"stdout":parsed,"stderr":result.stderr.strip(),"exit_code":result.returncode}


def experiment():
    result={"protocol":protocol_experiment(),"transports":transport_comparison(),
            "skills":skill_experiment(),"cooperation":cooperation_experiment(),
            "frameworks":framework_experiment(),"cli":cli_experiment(),
            "old_envelope_example":communication_cost()}
    assert result["transports"]["same_value"]
    return result


def exercise(code):
    # 稳定命名空间 PE（Protocol / Ecosystem），保留旧 05-04 的原含义。
    funcs={"PE-01":protocol_experiment,"PE-02":transport_comparison,
           "PE-03":skill_experiment,"PE-04":lambda:cooperate(),
           "PE-05":cooperation_experiment,"PE-06":framework_experiment}
    if code not in funcs:
        raise ValueError("未知 PE 题号")
    return funcs[code]()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["all","protocol","skills","cooperate","frameworks","cli"])
    parser.add_argument("--question",default="切片共享")
    args=parser.parse_args()
    if args.mode=="cli":
        print("正在检索并核验来源",file=sys.stderr)
        result=cooperate(args.question)["result"]
        print(encode(result))
        return 0 if result["status"]=="complete" else 2
    funcs={"all":experiment,"protocol":protocol_experiment,"skills":skill_experiment,
           "cooperate":lambda:cooperate(args.question),"frameworks":framework_experiment}
    print(json.dumps(funcs[args.mode](),ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

