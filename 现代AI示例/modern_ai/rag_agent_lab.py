"""新第七篇：来源核验、检索算账、工具纠错与跨进程 HTTP 实验。"""
from collections import Counter
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
import numpy as np
from .agent import TOOLS, BudgetExceeded, execute, run, strict_json
from .retrieval import DATA, VOCAB, embed, load_chunks, search, tokenize
from .application_details import rank_example, retrieval_evaluation, citation_examples


def digest(text):
    # 完整摘要用于版本核对；短编号只供阅读，不承担身份校验。
    return sha256(text.encode("utf-8")).hexdigest()


def expand_evidence(hit, owner="alice", as_of=None):
    # 重新从可访问的现行版本定位，拒绝过期命中和被修改的摘录。
    visible = load_chunks(owner=owner, as_of=as_of)
    keys = ("source", "owner", "revision", "id", "start", "end", "text", "hash")
    found = next((c for c in visible if all(c[k] == hit[k] for k in keys)), None)
    if found is None:
        raise ValueError("命中已过期、内容改变或当前身份不可读")
    siblings = [c for c in visible if all(c[k] == found[k]
                for k in ("source", "owner", "revision"))]
    siblings.sort(key=lambda c: c["start"])
    index = siblings.index(found)
    neighbors = siblings[max(0, index-1):index+2]
    start, end = neighbors[0]["start"], neighbors[-1]["end"]
    records = json.loads(DATA.read_text("utf-8-sig"))
    source = next(r for r in records if all(r[k] == found[k]
                  for k in ("source", "owner", "revision")))
    text = source["text"][start:end]
    return dict(source=found["source"], revision=found["revision"],
                owner=found["owner"], start=start, end=end, text=text,
                sha256=digest(text), members=[c["id"] for c in neighbors])


def embedding_contract(vocabulary=VOCAB):
    # 维数相同仍不足以判断坐标相同，词表顺序也进入契约摘要。
    return {"dimension":len(vocabulary), "normalization":"l2-zero-kept",
            "encoder":"substring-count-v1",
            "vocabulary_sha256":digest(json.dumps(list(vocabulary), ensure_ascii=False))}


def checked_cosine(query, document, query_contract, index_contract):
    if query_contract != index_contract:
        raise ValueError("查询编码器与索引坐标契约不一致")
    q, d = np.asarray(query), np.asarray(document)
    if q.shape != (index_contract["dimension"],) or d.shape != q.shape:
        raise ValueError("向量形状不符合契约")
    return float(q @ d)


def source_experiment():
    hit = next(c for c in load_chunks() if c["id"] == "numpy:0")
    expanded = expand_evidence(hit)
    tampered = dict(hit, text=hit["text"]+"自动复制")
    failures = []
    for candidate in [tampered, load_chunks(owner="bob")[-1],
                      load_chunks(as_of="2025-01-01")[0]]:
        try:
            expand_evidence(candidate)
        except ValueError as error:
            failures.append(str(error))
    # 固定打分表隔离“先过滤还是先截断”的影响；不是实测语料排名。
    ranking = [("private", .99, False), ("old", .98, False),
               ("current-a", .95, True), ("current-b", .90, True)]
    late = [r[0] for r in ranking[:2] if r[2]]
    early = [r[0] for r in ranking if r[2]][:2]
    return {"chunks":load_chunks(), "expanded":expanded,
            "rejected":failures, "postfilter_top2":late, "prefilter_top2":early}


def vector_experiment():
    q = embed("切片共享")
    d = embed("切片介绍：切片共享数据。")
    contract = embedding_contract()
    swapped = VOCAB.copy()
    swapped[0], swapped[2] = swapped[2], swapped[0]
    wrong = np.array([q[VOCAB.index(t)] for t in swapped])
    rejected = False
    try:
        checked_cosine(wrong, d, embedding_contract(swapped), contract)
    except ValueError:
        rejected = True
    return {"query":q.tolist(), "document":d.tolist(),
            "correct_cosine":checked_cosine(q,d,contract,contract),
            "unchecked_wrong_cosine":float(wrong@d), "contract_rejected":rejected,
            "zero_query":embed("火星天气").tolist(), "contract":contract}


def bm25_experiment():
    record = search("切片共享")
    first = next(c for c in record["candidates"] if c["id"] == "numpy:0")
    length = len(tokenize(first["text"]))
    average = record["average_length"]
    k = 1.2*(.25+.75*length/average)
    return {"N":record["corpus_size"], "average":average, "length":length,
            "K":k, "terms":first["bm25_terms"], "score":first["bm25"],
            "saturation":[{"tf":f,"factor":2.2*f/(f+k)} for f in [0,1,2,4,8,16,100]],
            "candidates":[{key:c[key] for key in
                 ["id","cosine","bm25","vector_rank","keyword_rank","fused"]}
                 for c in record["candidates"]],
            "selected":[c["id"] for c in record["selected"]]}


def finite_json(raw):
    # 标准 JSON 字面量仍可能在二进制浮点解码时溢出，例如 1e999。
    value = strict_json(raw)
    def inspect(item):
        if type(item) is float and not math.isfinite(item):
            raise ValueError("浮点数溢出或非有限")
        if type(item) is dict:
            for child in item.values():
                inspect(child)
        elif type(item) is list:
            for child in item:
                inspect(child)
    inspect(value)
    return value


def validation_experiment():
    raw_cases = ['{"pages":true,"questions":4}', '{"pages":3.0,"questions":4}',
                 '{"pages":3,"pages":9,"questions":4}', '{"pages":NaN,"questions":4}',
                 '{"pages":1e999,"questions":4}', '{"pages":-1,"questions":4}',
                 '{"pages":1001,"questions":4}', '{"pages":3}',
                 '{"pages":3,"questions":4,"owner":"bob"}', '[]']
    rows = []
    for raw in raw_cases:
        try:
            execute("reading_minutes", finite_json(raw))
        except (ValueError, TypeError) as error:
            rows.append({"raw":raw, "rejected":True, "error":str(error)})
        else:
            rows.append({"raw":raw, "rejected":False})
    return {"valid":execute("reading_minutes",finite_json('{"pages":3,"questions":4}')),
            "invalid":rows}


def loop_experiment():
    requests = []
    def repair(messages):
        # 适配器明确组装消息与工具声明；固定回放由工具回执驱动下一步。
        requests.append(deepcopy({"messages":messages, "tools":TOOLS}))
        if messages[-1]["role"] != "tool":
            pages, call = True, "estimate-1"
        else:
            result = json.loads(messages[-1]["content"])
            if result["ok"]:
                return {"type":"final", "text":f"预计 {result['value']['minutes']} 分钟。"}
            pages, call = 3, "estimate-2"
        return {"type":"tool_call", "id":call, "name":"reading_minutes",
                "arguments":json.dumps({"pages":pages,"questions":4})}
    repaired = run("读3页，做4题，需要多久？", model=repair)
    snapshot = deepcopy(repaired["trace"][0]["input"])
    repaired["messages"][0]["content"] = "事后修改"
    stable = repaired["trace"][0]["input"] == snapshot
    def endless(messages):
        # 每轮编号不同，确保真正触发预算而非重复编号检查。
        return {"type":"tool_call","id":f"loop-{len(messages)}",
                "name":"reading_minutes","arguments":'{"pages":3,"questions":4}'}
    try:
        run("一直调用", model=endless, limit=2)
    except BudgetExceeded as error:
        stopped = {"trace_count":len(error.trace), "message_count":len(error.messages),
                   "roles":[m["role"] for m in error.messages]}
    def duplicate(messages):
        return {"type":"tool_call","id":"same","name":"reading_minutes",
                "arguments":'{"pages":3,"questions":4}'}
    rejected = False
    try:
        run("重复编号", model=duplicate)
    except ValueError:
        rejected = True
    return {"answer":repaired["answer"], "trace":repaired["trace"],
            "requests":requests, "snapshot_stable":stable, "budget":stopped,
            "duplicate_rejected":rejected}


def integrated_experiment():
    # 固定任务回放依次检索、计算和引用；每步都读取上一次实际回执。
    def planner(messages):
        receipts = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        if not receipts:
            return {"type":"tool_call","id":"lookup-1","name":"search_notes",
                    "arguments":'{"query":"复制后原数组"}'}
        if not receipts[-1]["ok"]:
            return {"type":"final","text":"工具执行失败，停止当前任务。"}
        if len(receipts) == 1:
            return {"type":"tool_call","id":"estimate-1","name":"reading_minutes",
                    "arguments":'{"pages":3,"questions":4}'}
        evidence = receipts[0]["value"][0]
        minutes = receipts[1]["value"]["minutes"]
        return {"type":"final","text":f"{evidence['text']} [{evidence['id']}] 预计 {minutes} 分钟。"}
    return run("查复制与原数组的关系，再估算读3页、做4题的时间。",model=planner)


def gui_step(state, action):
    # 纯内存界面回放：不移动用户鼠标，也不连接真实桌面。
    if action.get("screen_id") != state["screen_id"]:
        raise ValueError("截图版本已过期")
    x, y = action["x"], action["y"]
    if type(x) is not int or type(y) is not int or not (0<=x<640 and 0<=y<400):
        raise ValueError("截图坐标越界")
    actual = [2*x, 2*y]
    if not (620<=actual[0]<680 and 300<=actual[1]<340):
        raise ValueError("坐标未落在当前按钮")
    return {"screen_id":state["screen_id"]+1, "status":"已保存",
            "actual_point":actual}


def gui_experiment():
    state = {"screen_id":7, "status":"待保存"}
    proposal = {"screen_id":7, "x":321, "y":159}
    after = gui_step(state, proposal)
    failures = []
    for s,a in [(after,proposal),(state,dict(proposal,x=640)),
                (state,dict(proposal,x=1))]:
        try:
            gui_step(s,a)
        except ValueError as error:
            failures.append(str(error))
    return {"before":state, "action":proposal, "after":after, "rejected":failures}


def wiki_experiment():
    # 摘要是预先写定的修订提案；支持关系来自人工核读，不冒充语义判定器。
    raw = expand_evidence(next(c for c in load_chunks() if c["id"]=="numpy:0"))
    proposal = {"text":"切片共享数据；修改视图会影响原数组。复制后再改，原数组保持不变。",
                "source_sha256":raw["sha256"],"review":"approved"}
    index = {"wiki_sha256":digest(proposal["text"]), "source_sha256":raw["sha256"]}
    revised = dict(proposal,text=proposal["text"]+" 请区分视图与复制。")
    fresh = lambda p: (p["review"]=="approved" and
                      p["source_sha256"]==raw["sha256"] and
                      digest(p["text"])==index["wiki_sha256"])
    stale = not fresh(revised)
    index["wiki_sha256"] = digest(revised["text"])
    return {"source":raw,"proposal":proposal,"stale_index_rejected":stale,
            "reindexed_ready":fresh(revised),
            "false_claim_review":{"text":"切片总会自动复制","review":"rejected"}}


def _http_worker(connection, stop):
    # 子进程独占模型和服务；端口由系统分配，退出时关闭监听与线程。
    from .transport import create_server
    import torch
    torch.set_num_threads(2)
    server = None
    thread = None
    try:
        server = create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection.send({"port":server.server_port,"pid":os.getpid()})
        stop.wait()
    except BaseException as error:
        connection.send({"error":repr(error)})
    finally:
        if server is not None:
            if thread is not None:
                server.shutdown()
                thread.join()
            server.server_close()
        connection.close()


@contextmanager
def local_service():
    # Windows 使用 spawn，继承当前 .venv 的解释器；不产生权重文件或数据库。
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    stop = ctx.Event()
    process = ctx.Process(target=_http_worker, args=(child,stop))
    process.start()
    child.close()
    try:
        if not parent.poll(60):
            raise TimeoutError("本地服务启动超时")
        info = parent.recv()
        if "error" in info:
            raise RuntimeError(info["error"])
        yield f"http://127.0.0.1:{info['port']}", info
    finally:
        stop.set()
        process.join(8)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        process.close()


def http_experiment():
    from .transport import post_json
    with local_service() as (base, info):
        generated = post_json(base,"/generate",{"prompt":"read ","max_new_tokens":12})
        result = post_json(base,"/tools/reading_minutes",{"pages":3,"questions":4})
        statuses = []
        for path,body in [("/missing",{}),("/generate",{"prompt":"","max_new_tokens":12}),
                          ("/tools/reading_minutes",{"pages":True,"questions":4})]:
            try:
                post_json(base,path,body)
            except HTTPError as error:
                statuses.append(error.code)
                error.close()
    return {"separate_process":info["pid"]!=os.getpid(), "generation":generated,
            "tool":result,"error_statuses":statuses,"service_closed":True}


def experiment(include_http=False):
    result = {"source":source_experiment(),"vectors":vector_experiment(),
              "bm25":bm25_experiment(),"rrf":rank_example(),
              "evaluation":retrieval_evaluation(),"citations":citation_examples(),
              "validation":validation_experiment(),"loop":loop_experiment(),
              "gui":gui_experiment(),"wiki":wiki_experiment(),
              "integrated":integrated_experiment()}
    if include_http:
        result["http"] = http_experiment()
    return result


def exercise(code):
    functions = {"RA-01":loop_experiment,"RA-02":source_experiment,
                 "RA-03":vector_experiment,"RA-04":bm25_experiment,
                 "RA-05":lambda:{"rrf":rank_example(),"evaluation":retrieval_evaluation()},
                 "RA-06":lambda:{"citations":citation_examples(),"wiki":wiki_experiment()},
                 "RA-07":lambda:{"validation":validation_experiment(),"gui":gui_experiment()},
                 "RA-08":loop_experiment}
    if code not in functions:
        raise ValueError("未知题号："+code)
    return functions[code]()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    data = experiment(args.http)
    encoded = json.dumps(data,ensure_ascii=False,indent=2)
    if args.output:
        args.output.write_text(encoded,encoding="utf-8")
    else:
        print(encoded)

