"""新第九篇：真实进程退出、事务、租约写入边界与图检查点实验。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
from typing import TypedDict
from .runtime import Store, encode
from .harness import Queue
from .runtime_details import ReceiptLedger, ready

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "现代AI示例"


def snapshot(store, run_id):
    """同时观察四张提交相关表，不能只看函数有没有报错。"""
    task = store.get(run_id)
    audit = store.audit(run_id)
    with store.connection() as connection:
        outbox = [dict(row) for row in connection.execute(
            "SELECT key,status FROM outbox WHERE key=?", ("plan:"+run_id,))]
    return {"status":task["status"], "version":task["version"],
            "has_proposal":"proposal" in task["payload"], "plans":audit["plan_count"],
            "events":[row["kind"] for row in audit["events"]],
            "event_seq":[row["seq"] for row in audit["events"]], "outbox":outbox}


def rejected(action):
    try:
        action()
    except ValueError as error:
        return str(error)
    raise AssertionError("预期拒绝，但操作成功")


def revise_waiting(store, run_id, title, expected_version):
    """本篇增加的提案标题修订；只允许 waiting 的同版本写入。"""
    if type(title) is not str or not title.strip():
        raise ValueError("标题必须为非空字符串")
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("修订版本必须为非负整数")
    with store.transaction() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id=? AND owner='alice'", (run_id,)).fetchone()
        if row is None or row["status"] != "waiting" or row["version"] != expected_version:
            raise ValueError("待确认提案已经变化")
        payload = json.loads(row["payload"])
        payload["proposal"]["title"] = title
        changed = connection.execute(
            "UPDATE runs SET payload=?,version=version+1 WHERE id=? AND owner='alice' AND status='waiting' AND version=?",
            (encode(payload), run_id, expected_version)).rowcount
        if changed != 1:
            raise ValueError("待确认提案已经变化")
        store.event(connection, run_id, "revised", {"from_version":expected_version,"title":title})
    return store.get(run_id)


def check_claim(connection, job_id, token, now):
    # 时刻由接受写入的一侧提供；实验传入显式逻辑时间，不读取 Worker 自报的时钟。
    if type(token) is not int or token < 1 or not math.isfinite(now):
        raise ValueError("领取身份或时刻无效")
    row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None or row["status"] != "running" or row["token"] != token or not now < row["lease_until"]:
        raise ValueError("写入被拒绝：租约到期或令牌已被替代")


class ClaimedStore(Store):
    """复用 Store.confirm，在其同一个写事务里先复核领取。"""
    def __init__(self, path, job_id, token, now):
        self.claim_identity = (job_id, token, now)
        super().__init__(path)

    @contextmanager
    def transaction(self):
        with super().transaction() as connection:
            check_claim(connection, *self.claim_identity)
            yield connection


def write_resource(path, job_id, token, now, body):
    # 此受保护资源与 jobs 位于同一数据库；比较领取与写入不能被重新领取插入其间。
    store = ClaimedStore(path, job_id, token, now)
    with store.transaction() as connection:
        connection.execute("INSERT INTO resource_effects VALUES(?,?,?) "
                           "ON CONFLICT(job_id) DO UPDATE SET token=excluded.token,body=excluded.body",
                           (job_id, token, body))


def bound_graph_action(path, action, run_id, thread_id, fail=None):
    # thread_id 选图历史，run_id 选业务结果；恢复前检查二者绑定。
    from .frameworks import build_graph, graph_action
    from langgraph.checkpoint.sqlite import SqliteSaver
    store = Store(path)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_graph(saver, store)
        saved = graph.get_state({"configurable":{"thread_id":thread_id}})
        if action == "start":
            if saved.values:
                raise ValueError("图线程已存在，请恢复原线程")
        else:
            if not saved.values:
                raise ValueError("没有可恢复的图线程")
            if saved.values["task_id"] != run_id:
                raise ValueError("图线程与业务任务不匹配")
            store.get(run_id, owner="alice")
    return graph_action(path, action, run_id, thread_id, fail)


class ProbeState(TypedDict):
    prepared: bool
    approved: bool


def interrupt_probe(path, resume=False):
    """持久化访问计数故意放在 interrupt 前后，显现重入位置。"""
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.sqlite import SqliteSaver
    store = Store(path)
    with store.transaction() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS visits(seq INTEGER PRIMARY KEY,label TEXT NOT NULL)")
    def mark(label):
        with store.transaction() as connection:
            connection.execute("INSERT INTO visits(label) VALUES(?)", (label,))
    def prepare(state):
        mark("prepare")
        return {"prepared":True}
    def review(state):
        mark("review:before")  # 审计用计数也会重跑；这里没有发送、付款等业务动作。
        approved = interrupt({"question":"是否接受提案？"})
        if type(approved) is not bool:
            raise ValueError("确认值必须为布尔值")
        mark("review:after")
        return {"approved":approved}
    builder = StateGraph(ProbeState)
    builder.add_node("prepare", prepare); builder.add_node("review", review)
    builder.add_edge(START,"prepare"); builder.add_edge("prepare","review"); builder.add_edge("review",END)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable":{"thread_id":"probe-thread"}}
        graph.invoke(Command(resume=True) if resume else {"prepared":False}, config)
        next_nodes = list(graph.get_state(config).next)
    with store.connection() as connection:
        visits = [row[0] for row in connection.execute("SELECT label FROM visits ORDER BY seq")]
    return {"visits":visits,"next":next_nodes}


def worker(request):
    """子进程入口只接受预先列出的动作，源码仍从正式项目目录导入。"""
    action, path = request["action"], Path(request["db"])
    task = request.get("run_id", "run-17")
    if action.startswith("graph-"):
        return bound_graph_action(path, action[6:], task, request["thread_id"], request.get("fail"))
    if action.startswith("probe-"):
        return interrupt_probe(path, action == "probe-resume")
    store = Store(path)
    if action == "create":
        return store.create(task, request.get("question", "检查点恢复"))
    if action == "step":
        return store.advance(task)
    if action == "confirm":
        return store.confirm(task, fail=request.get("fail"), expected_version=request.get("version"))
    if action == "claimed-confirm":
        return ClaimedStore(path,request["job_id"],request["token"],request["now"]).confirm(
            task, expected_version=request.get("version"))
    if action == "show":
        return snapshot(store, task)
    if action == "deliver":
        receiver = ReceiptLedger(Path(request["receiver"]))
        return store.deliver(receiver.accept, fail_after_send=request.get("fail_after_send",False))
    if action == "claim":
        queue = Queue(store)
        if request.get("barrier"):
            print(encode({"ready":os.getpid()}), flush=True)
            if sys.stdin.readline().strip() != "go":
                raise ValueError("未收到领取信号")
        return queue.claim(request["now"])
    raise ValueError("未知实验动作")


def process():
    child_env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    return subprocess.Popen([sys.executable,"-B","-m","modern_ai.persistence_lab","worker"],
                            cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,encoding="utf-8",env=child_env)


def child(action, path, expected=0, **kwargs):
    request = {"action":action,"db":str(path),**kwargs}
    p = process()
    try:
        output, errors = p.communicate(encode(request)+"\n", timeout=90)
        assert p.returncode == expected, (action,p.returncode,output,errors)
        result = json.loads(output) if output.strip() else None
        return {"pid":p.pid,"exit_code":p.returncode,"value":result.get("value") if result else None,
                "error":result.get("error") if result else None}
    finally:
        if p.poll() is None:
            p.kill();p.communicate(timeout=10)


def concurrent_claims(path):
    # 两个独立进程均完成准备后才放行，保证确实竞争同一个可领取任务。
    workers = [process(),process()]
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        for p in workers:
            p.stdin.write(encode({"action":"claim","db":str(path),"now":100,"barrier":True})+"\n")
            p.stdin.flush()
        futures=[pool.submit(p.stdout.readline) for p in workers]
        waiting=[json.loads(f.result(timeout=90)) for f in futures]
        assert all("ready" in row for row in waiting)
        for p in workers:
            p.stdin.write("go\n");p.stdin.flush()
        results=[]
        for p in workers:
            output,errors=p.communicate(timeout=30)
            assert p.returncode==0,errors
            results.append({"pid":p.pid,"value":json.loads(output)["value"]})
        return results
    finally:
        for p in workers:
            if p.poll() is None:
                p.kill()
            p.communicate(timeout=10)
        pool.shutdown(wait=True, cancel_futures=True)


def schema_experiment(work):
    store=Store(work/"schema.sqlite")
    def tables():
        with store.connection() as connection:
            return [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    base=tables()
    assert len(base)==7
    store.create("A","切片共享");store.create("B","检查点恢复")
    store.advance("A");store.advance("B")
    seq={task:store.audit(task)["events"] for task in ["A","B"]}
    assert [r["seq"] for r in seq["A"]]==[1,3]
    Queue(store)
    assert len(tables())==8
    return {"store_tables":base,"queue_tables":tables(),"interleaved_events":seq,
            "ids":{"message_id":"msg-17","call_id":"call-17","thread_id":"thread-17","run_id":"run-17"}}


def state_experiment(work):
    path=work/"states.sqlite";store=Store(path)
    records=[]
    for action in ["create","step","step","confirm"]:
        result=child(action,path,version=2 if action=="confirm" else None)
        records.append({"action":action,"pid":result["pid"],**snapshot(store,"run-17")})
    assert [r["status"] for r in records]==["created","prepared","waiting","done"]
    assert [r["version"] for r in records]==[0,1,2,3]
    final=snapshot(store,"run-17")
    child("confirm",path,version=2)
    assert snapshot(store,"run-17")==final
    errors={"changed_question":rejected(lambda:store.create("run-17","另一问题")),
            "other_owner":rejected(lambda:store.get("run-17","bob")),
            "cancel_done":rejected(lambda:store.cancel("run-17"))}
    for status,steps in [("created",0),("prepared",1),("waiting",2)]:
        task="cancel-"+status;store.create(task,"检查点恢复")
        for _ in range(steps):store.advance(task)
        store.cancel(task);store.cancel(task)
        assert store.get(task)["status"]=="cancelled" and store.audit(task)["plan_count"]==0
    store.create("empty","火星天气")
    errors["empty_evidence"]=rejected(lambda:store.advance("empty"))
    errors["early_confirm"]=rejected(lambda:store.confirm("empty"))
    return {"stages":records,"after_repeat":final,"rejected":errors,"empty_status":store.get("empty")["status"]}


def version_experiment(work):
    path=work/"versions.sqlite";store=Store(path);ready(store,"review-17")
    barrier=Barrier(2)
    def update(title):
        local=Store(path);barrier.wait(timeout=10)
        try:
            value=revise_waiting(local,"review-17",title,2)
            return {"title":title,"accepted":True,"version":value["version"]}
        except ValueError as error:
            return {"title":title,"accepted":False,"error":str(error)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(update,["方案 A","方案 B"]))
    assert sum(r["accepted"] for r in results)==1
    stale=rejected(lambda:store.confirm("review-17",expected_version=2))
    before=snapshot(store,"review-17")
    assert before["version"]==3 and before["plans"]==0
    store.confirm("review-17",expected_version=3)
    return {"concurrent_updates":results,"stale_confirmation":stale,
            "before_confirmation":before,"after_confirmation":snapshot(store,"review-17")}


def atomic_experiment(work):
    path=work/"atomic.sqlite";store=Store(path)
    ready(store,"visibility")
    with store.connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO plans VALUES(?,?,?)",("visibility","alice","{}"))
        writer_count=connection.execute("SELECT count(*) FROM plans").fetchone()[0]
        reader_count=store.audit("visibility")["plan_count"]
        connection.rollback()
    assert (writer_count,reader_count)==(1,0)
    faults=[]
    for failure,code in [("before_commit",2),("after_commit",2),("hard_before_commit",71),("hard_after_commit",72)]:
        task="fault-"+failure;ready(store,task)
        outcome=child("confirm",path,expected=code,run_id=task,fail=failure)
        observed=child("show",path,run_id=task)["value"]
        committed="after" in failure
        assert observed["plans"]==int(committed) and len(observed["outbox"])==int(committed)
        assert observed["status"]==("done" if committed else "waiting")
        child("confirm",path,run_id=task,version=2)
        final=snapshot(store,task)
        assert final["plans"]==1 and len(final["outbox"])==1 and final["events"].count("done")==1
        faults.append({"failure":failure,"process":outcome,"observed":observed,"after_retry":final})
    return {"uncommitted_visibility":{"writer":writer_count,"other_connection":reader_count},"faults":faults}


def outbox_experiment(work):
    sender=work/"sender.sqlite";receiver=work/"receiver.sqlite"
    store=Store(sender);ready(store,"delivery-17");store.confirm("delivery-17")
    failure=child("deliver",sender,expected=2,receiver=str(receiver),fail_after_send=True)
    ledger=ReceiptLedger(receiver)
    before={"sender":store.outbox()[0]["status"],"receiver_rows":ledger.count()}
    success=child("deliver",sender,receiver=str(receiver))
    after={"sender":store.outbox()[0]["status"],"receiver_rows":ledger.count()}
    assert before=={"sender":"pending","receiver_rows":1} and after=={"sender":"delivered","receiver_rows":1}
    conflict=rejected(lambda:ledger.accept("plan:delivery-17",{"changed":True}))
    return {"failed_process":failure,"retry_process":success,"before_retry":before,"after_retry":after,
            "delivery_attempts":2,"key":"plan:delivery-17","different_payload_rejected":conflict}


def lease_experiment(work):
    path=work/"lease.sqlite";store=Store(path);queue=Queue(store)
    queue.enqueue("job-17","timer:weekly:17",100);queue.enqueue("job-17","timer:weekly:17",100)
    claims=concurrent_claims(path)
    assert sum(r["value"] is not None for r in claims)==1
    old=next(r["value"] for r in claims if r["value"])
    with store.transaction() as connection:
        connection.execute("CREATE TABLE resource_effects(job_id TEXT PRIMARY KEY,token INTEGER NOT NULL,body TEXT NOT NULL)")
        connection.execute("CREATE TABLE unsafe_effects(job_id TEXT PRIMARY KEY,body TEXT NOT NULL)")
    expired=rejected(lambda:write_resource(path,"job-17",old["token"],130,"旧结果"))
    expired_finish=rejected(lambda:queue.finish("job-17",old["token"],130))
    newer=queue.claim(131)
    write_resource(path,"job-17",newer["token"],132,"新结果")
    stale=rejected(lambda:write_resource(path,"job-17",old["token"],133,"旧结果"))
    # 对照组省掉写入处的领取检查，实际演示晚到旧结果覆盖新结果。
    with store.transaction() as connection:
        for body in ["新结果","旧结果"]:
            connection.execute("INSERT INTO unsafe_effects VALUES('job-17',?) ON CONFLICT(job_id) DO UPDATE SET body=excluded.body",(body,))
        protected=dict(connection.execute("SELECT * FROM resource_effects").fetchone())
        unprotected=connection.execute("SELECT body FROM unsafe_effects").fetchone()[0]
    assert protected["body"]=="新结果" and unprotected=="旧结果"
    queue.finish("job-17",newer["token"],134)
    queue.enqueue("renew","manual",200);renewal=queue.claim(200)
    queue.renew("renew",renewal["token"],220)
    assert queue.get("renew")["lease_until"]==250 and queue.claim(231) is None
    queue.finish("renew",renewal["token"],240)
    return {"competing_processes":claims,"old":old,"new":newer,"expired_write":expired,
            "expired_finish":expired_finish,"stale_write":stale,"protected":protected,"unprotected":unprotected,
            "final_job":queue.get("job-17"),"renewed_until":250}


def graph_experiment(work):
    path=work/"graph.sqlite";task="graph-17";thread="thread-17"
    args={"run_id":task,"thread_id":thread}
    started=child("graph-start",path,**args)
    paused=child("graph-show",path,**args)["value"]
    wrong=child("graph-resume",path,expected=2,run_id="other-run",thread_id=thread)
    unknown=child("graph-resume",path,expected=2,run_id=task,thread_id="unknown")
    failed=child("graph-resume",path,expected=2,fail="after_commit",**args)
    after_failure=child("graph-show",path,**args)["value"]
    business=snapshot(Store(path),task)
    child("graph-continue",path,**args)
    done=child("graph-show",path,**args)["value"]
    assert paused["next"]==["review"] and after_failure["next"]==["commit"] and done["next"]==[]
    assert business["plans"]==1 and snapshot(Store(path),task)["plans"]==1
    child("graph-start",path,run_id="cancelled",thread_id="cancel-thread")
    child("graph-cancel",path,run_id="cancelled",thread_id="cancel-thread")
    assert snapshot(Store(path),"cancelled")["status"]=="cancelled"
    probe=work/"probe.sqlite"
    first=child("probe-start",probe);second=child("probe-resume",probe)
    assert first["value"]["visits"]==["prepare","review:before"]
    assert second["value"]["visits"]==["prepare","review:before","review:before","review:after"]
    return {"started_pid":started["pid"],"paused_next":paused["next"],"wrong_run":wrong["error"],
            "unknown_thread":unknown["error"],"failure":failed,"pending_after_failure":after_failure["next"],
            "business_before_retry":business,"final_next":done["next"],"final_business":snapshot(Store(path),task),
            "cancelled":snapshot(Store(path),"cancelled"),"probe_start":first,"probe_resume":second,
            "versions":{name:metadata.version(name) for name in ["langgraph","langgraph-checkpoint-sqlite"]}}


def integrated_experiment(work):
    path=work/"integrated.sqlite";receiver=work/"integrated-receiver.sqlite"
    store=Store(path);queue=Queue(store);task="run-17";job="job-17"
    queue.enqueue(job,"timer:weekly:17",100);old=queue.claim(100)
    trace=[]
    for action in ["create","step","step"]:
        value=child(action,path,run_id=task)
        trace.append({"action":action,"pid":value["pid"],"state":snapshot(store,task)})
    newer=queue.claim(131)
    stale=child("claimed-confirm",path,expected=2,run_id=task,job_id=job,token=old["token"],now=132,version=2)
    assert store.audit(task)["plan_count"]==0
    for _ in range(2):
        child("claimed-confirm",path,run_id=task,job_id=job,token=newer["token"],now=132,version=2)
    child("deliver",path,expected=2,receiver=str(receiver),fail_after_send=True)
    pending=store.outbox()[0]["status"]
    child("deliver",path,receiver=str(receiver))
    queue.finish(job,newer["token"],133)
    result={"trace":trace,"old_claim":old,"new_claim":newer,"stale_worker":stale,
            "sender_before_retry":pending,"business":snapshot(store,task),"queue":queue.get(job),
            "receiver_rows":ReceiptLedger(receiver).count()}
    assert result["business"]["plans"]==1 and result["receiver_rows"]==1
    assert result["business"]["outbox"][0]["status"]=="delivered"
    return result


EXPERIMENTS={"schema":schema_experiment,"states":state_experiment,"versions":version_experiment,
             "atomic":atomic_experiment,"outbox":outbox_experiment,"leases":lease_experiment,
             "graph":graph_experiment,"integrated":integrated_experiment}


def run_experiment(mode="all"):
    assert Path(sys.prefix).resolve()==(PROJECT/".venv").resolve()
    tmp=ROOT/".tmp";tmp.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="chapter09-",dir=tmp) as directory:
        work=Path(directory)
        if mode=="all":
            return {key:fn(work) for key,fn in EXPERIMENTS.items()}
        return EXPERIMENTS[mode](work)


def exercise(code):
    choices={f"RT-{i:02}":name for i,name in enumerate(EXPERIMENTS,1)}
    if code not in choices:
        raise ValueError("未知 RT 题号")
    return run_experiment(choices[code])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv)>1 and sys.argv[1]=="worker":
        try:
            value=worker(json.loads(sys.stdin.readline()))
            print(json.dumps({"pid":os.getpid(),"value":value},ensure_ascii=False,default=str),flush=True)
            return 0
        except (ValueError,RuntimeError) as error:
            print(encode({"pid":os.getpid(),"error":str(error)}),flush=True)
            return 2
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["all",*EXPERIMENTS],nargs="?",default="all")
    args=parser.parse_args()
    print(json.dumps(run_experiment(args.mode),ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
