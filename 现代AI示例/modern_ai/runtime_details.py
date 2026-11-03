"""第六篇逐项实验：同一源码、新进程、故障、租约、记忆与停止记录。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from .agent import BudgetExceeded, run
from .harness import Queue, communication_cost
from .runtime import Store, assemble, encode

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'现代AI示例'


def child(module, *args, expected=0):
    # 子进程复用唯一解释器与正式源码，不复制项目。
    process=subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],
                           cwd=PROJECT,capture_output=True,text=True,encoding='utf-8',timeout=40)
    if expected is None:
        assert process.returncode!=0
    else:
        assert process.returncode==expected,process.stderr
    return {'exit':process.returncode,
            'value':json.loads(process.stdout) if process.returncode==0 else None,
            'last_error':process.stderr.strip().splitlines()[-1] if process.stderr.strip() else ''}


def compact(store,task):
    run=store.get(task)
    return {'status':run['status'],'version':run['version'],
            'has_proposal':'proposal' in run['payload'],
            'plans':store.audit(task)['plan_count'],
            'events':[e['kind'] for e in store.audit(task)['events']]}


def ready(store,task):
    store.create(task,'检查点恢复')
    store.advance(task);store.advance(task)
    return store.get(task)


def state_experiment(work):
    path=work/'state.sqlite'
    store=Store(path)
    store.create('run-17','检查点恢复')
    stages=[compact(store,'run-17')]
    # 每个 step/show 都是新进程，明确在 prepared 位置退出后还能继续。
    child('modern_ai.runtime','step','--db',path,'--id','run-17')
    stages.append(compact(Store(path),'run-17'))
    child('modern_ai.runtime','step','--db',path,'--id','run-17')
    stages.append(compact(Store(path),'run-17'))
    child('modern_ai.runtime','confirm','--db',path,'--id','run-17')
    stages.append(compact(Store(path),'run-17'))
    child('modern_ai.runtime','confirm','--db',path,'--id','run-17')
    return {'stages':stages,'after_repeat':compact(store,'run-17')}


def fault_experiment(work):
    path=work/'fault.sqlite'
    store=Store(path)
    rows=[]
    for failure in ['before_commit','after_commit','hard_before_commit','hard_after_commit']:
        task='fault-'+failure
        ready(store,task)
        result=child('modern_ai.runtime','confirm','--db',path,'--id',task,
                     '--fail',failure,expected=None)
        # 查询也在新进程中发生；异常栈与数据库事实分开保留。
        observed=child('modern_ai.runtime','show','--db',path,'--id',task)['value']
        row={'failure':failure,'exit':result['exit'],'error':result['last_error'],
             'status_after_failure':observed['run']['status'],
             'plans_after_failure':observed['audit']['plan_count']}
        child('modern_ai.runtime','confirm','--db',path,'--id',task)
        row['after_retry']=compact(Store(path),task)
        rows.append(row)
    return rows


def memory_experiment(work):
    path=work/'memory.sqlite'
    store=Store(path)
    raw='检查点恢复：先确认，计划上限10分钟。'
    summary='检查点支持恢复。'
    store.remember('rule',raw,summary)
    history=[{'id':'early','text':raw},{'id':'recent','text':'最近一次运行已退出。'}]
    # 滑动窗口只留最后一项，早期数字与确认条件一起消失。
    window=history[-1:]
    recalled=Store(path).recall('检查点恢复')[0]
    old_digest=sha256(raw.encode()).hexdigest()
    store.remember('rule',raw.replace('10','20'),'检查点支持恢复。')
    versions=store.memory_versions('rule')
    latest=store.recall('检查点恢复')[0]
    # 当前索引已经对应第 2 版；按历史修订读回第 1 版后再组装。
    context=assemble('如何恢复',[{'id':'rule:1','text':versions[0]['raw']}])
    return {'window_ids':[x['id'] for x in window],
            'window_has_limit':any('10' in x['text'] for x in window),
            'summary_has_limit':'10' in summary,
            'recalled_raw':recalled['raw'],'digest_matches':recalled['digest']==old_digest,
            'context':context,'history_revisions':[v['revision'] for v in versions],
            'historical_raw':versions[0]['raw'],'latest_raw':latest['raw']}


def lease_experiment(work):
    path=work/'queue.sqlite'
    store=Store(path);queue=Queue(store)
    queue.enqueue('q','timer:weekly',100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims=list(pool.map(lambda _:Queue(Store(path)).claim(100),range(2)))
    old=next(c for c in claims if c)
    # 刚好到期也不可完成，即使尚未被另一执行者领取。
    try:
        queue.finish('q',old['token'],130)
    except ValueError as error:
        expired=str(error)
    else:
        raise AssertionError('到期领取仍能完成')
    new=queue.claim(131)
    try:
        queue.finish('q',old['token'],132)
    except ValueError as error:
        stale=str(error)
    else:
        raise AssertionError('旧令牌仍能完成')
    ready(store,'business-q')
    store.confirm('business-q');store.confirm('business-q')
    queue.finish('q',new['token'],132)
    queue.enqueue('renew','manual',200)
    renewed=queue.claim(200)
    queue.renew('renew',renewed['token'],220)
    until=queue.get('renew')['lease_until']
    assert queue.claim(231) is None
    queue.finish('renew',renewed['token'],240)
    return {'concurrent_claims':sum(c is not None for c in claims),'old':old,'new':new,
            'expired_rejected':expired,'old_rejected':stale,'final_job':queue.get('q'),
            'business_plans':store.audit('business-q')['plan_count'],
            'renewed_until':until,'communication':communication_cost()}


class ReceiptLedger:
    """第二个 SQLite 文件模拟接收方的持久化去重接口，不发送外部消息。"""
    def __init__(self,path):
        self.path=path
        self.calls=0
        connection=sqlite3.connect(path)
        try:
            with connection:
                connection.execute('CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY,body TEXT NOT NULL)')
        finally:
            connection.close()

    def accept(self,key,payload):
        self.calls+=1
        body=encode(payload)
        connection=sqlite3.connect(self.path,timeout=5)
        try:
            with connection:
                connection.execute('BEGIN IMMEDIATE')
                prior=connection.execute('SELECT body FROM receipts WHERE key=?',(key,)).fetchone()
                if prior and prior[0]!=body:
                    raise ValueError('相同幂等键对应不同内容')
                connection.execute('INSERT INTO receipts VALUES(?,?) ON CONFLICT DO NOTHING',(key,body))
            return {'receipt':key}
        finally:
            connection.close()

    def count(self):
        connection=sqlite3.connect(self.path)
        try:
            return connection.execute('SELECT count(*) FROM receipts').fetchone()[0]
        finally:
            connection.close()


def outbox_experiment(work):
    store=Store(work/'sender.sqlite')
    receiver=ReceiptLedger(work/'receiver.sqlite')
    ready(store,'delivery-17');store.confirm('delivery-17')
    try:
        store.deliver(receiver.accept,fail_after_send=True)
    except RuntimeError as error:
        failure=str(error)
    before={'sender':store.outbox()[0]['status'],'receiver_rows':receiver.count()}
    store.deliver(receiver.accept)
    after={'sender':store.outbox()[0]['status'],'receiver_rows':receiver.count(),'requests':receiver.calls}
    assert before=={'sender':'pending','receiver_rows':1}
    assert after=={'sender':'delivered','receiver_rows':1,'requests':2}
    return {'failure':failure,'before_retry':before,'after_retry':after,
            'key':store.outbox()[0]['key']}


def budget_experiment(work):
    store=Store(work/'budget.sqlite')
    store.create('budget-17','估算时长')
    optional=[{'id':'fact','text':'检查点保存状态。'},
              {'id':'long','text':'很长的历史。'*50}]
    context=assemble('如何恢复',optional)
    try:
        assemble('如何恢复',optional,total=24,output=8)
    except ValueError as error:
        tiny_stop=str(error)
    try:
        assemble('很长的问题。'*100,[],total=256,output=64)
    except ValueError as error:
        required_stop=str(error)
    def endless(messages):
        n=sum(m['role']=='tool' for m in messages)
        return {'type':'tool_call','id':f'call-{n+1}','name':'reading_minutes',
                'arguments':'{"pages":3,"questions":4}'}
    try:
        run('估算时长',model=endless,limit=2)
    except BudgetExceeded as error:
        detail={'limit':2,'used':len(error.trace),'trace':error.trace,'messages':error.messages,
                'checkpoint_version':store.get('budget-17')['version']}
        store.record_attempt('budget-17','attempt-1','model_call_budget',detail)
    saved=store.attempts('budget-17')[0]
    details=json.loads(saved['details'])
    assert details['used']==2 and len(details['messages'])==6
    return {'context':context,'tiny_context_rejected':tiny_stop,
            'required_rejected':required_stop,'attempt':saved,
            'business':compact(store,'budget-17')}


def graph_experiment(work):
    path=work/'graph.sqlite'
    records={}
    common=['--db',path,'--id','graph-17','--thread-id','thread-17']
    paused=child('modern_ai.frameworks','start',*common)['value']
    assert '__interrupt__' in paused
    records['paused']=child('modern_ai.frameworks','show',*common)['value']['next']
    failure=child('modern_ai.frameworks','resume',*common,'--fail','after_commit',expected=None)
    records['failure']=failure['last_error']
    store=Store(path)
    records['business_before_graph_retry']=compact(store,'graph-17')
    records['pending_after_failure']=child('modern_ai.frameworks','show',*common)['value']['next']
    resumed=child('modern_ai.frameworks','continue',*common)['value']
    records['resumed_status']=resumed['result']['status']
    records['final_next']=child('modern_ai.frameworks','show',*common)['value']['next']
    records['final_business']=compact(store,'graph-17')
    cancel=['--db',path,'--id','cancel-17','--thread-id','cancel-thread']
    child('modern_ai.frameworks','start',*cancel)
    cancelled=child('modern_ai.frameworks','cancel',*cancel)['value']
    records['cancelled_status']=cancelled['result']['status']
    records['cancelled_plans']=store.audit('cancel-17')['plan_count']
    return records


def exercise(code, work):
    return {'06-01':state_experiment,'06-02':fault_experiment,'06-03':memory_experiment,
            '06-04':lease_experiment,'06-05':budget_experiment}[code](work)


def main():
    tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='runtime-',dir=tmp) as directory:
        work=Path(directory)
        result={code:exercise(code,work) for code in ['06-01','06-02','06-03','06-04','06-05']}
        result['outbox']=outbox_experiment(work)
        result['graph']=graph_experiment(work)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

