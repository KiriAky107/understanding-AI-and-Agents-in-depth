"""针对数学一致性、协议边界与跨进程恢复的集中核验。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter
import numpy as np
from . import mathlab, networks, language, learning
from .agent import execute, strict_json, run
from .extensions import discover, load
from .harness import Queue, read_text, digest_process, communication_cost
from .mcp_stdio import client_demo
from .retrieval import DATA, search, load_chunks
from .runtime import Store, assemble
from .transport import create_server, post_json

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'现代AI示例'


def rejects(action, exception=ValueError):
    # 只有预期类型的异常才算正确拒绝。
    try:
        action()
    except exception:
        return
    raise AssertionError('应拒绝的输入被接受')


def child(module, *args, success=True):
    # 新进程继续使用当前虚拟环境，不复制源码或安装包。
    result = subprocess.run([sys.executable,'-B','-m',module,*map(str,args)],
                            cwd=PROJECT, text=True, encoding='utf-8',
                            capture_output=True, timeout=40)
    if success:
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)
    assert result.returncode != 0
    return result.stderr.splitlines()[-1]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    assert Path(sys.prefix).resolve() == (PROJECT/'.venv').resolve(), '核验必须使用唯一 .venv'
    started = perf_counter()
    report = {'date':'2026-10-31', 'python':sys.version.split()[0],
              'environment':'现代AI示例/.venv',
              'versions':{name:metadata.version(name) for name in
                          ['numpy','torch','scikit-learn','matplotlib','langchain','langgraph',
                           'langgraph-checkpoint-sqlite']}}
    m = mathlab.demo()
    assert m['dot']==12 and m['cosine']==.6
    assert abs(m['finite_difference']-60)<1e-6
    np.testing.assert_allclose(m['pca']['restored'], [[1,2],[2,4],[3,6]], atol=1e-12)
    # 行列式同时按代数规则和基向量围成的面积计算。
    w = np.array([[2.,1.],[-1.,1.]])
    assert np.isclose(np.linalg.det(w),3)
    np.testing.assert_allclose(w@np.array([3.,4.]),[10,1])
    np.testing.assert_allclose(np.linalg.solve(w,[10,1]),[3,4])
    report['math'] = {'dot':12,'cosine':.6,'difference_gradient':m['finite_difference'],
                      'determinant':3,'pca_max_error':float(np.abs(np.array(m['pca']['restored'])-
                                                               [[1,2],[2,4],[3,6]]).max())}
    n = networks.demo()
    assert n['gradient']=={'loss':50.,'dw':60.}
    assert n['temporal']['dr']==.75 and n['temporal']['lstm_c']==.9
    assert n['sequence']['predictions']==[0,1,0,1]
    assert n['cache']['max_error']<1e-12 and n['cache']['future_mass']==0
    report['networks'] = {key:n[key] for key in ['gradient','temporal','cache']}
    report['sequence_loss'] = [n['sequence']['initial_loss'],n['sequence']['final_loss']]
    model, lm = language.train_language(return_model=True)
    assert lm['final_loss']<lm['initial_loss']
    paired, lora = language.multimodal(), language.lora_roundtrip()
    assert paired['paired_predictions']==[0,1] and paired['input_shape']==[1,6,8]
    assert abs(paired['clip_hand']['loss']-.3132616875182228)<1e-12
    assert lora['base_unchanged'] and lora['reload_error']==0
    report.update(language=lm, multimodal={k:paired[k] for k in
                  ['patches','input_shape','alignment_initial','alignment_final','paired_predictions']},
                  lora=lora, learning=learning.experiment())
    retrieved = search('切片共享')
    assert retrieved['selected'][0]['id']=='numpy:0'
    assert all(c['owner']!='bob' for c in load_chunks())
    assert all(c['revision']!='2025-01-01' for c in load_chunks())
    assert not search('火星天气')['selected']
    assert execute('reading_minutes',{'pages':3,'questions':4})=={'minutes':10}
    rejects(lambda:execute('reading_minutes',{'pages':True,'questions':4}))
    rejects(lambda:execute('reading_minutes',{'pages':-1,'questions':4}))
    rejects(lambda:execute('search_notes',{'query':'切片','owner':'bob'}))
    rejects(lambda:strict_json('{"pages":3,"pages":30}'))
    rejects(lambda:strict_json('{"pages":NaN}'))
    loop = run('切片共享')
    assert [m['role'] for m in loop['messages']]==['system','user','assistant','tool','assistant']
    assert [[m['role'] for m in t['input']] for t in loop['trace']]==[
        ['system','user'],['system','user','assistant','tool']]
    rejects(lambda:run('问题', model=lambda _: {'type':'tool_call','id':'same','name':'bad','arguments':'{}'}))
    protocol = client_demo()
    assert json.loads(protocol['responses'][-1]['result']['content'][0]['text'])=={'minutes':10}
    assert discover()[0]['name']=='organize-notes' and '来源' in load('organize-notes')['body']
    assert digest_process(DATA.parent,DATA.name)==sha256(DATA.read_bytes()).hexdigest()
    rejects(lambda:read_text(DATA.parent,'../pyproject.toml'))
    report['retrieval'] = {'selected':[r['id'] for r in retrieved['selected']],
                           'cosine':[r['cosine'] for r in retrieved['selected']],
                           'missing':0,'private_leaks':0}
    report['protocol'] = {'mcp_version':protocol['responses'][0]['result']['protocolVersion'],
                          'reading_minutes':10, 'rejected_bad_inputs':7}
    # 真 HTTP 接入同一已训练模型，完成后关闭服务与线程。
    server = create_server(port=0, model=model)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        generated = post_json(base,'/generate',{'prompt':'read ','max_new_tokens':12})
        assert generated['output_tokens']==12
        assert post_json(base,'/tools/reading_minutes',{'pages':3,'questions':4})=={'minutes':10}
        report['http'] = generated
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    tmp = ROOT/'.tmp'
    tmp.mkdir(exist_ok=True)
    # 所有测试数据库共用一个任务目录，退出（包括失败）即删除。
    with TemporaryDirectory(prefix='check-',dir=tmp) as work:
        db = Path(work)/'state.sqlite'
        initial = child('modern_ai.runtime','start','--db',db)
        assert initial['status']=='waiting'
        store = Store(db)
        assert store.audit('run-17')['plan_count']==0
        before = child('modern_ai.runtime','confirm','--db',db,'--fail','before_commit',success=False)
        assert store.get('run-17')['status']=='waiting' and store.audit('run-17')['plan_count']==0
        after = child('modern_ai.runtime','confirm','--db',db,'--fail','after_commit',success=False)
        assert store.get('run-17')['status']=='done' and store.audit('run-17')['plan_count']==1
        child('modern_ai.runtime','confirm','--db',db)
        assert store.audit('run-17')['plan_count']==1
        kinds = [e['kind'] for e in store.audit('run-17')['events']]
        assert kinds==['created','prepared','waiting','done']
        store.remember('rule','检查点在 SQLite 保存状态。','保存状态')
        store.remember('rule','检查点在 SQLite 保存状态与下一步。','状态与位置')
        memory = store.recall('检查点')[0]
        assert memory['revision']==2 and memory['digest']==sha256(memory['raw'].encode()).hexdigest()
        queue = Queue(store)
        queue.enqueue('queue-1','timer:weekly',100)
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _:queue.claim(100),range(2)))
        old = next(c for c in claims if c)
        assert sum(c is not None for c in claims)==1
        new = queue.claim(131)
        rejects(lambda:queue.finish('queue-1',old['token'],132))
        queue.finish('queue-1',new['token'],132)
        paused = child('modern_ai.frameworks','start','--db',db)
        assert '__interrupt__' in paused
        resumed = child('modern_ai.frameworks','resume','--db',db)
        assert resumed['result']['status']=='done'
        child('modern_ai.frameworks','continue','--db',db)
        assert store.audit('graph-17')['plan_count']==1
        # 同机一次状态读取与上下文组装分别计时。
        tick = perf_counter()
        store.get('run-17')
        read_us = (perf_counter()-tick)*1e6
        tick = perf_counter()
        context = assemble('如何恢复',[{'id':'fact','text':'检查点保存状态。'},
                                      {'id':'long','text':'很长的历史。'*50}])
        context_us = (perf_counter()-tick)*1e6
        assert context['input_tokens']+context['output_reserved']<=256
        assert context['selected']==['fact'] and context['omitted']==['long']
        report['runtime'] = {'events':kinds,'plans_after_retry':1,
                             'before_commit':before,'after_commit':after,
                             'framework_resumed':'done','framework_plan_count':1,
                             'lease_tokens':[old['token'],new['token']],
                             'memory_revision':2, 'sqlite_read_us':read_us}
        report['context'] = dict(context, assemble_us=context_us)
    assert not list(tmp.iterdir()), '任务临时目录清理不完整'
    report['communication'] = communication_cost()
    report['elapsed_ms'] = (perf_counter()-started)*1000
    report['temporary_files_remaining'] = 0
    (ROOT/'docs'/'核验数据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','report':'docs/核验数据.json',
                      'elapsed_ms':report['elapsed_ms'],'temporary_files_remaining':0},ensure_ascii=False))


if __name__ == '__main__':
    main()
