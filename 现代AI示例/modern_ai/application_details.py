"""第五篇逐步观测：排名、检索评价、引用与三种调用封装。"""
import json
import threading
from .agent import execute
from .retrieval import search
from .transport import create_server, post_json
from .mcp_stdio import client_demo


def rank_example():
    # 两个合法排名排列；交换第二通道的 A/B，而非让两者同时成为第一。
    before = {'A':1/61+1/63, 'B':1/62+1/61}
    after = {'A':1/61+1/61, 'B':1/62+1/63}
    return {'before':before,'after':after,
            'orders':{name:sorted(values,key=lambda key:-values[key])
                      for name,values in [('before',before),('after',after)]}}


def retrieval_evaluation():
    # 日期与标准片段由问题标注确定，不拿检索输出反过来生成标准答案。
    cases = [
        {'id':'sharing','question':'切片共享','relevant':['numpy:0']},
        {'id':'copy','question':'复制后原数组','relevant':['numpy:14']},
        {'id':'current','question':'计划恢复','relevant':['runtime:0','runtime:27']},
        {'id':'history','question':'计划恢复','as_of':'2025-01-01','relevant':['old:0']},
        {'id':'missing','question':'火星天气','relevant':[]}
    ]
    rows = []
    for case in cases:
        result = search(case['question'],as_of=case.get('as_of'))
        ids = [item['id'] for item in result['selected']]
        gold = set(case['relevant'])
        # 无相关资料的题单独检查空结果，不混入 Recall/MRR 的平均。
        recall = len(gold.intersection(ids))/len(gold) if gold else None
        rank = next((i+1 for i,key in enumerate(ids) if key in gold),None)
        reciprocal = (1/rank if rank else 0.) if gold else None
        rows.append(dict(case,selected=ids,recall_at_3=recall,rr_at_3=reciprocal))
    answerable = [row for row in rows if row['relevant']]
    recall_at_1 = sum(len(set(row['relevant']).intersection(row['selected'][:1]))/
                      len(row['relevant']) for row in answerable)/len(answerable)
    return {'cases':rows,'mean_recall_at_1':recall_at_1,
            'mean_recall_at_3':sum(r['recall_at_3'] for r in answerable)/len(answerable),
            'mrr_at_3':sum(r['rr_at_3'] for r in answerable)/len(answerable),
            'missing_returns_empty':not rows[-1]['selected']}


def citation_examples():
    supplied = {item['id'] for item in search('切片共享')['selected']}
    # supported 是逐条核读后的人工标注，不用“出现了引用号”推定语义支持。
    annotations = [
        {'claim':'切片共享数据。','citation':'numpy:0','supported':True},
        {'claim':'切片会自动复制数据。','citation':'numpy:0','supported':False},
        {'claim':'复制后原数组保持不变。','citation':'numpy:14','supported':True}
    ]
    return [dict(item,in_input=item['citation'] in supplied,
                 accepted=item['citation'] in supplied and item['supported'])
            for item in annotations]


def transport_comparison():
    arguments = {'pages':3,'questions':4}
    direct = execute('reading_minutes',arguments)
    # 只访问工具路径，占位对象避免无关训练；端口 0 由系统分配空闲端口。
    server = create_server(port=0,model=object())
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        http = post_json(f'http://127.0.0.1:{server.server_port}',
                         '/tools/reading_minutes',arguments)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    mcp = client_demo()['responses'][-1]
    unwrapped = json.loads(mcp['result']['content'][0]['text'])
    return {'arguments':arguments,'function':direct,'http':http,'mcp':mcp,
            'same_value':direct == http == unwrapped}


def demo():
    return {'ranking':rank_example(),'retrieval':retrieval_evaluation(),
            'citations':citation_examples(),'transports':transport_comparison()}


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(demo(),ensure_ascii=False,indent=2))
