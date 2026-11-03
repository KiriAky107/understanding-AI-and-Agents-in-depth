"""可追踪的切分、计数表示、BM25、混合排序与来源回答。"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import numpy as np

DATA = Path(__file__).resolve().parents[1]/'data'/'notes.json'
VOCAB = ['切片', '共享', '复制', '数据库', '检查点', '事务', '幂等', '恢复', '计划']


def tokenize(text):
    # 英文按词、中文按字，规则明确可复现，不声称保留完整中文词义。
    return re.findall(r'[a-z_]+|[\u4e00-\u9fff]', text.lower())


def embed(text):
    # 固定词表计数再归一化，保留每列含义与零向量。
    vector = np.array([text.count(word) for word in VOCAB], dtype=float)
    length = np.linalg.norm(vector)
    return vector/length if length else vector


def load_chunks(owner='alice', latest=True, as_of=None):
    records = json.loads(DATA.read_text(encoding='utf-8-sig'))
    # 先按应用身份过滤；工具参数不能更改身份。
    allowed = [record for record in records if record['owner'] in ['public', owner]
               and (as_of is None or record['revision'] <= as_of)]
    versions = {}
    for record in allowed:
        versions[record['source']] = max(versions.get(record['source'], ''), record['revision'])
    chunks = []
    for record in allowed:
        if latest and record['revision'] != versions[record['source']]:
            continue
        offset = 0
        for paragraph in record['text'].split('\n\n'):
            # 短语料按段切分，编号含源版本和原文偏移。
            item = {key:record[key] for key in ['source', 'owner', 'revision']}
            item.update(id=f"{record['id']}:{offset}", start=offset, end=offset+len(paragraph),
                        text=paragraph, hash=sha256(paragraph.encode()).hexdigest()[:12])
            chunks.append(item)
            offset += len(paragraph)+2
    return chunks


def search(question, owner='alice', k=3, latest=True, as_of=None):
    if type(k) is not int or k < 1:
        raise ValueError('k 必须为正整数')
    chunks = load_chunks(owner, latest, as_of)
    if not chunks:
        return {'question':question, 'owner':owner, 'candidates':[], 'selected':[]}
    query, words = embed(question), tokenize(question)
    counts = [Counter(tokenize(item['text'])) for item in chunks]
    average = sum(sum(c.values()) for c in counts)/len(counts)
    lexical, vector, contributions = [], [], []
    for index, (item, count) in enumerate(zip(chunks, counts)):
        # BM25 用词频饱和与文档长度修正，参数 k1=1.2、b=.75。
        value, detail = 0., []
        for word in sorted(set(words)):
            frequency = sum(word in other for other in counts)
            inverse = math.log(1+(len(chunks)-frequency+.5)/(frequency+.5))
            tf = count[word]
            part = inverse*tf*2.2/(tf+1.2*(.25+.75*sum(count.values())/average))
            value += part
            detail.append({'term':word,'tf':tf,'df':frequency,'idf':inverse,'contribution':part})
        lexical.append(value)
        contributions.append(detail)
        vector.append(float(embed(item['text']) @ query))
    vector_order = sorted(range(len(chunks)), key=lambda i:(-vector[i], chunks[i]['id']))
    lexical_order = sorted(range(len(chunks)), key=lambda i:(-lexical[i], chunks[i]['id']))
    vr = {i:r+1 for r,i in enumerate(vector_order)}
    br = {i:r+1 for r,i in enumerate(lexical_order)}
    candidates = []
    for i,item in enumerate(chunks):
        if vector[i] <= 0 and lexical[i] <= 0:
            continue
        # 只给有正证据的检索通道加 RRF，避免零分列表制造相关性。
        fused = (1/(60+vr[i]) if vector[i]>0 else 0)+(1/(60+br[i]) if lexical[i]>0 else 0)
        candidates.append(dict(item, cosine=vector[i], bm25=lexical[i], fused=fused,
                               vector_rank=vr[i] if vector[i]>0 else None,
                               keyword_rank=br[i] if lexical[i]>0 else None,
                               bm25_terms=contributions[i]))
    candidates.sort(key=lambda item:(-item['fused'], item['id']))
    # 版本与权限已在召回前过滤；重排在候选中优先覆盖问题中的明确词项。
    terms = [word for word in VOCAB if word in question]
    selected = sorted(candidates[:max(k*2, 4)],
                      key=lambda item:(-sum(t in item['text'] for t in terms), -item['fused']))[:k]
    return {'question':question, 'owner':owner, 'candidates':candidates, 'selected':selected,
            'corpus_size':len(chunks), 'average_length':average}


def answer(question):
    # 确定性抽取回答让控制流与生成质量分开核验。
    result = search(question)
    evidence = result['selected']
    result['answer'] = ('\n'.join(f"{item['text']} [{item['id']}]" for item in evidence)
                        if evidence else '资料中没有可引用的证据。')
    return result


def demo():
    return {'ordinary':answer('切片共享'), 'missing':answer('火星天气'),
            'recovery':answer('检查点恢复'),
            'old_and_new':search('计划恢复', latest=False)}
