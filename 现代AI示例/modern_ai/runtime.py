"""SQLite 状态机、确认、幂等提交、外部记忆与上下文预算。"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from .retrieval import search, embed


def encode(value):
    # 紧凑序列化规则固定，字节预算才能逐项复现。
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.connection() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                version INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plans(
                run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                kind TEXT NOT NULL, body TEXT NOT NULL, at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories(
                owner TEXT NOT NULL, key TEXT NOT NULL, raw TEXT NOT NULL,
                summary TEXT NOT NULL, digest TEXT NOT NULL, revision INTEGER NOT NULL,
                PRIMARY KEY(owner,key));
            CREATE TABLE IF NOT EXISTS memory_history(
                owner TEXT NOT NULL, key TEXT NOT NULL, revision INTEGER NOT NULL,
                raw TEXT NOT NULL, summary TEXT NOT NULL, digest TEXT NOT NULL,
                PRIMARY KEY(owner,key,revision));
            CREATE TABLE IF NOT EXISTS outbox(
                key TEXT PRIMARY KEY, owner TEXT NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', receipt TEXT);
            CREATE TABLE IF NOT EXISTS attempts(
                run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, reason TEXT NOT NULL,
                details TEXT NOT NULL, PRIMARY KEY(run_id,attempt_id));
            INSERT OR IGNORE INTO memory_history(owner,key,revision,raw,summary,digest)
                SELECT owner,key,revision,raw,summary,digest FROM memories;
            """)

    @contextmanager
    def connection(self):
        # with sqlite3.connect 本身不负责 close，这里统一关闭句柄。
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def event(self, connection, run_id, kind, body):
        connection.execute('INSERT INTO events(run_id,kind,body,at) VALUES(?,?,?,?)',
                           (run_id, kind, encode(body), datetime.now(timezone.utc).isoformat()))

    def get(self, run_id, owner='alice'):
        with self.connection() as connection:
            row = connection.execute('SELECT * FROM runs WHERE id=? AND owner=?',
                                     (run_id, owner)).fetchone()
        if row is None:
            raise ValueError('没有可读任务')
        result = dict(row)
        result['payload'] = json.loads(result['payload'])
        return result

    def create(self, run_id, question, owner='alice'):
        with self.transaction() as connection:
            # 相同 id 的不同请求不能被静默合并。
            row = connection.execute('SELECT owner,payload FROM runs WHERE id=?', (run_id,)).fetchone()
            if row:
                if row['owner'] != owner or json.loads(row['payload'])['question'] != question:
                    raise ValueError('任务编号与原请求冲突')
            else:
                payload = {'question':question, 'messages':[{'role':'user','content':question}]}
                connection.execute('INSERT INTO runs VALUES(?,?,?,0,?)',
                                   (run_id, owner, 'created', encode(payload)))
                self.event(connection, run_id, 'created', {'next':'prepare'})
        return self.get(run_id, owner)

    def advance(self, run_id, owner='alice'):
        snapshot = self.get(run_id, owner)
        # 检索在写事务外完成，避免远程工作占住 SQLite 写锁。
        if snapshot['status'] == 'created':
            evidence = search(snapshot['payload']['question'], owner)['selected']
            proposal = {'title':'阅读计划', 'evidence_ids':[item['id'] for item in evidence],
                        'items':[item['text'] for item in evidence]}
            if not evidence:
                raise ValueError('没有来源，不能形成计划')
        with self.transaction() as connection:
            row = connection.execute('SELECT * FROM runs WHERE id=? AND owner=?',
                                     (run_id, owner)).fetchone()
            payload = json.loads(row['payload'])
            if row['status'] == 'created':
                payload['proposal'] = proposal
                connection.execute("UPDATE runs SET status='prepared',version=version+1,payload=? WHERE id=?",
                                   (encode(payload), run_id))
                self.event(connection, run_id, 'prepared', {'next':'request_confirmation'})
            elif row['status'] == 'prepared':
                connection.execute("UPDATE runs SET status='waiting',version=version+1 WHERE id=?", (run_id,))
                self.event(connection, run_id, 'waiting', {'next':'confirm'})
            # waiting、done 和 cancelled 都不会被自动推进。
        return self.get(run_id, owner)

    def confirm(self, run_id, owner='alice', fail=None, expected_version=None):
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise ValueError('确认版本必须为非负整数')
        with self.transaction() as connection:
            row = connection.execute('SELECT * FROM runs WHERE id=? AND owner=?',
                                     (run_id, owner)).fetchone()
            if row is None:
                raise ValueError('没有可确认任务')
            if row['status'] == 'done':
                return self.get(run_id, owner)
            if row['status'] != 'waiting':
                raise ValueError('任务不在等待确认状态')
            if expected_version is not None and row['version'] != expected_version:
                raise ValueError('确认版本已经变化')
            proposal = json.loads(row['payload'])['proposal']
            # 同一数据库事务内提交业务记录、完成状态与追踪。
            connection.execute('INSERT INTO plans VALUES(?,?,?) ON CONFLICT(run_id) DO NOTHING',
                               (run_id, owner, encode(proposal)))
            if fail == 'before_commit':
                raise RuntimeError('故障注入：提交前中断')
            if fail == 'hard_before_commit':
                # 仅用于读者显式选择的故障实验；退出后由 SQLite 恢复未提交事务。
                os._exit(71)
            connection.execute("UPDATE runs SET status='done',version=version+1 WHERE id=?", (run_id,))
            self.event(connection, run_id, 'done', {'plan_key':run_id, 'next':None})
            connection.execute('INSERT INTO outbox(key,owner,payload) VALUES(?,?,?)',
                               ('plan:'+run_id, owner, encode(proposal)))
        if fail == 'after_commit':
            raise RuntimeError('故障注入：已提交但响应丢失')
        if fail == 'hard_after_commit':
            os._exit(72)
        return self.get(run_id, owner)

    def cancel(self, run_id, owner='alice'):
        with self.transaction() as connection:
            row = connection.execute('SELECT status FROM runs WHERE id=? AND owner=?',
                                     (run_id, owner)).fetchone()
            if row is None or row['status'] == 'done':
                raise ValueError('任务不能取消')
            if row['status'] == 'cancelled':
                return self.get(run_id, owner)
            connection.execute("UPDATE runs SET status='cancelled',version=version+1 WHERE id=?", (run_id,))
            self.event(connection, run_id, 'cancelled', {'next':None})
        return self.get(run_id, owner)

    def audit(self, run_id, owner='alice'):
        self.get(run_id, owner)
        with self.connection() as connection:
            events = [dict(row) for row in connection.execute(
                'SELECT seq,kind,body FROM events WHERE run_id=? ORDER BY seq', (run_id,))]
            plans = connection.execute('SELECT count(*) FROM plans WHERE run_id=?', (run_id,)).fetchone()[0]
        return {'events':events, 'plan_count':plans}

    def remember(self, key, raw, summary, owner='alice'):
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.transaction() as connection:
            connection.execute("""
            INSERT INTO memories VALUES(?,?,?,?,?,1)
            ON CONFLICT(owner,key) DO UPDATE SET raw=excluded.raw,summary=excluded.summary,
                digest=excluded.digest,revision=memories.revision+1
            """, (owner, key, raw, summary, digest))
            current = connection.execute('SELECT * FROM memories WHERE owner=? AND key=?',
                                         (owner,key)).fetchone()
            connection.execute('INSERT INTO memory_history VALUES(?,?,?,?,?,?)',
                               (owner,key,current['revision'],raw,summary,digest))

    def memory_versions(self, key, owner='alice'):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM memory_history WHERE owner=? AND key=? ORDER BY revision',
                (owner,key))]

    def outbox(self, owner='alice'):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM outbox WHERE owner=? ORDER BY key',(owner,))]

    def deliver(self, send, owner='alice', fail_after_send=False):
        delivered = []
        for item in self.outbox(owner):
            if item['status'] == 'delivered':
                continue
            # 外部动作在本地写事务之外执行，以稳定 key 请求接收方去重。
            receipt = send(item['key'],json.loads(item['payload']))
            if fail_after_send:
                raise RuntimeError('故障注入：接收方已处理，本地尚未记回执')
            with self.transaction() as connection:
                connection.execute("UPDATE outbox SET status='delivered',receipt=? WHERE key=? AND owner=?",
                                   (encode(receipt),item['key'],owner))
            delivered.append(receipt)
        return delivered

    def record_attempt(self, run_id, attempt_id, reason, details, owner='alice'):
        self.get(run_id,owner)
        body = encode(details)
        with self.transaction() as connection:
            prior = connection.execute('SELECT reason,details FROM attempts WHERE run_id=? AND attempt_id=?',
                                       (run_id,attempt_id)).fetchone()
            if prior and (prior['reason']!=reason or prior['details']!=body):
                raise ValueError('尝试编号与停止记录冲突')
            connection.execute('INSERT INTO attempts VALUES(?,?,?,?) ON CONFLICT DO NOTHING',
                               (run_id,attempt_id,reason,body))

    def attempts(self, run_id, owner='alice'):
        self.get(run_id,owner)
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_id',(run_id,))]

    def recall(self, query, owner='alice'):
        with self.connection() as connection:
            records = [dict(row) for row in connection.execute(
                'SELECT * FROM memories WHERE owner=?', (owner,))]
        # 索引从原记录重建；摘要不替代原始证据。
        q = embed(query)
        for record in records:
            record['score'] = float(embed(record['raw'])@q)
        return sorted([r for r in records if r['score']>0], key=lambda r:-r['score'])


def assemble(question, optional, total=256, output=64):
    # 本项目分词器是字节表，JSON 序列化后每个 UTF-8 字节占一个 Token。
    if type(total) is not int or type(output) is not int or not 0 <= output < total:
        raise ValueError('预算必须为整数且满足 0 <= output < total')
    messages = [{'role':'system','content':'只根据资料回答。'},
                {'role':'user','content':question}]
    limit, selected, omitted, decisions = total-output, [], [], []
    def cost(value):
        return len(encode(value).encode('utf-8'))
    if cost(messages)>limit:
        raise ValueError('必选输入已超预算')
    for item in optional:
        candidate = messages+[{'role':'context','content':item['text']}]
        decisions.append({'id':item['id'],'before':cost(messages),'candidate':cost(candidate),
                          'limit':limit,'accepted':cost(candidate)<=limit})
        if cost(candidate)<=limit:
            messages = candidate
            selected.append(item['id'])
        else:
            omitted.append(item['id'])
    return {'messages':messages, 'input_tokens':cost(messages), 'output_reserved':output,
            'selected':selected, 'omitted':omitted, 'total':total,'decisions':decisions}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start','step','confirm','cancel','show'])
    parser.add_argument('--db', required=True)
    parser.add_argument('--id', default='run-17')
    parser.add_argument('--question', default='检查点恢复')
    parser.add_argument('--fail', choices=['before_commit','after_commit','hard_before_commit','hard_after_commit'])
    args = parser.parse_args()
    store = Store(args.db)
    if args.action == 'start':
        result = store.create(args.id, args.question)
        # 两次提交分别留下 prepared 与 waiting 恢复位置。
        while result['status'] in ['created','prepared']:
            result = store.advance(args.id)
    elif args.action == 'step':
        result = store.advance(args.id)
    elif args.action == 'confirm':
        result = store.confirm(args.id, fail=args.fail)
    elif args.action == 'cancel':
        result = store.cancel(args.id)
    else:
        result = {'run':store.get(args.id), 'audit':store.audit(args.id)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
