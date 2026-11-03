"""文件边界、固定进程入口、可租约领取的队列与协作通信计量。"""
from pathlib import Path
import json
import math
import subprocess
import sys
from .runtime import Store, encode


def checked_path(workspace, requested):
    # resolve 展开 .. 与符号链接，再检查最终目标仍处于工作区内。
    root = Path(workspace).resolve(strict=True)
    target = (root/requested).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError('文件不在获准工作区内')
    return target


def read_text(workspace, requested):
    target = checked_path(workspace, requested)
    if target.stat().st_size > 1048576:
        raise ValueError('文件超过读取预算')
    return target.read_text(encoding='utf-8-sig')


def digest_process(workspace, requested):
    target = checked_path(workspace, requested)
    if target.stat().st_size > 1048576:
        raise ValueError('文件超过处理预算')
    # 命令文本固定，模型只提供经过校验的路径；不启用 shell。
    code = 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())'
    result = subprocess.run([sys.executable, '-B', '-c', code, str(target)],
                            capture_output=True, text=True, timeout=3, check=True)
    return result.stdout.strip()


class Queue:
    def __init__(self, store):
        self.store = store
        with store.connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, origin TEXT NOT NULL, due REAL NOT NULL,
                lease_until REAL NOT NULL DEFAULT 0, token INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'ready')""")
            connection.commit()

    def enqueue(self, task_id, origin, due):
        # 相同任务来源只入队一次；origin 记录人工、事件或定时规则。
        with self.store.transaction() as connection:
            prior=connection.execute('SELECT origin,due FROM jobs WHERE id=?',(task_id,)).fetchone()
            if prior and (prior['origin']!=origin or prior['due']!=due):
                raise ValueError('队列编号与原入队记录冲突')
            connection.execute('INSERT INTO jobs(id,origin,due) VALUES(?,?,?) ON CONFLICT(id) DO NOTHING',
                               (task_id, origin, due))

    def claim(self, now, lease=30):
        if not math.isfinite(now) or not math.isfinite(lease) or lease <= 0:
            raise ValueError('领取时间和租期不合法')
        with self.store.transaction() as connection:
            row = connection.execute("""SELECT * FROM jobs
                WHERE status!='done' AND due<=? AND lease_until<=? ORDER BY due,id LIMIT 1""",
                                     (now, now)).fetchone()
            if row is None:
                return None
            token = row['token']+1
            connection.execute("UPDATE jobs SET lease_until=?,token=?,status='running' WHERE id=?",
                               (now+lease, token, row['id']))
            return {'id':row['id'], 'token':token, 'origin':row['origin'], 'lease_until':now+lease}

    def finish(self, task_id, token, now):
        with self.store.transaction() as connection:
            changed = connection.execute("""UPDATE jobs SET status='done'
                WHERE id=? AND token=? AND status='running' AND lease_until>?""", (task_id, token, now)).rowcount
            if not changed:
                raise ValueError('租约已过期、被替代或任务已经完成')

    def renew(self, task_id, token, now, lease=30):
        if not math.isfinite(now) or not math.isfinite(lease) or lease <= 0:
            raise ValueError('续租时间和租期不合法')
        with self.store.transaction() as connection:
            changed=connection.execute("""UPDATE jobs SET lease_until=?
                WHERE id=? AND token=? AND status='running' AND lease_until>?""",
                (now+lease,task_id,token,now)).rowcount
            if not changed:
                raise ValueError('不能续租失效的领取')

    def get(self, task_id):
        with self.store.connection() as connection:
            row=connection.execute('SELECT * FROM jobs WHERE id=?',(task_id,)).fetchone()
        if row is None:
            raise ValueError('没有队列任务')
        return dict(row)


def communication_cost():
    # 精确计量固定委派消息的序列化字节数，不实际启动子智能体。
    task = {'goal':'分别查找切片和检查点资料', 'context':'公共资料与引用规则'}
    delegation = [{'worker':'retrieval','task':task}, {'worker':'review','task':task}]
    results = [{'worker':'retrieval','evidence':['numpy:0']},
               {'worker':'review','evidence':['runtime:0']}]
    return {'single_task_bytes':len(encode(task).encode()),
            'delegation_bytes':sum(len(encode(x).encode()) for x in delegation),
            'result_bytes':sum(len(encode(x).encode()) for x in results),
            'messages':4}
