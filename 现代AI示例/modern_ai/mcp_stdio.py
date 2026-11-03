"""固定 2025-11-25 版本的最小 MCP stdio 服务与逐轮客户端。"""
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from .agent import TOOLS, execute, strict_json

VERSION = '2025-11-25'


def rpc_error(identifier, code, message):
    return {'jsonrpc':'2.0','id':identifier,'error':{'code':code,'message':message}}


def handle(request, state):
    # 请求结构、会话阶段、工具业务校验分开；通知一律不返回响应。
    if type(request) is not dict or request.get('jsonrpc') != '2.0' or type(request.get('method')) is not str:
        return rpc_error(None, -32600, 'Invalid Request')
    identifier = request.get('id')
    if 'id' in request and type(identifier) not in (str, int):
        return rpc_error(None, -32600, 'Invalid Request id')
    method = request['method']
    params = request.get('params', {})
    if 'id' not in request:
        if method == 'notifications/initialized' and state['ready'] and type(params) is dict:
            state['initialized'] = True
        return None
    if type(params) is not dict:
        return rpc_error(identifier, -32602, 'params 必须为对象')
    if method == 'ping':
        result = {}
    elif method == 'initialize':
        info = params.get('clientInfo')
        if (state['ready'] or type(params.get('protocolVersion')) is not str
                or type(params.get('capabilities')) is not dict or type(info) is not dict
                or type(info.get('name')) is not str or type(info.get('version')) is not str):
            return rpc_error(identifier, -32602, 'initialize 参数或阶段错误')
        # 若对方请求别的版本，返回本服务支持的版本，由客户端决定继续或断开。
        state['ready'] = True
        result = {'protocolVersion':VERSION, 'capabilities':{'tools':{}},
                  'serverInfo':{'name':'reading-notes','version':'1.0.0'}}
    elif not state['initialized']:
        return rpc_error(identifier, -32000, '尚未完成初始化')
    elif method == 'tools/list':
        result = {'tools':TOOLS}
    elif method == 'tools/call':
        name, arguments = params.get('name'), params.get('arguments', {})
        if type(name) is not str or type(arguments) is not dict:
            return rpc_error(identifier, -32602, 'tools/call 结构错误')
        if name not in {tool['name'] for tool in TOOLS}:
            return rpc_error(identifier, -32602, 'Unknown tool: '+name)
        try:
            value = execute(name, arguments)
            result = {'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],
                      'isError':False}
        except (ValueError, TypeError) as error:
            result = {'content':[{'type':'text','text':str(error)}], 'isError':True}
    else:
        return rpc_error(identifier, -32601, 'Method not found')
    return {'jsonrpc':'2.0','id':identifier,'result':result}


def serve():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    state = {'ready':False,'initialized':False}
    for line in sys.stdin:
        try:
            response = handle(strict_json(line), state)
        except (ValueError, TypeError) as error:
            code = -32700 if isinstance(error,json.JSONDecodeError) else -32600
            response = rpc_error(None, code, str(error))
        if response is not None:
            print(json.dumps(response,ensure_ascii=False), flush=True)


class StdioClient:
    """单连接、单个在途请求；后台只负责读取，主线程逐轮检查响应。"""
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable,'-B','-m','modern_ai.mcp_stdio','serve'],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,encoding='utf-8')
        self.incoming = queue.Queue()
        self.errors = []
        self.requests, self.responses = [], []
        def read_output():
            for line in self.process.stdout:
                self.incoming.put(line)
            self.incoming.put(None)
        def read_errors():
            self.errors.extend(self.process.stderr)
        self.readers = [threading.Thread(target=read_output,daemon=True),
                        threading.Thread(target=read_errors,daemon=True)]
        for reader in self.readers:
            reader.start()

    def send(self, message):
        self.requests.append(message)
        self.process.stdin.write(json.dumps(message,ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        if 'id' not in message:
            return None
        try:
            line = self.incoming.get(timeout=20)
        except queue.Empty as error:
            raise TimeoutError('等待 MCP 响应超时') from error
        if line is None:
            raise RuntimeError('服务端已关闭输出')
        response = strict_json(line)
        if response.get('jsonrpc') != '2.0' or response.get('id') != message['id']:
            raise ValueError('响应编号或版本不匹配')
        if ('result' in response) == ('error' in response):
            raise ValueError('响应必须且只能包含 result 或 error')
        self.responses.append(response)
        return response

    def initialize(self, version=VERSION):
        response = self.send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{
            'protocolVersion':version,'capabilities':{},
            'clientInfo':{'name':'notes-client','version':'1.0.0'}}})
        result = response.get('result', {})
        if result.get('protocolVersion') != version or 'tools' not in result.get('capabilities',{}):
            raise ValueError('协议版本或工具能力不匹配')
        # 检查通过后才发 initialized，不能预先整批发送。
        self.send({'jsonrpc':'2.0','method':'notifications/initialized'})

    def close(self):
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            for reader in self.readers:
                reader.join(timeout=5)
            self.process.stdout.close()
            self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def client_demo():
    with StdioClient() as client:
        client.initialize()
        client.send({'jsonrpc':'2.0','id':2,'method':'tools/list'})
        client.send({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{
            'name':'reading_minutes','arguments':{'pages':3,'questions':4}}})
        return {'requests':client.requests,'responses':client.responses}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv)>1 and sys.argv[1] == 'serve':
        serve()
    else:
        print(json.dumps(client_demo(),ensure_ascii=False,indent=2))
