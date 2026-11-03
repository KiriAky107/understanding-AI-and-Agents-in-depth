"""同一普通函数的 HTTP 接入，以及真正的本地小模型 HTTP 生成。"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
import torch
from .agent import execute, replay_model, strict_json
from .language import train_language


def post_json(base, path, body):
    # 请求与响应都明确 UTF-8；timeout 限制等待，不自动重复有副作用请求。
    request = Request(base+path, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type':'application/json'}, method='POST')
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read(1048576))


def create_server(port=8765, model=None):
    # 只监听本机。模型在服务器启动时训练一次，此后复用固定参数。
    if model is None:
        model, _ = train_language(return_model=True)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # 演示不把请求正文写进日志，调用方显式保存需要的追踪。
            pass

        def do_POST(self):
            status = 200
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('请求长度越界')
                body = strict_json(self.rfile.read(length).decode('utf-8'))
                if type(body) is not dict:
                    raise ValueError('正文必须为 JSON 对象')
                if self.path == '/generate':
                    prompt, count = body['prompt'], body.get('max_new_tokens', 12)
                    if type(prompt) is not str or not prompt or type(count) is not int or not 1 <= count <= 64:
                        raise ValueError('生成参数不合法')
                    tokens = torch.tensor([list(prompt.encode())], dtype=torch.long)
                    with torch.no_grad():
                        for _ in range(count):
                            next_id = model(tokens[:, -model.context:])[:, -1].argmax(-1, keepdim=True)
                            tokens = torch.cat((tokens, next_id), dim=1)
                    result = {'text':bytes(tokens[0].tolist()).decode('utf-8', errors='replace'),
                              'input_tokens':len(prompt.encode()), 'output_tokens':count}
                elif self.path == '/chat':
                    messages = body.get('messages')
                    if (type(messages) is not list or not messages
                            or any(type(m) is not dict or m.get('role') not in
                                   ['system','user','assistant','tool'] or not
                                   (type(m.get('content')) is str or
                                    (m.get('role') == 'assistant' and type(m.get('tool_call')) is dict))
                                   for m in messages)
                            or type(messages[-1].get('content')) is not str):
                        raise ValueError('messages 必须为非空消息列表')
                    result = replay_model(body['messages'])
                elif self.path == '/tools/reading_minutes':
                    result = execute('reading_minutes', body)
                else:
                    status, result = 404, {'error':'不存在的路径'}
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
                status, result = 400, {'error':str(error)}
            data = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['serve','client'])
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.action == 'client':
        print(json.dumps(post_json(f'http://127.0.0.1:{args.port}', '/generate',
                                   {'prompt':'read ', 'max_new_tokens':12}), ensure_ascii=False))
        return
    server = create_server(args.port)
    print(f'Listening http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
