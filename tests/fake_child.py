"""管理面板测试用的假团子：打运行目录、开 /status、按模式响应 /shutdown 或崩溃。"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--mode", default="normal", choices=["normal", "ignore", "crash", "gbk", "late"])
args = parser.parse_args()

print("本次运行的日志和截图: /tmp/runs/x", flush=True)
if args.mode == "crash":
    print("boom", flush=True)
    sys.exit(3)
if args.mode == "late":  # 启动慢：过一会儿才开端口
    import time

    time.sleep(1.5)
if args.mode == "gbk":
    sys.stdout.buffer.write("中文乱码\n".encode("gbk"))
    sys.stdout.buffer.write(b"x" * 5000 + b"\n")
    sys.stdout.flush()


class Handler(BaseHTTPRequestHandler):
    def _reply(self, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        self._reply({"seq": 0, "age": None, "info": {}})

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply({"ok": True})
        if args.mode in ("normal", "late"):
            print("收到退出", flush=True)
            os._exit(0)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
