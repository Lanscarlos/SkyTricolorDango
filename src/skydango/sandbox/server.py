"""沙盒子进程的 JSON 接口（brain-sandbox spec §3）：只监听 127.0.0.1，没有页面（页面在管理面板里）。

| 路由 | 内容 |
|---|---|
| GET /status | {"ok", "kind": "sandbox", "ready"}：身体没建好也回（管理面板据此判断子进程起来了） |
| GET /state?after=N[&wait=秒] | 长轮询（最多 25 秒）：沙盒时间、精力、身边、场景、聊天记录新行、idle、额度 |
| POST /op | 冒充发言、来去、快进……（SandboxControl.op）；参数不对 400 |
| GET /brain?after= | 大脑时间线（和 viewer 同一个处理函数） |
| GET /inner、POST /inner/forget | 内心页（和 viewer 同一个处理函数） |
| POST /shutdown | 下线（写日记）：走 watchdog.once 的只中断一次钩子 |

GET / POST 都校验 Host（防 DNS 重绑定），POST 走 post_guard（X-Skydango: 1、JSON、≤ 64 KB）。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..vision.viewer import JsonHandler, _Server, brain_body, forget_response, inner_response, post_guard

log = logging.getLogger(__name__)

MAX_BODY = 65536
STATE_WAIT = 25.0  # /state 长轮询最多等几秒
NOT_READY = {"ok": False, "text": "沙盒还在启动"}


class SandboxServer:
    def __init__(self, port: int, on_shutdown: Callable[[], None], host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = port
        self.on_shutdown = on_shutdown
        self.control = None  # sandbox.control.SandboxControl：身体建好后挂上
        self.trace = None  # brain.trace.BrainTrace
        self.inner: Callable[[], dict] | None = None
        self.forget: Callable[[str, str, str, str], str] | None = None
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> str:
        """起服务（后台线程），返回地址；端口被占用抛 OSError。"""
        box = self

        class Handler(JsonHandler):
            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if not self._local_host():
                    self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                    return
                if url.path == "/status":
                    self._json(200, {"ok": True, "kind": "sandbox", "ready": box.control is not None})
                elif url.path == "/state":
                    control = box.control
                    if control is None:
                        self._json(503, NOT_READY)
                        return
                    self._json(200, control.state(self._after(url), self._wait(url, STATE_WAIT)))
                elif url.path == "/brain":
                    if box.trace is None:
                        self._json(503, NOT_READY)
                        return
                    self._send(200, "application/json; charset=utf-8", brain_body(box.trace, self._after(url), self._wait(url)))
                elif url.path == "/inner":
                    if box.inner is None:
                        self._json(503, NOT_READY)
                        return
                    self._json(*inner_response(box.inner))
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path not in ("/op", "/inner/forget", "/shutdown"):
                    self._drain()
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                refused = post_guard(self.headers, self.server.server_address[1], MAX_BODY)
                if refused is not None:
                    self._drain()
                    self._json(refused[0], {"ok": False, "text": refused[1]})
                    return
                raw = self.rfile.read(int(self.headers.get("Content-Length")))
                if path == "/shutdown":
                    box.on_shutdown()  # interrupt_main：主线程走 Ctrl+C 的收尾，这里照样能回应
                    self._json(200, {"ok": True, "text": "正在下线（写日记）"})
                elif path == "/inner/forget":
                    if box.forget is None:
                        self._json(503, NOT_READY)
                    else:
                        self._json(*forget_response(box.forget, raw))
                else:
                    self._op(raw)

            def _op(self, raw: bytes) -> None:
                control = box.control
                if control is None:
                    self._json(503, NOT_READY)
                    return
                try:
                    req = json.loads(raw.decode("utf-8"))
                    result = control.op(req)
                except ValueError as exc:  # 坏 JSON、未知操作、参数不对
                    self._json(400, {"ok": False, "text": str(exc)})
                    return
                except Exception as exc:  # 身体停了之类
                    log.exception("沙盒操作出错")
                    self._json(503, {"ok": False, "text": f"沙盒没做成：{exc}"})
                    return
                self._json(200, result)

            def _wait(self, url, cap: float = 2.0) -> float:
                try:
                    value = float(parse_qs(url.query).get("wait", [cap])[0])
                except ValueError:
                    return cap
                return max(0.0, min(cap, value)) if value == value else cap

        self._server = _Server((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="sandbox-http", daemon=True).start()
        return self.url

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2] if self._server else (self.host, self.port)
        return f"http://{host}:{port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
