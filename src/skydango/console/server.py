"""管理面板的 HTTP 服务：页面、设置、设备检测、启停团子，/live/* 转发到团子子进程的 viewer。

只监听 127.0.0.1。除了页面本身，所有请求都校验 Host（防 DNS 重绑定）；POST 还要 X-Skydango 头 + JSON + 大小上限
（别的网页借浏览器发不过来）。防护函数和 vision/viewer.py 共用。
"""

from __future__ import annotations

import dataclasses
import http.client
import importlib.resources
import importlib.util
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config import Config, console_paths, read_secrets
from ..vision.viewer import is_local_host, post_guard
from . import probes
from .devicecheck import run_checks
from .inner_view import BUSY_ERROR, forget_offline, inner_state
from .preflight import preflight
from .runner import LOCAL, LaunchOptions, Runner, build_command, child_env, probe_status, send_shutdown
from .settings import SettingsStore

log = logging.getLogger(__name__)

MAX_BODY = 65536  # 设置可能比手动控制大
PROXY_TIMEOUT = 5.0  # 长轮询 2 秒 + 余量
BUSY = ("starting", "running", "stopping")
LIVE = ("running", "stopping")
NOT_RUNNING = {"ok": False, "text": "团子没在运行"}


class _Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"  # Windows 上 SO_REUSEADDR 允许两个进程绑同一个端口
    daemon_threads = True


def _page() -> bytes:
    return (importlib.resources.files("skydango.console") / "static" / "console.html").read_bytes()


def _launch(body: dict) -> LaunchOptions:
    """校验总览传来的启动选项；不对抛 ValueError（原因给页面看）。"""
    for key in ("brain", "live", "emotes"):
        if not isinstance(body.get(key), bool):
            raise ValueError(f"{key} 要是 true / false")
    duration = body.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        raise ValueError("duration 要是不小于 0 的数字")
    return LaunchOptions(body["brain"], body["live"], body["emotes"], float(duration))


class ConsoleServer:
    def __init__(
        self,
        config_path: Path,
        store: SettingsStore,
        runner: Runner,
        port: int,
        child_port: int,
        device_checks: Callable = run_checks,
        make_device: Callable[[Config], Any] | None = None,
        parent_pid: int | None = None,
        find_spec: Callable[[str], object] = importlib.util.find_spec,
    ) -> None:
        self.config_path, self.store, self.runner = Path(config_path), store, runner
        self.port, self.child_port = port, child_port
        self.device_checks, self.make_device, self.find_spec = device_checks, make_device, find_spec
        self.parent_pid = os.getpid() if parent_pid is None else parent_pid
        self.orphan = probe_status(child_port)  # 上次留下的团子还占着端口
        self.proxy_timeout = PROXY_TIMEOUT
        self._device_lock = threading.Lock()  # 检测设备期间不让启动（两边会同时碰设备）
        self._checking = False
        self._server: ThreadingHTTPServer | None = None

    # ---- 各个接口 ----
    def _busy(self) -> bool:
        return self.runner.status()["state"] in BUSY

    def launch_options(self) -> LaunchOptions:
        c = self.store._fallback().console
        return LaunchOptions(c.brain, c.live, c.emotes, c.duration)

    def state(self) -> dict:
        opts = self.launch_options()
        busy = self._busy()
        return {"run": self.runner.status(), "launch": dataclasses.asdict(opts),
                "problems": preflight(self.store, opts, busy, self.find_spec), "orphan": self.orphan and not busy,
                "emotes_allowed": self.store._fallback().emotes.enabled}  # config.toml 关了动作：面板上只能关不能开

    def start_run(self, body: dict) -> tuple[int, dict]:
        try:
            opts = _launch(body)
        except ValueError as exc:
            return 400, {"ok": False, "text": str(exc)}
        self.store.save({"console.brain": opts.brain, "console.live": opts.live, "console.emotes": opts.emotes,
                         "console.duration": opts.duration})  # 下次打开面板还是这次的选择
        with self._device_lock:  # 和设备检测互斥：检查完到真的起进程之间，检测不能插进来
            problems = preflight(self.store, opts, self._busy(), self.find_spec)
            if self._checking:
                problems.append("正在检测设备，等检测完再叫醒")
            if not self._busy() and probe_status(self.child_port):  # 上次留下的团子还占着端口：再起一个会有两个团子
                self.orphan = True
                problems.append(f"{self.child_port} 端口上有上次留下的团子，先点「让它退出」")
            try:
                secrets = read_secrets(console_paths(self.config_path)[1])
            except ValueError as exc:
                problems.append(str(exc))
            if problems:
                return 409, {"ok": False, "problems": problems}
            cmd = build_command(opts, self.config_path, self.child_port, self.parent_pid)
            try:
                self.runner.start(cmd, child_env(os.environ, secrets), opts)
            except (RuntimeError, OSError) as exc:
                return 409, {"ok": False, "problems": [str(exc)]}
        self.orphan = False
        return 200, {"ok": True}

    def save_settings(self, body: dict) -> tuple[int, dict]:
        values, revert = body.get("values") or {}, body.get("revert") or []
        if not isinstance(values, dict) or not isinstance(revert, list):
            return 400, {"ok": False, "text": "values 要是对象、revert 要是列表"}
        result = self.store.save(values, revert)
        if result["ok"] and self._busy():
            result["restart"] = True
        return 200, result

    def test_settings(self, body: dict) -> tuple[int, dict]:
        values = body.get("values") or {}
        what = body.get("what")
        if what not in ("llm", "claude") or not isinstance(values, dict):
            return 400, {"ok": False, "text": "what 要是 llm 或 claude"}
        try:
            cfg = self.store.effective()
        except ValueError as exc:
            return 200, {"ok": False, "text": str(exc)}
        form = {k: v for k, v in values.items() if isinstance(v, str)}
        if what == "llm":
            llm = dataclasses.replace(cfg.llm, **{k[4:]: form[k] for k in ("llm.provider", "llm.base_url", "llm.model") if k in form})
            return 200, probes.test_llm(llm, form.get("secret.llm") or self.store.secret("llm"))
        path = form.get("brain.claude_path") or cfg.brain.claude_path
        return 200, probes.test_claude(path, cfg.brain.config_dir, form.get("secret.claude") or self.store.secret("claude"))

    def check_device(self) -> tuple[int, dict]:
        with self._device_lock:
            if self._busy():
                return 409, {"ok": False, "text": "团子运行中，设备归它用"}
            if self._checking:
                return 409, {"ok": False, "text": "正在检测设备"}
            self._checking = True
        try:
            try:
                cfg = self.store.effective()
            except ValueError as exc:
                return 200, {"ok": False, "text": str(exc)}
            make = self.make_device
            if make is None:
                from ..cli import _device as make  # 延迟导入：cli 也导入这个模块
            checks = self.device_checks(cfg, make)
            return 200, {"ok": True, "checks": [dataclasses.asdict(c) for c in checks]}
        finally:
            self._checking = False

    def stop_orphan(self) -> tuple[int, dict]:
        ok = send_shutdown(self.child_port)
        self.orphan = False
        return 200, {"ok": ok}

    def proxy(self, method: str, rest: str, query: str, body: bytes | None = None) -> tuple[int, str, bytes]:
        """转发到子进程的 viewer；没在运行 / 连不上 / 超时都是 503（子进程可能刚好退出）。"""
        not_running = (503, "application/json; charset=utf-8", json.dumps(NOT_RUNNING, ensure_ascii=False).encode())
        if self.runner.status()["state"] not in LIVE:
            return not_running
        url = f"http://127.0.0.1:{self.child_port}/{rest}" + (f"?{query}" if query else "")
        headers = {"Content-Type": "application/json", "X-Skydango": "1"} if method == "POST" else {}
        req = urllib.request.Request(url, data=body, headers=headers, method=method)  # Host 自动是 127.0.0.1:<child_port>
        try:
            with LOCAL.open(req, timeout=self.proxy_timeout) as r:
                return r.status, r.headers.get("Content-Type", "application/octet-stream"), r.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers.get("Content-Type", "text/plain"), err.read()
        except (urllib.error.URLError, OSError, http.client.HTTPException):  # 连不上、超时、响应体传到一半断了
            return not_running

    # ---- 内心页（spec 2026-09-30-inner-viewer §2）----
    def _inner_dir_and_friends(self) -> tuple[Path, list[str]]:
        from ..chat.memory import MemoryStore

        cfg = self.store._fallback()  # console.toml 坏了也能看
        found = MemoryStore(cfg.reply.memory_dir).friend_names() if cfg.reply.memory_dir else []
        return Path(cfg.reply.memory_dir) / "inner", list(dict.fromkeys([*found, *cfg.reply.friends]))

    def _live_inner(self) -> dict | None:
        code, _, raw = self.proxy("GET", "inner", "")
        if code != 200:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def inner(self) -> tuple[int, dict]:
        inner_dir, friends = self._inner_dir_and_friends()
        return 200, inner_state(inner_dir, friends, self.runner.status()["state"], self._live_inner, time.time())

    def forget(self, body: dict) -> tuple[int, dict]:
        state = self.runner.status()["state"]
        if state in ("starting", "stopping"):  # 两边可能同时改 persona.json
            return 409, {"ok": False, "error": BUSY_ERROR}
        inner_dir, _ = self._inner_dir_and_friends()
        alive = self.store._fallback().inner.save_every * 3
        if state == "running":
            code, _, raw = self.proxy("POST", "inner/forget", "", json.dumps(body, ensure_ascii=False).encode())
            try:
                data = json.loads(raw.decode("utf-8"))
            except ValueError:
                data = None
            if not isinstance(data, dict):
                return 503, {"ok": False, "error": "团子没回应，稍后再试"}
            if not data.get("ok") and "error" not in data:
                data["error"] = data.get("text") or "团子没删成"
            options = self.runner.status().get("options") or {}
            if data.get("ok") and options.get("live") is False:
                # dry-run 的团子只删了内存里的（它从不写 persona.json）：面板顺手把文件也改了，停掉之后不会回来
                try:
                    forget_offline(inner_dir, body, time.time(), alive)
                except Exception:
                    log.exception("dry-run 时顺手改 persona.json 出错")
            return code, data
        if probe_status(self.child_port):  # 上次留下的团子还占着端口：它会把性格档案写回去
            return 409, {"ok": False, "error": "上次留下的团子还在跑（占着子进程端口），先在总览让它退出再删"}
        try:
            return 200, forget_offline(inner_dir, body, time.time(), alive)
        except ValueError as exc:
            return 400, {"ok": False, "error": str(exc)}

    # ---- HTTP ----
    def start(self) -> str:
        """起服务（后台线程），返回地址。端口被占用抛 OSError。"""
        console = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                if url.path == "/":
                    self._send(200, "text/html; charset=utf-8", _page())
                    return
                if not is_local_host(self.headers.get("Host") or "", console.port):
                    self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                elif url.path == "/api/state":
                    self._json(200, console.state())
                elif url.path == "/api/logs":
                    self._json(200, console.runner.logs(self._after(url)))
                elif url.path == "/api/settings":
                    self._json(200, console.store.view())
                elif url.path == "/api/inner":
                    self._json(*console.inner())
                elif url.path.startswith("/live/"):
                    self._send(*console.proxy("GET", url.path[len("/live/"):], url.query))
                else:
                    self._json(404, {"ok": False, "text": "没有这个地址"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                routes = {
                    "/api/settings": console.save_settings,
                    "/api/settings/test": console.test_settings,
                    "/api/device": lambda body: console.check_device(),
                    "/api/run/start": console.start_run,
                    "/api/run/stop": lambda body: (console.runner.stop(), (200, {"ok": True}))[1],
                    "/api/orphan/stop": lambda body: console.stop_orphan(),
                    "/api/inner/forget": console.forget,
                }
                if path not in routes and path != "/live/control":
                    self._drain()
                    self._json(404, {"ok": False, "text": "没有这个地址"})
                    return
                refused = post_guard(self.headers, console.port, MAX_BODY)
                if refused is not None:
                    self._drain()
                    self._json(refused[0], {"ok": False, "text": refused[1]})
                    return
                raw = self.rfile.read(int(self.headers.get("Content-Length")))
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = None
                if not isinstance(body, dict):
                    self._json(400, {"ok": False, "text": "请求要是一个 JSON 对象"})
                    return
                if path == "/live/control":
                    self._send(*console.proxy("POST", "control", "", raw))
                    return
                try:
                    self._json(*routes[path](body))
                except Exception as exc:  # 出错也给页面一句话，不让页面卡住
                    log.exception("面板请求 %s 出错", path)
                    self._json(500, {"ok": False, "text": f"出错了：{exc}"})

            def _after(self, url) -> int:
                try:
                    return int(parse_qs(url.query).get("after", ["0"])[0] or 0)
                except ValueError:
                    return 0

            def _drain(self) -> None:
                try:
                    left = min(int(self.headers.get("Content-Length") or 0), MAX_BODY + 1)
                    if left > 0:
                        self.rfile.read(left)
                except (ValueError, OSError):
                    pass

            def _json(self, code: int, data) -> None:
                self._send(code, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False).encode(errors="replace"))

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-store")
                    if code != 204:
                        self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if code != 204:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args) -> None:
                pass

        self._server = _Server(("127.0.0.1", self.port), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, name="console", daemon=True).start()
        return f"http://127.0.0.1:{self.port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
