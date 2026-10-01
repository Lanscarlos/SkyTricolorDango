"""管理面板的 HTTP 服务：页面、设置、设备检测、启停团子 / 大脑沙盒，/live/* 转发到团子子进程的 viewer，
/sandbox/* 转发到沙盒子进程的接口（团子和沙盒同一时间只能有一个）。

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
import re
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
from ..vision.viewer import is_local_host, post_guard, static_asset
from . import probes
from .devicecheck import run_checks
from .inner_view import BUSY_ERROR, forget_offline, inner_state
from .preflight import preflight, problem
from .reports import list_reports, read_report
from .replay import KEEP_NOTE, Recorder, Replayer, safe_name
from .runner import LOCAL, LaunchOptions, Runner, build_command, build_sandbox_command, child_env, probe_status, send_shutdown
from .settings import SettingsStore

log = logging.getLogger(__name__)

MAX_BODY = 65536  # 设置可能比手动控制大
PROXY_TIMEOUT = 5.0  # 长轮询 2 秒 + 余量
BUSY = ("starting", "running", "stopping")
LIVE = ("running", "stopping")
NOT_RUNNING = {"ok": False, "text": "团子没在运行"}
SANDBOX_NOT_RUNNING = {"ok": False, "text": "沙盒没在运行"}
SANDBOX_PROXY_TIMEOUT = 35.0  # /sandbox/state 长轮询最多 25 秒、/sandbox/op 身体线程最多 10 秒 + 余量
SANDBOX_GET = ("state", "brain", "inner", "status")  # 转发给沙盒的 GET
_STATIC_NAME = re.compile(r"[A-Za-z0-9_.-]+\.(css|js)")
_STATIC_TYPES = {"css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8"}


def _console_static(name: str) -> tuple[str, bytes] | None:
    """`console/static/` 下的 css / js；名字不合规矩或不存在返回 None。"""
    m = _STATIC_NAME.fullmatch(name)
    if m is None or ".." in name:
        return None
    try:
        return _STATIC_TYPES[m.group(1)], (importlib.resources.files("skydango.console") / "static" / name).read_bytes()
    except (OSError, FileNotFoundError):
        return None


SANDBOX_POST = ("op", "inner/forget")  # 转发给沙盒的 POST（/shutdown 不给页面直接发，走 /api/sandbox/stop）


def _kind(status: dict) -> str:
    return status.get("kind") or "dango"


def _start_choice(body: dict) -> str:
    """沙盒启动选项：resume / sleep / HH:MM / YYYY-MM-DD HH:MM；不对抛 ValueError。"""
    from ..sandbox.clock import is_time_text

    start = body.get("start", "resume")
    if not isinstance(start, str) or not (start in ("resume", "sleep") or is_time_text(start)):
        raise ValueError("起始时间要是 resume（接着上次）/ sleep（睡一晚）/ HH:MM / YYYY-MM-DD HH:MM")
    return start.strip()


class _Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"  # Windows 上 SO_REUSEADDR 允许两个进程绑同一个端口
    daemon_threads = True


def _page() -> bytes:
    return (importlib.resources.files("skydango.console") / "static" / "console.html").read_bytes()


def _launch(body: dict) -> LaunchOptions:
    """校验真机团子页传来的启动选项；不对抛 ValueError（原因给页面看）。"""
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
        # 沙盒剧本（spec 2026-09-30-brain-sandbox §6）：录制和回放都在面板进程里
        self.recorder = Recorder()
        self.replayer: Replayer | None = None
        self.last_report: str | None = None
        self._replay_lock = threading.Lock()

    # ---- 各个接口 ----
    def _busy(self) -> bool:
        return self.runner.status()["state"] in BUSY

    def _state_of(self, kind: str) -> str:
        """某一种子进程（dango / sandbox）现在的状态；槽被另一种占着时当它 idle。"""
        st = self.runner.status()
        return st["state"] if _kind(st) == kind else "idle"

    def _other_busy(self, kind: str) -> str | None:
        """另一种子进程在 starting / running / stopping 时给一句"先停它"。"""
        st = self.runner.status()
        if st["state"] in BUSY and _kind(st) != kind:
            return "沙盒在运行，先在「沙盒」页下线沙盒" if _kind(st) == "sandbox" else "团子在运行，先停团子"
        return None

    def launch_options(self) -> LaunchOptions:
        c = self.store._fallback().console
        return LaunchOptions(c.brain, c.live, c.emotes, c.duration)

    def _run_problems(self, opts: LaunchOptions) -> list[dict]:
        """叫醒团子前的问题：沙盒占着槽时说"先下线沙盒"，不说"团子已经在运行"。"""
        other = self._other_busy("dango")
        if other:
            return [problem(other), *preflight(self.store, opts, False, self.find_spec)]
        return preflight(self.store, opts, self._busy(), self.find_spec)

    def state(self) -> dict:
        opts = self.launch_options()
        busy = self._busy()
        return {"run": self.runner.status(), "launch": dataclasses.asdict(opts),
                "problems": self._run_problems(opts), "orphan": self.orphan and not busy,
                "emotes_allowed": self.store._fallback().emotes.enabled}  # config.toml 关了动作：面板上只能关不能开

    def start_run(self, body: dict) -> tuple[int, dict]:
        try:
            opts = _launch(body)
        except ValueError as exc:
            return 400, {"ok": False, "text": str(exc)}
        self.store.save({"console.brain": opts.brain, "console.live": opts.live, "console.emotes": opts.emotes,
                         "console.duration": opts.duration})  # 下次打开面板还是这次的选择
        with self._device_lock:  # 和设备检测互斥：检查完到真的起进程之间，检测不能插进来
            problems = self._run_problems(opts)
            if self._checking:
                problems.append(problem("正在检测设备，等检测完再叫醒"))
            if not self._busy() and probe_status(self.child_port):  # 上次留下的团子还占着端口：再起一个会有两个团子
                self.orphan = True
                problems.append(problem(f"{self.child_port} 端口上有上次留下的团子，先点「让它退出」"))
            if not self._busy() and probe_status(self.sandbox_port()):  # 上次留下的沙盒：共用令牌，也别同时在线
                problems.append(problem(f"{self.sandbox_port()} 端口上有上次留下的沙盒，先在「沙盒」页让它退出"))
            try:
                secrets = read_secrets(console_paths(self.config_path)[1])
            except ValueError as exc:
                problems.append(problem(str(exc)))
            if problems:
                return 409, {"ok": False, "problems": problems}
            cmd = build_command(opts, self.config_path, self.child_port, self.parent_pid)
            try:
                self.runner.start(cmd, child_env(os.environ, secrets), opts)
            except (RuntimeError, OSError) as exc:
                return 409, {"ok": False, "problems": [problem(str(exc))]}
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

    def stop_orphan(self, body: dict | None = None) -> tuple[int, dict]:
        """让上次留下的子进程退出；body {"kind": "sandbox"} 时是沙盒端口。"""
        if (body or {}).get("kind") == "sandbox":
            return 200, {"ok": send_shutdown(self.sandbox_port())}
        ok = send_shutdown(self.child_port)
        self.orphan = False
        return 200, {"ok": ok}

    def stop_run(self) -> tuple[int, dict]:
        """真机团子的「停止」：只停团子（沙盒在跑时不动它）。"""
        if _kind(self.runner.status()) == "dango":
            self.runner.stop()
        return 200, {"ok": True}

    # ---- 大脑沙盒（spec 2026-09-30-brain-sandbox §4）----
    def sandbox_port(self) -> int:
        return self.store._fallback().sandbox.port

    def sandbox_dir(self) -> Path:
        return Path(self.store._fallback().sandbox.dir)

    def replaying(self) -> bool:
        r = self.replayer
        return r is not None and bool(r.progress().get("running"))

    def start_sandbox(self, body: dict) -> tuple[int, dict]:
        try:
            start = _start_choice(body)
        except ValueError as exc:
            return 400, {"ok": False, "text": str(exc)}
        if self.replaying():
            return 409, {"ok": False, "problems": [problem("正在回放剧本，先停止回放")]}
        code, res = self.launch_sandbox(start)
        if code == 200:
            self.recorder.started(start)  # 第一次启动是剧本的 [start]，之后记成 online
        return code, res

    def sandbox_problems(self) -> tuple[list[dict], bool]:
        """起沙盒前的问题 + 沙盒端口上有没有孤儿（不加锁，要一致的调用方自己拿 _device_lock）。"""
        port = self.sandbox_port()
        problems: list[dict] = []
        other = self._other_busy("sandbox")
        if other:
            problems.append(problem(other))
        elif self._busy():
            problems.append(problem("沙盒已经在运行"))
        # 沙盒不碰设备：预检只看配置、令牌、mcp（和团子的大脑模式一样）
        problems += preflight(self.store, LaunchOptions(brain=True), False, self.find_spec)
        orphan = False
        if not self._busy() and probe_status(port):
            orphan = True
            problems.append(problem(f"{port} 端口上有上次留下的沙盒，先点「让它退出」"))
        if not self._busy() and probe_status(self.child_port):  # 上次留下的团子：共用令牌，别两个大脑同时在线
            self.orphan = True
            problems.append(problem(f"{self.child_port} 端口上有上次留下的团子，先在「真机团子」页让它退出"))
        try:
            read_secrets(console_paths(self.config_path)[1])
        except ValueError as exc:
            problems.append(problem(str(exc)))
        return problems, orphan

    def launch_sandbox(self, start: str) -> tuple[int, dict]:
        """起沙盒子进程（页面启动和回放共用；只有页面启动才记进录制）。"""
        port = self.sandbox_port()
        with self._device_lock:
            problems, orphan = self.sandbox_problems()
            if problems:
                return 409, {"ok": False, "problems": problems, "orphan": orphan}
            secrets = read_secrets(console_paths(self.config_path)[1])  # 坏了的话 sandbox_problems 已经拦下
            cmd = build_sandbox_command(self.config_path, port, self.parent_pid, start)
            try:
                self.runner.start(cmd, child_env(os.environ, secrets), {"start": start}, kind="sandbox", port=port)
            except (RuntimeError, OSError) as exc:
                return 409, {"ok": False, "problems": [problem(str(exc))]}
        return 200, {"ok": True}

    def stop_sandbox(self) -> tuple[int, dict]:
        if _kind(self.runner.status()) != "sandbox":
            return 409, {"ok": False, "text": "沙盒没在运行"}
        if self.replaying():  # 回放中手动下线：回放这一步做完就停
            self.replayer.stop_requested()
        elif self._state_of("sandbox") in ("starting", "running"):
            self.recorder.stopped()
        self.runner.stop()
        return 200, {"ok": True}

    def reset_sandbox(self) -> tuple[int, dict]:
        if self.replaying():
            return 409, {"ok": False, "text": "正在回放剧本，先停止回放"}
        code, res = self._reset_sandbox()
        if code == 200:
            self.recorder.new(reset=True)
        return code, res

    def _reset_sandbox(self) -> tuple[int, dict]:
        from .sandbox_view import reset

        with self._device_lock:  # 和起沙盒互斥：锁里再看一次状态，别一边删目录一边起子进程
            if self._state_of("sandbox") in BUSY:
                return 409, {"ok": False, "text": "先下线沙盒再重置记忆"}
            if probe_status(self.sandbox_port()):
                return 409, {"ok": False, "text": "上次留下的沙盒还在跑（占着沙盒端口），先让它退出"}
            cfg = self.store._fallback()
            try:
                reset(self.sandbox_dir(), Path(cfg.reply.memory_dir or "memory"))
            except ValueError as exc:  # 沙盒目录和真 memory 重合
                return 409, {"ok": False, "text": str(exc)}
        return 200, {"ok": True, "text": "沙盒记忆用 memory/ 重置好了"}

    def sandbox_info(self) -> dict:
        from .sandbox_view import friends, start_info

        cfg = self.store._fallback()
        root = self.sandbox_dir()
        state = self._state_of("sandbox")
        problems, orphan = ([], False) if state in BUSY else self.sandbox_problems()
        return {**start_info(root, time.time()), "friends": _safe_list(lambda: friends(root)), "owner_name": cfg.brain.owner_name,
                "dir": str(root), "state": state, "problems": problems, "orphan": orphan}  # orphan：页面在问题旁给「让它退出」

    def sandbox_proxy(self, method: str, rest: str, query: str, body: bytes | None = None) -> tuple[int, str, bytes]:
        """转发到沙盒子进程；槽被团子占着 / 沙盒没在跑 / 连不上都是 503。"""
        not_running = (503, "application/json; charset=utf-8", json.dumps(SANDBOX_NOT_RUNNING, ensure_ascii=False).encode())
        st = self.runner.status()
        if _kind(st) != "sandbox" or st["state"] not in LIVE:
            return not_running
        port = st.get("port") or self.sandbox_port()
        url = f"http://127.0.0.1:{port}/{rest}" + (f"?{query}" if query else "")
        headers = {"Content-Type": "application/json", "X-Skydango": "1"} if method == "POST" else {}
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with LOCAL.open(req, timeout=SANDBOX_PROXY_TIMEOUT) as r:
                return r.status, r.headers.get("Content-Type", "application/octet-stream"), r.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers.get("Content-Type", "text/plain"), err.read()
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            return not_running

    def sandbox_op(self, raw: bytes) -> tuple[int, str, bytes]:
        """页面的手动操作：回放中拒绝；沙盒收下了（200）就记进录制。"""
        if self.replaying():
            body = {"ok": False, "text": "正在回放剧本，先停止回放再手动操作"}
            return 409, "application/json; charset=utf-8", json.dumps(body, ensure_ascii=False).encode()
        code, ctype, out = self.sandbox_proxy("POST", "op", "", raw)
        if code == 200:
            try:
                req = json.loads(raw.decode("utf-8"))
                if isinstance(req, dict):
                    self.recorder.record(req)
            except ValueError:
                pass
        return code, ctype, out

    # ---- 剧本（spec §6）----
    def scenarios_dir(self) -> Path:
        return self.sandbox_dir() / "scenarios"

    def list_scenarios(self) -> dict:
        from .scenario import ScenarioError, load

        out = []
        folder = self.scenarios_dir()
        for path in sorted(folder.glob("*.toml")) if folder.is_dir() else []:
            item = {"name": path.stem, "note": "", "steps": 0, "memory": "", "error": ""}
            try:
                sc = load(path)
                item.update(note=sc.note, steps=len(sc.steps), memory=sc.start.memory)
            except ScenarioError as exc:
                item["error"] = str(exc)
            out.append(item)
        return {"scenarios": out, "dir": str(folder)}

    def save_scenario(self, body: dict) -> tuple[int, dict]:
        import tomllib

        from .scenario import ScenarioError, dumps, parse

        name, note = body.get("name"), body.get("note", "")
        if not safe_name(name):
            return 400, {"ok": False, "text": "剧本名只许中英文、数字和 - _（最多 40 个字）"}
        if not isinstance(note, str):
            return 400, {"ok": False, "text": "note 要是文字"}
        if not self.recorder.describe()["steps"]:
            return 400, {"ok": False, "text": "还没录到操作（启动沙盒、做点什么之后再存）"}
        path = self.scenarios_dir() / f"{name}.toml"
        if path.exists() and body.get("overwrite") is not True:
            return 409, {"ok": False, "exists": True, "text": f"已经有叫「{name}」的剧本了，要覆盖吗？"}
        scenario = self.recorder.snapshot(name, note.strip())
        text = dumps(scenario)
        try:
            parse(tomllib.loads(text), name)  # 存之前自己读一遍：录下来的也得是能回放的
        except (ScenarioError, tomllib.TOMLDecodeError) as exc:
            return 400, {"ok": False, "text": f"录下来的操作存不成剧本：{exc}"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        warning = KEEP_NOTE if scenario.start.memory == "keep" else ""
        return 200, {"ok": True, "path": str(path), "warning": warning}

    def start_replay(self, body: dict) -> tuple[int, dict]:
        from .scenario import ScenarioError, load

        name = body.get("name")
        if not safe_name(name):
            return 400, {"ok": False, "text": "剧本名不对"}
        path = self.scenarios_dir() / f"{name}.toml"
        if not path.is_file():
            return 404, {"ok": False, "text": f"找不到剧本「{name}」"}
        try:
            scenario = load(path)
        except ScenarioError as exc:
            return 400, {"ok": False, "text": str(exc)}
        with self._replay_lock:
            if self.replaying():
                return 409, {"ok": False, "text": "已经在回放了"}
            other = self._other_busy("sandbox")
            if other:
                return 409, {"ok": False, "text": other}
            cfg = self.store._fallback()
            replayer = Replayer(_SandboxApi(self), scenario, self.sandbox_dir() / "reports", cfg.sandbox.step_timeout,
                                cfg.console.stop_timeout)
            self.replayer = replayer
            self.recorder.new()  # 回放的操作不录；回放完接着手动玩的从这里重新录
        threading.Thread(target=self._run_replay, args=(replayer,), name="console-replay", daemon=True).start()
        return 200, {"ok": True}

    def _run_replay(self, replayer: Replayer) -> None:
        try:
            self.last_report = str(replayer.run())
        except Exception:
            log.exception("回放出错")

    def stop_replay(self) -> tuple[int, dict]:
        if not self.replaying():
            return 409, {"ok": False, "text": "没在回放"}
        self.replayer.stop_requested()
        return 200, {"ok": True, "text": "这一步做完就停"}

    def replay_state(self) -> dict:
        r = self.replayer
        progress = r.progress() if r is not None else {"running": False, "step": 0, "total": 0, "name": ""}
        return {"progress": progress, "report": self.last_report, "recording": self.recorder.describe()}

    def proxy(self, method: str, rest: str, query: str, body: bytes | None = None) -> tuple[int, str, bytes]:
        """转发到子进程的 viewer；没在运行 / 连不上 / 超时都是 503（子进程可能刚好退出）。"""
        not_running = (503, "application/json; charset=utf-8", json.dumps(NOT_RUNNING, ensure_ascii=False).encode())
        st = self.runner.status()
        if st["state"] not in LIVE or _kind(st) != "dango":  # 槽被沙盒占着：/live/* 不转
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

    def _sandbox_live_inner(self) -> dict | None:
        code, _, raw = self.sandbox_proxy("GET", "inner", "")
        if code != 200:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _sandbox_now(self) -> float:
        """沙盒没在跑时的"现在"：上次停下时的沙盒时间（可能比真实时间晚），别用真实时间把沙盒的别扭 / 心愿判成过期。"""
        from ..sandbox.clock import floor_time

        return max(time.time(), floor_time(self.sandbox_dir()))

    def inner(self, source: str = "dango") -> tuple[int, dict]:
        if source == "sandbox":
            from .sandbox_view import friends

            root = self.sandbox_dir()
            return 200, inner_state(root / "memory" / "inner", _safe_list(lambda: friends(root)), self._state_of("sandbox"),
                                    self._sandbox_live_inner, self._sandbox_now())
        inner_dir, friends = self._inner_dir_and_friends()
        return 200, inner_state(inner_dir, friends, self._state_of("dango"), self._live_inner, time.time())

    def _forget_sandbox(self, body: dict) -> tuple[int, dict]:
        state = self._state_of("sandbox")
        if state in ("starting", "stopping"):
            return 409, {"ok": False, "error": "沙盒正在启动 / 下线，稍等再删"}
        clean = {k: v for k, v in body.items() if k != "source"}
        if state == "running":
            code, _, raw = self.sandbox_proxy("POST", "inner/forget", "", json.dumps(clean, ensure_ascii=False).encode())
            try:
                data = json.loads(raw.decode("utf-8"))
            except ValueError:
                data = None
            if not isinstance(data, dict):
                return 503, {"ok": False, "error": "沙盒没回应，稍后再试"}
            if not data.get("ok") and "error" not in data:
                data["error"] = data.get("text") or "沙盒没删成"
            return code, data
        if probe_status(self.sandbox_port()):
            return 409, {"ok": False, "error": "上次留下的沙盒还在跑（占着沙盒端口），先让它退出再删"}
        try:
            # "别处在跑"按真实时间比 mtime；流水账这一条记沙盒时间（和沙盒自己记的排在一条线上）
            return 200, forget_offline(self.sandbox_dir() / "memory" / "inner", clean, time.time(),
                                       self.store._fallback().inner.save_every * 3, at=self._sandbox_now())
        except ValueError as exc:
            return 400, {"ok": False, "error": str(exc)}

    def forget(self, body: dict) -> tuple[int, dict]:
        if body.get("source") == "sandbox":
            return self._forget_sandbox(body)
        body = {k: v for k, v in body.items() if k != "source"}
        state = self._state_of("dango")
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
            return 409, {"ok": False, "error": "上次留下的团子还在跑（占着子进程端口），先在「真机团子」页让它退出再删"}
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
                if url.path.startswith("/static/"):  # 和 viewer 共用的大脑时间线脚本 / 样式（同页面一样公开）
                    asset = static_asset(url.path[len("/static/"):])
                    if asset is None:
                        self._json(404, {"ok": False, "text": "没有这个地址"})
                    else:
                        self._send(200, *asset)
                    return
                if url.path.startswith("/console/static/"):  # 面板自己的样式 / 脚本（同页面一样公开）
                    asset = _console_static(url.path[len("/console/static/"):])
                    if asset is None:
                        self._json(404, {"ok": False, "text": "没有这个地址"})
                    else:
                        self._send(200, *asset)
                    return
                if not is_local_host(self.headers.get("Host") or "", console.port):
                    self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                elif url.path == "/api/sandbox/reports":
                    self._json(200, {"reports": list_reports(console.sandbox_dir() / "reports")})
                elif url.path.startswith("/api/sandbox/reports/"):
                    self._json(*read_report(console.sandbox_dir() / "reports", url.path[len("/api/sandbox/reports/"):]))
                elif url.path == "/api/state":
                    self._json(200, console.state())
                elif url.path == "/api/logs":
                    self._json(200, console.runner.logs(self._after(url)))
                elif url.path == "/api/settings":
                    self._json(200, console.store.view())
                elif url.path == "/api/inner":
                    self._json(*console.inner(parse_qs(url.query).get("source", ["dango"])[0]))
                elif url.path == "/api/sandbox/info":
                    self._json(200, console.sandbox_info())
                elif url.path == "/api/sandbox/scenarios":
                    self._json(200, console.list_scenarios())
                elif url.path == "/api/sandbox/replay":
                    self._json(200, console.replay_state())
                elif url.path.startswith("/live/"):
                    self._send(*console.proxy("GET", url.path[len("/live/"):], url.query))
                elif url.path.startswith("/sandbox/"):
                    rest = url.path[len("/sandbox/"):]
                    if rest in SANDBOX_GET:
                        self._send(*console.sandbox_proxy("GET", rest, url.query))
                    else:
                        self._json(404, {"ok": False, "text": "没有这个地址"})
                else:
                    self._json(404, {"ok": False, "text": "没有这个地址"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                routes = {
                    "/api/settings": console.save_settings,
                    "/api/settings/test": console.test_settings,
                    "/api/device": lambda body: console.check_device(),
                    "/api/run/start": console.start_run,
                    "/api/run/stop": lambda body: console.stop_run(),
                    "/api/orphan/stop": console.stop_orphan,
                    "/api/inner/forget": console.forget,
                    "/api/sandbox/start": console.start_sandbox,
                    "/api/sandbox/stop": lambda body: console.stop_sandbox(),
                    "/api/sandbox/reset": lambda body: console.reset_sandbox(),
                    "/api/sandbox/save": console.save_scenario,
                    "/api/sandbox/replay": console.start_replay,
                    "/api/sandbox/replay/stop": lambda body: console.stop_replay(),
                    "/api/sandbox/record/new": lambda body: (console.recorder.new(), (200, {"ok": True}))[1],
                }
                forwards = {"/live/control", *(f"/sandbox/{p}" for p in SANDBOX_POST)}
                if path not in routes and path not in forwards:
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
                if path == "/sandbox/op":
                    self._send(*console.sandbox_op(raw))
                    return
                if path.startswith("/sandbox/"):
                    self._send(*console.sandbox_proxy("POST", path[len("/sandbox/"):], "", raw))
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


def _safe_list(fn) -> list:
    try:
        return list(fn())
    except Exception:
        log.exception("读沙盒好友名单出错")
        return []


class _SandboxApi:
    """回放引擎用的沙盒接口：直接调面板自己的方法（不经 HTTP、不录制）。"""

    def __init__(self, console: ConsoleServer) -> None:
        self.c = console

    def start(self, choice: str) -> None:
        code, res = self.c.launch_sandbox(choice or "resume")
        if code != 200:
            raise RuntimeError("沙盒起不来：" + "；".join([p["text"] for p in res.get("problems") or []] or [res.get("text", "")]))

    def stop(self) -> None:
        if _kind(self.c.runner.status()) == "sandbox":
            self.c.runner.stop()

    def reset(self) -> None:
        code, res = self.c._reset_sandbox()
        if code != 200:
            raise RuntimeError(res.get("text") or "重置记忆没成")

    def running(self) -> bool:
        return self.c._state_of("sandbox") in BUSY

    def op(self, req: dict) -> dict:
        code, _, raw = self.c.sandbox_proxy("POST", "op", "", json.dumps(req, ensure_ascii=False).encode())
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            data = None
        return data if isinstance(data, dict) else {"ok": False, "text": f"沙盒没回应（{code}）"}

    def state(self, after: int) -> dict | None:
        code, _, raw = self.c.sandbox_proxy("GET", "state", f"after={int(after)}&wait=1")
        if code != 200:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def inner(self) -> dict:
        return self.c.inner("sandbox")[1].get("now") or {}

    def diary(self) -> str:
        path = self.c.sandbox_dir() / "memory" / "inner" / "diary.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""
