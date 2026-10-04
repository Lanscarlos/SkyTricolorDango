"""管理面板起停子进程：团子（`python -m skydango run …`）或大脑沙盒（`python -m skydango sandbox …`）。
状态机、日志尾巴、先请它自己退出、超时再强杀。

子进程收到 POST /shutdown 走和 Ctrl+C 一样的收尾（复原镜头、换回轮盘、写日记、结束 Claude Code）；
面板没了，子进程的 --parent-pid 看门狗会让它自己退出。同一时间只有一个子进程（团子和沙盒互斥：共用令牌和 .brain-claude/）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from .logtail import LogTail

log = logging.getLogger(__name__)

IDLE, STARTING, RUNNING, STOPPING, EXITED, CRASHED = "idle", "starting", "running", "stopping", "exited", "crashed"
MAX_LINE = 2000  # 日志一行最多留几个字（进度条之类的超长行）
RUN_DIR = re.compile(r"本次运行的日志和截图: (.+)$")
LINE_END = re.compile(rb"\r\n|\n|\r")


@dataclass
class LaunchOptions:
    brain: bool = True  # False = 普通 Agent（调试用）
    live: bool = False
    emotes: bool = True
    duration: float = 0.0  # 0 = 一直跑


def build_command(opts: LaunchOptions, config_path: Path, parent_pid: int, python: str = sys.executable) -> list[str]:
    """模式和真发 / 只看总是显式传：config.toml 里的 dry_run 不会偷偷改掉面板上的选择。"""
    cmd = [python, "-m", "skydango", "-c", str(config_path), "run"]
    if not opts.brain:
        cmd.append("--no-brain")
    cmd.append("--live" if opts.live else "--dry-run")
    if not opts.emotes:
        cmd.append("--no-emotes")
    if opts.duration > 0:
        cmd += ["--duration", str(float(opts.duration))]
    return cmd + ["--parent-pid", str(parent_pid)]  # 接口端口来自同一份配置（[viewer] port）


def build_sandbox_command(config_path: Path, port: int, parent_pid: int, start: str, python: str = sys.executable) -> list[str]:
    """大脑沙盒子进程（spec 2026-09-30-brain-sandbox §2）：--start 透传（resume / sleep / HH:MM / YYYY-MM-DD HH:MM）。"""
    return [python, "-m", "skydango", "-c", str(config_path), "sandbox", "--port", str(port), "--no-browser",
            "--parent-pid", str(parent_pid), "--start", start]


KIND_NAMES = {"dango": "团子", "sandbox": "沙盒"}
TERMINAL_NOTE = "终端起的，日志来自 agent.log"
ATTACH_MISSES = 3  # 接管的团子连续几次探不通算退出了


def child_env(base: Mapping[str, str], secrets: Mapping[str, str]) -> dict[str, str]:
    """子进程的环境变量：面板的 + secrets.toml 的（覆盖）+ UTF-8 输出（Windows 默认 GBK，日志会花）。"""
    return {**base, **secrets, "PYTHONIOENCODING": "utf-8"}


# 本机请求不走代理：urllib 默认读 HTTP_PROXY 和 Windows 注册表里的代理（开着 Clash 时 127.0.0.1 也会被转走）
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe_status(port: int, timeout: float = 1.0) -> bool:
    try:
        with LOCAL.open(f"http://127.0.0.1:{port}/status", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def probe_run(port: int, timeout: float = 0.5) -> dict | None:
    """端口上是不是团子：/status 里带 run 节（dict）才算，返回它；别的程序、旧版本的团子、探不通都是 None。"""
    try:
        with LOCAL.open(f"http://127.0.0.1:{port}/status", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    run = data.get("run") if isinstance(data, dict) else None
    return run if isinstance(run, dict) else None


def port_free(port: int) -> bool:
    """在 127.0.0.1 上试绑定一下：绑得上 = 没人占着。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def send_shutdown(port: int, timeout: float = 2.0) -> bool:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/shutdown", data=b"{}", method="POST",
        headers={"Content-Type": "application/json", "X-Skydango": "1"},
    )
    try:
        with LOCAL.open(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def kill_tree(pid: int) -> None:
    """连子进程（Claude Code、MCP 服务）一起结束。"""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _popen_flags(platform: str) -> dict:
    """Windows：新进程组（面板终端的 Ctrl+C 不直接落到团子身上）+ 团子自己一个不显示的控制台
    （关掉面板的终端窗口不会连带把团子直接结束，而是由 --parent-pid 看门狗让它正常收尾）。"""
    if platform == "win32":
        return {"creationflags": 0x00000200 | 0x08000000}  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    return {"start_new_session": True}


class Runner:
    def __init__(
        self,
        cwd: Path,
        child_port: int,
        stop_timeout: float,
        log_lines: int,
        probe: Callable[[int], bool] = probe_status,
        shutdown: Callable[[int], bool] = send_shutdown,
        kill: Callable[[int], None] = kill_tree,
        start_notice: float = 60.0,
        probe_run: Callable[[int], dict | None] = probe_run,
        poll: float = 1.0,
    ) -> None:
        self.cwd, self.child_port, self.stop_timeout, self.start_notice = Path(cwd), child_port, stop_timeout, start_notice
        self._probe, self._shutdown, self._kill, self._probe_run, self._poll = probe, shutdown, kill, probe_run, poll
        self._lock = threading.Lock()
        self._lines: deque[tuple[int, str]] = deque(maxlen=log_lines)
        self._count = 0
        self._state = IDLE
        self._proc: subprocess.Popen | None = None
        self._started: float | None = None
        self._exit_code: int | None = None
        self._run_dir: str | None = None
        self._forced = False
        self._options: LaunchOptions | dict | None = None
        self._kind = "dango"  # 这个槽现在（或最近一次）是团子还是沙盒
        self._port = child_port  # 这次的子进程接口端口：探活、请它退出都用它
        # 接管终端起的团子（spec 2026-10-04-console-attach §2）：不是 Popen 出来的，靠探 /status 活着、按 pid 强杀
        self._source = "console"  # console = 面板自己起的；terminal = 接管的
        self._attached: dict | None = None  # 接管时 /status 里的 run 节
        self._tail: LogTail | None = None
        self._gen = 0  # 每次起 / 接管 / 放手加一：旧的后台线程看到对不上就收工

    # ---- 起 ----
    def start(self, cmd: list[str], env: dict[str, str], options, kind: str = "dango", port: int | None = None) -> None:
        """kind：dango 团子 / sandbox 沙盒；port 默认 child_port。已经有一个在跑（不管哪种）就 RuntimeError。"""
        if kind not in KIND_NAMES:
            raise ValueError(f"不认识的子进程类型：{kind}")
        with self._lock:
            if self._state not in (IDLE, EXITED, CRASHED):
                who = KIND_NAMES[self._kind]
                raise RuntimeError(f"{who}已经在运行，先停{who}")
            self._lines.clear()
            self._count, self._exit_code, self._run_dir, self._forced, self._options = 0, None, None, False, options
            self._kind, self._port = kind, self.child_port if port is None else port
            self._source, self._attached, self._tail = "console", None, None
            self._gen += 1
            self._proc = subprocess.Popen(
                cmd, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                **_popen_flags(sys.platform),
            )
            self._started = time.monotonic()
            self._state = STARTING
            proc = self._proc
        reader = threading.Thread(target=self._read, args=(proc,), name="console-child-out", daemon=True)
        reader.start()
        threading.Thread(target=self._monitor, args=(proc, reader), name="console-child", daemon=True).start()
        log.info("起%s：%s", KIND_NAMES[kind], " ".join(cmd))

    def attach(self, info: dict) -> bool:
        """接管终端起的团子（info = 它 /status 里的 run 节）。槽忙着返回 False。"""
        with self._lock:
            if self._state not in (IDLE, EXITED, CRASHED):
                return False
            self._lines.clear()
            self._count, self._exit_code, self._forced, self._proc = 0, None, False, None
            self._options = LaunchOptions(bool(info.get("brain", True)), bool(info.get("live")), bool(info.get("emotes", True)),
                                          float(info.get("duration") or 0))
            self._kind, self._port, self._source, self._attached = "dango", self.child_port, "terminal", dict(info)
            self._run_dir = str(info["run_dir"]) if info.get("run_dir") else None
            self._tail = LogTail(Path(self._run_dir) / "agent.log") if self._run_dir else None
            self._started = time.monotonic()
            self._state = RUNNING
            self._gen += 1
            gen = self._gen
        threading.Thread(target=self._watch_attached, args=(gen,), name="console-attached", daemon=True).start()
        log.info("接上了终端起的团子（pid %s，运行目录 %s）", info.get("pid"), self._run_dir)
        return True

    def _same_run(self, gen: int) -> dict | None:
        """还是这次接管的（没被放手、没换人）就返回它的 run 节；调用时已持锁。"""
        return self._attached if gen == self._gen and self._source == "terminal" else None

    def _alive(self, pid) -> bool:
        got = self._probe_run(self._port)
        return got is not None and got.get("pid") == pid

    def _gone(self, gen: int, forced: bool = False) -> None:
        with self._lock:
            if self._same_run(gen) is not None and self._state in (RUNNING, STOPPING):
                self._state, self._exit_code = EXITED, None
                self._forced = self._forced or forced
                log.info("终端起的团子退出了")

    def _watch_attached(self, gen: int) -> None:
        """接管期间每 poll 秒探一次：连续 ATTACH_MISSES 次探不通（或端口上换了个 pid）= 退出了。"""
        misses = 0
        while True:
            time.sleep(self._poll)
            with self._lock:
                run = self._same_run(gen)
                if run is None or self._state not in (RUNNING, STOPPING):
                    return
            misses = 0 if self._alive(run.get("pid")) else misses + 1
            if misses >= ATTACH_MISSES:
                self._gone(gen)
                return

    def _stop_attached(self, gen: int, pid) -> None:
        """先请它自己退出，stop_timeout 秒还在就按 pid 结束进程树。"""
        deadline = time.monotonic() + self.stop_timeout
        sent = False
        while time.monotonic() < deadline:
            with self._lock:
                if self._same_run(gen) is None or self._state != STOPPING:
                    return
            if not sent:
                sent = self._shutdown(self._port)
            if not self._alive(pid):
                self._gone(gen)
                return
            time.sleep(min(0.5, self._poll * 2))
        log.warning("终端起的团子 %.0f 秒没退出，强制结束（轮盘可能没换回）", self.stop_timeout)
        self._kill(pid)
        self._gone(gen, forced=True)

    def _read(self, proc: subprocess.Popen) -> None:
        """按 \n 或单独的 \r 分行：进度条只用 \r 刷新时，新的一段顶掉上一段（像终端那样），不攒成一整行。"""
        buf, overwrite = b"", False
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                end = LINE_END.search(buf)
                if end is None or (end.group() == b"\r" and end.end() == len(buf)):  # 可能是 \r\n 的前半个，等下一块
                    break
                self._line(buf[: end.start()], overwrite)
                buf, overwrite = buf[end.end():], end.group() == b"\r"
            buf = buf[: MAX_LINE * 4]  # 一直不换行的输出：只留开头，别无限攒（UTF-8 一个字最多 4 字节）
        if buf.rstrip(b"\r"):
            self._line(buf.rstrip(b"\r"), overwrite)

    def _line(self, raw: bytes, overwrite: bool) -> None:
        text = raw.decode("utf-8", "replace")[:MAX_LINE]
        with self._lock:
            if overwrite and self._lines:
                self._lines.pop()
            self._count += 1
            self._lines.append((self._count, text))
            found = RUN_DIR.search(text)
            if found and self._run_dir is None:
                self._run_dir = found.group(1).strip()

    def _monitor(self, proc: subprocess.Popen, reader: threading.Thread) -> None:
        while True:
            try:
                code = proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if self._state == STARTING and self._probe(self._port):
                with self._lock:
                    if self._state == STARTING:
                        self._state = RUNNING
        reader.join(2.0)  # 最后几行日志读完再改状态；孙进程还拿着管道也不死等
        with self._lock:
            self._exit_code = code
            self._state = EXITED if self._state == STOPPING or code == 0 else CRASHED
        log.info("%s退出了（退出码 %s）", KIND_NAMES[self._kind], code)

    # ---- 停 ----
    def stop(self) -> None:
        """马上返回；后台先请它自己退出（启动中还没开端口就一直重试），stop_timeout 秒还没退就强杀。"""
        with self._lock:
            if self._state not in (STARTING, RUNNING):
                return
            self._state = STOPPING
            proc, port, attached, gen = self._proc, self._port, self._same_run(self._gen), self._gen
        if attached is not None:
            threading.Thread(target=self._stop_attached, args=(gen, attached.get("pid")), name="console-stop", daemon=True).start()
            return
        threading.Thread(target=self._stop, args=(proc, port), name="console-stop", daemon=True).start()

    def _stop(self, proc: subprocess.Popen, port: int) -> None:
        deadline = time.monotonic() + self.stop_timeout
        sent = False
        while proc.poll() is None and time.monotonic() < deadline:
            if not sent:
                sent = self._shutdown(port)
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            log.warning("%s %.0f 秒没退出，强制结束（轮盘可能没换回）", KIND_NAMES[self._kind], self.stop_timeout)
            with self._lock:
                self._forced = True
            self._kill(proc.pid)

    def kill(self) -> None:
        """马上强杀（面板退出时又按了一次 Ctrl+C）。"""
        with self._lock:
            proc, attached, gen = self._proc, self._same_run(self._gen), self._gen
            if self._state not in (STARTING, RUNNING, STOPPING) or (proc is None and attached is None):
                return
            self._state, self._forced = STOPPING, True
        if attached is not None:
            self._kill(attached.get("pid"))
            self._gone(gen, forced=True)
            return
        self._kill(proc.pid)

    def close(self) -> None:
        """面板退出用：自己起的停掉并等到真的退出；接管的只放手（不是面板的子进程，接着跑）。"""
        with self._lock:
            if self._same_run(self._gen) is not None:
                self._gen += 1
                self._state, self._attached, self._tail, self._source = IDLE, None, None, "console"
                return
        self.stop()
        while self.status()["state"] in (STARTING, RUNNING, STOPPING):
            time.sleep(0.1)

    # ---- 看 ----
    def status(self) -> dict:
        with self._lock:
            running = self._state in (STARTING, RUNNING, STOPPING)
            uptime = time.monotonic() - self._started if running and self._started is not None else None
            pid = self._proc.pid if running and self._proc else None
            if running and self._attached is not None:  # 接管的：pid、运行了多久按它自己报的
                pid = self._attached.get("pid")
                started = self._attached.get("started")
                uptime = max(0.0, time.time() - float(started)) if started is not None else uptime
            return {
                "state": self._state,
                "pid": pid,
                "uptime": uptime,
                "exit_code": self._exit_code,
                "run_dir": self._run_dir,
                "forced": self._forced,
                "slow_start": self._state == STARTING and uptime is not None and uptime > self.start_notice,
                "options": _options_dict(self._options),
                "kind": self._kind,
                "source": self._source,
                "port": self._port,
                "error": self._error_reason() if self._state == CRASHED else None,
            }

    def _error_reason(self) -> str | None:
        """崩溃原因：倒着找带「错误:」的行（取冒号后面的），没有就取最后一行非空的；调用时已持锁。"""
        last = None
        for _, text in reversed(self._lines):
            line = text.strip()
            if not line:
                continue
            if last is None:
                last = line
            i = line.find("错误:")
            if i >= 0:
                return line[i + len("错误:"):].strip()[:200]
        return last[:200] if last else None

    def logs(self, after: int = 0) -> dict:
        """行号从 1 连续编号；after = 已经拿到的最后一行，环形缓冲丢掉的行不再返回。
        接管的团子拿不到终端输出：每次先把 agent.log 新写的（INFO 以上）编号进来，带上 note。"""
        with self._lock:
            tail = self._tail if self._source == "terminal" else None
        fresh = tail.read_new() if tail is not None else None
        with self._lock:
            for text in fresh or []:
                self._count += 1
                self._lines.append((self._count, text[:MAX_LINE]))
            out = {"next": self._count, "lines": [text for n, text in self._lines if n > after]}
            if self._source == "terminal":
                out["note"] = TERMINAL_NOTE
            return out


def _options_dict(options) -> dict | None:
    if not options:
        return None
    return asdict(options) if is_dataclass(options) else dict(options)
