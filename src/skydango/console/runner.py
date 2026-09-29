"""管理面板起停团子子进程（`python -m skydango run --view …`）：状态机、日志尾巴、先请它自己退出、超时再强杀。

子进程收到 POST /shutdown 走和 Ctrl+C 一样的收尾（复原镜头、换回轮盘、结束 Claude Code）；
面板没了，子进程的 --parent-pid 看门狗会让它自己退出。同一时间只有一个子进程。
"""

from __future__ import annotations

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
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

IDLE, STARTING, RUNNING, STOPPING, EXITED, CRASHED = "idle", "starting", "running", "stopping", "exited", "crashed"
MAX_LINE = 2000  # 日志一行最多留几个字（进度条之类的超长行）
RUN_DIR = re.compile(r"本次运行的日志和截图: (.+)$")


@dataclass
class LaunchOptions:
    brain: bool = True  # False = 普通 Agent（调试用）
    live: bool = False
    emotes: bool = True
    duration: float = 0.0  # 0 = 一直跑


def build_command(opts: LaunchOptions, config_path: Path, child_port: int, parent_pid: int, python: str = sys.executable) -> list[str]:
    """模式和真发 / 只看总是显式传：config.toml 里的 dry_run 不会偷偷改掉面板上的选择。"""
    cmd = [python, "-m", "skydango", "-c", str(config_path), "run"]
    if not opts.brain:
        cmd.append("--no-brain")
    cmd.append("--live" if opts.live else "--dry-run")
    if not opts.emotes:
        cmd.append("--no-emotes")
    if opts.duration > 0:
        cmd += ["--duration", str(float(opts.duration))]
    return cmd + ["--view", "--viewer-port", str(child_port), "--no-browser", "--parent-pid", str(parent_pid)]


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
    ) -> None:
        self.cwd, self.child_port, self.stop_timeout, self.start_notice = Path(cwd), child_port, stop_timeout, start_notice
        self._probe, self._shutdown, self._kill = probe, shutdown, kill
        self._lock = threading.Lock()
        self._lines: deque[tuple[int, str]] = deque(maxlen=log_lines)
        self._count = 0
        self._state = IDLE
        self._proc: subprocess.Popen | None = None
        self._started: float | None = None
        self._exit_code: int | None = None
        self._run_dir: str | None = None
        self._forced = False
        self._options: LaunchOptions | None = None

    # ---- 起 ----
    def start(self, cmd: list[str], env: dict[str, str], options: LaunchOptions) -> None:
        with self._lock:
            if self._state not in (IDLE, EXITED, CRASHED):
                raise RuntimeError("团子已经在运行")
            self._lines.clear()
            self._count, self._exit_code, self._run_dir, self._forced, self._options = 0, None, None, False, options
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
        log.info("起团子：%s", " ".join(cmd))

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip("\r\n")[:MAX_LINE]
            with self._lock:
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
            if self._state == STARTING and self._probe(self.child_port):
                with self._lock:
                    if self._state == STARTING:
                        self._state = RUNNING
        reader.join(2.0)  # 最后几行日志读完再改状态；孙进程还拿着管道也不死等
        with self._lock:
            self._exit_code = code
            self._state = EXITED if self._state == STOPPING or code == 0 else CRASHED
        log.info("团子退出了（退出码 %s）", code)

    # ---- 停 ----
    def stop(self) -> None:
        """马上返回；后台先请它自己退出（启动中还没开端口就一直重试），stop_timeout 秒还没退就强杀。"""
        with self._lock:
            if self._state not in (STARTING, RUNNING):
                return
            self._state = STOPPING
            proc = self._proc
        threading.Thread(target=self._stop, args=(proc,), name="console-stop", daemon=True).start()

    def _stop(self, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.stop_timeout
        sent = False
        while proc.poll() is None and time.monotonic() < deadline:
            if not sent:
                sent = self._shutdown(self.child_port)
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            log.warning("团子 %.0f 秒没退出，强制结束（轮盘可能没换回）", self.stop_timeout)
            with self._lock:
                self._forced = True
            self._kill(proc.pid)

    def kill(self) -> None:
        """马上强杀（面板退出时又按了一次 Ctrl+C）。"""
        with self._lock:
            proc = self._proc
            if self._state not in (STARTING, RUNNING, STOPPING) or proc is None:
                return
            self._state, self._forced = STOPPING, True
        self._kill(proc.pid)

    def close(self) -> None:
        """面板退出用：停掉并等到真的退出。"""
        self.stop()
        while self.status()["state"] in (STARTING, RUNNING, STOPPING):
            time.sleep(0.1)

    # ---- 看 ----
    def status(self) -> dict:
        with self._lock:
            running = self._state in (STARTING, RUNNING, STOPPING)
            uptime = time.monotonic() - self._started if running and self._started is not None else None
            return {
                "state": self._state,
                "pid": self._proc.pid if running and self._proc else None,
                "uptime": uptime,
                "exit_code": self._exit_code,
                "run_dir": self._run_dir,
                "forced": self._forced,
                "slow_start": self._state == STARTING and uptime is not None and uptime > self.start_notice,
                "options": asdict(self._options) if self._options else None,
            }

    def logs(self, after: int = 0) -> dict:
        """行号从 1 连续编号；after = 已经拿到的最后一行，环形缓冲丢掉的行不再返回。"""
        with self._lock:
            return {"next": self._count, "lines": [text for n, text in self._lines if n > after]}
