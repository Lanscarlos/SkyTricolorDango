"""面板的任务槽：整理难例 / 一键重训这类"跑几分钟、只关心进度和结果"的子进程。

和 runner.Runner 分开：任务没有 HTTP 接口，不探活、不请它自己退出；停就是进程树强杀。
子进程输出逐行写 `tmp/jobs/<时间>-<任务>.log`；最后一行 `PROGRESS ` 之后的文字是进度。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .runner import _popen_flags, kill_tree

JOBS = ("inbox", "retrain")
TAIL_LINES = 20
PROGRESS = "PROGRESS "
IDLE, RUNNING, DONE, FAILED, STOPPED = "idle", "running", "done", "failed", "stopped"


class JobRunner:
    def __init__(self, cwd: Path, kill: Callable[[int], None] = kill_tree, popen: Callable = subprocess.Popen) -> None:
        self.cwd, self._kill, self._popen = Path(cwd), kill, popen
        self._lock = threading.Lock()
        self._state, self._job = IDLE, None
        self._progress, self._exit_code, self._log = "", None, None
        self._tail: deque[str] = deque(maxlen=TAIL_LINES)
        self._proc = None
        self._gen = 0

    def start(self, job: str, cmd: list[str], env: dict) -> None:
        if job not in JOBS:
            raise ValueError(f"不认识的任务：{job}")
        with self._lock:
            if self._state == RUNNING:
                raise RuntimeError("任务还在跑，等它做完或先停掉")
            logdir = self.cwd / "tmp" / "jobs"
            logdir.mkdir(parents=True, exist_ok=True)
            self._log = logdir / f"{time.strftime('%Y%m%d-%H%M%S')}-{job}.log"
            self._proc = self._popen(
                cmd, cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                **_popen_flags(sys.platform),
            )
            self._gen += 1
            self._job, self._state = job, RUNNING
            self._progress, self._exit_code = "", None
            self._tail.clear()
            proc, gen, log = self._proc, self._gen, self._log
        threading.Thread(target=self._read, args=(proc, gen, log), name=f"console-job-{job}", daemon=True).start()

    def _read(self, proc, gen: int, log: Path) -> None:
        with log.open("w", encoding="utf-8", errors="replace") as out:
            for raw in proc.stdout:
                text = raw.decode("utf-8", "replace").rstrip("\r\n")
                out.write(text + "\n")
                out.flush()
                with self._lock:
                    if self._gen != gen:
                        continue
                    self._tail.append(text)
                    if text.startswith(PROGRESS):
                        self._progress = text[len(PROGRESS):].strip()
        code = proc.wait()
        with self._lock:
            if self._gen != gen:
                return
            self._exit_code = code
            if self._state == RUNNING:
                self._state = DONE if code == 0 else FAILED

    def stop(self) -> None:
        with self._lock:
            if self._state != RUNNING or self._proc is None:
                return
            self._state = STOPPED
            pid = self._proc.pid
        self._kill(pid)

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "job": self._job, "progress": self._progress, "tail": list(self._tail),
                    "exit_code": self._exit_code, "log": str(self._log) if self._log else None}
