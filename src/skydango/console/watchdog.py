"""子进程的父进程看门狗：面板（父进程）没了，团子自己走正常退出（复原镜头、换回轮盘），不留孤儿。"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable


def once(fn: Callable[[], None]) -> Callable[[], None]:
    """只触发一次：/shutdown 和看门狗都要团子退出时，第二次中断会落在收尾（换回轮盘、写经过）的中途把它打断。"""
    lock, done = threading.Lock(), []

    def call() -> None:
        with lock:
            if done:
                return
            done.append(True)
        fn()

    return call


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        synchronize, wait_timeout = 0x00100000, 0x102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch_parent(
    pid: int,
    interrupt: Callable[[], None],
    alive: Callable[[int], bool] = pid_alive,
    interval: float = 2.0,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """每 interval 秒看一次 pid 还在不在；不在了调一次 interrupt 就结束。"""
    stop = stop or threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            if not alive(pid):
                interrupt()
                return

    thread = threading.Thread(target=loop, name="parent-watch", daemon=True)
    thread.start()
    return thread
