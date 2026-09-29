import os
import subprocess
import sys
import threading
import time

from skydango.console.watchdog import pid_alive, watch_parent


def test_interrupts_once_when_parent_gone():
    state, hits = {"alive": True}, []
    t = watch_parent(123, lambda: hits.append(1), alive=lambda pid: state["alive"], interval=0.01)
    time.sleep(0.05)
    assert hits == []
    state["alive"] = False
    t.join(1)
    assert hits == [1] and not t.is_alive()


def test_stop_event_ends_watch():
    stop = threading.Event()
    t = watch_parent(1, lambda: None, alive=lambda pid: True, interval=0.01, stop=stop)
    stop.set()
    t.join(1)
    assert not t.is_alive()


def test_pid_alive_for_self_and_dead_child():
    assert pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not pid_alive(p.pid)
