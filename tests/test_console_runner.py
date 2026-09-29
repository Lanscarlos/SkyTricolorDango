import os
import socket
import sys
import time
from pathlib import Path

import pytest

from skydango.console.runner import (
    CRASHED,
    EXITED,
    RUNNING,
    STARTING,
    LaunchOptions,
    Runner,
    _popen_flags,
    build_command,
    child_env,
    probe_status,
    send_shutdown,
)

FAKE = Path(__file__).with_name("fake_child.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def runner(tmp_path, **kw):
    kw.setdefault("stop_timeout", 2.0)
    return Runner(tmp_path, free_port(), log_lines=kw.pop("log_lines", 50), **kw)


def child(r, mode="normal"):
    return [sys.executable, str(FAKE), "--port", str(r.child_port), "--mode", mode]


def wait_state(r, *states, timeout=10):
    end = time.monotonic() + timeout
    while r.status()["state"] not in states:
        assert time.monotonic() < end, r.status()
        time.sleep(0.05)


SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_build_command_is_explicit():
    cmd = build_command(LaunchOptions(brain=False, live=False, emotes=False, duration=90), Path("c.toml"), 8761, 42, python="py")
    assert cmd == ["py", "-m", "skydango", "-c", "c.toml", "run", "--no-brain", "--dry-run", "--no-emotes", "--duration", "90.0",
                   "--view", "--viewer-port", "8761", "--no-browser", "--parent-pid", "42"]
    assert "--live" in build_command(LaunchOptions(live=True), Path("c.toml"), 1, 2)
    assert "--no-brain" not in build_command(LaunchOptions(), Path("c.toml"), 1, 2)
    assert "--duration" not in build_command(LaunchOptions(), Path("c.toml"), 1, 2)


def test_child_env_injects_secrets_and_utf8():
    env = child_env({"PATH": "p", "K": "old"}, {"K": "new"})
    assert env == {"PATH": "p", "K": "new", "PYTHONIOENCODING": "utf-8"}


def test_start_run_stop_exits_cleanly(tmp_path):
    r = runner(tmp_path)
    r.start(child(r), dict(os.environ), LaunchOptions())
    wait_state(r, RUNNING)
    assert r.status()["run_dir"] == "/tmp/runs/x" and r.status()["options"]["brain"] is True
    with pytest.raises(RuntimeError, match="已经在运行"):
        r.start(child(r), dict(os.environ), LaunchOptions())
    r.stop()
    r.stop()  # 连点两次：Review Focus 3
    wait_state(r, EXITED)
    assert r.status()["forced"] is False and r.status()["exit_code"] == 0
    assert "收到退出" in r.logs()["lines"]


def test_ignored_shutdown_is_killed_after_timeout(tmp_path):
    r = runner(tmp_path, stop_timeout=1.0)
    r.start(child(r, "ignore"), dict(os.environ), LaunchOptions())
    wait_state(r, RUNNING)
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is True


def test_stop_while_starting_retries_then_kills(tmp_path):  # 子进程一直不开端口
    r = runner(tmp_path, stop_timeout=1.0)
    r.start(SLEEPER, dict(os.environ), LaunchOptions())
    assert r.status()["state"] == STARTING
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is True


def test_crash_keeps_logs(tmp_path):
    r = runner(tmp_path)
    r.start(child(r, "crash"), dict(os.environ), LaunchOptions())
    wait_state(r, CRASHED)
    assert r.status()["exit_code"] == 3 and "boom" in r.logs()["lines"]


def test_restart_after_exit_clears_old_logs(tmp_path):
    r = runner(tmp_path)
    r.start(child(r, "crash"), dict(os.environ), LaunchOptions())
    wait_state(r, CRASHED)
    r.start([sys.executable, "-c", "print('second')"], dict(os.environ), LaunchOptions())
    wait_state(r, EXITED)
    assert r.logs()["lines"] == ["second"] and r.status()["run_dir"] is None


def test_bad_bytes_and_long_lines_do_not_break_logs(tmp_path):  # Review Focus 1
    r = runner(tmp_path)
    r.start(child(r, "gbk"), dict(os.environ), LaunchOptions())
    wait_state(r, RUNNING)
    lines = r.logs()["lines"]
    assert any("�" in line for line in lines) and max(len(line) for line in lines) <= 2000
    r.stop()
    wait_state(r, EXITED)


def test_logs_ring_buffer_and_after(tmp_path):
    r = runner(tmp_path, log_lines=3)
    r.start([sys.executable, "-c", "print('\\n'.join(map(str, range(10))))"], dict(os.environ), LaunchOptions())
    wait_state(r, EXITED)
    first = r.logs()
    assert first["lines"] == ["7", "8", "9"] and first["next"] == 10
    assert r.logs(after=first["next"])["lines"] == []
    assert r.logs(after=9)["lines"] == ["9"]


def test_slow_start_flag(tmp_path):
    r = runner(tmp_path, stop_timeout=0.5, start_notice=0.1)
    r.start(SLEEPER, dict(os.environ), LaunchOptions())
    assert r.status()["slow_start"] is False
    time.sleep(0.3)
    assert r.status()["slow_start"] is True
    r.stop()
    wait_state(r, EXITED)


def test_close_blocks_until_stopped(tmp_path):
    r = runner(tmp_path)
    r.start(child(r), dict(os.environ), LaunchOptions())
    wait_state(r, RUNNING)
    r.close()
    assert r.status()["state"] == EXITED


def test_local_requests_ignore_proxy(tmp_path, monkeypatch):  # 终审 Important 1：开着 Clash 的机器
    r = runner(tmp_path)
    r.start(child(r, "ignore"), dict(os.environ), LaunchOptions())
    wait_state(r, RUNNING)
    try:
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        assert probe_status(r.child_port) and send_shutdown(r.child_port)
    finally:
        r.stop()
        wait_state(r, EXITED)


def test_windows_child_gets_hidden_console_of_its_own():  # 终审：关面板窗口不能连带把团子直接结束
    assert _popen_flags("win32") == {"creationflags": 0x00000200 | 0x08000000}
    assert _popen_flags("linux") == {"start_new_session": True}


def test_kill_forces_immediately(tmp_path):
    r = runner(tmp_path, stop_timeout=30.0)
    r.start(SLEEPER, dict(os.environ), LaunchOptions())
    r.kill()
    wait_state(r, EXITED, CRASHED, timeout=5)
    assert r.status()["forced"] is True


def test_carriage_return_progress_keeps_latest(tmp_path):  # 进度条只用 \r 刷新：留最新的，不攒成一行
    r = runner(tmp_path)
    code = "import sys; sys.stdout.write('10%\\r20%\\r30%\\ndone\\nx\\ry'); sys.stdout.flush()"
    r.start([sys.executable, "-c", code], dict(os.environ), LaunchOptions())
    wait_state(r, EXITED)
    assert r.logs()["lines"] == ["30%", "done", "y"]


def test_endless_line_without_newline_is_cut(tmp_path):
    r = runner(tmp_path)
    code = "import sys; sys.stdout.write('x' * 10000); sys.stdout.flush()"
    r.start([sys.executable, "-c", code], dict(os.environ), LaunchOptions())
    wait_state(r, EXITED)
    lines = r.logs()["lines"]
    assert lines and max(len(line) for line in lines) <= 2000


def test_stop_while_starting_then_port_opens_exits_cleanly(tmp_path):  # 重试的 shutdown 送到了，就不用强杀
    r = runner(tmp_path, stop_timeout=8.0)
    r.start(child(r, "late"), dict(os.environ), LaunchOptions())
    assert r.status()["state"] == STARTING
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is False and "收到退出" in r.logs()["lines"]
