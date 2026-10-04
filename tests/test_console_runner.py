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


def ENV() -> dict[str, str]:
    """和面板一样经 child_env：子进程输出 UTF-8。直接传 os.environ 时中文按 GBK 打印，runner 解码成乱码（中文 Windows）。"""
    return child_env(os.environ, {})


SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_build_command_is_explicit():
    cmd = build_command(LaunchOptions(brain=False, live=False, emotes=False, duration=90), Path("c.toml"), 42, python="py")
    assert cmd == ["py", "-m", "skydango", "-c", "c.toml", "run", "--no-brain", "--dry-run", "--no-emotes", "--duration", "90.0",
                   "--parent-pid", "42"]  # 接口端口来自同一份配置，不再传 --view / --viewer-port / --no-browser
    assert "--live" in build_command(LaunchOptions(live=True), Path("c.toml"), 2)
    assert "--no-brain" not in build_command(LaunchOptions(), Path("c.toml"), 2)
    assert "--duration" not in build_command(LaunchOptions(), Path("c.toml"), 2)


def test_child_env_injects_secrets_and_utf8():
    env = child_env({"PATH": "p", "K": "old"}, {"K": "new"})
    assert env == {"PATH": "p", "K": "new", "PYTHONIOENCODING": "utf-8"}


def test_start_run_stop_exits_cleanly(tmp_path):
    r = runner(tmp_path)
    r.start(child(r), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    assert r.status()["run_dir"] == "/tmp/runs/x" and r.status()["options"]["brain"] is True
    with pytest.raises(RuntimeError, match="已经在运行"):
        r.start(child(r), ENV(), LaunchOptions())
    r.stop()
    r.stop()  # 连点两次：Review Focus 3
    wait_state(r, EXITED)
    assert r.status()["forced"] is False and r.status()["exit_code"] == 0
    assert "收到退出" in r.logs()["lines"]


def test_ignored_shutdown_is_killed_after_timeout(tmp_path):
    r = runner(tmp_path, stop_timeout=1.0)
    r.start(child(r, "ignore"), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is True


def test_stop_while_starting_retries_then_kills(tmp_path):  # 子进程一直不开端口
    r = runner(tmp_path, stop_timeout=1.0)
    r.start(SLEEPER, ENV(), LaunchOptions())
    assert r.status()["state"] == STARTING
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is True


def test_crash_keeps_logs(tmp_path):
    r = runner(tmp_path)
    r.start(child(r, "crash"), ENV(), LaunchOptions())
    wait_state(r, CRASHED)
    assert r.status()["exit_code"] == 3 and "boom" in r.logs()["lines"]


def test_restart_after_exit_clears_old_logs(tmp_path):
    r = runner(tmp_path)
    r.start(child(r, "crash"), ENV(), LaunchOptions())
    wait_state(r, CRASHED)
    r.start([sys.executable, "-c", "print('second')"], ENV(), LaunchOptions())
    wait_state(r, EXITED)
    assert r.logs()["lines"] == ["second"] and r.status()["run_dir"] is None


def test_bad_bytes_and_long_lines_do_not_break_logs(tmp_path):  # Review Focus 1
    r = runner(tmp_path)
    r.start(child(r, "gbk"), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    lines = r.logs()["lines"]
    assert any("�" in line for line in lines) and max(len(line) for line in lines) <= 2000
    r.stop()
    wait_state(r, EXITED)


def test_logs_ring_buffer_and_after(tmp_path):
    r = runner(tmp_path, log_lines=3)
    r.start([sys.executable, "-c", "print('\\n'.join(map(str, range(10))))"], ENV(), LaunchOptions())
    wait_state(r, EXITED)
    first = r.logs()
    assert first["lines"] == ["7", "8", "9"] and first["next"] == 10
    assert r.logs(after=first["next"])["lines"] == []
    assert r.logs(after=9)["lines"] == ["9"]


def test_slow_start_flag(tmp_path):
    r = runner(tmp_path, stop_timeout=0.5, start_notice=0.1)
    r.start(SLEEPER, ENV(), LaunchOptions())
    assert r.status()["slow_start"] is False
    time.sleep(0.3)
    assert r.status()["slow_start"] is True
    r.stop()
    wait_state(r, EXITED)


def test_close_blocks_until_stopped(tmp_path):
    r = runner(tmp_path)
    r.start(child(r), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    r.close()
    assert r.status()["state"] == EXITED


def test_local_requests_ignore_proxy(tmp_path, monkeypatch):  # 终审 Important 1：开着 Clash 的机器
    r = runner(tmp_path)
    r.start(child(r, "ignore"), ENV(), LaunchOptions())
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
    r.start(SLEEPER, ENV(), LaunchOptions())
    r.kill()
    wait_state(r, EXITED, CRASHED, timeout=5)
    assert r.status()["forced"] is True


def test_carriage_return_progress_keeps_latest(tmp_path):  # 进度条只用 \r 刷新：留最新的，不攒成一行
    r = runner(tmp_path)
    code = "import sys; sys.stdout.write('10%\\r20%\\r30%\\ndone\\nx\\ry'); sys.stdout.flush()"
    r.start([sys.executable, "-c", code], ENV(), LaunchOptions())
    wait_state(r, EXITED)
    assert r.logs()["lines"] == ["30%", "done", "y"]


def test_endless_line_without_newline_is_cut(tmp_path):
    r = runner(tmp_path)
    code = "import sys; sys.stdout.write('x' * 10000); sys.stdout.flush()"
    r.start([sys.executable, "-c", code], ENV(), LaunchOptions())
    wait_state(r, EXITED)
    lines = r.logs()["lines"]
    assert lines and max(len(line) for line in lines) <= 2000


def test_stop_while_starting_then_port_opens_exits_cleanly(tmp_path):  # 重试的 shutdown 送到了，就不用强杀
    r = runner(tmp_path, stop_timeout=8.0)
    r.start(child(r, "late"), ENV(), LaunchOptions())
    assert r.status()["state"] == STARTING
    r.stop()
    wait_state(r, EXITED)
    assert r.status()["forced"] is False and "收到退出" in r.logs()["lines"]


# ---- 沙盒计划 Task 6：团子 / 沙盒一个槽，按 kind 用各自的端口 ----
def test_sandbox_kind_uses_its_port_and_blocks_dango(tmp_path):
    from skydango.console.runner import build_sandbox_command

    r = runner(tmp_path)
    port = free_port()
    r.start([sys.executable, str(FAKE), "--port", str(port)], ENV(), {"start": "resume"}, kind="sandbox", port=port)
    wait_state(r, RUNNING)  # 探活用这次的端口，不是 child_port
    st = r.status()
    assert st["kind"] == "sandbox" and st["port"] == port and st["options"] == {"start": "resume"}
    with pytest.raises(RuntimeError, match="沙盒已经在运行，先停沙盒"):
        r.start(child(r), ENV(), LaunchOptions())
    r.stop()
    wait_state(r, EXITED)  # 退出请求也发到沙盒端口
    assert r.status()["forced"] is False
    r.start(child(r), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    assert r.status()["kind"] == "dango" and r.status()["port"] == r.child_port
    with pytest.raises(RuntimeError, match="团子已经在运行，先停团子"):
        r.start(SLEEPER, ENV(), {"start": "resume"}, kind="sandbox", port=port)
    r.stop()
    wait_state(r, EXITED)
    assert build_sandbox_command(Path("c.toml"), 19392, 42, "sleep", python="py") == [
        "py", "-m", "skydango", "-c", "c.toml", "sandbox", "--port", "19392", "--no-browser", "--parent-pid", "42", "--start", "sleep"]


def _crash_with(tmp_path, lines: list[str]):
    r = runner(tmp_path)
    code = "import sys\nfor s in %r: print(s, flush=True)\nsys.exit(1)" % (lines,)
    r.start([sys.executable, "-c", code], ENV(), LaunchOptions())
    wait_state(r, CRASHED)
    return r.status()


def test_crashed_status_has_error_line(tmp_path):
    st = _crash_with(tmp_path, ["INFO 启动", "11:23:18 错误: 没有找到 API Key，请设置环境变量 DEEPSEEK_API_KEY", "bye"])
    assert st["state"] == "crashed" and st["error"] == "没有找到 API Key，请设置环境变量 DEEPSEEK_API_KEY"


def test_crashed_without_error_line_uses_last_line(tmp_path):
    st = _crash_with(tmp_path, ["a", "Traceback (most recent call last):", "ValueError: boom", ""])
    assert st["error"] == "ValueError: boom"


def test_running_has_no_error(tmp_path):
    r = runner(tmp_path)
    r.start(child(r), ENV(), LaunchOptions())
    wait_state(r, RUNNING)
    st = r.status()
    assert st["state"] == "running" and st["error"] is None
    r.stop()
    wait_state(r, EXITED)


# ---- 接管终端起的团子（spec 2026-10-04-console-attach §2）----
class Remote:
    """假的终端团子：probe 返回 info（None = 探不通），记下 shutdown / kill。"""

    def __init__(self, info):
        self.info, self.shutdowns, self.kills = info, [], []
        self.exit_on_shutdown = True

    def probe(self, port):
        return self.info

    def shutdown(self, port):
        self.shutdowns.append(port)
        if self.exit_on_shutdown:
            self.info = None
        return True

    def kill(self, pid):
        self.kills.append(pid)
        self.info = None


def info(tmp_path, pid=4242, **kw):
    return {"pid": pid, "run_dir": str(tmp_path), "live": True, "brain": True, "emotes": False, "duration": 0.0,
            "started": time.time() - 30, "console": False, **kw}


def attached(tmp_path, remote, **kw):
    kw.setdefault("stop_timeout", 0.5)
    return Runner(tmp_path, free_port(), log_lines=kw.pop("log_lines", 50), probe_run=remote.probe, shutdown=remote.shutdown,
                  kill=remote.kill, poll=0.05, **kw)


def test_attach_shows_terminal_run(tmp_path):
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    assert r.attach(remote.info) is True
    st = r.status()
    assert st["state"] == RUNNING and st["source"] == "terminal" and st["kind"] == "dango" and st["pid"] == 4242
    assert st["options"] == {"brain": True, "live": True, "emotes": False, "duration": 0.0}
    assert st["run_dir"] == str(tmp_path) and st["uptime"] >= 29
    r.close()


def test_own_child_has_console_source(tmp_path):
    r = runner(tmp_path)
    r.start(SLEEPER, ENV(), LaunchOptions())
    assert r.status()["source"] == "console"
    r.kill()
    wait_state(r, EXITED, CRASHED)


def test_attach_refused_while_busy(tmp_path):
    r = runner(tmp_path)
    r.start(SLEEPER, ENV(), LaunchOptions())
    assert r.attach(info(tmp_path)) is False and r.status()["source"] == "console"
    r.kill()
    wait_state(r, EXITED, CRASHED)


def test_attached_exits_after_three_failed_probes(tmp_path):
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    time.sleep(0.2)
    assert r.status()["state"] == RUNNING
    remote.info = None
    wait_state(r, EXITED, timeout=3)
    assert r.status()["exit_code"] is None and r.status()["forced"] is False


def test_attached_counts_another_pid_as_exited(tmp_path):  # 端口上换了一个团子：这个已经退了
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    remote.info = info(tmp_path, pid=1)
    wait_state(r, EXITED, timeout=3)


def test_attached_stop_clean(tmp_path):
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    r.stop()
    wait_state(r, EXITED, timeout=3)
    assert remote.shutdowns == [r.child_port] and remote.kills == [] and r.status()["forced"] is False


def test_attached_stop_sends_shutdown_then_kills_by_pid(tmp_path):
    remote = Remote(info(tmp_path))
    remote.exit_on_shutdown = False
    r = attached(tmp_path, remote, stop_timeout=0.3)
    r.attach(remote.info)
    r.stop()
    wait_state(r, EXITED, timeout=3)
    assert remote.shutdowns and remote.kills == [4242] and r.status()["forced"] is True


def test_attached_kill_is_immediate(tmp_path):
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    r.kill()
    wait_state(r, EXITED, timeout=3)
    assert remote.kills == [4242] and remote.shutdowns == []


def test_close_leaves_attached_running(tmp_path):  # Review Focus 4：面板退出不停终端起的团子
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    started = time.monotonic()
    r.close()
    assert time.monotonic() - started < 1.0
    assert remote.shutdowns == [] and remote.kills == [] and r.status()["state"] == "idle"


def test_attached_logs_come_from_agent_log(tmp_path):
    t = "2026-10-04 21:30:00,123"
    (tmp_path / "agent.log").write_text(f"{t} INFO a: 一\n{t} DEBUG a: 细\n{t} INFO a: 二\n", encoding="utf-8")
    remote = Remote(info(tmp_path))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    got = r.logs(0)
    assert got["lines"] == [f"{t} INFO a: 一", f"{t} INFO a: 二"] and got["next"] == 2
    assert got["note"] == "终端起的，日志来自 agent.log"
    with open(tmp_path / "agent.log", "a", encoding="utf-8") as f:
        f.write(f"{t} WARNING a: 三\n")
    assert r.logs(2)["lines"] == [f"{t} WARNING a: 三"]
    r.close()


def test_attached_logs_without_file_only_note(tmp_path):
    remote = Remote(info(tmp_path / "gone"))
    r = attached(tmp_path, remote)
    r.attach(remote.info)
    assert r.logs(0) == {"next": 0, "lines": [], "note": "终端起的，日志来自 agent.log"}
    r.close()


def test_own_child_logs_have_no_note(tmp_path):
    assert "note" not in runner(tmp_path).logs(0)


def test_probe_run_needs_run_section():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from skydango.console.runner import probe_run

    replies = [{"seq": 0, "info": {}}, {"seq": 0, "info": {}, "run": {"pid": 1}}, {"seq": 0, "run": "x"}]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            raw = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert probe_run(port) is None
        assert probe_run(port) == {"pid": 1}
        assert probe_run(port) is None
    finally:
        srv.shutdown()
        srv.server_close()
    assert probe_run(port) is None  # 探不通
