import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skydango.console.devicecheck import Check
from skydango.console.server import ConsoleServer
from skydango.console.settings import SettingsStore

GOOD = {"Content-Type": "application/json", "X-Skydango": "1"}
OPTS = {"brain": True, "live": False, "emotes": True, "duration": 0}


def request(url, body: bytes | None = None, headers: dict | None = None):
    """返回 (状态码, JSON 或 None)；HTTP 错误也返回状态码。"""
    req = urllib.request.Request(url, data=body, headers=headers or {}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as err:
        raw, status = err.read(), err.code
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw.decode("utf-8", "replace")


class FakeRunner:
    def __init__(self):
        self.state, self.started, self.stopped = "idle", [], 0

    def status(self):
        return {"state": self.state, "pid": None, "uptime": None, "exit_code": None, "run_dir": None, "forced": False,
                "slow_start": False, "options": None}

    def logs(self, after=0):
        return {"next": 2, "lines": ["a", "b"][after:]}

    def start(self, cmd, env, options):
        self.started.append((cmd, env, options))
        self.state = "starting"

    def stop(self):
        self.stopped += 1


class Upstream(BaseHTTPRequestHandler):
    """假的子进程 viewer：GET 回路径和 query，POST 回收到的 Host / X-Skydango / body。"""

    def _reply(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/hang":  # 子进程卡住
            import time

            time.sleep(2)
        if path == "/partial":  # 子进程在传响应体的半路上被结束
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"half')
            self.wfile.flush()
            self.close_connection = True
            return
        self._reply({"path": path, "query": query})

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
        self.server.posts.append(self.path)
        self._reply({"host": self.headers.get("Host"), "x": self.headers.get("X-Skydango"), "body": body})

    def log_message(self, *a):
        pass


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.posts = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_server(tmp_path, upstream, secrets="", checks=None, child_port=None):
    if secrets:
        (tmp_path / "secrets.toml").write_text(secrets, encoding="utf-8")
    store = SettingsStore(tmp_path / "config.toml", environ={})
    runner = FakeRunner()
    srv = ConsoleServer(tmp_path / "config.toml", store, runner, port=0, child_port=child_port or upstream.server_address[1],
                        device_checks=checks or (lambda cfg, make: []), make_device=lambda cfg: None,
                        find_spec=lambda name: object())
    srv.url = srv.start()
    srv.fake_runner, srv.store = runner, store
    return srv


@pytest.fixture
def srv(tmp_path, upstream):
    s = make_server(tmp_path, upstream)
    yield s
    s.stop()


@pytest.fixture
def srv_with_token(tmp_path, upstream):
    s = make_server(tmp_path, upstream, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "tok"\nDEEPSEEK_API_KEY = "k"\n', child_port=free_port())
    yield s
    s.stop()


def test_page_is_public_but_api_checks_host(srv):
    status, page = request(srv.url)
    assert status == 200 and "<title>" in page
    assert request(srv.url + "api/state", headers={"Host": f"evil.com:{srv.port}"})[0] == 403
    assert request(srv.url + "live/status", headers={"Host": f"evil.com:{srv.port}"})[0] == 403


def test_post_needs_header_and_json(srv):
    assert request(srv.url + "api/run/stop", b"{}", {"Content-Type": "application/json"})[0] == 403
    assert request(srv.url + "api/run/stop", b"[]", GOOD)[0] == 400
    assert request(srv.url + "api/run/stop", b"not json", GOOD)[0] == 400
    assert request(srv.url + "api/run/stop", b"x" * 70000, GOOD)[0] == 413
    assert srv.fake_runner.stopped == 0
    assert request(srv.url + "api/run/stop", b"{}", GOOD) == (200, {"ok": True})
    assert srv.fake_runner.stopped == 1


def test_state_and_logs(srv):
    status, state = request(srv.url + "api/state")
    assert status == 200 and state["run"]["state"] == "idle" and state["launch"] == OPTS | {"duration": 0.0}
    assert [p["text"] for p in state["problems"]] == ["大脑模式要 Claude 令牌：去设置页填", "大脑离线时的备用回复要大模型的 API Key：去设置页填"] and state["orphan"] is True  # 假上游占着 child_port
    assert request(srv.url + "api/logs?after=1") == (200, {"next": 2, "lines": ["b"]})


@pytest.mark.parametrize("body", [{"brain": True, "live": False, "emotes": True, "duration": -1},
                                  {"brain": True, "live": False, "emotes": True, "duration": "abc"},
                                  {"brain": True, "live": False, "emotes": True, "duration": True},
                                  {"brain": "yes", "live": False, "emotes": True, "duration": 0},
                                  {"live": False, "emotes": True, "duration": 0}])
def test_start_rejects_bad_options(srv, body):  # Review Focus 5
    status, res = request(srv.url + "api/run/start", json.dumps(body).encode(), GOOD)
    assert status == 400 and not res["ok"] and res["text"] and srv.fake_runner.started == []


def test_start_reports_preflight_problems(srv):  # 没有 Claude 令牌
    status, res = request(srv.url + "api/run/start", json.dumps(OPTS).encode(), GOOD)
    assert status == 409 and "Claude 令牌" in res["problems"][0]["text"] and srv.fake_runner.started == []


def test_start_remembers_options_and_injects_secrets(srv_with_token):
    body = json.dumps({**OPTS, "live": True, "duration": 60}).encode()
    status, res = request(srv_with_token.url + "api/run/start", body, GOOD)
    cmd, env, opts = srv_with_token.fake_runner.started[0]
    assert status == 200 and res == {"ok": True}
    assert "--live" in cmd and "--parent-pid" in cmd and env["SKYDANGO_CLAUDE_TOKEN"] == "tok" and opts.live and opts.duration == 60.0
    assert srv_with_token.store.effective().console.live is True


def test_device_check_refused_while_running(srv):
    srv.fake_runner.state = "running"
    status, res = request(srv.url + "api/device", b"{}", GOOD)
    assert status == 409 and res["text"] == "团子运行中，设备归它用"


def test_device_check_returns_checks(tmp_path, upstream):
    s = make_server(tmp_path, upstream, checks=lambda cfg, make: [Check("adb", "adb", "ok", "v1")])
    try:
        status, res = request(s.url + "api/device", b"{}", GOOD)
        assert status == 200 and res["ok"] and res["checks"][0] == {"key": "adb", "label": "adb", "status": "ok", "detail": "v1",
                                                                     "hint": "", "data": {}}
    finally:
        s.stop()


def test_live_proxy_passes_query_and_503s(srv, upstream):
    assert request(srv.url + "live/snapshot?after=3") == (503, {"ok": False, "text": "团子没在运行"})
    srv.fake_runner.state = "running"
    assert request(srv.url + "live/snapshot?after=3") == (200, {"path": "/snapshot", "query": "after=3"})
    assert request(srv.url + "live/")[1]["path"] == "/"
    upstream.shutdown()
    upstream.server_close()  # Review Focus 4：子进程刚退出
    assert request(srv.url + "live/snapshot?after=3") == (503, {"ok": False, "text": "团子没在运行"})


def test_live_control_rewrites_host_and_header(srv, upstream):
    srv.fake_runner.state = "running"
    status, body = request(srv.url + "live/control", b'{"action":"say"}', GOOD)
    assert status == 200 and body == {"host": f"127.0.0.1:{upstream.server_address[1]}", "x": "1", "body": '{"action":"say"}'}
    assert request(srv.url + "live/control", b'{"action":"say"}', {"Content-Type": "application/json"})[0] == 403


def test_orphan_stop_sends_shutdown(srv, upstream):
    status, res = request(srv.url + "api/orphan/stop", b"{}", GOOD)
    assert status == 200 and res["ok"] and upstream.posts == ["/shutdown"]
    assert request(srv.url + "api/state")[1]["orphan"] is False


def test_settings_round_trip_never_leaks_secret(srv):
    status, res = request(srv.url + "api/settings", json.dumps({"values": {"secret.llm": "sk-1234567890abcd"}}).encode(), GOOD)
    assert status == 200 and res["ok"] and "restart" not in res
    status, view = request(srv.url + "api/settings")
    assert status == 200 and "1234567890" not in json.dumps(view)


def test_settings_save_while_running_says_restart(srv):
    srv.fake_runner.state = "running"
    status, res = request(srv.url + "api/settings", json.dumps({"values": {"device.serial": "x"}}).encode(), GOOD)
    assert status == 200 and res["ok"] and res["restart"] is True


def test_settings_test_uses_form_values(srv, monkeypatch):
    seen = {}
    monkeypatch.setattr("skydango.console.probes.test_llm", lambda cfg, key: seen.update(model=cfg.model, key=key) or {"ok": True, "text": "成功"})
    request(srv.url + "api/settings", json.dumps({"values": {"secret.llm": "sk-saved"}}).encode(), GOOD)
    body = json.dumps({"what": "llm", "values": {"llm.model": "m2", "secret.llm": ""}}).encode()
    assert request(srv.url + "api/settings/test", body, GOOD) == (200, {"ok": True, "text": "成功"})
    assert seen == {"model": "m2", "key": "sk-saved"}


def test_settings_test_claude(srv, monkeypatch):
    seen = {}
    monkeypatch.setattr("skydango.console.probes.test_claude",
                        lambda path, config_dir, token: seen.update(path=path, dir=config_dir, token=token) or {"ok": False, "text": "x"})
    body = json.dumps({"what": "claude", "values": {"secret.claude": "tok2", "brain.claude_path": "/opt/claude"}}).encode()
    assert request(srv.url + "api/settings/test", body, GOOD)[0] == 200
    assert seen == {"path": "/opt/claude", "dir": ".brain-claude", "token": "tok2"}
    assert request(srv.url + "api/settings/test", b'{"what": "nope"}', GOOD)[0] == 400


def test_unknown_routes_404(srv):
    assert request(srv.url + "api/nope")[0] == 404
    assert request(srv.url + "api/nope", b"{}", GOOD)[0] == 404


# ---- 静态文件（页面本身的测试在 test_console_page.py）----


def test_serves_shared_brain_trace_assets(srv):  # 沙盒页的大脑时间线和 viewer 用同一份脚本
    for name, kind, word in (("brain_trace.js", "javascript", "function mountBrainTrace"), ("brain_trace.css", "text/css", ".turn")):
        with urllib.request.urlopen(srv.url + "static/" + name, timeout=5) as r:
            assert r.status == 200 and kind in r.headers["Content-Type"] and word in r.read().decode("utf-8")
    assert request(srv.url + "static/../server.py")[0] == 404
    assert request(srv.url + "static/nope.js")[0] == 404


def test_start_refused_while_orphan_holds_port(tmp_path, upstream):  # 终审 Important 3：别起第二个团子
    s = make_server(tmp_path, upstream, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "tok"\nDEEPSEEK_API_KEY = "k"\n')
    try:
        status, res = request(s.url + "api/run/start", json.dumps(OPTS).encode(), GOOD)
        assert status == 409 and "上次留下的团子" in res["problems"][0]["text"] and s.fake_runner.started == []
        assert request(s.url + "api/state")[1]["orphan"] is True
    finally:
        s.stop()


def test_live_proxy_body_cut_off_is_503(srv):  # Review Focus 4：子进程在半路退出
    srv.fake_runner.state = "running"
    assert request(srv.url + "live/partial") == (503, {"ok": False, "text": "团子没在运行"})


def test_state_says_when_emotes_are_off_in_config(tmp_path, upstream):  # spec §2：只能关不能开
    (tmp_path / "config.toml").write_text("[emotes]\nenabled = false\n", encoding="utf-8")
    s = make_server(tmp_path, upstream)
    try:
        assert request(s.url + "api/state")[1]["emotes_allowed"] is False
    finally:
        s.stop()


def test_start_refused_while_checking_device(tmp_path, upstream):  # 检测和团子不能同时碰设备
    started, release = threading.Event(), threading.Event()

    def slow_checks(cfg, make):
        started.set()
        release.wait(5)
        return []

    s = make_server(tmp_path, upstream, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "tok"\nDEEPSEEK_API_KEY = "k"\n', checks=slow_checks, child_port=free_port())
    try:
        t = threading.Thread(target=request, args=(s.url + "api/device", b"{}", GOOD))
        t.start()
        assert started.wait(5)
        status, res = request(s.url + "api/run/start", json.dumps(OPTS).encode(), GOOD)
        assert status == 409 and "正在检测设备" in res["problems"][0]["text"] and s.fake_runner.started == []
        assert request(s.url + "api/device", b"{}", GOOD)[0] == 409
        release.set()
        t.join(5)
        assert request(s.url + "api/run/start", json.dumps(OPTS).encode(), GOOD)[0] == 200
    finally:
        release.set()
        s.stop()


def test_live_proxy_timeout_is_503(srv):
    srv.fake_runner.state = "running"
    srv.proxy_timeout = 0.3
    assert request(srv.url + "live/hang") == (503, {"ok": False, "text": "团子没在运行"})


def test_sandbox_info_has_problems(srv):  # 没有 Claude 令牌
    info = request(srv.url + "api/sandbox/info")[1]
    assert any(p["setting"] == "secret.claude" for p in info["problems"])
    assert all(set(p) == {"text", "setting"} for p in info["problems"])


def test_serves_console_static(srv):
    with urllib.request.urlopen(srv.url + "console/static/console.css", timeout=5) as r:
        assert r.status == 200 and "text/css" in r.headers["Content-Type"]
    for bad in ("console/static/console.html", "console/static/../server.py", "console/static/%2e%2e/server.py", "console/static/nope.js", "console/static/con.js", "console/static/nul.js", "console/static/aux.css"):
        assert request(srv.url + bad)[0] == 404


def test_console_static_device_names_never_touch_fs():
    from skydango.console.server import _console_static

    for n in ("con.js", "nul.js", "aux.css", "com1.js", "nope.js"):
        assert _console_static(n) is None
    assert _console_static("console.css") is not None
