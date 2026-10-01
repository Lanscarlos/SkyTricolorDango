"""管理面板的沙盒部分（brain-sandbox 计划 Task 6）：起停和团子互斥、/sandbox/* 转发、重置记忆、内心页读沙盒。"""

from __future__ import annotations

import json
import time

import pytest
from test_console_server import GOOD, OPTS, FakeRunner, free_port, make_server, no_registry, request, upstream  # noqa: F401

from skydango.console.sandbox_view import friends, reset, start_info
from skydango.inner.persona import Persona, Trait
from skydango.inner.store import InnerStore
from skydango.sandbox.clock import save


class KindRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.kind, self.port = "dango", None

    def status(self):
        return {**super().status(), "kind": self.kind, "port": self.port}

    def start(self, cmd, env, options, kind="dango", port=None):
        self.started.append((cmd, env, options, kind, port))
        self.state, self.kind, self.port = "starting", kind, port


def make(tmp_path, upstream, sandbox_port=None):
    port = sandbox_port or free_port()
    (tmp_path / "config.toml").write_text(
        f'[sandbox]\ndir = "{(tmp_path / "sandbox").as_posix()}"\nport = {port}\n'
        f'[reply]\nmemory_dir = "{(tmp_path / "memory").as_posix()}"\n'
        '[brain]\nowner_name = "卡洛"\n', encoding="utf-8")
    srv = make_server(tmp_path, upstream, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "tok"\nDEEPSEEK_API_KEY = "k"\n', child_port=free_port())
    srv.runner = srv.fake_runner = KindRunner()
    srv.sandbox_port_value = port
    return srv


@pytest.fixture
def srv(tmp_path, upstream):
    s = make(tmp_path, upstream)
    yield s
    s.stop()


def post(srv, path, body):
    return request(srv.url + path, json.dumps(body, ensure_ascii=False).encode(), GOOD)


# ---- 起停互斥（Review Focus 3）----
def test_sandbox_start_refused_while_dango_runs(srv):
    srv.runner.state = "running"
    status, res = post(srv, "api/sandbox/start", {"start": "resume"})
    assert status == 409 and any("先停团子" in p["text"] for p in res["problems"])
    assert srv.runner.started == []


def test_dango_start_refused_while_sandbox_runs(srv):
    srv.runner.state, srv.runner.kind = "running", "sandbox"
    status, res = post(srv, "api/run/start", OPTS)
    assert status == 409 and any("先在「沙盒」页下线沙盒" in p["text"] for p in res["problems"])
    assert not any("团子已经在运行" in p["text"] for p in res["problems"])
    assert srv.runner.started == []
    assert request(srv.url + "api/state")[1]["problems"][0]["text"].startswith("沙盒在运行")
    assert post(srv, "api/run/stop", {}) == (200, {"ok": True}) and srv.runner.stopped == 0  # 真机的停止不动沙盒


def test_sandbox_start_builds_command(srv):
    status, res = post(srv, "api/sandbox/start", {"start": "2026-10-01 09:00"})
    assert status == 200 and res == {"ok": True}
    cmd, env, options, kind, port = srv.runner.started[0]
    assert kind == "sandbox" and port == srv.sandbox_port_value and options == {"start": "2026-10-01 09:00"}
    assert cmd[cmd.index("sandbox") + 1:] == ["--port", str(port), "--no-browser", "--parent-pid", str(srv.parent_pid),
                                              "--start", "2026-10-01 09:00"]
    assert env["SKYDANGO_CLAUDE_TOKEN"] == "tok"


@pytest.mark.parametrize("start", ["明天", "25:00", 9, None])
def test_sandbox_start_rejects_bad_choice(srv, start):
    status, res = post(srv, "api/sandbox/start", {"start": start})
    assert status == 400 and srv.runner.started == []


def test_sandbox_start_refused_with_orphan(tmp_path, upstream):
    s = make(tmp_path, upstream, sandbox_port=upstream.server_address[1])  # 假上游占着沙盒端口
    try:
        status, res = post(s, "api/sandbox/start", {"start": "resume"})
        assert status == 409 and res["orphan"] is True and s.runner.started == []
        info = request(s.url + "api/sandbox/info")[1]  # 沙盒页停着时也要能给「让它退出」（有问题时启动按钮是灰的）
        assert info["orphan"] is True and any("上次留下的沙盒" in p["text"] for p in info["problems"])
        assert post(s, "api/orphan/stop", {"kind": "sandbox"}) == (200, {"ok": True})
        assert upstream.posts[-1] == "/shutdown"
    finally:
        s.stop()


def test_sandbox_stop_only_stops_sandbox(srv):
    assert post(srv, "api/sandbox/stop", {})[0] == 409
    srv.runner.state, srv.runner.kind = "running", "sandbox"
    assert post(srv, "api/sandbox/stop", {}) == (200, {"ok": True}) and srv.runner.stopped == 1


# ---- 转发 ----
def test_proxy_follows_kind(tmp_path, upstream):
    s = make(tmp_path, upstream, sandbox_port=None)
    try:
        s.runner.state, s.runner.kind, s.runner.port = "running", "sandbox", upstream.server_address[1]
        assert request(s.url + "live/status")[0] == 503  # 槽被沙盒占着：/live/* 不转
        assert request(s.url + "sandbox/status") == (200, {"path": "/status", "query": ""})
        assert request(s.url + "sandbox/state?after=3") == (200, {"path": "/state", "query": "after=3"})
        assert request(s.url + "sandbox/shutdown")[0] == 404  # 只转白名单
        status, body = post(s, "sandbox/op", {"op": "say", "who": "小明", "text": "在吗"})
        assert status == 200 and json.loads(body["body"]) == {"op": "say", "who": "小明", "text": "在吗"} and body["x"] == "1"
        assert post(s, "sandbox/shutdown", {})[0] == 404
        s.runner.kind = "dango"
        assert request(s.url + "sandbox/status") == (503, {"ok": False, "text": "沙盒没在运行"})
        assert request(s.url + "sandbox/status", headers={"Host": f"evil.com:{s.port}"})[0] == 403
    finally:
        s.stop()


# ---- 重置 ----
def test_reset_copies_memory_except_archive(tmp_path):
    memory, sandbox = tmp_path / "memory", tmp_path / "sandbox"
    (memory / "archive").mkdir(parents=True)
    (memory / "archive" / "x.jsonl").write_text("旧的", encoding="utf-8")
    (memory / "inner").mkdir()
    (memory / "friends.md").write_text("## 小明\n", encoding="utf-8")
    (memory / "inner" / "days.jsonl").write_text('{"start": 1, "end": 2}\n', encoding="utf-8")
    (sandbox / "memory").mkdir(parents=True)
    (sandbox / "memory" / "多余.md").write_text("沙盒里聊出来的", encoding="utf-8")
    save(sandbox / "clock.json", time.time())
    reset(sandbox, memory)
    assert not (sandbox / "memory" / "archive").exists() and not (sandbox / "memory" / "多余.md").exists()
    assert (sandbox / "memory" / "friends.md").read_text(encoding="utf-8") == "## 小明\n"
    assert (sandbox / "memory" / "inner" / "days.jsonl").read_text(encoding="utf-8") == '{"start": 1, "end": 2}\n'
    assert not (sandbox / "clock.json").exists()
    assert (memory / "archive" / "x.jsonl").exists()  # 真的 memory/ 原样
    assert friends(sandbox) == ["小明"]


def test_reset_without_memory_is_empty(tmp_path):
    reset(tmp_path / "sandbox", tmp_path / "没有")
    assert (tmp_path / "sandbox" / "memory").is_dir() and not list((tmp_path / "sandbox" / "memory").iterdir())


def test_reset_api(srv, tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "friends.md").write_text("## 阿花\n", encoding="utf-8")
    srv.runner.state, srv.runner.kind = "running", "sandbox"
    assert post(srv, "api/sandbox/reset", {})[0] == 409
    srv.runner.state = "idle"
    assert post(srv, "api/sandbox/reset", {})[0] == 200
    info = request(srv.url + "api/sandbox/info")[1]
    assert info["friends"] == ["阿花"] and info["owner_name"] == "卡洛" and info["floor_text"] == ""


def test_start_info_floor_text(tmp_path):
    later = time.mktime((2030, 9, 30, 22, 0, 0, 0, 0, -1))
    save(tmp_path / "clock.json", later)
    info = start_info(tmp_path, time.time())
    assert info["floor"] == later and info["saved"] == later and info["floor_text"] == "9月30日 22:00"


# ---- 内心页读沙盒 ----
def test_inner_source_sandbox_reads_sandbox_dir(srv, tmp_path):
    InnerStore(tmp_path / "memory" / "inner").write_persona(Persona(catchphrases=[Trait("真的团子", since=time.time())]))
    InnerStore(tmp_path / "sandbox" / "memory" / "inner").write_persona(Persona(catchphrases=[Trait("沙盒团子", since=time.time())]))
    real = request(srv.url + "api/inner")[1]
    sand = request(srv.url + "api/inner?source=sandbox")[1]
    assert real["persona"]["catchphrases"][0]["text"] == "真的团子"
    assert sand["persona"]["catchphrases"][0]["text"] == "沙盒团子" and sand["source"] == "files"


def test_forget_source_sandbox_changes_sandbox_file(srv, tmp_path):
    InnerStore(tmp_path / "memory" / "inner").write_persona(Persona(catchphrases=[Trait("害", since=time.time())]))
    InnerStore(tmp_path / "sandbox" / "memory" / "inner").write_persona(Persona(catchphrases=[Trait("害", since=time.time())]))
    assert post(srv, "api/inner/forget", {"kind": "catchphrase", "text": "害", "source": "sandbox"}) == (200, {"ok": True})
    sand = json.loads((tmp_path / "sandbox" / "memory" / "inner" / "persona.json").read_text("utf-8"))
    real = json.loads((tmp_path / "memory" / "inner" / "persona.json").read_text("utf-8"))
    assert sand["catchphrases"] == [] and real["catchphrases"] != []


def test_inner_dango_while_sandbox_runs_reads_files(srv):
    srv.runner.state, srv.runner.kind = "running", "sandbox"
    assert request(srv.url + "api/inner")[1]["source"] == "files"  # 团子没在跑：不去 /live/inner


# ---- 报告接口（console-redesign Task 3）----
def test_reports_routes(srv, tmp_path):
    reports = tmp_path / "sandbox" / "reports"
    reports.mkdir(parents=True)
    (reports / "r-20261001-000000.md").write_text("hi", encoding="utf-8")
    (tmp_path / "sandbox" / "x.md").write_text("secret", encoding="utf-8")
    assert request(srv.url + "api/sandbox/reports")[1]["reports"][0]["name"] == "r-20261001-000000.md"
    assert request(srv.url + "api/sandbox/reports/r-20261001-000000.md")[1]["text"] == "hi"
    assert request(srv.url + "api/sandbox/reports/..%2fx.md")[0] == 404
    assert request(srv.url + "api/sandbox/reports", headers={"Host": f"evil.com:{srv.port}"})[0] == 403
