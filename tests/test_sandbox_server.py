"""沙盒子进程的 HTTP 接口（brain-sandbox 计划 Task 5）：路由、还没起好时的 503、安全校验。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from skydango.brain.trace import BrainTrace
from skydango.sandbox.server import SandboxServer

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(url, method="GET", body=None, headers=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = {"X-Skydango": "1", "Content-Type": "application/json"} if method == "POST" else {}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with NO_PROXY.open(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        text = e.read()
        try:
            return e.code, json.loads(text)
        except ValueError:
            return e.code, text


class FakeControl:
    def __init__(self):
        self.ops = []
        self.states = []

    def op(self, req):
        if req.get("op") == "飞":
            raise ValueError("不认识的操作：'飞'")
        self.ops.append(req)
        return {"ok": True, "text": "好"}

    def state(self, after, timeout):
        self.states.append((after, timeout))
        return {"version": 3, "lines": [], "idle": True}


@pytest.fixture
def srv():
    calls = []
    s = SandboxServer(0, on_shutdown=lambda: calls.append("bye"))
    url = s.start()
    s.calls = calls
    s.base = url
    yield s
    s.stop()


def test_status_before_ready(srv):
    assert call(srv.base + "status") == (200, {"ok": True, "kind": "sandbox", "ready": False})


def test_state_and_op_503_until_ready(srv):
    code, body = call(srv.base + "state?after=0")
    assert code == 503 and body == {"ok": False, "text": "沙盒还在启动"}
    code, body = call(srv.base + "op", "POST", {"op": "say", "who": "小明", "text": "在吗"})
    assert code == 503 and body["text"] == "沙盒还在启动"


def test_state_and_op_routed(srv):
    srv.control = FakeControl()
    assert call(srv.base + "status")[1]["ready"] is True
    code, body = call(srv.base + "state?after=2&wait=0.5")
    assert code == 200 and body["version"] == 3 and srv.control.states == [(2, 0.5)]
    call(srv.base + "state?after=2&wait=999")
    assert srv.control.states[-1] == (2, 25.0)  # 最多等 25 秒
    assert call(srv.base + "op", "POST", {"op": "say", "who": "小明", "text": "在吗"}) == (200, {"ok": True, "text": "好"})
    code, body = call(srv.base + "op", "POST", {"op": "飞"})
    assert code == 400 and "飞" in body["text"]
    code, body = call(srv.base + "op", "POST", raw=b"{bad json")
    assert code == 400


def test_security(srv):
    srv.control = FakeControl()
    port = srv.base.rstrip("/").rsplit(":", 1)[1]
    assert call(srv.base + "state?after=0", headers={"Host": f"evil.example:{port}"})[0] == 403
    assert call(srv.base + "status", headers={"Host": f"evil.example:{port}"})[0] == 403
    assert call(srv.base + "op", "POST", {"op": "say"}, headers={"X-Skydango": ""})[0] == 403
    assert call(srv.base + "op", "POST", {"op": "say"}, headers={"Content-Type": "text/plain"})[0] == 403
    assert call(srv.base + "op", "POST", raw=b"{" + b" " * 70000 + b"}")[0] == 413
    assert srv.control.ops == []


def test_shutdown(srv):
    assert call(srv.base + "shutdown", "POST", {})[0] == 200
    assert srv.calls == ["bye"]


def test_brain_and_inner(srv):
    assert call(srv.base + "brain?after=0")[0] == 503
    assert call(srv.base + "inner")[0] == 503
    srv.trace = BrainTrace()
    code, body = call(srv.base + "brain?after=0&wait=0")
    assert code == 200 and body["turns"] == []
    srv.inner = lambda: {"running": True, "mood": None}
    assert call(srv.base + "inner") == (200, {"running": True, "mood": None})
    seen = []
    srv.forget = lambda k, t, w, tp: seen.append((k, t, w, tp)) or ""
    assert call(srv.base + "inner/forget", "POST", {"kind": "catchphrase", "text": "害"}) == (200, {"ok": True})
    assert seen == [("catchphrase", "害", "", "")]
    assert call(srv.base + "inner/forget", "POST", {"kind": "飞"})[0] == 400


def test_unknown_paths(srv):
    assert call(srv.base + "nope")[0] == 404
    assert call(srv.base + "nope", "POST", {})[0] == 404


def test_usage_routed(srv):   # spec 2026-10-06-model-usage §6.1
    assert call(srv.base + "usage") == (503, {"ok": False, "text": "沙盒还在启动"})
    srv.usage = lambda: {"source": "sandbox"}
    assert call(srv.base + "usage") == (200, {"ok": True, "source": "sandbox"})
