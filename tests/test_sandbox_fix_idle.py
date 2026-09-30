"""大脑沙盒审查意见的回归测试：idle 和回放：idle 变了叫醒长轮询、额度用完、回放中途下线（审查 5 / 9）。"""


from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from test_brain_tools import FakeBody
from test_console_sandbox import make, post
from test_console_server import no_registry, request, upstream  # noqa: F401

from skydango.console.sandbox_view import reset
from skydango.sandbox.clock import save


# ---- 5. idle：变了就叫醒 /state 的等待者；额度用完时不看事件队列 ----
def test_state_returns_when_idle_flips(tmp_path):
    from test_sandbox_control import build

    s = build(tmp_path, start=False)
    c = s.control
    s.mono.t = 0.0
    assert c.idle() is False
    v = s.transcript.version
    out = {}

    def later():
        s.mono.t = 5.0  # 安静够了

    threading.Timer(0.3, later).start()
    started = time.monotonic()
    out = c.state(after=v, timeout=5.0)
    assert time.monotonic() - started < 2.5 and out["idle"] is True and out["lines"] == []


def test_idle_ignores_event_queue_while_quota_is_out(tmp_path):
    from test_sandbox_control import build

    s = build(tmp_path, start=False)
    c = s.control
    s.events.put("chat", "小明：在吗")
    assert c.idle() is False
    s.brain.left = 300  # 额度用完：大脑醒不来，事件会一直积着
    c.idle()
    s.mono.t += 3
    assert c.idle() is True


def test_replay_notes_quota(tmp_path):
    from test_replay import FakeApi, replayer, scenario

    from skydango.console.scenario import Start, Step

    api = FakeApi()
    real_state = api.state

    def state(after):
        st = real_state(after)
        if st is not None:
            st["limit"] = "额度用完，约 9 分钟后再试"
        return st

    api.state = state
    text = replayer(api, scenario(Step("say", "在吗", who="小明"), start=Start(memory="keep")), tmp_path)[0].run().read_text("utf-8")
    assert "额度用完，约 9 分钟后再试" in text.split("第 1 步")[1]


# ---- 9. 回放中途手动下线：回放线程不等满 step_timeout ----
def test_replay_stops_quickly_when_sandbox_is_taken_down(tmp_path):
    from test_replay import FakeApi, replayer, scenario

    from skydango.console.scenario import Start, Step

    api = FakeApi()
    sc = scenario(*(Step("say", str(i), who="小明") for i in range(1, 4)), start=Start(memory="keep"))
    r, clock = replayer(api, sc, tmp_path, step_timeout=180.0)

    def take_down():
        if any(c[0] == "op" and c[1].get("text") == "2" for c in api.calls) and api.is_running:
            api.never_idle = True  # 第 2 步发出去后一直不安静
            r.stop_requested()  # 页面点了「下线」：请求停止回放 + 停掉子进程
            api.is_running = False

    api.on_state = take_down
    text = r.run().read_text(encoding="utf-8")
    assert clock.t < 60  # 没等满 180 秒
    assert [o["text"] for o in [c[1] for c in api.calls if c[0] == "op"]] == ["1", "2"]
    assert "回放被停止，沙盒已下线" in text and r.progress()["running"] is False
