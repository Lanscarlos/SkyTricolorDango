"""团子跑道的事件（brain/flow.py）：格式、环形缓冲、长轮询、大脑一轮里的决定先攒着、用量换成「看一眼」和这一轮的花费。"""
import json
import threading
import time

import pytest

from skydango.brain.flow import FlowLog


def fl(limit=300):
    t = [1_790_000_000.0]
    f = FlowLog(lambda: t[0], limit=limit)
    return f, t


def nodes(events):
    return [(e["node"], e["outcome"]) for e in events]


def test_event_format_omits_empty_fields():
    f, _ = fl()
    e = f.emit("has_new", "2 条新", "ok", value=2)
    assert e == {"id": 1, "t": 1_790_000_000.0, "node": "has_new", "label": "2 条新", "outcome": "ok", "value": 2}
    e2 = f.emit("to_me", "跟你说：叫了团子", "ok", speaker="小明")
    assert e2["id"] == 2 and e2["speaker"] == "小明" and "value" not in e2 and "model" not in e2 and "cost" not in e2
    json.dumps(e2, ensure_ascii=False)  # 能直接序列化


def test_rejects_unknown_node_and_outcome():
    f, _ = fl()
    with pytest.raises(ValueError):
        f.emit("dance", "x", "ok")
    with pytest.raises(ValueError):
        f.emit("look", "x", "maybe")


def test_ring_buffer_keeps_latest():
    f, _ = fl(limit=3)
    for i in range(5):
        f.emit("look", str(i), "ok")
    boot, v, events = f.wait_since(0, 0)
    assert v == 5 and [e["label"] for e in events] == ["2", "3", "4"]


def test_wait_since_long_poll():
    f, _ = fl()
    started = time.monotonic()
    _, v, events = f.wait_since(0, 0.2)
    assert v == 0 and events == [] and time.monotonic() - started >= 0.15
    threading.Timer(0.1, lambda: f.emit("look", "看", "ok")).start()
    _, v, events = f.wait_since(0, 3.0)
    assert v == 1 and nodes(events) == [("look", "ok")]


def test_client_ahead_starts_over():  # 网页记的比这边大：团子重启过
    f, _ = fl()
    f.emit("look", "看", "ok")
    boot, v, events = f.wait_since(50, 0)
    assert v == 1 and len(events) == 1 and boot == f.boot


def test_decisions_in_a_turn_wait_for_its_end():
    f, t = fl()
    f.begin_turn()
    f.emit("say_or_act", "说话", "ok", decision=True)
    f.emit("can_send", "能发", "ok", decision=True)
    f.emit("send", "在呢", "sent", decision=True)
    assert nodes(f.wait_since(0, 0)[2]) == [("think", "ok")]  # 只有「在想」
    t[0] += 6.2
    f.end_turn(6.2, "好")
    assert nodes(f.wait_since(0, 0)[2]) == [
        ("think", "ok"), ("think", "ok"), ("reply_or_not", "ok"), ("say_or_act", "ok"), ("can_send", "ok"), ("send", "sent")]
    assert f.wait_since(0, 0)[2][1]["label"] == "想了 6.2 秒"


def test_turn_without_speaking_is_skip_with_reason():
    f, _ = fl()
    f.begin_turn()
    f.end_turn(3.0, "看了看周围\n不说：他们在聊别的")
    last = f.wait_since(0, 0)[2][-1]
    assert (last["node"], last["outcome"], last["label"]) == ("reply_or_not", "skip", "不说：他们在聊别的")
    f.begin_turn()
    f.end_turn(1.0, "")
    assert f.wait_since(0, 0)[2][-1]["label"] == "这轮没开口"


def test_failed_turn():
    f, _ = fl()
    f.begin_turn()
    f.emit("say_or_act", "说话", "ok", decision=True)
    f.end_turn(2.0, "", error="超时")
    got = f.wait_since(0, 0)[2]
    assert nodes(got)[1:] == [("think", "skip"), ("say_or_act", "ok")]  # 失败也把攒着的放出来，不编「回不回」
    assert "超时" in got[1]["label"]


def test_decisions_outside_a_turn_go_out_at_once():  # 手动控制、备用回复
    f, _ = fl()
    f.emit("say_or_act", "说话", "ok", decision=True)
    assert nodes(f.wait_since(0, 0)[2]) == [("say_or_act", "ok")]


def test_usage_eyes_is_a_look_and_brain_cost_goes_on_the_turn():
    f, _ = fl()
    f.on_usage("eyes", "deepseek", "deepseek-flash", 0.0012, True)
    f.on_usage("memory", "deepseek", "deepseek-flash", 0.5, True)  # 不是跑道上的一步
    f.begin_turn()
    f.on_usage("brain", "deepseek", "deepseek-flash", 0.003, True)
    f.on_usage("recap", "deepseek", "deepseek-flash", 0.001, True)
    f.end_turn(4.0, "")
    got = f.wait_since(0, 0)[2]
    assert (got[0]["node"], got[0]["model"], got[0]["cost"]) == ("look", "eye", 0.0012)
    done = got[2]
    assert done["node"] == "think" and done["model"] == "brain" and done["cost"] == pytest.approx(0.004)
    assert len(got) == 4


def test_unpriced_brain_has_no_cost():  # Claude Code 走订阅：没有钱数，不编
    f, _ = fl()
    f.on_usage("eyes", "claude", "haiku", None, True)
    f.begin_turn()
    f.on_usage("brain", "claude", "sonnet", None, True)
    f.end_turn(1.0, "")
    got = f.wait_since(0, 0)[2]
    assert "cost" not in got[0] and "cost" not in got[2]


# ---- 大脑一轮、用量监听 ----
def test_brain_wake_opens_and_closes_a_turn(clock):
    from test_brain_loop import FakeSession, make, ok

    brain, _, _, _ = make(clock, FakeSession(ok("不说：在忙")))
    f, _ = fl()
    brain.flow = f
    brain.wake(clock(), "heartbeat")
    got = f.wait_since(0, 0)[2]
    assert nodes(got) == [("think", "ok"), ("think", "ok"), ("reply_or_not", "skip")]
    assert got[-1]["label"] == "不说：在忙"


def test_brain_failed_turn_closes_with_error(clock):
    from test_brain_loop import FakeSession, make

    from skydango.brain.claude import ClaudeError

    brain, _, _, _ = make(clock, FakeSession(ClaudeError("超时")))
    f, _ = fl()
    brain.flow = f
    brain.wake(clock(), "events")
    got = f.wait_since(0, 0)[2]
    assert nodes(got) == [("think", "ok"), ("think", "skip")] and "超时" in got[-1]["label"]


def test_meter_listeners_get_each_call_with_cost():
    from test_models_usage import meter

    m = meter()
    calls = []
    m.listeners.append(lambda *a: calls.append(a))
    m.listeners.append(lambda *a: 1 / 0)  # 监听出错不影响记账
    m.record("eyes", "deepseek", "deepseek-flash", backup=False, usage={"input_tokens": 1000, "output_tokens": 100}, ok=True)
    m.record("brain", "claude", "sonnet", backup=False, usage=None, ok=True)
    (use, provider, model, cost, okk), (use2, _, _, cost2, _) = calls
    assert (use, provider, model, okk) == ("eyes", "deepseek", "deepseek-flash", True) and cost > 0
    assert use2 == "brain" and cost2 is None
    assert m.snapshot()["run"]["rows"]
