"""身体把听到 / 说 / 做 / 被拦 / 心里想的交给 on_line（真机聊天记录，spec 2026-10-01-console-live-page §3.2）；事件队列的 tap。"""
import pytest
from test_brain_body import FakeEmotes, body, msg
from test_brain_lull_body import chat_then_quiet, lb
from test_brain_reflex_body import OPEN, rx

from skydango.brain.body import ToolError
from skydango.brain.events import EventQueue


def collect(b):
    got = []
    b.on_line = lambda kind, text, who, why: got.append((kind, text, who, why))
    return got


def test_heard_lines(clock):
    b, _, reader, _ = body(clock)
    got = collect(b)
    reader.batches = [[msg("在吗"), msg("hi", speaker="")]]
    b.step()
    assert got == [("heard", "在吗", "懒洋洋大王", ""), ("heard", "hi", "（看不出是谁）", "")]


def test_said_live_and_dry_run(clock):
    b, _, _, _ = body(clock, live=True)
    got = collect(b)
    b.say("在呢")
    assert got == [("said", "在呢", "团子", "")]
    b2, _, _, _ = body(clock)  # 默认 dry-run
    got2 = collect(b2)
    b2.say("在呢")
    assert got2 == [("said", "在呢（dry-run，没真的发）", "团子", "")]


def test_blocked_line(clock):
    b, _, _, _ = body(clock, live=True)
    got = collect(b)
    with pytest.raises(ToolError):
        b.say("我是真人")
    assert got[0][0] == "blocked" and got[0][1] == "我是真人" and got[0][2] == "团子" and got[0][3]


def test_emote_line(clock):
    b, _, _, _ = body(clock, live=True, emotes=FakeEmotes())
    got = collect(b)
    b.emote("鞠躬")
    assert got == [("act", "（团子做了 鞠躬）", "团子", "")]


def test_reflex_and_bubble_lines(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"])
    got = collect(b)
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    kinds = [(k, t) for k, t, _, _ in got]
    assert ("heard", "团子") in kinds
    assert ("act", "（团子下意识地 点头）") in kinds and ("act", "（团子头顶冒出输入气泡）") in kinds
    assert dev.calls.count(OPEN) == 1


def test_musing_line(clock):
    b, env, reader, events = lb(clock)
    got = collect(b)
    chat_then_quiet(b, reader, events, clock)
    got.clear()
    b.mused("心里：他是不是去忙了")
    assert got == [("event", "── 心里：他是不是去忙了 ──", "", "")]


def test_on_line_errors_do_not_break_body(clock):
    b, _, reader, events = body(clock)
    b.on_line = lambda *a: 1 / 0
    reader.batches = [[msg("在吗")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]


def test_tap_sees_every_put(clock):
    q = EventQueue(clock=clock)
    seen = []
    q.tap(lambda kind, text, who: seen.append((kind, text, who)))
    q.put("arrive", "小明 来到身边", who="小明")
    q.put("leave", "小明 走开了", who="小明")  # 和 arrive 不抵消，照样调
    q.put("scene_change", "画面恢复了")
    q.put("scene_change", "画面恢复了")  # 合并了也调
    assert seen == [("arrive", "小明 来到身边", "小明"), ("leave", "小明 走开了", "小明"),
                    ("scene_change", "画面恢复了", ""), ("scene_change", "画面恢复了", "")]


def test_tap_errors_are_swallowed(clock):
    q = EventQueue(clock=clock)
    q.tap(lambda *a: 1 / 0)
    q.put("chat", "x")
    assert len(q) == 1
