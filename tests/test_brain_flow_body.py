"""身体往团子跑道（brain/flow.py）发事件：读到新消息、跟谁说、攒话、说话 / 做动作、被护栏拦下。"""
import pytest
from test_brain_body import FakeEmotes, body, msg
from test_brain_pacing_body import pace, say

from skydango.brain.body import ToolError
from skydango.brain.flow import FlowLog


def flow(b):
    b.flow = FlowLog(lambda: 1_790_000_000.0)
    return b.flow


def got(f):
    return [(e["node"], e["outcome"], e["label"]) for e in f.wait_since(0, 0)[2]]


def test_heard_has_new_read_and_to_me(clock):
    b, _, reader, _ = body(clock, addressee=True)
    f = flow(b)
    b.friend_names = lambda: ["懒洋洋大王"]
    reader.batches = [[msg("团子在吗"), msg("hi", speaker="")]]
    b.step()
    rows = f.wait_since(0, 0)[2]
    assert [(e["node"], e["outcome"]) for e in rows[:2]] == [("has_new", "ok"), ("read", "ok")]
    assert rows[0]["label"] == "2 条新" and rows[0]["value"] == 2
    to_me = [e for e in rows if e["node"] == "to_me"]
    assert [(e["outcome"], e.get("speaker")) for e in to_me] == [("ok", "懒洋洋大王"), ("unsure", None)]
    assert to_me[0]["label"].startswith("跟你说") and to_me[1]["label"].startswith("拿不准")


def test_no_new_messages_no_events(clock):  # 没新消息不发：团子停着呼吸
    b, _, _, _ = body(clock)
    f = flow(b)
    b.step()
    assert not [e for e in f.wait_since(0, 0)[2] if e["node"] in ("has_new", "read", "to_me")]


def test_other_is_skip(clock):
    from test_brain_addressee_body import ab
    b, env, reader, events = ab(clock)
    f = flow(b)
    reader.batches = [[msg("阿花你在哪", speaker="小明")]]
    b.step()
    (e,) = [e for e in f.wait_since(0, 0)[2] if e["node"] == "to_me"]
    assert e["outcome"] == "skip" and e["label"].startswith("跟别人说")


def test_batch_hold_and_release(clock):
    b, dev, reader, events = pace(clock)
    f = flow(b)
    say(reader, "哈哈", "嘻嘻")
    b.step()
    assert ("batch", "ok", "攒着 2 句") in got(f)
    clock.advance(3.0)
    b.step()
    assert got(f)[-1] == ("batch", "ok", "攒了 2 句 · 等了 3 秒")


def test_say_sent_and_dry_run(clock):
    b, _, _, _ = body(clock, live=True)
    f = flow(b)
    b.say("在呢")
    assert got(f) == [("say_or_act", "ok", "说话「在呢」"), ("can_send", "ok", "能发"), ("send", "sent", "在呢")]
    b2, _, _, _ = body(clock)
    f2 = flow(b2)
    b2.say("在呢")
    assert got(f2)[-1] == ("send", "sent", "dry-run 没真发：在呢")


def test_say_blocked_is_ask_with_reason(clock):
    b, _, _, _ = body(clock, live=True)
    f = flow(b)
    with pytest.raises(ToolError):
        b.say("我是真人")
    last = got(f)[-1]
    assert last[:2] == ("can_send", "ask") and "真人" in last[2]


def test_emote_done_and_refused(clock):
    b, _, _, _ = body(clock, live=True, emotes=FakeEmotes())
    f = flow(b)
    b.emote("鞠躬")
    assert got(f) == [("say_or_act", "ok", "动作「鞠躬」"), ("can_send", "ok", "能做"), ("send", "sent", "做了「鞠躬」")]
    with pytest.raises(ToolError):
        b.emote("跳舞")
    assert got(f)[-1][:2] == ("can_send", "ask")


def test_decisions_wait_for_the_brain_turn(clock):
    b, _, _, _ = body(clock, live=True)
    f = flow(b)
    f.begin_turn()
    b.say("在呢")
    assert [n for n, _, _ in got(f)] == ["think"]
    f.end_turn(2.0, "")
    assert [n for n, _, _ in got(f)] == ["think", "think", "reply_or_not", "say_or_act", "can_send", "send"]


def test_flow_errors_do_not_break_body(clock):
    b, _, reader, events = body(clock, live=True)
    b.flow = type("Bad", (), {"emit": lambda *a, **k: 1 / 0})()
    reader.batches = [[msg("在吗")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
    assert b.say("在呢").startswith("已发送")


def test_blocked_view_is_ask(clock):  # 面板挡着（clear_view 拒绝）：也是「不能发」，不能停在「选一选」上没下文
    b, _, _, _ = body(clock, live=True, emotes=FakeEmotes())
    f = flow(b)

    def refuse(*a, **k):
        raise ToolError("有面板挡着")

    b.clear_view = refuse
    with pytest.raises(ToolError):
        b.say("在呢")
    with pytest.raises(ToolError):
        b.emote("鞠躬")
    assert [(n, o) for n, o, _ in got(f)] == [("say_or_act", "ok"), ("can_send", "ask"), ("say_or_act", "ok"), ("can_send", "ask")]
    assert got(f)[1][2] == "不能发：有面板挡着"
