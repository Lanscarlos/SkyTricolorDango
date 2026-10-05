"""身体接上「好友在跟谁说话」的判断（spec 2026-10-05-addressee §4）：事件、反射、账本、接话 / 插话、status、聊天记录标注。"""

import logging
from types import SimpleNamespace

import pytest
from test_brain_body import FakeEnv, body, msg

from skydango.brain import addressee as addressee_mod
from skydango.brain.body import ToolError

FRIENDS_MD = "## 小明\n- 叫法：明哥\n\n## 阿花\n"


def ab(clock, live=True, proactive=False, **kw):
    env = FakeEnv()
    env.near = ["小明", "阿花"]
    b, dev, reader, events = body(clock, live=live, addressee=True, env=env, **kw)
    b.friend_names = lambda: ["小明", "阿花"]
    b.friends_text = lambda: FRIENDS_MD
    b.cfg.proactive.enabled = proactive
    b.step()  # 好友来了（arrive）：先消化掉
    events.drain()
    return b, env, reader, events


def say_batch(b, reader, *items):
    """items：(说话人, 内容)；一批读进来。"""
    reader.batches = [[msg(text, speaker=who) for who, text in items]]
    b.step()


def quota(left, blocked=""):
    return lambda: SimpleNamespace(left=left, blocked=blocked, line=lambda w: "场合：测试")


def test_aside_event_with_tag(clock):
    b, _, reader, events = ab(clock, proactive=True)
    b.occasion = quota(3)
    say_batch(b, reader, ("小明", "阿花你看"))
    ev = events.drain()
    assert [e.kind for e in ev] == ["aside"]
    assert ev[0].text == "聊天  小明：「阿花你看」（跟别人说：叫了阿花）"


def test_chat_event_with_tag(clock):
    b, _, reader, events = ab(clock)
    say_batch(b, reader, ("小明", "团子你说呢"))
    ev = events.drain()
    assert [e.kind for e in ev] == ["chat"]
    assert ev[0].text.endswith("（跟你说：叫了你）")


def test_aside_bg_when_quota_used(clock):
    b, _, reader, events = ab(clock, proactive=True)
    b.occasion = quota(0, "额度用完")
    say_batch(b, reader, ("小明", "阿花你看"))
    assert [e.kind for e in events.drain()] == ["aside_bg"]
    say_batch(b, reader, ("小明", "阿花快来"))
    assert events.urgent() is False


def test_aside_bg_when_proactive_off(clock):
    b, _, reader, events = ab(clock, proactive=False)
    say_batch(b, reader, ("小明", "阿花你看"))
    assert [e.kind for e in events.drain()] == ["aside_bg"]


def test_judge_log_line(clock, caplog):
    b, _, reader, _ = ab(clock)
    caplog.set_level(logging.INFO, logger="skydango.brain.body")
    say_batch(b, reader, ("小明", "阿花你看"))
    assert "跟谁说 小明「阿花你看」→ 跟别人说（叫了阿花）· 身边：小明、阿花" in caplog.text


def test_no_bubble_for_aside(clock):
    b, _, reader, _ = ab(clock)
    say_batch(b, reader, ("小明", "阿花你看"))
    assert not b.sender.opened
    say_batch(b, reader, ("小明", "团子在吗"))
    assert b.sender.opened


class FakeLedger:
    def __init__(self):
        self.heard_calls = []

    def heard(self, speaker, text, to_me, wall):
        self.heard_calls.append((text, to_me))

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_ledger_to_me_only_me(clock):
    b, _, reader, _ = ab(clock)
    b.ledger = FakeLedger()
    say_batch(b, reader, ("小明", "阿花你看"), ("小明", "团子在吗"))
    assert b.ledger.heard_calls == [("阿花你看", False), ("团子在吗", True)]


def test_pending_reply_skips_aside(clock):
    b, _, reader, _ = ab(clock, proactive=True)
    say_batch(b, reader, ("小明", "团子在吗"))
    say_batch(b, reader, ("阿花", "明哥你看"))
    assert b._pending_reply(b.wall()) is True
    b2, _, reader2, _ = ab(clock, proactive=True)
    say_batch(b2, reader2, ("阿花", "明哥你看"))
    assert b2._pending_reply(b2.wall()) is False


def test_aside_turn_say_is_proactive(clock):
    b, _, reader, _ = ab(clock, proactive=True)
    b.occasion = quota(0, "额度用完")
    say_batch(b, reader, ("小明", "阿花你看"))
    with pytest.raises(ToolError, match="额度用完"):
        b.say("我也看看")
    clock.advance(30)  # 过了 thread_window：下一句拿不准，算有话待接，不是主动开口
    say_batch(b, reader, ("小明", "嗯"))
    assert b.verdicts[-1][3].label == "unsure"
    assert "已发送" in b.say("怎么啦")


def test_blocked_say_not_followup(clock):
    b, _, reader, _ = ab(clock, proactive=True)
    b.occasion = quota(0, "额度用完")
    say_batch(b, reader, ("小明", "阿花你看"))
    with pytest.raises(ToolError):
        b.say("我也看看")
    clock.advance(1)
    say_batch(b, reader, ("小明", "哈哈"))
    assert b.verdicts[-1][3].reason != "在接你的话"
    # 说成功之后才算
    b.cfg.proactive.enabled = False
    b.say("好")
    clock.advance(1)
    say_batch(b, reader, ("小明", "哈哈哈"))
    assert b.verdicts[-1][3].reason == "在接你的话"


def test_owner_command_not_judged(clock):
    b, _, reader, events = ab(clock)
    b.cfg.brain.owner_name = "小明"
    say_batch(b, reader, ("小明", "#过来"), ("小明", "阿花你看"))
    assert [e.kind for e in events.drain()] == ["owner_command", "aside_bg"]
    assert not b.sender.opened


def test_status_thread_note(clock):
    b, _, reader, _ = ab(clock)
    for who, text in [("小明", "阿花你看"), ("阿花", "明哥怎么了"), ("小明", "阿花这个好玩"), ("阿花", "明哥我看到了")]:
        say_batch(b, reader, (who, text))
        clock.advance(2)
    assert "小明 和 阿花 在聊" in b.status()


def test_panel_busy_on_aside(clock):
    b, _, reader, events = ab(clock)
    say_batch(b, reader, ("小明", "阿花你看"))
    calls = []
    b.panel.busy = lambda now: calls.append(now)
    b.step()
    assert calls and events.has("aside_bg")


def test_heard_line_why_is_tag(clock):
    b, _, reader, _ = ab(clock)
    lines = []
    b.on_line = lambda kind, text, who, why="": lines.append((kind, text, who, why))
    say_batch(b, reader, ("小明", "阿花你看"))
    assert ("heard", "阿花你看", "小明", "跟别人说：叫了阿花") in lines


def test_disabled_is_unchanged(clock, caplog):
    env = FakeEnv()
    env.near = ["小明", "阿花"]
    b, _, reader, events = body(clock, live=True, env=env)
    b.friend_names = lambda: ["小明", "阿花"]
    b.step()
    events.drain()
    lines = []
    b.on_line = lambda kind, text, who, why="": lines.append((kind, why))
    caplog.set_level(logging.INFO, logger="skydango.brain.body")
    say_batch(b, reader, ("小明", "阿花你看"))
    ev = events.drain()
    assert [e.kind for e in ev] == ["chat"] and ev[0].text == "聊天  小明：「阿花你看」"
    assert "跟谁说" not in caplog.text
    assert ("heard", "") in lines


def test_judge_error_is_unsure(clock, monkeypatch):
    b, _, reader, events = ab(clock)

    def boom(*a, **k):
        raise RuntimeError("坏了")

    monkeypatch.setattr(addressee_mod.Addressee, "judge", boom)
    say_batch(b, reader, ("小明", "阿花你看"))
    ev = events.drain()
    assert ev[0].kind == "chat" and ev[0].text.endswith("（拿不准：判断出错）")
