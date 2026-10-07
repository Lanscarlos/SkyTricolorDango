"""身体攒话（spec 2026-10-07-chat-pacing §1）：攒几句再叫醒大脑、放行时才冒输入气泡、status 里的「在攒话」。"""

from test_brain_body import FakeEnv, body, msg
from test_brain_reflex import Rng

OPEN = ("hw_key", 28)


def pace(clock, pacing=True, live=True, typing=None, **pacing_cfg):
    from conftest import FakeDevice, scene

    dev = FakeDevice([scene()])
    env = FakeEnv()
    env.near = ["小明"]
    b, dev, reader, events = body(clock, live=live, env=env, rng=Rng(0.99), device_override=dev, pacing=pacing)
    b.cfg.pacing.quiet_min = b.cfg.pacing.quiet_max = 3.0
    for k, v in pacing_cfg.items():
        setattr(b.cfg.pacing, k, v)
    b.friend_names = lambda: ["小明"]
    if typing is not None:
        reader.typing = typing
    b.step()  # 小明来了（arrive）：先消化掉
    events.drain()
    return b, dev, reader, events


def say(reader, *texts, who="小明"):
    reader.batches = [[msg(t, speaker=who) for t in texts]]


def test_chat_held_until_quiet(clock):
    b, dev, reader, events = pace(clock)
    say(reader, "哈哈")
    b.step()
    assert events.held() and events.has("chat")
    clock.advance(2.9)
    b.step()
    assert events.held()
    clock.advance(0.1)
    b.step()
    assert not events.held()


def test_bubble_opens_at_release(clock):
    b, dev, reader, events = pace(clock)
    say(reader, "团子")  # 身边只有小明：在跟团子说
    b.step()
    assert not b.sender.opened and OPEN not in dev.calls
    clock.advance(3.0)
    b.step()
    assert b.sender.opened and OPEN in dev.calls and not events.held()


def test_bubble_blocked_still_releases(clock):
    b, dev, reader, events = pace(clock)
    say(reader, "团子")
    b.step()
    b._requests = {"小明": "hold"}  # 有互动请求挂着：不开框
    clock.advance(3.0)
    b.step()
    assert not b.sender.opened and not events.held()


def test_typing_keeps_holding(clock):
    typing = ["小明"]
    b, dev, reader, events = pace(clock, typing=lambda: list(typing))
    say(reader, "哈哈")
    b.step()
    clock.advance(5.0)
    b.step()
    assert events.held()
    typing.clear()
    clock.advance(2.9)
    b.step()
    assert events.held()  # 从最后看到他打字起重新算
    clock.advance(0.1)
    b.step()
    assert not events.held()


def test_upset_releases_same_step(clock):
    b, dev, reader, events = pace(clock)
    say(reader, "我今天好难过")
    b.step()
    assert not events.held() and events.has("chat")


def test_owner_command_flushes_held_batch(clock):
    b, dev, reader, events = pace(clock)
    b.cfg.brain.owner_name = "卡洛"
    say(reader, "哈哈")
    b.step()
    assert events.held()
    say(reader, "#过来", who="卡洛")
    b.step()
    assert not events.held() and events.has("owner_command") and events.has("chat")


def test_hold_released_by_max_wait_without_new_chat(clock):
    b, dev, reader, events = pace(clock, typing=lambda: ["小明"])
    say(reader, "哈哈")
    b.step()
    clock.advance(14.9)
    b.step()
    assert events.held()
    clock.advance(0.1)
    b.step()
    assert not events.held()


def test_status_shows_pacing(clock):
    b, dev, reader, events = pace(clock)
    say(reader, "哈哈")
    b.step()
    assert "在攒话：小明说了 1 句，再等 3 秒" in b.status()
    clock.advance(3.0)
    b.step()
    assert "在攒话" not in b.status()


def test_pacing_off_unchanged(clock):
    b, dev, reader, events = pace(clock, pacing=False)
    say(reader, "团子")
    b.step()
    assert not events.held() and b.sender.opened


# ---- say 标 jab、回法（spec 2026-10-07-chat-pacing §2）----
def test_jab_recorded_after_send(clock):
    b, dev, reader, events = pace(clock)
    b.say("你才废物", jab=True)
    clock.advance(10)
    b.say("好", jab=False)
    assert list(b._jabs) == [True, False]
    clock.advance(10)
    try:
        b.say("你才废物", jab=True)  # 刚说过：被拦下，不记
    except Exception:
        pass
    assert list(b._jabs) == [True, False]
    clock.advance(10)
    b.say("手动说的", live=True, jab=True)  # 手动控制：算不贱
    assert list(b._jabs)[-1] is False
    for text in ("雨林丑", "你是路痴", "我那是让着你", "锅是网的", "那当然", "斗篷啃了吧", "看了是路", "游戏针对我"):
        clock.advance(10)
        b.say(text, jab=True)
    assert len(b._jabs) == 6 and all(b._jabs)


def test_status_recent_jabs(clock):
    b, dev, reader, events = pace(clock)
    assert "最近说的" not in b.status()
    b.say("你才废物", jab=True)
    clock.advance(10)
    b.say("好")
    assert "最近说的：2 句里贱了 1 句" in b.status()
    b.cfg.pacing.manner = False
    assert "最近说的" not in b.status()


class LowMind:
    class mood:
        level = "低落"
        text = ""


def test_manner_line_uses_mood_and_energy(clock):
    b, dev, reader, events = pace(clock)
    assert b.manner_line([("小明", "今天去哪个图？")]) is None  # 没内心层：中性
    b.mind = LowMind()
    assert "没力气贫" in b.manner_line([("小明", "今天去哪个图？")])
