"""身体反射 §3：被叫到的小动作、回礼、闲着的小动作，额度和 min_gap（spec 2026-09-30-body-reflex）。"""

import pytest
from test_brain_body import FakeEnv, body, msg
from test_brain_reflex import Rng

from skydango.brain.body import ToolError

OPEN = ("hw_key", 28)


class ReflexEmotes:
    def __init__(self, dev):
        self.dev = dev
        self.done = []  # (动作, 是不是反射)
        self.pretended = []
        self.last_any = float("-inf")
        self.clock = None  # rx() 里设成测试的 clock

    def on_wheel(self):
        return ["点头", "挥手", "伸懒腰"]

    def available(self, ignore_interval=False):
        return self.on_wheel()

    def perform(self, name, reflex=False):
        self.done.append((name, reflex))
        self.dev.calls.append(("emote", name))
        self.last_any = self.clock()

    def pretend(self, name, reflex=False):
        self.pretended.append((name, reflex))
        self.last_any = self.clock()


class GestureEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.gestures = []

    def pop_gestures(self):
        out, self.gestures = self.gestures, []
        return out


def rx(clock, r=0.0, live=True, **reflex_cfg):
    from conftest import FakeDevice, scene

    dev = FakeDevice([scene()])
    emotes = ReflexEmotes(dev)
    emotes.clock = clock
    env = GestureEnv()
    env.near = ["小明"]
    b, dev, reader, events = body(clock, live=live, env=env, emotes=emotes, rng=Rng(r), device_override=dev)
    for k, v in reflex_cfg.items():
        setattr(b.cfg.reflex, k, v)
    b.cfg.reflex.return_map = reflex_cfg.get("return_map", {"wave": "挥手"})
    b.friend_names = lambda: ["小明"]
    b.reflexes.stir(clock())  # 闲着的计时按改过的配置重新算
    b.step()  # 小明来了（arrive）：先消化掉
    events.drain()
    return b, dev, reader, events, emotes


def kinds(events):
    return [e.kind for e in events.drain()]


def test_nod_then_bubble(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"])
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    assert emotes.done == [("点头", True)] and b.sender.opened
    assert dev.calls.index(("emote", "点头")) < dev.calls.index(OPEN)  # 先动作再开框
    assert kinds(events) == ["chat", "reflex"]


def test_one_nod_per_batch(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"], min_gap=0.0)
    reader.batches = [[msg("团子", speaker="小明"), msg("团子？", speaker="小明")]]
    b.step()
    assert len(emotes.done) == 1


def test_no_nod_when_chance_misses(clock):
    b, dev, reader, events, emotes = rx(clock, r=0.5, addressed=["点头"])  # addressed_chance 0.3
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    assert emotes.done == [] and b.sender.opened


def test_return_gesture_replaces_event(clock):
    b, dev, reader, events, emotes = rx(clock)
    b.env.gestures = [("小明", "wave")]
    b.step()
    assert emotes.done == [("挥手", True)]
    ev = events.drain()
    assert [e.kind for e in ev] == ["reflex"] and ev[0].text == "小明对你挥手，你回了个挥手"


def test_gesture_goes_to_brain_when_not_returned(clock):
    b, dev, reader, events, emotes = rx(clock, r=0.9)  # return_chance 0.7
    b.env.gestures = [("小明", "wave")]
    b.step()
    assert emotes.done == [] and kinds(events) == ["gesture"]


def test_no_return_to_holding_partner(clock):
    b, dev, reader, events, emotes = rx(clock)
    b.holding = "小明"
    b.env.gestures = [("小明", "wave")]
    b.step()
    assert emotes.done == [] and events.drain() == []


def test_return_reopens_bubble(clock):
    b, dev, reader, events, emotes = rx(clock)
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    opened = b._bubble_at
    clock.advance(1)
    b.env.gestures = [("小明", "wave")]
    b.step()
    assert emotes.done == [("挥手", True)] and b.sender.opened and b._bubble_at == opened


def test_idle_after_quiet_and_not_while_typing(clock):
    b, dev, reader, events, emotes = rx(clock, idle=["伸懒腰"], idle_min=10.0, idle_max=10.0)
    clock.advance(9)
    b.step()
    assert emotes.done == []
    clock.advance(1)
    b.step()
    assert emotes.done == [("伸懒腰", True)] and kinds(events) == ["reflex"]
    reader.batches = [[msg("团子", speaker="小明")]]  # 框开着：正在“打字”，闲着不做
    b.step()
    clock.advance(20)  # 过了 idle_min，但框还没到 bubble_max
    b.step()
    assert len(emotes.done) == 1 and b.sender.opened


def test_min_gap_and_quota(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"], quota=1, bubble=False)
    emotes.last_any = clock() - 3  # 3 秒前刚做过一个（min_gap 4）
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    assert emotes.done == []
    clock.advance(2)
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    assert emotes.done == [("点头", True)]
    clock.advance(10)
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    assert len(emotes.done) == 1  # quota 1 用完了


def test_brain_emote_waits_min_gap(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"])
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    clock.advance(1)
    with pytest.raises(ToolError, match="刚做完一个动作"):
        b.emote("挥手")
    clock.advance(3)
    assert b.emote("挥手") == "做了「挥手」" and emotes.done[-1] == ("挥手", False)


def test_status_shows_recent_reflex(clock):
    b, dev, reader, events, emotes = rx(clock)
    b.env.gestures = [("小明", "wave")]
    b.step()
    clock.advance(12)
    assert "刚才下意识：小明对你挥手，你回了个挥手（12 秒前）" in b.status()


def test_reflex_disabled_changes_nothing(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"], idle=["伸懒腰"])
    b.cfg.reflex.enabled = False
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    b.env.gestures = [("小明", "wave")]
    clock.advance(500)
    b.step()
    assert emotes.done == [] and OPEN not in dev.calls
    assert kinds(events) == ["chat", "gesture"]
    emotes.last_any = clock()
    assert b.emote("挥手") == "做了「挥手」"  # 不查 min_gap


def test_dry_run_pretends(clock):
    b, dev, reader, events, emotes = rx(clock, live=False)
    b.env.gestures = [("小明", "wave")]
    b.step()
    assert emotes.pretended == [("挥手", True)] and emotes.done == [] and kinds(events) == ["reflex"]
