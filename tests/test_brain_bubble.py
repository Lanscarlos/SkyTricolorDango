"""身体反射 §2：有人跟团子说话时身体马上冒输入气泡（spec 2026-09-30-body-reflex）。"""

from test_brain_body import FakeEnv, FakeSocial, body, msg

from skydango.brain.skills import SkillStep
from skydango.device.base import KEYCODE_BACK
from skydango.game.social import Request

OPEN = ("hw_key", 28)
BACK = ("key", KEYCODE_BACK)


def rb(clock, live=True, **kw):
    env = FakeEnv()
    env.near = ["小明"]
    b, dev, reader, events = body(clock, live=live, env=env, **kw)
    b.friend_names = lambda: ["小明"]
    return b, dev, reader, env


def called(b, text="团子在吗", speaker="小明"):
    b.reader.batches = [[msg(text, speaker=speaker)]]
    b.step()


def test_opens_bubble_when_addressed(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    assert OPEN in dev.calls and b.sender.opened and b._bubble_at == clock()
    assert "输入框：开着（身体替你开的" in b.status()


def test_does_not_reopen_open_bubble(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    called(b, "团子？")
    assert dev.calls.count(OPEN) == 1


def test_say_uses_the_open_bubble(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    b.say("在呀")
    b.step()
    assert dev.calls.count(OPEN) == 1 and b._bubble_at is None and BACK not in dev.calls


def test_closes_when_round_ends_silent(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    opened = clock()
    clock.advance(3)
    b.brain_turn = lambda: (opened + 1, opened + 3)
    b.step()
    assert BACK in dev.calls and not b.sender.opened and b._bubble_at is None


def test_bubble_survives_round_that_started_before_it(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    opened = clock()
    b.brain_turn = lambda: (opened - 5, opened + 0.5)  # 开框之前就在想上一批
    clock.advance(1)
    b.step()
    assert b.sender.opened and BACK not in dev.calls


def test_closes_on_timeout(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    clock.advance(44)
    b.step()
    assert b.sender.opened
    clock.advance(1)  # bubble_max 45
    b.step()
    assert not b.sender.opened and BACK in dev.calls


def test_closes_before_keys_but_not_before_say(clock):
    b, dev, _, _ = rb(clock)
    called(b)
    b.clear_view("say", live=True)
    assert b.sender.opened
    b.clear_view("move", live=True)
    assert not b.sender.opened and b._bubble_at is None


def test_request_closes_bubble(clock):
    social = FakeSocial()
    b, dev, _, env = rb(clock, social=social)
    called(b)
    env.requests = {"小明": Request("小明", "hand", (5, 5), clock())}
    b.step()
    assert not b.sender.opened


def test_no_bubble_when_not_addressed(clock):
    b, dev, _, env = rb(clock)
    env.near = ["小明", "阿花"]
    b.friend_names = lambda: ["小明", "阿花"]
    called(b, "今天好热")  # 两个好友在、没叫名字、团子也没刚说过话
    called(b, "团子", speaker="路人甲")  # 陌生人
    assert OPEN not in dev.calls


def test_followup_after_tuanzi_spoke(clock):
    b, dev, _, env = rb(clock)
    env.near = ["小明", "阿花"]
    b.friend_names = lambda: ["小明", "阿花"]
    b.say("我先挂会儿")
    clock.advance(10)
    called(b, "去哪")
    assert b.sender.opened


def test_no_bubble_in_dry_run(clock):
    b, dev, _, _ = rb(clock, live=False)
    called(b)
    assert OPEN not in dev.calls


def test_no_bubble_when_disabled(clock):
    for key in ("enabled", "bubble"):
        b, dev, _, _ = rb(clock)
        setattr(b.cfg.reflex, key, False)
        called(b)
        assert OPEN not in dev.calls, key


class Spinning:  # 一直在跑的假技能（比如 track）
    name, goal, timeout = "spin", "转着", 999.0

    def tick(self, body, frame, now):
        return SkillStep("running", "")

    def stop(self, body, reason):
        pass


def test_no_bubble_while_skill_active(clock):
    b, dev, _, _ = rb(clock)
    b.skills.active = Spinning()
    called(b)
    assert OPEN not in dev.calls


def test_no_bubble_when_brain_offline(clock):
    b, dev, _, _ = rb(clock)
    b.brain_offline = lambda now: True
    called(b)
    assert OPEN not in dev.calls
