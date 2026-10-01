"""点亮没点火的陌生人 + 点火后鞠躬（spec 2026-10-01-light-unlit-stranger §4~§6）。"""

from test_brain_body import FakeEnv, FakeSocial, body
from test_brain_reflex_body import ReflexEmotes

from skydango.game.social import LIGHT, LIGHT_KEY, Request

CANDLE_KEY = ("hw_key", 4)  # 3 号格 = slot_keys[2] = 4


class LightEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.tried = []
        self.lit_result = False

    def mark_tried(self, track_id):
        self.tried.append(track_id)
        self.requests.pop(LIGHT_KEY, None)

    def lit(self, track_id, pos, since):
        return self.lit_result


class LightEmotes(ReflexEmotes):
    def on_wheel(self):
        return ["鞠躬", "挥手"]

    def press_slot(self, slot):
        self.dev.calls.append(("hw_key", [2, 3, 4, 5, 6, 7, 8, 9][slot - 1]))


class AllowAll(FakeSocial):
    def allowed(self, req):
        return True


def lb(clock, live=True, social=None):
    from conftest import FakeDevice, scene

    dev = FakeDevice([scene()])
    emotes = LightEmotes(dev)
    emotes.clock = clock
    env = LightEnv()
    b, dev, reader, events = body(clock, live=live, env=env, emotes=emotes, social=social or AllowAll(), device_override=dev)
    b.step()
    events.drain()
    return b, dev, env, emotes, events


def offer(env, clock, track=7):
    env.requests[LIGHT_KEY] = Request("陌生人", LIGHT, (990, 620), clock(), track=track)


def presses(dev):
    return dev.calls.count(CANDLE_KEY)


def test_raise_candle_once_then_bow_when_lit(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert presses(dev) == 1 and env.tried == [7]
    b.step()
    assert presses(dev) == 1  # 请求被 mark_tried 拿掉了，不会再按
    env.lit_result = True
    b.step()
    assert [e.kind for e in events.drain()] == ["accepted"]
    clock.advance(2.0)
    b.step()
    assert emotes.done == []  # 还没到 bow_delay
    clock.advance(0.6)
    b.step()
    assert emotes.done == [("鞠躬", True)] and presses(dev) == 1  # 鞠躬放下蜡烛，不按 3


def test_timeout_lowers_candle_without_bow(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(8.1)
    b.step()
    assert presses(dev) == 2 and emotes.done == []  # 举一次、放一次


def test_no_lower_if_emoted_meanwhile(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(1.0)
    emotes.perform("挥手")  # 大脑做了个动作：蜡烛已经放下了
    clock.advance(7.1)
    b.step()
    assert presses(dev) == 1


def test_bow_gives_up_and_lowers(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()
    b._reflex_emote = lambda *a: False  # 一直做不了反射动作（被挡着、刚做过……）
    clock.advance(2.5)
    b.step()
    clock.advance(5.1)
    b.step()
    assert emotes.done == [] and presses(dev) == 2


def test_no_bow_name_on_wheel_lowers_right_away(clock):
    b, dev, env, emotes, events = lb(clock)
    b.cfg.social.after_light = ""
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()
    assert presses(dev) == 2 and emotes.done == []


def test_dry_run_only_logs(clock):
    b, dev, env, emotes, events = lb(clock, live=False)
    offer(env, clock)
    b.step()
    assert presses(dev) == 0 and env.tried == [7]


def test_policy_off_does_nothing(clock):
    class DenyLight(FakeSocial):
        def allowed(self, req):
            return req.kind != LIGHT

    b, dev, env, emotes, events = lb(clock, social=DenyLight())
    offer(env, clock)
    b.step()
    assert presses(dev) == 0 and env.tried == []


def test_not_while_holding_hands(clock):
    b, dev, env, emotes, events = lb(clock)
    b.holding = "小明"
    offer(env, clock)
    b.step()
    assert presses(dev) == 0


def test_light_request_makes_no_request_event_and_never_taps(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert "request" not in [e.kind for e in events.drain()]
    assert not [c for c in dev.calls if c[0] == "tap"]


def test_bow_after_accepting_candle(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    env.requests["陌生人"] = Request("陌生人", "candle", (990, 400), clock())
    social.to_handle = ["陌生人:candle"]
    b.step()
    env.requests.pop("陌生人")
    clock.advance(2.6)
    b.step()
    assert emotes.done == [("鞠躬", True)] and presses(dev) == 0


def test_reflex_waits_while_candle_raised(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert b._reflex_emote("挥手", "测试", clock()) is False


def test_status_mentions_raised_candle(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert "正在举蜡烛给陌生人点火" in b.status()


def test_policy_tool_accepts_stranger_light(clock):
    b, dev, env, emotes, events = lb(clock)
    b.set_policy("stranger", "light", False)
    assert b.social.policy_calls == [("stranger", "light", False)]


def test_no_second_candle_while_bow_pending(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()  # 点亮了，鞠躬排上，蜡烛还举着
    env.lit_result = False
    offer(env, clock, track=8)
    b.step()
    assert presses(dev) == 1  # 不能再按 3（那会把蜡烛放下）
    clock.advance(2.6)
    b.step()
    assert emotes.done == [("鞠躬", True)] and presses(dev) == 1


def test_shutdown_lowers_candle_when_bow_pending(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()  # 点亮了，鞠躬还没做
    b.shutdown()
    assert presses(dev) == 2


def test_shutdown_lowers_candle_when_raised_not_lit(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    b.shutdown()
    assert presses(dev) == 2


def test_schedule_bow_keeps_pending_raised_at(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()  # 排上鞠躬，带着要兜底放下的蜡烛
    env.requests["陌生人"] = Request("陌生人", "candle", (990, 400), clock())
    social.to_handle = ["陌生人:candle"]
    b.step()  # 接受别人点火又要排鞠躬：不能把上一个覆盖掉
    b._reflex_emote = lambda *a: False
    clock.advance(2.6)
    b.step()
    clock.advance(5.1)
    b.step()
    assert emotes.done == [] and presses(dev) == 2


def test_emote_while_raised_clears_state(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(1.0)
    emotes.perform("挥手")
    b.step()
    assert "正在举蜡烛" not in b.status() and presses(dev) == 1
    env.lit_result = True
    b.step()
    assert presses(dev) == 1 and b._bow is None  # 不鞠躬、不按 3


def test_min_gap_before_raising(clock):
    b, dev, env, emotes, events = lb(clock)
    emotes.perform("挥手")
    offer(env, clock)
    b.step()
    assert presses(dev) == 0
    clock.advance(b.cfg.reflex.min_gap)
    b.step()
    assert presses(dev) == 1


def test_dry_run_lower_candle_presses_nothing(clock):
    b, dev, env, emotes, events = lb(clock, live=False)
    b._lower_candle(clock() - 1)
    assert presses(dev) == 0
