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
    clock.advance(b.cfg.reflex.min_gap)
    b.step()
    assert presses(dev) == 2 and env.tried[-1] == 8  # 请求只是被推迟，没丢


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


def test_accepting_candle_while_bow_pending_keeps_one_bow_without_lowering(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()  # 排上鞠躬，带着要兜底放下的蜡烛
    env.requests["陌生人"] = Request("陌生人", "candle", (990, 400), clock())
    social.to_handle = ["陌生人:candle"]
    b.step()  # 接受别人点火又要排鞠躬：不能把上一个覆盖掉（点圆圈会放下蜡烛：不用兜底放下了，见最终审查 #4）
    assert b._bow is not None and b._bow[2] is None
    b._reflex_emote = lambda *a: False
    clock.advance(2.6)
    b.step()
    clock.advance(5.1)
    b.step()
    assert emotes.done == [] and presses(dev) == 1


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


def test_accepting_candle_while_raised_stops_waiting_for_lit(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()  # 举起蜡烛
    env.requests["陌生人"] = Request("陌生人", "candle", (990, 400), clock())
    social.to_handle = ["陌生人:candle"]
    b.step()  # 接受别人点火：先排上一个没有 raised_at 的鞠躬；点圆圈放下了蜡烛，不再等他亮起来（最终审查 #4）
    env.requests.pop("陌生人")
    social.to_handle = []
    env.lit_result = True
    b.step()
    assert b._raised is None and b._bow is not None and b._bow[2] is None
    b._reflex_emote = lambda *a: False
    clock.advance(2.6)
    b.step()
    clock.advance(5.1)
    b.step()
    assert emotes.done == [] and presses(dev) == 1


# ---- 最终审查 #2 / #4 / #5 ----
def test_no_raise_while_body_bubble_open(clock):
    """身体替大脑开着输入框（有人在跟团子说话）：不举蜡烛（按键会把框关掉，大脑那句就发不出去）。"""
    b, dev, env, emotes, events = lb(clock)
    assert b._open_bubble(clock()) and b._bubble_at is not None
    offer(env, clock)
    b.step()
    assert presses(dev) == 0 and env.tried == [] and b.sender.opened


def test_lowering_candle_closes_body_bubble_properly(clock):
    """举着蜡烛时身体开了框，超时放下蜡烛：按键前走 _close_bubble，sender 知道框关了（否则 say 往空里打字）。"""
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert presses(dev) == 1
    assert b._open_bubble(clock()) and b.sender.opened
    clock.advance(8.1)
    b.step()
    assert presses(dev) == 2
    assert not b.sender.opened and b._bubble_at is None


def test_accepting_other_interaction_while_raised_forgets_candle(clock):
    """举着蜡烛时接受了别的互动（点圆圈会放下蜡烛）：之后超时不能再按 3（那会把蜡烛举起来）。"""
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()
    env.requests["小明"] = Request("小明", "hand", (990, 400), clock())
    social.to_handle = ["小明:hand"]
    b.step()
    env.requests.pop("小明")
    assert "正在举蜡烛" not in b.status()
    clock.advance(8.1)
    b.step()
    assert presses(dev) == 1


def test_accepting_other_interaction_keeps_bow_but_not_lowering(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()  # 点亮了，鞠躬排上，带着兜底放下的蜡烛
    env.requests["小明"] = Request("小明", "hand", (990, 400), clock())
    social.to_handle = ["小明:hand"]
    b.step()
    env.requests.pop("小明")
    assert b._bow is not None and b._bow[2] is None  # 鞠躬留着，蜡烛当作已经放下
    b._reflex_emote = lambda *a: False
    clock.advance(2.6)
    b.step()
    clock.advance(5.1)
    b.step()
    assert presses(dev) == 1


def test_blackout_while_raised_does_not_press_on_timeout(clock):
    """举着蜡烛时黑过屏（切场景）：蜡烛状态不明，超时不按 3。"""
    import numpy as np
    from conftest import scene

    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    dev.frames = [np.zeros((720, 1280, 3), np.uint8)]
    clock.advance(1.5)
    b.step()
    assert b.blackout
    dev.frames = [scene()]
    clock.advance(1.5)
    b.step()
    assert not b.blackout
    clock.advance(5.2)
    b.step()
    assert presses(dev) == 1


def test_schedule_bow_merges_raised_at_into_pending_bow(clock):
    """已经排了一个鞠躬：再排只补上还缺的 raised_at，不覆盖时间、不丢已有的。"""
    b, dev, env, emotes, events = lb(clock)
    b._schedule_bow(clock(), None)
    due = b._bow[0]
    b._schedule_bow(clock() + 1, 5.0)
    assert b._bow[0] == due and b._bow[2] == 5.0
    b._schedule_bow(clock() + 2, None)
    assert b._bow[2] == 5.0
