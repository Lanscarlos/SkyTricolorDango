"""好友在不在场：身体那一边（spec 2026-10-06-friend-presence §3）。"""

from test_brain_call_body import CallEnv, call_body, pressed

from skydango.config import PerceptionConfig
from skydango.vision.people import CallSeen
from skydango.vision.presence import Presence


class PresenceEnv(CallEnv):
    """感知层开着 presence 时的样子：nearby = 在场，in_view = 画面里，need_call = 找不到、还没喊过的。"""

    def __init__(self):
        super().__init__()
        self.presence = object()
        self.view = []
        self.lost = []
        self.around_list = []
        self.notes = {}

    def in_view(self, now):
        return list(self.view)

    def need_call(self, now):
        return list(self.lost)

    def around(self, now):
        return list(self.around_list)

    def left_note(self, name, now):
        return self.notes.get(name, "")


def pbody(clock, **kw):
    return call_body(clock, env=PresenceEnv(), **kw)


def test_presence_call_presses_once_for_all_lost(clock):
    b, device, env, events = pbody(clock)
    env.lost = ["小明", "阿花"]
    b._watch_call(clock())
    assert pressed(device) == 1 and b.last_call.reason == "presence"
    b._watch_call(clock())
    assert pressed(device) == 1  # 上一声还没结果
    env.results[b.last_call.at] = CallSeen(b.last_call.at, {}, ended=True)
    b._watch_call(clock())
    assert not [e for e in events.drain() if e.kind == "call"]  # 确认在场的这一声不叫大脑


def test_presence_call_respects_min_gap(clock):
    b, device, env, _ = pbody(clock)
    env.lost = ["小明"]
    b._call_at = clock() - 5
    b._watch_call(clock())
    assert pressed(device) == 0


def test_presence_call_blocked_while_typing(clock):
    b, device, env, _ = pbody(clock)
    env.lost = ["小明"]
    device.shown = True  # 输入框开着：Q 会变成打字
    b._watch_call(clock())
    assert pressed(device) == 0


def test_presence_dry_run_never_presses(clock):
    b, device, env, _ = pbody(clock, live=False)
    env.lost = ["小明"]
    b._watch_call(clock())
    assert pressed(device) == 0


def test_leave_text_from_presence(clock):
    b, _, env, events = pbody(clock)
    env.near = ["小明"]
    env.view = ["小明"]
    b._watch_comings(clock())
    events.drain()
    env.near, env.view = [], []
    env.notes = {"小明": "喊了一声也没看到，15 秒了"}
    b._watch_comings(clock())
    leaves = [e.text for e in events.drain() if e.kind == "leave"]
    assert leaves == ["小明 走开了（喊了一声也没看到，15 秒了）"]


def test_out_of_view_starts_lost_search_without_leave(clock):
    b, _, env, events = pbody(clock)
    env.near, env.view = ["小明"], ["小明"]
    b._watch_comings(clock())
    events.drain()
    clock.advance(1)
    env.view = []  # 走出画面，但还在附近
    b._watch_comings(clock())
    assert not [e for e in events.drain() if e.kind == "leave"]
    assert b._out_at["小明"] == clock()
    started = []
    b.attention.start_lost = lambda name, side, edge, can_call, now: started.append(name) or True
    b._start_lost_search(clock())
    assert started == ["小明"]


def test_status_shows_view_and_around(clock):
    b, _, env, _ = pbody(clock)
    env.near = ["小明", "阿花"]
    env.view = ["小明"]
    env.around_list = [("阿花", "画面外·右边")]
    text = b.status()
    assert "身边的好友：小明" in text
    assert "附近：阿花（画面外·右边）" in text


class RealPresenceEnv(CallEnv):
    """身体接真的 Presence（感知层那几个方法照 PerceptionWatcher 的写法）。"""

    def __init__(self):
        super().__init__()
        self.seen = {}
        self.presence = Presence(PerceptionConfig(), self.seen, can_call=True)

    def nearby(self, now):
        return self.presence.present(["小明"], now)

    def in_view(self, now):
        return self.presence.in_view(["小明"], now)

    def need_call(self, now):
        return self.presence.need_call(["小明"], now)

    def around(self, now):
        return self.presence.around(["小明"], now)

    def left_note(self, name, now):
        return self.presence.left_note(name, now)


def test_out_of_view_is_not_leave_until_call_fails(clock):
    b, _, env, events = call_body(clock, env=RealPresenceEnv())
    env.seen["小明"] = clock()
    b._watch_comings(clock())
    assert [e.kind for e in events.drain()] == ["arrive"]
    clock.advance(6)  # 走出画面 6 秒：找不到，还不算走开
    b._watch_comings(clock())
    assert not [e for e in events.drain() if e.kind == "leave"]
    env.presence.call_done(clock(), {})  # 喊了一声没亮出他
    clock.advance(8)
    b._watch_comings(clock())
    assert not [e for e in events.drain() if e.kind == "leave"]
    clock.advance(1.5)
    b._watch_comings(clock())
    assert [e.text for e in events.drain() if e.kind == "leave"] == ["小明 走开了（喊了一声也没看到，16 秒了）"]


def test_search_adopts_pending_presence_call(clock):
    b, device, env, _ = pbody(clock)
    env.lost = ["小明"]
    b._watch_call(clock())  # 确认在场的那一声先按了
    assert b._pending_auto is not None and b._pending_auto.reason == "presence"
    got = []
    b.attention.search = type("S", (), {"who": "小明", "called": lambda self, seen, now: got.append(seen),
                                        "call_sent": lambda self, now: got.append("sent")})()
    b._search_call(clock())
    assert got == ["sent"] and b._search_call_pending  # 找人那一步等这一声的结果，不另喊、不当成"没喊"
    seen = CallSeen(b._pending_auto.at, {}, ended=True)
    env.results[b._pending_auto.at] = seen
    b._watch_call(clock())
    assert got == ["sent", seen]


def test_search_reuses_fresh_presence_result(clock):
    b, device, env, _ = pbody(clock)
    env.lost = ["小明"]
    b._watch_call(clock())
    seen = CallSeen(b._pending_auto.at, {}, ended=True)
    env.results[b._pending_auto.at] = seen
    env.lost = []
    b._watch_call(clock())  # 结果收回来了
    got = []
    b.attention.search = type("S", (), {"who": "小明", "called": lambda self, s, now: got.append(s),
                                        "call_sent": lambda self, now: got.append("sent")})()
    clock.advance(3)
    b._search_call(clock())
    assert got == [seen] and pressed(device) == 1
