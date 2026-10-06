"""好友在不在场：身体那一边（spec 2026-10-06-friend-presence §3）。"""

from test_brain_call_body import CallEnv, call_body, pressed

from skydango.vision.people import CallSeen


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
