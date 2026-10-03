"""身体接上有意识地找（plan 2026-10-03-attention-search Task 5）：好友走开往他那边找、中间淡掉先喊、环顾、search 事件。"""

from test_brain_attention_body import attn_body, steps

from skydango.brain.calling import CallResult
from skydango.vision.people import CallSeen, Seen


def quiet(b):
    """别让聊天面板定时看一眼挡住注意力；只看找。"""
    b.cfg.panel.idle_peek = 1e9


def leave(b, env, clock, name="小明", x=60):
    """小明在身边、名字标签最后在 x，然后走开（不经过"来了"，免得叫开聊天面板）。
    标签时间往前拨 6 秒：真机上"走开"是 keep（5 s）没看到名字才判的，标签早就过了 track.max_age，不算"还在画面里"。"""
    env.labels[name] = (x, 300, 80, 20, clock() - 6.0)
    b._nearby = {name}
    env.near = []


def search_events(events):
    return [e.text for e in events.drain() if e.kind == "search"]


def test_friend_leaving_from_left_edge_makes_attention_look_left(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.call.auto = False
    b.cfg.attention.scan_after = 1e9
    leave(b, env, clock, x=60)
    steps(b, clock, 3)
    assert "在找：小明（他刚从左边走了" in b.status()
    steps(b, clock, 30)
    assert cam.nudges and {d for d, s in cam.nudges} == {"left"}
    assert len(cam.nudges) == b.cfg.attention.lost_segments * b.cfg.attention.seg_presses
    assert search_events(events) == ["你往左边找了找刚走开的小明，没看到他"]


def test_middle_fade_calls_first_and_found_by_call(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.attention.scan_after = 1e9
    env.unnamed = lambda now: 1  # 有感知层的呼喊接口（call_available 看 hasattr）
    env.called = lambda at, by_self=True: None
    env.call_result = lambda at: CallSeen(at, {"小明": Seen("右边", "远")})
    calls = []

    def fake_call_out(reason, live=False):
        calls.append(reason)
        b._call_at = clock()
        return CallResult(clock(), reason)

    b.call_out = fake_call_out
    leave(b, env, clock, x=900)
    steps(b, clock, 6)
    assert calls == ["auto"] and cam.nudges == []
    assert "刚才找到了小明（右边·远）" in b.status()
    kinds = [e.kind for e in events.drain()]
    assert "call" in kinds and "search" not in kinds


def test_auto_call_left_to_search_unless_still_mode(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    env.unnamed = lambda now: 1
    env.called = lambda at, by_self=True: None
    env.call_result = lambda at: None
    calls = []
    b.call_out = lambda reason, live=False: (calls.append(reason), CallResult(clock(), reason))[1]
    b.attention.set_mode("别动", None)  # 注意力不找：老的自动喊照旧
    leave(b, env, clock, x=900)
    steps(b, clock, 2)
    assert calls == ["auto"]


def test_alone_scans_and_reports_empty_in_status_only(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.attention.scan_segments = [2, 2]
    steps(b, clock, 50)  # 35 s：20 s 后环顾一片，转完附近没人
    assert len(cam.nudges) == 2 * b.cfg.attention.seg_presses
    assert "附近没人" in b.status()
    assert search_events(events) == []  # 还是没人：只写 status


def test_dry_run_thinks_but_never_presses(clock):
    b, dev, reader, events, env, cam = attn_body(clock, live=False)
    quiet(b)
    b.cfg.call.auto = False
    leave(b, env, clock, x=60)
    steps(b, clock, 3)
    assert cam.nudges == [] and "在找：小明" in b.status() and b._attention_why == "dry-run"
