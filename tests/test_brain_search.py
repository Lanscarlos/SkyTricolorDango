"""有意识地找：纯计算（plan 2026-10-03-attention-search Task 2 / 3）。画面宽 1920，hfov 90°，press_deg 18°。"""

import random

import pytest

from skydango.brain.peek import Turn
from skydango.brain.search import SECTORS, TERMINAL, Heading, Obs, Search, bearing_name, event_text
from skydango.config import Config
from skydango.vision.people import CallSeen, Seen


def cfgs(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.attention, k, v)
    return cfg.attention, cfg.track


def heading(now=0.0):
    a, t = cfgs()
    return Heading(a.press_deg, t.nudge_max, 90.0, now)


# ---- Task 2：Heading ----

def test_bearing_names():
    assert [bearing_name(d) for d in (0, 60, 120, 180, 240, 300, 359)] == [
        "前面", "右前方", "右后方", "后面", "左后方", "左前方", "前面"]
    assert bearing_name(66) == "右前方" and bearing_name(-60) == "左前方"


def test_heading_starts_facing_front_only():
    h = heading()
    assert h.deg == 0.0 and h.seen[0] == 0.0 and all(t == float("-inf") for t in h.seen[1:])


def test_heading_turns_by_press_seconds():
    h = heading()
    nudge = Config().track.nudge_max
    h.turned("right", nudge * 3, 5.0)  # 3 下 × 18° = 54°
    assert h.deg == pytest.approx(54.0) and h.seen[1] == 5.0 and h.seen[0] == 0.0
    h.turned("left", nudge * 6, 10.0)  # 往左 108° → 306°
    assert h.deg == pytest.approx(306.0) and h.seen[5] == 10.0


def test_stalest_prefers_never_seen_then_nearest_then_prefer_side():
    h = heading()
    assert h.stalest("left") == ("left", "左前方")  # 左前、右前都没看过、一样近：按 prefer
    assert h.stalest("right") == ("right", "右前方")
    nudge = Config().track.nudge_max
    h.turned("right", nudge * 3, 5.0)
    h.turned("left", nudge * 6, 10.0)  # 现在朝 306°：看过 0（t=0）、60（t=5）、300（t=10）
    assert h.stalest("right") == ("left", "左前方")  # 没看过的 120 / 180 / 240 里 240 最近（往左 66°）


def test_heading_reset_forgets():
    h = heading()
    h.turned("right", Config().track.nudge_max * 3, 5.0)
    h.reset(9.0)
    assert h.deg == 0.0 and h.seen[0] == 9.0 and h.seen[1] == float("-inf") and SECTORS == 6


# ---- Task 3：Search ----

def no_call(s, t):
    s.called(None, t)  # 调用方没喊成（额度不够 / 关着）


def run(s, obs_fn, t0=0.0, until=60.0, dt=0.1, on_call=no_call):
    """模拟调用方：每 dt 秒 step 一次，press 就当真按了，call 交给 on_call(s, t)。返回 [(t, step)]，结束就停。"""
    out, t = [], t0
    while t <= until:
        st = s.step(obs_fn(t), t)
        out.append((t, st))
        if st.state == "press":
            s.pressed(st.turn, t)
        elif st.state == "call":
            on_call(s, t)
        if st.state in TERMINAL:
            break
        t = round(t + dt, 6)
    return out


def presses(log):
    return [st.turn.direction for _, st in log if st.state == "press"]


def test_lost_from_left_edge_turns_left_then_calls_then_gives_up():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, True, a, tr, heading(), 0.0)
    calls = []
    log = run(s, lambda t: Obs(), on_call=lambda s, t: (calls.append(t), s.called(None, t)))
    assert presses(log) == ["left"] * (2 * a.seg_presses)
    assert all(st.turn == Turn("left", tr.nudge_max) for _, st in log if st.state == "press")
    first_press = next(t for t, st in log if st.state == "press")
    assert len(calls) == 1 and calls[0] > first_press  # 先转、转完才喊
    assert log[-1][1].state == "none" and s.note == "往左边找了找，没看到小明"
    assert event_text(s, False) == "你往左边找了找刚走开的小明，没看到他"


def test_lost_call_reveals_offscreen_tag_then_turns_that_way():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, True, a, tr, heading(), 0.0)
    seen = CallSeen(0.0, {"小明": Seen("右边", None, on_screen=False)})

    def on_call(s, t):
        s.call_sent(t)
        s.called(seen, t)

    log = run(s, lambda t: Obs(), on_call=on_call)
    assert presses(log) == ["left"] * 6 + ["right"] * 6
    assert s.note == "往左边、右边找了找，没看到小明，喊了一声也没看到他的名字"


def test_lost_middle_calls_first_and_name_lights_up():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    assert s.step(Obs(), 0.0).state == "call"
    assert s.step(Obs(), 0.1).state == "call"  # 调用方这一圈没喊（被挡住）：还要喊
    s.call_sent(0.1)
    assert s.step(Obs(), 0.2).state == "wait"
    s.called(CallSeen(0.1, {"小明": Seen("右边", "远")}), 6.5)
    st = s.step(Obs(), 6.6)
    assert st.state == "found" and s.note == "小明（右边·远）"
    assert event_text(s, False) is None  # 找到了：走现有的 return
    assert s.result_line() == "找到了小明（右边·远）"


def test_lost_middle_with_unnamed_people_does_not_turn_away():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(strangers=1))
    assert presses(log) == []
    assert log[-1][1].state == "none" and "1 个没挂名字的人" in s.note
    assert event_text(s, False) is None  # 没转没喊（这一声被拦了）：什么都没做，不报事件，只留在 status
    assert "1 个没挂名字的人" in s.result_line()


def test_lost_called_with_unnamed_people_event_mentions_them():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)

    def sent(s, t):
        s.call_sent(t)
        s.called(CallSeen(t, {}), t + 6.0)  # 喊了、没认出他

    log = run(s, lambda t: Obs(strangers=2), on_call=sent)
    assert presses(log) == [] and log[-1][1].state == "none"
    assert event_text(s, False) == "你喊了一声找刚走开的小明，没看到他（画面里还有 2 个没挂名字的人，可能就是他）"


def test_lost_nothing_done_gives_no_event():
    a, tr = cfgs()
    s = Search.lost("小明", None, False, False, a, tr, heading(), 0.0)  # 不能喊、没有线索
    s.step(Obs(strangers=1), 0.0)
    assert s.state == "none" and event_text(s, False) is None


def test_lost_middle_alone_turns_toward_his_side_after_call():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs())
    assert presses(log) == ["right"] * (a.lost_segments * a.seg_presses)


def test_lost_found_mid_turn_stops():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(target_x=500.0, target_where="左边·中") if t >= 1.0 else Obs())
    assert log[-1][1].state == "found" and s.note == "小明（左边·中）"
    assert len(presses(log)) == 2  # 0.0、0.6 按了两下，1.0 看到了


def test_lost_maybe_is_reported_as_maybe():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(maybe_x=300.0) if t >= 1.0 else Obs())
    assert log[-1][1].state == "maybe" and s.note == "小明（没看到名字，左边）"
    assert event_text(s, False) == "你往左边找了找刚走开的小明，那边有个人可能是他（没看到名字）"


def test_lost_without_call_never_asks_to_call():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(), on_call=lambda s, t: pytest.fail("不该喊"))
    assert presses(log) == ["left"] * 6 and log[-1][1].state == "none"


def test_search_gives_up_when_not_advanced_for_resume_within():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.step(Obs(), 0.0).state == "press"  # 调用方一直没按（被挡住）
    assert s.step(None, 0.5).state == "wait"  # 画面暂停：等着
    assert s.step(Obs(), 9.9).state == "press"
    st = s.step(Obs(), 10.2)
    assert st.state == "none" and s.aborted and event_text(s, False) is None
    assert s.result_line() == "找小明，被打断太久，不找了"


def test_aborted_scan_does_not_claim_it_looked():
    a, tr = cfgs(scan_segments=[2, 2])
    s = Search.scan(a, tr, heading(), random.Random(1), "left", 0.0)
    assert s.step(Obs(), 0.0).state == "press"
    assert s.step(Obs(), 10.5).state == "none" and s.aborted
    assert s.result_line() == f"想往{s.where}看看，被打断了"


def test_scan_turns_toward_stalest_and_reports_empty():
    a, tr = cfgs(scan_segments=[2, 2])
    h = heading()
    h.turned("right", tr.nudge_max * 3, 5.0)
    h.turned("left", tr.nudge_max * 6, 10.0)
    s = Search.scan(a, tr, h, random.Random(1), "right", 11.0)
    assert s.where == "左前方" and s.describe() == "在看看周围：往左前方（那边很久没看了）"
    log = run(s, lambda t: Obs(), t0=11.0)
    assert presses(log) == ["left"] * 6 and log[-1][1].state == "empty" and s.note == "附近没人"
    assert s.result_line() == "往左前方看了看：附近没人"
    assert event_text(s, True) is None


def test_scan_segments_come_from_rng_range():
    a, tr = cfgs(scan_segments=[2, 3])
    counts = set()
    for seed in range(20):
        s = Search.scan(a, tr, heading(), random.Random(seed), "right", 0.0)
        counts.add(len(presses(run(s, lambda t: Obs()))) // a.seg_presses)
    assert counts == {2, 3}


def test_scan_sees_strangers_looks_then_reports():
    a, tr = cfgs()
    s = Search.scan(a, tr, heading(), random.Random(1), "right", 0.0)
    log = run(s, lambda t: Obs(strangers=2) if t >= 1.0 else Obs())
    end_t, end = log[-1]
    assert end.state == "seen" and s.note == "有 2 个陌生人" and end_t == pytest.approx(1.0 + a.scan_look)
    assert not any(st.state == "press" for t, st in log if t >= 1.0)
    assert event_text(s, True) == f"你往{s.where}看了看：有 2 个陌生人"
    assert event_text(s, False) is None  # 上次也有人：不算新鲜


def test_scan_sees_friend_stops_without_event():
    a, tr = cfgs()
    s = Search.scan(a, tr, heading(), random.Random(1), "right", 0.0)
    log = run(s, lambda t: Obs(friends=("小明",)) if t >= 1.0 else Obs())
    assert log[-1][1].state == "seen" and s.friends == ("小明",) and s.note == "看到了小明"
    assert event_text(s, True) is None  # 好友认出来就进身边名单，arrive / return 会说


def test_find_without_clue_calls_then_turns_full_circle():
    a, tr = cfgs()
    s = Search.find("小明", None, False, True, a, tr, heading(), 0.0)

    def on_call(s, t):
        s.call_sent(t)
        s.called(CallSeen(t, {}), t)

    log = run(s, lambda t: Obs(), until=120.0, on_call=on_call)
    dirs = presses(log)
    assert len(dirs) == SECTORS * a.seg_presses and len(set(dirs)) == 1
    assert s.kind == "find" and s.note == "转了一圈没找到小明，喊了一声也没看到他的名字"
    assert event_text(s, True) is None  # find 的结果走 task_done / task_failed


def test_find_with_leave_clue_uses_lost_plan():
    a, tr = cfgs()
    s = Search.find("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.kind == "find" and presses(run(s, lambda t: Obs())) == ["left"] * 6


def test_describe_progress():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.describe() == "在找：小明（他刚从左边走了）"
    st = s.step(Obs(), 0.0)
    s.pressed(st.turn, 0.0)
    assert s.describe() == "在找：小明（他刚从左边走了，往左边转了 1 段）"
    f = Search.find("小明", None, False, True, a, tr, heading(), 0.0)
    f.step(Obs(), 0.0)
    f.call_sent(0.0)
    assert f.describe() == "在找：小明（大脑让找的，喊了一声在等）"


def test_pressed_wait_keeps_search_alive():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    for t in range(1, 15):
        s.pressed_wait(float(t))
    assert s.step(Obs(), 15.0).state == "press"
