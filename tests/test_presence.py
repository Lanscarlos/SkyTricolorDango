"""好友在不在场（spec 2026-10-06-friend-presence §1）：身边 / 附近 / 找不到 / 走开。"""

from skydango.config import PerceptionConfig
from skydango.vision.people import Seen
from skydango.vision.presence import Presence

OUT = Seen("右边", None, on_screen=False)


def make(can_call=True, **cfg):
    seen: dict[str, float] = {}
    return Presence(PerceptionConfig(**cfg), seen, can_call), seen


def test_in_view_then_near_by_edge():
    pr, seen = make()
    seen["小明"] = 0.0
    assert pr.state("小明", 3) == "view"
    pr.edge("小明", "右边", 4)
    assert pr.state("小明", 8) == "near"
    assert pr.around(["小明"], 8) == [("小明", "画面外·右边")]
    assert pr.present(["小明"], 8) == ["小明"]
    assert pr.in_view(["小明"], 8) == []


def test_view_beats_edge():
    pr, seen = make()
    seen["小明"] = 4.0
    pr.edge("小明", "左边", 4)
    assert pr.state("小明", 4) == "view"
    assert pr.around(["小明"], 4) == []


def test_lost_needs_call_then_left_after_failed_call():
    pr, seen = make()
    seen["小明"] = 0.0
    assert pr.state("小明", 6) == "lost"
    assert pr.need_call(["小明"], 6) == ["小明"]
    assert pr.present(["小明"], 6) == ["小明"]  # 找不到还不算走开
    assert pr.around(["小明"], 6) == [("小明", "看不到了，在确认（6 秒）")]
    pr.call_done(6, {})
    assert pr.need_call(["小明"], 7) == []
    assert pr.state("小明", 14) == "lost"
    assert pr.state("小明", 15) == "left"
    assert pr.left_note("小明", 15) == "喊了一声也没看到，15 秒了"
    assert pr.need_call(["小明"], 15) == []
    assert pr.present(["小明"], 15) == []


def test_no_call_waits_confirm_max():
    pr, seen = make()
    seen["小明"] = 0.0
    assert pr.state("小明", 30) == "lost"
    assert pr.state("小明", 60) == "left"
    assert pr.left_note("小明", 60) == "60 秒没看到，没喊成"


def test_cannot_call_leaves_after_leave_after():
    pr, seen = make(can_call=False)
    seen["小明"] = 0.0
    assert pr.need_call(["小明"], 6) == []
    assert pr.state("小明", 14) == "lost"
    assert pr.state("小明", 15) == "left"
    assert pr.left_note("小明", 15) == "15 秒没看到"


def test_call_found_confirms_for_recheck():
    pr, seen = make()
    seen["小明"] = 0.0
    pr.call_done(6, {"小明": OUT})
    assert pr.state("小明", 95) == "near"
    assert pr.around(["小明"], 95) == [("小明", "画面外·右边，1 分钟前喊到")]
    assert pr.confirmed("小明") == 6
    assert pr.state("小明", 97) == "lost"
    assert pr.need_call(["小明"], 97) == ["小明"]
    pr.call_done(97, {})
    assert pr.state("小明", 110) == "lost"
    assert pr.state("小明", 111) == "left"


def test_found_on_screen_is_far():
    pr, seen = make()
    seen["小明"] = 0.0
    pr.call_done(6, {"小明": Seen("左边", "远")})
    assert pr.around(["小明"], 10) == [("小明", "远处，刚喊到")]


def test_call_before_lost_does_not_count():
    pr, seen = make()
    seen["小明"] = 5.0
    pr.call_done(3, {})
    assert pr.state("小明", 20) == "lost"
    assert pr.need_call(["小明"], 20) == ["小明"]


def test_shift_delays_everything():
    pr, seen = make()
    seen["小明"] = 0.0
    pr.call_done(6, {"小明": OUT})
    pr.shift(100, 150)
    assert pr.confirmed("小明") == 106
    assert pr.state("小明", 190) == "near"


def test_comes_back_after_left():
    pr, seen = make()
    seen["小明"] = 0.0
    pr.call_done(6, {})
    assert pr.state("小明", 30) == "left"
    seen["小明"] = 40.0
    assert pr.state("小明", 41) == "view"
    assert pr.present(["小明"], 41) == ["小明"]


def test_unknown_name_is_empty():
    pr, _ = make()
    assert pr.state("阿花", 0) == ""
    assert pr.present(["阿花"], 0) == []
    assert pr.need_call(["阿花"], 0) == []


def test_config_defaults():
    p = PerceptionConfig()
    assert (p.presence, p.leave_after, p.recheck, p.confirm_max, p.keep) == (True, 15.0, 90.0, 60.0, 5.0)


def test_edge_or_call_alone_does_not_start_tracking():
    pr, seen = make()
    pr.edge("小明", "右边", 1)  # 这次运行还没在画面里见过他：不算来了
    pr.call_done(2, {"阿花": OUT})
    assert pr.present(["小明", "阿花"], 3) == []
    seen["小明"] = 5.0
    pr.edge("小明", "右边", 10)
    assert pr.state("小明", 12) == "near"


def test_call_found_while_in_view_does_not_extend_near():
    pr, seen = make()
    seen["小明"] = 0.0
    pr.call_done(0, {"小明": Seen("前面", "近")})  # 喊的时候他就在画面里
    seen["小明"] = 3.0  # 窗口里还看得到
    assert pr.state("小明", 9) == "lost"  # 走出画面后不靠这次"喊到"多挂 90 秒
    assert pr.need_call(["小明"], 9) == ["小明"]

