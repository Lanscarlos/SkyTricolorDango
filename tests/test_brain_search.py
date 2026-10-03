"""有意识地找：纯计算（plan 2026-10-03-attention-search Task 2 / 3）。画面宽 1920，hfov 90°，press_deg 18°。"""

import pytest

from skydango.brain.search import SECTORS, Heading, bearing_name
from skydango.config import Config


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
