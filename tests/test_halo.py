"""呼唤光圈认团子的纯计算（spec 2026-10-01-q-call §2.2）。"""

import numpy as np

from skydango.config import CallConfig
from skydango.vision.bubbles import Rect
from skydango.vision.halo import HaloWatch, head_region

W, H = 1920, 1080
MID = Rect(940, 500, 60, 150)  # 在画面中间
LEFT = Rect(100, 500, 60, 150)  # 中心离中线 830 px > 0.35 × 1920


def gray(v=80):
    return np.full((H, W, 3), v, np.uint8)


def lit(boxes, v=80, rise=60, whole=0):
    """画面整体 v + whole，boxes 头顶区域再亮 rise。"""
    f = gray(v + whole)
    for b in boxes:
        r = head_region(b, W, H)
        f[r.y:r.y2, r.x:r.x2] = v + whole + rise
    return f


def watch(at=10.0, base=None, **cfg):
    return HaloWatch(base if base is not None else gray(), {1: MID, 2: LEFT}, at, CallConfig(**cfg))


def test_head_region_above_and_clamped():
    assert head_region(Rect(100, 200, 40, 100), W, H) == Rect(70, 150, 100, 80)
    assert head_region(Rect(0, 10, 40, 100), W, H) == Rect(0, 0, 70, 40)
    assert head_region(Rect(5, 0, 0, 0), W, H) is None


def test_single_center_halo_is_self():
    w = watch()
    w.add(gray(), 10.05)
    w.add(lit([MID]), 10.3)
    w.add(gray(), 10.6)
    assert w.result(W) == ("self", 1)
    assert w.peaks[1] >= 59 and abs(w.peaks[2]) < 1


def test_two_halos_are_others():
    w = watch()
    w.add(lit([MID, LEFT]), 10.3)
    assert w.result(W) == ("others", None)


def test_lit_in_base_does_not_count():
    base = lit([MID])
    w = watch(base=base)
    w.add(base.copy(), 10.3)
    assert w.result(W) == ("none", None)


def test_whole_frame_brightening_cancels():
    w = watch()
    w.add(gray(140), 10.3)
    assert w.result(W) == ("none", None)


def test_peak_outside_window_ignored():
    w = watch()
    w.add(lit([MID]), 9.95)  # 按键之前
    w.add(lit([MID]), 10.9)
    w.add(gray(), 10.5)
    assert w.result(W) == ("none", None)


def test_off_center_single_halo_is_not_self():
    w = watch()
    w.add(lit([LEFT]), 10.3)
    assert w.result(W) == ("others", None)


def test_second_person_half_risen_is_ambiguous():
    w = watch(halo_rise=25.0)
    f = lit([MID])
    r = head_region(LEFT, W, H)
    f[r.y:r.y2, r.x:r.x2] = 80 + 15  # 一半以上、门槛以下
    w.add(f, 10.3)
    assert w.result(W) == ("others", None)


def test_skipped_flag():
    w = watch()
    w.add(lit([MID]), 10.3)
    w.skipped = True
    assert w.result(W) == ("skipped", None)


def test_no_boxes_is_skipped_not_none():
    w = HaloWatch(gray(), {}, 10.0, CallConfig())
    w.add(lit([MID]), 10.3)
    assert w.result(W) == ("skipped", None)  # 没有能看的人：说不准，别报"没看到"


def test_peak_right_after_press_counts():
    w = watch()
    w.add(lit([MID]), 10.05)  # 按下后 0.05 s 就亮了（at 取的是按下之前）
    assert w.result(W) == ("self", 1)
