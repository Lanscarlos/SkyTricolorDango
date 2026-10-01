"""追踪器升级（spec 2026-10-01-tracking-relink-motion §2）：两段匹配、速度预测、中心距离兜底、画面平移、dropped / prune。"""

import cv2
import numpy as np
import pytest

from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.track import Tracker, estimate_shift


def P(x, y=400, w=30, h=60, s=0.9, cls="player"):
    return Detection(cls, Rect(x, y, w, h), s)


def test_defaults_unchanged():
    t = Tracker(buffer=1.0, min_iou=0.3)
    (a,) = t.update([P(100, 100, 50, 100)], 0.0)
    (b,) = t.update([P(110, 105, 50, 100)], 0.1)
    assert a.id == b.id and b.hits == 2
    (c,) = t.update([P(110, 105, 50, 100, cls="name_tag")], 0.2)
    assert c.id != a.id
    (d,) = t.update([P(112, 105, 50, 100)], 2.0)
    assert d.id not in (a.id, c.id)
    assert d.vx == 0 and d.weak_hits == 0


def test_low_box_only_extends():
    t = Tracker(buffer=1.0, min_iou=0.3)
    (a,) = t.update([P(100)], 0.0)
    out = t.update([], 0.1, low=[P(103, s=0.3)])
    assert [x.id for x in out] == [a.id] and a.weak_hits == 1 and a.box.x == 103
    assert t.update([], 0.2, low=[P(900, s=0.3)]) == []
    assert len(t.tracks) == 1


def _walk(tracker, n=5, step=15, dt=0.15):
    ids = []
    for i in range(n):
        (tr,) = tracker.update([P(100 + i * step, w=20, h=40)], i * dt)
        ids.append(tr.id)
    return ids


def test_center_gate_links_small_box_moving_past_iou():
    assert len(set(_walk(Tracker()))) > 1  # 20 px 宽的框挪 15 px：IoU 掉到 0.3 以下
    assert len(set(_walk(Tracker(center_gate=0.6, predict=True)))) == 1


def test_iou_candidate_beats_center_candidate():
    t = Tracker(center_gate=0.6, predict=True)
    (a,) = t.update([P(100, h=60)], 0.0)
    near = P(117)  # 横移 17 px：IoU 0.28，只能走中心距离
    good = P(84)  # 横移 16 px：IoU 0.30，刚过线
    out = t.update([near, good], 0.1)
    assert out[1].id == a.id and out[0].id != a.id


def test_shift_keeps_track_when_whole_frame_pans():
    t = Tracker()
    (a,) = t.update([P(100)], 0.0)
    (b,) = t.update([P(400)], 0.1, shift=(300, 0))
    assert b.id == a.id
    t2 = Tracker()
    (c,) = t2.update([P(100)], 0.0)
    (d,) = t2.update([P(400)], 0.1)
    assert d.id != c.id


def test_prediction_capped_at_half_second():
    t = Tracker(predict=True)
    (a,) = t.update([P(100)], 0.0)
    a.vx = 100.0
    assert t.predicted(a, 2.0)[0].x == 150
    assert Tracker().predicted(a, 2.0)[0].x == 100  # 不预测：就是最后的框


def test_velocity_updates_and_not_after_gap_or_calm():
    t = Tracker(predict=True)
    (a,) = t.update([P(100)], 0.0)
    t.update([P(110)], 0.1)
    assert a.vx == pytest.approx(50.0)  # 样本 100 px/s，α = 0.5 从 0 起
    t.update([P(130)], 0.9)  # 隔了 0.8 秒：不更新速度
    assert a.vx == pytest.approx(50.0)
    t.calm(1.3)
    assert a.vx == a.vy == a.vh == 0
    t.update([P(135)], 1.0)  # calm 期间
    assert a.vx == 0
    t.update([P(140)], 1.6)  # 隔 0.6 秒
    assert a.vx == 0
    t.update([P(150)], 1.7)
    assert a.vx == pytest.approx(50.0)


def test_dropped_and_prune_false():
    t = Tracker(buffer=1.0)
    (a,) = t.update([P(100)], 0.0)
    assert t.update([], 2.0, prune=False) == []
    assert a.id in t.tracks and t.dropped == []
    t.update([], 2.0)
    assert a.id not in t.tracks and t.dropped == [a]
    t.update([], 2.1)
    assert t.dropped == []


def test_prune_false_keeps_previous_dropped():
    t = Tracker(buffer=1.0)
    (a,) = t.update([P(100)], 0.0)
    t.update([], 2.0)
    t.update([P(900)], 2.0, prune=False)
    assert t.dropped == [a]


def test_shift_accumulates_on_unmatched_track():
    t = Tracker()
    (a,) = t.update([P(100)], 0.0)
    t.update([], 0.1, shift=(10, 0))
    t.update([], 0.2, shift=(5, 0))
    assert a.pan == (15, 0)
    (b,) = t.update([P(118)], 0.3, shift=(0, 0))
    assert b.id == a.id and a.pan == (0, 0)


def test_velocity_excludes_pan():
    t = Tracker(predict=True)
    (a,) = t.update([P(100)], 0.0)
    t.update([P(140)], 0.1, shift=(40, 0))  # 全是镜头转的：人没动
    assert a.vx == pytest.approx(0.0)


# ---- 画面平移估计（§2.4） ----
def _texture(w=240, h=67, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 40, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(img, (5, 5), 1.5)


def test_shift_estimates_synthetic_pan():
    prev = _texture()
    cur = np.roll(prev, 7, axis=1)
    dx, dy = estimate_shift(prev, cur, None)
    assert abs(dx - 7) < 1 and abs(dy) < 1


def test_shift_none_on_flat_image():
    flat = np.full((67, 240), 128, np.float32)
    assert estimate_shift(flat, flat.copy(), None) is None
    black = np.zeros((67, 240), np.float32)
    assert estimate_shift(black, black.copy(), None) is None


def test_shift_ignores_masked_region():
    prev = _texture(seed=1)
    cur = prev.copy()
    prev[20:50, 10:40] = 255
    cur[20:50, 30:60] = 255  # 左 1/3 里一块往右挪了 20 px（人在走），背景没动
    mask = np.ones(prev.shape, bool)
    mask[:, :80] = False
    dx, dy = estimate_shift(prev, cur, mask)
    assert abs(dx) < 1 and abs(dy) < 1


def test_shift_none_when_too_large():
    prev = _texture(seed=2)
    assert estimate_shift(prev, np.roll(prev, 100, axis=1), None) is None


# ---- 最终评审的修复 ----
def test_tie_order_unchanged_when_defaults():
    """开关全关逐字照旧：IoU 一样时还是序号大的检测先配（原来的 sort(reverse=True)）。"""
    t = Tracker()
    (a,) = t.update([P(100)], 0.0)
    out = t.update([P(85), P(115)], 0.1)
    assert out[1].id == a.id and out[0].id != a.id


def test_velocity_not_polluted_when_person_moves_with_camera():
    """镜头绕着团子转，身边的人在屏幕上不动（配上的是不带平移的预测框）：速度不该减掉背景平移。"""
    t = Tracker(predict=True, center_gate=0.6)
    (a,) = t.update([P(300, w=90, h=220)], 0.0)
    for i in range(1, 4):
        (b,) = t.update([P(300, w=90, h=220)], 0.15 * i, shift=(120, 0))
        assert b.id == a.id
    assert abs(a.vx) < 50 and a.drift == (0.0, 0.0)


def test_drift_accumulates_when_panned_box_wins():
    t = Tracker()
    (a,) = t.update([P(100)], 0.0)
    t.update([P(140)], 0.1, shift=(40, 0))
    assert a.drift == (40.0, 0.0)


def test_low_stage_is_iou_only():
    t = Tracker(center_gate=0.6, predict=True)
    t.update([P(100, w=20, h=40)], 0.0)
    assert t.update([], 0.15, low=[P(117, w=20, h=40, s=0.3)]) == []  # 中心距离够、IoU 不够：低分框不续


def test_strong_last_only_moves_on_high_score_hits():
    t = Tracker()
    (a,) = t.update([P(100)], 0.0)
    t.update([], 0.1, low=[P(101, s=0.3)])
    assert a.strong_last == 0.0 and a.last == 0.1


def test_shift_rejects_unrelated_images_with_strict_threshold():
    a, b = _texture(seed=11), _texture(seed=12)
    assert estimate_shift(a, b, None, min_response=0.4) is None
    assert estimate_shift(a, np.roll(a, 9, axis=1), None, min_response=0.4) is not None
