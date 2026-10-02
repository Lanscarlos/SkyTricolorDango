"""火焰圆盘（spec 2026-10-01-light-unlit-stranger §3）：黑影身上的火焰（深色实心圆 + 火焰，他举蜡烛时外面多一圈白圈）。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from skydango.config import SocialConfig
from skydango.game.social import cream, load_icons
from skydango.vision.bubbles import Rect
from skydango.vision.candle import Disk, black, find_flame, load_flame

FLAME = load_flame()
STRANGER = load_icons("assets/social")["stranger"]


def paste(frame, mask, cx, cy):
    """把米白剪影画到 (cx, cy) 为中心的位置。"""
    h, w = mask.shape
    y1, x1 = cy - h // 2, cx - w // 2
    region = frame[y1 : y1 + h, x1 : x1 + w]
    region[mask > 0] = (235, 245, 245)


def figure(ring=False, icon=None):
    """黑影（框 900,500,180,360）胸口一个深色圆盘 + 图标；ring = 外面再画一圈白色描边（举蜡烛的请求）。"""
    f = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)  # 草地
    cv2.rectangle(f, (900, 500), (1080, 860), (12, 12, 12), -1)
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)
    if ring:
        cv2.circle(f, (990, 620), 38, (240, 245, 245), 3)
    paste(f, FLAME if icon is None else icon, 990, 620)
    return f


AREA = Rect(810, 284, 360, 576)


def test_finds_dark_disk_with_flame():
    d = find_flame(figure(), AREA, FLAME)
    assert d is not None and abs(d.x - 990) <= 6 and abs(d.y - 620) <= 6 and d.score >= 0.9


def test_white_ring_still_counts():
    """白圈 = 这个黑影在举蜡烛（10-01 真机：白圈一闪一闪），他照样站在能点火的距离里，按 3 举蜡烛也安全。"""
    assert find_flame(figure(ring=True), AREA, FLAME) is not None


def test_stranger_icon_is_not_a_disk():
    assert find_flame(figure(icon=STRANGER), AREA, FLAME) is None


def test_nothing_on_plain_figure():
    f = figure()
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)  # 盖掉火焰
    assert find_flame(f, AREA, FLAME) is None


def test_disk_near_head_above_box_is_found():
    """录像 d 第 20 秒起圆盘在头顶附近（框上沿往上）。"""
    f = figure()
    cv2.circle(f, (990, 620), 52, (60, 90, 40), -1)  # 抹掉胸口那个
    cv2.circle(f, (990, 470), 52, (20, 20, 20), -1)
    paste(f, FLAME, 990, 470)
    d = find_flame(f, AREA, FLAME)
    assert d is not None and abs(d.y - 470) <= 6


# ---- 10-01 晚真机抓的帧（200×200，火焰在正中）：find_flame 不看外环亮度、不看白圈 ----
DISKS = Path(__file__).parent / "data" / "candle_disks"
CROP_AREA = Rect(0, 0, 200, 200)  # 搜索区域盖住整张小图
SURE = SocialConfig().disk_sure


@pytest.mark.parametrize("name", ["disk-plain-2206.png", "disk-white-2206.png", "disk-bright-2206.png"])
def test_real_disks_are_found_and_sure(name):
    """white：他在举蜡烛（白圈）；bright：半透明圆盘透出后面亮的火光，外环亮度 141（旧的"够暗"判断刷掉了）。"""
    d = find_flame(cv2.imread(str(DISKS / name)), CROP_AREA, FLAME)
    assert d is not None and abs(d.x - 100) <= 6 and abs(d.y - 100) <= 6 and d.score >= SURE, (name, d)


def test_lantern_is_never_sure():
    """场景里灯笼上的菱形（中间也是个黑洞）能匹配到 0.77：单帧可能算，但到不了 disk_sure，不会出请求。"""
    d = find_flame(cv2.imread(str(DISKS / "lantern-2202.png")), CROP_AREA, FLAME)
    assert d is None or d.score < SURE


# ---- 孤儿圆圈：有没有白圈（最终审查 #1：半透明圆盘透出亮背景，不能看亮度） ----

from skydango.vision.candle import ring_fraction, white_ring  # noqa: E402

RINGS = Path(__file__).parent / "data" / "candle_rings"  # 录像 candle-20260930-c 截的 140×140，圆心在 (70, 70)、圆圈半径约 50


def crop(name):
    return cv2.imread(str(RINGS / name))


def white(img, r=50):
    return white_ring(img, 70, 70, r)


def test_translucent_disk_over_bright_tower_has_no_white_ring():
    """5.3 s：圆盘透出后面亮的塔，外环亮度 97~109，旧的"够暗"判断过不了 → 被当成举蜡烛请求去点 → 团子跟着人走。"""
    for name in ("disk-c-05.3s.png", "disk-c-28.4s.png"):
        assert not white(crop(name)), name


def test_white_ring_candle_request_has_white_ring():
    for name in ("ring-c-22.5s.png", "ring-c-25.0s.png"):
        assert white(crop(name)), name


def test_ring_fraction_synthetic():
    assert ring_fraction(figure(ring=True), 990, 620, 32, 48) == 1.0
    assert ring_fraction(figure(), 990, 620, 32, 48) <= 0.2


def test_flame_outside_area_is_not_found():
    """范围由调用方给：火焰在范围外就当没有。"""
    assert find_flame(figure(), Rect(1300, 284, 300, 576), FLAME) is None


def test_area_clipped_to_frame():
    d = find_flame(figure(), Rect(700, -200, 2000, 2000), FLAME)
    assert d is not None and abs(d.x - 990) <= 6


def test_tiny_area_returns_none():
    assert find_flame(figure(), Rect(980, 610, 10, 10), FLAME) is None


def test_black_silhouette_vs_colored_person():
    f = np.full((400, 400, 3), (60, 90, 40), np.uint8)
    cv2.rectangle(f, (50, 50), (150, 350), (12, 12, 12), -1)  # 黑影
    cv2.rectangle(f, (250, 50), (350, 350), (40, 120, 220), -1)  # 点亮后：衣服有颜色
    assert black(f, Rect(50, 50, 100, 300)) > 0.9
    assert black(f, Rect(250, 50, 100, 300)) < 0.1


def test_black_uses_middle_half_and_top_70_percent():
    f = np.full((400, 400, 3), (40, 120, 220), np.uint8)
    cv2.rectangle(f, (0, 0), (24, 399), (0, 0, 0), -1)  # 框最左 1/4 黑：不在中间一半里
    cv2.rectangle(f, (0, 300), (399, 399), (0, 0, 0), -1)  # 下面 30% 黑：不算
    assert black(f, Rect(0, 0, 100, 400)) == 0.0


def test_black_empty_box_is_zero():
    assert black(np.zeros((100, 100, 3), np.uint8), Rect(200, 200, 50, 50)) == 0.0


def test_black_excludes_self_box():
    """框的左半边黑、右半边彩色（团子的身体）：不扣团子只有一半黑，扣掉后剩下的都是黑的。"""
    f = np.full((400, 400, 3), (60, 90, 40), np.uint8)
    cv2.rectangle(f, (0, 0), (99, 399), (12, 12, 12), -1)
    cv2.rectangle(f, (100, 0), (199, 399), (40, 120, 220), -1)
    box = Rect(0, 0, 200, 400)  # 区域 x 50~150
    assert black(f, box) == pytest.approx(0.5, abs=0.02)
    assert black(f, box, exclude=Rect(100, 0, 100, 400)) > 0.95


def test_black_mostly_covered_is_none():
    """团子盖住区域的 70%：剩下的不到 BLACK_MIN_VISIBLE，量不准 → None。"""
    f = np.full((400, 400, 3), (12, 12, 12), np.uint8)
    assert black(f, Rect(0, 0, 200, 400), exclude=Rect(80, 0, 100, 400)) is None


# ---- 找出范围里的几处火焰（spec 2026-10-03-light-flame-vanish §1）：跟踪"举蜡烛时那一团"、认出不动的灯笼都要看全 ----

from skydango.vision.candle import find_flames  # noqa: E402


def two_figures():
    """左边黑影胸口一团火焰（990, 620），右边另一个黑影胸口一团（1290, 600）。"""
    f = figure()
    cv2.rectangle(f, (1200, 480), (1380, 840), (12, 12, 12), -1)
    cv2.circle(f, (1290, 600), 52, (20, 20, 20), -1)
    paste(f, FLAME, 1290, 600)
    return f


WIDE = Rect(810, 284, 700, 576)


def test_find_flames_returns_every_flame_best_first():
    found = find_flames(two_figures(), WIDE, FLAME)
    assert len(found) == 2
    assert {(round(d.x, -1), round(d.y, -1)) for d in found} == {(990, 620), (1290, 600)}
    assert found[0].score >= found[1].score


def test_find_flames_one_flame_is_one_candidate():
    """同一团火焰在相邻位置、不同尺度上都有高分：只算一处。"""
    found = find_flames(figure(), AREA, FLAME)
    assert len(found) == 1 and abs(found[0].x - 990) <= 6


def test_find_flames_limit_and_empty():
    assert len(find_flames(two_figures(), WIDE, FLAME, limit=1)) == 1
    f = figure()
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)
    assert find_flames(f, AREA, FLAME) == []


def test_find_flame_is_the_best_of_find_flames():
    best = find_flame(two_figures(), WIDE, FLAME)
    assert best == find_flames(two_figures(), WIDE, FLAME)[0]
