"""火焰圆盘（spec 2026-10-01-light-unlit-stranger §3）：黑影身上的火焰（深色实心圆 + 火焰，他举蜡烛时外面多一圈白圈）。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from skydango.config import SocialConfig
from skydango.game.social import cream, load_icons
from skydango.vision.bubbles import Rect
from skydango.vision.candle import find_disk, load_flame

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


BOX = Rect(900, 500, 180, 360)


def test_finds_dark_disk_with_flame():
    d = find_disk(figure(), BOX, FLAME)
    assert d is not None and abs(d.x - 990) <= 6 and abs(d.y - 620) <= 6 and d.score >= 0.9


def test_white_ring_still_counts():
    """白圈 = 这个黑影在举蜡烛（10-01 真机：白圈一闪一闪），他照样站在能点火的距离里，按 3 举蜡烛也安全。"""
    assert find_disk(figure(ring=True), BOX, FLAME) is not None


def test_stranger_icon_is_not_a_disk():
    assert find_disk(figure(icon=STRANGER), BOX, FLAME) is None


def test_nothing_on_plain_figure():
    f = figure()
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)  # 盖掉火焰
    assert find_disk(f, BOX, FLAME) is None


def test_disk_near_head_above_box_is_found():
    """录像 d 第 20 秒起圆盘在头顶附近（框上沿往上）。"""
    f = figure()
    cv2.circle(f, (990, 620), 52, (60, 90, 40), -1)  # 抹掉胸口那个
    cv2.circle(f, (990, 470), 52, (20, 20, 20), -1)
    paste(f, FLAME, 990, 470)
    d = find_disk(f, BOX, FLAME)
    assert d is not None and abs(d.y - 470) <= 6


# ---- 10-01 晚真机抓的帧（200×200，火焰在正中）：find_disk 不看外环亮度、不看白圈 ----
DISKS = Path(__file__).parent / "data" / "candle_disks"
CROP_BOX = Rect(50, 80, 100, 120)  # 搜索区域盖住整张小图
SURE = SocialConfig().disk_sure


@pytest.mark.parametrize("name", ["disk-plain-2206.png", "disk-white-2206.png", "disk-bright-2206.png"])
def test_real_disks_are_found_and_sure(name):
    """white：他在举蜡烛（白圈）；bright：半透明圆盘透出后面亮的火光，外环亮度 141（旧的"够暗"判断刷掉了）。"""
    d = find_disk(cv2.imread(str(DISKS / name)), CROP_BOX, FLAME)
    assert d is not None and abs(d.x - 100) <= 6 and abs(d.y - 100) <= 6 and d.score >= SURE, (name, d)


def test_lantern_is_never_sure():
    """场景里灯笼上的菱形（中间也是个黑洞）能匹配到 0.77：单帧可能算，但到不了 disk_sure，不会出请求。"""
    d = find_disk(cv2.imread(str(DISKS / "lantern-2202.png")), CROP_BOX, FLAME)
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
