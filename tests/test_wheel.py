import cv2
import numpy as np
import pytest
from conftest import FakeDevice

from skydango.config import WheelConfig
from skydango.game.wheel import SLOT_SCALES, EmoteLibrary, Wheel, WheelError
from skydango.imageio import imwrite
from skydango.vision.icons import same_icon, silhouette, trim

CREAM = (220, 235, 245)


def icon(kind, size=90, scale=1.0):
    """深色格子里画一个米白色剪影。"""
    img = np.full((size, size, 3), 30, np.uint8)
    c, r = size // 2, int(28 * scale)
    if kind == "circle":
        cv2.circle(img, (c, c), r, CREAM, -1)
    elif kind == "cross":
        cv2.rectangle(img, (c - r, c - 6), (c + r, c + 6), CREAM, -1)
        cv2.rectangle(img, (c - 6, c - r), (c + 6, c + r), CREAM, -1)
    elif kind == "person":
        cv2.circle(img, (c, c - r + 8), 9, CREAM, -1)
        cv2.rectangle(img, (c - 8, c - r + 20), (c + 8, c + r), CREAM, -1)
        cv2.line(img, (c, c - 5), (c + r, c - r + 5), CREAM, 6)
    return img


@pytest.fixture
def library(tmp_path):
    for name, kind in (("鞠躬", "circle"), ("欢呼", "cross"), ("指向", "person")):
        imwrite(tmp_path / f"{name}.png", icon(kind))
    return EmoteLibrary(tmp_path)


def test_slot_geometry_matches_calibrated_editor():
    wheel = Wheel(FakeDevice([np.zeros((1080, 1920, 3), np.uint8)]), WheelConfig(), EmoteLibrary("missing"))
    x, y = wheel.slot_center(5, 1920, 1080)  # 正下方，实测 (643, 735)
    assert abs(x - 643) <= 3 and abs(y - 735) <= 5
    x, y = wheel.slot_center(1, 1920, 1080)  # 正上方
    assert abs(x - 643) <= 3 and y < 540
    x, y = wheel.slot_center(3, 1920, 1080)  # 右
    assert x > 643 and abs(y - 540) <= 3


def test_library_identifies_bigger_icon_in_wheel(library):
    assert sorted(library.names) == ["指向", "欢呼", "鞠躬"]
    big = icon("person", size=130, scale=1.25)  # 轮盘里的图标比列表大一圈
    name, score = library.identify(big, SLOT_SCALES, 0.8)
    assert name == "指向" and score >= 0.8
    name, _ = library.identify(np.full((130, 130, 3), 30, np.uint8), SLOT_SCALES, 0.8)
    assert name is None


def test_same_icon():
    a, b = trim(silhouette(icon("circle"))), trim(silhouette(icon("cross")))
    assert same_icon(a, a, 0.9)
    assert not same_icon(a, b, 0.9)


def make_wheel(library, shown=False):
    device = FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])
    device.shown = shown
    t = [100.0]
    wheel = Wheel(device, WheelConfig(), library, sleep=lambda s: None, clock=lambda: t[0])
    wheel.slots = {1: "鞠躬", 2: "欢呼", 3: None, 4: None, 5: None, 6: None, 7: None, 8: None}
    return wheel, device, t


def test_perform_known_emote_presses_number_key(library):
    wheel, device, _ = make_wheel(library)
    assert wheel.perform("欢呼") == 2
    assert device.calls == [("hw_key", 3)]  # KEY_2


def test_perform_refuses_when_typing(library):
    wheel, device, _ = make_wheel(library, shown=True)
    with pytest.raises(WheelError, match="输入框"):
        wheel.perform("鞠躬")
    assert device.calls == []


def test_victim_skips_locked_and_prefers_least_recent(library):
    wheel, _, t = make_wheel(library)
    assert 3 not in wheel.free_slots() and 8 not in wheel.free_slots()
    assert wheel._victim() == 1  # 都没用过：按编号
    wheel.last_used = {s: 100.0 + s for s in wheel.free_slots()}
    wheel.last_used[6] = 50.0
    assert wheel._victim() == 6


def test_assign_rejects_locked_and_unknown(library):
    wheel, _, _ = make_wheel(library)
    with pytest.raises(WheelError, match="锁定"):
        wheel.assign(3, "鞠躬")
    with pytest.raises(WheelError, match="图标库"):
        wheel.assign(5, "不存在")


def test_library_skips_underscore_files(tmp_path):
    imwrite(tmp_path / "鞠躬.png", icon("circle"))
    imwrite(tmp_path / "_sheet.png", icon("cross"))
    assert EmoteLibrary(tmp_path).names == ["鞠躬"]


def test_victim_limited_to_candidates(library):
    wheel, _, _ = make_wheel(library)
    assert wheel._victim([7, 3]) == 7  # 3 被锁定
    with pytest.raises(WheelError, match="锁定"):
        wheel._victim([3])


def test_press_slot_sends_its_digit_key(library):
    wheel, device, _ = make_wheel(library)
    wheel.press(3)  # 3 号格是锁定格（举蜡烛），也能按
    assert device.calls == [("hw_key", 4)]  # KEY_3


def test_press_refuses_when_input_box_open(library):
    wheel, device, _ = make_wheel(library, shown=True)
    with pytest.raises(WheelError, match="输入框"):
        wheel.press(3)
