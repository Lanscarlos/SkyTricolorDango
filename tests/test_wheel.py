from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import FakeDevice

from skydango.config import WheelConfig
from skydango.game.wheel import SLOT_SCALES, EmoteLibrary, Wheel, WheelError
from skydango.imageio import imread, imwrite
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


def test_slot_threshold_has_margin_over_measured_scores():
    """10-02 真机实测（tmp/manual-wheel-before.txt）：欢呼在格子里只有 0.72、空 / 锁定格子 0.47；
    门槛原来正好是 0.72，第二次启动就认不出欢呼。两边都要留余量。"""
    t = WheelConfig().slot_match_threshold
    assert 0.47 + 0.08 <= t <= 0.72 - 0.08


def test_refresh_logs_every_slot_score(library, caplog, monkeypatch):
    wheel, _, _ = make_wheel(library)
    monkeypatch.setattr(wheel, "open_editor", lambda: np.zeros((1080, 1920, 3), np.uint8))
    monkeypatch.setattr(wheel, "close_editor", lambda: None)
    scores = {s: (None, 0.47) for s in range(1, 9)}
    scores[1], scores[6] = ("鞠躬", 0.93), ("欢呼", 0.72)
    monkeypatch.setattr(wheel, "read_slots", lambda frame: scores)
    with caplog.at_level("INFO", logger="skydango.game.wheel"):
        wheel.refresh()
    assert "1 鞠躬 0.93" in caplog.text and "6 欢呼 0.72" in caplog.text and "3 ? 0.47" in caplog.text
    assert wheel.slots[6] == "欢呼" and wheel.slots[3] is None


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _diamond():
    img = np.full((90, 90, 3), 30, np.uint8)
    cv2.fillPoly(img, [np.array([(45, 12), (78, 45), (45, 78), (12, 45)])], CREAM)
    return img


def _list_frames(cfg, recent=True):
    """合成动作列表：最上面是「最近使用」，两行、每个图标右下角贴着真机截下来的时钟角标（第二行有个只在这里出现的菱形）；
    recent=False 时最上面一行是不带角标的菱形（没有最近使用）。下面是动作网格。
    按真机的样子翻页：长画布每屏滚 400 像素，最后一屏到底重复一次。"""
    height, width = 1080, 1920
    x1, y1 = int(cfg.list_roi[0] * width), int(cfg.list_roi[1] * height)
    canvas = np.zeros((2000, int(cfg.list_roi[2] * width) - x1, 3), np.uint8)
    clock = imread(ASSETS / "emotes" / "recent_clock.png")

    def put(img, cx, cy):
        h, w = img.shape[:2]
        canvas[cy - h // 2 : cy - h // 2 + h, cx - w // 2 : cx - w // 2 + w] = img

    if recent:
        for cy, row in ((130, [icon("person"), icon("circle")]), (250, [_diamond(), icon("cross")])):
            for col, img in enumerate(row):
                put(img, 60 + 119 * col, cy)
                put(clock, 60 + 119 * col + 40, cy + 33)  # 真机上角标在图标中心右下 (40, 33)
    else:
        put(_diamond(), 180, 130)
    kinds = ["circle", "cross", "person"]
    for row, cy in enumerate(range(400, 1950, 119)):
        for col in range(4):
            put(icon(kinds[(row + col) % 3]), 60 + 119 * col, cy)

    def page(offset):
        frame = np.zeros((height, width, 3), np.uint8)
        part = canvas[offset : offset + height - y1]
        frame[y1 : y1 + part.shape[0], x1 : x1 + part.shape[1]] = part
        return frame

    top, mid, bottom = page(0), page(400), page(2000 - (height - y1))
    return [top, top, top, mid, bottom, bottom]


def _scan(library, tmp_path, monkeypatch, recent):
    cfg = WheelConfig(recent_badge=str(ASSETS / "emotes" / "recent_clock.png"))
    wheel = Wheel(FakeDevice(_list_frames(cfg, recent)), cfg, library, sleep=lambda s: None)
    monkeypatch.setattr(wheel, "open_editor", lambda: None)
    monkeypatch.setattr(wheel, "close_editor", lambda: None)
    paths = wheel.scan_list(tmp_path / "scan")
    return [trim(silhouette(imread(p))) for p in paths]


def test_scan_list_skips_recently_used_rows(library, tmp_path, monkeypatch):
    """「最近使用」是后面动作的重复，带时钟角标、截歪了，有一行也可能两行（10-03 真机：4 + 2 个，
    只跳第一行时第二行的先祖群、跪坐漏进来成了 001 / 002），带角标的几行都不要。"""
    masks = _scan(library, tmp_path, monkeypatch, recent=True)
    assert not any(same_icon(m, trim(silhouette(_diamond())), 0.9) for m in masks)
    assert len(masks) == 3  # 圆、十字、人形各一张


def test_scan_list_keeps_top_row_without_recent_badges(library, tmp_path, monkeypatch):
    masks = _scan(library, tmp_path, monkeypatch, recent=False)
    assert any(same_icon(m, trim(silhouette(_diamond())), 0.9) for m in masks)
    assert len(masks) == 4


def test_scan_list_moves_previous_scan_into_old(library, tmp_path, monkeypatch):
    """重扫只覆盖 001~N：上次多出来的编号图会留下来混进「动作名」页（10-03 晚 178~182）。扫之前整批挪进 _old/<时间>/。"""
    out = tmp_path / "scan"
    out.mkdir()
    for name in ("001.png", "200.png", "_sheet.png"):
        imwrite(out / name, icon("circle"))
    (out / "说明.txt").write_text("留着", encoding="utf-8")
    cfg = WheelConfig(recent_badge=str(ASSETS / "emotes" / "recent_clock.png"))
    wheel = Wheel(FakeDevice(_list_frames(cfg)), cfg, library, sleep=lambda s: None)
    monkeypatch.setattr(wheel, "open_editor", lambda: None)
    monkeypatch.setattr(wheel, "close_editor", lambda: None)
    paths = wheel.scan_list(out)
    assert sorted(p.name for p in out.glob("*.png")) == ["001.png", "002.png", "003.png", "_sheet.png"]
    assert len(paths) == 3
    backups = list((out / "_old").iterdir())
    assert len(backups) == 1
    assert sorted(p.name for p in backups[0].iterdir()) == ["001.png", "200.png", "_sheet.png"]
    assert (out / "说明.txt").exists()
