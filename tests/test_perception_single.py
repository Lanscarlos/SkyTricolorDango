"""感知层的单类模式（YOLO 只框人，spec 2026-11-04-yolo-boxes-only §3）：点没点火来自外形头 / 亮度，不来自 YOLO 类别。"""

import numpy as np

from skydango.config import AttrsConfig, EnvConfig, PerceptionConfig
from skydango.vision.attrs import PersonAttrs
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.perception import LIT_LOW, PerceptionWatcher

from test_perception import FRIENDS, FakeDetector, FakeOcr, frame, player
from test_perception_attrs import FakeModel, probs, scene


class SingleDetector(FakeDetector):
    single = True


def watcher(det, model=None, attrs_cfg=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    acfg = attrs_cfg or AttrsConfig(enabled=True, every=0.0)
    attrs = PersonAttrs(acfg, model) if model is not None else None
    return PerceptionWatcher(
        det, FakeOcr({}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, attrs=attrs, attrs_cfg=acfg,
    )


def lit_frame(x=1500, y=400, w=90, h=220):
    img = frame()
    img[y:y + h, x:x + w] = (60, 200, 90)  # 彩色：V 远高于 lit_v
    return img


def run(w, img, times):
    for t in times:
        w.process(img, t, panel_visible=False)


def test_single_mode_detected():
    det = SingleDetector()
    det.frames = [[player(1500)]]
    model = FakeModel({})
    w = watcher(det, model)
    assert w.single and w.attrs.single
    assert not watcher(FakeDetector()).single  # 旧模型没有 single 属性


def test_single_dark_vote_marks_unlit():
    det = SingleDetector()
    det.frames = [[player(1500)]]
    w = watcher(det)  # 外形头没开
    run(w, frame(), (0.0, 0.5, 1.0, 1.5, 2.0))  # 全黑的画面：框里很暗
    assert w.unlit(2.0) == 1 and w.strangers(2.0) == 1


def test_single_lit_not_unlit():
    det = SingleDetector()
    det.frames = [[player(1500)]]
    w = watcher(det)
    run(w, lit_frame(), (0.0, 0.5, 1.0, 1.5, 2.0))
    assert w.unlit(2.0) == 0 and w.strangers(2.0) == 1


def test_single_attrs_dies_falls_back_to_dark():
    """外形头先说是黑影（画面也很暗），然后自己关掉：unlit 不丢，之后改按亮度投票，人亮起来后能翻回点亮。"""
    img, model = scene({1500: probs(unlit=0.95)})
    img[400:620, 1500:1590] = 1  # scene 涂的灰度 1：很暗
    det = SingleDetector()
    det.frames = [[player(1500)]]
    w = watcher(det, model, attrs_cfg=AttrsConfig(enabled=True, every=0.0, votes=3, flip_votes=2))
    run(w, img, (0.0, 0.5, 1.0))
    (t,) = [x for x in w.last_tracks if x.cls == "player"]
    assert t.data["unlit"] is True and w._unlit(t)
    w.attrs.enabled = False  # 连续出错自己关了
    run(w, img, (1.5, 2.0))
    assert t.data["unlit"] is True and w._unlit(t) and w.unlit(2.0) == 1
    run(w, lit_frame(), (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5))
    assert t.data["unlit"] is False and not w._unlit(t)


def test_single_people_boxes_flag_from_unlit():
    det = SingleDetector()
    low = Detection("player", Rect(300, 500, 60, 150), max(LIT_LOW, 0.3))
    det.frames = [[player(1500), low]]
    w = watcher(det)
    run(w, frame(), (0.0, 0.5, 1.0, 1.5))
    flags = {b.x: u for b, u in w._people_boxes}
    assert flags[1500] is True  # 黑影轨迹的框
    assert flags.get(300, False) is False  # 低分框没有轨迹：一律 False


def test_mode_note():
    det = SingleDetector()
    det.frames = [[player(1500)]]
    assert watcher(det).mode_note() == "单类模式；点没点火按亮度估（外形头没开）"
    w = watcher(det, FakeModel({}))
    assert w.mode_note() == "单类模式"
    w.attrs.enabled = False
    assert "亮度" in w.mode_note()
    assert watcher(FakeDetector()).mode_note() is None


def test_old_model_unchanged():
    det = FakeDetector()
    det.frames = [[Detection("player_unlit", Rect(300, 500, 60, 150), 0.9)]]
    w = watcher(det)
    run(w, lit_frame(300, 500, 60, 150), (0.0, 0.5, 1.0, 1.5))
    assert w.unlit(1.5) == 1  # 画面再亮也按 YOLO 类别判
    assert [u for _, u in w._people_boxes] == [True]
