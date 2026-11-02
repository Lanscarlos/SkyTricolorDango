import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from skydango.brain.events import Event
from skydango.brain.vision import (
    LEGEND_HEAD, MARK_KINDS, NO_BOXES, AutoVision, brain_sees, doubtful, mark_frame, parse_check, should_attach,
)
from skydango.config import BrainVisionConfig


def ev(kind):
    return Event(kind, "x", 0.0)


def frame():
    return np.zeros((1080, 1920, 3), np.uint8)


def test_numbers_left_to_right_and_legend():
    boxes = [
        {"x": 1278, "y": 331, "w": 124, "h": 157, "kind": "friend", "label": "懒洋洋大王"},
        {"x": 980, "y": 475, "w": 75, "h": 101, "kind": "self", "label": "团子"},
        {"x": 300, "y": 400, "w": 80, "h": 120, "kind": "maybe", "label": "像番茄炒蛋盖饭?"},
        {"x": 1500, "y": 500, "w": 80, "h": 120, "kind": "stranger", "label": "陌生人A"},
        {"x": 1300, "y": 250, "w": 60, "h": 60, "kind": "icon", "label": "牵手"},
    ]
    a = mark_frame(frame(), boxes, (960, 540), 80)
    assert a.legend.splitlines() == [
        LEGEND_HEAD,
        "① 像番茄炒蛋盖饭（没看到名字）",
        "② 你自己",
        "③ 懒洋洋大王（好友，看到名字）",
        "④ 图标：牵手",
        "⑤ 陌生人A",
    ]
    assert (a.w, a.h) == (960, 540) and a.src == "auto"
    assert a.block["type"] == "image" and a.block["source"]["media_type"] == "image/jpeg"
    assert a.thumb


def test_skips_tags_bubbles_rings():  # Review Focus 5
    boxes = [{"x": 10, "y": 10, "w": 50, "h": 20, "kind": k, "label": "x"} for k in ("name", "tag", "typing", "ring", "rejected")]
    assert mark_frame(frame(), boxes, (960, 540), 80).legend == NO_BOXES


def test_no_boxes_means_plain_image():
    assert mark_frame(frame(), None, (960, 540), 80).legend == NO_BOXES


def test_draws_on_copy_with_kind_color():
    f = frame()
    a = mark_frame(f, [{"x": 100, "y": 300, "w": 200, "h": 300, "kind": "friend", "label": "小明"}], (1920, 1080), 95)
    assert not f.any()  # 不改原图
    img = cv2.imdecode(np.frombuffer(base64.b64decode(a.block["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    px = img[450, 100].astype(int)
    assert np.abs(px - np.array(MARK_KINDS["friend"])).max() <= 40


def test_should_attach_triggers_and_gap():
    cfg = BrainVisionConfig()
    assert should_attach({"chat"}, False, float("-inf"), 100.0, cfg)
    assert not should_attach({"stranger", "leave"}, False, float("-inf"), 100.0, cfg)  # 只有背景事件
    assert should_attach(set(), True, float("-inf"), 100.0, cfg)  # 拿不准
    assert not should_attach({"chat"}, False, 90.0, 100.0, cfg)  # min_gap 20
    assert not should_attach({"chat"}, False, float("-inf"), 100.0, BrainVisionConfig(auto=False))


def test_doubtful():
    assert doubtful([{"kind": "unsure"}]) and doubtful([{"kind": "icon", "unknown": True}])
    assert not doubtful([{"kind": "friend"}, {"kind": "icon"}])


def test_pick_none_without_frame():  # Review Focus 3
    assert AutoVision(BrainVisionConfig(), 80).pick([ev("chat")], 100.0, None, None, False) is None


def test_pick_none_when_black():  # Review Focus 3
    assert AutoVision(BrainVisionConfig(), 80).pick([ev("chat")], 100.0, frame(), None, True) is None


def test_pick_stale_overlay_gives_plain_image_and_sets_last_at():
    env = SimpleNamespace(overlay_fresh=lambda now: False, overlay=lambda now: pytest.fail("不该读"))
    auto = AutoVision(BrainVisionConfig(), 80)
    a = auto.pick([ev("chat")], 100.0, frame(), env, False)
    assert a.legend == NO_BOXES and auto.last_at == 100.0
    assert auto.pick([ev("chat")], 110.0, frame(), env, False) is None  # 隔不到 20 秒


def test_pick_background_only_not_attached():
    auto = AutoVision(BrainVisionConfig(), 80)
    assert auto.pick([ev("stranger")], 100.0, frame(), None, False) is None
    assert auto.last_at == float("-inf")


def test_pick_doubtful_boxes_trigger():
    env = SimpleNamespace(overlay_fresh=lambda now: True,
                          overlay=lambda now: [{"x": 10, "y": 10, "w": 50, "h": 80, "kind": "unsure", "label": "可能是小明?"}])
    a = AutoVision(BrainVisionConfig(), 80).pick([ev("stranger")], 100.0, frame(), env, False)
    assert a is not None and "可能是小明（没看到名字）" in a.legend


def test_brain_sees():
    from skydango.brain.session import BrainSession

    tl = SimpleNamespace(vision_off="")
    assert brain_sees(tl, True, BrainVisionConfig(), False)
    assert not brain_sees(tl, True, BrainVisionConfig(enabled=False), False)
    assert not brain_sees(tl, True, BrainVisionConfig(), True)  # 沙盒
    assert not brain_sees(tl, False, BrainVisionConfig(), False)  # 模型没勾能看图
    assert not brain_sees(SimpleNamespace(vision_off="400"), True, BrainVisionConfig(), False)
    claude = BrainSession.__new__(BrainSession)  # Claude 大脑：不管 enabled，照旧看模型
    assert brain_sees(claude, True, BrainVisionConfig(enabled=False), False)
    assert not brain_sees(claude, False, BrainVisionConfig(), False)


def test_parse_check():
    assert parse_check("说了\n核对：A\n核对：B") == "A；B"
    assert parse_check("没有") == ""


def test_config_loads_brain_vision_section(tmp_path):
    from skydango.config import load_config

    f = tmp_path / "c.toml"
    f.write_text("[brain.vision]\nenabled = false\nsize = [640, 360]\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.brain.vision.enabled is False and cfg.brain.vision.size == [640, 360] and cfg.brain.vision.min_gap == 20.0
    assert cfg.vision.mode == "log"  # [vision]（读聊天）不受影响
