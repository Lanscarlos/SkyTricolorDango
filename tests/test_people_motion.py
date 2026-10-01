"""运动方向（spec 2026-10-01-tracking-relink-motion §5）：纯函数和 status 的写法。"""

import pytest

from skydango.config import PerceptionConfig
from skydango.vision.bubbles import Rect
from skydango.vision.people import Person, describe_people
from skydango.vision.perception import motion_of, settle_motion

CFG = PerceptionConfig()


def hist(h0, h1, x0=500.0, x1=500.0, n=11, span=1.5):
    return [(span * i / (n - 1), h0 + (h1 - h0) * i / (n - 1), x0 + (x1 - x0) * i / (n - 1)) for i in range(n)]


@pytest.mark.parametrize("h,expect", [
    (hist(100, 130), "走近"),
    (hist(100, 70), "走远"),
    (hist(100, 100, 500, 650), "往右走"),
    (hist(100, 100, 500, 350), "往左走"),
    (hist(100, 102, 500, 510), "站着"),
])
def test_motion_of_five_outcomes(h, expect):
    assert motion_of(h, 1.5, CFG) == expect


def test_motion_of_needs_samples():
    assert motion_of(hist(100, 130)[:2], 1.5, CFG) is None
    assert motion_of(hist(100, 130, span=0.6), 0.6, CFG) is None  # 只覆盖了 0.6 秒（< 窗口 60%）
    assert motion_of(hist(100, 130), 5.0, CFG) is None  # 都过了窗口


def test_settle_motion_debounce_and_none_immediate():
    d = {}
    assert settle_motion(d, "站着", 0.0, 0.5) is None
    assert settle_motion(d, "站着", 0.3, 0.5) is None
    assert settle_motion(d, "站着", 0.6, 0.5) == "站着"
    assert settle_motion(d, "走近", 1.0, 0.5) == "站着"
    assert settle_motion(d, "走远", 1.2, 0.5) == "站着"  # 换了新结论：重新计时
    assert settle_motion(d, "走远", 1.5, 0.5) == "站着"
    assert settle_motion(d, "走远", 1.7, 0.5) == "走远"
    assert settle_motion(d, None, 1.8, 0.5) is None


def P(**kw):
    base = dict(track_id=1, kind="friend", name="小明", box=Rect(0, 0, 10, 10), side="左边", distance="中")
    return Person(**(base | kw))


def test_describe_person_motion():
    assert describe_people([P(motion="走远")]) == "小明（左边·中，正在走远）"
    assert describe_people([P(motion="站着")]) == "小明（左边·中）"
    assert describe_people([P()]) == "小明（左边·中）"
    assert describe_people([P(motion="走远", sure=False)]) == "像小明（没看到名字，左边·中，正在走远）"
    assert describe_people([P(kind="stranger", name=None, sid="陌生人A", look="白斗篷", motion="往右走")]) == \
        "陌生人A（白斗篷，左边·中，正在往右走）"
