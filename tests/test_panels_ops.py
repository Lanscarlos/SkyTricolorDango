from dataclasses import replace

import pytest
from conftest import FakeDevice, scene
from panels_helpers import CARDS

from skydango.config import PanelsConfig
from skydango.game.panels import PanelOps
from skydango.vision.bubbles import Rect
from skydango.vision.panels import UNKNOWN, Button, Panel, PanelReading, PanelState, load_cards

CARD = {c.name: c for c in load_cards(CARDS)}
EMOTE = Panel("emote_panel", "动作面板", Rect(883, 0, 320, 720), False, 20)
WHEEL = Panel("wheel_editor", "轮盘编辑界面", Rect(0, 0, 1280, 720), False, 30)
DIALOG = Panel(UNKNOWN, "不认识的面板", Rect(200, 100, 800, 500), False, 100)
CANCEL = Button("取消", Rect(300, 400, 60, 30), "retreat")
JOIN = Button("加入", Rect(700, 400, 60, 30), "never")
OK = Button("确定", Rect(500, 400, 60, 30), "other")


class FakeWatcher:
    def __init__(self, present=(), close_at=None, reading=None, cards=None):
        self.cards = dict(cards or CARD)
        self.presents = list(present)
        self.close_at = close_at
        self.reading = reading
        self.closed = []
        self.reads = 0
        self.state = PanelState()

    def present(self, frame, panel):
        return self.presents.pop(0) if self.presents else True

    def find_close(self, frame):
        return self.close_at

    def mark_closed(self, name):
        self.closed.append(name)

    def read(self, frame, panel, now):
        self.reads += 1
        return self.reading


def ops(watcher, frames=None):
    device = FakeDevice(frames or [scene()])
    return PanelOps(device, watcher, PanelsConfig(), sleep=lambda s: None, clock=lambda: 100.0), device


def reading(*buttons, panel=DIALOG):
    return PanelReading(panel, "出错了", "网络连接断开", tuple(buttons), 100.0)


def taps(device):
    return [c for c in device.calls if c[0] == "tap"]


def test_close_uses_card_key():
    w = FakeWatcher(present=[False])
    o, device = ops(w)
    assert o.close(EMOTE)
    assert device.calls == [("hw_key", 18)] and w.closed == ["emote_panel"]


def test_close_tries_next_way():
    w = FakeWatcher(present=[True, False], cards={**CARD, "emote_panel": replace(CARD["emote_panel"], close_ways=("key:18", "esc"))})
    o, device = ops(w)
    assert o.close(EMOTE)
    assert device.calls == [("hw_key", 18), ("hw_key", 1)]


def test_close_unknown_uses_retreat_not_never():
    w = FakeWatcher(present=[True, False])
    o, device = ops(w)
    assert o.close(DIALOG, reading(JOIN, CANCEL))
    assert taps(device) == [("tap", 330, 415), ("tap", 330, 415)]  # 第一下只切触屏模式、面板还在 → 再点；没点过 加入
    assert w.closed == [UNKNOWN]


def test_close_reads_when_no_reading_given():
    w = FakeWatcher(present=[True, False], reading=reading(CANCEL))
    o, device = ops(w)
    assert o.close(DIALOG) and w.reads == 1 and taps(device) == [("tap", 330, 415), ("tap", 330, 415)]


def test_close_then_x():
    w = FakeWatcher(present=[True, False], close_at=(1250, 20))
    o, device = ops(w)
    assert o.close(DIALOG, reading(JOIN))
    assert taps(device) == [("tap", 1250, 20), ("tap", 1250, 20)]


def test_close_gives_up():
    w = FakeWatcher()
    o, device = ops(w)
    assert not o.close(DIALOG, reading(JOIN))
    assert device.calls == [] and w.closed == []


def test_close_fails_after_all_ways():
    w = FakeWatcher(present=[True, True])
    o, device = ops(w)
    assert not o.close(EMOTE) and w.closed == []


def test_wake_tap_only_once_when_it_took_effect():
    w = FakeWatcher(present=[False, False])
    o, device = ops(w)
    assert o.close(WHEEL)
    assert taps(device) == [("tap", 1258, 22)]


def test_press_reports_unchanged():
    w = FakeWatcher(present=[True, True])
    o, device = ops(w)
    state, changed = o.press(reading(OK), OK)
    assert state is w.state and not changed and w.closed == []
    assert taps(device) == [("tap", 530, 415), ("tap", 530, 415)]


def test_press_closes_panel_when_gone():
    other = scene()
    other[:] = 255
    w = FakeWatcher(present=[True, False])
    o, device = ops(w, frames=[scene(), scene(), scene(), other])
    state, changed = o.press(reading(OK), OK)
    assert changed and w.closed == [UNKNOWN]


def test_press_refuses_never():
    o, device = ops(FakeWatcher())
    with pytest.raises(ValueError):
        o.press(reading(JOIN), JOIN)
    assert device.calls == []
