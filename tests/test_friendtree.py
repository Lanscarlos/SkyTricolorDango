import numpy as np
import pytest
from conftest import FakeDevice, panel_manager

from skydango.config import FriendCheckConfig
from skydango.game import friendtree
from skydango.game.friendtree import LINUX_KEY_ESC, FriendChecker


class PanelDevice(FakeDevice):
    """点人物 → 右边出现深色面板；按 close_with 这个键才关上。"""

    def __init__(self, close_with=("hw_key", LINUX_KEY_ESC), opens=True):
        super().__init__([np.zeros((1080, 1920, 3), np.uint8)])
        self.open = False
        self.close_with = close_with
        self.opens = opens

    def screenshot(self):
        img = np.full((1080, 1920, 3), 150, np.uint8)
        if self.open:
            img[:, 1250:] = 30
        return img

    def tap(self, x, y):
        super().tap(x, y)
        if self.opens:
            self.open = True

    def hw_key(self, code):
        super().hw_key(code)
        if ("hw_key", code) == self.close_with:
            self.open = False

    def key(self, keycode):
        super().key(keycode)
        if ("key", keycode) == self.close_with:
            self.open = False


@pytest.fixture(autouse=True)
def touch(monkeypatch):
    monkeypatch.setattr(friendtree, "touch_mode", lambda frame: True)


def checker(dev, **kw):
    return FriendChecker(dev, FriendCheckConfig(**kw), sleep=lambda s: None)


def test_opens_panel_and_closes_with_esc():
    dev = PanelDevice()
    result = checker(dev).check(1500, 600)
    assert dev.calls == [("tap", 1500, 600), ("hw_key", LINUX_KEY_ESC)]
    assert result.changed > 0.06 and result.closed and result.closed_by == "esc"
    assert result.opened[500, 1800, 0] == 30 and result.after[500, 1800, 0] == 150


def test_falls_back_to_back_key():
    dev = PanelDevice(close_with=("key", 4))
    result = checker(dev, close=["esc", "back"]).check(1500, 600)
    assert result.closed_by == "back"
    assert [c[0] for c in dev.calls] == ["tap", "hw_key", "key"]


def test_reports_when_panel_cannot_be_closed():
    dev = PanelDevice(close_with=None)
    result = checker(dev).check(1500, 600)
    assert not result.closed and result.closed_by == ""
    assert dev.open


def test_no_panel_means_nothing_to_close():
    dev = PanelDevice(opens=False)
    c = checker(dev)
    result = c.check(1500, 600)
    assert not c.looks_open(result.changed) and result.closed
    assert dev.calls == [("tap", 1500, 600)]


def test_keyboard_mode_needs_a_wake_tap_first(monkeypatch):
    modes = iter([False, True])
    monkeypatch.setattr(friendtree, "touch_mode", lambda frame: next(modes))
    dev = PanelDevice(opens=False)
    checker(dev).check(1500, 600)
    assert dev.calls == [("tap", 1500, 600), ("tap", 1500, 600)]


def test_reopens_chat_panel_after_tapping():
    dev = PanelDevice()
    state = iter([True, False, True])  # 点之前开着 → 点完被关了 → 按 C 后开了
    c = FriendChecker(dev, FriendCheckConfig(), panel=panel_manager(dev, lambda: next(state)), sleep=lambda s: None)
    c.check(1500, 600)
    assert dev.calls[-1] == ("hw_key", 46)
