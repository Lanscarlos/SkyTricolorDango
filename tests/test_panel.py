from conftest import FakeDevice, scene

from skydango.chat.panel import PanelKeeper
from skydango.config import VisionConfig


class Reader:
    def __init__(self, visible):
        self.visible = list(visible)  # 每次 panel_visible 依次返回，最后一个一直返回
        self.panel_closed_since = None

    def panel_visible(self, frame):
        return self.visible.pop(0) if len(self.visible) > 1 else self.visible[0]


def keeper(visible, mode="log"):
    device = FakeDevice([scene()])
    return PanelKeeper(VisionConfig(mode=mode), device, Reader(visible), sleep=lambda s: None), device


def test_opens_panel_with_key_when_closed():
    k, device = keeper([False, True])
    assert k.ensure_open() is True
    assert device.calls == [("hw_key", 46)]


def test_does_not_press_while_typing():
    k, device = keeper([False])
    device.shown = True  # 输入框开着时按 C 会打出字母
    assert k.ensure_open() is False and device.calls == []


def test_other_modes_do_nothing():
    k, device = keeper([False], mode="bubble")
    assert k.ensure_open() is True and device.calls == []


def test_reopens_only_after_delay_and_cooldown():
    k, device = keeper([False, True])
    k.reader.panel_closed_since = 0.0
    k.maybe_reopen(3.0)  # 关了 3 秒，还没到 log_reopen_after（5 秒）
    assert device.calls == []
    k.maybe_reopen(6.0)
    assert device.calls == [("hw_key", 46)]
    k.reader.visible = [False]
    k.maybe_reopen(8.0)  # 冷却（10 秒）内不再按
    assert device.calls == [("hw_key", 46)]
