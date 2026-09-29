import pytest
from conftest import FakeDevice, scene

from skydango.chat.panel import PanelManager
from skydango.config import PanelConfig, VisionConfig


class Reader:
    def __init__(self, visible):
        self.visible = list(visible)  # 每次 panel_visible 依次返回，最后一个一直返回
        self.panel_closed_since = None

    def panel_visible(self, frame):
        return self.visible.pop(0) if len(self.visible) > 1 else self.visible[0]


def manager(visible, mode="log"):
    device = FakeDevice([scene()])
    return PanelManager(VisionConfig(mode=mode), PanelConfig(), device, Reader(visible), sleep=lambda s: None), device


def test_opens_panel_with_key_when_closed():
    m, device = manager([False, True])
    assert m.ensure_open() is True
    assert device.calls == [("hw_key", 46)]


def test_does_not_press_while_typing():
    m, device = manager([False])
    device.shown = True  # 输入框开着时按 C 会打出字母
    assert m.ensure_open() is False and device.calls == []


def test_other_modes_do_nothing():
    m, device = manager([False], mode="bubble")
    assert m.ensure_open() is True and device.calls == []


def test_reopens_only_after_delay_and_cooldown():
    m, device = manager([False, True])
    m.reader.panel_closed_since = 0.0
    m.tick(3.0, [], visible=False)  # 关了 3 秒，还没到 log_reopen_after（5 秒）
    assert device.calls == []
    m.tick(6.0, [], visible=False)
    assert device.calls == [("hw_key", 46)]
    m.reader.visible = [False]
    m.tick(8.0, [], visible=False)  # 冷却（10 秒）内不再按
    assert device.calls == [("hw_key", 46)]


def test_start_opens_panel_in_always_mode():
    m, device = manager([False, True])
    assert m.start(0.0) is True and device.calls == [("hw_key", 46)]


def test_borrow_closes_then_reopens_in_always_mode():
    m, device = manager([True, False, True])  # 借前开着 → 还的时候关着 → 按了之后开了
    with m.borrow("camera") as was_open:
        assert was_open is True and m.lent == "camera"
        assert device.calls == [("hw_key", 46)]
    assert device.calls == [("hw_key", 46), ("hw_key", 46)] and m.lent is None


def test_borrow_nested_restores_once():
    m, device = manager([True, False, False, True])
    with m.borrow("camera"):
        with m.borrow("emotes") as inner:
            assert inner is False  # 外层已经关了
        assert m.lent == "camera" and device.calls == [("hw_key", 46)]
    assert device.calls.count(("hw_key", 46)) == 2


def test_borrow_close_false_does_not_press_on_enter():  # social / friendtree：点屏幕自己会关
    m, device = manager([True, False, True])
    with m.borrow("social", close=False):
        assert device.calls == []
    assert device.calls == [("hw_key", 46)]


def test_borrow_restores_on_exception():
    m, device = manager([True, False, True])
    with pytest.raises(RuntimeError):
        with m.borrow("camera"):
            raise RuntimeError("转镜头出错")
    assert m.lent is None and device.calls == [("hw_key", 46), ("hw_key", 46)]


def test_no_reopen_while_lent():
    m, device = manager([False])  # 面板一直关着
    m.reader.panel_closed_since = 0.0
    with m.borrow("camera", close=False):
        m.tick(60.0, [], visible=False)  # 早过了 log_reopen_after，但借出中不重开
        assert device.calls == []


def test_borrow_does_not_reopen_while_typing():
    m, device = manager([True, False])
    with m.borrow("camera"):
        device.shown = True
    assert device.calls == [("hw_key", 46)]  # 归还时输入框开着：不按，交给 tick 以后重开


def test_restored_and_should_be_open():
    m, _ = manager([True])
    assert m.should_be_open() is True and m.restored() is True
    with m.borrow("camera", close=False):
        assert m.should_be_open() is False


def test_inactive_manager_never_presses():
    m, device = manager([True], mode="bubble")
    with m.borrow("camera") as was_open:
        assert was_open is False
    m.tick(100.0, [], visible=False)
    assert device.calls == [] and m.active is False
