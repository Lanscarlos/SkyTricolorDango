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


# ---- auto 模式：闲着关、定时 / 有触发看一眼、聊天中开着 ----
from conftest import fake_panel  # noqa: E402


def auto(open_=False):
    device = FakeDevice([scene()])
    m, state = fake_panel(device, open_=open_, mode="auto")
    m.start(0.0)
    return m, device, state


def presses(device):
    return device.calls.count(("hw_key", 46))


def peek(m, t):
    """在 t 打开看一眼，接着两帧都没有新消息 → 关上。"""
    m.tick(t, [], visible=False)
    m.tick(t + 0.2, [], visible=True)
    m.tick(t + 0.4, [], visible=True)


def chatting(m, state, t):
    state.open = True
    m.tick(t, ["小明：在吗"], visible=True)
    assert m.state == "chatting"


def test_idle_peeks_after_idle_peek_and_closes_without_new():
    m, dev, _ = auto()
    assert m.state == "idle" and dev.calls == []
    m.tick(29.0, [], visible=False)
    assert dev.calls == []
    m.tick(30.0, [], visible=False)
    assert presses(dev) == 1 and m.state == "peek"
    m.tick(30.2, [], visible=True)
    m.tick(30.4, [], visible=True)
    assert presses(dev) == 2 and m.state == "idle"


def test_peek_waits_two_visible_frames():
    m, dev, _ = auto()
    m.tick(30.0, [], visible=False)
    m.tick(30.2, [], visible=True)  # 第一帧面板可能还没画完
    assert presses(dev) == 1 and m.state == "peek"


def test_peek_with_new_message_enters_chatting():
    m, dev, _ = auto()
    m.tick(30.0, [], visible=False)
    m.tick(30.2, ["小明：在吗"], visible=True)
    assert m.state == "chatting" and presses(dev) == 1


def test_chatting_closes_after_quiet_close():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    m.tick(144.9, [], visible=True)
    assert m.state == "chatting" and presses(dev) == 0
    m.tick(145.0, [], visible=True)
    assert m.state == "idle" and presses(dev) == 1


def test_chatting_does_not_close_while_typing():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    dev.shown = True
    m.tick(145.0, [], visible=True)
    assert m.state == "chatting" and presses(dev) == 0  # 输入框开着：安静重新计时
    dev.shown = False
    m.tick(189.9, [], visible=True)
    assert m.state == "chatting"
    m.tick(190.0, [], visible=True)
    assert m.state == "idle"


def test_busy_resets_quiet_timer():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    m.busy(130.0)
    m.tick(145.0, [], visible=True)
    assert m.state == "chatting"
    m.tick(175.0, [], visible=True)
    assert m.state == "idle"


def test_trigger_in_idle_peeks_now_but_respects_cooldown():
    m, dev, _ = auto()
    m.trigger("arrive", 5.0)
    peek(m, 5.0)  # 触发了马上看；5.4 关上
    assert presses(dev) == 2 and m.state == "idle"
    m.trigger("approach", 7.0)
    m.tick(7.0, [], visible=False)
    assert presses(dev) == 2  # 冷却（5 秒）内先记着
    m.tick(10.4, [], visible=False)
    assert presses(dev) == 3 and m.state == "peek"


def test_trigger_ignored_while_chatting():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    m.trigger("arrive", 110.0)
    m.tick(145.0, [], visible=True)  # 安静关上
    m.tick(146.0, [], visible=False)
    assert presses(dev) == 1 and m.state == "idle"  # 聊天中的触发不留到闲着


def test_no_peek_during_blackout():
    m, dev, _ = auto()
    m.tick(30.0, [], visible=False, blackout=True)
    m.tick(40.0, [], visible=False, blackout=True)
    assert dev.calls == []
    m.trigger("scene", 41.0)
    m.tick(41.0, [], visible=False)
    assert presses(dev) == 1 and m.state == "peek"


def test_peek_gives_up_after_open_timeout():
    m, dev, _ = auto()
    m.tick(30.0, [], visible=False)
    m.tick(30.5, [], visible=False)
    m.tick(31.5, [], visible=False)  # 按了 1.5 秒还没看到面板
    assert m.state == "idle"
    m.tick(40.0, [], visible=False)
    assert presses(dev) == 1  # 不连按，到下个周期再试


def test_unexpected_open_panel_becomes_chatting():
    m, dev, state = auto()
    state.open = True
    m.tick(5.0, [], visible=True)
    assert m.state == "chatting" and dev.calls == []


def test_just_closed_panel_is_not_unexpected_open():
    m, dev, _ = auto()
    peek(m, 30.0)  # 30.4 关上
    m.tick(30.6, [], visible=True)  # 关的动画：这一帧还看得到
    assert m.state == "idle"


def test_idle_tick_does_not_poll_ime():
    m, dev, _ = auto()
    dev.ime_calls = 0
    for t in range(1, 21):
        m.tick(float(t), [], visible=False)
    assert dev.ime_calls == 0


def test_borrow_during_peek_retries_after_return():
    m, dev, state = auto()
    m.tick(30.0, [], visible=False)  # 按了，开始看一眼
    with m.borrow("camera"):
        assert m.state == "idle" and state.open is False
    assert state.open is False  # 闲着：归还不重开
    m.tick(30.5, [], visible=False)
    assert m.state == "peek" and state.open is True  # 欠着的那一眼马上补


def test_borrow_while_chatting_reopens_on_return():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    with m.borrow("camera"):
        assert state.open is False
    assert state.open is True and m.state == "chatting"


def test_missing_for_counts_only_when_should_be_open():
    m, dev, state = auto()
    assert m.missing_for(100.0) == 0
    chatting(m, state, 200.0)
    state.open = False
    m.tick(201.0, [], visible=False)
    assert m.missing_for(231.0) == 30.0


def test_shutdown_reopens_panel_in_auto():
    m, dev, state = auto()
    m.shutdown()
    assert state.open is True


def test_describe():
    m, dev, state = auto()
    assert m.describe(12.0) == "闲着（18 秒后看一眼）"
    with m.borrow("camera", close=False):
        assert m.describe(12.0) == "借出：camera"
    chatting(m, state, 20.0)
    assert m.describe(21.0) == "聊天中"
    always, _ = manager([True])
    assert always.describe(0.0) == "常开"
    assert manager([True], mode="bubble")[0].describe(0.0) == ""
