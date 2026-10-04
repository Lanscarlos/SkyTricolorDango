import pytest
from conftest import FakeDevice, scene

from conftest import fake_panel
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
    return PanelManager(VisionConfig(mode=mode), PanelConfig(mode="always"), device, Reader(visible), sleep=lambda s: None), device


def test_opens_panel_with_key_when_closed():
    m, device = manager([False, True])
    assert m.ensure_open() is True
    assert device.calls == [("hw_key", 46)]


def test_press_reports_to_on_press():
    m, device = manager([False, True])
    m.clock = lambda: 12.5
    seen = []
    m.on_press = seen.append
    m.ensure_open()
    assert seen == [12.5]


def test_on_press_error_does_not_break_press():
    m, device = manager([False, True])
    m.on_press = lambda at: 1 / 0
    assert m.ensure_open() is True and device.calls == [("hw_key", 46)]


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
    device = FakeDevice([scene()])
    m, state = fake_panel(device, open_=True)
    with m.borrow("camera") as was_open:
        assert was_open is True and m.lent == "camera" and state.open is False
        assert device.calls == [("hw_key", 46)]
    assert device.calls == [("hw_key", 46), ("hw_key", 46)] and m.lent is None and state.open is True


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
    device = FakeDevice([scene()])
    m, state = fake_panel(device, open_=True)
    with pytest.raises(RuntimeError):
        with m.borrow("camera"):
            raise RuntimeError("转镜头出错")
    assert m.lent is None and device.calls == [("hw_key", 46), ("hw_key", 46)] and state.open is True


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


def test_just_closed_panel_is_not_taken_as_opened_by_someone_else():
    """实测（2026-09-30 live）：看一眼刚关上 0.7 秒，被判成"面板开着（不是自己开的）"进了聊天中、又把面板打开。
    关的时间记的是这一圈开始的时刻，真按键晚了近 1 秒，读聊天确认面板关了又晚零点几秒，加起来超过 open_timeout。"""
    m, dev, _ = auto()
    t = [0.0]
    m.clock = lambda: t[0]
    m.tick(30.0, [], visible=False)
    m.tick(30.2, [], visible=True)
    t[0] = 31.3  # 这一圈读聊天、跑识别花了快 1 秒，按关的键时已经 31.3
    m.tick(30.4, [], visible=True)
    assert m.state == "idle" and presses(dev) == 2
    t[0] = 32.0
    m.tick(32.0, [], visible=True)  # 读聊天的还没确认面板关了
    assert m.state == "idle" and presses(dev) == 2
    t[0] = 34.0
    m.tick(34.0, [], visible=True)  # 过了 open_timeout 还开着：真是别人开的
    assert m.state == "chatting"


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


# ---- 好友头顶冒气泡：开着面板等消息 ----
def bubble_open(m, t):
    m.bubble_seen(t)
    m.tick(t, [], visible=False)
    assert m.state == "bubble"


def test_bubble_keeps_panel_open_until_message():
    m, dev, _ = auto()
    bubble_open(m, 10.0)
    for t in (10.5, 12.0, 14.0, 16.0):
        m.bubble_seen(t)
        m.tick(t, [], visible=True)
    assert m.state == "bubble" and presses(dev) == 1  # 对方还在打字：不关
    m.tick(16.2, ["小明：在吗"], visible=True)
    assert m.state == "chatting" and presses(dev) == 1


def test_bubble_closes_after_bubble_gone():
    m, dev, _ = auto()
    bubble_open(m, 10.0)
    m.bubble_seen(12.0)
    m.tick(12.0, [], visible=True)
    m.tick(14.9, [], visible=True)
    assert m.state == "bubble"
    m.tick(15.0, [], visible=True)  # 气泡没了 3 秒还没新消息
    assert m.state == "idle" and presses(dev) == 2


def test_bubble_closes_after_bubble_wait():
    m, dev, _ = auto()
    bubble_open(m, 10.0)
    for t in range(11, 25):
        m.bubble_seen(float(t))
        m.tick(float(t), [], visible=True)
    assert m.state == "bubble"
    m.bubble_seen(25.0)
    m.tick(25.0, [], visible=True)  # 一直在打字也最多等 15 秒
    assert m.state == "idle" and presses(dev) == 2


def test_bubble_seen_while_chatting_changes_nothing():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    m.bubble_seen(101.0)
    m.tick(101.0, [], visible=True)
    assert m.state == "chatting" and presses(dev) == 0


def test_bubble_describe():
    m, dev, _ = auto()
    bubble_open(m, 10.0)
    assert m.describe(16.0) == "等气泡（还剩 9 秒）"


def test_borrow_during_bubble_waits_again_after_return():
    m, dev, state = auto()
    bubble_open(m, 10.0)
    with m.borrow("camera"):
        pass
    m.tick(11.0, [], visible=False)
    assert m.state == "bubble"


# ---- 说话前先开面板 ----
def test_before_speak_opens_panel_first_when_idle():
    m, dev, state = auto()
    m.before_speak(5.0)
    assert dev.calls == [("hw_key", 46)] and state.open is True and m.state == "chatting"
    m.tick(49.0, [], visible=True)  # 说话算"不安静"
    assert m.state == "chatting"


def test_before_speak_does_not_press_while_typing():
    m, dev, state = auto()
    dev.shown = True
    m.before_speak(5.0)
    assert dev.calls == [] and m.state == "chatting"


def test_before_speak_while_lent_reopens_on_return():
    m, dev, state = auto()
    with m.borrow("camera", close=False):
        m.before_speak(5.0)
        assert dev.calls == []
    assert state.open is True and m.state == "chatting"


def test_before_speak_does_nothing_in_always_mode():
    m, dev = manager([False])
    m.before_speak(5.0)
    assert dev.calls == []


def test_before_speak_reopens_panel_when_chatting_but_closed():
    m, dev, state = auto()
    chatting(m, state, 100.0)
    state.open = False  # 聊天中面板被关了（点了屏幕之类）
    m.before_speak(101.0)
    assert state.open is True and presses(dev) == 1


# ---- 整分支评审的修复 ----
class SlowPanel:
    """按键后面板要过 lag 次截图才真的变（开 / 关的动画）。"""

    def __init__(self, device, open_=False, lag=2):
        self.open, self.target, self.countdown, self.lag = open_, open_, 0, lag
        press = device.hw_key

        def hw_key(code):
            press(code)
            if code == 46:
                self.target, self.countdown = not self.target, self.lag

        device.hw_key = hw_key

    def visible(self):
        if self.countdown > 0:
            self.countdown -= 1
        else:
            self.open = self.target
        return self.open


def slow(open_=False, mode="auto"):
    from conftest import panel_manager

    device = FakeDevice([scene()])
    panel = SlowPanel(device, open_)
    m = panel_manager(device, panel.visible, mode)
    m.start(0.0)
    return m, device, panel


def test_borrow_enter_failure_does_not_leave_panel_lent():  # I1
    m, device = manager([True])

    def boom(code):
        raise RuntimeError("adb 失败")

    device.hw_key = boom
    with pytest.raises(RuntimeError):
        with m.borrow("camera"):
            pass
    assert m.lent is None and m.should_be_open() is True


def test_borrow_right_after_peek_press_closes_the_opening_panel():  # I2 场景 A
    m, dev, panel = slow()
    m.tick(30.0, [], visible=False)  # 按了 C，面板还在打开的动画里
    with m.borrow("camera"):
        assert panel.target is False  # 等面板开出来再关掉，转镜头时面板是关着的
    assert presses(dev) == 2


def test_borrow_right_after_close_does_not_reopen():  # I2 场景 B
    m, dev, panel = slow(open_=True)
    m.tick(1.0, ["小明：在吗"], visible=True)
    m.tick(46.0, [], visible=True)  # 安静 45 秒：按 C 关，关的动画中
    with m.borrow("camera") as was_open:
        assert was_open is False
    assert presses(dev) == 1 and panel.target is False


def test_before_speak_right_after_peek_press_does_not_toggle_back():  # I2 场景 C
    m, dev, panel = slow()
    m.tick(30.0, [], visible=False)
    m.before_speak(30.0)
    assert presses(dev) == 1 and panel.target is True and m.state == "chatting"


def settling_auto():
    m, dev, state = auto()
    m.reader.settling = True  # ChatReader 看到新行，正在等下一帧确认
    return m, dev, state


def test_peek_does_not_close_while_reader_settling():  # I3
    m, dev, _ = settling_auto()
    peek(m, 30.0)
    m.tick(30.6, [], visible=True)
    assert m.state == "peek" and presses(dev) == 1
    m.reader.settling = False
    m.tick(30.8, [], visible=True)
    assert m.state == "idle"


def test_chatting_does_not_close_while_reader_settling():  # I3
    m, dev, state = settling_auto()
    chatting(m, state, 100.0)
    m.tick(145.0, [], visible=True)
    assert m.state == "chatting"


def test_bubble_does_not_close_while_reader_settling():  # I3
    m, dev, _ = settling_auto()
    bubble_open(m, 10.0)
    m.tick(14.0, [], visible=True)
    assert m.state == "bubble"


def test_always_restored_when_panel_was_closed_before_borrow():  # M1
    m, _ = manager([False])
    with m.borrow("camera", close=False):
        pass
    assert m.restored() is True  # 和原来一样：借之前就关着，不算"没恢复"


def test_borrow_does_not_press_while_typing():  # M4
    m, device = manager([True])
    device.shown = True
    with m.borrow("camera") as was_open:
        assert was_open is True
    assert device.calls == []


def test_peek_close_does_not_press_while_typing():  # M4
    m, dev, _ = auto()
    m.tick(30.0, [], visible=False)
    dev.shown = True  # 看一眼时用户按 Enter 开始打字
    m.tick(30.2, [], visible=True)
    m.tick(30.4, [], visible=True)
    assert presses(dev) == 1 and m.state == "chatting"


# ---- 空闲注意力用：安静 / 先看一眼推迟开面板（plan 2026-09-30-idle-attention Task 1） ----

def test_quiet_only_when_idle_nothing_pending_and_peek_not_due():
    m, _, _ = auto()
    assert m.quiet(1.0, margin=1.6)
    assert not m.quiet(m.cfg.idle_peek - 1.0, margin=1.6)  # 快到定时看一眼
    m.trigger("approach", 2.0)
    assert m.pending == "approach" and not m.quiet(2.0, 1.6)


def test_quiet_false_when_lent_or_not_idle_or_always():
    m, _, state = auto()
    with m.borrow("camera"):
        assert not m.quiet(1.0, 1.6)
    chatting(m, state, 2.0)
    assert m.state == "chatting" and not m.quiet(2.1, 1.6)
    always, _ = fake_panel(FakeDevice([scene()]), open_=True, mode="always")
    assert not always.quiet(1.0, 1.6)


def test_hold_off_delays_pending_peek_until_released():
    m, device, _ = auto()
    m.bubble_seen(5.0)
    assert m.pending == "bubble"
    assert m.hold_off(7.0, 5.0) is True
    m.tick(5.5, [], visible=False)
    assert presses(device) == 0  # 推迟中：不开
    m.hold_off(None, 5.6)
    m.tick(5.7, [], visible=False)
    assert presses(device) == 1 and m.state == "bubble"  # 放手后照常开


def test_hold_off_expires_on_its_own():
    m, device, _ = auto()
    m.trigger("approach", 5.0)
    m.hold_off(7.0, 5.0)
    m.tick(6.9, [], visible=False)
    assert presses(device) == 0
    m.tick(7.0, [], visible=False)
    assert presses(device) == 1


def test_hold_off_only_once_per_pending():
    m, _, _ = auto()
    m.bubble_seen(5.0)
    assert m.hold_off(7.0, 5.0) is True
    m.hold_off(None, 6.0)
    assert m.hold_off(9.0, 6.0) is False  # 同一个 pending 不再推迟


def test_hold_off_does_not_block_chatting():
    m, _, state = auto()
    m.bubble_seen(5.0)
    m.hold_off(7.0, 5.0)
    chatting(m, state, 5.5)  # 推迟中读到新消息
    assert m.state == "chatting"


# ---- 无障碍读法：跟画面里的好友聊天时面板关着（聊着 talking） ----
from skydango.chat.reader import Message  # noqa: E402


class BubbleReader:
    """无障碍读法的 reader（FallbackReader 的接口）：能读气泡、看得见哪些名字标签、想不想看一眼面板。"""

    def __init__(self, state):
        self.state = state
        self.panel_closed_since = None
        self.reads_bubbles = True
        self.tags: list[str] = []
        self.peek: str | None = None

    def panel_visible(self, frame):
        return self.state.open

    def tags_in_view(self):
        return list(self.tags)

    def want_peek(self):
        peek, self.peek = self.peek, None
        return peek


def a11y(open_=False, mode="auto"):
    device = FakeDevice([scene()])
    m, state = fake_panel(device, open_=open_, mode=mode)
    m.reader = BubbleReader(state)
    m.start(0.0)
    return m, device, state


def said(text="在吗", who="小明", source="bubble"):
    return Message(text, (0, 0, 10, 10), 0.0, who, source)


def apeek(m, t):
    """无障碍读法下看一眼：面板至少开 1 秒才关（A11Y_PEEK_MIN）。"""
    m.tick(t, [], visible=False)
    m.tick(t + 0.2, [], visible=True)
    m.tick(t + 1.2, [], visible=True)


def talking(m, t, who="小明"):
    m.tick(t, [said(who=who)], visible=False)
    assert m.state == "talking"


def test_bubble_message_enters_talking_without_key():
    m, dev, state = a11y()
    m.tick(5.0, [said()], visible=False)
    assert m.state == "talking" and dev.calls == [] and state.open is False
    assert m.should_be_open() is False


def test_panel_message_still_enters_chatting():
    m, dev, state = a11y(open_=True)
    m.tick(5.0, [said(source="panel")], visible=True)
    assert m.state == "chatting"


def test_mixed_message_enters_chatting():
    m, dev, state = a11y()
    m.tick(5.0, [said(), said("嗯", who="小红", source="panel")], visible=False)
    assert m.state == "chatting"


def test_talking_peeks_every_chat_peek():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.tick(19.9, [], visible=False)
    assert presses(dev) == 0 and m.state == "talking"
    m.tick(20.0, [], visible=False)  # 进「聊着」15 秒：看一眼面板（接住画面外的人）
    assert presses(dev) == 1 and m.state == "peek"
    m.tick(20.2, [], visible=True)
    m.tick(21.2, [], visible=True)  # 无障碍读法：看一眼至少开 1 秒
    assert presses(dev) == 2 and m.state == "talking" and state.open is False  # 没新消息：关上、接着聊
    m.tick(36.1, [], visible=False)
    assert presses(dev) == 2
    m.tick(36.2, [], visible=False)  # 上次读到面板 15 秒后再看
    assert presses(dev) == 3 and m.state == "peek"


def test_talking_peek_with_panel_message_enters_chatting():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.tick(20.0, [], visible=False)
    m.tick(20.2, [said("我在这边", who="小红", source="panel")], visible=True)
    assert m.state == "chatting" and state.open is True


def test_talking_goes_idle_after_quiet_close():
    m, dev, state = a11y()
    talking(m, 5.0)
    apeek(m, 20.0)
    apeek(m, 36.5)
    assert m.state == "talking"
    m.tick(49.9, [], visible=False)
    assert m.state == "talking"
    m.tick(50.0, [], visible=False)  # 45 秒没动静
    assert m.state == "idle" and state.open is False


def test_bubble_message_refreshes_talking():
    m, dev, state = a11y()
    m.cfg.chat_peek = 100.0  # 不看一眼，只看安静计时
    talking(m, 5.0)
    m.tick(40.0, [said("还在")], visible=False)
    m.tick(50.0, [], visible=False)
    assert m.state == "talking"
    m.tick(85.0, [], visible=False)
    assert m.state == "idle" and dev.calls == []


def test_peek_from_talking_after_quiet_close_goes_idle():
    m, dev, state = a11y()
    m.cfg.chat_peek = 100.0  # 只靠 trigger 看一眼
    talking(m, 5.0)
    m.trigger("approach", 49.0)
    m.tick(49.0, [], visible=False)
    assert m.state == "peek"
    m.tick(50.2, [], visible=True)
    m.tick(51.2, [], visible=True)  # 看完已经安静 45 秒了
    assert m.state == "idle" and state.open is False


def test_bubble_during_idle_peek_returns_to_talking():
    m, dev, state = a11y()
    m.tick(30.0, [], visible=False)  # 定时看一眼
    assert m.state == "peek"
    m.tick(30.1, [said()], visible=False)  # 面板还没出来，气泡先读到了
    assert m.state == "peek"
    m.tick(30.2, [], visible=True)
    m.tick(31.2, [], visible=True)
    assert m.state == "talking" and state.open is False


def test_before_speak_keeps_panel_closed_when_partner_in_view():
    m, dev, state = a11y()
    m.cfg.chat_peek = 100.0  # 不看一眼，只看安静计时
    talking(m, 5.0)
    m.reader.tags = ["小明", "小红"]
    m.before_speak(10.0)
    assert dev.calls == [] and m.state == "talking"
    m.tick(54.9, [], visible=False)  # 说话刷新了活动时间
    assert m.state == "talking"
    m.tick(55.0, [], visible=False)
    assert m.state == "idle"

    m, dev, state = a11y()
    talking(m, 5.0)
    m.before_speak(10.0)  # 他不在画面里了：照旧开面板
    assert presses(dev) == 1 and state.open is True and m.state == "chatting"


def test_before_speak_opens_panel_when_any_partner_out_of_view():
    m, dev, state = a11y()
    talking(m, 5.0)
    talking(m, 6.0, who="小红")
    m.reader.tags = ["小明"]
    m.before_speak(10.0)
    assert presses(dev) == 1 and m.state == "chatting"


def test_want_peek_triggers_peek():
    m, dev, state = a11y()
    m.reader.peek = "bubble_text"
    m.tick(5.0, [], visible=False)
    assert presses(dev) == 1 and m.state == "peek"


def test_want_peek_while_talking():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.reader.peek = "bubble_text"
    m.tick(6.0, [], visible=False)
    assert presses(dev) == 1 and m.state == "peek"


def test_fallback_to_ocr_while_talking_opens_panel():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.reader.reads_bubbles = False
    m.tick(6.0, [], visible=False)
    assert m.state == "chatting" and presses(dev) == 1 and state.open is True


def test_talking_does_not_peek_while_typing():
    """反射替大脑开着输入框时到点不看一眼：交给 _open_peek 会升级成聊天中，框一关就开面板。"""
    m, dev, state = a11y()
    talking(m, 5.0)
    dev.shown = True
    m.tick(20.0, [], visible=False)
    m.tick(30.0, [], visible=False)
    assert m.state == "talking" and presses(dev) == 0
    dev.shown = False  # 框关上了：照常看一眼
    m.tick(30.2, [], visible=False)
    assert m.state == "peek" and presses(dev) == 1


def test_talking_typing_counts_as_activity_for_quiet_close():
    m, dev, state = a11y()
    m.cfg.chat_peek = 100.0
    talking(m, 5.0)
    dev.shown = True
    m.tick(50.0, [], visible=False)  # 安静 45 秒到了，但输入框开着：算在说话
    assert m.state == "talking" and presses(dev) == 0
    dev.shown = False
    m.tick(94.9, [], visible=False)
    assert m.state == "talking"
    m.tick(95.0, [], visible=False)
    assert m.state == "idle" and presses(dev) == 0


def test_fallback_to_ocr_during_talking_peek_goes_chatting():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.tick(20.0, [], visible=False)
    assert m.state == "peek"
    m.reader.reads_bubbles = False  # 看的途中退回 OCR
    m.tick(20.2, [], visible=True)
    m.tick(20.4, [], visible=True)
    assert m.state == "chatting" and state.open is True and presses(dev) == 1  # 面板留着，不关


def test_borrow_during_talking_peek_returns_to_talking():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.tick(20.0, [], visible=False)
    assert m.state == "peek" and state.open is True
    with m.borrow("camera"):
        assert m.state == "talking"
    assert m.state == "talking" and state.open is False  # 聊着不该开面板：归还不重开


def test_borrow_during_talking_peek_after_fallback_returns_to_chatting():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.tick(20.0, [], visible=False)
    m.reader.reads_bubbles = False
    with m.borrow("camera"):
        assert m.state == "chatting"
    assert m.state == "chatting" and state.open is True  # 归还时按聊天中重开


def test_talking_panel_opened_by_someone_else_becomes_chatting():
    m, dev, state = a11y()
    talking(m, 5.0)
    state.open = True
    m.tick(8.0, [], visible=True)
    assert m.state == "chatting" and presses(dev) == 0


def test_bubble_seen_while_talking_changes_nothing():
    m, dev, state = a11y()
    talking(m, 5.0)
    m.bubble_seen(6.0)
    m.tick(6.0, [], visible=False)
    assert m.state == "talking" and presses(dev) == 0


def test_talking_describe():
    m, dev, state = a11y()
    talking(m, 5.0)
    assert m.describe(8.0) == "聊着（面板关着，12 秒后看一眼）"


def test_ocr_reader_unchanged():
    m, dev, _ = auto()  # 现有的假 reader：没有 reads_bubbles 这些属性
    m.tick(5.0, [said()], visible=False)
    assert m.state == "chatting"


def test_ocr_fallback_reader_unchanged():
    m, dev, _ = a11y()
    m.reader.reads_bubbles = False
    m.reader.peek = "bubble_text"
    m.tick(5.0, [said()], visible=False)
    assert m.state == "chatting" and dev.calls == []


def test_always_mode_unchanged():
    m, dev, state = a11y(open_=True, mode="always")
    m.tick(5.0, [said()], visible=True)
    assert m.state == "chatting" and dev.calls == []
    m.before_speak(6.0)
    assert dev.calls == []


def _ticker(m):
    """让面板管理器的时钟（按键时刻）和 tick 的 now 一致。"""
    clock = {"t": 0.0}
    m.clock = lambda: clock["t"]

    def tick(t, fresh=(), visible=False):
        clock["t"] = t
        m.tick(t, list(fresh), visible=visible)

    return tick


def test_a11y_peek_stays_open_until_rows_loaded():
    """无障碍读法每 0.15 秒一份快照，「连着 2 帧看到」不到半秒就满足：面板行还没加载全就关了（10-04 真机）。至少开 1 秒。"""
    m, dev, state = a11y()
    tick = _ticker(m)
    tick(30.0)  # 定时看一眼
    assert m.state == "peek" and presses(dev) == 1
    tick(30.2, visible=True)
    tick(30.4, visible=True)
    assert m.state == "peek" and presses(dev) == 1
    tick(31.3, visible=True)
    assert m.state == "idle" and presses(dev) == 2


def test_a11y_panel_lingering_after_own_close_is_not_user_opened():
    """关面板的动画约 2 秒，无障碍树里的面板节点这段时间还在、还会闪（10-04 真机：关了 2.6 秒后还"看得到"）。
    团子自己关的 3.5 秒内看到面板不算别人打开的。"""
    m, dev, state = a11y()
    tick = _ticker(m)
    tick(30.0)
    tick(30.2, visible=True)
    tick(31.3, visible=True)
    assert m.state == "idle"
    tick(32.0, visible=True)
    tick(33.9, visible=True)
    tick(34.5, visible=False)
    assert m.state == "idle" and presses(dev) == 2
    tick(36.0, visible=True)  # 关了 4.7 秒后又看到：这才是别人打开的
    assert m.state == "chatting"


def test_a11y_talking_peek_lingering_panel_stays_talking():
    m, dev, state = a11y()
    tick = _ticker(m)
    talking(m, 5.0)
    tick(20.0)
    tick(20.2, visible=True)
    tick(21.3, visible=True)
    assert m.state == "talking" and presses(dev) == 2
    tick(23.0, visible=True)  # 关面板的动画里还看得到
    assert m.state == "talking"


def test_ocr_reader_keeps_short_close_window():
    """OCR 读法照旧：关了 1.5 秒后看到面板就算别人打开的、看一眼 2 帧就关。"""
    m, dev, state = auto()
    tick = _ticker(m)
    tick(30.0)
    tick(30.2, visible=True)
    tick(30.4, visible=True)
    assert m.state == "idle"
    tick(32.0, visible=True)
    assert m.state == "chatting"
