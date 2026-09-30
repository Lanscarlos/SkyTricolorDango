import numpy as np
import pytest
from conftest import FakeDevice, fake_panel
from test_wheel import icon

from skydango.config import EmoteConfig, WheelConfig
from skydango.device.base import KEYCODE_BACK
from skydango.game.emotes import EmotePlayer
from skydango.game.wheel import EmoteLibrary, Wheel, WheelError
from skydango.imageio import imwrite

PANEL_KEY = 46  # C


@pytest.fixture
def library(tmp_path):
    for name, kind in (("鞠躬", "circle"), ("欢呼", "cross"), ("指向", "person")):
        imwrite(tmp_path / f"{name}.png", icon(kind))
    return EmoteLibrary(tmp_path)


def make_player(library, cfg=None, panel=True, shown=False, slots=None):
    """真 Wheel，但打开编辑界面的 refresh / assign 换成直接改状态（编辑界面的流程在 test_wheel 里测）。"""
    device = FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])
    device.shown = shown
    press = device.key

    def key(code):  # BACK 会关掉输入框
        press(code)
        if code == KEYCODE_BACK:
            device.shown = False

    device.key = key
    t = [100.0]
    wheel = Wheel(device, WheelConfig(), library, sleep=lambda s: None, clock=lambda: t[0])
    state = dict(slots or {1: "鞠躬", 2: None, 3: None, 4: None, 5: None, 6: None, 7: "欢呼", 8: None})

    def refresh():
        device.calls.append(("refresh",))
        wheel.slots = dict(state)
        return {s: (n, 1.0) for s, n in state.items()}

    def assign(slot, name, force=False):
        device.calls.append(("assign", slot, name))
        state[slot] = name
        wheel.slots[slot] = name

    wheel.refresh = refresh
    wheel.assign = assign
    cfg = cfg or EmoteConfig(extra=["指向"], swap_slots=[7])
    manager, _ = fake_panel(device, open_=panel)
    player = EmotePlayer(device, wheel, cfg, manager, sleep=lambda s: None, clock=lambda: t[0])
    return player, device, t, state


def test_start_reads_wheel_with_panel_closed_then_reopens(library):
    player, device, _, _ = make_player(library)
    player.start()
    assert device.calls == [("hw_key", PANEL_KEY), ("refresh",), ("hw_key", PANEL_KEY)]
    assert player.original == {7: "欢呼"}


def test_start_leaves_panel_alone_when_closed(library):
    player, device, _, _ = make_player(library, panel=False)
    player.start()
    assert device.calls == [("refresh",)]


def test_unrecognized_swap_slot_is_not_used(library):
    player, _, _, _ = make_player(library, cfg=EmoteConfig(extra=["指向"], swap_slots=[6]))
    player.start()
    assert player.swap_slots == [] and player.available() == ["鞠躬", "欢呼"]


def test_config_filters_unknown_extra_and_locked_slots(library):
    player, _, _, _ = make_player(library, cfg=EmoteConfig(extra=["指向", "不存在"], swap_slots=[3, 7]))
    assert player.extra == ["指向"] and player.swap_slots == [7]


def test_available_skips_locked_slots(library):
    slots = {1: "鞠躬", 2: None, 3: "欢呼", 4: None, 5: None, 6: None, 7: None, 8: None}
    player, _, _, _ = make_player(library, cfg=EmoteConfig(), slots=slots)
    player.start()
    assert player.available() == ["鞠躬"]  # 3 号格锁定（道具），不给模型


def test_perform_on_wheel_presses_number_key_only(library):
    player, device, _, _ = make_player(library)
    player.start()
    device.calls.clear()
    assert player.perform("鞠躬") == 1
    assert device.calls == [("hw_key", 2)]  # KEY_1


def test_perform_closes_input_box_first(library):
    player, device, _, _ = make_player(library, shown=True)
    player.start()
    device.shown = True
    device.calls.clear()
    player.perform("鞠躬")
    assert device.calls == [("key", KEYCODE_BACK), ("hw_key", 2)]


def test_perform_swaps_extra_into_swap_slot_with_panel_closed(library):
    player, device, _, state = make_player(library)
    player.start()
    device.calls.clear()
    assert player.perform("指向") == 7
    assert device.calls == [("hw_key", PANEL_KEY), ("assign", 7, "指向"), ("hw_key", PANEL_KEY), ("hw_key", 8)]
    assert state[7] == "指向"


def test_perform_rejects_emote_not_on_wheel_or_whitelist(library):
    player, device, _, _ = make_player(library, cfg=EmoteConfig())
    player.start()
    device.calls.clear()
    with pytest.raises(WheelError):
        player.perform("指向")
    assert device.calls == []


def test_available_respects_intervals(library):
    player, _, t, _ = make_player(library)
    player.start()
    assert player.available() == ["鞠躬", "欢呼", "指向"]
    player.perform("鞠躬")
    assert player.available() == []  # min_interval 20 s 内
    t[0] += 21
    assert player.available() == ["鞠躬", "欢呼", "指向"]
    player.perform("指向")  # 换掉了 7 号格的欢呼
    t[0] += 21
    assert player.available() == ["鞠躬", "指向"]  # 换轮盘还在 120 s 限速内：只给轮盘上的
    t[0] += 120
    assert player.available() == ["鞠躬", "指向"]  # 欢呼不在白名单，换下来就没了



def test_available_can_ignore_emote_interval_but_not_swap_interval(library):
    player, _, t, _ = make_player(library)
    player.start()
    player.perform("指向")  # 换掉了 7 号格的欢呼
    assert player.available() == []
    assert player.available(ignore_interval=True) == ["鞠躬", "指向"]  # 动作限速不管；换轮盘限速照旧

def test_pretend_counts_for_intervals_without_touching_device(library):
    player, device, t, _ = make_player(library)
    player.start()
    device.calls.clear()
    player.pretend("指向")
    assert device.calls == [] and player.available() == []
    t[0] += 21
    assert player.available() == ["鞠躬", "欢呼"]  # 假装换过轮盘：换轮盘也在限速


def test_restore_puts_original_back(library):
    player, device, _, state = make_player(library)
    player.start()
    player.perform("指向")
    device.calls.clear()
    player.restore()
    assert device.calls == [("hw_key", PANEL_KEY), ("assign", 7, "欢呼"), ("refresh",), ("hw_key", PANEL_KEY)]
    assert state[7] == "欢呼"


def test_restore_noop_when_unchanged(library):
    player, device, _, _ = make_player(library)
    player.start()
    device.calls.clear()
    player.restore()
    assert device.calls == []


def test_reflex_emote_does_not_start_brain_cooldown(library):
    player, device, t, _ = make_player(library)
    player.start()
    player.perform("鞠躬", reflex=True)
    assert player.last_any == 100.0 and player.available() == ["鞠躬", "欢呼", "指向"]  # 大脑照样能做
    player.pretend("欢呼", reflex=True)
    assert player.last_emote == float("-inf") and player.last_any == 100.0
    t[0] = 105.0
    player.perform("鞠躬")
    assert player.available() == [] and player.last_any == 105.0
