import pytest
from conftest import FakeDevice, scene

from skydango.brain.locomotion import Locomotion


def loco(shown=False):
    dev = FakeDevice([scene()])
    dev.shown = shown
    return Locomotion(dev, 0.3, sleep=lambda s: None), dev


def test_forward_presses_w_the_right_number_of_times():
    loc, dev = loco()
    assert loc.move("forward", 2) == "前进走了 2 步"
    assert dev.calls == [("hw_down", 17), ("hw_up", 17), ("hw_down", 17), ("hw_up", 17)]


def test_steps_clamped_to_default_max_steps():
    loc, dev = loco()
    loc.move("right", 10)
    assert dev.calls.count(("hw_down", 32)) == 3  # 默认 MAX_STEPS = 3


def test_custom_max_steps_used_for_owner_window():
    loc, dev = loco()
    loc.move("forward", 10, max_steps=6)
    assert dev.calls.count(("hw_down", 17)) == 6


def test_closes_input_box_first():
    loc, dev = loco(shown=True)
    loc.move("back", 1)
    assert dev.calls[0] == ("key", 4)


def test_unknown_direction_rejected():
    loc, _ = loco()
    with pytest.raises(ValueError):
        loc.move("up")
