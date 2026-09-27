import pytest
from conftest import FakeDevice, scene

from skydango.brain.camera import Camera


def cam(panel=True):
    dev = FakeDevice([scene()])
    state = {"panel": panel}

    def hw_key(code):
        dev.calls.append(("hw_key", code))
        if code == 46:
            state["panel"] = not state["panel"]

    dev.hw_key = hw_key
    return Camera(dev, 0.25, lambda: state["panel"], 46, sleep=lambda s: None), dev, state


def test_turn_closes_panel_holds_arrow_and_reopens():
    c, dev, state = cam()
    assert c.move("left", 2) == "左转了 2 步"
    assert dev.calls == [
        ("hw_key", 46), ("hw_down", 105), ("hw_up", 105), ("hw_down", 105), ("hw_up", 105), ("hw_key", 46)
    ]
    assert state["panel"] is True


def test_zoom_is_a_key_press_and_reset_undoes_everything():
    c, dev, _ = cam(panel=False)
    c.move("zoom_in")
    c.move("right")
    c.move("up")
    assert c.describe() == "右转了 1 步，往上看了 1 步，拉近了 1 步"
    dev.calls.clear()
    assert c.reset().startswith("镜头转回原位")
    assert ("hw_down", 105) in dev.calls and ("hw_down", 108) in dev.calls and ("hw_key", 13) in dev.calls
    assert c.describe() == "原位"
    assert c.reset() == "镜头已经在原位"


def test_closes_input_box_first_and_limits_steps():
    c, dev, _ = cam(panel=False)
    dev.shown = True
    c.move("right", 10)
    assert dev.calls[0] == ("key", 4)
    assert dev.calls.count(("hw_down", 106)) == 4


def test_unknown_action_rejected():
    c, _, _ = cam()
    with pytest.raises(ValueError):
        c.move("jump")


def test_arrow_key_is_released_even_when_interrupted():
    c, dev, _ = cam(panel=False)

    def boom(s):
        raise KeyboardInterrupt

    c.sleep = boom
    with pytest.raises(KeyboardInterrupt):
        c.move("left", 3)
    assert ("hw_up", 105) in dev.calls  # 方向键一定松开，不然镜头会一直转
    assert c.offset["turn"] == 0  # 第一步就被打断：这一步没走完，不记


def test_offset_counts_each_finished_step():
    c, dev, _ = cam(panel=False)
    presses = []

    def sleep(s):
        presses.append(s)
        if len(presses) == 3:  # 每步两次 sleep（按住、松开后）：第 3 次是第 2 步按住时，这时出错
            raise RuntimeError("adb 失败")

    c.sleep = sleep
    with pytest.raises(RuntimeError):
        c.move("right", 3)
    assert c.offset["turn"] == 1


def test_around_captures_four_directions_and_turns_full_circle():
    c, dev, state = cam()
    shots = []
    frames = c.around(lambda: shots.append(len(shots)) or len(shots))
    assert frames == [1, 2, 3, 4]
    assert dev.calls.count(("hw_down", 106)) == 8  # 每 90°（2 步）一张，最后再转 90° 回到原来的朝向
    assert c.offset["turn"] == 0 and state["panel"] is True
