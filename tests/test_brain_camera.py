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


def test_owner_window_can_raise_the_step_cap():
    c, dev, _ = cam(panel=False)
    c.move("right", 10, max_steps=8)
    assert dev.calls.count(("hw_down", 106)) == 8


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


# ---- spin：转一圈、按 fps 截图 ----
def spin_cam(panel=True):
    c, dev, state = cam(panel)
    clock = {"t": 0.0}
    c.clock = lambda: clock["t"]
    c.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    return c, dev, state


def test_spin_holds_right_and_captures_at_fps():
    c, dev, state = spin_cam()
    r = c.spin(lambda: scene(), turns=1, seconds_per_turn=1.0, fps=10)
    assert [round(t, 2) for t, _ in r.frames] == [round(0.1 * i, 2) for i in range(10)]
    assert dev.calls[0] == ("hw_key", 46) and dev.calls[-1] == ("hw_key", 46)
    assert dev.calls.count(("hw_down", 106)) == 1 and dev.calls.count(("hw_up", 106)) == 1
    assert r.seconds == pytest.approx(1.0) and r.panel_reopened and not r.blackout
    assert r.before is not None and r.after is not None
    assert c.describe() == "原位"  # 整圈不改偏移


def test_spin_releases_key_and_reopens_panel_on_error():
    c, dev, state = spin_cam()
    n = {"i": 0}

    def capture():
        n["i"] += 1
        if n["i"] == 3:
            raise RuntimeError("截图失败")
        return scene()

    with pytest.raises(RuntimeError):
        c.spin(capture, seconds_per_turn=1.0, fps=10)
    assert ("hw_up", 106) in dev.calls and state["panel"] is True


def test_spin_turns_multiply_duration():
    c, _, _ = spin_cam(panel=False)
    r = c.spin(lambda: scene(), turns=2, seconds_per_turn=1.0, fps=5)
    assert len(r.frames) == 10


def test_spin_reports_panel_not_reopened_and_blackout():
    c, dev, state = spin_cam()
    import numpy as np

    def hw_key(code):  # 按 C 只能关、开不回来
        dev.calls.append(("hw_key", code))
        if code == 46:
            state["panel"] = False

    dev.hw_key = hw_key
    frames = iter([scene()] + [np.zeros((1080, 1920, 3), np.uint8)] * 50)
    r = c.spin(lambda: next(frames), seconds_per_turn=0.5, fps=10)
    assert r.panel_reopened is False and r.blackout is True
