import pytest
from conftest import FakeDevice, fake_panel, scene

from skydango.brain.camera import Camera


class State(dict):
    """state["panel"] 读写假面板的开关（老测试的写法）。"""

    def __init__(self, panel_state):
        super().__init__()
        self.ps = panel_state

    def __getitem__(self, key):
        return self.ps.open

    def __setitem__(self, key, value):
        self.ps.open = value


def cam(panel=True, mode="always"):
    dev = FakeDevice([scene()])
    manager, ps = fake_panel(dev, open_=panel, mode=mode)
    return Camera(dev, 0.25, manager, sleep=lambda s: None), dev, State(ps)


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
def spin_cam(panel=True, mode="always"):
    c, dev, state = cam(panel, mode)
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


def test_spin_skips_failed_captures_and_finishes_the_turn():
    c, dev, state = spin_cam()
    n = {"i": 0}

    def capture():
        n["i"] += 1
        if n["i"] == 3:
            raise RuntimeError("截图失败")
        return scene()

    r = c.spin(capture, seconds_per_turn=1.0, fps=10)
    assert len(r.frames) == 9 and r.seconds == pytest.approx(1.0)  # 少一张，但照样转满一圈
    assert dev.calls.count(("hw_up", 106)) == 1 and state["panel"] is True


def test_spin_releases_key_and_reopens_panel_on_interrupt():
    c, dev, state = spin_cam()
    n = {"i": 0}

    def capture():
        n["i"] += 1
        if n["i"] == 3:
            raise KeyboardInterrupt
        return scene()

    with pytest.raises(KeyboardInterrupt):
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


def test_camera_without_panel_presses_nothing():
    dev = FakeDevice([scene()])
    Camera(dev, 0.25, None, sleep=lambda s: None).move("left")
    assert ("hw_key", 46) not in dev.calls


def test_camera_in_idle_auto_mode_leaves_panel_closed():
    c, dev, state = cam(panel=False, mode="auto")  # auto：闲着时面板本来关着
    c.move("left")
    assert ("hw_key", 46) not in dev.calls and state["panel"] is False


def test_spin_reports_restored_when_idle():  # 闲着不重开也算"回到该有的状态"
    c, dev, state = spin_cam(panel=False, mode="auto")
    r = c.spin(lambda: scene(), turns=1, seconds_per_turn=0.2, fps=10)
    assert r.panel_reopened is True and ("hw_key", 46) not in dev.calls


# ---- nudge：短按记秒数；reset：先粗转回去，再按参照图闭环细调 ----
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from skydango.brain.camera import REFINE_MAX, thumb_similarity  # noqa: E402


def texture(w=1920, h=1080, seed=1, cell=40):
    """平滑的随机纹理：平移几十像素内相关系数单调下降（模拟背景）。"""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // cell + 2, w // cell + 2), dtype=np.uint8)
    g = cv2.resize(small, ((w // cell + 2) * cell, (h // cell + 2) * cell), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return cv2.merge([g, g, g])


class TurnDevice(FakeDevice):
    """画面随镜头平移，每按一下移 30 + 360 × 秒 px（非线性，同 D0：0.02 s ≈ 37 px、0.1 s ≈ 66 px；左右对称）。"""

    MARGIN = 1000

    def __init__(self):
        super().__init__([scene()])
        self.pano = texture(1920 + 2 * self.MARGIN)
        self.px = 0.0  # 画面累计平移（右转为正）

    def hw_key_hold(self, code, seconds):
        super().hw_key_hold(code, seconds)
        step = 30 + 360 * seconds
        self.px += step if code == 106 else -step if code == 105 else 0.0

    def screenshot(self):
        x = self.MARGIN + int(round(self.px))
        return self.pano[:, x : x + 1920].copy()


class NoiseDevice(FakeDevice):
    """每次截图都是新的随机噪声：和参照图永远对不上。"""

    def __init__(self):
        super().__init__([scene()])
        self.rng = np.random.default_rng(7)

    def screenshot(self):
        return self.rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


def holds(dev):
    return [c for c in dev.calls if c[0] == "hw_hold"]


def test_nudge_holds_arrow_and_counts_seconds():
    c, dev, _ = cam(panel=False)
    assert c.nudge("right", 0.05) == pytest.approx(0.05)
    assert holds(dev) == [("hw_hold", 106, pytest.approx(0.05))]
    assert c.turn_seconds == pytest.approx(0.05)
    assert c.nudge("left", 0.5) == pytest.approx(0.1)  # 最长 0.1 s
    assert holds(dev)[-1] == ("hw_hold", 105, pytest.approx(0.1))
    assert c.turn_seconds == pytest.approx(-0.05)
    assert c.nudge("right", 0.001) == pytest.approx(0.02)  # 最短 0.02 s
    with pytest.raises(ValueError):
        c.nudge("up", 0.05)


def test_nudge_does_not_touch_panel():
    c, dev, state = cam(panel=True)
    dev.shown = True  # 输入框开着也不按 BACK（调用方是技能，已经借好面板）
    c.nudge("right", 0.05)
    assert state["panel"] is True
    assert ("hw_key", 46) not in dev.calls and ("key", 4) not in dev.calls


def test_nudge_remembers_reference_only_at_home():
    c, dev, _ = cam(panel=False)
    assert c.ref is None
    c.nudge("right", 0.05)
    first = c.ref
    assert first is not None
    dev.frames = [texture(seed=3)]  # 画面变了，但镜头已经不在原位：不覆盖
    c.nudge("right", 0.05)
    assert c.ref is first


def test_reset_undoes_nudges_then_refines_to_reference():
    dev = TurnDevice()
    c = Camera(dev, 0.25, None, sleep=lambda s: None)
    for _ in range(3):
        c.nudge("right", 0.1)
    dev.px += 40  # 反向按不准：多漂了 40 px（约一小步）
    result = c.reset()
    assert abs(dev.px) <= 20
    assert result.startswith("镜头转回原位") and "没对准" not in result
    assert c.turn_seconds == 0 and c.ref is None


def test_reset_replays_presses_with_same_durations():
    """转动和按键时长不成比例（D0）：秒数加总后分块反向按会差很远，要按同样的时长逐次反向重放。"""
    dev = TurnDevice()
    c = Camera(dev, 0.25, None, sleep=lambda s: None)
    for _ in range(10):
        c.nudge("right", 0.02)  # 10 × 37.2 px
    for _ in range(3):
        c.nudge("left", 0.1)  # 3 × 66 px；净秒数 -0.1 s，净平移却是 +174 px
    dev.px += 10  # 一点点误差
    result = c.reset()
    assert abs(dev.px) <= 20 and "没对准" not in result
    reverse = holds(dev)[13:26]  # 粗转：同样的时长反过来按
    assert sorted(reverse) == sorted([("hw_hold", 105, pytest.approx(0.02))] * 10 + [("hw_hold", 106, pytest.approx(0.1))] * 3)


def test_reset_without_reference_only_reverses():
    c, dev, _ = cam(panel=False)
    c.turns[0.1] = 2
    assert c.turn_seconds == pytest.approx(0.2)
    c.reset()
    assert holds(dev) == [("hw_hold", 105, pytest.approx(0.1))] * 2
    assert c.turn_seconds == 0 and not c.turns


def test_reference_taken_once_per_excursion():
    """净秒数 / 净次数回到 0 也不重拍参照图：每次离开原位只拍一次，reset 之后才清掉。"""
    c, dev, _ = cam(panel=False)
    for _ in range(5):
        c.nudge("right", 0.02)
    c.nudge("left", 0.1)  # 净秒数 0，但镜头其实不在原位
    first = c.ref
    dev.frames = [texture(seed=3)]
    c.nudge("right", 0.05)
    c.nudge("left", 0.05)
    for _ in range(5):
        c.nudge("left", 0.02)
    c.nudge("right", 0.1)  # 记账全部抵消
    c.nudge("right", 0.05)
    assert c.ref is first
    c.reset()
    assert c.ref is None
    c.nudge("right", 0.05)  # 复位之后是新的一次离开：重拍
    assert c.ref is not None and c.ref is not first


def test_forget_reference_keeps_accounting():
    c, dev, _ = cam(panel=False)
    c.nudge("right", 0.05)
    c.forget_reference()  # 身体走动 / 黑屏之后：参照图不作数了
    assert c.ref is None
    c.nudge("right", 0.05)
    assert c.ref is None  # 这次离开原位不再重拍
    dev.calls.clear()
    assert "没对准" not in c.reset()
    assert holds(dev) == [("hw_hold", 105, pytest.approx(0.05))] * 2  # 只粗转，不细调


def test_reset_refine_time_limit():
    dev = NoiseDevice()
    c = Camera(dev, 0.25, None)
    t = {"now": 0.0}
    c.clock = lambda: t["now"]
    c.sleep = lambda s: t.__setitem__("now", t["now"] + s)
    c.nudge("right", 0.05)
    dev.calls.clear()
    assert "没对准" in c.reset(refine_seconds=2.0)
    refine = holds(dev)[1:]
    assert len(refine) <= 2 * 6  # 每步等 0.4 s：2 秒内最多走五六步（含退回）
    assert sum(s if code == 106 else -s for _, code, s in refine) == pytest.approx(0.0)


def test_release_lifts_both_arrows():
    c, dev, _ = cam(panel=False)
    c.release()
    assert ("hw_up", 105) in dev.calls and ("hw_up", 106) in dev.calls


def test_reset_reports_when_reference_never_matches():
    dev = NoiseDevice()
    c = Camera(dev, 0.25, None, sleep=lambda s: None)
    c.nudge("right", 0.05)
    dev.calls.clear()
    result = c.reset()
    assert "没对准" in result
    refine = holds(dev)[1:]  # 第一下是粗转回去
    assert holds(dev)[0] == ("hw_hold", 105, pytest.approx(0.05))
    assert len(refine) <= REFINE_MAX * 2
    net = sum(s if code == 106 else -s for _, code, s in refine)
    assert net == pytest.approx(0.0)  # 找不到高峰：停在粗转的位置
    assert c.turn_seconds == 0 and c.ref is None


def test_thumb_similarity_ignores_chat_panel_area():
    a = texture(seed=5)
    b = a.copy()
    b[:, : int(1920 * 0.33)] = texture(seed=9)[:, : int(1920 * 0.33)]  # 只有左边聊天面板那块不一样
    b[int(1080 * 0.46) :, :] = 0  # 下半部分（人、按钮）不一样也不管
    assert thumb_similarity(a, b) > 0.99
    assert thumb_similarity(a, texture(seed=9)) < 0.5


# ---- look_person 换角度用：单步缩放、describe 说出小步转过 ----

def test_zoom_once_presses_one_step_and_counts():
    c, dev, state = cam(panel=True)
    dev.shown = True  # 不借面板、不按 BACK（调用方已经借好面板）
    assert "拉远了 1 步" in c.zoom_once("out")
    assert ("hw_key", 13) in dev.calls
    assert c.offset["zoom"] == -1
    c.zoom_once("in")
    assert c.offset["zoom"] == 0 and ("hw_key", 12) in dev.calls
    assert state["panel"] is True and ("key", 4) not in dev.calls
    with pytest.raises(ValueError):
        c.zoom_once("left")


def test_zoom_once_remembers_reference_before_leaving_home():
    c, _, _ = cam(panel=False)
    c.zoom_once("out")
    assert c.ref is not None


def test_describe_mentions_nudges():
    c, _, _ = cam(panel=False)
    c.nudge("right", 0.05)
    assert "右转了一点" in c.describe()
    c.nudge("left", 0.05)
    c.nudge("left", 0.05)
    assert "左转了一点" in c.describe()
    c.nudge("right", 0.05)
    assert c.describe() == "原位"


def test_describe_mixed_nudges_that_cancel_in_count_are_not_home():  # 右 0.05 + 左 0.1：净次数 0，但不在原位
    c, _, _ = cam(panel=False)
    c.nudge("right", 0.05)
    c.nudge("left", 0.1)
    assert "左转了一点" in c.describe()
