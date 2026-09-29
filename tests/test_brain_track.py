"""技能 track（盯着好友）：纯控制逻辑，假 body 按脚本给目标位置、假镜头记按键（plan 2026-09-29-brain-track Task 4）。"""

from types import SimpleNamespace

import numpy as np
import pytest

from skydango.brain.track import TrackSkill
from skydango.config import Config

FRAME = np.zeros((1080, 1920, 3), np.uint8)
NAME = "懒洋洋大王"


class FakeCam:
    def __init__(self):
        self.nudges = []
        self.resets = 0
        self.released = 0

    def nudge(self, direction, seconds):
        self.nudges.append((direction, seconds))
        return seconds

    def reset(self, refine_seconds=None):
        self.resets += 1
        return "镜头转回原位了"

    def release(self):
        self.released += 1


class FakeBody:
    """x = None 表示这一圈看不到目标。reader.panel_closed_since = None 表示聊天记录面板开着。"""

    def __init__(self, x=960.0):
        self.x = x
        self.cfg = Config()
        self.cfg.vision.mode = "log"
        self.reader = SimpleNamespace(panel_closed_since=0.0)
        self.camera = FakeCam()
        self.frame_width = 1920
        self.blackout = False
        self.asked = []

    def target_x(self, name, now):
        self.asked.append(name)
        return None if self.x is None else (float(self.x), "body")


def begin(clock, x=960.0, seconds=30):
    b = FakeBody(x)
    skill = TrackSkill(NAME, seconds)
    skill.start(b, clock())
    return skill, b


def at(skill, b, clock, t0, t):
    """把时钟拨到开始后第 t 秒，tick 一次。"""
    clock.t = t0 + t
    return skill.tick(b, FRAME, clock())


def test_skill_attributes():
    s = TrackSkill(NAME, 20)
    assert (s.name, s.goal, s.timeout) == ("track", f"盯着{NAME}", 25)
    assert s.needs_camera is True and s.quiet_people is True


def test_waits_settle_before_first_nudge(clock):
    t0 = clock()
    skill, b = begin(clock, x=1700)
    assert at(skill, b, clock, t0, 0.3).state == "running"
    assert b.camera.nudges == []  # 刚开始：借面板的动画、画面横移还没停
    assert at(skill, b, clock, t0, 0.7).state == "running"
    ((direction, seconds),) = b.camera.nudges
    assert direction == "right" and seconds == pytest.approx(0.1 * (1700 - 960) / 960, abs=0.001)
    assert b.asked and b.asked[-1] == NAME


def test_left_of_center_turns_left_and_clamps(clock):
    t0 = clock()
    skill, b = begin(clock, x=0)
    at(skill, b, clock, t0, 0.7)
    assert b.camera.nudges == [("left", pytest.approx(0.1))]  # 偏差 1 × 0.1 = 0.1，最长 0.1
    skill2, b2 = begin(clock, x=900)  # e ≈ -0.06，死区内
    t1 = clock()
    step = at(skill2, b2, clock, t1, 0.7)
    assert b2.camera.nudges == [] and step.state == "running" and "中间" in step.note


def test_small_error_uses_minimum_press(clock):
    t0 = clock()
    skill, b = begin(clock, x=960 + 0.16 * 960)  # 刚出死区：0.1 × 0.16 = 0.016 → 最短 0.02
    at(skill, b, clock, t0, 0.7)
    assert b.camera.nudges == [("right", pytest.approx(0.02))]


def test_no_second_nudge_until_settled(clock):
    t0 = clock()
    skill, b = begin(clock, x=1700)
    at(skill, b, clock, t0, 0.7)
    at(skill, b, clock, t0, 1.0)  # 按完才 0.3 s：画面还在动
    assert len(b.camera.nudges) == 1
    at(skill, b, clock, t0, 1.3)  # 0.6 s 后再按
    assert len(b.camera.nudges) == 2


def test_short_occlusion_is_not_lost(clock):
    t0 = clock()
    skill, b = begin(clock, x=960)
    assert at(skill, b, clock, t0, 1.0).state == "running"
    b.x = None  # 被别人挡住
    for t in (1.5, 2.0, 2.5, 3.0):
        step = at(skill, b, clock, t0, t)
        assert step.state == "running"
    assert "看不到" in step.note
    assert b.camera.nudges == []  # 看不到时不乱按键
    b.x = 1000
    assert at(skill, b, clock, t0, 3.0 + 0.1).state == "running"
    b.x = None
    assert at(skill, b, clock, t0, 5.5).state == "running"  # 从 3.1 s 重新算
    assert b.camera.nudges == []


def test_lost_after_three_seconds_reports_last_side(clock):
    t0 = clock()
    skill, b = begin(clock, x=1800)
    at(skill, b, clock, t0, 0.7)
    b.x = None
    assert at(skill, b, clock, t0, 2.0).state == "running"
    step = at(skill, b, clock, t0, 0.7 + 3.1)
    assert step.state == "failed" and "跟丢了" in step.note and "右边" in step.note


def test_stall_stops_nudging_near_target(clock):
    t0 = clock()
    skill, b = begin(clock, x=1400)  # 离团子太近：转了也移不动
    t = 0.7
    while len(b.camera.nudges) < 3:
        at(skill, b, clock, t0, t)
        t += 0.7
    assert [d for d, _ in b.camera.nudges] == ["right"] * 3
    step = at(skill, b, clock, t0, t)
    assert step.state == "running" and "转不动" in step.note
    end = t + 5.0
    while t < end:
        t += 0.5
        step = at(skill, b, clock, t0, t)
        assert step.state == "running" and "转不动" in step.note
    assert len(b.camera.nudges) == 3  # 停手期间不再按
    b.x = 1700  # 比停手时挪了 300 px（> 144）：解除
    at(skill, b, clock, t0, t + 0.5)
    assert len(b.camera.nudges) == 4 and b.camera.nudges[-1][0] == "right"


def test_progress_is_not_a_stall(clock):
    t0 = clock()
    skill, b = begin(clock, x=1700)
    t = 0.7
    for x in (1700, 1600, 1500, 1400):  # 每按一下都近了 100 px：不是转不动
        b.x = x
        at(skill, b, clock, t0, t)
        t += 0.7
    assert len(b.camera.nudges) == 4


def test_done_after_seconds(clock):
    t0 = clock()
    skill, b = begin(clock, x=960, seconds=5)
    assert at(skill, b, clock, t0, 4.9).state == "running"
    step = at(skill, b, clock, t0, 5.0)
    assert step.state == "done" and "盯了 5 秒" in step.note and "前面" in step.note


def test_stop_does_nothing_to_camera(clock):
    t0 = clock()
    skill, b = begin(clock, x=1700)
    at(skill, b, clock, t0, 0.7)
    before = list(b.camera.nudges)
    skill.stop(b, "大脑叫停")
    assert b.camera.nudges == before and b.camera.resets == 0  # 镜头不复原，交给 camera_reset


def test_stop_releases_arrow_keys(clock):
    skill, b = begin(clock)
    skill.stop(b, "身体停了")
    assert b.camera.released == 1  # Ctrl+C 打断按键时方向键可能没抬起：补一次


def test_waits_while_chat_panel_is_open(clock):
    """面板开着时方向键转不了视角：只等，不按键、不算转不动、不算跟丢。"""
    t0 = clock()
    skill, b = begin(clock, x=1400)
    b.reader.panel_closed_since = None  # 面板被重新打开了
    for t in (0.7, 1.4, 2.1, 2.8, 3.5):
        step = at(skill, b, clock, t0, t)
        assert step.state == "running" and "面板" in step.note
    b.x = None  # 面板挡住了左边：看不到也不算跟丢
    for t in (4.0, 6.0, 8.0):
        assert at(skill, b, clock, t0, t).state == "running"
    assert b.camera.nudges == []
    b.reader.panel_closed_since = t0 + 8.0  # 面板关上了（画面横移约 130 px）
    b.x = 1400
    at(skill, b, clock, t0, 8.1)
    assert b.camera.nudges == []  # 先等画面停稳
    for t in (8.7, 9.4, 10.1):
        at(skill, b, clock, t0, t)
    assert len(b.camera.nudges) == 3  # 面板开着那几圈没算进"转不动"


def test_panel_state_ignored_outside_log_mode(clock):
    t0 = clock()
    skill, b = begin(clock, x=1700)
    b.cfg.vision.mode = "bubble"  # 不读聊天记录面板：panel_closed_since 一直是 None，不代表面板开着
    b.reader.panel_closed_since = None
    at(skill, b, clock, t0, 0.7)
    assert len(b.camera.nudges) == 1


def test_runs_inside_skill_runner_and_borrows_panel(clock):
    from conftest import FakeDevice, fake_panel, scene

    from skydango.brain.events import EventQueue
    from skydango.brain.skills import SkillRunner

    dev = FakeDevice([scene()])
    panel, state = fake_panel(dev, open_=True)
    events = EventQueue(clock=clock)
    runner = SkillRunner(events, clock, panel=panel)
    b = FakeBody(x=960)
    runner.start(b, TrackSkill(NAME, 2))
    assert state.open is False  # 转镜头要关着面板
    t0 = clock()
    clock.t = t0 + 2.0
    runner.tick(b, FRAME, clock())
    assert runner.active is None and state.open is True
    (e,) = events.drain()
    assert e.kind == "task_done" and "盯了 2 秒" in e.text
