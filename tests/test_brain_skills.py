from types import SimpleNamespace

import numpy as np
import pytest

from skydango.brain.body import ToolError
from skydango.brain.events import EventQueue
from skydango.brain.skills import SkillRunner, SkillStep

FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeSkill:
    name = "track"
    goal = "盯着小明"

    def __init__(self, script=(), timeout=30.0):
        self.script = list(script)
        self.timeout = timeout
        self.started = 0
        self.ticks = 0
        self.stops = []

    def start(self, body, now):
        self.started += 1

    def tick(self, body, frame, now):
        self.ticks += 1
        step = self.script.pop(0) if self.script else SkillStep("running", "")
        if isinstance(step, Exception):
            raise step
        return step

    def stop(self, body, reason):
        self.stops.append(reason)


def setup(clock, skill):
    events = EventQueue(clock=clock)
    runner = SkillRunner(events, clock)
    body = SimpleNamespace(blackout=False)
    runner.start(body, skill)
    return runner, body, events


def kinds(events):
    return [(e.kind, e.text) for e in events.drain()]


def test_done_emits_event_and_clears(clock):
    skill = FakeSkill([SkillStep("running", "转向中"), SkillStep("done", "对准了")])
    runner, body, events = setup(clock, skill)
    runner.tick(body, FRAME, clock())
    assert runner.active is skill and kinds(events) == []
    runner.tick(body, FRAME, clock())
    assert kinds(events) == [("task_done", "盯着小明：对准了")]
    assert runner.active is None and len(skill.stops) == 1 and skill.started == 1


def test_failed_step_emits_task_failed(clock):
    skill = FakeSkill([SkillStep("failed", "跟丢了")])
    runner, body, events = setup(clock, skill)
    runner.tick(body, FRAME, clock())
    assert kinds(events) == [("task_failed", "盯着小明没做成：跟丢了")]


def test_timeout_fails(clock):
    skill = FakeSkill(timeout=5.0)
    runner, body, events = setup(clock, skill)
    clock.advance(6)
    runner.tick(body, FRAME, clock())
    [(kind, text)] = kinds(events)
    assert kind == "task_failed" and "超时" in text
    assert skill.ticks == 0 and runner.active is None and skill.stops


def test_exception_in_tick_fails_not_raises(clock):
    skill = FakeSkill([RuntimeError("坏了")])
    runner, body, events = setup(clock, skill)
    runner.tick(body, FRAME, clock())  # 不往外抛：身体主循环照常跑
    [(kind, text)] = kinds(events)
    assert kind == "task_failed" and "出错了" in text and runner.active is None and skill.stops


def test_exception_in_stop_still_clears(clock):
    skill = FakeSkill([SkillStep("done", "好了")])
    skill.stop = lambda body, reason: (_ for _ in ()).throw(RuntimeError("松键失败"))
    runner, body, events = setup(clock, skill)
    runner.tick(body, FRAME, clock())
    assert runner.active is None and kinds(events) == [("task_done", "盯着小明：好了")]


def test_blackout_fails(clock):
    skill = FakeSkill()
    runner, body, events = setup(clock, skill)
    body.blackout = True
    runner.tick(body, FRAME, clock())
    assert kinds(events) == [("task_failed", "盯着小明没做成：画面黑了")] and skill.ticks == 0


def test_second_start_refused(clock):
    runner, body, _ = setup(clock, FakeSkill())
    with pytest.raises(ToolError, match="stop_task"):
        runner.start(body, FakeSkill())


def test_start_failure_leaves_nothing_running(clock):
    runner = SkillRunner(EventQueue(clock=clock), clock)
    skill = FakeSkill()
    skill.start = lambda body, now: (_ for _ in ()).throw(ToolError("dry-run 不做"))
    with pytest.raises(ToolError):
        runner.start(SimpleNamespace(blackout=False), skill)
    assert runner.active is None


def test_cancel_calls_stop_and_emits_nothing(clock):
    skill = FakeSkill()
    runner, body, events = setup(clock, skill)
    assert "盯着小明" in runner.cancel(body, "大脑叫停")
    assert skill.stops == ["大脑叫停"] and runner.active is None and kinds(events) == []
    assert runner.cancel(body, "大脑叫停") == "没有在做的事"


def test_idle_tick_does_nothing(clock):
    runner = SkillRunner(EventQueue(clock=clock), clock)
    runner.tick(SimpleNamespace(blackout=True), FRAME, clock())
    assert runner.describe(clock()) == "没有在做的事"


def test_describe(clock):
    skill = FakeSkill([SkillStep("running", "已对准")])
    runner, body, _ = setup(clock, skill)
    clock.advance(6)
    runner.tick(body, FRAME, clock())
    assert runner.describe(clock()) == "正在做：盯着小明（第 6 秒）：已对准"
