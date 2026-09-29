"""技能层：大脑说"去做一件事"，身体跟着主循环的节拍一圈一圈地做，做完 / 做不成发事件叫醒大脑。

大脑一轮要几秒到十几秒，盯人、走过去、点火这种要每秒修正好几次的事它来不及做；技能在身体线程里闭环，
大脑只管决定做不做、目标是谁，结果以 task_done / task_failed 为准。同时最多一个技能。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .events import EventQueue

log = logging.getLogger(__name__)

IDLE = "没有在做的事"


@dataclass(frozen=True)
class SkillStep:
    state: str  # "running" / "done" / "failed"
    note: str  # running：进度（进 status）；done / failed：结果（进事件）


class Skill(Protocol):
    name: str  # "track"
    goal: str  # 给大脑看的："盯着小明"
    timeout: float  # 秒
    # needs_camera: bool = True 的技能（要转镜头）开始时向 PanelManager 借面板（关掉），结束时归还；没写就是 False

    def start(self, body: Any, now: float) -> None: ...

    def tick(self, body: Any, frame: np.ndarray, now: float) -> SkillStep: ...  # 不许 sleep 超过 0.3 s：身体还要读聊天

    def stop(self, body: Any, reason: str) -> None: ...  # 松开按着的键；镜头不复原（交给 camera_reset）


class SkillRunner:
    def __init__(self, events: EventQueue, clock: Callable[[], float], panel=None) -> None:
        self.events = events
        self.clock = clock
        self.panel = panel  # chat.panel.PanelManager
        self._lease = None  # 借面板的上下文（needs_camera 的技能）
        self.active: Skill | None = None
        self._started = 0.0
        self._note = ""

    def start(self, body: Any, skill: Skill) -> str:
        """start 抛异常（比如 dry-run 不做）就不算开始。"""
        from .body import ToolError  # body 导入本模块，这里晚一点导入免得循环

        if self.active is not None:
            raise ToolError(f"正在{self.active.goal}，先 stop_task 再换")
        now = self.clock()
        skill.start(body, now)
        self.active, self._started, self._note = skill, now, ""
        if self.panel is not None and getattr(skill, "needs_camera", False):
            self._lease = self.panel.borrow("skill")
            self._lease.__enter__()
        log.info("开始做：%s", skill.goal)
        return f"开始{skill.goal}了，做完或做不成会告诉你"

    def tick(self, body: Any, frame: np.ndarray, now: float) -> None:
        skill = self.active
        if skill is None:
            return
        if body.blackout:
            step = SkillStep("failed", "画面黑了")
        elif now - self._started > skill.timeout:
            step = SkillStep("failed", f"超时了（{skill.timeout:.0f} 秒）")
        else:
            try:
                step = skill.tick(body, frame, now)
            except Exception as exc:
                log.exception("技能 %s 出错", skill.name)
                step = SkillStep("failed", f"出错了：{exc}")
        if step.state == "running":
            self._note = step.note
            return
        self._finish(body, step.note)
        if step.state == "done":
            self.events.put("task_done", f"{skill.goal}：{step.note}")
        else:
            self.events.put("task_failed", f"{skill.goal}没做成：{step.note}")
        log.info("%s：%s（%s）", "做完了" if step.state == "done" else "没做成", skill.goal, step.note)

    def cancel(self, body: Any, reason: str) -> str:
        """大脑自己叫停 / 身体要退出：不发事件（叫停的一方自己知道）。"""
        skill = self.active
        if skill is None:
            return IDLE
        self._finish(body, reason)
        log.info("停下了：%s（%s）", skill.goal, reason)
        return f"停下了：{skill.goal}"

    def describe(self, now: float) -> str:
        if self.active is None:
            return IDLE
        text = f"正在做：{self.active.goal}（第 {now - self._started:.0f} 秒）"
        return text + (f"：{self._note}" if self._note else "")

    def _finish(self, body: Any, reason: str) -> None:
        skill, self.active, self._note = self.active, None, ""
        try:
            skill.stop(body, reason)
        except Exception:  # 松键失败也要清掉，不然大脑再也开始不了新技能
            log.exception("技能 %s 收尾出错", skill.name)
        lease, self._lease = self._lease, None
        if lease is not None:
            try:
                lease.__exit__(None, None, None)
            except Exception:
                log.exception("技能 %s 归还聊天面板出错", skill.name)
