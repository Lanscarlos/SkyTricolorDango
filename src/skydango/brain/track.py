"""技能 track：小步转镜头，把某个好友保持在画面中间（计划 docs/superpowers/plans/2026-09-29-brain-track.md）。

纯控制逻辑，不碰设备 / YOLO：位置来自 body.target_x（人物框优先、名字标签兜底），按键交给 body.camera.nudge。
每圈按名字重新找目标，不依赖轨迹 id；转镜头之后不平移轨迹框（D0：近处的人和远处的背景横移差好几倍）。

- 方向：右转画面里的东西往左移 → 目标在中线右边就按右键
- 比例控制：按键秒数 = gain × 偏差，夹到 nudge_min ~ nudge_max；偏差在死区里不动
- 按完等 settle 秒画面停稳再看（不 sleep：记下按键时间，之后几圈不动）；刚开始也先等 settle（借面板的动画、画面横移）
- 离团子太近的人转镜头时横移有上限，可能永远进不了死区：同方向连续按 stall_nudges 次误差没缩小 stall_px，就停手，
  直到目标相对停手时挪出死区宽度再重新开始
- lost_after 秒看不到 → 跟丢；到时间 → 做完。结束时不复原镜头（交给 camera_reset）
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..vision.people import side_of
from .skills import SkillStep

EPS = 1e-6  # 时间比较留一点余量（浮点）


class TrackSkill:
    name = "track"
    needs_camera = True  # SkillRunner 开始时借走聊天面板（开着时方向键转不了视角）
    quiet_people = True  # 跟踪期间身体不报人来人走（人进出画面是自己转的）

    def __init__(self, name: str, seconds: float) -> None:
        self.target = name
        self.seconds = float(seconds)
        self.goal = f"盯着{name}"
        self.timeout = self.seconds + 5

    def start(self, body: Any, now: float) -> None:
        self.cfg = body.cfg.track
        self._start = now
        self._last_nudge = float("-inf")
        self._seen_at = now  # 开始前身体已经确认看得到
        self._seen = True
        self._side = "前面"
        self._run_dir: str | None = None  # 这一串同方向按键：方向、按了几次、开始时的误差（像素）
        self._run_len = 0
        self._run_err = 0.0
        self._stalled_cx: float | None = None  # 停手时目标的 x；None = 没停手

    def tick(self, body: Any, frame: np.ndarray, now: float) -> SkillStep:
        cfg = self.cfg
        width = body.frame_width
        half = width / 2
        found = body.target_x(self.target, now)
        if found is not None:
            cx = found[0]
            self._seen_at, self._seen, self._side = now, True, side_of(cx, width)
        else:
            self._seen = False

        if now - self._start >= self.seconds - EPS:
            where = f"在{self._side}" if self._seen else f"最后在{self._side}"
            return SkillStep("done", f"盯了 {self.seconds:.0f} 秒，{self.target}{where}")
        if found is None:
            if now - self._seen_at > cfg.lost_after:
                return SkillStep("failed", f"跟丢了（最后在{self._side}）")
            return SkillStep("running", f"看不到{self.target}，等一下")
        if body.camera is None:
            return SkillStep("failed", "没有镜头，转不了")
        if now - self._start < cfg.settle - EPS or now - self._last_nudge < cfg.settle - EPS:
            return SkillStep("running", f"盯着{self.target}（{self._side}）")

        err = cx - half
        if abs(err) < cfg.deadband * half:
            self._run_dir, self._run_len, self._stalled_cx = None, 0, None
            return SkillStep("running", f"{self.target}在中间")
        if self._stalled_cx is not None:
            if abs(cx - self._stalled_cx) <= cfg.deadband * half:
                return SkillStep("running", f"{self.target}就在旁边，转不动了")
            self._stalled_cx, self._run_dir, self._run_len = None, None, 0  # 目标挪开了：重新开始

        direction = "right" if err > 0 else "left"
        if direction != self._run_dir:
            self._run_dir, self._run_len, self._run_err = direction, 0, abs(err)
        elif self._run_len >= cfg.stall_nudges:
            if self._run_err - abs(err) < cfg.stall_px:  # 按了几下误差几乎没变：转不动
                self._stalled_cx, self._run_dir, self._run_len = cx, None, 0
                return SkillStep("running", f"{self.target}就在旁边，转不动了")
            self._run_len, self._run_err = 0, abs(err)  # 在变近：重新数下一串
        seconds = min(max(cfg.gain * abs(err) / half, cfg.nudge_min), cfg.nudge_max)
        body.camera.nudge(direction, seconds)
        self._last_nudge = now
        self._run_len += 1
        return SkillStep("running", f"{self.target}在{self._side}，往{'右' if direction == 'right' else '左'}转一点")

    def stop(self, body: Any, reason: str) -> None:
        """一次按键按完即松，没有按着的键；镜头不复原（交给 camera_reset）。"""
