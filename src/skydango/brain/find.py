"""技能 find：大脑让找某个好友（spec 2026-10-03-attention-search §3）。

包一个 Search.find：他刚走开过就往他走的方向找，没线索就先喊一声、看名字贴在哪边，没有就转一圈；
找到（或者只是像他）就停下、task_done，没找到 task_failed。镜头不复原（交给 camera_reset），要一直盯着由大脑接着调 track。
用自己的 Heading：别和空闲注意力的方位记忆互相干扰（技能在跑时身体每圈都当"别人在动镜头"清注意力的朝向）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .calling import call_available
from .search import Heading, Search
from .skills import SkillStep
from .track import TrackSkill

log = logging.getLogger(__name__)


class FindSkill:
    name = "find"
    needs_camera = True  # SkillRunner 开始时借走聊天面板（开着时方向键转不了视角）
    quiet_people = True  # 找的时候人进出画面是自己转的：不报人来人走

    def __init__(self, name: str, seconds: float, side: str | None, edge_exit: bool) -> None:
        self.target = name
        self.seconds = float(seconds)
        self.side, self.edge_exit = side, edge_exit
        self.goal = f"找{name}"
        self.timeout = self.seconds

    def start(self, body: Any, now: float) -> None:
        cfg = body.cfg
        self.heading = Heading(cfg.attention.press_deg, cfg.track.nudge_max, cfg.spin.hfov, now)
        can_call = bool(cfg.call.enabled) and call_available(cfg, body.env)
        self.search = Search.find(self.target, self.side, self.edge_exit, can_call, cfg.attention, cfg.track, self.heading, now)
        self._pending = None  # 喊出去、还没结果的那一声（CallResult）

    def tick(self, body: Any, frame: np.ndarray, now: float) -> SkillStep:
        s = self.search
        if self._pending is not None:
            seen = body.env.call_result(self._pending.at)
            if seen is not None or now - self._pending.at > body.cfg.call.window + 10:
                s.called(seen, now)
                self._pending = None
        if TrackSkill._panel_open(body):  # 聊天面板开着方向键没用：等（不算没进展）
            s.pressed_wait(now)
            return SkillStep("running", "等聊天面板关上")
        st = s.step(body.search_obs(self.target, now), now)
        if st.state == "found":
            return SkillStep("done", f"找到了{st.note}")
        if st.state == "maybe":
            return SkillStep("done", f"可能是{st.note}")
        if st.state == "none":
            return SkillStep("failed", st.note)
        if st.state == "call":
            try:
                r = body.call_out("brain")
            except Exception:  # 喊不成不该拖垮找：当没喊成，接着下一步
                log.exception("找%s：喊一声出错", self.target)
                s.called(None, now)
                return SkillStep("running", s.describe())
            if r.refused or r.dry:
                log.debug("找%s：喊不了（%s）", self.target, r.refused or "dry-run")
                s.called(None, now)
            else:
                self._pending = r
                s.call_sent(now)
            return SkillStep("running", s.describe())
        if st.state == "press":
            body.camera.nudge(st.turn.direction, st.turn.seconds)
            s.pressed(st.turn, now)
        return SkillStep("running", s.describe())

    def stop(self, body: Any, reason: str) -> None:
        """同 track：左右各补一次抬起。镜头不复原。"""
        release = getattr(body.camera, "release", None)
        if release is not None:
            release()
