"""空闲注意力（东张西望）：团子闲着时"现在最想看什么"，纯决策（spec 2026-09-30-idle-attention §2 / §3）。

每圈给一组候选目标（说话的人、走近的人、对团子做动作的人、站着的好友），按基础兴趣 ×（1 − 看腻）挑一个看；
目标在画面中间带里就不动、慢慢看腻，不在就小步转过去（同 track：目标在右边按右，把它拉向中间）；
换目标要高出一截、刚换过的一会儿内不换（防抖）。不回位：看腻了就看下一个。
这里不碰设备：Body 判断闲不闲、真按了键再调 pressed()。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..config import AttentionConfig, TrackConfig
from .peek import Turn

log = logging.getLogger(__name__)

KINDS = ("talk_friend", "act_on_me", "approach", "talk_stranger", "friend_present")
KIND_NOTES = {"talk_friend": "在说话", "act_on_me": "在对你做动作", "approach": "朝你走过来",
              "talk_stranger": "在说话", "friend_present": "站在那"}
FRESH_KINDS = ("talk_friend", "talk_stranger", "approach", "act_on_me")  # 刚出现时值得"先看一眼再开面板"
MODES = ("随意", "好奇", "专心", "别动")


@dataclass(frozen=True)
class Target:
    key: str  # "t:<轨迹 id>" 或 "n:<名字>"：同一个人
    kind: str  # KINDS 之一
    x: float  # 框中心 x（整图像素）
    who: str | None  # 名字；陌生人 None
    fresh: float | None = None  # 这次刺激开始的时间（气泡 start / 走近 / 动作）；变了就把看腻清零


@dataclass(frozen=True)
class Thought:
    current: Target | None  # 现在在看的
    action: Turn | None  # 这一圈想按的（门槛不开时 Body 不按）
    centered: bool  # current 已经在中间带里
    wandering: bool  # 在随意看
    look_first: bool  # current 是刚出现的说话 / 走近 / 做动作：值得先看再开面板


class Attention:
    def __init__(self, cfg: AttentionConfig, track: TrackConfig, rng: random.Random, now: float) -> None:
        self.cfg, self.track, self.rng = cfg, track, rng
        self.width = 1920
        self.mode = "随意"
        self.focus: str | None = None
        self._last_now = now
        self._bored: dict[str, float] = {}
        self._fresh: dict[str, float | None] = {}
        self._current: Target | None = None
        self._switched_at = float("-inf")
        self._last_press = float("-inf")
        self._err = 0.0  # 这一圈 current 离中线多远（给 pressed 记转不动用）
        self._streak_dir: str | None = None  # 转不动（同 track）：同方向连按几下、开始时离中线多远
        self._streak_n = 0
        self._streak_err = 0.0
        self._thought = Thought(None, None, False, False, False)

    # ---- 兴趣 ----
    def base(self, t: Target) -> float:
        return float(getattr(self.cfg, t.kind))

    def _centered(self, t: Target) -> bool:
        return abs(t.x - self.width / 2) <= self.cfg.center_band * self.width / 2

    def interest(self, t: Target) -> float:
        return self.base(t) * (1.0 - self._bored.get(t.key, 0.0))

    def _merge(self, targets: list[Target]) -> dict[str, Target]:
        """同一个人几种目标：只留基础兴趣最高的那种。"""
        out: dict[str, Target] = {}
        for t in targets:
            if t.key not in out or self.base(t) > self.base(out[t.key]):
                out[t.key] = t
        return out

    def _update_boredom(self, merged: dict[str, Target], dt: float) -> None:
        for key, t in merged.items():
            if t.fresh is not None and self._fresh.get(key) != t.fresh:
                self._fresh[key] = t.fresh
                self._bored[key] = 0.0  # 新的一句 / 新的一次走近：又有意思了
        for key in set(self._bored) | set(merged):
            t = merged.get(key)
            b = self._bored.get(key, 0.0)
            if t is not None and self._centered(t):
                b += self.cfg.bore_rate * dt
            else:
                b -= self.cfg.recover_rate * dt
            b = min(max(b, 0.0), 1.0)
            if b == 0.0 and t is None:
                self._bored.pop(key, None)
                self._fresh.pop(key, None)
            else:
                self._bored[key] = b

    def _choose(self, merged: dict[str, Target], now: float) -> Target | None:
        ok = [t for t in merged.values() if self.interest(t) >= self.cfg.min_interest]
        best = max(ok, key=self.interest, default=None)
        cur = merged.get(self._current.key) if self._current is not None else None
        if cur is not None and self.interest(cur) < self.cfg.min_interest:
            cur = None
        if cur is None:
            pick = best
        elif (best is not None and best.key != cur.key and now - self._switched_at >= self.cfg.switch_hold
              and self.interest(best) >= self.interest(cur) + self.cfg.switch_margin):
            pick = best
        else:
            pick = cur
        if (pick is None) != (self._current is None) or (pick is not None and pick.key != self._current.key):
            self._switched_at = now
            self._streak_dir, self._streak_n = None, 0
            if pick is not None:
                log.debug("注意力换到 %s（%s，兴趣 %.2f）", pick.who or "陌生人", pick.kind, self.interest(pick))
        return pick

    # ---- 每圈 ----
    def think(self, targets: list[Target], now: float, wander_scale: float = 1.0) -> Thought:
        dt = min(max(now - self._last_now, 0.0), self.cfg.max_step)
        self._last_now = now
        merged = self._merge(targets)
        self._update_boredom(merged, dt)
        cur = self._current = self._choose(merged, now)
        action, centered, look_first = None, False, False
        if cur is not None:
            centered = self._centered(cur)
            look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
            if not centered and now - self._last_press >= self.track.settle:
                action = self._turn(cur)
                if action is None:  # 转不动、刚加了看腻：这一圈就重新挑（可能换走）
                    cur = self._current = self._choose(merged, now)
                    if cur is not None:
                        centered = self._centered(cur)
                        look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
        self._thought = Thought(cur, action, centered, False, look_first)
        return self._thought

    def _turn(self, cur: Target) -> Turn | None:
        half = self.width / 2
        err = abs(cur.x - half)
        direction = "right" if cur.x > half else "left"  # 按右键画面里的东西往左移：目标在右边就按右
        if (direction == self._streak_dir and self._streak_n >= self.track.stall_nudges
                and self._streak_err - err < self.track.stall_px):
            self._bored[cur.key] = min(1.0, self._bored.get(cur.key, 0.0) + self.cfg.stuck_bored)
            self._streak_dir, self._streak_n = None, 0
            log.debug("注意力转不动 %s，看腻一截", cur.who or "陌生人")
            return None
        self._err = err
        seconds = min(max(self.cfg.gain * err / half, self.track.nudge_min), self.track.nudge_max)
        return Turn(direction, round(seconds, 3))

    def pressed(self, turn: Turn, now: float) -> None:
        """Body 真按了这一下：记按键时间、转不动计数（隔太久说明别人可能动过镜头，重新计数）。"""
        if turn.direction != self._streak_dir or now - self._last_press > 3 * self.track.settle:
            self._streak_dir, self._streak_n, self._streak_err = turn.direction, 0, self._err
        self._streak_n += 1
        self._last_press = now

    def describe(self) -> str:
        cur = self._thought.current
        if cur is not None:
            return f"在看：{cur.who or '陌生人'}（{KIND_NOTES[cur.kind]}）"
        return "闲着随意看" if self._thought.wandering else "没在看什么"
