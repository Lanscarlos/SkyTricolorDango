"""身体反射：判断"是不是在跟团子说话"、抽概率、额度、闲着计时（spec docs/superpowers/specs/2026-09-30-body-reflex-design.md）。

纯决策、不碰设备：身体每圈把新聊天、身边的好友、轮盘上的动作交进来，拿回"要不要开框 / 做哪个动作"，自己去做。
时间都用身体的 clock（monotonic）。
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

from ..config import ReflexConfig


class Reflexes:
    def __init__(self, cfg: ReflexConfig, rng: random.Random, now: float) -> None:
        self.cfg = cfg
        self.rng = rng
        self.used: deque[float] = deque()  # 窗口内做过的反射动作的时间
        self.recent: deque[tuple[float, str]] = deque(maxlen=10)  # (时间, 说明)：status / 网页用
        self._last_idle = ""
        self._idle_due = now
        self.stir(now)

    def stir(self, now: float, scale: float = 1.0) -> None:
        """有动静（有人说话、团子说话或做事）：闲着的计时从现在重新算。scale：困的时候 < 1，更勤。"""
        span = max(0.0, self.cfg.idle_max - self.cfg.idle_min)
        self._idle_due = now + (self.cfg.idle_min + self.rng.random() * span) * scale

    def left(self, now: float) -> int:
        while self.used and now - self.used[0] >= self.cfg.quota_window:
            self.used.popleft()
        return max(0, self.cfg.quota - len(self.used))

    def done(self, now: float, text: str, scale: float = 1.0) -> None:
        self.left(now)
        self.used.append(now)
        self.recent.append((now, text))
        self.stir(now, scale)

    @staticmethod
    def usable(names: Sequence[str], on_wheel: Sequence[str]) -> list[str]:
        return [n for n in names if n in on_wheel]

    def _chance(self, p: float) -> bool:
        return self.rng.random() < p

    def pick_addressed(self, now: float, on_wheel: Sequence[str], scale: float = 1.0) -> str | None:
        """scale：心情的倍数（开心 1.5、烦 0）。"""
        names = self.usable(self.cfg.addressed, on_wheel)
        if not names or not self.left(now) or not self._chance(min(1.0, self.cfg.addressed_chance * scale)):
            return None
        return self.rng.choice(names)

    def pick_return(self, now: float, label: str, on_wheel: Sequence[str]) -> str | None:
        name = self.cfg.return_map.get(label)
        if name is None or name not in on_wheel or not self.left(now) or not self._chance(self.cfg.return_chance):
            return None
        return name

    def pick_idle(self, now: float, on_wheel: Sequence[str]) -> str | None:
        if now < self._idle_due or not self.left(now):
            return None
        names = self.usable(self.cfg.idle, on_wheel)
        fresh = [n for n in names if n != self._last_idle] or names  # 别和上一次一样（只剩一个时照做）
        if not fresh:
            return None
        self._last_idle = self.rng.choice(fresh)
        return self._last_idle
