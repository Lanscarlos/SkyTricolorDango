"""按每次调用的 usage 估算花费；最近一小时超了就少醒 / 停下。价格在 [brain] 里配（默认 Sonnet 5）。"""

from __future__ import annotations

from collections import deque

from ..config import BrainConfig

WINDOW = 3600.0


class Budget:
    def __init__(self, cfg: BrainConfig) -> None:
        self.cfg = cfg
        self._spent: deque[tuple[float, float]] = deque()  # (时间, 美元)

    def cost(self, usage: dict) -> float:
        fresh = usage.get("input_tokens") or 0
        read = usage.get("cache_read_input_tokens") or 0  # 缓存命中按 0.1 倍
        made = usage.get("cache_creation_input_tokens") or 0  # 写缓存按 1.25 倍
        out = usage.get("output_tokens") or 0
        return ((fresh + 0.1 * read + 1.25 * made) * self.cfg.price_input + out * self.cfg.price_output) / 1e6

    def record(self, usage: dict, now: float) -> float:
        usd = self.cost(usage)
        self._spent.append((now, usd))
        return usd

    def spent(self, now: float) -> float:
        while self._spent and now - self._spent[0][0] > WINDOW:
            self._spent.popleft()
        return sum(usd for _, usd in self._spent)

    def level(self, now: float) -> str:
        spent = self.spent(now)
        if spent >= self.cfg.pause_usd_per_hour:
            return "paused"
        if spent >= self.cfg.max_usd_per_hour:
            return "chat_only"
        return "ok"
