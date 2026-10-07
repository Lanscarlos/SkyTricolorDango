"""攒话（spec 2026-10-07-chat-pacing §1）：一批聊天什么时候交给大脑。纯计算，不碰设备、不读时钟（now 由调用方给）。

最后一句之后安静 quiet 秒（quiet_min ~ quiet_max 随机，有问句取 quiet_min）、批里说过话的人都不在打字才放；
在打字就从最后一次看到他打字起重新算；第一句进来过了 max_wait 一定放；urgent（难过类、卡洛的命令）马上放。"""

from __future__ import annotations

import math
import random
from collections.abc import Collection

from ..config import PacingConfig

QUESTION_WORDS = ("？", "?", "吗", "什么", "怎么", "为啥", "啥", "哪", "几")


def is_question(text: str) -> bool:
    return any(w in (text or "") for w in QUESTION_WORDS)


class Pacer:
    def __init__(self, cfg: PacingConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self._reset()

    def _reset(self) -> None:
        self._lines: list[tuple[str, str]] = []
        self._first = 0.0
        self._last = 0.0
        self._typed = float("-inf")  # 最后一次看到批里有人在打字
        self._quiet = 0.0
        self._urgent = False

    def heard(self, now: float, lines: list[tuple[str, str]], urgent: bool) -> None:
        """一批聊天进来：(说话人, 原文)。lines 为空时只是把攒着的这批标成马上放（没攒着就什么都不做）。"""
        if not lines:
            if urgent and self._lines:
                self._urgent = True
            return
        if not self._lines:
            self._first = now
            self._quiet = self.rng.uniform(self.cfg.quiet_min, self.cfg.quiet_max)
        self._lines.extend(lines)
        self._last = now
        if any(is_question(text) for _, text in lines):
            self._quiet = self.cfg.quiet_min  # 问了就答得快一点
        self._urgent = self._urgent or urgent

    def pending(self) -> bool:
        return bool(self._lines)

    def batch(self) -> list[tuple[str, str]]:
        return list(self._lines)

    def _speakers(self) -> list[str]:
        return list(dict.fromkeys(who for who, _ in self._lines if who))

    def _typing(self, typing: Collection[str]) -> bool:
        return any(who in typing for who in self._speakers())

    def ready(self, now: float, typing: Collection[str]) -> bool:
        """攒着的这批该不该交给大脑；typing = 头顶冒点点的人（OCR 读法给空的）。"""
        if not self._lines:
            return False
        if self._urgent or now - self._first >= self.cfg.max_wait:
            return True
        if self._typing(typing):
            self._typed = now
            return False
        return now - max(self._last, self._typed) >= self._quiet

    def released(self, now: float) -> None:
        self._reset()

    def describe(self, now: float, typing: Collection[str]) -> str:
        """status 里那一项：「在攒话：小明说了 3 句，等他说完（还在打字）/ 再等 4 秒」。"""
        if not self._lines:
            return ""
        speakers = self._speakers()
        who = "、".join(speakers) or "有人"
        head = f"在攒话：{who}说了 {len(self._lines)} 句，"
        if self._typing(typing):
            return head + f"等{'他们' if len(speakers) > 1 else '他'}说完（还在打字）"
        wait = self._quiet - (now - max(self._last, self._typed))
        wait = min(wait, self.cfg.max_wait - (now - self._first))
        return head + f"再等 {max(0, math.ceil(wait))} 秒"
