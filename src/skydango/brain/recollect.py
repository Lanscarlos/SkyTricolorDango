"""想起来的：按这一轮的人和话，从近况里挑几条放进唤醒消息（spec 2026-10-31-memory-entries §3.2）。

系统提示词只常驻长期条目（上线时生成一次、不变）；近况、被挪出常驻的长期条目、上线时 inbox 里没标长期的条目进候选池，
每次大脑醒来按这一批事件打分（人 + 话 + 新鲜度），带最多几条。一次上线（压缩换上之前）同一条只带一次。
纯规则、不调模型；出错只记日志、这一轮不带。身体线程里用。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime

from ..chat.entries import SELF, Entry, pinned_view, show
from ..config import ReplyConfig
from .events import Event

log = logging.getLogger(__name__)

STOP = {"一个", "这个", "那个", "什么", "不是", "没有", "我们", "你们", "他们", "自己", "就是", "还是", "可以", "知道", "今天", "明天"}
_RUN = re.compile(r"[0-9A-Za-z一-鿿]+")
PERSON_SCORE = 2.0  # 条目所在节是说话的人 / 刚来的人
SELF_SCORE = 0.5  # 自己那一节，有人在跟团子说话时
WORD_SCORE = 1.0  # 每个重合的两字片段
WORD_MAX = 3.0
NO_DATE_FRESH = 0.3  # 日期不详的新鲜度
ARRIVE_LATEST = 3  # 好友来了：先带他那一节最新的几条近况


def bigrams(text: str) -> set[str]:
    """只留汉字和字母数字，按连续片段取相邻两个字；单字不算，去掉太常见的词。"""
    out: set[str] = set()
    for run in _RUN.findall((text or "").lower()):
        out.update(run[i : i + 2] for i in range(len(run) - 1))
    return out - STOP


class Recollector:
    def __init__(
        self,
        load: Callable[[], tuple[str, str]],
        match: Callable[[str], str | None],
        cfg: ReplyConfig,
        wall: Callable[[], float],
    ) -> None:
        self.load = load  # () -> (notes.md 原文, inbox.md 原文)，每次现读
        self.match = match  # 说话人（OCR 可能错字 / 是叫法）→ 好友名（对应 notes 的 `## 节`）
        self.cfg = cfg
        self.wall = wall
        self._epoch = 0
        self._brought: set[tuple[str, str]] = set()
        notes, inbox = load()
        # 上线时 inbox 里的行才进候选池；之后新记的走「你刚记下：」，不重复
        self._inbox0 = {ln.strip() for ln in (inbox or "").splitlines() if ln.strip()}
        self._notes_seen: str | None = None
        self._pool: list[Entry] = []
        self._rebuild(notes, inbox)

    def _rebuild(self, notes: str, inbox: str) -> None:
        old_lines = "\n".join(ln for ln in (inbox or "").splitlines() if ln.strip() in self._inbox0)
        self._notes_seen = notes
        _text, self._pool = pinned_view(notes, old_lines, self._today(), self.cfg.pinned_max)

    def _today(self):
        return datetime.fromtimestamp(self.wall()).date()

    def line(self, events: list[Event], epoch: int) -> str | None:
        """唤醒消息里「想起来的：」那一段；什么都没想起来返回 None。"""
        try:
            return self._line(events, epoch)
        except Exception:
            log.warning("想起来的：挑选时出错，这一轮不带", exc_info=True)
            return None

    def _who(self, name: str) -> str:
        return self.match(name) or name

    def _line(self, events: list[Event], epoch: int) -> str | None:
        if epoch != self._epoch:  # 大脑历史压缩换上了：原话没了，可以再带
            self._epoch = epoch
            self._brought.clear()
        notes, inbox = self.load()
        if notes != self._notes_seen:
            self._rebuild(notes, inbox)
        today = self._today()
        speakers: set[str] = set()
        arrivals: list[str] = []
        said: list[str] = []
        talked_to = False
        for ev in events:
            if ev.kind in ("chat", "aside") and ev.speech:
                speakers.add(self._who(ev.speech[0]))
                said.append(ev.speech[1])
                talked_to = talked_to or ev.kind == "chat"
            elif ev.kind in ("arrive", "return") and ev.who:
                who = self._who(ev.who)
                if who not in arrivals:
                    arrivals.append(who)
        words = bigrams("\n".join(said))
        people = speakers | set(arrivals)
        if not people and not words:
            return None

        picked: list[tuple[Entry, str]] = []  # (条目, 日志里的分数说明)
        keys: set[tuple[str, str]] = set()

        def take(e: Entry, why: str) -> None:
            k = (e.section, e.raw)
            if k not in self._brought and k not in keys:
                keys.add(k)
                picked.append((e, why))

        # 好友来了：他那一节最新的几条近况，不管话题，也不受门槛限制（但算在上限里）
        for who in arrivals:
            mine = [e for e in self._pool if e.section == who]
            for e in sorted(mine, key=lambda e: e.date or datetime.min.date(), reverse=True)[:ARRIVE_LATEST]:  # sorted 稳定：一样新按文件顺序
                take(e, "来了")

        scored: list[tuple[float, int, Entry, str]] = []
        for i, e in enumerate(self._pool):
            person = PERSON_SCORE if e.section in people else (SELF_SCORE if e.section == SELF and talked_to else 0.0)
            hit = len(words & bigrams(e.text))
            word = min(WORD_SCORE * hit, WORD_MAX)
            if person == 0 and word == 0:
                continue
            fresh = NO_DATE_FRESH if e.date is None else 0.5 ** (max(0, (today - e.date).days) / self.cfg.recall_half_days)
            total = person + word + fresh
            if total < self.cfg.recall_min_score:
                continue
            scored.append((total, i, e, f"分数 {total:.2f} 人 {person:g} 话 {word:g} 新 {fresh:.2f}"))
        scored.sort(key=lambda x: (-x[0], x[1]))  # 一样高按文件顺序
        for _total, _i, e, why in scored:
            take(e, why)

        lines: list[str] = []
        chars = 0
        for e, why in picked:
            if len(lines) >= self.cfg.recall_per_turn:
                break
            text = f"- {'你自己' if e.section == SELF else e.section}：{show(e, today)}"
            if lines and chars + len(text) > self.cfg.recall_chars:
                break
            lines.append(text)
            chars += len(text)
            self._brought.add((e.section, e.raw))
            log.debug("想起来：%s：%s %s", e.section, e.text, why)
        if not lines:
            return None
        return "想起来的：\n" + "\n".join(lines)
