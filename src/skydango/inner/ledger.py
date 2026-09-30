"""内心账本（spec §1 §2）：好友关系卡 + 这一次上线，纯数据、不碰设备、不碰文件（读写在 store.py）。

身体线程每圈调 present / heard / said；拼给大脑看的文字（arrive 说明、status 的“身边的好友”）也在这里。
时间一律墙上时间（time.time()），日期按本地时区。
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from difflib import SequenceMatcher

from ..brain.occasion import ME
from ..chat.memory import format_gap
from ..chat.tracker import normalize, similar
from ..config import InnerConfig

log = logging.getLogger(__name__)

MATCH = 0.75  # 名字相似度门槛（同 occasion.is_friend_fn）
LINE_CHARS = 40  # last_line 最多存几个字


def day_of(t: float) -> str:
    """本地日期 "2026-09-30"。"""
    return time.strftime("%Y-%m-%d", time.localtime(t))


def ago(seconds: float) -> str:
    return "不到 1 分钟" if seconds < 60 else format_gap(seconds)


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def match_friend(name: str, friends: Sequence[str]) -> str | None:
    """OCR 出来的名字对到哪个好友：先过 similar，再取最像的（两个好友名字很像时别记错人）。"""
    name = (name or "").strip()
    if not name or name == ME:
        return None
    candidates = [f for f in friends if f and similar(name, f, MATCH)]
    if not candidates:
        return None
    n = normalize(name)
    return max(candidates, key=lambda f: SequenceMatcher(None, n, normalize(f)).ratio())


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in known})


@dataclass
class Card:
    """一个好友的关系卡。"""

    first_met: float
    last_seen: float | None = None  # None = 还没在身边见过（只在聊天里说过话）
    visits: int = 0
    days: list[str] = field(default_factory=list)  # 在身边或说过话的日期
    minutes: float = 0.0
    today: str = ""  # today_minutes / today_visits 是哪天的
    today_minutes: float = 0.0
    today_visits: int = 0
    lines: int = 0  # 他在聊天里说过几句
    to_me: int = 0  # 其中跟团子说的
    last_line: dict | None = None  # {"t": 时间, "text": 最多 40 字}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        return _from_dict(cls, d)


@dataclass
class Session:
    """一次上线（days.jsonl 的一行 / current.json）。"""

    start: float
    end: float | None = None
    live: bool = True
    ended: str = ""  # normal / crash / backfill
    friends: list[str] = field(default_factory=list)
    heard: int = 0
    said: int = 0
    summary: str = ""
    saved: float | None = None  # current.json 最后保存的时间

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        return _from_dict(cls, d)


def _today_minutes(card: Card, now: float) -> float:
    return card.today_minutes if card.today == day_of(now) else 0.0


def describe(name: str, card: Card, now: float) -> str:
    """小明（今天一起 40 分钟·认识 21 天·一起玩过 9 天）"""
    minutes = _today_minutes(card, now)
    bits = ["今天刚来" if minutes < 1 else f"今天一起 {int(minutes + 1e-6)} 分钟"]
    known = days_between(day_of(card.first_met), day_of(now))
    bits.append("今天刚认识" if known <= 0 else f"认识 {known} 天")
    if len(card.days) > 1:
        bits.append(f"一起玩过 {len(card.days)} 天")
    return f"{name}（{'·'.join(bits)}）"


def card_line(name: str, card: Card, now: float) -> str:
    """memory show 的一行。"""
    seen = "还没在身边见过" if card.last_seen is None else f"见过 {card.visits} 次，上次 {ago(now - card.last_seen)}前"
    return f"{describe(name, card, now)}；{seen}；说过 {card.lines} 句，跟你说过 {card.to_me} 句"


class Ledger:
    def __init__(
        self,
        cfg: InnerConfig,
        friends: Callable[[], list[str]],  # 好友名单（friends.md 的 ## 标题），每次现取
        now: float,
        cards: dict[str, Card] | None = None,
        history: list[Session] | None = None,  # 以前的上线（旧到新）
        store=None,  # inner.store.InnerStore
        persist: bool = False,  # 只有 live 写盘
        backfilled: float | None = None,  # people.json 是什么时候回填的（写回去时带上）
    ) -> None:
        self.cfg = cfg
        self.friends = friends
        self.cards: dict[str, Card] = dict(cards or {})
        self.history: list[Session] = list(history or [])
        self.store = store
        self.persist = persist
        self.backfilled = backfilled
        self.session = Session(start=now, live=persist)
        self._saved_at = float("-inf")
        self._closed = False
        self._prev: dict[str, float] = {}  # 每个好友上一次 present 的时刻（算时长）
        self._lock = threading.Lock()

    def card(self, name: str) -> Card | None:
        with self._lock:
            card = self.cards.get(name)
            return copy.deepcopy(card) if card is not None else None

    # ---- 身体调的 ----
    def present(self, names: Sequence[str], now: float) -> dict[str, str]:
        """这一圈身边的好友。返回这一圈算作新见面的好友 → arrive 说明（按更新前的卡写）。"""
        notes: dict[str, str] = {}
        with self._lock:
            friends = self.friends()
            done: set[str] = set()
            for raw in names:
                name = match_friend(raw, friends)
                if name is None or name in done:
                    continue
                done.add(name)
                card = self.cards.setdefault(name, Card(first_met=now))
                self._touch(card, name, now)
                last = card.last_seen
                if last is None or now - last > self.cfg.visit_gap:
                    card.visits += 1
                    card.today_visits += 1
                    notes[name] = self._arrive_note(card, last, now)
                    dt = 0.0
                else:
                    dt = min(max(0.0, now - self._prev.get(name, now)), self.cfg.max_step)
                card.minutes += dt / 60
                card.today_minutes += dt / 60
                card.last_seen = now if last is None else max(last, now)  # 时间往回跳时不往回改
                self._prev[name] = now
        return notes

    def heard(self, speaker: str, text: str, to_me: bool, now: float) -> None:
        """读到一句聊天（任何人说的；"我"不用传进来）。"""
        with self._lock:
            self.session.heard += 1
            name = match_friend(speaker, self.friends())
            if name is None:
                return
            card = self.cards.setdefault(name, Card(first_met=now))
            self._touch(card, name, now)
            card.lines += 1
            if to_me:
                card.to_me += 1
            card.last_line = {"t": now, "text": (text or "")[:LINE_CHARS]}

    def said(self, now: float) -> None:
        with self._lock:
            self.session.said += 1

    def save(self, now: float) -> bool:
        """live 时每 save_every 秒写一次 people.json + current.json（身体主循环每圈调）；返回写了没有。"""
        if not self.persist or self.store is None or self._closed or now - self._saved_at < self.cfg.save_every:
            return False
        with self._lock:
            self._saved_at = now
            self.session.saved = now
            self.store.write_people(self.cards, self.backfilled)
            self.store.write_current(self.session)
        return True

    def checkpoint(self, now: float) -> None:
        """身体收尾之后、让大脑写经过之前先落一次账：写经过时被强杀，下次启动也算正常下线（只丢经过）。"""
        if not self.persist or self.store is None or self._closed:
            return
        with self._lock:
            self.session.end = now
            self.session.ended = "normal"
            self.session.saved = now
            self.store.write_people(self.cards, self.backfilled)
            self.store.write_current(self.session)

    def close(self, summary: str, now: float) -> None:
        """退出：这一次追加进 days.jsonl、写 people.json、删 current.json。只生效一次；不写盘时什么都不做。"""
        if self._closed:
            return
        self._closed = True
        if not self.persist or self.store is None:
            return
        with self._lock:
            self.session.end = now
            self.session.ended = "normal"
            self.session.summary = summary or ""
            self.session.saved = now
            self.store.write_people(self.cards, self.backfilled)
            self.store.append_day(self.session)
            self.store.drop_current()
            self.history.append(self.session)

    # ---- 给大脑看的 ----
    def status_line(self, names: Sequence[str], now: float) -> str:
        """status 的“身边的好友”：有卡的带交情，没卡的只写名字。"""
        with self._lock:
            friends = self.friends()
            parts = []
            for raw in names:
                name = match_friend(raw, friends)
                card = self.cards.get(name) if name else None
                parts.append(describe(name, card, now) if card is not None else raw)
        return "、".join(parts)

    def days_prompt(self, now: float, diaries: list[str] | None = None) -> str:
        """系统提示词的「日子」一节（启动时算一次）；diaries：最近几篇日记（第 2 期）。"""
        from .days import days_prompt

        with self._lock:
            return days_prompt(self.history, self.cards, self.friends(), now, self.cfg.long_gap, diaries)

    # ---- 内部 ----
    def _touch(self, card: Card, name: str, now: float) -> None:
        today = day_of(now)
        if card.today != today:  # 跨了日期：今天的时长、次数清零
            card.today, card.today_minutes, card.today_visits = today, 0.0, 0
        if today not in card.days:
            card.days.append(today)
        if name not in self.session.friends:
            self.session.friends.append(name)

    def _arrive_note(self, card: Card, last: float | None, now: float) -> str:
        if last is None:
            return "（第一次在身边见到）"
        gap = now - last
        far = "，好久没见了" if gap > self.cfg.long_gap * 86400 else ""
        today = "今天第一次" if card.today_visits == 1 else f"今天第 {card.today_visits} 次"
        note = f"（第 {card.visits} 次见；上次 {ago(gap)}前{far}；{today}）"
        line = card.last_line
        if line and day_of(line["t"]) < day_of(now):  # 今天说的聊天里本来就看得到
            note += f"。他上次最后说的（{ago(now - line['t'])}前）：「{line['text']}」"
        return note
