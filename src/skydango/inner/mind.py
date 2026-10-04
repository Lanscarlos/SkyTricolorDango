"""心情、别扭、心愿（spec 2026-09-30-inner-phase2 §1 §4）：数据 + 代码守的规矩 + 解析反思结果。纯数据，不碰文件（读写在 store.py）。

反思（模型）只提建议；这里决定收不收：别扭只冲够熟的好友、有时限、最多 1 条，心愿最多几条、会过期，不认识的一律丢。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field

from ..chat.tracker import similar
from ..config import InnerConfig
from .ledger import Card, ago, match_friend

MOODS = ("开心", "平常", "低落", "烦")
KINDS = ("惦记", "想做", "小心思")
TEXT_CHARS = 40
_KIND_LABEL = {"惦记": "惦记", "想做": "想", "小心思": "小心思"}
# 别扭对象说了这些就立刻消气（提示词之外的代码兜底：光遇未成年玩家多）
DISTRESS = ("难过", "伤心", "不舒服", "难受", "想哭", "哭了", "委屈", "不开心", "心情不好", "认真的", "别闹了", "生气了")


def sounds_upset(text: str) -> bool:
    return any(w in (text or "") for w in DISTRESS)


@dataclass
class Mood:
    level: str = "平常"
    text: str = ""  # 一句带原因的话
    since: float = 0.0  # 这个档位从什么时候开始


@dataclass
class Grudge:
    who: str
    why: str
    since: float
    until: float


@dataclass
class Want:
    kind: str  # 惦记 / 想做 / 小心思
    text: str
    who: str = ""  # 只有“惦记”挂在好友名下
    since: float = 0.0
    until: float = 0.0  # 0 = 不过期


@dataclass
class Mind:
    mood: Mood = field(default_factory=Mood)
    grudge: Grudge | None = None
    wants: list[Want] = field(default_factory=list)
    updated: float | None = None
    calm: dict[str, float] = field(default_factory=dict)  # 刚消气的人 → 到什么时候之前不再对他闹别扭

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Mind:
        g = d.get("grudge")
        return cls(
            mood=Mood(**d.get("mood") or {}),
            grudge=Grudge(**g) if g else None,
            wants=[Want(**w) for w in d.get("wants") or []],
            updated=d.get("updated"),
            calm={str(k): float(v) for k, v in (d.get("calm") or {}).items()},
        )

    # ---- 套进反思结果 ----
    def apply(self, result: dict, cards: dict[str, Card], friends: list[str], now: float, cfg: InnerConfig) -> list[str]:
        """返回丢掉了哪些（记日志用）。"""
        dropped: list[str] = []
        self.updated = now
        if "mood" in result:
            self._apply_mood(result["mood"], now, dropped)
        if "grudge" in result:
            self._apply_grudge(result["grudge"], cards, friends, now, cfg, dropped)
        done = result.get("wants_done", [])
        if isinstance(done, list):
            for text in done:
                if isinstance(text, str) and text.strip():
                    self.wants = [w for w in self.wants if not similar(text, w.text, 0.75)]
        else:
            dropped.append("wants_done 不是列表")
        add = result.get("wants_add", [])
        if isinstance(add, list):
            for item in add:
                self._add_want(item, friends, now, cfg, dropped)
        else:
            dropped.append("wants_add 不是列表")
        while len(self.wants) > cfg.wants_max:
            dropped.append(f"心愿太多，挤掉最旧的：{self.wants.pop(0).text}")
        return dropped

    def _apply_mood(self, mood, now: float, dropped: list[str]) -> None:
        if not isinstance(mood, dict):
            dropped.append("mood 格式不对")
            return
        level = mood.get("level")
        if level not in MOODS:
            dropped.append(f"不认识的心情：{level!r}，当平常")
            level = "平常"
        if level != self.mood.level:
            self.mood.since = now
        self.mood.level = level
        self.mood.text = str(mood.get("text") or "").strip()[:TEXT_CHARS]

    def _apply_grudge(self, grudge, cards, friends, now, cfg, dropped) -> None:
        if grudge == "keep":
            return
        if grudge is None:
            self.grudge = None
            return
        if not isinstance(grudge, dict):
            dropped.append("grudge 格式不对")
            return
        who = match_friend(str(grudge.get("who") or ""), friends)
        card = cards.get(who) if who else None
        if who is None:
            dropped.append(f"别扭冲的不是好友：{grudge.get('who')!r}")
        elif card is None or len(card.days) < cfg.grudge_min_days:
            dropped.append(f"和 {who} 还不够熟，不闹别扭")
        else:
            why = str(grudge.get("why") or "").strip()[:TEXT_CHARS]
            if self.grudge is not None and self.grudge.who == who:  # 还是这个人：不续期，只换原因
                self.grudge.why = why or self.grudge.why
            elif self.calm.get(who, 0.0) > now:
                dropped.append(f"刚跟 {who} 消气，先不闹")
            else:
                self.grudge = Grudge(who, why, now, now + cfg.grudge_max)

    def _add_want(self, item, friends, now, cfg, dropped) -> None:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            dropped.append(f"不认识的心愿：{item!r}")
            return
        text = str(item.get("text") or "").strip()[:TEXT_CHARS]
        if not text:
            return
        who = ""
        if item["kind"] == "惦记":
            who = match_friend(str(item.get("who") or ""), friends) or ""
            if not who:
                dropped.append(f"惦记的不是好友：{item.get('who')!r}")
                return
        if any(similar(text, w.text, 0.75) for w in self.wants):
            return
        self.wants.append(Want(item["kind"], text, who, now, now + cfg.want_days * 86400))

    # ---- 身体每圈 ----
    def expire(self, now: float) -> None:
        g = self.grudge
        if g is not None and now >= g.until:
            self.calm[g.who] = g.until + (g.until - g.since)  # 消气后冷却同样长的时间
            self.grudge = None
        self.calm = {k: v for k, v in self.calm.items() if v > now}
        self.wants = [w for w in self.wants if not (w.until and now >= w.until)]

    def forgive(self, name: str, now: float) -> bool:
        """别扭对象难过了 / 认真了：立刻消气，并冷却一阵。返回有没有撤。"""
        g = self.grudge
        if g is None or g.who != name:
            return False
        self.calm[name] = now + (g.until - g.since)
        self.grudge = None
        return True

    def wake(self, now: float, rest_gap: float) -> bool:
        """睡过一觉（离上次反思超过 rest_gap）：心情回到平常（别扭、心愿按各自的时限）。返回有没有回落。"""
        if self.updated is None or now - self.updated < rest_gap or self.mood.level == "平常":
            return False
        self.mood = Mood(since=now)
        return True

    def grudge_on(self, name: str, now: float) -> bool:
        return self.grudge is not None and self.grudge.who == name and now < self.grudge.until

    # ---- 给大脑看的 ----
    def line(self, now: float, energy=None) -> str:
        """status 的“心里”（不带前缀）。"""
        parts = [self.mood.text or self.mood.level]
        if energy is not None:
            parts.append(energy.note)
        g = self.grudge
        if g is not None and now < g.until:
            parts.append(f"跟{g.who}闹别扭（{g.why}，还有 {ago(g.until - now)}消气）" if g.why else f"跟{g.who}闹别扭（还有 {ago(g.until - now)}消气）")
        parts += [f"{_KIND_LABEL[w.kind]}：{w.text}" for w in self.wants]
        return " · ".join(parts)

    def want_note(self, name: str) -> str:
        """好友来时接在 arrive 说明后面。"""
        texts = [w.text for w in self.wants if w.kind == "惦记" and w.who == name]
        return "。你惦记着：" + "；".join(texts) if texts else ""

    def copy(self) -> Mind:
        return copy.deepcopy(self)


_FENCE = re.compile(r"```[a-zA-Z]*")


def parse_reflection(text: str | None) -> dict | None:
    """反思的回答 → dict；容忍 ```json 围栏、前后有字；不是 JSON 对象就 None。
    模型有时把一份回答拆成几个对象回（10-03：心情 / 日记一个、性格一个），一个接一个往后读、按顺序合并；
    碰到坏的就停（不往坏对象里面找，免得把里面嵌套的 {"level": …} 当成整份回答）。"""
    if not text:
        return None
    text = _FENCE.sub("", text)
    dec = json.JSONDecoder()
    data: dict | None = None
    i = text.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            break
        if isinstance(obj, dict):
            data = {**(data or {}), **obj}
        i = text.find("{", end)
    return data
