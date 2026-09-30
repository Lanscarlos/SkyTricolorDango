"""性格档案（spec 2026-09-30-inner-phase3 §1）：口头禅、老梗、看法。纯数据，不碰文件（读写在 store.py）。

反思（模型）只提建议；这里决定收不收：字数截断、敏感词丢、老梗只挂在够熟的好友名下、收着点的人不记新梗、
每类有上限（先淘汰 hits 最少的，再淘汰最久没用的）、久不用会淡出。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chat.tracker import similar
from ..config import InnerConfig
from .ledger import Card, match_friend

TEXT_CHARS = 30  # 口头禅 / 老梗 / 立场最多几个字
TOPIC_CHARS = 10  # 话题最多几个字
SIMILAR = 0.75
# 这些词出现在条目里就不记（光遇未成年玩家多：外貌、家里、成绩、年龄都不拿来开玩笑）
SENSITIVE = ("胖", "瘦", "丑", "矮", "长相", "身材", "脸", "爸", "妈", "家里", "成绩", "考试", "分数", "几岁", "年纪", "学校", "班")
HEADING = "## 你攒下的性格（慢慢和大家玩出来的；人设里写的优先）"
TAIL = "用得自然，别每句都用；同一个梗一次上线最多用一两回。"


def touchy(*texts: str) -> bool:
    return any(w in (t or "") for t in texts for w in SENSITIVE)


@dataclass
class Trait:
    text: str  # 口头禅 / 老梗的原文；看法的立场（stance）
    who: str = ""  # 老梗挂在哪个好友名下
    topic: str = ""  # 看法的话题
    since: float = 0.0
    last_used: float = 0.0
    hits: int = 0

    def stamp(self) -> float:
        return self.last_used or self.since


def _stats(t: Trait) -> dict:
    return {"since": t.since, "last_used": t.last_used, "hits": t.hits}


def _trait(d: dict, **kw) -> Trait:
    return Trait(since=float(d.get("since") or 0.0), last_used=float(d.get("last_used") or 0.0),
                 hits=int(d.get("hits") or 0), **kw)


@dataclass
class Persona:
    catchphrases: list[Trait] = field(default_factory=list)
    jokes: list[Trait] = field(default_factory=list)
    opinions: list[Trait] = field(default_factory=list)

    # ---- 存盘格式 ----
    def to_dict(self) -> dict:
        return {
            "catchphrases": [{"text": t.text, **_stats(t)} for t in self.catchphrases],
            "jokes": [{"who": t.who, "text": t.text, **_stats(t)} for t in self.jokes],
            "opinions": [{"topic": t.topic, "stance": t.text, **_stats(t)} for t in self.opinions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Persona:
        return cls(
            catchphrases=[_trait(x, text=str(x["text"])) for x in d.get("catchphrases") or []],
            jokes=[_trait(x, text=str(x["text"]), who=str(x.get("who") or "")) for x in d.get("jokes") or []],
            opinions=[_trait(x, text=str(x["stance"]), topic=str(x.get("topic") or "")) for x in d.get("opinions") or []],
        )

    # ---- 套进反思结果 ----
    def apply(self, result: dict, cards: dict[str, Card], friends: list[str], now: float, cfg: InnerConfig,
              soft: set[str] = frozenset()) -> list[str]:
        """只看 persona_add / persona_used；返回丢掉了哪些（记日志用）。"""
        dropped: list[str] = []
        add = result.get("persona_add", {})
        if isinstance(add, dict):
            self._add_list(add, "catchphrases", dropped, lambda item: self._add_catchphrase(item, now, dropped))
            self._add_list(add, "jokes", dropped, lambda item: self._add_joke(item, cards, friends, now, cfg, soft, dropped))
            self._add_list(add, "opinions", dropped, lambda item: self._add_opinion(item, now, dropped))
        else:
            dropped.append("persona_add 不是对象")
        used = result.get("persona_used", [])
        if isinstance(used, list):
            for item in used:
                if isinstance(item, str) and item.strip():
                    self._bump(item.strip(), now)
        else:
            dropped.append("persona_used 不是列表")
        self._cap(cfg, dropped)
        self.fade(now, cfg)
        return dropped

    @staticmethod
    def _add_list(add: dict, key: str, dropped: list[str], fn) -> None:
        items = add.get(key, [])
        if not isinstance(items, list):
            dropped.append(f"{key} 不是列表")
            return
        for item in items:
            fn(item)

    def _add_catchphrase(self, item, now: float, dropped: list[str]) -> None:
        if not isinstance(item, str):
            dropped.append(f"口头禅格式不对：{item!r}")
            return
        text = item.strip()
        if not text:
            return
        if touchy(text):
            dropped.append(f"口头禅带敏感词：{text}")
            return
        text = text[:TEXT_CHARS]
        if any(similar(text, t.text, SIMILAR) for t in self.catchphrases):
            return
        self.catchphrases.append(Trait(text, since=now))

    def _add_joke(self, item, cards, friends, now, cfg, soft, dropped) -> None:
        if not isinstance(item, dict):
            dropped.append(f"老梗格式不对：{item!r}")
            return
        text = str(item.get("text") or "").strip()
        if not text:
            return
        who = match_friend(str(item.get("who") or ""), friends)
        card = cards.get(who) if who else None
        if who is None:
            dropped.append(f"老梗挂的不是好友：{item.get('who')!r}")
        elif card is None or len(card.days) < cfg.grudge_min_days:
            dropped.append(f"和 {who} 还不够熟，不记老梗")
        elif who in soft:
            dropped.append(f"对 {who} 收着点，不记老梗")
        elif touchy(text):
            dropped.append(f"老梗带敏感词：{text}")
        else:
            text = text[:TEXT_CHARS]
            if not any(t.who == who and similar(text, t.text, SIMILAR) for t in self.jokes):
                self.jokes.append(Trait(text, who=who, since=now))

    def _add_opinion(self, item, now: float, dropped: list[str]) -> None:
        if not isinstance(item, dict):
            dropped.append(f"看法格式不对：{item!r}")
            return
        topic = str(item.get("topic") or "").strip()
        stance = str(item.get("stance") or "").strip()
        if not topic or not stance:
            return
        if touchy(topic, stance):
            dropped.append(f"看法带敏感词：{topic}——{stance}")
            return
        topic, stance = topic[:TOPIC_CHARS], stance[:TEXT_CHARS]
        for t in self.opinions:
            if similar(topic, t.topic, SIMILAR):
                if t.text != stance:  # 同一个话题：换成新的立场
                    t.text, t.since = stance, now
                return
        self.opinions.append(Trait(stance, topic=topic, since=now))

    def _bump(self, item: str, now: float) -> None:
        for t in self.catchphrases + self.jokes:
            if similar(item, t.text, SIMILAR):
                t.hits, t.last_used = t.hits + 1, now
        for t in self.opinions:
            if similar(item, t.topic, SIMILAR):
                t.hits, t.last_used = t.hits + 1, now

    def _cap(self, cfg: InnerConfig, dropped: list[str]) -> None:
        _evict(self.catchphrases, cfg.catchphrases_max, dropped)
        for who in dict.fromkeys(t.who for t in self.jokes):
            mine = [t for t in self.jokes if t.who == who]
            gone = _evict(mine, cfg.jokes_per_friend, dropped)
            self.jokes = [t for t in self.jokes if not any(t is g for g in gone)]
        _evict(self.jokes, cfg.jokes_max, dropped)
        _evict(self.opinions, cfg.opinions_max, dropped)

    def fade(self, now: float, cfg: InnerConfig) -> None:
        """久不用的淡出：离上次用（没用过就是记下的时候）超过 fade_days 天。"""
        limit = cfg.fade_days * 86400
        for name in ("catchphrases", "jokes", "opinions"):
            setattr(self, name, [t for t in getattr(self, name) if now - t.stamp() <= limit])

    # ---- 给大脑看的 ----
    def section(self) -> str:
        """系统提示词「你攒下的性格」；空档案给空串。"""
        lines = []
        if self.catchphrases:
            lines.append("口头禅：" + "；".join(t.text for t in self.catchphrases))
        if self.opinions:
            lines.append("看法：" + "；".join(f"{t.topic}——{t.text}" for t in self.opinions))
        for who in dict.fromkeys(t.who for t in self.jokes):
            lines.append(f"和{who}的老梗：" + "；".join(t.text for t in self.jokes if t.who == who))
        if not lines:
            return ""
        return "\n".join([HEADING, *lines, TAIL])

    def joke_note(self, name: str) -> str:
        """好友来时接在 arrive 说明后面（最多 2 条，hits 多的先）。"""
        mine = sorted((t for t in self.jokes if t.who == name), key=lambda t: -t.hits)[:2]
        return "。你们的老梗：" + "；".join(t.text for t in mine) if mine else ""

    def show_lines(self) -> list[str]:
        """memory show 用。"""
        out = ["===== 性格档案 ====="]
        out += [f"口头禅：{t.text}（用过 {t.hits} 次）" for t in self.catchphrases]
        out += [f"看法：{t.topic}——{t.text}（用过 {t.hits} 次）" for t in self.opinions]
        out += [f"和{t.who}的老梗：{t.text}（用过 {t.hits} 次）" for t in self.jokes]
        if len(out) == 1:
            out.append("（空）")
        return out


def _evict(items: list[Trait], cap: int, dropped: list[str]) -> list[Trait]:
    """就地删到 cap 条：hits 最少的先删，一样多删最久没用的（再一样删排在前面的）。返回删掉的。"""
    gone = []
    while len(items) > max(cap, 0):
        victim = min(items, key=lambda t: (t.hits, t.stamp()))
        items.remove(victim)
        gone.append(victim)
        dropped.append(f"太多了，挤掉：{victim.topic + '——' if victim.topic else ''}{victim.text}")
    return gone
