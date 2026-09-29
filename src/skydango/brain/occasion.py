"""场合：现在是什么场合、团子上次主动开口有没有人接、这会儿还能不能主动开口（看场合主动开口，spec §2 / §4）。

纯计算、不碰设备：身体把身边的人、最近的聊天、自己说过的话交进来，拿回一个 Occasion ——
line() 写进 status 给大脑看，blocked 给 say 当护栏。时间都用墙钟（和 body.chat 同一个时钟）。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import ProactiveConfig

LEVEL_NAMES = {"busy": "热闹", "quiet": "安静", "alone": "没熟人"}
ME = "我"  # body.chat 里团子自己的说话人

Chat = Sequence[tuple[float, str, str]]  # (墙钟时间, 说话人, 内容)


@dataclass(frozen=True)
class Spoken:
    t: float  # 墙钟时间（和 body.chat 同一个时钟）
    text: str
    proactive: bool  # 主动开口（不是被聊天叫醒后接话）


@dataclass(frozen=True)
class Occasion:
    level: str  # "busy" 热闹 / "quiet" 安静 / "alone" 没熟人
    friends: tuple[str, ...]
    strangers: int
    others: int  # busy_window 内别人（说话人不是“我”）说了几句
    last: Spoken | None  # 上次主动开口
    last_reply: str  # "called" / "replied" / "waiting" / "none"；没有上次时 ""
    replier: str  # called / replied 时是谁
    recent: int  # quota_window 内主动说了几句
    recent_replied: int  # 其中几句有人接
    left: int  # 还能主动说几句（额度；没熟人 / 冷场时 0）
    blocked: str  # 现在不能主动开口的原因（给大脑看），"" = 可以
    busy_window: float = 180.0
    quota_window: float = 600.0

    def line(self, now: float) -> str:
        parts = [self._level_part()]
        if self.last is None:
            parts.append("还没主动开过口")
        else:
            reply = {
                "called": f"，{self.replier}叫了你",
                "replied": f"，{self.replier}接了话",
                "waiting": "，还在等人接",
                "none": "，没人接",
            }[self.last_reply]
            parts.append(f"上次主动开口 {_ago(now - self.last.t)}前「{self.last.text}」{reply}")
        parts.append(f"最近 {self.quota_window / 60:g} 分钟主动说了 {self.recent} 句，{self.recent_replied} 句有人接")
        parts.append(f"这会儿不能主动开口：{self.blocked}" if self.blocked else f"这会儿还能主动说 {self.left} 句")
        return " / ".join(parts)

    def _level_part(self) -> str:
        if self.level == "alone":
            return f"没熟人（身边只有 {self.strangers} 个陌生人）" if self.strangers else "没熟人（身边没人）"
        who = "、".join(self.friends)
        return f"{LEVEL_NAMES[self.level]}（身边 {who}；最近 {self.busy_window / 60:g} 分钟别人说了 {self.others} 句）"


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds)} 秒" if seconds < 60 else f"{int(seconds // 60)} 分钟"


def is_friend_fn(names: Sequence[str]) -> Callable[[str], bool]:
    """说话人是不是好友：和好友名单模糊匹配（OCR 可能错一两个字）；看不出是谁的永远不是。"""
    names = list(names)

    def is_friend(speaker: str) -> bool:
        return bool(speaker) and speaker != ME and any(similar(speaker, n, 0.75) for n in names)

    return is_friend


def reply_state(
    cfg: ProactiveConfig, spoken: Spoken, chat: Chat, now: float, is_friend: Callable[[str], bool]
) -> tuple[str, str]:
    """主动那句之后 reply_window 内有没有好友说话：(called / replied / waiting / none, 谁)。"""
    end = spoken.t + cfg.reply_window
    lines = [(who, text) for t, who, text in chat if spoken.t < t <= end and is_friend(who)]
    for who, text in lines:
        if any(name in text for name in cfg.self_names):
            return "called", who
    if lines:
        return "replied", lines[0][0]
    return ("waiting", "") if now < end else ("none", "")


def assess(
    cfg: ProactiveConfig,
    now: float,
    friends: Sequence[str],
    strangers: int,
    chat: Chat,
    spoken: Sequence[Spoken],
    is_friend: Callable[[str], bool],
) -> Occasion:
    others = sum(1 for t, who, _ in chat if who != ME and now - cfg.busy_window <= t <= now)
    if not friends:
        level = "alone"
    else:
        level = "busy" if others >= cfg.busy_lines else "quiet"
    mine = [s for s in spoken if s.proactive]
    states = [reply_state(cfg, s, chat, now, is_friend) for s in mine]
    last = mine[-1] if mine else None
    last_reply, replier = states[-1] if states else ("", "")
    in_window = [(s, st) for s, st in zip(mine, states) if now - s.t < cfg.quota_window]
    recent = len(in_window)
    recent_replied = sum(1 for _, (st, _) in in_window if st in ("called", "replied"))

    # 冷场：最后一次有好友说话之后，连着 cold_after 句主动的话都没人接（好友一开口就重新计数）
    heard_at = max((t for t, who, _ in chat if is_friend(who)), default=float("-inf"))
    since = [st for s, (st, _) in zip(mine, states) if s.t > heard_at]
    tail = since[-cfg.cold_after :] if cfg.cold_after > 0 else []
    cold = cfg.cold_after > 0 and len(tail) == cfg.cold_after and all(st == "none" for st in tail)
    limit = cfg.quota_busy if level == "busy" else cfg.quota_quiet
    left = 0 if level == "alone" or cold else max(0, limit - recent)

    blocked = ""
    if level == "alone":
        blocked = "身边没有好友，不主动开口"
    elif cold:
        blocked = f"连着 {cfg.cold_after} 句主动的话都没人接，等有好友说话再主动开口"
    elif recent >= limit:
        # 要等到窗口里只剩 limit - 1 句（档位变低时不止最早那一句）
        wait = math.ceil(in_window[recent - limit][0].t + cfg.quota_window - now) if in_window else 0
        blocked = f"最近 {cfg.quota_window / 60:g} 分钟已经主动说了 {recent} 句，{wait} 秒后才能再主动开口"
    elif last is not None and now - last.t < cfg.min_gap:
        blocked = f"刚主动说过，{math.ceil(last.t + cfg.min_gap - now)} 秒后才能再主动开口"

    return Occasion(
        level=level,
        friends=tuple(friends),
        strangers=strangers,
        others=others,
        last=last,
        last_reply=last_reply,
        replier=replier,
        recent=recent,
        recent_replied=recent_replied,
        left=left,
        blocked=blocked,
        busy_window=cfg.busy_window,
        quota_window=cfg.quota_window,
    )
