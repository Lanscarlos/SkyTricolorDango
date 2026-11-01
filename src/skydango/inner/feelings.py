"""对每个人的感觉（spec 2026-11-01-friend-feelings）：关系卡上一句印象 + 亲近度 + 走向。

下线那次的反思（模型）只提建议；这里决定收不收：只改这次见过的好友、亲近度一次最多一档且要够天数、
太长整条不记（不截半句）、敏感词 / 依恋吃醋 / 难过类的话丢、说过难过的人不写生疏和埋怨。
另有拼给大脑（arrive、status）和反思材料的文字。纯函数，不写盘。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..chat.responder import claims_human
from ..chat.tracker import similar
from ..config import InnerConfig
from .ledger import Card, day_of, match_friend
from .mind import sounds_upset
from .persona import SENSITIVE

CLOSE_LEVELS = ("刚认识", "熟了", "很铁")  # 有序
TRENDS = ("在变近", "平稳", "有点生疏")  # 写盘的只有这三种
STALE = "好久没见了"  # 显示时现算的一档，不写盘
FEEL_MAX = 40  # 印象最多几个字，超了整条不收
FEEL_PAST = 3  # 留最近几版旧印象
SAME = 0.9  # 新旧印象像到这个算没改
# 不写依恋、吃醋（好友里可能有未成年人，感情不往这边推）
CLINGY = ("离不开", "只喜欢", "只要他", "只要她", "吃醋", "嫉妒", "独占", "最爱")
# 说过难过的人：印象里不许带埋怨
GRUMBLE = ("烦", "讨厌", "放鸽子", "迟到", "没良心", "气死", "不理我")


def _bad_text(text: str) -> str:
    """印象文字不能收的原因；能收返回空串。"""
    if len(text) > FEEL_MAX:
        return f"超过 {FEEL_MAX} 字"
    if any(w in text for w in SENSITIVE):
        return "带敏感词"
    if any(w in text for w in CLINGY):
        return "写了依恋 / 吃醋"
    if sounds_upset(text):
        return "带难过的话"
    if claims_human(text):
        return "说自己是真人"
    return ""


def _close_cap(card: Card, cfg: InnerConfig) -> int:
    """按一起玩过几天，亲近度最高能到哪档（下标）。"""
    days = len(card.days)
    if days >= cfg.feel_close_days:
        return 2
    return 1 if days >= cfg.grudge_min_days else 0


def apply_feelings(
    cards: dict[str, Card], proposed, *, met: set[str], soft: set[str], today: str, friends: Sequence[str], cfg: InnerConfig,
) -> tuple[list[str], list[str]]:
    """把反思给的 feelings 套进关系卡（原地改 card.feeling）。返回（改了什么，丢掉的原因）。"""
    changes: list[str] = []
    dropped: list[str] = []
    if not isinstance(proposed, list):
        return changes, dropped
    done: set[str] = set()
    for item in proposed:
        if not isinstance(item, dict) or not isinstance(item.get("who"), str):
            continue
        name = match_friend(item["who"], friends)
        if name is None or name not in cards:
            dropped.append(f"感觉：{item['who']} 不在好友名单里")
            continue
        if name in done:
            continue
        done.add(name)
        if name not in met:
            dropped.append(f"感觉：{name} 这次没见到，不改")
            continue
        text = item.get("text").strip() if isinstance(item.get("text"), str) else ""
        close, trend = item.get("close"), item.get("trend")
        if close not in CLOSE_LEVELS or trend not in TRENDS:
            dropped.append(f"感觉：{name} 的档位看不懂（{close} / {trend}）")
            continue
        card = cards[name]
        old = card.feeling or {}
        old_text, old_close = old.get("text", ""), old.get("close")
        want = CLOSE_LEVELS.index(close)
        if old_close in CLOSE_LEVELS:  # 一次最多变一档
            base = CLOSE_LEVELS.index(old_close)
            want = max(base - 1, min(base + 1, want))
        cap = _close_cap(card, cfg)
        if want > cap:
            dropped.append(f"感觉：{name} 一起玩过 {len(card.days)} 天，{CLOSE_LEVELS[want]} 压回 {CLOSE_LEVELS[cap]}")
            want = cap
        close = CLOSE_LEVELS[want]
        if name in soft:  # 这次说过难过：不写生疏、不收埋怨
            if trend == "有点生疏":
                trend = "平稳"
            if any(w in text for w in GRUMBLE):
                dropped.append(f"感觉：{name} 这次说过难过，带埋怨的印象不收")
                text = ""
        if text:
            why = _bad_text(text)
            if why:
                dropped.append(f"感觉：{name} 的印象{why}，不收")
                text = ""
        if not text or (old_text and similar(text, old_text, SAME)):
            text = old_text
        if not text:  # 还没有印象、这次也没收下：不建空的
            continue
        if text == old_text and close == old_close and trend == old.get("trend"):
            continue
        past = list(old.get("past", []))
        if old_text and (text != old_text or close != old_close):
            past = (past + [{"text": old_text, "close": old_close, "date": old.get("since", "")}])[-FEEL_PAST:]
        card.feeling = {"text": text, "close": close, "trend": trend, "since": today, "past": past}
        changes.append(f"对{name}的感觉：" + (f"{old_close}→{close}，" if old_close and old_close != close else f"{close}，")
                       + f"{trend}（{text}）")
    return changes, dropped


def distressed(chat: Iterable[tuple[float, str, str]], friends: Sequence[str]) -> set[str]:
    """聊天里说过难过类的话的好友（不靠 [inner] persona 的收着点）。"""
    out: set[str] = set()
    for _, who, text in chat:
        name = match_friend(who, friends)
        if name and sounds_upset(text):
            out.add(name)
    return out


def feelings_lines(cards: dict[str, Card], names: Sequence[str], now: float, cfg: InnerConfig) -> list[str]:
    """下线反思材料「你对他们的感觉」：这次见过的好友每人一行（没有卡的不写）。"""
    out = []
    for name in names:
        card = cards.get(name)
        if card is None:
            continue
        f, days = card.feeling, f"一起玩过 {len(card.days)} 天"
        if f.get("text"):
            out.append(f"{name}：{f['text']}（{f['close']}，{f['trend']}，{f.get('since', '')} 写的；{days}）")
        else:
            out.append(f"{name}：还没有印象（{days}）")
    return out


def introspect_text(cards: dict[str, Card], friends: Sequence[str], now: float, cfg: InnerConfig) -> str:
    """introspect("关系")：每个有印象的好友一段——现在的印象、档位、走向、哪天写的，下面是以前的（旧到新）。"""
    out = []
    for name in friends:
        card = cards.get(name)
        if card is None or not card.feeling.get("text"):
            continue
        f = card.feeling
        out.append(f"{name}：{f['text']}（{f['close']}，{trend_shown(card, now, cfg)}，{f.get('since', '')} 写的）")
        out += [f"  以前（{p.get('date', '')}）：{p.get('text', '')}（{p.get('close', '')}）" for p in f.get("past", [])]
    return "\n".join(out) if out else "还没对谁有印象"


def trend_shown(card: Card, now: float, cfg: InnerConfig) -> str:
    """显示用的走向：太久没在身边见过就是「好久没见了」。"""
    if card.last_seen is not None and now - card.last_seen >= cfg.feel_stale_days * 86400:
        return STALE
    return card.feeling.get("trend", "")


def feeling_note(card: Card, now: float, cfg: InnerConfig, soft: bool) -> str:
    """arrive 事件里的一句：「。你对他：嘴贫但靠谱（熟了，最近平稳）」；收着点时只写亲近度。"""
    f = card.feeling
    if not f.get("text"):
        return ""
    if soft:
        return f"。你对他：（{f['close']}）"
    return f"。你对他：{f['text']}（{f['close']}，最近{trend_shown(card, now, cfg)}）"


def feeling_bit(card: Card, now: float, cfg: InnerConfig) -> str:
    """status「身边的好友」括号里的一段：档位，生疏 / 好久没见时带上走向。"""
    f = card.feeling
    if not f.get("text"):
        return ""
    trend = trend_shown(card, now, cfg)
    return f"{f['close']}·{trend}" if trend in ("有点生疏", STALE) else f["close"]
