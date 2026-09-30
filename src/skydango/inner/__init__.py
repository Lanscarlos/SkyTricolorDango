"""内心层（docs/superpowers/specs/2026-09-30-inner-phase1-design.md）：团子跨过每一轮、每一次上线都在的“自我”。

第 1 期只记账：给好友记关系卡（见过几次、上次什么时候、待了多久、说过几句），每次上线记一行（日子）。
不调模型、不往游戏里发输入；大脑在 arrive 事件、status、系统提示词「日子」里看到。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..chat.memory import Turn
from ..config import InnerConfig
from .backfill import backfill
from .ledger import Card, Ledger, Session
from .store import InnerStore

log = logging.getLogger(__name__)


def open_ledger(
    cfg: InnerConfig,
    directory: str | Path,
    friends: Callable[[], list[str]],
    turns: Callable[[], list[Turn]],  # history.jsonl 的全部轮次（回填用，只在要回填时读）
    persist: bool,
    now: float,
) -> Ledger:
    """启动时：补上意外结束的那次 → 没有 people.json 就回填 → 建账本。dry-run（persist = False）什么都不写。

    每一步出错只记日志、用空的继续：账本坏了不能让团子起不来。"""
    store = InnerStore(directory)
    crash: Session | None = None
    try:
        crash = store.take_current() if persist else store.peek_current()
        if crash is not None:
            crash.end = crash.saved or crash.start
            crash.ended = "crash"
            if persist:
                store.append_day(crash)
    except Exception:
        log.exception("内心账本：补上次意外结束的记录出错")

    cards: dict[str, Card] = {}
    backfilled: float | None = None
    filled: list[Session] = []
    try:
        if store.people_exists():
            cards, backfilled, _ = store.load_people()  # 读不了：改名放一边、空卡，不回填（免得错数据再算一遍）
        else:
            cards, filled = backfill(turns(), friends(), cfg.session_gap)
            backfilled = now
            if persist:
                store.write_people(cards, backfilled)
                if not store.days():
                    store.append_days(filled)
            log.info("内心账本：从聊天记录回填了 %d 个好友、%d 次上线", len(cards), len(filled))
    except Exception:
        log.exception("内心账本：读关系卡 / 回填出错，从空卡开始")
        cards, backfilled, filled = {}, None, []

    history: list[Session] = []
    try:
        history = store.days()
        if not persist:  # dry-run 什么都没写：回填出来的、意外结束的只放在内存里
            if not history:
                history = filled
            if crash is not None:
                history = [*history, crash]
    except Exception:
        log.exception("内心账本：读上线记录出错")
    log.info("内心账本：%d 个好友的关系卡、以前上线 %d 次%s", len(cards), len(history), "" if persist else "（dry-run，不写盘）")
    return Ledger(cfg, friends, now, cards, history, store, persist, backfilled)


def show_lines(cfg: InnerConfig, directory: str | Path, friends: list[str], now: float) -> list[str]:
    """memory show：关系卡 + 最近 10 次上线。只读：不建目录、坏文件也不改名。"""
    from .days import recent_days
    from .ledger import card_line

    store = InnerStore(directory)
    cards, _, ok = store.load_people(quarantine=False)
    lines = ["===== 关系卡 ====="]
    if not ok:
        lines.append("（people.json 读不了）")
    else:
        names = [n for n in friends if n in cards] + [n for n in cards if n not in friends]
        for n in names:
            label = n if n in friends else f"{n}（不在好友名单里）"
            lines.append(card_line(label, cards[n], now))
        if not names:
            lines.append("（空）")
    lines.append("===== 最近 10 次上线 =====")
    lines += recent_days(store.days(), 10) or ["（空）"]
    return lines
