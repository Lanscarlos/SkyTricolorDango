"""管理面板「内心」页的数据（spec 2026-09-30-inner-viewer §2 §5）：读 memory/inner/ 的文件，团子在跑时合并实时数据；删性格条目。

都按“内心目录 + 实时取数函数”参数化：之后的大脑沙盒换成 sandbox/memory/inner/ 和 /sandbox/inner 直接复用。
只读的地方不建目录、坏文件不改名（quarantine=False）；读不了的那一块给空的，别的照常。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..inner.api import forget_result, parse_forget
from ..inner.days import recent_days
from ..inner.log import MindLog, read as read_log
from ..inner.store import InnerStore

log = logging.getLogger(__name__)

LOG_SHOW_DAYS = 7  # 页面上看最近几天的流水账
RECENT = 10  # 最近几次上线 / 几篇日记
BUSY_ERROR = "团子正在启动 / 停止，稍等再删"


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        log.exception("内心页读文件出错")
        return default


def _cards(store: InnerStore, friends: list[str]) -> list[dict]:
    cards, _, _ = store.load_people(quarantine=False)
    names = [n for n in friends if n in cards] + [n for n in cards if n not in friends]
    return [{
        "name": n, "friend": n in friends, "first_met": c.first_met, "days": len(c.days), "visits": c.visits,
        "last_seen": c.last_seen, "said": c.lines, "to_me": c.to_me,
    } for n, c in ((n, cards[n]) for n in names)]


def _files_now(store: InnerStore, rows: list[dict], now: float) -> dict:
    """文件里的“现在”：mind.json 的心情 / 别扭 / 心愿，流水账最后一条精力；收着点不存盘，是空的。"""
    out: dict = {"mood": None, "energy": None, "grudge": None, "wants": [], "soft": []}
    if store.mind_path.is_file():
        m = store.load_mind(quarantine=False)
        out["mood"] = {"level": m.mood.level, "text": m.mood.text, "since": m.mood.since}
        g = m.grudge
        if g is not None and now < g.until:
            out["grudge"] = {"who": g.who, "why": g.why, "until": g.until}
        out["wants"] = [{"kind": w.kind, "text": w.text, "who": w.who, "until": w.until}
                        for w in m.wants if not (w.until and now >= w.until)]
    last = next((r for r in reversed(rows) if r.get("kind") == "energy" and r.get("level")), None)
    if last is not None:
        out["energy"] = {"level": last["level"], "score": last.get("score"), "note": last["level"]}
    return out


def _merge_log(files: list[dict], live: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for r in [*files, *live]:
        try:
            seen[(float(r["t"]), r.get("kind"))] = r  # 实时的同一条盖过文件里的
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(seen.values(), key=lambda r: float(r["t"]))


def inner_state(inner_dir: Path, friends: list[str], state: str, live: Callable[[], dict | None], now: float) -> dict:
    """内心页要的全部数据。state 是 runner 的状态；running 时调 live() 取团子的实时快照（取不到 → files_fallback）。"""
    inner_dir = Path(inner_dir)
    store = InnerStore(inner_dir)
    rows = _safe(lambda: read_log(inner_dir / "mind_log.jsonl", now - LOG_SHOW_DAYS * 86400), [])
    files_now = _safe(lambda: _files_now(store, rows, now), {"mood": None, "energy": None, "grudge": None, "wants": [], "soft": []})
    persona = None
    if store.persona_path.is_file():
        persona = _safe(lambda: store.load_persona(quarantine=False).to_dict(), None)
    saved_at = _safe(lambda: store.mind_path.stat().st_mtime if store.mind_path.is_file() else None, None)

    source, now_part, merged = "files", files_now, rows
    if state == "running":
        snap = None
        try:
            snap = live()
        except Exception:
            log.exception("取团子的实时内心出错")
        if isinstance(snap, dict):
            source = "live"
            now_part = {k: snap.get(k) for k in ("mood", "energy", "grudge")}
            now_part.update({k: snap.get(k) or [] for k in ("wants", "soft")})
            persona = snap.get("persona")
            merged = _merge_log(rows, snap.get("log") or [])
        else:
            source = "files_fallback"

    return {
        "source": source,
        "saved_at": saved_at,
        "now": now_part,
        "persona": persona,
        "log": merged,
        "cards": _safe(lambda: _cards(store, friends), []),
        "days": _safe(lambda: recent_days(store.days(), RECENT), []),
        "diaries": _safe(lambda: list(reversed(store.last_diaries(RECENT))), []),  # 新的在前
    }


def forget_offline(inner_dir: Path, req: dict, now: float) -> dict:
    """团子没在跑：直接改 persona.json（原子写）+ 流水账记一条 forget。请求不对抛 ValueError。"""
    kind, text, who, topic = parse_forget(req)
    store = InnerStore(inner_dir)
    if not store.persona_path.is_file():
        return forget_result("找不到这条（可能已经淡出了）")
    persona = store.load_persona(quarantine=False)  # 读不了就是空的：删不到，也不会把坏文件覆盖掉
    if not persona.remove(kind, text, who, topic):
        return forget_result("找不到这条（可能已经淡出了）")
    store.write_persona(persona)
    MindLog(store.dir / "mind_log.jsonl", persist=True).forget(now, kind, text, who, topic)
    log.info("管理面板删了性格条目：%s %s%s", kind, topic or who, text)
    return forget_result("")
