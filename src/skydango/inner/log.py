"""内心流水账（spec 2026-09-30-inner-viewer §1）：memory/inner/mind_log.jsonl，一行一条。

kind 只有三种：reflect（每次反思改了什么）、energy（每 ENERGY_EVERY 秒一条精力）、forget（网页上删了性格条目）。
live 追加写盘，dry-run 只在内存里；内存最多留 MEMORY_MAX 条。写不了只记日志，不影响团子。
`diff` 是纯函数：对比反思前后的 Mind / Persona 快照，写成给人看的一行行文字（不靠模型）。
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Callable

from .mind import Mind
from .persona import Persona

log = logging.getLogger(__name__)

ENERGY_EVERY = 300  # 秒（墙上时间）：多久记一条精力
LOG_DAYS = 30  # 启动时删掉多少天前的行
MEMORY_MAX = 200  # 内存里最多留几条


def diff(before_mind: Mind, after_mind: Mind, before_persona: Persona | None, after_persona: Persona | None) -> list[str]:
    """反思前后变了什么。顺序固定：心情 → 别扭 → 新心愿 → 心愿了结 → 新口头禅 → 新老梗 → 新看法 → 看法换了 → 淡出。"""
    out: list[str] = []
    a, b = before_mind.mood, after_mind.mood
    if a.level != b.level:
        out.append(f"心情 {a.level}→{b.level}" + (f"（{b.text}）" if b.text else ""))
    elif a.text != b.text and b.text:
        out.append(f"心情：{b.text}")

    ga, gb = before_mind.grudge, after_mind.grudge
    if gb is not None and (ga is None or ga.who != gb.who):
        if ga is not None:
            out.append(f"别扭撤了：{ga.who}")
        out.append(f"新别扭：{gb.who}" + (f"（{gb.why}）" if gb.why else ""))
    elif ga is not None and gb is None:
        out.append(f"别扭撤了：{ga.who}")

    wa = [w.text for w in before_mind.wants]
    wb = [w.text for w in after_mind.wants]
    out += [f"新心愿：{t}" for t in wb if t not in wa]
    out += [f"心愿了结：{t}" for t in wa if t not in wb]

    if before_persona is not None or after_persona is not None:
        pa, pb = before_persona or Persona(), after_persona or Persona()
        ca = [t.text for t in pa.catchphrases]
        cb = [t.text for t in pb.catchphrases]
        ja = [(t.who, t.text) for t in pa.jokes]
        jb = [(t.who, t.text) for t in pb.jokes]
        oa = {t.topic: t.text for t in pa.opinions}
        ob = {t.topic: t.text for t in pb.opinions}
        out += [f"新口头禅：{t}" for t in cb if t not in ca]
        out += [f"新老梗：{w}——{t}" for w, t in jb if (w, t) not in ja]
        out += [f"新看法：{k}——{v}" for k, v in ob.items() if k not in oa]
        out += [f"看法换了：{k} {oa[k]}→{v}" for k, v in ob.items() if k in oa and oa[k] != v]
        out += [f"淡出：{t}" for t in ca if t not in cb]
        out += [f"淡出：{w}——{t}" for w, t in ja if (w, t) not in jb]
        out += [f"淡出：{k}——{v}" for k, v in oa.items() if k not in ob]
    return out


def _energy_dict(energy) -> dict:
    if energy is None:
        return {"level": None, "score": None}
    return {"level": energy.level, "score": energy.score}


class MindLog:
    """path 为 None 或 persist = False 时只在内存里记。"""

    def __init__(self, path: Path | str | None, persist: bool) -> None:
        self.path = Path(path) if path is not None else None
        self.persist = persist and self.path is not None
        self._rows: deque[dict] = deque(maxlen=MEMORY_MAX)
        self.on_add: Callable[[dict], None] | None = None  # 每记一条就调（沙盒把反思改了什么写进聊天记录）；出错只记日志
        self._tail_ok = False  # 第一次追加前看一眼文件末尾：上次写到一半的半行后面先换行，别把新的一行也粘坏

    # ---- 记 ----
    def reflect(self, t: float, final: bool, mind: Mind, energy, changes: list[str], dropped: list[str]) -> dict:
        g = mind.grudge
        return self._add({
            "t": t, "kind": "reflect", "final": bool(final),
            "mood": {"level": mind.mood.level, "text": mind.mood.text},
            "energy": None if energy is None else _energy_dict(energy),
            "grudge": {"who": g.who, "why": g.why, "until": g.until} if g is not None else None,
            "wants": [w.text for w in mind.wants],
            "changes": list(changes), "dropped": list(dropped),
        })

    def energy(self, t: float, energy) -> dict:
        return self._add({"t": t, "kind": "energy", **_energy_dict(energy)})

    def forget(self, t: float, kind: str, text: str, who: str = "", topic: str = "") -> dict:
        """kind 是性格条目的类别（catchphrase / joke / opinion），记在 what 里（kind 字段留给记录类别 forget）。"""
        return self._add({"t": t, "kind": "forget", "what": kind, "text": text, "who": who, "topic": topic})

    def recent(self) -> list[dict]:
        """内存里的（旧到新）。"""
        return list(self._rows)

    # ---- 保留 ----
    def trim(self, now: float, days: int = LOG_DAYS) -> None:
        """删掉 days 天前的行（live 时原子重写文件：写 .tmp 再 replace）。"""
        cutoff = now - days * 86400
        kept = [r for r in self._rows if r.get("t", 0) >= cutoff]
        self._rows.clear()
        self._rows.extend(kept)
        if not self.persist or not self.path.is_file():
            return
        try:
            rows = read(self.path, cutoff)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("内心流水账整理不了：%s", exc)

    # ---- 内部 ----
    def _add(self, row: dict) -> dict:
        self._rows.append(row)
        if self.persist:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                lead = "" if self._tail_ok else _broken_tail(self.path)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(lead + json.dumps(row, ensure_ascii=False) + "\n")
                self._tail_ok = True
            except OSError as exc:
                log.warning("内心流水账写不了：%s", exc)
        if self.on_add is not None:
            try:
                self.on_add(dict(row))
            except Exception:
                log.exception("内心流水账的 on_add 出错")
        return row


def _broken_tail(path: Path) -> str:
    """文件末尾不是换行（被强杀写到一半）→ 返回 "\\n"，让新的一行另起。"""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            if fh.tell() == 0:
                return ""
            fh.seek(-1, 2)
            return "" if fh.read(1) == b"\n" else "\n"
    except FileNotFoundError:
        return ""


def read(path: Path | str, since: float) -> list[dict]:
    """读文件里 t >= since 的记录（旧到新）；文件不在返回 []，读不了的行跳过（上次被强杀写到一半）。"""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("内心流水账读不了：%s", exc)
        return []
    out = []
    for chunk in raw.splitlines():
        if not chunk.strip():
            continue
        try:
            row = json.loads(chunk.decode("utf-8"))  # 按行解码：截在中文字符中间的那半行单独跳过（UnicodeDecodeError 是 ValueError）
            t = float(row["t"])
        except (ValueError, TypeError, KeyError):
            continue
        if isinstance(row, dict) and t >= since:
            out.append(row)
    return out
