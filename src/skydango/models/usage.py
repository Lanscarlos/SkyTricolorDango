"""模型用量和额度（spec 2026-10-06-model-usage §2 §3 §5 §6.1）：每笔调用按 (用处, 供应商/模型, 主还是备) 记次数、token、钱；
Claude Code 的限额事件、DeepSeek 余额也放在这里，snapshot() 给团子 / 沙盒的 /usage。

一次进程一个，挂在 Registry.meter 上；线程安全。记账出错只记日志，绝不影响调用本身（调用方包 try）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .config import USE_BY_NAME, USES, ModelSetup, Price
from .gate import ProviderGates

if TYPE_CHECKING:
    from .usage_ledger import Ledger

log = logging.getLogger(__name__)

BEIJING = timezone(timedelta(hours=8))  # 固定 UTC+8：Windows 上 zoneinfo 要另装 tzdata
NUMS = ("calls", "fails", "input", "output", "cache_read", "cache_write")
_USAGE_KEYS = {"input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_input_tokens",
               "cache_write": "cache_creation_input_tokens"}


def _minutes(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def is_peak(at: float, peak_hours: tuple[str, ...]) -> bool:
    """北京时间周一到周五、落在某个 "HH:MM-HH:MM"（左闭右开）里。不认法定节假日。"""
    if not peak_hours:
        return False
    t = datetime.fromtimestamp(at, BEIJING)
    if t.weekday() >= 5:
        return False
    now = t.hour * 60 + t.minute
    for span in peak_hours:
        start, _, end = span.partition("-")
        try:
            if _minutes(start) <= now < _minutes(end):
                return True
        except ValueError:
            continue
    return False


def _int(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def tokens(usage: dict | None) -> dict[str, int]:
    """后端返回的 usage（缺的、乱填的当 0）→ input / output / cache_read / cache_write。"""
    usage = usage if isinstance(usage, dict) else {}
    return {k: _int(usage.get(src)) for k, src in _USAGE_KEYS.items()}


def call_cost(price: Price | None, usage: dict | None, peak: float) -> float | None:
    """一笔的钱（元）：OpenAI 兼容的 prompt_tokens 含命中缓存的部分，没命中 = 输入 − 命中（不小于 0）。"""
    if price is None:
        return None
    t = tokens(usage)
    miss = max(0, t["input"] - t["cache_read"])
    return (t["cache_read"] * price.hit + miss * price.miss + t["output"] * price.out) / 1e6 * peak


def _empty() -> dict:
    return {**{k: 0 for k in NUMS}, "cost": None, "est": False}


def add_row(into: dict, row: dict) -> None:
    """同一行相加：数字相加；cost 两边都是 None 才是 None。"""
    for k in NUMS:
        into[k] = into.get(k, 0) + _int(row.get(k))
    cost = row.get("cost")
    if cost is not None:
        into["cost"] = (into.get("cost") or 0.0) + float(cost)
    else:
        into.setdefault("cost", None)
    into["est"] = bool(into.get("est")) or bool(row.get("est"))


def merge_rows(a: dict, b: dict) -> dict:
    """{key: 行} 两份相加（不改原来的）。"""
    out = {k: dict(v) for k, v in a.items()}
    for key, row in b.items():
        add_row(out.setdefault(key, _empty()), row)
    return out


def row_key(use: str, provider: str, model: str, backup: bool) -> str:
    return f"{use}|{provider}/{model}|{'backup' if backup else 'main'}"


def rows_view(rows: dict) -> dict:
    """{key: 行} → snapshot 里的 {"rows": [...], "cost", "est"}：钱多的在前、再按次数。"""
    out = []
    for key, row in rows.items():
        use, _, rest = key.partition("|")
        model, _, which = rest.rpartition("|")
        spec = USE_BY_NAME.get(use)
        out.append({"use": use, "label": spec.label if spec else use, "model": model, "backup": which == "backup",
                    **{k: _int(row.get(k)) for k in NUMS}, "cost": row.get("cost"), "est": bool(row.get("est"))})
    out.sort(key=lambda r: (-(r["cost"] or 0.0), -r["calls"], r["use"], r["model"]))
    priced = [r["cost"] for r in out if r["cost"] is not None]
    return {"rows": out, "cost": sum(priced) if priced else None, "est": any(r["est"] for r in out)}


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_rate(info: dict) -> dict:
    """Claude Code 的 rate_limit_info → 面板用的样子；字段都可选，乱填的丢掉。"""
    out: dict = {}
    if isinstance(info.get("status"), str):
        out["status"] = info["status"]
    if isinstance(info.get("rateLimitType"), str):
        out["type"] = info["rateLimitType"]
    if (u := _num(info.get("utilization"))) is not None:
        out["utilization"] = u
    if (r := _num(info.get("resetsAt"))) is not None:
        out["resets_at"] = int(r)
    windows = {}
    for name, w in (info.get("unifiedWindows") or {}).items() if isinstance(info.get("unifiedWindows"), dict) else ():
        if not isinstance(w, dict):
            continue
        u, r = _num(w.get("utilization")), _num(w.get("resetsAt"))
        if u is None:
            continue
        windows[str(name)] = {"utilization": u, **({"resets_at": int(r)} if r is not None else {})}
    if windows:
        out["windows"] = windows
    return out


class UsageMeter:
    def __init__(self, setup: ModelSetup, gates: ProviderGates, *, source: str, wall: Callable[[], float] = time.time,
                 ledger: Ledger | None = None) -> None:
        self.setup = setup
        self.gates = gates
        self.source = source
        self.wall = wall
        self.ledger = ledger
        self.started = wall()
        self._lock = threading.Lock()
        self._run: dict[str, dict] = {}
        self._delta: dict[str, dict[str, dict[str, dict]]] = {}  # 日期 → 来源 → key → 行（还没写进账本的）
        self._rates: dict[str, dict] = {}
        self._balances: dict[str, dict] = {}
        self._base: tuple[str, dict] = ("", {})  # (日期, 账本里那天已经写盘的 {key: 行})：snapshot 的「今天」= 它 + 增量
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._saver: threading.Thread | None = None
        self._refresh_base(time.strftime("%Y-%m-%d", time.localtime(self.started)))

    # ---- 记 ----
    def record(self, use: str, provider: str, model: str, *, backup: bool, usage: dict | None, ok: bool,
               at: float | None = None) -> None:
        at = self.wall() if at is None else at
        p = self.setup.providers.get(provider)
        price = p.price(model) if p is not None else None
        row = _empty()
        row["calls"] = 1
        if ok:
            row.update(tokens(usage))
            if price is not None:
                row["cost"] = call_cost(price, usage, p.peak if is_peak(at, p.peak_hours) else 1.0)
                row["est"] = price.est
        else:
            row["fails"] = 1
            if price is not None:
                row["cost"], row["est"] = 0.0, price.est
        key = row_key(use, provider, model, backup)
        day = time.strftime("%Y-%m-%d", time.localtime(at))
        with self._lock:
            add_row(self._run.setdefault(key, _empty()), row)
            add_row(self._delta.setdefault(day, {}).setdefault(self.source, {}).setdefault(key, _empty()), row)

    def rate_limit(self, provider: str, info: dict, at: float | None = None) -> None:
        if not isinstance(info, dict):
            return
        rate = parse_rate(info)
        rate["at"] = self.wall() if at is None else at
        with self._lock:
            self._rates[provider] = rate

    def balance(self, provider: str, info: dict | None, error: str | None, at: float | None = None) -> None:
        at = self.wall() if at is None else at
        entry = {"available": (info or {}).get("available"), "items": list((info or {}).get("items") or []),
                 "at": at, "error": error}
        with self._lock:
            if error is not None and provider in self._balances and self._balances[provider].get("error") is None:
                entry = {**self._balances[provider], "error": error}  # 查失败：上次的数留着，标上原因
                entry["at"] = self._balances[provider]["at"]
            self._balances[provider] = entry

    # ---- 增量（账本用）----
    def take_delta(self) -> dict:
        with self._lock:
            out, self._delta = self._delta, {}
        return out

    def restore_delta(self, delta: dict) -> None:
        with self._lock:
            for day, sources in delta.items():
                for source, rows in sources.items():
                    into = self._delta.setdefault(day, {}).setdefault(source, {})
                    for key, row in rows.items():
                        add_row(into.setdefault(key, _empty()), row)

    def _today_rows(self, day: str) -> dict:
        """今天 = 账本里已经写盘的（所有来源、之前几次运行）+ 这个进程还没写盘的（调用方拿着 _lock）。"""
        base_day, base = self._base
        rows: dict = dict(base) if base_day == day else {}
        for source_rows in self._delta.get(day, {}).values():
            rows = merge_rows(rows, source_rows)
        return rows

    def _refresh_base(self, day: str) -> None:
        if self.ledger is None:
            return
        try:
            rows = self.ledger.day_rows(day)
        except Exception:  # noqa: BLE001 账本读不出来不影响记账
            log.debug("读用量账本出错", exc_info=True)
            return
        with self._lock:
            self._base = (day, rows)

    def flush(self) -> bool:
        """把增量加进账本；失败增量放回去、返回 False。没有账本返回 False。"""
        if self.ledger is None:
            return False
        with self._flush_lock:
            delta = self.take_delta()
            today = time.strftime("%Y-%m-%d", time.localtime(self.wall()))
            ok = False
            if delta:
                try:
                    ok = self.ledger.add(delta, today)
                except Exception:  # noqa: BLE001
                    log.warning("写用量账本出错", exc_info=True)
                if not ok:
                    self.restore_delta(delta)
                    return False
            self._refresh_base(today)
            return True

    def start_saver(self, every: float) -> None:
        """每 every 秒写一次账本（守护线程）。"""
        if self.ledger is None or self._saver is not None:
            return

        def loop() -> None:
            while not self._stop.wait(every):
                self.flush()

        self._saver = threading.Thread(target=loop, name="usage-saver", daemon=True)
        self._saver.start()

    def close(self) -> None:
        """停保存线程、最后写一次。"""
        self._stop.set()
        if self._saver is not None:
            self._saver.join(timeout=5)
        self.flush()

    # ---- 给 /usage ----
    def current(self, use: str) -> tuple[str | None, bool]:
        """这个用处现在实际走哪个（同 Registry.current）、是不是备用。"""
        u = self.setup.uses[use]
        if u.disabled:
            return None, False
        for ref, backup in ((u.main, False), (u.backup, True)):
            if ref is not None and self.gates.ok(ref.provider):
                return str(ref), backup
        return None, False

    def snapshot(self) -> dict:
        now = self.wall()
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        with self._lock:
            run = {k: dict(v) for k, v in self._run.items()}
            today = self._today_rows(day)
            rates = {k: dict(v) for k, v in self._rates.items()}
            balances = {k: dict(v) for k, v in self._balances.items()}
        providers = [{"id": p.id, "kind": p.kind, "closed": self.gates.reason(p.id),
                      "balance": balances.get(p.id), "rate": rates.get(p.id)}
                     for p in self.setup.providers.values() if p.kind != "echo"]
        uses = []
        for spec in USES:
            u = self.setup.uses[spec.name]
            current, backup = self.current(spec.name)
            uses.append({"use": spec.name, "label": spec.label, "main": str(u.main) if u.main else None,
                         "current": current, "backup": backup, "disabled": u.disabled or None})
        return {"source": self.source, "started": self.started, "run": rows_view(run), "today": rows_view(today),
                "providers": providers, "uses": uses}
