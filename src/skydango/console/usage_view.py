"""面板自己的「模型用量」（spec 2026-10-06-model-usage §5.1 §6.1）：团子 / 沙盒没在跑时读账本的今天，自己查 DeepSeek 余额。

结构和子进程的 /usage 一样（run / rate 为 null、闸都开着），页面同一套代码画。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from ..models.balance import BalanceError, balance_targets, fetch_balance
from ..models.config import resolve
from ..models.gate import ProviderGates
from ..models.usage import UsageMeter
from ..models.usage_ledger import ledger_for
from .preflight import SecretEnv
from .settings import SettingsStore

log = logging.getLogger(__name__)


class UsageView:
    def __init__(self, store: SettingsStore, wall: Callable[[], float] = time.time,
                 fetch: Callable[..., dict] = fetch_balance) -> None:
        self.store = store
        self.wall = wall
        self.fetch = fetch
        self._lock = threading.Lock()
        self._balances: dict[str, tuple[float, dict | None, str | None]] = {}  # id → (查的时间, 结果, 错误)

    def _balance(self, setup, every: float) -> dict[str, tuple[float, dict | None, str | None]]:
        now = self.wall()
        for p, key in balance_targets(setup, SecretEnv(self.store)):
            with self._lock:
                cached = self._balances.get(p.id)
            if cached is not None and now - cached[0] < every:
                continue
            try:
                entry = (now, self.fetch(p.base_url, key), None)
            except BalanceError as exc:
                entry = (now, None, str(exc))
            except Exception as exc:  # noqa: BLE001 查不到余额不影响页面
                log.debug("面板查余额出错", exc_info=True)
                entry = (now, None, f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._balances[p.id] = entry
        with self._lock:
            return dict(self._balances)

    def get(self) -> dict:
        cfg = self.store._fallback()  # console.toml 坏了也能看
        setup = resolve(cfg)
        meter = UsageMeter(setup, ProviderGates(), source="console", wall=self.wall, ledger=ledger_for(cfg))
        for pid, (at, info, error) in self._balance(setup, cfg.usage.balance_every).items():
            if pid in setup.providers:
                meter.balance(pid, info, error, at=at)
        snap = meter.snapshot()
        return {"ok": True, **snap, "source": None, "started": None, "run": None}
