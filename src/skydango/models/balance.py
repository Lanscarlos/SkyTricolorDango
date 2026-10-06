"""DeepSeek 余额（spec 2026-10-06-model-usage §5.1）：GET <base_url>/user/balance，每 balance_every 秒查一次交给 UsageMeter。

只查 DeepSeek 官方接口（is_deepseek）：别的 OpenAI 兼容家没有统一的余额接口。查失败只记 DEBUG，面板上显示原因。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from ..chat import llm as chat_llm
from .config import ModelSetup, ProviderConfig, is_deepseek

log = logging.getLogger(__name__)


class BalanceError(Exception):
    """查不到余额：原因给面板。"""


def _money(value) -> float:
    if isinstance(value, bool):
        raise ValueError(value)
    return float(value)


def fetch_balance(base_url: str, key: str, timeout: float = 10.0, opener=None) -> dict:
    """{"available": bool, "items": [{"currency", "total", "granted", "topped_up"}]}；查不到抛 BalanceError。"""
    url = base_url.rstrip("/") + "/user/balance"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    open_ = opener or urllib.request.urlopen
    try:
        with open_(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise BalanceError(f"{exc.code} {exc.reason}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BalanceError(f"连不上：{exc}") from None
    try:
        data = json.loads(raw)
        infos = data["balance_infos"]
        items = [{"currency": str(i.get("currency") or ""), "total": _money(i["total_balance"]),
                  "granted": _money(i.get("granted_balance", 0)), "topped_up": _money(i.get("topped_up_balance", 0))}
                 for i in infos]
        return {"available": bool(data.get("is_available", True)), "items": items}
    except (ValueError, KeyError, TypeError, AttributeError):
        raise BalanceError("看不懂余额接口的回答") from None


def provider_key(p: ProviderConfig, environ: Mapping[str, str]) -> str:
    """这家的 Key（同 build_client：环境变量，Windows 上再看 setx 写的用户变量）。"""
    if not p.key_env:
        return ""
    return environ.get(p.key_env, "") or chat_llm._user_env(p.key_env)


def balance_targets(setup: ModelSetup, environ: Mapping[str, str]) -> list[tuple[ProviderConfig, str]]:
    """要查余额的供应商：DeepSeek 官方接口、有 Key 的。"""
    out = []
    for p in setup.providers.values():
        if is_deepseek(p):
            key = provider_key(p, environ)
            if key:
                out.append((p, key))
    return out


class BalanceWatcher:
    """后台线程：启动时先查一次，之后每 every 秒查一次。"""

    def __init__(self, setup: ModelSetup, meter, environ: Mapping[str, str] = os.environ, every: float = 300.0,
                 fetch: Callable[..., dict] = fetch_balance) -> None:
        self.setup = setup
        self.meter = meter
        self.environ = environ
        self.every = every
        self.fetch = fetch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        for p, key in balance_targets(self.setup, self.environ):
            try:
                info = self.fetch(p.base_url, key)
            except BalanceError as exc:
                log.debug("%s 余额查不到：%s", p.id, exc)
                self._give(p.id, None, str(exc))
            except Exception as exc:  # noqa: BLE001 查余额出什么错都不影响团子
                log.debug("%s 余额查不到", p.id, exc_info=True)
                self._give(p.id, None, f"{type(exc).__name__}: {exc}")
            else:
                self._give(p.id, info, None)

    def _give(self, provider: str, info: dict | None, error: str | None) -> None:
        try:
            self.meter.balance(provider, info, error)
        except Exception:  # noqa: BLE001
            log.debug("记余额出错", exc_info=True)

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            self.poll_once()
            while not self._stop.wait(self.every):
                self.poll_once()

        self._thread = threading.Thread(target=loop, name="balance", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
