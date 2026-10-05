"""按供应商的闸：那家额度 / 余额用完或认证失败后关上，这次运行里不再用它（原 Claude 总闸推广）。线程安全，只关不开。"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class ProviderGates:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed: dict[str, str] = {}

    def ok(self, provider: str) -> bool:
        with self._lock:
            return provider not in self._closed

    def reason(self, provider: str) -> str | None:
        with self._lock:
            return self._closed.get(provider)

    def trip(self, provider: str, kind: str, detail: str) -> bool:
        """关这家的闸；返回这次是不是刚关上（已经关了返回 False，原因不改）。"""
        why = "额度 / 余额用完" if kind == "limit" else f"认证失败：{detail[:80]}"
        with self._lock:
            if provider in self._closed:
                return False
            self._closed[provider] = why
        log.warning("%s 不能用了（%s），这次运行里改用备用", provider, why)
        return True

    def closed(self) -> dict[str, str]:
        with self._lock:
            return dict(self._closed)
