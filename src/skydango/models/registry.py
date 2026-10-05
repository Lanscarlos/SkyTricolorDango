"""按用处给模型：Registry 管所有供应商和闸，call(用处) 给出 GatedCall（主 → 备自动切换）。

代码只从这里拿模型，别处不再自己拼 claude -p 命令或建 OpenAI 客户端。见 spec 2026-10-05-model-providers §2。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from ..chat import llm as chat_llm
from .config import ModelRef, ModelSetup, Problem, ProviderConfig, UseConfig
from .errors import ModelError, ModelUnavailable, down_kind
from .gate import ProviderGates

log = logging.getLogger(__name__)


class EchoBackend:
    """不调模型，原样回显（--echo 联调用）。"""

    def message(self, system: str, content, *, history=None, max_tokens=None, timeout=None) -> dict:
        messages = [*(history or []), {"role": "user", "content": content if isinstance(content, str) else ""}]
        return {"result": chat_llm.EchoClient().complete(system, messages), "usage": {}, "provider": "echo", "model": "echo"}


class Registry:
    def __init__(self, setup: ModelSetup, gates: ProviderGates, workdir: Path, environ: Mapping[str, str] = os.environ,
                 backends: Callable[[ProviderConfig, str, UseConfig, Path], object] | None = None) -> None:
        self.setup = setup
        self.gates = gates
        self.workdir = Path(workdir)
        self.environ = environ
        self._backends = backends

    def provider(self, name: str) -> ProviderConfig:
        return self.setup.providers[name]

    def sees(self, ref: ModelRef | None) -> bool:
        if ref is None or ref.provider not in self.setup.providers:
            return False
        return self.provider(ref.provider).sees(ref.model)

    def call(self, use: str, *, timeout: float | None = None, max_retries: int | None = None,
             cwd: Path | None = None) -> "GatedCall":
        return GatedCall(self, use, timeout=timeout, max_retries=max_retries, cwd=cwd or self.workdir / use)

    def backend(self, ref: ModelRef, use: str, cwd: Path, max_retries: int | None = None):
        """这个用处在这家这个模型上的后端（测试可以注入 backends）。"""
        provider = self.provider(ref.provider)
        usecfg = self.setup.uses[use]
        if self._backends is not None:
            return self._backends(provider, ref.model, usecfg, cwd)
        if provider.kind == "echo":
            return EchoBackend()
        if provider.kind == "claude-code":
            from .claude_code import ClaudeCodeBackend

            return ClaudeCodeBackend(provider, ref.model, cwd, environ=self.environ, persist=use != "image_label")
        from .openai_compat import OpenAIBackend

        return OpenAIBackend(provider, ref.model, temperature=usecfg.temperature, max_tokens=usecfg.max_tokens,
                             max_retries=max_retries, environ=self.environ)

    def current(self, use: str) -> ModelRef | None:
        """这个用处现在实际走哪个：主那家闸开着走主，否则备（备那家也开着）；都不行 None。"""
        u = self.setup.uses[use]
        if u.disabled:
            return None
        for ref in (u.main, u.backup):
            if ref is not None and self.gates.ok(ref.provider):
                return ref
        return None

    def describe(self, use: str) -> str:
        """给面板 / status / 日志的一行字：「deepseek/deepseek-chat（备 claude/sonnet）」，停用写「停用：原因」。"""
        u = self.setup.uses[use]
        if u.disabled:
            return f"停用：{u.disabled}"
        text = str(u.main) if u.main is not None else "（主模型用不了）"
        return text + (f"（备 {u.backup}）" if u.backup is not None else "")

    def requirements(self, uses: Iterable[str]) -> list[Problem]:
        """这些用处的主 / 备用到的每家：claude-code 查令牌和 claude 命令，openai 查 Key。只返回问题、不抛。"""
        out: list[Problem] = []
        seen: set[str] = set()
        for use in uses:
            u = self.setup.uses[use]
            for ref in (u.main, u.backup):
                if ref is None or ref.provider in seen:
                    continue
                seen.add(ref.provider)
                provider = self.provider(ref.provider)
                problem = self._missing(provider)
                if problem:
                    out.append(Problem(problem, use=use, provider=provider.id))
        return out

    def _missing(self, provider: ProviderConfig) -> str:
        if provider.kind == "claude-code":
            from .claude_code import claude_base

            try:
                claude_base(provider, self.environ)
            except ModelError as exc:
                return f"{provider.id}：{exc}"
        elif provider.kind == "openai":
            env = provider.key_env
            if not env or not (self.environ.get(env) or chat_llm._user_env(env)):
                return f"{provider.id} 缺 Key（环境变量 {env or '没写 key_env'}）"
        return ""

    def log_summary(self) -> None:
        for name, u in self.setup.uses.items():
            log.info("模型：%s = %s", name, self.describe(name))
        for line in self.setup.legacy:
            log.warning("配置：%s", line)
        for p in self.setup.problems:
            log.warning("模型配置：%s", p.text)


class GatedCall:
    """一个用处的调用：主那家闸开走主；主出关闸类错误（额度 / 余额 / 认证）→ 关主的闸、这一笔改走备；
    主的闸关着直接走备；备也关了 / 没有备 → ModelUnavailable。别的错（超时、限流、格式）原样抛、不关闸。"""

    def __init__(self, registry: Registry, use: str, *, timeout: float | None, max_retries: int | None, cwd: Path) -> None:
        self.registry = registry
        self.use = use
        u = registry.setup.uses[use]
        self.main, self.backup = u.main, u.backup
        self._timeout = timeout
        self.max_retries = max_retries
        self.cwd = cwd
        self._short = False  # timeout 改过（下线反思）：备用不重试
        self._cache: dict[ModelRef, object] = {}

    @property
    def timeout(self) -> float | None:
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        """下线反思压超时：主和备都用新值，备用不重试。"""
        self._timeout = value
        self._short = True
        self._cache.clear()

    def available(self) -> bool:
        return self.registry.current(self.use) is not None

    def _backend(self, ref: ModelRef, backup: bool):
        if ref not in self._cache:
            retries = 0 if (backup and self._short) else self.max_retries
            self._cache[ref] = self.registry.backend(ref, self.use, self.cwd, retries)
        return self._cache[ref]

    def _send(self, ref: ModelRef, backup: bool, system, content, history, max_tokens) -> dict:
        kw = {"history": history, "max_tokens": max_tokens or self.registry.setup.uses[self.use].max_tokens}
        if self._timeout is not None:
            kw["timeout"] = self._timeout
        return self._backend(ref, backup).message(system, content, **kw)

    def message(self, system: str, content: str | list[dict], *, history: list[dict] | None = None,
                max_tokens: int | None = None) -> dict:
        reg, gates = self.registry, self.registry.gates
        u = reg.setup.uses[self.use]
        if u.disabled:
            raise ModelUnavailable(f"{self.use} 用不了：{u.disabled}")
        main, backup = self.main, self.backup
        if main is not None and gates.ok(main.provider):
            try:
                return self._send(main, False, system, content, history, max_tokens)
            except ModelError as exc:
                kind = down_kind(exc)
                if kind is None:
                    raise
                gates.trip(main.provider, kind, str(exc))
                if backup is None or not gates.ok(backup.provider):
                    raise
        elif backup is None or not gates.ok(backup.provider):
            why = "；".join(f"{p}：{r}" for p, r in gates.closed().items()) or "主模型用不了"
            raise ModelUnavailable(f"{self.use} 用不了（{why}）")
        if main is not None:
            log.debug("%s 这一笔改走 %s（%s %s）", self.use, backup, main.provider, gates.reason(main.provider))
        try:
            return self._send(backup, True, system, content, history, max_tokens)
        except ModelError as exc:
            kind = down_kind(exc)
            if kind is not None:
                gates.trip(backup.provider, kind, str(exc))
            raise

    def text(self, system: str, content: str | list[dict]) -> str:
        return self.message(system, content)["result"]

    def complete(self, system: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        """LlmClient 接口：最后一条是这一次的内容，前面的当 history。"""
        if not messages:
            return self.message(system, "", max_tokens=max_tokens)["result"]
        return self.message(system, messages[-1]["content"], history=list(messages[:-1]) or None,
                            max_tokens=max_tokens)["result"]
