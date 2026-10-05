"""兼容旧的引用：Claude Code 的进程、命令、错误都搬到了 models/claude_code.py（新代码从 models 引）。

ClaudeLlm / GatedLlm / gated_describe 是 Claude 总闸那一期的写法，等 Registry 接上后删。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..models.claude_code import (AUTH_WORDS, LIMIT_WORDS, ClaudeError, StreamProcess, check_result, claude_env,  # noqa: F401
                                  one_shot, one_shot_message, resolve_claude, user_message)
from ..models.errors import down_kind
from ..models.gate import ProviderGates

log = logging.getLogger(__name__)


def claude_down(exc: BaseException) -> str | None:
    """兼容旧名：同 models.errors.down_kind。"""
    return down_kind(exc)


class ClaudeLlm:
    """记忆整理（随手记 inbox.md、整理 notes.md）用：和 chat.llm 的 LlmClient 同一个接口，每次起一个一次性 claude -p。

    令牌、配置目录同大脑（订阅，不花 API 钱）；不给任何工具。Claude Code 没有 max_tokens 参数，忽略。
    """

    def __init__(self, base: list[str], env: dict[str, str], model: str, cwd: Path, timeout: float = 120.0) -> None:
        self.base = base
        self.env = env
        self.model = model
        self.cwd = cwd
        self.timeout = timeout

    def command(self, system: str) -> list[str]:
        return [
            *self.base, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", self.model, "--effort", "low", "--tools", "", "--strict-mcp-config",
            "--permission-mode", "dontAsk", "--disable-slash-commands", "--system-prompt", system,
        ]

    def complete(self, system: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        content = "\n\n".join(m["content"] for m in messages)  # NotesKeeper 只发一条 user 消息
        return one_shot(self.command(system), self.env, self.cwd, content, self.timeout)


class GatedLlm:
    """给一次性文字调用套上总闸：闸开走 Claude；额度 / 认证出错就关闸、这一笔改走备用（DeepSeek）；闸关着直接走备用。

    超时等别的错原样抛出、不关闸。没有备用时闸关着抛 ClaudeError（不再起 claude 进程）。
    """

    def __init__(self, claude, backup, gate: ProviderGates, provider: str = "claude") -> None:
        self.claude = claude
        self.backup = backup
        self.gate = gate
        self.provider = provider

    @property
    def timeout(self) -> float | None:
        return getattr(self.claude, "timeout", None)

    @timeout.setter
    def timeout(self, value: float) -> None:
        """下线反思缩短超时用（cli._final_reflection）：Claude 改超时；备用换成同样超时、不重试的副本
        （记忆和反思共用一个备用，原来那个不动）。备用没有 with_timeout 就照旧。"""
        if hasattr(self.claude, "timeout"):
            self.claude.timeout = value
        shorten = getattr(self.backup, "with_timeout", None)
        if shorten is not None:
            self.backup = shorten(value, max_retries=0)

    def complete(self, system: str, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        if self.gate.ok(self.provider):
            try:
                return self.claude.complete(system, messages, max_tokens)
            except ClaudeError as exc:
                kind = claude_down(exc)
                if kind is None:
                    raise
                self.gate.trip(self.provider, kind, str(exc))
                if self.backup is None:
                    raise
        elif self.backup is None:
            raise ClaudeError(f"Claude 不能用了（{self.gate.reason(self.provider)}），也没有备用模型", limit=True)
        log.debug("Claude 不能用了：%s，这一笔改走 DeepSeek", self.gate.reason(self.provider))
        return self.backup.complete(system, messages, max_tokens)


def gated_describe(describe: Callable[[object], str], gate: ProviderGates, provider: str = "claude") -> Callable[[object], str]:
    """给看图（眼睛、装扮描述）套上总闸：闸关着不调 describe、直接抛；额度 / 认证出错关闸后原样抛。"""

    def wrapped(content):
        if not gate.ok(provider):
            raise ClaudeError("Claude 不能用了，看不了图", limit=True)
        try:
            return describe(content)
        except ClaudeError as exc:
            kind = claude_down(exc)
            if kind is not None:
                gate.trip(provider, kind, str(exc))
            raise

    return wrapped
