"""按供应商的接入方式建大脑会话：claude-code → 常驻的 BrainSession，openai → 每轮重发的 ToolLoopBrain。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import BrainConfig
from ..models.config import ModelRef
from ..models.registry import Registry


def make_session(registry: Registry, ref: ModelRef, *, prompt: str, toolbox, mcp_url: str, workdir: Path,
                 cfg: BrainConfig, addressee: bool, on_message: Callable[[dict], None] | None,
                 inbox: Callable[[], str] | None = None, trace=None, backup: bool = False) -> object:
    """建失败（缺令牌 / Key、找不到 claude）抛 ModelError(down="auth")。
    inbox / trace / backup 只给 OpenAI 兼容的大脑（历史压缩，spec 2026-10-06-brain-compact）：读 inbox.md、时间线、用量记主还是备。"""
    provider = registry.provider(ref.provider)
    if provider.kind == "claude-code":
        from ..models.claude_code import claude_base
        from .session import BrainSession

        base, env = claude_base(provider, registry.environ)
        return BrainSession(base, env, Path(workdir) / "session", mcp_url, prompt, ref.model, cfg.effort,
                            cfg.turn_timeout, on_message, provider=ref.provider, on_rate_limit=registry._rate_hook(ref.provider))
    from ..models.openai_compat import build_client
    from .llm_tools import openai_tools
    from .toolloop import ASIDE_NOTE, BLIND_NOTE, ToolLoopBrain

    use = registry.setup.uses["brain"]
    return ToolLoopBrain(
        build_client(provider, max_retries=0, environ=registry.environ),
        prompt + "\n\n" + BLIND_NOTE + (ASIDE_NOTE if addressee else ""), toolbox, openai_tools(toolbox),
        model=ref.model, temperature=use.temperature, max_tokens=use.max_tokens, max_steps=cfg.max_steps,
        turn_timeout=cfg.turn_timeout, on_message=on_message, history=cfg.history, provider=ref.provider,
        compact=cfg.compact if cfg.compact.enabled else None, inbox=inbox, meter=registry.meter, backup=backup,
        recap_path=Path(workdir) / "recap.jsonl", on_note=trace.note if trace is not None else None,
    )
