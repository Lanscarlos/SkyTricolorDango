"""兼容旧的引用：Claude Code 的进程、命令、错误都搬到了 models/claude_code.py（新代码从 models 引）。

调模型一律从 models.registry.Registry 拿（按用处、主 → 备），这里只留旧名字的转引。
"""

from __future__ import annotations

from ..models.claude_code import (AUTH_WORDS, LIMIT_WORDS, ClaudeError, StreamProcess, check_result, claude_env,  # noqa: F401
                                  one_shot, one_shot_message, resolve_claude, user_message)
from ..models.errors import down_kind


def claude_down(exc: BaseException) -> str | None:
    """兼容旧名：同 models.errors.down_kind。"""
    return down_kind(exc)

