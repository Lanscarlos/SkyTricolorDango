"""启动前预检：不过就不起子进程，原因直接显示在总览的启动按钮下面。"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from .runner import LaunchOptions
from .settings import SettingsStore


def preflight(
    store: SettingsStore, opts: LaunchOptions, busy: bool, find_spec: Callable[[str], object] = importlib.util.find_spec
) -> list[str]:
    problems: list[str] = []
    if busy:
        problems.append("团子已经在运行")
    error = store.view()["error"]
    if error is None:
        try:
            cfg = store.effective()
        except ValueError as exc:
            error = str(exc)
    if error is not None:
        problems.append(error)
        return problems
    if opts.brain:
        if not store.secret("claude"):
            problems.append("大脑模式要 Claude 令牌：去设置页填")
        if find_spec("mcp") is None:  # 和 run 的检查一致
            problems.append("大脑要用 mcp：先 pip install --user mcp")
    elif cfg.llm.provider != "echo" and not store.secret("llm"):
        problems.append("普通模式要大模型的 API Key：去设置页填")
    return problems
