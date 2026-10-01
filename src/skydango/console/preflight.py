"""启动前预检：不过就不起子进程，原因直接显示在真机团子页的启动按钮下面。"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from .runner import LaunchOptions
from .settings import SettingsStore


def problem(text: str, setting: str | None = None) -> dict:
    """一条预检问题；setting 是设置页上该去填的那一项（页面据此给「去设置」跳转），没有就是 None。"""
    return {"text": text, "setting": setting}


def preflight(
    store: SettingsStore, opts: LaunchOptions, busy: bool, find_spec: Callable[[str], object] = importlib.util.find_spec
) -> list[dict]:
    problems: list[dict] = []
    if busy:
        problems.append(problem("团子已经在运行"))
    error = store.view()["error"]
    if error is None:
        try:
            cfg = store.effective()
        except ValueError as exc:
            error = str(exc)
    if error is not None:
        problems.append(problem(error))
        return problems
    if opts.brain:
        if not store.secret("claude"):
            problems.append(problem("大脑模式要 Claude 令牌：去设置页填", "secret.claude"))
        if find_spec("mcp") is None:  # 和 run 的检查一致
            problems.append(problem("大脑要用 mcp：先 pip install --user mcp"))
        if cfg.llm.provider != "echo" and not store.secret("llm"):  # 大脑离线时聊天交给 [llm] 备用
            problems.append(problem("大脑离线时的备用回复要大模型的 API Key：去设置页填", "secret.llm"))
    elif cfg.llm.provider != "echo" and not store.secret("llm"):
        problems.append(problem("普通模式要大模型的 API Key：去设置页填", "secret.llm"))
    return problems
