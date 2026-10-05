"""启动前预检：不过就不起子进程，原因直接显示在真机团子页的启动按钮下面。"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from .runner import LaunchOptions
from .settings import SettingsStore


def problem(text: str, setting: str | None = None) -> dict:
    """一条预检问题；setting 是该去填的那一项（设置页的键，或「模型」页的 models.<用处>），页面据此给跳转，没有就是 None。"""
    return {"text": text, "setting": setting}


class SecretEnv(Mapping):
    """Registry 查 Key / 令牌用的环境：按名字从 secrets.toml → 环境变量 → Windows 用户环境变量取（同子进程拿到的）。"""

    def __init__(self, store: SettingsStore) -> None:
        self.store = store

    def __getitem__(self, name: str) -> str:
        value = self.store.secret_env(name)
        if not value:
            raise KeyError(name)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def model_problems(store: SettingsStore, cfg, brain: bool) -> list[dict]:
    """按用处查模型：大脑模式只在大脑的主和备都用不了时拦，普通模式查回复的主模型；别的用处有问题不拦（到时那一处停用）。"""
    from ..models.config import resolve
    from ..models.gate import ProviderGates
    from ..models.registry import Registry

    registry = Registry(resolve(cfg), ProviderGates(), Path("tmp"), environ=SecretEnv(store))
    use = "brain" if brain else "reply"
    u = registry.setup.uses[use]
    issues = registry.requirements([use])
    bad = {p.provider for p in issues}
    refs = [r for r in ((u.main, u.backup) if brain else (u.main,)) if r is not None]
    if u.disabled or not refs or all(r.provider in bad for r in refs):
        mine = {r.provider for r in refs}
        why = u.disabled or "；".join(p.text for p in issues if p.provider in mine) or "没有选模型"
        head = "大脑没有能用的模型：" if brain else "普通模式的回复模型用不了："
        return [problem(head + why, f"models.{use}")]
    return []


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
    problems += model_problems(store, cfg, opts.brain)
    if opts.brain and find_spec("mcp") is None:  # 和 run 的检查一致
        problems.append(problem("大脑要用 mcp：先 pip install --user mcp"))
    return problems
