"""“接世界的东西”（brain-sandbox spec §2）：设备、读聊天、身边、轮盘、说话……和时钟。

`_run_brain` 只管组装身体 / 大脑 / 内心层；它从 `World` 拿这些部件。
真机是 `cli._game_world`（原样搬现有的建法），沙盒是 `sandbox.world.sandbox_world`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


def _nothing() -> None:
    return None


@dataclass
class World:
    device: Any
    reader: Any
    self_filter: Any
    panel: Any  # chat.panel.PanelManager
    env: Any  # vision.env.EnvWatcher / PerceptionWatcher / 沙盒的名单
    social: Any
    emotes: Any
    camera: Any
    locomotion: Any
    sender: Any
    friend_checker: Any
    panels: Any
    panel_ops: Any
    clock: Callable[[], float] = time.monotonic  # 间隔用
    wall: Callable[[], float] = time.time  # 墙上时间
    describe: Callable[[list[dict]], str] | None = None  # 眼睛把画面写成文字；None = 用 Haiku
    name: str = "game"
    close: Callable[[], None] = _nothing  # 收尾：停 env、关 panels
    restore: Callable[[], None] = _nothing  # 身体收尾之后最后做：切回用户的输入法
    text_only: bool = False  # 沙盒：没有画面，look / look_person 只给文字


@dataclass
class BrainParts:
    """_run_brain 组装好的东西（身体建好、线程启动前交给 on_ready；沙盒拿去挂接口）。"""

    body: Any  # brain.body.Body
    eyes: Any  # brain.eyes.Eyes
    events: Any  # brain.events.EventQueue
    brain: Any  # brain.loop.Brain
    trace: Any  # brain.trace.BrainTrace | None
    reflector: Any  # inner.reflect.Reflector | None
    ledger: Any  # inner.ledger.Ledger | None
    store: Any  # chat.memory.MemoryStore | None
    mind_log: Any  # inner.log.MindLog | None
    usage: Any = None  # UsageMeter.snapshot：沙盒挂到 /usage（spec 2026-10-06-model-usage §6.1）
