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
