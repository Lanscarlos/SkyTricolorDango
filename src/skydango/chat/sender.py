"""把一句话发到游戏里：点开聊天框 → 输入 → 提交。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ..config import SenderConfig
from ..device.base import KEYCODE_BACK, KEYCODE_ENTER, Device

log = logging.getLogger(__name__)


def to_pixels(point: list[float], width: int, height: int) -> tuple[int, int]:
    if len(point) != 2:
        raise ValueError(f"坐标应为 [x, y]，实际是 {point}")
    x, y = point
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"坐标要用 0~1 的归一化值，实际是 {point}")
    return int(round(x * width)), int(round(y * height))


class ChatSender:
    def __init__(
        self,
        device: Device,
        cfg: SenderConfig,
        screen_size: Callable[[], tuple[int, int]],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.device = device
        self.cfg = cfg
        self.screen_size = screen_size  # 返回 (width, height)
        self.sleep = sleep

    def _wait_open(self) -> None:
        """等输入框打开：实测按键后约 0.07 s 就开了，开了就走；最多等 open_delay。"""
        step, waited = 0.05, 0.0
        while waited < self.cfg.open_delay:
            self.sleep(step)
            waited += step
            if self.device.ime_shown():
                self.sleep(0.1)  # 留一点时间让输入法连上输入框
                return
        log.warning("等了 %.1f 秒输入框还没打开，照样输入", self.cfg.open_delay)

    def send(self, text: str) -> None:
        width, height = self.screen_size()
        cfg = self.cfg
        if (cfg.open_chat_key or cfg.open_chat) and not self.device.ime_shown():
            if cfg.open_chat_key:
                self.device.hw_key(cfg.open_chat_key)
            else:
                self.device.tap(*to_pixels(cfg.open_chat, width, height))
            self._wait_open()
        self.device.input_text(text)
        self.sleep(cfg.type_delay)
        if cfg.submit == "editor_action":
            self.device.editor_action(cfg.editor_action)
        elif cfg.submit == "enter":
            self.device.key(KEYCODE_ENTER)
        elif cfg.submit == "tap":
            if not cfg.send_button:
                raise ValueError("submit = \"tap\" 时必须配置 sender.send_button")
            self.device.tap(*to_pixels(cfg.send_button, width, height))
        else:
            raise ValueError(f"未知的 sender.submit: {cfg.submit}")
        self.sleep(cfg.after_delay)
        if cfg.close_with_back:
            self.device.key(KEYCODE_BACK)
        log.info("已发送: %s", text)
