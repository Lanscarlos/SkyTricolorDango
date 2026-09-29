"""面板的动手部分：按特征卡上写的办法关面板、点面板上的按钮。每一步都截图确认面板真的变了。

识别在 vision/panels.py；该不该按某个按钮（安全规则）由身体（brain/body.py）判断，这里只负责按。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ..brain.images import difference, thumb
from ..config import PanelsConfig
from ..vision.panels import UNKNOWN, Button, Panel, PanelReading, PanelState, PanelWatcher
from .social import touch_mode

log = logging.getLogger(__name__)

LINUX_KEY_ESC = 1
PRESS_CHANGED = 0.02  # 按完画面缩略图差异低于这个 = 画面没变
WAKE_DELAY = 0.6  # 键盘模式下第一下触摸只切到触屏模式（game-ops §1），等它切过去


def _center(button: Button) -> tuple[int, int]:
    return button.box.x + button.box.w // 2, button.box.y + button.box.h // 2


class PanelOps:
    def __init__(
        self,
        device,
        watcher: PanelWatcher,
        cfg: PanelsConfig,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.watcher = watcher
        self.cfg = cfg
        self.sleep = sleep
        self.clock = clock

    def close(self, panel: Panel, reading: PanelReading | None = None) -> bool:
        """关掉面板：卡片上的关法 → 撤退类按钮 → 右上角 ×，依次试，每试一种截图看它还在不在。关上了返回 True。"""
        card = self.watcher.cards.get(panel.name) if panel.name != UNKNOWN else None
        attempts: list[tuple[str, Callable[[], bool]]] = []
        for way in card.close_ways if card is not None else ():
            attempts.append((way, lambda way=way: self._way(way, panel, reading)))
        attempts.append(("撤退类按钮", lambda: self._retreat(panel, reading)))
        attempts.append(("×", lambda: self._close_mark(panel)))
        for what, attempt in attempts:
            if not attempt():  # 这种办法用不上（没有这个按钮、没有 ×）：不发输入，试下一种
                continue
            self.sleep(self.cfg.ui_delay)
            if not self.watcher.present(self.device.screenshot(), panel):
                log.info("用 %s 关掉了%s", what, panel.label)
                self.watcher.mark_closed(panel.name)
                return True
            log.info("用 %s 关%s，没关上", what, panel.label)
        return False

    def press(self, reading: PanelReading, button: Button) -> tuple[PanelState, bool]:
        """点一个按钮，返回 (按完的面板状态, 画面变了没有)。never 按钮在这里也拦一道（安全规则在身体里）。"""
        if button.kind == "never":
            raise ValueError(f"「{button.text}」是不能按的按钮")
        panel = reading.panel
        before = self.device.screenshot()
        self._tap(*_center(button), panel)
        self.sleep(self.cfg.ui_delay)
        after = self.device.screenshot()
        changed = difference(thumb(before), thumb(after)) >= PRESS_CHANGED
        if not self.watcher.present(after, panel):
            self.watcher.mark_closed(panel.name)
        return self.watcher.state, changed

    # ---- 内部 ----
    def _way(self, way: str, panel: Panel, reading: PanelReading | None) -> bool:
        if way == "esc":
            self.device.hw_key(LINUX_KEY_ESC)
        elif way.startswith("key:"):
            self.device.hw_key(int(way[4:]))
        elif way.startswith("tap:"):
            x, y = (float(v) for v in way[4:].split(","))
            height, width = self.device.screenshot().shape[:2]
            self._tap(round(x * width), round(y * height), panel)
        elif way.startswith("button:"):
            word = way[7:]
            reading = reading or self.watcher.read(self.device.screenshot(), panel, self.clock())
            button = next((b for b in reading.buttons if word in b.text and b.kind != "never"), None)
            if button is None:
                return False
            self._tap(*_center(button), panel)
        else:
            log.warning("不认识的关面板办法：%s", way)
            return False
        return True

    def _retreat(self, panel: Panel, reading: PanelReading | None) -> bool:
        reading = reading or self.watcher.read(self.device.screenshot(), panel, self.clock())
        button = next((b for b in reading.buttons if b.kind == "retreat"), None) if reading is not None else None
        if button is None:
            return False
        self._tap(*_center(button), panel)
        return True

    def _close_mark(self, panel: Panel) -> bool:
        pos = self.watcher.find_close(self.device.screenshot())
        if pos is None:
            return False
        self._tap(*pos, panel)
        return True

    def _tap(self, x: int, y: int, panel: Panel) -> None:
        """点屏幕。键盘模式下第一下只切到触屏模式（game-ops §1）：只有确认切过去了、面板还在、面板上的内容也没变，
        才补点第二下。拿不准第一下有没有生效就不补点 —— 宁可没点中（调用方会看到"画面没变"），也别连点两下
        落到新弹出的东西上（第二个弹框的按钮没过安全规则）。"""
        first = self.device.screenshot()
        if touch_mode(first):
            self.device.tap(x, y)
            return
        self.device.tap(x, y)
        self.sleep(WAKE_DELAY)
        after = self.device.screenshot()
        if not touch_mode(after):
            log.info("点了一下还没切到触屏模式：不确定这一下有没有生效，不补点")
            return
        if not self.watcher.present(after, panel):
            return  # 第一下已经把面板关了
        if difference(thumb(panel.box.crop(first)), thumb(panel.box.crop(after))) >= PRESS_CHANGED:
            return  # 面板上的内容变了：第一下已经生效
        self.device.tap(x, y)
