"""识别环境：从画面里看出身边有谁、在哪张图，写进提示词。

- 身边有谁：3D 场景里好友头顶有名字标签（实测 OCR 置信度 1.00），拿好友名单去匹配
- 在哪张图：画面上出现已知的地图名就记下（进入新区域时的地名提示，未在真机验证）

整块画面 OCR 一次要 0.7~1 s，所以每隔几秒在后台线程里扫一次，不耽误读聊天和回复。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from ..chat.tracker import normalize, similar
from ..config import EnvConfig
from .bubbles import roi_rect
from .ocr import OcrEngine

log = logging.getLogger(__name__)


class EnvWatcher:
    def __init__(
        self,
        ocr: OcrEngine,
        cfg: EnvConfig,
        names: Callable[[], list[str]],
        log_roi: list[float],
        background: bool = True,
        icons=None,  # game.social.IconClassifier：顺带看好友名字下方圆圈里的互动请求
        icon_offset: float = 2.23,  # 圆圈中心在名字上沿往下这么多倍标签高度
    ) -> None:
        self.ocr = ocr
        self.cfg = cfg
        self.names = names  # 好友名单（游戏昵称），每次扫描时现取，用户改了 friends.md 马上生效
        self.log_roi = log_roi  # 聊天记录面板的位置：面板开着时这块被挡住，不扫
        self.background = background
        self.icons = icons
        self.icon_offset = icon_offset
        self.requests: dict = {}  # 好友名 → game.social.Request（圆圈里是牵手 / 拥抱 / 击掌等图标）
        self.last_seen: dict[str, float] = {}
        self.place = ""
        self.place_at = float("-inf")
        self._last_scan = float("-inf")
        self._busy = False
        self._lock = threading.Lock()

    def observe(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        """主循环每帧调一次；到了间隔就扫一次（默认在后台线程里）。"""
        if now - self._last_scan < self.cfg.interval:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._last_scan = now
        height, width = frame.shape[:2]
        area = roi_rect(self.cfg.roi, width, height)
        if panel_visible:  # 面板挡住的左边不扫
            left = roi_rect(self.log_roi, width, height).x2
            if left > area.x:
                area = type(area)(left, area.y, max(1, area.x2 - left), area.h)
        region = area.crop(frame).copy()
        if self.background:
            threading.Thread(target=self._scan, args=(region, now, (area.x, area.y)), name="env", daemon=True).start()
        else:
            self._scan(region, now, (area.x, area.y))

    def _scan(self, region: np.ndarray, now: float, offset: tuple[int, int] = (0, 0)) -> None:
        try:
            lines = [line for line in self.ocr.recognize(region) if line.score >= self.cfg.min_score]
            names = self.names()
            seen = []
            for line in lines:
                text = line.text.strip()
                if len(normalize(text)) < 2:
                    continue
                for name in names:
                    if similar(text, name, 0.75):
                        self.last_seen[name] = now
                        seen.append(name)
                        self._check_request(region, line.box, name, now, offset)
                place = next((p for p in self.cfg.places if normalize(text) == normalize(p)), "")
                if place:
                    if place != self.place:
                        log.info("到了：%s", place)
                    self.place, self.place_at = place, now
            if seen:
                log.debug("旁边看到: %s", "、".join(seen))
        except Exception:
            log.exception("识别环境出错")
        finally:
            with self._lock:
                self._busy = False

    def _check_request(self, region: np.ndarray, box, name: str, now: float, offset: tuple[int, int]) -> None:
        """名字标签正下方的圆圈：平时是 ✦，有人发起牵手 / 拥抱 / 击掌时换成对应图标。"""
        if self.icons is None:
            return
        from ..game.social import IDLE, Request

        cx, cy = box.x + box.w // 2, box.y + round(self.icon_offset * box.h)
        kind, _ = self.icons.classify(region[max(0, cy - 56) : cy + 56, max(0, cx - 56) : cx + 56])
        if kind and kind != IDLE:
            if name not in self.requests or self.requests[name].kind != kind:
                log.info("%s 发起了互动：%s", name, kind)
            self.requests[name] = Request(name, kind, (cx + offset[0], cy + offset[1]), now)
        else:
            self.requests.pop(name, None)

    def nearby(self, now: float) -> list[str]:
        """最近 keep 秒内看到过名字标签的好友（按名单顺序）。标签会被挡住、会闪，所以不要求每次都看到。"""
        return [n for n in self.names() if now - self.last_seen.get(n, float("-inf")) <= self.cfg.keep]

    def describe(self, now: float) -> str:
        parts = []
        if self.place and now - self.place_at <= self.cfg.place_keep:
            parts.append(f"- 刚才画面上出现过地名：{self.place}（可能已经走到别处了）")
        people = self.nearby(now)
        if people:
            parts.append(f"- 你身边现在有：{'、'.join(people)}（画面上能看到他们头顶的名字）")
            parts.append("- 别的人看不到名字不代表不在（可能被挡住或离得远）")
        if not parts:
            return ""
        return "\n".join(parts)
