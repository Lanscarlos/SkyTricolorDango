"""难例收集：YOLO 感知层运行时把"可能认错了"的画面存下来，下一轮只补标这些（设计见一期文档 §5）。

    runs/<这次>/hard/<时分秒>_<原因>.jpg
    runs/<这次>/hard.jsonl     每行：文件名、原因、细节、这一帧的检测框

原因：
- ocr_only        旁路核对：整图 OCR 读到了某个好友，YOLO 这一帧没认出
- yolo_only       反过来（可能只是 OCR 漏了，优先级低）
- low_conf        一条轨迹连续 0.5 s 都是低置信度（low_conf ~ conf 之间）
- flicker         同一位置 3 s 内人物框出现又消失 3 次以上
- unlit_vs_player 同一个人在 player 和 player_unlit 之间来回变
- appearance      按外观认成的好友（maybe）和后来挂上的名字标签对不上（感知层调 report）
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..brain.images import difference, thumb
from ..config import EnvConfig, PerceptionConfig
from ..imageio import imwrite
from .bubbles import Rect
from .detect import Detection
from .env import match_names, scan_area
from .track import Track, Tracker, iou

log = logging.getLogger(__name__)

SAVE_GAP = 5.0  # 两次保存至少隔几秒
SAME = 0.03  # 和上一张存下的图缩略图差异小于这个就不存
LOW_FOR = 0.5  # 低置信度要持续几秒
FLICKER_WINDOW = 3.0
FLICKER_BIRTHS = 3
PEOPLE = ("player", "player_unlit")


def _boxes(tracks: list[Track], low: list[Detection]) -> list[dict]:
    out = [{"cls": t.cls, "x": t.box.x, "y": t.box.y, "w": t.box.w, "h": t.box.h, "score": round(t.score, 3)} for t in tracks]
    out += [{"cls": d.cls, "x": d.box.x, "y": d.box.y, "w": d.box.w, "h": d.box.h, "score": round(d.score, 3), "low": True}
            for d in low]
    return out


class HardCaseCollector:
    def __init__(
        self,
        folder: Path,
        cfg: PerceptionConfig,
        env_cfg: EnvConfig,
        log_roi: list[float],
        names: Callable[[], list[str]],
        ocr=None,  # 旁路核对用的整图 OCR；None = 不核对
        background: bool = True,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.folder = folder
        self.jsonl = folder.parent / "hard.jsonl"
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.log_roi = log_roi
        self.names = names
        self.ocr = ocr
        self.background = background
        self.clock = clock
        self.wall = wall
        self.saved = 0
        self._low = Tracker(cfg.track_buffer, cfg.track_iou)  # 低置信度的框单独追踪
        self._fired: set[tuple[str, int]] = set()  # (原因, 轨迹 id)：每条轨迹只存一次
        self._births: deque[tuple[float, Rect]] = deque()
        self._last_save = float("-inf")
        self._last_thumb: np.ndarray | None = None
        self._last_audit = float("-inf")
        self._auditing = False
        self._lock = threading.Lock()

    def check(self, frame: np.ndarray, now: float, tracks: list[Track], low: list[Detection], seen: set[str],
              panel_visible: bool) -> None:
        """感知层每处理完一帧（没暂停时）调一次。"""
        reasons: list[tuple[str, str]] = []
        for t in self._low.update(low, now):
            if t.last - t.first >= LOW_FOR and ("low_conf", t.id) not in self._fired:
                self._fired.add(("low_conf", t.id))
                reasons.append(("low_conf", f"{t.cls} 置信度 {t.score:.2f}，持续 {t.last - t.first:.1f} 秒"))
        while self._births and now - self._births[0][0] > FLICKER_WINDOW:
            self._births.popleft()
        for t in tracks:
            if t.cls in PEOPLE and t.hits == 1:
                again = sum(iou(box, t.box) >= 0.3 for _, box in self._births) + 1
                self._births.append((now, t.box))
                if again >= FLICKER_BIRTHS:
                    reasons.append(("flicker", f"({t.box.x},{t.box.y}) 附近 {FLICKER_WINDOW:.0f} 秒内出现了 {again} 次"))
            if t.flips >= 2 and ("unlit_vs_player", t.id) not in self._fired:
                self._fired.add(("unlit_vs_player", t.id))
                reasons.append(("unlit_vs_player", f"轨迹 {t.id} 在 player / player_unlit 之间变了 {t.flips} 次"))
        if reasons:
            reason = reasons[0][0]
            self._save(frame, now, reason, "；".join(d for _, d in reasons), _boxes(tracks, low))
        self._maybe_audit(frame, now, tracks, low, seen, panel_visible)

    def report(self, frame: np.ndarray, now: float, reason: str, detail: str, tracks: list[Track]) -> bool:
        """别处发现的难例（比如按外观认错了人）：照常限量、去重后存下。存了返回 True。"""
        return self._save(frame, now, reason, detail, _boxes(tracks, []))

    # ---- 旁路核对：隔一阵整图 OCR 一次，和 YOLO 认出的好友比 ----
    def _maybe_audit(self, frame, now, tracks, low, seen, panel_visible) -> None:
        if self.ocr is None or self.cfg.audit_interval <= 0 or now - self._last_audit < self.cfg.audit_interval:
            return
        with self._lock:
            if self._auditing:
                return
            self._auditing = True
        self._last_audit = now
        args = (frame.copy(), now, _boxes(tracks, low), set(seen), panel_visible)
        if self.background:
            threading.Thread(target=self._audit, args=args, name="hardcases", daemon=True).start()
        else:
            self._audit(*args)

    def _audit(self, frame, now, boxes, seen, panel_visible) -> None:
        try:
            height, width = frame.shape[:2]
            area = scan_area(width, height, self.env_cfg, self.log_roi, panel_visible)
            read = {name for name, _ in match_names(self.ocr.recognize(area.crop(frame)), self.names(), 0.9)}
            if read - seen:
                detail = f"OCR 读到 {'、'.join(sorted(read))}，YOLO 认出 {'、'.join(sorted(seen)) or '（没有）'}"
                self._save(frame, now, "ocr_only", detail, boxes)
            elif seen - read:
                detail = f"YOLO 认出 {'、'.join(sorted(seen))}，OCR 读到 {'、'.join(sorted(read)) or '（没有）'}"
                self._save(frame, now, "yolo_only", detail, boxes)
        except Exception:
            log.exception("难例核对出错")
        finally:
            with self._lock:
                self._auditing = False

    def _save(self, frame: np.ndarray, now: float, reason: str, detail: str, boxes: list[dict]) -> bool:
        small = thumb(frame)
        with self._lock:
            if self.saved >= self.cfg.hardcase_max or now - self._last_save < SAVE_GAP:
                return False
            if self._last_thumb is not None and difference(self._last_thumb, small) < SAME:
                return False
            self.saved += 1
            self._last_save, self._last_thumb = now, small
        stamp = time.strftime("%H%M%S", time.localtime(self.wall()))  # Windows 文件名不能有冒号
        name = f"{stamp}_{reason}.jpg"
        self.folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, 100):
            if not (self.folder / name).exists():
                break
            name = f"{stamp}-{i}_{reason}.jpg"
        imwrite(self.folder / name, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entry = {"file": name, "reason": reason, "detail": detail, "boxes": boxes}
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.debug("存了难例 %s：%s", name, detail)
        if self.saved == self.cfg.hardcase_max:
            log.info("难例存满 %d 张，这次不再存", self.cfg.hardcase_max)
        return True
