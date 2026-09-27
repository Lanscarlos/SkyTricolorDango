"""每次 `run` 一个目录：日志、面板识别记录、回复记录、截图、配置快照都放在一起，事后按场次翻看。

    runs/20260927-153012-live/
      agent.log       文件里一律记 DEBUG（终端照旧 INFO）
      rows.log        面板每次变化时每行的识别结果、哪几行被判成新的
      replies.jsonl   每轮：收到的消息 → 回复 → 发没发
      frames/*.jpg    读到新消息时的面板截图（标了框）
      config.json     本次实际生效的配置（含命令行覆盖）
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .imageio import imwrite
from .vision.bubbles import Rect, annotate, roi_rect

log = logging.getLogger(__name__)

_RUN_NAME = re.compile(r"^\d{8}-\d{6}")


def prune(root: Path, keep: int) -> list[Path]:
    """只留最近 keep 次运行（按目录名里的时间排），返回删掉的目录。keep <= 0 不删。"""
    if keep <= 0 or not root.is_dir():
        return []
    runs = sorted(p for p in root.iterdir() if p.is_dir() and _RUN_NAME.match(p.name))
    old = runs[: max(0, len(runs) - keep)]
    for p in old:
        shutil.rmtree(p, ignore_errors=True)
    return old


class RunDir:
    def __init__(self, path: Path, cfg: Config) -> None:
        self.path = path
        self.cfg = cfg
        self.frames = path / "frames"
        self.rows_log = path / "rows.log"
        self._handler: logging.Handler | None = None

    @classmethod
    def create(cls, cfg: Config, mode: str, now: float | None = None) -> "RunDir":
        root = Path(cfg.run.dir)
        # 先清理再建：新目录不算在要删的里面
        prune(root, cfg.run.keep - 1 if cfg.run.keep > 0 else 0)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        root.mkdir(parents=True, exist_ok=True)
        for i in range(100):
            path = root / (f"{stamp}-{mode}" if i == 0 else f"{stamp}-{mode}-{i}")
            try:
                path.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"建不了运行目录: {root / stamp}")
        run = cls(path, cfg)
        (path / "config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return run

    def attach_log(self) -> None:
        """skydango 的日志全量（DEBUG）写进 agent.log；终端的输出级别不变。"""
        root = logging.getLogger()
        for h in root.handlers:  # 下面把 skydango 调到 DEBUG，终端的 handler 要守住原来的级别
            if h.level == logging.NOTSET:
                h.setLevel(root.level)
        handler = logging.FileHandler(self.path / "agent.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)  # 挂在根上：第三方库（如 httpx 的请求记录）INFO 以上也会进来
        logging.getLogger("skydango").setLevel(logging.DEBUG)
        self._handler = handler

    def close(self) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def save_frame(self, frame: np.ndarray, boxes: Sequence[Rect]) -> Path | None:
        """标出新消息的框，只截面板（非 log 模式截 vision.roi）存 JPG。"""
        if not self.cfg.run.save_frames:
            return None
        height, width = frame.shape[:2]
        vision = self.cfg.vision
        area = roi_rect(vision.log_roi if vision.mode == "log" else vision.roi, width, height)
        crop = area.crop(annotate(frame, list(boxes)))
        now = time.time()
        name = time.strftime("%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}.jpg"
        path = self.frames / name
        imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.run.jpeg_quality])
        return path

    def record_reply(self, messages: Sequence, reply: str | None, sent: bool) -> None:
        """记一轮回复。reply 为 None：模型选择不回复（或回复被过滤掉了）。"""
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [{"speaker": m.speaker, "text": m.text} for m in messages],
            "reply": reply,
            "sent": sent,
        }
        with (self.path / "replies.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
