"""真机运行时把身份板每帧的输入存下来（runs/<这次>/identity/），离线用 identity_eval.replay_board 逐帧重放。

frames.pkl 是一串 pickle（一帧一个 FrameIn，和 identity-eval 的 frames.pkl 读出来一样是 list）；
frames/<秒>.jpg 每 snap_every 秒一张缩小的画面，存了图的那一帧 tag 就是图名，重放时对得上画面、标答案。
被强杀时最后一帧可能只写了一半，读的时候丢掉。写盘出错只记一次日志、之后不再存，不影响身份板。
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from skydango.imageio import imwrite

from .identity import FrameIn

log = logging.getLogger(__name__)

SNAP_WIDTH = 960


class FrameLog:
    def __init__(self, folder: Path, snap_every: float = 1.0, snap_max: int = 900) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._fh = (self.folder / 'frames.pkl').open('ab')
        self.snap_every, self.snap_max = snap_every, snap_max
        self._snaps = 0
        self._snap_at: float | None = None
        self.off = False

    def record(self, fi: FrameIn, frame: np.ndarray | None) -> FrameIn:
        """存这一帧；到了存图的时候把 tag 换成图名再存。返回（可能带上 tag 的）这一帧。"""
        if self.off or self._fh.closed:  # 关了之后感知线程可能还送来最后一帧
            return fi
        try:
            if (frame is not None and self._snaps < self.snap_max
                    and (self._snap_at is None or fi.now - self._snap_at >= self.snap_every - 1e-9)):
                fi = replace(fi, tag=f'{fi.now:.3f}')
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (SNAP_WIDTH, round(h * SNAP_WIDTH / w))) if w > SNAP_WIDTH else frame
                imwrite(self.folder / 'frames' / f'{fi.tag}.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self._snap_at = fi.now
                self._snaps += 1
            pickle.dump(fi, self._fh, protocol=pickle.HIGHEST_PROTOCOL)
            self._fh.flush()
        except Exception:
            log.exception('存身份板的帧出错，这次运行不再存')
            self.off = True
        return fi

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def load_frames(path: Path) -> list[FrameIn]:
    frames = []
    with Path(path).open('rb') as fh:
        while True:
            try:
                frames.append(pickle.load(fh))
            except EOFError:
                break
            except (pickle.UnpicklingError, ValueError, AttributeError, IndexError):
                break  # 被强杀时写了一半的最后一帧
    return frames
