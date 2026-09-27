"""图片读写。cv2.imread / imwrite 在 Windows 上不支持中文路径，这里用 imencode / imdecode 绕开。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imwrite(path: str | Path, img: np.ndarray) -> None:
    ext = Path(path).suffix or ".png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    Path(path).write_bytes(buf.tobytes())


def imread(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不了图片: {path}")
    return img
