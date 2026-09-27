"""大脑看图用的图片处理：缩放、裁剪、编码成 API 要的 base64 JPEG；画面大变 / 黑屏的判断。"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_jpeg(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def image_block(img: np.ndarray, quality: int = 80) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_jpeg(img, quality)}}


def fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """缩到不超过 size（宽, 高），保持比例，不放大。1920×1080 → 1280×720 约 1.2k tokens。"""
    h, w = frame.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def crop_view(
    frame: np.ndarray, x: int, y: int, w: int, h: int, view_w: int, max_side: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按缩略图（宽 view_w）上的坐标截原图的一块，返回 (图, 原图上的 (x, y, w, h))；太大就缩到最长边 max_side。"""
    fh, fw = frame.shape[:2]
    s = fw / view_w
    x1, y1 = max(0, round(x * s)), max(0, round(y * s))
    x2, y2 = min(fw, round((x + w) * s)), min(fh, round((y + h) * s))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("区域太小或者在画面外（坐标按 look 返回的图给）")
    crop = frame[y1:y2, x1:x2]
    longest = max(crop.shape[:2])
    if longest > max_side:
        k = max_side / longest
        crop = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x2 - x1, y2 - y1)


def thumb(frame: np.ndarray) -> np.ndarray:
    """64×36 灰度缩略图，判断画面大变用。"""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """两张缩略图的平均差异，0~1。"""
    return float(np.mean(cv2.absdiff(a, b))) / 255.0


def is_black(frame: np.ndarray, max_mean: float = 20.0) -> bool:
    """整屏黑（牵手被带着走时见过，像切场景，只剩角色和圆圈）：平均亮度很低。阈值未在真机标定。"""
    return float(np.mean(thumb(frame))) < max_mean


def label_note(labels: dict, scale: float) -> str:
    """OCR 认出的好友名字 → 给大脑看的“名字在图里的位置”。labels 的值是 (x, y, w, h, ...)，scale = 图宽 / 原图宽。"""
    if not labels:
        return "图里没认出好友的名字（可能被挡住、离得远，或者没有好友在）"
    rows = [
        f"{name}：名字在 ({round((x + w / 2) * scale)}, {round(y * scale)})，人在名字下方"
        for name, (x, y, w, h, *_rest) in sorted(labels.items())
    ]
    return "图里认出的好友名字（名字在人头顶）：\n" + "\n".join(rows)
