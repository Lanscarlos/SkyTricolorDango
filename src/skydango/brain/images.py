"""大脑看图用的图片处理：缩放、裁剪、编码成 API 要的 base64 JPEG；画面大变 / 黑屏的判断。"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from ..vision.people import OBJECT_NAMES


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


_SCENE_KINDS = {"friend": 0, "maybe": 1, "unlit": 2, "stranger": 3, "self": 4}
_SCENE_WHO = {"unlit": "陌生人（没点火，黑影）", "stranger": "陌生人", "self": "团子（就是“你”自己）"}


def _scene_who(b: dict) -> str:
    if b["kind"] == "maybe":  # 按外观认的好友：label 是"像小明?"
        return f"{b['label'].rstrip('?')}（没看到名字）"
    return _SCENE_WHO.get(b["kind"], b["label"])


def scene_note(env, now: float, scale: float) -> str:
    """YOLO 感知层认出的人（env.overlay）→ 给眼睛 / 大脑看的"谁在图里哪儿"。scale = 图宽 / 原图宽。"""
    boxes = env.overlay(now)
    people = sorted((b for b in boxes if b["kind"] in _SCENE_KINDS), key=lambda b: (_SCENE_KINDS[b["kind"]], b["x"]))
    rows = [
        f"- {_scene_who(b)}：({round((b['x'] + b['w'] / 2) * scale)}, {round((b['y'] + b['h'] / 2) * scale)}) 附近"
        for b in people
    ]
    friends = {b["label"] for b in people if b["kind"] == "friend"}
    rows[len(friends):len(friends)] = [  # 只看到名字标签、人没框出来的好友，排在好友后面
        f"- {b['label']}：头顶名字在 ({round((b['x'] + b['w'] / 2) * scale)}, {round(b['y'] * scale)})，人在名字下方"
        for b in sorted((b for b in boxes if b["kind"] == "name" and b["label"] not in friends), key=lambda b: b["x"])
    ]
    tail = "没列出的人都叫“陌生人”。"
    text = ("画面里没认出人（可能被挡住、离得远，或者没人）\n" + tail) if not rows else (
        "画面里认出的人（坐标按这张图）：\n" + "\n".join(rows) + "\n" + tail)
    things = env.objects(now) if hasattr(env, "objects") else []
    if things:  # 物品（座位 / 篝火 / 乐器 / 先祖）：只是提示，眼睛看到别的照样说
        text += "\n画面里认出的东西（坐标按这张图）：\n" + "\n".join(
            f"- {OBJECT_NAMES.get(t.kind, t.kind)}：({round((t.box.x + t.box.w / 2) * scale)}, {round((t.box.y + t.box.h / 2) * scale)}) 附近"
            for t in things
        ) + "\n没列出的东西按你自己看到的说。"
    icons = env.icons(now) if hasattr(env, "icons") else []
    if icons:  # 地图交互图标：只说明那里能互动
        text += "\n画面里认出的图标（坐标按这张图）：\n" + "\n".join(
            f"- {i.label}：({round((i.box.x + i.box.w / 2) * scale)}, {round((i.box.y + i.box.h / 2) * scale)}) 附近" for i in icons
        ) + "\n图标只说明那里能互动。"
    return text
