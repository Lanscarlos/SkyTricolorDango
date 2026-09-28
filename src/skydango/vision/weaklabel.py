"""弱标注：用现有的识别器（整图 OCR 读名字标签 + 圆圈剪影模板）给录下来的画面自动出 YOLO 框，
人工只需要补 player / self 框、修正错框。输出 ultralytics 的 YOLO 数据集格式（X-AnyLabeling 可以直接导入）。

    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt   每行：类别号 cx cy w h（0~1 归一化）
    <out>/data.yaml
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..chat.tracker import normalize, similar
from .bubbles import Rect
from .detect import Detection
from .ocr import OcrLine
from .track import iou


def _clip(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Rect | None:
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    return Rect(x1, y1, x2 - x1, y2 - y1) if x2 > x1 and y2 > y1 else None


def weak_labels(
    frame: np.ndarray,
    lines: Iterable[OcrLine],
    names: list[str],
    icons=None,  # game.social.IconClassifier
    icon_offset: float = 2.23,
    skip: list[Rect] = (),  # 这些区域里的字不要（聊天面板、底部输入栏）
    keep: Rect | None = None,  # 只要这块区域里的字（3D 画面）
    all_text: bool = False,  # 不只要对得上好友名单的字：画面里读到的字都当名字标签候选（要人工删错的）
    min_score: float = 0.9,
) -> list[tuple[str, Rect]]:
    height, width = frame.shape[:2]
    k = height / 1080
    out: list[tuple[str, Rect]] = []
    for line in lines:
        text = line.text.strip()
        if line.score < min_score or len(normalize(text)) < 2:
            continue
        center = (line.box.x + line.box.w / 2, line.box.y + line.box.h / 2)
        inside = lambda r: r.x <= center[0] < r.x2 and r.y <= center[1] < r.y2  # noqa: E731
        if (keep is not None and not inside(keep)) or any(inside(r) for r in skip):
            continue
        if not all_text and not any(similar(text, n, 0.75) for n in names):
            continue
        tag = line.box.pad(4, width, height)
        out.append(("name_tag", tag))
        if icons is None:
            continue
        cx, cy = line.box.x + line.box.w // 2, line.box.y + round(icon_offset * line.box.h)
        kind, _ = icons.classify(frame[max(0, cy - 56) : cy + 56, max(0, cx - 56) : cx + 56])
        if kind:  # 认得出圆圈（包括平时的 ✦）才标
            r = round(50 * k)
            ring = _clip(cx - r, cy - r, cx + r, cy + r, width, height)
            if ring is not None:
                out.append(("social_ring", ring))
    return out


def yolo_line(cls: int, box: Rect, width: int, height: int) -> str:
    return (
        f"{cls} {(box.x + box.w / 2) / width:.6f} {(box.y + box.h / 2) / height:.6f} "
        f"{box.w / width:.6f} {box.h / height:.6f}"
    )


def split_of(name: str, val: float) -> str:
    """按文件名哈希分训练 / 验证：同一张图每次都分到同一边，补标注后重跑不会乱。"""
    return "val" if zlib.crc32(name.encode()) % 1000 < val * 1000 else "train"


def data_yaml(root: Path, classes: list[str]) -> str:
    lines = [f"path: {root.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    return "\n".join(lines) + "\n"


def merge_labels(weak: list[tuple[str, Rect]], predicted: list[Detection], min_iou: float = 0.5) -> list[tuple[str, Rect]]:
    """弱标注 + 模型预测（预标注）：同类框重叠（IoU > min_iou）时留弱标注的（OCR 的名字框更准），其余预测框都加上。"""
    out = list(weak)
    for det in predicted:
        if not any(c == det.cls and iou(box, det.box) > min_iou for c, box in weak):
            out.append((det.cls, det.box))
    return out


def with_self(boxes: list[tuple[str, Rect]], me: Rect, min_iou: float = 0.5) -> list[tuple[str, Rect]]:
    """转圈认出的团子（感知层二期 §1.3）：和它重叠的 player / self 框换成一个 self 框。"""
    out = [(c, b) for c, b in boxes if not (c in ("player", "self") and iou(b, me) >= min_iou)]
    return out + [("self", me)]


def hard_images(runs: Path) -> list[tuple[Path, str]]:
    """runs/*/hard/*.jpg（运行时收集的难例）→ (路径, "<运行目录名>_<文件名>")：不同次运行的文件名可能重复。"""
    out = []
    for path in sorted(Path(runs).glob("*/hard/*.jpg")):
        out.append((path, f"{path.parent.parent.name}_{path.stem}"))
    return out
