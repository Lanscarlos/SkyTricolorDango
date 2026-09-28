"""YOLO 检测器：视觉的第一道关卡，只回答"画面里有什么、在哪"（设计见 docs/superpowers/specs/…-v0.2.md）。

两种后端，按模型文件的扩展名选：
- `.onnx` → onnxruntime（有 onnxruntime-gpu 就用 CUDA，没有退回 CPU；不需要 torch）
- `.pt` / `.engine` → ultralytics（训练时本来就装了 torch；`.engine` 是 TensorRT）

类别名优先从模型里读（ultralytics 导出的 ONNX 元数据里带 names），读不到用配置里的 `perception.classes`。
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .bubbles import Rect

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    cls: str
    box: Rect  # 整张截图坐标
    score: float


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> list[Detection]: ...


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放后补边到 size×size（和 ultralytics 训练时一致，补灰 114）。返回 (图, 缩放比例, (左边补了多少, 上边补了多少))。"""
    height, width = img.shape[:2]
    scale = min(size / width, size / height)
    nw, nh = round(width * scale), round(height * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, (left, top)


def _to_rect(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Rect | None:
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return Rect(x1, y1, x2 - x1, y2 - y1)


def decode(
    out: np.ndarray,
    names: list[str],
    scale: float,
    pad: tuple[int, int],
    shape: tuple[int, int],
    conf: float,
    iou: float,
) -> list[Detection]:
    """把 YOLO 的原始输出换成整张截图坐标的检测框。

    支持两种输出：
    - YOLOv8 / 11：(1, 4 + 类别数, N)，每列 cx, cy, w, h + 各类分数，要自己做 NMS
    - 端到端（YOLO26 / v10 默认导出）：(1, N, 6)，每行 x1, y1, x2, y2, 分数, 类别，已经去重
    """
    pred = np.asarray(out, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    height, width = shape
    left, top = pad
    dets: list[Detection] = []
    if pred.shape[1] == 6 and pred.shape[0] != 4 + len(names):  # 端到端
        for x1, y1, x2, y2, score, c in pred:
            if score < conf or not 0 <= int(c) < len(names):
                continue
            rect = _to_rect((x1 - left) / scale, (y1 - top) / scale, (x2 - left) / scale, (y2 - top) / scale, width, height)
            if rect is not None:
                dets.append(Detection(names[int(c)], rect, float(score)))
        return dets
    rows = pred.T  # (N, 4 + 类别数)
    scores = rows[:, 4:]
    cls = scores.argmax(axis=1)
    best = scores[np.arange(len(rows)), cls]
    keep = best >= conf
    rows, cls, best = rows[keep], cls[keep], best[keep]
    if not len(rows):
        return dets
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    x1, y1 = (cx - w / 2 - left) / scale, (cy - h / 2 - top) / scale
    bw, bh = w / scale, h / scale
    boxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
    idx = cv2.dnn.NMSBoxesBatched(boxes, best.tolist(), cls.tolist(), conf, iou)
    for i in np.asarray(idx).reshape(-1):
        bx, by, bw_, bh_ = boxes[i]
        rect = _to_rect(bx, by, bx + bw_, by + bh_, width, height)
        if rect is not None and 0 <= int(cls[i]) < len(names):
            dets.append(Detection(names[int(cls[i])], rect, float(best[i])))
    return dets


def _parse_names(raw: str | None) -> list[str] | None:
    """ultralytics 写进 ONNX 元数据的 names 形如 "{0: 'player', 1: 'name_tag'}"。"""
    if not raw:
        return None
    try:
        data = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(data, dict):
        return [str(data[k]) for k in sorted(data)]
    if isinstance(data, (list, tuple)):
        return [str(n) for n in data]
    return None


class OnnxYoloDetector:
    def __init__(self, path: str | Path, classes: list[str], imgsz: int = 640, conf: float = 0.35, iou: float = 0.5,
                 device: str = "cuda") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError('没有安装 onnxruntime：pip install onnxruntime（N 卡用 onnxruntime-gpu）') from exc
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.providers = self.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.providers:
            log.warning("要求用 GPU，但 onnxruntime 没有 CUDA（装的是 CPU 版，或者 CUDA 版本不对）：现在在 CPU 上跑")
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = _parse_names(meta.get("names")) or list(classes)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        fixed = inp.shape[-1]
        self.imgsz = fixed if isinstance(fixed, int) and fixed > 0 else imgsz  # 导出时固定了尺寸就按模型的来
        self.conf, self.iou = conf, iou

    def detect(self, img: np.ndarray) -> list[Detection]:
        boxed, scale, pad = letterbox(img, self.imgsz)
        blob = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0]
        return decode(out, self.names, scale, pad, img.shape[:2], self.conf, self.iou)


class UltralyticsDetector:
    def __init__(self, path: str | Path, imgsz: int = 640, conf: float = 0.35, iou: float = 0.5, device: str = "cuda") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('没有安装 ultralytics：pip install "skydango[yolo]"（50 系显卡要先装 cu128 的 torch）') from exc
        self.model = YOLO(str(path), task="detect")
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = 0 if device == "cuda" else "cpu"
        self.providers = [f"ultralytics:{device}"]
        names = self.model.names
        self.names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    def detect(self, img: np.ndarray) -> list[Detection]:
        result = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        height, width = img.shape[:2]
        boxes = result.boxes
        dets = []
        for (x1, y1, x2, y2), c, s in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            rect = _to_rect(x1, y1, x2, y2, width, height)
            if rect is not None:
                dets.append(Detection(self.names[int(c)], rect, float(s)))
        return dets


def make_detector(path: str | Path, classes: list[str], imgsz: int, conf: float, iou: float, device: str) -> Detector:
    path = Path(path)
    if path.suffix == ".onnx":
        if not path.exists():
            raise FileNotFoundError(f"找不到 YOLO 模型 {path}（先训练并导出，见 docs/superpowers/specs/…-perception-yolo-architecture-v0.2.md）")
        return OnnxYoloDetector(path, classes, imgsz, conf, iou, device)
    # .pt / .engine：交给 ultralytics（yolo11n.pt 这类官方权重不存在时它会自己下载）
    return UltralyticsDetector(path, imgsz, conf, iou, device)
