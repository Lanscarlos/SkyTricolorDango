"""别人对团子做的动作（感知层三期 §3，**研究性质**）：好友对着团子挥手、鞠躬时认出来，发事件给大脑，由它决定回不回礼。

还没有模型，这里是数据工具和运行时接口：
- `extract_clips`（`perception clips`）：录像（`record --fps 8`）→ 按人物轨迹切成 16 帧（2 s）的片段，存成
  `<输出>/<序号>_track<轨迹>_t<开始秒>s/00.jpg…15.jpg`，人工把片段目录挪进 `<数据目录>/<动作>/`（`none` = 站着、走路、别的动作）
- `evaluate`（`perception gesture-eval`）：在分好类的片段上算每个动作的精确率 / 召回率（目标：精确率 ≥ 90%、召回率 ≥ 60%）
- `OnnxGestureClassifier`：模型约定——输入 `1×T×3×S×S`（RGB，0~1，T = frames、S = size），输出 `1×len(labels)` 的分数
  （logits 或概率，这里统一做 softmax），标签顺序同 `[gesture] labels`
- 运行时（`vision/perception.py`）：只对认出名字、距离近 / 中、在画面中间一半的好友，每条轨迹攒够一段、每 interval 秒判一次
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..config import GestureConfig
from ..imageio import imread, imwrite
from .bubbles import Rect
from .detect import Detection
from .onnxrt import open_session
from .sweep import distance
from .track import Tracker

log = logging.getLogger(__name__)

SUFFIXES = (".jpg", ".jpeg", ".png")


class GestureClassifier(Protocol):
    def classify(self, clip: list[np.ndarray]) -> tuple[str, float]: ...


def person_crop(img: np.ndarray, box: Rect, size: int) -> np.ndarray:
    """人物框四周放宽 15%（挥手时手会伸出框外），裁出来缩放成 size×size。"""
    height, width = img.shape[:2]
    dx, dy = round(box.w * 0.15), round(box.h * 0.15)
    x1, y1 = max(0, box.x - dx), max(0, box.y - dy)
    x2, y2 = min(width, box.x2 + dx), min(height, box.y2 + dy)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class ClipBuffer:
    """一条轨迹最近的 frames 张裁剪，按 fps 取（比 fps 快的帧丢掉）；中间断档太久就从头攒（片段必须是连续的一段）。"""

    def __init__(self, frames: int, fps: float) -> None:
        period = 1.0 / max(fps, 0.1)
        self.gap = 0.9 * period  # 留一点余量：身体的截图间隔会抖
        self.max_gap = 3 * period  # 超过这么久没新帧（人走开、被挡、感知太慢）：之前攒的作废
        self.items: deque[np.ndarray] = deque(maxlen=frames)
        self.last = float("-inf")

    def push(self, t: float, crop: np.ndarray) -> None:
        if t - self.last < self.gap:
            return
        if t - self.last > self.max_gap:
            self.items.clear()
        self.last = t
        self.items.append(crop)

    def ready(self) -> bool:
        return len(self.items) == self.items.maxlen

    def clip(self) -> list[np.ndarray]:
        return list(self.items)


def eligible(name: str | None, box: Rect, width: int, ref_h: float, near: float, far: float) -> bool:
    """要不要判这个人：认出是哪个好友、离得近 / 中（二期 §4）、在画面中间一半（大致正对着团子）。"""
    if not name:
        return False
    if distance(box.h, ref_h, near, far) == "远":
        return False
    cx = box.x + box.w / 2
    return width / 4 <= cx <= width * 3 / 4


class OnnxGestureClassifier:
    def __init__(self, path: str, labels: list[str], device: str = "cpu") -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"找不到动作识别模型：{path}（[gesture] model；还没训练就先别打开 [gesture]）")
        try:
            import onnxruntime  # noqa: F401 - 缺包时给出提示
        except ImportError as exc:
            raise ImportError("动作识别模型要 onnxruntime：pip install onnxruntime") from exc
        self.session = open_session(path, device, "perception.device")
        self.input = self.session.get_inputs()[0].name
        self.labels = labels
        size = self.session.get_outputs()[0].shape[-1]  # 启动时就对上类别数，别等到运行时每次判都报错
        if isinstance(size, int) and size > 0 and size != len(labels):
            raise ValueError(f"动作模型输出 {size} 类，[gesture] labels 有 {len(labels)} 个：{labels}")

    def classify(self, clip: list[np.ndarray]) -> tuple[str, float]:
        x = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]).astype(np.float32) / 255.0
        x = x.transpose(0, 3, 1, 2)[None]  # 1×T×3×S×S
        scores = np.asarray(self.session.run(None, {self.input: x})[0], np.float64).reshape(-1)
        if len(scores) != len(self.labels):
            raise ValueError(f"动作模型输出 {len(scores)} 类，[gesture] labels 有 {len(self.labels)} 个")
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        i = int(probs.argmax())
        return self.labels[i], float(probs[i])


def load_clip(folder: Path) -> list[np.ndarray]:
    return [imread(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in SUFFIXES]


def extract_clips(
    frames: Iterable[tuple[float, np.ndarray]], detector, out: Path, cfg: GestureConfig, conf: float
) -> int:
    """录像 [(秒, 图)]（可以是生成器，录像长了不用全读进内存）→ 每条 player 轨迹按 fps 取帧，切成不重叠的 frames 张一段存起来，返回段数。"""
    tracker = Tracker(buffer=1.0, min_iou=0.3)
    buffers: dict[int, tuple[float, list[np.ndarray]]] = {}  # 轨迹 → (这一段开始的秒数, 裁剪)
    last: dict[int, float] = {}
    gap = 0.9 / max(cfg.fps, 0.1)
    n = 0
    out.mkdir(parents=True, exist_ok=True)
    for t, img in frames:
        dets: list[Detection] = [d for d in detector.detect(img) if d.cls == "player" and d.score >= conf]
        for track in tracker.update(dets, t):
            if t - last.get(track.id, float("-inf")) < gap:
                continue
            last[track.id] = t
            start, crops = buffers.setdefault(track.id, (t, []))
            crops.append(person_crop(img, track.box, cfg.size))
            if len(crops) < cfg.frames:
                continue
            folder = out / f"{n:04d}_track{track.id}_t{start:.2f}s"
            folder.mkdir(parents=True, exist_ok=True)
            for i, crop in enumerate(crops):
                imwrite(folder / f"{i:02d}.jpg", crop)
            n += 1
            del buffers[track.id]
    return n


def _pr(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


def evaluate(root: Path, classifier: GestureClassifier, cfg: GestureConfig) -> dict:
    """root/<标签>/<片段>/*.jpg 上逐段判：概率不到 min_prob 或判成 none 算没报。返回每个动作和总的 tp / fp / fn / 精确率 / 召回率。"""
    counts: dict[str, list[int]] = {}
    clips = 0
    for label_dir in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")):
        truth = label_dir.name
        for clip_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            clip = load_clip(clip_dir)
            if not clip:
                continue
            clips += 1
            label, prob = classifier.classify(clip)
            said = label if label != "none" and prob >= cfg.min_prob else None
            if truth != "none":
                counts.setdefault(truth, [0, 0, 0])
            if said is not None:
                counts.setdefault(said, [0, 0, 0])
            if said is not None and said == truth:
                counts[said][0] += 1
            else:
                if said is not None:
                    counts[said][1] += 1  # 报错了
                if truth != "none":
                    counts[truth][2] += 1  # 漏了
    out: dict = {label: _pr(*c) for label, c in sorted(counts.items())}
    total = [sum(c[i] for c in counts.values()) for i in range(3)]
    out["all"] = _pr(*total)
    out["clips"] = clips
    return out
