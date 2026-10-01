"""感知层第二层：人物属性。

给 YOLO 的框裁图 → 冻住的 DINOv2 主干（`embed.OnnxEmbedder`）→ 一组线性头（存在 `.npz` 里）。
这个文件现在只有模型格式、裁剪和预测；轨迹上的投票 / 复核在后面加。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..config import AttrsConfig
from .bubbles import Rect

log = logging.getLogger(__name__)

FORMS = ("not_person", "lit", "unlit", "spirit", "shared", "morph")
PERSON_FORMS = FORMS[1:]
CROP_PAD = 0.15  # 裁图时框四周各放这么多倍宽 / 高（训练时写进每个头的 pad）


class Embedder(Protocol):
    size: int

    def embed(self, img: np.ndarray) -> np.ndarray: ...  # 单位向量


def crop(img: np.ndarray, box: Rect, pad: float, size: int) -> np.ndarray:
    """框四周各放 pad 倍宽 / 高，短边补到和长边一样（以框中心为中心），越界填黑，缩放成 size×size。"""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    side = max(box.w * (1 + 2 * pad), box.h * (1 + 2 * pad))
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(1, int(round(side)))
    out = np.zeros((s, s, 3), np.uint8)
    h, w = img.shape[:2]
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2, :3]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return (e / e.sum()).astype(np.float32)


class AttrModel:
    """冻住的主干 + 若干线性头。每个头只对它 applies_to 里的 YOLO 类别算。"""

    def __init__(self, data: dict[str, np.ndarray], embedder: Embedder) -> None:
        self.embedder = embedder
        self.heads: list[str] = [str(h) for h in data["heads"]]
        self._w = {h: np.asarray(data[f"{h}.W"], np.float32) for h in self.heads}
        self._b = {h: np.asarray(data[f"{h}.b"], np.float32) for h in self.heads}
        self._labels = {h: [str(x) for x in data[f"{h}.labels"]] for h in self.heads}
        self._applies = {h: [str(x) for x in data[f"{h}.applies_to"]] for h in self.heads}
        self._pad = {h: float(data[f"{h}.pad"]) for h in self.heads}

    @property
    def size(self) -> int:
        return self.embedder.size

    def labels(self, head: str) -> list[str]:
        return self._labels[head]

    def applies_to(self, head: str) -> list[str]:
        return self._applies[head]

    def pad(self, head: str) -> float:
        return self._pad[head]

    def predict(self, items: list[tuple[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """每项 = (YOLO 类别, 已裁好的图)；返回每项 {头: softmax 概率}，只含 applies_to 里有该类别的头。"""
        out: list[dict[str, np.ndarray]] = []
        for cls, img in items:
            heads = [h for h in self.heads if cls in self._applies[h]]
            res: dict[str, np.ndarray] = {}
            if heads:
                v = np.asarray(self.embedder.embed(img), np.float32).reshape(-1)
                for h in heads:
                    res[h] = _softmax(v @ self._w[h] + self._b[h])
            out.append(res)
        return out


def save_model(path: Path, heads: dict[str, dict], backbone_name: str, backbone_key: str, trained: str) -> None:
    """存成 npz：heads 列表、<头>.W / b / labels / applies_to / pad，加主干文件名、主干键、训练日期。"""
    data: dict[str, np.ndarray] = {
        "heads": np.array(list(heads)),
        "backbone_name": np.array(backbone_name),
        "backbone_key": np.array(backbone_key),
        "trained": np.array(trained),
    }
    for name, h in heads.items():
        data[f"{name}.W"] = np.asarray(h["W"], np.float32)
        data[f"{name}.b"] = np.asarray(h["b"], np.float32)
        data[f"{name}.labels"] = np.array(list(h["labels"]))
        data[f"{name}.applies_to"] = np.array(list(h["applies_to"]))
        data[f"{name}.pad"] = np.array(float(h["pad"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:  # 直接给文件对象：np.savez 不会再往文件名后面补 .npz
        np.savez(f, **data)


def load_model(cfg: AttrsConfig, device: str, embedder: Embedder | None = None) -> AttrModel | None:
    """读 cfg.model；文件没有 / 打不开 / 主干对不上就警告一次、返回 None（调用方当作没开）。"""
    path = Path(cfg.model)
    if not path.exists():
        log.warning("人物属性模型不存在：%s（[attrs] 不启用）", path)
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        name = str(data["backbone_name"])
        key = str(data["backbone_key"])
        heads = [str(h) for h in data["heads"]]
        for h in heads:  # 缺键就在这里暴露
            for part in ("W", "b", "labels", "applies_to", "pad"):
                data[f"{h}.{part}"]
    except Exception as exc:
        log.warning("人物属性模型打不开：%s（%s）", path, exc)
        return None
    if name != Path(cfg.backbone).name:
        log.warning("人物属性模型是用 %s 训练的，现在配的主干是 %s，对不上（[attrs] 不启用）", name, Path(cfg.backbone).name)
        return None
    if embedder is None:
        from .embed import OnnxEmbedder

        try:
            embedder = OnnxEmbedder(cfg.backbone, norm="imagenet", device=device, what="attrs.backbone")
        except Exception as exc:
            log.warning("人物属性主干加载失败：%s（%s）", cfg.backbone, exc)
            return None
    want = f"{embedder.size}:imagenet"
    if key != want:
        log.warning("人物属性模型的主干键 %s 和现在的 %s 对不上（[attrs] 不启用）", key, want)
        return None
    return AttrModel(data, embedder)
