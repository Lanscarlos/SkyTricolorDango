"""认地图（感知层三期 §2）：和参考截图比特征，认出现在在哪张图。

    places/<地名>/*.jpg     图库：每个地方几张截图（不同角度 / 时间），`places add <地名>` 截当前画面存进来
    places/_index.npz       图库每张图的特征向量缓存（按文件、修改时间、模型失效）

- 算特征前把人物、名字标签、圆圈（YOLO 结果）和固定 UI（底部按钮栏、聊天面板）涂成画面平均色，
  免得"有没有人、面板开没开"影响匹配；图库里的图存之前就遮好了
- 特征模型：现成的小型图像编码器（ONNX，候选 MobileCLIP-S0 / DINOv2-small，`places bench` 比了再定）；
  `"thumb"` 是内置的缩略图 + 颜色直方图基线，不用下载模型，只用来对比
- 判定：和图库逐张算余弦相似度，最像的 ≥ place_min、且比第二像的**别的地方**高 ≥ place_margin 才算认出（宁可不说，不能说错）
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import PlacesConfig
from ..imageio import imread, imwrite
from .bubbles import Rect, roi_rect
from .embed import NORMS, OnnxEmbedder as _OnnxEmbedder, unit

log = logging.getLogger(__name__)

INDEX = "_index.npz"
SUFFIXES = (".jpg", ".jpeg", ".png")


def mask_scene(img: np.ndarray, boxes: list[Rect], keep_roi: list[float], hide_rois: list[list[float]]) -> np.ndarray:
    """拷贝一份：keep_roi 外面（底部按钮栏）、hide_rois 里面（聊天面板）、每个框（四周放宽 10%）涂成原图的平均色。"""
    height, width = img.shape[:2]
    mean = img.reshape(-1, img.shape[2]).mean(axis=0).round().astype(img.dtype)
    out = img.copy()
    keep = roi_rect(keep_roi, width, height)
    hole = np.ones((height, width), bool)
    hole[keep.y : keep.y2, keep.x : keep.x2] = False
    for roi in hide_rois:
        r = roi_rect(roi, width, height)
        hole[r.y : r.y2, r.x : r.x2] = True
    for b in boxes:
        dx, dy = round(b.w * 0.1), round(b.h * 0.1)
        hole[max(0, b.y - dy) : b.y2 + dy, max(0, b.x - dx) : b.x2 + dx] = True
    out[hole] = mean
    return out


# 向后兼容：旧代码可以从 places 导入这些
_unit = unit
OnnxEmbedder = _OnnxEmbedder


class ThumbEmbedder:
    """基线：32×18 灰度缩略图（减均值）+ 色相 / 饱和度直方图。不用模型，只拿来和真的特征模型对比。"""

    key = "thumb"

    def embed(self, img: np.ndarray) -> np.ndarray:
        small = cv2.resize(img, (32, 18), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).astype(np.float32)
        return unit(np.concatenate([unit(gray), unit(hist)]))


def make_embedder(cfg: PlacesConfig):
    if cfg.model == "thumb":
        return ThumbEmbedder()
    if Path(cfg.model).suffix.lower() == ".onnx":
        return OnnxEmbedder(cfg.model, cfg.size, cfg.norm, cfg.device, what="places.model")
    raise ValueError(f"places.model 要是 .onnx 文件或者 \"thumb\"：{cfg.model}")


@dataclass(frozen=True)
class PlaceMatch:
    name: str | None  # 认出的地方；None = 不确定，不说
    score: float  # 最像的相似度
    best: str | None  # 最像的是哪儿（没认出时也给，调试用）
    second: float  # 第二像的别的地方的相似度


def decide(scores: list[tuple[str, float]], place_min: float, place_margin: float) -> PlaceMatch:
    """最像的 ≥ place_min、且比第二像的别的地方高 ≥ place_margin 才算认出；图库只有一个地方时只看 place_min。"""
    if not scores:
        return PlaceMatch(None, 0.0, None, 0.0)
    best, score = max(scores, key=lambda s: s[1])
    second = max((s for n, s in scores if n != best), default=-1.0)  # 没有别的地方：余弦相似度的下限
    ok = score >= place_min and score - second >= place_margin
    return PlaceMatch(best if ok else None, float(score), best, float(second))


_BAD_CHARS = set('\\/:*?"<>|')


def check_place_name(name: str) -> None:
    """地名要能当目录名：不能空、不能以 _ / . 开头（图库会跳过）、不能有路径符号和 Windows 不允许的字符。"""
    if not name.strip() or name != name.strip():
        raise ValueError(f"地名不能是空的，前后也别带空格：{name!r}")
    if name[0] in "_." or any(c in _BAD_CHARS or ord(c) < 32 for c in name):
        raise ValueError(f"地名不能以 _ 或 . 开头，也不能带 \\ / : * ? \" < > |：{name!r}")


class PlaceLibrary:
    def __init__(self, root: Path, embedder=None, wall: Callable[[], float] = time.time) -> None:
        """embedder 为 None 时只能 add（存图不用算特征：还没选好特征模型也能先攒图库）。"""
        self.root = Path(root)
        self.embedder = embedder
        self.wall = wall
        self.entries: list[tuple[str, Path]] = []  # (地名, 文件)
        self.vecs = np.zeros((0, 0), np.float32)

    def _files(self) -> list[tuple[str, Path]]:
        if not self.root.is_dir():
            return []
        out = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            out += [(folder.name, f) for f in sorted(folder.iterdir()) if f.suffix.lower() in SUFFIXES]
        return out

    def load(self) -> int:
        """读图库：缓存里有、文件没改、模型一样的直接用；别的重算，再写回缓存。返回张数。"""
        files = self._files()
        cache: dict[str, tuple[int, np.ndarray]] = {}
        index = self.root / INDEX
        if index.exists():
            try:
                data = np.load(index, allow_pickle=False)
                if str(data["key"]) == self.embedder.key:
                    cache = {str(f): (int(m), v) for f, m, v in zip(data["files"], data["mtimes"], data["vecs"])}
            except Exception:
                log.warning("图库缓存读不了，重算：%s", index, exc_info=True)
        rel_names, mtimes, vecs, changed = [], [], [], False
        for _, path in files:
            rel = path.relative_to(self.root).as_posix()
            mtime = path.stat().st_mtime_ns
            hit = cache.get(rel)
            if hit is not None and hit[0] == mtime:
                vec = hit[1]
            else:
                vec = self.embedder.embed(imread(path))
                changed = True
            rel_names.append(rel)
            mtimes.append(mtime)
            vecs.append(np.asarray(vec, np.float32))
        self.entries = files
        self.vecs = np.stack(vecs) if vecs else np.zeros((0, 0), np.float32)
        if files and (changed or len(cache) != len(files)):
            np.savez(index, files=np.array(rel_names), mtimes=np.array(mtimes, np.int64), vecs=self.vecs,
                     key=np.array(self.embedder.key))
        return len(files)

    def places(self) -> list[str]:
        return sorted({name for name, _ in self.entries})

    def scores(self, vec: np.ndarray, exclude: Path | None = None) -> list[tuple[str, float]]:
        """和图库每张图的余弦相似度（向量都归一化过），从高到低。exclude：留一法评估时跳过自己。"""
        if not self.entries:
            return []
        sims = self.vecs @ _unit(vec)
        out = [(name, float(s)) for (name, path), s in zip(self.entries, sims) if exclude is None or path != Path(exclude)]
        return sorted(out, key=lambda x: -x[1])

    def add(self, name: str, img: np.ndarray) -> Path:
        """存一张（已经遮好的）截图到 <root>/<地名>/<时间>.jpg，重新读图库（有特征模型时顺便算好缓存）。"""
        check_place_name(name)
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.wall()))
        path = folder / f"{stamp}.jpg"
        i = 1
        while path.exists():
            path = folder / f"{stamp}-{i}.jpg"
            i += 1
        imwrite(path, img)
        if self.embedder is None:
            self.entries = self._files()
        else:
            self.load()
        return path


class PlaceRecognizer:
    def __init__(self, library: PlaceLibrary, cfg: PlacesConfig, keep_roi: list[float], hide_rois: list[list[float]]) -> None:
        self.library = library
        self.cfg = cfg
        self.keep_roi = keep_roi
        self.hide_rois = hide_rois

    def mask(self, img: np.ndarray, boxes: list[Rect]) -> np.ndarray:
        return mask_scene(img, boxes, self.keep_roi, self.hide_rois)

    def recognize(self, img: np.ndarray, boxes: list[Rect]) -> PlaceMatch:
        if not self.library.entries:
            return PlaceMatch(None, 0.0, None, 0.0)
        vec = self.library.embedder.embed(self.mask(img, boxes))
        return decide(self.library.scores(vec), self.cfg.place_min, self.cfg.place_margin)
