"""离线数据增强（`perception augment`）：给训练集加转视角的运动模糊、暗场景样本（设计见一期文档 §2.3）。

真实数据优先，增强只用来放大这些样本：只动 images/train（验证集保持真实分布），
新图带后缀（<名字>_blur.jpg / <名字>_dark.jpg），框不变、标注原样复制。
每张图抽不抽中、核多长、压多暗都由 (seed, 文件名) 决定：目录里加了新图，旧图的结果也不会变；已有的不再生成。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..imageio import imread, imwrite

SUFFIXES = ("_blur", "_dark")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def motion_blur(img: np.ndarray, length: int) -> np.ndarray:
    """水平运动模糊（转视角时画面横着拖影）。"""
    kernel = np.full((1, max(1, length)), 1.0 / max(1, length), np.float32)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def darken(img: np.ndarray, gamma: float) -> np.ndarray:
    """gamma > 1 压暗（暮土、禁阁、夜晚）。"""
    table = ((np.arange(256) / 255.0) ** gamma * 255).round().astype(np.uint8)
    return cv2.LUT(img, table)


def augment_dataset(root: Path, seed: int = 0, blur: float = 0.3, dark: float = 0.2) -> dict[str, int]:
    """给 <root>/images/train 加增强图。返回 {"blur": 新生成几张, "dark": 几张, "skipped": 已经有了跳过几张}。"""
    images = Path(root) / "images" / "train"
    labels = Path(root) / "labels" / "train"
    if not images.is_dir():
        raise FileNotFoundError(f"找不到 {images}（先用 perception label 生成数据集）")
    counts = {"blur": 0, "dark": 0, "skipped": 0}
    originals = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith(SUFFIXES))
    for path in originals:
        rng = random.Random(f"{seed}:{path.stem}")
        picks = []
        if rng.random() < blur:
            picks.append(("_blur", lambda img, n=rng.randint(9, 25): motion_blur(img, n)))
        if rng.random() < dark:
            picks.append(("_dark", lambda img, g=rng.uniform(1.8, 2.5): darken(img, g)))
        img = None
        for suffix, fn in picks:
            out = images / f"{path.stem}{suffix}.jpg"
            if out.exists():
                counts["skipped"] += 1
                continue
            if img is None:
                img = imread(path)
            imwrite(out, fn(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            label = labels / f"{path.stem}.txt"
            target = labels / f"{path.stem}{suffix}.txt"
            labels.mkdir(parents=True, exist_ok=True)
            if label.exists():
                shutil.copyfile(label, target)
            else:
                target.write_text("", encoding="utf-8")
            counts[suffix[1:]] += 1
    return counts
