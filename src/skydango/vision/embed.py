"""ONNX 特征模型（被 places、appearance 等模块用）。

特征模型：现成的小型图像编码器（ONNX），或内置的缩略图 + 颜色直方图基线。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .onnxrt import open_session

NORMS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


def unit(v: np.ndarray) -> np.ndarray:
    """把向量归一化成单位向量（欧几里得范数）。"""
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """单位向量的点积（余弦相似度）。维度不同返回 0.0。"""
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


class OnnxEmbedder:
    """ONNX 图像编码器：输入 1×3×S×S（RGB，按 norm 归一化），输出里有二维的取二维的，否则取 [:, 0]（CLS token）。"""

    def __init__(self, path: str, size: int = 224, norm: str = "imagenet", device: str = "cpu", what: str = "places.model") -> None:
        if norm not in NORMS:
            raise ValueError(f"{what} 的 norm 只能是 {' / '.join(NORMS)}：{norm}")
        if not Path(path).exists():
            raise FileNotFoundError(f"找不到特征模型：{path}（{what}；先用 \"thumb\" 基线也行）")
        try:
            import onnxruntime  # noqa: F401 - 缺包时给出提示
        except ImportError as exc:
            raise ImportError(f"{what} 要 onnxruntime：pip install onnxruntime") from exc
        self.session = open_session(path, device, what)
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.size = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else size
        self.mean, self.std = (np.array(v, np.float32) for v in NORMS[norm])
        st = Path(path).stat()  # 缓存键带上完整路径和文件大小 / 修改时间：同名的另一个模型、覆盖过的模型都不能用旧向量
        self.key = f"{Path(path).resolve().as_posix()}:{st.st_size}:{st.st_mtime_ns}:{self.size}:{norm}"

    def embed(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None]
        outs = self.session.run(None, {self.input.name: x})
        flat = next((o for o in outs if o.ndim == 2), None)
        return unit(flat[0] if flat is not None else outs[0][:, 0][0])
