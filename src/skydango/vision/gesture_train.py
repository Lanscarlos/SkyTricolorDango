"""动作识别的训练（`perception gesture-train`）：读确认过的片段、按录像切训练 / 验证集、训练、导出 ONNX、写报告。

片段放在 `<数据集>/<标签>/<片段名>/00.jpg…`，片段名见 `gesture.clip_name`。
- 第一部分（数据、切分）不依赖 torch
- 第二部分（训练、导出）：每帧过冻住的特征提取器（DINOv2-small，`DinoExtractor`）→ 一段 = T×D 的特征（缓存起来，
  另算一份左右镜像），只训练小小的时序头 `TemporalHead`；导出时 提取器 + 时序头 一起进 ONNX，输入输出对上
  `gesture.OnnxGestureClassifier` 的约定（`1×T×3×S×S` RGB 0~1 → `1×类别数`，顺序 = 训练时的 labels）。
  torch / transformers 延迟导入：`TemporalHead`、`Exported`、`DinoExtractor` 第一次用到时才建（模块级 `__getattr__`）
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import random
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import GestureConfig
from .gesture import SUFFIXES, load_clip, recording_of

MIN_PER_CLASS = 20
VAL_MIN_SHARE = 0.1  # 整段进验证集的录像至少占这个类别的这么多
TRAIN_MIN_SHARE = 0.5  # 整段录像进验证集后，训练集至少还留着每个类别的这么多
SPLIT_FILE = "_split.json"
_START = re.compile(r"_t(\d+(?:\.\d+)?)s$")


@dataclass(frozen=True)
class Sample:
    clip: str
    label: str
    recording: str
    start: float
    path: Path


def _start_of(clip: str) -> float:
    m = _START.search(clip)
    return float(m.group(1)) if m else 0.0


def list_samples(root: Path, labels: list[str], frames: int = 16) -> tuple[list[Sample], list[str]]:
    """只看 labels 里的类别目录。帧数不等于 frames 的片段不算样本，名字放进第二个返回值。"""
    samples: list[Sample] = []
    skipped: list[str] = []
    for label in labels:
        folder = Path(root) / label
        if not folder.is_dir():
            continue
        for clip in sorted(p for p in folder.iterdir() if p.is_dir()):
            n = sum(1 for f in clip.iterdir() if f.suffix.lower() in SUFFIXES)
            if n != frames:
                skipped.append(clip.name)
                continue
            samples.append(Sample(clip.name, label, recording_of(clip.name), _start_of(clip.name), clip))
    return samples, skipped


def check_counts(samples: list[Sample], labels: list[str], names: dict[str, str] | None = None) -> list[str]:
    """每类至少 MIN_PER_CLASS 段；不够的给提示。"""
    names = {**GestureConfig().names, "none": "都不是", **(names or {})}
    count = {label: 0 for label in labels}
    for s in samples:
        if s.label in count:
            count[s.label] += 1
    return [
        f"{names.get(label, label)}只有 {n} 段，还差 {MIN_PER_CLASS - n} 段"
        for label, n in count.items() if n < MIN_PER_CLASS
    ]


def split(
    samples: list[Sample], labels: list[str], val_ratio: float = 0.2, gap: float = 2.0, seed: int = 0,
) -> dict[str, list]:
    """切训练 / 验证集：同一段录像整个进一边（防止相邻帧泄漏），同样的 seed 结果一样。

    - 每个有样本的类别：有 ≥ 2 段录像就至少有一段录像进验证集；整段进验证集的录像要份量合适：
      占这个类别 ≥ VAL_MIN_SHARE（别只在验证集里放零星一两段），且它里面每个类别训练集都还留着 ≥ TRAIN_MIN_SHARE。
      挑不出这样的录像，就把这个类别最多的那段录像当成"唯一的录像"按时间切（10-02：鞠躬录像整段进了验证集、
      训练集只剩 2 段鞠躬，召回率 0%）。其余录像随机补到验证集样本数约占 val_ratio（同样守 TRAIN_MIN_SHARE）。
    - 类别只有一段录像（或上一条退回来的）：该录像里这个类别的片段按 start 切——设 n 段按 start 排序，
      c = 第 ceil((1 − val_ratio)·n) 段（至多最后一段）的 start；train = start < c − gap，val = start ≥ c，
      中间 gap 秒内的不用。这段录像里所有类别的片段都按同一个 c 切（几个类别钉在同一段录像上取最小的 c），
      免得同一时刻的画面一边训练一边验证；被 gap 丢掉的段数写进 warnings；这段录像不进整录像验证候选。
    - 进不了验证（或训练）集的类别写进 warnings。
    返回 {"train": [片段名], "val": [片段名], "warnings": [...], "seed": seed}。
    """
    rng = random.Random(seed)
    samples = [s for s in samples if s.label in labels]
    key = lambda s: s.recording or s.clip  # 老格式没有录像名：每段自成一组
    by_rec: dict[str, list[Sample]] = {}
    for s in samples:
        by_rec.setdefault(key(s), []).append(s)
    recs_of: dict[str, set[str]] = {}
    for s in samples:
        recs_of.setdefault(s.label, set()).add(key(s))
    total: dict[str, int] = {}
    count: dict[tuple[str, str], int] = {}  # (录像, 类别) → 段数
    for s in samples:
        total[s.label] = total.get(s.label, 0) + 1
        count[key(s), s.label] = count.get((key(s), s.label), 0) + 1
    by_time = {lb for lb, r in recs_of.items() if len(r) == 1}  # 按时间切的类别
    pinned = {next(iter(recs_of[lb])) for lb in by_time}  # 它们的录像：不整段进验证集
    order = sorted(by_rec)
    rng.shuffle(order)
    val_recs: set[str] = set()

    def train_ok(r: str) -> bool:
        """r 整段进验证集后，它里面每个（不按时间切的）类别训练集都还留着 ≥ TRAIN_MIN_SHARE。"""
        gone = val_recs | {r}
        return all(
            sum(count.get((x, lb), 0) for x in recs_of[lb] - gone) >= TRAIN_MIN_SHARE * total[lb]
            for lb in {s.label for s in by_rec[r]} if lb not in by_time)

    # 覆盖：录像少的类别先挑
    for label in sorted(recs_of, key=lambda lb: (len(recs_of[lb]), lb)):
        if label in by_time or recs_of[label] & val_recs:
            continue
        for r in order:
            if (r in recs_of[label] and r not in pinned and count[r, label] >= VAL_MIN_SHARE * total[label]
                    and train_ok(r)):
                val_recs.add(r)
                break
        else:  # 没有份量合适的录像：最多的那段按时间切
            by_time.add(label)
            pinned.add(max(sorted(recs_of[label]), key=lambda x: count[x, label]))
    # 补到比例
    target = val_ratio * len(samples)
    got = sum(len(by_rec[r]) for r in val_recs)
    for r in order:
        if got >= target:
            break
        if r in val_recs or r in pinned:
            continue
        if train_ok(r):
            val_recs.add(r)
            got += len(by_rec[r])

    train: list[str] = []
    val: list[str] = []
    cut: dict[str, float] = {}  # 按时间切的类别所在的录像 → 切点 c（几个类别钉在同一段录像上取最小的）
    for label in sorted(by_time):
        rec = next(r for r in sorted(recs_of[label]) if r in pinned)
        starts = sorted(s.start for s in samples if s.label == label and key(s) == rec)
        if len(starts) >= 2:
            c = starts[min(math.ceil((1 - val_ratio) * len(starts)), len(starts) - 1)]
            cut[rec] = min(c, cut.get(rec, c))
    dropped: dict[str, int] = {}
    for s in samples:
        rec = key(s)
        if rec in val_recs:
            val.append(s.clip)
        elif rec in cut:  # 这段录像里所有类别的片段都按同一个时间切，免得相邻画面一边训练一边验证
            if s.start < cut[rec] - gap:
                train.append(s.clip)
            elif s.start >= cut[rec]:
                val.append(s.clip)
            else:
                dropped[rec] = dropped.get(rec, 0) + 1
        else:
            train.append(s.clip)
    in_val = set(val)
    in_train = set(train)
    warnings = [f"录像 {r or '(无名)'} 按时间切，切点前后 {gap:g} 秒内丢掉 {n} 段" for r, n in sorted(dropped.items())]
    for label in labels:
        mine = [s.clip for s in samples if s.label == label]
        if not mine:
            continue
        if not in_val.intersection(mine):
            warnings.append(f"{label} 进不了验证集（录像或片段太少），评估时看不到它")
        if not in_train.intersection(mine):
            warnings.append(f"{label} 进不了训练集")
    return {"train": train, "val": val, "warnings": warnings, "seed": seed}


def save_split(root: Path, result: dict, seed: int | None = None) -> None:
    data = dict(result)
    if seed is not None:
        data["seed"] = seed
    (Path(root) / SPLIT_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_split(root: Path) -> dict | None:
    path = Path(root) / SPLIT_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---- 第二部分：特征、训练、导出、报告（torch 延迟导入）----
DINO_NAME = "facebook/dinov2-small"
DINO_KEY = "dinov2-small-v1"  # 模型 + 预处理的名字（换了就改）；缓存目录还要带上尺寸，见 cache_key
PATIENCE = 10  # 验证集宏平均 F1 这么多轮没变好就停
_MEAN = (0.485, 0.456, 0.406)  # ImageNet 归一化（DINOv2 训练时用的）
_STD = (0.229, 0.224, 0.225)
_LAZY = ("TemporalHead", "Exported", "DinoExtractor")
_built: dict[str, type] = {}


class Extractor(Protocol):
    """冻住的逐帧特征提取器（实现都是 torch.nn.Module，导出时和时序头一起进 ONNX）：
    `T×3×S×S`（RGB 0~1）→ `T×D`。key = 模型 + 预处理的名字，特征缓存目录是 `<key>-<尺寸>`（见 cache_key）。"""

    key: str

    def __call__(self, frames: Any) -> Any: ...


def _nn(name: str) -> type:
    """torch 的那几个类：第一次用到时才 import torch、建出来。"""
    if not _built:
        _built.update(_build())
    return _built[name]


def __getattr__(name: str):  # `from gesture_train import TemporalHead` 也走这里
    if name in _LAZY:
        return _nn(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build() -> dict[str, type]:
    import torch
    from torch import nn

    class TemporalHead(nn.Module):
        """B×T×D → B×类别数：Conv1d(D→128, k3) → ReLU → Conv1d(128→128, k3) → ReLU → 时间上 max + mean 拼接 → Linear(256→类别数)。"""

        def __init__(self, dim: int, classes: int, hidden: int = 128) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(dim, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            )
            self.fc = nn.Linear(2 * hidden, classes)

        def forward(self, x):
            h = self.conv(x.transpose(1, 2))  # B×hidden×T
            return self.fc(torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1))

    class Exported(nn.Module):
        """导出用：`1×T×3×S×S`（RGB 0~1）→ 逐帧提特征 → 时序头 → `1×类别数`（对上 OnnxGestureClassifier）。"""

        def __init__(self, extractor: nn.Module, head: nn.Module) -> None:
            super().__init__()
            self.extractor = extractor
            self.head = head

        def forward(self, clip):
            b, t = clip.shape[0], clip.shape[1]
            feats = self.extractor(clip.reshape(b * t, *clip.shape[2:]))
            return self.head(feats.reshape(b, t, -1))

    class DinoExtractor(nn.Module):
        """冻住的 DINOv2-small：内部做 ImageNet 归一化，每帧取 CLS + 各块平均拼成 768 维。

        模型从 HF_HOME 读（在仓库里时 .pth 已指到仓库的 .cache/huggingface；没设就用 <当前目录>/.cache/huggingface）。"""

        key = DINO_KEY

        def __init__(self, device: str = "cuda", name: str = DINO_NAME) -> None:
            super().__init__()
            os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
            try:
                from transformers import Dinov2Model
            except ImportError as exc:
                raise ImportError("训练动作模型要 transformers：装进 .pydeps（见 CLAUDE.md「环境」）") from exc
            self.model = Dinov2Model.from_pretrained(name).float()
            self.model.requires_grad_(False)
            self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
            self.to(device).eval()

        def forward(self, frames):
            hidden = self.model(pixel_values=(frames - self.mean) / self.std).last_hidden_state
            return torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)

    return {"TemporalHead": TemporalHead, "Exported": Exported, "DinoExtractor": DinoExtractor}


def clip_array(frames: list[np.ndarray], size: int, flip: bool = False) -> np.ndarray:
    """load_clip 读出的 BGR 帧 → `T×3×S×S` float32（RGB 0~1），和 OnnxGestureClassifier.classify 的预处理一样；
    尺寸不对先缩放（切片段时已经是 size），flip = 左右镜像。"""
    out = []
    for f in frames:
        if f.shape[:2] != (size, size):
            f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        if flip:
            f = np.ascontiguousarray(f[:, ::-1])
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.ascontiguousarray((np.stack(out).astype(np.float32) / 255.0).transpose(0, 3, 1, 2))


def _device_of(module) -> Any:
    import torch

    for t in (*module.parameters(), *module.buffers()):
        return t.device
    return torch.device("cpu")


def cache_key(extractor: Extractor, size: int) -> str:
    """特征缓存的目录名：模型 + 预处理（extractor.key）+ 输入尺寸。是否镜像在文件名里（`_flip`）。"""
    return f"{extractor.key}-{size}"


def features_for(sample: Sample, extractor: Extractor, cache_dir: Path, flip: bool, size: int = 112) -> np.ndarray:
    """一段的特征 `T×D`，缓存在 `<cache_dir>/<key>-<size>/<片段>[_flip].npy`（有就直接读，读坏了重算）。
    目录带尺寸：改了 `[gesture] size` 不会拿旧尺寸提的特征来训练。"""
    import torch

    path = Path(cache_dir) / cache_key(extractor, size) / f"{sample.clip}{'_flip' if flip else ''}.npy"
    if path.exists():
        try:
            return np.load(path)
        except (OSError, ValueError):
            pass
    x = torch.from_numpy(clip_array(load_clip(sample.path), size, flip)).to(_device_of(extractor))
    with torch.no_grad():
        feats = extractor(x).float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npy")
    np.save(tmp, feats)
    os.replace(tmp, path)  # 写到一半被打断不留坏文件
    return feats


def macro_f1(truth: list[int], pred: list[int]) -> float:
    """宏平均 F1：只算 truth 里出现过的类别。"""
    scores = []
    for c in sorted(set(truth)):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def train(
    samples: list[Sample], split: dict, labels: list[str], extractor: Extractor, cache_dir: Path,
    epochs: int = 60, seed: int = 0, device: str = "cuda", *, size: int = 112, batch: int = 32,
    lr: float = 1e-3, patience: int = PATIENCE, progress: Callable[[int, float, float], None] | None = None,
) -> tuple[Any, dict]:
    """在缓存的特征上训练时序头（训练集原样 + 左右镜像，验证集只用原样）：按类别数量加权交叉熵、AdamW，
    按验证集宏平均 F1 早停（patience 轮没变好就停，留最好的那轮）。没有验证集时按训练集 F1。

    返回 (时序头（CPU、eval）, info)；info：loss / val_f1（每轮）、best_epoch（从 1 数）、best_f1、metric（val / train）、
    train / val（段数）、dim、key。"""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    index = {label: i for i, label in enumerate(labels)}
    by_name = {s.clip: s for s in samples if s.label in index}
    tr = [by_name[c] for c in split["train"] if c in by_name]
    va = [by_name[c] for c in split["val"] if c in by_name]
    if not tr:
        raise ValueError("训练集是空的")

    def stack(items: list[Sample], flips: tuple[bool, ...]):
        xs = [features_for(s, extractor, cache_dir, f, size) for s in items for f in flips]
        ys = [index[s.label] for s in items for _ in flips]
        return torch.from_numpy(np.stack(xs)).float(), torch.tensor(ys)

    xt, yt = stack(tr, (False, True))
    xv, yv = stack(va, (False,)) if va else (xt, yt)
    metric = "val" if va else "train"
    dim = xt.shape[-1]
    count = torch.bincount(yt, minlength=len(labels)).float()
    present = int((count > 0).sum())
    weight = torch.where(count > 0, count.sum() / (present * count.clamp(min=1)), torch.zeros_like(count))

    head = _nn("TemporalHead")(dim, len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device))
    gen = torch.Generator().manual_seed(seed)
    xt, yt, xv_d = xt.to(device), yt.to(device), xv.to(device)
    info: dict = {"loss": [], "val_f1": [], "best_epoch": 0, "best_f1": -1.0, "metric": metric,
                  "train": len(tr), "val": len(va), "dim": int(dim), "key": cache_key(extractor, size)}
    best_state = None
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for idx in torch.randperm(len(yt), generator=gen).split(batch):
            idx = idx.to(device)
            opt.zero_grad()
            loss = loss_fn(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            pred = head(xv_d).argmax(dim=1).cpu().tolist()
        f1 = macro_f1(yv.tolist(), pred)
        info["loss"].append(round(total / len(yt), 5))
        info["val_f1"].append(round(f1, 4))
        if progress:
            progress(epoch, total / len(yt), f1)
        if f1 > info["best_f1"]:
            info["best_f1"], info["best_epoch"] = f1, epoch
            best_state = copy.deepcopy(head.state_dict())
        elif epoch - info["best_epoch"] >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head.cpu().eval(), info


def export_onnx(extractor: Extractor, head: Any, path: Path, frames: int = 16, size: int = 112) -> None:
    """提取器 + 时序头 导出成一个 ONNX（opset 17）：输入 `clip` 1×frames×3×size×size（RGB 0~1）→ 输出 `logits` 1×类别数。
    会把 extractor / head 挪到 CPU 上。"""
    import torch

    model = _nn("Exported")(extractor, head).cpu().eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, frames, 3, size, size)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # 旧的 TorchScript 导出器：DINOv2 用它导出没问题
        torch.onnx.export(model, (dummy,), str(path), input_names=["clip"], output_names=["logits"],
                          opset_version=17, dynamo=False)


def onnx_diff(extractor: Extractor, head: Any, path: Path, clips: list[np.ndarray]) -> float:
    """同样几段（`T×3×S×S`，见 clip_array）分别过 ONNX（onnxruntime CPU）和 PyTorch，返回输出的最大差。"""
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    model = _nn("Exported")(extractor, head).cpu().eval()
    worst = 0.0
    for x in clips:
        got = sess.run(None, {name: x[None]})[0]
        with torch.no_grad():
            want = model(torch.from_numpy(x[None])).numpy()
        worst = max(worst, float(np.abs(got - want).max()))
    return worst


def default_out(models: Path, now: dt.datetime) -> Path:
    """默认输出 models/gesture-<日期>.onnx（永远不是 gesture.onnx）；同一天已经有了就带上时分秒。"""
    path = Path(models) / f"gesture-{now:%Y%m%d}.onnx"
    return path if not path.exists() else Path(models) / f"gesture-{now:%Y%m%d-%H%M%S}.onnx"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def report_md(
    *, data: Path, model: Path, labels: list[str], names: dict[str, str], samples: list[Sample], split: dict,
    info: dict, evaluation: dict | None, min_prob: float, skipped: list[str] = (), parity: float | None = None,
    notes: list[str] = (), when: dt.datetime | None = None,
) -> str:
    """训练报告（Markdown）：类别顺序、每类数量、切分、训练曲线要点、验证集每类精确率 / 召回率、认错的片段。"""
    names = {"none": "都不是", **names}
    when = when or dt.datetime.now()
    in_train, in_val = set(split.get("train", [])), set(split.get("val", []))
    lines = [
        f"# 动作识别训练 {when:%Y-%m-%d %H:%M}",
        "",
        f"- 数据：`{data}`；模型：`{model}`（特征 {info.get('key', '?')} 冻住 + 时序头）",
        f"- **类别顺序**（= 模型输出顺序，`[gesture] labels` 必须一模一样）：{', '.join(labels)}",
        f"- 换上之前：看下面验证集达不达标（精确率 ≥ 90%、召回率 ≥ 60%），再复制成 `[gesture] model` 指的文件、打开 `[gesture] enabled`",
    ]
    if skipped:
        lines.append(f"- 跳过 {len(skipped)} 段帧数不对的片段：{'、'.join(skipped)}")
    for n in notes:
        lines.append(f"- 注意：{n}")
    lines += ["", "## 每类片段数", "", "| 类别 | 中文 | 训练 | 验证 | 没用上 | 合计 |", "|---|---|---|---|---|---|"]
    for label in labels:
        mine = [s.clip for s in samples if s.label == label]
        t = sum(1 for c in mine if c in in_train)
        v = sum(1 for c in mine if c in in_val)
        lines.append(f"| {label} | {names.get(label, label)} | {t} | {v} | {len(mine) - t - v} | {len(mine)} |")
    lines += ["", "## 切分", "",
              f"按录像切（同一段录像整个进一边；某类只有一段录像就按时间切），seed {split.get('seed', 0)}："
              f"训练 {len(in_train)} 段、验证 {len(in_val)} 段，写在 `{SPLIT_FILE}`"]
    lines += [f"- 警告：{w}" for w in split.get("warnings", [])]
    loss, f1 = info.get("loss", []), info.get("val_f1", [])
    which = "验证集" if info.get("metric", "val") == "val" else "训练集（没有验证集）"
    lines += ["", "## 训练", "",
              f"训练 {info.get('train', 0)} 段（加镜像 ×2）、验证 {info.get('val', 0)} 段；跑了 {len(loss)} 轮，"
              f"最好第 {info.get('best_epoch', 0)} 轮：{which}宏平均 F1 {info.get('best_f1', 0):.3f}"]
    if loss:
        step = max(1, len(loss) // 10)
        shown = sorted({*range(0, len(loss), step), len(loss) - 1, max(info.get("best_epoch", 1) - 1, 0)})
        lines += ["", "| 轮 | 训练损失 | 宏平均 F1 |", "|---|---|---|"]
        lines += [f"| {i + 1}{' ★' if i + 1 == info.get('best_epoch') else ''} | {loss[i]:.4f} | {f1[i]:.3f} |"
                  for i in shown if i < len(f1)]
    if parity is not None:
        lines += ["", f"ONNX 和 PyTorch 在验证集前几段上的输出最大差：{parity:.2e}"
                  + ("（> 1e-3，导出可能有问题）" if parity > 1e-3 else "")]
    lines += ["", "## 验证集评估", ""]
    if evaluation is None:
        lines.append("没评估（没有验证集）")
    else:
        lines += [f"{evaluation['clips']} 段，概率 ≥ {min_prob} 才算报了：", "",
                  "| 动作 | 中文 | 精确率 | 召回率 | 对 | 错报 | 漏 |", "|---|---|---|---|---|---|---|"]
        for label, v in evaluation.items():
            if label in ("all", "clips", "wrong"):
                continue
            lines.append(f"| {label} | {names.get(label, label)} | {_pct(v['precision'])} | {_pct(v['recall'])} "
                         f"| {v['tp']} | {v['fp']} | {v['fn']} |")
        a = evaluation["all"]
        ok = (a["precision"] or 0) >= 0.9 and (a["recall"] or 0) >= 0.6
        lines += ["", f"总的：精确率 {_pct(a['precision'])}、召回率 {_pct(a['recall'])} → "
                  + ("达标（精确率 ≥ 90%、召回率 ≥ 60%）" if ok else "没达标")]
        wrong = evaluation.get("wrong", [])
        lines += ["", f"### 认错的片段（{len(wrong)}）", ""]
        lines += [f"- `{w['clip']}`：其实是 {w['truth']}，模型说 {w['said']}（{w['prob']:.2f}"
                  + ("，没到 min_prob）" if w["said"] == w["truth"] else "）") for w in wrong] or ["没有"]
    return "\n".join(lines) + "\n"
