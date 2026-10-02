"""第二层（外形头）的训练和整帧回放评估（`perception attrs-train` / `attrs-eval` / `bench --attrs`）。

训练样本 = `datasets/attrs/form/<外形>/` 里的图（文件夹就是标签）。主干（冻住的 DINOv2）只提特征并缓存，
头是 numpy 全批梯度下降的 softmax 回归，存成 `.npz`（`attrs.save_model`）。不依赖 torch。
回放（`replay`）在带 YOLO 标注的验证帧上比较"纯 YOLO"和"低分框 + 外形头复核"的人物精确率 / 召回率。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..imageio import imread
from . import attrs
from .attrs import FORMS, PERSON_FORMS
from .augment import IMAGE_EXTS
from .bubbles import Rect
from .gesture_train import macro_f1
from .track import iou

SPLIT_FILE = "_split.json"
MIN_PER_CLASS = 20  # 一类至少这么多张才单独成类
L2_CHOICES = (1e-4, 1e-3, 1e-2)
ACCEPT_GRID = (0.5, 0.6, 0.7, 0.8)
REJECT_GRID = (0.6, 0.7, 0.8, 0.9)
MATCH_IOU = 0.4  # 回放里预测框和标注框的 IoU 到这个算对
GT_PERSON = {0: "player", 4: "player_unlit", 9: "spirit"}  # 数据集里算"人"的类别（3 = self 两边都不算）
PERSON_DETS = ("player", "player_unlit")


# ---------- 数据 ----------

def list_samples(root: Path) -> list[tuple[Path, str]]:
    """form/<外形>/ 里的所有图 → [(路径, 外形)]。"""
    out: list[tuple[Path, str]] = []
    for form in FORMS:
        d = Path(root) / "form" / form
        if d.is_dir():
            out += [(p, form) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return out


def crop_rows(root: Path) -> dict[str, dict]:
    path = Path(root) / "_crops.jsonl"
    if not path.is_file():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["crop"]: r for r in rows}


def split_crops(root: Path, seed: int = 0, val_ratio: float = 0.2) -> dict:
    """切训练 / 验证集（只看 form/ 里的图，名字写成 "<外形>/<文件名>"）：
    来自数据集且有 split 的按数据集原来的切分；其余按 group 整组随机分到约 val_ratio（没有 _crops.jsonl 行的图用文件名当 group）。
    结果写 root/_split.json，同 seed 同数据结果相同。"""
    root = Path(root)
    rows = crop_rows(root)
    fixed: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    ds_splits: dict[str, set[str]] = {}  # 数据集来源的录像 group -> 它那些行的 split（防止同一段录像的近似帧跨集）
    for r in rows.values():
        if r.get("source") == "dataset" and r.get("split") in ("train", "val") and r.get("group"):
            ds_splits.setdefault(str(r["group"]), set()).add(r["split"])
    for p, form in list_samples(root):
        name = f"{form}/{p.name}"
        r = rows.get(p.name)
        if r and r.get("source") == "dataset" and r.get("split") in ("train", "val"):
            fixed[name] = r["split"]
        else:
            groups.setdefault(str((r or {}).get("group") or p.stem), []).append(name)
    total = sum(len(v) for v in groups.values())
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    val_groups: set[str] = set()
    got = 0
    for i in rng.permutation(len(keys)):
        if got >= val_ratio * total:
            break
        val_groups.add(keys[i])
        got += len(groups[keys[i]])
    sp = {"train": [], "val": []}
    pinned: dict[str, str] = {}
    for name, which in fixed.items():
        sp[which].append(name)
    for k, names in groups.items():
        if k in ds_splits:  # 和数据集某段录像同名的非数据集裁图：那段录像有帧在验证集 -> 整组进验证集，否则进训练集
            which = "val" if "val" in ds_splits[k] else "train"
            pinned[k] = which
        else:
            which = "val" if k in val_groups else "train"
        sp[which] += names
    sp = {"train": sorted(sp["train"]), "val": sorted(sp["val"]), "pinned": pinned}
    (root / SPLIT_FILE).write_text(json.dumps({**sp, "seed": seed, "val_ratio": val_ratio}, ensure_ascii=False, indent=1), encoding="utf-8")
    return sp


def default_out(models: Path, now) -> Path:
    """默认输出 models/attrs-<日期>.npz；同一天已经有了就带上时分秒。"""
    path = Path(models) / f"attrs-{now:%Y%m%d}.npz"
    return path if not path.exists() else Path(models) / f"attrs-{now:%Y%m%d-%H%M%S}.npz"


def check_out(out: Path, configured: str, force: bool) -> None:
    """最终输出路径就是 [attrs] model 正在用的模型、又没加 --force：拒绝覆盖。"""
    if Path(out).resolve() == Path(configured).resolve() and not force:
        raise SystemExit(f"输出 {out} 就是 [attrs] model 正在用的模型：没评估过的新模型别直接覆盖它。"
                         "换个路径训练、看完报告达标再复制过去；确实要覆盖就加 --force")


def merge_labels(counts: dict[str, int], min_per_class: int) -> tuple[dict[str, str | None], list[str]]:
    """每个外形训练时归到哪一类（None = 丢掉样本）和写进报告的说明。
    shared / morph 不够 → 并进 lit；spirit 不够 → 丢；not_person / lit / unlit 不够 → ValueError。"""
    need = [f for f in ("not_person", "lit", "unlit") if counts.get(f, 0) < min_per_class]
    if need:
        raise ValueError("这几类确认过的图不够（每类至少 %d 张）：%s" % (
            min_per_class, "、".join(f"{f} {counts.get(f, 0)} 张" for f in need)))
    mapping: dict[str, str | None] = {f: f for f in FORMS}
    notes: list[str] = []
    for f in ("shared", "morph"):
        if counts.get(f, 0) < min_per_class:
            mapping[f] = "lit"
            notes.append(f"{f} 只有 {counts.get(f, 0)} 张（< {min_per_class}），并进 lit")
    if counts.get("spirit", 0) < min_per_class:
        mapping["spirit"] = None
        notes.append(f"spirit 只有 {counts.get('spirit', 0)} 张（< {min_per_class}），样本丢掉不训练")
    return mapping, notes


def features(paths: list[Path], embedder, cache: Path, flip: bool) -> np.ndarray:
    """每张图过主干得单位向量，按 embedder.key 分目录缓存成 .npy（flip = 左右镜像，另存一份）。"""
    if not paths:
        return np.zeros((0, 0), np.float32)
    d = Path(cache) / hashlib.sha1(str(embedder.key).encode("utf-8")).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        f = d / f"{p.parent.name}__{p.stem}{'_flip' if flip else ''}.npy"
        if f.exists():
            out.append(np.load(f))
            continue
        img = imread(p)
        if flip:
            img = cv2.flip(img, 1)
        v = np.asarray(embedder.embed(img), np.float32).reshape(-1)
        np.save(f, v)
        out.append(v)
    return np.stack(out).astype(np.float32)


# ---------- 训练 ----------

def _softmax_rows(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train_head(X: np.ndarray, y: np.ndarray, classes: int, l2: float, class_weight: np.ndarray,
               epochs: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """softmax 回归，全批 Adam；按类别加权的交叉熵 + l2。返回 (W D×K, b K)。"""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    n, d = X.shape
    onehot = np.eye(classes)[y]
    sw = np.asarray(class_weight, np.float64)[y]
    sw = sw / sw.mean()
    W, b = np.zeros((d, classes)), np.zeros(classes)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        g = (_softmax_rows(X @ W + b) - onehot) * sw[:, None] / n
        gW, gb = X.T @ g + l2 * W, g.sum(axis=0)
        mW, mb = b1 * mW + (1 - b1) * gW, b1 * mb + (1 - b1) * gb
        vW, vb = b2 * vW + (1 - b2) * gW * gW, b2 * vb + (1 - b2) * gb * gb
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
    return W.astype(np.float32), b.astype(np.float32)


def class_weights(y: np.ndarray, classes: int) -> np.ndarray:
    """每类权重 = 平均每类样本数 / 这一类样本数（没样本的类给 1）。"""
    cnt = np.bincount(np.asarray(y, int), minlength=classes).astype(np.float64)
    return np.where(cnt > 0, len(y) / (classes * np.maximum(cnt, 1)), 1.0).astype(np.float32)


def fit_head(Xtr, ytr, Xva, yva, classes: int, epochs: int = 300) -> tuple[np.ndarray, np.ndarray, float]:
    """l2 从 L2_CHOICES 里按验证集宏平均 F1 挑（没有验证集按训练集；同分取大的 l2）。返回 (W, b, l2)。"""
    cw = class_weights(ytr, classes)
    ex, ey = (Xva, yva) if len(yva) else (Xtr, ytr)
    best = None
    for l2 in L2_CHOICES:
        W, b = train_head(Xtr, ytr, classes, l2, cw, epochs)
        f1 = macro_f1([int(i) for i in ey], [int(i) for i in (ex @ W + b).argmax(1)])
        if best is None or f1 >= best[0]:
            best = (f1, W, b, l2)
    return best[1], best[2], best[3]


def evaluate(truth: list[int], pred: list[int], labels: list[str]) -> dict:
    """每类精确率 / 召回率（没有的类是 None）、混淆矩阵（行 = 真，列 = 预测）、宏平均 F1。"""
    k = len(labels)
    cm = np.zeros((k, k), int)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    per = {}
    for i, lb in enumerate(labels):
        tp, col, row = int(cm[i, i]), int(cm[:, i].sum()), int(cm[i].sum())
        per[lb] = {"precision": tp / col if col else None, "recall": tp / row if row else None, "n": row}
    return {"per_class": per, "confusion": cm.tolist(), "macro_f1": macro_f1(truth, pred) if truth else 0.0,
            "n": len(truth)}


def run_training(root: Path, embedder, cache: Path, min_per_class: int = MIN_PER_CLASS, seed: int = 0) -> dict:
    """切分 → 合并类别 → 特征（训练集加左右镜像）→ 挑 l2 训练 → 验证集评估。
    返回 {head（给 attrs.save_model）, labels, split, counts, notes, l2, eval, on_train}；类别不够抛 ValueError。"""
    root = Path(root)
    samples = list_samples(root)
    counts = {f: sum(1 for _, ff in samples if ff == f) for f in FORMS}
    mapping, notes = merge_labels(counts, min_per_class)
    sp = split_crops(root, seed)
    labels = [f for f in FORMS if f in set(mapping.values())]
    index = {lb: i for i, lb in enumerate(labels)}
    by_name = {f"{form}/{p.name}": (p, mapping[form]) for p, form in samples}

    def pick(names: list[str]) -> tuple[list[Path], np.ndarray]:
        rows = [by_name[n] for n in names if n in by_name and by_name[n][1] is not None]
        return [p for p, _ in rows], np.array([index[lb] for _, lb in rows], int)

    ptr, ytr = pick(sp["train"])
    pva, yva = pick(sp["val"])
    Xtr = np.concatenate([features(ptr, embedder, cache, False), features(ptr, embedder, cache, True)])
    ytr2 = np.concatenate([ytr, ytr])
    Xva = features(pva, embedder, cache, False)
    on_train = not len(yva)
    if on_train:
        notes.append("没有验证集（图太少），指标是训练集上的")
    W, b, l2 = fit_head(Xtr, ytr2, Xva, yva, len(labels))
    ex, ey = (features(ptr, embedder, cache, False), ytr) if on_train else (Xva, yva)
    ev = evaluate(list(map(int, ey)), [int(i) for i in (ex @ W + b).argmax(1)], labels)
    head = {"W": W, "b": b, "labels": labels, "applies_to": list(PERSON_DETS), "pad": attrs.CROP_PAD}
    if sp.get("pinned"):
        notes.append("这些录像在数据集和裁图里都有，按数据集的切分整组定了训练 / 验证（避免近似帧跨集）："
                     + "、".join(f"{g}→{w}" for g, w in sp["pinned"].items()))
    return {"head": head, "labels": labels, "split": sp, "counts": counts, "mapping": mapping, "notes": notes,
            "l2": l2, "eval": ev, "on_train": on_train, "train_n": len(ytr), "val_n": len(yva)}


# ---------- 回放 ----------

def _label_file(frame: Path) -> Path:
    return frame.parents[2] / "labels" / frame.parent.name / f"{frame.stem}.txt"


def _gt_boxes(path: Path, w: int, h: int) -> list[tuple[Rect, int]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) not in GT_PERSON:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((Rect(round((cx - bw / 2) * w), round((cy - bh / 2) * h), round(bw * w), round(bh * h)), int(parts[0])))
    return out


def collect(frames: list[Path], detector, model, conf_low: float) -> list[dict]:
    """每帧跑一次检测（只留分 ≥ conf_low 的人物框）和外形头：[{gt: [(框, 类别号)], dets: [{cls, box, score, p}]}]。
    p = {外形: 概率}，外形头对这个类别没有头时是 None。没有 labels/ 对应文件的帧跳过。"""
    records = []
    labels = model.labels("form")
    for f in frames:
        lf = _label_file(Path(f))
        if not lf.is_file():
            continue
        img = imread(f)
        h, w = img.shape[:2]
        dets = [d for d in detector.detect(img) if d.cls in PERSON_DETS and d.score >= conf_low]
        probs = model.predict([(d.cls, attrs.crop(img, d.box, model.pad("form"), model.size)) for d in dets]) if dets else []
        records.append({"gt": _gt_boxes(lf, w, h), "dets": [
            {"cls": d.cls, "box": d.box, "score": d.score,
             "p": ({lb: float(v) for lb, v in zip(labels, r["form"])} if "form" in r else None)}
            for d, r in zip(dets, probs)]})
    return records


def _ratio(a: int, b: int) -> float:
    return a / b if b else 1.0


def _score(preds: list[dict], gt: list[tuple[Rect, int]]) -> tuple[int, int, int, int, int]:
    """贪心配对（IoU 从大到小，≥ MATCH_IOU）。返回 (tp, fp, fn, 配上且 GT 是亮 / 黑影的对数, 其中点没点火认反的数)。"""
    pairs = sorted(((iou(p["box"], g[0]), i, j) for i, p in enumerate(preds) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, i, j in pairs:
        if v < MATCH_IOU:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j))
    both = bad = 0
    for i, j in matched:
        if gt[j][1] in (0, 4):
            both += 1
            bad += preds[i]["unlit"] != (gt[j][1] == 4)
    return len(matched), len(preds) - len(matched), len(gt) - len(matched), both, bad


def _unlit_second(d: dict) -> bool:
    """两边平均：u = 0.5 × YOLO（player_unlit = 1）+ 0.5 × 外形头里 unlit 占 lit / unlit / shared / morph 的比例 ≥ 0.5。"""
    yolo = 1.0 if d["cls"] == "player_unlit" else 0.0
    p = d["p"]
    den = sum(p.get(k, 0.0) for k in ("lit", "unlit", "shared", "morph")) if p else 0.0
    if den <= 0:
        return yolo >= 0.5
    return 0.5 * yolo + 0.5 * p.get("unlit", 0.0) / den >= 0.5


def simulate(records: list[dict], conf: float, accept: float, reject: float) -> dict:
    """单帧版的复核：基线 = 分 ≥ conf 的框；第二层 = 分 ≥ conf 且外形头没有 ≥ reject 地说不是人，
    加上 conf_low ≤ 分 < conf、外形头判成人形且 ≥ accept 的框（每框当成复核过 reject_n 次同样的结果）。"""
    tot = {"baseline": [0] * 5, "second": [0] * 5}
    gts = 0
    for rec in records:
        gts += len(rec["gt"])
        base, second = [], []
        for d in rec["dets"]:
            p = d["p"]
            if d["score"] >= conf:
                base.append({**d, "unlit": d["cls"] == "player_unlit"})
                if not (p and p.get("not_person", 0.0) >= reject):
                    second.append({**d, "unlit": _unlit_second(d)})
            elif p:
                top = max(p, key=p.get)  # 和运行时 PersonAttrs.admit 一致：最高类要是人形、概率 ≥ accept
                if top in PERSON_FORMS and p[top] >= accept:
                    second.append({**d, "unlit": _unlit_second(d)})
        for key, preds in (("baseline", base), ("second", second)):
            for i, v in enumerate(_score(preds, rec["gt"])):
                tot[key][i] += v
    out: dict = {"frames": len(records), "gt": gts}
    for key, (tp, fp, fn, pairs, mismatch) in tot.items():
        out[key] = {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                    "pairs": pairs, "mismatch": mismatch}
    return out


def replay(frames: list[Path], detector, model, conf_low: float, conf: float, accept: float, reject: float,
           reject_n: int = 3) -> dict:
    """在带 YOLO 标注的验证帧（<数据集>/images/val/*）上比较"纯 YOLO"和"低分框 + 外形头复核"。
    reject_n 在单帧版里没有作用（相当于每框复核了 reject_n 次同样的结果），只记在结果里。"""
    r = simulate(collect(frames, detector, model, conf_low), conf, accept, reject)
    r.update(conf_low=conf_low, conf=conf, accept=accept, reject=reject, reject_n=reject_n)
    return r


def sweep_thresholds(records: list[dict], conf: float) -> tuple[float, float]:
    """accept × reject 网格里：精确率不低于纯 YOLO 的前提下召回最高的一组（同分取阈值大的）；
    没有满足的就取精确率最高的一组。"""
    base = simulate(records, conf, 0.6, 0.7)["baseline"]["precision"]
    scored = []
    for a in ACCEPT_GRID:
        for r in REJECT_GRID:
            s = simulate(records, conf, a, r)["second"]
            scored.append((s["precision"] >= base - 1e-9, s["recall"] if s["precision"] >= base - 1e-9 else s["precision"],
                           s["precision"], a, r))
    best = max(scored)
    return best[3], best[4]


def frames_in(dataset: Path, split: str = "val") -> list[Path]:
    d = Path(dataset) / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


# ---------- 测速 ----------

def bench_attrs(get_frame: Callable[[int], np.ndarray], detector, model, n: int, warmup: int = 5,
                max_crops: int = 4) -> tuple[list[float], list[float]]:
    """每帧：只检测的耗时 / 检测 + 最多 max_crops 张人物裁图过外形头的耗时（毫秒）。前 warmup 帧不计。"""
    plain, withattrs = [], []
    for i in range(n + warmup):
        frame = get_frame(i)
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        people = sorted((d for d in dets if d.cls in PERSON_DETS), key=lambda d: -d.score)[:max_crops]
        if people:
            model.predict([(d.cls, attrs.crop(frame, d.box, model.pad("form"), model.size)) for d in people])
        t2 = time.perf_counter()
        if i >= warmup:
            plain.append((t1 - t0) * 1000)
            withattrs.append((t2 - t0) * 1000)
    return plain, withattrs


# ---------- 报告 ----------

def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def replay_md(r: dict) -> list[str]:
    lines = [f"conf_low = {r['conf_low']:g}、conf = {r['conf']:g}、accept = {r['accept']:g}、reject = {r['reject']:g}"
             f"（{r['frames']} 帧、{r['gt']} 个人物标注）", "",
             "| | 对 | 错报 | 漏 | 精确率 | 召回率 | 点没点火认反 |", "|---|---|---|---|---|---|---|"]
    for key, name in (("baseline", "纯 YOLO（≥ conf）"), ("second", "低分框 + 外形头复核")):
        s = r[key]
        lines.append(f"| {name} | {s['tp']} | {s['fp']} | {s['fn']} | {_pct(s['precision'])} | {_pct(s['recall'])} | "
                     f"{s['mismatch']} / {s['pairs']} |")
    lines.append("注意：单帧回放里「点没点火认反」按 u = 0.5×YOLO + 0.5×外形头 ≥ 0.5 算，YOLO 的 player 框要外形头黑影占比恰好 1.0 才翻、"
                 "player_unlit 框永远保持黑影，所以第二层这一列基本就是 YOLO 自己的答案，看不出外形头对点火判断的帮助（要看真机投票）。")
    return lines + [""]


def report_md(*, data: Path, model: Path | None, when, result: dict | None, replays: list[dict],
              suggest: tuple[float, float] | None, notes: list[str]) -> str:
    L = [f"# 外形头训练报告 {when:%Y-%m-%d %H:%M:%S}", "", f"- 数据：{data}", f"- 模型：{model}" if model else "- 模型：（只评估）", ""]
    if result:
        labels = result["labels"]
        L += ["## 数据", "", "| 外形 | 张数 | 训练时归到 |", "|---|---|---|"]
        for f in FORMS:
            L.append(f"| {f} | {result['counts'][f]} | {result['mapping'][f] or '（丢掉）'} |")
        L += ["", f"训练 {result['train_n']} 张（加左右镜像）、验证 {result['val_n']} 张；l2 = {result['l2']:g}"
              + ("（指标是训练集上的）" if result["on_train"] else ""), ""]
        if result["notes"]:
            L += ["合并说明：", ""] + [f"- {n}" for n in result["notes"]] + [""]
        ev = result["eval"]
        L += [f"## 验证集（宏平均 F1 {ev['macro_f1']:.3f}）", "", "| 类别 | 张数 | 精确率 | 召回率 |", "|---|---|---|---|"]
        for lb in labels:
            c = ev["per_class"][lb]
            L.append(f"| {lb} | {c['n']} | {_pct(c['precision'])} | {_pct(c['recall'])} |")
        L += ["", "混淆矩阵（行 = 真，列 = 预测）：", "", "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for lb, row in zip(labels, ev["confusion"]):
            L.append(f"| {lb} | " + " | ".join(str(v) for v in row) + " |")
        L.append("")
    for r in replays:
        L += [f"## 整帧回放（conf_low = {r['conf_low']:g}）", ""] + replay_md(r)
    if suggest:
        L += ["## 建议阈值", "", f"`accept = {suggest[0]:g}`、`reject = {suggest[1]:g}`"
              "（精确率不低于纯 YOLO 前提下召回最高的一组；单帧回放，真机还要看投票和轨迹的效果）", ""]
    if notes:
        L += ["## 注意", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)
