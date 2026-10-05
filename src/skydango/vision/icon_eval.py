"""认交互图标的离线工具（spec 2026-10-05-icon-detection §2.2、§5）：
evaluate = 同一批截图上比较模板法和 DINOv2 最近邻；cut = 截模板 / 参考图。不碰设备。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import IconsConfig, PerceptionConfig
from ..imageio import imread, imwrite
from .icons_map import (MAP_OWNERS, RING_PX, UNKNOWN, IconGallery, classify_template, icon_crop, map_label,
                        owner_of)
from .track import Rect, iou

PEOPLE = ("player", "player_unlit")
THINGS = ("bench", "bonfire", "instrument")
SHEET_MAX = 100  # 一张拼图最多几格
SHEET_COLS = 10
CELL = 120
OWNERS_MAX = 50  # 画归属的整图最多几张


def _sheet(crops: list[tuple[int, np.ndarray]], out: Path, stem: str) -> None:
    """每格一张裁图（等比缩进 CELL×CELL，灰底）+ 左上角编号；一张最多 SHEET_MAX 格，多了分 -2、-3。"""
    for part, start in enumerate(range(0, len(crops), SHEET_MAX), 1):
        chunk = crops[start:start + SHEET_MAX]
        rows = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
        sheet = np.full((rows * CELL, SHEET_COLS * CELL, 3), 40, np.uint8)
        for n, (idx, img) in enumerate(chunk):
            y, x = (n // SHEET_COLS) * CELL, (n % SHEET_COLS) * CELL
            if img.size:
                k = min(CELL / img.shape[0], CELL / img.shape[1])
                small = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))))
                sheet[y:y + small.shape[0], x:x + small.shape[1]] = small
            cv2.putText(sheet, str(idx), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        imwrite(out / (f"{stem}.jpg" if part == 1 else f"{stem}-{part}.jpg"), sheet)


def _pct(vals: list[float], q: float) -> float | None:
    return float(np.percentile(vals, q)) if vals else None


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _recall(images: list[Path], rings_by_image: dict[str, list[Rect]], labels: Path, classes: list[str]) -> float | None:
    """标注里的 social_ring 被检测到的比例（IoU >= 0.5）。"""
    if "social_ring" not in classes:
        return None
    idx = classes.index("social_ring")
    total = hit = 0
    for p in images:
        f = Path(labels) / f"{p.stem}.txt"
        if not f.exists():
            continue
        h, w = imread(p).shape[:2]
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5 or int(float(parts[0])) != idx:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            gt = Rect(round((cx - bw / 2) * w), round((cy - bh / 2) * h), round(bw * w), round(bh * h))
            total += 1
            if any(iou(gt, r) >= 0.5 for r in rings_by_image.get(p.name, [])):
                hit += 1
    return hit / total if total else None


def evaluate(images: list[Path], detect: Callable, template, gallery: IconGallery | None, cfg: IconsConfig,
             out: Path, labels: Path | None = None, classes: list[str] | None = None) -> dict:
    """每张图 detect → 每个 social_ring 判归属；归属在 MAP_OWNERS 里的两种分类器都跑，结果写进 out。
    返回 {"rings", "map", "agree", "recall"}。template / gallery 为 None 就只跑另一种。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    crops: list[np.ndarray] = []
    rings_n = 0
    all_rings: dict[str, list[Rect]] = {}
    drawn = 0
    for path in images:
        frame = imread(path)
        dets = detect(frame)
        people = [d.box for d in dets if d.cls in PEOPLE]
        spirits = [d.box for d in dets if d.cls == "spirit"]
        things = [(d.cls, d.box) for d in dets if d.cls in THINGS]
        rings = [d.box for d in dets if d.cls == "social_ring"]
        all_rings[path.name] = rings
        rings_n += len(rings)
        view = frame.copy() if drawn < OWNERS_MAX and rings else None
        for ring in rings:
            owner = owner_of(ring, people, spirits, things, cfg.under_x, cfg.under_up)
            if view is not None:
                cv2.rectangle(view, (ring.x, ring.y), (ring.x2, ring.y + ring.h), (0, 255, 255), 2)
                cv2.putText(view, owner, (ring.x, max(12, ring.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if owner not in MAP_OWNERS:
                continue
            crop = icon_crop(frame, ring, cfg.crop_scale)
            ok = crop.size > 0
            t_kind, t_score = classify_template(template, crop, ring) if template is not None and ok else (None, None)
            d_kind, d_score = gallery.classify(crop) if gallery is not None and ok else (None, None)
            rows.append({"id": len(rows), "image": path.name, "box": [ring.x, ring.y, ring.w, ring.h], "owner": owner,
                         "template": t_kind, "template_score": None if t_score is None else round(float(t_score), 3),
                         "dino": d_kind, "dino_score": None if d_score is None else round(float(d_score), 3)})
            crops.append(crop)
        if view is not None:
            imwrite(out / "owners" / f"{path.stem}.jpg", view)
            drawn += 1

    with open(out / "crops.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = sorted({r[k] for r in rows for k in ("template", "dino") if r[k] is not None})
    for how in ("template", "dino"):
        for kind in kinds:
            picked = [(r["id"], crops[r["id"]]) for r in rows if r[how] == kind]
            if picked:
                _sheet(picked, out, f"{how}-{kind}")
    both = [r for r in rows if r["template"] is not None and r["dino"] is not None]
    disagree = [(r["id"], crops[r["id"]]) for r in both if r["template"] != r["dino"]]
    if disagree:
        _sheet(disagree, out, "disagree")
    agree = [r for r in both if r["template"] == r["dino"]]
    recall = _recall(images, all_rings, Path(labels), classes or PerceptionConfig().classes) if labels else None

    lines = ["# 图标分类器对比", "",
             f"- 截图 {len(images)} 张，social_ring {rings_n} 个，地图 / 先祖 / 物件上的 {len(rows)} 个",
             f"- 模板法：{'有' if template is not None else '没有（只看 DINOv2）'}；"
             f"DINOv2 底库：{'有' if gallery is not None else '没有（只看模板；底库为空或没建起来）'}",
             f"- 两种都跑了的 {len(both)} 个，结果一致 {len(agree)} 个"]
    if recall is not None:
        lines.append(f"- social_ring 召回（IoU ≥ 0.5）：{recall:.1%}")
    lines += ["", "| 种类 | 叫法 | 模板判出 | DINOv2 判出 |", "| --- | --- | --- | --- |"]
    for kind in kinds:
        lines.append(f"| {kind} | {map_label(kind, 'map')} | {sum(r['template'] == kind for r in rows)} | "
                     f"{sum(r['dino'] == kind for r in rows)} |")
    for how in ("template", "dino"):
        got = [r for r in rows if r[how] is not None]
        unk = sum(r[how] == UNKNOWN for r in got)
        scores = [r[f"{how}_score"] for r in got if r[f"{how}_score"] is not None]
        share = f"{unk / len(got):.1%}" if got else "-"
        lines.append(f"\n{how}：unknown {unk}/{len(got)}（{share}）；分数 5% {_fmt(_pct(scores, 5))}、"
                     f"中位 {_fmt(_pct(scores, 50))}、95% {_fmt(_pct(scores, 95))}")
    sug = _pct([r["dino_score"] for r in agree if r["dino_score"] is not None], 5)
    if sug is not None:
        lines.append(f"\n建议 dino_match：{_fmt(sug)}（两种结果一致的那些里 DINOv2 分数的 5% 分位数；现在 {cfg.dino_match}）")
    else:
        lines.append("\n建议 dino_match：-（没有两种结果一致的样本）")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rings": rings_n, "map": len(rows), "agree": len(agree), "recall": recall}


def cut(image: Path, kind: str, box: Rect, refs: Path, templates: Path | None) -> list[Path]:
    """从截图裁一个图标：存 refs/<kind>/<时间>.jpg；给了 templates 再按 RING_PX 缩放存 templates/<kind>.png
    （已有就 <kind>-2.png、-3…）。返回写出的文件。"""
    frame = imread(image)
    crop = icon_crop(frame, box, IconsConfig().crop_scale)
    if crop.size == 0:
        raise ValueError(f"框 {box} 在画面外")
    wall = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(wall)) + f"-{int(wall * 1000) % 1000:03d}"
    made: list[Path] = []
    ref = Path(refs) / kind / f"{stamp}.jpg"
    imwrite(ref, crop)
    made.append(ref)
    if templates is not None:
        k = RING_PX / max(box.w, box.h)
        small = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        tdir = Path(templates)
        path = tdir / f"{kind}.png"
        n = 2
        while path.exists():
            path = tdir / f"{kind}-{n}.png"
            n += 1
        imwrite(path, small)
        made.append(path)
    return made
