"""认地图交互图标（spec 2026-10-05-icon-detection §3.2–3.6）：归属、叫法、投票、裁图的纯计算，DINOv2 底库，
运行时 MapIcons（感知层每帧对先祖 / 物件 / 地图上的圈调一次）。不碰设备。"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import IconsConfig
from ..game.social import KIND_NAMES, IconClassifier
from ..imageio import imread, imwrite
from .embed import cosine
from .track import Rect, Track

log = logging.getLogger(__name__)
SAVE_GAP = 2.0  # 同一条轨迹两张裁图至少隔几秒
SAVE_UNSURE = 0.1  # 分数不到 门槛 + 这么多 也存（拿不准的）

# 圈的归属（好友在调用方用名字标签先判，不进这里）
MAP_OWNERS: tuple[str, ...] = ("spirit", "bonfire", "bench", "instrument", "map")
UNKNOWN = "unknown"
TEMPLATE_SCALES = (0.9, 1.0, 1.1)  # 模板匹配的缩放档（裁图先按框缩到 RING_PX，再微调）
RING_PX = 100  # 模板按近处 1080p 圆圈直径截

_OVERRIDES = {("candle", "bonfire"): "篝火点燃", ("candle", "map"): "可以点的蜡烛 / 灯"}
_UNKNOWN_LABEL = "不认识的图标"


@dataclass(frozen=True)
class Icon:
    track_id: int
    kind: str  # 种类（UNKNOWN = 没认出）
    owner: str  # person / spirit / 物件类别名 / map
    label: str
    box: Rect
    side: str  # 左边 / 中间 / 右边……


def under_gap(ring: Rect, box: Rect, under_x: float, under_up: float) -> float | None:
    """圈在框的头顶上方范围内时返回圈心到框顶的竖直距离，否则 None。

    横着：圈心在框左右各放宽 under_x 倍框宽内；竖着：圈心在框顶往上 under_up 倍框高到框中线之间。"""
    cx, cy = ring.x + ring.w / 2, ring.y + ring.h / 2
    if not (box.x - under_x * box.w <= cx <= box.x2 + under_x * box.w):
        return None
    if not (box.y - under_up * box.h <= cy <= box.y + 0.5 * box.h):
        return None
    return abs(cy - box.y)


def _any_under(ring: Rect, boxes: list[Rect], under_x: float, under_up: float) -> bool:
    return any(under_gap(ring, b, under_x, under_up) is not None for b in boxes)


def owner_of(ring: Rect, people: list[Rect], spirits: list[Rect], things: list[tuple[str, Rect]],
             under_x: float, under_up: float) -> str:
    """圈属于谁：person > spirit > 物件（最近的那个类别）> map。"""
    if _any_under(ring, people, under_x, under_up):
        return "person"
    if _any_under(ring, spirits, under_x, under_up):
        return "spirit"
    best: tuple[float, str] | None = None
    for name, box in things:
        gap = under_gap(ring, box, under_x, under_up)
        if gap is not None and (best is None or gap < best[0]):
            best = (gap, name)
    return best[1] if best else "map"


def map_label(kind: str | None, owner: str) -> str:
    if kind is None or kind == UNKNOWN:
        return _UNKNOWN_LABEL
    if (kind, owner) in _OVERRIDES:
        return _OVERRIDES[(kind, owner)]
    return KIND_NAMES.get(kind, kind)


def vote(history: list[str]) -> str | None:
    """多数票；UNKNOWN 不计票，平票取最近一次出现的；全是 UNKNOWN 返回 UNKNOWN，空返回 None。"""
    if not history:
        return None
    counts: dict[str, int] = {}
    last: dict[str, int] = {}
    for i, k in enumerate(history):
        if k == UNKNOWN:
            continue
        counts[k] = counts.get(k, 0) + 1
        last[k] = i
    if not counts:
        return UNKNOWN
    return max(counts, key=lambda k: (counts[k], last[k]))


def icon_crop(frame: np.ndarray, box: Rect, scale: float) -> np.ndarray:
    """以框中心裁 scale × 长边的正方形，超出画面的部分去掉；完全在外面返回空数组。"""
    side = max(box.w, box.h) * scale
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    h, w = frame.shape[:2]
    x1, x2 = max(0, round(cx - side / 2)), min(w, round(cx + side / 2))
    y1, y2 = max(0, round(cy - side / 2)), min(h, round(cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def describe_icons(icons: list[Icon]) -> str:
    """"坐下（右边）、留影（左边）、不认识的图标 1 个"：认得的按顺序，不认识的合成一项放最后。"""
    known = [f"{i.label}（{i.side}）" for i in icons if i.kind != UNKNOWN]
    unknown = sum(1 for i in icons if i.kind == UNKNOWN)
    if unknown:
        known.append(f"{_UNKNOWN_LABEL} {unknown} 个")
    return "、".join(known)


def classify_template(icons: IconClassifier, crop: np.ndarray, box: Rect) -> tuple[str, float]:
    """模板法认种类：裁图按 RING_PX / 框长边 缩放后交 icons.classify；认不出 / 空裁图返回 (UNKNOWN, 分数)。"""
    long_edge = max(box.w, box.h)
    if crop.size == 0 or long_edge <= 0:
        return UNKNOWN, 0.0
    k = RING_PX / long_edge
    region = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    if region.size == 0:
        return UNKNOWN, 0.0
    kind, score = icons.classify(region, TEMPLATE_SCALES)
    return (kind if kind is not None else UNKNOWN), score


class IconGallery:
    """DINOv2 最近邻底库：每种图标若干参考图，认图标取和参考图余弦最大的那一种。"""

    def __init__(self, embedder, refs: dict[str, list[np.ndarray]], match: float, margin: float) -> None:
        self.embedder = embedder
        self.match = match
        self.margin = margin
        self._feats: dict[str, list[np.ndarray]] = {}
        for kind, imgs in refs.items():
            feats = [f for f in (embedder.embed(img) for img in imgs) if f is not None]  # 参考图特征只提一次
            if feats:
                self._feats[kind] = feats

    @property
    def kinds(self) -> list[str]:
        return list(self._feats)

    @classmethod
    def load(cls, directory: str | Path, embedder, match: float, margin: float) -> "IconGallery":
        """<directory>/<kind>/*.jpg；_ 开头的目录跳过，没有图的目录不算一种。"""
        refs: dict[str, list[np.ndarray]] = {}
        root = Path(directory)
        if root.is_dir():
            for sub in sorted(root.iterdir()):
                if not sub.is_dir() or sub.name.startswith("_"):
                    continue
                imgs = [imread(p) for p in sorted(sub.glob("*.jpg"))]
                if imgs:
                    refs[sub.name] = imgs
        return cls(embedder, refs, match, margin)

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """每种取最像的一张；最好的 >= match 且（只有一种 或 领先第二种 >= margin）才算，否则 (UNKNOWN, 最好分)。"""
        feat = self.embedder.embed(crop)
        if feat is None or not self._feats:
            return UNKNOWN, 0.0
        scores = {kind: max(cosine(feat, f) for f in feats) for kind, feats in self._feats.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, score = ranked[0]
        if score < self.match:
            return UNKNOWN, score
        if len(ranked) > 1 and score - ranked[1][1] < self.margin:
            return UNKNOWN, score
        return best, score


class MapIcons:
    """运行时：先祖 / 物件 / 地图上的圈每帧认一次种类，按轨迹投票（spec §3.2、§3.5）；拿不准 / 认不出的裁图存盘攒参考图。

    classifier = "dino" 且有底库用底库，否则模板；两个都没有一律 UNKNOWN。分类出错只记 DEBUG、当 UNKNOWN（不拖垮感知线程）。"""

    def __init__(self, cfg: IconsConfig, template: IconClassifier | None, gallery: IconGallery | None,
                 save_dir: Path | None) -> None:
        self.cfg = cfg
        self.template = template
        self.gallery = gallery
        self.save_dir = save_dir
        self.saved = 0  # 这次运行存了几张

    @property
    def classifier(self) -> str:
        if self.cfg.classifier == "dino" and self.gallery is not None:
            return "dino"
        return "template" if self.template is not None else "none"

    def _threshold(self, how: str) -> float:
        if how == "dino":
            return self.gallery.match
        if how == "template":
            return getattr(self.template, "min_score", 0.75)
        return 0.0

    def _classify(self, crop: np.ndarray, box: Rect, how: str) -> tuple[str, float]:
        try:
            if how == "dino":
                return self.gallery.classify(crop) if crop.size else (UNKNOWN, 0.0)
            if how == "template":
                return classify_template(self.template, crop, box)
        except Exception:
            log.debug("认图标出错（%s）", how, exc_info=True)
        return UNKNOWN, 0.0

    def observe(self, frame: np.ndarray, ring: Track, owner: str, now: float) -> bool:
        """认这一帧、投票，结果写进 ring.data（votes / icon / icon_score）；
        返回 True = 这条轨迹第一次投出 UNKNOWN 且已经连续 min_hits 帧（调用方报难例）。"""
        crop = icon_crop(frame, ring.box, self.cfg.crop_scale)
        how = self.classifier
        kind, score = self._classify(crop, ring.box, how)
        votes = ring.data.setdefault("votes", [])
        votes.append(kind)
        del votes[: max(0, len(votes) - max(1, self.cfg.vote))]
        ring.data["icon"] = vote(votes)
        ring.data["icon_score"] = score
        if kind == UNKNOWN or score < self._threshold(how) + SAVE_UNSURE:
            self._save(crop, ring, kind, score, how, owner, now)
        if ring.data["icon"] == UNKNOWN and ring.hits >= self.cfg.min_hits and not ring.data.get("icon_unknown_told"):
            ring.data["icon_unknown_told"] = True
            return True
        return False

    def _save(self, crop: np.ndarray, ring: Track, kind: str, score: float, how: str, owner: str, now: float) -> None:
        if not self.cfg.save or self.save_dir is None or crop.size == 0 or self.saved >= self.cfg.save_max:
            return
        if now - ring.data.get("icon_saved_at", float("-inf")) < SAVE_GAP:
            return
        ring.data["icon_saved_at"] = now
        wall = time.time()
        stamp = time.strftime("%H%M%S", time.localtime(wall)) + f"-{int(wall * 1000) % 1000:03d}"
        rel = f"{kind}/{stamp}-t{ring.id}.jpg"
        b = ring.box
        line = {"t": round(now, 3), "track": ring.id, "kind": kind, "score": round(float(score), 3),
                "classifier": how, "owner": owner, "box": [b.x, b.y, b.w, b.h], "file": rel}
        try:
            imwrite(self.save_dir / rel, crop)
            with open(self.save_dir / "icons.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.saved += 1
        except Exception:
            log.debug("存图标裁图出错", exc_info=True)
