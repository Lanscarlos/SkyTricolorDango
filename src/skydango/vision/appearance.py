"""认装扮：人物框裁图、颜色特征（认人靠外观，名字标签看不到时用）。"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field

from pathlib import Path

import cv2
import numpy as np

from ..config import AppearanceConfig
from ..imageio import imwrite
from .bubbles import Rect
from .embed import OnnxEmbedder, cosine, unit
from .gallery import Gallery, Sample
from .track import iou

log = logging.getLogger(__name__)


def _inter(a: Rect, b: Rect) -> int:
    w = min(a.x2, b.x2) - max(a.x, b.x)
    h = min(a.y2, b.y2) - max(a.y, b.y)
    return w * h if w > 0 and h > 0 else 0


def clear_box(box: Rect, others: list[Rect], blocked: list[Rect], max_overlap: float) -> bool:
    """框没被别的人 / 团子盖住、没压在 blocked（聊天面板）上：认装扮的好样本和图鉴收集共用。空框算不行。"""
    area = box.w * box.h
    if box.w <= 0 or box.h <= 0:
        return False
    for o in others:
        if iou(box, o) > max_overlap or _inter(box, o) / area > max_overlap:
            return False
    for b in blocked:
        if _inter(box, b) / area > max_overlap:
            return False
    return True


def good_crop(
    frame: np.ndarray, box: Rect, others: list[Rect], blocked: list[Rect], min_height: float, max_overlap: float
) -> np.ndarray | None:
    """好样本才返回裁图（框中间 60% 宽、整高），否则 None：框太小、被别的人 / 团子盖住、压在聊天面板上都不要。"""
    fh, fw = frame.shape[:2]
    if box.w <= 0 or box.h <= 0 or box.h < min_height * fh:
        return None
    if not clear_box(box, others, blocked, max_overlap):
        return None
    margin = int(round(box.w * 0.2))
    crop = Rect(box.x + margin, box.y, box.w - 2 * margin, box.h)
    x1, y1, x2, y2 = max(0, crop.x), max(0, crop.y), min(fw, crop.x2), min(fh, crop.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def describe_crop(frame: np.ndarray, box: Rect) -> np.ndarray:
    """送去描述用的裁图：框四周各扩 15%（带上头部），夹到画面内。"""
    fh, fw = frame.shape[:2]
    pad = Rect(
        box.x - int(round(box.w * 0.15)),
        box.y - int(round(box.h * 0.15)),
        box.w + 2 * int(round(box.w * 0.15)),
        box.h + 2 * int(round(box.h * 0.15)),
    )
    x1, y1, x2, y2 = max(0, pad.x), max(0, pad.y), min(fw, pad.x2), min(fh, pad.y2)
    return frame[y1:y2, x1:x2]


class ColorEmbedder:
    """内置的颜色特征：头（上 30%）和身体（下 70%）各一份 HSV 色相 / 饱和度直方图。亮度不进直方图，所以同色深浅分不开（已知限制）。"""

    key = "color-v1"

    def embed(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        cut = max(1, int(round(h * 0.3)))
        parts = [self._hist(img[:cut]), self._hist(img[cut:])]
        return unit(np.concatenate(parts))

    @staticmethod
    def _hist(region: np.ndarray) -> np.ndarray:
        if region.size == 0:
            return np.zeros(16 * 4, np.float32)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mask = ((v >= 30) & (v <= 245)).astype(np.uint8) * 255  # 去掉过黑 / 过曝的像素
        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 4], [0, 180, 0, 256])
        return unit(hist)


def make_embedder(cfg: AppearanceConfig):
    if cfg.model == "color":
        return ColorEmbedder()
    if cfg.model.lower().endswith(".onnx"):
        return OnnxEmbedder(cfg.model, cfg.size, cfg.norm, cfg.device, what="appearance.model")
    raise ValueError(f"appearance.model 只能是 \"color\" 或 .onnx 路径：{cfg.model}")


@dataclass
class Profile:
    """一个人的外观档案（好友 / 陌生人 / 团子自己）。"""

    feat: np.ndarray
    n: int
    updated: float
    seen: float
    desc: str = ""
    desc_feat: np.ndarray | None = None  # 判换装的基准：描述时的特征；中途记过换装后换成新那套稳下来时的平均特征
    crops: deque = field(default_factory=lambda: deque(maxlen=5))  # (框高, 描述用裁图)
    checked: bool = False  # 好友：这次上线和关系卡比过没有
    redescribed: int = 0
    settling: bool = False  # 刚记过一次换装、平均特征还在往新那套挪：挪稳之前不再判换装
    gallery: Gallery = field(default_factory=lambda: Gallery(40))  # 多张样本（认人靠它，不靠平均特征）


def _letters(i: int) -> str:
    """0 -> A，25 -> Z，26 -> AA，27 -> AB……"""
    out = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


class AppearanceBook:
    """外观记忆簿：这次上线学到的外观 + 关系卡里的旧外观。纯数据、线程安全（一把锁）：感知线程写、描述器回调写、身体线程读。"""

    def __init__(self, cfg: AppearanceConfig, key: str, keep: float = 5.0) -> None:
        self.cfg = cfg
        self.key = key
        self.keep = keep
        self._lock = threading.RLock()
        self.friends: dict[str, Profile] = {}
        self.strangers: dict[str, Profile] = {}
        self.me: Profile | None = None
        self.card_feats: dict[str, np.ndarray] = {}
        self.card_desc: dict[str, str] = {}
        self._stranger_count = 0  # 单调增长：陌生人的字母这次上线内不复用

    # ---- 写 ----
    def _get(self, kind: str, who: str) -> Profile | None:
        if kind == "friend":
            return self.friends.get(who)
        if kind == "stranger":
            return self.strangers.get(who)
        return self.me

    def learn(self, kind: str, who: str, feat, now: float, crop: tuple[int, np.ndarray] | None = None,
              *, dino=None, h: float = 0.0, pinned: bool = False) -> Profile:
        """滑动平均更新（第一次直接用，判换装用），同时把这张样本放进底库（认人用）。kind：friend / stranger / me。
        feat 是颜色特征，dino 是 DINOv2 特征（没开就不传），h 是框高，pinned = 名字标签证实过的（底库满了也不挤）。"""
        feat = unit(feat)
        with self._lock:
            p = self._get(kind, who)
            if p is None:
                p = Profile(feat=feat, n=1, updated=now, seen=now, gallery=Gallery(self.cfg.gallery_max))
                if kind == "friend":
                    self.friends[who] = p
                elif kind == "stranger":
                    self.strangers[who] = p
                else:
                    self.me = p
            else:
                e = self.cfg.ema
                p.feat = unit((1 - e) * p.feat + e * feat)
                p.n += 1
                p.updated = now
                p.seen = now
                if p.settling and cosine(p.feat, feat) >= self.cfg.match:  # 平均特征追上了新那套：以它为新的基准
                    p.desc_feat = p.feat.copy()
                    p.settling = False
            p.gallery.add(Sample(feat, None if dino is None else unit(dino), now, h, pinned))
            if crop is not None:
                p.crops.append(crop)
            return p

    def load_cards(self, outfits: dict[str, list[dict]]) -> None:
        """启动时从关系卡载入：每个好友取最后一套。key 不同的特征不比（换了特征模型），描述照留。"""
        with self._lock:
            self.card_feats.clear()
            self.card_desc.clear()
            for name, items in outfits.items():
                if not items:
                    continue
                last = items[-1]
                if last.get("desc"):
                    self.card_desc[name] = last["desc"]
                if last.get("key") == self.key and last.get("feat"):
                    self.card_feats[name] = unit(np.asarray(last["feat"], np.float32))

    def set_desc(self, kind: str, who: str, desc: str, feat) -> None:
        with self._lock:
            p = self._get(kind, who)
            if p is not None:
                p.desc = desc
                p.desc_feat = unit(feat)  # 还在挪稳（settling）的话，挪稳时 learn 会再换成稳下来的平均特征

    def mark_changed(self, kind: str, who: str) -> None:
        """刚记了一次中途换装：等平均特征挪稳到新那套（和新样本像到 match）再拿它当基准，期间 drifted 为假，
        一次换装只算一次；不靠"描述回来"清掉（没开描述、描述器放弃了也照样能判下一次换装）。"""
        with self._lock:
            p = self._get(kind, who)
            if p is not None:
                p.settling = True

    def shift(self, d: float, now: float) -> None:
        """感知暂停了 d 秒：陌生人"最后看到"的时间往后挪（不超过 now），暂停的时间不算他走开。"""
        with self._lock:
            for p in self.strangers.values():
                p.seen = min(p.seen + d, now)

    def forget(self, now: float) -> None:
        """删掉太久没见的陌生人。"""
        with self._lock:
            for k in [k for k, p in self.strangers.items() if now - p.seen > self.cfg.stranger_forget]:
                del self.strangers[k]

    # ---- 认人 ----
    def _scores(self, feat, exclude: set[str]) -> list[tuple[float, str]]:
        """feat 对这次上线学到的每个好友（不含 exclude）的颜色底库打分，从高到低；底库里没有可比样本的不列。"""
        out = []
        for n, p in self.friends.items():
            if n in exclude:
                continue
            sc = p.gallery.best(feat, "color")
            if sc is not None:
                out.append((sc, n))
        out.sort(reverse=True)
        return out

    def friend_scores(self, feat) -> list[tuple[str, float]]:
        """feat 对这次上线学到的每个好友的颜色底库打分（名字, 分数），从高到低（只读）。"""
        with self._lock:
            return [(n, sc) for sc, n in self._scores(unit(feat), set())]

    def assign_friends(self, cands: dict[int, np.ndarray], exclude: set[str]) -> dict[int, str]:
        """没有名字标签的轨迹 -> 好友名（"maybe"）。底库最高分过 match、比第二像的好友高 margin；一个名字只给相似度最高的那条轨迹。
        只认这次上线名字标签证实过的样本：关系卡里的旧特征不拿来认人。"""
        with self._lock:
            best: dict[str, tuple[float, int]] = {}
            for tid, feat in cands.items():
                scored = self._scores(feat, exclude)
                if not scored:
                    continue
                top, name = scored[0]
                if top < self.cfg.match:
                    continue
                if len(scored) > 1 and top - scored[1][0] < self.cfg.margin:
                    continue
                if name not in best or top > best[name][0]:
                    best[name] = (top, tid)
            return {tid: name for name, (_, tid) in best.items()}

    def unsure_friends(self, cands: dict[int, np.ndarray], exclude: set[str]) -> dict[int, str]:
        """疑似：底库最高分落在 [unsure, match) 的轨迹 -> 好友名；一个名字只给分数最高的那条。
        已被 assign_friends 拿走的名字由调用方放进 exclude。"""
        with self._lock:
            best: dict[str, tuple[float, int]] = {}
            for tid, feat in cands.items():
                scored = self._scores(feat, exclude)
                if not scored:
                    continue
                top, name = scored[0]
                if not (self.cfg.unsure <= top < self.cfg.match):
                    continue
                if name not in best or top > best[name][0]:
                    best[name] = (top, tid)
            return {tid: name for name, (_, tid) in best.items()}

    def still_like(self, track_feat, name: str) -> bool:
        """这条轨迹现在还像不像 name：他的底库最高分 ≥ match（不看关系卡）。"""
        with self._lock:
            p = self.friends.get(name)
            if p is None:
                return False
            sc = p.gallery.best(unit(track_feat), "color")
            return sc is not None and sc >= self.cfg.match

    def best_friend(self, feat) -> tuple[str | None, float]:
        """和哪个好友最像、底库余弦多少（只读；没有候选 = (None, 0.0)）。"""
        with self._lock:
            scored = self._scores(unit(feat), set())
        if not scored:
            return None, 0.0
        score, name = scored[0]
        return name, float(score)

    def looks_like_dango(self, dino: np.ndarray | None) -> bool:
        """DINOv2 特征像不像团子自己：团子底库最高分 ≥ dango_match，且严格高于每个好友底库的最高分（没有分的好友不算）。"""
        if dino is None:
            return False
        with self._lock:
            if self.me is None:
                return False
            dino = unit(dino)
            mine = self.me.gallery.best(dino, "dino")
            if mine is None or mine < self.cfg.dango_match:
                return False
            for p in self.friends.values():
                other = p.gallery.best(dino, "dino")
                if other is not None and other >= mine:
                    return False
            return True

    def stranger_id(self, feat, now: float, exclude: frozenset[str] | set[str] = frozenset()) -> tuple[str, bool]:
        """陌生人编号：认回已有的（第二项 = 离开超过 keep 秒又回来了），否则新编号。
        exclude = 此刻别的轨迹正占着的编号：最像的被占着就起新编号（不退而求其次认第二像的）。"""
        with self._lock:
            feat = unit(feat)
            scored = []
            for k, p in self.strangers.items():
                sc = p.gallery.best(feat, "color")
                if sc is not None:
                    scored.append((sc, k))
            scored.sort(reverse=True)
            if (scored and scored[0][1] not in exclude and scored[0][0] >= self.cfg.match
                    and (len(scored) == 1 or scored[0][0] - scored[1][0] >= self.cfg.margin)):
                p = self.strangers[scored[0][1]]
                back = now - p.seen > self.keep
                p.seen = now
                return scored[0][1], back
            name = "陌生人" + _letters(self._stranger_count)
            self._stranger_count += 1
            p = Profile(feat=feat, n=1, updated=now, seen=now, gallery=Gallery(self.cfg.gallery_max))
            p.gallery.add(Sample(feat, None, now, 0.0))
            self.strangers[name] = p
            return name, False

    # ---- 换装 ----
    def card_state(self, name: str) -> str:
        """new = 关系卡里没有可比的特征；same / changed = 这次学到的和卡里的比。还没学到时按 same（没有换装的证据）。"""
        with self._lock:
            cf = self.card_feats.get(name)
            if cf is None:
                return "new"
            p = self.friends.get(name)
            if p is None:
                return "same"
            return "same" if cosine(p.feat, cf) >= self.cfg.changed else "changed"

    def drifted(self, kind: str, who: str) -> bool:
        """现在的平均特征和基准（上次描述时 / 上次换装稳下来时）差多了（中途换了装）。刚记过换装、还没稳下来时为假。"""
        with self._lock:
            p = self._get(kind, who)
            return (p is not None and not p.settling and p.desc_feat is not None
                    and cosine(p.feat, p.desc_feat) < self.cfg.changed)

    # ---- 读 ----
    def look(self, kind: str, who: str) -> str:
        with self._lock:
            p = self._get(kind, who)
            if p is None:
                return self.card_desc.get(who, "") if kind == "friend" else ""
            return p.desc

    def best_crop(self, kind: str, who: str, min_height: float = 0.0) -> np.ndarray | None:
        """最近几个好样本里框最高的那张描述裁图；框高不到 min_height（像素）就 None。"""
        with self._lock:
            p = self._get(kind, who)
            if p is None or not p.crops:
                return None
            h, crop = max(p.crops, key=lambda c: c[0])
            return crop if h >= min_height else None


_BAD_NAME = re.compile(r'[<>:"/\\|?*]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _folder_name(who: str) -> str:
    """名字 → Windows 上能用的文件夹名：换掉非法字符、去掉结尾的点和空格；空 / . / .. → "_"；
    保留名（CON、NUL、COM1……，带不带扩展名、不分大小写）前面加 "_"。"""
    name = _BAD_NAME.sub("_", who).rstrip(" .")
    if not name:
        return "_"
    if name.split(".", 1)[0].rstrip(" ").upper() in _RESERVED:
        return "_" + name
    return name


class CropSaver:
    """运行时攒认人模型的训练数据：好样本的裁图存 folder/crops/<名字>/<毫秒>.jpg，一行元数据追加进 folder/appearance.jsonl。
    同一条轨迹隔 save_every 秒才存一张，总数到 save_max 就停。"""

    def __init__(self, folder: Path, save_every: float, save_max: int, wall: Callable[[], float] = time.time) -> None:
        self.folder = Path(folder)
        self.save_every = save_every
        self.save_max = save_max
        self.wall = wall
        self.count = 0
        self._last: dict[int, float] = {}
        self._lock = threading.Lock()

    def offer(self, track_id: int, who: str, crop: np.ndarray, now: float, row: dict) -> bool:
        with self._lock:
            if self.count >= self.save_max:
                return False
            if now - self._last.get(track_id, float("-inf")) < self.save_every:
                return False
            self._last[track_id] = now  # 先记上：写失败也隔 save_every 才再试，别每个样本都报一次错
            folder = self.folder / "crops" / _folder_name(who)
            stamp = int(self.wall() * 1000)
            name, i = f"{stamp}.jpg", 0
            while (folder / name).exists():
                i += 1
                name = f"{stamp}-{i}.jpg"
            imwrite(folder / name, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rel = f"crops/{folder.name}/{name}"
            with open(self.folder / "appearance.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": rel, **row}, ensure_ascii=False) + "\n")
            self.count += 1
            if self.count == self.save_max:
                log.info("认装扮训练数据攒够 %d 张，不再存了", self.save_max)
            return True
