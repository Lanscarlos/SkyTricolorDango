"""装扮图鉴第 1 期：运行时收集近处的人的清楚整身裁图（spec 2026-10-02-catalog-collect-design）。

感知层每帧调 update：人物框过了门槛（够大、完整、没被挡、清楚）就打分，每个身份留最好的 per_who 张、两两隔 gap 秒；
陌生人轨迹断了写出他那份，每 flush_every 秒、close() 时全部写出。身份由感知层的 judge 给（Who），这里不碰黑影 / 第二层的细节。
只截图存盘：不发输入、不调模型；写盘出错只记 WARNING。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import CatalogConfig
from ..imageio import imread, imwrite
from .appearance import _folder_name, clear_box
from .bubbles import Rect
from .candle import black
from .track import Track

log = logging.getLogger(__name__)

KIND = "outfit"  # 以后地图 / 先祖 / 物品用别的值
SHARP_HEIGHT = 256  # 算清晰度前把裁图缩到这么高（不同远近的人可比）
FULL_HEIGHT = 0.5  # 框高到画面一半就不再加分
MAX_OVERLAP = 0.2  # 同 [appearance] max_overlap 的默认值
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 90]


@dataclass(frozen=True)
class Who:
    """感知层判的身份。name 也是目录名：好友名 / "团子" / "陌生人-t<轨迹>"。"""

    name: str
    kind: str  # friend / self / stranger
    sure: bool  # 名字被名字标签证实
    maybe: str | None = None  # 按外观 / 位置接回的"像谁"


@dataclass
class Shot:
    crop: np.ndarray
    row: dict  # 索引那一行（写盘时再加 file）
    score: float
    t: float


def sharpness(crop: np.ndarray) -> float:
    """清晰度：转灰度、缩到高 SHARP_HEIGHT，拉普拉斯方差（越模糊越小）。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    gray = cv2.resize(gray, (max(1, round(w * SHARP_HEIGHT / h)), SHARP_HEIGHT), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def padded(box: Rect, pad: float, width: int, height: int) -> Rect:
    """框四周各放宽 pad 比例，夹到画面内。"""
    dx, dy = round(box.w * pad), round(box.h * pad)
    x1, y1 = max(0, box.x - dx), max(0, box.y - dy)
    x2, y2 = min(width, box.x2 + dx), min(height, box.y2 + dy)
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def stranger_key(track_id: int) -> str:
    return f"陌生人-t{track_id}"


class CatalogCollector:
    """感知线程调 update，别的线程可能同时读 total / saved / buffer。"""

    def __init__(self, cfg: CatalogConfig, root: Path, run_name: str, day: str, *,
                 wall: Callable[[], float] = time.time, trace: bool = False) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.folder = self.root / "inbox" / day / run_name
        self.run_name = run_name
        self.wall = wall
        self.trace = trace
        self.candidates: list[dict] = []  # trace 时每个看过的候选一行（离线工具定门槛用）
        self._buffers: dict[str, list[Shot]] = {}
        self._dirty: set[str] = set()
        self._written: dict[str, list[dict]] = {}  # 身份 → 最近一次写盘的索引行
        self._archived = 0  # 写出后清掉缓冲的陌生人一共几张
        self._seen: dict[int, float] = {}  # 轨迹 id → 上次看的时间
        self._last_flush: float | None = None
        self._closed = False
        self._lock = threading.Lock()

    def _total(self) -> int:
        """内部计算总数（不加锁；仅供 _offer 调用）。"""
        return self._archived + sum(len(b) for b in self._buffers.values())

    @property
    def total(self) -> int:
        with self._lock:
            return self._total()

    def _saved(self) -> int:
        """内部计算盘上张数（不加锁；仅供内部方法调用）。"""
        return sum(len(rows) for rows in self._written.values())

    @property
    def saved(self) -> int:
        """现在盘上有几张。"""
        with self._lock:
            return self._saved()

    def buffer(self, key: str) -> list[Shot]:
        with self._lock:
            return list(self._buffers.get(key, []))

    def update(self, frame: np.ndarray, players: list[Track], selfs: list[Track], now: float,
               panel: Rect | None, place: str, judge: Callable[[Track], Who | None]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._last_flush is None:
                self._last_flush = now
            people = list(players) + list(selfs)
            height, width = frame.shape[:2]
            for t in people:
                if now - self._seen.get(t.id, float("-inf")) < self.cfg.every:
                    continue
                self._seen[t.id] = now
                who = judge(t)
                if who is None:
                    continue
                others = [o.box for o in people if o.id != t.id]
                self._consider(frame, t.box, who, others, panel, place, now, width, height)
            if now - self._last_flush >= self.cfg.flush_every:
                self._flush(list(self._buffers))
                self._last_flush = now

    def _gate(self, box: Rect, others: list[Rect], panel: Rect | None, width: int, height: int) -> str | None:
        c = self.cfg
        if box.w <= 0 or box.h < c.min_height * height:
            return "small"
        if box.x < c.edge or box.y < c.edge or box.x2 > width - c.edge or box.y2 > height - c.edge:
            return "edge"
        if not clear_box(box, others, [panel] if panel is not None else [], MAX_OVERLAP):
            return "blocked"
        return None

    def _consider(self, frame: np.ndarray, box: Rect, who: Who, others: list[Rect], panel: Rect | None,
                  place: str, now: float, width: int, height: int) -> None:
        h = box.h / height
        fail = self._gate(box, others, panel, width, height)
        if fail is None and who.kind == "stranger" and (black(frame, box) or 0.0) >= self.cfg.dark_max:
            fail = "dark"
        sharp = score = None
        crop = None
        if fail is None:
            r = padded(box, self.cfg.pad, width, height)
            crop = frame[r.y:r.y2, r.x:r.x2]
            sharp = sharpness(crop)
            score = sharp * min(h / FULL_HEIGHT, 1.0)
            if sharp < self.cfg.sharp_min:
                fail = "blurry"
        if self.trace:
            self.candidates.append({
                "t": round(now, 3), "who": who.name, "box": [box.x, box.y, box.w, box.h], "height": round(h, 4),
                "sharp": None if sharp is None else round(sharp, 1), "score": None if score is None else round(score, 1),
                "fail": fail,
            })
        if fail is not None:
            return
        row = {
            "kind": KIND, "who": who.name, "who_kind": who.kind, "sure": who.sure, "maybe": who.maybe,
            "run": self.run_name, "t": round(now, 3), "wall": round(self.wall(), 3), "box": [box.x, box.y, box.w, box.h],
            "height": round(h, 4), "sharp": round(sharp, 1), "score": round(score, 1), "place": place,
        }
        self._offer(who.name, Shot(crop.copy(), row, score, now))

    def _offer(self, key: str, shot: Shot) -> None:
        """放进 key 的缓冲：隔不够 gap 的只留更好的；满了换最差的；张数到 max_per_run 后只换不加。"""
        full = self._total() >= self.cfg.max_per_run
        buf = self._buffers.get(key)
        if buf is None:
            if full:
                return
            buf = self._buffers[key] = []
        close = [s for s in buf if abs(s.t - shot.t) < self.cfg.gap]
        if close:
            if all(shot.score > s.score for s in close):
                # 用列表推导式构造新缓冲，避免 list.remove() 对 numpy 数组的比较问题
                close_set = {id(s) for s in close}
                buf[:] = [s for s in buf if id(s) not in close_set]
                buf.append(shot)
                self._dirty.add(key)
            return
        if len(buf) < self.cfg.per_who and not full:
            buf.append(shot)
            self._dirty.add(key)
            return
        worst = min(buf, key=lambda s: s.score)  # 缓冲满了，或者这次运行的张数到顶了：只换不加
        if shot.score > worst.score:
            # 用列表推导式避免 list.remove() 对 numpy 数组的比较问题
            buf[:] = [s for s in buf if id(s) != id(worst)]
            buf.append(shot)
            self._dirty.add(key)

    def _write(self, key: str) -> None:
        """把 key 的缓冲按得分写成 1.jpg、2.jpg……；缓冲比上次少就删掉多出来的旧文件。"""
        buf = sorted(self._buffers.get(key, []), key=lambda s: -s.score)
        folder = self.folder / _folder_name(key)
        out = []
        for i, s in enumerate(buf, 1):
            path = folder / f"{i}.jpg"
            imwrite(path, s.crop, JPEG)
            out.append({"file": path.relative_to(self.root).as_posix(), **s.row})
        for i in range(len(buf) + 1, max(self.cfg.per_who, len(self._written.get(key, []))) + 1):
            (folder / f"{i}.jpg").unlink(missing_ok=True)
        self._written[key] = out
        self._dirty.discard(key)

    def _write_index(self) -> None:
        """索引整份重写（先写临时文件再替换），和目录里的图一一对应。"""
        lines = [json.dumps(r, ensure_ascii=False) + "\n" for k in sorted(self._written) for r in self._written[k]]
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / "index.jsonl.tmp"
        tmp.write_text("".join(lines), encoding="utf-8")
        os.replace(tmp, self.folder / "index.jsonl")

    def _flush(self, keys: Iterable[str]) -> None:
        keys = [k for k in keys if k in self._dirty]
        if not keys:
            return
        try:
            for k in keys:
                self._write(k)
            self._write_index()
        except Exception as exc:  # imwrite 编码失败是 RuntimeError / cv2.error，不只 OSError
            log.warning("图鉴收集写盘出错：%s", exc)

    def dropped(self, tracks: Iterable[Track]) -> None:
        """追踪器删掉的轨迹：陌生人那份写出去、清掉缓冲（写失败就留着，下次再写）。"""
        with self._lock:
            if self._closed:
                return
            for t in tracks:
                self._seen.pop(t.id, None)
                key = stranger_key(t.id)
                if key not in self._buffers:
                    continue
                self._flush([key])
                if key not in self._dirty:
                    self._archived += len(self._buffers.pop(key))

    def close(self) -> None:
        """全部写出；之后的 update / dropped 都不理。"""
        with self._lock:
            if self._closed:
                return
            self._flush(list(self._buffers))
            self._closed = True
            if self._saved():
                log.info("图鉴收集：存了 %d 张 → %s", self._saved(), self.folder)


def contact_sheet(folder: Path, cell: int = 240) -> tuple[np.ndarray, list[str]] | None:
    """一次运行存下的图拼成一张：每个身份一行、按名次排，每格 cell×cell（等比缩放、灰底补齐），左上角标清晰度。
    返回 (拼图, 每一行是谁)；没有索引 / 没有图返回 None。中文名字 cv2 画不了，所以名字放在 legend 里由调用方打印。"""
    index = Path(folder) / "index.jsonl"
    if not index.exists():
        return None
    groups: dict[str, list[dict]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        groups.setdefault(r["who"], []).append(r)
    if not groups:
        return None
    root = Path(folder).parents[2]  # <root>/inbox/<day>/<run>
    cols = max(len(rs) for rs in groups.values())
    sheet = np.full((cell * len(groups), cell * cols, 3), 40, np.uint8)
    legend = []
    for row, (who, rs) in enumerate(groups.items()):
        legend.append(f"{who}（{len(rs)} 张）")
        for col, r in enumerate(rs):
            try:
                img = imread(root / r["file"])
            except RuntimeError:
                continue
            scale = min(cell / img.shape[0], cell / img.shape[1])
            small = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
            y, x = row * cell, col * cell
            sheet[y:y + small.shape[0], x:x + small.shape[1]] = small
            cv2.putText(sheet, f"s{r['sharp']:.0f} c{r['score']:.0f} h{r['height']:.2f}", (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return sheet, legend
