"""点亮陌生人：团子身边的火焰线索和"他亮没亮"（spec 2026-10-03-light-flame-vanish）。

纯计算，不碰画面：感知层每次扫描把这一帧的火焰候选（`candle.find_flames`）和"某个位置下面那个人"（`PersonAt`）交给 `FlameWatch.scan`，
它负责：每处火焰一条线索（按距离配对）、认出不动的假火焰（灯笼）、出请求的那条、举蜡烛之后火焰怎么没的。
10-02 晚的 6 次存图抽成观测后在 `tests/test_light_replay.py` 里回放。
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from dataclasses import dataclass, field

import numpy as np

from ..config import SocialConfig
from .bubbles import Rect
from .candle import Disk
from .track import iou

DISK_EVERY = 0.3  # 团子周围最多隔这么久找一次火焰
DISK_GAP = 1.0  # 火焰断开不超过这么久算同一条线索（火焰会晃）
STATIC_SECS = 4.0  # 一条线索在原地待了这么久、从没到过 disk_sure：不动的假火焰（灯笼上的菱形）
STATIC_MOVE = 0.05  # ……"原地"：离第一次看到的位置不到这么多倍团子框高
STATIC_KEEP = 10.0  # 不动的假火焰断开这么久内还记着它的位置（候选先配给它，别被真火焰的线索捡走）
LIT_SAME_IOU = 0.3  # 判点亮：变亮的人物框要和见过他黑时的框重叠这么多（同一个人；别的亮人走过来不算）
LIT_SCANS = 2  # 加速判点亮：连续这么多次扫描都看到他变亮
LIT_MISSES = 2  # 火焰没了：至少连着这么多次扫描都没配上（单独一次可能是镜头一晃）
LIT_DARK_MARGIN = 0.05  # 见过他黑：black() 只差这么多到 lit_black 也算（10-02 21:42 举起时 0.34，差一点没算上）
LIT_SHRINK = 0.7  # 火焰缩到举起时的这么多以下再没的：往远处走了
LIT_EDGE_Y = 0.1  # 上下边只留这么多倍框高（真火焰离范围上沿常只有 0.3 倍框高：10-02 21:44）

# 某个位置下面的人：(人物框, black() 或 None = 量不准, YOLO 是不是认成 player_unlit)；没人 = None
PersonAt = Callable[[tuple[int, int], float], "tuple[Rect, float | None, bool] | None"]


def flame_area(me: Rect, cfg: SocialConfig, width: int, height: int) -> Rect:
    """在团子周围找火焰的范围：左右各 light_area_x 倍框高、往上 light_area_up 倍框高、下到框底，裁到画面里。"""
    cx = me.x + me.w / 2
    x1, x2 = round(cx - cfg.light_area_x * me.h), round(cx + cfg.light_area_x * me.h)
    y1 = round(me.y - cfg.light_area_up * me.h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, me.y2)
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def under_tag(pos: tuple[int, int], tag: Rect) -> bool:
    """火焰落在这个名字标签下面（好友举蜡烛给团子点火的圆圈）：左右各放宽 0.5 倍标签宽、往下 4.5 倍标签高。"""
    x1, y1, x2, y2 = round(tag.x - 0.5 * tag.w), tag.y, round(tag.x - 0.5 * tag.w) + 2 * tag.w, tag.y + round(4.5 * tag.h)
    return x1 <= pos[0] < x2 and y1 <= pos[1] < y2


def person_under(boxes: list[Rect], pos: tuple[int, int], r: float) -> int | None:
    """火焰往下那一块（横向 ±2r、纵向 −2r ~ +8r：火焰在胸口或头顶）里重叠最多的人物框是第几个；没有 None。"""
    area = Rect(round(pos[0] - 2 * r), round(pos[1] - 2 * r), round(4 * r), round(10 * r))
    best, most = None, 0
    for i, box in enumerate(boxes):
        w = min(area.x2, box.x2) - max(area.x, box.x)
        h = min(area.y2, box.y2) - max(area.y, box.y)
        if w > 0 and h > 0 and w * h > most:
            best, most = i, w * h
    return best


@dataclass(eq=False)  # 按对象比较、能当字典的键（_match 的结果）
class Clue:
    """一处火焰：连着看到的这一段。"""

    id: int
    first: float
    last: float
    pos: tuple[int, int]
    r: float
    score: float
    best: float
    home: tuple[int, int]  # 第一次看到的位置（判不动的假火焰）
    moved: float = 0.0  # 离 home 最远到过多远（像素）
    black: float | None = None  # 最近一次看到时下面那个人有多黑（None = 没人 / 量不准）
    person: Rect | None = None
    unlit: bool = False  # 最近一次看到时下面那个人 YOLO 认成 player_unlit
    radii: list[float] = field(default_factory=list)  # 最近 3 次的半高（尺度只有 8 档，判"变小了"取中位数）
    announced: bool = False
    static: bool = False


@dataclass
class Lighting:
    """举着蜡烛等结果（start ~ stop）：认准举蜡烛时那一团火焰（clue）。"""

    clue: int
    raised: float
    pos: tuple[int, int]
    r: float
    r0: float
    flame_last: float
    black0: float | None
    dark_box: Rect | None
    vel: tuple[float, float] = (0.0, 0.0)  # 火焰最后的速度（像素 / 秒）
    away: bool = False  # 最后看到时在范围边上 / 缩小了：没了 = 走开
    misses: int = 0  # 最后看到之后连着几次扫描没配上
    scan_at: float = float("-inf")  # 最近一次真的扫描过
    person: tuple[Rect, float | None] | None = None  # 最近一次在火焰位置找到的人和 black()
    person_at: float = float("-inf")
    dark_now: bool = False  # 最近一次扫描火焰位置下面的人量得到、还是黑的
    bright: int = 0  # 连着几次扫描看到他变亮（同一个人）
    trail: list[tuple[float, tuple[int, int]]] = field(default_factory=list)  # 火焰最近几次的位置（算速度）
    radii: list[float] = field(default_factory=list)  # 举起后最近 2 次看到的半高


class FlameWatch:
    def __init__(self, cfg: SocialConfig) -> None:
        self.cfg = cfg
        self.clues: list[Clue] = []
        self.lighting: Lighting | None = None
        self._seq = 0

    # ---- 每次扫描 ----
    def scan(self, now: float, me: Rect, area: Rect, flames: list[Disk], person_at: PersonAt,
             shift: tuple[float, float] = (0.0, 0.0)) -> None:
        """这一帧团子在 me、范围 area 里找到 flames（已经去掉了好友标签下、篝火上、面板挡着的）。
        shift：上次扫描以来画面整体平移了多少（转镜头、聊天面板开关；感知层估的背景平移）。不补的话平移超过 light_jump 时
        他的火焰配不上，会被当成"原地没了"。镜头绕着团子转时，贴着团子的人比远处背景移得少（同追踪接回的 drift），
        所以配对时按 0 / 一半 / 整个平移三种偏移取最近；没配上的线索（灯笼这种背景）按整个平移挪。"""
        h = me.h
        self.clues = [c for c in self.clues if self._alive(c, now)]
        matched = self._match(flames, h, shift)
        if shift[0] or shift[1]:
            self._shift_rest(shift, matched)
        for c, d in matched.items():
            self._see(c, d, now, h, person_at)
        for d in flames:
            if not any(d is m for m in matched.values()):
                self._seq += 1
                c = Clue(self._seq, now, now, (d.x, d.y), d.r, d.score, 0.0, (d.x, d.y))
                self.clues.append(c)
                self._see(c, d, now, h, person_at)
        if self.lighting is not None:
            self._watch_lighting(now, me, area, matched, person_at)

    def _shift_rest(self, shift: tuple[float, float], matched: dict[Clue, Disk]) -> None:
        """画面平移了：home 都挪（灯笼跟着背景走，还是"不动"），没配上的线索位置也挪；点亮中他的轨迹作废（速度重新算）。"""
        mv = lambda p: (round(p[0] + shift[0]), round(p[1] + shift[1]))  # noqa: E731
        for c in self.clues:
            c.home = mv(c.home)
            if c not in matched:
                c.pos = mv(c.pos)
        L = self.lighting
        if L is not None:
            if not any(c.id == L.clue for c in matched):
                L.pos = mv(L.pos)
            L.trail, L.vel = [], (0.0, 0.0)

    def _alive(self, c: Clue, now: float) -> bool:
        if self.lighting is not None and c.id == self.lighting.clue:
            return True  # 点亮中他的线索一直留着：火焰原地又出现了要接得上
        return now - c.last <= (STATIC_KEEP if c.static else DISK_GAP)

    def _match(self, flames: list[Disk], h: float, shift: tuple[float, float] = (0.0, 0.0)) -> dict[Clue, Disk]:
        """候选和线索按距离从近到远配对：一条线索一个候选，距离 ≤ light_jump × 框高（距离按 0 / 一半 / 整个平移取最近）。"""
        jump = self.cfg.light_jump * h
        offsets = [(0.0, 0.0)] + ([(shift[0] / 2, shift[1] / 2), shift] if shift[0] or shift[1] else [])
        pairs = sorted(
            (min(float(np.hypot(d.x - c.pos[0] - ox, d.y - c.pos[1] - oy)) for ox, oy in offsets), i, j)
            for i, c in enumerate(self.clues) for j, d in enumerate(flames)
        )
        used_c, used_d, out = set(), set(), {}
        for dist, i, j in pairs:
            if dist > jump:
                break
            if i in used_c or j in used_d:
                continue
            used_c.add(i)
            used_d.add(j)
            out[i] = flames[j]
        return {self.clues[i]: d for i, d in out.items()}

    def _see(self, c: Clue, d: Disk, now: float, h: float, person_at: PersonAt) -> None:
        c.last, c.pos, c.r, c.score, c.best = now, (d.x, d.y), d.r, d.score, max(c.best, d.score)
        c.radii = c.radii[-2:] + [d.r]
        c.moved = max(c.moved, float(np.hypot(d.x - c.home[0], d.y - c.home[1])))
        person = person_at(c.pos, c.r)
        c.person, c.black, c.unlit = (person[0], person[1], person[2]) if person is not None else (None, None, False)
        # 不动的假火焰：在原地待够久、从没到过 disk_sure（到过一次就是真火焰，黑影站着不动也会这样）
        c.static = (c.best < self.cfg.disk_sure and now - c.first >= STATIC_SECS and c.moved < STATIC_MOVE * h)

    def _watch_lighting(self, now: float, me: Rect, area: Rect, matched: dict[Clue, Disk], person_at: PersonAt) -> None:
        cfg, L = self.cfg, self.lighting
        L.scan_at = now
        mine = next((c for c in matched if c.id == L.clue), None)
        if mine is not None:  # 火焰还在：跟着他挪
            L.trail = [(t, p) for t, p in L.trail if now - t <= 1.0] + [(now, mine.pos)]
            (t0, p0) = L.trail[0]
            if now - t0 > 1e-6:
                L.vel = ((mine.pos[0] - p0[0]) / (now - t0), (mine.pos[1] - p0[1]) / (now - t0))
            L.pos, L.r, L.flame_last, L.misses = mine.pos, mine.r, now, 0
            L.radii = L.radii[-1:] + [mine.r]
            L.away = self._away(mine.pos, median(L.radii), L, me, area)
        else:
            L.misses += 1
            if not L.radii:  # 举起之后还没再看到过：按举起时的位置看在不在边上
                L.away = self._away(L.pos, L.r, L, me, area)
        person = person_at(L.pos, L.r)
        if person is None:
            L.bright, L.dark_now = 0, False
            return
        box, blk, unlit = person
        L.person, L.person_at = (box, blk), now
        if unlit or (blk is not None and blk >= cfg.lit_black - LIT_DARK_MARGIN):
            L.dark_box = box  # 他黑着：记下这时的框，之后变亮的得是他
        L.dark_now = blk is not None and blk >= cfg.lit_black
        # blk 为 None = 被团子挡住大半量不准：人还在，但不算变亮
        # 只数火焰没了之后的：火焰还在时夜里 black() 一抖攒下的"变亮"不算（再丢一帧就会被当成提前判点亮）
        bright = (L.misses > 0 and blk is not None and blk < cfg.lit_black
                  and (L.black0 is None or L.black0 - blk >= cfg.lit_drop)
                  and L.dark_box is not None and iou(box, L.dark_box) >= LIT_SAME_IOU)
        L.bright = L.bright + 1 if bright else 0

    def _away(self, pos: tuple[int, int], r: float, L: Lighting, me: Rect, area: Rect) -> bool:
        """火焰在范围边上（按最后的速度往前推一次扫描）或者缩小了（最近两次的中位数 < 举起时的 LIT_SHRINK）：这时候没了算走开。"""
        if r < LIT_SHRINK * L.r0:
            return True
        x, y = pos[0] + L.vel[0] * DISK_EVERY, pos[1] + L.vel[1] * DISK_EVERY
        mx, my = self.cfg.lit_edge * me.h, LIT_EDGE_Y * me.h
        return x - area.x < mx or area.x2 - x < mx or y - area.y < my or area.y2 - y < my

    # ---- 出请求 ----
    def ready(self, now: float) -> Clue | None:
        """够格出请求的线索：连着看到 ≥ light_after 秒、有一帧 ≥ disk_sure、这次还看得到；几条都够格取分数最好的。点亮中没有。"""
        if self.lighting is not None:
            return None
        cfg = self.cfg
        ok = [c for c in self.clues if not c.static and now - c.last <= DISK_GAP
              and now - c.first >= cfg.light_after and c.best >= cfg.disk_sure]
        return max(ok, key=lambda c: c.best) if ok else None

    def main(self, now: float) -> Clue | None:
        """现在最像真火焰的那条（状态 / 日志用）：够格的优先，再按最好分数。"""
        live = [c for c in self.clues if now - c.last <= DISK_GAP and not c.static]
        return self.ready(now) or (max(live, key=lambda c: c.best) if live else None)

    def get(self, clue_id: int) -> Clue | None:
        return next((c for c in self.clues if c.id == clue_id), None)

    # ---- 点亮中 ----
    def start(self, clue_id: int, now: float) -> Lighting:
        """身体举起蜡烛了：认准这条线索的火焰。线索已经没了（举之前刚断）：不知道火焰在哪，之后没看到就判走开（away）。"""
        cfg, c = self.cfg, self.get(clue_id)
        dark = c is not None and c.person is not None and (c.unlit or (c.black is not None and c.black >= cfg.lit_black - LIT_DARK_MARGIN))
        self.lighting = Lighting(
            clue=clue_id, raised=now, pos=c.pos if c else (0, 0), r=c.r if c else 20.0, r0=median(c.radii) if c else 20.0,
            flame_last=c.last if c else float("-inf"), black0=c.black if c else None, dark_box=c.person if dark else None,
            away=c is None,  # 举之前线索刚断：不知道火焰在哪，没了算走开
        )
        return self.lighting

    def verdict(self, clue_id: int, since: float, now: float, stale: float) -> bool | None:
        """举蜡烛（since）之后：True 他亮了；False 还在等；None 走开了。stale：最近一次扫描比这还旧就不判（感知没在跑）。

        火焰还在 → False。火焰没了（连着 LIT_MISSES 次扫描没配上、离最后看到 ≥ lit_vanish）：最后在范围边上 / 缩小了 → None；
        否则原地没了 → True，除非那个位置下面的人这次量得到、还是黑的（火焰可能只是被挡 / 晃丢了）→ False 接着等。
        火焰没了一次扫描、同一个人连着 LIT_SCANS 次看到变亮 → True（加速）。举起不满 lit_min 一律 False。"""
        cfg, L = self.cfg, self.lighting
        if L is None or L.clue != clue_id:
            return False
        if now - L.scan_at > stale or L.scan_at - since < cfg.lit_min or L.misses == 0:
            return False
        if L.bright >= LIT_SCANS:
            return True
        if L.misses < LIT_MISSES or L.scan_at - L.flame_last < cfg.lit_vanish:
            return False
        if L.away:
            return None
        return False if L.dark_now else True

    def stop(self) -> Lighting | None:
        """这次点亮结束：他的线索也作废（之后要重新连续看满 light_after 秒）。"""
        L, self.lighting = self.lighting, None
        if L is not None:
            self.clues = [c for c in self.clues if c.id != L.clue]
        return L

    def shift(self, d: float) -> None:
        """暂停了 d 秒：时间都往后挪，暂停的时间不算"火焰消失"。"""
        for c in self.clues:
            c.first += d
            c.last += d
        L = self.lighting
        if L is not None:
            L.raised += d
            L.flame_last += d
            L.person_at += d  # -inf 加 d 还是 -inf
            L.scan_at += d
            L.trail = [(t + d, p) for t, p in L.trail]
