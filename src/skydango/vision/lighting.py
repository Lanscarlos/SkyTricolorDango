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
# 10-03 晚真机复盘（docs/progress/2026-10-03-plan.md「10-03 晚真机复盘」）
LIT_JUMP = 0.25  # 举着蜡烛时他那团火焰只配离预测位置这么多倍框高以内的（204557：人挤，0.28 倍框高外别人的火焰被当成他的）
LIT_WEAK = 0.6  # 举着蜡烛时火焰候选放宽到这个分数（不到 disk_min_score 的只给他那团接续用：火焰淡下去时分数会掉）
LIT_DARK_WAIT = 2.5  # 火焰原地没了、下面的人还量得黑：最多再等这么久就放下、不鞠躬（212739 一直等到 8 秒；判点亮会对着黑影鞠躬）
# 10-04 19:51:45 真机：举蜡烛时聊天面板「看一眼」开了，画面横移、火焰接不上，误判走开，冷却 60 秒里他原地接着举了 20 多秒
GONE_RETRY = 3.0  # 判走开后冷却中：一团火焰连着看到这么久（且不短于 light_after）就不等冷却、再举一次（只一次）
PENDING_GRACE = 2.0  # 火焰冒出来后最多拦着聊天面板这么久（够出请求的时间再加这么多；不出请求的火焰别一直拦着）

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
    seen: bool = False  # 举起后看到过他的火焰（一次都没看到 = 举之前就没了，不能判点亮）
    darkest: float | None = None  # 他（dark_box 那个人）量到过最黑的 black()，取第二大（深色衣服按"降了多少"判还黑不黑）
    blacks: list[float] = field(default_factory=list)  # 他（dark_box 那个人）量到过的 black()


class FlameWatch:
    def __init__(self, cfg: SocialConfig) -> None:
        self.cfg = cfg
        self.clues: list[Clue] = []
        self.lighting: Lighting | None = None
        self._seq = 0

    # ---- 每次扫描 ----
    def scan(self, now: float, me: Rect, area: Rect, flames: list[Disk], person_at: PersonAt,
             shift: tuple[float, float] = (0.0, 0.0), extra: list[Disk] = ()) -> None:
        """这一帧团子在 me、范围 area 里找到 flames（已经去掉了好友标签下、篝火上、面板挡着的）。
        shift：上次扫描以来画面整体平移了多少（转镜头、聊天面板开关；感知层估的背景平移）。不补的话平移超过 light_jump 时
        他的火焰配不上，会被当成"原地没了"。镜头绕着团子转时，贴着团子的人比远处背景移得少（同追踪接回的 drift），
        所以配对时按 0 / 一半 / 整个平移三种偏移取最近；没配上的线索（灯笼这种背景）按整个平移挪。
        extra：平时不算的火焰候选（落在好友名字标签下面的 233045、分数不到 disk_min_score 的 204557）：不出线索，只给举着蜡烛时他那团接续用。"""
        h = me.h
        self.clues = [c for c in self.clues if self._alive(c, now)]
        matched = self._match(flames, h, shift, now)
        target = self._target()
        if target is not None and target not in matched and extra:
            got = self._match_target(target, list(extra), h, shift, now)
            if got is not None:
                matched[target] = got
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

    def _target(self) -> Clue | None:
        """举着蜡烛时他那条线索。"""
        L = self.lighting
        return None if L is None else self.get(L.clue)

    def _gap(self, c: Clue, d: Disk, h: float, shift: tuple[float, float], now: float) -> tuple[float, float]:
        """(距离, 上限)：距离按 0 / 一半 / 整个平移取最近。他那条线索（举着蜡烛、举起后认到过他之后）从按速度预测的位置量、
        上限收紧到 LIT_JUMP；举起后第一次还按 light_jump：出请求到真的举起常隔 1~2 秒，他可能挪了（10-02 21:54）。"""
        offsets = [(0.0, 0.0)] + ([(shift[0] / 2, shift[1] / 2), shift] if shift[0] or shift[1] else [])
        L, (px, py), limit = self.lighting, c.pos, self.cfg.light_jump * h
        if L is not None and c.id == L.clue and L.seen:  # seen 了 flame_last 一定是举起后的时间
            dt = min(max(now - L.flame_last, 0.0), 1.0)
            px, py, limit = L.pos[0] + L.vel[0] * dt, L.pos[1] + L.vel[1] * dt, LIT_JUMP * h
        return min(float(np.hypot(d.x - px - ox, d.y - py - oy)) for ox, oy in offsets), limit

    def _match_target(self, c: Clue, flames: list[Disk], h: float, shift: tuple[float, float], now: float) -> Disk | None:
        """extra 里给他那团配一个。
        认到他之前：只认分数不够的弱火焰、而且离他 LIT_JUMP 以内（证明举起后火焰还在，204557）；名字标签下面的不算——
        他的火焰举之前就没了、旁边好友举着点火圆圈，不能拿好友的圆圈当他的（评审）。
        认到之后：只认够分的（名字标签下面的，233045）；弱的不再续，点亮时冒的火花会被当成他的火焰（212739）。"""
        L = self.lighting
        if L.seen:
            pool = [d for d in flames if d.score >= self.cfg.disk_min_score]
        else:
            pool = [d for d in flames if d.score < self.cfg.disk_min_score
                    and min(float(np.hypot(d.x - L.pos[0] - ox, d.y - L.pos[1] - oy))
                            for ox, oy in [(0.0, 0.0), (shift[0] / 2, shift[1] / 2), shift]) <= LIT_JUMP * h]
        best = min(((self._gap(c, d, h, shift, now), k) for k, d in enumerate(pool)), default=None)
        return None if best is None or best[0][0] > best[0][1] else pool[best[1]]

    def _match(self, flames: list[Disk], h: float, shift: tuple[float, float] = (0.0, 0.0),
               now: float = 0.0) -> dict[Clue, Disk]:
        """候选和线索按距离从近到远配对：一条线索一个候选，距离 ≤ light_jump × 框高（他那条见 _gap）。"""
        pairs = sorted(
            (gap, i, j) for i, c in enumerate(self.clues) for j, d in enumerate(flames)
            for gap, limit in [self._gap(c, d, h, shift, now)] if gap <= limit
        )
        used_c, used_d, out = set(), set(), {}
        for dist, i, j in pairs:
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
            L.pos, L.r, L.flame_last, L.misses, L.seen = mine.pos, mine.r, now, 0, True
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
            if L.dark_box is None or iou(box, L.dark_box) < LIT_SAME_IOU:
                L.blacks = []  # 换了一个人（人挤，量到了旁边的黑影）：之前量的不算他的
            L.dark_box = box  # 他黑着：记下这时的框，之后变亮的得是他
        if blk is not None and L.dark_box is not None and iou(box, L.dark_box) >= LIT_SAME_IOU:
            L.blacks.append(blk)
            # 他最黑的时候取第二大的：夜里 black() 会抖，一次抖高不算（214919：同一个黑影 0.70~0.92）
            L.darkest = sorted(L.blacks)[-2] if len(L.blacks) >= 2 else L.blacks[0]
        # 还黑着：量得到、≥ lit_black，而且没比他最黑的时候降够 lit_drop（深色衣服点亮后也有 0.5~0.6）
        L.dark_now = blk is not None and blk >= cfg.lit_black and (L.darkest is None or L.darkest - blk < cfg.lit_drop)
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

    def pending(self, now: float, wait: float | None) -> bool:
        """点亮这件事在进行：举着蜡烛，或者有一团火焰（不是灯笼）连着看到还不到 wait + PENDING_GRACE 秒（可能马上出请求）。
        wait = None：只看举没举着（冷却中、不会出请求）。"""
        if self.lighting is not None:
            return True
        return wait is not None and any(
            not c.static and now - c.last <= DISK_GAP and now - c.first <= wait + PENDING_GRACE for c in self.clues)

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
            darkest=c.black if dark else None, blacks=[c.black] if dark and c.black is not None else [],
        )
        return self.lighting

    def verdict(self, clue_id: int, since: float, now: float, stale: float) -> bool | None:
        """举蜡烛（since）之后：True 他亮了；False 还在等；None 走开了。stale：最近一次扫描比这还旧就不判（感知没在跑）。

        火焰还在 → False。火焰没了（连着 LIT_MISSES 次扫描没配上、离最后看到 ≥ lit_vanish）：最后在范围边上 / 缩小了 → None；
        否则原地没了 → True，除非那个位置下面的人这次量得到、还是黑的（火焰可能只是被挡 / 晃丢了）→ False 接着等。
        火焰没了一次扫描、同一个人连着 LIT_SCANS 次看到变亮 → True（加速）。举起不满 lit_min 一律 False。
        举起后一次都没看到他的火焰（举之前就没了）：不判点亮，满 lit_min + lit_vanish → None（10-03 晚三次这样误判点亮）。
        还黑着最多等 LIT_DARK_WAIT 秒，之后 None（放下、不鞠躬）。"""
        cfg, L = self.cfg, self.lighting
        if L is None or L.clue != clue_id:
            return False
        if now - L.scan_at > stale or L.scan_at - since < cfg.lit_min or L.misses == 0:
            return False
        if not L.seen:
            return None if L.scan_at - since >= cfg.lit_min + cfg.lit_vanish else False
        if L.bright >= LIT_SCANS:
            return True
        if L.misses < LIT_MISSES or L.scan_at - L.flame_last < cfg.lit_vanish:
            return False
        if L.away:
            return None
        if not L.dark_now:
            return True
        # 还黑着：最多等 LIT_DARK_WAIT，之后放下、不鞠躬（他可能还黑着，对着黑影鞠躬最难看）
        return False if L.scan_at - L.flame_last < LIT_DARK_WAIT else None

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
