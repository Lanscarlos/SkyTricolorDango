"""look_person 换角度：好友躲在团子身后时，边转边看（每一步只看 YOLO）让人露出来，太小就拉近。

这里只有纯决策：给一次观测（好友标签 / 身体框、团子框），返回下一步按什么键或者停下。
按键、等画面稳定、喂感知层、裁图在 body.Body.look_person 里。镜头看完不复位（大脑自己决定要不要 camera_reset）。

镜头绕团子转：按右键时远处的东西在画面上往左移、团子（支点）不动 —— 好友在团子左边就按右，推得更开。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import PeekConfig, TrackConfig
from ..vision.bubbles import Rect


@dataclass(frozen=True)
class Obs:
    tag: Rect | None  # 好友的名字标签（新鲜的）
    body: Rect | None  # 好友的身体框（感知层 people()）
    me: Rect | None  # 团子框（pick_self 过滤过）


@dataclass(frozen=True)
class Turn:
    direction: str  # "left" / "right"
    seconds: float


@dataclass(frozen=True)
class Zoom:
    direction: str  # "in" 拉近 / "out" 拉远


@dataclass(frozen=True)
class Done:
    reason: str  # "revealed" 露出来了 / "stuck" 转不动 / "lost" 看不到人了


def pick_self(boxes: list[Rect], width: int, center: float) -> Rect | None:
    """团子是镜头支点，总在画面水平中间：中心离中线超过 center × 屏宽的框不采信（认错人了），有几个取最近的。"""
    mid = width / 2
    ok = [b for b in boxes if abs(b.x + b.w / 2 - mid) <= center * width]
    return min(ok, key=lambda b: abs(b.x + b.w / 2 - mid), default=None)


def occluded(tag: Rect, me: Rect) -> bool:
    """名字标签压在团子身上：水平落在团子框内（两边各放宽 1/4），竖直在团子头顶附近到上半身之间。"""
    cx = tag.x + tag.w / 2
    if not me.x - me.w / 4 <= cx <= me.x2 + me.w / 4:
        return False
    return tag.y2 >= me.y - me.h / 2 and tag.y <= me.y + me.h * 0.6


def overlap_x(a: Rect, b: Rect) -> float:
    """a 和 b 水平重叠的宽度占 a 宽的比例。"""
    inter = min(a.x2, b.x2) - max(a.x, b.x)
    return max(0, inter) / a.w if a.w > 0 else 0.0


def _near_edge(box: Rect, width: int) -> bool:
    return box.x <= width * 0.01 or box.x2 >= width * 0.99


class PeekPlanner:
    """一步一步决定：先露出来（转，贴太近 / 转不动时拉远），再看大小（太小拉近，拉近后挡住 / 出画面就退一步）。"""

    def __init__(self, cfg: PeekConfig, track: TrackConfig, width: int, height: int) -> None:
        self.cfg, self.track = cfg, track
        self.width, self.height = width, height
        self.zoom_outs = self.zoom_ins = 0
        self.enlarging = False  # 已经露出来，在调大小
        self.undone = False  # 拉近坏了事、退过一步：不再拉近
        self._last_me: Rect | None = None
        self._streak_dir: str | None = None  # 转不动的判断（同 track）：同方向连按的次数和开始时还差多少
        self._streak_n = 0
        self._streak_gap = 0.0

    def next(self, obs: Obs) -> Turn | Zoom | Done:
        if obs.tag is None and obs.body is None:
            return Done("lost")
        if obs.me is not None:
            self._last_me = obs.me
        if self.enlarging:
            return self._enlarge(obs)
        me = obs.me or self._last_me or Rect(round(self.width * 0.45), 0, round(self.width * 0.1), self.height)
        if obs.body is not None and overlap_x(obs.body, me) < self.cfg.clear_overlap:
            return self._revealed(obs)
        target = obs.body or obs.tag
        assert target is not None
        tx, mx = target.x + target.w / 2, me.x + me.w / 2
        gap = me.w / 2 + target.w / 2 - abs(tx - mx)  # 还差多少像素才完全离开团子框
        if obs.body is None and gap <= 0:
            return self._revealed(obs)
        if obs.me is not None and obs.me.h > self.cfg.too_close * self.height and self.zoom_outs < self.cfg.max_zoom_out:
            return self._zoom_out()
        direction = "right" if tx < mx else "left"
        if direction == self._streak_dir and self._streak_gap - gap < self.track.stall_px:
            if self._streak_n >= self.track.stall_nudges:
                if obs.me is not None and self.zoom_outs < self.cfg.max_zoom_out:
                    return self._zoom_out()
                return Done("stuck")
        else:  # 换了方向或者有进展：重新计数
            self._streak_dir, self._streak_n, self._streak_gap = direction, 0, gap
        self._streak_n += 1
        seconds = self.cfg.gain * max(gap, 0) / (self.width / 2)
        return Turn(direction, round(min(max(seconds, self.track.nudge_min), self.track.nudge_max), 3))

    def _zoom_out(self) -> Zoom:
        self.zoom_outs += 1
        self._streak_dir, self._streak_n = None, 0
        return Zoom("out")

    def _revealed(self, obs: Obs) -> Zoom | Done:
        self.enlarging = True
        return self._grow(obs)

    def _enlarge(self, obs: Obs) -> Zoom | Done:
        body = obs.body
        me = obs.me or self._last_me
        spoiled = body is None or _near_edge(body, self.width) or (
            me is not None and overlap_x(body, me) >= self.cfg.clear_overlap)
        if spoiled and self.zoom_ins and not self.undone:  # 刚拉近就又挡住 / 出画面：退回一步
            self.undone = True
            self.zoom_ins -= 1
            return Zoom("out")
        return self._grow(obs)

    def _grow(self, obs: Obs) -> Zoom | Done:
        body = obs.body
        if (body is not None and not self.undone and body.h < self.cfg.too_small * self.height
                and self.zoom_ins < self.cfg.max_zoom_in):
            self.zoom_ins += 1
            return Zoom("in")
        return Done("revealed")
