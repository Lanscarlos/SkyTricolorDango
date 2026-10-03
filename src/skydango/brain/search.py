"""有意识地找（spec docs/superpowers/specs/2026-10-03-attention-search-design.md §1）：纯计算，不碰设备。

三种找共用这里：找刚走开的好友（lost）、环顾看周围有没有人（scan）、大脑下的目标（find）。
每圈 step(obs, now) 给出这一圈要做的：按一下（press）/ 喊一声（call）/ 等着（wait）/ 结束（found / maybe / none / seen / empty）。
调用方真按了键调 pressed()，喊出去了调 call_sent()，拿到结果（或者没喊成）调 called()。
基本动作是分段转：一段连按 seg_presses 下（每下 nudge_max 秒），段和段之间停 dwell 秒让 YOLO 出框、名字标签读出来。

Heading 是粗略朝向：按键时长折成角度累计起来（往右为正），记 6 个方位上次在画面里的时间，只拿来挑"最久没看过"的方向，不求准；
走路、黑屏、别人转过镜头之后调用方 reset。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import AttentionConfig, TrackConfig
from ..vision.people import CallSeen, side_of
from .peek import Turn

log = logging.getLogger(__name__)

SECTORS = 6
SECTOR_DEG = 360.0 / SECTORS
NAMES = ("前面", "右前方", "右后方", "后面", "左后方", "左前方")  # 相对现在的朝向，每 60° 一个
EPS = 1e-6
SIDE = {"left": "左边", "right": "右边"}
TERMINAL = frozenset({"found", "maybe", "none", "seen", "empty"})


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def bearing_name(offset: float) -> str:
    """相对现在朝向的角度（往右为正）→ "右后方" 这样的叫法。"""
    return NAMES[round((offset % 360.0) / SECTOR_DEG) % SECTORS]


class Heading:
    def __init__(self, press_deg: float, nudge_max: float, hfov: float, now: float) -> None:
        self.press_deg, self.nudge_max, self.hfov = press_deg, nudge_max, hfov
        self.deg = 0.0
        self.seen: list[float] = []
        self.reset(now)

    def reset(self, now: float) -> None:
        """不知道朝哪了（走路、黑屏、别人转过镜头）：现在算 0°，只有眼前这个方位刚看过。"""
        self.deg = 0.0
        self.seen = [float("-inf")] * SECTORS
        self.mark(now)

    def turned(self, direction: str, seconds: float, now: float) -> None:
        """按了一下方向键：按住 seconds 秒折成角度（press_deg 对应 nudge_max 秒，线性估）。"""
        deg = self.press_deg * seconds / self.nudge_max
        self.deg = (self.deg + (deg if direction == "right" else -deg)) % 360.0
        self.mark(now)

    def mark(self, now: float) -> None:
        """视野里的方位记成刚看过（方位中心离朝向不到半个视野角）。"""
        for i in range(SECTORS):
            if _gap(i * SECTOR_DEG, self.deg) <= self.hfov / 2 + EPS:
                self.seen[i] = now

    def stalest(self, prefer: str) -> tuple[str, str]:
        """最久没看过的方位：(往哪边转, 相对现在的叫法)。一样久挑近的，一样近挑 prefer 那边（正后方也按 prefer）。"""
        best = None
        for i in range(SECTORS):
            center = i * SECTOR_DEG
            if _gap(center, self.deg) <= self.hfov / 2 + EPS:
                continue
            off = (center - self.deg) % 360.0
            dist = min(off, 360.0 - off)
            direction = prefer if abs(off - 180.0) < EPS else ("right" if off < 180.0 else "left")
            key = (self.seen[i], dist, direction != prefer)
            if best is None or key < best[0]:
                best = (key, direction, off)
        if best is None:  # 视野角 ≥ 360°：哪都看得到
            return prefer, "前面"
        return best[1], bearing_name(best[2])


@dataclass(frozen=True)
class Obs:
    """这一圈的感知结果（调用方拼好）。target / maybe 只对要找的那个人；friends / strangers 给环顾用。"""

    width: int = 1920
    target_x: float | None = None  # 要找的人：名字证实的（人物框或画面里亮着的名字标签）中心 x
    target_where: str = ""  # "右边·远"（调用方拼好；空就按 x 算左右）
    maybe_x: float | None = None  # 要找的人：只是"像他"（按位置接回 / 认装扮），没看到名字
    friends: tuple[str, ...] = ()  # 画面里认得出名字的好友
    strangers: int = 0  # 画面里别的人（陌生人、黑影、没看到名字的）


@dataclass(frozen=True)
class SearchStep:
    state: str  # press / call / wait，或 TERMINAL 之一
    turn: Turn | None = None  # press 才有
    note: str = ""  # 进度或结果


class Search:
    """一次找。计划是一串步骤：("call",) 喊一声；("turn", 方向, 段数) 分段转；
    ("edge", 段数, 后备) 喊一声认出他在画面外就往那边转，否则换成后备（None = 跳过、"circle" = 转一圈、或者另一个步骤）；
    ("alone", 方向, 段数) 画面里没有没挂名字的人才往那边转（有的话他多半就在其中，转走反而丢了）。"""

    def __init__(self, kind: str, steps: list[tuple], cfg: AttentionConfig, track: TrackConfig, heading: Heading, now: float,
                 who: str | None = None, side: str | None = None, where: str = "") -> None:
        self.kind, self.who, self.side, self.where = kind, who, side, where
        self.cfg, self.track, self.heading = cfg, track, heading
        self._steps = list(steps)
        self._i = 0
        self._segs_left: int | None = None
        self._presses_left = 0
        self._dwell_until: float | None = None
        self._last_press = float("-inf")
        self._progress = now  # 上次有进展（开始、按键、喊、一段转完）：resume_within 没进展就不找了
        self._calling = False
        self._called_once = False
        self._edge: str | None = None  # 喊一声时他的名字贴在屏幕哪边
        self._call_found: str | None = None  # 喊一声时他的名字亮在画面里："右边·远"
        self._crowd = 0  # 没往他那边转：画面里还有几个没挂名字的人
        self._look_until: float | None = None  # 环顾看到陌生人：看到什么时候
        self._presses = {"left": 0, "right": 0}
        self._obs: Obs | None = None
        self.state, self.note, self.aborted = "running", "", False
        self.friends: tuple[str, ...] = ()
        self.strangers = 0

    # ---- 三种计划 ----
    @classmethod
    def lost(cls, who: str, side: str | None, edge_exit: bool, can_call: bool, cfg: AttentionConfig, track: TrackConfig,
             heading: Heading, now: float) -> Search:
        """找刚走开的好友。从画面边上出去的：先往那边转，再喊一声看名字贴在哪边；在中间淡掉的：先喊一声，
        画面里没人再往他那边转。side = 他最后在画面哪半边（不知道是 None）。"""
        n = cfg.lost_segments
        if edge_exit and side:
            steps = [("turn", side, n), ("call",), ("edge", n, None)]
        else:
            steps = [("call",), ("edge", n, ("alone", side, n) if side else None)]
        if not can_call:
            steps = [s for s in steps if s[0] != "call"]
        return cls("lost", steps, cfg, track, heading, now, who=who, side=side)

    @classmethod
    def find(cls, who: str, side: str | None, edge_exit: bool, can_call: bool, cfg: AttentionConfig, track: TrackConfig,
             heading: Heading, now: float) -> Search:
        """大脑让找的人：刚走开过就按找走开的好友那样找；没线索就先喊一声，名字贴在哪边就往哪边转，没有就转一圈。"""
        if side is not None:
            s = cls.lost(who, side, edge_exit, can_call, cfg, track, heading, now)
            s.kind = "find"
            return s
        steps = [("call",), ("edge", cfg.lost_segments, "circle")]
        if not can_call:
            steps = steps[1:]
        return cls("find", steps, cfg, track, heading, now, who=who)

    @classmethod
    def scan(cls, cfg: AttentionConfig, track: TrackConfig, heading: Heading, rng: random.Random, prefer: str,
             now: float) -> Search:
        """环顾一片：往最久没看过的方位转 scan_segments 段。"""
        direction, where = heading.stalest(prefer)
        lo, hi = cfg.scan_segments
        return cls("scan", [("turn", direction, rng.randint(lo, hi))], cfg, track, heading, now, side=direction, where=where)

    # ---- 每圈 ----
    def step(self, obs: Obs | None, now: float) -> SearchStep:
        """obs = None：画面暂停 / 黑着，不判也不动（照样算没进展）。"""
        if self.state != "running":
            return SearchStep(self.state, note=self.note)
        if obs is not None:
            done = self._check(obs, now)
            if done is not None:
                return done
        if now - self._progress > self.cfg.resume_within + EPS:
            self.aborted = True
            return self._finish("none", "被打断太久，不找了")
        if obs is None:
            return SearchStep("wait", note="画面暂停着")
        if self._look_until is not None:
            return SearchStep("wait", note=self.describe())
        self._obs = obs
        return self._advance(now)

    def _check(self, obs: Obs, now: float) -> SearchStep | None:
        if self.kind == "scan":
            if obs.friends:
                self.friends = obs.friends
                return self._finish("seen", "看到了" + "、".join(obs.friends))
            if obs.strangers:
                self.strangers = max(self.strangers, obs.strangers)
                if self._look_until is None:
                    self._look_until, self._progress = now + self.cfg.scan_look, now
            if self._look_until is not None and now >= self._look_until - EPS:
                return self._finish("seen", f"有 {self.strangers} 个陌生人")
            return None
        if self._call_found is not None:
            return self._finish("found", f"{self.who}（{self._call_found}）")
        if obs.target_x is not None:
            return self._finish("found", f"{self.who}（{obs.target_where or side_of(obs.target_x, obs.width)}）")
        if obs.maybe_x is not None:
            return self._finish("maybe", f"{self.who}（没看到名字，{side_of(obs.maybe_x, obs.width)}）")
        return None

    def _advance(self, now: float) -> SearchStep:
        while self._i < len(self._steps):
            st = self._steps[self._i]
            if st[0] == "call":
                if self._calling:
                    return SearchStep("wait", note=self.describe())
                return SearchStep("call", note=f"喊一声找{self.who}")
            if st[0] == "edge":
                _, n, fallback = st
                if self._edge is not None:
                    self._steps[self._i] = ("turn", self._edge, n)
                elif fallback == "circle":
                    self._steps[self._i] = ("turn", self.heading.stalest("right")[0], SECTORS)
                elif fallback is not None:
                    self._steps[self._i] = fallback
                else:
                    self._i += 1
                continue
            if st[0] == "alone":
                _, direction, n = st
                crowd = self._obs.strangers if self._obs is not None else 0
                if crowd:
                    self._crowd = crowd
                    self._i += 1
                else:
                    self._steps[self._i] = ("turn", direction, n)
                continue
            _, direction, segs = st
            if self._segs_left is None:
                self._segs_left, self._presses_left = segs, self.cfg.seg_presses
                self._progress = now
            if self._dwell_until is not None:
                if now < self._dwell_until - EPS:
                    return SearchStep("wait", note=self.describe())
                self._dwell_until = None
                self._segs_left -= 1
                self._progress = now
                if self._segs_left <= 0:
                    self._i += 1
                    self._segs_left = None
                    continue
                self._presses_left = self.cfg.seg_presses
            if now - self._last_press < self.track.settle - EPS:
                return SearchStep("wait", note=self.describe())
            return SearchStep("press", Turn(direction, self.track.nudge_max), self.describe())
        return self._finish("empty" if self.kind == "scan" else "none", self._none_text())

    # ---- 调用方回报 ----
    def pressed(self, turn: Turn, now: float) -> None:
        self._last_press = self._progress = now
        self._presses[turn.direction] += 1
        self.heading.turned(turn.direction, turn.seconds, now)
        self._presses_left -= 1
        if self._presses_left <= 0:
            self._dwell_until = now + self.track.settle + self.cfg.dwell  # 最后一下停稳了再停 dwell 秒看

    def call_sent(self, now: float) -> None:
        """喊出去了，结果还要等呼喊窗口结束。"""
        self._calling, self._called_once, self._progress = True, True, now

    def called(self, seen: CallSeen | None, now: float) -> None:
        """喊一声有结果了；seen = None：没喊成（额度、关着）或一直没等到结果，接着下一步。"""
        self._calling, self._progress = False, now
        if self._i < len(self._steps) and self._steps[self._i][0] == "call":
            self._i += 1
        if seen is None or self.who is None:
            return
        for name, s in seen.friends.items():
            if name == self.who or similar(self.who, name, 0.75):
                if s.on_screen:
                    self._call_found = f"{s.side}·{s.distance}" if s.distance else s.side
                else:
                    self._edge = "left" if s.side == "左边" else "right"
                return

    # ---- 文字 ----
    def dirs(self) -> str:
        """往哪几边转过："左边、右边"；没转过是空串。"""
        return "、".join(SIDE[d] for d in ("left", "right") if self._presses[d])

    def describe(self) -> str:
        if self.kind == "scan":
            return f"在看看周围：往{self.where}（那边很久没看了）"
        if self.kind == "find":
            why = "大脑让找的"
        else:
            why = f"他刚从{SIDE[self.side]}走了" if self.side else "他刚走开"
        segs = math.ceil(sum(self._presses.values()) / self.cfg.seg_presses)
        doing = "，喊了一声在等" if self._calling else (f"，往{self.dirs()}转了 {segs} 段" if segs else "")
        return f"在找：{self.who}（{why}{doing}）"

    def _none_text(self) -> str:
        if self.kind == "scan":
            return "附近没人"
        if sum(self._presses.values()) >= SECTORS * self.cfg.seg_presses:
            head = f"转了一圈没找到{self.who}"
        elif self.dirs():
            head = f"往{self.dirs()}找了找，没看到{self.who}"
        else:
            head = f"没看到{self.who}"
        if self._called_once:
            head += "，喊了一声也没看到他的名字"
        if self._crowd:
            head += f"（画面里还有 {self._crowd} 个没挂名字的人，可能就是他）"
        return head

    def result_line(self) -> str:
        """status 里"刚才…"那一句。"""
        if self.kind == "scan":
            return f"往{self.where}看了看：{self.note}"
        if self.state == "found":
            return f"找到了{self.note}"
        if self.state == "maybe":
            return f"可能找到了{self.note}"
        if self.aborted:
            return f"找{self.who}，{self.note}"
        return self.note

    def _finish(self, state: str, note: str) -> SearchStep:
        self.state, self.note = state, note
        log.debug("找（%s%s）结束：%s %s", self.kind, f" {self.who}" if self.who else "", state, note)
        return SearchStep(state, note=note)


def event_text(s: Search, prev_empty: bool) -> str | None:
    """找完了值不值得告诉大脑（背景事件 search）；prev_empty = 上一次环顾是"附近没人"。find 走 task_done / task_failed，不在这里。"""
    if s.aborted or s.state == "running":
        return None
    if s.kind == "lost":
        act = f"往{s.dirs()}找了找" if s.dirs() else ("喊了一声找" if s._called_once else "找了找")
        if s.state == "maybe":
            return f"你{act}刚走开的{s.who}，那边有个人可能是他（没看到名字）"
        if s.state == "none":
            return f"你{act}刚走开的{s.who}，没看到他"
        return None  # 找到了：他回到身边，走现有的 return
    if s.kind == "scan" and s.state == "seen" and not s.friends and prev_empty:
        return f"你往{s.where}看了看：有 {s.strangers} 个陌生人"
    return None
