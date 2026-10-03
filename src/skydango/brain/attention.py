"""空闲注意力（东张西望）：团子闲着时"现在最想看什么、在找什么"，纯决策
（spec 2026-09-30-idle-attention §2 / §3；2026-10-03-attention-search §2）。

被动注意：每圈给一组候选目标（说话的人、走近的人、对团子做动作的人、站着的好友），按基础兴趣 ×（1 − 看腻）挑一个看；
目标在画面中间带里就不动、慢慢看腻，不在就小步转过去（同 track：目标在右边按右，把它拉向中间）；
换目标要高出一截、刚换过的一会儿内不换（防抖）。
有意识地找（search.py）：没有被动目标时推进正在做的找——找刚走开的好友（Body 调 start_lost 发起）、
画面里一阵子没人就往最久没看过的方向环顾一片（这里自己发起）。站着的好友不打断找。没有动机就不转。不回位。
这里不碰设备：Body 判断闲不闲、真按了键再调 pressed()；"喊一声"交给 Body，结果由 Body 交给 search.called()。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import AttentionConfig, TrackConfig
from .peek import Turn
from .search import TERMINAL, Heading, Obs, Search, SearchStep, event_text

log = logging.getLogger(__name__)

KINDS = ("talk_friend", "act_on_me", "approach", "talk_stranger", "friend_present")
KIND_NOTES = {"talk_friend": "在说话", "act_on_me": "在对你做动作", "approach": "朝你走过来",
              "talk_stranger": "在说话", "friend_present": "站在那"}
FRESH_KINDS = ("talk_friend", "talk_stranger", "approach", "act_on_me")  # 刚出现时值得"先看一眼再开面板"
MODES = ("随意", "好奇", "专心", "别动")
SCAN_MODE = {"随意": 1.0, "好奇": 0.5}  # 环顾间隔的倍数；专心 / 别动不环顾
SEARCH_MODES = ("随意", "好奇", "专心")  # 会找刚走开的好友
RESULT_KEEP = 120.0  # status 里"刚才…"留多久（秒）
EPS = 1e-6


@dataclass(frozen=True)
class Target:
    key: str  # "t:<轨迹 id>" 或 "n:<名字>"：同一个人
    kind: str  # KINDS 之一
    x: float  # 框中心 x（整图像素）
    who: str | None  # 名字；陌生人 None
    fresh: float | None = None  # 这次刺激开始的时间（气泡 start / 走近 / 动作）；变了就把看腻清零


@dataclass(frozen=True)
class Thought:
    current: Target | None  # 现在在看的（被动注意）
    action: Turn | None  # 这一圈想按的（门槛不开时 Body 不按）
    centered: bool  # current 已经在中间带里
    look_first: bool  # current 是刚出现的说话 / 走近 / 做动作：值得先看再开面板
    search: SearchStep | None = None  # 这一圈找的那一步（没在找 / 被被动注意打断是 None）
    event: str | None = None  # 找完了、值得告诉大脑的一句（背景事件 search）


class Attention:
    def __init__(self, cfg: AttentionConfig, track: TrackConfig, rng: random.Random, now: float, hfov: float = 90.0) -> None:
        self.cfg, self.track, self.rng = cfg, track, rng
        self.width = 1920
        self.mode = "随意"
        self.focus: str | None = None
        self._last_now = now
        self._bored: dict[str, float] = {}
        self._fresh: dict[str, float | None] = {}
        self._current: Target | None = None
        self._switched_at = float("-inf")
        self._last_press = float("-inf")
        self._err = 0.0  # 这一圈 current 离中线多远（给 pressed 记转不动用）
        self._streak_dir: str | None = None  # 转不动（同 track）：同方向连按几下、开始时离中线多远
        self._streak_n = 0
        self._streak_err = 0.0
        self._stuck: dict[str, float] = {}  # 转不动的人卡在哪（x）：他挪开之前不再朝他按，留在原地看着
        self._thought = Thought(None, None, False, False)
        # 有意识地找
        self.heading = Heading(cfg.press_deg, track.nudge_max, hfov, now)
        self.search: Search | None = None
        self.last_search: tuple[Search, float] | None = None  # 最近一次找完的：(那次找, 结束时间)
        self._scan_empty = False  # 上次环顾是"附近没人"
        self._empty_since: float | None = None  # 画面里从什么时候起一个人都没有（None = 有人 / 不知道）
        self._scan_end = float("-inf")
        self._scan_side = "left"  # 上次环顾往哪边：下次两边一样近时换一边

    def set_mode(self, mode: str, focus: str | None) -> None:
        """大脑定注意力模式（MODES 之一）和关注谁（空 = 不特别关注）。新模式不找的，正在找的也不找了。"""
        if mode not in MODES:
            raise ValueError(f"不认识的模式：{mode}")
        self.mode = mode
        self.focus = (focus or "").strip() or None
        s = self.search
        if s is not None and (mode not in SEARCH_MODES or (s.kind == "scan" and mode not in SCAN_MODE)):
            log.debug("注意力改成%s：不找了（%s）", mode, s.describe())
            self.search = None

    def _focused(self, t: Target) -> bool:
        return bool(self.focus and t.who and (t.who == self.focus or similar(self.focus, t.who, 0.75)))

    # ---- 兴趣 ----
    def base(self, t: Target) -> float:
        b = float(getattr(self.cfg, t.kind))
        if self.mode == "好奇" and t.kind == "talk_stranger":
            b = max(b, 0.7)
        if self._focused(t):
            b = max(b, self.cfg.focus_interest)
        return b

    def _centered(self, t: Target) -> bool:
        return abs(t.x - self.width / 2) <= self.cfg.center_band * self.width / 2

    def interest(self, t: Target) -> float:
        return self.base(t) * (1.0 - self._bored.get(t.key, 0.0))

    def _merge(self, targets: list[Target]) -> dict[str, Target]:
        """同一个人几种目标：只留基础兴趣最高的那种。"""
        out: dict[str, Target] = {}
        for t in targets:
            if t.key not in out or self.base(t) > self.base(out[t.key]):
                out[t.key] = t
        return out

    def _update_boredom(self, merged: dict[str, Target], dt: float) -> None:
        for key, t in merged.items():
            if t.fresh is not None and self._fresh.get(key) != t.fresh:
                self._fresh[key] = t.fresh
                self._bored[key] = 0.0  # 新的一句 / 新的一次走近：又有意思了
        for key in set(self._bored) | set(merged):
            t = merged.get(key)
            b = self._bored.get(key, 0.0)
            if t is not None and self._centered(t):
                b += self.cfg.bore_rate * dt * (0.5 if self._focused(t) else 1.0)
            else:
                b -= self.cfg.recover_rate * dt
            b = min(max(b, 0.0), 1.0)
            if b == 0.0 and t is None:
                self._bored.pop(key, None)
                self._fresh.pop(key, None)
            else:
                self._bored[key] = b

    def _choose(self, merged: dict[str, Target], now: float) -> Target | None:
        ok = [t for t in merged.values() if self.interest(t) >= self.cfg.min_interest]
        best = max(ok, key=self.interest, default=None)
        cur = merged.get(self._current.key) if self._current is not None else None
        if cur is not None and self.interest(cur) < self.cfg.min_interest:
            cur = None
        if cur is None:
            pick = best
        elif (best is not None and best.key != cur.key and now - self._switched_at >= self.cfg.switch_hold
              and self.interest(best) >= self.interest(cur) + self.cfg.switch_margin):
            pick = best
        else:
            pick = cur
        if (pick is None) != (self._current is None) or (pick is not None and pick.key != self._current.key):
            self._switched_at = now
            self._streak_dir, self._streak_n = None, 0
            if pick is not None:
                log.debug("注意力换到 %s（%s，兴趣 %.2f）", pick.who or "陌生人", pick.kind, self.interest(pick))
        return pick

    # ---- 每圈 ----
    def think(self, targets: list[Target], now: float, scale: float = 1.0, obs: Obs | None = None,
              scan_ok: bool = True) -> Thought:
        """scale：环顾间隔的倍数（心情精力）；obs：这一圈的感知结果（None = 画面暂停 / 不知道，不发起环顾）；
        scan_ok：False = 现在别发起环顾（技能在转镜头）。"""
        dt = min(max(now - self._last_now, 0.0), self.cfg.max_step)
        self._last_now = now
        self._note_empty(obs, now)
        merged = self._merge(targets)
        if self.mode == "专心":  # 只看分量重的（好友说话、对团子做事、关注的人）
            merged = {k: t for k, t in merged.items() if self.base(t) >= self.cfg.act_on_me}
        self._update_boredom(merged, dt)
        cur = self._current = self._choose(merged, now)
        action, centered, look_first = None, False, False
        if cur is not None:
            centered = self._centered(cur)
            look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
            if not centered and now - self._last_press >= self.track.settle:
                action = self._turn(cur)
                if action is None:  # 转不动、刚加了看腻：这一圈就重新挑（可能换走）
                    cur = self._current = self._choose(merged, now)
                    if cur is not None:
                        centered = self._centered(cur)
                        look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
        step = event = None
        passive = cur is not None and not (cur.kind == "friend_present" and self.search is not None)  # 站着的好友不打断找
        if not passive and self.mode != "别动":
            step, event = self._search_step(obs, now, scale, scan_ok)
            if step is not None:
                action = step.turn if step.state == "press" else None
        if self.mode == "别动":
            action, look_first = None, False
        self._thought = Thought(cur, action, centered, look_first, step, event)
        return self._thought

    def _note_empty(self, obs: Obs | None, now: float) -> None:
        if obs is None:
            return
        if obs.friends or obs.strangers:
            self._empty_since = None
        elif self._empty_since is None:
            self._empty_since = now

    def _scan_due(self, now: float, scale: float) -> bool:
        if not self.cfg.search or self.mode not in SCAN_MODE or self._empty_since is None:
            return False
        gap = self.cfg.scan_every * scale * SCAN_MODE[self.mode]
        return now - self._empty_since >= self.cfg.scan_after - EPS and now - self._scan_end >= gap - EPS

    def _search_step(self, obs: Obs | None, now: float, scale: float,
                     scan_ok: bool = True) -> tuple[SearchStep | None, str | None]:
        if self.search is None and scan_ok and self._scan_due(now, scale):
            prefer = "right" if self._scan_side == "left" else "left"
            self.search = Search.scan(self.cfg, self.track, self.heading, self.rng, prefer, now)
            self._scan_side = self.search.side or prefer
            log.debug("注意力：%s", self.search.describe())
        if self.search is None:
            return None, None
        step = self.search.step(obs, now)
        if step.state not in TERMINAL:
            return step, None
        return step, self._finish_search(now)

    def _finish_search(self, now: float) -> str | None:
        s, self.search = self.search, None
        self.last_search = (s, now)
        event = event_text(s, self._scan_empty)
        self._scan_end = now  # 刚环顾过 / 刚找过：同一片地方，隔 scan_every 再环顾
        if s.kind == "scan" and not s.aborted:
            self._scan_empty = s.state == "empty"
        return event

    def start_lost(self, who: str, side: str | None, edge_exit: bool, can_call: bool, now: float) -> bool:
        """Body：好友刚走开，去找找（换掉正在做的环顾 / 找别人）。开关关着、模式不找、已经在找他：返回 False。"""
        if not self.cfg.search or self.mode not in SEARCH_MODES:
            return False
        s = self.search
        if s is not None and s.kind == "lost" and s.who == who:
            return False
        if s is not None:
            log.debug("去找%s，先不%s", who, s.describe())
            if s.kind == "scan":
                self._scan_end = now
        self.search = Search.lost(who, side, edge_exit, can_call, self.cfg, self.track, self.heading, now)
        return True

    def lost_bearing(self, now: float) -> None:
        """走路、黑屏：不知道朝哪了。"""
        self.heading.reset(now)

    def external_move(self, now: float) -> None:
        """别人（大脑、track、look_person 换角度、面板开关）刚动过镜头：位置都变了，等画面停稳再按，转不动重新算，朝向也不算数了。"""
        self._streak_dir, self._streak_n = None, 0
        self._stuck.clear()
        self._last_press = max(self._last_press, now)
        self._scan_end = max(self._scan_end, now)  # 别人刚四处看过（找 / 环视 / 换角度）：算环顾过了
        self.heading.reset(now)

    def _turn(self, cur: Target) -> Turn | None:
        half = self.width / 2
        err = abs(cur.x - half)
        direction = "right" if cur.x > half else "left"  # 按右键画面里的东西往左移：目标在右边就按右
        stuck = self._stuck.get(cur.key)
        if stuck is not None:
            if abs(cur.x - stuck) < 2 * self.track.stall_px:
                return None  # 还卡在那：看着他就好，别一直往一边按
            del self._stuck[cur.key]  # 他挪开了，再试试
        if (direction == self._streak_dir and self._streak_n >= self.track.stall_nudges
                and self._streak_err - err < self.track.stall_px):
            self._bored[cur.key] = min(1.0, self._bored.get(cur.key, 0.0) + self.cfg.stuck_bored)
            self._stuck[cur.key] = cur.x
            self._streak_dir, self._streak_n = None, 0
            log.debug("注意力转不动 %s，看腻一截", cur.who or "陌生人")
            return None
        self._err = err
        seconds = min(max(self.cfg.gain * err / half, self.track.nudge_min), self.track.nudge_max)
        return Turn(direction, round(seconds, 3))

    def pressed(self, turn: Turn, now: float) -> None:
        """Body 真按了这一下：找的那一步交给 search；被动注意记按键时间、转不动计数（隔太久说明别人可能动过镜头，重新计数）。"""
        th = self._thought
        if th.search is not None and th.search.state == "press" and self.search is not None:
            self.search.pressed(turn, now)  # 里面会转 heading
            self._last_press = now
            return
        self.heading.turned(turn.direction, turn.seconds, now)
        if (turn.direction != self._streak_dir or now - self._last_press > 3 * self.track.settle
                or self._streak_err - self._err >= self.track.stall_px):  # 换方向 / 隔太久 / 有进展：重新计数（同 track）
            self._streak_dir, self._streak_n, self._streak_err = turn.direction, 0, self._err
        self._streak_n += 1
        self._last_press = now

    def describe(self, now: float | None = None) -> str:
        """status 一行；now 给了才写"刚才…（几秒前）"。"""
        cur = self._thought.current
        if cur is not None and not (cur.kind == "friend_present" and self.search is not None):
            line = f"在看：{cur.who or '陌生人'}（{KIND_NOTES[cur.kind]}）"
        elif self.search is not None:
            line = self.search.describe()
        elif self.last_search is not None and now is not None and now - self.last_search[1] <= RESULT_KEEP:
            s, t = self.last_search
            line = f"刚才{s.result_line()}（{now - t:.0f} 秒前）"
        else:
            line = "没在看什么"
        if self.mode != "随意" or self.focus:
            head = f"注意力：{self.mode}" + (f"，关注{self.focus}" if self.focus else "")
            line = f"{head}；{line}"
        return line
