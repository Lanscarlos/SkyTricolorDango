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

log = logging.getLogger(__name__)

SECTORS = 6
SECTOR_DEG = 360.0 / SECTORS
NAMES = ("前面", "右前方", "右后方", "后面", "左后方", "左前方")  # 相对现在的朝向，每 60° 一个
EPS = 1e-6


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
