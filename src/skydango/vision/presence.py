"""好友在不在场（spec docs/superpowers/specs/2026-10-06-friend-presence-design.md §1）：纯计算，不碰设备、不 sleep。

好友稍远一点头顶的名字标签就淡掉，走出画面时标签贴在屏幕边上（离得远连贴边都没有，按 Q 才贴出来），
所以"画面里看不到"不等于"走开了"。每个好友分四种：
- 身边（view）：画面里看得到（感知层的 last_seen 在 keep 内：读到标签、轨迹续着、按外观认的）
- 附近（near）：名字贴在屏幕边上（keep 内），或者 recheck 内按 Q 喊到过
- 找不到（lost）：都没有，开始计时
- 走开（left）：找不到满 leave_after 且之后喊过一声也没亮出他；一直喊不成满 confirm_max；不能喊时满 leave_after
在场 = 身边 + 附近 + 找不到（找不到还不算走开）。只认这次运行里在画面里见过的好友：光贴边 / 喊到不算"来了"。

只存时间戳，查询时现算；"画面里看到"直接读感知层的 last_seen 字典（同一个对象，只读）。
"""

from __future__ import annotations

from ..config import PerceptionConfig
from .people import Seen

NEVER = float("-inf")


class Presence:
    def __init__(self, cfg: PerceptionConfig, seen: dict[str, float], can_call: bool) -> None:
        self.cfg = cfg
        self.seen = seen  # 感知层的 last_seen：名字 → 最后在画面里看到的时间
        self.can_call = can_call  # 大脑模式、[call] auto 开着、不是 dry-run：找不到时身体会喊一声
        self._edge: dict[str, tuple[float, str]] = {}  # 名字 → (最后贴边的时间, 哪边)
        self._confirmed: dict[str, tuple[float, str]] = {}  # 名字 → (喊到他的那次呼喊开始的时间, "画面外·右边" / "远处")
        self._tried: dict[str, float] = {}  # 名字 → 最近一次没亮出他的呼喊开始的时间

    # ---- 输入 ----
    def edge(self, name: str, side: str, now: float) -> None:
        """这一帧他的名字标签贴在屏幕边上（side：左边 / 右边 / 上边 / 下边）。"""
        self._edge[name] = (now, side)

    def call_done(self, at: float, found: dict[str, Seen]) -> None:
        """一次呼喊窗口（at 开始）结束：found 里的人喊到了，记录里别的人记一次"喊过没亮出来"。"""
        for name, s in found.items():
            self._confirmed[name] = (at, "远处" if s.on_screen else f"画面外·{s.side}")
        for name in self._names() - set(found):
            self._tried[name] = at

    def shift(self, d: float, now: float) -> None:
        """暂停了 d 秒：时间戳往后挪（last_seen 由感知层自己挪）。"""
        self._edge = {n: (min(t + d, now), s) for n, (t, s) in self._edge.items()}
        self._confirmed = {n: (min(t + d, now), w) for n, (t, w) in self._confirmed.items()}
        self._tried = {n: min(t + d, now) for n, t in self._tried.items()}

    # ---- 查询 ----
    def state(self, name: str, now: float) -> str:
        """"view" / "near" / "lost" / "left"；没见过的名字是 ""。"""
        if name not in self._names():
            return ""
        if now - self.seen.get(name, NEVER) <= self.cfg.keep:
            return "view"
        if self._near(name, now):
            return "near"
        lost = now - self._lost_at(name)
        if lost >= self.cfg.confirm_max or (lost >= self.cfg.leave_after and (not self.can_call or self._was_tried(name))):
            return "left"
        return "lost"

    def present(self, names: list[str], now: float) -> list[str]:
        """在场：还没走开的（身边 + 附近 + 找不到）。"""
        return [n for n in names if self.state(n, now) in ("view", "near", "lost")]

    def in_view(self, names: list[str], now: float) -> list[str]:
        return [n for n in names if self.state(n, now) == "view"]

    def need_call(self, names: list[str], now: float) -> list[str]:
        """找不到、这次还没喊过的（不能喊时为空）。"""
        if not self.can_call:
            return []
        return [n for n in names if self.state(n, now) == "lost" and not self._was_tried(n)]

    def around(self, names: list[str], now: float) -> list[tuple[str, str]]:
        """画面里看不到、还在场的人和怎么知道的：贴边优先，否则喊到的那次（"远处，刚喊到"）；找不到的写"看不到了，在确认"。"""
        out = []
        for n in names:
            st = self.state(n, now)
            if st == "lost":
                out.append((n, f"看不到了，在确认（{now - self._lost_at(n):.0f} 秒）"))
                continue
            if st != "near":
                continue
            t, side = self._edge.get(n, (NEVER, ""))
            if now - t <= self.cfg.keep:
                out.append((n, f"画面外·{side}"))
                continue
            at, where = self._confirmed[n]  # near 又不贴边：一定是 _fresh_confirm 那条
            ago = now - at
            out.append((n, f"{where}，{'刚' if ago < 60 else f'{int(ago // 60)} 分钟前'}喊到"))
        return out

    def confirmed(self, name: str) -> float | None:
        c = self._confirmed.get(name)
        return c[0] if c is not None else None

    def left_note(self, name: str, now: float) -> str:
        """走开的原因（给 leave 事件的括号里）。"""
        lost = now - self._lost_at(name)
        if self.can_call and self._was_tried(name):
            return f"喊了一声也没看到，{lost:.0f} 秒了"
        if self.can_call:
            return f"{lost:.0f} 秒没看到，没喊成"
        return f"{lost:.0f} 秒没看到"

    # ---- 内部 ----
    def _names(self) -> set[str]:
        """只认这次运行里在画面里见过的：光贴边 / 喊到不算"来了"（不然一喊就冒出一串"来到身边"）。"""
        return set(self.seen)

    def _fresh_confirm(self, name: str) -> float:
        """喊到他的时间；喊的时候（或之后）他还在画面里的不算：那次"喊到"说明不了他走出画面后还在。"""
        at = self._confirmed.get(name, (NEVER, ""))[0]
        return at if at > self.seen.get(name, NEVER) else NEVER

    def _near(self, name: str, now: float) -> bool:
        return (now - self._edge.get(name, (NEVER, ""))[0] <= self.cfg.keep
                or now - self._fresh_confirm(name) <= self.cfg.recheck)

    def _lost_at(self, name: str) -> float:
        """找不到的起点：最后一条证据的时间（喊到的证据过期那一刻也算）。"""
        return max(self.seen.get(name, NEVER), self._edge.get(name, (NEVER, ""))[0],
                   self._fresh_confirm(name) + self.cfg.recheck)

    def _was_tried(self, name: str) -> bool:
        return self._tried.get(name, NEVER) >= self._lost_at(name)
