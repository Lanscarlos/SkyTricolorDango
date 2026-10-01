"""按 Q 喊一声（spec docs/superpowers/specs/2026-10-01-q-call-design.md §2、§3）：结果和给大脑看的文字。

身体 `Body.call_out` 同步按键 + 连拍、开感知层的呼喊窗口；窗口结束（约 6 秒）后 `env.call_result(at)` 才有结果。
大脑工具 / 手动控制在自己的线程里用 `wait_result` 等它，身体照常读聊天。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from ..vision.bubbles import Rect
from ..vision.people import CallSeen

HALO_TEXT = {
    "self": "光圈：认出了你自己（中间）",
    "others": "光圈：别人也在喊，没认出你自己",
    "none": "光圈：没看到",
}
WAIT_EXTRA = 4.0  # 窗口结束后最多再等几秒（感知层一帧没跑到、身体正忙）


@dataclass
class CallResult:
    at: float  # 按键时刻（身体时钟）；拒绝时是想喊的时刻
    reason: str  # "brain" / "brain-owner"（主人命令窗口里放宽了间隔）/ "auto" / "manual"
    refused: str = ""  # 非空 = 没喊，原因
    dry: bool = False  # dry-run：没按键
    halo: str = "off"  # "self" / "others" / "none" / "skipped" / "off"
    self_box: Rect | None = None  # 光圈认出的团子框
    seen: CallSeen | None = None  # 窗口结束后补上


def seen_text(seen: CallSeen) -> str:
    """"认出 小明（右边·远）、懒洋洋大王（左边·近）；小红在画面外（左边）；还有 1 个没挂名字的人"。"""
    on = [f"{n}（{s.side}·{s.distance}）" if s.distance else f"{n}（{s.side}）" for n, s in seen.friends.items() if s.on_screen]
    off = [f"{n}在画面外（{s.side}）" for n, s in seen.friends.items() if not s.on_screen]
    parts = []
    if on:
        parts.append("认出 " + "、".join(on))
    if off:
        parts.append("、".join(off))
    if not parts:
        parts.append("没看到谁的名字")
    if seen.unnamed:
        parts.append(f"还有 {seen.unnamed} 个没挂名字的人")
    return "；".join(parts)


def halo_text(halo: str) -> str:
    return HALO_TEXT.get(halo, "")


def tool_text(r: CallResult) -> str:
    """大脑工具 / 手动控制的结果。"""
    body = seen_text(r.seen) if r.seen is not None else "没等到结果（感知层暂停了？）"
    halo = halo_text(r.halo)
    return f"喊了一声：{body}。" + (f"{halo}。" if halo else "")


def event_text(seen: CallSeen) -> str:
    """身体自动喊完的背景事件。"""
    return f"你下意识喊了一声：{seen_text(seen)}。"


def status_text(r: CallResult, now: float) -> str:
    """status 一行："上次喊：2 分钟前（认出小明；你自己在中间）"。"""
    gap = max(0.0, now - r.at)
    when = f"{gap:.0f} 秒前" if gap < 60 else f"{gap // 60:.0f} 分钟前"
    names = list(r.seen.friends) if r.seen is not None else []
    what = "认出" + "、".join(names) if names else "没看到谁的名字"
    if r.halo == "self":
        what += "；你自己在中间"
    return f"上次喊：{when}（{what}）"


def wait_result(body, r: CallResult, poll: float = 0.25, sleep: Callable[[float], None] = time.sleep) -> CallResult:
    """在调用方的线程里等呼喊窗口结束：每 poll 秒经身体线程问一次感知层，最多 window + WAIT_EXTRA 秒（时钟不走也按次数停）。"""
    limit = body.cfg.call.window + WAIT_EXTRA
    deadline = body.clock() + limit
    for _ in range(int(limit / poll) + 1):
        seen = body.call(lambda: body.env.call_result(r.at))
        if seen is not None:
            r.seen = seen  # 一般 r 就是 body.last_call：status 跟着有了
            return r
        if body.clock() >= deadline:
            break
        sleep(poll)
    return r
