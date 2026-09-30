"""沙盒操作和状态（brain-sandbox spec §3）：冒充发言、来去、陌生人、地名、场景、新鲜事、快进、反思；/state 长轮询。

操作都经 `body.call` 在身体线程里做（和大脑的工具一样排队），返回 {"ok", "text"}；
参数不对（未知 op、缺参数、类型不对）抛 ValueError，HTTP 层转 400。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

from ..brain.body import ToolError
from .clock import parse_duration

log = logging.getLogger(__name__)

IDLE_POLL = 0.5  # 秒：长轮询里多久看一次 idle 变没变
QUIET = 2.0  # 秒（真实时间）：事件队列空、大脑不在想、反思不在跑、命令队列空，持续这么久才算安静
MAX_TEXT = 500  # 冒充发言 / 场景 / 新鲜事最多几个字
MAX_STRANGERS = 20
WEEKDAYS = "一二三四五六日"


def clock_text(wall: float) -> str:
    """沙盒时间给人看：9月30日 周三 23:30。"""
    t = time.localtime(wall)
    return f"{t.tm_mon}月{t.tm_mday}日 周{WEEKDAYS[t.tm_wday]} {t.tm_hour:02d}:{t.tm_min:02d}"


def span_text(seconds: float) -> str:
    """快进了多久：1 小时 / 10 分钟 / 1 小时 30 分钟 / 45 秒。"""
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    parts = [f"{h} 小时" if h else "", f"{m} 分钟" if m else "", f"{s} 秒" if s and not h else ""]
    return " ".join(p for p in parts if p) or "0 秒"


def _text(req: dict, key: str, allow_empty: bool = False) -> str:
    value = req.get(key)
    if not isinstance(value, str):
        raise ValueError(f"缺少参数 {key}（要字符串）")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"参数 {key} 不能是空的")
    if len(value) > MAX_TEXT:
        raise ValueError(f"参数 {key} 太长了（最多 {MAX_TEXT} 字）")
    return value


def _seconds(value) -> float:
    """快进多久：正的有限秒数，或者 30s / 10m / 2h。"""
    if isinstance(value, str):
        value = parse_duration(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("参数 seconds 要是秒数")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"快进的秒数要是正数：{value}")
    return value


class SandboxControl:
    def __init__(self, parts, world, sim, transcript, scene, mono: Callable[[], float] = time.monotonic) -> None:
        self.parts = parts
        self.body = parts.body
        self.world = world
        self.reader = world.reader
        self.env = world.env
        self.sim = sim
        self.transcript = transcript
        self.scene = scene
        self.mono = mono  # 判断"安静了 2 秒"用真实时间
        self._quiet_since: float | None = None
        self.body.on_blocked = lambda text, why: self.transcript.add("blocked", text, "团子", why=why)
        if parts.mind_log is not None:
            parts.mind_log.on_add = self._on_log

    # ---- 操作 ----
    def op(self, req: dict) -> dict:
        if not isinstance(req, dict):
            raise ValueError("请求要是 JSON 对象")
        name = req.get("op")
        make = {
            "say": self._say, "come": self._come, "leave": self._leave, "strangers": self._strangers, "place": self._place,
            "scene": self._scene, "notice": self._notice, "skip": self._skip, "time": self._time, "reflect": self._reflect,
        }.get(name) if isinstance(name, str) else None
        if make is None:
            raise ValueError(f"不认识的操作：{name!r}")
        run = make(req)  # 先在调用方线程校验参数
        try:
            return self.body.call(run, timeout=10)
        except ToolError as exc:
            return {"ok": False, "text": str(exc)}

    def _event(self, text: str) -> None:
        self.transcript.add("event", f"── {text} ──")

    def _say(self, req: dict):
        who, text = _text(req, "who"), _text(req, "text")

        def run():
            self.reader.say(who, text)
            self.transcript.add("heard", text, who)
            return {"ok": True, "text": f"{who} 说了：{text}"}
        return run

    def _come(self, req: dict):
        who = _text(req, "who")

        def run():
            if who in self.env.friends:
                return {"ok": False, "text": f"{who} 已经在身边了"}
            self.env.friends.append(who)
            self._event(f"{who}来了")
            return {"ok": True, "text": f"{who} 来了"}
        return run

    def _leave(self, req: dict):
        who = _text(req, "who")

        def run():
            if who not in self.env.friends:
                return {"ok": False, "text": f"{who} 本来就不在身边"}
            self.env.friends.remove(who)
            self._event(f"{who}走了")
            return {"ok": True, "text": f"{who} 走了"}
        return run

    def _strangers(self, req: dict):
        n = req.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= MAX_STRANGERS:
            raise ValueError(f"参数 n 要是 0~{MAX_STRANGERS} 的整数")

        def run():
            before, self.env.stranger_count = self.env.stranger_count, n
            self._event(f"陌生人 {before}→{n} 个")
            return {"ok": True, "text": f"身边陌生人 {n} 个"}
        return run

    def _place(self, req: dict):
        name = _text(req, "name", allow_empty=True)

        def run():
            self.env.place = name
            self._event(f"到了{name}" if name else "地名清空")
            return {"ok": True, "text": f"地名：{name}" if name else "地名清空了"}
        return run

    def _scene(self, req: dict):
        text = _text(req, "text", allow_empty=True)

        def run():
            self.scene.text = text
            self._event(f"场景：{text}" if text else "场景清空（团子看不清）")
            return {"ok": True, "text": "场景改好了" if text else "场景清空了"}
        return run

    def _notice(self, req: dict):
        text = _text(req, "text")

        def run():
            if not self.body.cfg.proactive.enabled:
                why = "没开主动开口（[proactive] enabled = false），新鲜事不会叫醒大脑"
                self.transcript.add("blocked", f"新鲜事：{text}", why=why)
                return {"ok": False, "text": why}
            self.body.news(text)
            blocked = self.body._watch_news(self.body.clock())  # 和眼睛发现新鲜事同一个入口、同一套护栏
            mine = [why for news, why in blocked if news.endswith(text)]
            if mine:
                self.transcript.add("blocked", f"新鲜事：{text}", why=mine[0])
                return {"ok": False, "text": f"新鲜事没放出去：{mine[0]}"}
            self._event(f"新鲜事：{text}")
            return {"ok": True, "text": f"放了新鲜事：{text}"}
        return run

    def _skip(self, req: dict):
        seconds = _seconds(req.get("seconds"))

        def run():
            self._log_energy()
            self.sim.skip(seconds)
            self._log_energy()
            self._event(f"快进 {span_text(seconds)}")
            return {"ok": True, "text": f"快进了 {span_text(seconds)}，现在 {clock_text(self.sim.wall())}"}
        return run

    def _time(self, req: dict):
        at = _text(req, "at")

        def run():
            before = self.sim.wall()
            self._log_energy()
            try:
                self.sim.set_time(at)
            except ValueError as exc:
                return {"ok": False, "text": str(exc)}
            self._log_energy()
            now = self.sim.wall()
            same_day = time.localtime(before)[:3] == time.localtime(now)[:3]
            shown = time.strftime("%H:%M", time.localtime(now)) if same_day else clock_text(now)
            self._event(f"拨到 {shown}")
            return {"ok": True, "text": f"拨到 {clock_text(now)}"}
        return run

    def _reflect(self, req: dict):
        def run():
            reflector = self.parts.reflector
            if reflector is None:
                return {"ok": False, "text": "这次没开反思（[inner] reflect = false）"}
            if reflector.running:
                return {"ok": False, "text": "正在反思"}
            content = self.body.reflect_materials(False)
            self.body._reflect_chat, self.body._reflect_comings = [], []  # 同身体自己开始反思时：这些已经交给这一次了
            reflector.start(content, self.sim.clock())
            self._event("开始反思")
            return {"ok": True, "text": "开始反思了"}
        return run

    def _log_energy(self) -> None:
        mind_log = self.parts.mind_log
        if mind_log is None or self.body.mind is None:
            return
        try:
            mind_log.energy(self.sim.wall(), self.body.energy_now())
        except Exception:
            log.exception("快进前后记精力出错")

    def _on_log(self, row: dict) -> None:
        if row.get("kind") == "reflect" and row.get("changes"):
            self._event("反思：" + "；".join(row["changes"]))

    # ---- 状态 ----
    def _limited(self) -> bool:
        try:
            return self.parts.brain.limit_left(self.sim.clock()) > 0
        except Exception:
            log.debug("读额度状态出错", exc_info=True)
            return False

    def idle(self) -> bool:
        """事件队列空 + 大脑不在一轮里 + 反思不在跑 + 身体命令队列空 + 冒充的话都读走了，持续 ≥ 2 秒（真实时间）。

        额度用完时大脑醒不来、事件会一直积在队列里：这时不看事件队列（回放照样往下走，报告里标"额度用完"）。"""
        reflector = self.parts.reflector
        busy = (
            (len(self.parts.events) > 0 and not self._limited())
            or bool(getattr(self.parts.brain, "in_turn", False))
            or (reflector is not None and reflector.running)
            or not self.body._commands.empty()
            or self.reader.pending() > 0
        )
        now = self.mono()
        if busy:
            self._quiet_since = None
            return False
        if self._quiet_since is None:
            self._quiet_since = now
        return now - self._quiet_since >= QUIET

    def limit_text(self) -> str:
        try:
            left = self.parts.brain.limit_left(self.sim.clock())
        except Exception:
            log.debug("读额度状态出错", exc_info=True)
            return ""
        if left <= 0:
            return ""
        return f"额度用完，约 {max(1, math.ceil(left / 60))} 分钟后再试"

    def state(self, after: int, timeout: float) -> dict:
        """等到聊天记录有 after 之后的新行或超时（after 比现在的版本还新 = 沙盒重启过，从头给）。"""
        t = self.transcript
        was_idle = self.idle()
        deadline = time.monotonic() + max(0.0, timeout)
        with t.cond:
            if after > t.version:
                after = 0
            else:  # 有新行、或者 idle 变了（安静下来 / 又忙起来）就回，回放靠它尽早往下走
                while t.version <= after:
                    left = deadline - time.monotonic()
                    if left <= 0 or self.idle() != was_idle:
                        break
                    t.cond.wait(min(IDLE_POLL, left))
        energy = self.body._energy
        reflector = self.parts.reflector
        wall = self.sim.wall()
        return {
            "version": t.version,
            "wall": wall,
            "clock_text": clock_text(wall),
            "energy": {
                "level": getattr(energy, "level", ""), "score": getattr(energy, "score", None), "note": getattr(energy, "note", ""),
            },
            "friends": list(self.env.friends),
            "strangers": int(self.env.stranger_count),
            "place": self.env.place,
            "scene": self.scene.text,
            "lines": t.since(after),
            "idle": self.idle(),
            "thinking": bool(getattr(self.parts.brain, "in_turn", False)),
            "reflecting": bool(reflector is not None and reflector.running),
            "limit": self.limit_text(),
        }
