"""沙盒的模拟时钟（brain-sandbox spec §2）。

`wall()` = 真实墙上时间 + 偏移，`clock()` = 真实 monotonic + 偏移；只能往前拨（`skip` / `set_time`）。
身体、账本、反思器、眼睛、流水账用它；大脑循环的超时计时仍用真实时间（不在这里管）。
本地时区一律用 `time.localtime` / `time.mktime`。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_HM = re.compile(r"^(\d{1,2}):(\d{2})$")
_FULL = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{2})$")
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)([smh])$")
_UNIT = {"s": 1.0, "m": 60.0, "h": 3600.0}


class SimClock:
    """模拟时钟：真实时钟加一个只增不减的偏移。"""

    def __init__(
        self,
        offset: float = 0.0,
        real_wall: Callable[[], float] = time.time,
        real_mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.offset = float(offset)
        self._real_wall = real_wall
        self._real_mono = real_mono
        self._lock = threading.Lock()

    def wall(self) -> float:
        return self._real_wall() + self.offset

    def clock(self) -> float:
        return self._real_mono() + self.offset

    def skip(self, seconds: float) -> None:
        """快进 seconds 秒；负数拒绝（只往前拨）。"""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError(f"不能往回拨：{seconds:g} 秒")
        with self._lock:
            self.offset += seconds

    def set_time(self, at: str) -> float:
        """拨到 at：`HH:MM` = 往后最近的这个时刻（今天已过就明天）；`YYYY-MM-DD HH:MM` 早于现在就拒绝。返回跳了多少秒。"""
        with self._lock:
            now = self.wall()
            target = _next_hm(at, now) if _HM.match(at.strip()) else parse_time(at)
            if target < now:
                raise ValueError(f"不能往回拨：{at} 早于现在的沙盒时间 {_fmt(now)}")
            jump = target - now
            self.offset += jump
            return jump


def parse_duration(text: str) -> float:
    """`30s` / `10m` / `2h` → 秒；别的写法 `ValueError`。"""
    m = _DURATION.match(str(text).strip())
    if not m:
        raise ValueError(f"时长写法不对：{text!r}（要 30s / 10m / 2h 这种）")
    return float(m.group(1)) * _UNIT[m.group(2)]


def parse_hm(text: str) -> tuple[int, int]:
    """`HH:MM` → (时, 分)；格式或范围不对 `ValueError`。"""
    m = _HM.match(str(text).strip())
    if not m:
        raise ValueError(f"时间写法不对：{text!r}（要 HH:MM）")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        raise ValueError(f"时间不对：{text!r}")
    return h, mi


def parse_time(text: str) -> float:
    """`YYYY-MM-DD HH:MM`（本地时间）→ 时间戳；格式不对 `ValueError`。"""
    m = _FULL.match(str(text).strip())
    if not m:
        raise ValueError(f"时间写法不对：{text!r}（要 HH:MM 或 YYYY-MM-DD HH:MM）")
    y, mo, d, h, mi = (int(g) for g in m.groups())
    try:
        time.strptime(f"{y}-{mo}-{d} {h}:{mi}", "%Y-%m-%d %H:%M")  # 校验月日时分的范围
    except ValueError:
        raise ValueError(f"时间不对：{text!r}") from None
    return time.mktime((y, mo, d, h, mi, 0, 0, 0, -1))


def is_time_text(text: str) -> bool:
    """是不是 `HH:MM` 或 `YYYY-MM-DD HH:MM`（只看格式和范围）。"""
    try:
        if _HM.match(str(text).strip()):
            parse_hm(text)
        else:
            parse_time(text)
    except ValueError:
        return False
    return True


def _next_hm(at: str, base: float) -> float:
    """base 之后（含 base）最近的 HH:MM。"""
    h, mi = parse_hm(at)
    lt = time.localtime(base)
    day = 0
    while True:  # 最多走两次；夏令时换算交给 mktime
        t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day, h, mi, 0, 0, 0, -1))
        if t >= base:
            return t
        day += 1


def _fmt(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def resolve_start(choice: str, floor: float, now: float, wake_hour: int) -> float:
    """启动时的沙盒墙上时间。

    - `resume` 或空：接着上次 = max(floor, now)（剧本 [start] 不写 time 就是空）
    - `sleep`：睡一晚 = floor 之后最近的 wake_hour:00（没有 floor 时从现在算）
    - `HH:MM`：floor 与 now 取大之后最近的这个时刻
    - `YYYY-MM-DD HH:MM`：原样，早于 floor 就 `ValueError`
    """
    choice = str(choice).strip()
    if choice in ("resume", ""):
        return max(floor, now)
    if choice == "sleep":
        base = floor if floor > 0 else now
        t = _next_hm(f"{int(wake_hour):02d}:00", base)
        return t if t > base else _next_hm(f"{int(wake_hour):02d}:00", base + 1)
    if _HM.match(choice):
        return _next_hm(choice, max(floor, now))
    if _FULL.match(choice):
        t = parse_time(choice)
        if t < floor:
            raise ValueError(f"不能往回拨：{choice} 早于最早能开始的时间 {_fmt(floor)}")
        return t
    raise ValueError(f"不认识的起始时间：{choice!r}（resume / sleep / HH:MM / YYYY-MM-DD HH:MM）")


def load_saved(path: Path) -> float | None:
    """clock.json 里上次停下时的沙盒墙上时间；没有 → None，坏了 → None + WARNING。"""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        wall = json.loads(path.read_text(encoding="utf-8"))["wall"]
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise TypeError("wall 不是数")
        return float(wall)
    except (OSError, ValueError, TypeError, KeyError) as e:
        log.warning("沙盒时钟文件 %s 读不了（%s），当作没有", path, e)
        return None


def save(path: Path, wall: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"wall": float(wall), "text": _fmt(wall)}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _last_end(days: Path) -> float:
    """days.jsonl 最后一行（读得了的）的 end；没有 → 0。"""
    if not days.is_file():
        return 0.0
    try:
        lines = days.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        log.warning("沙盒 days.jsonl 读不了：%s", e)
        return 0.0
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            end = json.loads(line).get("end")
        except (ValueError, AttributeError):
            continue
        if isinstance(end, (int, float)) and not isinstance(end, bool):
            return float(end)
    return 0.0


def _crashed_at(current: Path) -> float:
    """current.json（上次被强杀、还没补进 days.jsonl 的那一次）最后保存的时间，没有就用开始时间；读不了 → 0。
    下次启动时账本会拿它补一行"意外断了"，沙盒时间不能早于它。"""
    if not current.is_file():
        return 0.0
    try:
        data = json.loads(current.read_text(encoding="utf-8"))
        values = [data.get("saved"), data.get("start")]
    except (OSError, ValueError, AttributeError):
        return 0.0
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return 0.0


def floor_time(sandbox_dir: Path) -> float:
    """起始时间下限：clock.json、memory/inner/days.jsonl 最后一行的 end、current.json（被强杀的那次）取大；都没有 → 0.0。"""
    sandbox_dir = Path(sandbox_dir)
    inner = sandbox_dir / "memory" / "inner"
    saved = load_saved(sandbox_dir / "clock.json") or 0.0
    return max(saved, _last_end(inner / "days.jsonl"), _crashed_at(inner / "current.json"))
