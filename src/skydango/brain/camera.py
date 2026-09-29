"""转视角 / 缩放（光遇的键盘操作，见 game-ops §2），记下净偏移，能转回原位。

- 方向键 ←→ 水平转、↑ 镜头压低往上看、↓ 抬回来：按住 step 秒算一步（0.5 s 约 90°）
- 减号拉近、加号拉远（和直觉相反）；聊天记录面板开着时转视角、缩放都没反应 → 操作时向 PanelManager 借面板（关掉），做完归还
- 输入框开着时按键会变成打字 → 先按 BACK
- spin：按住 → 连续转整圈、边转边按 fps 截图（`#spin` / `camera spin` / 打开感知层时的 look_around）
- nudge：短按左 / 右 0.02~0.1 s（技能 track 用），按时长分别记净次数（`turns`）；不借面板（技能已经借了）
- reset 闭环：只靠反向按同样时长会差到 90°（D0 实测）。离开原位前存一张参照缩略图（每次离开只拍一次），
  复位时先按记账逐次反向重放，再左右小步试、哪边和参照图更像往哪边走（见 game-ops §2「转视角实测（D0）」）
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..device.base import KEYCODE_BACK
from .images import is_black

log = logging.getLogger(__name__)

KEYS = {"left": 105, "right": 106, "up": 103, "down": 108, "zoom_in": 12, "zoom_out": 13}  # Linux 键码
AXIS = {
    "left": ("turn", -1), "right": ("turn", 1),
    "up": ("pitch", 1), "down": ("pitch", -1),
    "zoom_in": ("zoom", 1), "zoom_out": ("zoom", -1),
}
UNDO = {"turn": ("right", "left"), "pitch": ("up", "down"), "zoom": ("zoom_in", "zoom_out")}  # (正方向, 反方向)
MAX_STEPS = 4

NUDGE_MIN, NUDGE_MAX = 0.02, 0.1  # 短按时长（秒）：0.02 s 一下就移 35~40 px（D0）
REFINE_STEP = 0.02  # 闭环细调每步按多久
REFINE_MAX = 30  # 细调最多走几步（按键总数不超过它的两倍）
REFINE_SETTLE = 0.4  # 细调每按一步等画面停稳
REFINE_MIN_SIM = 0.5  # 最高相似度低于它 = 没找到参照图的朝向
THUMB = (160, 90)
THUMB_Y = (0.0, 0.45)  # 只比画面上半部分的背景（下面是人和按钮）
THUMB_X = (0.34, 1.0)  # 避开左边的聊天记录面板


def _thumb(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    crop = img[int(h * THUMB_Y[0]) : int(h * THUMB_Y[1]), int(w * THUMB_X[0]) : int(w * THUMB_X[1])]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else 0.0


def thumb_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两张图上半部分背景（y 0~45%、x 34%~100%，缩成 160×90 灰度）的相关系数，-1~1。"""
    return _corr(_thumb(a), _thumb(b))


@dataclass
class SpinResult:
    before: np.ndarray  # 转之前
    frames: list[tuple[float, np.ndarray]]  # (按住后第几秒, 图)
    after: np.ndarray  # 转完停稳后
    seconds: float  # 实际按住了多久
    panel_reopened: bool  # 面板回到了该有的状态（原来就关着、闲着时不重开也算 True）
    blackout: bool  # 中途有整屏黑的帧（切场景）


class Camera:
    def __init__(
        self,
        device,
        step: float,
        panel,  # chat.panel.PanelManager；None 表示没有面板要管
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.step = step
        self.panel = panel
        self.sleep = sleep
        self.clock = clock
        self.offset = {"turn": 0, "pitch": 0, "zoom": 0}
        # nudge 的记账：{按键秒数: 净次数（右正左负）}。转动和时长不成比例（D0：0.02 s ≈ 37 px、0.1 s ≈ 66 px），
        # 但同一时长左右对称 —— 复位时按同样的时长逐次反向重放，不把秒数加总
        self.turns: Counter[float] = Counter()
        self.ref: np.ndarray | None = None  # 原位时的参照缩略图（_thumb），reset 闭环用
        self._away = False  # 这次离开原位已经拍过（或放弃了）参照图：reset 之前不再拍

    @property
    def turn_seconds(self) -> float:
        """nudge 净按了多少秒（右正左负），只用来显示；复位按 turns 重放。"""
        return sum(s * n for s, n in self.turns.items())

    def at_home(self) -> bool:
        return not any(self.offset.values()) and not any(self.turns.values())

    def remember(self) -> None:
        """离开原位前截一张图存成参照缩略图。每次离开原位只拍一次：记账抵消了镜头也不一定回到原位，
        reset 结束之前不再拍（forget_reference 作废之后也不补拍）。

        要在面板借走（关着）之后调：开 / 关聊天面板画面整体横移约 130 px，和复位时的画面对不上。"""
        if self._away:
            return
        self._away = True
        if not self.at_home():  # 不知道原位在哪：只能粗转
            return
        try:
            img = self.device.screenshot()
        except Exception as exc:  # 没参照图只是复位时少了细调
            log.warning("存镜头参照图失败：%s", exc)
            return
        if is_black(img):  # 切场景的黑屏不能当参照
            return
        self.ref = _thumb(img)

    def move(self, action: str, steps: int = 1, max_steps: int = MAX_STEPS) -> str:
        if action not in KEYS:
            raise ValueError(f"不认识的视角操作：{action}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), max_steps))
        with self._ready():
            self.remember()
            for _ in range(steps):
                self._step(action)
        return self.describe()

    def nudge(self, direction: str, seconds: float) -> float:
        """短按左 / 右 seconds 秒（夹到 0.02~0.1），返回实际按了多久。

        不借面板、不按 BACK：调用方是技能，开始时已经借好面板；按键在模拟器里 sleep，身体线程只等一次 adb。"""
        if direction not in ("left", "right"):
            raise ValueError(f"nudge 只能左右转：{direction}")
        seconds = round(min(max(float(seconds), NUDGE_MIN), NUDGE_MAX), 3)
        self.remember()
        self.device.hw_key_hold(KEYS[direction], seconds)
        self.turns[seconds] += 1 if direction == "right" else -1
        return seconds

    def forget_reference(self) -> None:
        """参照图不作数了（身体走动过、黑屏切过场景）：记账保留，复位时只粗转；这次离开原位也不再补拍。"""
        self.ref = None

    def release(self) -> None:
        """左右方向键各补一次抬起（Ctrl+C 打断 hw_key_hold 时键可能没抬起来）。"""
        for direction in ("left", "right"):
            try:
                self.device.hw_key_up(KEYS[direction])
            except Exception:
                log.debug("松开方向键失败", exc_info=True)

    def reset(self, refine_seconds: float | None = None) -> str:
        """撤销步数偏移 → 按记账逐次反向重放 nudge → 有参照图就闭环细调。

        refine_seconds：细调最多花多久（退出收尾时限时；None = 只受 REFINE_MAX 限制）。"""
        undo = [(UNDO[axis][1] if value > 0 else UNDO[axis][0], abs(value)) for axis, value in self.offset.items() if value]
        if not undo and not any(self.turns.values()) and self.ref is None:
            self._away = False
            return "镜头已经在原位"
        matched = None
        try:
            with self._ready():
                for action, n in undo:
                    for _ in range(n):
                        self._step(action)
                self._unturn()
                if self.ref is not None:
                    matched = self._refine(refine_seconds)
        finally:
            self.ref = None
            self._away = False
        if matched is None:
            return "镜头转回原位了（来回转会有一点偏差）"
        if matched:
            return "镜头转回原位了（和转之前的画面对准了）"
        return "镜头转回原位了（没对准，可能差一点）"

    def _unturn(self) -> None:
        """把 nudge 的记账按同样的时长逐次反向按回去（按完一下就从账上划掉：中途出错时下次复位接着按）。"""
        for seconds in sorted(self.turns, reverse=True):
            while self.turns[seconds]:
                n = self.turns[seconds]
                back = "left" if n > 0 else "right"
                self.device.hw_key_hold(KEYS[back], seconds)
                self.turns[seconds] -= 1 if n > 0 else -1
                self.sleep(0.2)
        self.turns.clear()

    def _similarity(self) -> float:
        assert self.ref is not None
        return _corr(_thumb(self.device.screenshot()), self.ref)

    def _refine(self, limit: float | None = None) -> bool:
        """左右各试一小步，哪边和参照图更像就往哪边走，直到不再变像、走满 REFINE_MAX 步或超过 limit 秒。

        按键总数不超过 REFINE_MAX 的两倍（含退回）；最高相似度 < 0.5 时退回粗转的位置，返回 False。"""
        budget = REFINE_MAX * 2
        presses = pos = moves = 0  # pos：相对粗转位置走了几步（右正）
        start = self.clock()

        def in_time() -> bool:
            return limit is None or self.clock() - start < limit

        def press(d: int) -> None:
            nonlocal presses, pos
            self.device.hw_key_hold(KEYS["right" if d > 0 else "left"], REFINE_STEP)
            presses += 1
            pos += d
            self.sleep(REFINE_SETTLE)

        try:
            self.sleep(REFINE_SETTLE)
            cur = self._similarity()
            for d in (1, -1):
                moved = False
                # 留够"退回一步 + 回到粗转位置"的按键
                while moves < REFINE_MAX and presses + 2 + abs(pos + d) <= budget and in_time():
                    press(d)
                    moves += 1
                    s = self._similarity()
                    if s > cur:
                        cur, moved = s, True
                        continue
                    press(-d)  # 这一步变差了：退回上一步（那里最像）
                    break
                if moved:
                    break
        except Exception as exc:  # 截图失败：当没对准
            log.warning("闭环复位时截图失败：%s", exc)
            cur = -1.0
        if cur >= REFINE_MIN_SIM:
            return True
        log.info("闭环复位没找到参照图的朝向（最高相似度 %.2f），退回粗转的位置", cur)
        while pos:
            d = -1 if pos > 0 else 1
            self.device.hw_key_hold(KEYS["right" if d > 0 else "left"], REFINE_STEP)
            pos += d
            self.sleep(0.2)
        return False

    def around(self, capture: Callable[[], Any], turns: int = 4, steps: int = 2) -> list:
        """环顾一圈：每转 steps 步（约 90°）截一张，共 turns 张，最后转满一圈回到原来的朝向（偏移不变）。"""
        frames = []
        with self._ready():
            for i in range(turns):
                if i:
                    for _ in range(steps):
                        self._press("right")
                    self.sleep(0.3)  # 等镜头停稳再截
                frames.append(capture())
            for _ in range(steps):
                self._press("right")
        return frames

    def spin(
        self, capture: Callable[[], np.ndarray], turns: int = 1, seconds_per_turn: float = 2.0, fps: float = 15.0
    ) -> SpinResult:
        """按住右键连续转 turns 整圈，按住期间按 fps 截图（每张记按住后第几秒）。整圈回到原朝向，偏移不变。"""
        total = turns * seconds_per_turn
        period = 1.0 / max(fps, 0.1)
        code = KEYS["right"]
        frames: list[tuple[float, np.ndarray]] = []
        with self._ready():
            before = capture()
            self.device.hw_key_down(code)
            start = self.clock()
            try:
                while (t := self.clock() - start) < total - 1e-6:
                    try:
                        frames.append((t, capture()))
                    except Exception as exc:  # 截图偶尔失败：少一张，照样转满一圈（半路停下镜头就歪了）
                        log.warning("转圈时截图失败，跳过这张：%s", exc)
                    self.sleep(max(0.0, period - (self.clock() - start - t)))
                seconds = self.clock() - start
            finally:
                self.device.hw_key_up(code)  # 出错 / Ctrl+C 也要松开，不然镜头会一直转
            self.sleep(0.3)  # 等镜头停稳
            after = capture()
        reopened = self.panel.restored() if self.panel is not None else True
        blackout = any(is_black(f) for _, f in frames)
        return SpinResult(before, frames, after, seconds, reopened, blackout)

    def _step(self, action: str) -> None:
        """走一步并马上记下偏移：中途出错时复原也准。"""
        self._press(action)
        axis, sign = AXIS[action]
        self.offset[axis] += sign

    def describe(self) -> str:
        turn, pitch, zoom = self.offset["turn"], self.offset["pitch"], self.offset["zoom"]
        parts = []
        if turn:
            parts.append(f"{'右' if turn > 0 else '左'}转了 {abs(turn)} 步")
        if pitch:
            parts.append(f"往{'上' if pitch > 0 else '下'}看了 {abs(pitch)} 步")
        if zoom:
            parts.append(f"拉{'近' if zoom > 0 else '远'}了 {abs(zoom)} 步")
        return "，".join(parts) or "原位"

    def _press(self, action: str) -> None:
        code = KEYS[action]
        if action.startswith("zoom"):
            self.device.hw_key(code)
            self.sleep(0.3)
            return
        self.device.hw_key_down(code)
        try:
            self.sleep(self.step)
        finally:
            self.device.hw_key_up(code)  # Ctrl+C / 出错时也要松开，不然镜头会一直转
        self.sleep(0.2)

    @contextmanager
    def _ready(self) -> Iterator[bool]:
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)
        if self.panel is None:
            yield False
            return
        with self.panel.borrow("camera") as was_open:
            yield was_open
