"""快捷动作轮盘：图标库、读取 / 编辑轮盘、按数字键做动作。

光遇里的操作（MuMu「PC端操作方案」）：
- 数字键 1~8：直接做轮盘对应格子的动作（格子从正上方开始顺时针编号）
- 长按 Z 打开轮盘，此时按 E 进入编辑界面；编辑界面里点右侧列表的动作、再点格子就换好了，
  改动实时生效；右上角 × 随时能关，中间的 ✓ 只有改过东西后才能关
- 聊天记录面板（C）开着时长按 Z 不会出轮盘；输入框开着时按键会变成打字
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..config import WheelConfig
from ..device.base import Device
from ..imageio import imread, imwrite
from ..vision.bubbles import Rect, roi_rect
from ..vision.icons import best_match, find_icons, same_icon, silhouette, trim

log = logging.getLogger(__name__)

SLOTS = range(1, 9)
LIST_SCALES = [0.9, 1.0, 1.1]  # 图标库就是从列表里截的，尺寸基本一致
SLOT_SCALES = [1.1, 1.2, 1.25, 1.3, 1.4]  # 编辑界面轮盘里的图标比列表里大约 1.25 倍
RECENT_BADGE_THRESHOLD = 0.7  # 10-03 真机：时钟角标 0.96~1.0，列表里别处最高 0.30


class WheelError(RuntimeError):
    pass


class EmoteLibrary:
    """图标库：目录下每张 png 是一个动作，文件名（不含扩展名）就是动作名；下划线开头的文件忽略。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.templates: dict[str, np.ndarray] = {}
        self.reload()

    def reload(self) -> None:
        self.templates = {}
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.png")):
            if path.stem.startswith("_"):
                continue
            mask = trim(silhouette(imread(path)))
            if mask.any():
                self.templates[path.stem] = mask

    @property
    def names(self) -> list[str]:
        return list(self.templates)

    def identify(self, icon: np.ndarray, scales: list[float], threshold: float) -> tuple[str | None, float]:
        """认出一张截图里的图标是库里的哪个动作；认不出返回 (None, 最高分)。"""
        haystack = silhouette(icon)
        best_name, best_score = None, 0.0
        for name, template in self.templates.items():
            score = best_match(haystack, template, scales).score
            if score > best_score:
                best_name, best_score = name, score
        return (best_name, best_score) if best_score >= threshold else (None, best_score)


def _same_frame(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < 3.0


class Wheel:
    def __init__(
        self,
        device: Device,
        cfg: WheelConfig,
        library: EmoteLibrary,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.cfg = cfg
        self.library = library
        self.sleep = sleep
        self.clock = clock
        self.slots: dict[int, str | None] = {}  # 已知的轮盘内容（None = 认不出 / 不在图标库里）
        self.last_used: dict[int, float] = {}

    # ---- 几何 ----
    def slot_center(self, slot: int, width: int, height: int) -> tuple[int, int]:
        cx, cy = self.cfg.editor_center[0] * width, self.cfg.editor_center[1] * height
        radius = self.cfg.editor_radius * height
        angle = math.radians(-90 + 45 * (slot - 1))  # 1 号在正上方，顺时针
        return int(round(cx + radius * math.cos(angle))), int(round(cy + radius * math.sin(angle)))

    def slot_rect(self, slot: int, width: int, height: int) -> Rect:
        x, y = self.slot_center(slot, width, height)
        half = int(self.cfg.slot_icon_size * height / 2)
        return Rect(x - half, y - half, 2 * half, 2 * half)

    def free_slots(self) -> list[int]:
        return [s for s in SLOTS if s not in self.cfg.locked_slots]

    # ---- 编辑界面 ----
    def _editor_visible(self, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        crop = roi_rect(self.cfg.list_roi, width, height).crop(frame)
        return len(find_icons(crop, int(0.025 * height), int(0.1 * height))) >= 8

    def open_editor(self) -> np.ndarray:
        if self.device.ime_shown():
            raise WheelError("输入框开着，按键会变成打字；先关掉输入框再操作轮盘")
        self.device.hw_key_down(self.cfg.open_key)
        try:
            self.sleep(1.2)  # 长按一会儿轮盘才出来
            self.device.hw_key(self.cfg.edit_key)
            self.sleep(self.cfg.ui_delay)
        finally:
            self.device.hw_key_up(self.cfg.open_key)
        self.sleep(self.cfg.ui_delay)
        frame = self.device.screenshot()
        if not self._editor_visible(frame):
            raise WheelError("没能打开轮盘编辑界面：聊天记录面板开着时长按 Z 没反应，先按 C 关掉再试")
        return frame

    def close_editor(self) -> None:
        width, height = self._size()
        # 键盘操作之后的第一次触摸只会让游戏切到触屏模式（左下角冒出摇杆），点击本身被吞掉，所以要重试
        for _ in range(3):
            self.device.tap(int(self.cfg.editor_close[0] * width), int(self.cfg.editor_close[1] * height))
            self.sleep(self.cfg.ui_delay)
            if not self._editor_visible(self.device.screenshot()):
                return
        raise WheelError("轮盘编辑界面没关掉（点右上角 × 没反应），请手动关闭")

    def _size(self) -> tuple[int, int]:
        height, width = self.device.screenshot().shape[:2]
        return width, height

    def read_slots(self, frame: np.ndarray) -> dict[int, tuple[str | None, float]]:
        """编辑界面截图 → 每个格子是什么动作。"""
        height, width = frame.shape[:2]
        return {
            s: self.library.identify(
                self.slot_rect(s, width, height).crop(frame), SLOT_SCALES, self.cfg.slot_match_threshold
            )
            for s in SLOTS
        }

    def refresh(self) -> dict[int, tuple[str | None, float]]:
        frame = self.open_editor()
        try:
            result = self.read_slots(frame)
        finally:
            self.close_editor()
        self.slots = {s: name for s, (name, _) in result.items()}
        log.info("轮盘：%s（门槛 %.2f）", "、".join(f"{s} {name or '?'} {score:.2f}" for s, (name, score) in sorted(result.items())),
                 self.cfg.slot_match_threshold)
        return result

    # ---- 动作列表 ----
    def _list_crop(self, frame: np.ndarray) -> tuple[Rect, np.ndarray]:
        height, width = frame.shape[:2]
        area = roi_rect(self.cfg.list_roi, width, height)
        return area, area.crop(frame)

    def _scroll(self, down: bool, width: int, height: int) -> None:
        x = int(self.cfg.list_scroll_x * width)
        a, b = int(self.cfg.list_scroll_from * height), int(self.cfg.list_scroll_to * height)
        if down:
            self.device.swipe(x, a, x, b, 500)
        else:
            self.device.swipe(x, b, x, a, 300)
        self.sleep(self.cfg.ui_delay)

    def _scroll_to_top(self) -> None:
        prev = None
        for _ in range(15):
            frame = self.device.screenshot()
            _, crop = self._list_crop(frame)
            if prev is not None and _same_frame(prev, crop):
                return
            prev = crop
            height, width = frame.shape[:2]
            self._scroll(False, width, height)

    def _find_in_list(self, name: str) -> tuple[int, int] | None:
        template = self.library.templates[name]
        self._scroll_to_top()
        prev = None
        for _ in range(20):
            frame = self.device.screenshot()
            area, crop = self._list_crop(frame)
            match = best_match(silhouette(crop), template, LIST_SCALES)
            if match.score >= self.cfg.match_threshold:
                return area.x + match.x, area.y + match.y
            if prev is not None and _same_frame(prev, crop):
                return None  # 到底了
            prev = crop
            height, width = frame.shape[:2]
            self._scroll(True, width, height)
        return None

    def _assign_in_editor(self, slot: int, name: str) -> None:
        pos = self._find_in_list(name)
        if pos is None:
            raise WheelError(f"动作列表里没找到「{name}」")
        self.device.tap(*pos)  # 选中动作（出现在轮盘中间）
        self.sleep(self.cfg.ui_delay)
        frame = self.device.screenshot()
        height, width = frame.shape[:2]
        self.device.tap(*self.slot_center(slot, width, height))  # 放进格子
        self.sleep(self.cfg.ui_delay)
        frame = self.device.screenshot()
        got, score = self.library.identify(
            self.slot_rect(slot, width, height).crop(frame), SLOT_SCALES, self.cfg.slot_match_threshold
        )
        if got != name:
            raise WheelError(f"放进格子 {slot} 后认出来的是 {got}（相似度 {score:.2f}），可能没放上")
        self.slots[slot] = name
        log.info("轮盘格子 %d 换成了「%s」", slot, name)

    # ---- 对外 ----
    def assign(self, slot: int, name: str, force: bool = False) -> None:
        if slot not in SLOTS:
            raise WheelError(f"格子编号是 1~8，收到 {slot}")
        if slot in self.cfg.locked_slots and not force:
            raise WheelError(f"格子 {slot} 被锁定（wheel.locked_slots），不会覆盖")
        if name not in self.library.templates:
            raise WheelError(f"图标库里没有「{name}」（{self.library.directory}）")
        self.open_editor()
        try:
            self._assign_in_editor(slot, name)
        finally:
            self.close_editor()

    def slot_of(self, name: str) -> int | None:
        # 锁定格是道具（3 火 = 举蜡烛、8 门），带动画常被认成某个动作（10-03：火 → 赞赏 0.64）：不当成动作，同 EmotePlayer.on_wheel
        return next((s for s, n in self.slots.items() if n == name and s not in self.cfg.locked_slots), None)

    def _victim(self, candidates: list[int] | None = None) -> int:
        free = self.free_slots()
        if candidates is not None:
            free = [s for s in candidates if s in free]
        if not free:
            raise WheelError("能换的格子都被锁定了，没法换动作")
        # 最久没用过的先换；都没用过就按编号
        return min(free, key=lambda s: (self.last_used.get(s, float("-inf")), s))

    def ensure(self, name: str, candidates: list[int] | None = None) -> int:
        """保证动作在轮盘上，返回格子编号；不在就换掉一个最久没用的空闲格子（candidates 限定从哪些格子里挑）。"""
        if not self.slots:
            self.refresh()
        slot = self.slot_of(name)
        if slot is None:
            slot = self._victim(candidates)
            self.assign(slot, name)
        return slot

    def press(self, slot: int) -> None:
        """按这一格的数字键（锁定格也能按：3 号格常驻举蜡烛）。"""
        if self.device.ime_shown():
            raise WheelError("输入框开着，按数字键会变成打字")
        self.device.hw_key(self.cfg.slot_keys[slot - 1])
        self.last_used[slot] = self.clock()

    def perform(self, name: str) -> int:
        slot = self.ensure(name)
        self.press(slot)
        log.info("做动作「%s」（格子 %d）", name, slot)
        return slot

    # ---- 建图标库 ----
    def _recent_bottom(self, crop: np.ndarray, height: int) -> int | None:
        """列表这一屏里最下面那个「最近使用」时钟角标的中心 y（crop 坐标），没有就 None。"""
        path = Path(self.cfg.recent_badge)
        if not path.is_file():
            log.warning("找不到最近使用的时钟角标模板 %s，扫描不跳过最近使用", path)
            return None
        badge = trim(silhouette(imread(path)))
        if height != 1080:  # 模板按 1920×1080 截的
            scale = height / 1080
            badge = cv2.resize(badge, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        sil = silhouette(crop)
        if sil.shape[0] < badge.shape[0] or sil.shape[1] < badge.shape[1]:
            return None
        score = cv2.matchTemplate(sil.astype(np.float32), badge.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        ys, _ = np.where(score >= RECENT_BADGE_THRESHOLD)
        return int(ys.max()) + badge.shape[0] // 2 if len(ys) else None

    def scan_list(self, out_dir: str | Path) -> list[Path]:
        """把动作列表里的所有图标截下来存成 001.png、002.png……，并生成一张带编号的总览图。

        上次扫的编号图和总览图先整批挪进 `_old/<时间>/`：只覆盖 001~N 的话，上次多出来的会混进「动作名」页（10-03）。
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        old = [f for f in out.glob("*.png") if f.is_file() and (f.stem.isdigit() or f.name == "_sheet.png")]
        if old:
            backup = out / "_old" / time.strftime("%Y%m%d-%H%M%S")
            backup.mkdir(parents=True, exist_ok=True)
            for f in old:
                f.replace(backup / f.name)
            log.info("上次扫描的 %d 张图挪进了 %s", len(old), backup)
        saved: list[tuple[Path, np.ndarray, np.ndarray]] = []  # (路径, 彩色图, 剪影)
        self.open_editor()
        try:
            self._scroll_to_top()
            prev = None
            for _ in range(30):
                frame = self.device.screenshot()
                height = frame.shape[0]
                area, crop = self._list_crop(frame)
                cell = int(0.1 * height)
                margin = int(0.055 * height)
                found = find_icons(crop, int(0.025 * height), cell)
                bottom = self._recent_bottom(crop, height)
                if bottom is not None:
                    # 「最近使用」（一两行）全是后面动作的重复，时钟角标和剪影并成一块、截歪了（10-03）：最下面那个角标那行及以上都不要
                    found = [r for r in found if r.y + r.h // 2 > bottom + cell // 2]
                for r in found:
                    # 列表上下边缘处图标会被裁掉一截；翻一页只滚约 400 像素，每个图标总有一页完整落在中间
                    if r.y < margin or r.y2 > crop.shape[0] - margin:
                        continue
                    cx, cy = r.x + r.w // 2, r.y + r.h // 2
                    box = Rect(cx - cell // 2, cy - cell // 2, cell, cell).pad(0, crop.shape[1], crop.shape[0])
                    icon = box.crop(crop).copy()
                    mask = trim(silhouette(icon))
                    if not mask.any():
                        continue
                    dup = next((i for i, (_, _, m) in enumerate(saved) if same_icon(m, mask, 0.9)), None)
                    if dup is not None:
                        # 带时钟角标的“最近使用”图标万一漏进来，干净的版本会被当成它的重复：留剪影小的那个
                        path, _, old = saved[dup]
                        if np.count_nonzero(mask) < 0.95 * np.count_nonzero(old):
                            imwrite(path, icon)
                            saved[dup] = (path, icon, mask)
                        continue
                    path = out / f"{len(saved) + 1:03d}.png"
                    imwrite(path, icon)
                    saved.append((path, icon, mask))
                if prev is not None and _same_frame(prev, crop):
                    break
                prev = crop
                self._scroll(True, frame.shape[1], height)
        finally:
            self.close_editor()
        if saved:
            imwrite(out / "_sheet.png", _contact_sheet([(p.stem, icon) for p, icon, _ in saved]))
        return [p for p, _, _ in saved]


def _contact_sheet(items: list[tuple[str, np.ndarray]], cols: int = 10, cell: int = 120) -> np.ndarray:
    rows = math.ceil(len(items) / cols)
    sheet = np.full((rows * (cell + 24), cols * cell, 3), 40, np.uint8)
    for i, (label, icon) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + 24)
        img = cv2.resize(icon, (cell - 8, cell - 8))
        sheet[y + 4 : y + cell - 4, x + 4 : x + cell - 4] = img
        cv2.putText(sheet, label, (x + 8, y + cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
    return sheet
