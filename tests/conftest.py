from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skydango.vision.bubbles import Rect  # noqa: E402
from skydango.vision.ocr import OcrLine  # noqa: E402

W, H = 1280, 720


def scene(bubbles: list[tuple[int, int, int, int]] = (), size=(W, H)) -> np.ndarray:
    """合成一帧：蓝紫色渐变天空 + 一些亮色小噪点 + 指定位置的白色圆角气泡（里面画深色“字”）。"""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (120, 70, 50)
    img[:, :, 0] = np.linspace(90, 170, h, dtype=np.uint8)[:, None]
    rng = np.random.default_rng(0)
    for _ in range(40):  # 星星 / 光点：亮但很小，应被过滤
        x, y = int(rng.integers(0, w - 4)), int(rng.integers(0, h - 4))
        img[y : y + 3, x : x + 3] = 255
    for x, y, bw, bh in bubbles:
        r = bh // 3
        color = (248, 250, 250)
        cv2.rectangle(img, (x + r, y), (x + bw - r, y + bh), color, -1)
        cv2.rectangle(img, (x, y + r), (x + bw, y + bh - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + bw - r, y + r), (x + r, y + bh - r), (x + bw - r, y + bh - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
        # 模拟文字笔画
        for i in range(x + r, x + bw - r, 14):
            cv2.rectangle(img, (i, y + bh // 3), (i + 8, y + 2 * bh // 3), (60, 60, 60), -1)
    return img


class FakeOcr:
    """按调用顺序返回预设文本；每次 recognize 返回一个片段。"""

    def __init__(self, texts: list[str] | None = None, score: float = 0.95) -> None:
        self.texts = list(texts or [])
        self.score = score
        self.calls = 0

    def recognize(self, img):
        self.calls += 1
        if not self.texts:
            return []
        text = self.texts.pop(0)
        if not text:
            return []
        h, w = img.shape[:2]
        return [OcrLine(text, self.score, Rect(2, 2, max(1, w - 4), max(1, h - 4)))]


class FakeDevice:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.calls: list[tuple] = []
        self.shown = False  # 输入框（软键盘）是否已经打开
        self.ime_calls = 0  # ime_shown 被调了几次（它是一次 adb dumpsys，很慢）

    def screenshot(self):
        frame = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return frame

    def tap(self, x, y):
        self.calls.append(("tap", x, y))

    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    def key(self, keycode):
        self.calls.append(("key", keycode))

    def input_text(self, text):
        self.calls.append(("text", text))

    def editor_action(self, code):
        self.calls.append(("editor", code))

    def hw_key(self, code):
        self.calls.append(("hw_key", code))

    def hw_key_down(self, code):
        self.calls.append(("hw_down", code))

    def hw_key_up(self, code):
        self.calls.append(("hw_up", code))

    def hw_key_hold(self, code, seconds):
        self.calls.append(("hw_hold", code, seconds))

    def ime_shown(self):
        self.ime_calls += 1
        return self.shown


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock():
    return Clock()


class FakePanelReader:
    """给 PanelManager 当 reader：面板开没开由 visible() 决定。"""

    def __init__(self, visible) -> None:
        self.visible = visible
        self.panel_closed_since = None

    def panel_visible(self, frame) -> bool:
        return self.visible()


def panel_manager(device, visible, mode="always"):
    from skydango.chat.panel import PanelManager
    from skydango.config import PanelConfig, VisionConfig

    return PanelManager(VisionConfig(mode="log"), PanelConfig(mode=mode), device, FakePanelReader(visible), sleep=lambda s: None)


class PanelState:
    def __init__(self, open_: bool = True) -> None:
        self.open = open_


def fake_panel(device, open_=True, mode="always"):
    """假面板：device.hw_key(46) 切换开关；返回 (PanelManager, 状态)。"""
    state = PanelState(open_)
    press = device.hw_key

    def hw_key(code):
        press(code)
        if code == 46:
            state.open = not state.open

    device.hw_key = hw_key
    return panel_manager(device, lambda: state.open, mode), state


class FakeLlm:
    """假的一次性模型（反思用）：complete() 记下调用，返回 reply（是异常就抛）；wait 给了就先等它。"""

    def __init__(self, reply="{}", wait=None) -> None:
        self.reply = reply
        self.wait = wait
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, messages, max_tokens=None):
        self.calls.append((system, messages[0]["content"]))
        if self.wait is not None:
            self.wait.wait(5)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply
