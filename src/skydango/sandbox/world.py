"""沙盒世界（brain-sandbox spec §2）：身体 / 大脑 / 内心层原样跑，只把"接世界的那一圈"换成假的。

- 设备：截图永远是一张 1920×1080 中灰图（不黑，免得判成黑屏），按键 / 点屏幕 / 输入法什么也不做
- 读聊天：线程安全队列，页面"冒充 X 说 Y"入队，身体下一圈取出
- 身边：好友名单、陌生人数、地名由页面设置，身体照常比出 arrive / leave / stranger
- 眼睛：手写的场景文字，空着 = 看不清
- 说话、动作、走路：写进沙盒的聊天记录
- 没有镜头、好友树、面板识别、互动请求
"""

from __future__ import annotations

import dataclasses
import logging
import queue
from contextlib import contextmanager
from typing import Callable

import numpy as np

from ..brain.world import World
from ..chat.reader import Message
from ..chat.tracker import SelfFilter
from ..vision.bubbles import Rect
from .transcript import Transcript

log = logging.getLogger(__name__)

GRAY = np.full((1080, 1920, 3), 128, np.uint8)
GRAY.setflags(write=False)
BLIND = "看不清，眼前什么也看不出来"
SELF = "团子"


class SandboxDevice:
    """假设备：什么输入都不发，截图是中灰图。"""

    def screenshot(self) -> np.ndarray:
        return GRAY.copy()

    def tap(self, x, y) -> None:
        pass

    def swipe(self, x1, y1, x2, y2, duration_ms=300) -> None:
        pass

    def key(self, keycode) -> None:
        pass

    def input_text(self, text) -> None:
        pass

    def text(self, text) -> None:
        pass

    def hw_key(self, code) -> None:
        pass

    def hw_key_hold(self, code, seconds=0.0) -> None:
        pass

    def hw_key_down(self, code) -> None:
        pass

    def hw_key_up(self, code) -> None:
        pass

    def ime_shown(self) -> bool:
        return False

    def ime_enable(self, *args, **kwargs) -> None:
        pass

    def ime_disable(self, *args, **kwargs) -> None:
        pass

    def ime_current(self) -> str:
        return ""

    def screen_size(self) -> tuple[int, int]:
        return 1920, 1080


class SandboxReader:
    """冒充的发言排队，身体每圈 read() 取走。聊天记录面板永远"开着"。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.panel_closed_since: float | None = None
        self.settling = False
        self.trace_path = None

    def say(self, who: str, text: str) -> None:
        self._queue.put((who, text))

    def pending(self) -> int:
        """还没被身体读走的冒充发言有几句（沙盒判断安静了没有）。"""
        return self._queue.qsize()

    def read(self, frame, now: float) -> list[Message]:
        out = []
        while True:
            try:
                who, text = self._queue.get_nowait()
            except queue.Empty:
                return out
            out.append(Message(text, Rect(0, 0, 1, 1), now, who))

    def panel_visible(self, frame) -> bool:
        return True


class SandboxEnv:
    """身边：页面设置的好友名单 / 陌生人数 / 地名。接口同 EnvWatcher / 感知层里身体用到的部分。"""

    def __init__(self) -> None:
        self.friends: list[str] = []
        self.stranger_count = 0
        self.place = ""
        self.labels: dict = {}
        self.requests: dict = {}
        self.circles: dict = {}

    def observe(self, frame, now: float, panel_visible: bool = True) -> None:
        pass

    def nearby(self, now: float) -> list[str]:
        return list(self.friends)

    def strangers(self, now: float) -> int:
        return int(self.stranger_count)

    def unlit(self, now: float) -> int:
        return 0

    def people(self, now: float) -> list:
        return []

    def overlay(self, now: float) -> list[dict]:
        return []

    def hold(self, reason: str) -> None:
        pass

    def release(self, reason: str) -> None:
        pass

    @contextmanager
    def held(self, reason: str):
        yield

    def stop(self) -> None:
        pass


class SandboxSender:
    """团子说的话、输入气泡都写进沙盒聊天记录（clean_reply、限速、主动护栏在 body.say 里照常过）。"""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.opened = False

    def open(self) -> bool:
        if not self.opened:
            self.transcript.add("act", "（团子头顶冒出输入气泡）", SELF)
        self.opened = True
        return True

    def cancel(self) -> None:
        self.opened = False

    def send(self, text: str) -> None:
        self.transcript.add("said", text, SELF)
        self.opened = False


class SandboxEmotes:
    """假轮盘：names 都当作在轮盘上。接口同 EmotePlayer 里身体用到的部分；动作限速照常（min_interval）。"""

    def __init__(self, names: list[str], transcript: Transcript, clock: Callable[[], float], min_interval: float = 0.0) -> None:
        self.names = list(names)
        self.transcript = transcript
        self.clock = clock
        self.min_interval = min_interval
        self.last_emote = float("-inf")  # 大脑的动作冷却从这里算；反射做的不算
        self.last_any = float("-inf")

    def start(self) -> None:
        pass

    def on_wheel(self) -> list[str]:
        return list(self.names)

    def available(self, ignore_interval: bool = False) -> list[str]:
        if not ignore_interval and self.clock() - self.last_emote < self.min_interval:
            return []
        return list(self.names)

    def perform(self, name: str, reflex: bool = False) -> int:
        if name not in self.names:
            raise ValueError(f"「{name}」不在轮盘上")
        self._mark(name, reflex)
        return self.names.index(name) + 1

    def pretend(self, name: str, reflex: bool = False) -> None:
        self._mark(name, reflex)

    def _mark(self, name: str, reflex: bool) -> None:
        now = self.clock()
        self.last_any = now
        if not reflex:
            self.last_emote = now
        self.transcript.add("act", f"（团子下意识地 {name}）" if reflex else f"（团子做了 {name}）", SELF)

    def restore(self) -> None:
        pass


MOVE_NAMES = {"forward": "前", "back": "后", "left": "左", "right": "右", "W": "前", "S": "后", "A": "左", "D": "右"}


class SandboxLocomotion:
    """走路：只记一笔。方向用身体的 forward / back / left / right，也认 W/A/S/D。"""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def move(self, direction: str, steps: int = 1, max_steps: int = 3) -> str:
        key = direction if direction in MOVE_NAMES else str(direction).upper()
        if key not in MOVE_NAMES:
            raise ValueError(f"不认识的移动方向：{direction}")
        steps = max(1, min(int(steps), max_steps))
        where = MOVE_NAMES[key]
        self.transcript.add("act", f"（团子往{where}走了 {steps} 步）", SELF)
        return f"往{where}走了 {steps} 步"


class Scene:
    """团子"看到"的：页面上手写的场景文字；空着 = 看不清。"""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def describe(self, content=None) -> str:
        return self.text.strip() or BLIND


def emote_names(cfg) -> list[str]:
    """能做的动作：emotes/ 图标库里的名字；没有图标库就用 [sandbox] emotes。"""
    from ..game.wheel import EmoteLibrary

    try:
        names = EmoteLibrary(cfg.wheel.library_dir).names
    except Exception:
        log.warning("沙盒读动作图标库出错，改用 [sandbox] emotes", exc_info=True)
        names = []
    return names or list(cfg.sandbox.emotes)


def sandbox_world(cfg, sim, transcript: Transcript, scene: Scene) -> World:
    from ..chat.panel import PanelManager

    device = SandboxDevice()
    reader = SandboxReader()
    panel = PanelManager(cfg.vision, dataclasses.replace(cfg.panel, mode="always"), device, reader, lambda s: None, sim.clock)
    env = SandboxEnv()
    names = emote_names(cfg) if cfg.emotes.enabled else []
    return World(
        device=device,
        reader=reader,
        self_filter=SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix),
        panel=panel,
        env=env,
        social=None,
        emotes=SandboxEmotes(names, transcript, sim.clock, cfg.emotes.min_interval) if names else None,
        camera=None,
        locomotion=SandboxLocomotion(transcript),
        sender=SandboxSender(transcript),
        friend_checker=None,
        panels=None,
        panel_ops=None,
        clock=sim.clock,
        wall=sim.wall,
        describe=scene.describe,
        name="sandbox",
        text_only=True,
    )
