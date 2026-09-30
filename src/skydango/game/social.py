"""社交互动：好友发起牵手 / 拥抱 / 击掌时接受。

光遇里（实测）：好友头顶名字正下方有个圆圈，平时里面是 ✦；对方发起互动时换成对应的图标
（两只手 = 牵手、抱在一起 = 拥抱、举手 = 击掌、一个背着另一个 = 背背）。点这个圆圈就接受，团子会自己走过去，走的时候
图标还在，到了之后圆圈消失。键盘操作后的第一下触摸会被吞掉（只是切到触屏模式），所以要看反应补点。

分工：发现请求在 EnvWatcher 的后台扫描里顺带做（它本来就要 OCR 名字标签），
接受请求在主循环里做，免得和打字发消息同时操作设备。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import SocialConfig
from ..device.base import Device
from ..imageio import imread
from ..vision.icons import best_match, trim

log = logging.getLogger(__name__)

KIND_NAMES = {
    "hand": "牵手", "hug": "拥抱", "highfive": "击掌", "piggyback": "背背",
    "candle": "点火",  # 火焰：陌生人举着蜡烛走到团子旁边，要给团子点火（2026-09-28 用户说明）
    "stranger": "陌生人", "eye": "在看留影 / 听音乐", "shared": "共享空间",
}
IDLE = "star"  # 没有请求时圆圈里是 ✦
# 只是状态、不是请求的图标：没点火的陌生人平时是蜡烛 + 两只手；眼睛 = 在看留影蜡烛或听音乐；
# 深色实心圆里的飞人 = 在其他共享空间（点了会问要不要加入，绝不能点）
PASSIVE = {IDLE, "stranger", "eye", "shared"}


def is_request(kind: str | None) -> bool:
    """圆圈里的图标是不是在请求团子做什么（牵手、拥抱、点火……）。"""
    return kind is not None and kind not in PASSIVE
SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
RETRY = 2.0  # 去点之前那一帧没认出图标：隔这么久再试（不进整段冷却）


def cream(img: np.ndarray) -> np.ndarray:
    """圆圈里的图标是米白色：亮、不饱和。旁边角色身上的黄绿色光会被排除。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 70)).astype(np.uint8) * 255


def load_icons(directory: str | Path) -> dict[str, np.ndarray]:
    """目录下每张 png 是一种圆圈图标，文件名就是类型（star / hand / hug / highfive / candle……）。
    同一种可以录好几张，编号接在减号后面（candle.png、candle-2.png）：认出来都算 candle（见 kind_of）。"""
    icons = {}
    for path in sorted(Path(directory).glob("*.png")):
        mask = trim(cream(imread(path)))
        if mask.any():
            icons[path.stem] = mask
    return icons


@dataclass(frozen=True)
class Request:
    name: str  # 发起人的游戏昵称
    kind: str  # hand / hug / highfive / candle
    pos: tuple[int, int]  # 圆圈中心（整张截图坐标）
    seen_at: float


def touch_mode(frame: np.ndarray) -> bool:
    """触屏模式时左下角有个摇杆圈（实测中心约 (230, 847)、半径约 40，1920×1080）；键盘模式下没有。
    圆环一圈比内外侧亮 30~40，没有时接近 0。"""
    height, width = frame.shape[:2]
    k = height / 1080
    cx, cy, r = round(230 * width / 1920), round(847 * k), 40 * k
    x1, y1 = max(0, int(cx - r - 15)), max(0, int(cy - r - 15))
    patch = cv2.cvtColor(frame[y1 : int(cy + r + 15), x1 : int(cx + r + 15)], cv2.COLOR_BGR2GRAY).astype(float)
    if patch.size == 0:
        return False
    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    d = np.hypot(xx - (cx - x1), yy - (cy - y1))
    ring = patch[(d >= r - 3) & (d <= r + 3)]
    side = patch[((d >= r - 12) & (d <= r - 7)) | ((d >= r + 7) & (d <= r + 12))]
    return ring.size > 0 and side.size > 0 and ring.mean() - side.mean() > 15


def kind_of(name: str) -> str:
    """模板名 → 图标类型：candle-2 → candle。"""
    return name.split("-", 1)[0]


def _moved(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) >= 25 or abs(a[1] - b[1]) >= 25


class IconClassifier:
    def __init__(self, icons: dict[str, np.ndarray], min_score: float = 0.75) -> None:
        self.icons = icons
        self.min_score = min_score

    def classify(self, region: np.ndarray) -> tuple[str | None, float]:
        """一小块画面（圆圈附近）里是哪种图标；认不出返回 (None, 最高分)。"""
        if region.size == 0:
            return None, 0.0
        mask = cream(region)
        best, score = None, 0.0
        for kind, icon in self.icons.items():
            s = best_match(mask, icon, SCALES).score
            if s > score:
                best, score = kind_of(kind), s
        return (best, score) if score >= self.min_score else (None, score)

    def find(self, frame: np.ndarray, kind: str, near: tuple[int, int], radius: int = 150) -> tuple[int, int] | None:
        """在 near 附近找某种图标（人可能走动了一点），返回中心；找不到返回 None。"""
        icons = [icon for name, icon in self.icons.items() if kind_of(name) == kind]
        if not icons:
            return None
        height, width = frame.shape[:2]
        x1, y1 = max(0, near[0] - radius), max(0, near[1] - radius)
        x2, y2 = min(width, near[0] + radius), min(height, near[1] + radius)
        mask = cream(frame[y1:y2, x1:x2])
        match = max((best_match(mask, icon, SCALES) for icon in icons), key=lambda m: m.score)
        if match.score < self.min_score:
            return None
        # 同一个位置别的图标更像：说明认错了（比如拥抱被当成牵手）
        region = frame[max(0, y1 + match.y - 56) : y1 + match.y + 56, max(0, x1 + match.x - 56) : x1 + match.x + 56]
        if self.classify(region)[0] != kind:
            return None
        return x1 + match.x, y1 + match.y


class SocialHandler:
    def __init__(
        self,
        device: Device,
        cfg: SocialConfig,
        classifier: IconClassifier,
        friends: Callable[[], list[str]],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        dry_run: bool = False,
        panel=None,  # chat.panel.PanelManager：点屏幕会关掉聊天记录面板，借着点、完了归还
    ) -> None:
        self.dry_run = dry_run  # 只打印“将会接受”，不点屏幕
        self.panel = panel
        self.device = device
        self.cfg = cfg
        self.classifier = classifier
        self.friends = friends
        self.sleep = sleep
        self.clock = clock
        self._done: dict[tuple[str, str], float] = {}  # (谁, 什么) → 什么时候处理过，冷却用
        self.last: tuple[str, str, float] | None = None  # 最近一次接受的（谁, 什么, 时间）
        self.policy: dict[tuple[str, str], bool] = {}  # 大脑定的规则：(谁, 哪种) → 接不接；谁 = 昵称 / "*" / "stranger"
        self._backoff_until = float("-inf")

    def allowed(self, req: Request) -> bool:
        friend = req.name in self.friends()
        for who in (req.name, "*" if friend else "stranger"):  # 点名的优先
            for kind in (req.kind, "*"):
                if (who, kind) in self.policy:
                    return self.policy[(who, kind)]
        if friend:
            return req.kind in self.cfg.accept_friends
        return req.kind in self.cfg.accept_strangers

    def set_policy(self, who: str, kind: str, accept: bool) -> None:
        self.policy[(who, kind)] = accept

    def describe_policy(self) -> str:
        """给大脑看的当前规则。"""
        base = "好友默认接受：" + ("、".join(KIND_NAMES.get(k, k) for k in self.cfg.accept_friends) or "都不接")
        rules = []
        for (who, kind), ok in self.policy.items():
            person = {"*": "所有好友", "stranger": "陌生人"}.get(who, who)
            what = "所有请求" if kind == "*" else KIND_NAMES.get(kind, kind)
            rules.append(f"{person}的{what}：{'接' if ok else '不接'}")
        return base + ("；" + "；".join(rules) if rules else "")

    def handle(self, requests: dict[str, Request], now: float) -> list[str]:
        """主循环每轮调一次：把还新鲜、允许、不在冷却里的请求接受掉。返回处理了哪些（“谁:什么”）。"""
        handled = []
        if now < self._backoff_until:
            return handled
        for req in list(requests.values()):
            key = (req.name, req.kind)
            if now - req.seen_at > self.cfg.max_age or not self.allowed(req):
                continue
            if now - self._done.get(key, float("-inf")) < self.cfg.cooldown:
                continue
            try:
                if self.device.ime_shown():  # 正在打字：点屏幕会打断输入，等打完再说
                    return handled
                self._done[key] = now
                if self.dry_run:
                    log.info("[dry-run] 将会接受 %s 的%s", req.name, KIND_NAMES.get(req.kind, req.kind))
                    continue
                result = self.accept(req)
                if result is None:  # 没点：别进整段冷却，过 RETRY 秒还在就再试（实测亮背景上时认时不认，陌生人的点火就这么错过了）
                    self._done[key] = now - self.cfg.cooldown + RETRY
                elif result:
                    self.last = (req.name, req.kind, now)
                    handled.append(f"{req.name}:{req.kind}")
            except Exception:  # 实测 adb 会连续失败一阵：退避，别每帧都起一个 adb 进程
                self._backoff_until = now + self.cfg.error_backoff
                log.warning("处理互动请求时出错，%.0f 秒内不再处理", self.cfg.error_backoff, exc_info=True)
                return handled
        return handled

    def accept(self, req: Request) -> bool | None:
        """点圆圈接受请求；点之前这一帧就没认出图标返回 None（没点，handle 过 RETRY 秒再试）。真正生效的点击永远只有一下：牵上之后再点就会松手（实测断手）。

        键盘模式下（左下角没有摇杆圈）第一下触摸只会切到触屏模式、顺带关掉聊天面板 → 先点一下"唤醒"，
        再点一下真正的；点完只等不补点，图标消失就是完成（团子会自己走过去，站得近时图标可能原地停一会儿）。
        """
        what = KIND_NAMES.get(req.kind, req.kind)
        frame = self.device.screenshot()
        pos = self.classifier.find(frame, req.kind, req.pos)
        if pos is None:
            log.info("%s 的%s请求已经没了（这一帧没认出图标，%.0f 秒后再看）", req.name, what, RETRY)
            return None
        with self.panel.borrow("social", close=False) if self.panel is not None else nullcontext():
            if not touch_mode(frame):
                self.device.tap(*pos)  # 只切到触屏模式
                self.sleep(self.cfg.check_delay)
                frame = self.device.screenshot()
                found = self.classifier.find(frame, req.kind, pos)
                if found is None or not touch_mode(frame) or _moved(found, pos):
                    # 这一下其实已经生效了（或者请求没了）：别再点
                    return self._wait_done(req, found, what)
                pos = found
            self.device.tap(*pos)
            return self._wait_done(req, pos, what)

    def _wait_done(self, req: Request, pos: tuple[int, int] | None, what: str) -> bool:
        deadline = self.clock() + self.cfg.accept_timeout
        while pos is not None and self.clock() < deadline:
            self.sleep(self.cfg.check_delay)
            pos = self.classifier.find(self.device.screenshot(), req.kind, pos)
        if pos is None:
            log.info("接受了 %s 的%s", req.name, what)
            return True
        log.warning("点了 %s 的%s请求，等了 %.0f 秒图标还在，不再补点（再点会取消）", req.name, what, self.cfg.accept_timeout)
        return False

    def describe(self, now: float) -> str:
        if not self.last or now - self.last[2] > self.cfg.remember:
            return ""
        name, kind, t = self.last
        return f"- {int(now - t)} 秒前你接受了 {name} 的{KIND_NAMES.get(kind, kind)}（是你自己点的，不用解释）"
