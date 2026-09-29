"""面板识别：画面上开着哪些面板（聊天记录面板、动作面板、轮盘编辑、好友树、各种弹框……）。

只看不动：每个已知面板一张"特征卡"（assets/panels/<名字>/card.toml + 模板小图），每帧几毫秒判断开没开（快看）；
面板上写了什么、有哪些按钮靠 OCR（细读）；认不出来的弹框由"通用兜底"按"有按钮词 + 一段正文"认成不认识的面板。
动手（关面板、按按钮）在 game/panels.py。设计见 docs/superpowers/specs/2026-09-29-panels-design.md。
"""

from __future__ import annotations

import logging
import re
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import PanelsConfig
from ..imageio import imread
from ..brain.images import difference, thumb
from .bubbles import Rect, roi_rect
from .icons import best_match, silhouette, trim
from .ocr import OcrEngine, OcrLine

log = logging.getLogger(__name__)

CHAT = "chat_log"  # 聊天记录面板：不算遮挡，开关沿用 Body._watch_panel
UNKNOWN = "unknown"  # 通用兜底认出的不认识的面板
ACTIONS = ("say", "emote", "camera", "check_friend", "social")  # 遮挡护栏 clear_view 的操作名
KINDS = ("template", "dark", "builtin", "text")

_TOP_KEYS = {"label", "verified", "layer", "region", "confirm_frames", "allows", "features", "close", "buttons"}
_FEATURE_KEYS = {"kind", "roi", "image", "threshold", "max_value", "max_std", "name", "any"}
_CLOSE_KEYS = {"ways", "auto"}
_BUTTON_KEYS = {"roi", "allow", "never", "title_roi"}
_CLOSE_WAY = re.compile(r"^(key:\d+|tap:[0-9.]+,[0-9.]+|esc|button:.+)$")


class CardError(ValueError):
    """特征卡写错了：信息里带卡片名和出错的配置项。"""


@dataclass(frozen=True)
class Feature:
    kind: str  # template / dark / builtin / text
    roi: tuple[float, ...] = ()  # 归一化 [x1, y1, x2, y2]；text 不用
    image: str = ""  # template：卡片目录里的文件名
    template: np.ndarray | None = field(default=None, compare=False, repr=False)  # 已 trim 的剪影；图不存在为 None
    threshold: float = 0.8
    max_value: int = 70
    max_std: int = 40
    name: str = ""  # builtin 的函数名
    any: tuple[str, ...] = ()  # text 的关键字


@dataclass(frozen=True)
class Card:
    name: str
    label: str
    verified: bool
    layer: int
    region: tuple[float, float, float, float]
    confirm_frames: int
    allows: tuple[str, ...]
    features: tuple[Feature, ...]
    close_ways: tuple[str, ...]
    close_auto: bool
    buttons_roi: tuple[float, ...] = ()
    allow: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    title_roi: tuple[float, ...] = ()

    @property
    def quick(self) -> tuple[Feature, ...]:
        """快看用的特征（每帧判断）。"""
        return tuple(f for f in self.features if f.kind != "text")

    @property
    def texts(self) -> tuple[Feature, ...]:
        """要细读才知道的特征。"""
        return tuple(f for f in self.features if f.kind == "text")


@dataclass(frozen=True)
class Panel:
    name: str  # 卡片目录名；不认识的面板是 UNKNOWN
    label: str  # 给人 / 大脑看的名字
    box: Rect  # 面板占的区域（整张图坐标）
    verified: bool  # 卡片核对过没有
    layer: int  # 叠放顺序，越大越靠上
    allows: tuple[str, ...] = ()  # 这个面板开着时仍然可以做的操作

    def describe(self) -> str:
        return self.label + ("" if self.verified or self.name == UNKNOWN else "（未核对）")


@dataclass(frozen=True)
class Button:
    text: str
    box: Rect
    kind: str  # retreat（撤退类，可直接按）/ allow（卡片登记过）/ other（要主人放行）/ never（永远不按）


@dataclass(frozen=True)
class PanelReading:
    panel: Panel
    title: str
    text: str  # 正文（标题、按钮以外的文字，按行用换行拼起来）
    buttons: tuple[Button, ...]
    t: float


@dataclass(frozen=True)
class PanelChange:
    kind: str  # open / close
    panel: Panel
    reading: PanelReading | None = None  # 细读才认出的面板（不认识的、文字卡）开的时候带上


@dataclass(frozen=True)
class PanelState:
    panels: tuple[Panel, ...] = ()  # 按 layer 从大到小

    def top(self) -> Panel | None:
        """最上面的一个（不算聊天记录面板）。"""
        return next((p for p in self.panels if p.name != CHAT), None)

    def others(self) -> tuple[Panel, ...]:
        """聊天记录面板以外的。"""
        return tuple(p for p in self.panels if p.name != CHAT)


# ---- 特征卡加载 ----
def load_template(path: Path) -> np.ndarray | None:
    """读模板图 → 剪影 → 裁掉空白；文件不存在返回 None。"""
    if not path.exists():
        return None
    return trim(silhouette(imread(path)))


def _check_keys(name: str, where: str, data: dict, allowed: set[str]) -> None:
    for key in data:
        if key not in allowed:
            raise CardError(f"面板卡片 {name}: {where}不认识的配置项 {key}")


def _floats(name: str, key: str, value, n: int | None = 4) -> tuple[float, ...]:
    if not isinstance(value, list) or (n is not None and len(value) != n) or not all(isinstance(v, (int, float)) for v in value):
        raise CardError(f"面板卡片 {name}: {key} 应该是 {n} 个数（归一化坐标）")
    return tuple(float(v) for v in value)


def _strings(name: str, key: str, value) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CardError(f"面板卡片 {name}: {key} 应该是字符串列表")
    return tuple(value)


def _feature(name: str, folder: Path, data: dict, verified: bool) -> Feature:
    if not isinstance(data, dict):
        raise CardError(f"面板卡片 {name}: features 里每一项应该是一个表")
    _check_keys(name, "features 里", data, _FEATURE_KEYS)
    kind = data.get("kind")
    if kind not in KINDS:
        raise CardError(f"面板卡片 {name}: 不认识的特征种类 {kind}（可以用 {' / '.join(KINDS)}）")
    roi = _floats(name, "features.roi", data["roi"]) if "roi" in data else ()
    if kind in ("template", "dark") and not roi:
        raise CardError(f"面板卡片 {name}: {kind} 特征要写 roi")
    template = None
    image = str(data.get("image", ""))
    if kind == "template":
        if not image:
            raise CardError(f"面板卡片 {name}: template 特征要写 image")
        template = load_template(folder / image)
        if template is None:
            if verified:
                raise CardError(f"面板卡片 {name}: 找不到模板图 {image}（已核对的卡不能缺图）")
            log.warning("面板卡片 %s 还没有模板图 %s（未核对）：这个特征先当不命中", name, image)
    if kind == "builtin" and not data.get("name"):
        raise CardError(f"面板卡片 {name}: builtin 特征要写 name")
    if kind == "text" and not data.get("any"):
        raise CardError(f"面板卡片 {name}: text 特征要写 any（关键字列表）")
    return Feature(
        kind=kind,
        roi=roi,
        image=image,
        template=template,
        threshold=float(data.get("threshold", 0.8)),
        max_value=int(data.get("max_value", 70)),
        max_std=int(data.get("max_std", 40)),
        name=str(data.get("name", "")),
        any=_strings(name, "features.any", data.get("any", [])),
    )


def load_card(folder: Path) -> Card:
    name = folder.name
    try:
        with (folder / "card.toml").open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CardError(f"面板卡片 {name}: card.toml 读不了（{exc}）") from None
    _check_keys(name, "", data, _TOP_KEYS)
    for key in ("label", "region"):
        if key not in data:
            raise CardError(f"面板卡片 {name}: 缺少 {key}")
    verified = bool(data.get("verified", False))
    features = data.get("features", [])
    if not isinstance(features, list) or not features:
        raise CardError(f"面板卡片 {name}: 至少要有一个 features")
    allows = _strings(name, "allows", data.get("allows", []))
    for action in allows:
        if action not in ACTIONS:
            raise CardError(f"面板卡片 {name}: allows 里不认识的操作 {action}（可以用 {' / '.join(ACTIONS)}）")
    close = data.get("close", {})
    buttons = data.get("buttons", {})
    if not isinstance(close, dict) or not isinstance(buttons, dict):
        raise CardError(f"面板卡片 {name}: close / buttons 应该是表")
    _check_keys(name, "close 里", close, _CLOSE_KEYS)
    _check_keys(name, "buttons 里", buttons, _BUTTON_KEYS)
    ways = _strings(name, "close.ways", close.get("ways", []))
    for way in ways:
        if not _CLOSE_WAY.match(way):
            raise CardError(f"面板卡片 {name}: 不认识的关法 {way}（key:<键码> / tap:<x>,<y> / esc / button:<按钮文字>）")
    return Card(
        name=name,
        label=str(data["label"]),
        verified=verified,
        layer=int(data.get("layer", 50)),
        region=_floats(name, "region", data["region"]),  # type: ignore[arg-type]
        confirm_frames=max(1, int(data.get("confirm_frames", 2))),
        allows=allows,
        features=tuple(_feature(name, folder, f, verified) for f in features),
        close_ways=ways,
        close_auto=bool(close.get("auto", False)),
        buttons_roi=_floats(name, "buttons.roi", buttons["roi"]) if "roi" in buttons else (),
        allow=_strings(name, "buttons.allow", buttons.get("allow", [])),
        never=_strings(name, "buttons.never", buttons.get("never", [])),
        title_roi=_floats(name, "buttons.title_roi", buttons["title_roi"]) if "title_roi" in buttons else (),
    )


def load_cards(directory: str | Path) -> list[Card]:
    """读 directory 下每个子目录的 card.toml（按目录名排序，跳过 _ 开头的）。"""
    root = Path(directory)
    return [load_card(d) for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_")]


# ---- 细读 ----
CLOSE_MARKS = ("×", "X", "x")
READ_CACHE_DIFF = 0.02  # 面板区域缩略图差异低于这个就复用上次的细读


def classify(text: str, cfg: PanelsConfig, allow: Sequence[str] = (), never: Sequence[str] = ()) -> str:
    """按钮类别：never（放行也不按）> retreat（撤退类）> allow（卡片登记）> other（要主人放行）。词在按钮文字里就算命中。"""
    t = text.strip()
    if any(w in t for w in (*cfg.never, *never)):
        return "never"
    if t in CLOSE_MARKS or any(w in t for w in cfg.retreat):
        return "retreat"
    if any(w in t for w in allow):
        return "allow"
    return "other"


def _inside(box: Rect, area: Rect) -> bool:
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    return area.x <= cx < area.x2 and area.y <= cy < area.y2


def _is_button(line: OcrLine, card: Card | None, cfg: PanelsConfig, area: Rect | None) -> bool:
    text = line.text.strip()
    if not text or len(text) > cfg.button_max_chars:
        return False
    if area is not None:  # 卡片划了按钮区域：里面的短字都算
        return _inside(line.box, area)
    words = (*cfg.button_words, *cfg.retreat, *cfg.never, *(card.allow if card else ()), *(card.never if card else ()))
    return text in CLOSE_MARKS or any(w in text for w in words)


def split_reading(
    lines: list[OcrLine], panel: Panel, card: Card | None, cfg: PanelsConfig, width: int, height: int, now: float
) -> PanelReading:
    """OCR 行（整张图坐标）→ 标题、正文、按钮。"""
    lines = sorted((ln for ln in lines if ln.text.strip()), key=lambda ln: (ln.box.y, ln.box.x))
    area = roi_rect(list(card.buttons_roi), width, height) if card and card.buttons_roi else None
    buttons = [ln for ln in lines if _is_button(ln, card, cfg, area)]
    rest = [ln for ln in lines if ln not in buttons]
    title = ""
    if card and card.title_roi:
        title_area = roi_rect(list(card.title_roi), width, height)
        titles = [ln for ln in rest if _inside(ln.box, title_area)]
        title = "".join(ln.text.strip() for ln in titles)
        rest = [ln for ln in rest if ln not in titles]
    elif rest:
        title, rest = rest[0].text.strip(), rest[1:]
    allow, never = (card.allow, card.never) if card else ((), ())
    return PanelReading(
        panel,
        title,
        "\n".join(ln.text.strip() for ln in rest),
        tuple(Button(ln.text.strip(), ln.box, classify(ln.text, cfg, allow, never)) for ln in buttons),
        now,
    )


def describe_reading(reading: PanelReading) -> str:
    """一行说清面板写了什么：「标题」正文前 40 字，按钮：取消、加入。"""
    text = reading.text.replace("\n", " ")
    head = f"「{reading.title}」" if reading.title else ""
    buttons = "、".join(b.text for b in reading.buttons) or "没认出按钮"
    return f"{head}{text[:40]}，按钮：{buttons}"


# ---- 快看 ----
TEMPLATE_SCALES = (0.9, 1.0, 1.1)  # 模板按 1080 高裁的；再乘上 截图高 / 1080


def check_feature(frame: np.ndarray, feature: Feature, builtins: dict[str, Callable[[np.ndarray], bool]]) -> tuple[bool, float]:
    """一个特征命中没有：(命中, 分数)。template 的分数是最高匹配分；dark 是 V 中位数；builtin 是 1 / 0；text 要细读，这里恒不中。"""
    height, width = frame.shape[:2]
    if feature.kind == "template":
        if feature.template is None:
            return False, 0.0
        crop = roi_rect(list(feature.roi), width, height).crop(frame)
        k = height / 1080
        match = best_match(silhouette(crop), feature.template, [k * s for s in TEMPLATE_SCALES])
        return match.score >= feature.threshold, match.score
    if feature.kind == "dark":
        crop = roi_rect(list(feature.roi), width, height).crop(frame)
        value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
        median = float(np.median(value))
        return median <= feature.max_value and float(value.std()) <= feature.max_std, median
    if feature.kind == "builtin":
        fn = builtins.get(feature.name)
        hit = bool(fn(frame)) if fn is not None else False
        return hit, 1.0 if hit else 0.0
    return False, 0.0


@dataclass(frozen=True)
class CardCheck:
    """一张卡在一帧上的判断（不去抖），panels scan 命令打印用。"""

    card: Card
    hits: tuple[tuple[Feature, bool, float], ...]  # 每个快看特征：(特征, 命中, 分数)
    open: bool


class PanelWatcher:
    """身体每圈调 observe(frame, now)：哪些面板开着（去抖后），开 / 关的变化用 pop_changes() 取走。"""

    def __init__(
        self,
        cfg: PanelsConfig,
        cards: list[Card],
        ocr: OcrEngine,
        builtins: dict[str, Callable[[np.ndarray], bool]],
        clock: Callable[[], float] = time.monotonic,
        background: bool = True,
    ) -> None:
        for card in cards:
            for f in card.quick:
                if f.kind == "builtin" and f.name not in builtins:
                    raise ValueError(f"面板卡片 {card.name}: 没有叫 {f.name} 的内置判断")
        self.cfg = cfg
        self.cards: dict[str, Card] = {c.name: c for c in cards}
        self.ocr = ocr
        self.builtins = builtins
        self.clock = clock
        self.background = background
        self.state = PanelState()
        self.readings: dict[str, PanelReading] = {}  # 各面板最近的细读
        self._streak: dict[str, int] = {}  # 正数 = 连续命中几帧，负数 = 连续不中几帧
        self._open: dict[str, Panel] = {}  # 特征卡认出、开着的
        self._changes: list[PanelChange] = []
        self._expect: Counter[str] = Counter()
        self._read_cache: dict[str, tuple[Rect, np.ndarray, PanelReading]] = {}

    # ---- 身体线程 ----
    def observe(self, frame: np.ndarray, now: float) -> PanelState:
        height, width = frame.shape[:2]
        for card in self.cards.values():
            if card.quick:
                hit = all(check_feature(frame, f, self.builtins)[0] for f in card.quick)
                self._step(card, hit, width, height)
        self._rebuild()
        return self.state

    def explain(self, frame: np.ndarray) -> list[CardCheck]:
        out = []
        for card in self.cards.values():
            hits = tuple((f, *check_feature(frame, f, self.builtins)) for f in card.quick)
            out.append(CardCheck(card, hits, bool(hits) and all(h for _, h, _ in hits)))
        return out

    def read(self, frame: np.ndarray, panel: Panel, now: float) -> PanelReading:
        """细读一个面板：OCR 它的区域，拆成标题、正文、按钮。画面没怎么变时复用上次的结果。"""
        crop = panel.box.crop(frame)
        small = thumb(crop)
        cached = self._read_cache.get(panel.name)
        if cached is not None and cached[0] == panel.box and difference(cached[1], small) < READ_CACHE_DIFF:
            return cached[2]
        height, width = frame.shape[:2]
        lines = [
            OcrLine(ln.text, ln.score, Rect(ln.box.x + panel.box.x, ln.box.y + panel.box.y, ln.box.w, ln.box.h))
            for ln in self.ocr.recognize(crop)
        ]
        reading = split_reading(lines, panel, self.cards.get(panel.name), self.cfg, width, height, now)
        self._read_cache[panel.name] = (panel.box, small, reading)
        self.readings[panel.name] = reading
        return reading

    def present(self, frame: np.ndarray, panel: Panel) -> bool:
        """单帧判断这个面板还在不在（不去抖；PanelOps 关面板、按按钮后确认用）。"""
        card = self.cards.get(panel.name)
        if card is None or not card.quick:
            return False
        return all(check_feature(frame, f, self.builtins)[0] for f in card.quick)

    def mark_closed(self, name: str) -> None:
        """PanelOps 确认关掉了：马上从状态里去掉，排一个 close 变化。"""
        panel = self._open.pop(name, None)
        if panel is None:
            return
        self._streak[name] = 0
        self._emit("close", panel)
        self._rebuild()

    @property
    def expected(self) -> set[str]:
        return {name for name, n in self._expect.items() if n > 0}

    @contextmanager
    def expect(self, name: str | None) -> Iterator[None]:
        """身体自己要打开这个面板（换轮盘、看好友树）：期间它不算遮挡、开关不发变化；结束时和进来时比，开关变了才补一个变化。"""
        if name is None:
            yield
            return
        was_open = name in self._open
        self._expect[name] += 1
        try:
            yield
        finally:
            self._expect[name] -= 1
            if self._expect[name] <= 0:
                del self._expect[name]
                panel = self._open.get(name)
                if panel is not None and not was_open:
                    self._emit("open", panel)
                elif panel is None and was_open:
                    self._emit("close", Panel(name, self.cards[name].label, Rect(0, 0, 1, 1), self.cards[name].verified, self.cards[name].layer))

    def blocking(self, action: str) -> list[Panel]:
        """挡着这个操作的面板（身体自己打开的、卡片 allows 里有这个操作的不算），从上到下。"""
        expected = self.expected
        return [p for p in self.state.panels if p.name not in expected and action not in p.allows]

    def pop_changes(self) -> list[PanelChange]:
        out, self._changes = self._changes, []
        return out

    def close(self) -> None:
        """停后台线程（通用兜底）。"""

    # ---- 内部 ----
    def _panel(self, card: Card, width: int, height: int) -> Panel:
        return Panel(card.name, card.label, roi_rect(list(card.region), width, height), card.verified, card.layer, card.allows)

    def _step(self, card: Card, hit: bool, width: int, height: int) -> None:
        streak = self._streak.get(card.name, 0)
        streak = max(streak, 0) + 1 if hit else min(streak, 0) - 1
        self._streak[card.name] = streak
        if card.name not in self._open and streak >= card.confirm_frames:
            panel = self._open[card.name] = self._panel(card, width, height)
            self._emit("open", panel)
        elif card.name in self._open and -streak >= card.confirm_frames:
            self._emit("close", self._open.pop(card.name))

    def _emit(self, kind: str, panel: Panel, reading: PanelReading | None = None) -> None:
        if panel.name not in self.expected:
            self._changes.append(PanelChange(kind, panel, reading))

    def _rebuild(self) -> None:
        panels = sorted(self._open.values(), key=lambda p: -p.layer)
        self.state = PanelState(tuple(panels))
