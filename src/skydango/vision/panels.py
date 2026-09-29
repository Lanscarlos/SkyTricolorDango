"""面板识别：画面上开着哪些面板（聊天记录面板、动作面板、轮盘编辑、好友树、各种弹框……）。

只看不动：每个已知面板一张"特征卡"（assets/panels/<名字>/card.toml + 模板小图），每帧几毫秒判断开没开（快看）；
面板上写了什么、有哪些按钮靠 OCR（细读）；认不出来的弹框由"通用兜底"按"有按钮词 + 一段正文"认成不认识的面板。
动手（关面板、按按钮）在 game/panels.py。设计见 docs/superpowers/specs/2026-09-29-panels-design.md。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import tomllib
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import PanelsConfig
from ..imageio import imread
from ..brain.images import difference, is_black, thumb
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
_PUNCT = re.compile(r"[\s，。！？、…~～,.!?:：;；\-—\"'“”‘’()（）\[\]【】]+")


def _norm(text: str) -> str:
    """去掉空白和标点：「取消。」「确定！」和「取消」「确定」算同一个词。"""
    return _PUNCT.sub("", text)


def classify(text: str, cfg: PanelsConfig, allow: Sequence[str] = (), never: Sequence[str] = ()) -> str:
    """按钮类别：never（放行也不按）> retreat（撤退类）> allow（卡片登记）> other（要主人放行）。词在按钮文字里就算命中。"""
    t = text.strip()
    if any(w in t for w in (*cfg.never, *never)):  # never 用"包含"：宁可多拦
        return "never"
    if t in CLOSE_MARKS or _norm(t) in cfg.retreat:  # 撤退类要整个按钮就是这个词：「返回遇境」「取消好友」不算
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


def looks_like_panel(lines: list[OcrLine], cfg: PanelsConfig) -> bool:
    """通用兜底的判定：至少一个按钮（整行就是按钮词：确定 / 取消……）+ 其余文字够一段正文。
    名字标签、零星字、句子里带着按钮词的（「好的，走吧」、昵称「知道了吗」）都不算按钮。"""
    words = {*cfg.button_words, *cfg.retreat, *cfg.never}
    has_button = False
    chars = 0
    for ln in lines:
        text = ln.text.strip()
        if text in CLOSE_MARKS or _norm(text) in words:
            has_button = True
        else:
            chars += len(text)
    return has_button and chars >= cfg.unknown_min_chars


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


_FAILED = object()  # 通用兜底扫描出错：这次没结论
CLOSE_AREA = (0.9, 0.0, 1.0, 0.1)  # 右上角：找通用的 × 模板
UNKNOWN_LAYER = 100


class PanelWatcher:
    """身体每圈调 observe(frame, now)：哪些面板开着（去抖后），开 / 关的变化用 pop_changes() 取走。

    特征卡每帧判断；不认识的面板（和只靠文字认的卡）由通用兜底在画面大变 / 右上角出现 × / 每 scan_interval 秒时
    OCR 屏幕中部认出来（background=True 时在后台线程里做，结果下一次 observe 合并）。
    """

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
        self._close_mark = Feature("template", CLOSE_AREA, "close.png", load_template(Path(cfg.cards_dir) / "_common" / "close.png"))
        self._clean: np.ndarray | None = None  # 没开面板时的画面缩略图（通用兜底比较用）
        self._last_scan = float("-inf")
        self._found: Panel | None = None  # 通用兜底认出的面板（不认识的，或文字卡）
        self._found_thumb: np.ndarray | None = None
        self._confirmed_at = float("-inf")
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._future_thumb: np.ndarray | None = None
        self._future_gen = 0
        self._gen = 0  # mark_closed 时加 1：关之前提交的后台扫描结果作废（不然会把刚关的面板"复活"）
        self._ocr_lock = threading.Lock()  # 身体线程（read / present）和后台扫描共用一个 OCR 实例

    # ---- 身体线程 ----
    def observe(self, frame: np.ndarray, now: float) -> PanelState:
        height, width = frame.shape[:2]
        if not is_black(frame):  # 整屏黑（切场景）什么面板都看不出来：别把黑当成暗底面板，状态先不动
            for card in self.cards.values():
                if card.quick:
                    hit = all(check_feature(frame, f, self.builtins)[0] for f in card.quick)
                    self._step(card, hit, width, height)
        self._unknown(frame, now)
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
        lines = self._ocr_lines(frame, panel.box, self._chat_box())
        reading = split_reading(lines, panel, self.cards.get(panel.name), self.cfg, width, height, now)
        self._read_cache[panel.name] = (panel.box, small, reading)
        self.readings[panel.name] = reading
        return reading

    def present(self, frame: np.ndarray, panel: Panel) -> bool:
        """单帧判断这个面板还在不在（不去抖；PanelOps 关面板、按按钮后确认用）。靠文字认的面板要同步 OCR 一次。"""
        card = self.cards.get(panel.name)
        if panel.name == UNKNOWN or (card is not None and card.texts):
            return looks_like_panel(self._ocr_lines(frame, panel.box, self._chat_box()), self.cfg)
        if card is None or not card.quick:
            return False
        return all(check_feature(frame, f, self.builtins)[0] for f in card.quick)

    def find_close(self, frame: np.ndarray) -> tuple[int, int] | None:
        """右上角的通用 ×（assets/panels/_common/close.png，没录就一直是 None）在哪，整张图坐标。"""
        tpl = self._close_mark.template
        if tpl is None:
            return None
        height, width = frame.shape[:2]
        area = roi_rect(list(CLOSE_AREA), width, height)
        k = height / 1080
        match = best_match(silhouette(area.crop(frame)), tpl, [k * s for s in TEMPLATE_SCALES])
        return (area.x + match.x, area.y + match.y) if match.score >= self._close_mark.threshold else None

    def mark_closed(self, name: str) -> None:
        """PanelOps 确认关掉了：马上从状态里去掉，排一个 close 变化。"""
        if self._found is not None and self._found.name == name:
            self._emit("close", self._found)
            self._found = None
            self._gen += 1
            self._rebuild()
            return
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
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # ---- 内部 ----
    def panel_for(self, card: Card, width: int, height: int) -> Panel:
        """这张卡开着时的 Panel（框 = 卡片 region）。"""
        return Panel(card.name, card.label, roi_rect(list(card.region), width, height), card.verified, card.layer, card.allows)

    def _step(self, card: Card, hit: bool, width: int, height: int) -> None:
        streak = self._streak.get(card.name, 0)
        streak = max(streak, 0) + 1 if hit else min(streak, 0) - 1
        self._streak[card.name] = streak
        if card.name not in self._open and streak >= card.confirm_frames:
            panel = self._open[card.name] = self.panel_for(card, width, height)
            self._emit("open", panel)
        elif card.name in self._open and -streak >= card.confirm_frames:
            self._emit("close", self._open.pop(card.name))

    def _emit(self, kind: str, panel: Panel, reading: PanelReading | None = None) -> None:
        if panel.name not in self.expected:
            self._changes.append(PanelChange(kind, panel, reading))

    def _rebuild(self) -> None:
        panels = list(self._open.values()) + ([self._found] if self._found is not None else [])
        self.state = PanelState(tuple(sorted(panels, key=lambda p: -p.layer)))

    # ---- 通用兜底 ----
    def _unknown(self, frame: np.ndarray, now: float) -> None:
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            if self._future_gen == self._gen:  # 提交之后关过面板：这次的结果是关之前的画面，作废
                self._apply(future.result(), self._future_thumb, now)
        if self._found is not None and now - self._confirmed_at > self.cfg.unknown_ttl:
            self._emit("close", self._found)  # 太久没再确认：当它关了，免得一直挡着
            self._found = None
        if is_black(frame):
            return
        small = thumb(frame)
        if self._future is None and self._should_scan(frame, small, now):
            self._last_scan = now
            chat = self._chat_box()
            if self.background:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="panels")
                self._future = self._executor.submit(self._scan, frame.copy(), now, chat)
                self._future_thumb, self._future_gen = small, self._gen
            else:
                self._apply(self._scan(frame, now, chat), small, now)
        clean = not any(p.name != CHAT for p in self._open.values()) and self._found is None
        if clean and (self._clean is None or difference(self._clean, small) < self.cfg.change):
            self._clean = small

    def _should_scan(self, frame: np.ndarray, small: np.ndarray, now: float) -> bool:
        cfg = self.cfg
        if now - self._last_scan < cfg.unknown_cooldown:
            return False
        if now - self._last_scan >= cfg.scan_interval:
            return True
        if self._clean is not None and difference(self._clean, small) >= cfg.change:
            return True
        if self._found is not None and self._found_thumb is not None:
            if difference(self._found_thumb, thumb(self._found.box.crop(frame))) >= cfg.change:
                return True
        return check_feature(frame, self._close_mark, {})[0]

    def _chat_box(self) -> Rect | None:
        """聊天记录面板开着时它的区域：里面的字（团子自己的「好的」、别人的消息）不算别的面板的。"""
        chat = self._open.get(CHAT)
        return chat.box if chat is not None else None

    def _ocr_lines(self, frame: np.ndarray, area: Rect, chat: Rect | None) -> list[OcrLine]:
        """OCR 一块区域，坐标换回整张图，去掉空行和落在聊天记录面板里的行。"""
        with self._ocr_lock:
            found = self.ocr.recognize(area.crop(frame))
        lines = [
            OcrLine(ln.text, ln.score, Rect(ln.box.x + area.x, ln.box.y + area.y, ln.box.w, ln.box.h))
            for ln in found
            if ln.text.strip()
        ]
        return [ln for ln in lines if chat is None or not _inside(ln.box, chat)]

    def _scan(self, frame: np.ndarray, now: float, chat: Rect | None = None):
        """OCR 屏幕中部，认出面板返回 (Panel, PanelReading, 面板区域缩略图)，没有返回 None，出错返回 _FAILED。"""
        try:
            height, width = frame.shape[:2]
            lines = self._ocr_lines(frame, roi_rect(self.cfg.unknown_roi, width, height), chat)
            if not looks_like_panel(lines, self.cfg):
                return None
            pad = round(self.cfg.unknown_pad * width)
            x1 = max(0, min(ln.box.x for ln in lines) - pad)
            y1 = max(0, min(ln.box.y for ln in lines) - pad)
            x2 = min(width, max(ln.box.x2 for ln in lines) + pad)
            y2 = min(height, max(ln.box.y2 for ln in lines) + pad)
            box = Rect(x1, y1, x2 - x1, y2 - y1)
            allows = () if self.cfg.unknown_blocks else ACTIONS  # 默认只报告、不拦操作（误报率还没在真机核对）
            panel = Panel(UNKNOWN, "不认识的面板", box, False, UNKNOWN_LAYER, allows)
            reading = split_reading(lines, panel, None, self.cfg, width, height, now)
            blob = " ".join([reading.title, reading.text, *(b.text for b in reading.buttons)])
            for card in sorted(self.cards.values(), key=lambda c: -c.layer):
                if not card.texts or not any(k in blob for f in card.texts for k in f.any):
                    continue
                if all(check_feature(frame, f, self.builtins)[0] for f in card.quick):
                    panel = Panel(card.name, card.label, box, card.verified, card.layer, card.allows)
                    reading = split_reading(lines, panel, card, self.cfg, width, height, now)
                    break
            return panel, reading, thumb(box.crop(frame))
        except Exception:
            log.exception("通用兜底认面板出错，这次当没结论")
            return _FAILED

    def _apply(self, result, small: np.ndarray | None, now: float) -> None:
        if result is _FAILED:
            return
        if result is None:
            if self._found is not None:
                self._emit("close", self._found)
                self._found = None
            if small is not None:
                self._clean = small  # 扫过了没有面板：这就是新的干净画面（换地方、转了镜头）
            return
        panel, reading, box_thumb = result
        if self._found is None or self._found.name != panel.name:
            if self._found is not None:
                self._emit("close", self._found)
            self._emit("open", panel, reading)
        self._found, self._found_thumb, self._confirmed_at = panel, box_thumb, now
        self.readings[panel.name] = reading
