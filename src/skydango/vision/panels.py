"""面板识别：画面上开着哪些面板（聊天记录面板、动作面板、轮盘编辑、好友树、各种弹框……）。

只看不动：每个已知面板一张"特征卡"（assets/panels/<名字>/card.toml + 模板小图），每帧几毫秒判断开没开（快看）；
面板上写了什么、有哪些按钮靠 OCR（细读）；认不出来的弹框由"通用兜底"按"有按钮词 + 一段正文"认成不认识的面板。
动手（关面板、按按钮）在 game/panels.py。设计见 docs/superpowers/specs/2026-09-29-panels-design.md。
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..imageio import imread
from .bubbles import Rect
from .icons import silhouette, trim

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
