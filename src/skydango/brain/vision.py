"""大脑直接看图（spec 2026-11-02-brain-vision-design）：给大脑看的编号截图和图例、唤醒消息该不该附图、大脑看不看得了图。纯计算，不碰设备。

编号图只有一种画法：身体认出的人 / 东西 / 图标画框 + 左上角外一个写着数字的圆，图例文字（跟在图前面）告诉它每个编号身体认的是什么。
名字标签、气泡、火焰候选这些没有"这个框是谁"的意义，不画不编号。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import BrainVisionConfig
from ..vision.people import OBJECT_NAMES
from .images import encode_jpeg, fit, image_block

LEGEND_HEAD = "图上的编号（身体认的，可能错）："
NO_BOXES = "图上没画框（身体这会儿没认出东西）"

_GREEN, _YELLOW, _WHITE, _GRAY, _CYAN = (0, 200, 0), (0, 215, 255), (255, 255, 255), (160, 160, 160), (255, 220, 0)
MARK_KINDS: dict[str, tuple[int, int, int]] = {  # BGR；不在这里的 kind 不画（name / tag / typing / ring / rejected……）
    "friend": _GREEN,
    "maybe": _YELLOW, "unsure": _YELLOW, "dango_look": _YELLOW,
    "stranger": _WHITE, "unlit": _WHITE, "player": _WHITE,
    "self": _GRAY,
    "icon": _CYAN, "request": _CYAN,
    **{k: _CYAN for k in OBJECT_NAMES},
}

TRIGGERS = frozenset({"chat", "arrive", "approach", "notice", "lull"})  # 取走了这些事件的唤醒才自动附图
THUMB_SIZE = (480, 270)  # 大脑控制台里的缩略图
CIRCLE = 14  # 编号圆的半径（画在缩小后的图上）


@dataclass
class Attachment:
    legend: str  # 图例文字（跟在图前面）
    block: dict  # images.image_block 那种 Claude 风格的图片块
    src: str  # "auto" 或工具名（look……）
    w: int
    h: int
    thumb: str  # 缩到 480 宽的 JPEG base64，给大脑控制台


def _number(i: int) -> str:
    return chr(0x2460 + i) if i < 20 else f"({i + 1})"  # ①~⑳，更多的用 (21)


def _who(b: dict) -> str:
    """图例里一个框的说法。"""
    kind, label = b["kind"], b.get("label") or ""
    if kind == "friend":
        return f"{label}（好友，看到名字）"
    if kind in ("maybe", "unsure"):
        return f"{label.rstrip('?')}（没看到名字）"
    if kind == "self":
        return "你自己"
    if kind == "dango_look":
        return "像你自己"
    if kind == "unlit":
        return "陌生人（没点火，黑影）"
    if kind == "player":
        return "没认出是谁的人"
    if kind in ("icon", "request"):
        return f"图标：{label}"
    return label or OBJECT_NAMES.get(kind, kind)  # 陌生人编号、物品


def mark_frame(frame: np.ndarray, boxes: list[dict] | None, size: tuple[int, int], quality: int, src: str = "auto") -> Attachment:
    """原图缩到 size，画上编号框；boxes 是 env.overlay(now) 的条目（整张截图坐标），None = 不画框。"""
    marks = sorted((b for b in boxes or [] if b.get("kind") in MARK_KINDS), key=lambda b: b["x"])
    view = fit(frame, size).copy()  # 缩小后再画：线和字不会被缩得看不清；copy 保证不改原图
    h, w = view.shape[:2]
    k = w / frame.shape[1]
    legend = [LEGEND_HEAD] if marks else [NO_BOXES]
    for i, b in enumerate(marks):
        color = MARK_KINDS[b["kind"]]
        x1, y1 = round(b["x"] * k), round(b["y"] * k)
        x2, y2 = round((b["x"] + b["w"]) * k), round((b["y"] + b["h"]) * k)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cx, cy = min(max(x1, CIRCLE), w - CIRCLE), min(max(y1 - CIRCLE, CIRCLE), h - CIRCLE)  # 圆在框左上角外，贴边时挪进画面
        cv2.circle(view, (cx, cy), CIRCLE, color, -1)
        text = str(i + 1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(view, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        legend.append(f"{_number(i)} {_who(b)}")
    return Attachment("\n".join(legend), image_block(view, quality), src, w, h, encode_jpeg(fit(view, THUMB_SIZE), 70))


def doubtful(boxes: list[dict]) -> bool:
    """感知层自己拿不准：有"像谁 / 可能是谁"的人，或认不出的图标。"""
    return any(b.get("kind") in ("maybe", "unsure") or (b.get("kind") == "icon" and b.get("unknown")) for b in boxes)


def should_attach(kinds: set[str], doubtful: bool, last_at: float, now: float, cfg: BrainVisionConfig) -> bool:
    """这一轮唤醒消息要不要自动附一张图：开着 auto、离上次自动附图够久，且有触发的事件或感知层拿不准。"""
    if not cfg.auto or now - last_at < cfg.min_gap:
        return False
    return bool(kinds & TRIGGERS) or doubtful


class AutoVision:
    """唤醒消息的自动附图：在身体线程里调 pick（经 body.call），只管挑时机和画图；last_at 记上次自动附图的时间。"""

    def __init__(self, cfg: BrainVisionConfig, quality: int) -> None:
        self.cfg = cfg
        self.quality = quality
        self.last_at = float("-inf")

    def pick(self, events: list, now: float, frame, env, black: bool) -> Attachment | None:
        if frame is None or black:  # 还没截到图 / 画面黑着（切场景）：这一轮照常只发文字
            return None
        boxes = None  # 感知层结果过时 / 没开：只给图、不画框
        if env is not None and hasattr(env, "overlay_fresh") and env.overlay_fresh(now):
            boxes = env.overlay(now)
        if not should_attach({e.kind for e in events}, doubtful(boxes or []), self.last_at, now, self.cfg):
            return None
        self.last_at = now
        return mark_frame(frame, boxes, tuple(self.cfg.size), self.quality, "auto")


def brain_sees(session, sees_model: bool, cfg: BrainVisionConfig, text_only: bool) -> bool:
    """大脑现在看不看得了图：Claude Code 大脑照旧看模型勾没勾；OpenAI 兼容的还要 [brain.vision] 开着、不是沙盒、没被接口拒收过。"""
    from .session import BrainSession

    if isinstance(session, BrainSession):
        return sees_model
    return cfg.enabled and not text_only and sees_model and not getattr(session, "vision_off", "")


def parse_check(text: str) -> str:
    """最后的文字里以「核对：」开头的行（去掉前缀，多行用「；」连起来）；没有是空字符串。"""
    rows = [line.strip().removeprefix("核对：").strip() for line in (text or "").splitlines() if line.strip().startswith("核对：")]
    return "；".join(r for r in rows if r)

