"""动作片段的 Claude 初分（`perception gesture-label`）：16 帧拼成 4×4 一张图，让 Claude 判断中间那个人在做什么。

设计见 docs/superpowers/specs/2026-10-01-gesture-labeling-training-design.md §2。复用 `vision/assist.py` 的
Reviewer（分批、并发、缓存、额度用完），这里只放提示词、拼图、解析和 `claude.json` 读写。

TODO（2026-10-01 晚）：`GESTURE_SYSTEM` 里四个动作的样子是按印象写的，等用户核对过游戏里真实的动画后再改；
改了就把 `GESTURE_PROMPT_VERSION` 加一（缓存按它失效）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..brain.images import image_block
from ..config import AssistConfig
from .assist import FrameInput, Protocol, extract_json

log = logging.getLogger(__name__)

GUESSES = ("wave", "bow", "cheer", "shy", "none", "unsure")
GESTURE_PROMPT_VERSION = 1
CELL = 112  # 拼图每格边长；4×4 → 448×448
_ACTION_NAMES = {"wave": "挥手", "bow": "鞠躬", "cheer": "欢呼", "shy": "害羞"}

GESTURE_SYSTEM = """你是游戏《光·遇》(Sky) 动作片段的标注员。每个片段是一张 4×4 的拼图：同一个人的 16 帧，覆盖 2 秒，
顺序是从左到右、从上到下，每格左上角的数字是帧号 0~15。判断画面中间那个人在这 2 秒里做的是哪个动作：
- wave 挥手：一只手举起来左右摆。
- bow 鞠躬：上身往前弯下去再起来。
- cheer 欢呼：双手往上举。
- shy 害羞：双手捂脸或者身体扭过去。
- none：站着、走路、跑、坐着或者别的动作，不是上面四个。
- unsure：动作没做完（只看到开头或结尾）、画面看不清、人被挡住，没法判断。
有的片段前面会给一句录像说明（比如"这段录像是好友在反复做挥手"），那是提示：画面里实在没有这个动作时照实判 none 或 unsure，别硬凑。
只输出一个 JSON 对象，每个片段一项，键是片段名：
{"<片段名>": {"label": "wave", "confidence": 0.8, "reason": "一句话说明看到了什么"}}
label 只能是 wave / bow / cheer / shy / none / unsure；confidence 是 0~1。不要输出别的文字。"""
# 不看录像名（`--blind`）：去掉录像说明那句，只凭画面判断。看录像名时 Claude 常照着名字判（10-02 实测挥手录像 114 段判了 113 段挥手，其中有一动不动的）
GESTURE_SYSTEM_BLIND = GESTURE_SYSTEM.replace('有的片段前面会给一句录像说明（比如"这段录像是好友在反复做挥手"），那是提示：画面里实在没有这个动作时照实判 none 或 unsure，别硬凑。\n', "")
GUESS_FILE = "claude.json"
BLIND_FILE = "claude-blind.json"  # 不看录像名的初分，和 claude.json 并存；标注页优先显示它


@dataclass(frozen=True)
class Guess:
    label: str
    confidence: float
    reason: str


def contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    """16 帧 → 4×4 拼图（每格 112×112，不是这个尺寸先缩放），每格左上角白底黑字写帧号 0~15。"""
    if len(frames) != 16:
        raise ValueError(f"拼图要正好 16 帧，给了 {len(frames)}")
    sheet = np.zeros((CELL * 4, CELL * 4, 3), np.uint8)
    for i, f in enumerate(frames):
        if f.shape[:2] != (CELL, CELL):
            f = cv2.resize(f, (CELL, CELL), interpolation=cv2.INTER_AREA)
        y, x = (i // 4) * CELL, (i % 4) * CELL
        sheet[y : y + CELL, x : x + CELL] = f
        label = str(i)
        cv2.rectangle(sheet, (x, y), (x + 10 * len(label) + 4, y + 16), (255, 255, 255), -1)
        cv2.putText(sheet, label, (x + 2, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return sheet


def recording_hint(recording: str) -> str:
    """录像名（不分大小写）里带动作名：这段录像是好友在反复做那个动作；带 none：里面没有这四个动作。"""
    name = recording.lower()
    for key, zh in _ACTION_NAMES.items():
        if key in name:
            return f"这段录像是好友在反复做{zh}（中间有停顿）"
    if "none" in name:
        return "这段录像里没有这四个动作"
    return ""


def build_gesture_message(frames: list[FrameInput], cfg: AssistConfig, hint: bool = True) -> list[dict]:
    """一批片段的内容块：规则在 --system-prompt 里，这里每个片段一行文字（名字 + 录像提示）加一张拼图。
    hint = False：不写录像提示（片段名本身也带录像名，`--blind` 时 stem 换成 `blind_key` 的代号）。"""
    from .gesture import recording_of

    content: list[dict] = []
    for f in frames:
        note = recording_hint(recording_of(f.stem)) if hint else ""
        content.append({"type": "text", "text": f"片段 {f.stem}：{note}"})
        content.append(image_block(f.image, 85))
    return content


def blind_key(clip: str) -> str:
    """`--blind` 时给 Claude 看的片段代号：片段名的哈希，看不出录像名，同一段每次一样（缓存认它）。"""
    return "c" + hashlib.sha1(clip.encode("utf-8")).hexdigest()[:10]


def parse_gesture_review(text: str, frames: list[FrameInput]) -> dict[str, Guess]:
    """Claude 的回答 → 片段名 → Guess。只收 GUESSES 里的 label；片段名对不上、label 不认识的不在结果里。"""
    data = extract_json(text, {f.stem for f in frames})
    if data is None:
        return {}
    out: dict[str, Guess] = {}
    for stem, item in data.items():
        if not isinstance(item, dict) or item.get("label") not in GUESSES:
            continue
        out[stem] = Guess(item["label"], _conf(item.get("confidence")), str(item.get("reason") or "")[:60])
    return out


def _conf(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v else min(1.0, max(0.0, v))  # NaN → 0


GESTURE_PROTOCOL = Protocol(GESTURE_PROMPT_VERSION, GESTURE_SYSTEM, build_gesture_message, parse_gesture_review)
GESTURE_BLIND_PROTOCOL = Protocol(
    GESTURE_PROMPT_VERSION, GESTURE_SYSTEM_BLIND,
    lambda frames, cfg: build_gesture_message(frames, cfg, hint=False), parse_gesture_review,
)


def load_guess(clip_dir: Path, name: str = GUESS_FILE) -> Guess | None:
    """读片段目录里的 claude.json（name = BLIND_FILE 读不看录像名的那份）；没有 / 坏了 / label 不认识返回 None。"""
    try:
        d = json.loads((Path(clip_dir) / name).read_text(encoding="utf-8"))
        label = d["label"]
        if label not in GUESSES:
            return None
        return Guess(label, _conf(d.get("confidence")), str(d.get("reason") or "")[:60])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_guess(clip_dir: Path, g: Guess, model: str, name: str = GUESS_FILE) -> None:
    entry = {"label": g.label, "confidence": g.confidence, "reason": g.reason, "model": model,
             "prompt_version": GESTURE_PROMPT_VERSION, "t": time.time()}
    if name == BLIND_FILE:
        entry["blind"] = True
    (Path(clip_dir) / name).write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
