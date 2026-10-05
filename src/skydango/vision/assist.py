"""Claude 辅助标注（`perception label --assist`）：挑帧 → 检测器出人物候选框 → claude -p 核对 → 合并成 YOLO 标注。

设计见 docs/superpowers/specs/2026-09-28-assist-labeling-design.md。这里只放能单独测的逻辑，
命令行编排在 cli.py 的 `_perception_label_assist`；真正调 Claude 的函数由外面传给 `Reviewer`。
"""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..models.errors import ModelError, ModelUnavailable
from ..brain.images import image_block
from ..config import AssistConfig
from .bubbles import Rect
from .detect import Detection
from .track import iou

log = logging.getLogger(__name__)

PROMPT_VERSION = 4  # 改了提示词里的规则就加一：缓存按它失效，重跑会重新核对

ASSIST_SYSTEM = "你是游戏截图的目标检测标注核对员。按用户给的规则逐帧核对人物候选框，只输出一个 JSON 对象，不要别的文字。"

# 规则来自 2026-09-28 两轮试标（tmp/assist、tmp/assist2）踩过的坑；{self_hint} 换成配置里团子的长相
RULES = """下面是游戏《光·遇》(Sky) 的截图，每帧一张，要给目标检测做标注。每张图上画了：
- 白色细网格：每 100 像素一条，边上的数字是原图像素坐标（原图 {width}×{height}）
- 检测器给的候选"人物"框：彩色框，编号写在框外；每个框的精确坐标在图前面的文字里

对每个候选框判一个类别：
- self：团子，玩家自己操控的角色。外观：{self_hint}。镜头跟着它，一般在画面中下部，多是背对或侧对镜头。每张图最多一个 self。
  团子常和别人挨在一起：候选框同时框住团子和别人时，不要把整个框判成 self —— 判成离得最近的那个人，另外给 fixed_box，并把没框到的那个人列进 missing。
  不要把别的白发角色判成团子：没有那个背包的就不是团子。
- player：其他点亮了的玩家（彩色、发光的小人，衣服各式各样）。
  规则：头顶有圆圈的人形一律是玩家（player 或 player_unlit，绝不是 not_person）—— 白色描边空心圈（里面是 ✦、动作小图标、火焰、眼睛等），
  以及深色实心圆里画着飞人 + 小人图标的（那是在其他共享空间里的玩家，整个人发蓝光、半透明，可能在空中飞）。头顶有名字标签的也是玩家。
  整个人发蓝光、半透明、个子和普通玩家一样矮小（大头、短身子）的人形，是其他共享空间里的玩家，判 player —— 图标时有时无，这一帧没显示图标也算。
  用了装扮魔法变身的玩家也判 player：外形可以完全不像人（比如围红围巾的圆滚滚雪人、白鹿），个头和玩家差不多，
  头顶常有名字标签 / 圆圈 / 气泡，会走动、会和别人挨着；框住整个变身后的身体。雪人、小鹿模样的不要当成摆设或宠物判 not_person。
- player_unlit：没点火的陌生人，整个人是纯黑的剪影 / 黑影。他们头顶也有圆圈（常见火焰、眼睛图标），有圆圈不影响判成 player_unlit，只看身体是不是纯黑
  （穿黑斗篷但身上有亮色花纹、脸是亮的，不算，那是 player）。
- not_person：不是玩家 —— 先祖、宠物 / 小动物（比如白色圆滚滚的小兽）、石像、灯笼、花草、特效、UI 按钮等。
  先祖：身体修长（比玩家高瘦得多，接近成年人的比例）、身体偏透明，常摆固定姿势（比如红袍、白帽子的那种）；
  它头顶可能有深色实心圆里三颗星的图标，这个不算玩家的圆圈。拿不准是先祖还是玩家时写进 unsure。
- duplicate：和另一个候选框是同一个人。同一个人只留一个框：留框得最贴合的那个，其余标 duplicate，note 里写保留的编号。

还要做：
- 漏掉的人：没被任何候选框框到的 self / player / player_unlit，用网格估一个框 [x1,y1,x2,y2]（原图像素，框住整个身体含头发和斗篷，不含头顶的名字 / 圆圈），写进 missing。
  太远太小（身高不到约 25 像素）、完全看不出来的就别列。被 UI 面板挡住一部分但看得出是人的要列。
- 框不准：候选框类别对、但明显太大或太小（框住了两个人、只框了半个身体），在 note 里说明并给 fixed_box [x1,y1,x2,y2]。
- 拿不准的地方写进 unsure，没有就空字符串。

只输出一个 JSON 对象，每帧一项，每个候选编号都要判：
{{"<帧名>": {{"boxes": {{"1": {{"cls": "self", "note": "", "fixed_box": null}}}},
            "missing": [{{"cls": "player", "box": [x1, y1, x2, y2], "note": ""}}],
            "unsure": ""}}}}"""


def pick_frames(thumbs: Sequence[np.ndarray], min_change: float, max_gap: int) -> list[int]:
    """录像每秒 2 帧、相邻帧很像：和上一张留下的帧比，变化够大才留；隔太久也强制留一张。第 0 帧总留。"""
    keep: list[int] = []
    last = None
    for i, t in enumerate(thumbs):
        small = t.astype(np.float32)
        if last is None or np.abs(small - last).mean() > min_change or i - keep[-1] >= max_gap:
            keep.append(i)
            last = small
    return keep


_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255)]


def draw_candidates(frame: np.ndarray, boxes: list[Rect]) -> np.ndarray:
    """给 Claude 看的图：坐标网格 + 编号候选框（编号在框外，不挡住人）。不改原图。"""
    out = frame.copy()
    h, w = out.shape[:2]
    for x in range(0, w, 100):
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(out, str(x), (x + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for y in range(100, h, 100):
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(out, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for i, b in enumerate(boxes, 1):
        color = _COLORS[(i - 1) % len(_COLORS)]
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), color, 2)
        ty = b.y - 4 if b.y >= 30 else b.y2 + 26  # 靠顶时编号写在框下方
        cv2.rectangle(out, (b.x, ty - 24), (b.x + 16 * len(str(i)) + 10, ty + 4), color, -1)
        cv2.putText(out, str(i), (b.x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _xyxy(b: Rect) -> str:
    return f"[{b.x},{b.y},{b.x2},{b.y2}]"


def build_message(frames: list[tuple[str, np.ndarray, list[Rect]]], self_hint: str) -> list[dict]:
    """一批帧的内容块：规则 + 每帧（候选框坐标文字 + 画了网格和编号的图）。规则里的原图尺寸按这批第一帧填。"""
    height, width = frames[0][1].shape[:2] if frames else (1080, 1920)
    content: list[dict] = [{"type": "text", "text": RULES.format(self_hint=self_hint, width=width, height=height)}]
    for stem, frame, boxes in frames:
        listed = "；".join(f"{i}={_xyxy(b)}" for i, b in enumerate(boxes, 1)) or "没有候选框，只看有没有漏掉的人"
        content.append({"type": "text", "text": f"帧 {stem}：候选框 {listed}"})
        content.append(image_block(draw_candidates(frame, boxes), 85))
    return content


VERDICTS = ("self", "player", "player_unlit", "not_person", "duplicate")
PEOPLE = ("player", "player_unlit", "self")


@dataclass
class Verdict:
    cls: str
    fixed: Rect | None  # Claude 给的修正框（原图坐标）
    note: str


@dataclass
class FrameReview:
    verdicts: dict[int, Verdict]  # 候选编号（从 1 起）→ 判断
    missing: list[tuple[str, Rect, str]]  # Claude 补的漏框：类别、框、说明
    unsure: str
    problems: list[str] = field(default_factory=list)  # 回答里缺编号、类别写错之类，列进待核对清单


def _coords(value) -> list[float] | None:
    """[x1,y1,x2,y2] → 4 个有限的数；格式不对、有 NaN / inf（JSON 里的 NaN、Infinity 也能解析出来）返回 None。"""
    try:
        coords = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return coords if len(coords) == 4 and all(math.isfinite(v) for v in coords) else None


def _box(value, width: int, height: int) -> Rect | None:
    """[x1,y1,x2,y2] → 裁到画面内的 Rect；反了就排序，格式不对 / 不是有限数 / 裁完太小返回 None。"""
    coords = _coords(value)
    if coords is None:
        return None
    x1, y1, x2, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1 = max(0, round(x1)), max(0, round(y1))
    x2, y2 = min(width, round(x2)), min(height, round(y2))
    return Rect(x1, y1, x2 - x1, y2 - y1) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def _claimed_box(value, width: int, height: int, what: str, problems: list[str]) -> Rect | None:
    """Claude 给的框：坐标不对（格式错、NaN / inf）丢掉并记进 problems（列进待核对清单）；裁到画面外太小的照旧静默丢掉。"""
    if _coords(value) is None:
        problems.append(f"{what} 坐标不对，丢掉了：{value!r}")
        return None
    return _box(value, width, height)


def _boxes_by_id(raw, count: int) -> dict | None:
    """回答里的 boxes → 编号（"1"、"2"…）→ 判断。Claude 偶尔回成列表 [{...}, {...}]：按顺序当 1 号、2 号……
    没有候选时可以不写；有候选却不是对象 / 列表（看不懂）返回 None：这帧当没核对，不写缓存。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {str(i): v for i, v in enumerate(raw, 1)}
    return {} if raw is None and count == 0 else None


_STEM_PREFIX = re.compile(r"^帧\s*[:：]?\s*")
_IMAGE_SUFFIX = re.compile(r"\.(jpe?g|png)$", re.IGNORECASE)


def _by_stem(data: dict, stems) -> dict:
    """回答里的帧名 → 本批的帧名：原样对得上的优先；对不上的去掉"帧 "前缀、.jpg / .png 后缀再对。对不上的键丢掉。"""
    out = {k: v for k, v in data.items() if k in stems}
    for k, v in data.items():
        if isinstance(k, str) and k not in stems:
            stem = _IMAGE_SUFFIX.sub("", _STEM_PREFIX.sub("", k.strip())).strip()
            if stem in stems:
                out.setdefault(stem, v)
    return out


def extract_json(text: str, stems) -> dict | None:
    """回答里第一个含本批帧名的 JSON 对象，键换成本批的帧名（"帧 xxx.jpg" → "xxx"），只留本批的帧。
    前后有说明文字、代码块、多余的 { } 都不影响；找不到返回 None。"""
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            data, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and (hit := _by_stem(data, stems)):
            return hit
        i = text.find("{", i + 1)
    return None


def parse_review(text: str, frames: dict[str, int], width: int, height: int) -> dict[str, FrameReview]:
    """Claude 的回答 → 每帧的核对结果。frames：帧名 → 候选框个数。取不出 JSON 返回 {}；回答里缺的帧不在结果里（= 没核对）。"""
    data = extract_json(text, frames)
    if data is None:
        return {}
    out: dict[str, FrameReview] = {}
    for stem, count in frames.items():
        item = data.get(stem)
        if not isinstance(item, dict):
            continue
        boxes = _boxes_by_id(item.get("boxes"), count)
        if boxes is None:  # 看不懂：当没核对（不能当"全不是人"写进缓存）
            continue
        verdicts: dict[int, Verdict] = {}
        problems: list[str] = []
        for i in range(1, count + 1):
            v = boxes.get(str(i))
            cls = v.get("cls") if isinstance(v, dict) else None
            if cls not in VERDICTS:
                problems.append(f"{i} 号没判" if cls is None else f"{i} 号类别 {cls!r} 不认识")
                verdicts[i] = Verdict("not_person", None, "")
                continue
            fixed = _claimed_box(v.get("fixed_box"), width, height, f"{i} 号的 fixed_box", problems) if v.get("fixed_box") else None
            verdicts[i] = Verdict(cls, fixed, str(v.get("note") or ""))
        missing = []
        for m in item.get("missing") or []:
            if (isinstance(m, dict) and m.get("cls") in PEOPLE
                    and (box := _claimed_box(m.get("box"), width, height, f"补的 {m['cls']}", problems))):
                missing.append((m["cls"], box, str(m.get("note") or "")))
        out[stem] = FrameReview(verdicts, missing, str(item.get("unsure") or ""), problems)
    return out


def apply_review(candidates: list[Rect], review: FrameReview) -> list[tuple[str, Rect]]:
    """核对结果 → 人物标注框：认可的候选（有修正框用修正框）+ Claude 补的漏框；不是人 / 重复的丢掉。"""
    out = []
    for i, box in enumerate(candidates, 1):
        v = review.verdicts.get(i)
        if v is not None and v.cls in PEOPLE:
            out.append((v.cls, v.fixed or box))
    out += [(cls, box) for cls, box, _ in review.missing]
    return out


PEOPLE_COLORS = {"player": (0, 220, 0), "player_unlit": (255, 0, 200), "self": (200, 200, 200)}
WEAK_COLORS = {"name_tag": (0, 200, 255), "social_ring": (255, 120, 0)}  # 同 perception label 的预览


def _dashed(img: np.ndarray, b: Rect, color, thickness: int = 4, dash: int = 12) -> None:
    for x in range(b.x, b.x2, dash * 2):
        cv2.line(img, (x, b.y), (min(x + dash, b.x2), b.y), color, thickness)
        cv2.line(img, (x, b.y2), (min(x + dash, b.x2), b.y2), color, thickness)
    for y in range(b.y, b.y2, dash * 2):
        cv2.line(img, (b.x, y), (b.x, min(y + dash, b.y2)), color, thickness)
        cv2.line(img, (b.x2, y), (b.x2, min(y + dash, b.y2)), color, thickness)


def draw_review(frame: np.ndarray, weak: list[tuple[str, Rect]], candidates: list[Rect], review: FrameReview | None) -> np.ndarray:
    """预览：弱标注照旧；绿 player、紫 player_unlit、灰白 self；红细框 = 去掉的候选；虚线 = Claude 补的框。"""
    out = frame.copy()
    for cls, b in weak:
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), WEAK_COLORS.get(cls, (0, 200, 255)), 2)
    if review is None:
        for b in candidates:
            cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (0, 0, 255), 1)
        cv2.rectangle(out, (0, 0), (260, 50), (0, 0, 0), -1)
        cv2.putText(out, "NOT REVIEWED", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 没核对（cv2 写不了中文）
        return out
    for i, b in enumerate(candidates, 1):
        v = review.verdicts.get(i)
        if v is not None and v.cls in PEOPLE:
            box, color = v.fixed or b, PEOPLE_COLORS[v.cls]
            cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, 4)
            cv2.putText(out, f"{i} {v.cls}", (box.x, max(24, box.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (0, 0, 255), 1)
            cv2.putText(out, f"{i} {v.cls if v else '?'}", (b.x + 4, b.y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for cls, b, _ in review.missing:
        _dashed(out, b, PEOPLE_COLORS[cls])
        cv2.putText(out, f"+{cls}", (b.x, max(24, b.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PEOPLE_COLORS[cls], 2)
    return out


def review_report(results: list[tuple[str, FrameReview | None]]) -> str:
    """待核对清单（Markdown），按优先级：没核对 → 有补框 → 拿不准 / 回答有问题 → 其余抽查。"""
    failed = [s for s, r in results if r is None]
    added = [(s, r) for s, r in results if r is not None and r.missing]
    unsure = [(s, r) for s, r in results if r is not None and (r.unsure or r.problems)]
    listed = set(failed) | {s for s, _ in added} | {s for s, _ in unsure}
    rest = sum(1 for s, _ in results if s not in listed)
    lines = ["# 待核对清单（Claude 辅助标注）", "", "预览在 _preview/：绿 player、紫 player_unlit、灰白 self、红细框 = 去掉的候选、虚线 = Claude 补的框。", ""]
    lines += ["## 1. 没核对（人物框要全补）", ""] + ([f"- {s}" for s in failed] or ["（无）"]) + [""]
    lines += ["## 2. 有 Claude 补的框（偏松，要拉紧）", ""]
    lines += [f"- {s}：" + "；".join(f"{c} {n}".strip() for c, _, n in r.missing) for s, r in added] or ["（无）"]
    lines += ["", "## 3. Claude 拿不准 / 回答有问题", ""]
    lines += [f"- {s}：" + "；".join(filter(None, [r.unsure, *r.problems])) for s, r in unsure] or ["（无）"]
    lines += ["", f"## 4. 其余 {rest} 帧：抽查几张就行", ""]
    return "\n".join(lines)


def assist_workdir() -> Path:
    """claude -p 的工作目录：放在仓库外面的空目录。Claude Code 会从工作目录往上找 CLAUDE.md，
    放在仓库里会把项目说明（约 1 万 token）塞进每一批，还会诱导它先写说明文字（2026-09-28 实测）。"""
    return Path(tempfile.gettempdir()) / "skydango-assist-claude"


@dataclass
class FrameInput:
    stem: str
    image: np.ndarray
    candidates: list[Rect]
    people: list[Rect] = field(default_factory=list)  # 物品模式：已有的人物框（P1、P2…）
    hints: list[str] = field(default_factory=list)  # 物品模式：候选框的类别猜测（和 candidates 一一对应）


@dataclass(frozen=True)
class Protocol:
    """一种核对：提示词怎么拼、回答怎么解析、提示词版本（缓存按它失效）。标人和物品模式共用 Reviewer 的分批 / 并发 / 缓存。"""

    version: int
    system: str  # claude -p 的 --system-prompt
    build: Callable[[list[FrameInput], AssistConfig], list[dict]]  # 一批帧 → 内容块
    parse: Callable[[str, list[FrameInput]], dict[str, Any]]  # 回答 → 帧名 → 核对结果（取不出的帧不在里面）


def _people_parse(text: str, frames: list[FrameInput]) -> dict[str, FrameReview]:
    h, w = frames[0].image.shape[:2]
    return parse_review(text, {f.stem: len(f.candidates) for f in frames}, w, h)


PEOPLE_PROTOCOL = Protocol(
    PROMPT_VERSION, ASSIST_SYSTEM,
    lambda frames, cfg: build_message([(f.stem, f.image, f.candidates) for f in frames], cfg.self_hint),
    _people_parse,
)


class AssistLimit(RuntimeError):
    """订阅额度用完：已核对的帧都在缓存里，额度恢复后重跑同一条命令会接着做。"""


class Reviewer:
    """分批、并发地让 Claude 核对；每帧结果缓存成 <cache_dir>/<帧名>.json，重跑跳过已核对的帧。

    run：内容块 → result 消息（{"result": 文字, "usage": {...}}），失败抛 ModelError（测试里换成假的）。
    source：候选框来源（模型路径），和提示词版本、候选框坐标一起决定缓存算不算数。
    """

    def __init__(self, run: Callable[[list[dict]], dict], cache_dir: Path, cfg: AssistConfig, source: str,
                 protocol: Protocol = PEOPLE_PROTOCOL) -> None:
        self.run = run
        self.protocol = protocol
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        self.source = source
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self._lock = threading.Lock()
        self._limited = threading.Event()  # 有批次撞上额度用完：排队还没开始的批次不再起进程

    def _key(self, f: FrameInput) -> dict:
        key: dict = {"prompt_version": self.protocol.version, "source": self.source,
                     "candidates": [[b.x, b.y, b.x2, b.y2] for b in f.candidates]}
        if f.people:  # 标人的键保持老样子：旧缓存照样命中
            key["people"] = [[b.x, b.y, b.x2, b.y2] for b in f.people]
        if f.hints:
            key["hints"] = list(f.hints)
        return key

    def _cached(self, f: FrameInput) -> FrameReview | None:
        path = self.cache_dir / f"{f.stem}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        key = self._key(f)
        if {k: data.get(k) for k in key} != key or any(k in data for k in ("people", "hints") if k not in key):
            return None
        return self.protocol.parse(json.dumps({f.stem: data.get("review")}), [f]).get(f.stem)

    def _batch(self, frames: list[FrameInput]) -> dict[str, FrameReview | None]:
        content = self.protocol.build(frames, self.cfg)
        for attempt in (1, 2):
            if self._limited.is_set():  # 别的批次已经撞上额度用完：这批不起进程（review 最后照样抛 AssistLimit）
                return {}
            try:
                m = self.run(content)
                break
            except ModelError as e:
                if e.down == "limit" or isinstance(e, ModelUnavailable):
                    self._limited.set()
                    raise AssistLimit(str(e)) from e
                log.warning("核对 %s 失败（第 %d 次）：%s", ",".join(f.stem for f in frames), attempt, e)
        else:
            return {f.stem: None for f in frames}
        with self._lock:
            for k in self.usage:
                self.usage[k] += int((m.get("usage") or {}).get(k) or 0)
        text = m.get("result") or ""
        parsed = self.protocol.parse(text, frames)
        raw = extract_json(text, parsed) or {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for f in frames:
            if f.stem in parsed:
                entry = {**self._key(f), "review": raw.get(f.stem)}
                (self.cache_dir / f"{f.stem}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        return {f.stem: parsed.get(f.stem) for f in frames}

    def review(self, frames: list[FrameInput]) -> dict[str, FrameReview | None]:
        out: dict[str, FrameReview | None] = {}
        self._limited.clear()
        todo = []
        for f in frames:
            hit = self._cached(f)
            if hit is not None:
                out[f.stem] = hit
            else:
                todo.append(f)
        batches = [todo[i : i + self.cfg.batch] for i in range(0, len(todo), self.cfg.batch)]
        limit: AssistLimit | None = None
        with ThreadPoolExecutor(max(1, self.cfg.jobs)) as pool:
            for fut in [pool.submit(self._batch, b) for b in batches]:
                try:
                    out.update(fut.result())
                except AssistLimit as e:  # 等别的批次跑完、写好缓存再报
                    limit = e
        if limit is not None:
            raise limit
        return out


def merge_proposals(groups: list[list[Detection]], iou_thr: float = 0.6) -> list[Detection]:
    """几个模型的候选合在一起，按分数从高到低贪心去重（同一个人只留分最高的框）。"""
    keep: list[Detection] = []
    for d in sorted((d for g in groups for d in g), key=lambda d: d.score, reverse=True):
        if all(iou(d.box, k.box) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def people_candidates(dets: list[Detection]) -> list[Rect]:
    """自己训的模型（--model）：人物类的框当候选。"""
    return [d.box for d in dets if d.cls in PEOPLE]


class CocoPeople:
    """还没有自己的模型时：官方 COCO 模型的 person 类当候选，几个模型合并（试验里 yolo11n + yolo11x 互补）。"""

    def __init__(self, paths: list[str], conf: float, imgsz: int) -> None:
        from ultralytics import YOLO

        self.models = [YOLO(p) for p in paths]
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        groups = []
        for m in self.models:
            r = m.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0], verbose=False)[0]
            groups.append([
                Detection("player", Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), float(s))
                for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
            ])
        return merge_proposals(groups)
