"""物品模式的辅助标注（`perception label <数据集> --objects`）：给已经标好人的数据集补标座位 / 篝火 / 乐器 / 先祖，
顺带补标头顶气泡 typing（气泡不是物品，但同样是"给标好人的数据集补另一类框"，这一轮一起标）。

设计见 docs/superpowers/specs/2026-09-29-object-recognition-design.md §2。分批、并发、缓存沿用 assist.Reviewer，
这里是物品模式自己的提示词、解析和写回；命令行编排在 cli.py 的 `_perception_label_objects`。

写回规则：编号不在 LABEL_CLASSES 里的行原样保留（Claude 判成先祖的人物行只改类别号）；LABEL_CLASSES 的旧行整体换成这次的结果 —— 重跑安全。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..brain.images import image_block
from ..config import AssistConfig
from .assist import FrameInput, Protocol, _box, _dashed, _xyxy, draw_candidates, extract_json
from .bubbles import Rect
from .people import OBJECT_NAMES
from .weaklabel import yolo_line

OBJECTS = tuple(OBJECT_NAMES)  # ("bench", "bonfire", "instrument", "spirit")：感知层 objects() 报的物品
# 这个模式要标的类别 = 物品 + 头顶气泡。typing 不进 OBJECT_NAMES：身体 / 大脑不能把气泡当物品报出来
LABEL_CLASSES = OBJECTS + ("typing",)
LABEL_NAMES = {**OBJECT_NAMES, "typing": "气泡"}  # 清单里的中文
OBJECT_PROMPT_VERSION = 3  # 改了提示词里的规则就加一：缓存按它失效
PEOPLE_CLASSES = ("player", "player_unlit")  # P 编号只数这两类（团子、已经是先祖的不算）
VERDICTS = LABEL_CLASSES + ("not_object", "duplicate")
EXISTING = "已标 "  # 候选的 hint 以它开头 = 数据集里已有的标注（可能人工修过）；"猜 " 开头 = 检测器的猜测

OBJECT_SYSTEM = "你是游戏截图的目标检测标注员。按用户给的规则逐帧标出物品和头顶气泡、核对候选框，只输出一个 JSON 对象，不要别的文字。"
OBJECT_RULES = """下面是游戏《光·遇》(Sky) 的截图，每帧一张，要给目标检测补标"物品"和头顶气泡。每张图上画了：
- 白色细网格：每 100 像素一条，边上的数字是原图像素坐标（原图 1920×1080）
- 灰色细框 P1、P2…：已经标好的人物框（玩家）
- 彩色框 1、2…：物品 / 气泡候选框（可能没有）。写着"已标"的是数据集里已有的标注（可能人工修过：对的就原样认可，不要为了几个像素给 fixed_box）；
  写着"猜"的是检测器的猜测。每个框的精确坐标在图前面的文字里

要标的一共五类：四类物品，加上头顶气泡 typing（气泡不是物品，这一轮一起标）：
- bench：座位。明显是给人坐的：长椅、石凳、秋千座、摆好的坐垫。台阶、石头、地面不算。
- bonfire：篝火。燃着的篝火堆，包括玩家放的篝火道具。蜡烛、烛火堆、灯笼不算。
- instrument：乐器。摆在场景里、没人拿着的乐器（钢琴、竖琴架……）。玩家手里拿着的不算。
- spirit：先祖。发光的先祖灵魂（站在原地、没被收集的），以及先祖回忆里半透明的人形。玩家、团子、宠物、跟着玩家的小光团不算。
- typing：头顶气泡。人物头顶（好友在名字标签下方、互动圆圈上方；陌生人在头顶上方）的深色半透明圆角小框，
  里面是 `..` / `...`（正在输入）或者一行文字（发出去的话），两种都标 typing。只框气泡本身，不含名字标签、不含圆圈。团子自己头顶的也标。

要做的：
1. 每个候选框判一个：bench / bonfire / instrument / spirit / typing / not_object（不是这五类）/ duplicate（和另一个候选是同一个东西）。类别对但框明显不准，给 fixed_box [x1,y1,x2,y2]。
2. 漏掉的物品和气泡（没有候选框框到的）：写进 missing，给类别和框 [x1,y1,x2,y2]（原图像素，框住整个物品，被挡住就框露出来的部分；只框先祖的身体，不含头顶光效）。框高不到 15 像素的太远，不用标。
3. 灰色人物框里其实是先祖的（先祖被当成了玩家）：把编号写进 spirits，比如 ["P2"]。只写确定的；是玩家的不要写。
拿不准的写进 unsure，一句话说明。

只输出一个 JSON 对象，每帧一项：
{"帧名": {"boxes": {"1": {"cls": "bench", "fixed_box": [x1,y1,x2,y2], "note": "…"}}, "missing": [{"cls": "spirit", "box": [x1,y1,x2,y2], "note": "…"}], "spirits": ["P2"], "unsure": ""}}
fixed_box、note 没有就不写；没有候选框时 boxes 写 {}。"""

OBJECT_COLORS = {"bench": (216, 78, 29), "bonfire": (12, 88, 234), "instrument": (175, 164, 253), "spirit": (255, 255, 255),
                 "typing": (255, 0, 255)}  # BGR：蓝、橙、粉、白、品红
_P_ID = re.compile(r"^P(\d+)$")


@dataclass
class ObjectVerdict:
    cls: str  # VERDICTS 之一；回答里没判 / 类别不认识是 "unjudged"
    fixed: Rect | None  # Claude 给的修正框（原图坐标）
    note: str


@dataclass
class ObjectReview:
    verdicts: dict[int, ObjectVerdict]  # 候选编号（从 1 起）→ 判断
    missing: list[tuple[str, Rect, str]]  # Claude 补的物品框：类别、框、说明
    spirits: list[int]  # 判成先祖的人物框编号（P 后面的数字，从 1 起）
    unsure: str
    problems: list[str] = field(default_factory=list)  # 回答里缺编号、类别写错、P 编号不对……列进待核对清单
    changed: list[str] = field(default_factory=list)  # 已有标注被删 / 改（apply_object_review 填），列进待核对清单


def _draw(frame: np.ndarray, people: list[Rect], candidates: list[Rect]) -> np.ndarray:
    """给 Claude 看的图：灰色 P 框（人物）+ 网格和编号候选框。不改原图。"""
    out = frame.copy()
    for i, b in enumerate(people, 1):
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (160, 160, 160), 1)
        cv2.putText(out, f"P{i}", (b.x + 2, max(14, b.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return draw_candidates(out, candidates)


def _hint(hints: list[str], i: int) -> str:
    if i - 1 >= len(hints):
        return ""
    h = hints[i - 1]
    return f"({h})" if h.startswith((EXISTING, "猜 ")) else f"(猜 {h})"


def build_object_message(frames: list[FrameInput], cfg: AssistConfig) -> list[dict]:
    """一批帧的内容块：规则 + 每帧（人物框、候选框坐标文字 + 画了网格 / P 框 / 编号候选框的图）。"""
    content: list[dict] = [{"type": "text", "text": OBJECT_RULES}]
    for f in frames:
        people = "；".join(f"P{i}={_xyxy(b)}" for i, b in enumerate(f.people, 1)) or "无"
        cands = "；".join(f"{i}={_xyxy(b)}" + _hint(f.hints, i) for i, b in enumerate(f.candidates, 1)) or "无，只看有没有漏掉的物品"
        content.append({"type": "text", "text": f"帧 {f.stem}：人物 {people}；候选 {cands}"})
        content.append(image_block(_draw(f.image, f.people, f.candidates), 85))
    return content


def parse_object_review(text: str, frames: list[FrameInput]) -> dict[str, ObjectReview]:
    """Claude 的回答 → 每帧的核对结果。取不出 JSON 返回 {}；回答里缺的帧不在结果里（= 没核对）。"""
    by_stem = {f.stem: f for f in frames}
    data = extract_json(text, by_stem)
    if data is None:
        return {}
    out: dict[str, ObjectReview] = {}
    for stem, f in by_stem.items():
        item = data.get(stem)
        if not isinstance(item, dict):
            continue
        height, width = f.image.shape[:2]
        boxes = item.get("boxes") if isinstance(item.get("boxes"), dict) else {}
        verdicts: dict[int, ObjectVerdict] = {}
        problems: list[str] = []
        for i in range(1, len(f.candidates) + 1):
            v = boxes.get(str(i))
            cls = v.get("cls") if isinstance(v, dict) else None
            if cls not in VERDICTS:  # 没判 ≠ 不是物品：已有的标注留着（apply_object_review）
                problems.append(f"{i} 号没判" if cls is None else f"{i} 号类别 {cls!r} 不认识")
                verdicts[i] = ObjectVerdict("unjudged", None, "")
                continue
            fixed = _box(v.get("fixed_box"), width, height) if v.get("fixed_box") else None
            verdicts[i] = ObjectVerdict(cls, fixed, str(v.get("note") or ""))
        missing = []
        raw_missing, raw_spirits = item.get("missing") or [], item.get("spirits") or []
        if not isinstance(raw_missing, list):
            problems.append(f"missing 应该是列表：{raw_missing!r}")
            raw_missing = []
        if not isinstance(raw_spirits, list):
            problems.append(f"spirits 应该是列表：{raw_spirits!r}")
            raw_spirits = []
        for m in raw_missing:
            if isinstance(m, dict) and m.get("cls") in LABEL_CLASSES and (box := _box(m.get("box"), width, height)):
                missing.append((m["cls"], box, str(m.get("note") or "")))
        spirits: list[int] = []
        for s in raw_spirits:
            hit = _P_ID.match(s) if isinstance(s, str) else None
            n = int(hit.group(1)) if hit else 0
            if 1 <= n <= len(f.people):
                if n not in spirits:
                    spirits.append(n)
            else:
                problems.append(f"spirits 里的 {s!r} 不是这帧的人物框编号")
        out[stem] = ObjectReview(verdicts, missing, spirits, str(item.get("unsure") or ""), problems)
    return out


def apply_object_review(candidates: list[Rect], review: ObjectReview, hints: list[str] | None = None) -> list[tuple[str, Rect]]:
    """核对结果 → 物品 / 气泡标注框：认可的候选（有修正框用修正框）+ Claude 补的；不是物品 / 重复的丢掉。

    hints 以 EXISTING 开头的候选是数据集里已有的标注：Claude 没判的原样留着（不能因为回答漏了就删），
    被删 / 改类别 / 改框的记进 review.changed，列进待核对清单。"""
    hints = hints or []
    out = []
    for i, box in enumerate(candidates, 1):
        v = review.verdicts.get(i)
        hint = hints[i - 1] if i - 1 < len(hints) else ""
        old = hint[len(EXISTING):] if hint.startswith(EXISTING) else None
        name = LABEL_NAMES.get(old or "", old or "")
        if v is None or v.cls == "unjudged":
            if old in LABEL_CLASSES:
                out.append((old, box))
            continue
        if v.cls in LABEL_CLASSES:
            out.append((v.cls, v.fixed or box))
            if old is not None and v.cls != old:
                review.changed.append(f"{i} 号已有的{name}改成了{LABEL_NAMES[v.cls]}")
            elif old is not None and v.fixed is not None:
                review.changed.append(f"{i} 号已有的{name}框改了")
        elif old is not None:
            review.changed.append(f"{i} 号已有的{name}删了（{v.cls}" + (f"：{v.note}）" if v.note else "）"))
    out += [(cls, box) for cls, box, _ in review.missing]
    return out


def _parse_line(line: str, classes: list[str]) -> tuple[str, list[float]] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        idx, nums = int(parts[0]), [float(p) for p in parts[1:]]
    except ValueError:
        return None
    return (classes[idx], nums) if 0 <= idx < len(classes) else None


def people_in_labels(text: str, classes: list[str], width: int, height: int) -> list[Rect]:
    """标注文件里的人物框（player / player_unlit），按文件顺序 → P1、P2…（团子、已经是先祖的不算）。"""
    out = []
    for line in text.splitlines():
        parsed = _parse_line(line, classes)
        if parsed is None or parsed[0] not in PEOPLE_CLASSES:
            continue
        cx, cy, w, h = parsed[1]
        out.append(Rect(round((cx - w / 2) * width), round((cy - h / 2) * height), round(w * width), round(h * height)))
    return out


def objects_in_labels(text: str, classes: list[str], width: int, height: int) -> list[tuple[str, Rect]]:
    """标注文件里已有的物品 / 气泡框（重跑 / 手工修过的）：交给 Claude 当候选再核对一遍 —— 写回时这些行整体替换，不当候选就丢了。"""
    out = []
    for line in text.splitlines():
        parsed = _parse_line(line, classes)
        if parsed is None or parsed[0] not in LABEL_CLASSES:
            continue
        cx, cy, w, h = parsed[1]
        box = Rect(round((cx - w / 2) * width), round((cy - h / 2) * height), round(w * width), round(h * height))
        out.append((parsed[0], box))
    return out


def rewrite_labels(text: str, classes: list[str], objects: list[tuple[str, Rect]], spirits: list[int], width: int, height: int) -> str:
    """写回一个标注文件：物品 / 气泡的旧行去掉、这次的结果追加在末尾；第 n 个人物行（n ∈ spirits）改成先祖；别的行原样保留。"""
    index = {c: i for i, c in enumerate(classes)}
    lines, person = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = _parse_line(line, classes)
        if parsed is None:  # 解析不了的行不认识，原样留着
            lines.append(line)
            continue
        cls = parsed[0]
        if cls in LABEL_CLASSES:
            continue
        if cls in PEOPLE_CLASSES:
            person += 1
            if person in spirits:
                line = " ".join([str(index["spirit"]), *line.split()[1:]])
        lines.append(line)
    lines += [yolo_line(index[c], b, width, height) for c, b in objects]
    return "\n".join(lines) + ("\n" if lines else "")


def draw_objects_preview(frame: np.ndarray, objects: list[tuple[str, Rect]], people: list[Rect], spirits: list[int],
                         candidates: list[Rect], review: ObjectReview | None) -> np.ndarray:
    """预览：人物灰细框、改成先祖的红虚线；物品按类别上色（座位蓝、篝火橙、乐器粉、先祖白、气泡品红），Claude 补的虚线。"""
    out = frame.copy()
    for i, b in enumerate(people, 1):
        if i in spirits:
            _dashed(out, b, (0, 0, 255))
            cv2.putText(out, f"P{i}->spirit", (b.x, max(24, b.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (160, 160, 160), 1)
    if review is None:
        for b in candidates:
            cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), (0, 0, 255), 1)
        cv2.rectangle(out, (0, 0), (260, 50), (0, 0, 0), -1)
        cv2.putText(out, "NOT REVIEWED", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 没核对（cv2 写不了中文）
        return out
    added = {(c, (b.x, b.y, b.w, b.h)) for c, b, _ in review.missing}
    for cls, b in objects:
        color = OBJECT_COLORS.get(cls, (0, 255, 255))
        if (cls, (b.x, b.y, b.w, b.h)) in added:
            _dashed(out, b, color)
        else:
            cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), color, 4)
        cv2.putText(out, cls, (b.x, max(24, b.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def objects_report(results: list[tuple[str, ObjectReview | None]], edited: list[str] | None = None) -> str:
    """待核对清单（Markdown）：人改先祖 → 已有标注被删 / 改 → Claude 补了框 → 没核对成 → 你改过、这次跳过 → 拿不准 / 回答有问题。"""
    spirit = [(s, r) for s, r in results if r is not None and r.spirits]
    changed = [(s, r) for s, r in results if r is not None and r.changed]
    added = [(s, r) for s, r in results if r is not None and r.missing]
    failed = [s for s, r in results if r is None]
    unsure = [(s, r) for s, r in results if r is not None and (r.unsure or r.problems)]
    lines = ["# 待核对清单（物品模式）", "",
             "预览在 _preview_objects/：座位蓝、篝火橙、乐器粉、先祖白、气泡品红；虚线 = Claude 补的框；红虚线 = 人物框改成了先祖。", ""]
    lines += ["## 1. 人物框改成了先祖（确认真的是先祖）", ""]
    lines += [f"- {s}：" + "、".join(f"P{n}" for n in r.spirits) for s, r in spirit] or ["（无）"]
    lines += ["", "## 2. 已有标注被删 / 改（确认删得对、改得对）", ""]
    lines += [f"- {s}：" + "；".join(r.changed) for s, r in changed] or ["（无）"]
    lines += ["", "## 3. Claude 补了物品框 / 气泡框（偏松，要拉紧）", ""]
    lines += [f"- {s}：" + "；".join(f"{LABEL_NAMES.get(c, c)} {n}".strip() for c, _, n in r.missing) for s, r in added] or ["（无）"]
    lines += ["", "## 4. 没核对成（标注没动，物品 / 气泡要自己补）", ""] + ([f"- {s}" for s in failed] or ["（无）"])
    lines += ["", "## 5. 你改过、这次跳过的（要重新核对加 --recheck）", ""] + ([f"- {s}" for s in edited or []] or ["（无）"])
    lines += ["", "## 6. Claude 拿不准 / 回答有问题", ""]
    lines += [f"- {s}：" + "；".join(filter(None, [r.unsure, *r.problems])) for s, r in unsure] or ["（无）"]
    return "\n".join(lines) + "\n"


OBJECTS_PROTOCOL = Protocol(OBJECT_PROMPT_VERSION, OBJECT_SYSTEM, build_object_message, parse_object_review)


def review_to_state(review: ObjectReview) -> dict:
    """写回后记下的核对结果（清单用）：重跑跳过这帧时照样列进清单。"""
    return {"spirits": review.spirits, "missing": [[c, b.x, b.y, b.w, b.h, n] for c, b, n in review.missing],
            "changed": review.changed, "unsure": review.unsure, "problems": review.problems}


def review_from_state(data: dict) -> ObjectReview:
    missing = [(m[0], Rect(*m[1:5]), m[5]) for m in data.get("missing") or []]
    return ObjectReview({}, missing, list(data.get("spirits") or []), str(data.get("unsure") or ""),
                        list(data.get("problems") or []), list(data.get("changed") or []))
