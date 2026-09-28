"""Claude 辅助标注（`perception label --assist`）：挑帧 → 检测器出人物候选框 → claude -p 核对 → 合并成 YOLO 标注。

设计见 docs/superpowers/specs/2026-09-28-assist-labeling-design.md。这里只放能单独测的逻辑，
命令行编排在 cli.py 的 `_perception_label_assist`；真正调 Claude 的函数由外面传给 `Reviewer`。
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ..brain.images import image_block
from .bubbles import Rect

PROMPT_VERSION = 1  # 改了提示词里的规则就加一：缓存按它失效，重跑会重新核对

ASSIST_SYSTEM = "你是游戏截图的目标检测标注核对员。按用户给的规则逐帧核对人物候选框，只输出一个 JSON 对象，不要别的文字。"

# 规则来自 2026-09-28 两轮试标（tmp/assist、tmp/assist2）踩过的坑；{self_hint} 换成配置里团子的长相
RULES = """下面是游戏《光·遇》(Sky) 的截图，每帧一张，要给目标检测做标注。每张图上画了：
- 白色细网格：每 100 像素一条，边上的数字是原图像素坐标（原图 1920×1080）
- 检测器给的候选"人物"框：彩色框，编号写在框外；每个框的精确坐标在图前面的文字里

对每个候选框判一个类别：
- self：团子，玩家自己操控的角色。外观：{self_hint}。镜头跟着它，一般在画面中下部，多是背对或侧对镜头。每张图最多一个 self。
  团子常和别人挨在一起：候选框同时框住团子和别人时，不要把整个框判成 self —— 判成离得最近的那个人，另外给 fixed_box，并把没框到的那个人列进 missing。
  不要把别的白发角色判成团子：没有那个背包的就不是团子。
- player：其他点亮了的玩家（彩色、发光的小人，衣服各式各样）。
  规则：头顶有圆圈的人形一律是玩家（player 或 player_unlit，绝不是 not_person）—— 白色描边空心圈（里面是 ✦、动作小图标、火焰、眼睛等），
  以及深色实心圆里画着飞人 + 小人图标的（那是在其他共享空间里的玩家，整个人发蓝光、半透明，可能在空中飞）。头顶有名字标签的也是玩家。
- player_unlit：没点火的陌生人，整个人是纯黑的剪影 / 黑影。他们头顶也有圆圈（常见火焰、眼睛图标），有圆圈不影响判成 player_unlit，只看身体是不是纯黑
  （穿黑斗篷但身上有亮色花纹、脸是亮的，不算，那是 player）。
- not_person：不是玩家 —— 宠物 / 小动物（比如白色圆滚滚的小兽）、先祖（半透明发蓝光、头顶没有任何圆圈、摆固定姿势）、石像、灯笼、特效、UI 按钮等。
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
    """一批帧的内容块：规则 + 每帧（候选框坐标文字 + 画了网格和编号的图）。"""
    content: list[dict] = [{"type": "text", "text": RULES.format(self_hint=self_hint)}]
    for stem, frame, boxes in frames:
        listed = "；".join(f"{i}={_xyxy(b)}" for i, b in enumerate(boxes, 1)) or "没有候选框，只看有没有漏掉的人"
        content.append({"type": "text", "text": f"帧 {stem}：候选框 {listed}"})
        content.append(image_block(draw_candidates(frame, boxes), 85))
    return content
