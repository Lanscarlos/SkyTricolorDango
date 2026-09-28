"""Claude 辅助标注（`perception label --assist`）：挑帧 → 检测器出人物候选框 → claude -p 核对 → 合并成 YOLO 标注。

设计见 docs/superpowers/specs/2026-09-28-assist-labeling-design.md。这里只放能单独测的逻辑，
命令行编排在 cli.py 的 `_perception_label_assist`；真正调 Claude 的函数由外面传给 `Reviewer`。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
