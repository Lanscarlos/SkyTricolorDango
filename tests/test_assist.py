"""Claude 辅助标注（vision/assist.py）：挑帧、候选框图、提示词、解析、合并、Reviewer。不连 Claude、不要 GPU。"""

import numpy as np

from skydango.config import Config
from skydango.vision.assist import pick_frames


def test_pick_frames_drops_similar_and_forces_gap():
    base = np.full((36, 64), 100, np.uint8)
    thumbs = [base] * 5 + [base + 50] + [base + 50] * 30
    assert pick_frames(thumbs, min_change=30, max_gap=20) == [0, 5, 25]


def test_assist_config_defaults():
    c = Config().assist
    assert (c.min_change, c.max_gap, c.batch, c.jobs, c.model) == (30.0, 20, 5, 3, "sonnet")
    assert c.proposal_models == ["models/yolo11n.pt", "models/yolo11x.pt"]
