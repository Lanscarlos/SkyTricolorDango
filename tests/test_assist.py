"""Claude 辅助标注（vision/assist.py）：挑帧、候选框图、提示词、解析、合并、Reviewer。不连 Claude、不要 GPU。"""

import numpy as np

from skydango.config import Config
from skydango.vision.assist import build_message, draw_candidates, parse_review, pick_frames
from skydango.vision.bubbles import Rect


def test_pick_frames_drops_similar_and_forces_gap():
    base = np.full((36, 64), 100, np.uint8)
    thumbs = [base] * 5 + [base + 50] + [base + 50] * 30
    assert pick_frames(thumbs, min_change=30, max_gap=20) == [0, 5, 25]


def test_assist_config_defaults():
    c = Config().assist
    assert (c.min_change, c.max_gap, c.batch, c.jobs, c.model) == (30.0, 20, 5, 3, "sonnet")
    assert c.proposal_models == ["models/yolo11n.pt", "models/yolo11x.pt"]


def test_draw_candidates_keeps_original():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = draw_candidates(frame, [Rect(100, 200, 50, 120)])
    assert frame.max() == 0 and out.shape == frame.shape and out.max() > 0


def test_build_message_lists_frames_and_rules():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    msg = build_message([("a", frame, [Rect(10, 20, 30, 40)]), ("b", frame, [])], "蓝紫色圆背包")
    texts = [b["text"] for b in msg if b["type"] == "text"]
    assert "头顶有圆圈" in texts[0] and "蓝紫色圆背包" in texts[0] and "duplicate" in texts[0]
    assert "帧 a" in texts[1] and "1=[10,20,40,60]" in texts[1]
    assert "帧 b" in texts[2] and "没有候选框" in texts[2]
    assert [b["type"] for b in msg].count("image") == 2


def test_parse_reads_verdicts_missing_unsure():
    text = '{"a": {"boxes": {"1": {"cls": "self", "note": "", "fixed_box": [0, 0, 50, 90]}, "2": {"cls": "duplicate", "note": "留 1"}},'
    text += ' "missing": [{"cls": "player_unlit", "box": [300, 300, 340, 400], "note": "黑影"}], "unsure": "看不清"}}'
    r = parse_review(text, {"a": 2}, 1920, 1080)["a"]
    assert r.verdicts[1].cls == "self" and r.verdicts[1].fixed == Rect(0, 0, 50, 90)
    assert r.verdicts[2].cls == "duplicate"
    assert r.missing == [("player_unlit", Rect(300, 300, 40, 100), "黑影")] and r.unsure == "看不清"


def test_parse_marks_missing_ids_and_bad_classes():
    r = parse_review('{"a": {"boxes": {"1": {"cls": "npc"}}, "missing": [], "unsure": ""}}', {"a": 2}, 1920, 1080)["a"]
    assert r.verdicts[1].cls == r.verdicts[2].cls == "not_person" and len(r.problems) == 2


def test_parse_normalizes_and_clips_boxes():
    text = ('{"a": {"boxes": {}, "missing": [{"cls": "player", "box": [2000, 900, 1800, 1200]}, '
            '{"cls": "player", "box": [5000, 10, 6000, 20]}, {"cls": "pet", "box": [1, 1, 50, 50]}]}}')
    assert parse_review(text, {"a": 0}, 1920, 1080)["a"].missing == [("player", Rect(1800, 900, 120, 180), "")]


def test_parse_tolerates_fences_and_prose():
    text = '好的，结果如下：\n```json\n{"a": {"boxes": {"1": {"cls": "player"}}, "missing": [], "unsure": ""}}\n```\n以上。'
    assert parse_review(text, {"a": 1, "b": 1}, 1920, 1080).keys() == {"a"}


def test_parse_garbage_returns_empty():
    assert parse_review('{"a": {"boxes": ', {"a": 1}, 1920, 1080) == {}
