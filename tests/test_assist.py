"""Claude 辅助标注（vision/assist.py）：挑帧、候选框图、提示词、解析、合并、Reviewer。不连 Claude、不要 GPU。"""

import json

import numpy as np
import pytest

from skydango.brain.claude import ClaudeError
from skydango.config import AssistConfig, Config
from skydango.vision.assist import (
    AssistLimit,
    FrameInput,
    Reviewer,
    assist_command,
    FrameReview,
    Verdict,
    apply_review,
    build_message,
    draw_candidates,
    draw_review,
    parse_review,
    pick_frames,
    review_report,
)
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


def _review(**kw):
    return FrameReview(verdicts=kw.get("v", {}), missing=kw.get("m", []), unsure=kw.get("u", ""), problems=kw.get("p", []))


def test_apply_review_keeps_people_uses_fixed_adds_missing():
    cands = [Rect(0, 0, 10, 10), Rect(20, 0, 10, 10), Rect(40, 0, 10, 10), Rect(60, 0, 10, 10)]
    r = _review(v={1: Verdict("self", Rect(0, 0, 12, 12), ""), 2: Verdict("duplicate", None, ""),
                   3: Verdict("not_person", None, ""), 4: Verdict("player_unlit", None, "")},
                m=[("player", Rect(90, 0, 10, 20), "")])
    assert apply_review(cands, r) == [
        ("self", Rect(0, 0, 12, 12)), ("player_unlit", Rect(60, 0, 10, 10)), ("player", Rect(90, 0, 10, 20))
    ]


def test_review_report_orders_sections():
    md = review_report([("f1", None), ("f2", _review(m=[("player", Rect(0, 0, 5, 5), "远处")])),
                        ("f3", _review(u="看不清")), ("f4", _review())])
    assert md.index("f1") < md.index("f2") < md.index("f3") and "远处" in md and "看不清" in md
    assert "其余 1 帧" in md


def test_draw_review_marks_unreviewed():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert draw_review(frame, [], [], None).max() > 0 and frame.max() == 0


def _fake_run(calls, fail_times=0, limit=False):
    state = {"fails": fail_times}

    def run(content):
        stems = [b["text"].split()[1].split("：")[0] for b in content if b["type"] == "text" and b["text"].startswith("帧 ")]
        calls.append(stems)
        if limit:
            raise ClaudeError("额度用完", limit=True)
        if state["fails"] > 0:
            state["fails"] -= 1
            raise ClaudeError("超时")
        body = {s: {"boxes": {"1": {"cls": "player"}}, "missing": [], "unsure": ""} for s in stems}
        return {"result": json.dumps(body), "usage": {"input_tokens": 10, "output_tokens": 2}}

    return run


def _frames(n, cands=1):
    return [FrameInput(f"f{i}", np.zeros((1080, 1920, 3), np.uint8), [Rect(0, 0, 10, 10)] * cands) for i in range(n)]


def test_reviewer_batches(tmp_path):
    calls = []
    out = Reviewer(_fake_run(calls), tmp_path, AssistConfig(batch=2, jobs=1), "m").review(_frames(5))
    assert sorted(map(len, calls)) == [1, 2, 2] and all(out[f"f{i}"] is not None for i in range(5))
    assert out["f0"].verdicts[1].cls == "player"


def test_reviewer_cache_hit_and_invalidation(tmp_path):
    calls = []
    cfg = AssistConfig(batch=5, jobs=1)
    Reviewer(_fake_run(calls), tmp_path, cfg, "m").review(_frames(3))
    out = Reviewer(_fake_run(calls), tmp_path, cfg, "m").review(_frames(3))
    assert len(calls) == 1 and out["f2"].verdicts[1].cls == "player"
    Reviewer(_fake_run(calls), tmp_path, cfg, "other").review(_frames(3))
    assert len(calls) == 2


def test_reviewer_retries_once_then_gives_up(tmp_path):
    calls = []
    out = Reviewer(_fake_run(calls, fail_times=1), tmp_path, AssistConfig(batch=5, jobs=1), "m").review(_frames(2))
    assert len(calls) == 2 and out["f0"] is not None
    calls2 = []
    out2 = Reviewer(_fake_run(calls2, fail_times=9), tmp_path / "b", AssistConfig(batch=5, jobs=1), "m").review(_frames(2))
    assert len(calls2) == 2 and out2 == {"f0": None, "f1": None}


def test_reviewer_limit_raises_and_keeps_done(tmp_path):
    with pytest.raises(AssistLimit):
        Reviewer(_fake_run([], limit=True), tmp_path, AssistConfig(batch=1, jobs=1), "m").review(_frames(2))


def test_reviewer_sends_frames_without_candidates(tmp_path):
    calls = []
    Reviewer(_fake_run(calls), tmp_path, AssistConfig(batch=5, jobs=1), "m").review(_frames(2, cands=0))
    assert calls == [["f0", "f1"]]


def test_reviewer_counts_usage(tmp_path):
    r = Reviewer(_fake_run([]), tmp_path, AssistConfig(batch=1, jobs=2), "m")
    r.review(_frames(3))
    assert r.usage["input_tokens"] == 30 and r.usage["output_tokens"] == 6


def test_assist_command_uses_model_and_no_tools():
    cmd = assist_command(["claude"], AssistConfig(model="sonnet"))
    assert cmd[cmd.index("--model") + 1] == "sonnet" and cmd[cmd.index("--tools") + 1] == ""
