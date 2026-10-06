import json

import numpy as np

from model_seams import fake_claude
from skydango.config import AssistConfig, PerceptionConfig
from skydango.vision.assist import FrameInput
from skydango.vision.bubbles import Rect
from skydango.vision.objlabel import (
    OBJECT_PROMPT_VERSION,
    OBJECT_RULES,
    OBJECT_SYSTEM,
    OBJECTS_PROTOCOL,
    ObjectReview,
    ObjectVerdict,
    apply_object_review,
    build_object_message,
    objects_report,
    parse_object_review,
    people_in_labels,
    rewrite_labels,
)

CLASSES = PerceptionConfig().classes
W, H = 1920, 1080


def img():
    return np.zeros((H, W, 3), np.uint8)


def frame_input(stem="f0", people=(), cands=(), hints=()):
    return FrameInput(stem, img(), list(cands), people=list(people), hints=list(hints))


def test_build_message_lists_people_and_candidates():
    f = frame_input(people=[Rect(100, 100, 50, 150), Rect(300, 100, 50, 150)], cands=[Rect(600, 700, 200, 100)], hints=["bench"])
    content = build_object_message([f], AssistConfig())
    assert content[0] == {"type": "text", "text": OBJECT_RULES}
    text = content[1]["text"]
    assert "人物 P1=[100,100,150,250]；P2=[300,100,350,250]" in text and "候选 1=[600,700,800,800](猜 bench)" in text
    assert sum(b["type"] == "image" for b in content) == 1
    empty = build_object_message([frame_input()], AssistConfig())[1]["text"]
    assert "人物 无" in empty and "候选 无，只看有没有漏掉的物品" in empty


def reply(**item):
    body = {"boxes": {}, "missing": [], "spirits": [], "unsure": ""}
    body.update(item)
    return json.dumps({"f0": body})


def test_parse_reads_boxes_missing_spirits():
    f = frame_input(people=[Rect(0, 0, 10, 10), Rect(20, 0, 10, 10)], cands=[Rect(600, 700, 200, 100)])
    out = parse_object_review(reply(boxes={"1": {"cls": "bench", "fixed_box": [590, 690, 810, 810]}},
                                    missing=[{"cls": "spirit", "box": [900, 300, 980, 500]}, {"cls": "candle", "box": [0, 0, 9, 9]}],
                                    spirits=["P2"]), [f])
    r = out["f0"]
    assert r.verdicts[1].cls == "bench" and r.verdicts[1].fixed == Rect(590, 690, 220, 120)
    assert [(c, b) for c, b, _ in r.missing] == [("spirit", Rect(900, 300, 80, 200))]  # 不认识的类别不收
    assert r.spirits == [2]


def test_parse_bad_spirit_ids_go_to_problems():
    f = frame_input(people=[Rect(0, 0, 10, 10), Rect(20, 0, 10, 10)])
    r = parse_object_review(reply(spirits=["P9", 3, "P1"]), [f])["f0"]
    assert r.spirits == [1] and len(r.problems) == 2


def test_parse_missing_candidate_verdict_is_a_problem():
    f = frame_input(cands=[Rect(600, 700, 200, 100)])
    r = parse_object_review(reply(), [f])["f0"]
    assert r.verdicts[1].cls == "unjudged" and r.problems  # 没判 ≠ 不是物品：已有的标注要留着


def test_apply_object_review_keeps_objects_uses_fixed_adds_missing():
    cands = [Rect(0, 0, 10, 10), Rect(50, 50, 10, 10), Rect(90, 90, 10, 10)]
    review = ObjectReview({1: ObjectVerdict("bench", Rect(1, 1, 8, 8), ""), 2: ObjectVerdict("not_object", None, ""),
                           3: ObjectVerdict("duplicate", None, "")},
                          [("bonfire", Rect(200, 200, 30, 30), "")], [], "")
    assert apply_object_review(cands, review) == [("bench", Rect(1, 1, 8, 8)), ("bonfire", Rect(200, 200, 30, 30))]


LABELS = (
    "0 0.100000 0.200000 0.050000 0.100000\n"   # player → P1
    "1 0.100000 0.100000 0.050000 0.020000\n"   # name_tag
    "6 0.500000 0.500000 0.100000 0.100000\n"   # 旧的 bench
    "\n"
    "4 1.200000 0.500000 0.050000 0.100000\n"   # 越界的 player_unlit → P2
    "3 0.500000 0.800000 0.050000 0.100000\n"   # self
)


def test_people_in_labels_order():
    boxes = people_in_labels(LABELS + "9 0.3 0.3 0.05 0.1\n", CLASSES, W, H)
    assert len(boxes) == 2 and boxes[0] == Rect(144, 162, 96, 108)  # 已经是 spirit 的、self 都不算


def test_rewrite_keeps_people_lines_and_replaces_objects():
    out = rewrite_labels(LABELS, CLASSES, [("bonfire", Rect(960, 540, 96, 108))], [], W, H).splitlines()
    assert out[:4] == ["0 0.100000 0.200000 0.050000 0.100000", "1 0.100000 0.100000 0.050000 0.020000",
                       "4 1.200000 0.500000 0.050000 0.100000", "3 0.500000 0.800000 0.050000 0.100000"]
    assert out[4].startswith("7 ") and len(out) == 5 and "" not in out


def test_rewrite_turns_player_into_spirit():
    out = rewrite_labels(LABELS, CLASSES, [], [2], W, H).splitlines()
    assert "9 1.200000 0.500000 0.050000 0.100000" in out and out[0].startswith("0 ")


def test_rewrite_twice_is_stable():
    from skydango.vision.objlabel import objects_in_labels

    once = rewrite_labels(LABELS, CLASSES, [("bench", Rect(100, 800, 300, 200))], [1], W, H)
    # 重跑：已有的物品（含人改成的先祖）当候选、Claude 认可 → 原样写回；原来的 P2 成了 P1、不再有 P2
    existing = objects_in_labels(once, CLASSES, W, H)
    assert [c for c, _ in existing] == ["spirit", "bench"]
    assert len(people_in_labels(once, CLASSES, W, H)) == 1
    twice = rewrite_labels(once, CLASSES, existing, [], W, H)
    assert sorted(l.split()[0] for l in twice.splitlines()) == sorted(l.split()[0] for l in once.splitlines())
    again = objects_in_labels(twice, CLASSES, W, H)
    assert [c for c, _ in again] == ["spirit", "bench"]
    assert all(abs(a.x - b.x) <= 1 and abs(a.y - b.y) <= 1 and abs(a.w - b.w) <= 1 for (_, a), (_, b) in zip(existing, again))


def test_rewrite_keeps_unparseable_lines():
    out = rewrite_labels("0 0.1 0.2 0.05 0.1\n乱码行\n", CLASSES, [], [], W, H)
    assert "乱码行" in out.splitlines()


def test_objects_report_orders_sections():
    spirit = ObjectReview({}, [], [1], "")
    added = ObjectReview({}, [("bench", Rect(0, 0, 10, 10), "长椅")], [], "拿不准")
    md = objects_report([("a", None), ("b", added), ("c", spirit), ("d", ObjectReview({}, [], [], ""))])
    assert md.startswith("# 待核对清单（物品模式）")
    assert md.index("人物框改成了先祖") < md.index("- c") < md.index("Claude 补了物品框") < md.index("- b") < md.index("没核对成") < md.index("- a")
    assert "拿不准" in md


def test_protocol_version_and_system():
    assert OBJECTS_PROTOCOL.version == OBJECT_PROMPT_VERSION and OBJECTS_PROTOCOL.system == OBJECT_SYSTEM


# ---- perception label <数据集> --objects ----
def _dataset(tmp_path, monkeypatch):
    """两张图的数据集：0000 有一个 player + 一个 name_tag，0001 没有标注文件。假 Claude：
    没有候选时补一张 bench、说 P1 是先祖；有候选时照它的猜测全部认可。"""
    import re

    from skydango import cli
    from skydango.imageio import imwrite

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for i in range(2):
        imwrite(root / "images" / "train" / f"{i:04d}.jpg", np.full((H, W, 3), 40 * i, np.uint8))
    (root / "labels" / "train" / "0000.txt").write_text("0 0.500000 0.500000 0.050000 0.200000\n1 0.500000 0.350000 0.060000 0.030000\n",
                                                        encoding="utf-8")
    sent = []

    def run(cmd, env, cwd, content, timeout, on_message=None):
        texts = [b["text"] for b in content if b["type"] == "text" and b["text"].startswith("帧 ")]
        sent.append(cmd[cmd.index("--system-prompt") + 1])
        out = {}
        for t in texts:
            stem = t.split()[1].split("：")[0]
            hints = re.findall(r"(\d+)=\[[^\]]*\]\((?:猜|已标) (\w+)\)", t)
            if hints:
                out[stem] = {"boxes": {i: {"cls": c} for i, c in hints}, "missing": [], "spirits": [], "unsure": ""}
            else:
                out[stem] = {"boxes": {}, "missing": [{"cls": "bench", "box": [100, 700, 400, 900]}], "spirits": ["P1"], "unsure": ""}
        return {"result": json.dumps(out), "usage": {"input_tokens": 1, "output_tokens": 1}}

    fake_claude(monkeypatch, ["claude"])
    monkeypatch.setattr("skydango.models.claude_code.one_shot_message", run)
    return root, sent


def classes_in(path):
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()]


def test_objects_mode_rewrites_labels_and_backs_up(tmp_path, monkeypatch):
    from skydango import cli

    root, sent = _dataset(tmp_path, monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])
    assert classes_in(root / "labels" / "train" / "0000.txt") == ["9", "1", "6"]  # player → 先祖，name_tag 不动，补一张座位
    assert classes_in(root / "labels" / "train" / "0001.txt") == ["6"]  # 没标注文件的新图：P1 不存在，只补座位
    (backup,) = (root / "_backup").iterdir()  # 只跑了一次：一份备份
    assert (backup / "train" / "0000.txt").read_text(encoding="utf-8").startswith("0 0.500000")
    assert (root / "_preview_objects" / "0000.jpg").exists()
    report = (root / "_assist" / "objects.md").read_text(encoding="utf-8")
    assert report.index("- 0000：P1") < report.index("Claude 补了物品框")
    assert "9: spirit" in (root / "data.yaml").read_text(encoding="utf-8")
    assert sent and all(s.startswith("你是游戏截图的目标检测标注员") for s in sent)


def test_objects_mode_rerun_is_stable(tmp_path, monkeypatch):
    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])
    first = {p.name: sorted(classes_in(p)) for p in (root / "labels" / "train").glob("*.txt")}
    cli.main(["perception", "label", str(root), "--objects"])  # 已有的物品当候选再核对一遍，认可 → 不丢、不重复
    # 行的顺序会变（改成先祖的那行这次当物品写在末尾），YOLO 不在乎顺序
    assert {p.name: sorted(classes_in(p)) for p in (root / "labels" / "train").glob("*.txt")} == first


def test_objects_mode_rejects_other_modes(tmp_path, monkeypatch):
    import pytest

    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["perception", "label", str(root), "--objects", "--assist"])


def test_objects_mode_needs_object_classes(tmp_path, monkeypatch):
    import pytest

    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    (tmp_path / "c.toml").write_text('[perception]\nclasses = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]\n',
                                     encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["-c", str(tmp_path / "c.toml"), "perception", "label", str(root), "--objects"])


# ---- 评审修正：已有标注的安全 ----
def test_parse_tolerates_non_list_fields():
    f = frame_input(people=[Rect(0, 0, 10, 10)])
    r = parse_object_review(reply(spirits=2, missing=3), [f])["f0"]
    assert r.spirits == [] and r.missing == [] and len(r.problems) == 2


def test_build_message_marks_existing_labels():
    f = frame_input(cands=[Rect(600, 700, 200, 100), Rect(10, 10, 50, 50)], hints=["已标 bench", "猜 spirit"])
    text = build_object_message([f], AssistConfig())[1]["text"]
    assert "1=[600,700,800,800](已标 bench)" in text and "2=[10,10,60,60](猜 spirit)" in text
    assert "已标" in OBJECT_RULES


def test_apply_keeps_unjudged_existing_and_reports_changes():
    cands = [Rect(0, 0, 10, 10), Rect(50, 50, 10, 10), Rect(90, 90, 10, 10), Rect(200, 200, 10, 10)]
    hints = ["已标 bench", "已标 spirit", "已标 instrument", "猜 bonfire"]
    review = ObjectReview({1: ObjectVerdict("unjudged", None, ""), 2: ObjectVerdict("not_object", None, "是玩家"),
                           3: ObjectVerdict("instrument", Rect(88, 88, 14, 14), ""), 4: ObjectVerdict("unjudged", None, "")},
                          [], [], "")
    objects = apply_object_review(cands, review, hints)
    assert objects == [("bench", Rect(0, 0, 10, 10)), ("instrument", Rect(88, 88, 14, 14))]  # 没判的已有标注留着，没判的猜测丢掉
    assert review.changed == ["2 号已有的先祖删了（not_object：是玩家）", "3 号已有的乐器框改了"]
    md = objects_report([("f0", review)])
    assert "已有标注被删 / 改" in md and "- f0：2 号已有的先祖删了" in md


def _count_calls(monkeypatch):
    """包一层假 Claude：数调用次数。"""
    import skydango.models.claude_code as claude

    real = claude.one_shot_message
    calls = []

    def run(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(claude, "one_shot_message", run)
    return calls


def test_objects_mode_second_run_skips_done_frames(tmp_path, monkeypatch):
    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])
    calls = _count_calls(monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])  # 额度用完后重跑 / 再跑一次：做过的不再送
    assert calls == []
    report = (root / "_assist" / "objects.md").read_text(encoding="utf-8")
    assert "- 0000：P1" in report  # 人改先祖的帧一直在清单里


def test_objects_mode_skips_frames_you_edited(tmp_path, monkeypatch):
    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])
    label = root / "labels" / "train" / "0000.txt"
    edited = label.read_text(encoding="utf-8") + "6 0.200000 0.900000 0.100000 0.050000\n"
    label.write_text(edited, encoding="utf-8")  # 在 X-AnyLabeling 里改过
    calls = _count_calls(monkeypatch)
    cli.main(["perception", "label", str(root), "--objects"])
    assert calls == [] and label.read_text(encoding="utf-8") == edited
    assert "你改过" in (root / "_assist" / "objects.md").read_text(encoding="utf-8")
    cli.main(["perception", "label", str(root), "--objects", "--recheck"])
    assert calls  # --recheck 才重新核对


def test_objects_mode_writes_data_yaml_first(tmp_path, monkeypatch):
    import pytest

    from skydango import cli
    from skydango.brain.claude import ClaudeError

    root, _ = _dataset(tmp_path, monkeypatch)

    def limit(*a, **k):
        raise ClaudeError("额度", limit=True)

    monkeypatch.setattr("skydango.models.claude_code.one_shot_message", limit)
    with pytest.raises(SystemExit):
        cli.main(["perception", "label", str(root), "--objects"])
    assert "9: spirit" in (root / "data.yaml").read_text(encoding="utf-8")


def test_objects_mode_skips_augmented_images(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.imageio import imwrite

    root, _ = _dataset(tmp_path, monkeypatch)
    imwrite(root / "images" / "train" / "0000_blur.jpg", np.zeros((H, W, 3), np.uint8))
    cli.main(["perception", "label", str(root), "--objects"])
    assert not (root / "labels" / "train" / "0000_blur.txt").exists()
    assert not (root / "_preview_objects" / "0000_blur.jpg").exists()


# ---- typing（头顶气泡）并进物品模式 ----
TYPING = CLASSES.index("typing")


def test_prompt_mentions_typing_and_version_bumped():
    assert "typing" in OBJECT_RULES and OBJECT_PROMPT_VERSION >= 3


def test_object_names_has_no_typing():
    from skydango.vision.objlabel import LABEL_CLASSES
    from skydango.vision.people import OBJECT_NAMES

    assert "typing" not in OBJECT_NAMES  # 感知层的 objects() 不能把气泡当物品报出来
    assert "typing" in LABEL_CLASSES and all(c in LABEL_CLASSES for c in OBJECT_NAMES)


def test_parse_reads_typing_candidates_and_missing():
    f = frame_input(cands=[Rect(600, 200, 80, 40)])
    r = parse_object_review(reply(boxes={"1": {"cls": "typing"}}, missing=[{"cls": "typing", "box": [900, 300, 980, 340]}]), [f])["f0"]
    assert r.verdicts[1].cls == "typing" and not r.problems
    assert [(c, b) for c, b, _ in r.missing] == [("typing", Rect(900, 300, 80, 40))]
    assert apply_object_review(f.candidates, r) == [("typing", Rect(600, 200, 80, 40)), ("typing", Rect(900, 300, 80, 40))]


PEOPLE_LABELS = (
    "0 0.100000 0.200000 0.050000 0.100000\n"   # player
    "4 0.300000 0.200000 0.050000 0.100000\n"   # player_unlit
    "1 0.100000 0.100000 0.050000 0.020000\n"   # name_tag
    "2 0.100000 0.130000 0.020000 0.020000\n"   # social_ring
    "3 0.500000 0.800000 0.050000 0.100000\n"   # self
)


def test_rewrite_writes_typing_lines_and_keeps_people():
    out = rewrite_labels(PEOPLE_LABELS, CLASSES, [("typing", Rect(960, 540, 96, 108))], [], W, H).splitlines()
    assert out[:5] == PEOPLE_LABELS.splitlines()
    assert len(out) == 6 and out[5].split()[0] == str(TYPING)


def test_rewrite_replaces_old_typing_lines_like_objects():
    old = PEOPLE_LABELS + f"{TYPING} 0.500000 0.100000 0.040000 0.030000\n6 0.500000 0.500000 0.100000 0.100000\n"
    out = rewrite_labels(old, CLASSES, [("bonfire", Rect(960, 540, 96, 108))], [], W, H).splitlines()
    assert out[:5] == PEOPLE_LABELS.splitlines()
    assert [l.split()[0] for l in out[5:]] == ["7"]  # 旧的气泡、座位都换掉了


def test_existing_typing_is_a_candidate_and_kept_when_unjudged():
    from skydango.vision.objlabel import EXISTING, objects_in_labels

    text = PEOPLE_LABELS + f"{TYPING} 0.500000 0.100000 0.040000 0.030000\n"
    existing = objects_in_labels(text, CLASSES, W, H)
    assert [c for c, _ in existing] == ["typing"]
    review = ObjectReview({1: ObjectVerdict("unjudged", None, "")}, [], [], "")
    assert apply_object_review([b for _, b in existing], review, [EXISTING + "typing"]) == existing
    review = ObjectReview({1: ObjectVerdict("not_object", None, "")}, [], [], "")
    apply_object_review([b for _, b in existing], review, [EXISTING + "typing"])
    assert review.changed == ["1 号已有的气泡删了（not_object）"]


def test_report_and_preview_know_typing():
    from skydango.vision.objlabel import OBJECT_COLORS, draw_objects_preview

    r = ObjectReview({}, [("typing", Rect(0, 0, 10, 10), "")], [], "")
    md = objects_report([("f0", r)])
    assert "- f0：气泡" in md and "气泡品红" in md
    assert OBJECT_COLORS["typing"] == (255, 0, 255)
    out = draw_objects_preview(img(), [("typing", Rect(100, 100, 50, 30))], [], [], [], ObjectReview({}, [], [], ""))
    assert tuple(out[100, 120]) == (255, 0, 255)


def test_objects_mode_only_processes_matching_frames(tmp_path, monkeypatch):
    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    before = (root / "labels" / "train" / "0000.txt").read_text(encoding="utf-8")
    cli.main(["perception", "label", str(root), "--objects", "--only", "0001"])
    assert (root / "labels" / "train" / "0000.txt").read_text(encoding="utf-8") == before
    assert classes_in(root / "labels" / "train" / "0001.txt") == ["6"]
    assert not (root / "_preview_objects" / "0000.jpg").exists()
    report = (root / "_assist" / "objects.md").read_text(encoding="utf-8")
    assert "0000" not in report and "- 0001" in report


def test_objects_mode_only_accepts_globs_and_commas(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.imageio import imwrite

    root, _ = _dataset(tmp_path, monkeypatch)
    imwrite(root / "images" / "train" / "rec_0002.jpg", np.zeros((H, W, 3), np.uint8))
    cli.main(["perception", "label", str(root), "--objects", "--only", "00*.jpg,nothing", "--only", "zzz"])
    assert (root / "labels" / "train" / "0001.txt").exists() and not (root / "labels" / "train" / "rec_0002.txt").exists()
    assert classes_in(root / "labels" / "train" / "0000.txt") == ["9", "1", "6"]


def test_objects_mode_only_without_match_exits(tmp_path, monkeypatch):
    import pytest

    from skydango import cli

    root, sent = _dataset(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--only"):
        cli.main(["perception", "label", str(root), "--objects", "--only", "nope*"])
    assert sent == [] and not (root / "_backup").exists()


def test_objects_mode_needs_typing_class(tmp_path, monkeypatch):
    import pytest

    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    (tmp_path / "c.toml").write_text('[perception]\nclasses = ["player", "name_tag", "social_ring", "self", "player_unlit", "x",'
                                     ' "bench", "bonfire", "instrument", "spirit"]\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="typing"):
        cli.main(["-c", str(tmp_path / "c.toml"), "perception", "label", str(root), "--objects"])


def test_only_needs_objects_mode(tmp_path, monkeypatch):
    import pytest

    from skydango import cli

    root, _ = _dataset(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--objects"):
        cli.main(["perception", "label", str(root), "--only", "0001"])


# ---- 辅助标注审查留下的小问题（进度文档「没做完 / 待办」第 6 条），物品模式同样的毛病 ----
def test_parse_object_boxes_as_list_uses_order_as_ids():
    f = frame_input(cands=[Rect(600, 700, 200, 100), Rect(0, 0, 50, 50)])
    r = parse_object_review(reply(boxes=[{"cls": "bench"}, {"cls": "typing"}]), [f])["f0"]
    assert r.verdicts[1].cls == "bench" and r.verdicts[2].cls == "typing" and r.problems == []


def test_parse_object_unreadable_boxes_means_not_reviewed():
    f = frame_input(cands=[Rect(600, 700, 200, 100)])
    assert parse_object_review(reply(boxes="1 号是座位"), [f]) == {}


def test_parse_object_nonfinite_coordinates_are_problems():
    f = frame_input(cands=[Rect(600, 700, 200, 100)])
    text = ('{"f0": {"boxes": {"1": {"cls": "bench", "fixed_box": [NaN, 0, 10, 10]}}, '
            '"missing": [{"cls": "spirit", "box": [0, 0, Infinity, 10]}, {"cls": "spirit", "box": [900, 300, 980, 500]}], '
            '"spirits": [], "unsure": ""}}')
    r = parse_object_review(text, [f])["f0"]
    assert r.verdicts[1].cls == "bench" and r.verdicts[1].fixed is None
    assert [(c, b) for c, b, _ in r.missing] == [("spirit", Rect(900, 300, 80, 200))]
    assert len(r.problems) == 2 and all("坐标" in p for p in r.problems)


def test_parse_object_normalizes_frame_names():
    f = frame_input(cands=[Rect(600, 700, 200, 100)])
    text = json.dumps({"帧 f0.jpg": {"boxes": {"1": {"cls": "bonfire"}}, "missing": [], "spirits": [], "unsure": ""}},
                      ensure_ascii=False)
    assert parse_object_review(text, [f])["f0"].verdicts[1].cls == "bonfire"


def test_object_prompt_uses_actual_frame_size():
    small = FrameInput("f0", np.zeros((720, 1280, 3), np.uint8), [])
    rules = build_object_message([small], AssistConfig())[0]["text"]
    assert "原图 1280×720" in rules and "1920×1080" not in rules
    assert build_object_message([frame_input()], AssistConfig())[0]["text"] == OBJECT_RULES  # 1920×1080 的提示词一字不变
    assert "原图 1920×1080" in OBJECT_RULES and OBJECT_PROMPT_VERSION == 4


def test_bench_rule_one_box_per_seat():
    """座位的框法（10-02 用户定的）：一个能坐的地方一个框，连成一件的长椅整件一个框，桌子和桌上的东西不框。"""
    assert "一个能坐的地方一个框" in OBJECT_RULES
    assert "桌子" in OBJECT_RULES and "不框" in OBJECT_RULES
