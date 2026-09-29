import json

import numpy as np

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
    assert r.verdicts[1].cls == "not_object" and r.problems


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

    def run(cmd, env, cwd, content, timeout):
        texts = [b["text"] for b in content if b["type"] == "text" and b["text"].startswith("帧 ")]
        sent.append(cmd[cmd.index("--system-prompt") + 1])
        out = {}
        for t in texts:
            stem = t.split()[1].split("：")[0]
            hints = re.findall(r"(\d+)=\[[^\]]*\]\(猜 (\w+)\)", t)
            if hints:
                out[stem] = {"boxes": {i: {"cls": c} for i, c in hints}, "missing": [], "spirits": [], "unsure": ""}
            else:
                out[stem] = {"boxes": {}, "missing": [{"cls": "bench", "box": [100, 700, 400, 900]}], "spirits": ["P1"], "unsure": ""}
        return {"result": json.dumps(out), "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (["claude"], {}))
    monkeypatch.setattr("skydango.brain.claude.one_shot_message", run)
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
