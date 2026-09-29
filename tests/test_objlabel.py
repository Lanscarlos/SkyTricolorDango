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
