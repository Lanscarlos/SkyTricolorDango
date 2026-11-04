"""检测器认出单类模型（person 当 player 输出）。"""
import numpy as np

from skydango.vision.detect import decode, single_class_names

NEW = ["person", "name_tag", "social_ring", "self", "unused_4", "typing", "bench", "bonfire", "instrument", "unused_9"]
OLD = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]


def test_single_class_names_new():
    names, single = single_class_names(NEW)
    assert single and names[0] == "player" and names[4] == "" and names[9] == "" and names[3] == "self"


def test_single_class_names_old():
    assert single_class_names(OLD) == (OLD, False)


def _pred(rows, nc):
    """rows: (cls, cx, cy, w, h, score) → (1, 4 + nc, N)"""
    out = np.zeros((4 + nc, len(rows)), dtype=np.float32)
    for i, (c, cx, cy, w, h, s) in enumerate(rows):
        out[0:4, i] = (cx, cy, w, h)
        out[4 + c, i] = s
    return out[None]


def test_decode_drops_unused():
    names, _ = single_class_names(NEW)
    pred = _pred([(4, 100, 100, 40, 80, 0.9), (0, 300, 300, 40, 80, 0.9)], len(names))
    dets = decode(pred, names, 1.0, (0, 0), (640, 640), 0.35, 0.5)
    assert [d.cls for d in dets] == ["player"]


def test_decode_end_to_end_drops_unused():
    names, _ = single_class_names(NEW)
    pred = np.array([[10, 10, 50, 90, 0.9, 9], [200, 200, 240, 280, 0.9, 0]], dtype=np.float32)[None]
    dets = decode(pred, names, 1.0, (0, 0), (640, 640), 0.35, 0.5)
    assert [d.cls for d in dets] == ["player"]
