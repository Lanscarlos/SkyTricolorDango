from skydango.vision.track import Rect
from skydango.vision.icons_map import (UNKNOWN, Icon, describe_icons, icon_crop, map_label, owner_of, under_gap, vote)

RING = Rect(950, 300, 100, 100)  # 圈心 (1000, 350)

def test_owner_order_person_beats_objects():
    person = Rect(955, 380, 90, 220)
    bench = Rect(900, 420, 200, 120)
    assert owner_of(RING, [person], [], [("bench", bench)], 0.25, 1.0) == "person"

def test_owner_spirit_then_object_then_map():
    sp = Rect(955, 380, 90, 220)
    assert owner_of(RING, [], [sp], [], 0.25, 1.0) == "spirit"
    assert owner_of(RING, [], [], [("bonfire", Rect(940, 400, 120, 100))], 0.25, 1.0) == "bonfire"
    assert owner_of(RING, [], [], [], 0.25, 1.0) == "map"

def test_owner_far_box_is_map():
    assert owner_of(RING, [Rect(1400, 380, 90, 220)], [], [], 0.25, 1.0) == "map"   # 横着差太远
    assert owner_of(RING, [Rect(955, 700, 90, 100)], [], [], 0.25, 1.0) == "map"    # 圈比人高出一个多身高

def test_owner_nearest_of_two_people():
    near, far = Rect(955, 360, 90, 220), Rect(955, 420, 90, 220)
    assert under_gap(RING, near, 0.25, 1.0) < under_gap(RING, far, 0.25, 1.0)

def test_map_label():
    assert map_label("candle", "bonfire") == "篝火点燃"
    assert map_label("candle", "map") == "可以点的蜡烛 / 灯"
    assert map_label(None, "map") == "不认识的图标" and map_label(UNKNOWN, "spirit") == "不认识的图标"
    assert map_label("hand", "map") == "牵手"  # 其余查 KIND_NAMES

def test_vote():
    assert vote(["candle", UNKNOWN, "candle", "sit"]) == "candle"
    assert vote([UNKNOWN, UNKNOWN]) == UNKNOWN
    assert vote([]) is None

def test_icon_crop_square_and_clipped():
    import numpy as np
    f = np.zeros((1080, 1920, 3), np.uint8)
    assert icon_crop(f, Rect(900, 300, 100, 80), 1.3).shape[:2] == (130, 130)
    edge = icon_crop(f, Rect(1880, 1050, 60, 60), 1.3)   # 贴边：只裁画面里的部分
    assert edge.shape[0] <= 78 and edge.shape[1] <= 78 and edge.size > 0
    assert icon_crop(f, Rect(3000, 3000, 50, 50), 1.3).size == 0   # 完全在外面：空

def test_describe_icons():
    b = Rect(0, 0, 10, 10)
    icons = [Icon(1, "sit", "map", "坐下", b, "右边"), Icon(2, UNKNOWN, "map", "不认识的图标", b, "左边"),
             Icon(3, "memory", "map", "留影", b, "左边")]
    assert describe_icons(icons) == "坐下（右边）、留影（左边）、不认识的图标 1 个"
    assert describe_icons([]) == ""
