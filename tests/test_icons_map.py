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


# ---- Task 2：两种分类器 ----
import cv2
import numpy as np

from skydango.game.social import SCALES, IconClassifier, load_icons
from skydango.vision.icons_map import IconGallery, classify_template

ICONS = load_icons("assets/social")


def _hand_region():
    frame = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)
    icon = cv2.imread("assets/social/hand.png")
    h, w = icon.shape[:2]
    frame[400 - h // 2 : 400 - h // 2 + h, 1400 - w // 2 : 1400 - w // 2 + w] = icon
    return frame[344:456, 1344:1456]


def test_classifier_scales_param_default_unchanged():
    clf = IconClassifier(ICONS)
    region = _hand_region()
    assert clf.classify(region) == clf.classify(region, SCALES)


def test_classify_template_rescales_by_box():
    clf = IconClassifier(ICONS)
    region = _hand_region()
    big = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)  # 近处的大圈
    kind, score = classify_template(clf, big, Rect(0, 0, 224, 224))
    assert kind == "hand" and score > 0.7
    assert classify_template(clf, np.zeros((0, 0, 3), np.uint8), Rect(0, 0, 10, 10)) == (UNKNOWN, 0.0)


class FakeEmbedder:
    """按裁图平均颜色出 3 维单位向量。"""
    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, img):
        if self.fail:
            return None
        v = img.reshape(-1, 3).mean(axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v


def _img(bgr):
    return np.full((20, 20, 3), bgr, np.uint8)


RED, BLUE, GREY = (0, 0, 255), (255, 0, 0), (200, 200, 200)


def test_gallery_nearest():
    g = IconGallery(FakeEmbedder(), {"sit": [_img(RED), _img(RED)], "memory": [_img(BLUE)]}, 0.9, 0.1)
    assert g.classify(_img(RED))[0] == "sit"
    assert g.classify(_img(GREY))[0] == UNKNOWN


def test_gallery_single_kind_no_margin():
    g = IconGallery(FakeEmbedder(), {"sit": [_img(RED)]}, 0.9, 0.5)
    assert g.classify(_img(RED))[0] == "sit"


def test_gallery_embed_error_is_unknown():
    g = IconGallery(FakeEmbedder(), {"sit": [_img(RED)]}, 0.9, 0.1)
    g.embedder = FakeEmbedder(fail=True)
    assert g.classify(_img(RED)) == (UNKNOWN, 0.0)


def test_gallery_load_skips_underscore_and_empty(tmp_path):
    (tmp_path / "sit").mkdir()
    (tmp_path / "_removed").mkdir()
    (tmp_path / "music").mkdir()
    cv2.imwrite(str(tmp_path / "sit" / "a.jpg"), _img(RED))
    cv2.imwrite(str(tmp_path / "_removed" / "b.jpg"), _img(BLUE))
    g = IconGallery.load(tmp_path, FakeEmbedder(), 0.9, 0.1)
    assert g.kinds == ["sit"]
