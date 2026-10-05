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


# ---- MapIcons：运行时（投票、选分类器、难例、存裁图） ----
import json  # noqa: E402

from skydango.config import IconsConfig  # noqa: E402
from skydango.vision.icons_map import MapIcons  # noqa: E402
from skydango.vision.track import Track  # noqa: E402


class FakeTemplate:
    """假模板分类器（IconClassifier 的接口）：依次返回 seq 里的 (kind, score)，用完了一直返回最后一个。"""

    def __init__(self, *seq, min_score=0.75):
        self.seq = list(seq)
        self.min_score = min_score
        self.calls = 0

    def classify(self, region, scales=None):
        self.calls += 1
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]


class FakeGallery:
    def __init__(self, kind, score, match=0.75):
        self.kind, self.score, self.match = kind, score, match
        self.calls = 0

    def classify(self, crop):
        self.calls += 1
        return self.kind, self.score


def _ring_track(tid=7, hits=1):
    return Track(tid, "social_ring", Rect(950, 300, 100, 100), 0.9, 0.0, 0.0, hits=hits)


def _frame():
    return np.full((1080, 1920, 3), 60, np.uint8)


def test_map_icons_votes_capped_and_icon_is_majority():
    tpl = FakeTemplate(("bench", 0.9), ("bench", 0.9), ("hand", 0.9), ("hand", 0.9), ("hand", 0.9), ("bench", 0.9))
    m = MapIcons(IconsConfig(enabled=True, save=False), tpl, None, None)
    ring = _ring_track()
    for i in range(6):
        ring.hits = i + 1
        m.observe(_frame(), ring, "map", float(i))
    assert len(ring.data["votes"]) == 5
    assert ring.data["votes"] == ["bench", "hand", "hand", "hand", "bench"]
    assert ring.data["icon"] == "hand" and ring.data["icon_score"] == 0.9


def test_map_icons_template_none_kind_is_unknown():
    m = MapIcons(IconsConfig(enabled=True, save=False), FakeTemplate((None, 0.3)), None, None)
    ring = _ring_track()
    m.observe(_frame(), ring, "map", 0.0)
    assert ring.data["votes"] == [UNKNOWN] and ring.data["icon"] == UNKNOWN


def test_map_icons_dino_uses_gallery_else_template():
    tpl, gal = FakeTemplate(("hand", 0.9)), FakeGallery("bench", 0.8)
    ring = _ring_track()
    MapIcons(IconsConfig(enabled=True, classifier="dino", save=False), tpl, gal, None).observe(_frame(), ring, "map", 0.0)
    assert ring.data["icon"] == "bench" and gal.calls == 1 and tpl.calls == 0
    ring = _ring_track()
    MapIcons(IconsConfig(enabled=True, classifier="dino", save=False), tpl, None, None).observe(_frame(), ring, "map", 0.0)
    assert ring.data["icon"] == "hand" and tpl.calls == 1  # 没有底库：退回模板
    ring = _ring_track()
    MapIcons(IconsConfig(enabled=True, classifier="template", save=False), tpl, gal, None).observe(_frame(), ring, "map", 0.0)
    assert ring.data["icon"] == "hand" and gal.calls == 1  # template 模式不用底库


def test_map_icons_no_classifier_is_unknown():
    ring = _ring_track()
    MapIcons(IconsConfig(enabled=True, save=False), None, None, None).observe(_frame(), ring, "map", 0.0)
    assert ring.data["icon"] == UNKNOWN


def test_map_icons_classifier_error_is_unknown():
    class Boom:
        min_score = 0.75

        def classify(self, region, scales=None):
            raise RuntimeError("坏了")

    ring = _ring_track()
    MapIcons(IconsConfig(enabled=True, save=False), Boom(), None, None).observe(_frame(), ring, "map", 0.0)
    assert ring.data["icon"] == UNKNOWN


def test_map_icons_unknown_reported_once_after_min_hits():
    m = MapIcons(IconsConfig(enabled=True, save=False), FakeTemplate((None, 0.2)), None, None)
    ring = _ring_track()
    got = []
    for i in range(6):
        ring.hits = i + 1
        got.append(m.observe(_frame(), ring, "map", float(i)))
    assert got == [False, False, True, False, False, False]


def test_map_icons_known_never_reported():
    m = MapIcons(IconsConfig(enabled=True, save=False), FakeTemplate(("bench", 0.9)), None, None)
    ring = _ring_track()
    for i in range(5):
        ring.hits = i + 1
        assert m.observe(_frame(), ring, "map", float(i)) is False


def test_map_icons_saves_unsure_crops_throttled(tmp_path):
    # 模板门槛 0.75：0.8 < 0.85 算拿不准、存；0.9 不存
    tpl = FakeTemplate(("bench", 0.8), ("bench", 0.8), ("bench", 0.8), ("bench", 0.9), (None, 0.3))
    m = MapIcons(IconsConfig(enabled=True), tpl, None, tmp_path)
    ring = _ring_track(tid=3)
    for t in (0.0, 1.0, 2.5, 5.0, 7.0):  # 0.0 存；1.0 离上次不到 2 秒不存；2.5 存；5.0 分数够高不存；7.0 认不出、存
        m.observe(_frame(), ring, "map", t)
    lines = [json.loads(s) for s in (tmp_path / "icons.jsonl").read_text("utf-8").splitlines()]
    assert [ln["t"] for ln in lines] == [0.0, 2.5, 7.0]
    assert [ln["kind"] for ln in lines] == ["bench", "bench", UNKNOWN]
    first = lines[0]
    assert set(first) == {"t", "track", "kind", "score", "classifier", "owner", "box", "file"}
    assert first["track"] == 3 and first["classifier"] == "template" and first["owner"] == "map"
    assert first["box"] == [950, 300, 100, 100] and first["score"] == 0.8
    for ln in lines:
        assert (tmp_path / ln["file"]).is_file()
        assert ln["file"].startswith(ln["kind"] + "/") and ln["file"].endswith("-t3.jpg")


def test_map_icons_save_max_and_save_off(tmp_path):
    m = MapIcons(IconsConfig(enabled=True, save_max=2), FakeTemplate((None, 0.1)), None, tmp_path / "a")
    for tid in range(4):  # 不同轨迹不受 2 秒节流
        m.observe(_frame(), _ring_track(tid=tid), "map", 0.0)
    assert len((tmp_path / "a" / "icons.jsonl").read_text("utf-8").splitlines()) == 2
    m = MapIcons(IconsConfig(enabled=True, save=False), FakeTemplate((None, 0.1)), None, tmp_path / "b")
    m.observe(_frame(), _ring_track(), "map", 0.0)
    assert not (tmp_path / "b").exists()
    m = MapIcons(IconsConfig(enabled=True), FakeTemplate((None, 0.1)), None, None)
    m.observe(_frame(), _ring_track(), "map", 0.0)  # 没有存盘目录：不存、不出错


def test_map_icons_save_uses_dino_threshold(tmp_path):
    m = MapIcons(IconsConfig(enabled=True, classifier="dino", dino_match=0.75), None, FakeGallery("bench", 0.84), tmp_path)
    m.observe(_frame(), _ring_track(), "map", 0.0)  # 0.84 < 0.85：存
    m2 = MapIcons(IconsConfig(enabled=True, classifier="dino", dino_match=0.75), None, FakeGallery("bench", 0.86),
                  tmp_path / "x")
    m2.observe(_frame(), _ring_track(), "map", 0.0)
    lines = (tmp_path / "icons.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["classifier"] == "dino"
    assert not (tmp_path / "x").exists()


# ---- Task 7：整图火焰预标 ----
from skydango.vision.candle import load_flame
from skydango.vision.icons_map import flame_rings
from skydango.vision.track import iou

_FLAME = load_flame()


def _flame_frame(centers):
    f = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)
    h, w = _FLAME.shape
    for cx, cy in centers:
        region = f[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w]
        region[_FLAME > 0] = (235, 245, 245)
    return f


def test_flame_rings_square_around_flame():
    out = flame_rings(_flame_frame([(1000, 500)]), _FLAME, 0.85, 2.0, [])
    assert len(out) == 1
    r = out[0]
    cx, cy = r.x + r.w / 2, r.y + r.h / 2
    assert abs(cx - 1000) <= 6 and abs(cy - 500) <= 6 and r.w == r.h
    assert abs(r.w - 2 * 2.0 * _FLAME.shape[0] / 2) <= 16  # 半径 = 火焰半高 × ring_k


def test_flame_rings_skip_and_existing():
    frame = _flame_frame([(300, 500), (1000, 500), (1500, 700)])
    skip = [Rect(0, 0, 640, 1080)]  # 面板区域里不找
    first = flame_rings(frame, _FLAME, 0.85, 2.0, skip)
    assert sorted(round(r.x + r.w / 2, -2) for r in first) == [1000, 1500]
    existing = [next(r for r in first if r.x > 1200)]
    again = flame_rings(frame, _FLAME, 0.85, 2.0, skip, existing=existing)
    assert len(again) == 1 and iou(again[0], existing[0]) < 0.5


def test_flame_rings_min_score():
    assert flame_rings(_flame_frame([]), _FLAME, 0.85, 2.0, []) == []


def test_cli_flame_prelabel_adds_rings_and_skips_without_template():
    from skydango import cli
    from skydango.config import Config

    frame, cfg = _flame_frame([(1000, 500)]), Config()
    got = cli._flame_prelabel(cfg, frame, [("player", Rect(900, 520, 200, 400))], _FLAME)
    assert [c for c, _ in got] == ["social_ring"]
    assert cli._flame_prelabel(cfg, frame, [("social_ring", got[0][1])], _FLAME) == []  # 已有圆圈：不重复
    assert cli._flame_prelabel(cfg, frame, [], None) == []  # 读不到模板：这一步跳过
