import cv2
import numpy as np
import pytest

from skydango.config import EnvConfig, PerceptionConfig
from skydango.game.social import IDLE
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection, _parse_names, decode, letterbox
from skydango.vision.ocr import OcrLine, RapidOcrEngine
from skydango.vision.perception import STRANGER, PerceptionWatcher, detector_conf
from skydango.vision.track import Tracker, iou
from skydango.vision.weaklabel import data_yaml, hard_images, merge_labels, ring_labels, split_of, weak_labels, yolo_line

FRIENDS = ["懒洋洋大王", "番茄炒蛋盖饭"]


# ---- 检测器：预处理和解码（不需要真模型） ----
def test_letterbox_keeps_aspect_and_pads_evenly():
    img = np.zeros((1080, 1920, 3), np.uint8)
    out, scale, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == pytest.approx(1 / 3)
    assert (left, top) == (0, 140)
    assert out[0, 0, 0] == 114 and out[320, 320, 0] == 0


def test_decode_yolo11_output_maps_back_and_runs_nms():
    names = ["player", "name_tag"]
    scale, pad = 1 / 3, (0, 140)
    # 两个几乎重叠的 player（NMS 留分高的）+ 一个 name_tag；letterbox 坐标系里的 cx, cy, w, h
    cols = [
        [100, 240, 30, 60, 0.9, 0.1],
        [101, 241, 30, 60, 0.6, 0.1],
        [300, 200, 40, 12, 0.05, 0.8],
        [500, 300, 40, 40, 0.1, 0.2],  # 分数太低
    ]
    out = np.array(cols, np.float32).T[None]  # (1, 4 + 2, N)
    dets = decode(out, names, scale, pad, (1080, 1920), conf=0.35, iou=0.5)
    by = {d.cls: d for d in dets}
    assert len(dets) == 2
    assert by["player"].score == pytest.approx(0.9)
    assert by["player"].box == Rect(255, 210, 90, 180)  # (100-15)*3, (240-30-140)*3
    assert by["name_tag"].box == Rect(840, 162, 120, 36)


def test_decode_end_to_end_output():
    names = ["player", "name_tag", "social_ring"]
    rows = np.array([[90, 200, 120, 260, 0.8, 0], [10, 10, 20, 20, 0.1, 1], [0, 0, 5, 5, 0.9, 7]], np.float32)[None]
    dets = decode(rows, names, 1 / 3, (0, 140), (1080, 1920), conf=0.35, iou=0.5)
    assert dets == [Detection("player", Rect(270, 180, 90, 180), pytest.approx(0.8))]


def test_parse_names_from_ultralytics_metadata():
    assert _parse_names("{0: 'player', 1: 'name_tag'}") == ["player", "name_tag"]
    assert _parse_names("garbage{") is None and _parse_names(None) is None


# ---- 追踪 ----
def test_tracker_keeps_id_while_overlapping_and_drops_after_buffer():
    t = Tracker(buffer=1.0, min_iou=0.3)
    (a,) = t.update([Detection("player", Rect(100, 100, 50, 100), 0.9)], 0.0)
    (b,) = t.update([Detection("player", Rect(110, 105, 50, 100), 0.9)], 0.1)
    assert a.id == b.id and b.hits == 2
    (c,) = t.update([Detection("name_tag", Rect(110, 105, 50, 100), 0.9)], 0.2)  # 类别不同不接
    assert c.id != a.id
    (d,) = t.update([Detection("player", Rect(112, 105, 50, 100), 0.9)], 2.0)  # 断了太久：新轨迹
    assert d.id not in (a.id, c.id)


def test_iou():
    assert iou(Rect(0, 0, 10, 10), Rect(5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert iou(Rect(0, 0, 10, 10), Rect(20, 20, 5, 5)) == 0.0


# ---- 感知层 ----
class FakeDetector:
    def __init__(self):
        self.frames = []  # 每次 detect 依次返回一组；用完了一直返回最后一组
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return self.frames.pop(0) if len(self.frames) > 1 else (self.frames[0] if self.frames else [])


class FakeOcr:
    """read_line 按名字标签框的宽度认字（测试里每个标签宽度不同）。"""

    def __init__(self, by_width):
        self.by_width = by_width
        self.calls = 0

    def read_line(self, img):
        self.calls += 1
        text = self.by_width.get(img.shape[1] - 8)  # 裁剪时四周各扩了 4 px
        return OcrLine(text, 0.99, Rect(0, 0, img.shape[1], img.shape[0])) if text else None


class FakeIcons:
    """圆圈里是什么：按圆圈中心的横坐标查表。"""

    def __init__(self, kinds):
        self.kinds = kinds

    def classify(self, region):
        return self.kinds.get("next"), 0.9


def frame():
    return np.zeros((1080, 1920, 3), np.uint8)


def player(x, y=400, w=90, h=220):
    return Detection("player", Rect(x, y, w, h), 0.9)


def tag(x, w, y=330, h=44):
    return Detection("name_tag", Rect(x, y, w, h), 0.9)


def ring(cx, tag_y=330, tag_h=44):
    cy = tag_y + round(2.23 * tag_h)
    return Detection("social_ring", Rect(cx - 50, cy - 50, 100, 100), 0.9)


def ringed(cx=1400, cy=428):
    """画面上 ring(cx) 那个位置画一圈白圈：头顶没名字的圆圈要有白圈才算陌生人的请求（没有 = 深色火焰圆盘）。"""
    f = frame()
    cv2.circle(f, (cx, cy), 48, (240, 245, 245), 3)
    return f


def watcher(detector, ocr=None, icons=None, clock=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    extra = {"clock": clock} if clock is not None else {}
    return PerceptionWatcher(
        detector, ocr or FakeOcr({}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], icons=icons, background=False, **extra,
    )


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_friend_is_recognized_by_reading_the_name_tag_once():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110)]]
    ocr = FakeOcr({110: "懒洋洋大王"})
    w = watcher(det, ocr, ocr_retry=1.0, ocr_votes=3)
    for i in range(5):
        w.process(frame(), i * 0.1, panel_visible=False)
    assert w.nearby(0.5) == ["懒洋洋大王"]
    assert ocr.calls == 1  # 同一条轨迹 ocr_retry 内不重复读
    assert w.labels["懒洋洋大王"][:4] == (990, 330, 110, 44)
    assert "懒洋洋大王" in w.describe(0.5) and "陌生人" not in w.describe(0.5)
    w.process(frame(), 1.2, panel_visible=False)  # 隔了 ocr_retry：再读一次投票
    assert ocr.calls == 2


def test_identity_follows_the_name_not_the_track():
    det = FakeDetector()
    ocr = FakeOcr({110: "懒洋洋大王"})
    w = watcher(det, ocr, track_buffer=0.5, keep=5.0)
    det.frames = [[tag(990, 110)]]
    w.process(frame(), 0.0, panel_visible=False)
    det.frames = [[]]  # 转视角：看不到了
    w.process(frame(), 1.0, panel_visible=False)
    det.frames = [[tag(300 + 990, 110)]]  # 换了位置，新轨迹
    w.process(frame(), 2.0, panel_visible=False)
    assert ocr.calls == 2 and w.nearby(2.0) == ["懒洋洋大王"]
    assert w.nearby(7.1) == []  # keep 秒没看到算走开


def test_player_without_name_tag_becomes_stranger_after_a_while():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), player(1500)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), stranger_after=1.0, keep=5.0)
    w.process(frame(), 0.0, panel_visible=False)
    assert w.strangers(0.0) == 0  # 刚出现，还没到 stranger_after
    w.process(frame(), 1.0, panel_visible=False)
    assert w.strangers(1.0) == 1
    assert "1 个陌生人" in w.describe(1.0)
    det.frames = [[player(1000), tag(990, 110)]]
    w.process(frame(), 2.0, panel_visible=False)
    assert w.strangers(5.9) == 1  # 最近 keep 秒里看到过
    assert w.strangers(6.1) == 0


def test_far_players_and_self_are_not_strangers():
    det = FakeDetector()
    det.frames = [[player(1000, h=40), player(900), Detection("self", Rect(900, 400, 90, 220), 0.9)]]
    w = watcher(det)
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(2.0) == 0


def test_unlit_silhouette_is_a_stranger_right_away_even_far_away():
    det = FakeDetector()
    det.frames = [[Detection("player_unlit", Rect(1400, 500, 20, 50), 0.9), player(1000), tag(990, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), stranger_after=1.0)
    w.process(frame(), 0.0, panel_visible=False)
    assert w.strangers(0.0) == 1 and w.unlit(0.0) == 1  # 不用等 stranger_after，也不管离得多远
    assert "其中 1 个还没点火" in w.describe(0.0)


def test_lit_stranger_without_tag_is_not_counted_as_unlit():
    det = FakeDetector()
    det.frames = [[player(1500)]]
    w = watcher(det)
    for t in (0.0, 1.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(1.0) == 1 and w.unlit(1.0) == 0
    assert "点火" not in w.describe(1.0)


def test_self_roi_excludes_the_player_in_the_middle():
    det = FakeDetector()
    det.frames = [[player(915, y=420)]]
    w = watcher(det, self_roi=[0.45, 0.35, 0.55, 0.75])
    for t in (0.0, 1.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(1.0) == 0


def test_tagged_player_stays_known_when_tag_is_briefly_hidden():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    det.frames = [[player(1000), tag(990, 110)]]
    w.process(frame(), 0.0, panel_visible=False)
    det.frames = [[player(1002)]]  # 标签被挡住
    for t in (0.5, 1.0, 1.5, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(2.0) == 0


def test_ring_under_friend_tag_becomes_a_request():
    det = FakeDetector()
    icons = FakeIcons({"next": "hug"})
    det.frames = [[tag(990, 110), ring(1045)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), icons=icons)
    w.process(frame(), 0.0, panel_visible=False)
    req = w.requests["懒洋洋大王"]
    assert req.kind == "hug" and req.pos == (1045, 428)
    assert w.circles["懒洋洋大王"][0] == "hug"
    icons.kinds["next"] = IDLE  # 对方取消了：变回 ✦
    w.process(frame(), 0.1, panel_visible=False)
    assert "懒洋洋大王" not in w.requests and w.circles["懒洋洋大王"] == (IDLE, 0.1)
    det.frames = [[tag(990, 110)]]  # 牵上手后圆圈消失
    w.process(frame(), 0.2, panel_visible=False)
    assert w.circles["懒洋洋大王"] == (None, 0.2)


def test_ring_without_tag_is_a_stranger_request():
    det = FakeDetector()
    det.frames = [[ring(1400)]]
    w = watcher(det, icons=FakeIcons({"next": "candle"}))
    w.process(ringed(), 0.0, panel_visible=False)
    assert w.requests[STRANGER].kind == "candle"
    det.frames = [[]]
    w.process(ringed(), 0.1, panel_visible=False)
    assert STRANGER not in w.requests


def test_stranger_request_survives_a_frame_where_the_icon_is_unclear():
    """实测（2026-09-30 22:56）：火焰图标会晃，这一帧认出、下一帧认不出，请求一闪就没，身体来不及去点。
    圆圈还在原地、只是这一帧没认出图标：请求留 REQUEST_HOLD 秒；圆圈没了 / 认成别的就马上撤。"""
    det = FakeDetector()
    det.frames = [[ring(1400)]]
    icons = FakeIcons({"next": "candle"})
    w = watcher(det, icons=icons)
    w.process(ringed(), 0.0, panel_visible=False)
    icons.kinds["next"] = None  # 这一帧没认出
    w.process(ringed(), 0.3, panel_visible=False)
    assert w.requests[STRANGER].kind == "candle"
    w.process(ringed(), 2.0, panel_visible=False)  # 一直认不出：过了 1.5 秒就撤
    assert STRANGER not in w.requests
    icons.kinds["next"] = "candle"
    w.process(ringed(), 2.1, panel_visible=False)
    icons.kinds["next"] = "stranger"  # 放下蜡烛了：图标换成了别的
    w.process(ringed(), 2.2, panel_visible=False)
    assert STRANGER not in w.requests


def test_panel_area_and_bottom_bar_are_ignored():
    det = FakeDetector()
    det.frames = [[tag(100, 110), tag(990, 120, y=1000)]]  # 面板里的"- 名字"、底部输入栏
    ocr = FakeOcr({110: "懒洋洋大王", 120: "番茄炒蛋盖饭"})
    w = watcher(det, ocr)
    w.process(frame(), 0.0, panel_visible=True)
    assert w.nearby(0.0) == [] and ocr.calls == 0
    w.process(frame(), 0.1, panel_visible=False)  # 面板关了：左边的标签算数，底部的还是不算
    assert w.nearby(0.1) == ["懒洋洋大王"]


def test_observe_throttles_to_fps():
    det = FakeDetector()
    w = watcher(det, fps=10.0)
    for t in (0.0, 0.05, 0.1, 0.12, 0.2):
        w.observe(frame(), t, panel_visible=False)
    assert len(w.timings) == 3  # 0.0、0.1、0.2


def test_own_capture_runs_in_a_thread():
    import threading

    det = FakeDetector()
    got = threading.Event()

    def capture():
        got.set()
        return frame()

    w = PerceptionWatcher(det, FakeOcr({}), PerceptionConfig(capture="own", fps=50.0), EnvConfig(), lambda: FRIENDS,
                          log_roi=[0.0, 0.0, 0.335, 0.855], capture=capture)
    w.observe(frame(), 0.0, panel_visible=True)
    try:
        assert got.wait(2.0)
    finally:
        w.stop()


# ---- OCR 只识别 ----
class _Legacy:
    def __call__(self, img, **kw):
        assert kw == {"use_det": False, "use_cls": False, "use_rec": True}
        return [["懒洋洋大王 ", 0.98]], [0.01]


class _New:
    def __call__(self, img, **kw):
        return type("Out", (), {"txts": ("番茄炒蛋盖饭",), "scores": (0.97,)})()


@pytest.mark.parametrize("engine,legacy,text", [(_Legacy(), True, "懒洋洋大王"), (_New(), False, "番茄炒蛋盖饭")])
def test_read_line_supports_both_rapidocr_apis(engine, legacy, text):
    ocr = RapidOcrEngine.__new__(RapidOcrEngine)
    ocr._engine, ocr._legacy = engine, legacy
    line = ocr.read_line(np.zeros((40, 120, 3), np.uint8))
    assert line.text == text and line.box == Rect(0, 0, 120, 40)
    assert ocr.read_line(np.zeros((0, 0, 3), np.uint8)) is None


# ---- 弱标注 ----
def test_weak_labels_from_friend_names_and_rings():
    lines = [
        OcrLine("懒洋详大王", 0.99, Rect(990, 330, 110, 44)),  # 错一个字也认
        OcrLine("2级", 0.99, Rect(1300, 300, 40, 30)),  # 不是好友名字
        OcrLine("番茄炒蛋盖饭", 0.99, Rect(100, 200, 120, 40)),  # 在面板里
        OcrLine("番茄炒蛋盖饭", 0.5, Rect(1500, 200, 120, 40)),  # 分数低
    ]
    boxes = weak_labels(frame(), lines, FRIENDS, FakeIcons({"next": IDLE}), skip=[Rect(0, 0, 640, 920)])
    assert boxes == [("name_tag", Rect(986, 326, 118, 52)), ("social_ring", Rect(995, 378, 100, 100))]
    assert len(weak_labels(frame(), lines, FRIENDS, None, skip=[Rect(0, 0, 640, 920)], all_text=True)) == 2


def test_yolo_label_format_and_split():
    assert yolo_line(1, Rect(960, 540, 192, 108), 1920, 1080) == "1 0.550000 0.550000 0.100000 0.100000"
    assert split_of("a_0001", 0.15) == split_of("a_0001", 0.15)
    share = sum(split_of(f"f{i}", 0.2) == "val" for i in range(2000)) / 2000
    assert 0.15 < share < 0.25


def test_data_yaml(tmp_path):
    text = data_yaml(tmp_path, ["player", "name_tag"])
    assert "train: images/train" in text and "  1: name_tag" in text


# ---- 追踪器：补时间、跨类别关联（一期 §4、§5） ----
def test_tracker_shift_keeps_tracks_alive():
    t = Tracker(buffer=1.0)
    first = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    t.shift(20.0)
    again = t.update([Detection("player", Rect(2, 0, 50, 100), 0.9)], 20.5)[0]
    assert again.id == first.id and again.first == 20.0


def test_tracker_cross_class_association_counts_flips():
    t = Tracker(cross=frozenset({"player", "player_unlit"}), cross_iou=0.5)
    a = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    b = t.update([Detection("player_unlit", Rect(1, 0, 50, 100), 0.9)], 0.1)[0]
    c = t.update([Detection("player", Rect(1, 0, 50, 100), 0.9)], 0.2)[0]
    assert a.id == b.id == c.id and c.cls == "player" and c.flips == 2


def test_tracker_without_cross_keeps_classes_apart():
    t = Tracker()
    a = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    b = t.update([Detection("player_unlit", Rect(0, 0, 50, 100), 0.9)], 0.1)[0]
    assert a.id != b.id


# ---- 画面被挡时暂停计时（一期 §4） ----
def test_hold_stacks_and_nests():
    w = watcher(FakeDetector())
    w.release("x")  # 没 hold 过：忽略
    assert not w.paused
    w.hold("a")
    w.hold("b")
    w.hold("a")  # 同一原因嵌套
    w.release("a")
    w.release("b")
    assert w.paused
    w.release("a")
    assert not w.paused


def test_held_releases_on_error():
    w = watcher(FakeDetector())
    with pytest.raises(RuntimeError):
        with w.held("camera"):
            assert w.paused
            raise RuntimeError
    assert not w.paused


def test_hold_freezes_nearby_and_shifts_last_seen():
    clock = Clock()
    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), clock=clock, keep=5.0)
    w.process(frame(), 0.0, panel_visible=False)
    clock.t = 1.0
    w.hold("blackout")
    clock.t = 21.0
    assert w.nearby(21.0) == ["懒洋洋大王"]  # 暂停期间照常返回暂停前的结果
    w.release("blackout")
    assert w.last_seen["懒洋洋大王"] <= 21.0
    assert w.nearby(25.0) == ["懒洋洋大王"]
    assert w.nearby(27.0) == []


def test_hold_freezes_strangers():
    clock = Clock()
    det = FakeDetector()
    det.frames = [[Detection("player_unlit", Rect(1000, 400, 90, 220), 0.9)]]
    w = watcher(det, clock=clock, keep=5.0)
    w.process(frame(), 0.0, panel_visible=False)
    clock.t = 1.0
    w.hold("camera")
    assert w.strangers(30.0) == 1 and w.unlit(30.0) == 1
    clock.t = 30.0
    w.release("camera")
    assert w.strangers(33.0) == 1


def test_hold_skips_detection():
    clock = Clock()
    det = FakeDetector()
    w = watcher(det, clock=clock)
    w.hold("camera")
    w.observe(frame(), 0.0, panel_visible=False)
    assert det.calls == 0
    w.release("camera")
    w.observe(frame(), 1.0, panel_visible=False)
    assert det.calls == 1


def test_hold_keeps_tracks_no_reocr():
    clock = Clock()
    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    ocr = FakeOcr({110: "懒洋洋大王"})
    w = watcher(det, ocr, clock=clock, ocr_votes=1, track_buffer=1.0)
    w.process(frame(), 0.0, panel_visible=False)
    clock.t = 0.5
    w.hold("friend_tree")
    clock.t = 20.5
    w.release("friend_tree")
    w.process(frame(), 21.0, panel_visible=False)
    assert ocr.calls == 1 and w.nearby(21.0) == ["懒洋洋大王"]


def test_hold_max_auto_releases(caplog):
    clock = Clock()
    w = watcher(FakeDetector(), clock=clock, hold_max=60.0)
    w.hold("camera")
    clock.t = 61.0
    with caplog.at_level("WARNING"):
        w.observe(frame(), 61.0, panel_visible=False)
    assert not w.paused
    assert any("暂停" in r.message for r in caplog.records)


# ---- 集体消失：玩家自己开了全屏界面（一期 §4） ----
def bright():
    return np.full((1080, 1920, 3), 200, np.uint8)


def test_two_people_vanish_with_big_change_holds():
    det = FakeDetector()
    det.frames = [[player(1000), player(1400)], []]
    w = watcher(det, clock=Clock())
    w.process(bright(), 0.0, panel_visible=False)
    w.process(frame(), 0.1, panel_visible=False)
    assert w.paused


def test_one_person_vanishing_does_not_hold():
    det = FakeDetector()
    det.frames = [[player(1000)], []]
    w = watcher(det, clock=Clock())
    w.process(bright(), 0.0, panel_visible=False)
    w.process(frame(), 0.1, panel_visible=False)
    assert not w.paused


def test_small_change_does_not_hold():
    det = FakeDetector()
    det.frames = [[player(1000), player(1400)], []]
    w = watcher(det, clock=Clock())
    w.process(bright(), 0.0, panel_visible=False)
    w.process(bright(), 0.1, panel_visible=False)
    assert not w.paused


def test_occlusion_released_when_someone_is_back():
    clock = Clock()
    det = FakeDetector()
    det.frames = [[tag(990, 110), tag(1390, 120)], [], [], [tag(990, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王", 120: "番茄炒蛋盖饭"}), clock=clock, keep=5.0)
    w.process(bright(), 0.0, panel_visible=False)
    clock.t = 0.1
    w.process(frame(), 0.1, panel_visible=False)
    assert w.paused
    w.observe(frame(), 0.2, panel_visible=False)  # occlusion 暂停期间检测照跑
    assert det.calls == 3
    clock.t = 20.0
    w.process(bright(), 20.0, panel_visible=False)
    assert not w.paused
    assert w.nearby(20.0) == FRIENDS  # 地图开了 20 s，谁都没走


def test_occlusion_gives_up_after_limit():
    clock = Clock()
    det = FakeDetector()
    det.frames = [[player(1000), player(1400)], []]
    w = watcher(det, clock=clock, occlusion_hold=30.0)
    w.process(bright(), 0.0, panel_visible=False)
    clock.t = 0.1
    w.process(frame(), 0.1, panel_visible=False)
    assert w.paused
    clock.t = 31.2
    w.process(frame(), 31.2, panel_visible=False)
    assert not w.paused


# ---- 出框阈值和判定阈值分开（一期 §5.1 low_conf） ----
def test_low_confidence_boxes_are_not_tracked_but_kept():
    det = FakeDetector()
    det.frames = [[player(1000), Detection("player", Rect(1500, 400, 90, 220), 0.3)]]
    w = watcher(det, conf=0.35)
    w.process(frame(), 0.0, panel_visible=False)
    assert [t.cls for t in w.last_tracks] == ["player"]
    assert [d.box.x for d in w.last_low] == [1500]


def test_detector_conf():
    assert detector_conf(PerceptionConfig(conf=0.35, low_conf=0.25)) == 0.25
    assert detector_conf(PerceptionConfig(conf=0.35, hardcases=False, track_low=False)) == 0.35


# ---- label --model / --from-runs（一期 §5.2） ----
def test_merge_labels_prefers_weak_boxes():
    weak = [("name_tag", Rect(990, 330, 110, 44))]
    predicted = [
        Detection("name_tag", Rect(992, 331, 108, 44), 0.8),  # 和弱标注重叠：丢掉，OCR 的名字框更准
        Detection("player", Rect(1000, 400, 90, 220), 0.6),
        Detection("name_tag", Rect(300, 330, 110, 44), 0.5),  # 没重叠：保留
    ]
    out = merge_labels(weak, predicted)
    assert out == [("name_tag", Rect(990, 330, 110, 44)), ("player", Rect(1000, 400, 90, 220)),
                   ("name_tag", Rect(300, 330, 110, 44))]


def test_hard_images_collects_from_runs(tmp_path):
    for run, name in (("20260928-100000-dry", "101010_low_conf.jpg"), ("20260928-110000-live", "111111_flicker.jpg")):
        (tmp_path / run / "hard").mkdir(parents=True)
        (tmp_path / run / "hard" / name).write_bytes(b"x")
    (tmp_path / "20260928-120000-dry").mkdir()
    out = hard_images(tmp_path)
    assert [stem for _, stem in out] == ["20260928-100000-dry_101010_low_conf", "20260928-110000-live_111111_flicker"]
    assert out[0][0] == tmp_path / "20260928-100000-dry" / "hard" / "101010_low_conf.jpg"


def test_no_occlusion_right_after_a_camera_hold():
    # 转完镜头人都不在画面里是正常的：不能拿转之前那一帧比，误判成"开了全屏界面"
    clock = Clock()
    det = FakeDetector()
    det.frames = [[player(1000), player(1400)], []]
    w = watcher(det, clock=clock)
    w.process(bright(), 0.0, panel_visible=False)
    with w.held("camera"):
        clock.t = 2.0
    w.process(frame(), 2.1, panel_visible=False)
    assert not w.paused


# ---- 二期：环绕扫描 ----
from skydango.config import SpinConfig  # noqa: E402
from skydango.vision.sweep import UNKNOWN_WHO, UNLIT_WHO  # noqa: E402


def spin_frames(n=20):  # t = i * 0.1；seconds_per_turn = 2.0、hfov = 90 → 每帧转 18°
    return [(i * 0.1, frame()) for i in range(n)]


def me():
    return player(900, 500)  # 团子：每帧都在中间不动


def test_sweep_reports_friend_direction_and_strangers():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    det.frames[5] = [me(), player(1155, 400), tag(1145, 110)]  # t=0.5 → 90° + 画面偏右 11° → 右
    det.frames[10] = [me(), Detection("player_unlit", Rect(1155, 400, 90, 220), 0.9)]  # t=1.0 → 191° → 后
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    r = w.sweep(spin_frames(), SpinConfig())
    assert {(e.who, e.direction) for e in r.entries} == {("懒洋洋大王", "右"), (UNLIT_WHO, "后")}
    assert r.self_box == Rect(900, 500, 90, 220) and w.self_box == r.self_box
    assert r.frames == 20 and r.seconds == pytest.approx(1.9)
    assert "右边：懒洋洋大王" in r.text()


def test_sweep_does_not_touch_runtime_state():
    det = FakeDetector()
    det.frames = [[me(), player(1155, 400), tag(1145, 110)] for _ in range(20)]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.sweep(spin_frames(), SpinConfig())
    assert w.last_seen == {} and w.tracker.tracks == {} and w.requests == {} and w.labels == {}


def test_holding_partner_is_reported_beside():
    det = FakeDetector()
    det.frames = [[me(), player(980, 500), tag(965, 120, y=430)] for _ in range(20)]
    w = watcher(det, FakeOcr({120: "番茄炒蛋盖饭"}))
    r = w.sweep(spin_frames(), SpinConfig())
    assert [(e.who, e.direction) for e in r.entries] == [("番茄炒蛋盖饭", "身边")]
    assert r.self_box is None and w.self_box is None


def test_unknown_tag_is_not_a_stranger():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    det.frames[3] = [me(), player(1155, 400), tag(1145, 130)]
    w = watcher(det, FakeOcr({130: "路人甲"}))
    r = w.sweep(spin_frames(), SpinConfig())
    assert [e.who for e in r.entries] == [UNKNOWN_WHO]


def test_small_untagged_player_is_skipped():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    det.frames[3] = [me(), player(1500, h=40)]
    r = watcher(det).sweep(spin_frames(), SpinConfig())
    assert r.entries == []


def test_self_box_from_sweep_excludes_self_at_runtime():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    w = watcher(det)
    w.sweep(spin_frames(), SpinConfig())
    det.frames = [[me()]]
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(2.0) == 0


# ---- 二期：认说话人（typing 气泡） ----
def bubble(x, y=180, w=80, h=60):
    return Detection("typing", Rect(x, y, w, h), 0.9)


def unlit(x, y=400, w=90, h=220):
    return Detection("player_unlit", Rect(x, y, w, h), 0.9)


def test_bubble_over_unlit_stranger_gives_hint():
    det = FakeDetector()
    det.frames = [[unlit(1500), bubble(1505)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.speaker_hint(12.0) == "（说话的可能是右边近处那个没点火的陌生人）"  # 220 ÷ (0.2 × 1080) ≈ 1.02 → 近
    assert w.speaker_hint(19.0) is None  # 过了 typing_window（8 秒）


def test_far_stranger_on_the_left():
    det = FakeDetector()
    det.frames = [[player(200, y=500, w=40, h=80), bubble(190, y=420, w=60, h=40)]]  # 80 ÷ 216 ≈ 0.37 → 远
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.speaker_hint(10.0) == "（说话的可能是左边远处那个陌生人）"


def test_bubble_over_friend_gives_no_hint():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), bubble(1005, y=260)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.process(frame(), 10.0, panel_visible=False)
    assert w.speaker_hint(10.0) is None


def test_bubble_over_self_is_ignored():
    det = FakeDetector()
    det.frames = [[Detection("self", Rect(900, 400, 90, 220), 0.9), player(905), bubble(905)]]  # 团子框和人物框叠在一起
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.speaker_hint(10.0) is None


def test_two_strangers_typing_gives_no_hint():
    det = FakeDetector()
    det.frames = [[unlit(1500), bubble(1505), unlit(300), bubble(305)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.speaker_hint(10.0) is None


def test_overlay_marks_typing():
    det = FakeDetector()
    det.frames = [[unlit(1500), bubble(1505)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert {"kind": "typing", "label": "正在输入"}.items() <= next(b for b in w.overlay(10.0) if b["kind"] == "typing").items()


def test_default_classes_end_with_typing():
    assert PerceptionConfig().classes[:6] == ["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]  # 物品类追加在后面
    assert PerceptionConfig().typing_window == 8.0


# ---- 二期：距离和走向 ----
from skydango.vision.perception import approaching  # noqa: E402


def test_approaching_rules():
    grow = [(i * 0.1, 100 + i * 5, 1400 - i * 20) for i in range(15)]
    assert approaching(grow, 1920, 0.25)
    jitter = [(i * 0.1, 100 * (1.1 if i % 2 else 0.9), 960) for i in range(15)]
    assert not approaching(jitter, 1920, 0.25)
    outward = [(i * 0.1, 100 + i * 5, 1300 + i * 30) for i in range(15)]
    assert not approaching(outward, 1920, 0.25)
    assert not approaching(grow[:2], 1920, 0.25)


def walk_up(w, det, t0, tagged=True, cls="player"):
    for i in range(15):
        h = 150 + 8 * i
        body = Detection(cls, Rect(1000, 620 - h, 90, h), 0.9)
        det.frames = [[body, tag(990, 110, y=620 - h - 70)] if tagged else [body]]
        w.process(frame(), t0 + i * 0.1, panel_visible=False)


def test_friend_walking_up_is_reported_once_per_cooldown():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    walk_up(w, det, 0.0)
    assert w.pop_approaches() == ["懒洋洋大王"]
    assert w.pop_approaches() == []  # 取走就清空
    walk_up(w, det, 2.0)  # 60 秒内又走过来一次
    assert w.pop_approaches() == []


def test_stranger_approach_follows_switch():
    det = FakeDetector()
    w = watcher(det)
    walk_up(w, det, 0.0, tagged=False)
    assert w.pop_approaches() == [STRANGER]
    det2 = FakeDetector()
    quiet = watcher(det2, approach_strangers=False)
    walk_up(quiet, det2, 0.0, tagged=False)
    assert quiet.pop_approaches() == []


def test_no_approach_across_a_pause():
    det = FakeDetector()
    clock = Clock()
    w = watcher(det, clock=clock)
    for i in range(5):
        det.frames = [[unlit(1000, y=400, h=100)]]
        w.process(frame(), i * 0.1, panel_visible=False)
    clock.t = 0.45
    w.hold("camera")
    clock.t = 0.6
    w.release("camera")
    for i in range(5):
        det.frames = [[unlit(1000, y=360, h=140)]]
        w.process(frame(), 0.6 + i * 0.1, panel_visible=False)
    assert w.pop_approaches() == []


def test_nearest_picks_the_tallest_known_person():
    det = FakeDetector()
    det.frames = [[player(1000, y=300, h=300), tag(990, 110, y=230), unlit(1500, h=150),
                   Detection("self", Rect(900, 420, 90, 220), 0.9)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert w.nearest(0.0) == ("懒洋洋大王", "近")  # 300 ÷ 220（团子框）≈ 1.36
    assert watcher(FakeDetector()).nearest(0.0) is None


# ---- 评审修正 ----
def test_sweep_finds_self_when_model_boxes_it_twice():
    det = FakeDetector()
    det.frames = [[me(), Detection("self", Rect(902, 502, 90, 218), 0.9)] for _ in range(20)]
    r = watcher(det).sweep(spin_frames(), SpinConfig())
    assert r.self_box is not None and r.entries == []


def test_sweep_merges_flickering_friend():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    det.frames[5] = [me(), player(1155, 400), tag(1145, 110)]
    det.frames[6] = [me(), player(1070, 400)]  # 标签这一帧没检测到
    det.frames[7] = [me(), player(990, 400), tag(980, 140)]  # 标签糊了读不出
    r = watcher(det, FakeOcr({110: "懒洋洋大王"})).sweep(spin_frames(), SpinConfig())
    assert [e.who for e in r.entries] == ["懒洋洋大王"]


def test_single_frame_sweep_excludes_known_self():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    w = watcher(det)
    w.sweep(spin_frames(), SpinConfig())  # 认出团子
    det.frames = [[player(905, 500)]]
    assert w.sweep([(0.0, frame())], SpinConfig()).entries == []


def test_single_frame_sweep_excludes_self_roi():
    det = FakeDetector()
    det.frames = [[player(905, 500)]]
    w = watcher(det, self_roi=[0.4, 0.4, 0.6, 0.8])
    assert w.sweep([(0.0, frame())], SpinConfig()).entries == []


def test_self_box_is_padded_and_self_roi_still_applies():
    det = FakeDetector()
    det.frames = [[me()] for _ in range(20)]
    w = watcher(det, self_roi=[0.0, 0.0, 0.1, 0.5])
    w.sweep(spin_frames(), SpinConfig())
    det.frames = [[player(960, 540), player(20, 100)]]  # 团子挪出了原框一点；self_roi 里另有一个
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert w.strangers(2.0) == 0


# ---- 三期 §5：没认出的名字 ----
class RecordingUnknown:
    def __init__(self):
        self.added = []

    def add(self, text, crop):
        self.added.append((text, crop.shape))


def test_clear_but_unknown_name_is_recorded_once_per_track():
    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    ocr = FakeOcr({110: "星星小铺"})
    unknown = RecordingUnknown()
    w = PerceptionWatcher(
        det, ocr, PerceptionConfig(ocr_retry=1.0, ocr_votes=3), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, unknown=unknown,
    )
    for i in range(8):  # 同一条轨迹（间隔 < track_buffer），隔 ocr_retry 重读
        w.process(frame(), i * 0.5, panel_visible=False)
    assert ocr.calls >= 3 and w.nearby(3.5) == []
    assert [a[0] for a in unknown.added] == ["星星小铺"]


class LowScoreOcr(FakeOcr):
    def read_line(self, img):
        line = super().read_line(img)
        return OcrLine(line.text, 0.8, line.box) if line else None


def test_low_confidence_unknown_text_is_not_recorded():
    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    unknown = RecordingUnknown()
    w = PerceptionWatcher(
        det, LowScoreOcr({110: "星星小铺"}), PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, unknown=unknown,
    )
    w.process(frame(), 0.0, panel_visible=False)
    assert unknown.added == []


def test_friend_names_are_not_recorded_as_unknown():
    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    unknown = RecordingUnknown()
    w = PerceptionWatcher(
        det, FakeOcr({110: "懒洋洋大王"}), PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, unknown=unknown,
    )
    w.process(frame(), 0.0, panel_visible=False)
    assert unknown.added == [] and w.nearby(0.0) == ["懒洋洋大王"]


# ---- 三期 §1：远处小目标二次检测 ----
from skydango.vision.perception import far_region  # noqa: E402


class CropDetector:
    """整张截图（1080×1920）返回 full；裁剪图返回 crop（裁剪图坐标），记下每次裁剪的尺寸。"""

    def __init__(self, full, crop=()):
        self.full = list(full)
        self.crop = list(crop)
        self.crops = []

    def detect(self, img):
        if img.shape[:2] == (1080, 1920):
            return list(self.full)
        self.crops.append(img.shape[:2])
        return list(self.crop)


def small(x, y=500, w=24, h=60, cls="player"):
    return Detection(cls, Rect(x, y, w, h), 0.9)


def far_watcher(det, ocr=None, **cfg):
    return PerceptionWatcher(
        det, ocr or FakeOcr({}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False,
    )


def test_far_region_is_above_the_head_and_clamped():
    assert far_region(Rect(1000, 500, 24, 60), 1920, 1080) == Rect(976, 380, 72, 150)
    assert far_region(Rect(0, 10, 20, 60), 1920, 1080) == Rect(0, 0, 40, 40)  # 左上越界夹回
    assert far_region(Rect(5, 0, 2, 4), 1920, 1080) is None  # 太小


def test_far_player_gets_a_second_look_above_its_head():
    det = CropDetector([small(1000)], [Detection("name_tag", Rect(10, 20, 50, 14), 0.9)])
    w = far_watcher(det, FakeOcr({50: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert det.crops == [(150, 72)] and w.far_runs == 1
    assert w.nearby(0.0) == ["懒洋洋大王"]
    assert w.labels["懒洋洋大王"][:4] == (986, 400, 50, 14)  # 映射回整图坐标
    assert any(t.cls == "player" and t.data.get("name") == "懒洋洋大王" for t in w.last_tracks)


def test_far_crop_is_rate_limited_per_track_and_per_frame():
    det = CropDetector([small(200 + 300 * i) for i in range(5)])
    w = far_watcher(det, far_crops=3)
    w.process(frame(), 0.0, panel_visible=False)
    assert w.far_runs == 3
    w.process(frame(), 0.5, panel_visible=False)
    assert w.far_runs == 5  # 剩下两个轮到了；前三个还没到 1 s
    w.process(frame(), 1.0, panel_visible=False)
    assert w.far_runs == 8


def test_far_crop_skips_tagged_big_and_unlit_players():
    det = CropDetector([
        small(1000), Detection("name_tag", Rect(990, 440, 50, 14), 0.9),  # 已经挂上了标签
        player(1400),  # 离得近（框大）
        small(300, cls="player_unlit"),  # 黑影不用认名字
    ])
    w = far_watcher(det, FakeOcr({50: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert w.far_runs == 0 and det.crops == []


def test_far_crop_drops_rings_already_seen_in_the_full_frame():
    seen = Detection("social_ring", Rect(986, 440, 30, 30), 0.9)  # 原图也框到了这个圆圈
    det = CropDetector([small(1000), seen], [
        Detection("social_ring", Rect(10, 60, 30, 30), 0.9),  # 裁剪里又找到一遍（映射回去和原图的重合）
        Detection("name_tag", Rect(10, 20, 50, 14), 0.9),
    ])
    w = far_watcher(det, FakeOcr({50: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert w.far_runs == 1 and w.nearby(0.0) == ["懒洋洋大王"]
    assert sum(t.cls == "social_ring" for t in w.last_tracks) == 1


def test_far_crops_zero_turns_it_off():
    det = CropDetector([small(1000)], [Detection("name_tag", Rect(10, 20, 50, 14), 0.9)])
    w = far_watcher(det, FakeOcr({50: "懒洋洋大王"}), far_crops=0)
    w.process(frame(), 0.0, panel_visible=False)
    assert det.crops == [] and w.nearby(0.0) == []


# ---- 三期 §2：认地图 ----
from skydango.vision.places import PlaceMatch  # noqa: E402


class FakePlaces:
    def __init__(self, answers):
        self.answers = list(answers)  # 依次返回的地名（None = 认不出）
        self.calls = []

    def recognize(self, img, boxes):
        self.calls.append(list(boxes))
        name = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return PlaceMatch(name, 0.9 if name else 0.5, name or "云野", 0.5)


def place_watcher(det, places, **cfg):
    return PerceptionWatcher(
        det, FakeOcr({}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, places=places, place_interval=30.0,
    )


def white():
    return np.full((1080, 1920, 3), 255, np.uint8)


def test_place_is_recognized_on_interval_and_after_scene_change():
    places = FakePlaces(["云野"])
    w = place_watcher(FakeDetector(), places)
    w.process(frame(), 0.0, panel_visible=False)
    assert len(places.calls) == 1 and w.place == "云野" and w.place_at == 0.0
    assert "看起来在：云野" in w.describe(0.0)
    w.process(frame(), 10.0, panel_visible=False)  # 画面没变、没到间隔
    assert len(places.calls) == 1
    w.process(white(), 12.0, panel_visible=False)  # 画面大变
    assert len(places.calls) == 2
    w.process(frame(), 13.0, panel_visible=False)  # 又大变，但离上次不到 3 s
    assert len(places.calls) == 2
    w.process(frame(), 16.0, panel_visible=False)  # 和上次认地图时比还是大变，隔够了
    assert len(places.calls) == 3
    w.process(frame(), 40.0, panel_visible=False)
    assert len(places.calls) == 3
    w.process(frame(), 46.0, panel_visible=False)  # 到 place_interval
    assert len(places.calls) == 4


def test_place_is_not_checked_while_paused():
    places = FakePlaces(["云野"])
    w = place_watcher(FakeDetector(), places)
    w.hold("camera")
    w.process(frame(), 0.0, panel_visible=False)
    assert places.calls == [] and w.place == ""


def test_unrecognized_place_keeps_the_old_one_until_place_keep():
    places = FakePlaces(["云野", None])
    w = place_watcher(FakeDetector(), places)
    w.process(frame(), 0.0, panel_visible=False)
    w.process(frame(), 30.0, panel_visible=False)
    assert len(places.calls) == 2 and w.place == "云野" and w.place_at == 0.0
    assert "云野" in w.describe(30.0) and "云野" not in w.describe(601.0)


def test_place_recognizer_gets_this_frames_boxes_to_mask():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110)]]
    places = FakePlaces(["云野"])
    w = place_watcher(det, places)
    w.process(frame(), 0.0, panel_visible=False)
    assert places.calls == [[Rect(1000, 400, 90, 220), Rect(990, 330, 110, 44)]]


def test_place_recognizer_error_does_not_break_the_frame():
    class Broken:
        def recognize(self, img, boxes):
            raise RuntimeError("模型坏了")

    det = FakeDetector()
    det.frames = [[tag(990, 110)]]
    w = PerceptionWatcher(det, FakeOcr({110: "懒洋洋大王"}), PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS),
                          log_roi=[0.0, 0.0, 0.335, 0.855], background=False, places=Broken())
    w.process(frame(), 0.0, panel_visible=False)
    assert w.nearby(0.0) == ["懒洋洋大王"] and w.place == ""


# ---- 三期 §3：别人对团子做的动作 ----
from skydango.config import GestureConfig  # noqa: E402


class FakeGesture:
    def __init__(self, answer=("wave", 0.95)):
        self.answer = answer
        self.calls = 0

    def classify(self, clip):
        self.calls += 1
        assert len(clip) == 16 and clip[0].shape == (112, 112, 3)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def gesture_watcher(det, gestures, ocr=None):
    return PerceptionWatcher(
        det, ocr or FakeOcr({110: "懒洋洋大王"}), PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, gestures=gestures, gesture_cfg=GestureConfig(),
    )


def feed(w, until, start=0.0, step=0.125):
    out = []
    t = start
    while t <= until + 1e-9:
        w.process(frame(), t, panel_visible=False)
        out += w.pop_gestures()
        t = round(t + step, 3)
    return out


def test_friend_waving_in_front_is_reported_once_per_cooldown():
    det = FakeDetector()
    det.frames = [[player(900), tag(890, 110)]]  # 正前方、近
    g = FakeGesture()
    w = gesture_watcher(det, g)
    assert feed(w, 1.75) == [] and g.calls == 0  # 还没攒够 16 张
    assert feed(w, 29.0, start=1.875) == [("懒洋洋大王", "wave")]
    assert g.calls > 1  # 每 2 s 还在判，只是冷却中不报
    assert feed(w, 32.5, start=29.125) == [("懒洋洋大王", "wave")]


def test_gesture_skips_strangers_far_people_and_side_people():
    det = FakeDetector()
    det.frames = [[
        player(900),  # 没有名字标签：不知道是谁
        player(1700), tag(1690, 110),  # 名字认得，但偏到右边
        player(600, h=60), tag(590, 120, y=350),  # 远
    ]]
    ocr = FakeOcr({110: "懒洋洋大王", 120: "番茄炒蛋盖饭"})
    g = FakeGesture()
    w = gesture_watcher(det, g, ocr)
    assert feed(w, 4.0) == [] and g.calls == 0


def test_gesture_needs_high_probability_and_not_none():
    for answer in (("wave", 0.5), ("none", 0.99)):
        det = FakeDetector()
        det.frames = [[player(900), tag(890, 110)]]
        w = gesture_watcher(det, FakeGesture(answer))
        assert feed(w, 4.0) == []


def test_gesture_classifier_error_does_not_break_the_frame():
    det = FakeDetector()
    det.frames = [[player(900), tag(890, 110)]]
    w = gesture_watcher(det, FakeGesture(RuntimeError("模型坏了")))
    assert feed(w, 4.0) == [] and w.nearby(4.0) == ["懒洋洋大王"]


def test_no_gestures_without_classifier():
    det = FakeDetector()
    det.frames = [[player(900), tag(890, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    assert feed(w, 3.0) == []


def test_far_tag_track_survives_between_crops_with_uneven_frame_times():
    det = CropDetector([small(1000)], [Detection("name_tag", Rect(10, 20, 50, 14), 0.9)])
    unknown = RecordingUnknown()
    ocr = FakeOcr({50: "星星小铺"})
    w = PerceptionWatcher(det, ocr, PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS),
                          log_roi=[0.0, 0.0, 0.335, 0.855], background=False, unknown=unknown)
    ids = set()
    for i in range(58):  # 0.07 s 一帧，4 s
        w.process(frame(), round(i * 0.07, 2), panel_visible=False)
        ids |= {t.id for t in w.last_tracks if t.cls == "name_tag"}
    assert len(ids) == 1  # 同一条标签轨迹接得上：投票和"只记一次"才有用
    assert [a[0] for a in unknown.added] == ["星星小铺"]


def test_place_is_cleared_when_scene_changed_and_nothing_matches():
    places = FakePlaces(["云野", None])
    w = place_watcher(FakeDetector(), places)
    w.process(frame(), 0.0, panel_visible=False)
    w.process(white(), 5.0, panel_visible=False)  # 画面大变后认不出：可能到了图库里没有的地方
    assert len(places.calls) == 2 and w.place == "" and "看起来在" not in w.describe(5.0)


def test_place_is_cleared_after_two_misses_in_a_row():
    places = FakePlaces(["云野", None])
    w = place_watcher(FakeDetector(), places)
    w.process(frame(), 0.0, panel_visible=False)
    w.process(frame(), 30.0, panel_visible=False)
    assert w.place == "云野"  # 偶尔一次认不出：先留着
    w.process(frame(), 60.0, panel_visible=False)
    assert w.place == ""


# ---- 评审暂缓项 ----
def test_gesture_clip_is_dropped_when_the_friend_walks_out_of_the_center():
    det = FakeDetector()
    g = FakeGesture()
    w = gesture_watcher(det, g)
    det.frames = [[player(900), tag(890, 110)]]
    feed(w, 0.625)  # 中间攒了 6 张
    det.frames = [[player(1700), tag(1690, 110)]]  # 走到右边（同一条轨迹接不上也没关系）
    feed(w, 1.5, start=0.75)
    det.frames = [[player(900), tag(890, 110)]]
    assert feed(w, 3.0, start=1.625) == [] and g.calls == 0  # 回来后重新攒，1.375 s 还不够 16 张


def test_pending_gestures_are_capped_when_nobody_takes_them():
    w = gesture_watcher(FakeDetector(), FakeGesture())
    for i in range(200):
        w._gesture_at.clear()
        with w._lock:
            w._gestures.append((f"好友{i}", "wave"))
    assert len(w.pop_gestures()) <= 50


def test_far_crop_backs_off_for_far_strangers_without_a_tag():
    det = CropDetector([small(1000)])  # 远处的人，裁剪里也从来没有标签（点过火的陌生人）
    w = far_watcher(det)
    for i in range(143):  # 0.07 s 一帧，10 s
        w.process(frame(), round(i * 0.07, 2), panel_visible=False)
    assert w.far_runs <= 5  # 前几次每 0.8 s 一次，之后每 5 s 一次


def test_far_crop_skipped_when_a_tag_already_sits_in_the_crop_area():
    near_miss = Detection("name_tag", Rect(986, 385, 50, 14), 0.9)  # 原图框到了，只是位置没挂上人
    det = CropDetector([small(1000), near_miss], [Detection("name_tag", Rect(10, 5, 50, 14), 0.9)])
    w = far_watcher(det, FakeOcr({50: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert w.far_runs == 0 and det.crops == []


def test_only_one_self_per_frame_rest_become_players():
    """实测（2026-09-28 view）：模型把躺在地上的橙衣玩家也认成了团子（0.71，真团子 0.97）。一帧只有一个团子。"""
    det = FakeDetector()
    det.frames = [[Detection("self", Rect(900, 400, 90, 220), 0.97), Detection("self", Rect(300, 700, 200, 120), 0.71)]]
    w = watcher(det)
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert [t.box for t in w.last_tracks if t.cls == "self"] == [Rect(900, 400, 90, 220)]
    assert w.strangers(2.0) == 1  # 另一个是没挂名字的人


def test_status_icons_are_not_requests():
    """眼睛（在看留影 / 听音乐）、陌生人平时的蜡烛图标、共享空间的入口：只是状态，不能当请求去点。"""
    for kind in ("eye", "stranger", "shared"):
        det = FakeDetector()
        det.frames = [[tag(990, 110), ring(1045), ring(1400)]]
        w = watcher(det, FakeOcr({110: "懒洋洋大王"}), icons=FakeIcons({"next": kind}))
        w.process(frame(), 0.0, panel_visible=False)
        assert w.requests == {}, kind
        assert w.circles["懒洋洋大王"][0] == kind


def _scene_with_icon(kind, center):
    import cv2

    img = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)
    icon = cv2.imread(f"assets/social/{kind}.png")
    img[center[1] - 40 : center[1] + 40, center[0] - 40 : center[0] + 40] = icon
    return img


def test_ring_labels_find_icons_above_people():
    """陌生人头顶的圆圈：弱标注只看好友名字下方，这里按人物框在头顶附近用图标模板找（2026-09-28 摸底 164 帧新找到 65 个）。"""
    from skydango.game.social import IconClassifier, load_icons

    icons = IconClassifier(load_icons("assets/social"))
    person = ("player", Rect(950, 400, 100, 220))  # 头顶 y=400，圆圈中心在头顶上方约 60 px
    img = _scene_with_icon("candle", (1000, 340))
    def near(rings, cx=1000, cy=340):
        return len(rings) == 1 and rings[0].w == rings[0].h == 100 and             abs(rings[0].x + 50 - cx) <= 4 and abs(rings[0].y + 50 - cy) <= 4

    assert near(ring_labels(img, [person], icons))
    assert near(ring_labels(img, [("player_unlit", person[1])], icons))
    assert ring_labels(img, [person, ("social_ring", Rect(955, 295, 100, 100))], icons) == []  # 已经有了不重复标
    assert ring_labels(img, [("self", person[1])], icons) == []  # 团子头顶不标
    assert ring_labels(img, [("player", Rect(300, 400, 100, 220))], icons) == []  # 图标不在这个人头顶
    assert ring_labels(np.full((1080, 1920, 3), (60, 90, 40), np.uint8), [person], icons) == []  # 只有背景


def test_one_name_tag_names_only_one_player():
    """实测（2026-09-28 view，新地图）：团子被认成 player、和好友挤在一起，两个人都挂上了"懒洋洋大王"。一个标签只给对得最正的那个人。"""
    det = FakeDetector()
    friend, me = player(990, y=400), player(1040, y=440)  # 好友在标签正下方；团子偏右、靠下，框有重叠
    det.frames = [[tag(975, 110), friend, me]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    named = [t.box for t in w.last_tracks if t.cls == "player" and t.data.get("name") == "懒洋洋大王"]
    assert named == [friend.box]
    assert w.strangers(2.0) == 1  # 另一个没挂上名字，过了 stranger_after 算陌生人


def test_name_leaves_a_player_when_its_tag_shows_up_elsewhere():
    """实测（2026-09-30 盯人）：好友走到镜头前的大个陌生人身后，名字标签挂到了陌生人身上；
    好友走远后标签清清楚楚在别处，陌生人还顶着好友的名字，track 就去盯陌生人、把好友转出画面。"""
    det = FakeDetector()
    big = player(0, y=350, w=760, h=720)  # 离镜头很近的陌生人
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), stranger_after=1.0)
    det.frames = [[big, tag(300, 110, y=300)]]  # 好友在他身后：标签落在他头顶的范围里
    w.process(frame(), 0.0, panel_visible=False)
    assert [t.data.get("name") for t in w.last_tracks if t.cls == "player"] == ["懒洋洋大王"]
    det.frames = [[big, tag(1400, 110, y=250)]]  # 好友走远了（远处的小人 YOLO 没框出来）
    for t in (0.5, 1.0, 1.5, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert [t.data.get("name") for t in w.last_tracks if t.cls == "player"] == [None]
    assert [p.kind for p in w.people(2.0)] == ["stranger"]  # 没了标签，过了 stranger_after 照常算陌生人
    assert w.labels["懒洋洋大王"][0] == 1400


def test_player_box_on_a_weak_self_box_is_the_dango():
    """实测（2026-09-30 21:47，好友站在团子正后方）：团子被认成 player（0.74），self 只有 0.29（低于 conf 被丢），
    好友的名字标签在团子头顶 → 名字挂到了团子身上，look_person 裁了团子的背影、也不换角度（找不到团子框）。
    同一个位置有个弱 self 框的 player 就是团子。"""
    det = FakeDetector()
    me = Rect(910, 346, 310, 669)
    det.frames = [[Detection("player", me, 0.74), Detection("self", Rect(900, 340, 325, 669), 0.29),
                   tag(936, 110, y=72, h=54)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    for t in (0.0, 1.0, 2.0):
        w.process(frame(), t, panel_visible=False)
    assert [(t.cls, t.data.get("name")) for t in w.last_tracks if t.box == me] == [("self", None)]
    assert w.people(2.0) == [] and w.strangers(2.0) == 0
    assert w.labels["懒洋洋大王"][0] == 936  # 标签照样认得（peek 靠它判断被挡住）


def test_weak_self_elsewhere_does_not_turn_a_player_into_the_dango():
    det = FakeDetector()
    det.frames = [[player(300), Detection("self", Rect(1300, 400, 90, 220), 0.29), tag(290, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    for t in (0.0, 1.0):
        w.process(frame(), t, panel_visible=False)
    assert [(p.kind, p.name) for p in w.people(1.0)] == [("friend", "懒洋洋大王")]


def test_people_lists_side_and_distance_left_to_right():
    from skydango.vision.people import describe_people

    det = FakeDetector()
    det.frames = [[player(300, y=300, h=300), tag(290, 110, y=230), player(1500, y=450, h=87),
                   Detection("self", Rect(900, 420, 90, 220), 0.9)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), stranger_after=1.0)
    for t in (0.0, 1.0):
        w.process(frame(), t, panel_visible=False)
    people = w.people(1.0)
    assert [(p.kind, p.name, p.side) for p in people] == [("friend", "懒洋洋大王", "左边"), ("stranger", None, "右边")]
    assert describe_people(people) == "懒洋洋大王（左边·近）、陌生人（右边·远）"
    assert watcher(FakeDetector()).people(0.0) == []


def test_describe_people_names_unlit_strangers():
    from skydango.vision.people import Person, describe_people

    assert describe_people([Person(3, "unlit", None, Rect(900, 400, 90, 220), "前面", "中")]) == "没点火的陌生人（前面·中）"
    assert describe_people([]) == ""


def test_people_is_empty_while_paused_or_stale():  # 评审：暂停 / 没跑检测时别一直报暂停前那一帧的人
    det = FakeDetector()
    det.frames = [[player(300, y=300, h=300), tag(290, 110, y=230)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.process(frame(), 0.0, panel_visible=False)
    assert [p.name for p in w.people(0.0)] == ["懒洋洋大王"]
    assert w.people(5.0) == []  # 5 秒没跑检测：这一帧已经过时了
    w.hold("camera")
    assert w.people(0.0) == []
    w.release("camera")


def test_people_does_not_count_self_as_unlit_stranger():  # 评审：暗图上团子可能同时出 self 和 player_unlit 两个框
    det = FakeDetector()
    det.frames = [[Detection("self", Rect(900, 420, 90, 220), 0.9), unlit(900, y=420)]]
    w = watcher(det)
    w.process(frame(), 0.0, panel_visible=False)
    assert w.strangers(0.0) == 0 and w.people(0.0) == []


def test_typing_seen_counts_friends_by_default():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), bubble(1005, y=260)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.process(frame(), 10.0, panel_visible=False)
    assert w.typing_seen(10.5) is True
    assert w.typing_seen(11.5) is False  # 超过 within（1 秒）：气泡已经不在了


def test_typing_seen_strangers_only_when_asked():
    det = FakeDetector()
    det.frames = [[unlit(1500), bubble(1505)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.typing_seen(10.0) is False
    assert w.typing_seen(10.0, strangers=True) is True


def test_typing_seen_ignores_self():
    det = FakeDetector()
    det.frames = [[Detection("self", Rect(900, 400, 90, 220), 0.9), player(905), bubble(905)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert w.typing_seen(10.0, strangers=True) is False


# ---- 物品（spec 2026-09-29-object-recognition） ----
def test_object_distance_and_describe_things():
    from skydango.vision.people import Thing, describe_things, object_distance

    assert object_distance(1000, 1080, 0.85, 0.65) == "近"
    assert object_distance(800, 1080, 0.85, 0.65) == "中"
    assert object_distance(500, 1080, 0.85, 0.65) == "远"
    things = [Thing(1, "bench", Rect(0, 0, 1, 1), "左边", "近"), Thing(2, "spirit", Rect(0, 0, 1, 1), "前面", "远")]
    assert describe_things(things) == "座位（左边·近）、先祖（前面·远）"
    assert describe_things([]) == ""


def obj(cls, x, y, w, h):
    return Detection(cls, Rect(x, y, w, h), 0.9)


def three_objects():
    return [obj("bench", 200, 900, 200, 100), obj("spirit", 900, 300, 80, 200), obj("bonfire", 1500, 700, 120, 100)]


def test_objects_need_hits_and_sort():
    det = FakeDetector()
    det.frames = [three_objects()]
    w = watcher(det)
    for t in (0.0, 0.1):
        w.process(frame(), t, panel_visible=False)
    assert w.objects(0.1) == []  # 才看到两帧
    w.process(frame(), 0.2, panel_visible=False)
    assert [(o.kind, o.side, o.distance) for o in w.objects(0.2)] == [
        ("bench", "左边", "近"), ("spirit", "前面", "远"), ("bonfire", "右边", "中")]


def test_objects_empty_when_stale_or_paused():
    det = FakeDetector()
    det.frames = [three_objects()]
    w = watcher(det)
    for t in (0.0, 0.1, 0.2):
        w.process(frame(), t, panel_visible=False)
    assert w.objects(2.2) == []  # 最近一帧过时了
    w.hold("x")
    assert w.objects(0.2) == []


def test_spirit_among_people_not_counted_as_stranger():
    det = FakeDetector()
    det.frames = [[player(1000, y=400, h=220), obj("spirit", 1045, 400, 90, 220)]]
    w = watcher(det)
    for t in (0.0, 0.1, 0.2):
        w.process(frame(), t, panel_visible=False)
    things, people = w.objects(0.2), w.people(0.2)
    assert [o.kind for o in things] == ["spirit"]
    assert all(p.track_id != things[0].track_id for p in people) and len(w.last_tracks) == 2


def test_overlay_has_object_labels():
    det = FakeDetector()
    det.frames = [three_objects()]
    w = watcher(det)
    w.process(frame(), 0.0, panel_visible=False)
    bench = next(b for b in w.overlay(0.0) if b["kind"] == "bench")
    assert bench["label"] == "座位"


def test_objects_empty_with_six_class_model():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110)]]
    w = watcher(det)
    for t in (0.0, 0.1, 0.2):
        w.process(frame(), t, panel_visible=False)
    assert w.objects(0.2) == []


def test_place_mask_keeps_objects_visible():
    # 评审：长椅、钢琴是认地图的地标，不遮（v4 建的图库也没遮它们）
    det = FakeDetector()
    det.frames = [[player(1000), obj("bench", 200, 900, 200, 100)]]
    places = FakePlaces(["云野"])
    w = place_watcher(det, places)
    w.process(frame(), 0.0, panel_visible=False)
    assert places.calls == [[Rect(1000, 400, 90, 220)]]


def test_scene_boxes_for_places_add_skip_objects(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.config import Config

    model = tmp_path / "m.onnx"
    model.write_bytes(b"")

    class Det:
        def detect(self, img):
            return [player(1000), obj("instrument", 300, 500, 200, 150)]

    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: Det())
    cfg = Config()
    cfg.perception.model = str(model)
    assert cli._scene_boxes(cfg)(frame()) == [Rect(1000, 400, 90, 220)]


# ---- 空闲注意力用：谁在说话、谁走近了（plan 2026-09-30-idle-attention Task 2） ----

def friend_typing(w, det, t, bubble_on=True):
    det.frames = [[player(1000), tag(990, 110)] + ([bubble(1005, y=260)] if bubble_on else [])]
    w.process(frame(), t, panel_visible=False)


def test_talkers_reports_friend_with_bubble_and_position():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    friend_typing(w, det, 10.0)
    t = w.talkers(10.0)
    assert [(x.name, x.friend) for x in t] == [("懒洋洋大王", True)]
    assert t[0].x == pytest.approx(1045) and t[0].last == 10.0 and t[0].start == 10.0


def test_talkers_reports_stranger_without_name():
    det = FakeDetector()
    det.frames = [[unlit(1500), bubble(1505)]]
    w = watcher(det)
    w.process(frame(), 10.0, panel_visible=False)
    assert [(x.name, x.friend) for x in w.talkers(10.0)] == [(None, False)]


def test_talker_bubble_start_resets_after_gap():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    for i in range(6):  # 10.0 ~ 10.5 一直有气泡
        friend_typing(w, det, 10.0 + i * 0.1)
    assert w.talkers(10.5)[0].start == 10.0
    for i in range(15):  # 气泡消失 1.5 秒
        friend_typing(w, det, 10.6 + i * 0.1, bubble_on=False)
    friend_typing(w, det, 12.2)
    assert w.talkers(12.2)[0].start == 12.2  # 新的一句


def test_talker_continuous_bubble_longer_than_typing_window_keeps_start():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    for i in range(121):  # 连续打 12 秒（typing_window 8 秒）
        friend_typing(w, det, 10.0 + i * 0.1)
    assert w.talkers(22.0)[0].start == 10.0


def test_talkers_excludes_stale():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    friend_typing(w, det, 10.0)
    assert w.talkers(11.5) == []


def test_recent_approaches_keep_position_and_pop_still_works():
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    walk_up(w, det, 0.0)
    (who, cx, t), = w.recent_approaches(1.5)
    assert who == "懒洋洋大王" and cx == pytest.approx(1045) and 0.0 < t <= 1.4
    assert w.pop_approaches() == ["懒洋洋大王"]
    assert w.recent_approaches(1.5)  # pop 不影响
    assert w.recent_approaches(t + 6.0) == []  # 默认只要 5 秒内的


def test_recent_approaches_include_strangers():
    det = FakeDetector()
    w = watcher(det)
    walk_up(w, det, 0.0, tagged=False)
    assert [a[0] for a in w.recent_approaches(1.5)] == [STRANGER]


def test_bubble_gap_default():
    assert PerceptionConfig().bubble_gap == 1.0


def test_recent_approaches_follow_current_position():  # 整分支评审 2：走近的人之后挪了 / 不见了
    det = FakeDetector()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    walk_up(w, det, 0.0)
    for k in range(1, 6):  # 走到后接着往左挪（轨迹接得上）
        x = 1000 - 20 * k
        det.frames = [[Detection("player", Rect(x, 358, 90, 262), 0.9), tag(x - 10, 110, y=288)]]
        w.process(frame(), 1.4 + 0.1 * k, panel_visible=False)
    (who, cx, t), = w.recent_approaches(1.9)
    assert cx == pytest.approx(945)  # 现在的位置
    det.frames = [[]]
    for i in range(30):  # 人走没了（轨迹过期）
        w.process(frame(), 1.9 + i * 0.1, panel_visible=False)
    assert w.recent_approaches(4.9) == []


# ---- 点亮陌生人：在团子周围找火焰（spec 2026-10-01-light-flame-around-self §3） ----
from skydango.config import SocialConfig
from skydango.vision import perception as perception_mod
from skydango.vision.candle import Disk
from skydango.game.social import LIGHT_KEY

SELF = Rect(1200, 500, 100, 250)  # 团子框：H = 250，cx = 1250


def self_det(box=SELF, score=0.7):
    return Detection("self", box, score)


def unlit(x, y=400, w=90, h=220, score=0.9):
    return Detection("player_unlit", Rect(x, y, w, h), score)


def light_watcher(monkeypatch, flames):
    """flames：每次 find_flame 依次返回什么（用完了一直返回最后一个）；记下每次给的范围。"""
    seq, areas = list(flames), []

    def fake(frame, area, flame, s):
        areas.append(area)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(perception_mod, "find_flame", fake)
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    w.light_cfg, w.flame = SocialConfig(), np.ones((4, 4), np.uint8)
    w.areas = areas
    return w, det, clock


FLAME = Disk(1045, 480, 20.0, 0.9)


def run(w, t, clock):
    clock.t = t
    w.process(frame(), t, panel_visible=False)


def run_frames(w, clock, start, end, step=0.1):
    k = 0
    while start + k * step <= end + 1e-9:
        run(w, round(start + k * step, 3), clock)
        k += 1


def test_light_request_after_flame_seen_long_enough(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det(), unlit(1000)]]
    for t in (0.0, 1.0, 2.0, 2.5):
        run(w, t, clock)
        assert LIGHT_KEY not in w.requests
    run(w, 3.1, clock)
    req = w.requests[LIGHT_KEY]
    assert req.kind == "light" and req.pos == (1045, 480) and req.track == 1
    assert isinstance(w._flame["black"], float)


def test_light_request_without_any_person_box(monkeypatch):
    """10-01 22:52:39：晚上 YOLO 一个人物框都没给，火焰照样出请求。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests
    assert w._flame["black"] is None


def test_flame_over_bonfire_is_ignored(monkeypatch):
    """篝火的点燃图标也是火焰圆圈：落在篝火框上方的不算；同样的帧没有篝火就出请求。"""
    bonfire = Detection("bonfire", Rect(990, 470, 120, 100), 0.8)
    for dets, expect in (([self_det(), bonfire], False), ([self_det()], True)):
        w, det, clock = light_watcher(monkeypatch, [Disk(1045, 440, 20.0, 0.98)])
        det.frames = [dets]
        for t in (0.0, 1.0, 2.0, 2.5, 3.1):
            run(w, t, clock)
        assert (LIGHT_KEY in w.requests) is expect


def test_search_area_around_self_box(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[self_det()]]
    run(w, 0.0, clock)
    a = w.areas[-1]
    cfg = SocialConfig()
    assert a.x == round(1250 - cfg.light_area_x * 250) and a.x2 == round(1250 + cfg.light_area_x * 250)
    assert a.y == round(500 - cfg.light_area_up * 250) and a.y2 == 750


def test_no_self_box_no_search(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[unlit(1000)]]
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert w.areas == [] and LIGHT_KEY not in w.requests


def test_stale_self_track_still_used_within_self_hold(monkeypatch):
    """黑影贴着团子认不出团子时沿用最近的团子框最多 SELF_HOLD 秒。
    SELF_HOLD 内用 _me_last 继续找，过了就停止。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    run(w, 0.0, clock)
    det.frames = [[]]
    run(w, 0.5, clock)  # 这一帧没认出团子，但 0.5 秒前有
    assert len(w.areas) == 2
    run(w, 1.6, clock)  # SELF_HOLD 内（1.6 < 3.0）还在找
    assert len(w.areas) == 3
    run(w, 3.1, clock)  # 过了 SELF_HOLD（3.1 > 3.0）：不找
    assert len(w.areas) == 3


def test_self_box_held_while_hidden(monkeypatch):
    """黑影贴着团子认不出团子的情况。0.0 有 self_det()，0.3~2.9 只有 unlit（没有 self），
    火焰一直在 → 这期间照样在找（w.areas 增加），线索不断（w._flame["id"] 不变），
    3.1 秒出请求。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det(), unlit(1000)]]
    run(w, 0.0, clock)
    assert len(w.areas) == 1
    flame_id_start = w._flame["id"] if w._flame else None
    det.frames = [[unlit(1000)]]  # 0.3~2.9 秒没有 self，只有 unlit
    run_frames(w, clock, 0.3, 2.9, step=0.1)  # 每 0.1 秒跑一帧，让 0.3, 0.6, 0.9 等都扫描到
    assert len(w.areas) >= 5  # 应该有多次搜索（0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7）
    assert w._flame is not None and w._flame["id"] == flame_id_start  # 线索没断
    run(w, 3.1, clock)
    assert LIGHT_KEY in w.requests  # 出请求了


def test_self_box_hold_expires(monkeypatch):
    """0.0 有 self，之后一直没有 → 3.0 秒之后不再找（w.areas 不再增加）。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    run(w, 0.0, clock)
    assert len(w.areas) == 1
    det.frames = [[]]  # 之后没有 self
    for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        run(w, t, clock)
    area_count_at_3_0 = len(w.areas)
    run(w, 3.1, clock)
    assert len(w.areas) == area_count_at_3_0  # 不再搜索


def test_scan_throttled(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[self_det()]]
    for t in (0.0, 0.1, 0.2, 0.3):
        run(w, t, clock)
    assert len(w.areas) == 2  # DISK_EVERY = 0.3：0.0 和 0.3


def test_needs_one_sure_flame(monkeypatch):
    weak = Disk(1045, 480, 20.0, 0.75)
    w, det, clock = light_watcher(monkeypatch, [weak])
    det.frames = [[self_det()]]
    for t in (0.0, 1.0, 2.0, 2.5, 3.1, 3.5):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests


def test_long_gap_starts_new_clue(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME, None, None, FLAME, FLAME])
    det.frames = [[self_det()]]
    for t in (0.0, 0.5, 1.2, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests and w._flame["id"] == 2


def test_short_gap_keeps_clue(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME, FLAME, None, FLAME, FLAME])
    det.frames = [[self_det()]]
    for t in (0.0, 1.0, 1.5, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests and w._flame["id"] == 1


def test_jump_too_far_starts_new_clue(monkeypatch):
    far = Disk(1045 + 200, 480, 20.0, 0.9)  # 0.8 × 团子框高 > light_jump
    w, det, clock = light_watcher(monkeypatch, [FLAME, FLAME, far, far])
    det.frames = [[self_det()]]
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests and w._flame["id"] == 2


def test_flame_under_friend_tag_is_ignored(monkeypatch):
    """好友举蜡烛给团子点火：火焰在他名字标签下面的圆圈里，不是黑影。"""
    w, det, clock = light_watcher(monkeypatch, [Disk(1045, 400, 20.0, 0.95)])
    w.ocr = FakeOcr({110: "懒洋洋大王"})
    det.frames = [[self_det(), tag(990, 110, y=300)]]  # 标签 990~1100、300~344；排除区 y 300 ~ 300 + 4.5×44
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests


def test_unread_tag_does_not_exclude(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [Disk(1045, 400, 20.0, 0.95)])
    det.frames = [[self_det(), tag(990, 110, y=300)]]  # 名字没认出来（FakeOcr 空表）
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests


def test_flame_under_open_panel_is_ignored(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [Disk(600, 480, 20.0, 0.95)])  # 面板右边到 x = 643
    det.frames = [[self_det(Rect(700, 500, 100, 250))]]  # 团子框中心在面板外（面板里的框会被 _filter 去掉）
    for t in (0.0, 1.0, 2.0, 2.5, 3.1):
        clock.t = t
        w.process(frame(), t, panel_visible=True)
    assert LIGHT_KEY not in w.requests


def test_overlay_shows_current_flame(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    run(w, 0.0, clock)
    rings = [e for e in w.overlay(0.0) if e["label"].startswith("火焰")]
    assert rings and rings[0]["kind"] == "ring" and rings[0]["x"] == 1025


def test_resume_shifts_clue_times(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    run_frames(w, clock, 0.0, 1.0)
    w.hold("blackout")
    clock.t = 4.0
    w.release("blackout")  # 暂停 3 秒
    assert w._flame["first"] == 3.0 and w._flame["last"] == 3.9  # 最后一次扫在 0.9（0.3 秒一次）


def test_detector_conf_lower_with_light():
    from skydango.vision.perception import LIT_LOW, detector_conf

    cfg = PerceptionConfig(hardcases=False, track_low=False)
    assert detector_conf(cfg) == cfg.conf
    assert detector_conf(cfg, light=True) == LIT_LOW
    assert detector_conf(PerceptionConfig(), light=True) == LIT_LOW  # 默认开着低分框续轨迹（low_conf）时也再放低


def test_people_boxes_keep_low_scores(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[self_det(), unlit(1000, score=0.27), unlit(600, score=0.1)]]
    run(w, 0.0, clock)
    assert [round(s, 2) for _, s in w._people_boxes] == [0.27]


# ---- 孤儿圆圈：深色火焰圆盘不是举蜡烛请求（最终审查 #1） ----
from pathlib import Path  # noqa: E402

RINGS = Path(__file__).parent / "data" / "candle_rings"


def ring_frame(name, cx=1400, cy=428):
    """录像 c 截的圆圈小图（140×140，圆心在正中）贴进整张画面，圆心对准 ring(1400) 的框中心。"""
    f = frame()
    f[cy - 70 : cy + 70, cx - 70 : cx + 70] = cv2.imread(str(RINGS / name))
    return f


@pytest.mark.parametrize("name", ["disk-c-05.3s.png", "disk-c-28.4s.png"])
def test_orphan_dark_disk_is_not_a_candle_request_even_over_bright_background(name):
    """录像 c：半透明圆盘透出后面亮的塔（外环亮度 97~109），以前按"够暗"判断漏过 → 当成陌生人举蜡烛 → 点了圆盘团子跟着他走。
    现在只看有没有白圈；也不管点亮陌生人开没开（没开时以前直接不查）。"""
    det = FakeDetector()
    det.frames = [[ring(1400)]]
    w = watcher(det, icons=FakeIcons({"next": "candle"}))
    w.process(ring_frame(name), 0.0, panel_visible=False)
    assert STRANGER not in w.requests


@pytest.mark.parametrize("name", ["ring-c-22.5s.png", "ring-c-25.0s.png"])
def test_orphan_white_ring_is_still_a_candle_request(name):
    det = FakeDetector()
    det.frames = [[ring(1400)]]
    w = watcher(det, icons=FakeIcons({"next": "candle"}))
    w.process(ring_frame(name), 0.0, panel_visible=False)
    assert w.requests[STRANGER].kind == "candle"


def test_scene_watcher_loads_flame_only_for_brain(monkeypatch, tmp_path):
    """只有大脑模式的身体会举蜡烛：普通 Agent 不找火焰圆盘（否则 light 请求一直挂着）。"""
    from skydango import cli
    from skydango.config import Config
    import skydango.vision.detect as detect
    import skydango.vision.ocr as ocr

    monkeypatch.setattr(detect, "make_detector", lambda *a, **k: object())
    monkeypatch.setattr(ocr, "make_ocr", lambda *a, **k: FakeOcr({}))
    cfg = Config()
    cfg.perception.enabled = True
    assert "light" in cfg.social.accept_strangers
    assert cli._scene_watcher(cfg).flame is None
    assert cli._scene_watcher(cfg, light=True).flame is not None
    from types import SimpleNamespace

    cfg.perception.hardcases = False
    run = SimpleNamespace(path=tmp_path, hard=tmp_path / "hard")
    assert cli._scene_watcher(cfg, run=run, light=True).light_dir == tmp_path / "light"
    assert cli._scene_watcher(cfg, run=run).light_dir is None


def lit_setup(monkeypatch, flames, black_seq=(0.9,)):
    """团子身边火焰够久、出请求，身体举蜡烛（mark_tried）；black() 依次返回 black_seq（用完了一直返回最后一个）。"""
    w, det, clock = light_watcher(monkeypatch, flames)
    seq = list(black_seq)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: seq.pop(0) if len(seq) > 1 else seq[0])
    det.frames = [[self_det(), unlit(1000)]]
    run_frames(w, clock, 0.0, 3.1)
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    return w, det, clock, cid


def test_mark_tried_removes_request_and_blocks_new_ones(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    assert LIGHT_KEY not in w.requests and w._lighting["black0"] == 0.9
    run_frames(w, clock, 3.2, 5.0)
    assert LIGHT_KEY not in w.requests


def test_lit_false_before_lit_min(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 4.5)
    assert w.lit(cid, 3.1) is False  # 火焰没了、人也亮了，但举起才 1.4 秒
    run_frames(w, clock, 4.6, 5.5)
    assert w.lit(cid, 3.1) is True


def test_lit_false_while_flame_still_there(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    run_frames(w, clock, 3.2, 6.0)
    assert w.lit(cid, 3.1) is False


def test_lit_true_when_flame_gone_and_person_brightens(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME], black_seq=[0.9])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is True


def test_lit_false_when_flame_gone_but_still_black(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME], black_seq=[0.9])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is False


def test_lit_none_when_flame_gone_and_nobody_there(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[self_det()]]
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is None


def test_lit_low_score_person_counts(monkeypatch):
    """晚上黑影只有 0.27 的框：判结果时也看低分框，不然会误判"走了"。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[self_det(), unlit(1000, score=0.27)]]
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is False


def test_lit_person_must_be_near_flame(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[self_det(), player(400)]]  # 亮着的人在远处，不是他
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is None


def test_lit_without_dark_sighting_is_never_true(monkeypatch):
    """举蜡烛时火焰下面没人、之后也没看到过黑的人：原地出现亮的 player 也永远不判 True（宁晚勿早，等身体超时）。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]  # 举蜡烛时火焰下面没人：black0 = None
    run_frames(w, clock, 0.0, 3.1)
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    assert w._lighting["black0"] is None and w._lighting["dark_box"] is None
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.05)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 6.0)
    assert w.lit(cid, 3.1) is not True


def test_lit_not_fooled_by_other_bright_person(monkeypatch):
    """黑影走开、另一个本来就亮的人站到火焰原位置（和黑影的框 IoU < 0.3）：不是同一个人，不判点亮。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.9 if box.x == 1000 else 0.05)
    det.frames = [[self_det(), unlit(1000)]]
    run_frames(w, clock, 3.2, 3.6)  # 火焰灭了，黑影还在原地（看到他黑）
    det.frames = [[self_det(), player(1065, y=430)]]  # 黑影走开，另一个亮的人站进来
    run_frames(w, clock, 3.7, 6.0)
    assert w.lit(cid, 3.1) is not True


def test_lit_when_same_person_brightens(monkeypatch):
    """同一个框先黑后亮（unlit(1000) → player(1000)）：判点亮。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.9 if det.frames[0][1].cls == "player_unlit" else 0.05)
    run_frames(w, clock, 3.2, 3.6)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.7, 5.5)
    assert w.lit(cid, 3.1) is True


def test_lit_false_while_paused_or_stale(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[self_det()]]
    run_frames(w, clock, 3.2, 5.5)
    assert w.lit(cid, 3.1) is None
    clock.t = 6.5  # 1.1 秒没扫描（> FRAME_STALE + DISK_EVERY）
    assert w.lit(cid, 3.1) is False
    clock.t = 5.4
    w.hold("panel")
    assert w.lit(cid, 3.1) is False
    w.release("panel")


def test_lit_unknown_clue_is_false(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    assert w.lit(cid + 5, 3.1) is False


def test_light_done_not_lit_starts_cooldown(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("timeout")
    assert w._lighting is None
    run_frames(w, clock, 3.2, 10.0)
    assert LIGHT_KEY not in w.requests  # 火焰还在，但在冷却里
    run_frames(w, clock, 10.1, 63.6)  # 一直跑着（中间断开超过 DISK_GAP 会重新计时）
    assert LIGHT_KEY in w.requests  # 冷却过了：light_done 后 3.2 秒起了一条新线索、火焰一直在、一直接着


def test_light_done_lit_no_cooldown(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("lit")
    assert w._cooldown_until == float("-inf") and w._flame is None
    run_frames(w, clock, 3.2, 6.5)
    assert LIGHT_KEY in w.requests  # 又一个人（新线索）站够了


def test_light_done_twice_is_harmless(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("gone")
    w.light_done("exit")
    assert w._lighting is None


def test_lit_needs_two_bright_scans(monkeypatch):
    """判点亮要连续 LIT_SCANS 次扫描都看到他变亮：亮、黑、亮、亮 → 只有最后一次之后才是 True。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    val = [0.9]
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: val[0])
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 4.5)  # 火焰没了，人还黑着；最后一次火焰在 3.0
    assert w.lit(cid, 0.0) is False
    for t, v, want in ((4.8, 0.1, False), (5.1, 0.9, False), (5.4, 0.1, False), (5.7, 0.1, True)):
        val[0] = v
        run(w, t, clock)
        assert w.lit(cid, 0.0) is want, t


def test_lit_false_when_self_box_lost(monkeypatch):
    """团子框丢了超过 SELF_HOLD 就不再找火焰：火焰"消失"不能自己成立，之后 lit 是 False（不是 True / None）。
    SELF_HOLD 之内沿用最后的团子框照样真扫描（见 test_self_box_held_while_hidden），那时判出点亮是对的。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[player(1000)]]
    run_frames(w, clock, 3.2, 7.0)  # 最后一次认出团子是 3.0 秒左右，沿用到 6 秒多就过期
    assert w.lit(cid, 0.0) is False


def test_lit_uses_scan_time_not_frame_time(monkeypatch):
    """process() 处理新一帧的途中（帧已经开始、扫描还没做）身体线程调 lit：要按最近一次真扫描的时间判。
    以前按帧时间判：两帧隔 > DISK_GAP 时拿新帧时间去比旧扫描找到人的时间，人明明还在却判成 None（走了）。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 4.5)
    assert w.lit(cid, 0.0) is True
    seen = []
    real_far_tags = w._far_tags

    def far_tags_and_ask(*args, **kwargs):  # 在 _watch_flames 之前被调用：这一帧的扫描还没做
        seen.append(w.lit(cid, 0.0))
        return real_far_tags(*args, **kwargs)

    w._far_tags = far_tags_and_ask
    run(w, 5.7, clock)  # 卡了 1.2 秒才来下一帧（> DISK_GAP）
    assert seen == [False]  # 扫描太旧：拿不准，等等；不能是 None


def test_lit_false_when_last_scan_found_nobody(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    det.frames = [[self_det(), player(1000)]]
    run_frames(w, clock, 3.2, 4.5)
    assert w.lit(cid, 0.0) is True
    det.frames = [[self_det()]]
    run(w, 4.8, clock)  # 最近一次扫描没找到人，但没找到不满 DISK_GAP
    assert w.lit(cid, 0.0) is False


def test_lit_survives_pause_resume(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    run_frames(w, clock, 3.2, 4.0)
    before = w._lighting["scan_at"]
    w.hold("blackout")
    clock.t = 7.0
    w.release("blackout")
    assert w._lighting["scan_at"] > before + 2.9  # 暂停的时间不算"没扫过"
    assert w.lit(cid, 0.0) is False  # 火焰还在，不是 None


def test_light_done_pops_request(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det(), unlit(1000)]]
    run_frames(w, clock, 0.0, 3.1)
    req = w.requests[LIGHT_KEY]
    w.mark_tried(req.track)
    assert LIGHT_KEY not in w.requests
    w.requests[LIGHT_KEY] = req  # 模拟 mark_tried 落在 _watch_flames 算完和写入之间、请求被写回
    w.light_done("timeout")
    assert LIGHT_KEY not in w.requests


def test_diag_images_and_summary(monkeypatch, tmp_path):
    import json

    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[self_det(), unlit(1000)]]
    run_frames(w, clock, 0.0, 3.1)
    (d,) = list(tmp_path.iterdir())
    assert len(list(d.glob("request-*.jpg"))) == 1  # 出请求那一刻一张（dry-run 也有）
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    run_frames(w, clock, 3.2, 5.2)
    raised = sorted(d.glob("raised-*.jpg"))
    assert 4 <= len(raised) <= 6  # 每 DIAG_EVERY 秒一张
    w.light_done("timeout")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert s["clue"] == cid and s["result"] == "timeout" and s["black_raised"] is not None
    run_frames(w, clock, 5.3, 6.0)
    assert len(sorted(d.glob("raised-*.jpg"))) == len(raised)  # 结束后不再存


def test_diag_max_per_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(perception_mod, "DIAG_MAX", 3)
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[self_det()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    run_frames(w, clock, 3.2, 8.0)
    (d,) = list(tmp_path.iterdir())
    assert len(list(d.glob("*.jpg"))) == 3


def test_diag_runs_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(perception_mod, "DIAG_RUNS", 1)
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[self_det()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    w.light_done("lit")
    run_frames(w, clock, 3.2, 6.5)  # 第二次出请求：超过 DIAG_RUNS 不再建目录
    assert LIGHT_KEY in w.requests and len(list(tmp_path.iterdir())) == 1


def test_no_light_dir_saves_nothing(monkeypatch, tmp_path):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[self_det()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    run_frames(w, clock, 3.2, 4.0)
    w.light_done("gone")  # 没有存图目录：不出错


def test_low_boxes_for_light_do_not_change_weak_self_promotion():
    """检测器为点亮陌生人放低到 LIT_LOW 后：低分 self 不能再触发 promote_weak_self（和以前一样只认 conf 以上的门槛），
    低分黑影照样进 _people_boxes。"""
    det = FakeDetector()
    det.frames = [[Detection("player", SELF, 0.9), Detection("self", SELF, 0.22), unlit(1500, score=0.22)]]
    w = watcher(det, hardcases=False)
    w.process(frame(), 1.0, panel_visible=False)
    assert not any(d.cls == "self" for d in w.last_dets)
    assert w.last_low == []
    assert any(b.x == 1500 and s == 0.22 for b, s in w._people_boxes)


def test_lit_ignores_friend_under_flame(monkeypatch):
    """火焰消失后，原位置只有一个认得出名字的好友（本来就是亮的）：不是"别人"，不能判点亮。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    w.ocr = FakeOcr({110: "懒洋洋大王"})
    det.frames = [[self_det(), player(1000), tag(990, 110)]]
    run_frames(w, clock, 3.2, 6.0)
    assert w.lit(cid, 3.1) is None


def test_black_called_with_self_box_excluded(monkeypatch):
    """量多黑时把团子框传进去扣掉。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    calls = []

    def fake(f, box, v, exclude=None):
        calls.append(exclude)
        return 0.9

    monkeypatch.setattr(perception_mod, "black", fake)
    run_frames(w, clock, 3.2, 4.0)
    assert calls and all(c == SELF for c in calls)


def test_lit_unknown_black_is_not_bright(monkeypatch):
    """被团子挡住大半、black 量不准（None）：人还在，但不算变亮。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: None)
    det.frames = [[self_det(), unlit(1000)]]
    run_frames(w, clock, 3.2, 6.0)
    assert w.lit(cid, 3.1) is False


def test_reappearing_flame_near_lighting_keeps_it_alive(monkeypatch):
    """火焰断开超过 DISK_GAP 后在附近重新出现成新线索：还是同一团火，没灭，不能判点亮。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v, exclude=None: 0.1)
    run_frames(w, clock, 3.2, 4.4, step=0.3)  # 断开 1.4 秒，线索被丢
    assert w._flame is None
    new = Disk(1045 + 60, 480, 20.0, 0.9)  # 离原位置 60 px < 0.5 × 250
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: new)
    run_frames(w, clock, 4.5, 6.0, step=0.3)
    assert w._lighting["flame_last"] > 5.5 and w._lighting["pos"] == (new.x, new.y)
    assert w.lit(cid, 3.1) is False


# ---- 按 Q 喊一声 §1.1：轨迹续命（spec 2026-10-01-q-call） ----
def _faded(sticky=True):
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110)]]
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), keep=5.0, sticky_names=sticky)
    w.process(frame(), 0.0, panel_visible=False)
    det.frames = [[player(1000)]]  # 标签淡掉了，人还在
    return det, w


def test_tag_fades_but_track_keeps_friend_nearby():
    det, w = _faded()
    for i in range(1, 100):
        w.process(frame(), i * 0.1, panel_visible=False)
    assert w.nearby(9.9) == ["懒洋洋大王"] and w.strangers(9.9) == 0


def test_sticky_stops_when_track_breaks():
    det, w = _faded()
    for i in range(1, 31):
        w.process(frame(), i * 0.1, panel_visible=False)
    det.frames = [[]]  # 人也不见了：从这里开始算 keep
    for i in range(31, 100):
        w.process(frame(), i * 0.1, panel_visible=False)
    assert w.nearby(7.9) == ["懒洋洋大王"]
    assert w.nearby(8.1) == []


def test_sticky_names_off_is_old_behavior():
    det, w = _faded(sticky=False)
    for i in range(1, 60):
        w.process(frame(), i * 0.1, panel_visible=False)
    assert w.nearby(5.1) == []


def test_sticky_ignores_maybe_and_unlit():
    det = FakeDetector()
    det.frames = [[player(1000), Detection("player_unlit", Rect(400, 400, 90, 220), 0.9)]]
    w = watcher(det, keep=5.0)
    w.process(frame(), 0.0, panel_visible=False)
    (a,) = [t for t in w.last_tracks if t.cls == "player"]
    (b,) = [t for t in w.last_tracks if t.cls == "player_unlit"]
    a.data["maybe"] = "懒洋洋大王"  # 按外观认的：不续
    b.data.update(name="番茄炒蛋盖饭", tagged=True)  # 黑影轨迹上残留的名字：不续
    for i in range(1, 20):
        w.process(frame(), i * 0.1, panel_visible=False)
    assert "番茄炒蛋盖饭" not in w.last_seen


# ---- 按 Q 喊一声 §1.2、§1.3：呼喊窗口、贴边标签 ----
from skydango.vision.people import Seen  # noqa: E402
from skydango.vision.perception import FAR_MISSES  # noqa: E402


def test_call_window_collects_names_and_unnamed():
    det = FakeDetector()
    det.frames = [[player(1500), small(400, y=400)]]  # 右边的人、左边远处的小人
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), far_crops=0)
    w.process(frame(), 0.0, panel_visible=False)
    w.called(0.5)
    det.frames = [[player(1500), tag(1490, 110), small(400, y=400)]]  # 一喊，右边那人头顶亮出名字
    w.process(frame(), 1.0, panel_visible=False)
    assert w.call_result(0.5) is None  # 窗口还没结束
    w.process(frame(), 6.6, panel_visible=False)
    r = w.call_result(0.5)
    assert r.ended and r.at == 0.5
    assert r.friends["懒洋洋大王"].side == "右边" and r.friends["懒洋洋大王"].on_screen
    assert r.friends["懒洋洋大王"].distance in ("近", "中", "远")
    assert r.unnamed == 1
    assert w.call_result(9.9) is None  # 对不上的 at


def test_new_call_replaces_old_window():
    det = FakeDetector()
    det.frames = [[]]
    w = watcher(det)
    w.called(0.0)
    w.called(1.0)
    w.process(frame(), 7.5, panel_visible=False)
    assert w.call_result(0.0) is None and w.call_result(1.0).ended


def test_edge_tag_not_nearby_but_in_call_result():
    det = FakeDetector()
    det.frames = [[tag(1810, 100, y=500)]]  # 名字贴在屏幕右边（中心 1860）、下面没人：好友在画面外
    w = watcher(det, FakeOcr({100: "懒洋洋大王"}))
    w.called(0.0)
    w.process(frame(), 0.1, panel_visible=False)
    assert w.nearby(0.1) == []
    assert "懒洋洋大王" in w.labels  # 位置照记：盯人（track）靠它往画面外转
    w.process(frame(), 6.1, panel_visible=False)
    assert w.call_result(0.0).friends["懒洋洋大王"] == Seen("右边", None, on_screen=False)


def test_edge_band_zero_is_old_behavior():
    det = FakeDetector()
    det.frames = [[tag(1810, 100, y=500)]]
    w = watcher(det, FakeOcr({100: "懒洋洋大王"}), edge_band=0.0)
    w.process(frame(), 0.1, panel_visible=False)
    assert w.nearby(0.1) == ["懒洋洋大王"]


def test_edge_tag_over_a_person_still_counts():
    det = FakeDetector()
    det.frames = [[player(1800), tag(1790, 110)]]  # 人站在画面最右边、标签挂在他头上：在画面里
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}))
    w.process(frame(), 0.1, panel_visible=False)
    assert w.nearby(0.1) == ["懒洋洋大王"]


def test_call_window_shifts_while_paused():
    clock = Clock(0.0)
    det = FakeDetector()
    det.frames = [[]]
    w = watcher(det, clock=clock)
    w.called(0.0)
    clock.t = 1.0
    w.hold("camera")
    clock.t = 5.0
    w.release("camera")  # 停了 4 秒：窗口变成 [4, 10]
    w.process(frame(), 7.0, panel_visible=False)
    assert w.call_result(0.0) is None
    w.process(frame(), 10.1, panel_visible=False)
    assert w.call_result(0.0).ended


def test_far_crops_ignore_backoff_and_double_in_window():
    def backed_off(call):
        det = CropDetector([small(200 + 300 * i) for i in range(5)])
        w = far_watcher(det, far_crops=3)
        w.process(frame(), 0.0, panel_visible=False)
        for t in w.tracker.tracks.values():  # 都连着几次没找到标签：平时要等 5 秒才再裁
            t.data["far_miss"] = FAR_MISSES
            t.data["far_at"] = 0.0
        if call:
            w.called(0.9)
        runs = w.far_runs
        w.process(frame(), 1.0, panel_visible=False)
        return w.far_runs - runs

    assert backed_off(False) == 0
    assert backed_off(True) == 5  # 窗口里不退避、每帧最多 3 × 2 块
