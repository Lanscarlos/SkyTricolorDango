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
    w.process(frame(), 0.0, panel_visible=False)
    assert w.requests[STRANGER].kind == "candle"
    det.frames = [[]]
    w.process(frame(), 0.1, panel_visible=False)
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
    assert detector_conf(PerceptionConfig(conf=0.35, hardcases=False)) == 0.35


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
