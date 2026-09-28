import numpy as np
import pytest

from skydango.config import EnvConfig, PerceptionConfig
from skydango.game.social import IDLE
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection, _parse_names, decode, letterbox
from skydango.vision.ocr import OcrLine, RapidOcrEngine
from skydango.vision.perception import STRANGER, PerceptionWatcher, detector_conf
from skydango.vision.track import Tracker, iou
from skydango.vision.weaklabel import data_yaml, hard_images, merge_labels, split_of, weak_labels, yolo_line

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
