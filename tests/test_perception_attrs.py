"""感知层接上第二层（spec 2026-10-02-perception-attrs §3.3、§3.4）：低分框复核后放行、误框撤下、点没点火两边投票、开关。"""

import numpy as np

from skydango.config import AttrsConfig, EnvConfig, PerceptionConfig
from skydango.vision.attrs import FORMS, PersonAttrs
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.people import Person
from skydango.vision.perception import PerceptionWatcher, detector_conf

from test_perception import FRIENDS, FakeDetector, FakeOcr, frame, player, tag

NAME = FRIENDS[0]  # 懒洋洋大王：标签宽 110


def probs(**kw) -> list[float]:
    """外形六类的概率：给定的几类照给，剩下的平分。"""
    rest = (1.0 - sum(kw.values())) / (len(FORMS) - len(kw))
    return [kw.get(k, rest) for k in FORMS]


class FakeModel:
    """按裁图中心的灰度（scene() 给每个人的框涂了不同的灰度）返回指定的外形概率。"""

    size = 32

    def __init__(self, by_value: dict[int, list[float]], fail: bool = False):
        self.by_value = by_value
        self.fail = fail
        self.calls = 0

    def pad(self, head):
        return 0.15

    def labels(self, head):
        return list(FORMS)

    def predict(self, items):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        out = []
        for _, img in items:
            v = int(img[img.shape[0] // 2, img.shape[1] // 2, 0])
            out.append({"form": np.array(self.by_value[v], np.float32)})
        return out


def scene(forms: dict[int, list[float]], y=400, w=90, h=220):
    """forms：人物框左上角 x → 外形概率。每个人的框涂成不同灰度（1, 2, …），返回 (画面, 假模型)。"""
    img = frame()
    by_value = {0: probs(not_person=0.9)}  # 没涂到的地方（不该被裁）
    for v, (x, p) in enumerate(forms.items(), start=1):
        img[y:y + h, x:x + w] = v
        by_value[v] = p
    return img, FakeModel(by_value)


def watcher(det, model=None, ocr=None, attrs_cfg=None, **cfg):
    cfg.setdefault("stranger_after", 1.0)
    attrs = PersonAttrs(attrs_cfg or AttrsConfig(enabled=True, every=0.0), model) if model is not None else None
    return PerceptionWatcher(
        det, ocr or FakeOcr({}), PerceptionConfig(**cfg), EnvConfig(), lambda: list(FRIENDS),
        log_roi=[0.0, 0.0, 0.335, 0.855], background=False, attrs=attrs,
    )


def low_player(x, s=0.3, cls="player", y=400, w=90, h=220):
    return Detection(cls, Rect(x, y, w, h), s)


def test_detector_conf_lowers_for_attrs():
    cfg = PerceptionConfig(hardcases=False, track_low=False)
    assert detector_conf(cfg) == cfg.conf
    assert detector_conf(cfg, attrs=True) == min(cfg.low_conf, cfg.conf)


def test_disabled_is_identical():
    """attrs=None：和接第二层之前逐字一样（数字是改动前跑出来的）；低分框不开轨迹，轨迹上不写 strong / admitted。"""
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), player(1500), Detection("player_unlit", Rect(300, 500, 60, 150), 0.9),
                   low_player(1700)]]
    w = watcher(det, ocr=FakeOcr({110: NAME}), keep=5.0)
    for t in (0.0, 0.5, 1.0):
        w.process(frame(), t, panel_visible=False)
    assert w.people(1.0) == [
        Person(4, "unlit", None, Rect(300, 500, 60, 150), "左边", "中"),
        Person(1, "friend", NAME, Rect(1000, 400, 90, 220), "前面", "近"),
        Person(3, "stranger", None, Rect(1500, 400, 90, 220), "右边", "近"),
    ]
    assert (w.strangers(1.0), w.unlit(1.0)) == (2, 1)
    assert w.overlay(1.0) == [
        {"x": 1000, "y": 400, "w": 90, "h": 220, "kind": "friend", "label": NAME, "score": 0.9},
        {"x": 990, "y": 330, "w": 110, "h": 44, "kind": "name", "label": NAME, "score": 0.9},
        {"x": 1500, "y": 400, "w": 90, "h": 220, "kind": "stranger", "label": "陌生人", "score": 0.9},
        {"x": 300, "y": 500, "w": 60, "h": 150, "kind": "unlit", "label": "陌生人（没点火）", "score": 0.9},
    ]
    assert w.nearby(1.0) == [NAME] and w.unnamed(1.0) == 1 and w.nearest(1.0) == ("陌生人", "近")
    assert w.tracker.open_low == frozenset()
    assert len(w.last_tracks) == 4
    assert not any(k in t.data for t in w.last_tracks for k in ("strong", "admitted"))


def test_low_score_person_appears_only_after_review():
    img, model = scene({1500: probs(lit=0.9)})
    det = FakeDetector()
    det.frames = [[low_player(1500)]]
    w = watcher(det, model)
    w.process(img, 0.0, panel_visible=False)
    assert w.people(0.0) == [] and w.strangers(0.0) == 0  # 复核一次不够
    (t,) = w.last_tracks
    assert t.data["admitted"] is False and not t.data.get("strong")
    w.process(img, 0.5, panel_visible=False)
    assert t.data["admitted"] is True  # 复核满 2 次放行
    assert w.strangers(0.5) == 0 and w.people(0.5) == []  # 还没到 stranger_after
    w.process(img, 1.0, panel_visible=False)
    assert w.strangers(1.0) == 1
    assert [(p.track_id, p.kind) for p in w.people(1.0)] == [(t.id, "stranger")]
    assert [e["kind"] for e in w.overlay(1.0)] == ["stranger"]


def test_high_score_tree_is_withdrawn():
    img, model = scene({1500: probs(not_person=0.9)})
    det = FakeDetector()
    det.frames = [[player(1500, h=220)]]
    w = watcher(det, model, stranger_after=0.0, keep=1.0)
    for i, t in enumerate((0.0, 0.5)):
        w.process(img, t, panel_visible=False)
        assert [p.kind for p in w.people(t)] == ["stranger"], i
    for t in (1.0, 1.5, 2.0, 2.5):
        w.process(img, t, panel_visible=False)
        assert w.people(t) == [] and w.overlay(t) == []
    (tr,) = w.last_tracks
    assert tr.data["rejected"] and tr.data["admitted"] is False
    assert w.strangers(2.5) == 0 and w.nearest(2.5) is None and w.unnamed(2.5) == 0


def test_withdrawn_box_is_not_a_person_under_the_flame():
    """撤下的误框不进 _people_boxes（点亮陌生人找"火焰下面那个人"用）；没接第二层时照旧在里面。"""
    box = Rect(1500, 400, 90, 220)
    img, model = scene({1500: probs(not_person=0.9)})
    det = FakeDetector()
    det.frames = [[player(1500)]]
    w = watcher(det, model)
    for t in (0.0, 0.5):
        w.process(img, t, panel_visible=False)
        assert [b for b, _ in w._people_boxes] == [box]
    w.process(img, 1.0, panel_visible=False)  # 第 3 票：撤下
    assert w.last_tracks[0].data["rejected"] and w._people_boxes == []

    off = watcher(FakeDetector(), None)
    off.detector.frames = [[player(1500)]]
    for t in (0.0, 0.5, 1.0):
        off.process(img, t, panel_visible=False)
    assert [b for b, _ in off._people_boxes] == [box]


def test_unvoted_weak_track_is_not_kept_alive():
    img, model = scene({1500: probs(not_person=0.9)})
    det = FakeDetector()
    det.frames = [[low_player(1500)]]
    w = watcher(det, model)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        w.process(img, t, panel_visible=False)
        assert w.people(t) == [] and w.strangers(t) == 0
    (tr,) = w.tracker.tracks.values()
    assert tr.strong_last == float("-inf") and tr.data["admitted"] is False
    assert not tr.data.get("rejected")  # 低分框判"不是人"：只是一直不放行，不算撤下


def test_dark_cloak_lit_player_not_counted_unlit():
    """YOLO 一直报黑影、外形头很肯定是点过火的（深色斗篷）：两边投票后算点过火的陌生人。"""
    img, model = scene({1500: probs(lit=0.95, unlit=0.01)})
    det = FakeDetector()
    det.frames = [[Detection("player_unlit", Rect(1500, 400, 90, 220), 0.9)]]
    w = watcher(det, model, attrs_cfg=AttrsConfig(enabled=True, every=0.0, yolo_w=0.3))
    for t in (0.0, 0.5, 1.0):
        w.process(img, t, panel_visible=False)
    assert w.unlit(1.0) == 0
    assert [p.kind for p in w.people(1.0)] == ["stranger"]
    assert [e["kind"] for e in w.overlay(1.0)] == ["stranger"]


def test_no_review_while_paused():
    img, model = scene({1500: probs(lit=0.9)})
    det = FakeDetector()
    det.frames = [[low_player(1500)]]
    w = watcher(det, model)
    w.process(img, 0.0, panel_visible=False)
    assert model.calls == 1
    w.hold("camera")
    w.process(img, 0.5, panel_visible=False)  # 暂停中（身体线程刚好 hold）：不裁、不投票，放行沿用
    assert model.calls == 1
    (t,) = w.last_tracks
    assert t.data["form_n"] == 1 and t.data["admitted"] is False


def test_cli_builds_attrs_only_when_model_loads(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.config import Config
    from skydango.vision import attrs as attrs_mod

    cfg = Config()
    assert cli._person_attrs(cfg) is None  # 默认关
    cfg.attrs.enabled = True
    cfg.attrs.model = str(tmp_path / "missing.npz")
    assert cli._person_attrs(cfg) is None  # 模型不存在：load_model 警告过，当作没开
    seen = {}

    def fake_load(acfg, device, embedder=None):
        seen["device"] = device
        return FakeModel({})

    monkeypatch.setattr(attrs_mod, "load_model", fake_load)
    cfg.perception.device = "cuda"
    pa = cli._person_attrs(cfg)
    assert isinstance(pa, PersonAttrs) and pa.enabled and seen["device"] == "cuda"  # device 空 = 跟 [perception]


def test_attrs_errors_turn_off_open_low():
    img, model = scene({1000: probs(lit=0.9)})
    model.fail = True
    det = FakeDetector()
    det.frames = [[player(1000)]]
    w = watcher(det, model, attrs_cfg=AttrsConfig(enabled=True, every=0.0, max_errors=2))
    assert w.tracker.open_low == frozenset({"player", "player_unlit"})
    for t in (0.0, 0.5):
        w.process(img, t, panel_visible=False)
    assert not w.attrs.enabled and w.tracker.open_low == frozenset()
    det.frames = [[player(1000), low_player(1500)]]
    w.process(img, 1.0, panel_visible=False)
    assert len(w.tracker.tracks) == 1  # 新出现的低分框不再开待复核的轨迹
    assert [p.kind for p in w.people(1.0)] == ["stranger"]  # 高分框照旧放行
