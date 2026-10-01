"""第二层人物属性：模型格式、裁剪、配置（不加载真模型，用假 embedder）。"""

import logging

import numpy as np

from skydango.config import AttrsConfig, Config, load_config
from skydango.vision import attrs
from skydango.vision.attrs import PersonAttrs, crop, load_model, save_model
from skydango.vision.bubbles import Rect
from skydango.vision.track import Track


class FakeEmbedder:
    size = 64
    key = "fake"

    def embed(self, img):
        return np.array([1.0, 0.0, 0.0, 0.0], np.float32)


def _heads():
    W = np.zeros((4, 3), np.float32)
    W[0] = [0.0, 3.0, 0.0]  # 第 0 维为 1：lit 最大
    return {
        "form": {"W": W, "b": np.zeros(3, np.float32), "labels": ["not_person", "lit", "unlit"],
                 "applies_to": ["player", "player_unlit"], "pad": 0.15},
        "icon": {"W": np.ones((4, 2), np.float32), "b": np.zeros(2, np.float32), "labels": ["a", "b"],
                 "applies_to": ["social_ring"], "pad": 0.3},
    }


def _model(tmp_path, embedder=None):
    p = tmp_path / "attrs.npz"
    save_model(p, _heads(), "dinov2-small.onnx", "64:imagenet", "2026-10-02")
    cfg = AttrsConfig(model=str(p), backbone="models/dinov2-small.onnx")
    return cfg, load_model(cfg, "cpu", embedder=embedder or FakeEmbedder())


def test_forms_constants():
    assert attrs.FORMS == ("not_person", "lit", "unlit", "spirit", "shared", "morph")
    assert attrs.PERSON_FORMS == attrs.FORMS[1:]
    assert attrs.CROP_PAD == 0.15


def test_crop_is_square_padded_and_sized():
    img = np.full((400, 400, 3), 255, np.uint8)
    out = crop(img, Rect(0, 100, 100, 200), 0.15, 64)
    assert out.shape == (64, 64, 3)
    assert (out[:, 0] == 0).all()  # 贴着图片左边：左侧补黑


def test_predict_uses_only_heads_for_that_class(tmp_path):
    _, m = _model(tmp_path)
    img = np.zeros((10, 10, 3), np.uint8)
    res = m.predict([("player", img), ("social_ring", img)])
    assert set(res[0]) == {"form"} and set(res[1]) == {"icon"}
    assert abs(float(res[0]["form"].sum()) - 1.0) < 1e-5
    assert m.labels("form")[int(np.argmax(res[0]["form"]))] == "lit"
    assert m.size == 64


def test_save_and_load_round_trip(tmp_path):
    _, m = _model(tmp_path)
    assert sorted(m.heads) == ["form", "icon"]
    assert m.labels("form") == ["not_person", "lit", "unlit"]
    assert m.applies_to("icon") == ["social_ring"]
    assert abs(m.pad("icon") - 0.3) < 1e-6
    img = np.zeros((10, 10, 3), np.uint8)
    p = m.predict([("player", img)])[0]["form"]
    assert np.allclose(p, np.exp([0, 3, 0]) / np.exp([0, 3, 0]).sum(), atol=1e-5)


def test_load_model_rejects_mismatched_backbone(tmp_path, caplog):
    cfg, _ = _model(tmp_path)

    class Other(FakeEmbedder):
        size = 32

    with caplog.at_level(logging.WARNING):
        assert load_model(cfg, "cpu", embedder=Other()) is None
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1
    # 主干文件名对不上
    cfg2 = AttrsConfig(model=cfg.model, backbone="models/other.onnx")
    assert load_model(cfg2, "cpu", embedder=FakeEmbedder()) is None
    # 文件不存在
    cfg3 = AttrsConfig(model=str(tmp_path / "nope.npz"))
    assert load_model(cfg3, "cpu", embedder=FakeEmbedder()) is None


def test_attrs_config_defaults(tmp_path):
    a = Config().attrs
    assert a.enabled is False and a.max_crops == 4 and a.accept == 0.6
    assert (a.model, a.backbone, a.device) == ("models/attrs.npz", "models/dinov2-small.onnx", "")
    assert (a.every, a.votes, a.reject, a.reject_n, a.yolo_w, a.flip_votes, a.max_errors) == (0.5, 5, 0.7, 3, 0.5, 3, 10)
    f = tmp_path / "c.toml"
    f.write_text("[attrs]\nenabled = true\nmax_crops = 2\n", encoding="utf-8")
    c = load_config(f)
    assert c.attrs.enabled is True and c.attrs.max_crops == 2


# ---- PersonAttrs.update ----


class FakeModel:
    """记录每次 predict 收到的项；probs 依次给（用完重复最后一个）。"""

    size = 32

    def __init__(self, probs=None, fail=False):
        self.calls: list[list[tuple[str, tuple]]] = []
        self.probs = probs or [np.array([0.1, 0.8, 0.1], np.float32)]
        self.fail = fail
        self.n = 0

    def pad(self, head):
        return 0.15

    def labels(self, head):
        return ["not_person", "lit", "unlit"]

    def predict(self, items):
        self.calls.append([(c, img.shape) for c, img in items])
        if self.fail:
            raise RuntimeError("boom")
        out = []
        for _ in items:
            out.append({"form": self.probs[min(self.n, len(self.probs) - 1)]})
            self.n += 1
        return out


FRAME = np.zeros((1080, 1920, 3), np.uint8)


def _track(tid, h, now, *, cls="player", form_n=0, crop_at=None, x=None):
    t = Track(id=tid, cls=cls, box=Rect(x if x is not None else 100 * tid, 300, h // 2, h), score=0.9,
              first=0.0, last=now)
    t.data["form_n"] = form_n
    if crop_at is not None:
        t.data["crop_at"] = crop_at
    return t


def _pa(model, **kw):
    return PersonAttrs(AttrsConfig(**kw), model)


def test_new_tracks_are_cropped_first_then_by_height():
    m = FakeModel()
    pa = _pa(m, max_crops=4)
    now = 10.0
    ts = [_track(1, 100, now), _track(2, 90, now)]  # 新的，矮
    ts += [_track(3 + i, 200 + 10 * i, now, form_n=2, crop_at=now - 0.6) for i in range(4)]  # 已复核，高 200..230
    pa.update(FRAME, ts, now, None)
    assert len(m.calls) == 1 and len(m.calls[0]) == 4
    done = {t.id for t in ts if t.data["form_n"] > (0 if t.id <= 2 else 2)}
    assert done == {1, 2, 5, 6}  # 2 条新的 + 最高的 2 条（高 220、230）


def test_recently_cropped_track_waits_every_seconds():
    m = FakeModel()
    pa = _pa(m, every=0.5)
    t = _track(1, 200, 10.0, form_n=1, crop_at=9.7)
    pa.update(FRAME, [t], 10.0, None)
    assert m.calls == []
    assert t.data["form_n"] == 1


def test_votes_average_last_n():
    probs = [np.array(p, np.float32) for p in
             ([0, 1, 0], [0, 1, 0], [0.9, 0.1, 0], [0.9, 0.1, 0], [0.9, 0.1, 0], [0, 0.2, 0.8], [0, 0.2, 0.8])]
    m = FakeModel(probs)
    pa = _pa(m, votes=5, every=0.5)
    t = _track(1, 200, 0.0)
    for i in range(7):
        t.last = float(i)
        pa.update(FRAME, [t], float(i), None)
    assert len(t.data["form_hist"]) == 5
    assert t.data["form_n"] == 7
    mean = np.mean(probs[2:], axis=0)
    label, p = t.data["form"]
    assert label == ["not_person", "lit", "unlit"][int(np.argmax(mean))]
    assert abs(p - float(mean.max())) < 1e-6
    assert t.data["crop_at"] == 6.0


def test_half_covered_by_panel_is_skipped():
    m = FakeModel()
    pa = _pa(m)
    t = _track(1, 200, 5.0, x=100)  # 框 100..200 × 300..500
    pa.update(FRAME, [t], 5.0, Rect(0, 0, 160, 1080))  # 盖住 60%
    assert m.calls == [] and "form" not in t.data
    pa.update(FRAME, [t], 5.0, Rect(0, 0, 140, 1080))  # 盖住 40%：要裁
    assert len(m.calls) == 1


def test_paused_style_skip_and_stale_tracks():
    m = FakeModel()
    pa = _pa(m)
    stale = _track(1, 200, 4.0)  # last < now
    pa.update(FRAME, [stale], 5.0, None)
    assert m.calls == []
    assert "cls_hist" not in stale.data


def test_non_person_and_rejected_not_cropped():
    m = FakeModel()
    pa = _pa(m)
    tag = _track(1, 100, 5.0, cls="name_tag")
    rej = _track(2, 200, 5.0)
    rej.data["rejected"] = True
    pa.update(FRAME, [tag, rej], 5.0, None)
    assert m.calls == []


def test_errors_disable_after_max_errors(caplog):
    m = FakeModel(fail=True)
    pa = _pa(m, max_errors=3, every=0.0)
    t = _track(1, 200, 0.0)
    with caplog.at_level(logging.WARNING):
        for i in range(3):
            t.last = float(i)
            assert pa.enabled
            pa.update(FRAME, [t], float(i), None)
    assert pa.enabled is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert t.data["form_n"] == 0 and "form_hist" not in t.data and "cls_hist" not in t.data
    n = len(m.calls)
    t.last = 9.0
    pa.update(FRAME, [t], 9.0, None)
    assert len(m.calls) == n


def test_success_resets_error_count():
    m = FakeModel()
    pa = _pa(m, max_errors=2, every=0.0)
    t = _track(1, 200, 0.0)
    for i, fail in enumerate((True, False, True)):
        m.fail = fail
        t.last = float(i)
        pa.update(FRAME, [t], float(i), None)
    assert pa.enabled


def test_cls_hist_records_every_updated_person_track():
    m = FakeModel()
    pa = _pa(m, votes=3, max_crops=1, every=10.0)
    ts = [_track(1, 200, 0.0), _track(2, 150, 0.0)]
    for i in range(5):
        for t in ts:
            t.last = float(i)
        ts[1].cls = "player_unlit" if i % 2 else "player"
        pa.update(FRAME, ts, float(i), None)
    assert ts[1].data["cls_hist"] == ["player", "player_unlit", "player"]
    assert "form_hist" not in ts[1].data or len(ts[1].data["form_hist"]) <= 3
