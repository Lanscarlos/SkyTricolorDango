"""第二层人物属性：模型格式、裁剪、配置（不加载真模型，用假 embedder）。"""

import logging

import numpy as np

from skydango.config import AttrsConfig, Config, load_config
from skydango.vision import attrs
from skydango.vision.attrs import crop, load_model, save_model
from skydango.vision.bubbles import Rect


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
