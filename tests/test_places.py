import os

import numpy as np
import pytest

from skydango.config import PlacesConfig
from skydango.imageio import imread, imwrite
from skydango.vision.bubbles import Rect
from skydango.vision.places import (
    PlaceLibrary,
    PlaceMatch,
    PlaceRecognizer,
    ThumbEmbedder,
    decide,
    make_embedder,
    mask_scene,
)

KEEP = [0.0, 0.0, 1.0, 0.89]
PANEL = [0.0, 0.0, 0.335, 0.855]


def noise(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


def gradient(color, seed=0):
    """一个"地方"：某种颜色的上下渐变 + 一点噪声（同一个地方不同截图）。"""
    ramp = np.linspace(0.3, 1.0, 1080)[:, None, None]
    img = (ramp * np.array(color, float)[None, None, :]).repeat(1920, axis=1)
    img = img + np.random.default_rng(seed).normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_mask_scene_paints_boxes_panel_and_bottom_with_mean_color():
    img = noise()
    before = img.copy()
    mean = img.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    out = mask_scene(img, [Rect(1000, 100, 50, 50)], KEEP, [PANEL])
    assert (img == before).all()  # 不改原图
    for x, y in ((1020, 120), (996, 96), (50, 50), (1500, 1070)):  # 框（放宽 10%）、面板、底部按钮栏
        assert (out[y, x] == mean).all(), (x, y)
    assert (out[500, 1500] == before[500, 1500]).all()


def test_decide_needs_min_and_margin_over_other_places():
    assert decide([("云野", 0.9), ("云野", 0.88), ("雨林", 0.8)], 0.8, 0.05) == PlaceMatch("云野", 0.9, "云野", 0.8)
    assert decide([("云野", 0.9), ("雨林", 0.87)], 0.8, 0.05).name is None  # 拉不开差距：宁可不说
    assert decide([("云野", 0.9), ("雨林", 0.87)], 0.8, 0.05).best == "云野"
    assert decide([("云野", 0.7)], 0.8, 0.05).name is None
    assert decide([("云野", 0.85)], 0.8, 0.05).name == "云野"  # 图库里只有一个地方：只看 place_min
    assert decide([], 0.8, 0.05) == PlaceMatch(None, 0.0, None, 0.0)


class CountingEmbedder:
    def __init__(self, key="fake"):
        self.key = key
        self.calls = 0

    def embed(self, img):
        self.calls += 1
        v = img.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.0
        return v / np.linalg.norm(v)


def gallery(root):
    for name, color in (("云野", (40, 200, 60)), ("雨林", (200, 80, 40))):
        (root / name).mkdir(parents=True)
        imwrite(root / name / "a.jpg", gradient(color)[::8, ::8])
    (root / "_trash").mkdir()
    imwrite(root / "_trash" / "x.jpg", gradient((0, 0, 0))[::8, ::8])


def test_library_caches_vectors_and_invalidates_on_new_file_or_model(tmp_path):
    gallery(tmp_path)
    emb = CountingEmbedder()
    lib = PlaceLibrary(tmp_path, emb)
    assert lib.load() == 2 and emb.calls == 2 and lib.places() == ["云野", "雨林"]
    assert (tmp_path / "_index.npz").exists()
    emb2 = CountingEmbedder()
    assert PlaceLibrary(tmp_path, emb2).load() == 2 and emb2.calls == 0  # 缓存命中
    imwrite(tmp_path / "云野" / "b.jpg", gradient((40, 200, 60), 1)[::8, ::8])
    emb3 = CountingEmbedder()
    assert PlaceLibrary(tmp_path, emb3).load() == 3 and emb3.calls == 1  # 只算新加的
    path = tmp_path / "雨林" / "a.jpg"
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))  # 图换过了
    emb4 = CountingEmbedder()
    PlaceLibrary(tmp_path, emb4).load()
    assert emb4.calls == 1
    emb5 = CountingEmbedder(key="other-model")
    PlaceLibrary(tmp_path, emb5).load()
    assert emb5.calls == 3  # 换了模型：全部重算


def test_library_scores_can_exclude_one_image(tmp_path):
    gallery(tmp_path)
    lib = PlaceLibrary(tmp_path, CountingEmbedder())
    lib.load()
    vec = lib.embedder.embed(imread(tmp_path / "云野" / "a.jpg"))
    assert [n for n, _ in lib.scores(vec)] == ["云野", "雨林"]
    assert [n for n, _ in lib.scores(vec, exclude=tmp_path / "云野" / "a.jpg")] == ["雨林"]
    assert max(s for _, s in lib.scores(vec)) == pytest.approx(1.0, abs=1e-5)


def test_empty_library_recognizes_nothing(tmp_path):
    lib = PlaceLibrary(tmp_path / "places", CountingEmbedder())
    assert lib.load() == 0
    r = PlaceRecognizer(lib, PlacesConfig(), KEEP, [PANEL])
    assert r.recognize(noise(), []).name is None


def test_add_writes_under_place_folder_without_colons(tmp_path):
    lib = PlaceLibrary(tmp_path, CountingEmbedder(), wall=lambda: 0.0)
    a = lib.add("云野", gradient((40, 200, 60)))
    b = lib.add("云野", gradient((40, 200, 60), 1))
    assert a.parent == tmp_path / "云野" and a != b and ":" not in a.name + b.name
    assert a.exists() and b.exists() and lib.places() == ["云野"] and len(lib.entries) == 2


def test_thumb_embedder_is_normalized_and_tells_places_apart():
    emb = ThumbEmbedder()
    a1, a2 = emb.embed(gradient((40, 200, 60), 0)), emb.embed(gradient((40, 200, 60), 1))
    b = emb.embed(gradient((200, 80, 40)))
    assert np.linalg.norm(a1) == pytest.approx(1.0, abs=1e-5)
    assert float(a1 @ a2) > float(a1 @ b) + 0.05


def test_recognizer_masks_boxes_and_decides(tmp_path):
    gallery(tmp_path)
    seen = []

    class Spy(CountingEmbedder):
        def embed(self, img):
            seen.append(img)
            return super().embed(img)

    lib = PlaceLibrary(tmp_path, Spy())
    lib.load()
    seen.clear()
    r = PlaceRecognizer(lib, PlacesConfig(place_min=0.99, place_margin=0.0), KEEP, [PANEL])
    img = gradient((40, 200, 60))
    img[400:600, 1000:1100] = 255  # 一个人
    m = r.recognize(img, [Rect(1000, 400, 100, 200)])
    assert m.name == "云野"
    assert not (seen[0][500, 1050] == 255).all()  # 人被涂掉了


def test_make_embedder_thumb_and_missing_model(tmp_path):
    assert isinstance(make_embedder(PlacesConfig(model="thumb")), ThumbEmbedder)
    with pytest.raises((FileNotFoundError, ImportError)):
        make_embedder(PlacesConfig(model=str(tmp_path / "nope.onnx")))
    with pytest.raises(ValueError):
        make_embedder(PlacesConfig(model="places.bin"))


def _pool_model(path, size=None, three_d=False):
    """一个最小的"编码器"：全局平均池化 → 1×3（或 1×1×3，模拟 token 输出）。"""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    dim = size or "S"
    inp = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, dim, dim])
    nodes = [helper.make_node("GlobalAveragePool", ["pixel_values"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"])]
    out_name = "f"
    if three_d:
        nodes.append(helper.make_node("Unsqueeze", ["f", "axes"], ["t"]))
        out_name = "t"
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 1, 3] if three_d else [1, 3])
    inits = [helper.make_tensor("axes", TensorProto.INT64, [1], [1])] if three_d else []
    model = helper.make_model(helper.make_graph(nodes, "pool", [inp], [out], inits),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_onnx_embedder_preprocesses_rgb_and_normalizes(tmp_path):
    pytest.importorskip("onnxruntime")
    _pool_model(tmp_path / "m.onnx", size=64)
    emb = make_embedder(PlacesConfig(model=str(tmp_path / "m.onnx"), norm="none"))
    assert emb.size == 64 and emb.key == "m.onnx:64:none"
    img = np.zeros((90, 160, 3), np.uint8)
    img[..., 0] = 255  # BGR 的蓝色
    v = emb.embed(img)
    assert v.shape == (3,) and v[2] == pytest.approx(1.0) and v[0] == pytest.approx(0.0)  # RGB 顺序：蓝色在最后


def test_onnx_embedder_takes_cls_token_from_3d_output(tmp_path):
    pytest.importorskip("onnxruntime")
    _pool_model(tmp_path / "t.onnx", three_d=True)
    emb = make_embedder(PlacesConfig(model=str(tmp_path / "t.onnx"), size=32))
    assert emb.size == 32
    assert emb.embed(np.full((90, 160, 3), 128, np.uint8)).shape == (3,)
