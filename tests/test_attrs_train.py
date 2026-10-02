import json
from argparse import Namespace

import numpy as np
import pytest

from skydango import cli
from skydango.config import Config
from skydango.imageio import imwrite
from skydango.vision import attrs_train as at
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection


def _touch(root, form, name, row=None, rows=None):
    p = root / "form" / form / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    if row is not None:
        rows.append({"crop": name, **row})


def _split_root(tmp_path):
    root = tmp_path / "attrs"
    rows: list[dict] = []
    for i in range(3):
        _touch(root, "lit", f"d{i}.jpg", {"source": "dataset", "split": "val" if i == 0 else "train", "group": "recA"}, rows)
    for g in range(10):  # 10 组录像来源的裁图，每组 3 张
        for i in range(3):
            _touch(root, "unlit", f"g{g}_{i}.jpg", {"source": "images", "split": None, "group": f"rec{g}"}, rows)
    _touch(root, "lit", "orphan.jpg")  # 没有 _crops.jsonl 行
    (root / "_crops.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


def test_split_crops_dataset_fixed_groups_whole_deterministic(tmp_path):
    root = _split_root(tmp_path)
    sp = at.split_crops(root, seed=1, val_ratio=0.3)
    assert "lit/d0.jpg" in sp["val"] and {"lit/d1.jpg", "lit/d2.jpg"} <= set(sp["train"])
    for g in range(10):
        names = {f"unlit/g{g}_{i}.jpg" for i in range(3)}
        assert names <= set(sp["val"]) or names <= set(sp["train"])  # 同组不跨集
    assert any(n.startswith("unlit/") for n in sp["val"]) and any(n.startswith("unlit/") for n in sp["train"])
    assert "lit/orphan.jpg" in sp["train"] + sp["val"]
    assert (root / "_split.json").is_file()
    assert at.split_crops(root, seed=1, val_ratio=0.3) == sp
    assert set(sp["train"]).isdisjoint(sp["val"])


def test_merge_labels_cases():
    ok = {"not_person": 30, "lit": 30, "unlit": 30, "spirit": 30, "shared": 30, "morph": 30}
    m, notes = at.merge_labels(ok, 20)
    assert m == {f: f for f in ok} and notes == []
    m, notes = at.merge_labels({**ok, "shared": 3, "morph": 0}, 20)
    assert m["shared"] == "lit" and m["morph"] == "lit" and len(notes) == 2
    m, notes = at.merge_labels({**ok, "spirit": 5}, 20)
    assert m["spirit"] is None and notes
    for bad in ("not_person", "lit", "unlit"):
        with pytest.raises(ValueError):
            at.merge_labels({**ok, bad: 2}, 20)


def test_train_head_separable():
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0, 0], 0.05, (40, 4))
    b = rng.normal([0, 1, 0, 0], 0.05, (40, 4))
    X = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 40 + [1] * 40)
    W, b_ = at.train_head(X[::2], y[::2], 2, 1e-3, np.ones(2, np.float32))
    assert W.shape == (4, 2) and b_.shape == (2,)
    assert ((X[1::2] @ W + b_).argmax(1) == y[1::2]).mean() == 1.0
    W2, b2, l2 = at.fit_head(X[::2], y[::2], X[1::2], y[1::2], 2)
    assert l2 in at.L2_CHOICES and ((X[1::2] @ W2 + b2).argmax(1) == y[1::2]).all()


class CountEmbedder:
    size = 8
    key = "C:/x/dino.onnx:1:2:8:imagenet"

    def __init__(self):
        self.calls = 0

    def embed(self, img):
        self.calls += 1
        v = np.array([float(img[:, :, 0].mean()), 1.0, 0.0], np.float32)
        return v / np.linalg.norm(v)


def test_features_cache(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / "f" / f"{i}.jpg"
        imwrite(p, np.full((16, 16, 3), 40 * (i + 1), np.uint8))
        paths.append(p)
    emb = CountEmbedder()
    a = at.features(paths, emb, tmp_path / "cache", flip=False)
    assert a.shape == (3, 3) and emb.calls == 3
    b = at.features(paths, emb, tmp_path / "cache", flip=False)
    assert emb.calls == 3 and np.allclose(a, b)
    at.features(paths, emb, tmp_path / "cache", flip=True)
    assert emb.calls == 6  # 镜像是另一份缓存
    assert at.features([], emb, tmp_path / "cache", flip=False).shape[0] == 0


class FakeDet:
    def __init__(self, per_frame):
        self.per_frame = per_frame

    def detect(self, img):
        return list(self.per_frame[int(img[0, 0, 0])])


class FakeModel:
    """按调用顺序给概率（头 form 的标签 not_person / lit / unlit）。"""
    size = 8

    def __init__(self, queue):
        self.queue = list(queue)

    def pad(self, head):
        return 0.15

    def labels(self, head):
        return ["not_person", "lit", "unlit"]

    def predict(self, items):
        return [{"form": np.array(self.queue.pop(0), np.float32)} for _ in items]


def _d(cls, x, s):
    return Detection(cls, Rect(x, 20, 40, 60), s)


def _replay_set(tmp_path):
    root = tmp_path / "sky"
    for tag, stem, label in ((1, "f1", "0 0.2 0.5 0.2 0.6\n4 0.6 0.5 0.2 0.6\n3 0.9 0.5 0.1 0.3\n"),
                             (2, "f2", "0 0.45 0.5 0.2 0.6\n")):
        img = np.full((100, 200, 3), 50, np.uint8)
        img[0, 0, 0] = tag
        imwrite(root / "images/val" / f"{stem}.png", img)
        (root / "labels/val").mkdir(parents=True, exist_ok=True)
        (root / "labels/val" / f"{stem}.txt").write_text(label, encoding="utf-8")
    return [root / "images/val/f1.png", root / "images/val/f2.png"]


def test_replay_counts(tmp_path):
    frames = _replay_set(tmp_path)
    # GT 框（200×100 画面）：f1 lit x20~60、unlit x100~140、self 忽略；f2 lit x70~110
    det = FakeDet({
        1: [_d("player", 20, 0.9), _d("player", 100, 0.9), _d("player", 150, 0.3)],  # 第二个 GT 是 unlit 却被认成 player；第三个是低分假框
        2: [_d("player", 70, 0.3), _d("player", 150, 0.8)],  # 低分真框 + 高分假框
    })
    model = FakeModel([
        [0.0, 1.0, 0.0],     # f1 A：lit
        [0.01, 0.01, 0.98],  # f1 B：unlit
        [0.9, 0.05, 0.05],   # f1 C 低分假框：不是人
        [0.1, 0.8, 0.1],     # f2 低分真框：lit .8
        [0.95, 0.03, 0.02],  # f2 高分假框：不是人 .95
    ])
    r = at.replay(frames, det, model, conf_low=0.2, conf=0.5, accept=0.6, reject=0.7)
    b, s = r["baseline"], r["second"]
    assert (b["tp"], b["fp"], b["fn"]) == (2, 1, 1)  # 基线：A、B 对，高分假框错，低分真框漏
    assert (s["tp"], s["fp"], s["fn"]) == (3, 0, 0)
    assert (b["pairs"], b["mismatch"]) == (2, 1)  # B 被 YOLO 认成 player（亮）
    # 单帧局限：player 框要外形头黑影占比恰好 1.0 才翻成黑影，0.98 翻不了 -> 第二层的认反数就是 YOLO 自己的
    assert (s["pairs"], s["mismatch"]) == (3, 1)
    assert "基本就是 YOLO 自己的答案" in "\n".join(at.replay_md(r))
    assert r["frames"] == 2 and r["gt"] == 3


def det_again():
    return FakeDet({
        1: [_d("player", 20, 0.9), _d("player", 100, 0.9), _d("player", 150, 0.3)],
        2: [_d("player", 70, 0.3), _d("player", 150, 0.8)],
    })


def model_queue():
    return [[0.0, 1.0, 0.0], [0.01, 0.01, 0.98], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.95, 0.03, 0.02]]


def test_sweep_picks_best_recall_not_below_baseline_precision(tmp_path):
    frames = _replay_set(tmp_path)
    records = at.collect(frames, det_again(), FakeModel(model_queue()), 0.2)
    accept, reject = at.sweep_thresholds(records, 0.5)
    s = at.simulate(records, 0.5, accept, reject)["second"]
    assert (s["tp"], s["fp"], s["fn"]) == (3, 0, 0)  # 这批数据上所有阈值组合里最好的


def test_attrs_train_refuses_overwrite_of_configured_model(tmp_path):
    cfg = Config()
    cfg.attrs.model = str(tmp_path / "attrs.npz")
    args = Namespace(data=str(tmp_path), out=cfg.attrs.model, force=False, device="cpu")
    with pytest.raises(SystemExit, match="--force"):
        cli._perception_attrs_train(cfg, args)


def test_bench_attrs_times_detect_and_crops():
    frame = np.full((100, 200, 3), 50, np.uint8)
    frame[0, 0, 0] = 1
    det = FakeDet({1: [_d("player", 20, 0.9), _d("player", 100, 0.8), _d("player_unlit", 150, 0.7), _d("name_tag", 0, 0.9)]})
    model = FakeModel([[0.0, 1.0, 0.0]] * 2)
    plain, withattrs = at.bench_attrs(lambda i: frame, det, model, n=1, warmup=0, max_crops=2)
    assert len(plain) == len(withattrs) == 1 and model.queue == []  # 只对 2 个人物框裁图预测


class BucketEmbedder:
    size = 8
    key = "C:/x/b.onnx:1:2:8:imagenet"

    def embed(self, img):
        m = float(img.mean())
        return np.eye(3, dtype=np.float32)[0 if m < 70 else 1 if m < 170 else 2]


def test_run_training_end_to_end(tmp_path):
    root = tmp_path / "attrs"
    for form, level in (("not_person", 20), ("lit", 120), ("unlit", 220)):
        for i in range(8):
            imwrite(root / "form" / form / f"{i}.jpg", np.full((16, 16, 3), level + i, np.uint8))
    res = at.run_training(root, BucketEmbedder(), tmp_path / "cache", min_per_class=3)
    assert res["labels"] == ["not_person", "lit", "unlit"] and res["mapping"]["spirit"] is None
    assert res["eval"]["macro_f1"] == 1.0 and res["val_n"] > 0 and len(res["notes"]) == 3
    from skydango.vision import attrs
    out = tmp_path / "m.npz"
    attrs.save_model(out, {"form": res["head"]}, "d.onnx", "8:imagenet", "20261002")
    assert out.is_file()
    md = at.report_md(data=root, model=out, when=__import__("datetime").datetime(2026, 10, 2), result=res, replays=[], suggest=None, notes=[])
    assert "混淆矩阵" in md and "not_person" in md


def test_default_out_and_final_path_guard(tmp_path, monkeypatch):
    import datetime as dt
    now = dt.datetime(2026, 10, 2, 9, 8, 7)
    assert at.default_out(tmp_path, now) == tmp_path / "attrs-20261002.npz"
    (tmp_path / "attrs-20261002.npz").write_bytes(b"x")
    assert at.default_out(tmp_path, now) == tmp_path / "attrs-20261002-090807.npz"
    with pytest.raises(SystemExit, match="--force"):
        at.check_out(tmp_path / "attrs-20261002.npz", str(tmp_path / "attrs-20261002.npz"), False)
    at.check_out(tmp_path / "attrs-20261002.npz", str(tmp_path / "attrs-20261002.npz"), True)
    # 默认路径恰好是 [attrs] model（当天还没有这个文件）：CLI 也要拒绝
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.attrs.model = str(tmp_path / "models" / f"attrs-{dt.datetime.now():%Y%m%d}.npz")
    (tmp_path / "d" / "form").mkdir(parents=True)
    with pytest.raises(SystemExit, match="--force"):
        cli._perception_attrs_train(cfg, Namespace(data=str(tmp_path / "d"), out=None, force=False, device="cpu"))


def test_split_pins_non_dataset_group_to_dataset_recording_split(tmp_path):
    root = tmp_path / "attrs"
    rows: list[dict] = []
    _touch(root, "lit", "ds_val.jpg", {"source": "dataset", "split": "val", "group": "recX"}, rows)
    _touch(root, "lit", "ds_tr.jpg", {"source": "dataset", "split": "train", "group": "recX"}, rows)
    _touch(root, "lit", "ds_tr2.jpg", {"source": "dataset", "split": "train", "group": "recY"}, rows)
    for i in range(4):
        _touch(root, "unlit", f"x{i}.jpg", {"source": "images", "split": None, "group": "recX"}, rows)
        _touch(root, "unlit", f"y{i}.jpg", {"source": "images", "split": None, "group": "recY"}, rows)
    (root / "_crops.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    for seed in range(5):
        sp = at.split_crops(root, seed=seed, val_ratio=0.5)
        assert all(f"unlit/x{i}.jpg" in sp["val"] for i in range(4))
        assert all(f"unlit/y{i}.jpg" in sp["train"] for i in range(4))
        assert sp["pinned"] == {"recX": "val", "recY": "train"}
