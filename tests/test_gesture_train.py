import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from skydango.imageio import imwrite
from skydango.vision.gesture_train import (
    Sample, check_counts, list_samples, load_split, save_split, split,
)

LABELS = ["none", "wave", "cheer"]


def mk(root: Path, label: str, clip: str, frames: int = 16) -> None:
    d = root / label / clip
    d.mkdir(parents=True)
    for i in range(frames):
        (d / f"{i:02d}.jpg").write_bytes(b"x")


def smp(label, rec, start, n=0):
    clip = f"{rec}__{n:04d}_track1_t{start:.2f}s"
    return Sample(clip, label, rec, float(start), Path(clip))


def test_list_samples_and_incomplete(tmp_path):
    mk(tmp_path, "wave", "recA__0001_track3_t12.50s")
    mk(tmp_path, "none", "recB__0002_track1_t3.00s")
    mk(tmp_path, "cheer", "recA__0003_track1_t1.00s")
    mk(tmp_path, "other", "recZ__0001_track1_t1.00s")
    samples, skipped = list_samples(tmp_path, ["none", "wave"])
    assert {(s.label, s.recording, s.start) for s in samples} == {
        ("wave", "recA", 12.5), ("none", "recB", 3.0)}
    assert skipped == []


def test_incomplete_clip_not_in_dataset(tmp_path):
    mk(tmp_path, "wave", "recA__0001_track3_t1.00s", frames=15)
    mk(tmp_path, "wave", "recA__0002_track3_t2.00s")
    samples, skipped = list_samples(tmp_path, LABELS)
    assert [s.clip for s in samples] == ["recA__0002_track3_t2.00s"]
    assert skipped == ["recA__0001_track3_t1.00s"]


def test_check_counts_message():
    samples = [smp("cheer", "r", i, i) for i in range(12)]
    samples += [smp(l, "r", i, 100 + i) for l in ("none", "wave") for i in range(30)]
    assert check_counts(samples, LABELS) == ["欢呼只有 12 段，还差 8 段"]


def test_check_counts_none_name():
    samples = [smp("wave", "r", i, i) for i in range(30)] + [smp("cheer", "r", i, 50 + i) for i in range(30)]
    assert check_counts(samples, LABELS) == ["都不是只有 0 段，还差 20 段"]


def _multi():
    out = []
    for r in range(5):
        for label in LABELS:
            for i in range(4):
                out.append(smp(label, f"rec{r}", i * 3, len(out)))
    return out


def test_split_by_recording_keeps_recordings_whole():
    s = split(_multi(), LABELS, seed=1)
    rec = lambda names: {n.split("__")[0] for n in names}
    assert not rec(s["train"]) & rec(s["val"])
    assert not set(s["train"]) & set(s["val"])
    assert len(s["train"]) + len(s["val"]) == 60


def test_split_every_class_has_val_and_deterministic():
    samples = []
    for r, label in enumerate(["none", "wave", "cheer"] * 2):
        for i in range(5):
            samples.append(smp(label, f"rec{r}", i, len(samples)))
    for seed in range(5):
        s = split(samples, LABELS, seed=seed)
        for label in LABELS:
            assert any(x.label == label and x.clip in s["val"] for x in samples)
            assert any(x.label == label and x.clip in s["train"] for x in samples)
        assert s == split(samples, LABELS, seed=seed)
        assert s["warnings"] == []


def test_split_single_recording_by_time_with_gap():
    samples = [smp(l, "one", t, t * 2 + k) for t in range(100) for k, l in enumerate(["none", "wave"])]
    s = split(samples, ["none", "wave"], gap=2.0)
    starts = lambda names: [float(n.rsplit("_t", 1)[1][:-1]) for n in names]
    assert max(starts(s["train"])) < 78 and min(starts(s["val"])) == 80
    assert len(s["train"]) + len(s["val"]) == 2 * 98


def test_split_warns_when_no_val():
    s = split([smp("wave", "a", 1.0)], ["none", "wave"])
    assert any("wave" in w for w in s["warnings"])


def test_save_load_split(tmp_path):
    assert load_split(tmp_path) is None
    s = split(_multi(), LABELS, seed=3)
    save_split(tmp_path, s)
    assert load_split(tmp_path)["train"] == s["train"]


def _starts(names):
    return [float(n.rsplit("_t", 1)[1][:-1]) for n in names]


def _mixed():
    # R1 是 wave 唯一的录像，里面还有 none；none 另有两段录像
    out = []
    for t in range(50):
        out.append(smp("wave", "R1", t, len(out)))
        out.append(smp("none", "R1", t, len(out)))
    for r in ("R2", "R3"):
        for t in range(10):
            out.append(smp("none", r, t, len(out)))
    return out


def test_pinned_recording_cut_applies_to_all_labels():
    for seed in range(4):
        out = _mixed()
        s = split(out, ["none", "wave"], gap=2.0, seed=seed)
        c = 40.0  # wave 有 50 段：第 ceil(0.8*50)=40 段的 start
        r1 = lambda names: [n for n in names if n.startswith("R1__")]
        assert all(t >= c for t in _starts(r1(s["val"])))
        assert all(t < c - 2.0 for t in _starts(r1(s["train"])))
        assert any(x.label == "none" and x.recording == "R1" and x.clip in s["val"] for x in out)
        assert not set(s["train"]) & set(s["val"])
        assert any("R1" in w and "丢掉" in w for w in s["warnings"])


def test_two_labels_pinning_same_recording_use_smallest_cut():
    out = []
    for t in range(50):
        out.append(smp("wave", "R1", t, len(out)))
    for t in range(20):
        out.append(smp("cheer", "R1", t, len(out)))
    s = split(out, ["wave", "cheer"], gap=1.0)
    # wave 切点 40，cheer 切点 16：取 16
    assert min(_starts(s["val"])) == 16.0
    assert max(_starts(s["train"])) < 15.0


# ---- 第二部分：训练、导出、报告（假特征提取器：每帧的平均颜色，不下载 DINOv2）----
def fake_extractor():
    torch = pytest.importorskip("torch")

    class FakeExtractor(torch.nn.Module):
        key = "fake"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, frames):  # T×3×S×S → T×3
            self.calls += 1
            return frames.mean(dim=(2, 3))

    return FakeExtractor()


COLORS = {"none": (40, 60, 80), "wave": (200, 160, 110), "bow": (90, 200, 60)}


def write_dataset(root: Path, labels=("none", "wave"), per=25, recs=5, size=112, seed=0):
    rng = np.random.default_rng(seed)
    n = 0
    for label in labels:
        for i in range(per):
            d = root / label / f"rec{i % recs}__{n:04d}_track1_t{i * 3:.2f}s"
            d.mkdir(parents=True)
            base = np.array(COLORS[label], np.int16)
            for f in range(16):
                img = np.clip(base + rng.integers(-20, 21, (size, size, 3)), 0, 255).astype(np.uint8)
                imwrite(d / f"{f:02d}.jpg", img)
            n += 1


def _trained(tmp_path, labels=("none", "wave"), epochs=30):
    from skydango.vision.gesture_train import train

    root = tmp_path / "data"
    write_dataset(root, labels)
    samples, _ = list_samples(root, list(labels))
    s = split(samples, list(labels))
    head, info = train(samples, s, list(labels), fake_extractor(), root / "_features", epochs=epochs, device="cpu")
    return root, samples, s, head, info


def test_train_learns_separable_fake_data(tmp_path):
    _, _, s, head, info = _trained(tmp_path)
    assert s["val"] and s["train"]
    assert info["best_f1"] > 0.9
    assert 1 <= info["best_epoch"] <= len(info["loss"]) <= 30
    assert len(info["val_f1"]) == len(info["loss"])


def test_features_cached(tmp_path):
    from skydango.vision.gesture_train import features_for

    root = tmp_path / "data"
    write_dataset(root, per=1, recs=1)
    samples, _ = list_samples(root, ["none", "wave"])
    ext = fake_extractor()
    cache = tmp_path / "_features"
    a = features_for(samples[0], ext, cache, flip=False)
    assert a.shape == (16, 3) and ext.calls == 1
    assert (cache / "fake-112" / f"{samples[0].clip}.npy").exists()
    b = features_for(samples[0], ext, cache, flip=False)
    assert ext.calls == 1 and (a == b).all()
    features_for(samples[0], ext, cache, flip=True)
    assert ext.calls == 2 and (cache / "fake-112" / f"{samples[0].clip}_flip.npy").exists()


def test_features_cache_follows_size(tmp_path):
    from skydango.vision.gesture_train import features_for, train

    root = tmp_path / "data"
    write_dataset(root)
    samples, _ = list_samples(root, ["none", "wave"])
    s = split(samples, ["none", "wave"])
    cache = root / "_features"
    train(samples, s, ["none", "wave"], fake_extractor(), cache, epochs=1, device="cpu", size=112)
    ext = fake_extractor()
    head, info = train(samples, s, ["none", "wave"], ext, cache, epochs=1, device="cpu", size=56)
    assert ext.calls == 2 * len(s["train"]) + len(s["val"])  # 换了尺寸：一个都不复用 112 的缓存
    assert info["key"] == "fake-56"
    assert (cache / "fake-56").is_dir() and (cache / "fake-112").is_dir()
    seen = []
    ext.forward = lambda frames: (seen.append(tuple(frames.shape)), frames.mean(dim=(2, 3)))[1]
    features_for(samples[0], ext, tmp_path / "fresh", flip=False, size=56)
    assert seen == [(16, 3, 56, 56)]  # 按 size 缩放后再提特征


def test_temporal_head_shape():
    torch = pytest.importorskip("torch")
    from skydango.vision.gesture_train import TemporalHead

    assert tuple(TemporalHead(768, 5)(torch.zeros(2, 16, 768)).shape) == (2, 5)


def test_export_onnx_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from skydango.vision.gesture_train import Exported, export_onnx

    _, _, _, head, _ = _trained(tmp_path, epochs=3)
    ext = fake_extractor()
    path = tmp_path / "g.onnx"
    export_onnx(ext, head, path, frames=16, size=112)
    x = np.random.default_rng(1).random((1, 16, 3, 112, 112), dtype=np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        want = Exported(ext, head).eval()(torch.from_numpy(x)).numpy()
    assert got.shape == (1, 2)
    assert float(np.abs(got - want).max()) < 1e-4


def test_export_follows_config_label_order(tmp_path):
    pytest.importorskip("onnxruntime")
    from skydango.vision.gesture import OnnxGestureClassifier, load_clip
    from skydango.vision.gesture_train import export_onnx, report_md

    labels = ["wave", "none"]  # 和默认反过来
    root, samples, s, head, info = _trained(tmp_path, labels=labels)
    path = tmp_path / "g.onnx"
    export_onnx(fake_extractor(), head, path, frames=16, size=112)
    clf = OnnxGestureClassifier(str(path), labels)
    for label in labels:
        clip = next(x for x in samples if x.label == label)
        assert clf.classify(load_clip(clip.path))[0] == label
    text = report_md(data=root, model=path, labels=labels, names={"wave": "挥手"}, samples=samples, split=s,
                     info=info, evaluation=None, min_prob=0.9)
    assert "类别顺序" in text and "wave, none" in text


def test_evaluate_only_val(tmp_path):
    from skydango.config import GestureConfig
    from skydango.vision.gesture import evaluate

    class Always:
        def classify(self, clip):
            return "wave", 0.99

    root = tmp_path / "data"
    write_dataset(root, per=3, recs=3)
    clips = sorted(p.name for p in (root / "wave").iterdir())
    r = evaluate(root, Always(), GestureConfig(), only={clips[0]})
    assert r["clips"] == 1 and r["wave"]["tp"] == 1 and r["all"]["fp"] == 0
    r = evaluate(root, Always(), GestureConfig())
    assert r["clips"] == 6 and r["all"]["fp"] == 3
    assert {w["clip"] for w in r["wrong"]} == {p.name for p in (root / "none").iterdir()}


def test_cli_refuses_when_class_too_small(tmp_path, monkeypatch):
    from skydango import cli

    root = tmp_path / "data"
    write_dataset(root, labels=("none", "wave"), per=25)
    write_dataset(root, labels=("bow",), per=5, seed=1)
    (tmp_path / "config.toml").write_text('[gesture]\nlabels = ["none", "wave", "bow"]\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(["-c", "config.toml", "perception", "gesture-train", str(root), "--device", "cpu"])
    assert "鞠躬只有 5 段，还差 15 段" in str(e.value)
    assert not (root / "_split.json").exists()


def test_cli_gesture_train_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("onnxruntime")
    from skydango import cli
    from skydango.vision import gesture_train as gt

    root = tmp_path / "data"
    write_dataset(root)
    (tmp_path / "config.toml").write_text('[gesture]\nlabels = ["none", "wave"]\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gt, "DinoExtractor", lambda device: fake_extractor())
    cli.main(["-c", "config.toml", "perception", "gesture-train", str(root), "--device", "cpu", "--epochs", "20",
              "--out", "models/g.onnx"])
    assert (tmp_path / "models" / "g.onnx").exists()
    assert not (tmp_path / "models" / "gesture.onnx").exists()
    assert gt.load_split(root)["val"]
    reports = list((tmp_path / "tmp" / "gesture-train").glob("*/report.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "类别顺序" in text and "none, wave" in text and "验证集" in text
    assert "报告" in capsys.readouterr().out

    # gesture-eval：有 _split.json 默认只评验证集，--all 评全部
    cli.main(["-c", "config.toml", "perception", "gesture-eval", str(root), "--model", "models/g.onnx"])
    val_out = capsys.readouterr().out
    assert f"{len(gt.load_split(root)['val'])} 段" in val_out and "只评验证集" in val_out
    cli.main(["-c", "config.toml", "perception", "gesture-eval", str(root), "--model", "models/g.onnx", "--all"])
    assert "50 段" in capsys.readouterr().out


def test_default_out_does_not_touch_gesture_onnx(tmp_path):
    from skydango.vision import gesture_train as gt

    when = dt.datetime(2026, 10, 1, 21, 5, 7)
    assert gt.default_out(tmp_path / "models", when) == tmp_path / "models" / "gesture-20261001.onnx"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "gesture-20261001.onnx").write_bytes(b"x")
    assert gt.default_out(tmp_path / "models", when).name == "gesture-20261001-210507.onnx"
