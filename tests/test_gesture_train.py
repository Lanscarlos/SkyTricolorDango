from pathlib import Path

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
