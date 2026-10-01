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
