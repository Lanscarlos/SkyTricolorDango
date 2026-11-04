from skydango.vision.cropgroups import frame_time, groups


def row(crop, image, box, split="train", source="dataset", dataset="D:/x/datasets/sky", group=None):
    return {"crop": crop, "image": image, "box": box, "split": split, "source": source, "dataset": dataset, "group": group}


def seq(n, prefix="rec1", step=1.5, x0=100, tag="", **kw):
    return [row(f"{prefix}{tag}c{i}", f"D:/x/{prefix}_{i:04d}_{i * step:06.2f}s.jpg", [x0 + i * 5, 100, 50, 150], **kw) for i in range(n)]


def test_frame_time():
    assert frame_time("20260929-223311-s3_0351_175.50s") == 175.5
    assert frame_time("abc") is None


def test_groups_chain_consecutive_frames():
    assert groups(seq(5)) == [[f"rec1c{i}" for i in range(5)]]


def test_groups_break_on_gap():
    rows = seq(3)
    rows += [row(f"late{i}", f"D:/x/rec1_{10 + i:04d}_{8.0 + i * 1.5:06.2f}s.jpg", [110 + i * 5, 100, 50, 150]) for i in range(3)]
    assert groups(rows) == [["rec1c0", "rec1c1", "rec1c2"], ["late0", "late1", "late2"]]


def test_groups_different_recordings_or_split_not_merged():
    rows = seq(2, "recA") + seq(2, "recB") + seq(2, "recA", split="val", tag="v")
    assert sorted(len(g) for g in groups(rows)) == [2, 2, 2]


def test_groups_no_time_singletons():
    rows = [row(f"c{i}", f"D:/x/frame{i}.jpg", [100, 100, 50, 150]) for i in range(3)]
    assert sorted(groups(rows)) == [["c0"], ["c1"], ["c2"]]


def test_groups_cap_splits():
    assert [len(g) for g in groups(seq(35, step=0.5))] == [30, 5]


def test_groups_side_by_side_not_merged():
    rows = []
    for i in range(5):
        img = f"D:/x/rec1_{i:04d}_{i * 1.0:06.2f}s.jpg"
        rows.append(row(f"L{i}", img, [100 + i * 3, 100, 50, 150]))
        rows.append(row(f"R{i}", img, [600 + i * 3, 100, 50, 150]))
    assert sorted(groups(rows)) == [[f"L{i}" for i in range(5)], [f"R{i}" for i in range(5)]]


def test_groups_bad_rows_are_singletons():
    rows = [
        {"crop": "noimg", "box": [1, 2, 3, 4]},
        {"crop": "nobox", "image": "D:/x/rec1_0001_1.00s.jpg"},
        {"crop": "badbox", "image": "D:/x/rec1_0002_2.00s.jpg", "box": [1, 2, 3]},
        {"crop": "strbox", "image": "D:/x/rec1_0003_3.00s.jpg", "box": ["a", 2, 3, 4]},
    ]
    assert sorted(groups(rows)) == [["badbox"], ["nobox"], ["noimg"], ["strbox"]]


def test_groups_inbox_by_run():
    a = seq(3, "r1", source="inbox", group="runA", tag="a")
    b = seq(3, "r1", source="inbox", group="runB", tag="b")  # 时间和位置都接得上，但运行不同
    assert sorted(len(g) for g in groups(a + b)) == [3, 3]
