"""训练用的合并视图：三类人合成一类，不动原数据集。"""

import os

from skydango.vision.merged import build


def _dataset(tmp_path):
    src = tmp_path / "sky"
    for split, stem in (("train", "a"), ("val", "b_blur")):
        (src / "images" / split).mkdir(parents=True, exist_ok=True)
        (src / "labels" / split).mkdir(parents=True, exist_ok=True)
        (src / "images" / split / f"{stem}.jpg").write_bytes(b"img-" + stem.encode())
    (src / "labels/train/a.txt").write_text("4 0.5 0.5 0.1 0.2\n0 0.2 0.2 0.1 0.1\n9 0.7 0.7 0.1 0.3\n3 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    (src / "labels/val/b_blur.txt").write_text("9 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    (src / "labels/train.cache").write_bytes(b"stale")
    return src


def test_build_merged(tmp_path):
    src = _dataset(tmp_path)
    before = (src / "labels/train/a.txt").read_bytes()
    yaml = build(src, tmp_path / "m")
    m = tmp_path / "m"
    assert [ln.split()[0] for ln in (m / "labels/train/a.txt").read_text(encoding="utf-8").splitlines()] == ["0", "0", "0", "3"]
    assert (m / "images/val/b_blur.jpg").read_bytes() == b"img-b_blur" and (m / "labels/val/b_blur.txt").is_file()
    assert not (m / "labels/train.cache").exists()
    text = yaml.read_text(encoding="utf-8")
    assert yaml == m / "data.yaml" and "0: person" in text and "4: unused_4" in text and "9: unused_9" in text
    assert "train: images/train" in text and "val: images/val" in text
    assert (src / "labels/train/a.txt").read_bytes() == before  # 原数据集没动


def test_build_keeps_odd_lines(tmp_path):
    src = _dataset(tmp_path)
    (src / "labels/train/a.txt").write_text("# 注释\n\n4 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    build(src, tmp_path / "m")
    assert (tmp_path / "m/labels/train/a.txt").read_text(encoding="utf-8").split("\n")[:3] == ["# 注释", "", "0 0.5 0.5 0.1 0.2"]


def test_build_merged_overwrites(tmp_path):
    src = _dataset(tmp_path)
    out = tmp_path / "m"
    (out / "labels/train").mkdir(parents=True)
    (out / "labels/train/stale.txt").write_text("4 0 0 0 0\n", encoding="utf-8")
    build(src, out)
    assert not (out / "labels/train/stale.txt").exists() and (out / "labels/train/a.txt").is_file()


def test_build_falls_back_to_copy_when_link_fails(tmp_path, monkeypatch):
    src = _dataset(tmp_path)

    def boom(*a, **k):
        raise OSError("cross-device")

    monkeypatch.setattr(os, "link", boom)
    build(src, tmp_path / "m")
    assert (tmp_path / "m/images/train/a.jpg").read_bytes() == b"img-a"
