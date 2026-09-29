"""perception label 写数据集：录像帧的命名（带录像目录名）、同名不同图不覆盖。不连 Claude、不要 GPU。"""

from pathlib import Path

import cv2
import numpy as np

from skydango.imageio import imread, imwrite
from skydango.vision.weaklabel import dataset_clash, label_items, write_sample


def gradient(shift: int = 0) -> np.ndarray:
    """平滑的渐变图：JPEG 重新压缩后和原图差得很少，像真实画面。"""
    x = np.linspace(0, 200, 160, dtype=np.float32)
    y = np.linspace(0, 50, 90, dtype=np.float32)
    g = (x[None, :] + y[:, None] + shift).clip(0, 255).astype(np.uint8)
    return np.dstack([g, np.roll(g, 7, axis=1), 255 - g])


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


# ---- 命名 ----
def test_single_recording_dir_gets_its_name_as_prefix(tmp_path):
    rec = tmp_path / "20260929-211807"
    files = [touch(rec / "0000_000.00s.jpg"), touch(rec / "0001_000.50s.jpg")]
    items = label_items(rec, files)
    assert [stem for _, stem in items] == ["20260929-211807_0000_000.00s", "20260929-211807_0001_000.50s"]
    assert [p for p, _ in items] == files


def test_parent_dir_naming_unchanged(tmp_path):
    # 对 tmp/record 跑：已有数据集里的名字是 "<录像>_<文件>"，不能变
    root = tmp_path / "record"
    files = [touch(root / "20260928-214948" / "0000_000.00s.jpg"), touch(root / "20260929-211807" / "0003_001.50s.jpg")]
    assert [stem for _, stem in label_items(root, files)] == ["20260928-214948_0000_000.00s", "20260929-211807_0003_001.50s"]


def test_single_image_keeps_its_own_name(tmp_path):
    img = touch(tmp_path / "rec" / "0000_000.00s.jpg")
    assert label_items(img, [img]) == [(img, "0000_000.00s")]


# ---- 防覆盖 ----
def test_same_name_different_image_is_a_clash(tmp_path):
    old = gradient()
    imwrite(tmp_path / "images" / "val" / "0000_000.00s.jpg", old)
    assert dataset_clash(tmp_path, "0000_000.00s", 255 - old) == tmp_path / "images" / "val" / "0000_000.00s.jpg"


def test_same_image_after_jpeg_recompression_is_not_a_clash(tmp_path):
    frame = gradient()
    imwrite(tmp_path / "images" / "train" / "a.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 重新压缩过，像素有细小差别
    assert dataset_clash(tmp_path, "a", frame) is None
    assert dataset_clash(tmp_path, "b", 255 - frame) is None  # 没有同名的


def test_write_sample_skips_clash_and_keeps_old_files(tmp_path):
    old = gradient()
    assert write_sample(tmp_path, "train", "a", old, ["0 0.5 0.5 0.1 0.1"]) is None
    img, label = tmp_path / "images" / "train" / "a.jpg", tmp_path / "labels" / "train" / "a.txt"
    before = img.read_bytes()
    clash = write_sample(tmp_path, "train", "a", 255 - old, ["1 0.2 0.2 0.1 0.1"])
    assert clash == img
    assert img.read_bytes() == before
    assert label.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"


def test_write_sample_same_image_overwrites_labels(tmp_path):
    frame = gradient()
    write_sample(tmp_path, "val", "a", frame, ["0 0.5 0.5 0.1 0.1"])
    assert write_sample(tmp_path, "val", "a", frame, ["2 0.3 0.3 0.1 0.1", "0 0.5 0.5 0.1 0.1"]) is None
    assert (tmp_path / "labels" / "val" / "a.txt").read_text(encoding="utf-8") == "2 0.3 0.3 0.1 0.1\n0 0.5 0.5 0.1 0.1\n"
    assert np.abs(imread(tmp_path / "images" / "val" / "a.jpg").astype(int) - frame.astype(int)).mean() < 2


def test_write_sample_empty_labels_writes_empty_file(tmp_path):
    write_sample(tmp_path, "train", "a", gradient(), [])
    assert (tmp_path / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == ""


# ---- perception label（弱标注路径）：OCR / 图标换成空的，只看命名和写盘 ----
def run_label(monkeypatch, source: Path, out: Path) -> None:
    import argparse

    import skydango.vision.ocr as ocr
    from skydango import cli
    from skydango.config import Config

    monkeypatch.setattr(ocr, "make_ocr", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    monkeypatch.setattr(cli, "_weak_boxes", lambda *a, **k: [])
    args = argparse.Namespace(source=str(source), output=str(out), model=None, from_runs=False, spin=False, val=0.15,
                              all_text=False, min_score=0.9, preview=False, assist=False, all_frames=False,
                              objects=False, recheck=False)
    cli._perception_label(Config(), args)


def stems(out: Path) -> list[str]:
    return sorted(p.stem for p in (out / "images").rglob("*.jpg"))


def test_label_two_recordings_with_same_file_names_both_kept(tmp_path, monkeypatch):
    out = tmp_path / "ds"
    imwrite(tmp_path / "rec" / "20260928-214948" / "0000_000.00s.jpg", gradient())
    imwrite(tmp_path / "rec" / "20260929-211807" / "0000_000.00s.jpg", 255 - gradient())
    run_label(monkeypatch, tmp_path / "rec" / "20260928-214948", out)
    run_label(monkeypatch, tmp_path / "rec" / "20260929-211807", out)
    assert stems(out) == ["20260928-214948_0000_000.00s", "20260929-211807_0000_000.00s"]


def test_label_skips_clash_and_reports(tmp_path, monkeypatch, capsys):
    out = tmp_path / "ds"
    rec = tmp_path / "rec" / "20260929-211807"
    imwrite(rec / "0000_000.00s.jpg", gradient())
    imwrite(rec / "0001_000.50s.jpg", gradient(30))
    run_label(monkeypatch, rec, out)
    (clash,) = [p for p in (out / "images").rglob("*.jpg") if p.stem.endswith("0000_000.00s")]
    label = out / "labels" / clash.parent.name / f"{clash.stem}.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")  # 人工修过的标注
    before = clash.read_bytes()
    imwrite(rec / "0000_000.00s.jpg", 255 - gradient())  # 同名换成另一张图
    capsys.readouterr()
    run_label(monkeypatch, rec, out)
    printed = capsys.readouterr().out
    assert clash.read_bytes() == before
    assert label.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert "20260929-211807_0000_000.00s" in printed and "数据集里已有同名的另一张图" in printed
