import json
from datetime import datetime

import numpy as np

from skydango.imageio import imread, imwrite
from skydango.vision import attrs_data as ad
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection


class FakeDetector:
    def __init__(self, by_tag):
        self.by_tag = {k * 40: v for k, v in by_tag.items()}  # 帧标记（画面左上角的蓝色值 = 标记 × 40） -> [Detection]

    def detect(self, img):
        # JPEG 有损：误差在 10 以内算同一个标记
        v = int(img[0, 0, 0])
        return next((list(d) for t, d in self.by_tag.items() if abs(t - v) < 10), [])


def _frame(path, tag, w=200, h=100):
    img = np.full((h, w, 3), 90, np.uint8)
    img[0:8, 0:8, 0] = tag * 40  # 用第一个像素告诉假检测器是哪一帧
    imwrite(path, img, [1, 100])


def _dataset(tmp_path):
    root = tmp_path / "sky"
    for split, stem, tag in (("train", "rec1_0001_1.00s", 1), ("val", "0002_2.00s", 2), ("train", "rec1_0001_1.00s_blur", 3)):
        _frame(root / "images" / split / f"{stem}.jpg", tag)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    # 帧 1：一个 lit（0）、一个 self（3）、一个 spirit（9）
    (root / "labels/train/rec1_0001_1.00s.txt").write_text(
        "0 0.25 0.5 0.1 0.4\n3 0.5 0.5 0.1 0.4\n9 0.75 0.5 0.1 0.4\n", encoding="utf-8")
    (root / "labels/val/0002_2.00s.txt").write_text("4 0.5 0.5 0.1 0.4\n", encoding="utf-8")
    (root / "labels/train/rec1_0001_1.00s_blur.txt").write_text("0 0.5 0.5 0.1 0.4\n", encoding="utf-8")
    return root


def _det(cls, x, y, w, h, s=0.9):
    return Detection(cls, Rect(x, y, w, h), s)


def _rows(out):
    return [json.loads(line) for line in (out / "_crops.jsonl").read_text(encoding="utf-8").splitlines()]


def _move(out, row, form):
    (out / "form" / form).mkdir(parents=True, exist_ok=True)
    (out / "_unlabeled" / row["crop"]).rename(out / "form" / form / row["crop"])


def test_crop_name():
    assert ad.crop_name("a_0001_1.00s", Rect(1, 2, 3, 4)) == "a_0001_1.00s__1_2_3_4.jpg"


def test_dataset_known_unlabeled_and_idempotent(tmp_path):
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    # 帧 1：检测器重复认出 lit 那个人（IoU 高，不进 _unlabeled）+ 一个标注里没有的人 + 低分框 + 非人物；帧 2 没有检测
    det = FakeDetector({1: [_det("player", 40, 30, 20, 40), _det("player", 150, 5, 20, 40), _det("name_tag", 0, 0, 5, 5),
                            _det("player", 100, 5, 20, 40, 0.1)]})
    counts = ad.crops_from_dataset(root, det, out, conf=0.2)
    assert counts["lit"] == 1 and counts["spirit"] == 1 and counts["unlit"] == 1
    assert counts["_unlabeled"] == 1
    assert len(list((out / "form/lit").glob("*.jpg"))) == 1
    assert len(list((out / "form/unlit").glob("*.jpg"))) == 1
    assert not (out / "form/self").exists()
    rows = _rows(out)
    assert len(rows) == 4  # 增强图被跳过、self 不要
    lit = next(r for r in rows if r["yolo_cls"] == "player" and r["known"])
    assert lit["source"] == "dataset" and lit["split"] == "train" and lit["group"] == "rec1"
    assert imread(out / "form/lit" / lit["crop"]).shape == (224, 224, 3)
    unl = next(r for r in rows if not r["known"])
    assert unl["crop"].startswith("rec1_0001_1.00s__150_5_") and (out / "_unlabeled" / unl["crop"]).exists()
    assert next(r for r in rows if r["split"] == "val")["group"] == ""  # 旧格式没有录像前缀
    # 重跑：不重复，也不把已经被挪走的裁图再造一份
    _move(out, unl, "lit")
    again = ad.crops_from_dataset(root, det, out, conf=0.2)
    assert len(_rows(out)) == 4 and sum(again.get(k, 0) for k in ("lit", "unlit", "spirit", "_unlabeled")) == 0
    assert not (out / "_unlabeled" / unl["crop"]).exists()


def test_images_dedup_and_group(tmp_path):
    folder = tmp_path / "runs" / "r1" / "hard"
    _frame(folder / "f1.jpg", 1)
    out = tmp_path / "attrs"
    det = FakeDetector({1: [_det("player", 10, 10, 40, 60, 0.5), _det("player_unlit", 12, 12, 40, 60, 0.9),
                            _det("player", 120, 10, 40, 60, 0.8)]})
    counts = ad.crops_from_images(folder, det, out, conf=0.2)
    assert counts["_unlabeled"] == 2
    rows = _rows(out)
    assert {r["yolo_cls"] for r in rows} == {"player_unlit", "player"}
    assert all(r["group"] == "r1" and r["source"] == "images" and r["split"] is None and r["known"] is False for r in rows)
    assert ad.crops_from_images(folder, det, out, conf=0.2)["_unlabeled"] == 0
    assert len(_rows(out)) == 2
    other = tmp_path / "rec2"
    _frame(other / "f1.jpg", 1)
    ad.crops_from_images(other, det, out, conf=0.2)
    assert {r["group"] for r in _rows(out)} == {"r1", "rec2"}


def test_writeback(tmp_path):
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    det = FakeDetector({1: [_det("player", 150, 5, 20, 40), _det("player", 40, 30, 20, 40)]})
    ad.crops_from_dataset(root, det, out, conf=0.2)
    hard = tmp_path / "hard"
    _frame(hard / "h.jpg", 1)
    ad.crops_from_images(hard, FakeDetector({1: [_det("player", 5, 5, 20, 40)]}), out, conf=0.2)
    rows = _rows(out)
    new = next(r for r in rows if r["source"] == "dataset" and not r["known"])
    hard_row = next(r for r in rows if r["source"] == "images")
    # 还在 _unlabeled：不写，也不备份
    assert ad.writeback(root, out, datetime(2026, 10, 2, 11, 0, 0)) == {"frames": 0, "boxes": 0}
    assert not (root / "_backup").exists()
    # 数据集来源的挪进 form/morph 才算确认；难例挪进去也不写
    _move(out, new, "morph")
    _move(out, hard_row, "shared")
    label = root / "labels/train/rec1_0001_1.00s.txt"
    before = label.read_text(encoding="utf-8")
    assert ad.writeback(root, out, datetime(2026, 10, 2, 12, 0, 0)) == {"frames": 1, "boxes": 1}
    backup = root / "_backup" / "labels-20261002-120000" / "train" / "rec1_0001_1.00s.txt"
    assert backup.read_text(encoding="utf-8") == before
    lines = label.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and lines[-1].split()[0] == "0"
    cx, cy, w, h = map(float, lines[-1].split()[1:])
    assert (round(cx * 200), round(cy * 100), round(w * 200), round(h * 100)) == (160, 25, 20, 40)
    # 再写一次：同帧已有 IoU ≥ 0.5 的框，不重复
    assert ad.writeback(root, out, datetime(2026, 10, 2, 12, 5, 0)) == {"frames": 0, "boxes": 0}
    assert len(label.read_text(encoding="utf-8").splitlines()) == 4


def test_writeback_class_map(tmp_path):
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    det = FakeDetector({2: [_det("player", 10, 10, 20, 40), _det("player", 60, 10, 20, 40), _det("player", 150, 10, 20, 40)]})
    ad.crops_from_dataset(root, det, out, conf=0.2)
    rows = [r for r in _rows(out) if r["split"] == "val" and not r["known"]]
    assert len(rows) == 3
    for r, form in zip(rows, ("unlit", "spirit", "lit")):
        _move(out, r, form)
    assert ad.writeback(root, out, datetime(2026, 10, 2, 1, 2, 3)) == {"frames": 1, "boxes": 3}
    got = [ln.split()[0] for ln in (root / "labels/val/0002_2.00s.txt").read_text(encoding="utf-8").splitlines()]
    assert got == ["4", "4", "9", "0"]


def test_rows_absolute_paths_and_other_dataset_not_written(tmp_path):
    from pathlib import Path

    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    ad.crops_from_dataset(root, FakeDetector({1: [_det("player", 150, 5, 20, 40)]}), out, conf=0.2)
    rows = _rows(out)
    assert all(Path(r["image"]).is_absolute() and Path(r["image"]).is_file() for r in rows)
    assert all(r["dataset"] == root.resolve().as_posix() for r in rows)
    _move(out, next(r for r in rows if not r["known"]), "lit")
    other = tmp_path / "other"
    (other / "labels" / "train").mkdir(parents=True)
    (other / "images" / "train").mkdir(parents=True)
    assert ad.writeback(other, out, datetime(2026, 10, 2, 9, 0, 0)) == {"frames": 0, "boxes": 0}
    assert not (other / "_backup").exists() and not list((other / "labels" / "train").iterdir())
    # 旧记录（没有 dataset）跳过
    (out / "_crops.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "dataset"}) + "\n" for r in rows), encoding="utf-8")
    assert ad.writeback(root, out, datetime(2026, 10, 2, 9, 0, 0)) == {"frames": 0, "boxes": 0}
