from pathlib import Path

import numpy as np
import pytest

from skydango.imageio import imread, imwrite
from skydango.vision.augment import augment_dataset, darken, motion_blur


def test_motion_blur_is_horizontal():
    vertical = np.zeros((40, 40, 3), np.uint8)
    vertical[:, ::2] = 255  # 竖条纹：横着糊会变平
    horizontal = np.zeros((40, 40, 3), np.uint8)
    horizontal[::2] = 255  # 横条纹：横着糊不变
    assert motion_blur(vertical, 9)[:, 5:-5].std() < 40
    assert np.array_equal(motion_blur(horizontal, 9), horizontal)


def test_darken_lowers_brightness():
    gray = np.full((10, 10, 3), 128, np.uint8)
    assert darken(gray, 2.0).mean() < 128


def make_dataset(root: Path, train: int = 20, val: int = 2) -> None:
    for split, n in (("train", train), ("val", val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            imwrite(root / "images" / split / f"{split}{i:02d}.jpg", np.full((36, 64, 3), 100 + i, np.uint8))
            (root / "labels" / split / f"{split}{i:02d}.txt").write_text(f"0 0.5 0.5 0.1 0.{i % 9 + 1}\n", encoding="utf-8")


def test_augment_only_train_copies_labels_and_is_idempotent(tmp_path):
    make_dataset(tmp_path)
    counts = augment_dataset(tmp_path, blur=1.0, dark=1.0)
    assert counts["blur"] == 20 and counts["dark"] == 20
    train = tmp_path / "images" / "train"
    assert len(list(train.glob("*_blur.jpg"))) == 20 and len(list(train.glob("*_dark.jpg"))) == 20
    labels = tmp_path / "labels" / "train"
    assert (labels / "train03_blur.txt").read_text(encoding="utf-8") == (labels / "train03.txt").read_text(encoding="utf-8")
    assert imread(train / "train03_dark.jpg").mean() < imread(train / "train03.jpg").mean()
    assert len(list((tmp_path / "images" / "val").iterdir())) == 2
    again = augment_dataset(tmp_path, blur=1.0, dark=1.0)
    assert again["blur"] == 0 and again["dark"] == 0  # 已有的不再生成，_blur / _dark 本身也不再增强


def test_augment_same_seed_same_picks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_dataset(a)
    make_dataset(b)
    augment_dataset(a, seed=7, blur=0.3, dark=0.2)
    augment_dataset(b, seed=7, blur=0.3, dark=0.2)
    names = lambda root: sorted(p.name for p in (root / "images" / "train").iterdir())  # noqa: E731
    assert names(a) == names(b) and len(names(a)) > 20


def test_augment_without_train_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        augment_dataset(tmp_path)
