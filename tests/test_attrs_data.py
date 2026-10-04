import json
from datetime import datetime

import numpy as np
import pytest

from skydango.config import AssistConfig
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
    assert ad.writeback(root, out, datetime(2026, 10, 2, 11, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
    assert not (root / "_backup").exists()
    # 数据集来源的挪进 form/morph 才算确认；难例挪进去也不写
    _move(out, new, "morph")
    _move(out, hard_row, "shared")
    label = root / "labels/train/rec1_0001_1.00s.txt"
    before = label.read_text(encoding="utf-8")
    assert ad.writeback(root, out, datetime(2026, 10, 2, 12, 0, 0)) == {"frames": 1, "boxes": 1, "relabeled": 0, "removed": 0}
    backup = root / "_backup" / "labels-20261002-120000" / "train" / "rec1_0001_1.00s.txt"
    assert backup.read_text(encoding="utf-8") == before
    lines = label.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and lines[-1].split()[0] == "0"
    cx, cy, w, h = map(float, lines[-1].split()[1:])
    assert (round(cx * 200), round(cy * 100), round(w * 200), round(h * 100)) == (160, 25, 20, 40)
    # 再写一次：同帧已有 IoU ≥ 0.5 的框，不重复
    assert ad.writeback(root, out, datetime(2026, 10, 2, 12, 5, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
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
    assert ad.writeback(root, out, datetime(2026, 10, 2, 1, 2, 3)) == {"frames": 1, "boxes": 3, "relabeled": 0, "removed": 0}
    got = [ln.split()[0] for ln in (root / "labels/val/0002_2.00s.txt").read_text(encoding="utf-8").splitlines()]
    assert got == ["4", "4", "9", "0"]


def _confirm(out, row, to, src=None):
    """照标注页：从 src（默认 form/<原外形>）挪到 to（同一类 = 原地确认），记进 _labels.jsonl。"""
    src = src or {"player": "lit", "player_unlit": "unlit", "spirit": "spirit"}[row["yolo_cls"]]
    sdir = out / "form" / src
    ddir = out / ("_discard" if to == "_discard" else f"form/{to}")
    if src != to:
        ddir.mkdir(parents=True, exist_ok=True)
        (sdir / row["crop"]).rename(ddir / row["crop"])
    with (out / "_labels.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"crop": row["crop"], "from": src, "to": to}) + "\n")


def test_writeback_relabels_and_removes_confirmed_known_boxes(tmp_path):
    # 10-04：datasets/sky 的人物标注点没点火标反、把椅子标成人的都有；标注页确认过的改回去，增强图的标注副本一起改
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    (root / "labels/train/rec1_0001_1.00s_dark.txt").write_text(
        "0 0.25 0.5 0.1 0.4\n3 0.5 0.5 0.1 0.4\n9 0.75 0.5 0.1 0.4\n", encoding="utf-8")
    (root / "labels/train/rec1_0001_1.00s_blur.txt").write_text(
        "0 0.25 0.5 0.1 0.4\n3 0.5 0.5 0.1 0.4\n9 0.75 0.5 0.1 0.4\n", encoding="utf-8")
    ad.crops_from_dataset(root, FakeDetector({}), out, conf=0.2)
    known = {r["yolo_cls"]: r for r in _rows(out) if r["known"] and r["split"] == "train"}
    val = next(r for r in _rows(out) if r["known"] and r["split"] == "val")
    assert set(known) == {"player", "spirit"}
    # 没人确认的不动
    assert ad.writeback(root, out, datetime(2026, 10, 4, 1, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
    _confirm(out, known["player"], "unlit")  # lit → 黑影
    _confirm(out, known["spirit"], "not_person")  # 先祖 → 不是人：删掉
    _confirm(out, val, "unlit")  # 原地确认、类别没变：不动
    res = ad.writeback(root, out, datetime(2026, 10, 4, 2, 0, 0))
    assert res == {"frames": 1, "boxes": 0, "relabeled": 1, "removed": 1}
    for stem in ("rec1_0001_1.00s", "rec1_0001_1.00s_dark", "rec1_0001_1.00s_blur"):
        got = (root / f"labels/train/{stem}.txt").read_text(encoding="utf-8").splitlines()
        assert [ln.split()[0] for ln in got] == ["4", "3"], stem  # self 行不碰
    assert (root / "labels/val/0002_2.00s.txt").read_text(encoding="utf-8") == "4 0.5 0.5 0.1 0.4\n"
    assert (root / "_backup" / "labels-20261004-020000" / "train" / "rec1_0001_1.00s_dark.txt").is_file()
    # 再写一次：已经改过了，什么都不做，也不再多一份备份
    assert ad.writeback(root, out, datetime(2026, 10, 4, 3, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
    assert not (root / "_backup" / "labels-20261004-030000").exists()


def test_writeback_discard_and_undone_keep_box(tmp_path):
    # 「不要」= 这张图看不清，不等于不是人：框留着；撤销掉的确认不算
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    ad.crops_from_dataset(root, FakeDetector({}), out, conf=0.2)
    known = {r["yolo_cls"]: r for r in _rows(out) if r["known"] and r["split"] == "train"}
    _confirm(out, known["player"], "_discard")
    _confirm(out, known["spirit"], "unlit")
    with (out / "_labels.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"crop": known["spirit"]["crop"], "from": "unlit", "to": "spirit", "undo": True}) + "\n")
    (out / "form/unlit" / known["spirit"]["crop"]).rename(out / "form/spirit" / known["spirit"]["crop"])
    assert ad.writeback(root, out, datetime(2026, 10, 4, 4, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}


def test_writeback_adds_to_augmented_copies(tmp_path):
    # 补漏标的人：增强图（同一帧的模糊 / 压暗副本）的标注一起补
    root = _dataset(tmp_path)
    out = tmp_path / "attrs"
    ad.crops_from_dataset(root, FakeDetector({1: [_det("player", 150, 5, 20, 40)]}), out, conf=0.2)
    _move(out, next(r for r in _rows(out) if not r["known"]), "lit")
    res = ad.writeback(root, out, datetime(2026, 10, 4, 5, 0, 0))
    assert res["boxes"] == 1 and res["frames"] == 1
    blur = (root / "labels/train/rec1_0001_1.00s_blur.txt").read_text(encoding="utf-8").splitlines()
    assert len(blur) == 2 and blur[-1].split()[0] == "0"


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
    assert ad.writeback(other, out, datetime(2026, 10, 2, 9, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}
    assert not (other / "_backup").exists() and not list((other / "labels" / "train").iterdir())
    # 旧记录（没有 dataset）跳过
    (out / "_crops.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "dataset"}) + "\n" for r in rows), encoding="utf-8")
    assert ad.writeback(root, out, datetime(2026, 10, 2, 9, 0, 0)) == {"frames": 0, "boxes": 0, "relabeled": 0, "removed": 0}


# ---- Claude 初分 ----

def _fi(names):
    crops = [np.full((224, 224, 3), 10 * i, np.uint8) for i in range(len(names))]
    return ad.make_sheet_input(names, crops)


def test_form_sheet_size_and_numbers():
    sheet = ad.form_sheet([np.full((224, 224, 3), 100, np.uint8)] * 16)
    assert sheet.shape == (640, 640, 3)
    for i in range(16):  # 每格左上角白底（编号底），格子中间是原色
        y, x = (i // 4) * 160, (i % 4) * 160
        assert sheet[y + 1, x + 1].tolist() == [255, 255, 255]
        assert sheet[y + 100, x + 100].tolist() == [100, 100, 100]
    partial = ad.form_sheet([np.full((50, 50, 3), 100, np.uint8)] * 3)  # 不足 16 张、尺寸不对也行
    assert partial.shape == (640, 640, 3) and partial[100, 260].tolist() == [100, 100, 100]
    assert partial[500, 500].tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        ad.form_sheet([])
    with pytest.raises(ValueError):
        ad.form_sheet([np.zeros((9, 9, 3), np.uint8)] * 17)


def test_sheet_stem_stable():
    assert ad.sheet_stem(["a.jpg", "b.jpg"]) == ad.sheet_stem(["a.jpg", "b.jpg"])
    assert ad.sheet_stem(["a.jpg", "b.jpg"]) != ad.sheet_stem(["b.jpg", "a.jpg"])


def test_build_form_message():
    f = _fi(["a.jpg", "b.jpg"])
    content = ad.build_form_message([f], AssistConfig())
    assert [c["type"] for c in content] == ["text", "image"]
    assert "0=a.jpg" in content[0]["text"] and "1=b.jpg" in content[0]["text"] and f.stem in content[0]["text"]
    assert "not_person" in ad.FORM_SYSTEM and "morph" in ad.FORM_SYSTEM and ad.FORM_PROTOCOL.system == ad.FORM_SYSTEM


def test_parse_form_review():
    f = _fi(["a.jpg", "b.jpg", "c.jpg"])
    text = '好的：\n{"a.jpg": {"label": "lit", "confidence": 0.9, "reason": "x"}, "b.jpg": {"label": "dragon"}, "zzz.jpg": {"label": "lit"}}\n完'
    got = ad.parse_form_review(text, [f])
    assert list(got) == ["a.jpg"]  # 未知类别 / 对不上的名字 / 缺项（c）都不在
    assert got["a.jpg"].label == "lit" and got["a.jpg"].confidence == 0.9
    assert ad.parse_form_review('{"c.jpg": {"label": "unsure", "confidence": 5}}', [f])["c.jpg"].confidence == 1.0
    assert ad.parse_form_review("没有 json", [f]) == {}
    assert ad.FORM_PROTOCOL.parse(text, [f]) == {f.stem: got}


def test_form_protocol_through_reviewer(tmp_path):
    from skydango.vision import assist

    f = _fi(["a.jpg", "b.jpg"])
    seen = []

    def run(content):
        seen.append(content)
        return {"result": '{"a.jpg": {"label": "spirit", "confidence": 0.7}, "b.jpg": {"label": "lit"}}', "usage": {"input_tokens": 3}}

    r = assist.Reviewer(run, tmp_path / "cache", AssistConfig(), "attrs", protocol=ad.FORM_PROTOCOL)
    out = r.review([f])
    assert out[f.stem]["a.jpg"].label == "spirit" and out[f.stem]["b.jpg"].label == "lit"
    assert len(seen) == 1 and r.usage["input_tokens"] == 3


def test_form_guess_file_skip_and_merge(tmp_path):
    unl = tmp_path / "_unlabeled"
    unl.mkdir()
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        imwrite(unl / n, np.zeros((8, 8, 3), np.uint8))
    assert ad.pending_crops(unl, ad.load_form_guesses(unl)) == ["a.jpg", "b.jpg", "c.jpg"]
    ad.write_form_guesses(unl, {"a.jpg": ad.Guess("lit", 0.9, "r")}, "sonnet")
    ex = ad.load_form_guesses(unl)
    assert ex["a.jpg"] == {"label": "lit", "confidence": 0.9, "reason": "r", "model": "sonnet", "version": ad.FORM_PROMPT_VERSION}
    assert ad.pending_crops(unl, ex) == ["b.jpg", "c.jpg"]
    assert ad.pending_crops(unl, ex, recheck=True) == ["a.jpg", "b.jpg", "c.jpg"]
    ex["b.jpg"] = {"label": "lit", "version": ad.FORM_PROMPT_VERSION - 1}  # 旧版提示词的不算
    assert ad.pending_crops(unl, ex) == ["b.jpg", "c.jpg"]
    ad.write_form_guesses(unl, {"b.jpg": ad.Guess("unsure", 0.0, "")}, "sonnet")  # 合并：a 还在
    assert set(ad.load_form_guesses(unl)) == {"a.jpg", "b.jpg"}
    assert ad.pending_crops(tmp_path / "nope", {}) == []
    (unl / "claude.json").write_text("坏的", encoding="utf-8")
    assert ad.load_form_guesses(unl) == {}
