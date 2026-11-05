import json

import numpy as np

from skydango.imageio import imwrite
from skydango.vision.bubbles import Rect
from skydango.vision.crops_audit import audit, write_focus, write_scores
from skydango.vision.detect import Detection

W, H = 400, 300


class FakeDetector:
    """按帧给事先定好的框：图是纯色，帧号编码在左上角像素的蓝通道（帧号 × 10）。"""

    def __init__(self, by_frame: dict[int, list[tuple[str, Rect, float]]]):
        self.by_frame = by_frame

    def detect(self, img):
        return [Detection(c, b, s) for c, b, s in self.by_frame.get(int(img[0, 0, 0]), [])]


class FakeHead:
    """外形头：每个框按左上角 x 查事先定好的概率。"""

    def __init__(self, probs: dict[int, dict[str, float]]):
        self.probs = probs
        self.size = 16

    def labels(self, head):
        return ["not_person", "lit", "unlit", "spirit"]

    def pad(self, head):
        return 0.15

    def keep(self, head):
        return None

    def predict(self, items):
        out = []
        for _, img in items:
            p = self.probs[int(img[8, 8, 0])]  # 裁图中心的颜色编码了框序号
            out.append({"form": np.array([p.get(k, 0.0) for k in self.labels("form")], np.float32)})
        return out


def label_line(cls, box):
    return f"{cls} {(box.x + box.w / 2) / W:.6f} {(box.y + box.h / 2) / H:.6f} {box.w / W:.6f} {box.h / H:.6f}"


def make_dataset(root, frames, split="train", prefix="rec", suffix=""):
    """frames: {帧号: [(类别号, 框)]}；图片的蓝通道 = 帧号 * 10，框内区域的蓝通道 = 100 + 框左上角 x（裁图中心据此认框）。"""
    for n, boxes in frames.items():
        img = np.full((H, W, 3), n * 10, np.uint8)
        for _, b in boxes:
            img[b.y:b.y + b.h, b.x:b.x + b.w, 0] = 100 + b.x % 100
        stem = f"{prefix}_{n:04d}_{n * 1.0:06.2f}s{suffix}"
        imwrite(root / "images" / split / f"{stem}.png", img)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(label_line(c, b) for c, b in boxes) + "\n", encoding="utf-8")
    return root


def det_key(n):
    return n * 10  # 与 make_dataset 的图片蓝通道一致


A = Rect(20, 50, 60, 120)
B = Rect(200, 50, 60, 120)
C = Rect(300, 50, 60, 120)


def crops_jsonl(attrs):
    p = attrs / "_crops.jsonl"
    return [json.loads(x) for x in p.read_text("utf-8").splitlines()] if p.exists() else []


def test_audit_missed_crops_written(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]})
    det = FakeDetector({det_key(1): [("player", A, 0.9), ("player", B, 0.8)]})  # B 标注里没有
    res = audit(ds, attrs, {"m1": det})
    assert len(res["missed"]) == 1 and res["focus"] == res["missed"]
    name = res["missed"][0]
    assert (attrs / "_unlabeled" / name).is_file()
    row = next(r for r in crops_jsonl(attrs) if r["crop"] == name)
    assert row["found_by"] == "m1" and row["source"] == "dataset" and row["split"] == "train" and row["known"] is False


def test_audit_missed_deduped_across_detectors(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]})
    d1 = FakeDetector({det_key(1): [("player", B, 0.9)]})
    d2 = FakeDetector({det_key(1): [("player", Rect(202, 52, 60, 120), 0.9)]})  # 几乎同一个框
    res = audit(ds, attrs, {"m1": d1, "m2": d2})
    assert len(res["missed"]) == 1 and len(crops_jsonl(attrs)) == 2  # B 一张 + 没人框到的标注 A（suspect）补裁


def test_audit_suspect_and_partial(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A), (0, B)]})
    d1 = FakeDetector({det_key(1): [("player", B, 0.5)]})
    d2 = FakeDetector({det_key(1): []})
    res = audit(ds, attrs, {"m1": d1, "m2": d2})
    assert len(res["suspect"]) == 1 and len(res["partial"]) == 1
    assert (attrs / "form" / "lit" / res["suspect"][0]).is_file()  # 标注的框补裁进 form/lit
    assert set(res["focus"]) == set(res["suspect"]) | set(res["partial"])


def test_audit_head_flags_not_person(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A), (0, C)]})
    det = FakeDetector({det_key(1): [("player", A, 0.9), ("player", C, 0.9)]})
    head = FakeHead({100 + A.x % 100: {"not_person": 0.9, "lit": 0.1}, 100 + C.x % 100: {"lit": 0.9, "not_person": 0.1}})
    res = audit(ds, attrs, {"m1": det}, head=head)
    assert len(res["head"]) == 1 and res["head"][0].startswith("rec_0001_") and f"__{A.x}_" in res["head"][0]
    assert res["suspect"] == [] and res["partial"] == []


def test_audit_skips_augmented_and_val(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]}, split="val")
    make_dataset(ds, {2: [(0, A)]}, suffix="_blur")  # 增强图
    det = FakeDetector({det_key(1): [("player", B, 0.9)], det_key(2): [("player", B, 0.9)]})
    res = audit(ds, attrs, {"m1": det})
    assert res == {"missed": [], "suspect": [], "partial": [], "head": [], "focus": [], "scores": {}}


def test_write_focus_backs_up(tmp_path):
    assert write_focus(tmp_path, ["a.jpg"]) is None
    assert json.loads((tmp_path / "_focus.json").read_text("utf-8")) == ["a.jpg"]
    backup = write_focus(tmp_path, ["b.jpg"])
    assert backup is not None and backup.name.startswith("_focus-") and backup.suffix == ".json"
    assert json.loads(backup.read_text("utf-8")) == ["a.jpg"]
    assert json.loads((tmp_path / "_focus.json").read_text("utf-8")) == ["b.jpg"]


def test_audit_scores_cover_all_label_boxes_and_missed(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    SELF = Rect(130, 150, 40, 80)
    make_dataset(ds, {1: [(0, A), (4, C), (3, SELF)]})
    det = FakeDetector({det_key(1): [("player", A, 0.9), ("player", C, 0.9), ("player", B, 0.8)]})  # B 标注里没有
    head = FakeHead({100 + A.x % 100: {"lit": 0.7, "not_person": 0.3}, 100 + C.x % 100: {"unlit": 0.9, "not_person": 0.1},
                     10: {"lit": 0.5, "spirit": 0.2, "not_person": 0.3}, 100 + SELF.x % 100: {"lit": 1.0}})
    res = audit(ds, attrs, {"m1": det}, head=head)
    a, c = f"rec_0001_001.00s__{A.x}_{A.y}_{A.w}_{A.h}.jpg", f"rec_0001_001.00s__{C.x}_{C.y}_{C.w}_{C.h}.jpg"
    missed = res["missed"][0]
    assert set(res["scores"]) == {a, c, missed}  # self 框没有分数
    assert res["scores"][a]["lit"] == 0.7 and res["scores"][c]["unlit"] == 0.9
    assert res["scores"][missed] == {"not_person": 0.3, "lit": 0.5, "unlit": 0.0, "spirit": 0.2}


def test_audit_no_head_no_scores(tmp_path):
    ds, attrs = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]})
    det = FakeDetector({det_key(1): [("player", A, 0.9), ("player", B, 0.8)]})
    assert audit(ds, attrs, {"m1": det})["scores"] == {}


def test_write_scores_backs_up(tmp_path):
    assert write_scores(tmp_path, {}) is None and not (tmp_path / "_scores.json").exists()
    assert write_scores(tmp_path, {"a.jpg": {"lit": 0.9}}) is None
    backup = write_scores(tmp_path, {"b.jpg": {"lit": 0.1}})
    assert backup is not None and backup.name.startswith("_scores-") and json.loads(backup.read_text("utf-8")) == {"a.jpg": {"lit": 0.9}}
    assert json.loads((tmp_path / "_scores.json").read_text("utf-8")) == {"b.jpg": {"lit": 0.1}}


def test_cli_crops_audit_writes_focus(tmp_path, monkeypatch, capsys):
    from skydango import cli
    from skydango.vision import attrs, attrs_train

    ds, out = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]})
    seen = []
    monkeypatch.setattr(attrs_train, "replay_detector", lambda p, model=None, conf=0.2: seen.append(model) or FakeDetector({det_key(1): [("player", B, 0.9)]}))
    monkeypatch.setattr(attrs, "load_model", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    cli.main(["perception", "crops-audit", str(ds), "--models", "a.pt,b.pt", "--attrs-data", str(out)])
    assert seen == ["a.pt", "b.pt"]
    focus = json.loads((out / "_focus.json").read_text("utf-8"))
    assert len(focus) == 2  # 漏标的 B + 两个模型都没框到标注 A
    assert "外形头没加载" in capsys.readouterr().out


def test_cli_crops_audit_writes_scores(tmp_path, monkeypatch, capsys):
    from skydango import cli
    from skydango.vision import attrs, attrs_train

    ds, out = tmp_path / "ds", tmp_path / "attrs"
    make_dataset(ds, {1: [(0, A)]})
    monkeypatch.setattr(attrs_train, "replay_detector", lambda p, model=None, conf=0.2: FakeDetector({det_key(1): [("player", A, 0.9)]}))
    monkeypatch.setattr(attrs, "load_model", lambda *a, **k: FakeHead({100 + A.x % 100: {"lit": 0.8, "not_person": 0.2}}))
    monkeypatch.chdir(tmp_path)
    cli.main(["perception", "crops-audit", str(ds), "--models", "a.pt", "--attrs-data", str(out)])
    scores = json.loads((out / "_scores.json").read_text("utf-8"))
    assert len(scores) == 1 and next(iter(scores.values()))["lit"] == 0.8
    assert "_scores.json" in capsys.readouterr().out
