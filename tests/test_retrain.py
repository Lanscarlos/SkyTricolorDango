import datetime as dt
import json
import time

import numpy as np
import pytest

from skydango import cli
from skydango.config import Config
from skydango.imageio import imwrite
from skydango.vision import attrs, inbox as ib, retrain
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection


# ---------- 小工具 ----------

def test_next_version(tmp_path):
    models, ty = tmp_path / "models", tmp_path / "tmp_yolo"
    assert retrain.next_version(models, ty) == 1  # 都没有
    models.mkdir()
    for name in ("sky-yolo-v7.pt", "sky-yolo-v10.pt", "sky-yolo-v1.onnx", "yolo11n.pt", "attrs-20261004.npz"):
        (models / name).write_bytes(b"x")
    assert retrain.next_version(models, ty) == 11
    (ty / "sky-v11").mkdir(parents=True)
    (ty / "sky-v9").mkdir()
    (ty / "train_v99.py").write_text("x")  # 不是 sky-v<N>/ 目录
    assert retrain.next_version(models, ty) == 12


def test_next_attrs_letter(tmp_path):
    now = dt.datetime(2026, 10, 5, 9, 0)
    assert retrain.next_attrs(tmp_path, now) == tmp_path / "attrs-20261005a.npz"
    (tmp_path / "attrs-20261005a.npz").write_bytes(b"x")
    (tmp_path / "attrs-20261005b-mask.npz").write_bytes(b"x")  # b 带后缀也算用过
    (tmp_path / "attrs-20261004c.npz").write_bytes(b"x")  # 别的日期不算
    assert retrain.next_attrs(tmp_path, now) == tmp_path / "attrs-20261005c.npz"


def test_stash_caches(tmp_path):
    ds = tmp_path / "sky"
    (ds / "labels" / "val").mkdir(parents=True)
    (ds / "labels" / "train.cache").write_bytes(b"a")
    (ds / "labels" / "val.cache").write_bytes(b"b")
    (ds / "labels" / "val" / "x.cache").write_bytes(b"c")
    (ds / "labels" / "val" / "f.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    moved = retrain.stash_caches(ds, tmp_path / "out" / "caches")
    assert sorted(p.relative_to(tmp_path / "out" / "caches").as_posix() for p in moved) == ["train.cache", "val.cache", "val/x.cache"]
    assert all(p.is_file() for p in moved)
    assert not list((ds / "labels").rglob("*.cache")) and (ds / "labels" / "val" / "f.txt").is_file()
    assert retrain.stash_caches(ds, tmp_path / "out2") == []


def test_mark_trained_and_passed_frames(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "r1").mkdir(parents=True)
    ib.save_frames(inbox, "r1", {
        "a": {"split": "val", "decision": {"what": "pass", "t": 100.0, "dataset": "val/a"}},
        "b": {"split": "train", "decision": {"what": "pass", "t": 300.0, "dataset": "train/b"}},
        "c": {"split": "train", "decision": {"what": "discard", "t": 100.0}},
        "d": {"split": "train", "decision": None},
    })
    assert [p["dataset"] for p in ib.passed_frames(inbox)] == ["val/a", "train/b"]
    assert [p["dataset"] for p in ib.passed_frames(inbox, before=200.0)] == ["val/a"]
    ib.mark_trained(inbox, 1, at=200.0)
    st = ib.read_stats(inbox)
    assert st["trained_at"] == 200.0 and st["trained_frames"] == 1
    assert ib.passed_frames(tmp_path / "nope") == []


# ---------- 整个流程 ----------

class FakeDet:
    def __init__(self, per_frame):
        self.per_frame = per_frame

    def detect(self, img):
        return list(self.per_frame.get(int(img[0, 0, 0]), []))


class BucketEmbedder:
    size = 8
    key = "b.onnx:8:imagenet"

    def embed(self, img):
        return np.array([1.0, 0.0, 0.0], np.float32)


def _head(row: int) -> dict:
    W = np.zeros((3, 3), np.float32)
    W[0, row] = 8.0
    return {"W": W, "b": np.zeros(3, np.float32), "labels": ["not_person", "lit", "unlit"],
            "applies_to": ["player", "player_unlit"], "pad": 0.15, "keep": None}


def _d(cls, x, s):
    return Detection(cls, Rect(x, 20, 40, 60), s)


OLD = {2: [_d("player", 70, 0.9)], 3: [_d("player_unlit", 100, 0.9)]}  # 漏了 f1（来自收件箱）里的人
NEW = {1: [_d("player", 20, 0.9)], 2: [_d("player", 70, 0.9)], 3: [_d("player_unlit", 100, 0.9)]}


def _world(tmp_path, monkeypatch, old_attrs=True):
    ds = tmp_path / "sky"
    for tag, stem, label in ((1, "f1", "0 0.2 0.5 0.2 0.6\n"), (2, "f2", "0 0.45 0.5 0.2 0.6\n"),
                             (3, "f3", "4 0.6 0.5 0.2 0.6\n")):
        img = np.full((100, 200, 3), 50, np.uint8)
        img[0, 0, 0] = tag
        imwrite(ds / "images/val" / f"{stem}.png", img)
        (ds / "labels/val").mkdir(parents=True, exist_ok=True)
        (ds / "labels/val" / f"{stem}.txt").write_text(label, encoding="utf-8")
    (ds / "labels" / "val.cache").write_bytes(b"stale")
    (ds / "data.yaml").write_text(f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n", encoding="utf-8")
    inbox = tmp_path / "inbox"
    (inbox / "r1").mkdir(parents=True)
    ib.save_frames(inbox, "r1", {
        "f1": {"split": "val", "decision": {"what": "pass", "t": 100.0, "dataset": "val/f1"}},
        "t1": {"split": "train", "decision": {"what": "pass", "t": 100.0, "dataset": "train/t1"}},
        "t2": {"split": "train", "decision": {"what": "pass", "t": time.time() + 1e6, "dataset": "train/t2"}},  # 训练开始后才通过
    })
    models, ty = tmp_path / "models", tmp_path / "tmp_yolo"
    models.mkdir()
    (models / "sky-yolo-v10.pt").write_bytes(b"old")
    (ty / "sky-v11").mkdir(parents=True)
    cfg = Config()
    cfg.perception.model = str(models / "sky-yolo-v10.pt")
    cfg.perception.low_conf = 0.2
    cfg.attrs.backbone = "b.onnx"
    cfg.attrs.model = str(models / "attrs-old.npz")
    if old_attrs:
        attrs.save_model(models / "attrs-old.npz", {"form": _head(1)}, "b.onnx", "8:imagenet", "20261004")
    monkeypatch.setattr("skydango.vision.detect.make_detector",
                        lambda path, *a, **kw: FakeDet(NEW if "v12" in str(path) else OLD))
    calls = {"train": [], "val": [], "attrs": []}

    def train(rcfg, data, out, progress):
        calls["train"].append((rcfg.epochs, data))
        (out / "weights").mkdir(parents=True)
        (out / "weights" / "best.pt").write_bytes(b"new")

    def val(weights, data, out):
        calls["val"].append((weights, data))
        new = "v12" in str(weights)
        return {"player": 0.82 if new else 0.80, "all": 0.9 if new else 0.7}

    def attrs_train_fn(c, embedder, attrs_root, out, now):
        calls["attrs"].append((attrs_root, out))
        attrs.save_model(out, {"form": _head(1)}, "b.onnx", f"{embedder.size}:imagenet", f"{now:%Y%m%d}")
        return {"eval": {"macro_f1": 0.9}, "train_n": 10, "val_n": 3, "notes": []}

    kw = dict(train=train, val=val, attrs_train_fn=attrs_train_fn, embedder=BucketEmbedder(),
              models=models, tmp_yolo=ty, attrs_root=tmp_path / "attrs")
    return cfg, ds, inbox, models, calls, kw


def test_run_retrain_end_to_end(tmp_path, monkeypatch):
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch)
    out = tmp_path / "out"
    lines = []
    before = time.time()
    now = dt.datetime(2026, 10, 5, 23, 59, 59)
    res = retrain.run_retrain(cfg, ds, inbox, out, lines.append, now=now, **kw)
    assert (models / "sky-yolo-v12.pt").read_bytes() == b"new"
    new_attrs = models / "attrs-20261005a.npz"
    assert new_attrs.is_file() and calls["attrs"][0][1] == new_attrs
    assert calls["train"][0] == (cfg.retrain.epochs, ds / "data.yaml")
    assert not (ds / "labels" / "val.cache").exists() and (out / "caches" / "val.cache").is_file()
    assert any(s == "PROGRESS 训练 YOLO" or s.startswith("PROGRESS 训练 YOLO") for s in lines)
    assert all(s.startswith("PROGRESS ") for s in lines)
    # 结果
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert saved["ok"] is True and saved["yolo"] == str(models / "sky-yolo-v12.pt") and saved["attrs"] == str(new_attrs)
    assert saved["old_yolo"] == cfg.perception.model and saved["old_attrs"] == cfg.attrs.model
    n = res["numbers"]
    assert n["all"]["frames"] == 3 and n["inbox"]["frames"] == 1
    assert n["all"]["rows"]["old"]["recall"] == pytest.approx(2 / 3) and n["all"]["rows"]["new"]["recall"] == 1.0
    assert n["inbox"]["rows"]["old"]["recall"] == 0.0 and n["inbox"]["rows"]["new"]["recall"] == 1.0
    assert n["all"]["rows"]["old_head"] is not None and n["all"]["rows"]["new_head"] is not None
    assert n["all"]["map50"]["new"]["player"] == 0.82
    # ultralytics val：旧 / 新 × 全部 / 收件箱；收件箱那份的清单里只有 f1
    assert len(calls["val"]) == 4
    sub = [d for _, d in calls["val"] if d != ds / "data.yaml"]
    assert len(sub) == 2
    listed = [line for line in (out / "inbox_val.txt").read_text(encoding="utf-8").splitlines() if line]
    assert [p.split("\\")[-1].split("/")[-1] for p in listed] == ["f1.png"]
    # 报告
    md = (out / "report.md").read_text(encoding="utf-8")
    for row in ("| 旧 YOLO |", "| 旧 YOLO + 旧外形头 |", "| 新 YOLO |", "| 新 YOLO + 新外形头 |"):
        assert md.count(row) == 2  # 全部验证帧、来自收件箱的验证帧各一张表
    assert "新旧用的是同一份答案" in md and "可能是波动" in md
    assert "82%（+2）（可能是波动）" in md  # player mAP50 0.80 → 0.82
    assert "90%（+20）" in md and "90%（+20）（可能是波动）" not in md
    assert "| 新 YOLO | 3 | 0 | 0 | 100%（±0）（可能是波动） | 100%（+33） |" in md  # 精确率一样也标
    # 训练帧数：开始之前通过的 2 帧（t2 是训练开始后才通过的，不算）
    st = ib.read_stats(inbox)
    assert st["trained_frames"] == 2 and before <= st["trained_at"] <= time.time()


def test_run_retrain_without_old_head(tmp_path, monkeypatch):
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch, old_attrs=False)
    res = retrain.run_retrain(cfg, ds, inbox, tmp_path / "out", lambda s: None, **kw)
    rows = res["numbers"]["all"]["rows"]
    assert rows["old_head"] is None and rows["old"] is not None and rows["new_head"] is not None
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "没有旧外形头" in md


def test_run_retrain_failure_writes_no_result(tmp_path, monkeypatch):
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("显卡炸了")

    kw["train"] = boom
    out = tmp_path / "out"
    with pytest.raises(retrain.RetrainFailed) as e:
        retrain.run_retrain(cfg, ds, inbox, out, lambda s: None, **kw)
    assert e.value.step == "训练 YOLO" and "显卡炸了" in str(e.value)
    assert not (out / "result.json").exists() and not (models / "sky-yolo-v12.pt").exists()
    assert "trained_at" not in ib.read_stats(inbox)


def test_run_retrain_mark_trained_failure_writes_no_result(tmp_path, monkeypatch):
    # result.json 最后写：记训练时间失败也算没做完（spec §9）
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise OSError("盘满了")

    monkeypatch.setattr(ib, "mark_trained", boom)
    out = tmp_path / "out"
    with pytest.raises(retrain.RetrainFailed) as e:
        retrain.run_retrain(cfg, ds, inbox, out, lambda s: None, **kw)
    assert e.value.step == "记下训练时间"
    assert not (out / "result.json").exists() and (out / "report.md").is_file()


def _load_head(path, head, cfg):
    import dataclasses

    attrs.save_model(path, {"form": head}, "b.onnx", "8:imagenet", "20261005")
    return attrs.load_model(dataclasses.replace(cfg.attrs, model=str(path)), "cpu", BucketEmbedder())


def _spirit_head() -> dict:
    W = np.zeros((3, 4), np.float32)
    W[0, 1] = 8.0
    return {"W": W, "b": np.zeros(4, np.float32), "labels": ["not_person", "lit", "unlit", "spirit"],
            "applies_to": ["player", "player_unlit"], "pad": 0.15, "keep": None}


@pytest.mark.parametrize("old_spirit,want_gt", [(False, 3), (True, 4)])
def test_compare_shares_one_answer_for_spirit(tmp_path, monkeypatch, old_spirit, want_gt):
    """旧头没有先祖类、新头有：四行都按同一份答案（去掉先祖），两边的人数一样；都有先祖类才留先祖。"""
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch)
    img = np.full((100, 200, 3), 50, np.uint8)
    img[0, 0, 0] = 4
    imwrite(ds / "images/val/f4.png", img)
    (ds / "labels/val/f4.txt").write_text("9 0.5 0.5 0.2 0.6\n", encoding="utf-8")  # 一个先祖，两边都没检出
    old = _load_head(tmp_path / "old.npz", _spirit_head() if old_spirit else _head(1), cfg)
    new = _load_head(tmp_path / "new.npz", _spirit_head(), cfg)
    n = retrain.compare(cfg.perception.model, models / "sky-yolo-v12.pt", old, new, ds, cfg, {"f1"}, attrs_root=None)
    rows = n["all"]["rows"]
    assert n["all"]["gt"] == want_gt and n["all"]["frames"] == 4
    for key in ("old", "old_head", "new", "new_head"):
        assert rows[key]["tp"] + rows[key]["fn"] == want_gt  # 每行的答案人数一样
    assert any("去掉了先祖" in x for x in n["notes"]) == (not old_spirit)
    md = retrain.report_md({"yolo": "y", "attrs": "a", "numbers": n})
    assert ("去掉了先祖" in md) == (not old_spirit)


def test_compare_without_old_yolo(tmp_path, monkeypatch):
    cfg, ds, inbox, models, calls, kw = _world(tmp_path, monkeypatch)
    new = _load_head(tmp_path / "new.npz", _head(1), cfg)
    n = retrain.compare(tmp_path / "nope.pt", models / "sky-yolo-v12.pt", None, new, ds, cfg, {"f1"},
                        attrs_root=None, val=kw["val"], work=tmp_path / "work")
    for key in ("all", "inbox"):
        rows = n[key]["rows"]
        assert rows["old"] is None and rows["old_head"] is None and rows["new"] is not None and rows["new_head"] is not None
        assert n[key]["map50"]["old"] is None and n[key]["map50"]["new"]["player"] == 0.82
    assert any("没有旧 YOLO" in x for x in n["notes"])
    assert all("v12" in str(w) for w, _ in calls["val"]) and len(calls["val"]) == 2
    md = retrain.report_md({"yolo": "y", "attrs": "a", "old_yolo": str(tmp_path / "nope.pt"), "numbers": n})
    assert md.count("| 旧 YOLO | 没有旧 YOLO |") == 2 and md.count("| 旧 YOLO + 旧外形头 | 没有旧 YOLO |") == 2
    assert "| player | — | 82% |" in md  # 没有旧值：不比


def test_cell_marks_on_displayed_difference():
    assert retrain._cell(0.70, 0.70) == "70%（±0）（可能是波动）"
    assert retrain._cell(0.80, 0.766) == "80%（+3）（可能是波动）"  # 实差 3.4 点，显示 +3：按显示的算
    assert retrain._cell(0.7351, 0.70) == "74%（+4）"
    assert retrain._cell(0.66, 0.70) == "66%（-4）"
    assert retrain._cell(None, 0.7) == "—" and retrain._cell(0.5, None) == "50%"


def test_report_md_marks_small_differences():
    def row(p, r):
        return {"tp": 1, "fp": 0, "fn": 0, "precision": p, "recall": r, "pairs": 1, "mismatch": 0}

    part = {"frames": 5, "gt": 7, "rows": {"old": row(0.78, 0.70), "old_head": None, "new": row(0.80, 0.90), "new_head": row(0.5, 0.5)},
            "map50": None}
    md = retrain.report_md({"yolo": "models/sky-yolo-v12.pt", "attrs": "models/attrs-20261005a.npz",
                            "old_yolo": "models/sky-yolo-v10.pt", "old_attrs": "models/attrs-x.npz",
                            "numbers": {"all": part, "inbox": None, "notes": ["没有旧外形头"]}})
    assert "80%（+2）（可能是波动）" in md and "90%（+20）" in md and "90%（+20）（可能是波动）" not in md
    assert "没有来自收件箱的验证帧" in md and "新旧用的是同一份答案" in md


def test_cli_retrain(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text('[inbox]\ndir = "ib"\n', encoding="utf-8")
    seen = {}

    def fake(cfg, dataset, inbox, out, progress, **kw):
        seen.update(epochs=cfg.retrain.epochs, dataset=dataset, inbox=inbox, out=out)
        progress("PROGRESS 训练 YOLO")
        return {"yolo": "models/sky-yolo-v12.pt", "attrs": "models/attrs-20261005a.npz"}

    monkeypatch.setattr(retrain, "run_retrain", fake)
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "retrain", "--epochs", "3"])
    out = capsys.readouterr().out
    assert seen["epochs"] == 3 and seen["dataset"].as_posix() == "datasets/sky" and seen["inbox"].as_posix() == "ib"
    assert seen["out"].parent.as_posix() == "tmp/retrain"
    assert "PROGRESS 训练 YOLO" in out and "sky-yolo-v12.pt" in out

    def bad(*a, **k):
        raise retrain.RetrainFailed("训练外形头", ValueError("这几类确认过的图不够"))

    monkeypatch.setattr(retrain, "run_retrain", bad)
    with pytest.raises(SystemExit) as e:
        cli.main(["-c", str(tmp_path / "config.toml"), "perception", "retrain"])
    assert e.value.code not in (0, None)
    assert "重训失败（训练外形头）" in capsys.readouterr().out
