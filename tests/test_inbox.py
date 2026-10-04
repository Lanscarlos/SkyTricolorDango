import pytest

from skydango import cli
from skydango.config import InboxConfig, load_config
from skydango.vision.bubbles import Rect
from skydango.vision.inbox import collect, collect_all, crop_place, final_boxes, frame_name, frame_state, pending_runs, route, split_of


def test_split_of_whole_run_same_side():
    assert split_of("20261004-192711-live-brain", 5) in ("train", "val")
    vals = [r for r in (f"run{i}" for i in range(100)) if split_of(r, 5) == "val"]
    assert 10 <= len(vals) <= 30 and all(split_of(r, 5) == "val" for r in vals)


def test_frame_name():
    assert frame_name("r1", "192007_attrs_disagree.jpg") == "r1_192007_attrs_disagree"


def test_collect_copies_and_is_idempotent(tmp_path):
    run = tmp_path / "runs" / "20261004-192711-live-brain"
    (run / "hard").mkdir(parents=True)
    for n in ("192007_attrs_disagree.jpg", "192102_low_conf.jpg"):
        (run / "hard" / n).write_bytes(b"x")
    (run / "hard.jsonl").write_text('{"file": "192007_attrs_disagree.jpg"}\n', encoding="utf-8")
    inbox = tmp_path / "inbox"
    assert collect(run, inbox) == 2 and collect(run, inbox) == 0
    assert sorted(p.name for p in (inbox / run.name / "raw").iterdir()) == ["192007_attrs_disagree.jpg", "192102_low_conf.jpg"]
    assert (inbox / run.name / "hard.jsonl").is_file() and pending_runs(inbox) == [run.name]
    assert len((inbox / "_index.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_collect_nothing(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    assert collect(run, tmp_path / "inbox") == 0 and not (tmp_path / "inbox" / "r").exists()


def test_collect_all_and_pending(tmp_path):
    runs = tmp_path / "runs"
    for name, n in (("a", 1), ("b", 0), ("c", 2)):
        (runs / name / "hard").mkdir(parents=True)
        for i in range(n):
            (runs / name / "hard" / f"{i}.jpg").write_bytes(b"x")
    inbox = tmp_path / "inbox"
    assert collect_all(runs, inbox) == ["a", "c"]
    assert collect_all(runs, inbox) == []
    with (inbox / "_index.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"run": "a", "processed_at": "2026-10-04T20:00:00"}\n')
    assert pending_runs(inbox) == ["c"]


class _Env:
    unknown = None
    catalog = None

    def __init__(self, folder, saved=1):
        class Hard:
            pass

        self.hardcases = Hard()
        self.hardcases.saved = saved
        self.hardcases.folder = folder

    def stop(self):
        pass


def _fake_run(tmp_path):
    run = tmp_path / "runs" / "r1"
    (run / "hard").mkdir(parents=True)
    (run / "hard" / "1.jpg").write_bytes(b"x")
    return run


def test_stop_scene_collects(tmp_path, capsys):
    run = _fake_run(tmp_path)
    cfg = InboxConfig(dir=str(tmp_path / "inbox"))
    cli._stop_scene(_Env(run / "hard"), cfg)
    assert (tmp_path / "inbox" / "r1" / "raw" / "1.jpg").is_file()
    assert "datasets/inbox" in capsys.readouterr().out


def test_stop_scene_disabled_or_none_does_not_collect(tmp_path):
    run = _fake_run(tmp_path)
    cli._stop_scene(_Env(run / "hard"), InboxConfig(enabled=False, dir=str(tmp_path / "inbox")))
    cli._stop_scene(_Env(run / "hard"))
    assert not (tmp_path / "inbox").exists()


def test_stop_scene_never_raises(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("盘满了")

    monkeypatch.setattr("skydango.vision.inbox.collect", boom)
    cli._stop_scene(_Env(run / "hard"), InboxConfig(dir=str(tmp_path / "inbox")))


def test_inbox_defaults():
    cfg = load_config(None)
    assert cfg.inbox.enabled and cfg.inbox.ask and cfg.inbox.dir == "datasets/inbox"
    assert (cfg.inbox.agree, cfg.inbox.dup_diff, cfg.inbox.dup_gap, cfg.inbox.val_every, cfg.inbox.retrain_min) == (0.9, 6.0, 5.0, 5, 50)
    assert (cfg.retrain.base, cfg.retrain.imgsz, cfg.retrain.epochs, cfg.retrain.batch, cfg.retrain.workers) == ("models/yolo11n.pt", 960, 120, 16, 2)


def test_inbox_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[inbox]\nask = false\nretrain_min = 30\n[retrain]\nepochs = 3\n", encoding="utf-8")
    cfg = load_config(p)
    assert not cfg.inbox.ask and cfg.inbox.retrain_min == 30 and cfg.retrain.epochs == 3


def test_cli_inbox_collect(tmp_path, capsys):
    run = tmp_path / "runs" / "r1"
    (run / "hard").mkdir(parents=True)
    (run / "hard" / "1.jpg").write_bytes(b"x")
    (tmp_path / "config.toml").write_text(f'[inbox]\ndir = "{(tmp_path / "inbox").as_posix()}"\n', encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "collect", str(tmp_path / "runs")])
    assert "收了 1 次运行 / 共 1 张" in capsys.readouterr().out
    assert (tmp_path / "inbox" / "r1" / "raw" / "1.jpg").is_file()


@pytest.mark.parametrize("cls,score,probs,want", [
    ("player", 0.8, {"lit": 0.95, "unlit": 0.03, "not_person": 0.02}, "agree"),
    ("player", 0.8, {"not_person": 0.97, "lit": 0.03}, None),  # 高分框判不是人：给人看
    ("player_unlit", 0.6, {"lit": 0.7, "unlit": 0.3}, None),  # 不一致
    ("player_unlit", 0.3, {"unlit": 0.93, "lit": 0.07}, "agree"),  # 低分框一致：补成正式框
    ("player", 0.3, {"not_person": 0.92, "lit": 0.08}, "drop_low"),
    ("player", 0.3, {"not_person": 0.6, "lit": 0.4}, None),
    ("player", 0.8, {"shared": 0.95, "lit": 0.05}, "agree"),  # shared 并进 lit
])
def test_route(cls, score, probs, want):
    assert route(cls, score, probs, conf=0.35, agree=0.9).auto == want


def test_frame_state_order_and_done_sticks():
    e = {"boxes": [{"cls": "player", "crop": "a.jpg", "auto": None}], "decision": None, "dup_of": None, "error": None, "editing": False}
    assert frame_state(e, lambda c: "_unlabeled") == "crops"
    assert frame_state(e, lambda c: "lit") == "glance"
    assert frame_state(e, lambda c: "_discard") == "edit"
    assert frame_state({**e, "editing": True}, lambda c: "lit") == "edit"
    assert frame_state({**e, "decision": {"what": "pass"}}, lambda c: "_unlabeled") == "done"  # 通过后外形页再改也不回 crops
    assert frame_state({**e, "decision": {"what": "discard"}}, lambda c: "lit") == "discarded"
    assert frame_state({**e, "dup_of": "x"}, lambda c: "lit") == "dup"
    assert frame_state({**e, "error": "boom"}, lambda c: "lit") == "error"


def test_crop_place(tmp_path):
    (tmp_path / "_unlabeled").mkdir()
    (tmp_path / "_unlabeled" / "a.jpg").write_bytes(b"x")
    (tmp_path / "_discard").mkdir()
    (tmp_path / "_discard" / "b.jpg").write_bytes(b"x")
    (tmp_path / "form" / "unlit").mkdir(parents=True)
    (tmp_path / "form" / "unlit" / "c.jpg").write_bytes(b"x")
    assert [crop_place(tmp_path, n) for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")] == ["_unlabeled", "_discard", "unlit", None]


def test_final_boxes():
    classes = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]
    e = {"boxes": [
        {"cls": "player", "box": [10, 20, 30, 40], "score": 0.8, "src": "yolo", "crop": "a.jpg", "auto": "agree"},
        {"cls": "player", "box": [50, 60, 30, 40], "score": 0.6, "src": "yolo", "crop": "b.jpg", "auto": None},  # 人判 unlit
        {"cls": "player", "box": [90, 60, 30, 40], "score": 0.6, "src": "yolo", "crop": "c.jpg", "auto": None},  # 人判 not_person
        {"cls": "player", "box": [130, 60, 30, 40], "score": 0.2, "src": "yolo", "crop": "d.jpg", "auto": "drop_low"},
        {"cls": "name_tag", "box": [5, 6, 7, 8], "score": 0.9, "src": "yolo", "crop": None, "auto": None},
    ]}
    where = {"a.jpg": "lit", "b.jpg": "unlit", "c.jpg": "not_person", "d.jpg": "not_person"}
    assert final_boxes(e, where.get, classes) == [(0, Rect(10, 20, 30, 40)), (4, Rect(50, 60, 30, 40)), (1, Rect(5, 6, 7, 8))]
    # 还没人判（_unlabeled）保持 YOLO 类别；挪进 _discard 的去掉
    assert final_boxes(e, lambda c: "_unlabeled", classes)[1] == (0, Rect(50, 60, 30, 40))
    assert len(final_boxes(e, lambda c: "_discard", classes)) == 2


# ---- 通过 / 不要 / 撤销 ----
import shutil as _shutil

import cv2
import numpy as np

from skydango.vision import inbox as _inbox
from skydango.vision.weaklabel import yolo_line

_CLASSES = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]


def _setup_frame(tmp_path, split="train"):
    inbox, ds, attrs_root = tmp_path / "inbox", tmp_path / "sky", tmp_path / "attrs"
    run, frame = "r1", "r1_a"
    (inbox / run / "raw").mkdir(parents=True)
    cv2.imwrite(str(inbox / run / "raw" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    entry = {"file": "raw/a.jpg", "split": split, "editing": False, "error": None, "dup_of": None, "decision": None,
             "boxes": [{"cls": "player", "box": [20, 10, 40, 60], "auto": "agree", "crop": "c1.jpg"},
                       {"cls": "bench", "box": [100, 50, 50, 20]}]}
    _inbox.save_frames(inbox, run, {frame: entry})
    return inbox, run, frame, attrs_root, ds


def test_pass_copies_into_dataset(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path, "val")
    assert _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES) == f"val/{frame}"
    lines = (ds / "labels" / "val" / f"{frame}.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [yolo_line(0, Rect(20, 10, 40, 60), 200, 100), yolo_line(6, Rect(100, 50, 50, 20), 200, 100)]
    assert (ds / "images" / "val" / f"{frame}.jpg").read_bytes() == (inbox / run / "raw" / "a.jpg").read_bytes()
    e = _inbox.load_frames(inbox, run)[frame]
    assert e["decision"]["what"] == "pass" and e["decision"]["dataset"] == f"val/{frame}" and e["decision"]["edited"] is False
    assert e["editing"] is False and (inbox / run / "labels" / f"{frame}.txt").is_file()


def test_pass_with_edited_boxes(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES, boxes=[(3, Rect(0, 0, 100, 50))])
    lines = (ds / "labels" / "train" / f"{frame}.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [yolo_line(3, Rect(0, 0, 100, 50), 200, 100)]
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["edited"] is True


def test_pass_refuses_clash(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "train" / f"{frame}.jpg").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None
    assert not (inbox / run / "labels" / f"{frame}.txt").exists()
    assert (ds / "images" / "train" / f"{frame}.jpg").read_bytes() == b"old"


def test_pass_rolls_back_on_copy_failure(tmp_path, monkeypatch):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    real, calls = _shutil.copyfile, []

    def flaky(src, dst, *a, **k):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(_inbox.shutil, "copyfile", flaky)
    with pytest.raises(OSError):
        _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert not (ds / "images" / "train" / f"{frame}.jpg").exists()
    assert not (ds / "labels" / "train" / f"{frame}.txt").exists()
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None


def test_undo_pass_removes_files(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    _inbox.undo_frame(inbox, run, frame, ds)
    assert not (ds / "images" / "train" / f"{frame}.jpg").exists()
    assert not (ds / "labels" / "train" / f"{frame}.txt").exists()
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None


def test_discard_and_undo(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    _inbox.discard_frame(inbox, run, frame)
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["what"] == "discard"
    _inbox.undo_frame(inbox, run, frame, ds)
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None
    _inbox.set_editing(inbox, run, frame, True)
    assert _inbox.load_frames(inbox, run)[frame]["editing"] is True
    assert _inbox.load_frames(inbox, "nope") == {}
