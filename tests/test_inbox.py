import json
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
    assert final_boxes(e, lambda c: "_discard", classes) == [(1, Rect(5, 6, 7, 8))]  # 自动确认的裁图被判不要也去掉（终审 3）



def test_final_boxes_follow_reclassified_auto_agree_crop():
    # 终审 3：自动确认的框也按裁图现在在哪算：人在外形页改判 / 判不要后，最终标注跟着变
    classes = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]
    e = {"boxes": [{"cls": "player", "box": [10, 20, 30, 40], "score": 0.8, "src": "yolo", "crop": "a.jpg", "auto": "agree"}]}
    assert final_boxes(e, lambda c: "lit", classes) == [(0, Rect(10, 20, 30, 40))]  # 没人改过：和以前一样
    assert final_boxes(e, lambda c: "unlit", classes) == [(4, Rect(10, 20, 30, 40))]
    assert final_boxes(e, lambda c: "not_person", classes) == []
    assert final_boxes(e, lambda c: "_discard", classes) == []


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


def test_pass_rolls_back_when_saving_decision_fails(tmp_path, monkeypatch):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    real = _inbox.save_frames
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(_inbox, "save_frames", flaky)
    with pytest.raises(OSError):
        _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert not (ds / "images" / "train" / f"{frame}.jpg").exists()
    assert not (ds / "labels" / "train" / f"{frame}.txt").exists()
    assert not (inbox / run / "labels" / f"{frame}.txt").exists()
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None
    assert _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES) == f"train/{frame}"


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


# ---- Task 5：整理流水线 ----
from skydango.config import Config as _Config
from skydango.vision.detect import Detection as _Det


def _img(shade: int, mark: int = 0) -> np.ndarray:
    im = np.full((1080, 1920, 3), shade, np.uint8)
    if mark:
        im[100:600, 200:900] = mark
    return im


def _setup_run(inbox, run, files):
    """files: {文件名: 图}；写进 inbox/<run>/raw/ 并登记进 _index.jsonl。"""
    raw = inbox / run / "raw"
    raw.mkdir(parents=True)
    for name, im in files.items():
        cv2.imwrite(str(raw / name), im)
    with (inbox / "_index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run, "collected_at": "2026-10-04T20:00:00", "frames": len(files)}) + "\n")


def _cfg():
    return _Config()


def _judge_by_order(table):
    """假外形头：按裁图出现顺序依次给概率。"""
    it = iter(table)
    return lambda frame, boxes: [next(it) for _ in boxes]


def test_process_routes_and_writes(tmp_path):
    inbox, attrs_root = tmp_path / "inbox", tmp_path / "attrs"
    _setup_run(inbox, "r1", {"192007_low_conf.jpg": _img(90)})
    (inbox / "r1" / "hard.jsonl").write_text('{"file": "192007_low_conf.jpg", "reason": "low_conf"}\n', encoding="utf-8")
    dets = [_Det("player", Rect(100, 200, 120, 260), 0.8), _Det("player", Rect(900, 300, 100, 240), 0.8)]
    weak = [("name_tag", Rect(80, 150, 160, 30))]
    judge = _judge_by_order([{"lit": 0.95, "unlit": 0.03, "not_person": 0.02}, {"lit": 0.2, "not_person": 0.75, "unlit": 0.05}])
    res = _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: weak, judge, progress=lambda s: None)
    frames = _inbox.load_frames(inbox, "r1")
    entry = frames["r1_192007_low_conf"]
    assert entry["file"] == "raw/192007_low_conf.jpg" and entry["reason"] == "low_conf" and entry["error"] is None
    assert sorted(b["cls"] for b in entry["boxes"]) == ["name_tag", "player", "player"]
    agree = [b for b in entry["boxes"] if b["auto"] == "agree"]
    human = [b for b in entry["boxes"] if b["cls"] == "player" and b["auto"] is None]
    assert len(agree) == 1 and len(human) == 1
    assert (attrs_root / "form" / "lit" / agree[0]["crop"]).is_file()
    assert (attrs_root / "_unlabeled" / human[0]["crop"]).is_file()
    labels = [json.loads(x) for x in (attrs_root / "_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert labels[0]["by"] == "auto-agree" and labels[0]["crop"] == agree[0]["crop"] and labels[0]["to"] == "lit"
    guess = json.loads((attrs_root / "_unlabeled" / "claude.json").read_text(encoding="utf-8"))
    assert "YOLO 判" in guess[human[0]["crop"]]["reason"] and guess[human[0]["crop"]]["label"] == "not_person"
    assert [b for b in entry["boxes"] if b["cls"] == "name_tag"][0]["crop"] is None
    assert res["frames"] == 1 and res["auto"] == 1 and res["to_judge"] == 1
    assert not pending_runs(inbox)


def test_process_dedupes(tmp_path):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(90), "192002_b.jpg": _img(91), "192012_c.jpg": _img(91)})
    calls = []
    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), lambda f: calls.append(1) or [], lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    frames = _inbox.load_frames(inbox, "r1")
    assert frames["r1_192002_b"]["dup_of"] == "r1_192000_a"
    assert not frames["r1_192012_c"].get("dup_of")
    assert len(calls) == 2


def test_process_resumes(tmp_path):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40), "192100_b.jpg": _img(200)})
    calls = []

    def boom(f):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return []

    with pytest.raises(KeyboardInterrupt):
        _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), boom, lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    assert list(_inbox.load_frames(inbox, "r1")) == ["r1_192000_a"] and pending_runs(inbox) == ["r1"]
    calls.clear()
    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), lambda f: calls.append(1) or [], lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    assert len(calls) == 1 and not pending_runs(inbox)


def test_process_frame_error_skips(tmp_path):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40), "192100_b.jpg": _img(200)})
    n = []

    def det(f):
        n.append(1)
        if len(n) == 1:
            raise ValueError("坏帧")
        return []

    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), det, lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    frames = _inbox.load_frames(inbox, "r1")
    assert "坏帧" in frames["r1_192000_a"]["error"] and frames["r1_192100_b"]["error"] is None
    assert frame_state(frames["r1_192000_a"], lambda c: None) == "error"


def test_process_collects_first(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r9" / "hard").mkdir(parents=True)
    cv2.imwrite(str(runs / "r9" / "hard" / "192000_x.jpg"), _img(60))
    inbox = tmp_path / "inbox"
    res = _inbox.process(inbox, tmp_path / "attrs", runs, _cfg(), lambda f: [], lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    assert "r9_192000_x" in _inbox.load_frames(inbox, "r9") and res["frames"] == 1


def test_process_drop_low_and_status(tmp_path):
    inbox, attrs_root = tmp_path / "inbox", tmp_path / "attrs"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    dets = [_Det("player", Rect(100, 200, 120, 260), 0.25)]
    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: [], lambda f, b: [{"not_person": 0.95, "lit": 0.05}], progress=lambda s: None)
    b = _inbox.load_frames(inbox, "r1")["r1_192000_a"]["boxes"][0]
    assert b["auto"] == "drop_low" and b["crop"] is None
    st = _inbox.status(inbox, attrs_root)
    assert st["runs"]["r1"]["glance"] == 1 and st["judge_left"] == 0 and st["passed_since_train"] == 0


def test_inbox_process_missing_model(tmp_path, capsys):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    before = {"x": 1}
    (inbox / "r1" / "frames.json").write_text(json.dumps(before), encoding="utf-8")
    (tmp_path / "yolo.onnx").write_bytes(b"x")
    (tmp_path / "config.toml").write_text(
        f'[inbox]\ndir = "{inbox.as_posix()}"\n[perception]\nmodel = "{(tmp_path / "yolo.onnx").as_posix()}"\n[attrs]\nmodel = "{(tmp_path / "nope.npz").as_posix()}"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "process", str(tmp_path / "runs")])
    assert e.value.code == 1 and "外形头" in capsys.readouterr().out
    assert json.loads((inbox / "r1" / "frames.json").read_text(encoding="utf-8")) == before


def test_process_judge_gets_frame_and_person_boxes(tmp_path):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    dets = [_Det("player", Rect(100, 200, 120, 260), 0.8), _Det("name_tag", Rect(90, 150, 150, 30), 0.9)]
    seen = []

    def judge(frame, boxes):
        seen.append((frame.shape, list(boxes)))
        return [{"lit": 0.5, "not_person": 0.5} for _ in boxes]

    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: [], judge, progress=lambda s: None)
    assert seen == [((1080, 1920, 3), [Rect(100, 200, 120, 260)])]


def test_process_flushes_crop_rows_per_frame(tmp_path):
    inbox, attrs_root = tmp_path / "inbox", tmp_path / "attrs"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40), "192100_b.jpg": _img(200)})
    dets = [_Det("player", Rect(100, 200, 120, 260), 0.8)]
    rows_at_second = []

    def judge(frame, boxes):
        path = attrs_root / "_crops.jsonl"
        rows_at_second.append(len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0)
        return [{"lit": 0.5, "not_person": 0.5} for _ in boxes]

    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: [], judge, progress=lambda s: None)
    assert rows_at_second == [0, 1]


def test_process_guess_has_prompt_version(tmp_path):
    from skydango.vision.attrs_data import FORM_PROMPT_VERSION

    inbox, attrs_root = tmp_path / "inbox", tmp_path / "attrs"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    dets = [_Det("player", Rect(100, 200, 120, 260), 0.8)]
    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: [], lambda f, b: [{"lit": 0.5, "not_person": 0.5}], progress=lambda s: None)
    guess = json.loads((attrs_root / "_unlabeled" / "claude.json").read_text(encoding="utf-8"))
    assert [g["version"] for g in guess.values()] == [FORM_PROMPT_VERSION]


def test_form_judge_masks_by_box_not_crop_size():
    class Model:
        size = 224

        def labels(self, head):
            return ["not_person", "lit"]

        def pad(self, head):
            return 0.15

        def keep(self, head):
            return 0.05

        def predict(self, items):
            self.items = items
            return [{"form": np.array([0.2, 0.8])} for _ in items]

    m = Model()
    frame = np.full((1080, 1920, 3), 255, np.uint8)
    out = cli._form_judge(m)(frame, [Rect(900, 300, 60, 400)])  # 瘦高的框：左右要填灰
    img = m.items[0][1]
    assert out == [{"not_person": pytest.approx(0.2), "lit": pytest.approx(0.8)}]
    assert img.shape[:2] == (224, 224) and (img[:, :5] == 114).all() and (img[:, -5:] == 114).all() and (img[100:120, 100:120] == 255).all()


def test_cli_inbox_process_model_load_failure(tmp_path, capsys, monkeypatch):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    for n in ("yolo.onnx", "attrs.npz", "bb.onnx"):
        (tmp_path / n).write_bytes(b"x")
    t = lambda n: (tmp_path / n).as_posix()  # noqa: E731
    lines = ["[inbox]", f'dir = "{inbox.as_posix()}"', "[perception]", f'model = "{t("yolo.onnx")}"',
             "[attrs]", f'model = "{t("attrs.npz")}"', f'backbone = "{t("bb.onnx")}"']
    (tmp_path / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import skydango.vision.embed as embed

    def boom(*a, **k):
        raise RuntimeError("坏主干")

    monkeypatch.setattr(embed, "OnnxEmbedder", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "process", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert e.value.code == 1 and "整理不了" in out and "坏主干" in out and not (inbox / "r1" / "frames.json").exists()


def test_process_skips_player_box_on_dango(tmp_path):
    inbox, attrs_root = tmp_path / "inbox", tmp_path / "attrs"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    dets = [_Det("self", Rect(1385, 458, 371, 433), 0.84), _Det("player", Rect(1386, 458, 373, 434), 0.354),
            _Det("player", Rect(200, 300, 100, 240), 0.8)]
    seen = []

    def judge(frame, boxes):
        seen.append(list(boxes))
        return [{"lit": 0.95, "not_person": 0.05} for _ in boxes]

    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: [(x.cls, x.box) for x in d], judge, progress=lambda s: None)
    assert seen == [[Rect(200, 300, 100, 240)]]
    boxes = _inbox.load_frames(inbox, "r1")["r1_192000_a"]["boxes"]
    assert sorted((b["cls"], tuple(b["box"])) for b in boxes) == [("player", (200, 300, 100, 240)), ("self", (1385, 458, 371, 433))]


def test_process_keeps_decisions_made_during_run(tmp_path):
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40), "192100_b.jpg": _img(200)})
    calls = []

    def detect(f):
        calls.append(1)
        if len(calls) == 2:  # 整理第二帧时，页面上不要了第一帧
            _inbox.discard_frame(inbox, "r1", "r1_192000_a")
        return []

    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), detect, lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    frames = _inbox.load_frames(inbox, "r1")
    assert frames["r1_192000_a"]["decision"]["what"] == "discard" and "r1_192100_b" in frames
