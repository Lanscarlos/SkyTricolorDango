import json
import pytest

from skydango import cli
from skydango.config import InboxConfig, load_config
from skydango.vision.bubbles import Rect
from skydango.vision.inbox import collect, collect_all, crop_place, final_boxes, frame_name, frame_state, pending_runs, route, save_frames, split_of


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
    save_frames(inbox, "a", {frame_name("a", "0.jpg"): {"file": "raw/0.jpg", "boxes": []}})  # 整理过：raw 里的图都在 frames.json 里
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


def test_cli_inbox_runs_default_from_config(tmp_path, capsys, monkeypatch):
    # 终审 8：collect / process 不给运行目录时用配置里的 [run] dir，不写死 runs
    run = tmp_path / "myruns" / "r1"
    (run / "hard").mkdir(parents=True)
    (run / "hard" / "1.jpg").write_bytes(b"x")
    lines = ["[run]", f'dir = "{(tmp_path / "myruns").as_posix()}"', "[inbox]", f'dir = "{(tmp_path / "inbox").as_posix()}"']
    (tmp_path / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "collect"])
    assert "收了 1 次运行 / 共 1 张" in capsys.readouterr().out
    seen = []
    monkeypatch.setattr(cli, "_inbox_process", lambda cfg, args: seen.append(args.runs))
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "process"])
    assert seen == [(tmp_path / "myruns").as_posix()]


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



@pytest.mark.parametrize("other", ["images", "labels"])
def test_pass_refuses_clash_in_other_split(tmp_path, other):
    # 终审 6：旧的 --from-runs 按帧名哈希分边，同名帧可能在另一边：两边都查，免得 train / val 泄漏
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path, "train")
    ext = "jpg" if other == "images" else "txt"
    (ds / other / "val").mkdir(parents=True)
    (ds / other / "val" / f"{frame}.{ext}").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert _inbox.load_frames(inbox, run)[frame]["decision"] is None
    assert not (ds / "images" / "train").exists() and not (inbox / run / "labels" / f"{frame}.txt").exists()


def test_pass_records_final_boxes_and_undo_keeps_edited_ones(tmp_path):
    # 终审 4：通过时把最终框（编号 + 框）记进 decision；撤销编辑过的通过，框留在 last_boxes，
    # 这一帧按它显示、按回车直接通过也写它；撤销没编辑的通过不留（照旧跟着外形页的判断）
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    e = _inbox.load_frames(inbox, run)[frame]
    assert e["decision"]["boxes"] == [{"cls": 0, "box": [20, 10, 40, 60]}, {"cls": 6, "box": [100, 50, 50, 20]}]
    _inbox.undo_frame(inbox, run, frame, ds)
    assert "last_boxes" not in _inbox.load_frames(inbox, run)[frame]
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES, boxes=[(3, Rect(0, 0, 100, 50))])
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["boxes"] == [{"cls": 3, "box": [0, 0, 100, 50]}]
    _inbox.undo_frame(inbox, run, frame, ds)
    e = _inbox.load_frames(inbox, run)[frame]
    assert e["decision"] is None and e["last_boxes"] == [{"cls": 3, "box": [0, 0, 100, 50]}]
    assert _inbox.frame_state(e, lambda c: "_discard") == "glance"  # 编辑过的帧：框已经人定了，不再要编辑
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)  # 不编辑直接通过：写上次编辑的框
    lines = (ds / "labels" / "train" / f"{frame}.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [yolo_line(3, Rect(0, 0, 100, 50), 200, 100)]
    e = _inbox.load_frames(inbox, run)[frame]
    assert e["decision"]["edited"] is True and "last_boxes" not in e


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



def test_raw_added_after_processing_is_pending(tmp_path):
    # 终审 7：运行标了 processed_at 之后，下线时补收进 raw/ 的图也算没整理，再整理只做新的那张
    inbox = tmp_path / "inbox"
    _setup_run(inbox, "r1", {"192000_a.jpg": _img(40)})
    calls = []
    run = lambda: _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), lambda f: calls.append(1) or [],
                                 lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    run()
    assert not pending_runs(inbox) and _inbox.pending_frames(inbox) == 0
    cv2.imwrite(str(inbox / "r1" / "raw" / "192100_b.jpg"), _img(200))
    assert pending_runs(inbox) == ["r1"] and _inbox.pending_frames(inbox) == 1
    calls.clear()
    run()
    assert len(calls) == 1 and list(_inbox.load_frames(inbox, "r1")) == ["r1_192000_a", "r1_192100_b"]
    assert not pending_runs(inbox) and _inbox.pending_frames(inbox) == 0


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


# ---- 认交互图标 Task 7：导入录像、老帧回炉 ----


def _write_dataset(ds, frames):
    """frames: [(split, 帧名, 图, 标注行或 None)]。"""
    for split, stem, im, lines in frames:
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ds / "images" / split / f"{stem}.jpg"), im)
        if lines is not None:
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split / f"{stem}.txt").write_text("".join(x + "\n" for x in lines), encoding="utf-8")


def test_add_images_every_and_idempotent(tmp_path):
    src, inbox = tmp_path / "icons-yunye-1005", tmp_path / "inbox"
    src.mkdir()
    for i in range(5):
        cv2.imwrite(str(src / f"{i:03d}.jpg"), _img(40 + 30 * i))
    cv2.imwrite(str(src / "005.png"), _img(250))
    (src / "notes.txt").write_text("x", encoding="utf-8")
    assert _inbox.add_images(src, inbox, every=2) == 3  # 000 / 002 / 004（按文件名排序隔 2 取 1）
    raw = inbox / src.name / "raw"
    assert sorted(p.name for p in raw.iterdir()) == ["000.jpg", "002.jpg", "004.jpg"]
    assert _inbox.add_images(src, inbox, every=1) == 3  # 再导一次：已有的跳过，补上 001 / 003 / 005（png 转 jpg）
    assert sorted(p.name for p in raw.iterdir()) == ["000.jpg", "001.jpg", "002.jpg", "003.jpg", "004.jpg", "005.jpg"]
    assert cv2.imread(str(raw / "005.jpg")) is not None
    assert _inbox.add_images(src, inbox) == 0
    rows = [json.loads(x) for x in (inbox / "_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["run"] == src.name and rows[0]["source"] == "import"
    assert pending_runs(inbox) == [src.name]


def test_process_marks_imported_reason(tmp_path):
    src, inbox = tmp_path / "rec", tmp_path / "inbox"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), _img(40))
    _inbox.add_images(src, inbox)
    _setup_run(inbox, "r1", {"192000_x.jpg": _img(90)})  # 普通运行没有 hard.jsonl：原因照旧是 None
    _inbox.process(inbox, tmp_path / "attrs", tmp_path / "runs", _cfg(), lambda f: [], lambda f, d: [], lambda f, b: [], progress=lambda s: None)
    assert _inbox.load_frames(inbox, "rec")["rec_a"]["reason"] == "import"
    assert _inbox.load_frames(inbox, "r1")["r1_192000_x"]["reason"] is None


def test_add_dataset_copies_frames_labels_and_redo_map(tmp_path):
    ds, inbox = tmp_path / "sky", tmp_path / "inbox"
    _write_dataset(ds, [("train", "a", _img(40), [yolo_line(0, Rect(100, 200, 120, 260), 1920, 1080)]),
                        ("train", "a_blur", _img(41), []),  # 增强图不回炉
                        ("val", "b", _img(200), None)])
    assert _inbox.add_dataset(ds, inbox) == 2
    runs = [p.name for p in inbox.iterdir() if p.is_dir()]
    assert len(runs) == 1 and runs[0].startswith("_redo-")
    run = runs[0]
    assert sorted(p.name for p in (inbox / run / "raw").iterdir()) == ["a.jpg", "b.jpg"]
    fa, fb = frame_name(run, "a.jpg"), frame_name(run, "b.jpg")
    assert (inbox / run / "labels" / f"{fa}.txt").read_text(encoding="utf-8") == (ds / "labels" / "train" / "a.txt").read_text(encoding="utf-8")
    assert not (inbox / run / "labels" / f"{fb}.txt").exists()
    redo = json.loads((inbox / run / "redo.json").read_text(encoding="utf-8"))
    assert redo == {fa: {"name": "a", "split": "train"}, fb: {"name": "b", "split": "val"}}
    assert not (inbox / run / "frames.json").exists()  # 不预写 frames.json：process 会跳过已有条目
    assert pending_runs(inbox) == [run]


def test_process_redo_keeps_split_and_merges_original_labels(tmp_path):
    ds, inbox, attrs_root = tmp_path / "sky", tmp_path / "inbox", tmp_path / "attrs"
    orig = [yolo_line(0, Rect(100, 200, 120, 260), 1920, 1080), yolo_line(1, Rect(90, 150, 140, 30), 1920, 1080)]
    _write_dataset(ds, [("train", "a", _img(40), orig), ("val", "b", _img(40), None)])  # 两张一模一样：回炉帧不去重
    _inbox.add_dataset(ds, inbox)
    run = next(p.name for p in inbox.iterdir() if p.is_dir())
    dets = [_Det("player", Rect(104, 204, 120, 260), 0.9), _Det("player", Rect(900, 300, 100, 240), 0.8)]
    weak = [("social_ring", Rect(1000, 100, 100, 100))]
    judge = lambda f, boxes: [{"lit": 0.95, "not_person": 0.05} for _ in boxes]
    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: dets, lambda f, d: weak, judge, progress=lambda s: None)
    frames = _inbox.load_frames(inbox, run)
    a, b = frames[frame_name(run, "a.jpg")], frames[frame_name(run, "b.jpg")]
    assert a["redo"] == {"name": "a", "split": "train"} and a["split"] == "train" and a["reason"] == "redo"
    assert b["redo"] == {"name": "b", "split": "val"} and b["split"] == "val" and not b["dup_of"]
    got = sorted((x["cls"], tuple(x["box"])) for x in a["boxes"])
    assert got == [("name_tag", (90, 150, 140, 30)), ("player", (100, 200, 120, 260)), ("player", (900, 300, 100, 240)),
                   ("social_ring", (1000, 100, 100, 100))]  # 原标注的人留着（同类重叠的 YOLO 框不再加）
    kept = next(x for x in a["boxes"] if x["box"] == [100, 200, 120, 260])
    assert kept["score"] == 1.0 and kept["src"] == "label" and kept["crop"]  # 原标注的人也过外形头复核
    assert sorted(x["cls"] for x in b["boxes"]) == ["player", "player", "social_ring"]


def _setup_redo(tmp_path, old="0 0.5 0.5 0.1 0.1\n"):
    inbox, ds, attrs_root = tmp_path / "inbox", tmp_path / "sky", tmp_path / "attrs"
    run, frame = "_redo-20261005-200000", "_redo-20261005-200000_a"
    (inbox / run / "raw").mkdir(parents=True)
    im = np.zeros((100, 200, 3), np.uint8)
    cv2.imwrite(str(inbox / run / "raw" / "a.jpg"), im)
    _write_dataset(ds, [("val", "a", im, None)])
    if old is not None:
        (ds / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (ds / "labels" / "val" / "a.txt").write_text(old, encoding="utf-8")
    entry = {"file": "raw/a.jpg", "split": "val", "redo": {"name": "a", "split": "val"}, "editing": False, "error": None,
             "dup_of": None, "decision": None, "boxes": [{"cls": "bench", "box": [100, 50, 50, 20]}]}
    _inbox.save_frames(inbox, run, {frame: entry})
    return inbox, run, frame, attrs_root, ds


def test_pass_redo_overwrites_with_backup_and_undo_restores(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_redo(tmp_path)
    bak = ds / "_backup" / "redo-20261005-200000" / "val" / "a.txt"
    assert _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES) == "val/a"
    assert (ds / "labels" / "val" / "a.txt").read_text(encoding="utf-8") == yolo_line(6, Rect(100, 50, 50, 20), 200, 100) + "\n"
    assert bak.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert not (ds / "images" / "val" / f"{frame}.jpg").exists()  # 写回原帧名，不按收件箱帧名另存一份
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["dataset"] == "val/a"
    _inbox.undo_frame(inbox, run, frame, ds)
    assert (ds / "labels" / "val" / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert (ds / "images" / "val" / "a.jpg").is_file()  # 不删图
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES, boxes=[(3, Rect(0, 0, 100, 50))])
    assert bak.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"  # 同一次回炉的备份已有：不覆盖
    _inbox.undo_frame(inbox, run, frame, ds)
    assert (ds / "labels" / "val" / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"


def test_pass_redo_rewrites_augmented_label_copies_and_undo_restores(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_redo(tmp_path)
    labels = ds / "labels" / "val"
    for s in ("_blur", "_dark"):
        (labels / f"a{s}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels / "a_blur_x.txt").write_text("keep\n", encoding="utf-8")  # 只认 <帧>_blur / _dark
    bak = ds / "_backup" / "redo-20261005-200000" / "val"
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    new = yolo_line(6, Rect(100, 50, 50, 20), 200, 100) + "\n"
    for s in ("", "_blur", "_dark"):
        assert (labels / f"a{s}.txt").read_text(encoding="utf-8") == new
        assert (bak / f"a{s}.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert (labels / "a_blur_x.txt").read_text(encoding="utf-8") == "keep\n"
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["aug"] == ["_blur", "_dark"]
    _inbox.undo_frame(inbox, run, frame, ds)
    for s in ("", "_blur", "_dark"):
        assert (labels / f"a{s}.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES, boxes=[(3, Rect(0, 0, 100, 50))])
    assert (bak / "a_dark.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"  # 备份已有：不覆盖
    _inbox.undo_frame(inbox, run, frame, ds)
    assert (labels / "a_dark.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"


def test_pass_redo_without_augmented_copies_creates_none(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_redo(tmp_path)
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    labels = ds / "labels" / "val"
    assert sorted(p.name for p in labels.iterdir()) == ["a.txt"]
    assert sorted(p.name for p in (ds / "_backup" / "redo-20261005-200000" / "val").iterdir()) == ["a.txt"]
    assert _inbox.load_frames(inbox, run)[frame]["decision"]["aug"] == []
    _inbox.undo_frame(inbox, run, frame, ds)
    assert sorted(p.name for p in labels.iterdir()) == ["a.txt"]


def test_pass_normal_frame_ignores_augmented_names(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_frame(tmp_path)
    (ds / "labels" / "train").mkdir(parents=True)
    (ds / "labels" / "train" / f"{frame}_blur.txt").write_text("old\n", encoding="utf-8")
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert (ds / "labels" / "train" / f"{frame}_blur.txt").read_text(encoding="utf-8") == "old\n"
    assert "aug" not in _inbox.load_frames(inbox, run)[frame]["decision"] and not (ds / "_backup").exists()


def test_undo_redo_without_original_label_removes_label(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_redo(tmp_path, old=None)
    _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert (ds / "labels" / "val" / "a.txt").is_file()
    _inbox.undo_frame(inbox, run, frame, ds)
    assert not (ds / "labels" / "val" / "a.txt").exists() and (ds / "images" / "val" / "a.jpg").is_file()


def test_pass_redo_still_refuses_clash_in_other_split(tmp_path):
    inbox, run, frame, attrs_root, ds = _setup_redo(tmp_path)
    _write_dataset(ds, [("train", "a", np.zeros((100, 200, 3), np.uint8), None)])
    with pytest.raises(FileExistsError):
        _inbox.pass_frame(inbox, run, frame, attrs_root, ds, _CLASSES)
    assert (ds / "labels" / "val" / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"


def test_cli_inbox_add(tmp_path, capsys):
    src = tmp_path / "rec"
    src.mkdir()
    for i in range(4):
        cv2.imwrite(str(src / f"{i}.jpg"), _img(40 + i))
    (tmp_path / "config.toml").write_text(f'[inbox]\ndir = "{(tmp_path / "inbox").as_posix()}"\n', encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "add", str(src), "--every", "2"])
    out = capsys.readouterr().out
    assert "导入了 2 张" in out and "perception inbox process" in out
    ds = tmp_path / "sky"
    _write_dataset(ds, [("train", "a", _img(40), [])])
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "add", str(ds), "--redo"])
    assert "导入了 1 张" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "add", str(src), "--redo"])  # 不是数据集


def test_process_redo_reuses_dataset_crop_name(tmp_path):
    """回炉帧的人物裁图按原帧名命名：datasets/sky 导进 datasets/attrs 的同一个人已经有裁图，就不再出第二张。
    标注是任意小数（手标的），两边换算成像素框要一致，裁图名才对得上。"""
    from skydango.vision import attrs_data
    from skydango.vision.attrs_data import crop_name

    ds, inbox, attrs_root = tmp_path / "sky", tmp_path / "inbox", tmp_path / "attrs"
    line = "0 0.326553 0.777606 0.013120 0.284935"  # 两种换算差一个像素的框
    _write_dataset(ds, [("train", "a", _img(40), [line])])

    class NoDet:
        def detect(self, img):
            return []

    attrs_data.crops_from_dataset(ds, NoDet(), attrs_root, 0.5)
    before = [r["crop"] for r in attrs_data._read_rows(attrs_root)]
    assert len(before) == 1 and before[0].startswith("a__")
    _inbox.add_dataset(ds, inbox)
    run = next(p.name for p in inbox.iterdir() if p.is_dir())
    judge = lambda f, boxes: [{"lit": 0.95, "not_person": 0.05} for _ in boxes]
    _inbox.process(inbox, attrs_root, tmp_path / "runs", _cfg(), lambda f: [], lambda f, d: [], judge, progress=lambda s: None)
    after = [r["crop"] for r in attrs_data._read_rows(attrs_root)]
    assert after == before  # 没有第二张
    a = _inbox.load_frames(inbox, run)[frame_name(run, "a.jpg")]
    person = next(x for x in a["boxes"] if x["cls"] == "player")
    box = attrs_data._label_boxes(ds / "labels" / "train" / "a.txt", 1920, 1080)[0][1]
    assert person["crop"] == before[0] == crop_name("a", box)
