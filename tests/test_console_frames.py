import json

import cv2
import numpy as np
import pytest

from skydango.console.frames import FramesApi
from skydango.vision import inbox as ib
from skydango.vision.bubbles import Rect
from skydango.vision.weaklabel import yolo_line
from test_console_server import GOOD, make_server, request, upstream  # noqa: F401

CLASSES = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]
FRAME = "r1_a"


def make(tmp_path, boxes=None, split="train", files=("a.jpg",)):
    inbox, attrs, ds = tmp_path / "inbox", tmp_path / "attrs", tmp_path / "sky"
    (inbox / "r1" / "raw").mkdir(parents=True)
    entries = {}
    for f in files:
        cv2.imwrite(str(inbox / "r1" / "raw" / f), np.full((100, 200, 3), 80, np.uint8))
        entries[f"r1_{f[:-4]}"] = {
            "file": f"raw/{f}", "reason": "low_conf", "split": split, "dup_of": None, "editing": False, "error": None, "decision": None,
            "boxes": boxes if boxes is not None else [
                {"cls": "player", "box": [20, 10, 40, 60], "score": 0.9, "src": "yolo", "crop": "c1.jpg", "auto": "agree"},
                {"cls": "bench", "box": [100, 50, 50, 20], "score": 0.8, "src": "yolo", "crop": None, "auto": None}]}
    ib.save_frames(inbox, "r1", entries)
    return FramesApi(inbox, attrs, ds, CLASSES), inbox, attrs, ds


def put_crop(attrs, where, name):
    d = attrs / where
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"x")


def test_state_counts_and_boxes(tmp_path):
    boxes = [
        {"cls": "player", "box": [1, 1, 10, 10], "score": 0.9, "crop": "agree.jpg", "auto": "agree"},
        {"cls": "player", "box": [2, 2, 10, 10], "score": 0.3, "crop": None, "auto": "drop_low"},
        {"cls": "player", "box": [3, 3, 10, 10], "score": 0.9, "crop": "gone.jpg", "auto": None},
        {"cls": "player", "box": [4, 4, 10, 10], "score": 0.9, "crop": "unlit.jpg", "auto": None},
        {"cls": "player", "box": [5, 5, 10, 10], "score": 0.9, "crop": "pending.jpg", "auto": None},
        {"cls": "bench", "box": [6, 6, 10, 10], "score": 0.9, "crop": None, "auto": None}]
    api, inbox, attrs, _ = make(tmp_path, boxes)
    put_crop(attrs, "form/lit", "agree.jpg")
    put_crop(attrs, "_discard", "gone.jpg")
    put_crop(attrs, "form/unlit", "unlit.jpg")
    put_crop(attrs, "_unlabeled", "pending.jpg")
    s = api.state()
    f = s["frames"][0]
    assert s["counts"] == {"crops": 1} and f["state"] == "crops" and f["w"] == 200 and f["h"] == 100
    assert f["frame"] == FRAME and f["run"] == "r1" and f["reason"] == "low_conf" and f["split"] == "train"
    got = [(b["cls"], b["box"][0], b["src"], b["dropped"]) for b in f["boxes"]]
    assert got == [(0, 1, "auto", False), (0, 2, "auto", True), (4, 4, "judged", False), (0, 5, "yolo", False), (6, 6, "yolo", False)]


def test_state_skips_dups_and_counts_glance(tmp_path):
    api, inbox, attrs, _ = make(tmp_path, files=("a.jpg", "b.jpg"))
    fr = ib.load_frames(inbox, "r1")
    fr["r1_b"]["dup_of"] = "r1_a"
    ib.save_frames(inbox, "r1", fr)
    s = api.state()
    assert [f["frame"] for f in s["frames"]] == ["r1_a"] and s["counts"] == {"glance": 1}


def test_crop_move_changes_state(tmp_path):
    api, inbox, attrs, _ = make(tmp_path)
    put_crop(attrs, "_unlabeled", "c1.jpg")
    assert api.state()["frames"][0]["state"] == "crops"
    (attrs / "form" / "lit").mkdir(parents=True)
    (attrs / "_unlabeled" / "c1.jpg").replace(attrs / "form" / "lit" / "c1.jpg")
    assert api.state()["frames"][0]["state"] == "glance"


@pytest.mark.parametrize("name", ["r1_zzz", "../x", "C:/x.jpg", "/etc/passwd", ""])
def test_image_only_registered(tmp_path, name):
    api, *_ = make(tmp_path)
    assert api.image(name) is None
    assert api.image(FRAME) == tmp_path / "inbox" / "r1" / "raw" / "a.jpg"


def test_image_refuses_path_outside_inbox(tmp_path):
    api, inbox, *_ = make(tmp_path)
    fr = ib.load_frames(inbox, "r1")
    fr[FRAME]["file"] = "../../outside.jpg"
    ib.save_frames(inbox, "r1", fr)
    (tmp_path / "outside.jpg").write_bytes(b"x")
    assert api.image(FRAME) is None


def test_act_pass_discard_undo(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    code, body = api.act({"frame": FRAME, "do": "pass"})
    assert code == 200 and (ds / "labels" / "train" / f"{FRAME}.txt").is_file()
    assert api.state()["frames"][0]["state"] == "done"
    assert api.act({"frame": FRAME, "do": "pass"})[0] == 409  # 已通过再通过 = 同名
    assert api.act({"frame": FRAME, "do": "undo"})[0] == 200
    assert not (ds / "labels" / "train" / f"{FRAME}.txt").exists()
    assert api.act({"frame": FRAME, "do": "discard"})[0] == 200
    assert api.state()["frames"][0]["state"] == "discarded"
    assert api.act({"frame": FRAME, "do": "undo"})[0] == 200
    assert api.state()["frames"][0]["state"] == "glance"


def test_act_edit_cancel_and_bad_requests(tmp_path):
    api, *_ = make(tmp_path)
    assert api.act({"frame": FRAME, "do": "edit"})[0] == 200
    assert api.state()["frames"][0]["state"] == "edit"
    assert api.act({"frame": FRAME, "do": "cancel_edit"})[0] == 200
    assert api.state()["frames"][0]["state"] == "glance"
    assert api.act({"frame": "nope", "do": "pass"})[0] == 404
    assert api.act({"frame": FRAME, "do": "zap"})[0] == 400
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 99, "box": [1, 1, 20, 20]}]})[0] == 400
    assert api.act({"frame": FRAME, "do": "pass", "boxes": "x"})[0] == 400


def test_edit_save_normalizes(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    boxes = [{"cls": 6, "box": [100, 100, -50, -60]}, {"cls": 6, "box": [150, 10, 100, 50]}, {"cls": 6, "box": [0, 0, 3, 50]}]
    assert api.act({"frame": FRAME, "do": "pass", "boxes": boxes})[0] == 200
    lines = (ds / "labels" / "train" / f"{FRAME}.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [yolo_line(6, Rect(50, 40, 50, 60), 200, 100), yolo_line(6, Rect(150, 10, 50, 50), 200, 100)]
    assert ib.load_frames(inbox, "r1")[FRAME]["decision"]["edited"] is True


def test_edit_save_crops_new_and_reclassified_people(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    boxes = [{"cls": 0, "box": [20, 10, 40, 60]},        # 原框，没变：不裁
             {"cls": 4, "box": [120, 10, 40, 60]},        # 新画的黑影
             {"cls": 9, "box": [20, 10, 40, 60]},         # 同位置改成先祖：类别变了，裁
             {"cls": 3, "box": [0, 0, 30, 30]}]           # 非人物：不裁
    assert api.act({"frame": FRAME, "do": "pass", "boxes": boxes})[0] == 200
    assert [p.name for p in (attrs / "form" / "unlit").iterdir()] == [f"{FRAME}__120_10_40_60.jpg"]
    assert [p.name for p in (attrs / "form" / "spirit").iterdir()] == [f"{FRAME}__20_10_40_60.jpg"]
    assert not (attrs / "form" / "lit").exists()
    rows = [json.loads(x) for x in (attrs / "_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(r["to"], r["by"], r["from"]) for r in rows} == {("unlit", "frame-edit", "_unlabeled"), ("spirit", "frame-edit", "_unlabeled")}
    crops = [json.loads(x) for x in (attrs / "_crops.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(c["source"] == "inbox" and c["group"] == "r1" and c["known"] is False for c in crops)
    assert {c["yolo_cls"] for c in crops} == {"player_unlit", "spirit"}


def test_edit_save_clash_is_409(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "train" / f"{FRAME}.jpg").write_bytes(b"x")
    code, body = api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 4, "box": [120, 10, 40, 60]}]})
    assert code == 409 and body["text"] == "datasets/sky 里已经有同名的帧"
    assert not (attrs / "form").exists()  # 没通过就不留裁图


def test_frames_api_routes(tmp_path, upstream, monkeypatch):  # noqa: F811
    import urllib.request

    inbox = tmp_path / "datasets" / "inbox"
    (inbox / "r1" / "raw").mkdir(parents=True)
    cv2.imwrite(str(inbox / "r1" / "raw" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    ib.save_frames(inbox, "r1", {FRAME: {"file": "raw/a.jpg", "split": "train", "dup_of": None, "editing": False, "error": None,
                                         "decision": None, "boxes": []}})
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = make_server(tmp_path, upstream)
    try:
        u = s.url
        st, d = request(u + "api/frames/state")
        assert st == 200 and d["counts"] == {"glance": 1} and d["frames"][0]["frame"] == FRAME
        with urllib.request.urlopen(u + "api/frames/image?frame=" + FRAME, timeout=10) as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read()[:2] == b"\xff\xd8"
        assert request(u + "api/frames/image?frame=..%2Fx")[0] == 404
        assert request(u + "api/frames/state", headers={"Host": f"evil.com:{s.port}"})[0] == 403
        body = json.dumps({"frame": FRAME, "do": "discard"}).encode()
        assert request(u + "api/frames/act", body, {"Content-Type": "application/json"})[0] == 403
        assert request(u + "api/frames/act", body, GOOD)[0] == 200
        assert request(u + "api/frames/state")[1]["counts"] == {"discarded": 1}
    finally:
        s.stop()