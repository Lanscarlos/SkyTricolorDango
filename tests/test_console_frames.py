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



def test_state_follows_reclassified_auto_agree_crop(tmp_path):
    # 终审 3：auto-agree 的裁图在外形页被改判 / 判不要，state() 显示的框跟着变（不再固定用 YOLO 类别）
    api, inbox, attrs, _ = make(tmp_path)
    put_crop(attrs, "form/lit", "c1.jpg")
    f = api.state()["frames"][0]
    assert [(b["cls"], b["src"]) for b in f["boxes"]] == [(0, "auto"), (6, "yolo")]
    (attrs / "form" / "unlit").mkdir(parents=True)
    (attrs / "form" / "lit" / "c1.jpg").replace(attrs / "form" / "unlit" / "c1.jpg")
    f = api.state()["frames"][0]
    assert [(b["cls"], b["src"]) for b in f["boxes"]] == [(4, "judged"), (6, "yolo")]
    (attrs / "_discard").mkdir(parents=True)
    (attrs / "form" / "unlit" / "c1.jpg").replace(attrs / "_discard" / "c1.jpg")
    f = api.state()["frames"][0]
    assert f["state"] == "edit" and [b["cls"] for b in f["boxes"]] == [6]


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
    put_crop(attrs, "form/lit", "c1.jpg")  # 原框（auto agree）的裁图
    boxes = [{"cls": 0, "box": [20, 10, 40, 60]},        # 原框，没变：不裁
             {"cls": 4, "box": [120, 10, 40, 60]},        # 新画的黑影
             {"cls": 9, "box": [150, 40, 40, 50]},        # 新画的先祖
             {"cls": 3, "box": [0, 0, 30, 30]}]           # 非人物：不裁
    assert api.act({"frame": FRAME, "do": "pass", "boxes": boxes})[0] == 200
    assert [p.name for p in (attrs / "form" / "unlit").iterdir()] == [f"{FRAME}__120_10_40_60.jpg"]
    assert [p.name for p in (attrs / "form" / "spirit").iterdir()] == [f"{FRAME}__150_40_40_50.jpg"]
    assert [p.name for p in (attrs / "form" / "lit").iterdir()] == ["c1.jpg"]
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

def _crop_row(attrs, name, yolo="player"):
    with (attrs / "_crops.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"crop": name, "image": "x", "source": "inbox", "yolo_cls": yolo}) + "\n")


def _labels(attrs):
    return [json.loads(x) for x in (attrs / "_labels.jsonl").read_text(encoding="utf-8").splitlines()]


def test_edit_moves_existing_crop_when_class_changes(tmp_path):
    name = f"{FRAME}__20_10_40_60.jpg"
    boxes = [{"cls": "player", "box": [20, 10, 40, 60], "score": 0.9, "crop": name, "auto": "agree"}]
    api, inbox, attrs, ds = make(tmp_path, boxes)
    put_crop(attrs, "form/lit", name)
    _crop_row(attrs, name)
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 9, "box": [20, 10, 40, 60]}]})[0] == 200
    assert not (attrs / "form" / "lit" / name).exists() and (attrs / "form" / "spirit" / name).is_file()
    row = _labels(attrs)[-1]
    assert (row["crop"], row["from"], row["to"], row["by"]) == (name, "lit", "spirit", "frame-edit")


def test_edit_compares_against_judged_class(tmp_path):
    name = f"{FRAME}__20_10_40_60.jpg"
    boxes = [{"cls": "player", "box": [20, 10, 40, 60], "score": 0.9, "crop": name, "auto": None}]
    api, inbox, attrs, ds = make(tmp_path, boxes)
    put_crop(attrs, "form/unlit", name)  # 人在外形页判成黑影
    _crop_row(attrs, name)
    # 编辑里改回 player(0)：和展示的黑影不同，要把裁图挪回 lit
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 0, "box": [20, 10, 40, 60]}]})[0] == 200
    assert (attrs / "form" / "lit" / name).is_file() and not (attrs / "form" / "unlit" / name).exists()
    assert _labels(attrs)[-1]["from"] == "unlit"


def test_edit_person_to_non_person_moves_crop_to_not_person(tmp_path):
    name = f"{FRAME}__20_10_40_60.jpg"
    boxes = [{"cls": "player", "box": [20, 10, 40, 60], "score": 0.9, "crop": name, "auto": "agree"}]
    api, inbox, attrs, ds = make(tmp_path, boxes)
    put_crop(attrs, "form/lit", name)
    _crop_row(attrs, name)
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 7, "box": [20, 10, 40, 60]}]})[0] == 200
    assert (attrs / "form" / "not_person" / name).is_file()
    assert _labels(attrs)[-1]["to"] == "not_person"


def test_edit_unchanged_class_leaves_crop(tmp_path):
    name = f"{FRAME}__20_10_40_60.jpg"
    boxes = [{"cls": "player", "box": [20, 10, 40, 60], "score": 0.9, "crop": name, "auto": "agree"}]
    api, inbox, attrs, ds = make(tmp_path, boxes)
    put_crop(attrs, "form/lit", name)
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 0, "box": [22, 12, 40, 60]}]})[0] == 200
    assert (attrs / "form" / "lit" / name).is_file() and not (attrs / "_labels.jsonl").exists()


def test_concurrent_acts_on_separate_instances_do_not_lose_decisions(tmp_path):
    import threading

    api, inbox, attrs, ds = make(tmp_path, files=("a.jpg", "b.jpg"))
    codes = []

    def go(frame, do):
        codes.append(FramesApi(inbox, attrs, ds, CLASSES).act({"frame": frame, "do": do})[0])

    # 每个请求现建一个实例，两个操作同时来
    ts = [threading.Thread(target=go, args=(f, d)) for f, d in (("r1_a", "pass"), ("r1_b", "discard")) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    fr = ib.load_frames(inbox, "r1")
    assert fr["r1_a"]["decision"]["what"] == "pass" and fr["r1_b"]["decision"]["what"] == "discard"


def test_discard_and_edit_refused_on_passed_frame(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    assert api.act({"frame": FRAME, "do": "pass"})[0] == 200
    for do in ("discard", "edit"):
        code, body = api.act({"frame": FRAME, "do": do})
        assert code == 409 and "撤销" in body["text"]
    assert ib.load_frames(inbox, "r1")[FRAME]["decision"]["what"] == "pass"
    assert api.act({"frame": FRAME, "do": "undo"})[0] == 200
    assert api.act({"frame": FRAME, "do": "pass"})[0] == 200


def test_plain_pass_refused_with_pending_crops_or_error(tmp_path):
    api, inbox, attrs, ds = make(tmp_path)
    put_crop(attrs, "_unlabeled", "c1.jpg")
    code, body = api.act({"frame": FRAME, "do": "pass"})
    assert code == 409 and "外形页" in body["text"] and not (ds / "labels").exists()
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 6, "box": [10, 10, 20, 20]}]})[0] == 200  # 编辑保存照样可以
    api.act({"frame": FRAME, "do": "undo"})
    fr = ib.load_frames(inbox, "r1")
    fr[FRAME]["error"] = "X"
    ib.save_frames(inbox, "r1", fr)
    code, body = api.act({"frame": FRAME, "do": "pass"})
    assert code == 409 and "出错" in body["text"]


def test_parse_rejects_infinity_and_state_survives_unknown_class(tmp_path):
    api, inbox, attrs, ds = make(tmp_path, [{"cls": "mystery", "box": [1, 1, 5, 5], "auto": None, "crop": None},
                                            {"cls": "bench", "box": [6, 6, 10, 10], "auto": None, "crop": None}])
    assert api.act({"frame": FRAME, "do": "pass", "boxes": [{"cls": 6, "box": [1, 1, float("inf"), 20]}]})[0] == 400
    assert [b["box"][0] for b in api.state()["frames"][0]["boxes"]] == [6]


# ---------- 重训：最近一次报告、换上 / 回退 ----------

import hashlib  # noqa: E402

from skydango.console.retrain_view import RetrainView  # noqa: E402
from skydango.console.settings import SettingsStore  # noqa: E402


def make_retrain(tmp_path, console_toml="", config_toml="", stamps=("20261005-010000",), result=True):
    (tmp_path / "config.toml").write_text(config_toml, encoding="utf-8")
    if console_toml:
        (tmp_path / "console.toml").write_text(console_toml, encoding="utf-8")
    root = tmp_path / "retrain"
    for s in stamps:
        d = root / s
        d.mkdir(parents=True)
        (d / "report.md").write_text(f"# 报告 {s}", encoding="utf-8")
    new_yolo, new_attrs = tmp_path / "new.pt", tmp_path / "new.npz"
    new_yolo.write_bytes(b"y")
    new_attrs.write_bytes(b"a")
    if result:
        (root / stamps[-1] / "result.json").write_text(
            json.dumps({"ok": True, "yolo": str(new_yolo), "attrs": str(new_attrs)}), encoding="utf-8")
    store = SettingsStore(tmp_path / "config.toml", environ={})
    return RetrainView(root, store), store, root / stamps[-1], new_yolo, new_attrs


def test_latest_reads_newest_dir(tmp_path):
    view, _, d, _, _ = make_retrain(tmp_path, stamps=("20261004-230000", "20261005-010000"))
    (tmp_path / "retrain" / "stray.txt").write_text("x")
    r = view.latest()
    assert r["ok"] and r["dir"] == "20261005-010000" and "报告 20261005" in r["report"]
    assert r["result"]["ok"] is True and r["adopted"] is None


def test_latest_empty(tmp_path):
    view = RetrainView(tmp_path / "none", SettingsStore(tmp_path / "config.toml", environ={}))
    r = view.latest()
    assert r == {"ok": True, "dir": None, "report": "", "result": None, "adopted": None}


def test_adopt_then_rollback_with_override(tmp_path):
    view, store, d, new_yolo, _ = make_retrain(
        tmp_path, console_toml='[perception]\nmodel = "models/old-console.pt"\n', config_toml='[perception]\nmodel = "models/cfg.pt"\n')
    cfg_hash = hashlib.sha256((tmp_path / "config.toml").read_bytes()).hexdigest()
    st, r = view.adopt("yolo", running=True)
    assert st == 200 and r["ok"] and r["note"] == "下次叫醒生效"
    assert store.effective().perception.model == str(new_yolo)
    rec = json.loads((d / "adopt.json").read_text(encoding="utf-8"))["yolo"]
    assert rec["old"] == "models/old-console.pt" and rec["new"] == str(new_yolo) and rec["had_override"] is True and rec["t"]
    assert view.latest()["adopted"]["yolo"]["old"] == "models/old-console.pt"
    st, r = view.rollback("yolo", running=False)
    assert st == 200 and "note" not in r
    assert store.effective().perception.model == "models/old-console.pt"
    assert not (d / "adopt.json").exists() or "yolo" not in json.loads((d / "adopt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((tmp_path / "config.toml").read_bytes()).hexdigest() == cfg_hash


def test_rollback_without_override_reverts(tmp_path):
    view, store, d, new_yolo, new_attrs = make_retrain(tmp_path, config_toml='[attrs]\nmodel = "models/cfg.npz"\n')
    assert view.adopt("attrs", running=False)[0] == 200
    assert store.effective().attrs.model == str(new_attrs)
    assert view.rollback("attrs", running=False)[0] == 200
    assert store.effective().attrs.model == "models/cfg.npz"
    assert "model" not in (tmp_path / "console.toml").read_text(encoding="utf-8")  # 没有钉进 console.toml


def test_adopt_refusals(tmp_path):
    view, store, d, new_yolo, _ = make_retrain(tmp_path, result=False)
    assert view.adopt("yolo", running=False)[0] == 409
    view, store, d, new_yolo, _ = make_retrain(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    new_yolo.unlink()
    st, r = view.adopt("yolo", running=False)
    assert st == 409 and "不存在" in r["text"]
    assert view.adopt("zzz", running=False)[0] == 400
    assert view.rollback("yolo", running=False)[0] == 409  # 没换过


def test_retrain_http(tmp_path, upstream, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    d = tmp_path / "tmp" / "retrain" / "20261005-010000"
    d.mkdir(parents=True)
    (d / "report.md").write_text("# hi", encoding="utf-8")
    s = make_server(tmp_path, upstream)
    try:
        u = s.url
        st, r = request(u + "api/retrain/latest")
        assert st == 200 and r["dir"] == "20261005-010000"
        assert request(u + "api/retrain/latest", headers={"Host": f"evil.com:{s.port}"})[0] == 403
        body = json.dumps({"what": "yolo"}).encode()
        assert request(u + "api/retrain/adopt", body, {"Content-Type": "application/json"})[0] == 403
        assert request(u + "api/retrain/adopt", body, GOOD)[0] == 409  # 没 result.json
    finally:
        s.stop()
