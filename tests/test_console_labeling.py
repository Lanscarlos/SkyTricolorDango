import json
import urllib.parse
from pathlib import Path

import pytest

from skydango.console.labeling import GestureLabels
from test_console_server import GOOD, make_server, request, upstream  # noqa: F401

LABELS = ["none", "wave", "bow", "cheer", "shy"]
CLIP = "gesture-挥手-1__0001_track1_t0.00s"


def make_clip(root, where="_unlabeled", clip=CLIP, guess=True):
    d = root / where / clip
    d.mkdir(parents=True)
    for i in range(16):
        (d / f"{i:02d}.jpg").write_bytes(b"\xff\xd8fake" + bytes([i]))
    if guess:
        (d / "claude.json").write_text(json.dumps({"label": "wave", "confidence": 0.8, "reason": "挥手"}), encoding="utf-8")
    return d


def log_lines(root):
    return [json.loads(x) for x in (root / "_labels.jsonl").read_text("utf-8").splitlines()]


def test_state_missing_dir(tmp_path):
    s = GestureLabels(tmp_path / "nope", LABELS).state()
    assert s["ok"] is False


def test_state_lists_clips_with_guess(tmp_path):
    make_clip(tmp_path)
    make_clip(tmp_path, "wave", "other__0002", guess=False)
    s = GestureLabels(tmp_path, LABELS).state()
    assert s["ok"] and s["counts"]["_unlabeled"] == 1 and s["counts"]["wave"] == 1
    c = next(x for x in s["clips"] if x["clip"] == CLIP)
    assert c["where"] == "_unlabeled" and c["recording"] == "gesture-挥手-1"
    assert c["guess"] == {"label": "wave", "confidence": 0.8, "reason": "挥手"}
    assert next(x for x in s["clips"] if x["where"] == "wave")["guess"] is None


def test_state_prefers_blind_guess(tmp_path):
    d = make_clip(tmp_path)
    (d / "claude-blind.json").write_text(json.dumps({"label": "none", "confidence": 0.6, "reason": "站着"}), encoding="utf-8")
    c = GestureLabels(tmp_path, LABELS).state()["clips"][0]
    assert c["guess"] == {"label": "none", "confidence": 0.6, "reason": "站着"}


def test_label_moves_and_logs(tmp_path):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    code, item = g.label(CLIP, "wave")
    assert code == 200 and item["where"] == "wave"
    assert (tmp_path / "wave" / CLIP).is_dir() and not (tmp_path / "_unlabeled" / CLIP).exists()
    assert log_lines(tmp_path)[-1]["to"] == "wave"


def test_relabel_and_discard(tmp_path):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    for to, where in (("wave", "wave"), ("bow", "bow"), ("discard", "_discard")):
        assert g.label(CLIP, to)[0] == 200
        assert (tmp_path / where / CLIP).is_dir()
    assert g.label(CLIP, "nonsense")[0] == 400


def test_undo_restores(tmp_path):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    g.label(CLIP, "wave")
    g.label(CLIP, "bow")
    assert g.undo()[0] == 200 and (tmp_path / "wave" / CLIP).is_dir()
    assert g.undo()[0] == 200 and (tmp_path / "_unlabeled" / CLIP).is_dir()  # 跳过已撤销的
    assert g.undo()[0] == 409
    assert len(log_lines(tmp_path)) == 4 and log_lines(tmp_path)[-1]["undo"] is True


def test_label_twice_is_conflict(tmp_path):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    assert g.label(CLIP, "wave")[0] == 200
    assert g.label(CLIP, "wave")[0] == 409
    assert len(log_lines(tmp_path)) == 1


def test_undo_after_manual_move_is_refused(tmp_path):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    g.label(CLIP, "wave")
    (tmp_path / "wave" / CLIP).rename(tmp_path / "bow" / CLIP) if (tmp_path / "bow").mkdir() is None else None
    assert g.undo()[0] == 409
    assert len(log_lines(tmp_path)) == 1


def test_frame_rejects_traversal(tmp_path):
    make_clip(tmp_path)
    (tmp_path / "secret").mkdir()
    g = GestureLabels(tmp_path, LABELS)
    assert g.frame("../secret", 0) is None and g.frame("..\\secret", 0) is None and g.frame("_unlabeled/" + CLIP, 0) is None
    assert g.frame(CLIP, 16) is None and g.frame(CLIP, -1) is None and g.frame(CLIP, 0)[:2] == b"\xff\xd8"


@pytest.fixture
def srv_with_gesture(tmp_path, upstream):  # noqa: F811
    data = tmp_path / "gdata"
    make_clip(data)
    (tmp_path / "config.toml").write_text(f'[gesture]\ndataset = "{data.as_posix()}"\n', encoding="utf-8")
    s = make_server(tmp_path, upstream)
    yield s
    s.stop()


def test_frame_with_chinese_clip_name(srv_with_gesture):
    import urllib.request

    url = srv_with_gesture.url + "api/gesture/frame?clip=" + urllib.parse.quote(CLIP) + "&i=3"
    with urllib.request.urlopen(url, timeout=10) as r:
        assert r.status == 200 and r.headers["Content-Type"] == "image/jpeg" and r.read()[:2] == b"\xff\xd8"
    assert request(srv_with_gesture.url + "api/gesture/frame?clip=nope&i=0")[0] == 404
    assert request(srv_with_gesture.url + "api/gesture/frame?clip=" + urllib.parse.quote(CLIP) + "&i=x")[0] == 404


def test_api_requires_local_host_and_post_guard(srv_with_gesture):
    u = srv_with_gesture.url
    st, d = request(u + "api/gesture/state")
    assert st == 200 and d["ok"] and d["counts"]["_unlabeled"] == 1
    assert request(u + "api/gesture/state", headers={"Host": f"evil.com:{srv_with_gesture.port}"})[0] == 403
    body = json.dumps({"clip": CLIP, "to": "wave"}).encode()
    assert request(u + "api/gesture/label", body, {"Content-Type": "application/json"})[0] == 403
    st, d = request(u + "api/gesture/label", body, GOOD)
    assert st == 200 and d["where"] == "wave"
    assert request(u + "api/gesture/label", body, GOOD)[0] == 409
    st, d = request(u + "api/gesture/undo", b"{}", GOOD)
    assert st == 200 and d["where"] == "_unlabeled"


def test_concurrent_label_exactly_one_wins(tmp_path):
    import threading

    make_clip(tmp_path)
    codes = []

    def go():
        codes.append(GestureLabels(tmp_path, LABELS).label(CLIP, "wave")[0])

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(codes) == [200] + [409] * 7 and len(log_lines(tmp_path)) == 1


def test_concurrent_label_and_undo_stay_consistent(tmp_path):
    import threading

    make_clip(tmp_path)
    GestureLabels(tmp_path, LABELS).label(CLIP, "wave")
    ts = [threading.Thread(target=lambda: GestureLabels(tmp_path, LABELS).undo()) for _ in range(4)]
    ts.append(threading.Thread(target=lambda: GestureLabels(tmp_path, LABELS).label(CLIP, "bow")))
    [t.start() for t in ts]
    [t.join() for t in ts]
    where = [d for d in ("_unlabeled", "wave", "bow") if (tmp_path / d / CLIP).is_dir()]
    assert len(where) == 1  # 片段只在一处
    stack = []
    for e in log_lines(tmp_path):
        stack.pop() if e.get("undo") else stack.append(e)
    assert (stack[-1]["to"] if stack else "_unlabeled") == where[0]  # 记录和目录一致


def test_log_failure_moves_clip_back(tmp_path, monkeypatch):
    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    monkeypatch.setattr(g, "_log", lambda e: (_ for _ in ()).throw(OSError("disk")))
    assert g.label(CLIP, "wave")[0] == 500
    assert (tmp_path / "_unlabeled" / CLIP).is_dir() and not (tmp_path / "wave" / CLIP).exists()


def test_failed_rename_does_not_copy(tmp_path, monkeypatch):
    """改名失败（比如文件被占用）就是挪不动：不能退回 复制 + 删除，免得片段两边都有。"""
    import os

    make_clip(tmp_path)
    g = GestureLabels(tmp_path, LABELS)
    real = os.rename

    def fail(src, dst, *a, **k):
        if CLIP in str(src):
            raise PermissionError("in use")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "rename", fail)
    code, body = g.label(CLIP, "wave")
    assert code == 409 and "挪不动" in body["text"]
    assert (tmp_path / "_unlabeled" / CLIP).is_dir() and not (tmp_path / "wave" / CLIP).exists()
    assert not (tmp_path / "_labels.jsonl").exists()


# ---- 外形页（FormLabels） ----
CROP = "c0001.jpg"


def make_crop(root, where="_unlabeled", name=CROP, guess=None, image=None, box=(10, 20, 30, 40)):
    d = root / where
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"\xff\xd8crop")
    if guess:
        p = root / "_unlabeled" / "claude.json"
        old = json.loads(p.read_text("utf-8")) if p.exists() else {}
        old[name] = guess
        p.write_text(json.dumps(old), encoding="utf-8")
    with (root / "_crops.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"crop": name, "image": image, "box": list(box)}) + "\n")


def test_form_state_lists_items(tmp_path):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, image="/x/a.jpg")
    make_crop(tmp_path, "form/unlit", "c0002.jpg")
    (tmp_path / "_unlabeled" / "claude.json").write_text(
        json.dumps({CROP: {"label": "lit", "confidence": 0.9, "reason": "有光"}, "gone.jpg": {"label": "lit"}}), encoding="utf-8")
    s = FormLabels(tmp_path).state()
    assert s["ok"] and s["counts"]["_unlabeled"] == 1 and s["counts"]["unlit"] == 1 and s["counts"]["lit"] == 0
    assert [i["crop"] for i in s["items"]] == [CROP, "c0002.jpg"]  # claude.json 不算裁图、消失的裁图不列
    a = s["items"][0]
    assert a["where"] == "_unlabeled" and a["guess"] == {"label": "lit", "confidence": 0.9, "reason": "有光"}
    assert a["image"] == "/x/a.jpg" and a["box"] == [10, 20, 30, 40]
    assert s["items"][1]["where"] == "unlit" and s["items"][1]["guess"] is None
    assert FormLabels(tmp_path / "nope").state()["ok"] is False


def test_form_state_buttons_skip_shared_and_morph(tmp_path):
    # 10-04 用户：认人阶段只分 不是人 / 点亮的人 / 黑影 / 先祖（先祖和共享空间长得像，混进不是人反而难学，以后图鉴也要），
    # 共享空间 / 变身不出按钮（按点没点火标；训练时本来就并进 lit）；筛选照旧列全部类别，以前标进去的还找得到
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, "form/shared", "c0003.jpg")
    s = FormLabels(tmp_path).state()
    assert s["buttons"] == ["not_person", "lit", "unlit", "spirit"]
    assert s["forms"] == ["not_person", "lit", "unlit", "spirit", "shared", "morph"] and s["counts"]["shared"] == 1


def test_form_label_relabel_discard_undo(tmp_path):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path)
    f = FormLabels(tmp_path)
    code, item = f.label(CROP, "lit")
    assert code == 200 and item["where"] == "lit" and (tmp_path / "form" / "lit" / CROP).is_file()
    assert not (tmp_path / "_unlabeled" / CROP).exists()
    assert f.label(CROP, "morph")[0] == 200 and (tmp_path / "form" / "morph" / CROP).is_file()
    assert f.label(CROP, "discard")[0] == 200 and (tmp_path / "_discard" / CROP).is_file()
    assert f.label(CROP, "discard")[0] == 409
    assert f.label(CROP, "nonsense")[0] == 400
    first = log_lines(tmp_path)[0]
    assert (first["crop"], first["from"], first["to"]) == (CROP, "_unlabeled", "lit")
    assert f.undo()[1]["where"] == "morph" and (tmp_path / "form" / "morph" / CROP).is_file()
    assert f.undo()[1]["where"] == "lit"
    assert f.undo()[1]["where"] == "_unlabeled" and (tmp_path / "_unlabeled" / CROP).is_file()
    assert f.undo()[0] == 409
    assert len(log_lines(tmp_path)) == 6 and log_lines(tmp_path)[-1]["undo"] is True


def test_form_unknown_names_are_404_without_fs_access(tmp_path, monkeypatch):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path)
    (tmp_path / "secret.jpg").write_bytes(b"x")
    f = FormLabels(tmp_path)
    for bad in ("../secret.jpg", "..\\secret.jpg", "_unlabeled/" + CROP, "con.jpg", "nul.jpg", "claude.json", ""):
        assert f.crop(bad) is None and f.context(bad) is None
        assert f.label(bad, "lit")[0] == 409
    assert f.crop(CROP) == b"\xff\xd8crop"
    opened = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (opened.append(self.name), real(self))[1])
    f.crop("con.jpg")
    assert opened == []


def test_form_context_draws_box_and_scales(tmp_path):
    import cv2
    import numpy as np

    from skydango.console.labeling import FormLabels

    src = tmp_path / "frame.png"
    cv2.imwrite(str(src), np.full((540, 960, 3), 128, np.uint8))
    make_crop(tmp_path, image=src.as_posix(), box=(100, 100, 200, 200))
    make_crop(tmp_path, name="c0002.jpg", image=(tmp_path / "missing.png").as_posix())
    make_crop(tmp_path, name="c0003.jpg", image=None)
    f = FormLabels(tmp_path)
    out = cv2.imdecode(np.frombuffer(f.context(CROP), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape[:2] == (270, 480)
    assert out[50, 70].tolist() != [128, 128, 128]  # 框的上边在 y≈50（缩放 0.5）
    assert out[130, 130].tolist() == [128, 128, 128]
    assert f.context("c0002.jpg") is None and f.context("c0003.jpg") is None


def test_form_log_failure_moves_back(tmp_path, monkeypatch):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path)
    f = FormLabels(tmp_path)
    monkeypatch.setattr(f, "_log", lambda e: (_ for _ in ()).throw(OSError("disk")))
    assert f.label(CROP, "lit")[0] == 500
    assert (tmp_path / "_unlabeled" / CROP).is_file() and not (tmp_path / "form" / "lit" / CROP).exists()


def test_form_confirm_in_place_and_confirmed_flag(tmp_path):
    # 10-04：datasets/sky 导进来的裁图直接在 form/<类别>/ 里、没人看过（标签混了不少错的）。
    # 标注页"导入未确认"筛选靠 confirmed 区分；在原类别上再标一次 = 原地确认（记一条 from == to），不挪文件
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, "form/lit", "imp.jpg")
    make_crop(tmp_path)
    f = FormLabels(tmp_path)
    items = {i["crop"]: i for i in f.state()["items"]}
    assert items["imp.jpg"]["confirmed"] is False and items[CROP]["confirmed"] is False
    code, item = f.label("imp.jpg", "lit")
    assert code == 200 and item["where"] == "lit" and item["confirmed"] is True
    assert (tmp_path / "form" / "lit" / "imp.jpg").is_file()
    e = log_lines(tmp_path)[-1]
    assert (e["crop"], e["from"], e["to"]) == ("imp.jpg", "lit", "lit")
    assert f.label("imp.jpg", "lit")[0] == 409  # 已经确认过了
    assert f.label(CROP, "unlit")[1]["confirmed"] is True  # 从待确认挪过去的也算确认
    items = {i["crop"]: i for i in f.state()["items"]}
    assert items["imp.jpg"]["confirmed"] and items[CROP]["confirmed"]
    assert f.undo()[1]["where"] == "_unlabeled"
    code, body = f.undo()  # 撤销原地确认：不挪文件，回到没确认
    assert code == 200 and body["where"] == "lit" and body["confirmed"] is False
    assert (tmp_path / "form" / "lit" / "imp.jpg").is_file()
    assert f.label("imp.jpg", "lit")[0] == 200
    assert f.label("imp.jpg", "unlit")[1]["confirmed"] is True  # 确认后又改类别
    code, body = f.undo()  # 撤销改类别：回到原地确认过的 lit
    assert body["where"] == "lit" and body["confirmed"] is True


def test_form_items_flag_replay_crops(tmp_path):
    # 「导入未确认 · 回放用」筛选：数据集验证集帧里的框是 attrs-train 整帧回放的标准答案，先过它们回放才算得准
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, "form/lit", "v.jpg")
    make_crop(tmp_path, "form/lit", "t.jpg")
    make_crop(tmp_path, "form/lit", "h.jpg")
    rows = [{"crop": "v.jpg", "source": "dataset", "split": "val"}, {"crop": "t.jpg", "source": "dataset", "split": "train"},
            {"crop": "h.jpg", "source": "images", "split": None}]
    (tmp_path / "_crops.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = {i["crop"]: i for i in FormLabels(tmp_path).state()["items"]}
    assert items["v.jpg"]["replay"] is True and items["t.jpg"]["replay"] is False and items["h.jpg"]["replay"] is False


def test_form_discard_in_place_is_still_conflict(tmp_path):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, "_discard")
    assert FormLabels(tmp_path).label(CROP, "discard")[0] == 409


def test_form_confirm_in_place_log_failure(tmp_path, monkeypatch):
    from skydango.console.labeling import FormLabels

    make_crop(tmp_path, "form/lit")
    f = FormLabels(tmp_path)
    monkeypatch.setattr(f, "_log", lambda e: (_ for _ in ()).throw(OSError("disk")))
    assert f.label(CROP, "lit")[0] == 500
    assert (tmp_path / "form" / "lit" / CROP).is_file()


def test_form_api_routes(tmp_path, upstream, monkeypatch):  # noqa: F811
    import urllib.request

    make_crop(tmp_path / "datasets" / "attrs")
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = make_server(tmp_path, upstream)
    try:
        u = s.url
        st, d = request(u + "api/form/state")
        assert st == 200 and d["ok"] and d["counts"]["_unlabeled"] == 1
        with urllib.request.urlopen(u + "api/form/crop?name=" + CROP, timeout=10) as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read() == b"\xff\xd8crop"
        assert request(u + "api/form/crop?name=..%2Fx.jpg")[0] == 404
        assert request(u + "api/form/crop?name=con.jpg")[0] == 404
        assert request(u + "api/form/context?name=" + CROP)[0] == 404  # 没有原图
        assert request(u + "api/form/state", headers={"Host": f"evil.com:{s.port}"})[0] == 403
        body = json.dumps({"name": CROP, "to": "lit"}).encode()
        assert request(u + "api/form/label", body, {"Content-Type": "application/json"})[0] == 403
        st, d = request(u + "api/form/label", body, GOOD)
        assert st == 200 and d["where"] == "lit"
        assert request(u + "api/form/label", body, GOOD)[0] == 409
        st, d = request(u + "api/form/undo", b"{}", GOOD)
        assert st == 200 and d["where"] == "_unlabeled"
    finally:
        s.stop()
