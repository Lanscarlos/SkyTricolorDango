import json
import urllib.parse

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
