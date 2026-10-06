"""动作片段 Claude 初分（vision/gesture_label.py + perception gesture-label）。不连 Claude。"""

import json
from types import SimpleNamespace

import numpy as np

from model_seams import fake_claude
from skydango.config import AssistConfig, Config
from skydango.imageio import imwrite
from skydango.vision.assist import FrameInput
from skydango.vision.gesture_label import (
    Guess,
    build_gesture_message,
    contact_sheet,
    load_guess,
    parse_gesture_review,
    recording_hint,
    write_guess,
)

SHEET = np.zeros((448, 448, 3), np.uint8)


def test_contact_sheet_size_and_order():
    frames = [np.full((112, 112, 3), i * 10, np.uint8) for i in range(16)]
    s = contact_sheet(frames)
    assert s.shape == (448, 448, 3) and s[60, 60, 0] == 0 and s[60 + 112 * 3, 60 + 112 * 3, 0] == 150


def test_contact_sheet_resizes_and_draws_numbers():
    s = contact_sheet([np.full((200, 150, 3), 100, np.uint8)] * 16)
    assert s.shape == (448, 448, 3) and s[2, 2, 0] == 255  # 左上角白底


def test_recording_hint():
    assert "挥手" in recording_hint("gesture-wave-1") and "没有" in recording_hint("gesture-none-2") and recording_hint("rec") == ""


def test_parse_accepts_known_labels_and_unsure():
    frames = [FrameInput("a__0001_track1_t0.00s", SHEET, []), FrameInput("a__0002_track1_t1.00s", SHEET, [])]
    got = parse_gesture_review('{"a__0001_track1_t0.00s": {"label": "wave", "confidence": 1.4, "reason": "右手举起来摆"},'
                               ' "a__0002_track1_t1.00s": {"label": "dance", "confidence": 0.5}}', frames)
    assert got == {"a__0001_track1_t0.00s": Guess("wave", 1.0, "右手举起来摆")}


def test_parse_unsure_and_long_reason():
    frames = [FrameInput("c", SHEET, [])]
    got = parse_gesture_review(json.dumps({"c": {"label": "unsure", "confidence": "x", "reason": "字" * 100}}), frames)
    assert got["c"].label == "unsure" and got["c"].confidence == 0.0 and len(got["c"].reason) == 60


def test_message_has_hint_and_one_image_per_clip():
    msg = build_gesture_message([FrameInput("gesture-bow-1__0001_track1_t0.00s", SHEET, [])], AssistConfig())
    assert sum(b["type"] == "image" for b in msg) == 1 and any("鞠躬" in b.get("text", "") for b in msg)


def test_guess_roundtrip(tmp_path):
    assert load_guess(tmp_path) is None
    write_guess(tmp_path, Guess("bow", 0.7, "弯腰"), "sonnet")
    assert load_guess(tmp_path) == Guess("bow", 0.7, "弯腰")
    (tmp_path / "claude.json").write_text("{坏", encoding="utf-8")
    assert load_guess(tmp_path) is None


def _clip(root, name, n=16):
    d = root / name
    d.mkdir(parents=True)
    for i in range(n):
        imwrite(d / f"{i:02d}.jpg", np.full((112, 112, 3), 90, np.uint8))
    return d


def _cli_env(tmp_path, monkeypatch):
    from skydango import cli

    sent = []

    def run(cmd, env, cwd, content, timeout, on_message=None):
        stems = [b["text"].split()[1].rstrip("：") for b in content if b["type"] == "text" and b["text"].startswith("片段 ")]
        sent.append(stems)
        return {"result": json.dumps({s: {"label": "wave", "confidence": 0.9, "reason": "r"} for s in stems}),
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    fake_claude(monkeypatch, ["claude"])
    monkeypatch.setattr("skydango.models.claude_code.one_shot_message", run)
    cfg = Config()
    cfg.gesture.dataset = str(tmp_path / "ds")
    return cli, cfg, sent


def _args(recheck=False, source=None, blind=False):
    return SimpleNamespace(source=source, recheck=recheck, blind=blind)


def test_cli_skips_existing_guess_and_recheck(tmp_path, monkeypatch):
    cli, cfg, sent = _cli_env(tmp_path, monkeypatch)
    un = tmp_path / "ds" / "_unlabeled"
    a, b = _clip(un, "r__0001_track1_t0.00s"), _clip(un, "r__0002_track1_t1.00s")
    write_guess(a, Guess("bow", 0.5, ""), "m")
    cli._perception_gesture_label(cfg, _args())
    assert sent == [["r__0002_track1_t1.00s"]]
    assert load_guess(b).label == "wave" and load_guess(a).label == "bow"
    cli._perception_gesture_label(cfg, _args(recheck=True))
    assert sorted(s for batch in sent[1:] for s in batch) == ["r__0001_track1_t0.00s", "r__0002_track1_t1.00s"]
    assert load_guess(a).label == "wave"


def test_incomplete_clip_skipped(tmp_path, monkeypatch, capsys):
    cli, cfg, sent = _cli_env(tmp_path, monkeypatch)
    un = tmp_path / "ds" / "_unlabeled"
    _clip(un, "r__0001_track1_t0.00s", n=15)
    _clip(un, "r__0002_track1_t1.00s")
    cli._perception_gesture_label(cfg, _args())
    assert sent == [["r__0002_track1_t1.00s"]]
    assert "r__0001_track1_t0.00s" in capsys.readouterr().out


def test_clip_labeled_in_page_while_running_is_skipped(tmp_path, monkeypatch, capsys):
    cli, cfg, sent = _cli_env(tmp_path, monkeypatch)
    from skydango.models import claude_code

    un = tmp_path / "ds" / "_unlabeled"
    a, b = _clip(un, "r__0001_track1_t0.00s"), _clip(un, "r__0002_track1_t1.00s")
    inner = claude_code.one_shot_message  # _cli_env 换上的假的

    def run(cmd, env, cwd, content, timeout, on_message=None):
        out = inner(cmd, env, cwd, content, timeout)
        (tmp_path / "ds" / "wave").mkdir(exist_ok=True)
        a.rename(tmp_path / "ds" / "wave" / a.name)  # Claude 还在看的时候，网页上标走了
        return out

    monkeypatch.setattr("skydango.models.claude_code.one_shot_message", run)
    cli._perception_gesture_label(cfg, _args())
    assert load_guess(b).label == "wave"
    assert load_guess(tmp_path / "ds" / "wave" / a.name) is None  # 标走的不写 claude.json
    assert "已被标走 1 段" in capsys.readouterr().out


def test_unreadable_clip_is_skipped(tmp_path, monkeypatch, capsys):
    cli, cfg, sent = _cli_env(tmp_path, monkeypatch)
    un = tmp_path / "ds" / "_unlabeled"
    bad = _clip(un, "r__0001_track1_t0.00s")
    (bad / "03.jpg").write_bytes(b"not a jpeg")
    good = _clip(un, "r__0002_track1_t1.00s")
    cli._perception_gesture_label(cfg, _args())
    assert sent == [["r__0002_track1_t1.00s"]]
    assert load_guess(good).label == "wave" and load_guess(bad) is None
    out = capsys.readouterr().out
    assert "读不了 1 段" in out and "r__0001_track1_t0.00s" in out


def test_blind_message_has_no_hint():
    from skydango.vision.gesture_label import GESTURE_BLIND_PROTOCOL, GESTURE_SYSTEM_BLIND

    msg = GESTURE_BLIND_PROTOCOL.build([FrameInput("gesture-bow-1__0001_track1_t0.00s", SHEET, [])], AssistConfig())
    assert sum(b["type"] == "image" for b in msg) == 1
    assert not any("鞠躬" in b.get("text", "") for b in msg)
    assert GESTURE_BLIND_PROTOCOL.system == GESTURE_SYSTEM_BLIND and "录像说明" not in GESTURE_SYSTEM_BLIND


def test_blind_guess_file_is_separate(tmp_path):
    from skydango.vision.gesture_label import BLIND_FILE

    write_guess(tmp_path, Guess("bow", 0.7, "弯腰"), "sonnet")
    write_guess(tmp_path, Guess("none", 0.6, "站着"), "sonnet", BLIND_FILE)
    assert load_guess(tmp_path) == Guess("bow", 0.7, "弯腰")
    assert load_guess(tmp_path, BLIND_FILE) == Guess("none", 0.6, "站着")
    assert json.loads((tmp_path / BLIND_FILE).read_text(encoding="utf-8"))["blind"] is True


def test_cli_blind_writes_blind_file_and_compares(tmp_path, monkeypatch, capsys):
    from skydango.vision.gesture_label import BLIND_FILE

    cli, cfg, sent = _cli_env(tmp_path, monkeypatch)
    un = tmp_path / "ds" / "_unlabeled"
    a, b = _clip(un, "gesture-bow-1__0001_track1_t0.00s"), _clip(un, "gesture-bow-1__0002_track1_t1.00s")
    write_guess(a, Guess("bow", 0.5, ""), "m")
    write_guess(b, Guess("wave", 0.5, ""), "m")
    cli._perception_gesture_label(cfg, _args(blind=True))
    stems = [s for batch in sent for s in batch]
    assert len(set(stems)) == 2 and not any("bow" in s or "gesture" in s for s in stems)  # 片段名里的录像名也不给看
    assert load_guess(a).label == "bow" and load_guess(a, BLIND_FILE).label == "wave"
    assert "和看录像名的那次一致 1/2" in capsys.readouterr().out
    cli._perception_gesture_label(cfg, _args(blind=True))  # 做过的跳过
    assert len(sent) == 1
