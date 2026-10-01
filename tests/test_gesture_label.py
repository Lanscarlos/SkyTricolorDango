"""动作片段 Claude 初分（vision/gesture_label.py + perception gesture-label）。不连 Claude。"""

import json
from types import SimpleNamespace

import numpy as np

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

    def run(cmd, env, cwd, content, timeout):
        stems = [b["text"].split()[1].rstrip("：") for b in content if b["type"] == "text" and b["text"].startswith("片段 ")]
        sent.append(stems)
        return {"result": json.dumps({s: {"label": "wave", "confidence": 0.9, "reason": "r"} for s in stems}),
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr(cli, "_brain_env", lambda cfg: (["claude"], {}))
    monkeypatch.setattr("skydango.brain.claude.one_shot_message", run)
    cfg = Config()
    cfg.gesture.dataset = str(tmp_path / "ds")
    return cli, cfg, sent


def _args(recheck=False, source=None):
    return SimpleNamespace(source=source, recheck=recheck)


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
