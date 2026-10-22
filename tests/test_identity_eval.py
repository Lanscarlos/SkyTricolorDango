"""identity-truth：半自动标准答案（草稿、拼图、--apply）。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skydango.imageio import imwrite
from skydango.vision.bubbles import Rect
from skydango.vision.identity_eval import (
    TrackLog, apply_draft, collect_logs, draft_identities, load_logs, save_logs, write_draft, write_sheet,
)


def log(tid, n=20, names=(), selfs=0, t0=0.0, step=0.15):
    frames = [(f"{i:04d}.jpg", Rect(10 + i, 20, 30, 60)) for i in range(n)]
    return TrackLog(tid, t0, t0 + (n - 1) * step, frames, list(names), selfs)


def names(who, n, t0=0.0, step=0.15):
    return [(t0 + i * step, who) for i in range(n)]


def test_draft_tag_held():
    lg = log(1, names=names("饶总", 8))
    assert draft_identities({1: lg}, 0.5) == {1: "饶总"}


def test_draft_flash_not_counted():
    lg = log(2, names=[(0.3, "小明")])
    assert draft_identities({2: lg}, 0.5) == {2: "?"}


def test_draft_most_held_wins():
    lg = log(3, n=30, names=names("A", 3) + names("B", 10, t0=1.0))
    assert draft_identities({3: lg}, 0.5) == {3: "B"}


def test_draft_dango_by_self_hits():
    assert draft_identities({4: log(4, selfs=12)}, 0.5) == {4: "团子"}
    assert draft_identities({5: log(5, selfs=5)}, 0.5) == {5: "?"}


def test_logs_roundtrip(tmp_path):
    logs = {1: log(1, names=names("A", 4), selfs=2)}
    save_logs(tmp_path / "logs.json", logs)
    assert load_logs(tmp_path / "logs.json") == logs


def test_apply_roundtrip(tmp_path):
    logs = {1: log(1, n=5), 2: log(2, n=3), 3: log(3, n=4), 4: log(4, n=2)}
    save_logs(tmp_path / "logs.json", logs)
    write_draft(tmp_path / "d.toml", {1: "饶总", 2: "?", 3: "团子", 4: "?"}, logs)
    text = (tmp_path / "d.toml").read_text(encoding="utf-8")
    assert '"饶总"' in text and "5 帧" in text
    text = text.replace('t2 = "?"', 't2 = "陌生人"').replace('t4 = "?"', 't4 = "不要"')
    (tmp_path / "d.toml").write_text(text, encoding="utf-8")
    out = tmp_path / "truth.jsonl"
    n = apply_draft(tmp_path / "d.toml", tmp_path / "logs.json", out, ["饶总", "小红"])
    assert n == 5 + 3 + 4
    rows = [json.loads(s) for s in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert rows[0] == {"frame": "0000.jpg", "box": [10, 20, 30, 60], "who": "饶总"}
    assert {r["who"] for r in rows} == {"饶总", "陌生人", "团子"}


def test_apply_unknown_value_has_line_no(tmp_path):
    logs = {1: log(1, n=2)}
    save_logs(tmp_path / "logs.json", logs)
    (tmp_path / "d.toml").write_text('# 注释\n\nt1 = "老登"  # x\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 3 行"):
        apply_draft(tmp_path / "d.toml", tmp_path / "logs.json", tmp_path / "t.jsonl", ["饶总"])


def test_sheet_written(tmp_path):
    frames = tmp_path / "rec"
    frames.mkdir()
    for i in range(6):
        imwrite(frames / f"{i:04d}.jpg", np.full((200, 300, 3), 90, np.uint8))
    logs = {1: log(1, n=6), 2: log(2, n=2)}
    write_sheet(tmp_path / "sheet.jpg", logs, frames)
    assert (tmp_path / "sheet.jpg").stat().st_size > 0


class FakeWatcher:
    """每帧给一个 player 轨迹 7。名字是黏的（第一次配上后一直留着），tag_at 只在标签真配上的帧 = t。"""
    def __init__(self, tag_frames=None):
        self.tag_frames = tag_frames if tag_frames is not None else {2, 3, 4, 5, 6, 7, 8}
        self.data = {}
        self.cfg = SimpleNamespace(keep=5.0)
        self.clock = None
        self.last_tracks = []
        self.n = 0

    def process(self, frame, t, panel):
        self.n += 1
        if self.n in self.tag_frames:
            self.data.update(name="小红", tagged=True, tag_at=t)
        self.last_tracks = [SimpleNamespace(id=7, cls="player", box=Rect(5, 5, 20, 40), data=self.data)]


def test_collect_logs():
    items = [(i * 0.2, f"{i}.jpg", np.zeros((10, 10, 3), np.uint8), False) for i in range(4)]
    logs = collect_logs(items, FakeWatcher())
    lg = logs[7]
    assert [f for f, _ in lg.frames] == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    assert lg.names == [(0.2, "小红"), (0.4, "小红"), (0.6000000000000001, "小红")]
    assert (lg.first, lg.last) == (0.0, pytest.approx(0.6))


def test_collect_logs_flash_not_sticky():
    items = [(i * 0.2, f"{i}.jpg", np.zeros((10, 10, 3), np.uint8), False) for i in range(6)]
    logs = collect_logs(items, FakeWatcher(tag_frames={2}))
    assert logs[7].names == [(0.2, "小红")]
    assert draft_identities(logs, 0.5) == {7: "?"}


def test_apply_duplicate_line(tmp_path):
    save_logs(tmp_path / "logs.json", {1: log(1, n=2)})
    (tmp_path / "d.toml").write_text('t1 = "团子"\nt1 = "不要"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        apply_draft(tmp_path / "d.toml", tmp_path / "logs.json", tmp_path / "t.jsonl", [])


def test_frame_key(tmp_path):
    from skydango.vision.identity_eval import frame_key
    assert frame_key(tmp_path / "a" / "x.jpg", tmp_path) == "a/x.jpg"
    assert frame_key(Path("/elsewhere/y.jpg"), tmp_path) == "y.jpg"


def test_cli_end_to_end(tmp_path, monkeypatch):
    from skydango import cli
    from skydango.config import Config

    rec = tmp_path / "rec"
    rec.mkdir()
    for i in range(8):
        imwrite(rec / f"{i:04d}_{i * 0.2:05.2f}s.jpg", np.full((120, 160, 3), 80, np.uint8))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_perception", lambda cfg, args: (None, FakeWatcher()))
    monkeypatch.setattr(cli, "_panel_open", lambda cfg, frame: False)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: ["小红"]))
    cfg = Config()
    args = SimpleNamespace(source=str(rec), model=None, fps=6.5, apply=False, output=None)
    cli._perception_identity_truth(cfg, args)
    out = tmp_path / "tmp" / "identity-truth" / "rec"
    assert (out / "sheet.jpg").exists() and (out / "logs.json").exists()
    assert 't7 = "小红"' in (out / "truth-draft.toml").read_text(encoding="utf-8")
    args.apply = True
    cli._perception_identity_truth(cfg, args)
    rows = (rec / "truth.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 8 and json.loads(rows[0])["who"] == "小红"
