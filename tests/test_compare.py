from pathlib import Path

import numpy as np

from skydango.config import EnvConfig, PerceptionConfig
from skydango.vision.bubbles import Rect
from skydango.vision.compare import FrameResult, compare_frames, frame_time, report_md, side_by_side, summarize, timed_files
from skydango.vision.detect import Detection
from skydango.vision.env import EnvWatcher
from skydango.vision.ocr import OcrLine
from skydango.vision.perception import PerceptionWatcher

FRIENDS = ["懒洋洋大王", "番茄炒蛋盖饭"]


def test_frame_time_parses_record_names():
    assert frame_time(Path("0012_006.00s.jpg")) == 6.0
    assert frame_time(Path("tmp/record/x/0003_120.50s.jpg")) == 120.5
    assert frame_time(Path("shot.png")) is None


class SeqOcr:
    def __init__(self, texts):
        self.texts = list(texts)

    def recognize(self, img):
        texts = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return [OcrLine(t, 0.99, Rect(700, 300, 120, 40)) for t in texts]


class TagDetector:
    def detect(self, img):
        return [Detection("name_tag", Rect(990, 330, 110, 44), 0.9)]


class NameOcr:
    def read_line(self, img):
        return OcrLine("懒洋洋大王", 0.99, Rect(0, 0, img.shape[1], img.shape[0]))


def test_compare_frames_env_scans_on_interval_and_diff_callback():
    log_roi = [0.0, 0.0, 0.335, 0.855]
    env = EnvWatcher(SeqOcr([["懒洋洋大王"], []]), EnvConfig(interval=3.0), lambda: list(FRIENDS), log_roi, background=False)
    yolo = PerceptionWatcher(TagDetector(), NameOcr(), PerceptionConfig(), EnvConfig(), lambda: list(FRIENDS), log_roi,
                             background=False)
    img = np.zeros((1080, 1920, 3), np.uint8)
    diffs = []
    results = compare_frames(((float(t), f"{t}.jpg", img, False) for t in range(5)), env, yolo,
                             on_diff=lambda r, frame: diffs.append(r.file))
    assert [r.env is not None for r in results] == [True, False, False, True, False]
    assert results[0].env == {"懒洋洋大王"} and results[3].env == set()
    assert all(r.yolo == {"懒洋洋大王"} for r in results)
    assert diffs == ["3.jpg"]


def fr(t, env=None, yolo=(), env_req=(), yolo_req=(), strangers=0):
    return FrameResult(f"{t}.jpg", float(t), None if env is None else set(env), set(yolo), set(env_req), set(yolo_req),
                       strangers, 5.0 if env is not None else None, 20.0)


def test_summarize_requests_delay_stranger_and_leave_events():
    hand = ("懒洋洋大王", "hand")
    results = [
        fr(0, env=["懒洋洋大王"], yolo=["懒洋洋大王"]),
        fr(1, yolo=["懒洋洋大王"], yolo_req=[hand], strangers=1),
        fr(2, yolo=["懒洋洋大王"], yolo_req=[hand], strangers=1),
        fr(3, env=["懒洋洋大王"], env_req=[hand]),
        fr(6, env=[]),
        fr(10, env=[]),
    ]
    s = summarize(results, keep_env=30.0, keep_yolo=5.0)
    assert s["requests"]["懒洋洋大王:hand"] == {"env": 3.0, "yolo": 1.0, "delay": 2.0}
    assert [e["t"] for e in s["stranger_events"]] == [1.0, 3.0]
    assert s["leave_events"]["yolo"] == [{"name": "懒洋洋大王", "t": 7.0}]
    assert s["leave_events"]["env"] == []
    assert s["friends"]["懒洋洋大王"] == {"env": 2, "yolo_same_frames": 1, "yolo_all": 3}
    assert s["diff_frames"] == ["3.jpg"]
    assert s["timing"] == {"env_ms": 5.0, "yolo_ms": 20.0}


def test_report_md_mentions_every_section():
    s = summarize([fr(0, env=["懒洋洋大王"], yolo=["懒洋洋大王"])], 30.0, 5.0)
    md = report_md(s)
    for word in ("好友认出率", "互动请求", "stranger", "leave", "耗时"):
        assert word in md


def test_side_by_side_is_twice_as_wide():
    img = np.zeros((100, 200, 3), np.uint8)
    out = side_by_side(img, [{"x": 10, "y": 10, "w": 20, "h": 20, "kind": "name", "label": "懒洋洋大王"}], [])
    assert out.shape == (100, 400, 3) and out.any()


def test_timed_files_sorts_by_time_and_rejects_untimed_dirs():
    import pytest

    files = [Path("0002_001.00s.jpg"), Path("shot.png"), Path("0001_000.50s.jpg")]
    timed, skipped = timed_files(files)
    assert timed == [(0.5, Path("0001_000.50s.jpg")), (1.0, Path("0002_001.00s.jpg"))]
    assert skipped == [Path("shot.png")]
    with pytest.raises(ValueError, match="record"):
        timed_files([Path("a.png"), Path("b.png")])


def test_summarize_reports_far_named_rate():
    results = [fr(0), fr(1)]
    for r in results:
        r.far, r.far_named = 2, 1
    s = summarize(results, 30.0, 5.0)
    assert s["far"] == {"players": 4, "named": 2, "rate": 0.5}
    md = report_md(s)
    assert "远处" in md and "50%" in md and "--far-crops 0" in md
    assert summarize([fr(0)], 30.0, 5.0)["far"] == {"players": 0, "named": 0, "rate": None}


class FarDetector:
    """远处一个小人；裁剪图里有它的名字标签（裁剪图坐标）。"""

    def detect(self, img):
        if img.shape[:2] == (1080, 1920):
            return [Detection("player", Rect(1000, 500, 24, 60), 0.9)]
        return [Detection("name_tag", Rect(10, 20, 50, 14), 0.9)]


class WidthOcr:
    def read_line(self, img):
        return OcrLine("懒洋洋大王", 0.99, Rect(0, 0, img.shape[1], img.shape[0])) if img.shape[1] == 58 else None


def test_compare_frames_counts_far_players_and_named_ones():
    log_roi = [0.0, 0.0, 0.335, 0.855]
    img = np.zeros((1080, 1920, 3), np.uint8)
    results = []
    for crops in (3, 0):
        env = EnvWatcher(SeqOcr([[]]), EnvConfig(interval=3.0), lambda: list(FRIENDS), log_roi, background=False)
        yolo = PerceptionWatcher(FarDetector(), WidthOcr(), PerceptionConfig(far_crops=crops), EnvConfig(),
                                 lambda: list(FRIENDS), log_roi, background=False)
        results.append(compare_frames([(0.0, "0.jpg", img, False)], env, yolo)[0])
    assert (results[0].far, results[0].far_named) == (1, 1)
    assert (results[1].far, results[1].far_named) == (1, 0)


def test_summarize_and_report_objects():
    rs = [fr(0, None, []), fr(1, None, []), fr(2, None, [])]
    rs[0].objects, rs[1].objects = {"bench": 2}, {"bench": 1, "spirit": 1}
    s = summarize(rs, 5.0, 5.0)
    assert s["objects"] == {"bench": {"frames": 2, "avg": 1.5}, "spirit": {"frames": 1, "avg": 1.0}}
    md = report_md(s)
    assert "## 物品" in md and "- 座位：出现在 2 帧，平均每帧 1.5 个" in md and "- 先祖：出现在 1 帧，平均每帧 1.0 个" in md
    assert "没有认出物品（模型里没有物品类别，或者录像里没有）" in report_md(summarize([fr(0, None, [])], 5.0, 5.0))
