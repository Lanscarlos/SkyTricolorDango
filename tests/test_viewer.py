import json
import threading
import urllib.request

import numpy as np
import pytest
from conftest import FakeDevice, scene

from skydango.brain.events import EventQueue
from skydango.chat.reader import Message
from skydango.config import Config, EnvConfig, PerceptionConfig, ViewerConfig
from skydango.game.social import IDLE, Request
from skydango.imageio import imwrite
from skydango.vision.bubbles import Rect
from skydango.vision.detect import Detection
from skydango.vision.env import EnvWatcher
from skydango.vision.ocr import OcrLine
from skydango.vision.perception import PerceptionWatcher
from skydango.vision.viewer import Viewer, describe_env, panel_box


def frame(w=1920, h=1080):
    return np.full((h, w, 3), 80, np.uint8)


def viewer(**cfg):
    cfg.setdefault("port", 0)  # 随机端口
    return Viewer(ViewerConfig(**cfg))


def snap(v, after=0):
    body = v.snapshot(after, timeout=0.0)
    return None if body is None else json.loads(body)


# ---- overlay：两种识别器给出的框 ----
class IconsStub:
    def __init__(self, kind):
        self.kind = kind

    def classify(self, region):
        return self.kind, 0.9


def test_env_overlay_draws_recent_friend_labels_and_rings():
    class LabelOcr:
        def recognize(self, region):
            return [OcrLine("懒洋洋大王", 0.99, Rect(1320, 300, 160, 44))]

    env = EnvWatcher(LabelOcr(), EnvConfig(interval=3.0), lambda: ["懒洋洋大王"], [0.0, 0.0, 0.335, 0.855],
                     background=False, icons=IconsStub("hand"), icon_offset=2.23)
    env.observe(frame(), 10.0, panel_visible=False)
    boxes = env.overlay(10.5)
    assert boxes[0] == {"x": 1320, "y": 300, "w": 160, "h": 44, "kind": "name", "label": "懒洋洋大王"}
    assert boxes[1]["kind"] == "request" and boxes[1]["label"] == "牵手"
    assert boxes[1]["x"] + 50 == 1400 and boxes[1]["y"] + 50 == 300 + round(2.23 * 44)
    env.icons.kind = IDLE
    env.observe(frame(), 14.0, panel_visible=False)
    assert [(b["kind"], b["label"]) for b in env.overlay(14.0)] == [("name", "懒洋洋大王"), ("ring", "✦")]
    assert env.overlay(14.0 + 3.0 * 2 + 2) == []  # 两次扫描都没再看到：不画了


class ScriptDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


class WidthOcr:
    def __init__(self, by_width):
        self.by_width = by_width

    def read_line(self, img):
        text = self.by_width.get(img.shape[1] - 8)
        return OcrLine(text, 0.99, Rect(0, 0, 1, 1)) if text else None


def test_perception_overlay_labels_every_track_kind():
    tag_h = 44
    dets = [
        Detection("player", Rect(1000, 400, 90, 220), 0.91),
        Detection("name_tag", Rect(990, 330, 110, tag_h), 0.88),  # 认出是好友
        Detection("name_tag", Rect(1600, 330, 130, tag_h), 0.7),  # 读到了字但不在名单里
        Detection("player", Rect(1500, 400, 90, 220), 0.8),  # 没标签：过一会儿算陌生人
        Detection("player_unlit", Rect(1300, 500, 40, 90), 0.66),
        Detection("self", Rect(900, 420, 90, 220), 0.95),
        Detection("social_ring", Rect(1045 - 50, 330 + round(2.23 * tag_h) - 50, 100, 100), 0.8),
    ]
    w = PerceptionWatcher(ScriptDetector(dets), WidthOcr({110: "懒洋洋大王", 130: "路人甲乙"}),
                          PerceptionConfig(stranger_after=1.0), EnvConfig(), lambda: ["懒洋洋大王"],
                          [0.0, 0.0, 0.335, 0.855], icons=IconsStub("hug"), background=False)
    for t in (0.0, 0.5, 1.0, 1.5):
        w.process(frame(), t, panel_visible=False)
    by = {(b["kind"], b["label"]) for b in w.overlay(1.5)}
    assert ("friend", "懒洋洋大王") in by  # 人物关联到头顶的名字
    assert ("name", "懒洋洋大王") in by and ("tag", "?路人甲乙") in by
    assert ("stranger", "陌生人") in by and ("unlit", "陌生人（没点火）") in by
    assert ("self", "团子") in by and ("request", "拥抱") in by
    unlit = next(b for b in w.overlay(1.5) if b["kind"] == "unlit")
    assert unlit["score"] == 0.66 and (unlit["x"], unlit["y"], unlit["w"], unlit["h"]) == (1300, 500, 40, 90)


def test_describe_env_for_both_recognizers():
    env = EnvWatcher(None, EnvConfig(interval=4.0), lambda: ["懒洋洋大王"], [0.0, 0.0, 0.3, 0.8])
    env.last_seen["懒洋洋大王"] = 1.0
    env.requests["懒洋洋大王"] = Request("懒洋洋大王", "hand", (0, 0), 1.0)
    info = describe_env(env, 2.0)
    assert info == {"识别": "整图 OCR（每 4 秒一次）", "身边的好友": ["懒洋洋大王"], "互动请求": ["懒洋洋大王：牵手"]}

    w = PerceptionWatcher(ScriptDetector([Detection("player_unlit", Rect(1300, 500, 40, 90), 0.7)]), WidthOcr({}),
                          PerceptionConfig(), EnvConfig(), lambda: [], [0.0, 0.0, 0.3, 0.8], background=False)
    w.process(frame(), 0.0, panel_visible=False)
    info = describe_env(w, 0.0)
    assert info["识别"] == "YOLO 感知层" and info["身边的好友"] == "没看到"
    assert info["陌生人"] == "1 个（1 个没点火）" and info["互动请求"] == "没有"
    assert "ms" in info["检测耗时"]


# ---- Viewer：快照、限帧率、只压一次 ----
def test_update_throttles_to_fps_and_keeps_messages_for_a_while():
    v = viewer(fps=10.0)
    m = Message("在吗在吗", Rect(10, 20, 300, 30), 0.0, "懒洋洋大王")
    assert v.update(frame(), 0.0, messages=[m])
    assert not v.update(frame(), 0.05)  # 超过 10 帧 / 秒：丢掉
    assert v.update(frame(), 0.1, panel=Rect(0, 0, 640, 900))
    s = snap(v)
    assert s["seq"] == 2 and (s["width"], s["height"]) == (1920, 1080)
    kinds = [(b["kind"], b["label"]) for b in s["boxes"]]
    assert kinds == [("panel", "聊天记录面板"), ("message", "懒洋洋大王：在吗在吗")]
    assert s["info"]["来源"] == "实时截图" and s["info"]["识别"].startswith("没开")
    v.update(frame(), 3.5)
    assert [b["kind"] for b in snap(v)["boxes"]] == []  # 新消息的框留 3 秒


def test_snapshot_returns_none_without_new_frame_and_encodes_once():
    v = viewer(width=640)
    assert v.snapshot(0, timeout=0.0) is None
    v.update(frame(), 0.0, info={"模式": "LIVE"})
    a, b = v.snapshot(0, timeout=0.0), v.snapshot(0, timeout=0.0)
    assert a is b and v.encodes == 1
    s = json.loads(a)
    assert s["info"]["模式"] == "LIVE" and s["width"] == 1920  # 框按原图坐标，图本身缩到 640 宽
    import base64

    import cv2

    img = cv2.imdecode(np.frombuffer(base64.b64decode(s["image"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640)
    assert v.snapshot(1, timeout=0.0) is None  # 没有比 1 新的
    assert v.snapshot(99, timeout=0.0) is not None  # 浏览器的序号比这边大：程序重启过，从头给


def test_update_survives_broken_env():
    class Broken:
        def overlay(self, now):
            raise RuntimeError("boom")

    v = viewer()
    assert v.update(frame(), 0.0, env=Broken(), info={"模式": "dry-run"})
    assert snap(v)["info"]["模式"] == "dry-run"


def test_long_poll_wakes_up_on_new_frame():
    v = viewer()
    out = []
    t = threading.Thread(target=lambda: out.append(v.snapshot(0, timeout=5.0)))
    t.start()
    v.update(frame(), 0.0)
    t.join(2.0)
    assert out and out[0] is not None


def test_http_server_serves_page_and_snapshot():
    v = viewer()
    url = v.start()
    try:
        assert url.startswith("http://127.0.0.1:") and not url.endswith(":0/")
        page = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "团子看到的" in page and "/snapshot" in page
        v.update(frame(), 0.0, panel=Rect(0, 0, 10, 10))
        with urllib.request.urlopen(url + "snapshot?after=0", timeout=5) as r:
            assert r.status == 200 and json.loads(r.read())["boxes"][0]["kind"] == "panel"
        with urllib.request.urlopen(url + "snapshot?after=1", timeout=5) as r:  # 没新帧：等满 2 秒回 204
            assert r.status == 204
    finally:
        v.stop()


def test_port_in_use_raises():
    v = viewer()
    v.start()
    try:
        port = v._server.server_address[1]
        with pytest.raises(OSError):
            viewer(port=port).start()
    finally:
        v.stop()


def test_panel_box_only_when_log_panel_is_open():
    cfg = Config()
    cfg.vision.mode = "log"

    class R:
        panel_closed_since = None

    r = R()
    assert panel_box(cfg.vision, r, frame()) is not None
    r.panel_closed_since = 1.0
    assert panel_box(cfg.vision, r, frame()) is None
    cfg.vision.mode = "roi"
    r.panel_closed_since = None
    assert panel_box(cfg.vision, r, frame()) is None


def test_event_history_survives_drain_and_counts_repeats():
    q = EventQueue(clock=lambda: 0.0)
    q.put("arrive", "懒洋洋大王 来了")
    q.put("error", "截图失败")
    q.put("error", "截图失败")
    q.drain()
    assert [e.line() for e in q.recent(6)] == ["懒洋洋大王 来了", "截图失败（×2）"]


# ---- Body / Agent 每圈交给可视化，出错不影响主循环 ----
class SpyViewer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def update(self, frame, now, **kw):
        self.calls.append((now, kw))
        if self.fail:
            raise RuntimeError("boom")
        return True


@pytest.mark.parametrize("fail", [False, True])
def test_body_shows_each_step(clock, fail):
    from test_brain_body import body, msg

    spy = SpyViewer(fail)
    b, _, reader, events = body(clock, viewer=spy)
    reader.batches = [[msg("在吗")]]
    b.step()
    clock.advance(0.2)
    b.step()
    assert len(spy.calls) == 2
    kw = spy.calls[0][1]
    assert kw["messages"][0].text == "在吗" and kw["panel"] is not None
    assert kw["info"]["模式"] == "dry-run" and kw["info"]["牵着手"] == "没有"
    assert kw["info"]["最近事件"][0].startswith("聊天")
    assert events.drain()  # 可视化出错时，事件照样进了队列


@pytest.mark.parametrize("fail", [False, True])
def test_agent_shows_each_step(clock, fail):
    from test_agent import build, live_config

    cfg = live_config()
    cfg.vision.mode = "roi"
    agent, _ = build(cfg, [scene()], [[]], clock)
    agent.viewer = spy = SpyViewer(fail)
    agent.step()
    assert len(spy.calls) == 1
    info = spy.calls[0][1]["info"]
    assert info["模式"] == "LIVE" and info["待回复"] == "0 条" and info["刚说过"] == "还没说话"


# ---- view --images：回放 ----
def test_view_replays_images_with_boxes(tmp_path):
    from skydango import cli

    for i in range(2):
        imwrite(tmp_path / f"{i:03d}.png", frame())
    cfg = Config()
    cfg.vision.mode = "roi"
    cfg.env.interval = 0.0

    class LabelOcr:
        def recognize(self, region):
            return [OcrLine("懒洋洋大王", 0.99, Rect(1320, 300, 160, 44))]

    class NoRead:
        panel_closed_since = None

        def read(self, frame, now):
            return []

    env = EnvWatcher(LabelOcr(), cfg.env, lambda: ["懒洋洋大王"], cfg.vision.log_roi, background=False)
    v = viewer()
    t = iter([0.0, 0.0, 1.0, 1.0])
    seen = []
    orig = v.update

    def update(*a, **kw):
        ok = orig(*a, **kw)
        seen.append(snap(v))
        return ok

    v.update = update
    shown = cli._view(cfg, cli._view_frames(cli._images(str(tmp_path)), None), NoRead(), env, v,
                      clock=lambda: next(t), sleep=lambda s: None)
    assert shown == 2
    assert [s["info"]["来源"] for s in seen] == ["回放 000.png（1/2）", "回放 001.png（2/2）"]
    assert seen[-1]["boxes"][0]["label"] == "懒洋洋大王" and seen[-1]["info"]["身边的好友"] == ["懒洋洋大王"]


def test_view_frames_skips_failed_screenshots():
    from skydango import cli

    class Flaky:
        def __init__(self):
            self.n = 0

        def screenshot(self):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("adb 超时")
            return frame()

    frames = cli._view_frames([], Flaky(), sleep=lambda s: None)
    img, source = next(frames)
    assert img.shape == (1080, 1920, 3) and "实时截图" in source


def test_view_marks_black_frames():
    from skydango import cli

    cfg = Config()
    cfg.vision.mode = "roi"

    class NoRead:
        panel_closed_since = None

        def read(self, frame, now):
            return []

    class NoEnv:
        requests = {}
        cfg = EnvConfig()

        def observe(self, *a, **kw):
            pass

        def nearby(self, now):
            return []

    v = viewer()
    cli._view(cfg, iter([(np.zeros((1080, 1920, 3), np.uint8), "x")]), NoRead(), NoEnv(), v, sleep=lambda s: None)
    assert snap(v)["info"]["画面"].startswith("黑着")


def test_view_command_is_registered():
    from skydango import cli

    with pytest.raises(SystemExit):
        cli.main(["view", "--help"])
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
