import logging

import numpy as np
from test_brain_body import FakeEnv, body
from test_brain_loop import FakeRun, FakeSession, make, ok
from test_brain_tools import FakeBody, FakeEyes

from skydango.brain.tools import ToolBox
from skydango.brain.vision import LEGEND_HEAD, NO_BOXES, mark_frame, parse_check


class MarkedEnv(FakeEnv):
    def __init__(self, fresh=True):
        super().__init__()
        self.fresh = fresh

    def overlay_fresh(self, now):
        return self.fresh

    def overlay(self, now):
        return [{"x": 500, "y": 300, "w": 100, "h": 200, "kind": "friend", "label": "懒洋洋大王"}]


def test_body_look_marked_returns_legend_then_image(clock):
    b, _, _, _ = body(clock, env=MarkedEnv())
    legend, img = b.look(marked=True)
    assert legend["type"] == "text" and legend["text"].startswith(LEGEND_HEAD) and "懒洋洋大王（好友，看到名字）" in legend["text"]
    assert img["type"] == "image"
    clock.advance(6)
    first, second = b.look()  # 不带 marked：逐字照旧，图在前、位置说明在后
    assert first["type"] == "image" and second["type"] == "text" and LEGEND_HEAD not in second["text"]


def test_body_look_marked_stale_overlay_draws_no_boxes(clock):
    b, _, _, _ = body(clock, env=MarkedEnv(fresh=False))
    legend, _ = b.look(marked=True)
    assert legend["text"] == NO_BOXES


def test_body_look_marked_without_perception_is_unchanged(clock):
    b, _, _, _ = body(clock, env=FakeEnv())  # 没有 overlay_fresh：EnvWatcher 那种
    img, note = b.look(marked=True)
    assert img["type"] == "image" and "名字" in note["text"]


class MarkBody(FakeBody):
    def look(self, marked=False):
        self.calls.append(("look", marked))
        return [{"type": "image"}]


def test_toolbox_look_passes_marked():
    body_ = MarkBody()
    ToolBox(body_, FakeEyes(), marked=lambda: True).run("look", {"image": True})
    assert body_.calls == [("look", True)]
    plain = MarkBody()
    ToolBox(plain, FakeEyes()).run("look", {"image": True})
    assert plain.calls == [("look", False)]  # 没开 marked：照旧调 b.look()


class ImageSession(FakeSession):
    def __init__(self, *results):
        super().__init__(*results)
        self.images = []

    def send(self, text, image=None):
        self.images.append(image)
        return super().send(text)


def att():
    return mark_frame(np.zeros((1080, 1920, 3), np.uint8), None, (960, 540), 80)


def test_brain_wake_sends_image_from_snapshot(clock):
    session = ImageSession()
    brain, _, _, _ = make(clock, session)
    a = att()
    brain.snapshot = lambda events: a
    brain.wake(clock(), "heartbeat")
    assert session.images == [a]


def test_brain_wake_without_snapshot_calls_plain_send(clock):
    session = FakeSession()  # BrainSession 那样的签名：send(text)
    brain, _, _, _ = make(clock, session)
    brain.snapshot = lambda events: None
    brain.wake(clock(), "heartbeat")
    assert len(session.sent) == 1


def test_brain_snapshot_error_still_sends_text(clock, caplog):
    session = FakeSession()

    def broken(events):
        raise RuntimeError("身体不理")

    brain, _, _, _ = make(clock, session)
    brain.snapshot = broken
    with caplog.at_level(logging.WARNING):
        brain.wake(clock(), "heartbeat")
    assert len(session.sent) == 1 and "身体不理" in caplog.text


def test_brain_log_records_images_and_check(clock, caplog):
    run = FakeRun()
    shots = [{"src": "auto", "w": 960, "h": 540}]
    session = FakeSession({**ok("不说：x\n核对：①其实是懒洋洋大王"), "images": shots})
    brain, _, _, _ = make(clock, session, run=run)
    with caplog.at_level(logging.INFO):
        brain.wake(clock(), "heartbeat")
    assert run.entries[0]["images"] == shots and run.entries[0]["check"] == "①其实是懒洋洋大王"
    assert "核对：①其实是懒洋洋大王" in caplog.text


def test_brain_log_without_images_or_check(clock):
    run = FakeRun()
    brain, _, _, _ = make(clock, FakeSession(ok("好")), run=run)
    brain.wake(clock(), "heartbeat")
    assert "images" not in run.entries[0] and "check" not in run.entries[0]


def test_parse_check():
    assert parse_check("说了\n核对：A\n核对：B") == "A；B"
    assert parse_check("没有") == ""
