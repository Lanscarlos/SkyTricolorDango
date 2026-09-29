import json
import threading
from types import SimpleNamespace

import pytest
from test_brain_body import FakeCamera, FakeEmotes, FakeEnv, body, friend_body

from skydango.brain.manual import ManualControl


def here(b):
    """让 body.call 在本线程直接执行（身体线程里调 call 就是直接执行）。"""
    b._thread = threading.get_ident()
    return b


def manual_events(events):
    return [e for e in events.drain() if e.kind == "manual"]


def test_say_runs_live_and_tells_brain(clock):
    b, device, _, events = body(clock)  # dry-run
    out = ManualControl(here(b), events=events).run("say", {"text": "晚安"})
    assert out == {"ok": True, "text": "已发送：晚安"}
    assert ("text", "晚安") in device.calls
    [e] = manual_events(events)
    assert "说了「晚安」" in e.text and "已发送" in e.text


def test_tool_errors_become_not_ok_and_no_event(clock):
    b, _, _, events = body(clock)
    out = ManualControl(here(b), events=events).run("say", {"text": "我是真人啊"})
    assert out["ok"] is False and "真人" in out["text"]
    assert manual_events(events) == []


@pytest.mark.parametrize(
    "action,args",
    [
        ("fly", {}),
        ("say", {}),
        ("say", {"text": 3}),
        ("say", {"text": "  "}),
        ("emote", {"name": ""}),
        ("camera", {"action": "left", "steps": 5}),
        ("camera", {"action": "left", "steps": 0}),
        ("camera", {"action": "spin"}),
        ("check_friend", {"x": "a", "y": 1}),
        ("check_friend", {"x": 1}),
    ],
)
def test_bad_arguments_raise_value_error(clock, action, args):
    b, _, _, _ = body(clock, emotes=FakeEmotes(), camera=FakeCamera())
    with pytest.raises(ValueError):
        ManualControl(here(b)).run(action, args)


def test_emote_force_and_camera(clock):
    emotes, cam = FakeEmotes(), FakeCamera()
    b, _, _, events = body(clock, emotes=emotes, camera=cam)
    m = ManualControl(here(b), events=events)
    b.holding = "懒洋洋大王"
    assert m.run("emote", {"name": "鞠躬"})["ok"] is False
    assert m.run("emote", {"name": "鞠躬", "force": True}) == {"ok": True, "text": "做了「鞠躬」"}
    assert m.run("camera", {"action": "left", "steps": 2})["ok"] is True and cam.moves == [("left", 2)]
    assert m.run("camera", {"action": "right"})["ok"] is True and cam.moves[-1] == ("right", 1)  # 步数默认 1
    assert m.run("camera_reset", {})["ok"] is True and cam.resets == 1
    texts = [e.text for e in manual_events(events)]
    assert any("做了动作「鞠躬」" in t for t in texts) and any("转了视角（左转 ×2）" in t for t in texts)
    assert any("把视角复位" in t for t in texts)


def test_look_around_uses_sweep_or_eyes(clock):
    cam = FakeCamera()

    class SweepEnv(FakeEnv):
        def sweep(self, frames, spin):
            return SimpleNamespace(text=lambda: "前面有懒洋洋大王")

    b, _, _, events = body(clock, camera=cam, env=SweepEnv())
    assert ManualControl(here(b), events=events).run("look_around", {}) == {"ok": True, "text": "前面有懒洋洋大王"}
    assert cam.spins == 1

    class Eyes:
        def describe_around(self, frames, now):
            return f"四周（{len(frames)} 张）：雨林"

    b, _, _, _ = body(clock, camera=cam)
    assert ManualControl(here(b), eyes=Eyes()).run("look_around", {})["text"] == "四周（4 张）：雨林"
    b, _, _, _ = body(clock, camera=cam)
    assert ManualControl(here(b)).run("look_around", {})["text"].startswith("转完了")


def test_track_and_stop_from_panel(clock):
    from test_brain_body import track_body

    b, _, events = track_body(clock, live=False)  # 大脑 dry-run：手动的照样真做
    m = ManualControl(here(b), events=events)
    out = m.run("track", {"name": "懒洋洋大王", "seconds": 20})
    assert out["ok"] is True and out["text"].startswith("开始盯着懒洋洋大王")
    assert b.skills.active.name == "track" and b.skills.active.timeout == 25
    assert m.run("stop_task", {}) == {"ok": True, "text": "停下了：盯着懒洋洋大王"}
    assert b.skills.active is None
    texts = [e.text for e in manual_events(events)]
    assert any("盯着懒洋洋大王（20 秒）" in t for t in texts) and any("停下" in t for t in texts)
    out = m.run("track", {"name": "阿白"})  # 画面里没有：身体的护栏照旧
    assert out["ok"] is False and "没看到" in out["text"]


@pytest.mark.parametrize("args", [{}, {"name": " "}, {"name": "小明", "seconds": 0}, {"name": "小明", "seconds": 61},
                                  {"name": "小明", "seconds": "30"}])
def test_track_bad_arguments(clock, args):
    b, _, _, _ = body(clock, camera=FakeCamera())
    with pytest.raises(ValueError):
        ManualControl(here(b)).run("track", args)


def test_options_track(clock):
    from test_brain_body import track_body

    b, _, _ = track_body(clock)
    opts = ManualControl(b).options()
    assert opts["track"] is True and opts["max_track_seconds"] == 60
    bare, _, _, _ = body(clock, camera=FakeCamera())  # 没开感知层
    assert ManualControl(bare).options()["track"] is False


def test_camera_reset_gets_longer_timeout(clock):
    from skydango.brain.tools import RESET_TIMEOUT

    b, _, _, _ = body(clock, camera=FakeCamera())
    assert ManualControl(b)._bind("camera_reset", {})[2] == RESET_TIMEOUT


def test_check_friend_returns_text_only(clock):
    b, checker, _, events = friend_body(clock, live=False)
    out = ManualControl(here(b), events=events).run("check_friend", {"x": 1200, "y": 450})
    assert out["ok"] is True and "已经关上" in out["text"] and checker.calls == [(1200, 450)]
    assert "base64" not in json.dumps(out) and "data" not in out
    assert any("点了画面上 (1200, 450) 的人" in e.text for e in manual_events(events))


def test_options(clock):
    b, _, _, _ = body(clock, emotes=FakeEmotes(), camera=FakeCamera())
    assert ManualControl(b).options() == {
        "emotes": ["鞠躬"],
        "camera": ["left", "right", "up", "down", "zoom_in", "zoom_out"],
        "max_steps": 4,
        "friend_check": False,
        "max_chars": 40,
        "dry_run": True,
        "panels": False,
        "track": False,
        "max_track_seconds": 60,
    }
    bare, _, _, _ = body(clock)
    opts = ManualControl(bare).options()
    assert opts["emotes"] == [] and opts["camera"] == []


def test_unexpected_error_is_reported(clock):
    b, _, _, events = body(clock)

    def boom(text, live=False):
        raise RuntimeError("设备断了")

    b.say = boom
    out = ManualControl(here(b), events=events).run("say", {"text": "晚安"})
    assert out["ok"] is False and out["text"].startswith("出错了") and "设备断了" in out["text"]
    assert manual_events(events) == []


def test_eyes_error_after_look_around_is_reported(clock):
    class BrokenEyes:
        def describe_around(self, frames, now):
            raise RuntimeError("眼睛超时")

    b, _, _, _ = body(clock, camera=FakeCamera())
    out = ManualControl(here(b), eyes=BrokenEyes()).run("look_around", {})
    assert out["ok"] is False and "眼睛超时" in out["text"]


def test_panel_close_is_live_and_tells_brain(clock):
    from test_brain_panels import DIALOG, DIALOG_READING, FakeOps, FakePanels

    from skydango.vision.panels import UNKNOWN, PanelState

    panels = FakePanels()
    panels.state = PanelState((DIALOG,))
    panels.readings = {UNKNOWN: DIALOG_READING}
    ops = FakeOps(panels)
    b, _, _, events = body(clock, panels=panels, panel_ops=ops)  # dry-run：手动的照样真关
    mc = ManualControl(here(b), events=events)
    assert mc.options()["panels"] is True
    assert mc.run("panel_read", {})["text"].startswith("不认识的面板")
    assert mc.run("panel_close", {}) == {"ok": True, "text": "关掉了不认识的面板"}
    assert ops.closed == [(UNKNOWN, DIALOG_READING)]
    assert [e.text for e in manual_events(events)][-1] == "主人在面板上手动让团子关了面板：关掉了不认识的面板"


def test_options_without_panels(clock):
    b, _, _, events = body(clock)
    assert ManualControl(here(b), events=events).options()["panels"] is False
