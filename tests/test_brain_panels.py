from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_brain_body import FakeCamera, FakeEnv, FakeSocial, body, msg

from skydango.brain.body import ToolError
from skydango.vision.bubbles import Rect
from skydango.vision.panels import CHAT, UNKNOWN, Button, Panel, PanelChange, PanelReading, PanelState

EMOTE = Panel("emote_panel", "动作面板", Rect(900, 0, 300, 720), False, 20)
EMOTE_OK = replace(EMOTE, verified=True)
CHAT_PANEL = Panel(CHAT, "聊天记录面板", Rect(0, 0, 400, 600), False, 10, ("say", "emote", "camera", "check_friend", "social"))
DIALOG = Panel(UNKNOWN, "不认识的面板", Rect(200, 100, 800, 500), False, 100)
CANCEL = Button("取消", Rect(300, 400, 60, 30), "retreat")
JOIN = Button("加入", Rect(700, 400, 60, 30), "never")
OK = Button("确定", Rect(500, 400, 60, 30), "other")
DIALOG_READING = PanelReading(DIALOG, "出错了", "网络连接断开，请重试", (CANCEL, JOIN, OK), 100.0)


class FakePanels:
    def __init__(self):
        self.state = PanelState()
        self.changes = []
        self.blocks = {}
        self.expected_log = []
        self.readings = {}
        self.reads = []
        self.cards = {"emote_panel": SimpleNamespace(close_auto=True)}

    def observe(self, frame, now):
        return self.state

    def pop_changes(self):
        out, self.changes = self.changes, []
        return out

    def blocking(self, action):
        return list(self.blocks.get(action, []))

    @contextmanager
    def expect(self, name):
        self.expected_log.append(name)
        yield

    def read(self, frame, panel, now):
        self.reads.append(panel.name)
        return self.readings[panel.name]


class FakeOps:
    def __init__(self, panels, ok=True):
        self.panels = panels
        self.ok = ok
        self.closed = []
        self.pressed = []
        self.changed = True

    def close(self, panel, reading=None):
        self.closed.append((panel.name, reading))
        if self.ok:
            for action in self.panels.blocks:
                self.panels.blocks[action] = [p for p in self.panels.blocks[action] if p.name != panel.name]
            self.panels.state = PanelState(tuple(p for p in self.panels.state.panels if p.name != panel.name))
        return self.ok

    def press(self, reading, button):
        self.pressed.append(button.text)
        return self.panels.state, self.changed


def setup(clock, live=True, ok=True, **kw):
    panels = FakePanels()
    ops = FakeOps(panels, ok)
    kw.setdefault("env", FakeEnv())
    b, device, reader, events = body(clock, live=live, panels=panels, panel_ops=ops, **kw)
    return b, panels, ops, reader, events


def texts(events, kind="panel"):
    return [e.text for e in events.drain() if e.kind == kind]


# ---- 事件和状态 ----
def test_panel_open_event_and_status(clock):
    b, panels, _, _, events = setup(clock)
    panels.changes = [PanelChange("open", EMOTE)]
    panels.state = PanelState((EMOTE,))
    b.step()
    assert texts(events) == ["开了：动作面板（未核对）"]
    assert "开着的面板：动作面板（未核对）" in b.status()


def test_panel_close_event(clock):
    b, panels, _, _, events = setup(clock)
    panels.changes = [PanelChange("close", EMOTE)]
    b.step()
    assert texts(events) == ["关了：动作面板"]
    assert "开着的面板" not in b.status()


def test_chat_log_changes_ignored(clock):
    b, panels, _, _, events = setup(clock)
    panels.changes = [PanelChange("open", CHAT_PANEL)]
    panels.state = PanelState((CHAT_PANEL,))
    b.step()
    assert texts(events) == [] and "开着的面板" not in b.status()


def test_unknown_open_event_text(clock):
    b, panels, _, _, events = setup(clock)
    reading = PanelReading(DIALOG, "出错了", "网络连接断开，请重试", (Button("确定", Rect(0, 0, 1, 1), "other"),), 100.0)
    panels.changes = [PanelChange("open", DIALOG, reading)]
    b.step()
    assert texts(events) == ["出现不认识的面板：「出错了」网络连接断开，请重试，按钮：确定"]


def test_text_card_open_event_has_reading(clock):
    b, panels, _, _, events = setup(clock)
    invite = Panel("shared_invite", "共享空间邀请", Rect(0, 0, 10, 10), False, 90)
    reading = PanelReading(invite, "", "这位旅人正在共享空间内", (CANCEL,), 100.0)
    panels.changes = [PanelChange("open", invite, reading)]
    b.step()
    assert texts(events) == ["开了：共享空间邀请（未核对）：这位旅人正在共享空间内，按钮：取消"]


def test_env_held_while_panel_open(clock):
    env = FakeEnv()
    b, panels, _, _, _ = setup(clock, env=env)
    panels.state = PanelState((CHAT_PANEL, EMOTE))
    b.step()
    b.step()
    assert env.holds == [("hold", "panel")]
    panels.state = PanelState((CHAT_PANEL,))
    b.step()
    assert env.holds == [("hold", "panel"), ("release", "panel")]


def test_held_wheel_expects_wheel_editor(clock):
    b, panels, _, _, _ = setup(clock)
    with b._held("wheel"):
        assert panels.expected_log == ["wheel_editor"]
    with b._held("camera"):
        pass
    assert panels.expected_log == ["wheel_editor", None]


# ---- 遮挡护栏 ----
def test_camera_autocloses_verified_panel(clock):
    cam = FakeCamera()
    b, panels, ops, _, events = setup(clock, camera=cam)
    panels.blocks["camera"] = [EMOTE_OK]
    assert b.camera_move("left", live=True).startswith("镜头现在")
    assert ops.closed == [("emote_panel", None)] and cam.moves == [("left", 1)]
    assert "顺手关掉了动作面板" in texts(events)


def test_camera_refused_by_unverified_panel(clock):
    cam = FakeCamera()
    b, panels, ops, _, _ = setup(clock, camera=cam)
    panels.blocks["camera"] = [EMOTE]
    with pytest.raises(ToolError, match="被「动作面板（未核对）」挡着：可以 panel_read 看看，或者 panel_close 关掉"):
        b.camera_move("left", live=True)
    assert ops.closed == [] and cam.moves == []


def test_unknown_panel_not_autoclosed(clock):
    b, panels, ops, _, _ = setup(clock, camera=FakeCamera())
    panels.blocks["camera"] = [DIALOG]
    with pytest.raises(ToolError, match="被「不认识的面板」挡着"):
        b.camera_move("left", live=True)
    assert ops.closed == []


def test_close_fails_raises_and_error_event(clock):
    b, panels, ops, _, events = setup(clock, ok=False, camera=FakeCamera())
    panels.blocks["camera"] = [EMOTE_OK]
    with pytest.raises(ToolError, match="想关没关上"):
        b.camera_move("left", live=True)
    assert "动作面板挡着，想关没关上" in texts(events, "error")


def test_dry_run_notes_would_close(clock):
    b, panels, ops, _, _ = setup(clock, live=False, camera=FakeCamera())
    panels.blocks["camera"] = [EMOTE_OK]
    assert "（真执行时会先关掉动作面板）" in b.camera_move("left")
    assert ops.closed == []


def test_say_refused_by_unknown_panel(clock):
    b, panels, _, _, _ = setup(clock)
    panels.blocks["say"] = [DIALOG]
    with pytest.raises(ToolError, match="挡着"):
        b.say("你好", live=True)
    assert b.said == []  # 被挡了不占说话的名额


def test_social_skipped_when_blocked(clock):
    class CountingSocial(FakeSocial):
        calls = 0

        def handle(self, requests, now):
            CountingSocial.calls += 1
            return super().handle(requests, now)

    env = FakeEnv()
    env.requests = {"k": SimpleNamespace(name="懒洋洋大王", kind="hand")}
    social = CountingSocial()
    b, panels, _, _, _ = setup(clock, env=env, social=social)
    panels.blocks["social"] = [EMOTE]
    b.step()
    assert CountingSocial.calls == 0
    panels.blocks["social"] = []
    b.step()
    assert CountingSocial.calls == 1


def test_owner_permit_recorded(clock):
    b, _, _, reader, events = setup(clock)
    b.cfg.brain.owner_name = "卡洛"
    reader.batches = [[msg("#允许 确定", speaker="卡洛")]]
    b.step()
    assert b._permits == [("确定", clock() + 60.0)]
    assert [e.kind for e in events.drain()] == ["owner_command"]


def test_no_panels_means_no_guard(clock):
    cam = FakeCamera()
    b, _, _, _ = body(clock, live=True, camera=cam)
    assert b.clear_view("camera") == ""
    b.camera_move("left", live=True)
    assert cam.moves == [("left", 1)]
