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
        self.last = button
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


# ---- 大脑的面板工具 ----
def dialog(clock, live=True, reading=DIALOG_READING, **kw):
    b, panels, ops, reader, events = setup(clock, live=live, **kw)
    panels.state = PanelState((DIALOG,))
    panels.readings = {UNKNOWN: reading}
    return b, panels, ops, events


def test_panel_read_lists_numbered_buttons(clock):
    b, *_ = dialog(clock)
    assert b.panel_read() == (
        "不认识的面板\n  标题：出错了\n  正文：网络连接断开，请重试\n"
        "  按钮：[1] 取消（可以按）  [2] 加入（不能按）  [3] 确定（要主人放行）"
    )


def test_panel_read_two_panels_and_blanks(clock):
    b, panels, _, _ = dialog(clock)
    panels.state = PanelState((DIALOG, EMOTE, CHAT_PANEL))
    panels.readings["emote_panel"] = PanelReading(EMOTE, "", "", (), 100.0)
    out = b.panel_read()
    assert out.endswith("动作面板（未核对）\n  标题：（没读到）\n  正文：（没有）\n  按钮：没认出按钮")
    assert "\n\n动作面板" in out and "聊天记录面板" not in out


def test_panel_read_none(clock):
    b, *_ = setup(clock)
    assert b.panel_read() == "没有开着的面板"


def test_panel_read_image(clock):
    b, *_ = dialog(clock)
    out = b.panel_read(image=True)
    assert out[0]["type"] == "text" and out[1]["type"] == "image"


def test_panel_tools_need_panels(clock):
    b, _, _, _ = body(clock)
    with pytest.raises(ToolError, match="没开面板识别"):
        b.panel_read()


def test_panel_press_retreat(clock):
    b, _, ops, _ = dialog(clock)
    b.panel_read()
    assert b.panel_press("1").startswith("按了「取消」。现在开着的面板：不认识的面板")
    assert ops.pressed == ["取消"]


def test_panel_press_unchanged_frame(clock):
    b, _, ops, _ = dialog(clock)
    ops.changed = False
    b.panel_read()
    assert "画面没变（可能没点中）" in b.panel_press("取消")


def test_panel_press_requires_recent_read(clock):
    b, *_ = dialog(clock)
    with pytest.raises(ToolError, match="先 panel_read"):
        b.panel_press("1")
    b.panel_read()
    clock.advance(16)
    with pytest.raises(ToolError, match="先 panel_read（15 秒内）"):
        b.panel_press("1")


def test_panel_press_bad_button_lists_choices(clock):
    b, *_ = dialog(clock)
    b.panel_read()
    for bad in ("9", "0", "不存在"):
        with pytest.raises(ToolError, match="现在的按钮：\\[1\\] 取消、\\[2\\] 加入、\\[3\\] 确定"):
            b.panel_press(bad)


def test_panel_press_same_name_takes_first(clock):
    other = Button("取消", Rect(900, 400, 60, 30), "retreat")
    b, _, ops, _ = dialog(clock, reading=replace(DIALOG_READING, buttons=(other, CANCEL)))
    b.panel_read()
    b.panel_press("取消")
    assert ops.pressed == ["取消"] and ops.last is other


def test_panel_press_never_even_with_permit(clock):
    b, _, ops, _ = dialog(clock)
    b._permits = [("加入", clock() + 60)]
    b.panel_read()
    with pytest.raises(ToolError, match="「加入」不能按（花钱、删好友、退出这类按钮，主人放行也不按）"):
        b.panel_press("加入")
    assert ops.pressed == []


def test_panel_press_other_needs_permit(clock):
    b, _, ops, _ = dialog(clock)
    b.panel_read()
    with pytest.raises(ToolError, match="卡洛在聊天里发「#允许 确定」后 60 秒内可以按一次"):
        b.panel_press("3")
    b._permits = [("确定", clock() + 60)]
    b.panel_press("3")
    assert ops.pressed == ["确定"] and b._permits == []
    b.panel_read()
    with pytest.raises(ToolError, match="#允许"):
        b.panel_press("3")


def test_panel_press_permit_expired(clock):
    b, *_ = dialog(clock)
    b._permits = [("确定", clock() + 60)]
    clock.advance(61)
    b.panel_read()
    with pytest.raises(ToolError, match="#允许"):
        b.panel_press("确定")


def test_panel_press_panel_gone(clock):
    b, panels, _, _ = dialog(clock)
    b.panel_read()
    panels.state = PanelState()
    with pytest.raises(ToolError, match="不认识的面板已经关了"):
        b.panel_press("1")


def test_panel_press_dry_run_keeps_permit(clock):
    b, _, ops, _ = dialog(clock, live=False)
    b._permits = [("确定", clock() + 60)]
    b.panel_read()
    assert b.panel_press("确定") == "dry-run：没真的点（会点「确定」）"
    assert b._permits and ops.pressed == []
    assert b.panel_press("确定", live=True).startswith("按了「确定」") and not b._permits


def test_panel_close_top(clock):
    b, _, ops, _ = dialog(clock)
    assert b.panel_close() == "关掉了不认识的面板"
    assert ops.closed == [(UNKNOWN, DIALOG_READING)]


def test_panel_close_fails(clock):
    b, *_ = dialog(clock, ok=False)
    with pytest.raises(ToolError, match="没关上不认识的面板"):
        b.panel_close()


def test_panel_close_none_and_dry(clock):
    b, *_ = setup(clock)
    assert b.panel_close() == "没有开着的面板"
    dry, _, ops, _ = dialog(clock, live=False)
    assert dry.panel_close() == "dry-run：没真的关（会关掉不认识的面板）" and ops.closed == []


def test_toolbox_panel_tools(clock):
    from skydango.brain.tools import ToolBox

    b, *_ = dialog(clock)
    tb = ToolBox(b)
    assert tb.run("panel_read", {})[1] is False
    out, err = tb.run("panel_press", {"button": 1})
    assert not err and out.startswith("按了「取消」") and tb.acted
    assert tb.run("panel_press", {"button": []})[1] is True
    assert tb.run("panel_close", {}) == ("关掉了不认识的面板", False)


def test_prompt_has_panel_rules():
    from skydango.brain.prompt import static_prompt
    from skydango.config import ReplyConfig

    assert "## 面板（panel_read / panel_press / panel_close）" in static_prompt(ReplyConfig())


def test_panel_press_rechecks_panel_before_tapping(clock):
    # 读完之后弹框换了（都叫 unknown）：原来「取消」的位置现在是别的按钮 → 不按，要求重新读
    b, panels, ops, _ = dialog(clock)
    b.panel_read()
    panels.readings[UNKNOWN] = replace(DIALOG_READING, buttons=(Button("加入", CANCEL.box, "never"), OK))
    with pytest.raises(ToolError, match="面板变了，重新 panel_read"):
        b.panel_press("取消")
    assert ops.pressed == []


def test_panel_press_uses_fresh_button_box(clock):
    b, panels, ops, _ = dialog(clock)
    b.panel_read()
    moved = Button("取消", Rect(310, 420, 60, 30), "retreat")
    panels.readings[UNKNOWN] = replace(DIALOG_READING, buttons=(moved, JOIN, OK))
    b.panel_press("取消")
    assert ops.last is moved
