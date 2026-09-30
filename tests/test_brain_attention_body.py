"""身体接上空闲注意力（plan 2026-09-30-idle-attention Task 5 / 6）：闲着才按、先看一眼再开面板、自己转出来的走近不算。"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_brain_body import FakeEnv, body, msg

from skydango.brain.body import ToolError
from skydango.vision.bubbles import Rect
from skydango.vision.people import Person
from skydango.vision.perception import Talker

FRIEND_X = 1600.0  # 好友在画面右边：注意力要按右把他拉向中间


class AttnEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.talkers_list: list[Talker] = []
        self.approach_list: list[tuple[str, float, float]] = []
        self.approaches: list[str] = []
        self.gestures: list[tuple[str, str]] = []

    def talkers(self, now):
        return list(self.talkers_list)

    def typing_seen(self, now, within=1.0, strangers=False):
        return any(t.friend or strangers for t in self.talkers_list)

    def recent_approaches(self, now, within=5.0):
        return [a for a in self.approach_list if now - a[2] <= within]

    def pop_approaches(self):
        out, self.approaches = self.approaches, []
        return out

    def pop_gestures(self):
        out, self.gestures = self.gestures, []
        return out


class NudgeCam:
    def __init__(self):
        self.nudges = []
        self.resets = 0

    def nudge(self, direction, seconds):
        self.nudges.append((direction, seconds))
        return seconds

    def reset(self, refine_seconds=None):
        self.resets += 1
        return "镜头转回原位了"

    def forget_reference(self):
        pass

    def describe(self):
        return "右转了一点" if self.nudges else "原位"


def friend_talking(env, start=0.0, x=FRIEND_X, name="小明"):
    env.talkers_list = [Talker(7, name, True, x, start, start)]


def attn_body(clock, live=True, panel_mode="auto", **kw):
    env = AttnEnv()
    cam = NudgeCam()
    b, dev, reader, events = body(clock, live=live, env=env, camera=cam, panel_mode=panel_mode,
                                  frames=[np.full((1080, 1920, 3), 90, np.uint8)], **kw)
    b.friend_names = lambda: ["小明", "小红"]
    return b, dev, reader, events, env, cam


def steps(b, clock, n, dt=0.7):
    for _ in range(n):
        clock.advance(dt)
        b.step()


def test_turns_toward_friend_talking_when_idle_in_auto_mode(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b.step()
    assert cam.nudges and cam.nudges[0][0] == "right"


def test_no_press_in_always_mode(clock):
    b, dev, reader, events, env, cam = attn_body(clock, panel_mode="always")
    friend_talking(env, clock())
    steps(b, clock, 3)
    assert cam.nudges == []


def test_no_press_when_panel_opened_this_round(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    reader.state.open = True  # 这一圈读到新消息：面板开着、进了聊天
    reader.batches = [[msg("在吗", speaker="小明")]]
    b.step()
    assert cam.nudges == []


def _block(kind, b, env, dev, clock):
    if kind == "ime":
        dev.shown = True
    elif kind == "bubble_box":
        b._bubble_at = clock()
    elif kind == "skill":
        from test_brain_skills import FakeSkill

        b.skills.start(b, FakeSkill())
    elif kind == "request":
        env.requests = {"小明": SimpleNamespace(name="小明", kind="hold", pos=(1600, 300))}
    elif kind == "pending_chat":
        b.events.put("chat", "小明：在吗")
    elif kind == "brain_busy":
        b.brain_busy = lambda: True
    elif kind == "blackout":
        b.blackout = True
    elif kind == "holding":
        b.holding = "小明"
    elif kind == "recent_emote":
        b.emotes = SimpleNamespace(last_any=clock(), on_wheel=lambda: [], available=lambda *a: [])
    elif kind == "disabled":
        b.cfg.attention.enabled = False
    elif kind == "still_mode":
        b.attention.set_mode("别动", None)
    elif kind == "no_camera":
        b.camera = None
    elif kind == "no_perception":
        b.env = FakeEnv()


@pytest.mark.parametrize("kind", ["ime", "bubble_box", "skill", "request", "pending_chat", "brain_busy", "blackout",
                                  "holding", "recent_emote", "disabled", "still_mode", "no_camera", "no_perception"])
def test_blocked_conditions_do_not_press(clock, kind):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    _block(kind, b, env, dev, clock)
    b._watch_attention(clock())
    assert cam.nudges == []


def test_dry_run_still_describes_target(clock):
    b, dev, reader, events, env, cam = attn_body(clock, live=False)
    friend_talking(env, clock())
    b.step()
    assert cam.nudges == [] and "在看：小明（在说话）" in b.status()


def test_look_first_holds_panel_then_releases_when_centered(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b.step()  # 看到气泡：面板等着开，注意力先转过去
    assert b.panel.pending == "bubble" and cam.nudges
    clock.advance(0.7)
    b.step()
    assert not reader.state.open  # 推迟中：面板没开
    friend_talking(env, clock(), x=1000)  # 转过来了，在中间
    clock.advance(0.7)
    b.step()
    clock.advance(0.2)
    b.step()
    assert reader.state.open  # 放手后面板照常开


def test_look_first_expires_after_two_seconds(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b.step()
    steps(b, clock, 4)  # 2.8 秒，人一直没转到中间
    assert reader.state.open


def test_self_caused_approach_is_dropped_before_panel_trigger(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b.step()
    assert cam.nudges
    env.talkers_list = []
    env.approaches = ["小红"]  # 刚按完键：框变大是自己转出来的
    env.approach_list = [("小红", 1500.0, clock())]
    clock.advance(0.1)
    b.step()
    assert "approach" not in [e.kind for e in events.drain()] and b.panel.pending != "approach"


def test_approach_after_settle_is_kept(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    clock.advance(5.0)
    env.approaches = ["小红"]
    env.approach_list = [("小红", 1500.0, clock())]
    b.step()
    assert "approach" in [e.kind for e in events.drain()]


def test_gesture_is_recorded_for_attention(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    env.labels = {"小红": (1550, 300, 100, 36, clock())}
    env.gestures = [("小红", "wave")]
    b.step()
    keys = [t.kind for t in b._attention_targets(clock())]
    assert "act_on_me" in keys


def test_pressing_does_not_restart_idle_emote_timer(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    due = b.reflexes._idle_due
    friend_talking(env, clock())
    b.step()
    assert cam.nudges and b.reflexes._idle_due == due


def test_pressing_clears_scene_change_reference(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b._ref_thumb = np.zeros((9, 16), np.uint8)
    b._watch_attention(clock())
    assert cam.nudges and b._ref_thumb is None


def test_status_and_viewer_show_attention_line(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    b.step()
    assert "在看：小明（在说话）" in b.status()
    shown = {}
    b.viewer = SimpleNamespace(update=lambda *a, **k: shown.update(k.get("info") or {}), wants=lambda: True)
    b._show(np.full((1080, 1920, 3), 90, np.uint8), clock(), [])
    assert shown.get("注意力", "").startswith("在看：小明")


def test_friends_use_name_keys_across_sources(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    friend_talking(env, clock())
    env.people_list = [Person(7, "friend", "小明", Rect(1550, 400, 100, 300), "右边", "近")]
    keys = {t.key for t in b._attention_targets(clock())}
    assert keys == {"n:小明"}
