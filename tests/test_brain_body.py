import threading

import numpy as np
import pytest
from conftest import FakeDevice, scene

from skydango.brain.body import Body, ToolError
from skydango.brain.events import EventQueue
from skydango.chat.reader import Message
from skydango.chat.sender import ChatSender
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.game.social import Request
from skydango.vision.bubbles import Rect


class FakeReader:
    def __init__(self):
        self.batches = []  # 每次 read 依次返回一批消息
        self.panel_closed_since = None

    def read(self, frame, now):
        return self.batches.pop(0) if self.batches else []

    def panel_visible(self, frame):
        return True


class FakeEnv:
    def __init__(self):
        self.near = []
        self.requests = {}
        self.circles = {}
        self.labels = {}

    def observe(self, frame, now, panel_visible):
        pass

    def nearby(self, now):
        return list(self.near)


class FakeSocial:
    def __init__(self):
        self.to_handle = []
        self.policy_calls = []

    def handle(self, requests, now):
        out, self.to_handle = self.to_handle, []
        return out

    def set_policy(self, who, kind, accept):
        self.policy_calls.append((who, kind, accept))

    def describe_policy(self):
        return "好友默认接受：牵手"


def msg(text, speaker="懒洋洋大王"):
    return Message(text, Rect(0, 0, 10, 10), 0.0, speaker)


def body(clock, live=False, frames=None, **kw):
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.reply.dry_run = not live
    cfg.reply.disclosure_prefix = ""
    cfg.sender.open_chat_key = 28
    device = FakeDevice(frames or [scene()])
    reader = FakeReader()
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, "")
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    events = EventQueue(clock=clock)
    b = Body(cfg, device, reader, sender, self_filter, events, clock=clock, sleep=lambda s: None,
             wall=lambda: 1_790_000_000.0, **kw)
    return b, device, reader, events


def test_chat_messages_become_events(clock):
    b, _, reader, events = body(clock)
    reader.batches = [[msg("在吗"), msg("hi", speaker="")]]
    b.step()
    assert [e.text for e in events.drain()] == ["聊天  懒洋洋大王：在吗", "聊天  （看不出是谁）：hi"]
    assert list(b.chat)[-1][1:] == ("", "hi") and len(b.heard) == 2


def test_people_arrive_leave_and_requests(clock):
    env, social = FakeEnv(), FakeSocial()
    b, _, _, events = body(clock, env=env, social=social)
    env.near = ["懒洋洋大王"]
    env.requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (0, 0), 100.0)}
    social.to_handle = ["懒洋洋大王:hand"]
    b.step()
    assert [e.kind for e in events.drain()] == ["arrive", "request", "accepted"]
    env.near, env.requests = [], {}
    b.step()
    (e,) = events.drain()
    assert e.kind == "leave" and "懒洋洋大王" in e.text


def test_holding_is_guessed_from_circle(clock):
    env, social = FakeEnv(), FakeSocial()
    b, _, _, events = body(clock, env=env, social=social)
    env.requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (0, 0), 100.0)}
    social.to_handle = ["懒洋洋大王:hand"]
    b.step()
    events.drain()
    env.requests = {}
    clock.advance(3)
    env.circles["懒洋洋大王"] = (None, clock())  # 走过去之后圆圈没了
    b.step()
    assert b.holding == "懒洋洋大王" and events.drain()[-1].kind == "holding"
    clock.advance(5)
    env.circles["懒洋洋大王"] = ("star", clock())  # ✦ 又出现了
    b.step()
    assert b.holding is None and events.drain()[-1].kind == "released"


def test_black_screen_and_scene_change(clock):
    b, device, _, events = body(clock)
    b.step()
    device.frames = [np.zeros((720, 1280, 3), np.uint8)]
    clock.advance(1.5)
    b.step()
    assert b.blackout and events.drain()[-1].text.startswith("画面整屏黑")
    device.frames = [scene()]
    clock.advance(1.5)
    b.step()
    assert not b.blackout and events.drain()[-1].text == "画面恢复了"
    device.frames = [np.full((720, 1280, 3), 230, np.uint8)]
    clock.advance(1.5)
    b.step()
    assert events.drain()[-1].text.startswith("画面变化很大")


def test_panel_lost_and_back(clock):
    b, _, reader, events = body(clock)
    reader.panel_closed_since = 60.0  # 已经关了 40 秒
    b.step()
    assert events.drain()[-1].text.startswith("聊天记录面板关了")
    reader.panel_closed_since = None
    b.step()
    assert events.drain()[-1].text == "聊天记录面板又开了"


def test_call_runs_in_body_thread_and_times_out(clock):
    b, _, _, _ = body(clock)
    assert b.call(lambda: 5) == 5  # 身体线程自己调用：直接执行
    results = []
    t = threading.Thread(target=lambda: results.append(b.call(threading.get_ident)))
    t.start()
    while t.is_alive():
        b.step()
    assert results == [threading.get_ident()]  # 在跑 step 的线程（身体线程）里执行

    b.cfg.brain.command_timeout = 0.05
    errors = []

    def nobody_steps():
        try:
            b.call(lambda: 1)
        except ToolError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=nobody_steps)
    t.start()
    t.join(2)
    assert errors and "超时" in errors[0]
    b.step()  # 超时的命令已经取消，不会再执行，也不报错


def test_say_filters_rate_limits_and_records(clock):
    b, device, reader, _ = body(clock, live=True)
    reader.batches = [[msg("你是真人吗")]]
    b.step()
    with pytest.raises(ToolError, match="真人"):
        b.say("我是真人啊")  # 身份底线：硬过滤
    assert b.say("哈哈你猜") == "已发送：哈哈你猜"
    assert ("text", "哈哈你猜") in device.calls
    with pytest.raises(ToolError, match="太快"):
        b.say("再说一句")  # reply.min_interval = 3 秒
    assert b.heard == []


def test_say_in_dry_run_does_not_touch_device(clock):
    b, device, _, _ = body(clock)
    assert b.say("在呢").startswith("dry-run")
    assert device.calls == [] and b.said == ["在呢"]


def test_say_writes_memory_when_live(clock, tmp_path):
    from skydango.chat.memory import MemoryStore

    store = MemoryStore(tmp_path)
    b, _, reader, _ = body(clock, live=True, store=store)
    reader.batches = [[msg("在吗")]]
    b.step()
    b.say("在呢")
    (turn,) = store.history.all()
    assert "懒洋洋大王：「在吗」" in turn.user and turn.reply == "在呢"


class FakeEmotes:
    def __init__(self):
        self.done = []

    def available(self):
        return ["鞠躬"]

    def perform(self, name):
        self.done.append(name)

    def pretend(self, name):
        pass


def test_emote_refused_while_holding_unless_forced(clock):
    emotes = FakeEmotes()
    b, _, _, _ = body(clock, live=True, emotes=emotes)
    with pytest.raises(ToolError, match="能做的：鞠躬"):
        b.emote("跳舞")
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="force=true"):
        b.emote("鞠躬")
    assert b.emote("鞠躬", force=True) == "做了「鞠躬」" and emotes.done == ["鞠躬"]


def test_look_rate_limited_and_lists_names(clock):
    env = FakeEnv()
    env.labels = {"懒洋洋大王": (560, 200, 160, 44, clock())}
    b, _, _, _ = body(clock, env=env)
    img, note = b.look()
    assert img["type"] == "image"
    assert "懒洋洋大王：名字在 (640, 200)" in note["text"]  # 截图本来就是 1280 宽，不缩放
    with pytest.raises(ToolError, match="刚看过"):
        b.look()
    clock.advance(6)
    b.look()


def test_look_at_crops_region(clock):
    b, _, _, _ = body(clock)
    img, note = b.look_at(100, 100, 200, 100)
    assert img["type"] == "image" and "(100, 100) 起 200×100" in note["text"]
    with pytest.raises(ToolError):
        b.look_at(5000, 0, 10, 10)


class FakeCamera:
    def __init__(self):
        self.moves = []

    def move(self, action, steps):
        self.moves.append((action, steps))
        return "左转了 1 步"

    def reset(self):
        return "镜头转回原位了"

    def describe(self):
        return "原位"


def test_camera_refused_in_blackout_and_dry_run(clock):
    cam = FakeCamera()
    b, _, _, _ = body(clock, live=True, camera=cam)
    assert b.camera_move("left", 1) == "镜头现在：左转了 1 步"
    b.blackout = True
    with pytest.raises(ToolError, match="黑"):
        b.camera_move("left", 1)
    dry, _, _, _ = body(clock, camera=FakeCamera())
    assert dry.camera_move("left", 1).startswith("dry-run")


def test_policy_and_status(clock):
    env, social = FakeEnv(), FakeSocial()
    env.near = ["懒洋洋大王"]
    b, _, _, _ = body(clock, env=env, social=social)
    with pytest.raises(ToolError, match="不认识"):
        b.set_policy("*", "dance", True)
    assert b.set_policy("懒洋洋大王", "piggyback", False).startswith("现在的规则")
    assert social.policy_calls == [("懒洋洋大王", "piggyback", False)]
    s = b.status()
    assert "身边的好友：懒洋洋大王" in s and "dry-run" in s and "互动规则" in s


def test_chat_log_includes_own_lines(clock):
    b, _, reader, _ = body(clock)
    reader.batches = [[msg("在吗")]]
    b.step()
    b.say("在呢")
    text = b.chat_log(10)
    assert "懒洋洋大王：在吗" in text and "我：在呢" in text


def test_fallback_replies_when_brain_offline(clock):
    from skydango.chat.responder import Reply

    class Responder:
        def __init__(self):
            self.batches = []

        def reply(self, batch):
            self.batches.append(batch)
            return Reply("在呢")

    fallback = Responder()
    b, device, reader, events = body(clock, live=True, fallback=fallback)
    b.brain_offline = lambda now: True
    reader.batches = [[msg("在吗")]]
    b.step()
    assert events.drain() == [] and fallback.batches == []  # 不排给大脑；先等 debounce
    clock.advance(2)
    b.step()
    assert ("text", "在呢") in device.calls
    assert events.drain()[-1].kind == "fallback"
