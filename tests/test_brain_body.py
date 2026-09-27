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
