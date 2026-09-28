import base64
import threading
from concurrent.futures import Future

import cv2
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
    assert [e.text for e in events.drain()] == ["聊天  懒洋洋大王：「在吗」", "聊天  （看不出是谁）：「hi」"]
    assert list(b.chat)[-1][1:] == ("", "hi") and len(b.heard) == 2


def test_owner_command_opens_authorization_window(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    b.cfg.brain.owner_window = 30.0
    reader.batches = [[msg("#过来")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["owner_command"]
    assert b._owner_window_until == clock() + 30.0


def test_owner_command_requires_exact_speaker_match(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("#过来", speaker="懒洋洋大王2")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
    assert b._owner_window_until == float("-inf")


def test_non_hash_message_from_owner_is_normal_chat(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("过来呀")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
    assert b._owner_window_until == float("-inf")


def test_owner_command_disabled_when_owner_name_empty(clock):
    b, _, reader, events = body(clock)
    assert b.cfg.brain.owner_name == ""
    reader.batches = [[msg("#过来")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]


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
    ran = []  # 取消成功的话这个命令不该被执行到

    def nobody_steps():
        try:
            b.call(lambda: ran.append(1))
        except ToolError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=nobody_steps)
    t.start()
    t.join(2)
    assert errors and "超时" in errors[0]
    b.step()  # 超时的命令已经取消，不会再执行，也不报错
    assert ran == []  # 真的没有被执行到


def test_call_timeout_message_uses_actual_timeout_value(clock):
    b, _, _, _ = body(clock)  # cfg.brain.command_timeout 默认 15 秒，不应该出现在消息里
    errors = []

    def nobody_steps():
        try:
            b.call(lambda: 1, timeout=0.05)
        except ToolError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=nobody_steps)
    t.start()
    t.join(2)
    assert errors and "0 秒" in errors[0] and "15" not in errors[0]


def test_call_timeout_after_command_already_started_no_retry(clock):
    """命令已经被身体线程取出并开始执行时调用者超时：不能说“超时了”，得说“别重试”。"""
    b, _, _, _ = body(clock)
    b.cfg.brain.command_timeout = 0.05
    started = threading.Event()
    release = threading.Event()

    def fn():
        started.set()
        assert release.wait(2)
        return 1

    def body_runner():
        # 用阻塞 get() 而不是 _run_commands()（get_nowait），避免和入队产生竞争
        cmd_fn, fut = b._commands.get()
        if fut.set_running_or_notify_cancel():
            try:
                fut.set_result(cmd_fn())
            except BaseException as exc:  # noqa: BLE001 - 和 _run_commands 保持一致
                fut.set_exception(exc)

    t_body = threading.Thread(target=body_runner)
    t_body.start()

    errors = []

    def caller():
        try:
            b.call(fn)
        except ToolError as exc:
            errors.append(str(exc))

    t_caller = threading.Thread(target=caller)
    t_caller.start()

    assert started.wait(2)  # 命令确实已经开始执行了
    t_caller.join(2)
    assert errors and "别重试" in errors[0] and "已经开始" in errors[0]

    release.set()
    t_body.join(2)


def test_keyboard_interrupt_in_command_propagates_out_of_step(clock):
    """Ctrl+C 是在身体线程 fn() 执行时来的：不能被吞进 Future 里让 step() 若无其事地继续。"""
    b, _, _, _ = body(clock)
    fut: Future = Future()

    def boom():
        raise KeyboardInterrupt()

    b._commands.put((boom, fut))
    with pytest.raises(KeyboardInterrupt):
        b.step()
    with pytest.raises(KeyboardInterrupt):
        fut.result(timeout=0)  # Future 也记下了这个异常


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


def test_sensing_error_does_not_block_chat_and_commands(clock):
    class BrokenEnv(FakeEnv):
        def observe(self, frame, now, panel_visible):
            raise RuntimeError("OCR 出错")

    b, _, reader, events = body(clock, env=BrokenEnv())
    reader.batches = [[msg("在吗")]]
    b.step()
    kinds = [e.kind for e in events.drain()]
    assert "chat" in kinds and "error" in kinds


def test_screenshot_error_with_empty_message(clock):
    b, device, _, events = body(clock)

    def broken():
        raise RuntimeError()

    device.screenshot = broken
    b.step()
    assert events.drain()[-1].text == "截图失败：RuntimeError"


def test_holding_is_cleared_when_friend_leaves(clock):
    env = FakeEnv()
    b, _, _, events = body(clock, env=env)
    env.near = ["懒洋洋大王"]
    b.step()
    b.holding = "懒洋洋大王"
    env.near = []
    b.step()
    assert b.holding is None
    assert any(e.kind == "released" for e in events.drain())


def test_strangers_can_only_get_candle(clock):
    social = FakeSocial()
    b, _, _, _ = body(clock, social=social)
    for kind in ("hand", "hug", "*"):
        with pytest.raises(ToolError, match="点火"):
            b.set_policy("stranger", kind, True)
    b.set_policy("stranger", "candle", True)
    b.set_policy("stranger", "hand", False)  # 不接总是可以的
    assert social.policy_calls == [("stranger", "candle", True), ("stranger", "hand", False)]


def test_shutdown_fails_queued_commands_and_refuses_new_ones(clock):
    b, _, _, _ = body(clock)
    results = []

    def caller():
        try:
            b.call(lambda: "做了")
        except ToolError as exc:
            results.append(str(exc))

    t = threading.Thread(target=caller)
    t.start()
    while b._commands.empty():
        pass
    b.shutdown()
    t.join(2)
    assert results and "停" in results[0]
    with pytest.raises(ToolError, match="停"):
        b.call(lambda: 1)


def test_look_at_crops_the_last_look(clock):
    b, device, _, _ = body(clock)
    b.look()
    device.frames = [np.zeros((720, 1280, 3), np.uint8)]  # 之后画面变黑了
    img, _ = b.look_at(100, 100, 200, 100)
    data = np.frombuffer(base64.standard_b64decode(img["source"]["data"]), np.uint8)
    assert cv2.imdecode(data, cv2.IMREAD_COLOR).mean() > 20  # 裁的是上次 look 的图，不是新截的黑图


def test_capture_around(clock):
    class AroundCamera(FakeCamera):
        def around(self, capture):
            return [capture() for _ in range(4)]

    dry, _, _, _ = body(clock, camera=AroundCamera())
    assert len(dry.capture_around()) == 1  # dry-run 不转，只看当前画面
    live, _, _, _ = body(clock, live=True, camera=AroundCamera())
    assert len(live.capture_around()) == 4
    live.blackout = True
    with pytest.raises(ToolError, match="黑"):
        live.capture_around()


def test_strangers_come_and_go(clock):
    class YoloEnv(FakeEnv):  # YOLO 感知层多一个 strangers()、keep
        keep = 5.0
        n = 0

        def strangers(self, now):
            return self.n

    env = YoloEnv()
    b, _, _, events = body(clock, env=env)
    env.n = 2
    b.step()
    b.step()  # 人数没从 0 变过来：不重复报
    (e,) = events.drain()
    assert e.kind == "stranger" and "2 个" in e.text
    assert "陌生人：2 个" in b.status()
    env.n, env.near = 0, ["懒洋洋大王"]
    b.step()
    env.near = []
    b.step()
    kinds = [(e.kind, e.text) for e in events.drain()]
    assert kinds[0] == ("arrive", "懒洋洋大王 来到身边") and ("stranger", "陌生人都走开了") in kinds
    assert any(k == "leave" and "5 秒" in t for k, t in kinds)
