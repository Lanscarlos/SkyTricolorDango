from contextlib import contextmanager
import base64
import json
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
        self.people_list = []
        self.holds = []  # ("hold" / "release", 原因)

    def observe(self, frame, now, panel_visible):
        pass

    def hold(self, reason):
        self.holds.append(("hold", reason))

    def release(self, reason):
        self.holds.append(("release", reason))

    @contextmanager
    def held(self, reason):
        self.hold(reason)
        try:
            yield
        finally:
            self.release(reason)

    def nearby(self, now):
        return list(self.near)

    def people(self, now):
        return list(self.people_list)


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


def test_release_is_seen_when_friend_shows_a_status_icon(clock):
    """松手后对方在看留影蜡烛：圆圈里是眼睛不是 ✦，也算松开了。"""
    env, social = FakeEnv(), FakeSocial()
    b, _, _, events = body(clock, env=env, social=social)
    env.requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (0, 0), 100.0)}
    social.to_handle = ["懒洋洋大王:hand"]
    b.step()
    env.requests = {}
    clock.advance(3)
    env.circles["懒洋洋大王"] = (None, clock())
    b.step()
    events.drain()
    clock.advance(5)
    env.circles["懒洋洋大王"] = ("eye", clock())
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


def test_manual_say_is_marked_and_kept_out_of_memory(clock, tmp_path):
    # 主人手动说的话不是 AI 的回复：运行记录里标 manual，不写聊天历史 / 长期记忆（模型会模仿 history），也不拿走大脑待回复的消息
    from skydango.chat.memory import MemoryStore

    class Run:
        def __init__(self):
            self.replies = []

        def record_reply(self, messages, reply, sent, manual=False):
            self.replies.append((list(messages), reply, sent, manual))

        def save_frame(self, frame, boxes):
            pass

    store, run = MemoryStore(tmp_path), Run()
    b, _, reader, _ = body(clock, live=True, store=store, run=run)
    reader.batches = [[msg("在吗")]]
    b.step()
    b.say("测试一下", live=True)
    assert store.history.all() == []
    assert run.replies == [([], "测试一下", True, True)]
    assert [m.text for m in b.heard] == ["在吗"]  # 大脑之后回复时还能配上这句


def test_run_dir_marks_manual_replies(tmp_path):
    from skydango.runlog import RunDir

    cfg = Config()
    cfg.run.dir = str(tmp_path)
    run = RunDir.create(cfg, "dry")
    run.record_reply([], "测试一下", sent=True, manual=True)
    run.record_reply([], "好呀", sent=True)
    lines = [json.loads(line) for line in (run.path / "replies.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["manual"] is True and "manual" not in lines[1]


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
        self.resets = 0
        self.arounds = 0
        self.spins = 0

    def move(self, action, steps):
        self.moves.append((action, steps))
        return "左转了 1 步"

    def reset(self):
        self.resets += 1
        return "镜头转回原位了"

    def around(self, capture):
        self.arounds += 1
        return [capture() for _ in range(4)]

    def spin(self, capture, turns, seconds, fps):
        from types import SimpleNamespace

        self.spins += 1
        frame = capture()
        return SimpleNamespace(before=frame, frames=[(1.0, frame)], after=frame, panel_reopened=True, blackout=False)

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


# ---- 手动控制：live=True 时 dry-run 下也真执行，护栏照旧 ----
def test_live_say_in_dry_run_sends_and_remembers(clock):
    b, device, _, _ = body(clock)  # dry-run
    assert b.say("晚安", live=True) == "已发送：晚安"
    assert ("text", "晚安") in device.calls
    assert b.self_filter.is_self("晚安", clock())  # 读聊天不会把团子自己这句当成别人说的


def test_live_say_still_filtered_and_rate_limited(clock):
    b, _, _, _ = body(clock)
    with pytest.raises(ToolError, match="真人"):
        b.say("我是真人啊", live=True)
    b.say("你好", live=True)
    with pytest.raises(ToolError, match="太快"):
        b.say("再说一句", live=True)


def test_live_emote_and_camera_in_dry_run(clock):
    emotes, cam = FakeEmotes(), FakeCamera()
    b, _, _, _ = body(clock, emotes=emotes, camera=cam)
    assert b.emote("鞠躬", live=True) == "做了「鞠躬」" and emotes.done == ["鞠躬"]
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="force=true"):
        b.emote("鞠躬", live=True)
    assert b.camera_move("left", 2, live=True) == "镜头现在：左转了 1 步" and cam.moves == [("left", 2)]
    assert not b.camera_reset(live=True).startswith("dry-run") and cam.resets == 1
    b.blackout = True
    with pytest.raises(ToolError, match="黑"):
        b.camera_move("left", 1, live=True)


def test_shutdown_resets_camera_moved_by_hand_in_dry_run(clock):
    cam = FakeCamera()
    b, _, _, _ = body(clock, camera=cam)  # dry-run
    b.camera_move("left", 1, live=True)
    b.shutdown()
    assert cam.resets == 1


def test_capture_and_sweep_around_live_in_dry_run(clock):
    from types import SimpleNamespace

    cam = FakeCamera()
    b, _, _, _ = body(clock, camera=cam)
    assert len(b.capture_around()) == 1 and cam.arounds == 0  # 大脑在 dry-run 下只截当前一张
    assert len(b.capture_around(live=True)) == 4 and cam.arounds == 1

    class SweepEnv(FakeEnv):
        def sweep(self, frames, spin):
            return SimpleNamespace(text=lambda: f"看了 {len(frames)} 张")

    b, _, _, _ = body(clock, camera=cam, env=SweepEnv())
    assert b.sweep_around().startswith("dry-run") and cam.spins == 0
    assert b.sweep_around(live=True) == "看了 2 张" and cam.spins == 1


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

        def unlit(self, now):
            return 1 if self.n else 0

    env = YoloEnv()
    b, _, _, events = body(clock, env=env)
    env.n = 2
    b.step()
    b.step()  # 人数没从 0 变过来：不重复报
    (e,) = events.drain()
    assert e.kind == "stranger" and "2 个" in e.text and "其中 1 个还没点火" in e.text
    assert "陌生人：2 个" in b.status()
    env.n, env.near = 0, ["懒洋洋大王"]
    b.step()
    env.near = []
    b.step()
    kinds = [(e.kind, e.text) for e in events.drain()]
    assert kinds[0] == ("arrive", "懒洋洋大王 来到身边") and ("stranger", "陌生人都走开了") in kinds
    assert any(k == "leave" and "5 秒" in t for k, t in kinds)


class FakeChecker:
    def __init__(self, opened=True, closed=True):
        from skydango.config import FriendCheckConfig

        self.cfg = FriendCheckConfig()
        self.calls = []
        self.opened, self.closed = opened, closed

    def check(self, x, y):
        from skydango.game.friendtree import CheckResult

        self.calls.append((x, y))
        img = np.zeros((1080, 1920, 3), np.uint8)
        return CheckResult(img, img, img, 0.3 if self.opened else 0.0, self.closed, "esc" if self.closed else "")

    def looks_open(self, changed):
        return changed >= 0.06


def friend_body(clock, live=True, **kw):
    checker = FakeChecker(**kw)
    b, device, reader, events = body(clock, live=live, frames=[scene(size=(1920, 1080))], friend_checker=checker)
    b.cfg.friend_check.enabled = True
    return b, checker, reader, events


def test_check_friend_needs_enabling_and_a_fresh_look(clock):
    b, checker, _, _ = friend_body(clock)
    b.cfg.friend_check.enabled = False
    with pytest.raises(ToolError, match="没开"):
        b.check_friend(800, 300)
    b.cfg.friend_check.enabled = True
    with pytest.raises(ToolError, match="look"):
        b.check_friend(800, 300)
    b.look()
    clock.advance(20)  # 图太旧了：人可能走了
    with pytest.raises(ToolError, match="look"):
        b.check_friend(800, 300)
    assert checker.calls == []


def test_check_friend_scales_coordinates_and_returns_images(clock):
    b, checker, _, events = friend_body(clock)
    b.look()
    out = b.check_friend(800, 300)
    assert checker.calls == [(1200, 450)]  # 1280×720 的图 → 1920×1080 原图
    assert [block["type"] for block in out] == ["image", "image", "text"]
    assert "已经关上" in out[-1]["text"]
    with pytest.raises(ToolError, match="刚确认过"):
        b.check_friend(800, 300)


def test_check_friend_refuses_panel_area_bottom_bar_and_holding(clock):
    b, checker, reader, _ = friend_body(clock)
    b.look()
    reader.panel_closed_since = None  # 聊天记录面板开着
    with pytest.raises(ToolError, match="聊天记录面板"):
        b.check_friend(100, 300)
    with pytest.raises(ToolError, match="按钮栏"):
        b.check_friend(800, 700)
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="牵着"):
        b.check_friend(800, 300)
    assert checker.calls == []


def test_check_friend_at_uses_frame_pixels(clock):
    b, checker, reader, _ = friend_body(clock, live=False)
    out = b.check_friend_at(1200, 450, live=True)  # 不用先 look：坐标就是原图像素
    assert checker.calls == [(1200, 450)] and "已经关上" in out[-1]["text"]
    b, checker, reader, _ = friend_body(clock, live=False)
    with pytest.raises(ToolError, match="按钮栏"):
        b.check_friend_at(100, 1050, live=True)
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="牵着"):
        b.check_friend_at(1200, 450, live=True)
    b.cfg.friend_check.enabled = False
    with pytest.raises(ToolError, match="没开"):
        b.check_friend_at(1200, 450, live=True)
    assert checker.calls == []


def test_check_friend_dry_run_and_unclosed_panel(clock):
    b, checker, _, _ = friend_body(clock, live=False)
    b.look()
    assert "dry-run" in b.check_friend(800, 300) and checker.calls == []
    b, checker, _, events = friend_body(clock, closed=False)
    b.look()
    out = b.check_friend(800, 300)
    assert "没关上" in out[-1]["text"]
    assert any(e.kind == "error" and "好友树" in e.text for e in events.drain())


# ---- 画面被挡时暂停感知计时（感知层一期 §4） ----
def test_blackout_holds_env_until_screen_is_back(clock):
    env = FakeEnv()
    b, device, _, _ = body(clock, env=env)
    b.step()
    device.frames = [np.zeros((720, 1280, 3), np.uint8)]
    clock.advance(1.5)
    b.step()
    assert env.holds == [("hold", "blackout")]
    device.frames = [scene()]
    clock.advance(1.5)
    b.step()
    assert env.holds == [("hold", "blackout"), ("release", "blackout")]


def test_camera_move_is_wrapped_in_held(clock):
    env = FakeEnv()
    b, _, _, _ = body(clock, live=True, camera=FakeCamera(), env=env)
    b.camera_move("left", 1)
    assert env.holds == [("hold", "camera"), ("release", "camera")]
    env.holds.clear()
    b.camera_reset()
    assert env.holds == [("hold", "camera"), ("release", "camera")]


def test_emote_is_wrapped_in_held(clock):
    env = FakeEnv()
    b, _, _, _ = body(clock, live=True, emotes=FakeEmotes(), env=env)
    b.emote("鞠躬")
    assert env.holds == [("hold", "wheel"), ("release", "wheel")]


def test_social_is_wrapped_in_held(clock):
    env, social = FakeEnv(), FakeSocial()
    b, _, _, _ = body(clock, env=env, social=social)
    env.requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (0, 0), 100.0)}
    b.step()
    assert ("hold", "social") in env.holds and env.holds[-1] == ("release", "social")


# ---- 二期：look_around 走环绕扫描 ----
class SpinCamera(FakeCamera):
    def __init__(self):
        super().__init__()
        self.spins = 0

    def spin(self, capture, turns=1, seconds_per_turn=2.0, fps=15.0):
        from skydango.brain.camera import SpinResult

        self.spins += 1
        f = capture()
        return SpinResult(f, [(0.1, f), (0.2, f)], f, 0.2, True, False)


class SweepEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.swept = []

    def sweep(self, frames, spin):
        from skydango.vision.sweep import SweepEntry, SweepResult

        self.swept.append([t for t, _ in frames])
        return SweepResult([SweepEntry("前", 0, "懒洋洋大王", 3, None)], None, len(frames), 0.2)


def test_sweep_around_spins_and_sweeps(clock):
    env, cam = SweepEnv(), SpinCamera()
    b, _, _, _ = body(clock, live=True, env=env, camera=cam)
    assert b.sweep_around() == "正前方：懒洋洋大王"
    assert cam.spins == 1 and env.swept == [[0.0, 0.1, 0.2]]  # 转前那张算 0 秒
    assert env.holds == [("hold", "camera"), ("release", "camera")]


def test_sweep_around_in_dry_run_does_not_turn(clock):
    env, cam = SweepEnv(), SpinCamera()
    b, _, _, _ = body(clock, env=env, camera=cam)
    text = b.sweep_around()
    assert text.startswith("dry-run") and "正前方：懒洋洋大王" in text
    assert cam.spins == 0 and env.swept == [[0.0]]


def test_sweep_around_refused_in_blackout(clock):
    b, _, _, _ = body(clock, live=True, env=SweepEnv(), camera=SpinCamera())
    b.blackout = True
    with pytest.raises(ToolError, match="黑"):
        b.sweep_around()


class SceneEnv(FakeEnv):
    def strangers(self, now):
        return 0

    def overlay(self, now):
        return [{"x": 500, "y": 300, "w": 100, "h": 200, "kind": "friend", "label": "懒洋洋大王"}]


def test_body_look_uses_scene_note_with_perception(clock):
    b, _, _, _ = body(clock, env=SceneEnv())
    _, note = b.look()
    assert "- 懒洋洋大王：(550, 400) 附近" in note["text"] and "没列出的人都叫" in note["text"]


# ---- 二期：陌生人的消息加注说话人 ----
class HintEnv(FakeEnv):
    def speaker_hint(self, now):
        return "（说话的可能是左边近处那个陌生人）"


def test_stranger_message_gets_speaker_hint_in_event(clock):
    b, _, reader, events = body(clock, env=HintEnv())
    reader.batches = [[msg("你好", "陌生人"), msg("在吗", "")]]
    b.step()
    texts = [e.text for e in events.drain()]
    assert "聊天  陌生人：「你好（说话的可能是左边近处那个陌生人）」" in texts
    assert "聊天  （看不出是谁）：「在吗（说话的可能是左边近处那个陌生人）」" in texts


def test_friend_message_gets_no_hint(clock):
    b, _, reader, events = body(clock, env=HintEnv())
    reader.batches = [[msg("你好", "懒洋洋大王")]]
    b.step()
    assert [e.text for e in events.drain()] == ["聊天  懒洋洋大王：「你好」"]


def test_no_hint_without_perception(clock):
    b, _, reader, events = body(clock, env=FakeEnv())
    reader.batches = [[msg("你好", "陌生人")]]
    b.step()
    assert [e.text for e in events.drain()] == ["聊天  陌生人：「你好」"]


# ---- 二期：有人走过来、最近的人 ----
class ApproachEnv(FakeEnv):
    def __init__(self, who=(), nearest=None):
        super().__init__()
        self.who = list(who)
        self.near_one = nearest

    def strangers(self, now):
        return 0

    def pop_approaches(self):
        out, self.who = self.who, []
        return out

    def nearest(self, now):
        return self.near_one


def test_approach_events(clock):
    b, _, _, events = body(clock, env=ApproachEnv(["懒洋洋大王", "陌生人"]))
    b.step()
    got = [(e.kind, e.text) for e in events.drain() if e.kind == "approach"]
    assert got == [("approach", "懒洋洋大王 朝你走过来了"), ("approach", "有个陌生人朝你走过来了")]


def test_holding_partner_approach_is_dropped(clock):
    b, _, _, events = body(clock, env=ApproachEnv(["卡洛"]))
    b.holding = "卡洛"
    b.step()
    assert not [e for e in events.drain() if e.kind == "approach"]


def test_status_shows_nearest(clock):
    b, _, _, _ = body(clock, env=ApproachEnv(nearest=("懒洋洋大王", "近")))
    assert "离你最近的：懒洋洋大王（近）" in b.status()
    b2, _, _, _ = body(clock, env=ApproachEnv())
    assert "离你最近的" not in b2.status()


def test_camera_move_forgets_self_box(clock):
    env = SweepEnv()
    env.self_box = "框"
    b, _, _, _ = body(clock, live=True, env=env, camera=SpinCamera())
    b.camera_move("zoom_in", 1)
    assert env.self_box is None
    env.self_box = "框"
    b.camera_reset()
    assert env.self_box is None


def test_sweep_around_mentions_blackout(clock):
    class DarkCamera(SpinCamera):
        def spin(self, capture, turns=1, seconds_per_turn=2.0, fps=15.0):
            from dataclasses import replace

            return replace(super().spin(capture, turns, seconds_per_turn, fps), blackout=True)

    b, _, _, _ = body(clock, live=True, env=SweepEnv(), camera=DarkCamera())
    assert "中途画面黑了" in b.sweep_around()


# ---- 三期：别人对团子做的动作 ----
class GestureEnv(ApproachEnv):
    def __init__(self, gestures=()):
        super().__init__()
        self.gestures = list(gestures)

    def pop_gestures(self):
        out, self.gestures = self.gestures, []
        return out


def test_gesture_events(clock):
    b, _, _, events = body(clock, env=GestureEnv([("懒洋洋大王", "wave"), ("番茄炒蛋盖饭", "clap")]))
    b.step()
    got = [(e.kind, e.text) for e in events.drain() if e.kind == "gesture"]
    assert got == [("gesture", "懒洋洋大王对你挥手"), ("gesture", "番茄炒蛋盖饭对你clap")]


def test_holding_partner_gesture_is_dropped(clock):
    b, _, _, events = body(clock, env=GestureEnv([("卡洛", "wave")]))
    b.holding = "卡洛"
    b.step()
    assert not [e for e in events.drain() if e.kind == "gesture"]


def test_status_lists_people_on_screen_with_side_and_distance(clock):
    from skydango.vision.people import Person

    env = FakeEnv()
    b, _, _, _ = body(clock, env=env)
    assert "画面里" not in b.status()
    env.people_list = [Person(1, "friend", "小明", Rect(100, 300, 90, 300), "左边", "近"),
                       Person(2, "stranger", None, Rect(1100, 400, 40, 90), "右边", "远")]
    assert "画面里：小明（左边·近）、陌生人（右边·远）" in b.status()


def test_look_person_uses_perception_box(clock):
    from skydango.vision.people import Person

    env = FakeEnv()
    env.people_list = [Person(1, "friend", "小明", Rect(400, 200, 120, 300), "前面", "近")]
    b, _, _, _ = body(clock, env=env)
    img, note = b.look_person("小明")
    assert img["type"] == "image" and "这是 小明" in note["text"] and "估的" not in note["text"]
    assert b.find_person("小明", clock()) == Rect(400, 200, 120, 300)


def test_look_person_falls_back_to_name_tag(clock):  # 没开感知层：从名字标签往下估一块
    env = FakeEnv()
    env.labels = {"小明": (560, 200, 160, 44, clock())}
    b, _, _, _ = body(clock, env=env)
    box = b.find_person("小明", clock())
    assert box.y == 244 and box.h == 264 and box.x + box.w / 2 == 640  # 标签下方、6 倍标签高、居中
    _, note = b.look_person("小明")
    assert "按名字标签估的位置" in note["text"]


def test_find_person_ignores_stale_name_tag(clock):
    env = FakeEnv()
    env.labels = {"小明": (560, 200, 160, 44, clock() - 60)}
    b, _, _, _ = body(clock, env=env)
    assert b.find_person("小明", clock()) is None


def test_look_person_not_found_suggests_look_around(clock):
    b, _, _, _ = body(clock, env=FakeEnv())
    with pytest.raises(ToolError, match="look_around"):
        b.look_person("小明")


def test_look_person_shares_look_rate_limit(clock):
    env = FakeEnv()
    env.labels = {"小明": (560, 200, 160, 44, clock())}
    b, _, _, _ = body(clock, env=env)
    b.look()
    with pytest.raises(ToolError, match="刚看过"):
        b.look_person("小明")


def test_step_ticks_running_skill_and_status_shows_it(clock):
    from test_brain_skills import FakeSkill

    b, _, _, _ = body(clock)
    assert "没有在做的事" in b.status()
    skill = FakeSkill()
    b.skills.start(b, skill)
    b.step()
    assert skill.ticks == 1
    assert "正在做：盯着小明" in b.status()


def test_shutdown_cancels_skill_before_resetting_camera(clock):
    from test_brain_skills import FakeSkill

    order = []
    camera = FakeCamera()
    camera.reset = lambda: order.append("reset") or "复原了"
    b, _, _, _ = body(clock, camera=camera)
    skill = FakeSkill()
    skill.stop = lambda body, reason: order.append("stop")
    b.skills.start(b, skill)
    b.shutdown()
    assert order == ["stop", "reset"] and b.skills.active is None


def test_stop_task_cancels_running_skill(clock):
    from test_brain_skills import FakeSkill

    b, _, _, _ = body(clock)
    skill = FakeSkill()
    b.skills.start(b, skill)
    assert b.stop_task() == "停下了：盯着小明"
    assert skill.stops == ["大脑叫停"] and b.stop_task() == "没有在做的事"
