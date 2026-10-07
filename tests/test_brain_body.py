from contextlib import contextmanager
import base64
import json
import logging
import threading
from concurrent.futures import Future

import cv2
import numpy as np
import pytest
from conftest import FakeDevice, fake_panel, scene

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


class SyncReader(FakeReader):
    """面板开没开跟着假面板（fake_panel 的状态）走。"""

    def __init__(self, state):
        super().__init__()
        self.state = state

    @property
    def panel_closed_since(self):
        return None if self.state.open else 0.0

    @panel_closed_since.setter
    def panel_closed_since(self, value):
        pass

    def panel_visible(self, frame):
        return self.state.open


def body(clock, live=False, frames=None, panel_mode=None, **kw):
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.reply.dry_run = not live
    cfg.reply.disclosure_prefix = ""
    cfg.sender.open_chat_key = 28
    cfg.proactive.enabled = False  # 旧行为的测试：不管主动开口；pro_body 打开
    cfg.lull.enabled = kw.pop("lull", False)  # 旧行为的测试：不认冷场；test_brain_lull_body 打开
    cfg.addressee.enabled = kw.pop("addressee", False)  # 旧行为的测试：不分跟谁说；test_brain_addressee_body 打开
    cfg.pacing.enabled = kw.pop("pacing", False)  # 旧行为的测试：不攒话；test_brain_pacing_body 打开
    cfg.panel.mode = "always"  # 旧行为的测试：面板常开（2026-09-30 起默认 auto）；按需打开的用 panel_mode="auto"
    device = kw.pop("device_override", None) or FakeDevice(frames or [scene()])
    reader = FakeReader()
    if panel_mode is not None:  # 用假面板：按 46 开关，reader 跟着它
        cfg.panel.mode = panel_mode
        panel, state = fake_panel(device, open_=False, mode=panel_mode)
        panel.clock = clock
        reader = SyncReader(state)
        kw["panel"] = panel
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, "")
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    events = EventQueue(clock=clock)
    kw.setdefault("wall", lambda: 1_790_000_000.0)
    b = Body(cfg, device, reader, sender, self_filter, events, clock=clock, sleep=lambda s: None, **kw)
    b.panel.start(clock())
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


def test_friend_back_soon_is_return_not_arrive(clock):
    """走出画面又走回来（rejoin 秒内）不再当成来到身边：背景事件，不叫醒大脑打招呼。"""
    env = FakeEnv()
    b, _, _, events = body(clock, env=env)
    b.cfg.brain.rejoin = 60.0
    env.near = ["懒洋洋大王"]
    b.step()
    (e,) = events.drain()
    assert (e.kind, e.who) == ("arrive", "懒洋洋大王")
    env.near = []
    b.step()
    (e,) = events.drain()
    assert (e.kind, e.who) == ("leave", "懒洋洋大王")
    clock.advance(59)
    env.near = ["懒洋洋大王"]
    b.step()
    (e,) = events.drain()
    assert (e.kind, e.who, e.text) == ("return", "懒洋洋大王", "懒洋洋大王 回来了")


def test_friend_back_after_rejoin_is_arrive_again(clock):
    env = FakeEnv()
    b, _, _, events = body(clock, env=env)
    b.cfg.brain.rejoin = 60.0
    env.near = ["懒洋洋大王"]
    b.step()
    env.near = []
    b.step()
    events.drain()
    clock.advance(61)
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.kind for e in events.drain()] == ["arrive"]


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


def test_post_runs_later_without_waiting(clock, caplog):
    """post：不等结果，交给身体线程下一圈做（大脑交“心里”文字用，身体忙着自动喊时不能卡住 / 报 ERROR）。"""
    b, _, _, _ = body(clock)
    ran = []
    t = threading.Thread(target=lambda: ran.append(b.post(lambda: ran.append(threading.get_ident()))))
    t.start()
    t.join(2)
    assert ran == [True]  # 没人跑 step 也马上返回
    b.step()
    assert ran == [True, threading.get_ident()]  # 在身体线程里做

    def boom():
        raise RuntimeError("坏了")

    t = threading.Thread(target=lambda: b.post(boom))
    t.start()
    t.join(2)
    with caplog.at_level(logging.ERROR):
        b.step()  # 出错只记日志，不影响身体这一圈
    assert "坏了" in caplog.text

    b.stopped = True
    assert b.post(lambda: ran.append("停了还做")) is False
    b.step()
    assert "停了还做" not in ran


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


def test_say_refuses_what_it_just_said(clock):
    """10-03 晚 DeepSeek 备用大脑：同一句话隔 8 秒说两遍、连说三句晚安。10 分钟内几乎一样的话身体拦下（Claude 大脑也一样）。"""
    b, device, _, _ = body(clock, live=True, wall=clock)  # 拦重复按墙钟算（spoken 记的是墙钟）
    assert b.say("卡洛回来啦，头发吹干没").startswith("已发送")
    clock.advance(8)
    with pytest.raises(ToolError, match="刚说过"):
        b.say("卡洛回来啦头发吹干没")
    assert b.said == ["卡洛回来啦，头发吹干没"]  # 被拦的不记
    clock.advance(5)
    assert b.say("你头发吹干了吗").startswith("已发送")  # 换个说法照发
    clock.advance(600)
    assert b.say("卡洛回来啦，头发吹干没").startswith("已发送")  # 过了 10 分钟可以再说


def test_manual_say_is_not_checked_for_repeats(clock):
    b, device, _, _ = body(clock, live=True, wall=clock)
    b.say("晚安")
    clock.advance(5)
    assert b.say("晚安", live=True).startswith("已发送")  # 主人在手动控制里让她说的照发


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


def test_followup_say_is_not_recorded_as_proactive(clock, tmp_path):
    """10-03 晚：一轮里接着说的第二句（隔 6~16 秒、中间没人说话）被记成「（没人说话，你主动开口）」，
    读回「上次聊到哪」像团子一直在主动找话、换个说法复读自己。隔得久的才算主动开口。"""
    from skydango.brain.body import FOLLOWUP_LABEL
    from skydango.chat.memory import MemoryStore

    store = MemoryStore(tmp_path)
    b, _, reader, _ = body(clock, live=True, store=store, wall=clock)
    reader.batches = [[msg("你去把他叫起来")]]
    b.step()
    b.say("不要啦我才不叫")
    clock.advance(8)
    b.say("让他睡饱嘛")
    clock.advance(120)
    b.say("这边好安静")
    assert [t.user for t in store.history.all()][1:] == [FOLLOWUP_LABEL, "（没人说话，你主动开口）"]


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
        self.cooling = False  # 动作限速中
        self.last_any = float("-inf")  # 同 EmotePlayer：任何动作最近一次（反射的 min_gap 用）

    def on_wheel(self):
        return ["鞠躬"]

    def available(self, ignore_interval=False):
        return [] if self.cooling and not ignore_interval else ["鞠躬"]

    def perform(self, name):
        self.done.append(name)

    def pretend(self, name):
        pass


def test_emote_not_blocked_while_holding(clock):
    emotes = FakeEmotes()
    b, _, _, _ = body(clock, live=True, emotes=emotes)
    with pytest.raises(ToolError, match="能做的：鞠躬"):
        b.emote("跳舞")
    b.holding = "懒洋洋大王"
    assert b.emote("鞠躬") == "做了「鞠躬」" and emotes.done == ["鞠躬"]  # 做动作不会松开牵手（用户实测）


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
        self.reset_args = []  # 每次 reset 的 refine_seconds
        self.forgets = 0

    def move(self, action, steps, max_steps=4):
        self.moves.append((action, steps))
        return "左转了 1 步"

    def reset(self, refine_seconds=None):
        self.resets += 1
        self.reset_args.append(refine_seconds)
        return "镜头转回原位了"

    def forget_reference(self):
        self.forgets += 1

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



class FakeLocomotion:
    def __init__(self):
        self.moves = []

    def move(self, direction, steps=1, max_steps=3):
        self.moves.append((direction, steps, max_steps))
        return f"前进走了 {min(steps, max_steps)} 步"


def test_move_walks_and_refuses_while_holding_unless_forced(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, live=True, locomotion=loco)
    assert b.move("forward", 2) == "前进走了 2 步" and loco.moves == [("forward", 2, 3)]
    clock.advance(10)
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="force=true"):
        b.move("forward")
    assert b.move("forward", force=True) == "前进走了 1 步"


def test_move_min_interval(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, live=True, locomotion=loco)
    b.cfg.brain.move_min_interval = 3.0
    b.move("forward")
    clock.advance(1)
    with pytest.raises(ToolError, match="刚走过"):
        b.move("back")
    clock.advance(2)
    b.move("back")
    assert [m[0] for m in loco.moves] == ["forward", "back"]


def test_move_refused_without_locomotion_in_blackout_and_bad_direction(clock):
    b, _, _, _ = body(clock, live=True)
    with pytest.raises(ToolError, match="没有移动"):
        b.move("forward")
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, live=True, locomotion=loco)
    with pytest.raises(ToolError, match="不认识的移动方向"):
        b.move("up")
    b.blackout = True
    with pytest.raises(ToolError, match="黑"):
        b.move("forward")
    assert loco.moves == []


def test_move_dry_run_does_not_press_but_live_does(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, locomotion=loco)  # dry-run
    assert b.move("left", 2).startswith("dry-run：没真的走（left ×2）")
    with pytest.raises(ToolError, match="不认识的移动方向"):
        b.move("up")
    assert loco.moves == []
    clock.advance(10)
    assert b.move("left", live=True) == "前进走了 1 步" and loco.moves == [("left", 1, 3)]


def test_owner_window_relaxes_move(clock):
    loco = FakeLocomotion()
    b, _, reader, _ = body(clock, live=True, locomotion=loco)
    b.cfg.brain.owner_name = "卡洛"
    b.cfg.brain.owner_window = 30.0
    b.holding = "懒洋洋大王"
    reader.batches = [[msg("#过来", speaker="卡洛")]]
    b.step()
    assert b.move("forward", 6).endswith("（主人命令模式）")  # 牵着手不用 force
    assert b.move("forward", 1) == "前进走了 1 步（主人命令模式）"  # 不用等间隔
    assert loco.moves == [("forward", 6, 6), ("forward", 1, 6)]
    clock.advance(31)
    with pytest.raises(ToolError, match="force=true"):
        b.move("forward")
    b.holding = None
    assert b.move("forward", 6) == "前进走了 3 步" and loco.moves[-1] == ("forward", 6, 3)


def test_owner_note_only_when_relaxation_used(clock):
    loco = FakeLocomotion()
    b, _, reader, _ = body(clock, live=True, locomotion=loco)
    b.cfg.brain.owner_name = "卡洛"
    reader.batches = [[msg("#过来", speaker="卡洛")]]
    b.step()
    assert b.move("forward", 2) == "前进走了 2 步"  # 没超过平时的限制：不标


def test_move_forgets_scene_reference(clock):
    b, _, _, _ = body(clock, live=True, locomotion=FakeLocomotion())
    b._ref_thumb = "旧"
    b.move("forward")
    assert b._ref_thumb is None  # 自己走的，不算画面大变


def owner_says(b, reader, text="#过来"):
    b.cfg.brain.owner_name = "卡洛"
    b.cfg.brain.owner_window = 30.0
    reader.batches = [[msg(text, speaker="卡洛")]]
    b.step()


def test_owner_window_relaxes_emote(clock):
    emotes = FakeEmotes()
    b, _, reader, _ = body(clock, live=True, emotes=emotes)
    emotes.cooling = True
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="现在做不了"):
        b.emote("鞠躬")
    owner_says(b, reader)
    assert b.emote("鞠躬") == "做了「鞠躬」（主人命令模式）"  # 限速中也不拦
    assert "能做的动作：鞠躬" in b.status()
    emotes.cooling = False
    b.holding = None
    assert b.emote("鞠躬") == "做了「鞠躬」"  # 没用到放宽：不标
    with pytest.raises(ToolError, match="能做的：鞠躬"):
        b.emote("跳舞")  # 不在轮盘 / 白名单上的照样做不了
    clock.advance(31)
    emotes.cooling = True
    with pytest.raises(ToolError, match="现在做不了"):
        b.emote("鞠躬")
    assert emotes.done == ["鞠躬", "鞠躬"]


class StepCamera(FakeCamera):
    def move(self, action, steps, max_steps=4):
        self.moves.append((action, steps, max_steps))
        return "原位"


def test_owner_window_relaxes_camera_steps(clock):
    cam = StepCamera()
    b, _, reader, _ = body(clock, live=True, camera=cam)
    assert b.camera_move("left", 8) == "镜头现在：原位" and cam.moves[-1] == ("left", 8, 4)
    owner_says(b, reader)
    assert b.camera_move("left", 8) == "镜头现在：原位（主人命令模式）" and cam.moves[-1] == ("left", 8, 8)
    assert b.camera_move("left", 2) == "镜头现在：原位"
    clock.advance(31)
    b.camera_move("left", 8)
    assert cam.moves[-1] == ("left", 8, 4)

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
    assert b.emote("鞠躬", live=True) == "做了「鞠躬」"  # 牵着手也照做
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
    camera.reset = lambda **kw: order.append("reset") or "复原了"
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


def test_look_person_matches_ocr_typos_in_chat_names(clock):  # 评审：聊天里的名字是 OCR 读的，可能差一个字
    from skydango.vision.people import Person

    env = FakeEnv()
    env.people_list = [Person(1, "friend", "懒洋洋大王", Rect(400, 200, 120, 300), "前面", "近")]
    b, _, _, _ = body(clock, env=env)
    assert b.find_person("懒洋洋大玉", clock()) == Rect(400, 200, 120, 300)
    env.labels = {"小明同学": (560, 200, 160, 44, clock())}
    env.people_list = []
    assert b.find_person("小明同学", clock()) is not None


def test_look_person_not_found_lists_who_is_recognized(clock):
    from skydango.vision.people import Person

    env = FakeEnv()
    env.people_list = [Person(1, "friend", "阿白", Rect(400, 200, 120, 300), "前面", "近")]
    b, _, _, _ = body(clock, env=env)
    with pytest.raises(ToolError, match="认得出：阿白"):
        b.look_person("小明")


def test_look_person_without_env_suggests_looking_yourself(clock):
    b, _, _, _ = body(clock)
    with pytest.raises(ToolError, match="look\\(image=true\\)") as err:
        b.look_person("小明")
    assert "look_around" not in str(err.value)


# ---- 聊天面板按需打开（[panel] mode = "auto"） ----
def auto_body(clock, live=False, **kw):
    return body(clock, live=live, panel_mode="auto", **kw)


def test_arrive_triggers_peek_in_auto(clock):
    env = FakeEnv()
    b, device, _, _ = auto_body(clock, env=env)
    b.step()
    env.near = ["小明"]
    b.step()  # 发 arrive → 记下要看一眼
    b.step()
    assert b.panel.state == "peek" and device.calls.count(("hw_key", 46)) == 1


def test_approach_triggers_peek(clock):
    env = FakeEnv()
    approaches = [["小明"]]
    env.pop_approaches = lambda: approaches.pop() if approaches else []
    b, device, _, _ = auto_body(clock, env=env)
    b.step()  # 发 approach → 记下要看一眼
    b.step()
    assert b.panel.state == "peek"


def test_friend_bubble_calls_bubble_seen(clock):
    env = FakeEnv()
    env.typing_seen = lambda now, within=1.0, strangers=False: True
    b, _, _, _ = auto_body(clock, env=env)
    b.step()
    b.step()
    assert b.panel.state == "bubble"


def test_typing_seen_ignored_when_reader_reads_bubbles(clock):
    env = FakeEnv()
    env.typing_seen = lambda now, within=1.0, strangers=False: True
    b, _, reader, _ = auto_body(clock, env=env)
    reader.reads_bubbles = True  # 无障碍读法：气泡直接读成消息，不拿 YOLO 的气泡叫面板
    b.step()
    b.step()
    assert b.panel.state == "idle"


def test_chat_log_peeks_while_talking(clock):
    b, device, _, _ = auto_body(clock)
    b.panel.reader.reads_bubbles = True  # 无障碍读法：跟画面里的好友聊着、面板关着
    b.panel.reader.tags_in_view = lambda: ["小明"]
    b.panel.state = "talking"
    b.panel._last_activity = clock()
    b.chat_log()  # 大脑要看聊天：面板关着也马上开一眼
    assert ("hw_key", 46) in device.calls


def test_status_shows_typing_and_reader(clock):
    b, _, reader, _ = body(clock)
    assert "在打字" not in b.status() and "读聊天" not in b.status()
    reader.typing = lambda: ["小明", "小红"]
    reader.describe = lambda: "无障碍"
    text = b.status()
    assert "在打字：小明、小红" in text and "读聊天：无障碍" in text
    assert text.index("聊天记录面板") < text.index("在打字") < text.index("读聊天") < text.index("输入框")


def chatting_body(clock, **kw):
    b, device, reader, events = auto_body(clock, **kw)
    reader.state.open = True
    reader.batches = [[msg("在吗")]]
    b.step()
    assert b.panel.state == "chatting"
    return b, device, reader, events


def test_pending_chat_event_keeps_panel_busy(clock):
    b, _, _, events = chatting_body(clock)
    clock.advance(50)
    b.step()
    assert b.panel.state == "chatting"  # 大脑还没取走这条聊天：在准备回复，不算安静
    events.drain()
    clock.advance(46)
    b.step()
    assert b.panel.state == "idle"


def test_brain_in_turn_keeps_panel_busy(clock):
    b, _, _, events = chatting_body(clock)
    events.drain()
    b.brain_busy = lambda: True
    clock.advance(50)
    b.step()
    assert b.panel.state == "chatting"


def test_say_opens_panel_before_enter_when_idle(clock):
    b, device, _, _ = auto_body(clock, live=True)
    b.say("你好")
    keys = [c for c in device.calls if c[0] == "hw_key"]
    assert keys[:2] == [("hw_key", 46), ("hw_key", 28)] and b.panel.state == "chatting"


def test_chat_log_peeks_when_idle(clock):
    b, _, reader, events = auto_body(clock)
    reader.batches = [[], [msg("刚才谁在叫我")]]
    text = b.chat_log()
    assert "刚才谁在叫我" in text and b.panel.state == "chatting"
    assert any(e.kind == "chat" for e in events.drain())


def test_panel_event_only_when_should_be_open(clock):
    b, device, reader, events = auto_body(clock)
    device.hw_key = lambda code: device.calls.append(("hw_key", code))  # 按了也开不了
    clock.advance(20)
    b.step()
    clock.advance(40)
    b.step()
    assert not [e for e in events.drain() if e.kind == "panel"]  # 闲着关着是正常的
    b2, device2, reader2, events2 = chatting_body(clock)
    device2.hw_key = lambda code: device2.calls.append(("hw_key", code))
    reader2.state.open = False
    b2.brain_busy = lambda: True
    b2.step()
    clock.advance(31)
    b2.step()
    assert any(e.kind == "panel" and e.text.startswith("聊天记录面板关了") for e in events2.drain())


def test_scene_restored_triggers_peek(clock):
    black = np.zeros((720, 1280, 3), np.uint8)
    b, device, _, _ = auto_body(clock, frames=[black, scene(), scene()])
    b.step()
    assert b.blackout
    b.step()
    b.step()
    assert b.panel.state == "peek"


def test_shutdown_reopens_panel_in_auto(clock):
    b, _, reader, _ = auto_body(clock)
    b.shutdown()
    assert reader.state.open is True


def test_viewer_info_has_panel_state(clock):
    class Viewer:
        info = None

        def update(self, frame, now, **kw):
            Viewer.info = kw["info"]

    b, _, _, _ = auto_body(clock, viewer=Viewer())
    b.step()
    assert Viewer.info["聊天面板"].startswith("闲着")


def test_viewer_info_has_task_and_recent_says(clock):
    # 管理面板真机团子页的状态卡片：正在做、刚说过（spec console §3）
    class Viewer:
        info = None

        def update(self, frame, now, **kw):
            Viewer.info = kw["info"]

    b, _, _, _ = auto_body(clock, viewer=Viewer())
    b.step()
    assert Viewer.info["正在做"] == "没有在做的事" and Viewer.info["刚说过"] == "还没说话"
    b.said[:] = ["一", "二", "三", "四"]
    b.step()
    assert Viewer.info["刚说过"] == ["四", "三", "二"]


# ---- 看场合主动开口（spec 2026-09-29-proactive-chat §2 / §4） ----
def pro_body(clock, near=("阿花",), **kw):
    """好友在身边、墙钟跟着假时钟走（旧的发言限速构造时就定了 3 秒，测试里两句之间推进够时间）。"""
    env = kw.pop("env", FakeEnv())
    env.near = list(near)
    b, device, reader, events = body(clock, env=env, wall=clock, **kw)
    b.cfg.proactive.enabled = True
    b.friend_names = lambda: ["阿花"]
    return b, env, reader


def test_proactive_say_blocked_when_alone(clock):
    b, _, _ = pro_body(clock, near=())
    with pytest.raises(ToolError, match="身边没有好友，不主动开口"):
        b.say("好无聊")
    assert b.said == []


def test_reply_turn_not_limited(clock):
    b, _, _ = pro_body(clock, near=())
    b.brain_busy = lambda: True
    b.say("在呢")
    assert b.spoken[-1].proactive is False


def test_min_gap_and_recovery(clock):
    b, _, _ = pro_body(clock)
    b.say("这图好黑")
    clock.advance(10)
    with pytest.raises(ToolError, match="刚主动说过"):
        b.say("真的好黑")
    clock.advance(55)
    b.say("真的好黑")
    assert [s.proactive for s in b.spoken] == [True, True]


def test_greeting_new_friend_skips_min_gap(clock):  # 沙盒：卡洛刚来时主动说了一句，饶总紧跟着上线，招呼被 60 秒间隔拦下
    b, env, _ = pro_body(clock)
    b.friend_names = lambda: ["阿花", "小明"]
    b.cfg.proactive.quota_quiet = 10  # 只测间隔
    b.step()  # 阿花来了
    b.say("阿花来啦")
    clock.advance(10)
    env.near = ["阿花", "小明"]
    b.step()  # 小明来了：招呼不受间隔限制
    assert b.occasion().blocked == ""
    b.say("小明也来啦")
    clock.advance(10)
    with pytest.raises(ToolError, match="刚主动说过"):  # 招呼过了：间隔照旧
        b.say("你们好呀")
    assert [s.proactive for s in b.spoken] == [True, True]


def test_greeting_window_expires(clock):
    b, env, _ = pro_body(clock)
    b.friend_names = lambda: ["阿花", "小明"]
    env.near = ["小明"]
    b.step()  # 小明来了，没打招呼
    clock.advance(b.cfg.proactive.greet_window + 1)
    env.near = ["小明", "阿花"]
    b.say("小明在吗")  # 主动说一句
    clock.advance(5)
    with pytest.raises(ToolError, match="刚主动说过"):  # 小明来了太久、阿花还没被身体认出来：不算刚来
        b.say("还有谁")


def cold(b, clock):
    b.cfg.proactive.min_gap = 0
    b.cfg.proactive.quota_quiet = 10
    for text in ("一", "二", "三"):
        b.say(text)
        clock.advance(100)


def test_cold_pause_then_friend_speaks(clock):
    b, _, reader = pro_body(clock)
    cold(b, clock)
    with pytest.raises(ToolError, match="连着 3 句"):
        b.say("四")
    reader.batches = [[msg("嗯？", speaker="阿花")]]
    b.step()
    b.say("四")


def test_manual_say_ignores_guard(clock):
    b, _, _ = pro_body(clock)
    cold(b, clock)
    b.say("主人让说的", live=True)
    assert len(b.spoken) == 3
    clock.advance(5)  # 旧的发言限速（构造时定的 3 秒）
    with pytest.raises(ToolError, match="连着 3 句"):
        b.say("四")


def test_dry_run_counts(clock):
    b, _, _ = pro_body(clock)
    b.cfg.proactive.min_gap = 0
    b.say("一")
    clock.advance(5)
    b.say("二")
    clock.advance(5)
    with pytest.raises(ToolError, match="已经主动说了 2 句"):
        b.say("三")


def test_status_has_occasion(clock):
    b, _, _ = pro_body(clock)
    assert "场合：安静（身边 阿花" in b.status()
    alone, _, _, _ = body(clock, wall=clock)
    alone.cfg.proactive.enabled = True
    assert "场合：没熟人（身边没人）" in alone.status()


def test_disabled_keeps_old_behavior(clock):
    b, _, _ = pro_body(clock, near=())
    b.cfg.proactive.enabled = False
    b.say("好无聊")
    assert "场合" not in b.status()


def test_viewer_info_has_occasion(clock):
    class Viewer:
        info = None

        def update(self, frame, now, **kw):
            Viewer.info = kw["info"]

    b, _, _ = pro_body(clock, viewer=Viewer())
    b.step()
    assert Viewer.info["场合"] == "安静 · 还能主动说 2 句"


def notices(events):
    return [e.text for e in events.drain() if e.kind == "notice"]


def test_news_becomes_notice(clock):
    b, _, _ = pro_body(clock)
    b.news("天黑了")
    b.step()
    assert notices(b.events) == ["眼睛注意到：天黑了"]


def test_notice_min_and_duplicate(clock):
    b, _, _ = pro_body(clock)
    b.news("天黑了")
    b.news("下雨了")
    b.step()
    assert notices(b.events) == ["眼睛注意到：天黑了"]  # 60 秒内的第二条不发
    clock.advance(61)
    b.news("天黑了")
    b.step()
    assert notices(b.events) == []  # 和上一条一样
    b.news("下雨了")
    b.step()
    assert notices(b.events) == ["眼睛注意到：下雨了"]


def test_notice_dropped_when_alone_or_no_quota(clock):
    b, env, _ = pro_body(clock, near=())
    b.news("天黑了")
    b.step()
    assert notices(b.events) == []
    env.near = ["阿花"]
    b.cfg.proactive.min_gap = 0
    b.say("一")
    clock.advance(5)
    b.say("二")  # 安静时额度 2 句用完
    b.news("下雨了")
    b.step()
    assert notices(b.events) == []


def test_place_change(clock):
    b, env, _ = pro_body(clock)
    got = []
    for place in ("", "云野", "", "雨林"):
        env.place = place
        b.step()
        got += notices(b.events)
        clock.advance(61)
    assert got == ["看起来到了雨林"]


def test_place_change_without_place_attribute(clock):
    b, _, _ = pro_body(clock)
    b.step()  # FakeEnv 没有 place：不报错
    none, _, _, events = body(clock, wall=clock)
    none.cfg.proactive.enabled = True
    none.news("天黑了")
    none.step()  # 没开 env：当成没熟人
    assert notices(events) == []


def test_notice_off_when_disabled(clock):
    b, _, _ = pro_body(clock)
    b.cfg.proactive.enabled = False
    b.news("天黑了")
    b.step()
    assert notices(b.events) == []


def test_fallback_reply_is_not_proactive(clock):
    # 评审 #1：大脑离线时的备用回复是接话，不过主动开口的护栏、不占主动额度
    from skydango.chat.responder import Reply

    class Responder:
        def reply(self, batch):
            return Reply("在呢")

    b, env, reader = pro_body(clock, near=(), live=True, fallback=Responder())
    b.brain_offline = lambda now: True
    reader.batches = [[msg("有人吗", speaker="路人")]]
    b.step()
    clock.advance(2)
    b.step()
    assert "在呢" in b.said[-1]
    assert [s.proactive for s in b.spoken] == [False]


def test_occasion_errors_do_not_break_the_loop(clock):
    # 算场合出错（env / friends.md）：网页和 status 显示算不出来，这一圈照常跑完命令
    class Viewer:
        info = None

        def update(self, frame, now, **kw):
            Viewer.info = kw["info"]

    b, _, _ = pro_body(clock, viewer=Viewer())

    def broken():
        raise OSError("friends.md 读不了")

    b.friend_names = broken
    ran = []
    b._commands.put((lambda: ran.append(1), __import__("concurrent.futures").futures.Future()))
    b.step()
    assert Viewer.info["场合"] == "算不出来" and ran == [1]
    assert "场合：算不出来（详见日志）" in b.status()


# ---- 物品（spec 2026-09-29-object-recognition §3） ----
def test_status_lists_things(clock):
    from skydango.vision.people import Thing

    class ThingEnv(FakeEnv):
        things = []

        def objects(self, now):
            return list(self.things)

    env = ThingEnv()
    b, _, _, _ = body(clock, env=env)
    assert "画面里的东西" not in b.status()
    env.things = [Thing(1, "bench", Rect(0, 0, 1, 1), "左边", "近"), Thing(2, "spirit", Rect(0, 0, 1, 1), "前面", "远")]
    assert "画面里的东西：座位（左边·近）、先祖（前面·远）" in b.status()
    plain, _, _, _ = body(clock, env=FakeEnv())
    assert "画面里的东西" not in plain.status()


# ---- 技能 track：找目标位置、跟踪期间不报人来人走（plan 2026-09-29-brain-track Task 3） ----
def test_target_x_prefers_body_box(clock):
    from skydango.vision.people import Person

    env = FakeEnv()
    env.people_list = [Person(1, "friend", "懒洋洋大王", Rect(900, 400, 100, 300), "前面", "近")]
    env.labels = {"懒洋洋大王": (1200, 300, 160, 50, clock())}
    b, _, _, _ = body(clock, env=env)
    assert b.target_x("懒洋洋大王", clock()) == (950.0, "body")


def test_target_x_falls_back_to_fresh_tag(clock):
    env = FakeEnv()
    env.labels = {"懒洋洋大王": (1200, 300, 160, 50, clock() - 0.2)}
    b, _, _, _ = body(clock, env=env)
    assert b.target_x("懒洋洋大王", clock()) == (1280.0, "tag")
    env.labels = {"懒洋洋大王": (1200, 300, 160, 50, clock() - 0.8)}  # 超过 max_age = 0.5 s：不算
    assert b.target_x("懒洋洋大王", clock()) is None


def test_target_x_tolerates_ocr_typo(clock):
    env = FakeEnv()
    env.labels = {"懒洋洋大玉": (1200, 300, 160, 50, clock())}
    b, _, _, _ = body(clock, env=env)
    assert b.target_x("懒洋洋大王", clock()) == (1280.0, "tag")


def test_target_x_ignores_strangers_and_missing_env(clock):
    from skydango.vision.people import Person

    env = FakeEnv()
    env.people_list = [Person(1, "stranger", None, Rect(900, 400, 100, 300), "前面", "近")]
    b, _, _, _ = body(clock, env=env)
    assert b.target_x("懒洋洋大王", clock()) is None
    plain, _, _, _ = body(clock)
    assert plain.target_x("懒洋洋大王", clock()) is None


def test_frame_width(clock):
    b, _, _, _ = body(clock, frames=[np.zeros((720, 1280, 3), np.uint8)])
    assert b.frame_width == 1920  # 还没截过图
    b.step()
    assert b.frame_width == 1280


def quiet_skill():
    from test_brain_skills import FakeSkill

    skill = FakeSkill()
    skill.quiet_people = True  # 跟踪中：转镜头时人进出画面是自己转的
    return skill


def test_no_people_events_while_quiet_skill_runs(clock):
    env = FakeEnv()
    b, _, _, events = body(clock, env=env)
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.kind for e in events.drain()] == ["arrive"]
    b.skills.start(b, quiet_skill())
    env.near = []
    b.step()
    assert "leave" not in [e.kind for e in events.drain()]
    b.stop_task()
    b.step()
    assert "leave" in [e.kind for e in events.drain()]


def test_approaches_during_quiet_skill_are_dropped(clock):
    env = ApproachEnv(["懒洋洋大王"])  # 转镜头时框变大，被当成"走过来"
    b, _, _, events = body(clock, env=env)
    b.skills.start(b, quiet_skill())
    b.step()
    b.stop_task()
    b.step()  # 跟踪结束后也不冒出过时的 approach
    assert not [e for e in events.drain() if e.kind == "approach"]


# ---- track 工具（Task 5） ----
def track_body(clock, live=True, visible=True, **kw):
    from skydango.vision.people import Person

    env = FakeEnv()
    if visible:
        env.people_list = [Person(1, "friend", "懒洋洋大王", Rect(1300, 400, 100, 300), "右边", "中")]
    kw.setdefault("camera", FakeCamera())
    b, device, reader, events = body(clock, live=live, env=env, **kw)
    return b, env, events


def test_track_starts_skill_when_target_visible(clock):
    b, _, _ = track_body(clock)
    out = b.track("懒洋洋大王", 20)
    assert out.startswith("开始盯着懒洋洋大王了")
    assert b.skills.active.name == "track" and b.skills.active.timeout == 25
    assert "正在做：盯着懒洋洋大王" in b.status()


def test_track_refused_in_dry_run(clock):
    b, _, _ = track_body(clock, live=False)
    with pytest.raises(ToolError, match="dry-run"):
        b.track("懒洋洋大王")
    assert b.skills.active is None
    assert b.track("懒洋洋大王", live=True).startswith("开始盯着")  # 手动控制照做


def test_track_refused_when_target_not_visible(clock):
    from skydango.vision.people import Person

    b, env, _ = track_body(clock, visible=False)
    env.people_list = [Person(1, "friend", "阿白", Rect(400, 200, 120, 300), "左边", "近")]
    with pytest.raises(ToolError, match="没看到") as info:
        b.track("懒洋洋大王")
    assert "阿白" in str(info.value) and "look_around" in str(info.value)
    assert b.skills.active is None


def test_track_refused_without_camera_or_perception(clock):
    b, _, _ = track_body(clock, camera=None)
    with pytest.raises(ToolError, match="视角"):
        b.track("懒洋洋大王")

    class OcrEnv:  # 整图 OCR 的 env：只有名字标签，没有 people()
        labels = {"懒洋洋大王": (1200, 300, 160, 50, clock())}
        requests = {}

    plain, _, _, _ = body(clock, live=True, env=OcrEnv(), camera=FakeCamera())
    with pytest.raises(ToolError, match="感知层"):
        plain.track("懒洋洋大王")
    with pytest.raises(ToolError, match="感知层"):
        body(clock, live=True, camera=FakeCamera())[0].track("懒洋洋大王")


def test_track_clamps_seconds(clock):
    b, _, _ = track_body(clock)
    b.track("懒洋洋大王", 999)
    assert b.skills.active.timeout == 60 + 5
    b.stop_task()
    b.track("懒洋洋大王", 0)
    assert b.skills.active.timeout == 1 + 5


def test_track_refused_while_another_task_runs(clock):
    b, _, _ = track_body(clock)
    b.track("懒洋洋大王")
    with pytest.raises(ToolError, match="stop_task"):
        b.track("懒洋洋大王")


def test_no_scene_change_while_skill_turns_camera(clock):
    b, device, _, events = body(clock)
    b.step()
    b.skills.start(b, quiet_skill())  # FakeSkill 没写 needs_camera；quiet_skill 补上
    b.skills.active.needs_camera = True
    device.frames = [np.full((720, 1280, 3), 230, np.uint8)]  # 自己转镜头：画面大变
    clock.advance(1.5)
    b.step()
    assert "scene_change" not in [e.kind for e in events.drain()]
    b.stop_task()
    device.frames = [scene()]
    clock.advance(1.5)
    b.step()  # 技能结束后重新拿参照，第一圈不比
    clock.advance(1.5)
    b.step()
    assert "scene_change" not in [e.kind for e in events.drain()]


# ---- 评审修正（I3 / I4 / I5 / M3） ----
def camera_skill():
    skill = quiet_skill()
    skill.needs_camera = True
    return skill


def test_requests_still_accepted_while_quiet_skill_runs(clock):
    env, social = FakeEnv(), FakeSocial()
    b, _, _, events = body(clock, live=True, env=env, social=social)
    env.near = ["懒洋洋大王"]
    b.step()
    events.drain()
    b.skills.start(b, quiet_skill())
    env.requests = {"懒洋洋大王": Request("懒洋洋大王", "hand", (0, 0), 100.0)}
    social.to_handle = ["懒洋洋大王:hand"]
    b.step()
    kinds = [e.kind for e in events.drain()]
    assert "request" in kinds and "accepted" in kinds  # 盯人时好友伸手照样接


def test_shutdown_restores_wheel_before_resetting_camera_with_time_limit(clock):
    order = []
    camera, emotes = FakeCamera(), FakeEmotes()
    camera.reset = lambda **kw: order.append(("reset", kw)) or "复原了"
    emotes.restore = lambda: order.append(("restore", {}))
    b, _, _, _ = body(clock, camera=camera, emotes=emotes)
    b.shutdown()
    assert [o[0] for o in order] == ["restore", "reset"]  # 轮盘先恢复：复位可能要好几秒
    assert 0 < order[1][1]["refine_seconds"] <= 8


@pytest.mark.parametrize("call", [
    lambda b: b.camera_reset(live=True),
    lambda b: b.camera_move("left", 1, live=True),
    lambda b: b.capture_around(live=True),
])
def test_camera_tools_stop_camera_skill_first(clock, call):
    cam = FakeCamera()
    b, _, _, _ = body(clock, live=True, camera=cam)
    skill = camera_skill()
    b.skills.start(b, skill)
    out = call(b)
    assert b.skills.active is None and skill.stops  # 技能先停下
    assert cam.resets + len(cam.moves) + cam.arounds == 1  # 然后照做
    if isinstance(out, str):
        assert "先停下了盯着小明" in out


def test_check_friend_stops_camera_skill_first(clock):
    b, checker, _, _ = friend_body(clock, live=False)
    skill = camera_skill()
    b.skills.start(b, skill)
    out = b.check_friend_at(1200, 450, live=True)
    assert b.skills.active is None and checker.calls == [(1200, 450)]
    assert "先停下了盯着小明" in out[-1]["text"]


def test_sweep_around_stops_camera_skill_first(clock):
    b, _, _, _ = body(clock, live=True, env=SweepEnv(), camera=SpinCamera())
    skill = camera_skill()
    b.skills.start(b, skill)
    out = b.sweep_around(live=True)
    assert b.skills.active is None and "先停下了盯着小明" in out


def test_camera_tools_leave_other_skills_alone(clock):
    cam = FakeCamera()
    b, _, _, _ = body(clock, live=True, camera=cam)
    skill = quiet_skill()  # 不转镜头的技能
    b.skills.start(b, skill)
    out = b.camera_reset(live=True)
    assert b.skills.active is skill and "先停下" not in out


def test_move_and_blackout_forget_camera_reference(clock):
    cam = FakeCamera()
    b, device, _, _ = body(clock, live=True, camera=cam, locomotion=FakeLocomotion())
    b.move("forward")
    assert cam.forgets == 1  # 走过之后镜头参照图对不上了：复位时只粗转
    b.step()
    device.frames = [np.zeros((720, 1280, 3), np.uint8)]
    clock.advance(1.5)
    b.step()
    assert cam.forgets == 2


# ---- look_person 换角度（peek）：好友躲在团子身后，边转边看让他露出来 ----
from types import SimpleNamespace  # noqa: E402

from skydango.brain.camera import Camera  # noqa: E402
from skydango.vision.people import Person  # noqa: E402

ME = Rect(860, 400, 200, 400)  # 团子：画面正中
BEHIND_TAG = (900, 360, 120, 36)  # 名字标签压在团子头上
CLEAR_BODY = Rect(1250, 420, 160, 400)  # 露出来之后的身体框（和团子不重叠、够大）


class PeekEnv(FakeEnv):
    """按"已经按了几次镜头键"切换场景：scenes[i] = 按了 i 次之后感知层看到的 (标签, 身体框)，最后一个一直保持。

    delay=True：像真的感知层（后台线程）一样晚一轮 —— 这次 observe 交进来的帧，下一次 observe 才出结果。"""

    def __init__(self, device, clock, scenes, me=ME, delay=False):
        super().__init__()
        self.device, self.clock, self.scenes, self.me, self.delay = device, clock, scenes, me, delay
        self.observed = 0
        self.pending = None
        self._publish(self._compute())

    def _compute(self):
        tag, body_box = self.scenes[min(len(camera_keys(self.device)), len(self.scenes) - 1)]
        return tag, body_box, self.clock()

    def _publish(self, state):
        tag, body_box, t = state
        self.labels = {"小明": (*tag, t)} if tag else {}
        self.people_list = [Person(2, "friend", "小明", body_box, "右边", "近")] if body_box else []
        self.last_tracks = ([SimpleNamespace(id=1, cls="self", box=self.me, last=t)] if self.me else []) + (
            [SimpleNamespace(id=2, cls="player", box=body_box, last=t)] if body_box else [])

    def _apply(self):
        self._publish(self._compute())

    def observe(self, frame, now, panel_visible):
        self.observed += 1
        state = self._compute()
        if not self.delay:
            self._publish(state)
            return
        if self.pending is not None:
            self._publish(self.pending)
        self.pending = state


class ShiftDevice(FakeDevice):
    """画面亮度跟着聊天面板开关变（开着 30、关着 200）：看裁出来的图是不是在面板关着时截的。"""

    def __init__(self):
        super().__init__([np.zeros((1080, 1920, 3), np.uint8)])
        self.reader = None

    def screenshot(self):
        closed = self.reader is not None and self.reader.panel_closed_since is not None
        return np.full((1080, 1920, 3), 200 if closed else 30, np.uint8)


def peek_body(clock, scenes, live=True, delay=False, **kw):
    kw.setdefault("frames", [np.full((1080, 1920, 3), 90, np.uint8)])  # 坐标按 1920×1080
    b, device, reader, events = body(clock, live=live, **kw)
    b.sleep = clock.advance  # 等待会让时间往前走：按键之后的结果才分得出新旧
    env = PeekEnv(device, clock, scenes, delay=delay)
    b.env = env
    b.camera = Camera(device, 0.25, None, sleep=lambda s: None)
    return b, device, env


def camera_keys(device):  # 镜头键：左右（105 / 106）、拉近拉远（12 / 13）；不算借面板按的 C（46）
    return [c for c in device.calls if c[0] in ("hw_hold", "hw_key") and c[1] in (105, 106, 12, 13)]


presses = camera_keys


def jpeg_mean(block):
    data = base64.b64decode(block["source"]["data"])
    return float(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE).mean())


def test_status_says_who_is_hidden_behind_you(clock):
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None)])
    assert "被你挡住：小明" in b.status()


def test_status_quiet_when_friend_is_visible(clock):
    b, _, _ = peek_body(clock, [((1270, 380, 120, 36), CLEAR_BODY)])
    assert "被你挡住" not in b.status()


def test_look_person_turns_until_friend_is_revealed(clock):
    b, device, env = peek_body(clock, [(BEHIND_TAG, None), ((980, 360, 120, 36), None), ((1270, 380, 120, 36), CLEAR_BODY)])
    img, note = b.look_person("小明")
    held = [c for c in presses(device) if c[0] == "hw_hold"]
    assert len(held) == 2 and all(c[1] == 105 for c in held)  # 标签在团子中心偏左…偏右 → 都按左（推开）
    assert "挡住" in note["text"] and "转了一下镜头" in note["text"]
    assert "按名字标签估的" not in note["text"]  # 露出来了：用身体框裁
    assert "camera_reset" in note["text"]
    assert not b.camera.at_home()  # 看完不复位
    assert img["type"] == "image"


def test_look_person_direction_follows_tag_side(clock):
    b, device, _ = peek_body(clock, [((880, 360, 120, 36), None), ((1270, 380, 120, 36), CLEAR_BODY)])
    b.look_person("小明")
    assert presses(device)[0][1] == 106  # 标签中心 940 在团子中心 960 左边：按右


def test_look_person_zooms_in_on_small_revealed_friend(clock):
    small = Rect(1300, 500, 60, 150)
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 460, 120, 36), small), ((1250, 380, 120, 36), CLEAR_BODY)])
    b.look_person("小明")
    assert ("hw_key", 12) in presses(device)  # 拉近一步
    assert "拉近了 1 步" in b.camera.describe()


def test_look_person_gives_up_within_budget(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None)])  # 怎么转都出不来
    _, note = b.look_person("小明")
    assert 0 < len(presses(device)) <= b.cfg.peek.max_presses
    assert "没看清" in note["text"] and "按名字标签估的" in note["text"]


def test_look_person_does_not_peek_in_dry_run(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None)], live=False)
    _, note = b.look_person("小明")
    assert presses(device) == []
    assert "被你挡住" in note["text"] and "dry-run" in note["text"]


def test_look_person_does_not_interrupt_camera_skill(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None)])
    skill = camera_skill()
    b.skills.start(b, skill)
    _, note = b.look_person("小明")
    assert presses(device) == [] and b.skills.active is skill
    assert "被你挡住" in note["text"]


def test_look_person_does_not_peek_in_blackout(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None)])
    b.blackout = True
    b.look_person("小明")
    assert presses(device) == []


def test_look_person_does_not_peek_when_disabled(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None)])
    b.cfg.peek.enabled = False
    b.look_person("小明")
    assert presses(device) == []


def test_look_person_does_not_peek_when_friend_is_beside_you(clock):
    b, device, _ = peek_body(clock, [((1500, 380, 120, 36), None)])
    _, note = b.look_person("小明")
    assert presses(device) == [] and "挡住" not in note["text"]


def test_look_person_ignores_self_box_off_center(clock):  # YOLO 把旁边的人认成团子：不采信，不算挡住
    b, device, env = peek_body(clock, [((150, 360, 120, 36), None)])
    env.me = Rect(100, 400, 200, 400)
    env._apply()
    b.look_person("小明")
    assert presses(device) == []


def test_self_box_well_right_of_center_still_counts(clock):
    """实测（2026-09-30 22:28）：镜头没跟正，团子站在画面右边（框中心 x≈1510，偏中线 29% 屏宽），
    YOLO 认得很稳（0.96），旧规定"偏 15% 以内"把它丢了 → 团子框 None，peek 永远不触发。"""
    b, _, env = peek_body(clock, [((1450, 420, 120, 36), None)])
    env.me = Rect(1446, 518, 130, 309)
    env._apply()
    assert "被你挡住：小明" in b.status()


def test_look_person_peek_clears_scene_change_reference(clock):
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 380, 120, 36), CLEAR_BODY)])
    b._ref_thumb = np.zeros((9, 16), np.uint8)
    b.look_person("小明")
    assert b._ref_thumb is None  # 自己转的，不算画面大变


# ---- look_person 换角度：评审修正 ----

def test_peek_crops_frame_taken_while_panel_is_lent(clock):  # 面板开 / 关画面横移约 130 px：裁图要用量框时那一帧
    device = ShiftDevice()
    b, _, env = peek_body(clock, [(BEHIND_TAG, None), ((1270, 380, 120, 36), CLEAR_BODY)],
                          frames=None, panel_mode="always", device_override=device)
    device.reader = b.reader
    assert b.reader.panel_closed_since is None  # always 模式：面板平时开着
    img, note = b.look_person("小明")
    assert "转了一下镜头" in note["text"]
    assert jpeg_mean(img) > 150  # 面板关着时截的（亮度 200），不是还回面板之后（30）


def test_peek_ignores_results_from_before_the_press(clock):  # 感知层晚一轮：拉近之后先读到的是拉近前的小框
    small = Rect(1300, 500, 60, 150)
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 460, 120, 36), small), ((1250, 380, 120, 36), CLEAR_BODY)],
                             delay=True)
    b.look_person("小明")
    assert [c for c in presses(device) if c == ("hw_key", 12)] == [("hw_key", 12)]  # 只拉近一次，没被旧的小框骗着再拉


def test_peek_budget_spent_while_enlarging_still_counts_as_seen(clock):
    small = Rect(1300, 500, 60, 150)
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 460, 120, 36), small)])
    b.cfg.peek.max_presses = 2  # 转一下露出来、拉近一下，预算就用完了
    _, note = b.look_person("小明")
    assert "没看清" not in note["text"] and "按名字标签估的" not in note["text"]


def test_peek_forgets_scene_reference_even_if_a_key_press_fails(clock):
    b, device, _ = peek_body(clock, [(BEHIND_TAG, None), ((980, 360, 120, 36), None)])
    b._ref_thumb = np.zeros((9, 16), np.uint8)
    calls = []

    def boom(code, seconds):
        calls.append(code)
        if len(calls) == 2:
            raise RuntimeError("adb 断了")

    device.hw_key_hold = boom
    with pytest.raises(RuntimeError):
        b.look_person("小明")
    assert b._ref_thumb is None


def test_peek_ignores_sweep_self_box_after_zooming(clock):  # 转圈认出的团子框在缩放后大小不对：循环里只信当前帧的 YOLO
    b, device, env = peek_body(clock, [(BEHIND_TAG, None)])
    env.me = Rect(880, 400, 200, 400)  # YOLO 的团子框（中心 980）
    env._apply()
    env.self_box = Rect(660, 100, 600, 900)  # 转圈认的旧框（中心 960，离中线更近）、很大：采信了会一直判"贴太近"拉远
    b.cfg.peek.max_presses = 3
    b.look_person("小明")
    assert ("hw_key", 13) not in presses(device)


def test_status_hidden_note_depends_on_whether_peek_can_run(clock):
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None)], live=False)
    assert "被你挡住：小明" in b.status() and "会自己换角度" not in b.status()


ON_ME = Rect(850, 395, 220, 410)  # 团子身上的 player 框（比 self 框宽一圈）


def test_body_box_on_self_is_not_the_friend(clock):
    """10-02 晚 22:26：好友在团子正后方，标签在团子头顶，团子身上的 player 框挂上了他的名字。
    这个框就是团子：不算他的身体（不算露出来），照样换角度。"""
    b, device, _ = peek_body(clock, [(BEHIND_TAG, ON_ME), ((980, 360, 120, 36), None), ((1270, 380, 120, 36), CLEAR_BODY)])
    assert "被你挡住：小明" in b.status()
    img, note = b.look_person("小明")
    assert len([c for c in presses(device) if c[0] == "hw_hold"]) == 2
    assert "转了一下镜头" in note["text"] and "按名字标签估的" not in note["text"]
    assert "(1218, 340)" in note["text"]  # 裁的是露出来之后的身体框（CLEAR_BODY 四周放宽 20%），不是团子


def test_look_person_saves_what_it_showed_the_brain(clock, tmp_path):
    """look_person 裁给大脑的图存进运行目录，事后能核对看到的是谁。"""
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 380, 120, 36), CLEAR_BODY)], run=SimpleNamespace(path=tmp_path))
    b.look_person("小明")
    saved = sorted(p.name for p in (tmp_path / "look_person").iterdir())
    assert len(saved) == 2 and saved[0].endswith("-小明-crop.jpg") and saved[1].endswith("-小明-frame.jpg")


def test_peek_logs_each_step(clock, caplog):
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None), ((980, 360, 120, 36), None), ((1270, 380, 120, 36), CLEAR_BODY)])
    with caplog.at_level(logging.DEBUG, logger="skydango.brain.body"):
        b.look_person("小明")
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("换角度")]
    assert len(steps) >= 3 and any("Turn" in s for s in steps) and "revealed" in steps[-1]


def test_peek_lost_after_undoing_zoom_still_uses_revealed_view(clock):  # 露出来过、拉近把人推出画面、退回后仍看不到
    small = Rect(1300, 500, 60, 150)
    b, _, _ = peek_body(clock, [(BEHIND_TAG, None), ((1270, 460, 120, 36), small), (None, None)])
    _, note = b.look_person("小明")
    assert "没看清" not in note["text"] and "按名字标签估的" not in note["text"]


# ---- 接还没回的好友聊天不算主动开口（spec 2026-10-01-lull-musing §4 修复 1） ----
def test_reply_to_pending_chat_not_proactive(clock):
    b, _, reader = pro_body(clock)
    reader.batches = [[msg("在吗", speaker="阿花")]]
    b.step()
    b.say("在呢")  # 这一轮是被别的事件叫醒的（brain_busy 为假），但前面有没回的话
    assert b.spoken[-1].proactive is False
    assert b.occasion().last is None


def test_pending_expires(clock):
    b, _, reader = pro_body(clock)
    reader.batches = [[msg("在吗", speaker="阿花")]]
    b.step()
    clock.advance(91)
    b.say("在呢")
    assert b.spoken[-1].proactive is True


def test_after_my_reply_back_to_proactive(clock):
    b, _, reader = pro_body(clock)
    reader.batches = [[msg("在吗", speaker="阿花")]]
    b.step()
    b.say("在呢")
    clock.advance(5)
    b.say("还有事吗")  # 已经接过了：再说就是主动的
    assert [s.proactive for s in b.spoken] == [False, True]


def test_stranger_line_not_pending(clock):
    b, _, reader = pro_body(clock)
    reader.batches = [[msg("hi", speaker="")]]
    b.step()
    b.say("你好")
    assert b.spoken[-1].proactive is True


def test_status_lists_icons(clock):
    from skydango.vision.icons_map import Icon

    class IconEnv(FakeEnv):
        icon_list = []

        def icons(self, now):
            return list(self.icon_list)

    env = IconEnv()
    b, _, _, _ = body(clock, env=env)
    assert "画面里的图标" not in b.status()
    env.icon_list = [Icon(1, "sit", "bench", "坐下", Rect(0, 0, 1, 1), "右边"), Icon(2, "unknown", "map", "", Rect(0, 0, 1, 1), "左边")]
    assert "画面里的图标：坐下（右边）、不认识的图标 1 个" in b.status()
    plain, _, _, _ = body(clock, env=FakeEnv())
    assert "画面里的图标" not in plain.status()


def test_status_and_scene_note_with_real_envwatcher(clock):
    # 真 EnvWatcher（非 YOLO 退路）没有 icons() 方法：状态和 scene_note 不报错、也没有图标行
    from skydango.brain.images import scene_note
    from skydango.config import EnvConfig
    from skydango.vision.env import EnvWatcher

    env = EnvWatcher(None, EnvConfig(interval=3.0), lambda: [], [0.0, 0.0, 0.335, 0.855], background=False, icons=object())
    assert not hasattr(env, "icons")
    b, _, _, _ = body(clock, env=env)
    assert "画面里的图标" not in b.status()
    assert "图标" not in scene_note(env, 0.0, 1.0)


def test_status_has_models_line(clock):  # spec 2026-10-05-model-providers §4
    b = body(clock)[0]
    plain = b.status()
    assert "模型：" not in plain
    b.models_line = lambda: "大脑 deepseek/deepseek-chat"
    assert "模型：大脑 deepseek/deepseek-chat" in b.status()
