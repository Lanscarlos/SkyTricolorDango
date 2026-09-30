import logging
import threading
import time

import pytest

from skydango.brain.claude import ClaudeError
from skydango.brain.events import EventQueue
from skydango.brain.loop import Brain, log_brain_message
from skydango.brain.prompt import SUMMARY_REQUEST
from skydango.chat.memory import MemoryStore
from skydango.config import BrainConfig, ChatConfig


def ok(text="好"):
    return {"subtype": "success", "result": text, "num_turns": 2, "total_cost_usd": 0.01, "usage": {"input_tokens": 5}}


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        r = self.results.pop(0) if self.results else ok()
        if isinstance(r, Exception):
            raise r
        return r


class FakeToolBox:
    def __init__(self):
        self.acted = False
        self.used = []
        self.status_calls = 0

    def begin_turn(self):
        self.acted = False
        self.used = []

    def status(self):
        self.status_calls += 1
        return "面板开"


class FakeEyes:
    def summary(self, now):
        return "场景（3 秒前看的）：\n在雨林"


class FakeRun:
    def __init__(self):
        self.entries = []

    def record_brain(self, entry):
        self.entries.append(entry)


class FakeTrace:
    def __init__(self, broken=False):
        self.calls = []
        self.state = None
        self.broken = broken

    def _record(self, *call):
        if self.broken:
            raise RuntimeError("记录坏了")
        self.calls.append(call)

    def begin(self, reason, prompt):
        self._record("begin", reason, prompt)

    def finish(self, result, seconds):
        self._record("finish", result, seconds)

    def fail(self, error, seconds):
        self._record("fail", error, seconds)


def make(clock, session, store=None, run=None, nearby=None, trace=None, **cfg):
    events = EventQueue(clock=clock)
    tb = FakeToolBox()
    near = [] if nearby is None else nearby
    brain = Brain(BrainConfig(**cfg), ChatConfig(), session, tb, events, lambda now: list(near), eyes=FakeEyes(),
                  clock=clock, wall=lambda: 0.0, run=run, store=store, trace=trace)
    return brain, events, tb, near


def test_first_wake_is_immediate_and_sends_text_only(clock):
    session = FakeSession()
    brain, _, _, _ = make(clock, session)
    assert brain.due(clock()) == "heartbeat"
    brain.wake(clock(), "heartbeat")
    text = session.sent[0]
    assert "没有新事件" in text and "状态：面板开" in text and "在雨林" in text


def test_events_wait_for_debounce_and_go_into_message(clock):
    session = FakeSession()
    brain, events, _, _ = make(clock, session)
    brain.last_wake = clock()
    events.put("chat", "聊天  懒懒：「在吗」")
    assert brain.due(clock() + 0.3) is None
    assert brain.due(clock() + 1.0) == "events"
    brain.wake(clock() + 1.0, "events")
    assert "- 聊天  懒懒：「在吗」" in session.sent[0]


def test_failure_backs_off_then_goes_offline(clock):
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("挂了"), ok()))
    t0 = clock()
    brain.wake(t0, "heartbeat")
    assert brain.due(t0 + 5) is None  # 退避 10 秒
    assert not brain.offline(t0 + 100) and brain.offline(t0 + 121)
    brain.wake(t0 + 130, "heartbeat")
    assert not brain.offline(t0 + 131) and brain.failures == 0


def test_limit_waits_longer(clock):
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("hit your limit", limit=True)))
    brain.wake(clock(), "heartbeat")
    assert brain.backoff_until == clock() + 600


def test_heartbeat_backs_off_when_idle_and_resets(clock):
    brain, _, tb, near = make(clock, FakeSession())
    assert brain.heartbeat(clock()) == 90  # 身边没好友：从第二档起
    brain.wake(clock(), "heartbeat")
    assert brain.heartbeat(clock()) == 180
    near.append("懒洋洋大王")
    assert brain.heartbeat(clock()) == 90
    brain.wake(clock(), "events")
    assert brain.heartbeat(clock()) == 45


def test_record_brain_entry(clock):
    run = FakeRun()
    brain, _, tb, _ = make(clock, FakeSession(), run=run)
    tb.used = ["say"]
    brain.toolbox.begin_turn = lambda: None  # 保留上面放进去的 used
    brain.wake(clock(), "heartbeat")
    (entry,) = run.entries
    assert entry["tools"] == ["say"] and entry["total_cost_usd"] == 0.01 and entry["num_turns"] == 2


def test_thread_survives_errors():
    class BrokenOnce(FakeToolBox):
        def status(self):
            if self.status_calls == 0:
                self.status_calls += 1
                raise RuntimeError("friends.md 读不了")
            return super().status()

    events = EventQueue()
    brain = Brain(BrainConfig(), ChatConfig(), FakeSession(), BrokenOnce(), events, lambda now: [])
    stop = threading.Event()
    t = threading.Thread(target=brain.run, args=(stop,), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while brain.failures == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert brain.failures == 1 and t.is_alive()  # 出错了但线程没死
    stop.set()
    t.join(3)
    assert not t.is_alive()


def test_farewell_writes_summary_to_inbox(clock, tmp_path):
    store = MemoryStore(tmp_path)
    session = FakeSession(ok("在雨林和懒懒 玩了一会儿"))
    brain, _, _, _ = make(clock, session, store=store)
    assert brain.farewell() == "在雨林和懒懒 玩了一会儿"
    assert session.sent == [SUMMARY_REQUEST] and "在雨林和懒懒 玩了一会儿" in store.inbox()
    brain.failing_since = 0.0
    assert brain.farewell() == ""  # 正在失败：不再发


# ---- 交给可视化网页的记录（trace） ----
def test_wake_reports_to_trace(clock):
    session, trace = FakeSession(ok("好")), FakeTrace()
    brain, _, _, _ = make(clock, session, trace=trace)
    brain.wake(clock(), "heartbeat")
    assert trace.calls[0] == ("begin", "heartbeat", session.sent[0])
    kind, result, seconds = trace.calls[1]
    assert kind == "finish" and result["result"] == "好" and seconds >= 0
    assert len(trace.calls) == 2


def test_failed_wake_reports_error(clock):
    trace = FakeTrace()
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("超时")), trace=trace)
    brain.wake(clock(), "events")
    assert trace.calls[0][:2] == ("begin", "events")
    assert trace.calls[-1][0] == "fail" and "超时" in trace.calls[-1][1]


def test_farewell_reports_to_trace(clock, tmp_path):
    trace = FakeTrace()
    brain, _, _, _ = make(clock, FakeSession(ok("玩了一会儿")), store=MemoryStore(tmp_path), trace=trace)
    assert brain.farewell() == "玩了一会儿"
    assert trace.calls[0] == ("begin", "farewell", SUMMARY_REQUEST)
    assert trace.calls[-1][0] == "finish"


def test_trace_state(clock):
    trace = FakeTrace()
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("挂了")), trace=trace)
    assert trace.state == brain.trace_state
    cfg = brain.cfg
    assert brain.trace_state() == {"model": cfg.model, "effort": cfg.effort, "failures": 0, "retry_in": None, "offline": False}
    brain.wake(clock(), "heartbeat")
    state = brain.trace_state()
    assert state["failures"] == 1 and state["retry_in"] == 10.0 and state["offline"] is False


def test_trace_errors_do_not_break_wake(clock):
    run, session = FakeRun(), FakeSession()
    brain, _, _, _ = make(clock, session, run=run, trace=FakeTrace(broken=True))
    brain.wake(clock(), "heartbeat")
    assert len(session.sent) == 1 and len(run.entries) == 1 and brain.failures == 0


def test_log_brain_message(caplog):
    caplog.set_level(logging.INFO, logger="skydango.brain.loop")
    log_brain_message({"type": "assistant", "message": {"content": [{"type": "text", "text": "先看看"}]}})
    log_brain_message({"type": "user", "message": {"content": []}})
    assert "大脑想：先看看" in caplog.text


def test_unexpected_error_still_closes_the_turn(clock, tmp_path):
    # 起不来进程之类（不是 ClaudeError）：run() 兜住退避，但时间线上这一轮要标失败、带上错误，不能一直"进行中"
    trace = FakeTrace()
    brain, _, _, _ = make(clock, FakeSession(FileNotFoundError("claude 不在")), store=MemoryStore(tmp_path), trace=trace)
    with pytest.raises(FileNotFoundError):
        brain.wake(clock(), "heartbeat")
    assert trace.calls[-1][0] == "fail" and "FileNotFoundError" in trace.calls[-1][1] and "claude 不在" in trace.calls[-1][1]
    brain.session = FakeSession(OSError("写不了"))
    with pytest.raises(OSError):
        brain.farewell()
    assert trace.calls[-1][0] == "fail" and "写不了" in trace.calls[-1][1]


def test_chat_turn_only_while_answering_chat(clock):  # I4：只有取走了聊天的那一轮才算"在回话"
    seen = []

    class Watching(FakeSession):
        def send(self, text):
            seen.append(brain.chat_turn)
            return super().send(text)

    brain, events, _, _ = make(clock, Watching(ok(), ok(), RuntimeError("进程起不来")))
    assert brain.chat_turn is False
    brain.wake(clock(), "heartbeat")  # 心跳：不算
    events.put("arrive", "小明 来到身边")
    brain.wake(clock(), "events")  # 只有来人：不算
    events.put("chat", "聊天  小明：「在吗」")
    with pytest.raises(RuntimeError):
        brain.wake(clock(), "events")
    assert seen == [False, False, True] and brain.chat_turn is False


# ---- 背景事件（events.BACKGROUND）：自己不叫醒，攒够 background_wait 秒兜底叫醒一次 ----


def test_background_events_alone_do_not_wake_right_away(clock):
    brain, events, _, _ = make(clock, FakeSession(), background_wait=20.0)
    brain.last_wake = clock()
    events.put("stranger", "身边来了陌生人（1 个，头顶没有名字）")
    events.put("leave", "懒懒 走开了（5 秒没看到名字）", who="懒懒")
    assert brain.due(clock() + 1.0) is None
    assert brain.due(clock() + 19.0) is None
    assert brain.due(clock() + 20.0) == "background"


def test_urgent_event_wakes_and_takes_background_along(clock):
    session = FakeSession()
    brain, events, _, _ = make(clock, session, background_wait=20.0)
    brain.last_wake = clock()
    events.put("stranger", "身边来了陌生人（1 个，头顶没有名字）")
    clock.advance(5)
    events.put("arrive", "懒懒 来到身边", who="懒懒")
    assert brain.due(clock() + 0.3) is None  # 照旧攒 debounce
    assert brain.due(clock() + 1.0) == "events"
    brain.wake(clock() + 1.0, "events")
    assert "- 身边来了陌生人" in session.sent[0] and "- 懒懒 来到身边" in session.sent[0]


def test_background_wake_does_not_reset_idle_heartbeat(clock):
    brain, events, _, _ = make(clock, FakeSession(), background_wait=20.0)
    brain._idle = 2
    events.put("stranger", "陌生人都走开了")
    brain.wake(clock() + 20.0, "background")
    assert brain._idle == 3  # 只是周围在变、大脑也没做事：心跳照样往后退


def test_last_turn_records_start_and_end(clock):
    class Slow(FakeSession):  # send 时时间走 5 秒
        def send(self, text):
            clock.advance(5.0)
            return super().send(text)

    brain, _, _, _ = make(clock, Slow(ok(), ClaudeError("挂了")))
    assert brain.last_turn == (float("-inf"), float("-inf"))
    start = clock()
    brain.wake(start, "heartbeat")
    assert brain.last_turn == (start, start + 5.0)
    brain.wake(clock(), "heartbeat")  # 失败的一轮也记
    assert brain.last_turn == (start + 5.0, start + 10.0)


def test_last_turn_starts_before_draining_events(clock):
    """审查 I1：拼消息（status 要等身体线程）期间身体读到的新消息不在这一轮里，这一轮的开始要早于它。"""
    brain, _, tb, _ = make(clock, FakeSession())
    status = tb.status

    def slow_status():
        clock.advance(2.0)  # body.call(status) 要等身体线程下一圈
        return status()

    tb.status = slow_status
    start = clock()
    brain.wake(start, "events")
    assert brain.last_turn[0] == start


def test_heartbeat_slower_when_sleepy(clock):
    brain, _, _, near = make(clock, FakeSession(), nearby=["阿花"])
    assert brain.heartbeat(clock()) == 45
    brain.slow = lambda: True
    assert brain.heartbeat(clock()) == 90
    brain._idle = 5
    assert brain.heartbeat(clock()) == 180  # 不越界


# ---- 沙盒计划 Task 2：in_turn、模拟时钟 ----
def test_in_turn_only_during_a_wake(clock):
    seen = []

    class Peek(FakeSession):
        def send(self, text):
            seen.append(brain.in_turn)
            return super().send(text)

    brain, *_ = make(clock, Peek(ok(), ClaudeError("坏了"), RuntimeError("进程起不来")))
    assert brain.in_turn is False
    brain.wake(clock(), "heartbeat")
    assert seen == [True] and brain.in_turn is False
    brain.wake(clock(), "heartbeat")  # ClaudeError：退避
    assert seen == [True, True] and brain.in_turn is False
    with pytest.raises(RuntimeError):
        brain.wake(clock(), "heartbeat")  # 别的异常：run() 兜住
    assert seen == [True, True, True] and brain.in_turn is False


def test_sim_clock_skip_then_chat_wakes_within_debounce():  # Review Focus 1：大脑和事件队列同一个钟
    from skydango.sandbox.clock import SimClock

    sim = SimClock()
    events = EventQueue(clock=sim.clock)
    session = FakeSession()
    chat = ChatConfig()
    brain = Brain(BrainConfig(heartbeat=[1e9]), chat, session, FakeToolBox(), events, lambda now: [],
                  clock=sim.clock, wall=sim.wall)
    stop = threading.Event()
    t = threading.Thread(target=brain.run, args=(stop,), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while not session.sent and time.monotonic() < deadline:  # 刚上线马上醒一次
            time.sleep(0.02)
        assert len(session.sent) == 1
        sim.skip(3600)
        started = time.monotonic()
        events.put("chat", "小明：在吗")
        while len(session.sent) < 2 and time.monotonic() - started < chat.debounce + 3:
            time.sleep(0.02)
        assert len(session.sent) == 2 and "在吗" in session.sent[1]
        assert time.monotonic() - started < chat.debounce + 1.5
    finally:
        stop.set()
        t.join(3)
