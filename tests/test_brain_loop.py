import logging
import threading
import time

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
    assert brain.farewell() is True
    assert session.sent == [SUMMARY_REQUEST] and "在雨林和懒懒 玩了一会儿" in store.inbox()
    brain.failing_since = 0.0
    assert brain.farewell() is False  # 正在失败：不再发


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
    assert brain.farewell() is True
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
