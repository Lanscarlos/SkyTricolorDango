import copy
from types import SimpleNamespace

from skydango.brain.budget import Budget
from skydango.brain.context import Context
from skydango.brain.events import EventQueue
from skydango.brain.loop import Brain
from skydango.chat.memory import MemoryStore
from skydango.config import BrainConfig, ChatConfig

IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "x"}}


class ScriptedClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, system, messages, tools=None, max_tokens=None):
        self.calls.append(copy.deepcopy(messages))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def text(t, input_tokens=100):
    return {"content": [{"type": "text", "text": t}], "stop_reason": "end_turn",
            "usage": {"input_tokens": input_tokens, "output_tokens": 10}}


def tools(*uses, stop="tool_use"):
    content = [{"type": "tool_use", "id": f"t{i}", "name": n, "input": a} for i, (n, a) in enumerate(uses)]
    return {"content": content, "stop_reason": stop, "usage": {"input_tokens": 100, "output_tokens": 10}}


class ToolBoxSpy:
    def __init__(self):
        self.ran = []
        self.body = SimpleNamespace(last_look=float("-inf"), blackout=False)

    def run(self, name, args):
        self.ran.append((name, args))
        if name == "look":
            return [dict(IMG), {"type": "text", "text": "名字"}], False
        if name == "status":
            return "面板开", False
        return f"{name} ok", False


def make(clock, client, store=None, **cfg):
    c = BrainConfig(**cfg)
    events = EventQueue(clock=clock)
    ctx = Context(c, "规则", lambda: "记忆")
    tb = ToolBoxSpy()
    near = []
    brain = Brain(c, ChatConfig(), client, ctx, tb, events, Budget(c), lambda now: list(near),
                  clock=clock, wall=lambda: 0.0, store=store)
    return brain, events, tb, ctx, near


def test_first_wake_is_immediate_and_looks_around(clock):
    brain, _, tb, ctx, _ = make(clock, ScriptedClient(text("先看看")))
    assert brain.due(clock()) == "heartbeat"
    brain.wake(clock(), "heartbeat")
    assert [n for n, _ in tb.ran] == ["look", "status"]
    first = ctx.messages[0]["content"]
    assert "没有新事件" in first[0]["text"] and "状态：面板开" in first[0]["text"] and first[1]["type"] == "image"
    assert ctx.messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "先看看"}]}


def test_events_wait_for_debounce(clock):
    brain, events, _, _, _ = make(clock, ScriptedClient())
    brain.last_wake = clock()
    events.put("chat", "聊天  懒懒：在吗")
    assert brain.due(clock() + 0.3) is None
    assert brain.due(clock() + 1.0) == "events"


def test_tool_loop_runs_tools_and_feeds_results(clock):
    client = ScriptedClient(tools(("say", {"text": "在呢"})), text("好"))
    brain, events, tb, ctx, _ = make(clock, client)
    tb.body.last_look = clock()  # 刚看过，不自动附图
    events.put("chat", "聊天  懒懒：在吗")
    brain.wake(clock() + 1, "events")
    assert ("say", {"text": "在呢"}) in tb.ran and ("look", {}) not in tb.ran
    assert client.calls[1][-1]["content"][0] == {"type": "tool_result", "tool_use_id": "t0", "content": "say ok"}
    assert ctx.messages[-1]["content"][0]["text"] == "好"


def test_says_and_steps_are_capped(clock):
    client = ScriptedClient(tools(("say", {"text": "a"}), ("say", {"text": "b"}), ("say", {"text": "c"})), text("好"))
    brain, _, tb, ctx, _ = make(clock, client, max_says=2)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    results = ctx.messages[-2]["content"]
    assert [r.get("is_error", False) for r in results] == [False, False, True]
    assert [n for n, _ in tb.ran].count("say") == 2

    client = ScriptedClient(tools(("say", {"text": "a"}), ("emote", {"name": "鞠躬"})))
    brain, _, tb, ctx, _ = make(clock, client, max_steps=1)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert len(client.calls) == 1  # 到上限就不再问它
    assert ctx.messages[-1]["content"][1]["is_error"] is True


def test_truncated_turn_is_dropped_and_nothing_runs(clock):
    brain, _, tb, ctx, _ = make(clock, ScriptedClient(tools(("say", {"text": "在"}), stop="max_tokens")))
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert "say" not in [n for n, _ in tb.ran]
    assert ctx.messages[-1]["role"] == "user"  # 没有结果的 tool_use 不能留在记录里


def test_api_failure_backs_off_then_goes_offline(clock):
    brain, _, tb, _, _ = make(clock, ScriptedClient(RuntimeError("网络断了"), text("好了")))
    tb.body.last_look = clock()
    t0 = clock()
    brain.wake(t0, "heartbeat")
    assert brain.due(t0 + 5) is None  # 退避 10 秒
    assert not brain.offline(t0 + 100) and brain.offline(t0 + 121)
    brain.wake(t0 + 130, "heartbeat")
    assert not brain.offline(t0 + 131) and brain.failures == 0


def test_heartbeat_backs_off_when_idle_and_resets_on_events(clock):
    brain, _, tb, _, near = make(clock, ScriptedClient(text("没事"), text("没事")))
    tb.body.last_look = clock()
    assert brain.heartbeat(clock()) == 90  # 身边没好友：从第二档起
    brain.wake(clock(), "heartbeat")
    assert brain.heartbeat(clock()) == 180
    near.append("懒洋洋大王")
    assert brain.heartbeat(clock()) == 90
    brain.wake(clock(), "events")
    assert brain.heartbeat(clock()) == 45


def test_budget_limits_what_wakes_the_brain(clock):
    brain, events, _, _, _ = make(clock, ScriptedClient(), max_usd_per_hour=0.0001, pause_usd_per_hour=1.0)
    brain.budget.record({"input_tokens": 100}, clock())  # $0.0002
    events.put("arrive", "懒懒 来到身边")
    assert brain.due(clock() + 1) is None  # 只为聊天醒，心跳也停了
    events.put("chat", "聊天  懒懒：在吗")
    assert brain.due(clock() + 2) == "events"
    brain.budget.record({"input_tokens": 1_000_000}, clock())  # $2
    assert brain.due(clock() + 3) is None and brain.offline(clock() + 3)


def test_long_context_is_compacted_into_summary(clock, tmp_path):
    store = MemoryStore(tmp_path)
    client = ScriptedClient(text("嗯", input_tokens=200), text("在雨林，懒懒在旁边"))
    brain, _, tb, ctx, _ = make(clock, client, store=store, compact_tokens=150)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert "不要调用工具" in client.calls[1][-1]["content"][0]["text"]
    assert len(ctx.messages) == 1 and "在雨林" in ctx.messages[0]["content"][0]["text"]
    assert "在雨林" in store.inbox()
