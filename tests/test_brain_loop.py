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


def make(clock, client, store=None, run=None, **cfg):
    c = BrainConfig(**cfg)
    events = EventQueue(clock=clock)
    ctx = Context(c, "规则", lambda: "记忆")
    tb = ToolBoxSpy()
    near = []
    brain = Brain(c, ChatConfig(), client, ctx, tb, events, Budget(c), lambda now: list(near),
                  clock=clock, wall=lambda: 0.0, store=store, run=run)
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


# ---- Group D review fixes ----

class FakeRun:
    """假运行目录：只收集 record_brain 记的条目。"""

    def __init__(self):
        self.entries = []

    def record_brain(self, entry):
        self.entries.append(entry)


class BoomRun:
    """假运行目录：record_brain 一调就炸，用来测它不该打断当前这一轮。"""

    def record_brain(self, entry):
        raise RuntimeError("写日志炸了")


class Http400(Exception):
    """模拟 anthropic.BadRequestError：带 status_code == 400。"""

    def __init__(self, msg="记录不合法"):
        super().__init__(msg)
        self.status_code = 400


def test_refusal_stop_drops_turn_and_nothing_runs(clock):
    brain, _, tb, ctx, _ = make(clock, ScriptedClient(tools(("say", {"text": "在"}), stop="refusal")))
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert "say" not in [n for n, _ in tb.ran]
    assert ctx.messages[-1]["role"] == "user"  # 没有结果的 tool_use 不能留在记录里


def test_record_brain_logs_one_entry_per_model_call(clock):
    run = FakeRun()
    client = ScriptedClient(tools(("say", {"text": "在呢"})), text("好"))
    brain, _, tb, ctx, _ = make(clock, client, run=run)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert len(run.entries) == 2  # 调了两次模型，各记一条
    assert run.entries[0]["tools"] == ["say"]
    assert run.entries[1]["text"] == "好"


def test_log_error_does_not_break_turn(clock):
    """_log 写 brain.jsonl 出错不该让 tool_use 没有配对结果留在记录里（否则下次请求 400）。"""
    client = ScriptedClient(tools(("say", {"text": "在呢"})), text("好"))
    brain, _, tb, ctx, _ = make(clock, client, run=BoomRun())
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert ("say", {"text": "在呢"}) in tb.ran
    assert ctx.messages[-1]["content"][0]["text"] == "好"


def test_runaway_say_retries_are_capped_by_requests(clock):
    """模型不停调 say（一直被 max_says 拒绝）：每次 tool_use 都要计进 calls，且请求次数有硬上限。"""
    responses = [tools(("say", {"text": "还是要说"})) for _ in range(10)]
    client = ScriptedClient(*responses)
    brain, _, tb, ctx, _ = make(clock, client, max_steps=3, max_says=0)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert len(client.calls) <= brain.cfg.max_steps + 1


def test_empty_assistant_content_is_not_stored(clock):
    empty = {"content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 0}}
    client = ScriptedClient(tools(("say", {"text": "好的"})), empty)
    brain, _, tb, ctx, _ = make(clock, client)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert not any(m["role"] == "assistant" and m["content"] == [] for m in ctx.messages)
    assert all(m["content"] for m in ctx.messages)  # 不留空 content 的消息


def test_400_error_resets_context(clock):
    client = ScriptedClient(tools(("say", {"text": "喂"})), Http400())
    brain, _, tb, ctx, _ = make(clock, client)
    tb.body.last_look = clock()
    brain.wake(clock(), "heartbeat")
    assert len(ctx.messages) == 1
    assert "清空" in ctx.messages[0]["content"][0]["text"]
    assert brain.failing_since is not None  # 仍然按普通失败退避


def test_failure_backoff_uses_clock_at_failure_time(clock):
    """慢请求失败：退避该从失败发生的那一刻算，不是这一轮醒来时的 now。"""

    class SlowFailClient:
        def __init__(self):
            self.calls = 0

        def create(self, system, messages, tools=None, max_tokens=None):
            self.calls += 1
            clock.advance(60.0)  # 模拟这次请求花了 60 秒才失败
            raise RuntimeError("网络断了")

    brain, _, tb, _, _ = make(clock, SlowFailClient())
    tb.body.last_look = clock()
    t0 = clock()
    brain.wake(t0, "heartbeat")
    # 用旧的 now（t0）算的话退避到 t0+10，此时早就过去了；用失败发生那一刻（t0+60）算才对
    assert brain.backoff_until == t0 + 60.0 + 10.0
    assert brain.due(t0 + 65) is None  # 还在退避里


def test_compact_skipped_while_failing(clock):
    client = ScriptedClient(text("摘要"))  # 若没做防护，这条会被拿去用，压缩“成功”
    brain, _, tb, ctx, _ = make(clock, client)
    brain.failing_since = clock()
    ctx.messages.append({"role": "user", "content": [{"type": "text", "text": "x"}]})
    ctx.messages.append({"role": "assistant", "content": [{"type": "text", "text": "y"}]})
    assert brain.compact() is False  # 正在退避，不该再去调模型
    assert client.calls == []


def test_compact_rejects_non_end_turn_stop(clock):
    partial = {"content": [{"type": "text", "text": "摘要写了一半"}], "stop_reason": "max_tokens",
               "usage": {"input_tokens": 50, "output_tokens": 5}}
    client = ScriptedClient(partial)
    brain, _, tb, ctx, _ = make(clock, client)
    ctx.messages.append({"role": "user", "content": [{"type": "text", "text": "x"}]})
    ctx.messages.append({"role": "assistant", "content": [{"type": "text", "text": "y"}]})
    before = len(ctx.messages)
    assert brain.compact() is False
    assert len(ctx.messages) == before  # 追加的摘要请求被弹出，没有留痕
