"""ToolLoopBrain 的历史压缩（spec 2026-10-06-brain-compact §1~§4 §9）：只往后接、新记的 inbox、压缩、失败退回滑动。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skydango.brain.toolloop import ToolLoopBrain
from skydango.config import CompactConfig
from skydango.models.errors import ModelError

DATA = Path(__file__).parent / "data"
COMPACT_MARK = "（这不是事件，是整理"


def stamp(i: int) -> str:
    return f"[2026年10月6日（周二） 21:{i:02d}:00]"


def wake(i: int) -> str:
    text = f"{stamp(i)} 事件：\n- 小明：第 {i} 句\n状态：身边 小明"
    return text + "长" * 2000 if i == 5 else text  # 第 5 轮超过 HISTORY_TEXT，历史里会被截


def reply(content=None, calls=(), usage=None):
    tcs = [SimpleNamespace(id=f"c{n}", function=SimpleNamespace(name=name, arguments=args)) for n, (name, args) in enumerate(calls)]
    return SimpleNamespace(content=content, tool_calls=tcs or None, fake_usage=usage)


def turn_script(i: int) -> list:
    """第 i 轮的模型回复：每三轮调一次 say，其余只回文字。"""
    if i % 3 == 0:
        return [reply(None, [("say", json.dumps({"text": f"你好{i}"}, ensure_ascii=False))]), reply(f"说完了{i}")]
    return [reply(f"想了想{i}")]


class FakeLLM:
    """记下每次 create 的参数；压缩指令按 compact 脚本回，别的按轮脚本回。usage = 每次请求报的 prompt_tokens。"""

    def __init__(self, usage=None, compact=()):
        self.requests: list[dict] = []
        self.queue: list = []
        self.compact = list(compact)
        self.usage = usage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        kw = copy.deepcopy(kw)
        self.requests.append(kw)
        last = kw["messages"][-1]
        if last["role"] == "user" and str(last["content"]).startswith(COMPACT_MARK):
            item = self.compact.pop(0) if self.compact else reply("前情提要正文")
        else:
            item = self.queue.pop(0) if self.queue else reply("好")
        if isinstance(item, Exception):
            raise item
        resp = SimpleNamespace(choices=[SimpleNamespace(message=item)])
        u = item.fake_usage or (self.usage and (self.usage, 10, self.usage - 100))
        if u:
            resp.usage = SimpleNamespace(prompt_tokens=u[0], completion_tokens=u[1], prompt_cache_hit_tokens=u[2])
        return resp

    def turn_requests(self):
        return [r for r in self.requests if not str(r["messages"][-1]["content"]).startswith(COMPACT_MARK)]

    def compact_requests(self):
        return [r for r in self.requests if str(r["messages"][-1]["content"]).startswith(COMPACT_MARK)]


def toolbox():
    return SimpleNamespace(run=lambda name, args: ("好", False))


SAY = {"type": "function", "function": {"name": "say", "description": "说一句",
       "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}


def make(llm=None, **kw):
    llm = llm or FakeLLM()
    kw.setdefault("history", 8)
    brain = ToolLoopBrain(llm, "系统提示词", toolbox(), [SAY], model="deepseek-flash", temperature=1.0, max_tokens=4096, **kw)
    brain.llm = llm
    return brain


def run_turns(brain, first: int, last: int) -> list[dict]:
    out = []
    for i in range(first, last + 1):
        brain.llm.queue.extend(turn_script(i))
        out.append(brain.send(wake(i)))
    return out


def record_requests(turns=12, **kw) -> list[list[dict]]:
    brain = make(**kw)
    run_turns(brain, 1, turns)
    return [r["messages"] for r in brain.llm.turn_requests()]


def last_user(brain) -> str:
    return [m for m in brain.llm.turn_requests()[-1]["messages"] if m["role"] == "user"][-1]["content"]


# ---- §9.9 关着时逐字照旧 ----
def test_disabled_matches_baseline():
    baseline = json.loads((DATA / "toolloop_baseline.json").read_text(encoding="utf-8"))
    assert record_requests(compact=None) == baseline
    assert record_requests(compact=None, inbox=lambda: "- 新的\n") == baseline   # 关着：不读 inbox


def _wake_index(messages: list[dict]) -> int:
    return max(i for i, m in enumerate(messages) if m["role"] == "user")


def assert_append_only(reqs: list[list[dict]]) -> None:
    for a, b in zip(reqs, reqs[1:]):
        ia, ib = _wake_index(a), _wake_index(b)
        if a[ia] == b[ib] and ia == ib:  # 同一轮：上一次请求整个是前缀
            assert b[:len(a)] == a
        else:  # 跨轮：上一轮的历史部分（唤醒消息之前）是前缀
            assert b[:ia] == a[:ia]


# ---- §9.1 只往后接 ----
def test_append_only_prefix():
    reqs = record_requests(compact=CompactConfig(budget=10**9))
    assert_append_only(reqs)
    assert len(reqs[-1]) > 24                                      # 不再按 history=8 / HISTORY_CHARS 截
    assert reqs[-1][0] == {"role": "system", "content": "系统提示词"} and reqs[-1][1]["content"].startswith(stamp(1))
    hist5 = next(m["content"] for m in reqs[-1] if m["role"] == "user" and m["content"].startswith(stamp(5)))
    assert hist5.endswith("……（后面截掉了）") and len(hist5) < 1600   # 历史里的唤醒消息照旧截


def test_append_result_keys():
    out = run_turns(make(compact=CompactConfig(budget=10**9)), 1, 1)[0]
    assert out["history_mode"] == "append" and out["recap"] == 0
    assert "history_mode" not in run_turns(make(compact=None), 1, 1)[0]


# ---- §9.7 新记的 inbox ----
def test_inbox_new_lines_once():
    inbox = ["- 旧的\n"]
    brain = make(compact=CompactConfig(), inbox=lambda: inbox[0])
    run_turns(brain, 1, 1)
    assert "你刚记下" not in last_user(brain)
    inbox[0] += "- 小明下周考试\n"
    run_turns(brain, 2, 2)
    assert last_user(brain).endswith("\n\n你刚记下：\n- 小明下周考试")
    run_turns(brain, 3, 3)
    assert "你刚记下" not in last_user(brain)
    hist2 = next(m["content"] for m in brain.llm.turn_requests()[-1]["messages"] if m["role"] == "user" and m["content"].startswith(stamp(2)))
    assert hist2.endswith("你刚记下：\n- 小明下周考试")              # 跟着进历史


def test_inbox_read_error_skips():
    calls = []

    def read():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("坏了")
        return ""

    brain = make(compact=CompactConfig(), inbox=read)
    run_turns(brain, 1, 1)
    assert "你刚记下" not in last_user(brain) and len(calls) == 2


# ---- §2 §4 压缩 ----
class FakeMeter:
    def __init__(self):
        self.calls = []

    def record(self, use, provider, model, *, backup, usage, ok):
        self.calls.append((use, provider, model, backup, usage, ok))


def make_c(budget=1000, keep=2, usage=1000, retry=300.0, compact=(), now=None, **kw):
    """budget / keep 小一点的压缩大脑；spawn 只把压缩任务存进 brain.jobs，测试自己决定什么时候跑。"""
    jobs: list = []
    now = now if now is not None else [0.0]
    brain = make(FakeLLM(usage=usage, compact=compact),
                 compact=CompactConfig(budget=budget, keep_turns=keep, recap_max=100, retry=retry),
                 spawn=jobs.append, clock=lambda: now[0], **kw)
    brain.jobs, brain.now = jobs, now
    return brain


def reqs_of(brain, i: int) -> list[dict]:
    return [r for r in brain.llm.turn_requests() if r["messages"][_wake_index(r["messages"])]["content"].startswith(stamp(i))]


def users(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "user"]


def test_no_compact_below_budget():                       # §9.2
    brain = make_c(usage=999)
    run_turns(brain, 1, 10)
    assert brain.jobs == [] and brain.llm.compact_requests() == []


def test_no_compact_without_usage():                      # 取不到 usage 的那一轮不判
    brain = make_c(usage=None)
    run_turns(brain, 1, 6)
    assert brain.jobs == []


def test_compact_request_shape():                         # §9.2
    brain = make_c()
    run_turns(brain, 1, 2)
    assert brain.jobs == []                               # 没比 keep_turns 多
    run_turns(brain, 3, 3)
    assert len(brain.jobs) == 1
    brain.llm.usage = 1500
    run_turns(brain, 4, 4)
    assert len(brain.jobs) == 1                           # 在跑：不起第二个
    brain.jobs[0]()
    req = brain.llm.compact_requests()[0]
    last = reqs_of(brain, 3)[-1]["messages"]
    assert req["messages"][:-2] == last
    assert req["messages"][-2] == {"role": "assistant", "content": "说完了3"}
    assert f"{stamp(2)} 那条消息之前" in req["messages"][-1]["content"]
    assert req["tool_choice"] == "none" and req["tools"] == [SAY] and req["temperature"] == 0.3
    assert req["max_tokens"] == 4096 and req["timeout"] == 120.0 and req["model"] == "deepseek-flash"


def test_swap_in_recap_keeps_turns_added_meanwhile():     # §9.3 + Review Focus 1
    brain = make_c()
    run_turns(brain, 1, 3)
    run_turns(brain, 4, 4)                                # 压缩期间接上的一轮
    brain.jobs[0]()
    assert brain.recap == "" and brain.compactions == 0   # 下一次 send 开头才换上
    out = run_turns(brain, 5, 5)[0]
    m = reqs_of(brain, 5)[0]["messages"]
    assert m[1]["content"] == "（这次上线到现在的前情提要，第 1 次整理，写到 21:02 为止；之后的原话在后面）\n前情提要正文"
    assert m[2] == {"role": "assistant", "content": "（知道了）"}
    assert [u[:len(stamp(2))] for u in users(m)[1:]] == [stamp(2), stamp(3), stamp(4), stamp(5)]
    assert brain.recap == "前情提要正文" and brain.compactions == 1 and out["history_mode"] == "append" and out["recap"] == 1
    run_turns(brain, 6, 6)                                # 之后又只往后接
    assert_append_only([r["messages"] for r in reqs_of(brain, 5) + reqs_of(brain, 6)])


def test_second_compact_includes_old_recap():             # §9.4
    brain = make_c(compact=[reply("第一份"), reply("第二份")])
    run_turns(brain, 1, 3)
    brain.jobs[0]()
    run_turns(brain, 4, 4)                                # 换上第一份；这一轮结束又起第二次
    assert len(brain.jobs) == 2
    brain.jobs[1]()
    second = brain.llm.compact_requests()[1]["messages"]
    assert second[1]["content"].startswith("（这次上线到现在的前情提要，第 1 次整理") and second[1]["content"].endswith("第一份")
    run_turns(brain, 5, 5)
    m = reqs_of(brain, 5)[0]["messages"]
    assert m[1]["content"].startswith("（这次上线到现在的前情提要，第 2 次整理") and m[1]["content"].endswith("第二份")
    assert brain.recap == "第二份" and brain.compactions == 2 and "第一份" not in json.dumps(m, ensure_ascii=False)


@pytest.mark.parametrize("bad", [ModelError("余额用完", down="limit"), reply(""), reply(None, [("say", "{}")])])
def test_compact_failure_slides_then_retries(bad):        # §9.5
    brain = make_c(compact=[bad])
    run_turns(brain, 1, 3)
    brain.jobs[0]()
    out = run_turns(brain, 4, 4)[0]
    assert out["history_mode"] == "sliding" and brain.sliding
    assert reqs_of(brain, 4)[0]["messages"] == record_requests(compact=None, turns=4)[-1]   # 和关着时一样的历史
    run_turns(brain, 5, 6)
    assert len(brain.jobs) == 1                           # retry 秒内不再试
    brain.now[0] = 301.0
    run_turns(brain, 7, 7)
    assert len(brain.jobs) == 2                           # 冷却过了：滑动里不看 prompt_tokens 也再试
    brain.jobs[1]()
    src = brain.llm.compact_requests()[-1]["messages"]
    assert [u[:len(stamp(1))] for u in users(src)[:-1]] == [stamp(i) for i in range(1, 8)]   # 用完整历史现拼
    out = run_turns(brain, 8, 8)[0]
    assert out["history_mode"] == "append" and brain.compactions == 1 and not brain.sliding


def test_stuck_compact_slides_at_double_budget():         # §9.6
    brain = make_c()
    run_turns(brain, 1, 3)
    brain.llm.usage = 2000
    out = run_turns(brain, 4, 4)[0]
    assert out["history_mode"] == "append" and brain.sliding
    assert run_turns(brain, 5, 5)[0]["history_mode"] == "sliding" and len(brain.jobs) == 1
    brain.jobs[0]()
    assert run_turns(brain, 6, 6)[0]["history_mode"] == "append"


def test_compact_meter_backup():                          # §9.8
    meter = FakeMeter()
    brain = make_c(meter=meter, backup=True, compact=[reply("好的提要"), ModelError("坏了")])
    run_turns(brain, 1, 3)
    brain.jobs[0]()
    assert meter.calls == [("recap", "deepseek", "deepseek-flash", True,
                            {"input_tokens": 1000, "output_tokens": 10, "cache_read_input_tokens": 900}, True)]
    run_turns(brain, 4, 4)
    brain.jobs[1]()
    assert meter.calls[-1][-1] is False and meter.calls[-1][0] == "recap"


def test_compact_skips_dangling_tool_calls():             # Review Focus 2：轮数到顶，最后那条 assistant 只留文字
    brain = make_c(max_steps=0)
    run_turns(brain, 1, 2)
    brain.llm.queue.extend([reply("先说", [("say", '{"text":"a"}')])] * 3)
    brain.send(wake(3))
    brain.jobs[0]()
    tail = brain.llm.compact_requests()[0]["messages"][-2]
    assert tail == {"role": "assistant", "content": "先说"}


def test_turn_error_not_recorded():                       # Review Focus 3
    brain = make_c()
    run_turns(brain, 1, 2)
    brain.llm.queue.append(RuntimeError("断了"))
    with pytest.raises(ModelError):
        brain.send(wake(3))
    assert len(brain._past) == 2 and brain.jobs == []


def test_recap_jsonl_and_note(tmp_path):
    notes: list = []
    path = tmp_path / "recap.jsonl"
    brain = make_c(compact=[reply("好的提要"), reply("")], recap_path=path, on_note=notes.append)
    run_turns(brain, 1, 3)
    brain.jobs[0]()
    run_turns(brain, 4, 4)
    assert notes[0].startswith("── 压缩：第 1 次，压掉 1 轮 → 前情提要 4 字，用时 0 秒 ──") and notes[0].endswith("好的提要")
    brain.jobs[1]()
    run_turns(brain, 5, 5)
    assert notes[1].startswith("── 压缩失败：")
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["n"] == 1 and lines[0]["turns"] == 1 and lines[0]["chars"] == 4 and lines[0]["text"] == "好的提要"
    assert lines[0]["usage"]["input_tokens"] == 1000 and "time" in lines[0] and "seconds" in lines[0]
    assert lines[1]["ok"] is False and lines[1]["error"]
