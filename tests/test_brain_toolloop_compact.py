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
