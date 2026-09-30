import threading
import time

from conftest import FakeLlm

from skydango.brain.claude import ClaudeError
from skydango.config import InnerConfig
from skydango.inner.reflect import REFLECT_SYSTEM, Reflector, materials

CFG = InnerConfig()
T0 = time.mktime((2026, 9, 30, 20, 0, 0, 0, 0, -1))


def test_due_needs_activity_and_interval(clock):
    llm = FakeLlm('{"mood": {"level": "开心"}}')
    r = Reflector(CFG, llm, clock, threaded=False)
    clock.advance(1300)
    assert not r.due(clock())  # 没动静
    r.stirred(clock())
    assert r.due(clock())
    r.start("材料", clock())
    assert llm.calls == [(REFLECT_SYSTEM, "材料")]
    assert r.poll() == {"mood": {"level": "开心"}} and r.poll() is None and not r.due(clock())


def test_due_after_conversation_quiet(clock):
    r = Reflector(CFG, FakeLlm("{}"), clock, threaded=False)
    for _ in range(6):
        r.heard(True, clock())
        clock.advance(5)
    assert not r.due(clock())
    clock.advance(180)
    assert r.due(clock())
    r2 = Reflector(CFG, FakeLlm("{}"), clock, threaded=False)
    for _ in range(6):
        r2.heard(False, clock())  # 陌生人说的不算
    clock.advance(200)
    assert not r2.due(clock())


def test_failure_keeps_old_and_limit_pauses(clock):
    r = Reflector(CFG, FakeLlm(ClaudeError("额度", limit=True)), clock, threaded=False)
    r.stirred(clock())
    clock.advance(1300)
    r.start("材料", clock())
    assert r.poll() is None
    for _ in range(6):
        r.heard(True, clock())
    clock.advance(200)
    assert not r.due(clock())  # 本来聊完安静该反思了，额度用完 600 秒内不反思
    clock.advance(500)
    assert r.due(clock())


def test_other_errors_do_not_pause(clock):
    r = Reflector(CFG, FakeLlm(RuntimeError("坏了")), clock, threaded=False)
    r.stirred(clock())
    clock.advance(1300)
    r.start("x", clock())
    assert r.poll() is None
    for _ in range(6):
        r.heard(True, clock())
    clock.advance(200)
    assert r.due(clock())


def test_bad_json_is_none(clock):
    r = Reflector(CFG, FakeLlm("今天挺好"), clock, threaded=False)
    r.stirred(clock())
    clock.advance(1300)
    r.start("x", clock())
    assert r.poll() is None


def test_only_one_at_a_time(clock):
    gate = threading.Event()
    r = Reflector(CFG, FakeLlm("{}", wait=gate), clock)
    r.stirred(clock())
    clock.advance(1300)
    r.start("x", clock())
    for _ in range(6):
        r.heard(True, clock())
    clock.advance(9999)
    assert r.running and not r.due(clock())
    gate.set()
    deadline = time.monotonic() + 3
    while r.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not r.running and r.poll() == {}


def test_final_discards_background(clock):  # Review Focus 3
    gate = threading.Event()
    llm = FakeLlm('{"mood": {"level": "烦"}}', wait=gate)
    r = Reflector(CFG, llm, clock)
    r.stirred(clock())
    clock.advance(1300)
    r.start("x", clock())
    assert r.running
    llm.reply = '{"mood": {"level": "开心"}, "diary": "今天不错", "memos": ["小明考试过了"]}'
    gate.set()
    assert r.final("最终材料")["diary"] == "今天不错"
    time.sleep(0.2)
    assert r.poll() is None and not r.due(clock() + 99999)  # 后台那次回来了也丢掉；之后不再反思


def test_final_runs_even_when_paused(clock):
    llm = FakeLlm(ClaudeError("额度", limit=True))
    r = Reflector(CFG, llm, clock, threaded=False)
    r.stirred(clock())
    clock.advance(1300)
    r.start("x", clock())
    llm.reply = '{"diary": "嗯"}'
    assert r.final("最后") == {"diary": "嗯"}


def test_materials_mentions_everything():
    text = materials(T0, "有点困（半夜了）", "平常 · 有点困", [(T0, "小明", "在吗"), (T0 + 1, "我", "在")],
                     ["小明 来了"], ["小明（…）"], ["- 小明在准备考试"], "人设……", final=True)
    for s in ("2026年9月30日（周三） 20:00", "有点困（半夜了）", "平常 · 有点困", "小明：在吗", "我：在", "小明 来了",
              "小明（…）", "- 小明在准备考试", "人设……", "日记"):
        assert s in text
    assert "日记" not in materials(T0, "", "", [], [], [], [], "", final=False)


def test_materials_keeps_latest_80_lines():
    chat = [(T0 + i, "小明", f"第{i}句") for i in range(100)]
    text = materials(T0, "", "", chat, [], [], [], "", final=False)
    assert "第19句" not in text and "第20句" in text and "第99句" in text


def test_system_prompt_rules():
    for s in ("JSON", "mood", "grudge", "wants_add", "wants_done", "diary", "memos", "不编", "keep"):
        assert s in REFLECT_SYSTEM
