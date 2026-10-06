import threading
import time

from conftest import FakeLlm

from skydango.brain.claude import ClaudeError
from skydango.config import InnerConfig
from skydango.inner.reflect import PERSONA_SYSTEM, REFLECT_SYSTEM, Reflector, friend_sections, materials

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


def test_bad_json_logs_full_text(clock, caplog):  # 2026-09-30 下线反思坏了，日志只有前 120 字，看不出坏在哪
    raw = '{"mood": {"level": "开心", "text": "好"}, "diary": "' + "今天玩得很开心" * 20 + '他说"来找我'  # 截断了，修引号也救不回
    r = Reflector(CFG, FakeLlm(raw), clock, threaded=False)
    with caplog.at_level("DEBUG", logger="skydango.inner.reflect"):
        assert r.final("x") is None
    assert any(rec.levelname == "DEBUG" and "来找我" in rec.getMessage() for rec in caplog.records)


def test_system_prompt_asks_for_corner_quotes():
    assert "「」" in REFLECT_SYSTEM


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


FRIENDS_MD = """# 好友

## 懒洋洋大王
- 本名卡洛，可以叫卡洛

## 番茄炒蛋盖饭
- 好朋友，女生
- 卡洛喊她“老登”
"""


def test_friend_sections_picks_named_friends():  # 日记里把女生写成“他”：反思要看到好友名单里写的
    assert friend_sections(FRIENDS_MD, ["番茄炒蛋盖饭"]) == ["番茄炒蛋盖饭：好朋友，女生；卡洛喊她“老登”"]
    assert friend_sections(FRIENDS_MD, ["路人"]) == [] and friend_sections("", ["番茄炒蛋盖饭"]) == []


def test_materials_friend_profiles():
    text = materials(T0, "", "", [], [], [], [], "", final=True, profiles=["番茄炒蛋盖饭：好朋友，女生"])
    assert "好友名单里写的" in text and "番茄炒蛋盖饭：好朋友，女生" in text
    assert "好友名单里写的" not in materials(T0, "", "", [], [], [], [], "", final=True)
    assert "性别" in REFLECT_SYSTEM


def test_system_prompt_rules():
    for s in ("JSON", "mood", "grudge", "wants_add", "wants_done", "diary", "memos", "不编", "keep"):
        assert s in REFLECT_SYSTEM


# ---- 第 3 期：反思带上性格 ----
def test_reflector_uses_given_system(clock):
    llm = FakeLlm("{}")
    r = Reflector(CFG, llm, clock, threaded=False, system="系统X")
    r.stirred(clock()); clock.advance(1300); r.start("m", clock())
    assert llm.calls[0][0] == "系统X"


def test_persona_system_mentions_keys():
    for s in ("persona_add", "persona_used", "catchphrases", "jokes", "opinions", "外貌"):
        assert s in PERSONA_SYSTEM


def test_materials_traits_section():
    base = dict(now=T0, energy_note="", mind_line="", chat=[], comings=[], cards=[], notes=[], persona="", final=False)
    assert "你攒下的性格" not in materials(**base)  # Review Focus 5：不传和第 2 期一样
    assert materials(**base) == materials(**base, traits=None)
    assert "你攒下的性格：\n（还没有）" in materials(**base, traits="")
    text = materials(**base, traits="口头禅：害")
    assert "口头禅：害" in text and text.index("你攒下的性格") < text.index("相关的好友")


def test_persona_system_cheeky():  # 贱兮兮：攒带贱味的口头禅 / 老梗，损事的可以记，拿外貌这些开玩笑的照样不记
    from skydango.inner.reflect import persona_system

    assert persona_system(False) == PERSONA_SYSTEM
    text = persona_system(True)
    assert "损人的、" not in text and "外貌" in text and "贱" in text
    assert text.endswith('"persona_used": ["..."]')  # JSON 说明照旧在最后
