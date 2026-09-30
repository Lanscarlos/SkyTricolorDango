"""沙盒剧本格式：读、校验、写（brain-sandbox 计划 Task 9，spec §6）。"""

from __future__ import annotations

import pytest

from skydango.console.scenario import Scenario, ScenarioError, Start, Step, dumps, load, to_op

# spec §6 的"放鸽子"示例
EXAMPLE = '''
name = "放鸽子"
note = "说好来又不来，看会不会闹别扭；认真说难过后别扭要撤"

[start]
memory = "reset"          # reset = 先从 memory/ 重置；keep = 用沙盒现在的记忆
time = "20:00"
nearby = ["小明"]
strangers = 0
place = "云野"
scene = "云野，傍晚，篝火旁"

[[steps]]
who = "小明"
say = "我 9 点再来找你玩"
[[steps]]
leave = "小明"
[[steps]]
skip = "2h"
[[steps]]
come = "小明"
[[steps]]
reflect = true
[[steps]]
who = "小明"
say = "对不起……我今天真的很难过"
expect = "别扭当场撤掉"
[[steps]]
offline = true
[[steps]]
online = "sleep"
'''


def write(tmp_path, text, name="剧本.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_example_loads(tmp_path):
    s = load(write(tmp_path, EXAMPLE))
    assert s.name == "放鸽子" and s.note.startswith("说好来又不来")
    assert s.start == Start(memory="reset", time="20:00", nearby=["小明"], strangers=0, place="云野", scene="云野，傍晚，篝火旁")
    assert [st.action for st in s.steps] == ["say", "leave", "skip", "come", "reflect", "say", "offline", "online"]
    assert s.steps[0] == Step("say", "我 9 点再来找你玩", who="小明")
    assert s.steps[5].expect == "别扭当场撤掉"
    assert s.steps[7].value == "sleep"


def test_roundtrip(tmp_path):
    s = load(write(tmp_path, EXAMPLE))
    again = load(write(tmp_path, dumps(s), "again.toml"))
    assert again == s


def test_roundtrip_escapes_and_wait(tmp_path):
    s = Scenario(
        name="引号",
        note='他说"嗨"\n第二行\\',
        start=Start(memory="keep", time="", nearby=["阿花", "小明"], strangers=3, place="", scene="有\t制表符"),
        steps=[
            Step("say", '“真的吗”\x7f', who="阿花", wait=5.0),
            Step("notice", "天黑了", expect="主动开口"),
            Step("strangers", 2),
            Step("time", "2026-10-01 09:00"),
            Step("skip", "30s"),
            Step("place", "雨林"),
            Step("scene", "下雨"),
        ],
    )
    assert load(write(tmp_path, dumps(s))) == s


def test_default_name_from_file(tmp_path):
    s = load(write(tmp_path, '[[steps]]\ncome = "小明"\n', "深夜.toml"))
    assert s.name == "深夜" and s.note == "" and s.start == Start()


@pytest.mark.parametrize(
    "steps, n, words",
    [
        ('[[steps]]\ncome = "小明"\n[[steps]]\nfly = true\n', 2, "不认识"),
        ('[[steps]]\ncome = "小明"\nleave = "阿花"\n', 1, "两个动作"),
        ('[[steps]]\nexpect = "啥也没有"\n', 1, "没有动作"),
        ('[[steps]]\nsay = "在吗"\n', 1, "who"),
        ('[[steps]]\nwho = "小明"\ncome = "小明"\n', 1, "who"),
        ('[[steps]]\ncome = "小明"\n[[steps]]\nskip = "2d"\n', 2, "时长"),
        ('[[steps]]\ntime = "25:00"\n', 1, "时间"),
        ('[[steps]]\ntime = "明早"\n', 1, "时间"),
        ('[[steps]]\nonline = "sleep"\n', 1, "offline"),
        ('[[steps]]\noffline = true\n[[steps]]\ncome = "小明"\n', 1, "online"),
        ('[[steps]]\noffline = true\n[[steps]]\nonline = "明天"\n', 2, "时间"),
        ('[[steps]]\nstrangers = 99\n', 1, "陌生人"),
        ('[[steps]]\nstrangers = "两个"\n', 1, "陌生人"),
        ('[[steps]]\nreflect = false\n', 1, "true"),
        ('[[steps]]\ncome = ""\n', 1, "名字"),
        ('[[steps]]\ncome = "小明"\nwait = -1\n', 1, "wait"),
        ('[[steps]]\ncome = "小明"\nexpect = 3\n', 1, "expect"),
    ],
)
def test_step_errors_name_the_step(tmp_path, steps, n, words):
    with pytest.raises(ScenarioError) as e:
        load(write(tmp_path, steps))
    assert f"第 {n} 步" in str(e.value) and words in str(e.value)


def test_offline_at_end_ok_and_online_after_offline(tmp_path):
    s = load(write(tmp_path, '[[steps]]\noffline = true\n[[steps]]\nonline = "22:00"\n[[steps]]\noffline = true\n'))
    assert [st.action for st in s.steps] == ["offline", "online", "offline"]


def test_start_errors(tmp_path):
    for text in (
        '[start]\nmemory = "maybe"\n',
        '[start]\ntime = "明早"\n',
        '[start]\nnearby = "小明"\n',
        '[start]\nstrangers = -1\n',
        '[start]\nweather = "雨"\n',
        'title = "多了个键"\n',
        'name = "坏的\n',  # TOML 语法错
    ):
        with pytest.raises(ScenarioError):
            load(write(tmp_path, text))


def test_start_accepts_resume_and_sleep(tmp_path):
    assert load(write(tmp_path, '[start]\ntime = "sleep"\n')).start.time == "sleep"
    assert load(write(tmp_path, '[start]\ntime = "resume"\n')).start.time == "resume"


def test_wait_accepts_duration_text(tmp_path):
    s = load(write(tmp_path, '[[steps]]\ncome = "小明"\nwait = "1m"\n'))
    assert s.steps[0].wait == 60.0


def test_to_op():
    assert to_op(Step("say", "在吗", who="小明")) == {"op": "say", "who": "小明", "text": "在吗"}
    assert to_op(Step("come", "小明")) == {"op": "come", "who": "小明"}
    assert to_op(Step("leave", "小明")) == {"op": "leave", "who": "小明"}
    assert to_op(Step("strangers", 2)) == {"op": "strangers", "n": 2}
    assert to_op(Step("place", "雨林")) == {"op": "place", "name": "雨林"}
    assert to_op(Step("scene", "下雨")) == {"op": "scene", "text": "下雨"}
    assert to_op(Step("notice", "天黑了")) == {"op": "notice", "text": "天黑了"}
    assert to_op(Step("skip", "2h")) == {"op": "skip", "seconds": 7200.0}
    assert to_op(Step("skip", 90)) == {"op": "skip", "seconds": 90.0}
    assert to_op(Step("time", "23:30")) == {"op": "time", "at": "23:30"}
    assert to_op(Step("reflect", True)) == {"op": "reflect"}
    assert to_op(Step("offline", True)) is None
    assert to_op(Step("online", "sleep")) is None


def test_example_scenarios_in_docs_load():  # 沙盒计划 Task 11：docs/sandbox-scenarios/ 的示例都能读
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "docs" / "sandbox-scenarios"
    files = sorted(folder.glob("*.toml"))
    assert {p.stem for p in files} >= {"放鸽子", "深夜犯困", "第二天上线"}
    for path in files:
        s = load(path)
        assert s.name == path.stem and s.note and s.steps
        assert all(n in ("小明", "阿花") for n in s.start.nearby)  # 人名是占位的
    assert (folder / "README.md").is_file()
