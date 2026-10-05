"""离线评估 addressee label / eval：合成 agent.log + 假 Claude，只用占位人名。"""
import json
from datetime import datetime

import pytest

from skydango.brain import claude
from skydango.brain.addressee import Verdict
from skydango.chat import addressee_eval as ev
from skydango.config import AddresseeConfig

FRIENDS = ["小明", "阿花", "小红"]
ALIASES = {n: [n] for n in FRIENDS}


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


LOG = """\
2026-10-04 21:00:00,100 INFO skydango.agent: 启动
2026-10-04 21:00:10,200 INFO skydango.brain.body: 读到: 小明：你觉得呢
2026-10-04 21:00:10,300 INFO skydango.brain.body: 跟谁说 小明「你觉得呢」→ 拿不准（）· 身边：小明、阿花
2026-10-04 21:00:20,000 INFO skydango.chat.sender: 已发送: 我觉得可以
2026-10-04 21:00:21,000 INFO skydango.brain.body: 工具 say {} → 已发送：我觉得可以
2026-10-04 21:00:30,000 INFO skydango.brain.body: [dry-run] 将会发送: 好呀
2026-10-04 21:00:40,000 INFO skydango.brain.body: 读到: 没有说话人的一行
2026-10-04 21:00:41,000 DEBUG skydango.brain.body: 读到: 阿花：忽略调试行不算
"""


def test_read_log(tmp_path):
    p = tmp_path / "agent.log"
    p.write_text(LOG, encoding="utf-8")
    lines = ev.read_log(p, "r1")
    assert len(lines) == 4  # 工具行、DEBUG 行、启动行都不算
    assert [l.id for l in lines] == [f"r1#{i}" for i in range(4)]
    first = lines[0]
    assert (first.who, first.text) == ("小明", "你觉得呢")
    assert first.nearby == ["小明", "阿花"]
    assert first.t == pytest.approx(_ts("2026-10-04 21:00:10,200"))
    assert (lines[1].who, lines[1].text) == ("我", "我觉得可以")
    assert (lines[2].who, lines[2].text) == ("我", "好呀")
    assert (lines[3].who, lines[3].text) == ("", "没有说话人的一行")
    assert lines[3].nearby is None


def test_read_log_empty_nearby(tmp_path):
    p = tmp_path / "agent.log"
    p.write_text(
        "2026-10-04 21:00:10,200 INFO skydango.brain.body: 读到: 小明：嗨\n"
        "2026-10-04 21:00:10,300 INFO skydango.brain.body: 跟谁说 小明「嗨」→ 拿不准（）· 身边：没人\n",
        encoding="utf-8",
    )
    assert ev.read_log(p, "r")[0].nearby == []


def _line(i, t, who, text="嗯", nearby=None, run="r"):
    return ev.Line(id=f"{run}#{i}", run=run, t=t, who=who, text=text, nearby=nearby)


def test_pick_multi_and_control():
    lines = [_line(0, 0, "小明"), _line(1, 60, "阿花")]  # 60 秒内两个人都说过 → 都选
    assert ev.pick(lines, FRIENDS) == ["r#0", "r#1"]
    solo = [_line(i, 1000 + i * 400, "小明") for i in range(10)]  # 单人、互相隔得远 → 每 5 句取 1
    assert ev.pick(solo, FRIENDS) == ["r#0", "r#5"]


def test_replay_uses_logged_nearby():
    cfg = AddresseeConfig()
    # 记录里身边有两个人：「身边只有他」不成立 → unsure；没记录时按 5 分钟内说过话的估 → 只有小明 → me
    logged = [_line(0, 100, "小明", "今天好热", nearby=["小明", "阿花"])]
    guessed = [_line(0, 100, "小明", "今天好热", nearby=None)]
    assert ev.replay(logged, cfg, ["团子"], 30.0, ALIASES)["r#0"].label == "unsure"
    assert ev.replay(guessed, cfg, ["团子"], 30.0, ALIASES)["r#0"].label == "me"


def test_replay_said_resets():
    cfg = AddresseeConfig()
    lines = [_line(0, 100, "我", "你好"), _line(1, 110, "小明", "哈哈", nearby=["小明", "阿花"])]
    assert ev.replay(lines, cfg, ["团子"], 30.0, ALIASES)["r#1"].label == "me"  # 团子刚说完，接话


def _fake(calls, drop=()):
    def run(content):
        calls.append(content)
        text = content[0]["text"]
        ids = [ln[1 : ln.index("]")] for ln in text.splitlines() if ln.startswith("[r#")]
        return {"result": json.dumps({i: "跟团子" for i in ids if i not in drop}, ensure_ascii=False)}

    return run


def test_claude_labels_cache(tmp_path):
    lines = [_line(i, i * 10, "小明", f"句{i}") for i in range(5)]
    cache = tmp_path / "claude.jsonl"
    calls = []
    got = ev.claude_labels(lines[:3], lines, _fake(calls), cache, batch=2)
    assert set(got) == {"r#0", "r#1", "r#2"} and len(calls) == 2
    calls.clear()
    again = ev.claude_labels(lines[:3], lines, _fake(calls), cache, batch=2)
    assert again == got and calls == []
    # 回答缺 id：不写缓存
    cache2 = tmp_path / "c2.jsonl"
    got2 = ev.claude_labels(lines[:2], lines, _fake([], drop={"r#1"}), cache2, batch=5)
    assert set(got2) == {"r#0"}
    rows = [json.loads(x) for x in cache2.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["r#0"] and rows[0]["v"] == ev.PROMPT_VERSION


def test_claude_limit(tmp_path):
    lines = [_line(i, i * 10, "小明", f"句{i}") for i in range(4)]
    cache = tmp_path / "claude.jsonl"
    n = {"k": 0}
    ok = _fake([])

    def run(content):
        n["k"] += 1
        if n["k"] == 2:
            raise claude.ClaudeError("limit", limit=True)
        return ok(content)

    with pytest.raises(SystemExit) as e:
        ev.claude_labels(lines, lines, run, cache, batch=2)
    assert "额度" in str(e.value)
    rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
    assert {r["id"] for r in rows} == {"r#0", "r#1"}


def test_review_roundtrip(tmp_path):
    lines = [_line(i, i * 10, "小明", f"句{i}") for i in range(30)]
    rule = {l.id: Verdict("me", "") for l in lines}
    cl = {l.id: "跟团子" for l in lines}
    cl["r#3"] = "跟别人"  # 唯一不一致
    path = tmp_path / "review.md"
    n = ev.write_review(path, lines, lines, rule, cl, seed=0)
    assert 1 <= n < 30
    text = path.read_text(encoding="utf-8")
    assert "### r#3\n" in text
    assert ev.read_review(path) == {}
    j = text.index("标：", text.index("### r#3\n"))
    path.write_text(text[: j + 2] + "跟别人" + text[j + 2 :], encoding="utf-8")
    assert ev.read_review(path) == {"r#3": "跟别人"}
    ev.write_review(path, lines, lines, rule, cl, seed=0)  # 再写：已填的保留
    assert ev.read_review(path) == {"r#3": "跟别人"}


def test_review_bad_value(tmp_path):
    path = tmp_path / "review.md"
    path.write_text("### r#7\n> 小明：嗯\n标：随便\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        ev.read_review(path)
    assert "r#7" in str(e.value)


def _eval(n_me_wrong, n_unsure=0, total_other=20, with_me=True):
    lines, rule, human = [], {}, {}
    for i in range(total_other):
        l = _line(i, i * 10, "小明")
        lines.append(l)
        rule[l.id] = Verdict("me" if i < n_me_wrong else "other", "")
        human[l.id] = "跟别人"
    for k in range(2 if with_me else 0):  # 「跟团子」的标准答案，规则判对
        l = _line(500 + k, 5000 + k * 10, "小明")
        lines.append(l)
        rule[l.id] = Verdict("me", "")
        human[l.id] = "跟团子"
    for k in range(n_unsure):
        l = _line(1000 + k, 9000 + k, "小明")
        lines.append(l)
        rule[l.id] = Verdict("unsure", "")
    return ev.evaluate(lines, [l.id for l in lines], rule, {}, human)


def test_evaluate_thresholds():
    assert _eval(1).passed  # 1/20 = 5%
    r = _eval(2)  # 10% > 5%
    assert not r.passed
    assert "跟别人" in r.markdown()
    assert not _eval(0, n_unsure=16).passed  # 16/38 = 42%
    assert _eval(0, n_unsure=10).passed  # 10/32 = 31%


def test_evaluate_gold_from_agreement_and_unreviewed():
    a, b, c = _line(0, 0, "小明"), _line(1, 10, "小明"), _line(2, 20, "小明")
    rule = {a.id: Verdict("me", ""), b.id: Verdict("other", ""), c.id: Verdict("me", "")}
    cl = {a.id: "跟团子", b.id: "跟团子", c.id: "看不出"}
    r = ev.evaluate([a, b, c], [a.id, b.id, c.id], rule, cl, {})
    assert "没核对" in r.markdown()
    assert r.confusion["跟团子"]["me"] == 1  # 只有 a 两边一致（b 不一致没核对，c 是看不出）
    assert sum(sum(v.values()) for v in r.confusion.values()) == 1


def test_evaluate_no_gold_not_passed():
    r = ev.evaluate([], [], {}, {}, {})
    assert not r.passed and "样本不足" in r.markdown()
    r2 = _eval(0, with_me=False)  # 只有「跟别人」的标准答案，没有「跟团子」的
    assert not r2.passed and "样本不足" in r2.markdown()


def test_review_keeps_filled_without_claude_label(tmp_path):
    lines = [_line(i, i * 10, "小明", f"句{i}") for i in range(4)]
    rule = {l.id: Verdict("me", "") for l in lines}
    cl = {l.id: "跟别人" for l in lines}
    path = tmp_path / "review.md"
    ev.write_review(path, lines, lines, rule, cl)
    text = path.read_text(encoding="utf-8")
    for i in ("r#1", "r#2"):
        j = text.index("标：", text.index(f"### {i}\n"))
        text = text[: j + 2] + "大家" + text[j + 2 :]
    path.write_text(text, encoding="utf-8")
    # r#1 的 Claude 标签没了，r#2 这次根本不在 items 里
    ev.write_review(path, [lines[0], lines[1]], lines, rule, {"r#0": "跟别人"})
    assert ev.read_review(path) == {"r#1": "大家", "r#2": "大家"}
    assert "Claude：没标" in path.read_text(encoding="utf-8")


def test_report_by_reason_and_old_log_note():
    ls, rule, human = [], {}, {}
    for i, (name, gold) in enumerate([("阿花", "跟别人"), ("小红", "跟别人")]):
        l = _line(i, i * 10, "小明")  # nearby None = 旧日志
        ls.append(l)
        rule[l.id] = Verdict("other", f"叫了{name}", name)
        human[l.id] = gold
    l = _line(5, 100, "小明")
    ls.append(l)
    rule[l.id] = Verdict("me", "在接你的话")
    human[l.id] = "跟别人"
    r = ev.evaluate(ls, [x.id for x in ls], rule, {}, human)
    text = r.markdown()
    assert "## 按规则分" in text
    assert "叫了X" in text and "叫了阿花" not in text.split("## 按规则分")[1].split("## 判错")[0]
    assert r.by_reason["叫了X"]["跟别人"]["other"] == 2
    assert r.by_reason["在接你的话"]["跟别人"]["me"] == 1
    assert "旧日志上的门槛只作参考" in text and "句句都接" in text
    # 新日志（有「跟谁说」记录 = nearby 不是 None）不写这行
    for x in ls:
        x.nearby = ["小明"]
    r2 = ev.evaluate(ls, [x.id for x in ls], rule, {}, human)
    assert "旧日志上的门槛" not in r2.markdown()
