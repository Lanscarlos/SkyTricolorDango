from skydango.inner.energy import Energy
from skydango.inner.log import MindLog, diff, read
from skydango.inner.mind import Grudge, Mind, Mood, Want
from skydango.inner.persona import Persona, Trait


def mind(level="平常", text="", grudge=None, wants=()):
    return Mind(mood=Mood(level, text), grudge=grudge, wants=[Want("想做", w) for w in wants])


def test_diff_mood_level_and_text_only():
    assert diff(mind("平常", "还行"), mind("开心", "有人来聊天"), None, None) == ["心情 平常→开心（有人来聊天）"]
    assert diff(mind("平常", "还行"), mind("平常", "有点无聊"), None, None) == ["心情：有点无聊"]
    assert diff(mind("平常", "还行"), mind("平常", "还行"), None, None) == []


def test_diff_same_reflection_only_mood_text():
    """Review Focus 1：反思结果和上一次一样、只换了那句话 → 只有一行"心情：…"。"""
    g = Grudge("小明", "放鸽子", 0, 100)
    p = Persona(catchphrases=[Trait("害")], opinions=[Trait("丑", topic="雨林")], jokes=[Trait("梗", who="小明")])
    import copy
    before = mind("开心", "好玩", grudge=g, wants=["看日落", "听弹琴"])
    after = mind("开心", "真好玩", grudge=copy.deepcopy(g), wants=["看日落", "听弹琴"])
    assert diff(before, after, p, copy.deepcopy(p)) == ["心情：真好玩"]


def test_diff_grudge_and_wants():
    g = Grudge("小明", "放鸽子", 0, 100)
    assert diff(mind(), mind(grudge=g), None, None) == ["新别扭：小明（放鸽子）"]
    assert diff(mind(grudge=g), mind(), None, None) == ["别扭撤了：小明"]
    assert diff(mind(wants=["看日落"]), mind(wants=["听弹琴"]), None, None) == ["新心愿：听弹琴", "心愿了结：看日落"]


def test_diff_persona():
    before = Persona(catchphrases=[Trait("害")], opinions=[Trait("丑", topic="雨林")], jokes=[Trait("旧梗", who="阿花")])
    after = Persona(catchphrases=[Trait("害"), Trait("懒得动")], opinions=[Trait("其实还行", topic="雨林"), Trait("好看", topic="云野")],
                    jokes=[Trait("冥龙嘴里", who="小明")])
    assert diff(mind(), mind(), before, after) == [
        "新口头禅：懒得动", "新老梗：小明——冥龙嘴里", "新看法：云野——好看", "看法换了：雨林 丑→其实还行", "淡出：阿花——旧梗"]


def test_reflect_record_fields(tmp_path):
    log = MindLog(None, persist=False)
    g = Grudge("小明", "放鸽子", 0, 100)
    r = log.reflect(5.0, True, mind("开心", "好", grudge=g, wants=["看日落"]), Energy("精神", 90, "精神"), ["x"], ["y"])
    assert r == {"t": 5.0, "kind": "reflect", "final": True, "mood": {"level": "开心", "text": "好"},
                 "energy": {"level": "精神", "score": 90}, "grudge": {"who": "小明", "why": "放鸽子", "until": 100},
                 "wants": ["看日落"], "changes": ["x"], "dropped": ["y"]}
    assert log.energy(6.0, None)["level"] is None
    f = log.forget(7.0, "opinion", "丑", topic="雨林")
    assert f["kind"] == "forget" and f["what"] == "opinion" and f["topic"] == "雨林"


def test_log_persist_and_dry_run(tmp_path):
    p = tmp_path / "mind_log.jsonl"
    live, dry = MindLog(p, persist=True), MindLog(tmp_path / "x.jsonl", persist=False)
    live.energy(10.0, None); dry.energy(10.0, None)
    assert len(read(p, 0)) == 1 and not (tmp_path / "x.jsonl").exists() and len(dry.recent()) == 1


def test_memory_cap_and_trim_and_bad_lines(tmp_path):
    p = tmp_path / "mind_log.jsonl"
    log = MindLog(p, persist=True)
    for i in range(210):
        log.forget(float(i), "catchphrase", "害")
    assert len(log.recent()) == 200
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"t": 5, "kind": "ene')             # 被强杀写到一半
    assert len(read(p, 0)) == 210                       # 坏行跳过
    assert len(read(p, 100)) == 110                     # since：只要 t >= 100 的
    log.trim(now=30 * 86400 + 100, days=30)            # 30 天前（t < 100）的删掉
    rows = read(p, 0)
    assert rows[0]["t"] == 100.0 and len(rows) == 110
    assert not (tmp_path / "mind_log.jsonl.tmp").exists()


def test_read_missing_and_dry_trim(tmp_path):
    assert read(tmp_path / "nope.jsonl", 0) == []
    p = tmp_path / "mind_log.jsonl"
    p.write_text('{"t": 1, "kind": "energy"}\n', encoding="utf-8")
    MindLog(p, persist=False).trim(now=10**9)           # dry-run 不改文件
    assert len(read(p, 0)) == 1


def test_half_multibyte_char_skipped(tmp_path):  # 终审 I1：截在一个中文字符中间
    p = tmp_path / "mind_log.jsonl"
    good = '{"t": 100, "kind": "energy"}\n'.encode()
    half = '{"t": 200, "kind": "reflect", "changes": ["新心愿：想看日落"]}'.encode()[:-4]
    p.write_bytes(good + half)
    assert [r["t"] for r in read(p, 0)] == [100]
    log = MindLog(p, persist=True)
    log.trim(now=150)
    log.energy(300.0, None)
    assert [r["t"] for r in read(p, 0)] == [100, 300.0]  # trim 把坏行清掉了，新行照常追加


def test_append_after_half_line_starts_new_line(tmp_path):
    p = tmp_path / "mind_log.jsonl"
    p.write_text('{"t": 1, "kind": "energy"}\n{"t": 2, "kind": "ene', encoding="utf-8")
    MindLog(p, persist=True).energy(3.0, None)
    assert [r["t"] for r in read(p, 0)] == [1, 3.0]


def test_musing_row():  # 冷场时的心理活动 §3
    assert MindLog(None, False).musing(1.0, ["懒洋洋大王"], "silent", "嗯") == {
        "t": 1.0, "kind": "musing", "who": ["懒洋洋大王"], "lull": "silent", "text": "嗯",
    }
