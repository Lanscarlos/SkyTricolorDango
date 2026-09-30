"""身体接上内心流水账（spec 2026-09-30-inner-viewer §4）：反思记 changes、每 300 秒记精力、删性格条目、内心快照。"""

import json
import time

from conftest import FakeLlm
from test_brain_body import FakeEnv, body

from skydango.config import Config
from skydango.inner.ledger import Card, Ledger
from skydango.inner.log import MindLog
from skydango.inner.mind import Mind
from skydango.inner.persona import Persona, Trait
from skydango.inner.reflect import Reflector
from skydango.inner.store import InnerStore

FRIENDS = ["懒洋洋大王", "阿花"]
WALL = time.mktime((2026, 9, 30, 15, 0, 0, 0, 0, -1))


def make(clock, tmp_path, live=False, reflect_reply="{}", reflect=True, persona=None, **kw):
    env = FakeEnv()
    cfg = Config().inner
    cfg.persona = True
    led = Ledger(cfg, lambda: FRIENDS, WALL, store=InnerStore(tmp_path), persist=live,
                 cards={"懒洋洋大王": Card(first_met=0, days=["d1", "d2", "d3"])})
    reflector = Reflector(cfg, FakeLlm(reflect_reply), clock, threaded=False) if reflect else None
    b, *_ = body(clock, live=live, env=env, ledger=led, mind=Mind() if reflect else None, reflector=reflector,
                 persona=(persona if persona is not None else Persona()) if reflect else None,
                 mind_log=MindLog(tmp_path / "mind_log.jsonl", persist=live), wall=lambda: WALL, **kw)
    b.friend_names = lambda: FRIENDS
    return b


def test_reflection_logs_changes(clock, tmp_path):
    b = make(clock, tmp_path)
    b.apply_reflection({"mood": {"level": "开心", "text": "有人来聊天"}, "grudge": "keep"})
    r = b.mind_log.recent()[-1]
    assert r["kind"] == "reflect" and r["final"] is False and r["changes"] == ["心情 平常→开心（有人来聊天）"]
    assert r["t"] == WALL and r["mood"] == {"level": "开心", "text": "有人来聊天"}


def test_reflection_logs_persona_changes_and_dropped(clock, tmp_path):
    b = make(clock, tmp_path)
    b.apply_reflection({"persona_add": {"catchphrases": ["懒得动"], "jokes": [{"who": "阿花", "text": "梗"}]}})
    r = b.mind_log.recent()[-1]
    assert r["changes"] == ["新口头禅：懒得动"]
    assert any("阿花" in d for d in r["dropped"])  # 和阿花还不够熟


def test_energy_every_300s(clock, tmp_path):
    b = make(clock, tmp_path)
    for dt in (0, 299, 300):
        b.wall = lambda dt=dt: WALL + dt
        b._inner_tick()
    assert [r["kind"] for r in b.mind_log.recent()] == ["energy", "energy"]
    assert b.mind_log.recent()[0]["score"] is not None


def test_forget_live_writes_persona_and_logs(clock, tmp_path):
    b = make(clock, tmp_path, live=True, persona=Persona(catchphrases=[Trait("害")]))
    assert b.forget("catchphrase", "害") == ""
    assert b.persona.catchphrases == [] and json.loads((tmp_path / "persona.json").read_text("utf-8"))["catchphrases"] == []
    assert b.mind_log.recent()[-1]["kind"] == "forget"
    assert b.forget("catchphrase", "害") == "找不到这条（可能已经淡出了）"  # Review Focus 3


def test_forget_dry_run_no_file(clock, tmp_path):
    b = make(clock, tmp_path, persona=Persona(opinions=[Trait("丑", topic="雨林")]))
    assert b.forget("opinion", "", topic="雨林") == ""
    assert b.persona.opinions == [] and not (tmp_path / "persona.json").exists()
    assert b.mind_log.recent()[-1]["topic"] == "雨林"


def test_forget_bad_kind_and_no_persona(clock, tmp_path):
    b = make(clock, tmp_path)
    assert b.forget("xx", "害") == "不认识的类别"
    assert make(clock, tmp_path, reflect=False).forget("catchphrase", "害") == "性格档案没开"


def test_snapshot_without_reflect(clock, tmp_path):
    s = make(clock, tmp_path, reflect=False).inner_snapshot()
    assert s["mood"] is None and s["persona"] is None and s["log"] == [] and s["running"] is True
    assert s["energy"] is None and s["grudge"] is None and s["wants"] == [] and s["soft"] == []


def test_snapshot_with_reflect(clock, tmp_path):
    b = make(clock, tmp_path, persona=Persona(catchphrases=[Trait("害")]))
    b._inner_tick()
    b._soft_until["阿花"] = (WALL + 60, "好难过")
    s = b.inner_snapshot()
    assert s["at"] == WALL and s["mood"]["level"] == "平常" and s["energy"]["level"]
    assert s["persona"]["catchphrases"][0]["text"] == "害"
    assert s["soft"] == [{"who": "阿花", "text": "好难过", "until": WALL + 60}]
    assert s["log"][-1]["kind"] == "energy"
    json.dumps(s, ensure_ascii=False)  # 能直接给网页


def test_cli_inner_log_only_with_reflector(tmp_path):
    from skydango import cli

    led = Ledger(Config().inner, lambda: FRIENDS, WALL, store=InnerStore(tmp_path), persist=True)
    assert cli._inner_log(None, object()) is None and cli._inner_log(led, None) is None
    (tmp_path / "mind_log.jsonl").write_text('{"t": 1, "kind": "energy"}\n', encoding="utf-8")
    m = cli._inner_log(led, object())
    assert m.persist and m.path == tmp_path / "mind_log.jsonl"
    assert (tmp_path / "mind_log.jsonl").read_text(encoding="utf-8") == ""  # 启动时删掉 30 天前的
