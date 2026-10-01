"""管理面板的内心页数据（spec 2026-09-30-inner-viewer §2 §5）：读 memory/inner/ 的文件、在跑时合并实时、删性格条目。"""

import json
import os

import pytest
from test_console_server import GOOD, free_port, make_server, request, upstream  # noqa: F401  (upstream 是 fixture)

from skydango.console.inner_view import forget_offline, inner_state
from skydango.inner.ledger import Card
from skydango.inner.log import MindLog, read
from skydango.inner.mind import Mind, Mood
from skydango.inner.persona import Persona, Trait
from skydango.inner.store import InnerStore

NOW = 1_790_000_000.0
FRIENDS = ["小明", "阿花"]


def fill(d, persona=True, mind=True):
    st = InnerStore(d)
    st.write_people({"路人甲": Card(first_met=NOW - 86400, lines=3), "阿花": Card(first_met=NOW - 5 * 86400, days=["a", "b"], visits=4,
                                                                            last_seen=NOW - 60, lines=10, to_me=6)}, None)
    if mind:
        st.write_mind(Mind(mood=Mood("开心", "有人陪")))
    if persona:
        st.write_persona(Persona(catchphrases=[Trait("害", since=NOW)], opinions=[Trait("丑", topic="雨林", since=NOW)]))
    st.append_diary("今天不错。", NOW - 3600)
    log = MindLog(d / "mind_log.jsonl", persist=True)
    log.energy(NOW - 10 * 86400, None)  # 7 天前的：不给
    log.energy(NOW - 600, None)
    log.reflect(NOW - 300, False, Mind(), None, ["心情：好"], [])
    return st


def live_snap(**kw):
    snap = {"running": True, "at": NOW, "mood": {"level": "烦", "text": "吵", "since": NOW}, "energy": {"level": "困", "score": 20, "note": "困"},
            "grudge": None, "wants": [], "soft": [{"who": "阿花", "text": "难过", "until": NOW + 60}],
            "persona": {"catchphrases": [], "jokes": [], "opinions": []},
            "log": [{"t": NOW - 300, "kind": "reflect", "changes": ["心情：好"]}, {"t": NOW - 5, "kind": "energy", "level": "困", "score": 20}]}
    snap.update(kw)
    return snap


def test_files_only(tmp_path):
    fill(tmp_path)
    s = inner_state(tmp_path, FRIENDS, "idle", lambda: pytest.fail("没在跑不该取实时"), NOW)
    assert s["source"] == "files" and s["saved_at"] is not None
    assert s["now"]["mood"]["level"] == "开心" and s["now"]["soft"] == [] and s["now"]["grudge"] is None
    assert [c["name"] for c in s["cards"]] == ["阿花", "路人甲"]  # 好友名单里的在前
    assert s["cards"][0] == {"name": "阿花", "friend": True, "first_met": NOW - 5 * 86400, "days": 2, "visits": 4,
                             "last_seen": NOW - 60, "said": 10, "to_me": 6, "outfits": []}
    assert s["cards"][1]["friend"] is False
    assert [r["kind"] for r in s["log"]] == ["energy", "reflect"]  # 7 天前的不给
    assert s["persona"]["catchphrases"][0]["text"] == "害"
    assert len(s["days"]) == 0 and s["diaries"] and "今天不错" in s["diaries"][0]


def test_files_missing_everything(tmp_path):
    s = inner_state(tmp_path / "nope", FRIENDS, "idle", lambda: None, NOW)
    assert s["source"] == "files" and s["persona"] is None and s["now"]["mood"] is None and s["saved_at"] is None
    assert s["cards"] == [] and s["log"] == [] and s["days"] == [] and s["diaries"] == []


def test_persona_missing_is_none(tmp_path):
    fill(tmp_path, persona=False)
    assert inner_state(tmp_path, FRIENDS, "idle", lambda: None, NOW)["persona"] is None


def test_running_merges_live(tmp_path):
    fill(tmp_path)
    s = inner_state(tmp_path, FRIENDS, "running", live_snap, NOW)
    assert s["source"] == "live" and s["now"]["mood"]["level"] == "烦" and s["now"]["energy"]["score"] == 20
    assert s["now"]["soft"][0]["who"] == "阿花" and s["persona"]["catchphrases"] == []  # 实时优先
    assert [(r["t"], r["kind"]) for r in s["log"]] == [(NOW - 600, "energy"), (NOW - 300, "reflect"), (NOW - 5, "energy")]  # 同一条只出现一次
    assert [c["name"] for c in s["cards"]] == ["阿花", "路人甲"]  # 关系卡照样来自文件


def test_running_live_without_reflect(tmp_path):  # Review Focus 5
    fill(tmp_path)
    s = inner_state(tmp_path, FRIENDS, "running", lambda: live_snap(mood=None, energy=None, persona=None, log=[], soft=[]), NOW)
    assert s["source"] == "live" and s["now"]["mood"] is None and s["persona"] is None
    assert s["cards"] and s["diaries"]


def test_running_but_live_unreachable(tmp_path):
    fill(tmp_path)
    s = inner_state(tmp_path, FRIENDS, "running", lambda: None, NOW)
    assert s["source"] == "files_fallback" and s["now"]["mood"]["level"] == "开心"


def test_bad_log_line_ok(tmp_path):  # Review Focus 2
    fill(tmp_path)
    with (tmp_path / "mind_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"t": 5, "kind": "ene')
    assert len(inner_state(tmp_path, FRIENDS, "idle", lambda: None, NOW)["log"]) == 2


def test_forget_offline(tmp_path):
    fill(tmp_path)
    assert forget_offline(tmp_path, {"kind": "catchphrase", "text": "害"}, NOW) == {"ok": True}
    assert json.loads((tmp_path / "persona.json").read_text("utf-8"))["catchphrases"] == []
    assert read(tmp_path / "mind_log.jsonl", 0)[-1] == {"t": NOW, "kind": "forget", "what": "catchphrase", "text": "害",
                                                        "who": "", "topic": ""}
    assert forget_offline(tmp_path, {"kind": "catchphrase", "text": "害"}, NOW) == {"ok": False, "error": "找不到这条（可能已经淡出了）"}
    assert forget_offline(tmp_path / "nope", {"kind": "opinion", "topic": "雨林"}, NOW)["ok"] is False
    assert not (tmp_path / "nope").exists()
    with pytest.raises(ValueError):
        forget_offline(tmp_path, {"kind": "xx"}, NOW)


# ---- ConsoleServer 的 /api/inner、/api/inner/forget ----
@pytest.fixture
def srv(tmp_path, upstream):  # noqa: F811
    mem = tmp_path / "memory"
    (tmp_path / "config.toml").write_text(f'[reply]\nmemory_dir = "{mem.as_posix()}"\nfriends = ["阿花"]\n', encoding="utf-8")
    fill(mem / "inner")
    s = make_server(tmp_path, upstream, child_port=free_port())  # 子进程端口上没人：不算有孤儿团子
    s.inner_dir = mem / "inner"
    yield s
    s.stop()


def fake_proxy(srv, replies):
    calls = []

    def proxy(method, rest, query, body=None):
        calls.append((method, rest, body))
        code, data = replies.get((method, rest), (503, {"ok": False, "text": "团子没在运行"}))
        return code, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False).encode()

    srv.proxy = proxy
    return calls


def test_api_inner_idle(srv):
    code, data = request(srv.url + "api/inner")
    assert code == 200 and data["source"] == "files" and data["now"]["mood"]["level"] == "开心"
    assert data["cards"][0]["name"] == "阿花" and data["cards"][0]["friend"] is True
    assert request(srv.url + "api/inner", headers={"Host": f"evil.com:{srv.port}"})[0] == 403


def test_api_inner_running(srv):
    srv.fake_runner.state = "running"
    fake_proxy(srv, {("GET", "inner"): (200, live_snap())})
    assert request(srv.url + "api/inner")[1]["source"] == "live"
    fake_proxy(srv, {})
    assert request(srv.url + "api/inner")[1]["source"] == "files_fallback"


def test_api_forget_idle_changes_file(srv):
    calls = fake_proxy(srv, {})
    body = json.dumps({"kind": "opinion", "topic": "雨林"}).encode()
    assert request(srv.url + "api/inner/forget", body, GOOD) == (200, {"ok": True})
    assert json.loads((srv.inner_dir / "persona.json").read_text("utf-8"))["opinions"] == [] and calls == []
    assert request(srv.url + "api/inner/forget", json.dumps({"kind": "xx"}).encode(), GOOD)[0] == 400
    assert request(srv.url + "api/inner/forget", body, {"Content-Type": "application/json"})[0] == 403


def test_api_forget_running_forwards(srv):
    srv.fake_runner.state = "running"
    calls = fake_proxy(srv, {("POST", "inner/forget"): (200, {"ok": True})})
    body = json.dumps({"kind": "catchphrase", "text": "害"}).encode()
    assert request(srv.url + "api/inner/forget", body, GOOD) == (200, {"ok": True})
    assert calls[0][:2] == ("POST", "inner/forget") and json.loads(calls[0][2]) == {"kind": "catchphrase", "text": "害"}
    assert json.loads((srv.inner_dir / "persona.json").read_text("utf-8"))["catchphrases"] != []  # 文件交给团子改


@pytest.mark.parametrize("state", ["starting", "stopping"])
def test_api_forget_refused_while_starting_or_stopping(srv, state):  # Review Focus 4
    srv.fake_runner.state = state
    calls = fake_proxy(srv, {})
    body = json.dumps({"kind": "catchphrase", "text": "害"}).encode()
    assert request(srv.url + "api/inner/forget", body, GOOD)[1] == {"ok": False, "error": "团子正在启动 / 停止，稍等再删"}
    assert calls == [] and json.loads((srv.inner_dir / "persona.json").read_text("utf-8"))["catchphrases"] != []


# ---- 终审修正 ----
def test_forget_offline_refused_when_someone_else_runs(tmp_path):  # I3：终端里跑着 run --live
    fill(tmp_path)
    cur = tmp_path / "current.json"
    cur.write_text("{}", encoding="utf-8")
    os.utime(cur, (NOW - 30, NOW - 30))
    r = forget_offline(tmp_path, {"kind": "catchphrase", "text": "害"}, NOW)
    assert r["ok"] is False and "在别处跑着" in r["error"]
    assert json.loads((tmp_path / "persona.json").read_text("utf-8"))["catchphrases"] != []
    os.utime(cur, (NOW - 3600, NOW - 3600))  # 上次被强杀留下的旧 current.json：不算
    assert forget_offline(tmp_path, {"kind": "catchphrase", "text": "害"}, NOW) == {"ok": True}


def test_api_forget_refused_with_orphan(tmp_path, upstream):  # noqa: F811  I3：上次留下的团子还占着端口
    mem = tmp_path / "memory"
    (tmp_path / "config.toml").write_text(f'[reply]\nmemory_dir = "{mem.as_posix()}"\n', encoding="utf-8")
    fill(mem / "inner")
    s = make_server(tmp_path, upstream)  # 子进程端口 = 假 viewer，/status 有响应
    try:
        code, data = request(s.url + "api/inner/forget", json.dumps({"kind": "catchphrase", "text": "害"}).encode(), GOOD)
        assert code == 409 and "上次留下的团子" in data["error"]
        assert json.loads((mem / "inner" / "persona.json").read_text("utf-8"))["catchphrases"] != []
    finally:
        s.stop()


def test_api_forget_dry_run_child_also_changes_file(srv):  # I4：dry-run 子进程只删内存，面板顺手改文件
    status = srv.fake_runner.status()
    srv.fake_runner.status = lambda: {**status, "state": "running", "options": {"brain": True, "live": False, "emotes": True, "duration": 0}}
    calls = fake_proxy(srv, {("POST", "inner/forget"): (200, {"ok": True})})
    body = json.dumps({"kind": "catchphrase", "text": "害"}).encode()
    assert request(srv.url + "api/inner/forget", body, GOOD) == (200, {"ok": True})
    assert calls and json.loads((srv.inner_dir / "persona.json").read_text("utf-8"))["catchphrases"] == []


def test_inner_view_shows_outfits(tmp_path):
    st = InnerStore(tmp_path)
    outs = [{"desc": f"套{i}", "feat": [0.1], "key": "k", "first": "2026-09-30", "last": "2026-10-01"} for i in range(4)]
    st.write_people({"阿花": Card(first_met=NOW, outfits=outs)}, None)
    s = inner_state(tmp_path, FRIENDS, "idle", lambda: None, NOW)
    got = s["cards"][0]["outfits"]
    assert [o["desc"] for o in got] == ["套1", "套2", "套3"]  # 最近 3 套，不带特征向量
    assert got[0] == {"desc": "套1", "first": "2026-09-30", "last": "2026-10-01"}
