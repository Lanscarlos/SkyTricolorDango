"""沙盒剧本的录制、回放、报告（brain-sandbox 计划 Task 10，spec §6）：回放引擎用假的沙盒接口测。"""

from __future__ import annotations

import json
import math

import pytest
from test_console_sandbox import KindRunner, make, post  # noqa: F401
from test_console_server import no_registry, request, upstream  # noqa: F401

from skydango.console.replay import Recorder, Replayer
from skydango.console.scenario import Scenario, ScenarioError, Start, Step, dumps, load, parse


class FakeApi:
    """假沙盒：记下调了什么；state() 按脚本给 idle / 新行。"""

    def __init__(self, idle_after=0, never_idle=False, running=False):
        self.calls: list = []
        self.is_running = running
        self.idle_after = idle_after  # 每次 op 之后前几次 state 不 idle
        self.never_idle = never_idle
        self.version = 0
        self.rows: list[dict] = []
        self.pending = 0
        self.diary_text = "旧日记\n"
        self.on_state = None

    def _row(self, kind, text, who=""):
        self.version += 1
        self.rows.append({"seq": self.version, "t": 1790800000.0 + self.version * 60, "kind": kind, "who": who, "text": text})

    def start(self, choice):
        self.calls.append(("start", choice))
        self.is_running = True

    def stop(self):
        self.calls.append(("stop",))
        self.is_running = False
        self.diary_text += "\n9月30日：今天小明放了我鸽子\n"

    def reset(self):
        self.calls.append(("reset",))

    def running(self):
        return self.is_running

    def op(self, req):
        self.calls.append(("op", req))
        if req["op"] == "say":
            self._row("heard", req["text"], req["who"])
            self._row("said", "哼", "团子")
        elif req["op"] == "reflect":
            self._row("event", "── 反思：心情 平常→烦 ──")
        else:
            self._row("event", f"── {req['op']} ──")
        self.pending = self.idle_after
        return {"ok": True, "text": "好"}

    def state(self, after):
        self.calls.append(("state", after))
        if self.on_state:
            self.on_state()
        if not self.is_running:
            return None
        idle = not self.never_idle and self.pending <= 0
        self.pending -= 1
        return {"version": self.version, "lines": [r for r in self.rows if r["seq"] > after], "idle": idle}

    def inner(self):
        return {"mood": {"level": "烦", "text": "小明放鸽子"}, "energy": {"level": "困", "score": 20}, "grudge": None, "wants": []}

    def diary(self):
        return self.diary_text


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.05)


def scenario(*steps, start=None) -> Scenario:
    return Scenario("放鸽子", "说好来又不来", start or Start(memory="reset", time="20:00", nearby=["小明"], place="云野"), list(steps))


def replayer(api, sc, tmp_path, step_timeout=30.0):
    c = Clock()
    return Replayer(api, sc, tmp_path / "reports", step_timeout=step_timeout, stop_timeout=10.0, sleep=c.sleep, clock=c), c


def ops(api):
    return [c[1] for c in api.calls if c[0] == "op"]


def test_order_start_then_steps(tmp_path):
    api = FakeApi()
    sc = scenario(Step("say", "我 9 点再来", who="小明"), Step("leave", "小明"), Step("skip", "2h"), Step("reflect", True))
    r, _ = replayer(api, sc, tmp_path)
    path = r.run()
    assert [c for c in api.calls if c[0] in ("reset", "start")] == [("reset",), ("start", "20:00")]
    assert ops(api) == [
        {"op": "come", "who": "小明"}, {"op": "place", "name": "云野"},  # [start] 的身边 / 场景先发
        {"op": "say", "who": "小明", "text": "我 9 点再来"}, {"op": "leave", "who": "小明"}, {"op": "skip", "seconds": 7200.0},
        {"op": "reflect"},
    ]
    text = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "reports" and path.name.startswith("放鸽子-") and path.suffix == ".md"
    assert "# 剧本回放：放鸽子" in text and "说好来又不来" in text
    assert "第 1 步：小明说「我 9 点再来」" in text and "小明：我 9 点再来" in text and "团子：哼" in text
    assert "── 反思：心情 平常→烦 ──" in text
    assert "## 最后的心里" in text and "小明放鸽子" in text


def test_waits_for_idle_before_next_step(tmp_path):
    api = FakeApi(idle_after=3)
    sc = scenario(Step("say", "一", who="小明"), Step("say", "二", who="小明"), start=Start(memory="keep"))
    r, _ = replayer(api, sc, tmp_path)
    r.run()
    kinds = [c[0] if c[0] != "op" else c[1]["text"] for c in api.calls if c[0] in ("op", "state")]
    i1, i2 = kinds.index("一"), kinds.index("二")
    assert kinds[i1 + 1:i2].count("state") >= 4  # 前 3 次不 idle、第 4 次 idle 之后才发下一步


def test_timeout_is_reported_and_continues(tmp_path):
    api = FakeApi(never_idle=True)
    sc = scenario(Step("say", "一", who="小明"), Step("say", "二", who="小明"), start=Start(memory="keep"))
    r, _ = replayer(api, sc, tmp_path, step_timeout=5.0)
    text = r.run().read_text(encoding="utf-8")
    assert [o["text"] for o in ops(api)] == ["一", "二"]
    assert text.count("超时") >= 2


def test_offline_then_online(tmp_path):
    api = FakeApi()
    sc = scenario(Step("say", "拜拜", who="小明"), Step("offline", True), Step("online", "sleep"), Step("say", "早", who="小明"),
                  start=Start(memory="keep"))
    r, _ = replayer(api, sc, tmp_path)
    text = r.run().read_text(encoding="utf-8")
    life = [c for c in api.calls if c[0] in ("start", "stop")]
    assert life == [("start", "resume"), ("stop",), ("start", "sleep")]
    stop_i = api.calls.index(("stop",))
    assert ("start", "sleep") in api.calls[stop_i:] and [o["text"] for o in ops(api)] == ["拜拜", "早"]
    assert "## 这次写的日记" in text and "今天小明放了我鸽子" in text and "旧日记" not in text.split("## 这次写的日记")[1]


def test_running_sandbox_is_stopped_first(tmp_path):
    api = FakeApi(running=True)
    r, _ = replayer(api, scenario(start=Start(memory="reset")), tmp_path)
    r.run()
    assert [c for c in api.calls if c[0] in ("stop", "reset", "start")] == [("stop",), ("reset",), ("start", "resume")]


def test_stop_requested_finishes_current_step(tmp_path):  # Review Focus 4
    api = FakeApi()
    sc = scenario(*(Step("say", str(i), who="小明") for i in range(1, 6)), start=Start(memory="keep"))
    r, _ = replayer(api, sc, tmp_path)

    def maybe_stop():
        if any(c[0] == "op" and c[1].get("text") == "2" for c in api.calls):
            r.stop_requested()

    api.on_state = maybe_stop
    text = r.run().read_text(encoding="utf-8")
    assert [o["text"] for o in ops(api)] == ["1", "2"]
    assert "第 2 步" in text and "第 3 步" not in text and "停下" in text
    assert r.progress()["running"] is False


def test_progress(tmp_path):
    api = FakeApi()
    sc = scenario(Step("say", "一", who="小明"), start=Start(memory="keep"))
    r, _ = replayer(api, sc, tmp_path)
    seen = []
    api.on_state = lambda: seen.append(r.progress())
    r.run()
    assert seen and seen[-1]["running"] is True and seen[-1]["total"] == 1 and seen[-1]["name"] == "放鸽子"
    assert any(p["step"] == 1 for p in seen)


def test_keep_memory_warns_in_report(tmp_path):
    api = FakeApi()
    text = replayer(api, scenario(start=Start(memory="keep")), tmp_path)[0].run().read_text(encoding="utf-8")
    assert "依赖沙盒当时的记忆" in text


# ---- 录制 ----
def test_recorder_roundtrip(tmp_path):
    rec = Recorder()
    rec.new(reset=True)
    rec.started("20:00")
    rec.record({"op": "come", "who": "小明"})
    rec.record({"op": "say", "who": "小明", "text": "在吗"})
    rec.record({"op": "skip", "seconds": 3600})
    rec.stopped()
    rec.started("sleep")
    s = rec.snapshot("录的", "随手录")
    assert s.start == Start(memory="reset", time="20:00")
    assert [(x.action, x.value) for x in s.steps] == [("come", "小明"), ("say", "在吗"), ("skip", 3600.0), ("offline", True),
                                                      ("online", "sleep")]
    path = tmp_path / "录的.toml"
    path.write_text(dumps(s), encoding="utf-8")
    assert load(path) == s


def test_recorder_keep_and_crash(tmp_path):
    rec = Recorder()
    rec.started("resume")  # 没重置记忆就启动
    rec.record({"op": "reflect"})
    rec.started("sleep")  # 上次崩了没记下线：先补一个 offline
    s = rec.snapshot("x", "")
    assert s.start.memory == "keep" and [x.action for x in s.steps] == ["reflect", "offline", "online"]
    assert rec.describe()["memory"] == "keep"
    rec.new()
    assert rec.snapshot("x", "").steps == []


def test_scenario_rejects_bad_numbers():  # 线 2 终审留下的：skip / wait 不是有限数、wait 为负
    for bad in ({"skip": math.nan}, {"skip": math.inf}, {"skip": 0}, {"skip": "0s"}, {"reflect": True, "wait": -1},
                {"reflect": True, "wait": math.inf}, {"reflect": True, "wait": math.nan}):
        with pytest.raises(ScenarioError, match="第 1 步"):
            parse({"steps": [bad]})


# ---- 管理面板 ----
def test_console_records_saves_lists_and_blocks_manual_ops_while_replaying(tmp_path, upstream):  # noqa: F811
    s = make(tmp_path, upstream)
    try:
        s.runner.state, s.runner.kind, s.runner.port = "running", "sandbox", upstream.server_address[1]
        s.recorder.started("resume")
        assert post(s, "sandbox/op", {"op": "say", "who": "小明", "text": "在吗"})[0] == 200
        status, res = post(s, "api/sandbox/save", {"name": "测试-1", "note": "录的"})
        assert status == 200 and res["ok"] and "依赖沙盒当时的记忆" in res["warning"]
        saved = load(tmp_path / "sandbox" / "scenarios" / "测试-1.toml")
        assert [(x.action, x.who, x.value) for x in saved.steps] == [("say", "小明", "在吗")]
        assert post(s, "api/sandbox/save", {"name": "测试-1", "note": ""})[0] == 409  # 已存在要 overwrite
        assert post(s, "api/sandbox/save", {"name": "测试-1", "note": "录的", "overwrite": True})[0] == 200
        assert post(s, "api/sandbox/save", {"name": "../坏", "note": ""})[0] == 400
        (tmp_path / "sandbox" / "scenarios" / "坏的.toml").write_text('[[steps]]\n"飞" = 1\n', encoding="utf-8")
        listed = {x["name"]: x for x in request(s.url + "api/sandbox/scenarios")[1]["scenarios"]}
        assert listed["测试-1"]["note"] == "录的" and "第 1 步" in listed["坏的"]["error"]
        assert post(s, "api/sandbox/replay", {"name": "坏的"})[0] == 400
        assert post(s, "api/sandbox/replay", {"name": "没有"})[0] == 404

        class Busy:
            def progress(self):
                return {"running": True, "step": 1, "total": 3, "name": "测试-1"}

            def stop_requested(self):
                self.stopped = True

        s.replayer = busy = Busy()
        status, res = post(s, "sandbox/op", {"op": "say", "who": "小明", "text": "我插一句"})
        assert status == 409 and "正在回放" in res["text"]
        assert post(s, "api/sandbox/replay", {"name": "测试-1"})[0] == 409
        assert request(s.url + "api/sandbox/replay")[1]["progress"]["running"] is True
        assert post(s, "api/sandbox/replay/stop", {})[0] == 200 and busy.stopped
        assert len(upstream.posts) == 1  # 回放中的手动操作没转给沙盒
    finally:
        s.stop()


def test_console_start_stop_reset_drive_recorder(tmp_path, upstream):  # noqa: F811
    s = make(tmp_path, upstream)
    try:
        assert post(s, "api/sandbox/reset", {})[0] == 200
        assert post(s, "api/sandbox/start", {"start": "20:00"})[0] == 200
        s.runner.state = "running"
        assert post(s, "api/sandbox/stop", {})[0] == 200
        s.runner.state = "exited"
        assert post(s, "api/sandbox/start", {"start": "sleep"})[0] == 200
        snap = s.recorder.snapshot("x", "")
        assert snap.start == Start(memory="reset", time="20:00") and [x.action for x in snap.steps] == ["offline", "online"]
        assert json.loads(json.dumps(request(s.url + "api/sandbox/replay")[1]))["recording"]["steps"] == 2
    finally:
        s.stop()
