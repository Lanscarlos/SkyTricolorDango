"""沙盒命令行客户端（sandbox-ctl）：假的沙盒 / 管理面板 HTTP 服务，不起真沙盒。"""

from __future__ import annotations

import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from skydango.sandbox import ctl


class Fake:
    """按 (方法, 路径) 回脚本里的响应；同一路由给列表时按顺序取，取到最后一个就一直回它。"""

    def __init__(self, routes: dict):
        self.routes = {k: (list(v) if isinstance(v, list) else [v]) for k, v in routes.items()}
        self.calls: list[tuple[str, str, dict, object]] = []
        self.times: list[float] = []  # 和 calls 一一对应：收到请求的 monotonic 时间
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self, method):
                url = urlparse(self.path)
                body = None
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"null")
                    assert self.headers.get("X-Skydango") == "1"
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                fake.calls.append((method, url.path, query, body))
                fake.times.append(time.monotonic())
                queue = fake.routes.get((method, url.path))
                if not queue:
                    code, data = 404, {"ok": False}
                else:
                    item = queue.pop(0) if len(queue) > 1 else queue[0]
                    code, data = item(query, body) if callable(item) else item
                raw = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                self._go("GET")

            def do_POST(self):  # noqa: N802
                self._go("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def paths(self, method=None):
        return [c[1] for c in self.calls if method is None or c[0] == method]


def free_port() -> int:
    s = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = s.server_address[1]
    s.server_close()
    return port


def state(version, lines=(), idle=True, thinking=False, **extra):
    data = {
        "version": version, "wall": 0, "clock_text": "10月6日 周二 21:30",
        "energy": {"level": "还行", "score": 0.6, "note": "晚上"},
        "friends": ["小明"], "strangers": 0, "place": "", "scene": "",
        "lines": list(lines), "idle": idle, "thinking": thinking, "reflecting": False, "limit": "",
    }
    data.update(extra)
    return 200, data


def row(seq, kind, text, who="", why=""):
    r = {"seq": seq, "t": 1_791_300_000 + seq, "kind": kind, "who": who, "text": text}
    if why:
        r["why"] = why
    return r


def brain(version, turns=()):
    return 200, {"boot": "b", "version": version, "oldest": 1, "state": {}, "turns": list(turns)}


def turn(tid, text, tools=("say",), reason="chat"):
    steps = [{"kind": "tool", "name": t, "input": {}} for t in tools] + [{"kind": "text", "text": text}]
    return {"id": tid, "reason": reason, "start": 0, "end": 1, "seconds": 3.2, "prompt": "小明：你好",
            "steps": steps, "tools": list(tools), "result": None, "error": None, "updated": 9}


@pytest.fixture
def run():
    fakes: list[Fake] = []

    def go(argv, sandbox=None, console=None):
        sb = Fake(sandbox) if sandbox is not None else None
        cs = Fake(console) if console is not None else None
        fakes.extend(f for f in (sb, cs) if f)
        out = io.StringIO()
        ports = ["--port", str(sb.port if sb else free_port()), "--console-port", str(cs.port if cs else free_port())]
        code = ctl.main(ports + ["--poll", "0.2"] + argv, out=out)
        return code, out.getvalue(), sb, cs

    yield go
    for f in fakes:
        f.close()


# ---- 一行聊天记录怎么显示 ----

def test_format_heard_with_addressee_note():
    text = ctl.format_line(row(1, "heard", "团子你在干嘛", who="小明", why="跟你说：叫了你"))
    assert "小明：团子你在干嘛" in text and "跟你说：叫了你" in text


def test_format_said_act_event_blocked():
    assert "团子：在发呆" in ctl.format_line(row(2, "said", "在发呆", who="团子"))
    assert "鞠躬" in ctl.format_line(row(3, "act", "（做了个鞠躬）", who="团子"))
    assert "── 小明来了 ──" in ctl.format_line(row(4, "event", "── 小明来了 ──"))
    blocked = ctl.format_line(row(5, "blocked", "我是真人", who="团子", why="不能说自己是真人"))
    assert "拦下" in blocked and "我是真人" in blocked and "不能说自己是真人" in blocked


# ---- 操作：发出去、等团子反应完 ----

def test_say_posts_op_and_prints_reaction_until_idle(run):
    code, out, sb, _ = run(["say", "小明", "你好"], sandbox={
        ("GET", "/state"): [
            state(5),  # 发之前：拿版本号
            state(7, [row(6, "heard", "你好", who="小明", why="跟你说：身边只有他"), row(7, "event", "── x ──")],
                  idle=False, thinking=True),
            state(8, [row(8, "said", "嗨", who="团子")], idle=True),
        ],
        ("GET", "/brain"): [brain(3), brain(5, [turn(2, "心里：小明来打招呼了")])],
        ("POST", "/op"): (200, {"ok": True, "text": "小明 说了：你好"}),
    })
    assert code == 0
    op = [c for c in sb.calls if c[1] == "/op"][0]
    assert op[3] == {"op": "say", "who": "小明", "text": "你好"}
    afters = [c[2].get("after") for c in sb.calls if c[1] == "/state"]
    assert afters[:3] == ["0", "5", "7"]
    assert "小明：你好" in out and "跟你说：身边只有他" in out
    assert "团子：嗨" in out
    assert "心里：小明来打招呼了" in out
    assert "10月6日 周二 21:30" in out and "还行" in out
    assert out.index("小明：你好") < out.index("团子：嗨")


def test_brain_text_only_from_turns_after_the_op(run):
    old = turn(1, "不说：上一轮的事")
    code, out, sb, _ = run(["say", "小明", "你好"], sandbox={
        ("GET", "/state"): [state(5), state(6, [row(6, "said", "嗨", who="团子")], idle=True)],
        ("GET", "/brain"): [brain(3, [old]), brain(5, [turn(2, "心里：这一轮")])],
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    brain_calls = [c[2] for c in sb.calls if c[1] == "/brain"]
    assert brain_calls[-1].get("after") == "3"
    assert "这一轮" in out and "上一轮的事" not in out


def test_no_wait_returns_after_op(run):
    code, out, sb, _ = run(["say", "小明", "你好", "--no-wait"], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (200, {"ok": True, "text": "小明 说了：你好"}),
    })
    assert code == 0
    assert "小明 说了：你好" in out
    assert not [c for c in sb.calls if c[1] == "/state" and c[2].get("after") == "5"]


def test_op_refused_exits_1_without_waiting(run):
    code, out, sb, _ = run(["come", "小明"], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (200, {"ok": False, "text": "小明 已经在身边了"}),
    })
    assert code == 1
    assert "小明 已经在身边了" in out
    assert len([c for c in sb.calls if c[1] == "/state"]) == 1


def test_bad_op_400_shows_reason(run):
    code, out, _, _ = run(["time", "25:99"], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (400, {"ok": False, "text": "时间格式不对"}),
    })
    assert code == 1 and "时间格式不对" in out


@pytest.mark.parametrize("argv, req", [
    (["come", "小明"], {"op": "come", "who": "小明"}),
    (["leave", "小明"], {"op": "leave", "who": "小明"}),
    (["skip", "30"], {"op": "skip", "seconds": 1800}),
    (["skip", "2h"], {"op": "skip", "seconds": "2h"}),
    (["skip", "1.5"], {"op": "skip", "seconds": 90}),
    (["time", "23:30"], {"op": "time", "at": "23:30"}),
    (["reflect"], {"op": "reflect"}),
    (["strangers", "2"], {"op": "strangers", "n": 2}),
    (["place", "雨林"], {"op": "place", "name": "雨林"}),
    (["place"], {"op": "place", "name": ""}),
    (["scene", "下雨了", "有篝火"], {"op": "scene", "text": "下雨了 有篝火"}),
    (["notice", "天黑了"], {"op": "notice", "text": "天黑了"}),
])
def test_ops_send_right_request(run, argv, req):
    code, _, sb, _ = run(argv + ["--no-wait"], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    assert code == 0
    assert [c[3] for c in sb.calls if c[1] == "/op"] == [req]


def test_wait_times_out(run):
    code, out, _, _ = run(["say", "小明", "你好", "--timeout", "0.5"], sandbox={
        ("GET", "/state"): [state(5), state(5, idle=False, thinking=True)],
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    assert code == 1 and "还没安静" in out


def test_json_output(run):
    code, out, _, _ = run(["--json", "say", "小明", "你好"], sandbox={
        ("GET", "/state"): [state(5), state(6, [row(6, "said", "嗨", who="团子")], idle=True)],
        ("GET", "/brain"): [brain(3), brain(5, [turn(2, "心里：嗯")])],
        ("POST", "/op"): (200, {"ok": True, "text": "小明 说了：你好"}),
    })
    data = json.loads(out)
    assert code == 0 and data["ok"] is True
    assert data["result"]["text"] == "小明 说了：你好"
    assert [r["text"] for r in data["lines"]] == ["嗨"]
    assert data["brain"][0]["texts"] == ["心里：嗯"]
    assert data["state"]["clock_text"] == "10月6日 周二 21:30" and "lines" not in data["state"]


# ---- 沙盒没开 / 还在启动 ----

def test_sandbox_unreachable_message(run):
    code, out, _, _ = run(["say", "小明", "你好"])
    assert code == 1
    assert "沙盒没在运行" in out and "sandbox-ctl start" in out


def test_sandbox_starting_503(run):
    code, out, _, _ = run(["state"], sandbox={("GET", "/state"): (503, {"ok": False, "text": "沙盒还在启动"})})
    assert code == 1 and "沙盒还在启动" in out


# ---- state / brain ----

def test_state_prints_summary_and_last_lines(run):
    lines = [row(i, "heard", f"第{i}句", who="小明") for i in range(1, 6)]
    code, out, sb, _ = run(["state", "--lines", "2"], sandbox={
        ("GET", "/state"): state(5, lines, place="雨林", limit="额度用完，约 3 分钟后再试"),
    })
    assert code == 0
    assert "10月6日 周二 21:30" in out and "小明" in out and "雨林" in out and "额度用完" in out
    assert "第5句" in out and "第4句" in out and "第3句" not in out
    assert sb.calls[0][2] == {"after": "0", "wait": "0"}


def test_brain_last_n(run):
    turns = [turn(i, f"心里：第{i}轮", tools=("status",)) for i in range(1, 4)]
    code, out, _, _ = run(["brain", "--last", "2"], sandbox={("GET", "/brain"): brain(9, turns)})
    assert code == 0
    assert "第3轮" in out and "第2轮" in out and "第1轮" not in out
    assert "status" in out


def test_brain_prompt_only_with_flag(run):
    t = turn(1, "心里：嗯")
    t["prompt"] = "很长的唤醒消息"
    _, out, _, _ = run(["brain"], sandbox={("GET", "/brain"): brain(9, [t])})
    assert "很长的唤醒消息" not in out
    _, out, _, _ = run(["brain", "--prompt"], sandbox={("GET", "/brain"): brain(9, [t])})
    assert "很长的唤醒消息" in out


# ---- start / stop：转发给管理面板 ----

def test_start_forwards_and_waits_ready(run):
    code, out, sb, cs = run(["start", "sleep"], console={
        ("POST", "/api/sandbox/start"): (200, {"ok": True}),
    }, sandbox={
        ("GET", "/status"): [(200, {"ok": True, "kind": "sandbox", "ready": False}),
                             (200, {"ok": True, "kind": "sandbox", "ready": True})],
        ("GET", "/state"): state(0),
    })
    assert code == 0
    assert [c[3] for c in cs.calls] == [{"start": "sleep"}]
    assert "10月6日 周二 21:30" in out


def test_start_problems_listed(run):
    code, out, _, _ = run(["start"], console={
        ("POST", "/api/sandbox/start"): (409, {"ok": False, "problems": [{"text": "团子在运行，先停团子"}]}),
    })
    assert code == 1 and "团子在运行，先停团子" in out


def test_start_console_not_running(run):
    code, out, _, _ = run(["start"])
    assert code == 1
    assert "管理面板没开" in out and "skydango console" in out


def test_stop_forwards_and_waits_gone(run):
    code, out, _, cs = run(["stop", "--no-wait"], console={("POST", "/api/sandbox/stop"): (200, {"ok": True})})
    assert code == 0 and cs.paths("POST") == ["/api/sandbox/stop"]


def test_stop_not_running(run):
    code, out, _, _ = run(["stop"], console={("POST", "/api/sandbox/stop"): (409, {"ok": False, "text": "沙盒没在运行"})})
    assert code == 1 and "沙盒没在运行" in out


def test_stop_waits_until_port_closed(run):
    # 沙盒端口上没有服务 = 已经下线：等的第一下就结束
    code, out, _, _ = run(["stop"], console={("POST", "/api/sandbox/stop"): (200, {"ok": True})})
    assert code == 0 and "下线了" in out


# ---- python -m skydango sandbox-ctl：端口按配置 ----

def test_cli_uses_config_ports(tmp_path, capsys):
    from skydango import cli

    sb = Fake({("GET", "/state"): state(3, [row(3, "said", "嗨", who="团子")])})
    try:
        cfg = tmp_path / "config.toml"
        cfg.write_text(f"[sandbox]\nport = {sb.port}\n\n[console]\nport = 1\n", encoding="utf-8")
        cli.main(["-c", str(cfg), "sandbox-ctl", "state", "--lines", "1"])
    finally:
        sb.close()
    assert "团子：嗨" in capsys.readouterr().out


def test_cli_exit_code_on_failure(tmp_path):
    from skydango import cli

    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[sandbox]\nport = {free_port()}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["-c", str(cfg), "sandbox-ctl", "state"])
    assert exc.value.code == 1


# ---- 对着真的 SandboxServer（假 control）：Host、X-Skydango 头、长轮询参数都对得上 ----

class ScriptedControl:
    def __init__(self):
        self.ops = []
        self.version = 1

    def op(self, req):
        self.ops.append(req)
        return {"ok": True, "text": f"{req['who']} 说了：{req['text']}"}

    def state(self, after, timeout):
        lines = []
        if self.ops and after < 2:
            self.version = 2
            lines = [row(2, "said", "在呢", who="团子")]
        return {**state(self.version, lines)[1], "version": self.version}


def test_against_real_sandbox_server():
    from skydango.brain.trace import BrainTrace
    from skydango.sandbox.server import SandboxServer

    srv = SandboxServer(0, on_shutdown=lambda: None)
    srv.start()
    srv.control = ScriptedControl()
    srv.trace = BrainTrace()
    try:
        out = io.StringIO()
        code = ctl.main(["--port", str(srv._server.server_address[1]), "say", "小明", "在吗"], out=out)
    finally:
        srv.stop()
    assert code == 0
    assert srv.control.ops == [{"op": "say", "who": "小明", "text": "在吗"}]
    assert "团子：在呢" in out.getvalue()


# ---- play：按真实时间线回放（spec 2026-10-21-chat-rhythm §3）----

def test_parse_script_formats():
    text = "﻿# 10-07\n0 小明：嘻嘻\r\n\n2.5 小明:团子\n12 小明：我说：好\n"
    assert ctl.parse_script(text) == [(0.0, "小明", "嘻嘻"), (2.5, "小明", "团子"), (12.0, "小明", "我说：好")]


@pytest.mark.parametrize("bad, n", [("0 小明 嘻嘻", 1), ("x 小明：嘻嘻", 1), ("0 小明：嘻嘻\n3 ：空", 2),
                                    ("5 小明：a\n2 小明：b", 2), ("0 小明：", 1), ("# 只有注释", 0)])
def test_parse_script_bad_line(bad, n):
    with pytest.raises(ctl.Failed) as e:
        ctl.parse_script(bad)
    assert (f"第 {n} 行" in e.value.text) if n else "一句都没有" in e.value.text


def _script(tmp_path, text):
    f = tmp_path / "play.txt"
    f.write_text(text, encoding="utf-8")
    return str(f)


def test_play_sends_on_time_without_waiting_then_waits_idle(run, tmp_path):
    code, out, sb, _ = run(["play", _script(tmp_path, "0 小明：嘻嘻\n0.4 小明：团子\n")], sandbox={
        ("GET", "/state"): [state(5), state(7, [row(6, "heard", "嘻嘻", who="小明"), row(7, "heard", "团子", who="小明")],
                                               idle=False, thinking=True),
                            state(8, [row(8, "said", "干嘛", who="团子")], idle=True)],
        ("GET", "/brain"): [brain(3), brain(5, [turn(2, "心里：他叫我")])],
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    assert code == 0
    ops = [(c[3], t) for c, t in zip(sb.calls, sb.times) if c[1] == "/op"]
    assert [o for o, _ in ops] == [{"op": "say", "who": "小明", "text": "嘻嘻"}, {"op": "say", "who": "小明", "text": "团子"}]
    assert ops[1][1] - ops[0][1] >= 0.35
    first_op = sb.calls.index(next(c for c in sb.calls if c[1] == "/op"))
    assert all(c[2].get("after") in (None, "0") for c in sb.calls[:first_op] if c[1] == "/state")  # 发之前只拿版本号
    assert "+0.0s 小明：嘻嘻" in out and "+0.4s 小明：团子" in out
    assert "团子：干嘛" in out and "心里：他叫我" in out


def _staged(ops):
    """/state 按已经发了几句回：发第一句前拿版本号；发了一句后（两句之间）一直是「干嘛」那行（重复回、要去重）；
    两句都发了之后才有「叫我啥事」、idle。"""
    def route(query, body):
        if not ops:
            return state(5)
        if len(ops) == 1:
            return state(6, [row(6, "said", "干嘛", who="团子")], idle=False, thinking=True)
        return state(7, [row(7, "said", "叫我啥事", who="团子")], idle=True)
    return route


def _counting(ops):
    def route(query, body):
        ops.append(body)
        return 200, {"ok": True, "text": "ok"}
    return route


def test_play_prints_reactions_between_lines_with_offsets(run, tmp_path):  # 终审：要能和团子的回复对时间
    ops: list = []
    code, out, _, _ = run(["--json", "play", _script(tmp_path, "0 小明：嘻嘻\n0.6 小明：团子\n")], sandbox={
        ("GET", "/state"): _staged(ops), ("GET", "/brain"): [brain(3), brain(5)], ("POST", "/op"): _counting(ops),
    })
    data = json.loads(out)
    assert code == 0
    assert [r["text"] for r in data["lines"]] == ["干嘛", "叫我啥事"]  # 两句之间读到的不重复
    assert data["lines"][0]["at"] < 0.6 <= data["lines"][1]["at"]
    ops = []
    code, out, _, _ = run(["play", _script(tmp_path, "0 小明：嘻嘻\n0.6 小明：团子\n")], sandbox={
        ("GET", "/state"): _staged(ops), ("GET", "/brain"): [brain(3), brain(5)], ("POST", "/op"): _counting(ops),
    })
    assert out.index("团子：干嘛") < out.index("+0.6s 小明：团子")
    assert re.search(r"\+0\.\ds .*团子：干嘛", out)


@pytest.mark.parametrize("encoding", ["utf-16", "gbk"])
def test_play_file_not_utf8_is_friendly(run, tmp_path, encoding):  # 终审：PowerShell 5.1 Out-File 默认 UTF-16
    f = tmp_path / "play.txt"
    f.write_text("0 小明：嘻嘻\n", encoding=encoding)
    code, out, sb, _ = run(["play", str(f)], sandbox={
        ("GET", "/state"): state(5), ("GET", "/brain"): brain(3), ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    assert code == 1 and "UTF-8" in out
    assert not [c for c in sb.calls if c[1] == "/op"]


def test_play_refused_line_stops(run, tmp_path):
    code, out, sb, _ = run(["play", _script(tmp_path, "0 小明：a\n0.1 阿花：b\n0.2 小明：c\n")], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): [(200, {"ok": True, "text": "ok"}), (200, {"ok": False, "text": "阿花 不在身边"})],
    })
    assert code == 1
    assert "第 2 句" in out and "不在身边" in out
    assert len([c for c in sb.calls if c[1] == "/op"]) == 2


def test_play_bad_file_sends_nothing(run, tmp_path):
    code, out, sb, _ = run(["play", _script(tmp_path, "0 小明：a\n不对\n")], sandbox={
        ("GET", "/state"): state(5),
        ("GET", "/brain"): brain(3),
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    assert code == 1 and "第 2 行" in out
    assert not [c for c in sb.calls if c[1] == "/op"]


def test_play_json(run, tmp_path):
    code, out, _, _ = run(["--json", "play", _script(tmp_path, "0 小明：嘻嘻\n")], sandbox={
        ("GET", "/state"): [state(5), state(6, [row(6, "said", "嗯哼", who="团子")], idle=True)],
        ("GET", "/brain"): [brain(3), brain(5, [turn(2, "心里：嗯")])],
        ("POST", "/op"): (200, {"ok": True, "text": "ok"}),
    })
    data = json.loads(out)
    assert code == 0 and data["ok"] is True
    assert data["sent"] == [{"t": 0.0, "who": "小明", "text": "嘻嘻"}]
    assert [r["text"] for r in data["lines"]] == ["嗯哼"]
    assert data["brain"][0]["texts"] == ["心里：嗯"]
