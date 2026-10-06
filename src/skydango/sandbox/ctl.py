"""沙盒命令行客户端（`python -m skydango sandbox-ctl ...`）：不开管理面板网页也能操作沙盒、看团子的反应。

只是沙盒子进程 JSON 接口（sandbox/server.py）的客户端：
- 操作（say / come / leave / skip / time / reflect …）→ POST /op；默认接着长轮询 /state 打印新行，
  等团子这一轮反应完（idle、不在想、不在反思），再打印大脑这几轮的文字（"心里：…""不说：…"）和沙盒时间、精力
- state → GET /state（不等）；brain → GET /brain（最近几轮的工具调用和文字）
- start / stop 转发给管理面板（/api/sandbox/start、/api/sandbox/stop）：沙盒由面板起停，面板没开就报清楚
本机请求一律不走代理；--json 输出一个 JSON 对象（给 Agent 解析）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, TextIO
from urllib.parse import urlencode

SANDBOX_PORT = 19392
CONSOLE_PORT = 19390
POLL = 20.0  # 秒：/state 长轮询一次最多等多久（服务端最多 25）
OP_TIMEOUT = 180.0  # 秒：默认最多等团子反应多久（同 [sandbox] step_timeout）
START_TIMEOUT = 120.0
STOP_TIMEOUT = 120.0  # 下线要走最终反思、写日记
RESULT_MAX = 200  # brain 里工具返回最多显示几个字（--full 不截）
WAIT_OPS = ("say", "come", "leave", "strangers", "place", "scene", "notice", "skip", "time", "reflect")

LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 开着 Clash 时 127.0.0.1 也会被转走


class Unreachable(Exception):
    pass


class Failed(Exception):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


def _reason(data) -> str:
    if isinstance(data, dict):
        problems = [p.get("text", "") if isinstance(p, dict) else str(p) for p in data.get("problems") or []]
        if problems:
            return "\n".join(f"- {p}" for p in problems)
        return str(data.get("text") or data.get("error") or "")
    return str(data or "")


def request(port: int, method: str, path: str, query: dict | None = None, body: dict | None = None,
            timeout: float = 15.0):
    url = f"http://127.0.0.1:{port}{path}" + (f"?{urlencode(query)}" if query else "")
    data = json.dumps(body if body is not None else {}, ensure_ascii=False).encode("utf-8") if method == "POST" else None
    headers = {"Content-Type": "application/json", "X-Skydango": "1"} if method == "POST" else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with LOCAL.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            got = json.loads(raw.decode("utf-8"))
        except ValueError:
            got = raw.decode("utf-8", "replace")
        raise Failed(_reason(got) or f"HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise Unreachable(str(exc)) from None


# ---- 显示 ----

def _hm(t) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "--:--"


def format_line(r: dict) -> str:
    """聊天记录一行：heard 带跟谁说的标注、blocked 带被拦的原因。"""
    kind, who, text, why = r.get("kind"), r.get("who") or "", r.get("text") or "", r.get("why") or ""
    head = _hm(r.get("t"))
    if kind == "heard":
        return f"{head} {who}：{text}" + (f"  〔{why}〕" if why else "")
    if kind == "said":
        return f"{head} {who or '团子'}：{text}"
    if kind == "act":
        return f"{head} * {who + ' ' if who else ''}{text}"
    if kind == "blocked":
        return f"{head} ✗ 拦下 {who + '：' if who else ''}{text}" + (f"（{why}）" if why else "")
    return f"{head} {text}"


def footer(s: dict) -> str:
    energy = s.get("energy") or {}
    parts = [f"沙盒时间 {s.get('clock_text', '')}"]
    if energy.get("level"):
        parts.append(f"精力 {energy['level']}" + (f"（{energy['note']}）" if energy.get("note") else ""))
    parts.append("身边 " + ("、".join(s.get("friends") or []) or "没人"))
    if s.get("strangers"):
        parts.append(f"陌生人 {s['strangers']} 个")
    if s.get("place"):
        parts.append(f"在 {s['place']}")
    text = "── " + " · ".join(parts) + " ──"
    if s.get("limit"):
        text += f"\n!! {s['limit']}"
    return text


def summarize_turn(t: dict) -> dict:
    texts = [st.get("text", "") for st in t.get("steps") or [] if st.get("kind") == "text"]
    return {"id": t.get("id"), "reason": t.get("reason"), "seconds": t.get("seconds"),
            "tools": list(t.get("tools") or []), "texts": texts, "error": t.get("error")}


def _turn_head(t: dict) -> str:
    bits = [str(t.get("reason") or "")]
    if t.get("seconds") is not None:
        bits.append(f"{t['seconds']:.1f} 秒")
    elif t.get("end") is None:
        bits.append("还在想")
    if t.get("tools"):
        bits.append("工具 " + "、".join(t["tools"]))
    if t.get("error"):
        bits.append(f"出错：{t['error']}")
    return f"大脑 #{t.get('id')}（{'，'.join(b for b in bits if b)}）"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines() or [""])


def _clip(text: str, full: bool) -> str:
    text = str(text)
    return text if full or len(text) <= RESULT_MAX else text[:RESULT_MAX] + f"…（共 {len(text)} 字）"


def format_turn(t: dict, prompt: bool = False, full: bool = False) -> str:
    lines = [f"{_hm(t.get('start'))} {_turn_head(t)}"]
    if prompt and t.get("prompt"):
        lines += ["  【收到】", _indent(t["prompt"], "    ")]
    for st in t.get("steps") or []:
        kind = st.get("kind")
        if kind == "tool":
            args = json.dumps(st.get("input") or {}, ensure_ascii=False)
            lines.append(f"  ▶ {st.get('name')}({'' if args == '{}' else args})")
        elif kind == "result":
            lines.append(_indent(("↳ 出错 " if st.get("error") else "↳ ") + _clip(st.get("text", ""), full), "    "))
        elif kind == "text":
            lines.append(_indent(st.get("text", "")))
        elif kind == "thinking" and full:
            lines.append(_indent("（思考）" + st.get("text", "")))
    return "\n".join(lines)


# ---- 命令 ----

class Ctl:
    def __init__(self, args, out: TextIO) -> None:
        self.args = args
        self.out = out
        self.sb = args.port
        self.console = args.console_port
        self.json = args.json

    def say(self, text: str = "") -> None:
        if not self.json:
            print(text, file=self.out, flush=True)

    def emit(self, data: dict) -> None:
        if self.json:
            print(json.dumps(data, ensure_ascii=False), file=self.out, flush=True)

    def fail(self, text: str) -> int:
        if self.json:
            self.emit({"ok": False, "text": text})
        else:
            self.say(text)
        return 1

    def sandbox(self, method: str, path: str, query: dict | None = None, body: dict | None = None, wait: float = 0.0):
        try:
            return request(self.sb, method, path, query, body, timeout=wait + 15)
        except Unreachable:
            raise Failed(f"沙盒没在运行（127.0.0.1:{self.sb} 连不上）：先 `python -m skydango sandbox-ctl start`，"
                         "或在管理面板「沙盒」页启动") from None

    def panel(self, path: str, body: dict):
        try:
            return request(self.console, "POST", path, body=body, timeout=30)
        except Unreachable:
            raise Failed(f"管理面板没开（127.0.0.1:{self.console} 连不上）：沙盒由面板起停，"
                         "先运行 `python -m skydango console`（可加 --no-browser）") from None

    def state(self, after: int, wait: float) -> dict:
        return self.sandbox("GET", "/state", {"after": after, "wait": _num(wait)}, wait=wait)

    def brain(self, after: int) -> dict | None:
        try:
            return self.sandbox("GET", "/brain", {"after": after, "wait": 0})
        except Failed:
            return None  # 大脑时间线拿不到不影响操作本身

    # -- 操作 --
    def op(self, req: dict) -> int:
        a = self.args
        before = self.state(0, 0)
        brain0 = self.brain(0)
        result = self.sandbox("POST", "/op", body=req)
        if not result.get("ok"):
            if self.json:
                self.emit({"ok": False, "result": result, "text": result.get("text", "")})
                return 1
            return self.fail(result.get("text") or "沙盒没做成")
        self.say(f"→ {result.get('text', '')}")
        if a.no_wait:
            self.emit({"ok": True, "result": result})
            return 0
        lines: list[dict] = []

        def seen(r: dict) -> None:
            lines.append(r)
            self.say(format_line(r))

        last, quiet = self.wait_quiet(before["version"], a.timeout, seen)
        turns = []
        if brain0 is not None:
            got = self.brain(brain0.get("version", 0))
            turns = [t for t in (got or {}).get("turns") or [] if t.get("id")]
        for t in turns:
            s = summarize_turn(t)
            self.say(_turn_head(t))
            for text in s["texts"]:
                self.say(_indent(text))
        last = last or before
        self.say(footer(last))
        if not quiet:
            self.say(f"!! 等了 {a.timeout:g} 秒还没安静下来（大脑可能还在想），用 state / brain 接着看")
        self.emit({"ok": quiet, "result": result, "lines": lines, "brain": [summarize_turn(t) for t in turns],
                   "state": _no_lines(last), "timed_out": not quiet})
        return 0 if quiet else 1

    def wait_quiet(self, after: int, timeout: float, seen: Callable[[dict], None]) -> tuple[dict | None, bool]:
        """长轮询 /state 打印新行，直到 idle、不在想、不在反思；超时返回 (最后的状态, False)。"""
        deadline = time.monotonic() + timeout
        last = None
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return last, False
            s = self.state(after, min(self.args.poll, left))
            for r in s.get("lines") or []:
                seen(r)
            after, last = s.get("version", after), s
            if s.get("idle") and not s.get("thinking") and not s.get("reflecting"):
                return s, True

    # -- 查看 --
    def show_state(self) -> int:
        s = self.state(0, 0)
        tail = (s.get("lines") or [])[-self.args.lines:] if self.args.lines > 0 else []
        if self.json:
            self.emit({"ok": True, "state": _no_lines(s), "lines": tail})
            return 0
        for r in tail:
            self.say(format_line(r))
        doing = "在反思" if s.get("reflecting") else "在想" if s.get("thinking") else "安静" if s.get("idle") else "在忙"
        self.say(footer(s))
        extra = [f"团子：{doing}"]
        if s.get("scene"):
            extra.append(f"场景：{s['scene']}")
        self.say("   " + " · ".join(extra))
        return 0

    def show_brain(self) -> int:
        a = self.args
        got = self.sandbox("GET", "/brain", {"after": 0, "wait": 0})
        turns = sorted((t for t in got.get("turns") or [] if t.get("id")), key=lambda t: t["id"])[-max(1, a.last):]
        if self.json:
            self.emit({"ok": True, "state": got.get("state") or {}, "turns": turns})
            return 0
        if not turns:
            self.say("大脑还没醒过")
        for t in turns:
            self.say(format_turn(t, prompt=a.prompt, full=a.full))
        return 0

    # -- 起停（转发给管理面板）--
    def start(self) -> int:
        a = self.args
        self.panel("/api/sandbox/start", {"start": a.start})
        if a.no_wait:
            self.say("沙盒正在启动")
            self.emit({"ok": True, "text": "沙盒正在启动"})
            return 0
        self.say("沙盒正在启动…")
        deadline = time.monotonic() + a.timeout
        while time.monotonic() < deadline:
            try:
                if request(self.sb, "GET", "/status", timeout=3).get("ready"):
                    s = self.state(0, 0)
                    self.say("沙盒起来了")
                    self.say(footer(s))
                    self.emit({"ok": True, "state": _no_lines(s)})
                    return 0
            except (Unreachable, Failed):
                pass
            time.sleep(min(1.0, a.poll))
        return self.fail(f"等了 {a.timeout:g} 秒沙盒还没起来：看管理面板「沙盒」页或 runs/ 里这次的 agent.log")

    def stop(self) -> int:
        a = self.args
        self.panel("/api/sandbox/stop", {})
        if a.no_wait:
            self.say("正在下线（写日记）")
            self.emit({"ok": True, "text": "正在下线"})
            return 0
        self.say("正在下线（写日记）…")
        deadline = time.monotonic() + a.timeout
        while time.monotonic() < deadline:
            try:
                request(self.sb, "GET", "/status", timeout=3)
            except Unreachable:
                self.say("沙盒下线了")
                self.emit({"ok": True, "text": "沙盒下线了"})
                return 0
            except Failed:
                pass
            time.sleep(min(1.0, a.poll))
        return self.fail(f"等了 {a.timeout:g} 秒沙盒还没退出：看管理面板「沙盒」页")


def _no_lines(s: dict | None) -> dict:
    return {k: v for k, v in (s or {}).items() if k != "lines"}


def _num(x: float):
    return int(x) if float(x).is_integer() else x


def _skip_seconds(text: str):
    """skip 的参数：纯数字 = 分钟；30s / 10m / 2h 原样交给沙盒解析。"""
    try:
        value = float(text)
    except ValueError:
        if not re.fullmatch(r"\s*\d+(\.\d+)?\s*[smhSMH]\s*", text):
            raise argparse.ArgumentTypeError(f"快进多久：分钟数，或 30s / 10m / 2h：{text!r}") from None
        return text.strip()
    if value <= 0:
        raise argparse.ArgumentTypeError("快进的分钟数要是正数")
    return _num(value * 60)


def _build_op(args) -> dict:
    name = args.action
    if name == "say":
        return {"op": "say", "who": args.who, "text": " ".join(args.text)}
    if name in ("come", "leave"):
        return {"op": name, "who": args.who}
    if name == "skip":
        return {"op": "skip", "seconds": args.seconds}
    if name == "time":
        return {"op": "time", "at": args.at}
    if name == "strangers":
        return {"op": "strangers", "n": args.n}
    if name == "place":
        return {"op": "place", "name": args.name}
    if name in ("scene", "notice"):
        return {"op": name, "text": " ".join(args.text)}
    return {"op": "reflect"}


def configure(p: argparse.ArgumentParser, sandbox_port: int | None = SANDBOX_PORT, console_port: int | None = CONSOLE_PORT) -> None:
    """给 sandbox-ctl 的解析器加参数（cli.py 和 main() 共用）；端口默认 None 时由调用方按配置补。"""
    p.add_argument("--port", type=int, default=sandbox_port, help=f"沙盒接口端口（默认 [sandbox] port = {SANDBOX_PORT}）")
    p.add_argument("--console-port", type=int, default=console_port, help=f"管理面板端口（默认 [console] port = {CONSOLE_PORT}）")
    p.add_argument("--json", action="store_true", help="输出一个 JSON 对象（给 Agent 解析）")
    p.add_argument("--poll", type=float, default=POLL, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="action", required=True)

    wait = argparse.ArgumentParser(add_help=False)
    wait.add_argument("--no-wait", action="store_true", help="发出去就返回，不等团子反应")
    wait.add_argument("--timeout", type=float, default=OP_TIMEOUT, help=f"最多等团子反应几秒（默认 {OP_TIMEOUT:g}）")

    q = sub.add_parser("say", parents=[wait], help="冒充某人说话，等团子这一轮反应完，打印新行和大脑的文字")
    q.add_argument("who")
    q.add_argument("text", nargs="+")
    for name, desc in (("come", "某个好友来到身边"), ("leave", "某个好友走开")):
        q = sub.add_parser(name, parents=[wait], help=desc)
        q.add_argument("who")
    q = sub.add_parser("skip", parents=[wait], help="快进：分钟数，或 30s / 10m / 2h")
    q.add_argument("seconds", type=_skip_seconds)
    q = sub.add_parser("time", parents=[wait], help="把沙盒时间拨到 HH:MM（只能拨到更晚，不能倒回去）")
    q.add_argument("at")
    sub.add_parser("reflect", parents=[wait], help="立刻反思一次")
    q = sub.add_parser("strangers", parents=[wait], help="身边陌生人个数")
    q.add_argument("n", type=int)
    q = sub.add_parser("place", parents=[wait], help="地名（不填 = 清空）")
    q.add_argument("name", nargs="?", default="")
    q = sub.add_parser("scene", parents=[wait], help="眼睛看到的场景（不填 = 清空，团子看不清）")
    q.add_argument("text", nargs="*")
    q = sub.add_parser("notice", parents=[wait], help="放一条新鲜事（同眼睛发现的，过主动开口的护栏）")
    q.add_argument("text", nargs="+")

    q = sub.add_parser("state", help="沙盒现在：时间、精力、身边、额度 + 最近几行聊天")
    q.add_argument("--lines", type=int, default=20, help="打印最近几行聊天记录（默认 20，0 = 不打印）")
    q = sub.add_parser("brain", help="大脑最近几轮：调了什么工具、返回了什么、写了什么")
    q.add_argument("--last", type=int, default=1, help="最近几轮（默认 1）")
    q.add_argument("--prompt", action="store_true", help="也打印这一轮收到的消息")
    q.add_argument("--full", action="store_true", help="工具返回不截断，带上思考")

    q = sub.add_parser("start", help="让管理面板起沙盒（面板要开着）")
    q.add_argument("start", nargs="?", default="resume", help="resume 接着上次（默认）/ sleep 睡一晚 / HH:MM / \"YYYY-MM-DD HH:MM\"")
    q.add_argument("--no-wait", action="store_true", help="不等沙盒起来")
    q.add_argument("--timeout", type=float, default=START_TIMEOUT)
    q = sub.add_parser("stop", help="让管理面板下线沙盒（走最终反思、写日记）")
    q.add_argument("--no-wait", action="store_true", help="不等沙盒退出")
    q.add_argument("--timeout", type=float, default=STOP_TIMEOUT)


def run(args, out: TextIO | None = None) -> int:
    if out is None:
        out = sys.stdout
        try:
            out.reconfigure(encoding="utf-8")  # Windows 终端默认 GBK，打不出 ✗ ▶ 之类
        except (AttributeError, ValueError):
            pass
    ctl = Ctl(args, out)
    try:
        if args.action in WAIT_OPS:
            return ctl.op(_build_op(args))
        if args.action == "state":
            return ctl.show_state()
        if args.action == "brain":
            return ctl.show_brain()
        if args.action == "start":
            return ctl.start()
        return ctl.stop()
    except Failed as exc:
        return ctl.fail(exc.text)


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    p = argparse.ArgumentParser(prog="sandbox-ctl", description="大脑沙盒的命令行客户端")
    configure(p)
    return run(p.parse_args(argv), out)
