"""沙盒剧本的录制、回放、报告（brain-sandbox spec §6）：都在管理面板进程里做。

- Recorder：记下转发过的 /sandbox/op、沙盒的启动 / 停止（停止 = offline、再启动 = online），"另存为"成剧本
- Replayer：按剧本再发一遍；每步发出后等沙盒安静（idle），超时记"超时"接着下一步；offline 等子进程退出、online 再起一个；
  写报告 sandbox/reports/<剧本>-<时间>.md。沙盒接口是一个小对象（api），面板里用真的，测试里用假的
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .scenario import Scenario, Start, Step, to_op

log = logging.getLogger(__name__)

POLL = 0.5  # 秒：等沙盒安静 / 退出时多久看一次
KEEP_NOTE = "这个剧本依赖沙盒当时的记忆（录的时候没先重置记忆）"
NAME_OK = re.compile(r"[0-9A-Za-z_\-一-鿿]{1,40}")


def safe_name(name: str) -> bool:
    """剧本名只许中英文、数字和 - _（直接当文件名用）。"""
    return isinstance(name, str) and bool(NAME_OK.fullmatch(name))


# ---- 录制 ----
def op_to_step(op: dict) -> Step | None:
    """/sandbox/op 的请求 → 剧本的一步；不认识的返回 None。"""
    kind = op.get("op")
    if kind == "say":
        return Step("say", op.get("text", ""), who=op.get("who", ""))
    if kind in ("come", "leave"):
        return Step(kind, op.get("who", ""))
    if kind == "strangers":
        return Step("strangers", op.get("n", 0))
    if kind == "place":
        return Step("place", op.get("name", ""))
    if kind in ("scene", "notice"):
        return Step(kind, op.get("text", ""))
    if kind == "skip":
        value = op.get("seconds")
        return Step("skip", float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
    if kind == "time":
        return Step("time", op.get("at", ""))
    if kind == "reflect":
        return Step("reflect", True)
    return None


class Recorder:
    """从沙盒启动开始自动录，停止 / 再启动接着录（offline / online），直到「新录制」或重置记忆。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.new()

    def new(self, reset: bool = False) -> None:
        """清空；reset = 刚重置过记忆（下次启动的剧本写 memory = "reset"）。"""
        with self._lock:
            self.start: Start | None = None
            self.steps: list[Step] = []
            self.online = False
            self._reset = reset

    def begin(self, start: Start) -> None:
        with self._lock:
            self.start, self.steps, self.online = start, [], True

    def record(self, op: dict) -> None:
        step = op_to_step(op)
        if step is None:
            return
        with self._lock:
            if self.start is None:  # 没从启动开始录（比如回放完接着手动玩）：当作用当时的记忆
                self.start, self.online = Start(memory="keep"), True
            self.steps.append(step)

    def offline(self) -> None:
        with self._lock:
            if self.start is not None and self.online:
                self.steps.append(Step("offline", True))
                self.online = False

    def online_again(self, choice: str) -> None:
        with self._lock:
            if self.online:  # 上次没记下线（子进程崩了）：先补一个 offline，剧本才读得回来
                self.steps.append(Step("offline", True))
            self.steps.append(Step("online", choice or "resume"))
            self.online = True

    def started(self, choice: str) -> None:
        """沙盒（被页面）启动了：第一次是剧本的 [start]，之后是 online。"""
        if self.start is None:
            self.begin(Start(memory="reset" if self._reset else "keep", time="" if choice == "resume" else choice))
            self._reset = False
        else:
            self.online_again(choice)

    def stopped(self) -> None:
        self.offline()

    def snapshot(self, name: str, note: str) -> Scenario:
        with self._lock:
            start = self.start or Start(memory="keep")
            steps = list(self.steps)
        return Scenario(name, note, Start(**vars(start)), steps)

    def describe(self) -> dict:
        with self._lock:
            return {"active": self.start is not None, "steps": len(self.steps),
                    "memory": self.start.memory if self.start is not None else ("reset" if self._reset else "keep")}


# ---- 回放 ----
def step_title(step: Step) -> str:
    a, v = step.action, step.value
    return {
        "say": lambda: f"{step.who}说「{v}」",
        "come": lambda: f"{v}来了",
        "leave": lambda: f"{v}走了",
        "strangers": lambda: f"陌生人变成 {v} 个",
        "place": lambda: f"到了{v}" if v else "地名清空",
        "scene": lambda: f"场景：{v}" if v else "场景清空",
        "notice": lambda: f"新鲜事：{v}",
        "skip": lambda: f"快进 {v}" if isinstance(v, str) else f"快进 {v:g} 秒",
        "time": lambda: f"拨到 {v}",
        "reflect": lambda: "立刻反思一次",
        "offline": lambda: "下线（写日记）",
        "online": lambda: {"sleep": "睡一晚再上线", "resume": "接着上线"}.get(v, f"再上线（{v}）"),
    }[a]()


def line_text(r: dict) -> str:
    t = time.strftime("%H:%M", time.localtime(r.get("t") or 0))
    kind, text = r.get("kind"), r.get("text", "")
    if kind == "heard":
        return f"{t} {r.get('who') or '？'}：{text}"
    if kind == "said":
        return f"{t} 团子：{text}"
    if kind == "blocked":
        why = f"（{r['why']}）" if r.get("why") else ""
        return f"{t} ~~{text}~~ 被拦下{why}"
    return f"{t} {text}"


class Replayer:
    def __init__(self, api, scenario: Scenario, report_dir: Path, step_timeout: float, stop_timeout: float,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.api = api
        self.scenario = scenario
        self.report_dir = Path(report_dir)
        self.step_timeout = step_timeout
        self.stop_timeout = stop_timeout
        self.sleep = sleep
        self.clock = clock
        self._stop = threading.Event()
        self._running = True  # 建好就算在回放（面板建好马上交给后台线程跑），run() 结束才变假
        self._step = 0
        self._after = 0
        self._limit = ""  # 最近一次 /state 说的额度状态
        self.report: Path | None = None

    def stop_requested(self) -> None:
        """当前这一步做完就停（沙盒保持运行，可以接着手动玩）。"""
        self._stop.set()

    def progress(self) -> dict:
        return {"running": self._running, "step": self._step, "total": len(self.scenario.steps), "name": self.scenario.name}

    # ---- 等 ----
    def _wait_stopped(self) -> bool:
        deadline = self.clock() + self.stop_timeout
        while self.api.running():
            if self.clock() >= deadline:
                return False
            self.sleep(POLL)
        return True

    def _collect(self, sink: list[dict]) -> dict | None:
        st = self.api.state(self._after)
        if st is None:
            return None
        sink += st.get("lines") or []
        self._after = st.get("version", self._after)  # 沙盒重启过时版本号从头来，它会从 0 给
        return st

    def _gave_up(self) -> bool:
        """点了停止回放、沙盒也已经下线（页面上手动下线）：别再等它安静 / 起来。"""
        return self._stop.is_set() and not self.api.running()

    def _wait_ready(self, sink: list[dict]) -> bool:
        """刚启动：等 /sandbox/state 能回话（身体建好了）。"""
        deadline = self.clock() + self.step_timeout
        self._after = 0
        while self._collect(sink) is None:
            if self.clock() >= deadline or self._gave_up():
                return False
            self.sleep(POLL)
        return True

    def _wait_idle(self, sink: list[dict]) -> bool:
        deadline = self.clock() + self.step_timeout
        while True:
            st = self._collect(sink)
            if st is not None:
                self._limit = st.get("limit") or ""
            if st is not None and st.get("idle"):
                return True
            if self.clock() >= deadline or self._gave_up():
                return False
            self.sleep(POLL)

    def _wait_note(self, section: dict, ok: bool, what: str) -> None:
        """等完一步之后往报告里写的话：被停止 / 超时 / 额度用完。"""
        if self._gave_up():
            section["notes"].append("回放被停止，沙盒已下线")
        elif not ok:
            section["notes"].append(what)
        if self._limit:
            section["notes"].append(self._limit)

    # ---- 跑 ----
    def run(self) -> Path:
        self._running = True
        sc = self.scenario
        started = time.time()
        sections: list[dict] = []
        notes: list[str] = []
        diary_before = ""
        try:
            if self.api.running():
                self.api.stop()
                if not self._wait_stopped():
                    notes.append(f"沙盒 {self.stop_timeout:.0f} 秒没下线，接着往下跑")
            if sc.start.memory == "reset":
                self.api.reset()
            diary_before = self._diary()
            prep = {"title": "准备", "lines": [], "notes": []}
            sections.append(prep)
            self.api.start(sc.start.time or "resume")
            if not self._wait_ready(prep["lines"]):
                prep["notes"].append("沙盒没起来（超时）")
            for n in sc.start.nearby:
                self._op({"op": "come", "who": n}, prep)
            if sc.start.strangers:
                self._op({"op": "strangers", "n": sc.start.strangers}, prep)
            if sc.start.place:
                self._op({"op": "place", "name": sc.start.place}, prep)
            if sc.start.scene:
                self._op({"op": "scene", "text": sc.start.scene}, prep)
            if not self._wait_idle(prep["lines"]):
                prep["notes"].append(f"超时：{self.step_timeout:.0f} 秒还没安静")
            for i, step in enumerate(sc.steps, 1):
                self._step = i
                sections.append(self._do(i, step))
                if self._stop.is_set() and i < len(sc.steps):
                    notes.append(f"回放在第 {i} 步做完后停下了（点了停止回放）")
                    break
                if self._gave_up():  # 最后一步时被停了：也别再等
                    break
        except Exception as exc:  # 面板接口出错：报告照样写，写到出错为止
            log.exception("回放出错")
            notes.append(f"回放出错停下了：{exc}")
        finally:
            self._running = False
        self.report = self._write(started, sections, notes, diary_before)
        return self.report

    def _op(self, req: dict, section: dict) -> None:
        try:
            res = self.api.op(req)
        except Exception as exc:
            res = {"ok": False, "text": str(exc)}
        if not res.get("ok"):
            section["notes"].append(f"没做成：{res.get('text') or '沙盒没回应'}")

    def _do(self, i: int, step: Step) -> dict:
        section = {"title": f"第 {i} 步：{step_title(step)}", "lines": [], "notes": [], "expect": step.expect}
        if step.action == "offline":
            self.api.stop()
            if not self._wait_stopped():
                section["notes"].append(f"超时：{self.stop_timeout:.0f} 秒还没下线")
        elif step.action == "online":
            self.api.start(step.value)
            if not self._wait_ready(section["lines"]):
                self._wait_note(section, False, "沙盒没起来（超时）")
            else:
                self._wait_note(section, self._wait_idle(section["lines"]), f"超时：{self.step_timeout:.0f} 秒还没安静")
        else:
            self._op(to_op(step), section)
            self._wait_note(section, self._wait_idle(section["lines"]), f"超时：{self.step_timeout:.0f} 秒还没安静")
        if step.wait > 0 and step.action != "offline" and not self._gave_up():
            self.sleep(step.wait)
            self._collect(section["lines"])
        return section

    # ---- 报告 ----
    def _diary(self) -> str:
        try:
            return self.api.diary() or ""
        except Exception:
            log.exception("读沙盒日记出错")
            return ""

    def _inner_lines(self) -> list[str]:
        try:
            now = self.api.inner() or {}
        except Exception:
            log.exception("取沙盒的心里出错")
            return ["（取不到）"]
        out = []
        m = now.get("mood")
        if m:
            out.append(f"- 心情：{m.get('level')}" + (f"（{m.get('text')}）" if m.get("text") else ""))
        e = now.get("energy")
        if e:
            out.append(f"- 精力：{e.get('level')}" + (f"（{e.get('note')}）" if e.get("note") and e.get("note") != e.get("level") else ""))
        g = now.get("grudge")
        out.append(f"- 别扭：跟{g.get('who')}（{g.get('why') or '没说为什么'}）" if g else "- 别扭：没有")
        wants = now.get("wants") or []
        out.append("- 心愿：" + ("；".join(w.get("text", "") for w in wants) if wants else "没有"))
        for s in now.get("soft") or []:
            out.append(f"- 收着点：{s.get('who')}")
        return out or ["（没开反思）"]

    def _write(self, started: float, sections: list[dict], notes: list[str], diary_before: str) -> Path:
        sc = self.scenario
        st = sc.start
        out = [f"# 剧本回放：{sc.name}", ""]
        if sc.note:
            out += [f"> {sc.note}", ""]
        out.append(f"- 回放时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}（真实时间）")
        out.append(f"- 开头：记忆 {'先从 memory/ 重置' if st.memory == 'reset' else '用沙盒当时的'}；"
                   f"起始时间 {st.time or '接着上次'}；身边 {'、'.join(st.nearby) or '没人'}；陌生人 {st.strangers} 个"
                   + (f"；地名 {st.place}" if st.place else "") + (f"；场景 {st.scene}" if st.scene else ""))
        if st.memory == "keep":
            out.append(f"- 注意：{KEEP_NOTE}")
        for n in notes:
            out.append(f"- **{n}**")
        for sec in sections:
            out += ["", f"## {sec['title']}"]
            if sec.get("expect"):
                out.append(f"期望：{sec['expect']}（这一版不检查，自己看）")
            for n in sec["notes"]:
                out.append(f"**{n}**")
            out.append("")
            out += [f"- {line_text(r)}" for r in sec["lines"]] or ["（没有新的聊天记录）"]
        out += ["", "## 最后的心里", "", *self._inner_lines()]
        diary = self._diary()
        added = diary[len(diary_before):] if diary.startswith(diary_before) else diary
        out += ["", "## 这次写的日记", "", added.strip() or "（这次没写日记：没下线过，或者没开反思）", ""]
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = sc.name if safe_name(sc.name) else "剧本"
        path = self.report_dir / f"{stem}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}.md"
        i = 1
        while path.exists():
            path = self.report_dir / f"{stem}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}-{i}.md"
            i += 1
        path.write_text("\n".join(out), encoding="utf-8")
        log.info("回放报告：%s", path)
        return path
