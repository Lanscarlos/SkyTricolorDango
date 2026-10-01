# 管理面板「真机团子」页改版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把管理面板 `#live` 页从「iframe 嵌识别网页 + 卡片 + 日志」改成和沙盒页一样的顶栏 + 三栏（团子 / 画面 + 大脑 / 聊天记录），并给真机加一份聊天记录。

**Architecture:** 后端：`Transcript` 挪到 `brain/`、加长轮询；身体加 `on_line` 回调、事件队列加 `tap`，真机组装时把它们接进一个 `Transcript`，viewer 用 `/chat` 提供。前端：画框代码抽成 viewer 和面板共用的 `vision/static/stage.js`；聊天行渲染抽成 `chatlog.js`（沙盒共用）；手动控制在 `livectl.js`；`live.js` 重写成三栏。

**Tech Stack:** Python 3.13 标准库 HTTP 服务、原生 JS（无构建）、pytest + node（`node --check`、`require` 测纯函数）。

**Spec:** `docs/superpowers/specs/2026-10-01-console-live-page-design.md`

## Global Constraints

- 用中文写注释、界面文字和提交说明；提交说明末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- 测试命令：`python -m pytest -q`（这台机器没有 `.venv`，用系统 Python；`tests/conftest.py` 会把 worktree 的 `src` 放最前面）。单个文件 `python -m pytest -q tests/<文件>`
- 管理面板页面：不许原生 `confirm(` / `prompt(` / `alert(`（用 `ask()` / `toast()`）；不引外部资源、请求一律相对路径；`console.css` 的颜色只能在 `:root` 里定义，别处 `var(--…)`（`tests/test_console_page.py` 有测试查）
- 数据（聊天原话、大脑的话、识别标签）只用 `textContent` / `fillText`，不拼 HTML
- 顶层不碰 `document` 的脚本才能在 node 里 `require` 测纯函数：`stage.js`、`chatlog.js`、`livectl.js` 都照 `brainlog.js` 的写法（`globalThis.X = …`，`if (typeof module !== "undefined" && module.exports) module.exports = …`）
- `/live/*` 转发超时 5 秒（`console/server.py` 的 `PROXY_TIMEOUT`）：viewer 的长轮询最多等 `viewer.WAIT`（2 秒）
- 不碰正在跑的团子：所有改动只在这个 worktree（`.claude/worktrees/console-live-page`，分支 `worktree-console-live-page`）

---

### Task 1: `Transcript` 挪到 `brain/` + 长轮询 + `event_line`

**Files:**
- Create: `src/skydango/brain/transcript.py`
- Delete: `src/skydango/sandbox/transcript.py`
- Modify: `src/skydango/sandbox/world.py:25`、`src/skydango/cli.py:1425`、`tests/test_sandbox_control.py:23`、`tests/test_sandbox_world.py:17`
- Test: `tests/test_brain_transcript.py`

**Interfaces:**
- Produces: `skydango.brain.transcript.Transcript(clock, limit=500)`（`clock` 有 `wall()`），方法 `add(kind, text, who="", why="") -> dict`、`since(seq) -> list[dict]`、`wait_since(seq, timeout) -> tuple[int, list[dict]]`；`KINDS`；`event_line(kind: str, text: str, who: str = "") -> str | None`

- [ ] **Step 1: 写失败的测试** `tests/test_brain_transcript.py`

```python
"""聊天记录（沙盒和真机共用，spec 2026-10-01-console-live-page §3.2）：长轮询、事件变成一行。"""
import threading
import time
from types import SimpleNamespace

from skydango.brain.transcript import Transcript, event_line


def tr():
    return Transcript(SimpleNamespace(wall=lambda: 1_790_000_000.0))


def test_wait_since_returns_new_lines_at_once():
    t = tr()
    t.add("heard", "在吗", "小明")
    v, lines = t.wait_since(0, 5.0)
    assert v == 1 and [r["text"] for r in lines] == ["在吗"]


def test_wait_since_times_out_empty():
    t = tr()
    t.add("heard", "在吗", "小明")
    started = time.monotonic()
    v, lines = t.wait_since(1, 0.2)
    assert v == 1 and lines == [] and time.monotonic() - started >= 0.15


def test_wait_since_wakes_on_add():
    t = tr()
    threading.Timer(0.1, lambda: t.add("said", "在呢", "团子")).start()
    v, lines = t.wait_since(0, 3.0)
    assert v == 1 and lines[0]["kind"] == "said"


def test_wait_since_restarts_when_client_is_ahead():  # 浏览器记的比这边大：这边重启过，从头给
    t = tr()
    t.add("heard", "一", "小明")
    v, lines = t.wait_since(99, 0.0)
    assert v == 1 and len(lines) == 1


def test_event_line():
    assert event_line("arrive", "小明 来到身边（第 3 次见）", "小明") == "── 小明 来到身边 ──"
    assert event_line("leave", "小明 走开了（20 秒没看到名字）", "小明") == "── 小明 走开了 ──"
    assert event_line("return", "小明 回来了", "小明") == "── 小明 回来了 ──"
    assert event_line("lull", "小明 回来了（走开了 3 分钟，你刚才在想：…）", "小明") == "── 小明 回来了 ──"
    assert event_line("lull", "冷场  小明 1 分钟没说话了。…") is None  # 冷场节点不记（心里想的另有一行）
    assert event_line("holding", "（推测）牵上了 小明 的手") == "── （推测）牵上了 小明 的手 ──"
    assert event_line("released", "（推测）和 小明 松手了") == "── （推测）和 小明 松手了 ──"
    assert event_line("scene_change", "画面整屏黑了（可能在切场景）") == "── 画面黑了（可能在切场景） ──"
    assert event_line("scene_change", "画面恢复了") == "── 画面恢复了 ──"
    assert event_line("scene_change", "画面变化很大（换了地方或者镜头动了）") is None
    assert event_line("stranger_back", "刚才那个陌生人A又回来了") == "── 刚才那个陌生人A又回来了 ──"
    assert event_line("chat", "聊天  小明：「在吗」", "") is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_transcript.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'skydango.brain.transcript'`

- [ ] **Step 3: 新建 `src/skydango/brain/transcript.py`**（`git mv` 再改，保留历史）

Run: `git mv src/skydango/sandbox/transcript.py src/skydango/brain/transcript.py`

然后把文件改成：

```python
"""聊天记录（沙盒 brain-sandbox spec §3；真机 console-live-page spec §3.2）：内存里最近 limit 行，每行 {seq, t, kind, who, text}。

kind：heard（听到的 / 冒充的发言）/ said（团子说的）/ act（动作、走路、气泡）/ event（来了走了、快进、反思、心里……）/ blocked（被护栏拦下的话）。
`cond` 在 add 时 notify_all，给沙盒 /state 和 viewer /chat 长轮询用。
"""

from __future__ import annotations

import threading
from collections import deque

KINDS = ("heard", "said", "act", "event", "blocked")


class Transcript:
    def __init__(self, clock, limit: int = 500) -> None:
        self.clock = clock  # 有 wall() 的东西：沙盒是 SimClock（沙盒墙上时间），真机是包着身体墙钟的小对象
        self.cond = threading.Condition()
        self.version = 0
        self._rows: deque[dict] = deque(maxlen=limit)

    def add(self, kind: str, text: str, who: str = "", why: str = "") -> dict:
        """why：blocked 行被拦下的原因（有才带这个键）。"""
        if kind not in KINDS:
            raise ValueError(f"不认识的聊天记录类型：{kind}")
        with self.cond:
            self.version += 1
            row = {"seq": self.version, "t": self.clock.wall(), "kind": kind, "who": who, "text": text}
            if why:
                row["why"] = why
            self._rows.append(row)
            self.cond.notify_all()
        return row

    def since(self, seq: int) -> list[dict]:
        """seq 之后的行（最多 limit 行，更早的已经丢了）。"""
        with self.cond:
            return [dict(r) for r in self._rows if r["seq"] > seq]

    def wait_since(self, seq: int, timeout: float) -> tuple[int, list[dict]]:
        """长轮询：seq 之后有新行马上返回，没有就最多等 timeout 秒。seq 比最新的还大（这边重启过）当 0。返回 (最新 seq, 新行)。"""
        with self.cond:
            if seq > self.version:
                seq = 0
            self.cond.wait_for(lambda: self.version > seq, max(0.0, timeout))
            return self.version, [dict(r) for r in self._rows if r["seq"] > seq]


def event_line(kind: str, text: str, who: str = "") -> str | None:
    """真机聊天记录里的分隔线：身体放给大脑的事件里，值得在聊天旁边看到的几种（来去、牵手、黑屏）；别的返回 None。"""
    if kind == "arrive":
        return f"── {who or text} 来到身边 ──"
    if kind == "leave":
        return f"── {who} 走开了 ──" if who else f"── {text} ──"
    if kind == "return" or (kind == "lull" and who):  # lull 带 who = 冷场后他回来了
        return f"── {who} 回来了 ──" if who else f"── {text} ──"
    if kind in ("holding", "released", "stranger_back"):
        return f"── {text} ──"
    if kind == "scene_change":
        if text.startswith("画面整屏黑了"):
            return "── 画面黑了（可能在切场景） ──"
        if text.startswith("画面恢复了"):
            return "── 画面恢复了 ──"
    return None
```

- [ ] **Step 4: 改 import**

- `src/skydango/sandbox/world.py:25`：`from .transcript import Transcript` → `from ..brain.transcript import Transcript`
- `src/skydango/cli.py:1425`：`from .sandbox.transcript import Transcript` → `from .brain.transcript import Transcript`
- `tests/test_sandbox_control.py:23`、`tests/test_sandbox_world.py:17`：`from skydango.sandbox.transcript import Transcript` → `from skydango.brain.transcript import Transcript`

Run: `grep -rn "sandbox.transcript\|sandbox import transcript\|from .transcript" src tests`
Expected: 没有输出

- [ ] **Step 5: 跑测试**

Run: `python -m pytest -q tests/test_brain_transcript.py tests/test_sandbox_control.py tests/test_sandbox_world.py tests/test_sandbox_server.py tests/test_cli_sandbox.py`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add -A src/skydango/brain/transcript.py src/skydango/sandbox src/skydango/cli.py tests/test_brain_transcript.py tests/test_sandbox_control.py tests/test_sandbox_world.py
git commit -m "refactor(brain): 聊天记录挪到 brain/，加长轮询和事件分隔线

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 身体 `on_line` + 事件队列 `tap`

**Files:**
- Modify: `src/skydango/brain/events.py`（`EventQueue.__init__`、新方法 `tap`、`put` 末尾）
- Modify: `src/skydango/brain/body.py`（`__init__` 里 `on_blocked` 旁边、新方法 `_line`、处理新消息处、`say`、`_blocked`、`emote`、`_reflex_emote`、`_open_bubble`、`mused`）
- Test: `tests/test_brain_body_lines.py`

**Interfaces:**
- Consumes: 无
- Produces: `Body.on_line: Callable[[str, str, str, str], None] | None`（参数 kind, text, who, why）；`EventQueue.tap(fn: Callable[[str, str, str], None]) -> None`

- [ ] **Step 1: 写失败的测试** `tests/test_brain_body_lines.py`

```python
"""身体把听到 / 说 / 做 / 被拦 / 心里想的交给 on_line（真机聊天记录，spec 2026-10-01-console-live-page §3.2）；事件队列的 tap。"""
import pytest
from test_brain_body import FakeEmotes, body, msg
from test_brain_lull_body import chat_then_quiet, lb
from test_brain_reflex_body import OPEN, rx

from skydango.brain.body import ToolError
from skydango.brain.events import EventQueue


def collect(b):
    got = []
    b.on_line = lambda kind, text, who, why: got.append((kind, text, who, why))
    return got


def test_heard_lines(clock):
    b, _, reader, _ = body(clock)
    got = collect(b)
    reader.batches = [[msg("在吗"), msg("hi", speaker="")]]
    b.step()
    assert got == [("heard", "在吗", "懒洋洋大王", ""), ("heard", "hi", "（看不出是谁）", "")]


def test_said_live_and_dry_run(clock):
    b, _, _, _ = body(clock, live=True)
    got = collect(b)
    b.say("在呢")
    assert got == [("said", "在呢", "团子", "")]
    b2, _, _, _ = body(clock)  # 默认 dry-run
    got2 = collect(b2)
    b2.say("在呢")
    assert got2 == [("said", "在呢（dry-run，没真的发）", "团子", "")]


def test_blocked_line(clock):
    b, _, _, _ = body(clock, live=True)
    got = collect(b)
    with pytest.raises(ToolError):
        b.say("我是真人")
    assert got[0][0] == "blocked" and got[0][1] == "我是真人" and got[0][2] == "团子" and got[0][3]


def test_emote_line(clock):
    b, _, _, _ = body(clock, live=True, emotes=FakeEmotes())
    got = collect(b)
    b.emote("鞠躬")
    assert got == [("act", "（团子做了 鞠躬）", "团子", "")]


def test_reflex_and_bubble_lines(clock):
    b, dev, reader, events, emotes = rx(clock, addressed=["点头"])
    got = collect(b)
    reader.batches = [[msg("团子", speaker="小明")]]
    b.step()
    kinds = [(k, t) for k, t, _, _ in got]
    assert ("heard", "团子") in kinds
    assert ("act", "（团子下意识地 点头）") in kinds and ("act", "（团子头顶冒出输入气泡）") in kinds
    assert dev.calls.count(OPEN) == 1


def test_musing_line(clock):
    b, env, reader, events = lb(clock)
    got = collect(b)
    chat_then_quiet(b, reader, events, clock)
    got.clear()
    b.mused("心里：他是不是去忙了")
    assert got == [("event", "── 心里：他是不是去忙了 ──", "", "")]


def test_on_line_errors_do_not_break_body(clock):
    b, _, reader, events = body(clock)
    b.on_line = lambda *a: 1 / 0
    reader.batches = [[msg("在吗")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]


def test_tap_sees_every_put(clock):
    q = EventQueue(clock=clock)
    seen = []
    q.tap(lambda kind, text, who: seen.append((kind, text, who)))
    q.put("arrive", "小明 来到身边", who="小明")
    q.put("leave", "小明 走开了", who="小明")  # 和 arrive 不抵消，照样调
    q.put("scene_change", "画面恢复了")
    q.put("scene_change", "画面恢复了")  # 合并了也调
    assert seen == [("arrive", "小明 来到身边", "小明"), ("leave", "小明 走开了", "小明"),
                    ("scene_change", "画面恢复了", ""), ("scene_change", "画面恢复了", "")]


def test_tap_errors_are_swallowed(clock):
    q = EventQueue(clock=clock)
    q.tap(lambda *a: 1 / 0)
    q.put("chat", "x")
    assert len(q) == 1
```

（`OPEN = ("hw_key", 28)` 是 `tests/test_brain_reflex_body.py` 第 9 行定义的：开输入框的按键记录。）

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_body_lines.py`
Expected: FAIL（`on_line` 没被调用：`assert [] == [...]`；`EventQueue` 没有 `tap`）

- [ ] **Step 3: `events.py` 加 `tap`**

`EventQueue.__init__` 里 `self._listeners` 下面加一行：

```python
        self._taps: list[Callable[[str, str, str], None]] = []  # 每个事件的全文（真机聊天记录）
```

`subscribe` 下面加方法：

```python
    def tap(self, fn: Callable[[str, str, str], None]) -> None:
        """每放一个事件就调 fn(kind, text, who)（合并 / 抵消了也调，在放事件的线程里调）；真机聊天记录用它记来去。"""
        self._taps.append(fn)
```

`put` 最后，`for fn in list(self._listeners):` 那段后面加：

```python
        for tap in list(self._taps):
            try:
                tap(kind, text, who)
            except Exception:
                log.exception("事件 tap 出错")
```

- [ ] **Step 4: `body.py` 加 `on_line` 和 `_line`**

`__init__` 里 `self.on_blocked: … = None` 那一行下面加：

```python
        self.on_line: Callable[[str, str, str, str], None] | None = None  # 真机聊天记录：(kind, text, who, why)；沙盒不接（沙盒在 world / control 里记）
```

`_blocked` 方法改成（先记聊天记录，再照旧调 `on_blocked`）：

```python
    def _blocked(self, text: str, why: str) -> None:
        self._line("blocked", text, "团子", why)
        if self.on_blocked is None:
            return
        try:
            self.on_blocked(text, why)
        except Exception:
            log.exception("on_blocked 出错")

    def _line(self, kind: str, text: str, who: str = "", why: str = "") -> None:
        """交给真机聊天记录；出错只记日志。"""
        if self.on_line is None:
            return
        try:
            self.on_line(kind, text, who, why)
        except Exception:
            log.exception("on_line 出错")
```

- [ ] **Step 5: 各调用点**

1. 处理新消息处：`self.heard = (self.heard + fresh)[-20:]` 这一行后面加

```python
        for m in fresh:
            self._line("heard", m.text, m.speaker or "（看不出是谁）")
```

2. `say`：dry-run 分支的 `return f"dry-run：没真的发…"` 前一行加 `self._line("said", full + "（dry-run，没真的发）", "团子")`；
   真发的 `self.sender.send(full)` 后面加 `self._line("said", full, "团子")`

3. `emote`：dry-run 分支 `return f"dry-run：没真的做「{name}」{note}"` 前一行加 `self._line("act", f"（团子做了 {name}）", "团子")`；
   真做的 `try: … perform(name) except …` 之后、`return f"做了「{name}」" …` 之前加同一行

4. `_reflex_emote`：`self.emoted.append(name)` 前一行加 `self._line("act", f"（团子下意识地 {name}）", "团子")`

5. `_open_bubble`：`if opened:` 块里 `log.info("有人在跟团子说话：先冒输入气泡")` 后面加 `self._line("act", "（团子头顶冒出输入气泡）", "团子")`

6. `mused`：`log.info("心里：%s", thought)` 后面加 `self._line("event", f"── 心里：{thought} ──")`

- [ ] **Step 6: 跑测试**

Run: `python -m pytest -q tests/test_brain_body_lines.py tests/test_brain_body.py tests/test_brain_events.py tests/test_brain_reflex_body.py tests/test_brain_lull_body.py`
Expected: 全部 PASS（`tests/test_brain_events.py` 不存在就去掉它）

- [ ] **Step 7: 提交**

```bash
git add src/skydango/brain/events.py src/skydango/brain/body.py tests/test_brain_body_lines.py
git commit -m "feat(brain): 身体把听到、说、做、被拦、心里想的交给 on_line；事件队列加 tap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: viewer `/chat` 长轮询

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`JsonHandler` 加 `_wait`；`Viewer.__init__` 加 `self.chat`；`do_GET` 加 `/chat`）
- Modify: `src/skydango/sandbox/server.py:115-120`（删掉自己的 `_wait`，用 `JsonHandler` 的）
- Test: `tests/test_viewer.py`（末尾加）

**Interfaces:**
- Consumes: `Transcript.wait_since(seq, timeout)`（Task 1）
- Produces: `Viewer.chat`（`Transcript | None`）；`GET /chat?after=<seq>&wait=<秒>` → `{"v": int, "lines": [...]}`，没挂 404，Host 不对 403；`JsonHandler._wait(url, cap=2.0) -> float`

- [ ] **Step 1: 写失败的测试**（加在 `tests/test_viewer.py` 末尾）

```python
# ---- 真机聊天记录（spec 2026-10-01-console-live-page §3.2）----
def test_chat_endpoint_404_without_transcript():
    v = viewer()
    url = v.start()
    try:
        assert request(url + "chat?after=0")[0] == 404
    finally:
        v.stop()


def test_chat_endpoint_long_polls():
    from types import SimpleNamespace

    from skydango.brain.transcript import Transcript

    v = viewer()
    v.chat = Transcript(SimpleNamespace(wall=lambda: 1_790_000_000.0))
    v.chat.add("heard", "在吗", "小明")
    url = v.start()
    port = url.rstrip("/").rsplit(":", 1)[1]
    try:
        status, body = request(url + "chat?after=0&wait=2")
        assert status == 200 and body["v"] == 1 and body["lines"][0]["text"] == "在吗"
        status, body = request(url + "chat?after=1&wait=0.1")  # 没有新的：等一下回空
        assert status == 200 and body == {"v": 1, "lines": []}
        status, body = request(url + "chat?after=9&wait=0")  # 浏览器记的比这边大：从头给
        assert len(body["lines"]) == 1
        assert request(url + "chat?after=0", None, {"Host": f"evil.example:{port}"})[0] == 403
    finally:
        v.stop()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_viewer.py -k chat_endpoint`
Expected: FAIL（`test_chat_endpoint_long_polls` 拿到 404）

- [ ] **Step 3: 实现**

`JsonHandler` 里 `_after` 下面加（和沙盒原来的一样）：

```python
    def _wait(self, url, cap: float = 2.0) -> float:
        """长轮询最多等几秒：query 里的 wait，夹到 [0, cap]；坏值用 cap。"""
        try:
            value = float(parse_qs(url.query).get("wait", [cap])[0])
        except ValueError:
            return cap
        return max(0.0, min(cap, value)) if value == value else cap
```

删掉 `src/skydango/sandbox/server.py` 里 `Handler` 自己的 `_wait`（115~120 行，签名、行为一样，调用方 `self._wait(url, STATE_WAIT)` 不用改）。

`Viewer.__init__` 里 `self.control = None …` 下面加：

```python
        self.chat = None  # brain.transcript.Transcript：真机聊天记录（管理面板真机页用），只在大脑真机模式挂
```

`do_GET` 里 `/brain` 分支前面加：

```python
                elif url.path == "/chat" and viewer.chat is not None:
                    if not self._local_host():
                        self._json(403, {"ok": False, "text": "只接受本机地址（Host 不对）"})
                    else:
                        version, lines = viewer.chat.wait_since(self._after(url), self._wait(url, WAIT))
                        self._json(200, {"v": version, "lines": lines})
```

- [ ] **Step 4: 跑测试**

Run: `python -m pytest -q tests/test_viewer.py tests/test_sandbox_server.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/viewer.py src/skydango/sandbox/server.py tests/test_viewer.py
git commit -m "feat(viewer): /chat 长轮询真机聊天记录

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 真机组装时接上聊天记录

**Files:**
- Modify: `src/skydango/cli.py`（`_run_brain` 里 `if viewer is not None:` 那一块，约 1777 行）
- Test: `tests/test_cli_brain.py`

**Interfaces:**
- Consumes: `Transcript`、`event_line`（Task 1）；`Body.on_line`、`EventQueue.tap`（Task 2）；`Viewer.chat`（Task 3）
- Produces: 真机大脑模式带 viewer 时 `viewer.chat` 是一个 `Transcript`

- [ ] **Step 1: 写失败的测试**（`tests/test_cli_brain.py`）

`FakeViewer` 加一个类属性 `chat = None`，然后在 `test_run_brain_on_lan_has_no_control` 后面加：

```python
def test_run_brain_with_viewer_attaches_chat(tmp_path, monkeypatch):
    from skydango.brain.transcript import Transcript

    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert isinstance(v.chat, Transcript)


def test_run_brain_on_lan_has_no_chat(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.viewer.host = "0.0.0.0"
    v = FakeViewer()
    cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    assert v.chat is None  # 聊天原话只给本机看
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_cli_brain.py -k chat`
Expected: `test_run_brain_with_viewer_attaches_chat` FAIL（`v.chat` 是 None）

- [ ] **Step 3: 实现**

`_run_brain` 里 `if cfg.viewer.host in LOCAL_HOSTS:` 块的最后（`viewer.forget = …` 后面）加：

```python
            if world.name != "sandbox":  # 真机聊天记录（spec 2026-10-01-console-live-page §3.2）：沙盒有自己的
                from types import SimpleNamespace

                from .brain.transcript import Transcript, event_line

                chat = Transcript(SimpleNamespace(wall=wall))
                body.on_line = lambda kind, text, who, why: chat.add(kind, text, who, why=why)

                def _event(kind: str, text: str, who: str) -> None:
                    line = event_line(kind, text, who)
                    if line is not None:
                        chat.add("event", line)

                events.tap(_event)
                viewer.chat = chat
```

（`wall`、`events`、`body`、`world` 在这个作用域里都已经有了；沙盒走 `_run_brain` 时 viewer 一般是 None，这个判断是保险。）

- [ ] **Step 4: 跑测试**

Run: `python -m pytest -q tests/test_cli_brain.py tests/test_cli_sandbox.py tests/test_sandbox_fix_clock.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/cli.py tests/test_cli_brain.py
git commit -m "feat(brain): 真机大脑模式把聊天记录挂到 viewer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 画框代码抽成 `vision/static/stage.js`

**Files:**
- Create: `src/skydango/vision/static/stage.js`
- Modify: `src/skydango/vision/viewer.py`（`STATIC`、`PAGE`）
- Test: `tests/test_viewer.py`（改 3 个读 PAGE 的测试、加 2 个）

**Interfaces:**
- Produces: 全局 `Stage = {COLORS, NAMES, nameAt(boxes, x, y) -> string|null, toFrame(clientX, clientY, rect, width, height) -> [x, y], draw(canvas, img, snapshot, {boxes?: bool, mark?: [x,y]|null, hover?: [px,py]|null})}`；`module.exports = Stage`；viewer 和管理面板都从 `static/stage.js` 取（viewer 是 `/stage.js`，面板是 `/static/stage.js`）

- [ ] **Step 1: 改 / 加测试**（`tests/test_viewer.py`）

加一个读 stage.js 的小工具（放在 `_trace_js` 旁边）：

```python
def _stage_path():
    import importlib.resources

    return importlib.resources.files("skydango.vision") / "static" / "stage.js"
```

`test_page_has_typing_colour_and_legend` 改成：

```python
def test_page_has_typing_colour_and_legend():
    stage = _stage_path().read_text(encoding="utf-8")
    assert 'typing:"#e879f9"' in stage and 'typing:"正在输入"' in stage
```

`test_click_maps_to_frame_pixels` 改成：

```python
def test_click_maps_to_frame_pixels():
    # 画布被 CSS 缩放显示：按显示尺寸换算回原图像素
    out = _node(f"const S=require({json.dumps(str(_stage_path()))});"
                "console.log(JSON.stringify(S.toFrame(650, 400, {left:50, top:100, width:960, height:540}, 1920, 1080)));")
    assert json.loads(out) == [1200, 600]
```

`test_click_picks_friend_name` 里把 `[fn] = …` 那行删掉，`script = fn + f"\nconst B=…"` 换成：

```python
    script = f"const S=require({json.dumps(str(_stage_path()))});const nameAt=S.nameAt;" + \
        f"const B={json.dumps(boxes, ensure_ascii=False)};" + \
        "console.log(JSON.stringify([nameAt(B,150,300),nameAt(B,980,450),nameAt(B,1550,400),nameAt(B,600,600)]));"
```

再加两个：

```python
def test_page_uses_shared_stage_script():
    assert 'src="stage.js"' in PAGE and "Stage.draw(" in PAGE
    for inline in ("function nameAt(", "function toFrame(", "function drawHover(", "const COLORS="):
        assert inline not in PAGE  # 不再内联一份


def test_viewer_serves_stage_script():
    v = viewer()
    url = v.start()
    try:
        with urllib.request.urlopen(url + "stage.js", timeout=5) as r:
            assert r.status == 200 and "javascript" in r.headers["Content-Type"] and "function nameAt" in r.read().decode("utf-8")
    finally:
        v.stop()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_viewer.py`
Expected: 上面 5 个 FAIL（stage.js 不存在 / PAGE 里还有内联）

- [ ] **Step 3: 新建 `src/skydango/vision/static/stage.js`**

```js
/* 识别画面：画图 + 识别框 + 标签 + 十字 + 悬停描述（viewer 网页和管理面板真机页共用，spec 2026-10-01-console-live-page §3.1）。
 * 快照格式见 vision/viewer.py 的 _render：{seq, width, height, image, boxes:[{x,y,w,h,kind,label?,score?,desc?}], info}。
 * 标签、描述来自识别 / 模型：只用 fillText 画。顶层不碰 document：node 里 require 它能测 nameAt / toFrame（tests/test_viewer.py）。 */
(function () {
"use strict";
const COLORS={friend:"#3ddc84",name:"#3ddc84",tag:"#facc15",stranger:"#ff9f43",unlit:"#a78bfa",player:"#60a5fa",self:"#cbd5e1",maybe:"#86efac",
ring:"#22d3ee",request:"#f43f5e",panel:"#6b7280",message:"#f472b6",typing:"#e879f9",
bench:"#1d4ed8",bonfire:"#ea580c",instrument:"#fda4af",spirit:"#ffffff",
panel_ok:"#3b82f6",panel_new:"#facc15",panel_unknown:"#ef4444",button_ok:"#22c55e",button_ask:"#9ca3af",button_never:"#dc2626"};
const NAMES={friend:"好友",tag:"没认出的名字",stranger:"陌生人",unlit:"没点火",player:"没判定的人",self:"团子",maybe:"按外观认的好友",ring:"互动圆圈",
request:"互动请求",panel:"聊天面板",message:"新消息",typing:"正在输入",
bench:"座位",bonfire:"篝火",instrument:"乐器",spirit:"先祖",
panel_ok:"面板（已核对）",panel_new:"面板（未核对）",panel_unknown:"不认识的面板",button_ok:"能按",button_ask:"要放行",button_never:"不能按"};

/* 在画面上点人：好友框，或名字标签往下一块（宽 3 倍、连标签 7 倍高，同身体 _below_tag）；重叠时取面积最小的 */
function nameAt(boxes,x,y){let best=null,area=Infinity;for(const b of boxes){if(!b.label||(b.kind!=="friend"&&b.kind!=="name"))continue;const tag=b.kind==="name",x1=tag?b.x-b.w:b.x,w=tag?b.w*3:b.w,h=tag?b.h*7:b.h;if(x<x1||x>=x1+w||y<b.y||y>=b.y+h)continue;if(w*h<area){best=b.label;area=w*h}}return best}
/* 画布被 CSS 缩放显示：按显示尺寸换算回原图像素 */
function toFrame(clientX,clientY,rect,width,height){return [Math.round((clientX-rect.left)*width/rect.width),Math.round((clientY-rect.top)*height/rect.height)]}

/* o.boxes = false 只画图（十字照画）；o.mark = 原图坐标的十字；o.hover = 画布像素坐标，落在有 desc 的框里时在框下方写一行 */
function draw(canvas,img,s,o){
  o=o||{};const ctx=canvas.getContext("2d");
  canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);
  const k=canvas.width/s.width,fs=Math.max(12,Math.round(canvas.width/80));
  if(o.mark){const x=o.mark[0]*k,y=o.mark[1]*k,r=Math.max(12,canvas.width/60);
    ctx.strokeStyle="#f472b6";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-r,y);ctx.lineTo(x+r,y);ctx.moveTo(x,y-r);ctx.lineTo(x,y+r);ctx.stroke()}
  if(o.boxes===false)return;
  ctx.font=`${fs}px system-ui,"Microsoft YaHei",sans-serif`;ctx.textBaseline="middle";
  const boxes=s.boxes||[];
  for(const b of boxes){const col=COLORS[b.kind]||"#fff",x=b.x*k,y=b.y*k,w=b.w*k,h=b.h*k;
    ctx.strokeStyle=col;ctx.lineWidth=b.kind==="request"?4:2;ctx.setLineDash(b.kind.startsWith("panel")?[8,5]:b.kind==="maybe"?[6,4]:[]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
    const t=(b.label||"")+(b.score!==undefined?` ${b.score.toFixed(2)}`:"");if(!t)continue;
    const below=b.kind==="ring"||b.kind==="request",tw=ctx.measureText(t).width+8,th=fs+6,ty=(below||y-th<0)?y+h:y-th;
    ctx.fillStyle=col;ctx.fillRect(x,ty,tw,th);ctx.fillStyle="#0b0d12";ctx.fillText(t,x+4,ty+th/2);}
  if(!o.hover)return;
  let hit=null;
  for(const b of boxes){if(!b.desc)continue;const x=b.x*k,y=b.y*k;if(o.hover[0]>=x&&o.hover[0]<=x+b.w*k&&o.hover[1]>=y&&o.hover[1]<=y+b.h*k)hit=b}
  if(!hit)return;const t=String(hit.desc),tw=ctx.measureText(t).width+8,th=fs+6,x=Math.min(hit.x*k,canvas.width-tw),y=Math.min((hit.y+hit.h)*k,canvas.height-th);
  ctx.fillStyle="rgba(11,13,18,.85)";ctx.fillRect(x,y,tw,th);ctx.fillStyle="#e6e8ee";ctx.fillText(t,x+4,y+th/2)}

const Stage={COLORS,NAMES,nameAt,toFrame,draw};
globalThis.Stage=Stage;
if(typeof module!=="undefined"&&module.exports)module.exports=Stage;
})();
```

- [ ] **Step 4: 改 viewer**

`STATIC = ("brain_trace.js", "brain_trace.css")` → `STATIC = ("brain_trace.js", "brain_trace.css", "stage.js")`，注释改成「vision/static/ 里给页面用的共用文件（大脑时间线、画框）」。

`PAGE` 里：
1. `<link rel="stylesheet" href="brain_trace.css"><script src="brain_trace.js"></script>` 后面加 `<script src="stage.js"></script>`
2. 删掉 `const COLORS={…};` 和 `const NAMES={…};` 两段（4+4 行）
3. `$("legend").innerHTML=Object.entries(NAMES).map(([k,v])=>`<span><i style="background:${COLORS[k]}"></i>${v}</span>`).join("");` 改成
   `$("legend").innerHTML=Object.entries(Stage.NAMES).map(([k,v])=>`<span><i style="background:${Stage.COLORS[k]}"></i>${v}</span>`).join("");`
4. 整个 `function draw(s){…}`（到 `drawHover(s,k,fs);\n}`）、`// 鼠标悬停…` 注释、`let hover=null;`、`function drawHover(…){…}` 换成：

```js
let hover=null;  // 鼠标在画布上的位置（画布像素）：落在有 desc 的框里时写一行装扮描述
function draw(s){Stage.draw(c,img,s,{boxes:showBoxes,mark:K.mark,hover})}
```

5. 删掉 control 段里的 `function nameAt(…){…}`、`function toFrame(…){…}`、`function drawMark(s){…}` 三行；
   两处 `toFrame(e.clientX,…)` 改成 `Stage.toFrame(e.clientX,…)`，`nameAt(last.boxes||[],x,y)` 改成 `Stage.nameAt(last.boxes||[],x,y)`

（`K` 是后面 `const K=…` 定义的；`draw` 只在拿到快照之后才调用，那时 `K` 已经定义好了。）

- [ ] **Step 5: 跑测试**

Run: `python -m pytest -q tests/test_viewer.py tests/test_console_server.py`
Expected: 全部 PASS（`test_viewer_page_script_parses` 也要过：内联脚本还能解析）

- [ ] **Step 6: 手动看一眼 viewer**

Run: `python -m skydango view --images tmp/record --no-browser --port 19398`（`tmp/record` 下随便挑一个有截图的录像目录；没有就 `ls tmp` 找一个有 jpg / png 的目录）
然后用浏览器面板打开 `http://127.0.0.1:19398/`，确认框和标签、图例、悬停描述和改之前一样；看完 `taskkill /F /T /PID <pid>` 停掉。

- [ ] **Step 7: 提交**

```bash
git add src/skydango/vision/static/stage.js src/skydango/vision/viewer.py tests/test_viewer.py
git commit -m "refactor(viewer): 画框代码抽成 stage.js（管理面板真机页共用）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 聊天行渲染抽成 `chatlog.js`（沙盒照用）

**Files:**
- Create: `src/skydango/console/static/chatlog.js`
- Modify: `src/skydango/console/static/sandbox.js`（删 `lineKind`、`sbLine`、`sbAppend`，改用 `Chat`）、`src/skydango/console/static/console.html`（加 `<script>`）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: `common.js` 的 `el`、`hhmm`、`dayTime`（调用时才用）
- Produces: 全局 `Chat = {lineKind(row) -> string, line(row) -> Element, append(box: Element, rows: object[]) -> void}`；`module.exports = {lineKind}`

- [ ] **Step 1: 改测试**（`tests/test_console_page.py`）

`JS` 列表在 `"brainlog.js"` 后面加 `"chatlog.js"`：

```python
JS = ["common.js", "markdown.js", "inner.js", "brainlog.js", "chatlog.js", "sandbox.js", "live.js", "scenarios.js", "settings.js", "device.js"]
```

`test_skeleton` 里顺序断言改成：

```python
    assert page.index('src="console/static/common.js"') < page.index('src="console/static/brainlog.js"') < page.index('src="console/static/sandbox.js"')
    assert page.index('src="console/static/chatlog.js"') < page.index('src="console/static/sandbox.js"')
```

`test_sandbox_line_kinds` 改成测 `chatlog.js`：

```python
def _chatlog_js(expr: str):  # 在 node 里载入 chatlog.js（没有 document），算 expr
    node = shutil.which("node") or pytest.skip("没有 node")
    js = f"const C=require({json.dumps(str(STATIC / 'chatlog.js'))});console.log(JSON.stringify({expr}))"
    return json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def test_chat_line_kinds():  # 聊天行的样式：反思 = 虚线框，别的事件 = 分隔线；团子说的（sb-me）靠左；被拦的删除线
    rows = [{"kind": "event", "text": "── 反思：心情 平常 → 开心 ──"}, {"kind": "event", "text": "── 开始反思 ──"},
            {"kind": "event", "text": "── 小明来了 ──"}, {"kind": "heard", "who": "小明"}, {"kind": "said", "who": "团子"},
            {"kind": "act", "who": "团子"}, {"kind": "blocked", "who": "团子"}, {"kind": "blocked", "who": ""}]
    assert _chatlog_js(f"{json.dumps(rows, ensure_ascii=False)}.map(C.lineKind)") == [
        "refl", "ev", "ev", "msg", "msg me", "act", "msg me blocked", "msg blocked"]
```

（删掉旧的 `test_sandbox_line_kinds`；`_sandbox_js` 里 require 了 `sandbox.js`，它现在要 `Chat`：把 `_sandbox_js` 的 js 改成先 require `chatlog.js`：

```python
    js = (f"require({json.dumps(str(STATIC / 'common.js'))});require({json.dumps(str(STATIC / 'chatlog.js'))});"
          f"require({json.dumps(str(STATIC / 'sandbox.js'))});console.log(JSON.stringify({expr}))")
```
）

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_console_page.py`
Expected: FAIL（`chatlog.js` 不存在）

- [ ] **Step 3: 新建 `src/skydango/console/static/chatlog.js`**

```js
/* 聊天记录的一行（沙盒页和真机页共用，spec 2026-10-01-console-live-page §2.4）：别人说的在右、团子说的在左（樱花底）、
 * 动作是旁白、事件是分隔线、反思是虚线框、被拦的话删除线 + 原因。行的格式见 brain/transcript.py：{seq, t, kind, who, text, why?}。
 * 样式类名沿用沙盒的 sb-*（console.css）。只用 textContent 放数据。顶层不碰 document：node 里 require 能测 lineKind。 */
(function () {
"use strict";
function lineKind(r) {  // 聊天行用哪种样式
  if (r.kind === "event") return /^──\s*反思/.test(r.text || "") ? "refl" : "ev";
  if (r.kind === "act") return "act";
  if (r.kind === "said") return "msg me";
  if (r.kind === "blocked") return r.who === "团子" ? "msg me blocked" : "msg blocked";
  return "msg";
}
function chatLine(r) {
  const k = lineKind(r), time = el("time", "", hhmm(r.t));
  if (k === "ev") { const d = el("div", "sb-ev"); d.append(time, " ", (r.text || "").replace(/^──\s*|\s*──$/g, "")); return d; }
  if (k === "refl") { const d = el("div", "sb-refl"); d.append(time, " ", r.text || ""); return d; }
  if (k === "act") { const d = el("div", "sb-act", r.text || ""); d.title = dayTime(r.t); return d; }
  const d = el("div", "sb-" + k.split(" ").join(" sb-")), who = el("div", "who");
  who.append(r.kind === "heard" ? r.who || "（不知道是谁）" : r.kind === "said" ? "团子" : r.who === "团子" ? "团子（没说出去）" : "（被拦下）", " ", time);
  d.append(who, el("div", "b", r.text || ""));
  if (r.kind === "blocked" && r.why) d.append(el("div", "why", r.why));
  return d;
}
function appendChat(box, rows) {  // 原本在底部才跟到底；占位的 .none 第一次有内容时去掉；最多留 500 行
  if (!rows.length) return;
  const follow = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  for (const p of box.querySelectorAll(":scope > .none")) p.remove();
  for (const r of rows) box.append(chatLine(r));
  while (box.childNodes.length > 500) box.firstChild.remove();
  if (follow) box.scrollTop = box.scrollHeight;
}
globalThis.Chat = {lineKind, line: chatLine, append: appendChat};
if (typeof module !== "undefined" && module.exports) module.exports = {lineKind};
})();
```

- [ ] **Step 4: 沙盒改用 `Chat`**（`sandbox.js`）

1. 删掉 `function lineKind(r) {…}`（7 行）、`function sbLine(r) {…}`、`function sbAppend(lines) {…}`
2. `SB` 初始值里去掉 `rows: 0`；`clearChat` 改成 `function clearChat() { SB.v = 0; $("sb-chat").textContent = ""; }`
3. `sbApply` 里 `sbAppend(d.lines || [])` 改成 `Chat.append($("sb-chat"), d.lines || [])`
4. 文件末尾 `Object.assign(globalThis, {Sandbox: {lineKind, summaryText}, …})` 改成 `Sandbox: {summaryText}`
5. 文件头注释里「node 里 require 它能测 Sandbox.lineKind / summaryText」改成「… Sandbox.summaryText」
6. `grep -n "SB.rows\|sbAppend\|sbLine\|lineKind" src/skydango/console/static/sandbox.js` 确认没有残留

`console.html` 里 `<script src="console/static/brainlog.js"></script>` 后面加 `<script src="console/static/chatlog.js"></script>`。

- [ ] **Step 5: 跑测试**

Run: `python -m pytest -q tests/test_console_page.py tests/test_console_sandbox.py`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add src/skydango/console/static/chatlog.js src/skydango/console/static/sandbox.js src/skydango/console/static/console.html tests/test_console_page.py
git commit -m "refactor(console): 聊天行渲染抽成 chatlog.js（真机页共用）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 真机页骨架：顶栏、三栏、画面、大脑、聊天、日志抽屉

**Files:**
- Modify: `src/skydango/console/static/console.html`（`#page-live` 整段替换；`<script>` 加 `static/stage.js`）
- Modify: `src/skydango/console/static/console.css`（删旧的真机样式，加新的）
- Rewrite: `src/skydango/console/static/live.js`
- Test: `tests/test_console_page.py`（改 `test_live_page`、`test_skeleton`）

**Interfaces:**
- Consumes: `Stage`（Task 5，经 `static/stage.js`）、`Chat`（Task 6）、`mountBrainConsole`（`brainlog.js`）、`Inner.renderNow` / `Inner.setSource`（`inner.js`）、common.js 的 `$ el getJSON post problemList onState refresh go S BUSY fmtUptime toast`；接口 `live/snapshot` `live/status` `live/brain` `live/chat` `api/inner` `api/logs` `api/run/start` `api/orphan/stop`
- Produces: 全局 `LiveView = {snapshot() -> object|null, pick(fn|null), mark([x,y]|null)}`（`pick` 的 `fn(x, y, snapshot)` 在用户点画面一次后调，原图坐标）；Task 8 的 `livectl.js` 用它，并由 live.js 在进入 / 离开运行时调 `LiveCtl.start()` / `LiveCtl.stop()`（Task 8 之前用 `globalThis.LiveCtl &&` 判断）

- [ ] **Step 1: 改测试**（`tests/test_console_page.py`）

`test_skeleton` 末尾加：

```python
    assert 'src="static/stage.js"' in page and page.index('src="static/stage.js"') < page.index('src="console/static/live.js"')
    assert page.index('src="console/static/chatlog.js"') < page.index('src="console/static/live.js"')
```

`test_live_page` 整个换成：

```python
def test_live_page():  # spec 2026-10-01-console-live-page：顶栏 + 三栏（团子 / 画面 + 大脑 / 聊天记录）+ 日志抽屉，不再用 iframe
    b = bundle()
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    live = page.split('id="page-live"', 1)[1].split('id="page-inner"', 1)[0]
    for id_ in ("launch", "opt-live", "opt-emotes", "opt-duration", "start", "problems", "banners", "lv-run", "live-chip", "rundir",
                "lv-log-btn", "lv-now", "cards", "lv-ctl", "lv-canvas", "lv-stage-none", "lv-legend", "lv-brain", "lv-brain-none",
                "lv-chat", "lv-drawer", "log"):
        assert f'id="{id_}"' in live, id_
    assert "<iframe" not in live and "live-frame" not in b
    assert live.index('id="lv-now"') < live.index('id="lv-canvas"') < live.index('id="lv-brain"') < live.index('id="lv-chat"')
    for api in ("api/run/start", "api/logs", "live/status", "live/snapshot", "live/chat", "live/brain", "api/inner", "api/orphan/stop"):
        assert api in b, api
    line = next(l for l in b.splitlines() if l.strip().startswith("const FACTS="))
    assert '"身边的好友"' in line and '"心情"' in line
    assert "emotes_allowed" in b and "沙盒在跑，先下线" in b and "Pages.live" in b and "globalThis.LiveView" in b
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_console_page.py -k "live_page or skeleton"`
Expected: FAIL

- [ ] **Step 3: 替换 `console.html` 的 `#page-live`**

从 `<section id="page-live" class="page" hidden>` 到它的 `</section>`（`<section id="page-inner"` 前面那个）整段换成：

```html
  <section id="page-live" class="page" hidden>
    <header class="sb-head lv-head">
      <form class="lv-start" id="launch" autocomplete="off">
        <div class="lv-opts">
          <span class="lv-group" role="radiogroup" aria-label="模式">
            <label><input type="radio" name="mode" value="brain"> 统管大脑</label>
            <label><input type="radio" name="mode" value="agent"> 普通 Agent <small>调试用</small></label>
          </span>
          <span class="lv-sep" aria-hidden="true"></span>
          <label><input type="checkbox" id="opt-live"> 真的发送</label>
          <span class="live-warn" id="live-warn" hidden>会在游戏里真的说话、做动作</span>
          <label><input type="checkbox" id="opt-emotes"> 聊天时做动作 <small id="emotes-note"></small></label>
          <span class="lv-sep" aria-hidden="true"></span>
          <label>时长 <input type="number" id="opt-duration" min="0" step="60" placeholder="一直跑" aria-label="运行时长（秒）"> 秒</label>
          <button class="btn go" id="start" type="submit">叫醒团子</button>
        </div>
        <div id="problems" hidden></div>
      </form>
      <div class="lv-run" id="lv-run" hidden>
        <span class="chip" id="live-chip"></span>
        <span id="lv-mode"></span>
        <span class="note lv-dir" id="rundir"></span>
      </div>
      <span class="grow"></span>
      <button class="btn sm" type="button" id="lv-log-btn" aria-expanded="false" aria-controls="lv-drawer">日志</button>
    </header>
    <div class="lv-banners" id="banners"></div>
    <div class="cols sb-cols lv-cols">
      <section class="pane sb-right lv-left" aria-label="团子">
        <div id="lv-now"></div>
        <div class="sb-sec"><h4>身边和状态</h4><dl class="lv-facts" id="cards"></dl></div>
        <div class="sb-sec" id="lv-ctl"><h4>手动控制</h4></div>
      </section>
      <section class="pane lv-mid" aria-label="画面和大脑">
        <div class="lv-stage">
          <div class="lv-bar">
            <button class="btn sm" type="button" id="lv-pause">暂停</button>
            <button class="btn sm" type="button" id="lv-boxes">隐藏框</button>
            <button class="btn sm" type="button" id="lv-save">存图</button>
            <button class="btn sm" type="button" id="lv-legend-btn" aria-expanded="false" aria-controls="lv-legend">图例</button>
            <span class="lv-fps" id="lv-fps"></span>
          </div>
          <div class="lv-screen">
            <canvas id="lv-canvas" aria-label="游戏画面和识别框"></canvas>
            <p class="none" id="lv-stage-none">叫醒之后这里是游戏画面</p>
            <div class="lv-legend" id="lv-legend" hidden></div>
          </div>
        </div>
        <div class="lv-brain">
          <p class="none" id="lv-brain-none">叫醒之后这里是大脑在想什么</p>
          <div id="lv-brain" hidden></div>
        </div>
      </section>
      <section class="pane sb-chatpane" aria-label="聊天记录">
        <h3>聊天记录</h3>
        <div class="scroll sb-chat" id="lv-chat" aria-live="polite"><p class="none">叫醒之后这里是聊天记录</p></div>
      </section>
    </div>
    <aside class="lv-drawer" id="lv-drawer" aria-label="日志">
      <header><span>日志</span><span class="grow"></span><button class="btn sm" type="button" id="lv-log-close">收起</button></header>
      <pre id="log" tabindex="0"></pre>
    </aside>
  </section>
```

（`#lv-ctl` 里的手动控制在 Task 8 填；这一步先只有标题。）

`<script>` 区：在 `<script src="console/static/common.js"></script>` 前面加 `<script src="static/stage.js"></script>`。

- [ ] **Step 4: 改 `console.css`**

删掉 `/* ---- 真机 ---- */` 下面这些规则：`#idle-cols`、`#idle-cols[hidden]`、`.launch-body`、`#launch fieldset`、`#launch legend`、`#launch .choice`、`#launch small`、`.launch-go`、`#watch`、`.live-view`、`#live-frame`、`#live-wait`、`.live-side`、`.logbox summary`、`.logbox pre`、`.logbox pre .e`、`.logbox pre .w`。
**保留** `.live-warn` 和 `.cards …`（`grep -n "cards" src/skydango/console/static/inner.js` 看内心页用不用；用就留，不用也留着无害）。

在 `/* ---- 真机 ---- */` 下面加：

```css
/* 和沙盒一样：页面不滚、占满视口；顶栏固定，三栏各自滚（spec 2026-10-01-console-live-page） */
#page-live{height:100vh;display:flex;flex-direction:column;padding:0}
.lv-start{display:flex;flex-direction:column;gap:6px;min-width:0}
.lv-opts{display:flex;align-items:center;gap:6px 16px;flex-wrap:wrap}
.lv-opts label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap}
.lv-opts small{color:var(--muted)}
.lv-group{display:inline-flex;gap:4px 14px;flex-wrap:wrap}
.lv-sep{width:1px;height:18px;background:var(--line)}
#opt-duration{width:84px}
.lv-head .live-warn{margin:0}
.lv-run{display:flex;align-items:center;gap:6px 14px;flex-wrap:wrap;min-width:0}
.lv-dir{font-family:var(--mono);font-size:12px;user-select:all;overflow-wrap:anywhere}
.lv-banners{flex:none;padding:0 20px}
.lv-banners:empty{display:none}
.lv-banners .banner{margin:10px 0 0}
.sb-cols.lv-cols{grid-template-columns:minmax(250px,.8fr) minmax(0,1.6fr) minmax(250px,.9fr)}

/* 左：现在 + 身边和状态 + 手动控制 */
#lv-now>h3{margin:0;padding:10px 14px 6px;font-size:12px;letter-spacing:.12em;color:var(--muted);font-weight:600;display:flex;align-items:center;gap:8px}
#lv-now .now-rows{grid-template-columns:52px minmax(0,1fr)}
.lv-facts{margin:0;display:grid;grid-template-columns:5.4em minmax(0,1fr);gap:3px 10px;font-size:13px}
.lv-facts dt{color:var(--muted);font-size:12px;padding-top:1px}
.lv-facts dd{margin:0;overflow-wrap:anywhere}
.lv-tag{display:inline-block;background:var(--matcha-bg);color:var(--matcha-ink);border-radius:999px;padding:0 9px;margin:0 4px 3px 0;font-size:12.5px}

/* 中：画面在上（16:9，最高半屏），大脑控制台占剩下的 */
.sb-cols>.lv-mid{padding:0}
.lv-stage{flex:none;padding:10px 12px;border-bottom:1px solid var(--line)}
.lv-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.lv-fps{margin-left:auto;font-size:12px;color:var(--muted)}
.lv-fps.bad{color:var(--brick-ink)}
.lv-screen{position:relative;width:min(100%,calc(50vh*16/9));aspect-ratio:16/9;margin:0 auto;background:var(--bubble);border-radius:8px;overflow:hidden}
#lv-canvas{display:block;width:100%;height:100%}
#lv-canvas.picking{cursor:crosshair}
.lv-screen>.none{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:0 16px}
.lv-legend{position:absolute;right:8px;top:8px;max-width:72%;display:flex;flex-wrap:wrap;gap:3px 10px;padding:7px 10px;background:var(--card);border:1px solid var(--line);border-radius:8px;box-shadow:var(--lift);font-size:12px;color:var(--muted)}
.lv-legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.lv-brain{flex:1;min-height:0;display:flex;flex-direction:column;position:relative;background:var(--term-bg)}
.lv-brain>.none{color:var(--term-dim);padding:12px 14px;font-family:var(--mono);font-size:12.5px}

/* 日志抽屉：从右边滑出，关着也照样拉日志 */
.lv-drawer{position:fixed;top:0;right:0;bottom:0;width:min(560px,92vw);z-index:30;display:flex;flex-direction:column;background:var(--card);
  border-left:1px solid var(--line);box-shadow:var(--lift);transform:translateX(100%);visibility:hidden;transition:transform .2s,visibility .2s}
.lv-drawer.open{transform:none;visibility:visible}
.lv-drawer>header{flex:none;display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);font-size:12px;letter-spacing:.12em;color:var(--muted)}
.lv-drawer pre{flex:1;min-height:0;margin:0;padding:8px 14px 14px;overflow:auto;font:12px/1.5 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere}
.lv-drawer pre .e{color:var(--brick-ink)}
.lv-drawer pre .w{color:var(--amber-ink)}
@media (prefers-reduced-motion:reduce){.lv-drawer{transition:none}}
@media (max-width:999px){
  #page-live{height:auto}
  .lv-banners{padding:0 12px}
  .sb-cols.lv-cols{grid-template-columns:1fr}
  .lv-cols>.pane{height:70vh}
  .lv-cols>.lv-mid{height:auto;min-height:70vh;order:0}
  .lv-screen{width:100%}
}
```

- [ ] **Step 5: 重写 `live.js`**

整个文件换成：

```js
/* 真机团子页（spec 2026-10-01-console-live-page）：顶栏（停着 = 启动选项，跑着 = 运行信息）+ 日志抽屉 + 三栏：
 * 左 现在（Inner.renderNow，api/inner）+ 身边和状态（live/status）+ 手动控制（livectl.js）；
 * 中 画面（live/snapshot 长轮询，stage.js 画框）+ 大脑控制台（brainlog.js，live/brain）；右 聊天记录（live/chat 长轮询，chatlog.js）。
 * 画面和聊天只在这一页开着、团子在跑时拉；停下后三栏留着最后的内容，下次叫醒才清空。
 * 给 livectl.js：globalThis.LiveView（最新快照、在画面上点一下、画十字）。「停止」只在左栏卡片上（common.js）。 */
(function () {
"use strict";
if (typeof document === "undefined") return;
const L = {launchLoaded: false, startProblems: false, on: false, gen: 0, shown: false, kind: "", state: "",
  seq: 0, snap: null, img: null, paused: false, boxes: true, mark: null, hover: null, pick: null, times: [],
  chatV: 0, trace: null, info: null, innerOk: false, logNext: 0, follow: true, drain: false, timers: []};
const FACTS = ["身边的好友", "陌生人", "互动请求", "开着的面板", "牵着手", "正在做", "刚说过", "场合", "聊天面板", "心情", "精力"];
const WITH_INNER = ["心情", "精力"];  // 「现在」里已经有了，内心层开着时不重复
const CHIP = {idle: "没在跑", starting: "正在启动", running: "运行中", stopping: "在收尾", crashed: "出错停下了", exited: "没在跑"};

function liveWarn() { $("live-warn").hidden = !$("opt-live").checked; }
function sleep(ms) { return new Promise(ok => setTimeout(ok, ms)); }
function alive(gen) { return gen === L.gen && L.on && L.shown; }

/* ---- 顶栏、横幅（每秒随 /api/state） ---- */
function render() {
  const st = S.state; if (!st) return;
  const run = st.run, busy = BUSY.includes(run.state), isSb = run.kind === "sandbox", mine = busy && !isSb;
  if (!L.launchLoaded && st.launch) {  // 启动选项：第一次按服务器记住的填，之后用户自己改
    const l = st.launch;
    document.querySelector(`input[name=mode][value=${l.brain ? "brain" : "agent"}]`).checked = true;
    $("opt-live").checked = l.live; $("opt-emotes").checked = l.emotes; $("opt-duration").value = l.duration > 0 ? l.duration : "";
    L.launchLoaded = true; liveWarn();
  }
  $("launch").hidden = mine; $("lv-run").hidden = !mine;
  const off = busy && isSb;  // 沙盒在跑：表单整个置灰
  for (const x of $("launch").querySelectorAll("input,button")) x.disabled = off;
  if (!st.emotes_allowed) { $("opt-emotes").checked = false; $("opt-emotes").disabled = true; }  // config.toml 关了动作：只能关不能开
  $("emotes-note").textContent = st.emotes_allowed ? "" : "config.toml 里关掉了";
  $("start").textContent = off ? "沙盒在跑，先下线" : "叫醒团子";
  if (mine) L.startProblems = false;
  if (!L.startProblems) problemList($("problems"), mine ? [] : st.problems || []);
  const chip = $("live-chip");
  chip.className = "chip" + (run.state === "running" ? " live dot" : "");
  chip.textContent = CHIP[run.state] || run.state;
  const o = run.options || {};
  $("lv-mode").textContent = [o.brain === false ? "普通 Agent" : "统管大脑", o.live ? "真的发送" : "只打印",
    run.uptime != null ? `已运行 ${fmtUptime(run.uptime)}` : ""].filter(Boolean).join(" · ");
  $("rundir").textContent = mine && run.run_dir ? run.run_dir : "";
  renderBanners(st, run, busy, isSb);
  if (!isSb && run.state === "crashed" && L.state !== "crashed") openDrawer(true);  // 出错停下：日志自己打开
  L.state = isSb ? "" : run.state;
  const on = !isSb && (run.state === "running" || run.state === "stopping");
  if (on && !L.on) begin(o.brain !== false);
  else if (!on && L.on) end();
}
function renderBanners(st, run, busy, isSb) {
  const banners = $("banners"); banners.textContent = "";
  if (run.forced && !busy && !isSb) {
    const b = el("div", "banner bad");
    b.append(el("span", "", "强制结束了，轮盘可能没换回，请用 python -m skydango emotes wheel 检查。")); banners.append(b);
  }
  if (st.orphan) {
    const b = el("div", "banner warn"), btn = el("button", "btn sm", "让它退出");
    b.append(el("span", "", "上次留下的团子还在运行（占着子进程端口），先让它退出再叫醒新的。"));
    btn.type = "button"; btn.onclick = async () => { btn.disabled = true; await post("api/orphan/stop"); refresh(); };
    b.append(btn); banners.append(b);
  }
}

/* ---- 进入 / 离开运行：清空上一次的内容、挂大脑、开手动控制 ---- */
function begin(brain) {
  L.on = true; L.gen++; L.seq = 0; L.snap = null; L.chatV = 0; L.info = null; L.mark = null; L.times = [];
  const chat = $("lv-chat"); chat.textContent = ""; chat.append(el("p", "none", "还没有聊天"));
  $("lv-stage-none").hidden = false; $("lv-stage-none").textContent = "画面出来之后显示在这里";
  const ctx = $("lv-canvas").getContext("2d"); ctx.clearRect(0, 0, $("lv-canvas").width, $("lv-canvas").height);
  renderFacts(); renderNow(null);
  if (L.trace) { L.trace.stop(); L.trace = null; }
  const box = $("lv-brain"), none = $("lv-brain-none"); box.textContent = ""; box.hidden = true;
  if (brain && typeof mountBrainConsole === "function") { none.hidden = true; L.trace = mountBrainConsole(box, "live/brain"); }
  else { none.hidden = false; none.textContent = "普通 Agent 没有大脑"; }
  if (globalThis.LiveCtl) LiveCtl.start();
  if (L.shown) loops();
}
function end() {  // 停下：内容留着，只停掉拉取
  L.on = false; L.gen++;
  if (L.trace) { L.trace.stop(); L.trace = null; }
  if (globalThis.LiveCtl) LiveCtl.stop();
  LiveView.pick(null);
  $("lv-fps").textContent = "停了"; $("lv-fps").className = "lv-fps";
}
function loops() { L.gen++; const g = L.gen; stageLoop(g); chatLoop(g); pullStatus(); pullInner(); }

/* ---- 画面 ---- */
async function stageLoop(gen) {
  if (typeof Stage === "undefined") { $("lv-stage-none").textContent = "画面脚本加载不了（static/stage.js）"; return; }
  while (alive(gen) && !L.paused) {
    try {
      const r = await fetch(`live/snapshot?after=${L.seq}`, {cache: "no-store"});
      if (!alive(gen)) return;
      if (r.status === 204) continue;
      if (!r.ok) throw new Error(r.status);
      const s = await r.json(); if (!alive(gen)) return;
      const img = new Image();
      await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = "data:image/jpeg;base64," + s.image; });
      if (!alive(gen)) return;
      L.seq = s.seq; L.snap = s; L.img = img; redraw(); $("lv-stage-none").hidden = true;
      const now = performance.now(); L.times.push(now); L.times = L.times.filter(t => now - t < 2000);
      $("lv-fps").textContent = `${(L.times.length / 2).toFixed(1)} 帧/秒 · ${s.width}×${s.height}`; $("lv-fps").className = "lv-fps";
    } catch (e) {
      if (!alive(gen)) return;
      $("lv-fps").textContent = "连不上"; $("lv-fps").className = "lv-fps bad"; await sleep(1000);
    }
  }
}
function redraw() { if (L.snap && L.img) Stage.draw($("lv-canvas"), L.img, L.snap, {boxes: L.boxes, mark: L.mark, hover: L.hover}); }
function canvasPoint(e) { const c = $("lv-canvas"), r = c.getBoundingClientRect(); return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height]; }

/* ---- 聊天记录 ---- */
async function chatLoop(gen) {
  while (alive(gen)) {
    try {
      const r = await fetch(`live/chat?after=${L.chatV}&wait=2`, {cache: "no-store"});
      if (!alive(gen)) return;
      if (r.status === 404) {  // 普通 Agent / 局域网模式没有聊天记录
        const box = $("lv-chat"); box.textContent = ""; box.append(el("p", "none", "这次运行没有聊天记录（普通 Agent，或者可视化开在局域网）")); return;
      }
      if (!r.ok) throw new Error(r.status);
      const d = await r.json(); if (!alive(gen)) return;
      if (d.v < L.chatV) { $("lv-chat").textContent = ""; }  // 团子重启过：从头来
      Chat.append($("lv-chat"), d.lines || []); L.chatV = d.v;
    } catch (e) { if (!alive(gen)) return; await sleep(1000); }
  }
}

/* ---- 左栏：现在 + 身边和状态 ---- */
function renderNow(d) {
  const box = $("lv-now");
  if (d && d.now) Inner.renderNow(box, d.now, d);
  else {
    box.textContent = ""; box.append(el("h3", "", "现在"));
    const b = el("div", "pane-body"); b.append(el("p", "none", L.on ? "（内心层没开，或者还没取到）" : "叫醒之后这里是团子现在的样子")); box.append(b);
  }
  const h = box.querySelector("h3"), link = el("button", "linkish sb-inner-link", "完整内心 →");
  link.type = "button"; link.onclick = () => { Inner.setSource("dango"); go("inner"); };
  h.append(el("span", "grow"), link);
}
async function pullInner() {
  if (!L.on) return;
  try { const d = await getJSON("api/inner"); if (!L.on) return; L.innerOk = !!(d && d.now); renderNow(d); }
  catch (e) { /* 下次再取 */ }
}
function renderFacts() {
  const dl = $("cards"); dl.textContent = ""; const info = L.info;
  if (!info) { dl.append(el("dt", "", "状态"), el("dd", "none", L.on ? "取状态中…" : "叫醒之后这里是身边有谁、在做什么")); return; }
  for (const key of FACTS) {
    if (!(key in info) || (L.innerOk && WITH_INNER.includes(key))) continue;
    const v = info[key], dd = el("dd");
    if (Array.isArray(v)) {
      if (!v.length) dd.textContent = "没有";
      else if (key === "身边的好友") for (const x of v) dd.append(el("span", "lv-tag", String(x)));
      else dd.textContent = v.map(String).join("、");
    } else dd.textContent = String(v);
    dl.append(el("dt", "", key), dd);
  }
}
async function pullStatus() {
  if (!L.on) return;
  try { const r = await fetch("live/status", {cache: "no-store"}); if (r.ok && L.on) { L.info = (await r.json()).info; renderFacts(); } } catch (e) {}
}

/* ---- 日志抽屉 ---- */
function openDrawer(open) {
  $("lv-drawer").classList.toggle("open", open); $("lv-log-btn").setAttribute("aria-expanded", String(open));
  if (open) { const b = $("log"); b.scrollTop = b.scrollHeight; }
}
async function pullLogs() {
  const run = S.state && S.state.run;
  if (!run || run.kind === "sandbox") return;
  if (!BUSY.includes(run.state)) { if (!L.drain) return; L.drain = false; }  // 刚停下：再拉最后一次，把收尾 / traceback 读进来
  else L.drain = true;
  try {
    const r = await getJSON(`api/logs?after=${L.logNext}`), box = $("log");
    if (r.next < L.logNext) { L.logNext = 0; box.textContent = ""; return; }  // 重新启动过：从头来
    for (const line of r.lines) box.append(el("span", /Traceback|ERROR|错误/.test(line) ? "e" : /WARNING/.test(line) ? "w" : "", line + "\n"));
    while (box.childNodes.length > 1000) box.firstChild.remove();
    L.logNext = r.next; if (r.lines.length && L.follow) box.scrollTop = box.scrollHeight;
  } catch (e) {}
}

async function start(e) {
  e.preventDefault();
  const d = $("opt-duration").value.trim();
  const body = {brain: document.querySelector("input[name=mode]:checked").value === "brain", live: $("opt-live").checked,
    emotes: $("opt-emotes").checked, duration: d === "" ? 0 : Number(d)};
  $("start").disabled = true;
  const r = await post("api/run/start", body);
  if (!r.data.ok) { L.startProblems = true; problemList($("problems"), r.data.problems || [r.data.text || r.data.error || "启动失败"]); }
  else { L.startProblems = false; L.logNext = 0; $("log").textContent = ""; }
  $("start").disabled = false;
  await refresh();
}

globalThis.LiveView = {
  snapshot: () => L.snap,
  pick(fn) { L.pick = fn || null; $("lv-canvas").classList.toggle("picking", !!L.pick); },
  mark(m) { L.mark = m || null; redraw(); },
};

function bind() {
  $("opt-live").onchange = liveWarn;
  $("launch").onsubmit = start;
  $("log").addEventListener("scroll", () => { const b = $("log"); L.follow = b.scrollTop + b.clientHeight >= b.scrollHeight - 8; });
  $("lv-log-btn").onclick = () => openDrawer(!$("lv-drawer").classList.contains("open"));
  $("lv-log-close").onclick = () => openDrawer(false);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && $("lv-drawer").classList.contains("open")) openDrawer(false); });
  $("lv-pause").onclick = () => {
    L.paused = !L.paused; $("lv-pause").textContent = L.paused ? "继续" : "暂停";
    if (!L.paused && L.on && L.shown) { L.gen++; const g = L.gen; stageLoop(g); chatLoop(g); }
  };
  $("lv-boxes").onclick = () => { L.boxes = !L.boxes; $("lv-boxes").textContent = L.boxes ? "隐藏框" : "显示框"; redraw(); };
  $("lv-save").onclick = () => {
    if (!L.snap) return;
    const a = el("a"); a.download = `dango-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}.png`;
    a.href = $("lv-canvas").toDataURL("image/png"); a.click();
  };
  $("lv-legend-btn").onclick = () => {
    const g = $("lv-legend"), open = g.hidden; g.hidden = !open; $("lv-legend-btn").setAttribute("aria-expanded", String(open));
  };
  if (typeof Stage !== "undefined") for (const [k, name] of Object.entries(Stage.NAMES)) {
    const s = el("span"), i = el("i"); i.style.background = Stage.COLORS[k]; s.append(i, name); $("lv-legend").append(s);
  }
  const c = $("lv-canvas");
  c.addEventListener("mousemove", e => { L.hover = canvasPoint(e); redraw(); });
  c.addEventListener("mouseleave", () => { L.hover = null; redraw(); });
  c.addEventListener("click", e => {
    if (!L.pick || !L.snap || typeof Stage === "undefined") return;
    const [x, y] = Stage.toFrame(e.clientX, e.clientY, c.getBoundingClientRect(), L.snap.width, L.snap.height), fn = L.pick, s = L.snap;
    LiveView.pick(null); fn(x, y, s);
  });
}

Pages.live = {
  init() { bind(); renderNow(null); renderFacts(); onState(render); },
  show() {
    L.shown = true; L.startProblems = false; render(); pullLogs();
    for (const t of L.timers) clearInterval(t);
    L.timers = [setInterval(pullStatus, 2000), setInterval(pullInner, 5000), setInterval(pullLogs, 1000)];
    if (L.on) loops();
  },
  hide() { L.shown = false; L.gen++; for (const t of L.timers) clearInterval(t); L.timers = []; },
};
})();
```

注意：`linkish` 类沙盒已经在用（`sb-inner-link`）；`Pages.live.show` 可能被重复调（点当前导航项），所以先清定时器。

- [ ] **Step 6: 跑测试**

Run: `python -m pytest -q tests/test_console_page.py tests/test_console_server.py`
Expected: 全部 PASS（`test_scripts_parse` 会 `node --check` live.js；`test_colors_only_in_root` 查 CSS；`test_offline_relative_and_no_native_dialogs` 查没有原生弹窗）

- [ ] **Step 7: 提交**

```bash
git add src/skydango/console/static/console.html src/skydango/console/static/console.css src/skydango/console/static/live.js tests/test_console_page.py
git commit -m "feat(console): 真机团子页照沙盒改成三栏，画面直接画在页面里，加聊天记录和日志抽屉

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 手动控制 `livectl.js`

**Files:**
- Create: `src/skydango/console/static/livectl.js`
- Modify: `src/skydango/console/static/console.html`（`#lv-ctl` 填内容、加 `<script>`）、`src/skydango/console/static/console.css`（手动控制样式）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: `LiveView`（Task 7）、`Stage.nameAt`（Task 5）、common.js 的 `$ el post ask`；接口 `live/control/options`（GET，404 = 身体还没建好）、`POST live/control` `{action, args}` → `{ok, text}`
- Produces: 全局 `LiveCtl = {start(), stop()}`；`module.exports = {controlLine}`

- [ ] **Step 1: 写失败的测试**（`tests/test_console_page.py`）

`JS` 列表在 `"live.js"` 前面加 `"livectl.js"`。加：

```python
def test_live_manual_control():
    page = (STATIC / "console.html").read_text(encoding="utf-8")
    ctl = page.split('id="lv-ctl"', 1)[1].split("</section>", 1)[0]
    for id_ in ("lc-warn", "lc-none", "lc-busy", "lc-say-text", "lc-count", "lc-say-go", "lc-emote-name", "lc-emote-go", "lc-camera",
                "lc-steps", "lc-reset", "lc-around", "lc-pick", "lc-pick-tip", "lc-track-name", "lc-track-pick", "lc-track-sec",
                "lc-track-go", "lc-stop", "lc-track-tip", "lc-panels", "lc-panel-read", "lc-panel-close", "lc-log"):
        assert f'id="{id_}"' in ctl, id_
    assert page.index('src="console/static/livectl.js"') < page.index('src="console/static/live.js"')
    js = (STATIC / "livectl.js").read_text(encoding="utf-8")
    for part in ("live/control/options", 'post("live/control"', "404", "3000", "5000", "ask(", "Stage.nameAt(", "LiveView.pick(", "globalThis.LiveCtl"):
        assert part in js, part
    assert "innerHTML" not in js


def test_live_control_line():  # 操作记录一行：做了什么 → 结果
    node = shutil.which("node") or pytest.skip("没有 node")
    js = (f"const C=require({json.dumps(str(STATIC / 'livectl.js'))});console.log(JSON.stringify(["
          "C.controlLine('say',{text:'hi'},{text:'ok'}),C.controlLine('camera',{action:'left',steps:2},{text:'ok'}),"
          "C.controlLine('track',{name:'ming',seconds:30},{text:'ok'}),C.controlLine('check_friend',{x:1,y:2},{text:'ok'})]))")
    out = json.loads(subprocess.run([node, "-e", js], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert out == ["说「hi」 → ok", "左转 ×2 → ok", "盯着ming（30 秒） → ok", "看人 (1, 2) → ok"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_console_page.py -k "manual_control or control_line"`
Expected: FAIL

- [ ] **Step 3: 填 `#lv-ctl`**（`console.html`）

`<div class="sb-sec" id="lv-ctl"><h4>手动控制</h4></div>` 换成：

```html
        <div class="sb-sec lc" id="lv-ctl">
          <h4>手动控制<span class="grow"></span><span class="lc-busy" id="lc-busy"></span></h4>
          <p class="lc-warn" id="lc-warn" hidden>大脑是 dry-run，手动操作照样会真的在游戏里执行</p>
          <p class="none" id="lc-none">叫醒之后能在这里手动让团子说话、做动作、转视角</p>
          <div class="lc-row"><span class="sb-label">说话</span><input type="text" id="lc-say-text" placeholder="让团子说一句…" aria-label="让团子说的话"><span class="lc-count" id="lc-count"></span><button class="btn sm" type="button" id="lc-say-go">说</button></div>
          <div class="lc-row"><span class="sb-label">动作</span><select id="lc-emote-name" aria-label="动作"></select><button class="btn sm" type="button" id="lc-emote-go">做</button></div>
          <div class="lc-row" id="lc-camera"><span class="sb-label">视角</span>
            <span class="lc-btns"><button class="btn sm" type="button" data-cam="left">左转</button><button class="btn sm" type="button" data-cam="right">右转</button><button class="btn sm" type="button" data-cam="up">抬头</button><button class="btn sm" type="button" data-cam="down">低头</button><button class="btn sm" type="button" data-cam="zoom_in">拉近</button><button class="btn sm" type="button" data-cam="zoom_out">拉远</button></span>
          </div>
          <div class="lc-row"><span class="sb-label"></span><label class="lc-num">步数 <input type="number" id="lc-steps" value="1" min="1"></label><button class="btn sm" type="button" id="lc-reset">复位</button><button class="btn sm" type="button" id="lc-around">环视一圈</button></div>
          <div class="lc-row"><span class="sb-label">看人</span><button class="btn sm" type="button" id="lc-pick">在画面上选人</button><span class="lc-tip" id="lc-pick-tip"></span></div>
          <div class="lc-row"><span class="sb-label">盯人</span><input type="text" id="lc-track-name" placeholder="好友名字" aria-label="盯谁"><button class="btn sm" type="button" id="lc-track-pick">在画面上选</button></div>
          <div class="lc-row"><span class="sb-label"></span><label class="lc-num"><input type="number" id="lc-track-sec" value="30" min="1"> 秒</label><button class="btn sm" type="button" id="lc-track-go">盯</button><button class="btn sm" type="button" id="lc-stop">停下</button><span class="lc-tip" id="lc-track-tip"></span></div>
          <div class="lc-row" id="lc-panels" hidden><span class="sb-label">面板</span><button class="btn sm" type="button" id="lc-panel-read">读面板</button><button class="btn sm" type="button" id="lc-panel-close">关面板</button></div>
          <ul class="lc-log" id="lc-log"></ul>
        </div>
```

`<script src="console/static/live.js"></script>` 前面加 `<script src="console/static/livectl.js"></script>`。

- [ ] **Step 4: 加样式**（`console.css`，接在 Task 7 的「左：…」那段后面）

```css
.lc h4{display:flex;align-items:center;gap:8px}
.lc-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:7px}
.lc-row>.sb-label{min-width:2.4em}
.lc-row input[type=text]{flex:1;min-width:0}
.lc-row select{flex:1;min-width:0}
.lc-btns{display:flex;flex-wrap:wrap;gap:4px;flex:1}
.lc-num{display:inline-flex;align-items:center;gap:4px;font-size:12.5px;color:var(--muted)}
.lc-num input{width:54px}
.lc-count,.lc-tip{font-size:12px;color:var(--muted)}
.lc-busy{font-size:12px;color:var(--indigo);letter-spacing:0;font-weight:400}
.lc-warn{margin:4px 0 0;padding:2px 10px;border-radius:6px;background:var(--amber-bg);color:var(--amber-ink);font-size:12.5px}
#lc-none{margin:4px 0 0}
.btn.on{border-color:var(--sakura);color:var(--sakura-ink);background:var(--sakura-bg)}
.lc-log{list-style:none;margin:8px 0 0;padding:6px 0 0;border-top:1px dashed var(--line);font:12px/1.6 var(--mono)}
.lc-log:empty{display:none}
.lc-log li{overflow-wrap:anywhere}
.lc-log .t{color:var(--muted)}
.lc-log .ok{color:var(--matcha-ink)}
.lc-log .bad{color:var(--brick-ink)}
```

- [ ] **Step 5: 新建 `src/skydango/console/static/livectl.js`**

```js
/* 真机团子页左栏的手动控制（spec 2026-10-01-console-live-page §2.2）：照 viewer 的手动控制栏（vision/viewer.py PAGE 的 control 段）重排，
 * 接口 live/control/options（404 = 身体还没建好，3 秒后再试）、POST live/control；在画面上选人走 live.js 的 LiveView；
 * 「看人」点了之后用 ask() 确认（内嵌浏览器里原生弹窗用不了）。总是真执行，照样过身体的护栏。
 * 顶层不碰 document：node 里 require 它能测 controlLine（tests/test_console_page.py）。 */
(function () {
"use strict";
const CAM = {left: "左转", right: "右转", up: "抬头", down: "低头", zoom_in: "拉近", zoom_out: "拉远"};
function controlLine(action, args, res) {
  const a = args || {}; let what;
  if (action === "say") what = `说「${a.text}」`;
  else if (action === "emote") what = `动作「${a.name}」`;
  else if (action === "camera") what = `${CAM[a.action] || a.action} ×${a.steps}`;
  else if (action === "camera_reset") what = "复位";
  else if (action === "look_around") what = "环视一圈";
  else if (action === "panel_read") what = "读面板";
  else if (action === "panel_close") what = "关面板";
  else if (action === "check_friend") what = `看人 (${a.x}, ${a.y})`;
  else if (action === "track") what = `盯着${a.name}（${a.seconds} 秒）`;
  else if (action === "stop_task") what = "停下";
  else what = action;
  return `${what} → ${res.text}`;
}
if (typeof module !== "undefined" && module.exports) module.exports = {controlLine};
if (typeof document === "undefined") return;

const K = {opts: null, busy: false, picking: false, trackPick: false, on: false, timer: 0, retry: 0};

function lock() {
  const o = K.opts, b = K.busy || !o, opts = o || {emotes: [], camera: []};
  $("lc-none").hidden = !!o;
  $("lc-warn").hidden = !(o && o.dry_run);
  $("lc-say-text").disabled = b; $("lc-say-go").disabled = b || !$("lc-say-text").value.trim();
  $("lc-emote-name").disabled = $("lc-emote-go").disabled = b || !opts.emotes.length;
  for (const x of document.querySelectorAll("#lc-camera button,#lc-steps,#lc-reset,#lc-around")) x.disabled = b || !(opts.camera || []).length;
  const pick = $("lc-pick");
  pick.disabled = b || !opts.friend_check; pick.classList.toggle("on", K.picking);
  $("lc-pick-tip").textContent = !o ? "" : !opts.friend_check ? "没开（[friend_check] enabled = false）" : K.picking ? "点一下画面上的人" : "";
  const tr = !!opts.track;
  $("lc-track-name").disabled = $("lc-track-sec").disabled = $("lc-track-pick").disabled = b || !tr;
  $("lc-track-go").disabled = b || !tr || !$("lc-track-name").value.trim(); $("lc-stop").disabled = b;
  $("lc-track-sec").max = opts.max_track_seconds || 60; $("lc-track-pick").classList.toggle("on", K.trackPick);
  if (o && !tr) $("lc-track-tip").textContent = "要开感知层（[perception]）和镜头";
  else if (K.trackPick) $("lc-track-tip").textContent = "点一下画面上的好友";
  $("lc-panels").hidden = !opts.panels; $("lc-panel-read").disabled = $("lc-panel-close").disabled = b || !opts.panels;
}
function apply() {
  const o = K.opts; if (!o) { lock(); return; }
  const sel = $("lc-emote-name"), keep = sel.value;
  if ([...sel.options].map(x => x.value).join("|") !== o.emotes.join("|")) {  // 列表变了才重建：定时刷新时别把打开的下拉框关掉
    sel.textContent = "";
    for (const n of o.emotes) { const op = el("option", "", n); op.value = n; sel.append(op); }
    if (o.emotes.includes(keep)) sel.value = keep;
  }
  $("lc-steps").max = o.max_steps; count();
}
function count() {
  const n = [...$("lc-say-text").value.trim()].length, max = K.opts ? K.opts.max_chars : 0;
  $("lc-count").textContent = K.opts ? `${n} / ${max}` : ""; lock();
}
function logLine(action, args, res) {
  const li = el("li");
  li.append(el("span", "t", new Date().toTimeString().slice(0, 8) + " "), el("span", res.ok ? "ok" : "bad", controlLine(action, args, res)));
  const ul = $("lc-log"); ul.prepend(li); while (ul.children.length > 10) ul.lastChild.remove();
}
async function options(retry) {
  try {
    const r = await fetch("live/control/options", {cache: "no-store"});
    if (!K.on) return;
    if (r.status === 404) {  // 身体还没建好：3 秒后再试
      K.opts = null; $("lc-none").textContent = "身体还没准备好…"; lock();
      if (retry) K.retry = setTimeout(() => options(true), 3000);
      return;
    }
    if (!r.ok) throw new Error(r.status);
    K.opts = await r.json(); apply();
  } catch (e) { if (retry && K.on) K.retry = setTimeout(() => options(true), 3000); }
}
async function send(action, args) {
  if (K.busy) return null;
  K.busy = true; lock();
  $("lc-busy").textContent = `正在做：${controlLine(action, args, {text: "…"}).split(" → ")[0]}…`;
  const r = await post("live/control", {action, args}), res = r.data && r.data.ok !== undefined ? r.data : {ok: false, text: `HTTP ${r.status}`};
  logLine(action, args, res); $("lc-busy").textContent = ""; K.busy = false; lock();
  await options(false);
  return res;
}

function bind() {
  $("lc-say-text").oninput = count;
  $("lc-say-text").onkeydown = e => { if (e.key === "Enter" && !e.isComposing) $("lc-say-go").click(); };
  $("lc-say-go").onclick = async () => {
    const t = $("lc-say-text").value.trim(); if (!t) return;
    const res = await send("say", {text: t}); if (res && res.ok) { $("lc-say-text").value = ""; count(); }
  };
  $("lc-emote-go").onclick = () => send("emote", {name: $("lc-emote-name").value});
  for (const x of document.querySelectorAll("#lc-camera button[data-cam]")) x.onclick = () => {
    const max = K.opts ? K.opts.max_steps : 4, n = Math.min(max, Math.max(1, parseInt($("lc-steps").value, 10) || 1));
    $("lc-steps").value = n; send("camera", {action: x.dataset.cam, steps: n});
  };
  $("lc-reset").onclick = () => send("camera_reset", {});
  $("lc-around").onclick = () => send("look_around", {});
  $("lc-panel-read").onclick = () => send("panel_read", {});
  $("lc-panel-close").onclick = () => send("panel_close", {});
  $("lc-stop").onclick = () => send("stop_task", {});
  $("lc-track-name").oninput = lock;
  $("lc-track-go").onclick = () => {
    const name = $("lc-track-name").value.trim(); if (!name) return;
    const max = (K.opts && K.opts.max_track_seconds) || 60, n = Math.min(max, Math.max(1, parseInt($("lc-track-sec").value, 10) || 30));
    $("lc-track-sec").value = n; send("track", {name, seconds: n});
  };
  $("lc-pick").onclick = () => {  // 看人：点画面 → 画十字 → ask() 确认 → check_friend
    K.trackPick = false; K.picking = !K.picking; lock();
    if (!K.picking) { LiveView.pick(null); return; }
    LiveView.pick(async (x, y) => {
      LiveView.mark([x, y]);
      const go = await ask(`点 (${x}, ${y}) 这个人？`, {ok: "点"});
      K.picking = false; lock();
      if (go) await send("check_friend", {x, y});
      LiveView.mark(null);
    });
  };
  $("lc-track-pick").onclick = () => {  // 盯人：按快照里的框认出点的是谁，填进名字
    K.picking = false; K.trackPick = !K.trackPick; $("lc-track-tip").textContent = ""; lock();
    if (!K.trackPick) { LiveView.pick(null); return; }
    LiveView.pick((x, y, s) => {
      const name = Stage.nameAt(s.boxes || [], x, y);
      K.trackPick = false;
      $("lc-track-tip").textContent = name ? "" : "那里没认出好友的名字，换个地方点，或者直接输入";
      if (name) $("lc-track-name").value = name;
      lock();
    });
  };
}

globalThis.LiveCtl = {
  start() {
    if (K.on) return;
    K.on = true; $("lc-none").textContent = "连接身体…"; options(true);
    K.timer = setInterval(() => { if (K.on && K.opts && !K.busy) options(false); }, 5000);  // 动作冷却后列表会变：定时刷新
  },
  stop() {
    K.on = false; clearInterval(K.timer); clearTimeout(K.retry); K.opts = null; K.picking = K.trackPick = false;
    $("lc-none").textContent = "叫醒之后能在这里手动让团子说话、做动作、转视角"; lock();
  },
};
function ready() { bind(); lock(); }
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready); else setTimeout(ready, 0);  // 同 common.js 末尾
})();
```

- [ ] **Step 6: 跑测试**

Run: `python -m pytest -q tests/test_console_page.py`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add src/skydango/console/static/livectl.js src/skydango/console/static/console.html src/skydango/console/static/console.css tests/test_console_page.py
git commit -m "feat(console): 真机团子页左栏的手动控制

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: 文档、全量测试、浏览器验证

**Files:**
- Modify: `CLAUDE.md`（「管理面板」「代码结构」两处）
- Modify: `docs/superpowers/specs/2026-10-01-console-live-page-design.md`（开头加状态行）

- [ ] **Step 1: 改 `CLAUDE.md`**

1. 「代码结构」表 `src/skydango/console/` 那一行：页面列表里加上 `chatlog.js`、`livectl.js`；`#live` 的描述「真机团子 = 原来的总览 + 实时画面…」改成「真机团子：照沙盒三栏（团子 / 画面 + 大脑 / 聊天记录）+ 日志抽屉」
2. 「代码结构」表 `src/skydango/brain/world.py` `src/skydango/sandbox/` 那一行：沙盒的 `聊天记录 transcript.py` 改成说明聊天记录在 `brain/transcript.py`（沙盒和真机共用）
3. 「管理面板」一节 **2026-10-01 重做** 那段里 `#live`（真机团子 = 原来的总览 + 实时画面：停着是启动表单 + 预检问题，跑着左边 iframe 嵌子进程 viewer、右边状态卡片 + 日志）改成：
   `#live`（真机团子，**2026-10-01 晚再改**，spec `docs/superpowers/specs/2026-10-01-console-live-page-design.md`：顶栏停着是启动选项、跑着是运行信息 + 「日志」抽屉；三栏 左 现在 + 身边和状态 + 手动控制（`livectl.js`）/ 中 画面（直接画在页面里，画框代码和 viewer 共用 `vision/static/stage.js`）+ 大脑控制台（`brainlog.js`）/ 右 聊天记录（viewer `/chat`，身体 `on_line` + 事件队列 `tap` 记进 `brain/transcript.py`，只在内存里、最近 500 行）；停下后内容留着；**还没在真机上用过**）

- [ ] **Step 2: spec 开头加状态**

标题下面第一行后加：`状态：**代码已完成**（计划 `docs/superpowers/plans/2026-10-01-console-live-page.md`），待真机验证（§6 第 4 步）`

- [ ] **Step 3: 全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS（基线是 2255 passed，加上新测试会更多）

- [ ] **Step 4: 浏览器验证（spec §6 第 1~3 步）**

1. **viewer 没改坏**：`python -m skydango view --images <有截图的录像目录> --no-browser --port 19398`，浏览器面板开 `http://127.0.0.1:19398/`，框、图例、悬停和以前一样
2. **沙盒聊天记录没改坏、真机页布局**：worktree 里起管理面板 `python -m skydango console --no-browser --port 19395`（别用 19390，主目录的面板可能还开着），浏览器面板开 `http://127.0.0.1:19395/#sandbox` 看聊天区；再开 `#live` 看停着时的顶栏和三栏占位、窄屏（`resize_window` 宽 900）变上下排
3. **跑着的样子**：只有在**主目录的团子已经停了**（`Get-CimInstance Win32_Process` 里没有 `skydango run`）时才做：在 19395 的面板上以 dry-run（不勾「真的发送」）叫醒团子，确认三栏都有内容、画面不裁、悬停 / 图例 / 暂停 / 存图能用、手动控制「读面板」之类无害操作能用、日志抽屉开关、停下后内容还在；截图。团子还在跑就跳过这一步，在报告里写明
4. 用完 `taskkill /F /T /PID <pid>` 停掉自己起的 view / console，`Get-CimInstance Win32_Process` 确认没残留

- [ ] **Step 5: 提交**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-10-01-console-live-page-design.md
git commit -m "docs: 真机团子页改版写进 CLAUDE.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

（合并进 main 并推送由 finishing-a-development-branch 那一步做。）
