# 可视化网页：大脑活动时间线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run --brain --view` 时，网页画面下方实时显示大脑每一轮：叫醒原因、收到的消息、说的话、思考、工具调用和返回、结果。

**Architecture:** 新模块 `brain/trace.py` 的 `BrainTrace` 在内存里记最近 50 轮（数据来自 `BrainSession.on_message` 和 `Brain.wake/farewell`）；
`Viewer` 新增 `/brain` 长轮询接口；页面画面下方加一栏时间线。只在有 viewer 时创建，大脑那边零开销。

**Tech Stack:** Python 3.13 标准库（threading、http.server、json）；页面是 `viewer.py` 里内嵌的原生 JS / CSS，无外部库。

**Spec:** `docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md`

## Global Constraints

- 不加依赖；页面不引外部脚本 / 字体，沿用 `viewer.py` 现有深色 CSS 变量（`--bg --panel --text --muted --line`）
- `MAX_TURNS = 50`，`MAX_CHARS = 20000`（超出截断，末尾加 `（截断，原长 N 字）`）
- 工具名去掉 `mcp__sky__` 前缀；工具返回里的图片块记成 `[图片]`
- 思考块只在内容非空时记
- `/brain`：没有 trace → 404；有 → 200 JSON；等满 `WAIT`（2 秒）没变化也回 200（`turns` 为空，带最新 `state`）
- trace 的每个入口都 `try/except`，出错只 `log.debug(..., exc_info=True)`，不能影响大脑
- 页面里所有来自数据的文字一律用 `textContent` 写入，不拼 HTML
- 代码注释、日志、页面文案用中文，风格照现有文件（短注释、行尾 `#` 说明）
- 测试命令：`python -m pytest -q`（不需要模拟器）

## Review Focus

1. **`look(image=true)` 的返回是图片块 + 文字块的列表**：应记成 `[图片]` 加文字，不能把 base64 塞进页面 → Task 1 `test_tool_result_blocks_with_image`
2. **Claude Code 在另一个线程往里喂消息，同时浏览器在 `since()`**：不能抛异常、不能丢步骤 → Task 1 `test_feed_from_another_thread_while_polling`
3. **上一轮超时后进程还在吐 `tool_result`**：应进"轮外"，不能挂到下一轮或丢掉 → Task 1 `test_messages_between_turns_go_outside`
4. **聊天内容里有 `<b>`、`<script>` 之类**：页面必须原样显示文字 → Task 4 `test_page_never_uses_innerhtml_for_brain_data`
5. **程序重启而浏览器没刷新**（浏览器带着更大的 `after`）：应返回全部轮次，页面清空重来 → Task 1 `test_after_larger_than_version_returns_everything`

---

### Task 1: `BrainTrace` 记录器

**Files:**
- Create: `src/skydango/brain/trace.py`
- Test: `tests/test_brain_trace.py`

**Interfaces:**
- Produces:
  - `BrainTrace(wall: Callable[[], float] = time.time)`；属性 `state: Callable[[], dict] | None = None`（由 Task 2 的 `Brain` 设置）
  - `begin(reason: str, prompt: str) -> None`：`reason` ∈ `"events" | "heartbeat" | "farewell"`
  - `feed(message: dict) -> None`
  - `finish(result: dict, seconds: float) -> None`
  - `fail(error: str, seconds: float) -> None`
  - `chain(fn: Callable[[dict], None]) -> Callable[[dict], None]`：返回先调 `fn(m)` 再 `self.feed(m)` 的函数（`fn` 抛异常也照样 feed）
  - `since(after: int, timeout: float) -> dict`：`{"version": int, "oldest": int | None, "state": dict, "turns": [turn, ...]}`
  - 每个 turn（dict，JSON 可序列化）：`id, reason, start(墙钟秒), end(None=进行中), seconds, prompt, steps, tools(工具名列表), result, error, updated`
  - step：`{"kind": "thinking"|"text", "text"}`、`{"kind": "tool", "id", "name", "input"}`、`{"kind": "result", "tool_use_id", "text", "error": bool}`
  - `result`：`{"subtype", "num_turns", "cost", "tokens": {"input", "output", "cache_read", "cache_write", "thinking"}}`，缺的字段为 `None`
  - 常量 `MAX_TURNS = 50`、`MAX_CHARS = 20000`

stream-json 消息的形状（测试和实现都按这个）：

```python
{"type": "assistant", "message": {"content": [
    {"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "t1", "name": "mcp__sky__say", "input": {"text": "你好"}}]}}
{"type": "user", "message": {"content": [
    {"type": "tool_result", "tool_use_id": "t1", "is_error": False,
     "content": "已发送"  # 或 [{"type": "text", "text": "..."}, {"type": "image", "source": {...}}]
    }]}}
# result.usage: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
#               output_tokens_details.thinking_tokens；另有 subtype、num_turns、total_cost_usd
```

- [ ] **Step 1: 写失败的测试** `tests/test_brain_trace.py`

辅助函数：`asst(*blocks)`、`user(*blocks)` 拼上面的消息；`turns(t, after=0) = t.since(after, 0.0)["turns"]`。

```python
def test_steps_in_order_and_tool_prefix_stripped():
    t = BrainTrace(wall=lambda: 100.0)
    t.begin("events", "[..] 事件：\n- 卡洛：你好")
    t.feed(asst({"type": "thinking", "thinking": "想想"}, {"type": "text", "text": " 嗯 "},
                {"type": "tool_use", "id": "t1", "name": "mcp__sky__say", "input": {"text": "你好"}}))
    t.feed(user({"type": "tool_result", "tool_use_id": "t1", "content": "已发送", "is_error": False}))
    [turn] = turns(t)
    assert turn["reason"] == "events" and turn["prompt"].endswith("卡洛：你好") and turn["end"] is None
    assert [s["kind"] for s in turn["steps"]] == ["thinking", "text", "tool", "result"]
    assert turn["steps"][1]["text"] == "嗯"
    assert turn["steps"][2] == {"kind": "tool", "id": "t1", "name": "say", "input": {"text": "你好"}}
    assert turn["steps"][3] == {"kind": "result", "tool_use_id": "t1", "text": "已发送", "error": False}
    assert turn["tools"] == ["say"]

def test_empty_thinking_and_blank_text_are_skipped(): ...  # {"thinking": ""}、{"text": "  "} → steps == []

def test_tool_result_blocks_with_image():
    # content=[{"type":"image","source":{"data":"AAAA"}}, {"type":"text","text":"画面里有卡洛"}]
    # → text == "[图片]\n画面里有卡洛"，且 "AAAA" 不在 json.dumps(since(...)) 里

def test_error_result_marked(): ...  # is_error=True → step["error"] is True

def test_long_text_truncated():
    # text 块 "字" * (MAX_CHARS + 5) → len(text) <= MAX_CHARS + 30 且以 f"（截断，原长 {MAX_CHARS + 5} 字）" 结尾

def test_finish_records_result_tokens_and_seconds():
    # finish({"subtype": "success", "num_turns": 3, "total_cost_usd": 0.02, "usage": {"input_tokens": 1460,
    #   "output_tokens": 88, "cache_read_input_tokens": 10, "cache_creation_input_tokens": 6369,
    #   "output_tokens_details": {"thinking_tokens": 25}}}, 3.2)
    # → end == 100.0, seconds == 3.2, result["tokens"] == {"input": 1460, "output": 88, "cache_read": 10,
    #   "cache_write": 6369, "thinking": 25}, result["cost"] == 0.02, error is None
    # finish({}, 1.0) 也不抛：tokens 各项为 None

def test_fail_keeps_steps():
    # begin → feed 一个 tool_use → fail("超时（120s）", 120.0) → error == "超时（120s）"，steps 仍有 1 项，end 不为 None

def test_begin_interrupts_unfinished_turn():
    # begin 两次 → 第一轮 error == "被打断（下一轮开始了）"，end 不为 None；第二轮进行中

def test_messages_between_turns_go_outside():
    # begin → finish → feed(user tool_result) → 返回里有 id == 0、reason == "outside" 的轮次，里面有这一步；
    # 第一轮的 steps 不变

def test_keeps_last_50_turns_and_reports_oldest():
    # begin/finish 55 轮 → since(0) 共 50 轮，ids == 6..55，oldest == 6

def test_since_is_incremental():
    # 两轮都 finish 后 v = since(0)["version"]；begin 第 3 轮 → since(v)["turns"] 只有 id 3
    # since(当前 version, 0.0) → turns == []，但 state 照样有

def test_after_larger_than_version_returns_everything(): ...  # since(10**9) → 所有轮次

def test_since_waits_for_change():
    # 另一个线程 0.1 秒后 begin；since(v, 2.0) 在 1 秒内返回且 turns 非空

def test_state_comes_from_callable_and_survives_errors():
    # t.state = lambda: {"failures": 2} → since(...)["state"]["failures"] == 2，且 state 里有 "turns"（已开始的轮数）
    # t.state = 抛异常的函数 → since 不抛，state == {"turns": ...}

def test_feed_survives_garbage():
    # feed(None)、feed({"type": "assistant"})、feed({"type": "assistant", "message": {"content": "字符串"}}) 都不抛

def test_chain_calls_fn_then_feeds_even_if_fn_raises(): ...

def test_feed_from_another_thread_while_polling():
    # 一个线程 begin 后 feed 200 条 text；主线程同时反复 since(0, 0.0)；结束后最后一轮 steps 恰好 200 条、没有异常
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_trace.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'skydango.brain.trace'`

- [ ] **Step 3: 实现 `src/skydango/brain/trace.py`**

一把 `threading.Condition` 保护：`deque` 存轮次（`maxlen` 不用：自己丢最早的并记 `oldest`）、`_version` 每次改动 +1 并写进该轮 `updated`、`notify_all`。
"轮外"轮次按需创建（`id = 0`，不计入 50 轮上限，在返回里排最前）。`since` 在锁里 `wait_for(lambda: self._version > after, timeout)`，
深拷贝要返回的轮次（`copy.deepcopy` 或 JSON 往返）后出锁；`state()` 在锁外调用，结果合并上 `{"turns": 已开始的轮数}`。
模块 docstring 写一段：它是干什么的、数据从哪来、只在 `run --brain --view` 时存在。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_trace.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/trace.py tests/test_brain_trace.py
git commit -m "feat(brain): BrainTrace——内存里记大脑最近 50 轮（收到的、说的、工具调用和返回、结果）"
```

---

### Task 2: `Brain` 接上 trace

**Files:**
- Modify: `src/skydango/brain/loop.py`（`Brain.__init__`、`wake`、`farewell`，新增 `trace_state`）
- Test: `tests/test_brain_loop.py`（加测试；`make()` 加 `trace=None` 参数透传）

**Interfaces:**
- Consumes: Task 1 的 `BrainTrace.begin / finish / fail / state`
- Produces:
  - `Brain(..., trace=None)`：有 trace 时在 `__init__` 里设 `trace.state = self.trace_state`
  - `Brain.trace_state() -> dict`：`{"model": cfg.model, "effort": cfg.effort, "failures": int, "retry_in": float | None（backoff 剩余秒数，没在退避为 None）, "offline": bool}`

- [ ] **Step 1: 写失败的测试**（假 trace 记下调用：`FakeTrace` 有 `calls` 列表和 `state` 属性）

```python
def test_wake_reports_to_trace(clock):
    # FakeSession(ok("好")) → brain.wake(clock(), "heartbeat")
    # → trace.calls == [("begin", "heartbeat", <发出去的 text>), ("finish", <result>, <seconds>)]
    #   且 begin 的 text == session.sent[0]，seconds >= 0

def test_failed_wake_reports_error(clock):
    # FakeSession(ClaudeError("超时")) → calls[-1][0] == "fail" 且 "超时" in calls[-1][1]

def test_farewell_reports_to_trace(clock, tmp_path):
    # 照 test_farewell_writes_summary_to_inbox 的做法 → calls[0] == ("begin", "farewell", SUMMARY_REQUEST)，calls[-1][0] == "finish"

def test_trace_state(clock):
    # make(..., trace=FakeTrace()) → trace.state is brain.trace_state
    # 正常：{"model": ..., "effort": ..., "failures": 0, "retry_in": None, "offline": False}
    # 失败一次后：failures == 1，retry_in ≈ BACKOFF[0]（10.0，clock 没动）

def test_trace_errors_do_not_break_wake(clock):
    # trace 的 begin/finish 都抛异常 → wake 照常把消息发出去、record_brain 照常记
```

现有测试不改、照样通过（它们不传 trace）。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_loop.py -q`
Expected: 新测试 FAIL（`unexpected keyword argument 'trace'`）

- [ ] **Step 3: 实现**

`wake` / `farewell`：发之前 `begin`，`session.send` 前后用 `self.clock()` 算耗时，成功 `finish`、`ClaudeError` 时 `fail(str(exc), 耗时)`。
三处调用都经一个小的私有方法 `_trace(name, *args)`：`trace is None` 直接返回，否则 `try/except Exception: log.debug(..., exc_info=True)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_loop.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/loop.py tests/test_brain_loop.py
git commit -m "feat(brain): 大脑每轮的开始 / 结果 / 失败交给 trace，trace_state 给网页栏头"
```

---

### Task 3: `/brain` 接口

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`Viewer.__init__` 加 `self.brain = None`；`Handler.do_GET` 加 `/brain`；模块 docstring 补一行）
- Test: `tests/test_viewer.py`

**Interfaces:**
- Consumes: Task 1 的 `BrainTrace.since(after, timeout) -> dict`
- Produces: `Viewer.brain` 属性（`BrainTrace | None`，Task 5 设置）；`GET /brain?after=<int>` → 404 / 200 JSON（`since` 的返回原样 `json.dumps(ensure_ascii=False)`）

- [ ] **Step 1: 写失败的测试**

```python
def test_brain_endpoint_404_without_trace():
    # viewer().start() → urlopen(url + "brain?after=0") 抛 HTTPError，code == 404

def test_brain_endpoint_serves_trace():
    # v.brain = BrainTrace()；begin("events", "你好")
    # → GET brain?after=0：200，json["turns"][0]["prompt"] == "你好"，有 "version" / "state"
    # → GET brain?after=<version>：约 2 秒后 200，turns == []
    # → GET brain?after=abc：当 0 处理，200
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viewer.py -q -k brain`
Expected: FAIL（`/brain` 现在回 404，第二个测试失败）

- [ ] **Step 3: 实现**：`after` 的解析照 `/snapshot`（`ValueError` → 0）；`since(after, WAIT)`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viewer.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): /brain 长轮询接口，给网页取大脑时间线"
```

---

### Task 4: 页面上的大脑时间线

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`PAGE` 的 HTML / CSS / JS）
- Test: `tests/test_viewer.py`
- 临时：`tmp/brain_page_demo.py`（手动核对页面用，不提交；`tmp/` 已 gitignore）

**Interfaces:**
- Consumes: Task 3 的 `/brain` JSON（字段见 Task 1 Interfaces）

页面行为（按 spec §3，全部照做）：
- 结构：`#stage` 里画面下面加 `<section id="brain" hidden>`：栏头 `#brain-state` + 开关 `#brain-acted`（「只看做了事的轮次」复选框）+ 列表 `#brain-turns`
- 启动时请求一次 `/brain?after=0`：404 → 保持 `hidden`、不再请求；否则去掉 `hidden`，进入长轮询循环（独立于画面的 `loop()`，不受「暂停」影响）；fetch 失败 → 栏头显示「连不上」，1 秒后重试
- 本地用 `Map(id → turn)` 合并增量；返回的 `version` 比上次小 → 清空重来；删掉 `id < oldest` 的（`id = 0` 的轮外除外）
- 栏头：`大脑 · {model} / {effort} · 已醒 {turns} 轮 · 在线`；`retry_in` 非空 → 「连续失败 {failures} 次，{retry_in 取整} 秒后重试」（黄）；`offline` → 「已转备用回复（DeepSeek）」（红）
- 列表按 `id` 倒序（轮外排最后）；每轮一个 `<details>`，`open` 状态用一个 `Set` 记住，重画后恢复；进行中（`end === null`）的默认展开
- 摘要行：`HH:MM:SS`、原因（`events`→新消息 / 事件，`heartbeat`→心跳，`farewell`→退出前总结，`outside`→轮外）、工具计数（`say×1 emote×1`，没有工具也没 text 步骤 → 「（什么都没做）」）、`seconds.toFixed(1)s`、`in 1.5k · out 88 · 思考 25`（数字 ≥1000 显示 `x.xk`）；有 `error` → 摘要写「失败：{error}」且整行红；进行中 → 「进行中…」
- 「只看做了事的轮次」勾上时隐藏：没有 `tool` 步骤、没有 `text` 步骤、也没 `error` 的轮次（进行中的不隐藏）
- 展开内容：「收到」（`<pre>` 原样）→ 各步骤（💭 思考 / 💬 说 / 🔧 调用 {name} + `JSON.stringify(input, null, 2)` / ↩ 返回，`error` 标黄，超过 10 行时外面再包一层 `<details>` 摘要显示前 10 行）→ 「结果」（subtype、耗时、tokens 各项、`cost` 保留 4 位小数「参考」；`error` 红色）→ 「复制这一轮」按钮（拼纯文本，`navigator.clipboard.writeText`，失败时退回 `document.execCommand("copy")` 用临时 textarea）
- 所有数据文字用 `textContent`；样式复用 CSS 变量，等宽用 `ui-monospace, Consolas, monospace`；窄屏（≤900px）照常在画面下方

- [ ] **Step 1: 写失败的测试**

```python
def test_page_has_brain_section():
    # 'id="brain"'、'id="brain-state"'、'id="brain-turns"'、'id="brain-acted"'、'/brain?after=' 都在 PAGE 里

def test_page_never_uses_innerhtml_for_brain_data():
    # 取 PAGE 里大脑那段脚本（在 "// ---- brain ----" 注释之后）→ 不含 "innerHTML"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viewer.py -q -k page`
Expected: 新测试 FAIL

- [ ] **Step 3: 实现页面**（大脑相关的 JS 放在 `// ---- brain ----` 注释之后）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viewer.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 在浏览器里核对页面**

写 `tmp/brain_page_demo.py`：起 `Viewer(ViewerConfig(port=8766))`，喂一帧灰图；`v.brain = BrainTrace()`，`state` 设成返回
`{"model": "sonnet", "effort": "low", "failures": 0, "retry_in": None, "offline": False}` 的函数；
造 4 轮：成功（text + say 工具 + 返回）、心跳什么都没做、失败（超时，带一个 tool 步骤）、进行中（每 2 秒 feed 一条 text，含 `<b>粗体?</b>`、一个 15 行的工具返回、一个 `[图片]` 返回）。
用内置浏览器打开 `http://127.0.0.1:8766/`，逐项核对：栏头、倒序、失败行红色、进行中自动展开并实时增长、展开状态刷新后保持、
「只看做了事的轮次」、长返回折叠、`<b>` 原样显示为文字、复制按钮、画面「暂停」时大脑栏照常更新、窄屏（`resize_window` mobile）布局；截图存 `tmp/`。
发现问题回到 Step 3 修。核对完停掉脚本进程（按进程树结束，确认没有残留 python）。

- [ ] **Step 6: 提交**

```bash
git add src/skydango/vision/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): 画面下方的大脑时间线（每轮可展开：收到的、说的、工具、结果；只看做了事的；复制这一轮）"
```

---

### Task 5: `run --brain --view` 接线、文档、真机验证

**Files:**
- Modify: `src/skydango/cli.py`（`_run_brain`）
- Modify: `CLAUDE.md`（「识别可视化」「统管大脑」两节各补一句）
- Modify: `docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md`（状态改「已实现」，真机核对的结论写进 §6）
- Test: `tests/test_cli_brain.py`

**Interfaces:**
- Consumes: Task 1 `BrainTrace`、`chain`；Task 2 `Brain(trace=...)`；Task 3 `Viewer.brain`

- [ ] **Step 1: 写失败的测试**

照 `tests/test_cli_brain.py::test_run_brain_wires_everything` 的做法（假 `claude` 进程 `fake_claude.py`，端到端跑 `_run_brain`），
把那段准备代码抽成一个辅助函数两个测试共用：

```python
class FakeViewer:
    brain = None
    def update(self, *a, **k): return True

def test_run_brain_with_viewer_records_turns(tmp_path, monkeypatch):
    # cli._run_brain(cfg, run, no_emotes=True, duration=3.0, viewer=v)
    # → isinstance(v.brain, BrainTrace)；turns = v.brain.since(0, 0.0)["turns"]
    #   至少一轮 reason == "heartbeat"、"没有新事件" in prompt、end 不为 None、error is None
    #   state 里 model == cfg.brain.model

def test_run_brain_without_viewer_still_works(...):  # 就是现有的 test_run_brain_wires_everything，不用新写
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cli_brain.py -q`
Expected: 新测试 FAIL

- [ ] **Step 3: 实现**：`trace = BrainTrace() if viewer is not None else None`；`on_message = trace.chain(log_brain_message) if trace else log_brain_message`；`Brain(..., trace=trace)`；`viewer.brain = trace`

- [ ] **Step 4: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 PASS（基线 557 passed, 5 skipped，加上新测试）

- [ ] **Step 5: 真机验证**（先按 `docs/game-ops.md` 确认游戏状态：聊天记录面板开着、输入框没开）

Run: `python -m skydango run --brain --view --duration 300`（dry-run；**不要**用 `timeout` 包）
在内置浏览器看 `http://127.0.0.1:8765/`：轮次、原因、收到的消息、工具调用和返回、结果都显示，进行中的一轮实时更新；截图存 `tmp/`。
翻 `runs/<这次>/agent.log` 或临时在 `feed` 里记 DEBUG，核对 stream-json 里**实际有没有 `user` / `tool_result` 消息、有没有思考原文**：
- 有 → 照现在的做法
- 没有 `tool_result` → 改成在 `ToolBox.run` 里把 (name, args, out, err) 交给 trace（新增 `BrainTrace.tool_result(name, text, error)`，按 spec §6，接口不变），补测试，再验证一次

结束后用 `Get-CimInstance Win32_Process` 确认没有残留 python。

- [ ] **Step 6: 更新文档**：CLAUDE.md 两节各一句；spec 状态改「已实现」，§6 写上核对结论

- [ ] **Step 7: 提交**

```bash
git add src/skydango/cli.py tests/test_cli_brain.py CLAUDE.md docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md
git commit -m "feat(brain): run --brain --view 时网页上有大脑时间线；文档"
```
