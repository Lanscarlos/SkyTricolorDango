# DeepSeek 备用大脑 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude 订阅额度不足（429）时，把大脑从 Claude Code 永久换成 DeepSeek function-calling 循环，复用同一个 `ToolBox`，盲跑 + 视觉留后路 + 不自动切回。

**Architecture:** 新增 `DeepSeekBrain`（`send(text) -> result` 签名对齐 `BrainSession`），内部跑 OpenAI function-calling 循环、逐个经 `ToolBox.run` 执行工具；`Brain` 加一个 `fallback_session`，`_failed` 里 `limit=True` 时永久 `self.session = fallback_session`。工具 schema 单一来源 `openai_tools(toolbox, blind=True)`，盲跑排除 5 个视觉工具。

**Tech Stack:** Python 3.13、openai SDK（已装）、现有 `brain/loop.py` + `brain/tools.py` + `chat/llm.py`、pytest（`.venv\Scripts\python.exe -m pytest -q`）。

**Spec:** [docs/superpowers/specs/2026-10-03-deepseek-fallback-brain-design.md](../specs/2026-10-03-deepseek-fallback-brain-design.md)

## Global Constraints

- 测试跑 `.venv\Scripts\python.exe -m pytest -q`（合成画面 + 假设备，不碰模拟器）。
- `fallback = False` / 没 Key / 没 `fallback_session` 时，`loop.py` 行为逐字照旧（现有测试不破）。
- 盲跑排除集固定：`{"look", "look_at", "look_person", "look_around", "check_friend"}`；`panel_read` 盲跑下去掉 `image` 参数。
- 不自动切回：DeepSeek 失败只走 `BACKOFF` 退避重试，`session` 一直是 DeepSeek。
- `openai` 模块按仓库惯例延迟导入（`build_client` 内 `from openai import OpenAI`），模块顶层不硬 import。

## Review Focus

1. **DeepSeek 返回的 `tool_calls[].function.arguments` 不是合法 JSON**（幻觉/截断）→ 不能 `json.loads` 崩掉整轮，当一次工具失败喂回模型。→ Task 2 测试 `test_bad_json_arguments_fed_back_as_error`。
2. **盲跑下模型硬调被排除的工具（如 `look`）** → schema 里没有，但模型可能硬调；`DeepSeekBrain` 只执行 schema 里出现的名字，别的回"没有这个工具"喂回。→ Task 2 测试 `test_unknown_tool_not_executed`。
3. **工具结果不是 `str`（图片块列表）** → OpenAI tool 消息的 `content` 必须是字符串；非 str 用 `json.dumps` 兜底。→ Task 2 测试 `test_non_string_result_serialized`。
4. **额度触发但没配 DeepSeek（`fallback=False` 或 Key 缺失）** → 退化成现有行为（退避 + 纯文字备用），不报错。→ Task 3 测试 `test_limit_without_fallback_still_backs_off` + Task 4 测试 `_fallback_brain` 返回 None。
5. **`force_fallback=True` 但 `fallback_session` 是 None** → 静默忽略（照旧用 Claude），不崩。→ Task 4 测试 `test_force_fallback_without_session_is_noop`。

---

### Task 1: `brain/llm_tools.py` — OpenAI 工具 schema 单一来源

**Files:**
- Create: `src/skydango/brain/llm_tools.py`
- Test: `tests/test_brain_llm_tools.py`

**Interfaces:**
- Produces: `openai_tools(toolbox, *, blind: bool = True) -> list[dict]`；模块常量 `BLIND_EXCLUDE: set[str]`、`PARAMETERS: dict[str, list[tuple[str, str, object]]]`。
- 依赖 `skydango.brain.tools` 的 `descriptions`、`CALL_DESCRIPTION`、`INTROSPECT_DESCRIPTION`（描述已有单一来源，直接复用）。

- [ ] **Step 1: 写失败测试** `tests/test_brain_llm_tools.py`

用 `types.SimpleNamespace(backstage=False, calling=False, body=SimpleNamespace(env=None))` 当假 toolbox（无 sweep）。断言：

```python
def test_blind_excludes_vision_tools():
    tools = openai_tools(fake_toolbox(), blind=True)
    names = [t["function"]["name"] for t in tools]
    assert not (set(names) & BLIND_EXCLUDE)
    assert set(names) >= {"say", "emote", "status", "camera", "track", "panel_read", "panel_press"}

def test_sighted_keeps_vision_tools():
    names = [t["function"]["name"] for t in openai_tools(fake_toolbox(), blind=False)]
    assert set(names) >= {"look", "look_person", "check_friend"}

def test_panel_read_has_no_image_when_blind():
    pr = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "panel_read")
    assert "image" not in pr["function"]["parameters"]["properties"]

def test_backstage_and_call_flags_add_tools():
    tb = fake_toolbox(backstage=True, calling=True)
    names = [t["function"]["name"] for t in openai_tools(tb, blind=True)]
    assert "introspect" in names and "call" in names

def test_required_and_defaults():
    say = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "say")
    assert say["function"]["parameters"]["required"] == ["text"]
    cl = next(t for t in openai_tools(fake_toolbox(), blind=True) if t["function"]["name"] == "chat_log")
    assert cl["function"]["parameters"]["properties"]["n"]["default"] == 20  # OpenAI 用 "default"，非 required
```

- [ ] **Step 2: 跑测试确认失败**（`openai_tools` 未定义）

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_llm_tools.py`
Expected: FAIL（`ModuleNotFoundError` / `NameError`）

- [ ] **Step 3: 实现 `openai_tools` 于 `src/skydango/brain/llm_tools.py`**

`PARAMETERS` 用下面这张表（`(参数名, JSON 类型, 默认值)`；默认值是 `None` 表示必填进 `required`）：

```python
PARAMETERS = {
    "look": [("image", "boolean", False)], "look_at": [("x", "integer", None), ("y", "integer", None), ("w", "integer", None), ("h", "integer", None)],
    "look_person": [("name", "string", None)], "look_around": [], "call": [], "status": [],
    "chat_log": [("n", "integer", 20)], "recall": [("query", "string", ""), ("who", "string", ""), ("days", "integer", 14)],
    "say": [("text", "string", None)], "emote": [("name", "string", None)],
    "set_request_policy": [("who", "string", None), ("kind", "string", None), ("accept", "boolean", None)],
    "camera": [("action", "string", None), ("steps", "integer", 1)], "camera_reset": [],
    "attention": [("mode", "string", None), ("focus", "string", "")], "move": [("direction", "string", None), ("steps", "integer", 1), ("force", "boolean", False)],
    "check_friend": [("x", "integer", None), ("y", "integer", None)],
    "track": [("name", "string", None), ("seconds", "integer", 30)], "find": [("name", "string", None), ("seconds", "integer", 30)], "stop_task": [],
    "panel_read": [("image", "boolean", False)], "panel_press": [("button", "string", None)], "panel_close": [], "introspect": [("topic", "string", None)],
}
```

`openai_tools`：按 `descriptions(sweep)` 里工具顺序输出（`sweep = hasattr(toolbox.body.env, "sweep")`，和 `mcp_server.build_server` 一致）；`blind=True` 时跳过 `BLIND_EXCLUDE` 且 `panel_read` 去掉 `image`；末尾按 `toolbox.calling` / `toolbox.backstage` 追加 `call` / `introspect`。每条 schema 形如 `{"type":"function","function":{"name","description","parameters":{"type":"object","properties","required"}}}`；默认值放进 `properties[name]["default"]`（不是 required）。文件顶部加注释：参数表要和 `tools.py` 的 `_bind`、`mcp_server.py` 的 `@srv.tool` 签名三处同步。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_llm_tools.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/llm_tools.py tests/test_brain_llm_tools.py
git commit -m "feat(brain): OpenAI 工具 schema 单一来源（盲跑排除视觉工具）"
```

---

### Task 2: `brain/deepseek.py` — `DeepSeekBrain` function-calling 循环

**Files:**
- Create: `src/skydango/brain/deepseek.py`
- Test: `tests/test_brain_deepseek.py`

**Interfaces:**
- Consumes: `skydango.brain.claude.ClaudeError`；`skydango.config.LlmConfig`；`skydango.chat.llm.read_key`。
- Produces:
  - `DeepSeekBrain(client, system: str, toolbox, tools: list[dict], *, model: str, temperature: float, max_tokens: int, max_steps: int = 6, turn_timeout: float = 120.0, clock: Callable[[], float] = time.monotonic)`，方法 `send(self, text: str) -> dict`（成功 `{"result": str, "subtype": "success"}`；失败抛 `ClaudeError(msg, limit=False)`）。
  - `FALLBACK_NOTE: str`（备用大脑提示词追加段，spec §4 原文）。
  - `build_client(llm: LlmConfig, api_key: str | None = None)`：延迟 `from openai import OpenAI`，返回 `OpenAI(base_url=llm.base_url or None, api_key=api_key or read_key(llm.api_key_env), timeout=llm.timeout, max_retries=0)`。

- [ ] **Step 1: 写失败测试** `tests/test_brain_deepseek.py`

假 client（`client.chat.completions.create(**kw)`）：脚本化的 `messages` 队列，每条是 `SimpleNamespace(content=str|None, tool_calls=list|None)`，`tool_calls` 每条 `SimpleNamespace(id="c1", function=SimpleNamespace(name="say", arguments='{"text":"你好"}'))`；队列元素是 `Exception` 就直接 raise。假 toolbox `run(name, args) -> (out, err)`。断言：

```python
def test_no_tool_calls_returns_text():
    brain = DeepSeekBrain(client(ok("好")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗") == {"result": "好", "subtype": "success"}

def test_single_tool_call_then_text():
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("say", {"text": "你好"})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    r = brain.send("在吗")
    assert r["result"] == "好" and tb.calls == [("say", {"text": "你好"})]

def test_iteration_cap():   # 模型不停调工具：到 max_steps+2 轮停
    brain = DeepSeekBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, max_steps=2)
    r = brain.send("在吗")
    assert r["subtype"] == "success"

def test_bad_json_arguments_fed_back_as_error():  # Review Focus 1
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("say", "not json")])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    assert brain.send("在吗")["subtype"] == "success"  # 不崩，把解析错误喂回模型

def test_unknown_tool_not_executed():  # Review Focus 2：模型硬调 look
    tb = toolbox()
    brain = DeepSeekBrain(client(calls_then_ok([("look", {"image": False})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")
    assert tb.calls == []  # 没执行，回"没有这个工具"

def test_non_string_result_serialized():  # Review Focus 3
    tb = toolbox(ret=([{"type": "image", "source": {}}], False))
    brain = DeepSeekBrain(client(calls_then_ok([("say", {})])), "sys", tb, tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096)
    brain.send("在吗")  # 不崩；tool 消息 content 被 json.dumps 成字符串

def test_api_error_raises_not_limit():
    brain = DeepSeekBrain(client(Exception("500")), "sys", toolbox(), [], model="deepseek-chat", temperature=0.8, max_tokens=4096)
    with pytest.raises(ClaudeError) as e:
        brain.send("在吗")
    assert e.value.limit is False

def test_turn_timeout():
    # deadline 已经是过去（turn_timeout=0）：循环开头就该抛超时，不等 create
    brain = DeepSeekBrain(client(always_calls()), "sys", toolbox(), tools=[say_schema()],
                          model="deepseek-chat", temperature=0.8, max_tokens=4096, turn_timeout=0.0, clock=lambda: 0.0)
    with pytest.raises(ClaudeError):
        brain.send("在吗")

def test_fallback_note_nonempty():
    assert "看不到画面" in FALLBACK_NOTE

def test_build_client(monkeypatch):
    made = {}
    mod = types.ModuleType("openai"); mod.OpenAI = lambda **kw: made.update(kw) or object()
    monkeypatch.setitem(sys.modules, "openai", mod)
    monkeypatch.setenv("SKYDANGO_TEST_KEY", "k")
    build_client(LlmConfig(base_url="https://api.deepseek.com", api_key_env="SKYDANGO_TEST_KEY"))
    assert made["base_url"] == "https://api.deepseek.com" and made["max_retries"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_deepseek.py`
Expected: FAIL（`DeepSeekBrain` 未定义）

- [ ] **Step 3: 实现 `src/skydango/brain/deepseek.py`**

`send()` 循环按 spec §1：构造函数把 `system` / `model` / `max_tokens`（及 client/toolbox/tools/max_steps/turn_timeout/clock）存成同名 `self.*`；`messages=[{"role":"system",...},{"role":"user","content":text}]`；循环**每轮开头先 `if clock() >= deadline: raise ClaudeError("超时")`**，再 `client.chat.completions.create(model=..., messages=messages, tools=tools, temperature=..., max_tokens=...)`，取 `resp.choices[0].message`；没有 `tool_calls` 就 break；有则逐个执行——**只执行 name 在 `{t["function"]["name"] for t in tools}` 里的**（别的回"没有这个工具：X"，不崩）；`arguments` 用 `try: json.loads(...) except ValueError: args=None`，解析失败也当一次工具失败喂回；`toolbox.run(name, args or {})` 的结果非 str 时 `json.dumps(out, ensure_ascii=False)`；append `{"role":"tool","tool_call_id":tc.id,"content":content}`。循环上限 `max_steps + 2` 轮工具。API 异常抛 `ClaudeError(msg, limit=False)`。最后 `return {"result": (final_content or ""), "subtype": "success"}`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_deepseek.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/deepseek.py tests/test_brain_deepseek.py
git commit -m "feat(brain): DeepSeekBrain function-calling 循环（盲跑 + 失败退避不切回）"
```

---

### Task 3: `loop.py` 切换逻辑 + 时间线标识

**Files:**
- Modify: `src/skydango/brain/loop.py`（`Brain.__init__`、`_failed`、`trace_state`）
- Modify: `src/skydango/console/static/brainlog.js`、`src/skydango/vision/static/brain_trace.js`（`on_fallback` 显示）
- Test: `tests/test_brain_loop.py`（新增 + 更新 `test_trace_state`）

**Interfaces:**
- Consumes: `DeepSeekBrain`（其实只要任何带 `send(text)` 的对象，测试用假 session）。
- Produces: `Brain(..., fallback_session=None)`；`self.on_fallback: bool`；`trace_state()` 多一个键 `"on_fallback"`。

- [ ] **Step 1: 写失败测试**（`tests/test_brain_loop.py`）

`make()` 加一个 `fallback_session=None` 透传。新增：

```python
def test_limit_swaps_to_fallback_session(clock):
    claude = FakeSession(ClaudeError("hit your limit", limit=True))
    fb = FakeSession(ok("deepseek 说的"))
    brain, _, _, _ = make(clock, claude, fallback_session=fb)
    brain.wake(clock(), "heartbeat")
    assert brain.session is fb and brain.on_fallback is True
    assert brain.failing_since is None and brain.failures == 0
    assert brain.backoff_until == float("-inf") and not brain.offline(clock())

def test_after_swap_next_wake_uses_fallback(clock):
    fb = FakeSession(ok("第一句"), ok("第二句"))
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("limit", limit=True)), fallback_session=fb)
    brain.wake(clock(), "heartbeat")
    assert len(fb.sent) == 0  # 这一轮是 Claude 挂的那轮
    brain.wake(clock(), "heartbeat")
    assert len(fb.sent) == 1

def test_fallback_failure_does_not_switch_back(clock):
    fb = FakeSession(ClaudeError("deepseek 挂了"))
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("limit", limit=True)), fallback_session=fb)
    brain.wake(clock(), "heartbeat")
    brain.wake(clock(), "heartbeat")  # DeepSeek 失败
    assert brain.session is fb and brain.on_fallback is True  # 不切回
    assert brain.offline(clock() + brain.cfg.offline_fallback + 1)

def test_limit_without_fallback_still_backs_off(clock):  # Review Focus 4
    brain, _, _, _ = make(clock, FakeSession(ClaudeError("hit your limit", limit=True)))
    brain.wake(clock(), "heartbeat")
    assert brain.backoff_until == clock() + 600 and brain.on_fallback is False
```

更新现有 `test_trace_state`：字典加 `"on_fallback": False`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_loop.py`
Expected: FAIL（`fallback_session` 未接 / `on_fallback` 缺失）

- [ ] **Step 3: 实现于 `src/skydango/brain/loop.py`**

`Brain.__init__` 加参数 `fallback_session=None`，存 `self.fallback_session = fallback_session`、`self.on_fallback = False`。`_failed` 里，在 `limit = bool(getattr(exc, "limit", False))` 之后加：`if limit and self.fallback_session is not None and not self.on_fallback: self.session = self.fallback_session; self.on_fallback = True; self.failures = 0; self.failing_since = None; self.backoff_until = float("-inf"); self.limited = False; log.warning("Claude 额度用完，切到 DeepSeek 备用大脑"); return`。`trace_state()` 返回字典加 `"on_fallback": self.on_fallback`。`offline()` 不改（切换后 `failing_since` 已清，DeepSeek 失败才重新累积，语义自然正确）。

前端两处：在现有 `if(st.offline){...已转备用回复（DeepSeek）...}` 前面加 `if(st.on_fallback){text+="备用大脑（DeepSeek）";cls="bad"}` 分支（不加测试，前端不在 pytest 里跑）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_brain_loop.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/loop.py src/skydango/console/static/brainlog.js src/skydango/vision/static/brain_trace.js tests/test_brain_loop.py
git commit -m "feat(brain): 额度用完切 DeepSeek 备用大脑（不自动切回）+ 时间线标识"
```

---

### Task 4: 配置字段 + `cli.py` 接线

**Files:**
- Modify: `src/skydango/config.py`（`BrainConfig`：加三字段）
- Modify: `src/skydango/cli.py`（`_run_brain`：提 prompt、建 fallback、传参、force_fallback 切换；新增 `_fallback_brain(cfg, toolbox, prompt) -> DeepSeekBrain | None`）
- Test: `tests/test_cli_brain.py`（新增）

**Interfaces:**
- Consumes: Task 1 `openai_tools`、Task 2 `DeepSeekBrain` / `build_client` / `FALLBACK_NOTE`、Task 3 `Brain(fallback_session=…)`。
- Produces: `_fallback_brain(cfg: Config, toolbox, prompt: str) -> DeepSeekBrain | None`；`BrainConfig.fallback` / `fallback_max_tokens` / `force_fallback`。

- [ ] **Step 1: 写失败测试**（`tests/test_cli_brain.py`）

```python
def test_fallback_brain_off_when_disabled(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.fallback = False
    assert cli._fallback_brain(cfg, object(), "p") is None

def test_fallback_brain_none_when_no_key(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    monkeypatch.setattr("skydango.brain.deepseek.read_key", lambda name: (_ for _ in ()).throw(RuntimeError("没 Key")))
    assert cli._fallback_brain(cfg, object(), "p") is None

def test_fallback_brain_built(tmp_path, monkeypatch):
    import skydango.brain.deepseek as ds
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    made = {}
    monkeypatch.setattr(ds, "build_client", lambda llm, key=None: made.update(key=key) or object())
    brain = cli._fallback_brain(cfg, object(), "prompt")
    assert isinstance(brain, ds.DeepSeekBrain) and brain.system == "prompt\n\n" + ds.FALLBACK_NOTE
    assert brain.model == cfg.llm.model and brain.max_tokens == cfg.brain.fallback_max_tokens

def test_run_brain_passes_fallback_session(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    sentinel = type("FB", (), {"send": lambda self, t: {"result": "好", "subtype": "success"}})()
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p: sentinel)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].fallback_session is sentinel

def test_force_fallback_swaps_session(tmp_path, monkeypatch):
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.force_fallback = True
    sentinel = type("FB", (), {"send": lambda self, t: {"result": "好", "subtype": "success"}})()
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p: sentinel)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].session is sentinel and seen[0].on_fallback is True

def test_force_fallback_without_session_is_noop(tmp_path, monkeypatch):  # Review Focus 5
    cfg, run, _ = fake_brain_run(tmp_path, monkeypatch)
    cfg.brain.force_fallback = True
    monkeypatch.setattr(cli, "_fallback_brain", lambda cfg, tb, p: None)
    seen = []
    cli._run_brain(cfg, run, no_emotes=True, duration=1.0, on_ready=lambda parts: seen.append(parts.brain))
    assert seen[0].session is not None and seen[0].on_fallback is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_cli_brain.py`
Expected: FAIL（`_fallback_brain` 未定义 / `fallback_max_tokens` 缺字段）

- [ ] **Step 3: 实现**

`config.py` 的 `BrainConfig` 加：`fallback: bool = True`、`fallback_max_tokens: int = 4096`、`force_fallback: bool = False`（放 `owner_window` 附近，注释照 spec §5）。

`cli.py` 新增模块级 `def _fallback_brain(cfg, toolbox, prompt)`：`if not cfg.brain.fallback: return None`；`try: client = build_client(cfg.llm)` `except RuntimeError: log.warning(...); return None`；返回 `DeepSeekBrain(client, prompt + "\n\n" + FALLBACK_NOTE, toolbox, openai_tools(toolbox, blind=True), model=cfg.llm.model, temperature=cfg.llm.temperature, max_tokens=cfg.brain.fallback_max_tokens, max_steps=cfg.brain.max_steps, turn_timeout=cfg.brain.turn_timeout)`。需要 `from .brain.deepseek import DeepSeekBrain, FALLBACK_NOTE, build_client` 和 `from .brain.llm_tools import openai_tools`（在函数内 import，仿 `_run_brain` 现有风格）。

`_run_brain` 里：把 `brain_prompt(...)` 现算成变量 `prompt`（现在内联在 `BrainSession(...)` 里，约 [cli.py:2429](src/skydango/cli.py#L2429)），`BrainSession(..., prompt, ...)` 用变量；`session` 之后加 `fallback_session = _fallback_brain(cfg, toolbox, prompt)`；`Brain(...)` 调用加 `fallback_session=fallback_session`；`brain` 构造完、`if viewer is not None:` 之前加 `if cfg.brain.force_fallback and fallback_session is not None: brain.session = fallback_session; brain.on_fallback = True`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest -q tests/test_cli_brain.py tests/test_brain_loop.py tests/test_brain_llm_tools.py tests/test_brain_deepseek.py`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS（`fallback` 关掉时旧行为逐字照旧）

- [ ] **Step 6: 提交**

```bash
git add src/skydango/config.py src/skydango/cli.py tests/test_cli_brain.py
git commit -m "feat(brain): [brain] fallback/fallback_max_tokens/force_fallback + cli 接线"
```

---

### 收尾

全部 Task 完成后：用 `verification-before-completion` 跑一遍全量测试拿证据，再按项目规矩合并进 main 并推送（`finishing-a-development-branch`）。真机/沙盒验证步骤见 spec §7（`force_fallback=true` 起沙盒/真机看时间线），留到实现后单独做。
