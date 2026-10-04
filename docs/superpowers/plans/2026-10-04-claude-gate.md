# Claude 总闸 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude 额度 / 认证出错后这次运行里不再起注定失败的 `claude -p`：大脑切 DeepSeek（认证错也切），随手记 / 整理 notes / 反思改走 DeepSeek，眼睛和装扮描述停用；时间线补全 DeepSeek 的中间文字、用量、模型名。

**Architecture:** `brain/claude.py` 加一个线程安全的 `ClaudeGate`（一次运行一个，关了不再开）和两个包装（文字的 `GatedLlm`、看图的 `gated_describe`）；`cli._run_brain` 建闸，交给大脑循环、NotesKeeper、Reflector、Eyes、Wardrobe。大脑循环在额度 / 认证错时关闸并切备用大脑，闸被别处先关时下一轮开始前就切。

**Tech Stack:** Python 3.13、pytest；DeepSeek 走现有 `chat/llm.py` 的 `make_llm`（OpenAI 兼容）。

**Spec:** `docs/superpowers/specs/2026-10-04-claude-gate-design.md`

## Global Constraints

- 只有额度（`limit`：429 / limit / 上限 / 额度，原有判断）和认证（`auth`：`api_error_status == 401`，或文字含 `authentication` / `invalid api key` / `invalid x-api-key` / `oauth token`，不分大小写）关闸；超时、进程起不来、其他错误不关
- 闸这次运行里关了就不再开；不存盘
- 改走 DeepSeek 的文字调用用 `make_llm(dataclasses.replace(cfg.llm, max_tokens=cfg.brain.fallback_max_tokens))`；`[llm]` 没 Key / 没装 openai → backup = None
- 整理 notes 失败后 `NOTES_RETRY = 600.0` 秒内不再试
- 关闸只记一次 WARNING："Claude 不能用了（…），这次运行里记忆、反思改走 DeepSeek，眼睛和装扮描述停用"；之后改道 / 跳过记 DEBUG
- `memory update` 命令、管理面板测试按钮、标注工具等单独的一次性命令不接闸（照旧）
- `[brain] force_fallback` 不关闸（Claude 可能还好着，记忆照走 Claude）
- 测试命令：`python -m pytest -q -p no:cacheprovider <文件>`；全量放后台跑；代码注释、日志用中文

## Review Focus

1. 好几个线程（记忆、反思、眼睛、大脑）几乎同时撞上 401：闸只关一次、WARNING 只一条、没有异常 —— Task 1 `test_gate_trip_from_many_threads_once`
2. Claude 只是一次超时：不能关闸、不能改道 DeepSeek（一次慢就整夜不用 Claude）—— Task 2 `test_gated_llm_timeout_does_not_trip`
3. 闸关了、`[llm]` 又没 Key：随手记 / 整理照样失败，但不起 claude 进程、整理不每轮重试 —— Task 2 `test_gated_llm_no_backup_raises_without_calling_claude` + Task 4
4. `force_fallback` 调试开关：大脑一上来就用 DeepSeek，但闸开着、记忆照走 Claude —— Task 6 `test_force_fallback_keeps_gate_open`
5. 反思改走 DeepSeek 时回复长度：`[llm] max_tokens` 默认 200 会把日记截断，backup 必须用 `fallback_max_tokens` —— Task 6 `test_gated_backup_uses_fallback_max_tokens`

---

### Task 1: 认证错误识别 + `ClaudeGate`

**Files:**
- Modify: `src/skydango/brain/claude.py`（`ClaudeError`、`check_result`，新 `ClaudeGate`、`claude_down`）
- Test: `tests/test_brain_claude.py`

**Interfaces:**
- Produces:
  - `ClaudeError(message: str, limit: bool = False, auth: bool = False)`，属性 `.limit` `.auth`
  - `claude_down(exc: BaseException) -> str | None`：`"limit"` / `"auth"` / `None`（只认 `ClaudeError`）
  - `class ClaudeGate`：`ok() -> bool`；`trip(kind: str, detail: str) -> bool`（这次是不是刚关上）；属性 `reason: str | None`（`"额度用完"` / `"认证失败：<detail 前 80 字>"`）；内部 `threading.Lock`

- [ ] **Step 1: 写失败的测试**

```python
def test_check_result_recognises_auth_errors():
    for m in ({"subtype": "error", "is_error": True, "result": "x", "api_error_status": 401},
              {"subtype": "success", "is_error": True, "result": "401 Authentication Fails, Your api key: ****gwAA is invalid"},
              {"subtype": "success", "is_error": True, "result": "OAuth token has expired"}):
        with pytest.raises(ClaudeError) as e:
            check_result(m)
        assert e.value.auth and not e.value.limit and claude_down(e.value) == "auth"

def test_check_result_limit_and_other():
    with pytest.raises(ClaudeError) as e:
        check_result({"subtype": "error", "is_error": True, "result": "x", "api_error_status": 429})
    assert claude_down(e.value) == "limit"
    assert claude_down(ClaudeError("超时")) is None and claude_down(RuntimeError("x")) is None

def test_gate_trips_once_and_stays_closed(caplog):
    g = ClaudeGate()
    assert g.ok() and g.reason is None
    assert g.trip("auth", "401 bad key") is True
    assert g.trip("limit", "429") is False
    assert not g.ok() and g.reason.startswith("认证失败")
    assert sum("Claude 不能用了" in r.getMessage() for r in caplog.records) == 1

def test_gate_trip_from_many_threads_once(caplog):  # Review Focus 1
    g = ClaudeGate()
    ts = [threading.Thread(target=g.trip, args=("auth", "401")) for _ in range(20)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not g.ok() and sum("Claude 不能用了" in r.getMessage() for r in caplog.records) == 1
```

- [ ] **Step 2: 跑测试确认失败**（`ImportError: claude_down` / `ClaudeGate`）

Run: `python -m pytest -q -p no:cacheprovider tests/test_brain_claude.py`

- [ ] **Step 3: 实现**：`AUTH_WORDS = ("authentication", "invalid api key", "invalid x-api-key", "oauth token")`；`check_result` 里 `auth = status == 401 or any(w in detail.lower() for w in AUTH_WORDS)`、`limit` 判断不变（两个都成立时只算 `limit`）；`ClaudeGate.trip` 首次关闸时 `log.warning` 全局约束里那句

- [ ] **Step 4: 跑测试确认通过**（同上命令）

- [ ] **Step 5: 提交** `git commit -m "feat(brain): 认出 Claude 认证错误；ClaudeGate 总闸"`

### Task 2: `GatedLlm`（文字）和 `gated_describe`（看图）

**Files:**
- Modify: `src/skydango/brain/claude.py`
- Test: `tests/test_brain_claude.py`

**Interfaces:**
- Consumes: Task 1 的 `ClaudeGate`、`claude_down`
- Produces:
  - `class GatedLlm`：`__init__(self, claude, backup, gate: ClaudeGate)`（`claude` / `backup` 都是有 `complete(system, messages, max_tokens=None) -> str` 的对象，`backup` 可为 None）；`complete(system, messages, max_tokens=None) -> str`
  - `gated_describe(describe: Callable[[object], str], gate: ClaudeGate) -> Callable[[object], str]`：闸关着抛 `ClaudeError("Claude 不能用了，看不了图", limit=True)`、不调 `describe`；`describe` 抛额度 / 认证错 → `gate.trip` 后原样抛

- [ ] **Step 1: 写失败的测试**（假 claude / backup：记调用次数，按脚本返回或抛）

```python
def test_gated_llm_uses_claude_while_open(): ...            # 闸开：claude 调 1 次、backup 0 次、返回 claude 的
def test_gated_llm_auth_error_trips_and_uses_backup(): ...  # claude 抛 auth → gate.ok() False、这一笔返回 backup 的
def test_gated_llm_closed_gate_skips_claude(): ...          # 先 trip；claude 调用 0 次、返回 backup 的
def test_gated_llm_timeout_does_not_trip():                  # Review Focus 2
    # claude 抛 ClaudeError("超时") → 原样抛出、gate.ok() 仍 True、backup 0 次
def test_gated_llm_no_backup_raises_without_calling_claude():  # Review Focus 3
    # backup=None、闸已关 → 抛 ClaudeError、claude 调用 0 次
def test_gated_describe_skips_and_trips(): ...              # 闸关 → 抛 limit=True 的 ClaudeError、describe 0 次；describe 抛 limit → 关闸
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现两个包装**（改道时 `log.debug("Claude 不能用了：…改走 DeepSeek")`）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(brain): GatedLlm / gated_describe——闸关了文字改走 DeepSeek、看图直接跳过`

### Task 3: 大脑循环接闸（认证错也切、闸先关就先切、时间线模型名）

**Files:**
- Modify: `src/skydango/brain/loop.py`（`Brain.__init__`、`_wake`、`_failed`、`trace_state`）
- Test: `tests/test_brain_loop.py`（沿用里面的假 session / toolbox 辅助）

**Interfaces:**
- Consumes: `ClaudeGate`、`claude_down`
- Produces: `Brain(..., gate: ClaudeGate | None = None)`；`trace_state()` 多一个 `"claude_gate": gate.reason`（没闸 / 没关是 None），`"model"` 在 `on_fallback` 时取 `getattr(self.session, "model", self.cfg.model)`

- [ ] **Step 1: 写失败的测试**

```python
def test_auth_error_switches_to_fallback_and_trips_gate(): ...  # session 抛 ClaudeError(auth=True) → on_fallback、session 换成 fallback、gate 关
def test_gate_closed_elsewhere_switches_before_next_turn(): ...  # gate.trip("auth", …) 后下一次 wake：Claude session 一次都没调，fallback 被调
def test_auth_error_without_fallback_backs_off_limit_retry(): ...  # 没 fallback：backoff_until == now + cfg.limit_retry
def test_trace_state_model_and_gate_on_fallback(): ...  # 切过去后 model == fallback.model（假 fallback 设 .model="deepseek-chat"），claude_gate == gate.reason
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`_failed` 里 `kind = claude_down(exc)`，`kind` 非 None 时 `gate.trip(kind, str(exc))`，有 fallback 且没切过就切（原来只认 `limit` 的那段扩成两种，日志写"Claude 认证失败 / 额度用完，切到 DeepSeek 备用大脑"）；没 fallback 时 `auth` 也按 `limit_retry` 退避；`_wake` 开头：`gate` 关着、有 fallback、还没切 → 先切再发
- [ ] **Step 4: 跑测试确认通过**（连同原有 `tests/test_brain_loop.py` 全部）
- [ ] **Step 5: 提交** `feat(brain): 认证错也切备用大脑、闸被别处关了下一轮就切；时间线显示备用大脑的模型和关闸原因`

### Task 4: 整理 notes 失败后别每轮重试

**Files:**
- Modify: `src/skydango/chat/memory.py`（`NotesKeeper.maybe_update` / `update_now`，常量 `NOTES_RETRY = 600.0`）
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `NotesKeeper` 内部 `_retry_after: float`（`wall()` 时间）；行为：`update_now` 失败 → `_retry_after = wall() + NOTES_RETRY`，`maybe_update` 在那之前直接 return；成功 → 清成 `-inf`；`update_now` 本身（`memory update` 命令直接调的）不看 `_retry_after`

- [ ] **Step 1: 写失败的测试** `test_notes_update_failure_waits_before_retry`：假 llm 第一次抛异常；攒够 `notes_every` 轮触发一次失败；再加一轮 `maybe_update` → llm 调用次数不变；`wall` 前进 601 秒再 `maybe_update` → 调用次数 +1
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**（`tests/test_memory.py` 全部）
- [ ] **Step 5: 提交** `fix(memory): 整理 notes 失败后 10 分钟内不再试`

### Task 5: DeepSeek 时间线补全（中间文字、用量）

**Files:**
- Modify: `src/skydango/brain/deepseek.py`（`DeepSeekBrain.send`）
- Test: `tests/test_brain_deepseek.py`

**Interfaces:**
- Produces: `send()` 返回 `{"result": str, "subtype": "success", "num_turns": int, "usage": {"input_tokens": int, "output_tokens": int} | None}`；一次回复同时有 `content` 和 `tool_calls` 时，先 `on_message({"type": "assistant", "message": {"content": [{"type": "text", "text": content}]}})`，再发 tool_use

- [ ] **Step 1: 写失败的测试**

```python
def test_text_alongside_tool_calls_goes_to_timeline(): ...  # 回 content="我想想" + say 调用 → on_message 第一条是 text "我想想"，然后 tool_use / tool_result
def test_usage_is_summed_over_requests(): ...  # 两次请求的假 resp.usage = (prompt 100, completion 10)、(120, 5) → usage == {"input_tokens": 220, "output_tokens": 15}，num_turns == 2
def test_usage_none_when_client_gives_none(): ...  # resp 没有 usage 属性 → usage is None
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（`getattr(resp, "usage", None)` 取不到就不累加；有一次取到就给 dict）
- [ ] **Step 4: 跑测试确认通过**（`tests/test_brain_deepseek.py` 全部）
- [ ] **Step 5: 提交** `feat(brain): DeepSeek 备用大脑的中间文字和用量进时间线`

### Task 6: 接线（cli）、眼睛和装扮描述停用、文档

**Files:**
- Modify: `src/skydango/cli.py`（`_run_brain`：建闸；`memory_llm` / `_inner_mind` 的 llm 包成 `GatedLlm`；眼睛 / `_wardrobe` 的 describe 包 `gated_describe`；`Brain(..., gate=gate)`）
- Modify: `src/skydango/brain/eyes.py`（`Eyes(..., available: Callable[[], bool] | None = None)`：`tick` 里 `available()` 为假直接 `return False`）
- Modify: `src/skydango/vision/wardrobe.py`（`Wardrobe(..., available=None)`：挑下一个描述前 `available()` 为假就不排）
- Modify: `CLAUDE.md`（「统管大脑」里 DeepSeek 那条后面加一句闸）、`docs/progress/2026-10-03-plan.md`（「下一步」第 4 项把"还没做"改掉）
- Test: `tests/test_cli_brain.py`、`tests/test_brain_eyes.py`、`tests/test_wardrobe.py`

**Interfaces:**
- Consumes: Task 1~3 的 `ClaudeGate`、`GatedLlm`、`gated_describe`、`Brain(gate=)`
- Produces: `_inner_mind(cfg, ledger, base, claude_vars, run, now=None, clock=..., gate=None, backup=None)`；`_wardrobe(..., gate=None)`；`_gated_backup(cfg) -> LlmClient | None`（`make_llm(replace(cfg.llm, max_tokens=cfg.brain.fallback_max_tokens))`，`RuntimeError` / `ImportError` → None 并 `log.warning`）

- [ ] **Step 1: 写失败的测试**

```python
def test_run_brain_shares_one_gate(...): ...  # 用 fake_brain_run：on_ready 拿到的 parts.brain.gate 和 notes.llm.gate、reflector.llm.gate 是同一个对象
def test_gated_backup_uses_fallback_max_tokens(...):  # Review Focus 5：make_llm 被调时 cfg.max_tokens == cfg.brain.fallback_max_tokens
def test_force_fallback_keeps_gate_open(...):  # Review Focus 4：force_fallback=True 时 brain.on_fallback 且 brain.gate.ok()
def test_eyes_skip_when_unavailable(): ...  # available=lambda: False → tick 返回 False、describe 0 次
def test_wardrobe_stops_queueing_when_unavailable(): ...  # available=lambda: False → describe 0 次
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现接线**（`available=gate.ok`；沙盒世界也走同一套：`world.describe` 是沙盒自己的场景描述时不包）
- [ ] **Step 4: 跑这几个测试文件确认通过，再后台跑全量**：`python -m pytest -q -p no:cacheprovider`，期望全部通过
- [ ] **Step 5: 文档**：CLAUDE.md「统管大脑」DeepSeek 那条后加"Claude 总闸（10-04，spec `2026-10-04-claude-gate-design.md`）：额度 / 认证错后这次运行里不再起 claude：大脑切 DeepSeek（认证错也切）、随手记 / 整理 / 反思改走 DeepSeek、眼睛和装扮描述停用；整理 notes 失败 10 分钟内不再试"；进度文档第 4 项标已做、写上 spec §8 的沙盒验证四步待做
- [ ] **Step 6: 提交** `feat(brain): 接上 Claude 总闸（记忆 / 反思改走 DeepSeek、眼睛和装扮描述停用）`
