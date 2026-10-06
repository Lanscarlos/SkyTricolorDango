# OpenAI 兼容大脑的历史压缩（`[brain.compact]`）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ToolLoopBrain`（DeepSeek 等 OpenAI 兼容的大脑）一次上线内历史只往后接、前缀缓存一路命中；上一次请求的 `prompt_tokens` 到 `budget` 就在后台把老的部分压成「前情提要」，中途新记的 inbox 接在下一轮唤醒消息末尾，压缩失败退回滑动 8 轮，最终反思多拿一份前情提要。

**Architecture:** 新模块 `brain/compact.py` 放纯函数和小部件（压缩指令、前情提要的格式、截断、唤醒消息时间戳、`InboxWatch`）；`ToolLoopBrain` 自己管状态机（只往后接 / 滑动、压缩在跑、冷却），压缩在后台线程跑、结果放进加锁的槽，**下一次 `send` 开头**在大脑线程里换上（成功）或进滑动模式（失败），所有状态都只在大脑线程里改。`models/config.py` 加一个跟着大脑走的用处 `recap`（只用来记用量）。

**Tech Stack:** Python 3.13、openai SDK（假客户端测试）、pytest；管理面板的原生 JS（node 里测纯函数）。

**Spec:** `docs/superpowers/specs/2026-10-06-brain-compact-design.md`

## Global Constraints

- `[brain.compact]` 只对 OpenAI 兼容的大脑生效；常驻 Claude Code 的 `BrainSession` 一行不动
- `enabled = false`：请求逐字和现在一样（滑动 8 轮、唤醒消息不带新记的 inbox、没有 recap 用量、最终反思材料不变、`send` 返回的 dict 不多键）
- 默认值：`enabled = true`、`budget = 64000`、`keep_turns = 6`、`recap_max = 1500`、`retry = 300`
- 前情提要不写进任何记忆文件（inbox / notes / history / 日记），只进 `runs/<…>/brain/recap.jsonl`
- 压缩请求：同一个 client、模型、`tools`；`tool_choice = "none"`、`temperature = 0.3`、`max_tokens` = `[models.brain]` 的、超时 `[brain] turn_timeout`
- 前情提要消息格式：`（这次上线到现在的前情提要，第 N 次整理，写到 HH:MM 为止；之后的原话在后面）\n正文`，后面跟一条 assistant `（知道了）`
- inbox 新行接在唤醒消息末尾：`\n\n你刚记下：\n- 一行\n- 一行`
- 最终反思那一节标题：`## 这次上线早些时候（前情提要，大脑自己写的）`
- 用处 `recap`：label「压缩」，help「大脑历史太长时写成前情提要（跟着大脑的模型，不单独选）」，`follows = "brain"`
- 测试跑 `python -m pytest -q`（云端：`/tmp/claude-0/venv/bin/python -m pytest -q`）；改完合并进 main 并推送（CLAUDE.md）

## 偏离 spec / spec 没说死的决定

- **滑动模式下的重试不看 `prompt_tokens`**：滑动的请求本来就小（系统提示词 + 8 轮），按 §2.1 的「≥ budget」永远不会再试。改成：滑动模式里冷却过了、`_past` 比 `keep_turns` 多，就在这一轮结束时再试一次；这时压缩请求用「只往后接」时的完整历史现拼（不走缓存，一次性的开销）
- **压缩请求的 messages** = 触发那一轮结束时的完整 `messages`（最后一次请求的 messages + 它回的 assistant）+ 压缩指令；那条 assistant 带没执行的工具调用（轮数到顶）时去掉 `tool_calls` 只留文字（API 不收没有 tool 回复的调用）
- **唤醒消息没有时间戳**（farewell 的 SUMMARY_REQUEST、测试）时，指令里说「最后 K 轮之前」，前情提要的「写到 … 为止」写「最后 K 轮之前」
- 压缩失败也在**下一次 `send` 开头**处理（进滑动、记 trace / recap.jsonl / 日志），和成功走同一个槽
- 时间线上的压缩是这一轮里的一个步骤（新步骤 kind `compact`），不是单独的一轮：换上发生在 `send` 里，那时大脑这一轮已经开始了

## Review Focus

1. **压缩期间大脑又过了几轮**：换上时 `_past = _past[cut:]` 要留下压缩期间接上的轮（cut 是起压缩时算的）——Task 5 测「压缩期间接上的轮还在」
2. **最后一次请求的 assistant 带没执行的工具调用（轮数到顶）**：压缩请求里不能带悬空的 `tool_calls`——Task 5 测
3. **大脑这一轮中途抛 `ModelError`**：这一轮不进 `_past`、不判压缩，压缩线程不受影响——Task 5 测
4. **inbox 被整理挪走后又出现同一行**：不重复带；inbox 读出错这一轮不带、不抛——Task 4 测
5. **`[models.recap]` 写了 / 大脑切了备用**：recap 永远跟 brain（主备都跟），用量 `backup` 照会话传——Task 2、Task 6 测

---

### Task 1: 配置 `[brain.compact]` 和设置页

**Files:**
- Modify: `src/skydango/config.py`（`BrainConfig` 前加 `CompactConfig`，`BrainConfig.compact`）
- Modify: `src/skydango/console/settings.py`（`FIELDS` 加三行，group `brain`，放在 `addressee.enabled` 后面）
- Modify: `config.example.toml`（加 `[brain.compact]` 一段，带注释）
- Test: `tests/test_brain_compact.py`（新建）

**Interfaces:**
- Produces: `CompactConfig(enabled: bool = True, budget: int = 64000, keep_turns: int = 6, recap_max: int = 1500, retry: float = 300.0)`；`BrainConfig.compact: CompactConfig`

- [ ] **Step 1: 写失败的测试**

```python
def test_compact_config_defaults_and_toml(tmp_path):
    from skydango.config import CompactConfig, Config, load_config
    assert Config().brain.compact == CompactConfig(enabled=True, budget=64000, keep_turns=6, recap_max=1500, retry=300.0)
    p = tmp_path / "c.toml"
    p.write_text("[brain.compact]\nenabled = false\nbudget = 32000\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.compact.enabled is False and cfg.brain.compact.budget == 32000 and cfg.brain.compact.keep_turns == 6
    assert cfg.sources["brain.compact.budget"] == "config"

def test_compact_settings_fields():
    from skydango.console.settings import KNOWN
    assert {"brain.compact.enabled", "brain.compact.budget", "brain.compact.keep_turns"} <= set(KNOWN)
    assert KNOWN["brain.compact.enabled"].kind == "bool" and KNOWN["brain.compact.budget"].kind == "int"
```

- [ ] **Step 2: 跑测试确认失败**（`ImportError: CompactConfig`）
- [ ] **Step 3: 实现**：`CompactConfig` dataclass（docstring 指向 spec）、`BrainConfig.compact = field(default_factory=CompactConfig)`；设置页三行：「大脑历史压缩」/「压缩门槛（token）」/「压完留几轮原话」，help 写清只对 OpenAI 兼容的大脑、关掉回到滑动 8 轮。`config.example.toml` 照 spec §8 写
- [ ] **Step 4: 跑测试通过**，再跑 `tests/test_console_settings.py tests/test_config*.py`
- [ ] **Step 5: 提交** `feat(brain): [brain.compact] 配置和设置页`

---

### Task 2: 跟着大脑的用处 `recap`

**Files:**
- Modify: `src/skydango/models/config.py`（`Use.follows`、`USES` 加 recap、`resolve` 跟随）
- Modify: `src/skydango/models/registry.py`（`log_summary` 跳过跟随的）
- Modify: `src/skydango/cli.py`（`_check_brain_models`、`_models_line` 跳过跟随的）
- Modify: `src/skydango/models/usage.py`（`snapshot()["uses"]` 跳过跟随的；rows 照记）
- Modify: `src/skydango/console/models_view.py`（view 的 recap 行带 `follows`、`used_by` 不算它；`_save` 忽略跟随的用处、删掉 console.toml 里的 `[models.recap]`）
- Modify: `src/skydango/console/static/models.js`（`follows` 的行只显示「跟着大脑」，不进 `ST.uses`）
- Test: `tests/test_models_config.py` 或现有的 models 配置测试文件、`tests/test_console_models.py`

**Interfaces:**
- Produces: `Use.follows: str = ""`；`USE_BY_NAME["recap"]`（`vision=False`，main/backup 字段写 `""`，实际值由 resolve 抄 brain）；`ModelSetup.uses["recap"]` 的 main / backup / temperature / max_tokens / disabled 等于 brain 的、`source = "follows"`

- [ ] **Step 1: 写失败的测试**

```python
def test_recap_follows_brain():
    from skydango.config import Config
    from skydango.models.config import resolve
    cfg = Config(); cfg.models = {"brain": {"main": "claude/sonnet", "backup": "deepseek/deepseek-flash"},
                                  "recap": {"main": "deepseek/deepseek-v4-pro"}}
    cfg.sources["models.recap.main"] = "config"
    s = resolve(cfg)
    r, b = s.uses["recap"], s.uses["brain"]
    assert (str(r.main), str(r.backup)) == ("claude/sonnet", "deepseek/deepseek-flash") and r.max_tokens == b.max_tokens
    assert any("[models.recap]" in p.text and p.warn for p in s.problems)

def test_recap_hidden_in_summaries(caplog):  # log_summary 不报 recap
    ...registry.log_summary() 后 "模型：recap" 不在 caplog.text

def test_models_view_recap_row_follows(store):  # 用 test_console_models.py 现成的 store fixture
    row = next(u for u in ModelsView(store).view()["uses"] if u["name"] == "recap")
    assert row["follows"] == "brain"
    # 保存时带上 recap（哪怕值和 brain 不同）也不写 [models.recap]
    status, _ = ModelsView(store).save({..., "uses": {"recap": {"main": "claude/haiku", "backup": ""}}})
    assert status == 200 and "recap" not in read(store.console_path).get("models", {})
```

另在 `tests/test_models_usage.py` 加：`UsageMeter.snapshot()["uses"]` 里没有 `recap`；`record("recap", …)` 后 `rows_view` 里那一行 label 是「压缩」。

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`resolve` 里跟随的用处不读 `[models.<它>]`（写了就 `Problem("[models.recap] 不生效：压缩跟着大脑的模型", use="recap", warn=True)`），直接 `dataclasses.replace(uses["brain"], name="recap", source="follows")`；需要 brain 先算（USES 里 brain 在前，recap 放最后）。`log_summary` / `_check_brain_models` 的循环 / `_models_line` / `snapshot` 的 uses / `models_view` 的 `used_by` 都按 `USE_BY_NAME[name].follows` 跳过。models.js：`ST.uses` 只收没 `follows` 的；`useRow` 遇到 `follows` 只画标签、help 和一行「跟着大脑」。`_save` 遇到跟随的用处 `continue`，最后 `models.pop("recap", None)`
- [ ] **Step 4: 跑测试通过**，再跑 `tests/test_console_models.py tests/test_models_*.py tests/test_console_usage*.py tests/test_preflight*.py`
- [ ] **Step 5: 提交** `feat(models): 跟着大脑的用处 recap（记压缩的用量）`

---

### Task 3: `brain/compact.py` 纯函数和 `InboxWatch`

**Files:**
- Create: `src/skydango/brain/compact.py`
- Test: `tests/test_brain_compact.py`

**Interfaces:**
- Produces:
  - `COMPACT_REQUEST: str`（模板，占位 `{where}`、`{max}`；内容照 spec §2.2 三条要点）
  - `RECAP_ACK = "（知道了）"`
  - `wake_stamp(text: str) -> str`：唤醒消息开头 `[…]` 里的字；没有返回 `""`
  - `until_of(stamp: str) -> str`：stamp 里最后一个 `HH:MM`；没有返回 `""`
  - `compact_request(first_kept: str, keep: int, recap_max: int) -> str`：`where` = `[stamp] 那条消息之前`，没 stamp 时 `最后 {keep} 轮之前`
  - `recap_message(text: str, n: int, until: str, keep: int) -> str`：spec §2.3 的格式；`until` 空时「写到最后 {keep} 轮之前为止」
  - `clean_recap(text: str | None, recap_max: int) -> str`：去首尾空白；超过 `recap_max * 1.5` 字截到 `recap_max` 字加「……」；`None` → `""`
  - `inbox_note(lines: list[str]) -> str`：`"你刚记下：\n- a\n- b"`（行首已有的 `- ` 去掉再加）；空列表 `""`
  - `class InboxWatch(read: Callable[[], str])`：建的时候读一次、非空行（strip）记进 `seen`；`fresh() -> list[str]` 读一次，返回不在 `seen` 的新行（保持顺序、去重）并记进 `seen`；读出错 `log.warning` 返回 `[]`；建的时候读出错当空

- [ ] **Step 1: 写失败的测试**

```python
def test_wake_stamp_and_until():
    t = "[2026年10月6日（周二） 21:03:15] 事件：\n- 小明：在吗"
    assert wake_stamp(t) == "2026年10月6日（周二） 21:03:15" and until_of(wake_stamp(t)) == "21:03"
    assert wake_stamp("写一份经过") == "" and until_of("") == ""

def test_compact_request_mentions_cut_and_limit():
    q = compact_request("[2026年10月6日（周二） 21:03:15] 事件", keep=6, recap_max=1500)
    assert "[2026年10月6日（周二） 21:03:15] 那条消息之前" in q and "1500" in q and "前情提要" in q
    assert "最后 6 轮之前" in compact_request("没有时间戳", keep=6, recap_max=1500)

def test_recap_message_format():
    assert recap_message("正文", 2, "21:03", 6) == "（这次上线到现在的前情提要，第 2 次整理，写到 21:03 为止；之后的原话在后面）\n正文"

def test_clean_recap():
    assert clean_recap("  好  ", 10) == "好" and clean_recap(None, 10) == ""
    assert clean_recap("一" * 15, 10) == "一" * 15          # 没超 1.5 倍：整句收下
    assert clean_recap("一" * 16, 10) == "一" * 10 + "……"

def test_inbox_watch():
    text = ["- 旧的一行\n"]
    w = InboxWatch(lambda: text[0])
    assert w.fresh() == []
    text[0] = "- 旧的一行\n- 小明下周考试\n"
    assert w.fresh() == ["- 小明下周考试"] and w.fresh() == []
    text[0] = "- 小明下周考试\n"; w.fresh(); text[0] = "- 旧的一行\n- 小明下周考试\n"
    assert w.fresh() == []                                   # 挪走再加回：不重复
    assert inbox_note(["- 小明下周考试", "阿花生日"]) == "你刚记下：\n- 小明下周考试\n- 阿花生日"

def test_inbox_watch_read_error(caplog):
    def boom(): raise OSError("坏了")
    assert InboxWatch(boom).fresh() == []
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: 提交** `feat(brain): 压缩的指令、前情提要格式和 inbox 新行（brain/compact.py）`

---

### Task 4: `ToolLoopBrain` 只往后接 + 新记的 inbox

**Files:**
- Modify: `src/skydango/brain/toolloop.py`
- Create: `tests/data/toolloop_baseline.json`（改代码**之前**用现在的代码录）
- Test: `tests/test_brain_toolloop_compact.py`（新建；假客户端照 `tests/test_brain_toolloop.py` 的写法，但每次 `create` 记下 `copy.deepcopy(kw)`）

**Interfaces:**
- Consumes: `CompactConfig`（Task 1）、`InboxWatch` / `inbox_note` / `recap_message` / `RECAP_ACK`（Task 3）
- Produces: `ToolLoopBrain.__init__` 新的仅关键字参数 `compact: CompactConfig | None = None`、`inbox: Callable[[], str] | None = None`（Task 5 再加 meter / backup / recap_path / on_note / spawn）；属性 `recap: str`（没压过 `""`）、`compactions: int`、`sliding: bool`；内部 `_append_messages() -> list[dict]`（system + 前情提要两条 + `_past` 全部轮，每轮 `_turn_messages(user, turn)`）

- [ ] **Step 1: 录基准**：在 scratchpad 写个脚本，用**现在的** `ToolLoopBrain`（`history=8`）、脚本化假客户端跑 12 轮（有的轮调 say 工具、有的只回文字、有一轮超过 `HISTORY_TEXT` 的唤醒消息），把每次 `create` 的 `messages` 存进 `tests/data/toolloop_baseline.json`（连同用到的脚本参数，测试照同样的脚本重放）
- [ ] **Step 2: 写失败的测试**

```python
def test_disabled_matches_baseline():          # §9.9：compact=None 逐字照旧
    assert record_requests(compact=None) == json.loads((DATA / "toolloop_baseline.json").read_text("utf-8"))["requests"]

def test_append_only_prefix():                 # §9.1
    reqs = record_requests(compact=CompactConfig(budget=10**9), turns=12)
    for a, b in zip(reqs, reqs[1:]):
        if b["turn"] == a["turn"]:             # 同一轮：上一次请求整个是前缀
            assert b["messages"][:len(a["messages"])] == a["messages"]
        else:                                  # 跨轮：上一轮的历史部分（去掉那一轮的唤醒消息起）是前缀
            assert b["messages"][:a["start"] - 1] == a["messages"][:a["start"] - 1]
    assert len(reqs[-1]["messages"]) > 2 * 8 + 2    # 不再按 history=8 截

def test_inbox_new_lines_once():               # §9.7
    inbox = ["- 旧的\n"]
    brain = make(compact=CompactConfig(), inbox=lambda: inbox[0])
    brain.send("[… 21:00:00] 一"); assert "你刚记下" not in last_user(brain)
    inbox[0] += "- 小明下周考试\n"
    brain.send("[… 21:01:00] 二"); assert last_user(brain).endswith("\n\n你刚记下：\n- 小明下周考试")
    brain.send("[… 21:02:00] 三"); assert "你刚记下" not in last_user(brain)

def test_inbox_ignored_when_disabled():
    brain = make(compact=None, inbox=lambda: "- 新的\n")   # 关着：不读 inbox
```

（`record_requests` / `make` / `last_user` 是测试文件里的帮手：`last_user` = 最近一次请求最后一条 user 消息的 content；`start` = 那次请求里这一轮唤醒消息的下标 + 1。）

- [ ] **Step 3: 跑测试确认失败**（基准那条应该**通过**——它是护栏，别的失败）
- [ ] **Step 4: 实现**：`compact is None` 走原路径一字不动；开着时 `send` 开头读 `InboxWatch.fresh()`，有新行就 `text += "\n\n" + inbox_note(lines)`（这个 `text` 同时进请求和 `_past`）；请求前缀 = `_append_messages()`（滑动时 = system + 前情提要两条 + `_history_messages()`）；轮末 `_past.append(...)` 不截。把 `_history_messages` 里截唤醒消息的那段抽成 `_turn_messages` 两边共用（不改它的结果）
- [ ] **Step 5: 跑测试通过**，再跑 `tests/test_brain_toolloop.py`
- [ ] **Step 6: 提交** `feat(brain): OpenAI 兼容大脑的历史只往后接、新记的 inbox 接在唤醒消息末尾`

---

### Task 5: 压缩：触发、请求、换上、失败退回滑动

**Files:**
- Modify: `src/skydango/brain/toolloop.py`
- Test: `tests/test_brain_toolloop_compact.py`

**Interfaces:**
- Consumes: Task 3 的 `compact_request` / `clean_recap` / `recap_message` / `wake_stamp` / `until_of`；Task 4 的 `_append_messages`
- Produces: `ToolLoopBrain.__init__` 再加 `meter=None`、`backup: bool = False`、`recap_path: Path | None = None`、`on_note: Callable[[str], None] | None = None`、`wall: Callable[[], float] = time.time`、`spawn: Callable[[Callable[[], None]], None] | None = None`（默认起 daemon 线程，测试注入「先存着、手动跑」）；`send` 返回 dict 在开着时多 `history_mode`（`"append"` / `"sliding"`）和 `recap`（`compactions`）

状态机（只在大脑线程里改）：

```
send 开头 _take()：槽里有结果 →
    成功：recap = 文字；_past = _past[cut:]；compactions += 1；sliding = False；
          on_note("── 压缩：第 N 次，压掉 K 轮 → 前情提要 C 字，用时 S 秒 ──\n" + 文字)；recap.jsonl 一行；log.info
    失败：sliding = True；retry_at = clock() + retry；on_note("── 压缩失败：原因，先用最近几轮 ──")；recap.jsonl 一行；log.warning
    _pending = False
send 结束 _after(messages, prompt_tokens)：
    _pending 且 prompt_tokens ≥ 2×budget → sliding = True（log.warning 一次）
    不起：_pending、clock() < retry_at、len(_past) ≤ keep_turns
    只往后接：prompt_tokens 为 None 或 < budget 不起；起 = 用这一轮完整 messages
    滑动：（冷却过了就）起 = 用 _append_messages() 现拼
```

压缩任务（在 `spawn` 里跑）：`create(model, messages=源 + [{"role": "user", "content": compact_request(...)}], tools=self.tools, tool_choice="none", temperature=0.3, max_tokens=self.max_tokens, timeout=self.turn_timeout)`；空结果 / 带 `tool_calls` / 异常（包成 `ModelError` 的文字）= 失败；`meter.record("recap", provider, model, backup=self.backup, usage=…, ok=…)`（usage 同 `send` 的三个键；meter 出错只记 DEBUG）；结果连同 cut、轮数、用时、usage 放进加锁的槽。

- [ ] **Step 1: 写失败的测试**（假客户端按 `messages` 最后一条是不是压缩指令分流：是就回压缩脚本里的下一项）

```python
def test_no_compact_below_budget():                       # §9.2
    b, jobs = make_c(budget=1000, usage=999); run_turns(b, 10); assert jobs == []

def test_compact_request_shape():                         # §9.2
    b, jobs = make_c(budget=1000, usage=1000, keep=2); run_turns(b, 3)
    assert len(jobs) == 1; run_turns(b, 1, usage=5000); assert len(jobs) == 1   # 在跑：不起第二个
    jobs[0]()
    req = compact_reqs(b)[0]
    assert req["messages"][:-2] == last_turn_req(b, turn=3)["messages"]
    assert req["messages"][-2]["role"] == "assistant" and req["messages"][-1]["content"].startswith(...)
    assert req["tool_choice"] == "none" and req["tools"] == b.tools and req["temperature"] == 0.3

def test_swap_in_recap():                                 # §9.3 + Review Focus 1
    # keep=2，第 3 轮后起压缩、第 4 轮（压缩期间）接上、跑 job、第 5 轮开头换上
    m = req_of_turn(b, 5)["messages"]
    assert m[1]["content"].startswith("（这次上线到现在的前情提要，第 1 次整理，写到 21:02 为止") and m[2] == {"role": "assistant", "content": "（知道了）"}
    assert [x["content"] for x in m if x["role"] == "user"][1:] == [轮2, 轮3, 轮4 的唤醒消息（截后）, 轮5]
    # 之后又只往后接：第 6 轮的请求以第 5 轮的历史部分为前缀

def test_second_compact_includes_old_recap():             # §9.4
    assert "前情提要" in second_compact_req["messages"][1]["content"] and b.recap == "第二份" and b.compactions == 2

@pytest.mark.parametrize("bad", [ModelError("撞墙"), "", "TOOL"])
def test_compact_failure_slides_then_retries(bad):        # §9.5
    # 失败后下一轮：请求 = system + _history_messages()（和 compact=None 时一样的历史）；history_mode == "sliding"
    # retry 秒内（假 clock）不再起；过了下一轮结束起、成功后回到 "append"

def test_stuck_compact_slides_at_double_budget():         # §9.6
    b, jobs = make_c(budget=1000, usage=1000); run_turns(b, 3); run_turns(b, 1, usage=2000)
    assert b.sliding and send_result(b)["history_mode"] == "sliding"; assert len(jobs) == 1

def test_compact_meter_backup():                          # §9.8
    meter = FakeMeter(); b, jobs = make_c(meter=meter, backup=True, ...); ...; jobs[0]()
    assert meter.calls == [("recap", "deepseek", "deepseek-flash", True, {"input_tokens": 3000, "output_tokens": 200, "cache_read_input_tokens": 2800}, True)]

def test_compact_skips_dangling_tool_calls():             # Review Focus 2：轮数到顶那条 assistant 只留文字
def test_turn_error_not_recorded():                       # Review Focus 3：send 抛 ModelError → _past 不变、不起压缩
def test_recap_jsonl_and_note(tmp_path):                  # recap.jsonl 成功 / 失败各一行，字段 time / n / turns / chars / seconds / usage / text 或 error；on_note 收到一条
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（照上面的状态机；`ModelError` 不关闸，日志写清原因）
- [ ] **Step 4: 跑测试通过**，再跑 `tests/test_brain_toolloop*.py tests/test_brain_loop*.py`
- [ ] **Step 5: 提交** `feat(brain): 历史超预算在后台压成前情提要，失败退回滑动 8 轮`

---

### Task 6: 接线：make_session、主 / 备、时间线、brain.jsonl

**Files:**
- Modify: `src/skydango/brain/sessions.py`（`make_session` 加 `inbox=None`、`trace=None`、`backup: bool = False`）
- Modify: `src/skydango/cli.py`（`_brain_sessions` 传 `backup`；`functools.partial(make_session, …)` 加 `inbox=store.inbox if store else None`、`trace=trace`）
- Modify: `src/skydango/brain/loop.py`（`_log` 把 result 里有的 `history_mode` / `recap` 写进 brain.jsonl）
- Modify: `src/skydango/brain/trace.py`（`BrainTrace.note(text: str)`：往当前轮（没有就「轮外」）加一步 `{"kind": "compact", "text": _clip(text)}`）
- Modify: `src/skydango/console/static/brainlog.js`（`stepLine` 认 `compact`：`{cls: "dim", mark: "─", text, fold: true}`；`line()` 多一个 `fold` 参数，`fold` 且多于一行时总是折起来，摘要是第一行）
- Test: `tests/test_brain_toolloop.py`（make_session）、`tests/test_brain_trace.py` / 现有 trace 测试文件、`tests/test_console_page.py`（brainlog 纯函数）、`tests/test_brain_loop*.py`

**Interfaces:**
- Consumes: Task 5 的 `ToolLoopBrain` 参数
- Produces: `make_session(..., inbox=None, trace=None, backup=False)`；`_brain_sessions` 调 `build(ref, backup=bool)`

- [ ] **Step 1: 写失败的测试**

```python
def test_make_session_compact_wiring(tmp_path):
    s = make_session(reg, ModelRef("deepseek", "deepseek-flash"), ..., workdir=tmp_path, cfg=BrainConfig(),
                     inbox=lambda: "", trace=BrainTrace(), backup=True)
    assert s.compact == BrainConfig().compact and s.backup is True and s.meter is reg.meter
    assert s.recap_path == tmp_path / "recap.jsonl" and s.on_note is not None
    off = BrainConfig(); off.compact.enabled = False
    assert make_session(..., cfg=off).compact is None

def test_brain_sessions_backup_flag():
    calls = []; build = lambda ref, backup=False: calls.append((str(ref), backup)) or object()
    session, later = cli._brain_sessions(registry, build); later()
    assert calls == [("deepseek/deepseek-flash", False), ("claude/sonnet", True)]

def test_trace_note_step():  # 当前轮里多一步 compact；没有当前轮进「轮外」
def test_brainlog_compact_step():  # node：stepLine({kind:"compact", text:"── 压缩 ──\n正文"}) → mark "─"、fold true
def test_brain_jsonl_history_mode(tmp_path):  # result 带 history_mode / recap → brain.jsonl 那行有；不带就没有这两个键
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**；`make_session` 里 `compact=cfg.compact if cfg.compact.enabled else None`、`meter=registry.meter`、`recap_path=Path(workdir) / "recap.jsonl"`、`on_note=trace.note if trace else None`、`wall` 不传（真时间）；`_brain_sessions` 主建不起来改用备时 `backup=True`
- [ ] **Step 4: 跑测试通过**，再跑 `tests/test_cli*.py tests/test_sandbox*.py tests/test_brain_*.py tests/test_console_page.py`
- [ ] **Step 5: 提交** `feat(brain): 压缩接上 make_session、用量主备、时间线和 brain.jsonl`

---

### Task 7: 最终反思带上前情提要

**Files:**
- Modify: `src/skydango/inner/reflect.py`（`materials(..., recap: str = "")`）
- Modify: `src/skydango/brain/body.py`（`Body.recap_text: Callable[[], str] = lambda: ""`；`reflect_materials(final=True)` 传 `recap=self.recap_text()`，出错当空）
- Modify: `src/skydango/cli.py`（`body.recap_text = lambda: getattr(brain.session, "recap", "") or ""`）
- Test: `tests/test_inner_reflect.py`、身体的反思材料测试

**Interfaces:**
- Produces: `materials(..., recap: str = "")`：`recap` 非空时在「这段时间的聊天」那块**前面**插 `"## 这次上线早些时候（前情提要，大脑自己写的）\n" + recap`

- [ ] **Step 1: 写失败的测试**

```python
def test_materials_recap_block():               # §9.10
    base = materials(NOW, "", "", CHAT, [], [], [], "人设", True)
    assert materials(NOW, "", "", CHAT, [], [], [], "人设", True, recap="") == base   # 逐字照旧
    with_recap = materials(NOW, "", "", CHAT, [], [], [], "人设", True, recap="小明说下周三考物理")
    head = "## 这次上线早些时候（前情提要，大脑自己写的）\n小明说下周三考物理"
    assert head in with_recap and with_recap.index(head) < with_recap.index("这段时间的聊天")

def test_body_final_materials_recap(...):       # 身体：final 带、中途不带、recap_text 抛错当空
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试通过**，再跑 `tests/test_inner_*.py tests/test_brain_body*.py`
- [ ] **Step 5: 提交** `feat(inner): 下线反思带上大脑这次上线的前情提要`

---

### Task 8: 文档和收尾

**Files:**
- Modify: `CLAUDE.md`（「代码结构」表 `brain/` 那行加 `compact.py`；「统管大脑」一节加一段「历史压缩」：只往后接、预算、前情提要放哪、失败滑动、inbox 新行、recap 用处、recap.jsonl、**还没在沙盒 / 真机验证，spec §10 六步没走**；「运行目录」表 `brain.jsonl` 那行补 `history_mode` / `recap` 和 `brain/recap.jsonl`）

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 全量测试** `python -m pytest -q`，全部通过
- [ ] **Step 3: 提交** `docs: CLAUDE.md 记上大脑历史压缩`
- [ ] **Step 4: 收尾**：whole-branch review（requesting-code-review），处理意见；合并进 main 并推送（CLAUDE.md 规矩）
