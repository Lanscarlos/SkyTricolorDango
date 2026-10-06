# 模型用量和额度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 管理面板的真机团子页、沙盒页、模型页显示每个用处的模型用量、折算花费、DeepSeek 余额、Claude 订阅额度和闸的状态；默认模型换成 deepseek-flash。

**Architecture:** `models/usage.py` 的 `UsageMeter` 挂在 `Registry` 上，`GatedCall` / 大脑 / Claude Code 进程把每笔用量和限额事件交给它；`models/usage_ledger.py` 把增量合并进 `runs/usage.json`（今天累计）；`models/balance.py` 后台查 DeepSeek 余额。团子 / 沙盒接口各加 `GET /usage`，面板 `GET /api/usage`（停着读账本），共用页面组件 `console/static/usage.js`。

**Tech Stack:** Python 3.13 标准库（threading、urllib、json、datetime），openai SDK 返回的 usage，原生 JS（node 测渲染）。

**Spec:** `docs/superpowers/specs/2026-10-06-model-usage-design.md`

## Global Constraints

- 记账、算钱、余额、限额、写账本任何一步出错只记日志，绝不影响模型调用本身。
- 单价单位：元 / 百万 token，空闲价三档 `[命中缓存的输入, 没命中的输入, 输出]`；高峰价 = 空闲价 × `peak`。
- 内置 DeepSeek 单价：`deepseek-flash` = [0.02, 1, 4]，`deepseek-v4-pro` = [0.15, 4.5, 13.5]；`deepseek-chat` / `deepseek-reasoner` 按 Flash 价、标估；`peak = 2.0`，`peak_hours = ["09:00-12:00", "14:00-18:00"]`。
- 高峰按北京时间（固定 UTC+8，不用 zoneinfo：Windows 没装 tzdata）周一到周五判；不认节假日。
- `claude-code` 接入方式不算钱（cost 为 `None`）。
- 余额只查 `kind = "openai"` 且 `base_url` 主机是 `api.deepseek.com` 的供应商：`GET <base_url>/user/balance`。
- `[usage]`：`ledger = "runs/usage.json"`、`keep_days = 7`、`save_every = 60.0`、`balance_every = 300.0`。
- 页面颜色只用 `console.css` `:root` 的变量；不用原生 confirm / prompt / alert；请求一律相对路径。
- 来源：`live`（`run`，含 `--no-brain`、dry-run）、`sandbox`、`offline`（`_registry()` 默认）。
- 默认模型：brain / memory / reflect / reply / text_label 的 main 改成 `deepseek/deepseek-flash`。
- 测试命令：`python -m pytest -q`（云端是系统 Python；本机用 `.venv\Scripts\python.exe -m pytest -q`）。

## Review Focus

- console.toml 里整张写着的 deepseek 供应商（面板保存过，没有 `prices`）：仍然按内置 DeepSeek 单价算钱，而不是全部「没填单价」。→ Task 2 `test_explicit_deepseek_without_prices_gets_builtin`
- 账本被另一个进程（离线命令）同时写：两个 meter 先后 flush，数字相加、不丢。→ Task 4 `test_two_meters_add_up`
- 后端返回的 usage 是空 dict / None / 缺字段（echo、旧 Claude 结果）：记一次调用、token 记 0，不抛。→ Task 3 `test_record_tolerates_missing_usage`
- 余额接口返回 200 但内容不是预期的 JSON（代理页面、字段变了）：记成查不到 + 原因，不抛。→ Task 6 `test_balance_bad_json`
- 页面拉 `/usage` 时子进程刚好退出（503）：卡片退回 `/api/usage` 的今天，不清空、不报错弹窗。→ Task 8 `usage.js` 的 `test_usage_card_falls_back`

---

### Task 1: 默认模型换成 deepseek-flash

**Files:**
- Modify: `src/skydango/models/config.py`（`USES` 五行的 main、`_builtin` 的模型表）
- Modify: `src/skydango/config.py:153`（`LlmConfig.model` 默认）
- Modify: `src/skydango/console/models_view.py:18-20`（DeepSeek 模板）
- Modify: `config.example.toml`（`[models.*]` 注释里的默认、`[providers.deepseek]` 模型表）
- Test: `tests/test_models_config.py`、`tests/test_console_models.py`（凡是断言默认 `deepseek-chat` 的改成 flash）

**Interfaces:**
- Produces: 内置 deepseek `models = ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner")`，`vision = ("deepseek-flash",)`

- [ ] **Step 1: 改测试**：`test_defaults` 断言 `str(s.uses[u].main) == "deepseek/deepseek-flash"`（u ∈ brain / memory / reflect / reply / text_label），备用不变；新增 `test_builtin_deepseek_models` 断言模型表顺序如上；`test_local_layout_brain_on_deepseek` 改成 brain = `deepseek/deepseek-flash`、reply 仍 = `deepseek/deepseek-chat`（旧 `[llm] model` 照旧换算）。
- [ ] **Step 2:** `python -m pytest -q tests/test_models_config.py` → 上面几条 FAIL
- [ ] **Step 3: 实现**：改 `USES`、`_builtin`、`LlmConfig.model = "deepseek-flash"`、模板、example。`grep -rn "deepseek-chat" src tests` 逐个看，只改"默认值"含义的地方。
- [ ] **Step 4:** `python -m pytest -q` → 全过（除了基线里本来就挂的 `test_gesture_train.py::test_cli_gesture_train_end_to_end`，缺 torch）
- [ ] **Step 5: Commit** `feat(models): 大脑 / 记忆 / 反思 / 回复 / 文字标注默认改用 deepseek-flash`

### Task 2: 单价配置

**Files:**
- Modify: `src/skydango/models/config.py`（`Price`、`ProviderConfig.prices / peak / peak_hours`、内置和显式供应商解析）
- Modify: `src/skydango/console/models_view.py`（view / `provider_from` / `provider_table` 带上单价和高峰倍数）
- Modify: `src/skydango/console/static/models.js`（每个模型一行三个数字框、供应商一格高峰倍数）
- Test: `tests/test_models_config.py`、`tests/test_console_models.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Price: hit: float; miss: float; out: float; est: bool = False`
  - `ProviderConfig.prices: tuple[tuple[str, Price], ...] = ()`、`peak: float = 1.0`、`peak_hours: tuple[str, ...] = ()`、`def price(self, model: str) -> Price | None`
  - `DEEPSEEK_PRICES: dict[str, Price]`（含估价的 deepseek-chat / deepseek-reasoner，`est=True`）、`DEEPSEEK_PEAK = 2.0`、`DEEPSEEK_PEAK_HOURS = ("09:00-12:00", "14:00-18:00")`
  - `def is_deepseek(p: ProviderConfig) -> bool`（kind openai 且 `urlparse(base_url).hostname == "api.deepseek.com"`）
  - models_view 的供应商行多 `"prices": {模型: [hit, miss, out]}`、`"peak": float`

- [ ] **Step 1: 写测试**
  - `test_builtin_deepseek_prices`：`p.price("deepseek-flash") == Price(0.02, 1, 4)`，`p.price("deepseek-chat").est is True`，`p.peak == 2.0`，`p.peak_hours == DEEPSEEK_PEAK_HOURS`，`resolve(...).providers["claude"].price("sonnet") is None`
  - `test_explicit_prices_parsed`：`[providers.x] kind="openai" models=["m"] prices={m=[1,2,3]} peak=1.5 peak_hours=["10:00-11:00"]` → `price("m") == Price(1,2,3)`、peak / hours 对
  - `test_bad_prices_reported`：`prices={m=[1,2]}`、`prices={m=["a",1,2]}`、`peak_hours=["25:00-26:00"]` 各报一条 `Problem`（provider = x），供应商照样建出来（坏的那项丢掉）
  - `test_explicit_deepseek_without_prices_gets_builtin`：`[providers.deepseek] kind="openai" base_url="https://api.deepseek.com" models=["deepseek-flash"]`（没 prices）→ `price("deepseek-flash") == DEEPSEEK_PRICES["deepseek-flash"]`、`peak == 2.0`
  - `test_console_models_roundtrip_prices`（test_console_models）：view 里 deepseek 行有 `prices["deepseek-flash"] == [0.02, 1, 4]`；`save` 改成 `[0.03, 1, 4]` 后 console.toml 里 `providers.deepseek.prices["deepseek-flash"] == [0.03, 1, 4]`、`peak` 写进去；提交负数 → 400
- [ ] **Step 2:** 跑这几条 → FAIL
- [ ] **Step 3: 实现**：`_explicit` 里 `prices` 表 → 元组（每项必须 3 个非负数）、`peak` ≥ 1、`peak_hours` 每项 `HH:MM-HH:MM`；`prices` / `peak` / `peak_hours` 都没写且 `is_deepseek` → 用内置值。和内置估价完全相同的数（同模型名）标 `est=True`（`_mark_est`），这样面板保存后再读出来照样是估。models.js 数字框 `step="any" min="0"`，空着 = 不填单价；`provider_table` 只写有单价的模型。
- [ ] **Step 4:** `python -m pytest -q tests/test_models_config.py tests/test_console_models.py tests/test_console_page.py` → PASS
- [ ] **Step 5: Commit** `feat(models): 供应商按模型配单价和高峰倍数，内置 DeepSeek 价格页单价`

### Task 3: UsageMeter（记账和算钱，纯内存）

**Files:**
- Create: `src/skydango/models/usage.py`
- Test: `tests/test_models_usage.py`

**Interfaces:**
- Consumes: Task 2 的 `ProviderConfig.price / peak / peak_hours`、`ModelSetup`、`ProviderGates`、`USE_BY_NAME`
- Produces:
  - `def is_peak(at: float, peak_hours: tuple[str, ...]) -> bool`（UTC+8，周一到周五，区间左闭右开）
  - `def call_cost(price: Price | None, usage: dict, peak: float) -> float | None`
  - `class UsageMeter(setup: ModelSetup, gates: ProviderGates, *, source: str, wall: Callable[[], float] = time.time, ledger: "Ledger | None" = None)`
    - `record(use: str, provider: str, model: str, *, backup: bool, usage: dict | None, ok: bool, at: float | None = None) -> None`
    - `rate_limit(provider: str, info: dict, at: float | None = None) -> None`
    - `balance(provider: str, info: dict | None, error: str | None, at: float | None = None) -> None`
    - `take_delta() -> dict` / `restore_delta(delta: dict) -> None`（Task 4 用）
    - `snapshot(current: Callable[[str], ModelRef | None] | None = None) -> dict`（spec §6.1 的结构）
  - 行的字段：`use, label, model（"供应商/模型"）, backup, calls, fails, input, output, cache_read, cache_write, cost, est`
  - 增量格式（账本同款）：`{日期: {来源: {"use|供应商/模型|main或backup": {calls, fails, input, output, cache_read, cache_write, cost, est}}}}`，cost 为 `None` 表示没单价

- [ ] **Step 1: 写测试**
  - `test_is_peak`：北京时间 2026-10-06（周二）10:00 → True；12:00 → False；14:00 → True；2026-10-04（周日）10:00 → False；`peak_hours=()` → 永远 False
  - `test_call_cost_split_cache`：Flash 价、usage `{input_tokens: 1_000_000, cache_read_input_tokens: 400_000, output_tokens: 100_000}`、peak 1 → `0.4*0.02 + 0.6*1 + 0.1*4 == pytest.approx(1.008)`；peak 2 → 2.016；`price=None` → None；cache 大于 input 时没命中按 0
  - `test_record_rows_and_snapshot`：两笔 brain 主、一笔 brain 备、一笔 eyes 失败 → `run.rows` 三行，brain 主 calls=2，eyes calls=1 fails=1 input=0；行按 cost 降序、再按 calls；`run.cost` 是有单价的行之和，`est` 有任一行估就 True
  - `test_record_tolerates_missing_usage`：`usage=None`、`{}`、`{"input_tokens": None}` 都记一次、token 0、不抛
  - `test_claude_has_no_cost`：claude/sonnet 一笔 → 行 cost is None，`run.cost` 不受影响
  - `test_snapshot_providers`：gates 关掉 claude（原因「额度 / 余额用完」）→ providers 里 claude `closed == "额度 / 余额用完"`；`rate_limit("claude", {"status": "allowed_warning", "rateLimitType": "five_hour", "utilization": 0.8, "resetsAt": 123, "unifiedWindows": {"seven_day": {"utilization": 0.3, "resetsAt": 456}}, "junk": 1})` → `rate == {"status": "allowed_warning", "type": "five_hour", "utilization": 0.8, "resets_at": 123, "windows": {"seven_day": {"utilization": 0.3, "resets_at": 456}}, "at": …}`；乱填（`utilization: "x"`）的字段丢掉
  - `test_snapshot_uses`：`uses` 每个用处一行，`current` 用传进来的 `current` 函数，`backup` = current 是不是 backup，`disabled` = `UseConfig.disabled or None`
  - `test_delta_take_restore`：take 后再 take 是空的；restore 后合回去
- [ ] **Step 2:** `python -m pytest -q tests/test_models_usage.py` → FAIL（模块不存在）
- [ ] **Step 3: 实现** `models/usage.py`。一把 `threading.Lock` 管所有状态；日期按 `time.localtime(at)` 的 `%Y-%m-%d`；`snapshot` 的 `today` 先放 Task 4 的占位：没有 ledger 时 = 这个进程今天的全部。
- [ ] **Step 4:** 测试 → PASS
- [ ] **Step 5: Commit** `feat(models): UsageMeter 按用处 / 模型记用量、折算花费、限额和余额`

### Task 4: 今天累计的账本

**Files:**
- Create: `src/skydango/models/usage_ledger.py`
- Modify: `src/skydango/models/usage.py`（`flush()`、`snapshot()["today"]`、保存线程、atexit）
- Modify: `src/skydango/config.py`（`UsageConfig`、`Config.usage`）
- Test: `tests/test_models_usage_ledger.py`

**Interfaces:**
- Consumes: Task 3 的增量格式、`take_delta / restore_delta`
- Produces:
  - `@dataclass class UsageConfig: ledger: str = "runs/usage.json"; keep_days: int = 7; save_every: float = 60.0; balance_every: float = 300.0`
  - `class Ledger(path: Path, keep_days: int = 7)`：`add(delta: dict, today: str) -> bool`（拿不到锁 / 写失败返回 False）、`read() -> dict`（坏文件改名 `.bad-<时间>` 后返回空）、`day_rows(day: str) -> dict`（所有来源合并成 `{key: 数}`）
  - `def merge_rows(a: dict, b: dict) -> dict`（同 key 相加；cost 任一边 None 且另一边也 None 才是 None）
  - `UsageMeter.flush() -> bool`、`UsageMeter.start_saver(every: float) -> None`（守护线程）、`UsageMeter.close() -> None`（停线程 + 最后 flush）
  - `def rows_view(rows: dict) -> list[dict]`（key → snapshot 的行，带 label；Task 7 的面板复用）

- [ ] **Step 1: 写测试**
  - `test_add_creates_and_merges`：空目录 add 两次同一 key → calls 相加，文件 `version == 1`
  - `test_two_meters_add_up`：两个 meter 各 record 后先后 flush → 文件里相加；`Ledger.day_rows(今天)` 合并了 live 和 offline 两个来源
  - `test_lock_busy_keeps_delta`：手动建 `usage.json.lock`（新的）→ `flush()` 返回 False、增量还在；锁文件 mtime 改到 31 秒前 → flush 成功、锁被删
  - `test_drop_old_days`：文件里有 8 天前的日期 → add 后没了，7 天内的留着
  - `test_bad_file_renamed`：文件写 `{坏的` → add 成功、旁边出现 `usage.json.bad-*`
  - `test_snapshot_today_includes_ledger`：账本里今天已有 sandbox 的 brain 1 次 → live meter record 1 次没 flush → `snapshot()["today"]` 里 brain 的那行 calls == 2；flush 之后还是 2（不重复算）
  - `test_flush_error_not_raised`：ledger 路径是个目录 → flush 返回 False、不抛
- [ ] **Step 2:** 跑 → FAIL
- [ ] **Step 3: 实现**：锁 = `os.open(path + ".lock", O_CREAT|O_EXCL|O_WRONLY)`，最多等 2 秒（每 0.05 秒试），超过 30 秒的旧锁删掉；写 `usage.json.tmp` 后 `os.replace`。meter 记一份 `_base`（上次 flush 后读到的今天）用于 `today`；没 flush 过时构造时读一次。`flush` 失败 `restore_delta`。
- [ ] **Step 4:** 测试 → PASS
- [ ] **Step 5: Commit** `feat(models): 用量账本 runs/usage.json（今天累计，按天合并、带锁）`

### Task 5: 接上记账点

**Files:**
- Modify: `src/skydango/models/registry.py`（`Registry(..., meter=None)`、`GatedCall._send` 记账、`backend()` 把限额回调交给 Claude Code 后端）
- Modify: `src/skydango/models/claude_code.py`（`one_shot_message(..., on_message=None)`、`ClaudeCodeBackend(..., on_event=None)`）
- Modify: `src/skydango/brain/session.py`（`BrainSession(..., on_rate_limit=None)`，`_seen` 里转交 `rate_limit_event`）
- Modify: `src/skydango/brain/sessions.py`（`make_session` 传 `registry.meter`）
- Modify: `src/skydango/brain/toolloop.py`（累加 `prompt_cache_hit_tokens` → `cache_read_input_tokens`）
- Modify: `src/skydango/brain/loop.py`（`Brain(..., meter=None)`：每轮成功 / 失败、farewell 记 `brain`）
- Modify: `src/skydango/cli.py`（`_registry(cfg, workdir, environ=None, source="offline")` 建 meter + ledger + offline 时 atexit；`_run_brain` 用 `"live"` / `"sandbox"`、`start_saver`、退出 `close()`；`run --no-brain` 用 `"live"`）
- Test: `tests/test_models_registry.py`、`tests/test_models_claude_code.py`、`tests/test_brain_toolloop.py`、`tests/test_brain_loop.py`、`tests/test_brain_claude.py`、`tests/test_cli_models.py`

**Interfaces:**
- Consumes: Task 3 `UsageMeter.record / rate_limit`，Task 4 `UsageConfig`、`Ledger`、`start_saver / close`
- Produces: `Registry.meter: UsageMeter | None`；`Brain(..., meter=None)`

- [ ] **Step 1: 写测试**
  - `test_gated_call_records_main_and_backup`（registry）：主 FakeBackend 返回 usage `{input_tokens: 10, output_tokens: 2}` → meter 行 `memory|deepseek/deepseek-flash|main` calls 1 input 10；主抛 `ModelError(down="limit")` → 主那行 fails 1、备那行 calls 1（backup=True）
  - `test_gated_call_meter_error_ignored`：meter.record 抛异常 → `message()` 照样返回结果
  - `test_one_shot_rate_limit_event`（claude_code，用 `tests/fake_claude.py` 的假进程先吐一条 `{"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}` 再吐 result）→ `on_message` 收到它；经 `Registry.backend` 建的后端把它交给 `meter.rate_limit("claude", {...})`
  - `test_toolloop_cache_hit_tokens`：假客户端两次返回 `usage(prompt_tokens=100, completion_tokens=5, prompt_cache_hit_tokens=80)` → 结果 usage `cache_read_input_tokens == 160`
  - `test_brain_records_usage`（loop）：假 session 返回 `{"result": "", "usage": {...}, "provider": "deepseek", "model": "deepseek-flash"}` → meter 一行 brain main；session 抛 ModelError → fails 1；`on_fallback` 后记成 backup
  - `test_brain_session_forwards_rate_limit`（brain_claude）：假进程吐 rate_limit_event → `on_rate_limit` 收到 info
  - `test_registry_offline_meter`（cli_models）：`_registry(cfg, tmp)` 的 `registry.meter.source == "offline"`、ledger 路径 = `cfg.usage.ledger`
- [ ] **Step 2:** 跑 → FAIL
- [ ] **Step 3: 实现**：所有 meter 调用包 `try/except Exception: log.debug(...)`。`Brain` 里 provider / model 取结果里的，缺了用 `self.provider()` / session.model。
- [ ] **Step 4:** `python -m pytest -q` → 全过（基线那条除外）
- [ ] **Step 5: Commit** `feat(models): GatedCall、大脑、Claude Code 进程把用量和限额事件记进 UsageMeter`

### Task 6: DeepSeek 余额

**Files:**
- Create: `src/skydango/models/balance.py`
- Modify: `src/skydango/cli.py`（`_run_brain` 和 `run --no-brain` 起 `BalanceWatcher`，退出停）
- Test: `tests/test_models_balance.py`

**Interfaces:**
- Consumes: Task 2 `is_deepseek`，Task 3 `UsageMeter.balance`
- Produces:
  - `def fetch_balance(base_url: str, key: str, timeout: float = 10.0, opener=None) -> dict`：返回 `{"available": bool, "items": [{"currency", "total", "granted", "topped_up"}]}`，失败抛 `BalanceError(str)`
  - `def balance_targets(setup: ModelSetup, environ) -> list[tuple[ProviderConfig, str]]`（是 DeepSeek 且有 Key 的；Key 取法同 `build_client`）
  - `class BalanceWatcher(setup, meter, environ, every: float)`：`start()` / `stop()` / `poll_once()`

- [ ] **Step 1: 写测试**（`opener` 注入假的 `urlopen`）
  - `test_fetch_balance_ok`：返回 `{"is_available": true, "balance_infos": [{"currency": "CNY", "total_balance": "12.34", "granted_balance": "0.00", "topped_up_balance": "12.34"}]}` → items[0] total == 12.34（字符串转 float）；请求头 `Authorization: Bearer k`、URL `https://api.deepseek.com/user/balance`
  - `test_fetch_balance_http_error`：401 → `BalanceError` 里有 401
  - `test_balance_bad_json`：200 但内容 `<html>` / `{"x":1}` → `BalanceError("看不懂…")`
  - `test_targets_only_deepseek_with_key`：openai 官方、Ollama、没 Key 的 deepseek 都不在
  - `test_watcher_poll_once_records`：成功 → `meter.balance("deepseek", info, None)`；失败 → `meter.balance("deepseek", None, "原因")`
- [ ] **Step 2:** 跑 → FAIL
- [ ] **Step 3: 实现**：`urllib.request`（走系统代理，外网地址）；线程每 `every` 秒 `poll_once`，`stop` 用 `threading.Event`；守护线程。
- [ ] **Step 4:** 测试 → PASS
- [ ] **Step 5: Commit** `feat(models): 每 5 分钟查 DeepSeek 余额`

### Task 7: 接口：/usage 和 /api/usage

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`self.usage: Callable[[], dict] | None`，GET `/usage`，本机 Host 校验同 `/inner`）
- Modify: `src/skydango/sandbox/server.py`（`self.usage`、GET `/usage`；文档表格加一行）
- Modify: `src/skydango/brain/world.py`（`BrainParts.usage: Any = None`）
- Modify: `src/skydango/cli.py`（live 时 `viewer.usage = lambda: meter.snapshot(registry.current)`；`BrainParts(usage=…)`；沙盒 `ready` 里 `server.usage = parts.usage`；`--no-brain` 同样挂）
- Create: `src/skydango/console/usage_view.py`
- Modify: `src/skydango/console/server.py`（`SANDBOX_GET` 加 `"usage"`；GET `/api/usage`）
- Test: `tests/test_viewer.py`、`tests/test_sandbox_server.py`（没有就建）、`tests/test_console_usage.py`

**Interfaces:**
- Consumes: Task 3 `snapshot`，Task 4 `Ledger.day_rows`、`rows_view`，Task 6 `fetch_balance`、`balance_targets`
- Produces:
  - `/usage` → `{"ok": true, **snapshot}`；没挂 → 404
  - `class UsageView(store: SettingsStore, wall=time.time, fetch=fetch_balance)`：`get() -> dict` = `{"ok": true, "source": null, "run": null, "today": {...}, "providers": [...带 balance、rate=None、closed=None], "uses": [...当前配置的 main]}`；余额缓存 `balance_every` 秒，Key 用 `SecretEnv(store)`
  - 面板 `GET /api/usage`：槽在跑（团子）→ 转发 `/live/usage`；（沙盒）→ 转发 `/sandbox/usage`；转发失败或没在跑 → `UsageView.get()`

- [ ] **Step 1: 写测试**
  - `test_viewer_usage`：挂上 `viewer.usage = lambda: {"source": "live"}` → GET `/usage` 200 且 `source == "live"`；不挂 → 404；Host 不对 → 403
  - `test_sandbox_usage`：同上（没挂 → 503 `NOT_READY`）
  - `test_console_usage_stopped_reads_ledger`：tmp 里写账本（今天 brain 2 次）→ `UsageView.get()["today"]["rows"][0]["calls"] == 2`、`run is None`；假 fetch 返回余额 → providers 里 deepseek 有 balance；60 秒内第二次 get 不再调 fetch
  - `test_console_api_usage_forwards`（test_console_server 风格）：runner 状态为 running dango 且假 proxy 返回 200 → 原样返回；proxy 503 → 回落到 UsageView
- [ ] **Step 2:** 跑 → FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4:** `python -m pytest -q` → 全过（基线那条除外）
- [ ] **Step 5: Commit** `feat(console): 团子 / 沙盒接口 GET /usage，面板 /api/usage（停着读账本、自己查余额）`

### Task 8: 页面卡片

**Files:**
- Create: `src/skydango/console/static/usage.js`
- Modify: `src/skydango/console/static/console.html`（真机页左栏「现在」下、沙盒页左栏同位置各一个 `<section class="usage-card" id="live-usage">` / `id="sb-usage"`，模型页标题下 `<div id="models-usage">`；加 `<script src="console/static/usage.js">`）
- Modify: `src/skydango/console/static/live.js`、`sandbox.js`、`models.js`（挂载）
- Modify: `src/skydango/console/static/console.css`（`.usage-*`，等宽数字 `font-variant-numeric: tabular-nums`）
- Test: `tests/test_console_usage_js.py`（node，同 `test_console_markdown.py` 的跑法）、`tests/test_console_page.py`（新文件在静态列表里、没有原生弹窗）

**Interfaces:**
- Consumes: Task 7 的 `/live/usage`、`/sandbox/usage`、`/api/usage` 结构
- Produces（`usage.js`，node 能 `require`）：
  - `usageHtml(data, opts) -> HTMLElement`（纯渲染，`opts.compact` 给模型页只出一行合计 + 余额）
  - `fmtTokens(n) -> string`（`950` → `"950"`，`12300` → `"12.3k"`，`4_560_000` → `"4.56M"`）、`fmtYuan(x) -> string`（`null` → `"—"`，`0.0042` → `"¥0.004"`，`1.5` → `"¥1.50"`）
  - `mountUsage(container, urls: string[], opts) -> {stop()}`：按顺序试 `urls`（比如 `["live/usage", "api/usage"]`），第一个成功的渲染；全失败保留上一次内容；每 5 秒，`document.hidden` 时跳过

- [ ] **Step 1: 写测试**（node 里给最小 DOM 桩，或者 `usageHtml` 返回结构可以用 `textContent` 检查）
  - `test_fmt`：上面 fmtTokens / fmtYuan 的例子
  - `test_usage_card_rows`：两家（claude 关闸、deepseek 余额 12.34）、一行备用 → 文本里有「关了：额度 / 余额用完」「¥12.34」「备」；没单价的行显示「没填单价」；`est` 时合计带「估」
  - `test_usage_card_rate`：rate `windows.five_hour.utilization = 0.62` → 文本有「5 小时 62%」
  - `test_usage_card_falls_back`：`mountUsage` 的第一个 url 返回 503、第二个 200 → 渲染第二个；全部失败 → 容器内容不变
- [ ] **Step 2:** 跑 → FAIL
- [ ] **Step 3: 实现**：用 `common.js` 的 `el`；卡片可折叠（`<details open>`，开合记 localStorage，try/catch 包住）；关闸红字用 `var(--bad)` 一类现有变量（先查 `:root` 里有哪些）
- [ ] **Step 4:** `python -m pytest -q tests/test_console_usage_js.py tests/test_console_page.py` → PASS
- [ ] **Step 5: Commit** `feat(console): 真机 / 沙盒 / 模型页的「模型用量」卡片`

### Task 9: 文档和收尾

**Files:**
- Modify: `CLAUDE.md`（「模型供应商」一节加「用量和额度」小段：账本位置、单价、余额、Claude 额度、默认模型换 flash；代码结构表加 `models/usage.py` 等；运行目录表加 `runs/usage.json`）
- Modify: `config.example.toml`（`[usage]` 段、`[providers.deepseek]` 的 `prices` / `peak` / `peak_hours` 示例）
- Modify: `docs/superpowers/specs/2026-10-06-model-usage-design.md` §4 账本格式改成和实现一致的扁平 key（`"use|供应商/模型|main"`）

- [ ] **Step 1:** 改文档
- [ ] **Step 2:** `python -m pytest -q` → 全过（基线那条除外）
- [ ] **Step 3: Commit** `docs: 模型用量和额度`
