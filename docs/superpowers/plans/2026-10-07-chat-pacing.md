# 聊天的节奏和分寸（`[pacing]`）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 单聊时团子攒几句再回、回法有提示（说一句 / 只做动作 / 挑一句回）、大脑自己标 `jab` 让代码数着最近贱了几句，提醒它收一收。

**Architecture:** 新增两个纯计算模块：`brain/pacing.py`（`Pacer`：一批聊天什么时候交给大脑）和 `brain/manner.py`（`manner()`：拼「回法：…」一行）。
身体在 `_heard` 里把聊天交给 `Pacer`、先 `events.hold()` 再入队，每圈查 `ready()`，到了先冒输入气泡再 `events.release()`；大脑的 `due()` 在 `events.held()` 时不醒。
大脑醒来拼唤醒消息时经 `Brain.manner` 钩子（cli 接成 `body.call(body.manner_line)`）拿「回法」一行，插在「状态：」前；`say` 多一个 `jab` 参数，身体记最近几句。

**Tech Stack:** Python 3.13，pytest（假时钟、假设备、假事件队列），现有的 `Body` / `Brain` / `ToolBox` / `brain_prompt`。

**Spec:** `docs/superpowers/specs/2026-10-07-chat-pacing-design.md`

## Global Constraints

- 只在大脑模式生效；`--no-brain` 的普通 Agent（`agent.py`）不改。沙盒照样生效（沙盒读聊天没有 `typing`，只按安静时间）
- `[pacing] enabled = false`：0.8 秒（`chat.debounce`）叫醒、被叫到当圈冒气泡，**逐字照旧**
- `[pacing] manner = false`：唤醒消息没有「回法」行、status 没有「最近说的」、`say` 没有 `jab`、提示词「说话」一节和 `CHEEKY_RULES` **逐字照旧**
- 默认值（spec §4，数字都是估的）：`enabled = true`、`quiet_min = 2.0`、`quiet_max = 6.0`、`max_wait = 15.0`、`manner = true`、`emote_chance = 0.3`、`short_len = 6`、`jab_window = 6`、`jab_limit = 2`
- 问句词（spec §1）：「？」「?」「吗」「什么」「怎么」「为啥」「啥」「哪」「几」
- 难过类的话一律用 `inner/mind.py` 的 `sounds_upset()`（`DISTRESS`），不另列词表
- 只管 `chat` / `aside` 两种事件；`owner_command`、`aside_bg` 和别的事件不经过 `Pacer`
- 「回法」行只在这一轮取走了 `chat` / `aside`、而且没取走 `owner_command` 时加；位置在事件列表之后、「状态：」之前
- 身份底线（`responder.RULES`「身份」、`_CLAIMS_HUMAN`）、「底线」一节、`CHEEKY_RULES` 开头那句以外的招数和刹车，一字不动
- 测试命令：`python -m pytest -q`（云端是 Linux，用系统 python）

## 实现时定下的几处（spec 没写死）

- **攒着聊天时 `due()` 什么都不返回**（events / background / heartbeat 都等）：spec 只写了紧急那一条，但心跳或背景兜底这时醒来会 `drain()` 掉攒了一半的那批；最多多等 `max_wait`
- **卡洛的 `#` 命令遇到攒着的一批**：`Pacer.heard(now, [], urgent=True)`，那批当圈放掉，命令照旧马上叫醒
- **「回法」一行在大脑线程拼唤醒消息时现算**：`Event` 多一个 `speech: tuple[str, str] | None`（说话人, 原文），`Brain` 从这一轮取走的 `chat` / `aside` 里收集成 batch，调 `Brain.manner(batch)`；
  cli 把它接成 `body.call(lambda: body.manner_line(batch), timeout=3)`，出错 / 超时只记日志、这一轮不加
- 「最近 N 句里贱了 K 句」的 N 用实际记了几句（满了就是 `jab_window`）
- `jab` 参数只在 `manner` 开着时进工具（`ToolBox(jab=True)`），关掉时 `b.say(text)` 的调用和 schema 都逐字照旧
- 手动控制 / 备用回复的 `say` 记成不贱（spec：「手动控制的 say 不带，算不贱」）
- 打字中的名字和说话人按原样相等比较（无障碍读法下两边都来自名字标签）
- spec §5（`private/tools/curate_history.py` 加 `--since` / `--until`）在私有仓库里，这次云端会话拿不到，**不在这个计划里**，留给用户本机做

## Review Focus

1. **攒话时卡洛发 `#` 命令**：命令要马上叫醒大脑，不能被 hold 卡到 `max_wait` → Task 4 `test_owner_command_flushes_held_batch`
2. **攒话时到了心跳 / 背景兜底**：不能把半批聊天交给大脑 → Task 3 `test_held_blocks_heartbeat_and_background`
3. **「在打字」一直不消失**（点点卡在头顶、读法出错）：`max_wait` 到了照样放 → Task 1 `test_max_wait_releases_while_typing`
4. **大脑在攒话时转离线 / 身体忙**：攒着的那批靠 `max_wait` 放掉，`events.held()` 不会一直卡着 → Task 4 `test_hold_released_by_max_wait_without_new_chat`
5. **`manner` 钩子出错或超时**（身体线程忙）：唤醒消息照样发，只是没有「回法」行 → Task 3 `test_manner_hook_error_skips_line`

---

### Task 1: `[pacing]` 配置 + `Pacer`

**Files:**
- Modify: `src/skydango/config.py`（`LullConfig` 后面加 `PacingConfig`；`Config` 里 `reflex` 后面加 `pacing`）
- Create: `src/skydango/brain/pacing.py`
- Test: `tests/test_pacing.py`

**Interfaces:**
- Produces:
  - `config.PacingConfig`（dataclass，字段和默认值见 Global Constraints），`Config.pacing`
  - `brain/pacing.py`：`QUESTION_WORDS: tuple[str, ...]`、`is_question(text: str) -> bool`
  - `class Pacer(cfg: PacingConfig, rng: random.Random)`：
    `heard(now, lines: list[tuple[str, str]], urgent: bool) -> None`、`ready(now, typing: Collection[str]) -> bool`、`pending() -> bool`、`released(now) -> None`、
    `batch() -> list[tuple[str, str]]`（攒着的这批，从旧到新）、`describe(now, typing) -> str`（status 那一项）

- [ ] **Step 1: 写失败的测试** `tests/test_pacing.py`（`cfg = PacingConfig()`、`random.Random(1)`，时间都是传进去的数）：
  - `test_quiet_random_in_range`：多批各 `heard(t, [("小明","哈哈")], False)`，每批放行时刻 − 最后一句时刻落在 `[2.0, 6.0]` 里，且不全相等（固定种子）
  - `test_not_ready_before_quiet`：`heard(0, …)` 后 `ready(1.9, [])` 为 False
  - `test_typing_waits_then_restarts`：`quiet` 用 `quiet_min = quiet_max = 3` 固定；`heard(0)`，`ready(2, ["小明"])` False、`ready(5, ["小明"])` False（还在打字），`ready(6, [])` False（从 5 起算），`ready(8, [])` True
  - `test_other_typist_does_not_block`：批里只有小明，`ready(t, ["阿花"])` 照样按时间放
  - `test_max_wait_releases_while_typing`：`heard(0)`，`ready(14.9, ["小明"])` False，`ready(15.0, ["小明"])` True
  - `test_urgent_releases_now`：`heard(0, [("小明","我好难过")], True)` → `ready(0, ["小明"])` True
  - `test_question_uses_quiet_min`：`quiet_max = 6`，`heard(0, [("小明","你在干嘛？")], False)` → `ready(2.0, [])` True；后来一句问句也让整批改取 `quiet_min`
  - `test_ocr_no_typing`：`typing` 传空，只按时间
  - `test_released_clears_and_batches_independent`：`released(t)` 后 `pending()` False、`batch() == []`；下一批 `heard` 重新随机、重新从第一句算 `max_wait`
  - `test_urgent_without_lines_flushes_pending`：攒着一批时 `heard(t, [], True)` → `ready` True；没攒着时 `heard(t, [], True)` 后 `pending()` 仍 False
  - `test_describe`：`describe` 返回 `"在攒话：小明说了 2 句，等他说完（还在打字）"`（他在 typing 里）和 `"在攒话：小明说了 2 句，再等 3 秒"`（向上取整）；多人写 `"小明、阿花说了 3 句"`，在打字的那段写 `"等他们说完（还在打字）"`

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_pacing.py -q` → ImportError / FAIL

- [ ] **Step 3: 实现** `PacingConfig`（docstring 写 spec 路径和"数字都是估的"）和 `Pacer`。
  `Pacer` 记：第一句时刻、最后一句时刻、`quiet`（第一句进来时 `rng.uniform(quiet_min, quiet_max)`；批里有问句改成 `quiet_min`）、最后一次看到批里有人在打字的时刻、`urgent`、batch。
  `ready`：没攒着 → False；`urgent` 或 `now − 第一句 ≥ max_wait` → True；批里有人在 `typing` → 记下 now、False；否则 `now − max(最后一句, 最后看到打字) ≥ quiet`。

- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_pacing.py -q` → 全过

- [ ] **Step 5: 提交** `git commit -m "feat(brain): 攒话的纯计算 Pacer 和 [pacing] 配置"`

### Task 2: 回法 `manner()`

**Files:**
- Create: `src/skydango/brain/manner.py`
- Test: `tests/test_manner.py`

**Interfaces:**
- Consumes: `PacingConfig`、`pacing.is_question`、`inner.mind.sounds_upset`
- Produces: `manner(batch: list[tuple[str, str]], recent: list[bool], mood: str | None, sleepy: bool, rng: random.Random, cfg: PacingConfig) -> str | None`
  （`mood` 是 `Mind.mood.level`：开心 / 平常 / 低落 / 烦；`sleepy` = 精力档是「困」）；
  `jab_note(recent: list[bool]) -> str`（status：`"最近说的：6 句里贱了 2 句"`，空列表返回 `""`）

固定文字（各项用「；」连，整行前缀 `回法：`）：
- `最近 {len(recent)} 句里贱了 {sum} 句，这轮好好说`
- `没力气贫，懒一点回`
- `想贫可以贫一句`
- `这几句可以只回个动作，不用说话`
- `{他|他们}连着说了 {n} 句，挑最想接的回一句就行，不用每句都回`

- [ ] **Step 1: 写失败的测试** `tests/test_manner.py`（`Rng(r)` 用固定返回值的假随机数，像 `test_brain_reflex.Rng`）：
  - `test_too_many_jabs`：`recent=[T,F,T,F,F,F]` → 以 `"回法：最近 6 句里贱了 2 句，这轮好好说"` 开头
  - `test_last_jab_counts`：`recent=[F,F,T]` → 含 `"最近 3 句里贱了 1 句，这轮好好说"`
  - `test_tired_or_low`：`recent=[]`、`mood="低落"` → 含 `"没力气贫，懒一点回"`；`mood="烦"`、`sleepy=True` 同；贱太多时只出第一档（三选一，命中即停）
  - `test_happy_may_jab`：`mood="开心"`、`recent=[T,F,F]`（最后一句不贱）→ 含 `"想贫可以贫一句"`；`recent=[T,T,F]` 时是第一档
  - `test_nothing_to_say`：`batch=[("小明","今天去哪个图？")]`、`recent=[F]`、`mood="平常"` → `None`
  - `test_emote_hint_short_no_question`：`batch=[("小明","哈哈"),("小明","笑死")]`、`Rng(0.2)` → 含 `"这几句可以只回个动作，不用说话"`；`Rng(0.4)` 不含
  - `test_emote_hint_not_for_question_or_long_or_upset`：有问句 / 有一句去掉空白 > 6 字 / 有「难过」时都不含，且这时不消耗随机数（`Rng` 记调用次数为 0）
  - `test_sleepy_emote_chance_capped`：`sleepy=True`、`Rng(0.44)` 含（0.3×1.5=0.45）；`emote_chance=0.5`、`sleepy=True`、`Rng(0.59)` 含、`Rng(0.61)` 不含（上限 0.6）
  - `test_many_lines`：同一人 3 句 → 含 `"他连着说了 3 句，挑最想接的回一句就行，不用每句都回"`；两个人 → `"他们连着说了 2 句…"`；1 句不含
  - `test_order_and_join`：贱太多 + 动作 + 多句同时命中 → `"回法：最近 6 句里贱了 3 句，这轮好好说；这几句可以只回个动作，不用说话；他连着说了 2 句，挑最想接的回一句就行，不用每句都回"`
  - `test_jab_note`：`jab_note([T,F])` == `"最近说的：2 句里贱了 1 句"`，`jab_note([]) == ""`

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_manner.py -q`
- [ ] **Step 3: 实现** `manner` / `jab_note`（顺序：贱的分寸 → 只做动作 → 攒了好几句；只做动作的概率 `min(emote_chance × (1.5 if sleepy else 1), 0.6)`，`sleepy` 时才乘，没乘时也不超过原值）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `git commit -m "feat(brain): 回法提示 manner()"`

### Task 3: 事件队列 hold / speech + 大脑醒来的条件和「回法」行

**Files:**
- Modify: `src/skydango/brain/events.py`（`Event.speech`、`put(..., speech=None)`、`hold()` / `release()` / `held()`）
- Modify: `src/skydango/brain/loop.py`（`due()`、`message()`、`Brain.manner`）
- Test: `tests/test_brain_events.py`、`tests/test_brain_loop.py`（各加几条）

**Interfaces:**
- Produces:
  - `Event.speech: tuple[str, str] | None = None`；`EventQueue.put(kind, text, who="", speech=None)`（合并 / 抵消的逻辑不变，新建 Event 时带上 speech）
  - `EventQueue.hold() -> None`、`EventQueue.release() -> None`、`EventQueue.held() -> bool`（加锁读写；`release()` 也 `notify_all`）
  - `Brain.manner: Callable[[list[tuple[str, str]]], str | None] | None = None`（构造后由 cli 设，像 `on_text`）
- `due()`：开头 `if self.events.held(): return None`（在 backoff 检查之后），其余不变
- `message()`：事件行之后、`"状态："` 之前，若 `self.manner` 不为空、这批有 `chat` / `aside` 且有 `speech`、没有 `owner_command`：`line = self.manner(batch)`，非空就 `lines.append(line)` 并 `log.info(line)`；钩子抛异常 → `log.warning` 一行、不加

- [ ] **Step 1: 写失败的测试**
  - `test_brain_events.py::test_hold_release`：`held()` 初始 False，`hold()` 后 True，`release()` 后 False；`put("chat", "x", speech=("小明","x"))` 后 `drain()[0].speech == ("小明","x")`
  - `test_brain_loop.py::test_held_blocks_urgent_until_release`：放一条 `chat`、`hold()`，过了 debounce `due()` 是 None；`release()` 后是 `"events"`
  - `test_brain_loop.py::test_held_blocks_heartbeat_and_background`：`hold()` 时到了心跳 / 背景兜底时间 `due()` 都是 None，`release()` 后照旧
  - `test_brain_loop.py::test_not_held_urgent_still_debounce`：没 hold 时 `arrive` 照旧 debounce 后醒（老行为）
  - `test_brain_loop.py::test_manner_line_before_status`：`brain.manner = lambda b: "回法：想贫可以贫一句"`，两条 `chat`（带 speech）一轮 → `session.sent[0]` 里「回法：想贫可以贫一句」那行紧挨在「状态：」那行之前，钩子收到的 batch 是两条 speech
  - `test_brain_loop.py::test_manner_only_on_chat_turns`：只有 `arrive` / 心跳 / 有 `owner_command` 的那一轮，钩子没被调、消息里没有「回法」
  - `test_brain_loop.py::test_manner_hook_error_skips_line`：钩子抛 `RuntimeError` → 照样发、没有「回法」
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_events.py tests/test_brain_loop.py -q`
- [ ] **Step 3: 实现**（见 Interfaces）
- [ ] **Step 4: 跑测试确认通过**，再跑 `python -m pytest tests/test_brain_toolloop.py tests/test_brain_toolloop_compact.py -q` 确认 inbox 行仍在「状态：」前、基准没变
- [ ] **Step 5: 提交** `git commit -m "feat(brain): 事件队列能扣住聊天、唤醒消息带回法行"`

### Task 4: 身体攒话、放行时冒气泡、status「在攒话」

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__` 建 `self.pacer = Pacer(cfg.pacing, self.rng)`、`self._pace_bubble = False`；`_heard`、`_on_heard`、`step`、新的 `_watch_pacing` / `_release_pacing`、`status`）
- Modify: `tests/test_brain_body.py`（`body()` 辅助函数加 `cfg.pacing.enabled = kw.pop("pacing", False)`，注释「旧行为的测试：不攒话；test_brain_pacing_body 打开」）
- Test: `tests/test_brain_pacing_body.py`

**Interfaces:**
- Consumes: Task 1 `Pacer`、Task 3 `events.hold/release/held`、`put(..., speech=)`
- 行为：
  - `_heard`（只在 `cfg.pacing.enabled`）：入队之前算出这一批要成为 `chat` / `aside` 的 `(说话人, 原文)`；非空就先 `events.hold()` 再照旧 `put`（`chat` / `aside` 都带 `speech`），之后 `pacer.heard(now, paced, urgent=any(sounds_upset(t) for _, t in paced))`；
    这批里有卡洛的 `#` 命令且 `pacer.pending()` → `pacer.heard(now, [], urgent=True)`。`enabled = false` 时 `put` 也不带 speech 以外的变化（带 speech 不影响事件文字）
  - `_on_heard`：`pacing.enabled` 时做完小动作反射后不开框，改成 `self._pace_bubble = True` 返回（前面的别扭 / 技能 / `_bubble_blocked` 拦法不变）
  - `_watch_pacing(now)`（`step` 里放在 `_run_commands()` 之前）：`pacer.pending()` 且 `pacer.ready(now, self._typists())` → `_release_pacing(now)`
  - `_release_pacing(now)`：`_pace_bubble` 时再查一遍技能在跑 / `_bubble_blocked()` / `cfg.reflex.bubble` / dry-run（dry-run 照旧只记日志），能冒就 `_open_bubble(now)`；然后 `_pace_bubble = False`、`pacer.released(now)`、`events.release()`；冒气泡出错也照样放
  - `_typists() -> list[str]`：`reader.typing()`（没有这个方法返回 `[]`；`status` 里原来那两行改用它）
  - `status`：`pacer.pending()` 时加一项 `pacer.describe(now, self._typists())`（放在「在打字」后面）

- [ ] **Step 1: 写失败的测试** `tests/test_brain_pacing_body.py`（照 `test_brain_reflex_body.rx` 搭：`body(clock, live=True, pacing=True, …)`、`cfg.pacing.quiet_min = cfg.pacing.quiet_max = 3.0`；`clock` 是 conftest 的假钟，能 `advance`）：
  - `test_chat_held_until_quiet`：小明说一句 → `step()` 后 `events.held()` True、`events` 里有 `chat`；推 2.9 秒 `step()` 仍 held；推到 3 秒后 `step()` → `held()` False
  - `test_bubble_opens_at_release`：反射开着、被叫到（`addressed` 空、`bubble = True`）：说话那圈 `sender.opened` False；放行那圈 True、`OPEN` 按键在放行那圈才出现
  - `test_bubble_blocked_still_releases`：放行时 `_requests` 非空（有互动请求挂着）→ 不开框、`held()` 照样 False
  - `test_typing_keeps_holding`：`reader.typing = lambda: ["小明"]`，过 3 秒仍 held；改成 `[]` 后再过 3 秒放
  - `test_upset_releases_same_step`：「我今天好难过」→ 同一圈 `held()` False
  - `test_owner_command_flushes_held_batch`：`owner_name` 设好，小明一句被扣着，卡洛 `#过来` 同一批或下一圈进来 → 那一圈 `held()` False、队列里 `owner_command` 和 `chat` 都在
  - `test_hold_released_by_max_wait_without_new_chat`：`reader.typing` 一直是 `["小明"]`，推到 15 秒 `step()` → 放
  - `test_status_shows_pacing`：攒着时 `status()` 含 `"在攒话：小明说了 1 句，再等 3 秒"`；放了之后不含
  - `test_pacing_off_unchanged`：`pacing=False` 时说话那圈 `held()` False、气泡当圈开（同 `test_brain_reflex_body.test_nod_then_bubble`）
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_brain_pacing_body.py -q`
- [ ] **Step 3: 实现**（见 Interfaces）
- [ ] **Step 4: 跑测试确认通过**，再跑全部身体相关测试 `python -m pytest tests -q -k "body or reflex or lull or addressee"` 确认老行为没变
- [ ] **Step 5: 提交** `git commit -m "feat(brain): 身体攒几句再叫醒大脑、放行时才冒输入气泡（[pacing]）"`

### Task 5: `say(jab)`、身体记账、回法和 status「最近说的」

**Files:**
- Modify: `src/skydango/brain/tools.py`（`ToolBox(..., jab: bool = False)`、`JAB_NOTE`、`_bind("say")`）
- Modify: `src/skydango/brain/llm_tools.py`（`openai_tools`：`toolbox.jab` 时 say 多 `("jab", "boolean", False)`、描述接 `JAB_NOTE`）
- Modify: `src/skydango/brain/mcp_server.py`（`toolbox.jab` 时注册带 `jab: bool = False` 的 say，否则原样）
- Modify: `src/skydango/brain/body.py`（`say(..., jab: bool = False)`、`self._jabs: deque[bool]`、`manner_line(batch)`、status）
- Test: `tests/test_brain_llm_tools.py`、`tests/test_brain_tools.py`（若没有就放进 `tests/test_brain_pacing_body.py`）、`tests/test_brain_pacing_body.py`

**Interfaces:**
- Consumes: Task 2 `manner` / `jab_note`
- Produces:
  - `tools.JAB_NOTE = "jab：这句算不算犯贱（损人、故意曲解装傻、甩锅、自恋嘴硬、明褒暗贬都算），照实标。"`（接在 say 描述后面，前面一个空格）
  - `Body.say(text, live=False, reply=False, jab=False)`：过了所有拦截（`limiter.record` 那里）之后 `self._jabs.append(bool(jab) and not live and not reply)`；`_jabs` 是 `deque(maxlen=cfg.pacing.jab_window)`，dry-run 也记
  - `Body.manner_line(batch: list[tuple[str, str]]) -> str | None`：`manner(batch, list(self._jabs), mind.mood.level 或 None, 精力档 == "困", self.rng, cfg.pacing)`，内心层出错按没有
  - `status`：`cfg.pacing.manner` 且 `_jabs` 非空时加 `jab_note(list(self._jabs))`（放在「刚说过」后面）

- [ ] **Step 1: 写失败的测试**
  - `test_brain_llm_tools.py::test_say_jab_only_when_on`：`ToolBox(..., jab=False)` 时 say 的 schema 和改之前逐字一样（和现有断言比 / 存一份 `json.dumps` 基准）；`jab=True` 时 properties 多 `jab: {"type": "boolean", "default": False}`、描述以 `JAB_NOTE` 结尾
  - `test_toolbox_say_passes_jab`：假 body 的 `say(text, jab=False)` 记参数；`ToolBox(jab=True).run("say", {"text": "嗯", "jab": True})` → 收到 `jab=True`；`ToolBox(jab=False)` 时 `say` 只收到 text（假 body 的 say 只接 text 也不报错）
  - `test_brain_pacing_body.py::test_jab_recorded_after_send`：`say("你才废物", jab=True)`、`say("好", jab=False)` → `_jabs == [True, False]`；被重复拦下的（同一句再说）不记；手动 `say(..., live=True, jab=True)` 记 False；超过 `jab_window` 只留最近 6 个
  - `test_status_recent_jabs`：说过两句（一句贱）→ status 含 `"最近说的：2 句里贱了 1 句"`；`manner = False` 时不含；没说过不含
  - `test_manner_line_uses_mood_and_energy`：`b.mind` 换成心情「低落」的假对象 → `manner_line([("小明","今天去哪个图？")])` 含 `"没力气贫"`；`b.mind = None` 不报错
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（见 Interfaces）
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest tests/test_brain_llm_tools.py tests/test_brain_pacing_body.py tests -q -k "tools or pacing"`
- [ ] **Step 5: 提交** `git commit -m "feat(brain): say 标 jab、身体记最近贱了几句、回法按心情精力给"`

### Task 6: 提示词、接线、设置、文档

**Files:**
- Modify: `src/skydango/brain/prompt.py`（`MANNER_RULES`、`CHEEKY_FIRST_OLD` / `CHEEKY_FIRST_NEW`、`brain_prompt(..., manner: bool = False)`）
- Modify: `src/skydango/cli.py`（`_run_brain`：`ToolBox(..., jab=cfg.pacing.manner)`、`brain_prompt(..., manner=cfg.pacing.manner)`、`if cfg.pacing.manner: brain.manner = …`）
- Modify: `src/skydango/console/settings.py`（`pacing.enabled` / `pacing.manner` 两个 bool，`brain` 组）
- Modify: `config.example.toml`（`[pacing]` 一节，放在 `[reflex]` 后面）、`CLAUDE.md`（新一节「聊天的节奏和分寸」+ 代码结构表加 `pacing.py` / `manner.py`）
- Test: `tests/test_brain_prompt.py`、`tests/test_console_settings.py`（若有字段清单断言）

**Interfaces:**
- `MANNER_RULES`（插在 `SAY_FIRST` 后面，在 `bubble` 替换之前做，结果顺序 SAY_FIRST → BUBBLE_NOTE → MANNER_RULES）：
  ```
  - 回法有三种，都是正常的：说一句；只做个动作不说话（用 emote，不 say）；对方连着说了好几句，挑最想接的那句回，别每句都回。
  - 唤醒消息里的「回法：…」是身体根据你最近的样子给的建议，照着来，除非对方在认真问你事情。
  - say 的时候用 jab 照实标出这句算不算犯贱（损人、故意曲解装傻、甩锅、自恋嘴硬、明褒暗贬都算）。
  ```
- `CHEEKY_FIRST_NEW`（`manner` 且 `cheeky` 时替换 `CHEEKY_RULES` 第一行，其余行不动）：
  `- 你平时正常聊天，偶尔犯一下贱才好笑（像个爱犯贱的损友，不是刻薄）。什么时候可以贱：他先损你、你心情好、气氛正热；连着贱了两句就收一收，正常说几句再说。常用的几招：`
- cli：`brain.manner = lambda batch: body.call(lambda: body.manner_line(batch), timeout=3)`（`body.call` 超时抛的 `ToolError` 由 Task 3 的 `message()` 兜住）

- [ ] **Step 1: 写失败的测试** `tests/test_brain_prompt.py`：
  - `test_manner_off_is_verbatim`：各种开关组合下 `brain_prompt(..., manner=False)` 和不传 `manner` 完全相等（包括 `cheeky=True, temper=True`）
  - `test_manner_rules_after_say_first`：`manner=True, bubble=True` → `SAY_FIRST + BUBBLE_NOTE + MANNER_RULES` 是子串
  - `test_cheeky_first_line_replaced`：`manner=True, temper=True, cheeky=True, inner=True` → 含 `CHEEKY_FIRST_NEW`、不含「别句句都贱」，「故意曲解、装傻」「刹车：」「刚认识的、陌生人照常好好说话，不贱。」还在；`manner=True, cheeky=False` 时没有贱的那几行
  - `test_console_settings.py`：`pacing.enabled`、`pacing.manner` 在设置清单里、类型 bool（照现有断言的写法）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** prompt、settings、cli 接线；写 `config.example.toml` 的 `[pacing]`（每项一行注释，照 `[lull]` 的格式，开头一行说明 + spec 路径 + "数字都是估的，没在真机验证"）；
  CLAUDE.md 加「聊天的节奏和分寸（`[pacing]`，大脑模式）」一节（设计 / 计划路径、攒话规则、回法、`jab`、两个开关逐字照旧、**代码做完，spec §7 沙盒 / 真机验证没走，spec §5 清 history 要用户在本机做**）
- [ ] **Step 4: 全量测试**：`python -m pytest -q` → 全过
- [ ] **Step 5: 提交** `git commit -m "feat(brain): 回法写进提示词、贱的开头一句改成以正常为底色；接线、设置、文档"`

### 收尾

- [ ] `requesting-code-review`：整支分支一次评审（Native 方式），按 `receiving-code-review` 处理意见
- [ ] `verification-before-completion`：`python -m pytest -q` 全过的输出
- [ ] 按 CLAUDE.md：分支合并进 `main` 并推送；功能分支 `claude/busy-wright-rmohnu` 也推送
