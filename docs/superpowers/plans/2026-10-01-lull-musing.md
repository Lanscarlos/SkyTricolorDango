# 冷场时的心理活动 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 好友在身边不说话了 / 聊着聊着走了时，身体认出冷场、按节点叫醒大脑，大脑在心里写一句“心里：…”，想法连贯推进、冷场结束时带给大脑、进反思材料和各处可视化；顺带修“接话被算成主动开口”和规则被套错。

**Architecture:** 新的纯计算模块 `inner/lull.py`（`LullTracker`）管冷场的开始 / 节点 / 结束和想过的话；身体每圈喂它（身边的人、`body.chat`、黑屏、跟踪中），把它返回的文字放成新的立刻叫醒事件 `lull`；大脑循环每轮结束把最后的文字交回身体（`Brain.on_text` → `body.mused`），身体取出“心里：”记到冷场上。提示词加「冷场的时候」一节。

**Tech Stack:** Python 3.13、pytest（合成画面 + 假设备）、管理面板原生 JS。

**Spec:** `docs/superpowers/specs/2026-10-01-lull-musing-design.md`

## Global Constraints

- 用中文写注释、日志、提示词、给大脑看的文字；注释密度、命名照周围代码。
- 测试命令：`.venv\Scripts\python.exe -m pytest -q`（系统 Python 没装 pytest 时）；git worktree 里也用主目录的 `.venv`。
- 时间：冷场全部用**墙钟**（`body.wall()`，和 `body.chat`、`occasion` 同一个时钟）。
- `[lull] enabled = false`：不认冷场、`leave` / `return` 照旧是背景事件、提示词和 status **逐字照旧**（只有修复 2 那一句的“没人接”定义变了，修复 1 照样生效）。
- 配置默认值：`enabled = true`、`talk_window = 300.0`、`stages = [60.0, 180.0, 360.0]`、`leave_spoke = 120.0`、`leave_said = 60.0`、`musing_max = 60`。
- 心理活动**不写** `history.jsonl`、随手记、关系卡。
- 只在大脑模式生效；**不依赖内心层**（`mind` 为 None 时照样叫醒、记“在想”，只是不进反思材料、不记流水账）。
- 流水账 `mind_log` 只在 live 写盘（`MindLog` 本身已经按 persist 处理）。
- 每个任务改完跑全量测试再提交；提交信息结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## Review Focus

- **好友名字 OCR 错一两个字**（聊天里“懒洋洋大玉”、身边名单里“懒洋洋大王”）：冷场对象、结束判断、走开判断都要按 `similar(…, 0.75)` 模糊匹配，不能精确比较 → Task 1、2 各有一条测试。
- **好友说的话被认成“看不出是谁”（speaker 为空）**：不算好友开口、不结束冷场、不算“刚才在聊” → Task 1 测试。
- **大脑最后的文字里有多行“心里：”或半角冒号、前后空格、超长**：取最后一行、两种冒号都认、截 `musing_max` 字 → Task 1 测试。
- **冷场刚结束（好友开口）后又安静 60 秒**：要能开始新的一次冷场，但同一句“最后一句”不重复开始 → Task 1 测试。
- **大脑这一轮正好跨过冷场结束**（大脑在想的时候好友开口了，文字回来时冷场已经没了）：这句“心里：”丢掉，不挂到下一次冷场上 → Task 1 测试（`muse` 在没有冷场时返回 False）。

---

### Task 1: 配置 + 冷场追踪（情况①、想法、状态、总结）

**Files:**
- Modify: `src/skydango/config.py`（`ProactiveConfig` 之后加 `LullConfig`；`Config` 加 `lull: LullConfig`，放在 `proactive` 后）
- Create: `src/skydango/inner/lull.py`
- Test: `tests/test_inner_lull.py`

**Interfaces:**
- Produces:
  - `LullConfig`（dataclass，字段见 Global Constraints，docstring 指向 spec）
  - `Lull`（dataclass）：`kind: str`（"silent" / "left"）、`who: tuple[str, ...]`、`t0: float`、`last: tuple[str, str] | None`（(说话人, 内容)，团子是 `"我"`）、`stage: int = -1`（已叫醒到第几个节点的下标）、`musings: list[tuple[float, float, str]]`（(墙钟, 冷了几秒, 想法)）、`said: list[str]`（冷场中团子又说的）、`ending: str = ""`（怎么结束的，给总结用）
  - `Cue`（frozen dataclass）：`text: str`、`final: bool`（这是这次冷场的最后一个节点）
  - `parse_musing(text: str, limit: int) -> str`：取最后一行 `^\s*心里[：:]\s*(.+)$`，strip，截 `limit` 字；没有返回 `""`
  - `class LullTracker(cfg: LullConfig, is_friend: Callable[[str], bool])`：
    - `tick(wall: float, nearby: Sequence[str], chat: Sequence[tuple[float, str, str]], paused: bool = False) -> list[Cue]`
    - `heard(wall: float, speaker: str, text: str) -> str`：好友开口时结束冷场，返回附在这条聊天后面的话（没有冷场返回 `""`）
    - `muse(text: str, wall: float) -> bool`：挂到所有活着的冷场上；没有冷场返回 False
    - `status(wall: float) -> str`：没有冷场返回 `""`
    - `active() -> list[Lull]`、`pop_finished() -> list[Lull]`、`flush(wall: float) -> list[Lull]`（下线前：把活着的都当作结束取走，`ending = "到下线还没结束"`）
    - `summary(lull: Lull, wall: float) -> str`（给反思材料的一行）
    - `snapshot(wall: float) -> list[dict]`：`{"kind", "who": [...], "since": t0, "last": [说话人, 内容] | None, "musings": [{"t", "text"}]}`

情况① 的规则照 spec §1；实现要点：
- 开始：`chat` 最后一行 `(t, who, text)`，`wall - t >= stages[0]`，`[t - talk_window, t]` 内有 `is_friend(说话人)` 的行，且这些说话人里至少一个能在 `nearby` 里模糊匹配上（`similar(说话人, 名字, 0.75)`，来自 `..chat.tracker`）；对象 = 匹配上的 `nearby` 名字（去重、按最后说话时间排，最后说话的在最后）。`t0 = t`；`t0 <= self._done_t0`（上一次冷场的 t0）时不开始。
- 节点：`k = 已过的最大节点下标`，`k > lull.stage` 时发一条 `Cue`（只发最新的），`final = (k == len(stages) - 1)`。
- `paused`（黑屏）：本圈不开始、不发节点，所有活着的冷场 `t0 += wall - 上一圈 wall`（不推进）。
- 团子在冷场中说的（`chat` 里 `t > t0`、说话人 `"我"`）记进 `said`，不重新计时。
- 对象都不在 `nearby` 了：冷场静静结束（不进 `pop_finished`，交给情况②，Task 2 把想法转过去）。
- 时长写法：`< 60` 秒写“N 秒”，否则“N 分钟”（`int(秒 // 60)`）；引用的话截 30 字。

文字（逐字，`{}` 是填充）：
- Cue（团子最后说的）：`冷场  {对象} {时长}没说话了。最后是你说的「{内容}」，{没人接|他没接}。` 对象一人时“他没接”，几人时“没人接”；几人时对象写 `大家（{名字、名字}）`
- Cue（好友最后说的）：`冷场  {对象} {时长}没说话了。最后是{说话人}说的「{内容}」，你没接。`
- 有 `said` 时句末追加：`这之间你又说了「{said[-1]}」。`
- `status`：`冷场：{对象} {时长}没说话（最后是{你|说话人}说的「{内容}」）· 在想：{最后一条想法}（{冷了几秒的时长}时）`；还没想过写 `· 在想：（还没想过）`
- `heard` 的返回：`（冷场了 {时长}，你刚才在想：{最后一条想法}）`；没想过 `（冷场了 {时长}）`
- `summary`：`{时长}，{对象}没接你的话。你心里想过：{想法1} → {想法2}。{ending}`（好友最后说的写 `{对象}说完你没接，大家都没再说话。`；没想过省掉“你心里想过”那句）；`ending` 由 `heard` 写成 `后来{说话人}说「{内容}」`

- [ ] **Step 1: 写失败的测试** `tests/test_inner_lull.py`（`T0 = 1_790_000_000.0`；`is_friend = lambda w: bool(w) and w != "我" and any(similar(w, n, 0.75) for n in ["懒洋洋大王", "阿花"])`）：
  - `test_starts_after_first_stage`：chat `[(T0, "懒洋洋大王", "在吗"), (T0+5, "我", "在呢")]`，nearby `["懒洋洋大王"]`；`tick(T0+64)` → `[]`；`tick(T0+65)` → 一条，文字 == `"冷场  懒洋洋大王 1 分钟没说话了。最后是你说的「在呢」，他没接。"`，`final is False`
  - `test_each_stage_once`：接上，`tick(T0+100)` → `[]`；`tick(T0+185)` 一条含“3 分钟”；`tick(T0+365)` 一条且 `final is True`；`tick(T0+900)` → `[]`
  - `test_fast_forward_only_latest`：开始前直接 `tick(T0+400)` → 只有一条、含“6 分钟”、`final is True`
  - `test_friend_spoke_last`：chat 最后是 `(T0, "懒洋洋大王", "我去倒水")` → 文字 == `"冷场  懒洋洋大王 1 分钟没说话了。最后是懒洋洋大王说的「我去倒水」，你没接。"`
  - `test_group`：阿花、懒洋洋大王都在 300 秒内说过、都在身边 → 文字以 `"冷场  大家（阿花、懒洋洋大王）"` 开头（按最后说话时间排）
  - `test_no_recent_friend_talk`：好友那句在最后一句之前 301 秒 → 永远不开始；speaker 为 `""` 的行不算好友
  - `test_ocr_typo`：chat 里说话人 `"懒洋洋大玉"`、nearby `["懒洋洋大王"]` → 开始，对象写 `懒洋洋大王`
  - `test_said_during_lull_no_restart`：开始后 chat 追加 `(T0+100, "我", "人呢")`，`tick(T0+185)` 文字以 `"冷场  懒洋洋大王 3 分钟没说话了。"` 开头、以 `"这之间你又说了「人呢」。"` 结尾
  - `test_heard_ends_with_note`：开始后 `muse("他是不是去忙了", T0+70)` 为 True；`heard(T0+130, "懒洋洋大王", "刚去倒水")` == `"（冷场了 2 分钟，你刚才在想：他是不是去忙了）"`；`active() == []`；`pop_finished()` 一个，其 `summary` 含 `"你心里想过：他是不是去忙了。后来懒洋洋大王说「刚去倒水」"`；第二次 `heard` 返回 `""`
  - `test_heard_stranger_does_not_end`：`heard(…, "", "hi")` 返回 `""`，冷场还在
  - `test_same_last_line_not_restarted`：结束后不加新聊天，`tick(T0+500)` → `[]`；再追加 `(T0+510, "懒洋洋大王", "嗯")`，`tick(T0+575)` → 新的一条
  - `test_paused_does_not_advance`：开始（T0+65）后 `tick(T0+100, paused=True)`、`tick(T0+300, paused=True)`、`tick(T0+310)` → `[]`（暂停的 200 秒不算）；`tick(T0+385)` → 一条“3 分钟”
  - `test_objects_left_silent_end`：开始后 `tick(…, nearby=[])` → `[]`、`active() == []`、`pop_finished() == []`
  - `test_muse_without_lull`：`muse("随便", T0)` 是 False
  - `test_status`：开始前 `status == ""`；开始后 `"冷场：懒洋洋大王 1 分钟没说话（最后是你说的「在呢」）· 在想：（还没想过）"`；muse 后结尾是 `"· 在想：他是不是去忙了（1 分钟时）"`
  - `test_parse_musing`：`parse_musing("想了想\n心里：他忙去了吧\n不说：等", 60) == "他忙去了吧"`；`"心里: 嗯 "` → `"嗯"`；两行取最后一行；`"心里：" + "长"*80` 截成 60 字；没有 → `""`
  - `test_flush`：活着的冷场 `flush(T0+200)` 返回一个、`ending == "到下线还没结束"`、之后 `active() == []`
- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_inner_lull.py -q` → ImportError / FAIL
- [ ] **Step 3: 实现** `LullConfig`、`inner/lull.py`（模块 docstring 指向 spec §1~§3）
- [ ] **Step 4: 跑测试确认通过**：同上 → PASS；再跑全量 `.venv\Scripts\python.exe -m pytest -q` → 全过
- [ ] **Step 5: 提交** `feat(inner): 冷场追踪（情况①、想法、总结）`

### Task 2: 冷场追踪：情况②（聊着聊着走了）

**Files:**
- Modify: `src/skydango/inner/lull.py`
- Test: `tests/test_inner_lull.py`

**Interfaces:**
- Consumes: Task 1 的 `LullTracker`、`Lull`、`Cue`
- Produces:
  - `LullTracker.left(name: str, wall: float, chat: Sequence[tuple[float, str, str]]) -> str | None`：算“聊着聊着走了”就开始一个 `kind="left"` 冷场（`t0 = wall`，`stage = 0`）、返回事件文字，否则 None
  - `LullTracker.returned(name: str, wall: float) -> str | None`：有这个人的情况② 冷场就结束它、返回附注，否则 None
  - `tick` 也推进情况②：节点只用 `stages[1:]`（下标 1、2），`final` 同样是最后一个节点；黑屏同样不推进
  - `heard` 也结束说话人对应的情况② 冷场（模糊匹配）

规则：`left` 条件 = `chat` 里 `wall - leave_spoke` 之后有说话人模糊匹配 `name` 的行，或 `wall - leave_said` 之后有 `"我"` 的行。这个人是情况① 冷场的对象时：从对象里去掉；对象空了就把它的 `musings` 转给新的情况② 冷场、情况① 静静结束。

文字（逐字）：
- `left`：`冷场  {name} 聊着聊着走开了。走之前最后是{你|他|说话人}说的「{内容}」。`（最后一句取 `chat` 最后一行；说话人是 name 本人写“他”）
- 情况② 节点：`冷场  {name} 走开 {时长}了，还没回来。`
- `returned` 返回：`（走开了 {时长}，你刚才在想：{最后一条想法}）`；没想过 `（走开了 {时长}）`
- `status` 里情况② 那段：`冷场：{name} 走开 {时长}了 · 在想：…`（两种都有时用 `；` 连起来，共用一个“冷场：”前缀）
- `summary`：`{name}聊着聊着走开了 {时长}。你心里想过：…。{ending}`；`returned` 把 `ending` 写成 `后来他回来了`，`heard` 写成 `后来他说「{内容}」`

- [ ] **Step 1: 写失败的测试**（加到 `tests/test_inner_lull.py`）：
  - `test_left_after_he_spoke`：chat `[(T0, "懒洋洋大王", "我去拿个东西")]`，`left("懒洋洋大王", T0+100, chat)` == `"冷场  懒洋洋大王 聊着聊着走开了。走之前最后是他说的「我去拿个东西」。"`
  - `test_left_after_i_spoke`：chat 最后 `(T0+50, "我", "你要牵好我哦")`，好友那句在 T0-200；`left(…, T0+100)` 含 `"最后是你说的「你要牵好我哦」"`
  - `test_left_not_chatting`：好友 121 秒前说过、团子 61 秒前说过 → None；名字 OCR 错（chat 里“懒洋洋大玉”）照样算
  - `test_left_stages`：`left` 后 `tick(T0+100+179)` → `[]`；`tick(T0+100+180)` == `[Cue("冷场  懒洋洋大王 走开 3 分钟了，还没回来。", False)]`；`+360` 一条 `final=True`
  - `test_returned`：`muse("是不是我说错话了", …)` 后 `returned("懒洋洋大王", T0+100+200)` == `"（走开了 3 分钟，你刚才在想：是不是我说错话了）"`；`pop_finished()` 的 summary 以 `"懒洋洋大王聊着聊着走开了 3 分钟。"` 开头、以 `"后来他回来了"` 结尾；不在冷场的人 `returned` 返回 None
  - `test_left_takes_over_silent`：情况① 冷场（对象只有他）、muse 过 → `left` 后 `active()` 只有一个 `kind == "left"`，其 `musings` 带着之前那条
  - `test_both_kinds_status`：阿花情况① + 懒洋洋大王情况② 同时 → `status` 以 `"冷场："` 开头、含 `"；"`、两人名字都在
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**（含全量）
- [ ] **Step 5: 提交** `feat(inner): 冷场追踪：聊着聊着走了`

### Task 3: 身体接上冷场（事件、附注、想法、状态、反思材料）

**Files:**
- Modify: `src/skydango/brain/body.py`；`src/skydango/brain/events.py`（`Event.kind` 注释加 `lull`，**不**加进 `BACKGROUND`）；`src/skydango/inner/log.py`（`MindLog.musing`）
- Test: `tests/test_brain_lull_body.py`（helper 照 `tests/test_brain_reflex_body.py`：`from test_brain_body import FakeEnv, body, msg`；`wall=clock`，`b.cfg.lull.enabled = True`，`b.friend_names = lambda: ["懒洋洋大王"]`）；`tests/test_inner_log.py`

**Interfaces:**
- Consumes: Task 1、2 的 `LullTracker` 全部方法、`parse_musing`
- Produces:
  - `Body.lulls: LullTracker | None`（`cfg.lull.enabled` 时建；`is_friend` 用 `lambda who: is_friend_fn(self.friend_names())(who)`，每次现取好友名单）
  - `Body.on_musing: Callable[[str], None] | None = None`（沙盒设；出错只记日志）
  - `Body.mused(text: str) -> None`：身体线程里调；`parse_musing(text, cfg.lull.musing_max)`，非空且 `lulls.muse(...)` 为 True 时：`log.info("心里：%s", m)`、`mind_log.musing(...)`（有就记）、`on_musing(m)`
  - `MindLog.musing(t: float, who: list[str], kind: str, text: str) -> dict`：行 `{"t", "kind": "musing", "who", "lull": kind, "text"}`
  - `Body.inner_snapshot()` 多一项 `"musing": lulls.snapshot(wall)`（没有 lulls 时 `[]`）

接法：
- `step()`：`_watch_attention` 之后加 `self._watch_lulls(self.clock())`：`lulls.tick(self.wall(), env.nearby(now), list(self.chat), paused=self.blackout)`，每个 Cue → `events.put("lull", cue.text)`；`cue.final` 且有 `reflector` → `reflector.stirred(now)`；然后 `pop_finished()` 的每个 → `_lull_note(lull)`
- `_lull_note(lull)`：`mind` 不为 None 时把 `(lull.t0, "（冷场）", lulls.summary(lull, wall))` 按时间插进 `_reflect_chat` 和 `_session_chat`（`bisect.insort` 按第一个元素）
- `_heard`：放 `chat` 事件之前，对每条消息 `note = lulls.heard(self.wall(), m.speaker, m.text)`，非空就接在这条聊天事件文字后面（`聊天  懒洋洋大王：「刚去倒水」（冷场了 2 分钟，你刚才在想：…）`）；主人命令同样附
- `_watch_comings`：走开那段 `text = lulls.left(name, self.wall(), list(self.chat))`，有就 `events.put("lull", text, who=name)` 代替 `leave`（`_arrive_notes`、`_reflect_note`、`_left_at`、牵手推测照旧）；回来那段 `note = lulls.returned(name, self.wall())`，有就 `events.put("lull", f"{name} 回来了{note}", who=name)` 代替 `return`；`arrive` 那段有 note 就接在文字末尾。`_watch_comings` 只在不跟踪时跑，所以跟踪中自然不判情况②
- `status()`：「场合」那一项之后、「心里」之前加 `lulls.status(self.wall())`（非空才加）
- `reflect_materials(final=True)` 开头：`lulls.flush(self.wall())` 的每个 → `_lull_note`
- 所有 lulls 调用包在 `_inner_call` 风格的 try 里（出错只记日志，这一圈照常）

- [ ] **Step 1: 写失败的测试** `tests/test_brain_lull_body.py`：
  - `test_stage_event_is_urgent`：聊一句 → 推进 65 秒 `step()` → `events.drain()` 里有 kind `"lull"`；`"lull" not in BACKGROUND`
  - `test_chat_after_lull_has_note`：冷场开始后 `b.mused("不说：等\n心里：他是不是去忙了")`；好友说话 `step()` → 聊天事件文字以 `"（冷场了 1 分钟，你刚才在想：他是不是去忙了）"` 结尾
  - `test_mused_outside_lull_ignored`：没冷场时 `b.mused("心里：嗯")` → `on_musing` 没被调
  - `test_on_musing_and_mind_log`：冷场中 `mused` → `on_musing` 收到 `"他是不是去忙了"`；有 `mind_log`（`MindLog(None, False)`）时 `recent()[-1]["kind"] == "musing"`
  - `test_leave_becomes_lull`：好友刚说过话后从 `env.near` 去掉 → 事件 kind `"lull"`、不是 `"leave"`；没说过话的好友走开仍是 `"leave"`
  - `test_return_becomes_lull`：接上，他回来 → kind `"lull"`，文字以 `"懒洋洋大王 回来了（走开了"` 开头
  - `test_status_has_lull`：冷场中 `"冷场：懒洋洋大王" in b.status()`
  - `test_reflect_chat_gets_summary`：带 `mind` 的身体（照 `tests/test_brain_mind_body.py` 的 helper）冷场后好友开口 → `b._reflect_chat` 里有说话人 `"（冷场）"` 的一行
  - `test_final_stage_stirs_reflector`：假 reflector 记 `stirred` 调用；到 360 秒节点后被调过
  - `test_disabled_unchanged`：`cfg.lull.enabled = False` → `b.lulls is None`；同样的剧情 leave 照旧、status 不含“冷场”
  - `test_inner_snapshot_musing`：冷场中 `inner_snapshot()["musing"][0]["musings"][0]["text"] == "他是不是去忙了"`
  - `tests/test_inner_log.py::test_musing_row`：`MindLog(None, False).musing(1.0, ["懒洋洋大王"], "silent", "嗯")` == `{"t": 1.0, "kind": "musing", "who": ["懒洋洋大王"], "lull": "silent", "text": "嗯"}`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**（含全量；注意 `test_brain_body.py` 的 `body()` 默认没开 lull 时旧测试不受影响——`body()` helper 里加 `cfg.lull.enabled = False`，注释“旧行为的测试：不认冷场”，和 proactive 同样处理）
- [ ] **Step 5: 提交** `feat(brain): 身体认冷场、记心里想的`

### Task 4: 大脑每轮的文字交回身体

**Files:**
- Modify: `src/skydango/brain/loop.py`（`Brain.on_text`）；`src/skydango/cli.py`（`_run_brain` 里 `body.brain_turn = …` 旁边接上）
- Test: `tests/test_brain_loop.py`

**Interfaces:**
- Produces: `Brain.on_text: Callable[[str], None] | None = None`；`_wake` 成功后（`_ok()` 之后）有文字就调，出错 `log.exception` 不影响这一轮；`farewell` 不调
- cli：`brain.on_text = lambda text: body.call(lambda: body.mused(text), timeout=3)`（`body.lulls` 为 None 时不接）

- [ ] **Step 1: 写失败的测试**（`tests/test_brain_loop.py`，用文件里已有的假 session）：
  - `test_on_text_gets_final_text`：假 session 回 `{"result": "心里：嗯", …}` → `on_text` 收到 `"心里：嗯"`
  - `test_on_text_error_does_not_break_turn`：`on_text` 抛异常 → 这一轮照常结束（`failures == 0`）
  - `test_on_text_not_called_on_failure`：session 抛 `ClaudeError` → 没调
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（loop + cli 接线）
- [ ] **Step 4: 跑测试确认通过**（含全量）
- [ ] **Step 5: 提交** `feat(brain): 每轮最后的文字交回身体`

### Task 5: 提示词：「冷场的时候」+ 规则写清（修复 2）

**Files:**
- Modify: `src/skydango/brain/prompt.py`；`src/skydango/cli.py`（`brain_prompt(…, lull=cfg.lull.enabled)`）
- Test: `tests/test_brain_prompt.py`

**Interfaces:**
- Produces: `brain_prompt(..., lull: bool = False)`；常量 `LULL_RULES`、`LULL_POINTER`

`PROACTIVE_RULES` 的“分寸”一行改成（逐字）：
`- 分寸：状态里“上次主动开口”那一行写着没人接，就收一收，过一阵再说，别连着抛话题；你接了别人的话、对方没再说，不算主动开口没人接。一次只主动说一句；别用问句硬找话。`

`lull` 时把 `不算主动开口没人接。` 换成 `LULL_POINTER = "不算主动开口没人接，那是冷场，看下面「冷场的时候」。"`，并把 `LULL_RULES` 插在 `## 身份` 之前（proactive 开着时就在「主动开口」之后；proactive 关着时也插在 `## 身份` 之前）。`LULL_RULES`（逐字）：

```
## 冷场的时候
- 好友在身边不说话了、聊着聊着走开了，身体会用“冷场”叫醒你，写着冷了多久、最后一句是谁说的。
- 被冷场叫醒时，在最后的文字里写一行“心里：……”：这一刻真实的猜测和感受，贴着最后那句话、你们的交情、你的心情和精力来想（困了就懒得多想）。不用每次都往坏处想：“他应该是去忙了”“是不是我那句太黏人了”“算了我自己看会儿风景”都行。
- 想法跟着冷场往前走：刚开始不在意，过一会儿开始猜，再久了有点失落或者自己找乐子；状态里“在想”是你上次想的，接着往下想，别重复。
- 说不说、说什么照旧自己定；心里想的可以流露出来（“你还在吗”“是不是我说错话啦”、做个动作），别句句追问。主动开口的规矩照旧。
- 是好友说完你没接的那种冷场：是你自己没接，想的是要不要接，而不是“他怎么不理我”。
- 冷场结束时，聊天后面会附上“你刚才在想：…”，可以顺口接（“我还以为你不理我了”），也可以当没这回事。
```

- [ ] **Step 1: 写失败的测试**：
  - `test_lull_rules_added`：`brain_prompt(reply, None, lull=True)` 含 `"## 冷场的时候"` 和 `LULL_POINTER`，且 `"## 冷场的时候"` 在 `"## 主动开口"` 之后、`"## 身份"` 之前
  - `test_lull_off_no_rules`：`lull=False` 不含 `"冷场"`；含 `"不算主动开口没人接。"`
  - `test_lull_without_proactive`：`proactive=False, lull=True` 含 `"## 冷场的时候"`
  - 现有对 `PROACTIVE_RULES` 逐字比较的测试按新的“分寸”一行更新
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（prompt + cli 传参）
- [ ] **Step 4: 跑测试确认通过**（含全量）
- [ ] **Step 5: 提交** `feat(brain): 提示词加「冷场的时候」，写清“没人接”`

### Task 6: 接话不算主动开口（修复 1）

**Files:**
- Modify: `src/skydango/brain/body.py`（`say`、新方法 `_pending_reply`）
- Test: `tests/test_brain_body.py`（看场合主动开口那一段）

**Interfaces:**
- Produces: `Body._pending_reply(wall: float) -> bool`：`self.chat` 里最后一条好友（`is_friend_fn(self.friend_names())`）的行比最后一条 `"我"` 的行新，且 `wall - 它的时间 <= cfg.proactive.reply_window`
- `say`：`proactive = … and not self.brain_busy() and not self._pending_reply(self.wall())`

- [ ] **Step 1: 写失败的测试**（用 `pro_body`）：
  - `test_reply_to_pending_chat_not_proactive`：好友说“在吗”→ `step()`；`brain_busy` 为 False；`say("在呢")` → `spoken[-1].proactive is False`，且 `occasion().last is None`
  - `test_pending_expires`：好友说话后推进 91 秒 → `say` 算主动
  - `test_after_my_reply_back_to_proactive`：接过一句后再 `say` → 算主动（受 min_gap 等护栏）
  - `test_stranger_line_not_pending`：说话人 `""` 的行 → `say` 算主动
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**（含全量）
- [ ] **Step 5: 提交** `fix(brain): 接还没回的好友聊天不算主动开口`

### Task 7: 管理面板、沙盒、配置样例

**Files:**
- Modify: `src/skydango/console/settings.py`（`Field("lull.enabled", "冷场时的心理活动", "好友不说话了、聊着聊着走了时团子心里会犯嘀咕，按节点叫醒大脑；关掉就回到老样子", "bool", "brain")`，放在 `inner.persona` 之后）
- Modify: `config.example.toml`（`[proactive]` 之后加 `[lull]` 一节，键和默认值照 Global Constraints，每个键一行中文注释）
- Modify: `src/skydango/sandbox/control.py`（`__init__` 里 `self.body.on_musing = lambda m: self._event("心里：" + m)`）
- Modify: `src/skydango/console/inner_view.py`（`now` 部分多一项 `musing`：默认 `[]`，在跑时取实时快照的 `musing`）
- Modify: `src/skydango/console/static/inner.js`（`renderNow`：`now.musing` 非空时加一行 `row("在想", …)`，每个冷场一个 tag：`{who 用、连} · 冷了 {span(t - since)} · 「最后一条想法」`；`renderLog`：`musing` 行也列出，显示 `心里：{text}（{who}）`，“只看改了的”筛选下也显示）
- Test: `tests/test_console_settings.py`、`tests/test_sandbox_control.py`、`tests/test_console_inner.py`、`tests/test_console_page.py`（已有的 `node --check` 覆盖 inner.js 语法）

- [ ] **Step 1: 写失败的测试**：
  - settings：`"lull.enabled"` 在字段清单里、类型 bool
  - sandbox control：建好 control 后调 `body.on_musing("他忙去了吧")` → transcript 最后一行 `{"kind": "event", "text": "── 心里：他忙去了吧 ──"}`
  - console inner：`live_snap(musing=[{…}])` 在跑时 `s["now"]["musing"]` 原样带出；不在跑时是 `[]`
  - 配置样例：`tests` 里已有加载 `config.example.toml` 的测试就加断言 `cfg.lull.stages == [60.0, 180.0, 360.0]`（没有就跳过这一条）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**（含全量）；`node --check src/skydango/console/static/inner.js`
- [ ] **Step 5: 提交** `feat(console): 冷场开关、沙盒旁白、内心页“在想”`

### Task 8: 文档 + 收尾验证

**Files:**
- Modify: `CLAUDE.md`（「身体反射」之后、「内心层」之前加一节「冷场时的心理活动（`[lull]`，大脑模式）」：设计 / 计划路径、两种情况和节点、“心里：”怎么记、看得到的地方、不写的地方、修复 1 / 2、`enabled = false` 照旧、**还没在真机 / 真 Claude 上跑过，数字都是估的**（spec「用真 Claude 在沙盒验证」四步）；代码结构表 `src/skydango/inner/` 那一行加上 `lull.py` 冷场追踪）
- Modify: spec 状态行改成 `**代码已完成，还没用真 Claude 验证**`

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 全量测试**：`.venv\Scripts\python.exe -m pytest -q` → 全过（贴出最后一行）
- [ ] **Step 3: 提交** `docs: 冷场时的心理活动`
- [ ] **Step 4: 按 CLAUDE.md 合并进 main 并推送**（`finishing-a-development-branch`，本仓库规矩：合并进 main、推送）
- [ ] **Step 5: 告诉用户用真 Claude 在沙盒里跑 spec 的验证四步**（重启沙盒才会加载新代码）
