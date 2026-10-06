# 好友在不在场 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 好友走出画面不再 5 秒就算"走开"：贴边 / 喊到算"附近"，找不到满 15 秒且喊过一声也没亮出名字才算走开。

**Architecture:** 新的纯计算模块 `vision/presence.py`（`Presence`）只存时间戳、查询时现算状态；`PerceptionWatcher` 喂它（贴边、呼喊窗口结束、暂停顺延，"画面里看到"直接共用 `last_seen` 字典），`nearby()` 改成"在场"、新增 `in_view()`；身体的自动喊、注意力找人、`find`、status 跟着改。`[perception] presence = false` 逐字照旧。

**Tech Stack:** Python 3.13，pytest（合成画面 + 假设备）。

**Spec:** `docs/superpowers/specs/2026-10-06-friend-presence-design.md`

## Global Constraints

- 配置（`[perception]`）：`presence = true`、`leave_after = 15.0`、`recheck = 90.0`、`confirm_max = 60.0`；`keep`（5.0）、`edge_band`（0.06）默认值不变
- `presence = false`：行为、文字、status 逐字照旧；`EnvWatcher`（感知层没开）不受影响
- 能不能喊（`can_call`）= 大脑模式 且 `[call] enabled` 且 `[call] auto` 且不是 dry-run；不能喊时找不到满 `leave_after` 直接算走开
- 在场确认的那一声不发 `call` 背景事件；照旧受 `[call] min_gap`、`auto_quota` / `auto_window`、`_auto_call_blocked` 限制
- "拿不准是不是小明"那种自动喊（含它的 `auto_again`）不变
- 注释、日志、文字用中文，照周围代码的写法
- 测试：`python -m pytest -q`（本机 Windows 用 `.venv\Scripts\python.exe -m pytest -q`）

## Review Focus

- 好友在画面里时又贴边又有标签（交叉帧）：算身边，不能因为贴边降成附近 → Task 1 `test_view_beats_edge`
- 暂停（`held`）期间 `recheck` / `leave_after` 不走：恢复后不该立刻判走开 → Task 1 `test_shift_delays_everything`、Task 2 `test_presence_pauses_with_hold`
- 喊的窗口开始早于"找不到"起点（比如大脑刚好喊过、那时他还在画面里）不算喊过 → Task 1 `test_call_before_lost_does_not_count`
- 走开以后又亮出标签：重新在场，身体按 `rejoin` 发 return / arrive（不能卡在"走开"） → Task 1 `test_comes_back_after_left`
- dry-run：不按键也不能永远挂着不走开 → Task 1 `test_cannot_call_leaves_after_leave_after`、Task 3 `test_presence_dry_run_never_presses`

---

### Task 1: 配置 + `Presence` 纯计算

**Files:**
- Modify: `src/skydango/config.py`（`PerceptionConfig`，`edge_band` 那段后面）
- Create: `src/skydango/vision/presence.py`
- Test: `tests/test_presence.py`

**Interfaces:**
- Consumes: `PerceptionConfig`（`keep` / `presence` / `leave_after` / `recheck` / `confirm_max`）、`vision.people.Seen`（`side`、`on_screen`）
- Produces（Task 2、3 用）：
  - `PerceptionConfig.presence: bool = True`、`leave_after: float = 15.0`、`recheck: float = 90.0`、`confirm_max: float = 60.0`
  - `class Presence(cfg: PerceptionConfig, seen: dict[str, float], can_call: bool)` —— `seen` 是感知层的 `last_seen`（同一个字典对象，`Presence` 只读）
  - `edge(name: str, side: str, now: float) -> None`（side：左边 / 右边 / 上边 / 下边）
  - `call_done(at: float, found: dict[str, Seen]) -> None`
  - `shift(d: float, now: float) -> None`
  - `state(name: str, now: float) -> str`：`"view"` / `"near"` / `"lost"` / `"left"` / `""`（没见过）
  - `present(names: list[str], now) -> list[str]`（view + near，按 names 的顺序）、`in_view(names, now) -> list[str]`、`need_call(names, now) -> list[str]`（lost 且没喊过）
  - `around(names, now) -> list[tuple[str, str]]`：附近的人 + 说明（`"画面外·右边"` / `"远处，刚喊到"` / `"画面外·上边，2 分钟前喊到"`）
  - `confirmed(name: str) -> float | None`、`left_note(name: str, now) -> str`

规则（照 spec §1.2，查询时现算，不存状态机）：每人存 `last_edge`（时间, side）、`confirmed`（时间, 说明）、`tried`（最近一次没亮出他的呼喊窗口开始时间）。
- view：`now - seen[name] ≤ keep`；near：不是 view，且贴边在 `keep` 内或喊到在 `recheck` 内
- lost 起点 `lost_at = max(seen, last_edge, confirmed + recheck)`（没有的项当 −∞）
- left：lost 且（`tried ≥ lost_at` 且 `now − lost_at ≥ leave_after`）或 `now − lost_at ≥ confirm_max` 或（`not can_call` 且 `now − lost_at ≥ leave_after`）
- `call_done`：`found` 里的人 → `confirmed = (at, "画面外·<side>" 如果 not on_screen 否则 "远处")`；记录里其余的人 → `tried = at`
- `left_note`：`"喊了一声也没看到，15 秒了"` / `"60 秒没看到，没喊成"` / `"15 秒没看到"`（秒数 = `now − lost_at`，`:.0f`）
- `around` 说明：贴边在 `keep` 内用贴边那条（`画面外·右边`），否则喊到那条 + `，刚喊到`（< 60 秒）/ `，N 分钟前喊到`

- [ ] **Step 1: 写失败的测试** `tests/test_presence.py`（`P = PerceptionConfig()`，`seen = {}`，`pr = Presence(P, seen, can_call=True)`）：
  - `test_in_view_then_near_by_edge`：`seen["小明"]=0`；`state(…,3)=="view"`；`edge("小明","右边",4)`；`state(…,8)=="near"`；`around(["小明"],8)==[("小明","画面外·右边")]`；`present(["小明"],8)==["小明"]`、`in_view(["小明"],8)==[]`
  - `test_view_beats_edge`：同一时刻既 seen 又 edge → `"view"`
  - `test_lost_needs_call_then_left_after_failed_call`：`seen=0`；t=6 `state=="lost"`、`need_call==["小明"]`；`call_done(6, {})`；t=14 还是 `"lost"`（不到 15）；t=15 `"left"`、`left_note=="喊了一声也没看到，15 秒了"`、`need_call==[]`
  - `test_no_call_waits_confirm_max`：`seen=0`、从不 call_done；t=30 `"lost"`；t=60 `"left"`、`left_note=="60 秒没看到，没喊成"`
  - `test_cannot_call_leaves_after_leave_after`：`can_call=False`；t=15 `"left"`、`need_call==[]`、`left_note=="15 秒没看到"`
  - `test_call_found_confirms_for_recheck`：`seen=0`；`call_done(6, {"小明": Seen("右边", None, on_screen=False)})`；t=95 `"near"`、`around==[("小明","画面外·右边，1 分钟前喊到")]`；t=97 `"lost"`、`need_call==["小明"]`；lost 起点 = 96 → `call_done(97, {})` 后 t=110 `"lost"`、t=111 `"left"`
  - `test_found_on_screen_is_far`：`call_done(6, {"小明": Seen("左边", "远")})` → t=10 `around==[("小明","远处，刚喊到")]`
  - `test_call_before_lost_does_not_count`：`seen=0`；`seen=5`、`call_done(3, {})`→ t=20 仍 `"lost"`（tried 3 < lost_at 5）、`need_call==["小明"]`
  - `test_shift_delays_everything`：`seen=0`、`call_done(6, {"小明": Seen("右边", None, on_screen=False)})`；`shift(100, 150)`（confirmed → 106）；t=190 `"near"`
  - `test_comes_back_after_left`：走到 `"left"` 之后 `seen["小明"]=40` → t=41 `"view"`
  - `test_unknown_name_is_empty`：`state("阿花", 0) == ""`、`present(["阿花"], 0) == []`
- [ ] **Step 2: 跑测试确认失败** `python -m pytest tests/test_presence.py -q` → ImportError
- [ ] **Step 3: 加配置四项**（`config.py`，注释写清意思，数字标"估的"），**实现 `vision/presence.py`**（模块 docstring 指向 spec）
- [ ] **Step 4: 跑测试确认通过** `python -m pytest tests/test_presence.py tests/test_config.py -q` → PASS
- [ ] **Step 5: 提交** `feat(perception): 好友在不在场的纯计算 Presence（身边 / 附近 / 找不到 / 走开）`

### Task 2: 感知层接上 Presence，贴边四条边

**Files:**
- Modify: `src/skydango/vision/perception.py`（构造、`_offscreen`、标签循环的贴边分支 ≈810、`_call_tick` 收尾 ≈1170、`_resume` ≈528、走近判断 ≈2033、`nearby` ≈2635、`describe` ≈2657）
- Modify: `src/skydango/vision/env.py`（`EnvWatcher.in_view = nearby`）、`src/skydango/cli.py`（`_scene_watcher` 传 `presence_call`、启动日志一行）
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 1 的 `Presence`
- Produces（Task 3 用，都按 `_frozen(now)` 算）：
  - `PerceptionWatcher(..., presence_call: bool = False)`；属性 `presence: Presence | None`（`cfg.presence` 关时 None）
  - `nearby(now)`：presence 开 = `presence.present(self.names(), now)`，关 = 原来
  - `in_view(now) -> list[str]`：presence 开 = `presence.in_view(...)`，关 = `nearby(now)`
  - `around(now) -> list[tuple[str, str]]`、`need_call(now) -> list[str]`、`left_note(name, now) -> str`（presence 关时 `[]` / `[]` / `""`）
  - `_offscreen(tag, width, height) -> str | None`：左右（按屏宽 × `edge_band`）优先，再上下（按屏高 × `edge_band`），返回 左边 / 右边 / 上边 / 下边
  - `EnvWatcher.in_view(now)` = `nearby(now)`

- [ ] **Step 1: 写失败的测试**（`tests/test_perception.py`；老测试里断言"贴边不在身边"的 `test_edge_tag_not_nearby_but_in_call_result` 改成传 `presence=False`，另写新的）：
  - `test_edge_tag_top_and_bottom`：标签在 y 中心 < 0.06×1080、下面没人 → `w.labels` 有他、`w.in_view(t)==[]`、`w.nearby(t)==["懒洋洋大王"]`、`w.around(t)==[("懒洋洋大王","画面外·上边")]`；底边同理"下边"
  - `test_edge_tag_counts_as_present`：右边贴边（同老测试的画面）→ `nearby==["懒洋洋大王"]`、`in_view==[]`
  - `test_presence_off_is_old_behavior`：`presence=False` 时贴边 `nearby==[]`、`in_view==nearby`、`need_call==[]`
  - `test_call_window_feeds_presence`：名字在画面里 t=0.1 → 空帧到 t=6 → `need_call(6)==["懒洋洋大王"]`（`presence_call=True`）；`called(6)`，窗口里空帧，t=12.1 收尾 → `need_call==[]`；t=15.2 `nearby==[]`、`left_note` 以"喊了一声也没看到"开头
  - `test_presence_pauses_with_hold`：t=0 看到，`hold("x")` 在 t=1，`clock.t=100` 时 `release` → `nearby(101)==["懒洋洋大王"]`（暂停的 99 秒不算）
  - `test_presence_call_false_by_default`：默认构造 `presence.can_call is False`
- [ ] **Step 2: 跑测试确认失败** `python -m pytest tests/test_perception.py -q -k "edge or presence or call_window_feeds"`
- [ ] **Step 3: 实现**：构造时 `self.presence = Presence(cfg, self.last_seen, presence_call) if cfg.presence else None`；贴边分支调 `presence.edge`；`_call_tick` 收尾 `presence.call_done(c.at, dict(c.friends))`；`_resume` 里 `presence.shift(d, now)`；两处 `_offscreen` 调用补 `height`；走近判断（≈2033）的 `maybe in self.nearby(now)` 改成 `now - self.last_seen.get(maybe, -inf) <= self.cfg.keep`（按外观认的还是只认画面里的）；`describe` 的"你身边现在有"用 `in_view`，有 `around` 时多一行 `- 附近（画面里看不到）：…`；cli `presence_call=light and cfg.call.enabled and cfg.call.auto and not cfg.reply.dry_run`，`cfg.perception.presence` 开时 INFO「好友在不在场：走出画面不算走开，找不到满 N 秒（喊过一声）才算」
- [ ] **Step 4: 跑测试** `python -m pytest tests/test_perception.py tests/test_env.py -q` → PASS（老测试因默认开 presence 挂掉的，确认是语义变化后改成显式 `presence=False`，不改断言）
- [ ] **Step 5: 提交** `feat(perception): nearby 改成在场（贴边四条边 / 喊到算附近），新增 in_view`

### Task 3: 身体：自动喊、走开文字、找人、status

**Files:**
- Modify: `src/skydango/brain/body.py`（`_watch_comings` ≈1201、`_start_lost_search` ≈817、`_search_call` ≈831、`_watch_call` ≈2541、`_collect_auto_call` ≈2613、`find` ≈2698、status ≈1938）
- Modify: `src/skydango/vision/viewer.py`（≈404）
- Test: Create `tests/test_brain_presence.py`（复用 `test_brain_call_body.call_body` / `CallEnv`）

**Interfaces:**
- Consumes: Task 2 的 `env.presence`（有没有）、`env.in_view` / `need_call` / `around` / `left_note`；身体用 `getattr` 判断，老的 `FakeEnv` 没有这些 = 老行为
- Produces：`Body._out_at: dict[str, float]`（离开画面的时间）、`Body._in_view: set[str]`；`call_out` 新 reason `"presence"`

- [ ] **Step 1: 写失败的测试** `tests/test_brain_presence.py`（`PresenceEnv(CallEnv)`：`presence = object()`，`view` / `lost` / `around_list` / `notes` 字段，`in_view` / `need_call` / `around` / `left_note` 返回它们）：
  - `test_presence_call_presses_once_for_all_lost`：`lost=["小明","阿花"]`、`unnamed_n=0` → `_watch_call` 按一次 Q、`last_call.reason=="presence"`；窗口结果到了后 `events` 里没有 `call` 事件
  - `test_presence_call_respects_min_gap_and_blocks`：刚喊过（`_call_at = now-5`）不按；`sender.opened` 为真不按
  - `test_presence_dry_run_never_presses`：`live=False` → `need_call` 有人也不按键
  - `test_leave_text_from_presence`：`near` 从 `["小明"]` 变 `[]`、`notes={"小明":"喊了一声也没看到，15 秒了"}` → `leave` 事件文字 `"小明 走开了（喊了一声也没看到，15 秒了）"`
  - `test_out_of_view_starts_lost_search_without_leave`：`near=["小明"]` 不变、`view` 从 `["小明"]` 变 `[]` → 没有 `leave` 事件、`_out_at["小明"]==now`；注意力开着时 `_start_lost_search` 用它发起找人（`attention.search.who=="小明"`）
  - `test_status_shows_view_and_around`：`near=["小明","阿花"]`、`view=["小明"]`、`around_list=[("阿花","画面外·右边")]` → status 有 `"身边的好友：小明"` 和 `"附近：阿花（画面外·右边）"`
  - `test_old_env_unchanged`：`CallEnv`（没 presence）时 `test_auto_call_after_friend_leaves_with_unnamed_people` 的流程照旧（直接跑现有 `tests/test_brain_call_body.py` 全绿即可，不另写）
- [ ] **Step 2: 跑测试确认失败** `python -m pytest tests/test_brain_presence.py -q`
- [ ] **Step 3: 实现**：
  - `_watch_comings`：`view = set(getattr(self.env, "in_view", self.env.nearby)(now))`，掉出去的记 `_out_at[n] = now`，存 `_in_view`；`leave` 文字有 `left_note` 且非空时用 `f"{name} 走开了（{note}）"`，否则原文字
  - `_start_lost_search` / `find`：`_left_at` → `_out_at`、`n not in self._nearby` → `n not in self._in_view`（窗口照旧 `auto_after_leave`；presence 关时两个时间同一刻，行为不变）；`_search_call` 里 `_auto_called` 也按 `_out_at`；presence 开时"刚喊回来过他"改成 `env.presence.confirmed(who)` 在 `recheck` 内 → `why = "刚喊到过他"`
  - `_watch_call`：presence 开时起因 ① 换成 `who = self.env.need_call(now)`（不看 `unnamed`、`auto_after_leave`、`auto_again`、`_auto_called`，日志「看不到 小明、阿花 了：喊一声看看还在不在」，`call_out("presence")`）；起因 ② 不变；`_search_takes_call()` 不再挡 presence 这一声
  - `_collect_auto_call`：`r.reason == "presence"` 时不放 `call` 事件（其余照旧）
  - status：presence 开时"身边的好友"用 `in_view`（ledger 的 `status_line` 也传 `in_view`），`around` 非空时多一行 `"附近：" + "、".join(f"{n}（{why}）")`；viewer `/status` 同样 `身边的好友` 用 `in_view`、多 `附近`
- [ ] **Step 4: 跑测试** `python -m pytest tests/test_brain_presence.py tests/test_brain_call_body.py tests/test_brain_call_again.py tests/test_brain_body.py -q` → PASS
- [ ] **Step 5: 提交** `feat(brain): 好友走出画面不算走开——自动喊确认在场、status 分身边 / 附近`

### Task 4: 设置页、文档、全量测试

**Files:**
- Modify: `src/skydango/console/settings.py`（`perception.enabled` 后面加三项；`call.auto` 的说明改）
- Modify: `CLAUDE.md`（「按 Q 喊一声」「YOLO 感知层」两节）、`docs/game-ops.md`（「呼喊找好友」：远近两种贴边、四条边）、spec（`auto_after_leave` 仍是"刚离开画面"的窗口，§3.3 那句改掉 `OUT_RECENT`）
- Test: `tests/test_console*.py`（设置清单的现有测试）

- [ ] **Step 1: 加设置项** `Field("perception.presence", "走出画面不算走开", "好友贴边 / 按 Q 能喊到就算还在，找不到满 15 秒且喊过一声才算走开", "bool", "features")`、`perception.leave_after`（"找不到多久算走开（秒）"，float）、`perception.recheck`（"喊到一次管多久（秒）"，"好友在远处时大约这么久喊一声确认，60~120"，float），组照 `perception.enabled`；`call.auto` 说明改成"好友看不到了 / 拿不准是谁时，身体自己喊一声确认（1 分钟最多 3 次）"
- [ ] **Step 2: 改文档**（照现有写法：代码完成、还没在真机验证，真机验证见 spec §6）
- [ ] **Step 3: 全量测试** `python -m pytest -q` → 全部 PASS
- [ ] **Step 4: 提交** `docs: 好友在不在场——设置页、CLAUDE.md、game-ops`
