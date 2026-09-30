# 空闲注意力（东张西望）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 团子在面板 auto 模式、闲着的时候，身体按兴趣模型小步转镜头看说话 / 走近 / 对它做事的人，闲着随意看看；大脑用 `attention(mode, focus)` 定注意力模式。

**Architecture:** 纯决策在新模块 `brain/attention.py`（候选目标 → 看腻 / 实际兴趣 / 防抖 → 下一下按什么），`Body` 在 `_watch_idle` 之后判断闲不闲、收集候选、按键；感知层补 `talkers()` / `recent_approaches()`，面板补 `quiet()` / `hold_off()` / `pending`，内心层 `Effects` 补 `wander` 倍数。

**Tech Stack:** Python 3.13、pytest（`python -m pytest -q`，合成画面 + 假设备）。

**Spec:** `docs/superpowers/specs/2026-09-30-idle-attention-design.md`

## Global Constraints

- 用中文写注释、日志、提示词、status；代码风格照周围（`brain/peek.py`、`brain/track.py`、`brain/reflex.py`）
- 只在 `self.panel.auto` 时按键；always 模式、沙盒（camera / panels 为 None、env 没有 `people`）什么都不按
- 注意力按键 **不借面板、不走 `clear_view`、不调 `reflexes.stir()`、不套 `env.held()`**；只用 `Camera.nudge`（左右，0.02～0.1 s）
- 不回位：注意力从不调 `camera.reset`
- 模式名：`随意`（默认）/ `好奇` / `专心` / `别动`；工具名 `attention(mode, focus)`；status 写"注意力"，不叫"心思"
- `[attention]` 默认值照 spec §4 表，**一处偏离**：`bubble_gap = 1.0` 放在 `[perception]`（新气泡的判定在感知层每帧做，见 Task 2），`[attention]` 里没有它
- 按键长短、`settle`、转不动（`stall_nudges` / `stall_px`）沿用 `[track]`；"离上个动作多久"沿用 `[reflex] min_gap`
- `attention` 工具 **不进** `tools.ACTIONS`
- 时间一律 `self.clock()`；注意力内部每圈时间差夹到 `max_step`（1 s）
- 每个任务结束时 `python -m pytest -q` 全绿；在分支 `feat/idle-attention` 上提交，全部完成、评审后合并进 main 并推送（CLAUDE.md 规矩）

## Review Focus

1. **按键那一刻面板已经开了**（同一圈里读到新消息进了 chatting、大脑要说话）→ 不按。门槛判断要在按键前用当圈最新的面板状态（Task 5 `test_no_press_when_panel_opened_this_round`）
2. **感知层晚一轮**（后台线程）：按完一下后立刻看到的还是按之前的位置 → 只在 `settle` 之后才用新位置决定下一下、算转不动（Task 3 `test_no_second_press_before_settle`、`test_stall_uses_positions_after_settle`）
3. **模拟时钟一下跳几小时**：看腻 / 随意看计时不能一步到顶或连按（Task 3 `test_time_jump_is_clamped`）
4. **目标在按键后离开画面**：不再朝原方向按，当前目标清掉（Task 3 `test_target_gone_stops_turning`）
5. **大脑 / track / look_person 转过镜头之后**：注意力的转不动计数不能把别人的按键当成自己的进展 → 两次自己按键隔了超过 `3 × settle` 就重新计数（Task 3 `test_stall_streak_resets_after_long_gap`）

---

## File Structure

| 文件 | 变化 | 职责 |
|---|---|---|
| `src/skydango/chat/panel.py` | 改 | `PanelManager.quiet(now)`、`pending`、`hold_off(until)` |
| `src/skydango/vision/perception.py` | 改 | `Talker`、`talkers(now)`、`recent_approaches(now, within)`；`_watch_typing` 记气泡开始时间 |
| `src/skydango/config.py`、`config.example.toml` | 改 | `PerceptionConfig.bubble_gap`、`AttentionConfig`、`Config.attention` |
| `src/skydango/inner/effects.py` | 改 | `Effects.wander` + 常量表 |
| `src/skydango/brain/attention.py` | 新 | 纯决策 `Attention` |
| `src/skydango/brain/body.py` | 改 | `_watch_attention`、门槛、候选收集、approach 丢弃、gesture 另记、status / `_show`、`set_attention` |
| `src/skydango/brain/tools.py`、`mcp_server.py`、`prompt.py` | 改 | `attention` 工具和提示词 |
| `CLAUDE.md` | 改 | 「统管大脑」里加一段注意力 |
| 测试 | 新 / 改 | `tests/test_panel.py`、`tests/test_perception.py`、`tests/test_inner_effects.py`、`tests/test_brain_attention.py`（新）、`tests/test_brain_attention_body.py`（新）、`tests/test_brain_tools.py`、`tests/test_brain_prompt.py` |

---

### Task 1: 面板的"安静"和"先看一眼"推迟

**Files:**
- Modify: `src/skydango/chat/panel.py`（`trigger` / `bubble_seen` 附近 100-116、`describe` 147-158、`_tick_idle` 205-215、`_open_peek`）
- Test: `tests/test_panel.py`

**Interfaces:**
- Produces:
  - `PanelManager.pending -> str | None`（只读属性，就是 `_pending`）
  - `PanelManager.quiet(now: float, margin: float) -> bool`：auto、state == "idle"、`lent is None`、`_pending is None`、`self.cfg.idle_peek - (now - self._last_read) >= margin`；非 auto 返回 False
  - `PanelManager.hold_off(until: float | None, now: float) -> bool`：`until` 为数值时推迟 idle 状态下的开面板（定时和 pending 都不开）到 `until`；同一个 `_pending` 只接受一次（返回 False = 拒绝）；`None` 立刻放手。`_open_peek` 消费 pending 时清掉推迟和"已推迟过"标记
  - `describe()` 的"N 秒后看一眼"改用和 `quiet` 同一个私有算式 `_until_peek(now) -> float`

- [ ] **Step 1: 写失败的测试**（照 `tests/test_panel.py` 现有的 auto 夹具）

```python
def test_quiet_only_when_idle_nothing_pending_and_peek_not_due(...):
    # auto、idle、没 pending、离定时看一眼 > margin → True
    assert pm.quiet(now, margin=1.6)
    pm.trigger("approach", now); assert not pm.quiet(now, 1.6)          # 有 pending
    # 快到定时看一眼（_last_read 在 idle_peek - 1 秒前）→ False
    # 借出中 → False；state 为 peek / bubble / chatting → False；always 模式 → False

def test_hold_off_delays_pending_peek_until_released(...):
    pm.bubble_seen(now)
    assert pm.pending == "bubble"
    assert pm.hold_off(now + 2.0, now) is True
    pm.tick(now + 0.5, [], visible=False, blackout=False); assert 46 not in pressed_keys   # 没开
    pm.hold_off(None, now + 0.6)
    pm.tick(now + 0.7, [], visible=False, blackout=False); assert pm.state == "bubble"   # 放手后照常开

def test_hold_off_expires_on_its_own(...):      # 到了 until 不用放手也开
def test_hold_off_only_once_per_pending(...):   # 第二次 hold_off 返回 False
def test_hold_off_does_not_block_chatting(...): # 推迟中 tick 带 fresh 消息 → chatting；before_speak() 照常
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_panel.py -k "quiet or hold_off"`
Expected: FAIL（`AttributeError: 'PanelManager' object has no attribute 'quiet'`）

- [ ] **Step 3: 实现上面 Interfaces 里的三个成员**；`_tick_idle` 在 `blackout or not (due or wanted)` 判断之前加：推迟中（`self._hold_until is not None and now < self._hold_until`）直接 return

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_panel.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/chat/panel.py tests/test_panel.py
git commit -m "feat(panel): quiet / pending / hold_off，给空闲注意力先看一眼再开面板用"
```

---

### Task 2: 感知层告诉外面谁在说话、谁走近了

**Files:**
- Modify: `src/skydango/vision/perception.py`（`_watch_typing` 732-747、`_watch_approach` 749-764、`pop_approaches` 附近、`__init__` 的 171-173）
- Modify: `src/skydango/config.py`（`PerceptionConfig` 加 `bubble_gap: float = 1.0`，注释：同一个人的气泡消失超过这么久再出现算新的一句）、`config.example.toml` 的 `[perception]`
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces（`perception.py` 模块级）：

```python
@dataclass(frozen=True)
class Talker:
    track_id: int
    name: str | None      # 好友名；陌生人 None
    friend: bool
    x: float              # 人物框中心 x（整图像素）
    last: float           # 最近一次看到气泡
    start: float          # 这次气泡开始（新的一句）
```

  - `PerceptionWatcher.talkers(now: float) -> list[Talker]`：`last_tracks` 里 `now - data["typing_at"] <= 1.0` 的人（不含团子自己，`_watch_typing` 本来就排除了）
  - `PerceptionWatcher.recent_approaches(now: float, within: float = 5.0) -> list[tuple[str, float, float]]`：`(who, 框中心 x, 时间)`，`who` 同 `pop_approaches`（好友名 / `STRANGER`）；`pop_approaches()` 不变
  - `_watch_typing`：`now - owner.data.get("typing_at", -inf) > cfg.bubble_gap` 时 `owner.data["bubble_start"] = now`，再写 `typing_at`

- [ ] **Step 1: 写失败的测试**（照 `tests/test_perception.py` 里喂 typing / 走近的现有写法）

```python
def test_talkers_reports_friend_with_bubble_and_position(...):
    t = w.talkers(now)
    assert [(x.name, x.friend) for x in t] == [("小明", True)] and abs(t[0].x - 960) < 5

def test_talker_bubble_start_resets_after_gap(...):
    # 气泡持续出现：start 不变；消失 1.5 s 再出现：start 变成新的时间
def test_talker_continuous_bubble_longer_than_typing_window_keeps_start(...):  # 连续打 12 秒，start 不漂
def test_talkers_excludes_stale(...):          # 超过 1 秒没看到气泡 → 不在列表
def test_recent_approaches_keep_position_and_pop_still_works(...):
    assert w.recent_approaches(now) == [("小明", pytest.approx(cx), now)]
    assert w.pop_approaches() == ["小明"] and w.recent_approaches(now)   # pop 不影响 recent
def test_recent_approaches_include_strangers(...):
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_perception.py -k "talker or recent_approaches"`
Expected: FAIL（没有 `talkers`）

- [ ] **Step 3: 实现**：走近另存 `self._approach_log: deque[(who, cx, t)]`（在 `_lock` 里追加，保留最近 30 s）；`cx` 取 `hist[-1][2]`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_perception.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/perception.py src/skydango/config.py config.example.toml tests/test_perception.py
git commit -m "feat(perception): talkers() 谁在说话（带新气泡开始时间）、recent_approaches() 带位置"
```

---

### Task 3: 注意力的纯决策——目标、看腻、转过去、防抖

**Files:**
- Create: `src/skydango/brain/attention.py`
- Modify: `src/skydango/config.py`（`AttentionConfig` + `Config.attention`）、`config.example.toml`（`[attention]` 节，照 spec §4 表，不含 `bubble_gap`）
- Test: `tests/test_brain_attention.py`（新）

**Interfaces:**
- Consumes: `brain.peek.Turn(direction, seconds)`；`config.TrackConfig`
- Produces:

```python
KINDS = ("talk_friend", "act_on_me", "approach", "talk_stranger", "friend_present")
KIND_NOTES = {"talk_friend": "在说话", "act_on_me": "在对你做动作", "approach": "朝你走过来",
              "talk_stranger": "在说话", "friend_present": "站在那"}
MODES = ("随意", "好奇", "专心", "别动")

@dataclass(frozen=True)
class Target:
    key: str               # "t:<轨迹 id>" 或 "n:<名字>"：同一个人
    kind: str              # KINDS 之一
    x: float               # 框中心 x（整图像素）
    who: str | None        # 名字；陌生人 None
    fresh: float | None = None  # 这次刺激开始的时间（气泡 start / 走近时间 / 动作时间）；变了就把看腻清零

@dataclass(frozen=True)
class Thought:
    current: Target | None     # 现在在看的
    action: Turn | None        # 这一圈想按的（门槛不开时 Body 不按）
    centered: bool             # current 已经在中间带里
    wandering: bool            # 在随意看
    look_first: bool           # current 是刚出现的说话 / 走近 / 做动作（fresh 在 cfg.look_first 内），值得先看再开面板

class Attention:
    def __init__(self, cfg: AttentionConfig, track: TrackConfig, rng: random.Random, now: float): ...
    width: int                 # Body 每圈更新（默认 1920）
    mode: str                  # 默认 "随意"
    focus: str | None
    def think(self, targets: list[Target], now: float, wander_scale: float = 1.0) -> Thought: ...
    def pressed(self, turn: Turn, now: float) -> None: ...   # Body 真按了才调：记按键时间、转不动计数、随意看进度
    def describe(self) -> str: ...                           # "在看：小明（在说话）" / "闲着随意看" / "没在看什么"
```

  - 同一个 `key` 有几种 kind：只留基础兴趣最高的那种（`think` 开头合并）
  - 实际兴趣 = 基础 ×（1 − bored）；在中间带（`|x − width/2| ≤ center_band × width / 2`）时 `bored += bore_rate × dt`，不在时 `-= recover_rate × dt`，夹到 [0, 1]；`dt = min(now − 上一圈, max_step)`
  - `fresh` 和上次记的不同 → 这个 key 的 bored 清零
  - 当前目标：实际兴趣最高且 ≥ `min_interest`；换目标要新目标比当前高出 `switch_margin`，且距上次换 ≥ `switch_hold`；当前目标这一圈不在候选里 → 清掉
  - `action`：current 不在中间带、且距上次按键 ≥ `track.settle` 时，`Turn("right" if x < width/2 else "left", clamp(gain × |x − width/2| / (width/2), nudge_min, nudge_max))`（按右远处往左，同 peek）
  - 转不动：`pressed` 时记下当时的偏差；同方向连续 `track.stall_nudges` 次偏差缩小 < `track.stall_px` → current 的 bored += `stuck_bored`、计数清零；两次按键隔了 > `3 × track.settle` 计数也清零
  - `pressed` 之前 `think` 不再给同一个 action 第二次（`settle` 没过就 `action=None`）
  - 本任务 `mode` 只实现 `随意`，随意看返回 `wandering=False, action=None`（Task 4 做）

- [ ] **Step 1: 写失败的测试**（`FakeClock` 就用数字 `now`；`W = 1920`）

```python
def tgt(key, kind, x, who=None, fresh=None): return Target(key, kind, x, who, fresh)

def test_friend_talking_beats_stranger_talking():
    a = attn(); th = a.think([tgt("t:1", "talk_stranger", 300), tgt("t:2", "talk_friend", 1600, "小明", 0.0)], 0.0)
    assert th.current.key == "t:2" and th.action.direction == "left"  # 目标在右边：按左（远处往右移）

def test_centered_target_needs_no_turn():            # x = 1000 在中间 40% 内 → action None, centered True
def test_turn_seconds_scale_with_offset_and_clamp():  # 偏差大 → 0.1；偏差小 → ≥ 0.02
def test_no_second_press_before_settle():            # think → pressed(t=0) → think(t=0.3) action None；think(t=0.7) 有 action
def test_bored_of_centered_target_then_switch():     # 两个目标，当前在中间带，~5 s 后换到另一个
def test_new_bubble_resets_boredom():                # fresh 从 0.0 变 6.0 → 又回到这个人
def test_switch_margin_and_hold_prevent_flip_flop(): # 两个 talk_friend 同分：10 s 内换目标 ≤ 2 次
def test_stall_adds_boredom_and_moves_on():          # 同方向 3 下偏差不变 → current 换走
def test_stall_uses_positions_after_settle():        # pressed 后 settle 内的位置不参与转不动判断
def test_stall_streak_resets_after_long_gap():       # 两下之间隔 3 s → 不算连续
def test_target_gone_stops_turning():                # 候选里没有它了 → current None、action None
def test_time_jump_is_clamped():                     # now 从 0 跳到 36000：bored 只涨 max_step 份，不连按
def test_same_person_multiple_kinds_keeps_highest(): # friend_present + talk_friend 同 key → kind == talk_friend
def test_look_first_only_for_fresh_talk_or_approach():  # fresh 在 look_first 内 → True；friend_present / 旧的 → False
def test_below_min_interest_is_not_a_target():
def test_describe_lines():                           # "在看：小明（在说话）"、"在看：陌生人（朝你走过来）"、"没在看什么"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_attention.py`
Expected: FAIL（`ModuleNotFoundError: skydango.brain.attention`）

- [ ] **Step 3: 实现 `AttentionConfig`（spec §4 表的字段和默认值，去掉 `bubble_gap`）和 `Attention`**

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_brain_attention.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/attention.py src/skydango/config.py config.example.toml tests/test_brain_attention.py
git commit -m "feat(attention): 纯决策——候选目标、看腻、比例转过去、转不动、防抖"
```

---

### Task 4: 随意看、四种模式、focus、心情精力倍数

**Files:**
- Modify: `src/skydango/brain/attention.py`
- Modify: `src/skydango/inner/effects.py`（`Effects` 末尾加 `wander: float = 1.0`；`_MOOD` / `_ENERGY` 元组加第 5 项；`effects()` 里 `m[4] * e[4]`）
- Test: `tests/test_brain_attention.py`、`tests/test_inner_effects.py`

**Interfaces:**
- Produces:
  - `Attention.set_mode(mode: str, focus: str | None) -> None`：`mode` 不在 `MODES` 抛 `ValueError`；`focus=""` 或 None 清掉
  - `Effects.wander` 常量：开心 0.8、平常 1.0、低落 1.5、烦 1.2；精神 1.0、还行 1.0、有点累 1.3、困 2.0（心情 × 精力）
  - 随意看：没有 ≥ `min_interest` 的目标、模式允许时，距上一次随意看结束 ≥ 随机 `U(wander_min, wander_max) × wander_scale × 模式倍数`（好奇 0.5）→ 开始一次：方向随机（同一边连着不超过 `wander_same_side` 次，超了就换边）、次数 `randint(*wander_presses)`；每次 `think` 在 settle 过后给一个 `Turn(方向, track.nudge_max)`，`wandering=True`；`pressed` 计数，满了结束并抽下一次的间隔；有目标出现立刻打断
  - 模式：`好奇`：`talk_stranger` 基础兴趣按 0.7；`专心`：不随意看，基础兴趣 < `act_on_me` 的目标忽略（focus 的人除外）；`别动`：`action=None`、`look_first=False`（仍然算 current，describe 照常）
  - focus：`who` 和 focus 一模一样或 `chat.tracker.similar(focus, who, 0.75)` → 基础兴趣取 `max(原来, focus_interest)`、看腻速度减半
  - `describe()`：模式不是随意或有 focus 时前缀 `注意力：专心，关注小明；`

- [ ] **Step 1: 写失败的测试**

```python
def test_wander_after_random_quiet_interval():      # rng 固定；间隔落在 [8, 20]；按 2~4 下后结束，wandering True→False
def test_wander_never_same_side_more_than_limit():  # 连着 10 次随意看，同一边连续 ≤ 2
def test_target_interrupts_wander():
def test_wander_scale_and_curious_mode_shorten_interval():   # wander_scale=2.0 间隔翻倍；好奇减半
def test_focused_mode_ignores_low_interest_and_no_wander():  # 只有 friend_present → 不转、不随意看
def test_still_mode_never_moves_nor_looks_first():
def test_curious_raises_stranger_talk():
def test_focus_boosts_person_and_slows_boredom():    # focus "小明" 的 friend_present 压过陌生人 approach；OCR 错一个字也认
def test_set_mode_rejects_unknown():
def test_describe_with_mode_and_focus():             # "注意力：专心，关注小明；在看：小明（站在那）"
```

`tests/test_inner_effects.py`：

```python
def test_wander_multiplier():
    assert effects("开心", "精神").wander == pytest.approx(0.8)
    assert effects("平常", "困").wander == pytest.approx(2.0)
    assert effects("低落", "有点累").wander == pytest.approx(1.95)
    assert effects("奇怪", "奇怪").wander == 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_attention.py tests/test_inner_effects.py`
Expected: FAIL（`set_mode` / `wander` 不存在）

- [ ] **Step 3: 实现**

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_brain_attention.py tests/test_inner_effects.py tests/test_brain_reflex_body.py`
Expected: PASS（反射那边用的 `Effects` 字段没受影响）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/attention.py src/skydango/inner/effects.py tests/test_brain_attention.py tests/test_inner_effects.py
git commit -m "feat(attention): 随意看、四种模式、focus；内心层 Effects.wander 倍数"
```

---

### Task 5: 身体接上注意力

**Files:**
- Modify: `src/skydango/brain/body.py`：`__init__`（存 `self.rng`，`Reflexes` 用同一个；建 `self.attention`；`_attention_pressed_at = -inf`、`_gestures_seen: deque`、`_held_pending: str | None`）、`step()`（`_watch_idle` 之后调 `_watch_attention(self.clock())`）、`_watch_comings`（approach 丢弃、gesture 另记）、`status()`（"刚才下意识"之后）、`_show()`（`info["注意力"]`）
- Test: `tests/test_brain_attention_body.py`（新；夹具照 `tests/test_brain_reflex_body.py` / `test_brain_body.py` 的 `body()` + `fake_panel(mode="auto")` + 带 `talkers` / `recent_approaches` / `people` / `requests` 的假 env + 记 `nudge` 的假镜头）

**Interfaces:**
- Consumes: Task 1 `panel.quiet(now, margin)` / `panel.pending` / `panel.hold_off(until, now)`；Task 2 `env.talkers(now)`、`env.recent_approaches(now)`；Task 3/4 `Attention`、`Target`、`Thought`；`Body.effects().wander`；`Body.target_x(name, now)`
- Produces:
  - `Body._attention_targets(now: float) -> list[Target]`：
    - `talkers`：好友 → `Target(f"t:{id}", "talk_friend", x, name, start)`；陌生人 → `"talk_stranger"`
    - `recent_approaches(now, 5.0)`：丢掉 `t - _attention_pressed_at < track.settle` 的（自己转出来的）；好友 key `n:<名字>`、陌生人 `a:<cx 取整到 50>`；`fresh = t`
    - `_gestures_seen` 里 5 s 内的 `(who, label, t)`：x 用 `target_x(who, now)`（拿不到就跳过），kind `act_on_me`，`fresh = t`
    - `people()` 里的好友：`friend_present`，key `t:<track_id>`
    - 各接口 `hasattr` 判断，没有就跳过
  - `Body._attention_blocked(now: float) -> str`：返回不能按的原因（空串 = 能按），按 spec §1 顺序：没开 / 模式别动、没镜头 / 没感知层、不是 auto、面板开着（`reader.panel_closed_since is None`）、不安静（`not panel.quiet(now, track.settle + 1)` 且不在先看一眼窗口）、输入框（`_bubble_at` / `sender.opened` / `device.ime_shown()`）、技能在跑、互动请求挂着（`env.requests`）、有攒着的 chat / owner_command 事件或 `brain_busy()`、黑屏、牵手、别的面板开着、刚做过动作（`now - emotes.last_any < reflex.min_gap`）、dry-run
  - `Body._watch_attention(now: float) -> None`：`attention.width = self.frame_width`；`th = attention.think(targets, now, self.effects().wander)`；
    先看一眼：`th.look_first and panel.pending in ("bubble", "approach") and _held_pending != panel.pending` → `panel.hold_off(now + cfg.attention.look_first, now)`，记 `_held_pending`；窗口里 `th.centered` 或目标没了 → `panel.hold_off(None, now)`；
    `th.action` 且 `_attention_blocked(now) == ""` → `camera.nudge(...)`、`attention.pressed(...)`、`_attention_pressed_at = now`、`_ref_thumb = None`；DEBUG 日志记换目标 / 按键 / 推迟放手
  - `_watch_comings`：`for who in pop_approaches()` 循环开头，`now - _attention_pressed_at < track.settle` → `continue`（在 `panel.trigger("approach")` 之前）；`pop_gestures()` 每条先 `_gestures_seen.append((who, label, now))`
  - status：`attention.describe()`；网页 `info["注意力"]` 同一句，不闲时后面加 `（先不动：<原因>）`

- [ ] **Step 1: 写失败的测试**

```python
def test_turns_toward_friend_talking_when_idle_in_auto_mode():     # 一圈后 cam.nudges == [("left", ...)]
def test_no_press_in_always_mode():
def test_no_press_when_panel_opened_this_round():                  # 同一圈 reader 读到新消息 → chatting → 不按
@pytest.mark.parametrize("block", ["ime", "bubble_box", "skill", "request", "pending_chat", "brain_busy",
                                   "blackout", "holding", "other_panel", "recent_emote", "dry_run", "no_camera", "sandbox_env"])
def test_blocked_conditions_do_not_press(block):
def test_dry_run_still_describes_target():                         # status 有"在看：小明（在说话）"
def test_look_first_holds_panel_then_releases_when_centered():     # bubble pending → 推迟、按键、centered 后放手、下一圈面板开
def test_look_first_expires_after_two_seconds():
def test_self_caused_approach_is_dropped_before_panel_trigger():   # 按键后 settle 内的 approach：没事件、panel.pending 没变
def test_approach_after_settle_is_kept():
def test_gesture_is_recorded_for_attention_even_when_reflex_returns_it():
def test_pressing_does_not_restart_idle_emote_timer():             # reflexes 的 _idle_due 不变
def test_pressing_clears_scene_change_reference():
def test_status_and_viewer_show_attention_line():
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_attention_body.py`
Expected: FAIL（没有 `_watch_attention`）

- [ ] **Step 3: 实现上面 Interfaces 里的成员和接入点**

- [ ] **Step 4: 跑测试确认通过，再跑全量**

Run: `python -m pytest -q tests/test_brain_attention_body.py && python -m pytest -q`
Expected: PASS，全量全绿（`tests/test_viewer.py::test_port_in_use_raises` 偶发失败就重跑一次并记下）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_attention_body.py
git commit -m "feat(body): 接上空闲注意力——闲着才按、先看一眼再开面板、自己转出来的走近不算、status / 网页显示在看什么"
```

---

### Task 6: 大脑的 `attention` 工具、提示词、文档

**Files:**
- Modify: `src/skydango/brain/body.py`（`set_attention`）、`src/skydango/brain/tools.py`（`DESCRIPTIONS`、`_bind`，**不进** `ACTIONS`）、`src/skydango/brain/mcp_server.py`（注册）、`src/skydango/brain/prompt.py`（一段）、`CLAUDE.md`（「统管大脑」里加一段、代码结构表加 `brain/attention.py`）
- Test: `tests/test_brain_tools.py`、`tests/test_brain_prompt.py`、`tests/test_brain_attention_body.py`

**Interfaces:**
- Consumes: `Attention.set_mode(mode, focus)`、`MODES`
- Produces:
  - `Body.set_attention(mode: str, focus: str | None = None) -> str`：`ValueError` 转 `ToolError("模式只能是：随意 / 好奇 / 专心 / 别动")`；返回 `"注意力：" + attention.describe()`
  - 工具 `attention(mode: str, focus: str = "")`，说明文字（原文照写）：`"改你闲着时东张西望的习惯：mode 随意（默认）/ 好奇（多看看）/ 专心（只看跟你说话、冲你来的人）/ 别动（不转镜头）；focus 填一个名字表示更想看他（空 = 不特别关注）。身体闲着会自己看，只在想改习惯时设，不用每轮设。"`
  - 提示词在「视角」一节后加：`"你闲着的时候身体会自己东张西望（有人说话、朝你走过来就转过去看一眼），不用你管；想换个习惯（主人说别乱转、想多看看某人）才用 attention。"`
  - 沙盒：不进 `SANDBOX_MISSING`（只改状态）

- [ ] **Step 1: 写失败的测试**

```python
# test_brain_tools.py
def test_attention_tool_is_listed_and_not_an_action():
    assert "attention" in DESCRIPTIONS and "attention" not in ACTIONS
def test_attention_tool_calls_body():   # FakeBody.set_attention 收到 ("专心", "小明")；focus "" → None
# test_brain_prompt.py：现有"每个工具名都出现在提示词里"的测试自动覆盖；再加
def test_prompt_mentions_idle_looking():
# test_brain_attention_body.py
def test_set_attention_changes_mode_and_status():
def test_set_attention_rejects_unknown_mode():
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_tools.py tests/test_brain_prompt.py tests/test_brain_attention_body.py`
Expected: FAIL

- [ ] **Step 3: 实现工具、提示词；CLAUDE.md 加一段**（要点：`[attention]`、只在 auto 模式、身体做反射大脑定模式、不回位、先看一眼再开面板、不打扰反射、**未在真机验证**、spec / plan 路径）

- [ ] **Step 4: 跑全量**

Run: `python -m pytest -q`
Expected: 全绿

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/tools.py src/skydango/brain/mcp_server.py src/skydango/brain/prompt.py CLAUDE.md tests/test_brain_tools.py tests/test_brain_prompt.py tests/test_brain_attention_body.py
git commit -m "feat(brain): attention 工具定东张西望的模式；提示词和文档"
```

---

## 收尾

- [ ] 整个分支评审（requesting-code-review），按意见修
- [ ] 合并进 main、推送（CLAUDE.md 规矩）；进度记忆里把"东张西望"加进真机清单（等 auto 模式验收后）
