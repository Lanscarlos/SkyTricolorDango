# 聊天记录面板按需打开 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 聊天记录面板平时关着、定时 / 有触发时看一眼、聊天中保持打开，由 `PanelManager` 统一开关；转镜头、换轮盘、接互动、好友树、技能改成向它借面板。

**Architecture:** `chat/panel.py` 的 `PanelKeeper` 扩成 `PanelManager`：`always` 模式保持现状（常开 + 关久了重开），`auto` 模式是状态机（idle / peek / bubble / chatting + 借出计数）。它不自己读聊天：身体 / Agent 每圈照常 `reader.read()`，把结果和"这一帧面板开没开"交给 `tick()`，管理者只决定按不按 C。在 cli 里建一个实例，传给 Body / Agent 和各组件。

**Tech Stack:** Python 3.13、pytest（假设备 + 假时钟，`tests/conftest.py` 的 `FakeDevice` / `scene`）

**Spec:** `docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md`

## Global Constraints

- 默认 `[panel] mode = "always"`：不改配置时，按键行为和现在一致（只允许多几次截图）
- `[panel]` 默认值：`idle_peek = 30.0`、`quiet_close = 45.0`、`peek_cooldown = 5.0`、`bubble_wait = 15.0`、`bubble_gone = 3.0`、`bubble_strangers = false`、`open_timeout = 1.5`
- `vision.mode != "log"` 或 `vision.log_open_key == 0`：管理者什么键都不按（`borrow` 只计数、`tick` / `before_speak` 什么都不做）
- 输入框开着（`device.ime_shown()`）时绝不按开面板的键（会打出字母 c）
- 只有 `PanelManager` 按 `vision.log_open_key`；其他模块不再自己开关面板
- 管理者只在身体线程 / Agent 主线程调用，不加锁；`tick()` 不 sleep（不阻塞主循环），`borrow` / `before_speak` / `ensure_open` 可以等（最多 `open_timeout`）
- 日志、提示词、注释用中文，风格同周围代码；不往仓库写模型名
- 跑测试：`python -m pytest -q`；每个任务结束全部测试要过

## Review Focus

1. **闲着时 `ime_shown()` 被频繁调用**：它是一次 adb `dumpsys`（慢）。只在要按键前、或聊天中安静计时到点要关之前调 —— 测试：Task 3 `test_idle_tick_does_not_poll_ime`
2. **刚按 C 关上后，读面板的下一帧还看得到面板**（关的动画）→ 不能被当成"意外开着"又进聊天中 —— 测试：Task 3 `test_just_closed_panel_is_not_unexpected_open`
3. **看一眼刚打开的第一帧面板没画完**，读不到新消息就关，漏读 → 连续 2 帧看到面板且没新消息才关 —— 测试：Task 3 `test_peek_waits_two_visible_frames`
4. **借出期间看一眼被打断**（peek 中途来了 `borrow`）→ 归还后要补看，不能丢 —— 测试：Task 3 `test_borrow_during_peek_retries_after_return`
5. **技能 / 转镜头途中出异常**，`borrow` 的计数和面板要复原，不然面板永远关着读不到聊天 —— 测试：Task 1 `test_borrow_restores_on_exception`

---

### Task 1: `PanelConfig` + `PanelManager`（always 模式、`borrow`），接进 Body / Agent

**Files:**
- Modify: `src/skydango/config.py`（新增 `PanelConfig`，`Config.panel`）
- Modify: `src/skydango/chat/panel.py`（`PanelKeeper` → `PanelManager`）
- Modify: `src/skydango/brain/body.py:20,91,166-169,176`、`src/skydango/agent.py:14,96,123-126,265-276`
- Test: `tests/test_panel.py`、`tests/test_config.py`

**Interfaces:**
- Produces:
  - `PanelConfig`（dataclass，字段和默认值见 Global Constraints，`mode: str = "always"`）；`Config.panel: PanelConfig`
  - `PanelManager(vision: VisionConfig, cfg: PanelConfig, device, reader, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic)`；`reader` 只用 `panel_visible(frame) -> bool` 和 `panel_closed_since`
  - `.active -> bool`（`vision.mode == "log" and vision.log_open_key`）
  - `.ensure_open() -> bool`（原样保留）
  - `.visible_now() -> bool`（新截一张图看面板开没开；不 active 时 False）
  - `.start(now: float) -> bool`（always：返回 `ensure_open()`；auto 见 Task 3）
  - `.tick(now: float, fresh: list, visible: bool, blackout: bool = False) -> None`（always：等于原来的 `maybe_reopen(now)`，借出中不重开）
  - `.borrow(who: str, close: bool = True)` → 上下文管理器，`as was_open: bool`（最外层借之前面板开没开）
  - `.lent -> str | None`（正借给谁；嵌套时是最外层的 who）
  - `.should_be_open() -> bool`（always：没借出时 True）
  - `.restored() -> bool`（`not should_be_open() or visible_now()`）
  - Body / Agent 构造多一个参数 `panel: PanelManager | None = None`，`None` 时自己建一个（`PanelManager(cfg.vision, cfg.panel, device, reader, lambda s: self.sleep(s), clock)`）；属性名仍是 `self.panel`

- [ ] **Step 1: 写失败的测试**（`tests/test_panel.py`：把现有 4 个测试改成用 `PanelManager(VisionConfig(mode=mode), PanelConfig(), device, Reader(visible), sleep=...)`，`maybe_reopen(t)` 改成 `tick(t, [], visible=False)`；新增：）

```python
def test_borrow_closes_then_reopens_in_always_mode():
    m, device = manager([True, False, True])   # 借前开着 → 关了 → 还的时候看到开了
    with m.borrow("camera") as was_open:
        assert was_open is True and m.lent == "camera"
        assert device.calls == [("hw_key", 46)]
    assert device.calls == [("hw_key", 46), ("hw_key", 46)] and m.lent is None

def test_borrow_nested_restores_once():
    m, device = manager([True, False, False, True])
    with m.borrow("camera"):
        with m.borrow("emotes") as inner:
            assert inner is False              # 外层已经关了
        assert m.lent == "camera" and device.calls == [("hw_key", 46)]
    assert device.calls.count(("hw_key", 46)) == 2

def test_borrow_close_false_does_not_press_on_enter():   # social / friendtree：点屏幕自己会关
    m, device = manager([True, False, True])
    with m.borrow("social", close=False):
        assert device.calls == []
    assert device.calls == [("hw_key", 46)]

def test_borrow_restores_on_exception():
    m, device = manager([True, False, True])
    with pytest.raises(RuntimeError):
        with m.borrow("camera"):
            raise RuntimeError("转镜头出错")
    assert m.lent is None and device.calls == [("hw_key", 46), ("hw_key", 46)]

def test_no_reopen_while_lent():
    m, device = manager([False])               # 面板一直关着
    m.reader.panel_closed_since = 0.0
    with m.borrow("camera", close=False):
        m.tick(60.0, [], visible=False)        # 早过了 log_reopen_after，但借出中不重开
        assert device.calls == []

def test_inactive_manager_never_presses():
    m, device = manager([True], mode="bubble")
    with m.borrow("camera") as was_open:
        assert was_open is False
    m.tick(100.0, [], visible=False)
    assert device.calls == []
```

`tests/test_config.py` 加：`load_config` 读 `[panel] mode = "auto"` 后 `cfg.panel.mode == "auto"`，其余字段是默认值。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_panel.py tests/test_config.py -q`
Expected: FAIL（`ImportError: cannot import name 'PanelManager'` / `PanelConfig`）

- [ ] **Step 3: 实现**

- `borrow` 进入：计数 0 → 1 时 `was_open = visible_now()`，`close and was_open` 就按键并 `sleep(0.8)`；归还（`finally`）：计数回 0 且 `should_be_open()` 且 `not visible_now()` → 按键，每 0.1 s 截图看，最多 `open_timeout`；没开成记 WARNING，交给 `tick` 重试（重试沿用原 `maybe_reopen` 的 `log_reopen_after` / `log_reopen_cooldown`）
- 归还前如果 `device.ime_shown()` 就不按（照旧交给 `tick`）
- Body：`_sense(frame, now)` 改成 `_sense(frame, now, fresh)`，把 `self.panel.maybe_reopen(now)` 换成 `self.panel.tick(now, fresh, visible=self.reader.panel_closed_since is None, blackout=self.blackout)`；`run()` 里 `self.panel.ensure_open()` → `self.panel.start(self.clock())`
- Agent：同样换成 `tick(...)`（`blackout=False`，Task 6 再接）；`ensure_log_open()` 返回 `self.panel.start(self.clock())`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS（`test_agent.py` / `test_brain_body.py` 不用改：Body / Agent 默认自己建管理者）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py src/skydango/chat/panel.py src/skydango/brain/body.py src/skydango/agent.py tests/test_panel.py tests/test_config.py
git commit -m "feat(panel): PanelManager 统一开关聊天记录面板（always 模式 + borrow）"
```

---

### Task 2: 转镜头 / 换轮盘 / 接互动 / 好友树改成借面板，cli 只建一个管理者

**Files:**
- Modify: `src/skydango/brain/camera.py`（构造、`_ready()`、`spin()` 的 `panel_reopened`）
- Modify: `src/skydango/game/emotes.py`（构造、`_panel_closed()`）
- Modify: `src/skydango/game/social.py`（构造、`accept()`，删 `_reopen_panel`）
- Modify: `src/skydango/game/friendtree.py`（构造、`check()`，删 `_reopen_panel`）
- Modify: `src/skydango/cli.py`（`_panel` 新函数；`_camera`、`_friend_checker`、`_build_emotes`、`_run_agent`、`_run_brain`、`cmd_camera`、`cmd_friend_check`、`emotes` 相关命令里构造这些类的地方）
- Test: `tests/test_brain_camera.py`、`tests/test_emotes.py`、`tests/test_social.py`、`tests/test_friendtree.py`、`tests/test_cli_*.py`（按需）

**Interfaces:**
- Consumes: Task 1 的 `PanelManager.borrow(who, close)`、`.restored()`、`.active`
- Produces:
  - `Camera(device, step, panel: PanelManager | None, sleep=..., clock=...)`（`None` = 没有面板要管）
  - `EmotePlayer(device, wheel, cfg, panel: PanelManager | None, sleep=..., clock=...)`
  - `SocialHandler(..., panel: PanelManager | None = None)`（替换 `panel_visible` 参数；`SocialConfig.panel_key` 不再使用，保留字段免得旧配置报错，注释写"已不用，面板键见 vision.log_open_key"）
  - `FriendChecker(device, cfg, panel: PanelManager | None = None, sleep=...)`（替换 `panel_visible` / `panel_key`）
  - `cli._panel(cfg, dev, reader, mode: str | None = None) -> PanelManager`（`mode` 覆盖 `cfg.panel.mode`；单独的命令 `camera spin` / `friend-check` / `emotes *` 用 `mode="always"`：做完恢复成原样）
  - `_run_agent` / `_run_brain` 建一个 `panel = _panel(cfg, dev, reader)`，传给 Body / Agent 和上面四个类

- [ ] **Step 1: 改测试**（行为不变，只换构造方式）：四个测试文件里用 `tests/conftest.py` 新增的帮手

```python
class PanelState:
    """假面板：hw_key(46) 切换开关；给 PanelManager 当 reader。"""
    def __init__(self, device, open_=True): ...
    def panel_visible(self, frame) -> bool: ...
    panel_closed_since = None

def fake_panel(device, open_=True, mode="always") -> tuple["PanelManager", PanelState]:
    """包一层 device.hw_key：按 46 切换 PanelState；返回 (管理者, 状态)。"""
```

原来断言按键序列的测试保持同样的按键序列（例：`test_turn_closes_panel_holds_arrow_and_reopens` 仍是 `[("hw_key", 46), 转 ×2, ("hw_key", 46)]`）；`test_social.py:111` / `test_friendtree.py:88` 仍断言面板最后是开着的。新增：

```python
def test_camera_in_idle_auto_mode_leaves_panel_closed():   # tests/test_brain_camera.py
    dev = FakeDevice([scene()])
    panel, state = fake_panel(dev, open_=False, mode="auto")   # auto + 闲着：面板本来关着
    c = Camera(dev, 0.25, panel, sleep=lambda s: None)
    c.move("left")
    assert ("hw_key", 46) not in dev.calls and state.open is False

def test_spin_reports_restored_when_idle():                 # 闲着不重开也算"回到该有的状态"
    ...  # 同上构造，c.spin(dev.screenshot, 1, 0.1, 10).panel_reopened is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_camera.py tests/test_emotes.py tests/test_social.py tests/test_friendtree.py -q`
Expected: FAIL（构造参数对不上 / `fake_panel` 不存在）

- [ ] **Step 3: 实现**

- `Camera._ready()`：先照旧关输入框（`ime_shown()` → BACK），再 `with self.panel.borrow("camera") as was_open: yield was_open`（`panel is None` 时 `yield False`），出来后照旧 BACK
- `Camera.spin`：`reopened = self.panel.restored() if self.panel else True`；`SpinResult.panel_reopened` 注释改成"面板回到了该有的状态（闲着时关着也算）"
- `EmotePlayer._panel_closed()` → `borrow("emotes")`；`SocialHandler.accept()` 整段包进 `borrow("social", close=False)`；`FriendChecker.check()` 包进 `borrow("friend_check", close=False)`；删两个 `_reopen_panel`
- `social.py` / `friendtree.py` 不再 import / 用 `panel_key`；`cli.py` 里所有 `panel_visible=`、`panel_key=` 构造参数删掉

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS；`grep -rn "log_open_key\|panel_key" src/skydango --include=*.py` 只剩 `config.py` 和 `chat/panel.py`

- [ ] **Step 5: 提交**

```bash
git add -A src/skydango tests
git commit -m "refactor(panel): 转镜头、换轮盘、接互动、好友树改成向 PanelManager 借面板"
```

---

### Task 3: auto 模式状态机（闲着 / 看一眼 / 聊天中）

**Files:**
- Modify: `src/skydango/chat/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: Task 1 的 `PanelManager`
- Produces（auto 模式下的语义；always 模式不变）：
  - `.state -> str`：`"idle"` / `"peek"` / `"bubble"` / `"chatting"`（always 模式恒为 `"chatting"`）
  - `.start(now) -> bool`：auto：`state = "idle"`、`last_read = now`，不按键，返回 `True`
  - `.trigger(reason: str, now: float) -> None`：`reason ∈ {"arrive", "approach", "chat_log", "scene"}`；只在闲着时记为待看
  - `.busy(now: float) -> None`：安静计时清零
  - `.missing_for(now: float) -> float`：该开（`should_be_open()`）却没开了多少秒；其余情况 0
  - `.describe(now: float) -> str`：`"聊天中"` / `"闲着（18 秒后看一眼）"` / `"看一眼"` / `"借出：camera"`；always 模式 `"常开"`；不 active 时 `""`
  - `.shutdown() -> None`：auto 下面板关着就 `ensure_open()`（退出时恢复成开着）
  - `should_be_open()`（auto）：没借出且 `state == "chatting"`

- [ ] **Step 1: 写失败的测试**（新帮手 `auto(visible, clock)` 用 `PanelConfig(mode="auto")`；`tick` 的 `visible` 参数直接给，不走 reader）

```python
def test_idle_peeks_after_idle_peek_and_closes_without_new(clock):
    m, dev = auto()
    m.start(0.0)
    m.tick(29.0, [], visible=False); assert dev.calls == []
    m.tick(30.0, [], visible=False); assert dev.calls == [("hw_key", 46)] and m.state == "peek"
    m.tick(30.2, [], visible=True); m.tick(30.4, [], visible=True)
    assert dev.calls == [("hw_key", 46), ("hw_key", 46)] and m.state == "idle"

def test_peek_waits_two_visible_frames():
    ...  # 只看到 1 帧面板时不关；第 2 帧才关

def test_peek_with_new_message_enters_chatting():
    ...  # peek 中 tick(t, [msg], visible=True) → state == "chatting"，不按键

def test_chatting_closes_after_quiet_close_and_checks_ime_first():
    ...  # 聊天中最后一条消息在 t=100；t=144.9 不关；t=145 关（按一次键）、state == "idle"
    ...  # 另一个用例：t=145 时 ime_shown() 为 True → 不关、安静重新计时

def test_busy_resets_quiet_timer():
    ...  # busy(130) 之后 t=145 不关、t=175 关

def test_trigger_in_idle_peeks_now_but_respects_cooldown():
    ...  # peek 刚在 t=10 结束；trigger("arrive", 12) → t=12 不按；t=15 按（peek_cooldown 5 s）

def test_trigger_ignored_while_chatting():
    ...

def test_no_peek_during_blackout_then_scene_trigger():
    ...  # blackout=True 时到点也不按；恢复后 trigger("scene") → 冷却外立刻看一眼

def test_peek_gives_up_after_open_timeout():
    ...  # 按了 C 1.5 s 内一直 visible=False → 回 idle、不连按（到下个 idle_peek 才再按）

def test_unexpected_open_panel_becomes_chatting():
    ...  # idle 且没在打开 / 刚关，tick(visible=True) → "chatting"，不按键

def test_just_closed_panel_is_not_unexpected_open():
    ...  # peek 关上后 open_timeout 秒内还 visible=True → 仍是 idle

def test_idle_tick_does_not_poll_ime():
    ...  # 闲着 tick 20 次（没到点）：device 上 ime_shown 调用次数为 0（FakeDevice 计数）

def test_borrow_during_peek_retries_after_return():
    ...  # peek 按了 C 之后 borrow → 状态回 idle 且待看；归还不重开（闲着）；下一次 tick 立刻再看一眼

def test_borrow_while_chatting_reopens_on_return():
    ...

def test_missing_for_counts_only_when_should_be_open():
    ...  # idle 关着 → 0；chatting 且 visible=False 从 t=200 起 → missing_for(230) == 30

def test_shutdown_reopens_panel_in_auto():
    ...

def test_describe():
    ...  # "闲着（18 秒后看一眼）"（start(0) 后 describe(12)）、"聊天中"、"借出：camera"
```

`FakeDevice` 若没有 `ime_shown` 调用计数，在 `tests/conftest.py` 加 `self.ime_calls`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_panel.py -q`
Expected: FAIL（`state` / `trigger` 等不存在或行为不对）

- [ ] **Step 3: 实现**（`tick` 的 auto 分支，每圈最多按一次键，不 sleep）

```
借出中：只记"欠着"（到点或有待看）→ return
fresh 非空：last_read = last_activity = now；state = "chatting"
visible：last_read = now
idle：
  visible 且 now - closed_at > open_timeout 且不在打开中 → "chatting"（意外开着），last_activity = now
  否则 (now - last_read >= idle_peek 或 有待看) 且 not blackout 且 now - last_peek >= peek_cooldown（定时不看冷却）
    → ime_shown() 为 True：state = "chatting"，return；否则按键，state = "peek"，opened_at = now，seen = 0，清待看
peek：
  visible：seen += 1；seen >= 2 → 按键关，state = "idle"，closed_at = last_peek = now
  not visible 且 now - opened_at >= open_timeout → WARNING，state = "idle"，last_read = last_peek = now
chatting：
  now - last_activity >= quiet_close：ime_shown() → last_activity = now；否则 visible 才按键关，state = "idle"，closed_at = now
  not visible：沿用 always 的重开（log_reopen_after / log_reopen_cooldown），并维护 missing_since
```

`borrow` 进入时若 `state in ("peek", "bubble")`：`state = "idle"`，记"待看"（bubble 记成待看的气泡，Task 4 用）。
状态切换打 INFO 日志：`面板：闲着 → 看一眼（定时）`、`看一眼：没有新消息，关上`、`面板：聊天中 → 闲着（安静 45 秒）`、`面板借给 camera`（最外层借出时）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/chat/panel.py tests/test_panel.py tests/conftest.py
git commit -m "feat(panel): auto 模式：闲着定时看一眼、聊天中保持打开、安静 45 秒关"
```

---

### Task 4: 气泡触发（等气泡状态）+ 说话前开面板

**Files:**
- Modify: `src/skydango/chat/panel.py`
- Modify: `src/skydango/vision/perception.py`（`typing_seen`）
- Test: `tests/test_panel.py`、`tests/test_perception.py`

**Interfaces:**
- Consumes: Task 3 的状态机
- Produces:
  - `PanelManager.bubble_seen(now: float) -> None`：这一圈看到了该触发的气泡。闲着 → 待看（气泡，最强）；等气泡中 → 更新 `last_bubble`
  - `PanelManager.before_speak(now: float) -> None`：auto、没借出、`state != "chatting"` → 面板没开就按键并等（每 0.1 s 截图，最多 `open_timeout`），`state = "chatting"`、`last_activity = now`；借出中只把"借出前的状态"记成 chatting（归还时重开）。always / 不 active：什么都不做
  - `PerceptionWatcher.typing_seen(now: float, within: float = 1.0, strangers: bool = False) -> bool`：`_typing` 里 `now - t <= within` 且（是好友或 `strangers`）的记录存在。`EnvWatcher` 不加（调用方用 `hasattr`）

- [ ] **Step 1: 写失败的测试**

```python
def test_bubble_keeps_panel_open_until_message():
    ...  # idle，bubble_seen(10) → tick(10) 按键、state == "bubble"；
    ...  # visible 且持续 bubble_seen 到 t=16 都不关；tick(16, [msg], True) → "chatting"

def test_bubble_closes_after_bubble_gone():
    ...  # 最后一次 bubble_seen 在 t=12；t=14.9 还开着；t=15 关、回 idle

def test_bubble_closes_after_bubble_wait():
    ...  # 一直 bubble_seen；t=10 打开，t=25 关

def test_bubble_seen_while_chatting_changes_nothing(): ...

def test_before_speak_opens_panel_first_when_idle():
    ...  # idle、面板关：before_speak(5) 按一次 46、state == "chatting"；随后 sender 的 Enter 在它之后

def test_before_speak_while_lent_reopens_on_return(): ...

def test_before_speak_does_nothing_in_always_mode(): ...
```

`tests/test_perception.py`：用已有的构造方式塞 `_typing` 记录（好友 / 陌生人 / 过期各一条），断言 `typing_seen(now)`、`typing_seen(now, strangers=True)`、`within` 过期为 False。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_panel.py tests/test_perception.py -q`
Expected: FAIL（`bubble_seen` / `before_speak` / `typing_seen` 不存在）

- [ ] **Step 3: 实现**

- 待看是气泡时，`tick` 打开后进 `"bubble"`（`bubble_start = last_bubble = now`），其余进 `"peek"`
- `"bubble"`：fresh 非空 → chatting（通用规则）；`visible` 且（`now - last_bubble >= bubble_gone` 或 `now - bubble_start >= bubble_wait`）→ 关、idle；没打开的超时同 peek
- `describe`：`"等气泡（还剩 9 秒）"`（按 `bubble_wait` 算）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/chat/panel.py src/skydango/vision/perception.py tests/test_panel.py tests/test_perception.py
git commit -m "feat(panel): 好友头顶冒气泡时开面板等消息；说话前先开面板"
```

---

### Task 5: 接进大脑的身体（触发、忙、说话、chat_log、事件、技能、提示词、网页）

**Files:**
- Modify: `src/skydango/brain/body.py`
- Modify: `src/skydango/brain/skills.py`
- Modify: `src/skydango/brain/loop.py`（`Brain.in_turn`）
- Modify: `src/skydango/brain/prompt.py`（`brain_prompt(..., panel_auto: bool = False)`）
- Modify: `src/skydango/cli.py`（`_run_brain`：`body.brain_busy`、`brain_prompt(..., panel_auto=cfg.panel.mode == "auto")`）
- Test: `tests/test_brain_body.py`、`tests/test_brain_skills.py`、`tests/test_brain_loop.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 3 / 4 的 `trigger`、`bubble_seen`、`busy`、`before_speak`、`missing_for`、`should_be_open`、`state`、`describe`、`shutdown`；`PerceptionWatcher.typing_seen`
- Produces:
  - `Body.brain_busy: Callable[[], bool]`（默认 `lambda: False`，cli 设成 `lambda: brain.in_turn`）
  - `Brain.in_turn: bool`（`wake()` 里 `session.send` 前后，`try/finally`）
  - `Skill.needs_camera: bool`（协议里加，默认按 `getattr(skill, "needs_camera", False)` 读）；`SkillRunner(events, clock, panel=None)`
  - `brain_prompt(reply, store, quick_around=False, panel_auto=False)`

- [ ] **Step 1: 写失败的测试**（`tests/test_brain_body.py` 的 `body()` 帮手加 `panel_mode="always"` 参数，auto 时传 `cfg.panel.mode = "auto"`；需要时给 `FakeEnv` 加 `typing_seen` / `pop_approaches`）

```python
def test_arrive_triggers_peek_in_auto(clock): ...        # env.near 多了"小明" → panel.state == "peek"（或按了 46）
def test_approach_triggers_peek(clock): ...
def test_friend_bubble_calls_bubble_seen(clock): ...     # FakeEnv.typing_seen 返回 True → 下一圈 state == "bubble"
def test_pending_chat_event_keeps_panel_busy(clock): ... # events 里有未取走的 chat → 聊天中过了 45 s 也不关
def test_brain_in_turn_keeps_panel_busy(clock): ...
def test_say_opens_panel_before_enter_when_idle(clock):
    ...  # live、auto、闲着：say("你好") 后 device.calls 里 ("hw_key", 46) 在 sender 打开输入框之前
def test_chat_log_peeks_when_idle(clock):
    ...  # 闲着时 chat_log() 同步看一眼：reader 这时返回的新消息进了 body.chat 和事件，返回里有它
def test_panel_event_only_when_should_be_open(clock):
    ...  # auto 闲着、面板关 60 s：没有 "panel" 事件；聊天中关了 30 s：有
def test_scene_restored_triggers_peek(clock): ...        # blackout True → False 后 trigger("scene")
def test_shutdown_reopens_panel_in_auto(clock): ...
def test_viewer_info_has_panel_state(clock): ...         # _show 的 info["聊天面板"] == panel.describe(now)
```

`tests/test_brain_skills.py`：`needs_camera = True` 的假技能开始时 `panel.lent == "skill"`，done / failed / cancel 后 `panel.lent is None`；没这个属性的技能不借。
`tests/test_brain_loop.py`：`wake()` 期间假 session 里读 `brain.in_turn is True`，返回后 False（异常时也 False）。
`tests/test_brain_prompt.py`：`panel_auto=True` 时包含"聊天面板平时关着，画面外的人说话可能晚半分钟才看到"，False 时不包含。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_body.py tests/test_brain_skills.py tests/test_brain_loop.py tests/test_brain_prompt.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

- `_watch_people`：新到的每个名字 `self.panel.trigger("arrive", now)`（一圈一次就够）；`pop_approaches` 有结果 → `trigger("approach", now)`；`hasattr(self.env, "typing_seen") and self.env.typing_seen(now, strangers=self.cfg.panel.bubble_strangers)` → `bubble_seen(now)`
- `step()`：`self.events.has("chat") or self.events.has("owner_command") or self.brain_busy()` → `self.panel.busy(now)`
- `_watch_screen`：blackout 变回 False 时 `self.panel.trigger("scene", now)`
- `_watch_panel`：`lost = self.panel.missing_for(now) >= PANEL_LOST_AFTER`（事件文案不变）
- `say()` live 分支：`self.panel.before_speak(now)` 在 `self.sender.send(full)` 前
- `chat_log()`：`self.panel.state == "idle"` 且 active 时同步看一眼：`trigger("chat_log", now)`，然后最多 `open_timeout + 1` 秒循环「截图 → `reader.read` → `panel.tick` → `_heard`」，`state` 离开 `"peek"` 就停；再照旧拼聊天记录
- `shutdown()`：最后调 `self.panel.shutdown()`（在复原镜头、恢复轮盘之后）
- `_show()`：`info["聊天面板"] = self.panel.describe(now)`（空字符串不加）
- `SkillRunner.start`：`needs_camera` 的技能开始后 `self._lease = self.panel.borrow("skill"); self._lease.__enter__()`；`_finish` 里 `__exit__(None, None, None)`（包 try，出错也清掉）；Body 构造 `SkillRunner(events, clock, panel=self.panel)`
- 提示词加的那句只在 `panel_auto` 时加，放在规则里讲聊天的地方

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain src/skydango/cli.py tests
git commit -m "feat(panel): 大脑的身体按需开面板：来人 / 走近 / 气泡时看、说话前开、技能借面板"
```

---

### Task 6: 接进普通 Agent

**Files:**
- Modify: `src/skydango/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 3 / 4 的 `tick`、`trigger`、`bubble_seen`、`busy`、`before_speak`、`describe`、`shutdown`
- Produces: 无新接口

- [ ] **Step 1: 写失败的测试**（`build()` 帮手能传 `cfg.panel.mode = "auto"`；需要时给 env 假对象）

```python
def test_agent_opens_panel_before_type_ahead(clock): ...  # auto 闲着、live、type_ahead：("hw_key", 46) 在输入框打开之前
def test_agent_pending_keeps_panel_busy(clock): ...       # pending 非空时聊天中不因安静而关
def test_agent_new_friend_triggers_peek(clock): ...       # env.nearby 多了人 → state == "peek"
def test_agent_bubble_triggers(clock): ...
def test_agent_no_peek_during_blackout(clock): ...        # 全黑帧：到点不按
def test_agent_command_reply_opens_panel_first(clock): ...# 主人 # 命令的确认语也先开面板
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_agent.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

- `step()`：`tick(now, fresh, visible=self.reader.panel_closed_since is None, blackout=is_black(frame))`（只在 `frame is not None` 时；截图失败这圈不 tick）；`self.pending` 非空 → `busy(now)`；`self.env` 有 `nearby` 时和上一圈比，多了人 → `trigger("arrive", now)`（`self._nearby: set[str]`）；`typing_seen` 同 Body
- `before_speak(now)`：在 `self.sender.open()`（type_ahead）、`self.sender.send(text)`、`_handle_command` 的 `self.sender.send(text)` 前，都只在非 dry-run 分支
- `_show()` 加 `info["聊天面板"]`；`run()` 结束（`log.info("到时间了…")` 前后都行，含 `KeyboardInterrupt` 由 cli 的 `finally`）调 `self.panel.shutdown()`：在 `cli._run_agent` 的 `finally` 里、恢复轮盘之后调 `agent.panel.shutdown()`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/agent.py src/skydango/cli.py tests/test_agent.py
git commit -m "feat(panel): 普通 Agent 也按需开面板"
```

---

### Task 7: 配置模板、文档，合并

**Files:**
- Modify: `config.example.toml`（`[panel]` 一节，默认值 + 中文注释，同 spec §6）
- Modify: `CLAUDE.md`（代码结构表 `chat/` 那行提 `panel.py`；新增一小节「聊天面板（`[panel]`）」：默认常开、`auto` 的行为一句话、真机验收前别改默认、借面板的规矩 —— 新代码要关面板一律 `panel.borrow(...)`，不许自己按 C）
- Modify: `docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md`（状态改成"代码已完成，待真机验收（§9）"）
- Test: `tests/test_config.py`（`config.example.toml` 能被 `load_config` 读、`[panel]` 字段齐全 —— 若已有"模板能加载"的测试，确认它覆盖到）

- [ ] **Step 1: 写模板 / 文档**
- [ ] **Step 2: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 3: 提交、推送、合并进 main**

```bash
git add config.example.toml CLAUDE.md docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md tests/test_config.py
git commit -m "docs(panel): 配置模板、CLAUDE.md、spec 状态"
git push -u origin claude/intelligent-cerf-e9tsim
git push origin claude/intelligent-cerf-e9tsim:main   # 远程 main 是这个分支的祖先时快进；否则先 merge origin/main 再推
```
