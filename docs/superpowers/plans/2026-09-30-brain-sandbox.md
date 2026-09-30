# 大脑沙盒（管理面板「沙盒」页）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**前置和并行：** 本计划用内心页计划（`docs/superpowers/plans/2026-09-30-inner-viewer.md`）的 `MindLog`、`Body.inner_snapshot` / `forget`、`inner/api.py`、`console/inner_view.py`、页面的 `renderInnerNow` / `loadInner`。
分两段执行：
- **线 2（和内心页计划并行，不碰 `cli.py`、不依赖内心页）：Task 1 → 3 → 9 → 7** —— 已完成并合进 main（f9fe3fc）；实现时和计划的出入：`.gitignore` 写 `/sandbox/`、剧本 `[start] time` 可空 / `resume` / `sleep`、`brain_trace.js` 只暴露 `mountBrainTrace`、起始时间下限也看 `current.json`
- **顺序段（内心页计划和线 2 都合进 main 之后）：Task 2 → 4 → 5 → 6 → 8 → 10 → 11**

**Goal:** 不开 MuMu，在管理面板上跑真的大脑 + 身体 + 内心层：冒充好友说话、造来去、快进时间、看团子的反应和心里；能把一段操作录成剧本回放、出报告。

**Architecture:** 把 `cli._run_brain` 拆成"组装身体 / 大脑 / 内心层"和"接世界的东西"（`World`）：真机是 `GameWorld`（原样搬现有代码），沙盒是 `SandboxWorld`（假设备、队列读聊天、名单当身边、场景文字当眼睛、模拟时钟）。沙盒是单独的子进程（`python -m skydango sandbox`），只给 JSON 接口；管理面板管起停（和团子互斥）、转发 `/sandbox/*`、原生「沙盒」页；剧本的录制 / 回放在管理面板进程里做。

**Tech Stack:** Python 3.13 标准库（`tomllib` 读剧本）；pytest；原生 JS。

**Spec:** `docs/superpowers/specs/2026-09-30-brain-sandbox-design.md`

## Global Constraints

- 目录：`[sandbox] dir = "sandbox"` → `sandbox/memory/`、`sandbox/scenarios/`、`sandbox/reports/`、`sandbox/clock.json`；`.gitignore` 加 `sandbox/`；**沙盒永远不写 `memory/`**
- `[sandbox]`：`dir = "sandbox"`、`port = 19392`、`step_timeout = 180.0`、`wake_hour = 9`、`emotes = []`（管理面板设置清单不加）
- 沙盒总是 live（`reply.dry_run = False`，`reply.memory_dir = <dir>/memory`）；运行目录 `runs/<时间>-sandbox/`
- 模拟时钟：`wall() = time.time() + offset`、`clock() = time.monotonic() + offset`；只往前拨；身体、事件队列、账本、反思器、眼睛、`MindLog`、**大脑循环（`Brain` 的 `clock` 和 `wall`）都用它**——大脑拿自己的时钟和事件队列的时间比（`now - events.last_put >= debounce`），两边必须是同一个钟。一轮的超时在 `brain/claude.py` 里直接用 `time.monotonic`，不受快进影响（线 2 终审核实）
- 起始时间下限 = max(`clock.json` 里上次停下的时间, 沙盒 `days.jsonl` 最后一行的 `end`)；早于下限拒绝
- 假设备的截图：1920×1080 三通道中灰（`128`），不是黑屏
- 场景文字空着时眼睛返回 `"看不清，眼前什么也看不出来"`
- 聊天记录行：`{seq, t, kind, who, text}`，`kind` ∈ `heard` / `said` / `act` / `event` / `blocked`；内存最多 500 行
- `idle` = 事件队列空 + 大脑不在一轮里 + 反思不在跑 + 身体命令队列空，持续 ≥ 2 秒（真实时间）
- 沙盒 HTTP：只监听 127.0.0.1，GET / POST 都校验 Host，POST 走 `post_guard`（`X-Skydango: 1`、JSON、≤ 64 KB）；`/state` 长轮询最多等 25 秒
- 团子和沙盒同一时间只能有一个子进程在 `starting` / `running` / `stopping`
- 剧本步骤键：`say`（配 `who`）/ `come` / `leave` / `strangers` / `place` / `scene` / `notice` / `skip` / `time` / `reflect` / `offline` / `online`；可选 `expect`（只进报告）、`wait`（秒，真实时间）；时长写法 `30s` / `10m` / `2h`
- 中文注释 / 日志；页面不引外部资源；出错只记日志、不拖垮面板

## Review Focus

1. **快进之后来一条聊天**：大脑照样在 `chat.debounce` 内醒来（大脑和事件队列同一个钟）；快进时正在想的一轮不被判超时（超时在 `claude.py` 用真实时间）——Task 2 测
2. **重置后沙盒时间早于真记忆里最后一次上线**（比如剧本写 `time = "20:00"`，真实最后一次上线是今晚 22:00）：拨到下一个 20:00，不让账本倒着走——Task 1 测
3. **团子在跑时点"启动沙盒"，或沙盒在跑时从总览叫醒团子**：都拒绝，提示先停另一个——Task 6 测
4. **剧本回放中间点了"停止回放"又手动发言**：当前这步做完就停，之后的手动操作照常，报告只写到停下为止——Task 10 测
5. **沙盒跑完 `memory/` 一个字节都没变**（包括 `inner/`、`history.jsonl`、`inbox.md`）——Task 5 测（跑前跑后比哈希）

---

## A 部分：沙盒本体 + 页面

### Task 1: 模拟时钟 `SimClock`

**Files:**
- Create: `src/skydango/sandbox/__init__.py`（空）、`src/skydango/sandbox/clock.py`
- Test: `tests/test_sandbox_clock.py`

**Interfaces:**
- Produces:
  - `class SimClock(offset: float = 0.0, real_wall=time.time, real_mono=time.monotonic)`：`wall() -> float`、`clock() -> float`、`skip(seconds: float) -> None`（负数 `ValueError`）
  - `SimClock.set_time(at: str) -> float`：`"HH:MM"` = 往后最近的这个时刻（今天已过就明天）；`"YYYY-MM-DD HH:MM"` 早于当前沙盒时间 → `ValueError("不能往回拨：…")`；返回跳了多少秒
  - `parse_duration(text: str) -> float`（`"30s"` / `"10m"` / `"2h"`，别的 `ValueError`）
  - `resolve_start(choice: str, floor: float, now: float, wake_hour: int) -> float`：`choice` ∈ `"resume"`（= max(floor, now)）/ `"sleep"`（floor 之后最近的 `wake_hour:00`，且至少晚于 floor）/ `"HH:MM"`（floor 与 now 取大之后最近的这个时刻）/ 完整时间（早于 floor → `ValueError`）
  - `load_saved(path: Path) -> float | None`、`save(path: Path, wall: float) -> None`（坏文件 → `None` + WARNING）
  - `floor_time(sandbox_dir: Path) -> float`：`clock.json` 与 `memory/inner/days.jsonl` 最后一行 `end` 取大，都没有返回 `0.0`

- [ ] **Step 1: 写失败的测试**

```python
from skydango.sandbox.clock import SimClock, parse_duration, resolve_start

T = time.mktime((2026, 9, 30, 21, 0, 0, 0, 0, -1))   # 真实：21:00

def test_skip_moves_both_clocks():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 100.0)
    c.skip(3600)
    assert c.wall() == T + 3600 and c.clock() == 3700.0

def test_set_time_forward_only():
    c = SimClock(real_wall=lambda: T, real_mono=lambda: 0.0)
    assert c.set_time("23:30") == 2.5 * 3600
    assert c.set_time("20:00") == 20.5 * 3600               # 今天已过：明天 20:00
    with pytest.raises(ValueError):
        c.set_time("2026-09-30 22:00")

def test_resolve_start_respects_floor():                      # Review Focus 2
    floor = T + 3600                                          # 真记忆里最后一次上线结束在 22:00
    assert resolve_start("20:00", floor, T, 9) == time.mktime((2026, 10, 1, 20, 0, 0, 0, 0, -1))
    assert resolve_start("sleep", floor, T, 9) == time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))
    assert resolve_start("resume", floor, T, 9) == floor
    with pytest.raises(ValueError):
        resolve_start("2026-09-30 21:30", floor, T, 9)

def test_parse_duration():
    assert [parse_duration(x) for x in ("30s", "10m", "2h")] == [30, 600, 7200]
    with pytest.raises(ValueError):
        parse_duration("2d")
```

外加：`save` / `load_saved` 来回一致、坏 JSON 返回 `None`；`floor_time` 取 `days.jsonl` 最后一行的 `end`。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_sandbox_clock.py -q`
- [ ] **Step 3: 实现 `sandbox/clock.py`**（本地时区用 `time.localtime` / `time.mktime`）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交**：`git commit -m "feat(sandbox): 模拟时钟（只往前拨、起始时间下限）"`

---

### Task 2: 拆出 `World`，`_run_brain` 只管组装

**Files:**
- Modify: `src/skydango/brain/world.py`（Task 3 建好了 `World`；这里加 `BrainParts`）
- Modify: `src/skydango/cli.py`（`_run_brain`、`_inner_ledger`、`_inner_mind`、`_inner_persona`、`_final_reflection`、`_days_prompt`、`cmd_run`；新 `_game_world`）
- Modify: `src/skydango/brain/loop.py`（`Brain.in_turn`）
- Test: `tests/test_cli_brain.py`（现有全过 + 追加）、`tests/test_brain_loop.py`（追加）

**Interfaces:**
- Consumes: Task 3 的 `World`
- Produces:
  - `@dataclass BrainParts`：`body`、`eyes`、`events`、`brain`、`trace`、`reflector`、`ledger`、`store`、`mind_log`
  - `cli._game_world(cfg, run, no_emotes: bool) -> World`：把现有的 `_device` / `_build_reader` / `_panel` / `_scene_watcher` / `SocialHandler` / `_build_emotes` / `_camera` / `_friend_checker` / `_panels` / `ChatSender` / `Locomotion` 原样搬进来（**仍按模块名调 `_device`、`_build_reader`**，现有测试的 monkeypatch 继续生效）
  - `cli._run_brain(cfg, run, world: World, duration=0.0, viewer=None, on_ready: Callable[[BrainParts], None] | None = None) -> None`；`cmd_run` 改成 `_run_brain(cfg, run, _game_world(cfg, run, args.no_emotes), args.duration, viewer)`
  - `_run_brain` 里所有 `time.time()` 换成 `world.wall()`；`Body(clock=world.clock, wall=world.wall)`、`Reflector(clock=world.clock)`、`Eyes(clock=world.clock)`、`EventQueue(clock=world.clock)`、`MindLog` 的时间来自身体；`Brain(..., clock=world.clock, wall=world.wall)`
  - `World` 加 `text_only: bool = False`（沙盒为真）：`ToolBox(text_only=True)` 时 `look` 不管 `image` 参数都走眼睛的文字（`eyes.describe_frame`，不附图、不附"画面里没认出人"这类位置说明），`look_person` 返回 `"沙盒里看不到人，只能靠聊天和场景"`；眼睛的 `note` 在 `text_only` 时返回空字符串
  - `_inner_ledger(cfg, store, now)`、`_inner_mind(cfg, ledger, base, claude_vars, run, now, clock)`、`_inner_persona(cfg, ledger, now)`、`_final_reflection(cfg, body, reflector, ledger, live_store, now)`、`_days_prompt(ledger, cfg, now)`：多一个时间参数
  - `Brain.in_turn: bool`：`wake` 开始时置真，结束（成功 / 失败 / 超时）置假
  - 身体建好后、线程启动前调 `on_ready(parts)`

- [ ] **Step 1: 写失败的测试**
  - `test_cli_brain.py` 追加：`fake_brain_run` 之后用 `cli._game_world(cfg, run, True)` 替换原来的参数，`world.wall = lambda: FIXED`，开 `inner`（`cfg.reply.memory_dir = tmp_path / "mem"`、live）跑 3 秒后，`days.jsonl` 那一行的 `start == FIXED`（证明账本用的是 world 的时间）；`on_ready` 被调一次，拿到的 `parts.body` 是 `Body`
  - `test_brain_loop.py` 追加：假 session 的一轮里读 `brain.in_turn` 为真，结束后为假；抛异常的一轮之后也为假；**`SimClock` 当 `Brain` 和 `EventQueue` 的钟，`skip(3600)` 之后放一条 `chat`，真实时间过了 `chat.debounce` 秒内 `Brain` 醒来**（Review Focus 1）
  - `test_brain_tools.py` 追加：`ToolBox(text_only=True)` 的 `look(image=True)` 返回里没有图片块、是眼睛的文字；`look_person` 返回沙盒那句话
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_cli_brain.py tests/test_brain_loop.py -q`
- [ ] **Step 3: 实现**：纯搬移为主；`_run_brain` 的 `finally` 里原来的 `_stop_scene(env)` / `panels.close()` 换成 `world.close()`；眼睛 `describe = world.describe or (lambda content: one_shot(...))`
- [ ] **Step 4: 跑测试确认通过**：先跑上面两个文件，再全量 `python -m pytest -q`（**现有测试一个都不能改**，只能追加）
- [ ] **Step 5: 提交**：`git commit -m "refactor(cli): _run_brain 拆出 World（接世界的东西），时间都从 world 取"`

---

### Task 3: 沙盒世界的部件（含 `World` 数据类）

**Files:**
- Create: `src/skydango/brain/world.py`（只有 `World` 数据类，见下；Task 2 再加 `BrainParts`、接进 `_run_brain`）
- Create: `src/skydango/sandbox/world.py`、`src/skydango/sandbox/transcript.py`
- Test: `tests/test_sandbox_world.py`

**Interfaces:**
- Consumes: Task 1 `SimClock`
- Produces:
  - `brain/world.py`：`@dataclass World`：`device`、`reader`、`self_filter`、`panel`、`env`、`social`、`emotes`、`camera`、`locomotion`、`sender`、`friend_checker`、`panels`、`panel_ops`、`clock: Callable[[], float] = time.monotonic`、`wall: Callable[[], float] = time.time`、`describe: Callable[[list[dict]], str] | None = None`（`None` = 眼睛用 Haiku）、`name: str = "game"`、`close: Callable[[], None] = lambda: None`（收尾：停 env、关 panels）
  - `transcript.py`：`class Transcript(clock: SimClock, limit=500)`：`add(kind: str, text: str, who: str = "") -> dict`、`since(seq: int) -> list[dict]`、`version: int`、`cond: threading.Condition`（`add` 时 `notify_all`，给长轮询）
  - `world.py`：
    - `GRAY: np.ndarray`（1080×1920×3，`uint8`，全 128）
    - `SandboxDevice`：`screenshot() -> GRAY.copy()`；`tap` / `swipe` / `hw_key` / `hw_key_hold` / `hw_key_down` / `hw_key_up` / `text` / `ime_*` 都什么也不做（照 `tests/conftest.py` 的 `FakeDevice` 列方法，但不记录）；`ime_shown() -> False`
    - `SandboxReader`：`say(who: str, text: str) -> None`（入队）；`read(frame, now) -> list[Message]`（出队，`Message(text, Rect(0,0,1,1), now, who)`）；`panel_visible(frame) -> True`；其余照 `tests/test_brain_body.py` 的 `FakeReader` 补齐身体会调的属性
    - `SandboxEnv`：`friends: list[str]`、`stranger_count: int`、`place: str`；`observe(...)` 空、`nearby(now)`、`strangers(now)`、`unlit(now) -> 0`、`people(now) -> []`、`labels = {}`、`requests = {}`、`circles = {}`、`hold` / `release` / `held` 空实现
    - `SandboxSender(transcript)`：`opened: bool`、`open() -> True`（记 `act`："（团子头顶冒出输入气泡）"）、`cancel()`、`send(text)`（记 `said`、关气泡）
    - `SandboxEmotes(names: list[str], transcript, clock)`：接口同 `EmotePlayer` 身体用到的部分（`on_wheel()`、`available(ignore_interval=False)`、`perform(name, reflex=False) -> int`、`pretend(name, reflex=False)`、`restore()`、`last_any`、`last_emote`）；`perform` / `pretend` 都记 `act`："（团子做了 鞠躬）"，反射的写"（团子下意识地 鞠躬）"
    - `SandboxLocomotion(transcript)`：`move(direction, steps=1, max_steps=3) -> str`（记 `act`："（团子往前走了 2 步）"，W/A/S/D → 前 / 左 / 后 / 右）
    - `class Scene`：`text: str = ""`；`describe(content) -> str` 返回 `text` 或 `"看不清，眼前什么也看不出来"`
    - `emote_names(cfg) -> list[str]`：`emotes/` 图标库里的名字（`EmoteLibrary(cfg.wheel.library_dir).names`），空的就用 `cfg.sandbox.emotes`
    - `sandbox_world(cfg, sim: SimClock, transcript: Transcript, scene: Scene) -> World`：`panel` 用真的 `PanelManager`（`mode = "always"`，reader 是 `SandboxReader`）；`camera`、`friend_checker`、`panels`、`panel_ops`、`social` 为 `None`；`clock=sim.clock`、`wall=sim.wall`、`describe=scene.describe`、`name="sandbox"`
- Also: `src/skydango/config.py` 加 `SandboxConfig`（Global Constraints 的五个键）挂在 `Config.sandbox`；`config.example.toml` 加 `[sandbox]` 一节；`.gitignore` 加 `sandbox/`

- [ ] **Step 1: 写失败的测试**（真 `Body` + `sandbox_world` 的部件 + `SimClock`，不起大脑；照 `tests/test_brain_body.py` 的方式单步 `b.step()`）

```python
def test_heard_becomes_chat_event(sb):
    sb.reader.say("小明", "在吗"); sb.body.step()
    assert any(e.kind == "chat" and "在吗" in e.text for e in sb.events.drain())

def test_nearby_list_drives_arrive_and_ledger(sb):
    sb.env.friends = ["小明"]; sb.body.step()
    assert any(e.kind == "arrive" for e in sb.events.drain())
    assert sb.ledger.card("小明").visits == 1

def test_say_and_emote_go_to_transcript(sb):
    sb.body.say("你好呀", live=True)
    sb.body.emote("鞠躬", live=True)
    kinds = [(r["kind"], r["text"]) for r in sb.transcript.since(0)]
    assert ("said", "你好呀") in kinds and ("act", "（团子做了 鞠躬）") in kinds

def test_scene_text_or_blind():
    s = Scene(); assert s.describe([]) == "看不清，眼前什么也看不出来"
    s.text = "云野，篝火旁"; assert s.describe([]) == "云野，篝火旁"

def test_gray_frame_is_not_blackout(sb):
    sb.body.step(); assert sb.body.blackout is False
```

`sb` fixture：`SimClock(real_wall=lambda: WALL)`、`Transcript`、`Scene`、`sandbox_world(cfg, …)`，用它的部件建 `Body`（带 `Ledger`，照 `tests/test_brain_inner_body.py` 的造法），返回一个带 `body / reader / env / events / transcript / ledger` 的对象。

- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_sandbox_world.py -q`
- [ ] **Step 3: 实现**（身体调了哪个缺的属性就补哪个，不改 `Body`）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交**：`git commit -m "feat(sandbox): 沙盒世界（假设备、队列读聊天、名单当身边、场景当眼睛、聊天记录）"`

---

### Task 4: 沙盒操作和状态（`SandboxControl`）

**Files:**
- Create: `src/skydango/sandbox/control.py`
- Modify: `src/skydango/brain/loop.py`（`limit_left`）、`src/skydango/brain/body.py`（`on_blocked`、`_watch_news` 返回拦截原因）、`src/skydango/inner/log.py`（`on_add`）
- Test: `tests/test_sandbox_control.py`、`tests/test_brain_loop.py`（追加 `limit_left`）

**Interfaces:**
- Consumes: Task 1–3；`BrainParts`（Task 2）；前置计划的 `MindLog.energy`、`Body.energy_now`
- Produces:
  - `class SandboxControl(parts: BrainParts, world_parts, sim: SimClock, transcript: Transcript, scene: Scene)`（`world_parts` = Task 3 的 reader / env 等，按需传）
  - `op(req: dict) -> dict`（`{"ok": bool, "text": str}`）：经 `parts.body.call(fn, timeout=10)` 在身体线程执行；各 op：
    - `say {who, text}` → `reader.say`，记 `heard`
    - `come {who}` / `leave {who}` → 改 `env.friends`，记 `event`："── 小明来了 ──" / "── 小明走了 ──"
    - `strangers {n}`（0~20）、`place {name}`、`scene {text}` → 改状态，记 `event`
    - `notice {text}` → `body.news(text)`；下一圈没变成 `notice` 事件时，记 `blocked` 并附原因（直接复用 `Body._watch_news` 的拦截原因：给 `_watch_news` 加一个返回值 `list[str]`，被拦的原因；不另写一套判断）
    - `skip {seconds}`（不是有限正数 → `ValueError`，挡 `nan` / `inf`）/ `time {at}` → 先记一条 `energy`（`mind_log.energy(sim.wall(), body.energy_now())`），`sim.skip` / `sim.set_time`，再记一条；记 `event`："── 快进 1 小时 ──" / "── 拨到 23:30 ──"；拨不回去 → `{"ok": False, "text": "不能往回拨：…"}`
    - `reflect` → `reflector.running` 为真时 `{"ok": False, "text": "正在反思"}`；否则强制开始一次（`reflector.start(body.reflect_materials(False), sim.clock())`）
    - 未知 op / 缺参数 → `ValueError`（HTTP 层转 400）
  - `idle() -> bool`（Global Constraints 的定义；"持续 ≥ 2 秒"用真实 `time.monotonic`）
  - `state(after: int, timeout: float) -> dict`：等到 `transcript.version > after` 或超时，返回 `{"version", "wall", "clock_text"（如 "9月30日 周三 23:30"）, "energy": {level, score, note}, "friends", "strangers", "place", "scene", "lines": transcript.since(after), "idle", "limit": 额度用完时的一句话 | ""}`；`limit` 来自新加的 `Brain.limit_left(now: float) -> float`（最近一次失败是额度用完时，离 `backoff_until` 还有几秒，否则 0；`_failed` 里记下 `self.limited = limit`，`_ok` 清掉），文字"额度用完，约 N 分钟后再试"
  - 反思结果一句话进聊天记录：`MindLog.reflect` 之后若 `changes` 非空，记 `event`："── 反思：" + "；".join(changes) + " ──"（在 `SandboxControl` 里订阅：给 `MindLog` 加 `on_add: Callable[[dict], None] | None`）
  - 说话被护栏拦下：`Body.say` 返回的拦截文字记 `blocked`（给 `SandboxSender` 之外再挂一个 `Body.on_blocked: Callable[[str, str], None] | None`，`say` 被主动护栏 / 过滤拦下时调 `(原话, 原因)`；`None` 时行为不变）

- [ ] **Step 1: 写失败的测试**（真 `Body` + 假 `Brain`（只有 `in_turn`）+ `Reflector(threaded=False, llm=FakeLlm)`；身体线程用 `threading.Thread(target=body.run)` 起、测完 `stop`）
  - `say` → 聊天记录有 `heard`；`come` → `arrive` 事件 + `event` 行
  - `skip 3600` → `sim.wall()` 多了 3600、`mind_log` 多了两条 `energy`；`time "08:00"` 在 23:00 → 拨到明天 8 点；往回拨的完整时间 → `ok False`
  - `reflect` 两次连着发，第二次（反思还在跑）→ "正在反思"
  - `notice` 在没熟人时 → `blocked` 行带原因
  - `idle()`：事件队列有东西 → 假；`brain.in_turn` → 假；都空且过了 2 秒 → 真
  - `state(after=v, timeout=0.1)` 没新行时 0.1 秒后返回、`lines == []`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_sandbox_control.py -q`
- [ ] **Step 3: 实现**（`Body` 只加 `on_blocked` 钩子和 `_watch_news` 的返回值，`MindLog` 只加 `on_add`；现有测试不改）
- [ ] **Step 4: 跑测试确认通过**，再全量 `python -m pytest -q`
- [ ] **Step 5: 提交**：`git commit -m "feat(sandbox): 沙盒操作（冒充发言、来去、快进、新鲜事、反思）和长轮询状态"`

---

### Task 5: 沙盒 HTTP 接口 + `sandbox` 命令

**Files:**
- Create: `src/skydango/sandbox/server.py`
- Modify: `src/skydango/vision/viewer.py`（把 `/brain`、`/inner`、`/inner/forget`、`/status`、`/shutdown` 的处理抽成模块函数，viewer 和沙盒共用）
- Modify: `src/skydango/cli.py`（`cmd_sandbox` + 子命令参数）
- Test: `tests/test_sandbox_server.py`、`tests/test_cli_sandbox.py`

**Interfaces:**
- Consumes: Task 2 `_run_brain(on_ready=…)`、Task 3 `sandbox_world`、Task 4 `SandboxControl`、前置计划 `inner/api.py`
- Produces:
  - `class SandboxServer(port: int, on_shutdown: Callable[[], None])`：`control: SandboxControl | None`、`trace: BrainTrace | None`、`inner` / `forget`（同 viewer 的两个可调用属性）；`start() -> str`、`stop()`
  - 路由：`GET /status`（`{"ok": True, "kind": "sandbox"}`，身体没建好时也回）、`GET /state?after=`、`GET /brain?after=`、`GET /inner`、`POST /op`、`POST /inner/forget`、`POST /shutdown`；`control` 还是 `None` 时 `/state`、`/op` 回 503 `{"ok": false, "text": "沙盒还在启动"}`
  - `python -m skydango [-c 配置] sandbox --port 19392 [--no-browser] [--parent-pid PID] [--start resume|sleep|HH:MM|"YYYY-MM-DD HH:MM"]`：
    `cfg.reply.dry_run = False`、`cfg.reply.memory_dir = <dir>/memory`（目录不存在就建空的）、`cfg.run.save_frames = False`（灰图不用存）；`[sandbox] wake_hour` 不在 0~23 → 启动报错、`floor_time` + `resolve_start` 算出 offset 建 `SimClock`；`RunDir.create(cfg, "sandbox")`；先起 `SandboxServer`（`/status` 立刻可用，管理面板据此判断起来了），`_run_brain(cfg, run, sandbox_world(…), viewer=None, on_ready=挂 control / trace / inner / forget)`；`finally` 里 `save(clock.json, sim.wall())`、`server.stop()`；看门狗和 `/shutdown` 共用 `watchdog.once(_thread.interrupt_main)`（同 `cmd_run`）
  - `BrainTrace`：`_run_brain` 在 `viewer is None` 时不建 trace——`cmd_sandbox` 通过 `on_ready` 之前传一个 `trace=` 参数进去（给 `_run_brain` 加 `trace: BrainTrace | None = None`，有就用它）

- [ ] **Step 1: 写失败的测试**
  - `test_sandbox_server.py`：`/status` 在 `control is None` 时 200；`/op` 在 `control is None` 时 503；非本机 Host 的 GET `/state` → 403；缺头 / 非 JSON / 超 64 KB 的 POST → 403 / 403 / 413；`/op` 未知 op → 400；`/shutdown` 调 `on_shutdown`
  - `test_cli_sandbox.py`（端到端，照 `tests/test_cli_brain.py` 的 `fake_brain_run` 用 `fake_claude.py`）：准备 `memory/`（有 `friends.md`、`history.jsonl`、`inner/days.jsonl`）并算出所有文件的 sha256；`cfg.sandbox.dir = tmp_path / "sandbox"`；在后台线程跑 `cli.cmd_sandbox(cfg, args)`，等 `/status` 通，POST `/op` `say 小明 在吗`，轮询 `/state` 直到出现 `said` 行（fake claude 会调 `say`；超时 30 秒），POST `/shutdown`，等线程结束；断言：`sandbox/memory/inner/days.jsonl` 多了一行、`sandbox/clock.json` 存在、**`memory/` 下所有文件的 sha256 不变**（Review Focus 5）
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_sandbox_server.py tests/test_cli_sandbox.py -q`
- [ ] **Step 3: 实现**（工具因为部件是 `None` 被拒时——互动请求、好友树、面板、镜头——沙盒下文案换成"沙盒里没有这个"：`ToolBox` / `Body` 按 `World.name == "sandbox"` 换一句，别处照旧；`fake_claude.py` 如果还不会调 `say`，给它加一个 `FAKE_CLAUDE_MODE=say` 模式：收到带"在吗"的消息就调一次 `say` 工具）
- [ ] **Step 4: 跑测试确认通过**，再全量
- [ ] **Step 5: 提交**：`git commit -m "feat(sandbox): sandbox 子进程命令和 JSON 接口"`

---

### Task 6: 管理面板：起停互斥、转发、重置、内心页切换

**Files:**
- Modify: `src/skydango/console/runner.py`（`kind`、按 kind 的端口）
- Create: `src/skydango/console/sandbox_view.py`
- Modify: `src/skydango/console/server.py`（`/sandbox/*` 转发、`/api/sandbox/*`、`/api/inner?source=sandbox`）
- Test: `tests/test_console_runner.py`、`tests/test_console_sandbox.py`

**Interfaces:**
- Consumes: 前置计划 `inner_state` / `forget_offline`；Task 1 `floor_time`、`load_saved`
- Produces:
  - `Runner.start(cmd, env, options, kind: str = "dango", port: int | None = None)`：`port` 默认 `child_port`；探活 / 关停用这次的 `port`；`status()` 多 `"kind"`、`"port"`；已经有一个在 `starting` / `running` / `stopping` → `RuntimeError("团子在运行，先停团子" | "沙盒在运行，先停沙盒")`
  - `build_sandbox_command(config_path, port, parent_pid, start: str, python=sys.executable) -> list[str]`
  - `sandbox_view.py`：`reset(sandbox_dir: Path, memory_dir: Path) -> None`（删 `sandbox/memory/`、`clock.json`，复制 `memory/`，跳过 `memory/archive/`；`memory/` 不存在就建空目录）、`friends(sandbox_dir: Path) -> list[str]`（沙盒 `friends.md` 的 `## 标题`，用 `MemoryStore.friend_names`）、`start_info(sandbox_dir: Path, now: float) -> dict`（`{"floor": floor_time, "saved": load_saved, "floor_text": "9月30日 22:00" | ""}`）
  - `ConsoleServer`：`POST /api/sandbox/start {"start": …}`（预检同团子的令牌 / mcp 两项，不查设备；另一种在跑 → 409；19392 上有上次留下的沙盒 → 409 + `orphan`）、`POST /api/sandbox/stop`、`POST /api/sandbox/reset`（沙盒不是停着 → 409 "先下线沙盒"）、`GET /api/sandbox/info`（`start_info` + `friends` + `owner_name`）、`GET /sandbox/<rest>` 转发、`POST /sandbox/op` 和 `POST /sandbox/inner/forget` 转发；`GET /api/inner?source=sandbox` 用 `inner_state(sandbox/memory/inner, …, live=proxy 到 /sandbox/inner)`、`POST /api/inner/forget` 带 `source` 同理；团子的 `/api/run/start` 在沙盒跑着时也拒绝
  - 转发按 runner 当前的 `kind`：`/live/*` 只在 `kind == "dango"` 时转，`/sandbox/*` 只在 `kind == "sandbox"` 时转，否则 503

- [ ] **Step 1: 写失败的测试**
  - runner：`kind="sandbox", port=19392` 起了之后再起 `dango` → `RuntimeError` 带"沙盒在运行"；`status()["kind"] == "sandbox"`（用 `tests/fake_child.py` 那套）
  - 面板：团子 running 时 `POST /api/sandbox/start` → 409 且 problems 里有"先停团子"（Review Focus 3，两个方向都测）；沙盒 running 时 `/live/status` → 503、`/sandbox/status` 转发成功
  - `reset`：`memory/archive/x` 不复制、其余文件内容一致、旧的 `sandbox/memory/多余.md` 被删、`clock.json` 被删；沙盒 running 时 → 409
  - `/api/inner?source=sandbox` 读的是 `sandbox/memory/inner/`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_console_runner.py tests/test_console_sandbox.py -q`
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**，再全量
- [ ] **Step 5: 提交**：`git commit -m "feat(console): 沙盒起停（和团子互斥）、转发、重置记忆、内心页读沙盒"`

---

### Task 7: 大脑时间线抽成共用脚本

**Files:**
- Create: `src/skydango/vision/static/brain_trace.js`、`src/skydango/vision/static/brain_trace.css`
- Modify: `src/skydango/vision/viewer.py`（`PAGE` 里的时间线 JS / CSS 换成 `<link>` / `<script src>`；新路由 `GET /brain_trace.js`、`GET /brain_trace.css`）
- Modify: `src/skydango/console/server.py`（`GET /static/brain_trace.js`、`/static/brain_trace.css`，读同一个文件）
- Modify: `pyproject.toml`（`[tool.setuptools.package-data]` 的 `skydango` 列表加 `"vision/static/*.js"`、`"vision/static/*.css"`）
- Test: `tests/test_viewer.py`、`tests/test_console_server.py`（追加）

**Interfaces:**
- Produces：`brain_trace.js` 定义全局 `mountBrainTrace(root: HTMLElement, url: string) -> {stop()}`：在 `root` 里画出现在 viewer 的那套（头部状态、"只看做了事的轮次"、轮次列表），对 `url + "?after=N"` 长轮询；行为和现在逐项一样（展开规则、复制一整轮、折叠长文本）

- [ ] **Step 1: 写失败的测试**：viewer `GET /brain_trace.js` 200、内容含 `function mountBrainTrace`；`PAGE` 里不再有 `const B={turns:` 这段内联代码、有 `brain_trace.js`；面板 `GET /static/brain_trace.js` 200；用现有 `test_page_script_parses` 的办法让 node 解析 `brain_trace.js`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 搬代码**（从 `viewer.py` 的 `PAGE` 里 `// ---- brain ----` 起的那段，Python 字符串里的 `\\n` 转回 `\n`）
- [ ] **Step 4: 跑测试确认通过**；用内置浏览器开一次 `python -m skydango view --images <任一目录> --no-browser` 确认页面没报错（view 没有大脑时间线：只确认脚本加载不出错）
- [ ] **Step 5: 提交**：`git commit -m "refactor(viewer): 大脑时间线抽成共用的 brain_trace.js"`

---

### Task 8: 「沙盒」页

**Files:**
- Modify: `src/skydango/console/static/console.html`
- Test: `tests/test_console_server.py`（追加）

**Interfaces:**
- Consumes: Task 6 的 `/api/sandbox/*`、`/sandbox/state`、`/sandbox/op`、`/sandbox/brain`；Task 7 `mountBrainTrace`；前置计划的 `renderInnerNow`、`loadInner`（加 `source = "sandbox"`）
- Produces：容器 id `tab-sandbox`、`sb-start`、`sb-clock`、`sb-chat`、`sb-say`、`sb-now`、`sb-nearby`、`sb-scene`、`sb-scenario`、`sb-brain`；「内心」页顶上加 团子 / 沙盒 切换（`inner-source-toggle`）

- [ ] **Step 1: 写失败的测试**：导航里有 `data-tab="sandbox"`（在「内心」后面）；上面的 id 都在；页面引用 `static/brain_trace.js`；脚本能被 node 解析
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 用 `frontend-design` skill 做排版，照 spec §5 实现**：
  - 没在跑：启动选项（接着上次 / 睡一晚 / 自定义；显示 `floor_text`"最早只能从 9月30日 22:00 开始"）+ [启动]；团子在跑时置灰并提示
  - 在跑：顶部状态 + 「下线（写日记）」+ 沙盒时间 / 精力 + [+10分钟] [+1小时] [到明早 9 点]（`time` op，`at` = `"09:00"`）[设成 __:__]
  - 聊天记录：`heard` 左、`said` 右、`act` / `event` 居中灰字、`blocked` 删除线 + 原因；说话人下拉 = 好友 + "陌生人（自己填名字）" + 主人
  - 右栏：`renderInnerNow`（每 5 秒取 `api/inner?source=sandbox` 的 `now`）+ "看完整内心 →"（切到「内心」页并选中沙盒）；身边、场景、新鲜事
  - 「重置记忆」：只在停着时可点，`confirm` 文案照 spec
  - 剧本区先放容器和"（下一步做）"占位，B 部分接上
  - 底部 `mountBrainTrace($("sb-brain"), "sandbox/brain")`
- [ ] **Step 4: 跑测试确认通过**；然后**真跑一次**：用内置浏览器开 `python -m skydango console --no-browser`，沙盒页重置记忆 → 启动 → 冒充好友说两句 → 快进到 23:30 → 下线；截图核对聊天记录、"现在"、时间线都在动，`memory/` 没变（`git status` 看不到、比一下修改时间）
- [ ] **Step 5: 提交**：`git commit -m "feat(console): 沙盒页（冒充发言、来去、场景、快进、现在、大脑时间线）"`

---

## B 部分：剧本

### Task 9: 剧本格式（读、校验、写）

**Files:**
- Create: `src/skydango/console/scenario.py`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: Task 1 `parse_duration`
- Produces:
  - `@dataclass Start(memory: str = "reset", time: str = "", nearby: list[str], strangers: int = 0, place: str = "", scene: str = "")`（`memory` ∈ `reset` / `keep`）
  - `@dataclass Step(action: str, value, who: str = "", expect: str = "", wait: float = 0.0)`（`action` 是 Global Constraints 里的步骤键之一；`value` 是那个键的值）
  - `@dataclass Scenario(name: str, note: str, start: Start, steps: list[Step])`
  - `load(path: Path) -> Scenario`：错误 → `ScenarioError(f"第 {i} 步：…")`（不认识的键、一步里两个动作、`say` 没 `who`、时长 / 时间格式不对、`online` 前面没有 `offline`、`offline` 后面紧跟的不是 `online` 也不是结尾）
  - `dumps(s: Scenario) -> str`：写成 spec §6 那种格式（`[start]` + `[[steps]]`，字符串用 `tomlfile._string` 转义）；`load(dumps(s)) == s`
  - `to_op(step: Step) -> dict | None`：`say` / `come` / … → `/sandbox/op` 的请求；`offline` / `online` → `None`（由回放引擎处理）

- [ ] **Step 1: 写失败的测试**：spec §6 的"放鸽子"示例能读、`dumps` 再 `load` 一样；每种错误各一条（断言信息里有"第 N 步"）；`to_op` 各一条
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_scenario.py -q`
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交**：`git commit -m "feat(console): 沙盒剧本格式（读、校验、写）"`

---

### Task 10: 录制、回放、报告

**Files:**
- Create: `src/skydango/console/replay.py`
- Modify: `src/skydango/console/server.py`（转发 `/sandbox/op` 时交给录制器；`/api/sandbox/start` / `stop` 也记）
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: Task 9；Task 6 的起停 / 转发
- Produces:
  - `class Recorder`：`begin(start: Start) -> None`（沙盒启动时，带启动选项和当时的身边 / 场景）、`record(op: dict) -> None`、`offline()`、`online(start: str)`、`new()`、`snapshot(name: str, note: str) -> Scenario`；重置记忆时 `new()`
  - `class Replayer(api, scenario: Scenario, report_dir: Path, step_timeout: float, stop_timeout: float, sleep=time.sleep, clock=time.monotonic)`：`api` 是一个小接口（`start(start_choice) -> None`、`stop() -> None`、`reset() -> None`、`running() -> bool`、`op(req) -> dict`、`state(after) -> dict`），面板里用真实实现、测试里用假的
    - `run() -> Path`（报告路径）：沙盒在跑先 `stop()` 等退出；`memory == "reset"` 就 `reset()`；`start(time)`；按 `[start]` 发 `come` / `strangers` / `place` / `scene`；逐步 `op` → 等 `state()["idle"]`（`step_timeout` 超时记"超时"）→ 再睡 `wait` → 收集这段的新聊天记录行；`offline` → `stop()` 并等 `running()` 为假（最多 `stop_timeout`）；`online` → `start(value)`
    - `stop_requested()`：设标志，当前步做完就停（Review Focus 4）
    - `progress() -> dict`：`{"running", "step", "total", "name"}`
  - 报告 `sandbox/reports/<剧本名>-<YYYYmmdd-HHMMSS>.md`：标题、`note`、每步一节（"第 N 步：小明说「在吗」"、这步之后的聊天记录行、这段时间里的 `event` 反思行、`expect`、"超时"标记），末尾"最后的心里"（`/sandbox/inner` 的 `now`）和"这次写的日记"（`sandbox/memory/inner/diary.md` 回放期间新增的段落）
  - `ConsoleServer`：`GET /api/sandbox/scenarios`（`sandbox/scenarios/*.toml` 的名字和 `note`，读不了的带错误）、`POST /api/sandbox/save {"name", "note"}`（`Recorder.snapshot` → 写文件，名字只许中英文数字和 `-_`，已存在要 `overwrite: true`）、`POST /api/sandbox/replay {"name"}`（后台线程跑 `Replayer`，已经在回放 → 409）、`POST /api/sandbox/replay/stop`、`GET /api/sandbox/replay`（`progress` + 最近一份报告的路径）；回放中页面的手动 `/sandbox/op` 请求 → 409 "正在回放"

- [ ] **Step 1: 写失败的测试**（假 api：`state()` 按脚本返回 idle / 不 idle 和新行）
  - 顺序：发出的 op 顺序和剧本一致；`[start]` 的身边 / 场景先发
  - 等 idle：假 api 前 3 次 `idle=False` 第 4 次 `True` → 第 4 次之后才发下一步；一直不 idle → 报告里这步有"超时"、继续下一步
  - `offline` → `online`：`stop` 之后等 `running()` 变假，再 `start("sleep")`
  - 停止回放：第 2 步时 `stop_requested()` → 第 2 步做完就结束，报告只有 2 步
  - 录制：`begin` + 三次 `record` + `offline` + `online("sleep")` → `snapshot` 出来的 `dumps` 能被 `load`，`memory = "keep"` 时报告 / 返回里带"依赖沙盒当时的记忆"提示
  - 回放中手动 `/sandbox/op` → 409
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest tests/test_replay.py -q`
- [ ] **Step 3: 实现**（顺带补 `scenario.py` 的校验：`skip` / `wait` 不是有限数、`wait` 为负 → `ScenarioError`，线 2 终审留下的）
- [ ] **Step 4: 跑测试确认通过**，再全量
- [ ] **Step 5: 提交**：`git commit -m "feat(console): 沙盒剧本录制、回放和报告"`

---

### Task 11: 剧本区页面、示例剧本、文档

**Files:**
- Modify: `src/skydango/console/static/console.html`（`sb-scenario` 区：录制状态、[新录制]、[另存为…]、剧本下拉 + [回放] / [停止回放]、进度、最近报告路径；回放中手动操作置灰）
- Create: `docs/sandbox-scenarios/放鸽子.toml`、`深夜犯困.toml`、`第二天上线.toml`、`README.md`（人名占位 `小明` / `阿花`，说明用前换成自己的好友名、复制到 `sandbox/scenarios/`）
- Modify: `CLAUDE.md`（新增「大脑沙盒」一节、代码结构表加 `src/skydango/sandbox/` 和 `brain/world.py`、常用命令加 `sandbox`）、spec 状态改"代码已完成"
- Test: `tests/test_console_server.py`（剧本区控件 id）、`tests/test_scenario.py`（`docs/sandbox-scenarios/*.toml` 都能 `load`）

- [ ] **Step 1: 写失败的测试**
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（页面部分继续用 `frontend-design` skill 的排版方向）
- [ ] **Step 4: 验证**：全量 `python -m pytest -q`；然后**真跑 spec「验证」四步**（内置浏览器 + 真 Claude）：把"放鸽子"里的 `小明` 换成本机好友名复制进 `sandbox/scenarios/` 回放，打开报告、「内心」页切沙盒看流水账；下线后"睡一晚"上线看「日子」带日记；最后确认 `memory/` 没被动过。截图给用户
- [ ] **Step 5: 提交并合进 main、推送**：`git commit -m "feat(console): 沙盒剧本区、示例剧本；docs: 大脑沙盒"`，按 CLAUDE.md 合并进 main 并 `git push`
