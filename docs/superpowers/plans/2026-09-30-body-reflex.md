# 身体反射 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 身体对常见时刻当场反应：有人跟团子说话马上冒"正在输入"气泡（偶尔先点个头），别人挥手回礼，闲着做小动作；大脑随后补上说什么。

**Architecture:** 纯决策放新文件 `brain/reflex.py`（判断"在跟团子说话"、抽概率、额度、闲着计时），身体（`brain/body.py`）每圈问它、自己执行。
输入框复用 `ChatSender.open()` / `cancel()`；"任何按键前先关框"放在 `Body.clear_view()` 这个所有按键类操作都经过的口子上。
大脑这一轮什么时候结束由 `BrainLoop.last_turn` 告诉身体。

**Tech Stack:** Python 3.11+，pytest（`tests/conftest.py` 的 `FakeDevice`、`Clock`）

**Spec:** `docs/superpowers/specs/2026-09-30-body-reflex-design.md`

## Global Constraints

- 总开关 `[reflex] enabled = false` 时行为和现在**完全一样**：不开框、不做反射动作、`gesture` 事件照旧交给大脑、提示词不变、大脑 `emote` 不查 `min_gap`
- dry-run（`cfg.reply.dry_run` 且不是手动控制）：不开输入框、反射动作走 `EmotePlayer.pretend(name, reflex=True)` 不按键，只写日志
- 反射只用轮盘上已有的动作（`EmotePlayer.on_wheel()`），从不换轮盘
- 反射**不占**大脑的动作冷却（`emotes.min_interval`）；任何两个动作之间至少 `reflex.min_gap` 秒
- 只关团子自己开的框（`sender.cancel()`，不调 `ime_shown()`）
- 做完反射放 `reflex` 背景事件（加进 `brain/events.py` 的 `BACKGROUND`），不单独叫醒大脑
- 默认值（spec「配置」）：`enabled=true`、`bubble=true`、`followup_window=30`、`bubble_max=45`、`addressed=[]`、`addressed_chance=0.3`、`return_map={}`、`return_chance=0.7`、`idle=[]`、`idle_min=180`、`idle_max=420`、`quota_window=600`、`quota=4`、`min_gap=4`
- "叫到了团子"的名字用现有的 `[proactive] self_names`，不另加配置
- 注释、日志、事件文字、提示词用中文，风格照 `occasion.py` / `peek.py`

## Review Focus

1. **track 在跑时来了消息**：框开着按方向键不会转镜头 → track 期间不开框（Task 4 测 `test_no_bubble_while_skill_active`）
2. **开框时大脑已经在想上一批**：那一轮结束时不能把新框关掉，要等"开框之后才开始"的那一轮结束（Task 4 测 `test_bubble_survives_round_that_started_before_it`）
3. **一批里好几句都在叫团子**：只开一次框、最多一个小动作（Task 5 测 `test_one_nod_per_batch`）
4. **大脑离线（备用回复接管）**：不开框，备用回复自己开（Task 4 测 `test_no_bubble_when_brain_offline`）
5. **牵着手的人对团子挥手**：现有代码跳过牵手对象的 gesture，反射也不回（Task 5 测 `test_no_return_to_holding_partner`）

---

### Task 1: `[reflex]` 配置和纯决策模块

**Files:**
- Modify: `src/skydango/config.py`（`ProactiveConfig` 后面加 `ReflexConfig`；`Config` 加 `reflex` 字段）
- Modify: `config.example.toml`（`[proactive]` 后面加 `[reflex]` 一节，清单写成注释好的例子：`# idle = ["伸懒腰", "坐下"]`、`# addressed = ["点头"]`、`# [reflex.return_map]` / `# wave = "挥手"` / `# bow = "鞠躬"`）
- Create: `src/skydango/brain/reflex.py`
- Test: `tests/test_brain_reflex.py`、`tests/test_config.py`

**Interfaces:**
- Produces:
  - `ReflexConfig`（dataclass，字段和默认值见 Global Constraints；`addressed` / `idle: list[str]`，`return_map: dict[str, str]`）
  - `addressed(speaker: str, text: str, *, is_friend: Callable[[str], bool], self_names: Sequence[str], nearby: Sequence[str], since_said: float | None, followup_window: float, owner: str = "") -> bool`
  - `class Reflexes(cfg: ReflexConfig, rng: random.Random, now: float)`：
    - `pick_addressed(now: float, on_wheel: Sequence[str]) -> str | None`
    - `pick_return(now: float, label: str, on_wheel: Sequence[str]) -> str | None`
    - `pick_idle(now: float, on_wheel: Sequence[str]) -> str | None`
    - `stir(now: float) -> None`（有动静：闲着计时重新抽一个 `idle_min`~`idle_max` 的截止时间）
    - `done(now: float, text: str) -> None`（记一次额度 + `stir` + 进 `recent`）
    - `left(now: float) -> int`（窗口内还能做几个）
    - `recent: deque[tuple[float, str]]`（maxlen 10，status / 网页用）
    - `usable(names: Sequence[str], on_wheel: Sequence[str]) -> list[str]`（清单里在轮盘上的）

- [ ] **Step 1: 写失败的测试** `tests/test_brain_reflex.py`

```python
import random
from skydango.brain.reflex import Reflexes, addressed
from skydango.config import Config, ReflexConfig

FRIENDS = lambda s: s in ("小明", "阿花")
KW = dict(is_friend=FRIENDS, self_names=["团子", "三彩"], followup_window=30.0)

def test_addressed_by_name_followup_or_only_friend():
    assert addressed("小明", "团子在吗", nearby=["小明", "阿花"], since_said=None, **KW)
    assert addressed("小明", "哈哈", nearby=["小明", "阿花"], since_said=12.0, **KW)
    assert addressed("小明", "哈哈", nearby=["小明"], since_said=None, **KW)

def test_not_addressed():
    assert not addressed("小明", "哈哈", nearby=["小明", "阿花"], since_said=31.0, **KW)  # 过了窗口
    assert not addressed("路人甲", "团子", nearby=[], since_said=None, **KW)  # 陌生人
    assert not addressed("我", "团子", nearby=["小明"], since_said=None, **KW)  # 自己
    assert not addressed("小明", "#过来", nearby=["小明"], since_said=None, owner="小明", **KW)  # 主人命令

class Rng(random.Random):
    def __init__(self, r):
        super().__init__(0); self.r = r
    def random(self):
        return self.r

def make(r=0.0, **kw):
    cfg = ReflexConfig(addressed=["点头"], idle=["伸懒腰", "坐下"], return_map={"wave": "挥手"}, **kw)
    return Reflexes(cfg, Rng(r), now=0.0)

def test_chances_and_wheel():
    assert make(0.29).pick_addressed(0.0, ["点头"]) == "点头"
    assert make(0.31).pick_addressed(0.0, ["点头"]) is None  # addressed_chance 0.3
    assert make(0.0).pick_addressed(0.0, ["鞠躬"]) is None  # 不在轮盘上
    assert make(0.69).pick_return(0.0, "wave", ["挥手"]) == "挥手"
    assert make(0.71).pick_return(0.0, "wave", ["挥手"]) is None  # return_chance 0.7
    assert make(0.0).pick_return(0.0, "bow", ["挥手"]) is None  # return_map 里没有

def test_quota():
    rx = make(0.0, quota=2)
    rx.done(0.0, "a"); rx.done(1.0, "b")
    assert rx.left(2.0) == 0 and rx.pick_addressed(2.0, ["点头"]) is None
    assert rx.left(601.0) == 1  # quota_window 600

def test_idle_waits_random_interval_and_avoids_repeat():
    rx = make(0.0)  # idle_min 180：Rng 0.0 → 截止 = stir 时刻 + 180
    assert rx.pick_idle(179.0, ["伸懒腰", "坐下"]) is None
    first = rx.pick_idle(180.0, ["伸懒腰", "坐下"])
    rx.done(180.0, first)
    assert rx.pick_idle(360.0, ["伸懒腰", "坐下"]) not in (None, first)
    rx.stir(400.0)
    assert rx.pick_idle(579.0, ["伸懒腰", "坐下"]) is None

def test_example_config_has_reflex():
    from pathlib import Path
    from skydango.config import load_config
    cfg = load_config(Path(__file__).parent.parent / "config.example.toml")
    assert cfg.reflex.enabled is True and cfg.reflex.idle == [] and Config().reflex.min_gap == 4.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_reflex.py`
Expected: FAIL，`ImportError: cannot import name 'Reflexes'`

- [ ] **Step 3: 实现**
  - `ReflexConfig` 放进 `config.py`，docstring 写"身体反射（见 docs/superpowers/specs/2026-09-30-body-reflex-design.md）"，每个字段行尾中文注释
  - `addressed()`：说话人是 `ME`（`occasion.ME`，"我"）或空 → False；`owner` 且说话人是 owner 且以 `#` 开头 → False；不是好友 → False；然后三条任一：文字含 `self_names` 任一；`since_said is not None and since_said <= followup_window`；`len(nearby) == 1 and similar(nearby[0], speaker, 0.75)`（`chat.tracker.similar`）
  - `Reflexes`：概率用 `rng.random() < chance`；闲着截止 = `stir 时刻 + idle_min + rng.random() * (idle_max - idle_min)`；`pick_idle` 在 `usable` 里去掉上一次的动作（只剩一个时允许重复），用 `rng.choice`；所有 `pick_*` 在 `left(now) == 0` 时返回 None
- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_brain_reflex.py tests/test_config.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py config.example.toml src/skydango/brain/reflex.py tests/test_brain_reflex.py
git commit -m "feat(reflex): [reflex] 配置和纯决策（在跟团子说话、概率、额度、闲着计时）"
```

### Task 2: 反射动作不占大脑冷却（`EmotePlayer`）

**Files:**
- Modify: `src/skydango/game/emotes.py:77-95`（`perform`、`pretend`）
- Test: `tests/test_emotes.py`

**Interfaces:**
- Produces: `EmotePlayer.perform(name: str, reflex: bool = False) -> int`、`EmotePlayer.pretend(name: str, reflex: bool = False) -> None`、属性 `EmotePlayer.last_any: float`（任何动作——反射或大脑——最近一次的时间，初值 `-inf`）

- [ ] **Step 1: 写失败的测试**（用现有 `make_player`）

```python
def test_reflex_emote_does_not_start_brain_cooldown(library):
    player, device, t, _ = make_player(library)
    player.start()
    player.perform("鞠躬", reflex=True)
    assert player.last_any == 100.0 and player.available() == ["鞠躬", "欢呼", "指向"]  # 大脑照样能做
    player.pretend("欢呼", reflex=True)
    assert player.last_emote == float("-inf") and player.last_any == 100.0
    player.perform("鞠躬")
    assert player.available() == [] and player.last_any == 100.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_emotes.py::test_reflex_emote_does_not_start_brain_cooldown`
Expected: FAIL，`TypeError: ... unexpected keyword argument 'reflex'`

- [ ] **Step 3: 实现**：两个方法都更新 `last_any`；`reflex=True` 时不更新 `last_emote`。反射不会走换轮盘分支（调用方只给轮盘上的动作），不用特殊处理
- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_emotes.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/game/emotes.py tests/test_emotes.py
git commit -m "feat(emotes): 反射做的动作不占大脑的动作冷却，记下任意动作的时间"
```

### Task 3: 让身体知道框是谁开的、大脑哪一轮结束了

**Files:**
- Modify: `src/skydango/chat/sender.py`（加 `opened` 属性）
- Modify: `src/skydango/brain/loop.py`（`wake` 记 `last_turn`）
- Test: `tests/test_sender.py`（新建，仓库里还没有单独的 sender 测试）、`tests/test_brain_loop.py`

**Interfaces:**
- Produces:
  - `ChatSender.opened -> bool`（只读 property，返回 `_opened`：框是我们 `open()` 开的、还没发 / 没 `cancel()`）
  - `Brain.last_turn: tuple[float, float]`（最近一轮完整结束的 (开始, 结束) 时间，用 `self.clock`；初值 `(-inf, -inf)`；失败的轮次也记）

- [ ] **Step 1: 写失败的测试**

```python
def test_last_turn_records_start_and_end(clock):
    class Slow(FakeSession):  # 本测试里定义：send 时时间走 5 秒
        def send(self, text):
            clock.advance(5.0)
            return super().send(text)
    brain, _, _, _ = make(clock, Slow())
    assert brain.last_turn == (float("-inf"), float("-inf"))
    start = clock()
    brain.wake(start, "heartbeat")
    assert brain.last_turn == (start, start + 5.0)

def test_sender_opened_flag():
    dev = FakeDevice([np.zeros((720, 1280, 3), np.uint8)])  # from conftest import FakeDevice
    s = ChatSender(dev, SenderConfig(open_chat_key=28), lambda: (1280, 720), sleep=lambda s: None)
    assert not s.opened
    s.open(); assert s.opened
    s.send("好"); assert not s.opened
    s.open(); s.cancel(); assert not s.opened
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_loop.py::test_last_turn_records_start_and_end tests/test_sender.py::test_sender_opened_flag`
Expected: FAIL，`AttributeError`

- [ ] **Step 3: 实现**：`wake` 开头记 `start = self.clock()`（已有），`finally` 里 `self.last_turn = (start, self.clock())`（和 `chat_turn = False` 放一起）；`ChatSender.opened` 返回 `self._opened`
- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_brain_loop.py tests/test_sender.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/chat/sender.py src/skydango/brain/loop.py tests/
git commit -m "feat(brain): 大脑记下最近一轮的起止时间；sender 暴露框是不是自己开的"
```

### Task 4: 输入气泡（身体开框、关框）

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__`、`step`、`_heard`、`clear_view`、`say`、`panel_press`、`panel_close`、`status`）
- Test: `tests/test_brain_bubble.py`（新建；从 `test_brain_body` 导入 `body`、`msg`、`FakeEnv`）

**Interfaces:**
- Consumes: Task 1 `addressed`、`ReflexConfig`；Task 3 `ChatSender.opened`、`Brain.last_turn`
- Produces（Task 5、6 用）：
  - `Body.brain_turn: Callable[[], tuple[float, float]]`（默认 `lambda: (-inf, -inf)`；cli 设成 `lambda: brain.last_turn`）
  - `Body._bubble_at: float | None`（身体替大脑开框的时间；None = 没开）
  - `Body._open_bubble(now: float) -> bool`、`Body._close_bubble(why: str) -> None`
  - `Body._said_at: float`（团子上次说话的 `clock` 时间，`say` 里记，初值 `-inf`）
  - `Body._addressed(m: Message, now: float) -> bool`（包 Task 1 的 `addressed`，`is_friend` 用 `is_friend_fn(self.friend_names())`、`nearby` 用 `env.nearby(now)`、`owner` 用 `cfg.brain.owner_name`）

行为（spec §2）：
- `_heard` 放完 chat 事件后：`reflex.enabled and reflex.bubble`、不 dry-run、大脑没离线、`self.skills.active is None`、这一批里有 `_addressed` 的消息 → `_open_bubble(now)`（先 `panel.before_speak(now)` 再 `sender.open()`；已经 `sender.opened` 就不重复开）。Task 5 会在开框前插入小动作
- `step` 里每圈 `_watch_bubble(now)`：`_bubble_at` 不为 None 时——`not sender.opened`（`say` 用掉了）→ 清掉；`brain_turn()` 的开始 ≥ `_bubble_at` 且结束 ≥ 开始 → 关（"大脑没说话"）；`now - _bubble_at >= bubble_max` → 关（"超时"）
- `clear_view(action, …)` 开头：`action != "say"` 时 `_close_bubble(action)`；`panel_press` / `panel_close` 开头也调
- dry-run 的 `_heard` 只 `log.info("[dry-run] 会冒输入气泡")`
- `status` 在 `_bubble_at` 不为 None 时加"输入框：开着（身体替你开的，想好就 say）"

- [ ] **Step 1: 写失败的测试**（`live=True`，`b.friend_names = lambda: ["小明"]`，`env.near = ["小明"]`）

```python
def test_opens_bubble_when_addressed(clock):
    b, dev, reader, env = rb(clock)  # 本文件的小工厂：live、好友小明在身边、reflex 默认
    reader.batches = [[msg("团子在吗", speaker="小明")]]
    b.step()
    assert ("hw_key", 28) in dev.calls and b.sender.opened and b._bubble_at == clock()

def test_say_uses_the_open_bubble(clock):
    b, dev, reader, _ = rb(clock); reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    n = dev.calls.count(("hw_key", 28))
    b.say("在呀"); b.step()
    assert dev.calls.count(("hw_key", 28)) == n and b._bubble_at is None

def test_closes_when_round_ends_silent(clock):
    b, dev, reader, _ = rb(clock); reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    opened = clock(); clock.advance(3)
    b.brain_turn = lambda: (opened + 1, opened + 3); b.step()
    assert ("key", KEYCODE_BACK) in dev.calls and not b.sender.opened

def test_bubble_survives_round_that_started_before_it(clock):
    b, dev, reader, _ = rb(clock); reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    b.brain_turn = lambda: (clock() - 5, clock() + 0.5); clock.advance(1); b.step()
    assert b.sender.opened

def test_closes_on_timeout_and_before_keys(clock):
    b, dev, reader, _ = rb(clock); reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    clock.advance(45); b.step()
    assert not b.sender.opened  # bubble_max
    reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    b.clear_view("move", live=True)
    assert not b.sender.opened

def test_request_closes_bubble(clock):  # env.requests 非空 → _watch_requests → clear_view("social")
    ...
    assert not b.sender.opened

def test_no_bubble_when_not_addressed_dry_run_offline_or_disabled(clock): ...
def test_no_bubble_while_skill_active(clock): ...   # b.skills.active = object()
def test_no_bubble_when_brain_offline(clock): ...   # b.brain_offline = lambda now: True
```

后三个各自断言 `("hw_key", 28) not in dev.calls`；disabled 用 `cfg.reflex.enabled = False` 和 `cfg.reflex.bubble = False` 各一次。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_bubble.py`
Expected: FAIL（没有开框）

- [ ] **Step 3: 按上面的行为实现**
- [ ] **Step 4: 跑测试确认通过，旧测试不坏**

Run: `python -m pytest -q tests/test_brain_bubble.py tests/test_brain_body.py tests/test_brain_manual.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_bubble.py
git commit -m "feat(body): 有人跟团子说话马上冒输入气泡，大脑说话时用这个框，没说话 / 超时 / 要按键时关掉"
```

### Task 5: 动作反射（被叫到、回礼、闲着）

**Files:**
- Modify: `src/skydango/brain/body.py`（`__init__` 加 `rng: random.Random | None = None` 参数；`_heard`、`_watch_comings` 的 gesture 段、`step`、`say`、`emote`、`status`、`_show`）
- Modify: `src/skydango/brain/events.py`（`BACKGROUND` 加 `"reflex"`）
- Test: `tests/test_brain_reflex_body.py`（新建）、`tests/test_brain_events.py`

**Interfaces:**
- Consumes: Task 1 `Reflexes`；Task 2 `perform(name, reflex=True)` / `pretend(name, reflex=True)` / `last_any`；Task 4 `_open_bubble` / `_close_bubble` / `_bubble_at`
- Produces: `Body.reflexes: Reflexes`、`Body._reflex_emote(name: str, why: str, now: float) -> bool`

`_reflex_emote` 做不了就返回 False（不抛）：`emotes is None`、`blackout`、`skills.active`、`self._requests` 非空、`now - emotes.last_any < min_gap`、`clear_view("emote")` 抛 `ToolError`。做了：dry-run → `pretend(name, reflex=True)`，否则 `with self._held("wheel"): perform(name, reflex=True)`；`reflexes.done(now, why)`；`events.put("reflex", why)`。

- **被叫到**：Task 4 开框之前，`pick_addressed(now, emotes.on_wheel())` 有结果 → `_reflex_emote(name, f"有人叫你，你下意识{name}", now)`，再开框。一批最多一次
- **回礼**（`_watch_comings`，跳过牵手对象的那段逻辑不动）：`pick_return(now, label, on_wheel)` 有结果且 `_reflex_emote(...)` 成功 → 事件文字"{who}对你{中文名}，你回了个{name}"（`reflex` 事件），**不**再放 `gesture`；否则照旧放 `gesture`。框原来开着（`_bubble_at` 不为 None）→ 回礼后 `_open_bubble(now)` 再开
- **闲着**：`step` 每圈 `_watch_idle(now)`：`_bubble_at` 不为 None 就不做；`pick_idle` 有结果 → `_reflex_emote(name, f"闲着，你{name}", now)`
- **有动静**：`_heard`（有新消息）、`say`、`clear_view`（任何 action）里 `reflexes.stir(now)`
- **大脑的 `emote`**：`reflex.enabled` 时 `now - emotes.last_any < min_gap` → `ToolError("刚做完一个动作，{n:.0f} 秒后再做")`
- `status`：`reflexes.recent` 里 300 秒内最近一条 → "刚才下意识：{text}（{n} 秒前）"；`_show` 的 info 加 `"反射"`

- [ ] **Step 1: 写失败的测试**

本文件的测试辅助（在文件里定义）：
- `ReflexEmotes(dev)`：`on_wheel()` 返回 `["点头", "挥手", "伸懒腰"]`；`available()` 同 `test_brain_body.FakeEmotes`；`perform(name, reflex=False)` 记 `done.append((name, reflex))` 并 `dev.calls.append(("emote", name))`、更新 `last_any`；`pretend(name, reflex=False)` 记 `pretended`、更新 `last_any`
- `GestureEnv(FakeEnv)`：多一个 `gestures` 列表和 `pop_gestures()`（取走并清空）
- `rx(clock, r, **reflex_cfg)`：用 `test_brain_body.body(clock, live=True, env=GestureEnv(), emotes=ReflexEmotes(dev), rng=Rng(r))` 建身体，`cfg.reflex` 按参数改，`friend_names = lambda: ["小明"]`、`env.near = ["小明"]`；返回 `(b, dev, reader, events, emotes)`
- `Rng` 同 Task 1

```python
def test_nod_then_bubble(clock):
    b, dev, reader, events, emotes = rx(clock, r=0.0, addressed=["点头"])
    reader.batches = [[msg("团子", speaker="小明")]]; b.step()
    assert emotes.done == [("点头", True)] and b.sender.opened
    assert dev.calls.index(("emote", "点头")) < dev.calls.index(("hw_key", 28))  # 先动作再开框
    assert [e.kind for e in events.drain()] == ["chat", "reflex"]

def test_one_nod_per_batch(clock):
    ... reader.batches = [[msg("团子", speaker="小明"), msg("团子？", speaker="小明")]]
    assert len(emotes.done) == 1

def test_return_gesture_replaces_event(clock):
    ... env.gestures = [("小明", "wave")]; b.step()
    assert emotes.done == [("挥手", True)] and [e.kind for e in events.drain()] == ["reflex"]

def test_gesture_goes_to_brain_when_not_returned(clock):   # r=0.9 > return_chance
    ... assert [e.kind for e in events.drain()] == ["gesture"]

def test_no_return_to_holding_partner(clock):
    ... b.holding = "小明"; assert emotes.done == [] and events.drain() == []

def test_return_reopens_bubble(clock): ...                  # 框开着 → 回礼后 sender.opened 仍为 True
def test_idle_after_quiet_and_not_while_typing(clock): ...  # advance(179) 没做、advance(1) 做了；框开着不做
def test_min_gap_and_quota(clock): ...                     # last_any 4 秒内不做；quota 用完不做
def test_brain_emote_waits_min_gap_but_not_cooldown(clock): ...
def test_reflex_disabled_changes_nothing(clock): ...       # enabled=False：不动作、gesture 照旧、大脑 emote 不查 min_gap
def test_dry_run_pretends(clock): ...                      # pretend(name, reflex=True)，不按数字键
```

`tests/test_brain_events.py` 的 `BACKGROUND` 断言加 `"reflex"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_reflex_body.py`
Expected: FAIL

- [ ] **Step 3: 按上面的行为实现**
- [ ] **Step 4: 跑全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/events.py tests/
git commit -m "feat(body): 动作反射——被叫到点头、回礼、闲着的小动作，自己的额度、不占大脑冷却"
```

### Task 6: 接上大脑（提示词、cli、管理面板、文档）

**Files:**
- Modify: `src/skydango/brain/prompt.py`（`brain_prompt(..., bubble: bool = False)`）
- Modify: `src/skydango/cli.py:1539`（传 `bubble=cfg.reflex.enabled and cfg.reflex.bubble`）、`:1563` 附近（`body.brain_turn = lambda: brain.last_turn`）
- Modify: `src/skydango/console/settings.py`（`proactive.*` 后面加两条）
- Modify: `CLAUDE.md`（「统管大脑」后面加「身体反射（`[reflex]`）」一节，写明**还没在真机上跑过**、spec 路径、四条真机验证）
- Test: `tests/test_brain_prompt.py`、`tests/test_console_settings.py`（或现有的设置清单测试）

**Interfaces:**
- Consumes: Task 3 `Brain.last_turn`、Task 4 `Body.brain_turn`

- [ ] **Step 1: 写失败的测试**

```python
BUBBLE = "有人跟你说话时，身体已经替你冒了输入气泡"

def test_bubble_note_only_when_enabled():
    assert BUBBLE in brain_prompt(ReplyConfig(), None, bubble=True)
    assert BUBBLE not in brain_prompt(ReplyConfig(), None)
```

设置清单：`"reflex.enabled"` 和 `"reflex.bubble"` 在 `FIELDS` 的 key 里，group 是 `"brain"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_brain_prompt.py`
Expected: FAIL

- [ ] **Step 3: 实现**
  - 提示词：`bubble` 为真时，在「说话（say）」一节第一条后面加一条（整句照抄）：
    `- 有人跟你说话时，身体已经替你冒了输入气泡（对方看到你在打字），不用急：想好就 say；想用动作回应就先 emote 再 say；不想回也行，身体会关掉。`
  - 设置：`Field("reflex.enabled", "身体反射", "有人叫团子马上冒输入气泡、回礼、闲着做小动作；关掉就回到老样子", "bool", "brain")`、
    `Field("reflex.bubble", "替大脑冒输入气泡", "有人跟团子说话时马上打开输入框，头顶显示正在输入", "bool", "brain")`
  - cli 两处接线
  - CLAUDE.md 新一节 + 「代码结构」表里 `brain/` 那行加 `reflex.py` 反射
- [ ] **Step 4: 跑全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/prompt.py src/skydango/cli.py src/skydango/console/settings.py CLAUDE.md tests/
git commit -m "feat(brain): 反射接上大脑（提示词、last_turn、管理面板开关），CLAUDE.md 加身体反射一节"
```

---

完成后：按 CLAUDE.md 合并进 main 并推送；真机验证（spec 最后一节四步）由用户在 Windows + MuMu 上跑，结论再写进 game-ops。
