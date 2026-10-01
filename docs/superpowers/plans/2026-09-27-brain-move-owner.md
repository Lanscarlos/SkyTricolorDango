# 大脑移动能力 + 主人命令模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

> **状态（2026-10-01 核对）**：Task 1~9 已落实（代码后来又改过：主人命令只在真用到放宽时才标、`emote` / `camera_move` / `move` 多了 `live` 参数给手动控制用、提示词「移动（move）」一节重写过）；
> Task 9 的 `AGENTS.md` 现在只指向 CLAUDE.md，改成在 CLAUDE.md 代码结构表里补 `locomotion.py`。**Task 10 真机验证还没做**（步长没标定、A/D 未测）。

**Goal:** 给统管大脑加一个 `move` 工具（小步走，不做自动寻路），再加一个"主人命令模式"——卡洛发的、
以 `#` 开头的聊天消息在 30 秒窗口内放宽 `move`/`emote`/`camera` 的步数、频率、牵手拦截限制。

**Architecture:** 新增 `brain/locomotion.py`（照抄 `camera.py` 的"按键小类"结构），`brain/body.py` 接入
牵手拦截 + 频率限制 + 授权窗口判断，`brain/tools.py` 暴露 `move` 工具，`brain/prompt.py` 更新能力描述和
主人命令说明。授权窗口的判断在 `Body._heard()` 里对 OCR 出来的 `(speaker, text)` 做精确匹配，不经过大脑
的"理解"这一步。

**Tech Stack:** Python 3.13，pytest，现有的 `FakeDevice`/`Clock` 测试夹具（`tests/conftest.py`）。

## Global Constraints

- 精确来源：`docs/superpowers/specs/2026-09-27-brain-move-design.md`（已用户批准）；这个计划里的每处数值
  （键码、默认时长、上限）都要和 spec 一致，spec 和这份计划冲突时以 spec 为准。
- **不放宽**：`brain/prompt.py` 的"底线"和"身份"两节内容，任何任务都不能碰。
- 每个任务改完都要跑一次相关测试文件，全绿才进下一个任务；最后一个任务之前跑一次全量 `pytest -q`。
- 涉及真机验证的步骤（Task 11）需要模拟器开着、游戏在前台，是本计划里唯一不能只靠单元测试完成的部分。

---

## File Structure

| 文件 | 改动 |
|---|---|
| `src/skydango/brain/locomotion.py` | 新建：`Locomotion` 类，WASD 小步移动 |
| `tests/test_brain_locomotion.py` | 新建：`Locomotion` 的单元测试 |
| `src/skydango/brain/camera.py` | 改：`Camera.move()` 加 `max_steps` 参数（默认 `MAX_STEPS`），供主人命令模式覆盖 |
| `tests/test_brain_camera.py` | 改：加一个 `max_steps` 覆盖的测试 |
| `src/skydango/config.py` | 改：`BrainConfig` 加 `move_step`/`move_min_interval`/`owner_name`/`owner_window` |
| `config.example.toml` | 改：`[brain]` 段补新字段的示例和注释 |
| `tests/test_config.py` | 改：`test_brain_section` 断言新字段默认值 |
| `src/skydango/brain/events.py` | 改：`Event.kind` 的注释加 `owner_command` |
| `src/skydango/brain/body.py` | 改：构造函数加 `locomotion` 参数、新状态字段；`_heard()` 识别 `#` 命令；
新增 `move()`；`emote()`/`camera_move()` 接入授权窗口 |
| `tests/test_brain_body.py` | 改：加 `FakeLocomotion`，新增/修改若干测试 |
| `src/skydango/brain/tools.py` | 改：`TOOLS` 加 `move`、`ACTIONS` 加 `"move"`、`ToolBox._bind` 加分支 |
| `tests/test_brain_tools.py` | 改：断言里加 `move`，`FakeBody` 加 `move` 方法，加一个 dispatch 测试 |
| `src/skydango/brain/prompt.py` | 改：移动能力描述替换"不能走"那段；新增 `## 主人命令` 小节 |
| `tests/test_brain_prompt.py` | 改：断言里加 `move`；新增一个断言"主人命令"小节存在的测试 |
| `AGENTS.md` | 改：代码结构表 `brain/` 那行补 `locomotion.py` |

---

### Task 1: `Locomotion` 类

**Files:**
- Create: `src/skydango/brain/locomotion.py`
- Test: `tests/test_brain_locomotion.py`

**Interfaces:**
- Consumes：`tests/conftest.py` 的 `FakeDevice`（`hw_key_down/hw_key_up/key/ime_shown`，见
  `tests/test_brain_camera.py` 里的用法）、`scene()`。
- Produces：`Locomotion(device, step, sleep=time.sleep)`，方法 `move(direction: str, steps: int = 1,
  max_steps: int = MAX_STEPS) -> str`；模块常量 `KEYS: dict[str, int]`、`MAX_STEPS: int = 3`。
  `body.py`（Task 5）、`tests/test_brain_body.py` 的 `FakeLocomotion` 都要用这些名字。

- [x] **Step 1: 写失败的测试**

创建 `tests/test_brain_locomotion.py`：

```python
import pytest
from conftest import FakeDevice, scene

from skydango.brain.locomotion import Locomotion


def loco(shown=False):
    dev = FakeDevice([scene()])
    dev.shown = shown
    return Locomotion(dev, 0.3, sleep=lambda s: None), dev


def test_forward_presses_w_the_right_number_of_times():
    loc, dev = loco()
    assert loc.move("forward", 2) == "前进走了 2 步"
    assert dev.calls == [("hw_down", 17), ("hw_up", 17), ("hw_down", 17), ("hw_up", 17)]


def test_steps_clamped_to_default_max_steps():
    loc, dev = loco()
    loc.move("right", 10)
    assert dev.calls.count(("hw_down", 32)) == 3  # 默认 MAX_STEPS = 3


def test_custom_max_steps_used_for_owner_window():
    loc, dev = loco()
    loc.move("forward", 10, max_steps=6)
    assert dev.calls.count(("hw_down", 17)) == 6


def test_closes_input_box_first():
    loc, dev = loco(shown=True)
    loc.move("back", 1)
    assert dev.calls[0] == ("key", 4)


def test_unknown_direction_rejected():
    loc, _ = loco()
    with pytest.raises(ValueError):
        loc.move("up")
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_locomotion.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'skydango.brain.locomotion'`

- [x] **Step 3: 写实现**

创建 `src/skydango/brain/locomotion.py`：

```python
"""角色移动（光遇的键盘操作，见 game-ops §2）：W/A/S/D 按住一小段时间模拟一小步，没有"复位"这回事。

- 前进 / 后退 / 左右对应 W / A / S / D；聊天记录面板开着也有效，不用像 camera 那样先关面板再开
- 2026-09-27 实测：按住 0.3 s 几乎看不出移动，1 s 幅度很大（从好友群里走到几米外），默认给一个短的单步时长
- 输入框开着时按键会变成打字 → 先按 BACK
- 走出去之后回不去：不维护位移量、不提供 reset()，避免造成"能一键复位"的错误预期
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..device.base import KEYCODE_BACK

KEYS = {"forward": 17, "back": 31, "left": 30, "right": 32}  # Linux 键码：W A S D（A/D 未在真机验证，按对称假设）
NAMES = {"forward": "前进", "back": "后退", "left": "向左", "right": "向右"}
MAX_STEPS = 3


class Locomotion:
    def __init__(
        self,
        device,
        step: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.device = device
        self.step = step
        self.sleep = sleep

    def move(self, direction: str, steps: int = 1, max_steps: int = MAX_STEPS) -> str:
        if direction not in KEYS:
            raise ValueError(f"不认识的移动方向：{direction}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), max_steps))
        if self.device.ime_shown():
            self.device.key(KEYCODE_BACK)
            self.sleep(0.3)
        code = KEYS[direction]
        for _ in range(steps):
            self.device.hw_key_down(code)
            self.sleep(self.step)
            self.device.hw_key_up(code)
            self.sleep(0.2)
        return f"{NAMES[direction]}走了 {steps} 步"
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_locomotion.py -v`
Expected: 5 个测试全部 PASS

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/locomotion.py tests/test_brain_locomotion.py
git commit -m "feat: add Locomotion for small-step WASD movement"
```

---

### Task 2: `Camera.move()` 支持覆盖步数上限

**Files:**
- Modify: `src/skydango/brain/camera.py:39-46`
- Test: `tests/test_brain_camera.py`

**Interfaces:**
- Consumes：`Camera` 现有构造和 `KEYS`/`MAX_STEPS`（不变）。
- Produces：`Camera.move(action, steps=1, max_steps=MAX_STEPS)`——`body.camera_move()`（Task 7）在主人
  命令窗口内会传 `max_steps=MAX_STEPS * 2`。

- [x] **Step 1: 写失败的测试**

在 `tests/test_brain_camera.py` 末尾加：

```python
def test_owner_window_can_raise_the_step_cap():
    c, dev, _ = cam(panel=False)
    c.move("right", 10, max_steps=8)
    assert dev.calls.count(("hw_down", 106)) == 8
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_camera.py::test_owner_window_can_raise_the_step_cap -v`
Expected: FAIL，`TypeError: move() got an unexpected keyword argument 'max_steps'`

- [x] **Step 3: 改实现**

`src/skydango/brain/camera.py` 里把：

```python
    def move(self, action: str, steps: int = 1) -> str:
        if action not in KEYS:
            raise ValueError(f"不认识的视角操作：{action}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), MAX_STEPS))
```

改成：

```python
    def move(self, action: str, steps: int = 1, max_steps: int = MAX_STEPS) -> str:
        if action not in KEYS:
            raise ValueError(f"不认识的视角操作：{action}（可以用 {'、'.join(KEYS)}）")
        steps = max(1, min(int(steps), max_steps))
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_camera.py -v`
Expected: 全部 PASS（含之前已有的用例）

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/camera.py tests/test_brain_camera.py
git commit -m "feat: let Camera.move accept a max_steps override"
```

---

### Task 3: 配置新增字段

**Files:**
- Modify: `src/skydango/config.py:260`（`BrainConfig`，紧跟 `camera_step` 之后）
- Modify: `config.example.toml:73-81`（`[brain]` 段）
- Test: `tests/test_config.py:39-45`（`test_brain_section`）

**Interfaces:**
- Produces：`cfg.brain.move_step: float`、`cfg.brain.move_min_interval: float`、
  `cfg.brain.owner_name: str`、`cfg.brain.owner_window: float`——Task 5/6/7 都要读这几个字段。

- [x] **Step 1: 写失败的测试**

把 `tests/test_config.py` 里的 `test_brain_section` 改成：

```python
def test_brain_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[brain]\nenabled = true\nheartbeat = [30, 60]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.enabled is True and cfg.brain.heartbeat == [30, 60]
    assert cfg.brain.model == "claude-sonnet-5" and cfg.brain.effort == "low"
    assert cfg.brain.image_size == [1280, 720] and cfg.brain.max_steps == 6
    assert cfg.brain.move_step == 0.3 and cfg.brain.move_min_interval == 3.0
    assert cfg.brain.owner_name == "" and cfg.brain.owner_window == 30.0
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_config.py::test_brain_section -v`
Expected: FAIL，`AttributeError: 'BrainConfig' object has no attribute 'move_step'`

- [x] **Step 3: 改实现**

`src/skydango/config.py` 里 `BrainConfig` 的 `camera_step` 那行之后加：

```python
    camera_step: float = 0.25  # 转视角每步按住方向键的秒数（0.5 s 约 90°）
    scene_change: float = 0.25  # 缩略图平均差异（0~1）超过这个算画面大变
    move_step: float = 0.3  # 移动每步按住方向键的秒数（实测 0.3 s 几乎不动、1 s 幅度很大，先取偏小的默认值）
    move_min_interval: float = 3.0  # 两次 move 调用最小间隔（秒），避免连续走位
    owner_name: str = ""  # 卡洛的游戏昵称，精确匹配；留空 = 主人命令模式关闭
    owner_window: float = 30.0  # 收到一条 # 开头的命令后，放宽 move/emote/camera 限制多少秒
```

（`scene_change` 是原来就有的最后一行，只是贴出来对齐插入位置；实际只在它后面新增 4 行。）

- [x] **Step 4: 更新示例配置**

`config.example.toml` 的 `[brain]` 段（`pause_usd_per_hour = 5.0` 那行之后）加：

```toml
move_step = 0.3                # 移动每步按住方向键的秒数（0.3 s 几乎不动、1 s 幅度很大，见 game-ops §2）
move_min_interval = 3.0        # 两次 move 调用最小间隔（秒）
owner_name = ""                # 卡洛的游戏昵称，精确匹配；留空 = 主人命令模式关闭
owner_window = 30.0            # 收到卡洛一条 # 开头的命令后，放宽 move/emote/camera 限制多少秒
```

- [x] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: 全部 PASS（含 `test_example_config_loads`，确认新增的 toml 字段能被 `BrainConfig` 识别）

- [x] **Step 6: 提交**

```bash
git add src/skydango/config.py config.example.toml tests/test_config.py
git commit -m "feat: add brain config fields for move and owner command mode"
```

---

### Task 4: `Body` 识别卡洛的 `#` 命令，开授权窗口

**Files:**
- Modify: `src/skydango/brain/body.py:38-98`（构造函数和状态字段）
- Modify: `src/skydango/brain/body.py:182-196`（`_heard`）
- Modify: `src/skydango/brain/events.py:13`（注释）
- Test: `tests/test_brain_body.py`

**Interfaces:**
- Consumes：`Config.brain.owner_name` / `Config.brain.owner_window`（Task 3）。
- Produces：`Body._owner_window_until: float`（其他任务的 `move`/`emote`/`camera_move` 都读它判断
  `now < self._owner_window_until`）；命中时的事件 `Event(kind="owner_command", ...)`。

- [x] **Step 1: 写失败的测试**

在 `tests/test_brain_body.py` 里，`test_chat_messages_become_events` 附近加：

```python
def test_owner_command_opens_authorization_window(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    b.cfg.brain.owner_window = 30.0
    reader.batches = [[msg("#过来")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["owner_command"]
    assert b._owner_window_until == clock() + 30.0


def test_owner_command_requires_exact_speaker_match(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("#过来", speaker="懒洋洋大王2")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
    assert b._owner_window_until == float("-inf")


def test_non_hash_message_from_owner_is_normal_chat(clock):
    b, _, reader, events = body(clock)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("过来呀")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
    assert b._owner_window_until == float("-inf")


def test_owner_command_disabled_when_owner_name_empty(clock):
    b, _, reader, events = body(clock)
    assert b.cfg.brain.owner_name == ""
    reader.batches = [[msg("#过来")]]
    b.step()
    assert [e.kind for e in events.drain()] == ["chat"]
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_body.py -k owner_command -v`
Expected: FAIL，`AttributeError: 'Body' object has no attribute '_owner_window_until'`

- [x] **Step 3: 改实现**

`src/skydango/brain/body.py` 构造函数里，`self._holding_since = 0.0` 那行之后加一行：

```python
        self._holding_since = 0.0
        self._accepted_hand: tuple[str, float] | None = None
        self._owner_window_until = float("-inf")  # now < 这个值 = 卡洛的 # 命令还在生效
```

`_heard` 方法里，把最后两行：

```python
        for m in fresh:
            self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：{m.text}")
```

改成：

```python
        owner = self.cfg.brain.owner_name
        for m in fresh:
            if owner and m.speaker == owner and m.text.startswith("#"):
                self._owner_window_until = now + self.cfg.brain.owner_window
                self.events.put("owner_command", f"卡洛的命令：{m.text}")
                log.info("识别到卡洛的命令：%s（授权窗口延长到 %.0f 秒后）", m.text, self.cfg.brain.owner_window)
            else:
                self.events.put("chat", f"聊天  {m.speaker or '（看不出是谁）'}：{m.text}")
```

`src/skydango/brain/events.py` 里 `Event` 的 `kind` 字段注释：

```python
    kind: str  # chat / arrive / leave / request / accepted / holding / released / scene_change / panel / error / fallback / dropped
```

改成：

```python
    kind: str  # chat / owner_command / arrive / leave / request / accepted / holding / released / scene_change / panel / error / fallback / dropped
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_body.py -v`
Expected: 全部 PASS（含之前已有的 body 测试，确认没改坏别的行为）

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/events.py tests/test_brain_body.py
git commit -m "feat: recognize owner's # commands and open an authorization window"
```

---

### Task 5: `Body.move()` 工具方法

**Files:**
- Modify: `src/skydango/brain/body.py`（构造函数加 `locomotion` 参数、导入、新增 `move()`）
- Test: `tests/test_brain_body.py`

**Interfaces:**
- Consumes：`Locomotion`（Task 1，鸭子类型，测试里用 `FakeLocomotion`）、`self._owner_window_until`
  （Task 4）、`self.cfg.brain.move_min_interval`（Task 3）。
- Produces：`Body.move(direction: str, steps: int = 1, force: bool = False) -> str`，
  `Body._last_move: float`——`brain/tools.py`（Task 8）的 `ToolBox._bind` 会调用它。

- [x] **Step 1: 写失败的测试**

在 `tests/test_brain_body.py` 里，`FakeCamera` 类附近加一个 `FakeLocomotion`，并加测试：

```python
class FakeLocomotion:
    def __init__(self):
        self.moves = []

    def move(self, direction, steps, max_steps):
        self.moves.append((direction, steps, max_steps))
        return f"{direction}走了 {min(steps, max_steps)} 步"


def test_move_refused_while_holding_unless_forced(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, live=True, locomotion=loco)
    b.holding = "懒洋洋大王"
    with pytest.raises(ToolError, match="force=true"):
        b.move("forward")
    assert b.move("forward", force=True) == "forward走了 1 步"
    assert loco.moves == [("forward", 1, 3)]  # 默认 MAX_STEPS = 3


def test_move_rate_limited(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, live=True, locomotion=loco)
    assert b.move("forward") == "forward走了 1 步"
    with pytest.raises(ToolError, match="刚走过"):
        b.move("forward")  # cfg.brain.move_min_interval 默认 3 秒
    clock.advance(3)
    b.move("forward")  # 不再报错


def test_move_in_dry_run_does_not_touch_device(clock):
    loco = FakeLocomotion()
    b, _, _, _ = body(clock, locomotion=loco)
    assert b.move("forward").startswith("dry-run")
    assert loco.moves == []


def test_owner_window_bypasses_move_limits_and_tags_result(clock):
    loco = FakeLocomotion()
    b, _, reader, _ = body(clock, live=True, locomotion=loco)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("#过来")]]
    b.step()  # 打开授权窗口
    b.holding = "懒洋洋大王"  # 牵着手也不用传 force
    assert b.move("forward", steps=10).endswith("（主人命令模式）")
    assert loco.moves == [("forward", 10, 6)]  # max_steps 翻倍到 6
    b.move("forward")  # 频率限制也不生效，不报错
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_body.py -k move -v`
Expected: FAIL，`TypeError: Body.__init__() got an unexpected keyword argument 'locomotion'`

- [x] **Step 3: 改实现**

`src/skydango/brain/body.py` 顶部导入区（`from .events import EventQueue` 那行附近）加：

```python
from .locomotion import MAX_STEPS as MOVE_MAX_STEPS
```

构造函数签名（`camera=None,  # brain.camera.Camera` 那行之后）加一个参数：

```python
        camera=None,  # brain.camera.Camera
        locomotion=None,  # brain.locomotion.Locomotion
```

构造函数体里，`self.camera = camera` 之后加：

```python
        self.camera = camera
        self.locomotion = locomotion
```

状态字段区（`self._accepted_hand` 那行之后）加：

```python
        self._last_move = float("-inf")
```

新增方法，放在 `camera_move`/`camera_reset` 附近（`camera_reset` 方法之后）：

```python
    def move(self, direction: str, steps: int = 1, force: bool = False) -> str:
        if self.locomotion is None:
            raise ToolError("没有移动能力")
        now = self.clock()
        in_window = now < self._owner_window_until
        if self.holding and not force and not in_window:
            raise ToolError(f"正牵着 {self.holding} 的手，移动会松手；确定要松手再走就传 force=true")
        if not in_window and now - self._last_move < self.cfg.brain.move_min_interval:
            wait = self.cfg.brain.move_min_interval - (now - self._last_move)
            raise ToolError(f"刚走过，等 {wait:.0f} 秒再走")
        self._last_move = now
        tag = "（主人命令模式）" if in_window else ""
        if self.cfg.reply.dry_run:
            return f"dry-run：没真的走（{direction} ×{steps}）{tag}"
        max_steps = MOVE_MAX_STEPS * 2 if in_window else MOVE_MAX_STEPS
        try:
            result = self.locomotion.move(direction, steps, max_steps)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return result + tag
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_body.py -v`
Expected: 全部 PASS

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_body.py
git commit -m "feat: add Body.move() with hand-hold and rate-limit guards"
```

---

### Task 6: `Body.emote()` / `Body.camera_move()` 接入授权窗口

**Files:**
- Modify: `src/skydango/brain/body.py`（`emote` 和 `camera_move` 方法）
- Test: `tests/test_brain_body.py`

**Interfaces:**
- Consumes：`self._owner_window_until`（Task 4）、`EmotePlayer.on_wheel()`/`.extra`（`game/emotes.py`
  已有）、`from .camera import MAX_STEPS as CAMERA_MAX_STEPS`。
- Produces：窗口内 `emote()` 跳过 `min_interval`/`swap_min_interval`（`available()` 的隐含冷却）和牵手拦截；
  `camera_move()` 的 `steps` 上限翻倍；两者的返回文案在窗口内追加"（主人命令模式）"。

- [x] **Step 1: 写失败的测试**

在 `tests/test_brain_body.py` 里加：

```python
def test_owner_window_bypasses_emote_hold_and_cooldown(clock):
    emotes = FakeEmotes()
    b, _, reader, _ = body(clock, live=True, emotes=emotes)
    b.cfg.brain.owner_name = "懒洋洋大王"
    b.holding = "懒洋洋大王"
    reader.batches = [[msg("#鞠躬")]]
    b.step()  # 打开授权窗口
    assert b.emote("鞠躬").endswith("（主人命令模式）")
    assert emotes.done == ["鞠躬"]  # 没传 force 也做成了


def test_owner_window_raises_camera_step_cap(clock):
    cam = FakeCamera()
    b, _, reader, _ = body(clock, live=True, camera=cam)
    b.cfg.brain.owner_name = "懒洋洋大王"
    reader.batches = [[msg("#转过去")]]
    b.step()
    assert b.camera_move("left", 10).endswith("（主人命令模式）")
    assert cam.moves == [("left", 10, 8)]  # CAMERA_MAX_STEPS(4) * 2
```

`FakeCamera.move` 目前签名是 `def move(self, action, steps)`；改成 `def move(self, action, steps, max_steps):
self.moves.append((action, steps, max_steps)); return "左转了 1 步"`（同时改掉 `test_camera_refused_in_blackout_and_dry_run`
里对 `FakeCamera` 的假设——那个测试只检查返回值前缀，不用改断言，只是 `move` 签名要匹配新调用方式）。

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_body.py -k owner_window_bypasses -v`
Expected: FAIL（`emote()`/`camera_move()` 还不知道 `_owner_window_until`，`FakeCamera.move` 签名不匹配）

- [x] **Step 3: 改实现**

先改 `tests/test_brain_body.py` 里的 `FakeCamera`：

```python
class FakeCamera:
    def __init__(self):
        self.moves = []

    def move(self, action, steps, max_steps):
        self.moves.append((action, steps, max_steps))
        return "左转了 1 步"

    def reset(self):
        return "镜头转回原位了"

    def describe(self):
        return "原位"
```

`src/skydango/brain/body.py` 顶部导入区加：

```python
from .camera import MAX_STEPS as CAMERA_MAX_STEPS
```

`emote()` 方法整段改成：

```python
    def emote(self, name: str, force: bool = False) -> str:
        if self.emotes is None:
            raise ToolError("这次没开动作（--no-emotes 或者图标库是空的）")
        now = self.clock()
        in_window = now < self._owner_window_until
        available = self.emotes.available()
        allowed = name in available
        if not allowed and in_window:
            allowed = name in self.emotes.on_wheel() or name in self.emotes.extra
        if not allowed:
            raise ToolError(f"「{name}」现在做不了；能做的：{'、'.join(available) or '暂时没有（刚做过动作，要等一会儿）'}")
        if self.holding and not force and not in_window:
            raise ToolError(f"正牵着 {self.holding} 的手，做动作会松手；确定要松手再做就传 force=true")
        self.emoted.append(name)
        self.emoted[:] = self.emoted[-50:]  # 只留最近 50 条，别无限长
        tag = "（主人命令模式）" if in_window else ""
        if self.cfg.reply.dry_run:
            self.emotes.pretend(name)
            return f"dry-run：没真的做「{name}」{tag}"
        try:
            self.emotes.perform(name)
        except Exception as exc:
            raise ToolError(f"「{name}」没做成：{exc}") from None
        return f"做了「{name}」{tag}"
```

`camera_move()` 方法整段改成：

```python
    def camera_move(self, action: str, steps: int = 1) -> str:
        if self.camera is None:
            raise ToolError("没有视角控制")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在转不了")
        in_window = self.clock() < self._owner_window_until
        tag = "（主人命令模式）" if in_window else ""
        if self.cfg.reply.dry_run:
            return f"dry-run：没真的转（{action} ×{steps}）{tag}"
        max_steps = CAMERA_MAX_STEPS * 2 if in_window else CAMERA_MAX_STEPS
        try:
            result = self.camera.move(action, steps, max_steps)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        self._ref_thumb = None  # 自己转的镜头，不算画面大变
        return "镜头现在：" + result + tag
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_body.py -v`
Expected: 全部 PASS（含 `test_emote_refused_while_holding_unless_forced`、
`test_camera_refused_in_blackout_and_dry_run` 这些原来就有的测试，确认没改坏平时的行为）

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_body.py
git commit -m "feat: let owner's authorization window relax emote/camera limits"
```

---

### Task 7: `move` 工具定义 + 分发

**Files:**
- Modify: `src/skydango/brain/tools.py`
- Test: `tests/test_brain_tools.py`

**Interfaces:**
- Consumes：`Body.move(direction, steps, force)`（Task 5）、`skydango.brain.locomotion.KEYS`（Task 1）。
- Produces：`TOOLS` 列表和 `ACTIONS` 集合里多一个 `"move"`，供 `brain/loop.py` 里把它交给 Claude 的
  `tools=` 参数（不用改 `loop.py`，它是直接引用 `tools.TOOLS`）。

- [x] **Step 1: 写失败的测试**

把 `tests/test_brain_tools.py` 改成：

```python
from skydango.brain.body import ToolError
from skydango.brain.tools import ACTIONS, TOOLS, ToolBox


class FakeBody:
    def __init__(self):
        self.calls = []

    def call(self, fn):
        return fn()

    def look(self):
        return [{"type": "image"}]

    def look_at(self, x, y, w, h):
        self.calls.append(("look_at", x, y, w, h))
        return "ok"

    def status(self):
        return "状态"

    def chat_log(self, n):
        self.calls.append(("chat_log", n))
        return "log"

    def say(self, text):
        return f"说了{text}"

    def emote(self, name, force):
        self.calls.append(("emote", name, force))
        return "ok"

    def set_policy(self, who, kind, accept):
        self.calls.append(("policy", who, kind, accept))
        return "ok"

    def camera_move(self, action, steps):
        self.calls.append(("camera", action, steps))
        return "ok"

    def camera_reset(self):
        raise ToolError("没有视角控制")

    def move(self, direction, steps, force):
        self.calls.append(("move", direction, steps, force))
        return "ok"


def test_tool_definitions_cover_dispatch():
    names = [t["name"] for t in TOOLS]
    assert names == [
        "look", "look_at", "status", "chat_log", "say", "emote",
        "set_request_policy", "camera", "camera_reset", "move",
    ]
    assert ACTIONS == {"say", "emote", "set_request_policy", "camera", "camera_reset", "move"}
    assert all(t["input_schema"]["type"] == "object" and t["description"] for t in TOOLS)


def test_run_dispatches_with_defaults():
    body = FakeBody()
    tb = ToolBox(body)
    assert tb.run("say", {"text": "在呢"}) == ("说了在呢", False)
    assert tb.run("look", {}) == ([{"type": "image"}], False)
    tb.run("chat_log", {})
    tb.run("emote", {"name": "鞠躬"})
    tb.run("camera", {"action": "left"})
    tb.run("set_request_policy", {"who": "*", "kind": "hug", "accept": False})
    tb.run("move", {"direction": "forward"})
    assert body.calls == [
        ("chat_log", 20), ("emote", "鞠躬", False), ("camera", "left", 1),
        ("policy", "*", "hug", False), ("move", "forward", 1, False),
    ]


def test_bad_arguments_and_errors_become_error_results():
    tb = ToolBox(FakeBody())
    content, err = tb.run("say", {})
    assert err and "缺少参数 text" in content
    content, err = tb.run("look_at", {"x": "1", "y": 0, "w": 10, "h": 10})
    assert err and "x" in content and "整数" in content
    content, err = tb.run("look_at", {"x": True, "y": 0, "w": 10, "h": 10})  # bool 不算整数
    assert err
    assert tb.run("camera_reset", {}) == ("没有视角控制", True)
    content, err = tb.run("fly", {})
    assert err and "没有这个工具" in content


def test_crash_in_body_does_not_escape():
    body = FakeBody()
    body.status = lambda: 1 / 0
    content, err = ToolBox(body).run("status", {})
    assert err and content.startswith("出错了")
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_tools.py -v`
Expected: FAIL，`test_tool_definitions_cover_dispatch` 断言的列表/集合和实际的 `TOOLS`/`ACTIONS` 不一致

- [x] **Step 3: 改实现**

`src/skydango/brain/tools.py` 里，`from .camera import KEYS` 那行之后加：

```python
from .locomotion import KEYS as MOVE_KEYS
```

`TOOLS` 列表里 `camera_reset` 那一条之后加：

```python
    {
        "name": "move",
        "description": (
            "小步移动角色（前进/后退/左右），配合 status/look 自己看方向对不对；"
            "不能精确导航到某个位置，也没法一键退回原位。steps 1~3，默认 1。"
            "牵着手时移动会松手，确定要松手再走就传 force=true。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": list(MOVE_KEYS)},
                "steps": {"type": "integer"},
                "force": {"type": "boolean"},
            },
            "required": ["direction"],
        },
    },
```

`ACTIONS` 那行改成：

```python
ACTIONS = {"say", "emote", "set_request_policy", "camera", "camera_reset", "move"}  # 算“做了事”的工具（心跳退档用）
```

`ToolBox._bind` 里 `camera_reset` 分支之后加：

```python
        if name == "move":
            direction, steps, force = _str(a, "direction"), _int(a, "steps", 1), _bool(a, "force", False)
            return lambda: b.move(direction, steps, force)
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_tools.py -v`
Expected: 全部 PASS

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/tools.py tests/test_brain_tools.py
git commit -m "feat: expose move as a brain tool"
```

---

### Task 8: 提示词更新

**Files:**
- Modify: `src/skydango/brain/prompt.py:29-30`（移动能力描述）
- Modify: `src/skydango/brain/prompt.py`（新增 `## 主人命令` 小节，插在"互动请求"和"视角"之间）
- Test: `tests/test_brain_prompt.py`

**Interfaces:**
- Consumes：无新接口，纯文本。
- Produces：`static_prompt()` 返回的文本里包含 `move`、`主人命令`、`# 开头` 等关键词，供人工/后续测试核对。

- [x] **Step 1: 写失败的测试**

把 `tests/test_brain_prompt.py` 里的 `test_static_prompt_keeps_identity_rules_and_explains_tools` 改成：

```python
def test_static_prompt_keeps_identity_rules_and_explains_tools():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "不要说“我是真人”" in text and "老实承认是 AI" in text  # 身份底线不能丢
    assert "不会发进游戏" in text  # 普通文字是想法，只有 say 才说话
    for tool in ("say", "look", "look_at", "emote", "set_request_policy", "camera", "move"):
        assert tool in text
    assert "40 个字" in text


def test_static_prompt_explains_move_and_owner_command():
    text = static_prompt(ReplyConfig(max_chars=40))
    assert "小步移动" in text and "做不到" in text and "精确导航" in text
    assert "## 主人命令" in text
    assert "# 开头" in text and "只有卡洛的才算" in text
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_prompt.py -v`
Expected: FAIL（`move`、`## 主人命令` 还不在 `BRAIN_RULES` 里）

- [x] **Step 3: 改实现**

`src/skydango/brain/prompt.py` 里，把：

```python
- 你现在不能走、飞、跑图、跟着别人走、弹琴、送东西（身体没有这些能力）。别答应这些（不说“我跟着你”“走呗”），
  被叫去就自然地推掉，比如“我先挂会儿”“今天懒得动，你们去吧”“我在这儿等你们”。
```

改成：

```python
- 你能小步移动（move）：前进 / 后退 / 左右，一次走一小步，走完自己用 status/look 看看方向对不对、
  有没有走偏、周围安不安全，再决定要不要接着走。做不到精确导航到某个位置、长距离跑图、跟着别人走、
  飞、弹琴、送东西；走丢了、走过头了，回不去就老实说（比如“糟糕我走过了”“不知道走哪儿去了”），别硬编方向。
  牵着手的时候移动会松手，跟做动作一样。懒人设定还在——现在是“能选择动或不动”，被喊过去时仍然可以自然地
  推掉（“我先挂会儿”“懒得动”），只是这个推拒现在是个真实的选择，不是因为身体做不到。
```

把：

```python
## 视角（camera）
```

改成（在它前面插入一整节）：

```python
## 主人命令
卡洛的消息如果以 # 开头，是他在直接指挥你（不是普通聊天），这时候你调用 move / emote / camera 会暂时
不受平时的步数 / 频率限制。但这只是让你“能更痛快地做”，不代表必须照做——你还是自己判断要不要听、做什么；
懒人设定、底线规则完全不受影响。别人发 # 开头的话不算数，只有卡洛的才算。

## 视角（camera）
```

- [x] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_prompt.py -v`
Expected: 全部 PASS

- [x] **Step 5: 提交**

```bash
git add src/skydango/brain/prompt.py tests/test_brain_prompt.py
git commit -m "docs(prompt): describe move capability and owner command mode"
```

---

### Task 9: 文档收尾 + 全量测试

**Files:**
- Modify: `AGENTS.md:36`

- [x] **Step 1: 改 `AGENTS.md`**

把：

```
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`tools.py` 工具、`context.py` 上下文和缓存、`camera.py` 视角 |
```

改成：

```
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`tools.py` 工具、`context.py` 上下文和缓存、`camera.py` 视角、`locomotion.py` 移动 |
```

- [x] **Step 2: 跑全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS，没有因为本次改动引入的失败

- [x] **Step 3: 提交**

```bash
git add AGENTS.md
git commit -m "docs: mention locomotion.py in the code layout table"
```

---

### Task 10: 真机验证（手动，不是自动化测试）

这一步不是 pytest，是照 `docs/superpowers/specs/2026-09-27-brain-move-design.md` 的"真机验证"清单实测，
需要 MuMu 开着、游戏在前台。

- [ ] **Step 1: 校准 `move_step` 默认值**

在安全开阔地带（没有水域/悬崖），用类似 Task 1 之前那次 adb 实测的方式，手动跑几次
`Locomotion(device, cfg.brain.move_step).move("forward", 1)`，确认 0.3 s 的实际幅度是不是符合预期
（比之前测的 1 s 明显小、比 0.3 s 单次裸测更可控）。如果幅度不对，回到 Task 3 调整
`BrainConfig.move_step` 默认值和 `config.example.toml` 的注释，重新走一遍该任务的测试。

- [ ] **Step 2: 验证大脑真的会用 move**

```bash
python -m skydango run --brain --live --duration 300
```

被朋友喊"过来"时观察：大脑会不会调用 `move`、方向对不对、有没有连续好几步不看 `status` 就失控地走。

- [ ] **Step 3: 验证主人命令模式**

`config.toml` 里填好 `[brain] owner_name = "<卡洛的实际游戏昵称>"`，重复 Step 2 的跑法，在游戏里用卡洛
的账号发一条 `#过来` 之类的消息，确认：
- `agent.log` 里出现"识别到卡洛的命令"；
- 随后的 `move`/`emote`/`camera` 调用结果里带"（主人命令模式）"；
- 换一个好友账号发同样的 `#` 开头消息，确认不会触发（`agent.log` 里没有"识别到卡洛的命令"这行）。

- [ ] **Step 4: 核对审计日志**

打开这次 `runs/<时间戳>-live-brain/brain.jsonl` 和 `agent.log`，确认 `move` 的调用参数、触发命令模式的
时间点都能对上，跟本文档"背景"一节里核对 `say`/`status` 调用的方法一样。

- [ ] **Step 5: 记录结果**

把这次真机验证里任何和 spec 假设不符的地方（比如 `move_step` 实际最优值、A/D 键码是否正确）更新进
`docs/game-ops.md` §2 和这份 spec 文档，跟本次开发之前那次 WASD 实测一样"先验证、再定案"。

---

## 预估用时

- Task 1–9（自动化实现 + 单测）：约 60~90 分钟 agent 工作时间，每个任务都有独立的 TDD 循环和提交。
- Task 10（真机验证）：至少两次 `run --brain --live --duration 300`（各 5 分钟实时等待）+ 若干次手动
  adb 移动校准，预计 20~30 分钟真实时间。
- **合计约 1.5~2 小时**，其中大头是真机验证的等待时间，不是代码本身的复杂度。
