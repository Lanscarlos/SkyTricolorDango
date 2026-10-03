# 东张西望改成"有意识地找" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删掉空闲注意力里纯随机的"随意看"，换成有动机的找：找刚走开的好友、一个人时往最久没看过的方向环顾一片、大脑用 `find(名字)` 让身体找人；有意义的结果交给大脑。

**Architecture:** 新的纯计算模块 `brain/search.py`（粗略朝向 `Heading` + 三种找的计划 `Search`），被两处共用：注意力（`brain/attention.py`，闲着的找）和新技能 `brain/find.py`（大脑的目标）。身体（`brain/body.py`）只负责发起"找走开的好友"、把感知结果拼成 `Obs`、真按键 / 喊一声、发背景事件 `search`。

**Tech Stack:** Python 3.13、pytest（合成数据 + 假设备，不需要模拟器）。

**Spec:** `docs/superpowers/specs/2026-10-03-attention-search-design.md`

## Global Constraints

- 用中文写注释、日志、给大脑看的文字；代码风格照周围的代码（注释密度、命名、`log.debug` 的写法）
- 测试命令：`.venv\Scripts\python.exe -m pytest -q`（worktree 里用主目录的 `.venv`：`D:\Lanscarlos\Develop\SkyTricolorDango\.venv\Scripts\python.exe`；`tests/conftest.py` 会把 worktree 的 `src` 放在最前面）。**全量测试放后台跑**（用户要求），单个测试文件可以前台跑
- 镜头照旧**不回位**；没有动机就不转（不保留"发呆换个朝向"）
- 背景事件 `search` 不单独叫醒大脑；大脑的 `find` 结果走技能层的 `task_done` / `task_failed`（立刻叫醒）
- 数字都是估的（`press_deg` 18°、`seg_presses` 3、`dwell` 1.5 s、`scan_after` 20 s、`scan_every` 60 s……），不在这次调
- 每个任务提交一次；全部做完按 CLAUDE.md 合并进 main 并推送

## 偏离 spec 的地方（写计划时查代码定的）

1. **走开的好友最后在哪**：spec 写"失踪记录 `Lost` 里最后的框和速度"，但 `Lost` 只留 `keep`（5 秒），而"走开"本身就是 `keep` 秒没看到才判的，发起找的时候记录多半已经删了。改用感知层 `env.labels[名字]`（名字标签最后的位置，一直留着）：中心 x 在左半边就往左找；在最左 / 最右 15%（`EXIT_EDGE`）里算"从画面边上出去的"。
2. **中间淡掉、Q 也没认出来时**：spec 写"当作不在画面里，往他那边转"。但这时画面里多半还站着没挂名字的人（他本人走远了、标签淡了），往一边转会把他转出画面。改成：画面里**还有没挂名字的人就不转**，结果写"画面里还有 N 个没挂名字的人，可能就是他"；画面里没人才往他那边转。
3. **环顾看到好友不发事件**：spec 表里写"看到了好友（他本来就不在身边时才发）"。可是名字认出来了，他马上就进了身边名单，`_watch_comings` 会发 `arrive` / `return`，所以这一条永远轮不到，直接不发。
4. **`find` 总是注册**（同 `track`）：spec 写"感知层开着才注册"。现有的 `track` 是总注册、运行时没感知层就拒绝，`find` 照它做，工具列表测试也只改一处。沙盒里回"沙盒里没有这个：镜头"。
5. **自动喊让给注意力的条件**多一条"注意力模式会找走开的好友"（随意 / 好奇 / 专心）：模式是"别动"时注意力不找，`_watch_call` 照旧自己喊，不然这种情况下谁都不喊了。

## File Structure

| 文件 | 改动 | 职责 |
|---|---|---|
| `src/skydango/brain/search.py` | 新建 | `Heading`（粗略朝向、6 个方位上次看过的时间）、`Obs`（一圈的感知结果）、`SearchStep`、`Search`（三种计划的分段转 / 喊一声 / 判定）、`event_text` |
| `src/skydango/brain/attention.py` | 改 | 删随意看；没有被动目标时推进 `Search`；自己发起环顾；`start_lost`；`describe(now)` 加"在找 / 刚才" |
| `src/skydango/brain/find.py` | 新建 | 技能 `FindSkill`：包一个 `Search.find`，喊一声的结果自己收 |
| `src/skydango/brain/body.py` | 改 | `search_obs`、`_left_where`、`_start_lost_search`、`_search_call`、`_watch_attention` 接线、`_watch_call` 让位、`find`、`move` 后清朝向 |
| `src/skydango/brain/events.py` | 改 | `search` 进 `BACKGROUND`，攒着只留最新一条 |
| `src/skydango/brain/tools.py` `mcp_server.py` `prompt.py` | 改 | 注册 `find`、改 `attention` 说明、提示词「视角」「做事」两节 |
| `src/skydango/config.py` | 改 | `AttentionConfig` 增删字段；废弃键跳过 |
| `src/skydango/inner/effects.py` | 改 | `wander` 注释改成乘环顾间隔 |
| `src/skydango/console/settings.py` | 改 | 加 `attention.search` 开关 |
| `config.example.toml` | 改 | `[attention]` 段 |
| `tests/test_brain_search.py` | 新建 | `search.py` 单测 |
| `tests/test_brain_attention.py` | 改 | 删随意看的测试、加找的测试 |
| `tests/test_brain_search_body.py` | 新建 | 身体接线 |
| `tests/test_brain_find.py` | 新建 | `find` 技能和工具 |
| `tests/test_config.py` `tests/test_brain_tools.py` `tests/test_brain_prompt.py` `tests/test_brain_events.py`（没有就建） | 改 | 对应的断言 |
| `CLAUDE.md` | 改 | 「统管大脑」里空闲注意力那一条 |

---

### Task 1: 配置加新字段

**Files:**
- Modify: `src/skydango/config.py`（`class AttentionConfig`，约 685~713 行）
- Modify: `src/skydango/console/settings.py:59`（`call.auto` 那行后面）
- Modify: `config.example.toml:350-372`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `AttentionConfig.search: bool`、`press_deg: float`、`seg_presses: int`、`dwell: float`、`scan_after: float`、`scan_every: float`、`scan_segments: list[int]`、`scan_look: float`、`lost_segments: int`、`resume_within: float`（后面所有任务用）

- [ ] **Step 1: 写失败的测试**（加在 `tests/test_config.py` 末尾）

```python
def test_attention_search_defaults():  # spec 2026-10-03-attention-search §4
    a = Config().attention
    assert a.search is True and a.press_deg == 18.0 and a.seg_presses == 3 and a.dwell == 1.5
    assert a.scan_after == 20.0 and a.scan_every == 60.0 and a.scan_segments == [2, 3] and a.scan_look == 3.0
    assert a.lost_segments == 2 and a.resume_within == 10.0
```

（`Config` 已经在文件顶上导入了；没有就加 `from skydango.config import Config`。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_attention_search_defaults -q`
Expected: FAIL（`AttributeError: 'AttentionConfig' object has no attribute 'search'`）

- [ ] **Step 3: 加字段**

在 `AttentionConfig` 的 `max_step` 那行**前面**加（`wander_*` 四项这一步先留着，Task 4 删）：

```python
    # 有意识地找（spec 2026-10-03-attention-search）：找刚走开的好友、一个人时环顾一片；都没有就不转
    search: bool = True  # 关掉只剩被动注意（有人说话 / 走近 / 对团子做动作就转过去看）
    press_deg: float = 18.0  # 每下 [track] nudge_max 约转几度（按 [spin] seconds_per_turn 2 秒一圈估的，待 camera spin 标定）
    seg_presses: int = 3  # 分段转：一段按几下
    dwell: float = 1.5  # 一段转完停几秒（等 YOLO 出框、名字标签读出来）
    scan_after: float = 20.0  # 画面里没人多久才环顾
    scan_every: float = 60.0  # 两次环顾至少隔几秒（再乘心情精力、模式的倍数）
    scan_segments: list[int] = field(default_factory=lambda: [2, 3])  # 环顾一次转几段（随机取一个）
    scan_look: float = 3.0  # 环顾看到陌生人停下看几秒
    lost_segments: int = 2  # 找走开的好友往一边最多转几段（Q 之后看到贴边标签再给同样多）
    resume_within: float = 10.0  # 找到一半被挡住 / 打断，这么久内接着找，超过就不找了
```

类的文档字符串改成：

```python
    """空闲注意力（东张西望，spec 2026-09-30-idle-attention、2026-10-03-attention-search）：闲着时按兴趣小步转镜头看说话 / 走近 /
    对团子做事的人；没有这些时找刚走开的好友、一个人待着时往最久没看过的方向看一片，没有动机就不转。
    只在聊天面板 auto 模式、面板关着时动。数字都是估的，真机调。按键长短、settle、转不动沿用 [track]。"""
```

- [ ] **Step 4: 管理面板开关**

`src/skydango/console/settings.py` 里 `Field("call.auto", …)` 那行后面加一行：

```python
    Field("attention.search", "有意识地找", "闲着时找刚走开的好友、一个人待着时隔一会儿往没看过的方向看看；关掉只剩“有人说话 / 走近就转过去看”", "bool", "brain"),
```

- [ ] **Step 5: 示例配置**

`config.example.toml` 的 `[attention]` 段：开头两行注释换成

```toml
# 空闲注意力（东张西望，spec docs/superpowers/specs/2026-09-30-idle-attention-design.md、2026-10-03-attention-search-design.md）：
# 闲着时转镜头看说话 / 走近 / 对团子做动作的人；没有这些就找刚走开的好友、一个人时往没看过的方向看一片，没有动机就不转。
# 只在 [panel] mode = "auto"、聊天面板关着时动。数字都是估的
```

`wander_*` 四行这一步先不动（Task 4 删），在它们**后面**加：

```toml
search = true                   # 有意识地找（找刚走开的好友 + 环顾）；关掉只剩上面的"看说话 / 走近的人"
press_deg = 18.0                # 每下 [track] nudge_max 约转几度（待 camera spin 标定）
seg_presses = 3                 # 分段转：一段按几下
dwell = 1.5                     # 一段转完停几秒
scan_after = 20.0               # 画面里没人多久才环顾
scan_every = 60.0               # 两次环顾至少隔几秒（再乘心情精力 / 模式的倍数）
scan_segments = [2, 3]          # 环顾一次转几段
scan_look = 3.0                 # 环顾看到陌生人停下看几秒
lost_segments = 2               # 找走开的好友往一边最多转几段
resume_within = 10.0            # 找到一半被打断，这么久内接着找
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_console_settings.py -q`（`test_console_settings.py` 不存在就只跑前一个；用 `ls tests | grep console` 找设置清单的测试一起跑）
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/skydango/config.py src/skydango/console/settings.py config.example.toml tests/test_config.py
git commit -m "feat(attention): 有意识地找的配置项

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `Heading`：粗略朝向和"最久没看过"

**Files:**
- Create: `src/skydango/brain/search.py`
- Test: `tests/test_brain_search.py`

**Interfaces:**
- Consumes: Task 1 的 `AttentionConfig.press_deg`
- Produces:
  - `SECTORS = 6`、`SECTOR_DEG = 60.0`、`bearing_name(offset: float) -> str`（"前面" / "右前方" / "右后方" / "后面" / "左后方" / "左前方"）
  - `class Heading(press_deg: float, nudge_max: float, hfov: float, now: float)`：属性 `deg: float`（0~360，往右为正）、`seen: list[float]`；方法 `reset(now)`、`turned(direction: str, seconds: float, now)`、`mark(now)`、`stalest(prefer: str) -> tuple[str, str]`（返回 (往哪边转 "left"/"right", 相对现在的叫法)）

- [ ] **Step 1: 写失败的测试**（新建 `tests/test_brain_search.py`）

```python
"""有意识地找：纯计算（plan 2026-10-03-attention-search Task 2 / 3）。画面宽 1920，hfov 90°，press_deg 18°。"""

import pytest

from skydango.brain.search import SECTORS, Heading, bearing_name
from skydango.config import Config


def cfgs(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.attention, k, v)
    return cfg.attention, cfg.track


def heading(now=0.0):
    a, t = cfgs()
    return Heading(a.press_deg, t.nudge_max, 90.0, now)


# ---- Task 2：Heading ----

def test_bearing_names():
    assert [bearing_name(d) for d in (0, 60, 120, 180, 240, 300, 359)] == [
        "前面", "右前方", "右后方", "后面", "左后方", "左前方", "前面"]
    assert bearing_name(66) == "右前方" and bearing_name(-60) == "左前方"


def test_heading_starts_facing_front_only():
    h = heading()
    assert h.deg == 0.0 and h.seen[0] == 0.0 and all(t == float("-inf") for t in h.seen[1:])


def test_heading_turns_by_press_seconds():
    h = heading()
    nudge = Config().track.nudge_max
    h.turned("right", nudge * 3, 5.0)  # 3 下 × 18° = 54°
    assert h.deg == pytest.approx(54.0) and h.seen[1] == 5.0 and h.seen[0] == 0.0
    h.turned("left", nudge * 6, 10.0)  # 往左 108° → 306°
    assert h.deg == pytest.approx(306.0) and h.seen[5] == 10.0


def test_stalest_prefers_never_seen_then_nearest_then_prefer_side():
    h = heading()
    assert h.stalest("left") == ("left", "左前方")  # 左前、右前都没看过、一样近：按 prefer
    assert h.stalest("right") == ("right", "右前方")
    nudge = Config().track.nudge_max
    h.turned("right", nudge * 3, 5.0)
    h.turned("left", nudge * 6, 10.0)  # 现在朝 306°：看过 0（t=0）、60（t=5）、300（t=10）
    assert h.stalest("right") == ("left", "左前方")  # 没看过的 120 / 180 / 240 里 240 最近（往左 66°）


def test_heading_reset_forgets():
    h = heading()
    h.turned("right", Config().track.nudge_max * 3, 5.0)
    h.reset(9.0)
    assert h.deg == 0.0 and h.seen[0] == 9.0 and h.seen[1] == float("-inf") and SECTORS == 6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'skydango.brain.search'`）

- [ ] **Step 3: 写 `Heading`**（新建 `src/skydango/brain/search.py`）

```python
"""有意识地找（spec docs/superpowers/specs/2026-10-03-attention-search-design.md §1）：纯计算，不碰设备。

三种找共用这里：找刚走开的好友（lost）、环顾看周围有没有人（scan）、大脑下的目标（find）。
每圈 step(obs, now) 给出这一圈要做的：按一下（press）/ 喊一声（call）/ 等着（wait）/ 结束（found / maybe / none / seen / empty）。
调用方真按了键调 pressed()，喊出去了调 call_sent()，拿到结果（或者没喊成）调 called()。
基本动作是分段转：一段连按 seg_presses 下（每下 nudge_max 秒），段和段之间停 dwell 秒让 YOLO 出框、名字标签读出来。

Heading 是粗略朝向：按键时长折成角度累计起来（往右为正），记 6 个方位上次在画面里的时间，只拿来挑"最久没看过"的方向，不求准；
走路、黑屏、别人转过镜头之后调用方 reset。
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SECTORS = 6
SECTOR_DEG = 360.0 / SECTORS
NAMES = ("前面", "右前方", "右后方", "后面", "左后方", "左前方")  # 相对现在的朝向，每 60° 一个
EPS = 1e-6


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def bearing_name(offset: float) -> str:
    """相对现在朝向的角度（往右为正）→ "右后方" 这样的叫法。"""
    return NAMES[round((offset % 360.0) / SECTOR_DEG) % SECTORS]


class Heading:
    def __init__(self, press_deg: float, nudge_max: float, hfov: float, now: float) -> None:
        self.press_deg, self.nudge_max, self.hfov = press_deg, nudge_max, hfov
        self.deg = 0.0
        self.seen: list[float] = []
        self.reset(now)

    def reset(self, now: float) -> None:
        """不知道朝哪了（走路、黑屏、别人转过镜头）：现在算 0°，只有眼前这个方位刚看过。"""
        self.deg = 0.0
        self.seen = [float("-inf")] * SECTORS
        self.mark(now)

    def turned(self, direction: str, seconds: float, now: float) -> None:
        """按了一下方向键：按住 seconds 秒折成角度（press_deg 对应 nudge_max 秒，线性估）。"""
        deg = self.press_deg * seconds / self.nudge_max
        self.deg = (self.deg + (deg if direction == "right" else -deg)) % 360.0
        self.mark(now)

    def mark(self, now: float) -> None:
        """视野里的方位记成刚看过（方位中心离朝向不到半个视野角）。"""
        for i in range(SECTORS):
            if _gap(i * SECTOR_DEG, self.deg) <= self.hfov / 2 + EPS:
                self.seen[i] = now

    def stalest(self, prefer: str) -> tuple[str, str]:
        """最久没看过的方位：(往哪边转, 相对现在的叫法)。一样久挑近的，一样近挑 prefer 那边（正后方也按 prefer）。"""
        best = None
        for i in range(SECTORS):
            center = i * SECTOR_DEG
            if _gap(center, self.deg) <= self.hfov / 2 + EPS:
                continue
            off = (center - self.deg) % 360.0
            dist = min(off, 360.0 - off)
            direction = prefer if abs(off - 180.0) < EPS else ("right" if off < 180.0 else "left")
            key = (self.seen[i], dist, direction != prefer)
            if best is None or key < best[0]:
                best = (key, direction, off)
        if best is None:  # 视野角 ≥ 360°：哪都看得到
            return prefer, "前面"
        return best[1], bearing_name(best[2])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search.py -q`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/search.py tests/test_brain_search.py
git commit -m "feat(attention): 粗略朝向 Heading，挑最久没看过的方向

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Search`：三种找的计划

**Files:**
- Modify: `src/skydango/brain/search.py`
- Test: `tests/test_brain_search.py`

**Interfaces:**
- Consumes: Task 2 的 `Heading`；`skydango.vision.people.CallSeen` / `Seen` / `side_of`；`skydango.brain.peek.Turn(direction, seconds)`；`skydango.chat.tracker.similar(a, b, threshold)`
- Produces（Task 4 / 5 / 6 用）：
  - `@dataclass(frozen=True) class Obs(width: int = 1920, target_x: float | None = None, target_where: str = "", maybe_x: float | None = None, friends: tuple[str, ...] = (), strangers: int = 0)`
  - `@dataclass(frozen=True) class SearchStep(state: str, turn: Turn | None = None, note: str = "")`；`TERMINAL = frozenset({"found", "maybe", "none", "seen", "empty"})`
  - `class Search`：类方法 `lost(who, side, edge_exit, can_call, cfg, track, heading, now)`、`find(who, side, edge_exit, can_call, cfg, track, heading, now)`、`scan(cfg, track, heading, rng, prefer, now)`；方法 `step(obs: Obs | None, now) -> SearchStep`、`pressed(turn, now)`、`call_sent(now)`、`called(seen: CallSeen | None, now)`、`describe() -> str`、`result_line() -> str`、`dirs() -> str`；属性 `kind`（"lost"/"scan"/"find"）、`who`、`side`、`where`、`state`（"running" 或 TERMINAL 之一）、`note`、`aborted`、`friends`、`strangers`
  - `event_text(s: Search, prev_empty: bool) -> str | None`

- [ ] **Step 1: 写失败的测试**（加在 `tests/test_brain_search.py`：顶上的 import 换成下面这组，测试接在 Task 2 的后面）

```python
import random

import pytest

from skydango.brain.peek import Turn
from skydango.brain.search import SECTORS, TERMINAL, Heading, Obs, Search, bearing_name, event_text
from skydango.config import Config
from skydango.vision.people import CallSeen, Seen
```

```python
# ---- Task 3：Search ----

def no_call(s, t):
    s.called(None, t)  # 调用方没喊成（额度不够 / 关着）


def run(s, obs_fn, t0=0.0, until=60.0, dt=0.1, on_call=no_call):
    """模拟调用方：每 dt 秒 step 一次，press 就当真按了，call 交给 on_call(s, t)。返回 [(t, step)]，结束就停。"""
    out, t = [], t0
    while t <= until:
        st = s.step(obs_fn(t), t)
        out.append((t, st))
        if st.state == "press":
            s.pressed(st.turn, t)
        elif st.state == "call":
            on_call(s, t)
        if st.state in TERMINAL:
            break
        t = round(t + dt, 6)
    return out


def presses(log):
    return [st.turn.direction for _, st in log if st.state == "press"]


def test_lost_from_left_edge_turns_left_then_calls_then_gives_up():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, True, a, tr, heading(), 0.0)
    calls = []
    log = run(s, lambda t: Obs(), on_call=lambda s, t: (calls.append(t), s.called(None, t)))
    assert presses(log) == ["left"] * (2 * a.seg_presses)
    assert all(st.turn == Turn("left", tr.nudge_max) for _, st in log if st.state == "press")
    first_press = next(t for t, st in log if st.state == "press")
    assert len(calls) == 1 and calls[0] > first_press  # 先转、转完才喊
    assert log[-1][1].state == "none" and s.note == "往左边找了找，没看到小明"
    assert event_text(s, False) == "你往左边找了找刚走开的小明，没看到他"


def test_lost_call_reveals_offscreen_tag_then_turns_that_way():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, True, a, tr, heading(), 0.0)
    seen = CallSeen(0.0, {"小明": Seen("右边", None, on_screen=False)})

    def on_call(s, t):
        s.call_sent(t)
        s.called(seen, t)

    log = run(s, lambda t: Obs(), on_call=on_call)
    assert presses(log) == ["left"] * 6 + ["right"] * 6
    assert s.note == "往左边、右边找了找，没看到小明，喊了一声也没看到他的名字"


def test_lost_middle_calls_first_and_name_lights_up():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    assert s.step(Obs(), 0.0).state == "call"
    assert s.step(Obs(), 0.1).state == "call"  # 调用方这一圈没喊（被挡住）：还要喊
    s.call_sent(0.1)
    assert s.step(Obs(), 0.2).state == "wait"
    s.called(CallSeen(0.1, {"小明": Seen("右边", "远")}), 6.5)
    st = s.step(Obs(), 6.6)
    assert st.state == "found" and s.note == "小明（右边·远）"
    assert event_text(s, False) is None  # 找到了：走现有的 return
    assert s.result_line() == "找到了小明（右边·远）"


def test_lost_middle_with_unnamed_people_does_not_turn_away():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(strangers=1))
    assert presses(log) == []
    assert log[-1][1].state == "none" and "1 个没挂名字的人" in s.note
    assert event_text(s, False) == "你找了找刚走开的小明，没看到他"


def test_lost_middle_alone_turns_toward_his_side_after_call():
    a, tr = cfgs()
    s = Search.lost("小明", "right", False, True, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs())
    assert presses(log) == ["right"] * (a.lost_segments * a.seg_presses)


def test_lost_found_mid_turn_stops():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(target_x=500.0, target_where="左边·中") if t >= 1.0 else Obs())
    assert log[-1][1].state == "found" and s.note == "小明（左边·中）"
    assert len(presses(log)) == 2  # 0.0、0.6 按了两下，1.0 看到了


def test_lost_maybe_is_reported_as_maybe():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(maybe_x=300.0) if t >= 1.0 else Obs())
    assert log[-1][1].state == "maybe" and s.note == "小明（没看到名字，左边）"
    assert event_text(s, False) == "你往左边找了找刚走开的小明，那边有个人可能是他（没看到名字）"


def test_lost_without_call_never_asks_to_call():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    log = run(s, lambda t: Obs(), on_call=lambda s, t: pytest.fail("不该喊"))
    assert presses(log) == ["left"] * 6 and log[-1][1].state == "none"


def test_search_gives_up_when_not_advanced_for_resume_within():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.step(Obs(), 0.0).state == "press"  # 调用方一直没按（被挡住）
    assert s.step(None, 0.5).state == "wait"  # 画面暂停：等着
    assert s.step(Obs(), 9.9).state == "press"
    st = s.step(Obs(), 10.2)
    assert st.state == "none" and s.aborted and event_text(s, False) is None
    assert s.result_line() == "找小明，被打断太久，不找了"


def test_scan_turns_toward_stalest_and_reports_empty():
    a, tr = cfgs(scan_segments=[2, 2])
    h = heading()
    h.turned("right", tr.nudge_max * 3, 5.0)
    h.turned("left", tr.nudge_max * 6, 10.0)
    s = Search.scan(a, tr, h, random.Random(1), "right", 11.0)
    assert s.where == "左前方" and s.describe() == "在看看周围：往左前方（那边很久没看了）"
    log = run(s, lambda t: Obs(), t0=11.0)
    assert presses(log) == ["left"] * 6 and log[-1][1].state == "empty" and s.note == "附近没人"
    assert s.result_line() == "往左前方看了看：附近没人"
    assert event_text(s, True) is None


def test_scan_segments_come_from_rng_range():
    a, tr = cfgs(scan_segments=[2, 3])
    counts = set()
    for seed in range(20):
        s = Search.scan(a, tr, heading(), random.Random(seed), "right", 0.0)
        counts.add(len(presses(run(s, lambda t: Obs()))) // a.seg_presses)
    assert counts == {2, 3}


def test_scan_sees_strangers_looks_then_reports():
    a, tr = cfgs()
    s = Search.scan(a, tr, heading(), random.Random(1), "right", 0.0)
    log = run(s, lambda t: Obs(strangers=2) if t >= 1.0 else Obs())
    end_t, end = log[-1]
    assert end.state == "seen" and s.note == "有 2 个陌生人" and end_t == pytest.approx(1.0 + a.scan_look)
    assert not any(st.state == "press" for t, st in log if t >= 1.0)
    assert event_text(s, True) == f"你往{s.where}看了看：有 2 个陌生人"
    assert event_text(s, False) is None  # 上次也有人：不算新鲜


def test_scan_sees_friend_stops_without_event():
    a, tr = cfgs()
    s = Search.scan(a, tr, heading(), random.Random(1), "right", 0.0)
    log = run(s, lambda t: Obs(friends=("小明",)) if t >= 1.0 else Obs())
    assert log[-1][1].state == "seen" and s.friends == ("小明",) and s.note == "看到了小明"
    assert event_text(s, True) is None  # 好友认出来就进身边名单，arrive / return 会说


def test_find_without_clue_calls_then_turns_full_circle():
    a, tr = cfgs()
    s = Search.find("小明", None, False, True, a, tr, heading(), 0.0)

    def on_call(s, t):
        s.call_sent(t)
        s.called(CallSeen(t, {}), t)

    log = run(s, lambda t: Obs(), until=120.0, on_call=on_call)
    dirs = presses(log)
    assert len(dirs) == SECTORS * a.seg_presses and len(set(dirs)) == 1
    assert s.kind == "find" and s.note == "转了一圈没找到小明，喊了一声也没看到他的名字"
    assert event_text(s, True) is None  # find 的结果走 task_done / task_failed


def test_find_with_leave_clue_uses_lost_plan():
    a, tr = cfgs()
    s = Search.find("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.kind == "find" and presses(run(s, lambda t: Obs())) == ["left"] * 6


def test_describe_progress():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    assert s.describe() == "在找：小明（他刚从左边走了）"
    st = s.step(Obs(), 0.0)
    s.pressed(st.turn, 0.0)
    assert s.describe() == "在找：小明（他刚从左边走了，往左边转了 1 段）"
    f = Search.find("小明", None, False, True, a, tr, heading(), 0.0)
    f.step(Obs(), 0.0)
    f.call_sent(0.0)
    assert f.describe() == "在找：小明（大脑让找的，喊了一声在等）"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search.py -q`
Expected: FAIL（`ImportError: cannot import name 'TERMINAL'`）

- [ ] **Step 3: 写 `Search`**

`src/skydango/brain/search.py`：import 那段换成

```python
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import AttentionConfig, TrackConfig
from ..vision.people import CallSeen, side_of
from .peek import Turn
```

常量区（`EPS` 后面）加：

```python
SIDE = {"left": "左边", "right": "右边"}
TERMINAL = frozenset({"found", "maybe", "none", "seen", "empty"})
```

文件末尾加：

```python
@dataclass(frozen=True)
class Obs:
    """这一圈的感知结果（调用方拼好）。target / maybe 只对要找的那个人；friends / strangers 给环顾用。"""

    width: int = 1920
    target_x: float | None = None  # 要找的人：名字证实的（人物框或画面里亮着的名字标签）中心 x
    target_where: str = ""  # "右边·远"（调用方拼好；空就按 x 算左右）
    maybe_x: float | None = None  # 要找的人：只是"像他"（按位置接回 / 认装扮），没看到名字
    friends: tuple[str, ...] = ()  # 画面里认得出名字的好友
    strangers: int = 0  # 画面里别的人（陌生人、黑影、没看到名字的）


@dataclass(frozen=True)
class SearchStep:
    state: str  # press / call / wait，或 TERMINAL 之一
    turn: Turn | None = None  # press 才有
    note: str = ""  # 进度或结果


class Search:
    """一次找。计划是一串步骤：("call",) 喊一声；("turn", 方向, 段数) 分段转；
    ("edge", 段数, 后备) 喊一声认出他在画面外就往那边转，否则换成后备（None = 跳过、"circle" = 转一圈、或者另一个步骤）；
    ("alone", 方向, 段数) 画面里没有没挂名字的人才往那边转（有的话他多半就在其中，转走反而丢了）。"""

    def __init__(self, kind: str, steps: list[tuple], cfg: AttentionConfig, track: TrackConfig, heading: Heading, now: float,
                 who: str | None = None, side: str | None = None, where: str = "") -> None:
        self.kind, self.who, self.side, self.where = kind, who, side, where
        self.cfg, self.track, self.heading = cfg, track, heading
        self._steps = list(steps)
        self._i = 0
        self._segs_left: int | None = None
        self._presses_left = 0
        self._dwell_until: float | None = None
        self._last_press = float("-inf")
        self._progress = now  # 上次有进展（开始、按键、喊、一段转完）：resume_within 没进展就不找了
        self._calling = False
        self._called_once = False
        self._edge: str | None = None  # 喊一声时他的名字贴在屏幕哪边
        self._call_found: str | None = None  # 喊一声时他的名字亮在画面里："右边·远"
        self._crowd = 0  # 没往他那边转：画面里还有几个没挂名字的人
        self._look_until: float | None = None  # 环顾看到陌生人：看到什么时候
        self._presses = {"left": 0, "right": 0}
        self._obs: Obs | None = None
        self.state, self.note, self.aborted = "running", "", False
        self.friends: tuple[str, ...] = ()
        self.strangers = 0

    # ---- 三种计划 ----
    @classmethod
    def lost(cls, who: str, side: str | None, edge_exit: bool, can_call: bool, cfg: AttentionConfig, track: TrackConfig,
             heading: Heading, now: float) -> Search:
        """找刚走开的好友。从画面边上出去的：先往那边转，再喊一声看名字贴在哪边；在中间淡掉的：先喊一声，
        画面里没人再往他那边转。side = 他最后在画面哪半边（不知道是 None）。"""
        n = cfg.lost_segments
        if edge_exit and side:
            steps = [("turn", side, n), ("call",), ("edge", n, None)]
        else:
            steps = [("call",), ("edge", n, ("alone", side, n) if side else None)]
        if not can_call:
            steps = [s for s in steps if s[0] != "call"]
        return cls("lost", steps, cfg, track, heading, now, who=who, side=side)

    @classmethod
    def find(cls, who: str, side: str | None, edge_exit: bool, can_call: bool, cfg: AttentionConfig, track: TrackConfig,
             heading: Heading, now: float) -> Search:
        """大脑让找的人：刚走开过就按找走开的好友那样找；没线索就先喊一声，名字贴在哪边就往哪边转，没有就转一圈。"""
        if side is not None:
            s = cls.lost(who, side, edge_exit, can_call, cfg, track, heading, now)
            s.kind = "find"
            return s
        steps = [("call",), ("edge", cfg.lost_segments, "circle")]
        if not can_call:
            steps = steps[1:]
        return cls("find", steps, cfg, track, heading, now, who=who)

    @classmethod
    def scan(cls, cfg: AttentionConfig, track: TrackConfig, heading: Heading, rng: random.Random, prefer: str,
             now: float) -> Search:
        """环顾一片：往最久没看过的方位转 scan_segments 段。"""
        direction, where = heading.stalest(prefer)
        lo, hi = cfg.scan_segments
        return cls("scan", [("turn", direction, rng.randint(lo, hi))], cfg, track, heading, now, side=direction, where=where)

    # ---- 每圈 ----
    def step(self, obs: Obs | None, now: float) -> SearchStep:
        """obs = None：画面暂停 / 黑着，不判也不动（照样算没进展）。"""
        if self.state != "running":
            return SearchStep(self.state, note=self.note)
        if obs is not None:
            done = self._check(obs, now)
            if done is not None:
                return done
        if now - self._progress > self.cfg.resume_within + EPS:
            self.aborted = True
            return self._finish("none", "被打断太久，不找了")
        if obs is None:
            return SearchStep("wait", note="画面暂停着")
        if self._look_until is not None:
            return SearchStep("wait", note=self.describe())
        self._obs = obs
        return self._advance(now)

    def _check(self, obs: Obs, now: float) -> SearchStep | None:
        if self.kind == "scan":
            if obs.friends:
                self.friends = obs.friends
                return self._finish("seen", "看到了" + "、".join(obs.friends))
            if obs.strangers:
                self.strangers = max(self.strangers, obs.strangers)
                if self._look_until is None:
                    self._look_until, self._progress = now + self.cfg.scan_look, now
            if self._look_until is not None and now >= self._look_until - EPS:
                return self._finish("seen", f"有 {self.strangers} 个陌生人")
            return None
        if self._call_found is not None:
            return self._finish("found", f"{self.who}（{self._call_found}）")
        if obs.target_x is not None:
            return self._finish("found", f"{self.who}（{obs.target_where or side_of(obs.target_x, obs.width)}）")
        if obs.maybe_x is not None:
            return self._finish("maybe", f"{self.who}（没看到名字，{side_of(obs.maybe_x, obs.width)}）")
        return None

    def _advance(self, now: float) -> SearchStep:
        while self._i < len(self._steps):
            st = self._steps[self._i]
            if st[0] == "call":
                if self._calling:
                    return SearchStep("wait", note=self.describe())
                return SearchStep("call", note=f"喊一声找{self.who}")
            if st[0] == "edge":
                _, n, fallback = st
                if self._edge is not None:
                    self._steps[self._i] = ("turn", self._edge, n)
                elif fallback == "circle":
                    self._steps[self._i] = ("turn", self.heading.stalest("right")[0], SECTORS)
                elif fallback is not None:
                    self._steps[self._i] = fallback
                else:
                    self._i += 1
                continue
            if st[0] == "alone":
                _, direction, n = st
                crowd = self._obs.strangers if self._obs is not None else 0
                if crowd:
                    self._crowd = crowd
                    self._i += 1
                else:
                    self._steps[self._i] = ("turn", direction, n)
                continue
            _, direction, segs = st
            if self._segs_left is None:
                self._segs_left, self._presses_left = segs, self.cfg.seg_presses
                self._progress = now
            if self._dwell_until is not None:
                if now < self._dwell_until - EPS:
                    return SearchStep("wait", note=self.describe())
                self._dwell_until = None
                self._segs_left -= 1
                self._progress = now
                if self._segs_left <= 0:
                    self._i += 1
                    self._segs_left = None
                    continue
                self._presses_left = self.cfg.seg_presses
            if now - self._last_press < self.track.settle - EPS:
                return SearchStep("wait", note=self.describe())
            return SearchStep("press", Turn(direction, self.track.nudge_max), self.describe())
        return self._finish("empty" if self.kind == "scan" else "none", self._none_text())

    # ---- 调用方回报 ----
    def pressed(self, turn: Turn, now: float) -> None:
        self._last_press = self._progress = now
        self._presses[turn.direction] += 1
        self.heading.turned(turn.direction, turn.seconds, now)
        self._presses_left -= 1
        if self._presses_left <= 0:
            self._dwell_until = now + self.track.settle + self.cfg.dwell  # 最后一下停稳了再停 dwell 秒看

    def call_sent(self, now: float) -> None:
        """喊出去了，结果还要等呼喊窗口结束。"""
        self._calling, self._called_once, self._progress = True, True, now

    def called(self, seen: CallSeen | None, now: float) -> None:
        """喊一声有结果了；seen = None：没喊成（额度、关着）或一直没等到结果，接着下一步。"""
        self._calling, self._progress = False, now
        if self._i < len(self._steps) and self._steps[self._i][0] == "call":
            self._i += 1
        if seen is None or self.who is None:
            return
        for name, s in seen.friends.items():
            if name == self.who or similar(self.who, name, 0.75):
                if s.on_screen:
                    self._call_found = f"{s.side}·{s.distance}" if s.distance else s.side
                else:
                    self._edge = "left" if s.side == "左边" else "right"
                return

    # ---- 文字 ----
    def dirs(self) -> str:
        """往哪几边转过："左边、右边"；没转过是空串。"""
        return "、".join(SIDE[d] for d in ("left", "right") if self._presses[d])

    def describe(self) -> str:
        if self.kind == "scan":
            return f"在看看周围：往{self.where}（那边很久没看了）"
        if self.kind == "find":
            why = "大脑让找的"
        else:
            why = f"他刚从{SIDE[self.side]}走了" if self.side else "他刚走开"
        segs = math.ceil(sum(self._presses.values()) / self.cfg.seg_presses)
        doing = "，喊了一声在等" if self._calling else (f"，往{self.dirs()}转了 {segs} 段" if segs else "")
        return f"在找：{self.who}（{why}{doing}）"

    def _none_text(self) -> str:
        if self.kind == "scan":
            return "附近没人"
        if sum(self._presses.values()) >= SECTORS * self.cfg.seg_presses:
            head = f"转了一圈没找到{self.who}"
        elif self.dirs():
            head = f"往{self.dirs()}找了找，没看到{self.who}"
        else:
            head = f"没看到{self.who}"
        if self._called_once:
            head += "，喊了一声也没看到他的名字"
        if self._crowd:
            head += f"（画面里还有 {self._crowd} 个没挂名字的人，可能就是他）"
        return head

    def result_line(self) -> str:
        """status 里"刚才…"那一句。"""
        if self.kind == "scan":
            return f"往{self.where}看了看：{self.note}"
        if self.state == "found":
            return f"找到了{self.note}"
        if self.state == "maybe":
            return f"可能找到了{self.note}"
        if self.aborted:
            return f"找{self.who}，{self.note}"
        return self.note

    def _finish(self, state: str, note: str) -> SearchStep:
        self.state, self.note = state, note
        log.debug("找（%s%s）结束：%s %s", self.kind, f" {self.who}" if self.who else "", state, note)
        return SearchStep(state, note=note)


def event_text(s: Search, prev_empty: bool) -> str | None:
    """找完了值不值得告诉大脑（背景事件 search）；prev_empty = 上一次环顾是"附近没人"。find 走 task_done / task_failed，不在这里。"""
    if s.aborted or s.state == "running":
        return None
    if s.kind == "lost":
        act = f"往{s.dirs()}找了找" if s.dirs() else ("喊了一声找" if s._called_once else "找了找")
        if s.state == "maybe":
            return f"你{act}刚走开的{s.who}，那边有个人可能是他（没看到名字）"
        if s.state == "none":
            return f"你{act}刚走开的{s.who}，没看到他"
        return None  # 找到了：他回到身边，走现有的 return
    if s.kind == "scan" and s.state == "seen" and not s.friends and prev_empty:
        return f"你往{s.where}看了看：有 {s.strangers} 个陌生人"
    return None
```

注意 `test_lost_middle_with_unnamed_people_does_not_turn_away` 里 `run` 默认的 `no_call` 没调 `call_sent`，所以 `_called_once` 是 False，事件是"你找了找…"。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search.py -q`
Expected: 全部通过（Task 2 的 5 个 + Task 3 的 18 个）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/search.py tests/test_brain_search.py
git commit -m "feat(attention): Search——找走开的好友、环顾、大脑的目标三种计划

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 注意力：删随意看、接上找

**Files:**
- Modify: `src/skydango/brain/attention.py`（整个文件按下面重写）
- Modify: `src/skydango/config.py`（删 `wander_*` 四项、废弃键跳过）
- Modify: `config.example.toml`（删 `wander_*` 四行）
- Modify: `src/skydango/inner/effects.py:17`（注释）
- Test: `tests/test_brain_attention.py`、`tests/test_config.py`

**Interfaces:**
- Consumes: Task 3 的 `Heading` `Obs` `Search` `SearchStep` `TERMINAL` `event_text`
- Produces（Task 5 用）：
  - `Attention(cfg, track, rng, now, hfov: float = 90.0)`；属性 `heading: Heading`、`search: Search | None`、`last_search: tuple[Search, float] | None`
  - `think(targets, now, scale: float = 1.0, obs: Obs | None = None) -> Thought`
  - `Thought(current, action, centered, look_first, search: SearchStep | None = None, event: str | None = None)`（**删掉了 `wandering`**）
  - `start_lost(who: str, side: str | None, edge_exit: bool, can_call: bool, now: float) -> bool`
  - `lost_bearing(now)`（只清朝向）、`external_move(now)`（原来的 + 清朝向）
  - `describe(now: float | None = None) -> str`
  - 常量 `SEARCH_MODES = ("随意", "好奇", "专心")`

- [ ] **Step 1: 改测试**

`tests/test_brain_attention.py`：

1. 顶上加 `from skydango.brain.search import Obs`
2. **删掉**这几个随意看的测试和辅助函数：`test_wander_after_random_quiet_interval`、`test_wander_never_same_side_more_than_limit`、`test_target_interrupts_wander`、`first_wander`、`test_wander_scale_and_curious_mode_shorten_interval`
3. `test_focused_mode_ignores_low_interest_and_no_wander` 改名 `test_focused_mode_ignores_low_interest`，第一行改成 `a = attn()`，断言里去掉 `and not th.wandering`
4. `test_still_mode_never_moves_nor_looks_first` 第一行改成 `a = attn()`
5. 改完 `grep -n wander tests/test_brain_attention.py` 应该没有输出
6. `run_idle` 加一个 `obs` 参数（默认 None = 不知道画面里有没有人，不会环顾，老测试照旧）：

```python
def run_idle(a, t0, t1, step=0.7, targets=(), obs=None):
    """空转：每圈 think，给了按键就当真按了；返回 [(时间, Thought)]。obs 是函数 t → Obs（None：不给，注意力不知道画面里有没有人）。"""
    out, t = [], t0
    while t <= t1:
        th = a.think(list(targets), t, obs=obs(t) if obs else None)
        if th.action is not None:
            a.pressed(th.action, t)
        out.append((t, th))
        t += step
    return out
```

7. 文件末尾加：

```python
# ---- 有意识地找（plan 2026-10-03-attention-search Task 4） ----

def empty(t):
    return Obs()


def test_no_motive_no_press():  # 以前到点就随意看；现在没人、还没到环顾时间就不动
    a = attn()
    assert all(th.action is None for t, th in run_idle(a, 0.0, 15.0, obs=empty))
    assert all(th.action is None for t, th in run_idle(attn(), 0.0, 60.0))  # 不知道画面里有没有人：不环顾


def test_scan_after_empty_for_a_while_then_every_interval():
    a = attn(scan_segments=[2, 2])
    log = run_idle(a, 0.0, 100.0, obs=empty)
    pressed = [t for t, th in log if th.action is not None]
    assert pressed[0] == pytest.approx(Config().attention.scan_after, abs=0.7)
    ends = [t for t, th in log if th.search is not None and th.search.state == "empty"]
    assert len(ends) == 2 and ends[1] - ends[0] >= Config().attention.scan_every  # 约 27 s、94 s
    assert a.describe(ends[1] + 1).startswith("刚才往") and a.describe(ends[1] + 1).endswith("附近没人（1 秒前）")


def scan_ends(a, scale=1.0, t1=300.0):
    """空场景跑 t1 秒（每 0.7 s 一圈、给了按键就当按了），返回每次环顾结束（附近没人）的时间。"""
    out, t = [], 0.0
    while t <= t1:
        th = a.think([], t, scale=scale, obs=Obs())
        if th.action is not None:
            a.pressed(th.action, t)
        if th.search is not None and th.search.state == "empty":
            out.append(t)
        t += 0.7
    return out


def test_scan_modes_and_scale():
    base = scan_ends(attn(scan_segments=[2, 2]))  # 间隔 60 s：约 5 次
    curious = attn(scan_segments=[2, 2])
    curious.set_mode("好奇", None)  # 间隔 ×0.5
    sleepy = scan_ends(attn(scan_segments=[2, 2]), scale=3.0)  # 困了：间隔 ×3
    assert len(scan_ends(curious)) > len(base) > len(sleepy) >= 1
    focused = attn()
    focused.set_mode("专心", None)
    assert scan_ends(focused) == []


def test_passive_target_interrupts_search_and_search_resumes():
    a = attn()
    assert a.start_lost("小明", "left", True, False, 0.0)
    th = a.think([], 0.0, obs=Obs())
    assert th.action == Turn("left", Config().track.nudge_max)
    a.pressed(th.action, 0.0)
    th = a.think([tgt("t:1", "talk_friend", 1600, "小红", 1.0)], 1.0, obs=Obs())
    assert th.current.key == "t:1" and th.action.direction == "right"  # 有人说话：先看他
    a.pressed(th.action, 1.0)
    th = a.think([], 3.0, obs=Obs())
    assert th.current is None and th.search is not None and a.search is not None  # 说完了：接着找


def test_search_dropped_after_long_interruption():
    a = attn()
    a.start_lost("小明", "left", True, False, 0.0)
    for i in range(20):  # 小红一直在说话（每秒新的一句：不腻）
        a.think([tgt("t:1", "talk_friend", 1000, "小红", float(i))], float(i), obs=Obs())
    th = a.think([], 20.0, obs=Obs())
    assert th.search.state == "none" and a.search is None and th.event is None  # 被打断太久：不找了、不报


def test_standing_friend_does_not_block_lost_search():
    a = attn()
    a.start_lost("小明", "left", True, False, 0.0)
    th = a.think([tgt("n:小红", "friend_present", 1600, "小红")], 0.0, obs=Obs(friends=("小红",)))
    assert th.action == Turn("left", Config().track.nudge_max) and a.describe(0.0).startswith("在找：小明")


def test_lost_search_finishes_with_event():
    a = attn()
    a.start_lost("小明", "left", True, False, 0.0)
    events = [th.event for t, th in run_idle(a, 0.0, 15.0, obs=empty) if th.event]  # 15 s：还没到环顾
    assert events == ["你往左边找了找刚走开的小明，没看到他"]
    assert a.search is None and a.describe(15.0).startswith("刚才往左边找了找，没看到小明（")


def test_scan_event_only_after_empty_scan():
    a = attn(scan_segments=[2, 2])
    run_idle(a, 0.0, 35.0, obs=empty)  # 第一次环顾：附近没人
    assert a.last_search[0].state == "empty"
    log = run_idle(a, 35.7, 120.0, obs=lambda t: Obs(strangers=1) if a.search is not None else Obs())
    assert [th.event for t, th in log if th.event] == [f"你往{a.last_search[0].where}看了看：有 1 个陌生人"]


def test_start_lost_respects_switch_and_mode():
    assert not attn(search=False).start_lost("小明", "left", True, False, 0.0)
    a = attn()
    a.set_mode("别动", None)
    assert not a.start_lost("小明", "left", True, False, 0.0)
    b = attn()
    assert b.start_lost("小明", "left", True, False, 0.0) and not b.start_lost("小明", "left", True, False, 1.0)
    b.set_mode("别动", None)
    assert b.search is None  # 改成别动：正在找的也不找了


def test_external_move_resets_heading():
    a = attn()
    a.start_lost("小明", "left", True, False, 0.0)
    th = a.think([], 0.0, obs=Obs())
    a.pressed(th.action, 0.0)
    assert a.heading.deg != 0.0
    a.external_move(1.0)
    assert a.heading.deg == 0.0


def test_passive_turn_moves_heading_too():
    a = attn()
    th = a.think([tgt("t:1", "talk_friend", 1600, "小明", 0.0)], 0.0)
    a.pressed(th.action, 0.0)
    assert a.heading.deg > 0.0  # 按右
```

`tests/test_config.py` 末尾加（文件顶上没有 `import logging` 就加上）：

```python
def test_removed_wander_keys_are_skipped_with_warning(tmp_path, caplog):  # spec 2026-10-03-attention-search §2
    p = tmp_path / "c.toml"
    p.write_text("[attention]\nwander_min = 8.0\nwander_presses = [2, 4]\nsearch = false\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(p)
    assert not hasattr(cfg.attention, "wander_min") and cfg.attention.search is False
    assert "attention.wander_min" in caplog.text and "attention.wander_presses" in caplog.text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_attention.py tests/test_config.py -q`
Expected: FAIL（`think() got an unexpected keyword argument 'obs'`、`no attribute 'start_lost'` 之类；`test_removed_wander_keys…` 断言 `hasattr` 失败）

- [ ] **Step 3: 配置删 `wander_*`、废弃键跳过**

`src/skydango/config.py`：

1. 删 `AttentionConfig` 里 `wander_min` `wander_max` `wander_presses` `wander_same_side` 四行
2. 顶上 import 区加 `import logging`，import 之后加 `log = logging.getLogger(__name__)`（已经有就不加）
3. `_merge` 前面加：

```python
# 删掉的配置项：旧的 config.toml / console.toml 里还写着时跳过、提醒一句（别的未知键照旧报错）
DEPRECATED = frozenset({
    "attention.wander_min", "attention.wander_max", "attention.wander_presses", "attention.wander_same_side",  # 2026-10-03 随意看删了
})
```

4. `_merge` 里 `if key not in known:` 改成：

```python
        if key not in known:
            if f"{path}{key}" in DEPRECATED:
                log.warning("配置项 %s%s 已经不用了，可以删掉", path, key)
                continue
            raise ValueError(f"未知配置项: {path}{key}")
```

`config.example.toml` 删 `wander_min` `wander_max` `wander_presses` `wander_same_side` 四行。

`src/skydango/inner/effects.py:17` 注释改成：

```python
    wander: float = 1.0  # 空闲注意力环顾的间隔倍数（> 1 看得少；困了看得少，和 idle 方向相反）
```

- [ ] **Step 4: 重写 `src/skydango/brain/attention.py`**

整个文件换成（被动注意那部分逐字照旧，只删了随意看、加了找）：

```python
"""空闲注意力（东张西望）：团子闲着时"现在最想看什么、在找什么"，纯决策
（spec 2026-09-30-idle-attention §2 / §3；2026-10-03-attention-search §2）。

被动注意：每圈给一组候选目标（说话的人、走近的人、对团子做动作的人、站着的好友），按基础兴趣 ×（1 − 看腻）挑一个看；
目标在画面中间带里就不动、慢慢看腻，不在就小步转过去（同 track：目标在右边按右，把它拉向中间）；
换目标要高出一截、刚换过的一会儿内不换（防抖）。
有意识地找（search.py）：没有被动目标时推进正在做的找——找刚走开的好友（Body 调 start_lost 发起）、
画面里一阵子没人就往最久没看过的方向环顾一片（这里自己发起）。站着的好友不打断找。没有动机就不转。不回位。
这里不碰设备：Body 判断闲不闲、真按了键再调 pressed()；"喊一声"交给 Body，结果由 Body 交给 search.called()。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import AttentionConfig, TrackConfig
from .peek import Turn
from .search import TERMINAL, Heading, Obs, Search, SearchStep, event_text

log = logging.getLogger(__name__)

KINDS = ("talk_friend", "act_on_me", "approach", "talk_stranger", "friend_present")
KIND_NOTES = {"talk_friend": "在说话", "act_on_me": "在对你做动作", "approach": "朝你走过来",
              "talk_stranger": "在说话", "friend_present": "站在那"}
FRESH_KINDS = ("talk_friend", "talk_stranger", "approach", "act_on_me")  # 刚出现时值得"先看一眼再开面板"
MODES = ("随意", "好奇", "专心", "别动")
SCAN_MODE = {"随意": 1.0, "好奇": 0.5}  # 环顾间隔的倍数；专心 / 别动不环顾
SEARCH_MODES = ("随意", "好奇", "专心")  # 会找刚走开的好友
RESULT_KEEP = 120.0  # status 里"刚才…"留多久（秒）
EPS = 1e-6


@dataclass(frozen=True)
class Target:
    key: str  # "t:<轨迹 id>" 或 "n:<名字>"：同一个人
    kind: str  # KINDS 之一
    x: float  # 框中心 x（整图像素）
    who: str | None  # 名字；陌生人 None
    fresh: float | None = None  # 这次刺激开始的时间（气泡 start / 走近 / 动作）；变了就把看腻清零


@dataclass(frozen=True)
class Thought:
    current: Target | None  # 现在在看的（被动注意）
    action: Turn | None  # 这一圈想按的（门槛不开时 Body 不按）
    centered: bool  # current 已经在中间带里
    look_first: bool  # current 是刚出现的说话 / 走近 / 做动作：值得先看再开面板
    search: SearchStep | None = None  # 这一圈找的那一步（没在找 / 被被动注意打断是 None）
    event: str | None = None  # 找完了、值得告诉大脑的一句（背景事件 search）


class Attention:
    def __init__(self, cfg: AttentionConfig, track: TrackConfig, rng: random.Random, now: float, hfov: float = 90.0) -> None:
        self.cfg, self.track, self.rng = cfg, track, rng
        self.width = 1920
        self.mode = "随意"
        self.focus: str | None = None
        self._last_now = now
        self._bored: dict[str, float] = {}
        self._fresh: dict[str, float | None] = {}
        self._current: Target | None = None
        self._switched_at = float("-inf")
        self._last_press = float("-inf")
        self._err = 0.0  # 这一圈 current 离中线多远（给 pressed 记转不动用）
        self._streak_dir: str | None = None  # 转不动（同 track）：同方向连按几下、开始时离中线多远
        self._streak_n = 0
        self._streak_err = 0.0
        self._stuck: dict[str, float] = {}  # 转不动的人卡在哪（x）：他挪开之前不再朝他按，留在原地看着
        self._thought = Thought(None, None, False, False)
        # 有意识地找
        self.heading = Heading(cfg.press_deg, track.nudge_max, hfov, now)
        self.search: Search | None = None
        self.last_search: tuple[Search, float] | None = None  # 最近一次找完的：(那次找, 结束时间)
        self._scan_empty = False  # 上次环顾是"附近没人"
        self._empty_since: float | None = None  # 画面里从什么时候起一个人都没有（None = 有人 / 不知道）
        self._scan_end = float("-inf")
        self._scan_side = "left"  # 上次环顾往哪边：下次两边一样近时换一边

    def set_mode(self, mode: str, focus: str | None) -> None:
        """大脑定注意力模式（MODES 之一）和关注谁（空 = 不特别关注）。新模式不找的，正在找的也不找了。"""
        if mode not in MODES:
            raise ValueError(f"不认识的模式：{mode}")
        self.mode = mode
        self.focus = (focus or "").strip() or None
        s = self.search
        if s is not None and (mode not in SEARCH_MODES or (s.kind == "scan" and mode not in SCAN_MODE)):
            log.debug("注意力改成%s：不找了（%s）", mode, s.describe())
            self.search = None

    def _focused(self, t: Target) -> bool:
        return bool(self.focus and t.who and (t.who == self.focus or similar(self.focus, t.who, 0.75)))

    # ---- 兴趣 ----
    def base(self, t: Target) -> float:
        b = float(getattr(self.cfg, t.kind))
        if self.mode == "好奇" and t.kind == "talk_stranger":
            b = max(b, 0.7)
        if self._focused(t):
            b = max(b, self.cfg.focus_interest)
        return b

    def _centered(self, t: Target) -> bool:
        return abs(t.x - self.width / 2) <= self.cfg.center_band * self.width / 2

    def interest(self, t: Target) -> float:
        return self.base(t) * (1.0 - self._bored.get(t.key, 0.0))

    def _merge(self, targets: list[Target]) -> dict[str, Target]:
        """同一个人几种目标：只留基础兴趣最高的那种。"""
        out: dict[str, Target] = {}
        for t in targets:
            if t.key not in out or self.base(t) > self.base(out[t.key]):
                out[t.key] = t
        return out

    def _update_boredom(self, merged: dict[str, Target], dt: float) -> None:
        for key, t in merged.items():
            if t.fresh is not None and self._fresh.get(key) != t.fresh:
                self._fresh[key] = t.fresh
                self._bored[key] = 0.0  # 新的一句 / 新的一次走近：又有意思了
        for key in set(self._bored) | set(merged):
            t = merged.get(key)
            b = self._bored.get(key, 0.0)
            if t is not None and self._centered(t):
                b += self.cfg.bore_rate * dt * (0.5 if self._focused(t) else 1.0)
            else:
                b -= self.cfg.recover_rate * dt
            b = min(max(b, 0.0), 1.0)
            if b == 0.0 and t is None:
                self._bored.pop(key, None)
                self._fresh.pop(key, None)
            else:
                self._bored[key] = b

    def _choose(self, merged: dict[str, Target], now: float) -> Target | None:
        ok = [t for t in merged.values() if self.interest(t) >= self.cfg.min_interest]
        best = max(ok, key=self.interest, default=None)
        cur = merged.get(self._current.key) if self._current is not None else None
        if cur is not None and self.interest(cur) < self.cfg.min_interest:
            cur = None
        if cur is None:
            pick = best
        elif (best is not None and best.key != cur.key and now - self._switched_at >= self.cfg.switch_hold
              and self.interest(best) >= self.interest(cur) + self.cfg.switch_margin):
            pick = best
        else:
            pick = cur
        if (pick is None) != (self._current is None) or (pick is not None and pick.key != self._current.key):
            self._switched_at = now
            self._streak_dir, self._streak_n = None, 0
            if pick is not None:
                log.debug("注意力换到 %s（%s，兴趣 %.2f）", pick.who or "陌生人", pick.kind, self.interest(pick))
        return pick

    # ---- 每圈 ----
    def think(self, targets: list[Target], now: float, scale: float = 1.0, obs: Obs | None = None) -> Thought:
        """scale：环顾间隔的倍数（心情精力）；obs：这一圈的感知结果（None = 画面暂停 / 不知道，不发起环顾）。"""
        dt = min(max(now - self._last_now, 0.0), self.cfg.max_step)
        self._last_now = now
        self._note_empty(obs, now)
        merged = self._merge(targets)
        if self.mode == "专心":  # 只看分量重的（好友说话、对团子做事、关注的人）
            merged = {k: t for k, t in merged.items() if self.base(t) >= self.cfg.act_on_me}
        self._update_boredom(merged, dt)
        cur = self._current = self._choose(merged, now)
        action, centered, look_first = None, False, False
        if cur is not None:
            centered = self._centered(cur)
            look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
            if not centered and now - self._last_press >= self.track.settle:
                action = self._turn(cur)
                if action is None:  # 转不动、刚加了看腻：这一圈就重新挑（可能换走）
                    cur = self._current = self._choose(merged, now)
                    if cur is not None:
                        centered = self._centered(cur)
                        look_first = cur.kind in FRESH_KINDS and cur.fresh is not None and now - cur.fresh <= self.cfg.look_first
        step = event = None
        passive = cur is not None and not (cur.kind == "friend_present" and self.search is not None)  # 站着的好友不打断找
        if not passive and self.mode != "别动":
            step, event = self._search_step(obs, now, scale)
            if step is not None:
                action = step.turn if step.state == "press" else None
        if self.mode == "别动":
            action, look_first = None, False
        self._thought = Thought(cur, action, centered, look_first, step, event)
        return self._thought

    def _note_empty(self, obs: Obs | None, now: float) -> None:
        if obs is None:
            return
        if obs.friends or obs.strangers:
            self._empty_since = None
        elif self._empty_since is None:
            self._empty_since = now

    def _scan_due(self, now: float, scale: float) -> bool:
        if not self.cfg.search or self.mode not in SCAN_MODE or self._empty_since is None:
            return False
        gap = self.cfg.scan_every * scale * SCAN_MODE[self.mode]
        return now - self._empty_since >= self.cfg.scan_after - EPS and now - self._scan_end >= gap - EPS

    def _search_step(self, obs: Obs | None, now: float, scale: float) -> tuple[SearchStep | None, str | None]:
        if self.search is None and self._scan_due(now, scale):
            prefer = "right" if self._scan_side == "left" else "left"
            self.search = Search.scan(self.cfg, self.track, self.heading, self.rng, prefer, now)
            self._scan_side = self.search.side or prefer
            log.debug("注意力：%s", self.search.describe())
        if self.search is None:
            return None, None
        step = self.search.step(obs, now)
        if step.state not in TERMINAL:
            return step, None
        return step, self._finish_search(now)

    def _finish_search(self, now: float) -> str | None:
        s, self.search = self.search, None
        self.last_search = (s, now)
        event = event_text(s, self._scan_empty)
        if s.kind == "scan":
            self._scan_end = now
            if not s.aborted:
                self._scan_empty = s.state == "empty"
        return event

    def start_lost(self, who: str, side: str | None, edge_exit: bool, can_call: bool, now: float) -> bool:
        """Body：好友刚走开，去找找（换掉正在做的环顾 / 找别人）。开关关着、模式不找、已经在找他：返回 False。"""
        if not self.cfg.search or self.mode not in SEARCH_MODES:
            return False
        s = self.search
        if s is not None and s.kind == "lost" and s.who == who:
            return False
        if s is not None:
            log.debug("去找%s，先不%s", who, s.describe())
            if s.kind == "scan":
                self._scan_end = now
        self.search = Search.lost(who, side, edge_exit, can_call, self.cfg, self.track, self.heading, now)
        return True

    def lost_bearing(self, now: float) -> None:
        """走路、黑屏：不知道朝哪了。"""
        self.heading.reset(now)

    def external_move(self, now: float) -> None:
        """别人（大脑、track、look_person 换角度、面板开关）刚动过镜头：位置都变了，等画面停稳再按，转不动重新算，朝向也不算数了。"""
        self._streak_dir, self._streak_n = None, 0
        self._stuck.clear()
        self._last_press = max(self._last_press, now)
        self.heading.reset(now)

    def _turn(self, cur: Target) -> Turn | None:
        half = self.width / 2
        err = abs(cur.x - half)
        direction = "right" if cur.x > half else "left"  # 按右键画面里的东西往左移：目标在右边就按右
        stuck = self._stuck.get(cur.key)
        if stuck is not None:
            if abs(cur.x - stuck) < 2 * self.track.stall_px:
                return None  # 还卡在那：看着他就好，别一直往一边按
            del self._stuck[cur.key]  # 他挪开了，再试试
        if (direction == self._streak_dir and self._streak_n >= self.track.stall_nudges
                and self._streak_err - err < self.track.stall_px):
            self._bored[cur.key] = min(1.0, self._bored.get(cur.key, 0.0) + self.cfg.stuck_bored)
            self._stuck[cur.key] = cur.x
            self._streak_dir, self._streak_n = None, 0
            log.debug("注意力转不动 %s，看腻一截", cur.who or "陌生人")
            return None
        self._err = err
        seconds = min(max(self.cfg.gain * err / half, self.track.nudge_min), self.track.nudge_max)
        return Turn(direction, round(seconds, 3))

    def pressed(self, turn: Turn, now: float) -> None:
        """Body 真按了这一下：找的那一步交给 search；被动注意记按键时间、转不动计数（隔太久说明别人可能动过镜头，重新计数）。"""
        th = self._thought
        if th.search is not None and th.search.state == "press" and self.search is not None:
            self.search.pressed(turn, now)  # 里面会转 heading
            self._last_press = now
            return
        self.heading.turned(turn.direction, turn.seconds, now)
        if (turn.direction != self._streak_dir or now - self._last_press > 3 * self.track.settle
                or self._streak_err - self._err >= self.track.stall_px):  # 换方向 / 隔太久 / 有进展：重新计数（同 track）
            self._streak_dir, self._streak_n, self._streak_err = turn.direction, 0, self._err
        self._streak_n += 1
        self._last_press = now

    def describe(self, now: float | None = None) -> str:
        """status 一行；now 给了才写"刚才…（几秒前）"。"""
        cur = self._thought.current
        if cur is not None and not (cur.kind == "friend_present" and self.search is not None):
            line = f"在看：{cur.who or '陌生人'}（{KIND_NOTES[cur.kind]}）"
        elif self.search is not None:
            line = self.search.describe()
        elif self.last_search is not None and now is not None and now - self.last_search[1] <= RESULT_KEEP:
            s, t = self.last_search
            line = f"刚才{s.result_line()}（{now - t:.0f} 秒前）"
        else:
            line = "没在看什么"
        if self.mode != "随意" or self.focus:
            head = f"注意力：{self.mode}" + (f"，关注{self.focus}" if self.focus else "")
            line = f"{head}；{line}"
        return line
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_attention.py tests/test_config.py tests/test_brain_search.py tests/test_inner_effects.py -q`
Expected: PASS。有失败先看是不是测试里的时间点和实现差一圈（`run_idle` 步长 0.7 s），按实现的真实行为改断言里的容差，别改实现去凑测试；行为本身不对才改实现。

- [ ] **Step 6: 身体那边先接上新签名**（让全量测试不挂；真正的接线在 Task 5）

`src/skydango/brain/body.py`：
- `Attention(cfg.attention, cfg.track, self.rng, clock())` 改成 `Attention(cfg.attention, cfg.track, self.rng, clock(), hfov=cfg.spin.hfov)`
- `_watch_attention` 里 `a.think(self._attention_targets(now), now, self.effects().wander)` 不用改（`scale` 还是第三个位置参数）
- 三处 `self.attention.describe()`（约 311、641、1663 行）改成 `self.attention.describe(self.clock())`

Run（后台）：`.venv\Scripts\python.exe -m pytest -q`
Expected: 全部通过。挂掉的如果是 `tests/test_brain_attention_body.py` 里依赖随意看的（目前 grep 没有），按"没动机就不动"改断言。

- [ ] **Step 7: 提交**

```bash
git add src/skydango/brain/attention.py src/skydango/brain/body.py src/skydango/config.py src/skydango/inner/effects.py config.example.toml tests/test_brain_attention.py tests/test_config.py
git commit -m "feat(attention): 删掉随意看，没有被动目标时推进有意识的找；环顾由注意力自己发起

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 身体接线：找走开的好友、喊一声、`search` 事件

**Files:**
- Modify: `src/skydango/brain/body.py`
- Modify: `src/skydango/brain/events.py`
- Test: `tests/test_brain_search_body.py`（新建）、`tests/test_brain_events.py`（没有就新建；有就加在末尾）

**Interfaces:**
- Consumes: Task 4 的 `Attention.start_lost` / `think(..., obs=)` / `Thought.search` / `Thought.event` / `lost_bearing` / `SEARCH_MODES`；Task 3 的 `Obs`、`TERMINAL`；`CallResult`（`brain/calling.py`）
- Produces（Task 6 用）：
  - `Body.search_obs(who: str | None, now: float) -> Obs | None`
  - `Body._left_where(name: str) -> tuple[str | None, bool]`
  - 模块常量 `EXIT_EDGE = 0.15`

- [ ] **Step 1: 写失败的测试**

`tests/test_brain_events.py`（没有就新建，带 `from skydango.brain.events import BACKGROUND, EventQueue`）末尾：

```python
def test_search_events_are_background_and_keep_latest():
    t = [0.0]
    q = EventQueue(clock=lambda: t[0])
    q.put("search", "你往左边找了找刚走开的小明，没看到他")
    t[0] = 5.0
    q.put("search", "你往右前方看了看：有 1 个陌生人")
    assert "search" in BACKGROUND
    assert [e.text for e in q.drain()] == ["你往右前方看了看：有 1 个陌生人"]
```

（先看一眼 `EventQueue` 有没有 `drain()`，现有测试用的就是它；没有就照现有测试取事件的写法。）

新建 `tests/test_brain_search_body.py`：

```python
"""身体接上有意识地找（plan 2026-10-03-attention-search Task 5）：好友走开往他那边找、中间淡掉先喊、环顾、search 事件。"""

from test_brain_attention_body import attn_body, steps

from skydango.brain.calling import CallResult
from skydango.vision.people import CallSeen, Seen


def quiet(b):
    """别让聊天面板定时看一眼挡住注意力；只看找。"""
    b.cfg.panel.idle_peek = 1e9


def leave(b, env, clock, name="小明", x=60):
    """小明在身边、名字标签最后在 x，然后走开（不经过"来了"，免得叫开聊天面板）。
    标签时间往前拨 6 秒：真机上"走开"是 keep（5 s）没看到名字才判的，标签早就过了 track.max_age，不算"还在画面里"。"""
    env.labels[name] = (x, 300, 80, 20, clock() - 6.0)
    b._nearby = {name}
    env.near = []


def search_events(events):
    return [e.text for e in events.drain() if e.kind == "search"]


def test_friend_leaving_from_left_edge_makes_attention_look_left(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.call.auto = False
    b.cfg.attention.scan_after = 1e9
    leave(b, env, clock, x=60)
    steps(b, clock, 3)
    assert "在找：小明（他刚从左边走了" in b.status()
    steps(b, clock, 30)
    assert cam.nudges and {d for d, s in cam.nudges} == {"left"}
    assert len(cam.nudges) == b.cfg.attention.lost_segments * b.cfg.attention.seg_presses
    assert search_events(events) == ["你往左边找了找刚走开的小明，没看到他"]


def test_middle_fade_calls_first_and_found_by_call(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.attention.scan_after = 1e9
    env.unnamed = lambda now: 1  # 有感知层的呼喊接口（call_available 看 hasattr）
    env.called = lambda at, by_self=True: None
    env.call_result = lambda at: CallSeen(at, {"小明": Seen("右边", "远")})
    calls = []

    def fake_call_out(reason, live=False):
        calls.append(reason)
        b._call_at = clock()
        return CallResult(clock(), reason)

    b.call_out = fake_call_out
    leave(b, env, clock, x=900)
    steps(b, clock, 6)
    assert calls == ["auto"] and cam.nudges == []
    assert "刚才找到了小明（右边·远）" in b.status()
    kinds = [e.kind for e in events.drain()]
    assert "call" in kinds and "search" not in kinds


def test_auto_call_left_to_search_unless_still_mode(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    env.unnamed = lambda now: 1
    env.called = lambda at, by_self=True: None
    env.call_result = lambda at: None
    calls = []
    b.call_out = lambda reason, live=False: (calls.append(reason), CallResult(clock(), reason))[1]
    b.attention.set_mode("别动", None)  # 注意力不找：老的自动喊照旧
    leave(b, env, clock, x=900)
    steps(b, clock, 2)
    assert calls == ["auto"]


def test_alone_scans_and_reports_empty_in_status_only(clock):
    b, dev, reader, events, env, cam = attn_body(clock)
    quiet(b)
    b.cfg.attention.scan_segments = [2, 2]
    steps(b, clock, 50)  # 35 s：20 s 后环顾一片，转完附近没人
    assert len(cam.nudges) == 2 * b.cfg.attention.seg_presses
    assert "附近没人" in b.status()
    assert search_events(events) == []  # 还是没人：只写 status


def test_dry_run_thinks_but_never_presses(clock):
    b, dev, reader, events, env, cam = attn_body(clock, live=False)
    quiet(b)
    b.cfg.call.auto = False
    leave(b, env, clock, x=60)
    steps(b, clock, 3)
    assert cam.nudges == [] and "在找：小明" in b.status() and b._attention_why == "dry-run"
```

（`clock` 是 `tests/conftest.py:115` 的 fixture，直接当参数用。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search_body.py tests/test_brain_events.py -q`
Expected: FAIL（没有发起找、`search` 不在 `BACKGROUND`）

- [ ] **Step 3: 事件**

`src/skydango/brain/events.py`：
- `BACKGROUND` 里加 `"search"`
- `Event.kind` 那行注释末尾加 `/ search`
- `put` 里 `elif kind == "stranger":` 改成 `elif kind in LATEST:`，那一段里两处 `e.kind != "stranger"` / `e.kind == "stranger"` 改成 `e.kind != kind` / `e.kind == kind`；文件顶上 `BACKGROUND` 后面加

```python
LATEST = frozenset({"stranger", "search"})  # 攒着时只留最新一条（陌生人数量来回跳、找的结果只看最近的）
```

- [ ] **Step 4: 身体**

`src/skydango/brain/body.py`：

1. import 区：`from .attention import Attention, Target as AttnTarget` 改成 `from .attention import SEARCH_MODES, Attention, Target as AttnTarget`；加 `from .search import Obs, SIDE as SEARCH_SIDE`；确认已经导入了 `similar`（`from ..chat.tracker import similar`）、`side_of`（`from ..vision.people import side_of`，没有就加）、`call_available`
2. 模块常量（`CALL_TRACK_STALE` 旁边）加：

```python
EXIT_EDGE = 0.15  # 好友走开时名字标签最后在最左 / 最右这么宽里：算从画面边上出去的（spec 2026-10-03-attention-search §1）
```

3. `__init__` 里注意力那几行后面加：

```python
        self._searched_for: dict[str, float] = {}  # 好友 → 为他哪一次走开（_left_at 的时间）找过
        self._search_call_pending = False  # _pending_auto 是"找走开的好友"喊的那一声
        self._attn_black = False  # 注意力上一圈看到的是不是黑屏（黑屏一开始就清朝向）
```

4. 新方法（放在 `_attention_targets` 后面）：

```python
    def search_obs(self, who: str | None, now: float) -> Obs | None:
        """有意识地找这一圈的感知结果：要找的人在不在（名字证实 / 只是像他）、画面里有哪些好友、几个别的人。
        画面暂停、黑屏、没有感知层：None。名字标签贴在屏幕边上的是人在画面外，不算找到。"""
        env = self.env
        if env is None or not hasattr(env, "people") or getattr(env, "paused", False) or self.blackout:
            return None
        width = self.frame_width
        people = env.people(now)
        friends = tuple(p.name for p in people if p.kind == "friend" and p.name and p.sure)
        others = sum(1 for p in people if p.kind in ("stranger", "unlit") or (p.kind == "friend" and not p.sure))
        target = maybe = None
        where = ""
        if who:
            def same(n):
                return bool(n) and (n == who or similar(who, n, 0.75))

            for p in people:
                if p.kind != "friend" or not same(p.name):
                    continue
                if p.sure:
                    target, where = p.box.x + p.box.w / 2, f"{p.side}·{p.distance}"
                    break
                if maybe is None:
                    maybe = p.box.x + p.box.w / 2
            if target is None:
                band = self.cfg.perception.edge_band * width
                for n, (x, _y, w, _h, t) in dict(getattr(env, "labels", {}) or {}).items():
                    cx = x + w / 2
                    if same(n) and now - t <= self.cfg.track.max_age and band <= cx <= width - band:
                        target, where = cx, side_of(cx, width)
                        break
        return Obs(width, target, where, maybe, friends, others)

    def _left_where(self, name: str) -> tuple[str | None, bool]:
        """好友走开时最后在哪：名字标签最后的位置 → (往哪边找 "left" / "right", 是不是从画面边上出去的)；没记录 (None, False)。
        （失踪记录 Lost 只留 keep 秒，判"走开"时多半已经删了，所以看标签。）"""
        v = dict(getattr(self.env, "labels", {}) or {}).get(name)
        if v is None:
            return None, False
        x, _y, w, _h, _t = v
        width = self.frame_width
        cx = x + w / 2
        return ("left" if cx < width / 2 else "right"), (cx < EXIT_EDGE * width or cx > (1 - EXIT_EDGE) * width)

    def _search_takes_call(self) -> bool:
        """好友走开后喊一声的事归注意力的"找走开的好友"管（_watch_call 不再自己判断）。"""
        return self._attention_on() and self.cfg.attention.search and self.attention.mode in SEARCH_MODES

    def _start_lost_search(self, now: float) -> None:
        """好友刚走开、还没回来、这次走开没找过：交给注意力去找。一次只找最近走开的那一个，同时走开的别人这次不找。"""
        window = self.cfg.call.auto_after_leave
        fresh = sorted(((t, n) for n, t in self._left_at.items()
                        if now - t <= window and n not in self._nearby and self._searched_for.get(n) != t), reverse=True)
        if not fresh:
            return
        for t, n in fresh:
            self._searched_for[n] = t
        name = fresh[0][1]
        side, edge = self._left_where(name)
        cfg = self.cfg.call
        can_call = cfg.enabled and cfg.auto and call_available(self.cfg, self.env)
        if self.attention.start_lost(name, side, edge, can_call, now):
            log.info("%s 刚走开（最后在%s%s）：找找他", name, SEARCH_SIDE.get(side, "不知道哪边"), "，从画面边上出去的" if edge else "")

    def _search_call(self, now: float) -> None:
        """找走开的好友那一步"喊一声"：照自动喊的额度（min_gap、auto_quota、auto_again）；喊不了就告诉 search 跳过这一步。"""
        s, cfg = self.attention.search, self.cfg.call
        while self._call_times and now - self._call_times[0] > cfg.auto_window:
            self._call_times.popleft()
        why = ""
        if self._pending_auto is not None:
            why = "上一声还没结果"
        elif now - self._call_at < cfg.min_gap:
            why = "刚喊过"
        elif len(self._call_times) >= cfg.auto_quota:
            why = "这一阵喊够了"
        elif s.who and now - self._auto_found.get(s.who, float("-inf")) <= cfg.auto_again:
            why = "刚喊回来过他"
        if not why:
            self._call_times.append(now)
            if s.who in self._left_at:
                self._auto_called[s.who] = self._left_at[s.who]
            log.info("找刚走开的%s：喊一声", s.who)
            r = self.call_out("auto")
            if r.refused:
                why = r.refused
            elif r.dry:
                why = "dry-run"
            else:
                self._pending_auto, self._search_call_pending = r, True
                s.call_sent(now)
                return
        log.debug("找%s：这一声不喊（%s）", s.who, why)
        s.called(None, now)

    def _search_called(self, seen, now: float) -> None:
        """自动喊的那一声有结果了（或者等不到了）：是找的那一步喊的就交回去。"""
        s = self.attention.search
        if self._search_call_pending and s is not None:
            s.called(seen, now)
        self._search_call_pending = False
```

5. `_watch_attention` 整个换成：

```python
    def _watch_attention(self, now: float) -> None:
        """每圈最后：想看什么、在找什么（dry-run 也算，进 status）；闲着就小步转过去 / 接着找 / 喊一声。不借面板、不算"有动静"。"""
        if not self._attention_on():
            return
        a = self.attention
        a.width = self.frame_width
        if self.skills.active is not None:  # 技能（track / find）在动镜头
            self._camera_moved("turn", now)
        if self.blackout and not self._attn_black:
            a.lost_bearing(now)
        self._attn_black = self.blackout
        if self._camera_moved_at > self._attn_seen_move:
            self._attn_seen_move = self._camera_moved_at
            a.external_move(self._camera_moved_at)
        if self.cfg.attention.search:
            self._start_lost_search(now)
        who = a.search.who if a.search is not None else None
        th = a.think(self._attention_targets(now), now, self.effects().wander, self.search_obs(who, now))
        if th.event:
            self.events.put("search", th.event)
        self._attention_look_first(th, now)
        calling = th.search is not None and th.search.state == "call"
        self._attention_why = self._attention_blocked(now, check_ime=th.action is not None or calling)
        if self._attention_why:
            return
        if calling:
            self._search_call(now)
            return
        if th.action is None:
            return
        try:
            self.camera.nudge(th.action.direction, th.action.seconds, record=False)  # 原位挪到这里：不进复位账
        except Exception:
            log.exception("注意力转镜头出错")
            return
        self._notify_camera("turn")  # 只告诉感知层；不设 _camera_moved_at（那是"别人"动镜头，会挡住注意力自己）
        a.pressed(th.action, now)
        self._attention_pressed_at = now
        self._ref_thumb = None  # 自己转的，不算画面大变
        log.debug("注意力按%s %.2f s（%s）", "右" if th.action.direction == "right" else "左", th.action.seconds, a.describe(now))
```

6. `_watch_call`：`if self._pending_auto is not None:` 那两行改成

```python
            if self._pending_auto is not None or self._search_takes_call():
                return
```

7. `_collect_auto_call`：拿到结果那一支末尾（`self._pending_auto = None` 后面）加 `self._search_called(seen, now)`；等不到结果那一支末尾加 `self._search_called(None, now)`

8. `move` 里 `self._notify_camera("move")` 后面加：

```python
        self.attention.lost_bearing(self.clock())  # 走过就不知道朝哪了（有意识地找的方位记忆）
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_search_body.py tests/test_brain_events.py tests/test_brain_attention_body.py tests/test_brain_call_body.py tests/test_brain_call_again.py -q`
Expected: PASS。`test_brain_attention_body.py` 里有测试挂了，先看是不是空场景跑了 20 秒以上、环顾按了键：是的话在那个测试里加 `b.cfg.attention.scan_after = 1e9`（老测试测的是被动注意）。

然后后台跑全量：`.venv\Scripts\python.exe -m pytest -q`，Expected: 全部通过。

- [ ] **Step 6: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/events.py tests/test_brain_search_body.py tests/test_brain_events.py tests/test_brain_attention_body.py
git commit -m "feat(attention): 身体接上找——好友走开往他那边找、中间淡掉先喊、找完发 search 背景事件

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 大脑工具 `find`

**Files:**
- Create: `src/skydango/brain/find.py`
- Modify: `src/skydango/brain/body.py`（`find` 方法，放在 `track` 后面）
- Modify: `src/skydango/brain/tools.py`、`src/skydango/brain/mcp_server.py`、`src/skydango/brain/prompt.py`
- Test: `tests/test_brain_find.py`（新建）、`tests/test_brain_tools.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 3 的 `Heading` `Search` `SearchStep`；Task 5 的 `Body.search_obs` / `_left_where`；`skills.SkillStep`；`track.TrackSkill._panel_open`（静态方法）；`calling.call_available`
- Produces: `class FindSkill(name: str, seconds: float, side: str | None, edge_exit: bool)`；`Body.find(name: str, seconds: int = 30, live: bool = False) -> str`；工具 `find(name, seconds=30)`

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_brain_find.py`：

```python
"""大脑工具 find（plan 2026-10-03-attention-search Task 6）：技能层里转镜头找人，找到 / 没找到发 task_done / task_failed。"""

import pytest
from test_brain_attention_body import attn_body, steps

from skydango.brain.body import ToolError
from skydango.vision.bubbles import Rect
from skydango.vision.people import Person

XIAOMING = Person(9, "friend", "小明", Rect(1500, 400, 100, 300), "右边", "近")


def find_body(clock, **kw):
    b, dev, reader, events, env, cam = attn_body(clock, **kw)
    b.cfg.panel.idle_peek = 1e9
    b.cfg.call.enabled = False  # 不喊：只转（喊一声的路线在 search 单测里）
    return b, events, env, cam


def done_events(events):
    return [(e.kind, e.text) for e in events.drain() if e.kind in ("task_done", "task_failed")]


def test_find_refuses_stranger_names_and_dry_run(clock):
    b, events, env, cam = find_body(clock)
    with pytest.raises(ToolError, match="好友名单里没有"):
        b.find("阿白")
    dry, _, _, _ = find_body(clock, live=False)
    with pytest.raises(ToolError, match="dry-run"):
        dry.find("小明")


def test_find_target_already_in_view(clock):
    b, events, env, cam = find_body(clock)
    env.people_list = [XIAOMING]
    assert b.find("小明") == "小明就在画面里（右边·近），不用找"
    assert b.skills.active is None


def test_find_turns_full_circle_then_fails(clock):
    b, events, env, cam = find_body(clock)
    out = b.find("小明", 60)
    assert out.startswith("开始找小明了")
    steps(b, clock, 80)
    assert len(cam.nudges) == 6 * b.cfg.attention.seg_presses and len({d for d, s in cam.nudges}) == 1
    assert done_events(events) == [("task_failed", "找小明没做成：转了一圈没找到小明")]


def test_find_found_mid_way(clock):
    b, events, env, cam = find_body(clock)
    b.find("小明", 60)
    steps(b, clock, 5)
    env.people_list = [XIAOMING]
    steps(b, clock, 2)
    assert done_events(events) == [("task_done", "找小明：找到了小明（右边·近）")]
    assert b.skills.active is None
```

`tests/test_brain_tools.py`：`test_tool_names_and_actions` 里 `TOOL_NAMES` 的列表在 `"track",` 后面加 `"find",`；`ACTIONS` 的集合加 `"find"`。末尾加：

```python
def test_tools_find_passes_args():
    class B(FakeBody):
        def find(self, name, seconds=30, live=False):
            self.calls.append(("find", name, seconds))
            return f"开始找{name}了"

    body = B()
    tb = ToolBox(body)
    out, err = tb.run("find", {"name": "小明", "seconds": 20})
    assert not err and out == "开始找小明了" and tb.acted
    tb.run("find", {"name": "阿白"})
    assert body.calls == [("find", "小明", 20), ("find", "阿白", 30)]
    out, err = ToolBox(B(), FakeEyes(), max_steps=50, sandbox=True).run("find", {"name": "小明"})
    assert err and "沙盒里没有这个" in out
```

（沙盒 `ToolBox` 的写法同 `test_brain_tools.py:306`。）

`tests/test_brain_prompt.py:260` 那个测试里加一句断言：`assert "find(" in text`

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_find.py tests/test_brain_tools.py tests/test_brain_prompt.py -q`
Expected: FAIL（`Body` 没有 `find`、工具列表对不上）

- [ ] **Step 3: 写 `src/skydango/brain/find.py`**

```python
"""技能 find：大脑让找某个好友（spec 2026-10-03-attention-search §3）。

包一个 Search.find：他刚走开过就往他走的方向找，没线索就先喊一声、看名字贴在哪边，没有就转一圈；
找到（或者只是像他）就停下、task_done，没找到 task_failed。镜头不复原（交给 camera_reset），要一直盯着由大脑接着调 track。
用自己的 Heading：别和空闲注意力的方位记忆互相干扰（技能在跑时身体每圈都当"别人在动镜头"清注意力的朝向）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .calling import call_available
from .search import Heading, Search
from .skills import SkillStep
from .track import TrackSkill

log = logging.getLogger(__name__)


class FindSkill:
    name = "find"
    needs_camera = True  # SkillRunner 开始时借走聊天面板（开着时方向键转不了视角）
    quiet_people = True  # 找的时候人进出画面是自己转的：不报人来人走

    def __init__(self, name: str, seconds: float, side: str | None, edge_exit: bool) -> None:
        self.target = name
        self.seconds = float(seconds)
        self.side, self.edge_exit = side, edge_exit
        self.goal = f"找{name}"
        self.timeout = self.seconds

    def start(self, body: Any, now: float) -> None:
        cfg = body.cfg
        self.heading = Heading(cfg.attention.press_deg, cfg.track.nudge_max, cfg.spin.hfov, now)
        can_call = bool(cfg.call.enabled) and call_available(cfg, body.env)
        self.search = Search.find(self.target, self.side, self.edge_exit, can_call, cfg.attention, cfg.track, self.heading, now)
        self._pending = None  # 喊出去、还没结果的那一声（CallResult）

    def tick(self, body: Any, frame: np.ndarray, now: float) -> SkillStep:
        s = self.search
        if self._pending is not None:
            seen = body.env.call_result(self._pending.at)
            if seen is not None or now - self._pending.at > body.cfg.call.window + 10:
                s.called(seen, now)
                self._pending = None
        if TrackSkill._panel_open(body):  # 聊天面板开着方向键没用：等（不算没进展）
            s.pressed_wait(now)
            return SkillStep("running", "等聊天面板关上")
        st = s.step(body.search_obs(self.target, now), now)
        if st.state == "found":
            return SkillStep("done", f"找到了{st.note}")
        if st.state == "maybe":
            return SkillStep("done", f"可能是{st.note}")
        if st.state == "none":
            return SkillStep("failed", st.note)
        if st.state == "call":
            r = body.call_out("brain")
            if r.refused or r.dry:
                log.debug("找%s：喊不了（%s）", self.target, r.refused or "dry-run")
                s.called(None, now)
            else:
                self._pending = r
                s.call_sent(now)
            return SkillStep("running", s.describe())
        if st.state == "press":
            body.camera.nudge(st.turn.direction, st.turn.seconds)
            s.pressed(st.turn, now)
        return SkillStep("running", s.describe())

    def stop(self, body: Any, reason: str) -> None:
        """同 track：左右各补一次抬起。镜头不复原。"""
        release = getattr(body.camera, "release", None)
        if release is not None:
            release()
```

上面用到一个 `Search.pressed_wait(now)`：面板开着等的时候只刷新"有进展"的时间、不算按键。在 `src/skydango/brain/search.py` 的 `call_sent` 前面加：

```python
    def pressed_wait(self, now: float) -> None:
        """调用方在等一件和找无关的事（聊天面板关上）：别算成没进展。"""
        self._progress = now
```

并在 `tests/test_brain_search.py` 末尾加：

```python
def test_pressed_wait_keeps_search_alive():
    a, tr = cfgs()
    s = Search.lost("小明", "left", True, False, a, tr, heading(), 0.0)
    for t in range(1, 15):
        s.pressed_wait(float(t))
    assert s.step(Obs(), 15.0).state == "press"
```


- [ ] **Step 4: `Body.find`**

`src/skydango/brain/body.py`：import 区加 `from .find import FindSkill`；`track` 方法后面加：

```python
    def find(self, name: str, seconds: int = 30, live: bool = False) -> str:
        """开始技能 find：转镜头找这个好友（刚走开过就往他走的方向，没线索先喊一声再转一圈），找到 / 没找到发 task_done / task_failed。"""
        if self._dry(live):
            raise ToolError("dry-run 不转镜头找人")
        if self.camera is None:
            raise ToolError("没有视角控制，找不了人")
        if self.env is None or not hasattr(self.env, "people"):
            raise ToolError("没开感知层（[perception]），认不准人，找不了")
        if self.blackout:
            raise ToolError("画面黑着（在切场景），现在找不了")
        friends = list(self.friend_names())
        match = next((f for f in friends if f == name), None) or next((f for f in friends if similar(name, f, 0.75)), None)
        if match is None:
            raise ToolError(f"好友名单里没有 {name}")
        seconds = max(5, min(int(seconds), 60))
        now = self.clock()
        obs = self.search_obs(match, now)
        if obs is not None and obs.target_x is not None:
            return f"{match}就在画面里（{obs.target_where}），不用找"
        side, edge = None, False
        if now - self._left_at.get(match, float("-inf")) <= self.cfg.call.auto_after_leave:  # 刚走开：按找走开的好友那样找
            side, edge = self._left_where(match)
        note = self.clear_view("camera", live)
        return self.skills.start(self, FindSkill(match, seconds, side, edge)) + note
```

注意 `SkillRunner.start` 返回 `f"开始{skill.goal}了，做完或做不成会告诉你"`，goal 是"找小明"，所以是"开始找小明了，…"（测试用 `startswith`）。

- [ ] **Step 5: 工具、MCP、提示词**

`src/skydango/brain/tools.py`：

1. `DESCRIPTIONS` 里 `"track": …` 那条**后面**加：

```python
    "find": "找某个好友：身体转镜头找他（他刚走开就往他走的方向找，没线索先喊一声、再转一圈），找到就停下，"
            "用 task_done / task_failed 告诉你他在哪 / 没找到。name 是好友名字，seconds 最多找多久（5~60，默认 30）。"
            "画面里已经有他会直接告诉你；找到后要一直盯着用 track，看完想转回来用 camera_reset。",
```

2. `"attention"` 那条换成：

```python
    "attention": "改你闲着时东张西望的习惯：mode 随意（默认：看说话 / 走近的人，找刚走开的好友，一个人时隔一会儿看看周围）/ "
                 "好奇（周围看得更勤，陌生人说话也看）/ 专心（只看跟你说话、冲你来的人，不看周围）/ 别动（不转镜头）；"
                 "focus 填一个名字表示更想看他（空 = 不特别关注）。身体闲着会自己看，只在想改习惯时设，不用每轮设。",
```

3. `ACTIONS` 加 `"find"`
4. `SANDBOX_MISSING` 加 `"find": "镜头",`
5. `_bind` 里 `if name == "track":` 那段后面加：

```python
        if name == "find":
            who, seconds = _str(a, "name"), _int(a, "seconds", 30)
            return lambda: b.find(who, seconds)
```

`src/skydango/brain/mcp_server.py`：`track` 注册那段后面加（注册顺序 = DESCRIPTIONS 顺序）：

```python
    @srv.tool(name="find", description=DESCRIPTIONS["find"])
    def find(name: str, seconds: int = 30):
        return call("find", name=name, seconds=seconds)
```

`src/skydango/brain/prompt.py`：
- 「视角」一节最后一条换成：

```
- 你闲着的时候身体会自己东张西望：有人说话、朝你走过来就转过去看一眼；好友刚走开会往他走的方向找找；一个人待着时隔一会儿往没看过的方向看看。有结果会作为事件告诉你，不用你管；想换个习惯（主人说别乱转、想多看看某人）才用 attention。
```

- 「做事」一节 `track(名字, 秒)` 那条后面加一条：

```
- find(名字, 秒) 也是一件：想知道某个好友在不在附近、他走开了想看看去哪了，身体会转镜头找（必要时喊一声），找到就停下，结果用 task_done / task_failed 告诉你；要一直盯着他再用 track。
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_find.py tests/test_brain_tools.py tests/test_brain_prompt.py tests/test_brain_mcp.py tests/test_brain_call_tool.py tests/test_brain_introspect.py tests/test_brain_search.py -q`
Expected: PASS

后台跑全量：`.venv\Scripts\python.exe -m pytest -q`，Expected: 全部通过。

- [ ] **Step 7: 提交**

```bash
git add src/skydango/brain/find.py src/skydango/brain/search.py src/skydango/brain/body.py src/skydango/brain/tools.py src/skydango/brain/mcp_server.py src/skydango/brain/prompt.py tests/test_brain_find.py tests/test_brain_search.py tests/test_brain_tools.py tests/test_brain_prompt.py
git commit -m "feat(brain): find 工具——转镜头找某个好友，找到 / 没找到用 task_done / task_failed 告诉大脑

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 文档

**Files:**
- Modify: `CLAUDE.md`（「统管大脑」里"空闲注意力（东张西望）"那一条；工具清单那一句；代码结构表 `brain/` 那行）
- Modify: `docs/progress/2026-10-03-plan.md`（⑤ 一节末尾）

- [ ] **Step 1: CLAUDE.md**

1. 「统管大脑」第二条工具清单 `… / track / stop_task / …` 里 `track` 后面加 `find`
2. "空闲注意力（东张西望）"那一条整条换成：

```markdown
- **空闲注意力（东张西望）**（`[attention]`，`brain/attention.py` 纯决策 + `brain/search.py` 怎么找 + `Body._watch_attention`；spec `docs/superpowers/specs/2026-09-30-idle-attention-design.md`、
  **10-03 改成有意识地找** `docs/superpowers/specs/2026-10-03-attention-search-design.md`、计划 `docs/superpowers/plans/2026-10-03-attention-search.md`；**未在真机验证，数字都是估的**）：
  **只在 `[panel] mode = "auto"`、聊天面板关着、身体闲着时动**（always 模式下完全不动）。
  被动注意照旧：候选目标（好友 / 陌生人说话、走近、挥手鞠躬、站着的好友）按基础兴趣 ×（1 − 看腻）挑一个，`Camera.nudge` 小步拉向画面中间；冒气泡 / 有人走近时**先看一眼再开面板**（`PanelManager.hold_off`，最多 `look_first` 2 s）。
  **随意看删了，没有动机就不转**；没有被动目标时做"找"（`search.py`，分段转：一段 `seg_presses` 下、停 `dwell` 秒）：
  ① **找刚走开的好友**（身体按 `env.labels` 里他名字标签最后的位置发起）：从画面边上出去的先往那边转 `lost_segments` 段、再按 Q 看名字贴在哪边；在中间淡掉的先按 Q（**自动喊并进来了**，额度照 `[call]`；注意力接管时 `_watch_call` 不再自己判断），画面里还有没挂名字的人就不转；
  ② **环顾**：画面里 `scan_after` 秒没人、离上次 `scan_every` 秒（× 精力 `Effects.wander` × 模式），往 `Heading` 记的**最久没看过的方位**转 `scan_segments` 段，看到人就停；
  站着的好友不打断找，有人说话 / 走近打断，`resume_within` 秒没进展就不找了。结果：没找到 / 可能是他、环顾从"没人"变"有人"才发**背景事件 `search`**，其余只写 status（"在找：小明（他刚从左边走了…）""刚才往右后方看了看：附近没人"）。
  朝向估计（`press_deg` 待 `camera spin` 标定）在走路、黑屏、别人转镜头后清零。输入框开着、技能在跑、有互动请求、在回聊天、黑屏、牵手、别的面板开着、刚做完动作、dry-run 时不按（dry-run 照样算）。
  大脑工具 `attention(mode, focus)`：随意 / 好奇（环顾更勤、陌生人说话更有意思）/ 专心（只看好友说话、冲团子来的、关注的人，不环顾、照样找走开的好友）/ 别动；**不算"做了事"**。
  旧配置里的 `wander_*` 四项加载时跳过并警告（`config.DEPRECATED`）
- **`find(名字, 秒)`**（技能，`brain/find.py`）：转镜头找某个好友——刚走开过按①找，没线索先按 Q、名字贴边就往那边转、没有就转一圈；找到 / "可能是他"发 `task_done`，没找到 `task_failed`；
  镜头不复位、要一直盯着由大脑接着调 `track`；dry-run 拒绝；不在好友名单里拒绝；画面里已经有他直接回"就在画面里"
```

3. 代码结构表 `src/skydango/brain/` 那行，`attention.py` 空闲注意力 后面加 `/ \`search.py\` 有意识地找 / \`find.py\` 找人技能`

- [ ] **Step 2: 进度文档**

`docs/progress/2026-10-03-plan.md` 「⑤ 讨论」一节末尾加一行：

```markdown
**进展**：spec `docs/superpowers/specs/2026-10-03-attention-search-design.md`、计划 `docs/superpowers/plans/2026-10-03-attention-search.md`；代码已合并（离线测试通过），晚上真机按 spec「真机验证」五步看，先 `camera spin` 标定 `press_deg`。
```

（如果实施时代码还没合并，就只写 spec 和计划的路径。）

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md docs/progress/2026-10-03-plan.md
git commit -m "docs: 有意识地找写进 CLAUDE.md 和 10-03 进度

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: 合并推送**

按 CLAUDE.md「每次任务改完合并进 main 并推送」：先 `git pull`（在主目录 `D:\Lanscarlos\Develop\SkyTricolorDango` 的 main 上），把功能分支合进去，有冲突在功能分支上解决、重跑全量测试，再 `git push`。
