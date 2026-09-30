# 团子主动点亮没点火的陌生人 + 点火后鞠躬 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 没点火的黑影在团子身边站够 3 秒，身体按 3 号键举蜡烛，看 YOLO 把他从 `player_unlit` 翻成 `player` 就算点亮了，然后鞠躬（顺带放下蜡烛）；团子接受了别人点火之后也鞠躬。

**Architecture:** `vision/candle.py` 纯函数在黑影框附近找"深色圆盘 + 火焰"（只用来判断距离）；`PerceptionWatcher` 每条黑影轨迹记圆盘出现多久，够了出一个 `light` 请求，并提供 `lit()` / `mark_tried()` 给身体核实；
身体（大脑模式）用 `EmotePlayer.press_slot(3)` 举蜡烛、每圈看 `env.lit()`、到点用反射动作鞠躬；`SocialHandler` 只管规则（`allowed`），不处理 `light`。

**Tech Stack:** Python 3.13、OpenCV、numpy、pytest（合成画面 + 假设备）

**Spec:** `docs/superpowers/specs/2026-10-01-light-unlit-stranger-design.md`

## Global Constraints

- **绝不点火焰圆盘**（点了团子会一直跟着陌生人走）：整个功能里不许有 `device.tap`；点火只靠按 3 号键
- 3 号键是开关：只在"身体自己刚按 3 举起、之后 `emotes.last_any` 没更新过"时才按 3 放下，别的时候一律不按
- 点没点亮只看 YOLO：同一条轨迹 `player` 连续 `lit_frames` 帧，或原地新出现、没挂名字的 `player`；不看彩虹爱心
- 同一个陌生人（轨迹 id）不管成没成只举一次蜡烛（`mark_tried`）
- dry-run 不按任何键，只打日志；`mark_tried` 照记
- 所有新数字写进 `SocialConfig`，注释写"未在真机验证"
- 测试命令：`python -m pytest -q`（本机系统 Python 有 pytest 和 skydango 开发模式安装；没有的话用 `.venv\Scripts\python.exe -m pytest -q`）
- 临时脚本和输出放 `tmp/`（gitignore），不放系统 Temp
- 用中文写注释、日志、提交信息；提交信息末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- 只 `git add` 自己改的文件（工作区里有别人没提交的 `.claude/launch.json`、`emotes/`、`docs/superpowers/plans/2026-09-27-brain-move-owner.md`，别碰）

## 文件结构

| 文件 | 改动 |
|---|---|
| `assets/candle/flame.png` | 新：圆盘里的火焰模板（从录像 d 截，只有火焰、没有白圈） |
| `src/skydango/vision/candle.py` | 新：`Disk`、`load_flame`、`dark_ring`、`find_disk` |
| `src/skydango/config.py` | `SocialConfig` 加 9 个键，`accept_strangers` 默认加 `light` |
| `src/skydango/game/social.py` | `KIND_NAMES["light"]`、`Request.track`、`LIGHT` 常量、`handle` 跳过 `light` |
| `src/skydango/vision/perception.py` | 黑影身上的圆盘计时 → `light` 请求；`mark_tried`、`lit`；孤儿圆圈外环暗的不算 `candle` |
| `src/skydango/cli.py` | `_scene_watcher` 把 `cfg.social` 和火焰模板交给感知层 |
| `src/skydango/game/wheel.py`、`game/emotes.py` | `Wheel.press(slot)`、`EmotePlayer.press_slot(slot)` |
| `src/skydango/brain/body.py` | `_watch_light`、`_watch_bow`、`_schedule_bow`、`_lower_candle`；`light` 不进请求事件；`set_policy` 放行陌生人 `light`；status；退出时放下蜡烛 |
| `src/skydango/brain/tools.py`、`brain/prompt.py` | kind 说明加 `light` |
| `tests/test_candle.py` | 新 |
| `tests/test_perception.py`、`tests/test_social.py`、`tests/test_wheel.py`、`tests/test_emotes.py`、`tests/test_brain_light_body.py`（新） | 测试 |
| `docs/game-ops.md`、`CLAUDE.md`、`docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md` | 文档 |

---

### Task 1: 火焰模板 + `vision/candle.py`（找圆盘）

**Files:**
- Create: `assets/candle/flame.png`
- Create: `src/skydango/vision/candle.py`
- Test: `tests/test_candle.py`

**Interfaces:**
- Produces:
  - `FLAME = "assets/candle/flame.png"`
  - `@dataclass(frozen=True) class Disk: x: int; y: int; r: float; score: float`（圆心整图坐标、火焰半高、匹配分）
  - `load_flame(path: str | Path = FLAME) -> np.ndarray`（米白剪影，已 trim）
  - `dark_ring(frame, cx: int, cy: int, r_in: float, r_out: float, dark: float) -> bool`
  - `find_disk(frame, box: Rect, flame: np.ndarray, min_score: float = 0.7, dark: float = 80.0) -> Disk | None`

- [ ] **Step 1: 截火焰模板**

录像 d 第 16 秒的圆盘火焰中心约 (1030, 680)，火焰半高约 27 px。在仓库目录跑：

```bash
python - <<'EOF'
import cv2
from pathlib import Path
from skydango.game.social import cream
from skydango.vision.icons import trim
im = cv2.imread('tmp/record/candle-20260930-d/0160_016.00s.jpg')
crop = im[648:712, 998:1062]
Path('assets/candle').mkdir(parents=True, exist_ok=True)
cv2.imwrite('assets/candle/flame.png', crop)
m = trim(cream(crop))
print(m.shape)
cv2.imwrite('tmp/flame_mask.png', cv2.resize(m, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
EOF
```

Expected：打印的形状两边都在 35~60 之间。用 Read 打开 `tmp/flame_mask.png` 看：只有一个火焰（带左上的小水滴、中间的洞），**没有圆弧、头发、别的亮块**。有多余的东西就把裁剪框往火焰收紧再跑，直到干净。

- [ ] **Step 2: 写失败的测试**

`tests/test_candle.py`：

```python
"""火焰圆盘（spec 2026-10-01-light-unlit-stranger §3）：黑影身上深色实心圆 + 火焰、没有白圈。"""

import cv2
import numpy as np

from skydango.game.social import cream, load_icons
from skydango.vision.bubbles import Rect
from skydango.vision.candle import dark_ring, find_disk, load_flame

FLAME = load_flame()
STRANGER = load_icons("assets/social")["stranger"]


def paste(frame, mask, cx, cy):
    """把米白剪影画到 (cx, cy) 为中心的位置。"""
    h, w = mask.shape
    y1, x1 = cy - h // 2, cx - w // 2
    region = frame[y1 : y1 + h, x1 : x1 + w]
    region[mask > 0] = (235, 245, 245)


def figure(ring=False, icon=None):
    """黑影（框 900,500,180,360）胸口一个深色圆盘 + 图标；ring = 外面再画一圈白色描边（举蜡烛的请求）。"""
    f = np.full((1080, 1920, 3), (60, 90, 40), np.uint8)  # 草地
    cv2.rectangle(f, (900, 500), (1080, 860), (12, 12, 12), -1)
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)
    if ring:
        cv2.circle(f, (990, 620), 38, (240, 245, 245), 3)
    paste(f, FLAME if icon is None else icon, 990, 620)
    return f


BOX = Rect(900, 500, 180, 360)


def test_finds_dark_disk_with_flame():
    d = find_disk(figure(), BOX, FLAME)
    assert d is not None and abs(d.x - 990) <= 6 and abs(d.y - 620) <= 6 and d.score >= 0.9


def test_white_ring_candle_is_not_a_disk():
    assert find_disk(figure(ring=True), BOX, FLAME) is None


def test_stranger_icon_is_not_a_disk():
    assert find_disk(figure(icon=STRANGER), BOX, FLAME) is None


def test_nothing_on_plain_figure():
    f = figure()
    cv2.circle(f, (990, 620), 52, (20, 20, 20), -1)  # 盖掉火焰
    assert find_disk(f, BOX, FLAME) is None


def test_disk_near_head_above_box_is_found():
    """录像 d 第 20 秒起圆盘在头顶附近（框上沿往上）。"""
    f = figure()
    cv2.circle(f, (990, 620), 52, (60, 90, 40), -1)  # 抹掉胸口那个
    cv2.circle(f, (990, 470), 52, (20, 20, 20), -1)
    paste(f, FLAME, 990, 470)
    d = find_disk(f, BOX, FLAME)
    assert d is not None and abs(d.y - 470) <= 6


def test_dark_ring():
    f = figure()
    assert dark_ring(f, 990, 620, 32, 48, dark=80.0)
    assert not dark_ring(figure(ring=True), 990, 620, 32, 48, dark=80.0)
```

- [ ] **Step 3: 跑测试看它失败**

Run: `python -m pytest tests/test_candle.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'skydango.vision.candle'`

- [ ] **Step 4: 实现**

`src/skydango/vision/candle.py`：

```python
"""火焰圆盘：没点火的黑影站到团子身边时，他身上出现深色实心圆 + 白色空心火焰（没有白圈）。

只用来判断"这个黑影站在能点火的距离里"（spec 2026-10-01-light-unlit-stranger）。**绝不点它**：
点了团子会一直跟着那个陌生人走，点火靠按 3 号键举蜡烛。
白圈 + 火焰是陌生人举蜡烛要给团子点火（social 的 candle），外环是亮的，这里不算。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..game.social import cream
from ..imageio import imread
from .bubbles import Rect
from .icons import best_match, trim

FLAME = "assets/candle/flame.png"  # 录像 candle-20260930-d 第 16 秒截的，只有火焰
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]  # 人远近不同，圆盘大小跟着变
RING_IN, RING_OUT = 1.2, 1.8  # 火焰半高的这么多倍之间是圆盘的外环（白圈在约 1.35 倍）
BRIGHT_MAX = 0.05  # 外环里米白像素超过这个比例 = 有白圈


@dataclass(frozen=True)
class Disk:
    x: int  # 圆心（整图坐标）
    y: int
    r: float  # 火焰半高（像素）
    score: float


def load_flame(path: str | Path = FLAME) -> np.ndarray:
    return trim(cream(imread(path)))


def dark_ring(frame: np.ndarray, cx: int, cy: int, r_in: float, r_out: float, dark: float) -> bool:
    """(cx, cy) 周围 r_in~r_out 的环带够暗、没有白色描边。"""
    x1, y1 = max(0, int(cx - r_out)), max(0, int(cy - r_out))
    patch = frame[y1 : int(cy + r_out) + 1, x1 : int(cx + r_out) + 1]
    if patch.size == 0:
        return False
    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    d = np.hypot(xx - (cx - x1), yy - (cy - y1))
    ring = (d >= r_in) & (d <= r_out)
    if not ring.any():
        return False
    value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    bright = cream(patch) > 0
    return float(value[ring].mean()) < dark and float(bright[ring].mean()) < BRIGHT_MAX


def _region(box: Rect, width: int, height: int) -> tuple[int, int, int, int] | None:
    """黑影框左右各扩 0.5 倍框宽、往上扩 0.6 倍框高、下到框底（圆盘在胸口或头顶附近）。"""
    x1, x2 = max(0, round(box.x - 0.5 * box.w)), min(width, round(box.x2 + 0.5 * box.w))
    y1, y2 = max(0, round(box.y - 0.6 * box.h)), min(height, box.y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def find_disk(frame: np.ndarray, box: Rect, flame: np.ndarray, min_score: float = 0.7, dark: float = 80.0) -> Disk | None:
    """在这个黑影身上 / 头顶找火焰圆盘；没有（或者是白圈的举蜡烛请求）返回 None。"""
    area = _region(box, frame.shape[1], frame.shape[0])
    if area is None:
        return None
    x1, y1, x2, y2 = area
    mask = cream(frame[y1:y2, x1:x2])
    best, scale = None, 1.0
    for s in SCALES:
        m = best_match(mask, flame, [s])
        if best is None or m.score > best.score:
            best, scale = m, s
    if best is None or best.score < min_score:
        return None
    cx, cy, r = x1 + best.x, y1 + best.y, flame.shape[0] * scale / 2
    if not dark_ring(frame, cx, cy, RING_IN * r, RING_OUT * r, dark):
        return None
    return Disk(cx, cy, r, best.score)
```

- [ ] **Step 5: 跑测试看它通过**

Run: `python -m pytest tests/test_candle.py -q`
Expected: 6 passed。`test_stranger_icon_is_not_a_disk` 过不了时，先看 `find_disk` 在它上面的分数（打印 `best.score`），把 `min_score` 默认值调到介于两者之间，并在 Step 6 的录像上复核。

- [ ] **Step 6: 录像上量一遍（离线，不进 git）**

写 `tmp/eval_disk.py`：对 `tmp/record/candle-20260930-d/` 第 140~260 帧、`candle-20260930-c/` 第 0~60 帧，用 `models/sky-yolo-v7.pt` 跑 `skydango.vision.detect.make_detector(...)`（参数照 `cli._scene_watcher`，读 `config.toml` 的 `[perception]`），
对每个 `player_unlit` 框调 `find_disk`，把结果（框 + 圆盘圈 + 分数）画在图上存 `tmp/eval_disk/<帧名>.jpg`，打印每帧 `帧名 框数 圆盘数 分数`。
人工看：录像 d 第 15~17 秒、第 20~25 秒那个披风黑影要认出；左下角"蜡烛 + 两只手"那个人不能认出。
认出率明显偏低（< 60%）时调 `SCALES` / `min_score` / `RING_IN` / `RING_OUT` / `dark`，把最终数字写回 `candle.py` 和 Task 2 的配置默认值，并在提交信息里写上认出率。

- [ ] **Step 7: 提交**

```bash
git add assets/candle/flame.png src/skydango/vision/candle.py tests/test_candle.py
git commit -m "feat(candle): 认黑影身上的火焰圆盘（深色实心圆 + 火焰、没有白圈）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 配置 + 社交层的 `light` 类型

**Files:**
- Modify: `src/skydango/config.py`（`SocialConfig`，约第 235~250 行）
- Modify: `src/skydango/game/social.py`
- Test: `tests/test_social.py`

**Interfaces:**
- Produces:
  - `SocialConfig` 新键：`light_after: float = 3.0`、`light_timeout: float = 8.0`、`lit_frames: int = 3`、`after_light: str = "鞠躬"`、`bow_delay: float = 2.5`、`candle_slot: int = 3`、`disk_min_score: float = 0.7`、`disk_dark: float = 80.0`、`flame: str = "assets/candle/flame.png"`；`accept_strangers` 默认 `["candle", "light"]`
  - `social.LIGHT = "light"`、`social.LIGHT_KEY = "陌生人·点亮"`（`requests` 里这条请求的键；感知层写、身体读，放在 social 免得 body 去 import perception）；`KIND_NAMES["light"] = "点亮陌生人"`
  - `Request(name, kind, pos, seen_at, track: int | None = None)`
  - `SocialHandler.handle` 跳过 `kind == LIGHT` 的请求（身体自己处理）

- [ ] **Step 1: 写失败的测试**（加到 `tests/test_social.py` 末尾；`handler(...)` 是文件里现成的辅助函数，第 130 行附近，照它的用法）

```python
def test_light_is_not_handled_by_social_handler():
    """light（团子举蜡烛点亮陌生人）由身体按 3 号键做，SocialHandler 不点屏幕。"""
    from skydango.game.social import LIGHT, Request

    dev = FakeDevice([np.zeros((1080, 1920, 3), np.uint8)])
    h = SocialHandler(dev, SocialConfig(), IconClassifier(ICONS), lambda: [])
    req = Request("陌生人", LIGHT, (990, 620), 0.0, track=7)
    assert h.allowed(req)  # 默认 accept_strangers 里有 light
    assert h.handle({"陌生人·点亮": req}, 0.1) == []
    assert not [c for c in dev.calls if c[0] == "tap"]


def test_light_policy_can_turn_it_off():
    from skydango.game.social import LIGHT, Request

    h = SocialHandler(FakeDevice([np.zeros((10, 10, 3), np.uint8)]), SocialConfig(), IconClassifier(ICONS), lambda: [])
    h.set_policy("stranger", LIGHT, False)
    assert not h.allowed(Request("陌生人", LIGHT, (0, 0), 0.0, track=1))
```

先确认 `tests/test_social.py` 顶部已经 import 了 `FakeDevice`、`np`、`SocialHandler`、`SocialConfig`、`IconClassifier`；缺哪个补哪个（`from conftest import FakeDevice`）。

- [ ] **Step 2: 跑测试看它失败**

Run: `python -m pytest tests/test_social.py -q -k light`
Expected: FAIL，`ImportError: cannot import name 'LIGHT'`

- [ ] **Step 3: 实现**

`config.py` 的 `SocialConfig`，`accept_strangers` 那行改成并在 `error_backoff` 后面加：

```python
    accept_strangers: list[str] = field(default_factory=lambda: ["candle", "light"])  # 陌生人：接受点火；light = 团子举蜡烛点亮没点火的陌生人（2026-10-01）
```

```python
    # 点亮没点火的陌生人（spec 2026-10-01-light-unlit-stranger，都**未在真机验证**）
    light_after: float = 3.0  # 黑影身上的火焰圆盘连续看到这么久才举蜡烛（别点路过的）
    light_timeout: float = 8.0  # 举着蜡烛最多等这么久看他亮起来（YOLO player_unlit → player）
    lit_frames: int = 3  # YOLO 连续几帧认成 player 才算点亮
    after_light: str = "鞠躬"  # 点亮别人 / 接受别人点火后做的动作（轮盘上要有；鞠躬顺带放下蜡烛），空 = 不做
    bow_delay: float = 2.5  # 看到他亮起来 / 接受点火后等这么久再鞠躬（等闪光动画）
    candle_slot: int = 3  # 轮盘上"举蜡烛"在第几格：按一下举起、再按一下放下（用户 2026-10-01）
    disk_min_score: float = 0.7  # 火焰圆盘的模板匹配分
    disk_dark: float = 80.0  # 圆盘外环亮度均值低于这个才算深色圆盘（有白圈的是举蜡烛的请求）
    flame: str = "assets/candle/flame.png"  # 圆盘里的火焰模板
```

（Task 1 Step 6 调过数字的话，用调过的默认值。）

`game/social.py`：

```python
KIND_NAMES = {
    "hand": "牵手", "hug": "拥抱", "highfive": "击掌", "piggyback": "背背",
    "candle": "点火",  # 火焰：陌生人举着蜡烛走到团子旁边，要给团子点火（2026-09-28 用户说明）
    "light": "点亮陌生人",  # 团子举蜡烛给身边没点火的黑影点火（2026-10-01，身体按 3 号键做，见 brain/body.py）
    "stranger": "陌生人", "eye": "在看留影 / 听音乐", "shared": "共享空间",
}
LIGHT = "light"
LIGHT_KEY = "陌生人·点亮"  # 感知层 requests 里"团子该举蜡烛点亮身边这个黑影"的键（和陌生人举蜡烛的 "陌生人" 分开，两件事可能同时有）
```

`Request` 加字段（放最后，有默认值）：

```python
@dataclass(frozen=True)
class Request:
    name: str  # 发起人的游戏昵称
    kind: str  # hand / hug / highfive / candle / light
    pos: tuple[int, int]  # 圆圈中心（整张截图坐标）；light 是火焰圆盘的中心
    seen_at: float
    track: int | None = None  # light：那个黑影的轨迹 id（感知层）
```

`SocialHandler.handle` 的循环开头：

```python
        for req in list(requests.values()):
            if req.kind == LIGHT:  # 举蜡烛点亮陌生人：身体按 3 号键做，这里不点屏幕（点圆盘会跟着人走）
                continue
```

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_social.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py src/skydango/game/social.py tests/test_social.py
git commit -m "feat(social): 新请求类型 light（团子举蜡烛点亮陌生人），SocialHandler 不处理它

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 感知层——圆盘计时、`light` 请求、`mark_tried`、`lit`

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Modify: `src/skydango/cli.py`（`_scene_watcher`，约第 329 行的 `return PerceptionWatcher(`）
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 1 `find_disk`、`dark_ring`、`load_flame`；Task 2 `SocialConfig`、`LIGHT`、`Request.track`
- Produces:
  - `requests[LIGHT_KEY]`（`LIGHT_KEY` 从 `game.social` 来）是 `Request(STRANGER, "light", 圆心, now, track=id)`，一次最多一个
  - `PerceptionWatcher(..., social_cfg: SocialConfig | None = None, flame: np.ndarray | None = None)`：两个都有才找圆盘
  - `PerceptionWatcher.mark_tried(track_id: int) -> None`
  - `PerceptionWatcher.lit(track_id: int, pos: tuple[int, int], since: float) -> bool | None`：True = 点亮了；False = 还是黑的；None = 这条轨迹没了、原地也没冒出新的亮人

- [ ] **Step 1: 写失败的测试**（加到 `tests/test_perception.py`；`watcher`、`FakeDetector`、`Clock`、`frame` 是文件里现成的）

```python
# ---- 点亮陌生人（spec 2026-10-01-light-unlit-stranger §3） ----
from skydango.config import SocialConfig
from skydango.vision import perception as perception_mod
from skydango.vision.candle import Disk
from skydango.game.social import LIGHT_KEY


def unlit(x, y=400, w=90, h=220):
    return Detection("player_unlit", Rect(x, y, w, h), 0.9)


def light_watcher(monkeypatch, disks):
    """disks：每次 find_disk 依次返回什么（用完了一直返回最后一个）。"""
    seq = list(disks)
    monkeypatch.setattr(perception_mod, "find_disk", lambda frame, box, flame, s, d: seq.pop(0) if len(seq) > 1 else seq[0])
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    w.light_cfg, w.flame = SocialConfig(), np.ones((4, 4), np.uint8)
    return w, det, clock


DISK = Disk(1045, 480, 20.0, 0.9)


def run(w, t, clock):
    clock.t = t
    w.process(frame(), t, panel_visible=False)


def test_light_request_after_disk_seen_long_enough(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK])
    det.frames = [[unlit(1000)]]
    for t in (0.0, 1.0, 2.0):
        run(w, t, clock)
        assert LIGHT_KEY not in w.requests
    run(w, 3.1, clock)
    req = w.requests[LIGHT_KEY]
    assert req.kind == "light" and req.pos == (1045, 480) and req.track is not None


def test_short_gap_keeps_timer_long_gap_resets(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK, DISK, None, DISK, DISK])
    det.frames = [[unlit(1000)]]
    for t in (0.0, 1.0, 1.5, 2.0, 3.1):  # 1.5 s 那次没看到，但离上次 0.5 s
        run(w, t, clock)
    assert LIGHT_KEY in w.requests
    w2, det2, clock2 = light_watcher(monkeypatch, [DISK, None, None, DISK, DISK])
    det2.frames = [[unlit(1000)]]
    for t in (0.0, 0.5, 1.2, 2.5, 3.1):  # 0 s 之后 2.5 s 才又看到：重新计时
        run(w2, t, clock2)
    assert LIGHT_KEY not in w2.requests


def test_mark_tried_stops_requests(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK])
    det.frames = [[unlit(1000)]]
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    assert LIGHT_KEY not in w.requests
    run(w, 4.0, clock)
    assert LIGHT_KEY not in w.requests


def test_far_or_lit_people_get_no_disk_check(monkeypatch):
    calls = []
    monkeypatch.setattr(perception_mod, "find_disk", lambda *a: calls.append(a) or DISK)
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    w.light_cfg, w.flame = SocialConfig(), np.ones((4, 4), np.uint8)
    det.frames = [[unlit(1000, h=40), player(300)]]  # 远处的小黑影（< stranger_min_height）、亮着的人
    run(w, 0.0, clock)
    assert calls == []


def test_lit_when_same_track_turns_player(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK])
    det.frames = [[unlit(1000)]]
    run(w, 0.0, clock)
    tid = w.last_tracks[0].id
    assert w.lit(tid, (1045, 480), 0.0) is False
    det.frames = [[player(1000)]]  # 同一个位置翻成 player：cross 组保证 id 不变
    run(w, 0.5, clock)
    run(w, 0.6, clock)
    assert w.lit(tid, (1045, 480), 0.0) is False  # 才 2 帧
    run(w, 0.7, clock)
    assert w.lit(tid, (1045, 480), 0.0) is True


def test_lit_fallback_when_track_breaks(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK])
    det.frames = [[unlit(1000)]]
    run(w, 0.0, clock)
    tid = w.last_tracks[0].id
    det.frames = [[]]
    run(w, 1.5, clock)  # 闪光：一帧都没认出来，轨迹过了 track_buffer 被删
    assert w.lit(tid, (1045, 480), 0.0) is None
    det.frames = [[player(1010)]]  # 原地冒出一个亮着的人，没有名字标签
    for t in (1.6, 1.7, 1.8):
        run(w, t, clock)
    assert w.lit(tid, (1045, 480), 0.0) is True


def test_lit_fallback_ignores_friend(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [DISK])
    det.frames = [[unlit(1000)]]
    run(w, 0.0, clock)
    tid = w.last_tracks[0].id
    det.frames = [[]]
    run(w, 1.5, clock)
    w.ocr = FakeOcr({110: "懒洋洋大王"})
    det.frames = [[player(1010), tag(1000, 110)]]  # 冒出来的是挂着好友名字的
    for t in (1.6, 1.7, 1.8):
        run(w, t, clock)
    assert w.lit(tid, (1045, 480), 0.0) is None
```

`tag`、`player`、`FakeOcr` 也是文件里现成的。`test_lit_fallback_when_track_breaks` 依赖 `PerceptionConfig.track_buffer` 默认 1.0 s（1.5 s 时原轨迹已删）。

- [ ] **Step 2: 跑测试看它失败**

Run: `python -m pytest tests/test_perception.py -q -k "light or lit or disk"`
Expected: FAIL，`ImportError: cannot import name 'LIGHT_KEY'`

- [ ] **Step 3: 实现**

`perception.py` 顶部 import 和常量：

```python
from ..config import EnvConfig, GestureConfig, PerceptionConfig, SocialConfig, SpinConfig
from ..game.social import IDLE, KIND_NAMES, LIGHT, LIGHT_KEY, Request, is_request
from .candle import dark_ring, find_disk
```

```python
DISK_EVERY = 0.3  # 同一个黑影最多隔这么久找一次火焰圆盘
DISK_GAP = 1.0  # 圆盘断开不超过这么久不重新计时（火焰会晃）
```

`__init__` 参数加在 `gesture_cfg` 后面：

```python
        social_cfg: SocialConfig | None = None,  # 点亮陌生人（spec 2026-10-01）：和 flame 都有才找火焰圆盘
        flame: np.ndarray | None = None,
```

`__init__` 里：

```python
        self.light_cfg = social_cfg
        self.flame = flame
        self._tried: set[int] = set()  # 身体举过蜡烛的黑影轨迹（不管成没成都不再出请求）
```

`process` 里，`strangers = unlit = 0` 那个 for 循环结束后、`self._watch_typing(...)` 之前加一行：

```python
        self._watch_disks(frame, players, now, height)
```

孤儿圆圈那段改成（外环暗的是火焰圆盘，不是举蜡烛的请求）：

```python
        for ring in orphans:
            kind = ring.data["kind"] = self._classify(frame, ring)
            if kind == "candle" and self._disk_like(frame, ring):
                kind = ring.data["kind"] = None  # 深色圆盘里的火焰：团子能去点亮他，不是他要给团子点火
            if is_request(kind):
```

新方法（放在 `_ring_center` 前面）：

```python
    # ---- 点亮陌生人（spec 2026-10-01-light-unlit-stranger §3） ----
    def _disk_like(self, frame: np.ndarray, ring: Track) -> bool:
        if self.light_cfg is None:
            return False
        cx, cy = self._ring_center(ring)
        r = ring.box.w / 2
        return dark_ring(frame, cx, cy, 0.8 * r, 1.0 * r, self.light_cfg.disk_dark)

    def _watch_disks(self, frame: np.ndarray, players: list[Track], now: float, height: int) -> None:
        """每条黑影轨迹记火焰圆盘出现了多久；够 light_after 秒出一个 light 请求（一次一个，挑看到最久的）。
        顺带给每条轨迹记"连续几帧是 player"，lit() 用。"""
        for p in players:
            p.data["player_run"] = p.data.get("player_run", 0) + 1 if p.cls == "player" else 0
        live = set(self.tracker.tracks)
        with self._lock:
            self._tried &= live
            tried = set(self._tried)
        cfg = self.light_cfg
        if cfg is None or self.flame is None:
            return
        best: Track | None = None
        for p in players:
            if p.cls != UNLIT or p.id in tried or p.box.h < self.cfg.stranger_min_height * height:
                continue
            d = p.data
            if now - d.get("disk_check", float("-inf")) >= DISK_EVERY:
                d["disk_check"] = now
                disk = find_disk(frame, p.box, self.flame, cfg.disk_min_score, cfg.disk_dark)
                if disk is not None:
                    if now - d.get("disk_last", float("-inf")) > DISK_GAP:
                        d["disk_first"] = now
                    d["disk_last"], d["disk_pos"] = now, (disk.x, disk.y)
            if now - d.get("disk_last", float("-inf")) <= DISK_GAP and now - d["disk_first"] >= cfg.light_after:
                if best is None or d["disk_first"] < best.data["disk_first"]:
                    best = p
        if best is None:
            self.requests.pop(LIGHT_KEY, None)
            return
        if LIGHT_KEY not in self.requests:
            log.info("没点火的陌生人在身边站了 %.0f 秒（轨迹 %d）", now - best.data["disk_first"], best.id)
        self.requests[LIGHT_KEY] = Request(STRANGER, LIGHT, best.data["disk_pos"], now, track=best.id)

    def mark_tried(self, track_id: int) -> None:
        """身体给这个黑影举过蜡烛了（不管成没成）：不再出请求。"""
        with self._lock:
            self._tried.add(track_id)
        self.requests.pop(LIGHT_KEY, None)

    def lit(self, track_id: int, pos: tuple[int, int], since: float) -> bool | None:
        """举蜡烛之后他亮起来没有（全靠 YOLO：player_unlit → player）。

        True：同一条轨迹连续 lit_frames 帧是 player；或者轨迹断了（闪光时认不出），since 之后在 pos 附近冒出一条
        没挂名字标签的 player、也连续够帧。False：还是黑的。None：人没了，原地也没冒出亮人。"""
        need = self.light_cfg.lit_frames if self.light_cfg is not None else 3
        tracks = list(self.tracker.tracks.values())
        for t in tracks:
            if t.id == track_id:
                return t.cls == "player" and t.data.get("player_run", 0) >= need
        for t in tracks:
            if (t.cls == "player" and t.first >= since and not t.data.get("name") and not t.data.get("tagged")
                    and abs(t.box.x + t.box.w / 2 - pos[0]) < t.box.w and t.data.get("player_run", 0) >= need):
                return True
        return None
```

注意：`players` 已经去掉了团子（`_is_self`），`player_run` 只给人物轨迹记。`lit()` 在身体线程调，只读，`list(...)` 先拷一份。

`cli.py` 的 `_scene_watcher`，`return PerceptionWatcher(` 前加：

```python
    flame = None
    if cfg.social.enabled and "light" in cfg.social.accept_strangers:
        from .vision.candle import load_flame

        flame = load_flame(cfg.social.flame)
```

`PerceptionWatcher(...)` 的参数末尾加 `social_cfg=cfg.social, flame=flame,`。

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_perception.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/perception.py src/skydango/cli.py tests/test_perception.py
git commit -m "feat(perception): 黑影身上的火焰圆盘够 3 秒出 light 请求；lit() 按 YOLO 判断点亮没有

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 按轮盘某一格（`Wheel.press` / `EmotePlayer.press_slot`）

**Files:**
- Modify: `src/skydango/game/wheel.py`（`perform`，约第 261 行）
- Modify: `src/skydango/game/emotes.py`
- Test: `tests/test_wheel.py`、`tests/test_emotes.py`

**Interfaces:**
- Produces:
  - `Wheel.press(slot: int) -> None`：输入框开着抛 `WheelError`；按 `cfg.slot_keys[slot - 1]`；更新 `last_used[slot]`
  - `EmotePlayer.press_slot(slot: int) -> None`：先关输入框，再 `wheel.press`；**不更新** `last_any` / `last_emote`（举蜡烛不算动作）

- [ ] **Step 1: 写失败的测试**

先看 `tests/test_wheel.py`、`tests/test_emotes.py` 里怎么建 `Wheel` / `EmotePlayer`（有现成的辅助函数），照着写：

```python
# tests/test_wheel.py（make_wheel 是文件里现成的：返回 wheel, device, t）
def test_press_slot_sends_its_digit_key(library):
    wheel, device, _ = make_wheel(library)
    wheel.press(3)  # 3 号格是锁定格（举蜡烛），也能按
    assert device.calls == [("hw_key", 4)]  # KEY_3


def test_press_refuses_when_input_box_open(library):
    wheel, device, _ = make_wheel(library, shown=True)
    with pytest.raises(WheelError, match="输入框"):
        wheel.press(3)
```

```python
# tests/test_emotes.py（make_player 是文件里现成的：返回 player, device, t, state）
def test_press_slot_is_not_an_emote(library):
    player, device, _, _ = make_player(library, shown=True)
    player.press_slot(3)
    assert device.calls[-1] == ("hw_key", 4)  # 先 BACK 关输入框，再按 KEY_3
    assert player.last_any == float("-inf") and player.last_emote == float("-inf")
```

- [ ] **Step 2: 跑测试看它失败**

Run: `python -m pytest tests/test_wheel.py tests/test_emotes.py -q -k press`
Expected: FAIL，`AttributeError: 'Wheel' object has no attribute 'press'`

- [ ] **Step 3: 实现**

`wheel.py`：

```python
    def press(self, slot: int) -> None:
        """按这一格的数字键（锁定格也能按：3 号格常驻举蜡烛）。"""
        if self.device.ime_shown():
            raise WheelError("输入框开着，按数字键会变成打字")
        self.device.hw_key(self.cfg.slot_keys[slot - 1])
        self.last_used[slot] = self.clock()

    def perform(self, name: str) -> int:
        slot = self.ensure(name)
        self.press(slot)
        log.info("做动作「%s」（格子 %d）", name, slot)
        return slot
```

`emotes.py`（放在 `pretend` 后面）：

```python
    def press_slot(self, slot: int) -> None:
        """按轮盘某一格的数字键，不算动作（不占冷却、不更新 last_any）：3 号格举蜡烛 / 放下蜡烛（spec 2026-10-01-light-unlit-stranger）。"""
        self._close_input()
        self.wheel.press(slot)
```

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_wheel.py tests/test_emotes.py -q`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add src/skydango/game/wheel.py src/skydango/game/emotes.py tests/test_wheel.py tests/test_emotes.py
git commit -m "feat(wheel): 按轮盘某一格的数字键（举蜡烛用），不算动作

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 身体——举蜡烛、核实、鞠躬、放下蜡烛

**Files:**
- Modify: `src/skydango/brain/body.py`
- Modify: `src/skydango/brain/tools.py`（第 31~35 行 `set_request_policy` 说明）、`src/skydango/brain/prompt.py`（「互动请求」一节，第 61 行起）
- Test: `tests/test_brain_light_body.py`（新）

**Interfaces:**
- Consumes: Task 2 `LIGHT`、`LIGHT_KEY`、`SocialConfig` 新键；Task 3 `env.mark_tried`、`env.lit`；Task 4 `emotes.press_slot`
- Produces（身体内部）：
  - `Body._raised: tuple[int, tuple[int, int], float] | None`（轨迹 id、圆盘位置、举起时间）
  - `Body._bow: tuple[float, float, float | None] | None`（到点时间、放弃时间、要兜底放下的蜡烛是几时举的）
  - `Body._lit_at: float`（最近一次点亮成功的时间）
  - `Body._watch_light(now)`、`Body._watch_bow(now)`、`Body._schedule_bow(now, raised_at)`、`Body._lower_candle(raised_at)`
  - `REQUEST_KINDS` 加 `"light"`

- [ ] **Step 1: 写失败的测试**

`tests/test_brain_light_body.py`：

```python
"""点亮没点火的陌生人 + 点火后鞠躬（spec 2026-10-01-light-unlit-stranger §4~§6）。"""

from test_brain_body import FakeEnv, FakeSocial, body
from test_brain_reflex_body import ReflexEmotes

from skydango.game.social import LIGHT, LIGHT_KEY, Request

CANDLE_KEY = ("hw_key", 4)  # 3 号格 = slot_keys[2] = 4


class LightEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.tried = []
        self.lit_result = False

    def mark_tried(self, track_id):
        self.tried.append(track_id)
        self.requests.pop(LIGHT_KEY, None)

    def lit(self, track_id, pos, since):
        return self.lit_result


class LightEmotes(ReflexEmotes):
    def on_wheel(self):
        return ["鞠躬", "挥手"]

    def press_slot(self, slot):
        self.dev.calls.append(("hw_key", [2, 3, 4, 5, 6, 7, 8, 9][slot - 1]))


class AllowAll(FakeSocial):
    def allowed(self, req):
        return True


def lb(clock, live=True, social=None):
    from conftest import FakeDevice, scene

    dev = FakeDevice([scene()])
    emotes = LightEmotes(dev)
    emotes.clock = clock
    env = LightEnv()
    b, dev, reader, events = body(clock, live=live, env=env, emotes=emotes, social=social or AllowAll(), device_override=dev)
    b.step()
    events.drain()
    return b, dev, env, emotes, events


def offer(env, clock, track=7):
    env.requests[LIGHT_KEY] = Request("陌生人", LIGHT, (990, 620), clock(), track=track)


def presses(dev):
    return dev.calls.count(CANDLE_KEY)


def test_raise_candle_once_then_bow_when_lit(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert presses(dev) == 1 and env.tried == [7]
    b.step()
    assert presses(dev) == 1  # 请求被 mark_tried 拿掉了，不会再按
    env.lit_result = True
    b.step()
    assert [e.kind for e in events.drain()] == ["accepted"]
    clock.advance(2.0)
    b.step()
    assert emotes.done == []  # 还没到 bow_delay
    clock.advance(0.6)
    b.step()
    assert emotes.done == [("鞠躬", True)] and presses(dev) == 1  # 鞠躬放下蜡烛，不按 3


def test_timeout_lowers_candle_without_bow(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(8.1)
    b.step()
    assert presses(dev) == 2 and emotes.done == []  # 举一次、放一次


def test_no_lower_if_emoted_meanwhile(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(1.0)
    emotes.perform("挥手")  # 大脑做了个动作：蜡烛已经放下了
    clock.advance(7.1)
    b.step()
    assert presses(dev) == 1


def test_bow_gives_up_and_lowers(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()
    b._reflex_emote = lambda *a: False  # 一直做不了反射动作（被挡着、刚做过……）
    clock.advance(2.5)
    b.step()
    clock.advance(5.1)
    b.step()
    assert emotes.done == [] and presses(dev) == 2


def test_no_bow_name_on_wheel_lowers_right_away(clock):
    b, dev, env, emotes, events = lb(clock)
    b.cfg.social.after_light = ""
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()
    assert presses(dev) == 2 and emotes.done == []


def test_dry_run_only_logs(clock):
    b, dev, env, emotes, events = lb(clock, live=False)
    offer(env, clock)
    b.step()
    assert presses(dev) == 0 and env.tried == [7]


def test_policy_off_does_nothing(clock):
    class DenyLight(FakeSocial):
        def allowed(self, req):
            return req.kind != LIGHT

    b, dev, env, emotes, events = lb(clock, social=DenyLight())
    offer(env, clock)
    b.step()
    assert presses(dev) == 0 and env.tried == []


def test_not_while_holding_hands(clock):
    b, dev, env, emotes, events = lb(clock)
    b.holding = "小明"
    offer(env, clock)
    b.step()
    assert presses(dev) == 0


def test_light_request_makes_no_request_event_and_never_taps(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert "request" not in [e.kind for e in events.drain()]
    assert not [c for c in dev.calls if c[0] == "tap"]


def test_bow_after_accepting_candle(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    env.requests["陌生人"] = Request("陌生人", "candle", (990, 400), clock())
    social.to_handle = ["陌生人:candle"]
    b.step()
    env.requests.pop("陌生人")
    clock.advance(2.6)
    b.step()
    assert emotes.done == [("鞠躬", True)] and presses(dev) == 0


def test_reflex_waits_while_candle_raised(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert b._reflex_emote("挥手", "测试", clock()) is False


def test_status_mentions_raised_candle(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    assert "正在举蜡烛给陌生人点火" in b.status()


def test_policy_tool_accepts_stranger_light(clock):
    b, dev, env, emotes, events = lb(clock)
    b.set_policy("stranger", "light", False)
    assert b.social.policy_calls == [("stranger", "light", False)]
```

`body(...)` 支持 `social=` 关键字（它把 `**kw` 传给 `Body`）；`FakeSocial` 没有 `allowed`，所以测试里用子类。

- [ ] **Step 2: 跑测试看它失败**

Run: `python -m pytest tests/test_brain_light_body.py -q`
Expected: FAIL（`presses(dev) == 0`、`_reflex_emote` 返回 True 之类）

- [ ] **Step 3: 实现**

`body.py` 顶部：

```python
from ..game.social import IDLE, KIND_NAMES, LIGHT, LIGHT_KEY, PASSIVE  # 改原来第 28 行
```

```python
REQUEST_KINDS = ("hand", "hug", "highfive", "piggyback", "candle", "light", "*")
```

`__init__` 里（`self.skills = SkillRunner(...)` 附近）：

```python
        self._raised: tuple[int, tuple[int, int], float] | None = None  # 举着蜡烛等他亮起来：(轨迹 id, 圆盘位置, 举起时间)
        self._bow: tuple[float, float, float | None] | None = None  # 点火后要鞠躬：(到点, 放弃, 兜底放下的蜡烛几时举的)
        self._lit_at = float("-inf")
```

`step()` 里 `self._watch_idle(self.clock())` 后面加：

```python
        self._watch_light(self.clock())
        self._watch_bow(self.clock())
```

`_watch_requests` 改成（`light` 不进请求事件、不进 `_requests`、不交给 social；接受了点火就排鞠躬）：

```python
    def _watch_requests(self, now: float) -> None:
        """互动请求、按规则自动接受、牵手状态（跟踪中也照常）。light（团子举蜡烛点亮陌生人）在 _watch_light 里做。"""
        requests = {k: r for k, r in dict(self.env.requests).items() if r.kind != LIGHT}
        ...（原来的内容不变）...
            for item in handled:
                name, kind = item.split(":", 1)
                self.events.put("accepted", f"身体按规则接受了 {name} 的{KIND_NAMES.get(kind, kind)}")
                if kind == "hand":
                    self._accepted_hand = (name, now)
                if kind == "candle":
                    self._schedule_bow(now, None)
        self._watch_holding(now)
```

新方法（放在 `_watch_requests` 后面）：

```python
    # ---- 点亮没点火的陌生人（spec 2026-10-01-light-unlit-stranger） ----
    def _watch_light(self, now: float) -> None:
        """黑影在身边站够了：按 3 号键举蜡烛（绝不点他身上的火焰圆盘：点了会跟着他走），等 YOLO 看到他亮起来。"""
        if self._raised is not None:
            self._check_lit(now)
            return
        req = dict(self.env.requests).get(LIGHT_KEY)
        if req is None or req.track is None or self.social is None or self.emotes is None:
            return
        if not self.social.allowed(req):
            return
        if self.skills.active is not None or self.holding or self._bubble_blocked():
            return
        try:
            self.clear_view("emote")
        except ToolError as exc:
            log.debug("先不举蜡烛：%s", exc)
            return
        self.env.mark_tried(req.track)  # 不管成没成，这个人只举一次
        if self._dry(False):
            log.info("[dry-run] 会举蜡烛点亮身边的陌生人")
            return
        try:
            with self._held("wheel"):
                self.emotes.press_slot(self.cfg.social.candle_slot)
        except Exception:
            log.warning("举蜡烛没成功", exc_info=True)
            return
        self._raised = (req.track, req.pos, now)
        log.info("举起蜡烛给身边没点火的陌生人点火")

    def _check_lit(self, now: float) -> None:
        track, pos, raised_at = self._raised
        if self.env.lit(track, pos, raised_at):
            self._raised = None
            self._lit_at = now
            log.info("陌生人亮起来了（举蜡烛 %.1f 秒）", now - raised_at)
            self.events.put("accepted", "你举起蜡烛给身边一个没点火的陌生人点了火（他亮起来了）")
            self._schedule_bow(now, raised_at)
        elif now - raised_at >= self.cfg.social.light_timeout:
            self._raised = None
            log.warning("举了蜡烛 %.0f 秒他还是黑的（或者走了），不再点他", self.cfg.social.light_timeout)
            self._lower_candle(raised_at)

    def _schedule_bow(self, now: float, raised_at: float | None) -> None:
        """点亮了别人（raised_at = 举蜡烛的时间）/ 接受了别人点火（None）：过 bow_delay 鞠躬；做不了鞠躬时把自己举的蜡烛放下。"""
        name = self.cfg.social.after_light
        if not name or name not in self._wheel() or self.holding:
            if raised_at is not None:
                self._lower_candle(raised_at)
            return
        due = now + self.cfg.social.bow_delay
        self._bow = (due, due + 5.0, raised_at)

    def _watch_bow(self, now: float) -> None:
        if self._bow is None or now < self._bow[0]:
            return
        _, give_up, raised_at = self._bow
        if self._reflex_emote(self.cfg.social.after_light, f"点完火你{self.cfg.social.after_light}了一下", now):
            self._bow = None
        elif now >= give_up:
            self._bow = None
            log.info("点火后的%s一直做不了，算了", self.cfg.social.after_light)
            if raised_at is not None:
                self._lower_candle(raised_at)

    def _lower_candle(self, raised_at: float) -> None:
        """按 3 放下自己举的蜡烛——只在举起之后没做过任何动作时（做动作会放下蜡烛，再按就又举起来了）。"""
        if self.emotes is None or self.emotes.last_any > raised_at or self._dry(False):
            return
        try:
            with self._held("wheel"):
                self.emotes.press_slot(self.cfg.social.candle_slot)
            log.info("放下蜡烛")
        except Exception:
            log.warning("放下蜡烛没成功", exc_info=True)
```

`_reflex_emote` 开头的判断加上 `self._raised is not None`：

```python
        if self.emotes is None or self.blackout or self.skills.active is not None or self._requests or self._raised is not None:
            return False
```

`set_policy` 里：

```python
        if who == "stranger" and accept and kind not in ("candle", "light"):
            raise ToolError("陌生人只能接点火（candle）、点亮他（light），牵手 / 拥抱 / 击掌 / 背背都不接陌生人的")
```

`status()` 里 `if self.holding:` 那一行前面加：

```python
        if self._raised is not None:
            parts.append("正在举蜡烛给陌生人点火")
        elif now - self._lit_at <= self.cfg.social.remember:
            parts.append(f"{now - self._lit_at:.0f} 秒前你给一个陌生人点了火")
```

`shutdown()` 里 `self.emotes.restore()` 之前：

```python
        if self._raised is not None:  # 退出时还举着蜡烛：放下
            self._lower_candle(self._raised[2])
            self._raised = None
```

`tools.py` 的 `set_request_policy` 说明：

```python
        f"kind：{' / '.join(REQUEST_KINDS)}（hand 牵手、hug 拥抱、highfive 击掌、piggyback 背背、candle 点火、light 点亮没点火的陌生人、* 所有）；"
        "accept：接不接。陌生人只能用 candle 和 light。"
```

`prompt.py` 的「互动请求」一节末尾加一行：

```
- 没点火的陌生人（黑影）站到你身边一会儿，身体会自动举蜡烛给他点火、点上后鞠个躬，会告诉你；不想点可以 set_request_policy("stranger", "light", false)。
```

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_brain_light_body.py tests/test_brain_body.py tests/test_brain_reflex_body.py -q`
Expected: 全部 passed

- [ ] **Step 5: 全量测试**

Run: `python -m pytest -q`
Expected: 全部 passed（提示词快照之类的测试因为「互动请求」多了一行而挂的话，按新文字更新快照）

- [ ] **Step 6: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/tools.py src/skydango/brain/prompt.py tests/test_brain_light_body.py
git commit -m "feat(body): 黑影在身边站够了按 3 举蜡烛，YOLO 看到他亮起来就鞠躬；接受点火后也鞠躬

绝不点火焰圆盘（会跟着人走）；8 秒没亮起来按 3 放下；做过动作就不再按 3。

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: 录像 c 上核对 `lit()` + 文档

**Files:**
- Modify: `docs/game-ops.md`（§4 快捷轮盘、§6 圆圈图标表）
- Modify: `CLAUDE.md`（「识别环境」一节末尾的社交互动那段）
- Modify: `docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md`（G 期）
- Modify（可能）：`src/skydango/config.py`（`bow_delay`）

- [ ] **Step 1: 录像 c 离线跑一遍感知层**

写 `tmp/eval_lit.py`：用 `cli._scene_watcher` 同样的参数（读 `config.toml`，`background=False`，`capture=None`）建 `PerceptionWatcher`，按 `tmp/record/candle-20260930-c/` 文件名里的秒数依次 `process(frame, t, panel_visible=False)` 第 300~420 帧（30~42 秒），每帧打印：
每条人物轨迹 `id cls player_run 框中心x`、`requests` 里有没有 `LIGHT_KEY`。在第一次出现 `player_unlit` 那条轨迹时记下 `(id, 框中心, t)`，之后每帧打印 `w.lit(id, (框中心x, 框中心y), t0)`。
看：第 35.5~37 秒闪光时 id 断没断；`lit()` 从第几秒开始为 True；有没有过早为 True（闪光前）。

- [ ] **Step 2: 按结果调**

- 闪光时 `lit()` 一直是 None / False、明明亮了：看是 id 断了兜底没接上（调 `lit()` 里的距离条件）还是 `player_run` 不够（调 `lit_frames`）
- 从"黑影开始闪光"到"亮了"大约几秒 → 如果明显不是 2.5 s 量级，改 `SocialConfig.bow_delay`（等闪光动画播完）
- 改了代码就补一个对应的单元测试，`python -m pytest -q` 全过再往下

- [ ] **Step 3: 写文档**

`docs/game-ops.md` §4「快捷轮盘」加一条：

```
- **3 号格常驻"举蜡烛"**（用户 2026-10-01）：按数字键 3 举起蜡烛，再按一次放下；做任何动作（比如鞠躬）会顺带放下蜡烛。原地举着、不会跟人走
```

§6 圆圈图标表里"深色实心圆 + 白色火焰"那一行改成：

```
| 深色实心圆 + 白色空心火焰（**没有白圈**） | 模板 `assets/candle/flame.png`（只有火焰；`candle*.png` 带白圈，认不好） | 没点火的黑影站到团子身边时出现在他身上（录像 `tmp/record/candle-20260930-d/` 第 16 秒） | **绝不点**：点了团子会一直跟着这个陌生人走、直到点上为止。团子按 3 号键举蜡烛给他点（`[social]` 的 light，spec 2026-10-01）；点上 = 黑影显出外观（YOLO `player_unlit` → `player`，录像 c 第 35.5~37 秒）。上方冒彩虹爱心**不代表**点上了（录像 d 冒了爱心的人到第 38 秒还是黑的） |
```

再把 Step 1 量到的结果（闪光多久、id 断没断、`lit()` 第几秒为真）写在表格下面一条。

`CLAUDE.md`「识别环境」一节里讲 `[social]` 的那段末尾加：

```
没点火的黑影站到团子身边 `light_after` 秒（他身上出现深色圆盘 + 火焰，`vision/candle.py` 认），大脑模式下身体按 3 号键举蜡烛（**绝不点那个圆盘**：会跟着人走），
YOLO 看到他从 `player_unlit` 变成 `player` 就算点亮、过 `bow_delay` 鞠躬（顺带放下蜡烛）；`light_timeout` 秒没亮就按 3 放下、这个人不再点。接受别人点火后也鞠躬。
大脑能用 `set_request_policy("stranger", "light", false)` 关掉；**未在真机验证**（spec `docs/superpowers/specs/2026-10-01-light-unlit-stranger-design.md` §9）
```

`2026-09-28-brain-skills-roadmap.md` 的 G 期「用户 09-30 定下的待办」那段后面加：

```
**2026-10-01 做了第一步**（spec `docs/superpowers/specs/2026-10-01-light-unlit-stranger-design.md`，计划 `docs/superpowers/plans/2026-10-01-light-unlit-stranger.md`）：
身体反射版，不走技能、不走过去——黑影自己站到身边才点；按 3 举蜡烛、YOLO 核实、鞠躬。下面的 `LightCandleSkill`（走过去点远处的人）还没做，要等 F 期。
```

- [ ] **Step 4: 全量测试 + 提交**

Run: `python -m pytest -q`
Expected: 全部 passed

```bash
git add docs/game-ops.md CLAUDE.md docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md src/skydango/config.py
git commit -m "docs: 点亮陌生人——3 号键举蜡烛、火焰圆盘绝不点、录像 c 上 lit() 的核对结果

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

（`config.py` 没改就别 add 它。）

---

## 晚上真机验证（用户在时，不在本计划的任务里）

照 spec §9：dry-run 看日志 → live 看举蜡烛多久亮、鞠躬、蜡烛放下 → 陌生人给团子点火后鞠躬 → 大脑关掉 light 后不举。结果写进 game-ops §6。
