# 点亮陌生人：在团子周围找火焰 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 点亮陌生人不再依赖 YOLO 认出黑影：在团子框周围直接找火焰，连续够久就举蜡烛；举起后按"火焰消失 + 原地的人变亮 / 没人"判结果，每次都存图留证据。

**Architecture:** `vision/candle.py` 出两个纯函数（`find_flame` 在给定范围里找火焰、`black` 量人物框有多黑）；`vision/perception.py` 用一条"火焰线索"（不挂人物轨迹）代替 `_watch_disks` 那一套，`mark_tried` / `lit` / `light_done` 改成按线索编号，加冷却和存图；`brain/body.py` 处理"他走了"和各种结局都调 `light_done`。

**Tech Stack:** Python 3.13、numpy、OpenCV、pytest（合成画面 + 假检测器）。

Spec：`docs/superpowers/specs/2026-10-01-light-flame-around-self-design.md`（下称 spec）。

## Global Constraints

- **绝不点火焰圆盘**（点了团子会跟着人走）；只按 3 号键举 / 放蜡烛。接受陌生人点火（白圈孤儿圆圈）那条路（`_disk_like`、`white_ring`、`social.py`）不动
- 只有大脑模式 + 开着感知层才找火焰（`_scene_watcher(cfg, light=True)` 才加载火焰模板，不变）
- 数字（spec §8）：`light_area_x = 1.2`、`light_area_up = 0.6`、`light_jump = 0.5`、`light_cooldown = 60.0`、`lit_v = 50`、`lit_black = 0.35`、`lit_drop = 0.25`；`disk_min_score 0.68` / `disk_sure 0.85` / `light_after 3.0` / `lit_min 2.0` / `light_timeout 8.0` 不变。Task 1 量完可能改前三个和 `lit_*` 的默认值，以 Task 1 写回 config 的为准
- 代码常量：`DISK_EVERY = 0.3`、`DISK_GAP = 1.0`、`SELF_MAX_AGE = 1.0`、`LIT_LOW = 0.2`、`DIAG_EVERY = 0.5`、`DIAG_MAX = 30`、`DIAG_RUNS = 50`、`FRAME_STALE = 0.5`
- 删掉的配置键：`lit_frames`、`lit_iou`、`lit_stale`
- 测试：`.venv\Scripts\python.exe -m pytest -q`（系统 Python 可能没装 pytest）；临时脚本放 `tmp/`，不进 git
- 用中文写注释、日志、文档；注释密度和现有代码一样
- 分支 `feat/light-flame`，做完合并进 main 并推送

## 文件

| 文件 | 改动 |
| --- | --- |
| `src/skydango/vision/candle.py` | `find_disk` + `_region` → `find_flame(frame, area, flame, min_score)`；新增 `black(frame, box, v)` |
| `src/skydango/config.py` | `[social]` 加 7 个键、删 3 个键 |
| `src/skydango/vision/perception.py` | 删 `_watch_disks` / `_tried` / `_lit_mem` / `LIT_MEM_MAX` / `player_run` / 旧 `lit()`；新增火焰线索、`mark_tried` / `lit` / `light_done`、冷却、存图、`overlay` 画火焰、`detector_conf(light=)` |
| `src/skydango/cli.py` | `_scene_watcher`：检测器阈值带上 light、传 `light_dir=run.path / "light"` |
| `src/skydango/brain/body.py` | `_check_lit` 处理 None；各种结局调 `env.light_done` |
| `tests/test_candle.py` `tests/test_perception.py` `tests/test_brain_light_body.py` | 改 / 加测试 |
| `docs/game-ops.md` §6、`CLAUDE.md` | 改成新做法 + Task 1 量到的数 |

---

### Task 1: 离线先量（spec §9，controller 自己做，不派子代理）

**Files:**
- Create: `tmp/lightflame/measure_area.py`、`tmp/lightflame/measure_black.py`（不进 git）
- 结果记在 `tmp/lightflame/results.md`，Task 7 写进 game-ops §6；数字写回 Task 3 的 config 默认值

- [ ] **Step 1: 量搜索范围。** 对 `tmp/lightdbg/w_*.jpg`（约 1860 张，今晚三次运行的帧）和三次运行的 `runs/20261001-2*-live-brain/hard/*.jpg`：用 `models/sky-yolo-v7.onnx`（`skydango.vision.detect.make_detector`）取分数最高的 `self` 框；整张图（去掉左边聊天面板和底部输入栏）用 `find_disk` 同样的匹配（米白剪影 + 模糊 + 多尺度）找火焰 ≥ 0.68 的所有位置（非极大值抑制，半径 40 px）。输出每张：团子框、火焰候选（位置、分数）、火焰相对团子框的位置 `dx = (fx − cx) / H`、`dy = (fy − self.y) / H`
- [ ] **Step 2: 人工核对。** 把 ≥ 0.85 的候选和随机抽的 0.68~0.85 候选裁成小图拼成总览（每张 40 个，带文件名和分数），看图标出哪些是真火焰、哪些是灯笼 / 别的；统计真火焰 `|dx|` 最大值、`dy` 最小值（往上最多多少）、最大值
- [ ] **Step 3: 定范围。** `light_area_x = max|dx| × 1.1`、`light_area_up = max(−dy) × 1.1`（向上取到 0.05）；列出范围内假火焰的最高分（应 < 0.85，否则记进风险）
- [ ] **Step 4: 量 black()。** 录像 `tmp/record/candle-20260930-c/` 第 30~40 秒：每帧跑 YOLO，取那个黑影的框（`player` / `player_unlit`，按位置连起来；没框的帧手标一个固定框），逐帧算"框中间一半宽、上 70% 高里 HSV V < v 的像素占比"，v 取 40 / 50 / 60 三档；再算今晚举蜡烛前（22:50:41 前后，`tmp/lightdbg/w_2250*.jpg`）那个黑影的值作对照。定 `lit_v`、`lit_black`、`lit_drop`：点亮后的值 < `lit_black` < 点亮前的值，`lit_drop` ≤ 前后差的一半
- [ ] **Step 5: 写 `tmp/lightflame/results.md`**：样本数、真火焰 dx / dy 分布、选的范围、假火焰最高分、black 点亮前后曲线（数字表）、选的三个数；量不出明显差别就写明"退回火焰消失 + 那里有人"（这时 `lit_black = 1.0`、`lit_drop = 0.0`，让 black 判断恒过）

---

### Task 2: candle.py：`find_flame` + `black`

**Files:**
- Modify: `src/skydango/vision/candle.py`（删 `_region`、`find_disk`；加 `find_flame`、`black`；改模块说明）
- Test: `tests/test_candle.py`

**Interfaces:**
- Produces: `find_flame(frame: np.ndarray, area: Rect, flame: np.ndarray, min_score: float = 0.68) -> Disk | None`（`area` 是整图坐标，裁到画面内；宽或高 < 16 返回 None）；`black(frame: np.ndarray, box: Rect, v: int = 50) -> float`（0~1；框裁完是空的返回 0.0）；`Disk(x, y, r, score)` 不变

- [ ] **Step 1: 改测试。** `tests/test_candle.py`：import 改成 `from skydango.vision.candle import Disk, black, find_flame, load_flame`；`BOX` 改成搜索范围 `AREA = Rect(810, 284, 360, 576)`（等于旧 `_region(BOX)`），所有 `find_disk(f, BOX, FLAME)` → `find_flame(f, AREA, FLAME)`；`CROP_BOX` → `CROP_AREA = Rect(0, 0, 200, 200)`，`find_disk(img, CROP_BOX, FLAME)` → `find_flame(img, CROP_AREA, FLAME)`。再加：

```python
def test_flame_outside_area_is_not_found():
    """范围由调用方给：火焰在范围外就当没有。"""
    assert find_flame(figure(), Rect(1300, 284, 300, 576), FLAME) is None


def test_area_clipped_to_frame():
    d = find_flame(figure(), Rect(700, -200, 2000, 2000), FLAME)
    assert d is not None and abs(d.x - 990) <= 6


def test_tiny_area_returns_none():
    assert find_flame(figure(), Rect(980, 610, 10, 10), FLAME) is None


def test_black_silhouette_vs_colored_person():
    f = np.full((400, 400, 3), (60, 90, 40), np.uint8)
    cv2.rectangle(f, (50, 50), (150, 350), (12, 12, 12), -1)  # 黑影
    cv2.rectangle(f, (250, 50), (350, 350), (40, 120, 220), -1)  # 点亮后：衣服有颜色
    assert black(f, Rect(50, 50, 100, 300)) > 0.9
    assert black(f, Rect(250, 50, 100, 300)) < 0.1


def test_black_uses_middle_half_and_top_70_percent():
    f = np.full((400, 400, 3), (40, 120, 220), np.uint8)
    cv2.rectangle(f, (0, 0), (24, 399), (0, 0, 0), -1)  # 框最左 1/4 黑：不在中间一半里
    cv2.rectangle(f, (0, 300), (399, 399), (0, 0, 0), -1)  # 下面 30% 黑：不算
    assert black(f, Rect(0, 0, 100, 400)) == 0.0


def test_black_empty_box_is_zero():
    assert black(np.zeros((100, 100, 3), np.uint8), Rect(200, 200, 50, 50)) == 0.0
```

- [ ] **Step 2: 跑测试看它失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_candle.py -q`
Expected: ImportError（`find_flame` / `black` 不存在）

- [ ] **Step 3: 实现。** 删 `_region`、`find_disk`，加：

```python
def find_flame(frame: np.ndarray, area: Rect, flame: np.ndarray, min_score: float = 0.68) -> Disk | None:
    """在 area（整图坐标，调用方给：团子周围那一块）里找火焰，只取最好的一处；没有返回 None。
    有没有白圈、外环亮不亮都不看（10-01 晚真机）；分数够不够确定由调用方按 disk_sure 看。"""
    x1, y1 = max(0, area.x), max(0, area.y)
    x2, y2 = min(frame.shape[1], area.x2), min(frame.shape[0], area.y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    mask = cv2.GaussianBlur(cream(frame[y1:y2, x1:x2]), (0, 0), BLUR)
    flame = cv2.GaussianBlur(flame, (0, 0), BLUR)
    best, scale = None, 1.0
    for s in SCALES:
        m = best_match(mask, flame, [s])
        if best is None or m.score > best.score:
            best, scale = m, s
    if best is None or best.score < min_score:
        return None
    return Disk(x1 + best.x, y1 + best.y, flame.shape[0] * scale / 2, best.score)


def black(frame: np.ndarray, box: Rect, v: int = 50) -> float:
    """人物框有多黑：中间一半宽、上 70% 高里很暗（HSV 的 V < v）的像素占比。黑影 ≈ 高，点亮后显出衣服颜色 ≈ 低。
    下面 30% 不看（脚下的影子），两边不看（背景）。框和画面不重叠返回 0。"""
    x1 = max(0, round(box.x + box.w / 4))
    x2 = min(frame.shape[1], round(box.x + 3 * box.w / 4))
    y1 = max(0, box.y)
    y2 = min(frame.shape[0], round(box.y + 0.7 * box.h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    value = frame[y1:y2, x1:x2].max(axis=2)  # HSV 的 V = BGR 三通道最大值
    return float((value < v).mean())
```

模块说明第二段改成：`find_flame` 在调用方给的范围（团子周围）里找火焰（spec 2026-10-01-light-flame-around-self），只用来判断"身边有个能点火的黑影"……；加一句 `black` 判点亮。

- [ ] **Step 4: 跑测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_candle.py -q`
Expected: 全过（`test_perception.py` 这时会因为 `find_disk` 没了而报错，Task 3 修）

- [ ] **Step 5: Commit**

```bash
git add src/skydango/vision/candle.py tests/test_candle.py
git commit -m "feat(candle): find_flame 在给定范围找火焰、black 量人物有多黑"
```

---

### Task 3: 配置 + 感知层的火焰线索和请求（spec §3）

**Files:**
- Modify: `src/skydango/config.py`（`SocialConfig`）
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception.py`（删旧的点亮测试，加新的）

**Interfaces:**
- Consumes: `find_flame`、`black`（Task 2）
- Produces（Task 4~6 用）：
  - `PerceptionWatcher._flame: dict | None`，键 `id`（int，自增从 1 起）、`first`、`last`、`best`、`score`、`pos`（(x, y)）、`r`、`black`（火焰下面那个人的 `black()`，没人是 None）、`announced`（bool）
  - `PerceptionWatcher._lighting: dict | None`（Task 4 填，本 task 只在出请求时检查 `is None`）
  - `PerceptionWatcher._cooldown_until: float`（初值 `-inf`）
  - `PerceptionWatcher._scan: dict | None`：最近一次找火焰的 `area`（Rect）、`me`（Rect）、`flame`（Disk | None）——存图（Task 5）用
  - `PerceptionWatcher._people_boxes: list[tuple[Rect, float]]`：这一帧 player / player_unlit 框（含 ≥ `LIT_LOW` 的低分框，去掉和 self 重叠 ≥ 0.5 的）
  - `PerceptionWatcher._person_at(pos: tuple[int, int], r: float) -> Rect | None`
  - 模块常量 `SELF_MAX_AGE`、`LIT_LOW`、`FRAME_STALE`
  - `detector_conf(cfg, light: bool = False) -> float`
  - 钩子 `self._on_request(now)`（本 task 写成空方法 `pass`，Task 5 填存图）

- [ ] **Step 1: config。** `SocialConfig` 里点亮陌生人那段：删 `lit_frames`、`lit_iou`、`lit_stale` 三行；改 `light_after` / `light_timeout` 注释；在 `lit_min` 后加：

```python
    light_area_x: float = 1.2  # 在团子周围找火焰：左右各多少倍团子框高（spec 2026-10-01-light-flame-around-self §9 量）
    light_area_up: float = 0.6  # 从团子框上沿往上多少倍框高（下到框底）
    light_jump: float = 0.5  # 相邻两次火焰位置差多少倍团子框高以内算同一个人
    light_cooldown: float = 60.0  # 没点亮（走了 / 超时 / 被打断）之后多久不再举：认不出是谁，只能按时间
    lit_v: int = 50  # 判点亮：HSV 的 V 低于这个算"很暗"的像素
    lit_black: float = 0.35  # 判点亮：人物框里很暗的像素占比低于这个……
    lit_drop: float = 0.25  # ……而且比举蜡烛时降了这么多以上，才算点亮
```

（数字以 Task 1 结果为准；`light_after` 注释改成"团子身边的火焰连续看到这么久才举蜡烛（别点路过的）"，`light_timeout` 注释改成"举着蜡烛最多等这么久看他亮起来"。）

- [ ] **Step 2: 写失败测试。** 删 `tests/test_perception.py` 里 `# ---- 点亮陌生人（spec 2026-10-01-light-unlit-stranger §3） ----` 到 `# ---- 孤儿圆圈：深色火焰圆盘不是举蜡烛请求（最终审查 #1） ----` 之间的全部内容（保留孤儿圆圈那段和 `test_scene_watcher_loads_flame_only_for_brain`），换成：

```python
# ---- 点亮陌生人：在团子周围找火焰（spec 2026-10-01-light-flame-around-self §3） ----
from skydango.config import SocialConfig
from skydango.vision import perception as perception_mod
from skydango.vision.candle import Disk
from skydango.game.social import LIGHT_KEY

SELF = Rect(1200, 500, 100, 250)  # 团子框：H = 250，cx = 1250


def me(box=SELF, score=0.7):
    return Detection("self", box, score)


def unlit(x, y=400, w=90, h=220, score=0.9):
    return Detection("player_unlit", Rect(x, y, w, h), score)


def light_watcher(monkeypatch, flames):
    """flames：每次 find_flame 依次返回什么（用完了一直返回最后一个）；记下每次给的范围。"""
    seq, areas = list(flames), []

    def fake(frame, area, flame, s):
        areas.append(area)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(perception_mod, "find_flame", fake)
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    w.light_cfg, w.flame = SocialConfig(), np.ones((4, 4), np.uint8)
    w.areas = areas
    return w, det, clock


FLAME = Disk(1045, 480, 20.0, 0.9)


def run(w, t, clock):
    clock.t = t
    w.process(frame(), t, panel_visible=False)


def run_frames(w, clock, start, end, step=0.1):
    k = 0
    while start + k * step <= end + 1e-9:
        run(w, round(start + k * step, 3), clock)
        k += 1


def test_light_request_after_flame_seen_long_enough(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me(), unlit(1000)]]
    for t in (0.0, 1.0, 2.0, 2.5):
        run(w, t, clock)
        assert LIGHT_KEY not in w.requests
    run(w, 3.1, clock)
    req = w.requests[LIGHT_KEY]
    assert req.kind == "light" and req.pos == (1045, 480) and req.track == 1


def test_light_request_without_any_person_box(monkeypatch):
    """10-01 22:52:39：晚上 YOLO 一个人物框都没给，火焰照样出请求。"""
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests


def test_search_area_around_self_box(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[me()]]
    run(w, 0.0, clock)
    a = w.areas[-1]
    cfg = SocialConfig()
    assert a.x == round(1250 - cfg.light_area_x * 250) and a.x2 == round(1250 + cfg.light_area_x * 250)
    assert a.y == round(500 - cfg.light_area_up * 250) and a.y2 == 750


def test_no_self_box_no_search(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[unlit(1000)]]
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    assert w.areas == [] and LIGHT_KEY not in w.requests


def test_stale_self_track_still_used_within_a_second(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]
    run(w, 0.0, clock)
    det.frames = [[]]
    run(w, 0.5, clock)  # 这一帧没认出团子，但 0.5 秒前有
    assert len(w.areas) == 2
    run(w, 1.6, clock)  # 超过 SELF_MAX_AGE：不找
    assert len(w.areas) == 2


def test_scan_throttled(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[me()]]
    for t in (0.0, 0.1, 0.2, 0.3):
        run(w, t, clock)
    assert len(w.areas) == 2  # DISK_EVERY = 0.3：0.0 和 0.3


def test_needs_one_sure_flame(monkeypatch):
    weak = Disk(1045, 480, 20.0, 0.75)
    w, det, clock = light_watcher(monkeypatch, [weak])
    det.frames = [[me()]]
    for t in (0.0, 1.0, 2.0, 2.5, 3.1, 3.5):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests


def test_long_gap_starts_new_clue(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME, None, None, FLAME, FLAME])
    det.frames = [[me()]]
    for t in (0.0, 0.5, 1.2, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests and w._flame["id"] == 2


def test_short_gap_keeps_clue(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME, FLAME, None, FLAME, FLAME])
    det.frames = [[me()]]
    for t in (0.0, 1.0, 1.5, 2.0, 2.5, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests and w._flame["id"] == 1


def test_jump_too_far_starts_new_clue(monkeypatch):
    far = Disk(1045 + 200, 480, 20.0, 0.9)  # 0.8 × 团子框高 > light_jump
    w, det, clock = light_watcher(monkeypatch, [FLAME, FLAME, far, far])
    det.frames = [[me()]]
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests and w._flame["id"] == 2


def test_flame_under_friend_tag_is_ignored(monkeypatch):
    """好友举蜡烛给团子点火：火焰在他名字标签下面的圆圈里，不是黑影。"""
    w, det, clock = light_watcher(monkeypatch, [Disk(1045, 400, 20.0, 0.95)])
    w.ocr = FakeOcr({110: "懒洋洋大王"})
    det.frames = [[me(), tag(990, 110, y=300)]]  # 标签 990~1100、300~344；排除区 y 300 ~ 300 + 4.5×44
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY not in w.requests


def test_unread_tag_does_not_exclude(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [Disk(1045, 400, 20.0, 0.95)])
    det.frames = [[me(), tag(990, 110, y=300)]]  # 名字没认出来（FakeOcr 空表）
    for t in (0.0, 1.0, 2.0, 3.1):
        run(w, t, clock)
    assert LIGHT_KEY in w.requests


def test_flame_under_open_panel_is_ignored(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [Disk(600, 480, 20.0, 0.95)])  # 面板右边到 x = 643
    det.frames = [[me(Rect(700, 500, 100, 250))]]  # 团子框中心在面板外（面板里的框会被 _filter 去掉）
    for t in (0.0, 1.0, 2.0, 3.1):
        clock.t = t
        w.process(frame(), t, panel_visible=True)
    assert LIGHT_KEY not in w.requests


def test_overlay_shows_current_flame(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]
    run(w, 0.0, clock)
    rings = [e for e in w.overlay(0.0) if e["label"].startswith("火焰")]
    assert rings and rings[0]["kind"] == "ring" and rings[0]["x"] == 1025


def test_resume_shifts_clue_times(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]
    run_frames(w, clock, 0.0, 1.0)
    w.hold("blackout")
    clock.t = 4.0
    w.release("blackout")  # 暂停 3 秒
    assert w._flame["first"] == 3.0 and w._flame["last"] == 4.0


def test_detector_conf_lower_with_light():
    from skydango.vision.perception import LIT_LOW, detector_conf

    cfg = PerceptionConfig(hardcases=False)
    assert detector_conf(cfg) == cfg.conf
    assert detector_conf(cfg, light=True) == LIT_LOW


def test_people_boxes_keep_low_scores(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [None])
    det.frames = [[me(), unlit(1000, score=0.27), unlit(600, score=0.1)]]
    run(w, 0.0, clock)
    assert [round(s, 2) for _, s in w._people_boxes] == [0.27]
```

- [ ] **Step 3: 跑测试看失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py -q -k "light or flame or self_box or scan or clue or jump or tag or panel or overlay_shows or resume_shifts or detector_conf or people_boxes"`
Expected: 失败（`find_flame` 没被感知层用、`_flame` 不存在……）

- [ ] **Step 4: 实现。** `perception.py`：
  1. import：`from .candle import black, find_flame, white_ring`（删 `find_disk`）；加 `from pathlib import Path`
  2. 常量：删 `LIT_MEM_MAX`；`DISK_EVERY` 注释改"团子周围最多隔这么久找一次火焰"、`DISK_GAP` 注释改"火焰断开不超过这么久算同一条线索（火焰会晃）"；加

```python
SELF_MAX_AGE = 1.0  # 找火焰：团子框这么久没更新就不用（宁可漏，不全屏乱找）
LIT_LOW = 0.2  # 点亮陌生人开着时检测器按这个出框：晚上黑影分数低（10-01 晚 0.27 / 0.28），找"火焰下面那个人"时也看低分框
FRAME_STALE = 0.5  # lit()：最近一帧比这更旧（感知没在跑）就不判
```

  3. `detector_conf`：

```python
def detector_conf(cfg: PerceptionConfig, light: bool = False) -> float:
    """检测器的出框阈值：收集难例时要看到 low_conf ~ conf 之间的框，点亮陌生人要看到 LIT_LOW 以上的黑影；判定仍按 conf。"""
    conf = min(cfg.low_conf, cfg.conf) if cfg.hardcases else cfg.conf
    return min(conf, LIT_LOW) if light else conf
```

  4. `__init__`：删 `self._lit_mem`、`self._tried` 两行，`_frame_at` 注释改成"lit() 判最近一帧够不够新"；加

```python
        self._flame: dict | None = None  # 团子身边现在的火焰线索（spec 2026-10-01-light-flame-around-self §3；self._lock 保护）
        self._flame_seq = 0
        self._flame_check = float("-inf")
        self._flame_log = float("-inf")  # DEBUG 日志每秒最多一行
        self._lighting: dict | None = None  # 举着蜡烛等结果（mark_tried ~ light_done）
        self._cooldown_until = float("-inf")  # 没点亮之后这之前不出请求
        self._scan: dict | None = None  # 最近一次找火焰：area / me / flame（存图用）
        self._people_boxes: list[tuple[Rect, float]] = []  # 这一帧的人物框（含 LIT_LOW 以上的低分框）
```

  5. `_resume`：删挪 `disk_*` 键和 `_lit_mem` 的两段，换成

```python
        with self._lock:  # 火焰线索 / 点亮中的时间也跟着挪：暂停的时间不算"火焰消失"
            if self._flame is not None:
                self._flame["first"] += d
                self._flame["last"] += d
            if self._lighting is not None:
                for key in ("raised", "flame_last", "person_at"):
                    self._lighting[key] += d  # -inf 加 d 还是 -inf
            self._cooldown_until += d
```

（`_lighting` 的这三个键 Task 4 建；本 task 先写上，`_lighting` 一直是 None。）

  6. `process()`：把 `self._watch_disks(frame, players, now, height)` 换成

```python
        self._people_boxes = [
            (d.box, d.score) for d in people_boxes(dets + [x for x in low if x.score >= LIT_LOW]) if d.cls != "self"
        ]
        self._watch_flames(frame, tags, now, width, height, panel_visible)
```

  7. 删 `_watch_disks`、旧 `mark_tried`、旧 `lit`，在 `_disk_like` 后面加：

```python
    def _self_now(self, now: float) -> Rect | None:
        """团子现在在哪：最近 SELF_MAX_AGE 秒内更新过的 self 轨迹里分数最高的（不用 self_box / self_roi：团子会被镜头带着偏）。"""
        selfs = [t for t in self.tracker.tracks.values() if t.cls == "self" and now - t.last <= SELF_MAX_AGE]
        return max(selfs, key=lambda t: t.score).box if selfs else None

    def _flame_area(self, me: Rect, width: int, height: int) -> Rect:
        cfg, cx = self.light_cfg, me.x + me.w / 2
        x1, x2 = round(cx - cfg.light_area_x * me.h), round(cx + cfg.light_area_x * me.h)
        y1 = round(me.y - cfg.light_area_up * me.h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, me.y2)
        return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _flame_excluded(self, pos: tuple[int, int], tags: list[Track], width: int, height: int,
                        panel_visible: bool) -> str | None:
        """火焰落在这些地方当没看到：认出了名字的标签下面（好友举蜡烛给团子点火的圆圈）、开着的聊天面板。"""
        for t in tags:
            b = t.box
            if t.data.get("name") and _inside(pos, Rect(round(b.x - 0.5 * b.w), b.y, 2 * b.w, round(4.5 * b.h))):
                return f"在 {t.data['name']} 的名字标签下面"
        if panel_visible and _inside(pos, roi_rect(self.log_roi, width, height)):
            return "聊天面板挡着"
        return None

    def _person_at(self, pos: tuple[int, int], r: float) -> Rect | None:
        """火焰往下那一块（横向 ±2r、纵向 −2r ~ +8r：火焰在胸口或头顶）里重叠最多的人物框（含低分框）。"""
        area = Rect(round(pos[0] - 2 * r), round(pos[1] - 2 * r), round(4 * r), round(10 * r))
        best, most = None, 0
        for box, _ in self._people_boxes:
            w = min(area.x2, box.x2) - max(area.x, box.x)
            h = min(area.y2, box.y2) - max(area.y, box.y)
            if w > 0 and h > 0 and w * h > most:
                best, most = box, w * h
        return best

    def _watch_flames(self, frame: np.ndarray, tags: list[Track], now: float, width: int, height: int,
                      panel_visible: bool) -> None:
        """团子身边连续 light_after 秒冒着火焰（够近的黑影身上才有）→ 出 light 请求，不管 YOLO 认没认出这个人。"""
        cfg = self.light_cfg
        if cfg is None or self.flame is None or now - self._flame_check < DISK_EVERY - 1e-6:  # 1e-6：0.6 - 0.3 < 0.3
            return
        self._flame_check = now
        why, flame = "", None
        me = self._self_now(now)
        if me is None:
            why = "没有团子框"
            self._scan = None
        else:
            area = self._flame_area(me, width, height)
            flame = find_flame(frame, area, self.flame, cfg.disk_min_score)
            if flame is not None:
                skip = self._flame_excluded((flame.x, flame.y), tags, width, height, panel_visible)
                if skip:
                    why, flame = f"火焰 {flame.score:.2f} {skip}", None
            self._scan = {"area": area, "me": me, "flame": flame}
        with self._lock:
            clue = self._flame
            if flame is not None:
                pos = (flame.x, flame.y)
                person = self._person_at(pos, flame.r)
                blk = black(frame, person, cfg.lit_v) if person is not None else None
                same = (clue is not None and now - clue["last"] <= DISK_GAP
                        and np.hypot(pos[0] - clue["pos"][0], pos[1] - clue["pos"][1]) <= cfg.light_jump * me.h)
                if not same:
                    self._flame_seq += 1
                    clue = self._flame = {"id": self._flame_seq, "first": now, "best": 0.0, "announced": False}
                clue.update(last=now, pos=pos, r=flame.r, score=flame.score, black=blk, best=max(clue["best"], flame.score))
            elif clue is not None and now - clue["last"] > DISK_GAP:
                clue = self._flame = None
            lighting, cooling = self._lighting, now < self._cooldown_until
            if lighting is not None and clue is not None and clue["id"] == lighting["id"]:
                lighting["flame_last"], lighting["pos"], lighting["r"] = clue["last"], clue["pos"], clue["r"]
            ready = (clue is not None and lighting is None and not cooling and now - clue["last"] <= DISK_GAP
                     and now - clue["first"] >= cfg.light_after and clue["best"] >= cfg.disk_sure)
            first = ready and not clue["announced"]
            if ready:
                clue["announced"] = True
        if now - self._flame_log >= 1.0:
            self._flame_log = now
            if clue is not None:
                log.debug("火焰线索 %d：(%d, %d) %.2f，已 %.1f 秒，best %.2f%s%s", clue["id"], *clue["pos"], clue["score"],
                          now - clue["first"], clue["best"], "（冷却中）" if cooling else "", "（正在点亮）" if lighting else "")
            elif why:
                log.debug("没找火焰 / 不算：%s", why)
        if not ready:
            self.requests.pop(LIGHT_KEY, None)
            return
        if first:
            log.info("身边有没点火的陌生人：火焰出现了 %.0f 秒（线索 %d）", now - clue["first"], clue["id"])
            self._on_request(frame, now)
        self.requests[LIGHT_KEY] = Request(STRANGER, LIGHT, clue["pos"], now, track=clue["id"])

    def _on_request(self, frame: np.ndarray, now: float) -> None:
        """第一次出请求：存图（Task 5）。"""
```

（`_on_request` 本 task 只有文档串；Task 5 填。本 task 里 `mark_tried` / `lit` / `light_done` 暂不存在——`tests/test_brain_light_body.py` 用假 env，不受影响。）

  8. `overlay()`：在 `return out` 前加

```python
        with self._lock:
            clue = self._flame
            if clue is not None and now - clue["last"] <= DISK_GAP:
                (x, y), r = clue["pos"], clue["r"]
                out.append({"x": round(x - r), "y": round(y - r), "w": round(2 * r), "h": round(2 * r),
                            "kind": "ring", "label": f"火焰 {clue['score']:.2f}", "score": round(clue["score"], 2)})
```

- [ ] **Step 5: 跑测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py tests/test_candle.py -q`
Expected: 全过（`test_scene_watcher_loads_flame_only_for_brain` 也过）

- [ ] **Step 6: Commit**

```bash
git add src/skydango/config.py src/skydango/vision/perception.py tests/test_perception.py
git commit -m "feat(perception): 点亮陌生人改成在团子周围找火焰，不依赖 YOLO 认出黑影"
```

---

### Task 4: 感知层 `mark_tried` / `lit` / `light_done` / 冷却（spec §4）

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 3 的 `_flame`、`_lighting`、`_cooldown_until`、`_person_at`、`_people_boxes`、`FRAME_STALE`
- Produces（身体用）：
  - `mark_tried(clue_id: int) -> None`
  - `lit(clue_id: int, since: float) -> bool | None`：True 点亮了、False 还在等、None 他走了
  - `light_done(result: str) -> None`，result ∈ `"lit" | "gone" | "timeout" | "interrupted" | "dry-run" | "exit" | "failed"`
  - `_lighting` 的键：`id`、`raised`、`pos`、`r`、`black0`、`flame_last`、`person`（(Rect, black) | None，最近一次找到的人）、`person_at`（最近一次找到人的帧时间，初值 -inf）、`requested`（出请求的时间，没出过 = raised）

- [ ] **Step 1: 写失败测试**（接在 Task 3 的测试后面）

```python
def lit_setup(monkeypatch, flames, black_seq=(0.9,)):
    """团子身边火焰够久、出请求，身体举蜡烛（mark_tried）；black() 依次返回 black_seq（用完了一直返回最后一个）。"""
    w, det, clock = light_watcher(monkeypatch, flames)
    seq = list(black_seq)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v: seq.pop(0) if len(seq) > 1 else seq[0])
    det.frames = [[me(), unlit(1000)]]
    run_frames(w, clock, 0.0, 3.1)
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    return w, det, clock, cid


def test_mark_tried_removes_request_and_blocks_new_ones(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    assert LIGHT_KEY not in w.requests and w._lighting["black0"] == 0.9
    run_frames(w, clock, 3.2, 5.0)
    assert LIGHT_KEY not in w.requests


def test_lit_false_before_lit_min(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v: 0.1)
    det.frames = [[me(), player(1000)]]
    run_frames(w, clock, 3.2, 4.5)
    assert w.lit(cid, 3.1) is False  # 火焰没了、人也亮了，但举起才 1.4 秒
    run_frames(w, clock, 4.6, 5.3)
    assert w.lit(cid, 3.1) is True


def test_lit_false_while_flame_still_there(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    run_frames(w, clock, 3.2, 6.0)
    assert w.lit(cid, 3.1) is False


def test_lit_true_when_flame_gone_and_person_brightens(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME], black_seq=[0.9])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v: 0.1)
    det.frames = [[me(), player(1000)]]
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is True


def test_lit_false_when_flame_gone_but_still_black(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME], black_seq=[0.9])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is False


def test_lit_none_when_flame_gone_and_nobody_there(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[me()]]
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is None


def test_lit_low_score_person_counts(monkeypatch):
    """晚上黑影只有 0.27 的框：判结果时也看低分框，不然会误判"走了"。"""
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[me(), unlit(1000, score=0.27)]]
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is False


def test_lit_person_must_be_near_flame(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v: 0.1)
    det.frames = [[me(), player(400)]]  # 亮着的人在远处，不是他
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is None


def test_lit_without_black_baseline_uses_lit_black_only(monkeypatch):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]  # 举蜡烛时火焰下面没人：black0 = None
    run_frames(w, clock, 0.0, 3.1)
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    assert w._lighting["black0"] is None
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    monkeypatch.setattr(perception_mod, "black", lambda f, box, v: 0.1)
    det.frames = [[me(), player(1000)]]
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is True


def test_lit_false_while_paused_or_stale(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    monkeypatch.setattr(perception_mod, "find_flame", lambda *a: None)
    det.frames = [[me()]]
    run_frames(w, clock, 3.2, 5.3)
    assert w.lit(cid, 3.1) is None
    clock.t = 6.0  # 0.7 秒没跑 process
    assert w.lit(cid, 3.1) is False
    clock.t = 5.3
    w.hold("panel")
    assert w.lit(cid, 3.1) is False
    w.release("panel")


def test_lit_unknown_clue_is_false(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    assert w.lit(cid + 5, 3.1) is False


def test_light_done_not_lit_starts_cooldown(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("timeout")
    assert w._lighting is None
    run_frames(w, clock, 3.2, 10.0)
    assert LIGHT_KEY not in w.requests  # 火焰还在，但在冷却里
    run_frames(w, clock, 10.1, 63.6)  # 一直跑着（中间断开超过 DISK_GAP 会重新计时）
    assert LIGHT_KEY in w.requests  # 冷却过了，线索还接着（火焰一直在）


def test_light_done_lit_no_cooldown(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("lit")
    assert w._cooldown_until == float("-inf") and w._flame is None
    run_frames(w, clock, 3.2, 6.5)
    assert LIGHT_KEY in w.requests  # 又一个人（新线索）站够了


def test_light_done_twice_is_harmless(monkeypatch):
    w, det, clock, cid = lit_setup(monkeypatch, [FLAME])
    w.light_done("gone")
    w.light_done("exit")
    assert w._lighting is None
```

- [ ] **Step 2: 跑测试看失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py -q -k "lit or mark_tried or light_done"`
Expected: AttributeError（没有 `mark_tried` / `lit` / `light_done`）

- [ ] **Step 3: 实现。** `_watch_flames` 的 `with self._lock:` 块里，`lighting, cooling = ...` 之后、`if lighting is not None and clue ...` 之后加（每次找火焰时顺带更新"火焰下面那个人"；不找火焰的帧不更新，0.3 秒一次够用）：

```python
            if lighting is not None:  # 举着蜡烛：火焰最后的位置下面现在有没有人、多黑
                person = self._person_at(lighting["pos"], lighting["r"])
                if person is not None:
                    lighting["person"], lighting["person_at"] = (person, black(frame, person, cfg.lit_v)), now
```

注意 `_watch_flames` 开头 `me is None` 时也要走到这一段（没有团子框也要判他亮没亮）：把 `with self._lock:` 块整个放在 `if me is None` 分支之后（现在的结构已经这样，只要确认 `return` 只在节流处）。

在 `_on_request` 后加：

```python
    def mark_tried(self, clue_id: int) -> None:
        """身体举起蜡烛了：开始"点亮中"（期间不出新请求），记下这时火焰在哪、他有多黑。"""
        now = self.clock()
        with self._lock:
            clue = self._flame if self._flame is not None and self._flame["id"] == clue_id else None
            self._lighting = {
                "id": clue_id, "raised": now, "requested": now,
                "pos": clue["pos"] if clue else (0, 0), "r": clue["r"] if clue else 20.0,
                "black0": clue["black"] if clue else None,
                "flame_last": clue["last"] if clue else float("-inf"),
                "person": None, "person_at": float("-inf"),
            }
        self.requests.pop(LIGHT_KEY, None)

    def lit(self, clue_id: int, since: float) -> bool | None:
        """举蜡烛（since 时刻）之后：True 他亮了；False 还在等（火焰还在 / 还黑着 / 拿不准）；None 火焰没了、那里也没人（走了）。

        拿不准一律 False：感知暂停中、最近一帧比 FRAME_STALE 还旧、举起不满 lit_min 秒、不是现在点亮中的线索。
        火焰消失超过 DISK_GAP 后看它最后位置下面的人（含低分框）：这一帧找到、black() 比举蜡烛时降了 lit_drop 以上
        且低于 lit_black → True（举蜡烛时下面没人就只看 lit_black）；找到但还黑 → False；DISK_GAP 内都没找到 → None。"""
        cfg = self.light_cfg
        frame_at = self._frame_at
        if cfg is None or self.paused or self.clock() - frame_at > FRAME_STALE or frame_at - since < cfg.lit_min:
            return False
        with self._lock:
            L = self._lighting
            if L is None or L["id"] != clue_id or frame_at - L["flame_last"] <= DISK_GAP:
                return False
            person, person_at, black0 = L["person"], L["person_at"], L["black0"]
        if frame_at - person_at > DISK_GAP:
            return None
        if person_at < frame_at - DISK_EVERY - 1e-6:  # 最近一次找火焰没找到人：先等等
            return False
        blk = person[1]
        return blk < cfg.lit_black and (black0 is None or black0 - blk >= cfg.lit_drop)

    def light_done(self, result: str) -> None:
        """身体这次点亮结束了（lit / gone / timeout / interrupted / dry-run / exit / failed）：结束"点亮中"；
        没点亮就冷却 light_cooldown 秒（认不出是谁，只能按时间）。重复调用无害。"""
        cfg = self.light_cfg
        with self._lock:
            L, self._lighting = self._lighting, None
            if L is None:
                return
            self._flame = None  # 这条线索用过了：之后要重新连续看满 light_after 秒
            if result != "lit" and cfg is not None:
                self._cooldown_until = self.clock() + cfg.light_cooldown
        log.info("点亮陌生人结束：%s（线索 %d）", result, L["id"])
        self._light_finished(L, result)

    def _light_finished(self, lighting: dict, result: str) -> None:
        """写存图的 summary.json（Task 5）。"""
```

注意 `lit()` 里"最近一次找火焰没找到人"那行：`person_at` 是找火焰那一帧的时间，找火焰每 `DISK_EVERY` 一次，所以离最近一帧超过 `DISK_EVERY` 就是最近一次没找到。

- [ ] **Step 4: 跑测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py tests/test_candle.py -q`
Expected: 全过

- [ ] **Step 5: Commit**

```bash
git add src/skydango/vision/perception.py tests/test_perception.py
git commit -m "feat(perception): 举蜡烛后按火焰消失 + 原地的人变亮判点亮，没点亮冷却"
```

---

### Task 5: 存图留证据（spec §5）+ cli 接线

**Files:**
- Modify: `src/skydango/vision/perception.py`（构造参数 `light_dir`、`_on_request`、`_light_finished`、点亮中按间隔存图）
- Modify: `src/skydango/cli.py`（`_scene_watcher`）
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 3/4 的 `_scan`、`_flame`、`_lighting`
- Produces: `PerceptionWatcher(..., light_dir: Path | None = None)`；常量 `DIAG_EVERY = 0.5`、`DIAG_MAX = 30`、`DIAG_RUNS = 50`；目录 `light_dir/<HHMMSS>/`（同一秒重名加 `-2`），图 `<标签>-<+秒>.jpg`（标签 `request` / `raised`；秒 = 离举起，举起前为负，格式 `+01.5` / `-00.3`），`summary.json`

- [ ] **Step 1: 写失败测试**

```python
def test_diag_images_and_summary(monkeypatch, tmp_path):
    import json

    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[me(), unlit(1000)]]
    run_frames(w, clock, 0.0, 3.1)
    (d,) = list(tmp_path.iterdir())
    assert len(list(d.glob("request-*.jpg"))) == 1  # 出请求那一刻一张（dry-run 也有）
    cid = w.requests[LIGHT_KEY].track
    w.mark_tried(cid)
    run_frames(w, clock, 3.2, 5.2)
    raised = sorted(d.glob("raised-*.jpg"))
    assert 4 <= len(raised) <= 6  # 每 DIAG_EVERY 秒一张
    w.light_done("timeout")
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert s["clue"] == cid and s["result"] == "timeout" and s["black_raised"] is not None
    run_frames(w, clock, 5.3, 6.0)
    assert len(sorted(d.glob("raised-*.jpg"))) == len(raised)  # 结束后不再存


def test_diag_max_per_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(perception_mod, "DIAG_MAX", 3)
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[me()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    run_frames(w, clock, 3.2, 8.0)
    (d,) = list(tmp_path.iterdir())
    assert len(list(d.glob("*.jpg"))) == 3


def test_diag_runs_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(perception_mod, "DIAG_RUNS", 1)
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    w.light_dir = tmp_path
    det.frames = [[me()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    w.light_done("lit")
    run_frames(w, clock, 3.2, 6.5)  # 第二次出请求：超过 DIAG_RUNS 不再建目录
    assert LIGHT_KEY in w.requests and len(list(tmp_path.iterdir())) == 1


def test_no_light_dir_saves_nothing(monkeypatch, tmp_path):
    w, det, clock = light_watcher(monkeypatch, [FLAME])
    det.frames = [[me()]]
    run_frames(w, clock, 0.0, 3.1)
    w.mark_tried(w.requests[LIGHT_KEY].track)
    run_frames(w, clock, 3.2, 4.0)
    w.light_done("gone")  # 没有存图目录：不出错
```

再改 `test_scene_watcher_loads_flame_only_for_brain`：末尾加

```python
    from types import SimpleNamespace

    run = SimpleNamespace(path=tmp_path, hard=tmp_path / "hard")
    assert cli._scene_watcher(cfg, run, light=True).light_dir == tmp_path / "light"
    assert cli._scene_watcher(cfg, run).light_dir is None
```

（函数签名加 `tmp_path` 参数；`_scene_watcher(cfg, icons=None, dev=None, background=True, run=None, light=False)`，`run` 只能按关键字传；`cfg.perception.hardcases` 默认开着会建 `HardCaseCollector`，测试里先 `cfg.perception.hardcases = False`。）

- [ ] **Step 2: 跑测试看失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py -q -k "diag or scene_watcher"`
Expected: 失败（`light_dir` 不存在 / 没存图）

- [ ] **Step 3: 实现。**
  1. 常量：

```python
DIAG_EVERY = 0.5  # 点亮中每隔这么久存一张图（spec 2026-10-01-light-flame-around-self §5）
DIAG_MAX = 30  # 一次最多存这么多张
DIAG_RUNS = 50  # 一次运行最多存这么多次
```

  2. 构造参数 `light_dir: Path | None = None`（放在 `flame` 后面），`self.light_dir = light_dir`、`self._diag: dict | None = None`（当前这次存图：`dir`、`count`、`next`、`requested`）、`self._diag_runs = 0`
  3. 实现：

```python
    def _on_request(self, frame: np.ndarray, now: float) -> None:
        """第一次出请求：新建这次的存图目录、存一张（dry-run 不按键也存：看得出该不该举）。"""
        if self.light_dir is None or self._diag_runs >= DIAG_RUNS:
            self._diag = None
            return
        self._diag_runs += 1
        stamp = time.strftime("%H%M%S")
        path, k = self.light_dir / stamp, 1
        while path.exists():
            k += 1
            path = self.light_dir / f"{stamp}-{k}"
        self._diag = {"dir": path, "count": 0, "next": float("-inf"), "requested": now}
        self._save_light(frame, now, "request")

    def _save_light(self, frame: np.ndarray, now: float, label: str) -> None:
        """画上搜索范围（灰）、团子框（白）、火焰（青圈 + 分数）、判点亮找到的人（绿 = 不黑 / 紫 = 黑 + black 值）。"""
        diag = self._diag
        if diag is None or diag["count"] >= DIAG_MAX:
            return
        import cv2

        from ..imageio import imwrite

        img = frame.copy()
        scan = self._scan or {}
        if scan.get("area") is not None:
            a = scan["area"]
            cv2.rectangle(img, (a.x, a.y), (a.x2, a.y2), (160, 160, 160), 2)
        if scan.get("me") is not None:
            m = scan["me"]
            cv2.rectangle(img, (m.x, m.y), (m.x2, m.y2), (255, 255, 255), 2)
        f = scan.get("flame")
        if f is not None:
            cv2.circle(img, (f.x, f.y), round(f.r * 1.5), (255, 255, 0), 2)
            cv2.putText(img, f"{f.score:.2f}", (f.x + round(f.r * 1.5), f.y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        with self._lock:
            person = self._lighting["person"] if self._lighting is not None else None
            raised = self._lighting["raised"] if self._lighting is not None else None
        if person is not None:
            box, blk = person
            color = (200, 60, 200) if blk >= self.light_cfg.lit_black else (60, 220, 60)
            cv2.rectangle(img, (box.x, box.y), (box.x2, box.y2), color, 2)
            cv2.putText(img, f"{blk:.2f}", (box.x, box.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        offset = now - (raised if raised is not None else now)
        try:
            diag["dir"].mkdir(parents=True, exist_ok=True)
            imwrite(diag["dir"] / f"{label}-{offset:+05.1f}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            log.debug("存点亮图出错", exc_info=True)
        diag["count"] += 1
        diag["next"] = now + DIAG_EVERY

    def _light_finished(self, lighting: dict, result: str) -> None:
        """写这次的 summary.json：线索、出请求 / 举起时间、结果、举起时和最后的 black()、火焰最后看到的时间（都相对举起，秒）。"""
        diag, self._diag = self._diag, None
        if diag is None:
            return
        import json

        raised = lighting["raised"]
        rel = lambda t: None if t == float("-inf") else round(t - raised, 2)  # noqa: E731
        summary = {
            "clue": lighting["id"], "result": result, "images": diag["count"],
            "requested": rel(diag["requested"]), "raised": 0.0, "done": rel(self.clock()),
            "black_raised": lighting["black0"],
            "black_end": round(lighting["person"][1], 3) if lighting["person"] is not None else None,
            "flame_last": rel(lighting["flame_last"]),
        }
        try:
            diag["dir"].mkdir(parents=True, exist_ok=True)
            (diag["dir"] / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            log.debug("写点亮 summary 出错", exc_info=True)
```

  4. `_watch_flames`：在最后（出请求逻辑之前，`if not ready:` 之前）加点亮中按间隔存图：

```python
        if lighting is not None and self._diag is not None and now >= self._diag["next"]:
            self._save_light(frame, now, "raised")
```

  注意 `_watch_flames` 节流在 `DISK_EVERY`（0.3 s），所以实际间隔是 0.6 s 左右（≥ `DIAG_EVERY` 的第一次扫描）；测试 3.2~5.2 秒约 4 张，和断言 4~6 匹配。若实测不符，以"按 `DIAG_EVERY` 节流、不超过 `DIAG_MAX`"为准调整断言下限。
  5. `mark_tried`：在 `self.requests.pop` 前，如果 `self._diag is None and self.light_dir is not None`（没出过请求就举了：理论上不会）不处理；`self._diag` 存在就 `self._diag["next"] = float("-inf")`（举起后第一帧就存）。
  6. `cli._scene_watcher`：

```python
    want_light = light and cfg.social.enabled and "light" in cfg.social.accept_strangers
    detector = make_detector(p.model, p.classes, p.imgsz, detector_conf(p, want_light), p.iou, p.device)
    ...
    flame = None
    if want_light:
        from .vision.candle import load_flame

        flame = load_flame(cfg.social.flame)
    ...
        social_cfg=cfg.social, flame=flame, light_dir=run.path / "light" if want_light and run is not None else None,
```

- [ ] **Step 4: 跑测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_perception.py tests/test_candle.py -q`
Expected: 全过

- [ ] **Step 5: Commit**

```bash
git add src/skydango/vision/perception.py src/skydango/cli.py tests/test_perception.py
git commit -m "feat(perception): 每次点亮陌生人存图和 summary.json 到 runs/<…>/light/"
```

---

### Task 6: 身体：他走了 / 各种结局调 `light_done`（spec §4）

**Files:**
- Modify: `src/skydango/brain/body.py`（`_watch_light`、`_check_lit`、`_watch_requests` 里接受互动那段、`shutdown`）
- Test: `tests/test_brain_light_body.py`

**Interfaces:**
- Consumes: `env.mark_tried(clue_id)`、`env.lit(clue_id, since) -> bool | None`、`env.light_done(result)`（Task 4）

- [ ] **Step 1: 改假 env + 写失败测试。** `LightEnv`：

```python
class LightEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.tried = []
        self.done = []
        self.lit_result = False

    def mark_tried(self, clue_id):
        self.tried.append(clue_id)
        self.requests.pop(LIGHT_KEY, None)

    def lit(self, clue_id, since):
        return self.lit_result

    def light_done(self, result):
        self.done.append(result)
```

加测试：

```python
def test_gone_lowers_candle_without_bow(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = None
    clock.advance(2.5)
    b.step()
    assert presses(dev) == 2 and emotes.done == [] and env.done == ["gone"] and b._raised is None


def test_done_results(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    env.lit_result = True
    b.step()
    assert env.done == ["lit"]


def test_timeout_reports_done(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(8.1)
    b.step()
    assert env.done == ["timeout"]


def test_emote_while_raised_reports_interrupted(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    clock.advance(1.0)
    emotes.perform("挥手")
    b.step()
    assert env.done == ["interrupted"]


def test_accepting_other_interaction_reports_interrupted(clock):
    social = AllowAll()
    b, dev, env, emotes, events = lb(clock, social=social)
    offer(env, clock)
    b.step()
    env.requests["小明"] = Request("小明", "hand", (990, 400), clock())
    social.to_handle = ["小明:hand"]
    b.step()
    assert env.done == ["interrupted"]


def test_dry_run_reports_done_right_away(clock):
    b, dev, env, emotes, events = lb(clock, live=False)
    offer(env, clock)
    b.step()
    assert env.tried == [7] and env.done == ["dry-run"]


def test_shutdown_while_raised_reports_exit(clock):
    b, dev, env, emotes, events = lb(clock)
    offer(env, clock)
    b.step()
    b.shutdown()
    assert env.done == ["exit"]
```

并把已有的 `test_dry_run_only_logs` 断言改成 `assert presses(dev) == 0 and env.tried == [7] and env.done == ["dry-run"]`。

- [ ] **Step 2: 跑测试看失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_light_body.py -q`
Expected: 新测试失败（`env.done` 是空的；`lit()` 调用参数个数不对时 TypeError）

- [ ] **Step 3: 实现。** `body.py`：
  1. 加小工具：

```python
    def _light_done(self, result: str) -> None:
        """这次点亮结束了，告诉感知层（结束"点亮中"、写存图 summary、没点亮就冷却）。"""
        try:
            self.env.light_done(result)
        except Exception:
            log.exception("light_done 出错")
```

  2. `_watch_light`：`self.env.mark_tried(req.track)` 注释改"开始点亮（感知层停止出请求、开始存图）"；dry-run 分支 `return` 前加 `self._light_done("dry-run")`；按键失败的 `except` 里 `return` 前加 `self._light_done("failed")`；docstring 改成"团子身边冒火焰够久了：按 3 号键举蜡烛（绝不点火焰圆盘：点了会跟着他走），等感知层看他亮起来。"
  3. `_check_lit`：

```python
    def _check_lit(self, now: float) -> None:
        track, pos, raised_at = self._raised
        if self.emotes is not None and self.emotes.last_any > raised_at:  # 做动作已经把蜡烛放下了
            self._raised = None
            log.info("举蜡烛时做了别的动作，蜡烛已经放下，不再等他亮起来")
            self._light_done("interrupted")
            return
        result = self.env.lit(track, raised_at)
        if result:
            self._raised = None
            self._lit_at = now
            log.info("陌生人亮起来了（举蜡烛 %.1f 秒）", now - raised_at)
            self.events.put("accepted", "你举起蜡烛给身边一个没点火的陌生人点了火（他亮起来了）")
            self._light_done("lit")
            self._schedule_bow(now, raised_at)
        elif result is None:
            self._raised = None
            log.info("他走开了（举蜡烛 %.1f 秒），放下蜡烛", now - raised_at)
            self._lower_candle(raised_at)
            self._light_done("gone")
        elif now - raised_at >= self.cfg.social.light_timeout:
            self._raised = None
            log.warning("举了蜡烛 %.0f 秒他还是黑的，放下", self.cfg.social.light_timeout)
            self._lower_candle(raised_at)
            self._light_done("timeout")
```

  4. `_watch_requests`：`if handled and (self._raised is not None or ...)` 块里，`self._raised = None` 之前：`if self._raised is not None: self._light_done("interrupted")`（注意只在 `_raised` 不为 None 时报，只排着鞠躬的不报）。
  5. `shutdown`：`if self._raised is not None:` 块里 `self._raised = None` 后加 `self._light_done("exit")`。

- [ ] **Step 4: 跑测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_light_body.py tests/test_brain_body.py -q`
Expected: 全过

- [ ] **Step 5: Commit**

```bash
git add src/skydango/brain/body.py tests/test_brain_light_body.py
git commit -m "feat(body): 点亮陌生人时他走开了就放下蜡烛，各种结局都告诉感知层"
```

---

### Task 7: 文档、离线回放、全量测试

**Files:**
- Modify: `docs/game-ops.md` §6（点火 / 火焰那几段）、`CLAUDE.md`（「识别环境」里点火那段）
- Create: `tmp/lightflame/replay.py`（不进 git）

- [ ] **Step 1: 离线回放。** `tmp/lightflame/replay.py`：用 `models/sky-yolo-v7.onnx` 建 `PerceptionWatcher`（`social_cfg=SocialConfig()`、`flame=load_flame()`、`light_dir=tmp/lightflame/replay/light`、`background=False`、`clock` 用帧文件名里的时间），按时间顺序喂 `tmp/lightdbg/w_*.jpg`（文件名 `w_HHMMSS_序号.jpg`，同一秒内按序号均分时间）；每出一次请求打印时间和线索编号，并假装身体：出请求后 0.5 秒 `mark_tried`，之后每帧 `lit()`，到结果或 8 秒 `light_done`。列出：每次出请求的时间、离第一次看到火焰多久、结果；对照 spec §0 表里的三次（22:06、22:31、22:48~22:52），看是否每次都在 3~5 秒内出请求、有没有在别的时间误报
- [ ] **Step 2: 回放结果记进 `tmp/lightflame/results.md`**；有误报就看存下的图找原因，必要时回 Task 3 调范围 / 排除规则（改了要补测试）
- [ ] **Step 3: game-ops §6** 改：火焰那一行的"处理"列和下面的核对说明换成新做法——在团子框周围找（范围、数字来自 Task 1）、不管 YOLO 认没认出人、好友标签下 / 面板下的排除、线索连接（`light_jump`）、出请求条件；判点亮改成火焰消失 + 原地的人 `black()` 变低（Task 1 的数字），没人 = 走了；冷却 60 秒；存图位置；删掉旧的 `lit_frames` / `lit_iou` / `lit_stale` / 替身那段描述，保留录像 c / d 的事实（点亮动画时间、彩虹爱心不代表点上）
- [ ] **Step 4: CLAUDE.md**「识别环境」里"没点火的黑影站到团子身边……"那一大句改成新做法（短一些）：大脑模式下，团子框周围（左右 `light_area_x`、往上 `light_area_up` 倍框高）连续 `light_after` 秒冒着火焰（不管 YOLO 认没认出这个人；认出名字的好友标签下、聊天面板下的不算）→ 身体按 3 举蜡烛（绝不点圆盘）；火焰消失后原地的人 `black()` 变低算点亮、鞠躬，原地没人算走了、放下，`light_timeout` 没结果放下；没点亮冷却 `light_cooldown` 秒；每次存图到 `runs/<…>/light/`；**未在真机验证**（spec `docs/superpowers/specs/2026-10-01-light-flame-around-self-design.md` §11）。「运行目录」表加一行 `light/<时间>/`
- [ ] **Step 5: 全量测试**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全过（记下通过数）

- [ ] **Step 6: Commit、合并、推送**

```bash
git add docs/game-ops.md CLAUDE.md docs/superpowers/plans/2026-10-01-light-flame-around-self.md
git commit -m "docs: 点亮陌生人改成在团子周围找火焰（game-ops、CLAUDE.md）"
git checkout main
git merge --no-ff feat/light-flame -m "Merge branch 'feat/light-flame'：点亮陌生人改成在团子周围找火焰"
git push origin main
```
