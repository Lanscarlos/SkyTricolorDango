# 装扮图鉴第 1 期：运行时收集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 团子运行时（`run` 的 dry-run 和 live）把近处的人清楚的整身裁图存进本机 `catalog/inbox/<日期>/<运行>/<身份>/`，每个身份每次运行留最好的几张，配一份索引；外加离线工具 `catalog collect` 在录像上跑同一套逻辑、定门槛。

**Architecture:** 新模块 `vision/catalog.py` 的 `CatalogCollector` 只管"门槛 → 挑图 → 写盘"，身份由感知层通过 `judge` 回调给（`Who`）。`PerceptionWatcher.process()` 在一帧处理完后调 `update` / `dropped`，`stop()` 调 `close()`。`cli._scene_watcher` 在有运行目录时建收集器；离线工具复用 `_perception` 和 `track-eval` 的读录像方式。

**Tech Stack:** Python 3.13、numpy、OpenCV（`cv2.Laplacian` 算清晰度）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-02-catalog-collect-design.md`

## Global Constraints

- 收集只截图存盘：**不发任何输入、不说话、不调模型**；任何写盘错误只记 WARNING，不抛出、不影响感知
- `[catalog] enabled = true` 默认开；要配合 `[perception] enabled`，没开时不建收集器、不提示
- `catalog = None`（或 `[catalog] enabled = false`）时感知层行为**逐字照旧**
- 收集器只读轨迹，**不改任何 `track.data`**
- 目录在第一次真的写图时才建（没收到图的运行不留空目录）
- 索引 `kind` 一律 `"outfit"`；文件路径相对 `catalog/`（`cfg.catalog.dir`），用 `/` 分隔
- 中文路径一律用 `skydango.imageio.imwrite` / `imread`（cv2 原生的不支持中文路径）；JPEG 质量 90
- 测试命令：`.venv\Scripts\python.exe -m pytest -q`（本机系统 Python 也能跑：`python -m pytest -q`）
- 回复 / 注释 / 提交信息用中文；提交信息末尾带 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/skydango/config.py` | `CatalogConfig` 数据类、`Config.catalog` |
| `config.example.toml` | `[catalog]` 一节 |
| `src/skydango/console/settings.py` | 设置页 `catalog.enabled` 开关 |
| `.gitignore` | `catalog/` |
| `src/skydango/vision/appearance.py` | 拆出 `clear_box`（`good_crop` 改为调用它） |
| `src/skydango/vision/catalog.py`（新） | `Who`、`Shot`、`sharpness`、`padded`、`stranger_key`、`CatalogCollector`、`contact_sheet` |
| `src/skydango/vision/perception.py` | `catalog` 参数、`_catalog_who`、在 `process()` / `stop()` 里调用 |
| `src/skydango/cli.py` | `_catalog_collector`、`_scene_watcher` 接线、`_stop_scene` 打印、`catalog collect` 子命令 |
| `CLAUDE.md` | 代码结构表一行、一节说明、常用命令 |
| `tests/test_catalog.py`（新） | 收集器单元测试（Task 3、4、7 的 `contact_sheet`） |
| `tests/test_perception_catalog.py`（新） | 感知层接线（Task 5） |
| `tests/test_cli_catalog.py`（新） | cli 接线和离线工具（Task 6、7） |

---

### Task 1: 配置 `[catalog]`、设置页开关、gitignore

**Files:**
- Modify: `src/skydango/config.py`（`AttrsConfig` 之后加 `CatalogConfig`；`Config` 里 `attrs` 之后加一行）
- Modify: `config.example.toml`（`[attrs]` 一节之后加 `[catalog]`）
- Modify: `src/skydango/console/settings.py:68`（`appearance.describe` 那一行之后）
- Modify: `.gitignore`
- Test: `tests/test_config.py`、`tests/test_console_settings.py`

**Interfaces:**
- Produces: `skydango.config.CatalogConfig`（字段见下），`Config.catalog`

- [ ] **Step 1: 写失败的测试**

`tests/test_config.py` 末尾加：

```python
def test_catalog_defaults_and_example():
    from skydango.config import CatalogConfig

    c = CatalogConfig()
    assert c.enabled is True and c.dir == "catalog"
    assert (c.every, c.min_height, c.edge, c.pad, c.sharp_min) == (0.5, 0.25, 8, 0.1, 50.0)
    assert (c.per_who, c.gap, c.flush_every, c.max_per_run) == (6, 3.0, 300.0, 300)
    ex = load_config(ROOT / "config.example.toml").catalog
    assert ex == CatalogConfig()
```

`tests/test_console_settings.py` 的 `test_every_spec_field_is_listed` 里，`"appearance.describe",` 后面插入 `"catalog.enabled",`（列表其余不动），并在文件里加：

```python
def test_console_has_catalog_switch():
    from skydango.console.settings import KNOWN

    f = KNOWN["catalog.enabled"]
    assert (f.label, f.kind, f.group) == ("收集装扮图鉴", "bool", "features")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_config.py::test_catalog_defaults_and_example tests/test_console_settings.py -k "catalog or every_spec"`
Expected: FAIL（`ImportError: cannot import name 'CatalogConfig'`、`KeyError: 'catalog.enabled'`）

- [ ] **Step 3: 实现**

`src/skydango/config.py`，`AttrsConfig` 类后面加：

```python
@dataclass
class CatalogConfig:
    """装扮图鉴第 1 期（spec 2026-10-02-catalog-collect）：运行时把近处的人清楚的整身裁图存进 catalog/inbox/。

    只截图存盘、不发输入，所以默认开；要配合 [perception]。门槛是估的，用 `catalog collect` 在录像上定。"""

    enabled: bool = True
    dir: str = "catalog"  # 存储目录（相对当前目录，gitignore）
    every: float = 0.5  # 每条轨迹多久看一次（秒）
    min_height: float = 0.25  # 框高至少占画面多少（估的）
    edge: int = 8  # 框离画面四边至少几像素（人没被画面切掉）
    pad: float = 0.1  # 裁图四周放宽的比例
    sharp_min: float = 50.0  # 清晰度门槛：裁图缩到高 256 后拉普拉斯方差（估的）
    per_who: int = 6  # 每个身份每次运行最多存几张
    gap: float = 3.0  # 留下的图两两至少隔几秒
    flush_every: float = 300.0  # 多久写一次盘（秒）
    max_per_run: int = 300  # 每次运行最多存几张
```

`Config` 里 `attrs: AttrsConfig = field(default_factory=AttrsConfig)` 后面加：

```python
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
```

`config.example.toml`，`[attrs]` 一节（到 `max_errors` 那一行）之后加：

```toml
[catalog]
# 装扮图鉴第 1 期：运行时把近处的人清楚的整身裁图存进 catalog/inbox/<日期>/<运行>/<身份>/（只截图存盘、不发输入）。
# 要配合 [perception]；门槛是估的，用 `python -m skydango catalog collect <录像目录>` 在录像上定
enabled = true
dir = "catalog"                 # 存储目录（gitignore）
every = 0.5                     # 每条轨迹多久看一次（秒）
min_height = 0.25               # 框高至少占画面多少
edge = 8                        # 框离画面四边至少几像素
pad = 0.1                       # 裁图四周放宽的比例
sharp_min = 50.0                # 清晰度门槛（裁图缩到高 256 后的拉普拉斯方差）
per_who = 6                     # 每个身份每次运行最多几张
gap = 3.0                       # 留下的图两两至少隔几秒
flush_every = 300.0             # 多久写一次盘（秒）
max_per_run = 300               # 每次运行最多几张
```

`src/skydango/console/settings.py`，`Field("appearance.describe", ...)` 那一行后面加：

```python
    Field("catalog.enabled", "收集装扮图鉴", "近处的人清楚的整身裁图存进 catalog/inbox/（只存图、不发输入）；要配合 YOLO 感知层", "bool", "features"),
```

`.gitignore`：在 `datasets/` 那一行后面加一行 `catalog/`。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_config.py tests/test_console_settings.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py config.example.toml src/skydango/console/settings.py .gitignore tests/test_config.py tests/test_console_settings.py
git commit -m "feat(catalog): [catalog] 配置、设置页开关、gitignore

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 认装扮拆出 `clear_box`

**Files:**
- Modify: `src/skydango/vision/appearance.py:34-53`（`good_crop`）
- Test: `tests/test_appearance.py`（已有的认装扮测试文件；没有就新建 `tests/test_appearance_clear_box.py`）

**Interfaces:**
- Produces: `skydango.vision.appearance.clear_box(box: Rect, others: list[Rect], blocked: list[Rect], max_overlap: float) -> bool`

- [ ] **Step 1: 写失败的测试**

```python
from skydango.vision.appearance import clear_box
from skydango.vision.bubbles import Rect


def test_clear_box():
    box = Rect(100, 100, 100, 200)
    assert clear_box(box, [], [], 0.2)
    assert clear_box(box, [Rect(500, 100, 100, 200)], [], 0.2)  # 离得远
    assert not clear_box(box, [Rect(120, 100, 100, 200)], [], 0.2)  # 和别人重叠
    assert not clear_box(box, [Rect(150, 150, 30, 40)], [], 0.0)  # 小框压在身上，门槛 0 也不行
    assert not clear_box(box, [], [Rect(0, 0, 160, 1080)], 0.2)  # 压在聊天面板上
    assert clear_box(box, [], [Rect(0, 0, 105, 1080)], 0.2)  # 面板只碰到一点边
    assert not clear_box(Rect(0, 0, 0, 10), [], [], 0.2)  # 空框
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q -k clear_box`
Expected: FAIL（`ImportError: cannot import name 'clear_box'`）

- [ ] **Step 3: 实现**

`appearance.py` 里，`good_crop` 前面加 `clear_box`，`good_crop` 改成调用它（返回值和以前逐字一样）：

```python
def clear_box(box: Rect, others: list[Rect], blocked: list[Rect], max_overlap: float) -> bool:
    """框没被别的人 / 团子盖住、没压在 blocked（聊天面板）上：认装扮的好样本和图鉴收集共用。空框算不行。"""
    area = box.w * box.h
    if box.w <= 0 or box.h <= 0:
        return False
    for o in others:
        if iou(box, o) > max_overlap or _inter(box, o) / area > max_overlap:
            return False
    for b in blocked:
        if _inter(box, b) / area > max_overlap:
            return False
    return True


def good_crop(
    frame: np.ndarray, box: Rect, others: list[Rect], blocked: list[Rect], min_height: float, max_overlap: float
) -> np.ndarray | None:
    """好样本才返回裁图（框中间 60% 宽、整高），否则 None：框太小、被别的人 / 团子盖住、压在聊天面板上都不要。"""
    fh, fw = frame.shape[:2]
    if box.w <= 0 or box.h <= 0 or box.h < min_height * fh:
        return None
    if not clear_box(box, others, blocked, max_overlap):
        return None
    margin = int(round(box.w * 0.2))
    crop = Rect(box.x + margin, box.y, box.w - 2 * margin, box.h)
    x1, y1, x2, y2 = max(0, crop.x), max(0, crop.y), min(fw, crop.x2), min(fh, crop.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
```

（`iou`、`_inter` 在 `appearance.py` 里已经有：`_inter` 定义在 `good_crop` 上方，`iou` 已从 `.track` 导入；动手前确认一下导入行。）

- [ ] **Step 4: 跑测试确认通过（含原有认装扮测试）**

Run: `python -m pytest -q tests/ -k "appearance or clear_box"`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/appearance.py tests/
git commit -m "refactor(appearance): 拆出 clear_box（好样本的遮挡判断），图鉴收集要共用

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 收集器——门槛和挑图（内存里）

**Files:**
- Create: `src/skydango/vision/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `CatalogConfig`（Task 1）、`clear_box`（Task 2）、`skydango.vision.track.Track`、`skydango.vision.bubbles.Rect`
- Produces（Task 4~7 用到）：
  - `Who(name: str, kind: str, sure: bool, maybe: str | None = None)`（frozen dataclass；`kind` ∈ `friend` / `self` / `stranger`）
  - `Shot(crop: np.ndarray, row: dict, score: float, t: float)`
  - `sharpness(crop: np.ndarray) -> float`、`padded(box: Rect, pad: float, width: int, height: int) -> Rect`、`stranger_key(track_id: int) -> str`（`"陌生人-t<id>"`）
  - `CatalogCollector(cfg, root: Path, run_name: str, day: str, *, wall=time.time, trace: bool = False)`
    - `update(frame, players: list[Track], selfs: list[Track], now: float, panel: Rect | None, place: str, judge: Callable[[Track], Who | None]) -> None`
    - `buffer(key: str) -> list[Shot]`（测试用，拷贝）
    - `total: int`（property）、`candidates: list[dict]`（`trace=True` 时每个看过的候选一行）、`folder: Path`（`root/inbox/<day>/<run_name>`）

- [ ] **Step 1: 写失败的测试**

`tests/test_catalog.py`：

```python
"""图鉴收集（spec 2026-10-02-catalog-collect）：门槛、挑图、写盘。"""

import cv2
import numpy as np
import pytest

from skydango.config import CatalogConfig
from skydango.vision import catalog as cat
from skydango.vision.bubbles import Rect
from skydango.vision.catalog import CatalogCollector, Who, sharpness, stranger_key
from skydango.vision.track import Track

H, W = 1080, 1920


def frame(v=100):
    """纯色画面：测挑图时把 sharpness 换成"裁图平均亮度"，用亮度控制清晰度。"""
    return np.full((H, W, 3), v, np.uint8)


def track(tid, x=800, y=300, w=200, h=400, cls="player"):
    return Track(tid, cls, Rect(x, y, w, h), 0.9, 0.0, 0.0)


def friend(t):
    return Who("懒洋洋大王", "friend", True)


def stranger(t):
    return Who(stranger_key(t.id), "stranger", False)


def collector(tmp_path, trace=True, **kw):
    cfg = CatalogConfig(**{"sharp_min": 10.0, **kw})
    return CatalogCollector(cfg, tmp_path / "catalog", "20261002-210000-dry-brain", "2026-10-02",
                            wall=lambda: 1700000000.0, trace=trace)


@pytest.fixture
def brightness(monkeypatch):
    monkeypatch.setattr(cat, "sharpness", lambda crop: float(crop.mean()))


def test_sharpness_noise_vs_blur():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (400, 200, 3), dtype=np.uint8)
    assert sharpness(noise) > 1000
    assert sharpness(cv2.GaussianBlur(noise, (0, 0), 8)) < 50
    assert sharpness(np.full((400, 200, 3), 90, np.uint8)) == 0.0


def test_padded_clamps_to_frame():
    assert cat.padded(Rect(100, 100, 100, 200), 0.1, W, H) == Rect(90, 80, 120, 240)
    assert cat.padded(Rect(0, 0, 100, 200), 0.1, W, H) == Rect(0, 0, 110, 220)


def test_gates(tmp_path, brightness):
    c = collector(tmp_path)
    f = frame(100)
    c.update(f, [track(1, h=200)], [], 0.0, None, "", stranger)  # 太小：200 / 1080 < 0.25
    c.update(f, [track(2, x=2)], [], 0.0, None, "", stranger)  # 贴左边
    c.update(f, [track(3, y=H - 400)], [], 0.0, None, "", stranger)  # 贴下边
    c.update(f, [track(4), track(5, x=850)], [], 0.0, None, "", stranger)  # 两个人叠在一起
    c.update(f, [track(6, x=100)], [], 0.0, Rect(0, 0, 640, 920), "", stranger)  # 压在聊天面板上
    c.update(frame(5), [track(7)], [], 0.0, None, "", stranger)  # 模糊（亮度 5 < sharp_min 10）
    c.update(f, [track(8)], [], 0.0, None, "", lambda t: None)  # judge 说不收
    assert c.total == 0
    fails = {r["who"]: r["fail"] for r in c.candidates}
    assert fails == {"陌生人-t1": "small", "陌生人-t2": "edge", "陌生人-t3": "edge", "陌生人-t4": "blocked",
                     "陌生人-t5": "blocked", "陌生人-t6": "blocked", "陌生人-t7": "blurry"}
    c.update(f, [track(9)], [], 0.0, None, "", stranger)
    assert [s.row["who"] for s in c.buffer("陌生人-t9")] == ["陌生人-t9"]


def test_every_limits_how_often_a_track_is_looked_at(tmp_path, brightness):
    c = collector(tmp_path)
    for now in (0.0, 0.2, 0.4, 0.5, 0.7, 1.0):
        c.update(frame(), [track(1)], [], now, None, "", stranger)
    assert [r["t"] for r in c.candidates] == [0.0, 0.5, 1.0]


def test_keeps_at_most_per_who_spread_by_gap(tmp_path, brightness):
    c = collector(tmp_path, per_who=3, gap=3.0)
    for i, now in enumerate([0.0, 1.0, 3.0, 6.0, 9.0, 12.0]):
        c.update(frame(100 + i), [track(1)], [], now, None, "", friend)
    shots = c.buffer("懒洋洋大王")
    assert len(shots) == 3
    ts = sorted(s.t for s in shots)
    assert all(b - a >= 3.0 for a, b in zip(ts, ts[1:]))
    assert ts == [6.0, 9.0, 12.0]  # 越往后越亮 = 越清楚：留下最好的三张


def test_close_in_time_keeps_the_better_one(tmp_path, brightness):
    c = collector(tmp_path, gap=3.0)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(150), [track(1)], [], 1.0, None, "", friend)  # 1 秒后、更清楚：换掉
    c.update(frame(120), [track(1)], [], 2.0, None, "", friend)  # 不如 1.0 那张：丢掉
    (s,) = c.buffer("懒洋洋大王")
    assert s.t == 1.0 and s.row["sharp"] == 150.0


def test_one_new_shot_can_replace_two_close_ones(tmp_path, brightness):
    c = collector(tmp_path, gap=3.0)
    # 时间不按顺序来，所以每次换一条轨迹（同一条轨迹 every 秒内只看一次）；judge 都说是他
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(100), [track(2)], [], 3.0, None, "", friend)
    c.update(frame(200), [track(3)], [], 1.5, None, "", friend)  # 离两张都不到 3 秒、比两张都好
    assert [s.t for s in c.buffer("懒洋洋大王")] == [1.5]


def test_score_grows_with_height_until_half_frame(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1, x=300, y=100, h=300)], [], 0.0, None, "", stranger)
    c.update(frame(100), [track(2, x=1300, y=100, h=700)], [], 0.0, None, "", stranger)
    (a,), (b,) = c.buffer("陌生人-t1"), c.buffer("陌生人-t2")
    assert b.score == pytest.approx(100.0)  # 超过半个画面高不再加分
    assert a.score == pytest.approx(100.0 * (300 / H) / 0.5, rel=0.02)


def test_friend_keeps_one_buffer_across_tracks(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(), [track(2)], [], 5.0, None, "", friend)  # 轨迹断了换了编号，还是他
    assert len(c.buffer("懒洋洋大王")) == 2


def test_self_and_row_fields(tmp_path, brightness):
    c = collector(tmp_path)
    me = track(3, cls="self")
    c.update(frame(), [], [me], 2.0, None, "雨林", lambda t: Who("团子", "self", True))
    (s,) = c.buffer("团子")
    assert s.row == {
        "kind": "outfit", "who": "团子", "who_kind": "self", "sure": True, "maybe": None,
        "run": "20261002-210000-dry-brain", "t": 2.0, "wall": 1700000000.0, "box": [800, 300, 200, 400],
        "height": round(400 / H, 4), "sharp": 100.0, "score": round(100.0 * (400 / H) / 0.5, 1), "place": "雨林",
    }
    assert s.crop.shape == (480, 240, 3)  # 四周各放宽 10%


def test_maybe_is_recorded_on_stranger(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(4)], [], 0.0, None, "", lambda t: Who(stranger_key(t.id), "stranger", False, "懒洋洋大王"))
    assert c.buffer("陌生人-t4")[0].row["maybe"] == "懒洋洋大王"


def test_max_per_run_stops_growth_but_allows_replacement(tmp_path, brightness):
    c = collector(tmp_path, max_per_run=2, gap=3.0)
    c.update(frame(100), [track(1)], [], 0.0, None, "", stranger)
    c.update(frame(100), [track(1)], [], 5.0, None, "", stranger)
    c.update(frame(100), [track(2, x=1300)], [], 5.0, None, "", stranger)  # 满了：新身份开不了
    c.update(frame(100), [track(1)], [], 10.0, None, "", stranger)  # 满了：同一个人也不再加
    assert c.total == 2 and c.buffer("陌生人-t2") == []
    c.update(frame(200), [track(1)], [], 15.0, None, "", stranger)  # 但比最差的好：替换
    assert c.total == 2 and max(s.t for s in c.buffer("陌生人-t1")) == 15.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_catalog.py`
Expected: FAIL（`ModuleNotFoundError: No module named 'skydango.vision.catalog'`）

- [ ] **Step 3: 实现 `src/skydango/vision/catalog.py`**

```python
"""装扮图鉴第 1 期：运行时收集近处的人的清楚整身裁图（spec 2026-10-02-catalog-collect-design）。

感知层每帧调 update：人物框过了门槛（够大、完整、没被挡、清楚）就打分，每个身份留最好的 per_who 张、两两隔 gap 秒；
陌生人轨迹断了写出他那份，每 flush_every 秒、close() 时全部写出。身份由感知层的 judge 给（Who），这里不碰黑影 / 第二层的细节。
只截图存盘：不发输入、不调模型；写盘出错只记 WARNING。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import CatalogConfig
from ..imageio import imwrite
from .appearance import _folder_name, clear_box
from .bubbles import Rect
from .track import Track

log = logging.getLogger(__name__)

KIND = "outfit"  # 以后地图 / 先祖 / 物品用别的值
SHARP_HEIGHT = 256  # 算清晰度前把裁图缩到这么高（不同远近的人可比）
FULL_HEIGHT = 0.5  # 框高到画面一半就不再加分
MAX_OVERLAP = 0.2  # 同 [appearance] max_overlap 的默认值
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 90]


@dataclass(frozen=True)
class Who:
    """感知层判的身份。name 也是目录名：好友名 / "团子" / "陌生人-t<轨迹>"。"""

    name: str
    kind: str  # friend / self / stranger
    sure: bool  # 名字被名字标签证实
    maybe: str | None = None  # 按外观 / 位置接回的"像谁"


@dataclass
class Shot:
    crop: np.ndarray
    row: dict  # 索引那一行（写盘时再加 file）
    score: float
    t: float


def sharpness(crop: np.ndarray) -> float:
    """清晰度：转灰度、缩到高 SHARP_HEIGHT，拉普拉斯方差（越模糊越小）。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    gray = cv2.resize(gray, (max(1, round(w * SHARP_HEIGHT / h)), SHARP_HEIGHT), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def padded(box: Rect, pad: float, width: int, height: int) -> Rect:
    """框四周各放宽 pad 比例，夹到画面内。"""
    dx, dy = round(box.w * pad), round(box.h * pad)
    x1, y1 = max(0, box.x - dx), max(0, box.y - dy)
    x2, y2 = min(width, box.x2 + dx), min(height, box.y2 + dy)
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def stranger_key(track_id: int) -> str:
    return f"陌生人-t{track_id}"


class CatalogCollector:
    """线程安全（感知线程调 update / dropped，退出时别的线程调 close）。"""

    def __init__(self, cfg: CatalogConfig, root: Path, run_name: str, day: str, *,
                 wall: Callable[[], float] = time.time, trace: bool = False) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.folder = self.root / "inbox" / day / run_name
        self.run_name = run_name
        self.wall = wall
        self.trace = trace
        self.candidates: list[dict] = []  # trace 时每个看过的候选一行（离线工具定门槛用）
        self._buffers: dict[str, list[Shot]] = {}
        self._dirty: set[str] = set()
        self._written: dict[str, list[dict]] = {}  # 身份 → 最近一次写盘的索引行
        self._archived = 0  # 写出后清掉缓冲的陌生人一共几张
        self._seen: dict[int, float] = {}  # 轨迹 id → 上次看的时间
        self._last_flush: float | None = None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._archived + sum(len(b) for b in self._buffers.values())

    @property
    def saved(self) -> int:
        """现在盘上有几张。"""
        return sum(len(rows) for rows in self._written.values())

    def buffer(self, key: str) -> list[Shot]:
        return list(self._buffers.get(key, []))

    def update(self, frame: np.ndarray, players: list[Track], selfs: list[Track], now: float,
               panel: Rect | None, place: str, judge: Callable[[Track], Who | None]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._last_flush is None:
                self._last_flush = now
            people = list(players) + list(selfs)
            height, width = frame.shape[:2]
            for t in people:
                if now - self._seen.get(t.id, float("-inf")) < self.cfg.every:
                    continue
                self._seen[t.id] = now
                who = judge(t)
                if who is None:
                    continue
                others = [o.box for o in people if o.id != t.id]
                self._consider(frame, t.box, who, others, panel, place, now, width, height)
            if now - self._last_flush >= self.cfg.flush_every:
                self._flush(list(self._buffers))
                self._last_flush = now

    def _gate(self, box: Rect, others: list[Rect], panel: Rect | None, width: int, height: int) -> str | None:
        c = self.cfg
        if box.w <= 0 or box.h < c.min_height * height:
            return "small"
        if box.x < c.edge or box.y < c.edge or box.x2 > width - c.edge or box.y2 > height - c.edge:
            return "edge"
        if not clear_box(box, others, [panel] if panel is not None else [], MAX_OVERLAP):
            return "blocked"
        return None

    def _consider(self, frame: np.ndarray, box: Rect, who: Who, others: list[Rect], panel: Rect | None,
                  place: str, now: float, width: int, height: int) -> None:
        h = box.h / height
        fail = self._gate(box, others, panel, width, height)
        sharp = score = None
        crop = None
        if fail is None:
            r = padded(box, self.cfg.pad, width, height)
            crop = frame[r.y:r.y2, r.x:r.x2]
            sharp = sharpness(crop)
            score = sharp * min(h / FULL_HEIGHT, 1.0)
            if sharp < self.cfg.sharp_min:
                fail = "blurry"
        if self.trace:
            self.candidates.append({
                "t": round(now, 3), "who": who.name, "box": [box.x, box.y, box.w, box.h], "height": round(h, 4),
                "sharp": None if sharp is None else round(sharp, 1), "score": None if score is None else round(score, 1),
                "fail": fail,
            })
        if fail is not None:
            return
        row = {
            "kind": KIND, "who": who.name, "who_kind": who.kind, "sure": who.sure, "maybe": who.maybe,
            "run": self.run_name, "t": round(now, 3), "wall": round(self.wall(), 3), "box": [box.x, box.y, box.w, box.h],
            "height": round(h, 4), "sharp": round(sharp, 1), "score": round(score, 1), "place": place,
        }
        self._offer(who.name, Shot(crop.copy(), row, score, now))

    def _offer(self, key: str, shot: Shot) -> None:
        """放进 key 的缓冲：隔不够 gap 的只留更好的；满了换最差的；张数到 max_per_run 后只换不加。"""
        full = self.total >= self.cfg.max_per_run
        buf = self._buffers.get(key)
        if buf is None:
            if full:
                return
            buf = self._buffers[key] = []
        close = [s for s in buf if abs(s.t - shot.t) < self.cfg.gap]
        if close:
            if all(shot.score > s.score for s in close):
                for s in close:
                    buf.remove(s)
                buf.append(shot)
                self._dirty.add(key)
            return
        if len(buf) < self.cfg.per_who and not full:
            buf.append(shot)
            self._dirty.add(key)
            return
        worst = min(buf, key=lambda s: s.score)  # 缓冲满了，或者这次运行的张数到顶了：只换不加
        if shot.score > worst.score:
            buf.remove(worst)
            buf.append(shot)
            self._dirty.add(key)

    def _flush(self, keys: Iterable[str]) -> None:
        """写盘（Task 4 补上）。这一个 Task 先什么都不做：测试只看内存里的缓冲。"""
        return
```

注意：`_flush` 在这一步是空方法（Task 4 补上），这一个 Task 的测试不依赖写盘。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_catalog.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): 图鉴收集器的门槛和挑图（够大、完整、没被挡、清楚；每人留最好的几张、两两隔开）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: 收集器——写盘、陌生人走开、收尾

**Files:**
- Modify: `src/skydango/vision/catalog.py`（补 `_flush`、加 `_write`、`_write_index`、`dropped`、`close`）
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: Task 3 的 `CatalogCollector`、`Shot`、`stranger_key`
- Produces: `CatalogCollector.dropped(tracks: Iterable[Track]) -> None`、`CatalogCollector.close() -> None`、`CatalogCollector.saved`（盘上几张）；
  磁盘布局 `<root>/inbox/<day>/<run>/<_folder_name(身份)>/<名次>.jpg` + `<root>/inbox/<day>/<run>/index.jsonl`

- [ ] **Step 1: 写失败的测试**

`tests/test_catalog.py` 末尾加：

```python
import json

from skydango.imageio import imread


def rows(c):
    return [json.loads(line) for line in (c.folder / "index.jsonl").read_text(encoding="utf-8").splitlines()]


def test_close_writes_files_and_index(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(150), [track(1)], [], 5.0, None, "", friend)
    c.close()
    folder = tmp_path / "catalog" / "inbox" / "2026-10-02" / "20261002-210000-dry-brain"
    assert c.folder == folder
    assert sorted(p.name for p in (folder / "懒洋洋大王").iterdir()) == ["1.jpg", "2.jpg"]
    assert imread(folder / "懒洋洋大王" / "1.jpg").mean() == pytest.approx(150, abs=2)  # 名次按得分
    rs = rows(c)
    assert [r["file"] for r in rs] == [
        "inbox/2026-10-02/20261002-210000-dry-brain/懒洋洋大王/1.jpg",
        "inbox/2026-10-02/20261002-210000-dry-brain/懒洋洋大王/2.jpg",
    ]
    assert rs[0]["kind"] == "outfit" and rs[0]["who"] == "懒洋洋大王" and rs[0]["t"] == 5.0
    assert c.saved == 2


def test_nothing_collected_leaves_no_folder(tmp_path):
    c = collector(tmp_path)
    c.update(frame(), [], [], 0.0, None, "", friend)
    c.close()
    assert not (tmp_path / "catalog").exists()


def test_dropped_stranger_is_written_and_cleared(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(), [track(7)], [], 0.0, None, "", stranger)
    c.dropped([track(7), track(99)])  # 99 没收过：不管
    assert (c.folder / "陌生人-t7" / "1.jpg").exists()
    assert c.buffer("陌生人-t7") == [] and c.total == 1  # 清掉了，但还算在这次运行的张数里
    c.update(frame(), [track(8, x=1300)], [], 1.0, None, "", stranger)
    c.close()
    assert [r["who"] for r in rows(c)] == ["陌生人-t7", "陌生人-t8"]  # 清掉的那份还在索引里


def test_periodic_flush_and_shrink_removes_extra_file(tmp_path, brightness):
    c = collector(tmp_path, flush_every=10.0, gap=3.0)
    # 时间不按顺序来，每次换一条轨迹（同一条轨迹 every 秒内只看一次）
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(100), [track(2)], [], 3.0, None, "", friend)
    c.update(frame(100), [track(3)], [], 10.0, None, "", friend)  # 到 10 秒：写一次盘
    assert len(list((c.folder / "懒洋洋大王").iterdir())) == 3
    c.update(frame(250), [track(4)], [], 11.5, None, "", friend)  # 离 10.0 不到 3 秒、更好：换掉 10.0 那张
    c.update(frame(255), [track(5)], [], 1.5, None, "", friend)  # 离 0.0 和 3.0 都不到 3 秒、都更好：两张换一张
    c.close()
    assert sorted(p.name for p in (c.folder / "懒洋洋大王").iterdir()) == ["1.jpg", "2.jpg"]
    assert len(rows(c)) == 2


def test_write_error_is_logged_not_raised(tmp_path, brightness, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(cat, "imwrite", boom)
    c = collector(tmp_path)
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    with caplog.at_level("WARNING"):
        c.close()
    assert "图鉴收集写盘出错" in caplog.text


def test_closed_collector_ignores_updates(tmp_path, brightness):
    c = collector(tmp_path)
    c.close()
    c.update(frame(), [track(1)], [], 0.0, None, "", friend)
    c.dropped([track(1)])
    assert c.total == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_catalog.py`
Expected: 新加的 6 个 FAIL（`AttributeError: ... 'close'` 等）

- [ ] **Step 3: 实现**

把 Task 3 留空的 `_flush` 换成下面这一组方法（放在 `_offer` 后面）：

```python
    def _write(self, key: str) -> None:
        """把 key 的缓冲按得分写成 1.jpg、2.jpg……；缓冲比上次少就删掉多出来的旧文件。"""
        buf = sorted(self._buffers.get(key, []), key=lambda s: -s.score)
        folder = self.folder / _folder_name(key)
        out = []
        for i, s in enumerate(buf, 1):
            path = folder / f"{i}.jpg"
            imwrite(path, s.crop, JPEG)
            out.append({"file": path.relative_to(self.root).as_posix(), **s.row})
        for i in range(len(buf) + 1, max(self.cfg.per_who, len(self._written.get(key, []))) + 1):
            (folder / f"{i}.jpg").unlink(missing_ok=True)
        self._written[key] = out
        self._dirty.discard(key)

    def _write_index(self) -> None:
        """索引整份重写（先写临时文件再替换），和目录里的图一一对应。"""
        lines = [json.dumps(r, ensure_ascii=False) + "\n" for k in sorted(self._written) for r in self._written[k]]
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / "index.jsonl.tmp"
        tmp.write_text("".join(lines), encoding="utf-8")
        os.replace(tmp, self.folder / "index.jsonl")

    def _flush(self, keys: Iterable[str]) -> None:
        keys = [k for k in keys if k in self._dirty]
        if not keys:
            return
        try:
            for k in keys:
                self._write(k)
            self._write_index()
        except OSError as exc:
            log.warning("图鉴收集写盘出错：%s", exc)

    def dropped(self, tracks: Iterable[Track]) -> None:
        """追踪器删掉的轨迹：陌生人那份写出去、清掉缓冲（写失败就留着，下次再写）。"""
        with self._lock:
            if self._closed:
                return
            for t in tracks:
                self._seen.pop(t.id, None)
                key = stranger_key(t.id)
                if key not in self._buffers:
                    continue
                self._flush([key])
                if key not in self._dirty:
                    self._archived += len(self._buffers.pop(key))

    def close(self) -> None:
        """全部写出；之后的 update / dropped 都不理。"""
        with self._lock:
            if self._closed:
                return
            self._flush(list(self._buffers))
            self._closed = True
            if self.saved:
                log.info("图鉴收集：存了 %d 张 → %s", self.saved, self.folder)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_catalog.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): 图鉴收集写盘——陌生人走开写出、定时写、收尾全部写，索引整份重写

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 感知层接上收集器

**Files:**
- Modify: `src/skydango/vision/perception.py`（`__init__` 加参数；`process()` 在难例收集之后、`self.timings.append` 之前调用；`stop()`；加 `_catalog_who`）
- Test: `tests/test_perception_catalog.py`

**Interfaces:**
- Consumes: `Who`、`stranger_key`（Task 3）；`CatalogCollector.update / dropped / close`（Task 3、4）
- Produces: `PerceptionWatcher(..., catalog=None)`、`PerceptionWatcher.catalog` 属性（离线工具会直接赋值）、`PerceptionWatcher._catalog_who(t: Track) -> Who | None`

- [ ] **Step 1: 写失败的测试**

`tests/test_perception_catalog.py`：

```python
"""感知层接图鉴收集（spec 2026-10-02-catalog-collect §4）。"""

from test_perception import FakeDetector, FakeOcr, frame, player, tag, watcher

from skydango.vision.bubbles import Rect
from skydango.vision.catalog import Who
from skydango.vision.detect import Detection
from skydango.vision.track import Track


class FakeCatalog:
    def __init__(self):
        self.updates, self.drops, self.closed = [], [], 0

    def update(self, frame, players, selfs, now, panel, place, judge):
        self.updates.append((now, [judge(t) for t in players], [judge(t) for t in selfs], panel, place))

    def dropped(self, tracks):
        self.drops += [t.id for t in tracks]

    def close(self):
        self.closed += 1


def test_judge_friend_stranger_unlit_self():
    w = watcher(FakeDetector())
    friend = Track(1, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"tagged": True, "name": "懒洋洋大王"})
    lit = Track(2, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"maybe": "懒洋洋大王", "maybe_by": "relink"})
    unlit = Track(3, "player_unlit", Rect(0, 0, 10, 10), 0.9, 0, 0)
    me = Track(4, "self", Rect(0, 0, 10, 10), 0.9, 0, 0)
    unknown = Track(5, "player", Rect(0, 0, 10, 10), 0.9, 0, 0, data={"tagged": True})  # 有标签但不在好友名单
    assert w._catalog_who(friend) == Who("懒洋洋大王", "friend", True)
    assert w._catalog_who(lit) == Who("陌生人-t2", "stranger", False, "懒洋洋大王")
    assert w._catalog_who(unlit) is None
    assert w._catalog_who(me) == Who("团子", "self", True)
    assert w._catalog_who(unknown) == Who("陌生人-t5", "stranger", False)


def test_process_calls_update_with_frame_people():
    det = FakeDetector()
    det.frames = [[player(1000), tag(990, 110), Detection("self", Rect(1500, 400, 90, 220), 0.9)]]
    w = watcher(det, ocr=FakeOcr({110: "懒洋洋大王"}))
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, True)
    (now, players, selfs, panel, place), = w.catalog.updates
    assert now == 0.0 and players == [Who("懒洋洋大王", "friend", True)] and selfs == [Who("团子", "self", True)]
    assert panel is not None  # 面板开着：给面板区域
    w.process(frame(), 0.1, False)
    assert w.catalog.updates[-1][3] is None  # 面板关着：None


def test_paused_skips_update_and_dropped_is_forwarded():
    det = FakeDetector()
    det.frames = [[player(1000)], []]
    w = watcher(det, track_buffer=0.5)
    w.catalog = FakeCatalog()
    w.process(frame(), 0.0, False)
    w.hold("test")
    w.process(frame(), 0.1, False)
    assert len(w.catalog.updates) == 1  # 暂停时不收
    w.release("test")
    w.process(frame(), 2.0, False)  # 人不见了、过了 track_buffer：轨迹删掉
    assert w.catalog.drops


def test_stop_closes_catalog():
    w = watcher(FakeDetector())
    w.catalog = FakeCatalog()
    w.stop()
    assert w.catalog.closed == 1


def test_no_catalog_is_fine():
    det = FakeDetector()
    det.frames = [[player(1000)]]
    w = watcher(det)
    assert w.catalog is None
    w.process(frame(), 0.0, False)
    w.stop()
```

（`watcher()` 辅助函数把关键字参数传给 `PerceptionConfig`，`track_buffer` 是它的字段；`hold` / `release` 是感知层已有的暂停接口。
朋友那条依赖第一帧就读到名字标签——和 `test_perception.py::test_friend_is_recognized_by_reading_the_name_tag_once` 同样的布局；如果第一帧还没挂上名字，就多 `process` 一帧再断言。）

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_perception_catalog.py`
Expected: FAIL（`AttributeError: 'PerceptionWatcher' object has no attribute '_catalog_who'` / `catalog`）

- [ ] **Step 3: 实现**

`perception.py` 顶部导入区加：

```python
from .catalog import Who, stranger_key
```

`__init__` 参数表最后（`attrs=None,` 之后）加：

```python
        catalog=None,  # vision.catalog.CatalogCollector：图鉴收集（None = 不收，行为照旧）
```

`__init__` 方法体里（`self.saver = saver` 附近）加：

```python
        self.catalog = catalog
```

`process()` 里，难例那段 `try: self.hardcases.check(...)` 的 `except` 之后、`self.timings.append(...)` 之前加：

```python
        if self.catalog is not None:
            try:
                if self.tracker.dropped:
                    self.catalog.dropped(self.tracker.dropped)
                if not self.paused:
                    panel = roi_rect(self.log_roi, width, height) if panel_visible else None
                    self.catalog.update(frame, players, selfs, now, panel, self.place, self._catalog_who)
            except Exception:
                log.exception("图鉴收集出错")
```

`stop()` 改成：

```python
    def stop(self) -> None:
        self._stop.set()
        if self.catalog is not None:
            self.catalog.close()
```

在 `_admitted` 之后加：

```python
    def _catalog_who(self, t: Track) -> Who | None:
        """图鉴收集的身份（spec 2026-10-02-catalog-collect §2）：团子 / 名字标签证实的好友 / 其余点过火的按陌生人；黑影不收。
        process() 交给收集器的 players 已经去掉了第二层没放行和先祖 / 共享空间的。"""
        if t.cls == "self":
            return Who("团子", "self", True)
        if self._unlit(t):
            return None
        d = t.data
        if d.get("tagged") and d.get("name"):
            return Who(d["name"], "friend", True)
        return Who(stranger_key(t.id), "stranger", False, d.get("maybe"))
```

- [ ] **Step 4: 跑测试确认通过（含感知层原有测试）**

Run: `python -m pytest -q tests/test_perception_catalog.py tests/test_perception.py tests/test_perception_attrs.py tests/test_trackeval.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/perception.py tests/test_perception_catalog.py
git commit -m "feat(perception): 接上图鉴收集（每帧交给收集器、轨迹删掉转告、停下时收尾；身份由感知层判）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: cli 接线（`run` 时建收集器、退出时报告）

**Files:**
- Modify: `src/skydango/cli.py`（`_appearance_parts` 后面加 `_catalog_collector`；`_scene_watcher` 里 `PerceptionWatcher(...)` 加 `catalog=`；`_stop_scene` 打印）
- Test: `tests/test_cli_catalog.py`

**Interfaces:**
- Consumes: `CatalogCollector`（Task 3、4）、`PerceptionWatcher(catalog=)`（Task 5）、`RunDir`（`run.path.name` 是运行目录名）
- Produces: `cli._catalog_collector(cfg: Config, run: RunDir | None) -> CatalogCollector | None`

- [ ] **Step 1: 写失败的测试**

`tests/test_cli_catalog.py`：

```python
"""图鉴收集的接线和离线工具（spec 2026-10-02-catalog-collect §4、§6）。"""

import time
from pathlib import Path

from test_cli_appearance import fake_models

from skydango import cli
from skydango.config import Config
from skydango.runlog import RunDir
from skydango.vision.catalog import CatalogCollector


def cfg_for(tmp_path):
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.catalog.dir = str(tmp_path / "catalog")
    cfg.perception.enabled = True
    return cfg


def test_scene_watcher_attaches_collector_only_with_run(monkeypatch, tmp_path):
    fake_models(monkeypatch)
    cfg = cfg_for(tmp_path)
    run = RunDir.create(cfg, "dry")
    w = cli._scene_watcher(cfg, run=run)
    assert isinstance(w.catalog, CatalogCollector)
    assert w.catalog.folder == Path(cfg.catalog.dir) / "inbox" / time.strftime("%Y-%m-%d") / run.path.name
    assert cli._scene_watcher(cfg).catalog is None  # 没有运行目录（view 等）：不收
    cfg.catalog.enabled = False
    assert cli._scene_watcher(cfg, run=run).catalog is None
    w.stop()


def test_stop_scene_reports_saved(capsys, tmp_path):
    class Env:
        hardcases = None

        def __init__(self, catalog):
            self.catalog = catalog

        def stop(self):
            pass

    class Saved:
        saved = 3
        folder = tmp_path / "x"

    cli._stop_scene(Env(Saved()))
    assert "图鉴收集：存了 3 张" in capsys.readouterr().out
    cli._stop_scene(Env(None))  # 没收集器：不报错
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_cli_catalog.py`
Expected: FAIL（`w.catalog` 是 None；`_stop_scene` 没打印"图鉴收集：存了 3 张"）

- [ ] **Step 3: 实现**

`cli.py`，`_appearance_parts` 函数后面加：

```python
def _catalog_collector(cfg: Config, run: RunDir | None):
    """装扮图鉴收集（spec 2026-10-02-catalog-collect §4）：[catalog] 开着、有运行目录（run 的 dry-run / live）才建；
    view、perception detect 没有运行目录，不收。[perception] 没开时 _scene_watcher 根本不走到这里。"""
    if not cfg.catalog.enabled or run is None:
        return None
    from .vision.catalog import CatalogCollector

    collector = CatalogCollector(cfg.catalog, Path(cfg.catalog.dir), run.path.name, time.strftime("%Y-%m-%d"))
    log.info("图鉴收集：近处的人清楚的裁图存进 %s", collector.folder)
    return collector
```

`_scene_watcher` 里 `return PerceptionWatcher(` 那一段，最后一行 `call_window=cfg.call.window, camera_settle=cfg.track.settle, attrs=attrs,` 改成：

```python
        call_window=cfg.call.window, camera_settle=cfg.track.settle, attrs=attrs,
        catalog=_catalog_collector(cfg, run),
```

`_stop_scene` 末尾加：

```python
    catalog = getattr(env, "catalog", None)
    if catalog is not None and catalog.saved:
        print(f"图鉴收集：存了 {catalog.saved} 张 → {catalog.folder}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_cli_catalog.py tests/test_cli_appearance.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/cli.py tests/test_cli_catalog.py
git commit -m "feat(cli): run 时建图鉴收集器，退出时报告存了几张

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 离线工具 `catalog collect` 和总览拼图

**Files:**
- Modify: `src/skydango/vision/catalog.py`（加 `contact_sheet`）
- Modify: `src/skydango/cli.py`（`cmd_catalog`；`places` 子命令注册之后注册 `catalog`）
- Test: `tests/test_catalog.py`、`tests/test_cli_catalog.py`

**Interfaces:**
- Consumes: `CatalogCollector(..., wall=, trace=True)`、`candidates`、`folder`、`close()`（经 `watcher.stop()`）；`cli._perception`、`cli._images`、`cli._panel_open`、`vision.compare.timed_files`、`vision.trackeval.subsample`
- Produces: `contact_sheet(folder: Path, cell: int = 240) -> tuple[np.ndarray, list[str]] | None`（拼图 + 每一行是谁）；命令 `python -m skydango catalog collect <录像目录> [--model] [--device] [--imgsz] [--fps 6.5] [-o 输出目录]`

- [ ] **Step 1: 写失败的测试**

`tests/test_catalog.py` 末尾加：

```python
def test_contact_sheet_one_row_per_identity(tmp_path, brightness):
    c = collector(tmp_path)
    c.update(frame(100), [track(1)], [], 0.0, None, "", friend)
    c.update(frame(120), [track(1)], [], 5.0, None, "", friend)
    c.update(frame(140), [track(2, x=1300)], [], 0.0, None, "", stranger)
    c.close()
    sheet, legend = cat.contact_sheet(c.folder, cell=100)
    assert legend == ["懒洋洋大王（2 张）", "陌生人-t2（1 张）"]
    assert sheet.shape == (2 * 100, 2 * 100, 3)  # 2 行；最多的一行 2 张
    assert cat.contact_sheet(tmp_path / "nothing") is None
```

`tests/test_cli_catalog.py` 末尾加：

```python
def test_cli_catalog_collect_on_recording(tmp_path, monkeypatch):
    import json

    import numpy as np
    from conftest import FakeOcr

    from skydango.imageio import imwrite
    from skydango.vision.bubbles import Rect
    from skydango.vision.detect import Detection

    class BigStranger:
        providers = ["Fake"]
        imgsz = 960

        def detect(self, img):
            return [Detection("player", Rect(800, 300, 200, 400), 0.9)]

    rng = np.random.default_rng(0)
    src = tmp_path / "rec-1"
    src.mkdir()
    for i in range(20):
        imwrite(src / f"{i:04d}_{i * 0.5:06.2f}s.jpg", rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))
    monkeypatch.setattr("skydango.vision.detect.make_detector", lambda *a, **k: BigStranger())
    monkeypatch.setattr("skydango.vision.ocr.make_ocr", lambda *a, **k: FakeOcr())
    monkeypatch.setattr(cli, "_icon_classifier", lambda cfg: None)
    monkeypatch.setattr(cli, "_friend_names", lambda cfg: (lambda: []))
    monkeypatch.setattr(cli, "_person_attrs", lambda cfg: None)
    out = tmp_path / "out"
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "catalog", "collect", str(src), "--fps", "2", "-o", str(out)])
    folder = out / "inbox" / "offline" / "rec-1"
    index = [json.loads(x) for x in (folder / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert index and all(r["who"].startswith("陌生人-t") for r in index)
    cands = [json.loads(x) for x in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(cands) >= len(index) and {"sharp", "height", "fail"} <= set(cands[0])
    assert (out / "sheet.jpg").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest -q tests/test_catalog.py::test_contact_sheet_one_row_per_identity tests/test_cli_catalog.py::test_cli_catalog_collect_on_recording`
Expected: FAIL（`AttributeError: module ... has no attribute 'contact_sheet'`；cli 报 `invalid choice: 'catalog'`）

- [ ] **Step 3: 实现**

`catalog.py` 末尾加（需要在顶部导入 `from ..imageio import imread, imwrite`，把原来的 `from ..imageio import imwrite` 改掉）：

```python
def contact_sheet(folder: Path, cell: int = 240) -> tuple[np.ndarray, list[str]] | None:
    """一次运行存下的图拼成一张：每个身份一行、按名次排，每格 cell×cell（等比缩放、灰底补齐），左上角标清晰度。
    返回 (拼图, 每一行是谁)；没有索引 / 没有图返回 None。中文名字 cv2 画不了，所以名字放在 legend 里由调用方打印。"""
    index = Path(folder) / "index.jsonl"
    if not index.exists():
        return None
    groups: dict[str, list[dict]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        groups.setdefault(r["who"], []).append(r)
    if not groups:
        return None
    root = Path(folder).parents[2]  # <root>/inbox/<day>/<run>
    cols = max(len(rs) for rs in groups.values())
    sheet = np.full((cell * len(groups), cell * cols, 3), 40, np.uint8)
    legend = []
    for row, (who, rs) in enumerate(groups.items()):
        legend.append(f"{who}（{len(rs)} 张）")
        for col, r in enumerate(rs):
            img = imread(root / r["file"])
            scale = min(cell / img.shape[0], cell / img.shape[1])
            small = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
            y, x = row * cell, col * cell
            sheet[y:y + small.shape[0], x:x + small.shape[1]] = small
            cv2.putText(sheet, f"s{r['sharp']:.0f} h{r['height']:.2f}", (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return sheet, legend
```

`cli.py`，`cmd_places` 后面加：

```python
def cmd_catalog(cfg: Config, args) -> None:
    """图鉴收集的离线工具（spec 2026-10-02-catalog-collect §6）：录像上跑感知层 + 收集器（录像时间当时钟），
    存下的图和运行时同样的结构，另写 candidates.jsonl（每个看过的候选过没过门槛）和 sheet.jpg，用来定 min_height / sharp_min。"""
    import json
    from collections import Counter

    from .vision.catalog import CatalogCollector, contact_sheet
    from .vision.compare import timed_files
    from .vision.trackeval import subsample

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    out = Path(args.output or f"tmp/catalog/{time.strftime('%Y%m%d-%H%M%S')}")
    _, watcher = _perception(cfg, args)
    now = [0.0]
    watcher.clock = lambda: now[0]
    collector = CatalogCollector(cfg.catalog, out, Path(args.source).name, "offline", wall=lambda: now[0], trace=True)
    watcher.catalog = collector
    kept = list(subsample(timed, args.fps))
    print(f"{len(timed)} 帧里按 {args.fps:g} 帧 / 秒抽了 {len(kept)} 帧 → {out}")
    for n, (t, path) in enumerate(kept, 1):
        now[0] = t
        frame = imread(path)
        watcher.process(frame, t, _panel_open(cfg, frame))
        if n % 100 == 0:
            print(f"  {n}/{len(kept)}")
    watcher.stop()  # 收集器全部写出
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in collector.candidates), encoding="utf-8")
    fails = Counter(r["fail"] or "收下" for r in collector.candidates)
    print(f"看了 {len(collector.candidates)} 个候选：" + "、".join(f"{k} {v}" for k, v in fails.most_common()))
    made = contact_sheet(collector.folder)
    if made is None:
        print("一张都没存下（看 candidates.jsonl 是哪条门槛挡住的）")
        return
    sheet, legend = made
    imwrite(out / "sheet.jpg", sheet)
    print(f"存了 {collector.saved} 张 → {collector.folder}；总览 {out / 'sheet.jpg'}")
    for i, line in enumerate(legend, 1):
        print(f"  第 {i} 行：{line}")
```

（`imread` / `imwrite` 在 `cli.py` 顶部已经从 `.imageio` 导入；动手前确认一下，没有就加上。）

`places` 子命令注册（`p.set_defaults(func=cmd_places)`）之后加：

```python
    p = sub.add_parser("catalog", help="装扮图鉴：在录像上试跑收集（定门槛用）")
    psub = p.add_subparsers(dest="action", required=True)
    q = psub.add_parser("collect", help="录像上跑感知层 + 图鉴收集器，输出存下的图、candidates.jsonl、sheet.jpg")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("--imgsz", type=int)
    q.add_argument("--fps", type=float, default=6.5, help="按录像时间每秒抽几帧（模拟 run 时身体截图，默认 6.5）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/catalog/<时间>）")
    p.set_defaults(func=cmd_catalog)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest -q tests/test_catalog.py tests/test_cli_catalog.py`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/catalog.py src/skydango/cli.py tests/test_catalog.py tests/test_cli_catalog.py
git commit -m "feat(catalog): catalog collect 离线工具——录像上试跑收集，出候选明细和总览拼图

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: 文档、全量测试、在真录像上试跑

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/progress/2026-09-28-yolo-training.md`（加一小节试跑结果）

**Interfaces:**
- Consumes: 前面所有 Task

- [ ] **Step 1: 改 `CLAUDE.md`**

1. 「代码结构」表里，`src/skydango/vision/appearance.py` 那一行后面加一行：

```markdown
| `src/skydango/vision/catalog.py` | 装扮图鉴第 1 期（见「装扮图鉴」）：运行时把近处的人清楚的整身裁图存进 `catalog/inbox/`（门槛、每人留最好几张、写盘、索引）、离线工具的总览拼图 |
```

2. 「认装扮」一节之后加一节：

```markdown
## 装扮图鉴（`[catalog]`，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-02-catalog-collect-design.md`（开头有五期路线图：收集 → 归类 + 部位 + 管理面板起名 → 游戏里主动问 → 用图鉴认 → 地图 / 先祖 / 物品），计划 `docs/superpowers/plans/2026-10-02-catalog-collect.md`。
目标是让团子认出好友身上发型 / 斗篷 / 面具的**俗称**：图鉴从团子自己的经历里长出来，认法是 DINOv2 特征检索 + Claude 对图确认（新加单品不用重训）。**这一期只做收集，还没在真机上跑过，门槛是估的**。
- `run`（dry-run 和 live 都算）时感知层每帧把人物框交给 `CatalogCollector`：够大（`min_height`）、不贴边、没被别人 / 聊天面板挡（和认装扮共用 `clear_box`）、清楚（拉普拉斯方差 ≥ `sharp_min`）才收；
  好友（名字标签证实的）、团子、点过火的陌生人收，黑影 / 先祖 / 共享空间 / 第二层没放行的不收；"像小明"的按陌生人存、索引记 `maybe`
- 每个身份每次运行留最好的 `per_who` 张、两两隔 `gap` 秒；陌生人轨迹断了写出、每 `flush_every` 秒和退出时全部写出；存进 `catalog/inbox/<日期>/<运行>/<身份>/<名次>.jpg` + 同目录 `index.jsonl`（`catalog/` 不进 git，只在本机；第 2 期起完名的精选图才进私有仓库）
- 和认装扮（`[appearance]`）完全分开，不带它的副作用；`enabled = false` 时感知层逐字照旧；管理面板有 `catalog.enabled`
- 定门槛：`catalog collect <录像目录>` → `tmp/catalog/<时间>/`（`sheet.jpg` 总览、`candidates.jsonl` 每个候选过没过哪条门槛）
```

3. 「常用命令」代码块里，`perception appearance-eval` 那一行后面加：

```bash
python -m skydango catalog collect <录像目录> [--model 模型] [--fps 6.5]  # 装扮图鉴：录像上试跑收集 → tmp/catalog/<时间>/（sheet.jpg、candidates.jsonl），定 min_height / sharp_min
```

- [ ] **Step 2: 全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS（失败的话先修，不许跳过）

- [ ] **Step 3: 在真录像上试跑（本机有 GPU 和 `models/sky-yolo-v7.pt`、`tmp/record/` 时）**

```bash
python -m skydango catalog collect tmp/record/q-call-20260930-c -o tmp/catalog/q-call-c
python -m skydango catalog collect tmp/record/gesture-bow-1 -o tmp/catalog/bow-1
python -m skydango catalog collect tmp/record/play-1001-1 -o tmp/catalog/play-1
python -m skydango catalog collect tmp/record/candle-20260930-b -o tmp/catalog/candle-b
```

每段看终端里"看了 N 个候选：small …、edge …、blurry …、收下 …"和 `sheet.jpg`。用 `candidates.jsonl` 统计：

```bash
python - <<'EOF'
import json, glob, statistics as st
for f in sorted(glob.glob("tmp/catalog/*/candidates.jsonl")):
    rs = [json.loads(x) for x in open(f, encoding="utf-8")]
    sharp = sorted(r["sharp"] for r in rs if r["sharp"] is not None)
    hs = sorted(r["height"] for r in rs)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    print(f, len(rs), "sharp p10/p50/p90", q(sharp, .1), q(sharp, .5), q(sharp, .9), "height p50/p90", q(hs, .5), q(hs, .9))
EOF
```

把结果（每段收下几张、各门槛挡了多少、清晰度分布、看 `sheet.jpg` 有没有糊的 / 半个人 / 不是人的）写进 `docs/progress/2026-09-28-yolo-training.md` 末尾新的一节「装扮图鉴收集试跑（10-02）」。
**默认门槛不在这一步改**：如果明显不合适（比如一张都收不下、或者糊的大量被收），在进度文档里写建议值，交给用户决定。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md docs/progress/2026-09-28-yolo-training.md
git commit -m "docs: 装扮图鉴第 1 期写进 CLAUDE.md；录像上试跑收集的结果

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
