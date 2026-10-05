# 认交互图标（路线图 ②b）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO 的 `social_ring` 扩成通用图标框；没名字的圆圈按几何关系判归属，只有「下面是人」的才可能成为互动请求；先祖 / 物件 / 地图上的圆圈用模板或 DINOv2 最近邻认种类，写进 status、眼睛和管理面板画面，**只认不点**；再补离线工具和收件箱的数据入口。

**Architecture:** 纯计算放新模块 `vision/icons_map.py`（归属、叫法、投票、裁图、DINOv2 底库、`MapIcons` 运行时）；`PerceptionWatcher.process` 里把「没名字的圈」先判归属再分流（防误点规则不受开关管）；`icons()` 和 `objects()` 平行给身体 / 眼睛；离线工具 `vision/icon_eval.py`；收件箱加 `add` / `--redo` / 整图火焰预标。

**Tech Stack:** Python 3.13、OpenCV、numpy、onnxruntime（DINOv2-small，`vision/embed.py` 的 `OnnxEmbedder`）、pytest；前端 `console/static/stage.js`（node 可 require）。

**Spec:** `docs/superpowers/specs/2026-10-05-icon-detection-design.md`

## Global Constraints

- 只认不点：任何代码路径都不能让「归属不是 person」的没名字圆圈进 `PerceptionWatcher.requests` / `circles`；这条**不受 `[icons] enabled` 管**
- `candle.py` / `lighting.py` / `_watch_flames` 一行不动（②c 的事）；好友圈（`_tag_above` 挂上的）路径一行不动
- `[icons] enabled` 代码默认 `false`；`false` 时除防误点外逐字照旧（不报图标、不攒裁图、提示词 / status / overlay 不变）
- `[perception] enabled = false` 时整个不生效（`EnvWatcher` 不加任何东西）
- YOLO 类别表不改（`social_ring` 仍是 2 号，`PerceptionConfig.classes` 不动）
- 数字：`dino_match` 0.75、`dino_margin` 0.05、`vote` 5、`min_hits` 3、`under_x` 0.25、`under_up` 1.0、`save_max` 200、裁图边长 = 框长边 × 1.3、模板尺度 (0.9, 1.0, 1.1)、每条轨迹 2 秒最多存一张裁图
- 测试跑 `.venv\Scripts\python.exe -m pytest -q`（worktree 里先设 `PYTHONPATH=src`，见 CLAUDE.md「环境」）；全量测试放后台
- 中文注释 / 日志 / 文案；颜色只在 `stage.js` 的 `COLORS` 里加

## Review Focus

1. **地图上的火焰图标 + 白圈、下面没人**（灯、篝火点燃）：绝不能变成陌生人 `candle` 请求——Task 3 有测试
2. **图标框贴着画面边、裁图越界**：`icon_crop` 要裁到画面里、不抛异常、空裁图返回 `unknown`——Task 1 有测试
3. **DINOv2 模型不存在 / 推理出错**：`classifier = "dino"` 时启动警告、退回模板，运行中出错不拖垮感知线程——Task 2、Task 4 有测试
4. **参考图目录为空或某种只有 1 张**：底库照样能建、只有一种时 margin 按 0 处理——Task 2 有测试
5. **聊天面板开着**：面板区域里的圆圈本来就被 `PANEL_DROP` 丢掉，不能在 `icons()` 里冒出来——Task 4 有测试

---

### Task 1: `[icons]` 配置和 `icons_map` 的纯计算

**Files:**
- Create: `src/skydango/vision/icons_map.py`
- Modify: `src/skydango/config.py`（新增 `IconsConfig`，`Config` 里加 `icons: IconsConfig = field(default_factory=IconsConfig)`，放在 `catalog` 后面）
- Modify: `config.example.toml`（`[catalog]` 之后加 `[icons]` 一节，每项一行注释，照 spec §3.6 的说明）
- Modify: `src/skydango/console/settings.py`（加 `Field("icons.enabled", "认地图交互图标", "YOLO 框出的图标认出种类、写进状态和画面（只认不点）；要配合 YOLO 感知层", "bool", "features")`，紧挨 `catalog.enabled`）
- Modify: `src/skydango/game/social.py`（`KIND_NAMES` 后加 `MAP_KINDS: frozenset[str] = frozenset()` 和一行注释：真机看完图标再往里加，见 spec §1）
- Test: `tests/test_icons_map.py`

**Interfaces:**
- Produces:
  - `IconsConfig` 字段：`enabled: bool = False`、`classifier: str = "template"`、`dino: str = "models/dinov2-small.onnx"`、`refs: str = "assets/icons"`、`dino_match: float = 0.75`、`dino_margin: float = 0.05`、`vote: int = 5`、`min_hits: int = 3`、`under_x: float = 0.25`、`under_up: float = 1.0`、`crop_scale: float = 1.3`、`flame_ring: float = 2.0`（Task 7 量了再改）、`save: bool = True`、`save_max: int = 200`；`classifier` 不是 `template` / `dino` 时加载配置报错（照 config.py 现有的校验写法）
  - `MAP_OWNERS: tuple[str, ...] = ("spirit", "bonfire", "bench", "instrument", "map")`、`UNKNOWN = "unknown"`
  - `@dataclass(frozen=True) class Icon: track_id: int; kind: str; owner: str; label: str; box: Rect; side: str`
  - `owner_of(ring: Rect, people: list[Rect], spirits: list[Rect], things: list[tuple[str, Rect]], under_x: float, under_up: float) -> str`：返回 `"person"` / `"spirit"` / 物件类别名 / `"map"`（好友在调用方用 `_tag_above` 先判，不进这里）
  - `under_gap(ring: Rect, box: Rect, under_x: float, under_up: float) -> float | None`：圈心 x 在 `[box.x − under_x·w, box.x2 + under_x·w]`、圈心 y 在 `[box.y − under_up·h, box.y + 0.5·h]` 时返回 `abs(圈心y − box.y)`，否则 None
  - `map_label(kind: str | None, owner: str) -> str`
  - `vote(history: list[str]) -> str | None`
  - `icon_crop(frame: np.ndarray, box: Rect, scale: float) -> np.ndarray`
  - `describe_icons(icons: list[Icon]) -> str`

- [ ] **Step 1: 写失败的测试**（`tests/test_icons_map.py`）

```python
from skydango.vision.track import Rect
from skydango.vision.icons_map import (UNKNOWN, Icon, describe_icons, icon_crop, map_label, owner_of, under_gap, vote)

RING = Rect(950, 300, 100, 100)  # 圈心 (1000, 350)

def test_owner_order_person_beats_objects():
    person = Rect(955, 380, 90, 220)
    bench = Rect(900, 420, 200, 120)
    assert owner_of(RING, [person], [], [("bench", bench)], 0.25, 1.0) == "person"

def test_owner_spirit_then_object_then_map():
    sp = Rect(955, 380, 90, 220)
    assert owner_of(RING, [], [sp], [], 0.25, 1.0) == "spirit"
    assert owner_of(RING, [], [], [("bonfire", Rect(940, 400, 120, 100))], 0.25, 1.0) == "bonfire"
    assert owner_of(RING, [], [], [], 0.25, 1.0) == "map"

def test_owner_far_box_is_map():
    assert owner_of(RING, [Rect(1400, 380, 90, 220)], [], [], 0.25, 1.0) == "map"   # 横着差太远
    assert owner_of(RING, [Rect(955, 700, 90, 100)], [], [], 0.25, 1.0) == "map"    # 圈比人高出一个多身高

def test_owner_nearest_of_two_people():
    near, far = Rect(955, 360, 90, 220), Rect(955, 420, 90, 220)
    assert under_gap(RING, near, 0.25, 1.0) < under_gap(RING, far, 0.25, 1.0)

def test_map_label():
    assert map_label("candle", "bonfire") == "篝火点燃"
    assert map_label("candle", "map") == "可以点的蜡烛 / 灯"
    assert map_label(None, "map") == "不认识的图标" and map_label(UNKNOWN, "spirit") == "不认识的图标"
    assert map_label("hand", "map") == "牵手"  # 其余查 KIND_NAMES

def test_vote():
    assert vote(["candle", UNKNOWN, "candle", "sit"]) == "candle"
    assert vote([UNKNOWN, UNKNOWN]) == UNKNOWN
    assert vote([]) is None

def test_icon_crop_square_and_clipped():
    import numpy as np
    f = np.zeros((1080, 1920, 3), np.uint8)
    assert icon_crop(f, Rect(900, 300, 100, 80), 1.3).shape[:2] == (130, 130)
    edge = icon_crop(f, Rect(1880, 1050, 60, 60), 1.3)   # 贴边：只裁画面里的部分
    assert edge.shape[0] <= 78 and edge.shape[1] <= 78 and edge.size > 0
    assert icon_crop(f, Rect(3000, 3000, 50, 50), 1.3).size == 0   # 完全在外面：空

def test_describe_icons():
    b = Rect(0, 0, 10, 10)
    icons = [Icon(1, "sit", "map", "坐下", b, "右边"), Icon(2, UNKNOWN, "map", "不认识的图标", b, "左边"),
             Icon(3, "memory", "map", "留影", b, "左边")]
    assert describe_icons(icons) == "坐下（右边）、留影（左边）、不认识的图标 1 个"
    assert describe_icons([]) == ""
```

另加 `tests/test_config.py`（已有就追加）：`IconsConfig().enabled is False`、`classifier = "foo"` 加载报错。

- [ ] **Step 2: 跑测试确认失败** — `.venv\Scripts\python.exe -m pytest tests/test_icons_map.py -q`，预期 ImportError
- [ ] **Step 3: 实现** `config.py` 的 `IconsConfig` 和上面 Interfaces 里的函数。`map_label`：`None` / `UNKNOWN` → 「不认识的图标」；`_OVERRIDES = {("candle", "bonfire"): "篝火点燃", ("candle", "map"): "可以点的蜡烛 / 灯"}`；其余 `KIND_NAMES.get(kind, kind)`。`vote` 只数非 `UNKNOWN` 的，平票取最近一次出现的。`describe_icons` 认得的按输入顺序、不认识的合成一项放最后
- [ ] **Step 4: 跑测试确认通过**（同 Step 2 命令 + `tests/test_config.py`）
- [ ] **Step 5: 提交** — `git commit -m "feat(icons): [icons] 配置和图标归属 / 叫法 / 投票 / 裁图的纯计算"`

---

### Task 2: 认种类的两种分类器

**Files:**
- Modify: `src/skydango/game/social.py`（`IconClassifier.classify(region, scales=SCALES)` 加可选参数，默认行为不变）
- Modify: `src/skydango/vision/icons_map.py`
- Test: `tests/test_icons_map.py`

**Interfaces:**
- Consumes: Task 1 的 `icon_crop`、`UNKNOWN`
- Produces:
  - `TEMPLATE_SCALES = (0.9, 1.0, 1.1)`、`RING_PX = 100`（模板按近处 1080p 圆圈直径截）
  - `classify_template(icons: IconClassifier, crop: np.ndarray, box: Rect) -> tuple[str, float]`：把裁图按 `RING_PX / max(box.w, box.h)` 缩放后交 `icons.classify(region, TEMPLATE_SCALES)`；认不出 / 空裁图返回 `(UNKNOWN, 分数)`
  - `class IconGallery`：`__init__(self, embedder, refs: dict[str, list[np.ndarray]], match: float, margin: float)`（embedder 有 `embed(img) -> np.ndarray | None`，即 `appearance.DinoGuard`）；`@classmethod load(cls, directory: str | Path, embedder, match, margin) -> "IconGallery"`（`<directory>/<kind>/*.jpg`，`_` 开头的目录跳过）；`kinds: list[str]`；`classify(self, crop: np.ndarray) -> tuple[str, float]`
  - 判法：每种取和参考图余弦最大的一张；最好的 ≥ match 且（只有一种时）或（领先第二种 ≥ margin）才算，否则 `(UNKNOWN, 最好分)`；`embed` 返回 None（出错）→ `(UNKNOWN, 0.0)`；参考图特征在 `__init__` 时提一次

- [ ] **Step 1: 写失败的测试**：
  - `test_classifier_scales_param_default_unchanged`：`IconClassifier.classify(region)` 和 `classify(region, SCALES)` 结果一样（用 `tests/test_social.py` 里现有的合成图标造法）
  - `test_classify_template_rescales_by_box`：把一张模板画进 200×200 的圈（框 200）→ 认对；空裁图 → `(UNKNOWN, 0.0)`
  - `test_gallery_nearest`：假 embedder（按裁图平均颜色出 3 维单位向量），refs `{"sit": [红图×2], "memory": [蓝图]}` → 红色裁图判 `sit`；灰色裁图（和两种都差不多）判 `UNKNOWN`
  - `test_gallery_single_kind_no_margin`：只有一种参考 → 够像就认
  - `test_gallery_embed_error_is_unknown`：embedder 返回 None → `(UNKNOWN, 0.0)`
  - `test_gallery_load_skips_underscore_and_empty`：`tmp_path` 下 `sit/a.jpg`、`_removed/b.jpg`、空目录 `music/` → `kinds == ["sit"]`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（余弦用 `vision/embed.py` 的 `cosine`）
- [ ] **Step 4: 跑 `tests/test_icons_map.py tests/test_social.py` 确认通过**
- [ ] **Step 5: 提交** — `feat(icons): 模板按框缩放认图标、DINOv2 最近邻底库`

---

### Task 3: 没名字的圆圈先判归属（防误点，不受开关管）

**Files:**
- Modify: `src/skydango/vision/perception.py`（`__init__` 加 `icons_cfg: IconsConfig | None = None`，存 `self.icons_cfg = icons_cfg or IconsConfig()`；`process` 里 orphans 那段，约 800~818 行）
- Modify: `tests/test_perception.py`（下面列的旧测试补一个人）
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 1 的 `owner_of`、`IconsConfig`
- Produces: 每个圆圈轨迹 `ring.data["owner"]`：`"friend"`（`_tag_above` 挂上的）/ `"person"` / `MAP_OWNERS` 之一；Task 4 读它

改法：
- 判归属用的框：人 = `players`（已去掉团子和外形头判成先祖 / 共享空间的）+ `self._others` 里外形不是 `spirit` 的；先祖 = 这一帧 `cls == "spirit"` 的轨迹 + `self._others` 里外形是 `spirit` 的；物件 = `cls in ("bench", "bonfire", "instrument")` 的轨迹（都用这一帧 `t.last == now` 的）
- 挂上名字标签的圈 `ring.data["owner"] = "friend"`（原逻辑不变）
- orphans：先 `owner = owner_of(...)` 记进 `ring.data["owner"]`；**只有 `owner == "person"` 的进原来那段循环**（分类、`_disk_like`、生成 `Request(STRANGER, …)`、`REQUEST_HOLD` 的"圆圈还在原地"判断也只看 person 的圈）；其余这一步什么都不做（Task 4 接手）

- [ ] **Step 1: 改旧测试**：`test_ring_without_tag_is_a_stranger_request`、`test_stranger_request_survives_a_frame_where_the_icon_is_unclear`、`test_orphan_dark_disk_is_not_a_candle_request_even_over_bright_background`、`test_orphan_white_ring_is_still_a_candle_request`，以及 1940~1960 行附近其他用 `ring(1400)` 期望陌生人请求的测试：`det.frames` 里同一帧加 `player(1355, y=440)`（人在圆圈正下方）
- [ ] **Step 2: 写失败的新测试**
  - `test_orphan_ring_without_person_is_never_a_request`：`det.frames = [[ring(1400)]]`、`FakeIcons({"next": "candle"})`、`ringed()` 画面（有白圈）→ `STRANGER not in w.requests`，`[t for t in w.last_tracks if t.cls == "social_ring"][0].data["owner"] == "map"`
  - `test_orphan_ring_over_bonfire_is_not_a_request`：同上再加 `Detection("bonfire", Rect(1340, 470, 120, 100), 0.9)` → 没有请求、`owner == "bonfire"`
  - `test_orphan_ring_over_spirit_is_not_a_request`：加 `Detection("spirit", Rect(1355, 440, 90, 220), 0.9)` → 没有请求、`owner == "spirit"`
  - `test_ring_owner_rule_ignores_icons_enabled`：`icons_cfg=IconsConfig(enabled=False)` 时第一个测试照样没有请求
  - `test_friend_ring_owner_is_friend`：沿用现有好友圈测试的造法，断言 `owner == "friend"`
- [ ] **Step 3: 跑测试确认新测试失败、改过的旧测试通过**
- [ ] **Step 4: 实现**
- [ ] **Step 5: 跑 `tests/test_perception*.py tests/test_social.py tests/test_light_replay.py` 确认全过**（点亮回放必须不变）
- [ ] **Step 6: 提交** — `fix(perception): 没名字的圆圈只有下面是人才可能是互动请求（地图图标不会被点）`

---

### Task 4: 地图图标的运行时（`MapIcons`）、`icons()`、overlay、接线

**Files:**
- Modify: `src/skydango/vision/icons_map.py`（`MapIcons`）
- Modify: `src/skydango/vision/perception.py`（`__init__` 加 `map_icons=None`；`process` 里对 `MAP_OWNERS` 的圈调 `map_icons.observe`；新方法 `icons(now)`；`overlay` 里画成 `icon`；难例 `icon_unknown`）
- Modify: `src/skydango/cli.py`（`_scene_watcher` 里建 `MapIcons` 和传 `icons_cfg`；新函数 `_map_icons(cfg, icons, dino, run)`）
- Test: `tests/test_icons_map.py`、`tests/test_perception.py`

**Interfaces:**
- Consumes: Task 1~3
- Produces:
  - `class MapIcons`：`__init__(self, cfg: IconsConfig, template: IconClassifier | None, gallery: IconGallery | None, save_dir: Path | None)`；`observe(self, frame: np.ndarray, ring: Track, owner: str, now: float) -> bool`（更新 `ring.data["votes"]`（最多 `cfg.vote` 条）、`ring.data["icon"]`（投票结果）、`ring.data["icon_score"]`；按需存裁图；返回「这条轨迹第一次投出 `UNKNOWN` 且 `hits ≥ min_hits`」，给调用方报难例）
  - 分类器选择：`cfg.classifier == "dino"` 且 `gallery` 不为 None 用底库，否则模板；两者都没有 → `UNKNOWN`
  - 存裁图：分数 < 门槛 + 0.1 或 `UNKNOWN`、这条轨迹上次存 ≥ 2 秒、本次运行累计 < `save_max` → `save_dir/<kind>/<HHMMSS-mmm>-t<轨迹>.jpg`（原始彩色裁图），`save_dir/icons.jsonl` 一行 `{t, track, kind, score, classifier, owner, box, file}`；`cfg.save = False` 或 `save_dir` 为 None 不存；写盘出错只记 DEBUG
  - `PerceptionWatcher.icons(now: float) -> list[Icon]`：不是 enabled / `map_icons is None` / 暂停中 → `[]`；取 `last_tracks` 里 `cls == "social_ring"`、`data["owner"] in MAP_OWNERS`、有 `data["icon"]`、`hits ≥ cfg.min_hits`、`now − t.last ≤ PEOPLE_STALE` 的；`side = side_of(...)`、`label = map_label(icon, owner)`；排序同 `objects()`（左 / 前 / 右，同一边按 x）
  - overlay：上面这些圈的条目 `kind = "icon"`、`label = map_label(...)`、`unknown = True`（是 `UNKNOWN` 时）；enabled 关着时照旧画 `ring`
  - 难例：`observe` 返回 True 时 `self._report_hard(frame, now, "icon_unknown", f"轨迹 {id} 的图标认不出（{owner}）", tracks)`
  - `cli._map_icons(cfg, icons, dino, run) -> MapIcons | None`：`cfg.icons.enabled` 才建；`classifier == "dino"` 时：`dino`（认装扮建好的）路径和 `cfg.icons.dino` 是同一个文件就包一层 `DinoGuard` 共用，否则 `OnnxEmbedder(cfg.icons.dino, norm="imagenet", device=cfg.perception.device, what="icons.dino")`；建不起来 / 参考图一张都没有 → WARNING「认图标：DINOv2 用不了（原因），退回模板」、`gallery = None`；`save_dir = run.path / "icons" if run else None`；`_scene_watcher` 传 `icons_cfg=cfg.icons, map_icons=_map_icons(...)`

- [ ] **Step 1: 写失败的测试**
  - `MapIcons`（`tests/test_icons_map.py`，假模板分类器 / 假底库）：投票 5 条封顶、`icon` 是投票结果；`classifier="dino"` 用底库、`gallery=None` 退回模板；`UNKNOWN` 第一次满 `min_hits` 时 `observe` 返回 True、之后不再返回；存裁图 2 秒节流、`save_max` 封顶、`icons.jsonl` 字段齐、`save=False` 不写
  - 感知层（`tests/test_perception.py`，`watcher(..., icons_cfg=IconsConfig(enabled=True), map_icons=MapIcons(...假分类器...))`）：
    - `test_map_icon_reported_after_min_hits`：同一个圈连续 3 帧 → `w.icons(now)` 一条、`label` 对、`owner == "map"`；第 2 帧时还是空
    - `test_map_icons_empty_when_disabled`：`enabled=False` → `w.icons(now) == []`，overlay 里那个圈的 kind 是 `ring`
    - `test_map_icon_overlay_kind`：enabled 时 overlay 有 `kind == "icon"`，认不出时 `unknown is True`
    - `test_map_icon_under_open_panel_not_reported`：圈中心在 `log_roi` 里、`panel_visible=True` → `icons()` 空（Review Focus 5）
    - `test_unknown_icon_reports_hard_case_once`：假 hardcases 收集器，认不出的圈 5 帧 → `report` 只调一次、reason `icon_unknown`
  - `cli`（`tests/test_perception.py` 里已有 `test_scene_watcher_*` 的造法）：`test_scene_watcher_map_icons_falls_back_to_template`：`classifier="dino"`、`dino` 指向不存在的文件 → 建出 `MapIcons` 且 `gallery is None`、日志有「退回模板」
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑 `tests/test_icons_map.py tests/test_perception*.py tests/test_light_replay.py` 确认通过**
- [ ] **Step 5: 提交** — `feat(perception): 地图交互图标认种类、icons()、管理面板画面`

---

### Task 5: 给大脑和画面

**Files:**
- Modify: `src/skydango/brain/body.py`（「画面里的东西」那行后面，约 1893~1895 行）
- Modify: `src/skydango/brain/images.py`（`scene_note` 末尾）
- Modify: `src/skydango/brain/prompt.py`（`brain_prompt` 加参数 `icons: bool = False`；常量 `ICONS_RULE`）
- Modify: `src/skydango/cli.py`（`brain_prompt(...)` 调用处传 `icons=cfg.perception.enabled and cfg.icons.enabled`）
- Modify: `src/skydango/console/static/stage.js`（`COLORS.icon = "#a3e635"`、`NAMES.icon = "交互图标"`；`b.unknown` 时虚线 `[3,3]`；`icon` 的标签和 ring 一样画在框下方）
- Test: `tests/test_brain_body*.py`（找 status 测试所在文件，有「画面里的东西」断言的那个）、`tests/test_brain_images.py`、`tests/test_brain_prompt.py`、`tests/test_viewer.py`（node 跑 stage.js 的那组）

**Interfaces:**
- Consumes: Task 1 的 `describe_icons`、Task 4 的 `env.icons(now)`
- Produces:
  - status：`icons = self.env.icons(now) if hasattr(self.env, "icons") else []`，非空时 `parts.append("画面里的图标：" + describe_icons(icons))`
  - `scene_note`：有图标时追加「画面里认出的图标（坐标按这张图）：」+ 每个一行 `- {label}：({x}, {y}) 附近` + 「图标只说明那里能互动。」
  - `ICONS_RULE = "图标只说明那里能互动，你现在不会去点它，别答应去坐下或者去点。"`：`icons=True` 时接在「光遇常识」里「坐下、弹琴还不会，别答应。」那句后面（同一行末尾加空格接上）

- [ ] **Step 1: 写失败的测试**：status 有图标时多一行、没有 / env 没有 `icons` 方法时不变；`scene_note` 有图标的段、没有时逐字照旧；`brain_prompt(icons=True)` 含 `ICONS_RULE`、`icons=False` 时和改之前逐字一样；stage.js 的 `COLORS.icon` / `NAMES.icon` 存在（照 `tests/test_viewer.py` 里 node require 的写法）
- [ ] **Step 2: 跑确认失败** → **Step 3: 实现** → **Step 4: 跑这几个测试文件确认通过**
- [ ] **Step 5: 提交** — `feat(brain): 状态、眼睛、提示词和管理面板画面加上地图交互图标`

---

### Task 6: 离线工具 `perception icon-eval` / `icon-cut`

**Files:**
- Create: `src/skydango/vision/icon_eval.py`
- Modify: `src/skydango/cli.py`（`psub.add_parser("icon-eval", …)`、`psub.add_parser("icon-cut", …)` 和两个 `_perception_icon_eval` / `_perception_icon_cut`）
- Test: `tests/test_icon_eval.py`

**Interfaces:**
- Consumes: Task 1~2（`owner_of`、`icon_crop`、`classify_template`、`IconGallery`、`map_label`）
- Produces:
  - `evaluate(images: list[Path], detect: Callable, template: IconClassifier | None, gallery: IconGallery | None, cfg: IconsConfig, out: Path, labels: Path | None = None) -> dict`：每张图跑 `detect` → 人 / 先祖 / 物件框 → 每个 `social_ring` 判归属；`MAP_OWNERS` 的两种分类器都跑；写
    - `out/template-<kind>.jpg`、`out/dino-<kind>.jpg`（每格一张裁图 + 左上角编号，网格同 `catalog.contact_sheet` 的排法，一张最多 100 格，多了分 `-2`、`-3`）
    - `out/disagree.jpg`（两种结果不同的）、`out/owners/<图名>.jpg`（每个圈画框 + 归属文字，最多 50 张）
    - `out/crops.jsonl`（编号、图、框、归属、两种结果和分数）、`out/report.md`（每种两种分类器各判出多少、`unknown` 占比、两种分数的分位数、建议 `dino_match` = 两者一致的那些 dino 分数的 5% 分位数）
    - `labels` 给了（数据集的 `labels/<split>/`）时再算 `social_ring` 召回（IoU ≥ 0.5）写进报告
    - 返回 `{"rings": n, "map": m, "agree": k, "recall": r | None}`
  - `cut(image: Path, kind: str, box: Rect, refs: Path, templates: Path | None) -> list[Path]`：裁图（`icon_crop`，`IconsConfig().crop_scale`）存 `refs/<kind>/<时间>.jpg`；给了 `templates` 再把裁图按 `RING_PX` 缩放存 `templates/<kind>.png`（已有就 `<kind>-2.png`、`-3`…）
  - CLI：`perception icon-eval <目录> [--model] [--labels] [-o]`（默认输出 `tmp/icon-eval/<时间>/`；目录里有 `images/val` 时取它和 `labels/val`）；`perception icon-cut <截图> <kind> --box x,y,w,h [--template]`（refs 用 `[icons] refs`，模板目录用 `[social] icons_dir`）
- [ ] **Step 1: 写失败的测试**：假 detect（一张图两个圈：一个下面有人、一个下面空）→ 只有空的那个进 `crops.jsonl`、`report.md` 存在、`template-*.jpg` 生成；给 labels 时召回 = 1.0；`cut` 的编号续接（已有 `sit.png` → 写 `sit-2.png`）
- [ ] **Step 2: 跑确认失败** → **Step 3: 实现** → **Step 4: 跑确认通过**
- [ ] **Step 5: 提交** — `feat(icons): icon-eval 离线比两种分类器、icon-cut 截模板和参考图`

---

### Task 7: 收件箱的数据入口：导入录像、老帧回炉、整图火焰预标

**Files:**
- Modify: `src/skydango/vision/inbox.py`（`add_images`、`add_dataset`；`pass_frame` 支持回炉帧覆盖；`frame_name` 不动）
- Modify: `src/skydango/cli.py`（`inbox add` 子命令；`_inbox_process` 的预标注多一步整图火焰）
- Modify: `src/skydango/config.py`（Step 5 量出来的 `flame_ring` 默认值）
- Test: `tests/test_inbox.py`

**Interfaces:**
- Consumes: 收件箱现有的 `collect` / `process` / `pass_frame` / `similar` / `split_of`
- Produces:
  - `add_images(src: Path, inbox: Path, every: int = 1) -> int`：`src/*.jpg|*.png` 按文件名排序、隔 `every` 取一张，复制进 `inbox/<src 目录名>/raw/`（png 转 jpg），`_index.jsonl` 记一行（`"source": "import"`）；已有的跳过；返回新复制张数。去重交给 `process`（照旧）；`process` 的 `reason` 没有 `hard.jsonl` 时是 None，页面上显示「导入」（`frames.json` 写 `"reason": "import"`：`_hard_reasons` 为空且 `_index.jsonl` 那行 `source == "import"` 时）
  - `add_dataset(dataset: Path, inbox: Path, splits=("train", "val")) -> int`：回炉。放进 `inbox/_redo-<时间>/raw/`，文件名 = 原帧名；同时把原标注复制进 `labels/<帧>.txt`；`frames.json` 预先写好每帧 `{"redo": {"name": 原帧名, "split": 原 split}, "split": 原 split, …}`（`process` 见到 `redo` 时：分边用它、不按哈希；预标注 = 原标注 + YOLO 预标，按现有 `merge_labels` 合并）
  - `pass_frame`：entry 有 `redo` 时目标路径用 `redo.name` / `redo.split`、**允许覆盖**；覆盖前把原 `labels/<split>/<name>.txt` 复制到 `dataset/_backup/redo-<inbox 运行名>/<split>/<name>.txt`（同一次回炉同一个目录，已有不覆盖）；不是回炉帧时同名照旧抛 `FileExistsError`。`undo_frame` 对回炉帧：把备份的标注拷回去（不删图）
  - 整图火焰预标：新函数 `flame_rings(frame, flame_tpl, min_score: float, ring_k: float, skip: list[Rect]) -> list[Rect]`（放 `vision/icons_map.py`）：`find_flames` 在整张图（减去 `skip`，即聊天面板开着时的面板区域）里找，分数 ≥ `min_score`（= `[social] disk_sure`）的每个 `Disk` 补成以圆心为中心、半径 `r × ring_k` 的正方形；和已有 `social_ring` 框 IoU ≥ 0.5 的不加。`_inbox_process` 的 `weak` 回调在 `_prelabel` 结果后面接上它（`flame` 模板用 `candle.load_flame(cfg.social.flame)`；读不到模板就跳过这一步、打印一行）
  - CLI：`perception inbox add <目录> [--every N] [--redo]`：`--redo` 时 `<目录>` 必须是含 `images/` `labels/` 的数据集，调 `add_dataset`；否则 `add_images`；打印「导入了 N 张 → <收件箱>/<运行>，接着跑 perception inbox process」
- [ ] **Step 1: 写失败的测试**：`add_images` 抽帧和幂等；`add_dataset` 带原标注和 `redo` 字段；`process` 对回炉帧沿用原 split；`pass_frame` 回炉帧覆盖 + 备份、普通帧同名照旧报错；`undo_frame` 回炉帧还原标注；`flame_rings` 在合成画面（画一个火焰模板）上补出框、和已有框重叠时不重复、`skip` 区域里不找
- [ ] **Step 2: 跑确认失败** → **Step 3: 实现** → **Step 4: 跑 `tests/test_inbox.py tests/test_icons_map.py` 确认通过**
- [ ] **Step 5: 量 `flame_ring`**：写一个一次性脚本放 `tmp/`（不进 git）：遍历 `datasets/sky/labels/*/` 里的 `social_ring` 框，在对应图上框内 `find_flames`，取「圆圈半径（框宽 / 2）/ 火焰 r」的中位数（只算找到恰好一团火焰的框）；把结果写成 `IconsConfig.flame_ring` 默认值（保留两位小数）和 `config.example.toml`，注释写「10-05 在 datasets/sky 的 N 个火焰圆圈上量的中位数」。数据集或模板不在时跳过这一步，默认值保持 2.0 并在提交信息里说明
- [ ] **Step 6: 提交** — `feat(inbox): 导入录像、老帧回炉、整图火焰补圆圈框`

---

### Task 8: 文档

**Files:**
- Modify: `CLAUDE.md`（代码结构表加 `vision/icons_map.py` `vision/icon_eval.py` 一行；YOLO 感知层一节后面加「## 认交互图标（`[icons]`，要配合 `[perception]`）」一节：spec / plan 路径、三步、防误点规则不受开关管、`icons()` / status / 眼睛 / 画面、代码默认关、**还没有地图图标的模板 / 参考图和新 YOLO，真机验证见 spec §8**；常用命令加 `icon-eval` / `icon-cut` / `inbox add`；运行目录表加 `icons/`）
- Modify: `docs/game-ops.md`（加「地图交互图标」一节骨架：一张空表「图标 | 长什么样 | 多远冒出来 | 大小随距离变吗 | 点了会怎样 | 在哪见过」+ 待看清的「举着的蜡烛火苗和头顶火焰圆盘」一条，标「待真机」）
- Modify: `docs/progress/2026-09-28-yolo-training.md`（标注规则加：所有可点的圆圈 / 图标都标 `social_ring`、框贴圆圈外沿、老的 100×100 框回炉时拉紧）
- Modify: `docs/superpowers/specs/2026-10-05-icon-detection-design.md`（§0 第 3 条后补「10-05：收件箱已合进 main」）
- Modify: `docs/superpowers/specs/2026-10-04-future-roadmap-design.md`（「已定」下记一行：②b 代码完成、等真机）

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 跑全量测试（后台）** `.venv\Scripts\python.exe -m pytest -q`，预期全过
- [ ] **Step 3: 提交** — `docs(icons): CLAUDE.md、game-ops 骨架、标注规则、路线图进度`
