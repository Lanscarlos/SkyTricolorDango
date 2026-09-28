# 感知层三期（代码部分）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把三期设计里不需要模型、不需要真机就能写完的部分做完：远处小目标二次检测（§1）、认地图（§2）、动作识别的数据工具和运行时接口（§3，研究性质，默认关）、跟随第 1 步（§4，只改提示词）、没认出的名字清单（§5）。各项验收（GPU / 真机 / 录数据）留给用户。

**Architecture:** 每一项都挂在 `PerceptionWatcher` 上、用可选的构造参数注入（`places=`、`gestures=`、`unknown=`），不传就和现在一样；计算都拆到独立模块（`vision/places.py`、`vision/gesture.py`、`vision/unknownnames.py`），用假检测器 / 假 OCR / 假嵌入模型测。身体只通过 `hasattr(env, "pop_gestures")` 接新事件，`[perception] enabled = false` 时什么都不变。

**Tech Stack:** Python 3.11+，numpy、opencv、onnxruntime（可选）；pytest（合成画面 + 假模型）。

**Spec:** `docs/superpowers/specs/2026-09-28-perception-phase3-design.md`（总纲 `…-perception-yolo-architecture-v0.2.md`；一期 §4 暂停、二期 §4 远近）

## 已替用户拍板的待确认项（写回设计 §7，用户可改）

1. §2 图库覆盖哪些地方：代码不依赖这个答案，用户自己 `places add`；文档建议先做常去的 5~8 个
2. §3 认哪几个动作：`[gesture] labels` 可配，默认 `["none", "wave", "bow"]`（挥手、鞠躬）；**本期只做数据工具、离线评估和运行时接口，不训练模型**（这台机器没有 torch / GPU，也还没有录数据），`[gesture] enabled` 默认关，模型的输入输出约定写进设计 §3
3. §4 第 2 步（视觉伺服 `follow`）**本期不做**：前提"大脑的 `move` 工具已实现并在真机标定过步长"不成立（`brain/locomotion.py` 有了，但 `Body` / `ToolBox` 没接 `move`，步长没标定）。只做第 1 步（提示词：想跟谁走就请他牵手）

## Global Constraints

- 不传新参数 / 新配置默认值时，`PerceptionWatcher`、身体、Agent、大脑提示词以外的行为和现在完全一样；现有 478 个测试全过
- `[perception]` 新增：`far_height = 0.08`（人物框高 < 画面高 × 这个才做二次检测）、`far_crops = 3`（每帧最多几个裁剪，0 = 关）
- 新配置 `[places]`：`enabled = false`、`dir = "places"`、`model = "models/places.onnx"`（也可以写 `"thumb"` 用内置的缩略图基线）、`size = 224`、`norm = "imagenet"`（`imagenet` / `clip` / `none`）、`device = "cpu"`、`place_min = 0.8`、`place_margin = 0.05`、`place_interval = 30.0`
- 新配置 `[gesture]`：`enabled = false`、`model = "models/gesture.onnx"`、`labels = ["none", "wave", "bow"]`、`names = {wave = "挥手", bow = "鞠躬"}`、`frames = 16`、`fps = 8.0`、`size = 112`、`interval = 2.0`、`min_prob = 0.9`、`cooldown = 30.0`
- `places/` 加进 `.gitignore`；`datasets/`、`models/` 本来就忽略
- 二次检测：每条轨迹每秒最多一次（`FAR_RETRY = 1.0`），只对 `player`（不对 `player_unlit`，黑影不用认名字）；裁剪区域 = 人物框上方，宽 3 × 框宽、高 2.5 × 框高（从 `p.y - 2h` 到 `p.y + 0.5h`，水平居中），只收 `name_tag` / `social_ring`；裁剪直接交给检测器（它自己 letterbox 放大到 `imgsz`，等价于设计里"放大再检测"）
- 认地图：`scene_change` 那种画面大变（缩略图差 > `scene_change`）后、离上次 ≥ 3 s，或每 `place_interval` 秒跑一次；暂停期间不跑；`describe()` 里加 `- 看起来在：<地名>`（`env.place_keep` 秒内）
- 没认出的名字：OCR 置信度 ≥ 0.9、规范化后 ≥ 2 个字、对不上 friends.md；同一条标签轨迹只记一次；和已记下的名字 `similar(…, 0.75)` 算同一个；**只列出，不写 friends.md**
- `gesture` 事件文字：`"<好友名>对你<动作中文名>"`；同一人同一动作 `cooldown`（30 s）内只报一次；只对好友、距离"近 / 中"、框中心在画面中间一半（`width/4 ~ 3width/4`）的人跑，每条轨迹每 `interval` 秒判一次；概率 < `min_prob` 或是 `none` 不报
- 文件名不能带 `:` 等 Windows 不允许的字符
- 注释、日志、CLI 输出、给大脑的文字用中文，风格照现有代码；测试命令 `python -m pytest -q`

## Review Focus

1. **二次检测找到的标签和原图里已有的标签重复**（原图也框到了，只是没挂上人）→ 不能多出一条轨迹、多读一次 OCR：映射回来的框和已有 `name_tag` 框 IoU ≥ 0.3 就丢掉。测试在 Task 2。
2. **裁剪区域越界**（人站在画面最上沿 / 最左边）→ 裁剪夹到画面内，宽或高 < 8 px 就不裁，不崩。测试在 Task 2。
3. **图库只有一个地方 / 图库是空的** → 一个地方时只看 `place_min`；空图库时 `match` 返回"认不出"、`_scene_watcher` 记警告不挂认地图。测试在 Task 4。
4. **图库里加了新图、换了模型** → 缓存 `_index.npz` 按（文件路径、修改时间、模型标识）失效重算，不拿旧向量。测试在 Task 4。
5. **动作分类器报错 / 裁剪帧不够 16 张** → 不报事件、不影响这一帧别的识别（异常记日志吞掉）。测试在 Task 7。

---

### Task 1: 没认出的名字 → `runs/<…>/unknown_names/` + `perception unknown-names`

**Files:**
- Create: `src/skydango/vision/unknownnames.py`
- Modify: `src/skydango/vision/perception.py`（`_ocr` 返回分数、`_read_name` 记下没认出的名字、构造参数 `unknown=None`）、`src/skydango/runlog.py`（docstring 加一行）、`src/skydango/cli.py`（`_scene_watcher` 挂上、`perception unknown-names`）
- Test: `tests/test_unknownnames.py`、`tests/test_perception.py`

**Interfaces:**
- Produces:
  - `UnknownNames(folder: Path, names: Callable[[], list[str]], wall: Callable[[], float] = time.time)`；`add(text: str, crop: np.ndarray) -> None`（和已记的 `similar(…, 0.75)` 就计数 +1，否则新建一条并存 `<序号>.jpg`，然后重写 `names.jsonl`）；属性 `entries: dict[str, dict]`（键 = 名字，值 `{"name", "count", "first", "last", "image"}`，时间是 `"%Y-%m-%d %H:%M:%S"` 字符串，`image` 是相对 folder 的文件名）
  - `collect(runs_root: Path, last: int, friends: list[str]) -> list[dict]`：最近 `last` 次运行（按目录名排）的 `unknown_names/names.jsonl` 合并（名字规范化后相同算一个，`count` 相加、`last` 取最大、`image` 取最近一次运行的绝对路径），去掉现在已经对得上 friends 的，按 `count` 降序
  - `PerceptionWatcher(..., unknown=None)`；`_ocr(crop) -> tuple[str, float]`（`read_line` 的分数；`recognize` 路径取各行最低分，没字时 0.0）
  - `perception unknown-names [--runs runs] [--last 5]`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_unknownnames.py
def test_add_counts_similar_names_once_and_writes_jsonl(tmp_path):
    u = UnknownNames(tmp_path / "unknown_names", lambda: ["懒洋洋大王"], wall=lambda: 0.0)
    img = np.zeros((20, 60, 3), np.uint8)
    u.add("星星小铺", img); u.add("星星小铺", img); u.add("星星小铺子", img)
    assert list(u.entries) == ["星星小铺"] and u.entries["星星小铺"]["count"] == 3
    rows = [json.loads(l) for l in (tmp_path / "unknown_names" / "names.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["name"] == "星星小铺" and (tmp_path / "unknown_names" / rows[0]["image"]).exists()

def test_collect_merges_runs_sorts_by_count_and_drops_known_friends(tmp_path):
    # 两次运行：A 名字各 2 次、B 名字 1 次；B 现在已经在 friends 里
    ...
    assert [r["name"] for r in collect(tmp_path, 5, ["乙"])] == ["甲"] and out[0]["count"] == 4

# tests/test_perception.py
def test_clear_but_unknown_name_is_recorded_once_per_track(tmp_path):
    # FakeOcr 读出 "星星小铺"（0.99），不在 FRIENDS 里；同一轨迹读 3 次只记 1 次
    assert u.entries["星星小铺"]["count"] == 1

def test_low_confidence_text_is_not_recorded(...):  # 分数 0.8 → 不记
```

- [ ] **Step 2: 跑测试确认失败** — `python -m pytest -q tests/test_unknownnames.py tests/test_perception.py -k unknown` → ImportError / 断言失败
- [ ] **Step 3: 实现** `unknownnames.py`；`_read_name` 里：`text, score = self._ocr(...)`，没对上好友、`score >= 0.9`、`not tag.data.get("unknown_logged")` 时 `self.unknown.add(text, crop)` 并标记；`sweep` 里改用 `_ocr(...)[0]`。`_scene_watcher` 在有 `run` 时传 `UnknownNames(run.path / "unknown_names", _friend_names(cfg))`；`_stop_scene` 有记录时打印"没认出的名字：N 个 → 路径（perception unknown-names 汇总）"
- [ ] **Step 4: 跑测试确认通过**，再跑全量 `python -m pytest -q`
- [ ] **Step 5: 提交** `feat(perception): 记下读得清楚但不在好友名单里的名字，perception unknown-names 汇总`

### Task 2: 远处小目标二次检测

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/config.py`、`config.example.toml`
- Test: `tests/test_perception.py`、`tests/test_config.py`

**Interfaces:**
- Produces: `PerceptionConfig.far_height: float = 0.08`、`far_crops: int = 3`；`far_region(p: Rect, width: int, height: int) -> Rect | None`（模块级函数，越界夹回、太小返回 None）；`PerceptionWatcher._far_tags(frame, players, tags, now, width, height, panel_visible) -> list[Track]`（新增的 name_tag / social_ring 轨迹，已经过 `tracker.update`）；统计 `self.far_runs: int`（跑了几次裁剪，测速 / compare 用）
- 人物轨迹上记 `data["tag_at"]`（最近一次挂上名字标签的时间）、`data["far_at"]`（最近一次二次检测）

- [ ] **Step 1: 写失败的测试**

```python
def test_far_player_gets_a_second_look_above_its_head():
    # 主检测只看到远处小人（h=60 < 0.08*1080）；FakeDetector 对裁剪图（形状不是 1080×1920）返回裁剪坐标里的 name_tag
    w.process(frame(), 0.0, False)
    assert w.nearby(0.0) == ["懒洋洋大王"] and w.far_runs == 1
    assert w.labels["懒洋洋大王"][:2] == (crop.x + 10, crop.y + 20)  # 映射回整图坐标

def test_far_crop_is_rate_limited_per_track_and_per_frame():
    # 5 个远处小人、far_crops=3 → 第一帧 3 次；0.5 s 后同一批不再裁；1.0 s 后再裁
def test_far_crop_skips_players_with_a_tag_and_big_players_and_unlit(): ...
def test_far_crop_drops_tags_already_seen_in_the_full_frame():  # IoU ≥ 0.3 → 不新建轨迹
def test_far_region_clamps_to_frame_and_rejects_tiny():
    assert far_region(Rect(0, 10, 20, 60), 1920, 1080) == Rect(0, 0, 50, 40)  # 左上越界夹回
    assert far_region(Rect(5, 0, 2, 4), 1920, 1080) is None
def test_far_crops_zero_turns_it_off(): ...
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`process()` 里 `tracks = self.tracker.update(dets, now)` 并分好 `players / tags / rings` 之后，`extra = self._far_tags(...)`，把其中 `name_tag` 并进 `tags`、`social_ring` 并进 `rings`、全部并进 `tracks`（第二次 `tracker.update(extra_dets, now)` 是安全的：同一个 `now` 不会删别的轨迹）。候选 = `cls == "player"`、`box.h < far_height * height`、`_tag_over` 这一帧为空、`now - tag_at >= FAR_RETRY`、`now - far_at >= FAR_RETRY`，按 `far_at` 从旧到新取前 `far_crops` 个。检测结果加上裁剪偏移、过 `_filter(..., drop_self=False)` 和 `conf`、和已有标签 IoU ≥ 0.3 的丢掉。人物循环里 `_tag_over` 找到标签时写 `tag_at = now`
- [ ] **Step 4: 跑测试确认通过**，全量测试
- [ ] **Step 5: 提交** `feat(perception): 远处小人头顶裁一块再检测一次，认远处好友的名字`

### Task 3: `perception compare` / `bench` 报远处认出率，可切换二次检测

**Files:**
- Modify: `src/skydango/vision/compare.py`、`src/skydango/cli.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: Task 2 的 `far_height`、`far_runs`
- Produces: `FrameResult.far: int`（这一帧 `player` 轨迹里框高 < `far_height` 的个数）、`far_named: int`（其中挂上了好友名的）；`summarize()` 多一个键 `"far": {"players": int, "named": int, "rate": float | None}`；`report_md` 多一节"## 远处的人（框高 < 8%）"；`perception compare / bench` 多一个 `--far-crops N`（覆盖配置，0 = 关）；`bench` 输出里加"二次检测 N 次"

- [ ] **Step 1: 写失败的测试** — `test_summarize_reports_far_named_rate`：两帧 `far=2, far_named=1` → `{"players": 4, "named": 2, "rate": 0.5}`；`report_md` 里有"远处"和"50%"；`compare_frames` 用 Task 2 测试里的远处小人场景填 `far` / `far_named`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**；报告里说明"对比开 / 关二次检测：同一批录像跑两次 `--far-crops 0` 和默认，看这一节的认出率差（目标 ≥ 30 个百分点），整帧耗时看 bench 的 p95（≤ 66 ms）"
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(compare): 报告远处人物的认出率，compare / bench 可以关掉二次检测对比`

### Task 4: `vision/places.py`：遮挡、特征、图库、判定

**Files:**
- Create: `src/skydango/vision/places.py`
- Modify: `src/skydango/config.py`（`PlacesConfig`、`Config.places`）、`config.example.toml`、`.gitignore`
- Test: `tests/test_places.py`、`tests/test_config.py`

**Interfaces:**
- Produces:
  - `PlacesConfig`（字段见 Global Constraints）
  - `mask_scene(img, boxes: list[Rect], keep_roi: list[float], hide_rois: list[list[float]]) -> np.ndarray`：拷贝一份，`keep_roi` 外面、`hide_rois` 里面、每个框（四周放宽 10%）都涂成原图的平均色
  - `ThumbEmbedder`（`key = "thumb"`）和 `OnnxEmbedder(path, size, norm, device)`（`key = "<文件名>:<size>:<norm>"`），都有 `embed(img: BGR) -> np.ndarray`（float32、L2 归一化）；`make_embedder(cfg: PlacesConfig)`
  - `PlaceMatch(name: str | None, score: float, best: str | None, second: float)`；`decide(scores: list[tuple[str, float]], place_min, place_margin) -> PlaceMatch`
  - `PlaceLibrary(root: Path, embedder)`：`load() -> int`（张数；缓存 `root/_index.npz`，键是相对路径 + mtime + `embedder.key`，变了就重算；`_` 开头的目录跳过）、`scores(vec, exclude: Path | None = None) -> list[tuple[str, float]]`、`add(name, img) -> Path`（存 `root/<name>/<YYYYmmdd-HHMMSS>.jpg`，重名加序号）、`places() -> list[str]`
  - `PlaceRecognizer(library, cfg: PlacesConfig, keep_roi, hide_rois)`：`recognize(img, boxes) -> PlaceMatch`（遮挡 → 特征 → `decide`）

- [ ] **Step 1: 写失败的测试**

```python
def test_mask_scene_paints_boxes_panel_and_bottom_with_mean_color(): ...
def test_decide_needs_min_and_margin_over_other_places():
    assert decide([("云野", 0.9), ("云野", 0.88), ("雨林", 0.8)], 0.8, 0.05).name == "云野"   # 同一地方的第二张不算
    assert decide([("云野", 0.9), ("雨林", 0.87)], 0.8, 0.05).name is None                    # 差得不够
    assert decide([("云野", 0.7)], 0.8, 0.05).name is None
    assert decide([("云野", 0.85)], 0.8, 0.05).name == "云野"                                 # 只有一个地方
    assert decide([], 0.8, 0.05).name is None
def test_library_caches_vectors_and_invalidates_on_new_file_or_model(tmp_path):  # 计数假 embedder 被调几次
def test_library_leave_one_out_scores_exclude_self(tmp_path): ...
def test_thumb_embedder_is_normalized_and_tells_different_scenes_apart(): ...
def test_add_writes_under_place_folder_without_colons(tmp_path): ...
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**。`OnnxEmbedder`：BGR→RGB、缩放到 `size×size`（模型输入是固定尺寸时以模型为准）、按 `norm` 减均值除方差（imagenet `(0.485,0.456,0.406)/(0.229,0.224,0.225)`，clip `(0.4815,0.4578,0.4082)/(0.2686,0.2613,0.2758)`，none 只除 255）、NCHW；输出里有二维的取二维的，否则取 `[:, 0]`（CLS token）。`ThumbEmbedder`：32×18 灰度减均值 + HSV 色相 / 饱和度 8×4 直方图，各自归一化后拼接再 L2 归一化。onnxruntime 没装时 `OnnxEmbedder` 抛 `ImportError("…pip install onnxruntime")`
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(places): 参考截图认地图——遮人和 UI、算特征、图库缓存、最像且拉开差距才算`

### Task 5: 认地图接进感知层 + `places add / test / bench`

**Files:**
- Modify: `src/skydango/vision/perception.py`（构造参数 `places=None`、`_watch_place`、`describe`）、`src/skydango/cli.py`（`_scene_watcher` 挂上；`places` 子命令）
- Test: `tests/test_perception.py`、`tests/test_places.py`（CLI 部分用假设备 / 图片目录）

**Interfaces:**
- Consumes: Task 4 的 `PlaceRecognizer.recognize`、`PlaceLibrary`、`make_embedder`
- Produces: `PerceptionWatcher.place` / `place_at` 被更新；`describe()` 加 `- 看起来在：云野`；CLI：`places add <地名> [--image 图]`、`places test [目录] [--model]`、`places bench [--model m ...]`（留一法：每张图和其余图比，打印每个模型的认对 / 认错 / 不说的张数和比例、每张耗时）

- [ ] **Step 1: 写失败的测试**

```python
def test_place_is_recognized_on_interval_and_after_scene_change():
    # 假 recognizer 记调用次数：t=0 跑一次；t=10 画面没变不跑；t=12 画面大变跑；t=13 大变但离上次 < 3 s 不跑；t=42 到 interval 跑
    assert w.place == "云野" and "看起来在：云野" in w.describe(12.0)
def test_place_is_not_checked_while_paused(): ...
def test_unrecognized_place_keeps_the_old_one_until_place_keep(): ...
def test_places_recognizer_masks_people_boxes(): # recognize 收到的 boxes = 这一帧的人物 / 标签 / 圆圈
def test_cli_places_add_and_test(tmp_path, capsys):  # --image 加两张、test 打印地名和相似度（假 embedder = thumb）
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**。`_watch_place` 在 `process()` 末尾、`not self.paused` 时跑；和上次认地图时的缩略图差 > `self.scene_change` 且离上次 ≥ `PLACE_GAP = 3.0` s，或离上次 ≥ `place_interval`；出错记日志吞掉。`_scene_watcher`：`cfg.places.enabled` 时 `load()`，0 张就警告"图库是空的（places add 加图）"、不挂
- [ ] **Step 4: 跑测试确认通过**，全量测试
- [ ] **Step 5: 提交** `feat(places): 感知层每 30 秒 / 画面大变后认一次地图；places add / test / bench`

### Task 6: `vision/gesture.py`：片段缓冲、谁该判、分类器、切片段、离线评估

**Files:**
- Create: `src/skydango/vision/gesture.py`
- Modify: `src/skydango/config.py`（`GestureConfig`、`Config.gesture`）、`config.example.toml`
- Test: `tests/test_gesture.py`、`tests/test_config.py`

**Interfaces:**
- Produces:
  - `GestureConfig`（字段见 Global Constraints）
  - `person_crop(img, box: Rect, size: int) -> np.ndarray`（框放宽 15% 裁出、缩放到 `size×size`）
  - `ClipBuffer(frames: int, fps: float)`：`push(t, crop) -> None`（离上一张 < 1/fps 就丢）、`ready() -> bool`、`clip() -> list[np.ndarray]`（最近 `frames` 张）
  - `eligible(name: str | None, box: Rect, width: int, ref_h: float, near: float, far: float) -> bool`
  - `OnnxGestureClassifier(path, labels, device="cpu")`：`classify(clip) -> tuple[str, float]`（输入 `(1, T, 3, S, S)` RGB 0~1 float32，输出 logits 或概率，做 softmax）
  - `extract_clips(frames: list[tuple[float, np.ndarray]], detector, out: Path, cfg: GestureConfig, conf: float) -> int`：跑检测 + `Tracker`，每条 `player` 轨迹按 `fps` 取帧、切成不重叠的 `frames` 张一段，存 `out/<序号>_t<开始秒>/00.jpg…`
  - `evaluate(root: Path, classifier, cfg: GestureConfig) -> dict`：`root/<标签>/<片段>/*.jpg`；按 `min_prob` 判（不到或 `none` 算没报），返回每个非 `none` 标签的 `{"tp", "fp", "fn", "precision", "recall"}` 和总的
- Classifier 协议：`classify(clip: list[np.ndarray]) -> tuple[str, float]`（测试用假的）

- [ ] **Step 1: 写失败的测试** — `ClipBuffer` 按 fps 丢帧、够 16 张才 ready；`eligible`：陌生人 / 远 / 偏到边上都 False；`evaluate` 在假分类器上算出的 precision / recall（例：wave 3 对 1 错报、bow 漏 1）；`extract_clips` 用假检测器（一个人 20 帧 @8fps）→ 1 段 16 张
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(gesture): 动作识别的数据工具——按轨迹切片段、离线评估、ONNX 分类器接口（研究性质，默认关）`

### Task 7: 动作识别接进感知层和身体，`gesture` 事件、提示词；CLI `perception clips / gesture-eval`

**Files:**
- Modify: `src/skydango/vision/perception.py`（构造参数 `gestures=None, gesture_cfg=None`；`_watch_gestures`；`pop_gestures`）、`src/skydango/brain/body.py`（`_watch_people` 里发 `gesture`）、`src/skydango/brain/events.py`（注释里的 kind 列表）、`src/skydango/brain/prompt.py`、`src/skydango/cli.py`
- Test: `tests/test_perception.py`、`tests/test_brain_body.py`、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 6 全部
- Produces: `PerceptionWatcher.pop_gestures() -> list[tuple[str, str]]`（(好友名, 动作标签)）；身体事件 `gesture`：`"懒洋洋大王对你挥手"`（中文名查 `gesture.names`，查不到用标签原文）；牵着手的那个人不报（同 approach）

- [ ] **Step 1: 写失败的测试**

```python
def test_friend_waving_in_front_is_reported_once_per_cooldown():
    # 假分类器永远 ("wave", 0.95)；好友在画面中间、近；8fps 喂 2 s → 第一次报；cooldown 内不再报；过了 30 s 再报
def test_gesture_skips_strangers_far_people_and_low_probability(): ...
def test_gesture_classifier_error_does_not_break_the_frame():  # 分类器抛异常 → 名字照样认出、没有事件
def test_body_turns_gestures_into_events(): # 假 env 有 pop_gestures → events 里有 "gesture" "懒洋洋大王对你挥手"
def test_brain_prompt_mentions_gestures_and_following_by_holding_hands():
    text = static_prompt(ReplyConfig())
    assert "对你挥手" in text and "牵" in text and "emote" in text
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**。`_watch_gestures(frame, players, now, width, height)`：对每个 `eligible` 的好友轨迹往 `track.data["clip"]`（`ClipBuffer`）里推裁剪；`ready()` 且离上次判 ≥ `interval` 就 `classify`；异常记日志。提示词（`BRAIN_RULES` 的"做动作"一节加一条）：`- 身体告诉你有人对你挥手、鞠躬时，想回礼就用 emote 回一个，不回也行。`；"说话"一节里"你现在不能走、飞、跑图、跟着别人走"那条改成：想跟着谁走，可以请对方牵你（"牵我一下"），牵上手他就能带着你走；别的走、飞、跑图还是做不到。CLI：`perception clips <录像目录> [-o datasets/gesture/_unlabeled] [--model]`、`perception gesture-eval <数据目录> [--model]`（打印每个动作的精确率 / 召回率，标出是否达到 90% / 60%）。`_scene_watcher`：`cfg.gesture.enabled` 时 `OnnxGestureClassifier` 传进去
- [ ] **Step 4: 跑测试确认通过**，全量测试
- [ ] **Step 5: 提交** `feat(brain): gesture 事件（好友对团子挥手 / 鞠躬），提示词说明可以回礼、想跟谁走就请他牵手`

### Task 8: 文档

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-perception-phase3-design.md`（状态、§7 写回决定、§3 模型输入输出约定和数据目录格式、§4 第 2 步暂缓的原因）、`docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`（§11 地名缺口、§15 三期状态）、`CLAUDE.md`（代码结构表、常用命令、感知层一节、运行目录 `unknown_names/`）、`docs/game-ops.md`（只加"未在真机验证"的待测项，不编结果）

- [ ] **Step 1: 改文档**；三期状态写"代码部分已完成；§1 / §2 / §5 验收待 GPU 机器和真机，§3 待录数据训练，§4 第 2 步待 `move` 工具"
- [ ] **Step 2: 全量测试** `python -m pytest -q`，全过
- [ ] **Step 3: 提交** `docs: 感知层三期代码部分完成；places / unknown-names / clips / gesture-eval 用法`
