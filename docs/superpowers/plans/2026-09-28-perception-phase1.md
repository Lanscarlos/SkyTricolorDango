# 感知层一期（代码部分）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 本次用户选了 **Native**（当前会话直接执行，不派 subagent）。

**Goal:** 把一期设计里不需要 GPU / 真机的部分全部做完：画面被挡时冻结状态、难例收集、`perception compare` / `augment` / `label --model --from-runs`，让 GPU 机器到手后只剩 M0 测速、M1 训练、M3 上线检查。

**Architecture:** 冻结（hold）做在 `PerceptionWatcher` 里，`EnvWatcher` 给空实现，身体 / Agent 在会挡画面的操作外面包 `env.held(...)`。难例收集是独立的 `vision/hardcases.py`，由 `PerceptionWatcher.process` 每帧喂数据、自己决定存不存。离线工具（compare / augment / label 扩展）的逻辑放在 `vision/` 下的纯函数模块，`cli.py` 只做参数和打印。

**Tech Stack:** Python 3.11+，numpy、opencv；测试 pytest（合成画面 + 假检测器 / 假 OCR）。

**Spec:** `docs/superpowers/specs/2026-09-28-perception-phase1-design.md`（总纲 `…-perception-yolo-architecture-v0.2.md`）

## Global Constraints

- 不往游戏里发任何新输入；所有新行为在 `[perception] enabled = false` 时对身体 / Agent 无影响（EnvWatcher 的 hold 系列是空操作）
- YOLO 类别顺序不变：`["player", "name_tag", "social_ring", "self", "player_unlit"]`
- 新配置（`[perception]`）：`hold_max = 60.0`、`occlusion_hold = 30.0`、`hardcases = true`、`audit_interval = 30.0`、`hardcase_max = 200`、`low_conf = 0.25`（§10 已由用户确认：难例默认开、旁路核对 30 s 一次）
- 集体消失的画面差异阈值用 `brain.scene_change`（0.25）
- 难例：两次保存至少隔 5 s；和上一张缩略图差异 < 0.03 不存；暂停期间不收集
- 文件名不能带 `:`（Windows）：难例图存成 `hard/<HHMMSS>_<原因>.jpg`
- 注释、日志、CLI 输出用中文，风格照现有代码
- 测试命令 `python -m pytest -q`

## Review Focus

1. **release 一个没 hold 过的原因 / 同一原因嵌套 hold 两次**（比如 `camera_move` 里又调了 `capture_around`）→ 不崩、不重复补时间；嵌套时两层都 release 才恢复（按计数，不按集合）。测试加在 Task 2。
2. **后台感知线程正在跑 process 时开始 hold** → 这一帧跑完可以更新状态，但恢复时的补时间按 hold 开始时刻算，不会把 `last_seen` 推到未来之后。测试加在 Task 2（`last_seen` 不超过 release 时刻）。
3. **暂停期间身体照样问 `nearby()` / `strangers()`**（blackout 持续十几秒，身体循环不停）→ 返回暂停开始那一刻的结果，不因为时间在走就把人判成走开。测试加在 Task 2。
4. **没有运行目录的命令**（`perception detect / bench`、`view`）→ 不建收集器、不写文件。测试加在 Task 6（`run_dir=None` 时 `_scene_watcher` 返回的 watcher 没有收集器）。
5. **录像文件名里没有时间**（用户自己截的图、别的工具录的）→ `compare` 明确报错说要用 `record` 录的目录，不静默按 0 秒全堆在一起。测试加在 Task 9。

---

### Task 1: 追踪器补时间 + 跨类别关联

**Files:**
- Modify: `src/skydango/vision/track.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces: `Tracker.shift(d: float) -> None`（所有轨迹 `first`、`last` 加 d）；`Tracker(buffer, min_iou, cross: frozenset[str] = frozenset(), cross_iou: float = 0.5)`；`Track.flips: int = 0`（类别在 cross 组内变过几次）

- [ ] **Step 1: 写失败的测试**

```python
def test_tracker_shift_keeps_tracks_alive():
    t = Tracker(buffer=1.0)
    first = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    t.shift(20.0)
    again = t.update([Detection("player", Rect(2, 0, 50, 100), 0.9)], 20.5)[0]
    assert again.id == first.id and again.first == 20.0

def test_tracker_cross_class_association_counts_flips():
    t = Tracker(cross=frozenset({"player", "player_unlit"}), cross_iou=0.5)
    a = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    b = t.update([Detection("player_unlit", Rect(1, 0, 50, 100), 0.9)], 0.1)[0]
    c = t.update([Detection("player", Rect(1, 0, 50, 100), 0.9)], 0.2)[0]
    assert a.id == b.id == c.id and c.cls == "player" and c.flips == 2

def test_tracker_without_cross_keeps_classes_apart():
    t = Tracker()
    a = t.update([Detection("player", Rect(0, 0, 50, 100), 0.9)], 0.0)[0]
    b = t.update([Detection("player_unlit", Rect(0, 0, 50, 100), 0.9)], 0.1)[0]
    assert a.id != b.id
```

- [ ] **Step 2: 跑测试确认失败** — `python -m pytest tests/test_perception.py -q -k tracker` → 3 个 FAIL
- [ ] **Step 3: 实现**：配对时同类别用 `min_iou`，不同类别但都在 `cross` 里用 `cross_iou`；匹配上且类别不同时改 `track.cls`、`flips += 1`
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): 追踪器补时间、player/player_unlit 跨类别关联`

### Task 2: 暂停计时（hold / release / held）

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/vision/env.py`、`src/skydango/config.py`
- Test: `tests/test_perception.py`、`tests/test_env.py`

**Interfaces:**
- Consumes: `Tracker.shift`
- Produces（`PerceptionWatcher` 和 `EnvWatcher` 都有，EnvWatcher 什么都不做）：
  `hold(reason: str) -> None`、`release(reason: str) -> None`、`held(reason: str)`（contextmanager）、`paused -> bool`（property）；
  `PerceptionConfig.hold_max: float = 60.0`、`occlusion_hold: float = 30.0`

规则（spec §4）：
- 原因按计数叠加；全部清零才恢复。release 不存在的原因 → 忽略
- 恢复时 `d = clock() - 暂停开始`，`last_seen` 每个值、`_strangers` 每条的时间、`labels` / `circles` 的时间、`Tracker.shift(d)` 都加 d；补完不超过恢复时刻
- 暂停期间 `observe()` 直接返回、`_loop` 跳过检测；`nearby(now)` / `strangers(now)` / `unlit(now)` / `describe(now)` 用暂停开始时刻代替 now
- `observe()` / `_loop` 里发现暂停超过 `hold_max` → `log.warning` 并全部 release

- [ ] **Step 1: 写失败的测试**（`watcher()` 测试工具加 `clock` 参数，用可变的假时钟）

```python
def test_hold_stacks_and_nests():            # hold a、hold b、hold a → release a、release b 仍暂停 → release a 才恢复；release "x" 不报错
def test_held_releases_on_error():           # with w.held("camera"): raise → w.paused is False
def test_hold_freezes_nearby_and_shifts_last_seen():
    # t=0 看到懒洋洋大王；t=1 hold；t=21 nearby(21) 仍含她；t=21 release；nearby(25) 仍含（1+20+5 边界内）；nearby(27) 不含
    # 且 release 后 w.last_seen["懒洋洋大王"] <= 21
def test_hold_skips_detection():             # 暂停期间 observe 不调 detector.detect（FakeDetector 计数）
def test_hold_keeps_tracks_no_reocr():        # t=0 标签读出名字；hold 20 s；恢复后同一位置的标签不再调 ocr（FakeOcr.calls 不变，ocr_votes=1）
def test_hold_max_auto_releases(caplog):      # hold_max=60：t=61 observe 后 paused is False，有警告日志
def test_env_watcher_hold_is_noop():          # EnvWatcher.held("x") 可用，paused 恒为 False（tests/test_env.py）
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（状态用 `self._holds: Counter`、`self._held_since: float | None`，改动在 `self._lock` 下）
- [ ] **Step 4: 跑测试确认通过**（含原有 perception / env 测试）
- [ ] **Step 5: Commit** `feat(perception): 画面被挡时暂停计时（hold / release / held）`

### Task 3: 集体消失规则

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: Task 2 的 hold / release；`brain.images.thumb` / `difference`
- Produces: `PerceptionWatcher(..., scene_change: float = 0.25)`；原因名 `"occlusion"`

规则：`process` 里数这一帧的人物 + 名字标签（过滤后，不含 self）。上一帧 ≥ 2、这一帧 0、且两帧缩略图差异 > `scene_change` → `hold("occlusion")`。
**occlusion 暂停期间检测照跑**（只有它是这样，别的原因仍跳过检测）：一旦又检测到人物或标签 → `release("occlusion")` 再照常处理这一帧；超过 `occlusion_hold` → release 并记 info 日志。

- [ ] **Step 1: 写失败的测试**

```python
def test_two_people_vanish_with_big_change_holds():   # 帧1：2 个 player（亮图）；帧2：什么都没有（黑图）→ w.paused
def test_one_person_vanishing_does_not_hold():        # 帧1：1 个 player；帧2：空 + 黑图 → not paused
def test_small_change_does_not_hold():                # 帧1：2 个 player；帧2：空，同一张图 → not paused
def test_occlusion_released_when_someone_is_back():   # 暂停后再来一帧有 name_tag → not paused，nearby 里的人没走
def test_occlusion_gives_up_after_limit():            # occlusion_hold=30：暂停后 31 s 仍空 → not paused
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): 多人同时消失且画面大变时自动暂停（玩家开了全屏界面）`

### Task 4: 身体 / Agent 调用 held

**Files:**
- Modify: `src/skydango/brain/body.py`、`src/skydango/agent.py`、`src/skydango/cli.py`（`_scene_watcher` 传 `scene_change=cfg.brain.scene_change`）
- Test: `tests/test_brain_body.py`、`tests/test_agent.py`

**Interfaces:**
- Consumes: `env.hold / release / held`（Task 2）

| 位置 | 做法 |
|---|---|
| `Body._watch_screen` | `blackout` 变 True → `env.hold("blackout")`；变 False → `env.release("blackout")` |
| `Body.camera_move` / `camera_reset` / `capture_around` | 真转镜头的那段包 `with self._held("camera")` |
| `Body.check_friend` | `friend_checker.check` 包 `_held("friend_tree")` |
| `Body.emote` | `emotes.perform` 包 `_held("wheel")` |
| `Body._watch_people` | `social.handle` 包 `_held("social")` |
| `Agent.step` / `Agent._emote` | `social.handle` 包 `"social"`、`emotes.perform` 包 `"wheel"` |

`Body._held(reason)` / `Agent._held(reason)`：env 为 None 或没有 `held` 属性（测试里的旧 FakeEnv）时返回 `contextlib.nullcontext()`。

- [ ] **Step 1: 写失败的测试**：`FakeEnv` 加 `holds: list[tuple[str, str]]` 记录 `("hold"/"release", 原因)` 和 `held()`

```python
def test_blackout_holds_env_until_screen_is_back(clock):  # 黑帧 → holds == [("hold","blackout")]；亮帧 → 追加 ("release","blackout")
def test_camera_move_is_wrapped_in_held(clock):            # live + FakeCamera：camera_move 后 holds == [("hold","camera"),("release","camera")]
def test_emote_and_social_are_wrapped_in_held(clock):      # live emote → "wheel"；有 request 时 _watch_people → "social"
def test_agent_emote_wrapped_in_held(clock):               # tests/test_agent.py：live 做动作 → env.holds 含 "wheel"
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑全部测试确认通过** — `python -m pytest -q`
- [ ] **Step 5: Commit** `feat(brain): 黑屏、转镜头、好友树、换轮盘、接互动时暂停感知计时`

### Task 5: 出框阈值和判定阈值分开

**Files:**
- Modify: `src/skydango/config.py`、`src/skydango/vision/perception.py`、`src/skydango/cli.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces: `PerceptionConfig.hardcases: bool = True`、`audit_interval: float = 30.0`、`hardcase_max: int = 200`、`low_conf: float = 0.25`；
  `detector_conf(p: PerceptionConfig) -> float`（`perception.py`，`hardcases` 开着返回 `min(p.low_conf, p.conf)`，否则 `p.conf`）；
  `PerceptionWatcher.last_low: list[Detection]`（这一帧 `score < cfg.conf` 的框，已过 `_filter`）；
  `PerceptionWatcher(..., hardcases=None)`：有收集器时每帧（非暂停）调 `hardcases.check(...)`（签名见 Task 6）

- [ ] **Step 1: 写失败的测试**

```python
def test_low_confidence_boxes_are_not_tracked_but_kept():
    # FakeDetector 出 player 0.9 和 player 0.3（conf=0.35）→ last_tracks 只有 1 条 player，last_low 有 1 条
def test_detector_conf():
    assert detector_conf(PerceptionConfig(conf=0.35, low_conf=0.25)) == 0.25
    assert detector_conf(PerceptionConfig(conf=0.35, hardcases=False)) == 0.35
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**；`cli._scene_watcher` 和 `_perception` 里 `make_detector` 的 conf 改用 `detector_conf(p)`
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): 检测器按 low_conf 出框，低置信度框留给难例收集`

### Task 6: 难例收集（`vision/hardcases.py`）

**Files:**
- Create: `src/skydango/vision/hardcases.py`
- Modify: `src/skydango/vision/env.py`（抽出 `match_names`）、`src/skydango/vision/perception.py`、`src/skydango/runlog.py`、`src/skydango/cli.py`
- Test: `tests/test_hardcases.py`（新）、`tests/test_runlog.py`

**Interfaces:**
- Consumes: `Tracker(cross=...)`、`Track.flips`（Task 1）、`last_low` / `hardcases=` 钩子（Task 5）
- Produces:
  - `env.match_names(lines: Iterable[OcrLine], names: list[str], min_score: float) -> list[tuple[str, OcrLine]]`（EnvWatcher._scan 改用它）
  - `RunDir.hard: Path`（`path / "hard"`，用到时才建）
  - `HardCaseCollector(folder: Path, cfg: PerceptionConfig, env_cfg: EnvConfig, log_roi, names: Callable[[], list[str]], ocr=None, background=True, clock=time.monotonic, wall=time.time)`
    - `check(frame, now, tracks: list[Track], low: list[Detection], seen: set[str], panel_visible: bool) -> None`
    - `saved: int`、`folder / "<HHMMSS>_<原因>.jpg"`、`folder.parent / "hard.jsonl"`（每行 `file`、`reason`、`detail`、`boxes`：`[{cls, x, y, w, h, score}]`）
  - `_scene_watcher(cfg, icons, dev, background, run: RunDir | None = None)`：`perception.enabled and hardcases and run is not None` 时建收集器；`PerceptionWatcher` 的 Tracker 用 `cross=frozenset({"player", "player_unlit"})`

判据（spec §5.1）：
- `ocr_only` / `yolo_only`：距上次核对 ≥ `audit_interval` 就拷一份帧，（后台线程）对 `env_cfg.roi` 区域（面板开着时去掉左边，同 EnvWatcher）整图 OCR，`match_names(..., 0.9)` 得到 OCR 名字集合；和这帧的 `seen` 比：OCR 有 YOLO 无 → `ocr_only`，YOLO 有 OCR 无 → `yolo_only`（同一帧两者都有时只存 `ocr_only`，detail 里两边名字都写）。`ocr` 为 None 时不核对
- `low_conf`：收集器自带一个 Tracker 追 `low` 里的框；某条低置信度轨迹 `last - first >= 0.5` → 存（每条轨迹只触发一次）
- `flicker`：`player` / `player_unlit` 新轨迹（`hits == 1`）出生时，记 (时间, 框)；3 s 内和它 IoU ≥ 0.3 的出生 ≥ 3 次 → 存
- `unlit_vs_player`：某条轨迹 `flips >= 2` → 存（每条轨迹只触发一次）
- 限额：`saved >= hardcase_max` 不存；距上次保存 < 5 s 不存；和上一张存下的图 `difference(thumb) < 0.03` 不存

- [ ] **Step 1: 写失败的测试**

```python
def test_match_names_filters_score_and_fuzzy_matches():          # test_env.py
def test_low_conf_track_saved_after_half_second(tmp_path):        # t=0、0.3 不存；t=0.6 存 1 张，原因 low_conf，hard.jsonl 一行
def test_flicker_saved_on_third_birth(tmp_path):                  # 同一位置 player 出生 3 次（中间隔空帧让轨迹断）→ 存 flicker
def test_unlit_vs_player_saved(tmp_path):                         # 一条轨迹 flips=2 → 存
def test_audit_ocr_only_and_yolo_only(tmp_path):                  # FakeOcr 读到懒洋洋大王、seen 为空 → ocr_only；反过来 → yolo_only；background=False
def test_limits_interval_dedupe_and_max(tmp_path):                # 5 s 内第二次不存；同一张图隔 6 s 不存（差异 < 0.03）；hardcase_max=1 后不存
def test_no_collection_while_held(tmp_path):                      # PerceptionWatcher 暂停时 process 不调 check（假收集器计数）
def test_no_collector_without_run_dir():                          # _scene_watcher(cfg 打开 perception + 假 detector, run=None) → watcher.hardcases is None
def test_run_dir_hard_path(tmp_path):                             # test_runlog.py：RunDir.hard == path / "hard"
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**；`_run_agent` / `_run_brain` 把 `run` 传给 `_scene_watcher`；退出时打印一行 `难例：存了 N 张 → runs/<…>/hard/`
- [ ] **Step 4: 跑全部测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): 运行时收集难例（旁路 OCR 核对、低置信度、闪烁、黑影来回变）`

### Task 7: 离线增强 `perception augment`

**Files:**
- Create: `src/skydango/vision/augment.py`
- Modify: `src/skydango/cli.py`
- Test: `tests/test_augment.py`（新）

**Interfaces:**
- Produces: `motion_blur(img, length: int) -> np.ndarray`（水平核）、`darken(img, gamma: float) -> np.ndarray`、
  `augment_dataset(root: Path, seed: int = 0, blur: float = 0.3, dark: float = 0.2) -> dict[str, int]`（`{"blur": n, "dark": n, "skipped": n}`）
- CLI：`perception augment <数据集目录> [--seed 0] [--blur 0.3] [--dark 0.2]`

规则：只处理 `images/train` 下不以 `_blur` / `_dark` 结尾的图；每张图用 `random.Random(f"{seed}:{stem}")` 决定抽不抽中、核长（9~25）、gamma（1.8~2.5）—— 结果不随目录里别的图变化；输出已存在就跳过；标注从 `labels/train/<stem>.txt` 原样复制（没有标注文件就写空文件）。`images/train` 不存在 → `FileNotFoundError`（CLI 打印中文提示）。

- [ ] **Step 1: 写失败的测试**

```python
def test_motion_blur_is_horizontal():         # 竖条纹图模糊后变平，横条纹图不变
def test_darken_lowers_brightness():          # 灰 128 图 gamma 2 → 均值 < 128
def test_augment_only_train_copies_labels_and_is_idempotent(tmp_path):
    # 20 张 train、2 张 val；blur=1.0 dark=1.0 → 20 个 _blur、20 个 _dark，标注内容相同；val 不变；再跑一次 blur=dark=0 个新增
def test_augment_same_seed_same_picks(tmp_path):   # blur=0.3：两个相同目录各跑一次，生成的文件名集合相同
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): perception augment —— 训练集加运动模糊 / 压暗样本`

### Task 8: `perception label --model / --from-runs`

**Files:**
- Modify: `src/skydango/vision/weaklabel.py`、`src/skydango/cli.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces: `merge_labels(weak: list[tuple[str, Rect]], predicted: list[Detection], min_iou: float = 0.5) -> list[tuple[str, Rect]]`（预测框和同类弱标注 IoU > 0.5 时丢预测框）；
  `hard_images(runs: Path) -> list[tuple[Path, str]]`（`runs/*/hard/*.jpg` → (路径, `<运行目录名>_<文件名去扩展>`)，按名字排序）
- CLI：`--model M`（`make_detector(M, …, conf=0.25)`）、`--from-runs`

- [ ] **Step 1: 写失败的测试**

```python
def test_merge_labels_prefers_weak_boxes():   # 弱 name_tag + 预测的同位置 name_tag + 预测 player → name_tag 只剩弱的那个，player 保留
def test_hard_images_collects_from_runs(tmp_path):  # runs/A/hard/1.jpg、runs/B/hard/2.jpg、runs/C（无 hard）→ [("A_1"), ("B_2")]
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**；`_perception_label` 的文件列表来源分两种（`--from-runs` 用 `hard_images`），stem 用返回的名字；有 `--model` 时把预测结果合并后写入，预览图里模型框用另一种颜色
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): label 支持模型预标注、从 runs/ 收集难例`

### Task 9: 离线对比 `perception compare`

**Files:**
- Create: `src/skydango/vision/compare.py`
- Modify: `src/skydango/cli.py`
- Test: `tests/test_compare.py`（新）

**Interfaces:**
- Produces:
  - `frame_time(path: Path) -> float | None`（`0012_006.00s.jpg` → 6.0）
  - `@dataclass FrameResult(file: str, t: float, env: set[str] | None, yolo: set[str], env_requests: set[tuple[str, str]], yolo_requests: set[tuple[str, str]], strangers: int, env_ms: float | None, yolo_ms: float)`（`env is None` = 现有方案这帧没扫）
  - `compare_frames(items: Iterable[tuple[float, str, np.ndarray, bool]], env: EnvWatcher, yolo: PerceptionWatcher, on_diff: Callable[[FrameResult, np.ndarray], None] | None = None) -> list[FrameResult]`（每项：时间、文件名、帧、面板开没开；两边都 `background=False`；`env.observe` 按自己的 interval 决定扫不扫，扫没扫看 `_last_scan == t`；认出的名字 = `last_seen == t` 的；请求 = `requests` 里 `time == t` 的；`env` 扫了且两边名字不同 → `on_diff`）
  - `summarize(results: list[FrameResult], keep_env: float, keep_yolo: float) -> dict`：`friends`（每个好友：现有方案扫描帧里认出几次、同样这些帧 YOLO 认出几次、不一致的帧名列表）、`requests`（每个 (谁, 哪种)：两边第一次出现的时间、差值）、`stranger_events`（陌生人数 0→有、有→0 的时间点）、`leave_events`（两边各自按 keep 模拟：名字最后一次看到后 keep 秒还没再看到 → 一个 leave，时间点 = 最后看到 + keep）、`timing`（两边平均 ms）
  - `report_md(summary: dict) -> str`
  - `side_by_side(frame, env_items: list[dict], yolo_items: list[dict]) -> np.ndarray`（左右拼图，框用 cv2 画，标签 ASCII 化）
- CLI：`perception compare <录像目录> [--model M] [--interval 3] [-o tmp/compare/<时间>]` → `report.md`、`summary.json`、`diff/<帧名>.jpg`；目录里没有一张带时间的图 → 打印"要用 record 录的目录（文件名带时间）"并退出码 1；个别没时间的跳过并提示

- [ ] **Step 1: 写失败的测试**

```python
def test_frame_time_parses_record_names():    # "0012_006.00s.jpg" → 6.0；"shot.png" → None
def test_compare_frames_env_scans_on_interval_and_diff_callback():
    # 假 EnvWatcher（FakeOcr 读到懒洋洋大王）interval=3、假 YOLO（每帧都认出）；t=0,1,2,3,4
    # → env 在 t=0、3 扫（其他帧 env is None）；yolo 每帧都有名字；有一帧 env 没认出时 on_diff 被调
def test_summarize_requests_delay_stranger_and_leave_events():
    # 手工构造 FrameResult 序列：YOLO t=1 报 (懒洋洋大王, hand)、env t=3 报 → delay 2.0；
    # strangers 0,1,1,0 → 两个事件；YOLO 最后看到 t=2、keep=5、序列到 t=10 → leave 在 7.0
def test_report_md_mentions_every_section():  # 含 "好友认出率"、"互动请求"、"stranger"、"leave"、"耗时"
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: Commit** `feat(perception): perception compare —— 同一批录像上对比 YOLO 和整图 OCR`

### Task 10: 配置模板和文档

**Files:**
- Modify: `config.example.toml`、`CLAUDE.md`、`docs/superpowers/specs/2026-09-28-perception-phase1-design.md`、`docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`

- [ ] **Step 1:** `config.example.toml` 的 `[perception]` 加新配置（带中文注释）
- [ ] **Step 2:** CLAUDE.md：代码结构表加 `hardcases.py` / `augment.py` / `compare.py`；运行目录表加 `hard/`、`hard.jsonl`；常用命令加 `perception compare / augment`、`label --model / --from-runs`
- [ ] **Step 3:** 一期文档：状态改为"代码部分已完成，待 GPU 机器"；§10 写上用户的确认；文件名格式改成 `<HHMMSS>_<原因>.jpg`（Windows 不能用冒号）
- [ ] **Step 4:** 总纲 §15 一期状态、§13 代码表补新模块、§12「遮挡时误报」标已做
- [ ] **Step 5:** `python -m pytest -q` 全绿
- [ ] **Step 6: Commit** `docs: 感知层一期代码部分完成`，合并进 main 并推送（CLAUDE.md 规矩）
