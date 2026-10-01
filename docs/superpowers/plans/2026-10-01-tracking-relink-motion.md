# 追踪器升级 + 失踪好友接回 + 运动方向 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 好友走远、名字标签淡掉时，YOLO 感知层的轨迹少断、断了在 `keep` 内按位置接回成"像小明"，不再冒假的"陌生人走过来""小明走开了"；顺带给每个人算运动方向。

**Architecture:** 在现有 `vision/track.py` 的 IoU 贪心追踪上加两段匹配（低分框只续不开）、速度预测、中心距离兜底和画面平移补偿（`estimate_shift`，相位相关）；感知层（`vision/perception.py`）加轨迹续命、失踪记录 + 按位置接回（复用认装扮的 `maybe` 那套保守后果）、`motion_hist` / `motion_of`；身体动镜头时调 `env.camera_moved`；新的离线工具 `perception track-eval` 用来定阈值。

**Tech Stack:** Python 3.11+、numpy、OpenCV（`cv2.phaseCorrelate`）、pytest（合成框 / 合成画面 + 假检测器）。

**Spec:** `docs/superpowers/specs/2026-10-01-tracking-relink-motion-design.md`

## Global Constraints

- 新开关**全关时逐字等于现在的行为**（`sticky_names` / `track_low` / `track_predict` / `track_pan` / `relink` / `motion` 都 `False`），有测试保证；`gesture.py`、`hardcases.py` 里另建的 Tracker 不改
- `Tracker.__init__` 新参数都是关键字、默认 `center_gate=0.0`、`predict=False`；`update` 新参数 `low=()`、`shift=None`、`prune=True`
- 速度：指数平均 `α = 0.5`；时间差 > 0.5 秒不更新；预测最多往前 0.5 秒
- 中心距离候选：中心距离 ≤ `center_gate` × 预测框高、框高比 0.67~1.5、跨类别不走；**永远排在 IoU 候选之后**
- 平移估计：缩略图 1/8 灰度、画面上半、面板开着去掉左 1/3、人物框填均值、Hanning 窗；`PAN_MIN_RESPONSE = 0.1`；位移 > 缩略图宽 1/3 → `None`
- 接回：中心距离 ≤ `min(0.6 + 0.5 × dt, 2.5)` × 最后框高、框高比 0.5~2、`first >= lost.last − 0.2`、预测最多 1.0 秒、近的比远的不到 1.3 倍算歧义
- 运动方向：窗口 `motion_window` 1.5 秒、样本 ≥ 3 且覆盖 ≥ 60% 窗口、`motion_grow` 0.15、`motion_side` 0.6、防抖 `motion_hold` 0.5；结论只有 `走近 / 走远 / 往左走 / 往右走 / 站着`
- 配置默认值照 spec §7 表：`sticky_names = true`、`track_low = true`、`track_predict = true`、`track_center_gate = 0.6`、`track_pan = true`、`relink = true`、`motion = true`、`motion_window = 1.5`、`motion_grow = 0.15`、`motion_side = 0.6`、`motion_hold = 0.5`
- 测试命令 `python -m pytest -q`；不碰真机；临时文件放 `tmp/`

## Review Focus

1. **同一帧两次 `tracker.update`**（`_far_tags` 第二次用 `prune=False`）：`dropped` 只在 `prune=True` 时清空重填，第二次调用不能把第一次的掉线记录冲掉、也不能再加一次平移 —— Task 3 的测试 `test_far_tags_second_update_keeps_dropped` 钉住
2. **接回之后标签亮出来是别的名字**：名字说了算，`maybe` / `maybe_by` 都要摘、不报 appearance 冲突 —— Task 5 `test_relinked_track_takes_other_tag_name`
3. **两个好友同时断、只冒出一条新轨迹**（交叉走过）：有歧义不接，宁可判陌生人 —— Task 5 `test_relink_ambiguous_two_lost_one_candidate`
4. **暂停（held）期间失踪记录不过期、恢复后速度清零** —— Task 5 `test_relink_survives_pause`、Task 3 `test_resume_zeroes_velocity`
5. **纯色 / 黑屏画面的平移估计**不能返回乱值（NaN、巨大位移）—— Task 2 `test_shift_none_on_flat_image`

---

### Task 1: 追踪器：两段匹配、速度预测、中心距离、`dropped` / `prune`

**Files:**
- Modify: `src/skydango/vision/track.py`
- Test: `tests/test_track_motion.py`（新）

**Interfaces:**
- Produces:
  - `Track` 多 `vx: float = 0.0`、`vy: float = 0.0`、`vh: float = 0.0`（像素 / 秒）、`weak_hits: int = 0`、`pan: tuple[float, float] = (0.0, 0.0)`（上次匹配以来累计的画面平移）
  - `Tracker(buffer=1.0, min_iou=0.3, cross=frozenset(), cross_iou=0.5, *, center_gate: float = 0.0, predict: bool = False)`
  - `Tracker.update(dets, now, *, low=(), shift: tuple[float, float] | None = None, prune: bool = True) -> list[Track]`
  - `Tracker.dropped: list[Track]`（`prune=True` 的 update 开头清空，填这次删掉的轨迹；`prune=False` 不动它）
  - `Tracker.predicted(track, now) -> list[Rect]`（带平移的预测框；平移非零时再加一个不带平移的）
  - `Tracker.calm(until: float) -> None`：所有轨迹速度清零，`now < until` 的 update 不更新速度

- [ ] **Step 1: 写失败的测试**（`tests/test_track_motion.py`，`P(x, y=400, w=30, h=60, s=0.9)` 造 player 检测）
  - `test_defaults_unchanged`：`Tracker()` 跑 test_perception 里那组（重叠接上、跨类不接、断 2 秒新开）结果一样，`t.vx == 0`
  - `test_low_box_only_extends`：`update([P(100)])` 后 `update([], 0.1, low=[P(103, s=0.3)])` 返回同 id、`weak_hits == 1`；`update([], 0.2, low=[P(900, s=0.3)])` 不开新轨迹（`len(tracker.tracks) == 1`）
  - `test_center_gate_links_small_box_moving_past_iou`：w=20 h=40 的框每 0.15 秒右移 15 px：`center_gate=0`、`predict=False` 时第二帧 id 变；`center_gate=0.6, predict=True` 时 5 帧都是同一个 id
  - `test_iou_candidate_beats_center_candidate`：一条轨迹、两个检测（一个 IoU 0.5、一个中心距离 0、但 IoU 0.29 由框高 ×1.4 造成）→ 接的是 IoU 那个
  - `test_shift_keeps_track_when_whole_frame_pans`：框 x=100 → 下一帧 x=400，传 `shift=(300, 0)` → 同 id；不传 → 新 id
  - `test_prediction_capped_at_half_second`：速度 vx=100 px/s 的轨迹，`predicted(t, last + 2.0)[0].x == box.x + 50`
  - `test_velocity_not_updated_after_gap_or_calm`：间隔 0.8 秒的匹配 `vx` 不变；`calm(until)` 后 `vx == 0` 且 until 前的匹配不改速度
  - `test_dropped_and_prune_false`：轨迹过期后 `update([], t, prune=False)` 不删、`dropped == []`；再 `update([], t)` 删掉、`dropped` 里有它
  - `test_shift_accumulates_on_unmatched_track`：两帧都没匹配、`shift=(10,0)`、`(5,0)` → `track.pan == (15, 0)`；匹配后 `pan == (0, 0)`

- [ ] **Step 2: 跑测试，确认失败**：`python -m pytest tests/test_track_motion.py -q` → FAIL（`unexpected keyword` / 属性不存在）

- [ ] **Step 3: 实现**
  - update 顺序：① `shift` 不为 None → 所有轨迹 `pan += shift` ② `prune` → 删过期轨迹进 `dropped` ③ 高分框配对 ④ 剩下的轨迹和 `low` 配对（`weak_hits += 1`）⑤ 剩下的高分框开新轨迹。返回：按 `dets` 顺序，低分框续上的附在后面
  - 配对分数用元组 `(1, iou)` / `(0, 0.3 × (1 − d / gate))` 降序贪心，保证 IoU 候选永远优先
  - `predicted`：`last box + (vx, vy, vh) × min(now − last, 0.5)`（`predict=False` 时为 0），中心不动、高度变化左右上下各分一半；再 `+ pan`；`pan != (0, 0)` 时多给一个不加 pan 的
  - 匹配上时：`0 < dt <= 0.5` 且 `now >= calm_until` → 样本 `((新中心 − 旧中心 − pan) / dt, (新高 − 旧高) / dt)`，`v = 0.5·v + 0.5·样本`；然后 `pan = (0, 0)`

- [ ] **Step 4: 跑测试通过**：`python -m pytest tests/test_track_motion.py tests/test_perception.py -q` → PASS

- [ ] **Step 5: 提交** `feat(perception): 追踪器两段匹配、速度预测、中心距离兜底`

### Task 2: 画面平移估计 `estimate_shift`

**Files:**
- Modify: `src/skydango/vision/track.py`
- Test: `tests/test_track_motion.py`

**Interfaces:**
- Produces: `estimate_shift(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray | None) -> tuple[float, float] | None`（灰度缩略图、同尺寸；`mask` 为 bool，True = 可用；返回缩略图像素、内容往右 / 下移为正）；`PAN_MIN_RESPONSE = 0.1`

- [ ] **Step 1: 失败的测试**
  - `test_shift_estimates_synthetic_pan`：随机纹理 240×67（`np.random.default_rng(0)` 高斯模糊过）、`cur = np.roll(prev, 7, axis=1)` → 结果 `dx ≈ 7`（误差 < 1）、`dy ≈ 0`
  - `test_shift_none_on_flat_image`：全 128 → `None`；全 0 → `None`
  - `test_shift_ignores_masked_region`：背景不动、mask 掉的左 1/3 里贴一块往右移 20 px 的亮块 → `|dx| < 1`
  - `test_shift_none_when_too_large`：`roll` 100（> 宽 1/3）→ `None`

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：float32；mask 外两张都填 mask 内均值；可用像素标准差 < 1 → `None`；`cv2.createHanningWindow`；`cv2.phaseCorrelate(prev, cur, win)`；`response` 不是有限数或 < `PAN_MIN_RESPONSE` → `None`
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: 提交** `feat(perception): 画面平移估计（相位相关）`

### Task 3: 配置 + 感知层接上新追踪器、平移、镜头事件

**Files:**
- Modify: `src/skydango/config.py`（`PerceptionConfig`，spec §7 全部 11 项）
- Modify: `src/skydango/vision/perception.py`、`src/skydango/vision/env.py`、`src/skydango/cli.py`（建 watcher 时传 `camera_settle=cfg.track.settle`）
- Test: `tests/test_perception_tracking.py`（新）

**Interfaces:**
- Consumes: Task 1、2
- Produces:
  - `PerceptionWatcher(..., camera_settle: float = 0.6)`；`self._pan: tuple[float, float]`（累计平移，整图像素）；`self.last_shift: tuple[float, float] | None`（这一帧用的平移，track-eval 用）；`self.last_dropped: list[Track]`（这一帧第一次 update 删掉的轨迹）
  - `PerceptionWatcher.camera_moved(at: float, kind: str) -> None`；`EnvWatcher.camera_moved` 空实现
  - 每个人物轨迹 `data["pan_at"] = self._pan`（每帧刷新，失踪记录用）
  - `detector_conf(cfg)`：`hardcases` 或 `track_low` 时返回 `min(low_conf, conf)`

- [ ] **Step 1: 失败的测试**（复用 `tests/test_perception.py` 的 `FakeDetector` / `FakeOcr` / `watcher()`，import 进来）
  - `test_all_switches_off_is_unchanged`：同一段合成序列（好友带标签 → 标签消失、分数降到 0.3 → 回 0.9），`all_off` 配置和"旧逻辑"期望一致：标签消失后 1 秒轨迹断、新轨迹 `stranger_after` 后判陌生人、`nearby` 在 keep 后变空
  - `test_low_conf_box_continues_track_in_watcher`：默认配置下同一序列轨迹 id 不变
  - `test_detector_conf_low_when_track_low`：`detector_conf(PerceptionConfig(hardcases=False, track_low=True)) == 0.25`；都关 → `0.35`
  - `test_pan_feeds_tracker`：纹理画面整体右移 160 px（不经人物框）、人物框也右移 160 px → 同一 id；`track_pan=False` → 新 id
  - `test_far_tags_second_update_keeps_dropped`：一帧里第一次 update 删了轨迹 A、`_far_tags` 又 update 一次 → `last_dropped` 里还有 A
  - `test_camera_moved_zoom_clears_hist_and_calms`：`camera_moved(t, "zoom")` 后 settle 内的帧 `hist` / `motion_hist` 为空、速度 0；`"turn"` 不清
  - `test_resume_zeroes_velocity`：`held()` 前后：恢复后所有轨迹 `vx == vy == vh == 0`、平移缩略图作废（恢复后第一帧 `last_shift is None`）

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
  - `__init__`：`Tracker(cfg.track_buffer, cfg.track_iou, cross=…, center_gate=cfg.track_center_gate if cfg.track_predict else 0.0, predict=cfg.track_predict)`
  - `_pan_step(frame, dets, panel_visible) -> tuple|None`：缩略图存 `self._pan_thumb`；mask 用这一帧的人物 / 团子 / 名字标签框（缩放到缩略图）；结果 ×8 加进 `self._pan`
  - `process`：`shift = self._pan_step(...) if cfg.track_pan else None`；`self.tracker.update(dets, now, low=low if cfg.track_low else (), shift=shift)`；`self.last_dropped = list(self.tracker.dropped)`；`_far_tags` 的 update 加 `prune=False`；`now < self._quiet_until` 时清 players 的 `hist` / `motion_hist`
  - `camera_moved`：`kind in ("zoom", "move", "spin")` → `self._quiet_until = max(…, at + camera_settle)`、`tracker.calm(at + camera_settle)`
  - `_resume`：`tracker.calm(now)`、`self._pan_thumb = None`、清 `motion_hist`
- [ ] **Step 4: 跑测试通过**（连同 `tests/test_perception*.py`）
- [ ] **Step 5: 提交** `feat(perception): 感知层用低分框、速度预测和画面平移续轨迹；镜头事件`

### Task 4: 轨迹续命（`sticky_names`）

**Files:**
- Modify: `src/skydango/vision/perception.py`（`process` 里陌生人那段循环之后）
- Test: `tests/test_perception_tracking.py`

- [ ] **Step 1: 失败的测试**
  - `test_sticky_keeps_friend_nearby_while_track_alive`：好友带标签 1 秒、之后 10 秒只有同一位置的人物框 → `nearby` 一直有他
  - `test_sticky_off_drops_after_keep`：同上 `sticky_names=False` → 标签消失 5 秒后 `nearby` 为空
  - `test_sticky_ends_after_track_dropped`：标签消失后人也消失 → 轨迹删掉（1 秒）后再过 `keep`（5 秒）才不在
  - `test_sticky_ignores_maybe`：只有 `maybe`（无 `tagged`）的轨迹不靠续命刷新（走现有 `maybe` 规则）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**：`cfg.sticky_names` 且 `data["tagged"]`、`data["name"]`、`now − t.last <= PEOPLE_STALE` → `last_seen[name] = now`
- [ ] **Step 4: 通过**
- [ ] **Step 5: 提交** `feat(perception): 挂过名字的轨迹没断就算在身边（续命）`

### Task 5: 失踪好友接回（`relink`）

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception_tracking.py`

**Interfaces:**
- Consumes: Task 3 的 `last_dropped`、`data["pan_at"]`、`self._pan`
- Produces: `Lost` dataclass（`name, box, vx, vy, vh, last, pan`）；`self._lost: dict[str, Lost]`；轨迹 `data["maybe_by"] = "relink"`

- [ ] **Step 1: 失败的测试**（序列：好友带标签站在 x=800；然后检测器 1.2 秒什么都不出（轨迹删掉）；之后在附近冒出没标签的人）
  - `test_relink_within_gate`：冒在 x=830 → 那条轨迹 `data["maybe"] == 名字`、`maybe_by == "relink"`、`stranger` 为假、`people()` 里 `sure is False`、`nearby` 一直有他（不冒 leave）
  - `test_relink_rejects_far_or_late_or_unlit`：冒在 x=1600 / 超过 `keep` 才冒 / 冒的是 `player_unlit` → 都不接
  - `test_relink_cancelled_when_tag_shown_elsewhere`：同名标签此刻挂在别人头上 → 不接、记录作废
  - `test_relink_ambiguous_two_lost_one_candidate`：两个好友同时断、只冒一条 → 不接；一个好友断、冒两条距离差不到 1.3 倍 → 不接
  - `test_relinked_track_confirmed_by_tag` / `test_relinked_track_takes_other_tag_name`：标签亮出来 → `name` 是标签上的、`maybe` / `maybe_by` 都没了
  - `test_relinked_track_can_relink_again`：接回的轨迹再断 → 再冒出来还能接
  - `test_relink_survives_pause`：断了以后 `held()` 10 秒，恢复后 1 秒内冒出来 → 照接
  - `test_appearance_does_not_drop_relinked`（挂 `AppearanceBook`，复用 `tests/test_perception_appearance.py` 的 `make()`）：外观连续不像也不摘
  - `test_relink_off`：`relink=False` → 不接，按现有规则判陌生人
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**
  - `last_dropped` 里 `cls == "player"`、名字取 `tagged` 的 `name`，或 `maybe_by == "relink"` 的 `maybe` → `_lost[name] = Lost(...)`（`pan = data["pan_at"]`）；过期 `now − last > keep` 删
  - 顺序：标签挂完 → 摘错挂的名字（`maybe_by == "relink"` 且名字在 `shown` 里也摘）→ `_relink(players, shown, now)` → 认装扮 → 判陌生人
  - 标签有名字挂上时 `maybe_by == "relink"` 的轨迹 pop `maybe` / `maybe_by`；`_appearance_identify` 里 `maybe_by == "relink"` 的跳过"连续不像就摘"
  - `_relink`：`shown` 里或别的活轨迹 `name` / `maybe` 占着的名字 → 记录作废；候选和预测中心按 Global Constraints；先收集所有过线的 (记录, 候选, 距离)，候选对上 ≥ 2 条记录 / 记录的最近两个候选距离比 < 1.3 → 跳过；接上 pop `sid`、INFO 日志"轨迹 %d 像是 %s（断了 %.1f 秒，按位置接回）"
  - `_resume`：`lost.last = min(lost.last + d, now)`
- [ ] **Step 4: 通过**（连同 `tests/test_perception_appearance.py`）
- [ ] **Step 5: 提交** `feat(perception): 断掉的好友轨迹按位置接回成"像他"`

### Task 6: 运动方向（`motion`）

**Files:**
- Modify: `src/skydango/vision/perception.py`（`motion_of`、`_watch_motion`、`people()`、`overlay()`）、`src/skydango/vision/people.py`、`src/skydango/vision/viewer.py`（悬停文字）
- Test: `tests/test_perception_tracking.py`、`tests/test_people_motion.py`（新）

**Interfaces:**
- Produces: `motion_of(hist: list[tuple[float, float, float]], now: float, cfg: PerceptionConfig) -> str | None`；`Person.motion: str | None = None`；overlay 条目可选 `"motion"` 键

- [ ] **Step 1: 失败的测试**
  - `test_motion_of_five_outcomes`：1.5 秒 10 个样本，框高 100→130 → `走近`；100→80 → `走远`；x 右移 1 个身高 → `往右走`；左移 → `往左走`；不动 → `站着`
  - `test_motion_of_needs_samples`：2 个样本 / 覆盖 0.6 秒 → `None`
  - `test_motion_debounce_and_none_immediate`：感知层里结论变了 0.3 秒还没换、0.6 秒换了；样本不够立刻 `None`
  - `test_motion_uses_pan_compensated_x`：画面平移 +100、人在屏幕上也 +100 → `站着`
  - `test_motion_cleared_after_zoom`：`camera_moved(t, "zoom")` 后 `motion` 变 `None`
  - `test_describe_person_motion`：`Person(..., motion="走远")` → `"小明（左边·中，正在走远）"`；`motion="站着"` / `None` 逐字照旧；`sure=False` → `"像小明（没看到名字，左边·中，正在走远）"`
  - `test_motion_off`：`motion=False` → `people()` 的 `motion` 全 `None`
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**：`_watch_motion(players, now)` 和 `_watch_approach` 一起在 `not self.paused` 时调；`data["motion_hist"]` 存 `(t, h, cx − self._pan[0])`；防抖用 `data["motion_cand"] = (结论, 起始时间)`；`_describe_person` 的 `where` 在 `motion in ("走近","走远","往左走","往右走")` 时加 `，正在{motion}`；viewer 悬停 `[b.desc, b.motion].filter(Boolean).join(" · ")`
- [ ] **Step 4: 通过**
- [ ] **Step 5: 提交** `feat(perception): 人物运动方向（走近 / 走远 / 往左 / 往右 / 站着）`

### Task 7: 身体动镜头时告诉感知层

**Files:**
- Modify: `src/skydango/brain/body.py`
- Test: `tests/test_body_camera_moved.py`（新，参照 `tests/test_brain_track.py` 里造 Body 的方式）

**Interfaces:**
- Produces: `Body._camera_moved(kind: str, at: float | None = None) -> None`：设 `_camera_moved_at`，`env` 有 `camera_moved` 就调（出错只记日志）

- [ ] **Step 1: 失败的测试**：假 env 记录调用；分别调 `camera_move("left")` → `"turn"`、`camera_move("zoom_in")` → `"zoom"`、`move("W")` → `"move"`、`camera_reset` → `"zoom"`（复位会重放拉近拉远）、`sweep_around` → `"spin"`；注意力 nudge 一次 → `"turn"`
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**：所有 `self._camera_moved_at = …` 换成 `_camera_moved(kind)`；`move` 工具真走了之后补 `_camera_moved("move")`；注意力 `nudge` 成功后 `_camera_moved("turn")`；peek 按过 `zoom_once` 用 `"zoom"` 否则 `"turn"`；`camera_move` 的 `zoom_*` 用 `"zoom"`、其余 `"turn"`；look_around / sweep_around 用 `"spin"`；track 在跑的那条照旧只设时间、再调 `"turn"`
- [ ] **Step 4: 通过**（连同 `tests/test_brain*.py`）
- [ ] **Step 5: 提交** `feat(brain): 动镜头时告诉感知层（camera_moved）`

### Task 8: 离线评估 `perception track-eval`

**Files:**
- Create: `src/skydango/vision/trackeval.py`
- Modify: `src/skydango/cli.py`（子命令 + `_perception_track_eval`，照 `_perception_compare` 建 watcher）
- Test: `tests/test_trackeval.py`（新）

**Interfaces:**
- Produces:
  - `baseline(cfg: PerceptionConfig) -> PerceptionConfig`（六个开关全关的副本）
  - `subsample(items, fps) -> Iterator`（按录像时间抽帧）
  - `evaluate(items, watcher) -> dict`：每个好友 `breaks: {原因: 次数}`（`低分框 / 位移 / 画面平移 / 漏检`）、`false_leaves`、`wronged`（判过陌生人、后来挂上他的标签）、`relinks: {对, 错, 未证实}`，以及 `timeline: list[str]`（运动方向，每条轨迹一行）
  - `report_md(base: dict, cur: dict, meta: dict) -> str`（并排两列）

- [ ] **Step 1: 失败的测试**：合成"好友带标签 → 标签消失、框变小、分数 0.3 → 分数回 0.9、标签再亮"；`evaluate` 用基线配置报 `breaks["低分框"] >= 1`、`wronged == 1`；当前配置都是 0；`report_md` 里有"基线""当前配置"和好友名；`subsample` 按 `fps=2` 从 0.1 秒间隔的 20 帧里取 ~10 帧
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**：时钟同 `compare_frames`（录像时间）；断开原因在一条有名字的轨迹**第一次没接上**的那一帧判（预测框附近 1 个框高内有低分框 → 低分框；2 个框高内有高分 player → 位移；`last_shift` 的 `|dx|` > 框宽 → 画面平移；否则漏检），被删时计数；假走开 = 名字离开 `nearby` 后 `keep` 内又出现；CLI：`perception track-eval <dir> [--model] [--fps 6.5] [-o]`，报告写 `tmp/track-eval/<时间>/report.md`
- [ ] **Step 4: 通过**
- [ ] **Step 5: 提交** `feat(perception): track-eval 离线评估追踪和接回`

### Task 9: 文档、全量测试、合并

**Files:**
- Modify: `CLAUDE.md`（YOLO 感知层一节加"追踪和接回"一段 + 常用命令加 `perception track-eval`；代码结构表 `track.py` / `trackeval.py`）、`config.example.toml`（如有 `[perception]` 段则加新项）

- [ ] **Step 1**：写文档（标明"还没在真机标定，spec §8 真机验证四步"）
- [ ] **Step 2**：`python -m pytest -q` 全绿
- [ ] **Step 3**：提交、请求代码评审、按意见修
- [ ] **Step 4**：合并进 main 并推送（CLAUDE.md 规矩）
