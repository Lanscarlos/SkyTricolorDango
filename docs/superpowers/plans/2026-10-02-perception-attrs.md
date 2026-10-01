# 感知层第二层：人物属性 + 复核 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO 人物框之后加一层"裁图 → 冻住的 DINOv2 → 线性头"，第一个头"外形"负责复核误框、捞回低分框、和 YOLO 一起判点没点火、认出先祖 / 共享空间 / 变身；配齐裁图、Claude 初分、标注页、训练和回放评估。

**Architecture:** 运行时全在新模块 `vision/attrs.py`（`AttrModel` 读 `.npz` 头 + 复用 `OnnxEmbedder` 主干；`PersonAttrs` 按轨迹裁图、投票、放行、黑影滞回），由 `PerceptionWatcher` 持有、在感知线程里调用；`Tracker` 加 `open_low` 让人物类低分框开"待复核"轨迹。离线工具分到 `vision/attrs_data.py`（裁图、写回、Claude 初分）和 `vision/attrs_train.py`（切分、特征缓存、numpy 逻辑回归、整帧回放报告）。

**Tech Stack:** Python 3.13、numpy、OpenCV、onnxruntime（DINOv2-small ONNX）、ultralytics（只在离线工具里，延迟导入）、`claude -p`（经 `vision/assist.Reviewer`）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-02-perception-attrs-design.md`

## Global Constraints

- `[attrs] enabled = false`（默认）时：检测阈值、`Tracker` 行为、`process()` 结果、`people()` / `overlay()` / status 文字**逐字照旧**
- 外形类别名和顺序固定：`FORMS = ("not_person", "lit", "unlit", "spirit", "shared", "morph")`；人形 = 后五类
- 默认值：`every` 0.5、`max_crops` 4、`votes` 5、`accept` 0.6、`reject` 0.7、`reject_n` 3、`yolo_w` 0.5、`flip_votes` 3、`max_errors` 10、`min_per_class` 20、`model` `"models/attrs.npz"`、`backbone` `"models/dinov2-small.onnx"`
- 低分框门槛不另设，用 `[perception] low_conf`（0.25）；离线工具的 `--conf` 默认 0.2
- 训练输出 `models/attrs-<YYYYMMDD>.npz`，不覆盖 `[attrs] model` 指的文件（覆盖要 `--force`）；报告 `tmp/attrs-train/<YYYYMMDD-HHMMSS>/report.md`
- 写回 `datasets/sky` 前先备份 `labels/` 到 `_backup/labels-<YYYYMMDD-HHMMSS>`
- 单元测试不加载真模型、不调 Claude、不碰 GPU：假主干（固定向量）+ 手写线性头；跑法 `.venv\Scripts\python.exe -m pytest -q`（没有 .venv 的机器用 `python -m pytest -q`）
- 中文注释 / 日志 / 用户可见文字，和周围代码一致
- **Task 5 起要动 `vision/perception.py`：先确认 `feat/light-flame` 已经合进 main**（`git log main --oneline | grep -i 火焰`），没合就停下问用户

## Review Focus

1. **只有低分框的人一直站着**：待复核轨迹靠低分框续着，复核通过一次就刷新 `strong_last`；没通过的不能因为 `LOW_ONLY_MAX` 被续命 —— Task 5 测"复核前不进 `players`、不续命"
2. **主干和 npz 不配套**（换了 DINOv2 文件、尺寸不同）：启动时退回、不崩，`enabled` 关着一样 —— Task 1 测
3. **同一条轨迹被撤后又冒出高分检测**：不翻案（`rejected` 一直为真）—— Task 3 测
4. **聊天面板挡住一半的人**：不裁图、不投票，已有结论保留 —— Task 2 测
5. **标注页并发 / 文件名里带奇怪字符**：确认、撤销只按目录遍历比对名字，不把请求里的名字拼进路径（同动作标注页）—— Task 8 测 `../x.jpg`、`con.jpg`

---

### Task 1: 配置和 `AttrModel`

**Files:**
- Modify: `src/skydango/config.py`（加 `AttrsConfig`，`Config.attrs`）
- Create: `src/skydango/vision/attrs.py`
- Test: `tests/test_attrs.py`

**Interfaces:**
- Produces:
  - `config.AttrsConfig`（dataclass，字段见 Global Constraints，外加 `device: str = ""` —— 空 = 跟 `[perception] device`）；`Config.attrs: AttrsConfig`
  - `attrs.FORMS`、`attrs.PERSON_FORMS = FORMS[1:]`
  - `attrs.crop(img: np.ndarray, box: Rect, pad: float, size: int) -> np.ndarray`：框四周各放 `pad` 倍宽 / 高，再把短边补到和长边一样（以框中心为中心），越界部分填黑，缩放到 `size×size`
  - `class Embedder(Protocol)`：`size: int`、`key: str`、`embed(img) -> np.ndarray`（单位向量）
  - `class AttrModel`：`__init__(self, data: dict[str, np.ndarray], embedder: Embedder)`；`heads: list[str]`；`labels(head) -> list[str]`；`applies_to(head) -> list[str]`；`pad(head) -> float`；`predict(items: list[tuple[str, np.ndarray]]) -> list[dict[str, np.ndarray]]`（每项 = (YOLO 类别, 原图上已裁好的图)；只算 `applies_to` 含该类别的头；softmax 概率）
  - `save_model(path: Path, heads: dict[str, dict], backbone_name: str, backbone_key: str, trained: str) -> None`；每个头 dict 键 `W`(D×K)、`b`(K)、`labels`、`applies_to`、`pad`
  - npz 键：`heads`、`<头>.W`、`<头>.b`、`<头>.labels`、`<头>.applies_to`、`<头>.pad`、`backbone_name`、`backbone_key`、`trained`
  - `load_model(cfg: AttrsConfig, device: str, embedder: Embedder | None = None) -> AttrModel | None`：文件不存在 / 打不开 / `backbone_name != Path(cfg.backbone).name` / `backbone_key` 和 embedder 的 `f"{size}:{norm}"` 对不上 → `log.warning` 一次、返回 `None`。`embedder=None` 时建 `OnnxEmbedder(cfg.backbone, norm="imagenet", device=device, what="attrs.backbone")`；`backbone_key` 用 `f"{embedder.size}:imagenet"`（测试注入的假 embedder 自带 `key` 属性时用它的 `key` 比）

- [ ] **Step 1: 写失败的测试**（`tests/test_attrs.py`）
  - `test_crop_is_square_padded_and_sized`：100×200 的框、`pad=0.15`、`size=64` → 输出 64×64×3；框贴图片左边时左侧补黑（输出第 0 列全 0）
  - `test_predict_uses_only_heads_for_that_class`：假 embedder 返回固定 4 维向量；两个头 `form`（`applies_to=["player","player_unlit"]`）、`icon`（`["social_ring"]`）；`predict([("player", img), ("social_ring", img)])` → 第一项只有 `form`、第二项只有 `icon`；概率和为 1；`W` 设成让 `lit` 最大时 argmax 是 `lit`
  - `test_save_and_load_round_trip`（`tmp_path`）：`save_model` → `load_model(cfg, "cpu", embedder=fake)` 得到同样的 `heads` / `labels` / 概率
  - `test_load_model_rejects_mismatched_backbone`：npz 的 `backbone_key` 和 fake 的 `key` 不同 → 返回 `None`，`caplog` 里有一条 WARNING；文件不存在 → `None`
  - `test_attrs_config_defaults`：`Config().attrs.enabled is False`、`max_crops == 4`、`accept == 0.6`；`load_config` 读 `[attrs]` 表能覆盖

- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_attrs.py -q` → ImportError / AttributeError
- [ ] **Step 3: 实现** `AttrsConfig`（照 `GestureConfig` 的写法、文档串写清"默认关、过了上线门槛再开"）和 `attrs.py` 的上述函数 / 类
- [ ] **Step 4: 跑测试确认通过**：同上命令 → 全过；再跑 `tests/test_config.py -q` 不回归
- [ ] **Step 5: 提交** `git commit -m "feat(attrs): 第二层的模型格式（DINOv2 主干 + npz 线性头）和配置"`

### Task 2: `PersonAttrs.update`（挑裁图、投票、出错关掉）

**Files:**
- Modify: `src/skydango/vision/attrs.py`
- Test: `tests/test_attrs.py`

**Interfaces:**
- Consumes: Task 1 的 `AttrModel`、`crop`；`vision.track.Track`
- Produces:
  - `class PersonAttrs`：`__init__(self, cfg: AttrsConfig, model: AttrModel)`；属性 `enabled: bool`（出错满 `max_errors` 后变 False）
  - `update(self, frame: np.ndarray, tracks: list[Track], now: float, panel: Rect | None) -> None`
  - 轨迹 `data` 键：`"form_hist"`（最近 `votes` 次概率数组的列表）、`"form"`（`(类别, 平均概率)`）、`"form_n"`（累计次数）、`"crop_at"`（上次裁图时间）、`"cls_hist"`（最近 `votes` 帧这条轨迹的 YOLO 类别，每帧更新过的人物轨迹都记，不管裁没裁）

**挑裁图的规则**（测试钉住）：候选 = `cls in ("player", "player_unlit")`、`t.last == now`、没有 `data["rejected"]`、被 `panel` 盖住的面积 < 框面积一半；排序键 `(form_n > 0, now - crop_at < every, -box.h)`，跳过 `form_n > 0 and now - crop_at < every` 的；取前 `max_crops` 个，一次 `model.predict`。

- [ ] **Step 1: 写失败的测试**
  - `test_new_tracks_are_cropped_first_then_by_height`：6 条人物轨迹（2 条 `form_n=0`，4 条已复核、`crop_at` 早于 `now - 0.5`、框高不同）、`max_crops=4` → 被裁的是 2 条新的 + 最高的 2 条（用记录调用的假模型核对）
  - `test_recently_cropped_track_waits_every_seconds`：`crop_at = now - 0.3` 的已复核轨迹这一帧不裁
  - `test_votes_average_last_n`：`votes=5`，连喂 7 次概率 → `form_hist` 长 5、`form` 是后 5 次平均的 argmax、`form_n == 7`
  - `test_half_covered_by_panel_is_skipped`：框 60% 在 `panel` 里 → 不裁、`form` 不变
  - `test_paused_style_skip_and_stale_tracks`：`t.last < now` 的轨迹不裁
  - `test_errors_disable_after_max_errors`：假模型 `predict` 一直抛错、`max_errors=3` → 第 3 帧后 `enabled is False`，`caplog` 有 WARNING；之后 `update` 不再调模型
  - `test_cls_hist_records_every_updated_person_track`：没被裁的轨迹也有 `cls_hist`，长度不超过 `votes`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现 `PersonAttrs.__init__` / `update`**（单帧出错 `log.exception`、这一帧不改任何轨迹；`errors` 成功一次清零）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(attrs): 按轨迹挑裁图、投票，连续出错自己关掉`

### Task 3: 放行和点没点火

**Files:**
- Modify: `src/skydango/vision/attrs.py`
- Test: `tests/test_attrs.py`

**Interfaces:**
- Consumes: Task 2 写的 `data` 键；`data["strong"]`（Task 5 在感知层设：这条轨迹出现过 ≥ `conf` 的检测）
- Produces:
  - `PersonAttrs.admit(self, t: Track) -> bool`
  - `PersonAttrs.is_unlit(self, t: Track) -> bool`：`enabled` 且 `"unlit" in t.data` → `t.data["unlit"]`；否则 `t.cls == "player_unlit"`
  - `unlit_score(t: Track, yolo_w: float) -> float | None`（模块函数）
  - `update` 里每帧对候选人物轨迹顺带：判撤下（写 `data["rejected"] = True`）、复核通过时 `t.strong_last = now`、更新 `data["unlit"]` / `data["unlit_streak"]`

**规则**（spec §3.3、§3.4）：
- `admit`：`rejected` → False；`data.get("strong")` → True；否则 `form` 在 `PERSON_FORMS`、概率 ≥ `accept`、`form_n >= 2` → True；其余 False。`enabled` 为 False 时 = `bool(data.get("strong"))`
- 撤下：`strong` 轨迹的 `form` 是 `not_person`、概率 ≥ `reject`、`form_n >= reject_n`
- 复核通过（`form` 人形且概率 ≥ `accept`）→ `t.strong_last = now`
- `unlit_score`：YOLO 侧 = `cls_hist` 里 `player_unlit` 的比例；外形侧 = 平均概率里 `unlit / (lit + unlit + shared + morph)`（头里没有的类别按 0；分母 0 → 只用 YOLO 侧）；`form_n == 0` 只用 YOLO 侧；`cls_hist` 空 → `None`
- 滞回：`"unlit" not in data` → `u >= 0.5`；之后 `u > 0.6`（当前不是黑影）或 `u < 0.4`（当前是）→ `unlit_streak += 1`，到 `flip_votes` 翻转并清零；否则清零

- [ ] **Step 1: 写失败的测试**
  - `test_strong_track_admitted_until_rejected_and_never_again`：`strong` 轨迹喂 3 次 `not_person` 0.9 → `admit` False；之后再喂 5 次 `lit` 0.99 → 仍 False
  - `test_weak_track_needs_two_confident_votes`：非 `strong`、`form_n=1` `lit` 0.9 → False；`form_n=2` → True；`spirit` 0.7 → True；`not_person` → False；`lit` 0.5（< accept）→ False
  - `test_verified_weak_track_refreshes_strong_last`：复核通过那一帧 `t.strong_last == now`；`not_person` 时不变
  - `test_unlit_score_mixes_yolo_and_form`：`cls_hist` 全 `player`、外形 `unlit` 概率占 0.8、`yolo_w=0.5` → 0.4；`form_n=0` → 0.0
  - `test_unlit_flip_needs_flip_votes_frames`：初始判点过火；连续 2 帧 u=0.7 不翻、第 3 帧翻；中间夹一帧 u=0.5 计数清零
  - `test_is_unlit_falls_back_to_yolo_class_when_disabled`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** `feat(attrs): 放行 / 撤下、复核刷新 strong_last、点没点火两边投票带滞回`

### Task 4: `Tracker.open_low`

**Files:**
- Modify: `src/skydango/vision/track.py`（`Tracker.__init__`、`update`）
- Test: `tests/test_tracker.py`

**Interfaces:**
- Produces: `Tracker(..., open_low: frozenset[str] = frozenset())`，属性 `open_low` 可改（感知层出错关掉第二层时清空）。`update` 里低分框配完旧轨迹后，类别在 `open_low` 里、没被用掉的低分框各开一条新轨迹：`strong_last = float("-inf")`，附在返回值末尾（排在被低分框续上的轨迹之后）

- [ ] **Step 1: 写失败的测试**：
  - `test_open_low_creates_provisional_track`：没有旧轨迹、`low=[player 0.3]`、`open_low={"player","player_unlit"}` → 返回 1 条，`strong_last == -inf`；下一帧同位置高分框接上它（同一 id）且 `strong_last == now`
  - `test_open_low_ignores_other_classes`：`low=[name_tag 0.3]` → 不开
  - `test_default_still_only_extends`：`open_low` 默认空 → 和现在一样不开（现有测试也要全过）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑 `tests/test_tracker.py tests/test_perception_tracking.py tests/test_track_motion.py -q` 全过**
- [ ] **Step 5: 提交** `feat(track): open_low——指定类别的低分框可以开待复核的新轨迹`

### Task 5: 接进感知层（放行、黑影、开关）

**前置**：`feat/light-flame` 已合进 main（Global Constraints）。

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/cli.py`（建 `PerceptionWatcher` 的两处：约 354、660 行）
- Test: `tests/test_perception_attrs.py`（新建；沿用 `tests/test_perception.py` 的 `FakeDetector` / `FakeOcr` / `watcher()` 写法）

**Interfaces:**
- Consumes: Task 1~4
- Produces:
  - `PerceptionWatcher(..., attrs: PersonAttrs | None = None)`；`self.attrs`
  - `detector_conf()`：`attrs` 开着时也用 `min(low_conf, conf)`
  - 轨迹 `data["strong"] = True`：`tracker.update` 返回的前 `len(dets)` 条（高分框接上 / 新开的）
  - 轨迹 `data["admitted"]`（`attrs` 开着时每帧给人物轨迹写；关着时不写，读的地方 `d.get("admitted", True)`）
  - `self._unlit(t) -> bool`：`self.attrs.is_unlit(t)` if `self.attrs and self.attrs.enabled` else `t.cls == UNLIT`

**改法：**
1. `__init__`：`attrs` 不为 None 时 `Tracker(..., open_low=frozenset({"player", UNLIT}))`
2. `process()`：`tracker.update(dets, now, low=low if (self.cfg.track_low or self.attrs) else (), ...)` → 标 `strong` → `self.attrs.update(frame, tracks, now, panel_rect)`（`panel_rect` = 面板开着时的 `log_roi` 矩形，否则 None；感知暂停时不调）→ `players` 只留 `admit` 为真的、写 `data["admitted"]`；`attrs.enabled` 变 False 的那一帧把 `tracker.open_low` 清空
3. 把人物轨迹上的 `t.cls == UNLIT` / `t.cls != UNLIT` / 人物语境里的 `t.cls == "player"` 换成 `self._unlit(t)` / `not self._unlit(t)`：`_assign_tags` 入参、relink 那行、挂标签那行、陌生人计数、`_far_tags` 人选、`_appearance_features` 人选、`_watch_typing` 的 `owner.cls == UNLIT`、`people()`、`overlay()`、点火那段里 `dark = p.cls == UNLIT or …`。**不换**：`_filter` / `people_boxes` / `one_self` 这些对检测结果（不是轨迹）的判断；`sweep()`（逐帧检测，没有轨迹）；`Tracker(cross=…)`
4. `people()` / `overlay()` 跳过 `d.get("admitted", True)` 为假的人物轨迹（`overlay` 见 Task 7 的画法，这一步先跳过）
5. `cli.py`：`cfg.attrs.enabled` 时 `attrs.load_model(cfg.attrs, cfg.attrs.device or cfg.perception.device)`，得到模型就建 `PersonAttrs` 传进去；`None` 就不传（已经警告过）

- [ ] **Step 1: 写失败的测试**（`watcher(..., attrs=PersonAttrs(cfg, FakeModel(...)))`，`FakeModel.predict` 按框 x 坐标返回指定概率）
  - `test_disabled_is_identical`：同一串检测，`attrs=None` 和改动前的结果一致：`people()`、`strangers()`、`overlay()` 相等（用现有一个陌生人 + 一个好友的场景）
  - `test_low_score_person_appears_only_after_review`：只有 0.3 的 `player` 框 → 第 1 帧 `people()` 为空、`strangers()` 0；假模型判 `lit` 0.9，第 2 帧复核满 2 次后（`every` 设 0）出现、满 `stranger_after` 算陌生人
  - `test_high_score_tree_is_withdrawn`：0.8 的 `player` 框、假模型判 `not_person` 0.9 → 前 2 帧在 `people()` 里，第 3 帧起消失、不再算陌生人
  - `test_unvoted_weak_track_is_not_kept_alive`：待复核轨迹一直判 `not_person` → `strong_last` 保持 `-inf`，不进 `players`
  - `test_dark_cloak_lit_player_not_counted_unlit`：YOLO 一直报 `player_unlit`、外形 `lit` 0.95、`yolo_w=0.3` → 3 帧后 `unlit(now) == 0`、`people()` 里 kind 是 `stranger`
  - `test_attrs_errors_turn_off_open_low`：模型一直抛错、`max_errors=2` → 之后 `tracker.open_low == frozenset()`、新出现的低分框不开轨迹
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑 `.venv\Scripts\python.exe -m pytest -q` 全量通过**（感知层相关测试多，必须全量）
- [ ] **Step 5: 提交** `feat(perception): 接上第二层——低分框复核后放行、误框撤下、点没点火两边投票`

### Task 6: 先祖 / 共享空间 / 变身

**Files:**
- Modify: `src/skydango/vision/people.py`、`src/skydango/vision/perception.py`
- Test: `tests/test_perception_attrs.py`、`tests/test_people_motion.py`（`describe_people` 的现有测试要照样过）

**Interfaces:**
- Produces:
  - `Person` 加 `form: str | None = None`、`form_p: float = 0.0`；`kind` 多两种 `"spirit"`、`"shared"`
  - `WHO` 加 `"spirit": "先祖"`、`"shared": "共享空间的人"`；`_describe_person` 对这两种写成 `"一个先祖（右边·远）"`
  - 感知层：外形是 `spirit` / `shared` 的放行轨迹不进 `players`（所以不算陌生人、不挂名字、不进 relink / 续命 / approach），另放 `self._others`；`people()` 给它们 `kind` = 外形；`morph` 照常进 `players`，`Person.form = "morph"`
  - `objects()`：外形 `spirit` 的轨迹也出一个 `Thing(kind="spirit")`，和 YOLO `spirit` 类轨迹框 IoU ≥ 0.5 的只留 YOLO 那个

- [ ] **Step 1: 写失败的测试**
  - `test_spirit_form_is_not_a_stranger`：高分 `player` 框、外形 `spirit` 0.9 → `strangers()` 0、没有 `stranger` 事件数据、`people()` 里 `kind == "spirit"`、`describe_people` 输出含"一个先祖"
  - `test_shared_space_player_not_relinked`：开 `relink`，`shared` 轨迹断掉不进失踪记录
  - `test_morph_keeps_name_tag`：外形 `morph`、头顶好友名字标签 → `kind == "friend"`、`form == "morph"`
  - `test_objects_dedupes_spirit`：同一位置 YOLO `spirit` 类 + 外形 `spirit` 人物轨迹 → `objects()` 只有一个先祖；不重叠时两个
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 全量 pytest 通过**
- [ ] **Step 5: 提交** `feat(perception): 外形头认出的先祖 / 共享空间玩家不算陌生人，变身照常认人`

### Task 7: 识别可视化、难例、设置开关

**Files:**
- Modify: `src/skydango/vision/perception.py`（`overlay()`、调 `hardcases`）、`src/skydango/vision/hardcases.py`、`src/skydango/vision/static/stage.js`、`src/skydango/console/settings.py`、`config.example.toml`
- Test: `tests/test_perception_attrs.py`、`tests/test_hardcases.py`、`tests/test_console_settings.py`（没有就放 `tests/test_cli_console.py` 里已有的设置测试旁）

**Interfaces:**
- Produces:
  - `overlay()` 条目：被撤的人物轨迹 `kind = "rejected"`、`label = f"不是人 {p:.1f}"`；靠复核放行的（非 `strong`）`label` 前加"复核·"；外形 `spirit` / `shared` 的 `kind` 同名；开着第二层时每个人物条目多 `"form": {类别: 概率}` 和 `"u": 黑影分`（悬停显示）
  - `stage.js`：`COLORS.rejected` 灰、虚线；`shared` 浅蓝；`spirit` 沿用物品先祖色；悬停提示里列 `form` 概率和 `u`
  - `HardCaseCollector.report(...)` 新原因：`attrs_reject`（一条高分轨迹被撤的那一帧）、`attrs_disagree`（放行轨迹 YOLO 侧比例和外形侧对点没点火相反、持续 2 s；轨迹 `data["disagree_since"]` 记开始时间，报一次后 `data["disagree_reported"] = True`）
  - 设置页 `Field("attrs.enabled", "人物复核（第二层）", "YOLO 人物框裁图再判一次：挡误框、捞低分框、认先祖 / 共享空间；要配合 YOLO 感知层和训练好的模型", "bool", "features")`
  - `config.example.toml` 加 `[attrs]` 一节（全部键 + 注释，`enabled = false`）

- [ ] **Step 1: 写失败的测试**：`test_overlay_marks_rejected_and_reviewed`、`test_hardcase_on_reject_once`、`test_disagree_reported_after_two_seconds`、设置项存在且类型是 bool
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（js 改动无单测，Step 4 后用 `view --images` 目测，见 Task 12）
- [ ] **Step 4: 全量 pytest 通过**
- [ ] **Step 5: 提交** `feat(viewer): 画出复核撤下 / 放行的框；难例多两种原因；设置页开关`

### Task 8: 裁图（`perception crops`）和写回

**Files:**
- Create: `src/skydango/vision/attrs_data.py`
- Modify: `src/skydango/cli.py`（`perception crops`）
- Test: `tests/test_attrs_data.py`

**Interfaces:**
- Consumes: `attrs.crop`；检测器用 `vision.detect` 现有的 ultralytics / ONNX 封装（同 `perception detect`），测试里注入假检测器
- Produces:
  - `KNOWN = {0: "lit", 4: "unlit", 9: "spirit"}`（`datasets/sky` 类别编号 → 外形）
  - `crop_name(frame_stem: str, box: Rect) -> str` = `f"{frame_stem}__{x}_{y}_{w}_{h}.jpg"`
  - `crops_from_dataset(root: Path, detector, out: Path, conf: float, size: int = 224, pad: float = 0.15) -> dict`：遍历 `images/train` 和 `images/val`（跳过 stem 以 `augment.SUFFIXES` 结尾的）；已知类别框 → `out/form/<外形>/`；检测器在 `conf` 跑出的 `player` / `player_unlit` 框和任何人物标注（0 / 3 / 4 / 9）IoU < 0.4 → `out/_unlabeled/`；返回各类张数
  - `crops_from_images(folder: Path, detector, out: Path, conf: float, ...) -> dict`：录像 / `runs/*/hard` 目录，人物框全进 `_unlabeled/`，同一图里 IoU ≥ 0.5 只留分高的
  - `out/_crops.jsonl` 每张一行：`{"crop", "image", "box", "score", "yolo_cls", "source": "dataset"|"images", "split": "train"|"val"|null, "group": 录像或 run 目录名}`；文件已存在就跳过（重跑幂等）
  - `writeback(dataset: Path, out: Path, now: datetime) -> dict`：读 `out/form/<人形类>/` 里、`_crops.jsonl` 里 `source == "dataset"` 且 `_labels.jsonl` 里有确认记录的裁图，按 `lit/morph/shared → 0`、`unlit → 4`、`spirit → 9` 写回 `labels/<split>/<帧>.txt`；先整个复制 `labels/` 到 `_backup/labels-<时间>`；同帧已有 IoU ≥ 0.5 的人物框不写；返回 `{"frames": n, "boxes": m}`
  - CLI：`perception crops <来源...> [--model 模型] [--conf 0.2] [--out datasets/attrs] [--writeback]`（来源是含 `images/` 和 `labels/` 的目录就当数据集）

- [ ] **Step 1: 写失败的测试**（`tmp_path` 里造一个 2 帧的小数据集 + 假检测器）：已知类别入对应目录、`self` 不要、增强图跳过；对不上的框进 `_unlabeled`；重跑不重复；`crops_from_images` 去重；`writeback` 先备份、写对编号、不重复写、难例来源不写
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 测试通过**；`tests/test_cli_phase3.py -q` 不回归
- [ ] **Step 5: 提交** `feat(attrs): perception crops——从数据集 / 难例裁人物图，确认后写回数据集`

### Task 9: Claude 初分（`perception attrs-label`）

**Files:**
- Modify: `src/skydango/vision/attrs_data.py`、`src/skydango/cli.py`
- Test: `tests/test_attrs_data.py`

**Interfaces:**
- Consumes: `vision.assist`（`FrameInput`、`Protocol`、`Reviewer`、`extract_json`），照 `vision/gesture_label.py` 的结构
- Produces:
  - `FORM_PROMPT_VERSION = 1`、`FORM_SYSTEM`（六类定义：`not_person` 举例树 / 椅子 / UI 图标 / 篝火 / 茶壶 / 雕像壁画；`spirit` 修长偏透明发光；`shared` 矮小蓝色半透明；`morph` 雪人 / 白鹿等变身，是玩家；`unsure` 看不清）
  - `form_sheet(crops: list[np.ndarray]) -> np.ndarray`：≤16 张拼 4×4（每格 160×160、左上角白底黑字编号 0~15）
  - `build_form_message(batch: list[FrameInput], cfg: AssistConfig) -> list[dict]`、`parse_form_review(text, batch) -> dict[str, Guess]`（`Guess` 复用 `gesture_label.Guess`；label 只收 `FORMS + ("unsure",)`，别的当没核对）
  - 结果写 `<out>/_unlabeled/claude.json`：`{裁图名: {"label", "confidence", "reason", "model", "version"}}`；已有且 version 相同的跳过
  - CLI：`perception attrs-label [目录，默认 datasets/attrs]`（令牌、并发、额度用完续跑同 `gesture-label`）
- [ ] **Step 1: 写失败的测试**：拼图尺寸和编号格、消息里带系统提示和图片块、解析（合法 / 未知类别 / 缺项 / 外面包了文字）、`claude.json` 跳过同版本
- [ ] **Step 2~4**：确认失败 → 实现 → 通过
- [ ] **Step 5: 提交** `feat(attrs): perception attrs-label——Claude 按 4×4 拼图初分外形`

### Task 10: 标注页「外形」

**Files:**
- Modify: `src/skydango/console/labeling.py`、`src/skydango/console/server.py`、`src/skydango/console/static/labeling.js`、`console.css`（如需）
- Test: `tests/test_console_labeling.py`

**Interfaces:**
- Produces:
  - `labeling.py` 抽出 `_Moves`（挪条目、`_labels.jsonl` 记账、撤销，按"条目 = 目录"或"条目 = 文件"参数化）；`GestureLabels` 改用它，行为和接口不变（现有测试全过）
  - `class FormLabels`：`__init__(self, root: Path)`（root = `datasets/attrs`，目录 `_unlabeled/`、`form/<类别>/`、`_discard/`）；`state() -> dict`（`counts`、`items: [{"crop", "where", "guess", "image", "box"}]`，`image` / `box` 来自 `_crops.jsonl`）；`crop(name) -> bytes | None`；`context(name) -> bytes | None`（原图缩到宽 480、画出框）；`label(name, to) -> (int, dict)`；`undo() -> (int, dict)`
  - 路由：`GET /api/form/state`、`GET /api/form/crop?name=`、`GET /api/form/context?name=`、`POST /api/form/label {name, to}`、`POST /api/form/undo`（POST 照旧走 `post_guard`）
  - `labeling.js`：页顶「动作 / 外形」两个标签；外形页大图 + 原图缩略 + Claude 猜测和理由；按键：回车 = 同意 Claude（`unsure` 时不动）、1~6 = `FORMS` 顺序、0 = 丢弃、Z = 撤销
- [ ] **Step 1: 写失败的测试**：`FormLabels` 确认 / 改类别 / 丢弃 / 撤销挪对文件并记账；名字不在任何目录里 → 404；`../x.jpg`、`con.jpg` → 404 且不碰文件系统；`GestureLabels` 现有测试照过
- [ ] **Step 2~4**：确认失败 → 实现 → `tests/test_console_labeling.py tests/test_cli_console.py -q` 通过
- [ ] **Step 5: 提交** `feat(console): 标注页加「外形」，和动作页共用挪文件 / 记账 / 撤销`

### Task 11: 训练和回放评估

**Files:**
- Create: `src/skydango/vision/attrs_train.py`
- Modify: `src/skydango/cli.py`（`perception attrs-train`、`attrs-eval`、`bench --attrs`）
- Test: `tests/test_attrs_train.py`

**Interfaces:**
- Consumes: Task 1 `save_model` / `AttrModel`；Task 8 `_crops.jsonl`；`gesture_train.macro_f1`
- Produces:
  - `split_crops(root: Path, seed: int = 0, val_ratio: float = 0.2) -> dict`：`source == "dataset"` 按它的 `split`；其他按 `group` 整组随机分到约 `val_ratio`；写 `root/_split.json`
  - `merge_labels(counts: dict[str, int], min_per_class: int) -> tuple[dict[str, str | None], list[str]]`：`shared` / `morph` 不够 → 映射到 `lit`；`spirit` 不够 → `None`（丢掉样本）；`not_person` / `lit` / `unlit` 不够 → 抛 `ValueError`；第二项是写进报告的说明
  - `features(paths: list[Path], embedder, cache: Path, flip: bool) -> np.ndarray`：按 `embedder.key` 分目录缓存 `.npy`
  - `train_head(X, y, classes: int, l2: float, class_weight: np.ndarray, epochs: int = 300) -> tuple[np.ndarray, np.ndarray]`：numpy 全批梯度下降的 softmax 回归；`l2` 从 `(1e-4, 1e-3, 1e-2)` 里按验证集 macro F1 挑
  - `replay(frames: list[Path], detector, model: AttrModel, conf_low: float, conf: float, accept: float, reject: float, reject_n: int = 3) -> dict`：在带标注的帧上比较"YOLO ≥ conf"和"YOLO ≥ conf_low + 复核（单帧版：每框当成复核了 `reject_n` 次同样结果）"的人物精确率 / 召回率（IoU ≥ 0.4 算对）和点没点火认反数
  - `sweep_thresholds(...) -> tuple[float, float]`：`accept` ∈ {0.5, 0.6, 0.7, 0.8}、`reject` ∈ {0.6, 0.7, 0.8, 0.9}，精确率不低于纯 YOLO 前提下召回最高的一组
  - CLI `perception attrs-train [datasets/attrs] [--out] [--force] [--device]`：切分 → 特征 → 训 → `save_model` → 报告（每类 P / R、混淆矩阵、合并说明、`replay` 在 0.2 和 `low_conf` 两档、建议阈值）；`--out` 等于 `[attrs] model` 且没 `--force` → 拒绝
  - CLI `perception attrs-eval <数据集> --model 模型`：只跑 `replay` + 报告
  - `perception bench --attrs`：每帧额外跑 `max_crops` 张裁图的 `predict`，报告前后 fps
- [ ] **Step 1: 写失败的测试**：`split_crops` 数据集来源按原切分、同 group 不跨集、同 seed 结果相同；`merge_labels` 三种情况；`train_head` 在两团可分的合成点上验证集准确率 = 1；特征缓存第二次不调 embedder；`replay` 在手写的 2 帧 + 假检测器 + 假模型上数对 TP / FP / FN；`--out` 覆盖保护
- [ ] **Step 2~4**：确认失败 → 实现 → 通过
- [ ] **Step 5: 提交** `feat(attrs): attrs-train / attrs-eval——numpy 线性头、按来源切分、整帧回放报告`

### Task 12: 文档、实测、收尾

**Files:**
- Modify: `CLAUDE.md`（代码结构表加 `vision/attrs.py` `attrs_data.py` `attrs_train.py`；「YOLO 感知层」加一条第二层；常用命令加 `perception crops / attrs-label / attrs-train / attrs-eval`）、`docs/progress/2026-09-28-yolo-training.md`（加"第二层"一节：上线门槛四条 = spec §7）
- 不改代码

- [ ] **Step 1: 测速**（GPU 机器）：`.venv\Scripts\python.exe -c` 一段脚本测 `models/dinov2-small.onnx` 在 cuda 上单张 / 4 张逐张的 ms，结果写进进度文档；如果逐张 4 张 > 40 ms，在进度文档里记下"要导出带 batch 维的主干"，**不在这个计划里做**
- [ ] **Step 2: 真数据跑一遍工具链**：`perception crops datasets/sky --model models/sky-yolo-v7.pt` 和 `perception crops runs/*/hard --model …`，打印的张数记进进度文档（不调 Claude，`attrs-label` 留给用户决定什么时候花额度）
- [ ] **Step 3: 文档**写好，`.venv\Scripts\python.exe -m pytest -q` 全量通过
- [ ] **Step 4: 提交** `docs: 感知层第二层写进 CLAUDE.md 和训练进度`
- [ ] **Step 5: 用 finishing-a-development-branch 收尾**，按 CLAUDE.md 合进 main 并推送；告诉用户剩下要人做的：`attrs-label` → 标注页确认 → `--writeback` → `attrs-train` → 看报告 → 晚上真机三步（spec §9）
