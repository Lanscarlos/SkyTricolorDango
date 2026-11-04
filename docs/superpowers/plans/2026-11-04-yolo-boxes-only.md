# YOLO 只框人（视觉重构 A）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让感知层能用"人只有一类"的 YOLO 模型：点没点火 / 先祖只由外形头判（外形头用不了时按亮度估），训练时临时把 `datasets/sky` 的三类人合成一类。

**Architecture:** 检测器加载时认出单类模型（0 号叫 `person`），把它的框改名成 `player` 往下传，下游按类别名的代码不动；点没点火统一走 `PerceptionWatcher._unlit`，单类模式下 YOLO 那一侧权重为 0、外形头关着时用 `candle.black()` 投票。训练侧新加合并视图（硬链接图片 + 改类别号），`retrain` / 回放对比 / 收件箱跟着认单类模式。旧模型（有 `player_unlit`）逐字照旧。

**Tech Stack:** Python 3.13、numpy、OpenCV、onnxruntime / ultralytics、pytest（合成画面 + 假检测器）。

**Spec:** `docs/superpowers/specs/2026-11-04-yolo-boxes-only-design.md`

## Global Constraints

- 单类模式只由模型类别名决定：类别 0 叫 `person` → 单类；不加配置开关。
- 类别号不变：0 `person`、4 `unused_4`、9 `unused_9`，其余同 `datasets/sky/data.yaml`（1 name_tag、2 social_ring、3 self、5 typing、6 bench、7 bonfire、8 instrument）。
- 单类模式下检测器把 `person` 改名成 `player` 输出，`unused_*` 的框丢掉；`self` 类保留、不动。
- 新配置：`[attrs] dark_unlit = 0.5`（`black()` 中位数 ≥ 它算黑影）、`[retrain] merge = true`。
- 亮度投票复用 `[attrs] every` / `votes` / `flip_votes`，滞回按 `dark_unlit ± 0.1`（见 Task 2），量法用 `candle.black(frame, box, light.lit_v, exclude=团子框)`，`None` 不投票。
- `datasets/sky` 一个字节都不改。
- 旧模型行为逐字照旧：现有测试全部照样过。
- 测试跑法（worktree 里）：`D:\Lanscarlos\Develop\SkyTricolorDango\.venv\Scripts\python.exe -m pytest -q`（`tests/conftest.py` 把 worktree 的 `src` 放最前）。全量测试放后台跑。
- 注释、日志、status 文案用中文，风格照周围代码。

## Review Focus

1. 外形头运行中途自己关掉（连续出错）：点没点火要无缝切到亮度投票，已有轨迹的 `unlit` 不能被清掉或卡死 —— Task 3 测试 `test_single_attrs_dies_falls_back_to_dark`。
2. 单类模式下点亮陌生人找"下面的人是不是黑影"：`_people_boxes` 的第二位要来自 `_unlit`，低分框（没有轨迹）一律 False —— Task 3 测试 `test_single_people_boxes_flag_from_unlit`。
3. 合并视图建在已有目录上（重训重跑）：要先清空，不能留上次的标签 —— Task 5 测试 `test_build_merged_overwrites`。
4. 收件箱里原标注带来的 `player_unlit` 框（`src=label`）在单类模式下仍按老规则配对，只有 YOLO 出的框走单类规则 —— Task 4 测试 `test_route_single_only_for_yolo`。
5. 回放对比时新旧两个检测器一个单类一个不是：各自用自己的 `yolo_w`（单类 0）—— Task 6 测试 `test_replay_row_single_ignores_yolo_w`。

---

### Task 1: 检测器认出单类模型

**Files:**
- Modify: `src/skydango/vision/detect.py`（`OnnxYoloDetector.__init__`、`UltralyticsDetector.__init__`、`decode`、`UltralyticsDetector.detect`）
- Test: `tests/test_detect_single.py`（新建）

**Interfaces:**
- Produces: `single_class_names(names: list[str]) -> tuple[list[str], bool]`（单类时返回把 `person` 换成 `player`、`unused_*` 换成 `""` 的新列表和 True；否则原样 + False）；两种检测器都有属性 `single: bool`；名字为 `""` 的类别框不出（`decode` / `UltralyticsDetector.detect` 里跳过）。

- [ ] **Step 1: 写失败的测试**

```python
from skydango.vision.detect import single_class_names, decode

NEW = ["person", "name_tag", "social_ring", "self", "unused_4", "typing", "bench", "bonfire", "instrument", "unused_9"]
OLD = ["player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit"]

def test_single_class_names_new():
    names, single = single_class_names(NEW)
    assert single and names[0] == "player" and names[4] == "" and names[9] == "" and names[3] == "self"

def test_single_class_names_old():
    assert single_class_names(OLD) == (OLD, False)

def test_decode_drops_unused():
    # 用一个 4+nc 行的假输出：一个 cls=4 的框、一个 cls=0 的框；单类名字下只剩一个 player
    ...
```

`test_decode_drops_unused` 按 `decode` 现有的非端到端输出格式（`pred.shape[0] == 4 + len(names)`）造 numpy 数组，断言 `[d.cls for d in dets] == ["player"]`。

- [ ] **Step 2: 跑测试确认失败** — `… -m pytest tests/test_detect_single.py -q`，期望 ImportError。
- [ ] **Step 3: 实现** `single_class_names`；两个检测器 `__init__` 里 `self.names, self.single = single_class_names(self.names)`；出框时类别名为 `""` 就跳过。
- [ ] **Step 4: 跑测试确认通过**，再跑 `tests/test_perception*.py` 确认旧模型没变。
- [ ] **Step 5: 提交** `feat(perception): 检测器认出单类模型（person 当 player 输出）`

### Task 2: 外形头单类模式 + 亮度投票的纯函数

**Files:**
- Modify: `src/skydango/vision/attrs.py`（`unlit_score`、`PersonAttrs`）、`src/skydango/config.py`（`AttrsConfig`）、`config.example.toml`（`[attrs]`）
- Test: `tests/test_attrs.py`

**Interfaces:**
- Produces:
  - `AttrsConfig.dark_unlit: float = 0.5`（注释：单类模式、外形头用不了时 `black()` 中位数 ≥ 这个算黑影，同 `[catalog] dark_max` 的量）
  - `PersonAttrs.single: bool = False`（属性，由感知层设置）；`single` 时 `unlit_score` 的 YOLO 权重为 0（`unlit_score(t, 0.0)`）；`single` 时 `data["unlit"]` 每翻转一次 `t.flips += 1`
  - `dark_vote(t: Track, value: float | None, cfg: AttrsConfig) -> None`：`value` 进 `t.data["dark_hist"]`（留最近 `cfg.votes` 个，`None` 不进）；中位数 `m`：第一次直接 `unlit = m >= dark_unlit`；之后当前黑影时 `m < dark_unlit - 0.1`、当前点过火时 `m > dark_unlit + 0.1` 才累计 `unlit_streak`，满 `flip_votes` 翻转并 `t.flips += 1`；否则清零 streak。和外形头共用 `data["unlit"]` / `unlit_streak` 两个键。

- [ ] **Step 1: 写失败的测试**

```python
def test_unlit_score_single_ignores_yolo():
    # cls_hist 全是 player，form_mean unlit 0.9 / lit 0.1 → single 时分数 0.9（yolo_w 不生效）

def test_dark_vote_first_and_hysteresis():
    cfg = AttrsConfig(votes=5, flip_votes=3, dark_unlit=0.5)
    t = Track(...)  # 照 test_attrs.py 已有的建法
    dark_vote(t, 0.8, cfg); assert t.data["unlit"] is True
    for v in (0.45, 0.45, 0.45, 0.45): dark_vote(t, v, cfg)
    assert t.data["unlit"] is True          # 0.45 在滞回区里，不翻
    for _ in range(5): dark_vote(t, 0.1, cfg)
    assert t.data["unlit"] is False and t.flips == 1

def test_dark_vote_none_ignored():
    # value None 不进 dark_hist、不改 unlit

def test_person_attrs_single_counts_flips():
    # single=True，外形头先给黑影几票再给点亮几票，翻转后 t.flips == 1；single=False 时 flips 不动
```

- [ ] **Step 2: 跑测试确认失败。**
- [ ] **Step 3: 实现**（`config.example.toml` 的 `[attrs]` 加 `dark_unlit = 0.5` 一行带注释）。
- [ ] **Step 4: 跑 `tests/test_attrs.py tests/test_perception_attrs.py` 通过。**
- [ ] **Step 5: 提交** `feat(attrs): 单类模式下只看外形头，加亮度投票`

### Task 3: 感知层接上单类模式

**Files:**
- Modify: `src/skydango/vision/perception.py`（`__init__` 加参数 `attrs_cfg: AttrsConfig | None = None`；`_unlit`；`_other_form` 不变；`_people_boxes` 的计算；新方法 `mode_note`）、`src/skydango/cli.py`（两处 `PerceptionWatcher(` 传 `attrs_cfg=cfg.attrs`，建好后单类模式 `log.info`）、`src/skydango/brain/body.py`（`status()` 在"读聊天："后面加一行）
- Test: `tests/test_perception_single.py`（新建，照 `tests/test_perception_attrs.py` 的假检测器 / 合成画面写法）

**Interfaces:**
- Consumes: Task 1 的 `detector.single`（用 `getattr(detector, "single", False)`）；Task 2 的 `PersonAttrs.single`、`dark_vote`、`AttrsConfig.dark_unlit`。
- Produces:
  - `PerceptionWatcher.single: bool`；`__init__` 里若 `attrs` 不为 None 就设 `attrs.single = self.single`
  - 单类模式每帧（不暂停时）对外形头关着 / 没有时的人物轨迹（`players` + 团子轨迹以外）按 `attrs_cfg.every` 节流调 `dark_vote(t, black(frame, t.box, self.light_cfg.lit_v, exclude=团子框), attrs_cfg)`；节流时间记 `t.data["dark_at"]`
  - `_unlit(t)`：外形头开着 → `attrs.is_unlit(t)`（外形头还没投出 `unlit` 时 `is_unlit` 在单类模式下返回 `t.data.get("unlit", False)`，不再看 `cls`）；外形头关着且单类 → `t.data.get("unlit", False)`；否则照旧 `t.cls == UNLIT`
  - `_people_boxes` 第二位：单类模式下 = 和这一帧某条人物轨迹 IoU ≥ 0.5 且 `_unlit(那条轨迹)`；配不上轨迹的（低分框）= False。旧模式照旧
  - `mode_note() -> str | None`：旧模式 None；单类且外形头开着 → `"单类模式"`；单类且外形头没开 / 自己关了 → `"单类模式；点没点火按亮度估（外形头没开）"`
  - `Body.status()`：`note = getattr(self.env, "mode_note", None)`，有内容就 `parts.append("感知：" + note)`

- [ ] **Step 1: 写失败的测试**

```python
def test_single_mode_detected():            # 假检测器 single=True → watcher.single；attrs.single 也被设上
def test_single_dark_vote_marks_unlit():    # 外形头 None，合成画面里一个纯黑人物框连着几帧 → 算陌生人、unlit(now) == 1
def test_single_lit_not_unlit():            # 同上但人物框是彩色 → unlit(now) == 0
def test_single_attrs_dies_falls_back_to_dark():  # 外形头先投黑影，再让它 enabled=False → 之后按亮度投票，原来的 unlit 不丢
def test_single_people_boxes_flag_from_unlit():   # 黑影轨迹的框第二位 True、低分框 False
def test_mode_note():                        # 三种情况的文字；旧模式 None
def test_old_model_unchanged():              # single=False 的假检测器出 player_unlit → 和现在一样按类别判
```

- [ ] **Step 2: 跑测试确认失败。**
- [ ] **Step 3: 实现**（`body.py` 那一行和 cli 的日志 `"感知层：单类模式（点没点火看外形头%s）"` 顺带加；`tests/test_brain_*` 里若有 status 全文比对的，按新行更新——只在单类时出现，旧模式文字不变）。
- [ ] **Step 4: 跑 `tests/test_perception*.py tests/test_light*.py tests/test_brain_light_body.py tests/test_hardcases.py` 通过。**
- [ ] **Step 5: 提交** `feat(perception): 单类模式下点没点火走外形头 / 亮度，status 写明`

### Task 4: 收件箱整理认单类模式

**Files:**
- Modify: `src/skydango/vision/inbox.py`（`route` 加关键字参数 `single: bool = False`；`process` 里对 YOLO 出的框传 `single=getattr(检测器, "single", False)`，原标注来的框 `single=False`）
- Test: `tests/test_inbox.py`

**Interfaces:**
- Produces: `route(yolo_cls, score, probs, conf, agree, *, single=False)`：`single` 时 `form in ("lit", "unlit", "spirit") and p >= agree` → `auto = "agree"`（不看 `PAIR`）；`not_person` 规则不变。写回照旧用 `FORM_CLS[form]`。

- [ ] **Step 1: 写失败的测试** `test_route_single_agree_unlit`（`route("player", 0.9, {"unlit": 0.95, "lit": 0.05}, 0.5, 0.9, single=True).auto == "agree"`，`single=False` 时是 None）、`test_route_single_only_for_yolo`（`process` 走一帧：原标注 `player_unlit` 框 + 假单类检测器的框，原标注框按老规则、YOLO 框按单类规则；照现有 `process` 测试的假 ctx 写法）。
- [ ] **Step 2: 确认失败。** **Step 3: 实现。** **Step 4: `tests/test_inbox.py` 通过。**
- [ ] **Step 5: 提交** `feat(inbox): 单类模型下 YOLO 说是人、外形头有把握就自动确认`

### Task 5: 合并视图 + retrain 用它训练

**Files:**
- Create: `src/skydango/vision/merged.py`
- Modify: `src/skydango/config.py`（`RetrainConfig.merge: bool = True`）、`config.example.toml`（`[retrain]`）、`src/skydango/vision/retrain.py`（`_run`、`compare`）
- Test: `tests/test_merged.py`（新建）、`tests/test_retrain.py`

**Interfaces:**
- Produces:
  - `MERGE_IDS = {4: 0, 9: 0}`；`MERGED_NAMES`（Global Constraints 那份 10 个名字）
  - `build(dataset: Path, out: Path) -> Path`：先删掉 `out`（存在的话）再建；`images/{train,val}` 每张 `os.link`，失败 `shutil.copy2`；`labels/{train,val}/*.txt` 逐行把第一列按 `MERGE_IDS` 改，其它行原样（空行、非数字开头的行原样）；`*.cache` 不复制；写 `out/data.yaml`（`path:` 指 `out` 绝对路径，`train: images/train`、`val: images/val`、names 用 `MERGED_NAMES`）；返回 `out / "data.yaml"`
  - `compare(..., yolo_dataset: Path | None = None)`：**新模型**的 ultralytics val 和它的 `_inbox_yaml` 用 `yolo_dataset or dataset`；**旧模型**（不是单类）的 val 照旧用 `dataset`（ultralytics val 没法把旧模型的 4 / 9 号映射成 0，放在合并视图上会算错）；外形头回放、取帧仍用 `dataset`。spec §3.1 门槛 1（找人不变差）看回放里 `baseline` 的召回 / 精确率（按框配对、不分类别），不看 mAP
  - `_run`：`cfg.retrain.merge` 时在 `stash_caches` 之后 `merged.build(dataset, out / "merged")`，`train_yolo` 和 `compare(yolo_dataset=…)` 用它；报告加一行"YOLO 用合并视图训练（单类：player / player_unlit / spirit → person）"

- [ ] **Step 1: 写失败的测试**

```python
def test_build_merged(tmp_path):
    # 造 datasets/sky：images/train/a.jpg、labels/train/a.txt = "4 0.5 0.5 0.1 0.2\n0 …\n9 …\n3 …\n"、
    # images/val/b_blur.jpg + labels、labels/train.cache
    yaml = build(src, tmp_path / "m")
    assert [l.split()[0] for l in (tmp_path/"m/labels/train/a.txt").read_text().splitlines()] == ["0", "0", "0", "3"]
    assert (tmp_path/"m/images/val/b_blur.jpg").is_file() and not (tmp_path/"m/labels/train.cache").exists()
    assert "0: person" in yaml.read_text(encoding="utf-8") and "4: unused_4" in yaml.read_text(encoding="utf-8")
    assert src_label_bytes == (src/"labels/train/a.txt").read_bytes()   # 原数据集没动

def test_build_merged_overwrites(tmp_path):   # out 里有上次留下的 stale.txt → 建完不在了

def test_run_retrain_uses_merged(tmp_path):   # 照 test_retrain.py 现有的假 train/val：merge=True 时假 train 收到的 data 是 out/merged/data.yaml；merge=False 时是 dataset/data.yaml
```

- [ ] **Step 2: 确认失败。** **Step 3: 实现。** **Step 4: `tests/test_merged.py tests/test_retrain.py` 通过。**
- [ ] **Step 5: 提交** `feat(retrain): 训练时把三类人合成一类（合并视图，不动 datasets/sky）`

### Task 6: 回放评估：只拿明确的答案计点没点火 + 单类 yolo_w + 亮度准确率

**Files:**
- Modify: `src/skydango/vision/attrs_train.py`（`_gt_boxes`、`_score`、`collect`、`replay_row`、`simulate` 的调用方、报告）、`src/skydango/vision/retrain.py`（`compare` 里给新旧检测器各自传 `single`）
- Test: `tests/test_attrs_train.py`、`tests/test_retrain.py`

**Interfaces:**
- Produces:
  - `_gt_boxes(...) -> list[tuple[Rect, int, bool]]`：第三位 `sure` = 类别 4 / 9，或这一框来自 `fixes`（改过类别或补进来的）；只标 `0` 且没被 fixes 碰过的 `sure = False`。所有读 `rec["gt"]` 的地方跟着改成三元组
  - `_score`：点没点火只在 `sure` 且 GT 是 0 / 4 的配对上计（`pairs` / `mismatch`）；tp / fp / fn 照旧算所有框
  - `collect(...)` 每个 det 多 `"dark": black(img, d.box)`（`candle.black` 默认参数，不排除团子）
  - `replay_row(records, conf_low, conf, a, *, single=False)`：`single` 时 `yolo_w` 按 0 算并记在结果里
  - `dark_sweep(records) -> dict`：在 `sure` 的 0 / 4 配对上（按 `MATCH_IOU` 贪心配，用 `dark` 不是 None 的框）对门槛 `0.30, 0.35, …, 0.80` 算准确率，返回 `{"best": 门槛, "acc": 准确率, "n": 对数, "rows": [(门槛, 准确率), …]}`；`attrs-eval` 和重训报告各加一节"只用亮度判点没点火（外形头用不了时）"，列 best / acc / n

- [ ] **Step 1: 写失败的测试** `test_gt_sure_flags`（标注 0 / 4 / 9 三框 + 一条 fixes 把 0 改 4 → sure 依次正确）、`test_score_ignores_unsure_player`（只标 0 的框判成黑影不计 mismatch，sure 的计）、`test_replay_row_single_ignores_yolo_w`（单类 records 全是 player、外形头 unlit 0.6 → single=True 判黑影，single=False 判点亮）、`test_dark_sweep_picks_threshold`（造 dark 值：黑影 0.7、点亮 0.2 → best 在 0.25~0.7 之间、acc == 1.0）。
- [ ] **Step 2: 确认失败。** **Step 3: 实现。** **Step 4: `tests/test_attrs_train.py tests/test_retrain.py tests/test_attrs_data.py` 通过。**
- [ ] **Step 5: 提交** `feat(attrs): 回放只拿明确的答案计点没点火，报告加亮度判法的准确率`

### Task 7: 标注提示词、文档

**Files:**
- Modify: `src/skydango/vision/assist.py`（提示词第 47~60 行附近：加一句"点没点火拿不准就标 player，外形页会再确认"）、`src/skydango/cli.py`（约 2552 行 `perception label` 的提示：同样一句）、`CLAUDE.md`（「YOLO 感知层」加一条"单类模式"小节：spec / 计划路径、怎么认、点没点火来源、合并视图、`dark_unlit` / `retrain.merge`、**还没训出单类模型、真机 spec §3.2 五步没走**；「常用命令」不变）、`docs/progress/2026-09-28-yolo-training.md`（末尾加 11-04 一节：做了什么、下一步 `perception retrain` 训单类模型看三条门槛）
- Test: 无新增（提示词有版本号的话照规矩加一；`tests/test_assist*.py` 若比对提示词全文则更新）

- [ ] **Step 1: 改文案和文档。**
- [ ] **Step 2: 后台跑全量测试** `… -m pytest -q`，期望全部通过（和开始前比，失败数不增加；开始前先记一次基线）。
- [ ] **Step 3: 提交** `docs: 单类模式（YOLO 只框人）`
