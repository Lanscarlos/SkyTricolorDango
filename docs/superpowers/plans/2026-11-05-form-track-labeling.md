# 外形页按轨迹判 + 训练集核对名单 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 外形页能把同一个人连着几帧的裁图串成一组、按一下键整组判；再加一个命令列出训练集里该人工核对的人物框（「重点核对」）。

**Architecture:** 分组是纯函数（`vision/cropgroups.py`，复用 `vision/track.py` 的 `Tracker`），外形页后端 `FormLabels.state` 现算（按 `_crops.jsonl` 缓存）；整组标注逐张记账、多一个 `batch` 字段，撤销按 `batch` 整组撤。前端纯逻辑放新文件 `static/formgroup.js`（node 可测），`labeling.js` 只接界面。核对名单是离线命令 `perception crops-audit`（`vision/crops_audit.py`）。

**Tech Stack:** Python 3.13、OpenCV、pytest；原生 JS（node 跑纯函数测试）。

**Spec:** `docs/superpowers/specs/2026-11-05-form-track-labeling-design.md`

## Global Constraints

- 分组：只在同一来源里串；相邻两帧时间差 > **3 秒**断开；一组最多 **30** 张；取不到时间的单独一组；分组不写进任何数据文件。
- `_labels.jsonl` 照旧每张一行，整组那一下的几行多一个相同的 `batch`（字符串）；写回、训练、回放的代码不改。
- 「按轨迹」开关默认开，存在 `localStorage` 键 `skydango.form.byTrack`（读写包 try/catch，读不到 = 开）；关掉 = 外形页逐字照旧。
- 「重点核对」名单文件 `datasets/attrs/_focus.json`（JSON 字符串数组）；不存在 = 这个筛选不出现。
- 动作页（`GestureLabels`）一点不动。
- 页面不引外部资源、不用原生 `confirm` / `prompt` / `alert`；颜色只用 `console.css` 里 `:root` 的变量。
- 测试命令（worktree 根目录）：`python -m pytest -q <文件>`（系统 python 装了 pytest；`tests/conftest.py` 把 worktree 的 `src` 放最前）。全量测试放后台跑，跑的时候别改文件。
- 注释、提示文案用中文，照周围代码的风格。

## Review Focus

1. 整组里有一张已经被别处挪走 / 确认过：这一张跳过、其余照标，返回里列出跳过的 —— Task 2 `test_form_label_batch_skips_missing_and_confirmed`。
2. 整组撤销时其中一张被手动挪过：停在那一张、已撤的保持撤掉、报错说明撤了几张 —— Task 2 `test_form_undo_batch_stops_at_moved`。
3. `_crops.jsonl` 里有坏行 / 缺 `image` / `box` 的行：分组不崩，这些裁图各自一组 —— Task 1 `test_groups_bad_rows_are_singletons`。
4. `_focus.json` 坏了（不是 JSON、不是数组）：`state` 照常返回、`focus` 为空列表 —— Task 2 `test_form_state_focus_bad_file`。
5. 同一帧两个人并排、各自连着几帧：不串成一组 —— Task 1 `test_groups_side_by_side_not_merged`。

---

### Task 1: 裁图分组（纯函数）

**Files:**
- Create: `src/skydango/vision/cropgroups.py`
- Test: `tests/test_cropgroups.py`

**Interfaces:**
- Consumes: `vision.track.Tracker`（`Tracker(buffer, min_iou, center_gate=…)`、`update(dets, now) -> list[Track]`，返回顺序同 dets）、`vision.detect.Detection(cls, box, score)`、`vision.bubbles.Rect`、`vision.attrs_data.frame_group(stem)`。
- Produces: `GAP = 3.0`、`CAP = 30`、`groups(rows: Iterable[dict], gap: float = GAP, cap: int = CAP) -> list[list[str]]`（每组是按时间排的裁图名；输入里每个有 `crop` 名的行恰好出现在一组里）、`frame_time(stem: str) -> float | None`（帧名末尾 `_<秒>s`，如 `0351_175.50s` → 175.5）。

- [ ] **Step 1: 写失败的测试**

```python
def row(crop, image, box, split="train", source="dataset", dataset="D:/x/datasets/sky", group=None):
    return {"crop": crop, "image": image, "box": box, "split": split, "source": source, "dataset": dataset, "group": group}

def test_frame_time():
    assert frame_time("20260929-223311-s3_0351_175.50s") == 175.5
    assert frame_time("abc") is None

def test_groups_chain_consecutive_frames():
    rows = [row(f"c{i}", f"D:/x/rec1_{i:04d}_{i*1.5:06.2f}s.jpg", [100 + i * 5, 100, 50, 150]) for i in range(5)]
    assert groups(rows) == [[f"c{i}" for i in range(5)]]

def test_groups_break_on_gap():       # 第 3 帧和第 4 帧隔 4 秒 → 两组
def test_groups_different_recordings_or_split_not_merged():  # 前缀不同 / split 不同 → 各自一组
def test_groups_no_time_singletons(): # 帧名没有 _<秒>s → 每张一组
def test_groups_cap_splits():         # 35 帧同一个人 → [30, 5]
def test_groups_side_by_side_not_merged():  # 每帧两个框（x=100 和 x=600，各自小幅移动）× 5 帧 → 两组各 5 张
def test_groups_bad_rows_are_singletons():  # 缺 image / box、box 不是 4 个数 → 各自一组，不抛异常
def test_groups_inbox_by_run():       # source="inbox"，group 相同的一起串，不同 group 不串
```

- [ ] **Step 2:** `python -m pytest tests/test_cropgroups.py -q`，期望 ImportError。
- [ ] **Step 3: 实现**
  - 来源键：`source == "dataset"` → `("dataset", dataset, split, frame_group(stem))`；其它来源 → `(source, group, split)`。`stem` 是 `image` 的文件名去扩展名。
  - 每个来源内按 `(time, crop)` 排序，按时间分帧（同一 `image` 的裁图同一帧）；每帧 `Tracker.update([Detection("player", box, 1.0) …], time)`，用返回的轨迹 id 归组。
  - 两帧时间差 > `gap` 就换一个新的 `Tracker`（之前的轨迹全部结束）。`Tracker(buffer=gap, min_iou=0.3, center_gate=1.0)`。
  - 每条轨迹的裁图按时间、每 `cap` 张切一段。输出按（来源、第一张的时间）排序。
- [ ] **Step 4:** 测试通过。
- [ ] **Step 5: 提交** `feat(labeling): 外形裁图按轨迹分组`

### Task 2: 外形页后端：分组、重点核对、整组标注和撤销

**Files:**
- Modify: `src/skydango/console/labeling.py`（`_Moves._relabel` / `_revert` 加可选的额外记账字段；`FormLabels.state` / `label` / `undo`）、`src/skydango/console/server.py:1098`（`/api/form/label` 把 `body.get("names")` 传进去）
- Test: `tests/test_console_labeling.py`

**Interfaces:**
- Consumes: Task 1 的 `cropgroups.groups(rows)`。
- Produces:
  - `FormLabels.state()` 多两个键：`groups: list[list[str]]`（只含现在还在某个位置里的裁图；按 `_crops.jsonl` 的 `(路径, mtime, size)` 做模块级缓存）和 `focus: list[str]`（`_focus.json` 里现在还存在的裁图名；文件不存在 / 坏了 = `[]`）。
  - `FormLabels.label(name: str, to: str, names: list[str] | None = None) -> tuple[int, dict]`：`names` 非空时整组：生成 `batch = uuid4().hex[:12]`，持锁逐张做单张的逻辑（原地确认或挪动），每行记账带 `"batch": batch`；找不到 / 已确认是这一类 / 挪不动的跳过。返回 `200, {"ok": True, "batch": batch, "items": [每张成功的 _fresh 结果], "skipped": [{"crop": 名, "text": 原因}]}`；一张都没成功返回 `409, {"ok": False, "text": …, "skipped": […]}`。`names` 为空 / None 时逐字照旧。
  - `FormLabels.undo()`：撤销栈顶那条若带 `batch`，就连续撤销直到栈顶的 `batch` 不同（每条撤销记录也写上同一个 `batch`）；返回 `200, {"ok": True, "batch": batch, "items": [撤回后各张的 _fresh]}`；中途某张撤不了：停下、返回 `409, {"ok": False, "text": "撤了 N 张，第 N+1 张已经不在原来的位置……", "items": [已撤的]}`。没有 `batch` 的逐字照旧。

- [ ] **Step 1: 写失败的测试**（照文件里现有 `test_form_*` 的建数据写法）

```python
def test_form_state_groups_and_focus(tmp_path):        # 3 张同一人连续帧 + 1 张别的录像 → groups 里有 [a,b,c] 和 [d]；_focus.json=["b","zzz"] → focus == ["b"]
def test_form_state_focus_bad_file(tmp_path):         # _focus.json 写 "{oops" → state ok、focus == []
def test_form_label_batch_logs_one_line_each(tmp_path):  # names=[a,b,c] 标 lit → _labels.jsonl 新增 3 行、batch 相同、都挪进 form/lit
def test_form_label_batch_skips_missing_and_confirmed(tmp_path):  # names 里有不存在的和已确认是 lit 的 → 跳过这两张、其余成功、skipped 列出
def test_form_undo_batch(tmp_path):                    # 整组标完再 undo → 3 张都回 _unlabeled，hand_labels 里没有它们
def test_form_undo_batch_stops_at_moved(tmp_path):     # 标完后手动把其中一张挪走 → undo 返回 409，已撤的那几张回去了
def test_form_undo_single_unchanged(tmp_path):         # 不带 names 的标注 → undo 照旧只撤一张、返回形状同以前
def test_form_api_label_names(tmp_path, upstream, monkeypatch):  # POST /api/form/label {"names":[…], "to":"lit"} 走到整组（照 test_form_api_routes）
```

- [ ] **Step 2:** 跑这几个测试，确认失败。
- [ ] **Step 3: 实现**（`_relabel(name, dst, extra=None)`、`_confirm(name, form, extra=None)`、`_revert(extra=None)` 把 `extra` 并进记账行；整组撤销循环调 `_revert`，每次先看 `live_records` 栈顶的 `batch`）。
- [ ] **Step 4:** `python -m pytest tests/test_console_labeling.py tests/test_attrs_data.py -q` 通过。
- [ ] **Step 5: 提交** `feat(labeling): 外形页后端给分组和重点核对，整组标注 / 撤销`

### Task 3: 外形页前端：按轨迹、拆出去、重点核对

**Files:**
- Create: `src/skydango/console/static/formgroup.js`（纯函数，末尾 `if (typeof module !== "undefined" && module.exports) module.exports = {…}`）
- Modify: `src/skydango/console/static/labeling.js`（外形部分：开关、列表、缩略图条、按键、撤销、「重点核对」筛选）、`src/skydango/console/static/console.html`（在 `labeling.js` 前引 `formgroup.js`；外形页加开关和缩略图条的容器）、`src/skydango/console/static/console.css`（缩略图条、灰掉 / 拆出去的样式，颜色用变量）
- Test: `tests/test_console_formgroup_js.py`（照 `tests/test_console_markdown.py` 用 node `require` 跑；没有 node 就 skip）、`tests/test_console_page.py` 里已有的静态检查跟着过

**Interfaces:**
- Consumes: Task 2 的 `state.groups` / `state.focus`、`api/form/label` 的 `names` 和返回、`api/form/undo` 的 `batch` 返回。
- Produces（`formgroup.js`）：
  - `buildGroups(items, groups, match) -> [{key, members, shown}]`：`items` 是 state 的条目，`groups` 是名字列表的列表，`match(item) -> bool` 是当前筛选；`members` = 这组所有条目（按组里顺序），`shown` = 其中 `match` 为真的；`shown` 为空的组不返回；没进任何组的条目各自一组；`key` 用组里第一张的名字。
  - `targets(group, excluded: Set<string>) -> string[]`：`shown` 里不在 `excluded` 的名字。
  - `centerIndex(n) -> number`：`Math.floor((n - 1) / 2)`。
  - `focusMatch(item, focus: Set<string>, forms: string[]) -> bool`：在名单里，且（在 `_unlabeled`，或在 `forms` 里某一类但 `confirmed` 为假）。
  - `readByTrack(storage) / writeByTrack(storage, on)`：读写 `skydango.form.byTrack`，异常时读返回 true、写忽略。

- [ ] **Step 1: 写失败的测试**

```python
def test_build_groups_filters_and_singletons(): # 3 张一组其中 1 张不匹配 → shown 2 张；没分组的条目自成一组；shown 空的组不出现
def test_targets_respects_excluded():
def test_center_index():                        # 1→0, 2→0, 5→2
def test_focus_match():                         # _unlabeled 在名单里 → true；form 里已确认 → false；不在名单 → false
def test_read_by_track_defaults_on():           # storage.getItem 抛异常 / 返回 null → true；"0" → false
```

- [ ] **Step 2:** 跑，确认失败（或 node 不在时 skip——本机有 node）。
- [ ] **Step 3: 实现 `formgroup.js`，再接 `labeling.js`：**
  - 开关在外形页顶上（「按轨迹」复选框），状态用 `readByTrack(localStorage)`；关着时外形页的渲染、按键、撤销走原来的代码路径不变。
  - 开着时列表条目 = `buildGroups(...)` 的组，显示张数；当前组上方一排缩略图（`cropUrl`），默认 `centerIndex`；点缩略图换大图和上下文；Shift + 点 / 点角上的 × = 加进 `excluded`（只存在页面内存里，换组清空）。
  - 按键（1~4 / Enter / 0）：开着时发 `post("api/form/label", {names: targets(cur, excluded), to})`，用返回的 `items` 更新本地条目；`skipped` 非空就 `toast` 一句"跳过 N 张：…"。Z 撤销：返回带 `batch` 时用 `items` 更新多条。
  - 筛选下拉加「重点核对（N）」（`state.focus` 非空才出现），匹配用 `focusMatch`。
- [ ] **Step 4:** `python -m pytest tests/test_console_formgroup_js.py tests/test_console_page.py tests/test_console_labeling.py -q` 通过。
- [ ] **Step 5: 提交** `feat(console): 外形页按轨迹整组判、拆出去、重点核对筛选`

### Task 4: 训练集核对名单 `perception crops-audit`

**Files:**
- Create: `src/skydango/vision/crops_audit.py`
- Modify: `src/skydango/cli.py`（`perception` 子命令里加 `crops-audit`）
- Test: `tests/test_crops_audit.py`

**Interfaces:**
- Consumes: `attrs_data` 的 `_Writer`、`_images_in`、`_label_boxes`、`_people`、`crop_name`、`frame_group`、`imread`、`KNOWN`、`PERSON_IDS`、`ID_NAMES`、`UNLABELED_IOU`、`DEDUP_IOU`；`augment.SUFFIXES`；`attrs.crop` / `attrs.CROP_PAD` / `attrs.PERSON_FORMS`；`attrs_train.replay_detector`；`attrs.load_model`。
- Produces:
  - `audit(dataset: Path, attrs_root: Path, detectors: dict[str, Detector], head=None, conf: float = 0.25, low: float = 0.15, accept: float = 0.6) -> dict`：只处理 `images/train`、跳过增强图。返回 `{"missed": [裁图名], "suspect": [...], "partial": [...], "head": [...], "focus": [全部去重后的名单]}`。
    - **missed**：任一检测器（分 ≥ `conf`，`_people` 合过框）出的框和这帧所有人物标注 IoU < `UNLABELED_IOU`、和这张图已裁过的框 IoU < `DEDUP_IOU` → `_Writer.add(..., "_unlabeled", {..., "source": "dataset", "split": "train", "known": False, "found_by": 检测器名})`（检测器之间去重靠 `_Writer.boxes`）。
    - 每个人物标注框（`PERSON_IDS` 里除 3 号 `self`）：数有几个检测器（分 ≥ `low`）出了 IoU ≥ 0.5 的框；0 个 → **suspect**，1 ~ n-1 个 → **partial**，全部 → 交外形头：`head` 不为 None 且（最高类不在 `PERSON_FORMS` 或人形概率之和 < `accept`）→ **head**。
    - suspect / partial / head 用的裁图名 = `crop_name(stem, box)`；`form/` 和 `_unlabeled/` 里都没有这张就照 `crops_from_dataset` 的写法补裁进 `form/<KNOWN[类别]>/`（`known: True`）。
  - `write_focus(attrs_root: Path, names: list[str]) -> Path | None`：已有 `_focus.json` 先改名成 `_focus-<YYYYmmdd-HHMMSS>.json`，再写新名单；返回备份路径或 None。
  - CLI：`python -m skydango perception crops-audit DATASET [--models a,b,c] [--conf 0.25] [--attrs-data datasets/attrs]`；`--models` 不给就只用 `[perception] model`；外形头用 `[attrs] model` + `[attrs] backbone`（加载不了就打印一句"外形头没加载，三个都框到的不再筛"、`head=None`）；最后打印四类数量和 `_focus.json` 路径。检测器用 `attrs_train.replay_detector(cfg.perception, 路径, conf=low)`。

- [ ] **Step 1: 写失败的测试**（假检测器：按图片名返回事先给定的 `Detection` 列表；用 `tmp_path` 造 2~3 帧的 `images/train` + `labels/train`，图片用 numpy 写 jpg）

```python
def test_audit_missed_crops_written():        # 检测器框出标注里没有的人 → _unlabeled 多一张、_crops.jsonl 记 found_by、在 missed 和 focus 里
def test_audit_suspect_and_partial():         # 两个检测器：标注框 A 都没框到 → suspect；框 B 只一个框到 → partial
def test_audit_head_flags_not_person():       # 假外形头对 C 给 not_person 0.9 → head；给 lit 0.9 → 不在名单
def test_audit_skips_augmented_and_val():     # _blur 图、val 帧不处理
def test_write_focus_backs_up(tmp_path):      # 已有 _focus.json → 备份成 _focus-*.json、新文件是新名单
```

- [ ] **Step 2:** 确认失败。 **Step 3:** 实现 `crops_audit.py` 和 CLI。 **Step 4:** `python -m pytest tests/test_crops_audit.py tests/test_attrs_data.py -q` 通过。
- [ ] **Step 5: 提交** `feat(perception): crops-audit 列出训练集该人工核对的人物框`

### Task 5: 文档 + 全量测试

**Files:**
- Modify: `CLAUDE.md`（「代码结构」表 `console/labeling.py` 那行和「管理面板」的「标注」页一段加：外形页按轨迹判（`vision/cropgroups.py`、`static/formgroup.js`）、整组 `batch`、「重点核对」；「常用命令」加 `perception crops-audit` 一行）、`docs/progress/2026-09-28-yolo-training.md`（末尾 11-05 一节：做了什么；下一步 = `crops-audit datasets/sky --models models/sky-yolo-v10.pt,tmp/retrain/control-20261104/yolo/weights/best.pt,models/sky-yolo-v12.pt` → 外形页「重点核对」按轨迹判 → `crops --writeback` → 单类 / 多类各训 3 次比平均）

- [ ] **Step 1:** 改文档。
- [ ] **Step 2:** 后台跑全量 `python -m pytest -q`，和开始前的基线比失败数不增加。
- [ ] **Step 3: 提交** `docs: 外形页按轨迹判、crops-audit`
