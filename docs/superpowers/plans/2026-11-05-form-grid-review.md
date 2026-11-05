# 外形页网格过目 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 外形页加「网格」模式：一页 36 张、默认是人、只点错的、整页确认，用来快速核对训练集人物框"是不是人"；记录带 `scope: "person"`，不算外形确认。

**Architecture:** 记账语义在 `vision/attrs_data.py`（`hand_labels` 跳过 scope 记录、新 `person_checked`）；外形页后端 `FormLabels` 加 `check()` 和 `state` 的 `scores` / `person`；`crops-audit` 顺手写 `_scores.json`；前端纯逻辑放 `static/formgrid.js`（node 可测），`labeling.js` 接界面。写回和外形头训练的代码不改（现有逻辑天然对，用测试钉住）。

**Tech Stack:** Python 3.13、pytest；原生 JS（node 测纯函数）。

**Spec:** `docs/superpowers/specs/2026-11-05-form-grid-review-design.md`

## Global Constraints

- 第一步记录：`{"t", "crop", "from", "to", "scope": "person", "batch"}`；`scope` 只有 `"person"` 这一个值。
- 「是人的概率」= `lit + unlit + spirit + shared + morph`（缺的按 0）；漏标核过是人挪去 `lit / unlit / spirit` 里概率最高的，没有分数 = `lit`。
- 网格一页 **36** 张；排序按是人的概率从低到高，没分数的排最后（同分按裁图名）；已核过是人的不显示。
- 模式开关 `localStorage` 键 `skydango.form.mode`（值 `"single"` / `"grid"`，读写包 try/catch，读不到 = `"single"`）；单张模式逐字照旧。
- `_scores.json` 覆盖写，旧文件先改名 `_scores-<YYYYmmdd-HHMMSS>.json`。
- 页面不引外部资源、不用原生 `confirm` / `prompt` / `alert`，颜色只用 `console.css` `:root` 的变量。
- 测试：worktree 根目录 `python -m pytest -q <文件>`；全量放后台、跑的时候别改文件。注释和文案中文。

## Review Focus

1. 网格里一张裁图在确认前被别处挪走 / 已核过：跳过、其余照做、toast 报出来 —— Task 2 `test_form_check_skips_moved_and_checked`。
2. 同一张出现在两份名单（"是人"和"不是人"）：整次 400、什么都不做 —— Task 2 `test_form_check_duplicate_is_400`。
3. 漏标裁图核过是人、后来被撤销：回到 `_unlabeled`、不再算核过、写回不补框 —— Task 2 `test_form_check_undo_restores_unlabeled`。
4. `_scores.json` 坏了 / 不是对象：`state.scores` 为 `{}`，网格照样能用（全排最后）—— Task 2 `test_form_state_scores_bad_file`。
5. 「是人」名单里有已经在 `form/not_person` 的裁图：跳过（"已经标成不是人"），不改它 —— Task 2 `test_form_check_person_on_not_person_skipped`。

---

### Task 1: 记账语义：`hand_labels` 跳过 scope、`person_checked`

**Files:**
- Modify: `src/skydango/vision/attrs_data.py`（`hand_labels`、新 `person_checked`）
- Test: `tests/test_attrs_data.py`

**Interfaces:**
- Produces: `PERSON_SCOPE = "person"`；`hand_labels(root)` 返回值里不含 `scope` 记录（但 `live_records` 配对时它们照样参与：撤销记录照旧抵消最近一条人标的记录，不管有没有 scope）；`person_checked(root: Path) -> set[str]`：`live_records` 之后最后一条是 `scope == "person"` 的裁图里，现在文件确实在 `form/<to>/` 下的那些。

- [ ] **Step 1: 写失败的测试**

```python
def test_hand_labels_skips_person_scope(tmp_path):   # 一条 from=lit,to=lit,scope=person → hand_labels 里没有它
def test_person_checked_basic(tmp_path):            # 同上且文件在 form/lit → person_checked == {"a.jpg"}
def test_person_checked_undone(tmp_path):           # 后面跟一条 undo → 不在 person_checked
def test_person_checked_moved_away(tmp_path):       # 文件已不在 form/lit → 不在
def test_person_then_form_confirm(tmp_path):        # scope 记录之后又有一条普通 to=unlit → hand_labels == {"a.jpg": "unlit"}，person_checked 不含它
def test_writeback_ignores_person_scope_known(tmp_path):    # known=True 的裁图只有 scope 记录 → writeback 不改标注
def test_writeback_adds_person_checked_missed(tmp_path):    # known=False 的裁图在 form/unlit（scope 记录挪过去的）→ writeback 补一个 4 号框
```

- [ ] **Step 2:** 确认失败（前两条、后两条写回的可能本来就过——过了就保留当回归测试）。
- [ ] **Step 3:** 实现。 **Step 4:** `python -m pytest tests/test_attrs_data.py tests/test_attrs_train.py -q` 通过。
- [ ] **Step 5: 提交** `feat(attrs): 「核过是人」的记录不算外形确认（scope: person）`

### Task 2: 外形页后端：`check`、`state.scores` / `state.person`

**Files:**
- Modify: `src/skydango/console/labeling.py`（`FormLabels.state`、新 `check`）、`src/skydango/console/server.py`（POST `/api/form/check`）
- Test: `tests/test_console_labeling.py`

**Interfaces:**
- Consumes: Task 1 的 `person_checked`、`PERSON_SCOPE`；已有 `_relabel(name, dst, extra)`、`_label_one(name, dst, extra)`、`_Moves._log`、整组 `undo`（按 `batch`）。
- Produces:
  - `SCORES_FILE = "_scores.json"`；`state()` 多 `scores: dict[str, dict[str, float]]`（只留现在还存在的裁图；文件不存在 / 坏了 / 不是对象 = `{}`）、`person: list[str]`（`person_checked` 里现在还存在的）。
  - `FormLabels.check(person: list[str], not_person: list[str], discard: list[str]) -> tuple[int, dict]`：三份名单两两有交集 → `400`；一个 `batch`、持锁逐张：
    - `person`：已核过 / 在 `not_person` / 在 `_discard` / 找不到 → 跳过；在 `_unlabeled` → `_relabel(name, 目标, {"scope": "person", "batch"})`，目标 = `scores` 里 lit / unlit / spirit 最高的，没有 = `lit`；在别的人形类别 → 只记一行 `{"t", "crop", "from": where, "to": where, "scope": "person", "batch"}`。
    - `not_person`：`_label_one(name, "not_person", {"batch"})`；`discard`：`_relabel(name, DISCARD, {"batch"})`；失败的跳过。
    - 返回 `200, {"ok": True, "batch", "items": [_fresh…], "skipped": [{"crop", "text"}]}`；一张都没做成 `409`。
  - `undo()` 不改：scope 原地记录撤销 = 记一条撤销不挪（现有 `_revert` 的 src == dst 分支），挪过的挪回。
  - `server.py`：`"/api/form/check": lambda body: console.form_labels().check(_names(body, "person"), _names(body, "not_person"), _names(body, "discard"))`，`_names` 只收字符串列表、别的当空。

- [ ] **Step 1: 写失败的测试**（照文件里现有 `test_form_*` 的造数据写法）

```python
def test_form_state_scores_and_person(tmp_path):
def test_form_state_scores_bad_file(tmp_path):            # "{oops" → scores == {}
def test_form_check_person_in_place(tmp_path):            # form/lit 的 a → 记一行 scope=person、文件不动、person_checked 含 a、hand_labels 不含 a
def test_form_check_person_unlabeled_moves_by_score(tmp_path):  # _unlabeled 的 b，scores 里 unlit 最高 → 挪进 form/unlit；没分数的 c → form/lit
def test_form_check_not_person_and_discard(tmp_path):     # 各自去 form/not_person / _discard，三类同一个 batch
def test_form_check_duplicate_is_400(tmp_path):
def test_form_check_skips_moved_and_checked(tmp_path):
def test_form_check_person_on_not_person_skipped(tmp_path):
def test_form_check_undo_restores_unlabeled(tmp_path):    # check 之后 undo → b 回 _unlabeled，a 不再算核过，not_person 的回原处
def test_form_api_check_route(tmp_path, upstream, monkeypatch):  # 照 test_form_api_routes
```

- [ ] **Step 2:** 确认失败。 **Step 3:** 实现。 **Step 4:** `python -m pytest tests/test_console_labeling.py -q` 通过。
- [ ] **Step 5: 提交** `feat(labeling): 外形页后端整页「核过是人」确认（api/form/check）`

### Task 3: `crops-audit` 写 `_scores.json`

**Files:**
- Modify: `src/skydango/vision/crops_audit.py`（`_head_flag` 拆出算概率的部分；`audit` 对训练集所有人物标注框和这次的漏标都算概率；新 `write_scores`）、`src/skydango/cli.py`（`crops-audit` 最后写分数、打印路径）
- Test: `tests/test_crops_audit.py`

**Interfaces:**
- Produces: `head_probs(head, img, cls: int | None, box: Rect) -> dict[str, float]`（`cls` 是数据集类别号，漏标传 None 当 `player`）；`audit(...)` 返回值多 `"scores": dict[str, dict[str, float]]`（`head` 为 None 时 `{}`）；`write_scores(attrs_root: Path, scores: dict) -> Path | None`（备份规则同 `write_focus`，空字典不写、返回 None）。原有 `head` 名单的判定（最高类不是人形或人形概率之和 < `accept`）不变，只是改用 `head_probs` 的结果。

- [ ] **Step 1: 写失败的测试** `test_audit_scores_cover_all_label_boxes_and_missed`（假外形头：每个人物标注框和漏标都有分数，`self` 框没有）、`test_audit_no_head_no_scores`、`test_write_scores_backs_up`、`test_cli_crops_audit_writes_scores`。
- [ ] **Step 2:** 确认失败。 **Step 3:** 实现。 **Step 4:** `python -m pytest tests/test_crops_audit.py -q` 通过。
- [ ] **Step 5: 提交** `feat(perception): crops-audit 把外形概率写进 _scores.json`

### Task 4: 前端网格

**Files:**
- Create: `src/skydango/console/static/formgrid.js`（纯函数，末尾 `module.exports`）
- Modify: `src/skydango/console/static/labeling.js`（外形部分加模式切换和网格渲染、键盘 Enter / Z）、`console.html`（在 `labeling.js` 前引 `formgrid.js`；网格容器、底部确认按钮、进度）、`console.css`
- Test: `tests/test_console_formgrid_js.py`（照 `tests/test_console_formgroup_js.py`）、`tests/test_console_page.py` 的 `JS` 列表加 `formgrid.js`

**Interfaces:**
- Consumes: Task 2 的 `state.scores` / `state.person`、`POST api/form/check` 及其返回、`api/form/undo` 的 `batch` 返回、`api/form/context`。
- Produces（`formgrid.js`）：
  - `PAGE = 36`
  - `personProb(scores) -> number | null`（五项之和；`scores` 为空 / undefined = null）
  - `gridItems(items, match, person: Set, scores) -> item[]`：`match` 为真且不在 `person` 里，按 `personProb` 升序、null 排最后、同分按 `crop`。
  - `page(list, i) -> item[]`（第 i 页，从 0 起）
  - `toggle(marks: Map, name, kind) -> Map`：`kind` 是 `"not_person"` / `"discard"`；同一种再点 = 去掉，另一种 = 改成这一种。
  - `submission(pageItems, marks) -> {person, not_person, discard}`：没标的进 `person`。
  - `readMode(storage) / writeMode(storage, mode)`：键 `skydango.form.mode`，异常读返回 `"single"`、写忽略。
- 界面：外形页顶上「单张 / 网格」切换；网格模式隐藏左列表和右按钮区；6 × 6 格子（裁图、角上类别名和概率百分比）；单击 = 不是人（红 ×）、Shift + 单击 = 不要（灰）、双击 = 右侧显示 `api/form/context` 整帧（点别处收起）；底部「这一页确认（Enter）」→ `post("api/form/check", submission(...))`，成功后刷新数据、进下一页，`skipped` 非空 `toast`；**Z** = `api/form/undo`，成功后回到撤掉的那一页并按返回的 `items` 恢复标记（`not_person` / `_discard` 的重新标上）；顶上「核过是人 N / 这个筛选共 M」。

- [ ] **Step 1: 写失败的测试** `test_person_prob`、`test_grid_items_order_and_filter`、`test_page`、`test_toggle_cycles`、`test_submission_defaults_person`、`test_read_mode_defaults_single`。
- [ ] **Step 2:** 确认失败。 **Step 3:** 实现 `formgrid.js` 和界面。 **Step 4:** `python -m pytest tests/test_console_formgrid_js.py tests/test_console_page.py tests/test_console_formgroup_js.py -q` 通过。
- [ ] **Step 5: 提交** `feat(console): 外形页网格过目（一页 36 张，只点错的，整页确认）`

### Task 5: 文档 + 全量测试

**Files:**
- Modify: `CLAUDE.md`（「管理面板」标注页一段加网格过目：两步核对、`scope: person`、`_scores.json`、Enter / Z；代码结构表 `console/labeling.py` 那行提 `formgrid.js`；「感知层第二层」写一句 `scope: person` 不进外形头训练）、`docs/progress/2026-09-28-yolo-training.md`（11-05 一节补：网格过目做完，下一步跑 `crops-audit` → 网格核「重点核对」和训练集「导入未确认」→ `crops --writeback` → 单类 / 多类各训 3 次比平均）

- [ ] **Step 1:** 改文档。 **Step 2:** 后台跑全量 `python -m pytest -q`，和开始前的基线比失败数不增加。
- [ ] **Step 3: 提交** `docs: 外形页网格过目`
