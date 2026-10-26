# 身份答案标注页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在管理面板「标注」页加第五个标签页「认人」，按键给身份板拥挤录像的每条轨迹标标准答案（能在某一帧切段），写回 `truth-draft.toml`，命令行 `--apply` 按段落到每一帧。

**Architecture:** 草稿格式和落帧逻辑放在 `vision/identity_eval.py`（命令行和面板共用）；面板后端 `console/identity_truth.py` 一个类 `IdentityTruth`，读 `tmp/identity-truth/<录像>/` 的草稿和 `logs.json`、现场从录像裁图，`server.py` 挂 `/api/identity/*`；前端 `console/static/identity.js`（纯函数能被 node require）由 `labeling.js` 切标签时调用。

**Tech Stack:** Python 3.13（tomllib、cv2、numpy）、标准库 HTTP 服务（现有 `console/server.py`）、原生 JS（不引外部资源）、pytest + node。

**Spec:** `docs/superpowers/specs/2026-10-27-identity-truth-labeling-design.md`（先读它）。另读 CLAUDE.md「身份板」「管理面板」两节。

## Global Constraints

- 在自己的 git worktree 里做（`using-git-worktrees`），**不许在主目录 `D:\Lanscarlos\Develop\SkyTricolorDango` 切分支**；做完合并进 main 并推送，再按 CLAUDE.md「从 worktree 直接推到远程 main 后，回主目录更新」处理主目录。
- 测试命令：`D:\Lanscarlos\Develop\SkyTricolorDango\.venv\Scripts\python.exe -m pytest -q`（worktree 里也用主目录的 `.venv`，`tests/conftest.py` 会把 worktree 的 `src` 放最前）。全量测试放后台跑。
- 页面文字、报错、注释、提交信息都用中文；提交信息结尾加 `Co-Authored-By` 行（执行方自己的署名）。
- `console.css` 颜色只在 `:root` 定义，别处用 `var(--…)`（有测试查）；页面不引外部资源、请求一律相对路径（`api/identity/...`，不带前导 `/`）；不许用原生 `confirm` / `prompt` / `alert`，用 `common.js` 的 `ask()` / `toast()`；名字、帧名等数据只用 `textContent`。
- 草稿里名字只能是：好友名（friends.md 的 `## 标题` + `reply.friends`，即 `cli._friend_names(cfg)()`）/ `团子` / `陌生人` / `不要` / `?`。
- 分段写法：`t528 = [[0, "玛卡巴卡"], [150.3, "懒洋洋大王"]]`；单个名字 `t147 = "玛卡巴卡"` 等价于 `[[0, "玛卡巴卡"]]`。帧落在「开始 ≤ 帧时间」的最后一段；`不要` / `?` 段不写进 `truth.jsonl`。
- 页面标过的轨迹记在同目录 `reviewed.json`（`{"tracks": [528, 147]}`）；`apply_draft` 不读它。
- 覆盖率 = 看过的轨迹里不在 `?` 段上的框数 / 总框数；≥ 90% 变绿。看过 = 在 `reviewed.json` 里，或草稿值和初判（`draft_identities(logs, cfg.identity.tag_hold)`）不一样。
- 生成草稿、`--apply`、`identity-eval` 留在命令行，页面不做。

## Review Focus

- **旧 `logs.json` 没有 `times`**（`load_logs` 里 `row.get('times', [])`）：帧时间要从帧名 `NNNN_SSS.SSs.jpg` 解析，切段照样落对 → Task 1 `test_frame_times_fallback_to_filename`。
- **切点正好等于某一帧的时间**：那一帧归后一段（开始 ≤ 帧时间）→ Task 1 `test_apply_segments_cut_on_frame`、Task 4 `splitAt` 测试。
- **好友名里带引号 / 反斜杠 / 空格**：渲染 TOML 要转义、读回一致 → Task 1 `test_render_round_trip_odd_names`。
- **好友超过 9 个**：数字键只给前 9 个，其余靠点按钮标；`keyLabel("0")` 是 `?` 不是第 10 个好友 → Task 4 `test_identity_key_labels`。
- **页面打开期间在编辑器里改了草稿**：保存返回 409、草稿不动，页面提示刷新 → Task 2 `test_label_stale_mtime_409`。

---

### Task 1: 草稿的分段写法和按段落帧（`vision/identity_eval.py`）

**Files:**
- Modify: `src/skydango/vision/identity_eval.py`（`write_draft`、`apply_draft`，新加下面几个函数）
- Test: `tests/test_identity_eval.py`

**Interfaces:**
- Produces（Task 2 用）：
  - `Segments = list[tuple[float, str]]`（模块级类型别名）
  - `frame_times(log: TrackLog) -> list[float]`：`len(log.times) == len(log.frames)` 时就是 `log.times`，否则从每个帧名用 `r'_(\d+(?:\.\d+)?)s\.jpg$'` 解析，解析不出抛 `ValueError`。
  - `parse_label(value, line: int) -> Segments`：字符串 → `[(0.0, value)]`；数组 → 每项必须是 `[数字, 字符串]`、开始时间严格递增，否则 `ValueError(f'第 {line} 行：…')`。不在这里查名字是否合法。
  - `check_label(segs: Segments, log: TrackLog, allowed: set[str], line: int, key: str) -> None`：名字不在 `allowed` → `第 {line} 行：{who!r} 不是好友名，也不是团子 / 陌生人 / 不要 / ?`（沿用现在的文字）；第一段开始晚于 `frame_times(log)[0]` → `第 {line} 行：{key} 第一段要从 0 开始`。
  - `label_at(segs: Segments, at: float) -> str`：开始 ≤ `at` 的最后一段；`at` 比第一段还早就是第一段。
  - `read_draft(path: Path) -> tuple[dict[int, Segments], dict[int, int]]`：返回 每条轨迹的分段 + 每条轨迹在文件里的行号（沿用 `apply_draft` 里现在的正则）；TOML 坏了抛 `ValueError`（带 tomllib 报的行号）。
  - `render_draft(labels: dict[int, Segments], logs: dict[int, TrackLog]) -> str`：说明行（现在 `write_draft` 那一句）+ 按编号每行一条；一段写成 `t3 = "名字"`，多段写成 `t528 = [[0, "甲"], [150.3, "乙"]]`（开始时间去掉多余的 0：`0`、`150.3`；名字用 `json.dumps(..., ensure_ascii=False)`），注释照现在的 `# 出现 a~b s，N 帧`。
  - `write_draft(path, ids: dict[int, str], logs)` 保留签名，改成调 `render_draft({id: [(0.0, who)]})`，输出逐字不变。
  - `apply_draft(draft, logs_path, out, *, friends)`：签名不变；用 `read_draft` + `check_label`，每帧按 `label_at(segs, frame_times(log)[i])` 取名字，`不要` / `?` 跳过。仍然先全部验完再写文件。

- [ ] **Step 1: 写失败的测试**（加到 `tests/test_identity_eval.py`，复用文件里现有的 `logs()` 夹具；需要多帧轨迹时自己造 `TrackLog`，帧名用 `f'{i}_{t:.2f}s.jpg'` 格式）

```python
def test_apply_segments_cut_on_frame(tmp_path):
    # 一条 4 帧轨迹，时间 0 / 1 / 2 / 3；切在 2.0：第 0、1 帧是 小明，第 2、3 帧是 陌生人
    ...
    draft.write_text('t1 = [[0, "小明"], [2.0, "陌生人"]]\n', encoding='utf-8')
    assert apply_draft(draft, seen, out, friends=['小明']) == 4
    assert [r['who'] for r in rows(out)] == ['小明', '小明', '陌生人', '陌生人']

def test_apply_segments_skip_unsure_part(tmp_path):
    # [[0, "小明"], [2.0, "?"]] → 只写前两帧

def test_apply_segment_errors_have_line_and_keep_old(tmp_path):
    # 依次写入下面三种第 2 行，各自 pytest.raises(ValueError, match='第 2 行')，out 里原来的「保留」不变：
    #   't1 = [[0, "小明"], [0, "陌生人"]]'   （不递增）
    #   't1 = [[1.5, "小明"]]'                （第一帧时间是 0，第一段晚于第一帧）
    #   't1 = [["小明", 0]]'                  （不是 [数字, 字符串]）

def test_frame_times_fallback_to_filename():
    log = TrackLog(frames=[('0003_000.46s.jpg', Rect(0, 0, 1, 1)), ('0005_000.77s.jpg', Rect(0, 0, 1, 1))])
    assert frame_times(log) == [0.46, 0.77]

def test_render_round_trip_odd_names(tmp_path):
    labels = {1: [(0.0, '小"明\\'), (2.5, '陌生人')], 2: [(0.0, '团 子')]}
    path.write_text(render_draft(labels, logs_for(1, 2)), encoding='utf-8')
    assert read_draft(path)[0] == labels
    assert 't1 = [[0, ' in path.read_text(encoding='utf-8')

def test_write_draft_output_unchanged(tmp_path):
    # write_draft 对现有 logs() 夹具的输出和改之前逐字一样：先在改代码前跑一次，把结果写成字面量断言
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_identity_eval.py -q`
Expected: 新测试 FAIL（`ImportError` / 断言不对），旧测试照常 PASS。

- [ ] **Step 3: 实现上面 Interfaces 里的函数，改 `write_draft` / `apply_draft`**

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_identity_eval.py -q`，再把 `rg -l "apply_draft|write_draft" tests` 列出的其他测试文件一起跑
Expected: 全 PASS。

- [ ] **Step 5: 提交** `feat(identity): 草稿能按帧切段，--apply 按段落到每一帧`

---

### Task 2: 面板后端 `console/identity_truth.py`

**Files:**
- Create: `src/skydango/console/identity_truth.py`
- Test: `tests/test_console_identity_truth.py`

**Interfaces:**
- Consumes：Task 1 的 `read_draft`、`render_draft`、`parse_label`、`check_label`、`frame_times`、`label_at`；现有 `load_logs`、`draft_identities`。
- Produces（Task 3 用）：

```python
class IdentityTruth:
    def __init__(self, root: Path, friends: Callable[[], list[str]], tag_hold: float) -> None
    def runs(self) -> dict                      # {"runs": ["20261026-224826", ...]}：root 下同时有 truth-draft.toml 和 logs.json 的子目录名，按名字倒序
    def state(self, run: str) -> tuple[int, dict]
    def crop(self, run: str, track: int, frame: str) -> bytes | None   # JPEG；None = 404
    def frame(self, run: str, frame: str) -> bytes | None              # JPEG；None = 404
    def label(self, body: dict) -> tuple[int, dict]   # body = {run, track, segments: [[start, who], ...], mtime}
    def undo(self, body: dict) -> tuple[int, dict]    # body = {run, mtime}
```

- `state` 返回 `200, {"ok": True, "run", "mtime", "friends": [...], "friends_note": str|None, "source": str, "coverage": {"done": int, "total": int}, "tracks": [...]}`；每条轨迹：`{"id", "frames": [{"name", "t", "box": [x,y,w,h], "tag": 名字或 null}], "guess", "segments": [[start, who], ...], "reviewed": bool, "two_names": bool}`，按框数从多到少排。`tag` 用 `log.names` 里同一时间的那个名字；`two_names` = `log.names` 里不同名字 ≥ 2；`reviewed` = 在 `reviewed.json` 里，或分段和 `[[0, guess]]` 不一样（Global Constraints 的「看过」）。
- 错误统一 `(状态码, {"ok": False, "text": 中文原因})`：`run` 不合法或不存在 404「没有这段录像的草稿」；草稿 TOML 坏了 / 校验不过 400（带行号的原文）；录像目录（`logs.json` 的 `source`）不在 404「录像目录不在了：…」（只影响 crop / frame，`state` 照常返回、`source_missing: true`）；`mtime` 和文件 `stat().st_mtime_ns` 对不上 409「草稿在别处改过，刷新一下」。
- `run` 校验：`re.fullmatch(r'[\w\-.]+', run)`、不是 `.` / `..`、`(root / run).is_dir()`。`frame` 必须是 logs.json 里出现过的帧名；`crop` 还要求这一帧属于 `track`。
- `friends` 读不到 / 为空：`friends_note = "读不到好友名单（friends.md），只能标团子 / 陌生人 / 不要 / ?"`。
- `crop`：按框四周各扩 15%（夹在画面内），高度 < 160 px 就等比放大到 160，`cv2.imencode('.jpg', …, [IMWRITE_JPEG_QUALITY, 85])`。`frame`：宽度缩到 1280（比它小不放大）。读图用 `cv2.imdecode(np.fromfile(...))`（路径可能有中文）。
- `label`：`segments` 先 `parse_label` 再 `check_label`（`allowed` = 好友 + 四个保留词），首段开始改写成 0；整份草稿读出来、替换这一条、`render_draft` 重写；写之前把当前草稿复制成 `truth-draft.toml.bak`、`reviewed.json` 复制成 `reviewed.json.bak`（没有就写一个空的 `{"tracks": []}` 当 bak）；写临时文件再 `os.replace`；这条轨迹加进 `reviewed.json`。返回 `200, {"ok": True, "mtime": 新的}`。
- `undo`：没有 `.bak` 返回 400「没有可以撤销的」；有就把两个 `.bak` 换回去（换完删掉 `.bak`，只退一步）。
- 草稿里没写的轨迹（logs.json 里有、草稿里没有）当作 `[[0, "?"]]`。

- [ ] **Step 1: 写失败的测试**（`tests/test_console_identity_truth.py`；夹具：`tmp_path/"identity"/"r1"/` 下用 `save_logs` + `write_draft` 造 2~3 条轨迹，录像目录放几张 `cv2.imwrite` 的纯色图，logs 的 `source` 指向它）

```python
def test_state_sorted_coverage_and_two_names(tmp_path)
    # 框多的排前；一条挂过 小明、小红 两个名字 → two_names True；初判写进 guess；什么都没标时 coverage.done == 0
def test_label_splits_and_marks_reviewed(tmp_path)
    # label 切成两段 → 草稿里是 t1 = [[0, "小明"], [2, "陌生人"]]，reviewed.json 有 1，state 里 reviewed True、coverage 增加
def test_adopt_guess_counts_as_reviewed(tmp_path)
    # 用初判原样 label 一次：草稿内容不变，但 coverage 算上这条
def test_hand_edited_line_counts_as_reviewed(tmp_path)
    # 直接改草稿文件里一行（和初判不一样）→ 不在 reviewed.json 也算看过
def test_label_rejects_bad_name_keeps_draft(tmp_path)
    # 名字不是好友 → 400，草稿和 reviewed.json 字节不变，没有 .bak
def test_label_stale_mtime_409(tmp_path)
    # 拿到 state.mtime 后改一下草稿文件 → label 409，文件是手改后的内容
def test_undo_restores_one_step(tmp_path)
    # label 两次 → undo → 草稿回到第一次之后；再 undo → 400「没有可以撤销的」
def test_friends_missing_note(tmp_path)
    # friends 返回 [] → friends_note 有字，label 一个好友名 400
def test_bad_toml_reports_line(tmp_path)
    # 草稿写坏 → state 400，text 里有「行」
def test_paths_refused(tmp_path)
    # run = "..", "a/b", "不存在" → 404；frame 不在 logs 里、frame 不属于 track → crop 返回 None
def test_crop_and_frame_jpeg(tmp_path)
    # crop 返回 b"\xff\xd8" 开头、解码后高 ≥ 160；frame 宽 ≤ 1280；源目录删掉后 crop None、state 照常且 source_missing True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_identity_truth.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `IdentityTruth`**

- [ ] **Step 4: 跑测试确认通过**

Run: 同 Step 2。Expected: 全 PASS。

- [ ] **Step 5: 提交** `feat(console): 认人标注的后端（读草稿、切段写回、撤销、裁图）`

---

### Task 3: 挂到面板服务（`console/server.py`）

**Files:**
- Modify: `src/skydango/console/server.py`（`frames_api` 旁边加工厂方法；`do_GET` 加 4 个路由；`do_POST` 的 `routes` 加 2 个）
- Test: `tests/test_console_identity_truth.py`（追加）

**Interfaces:**
- Consumes：Task 2 的 `IdentityTruth`。
- Produces：
  - `ConsoleServer.identity_truth(self) -> IdentityTruth`：`cfg = self.store._fallback()`；`IdentityTruth(Path("tmp/identity-truth"), _friend_names(cfg), cfg.identity.tag_hold)`，`_friend_names` 从 `skydango.cli` 延迟导入（和 `cli._perception_identity_truth --apply` 用同一份名单）。
  - GET `api/identity/runs` → `runs()`；`api/identity/state?run=` → `state(run)`；`api/identity/crop?run=&track=&frame=` → JPEG 或 404；`api/identity/frame?run=&frame=` → JPEG 或 404（`track` 不是整数当 404）。
  - POST `api/identity/label` → `label(body)`；`api/identity/undo` → `undo(body)`。

- [ ] **Step 1: 写失败的测试** `test_identity_routes(tmp_path, upstream, monkeypatch)`，照 `tests/test_console_frames.py::test_frames_api_routes` 的写法（`monkeypatch.chdir(tmp_path)`，在 `tmp_path/tmp/identity-truth/r1/` 造夹具，`config.toml` 里写 `[reply] memory_dir = "mem"`、`mem/friends.md` 写 `## 小明`）：runs 列出 r1；state 200；crop 的 Content-Type 是 `image/jpeg`；`run=..%2Fx` 404；错 Host 403；POST 不带 `X-Skydango` 403；带 `GOOD` 的 label 200、之后 state 里 reviewed True。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_identity_truth.py::test_identity_routes -q`
Expected: FAIL（404）。

- [ ] **Step 3: 加工厂方法和路由**

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_identity_truth.py tests/test_console_server.py tests/test_console_frames.py -q`
Expected: 全 PASS。

- [ ] **Step 5: 提交** `feat(console): 挂上 /api/identity/*`

---

### Task 4: 前端「认人」标签页

**Files:**
- Create: `src/skydango/console/static/identity.js`
- Modify: `src/skydango/console/static/console.html`（`#lb-tabs` 加按钮；标注页加这一页的容器；`<script>` 放在 `frames.js` 后、`labeling.js` 前）
- Modify: `src/skydango/console/static/labeling.js`（`NOTES`、`setTab`、`reload`、`onKey`、离开页面时）
- Modify: `src/skydango/console/static/console.css`
- Test: `tests/test_console_page.py`（`JS` 列表加 `identity.js`，加 node 测试）

**Interfaces:**
- Consumes：Task 3 的接口（相对路径 `api/identity/...`）；`common.js` 的 `$` `el` `post` `toast`。
- Produces：
  - `window.IdentityTab = {load(), hasData(), key(e), hide()}`，和 `window.FramesTab` 一样的用法：`labeling.js` 切到 `identity` 时 `load()`、按键转给 `key(e)`、离开时 `hide()`。
  - node 导出（`module.exports`，顶层不碰 `document` / `window`）：
    - `splitAt(segs, t) -> segs`：在 `t` 处切一刀，新段沿用原来那段的名字；`t` 已经是切点或 ≤ 0 时原样返回。
    - `removeCut(segs, t) -> segs`：删掉离 `t` 最近的切点（不删第一段的 0）；只有一段时原样返回。
    - `relabel(segs, t, who) -> segs`：把 `t` 所在那段改名，然后合并相邻同名段。
    - `segAt(segs, t) -> index`：开始 ≤ t 的最后一段。
    - `sampleIdx(n, k = 12) -> number[]`：在 0..n-1 里均匀挑 min(n, k) 个下标，含首尾、去重、递增。
    - `keyLabel(key, friends) -> string | null`：`"1"`~`"9"` → `friends[i-1]`（没有就 null）；`d`/`D` 团子、`s`/`S` 陌生人、`x`/`X` 不要、`"0"` → `?`；别的 null。
    - `coverage(tracks) -> {done, total}`：同 Global Constraints 的定义（`reviewed` 或 `segments` 和 `[[0, guess]]` 不同算看过；`?` 段上的帧不算）。
- 段在前端的表示和后端一样：`[[start, who], ...]`，第一段 start = 0。

页面（照 spec §3，布局复用标注页现有的 `cols` / `pane` / `lb-listpane` / `lb-list` / `lb-side` 等类）：
- `id-empty`（没有录像 / 出错的提示条）、`id-cols` 三栏：左 `id-list`（顶上 `id-run` 下拉、`id-filter` 下拉：全部 / 没标的 / 挂过两个名字 / ?，`id-progress` 进度）；中 `id-tagline`（每帧一格的名字时间线，按名字上色、没挂灰色，点一格选中那一帧）、`id-strip`（`sampleIdx` 挑的裁图，每张下写时间和挂的名字，选中的加框；「展开」按钮切换显示全部帧）、`id-segbar`（分段色带）；右 `id-frame`（整帧 `<img>` + 叠一层 `<canvas>` 画框：这条轨迹加粗、同一帧别的轨迹淡色）、`id-keys`（按键说明，好友编号列表；好友每个名字也是一个按钮，点了等同按键）、`id-note`（`friends_note`）。
- 名字到颜色：好友按名单顺序取 `--…` 变量里的一组颜色（在 `console.css` 的 `:root` 加 `--id-c1`…`--id-c6`、`--id-dango`、`--id-stranger`、`--id-skip`、`--id-unsure`，暗色主题也定义）。
- 按键：`1`~`9`/`D`/`S`/`X`/`0` 改选中裁图所在段 → 保存；回车 = `[[0, guess]]` → 保存；`/` 在选中裁图的时间切一刀 → 保存；`Backspace` → `removeCut` → 保存；`←`/`→` 移动选中；`J`/`K` 下一条 / 上一条；`Z` 撤销。按住不放不连着标（`e.repeat` 时只有方向键生效）。
- 保存：`post("api/identity/label", {run, track, segments, mtime})`；成功更新 `mtime`、把改之前的段压进撤销栈；失败（含 409）`toast` 原因、页面上的段退回改之前。标完（没有 `?` 段）后自动跳到下一条没看过的。
- 撤销：栈里有就把上一份段 `label` 回去（不压栈）；栈空（刷新过页面）调 `api/identity/undo` 再重读 `state`。
- 覆盖率 ≥ 90% 时 `id-progress` 加 `data-done="1"`（CSS 变绿），旁边一句「可以去命令行跑 perception identity-truth <录像> --apply，再跑 identity-eval」。

- [ ] **Step 1: 写失败的测试**（`tests/test_console_page.py`；`_node` 写法照文件里已有的）

```python
def test_identity_segments():
    # splitAt([[0,"甲"]], 2) == [[0,"甲"],[2,"甲"]]；splitAt 在已有切点 / t<=0 原样
    # relabel([[0,"甲"],[2,"甲"]], 3, "乙") == [[0,"甲"],[2,"乙"]]；再 relabel(..., 3, "甲") == [[0,"甲"]]（合并）
    # removeCut([[0,"甲"],[2,"乙"],[5,"丙"]], 4.2) == [[0,"甲"],[2,"乙"]]；removeCut([[0,"甲"]], 1) == [[0,"甲"]]
    # segAt([[0,"甲"],[2,"乙"]], 2) == 1（切点正好落在帧上归后一段）
def test_identity_sample_and_coverage():
    # sampleIdx(100) 长 12、首 0 尾 99、递增；sampleIdx(5) == [0,1,2,3,4]
    # coverage：一条 reviewed 的 [[0,"甲"],[2,"?"]] 4 帧(t 0,1,2,3) + 一条没看过、段等于初判的 3 帧 → {done: 2, total: 7}
def test_identity_key_labels():
    # 好友 10 个：keyLabel("9") 是第 9 个，keyLabel("0") == "?"，keyLabel("d") == "团子"，keyLabel("S") == "陌生人"，keyLabel("x") == "不要"，keyLabel("q") is None
def test_identity_tab_in_page():
    # console.html 里有 data-tab="identity"、id="id-cols"；identity.js 在 frames.js 之后、labeling.js 之前；labeling.js 里有 IdentityTab
```

`JS` 列表加 `"identity.js"`（放在 `"frames.js"` 后面），`test_skeleton` 等现有测试会一起查它能解析、颜色只用变量、没有原生弹窗。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_page.py -q`
Expected: 新测试 FAIL。

- [ ] **Step 3: 写 `identity.js`、改 `console.html` / `labeling.js` / `console.css`**

`labeling.js` 的改法：`NOTES.identity = "答案写回 tmp/identity-truth/<录像>/truth-draft.toml，标过的记在 reviewed.json；写完去命令行 --apply"`；`setTab` 认 `identity`、显示 / 隐藏 `id-cols` / `id-empty`、切走时调 `IdentityTab.hide()`；`reload` 里 `identity` → `IdentityTab.load()`；`onKey` 里 `identity` → `IdentityTab.key(e); return`（放在已有的输入框 / 对话框过滤之后，焦点在 `select` 上时不吃键）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_console_page.py tests/test_console_server.py -q`
Expected: 全 PASS。

- [ ] **Step 5: 在浏览器里过一遍**

起面板（在 worktree 根目录：`$env:PYTHONPATH="src"; D:\Lanscarlos\Develop\SkyTricolorDango\.venv\Scripts\python.exe -m skydango console --port 19395 --no-browser`，避开主目录可能在用的 19390；不设 `PYTHONPATH` 跑的是主目录的代码），开 `#labeling` → 认人。没有 `tmp/identity-truth/` 时应显示提示条；在 worktree 的 `tmp/identity-truth/demo/` 用 Task 2 测试的夹具办法造一份小数据再看：列表、裁图、时间线、切段、撤销、409 提示都能用。截图放 `tmp/`。（真录像只在个人电脑主目录的 `tmp/`，不用复制。）

- [ ] **Step 6: 提交** `feat(console): 标注页「认人」：按键标身份答案、在某一帧切段`

---

### Task 5: 文档和收尾

**Files:**
- Modify: `CLAUDE.md`（「代码结构」表加 `console/identity_truth.py` / `static/identity.js` 一行；「身份板」一节「离线」那条后面加一句：草稿可以在管理面板「标注 → 认人」里标，分段写法，`reviewed.json`；「管理面板」一节「标注」页那条提到第五个标签页）
- Modify: `docs/progress/2026-10-22-identity.md`（「10-26 晚」一节后加一行：标注页已实现、还没在真录像上标）

- [ ] **Step 1: 改文档**

- [ ] **Step 2: 全量测试（后台）**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全 PASS。

- [ ] **Step 3: 提交** `docs(identity): 认人标注页的用法`

- [ ] **Step 4: 合并进 main 并推送**，按 Global Constraints 处理主目录（19390 / 19391 / 19392 都没人听、主目录没有未提交改动才 `git pull --ff-only`，否则告诉用户之后再拉）。

## 留给用户的（不在这份计划里做）

spec §6 验证四步：个人电脑上开面板，选 `20261026-224826` 标几条长轨迹和一条挂过两个名字的；编辑器同时改草稿看 409；标到 90% 跑 `perception identity-truth tmp/record/20261026-224826 --apply`；`perception identity-eval tmp/record/20261026-224826`。
