# 难例收件箱 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** live 存下的难例自动进收件箱，外形头先筛、人在标注页逐张判裁图 + 整帧过目 / 编辑，通过的帧进 `datasets/sky`，攒够了在面板上一键重训 YOLO + 外形头并出对比报告，换不换模型由人点。

**Architecture:** 纯数据和文件操作放 `vision/inbox.py`（收、分边、去重、分流、帧状态、通过 / 撤销）和 `vision/retrain.py`（版本号、训练、对比、报告）；面板侧 `console/jobs.py`（任务槽：不带接口的子进程）和 `console/frames.py`（整帧页接口）；前端 `static/frames.js`。帧状态不存盘、每次按 `datasets/attrs` 里裁图现在在哪现算。

**Tech Stack:** Python 3.13、OpenCV、ultralytics（YOLO 训练 / val）、onnxruntime（DINOv2 主干）、原生 JS（无框架）、pytest、node（测 JS 纯函数）。

**Spec:** `docs/superpowers/specs/2026-10-04-hardcase-inbox-design.md`（下文 §n 指它的章节）

## Global Constraints

- 界面、日志、报告、提交信息都用中文；代码风格照周围代码（中文注释、`from __future__ import annotations`）
- 测试：`.venv\Scripts\python.exe -m pytest -q`（在 worktree 里同样用主目录的 `.venv`；`tests/conftest.py` 会把 worktree 的 `src` 放最前面）；GPU / 真模型的部分用假检测器、假特征模型、假 train 函数
- 页面：不许原生 `confirm` / `prompt` / `alert`（用 `ask()` / `toast()`）；颜色只在 `console.css` 的 `:root` 定义；请求一律相对路径；不引外部资源
- 面板接口：GET 校验 Host；POST 要 `X-Skydango: 1` + JSON + ≤ 64 KB（复用现有 `post_guard` / Host 校验）
- `config.toml` 面板只读不写；面板改设置写 `console.toml`
- 默认值（§8）：`inbox.enabled = true`、`inbox.dir = "datasets/inbox"`、`inbox.ask = true`、`inbox.agree = 0.9`、`inbox.dup_diff = 6`、`inbox.dup_gap = 5`（秒）、`inbox.val_every = 5`、`inbox.retrain_min = 50`；`retrain.base = "models/yolo11n.pt"`、`imgsz = 960`、`epochs = 120`、`batch = 16`、`workers = 2`
- 外形类别对应（§4）：player ↔ lit、player_unlit ↔ unlit、spirit ↔ spirit；外形头的 shared / morph 并进 lit
- `datasets/attrs` 的位置：`Path("datasets/attrs")`（同 `FormLabels` / `perception crops` 默认）；数据集：`Path("datasets/sky")`
- 收件箱帧名 = `<运行目录名>_<原文件名去后缀>`（同 `weaklabel.hard_images`）；分边 `zlib.crc32(run.encode()) % val_every == 0` → `"val"`
- 编辑模式按键（§5.2）：1 player 2 player_unlit 3 self 4 spirit 5 name_tag 6 social_ring 7 typing 8 bench 9 bonfire 0 instrument；过目模式：回车 通过、E 编辑、0 整帧不要、Z 撤销、← → 翻帧、H / 按住空格 隐藏框
- 改完合并进 main 并推送（CLAUDE.md）；不在主目录切分支

## Review Focus

- **整帧页取图的路径**：`/api/frames/image?frame=…` 只能读收件箱里 `frames.json` 登记过的原图，`..`、绝对路径、没登记的帧名一律 404（Task 7 加测试）
- **帧通过之后外形页又改了那张裁图**：通过时的标注已经写死进 `datasets/sky`，帧状态保持 `done`、不回到 `crops`（Task 3 加测试）
- **编辑器画出退化的框**（反向拖、宽高 < 4 px、拖出画面）：保存前归一（左上角 + 正宽高）、裁进画面、太小的丢掉（Task 7 后端校验 + Task 8 JS 纯函数各加测试）
- **模型文件缺失 / 加载失败**：整理任务一开始就失败、给出中文原因，已有的 `frames.json` 不被改动（Task 5 加测试）
- **团子被强杀（没走 `_stop_scene`）后点「整理」**：整理开头先补收所有 `runs/*/hard/`（Task 5 加测试）

---

### Task 1: 配置和设置清单

**Files:**
- Modify: `src/skydango/config.py`（`AttrsConfig` 附近加两个 dataclass，`Config` 加两个字段）
- Modify: `config.example.toml`（加 `[inbox]`、`[retrain]` 两节，注释照 §8）
- Modify: `src/skydango/console/settings.py`（`Field` 清单）
- Test: `tests/test_inbox.py`（新）、`tests/test_console_settings.py`

**Interfaces:**
- Produces: `InboxConfig(enabled: bool = True, dir: str = "datasets/inbox", ask: bool = True, agree: float = 0.9, dup_diff: float = 6.0, dup_gap: float = 5.0, val_every: int = 5, retrain_min: int = 50)`、`RetrainConfig(base: str = "models/yolo11n.pt", imgsz: int = 960, epochs: int = 120, batch: int = 16, workers: int = 2)`；`Config.inbox`、`Config.retrain`
- Produces: 设置清单新增 `inbox.enabled`、`inbox.ask`、`inbox.retrain_min`（features / 数据分组，照现有分组挑最近的）和 `perception.model`、`attrs.model`（字符串，供 Task 11 换模型写 `console.toml`）

- [ ] **Step 1: 写失败的测试**

```python
def test_inbox_defaults():
    cfg = load_config(None)
    assert cfg.inbox.enabled and cfg.inbox.ask and cfg.inbox.dir == "datasets/inbox"
    assert (cfg.inbox.agree, cfg.inbox.dup_diff, cfg.inbox.dup_gap, cfg.inbox.val_every, cfg.inbox.retrain_min) == (0.9, 6.0, 5.0, 5, 50)
    assert (cfg.retrain.base, cfg.retrain.imgsz, cfg.retrain.epochs, cfg.retrain.batch, cfg.retrain.workers) == ("models/yolo11n.pt", 960, 120, 16, 2)

def test_inbox_from_toml(tmp_path):
    p = tmp_path / "c.toml"; p.write_text("[inbox]\nask = false\nretrain_min = 30\n[retrain]\nepochs = 3\n", encoding="utf-8")
    cfg = load_config(p)
    assert not cfg.inbox.ask and cfg.inbox.retrain_min == 30 and cfg.retrain.epochs == 3
```

`tests/test_console_settings.py` 加：清单里有 `inbox.enabled` / `inbox.ask` / `inbox.retrain_min` / `perception.model` / `attrs.model` 五个键；`config.example.toml` 能被 `load_config` 读且 `[inbox]`、`[retrain]` 的值等于默认值。

- [ ] **Step 2: 跑测试确认失败** — `.venv\Scripts\python.exe -m pytest tests/test_inbox.py tests/test_console_settings.py -q`，预期 `AttributeError: 'Config' object has no attribute 'inbox'`
- [ ] **Step 3: 实现**（照 `AttrsConfig` 的写法；`load_config` 已按字段名映射 toml 节，确认新节不需要额外注册）
- [ ] **Step 4: 跑测试确认通过**（同 Step 2 命令，全过）
- [ ] **Step 5: 提交** — `git commit -m "feat(inbox): [inbox] [retrain] 配置和设置清单"`

---

### Task 2: 收（collect）+ 下线时自动收 + `perception inbox collect`

**Files:**
- Create: `src/skydango/vision/inbox.py`
- Modify: `src/skydango/cli.py`（`_stop_scene` 加参数；`perception` 子命令加 `inbox collect`）
- Test: `tests/test_inbox.py`

**Interfaces:**
- Produces（`vision/inbox.py`）：
  - `split_of(run: str, every: int) -> str`：`"val"` / `"train"`（Global Constraints 的公式）
  - `frame_name(run: str, file: str) -> str`：`f"{run}_{Path(file).stem}"`
  - `collect(run_dir: Path, inbox: Path) -> int`：复制 `run_dir/hard/*.jpg` 到 `inbox/<run>/raw/`、`hard.jsonl` 到 `inbox/<run>/hard.jsonl`，已存在的文件跳过；没有 `hard/` 或 0 张返回 0 且不建目录；第一次收某个运行时往 `inbox/_index.jsonl` 追加 `{"run", "collected_at", "frames"}`；返回新复制的张数
  - `collect_all(runs: Path, inbox: Path) -> list[str]`：对 `runs/*/` 逐个 `collect`，返回有新复制的运行名
  - `pending_runs(inbox: Path) -> list[str]`：收了、还没 `processed_at` 的运行（读 `_index.jsonl`，同一运行多行取最后一行的字段合并）
- Produces（`cli.py`）：`_stop_scene(env, inbox: InboxConfig | None = None)`；`inbox` 非 None 且 `enabled` 且 `env.hardcases.saved > 0` 时 `collect(env.hardcases.folder.parent, Path(inbox.dir))`，任何异常 `log.warning` 吞掉；调用处都传 `cfg.inbox`

- [ ] **Step 1: 写失败的测试**

```python
def test_split_of_whole_run_same_side():
    assert split_of("20261004-192711-live-brain", 5) in ("train", "val")
    vals = [r for r in (f"run{i}" for i in range(100)) if split_of(r, 5) == "val"]
    assert 10 <= len(vals) <= 30 and all(split_of(r, 5) == "val" for r in vals)

def test_collect_copies_and_is_idempotent(tmp_path):
    run = tmp_path / "runs" / "20261004-192711-live-brain"; (run / "hard").mkdir(parents=True)
    for n in ("192007_attrs_disagree.jpg", "192102_low_conf.jpg"): (run / "hard" / n).write_bytes(b"x")
    (run / "hard.jsonl").write_text('{"file": "192007_attrs_disagree.jpg"}\n', encoding="utf-8")
    inbox = tmp_path / "inbox"
    assert collect(run, inbox) == 2 and collect(run, inbox) == 0
    assert sorted(p.name for p in (inbox / run.name / "raw").iterdir()) == ["192007_attrs_disagree.jpg", "192102_low_conf.jpg"]
    assert (inbox / run.name / "hard.jsonl").is_file() and pending_runs(inbox) == [run.name]

def test_collect_nothing(tmp_path):
    run = tmp_path / "r"; run.mkdir()
    assert collect(run, tmp_path / "inbox") == 0 and not (tmp_path / "inbox" / "r").exists()

def test_stop_scene_collects_and_never_raises(tmp_path, monkeypatch):
    # 假 env：hardcases.saved = 1、folder = <run>/hard；collect 抛异常时 _stop_scene 照常返回；enabled = False 不收
    ...
```

（最后一个测试照 `tests/` 里现有假 env 的写法补全：一个只有 `stop()`、`hardcases`、`unknown = None`、`catalog = None` 的对象。）

- [ ] **Step 2: 跑测试确认失败** — `.venv\Scripts\python.exe -m pytest tests/test_inbox.py -q`，预期 `ImportError`
- [ ] **Step 3: 实现** `inbox.py` 上面四个函数 + `_stop_scene` 参数 + `perception inbox collect [runs]`（默认 `runs`，打印「收了 N 次运行 / 共 M 张 → datasets/inbox」）；`_stop_scene` 原来那句「收进数据集：perception label runs --from-runs」改成「已收进 datasets/inbox，管理面板里整理」（收了时）
- [ ] **Step 4: 跑测试确认通过**；再跑 `tests/test_cli*.py -q` 确认 `_stop_scene` 调用处没漏传
- [ ] **Step 5: 提交** — `feat(inbox): 下线时把难例收进 datasets/inbox，perception inbox collect`

---

### Task 3: 分流、帧状态、最终标注（纯计算）

**Files:**
- Modify: `src/skydango/vision/inbox.py`
- Test: `tests/test_inbox.py`

**Interfaces:**
- Consumes: `attrs.FORMS`、`attrs.PERSON_FORMS`、`attrs_data.hand_labels`
- Produces:
  - `PAIR = {"player": "lit", "player_unlit": "unlit", "spirit": "spirit"}`；`MERGE_FORM = {"shared": "lit", "morph": "lit"}`
  - `@dataclass Route: auto: str | None  # "agree" / "drop_low" / None（给人判）；form: str  # 外形头的类（并过 shared/morph）；p: float`
  - `route(yolo_cls: str, score: float, probs: dict[str, float], conf: float, agree: float) -> Route`：§4 的六行表
  - `crop_place(attrs_root: Path, crop: str) -> str | None`：裁图现在在 `"_unlabeled"` / `"_discard"` / `FORMS` 里的哪一个；都不在返回 None
  - `frame_state(entry: dict, place: Callable[[str], str | None]) -> str`：`"done" | "discarded" | "crops" | "edit" | "glance" | "dup" | "error"`（§2.2 的顺序；`dup_of` 非空 = `dup`，`error` 非空 = `error`）
  - `final_boxes(entry: dict, place: Callable[[str], str | None], classes: list[str]) -> list[tuple[int, Rect]]`：自动一致的用 YOLO 框和类别；有裁图且人判过的按判的类（`WRITEBACK_ID` 的映射：lit/morph/shared → player、unlit → player_unlit、spirit → spirit；not_person → 去掉）；`drop_low` 去掉；非人物框原样；返回类别编号（`classes.index`）

- [ ] **Step 1: 写失败的测试**

```python
@pytest.mark.parametrize("cls,score,probs,want", [
    ("player", 0.8, {"lit": 0.95, "unlit": 0.03, "not_person": 0.02}, "agree"),
    ("player", 0.8, {"not_person": 0.97, "lit": 0.03}, None),          # 高分框判不是人：给人看
    ("player_unlit", 0.6, {"lit": 0.7, "unlit": 0.3}, None),           # 不一致
    ("player_unlit", 0.3, {"unlit": 0.93, "lit": 0.07}, "agree"),      # 低分框一致：补成正式框
    ("player", 0.3, {"not_person": 0.92, "lit": 0.08}, "drop_low"),
    ("player", 0.3, {"not_person": 0.6, "lit": 0.4}, None),
    ("player", 0.8, {"shared": 0.95, "lit": 0.05}, "agree"),           # shared 并进 lit
])
def test_route(cls, score, probs, want):
    assert route(cls, score, probs, conf=0.35, agree=0.9).auto == want

def test_frame_state_order_and_done_sticks():
    e = {"boxes": [{"cls": "player", "crop": "a.jpg", "auto": None}], "decision": None, "dup_of": None, "error": None, "editing": False}
    assert frame_state(e, lambda c: "_unlabeled") == "crops"
    assert frame_state(e, lambda c: "lit") == "glance"
    assert frame_state(e, lambda c: "_discard") == "edit"
    assert frame_state({**e, "editing": True}, lambda c: "lit") == "edit"
    assert frame_state({**e, "decision": {"what": "pass"}}, lambda c: "_unlabeled") == "done"   # 通过后外形页再改也不回 crops

def test_final_boxes():
    # 一帧四个框：自动一致的 player、人判成 unlit 的 player、人判 not_person 的、drop_low 的，外加一个 name_tag
    # → [(0, A), (4, B), (1, tag)]（顺序按 boxes 原顺序）
    ...
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `feat(inbox): 外形头分流、帧状态现算、最终标注`

---

### Task 4: 通过 / 不要 / 撤销（文件操作）

**Files:**
- Modify: `src/skydango/vision/inbox.py`
- Test: `tests/test_inbox.py`

**Interfaces:**
- Consumes: Task 3 的 `final_boxes`、`crop_place`；`weaklabel.yolo_line(cls, box, width, height)`（YOLO 行格式）
- Produces（都以 `inbox: Path`、`run: str`、`frame: str` 定位；`frames.json` 读写经 `load_frames(inbox, run) -> dict` / `save_frames(inbox, run, frames)`，写法：先写 `.tmp` 再 `os.replace`）：
  - `pass_frame(inbox, run, frame, attrs_root, dataset, classes, boxes: list[tuple[int, Rect]] | None = None) -> str`：`boxes` 为 None 用 `final_boxes`，否则用给的（编辑保存）并标 `edited = True`；写 `inbox/<run>/labels/<frame>.txt`；复制原图和标注到 `dataset/images/<split>/<frame>.jpg`、`dataset/labels/<split>/<frame>.txt`；任一目标已存在 → `FileExistsError`（不写任何东西）；复制到一半失败 → 删掉已复制的、再抛；成功后 `decision = {"what": "pass", "edited": …, "t": time.time(), "dataset": f"{split}/{frame}"}`、`editing = False`；返回 `"<split>/<frame>"`
  - `discard_frame(inbox, run, frame) -> None`：`decision = {"what": "discard", "t": …}`
  - `undo_frame(inbox, run, frame, dataset) -> None`：通过的删掉 `datasets/sky` 那两个文件；`decision = None`
  - `set_editing(inbox, run, frame, on: bool) -> None`

- [ ] **Step 1: 写失败的测试** — `test_pass_copies_into_dataset`（labels 内容 = `final_boxes` 的 YOLO 行、split 对、decision 写了）、`test_pass_with_edited_boxes`、`test_pass_refuses_clash`（预先放一个同名 `images/train/<frame>.jpg` → `FileExistsError`、`decision` 还是 None、标注没写）、`test_pass_rolls_back_on_copy_failure`（monkeypatch `shutil.copyfile` 第二次抛 `OSError` → 两个目标都不在）、`test_undo_pass_removes_files`、`test_discard_and_undo`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `feat(inbox): 整帧通过 / 不要 / 撤销`

---

### Task 5: 整理流水线 + `perception inbox process|status`

**Files:**
- Modify: `src/skydango/vision/inbox.py`、`src/skydango/cli.py`（`attrs_data._Writer` 原样复用：`add(img, stem=帧名, box, folder, row)` 已接受任意 `row`，不改它）
- Test: `tests/test_inbox.py`、`tests/test_attrs_data.py`

**Interfaces:**
- Consumes: Task 2 `collect_all` / `pending_runs` / `split_of` / `frame_name`，Task 3 `route`，`perception.merge_people`，`attrs_data._Writer`、`attrs.crop` / `attrs.mask_crop` / `attrs.CROP_PAD`，`detect.Detection`
- Produces:
  - `similar(a: np.ndarray, b: np.ndarray, diff: float) -> bool`：两张图各缩成 1/8 灰度、平均绝对差 < diff
  - `frame_time(file: str) -> float | None`：文件名开头 `HHMMSS` → 秒；没有 → None（不按时间去重，只按像素差）
  - `process(inbox: Path, attrs_root: Path, runs: Path, cfg: Config, detect: Callable[[np.ndarray], list[Detection]], weak: Callable[[np.ndarray], list[tuple[str, Rect]]], judge: Callable[[list[np.ndarray]], list[dict[str, float]]], progress: Callable[[str], None] = print) -> dict`：
    1. `collect_all(runs, inbox)`
    2. 对 `pending_runs(inbox)` 按名字排序逐个：读 `frames.json`（没有就空），对 `raw/` 里按文件名排序的每张、`frames.json` 里还没有的帧：去重（和上一张保留帧 `similar` 且 `frame_time` 相差 ≤ `dup_gap` → `{"dup_of": 那帧}`）→ `detect`（低于 `low_conf` 的丢）+ `weak` → 人物框（player / player_unlit / spirit，`merge_people`）裁图 → `judge` 一次给这帧所有裁图 → `route` → 人物框写 `boxes`（`crop` = `_Writer` 生成的名字，`source = "inbox"`、`image` = 原图绝对路径、`group` = 运行名；`auto = "agree"` 的裁图放 `form/<form>/` 并往 `attrs_root/_labels.jsonl` 追加 `{"t", "crop", "from": "_unlabeled", "to": form, "by": "auto-agree", "p"}`；给人判的放 `_unlabeled/` 并往 `_unlabeled/claude.json` 写 `{"label", "confidence", "model": "attrs-screen", "reason": "YOLO 判<中文> <分>，外形头判<中文> <p>"}`；`drop_low` 不裁图）；非人物框 ≥ `conf` 的写进 `boxes`（`crop = None`、`auto = None`）；单帧异常记 `error` 跳过；**每帧写一次 `frames.json`**；每帧 `progress(f"PROGRESS {i}/{n} {run}")`
    3. 一次运行做完：`_index.jsonl` 追加 `{"run", "processed_at", "counts": {各状态}}`；`_stats.json` 更新 `sec_per_frame`
    4. 返回 `{"runs", "frames", "dups", "auto", "to_judge", "glance"}`
  - `status(inbox: Path, attrs_root: Path) -> dict`：每次运行各状态帧数、`judge_left`（来自整理、还在 `_unlabeled` 的裁图数）、`passed_since_train`（`decision.what == "pass"` 且 `t` > `_stats.json` 的 `trained_at`）
  - CLI：`perception inbox process` 建检测器（`[perception] model`、`low_conf`、`device`）、外形头（同 `screen_new.py`：`OnnxEmbedder(cfg.attrs.backbone, …)` + `attrs.load_model`，`judge` 里按 `model.keep("form")` 先 `mask_crop`）、`weak` = `_weak_boxes`（`args` 用 `SimpleNamespace(all_text=False, min_score=默认)`）；模型文件不存在 / 加载失败 → 打印中文原因、退出码 1、不碰收件箱；结束打印 §4 第 6 条那句；`perception inbox status` 打印 `status()`

- [ ] **Step 1: 写失败的测试**（假 `detect` / `weak` / `judge`，合成 1920×1080 图）
  - `test_process_routes_and_writes`：一帧两个人（一个一致 0.95、一个不一致）+ 一个 name_tag → `frames.json` 三个框；一张裁图在 `form/lit/`、`_labels.jsonl` 有 `by = auto-agree`；一张在 `_unlabeled/`、`claude.json` 理由含"YOLO 判"
  - `test_process_dedupes`：两张几乎一样、文件名相隔 2 秒 → 第二张 `dup_of`；相隔 10 秒 → 不去重
  - `test_process_resumes`：第一次处理到第 2 帧时 `detect` 抛 `KeyboardInterrupt`；再跑一次 → 第 1 帧不再调用 `detect`
  - `test_process_frame_error_skips`：`detect` 对某帧抛 `ValueError` → 那帧 `error`、其余照常
  - `test_process_collects_first`：`runs/<r>/hard/` 有图、收件箱是空的 → 处理到它（强杀后补收）
  - `test_inbox_process_missing_model`（CLI）：`[attrs] model` 指向不存在的文件 → 退出码 1、输出含"外形头"、收件箱里已有的 `frames.json` 内容不变
  - `test_writeback_skips_inbox_rows`（`tests/test_attrs_data.py`）：`_crops.jsonl` 里一条 `source = "inbox"` 的行、裁图在 `form/lit/` → `writeback` 不改任何 labels
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；`tests/test_attrs_data.py` 全过
- [ ] **Step 5: 提交** — `feat(inbox): 整理流水线（去重、预标注、外形头分流），perception inbox process / status`

---

### Task 6: 面板任务槽、停止时整理、收件箱摘要

**Files:**
- Create: `src/skydango/console/jobs.py`
- Modify: `src/skydango/console/server.py`
- Test: `tests/test_console_jobs.py`（新）、`tests/test_console_server.py`

**Interfaces:**
- Consumes: `runner._popen_flags`、`runner.kill_tree`、`child_env`；Task 5 的 `inbox.status`、`inbox.pending_runs`
- Produces:
  - `class JobRunner(cwd: Path, kill=kill_tree, popen=subprocess.Popen)`：`start(job: str, cmd: list[str], env: dict) -> None`（`job ∈ {"inbox", "retrain"}`，忙时 `RuntimeError("…还在跑")`）；`stop() -> None`（进程树强杀，状态 `stopped`）；`status() -> dict`：`{"state": "idle|running|done|failed|stopped", "job", "progress": 最后一行 "PROGRESS " 之后的文字, "tail": 最后 20 行, "exit_code", "log": 日志文件路径}`；输出逐行写 `tmp/jobs/<时间>-<job>.log`
  - `ConsoleServer`：`self.jobs = JobRunner(...)`；`POST /api/jobs/start {"job": "inbox"|"retrain"}`、`POST /api/jobs/stop`；`/api/state` 多 `"job": jobs.status()` 和 `"inbox": {"pending": len(pending_runs), "judge_left", "glance", "edit", "passed_since_train", "retrain_min"}`（`inbox.enabled = false` 时 `"inbox": null`）
  - 互斥：`start_run` / `start_sandbox` 在任务 `running` 时返回 409 `{"ok": False, "job": jobs.status(), "error": "整理还没完…"}`，body 带 `"stop_job": true` 时先停任务再起；`/api/jobs/start` 在团子 / 沙盒槽非空闲时 409
  - `GET /api/inbox/stop-info`：`{"hard": 当前运行目录 hard/ 的 jpg 数, "pending": 没整理的运行数, "eta_min": 估计分钟（_stats.json 的 sec_per_frame × (hard + 待整理帧数) / 60，没有记录按 1.5 秒 / 帧）, "ask": inbox.ask}`
  - `stop_run(body)`：`body.get("curate")` 为真 → 记下 `self._curate_after = True`，起一个守护线程等团子槽变 `exited` / `crashed`（最多 `stop_timeout + 30` 秒）后 `jobs.start("inbox", [sys.executable, "-m", "skydango", "-c", <config>, "perception", "inbox", "process"], env)`
- [ ] **Step 1: 写失败的测试** — `JobRunner` 用假 `popen`（stdout 是几行文字、其中 `PROGRESS 3/10 r1`）测 `status()` 的 progress / tail / done / failed；忙时再 `start` 抛；`stop` 调了 `kill`；server：任务 running 时叫醒 409、带 `stop_job` 先停再起；团子在跑时 `/api/jobs/start` 409；`stop_run({"curate": true})` 后把假 runner 状态改成 `exited` → 任务被起、命令行含 `perception inbox process`；`stop-info` 的张数和估时
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；`tests/test_console_*.py -q` 全过
- [ ] **Step 5: 提交** — `feat(console): 任务槽（整理 / 重训）、停团子后接着整理、收件箱摘要`

---

### Task 7: 整帧页后端 + 外形页「来自整理」

**Files:**
- Create: `src/skydango/console/frames.py`
- Modify: `src/skydango/console/server.py`（路由）、`src/skydango/console/labeling.py`（`FormLabels._item` 多 `"inbox": r.get("source") == "inbox"`）
- Test: `tests/test_console_frames.py`（新）、`tests/test_console_labeling.py`

**Interfaces:**
- Consumes: Task 3 / 4 的全部；`attrs.crop`、`attrs_data._Writer`
- Produces（`class FramesApi(inbox: Path, attrs_root: Path, dataset: Path, classes: list[str])`）：
  - `state() -> dict`：`{"frames": [{"frame", "run", "state", "reason", "split", "boxes": [{"cls", "box", "src": "auto|judged|yolo|drawn", "dropped": bool}], "w", "h"}], "counts": {各状态}}`（`dup` 不列出）；`boxes` 是按当前判断合成的（`final_boxes` 之前的样子，`drop_low` 标 `dropped`、not_person 的不列）
  - `image(frame: str) -> Path | None`：只认 `frames.json` 里登记过的帧，返回 `inbox/<run>/<file>`；否则 None（server 回 404）
  - `act(body: dict) -> tuple[int, dict]`：`{"frame", "do": "pass"|"discard"|"edit"|"cancel_edit"|"undo", "boxes"?: [{"cls": int, "box": [x, y, w, h]}]}`；`pass` 带 `boxes` = 编辑保存：后端先归一（负宽高翻正、裁进 `[0, w) × [0, h)`、宽或高 < 4 px 丢掉、`cls` 不在 `range(len(classes))` 的 400），人物类（0 / 4 / 9）里和原框 IoU < 0.5 或类别变了的框裁一张图进 `attrs_root/form/<lit|unlit|spirit>/`、`_labels.jsonl` 记 `by = "frame-edit"`、`_crops.jsonl` 记 `source = "inbox"`；`FileExistsError` → 409 `"datasets/sky 里已经有同名的帧"`
  - 路由：`GET /api/frames/state`、`GET /api/frames/image?frame=`、`POST /api/frames/act`
- [ ] **Step 1: 写失败的测试** — 状态列表和计数；外形页挪裁图后状态从 crops 变 glance；`image("../x")`、绝对路径、没登记的帧 → None；编辑保存的归一（`[100, 100, -50, -60]` → `[50, 40, 50, 60]`、`[1900, 10, 100, 50]` 裁成宽 20、`[0, 0, 3, 50]` 丢掉）；新画的黑影框进 `form/unlit/` 且记 `frame-edit`；撤销；重名 409；`FormLabels` 条目有 `inbox`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `feat(console): 整帧页接口（过目、编辑保存、撤销），外形页标出来自整理的裁图`

---

### Task 8: 整帧页前端（过目 + 编辑）和外形页筛选

**Files:**
- Create: `src/skydango/console/static/frames.js`
- Modify: `src/skydango/console/static/console.html`（`lb-tabs` 加「整帧」、整帧页的骨架）、`labeling.js`（「来自整理」筛选：`c.inbox === true`，计数照 `replay` 的写法）、`console.css`（框颜色用 `stage.js` 现有配色的变量；新颜色在 `:root` 定义）
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 7 的三个接口；`common.js` 的 `$` `el` `post` `ask` `toast`；`stage.js` 的类别配色
- Produces（`frames.js` 末尾 `if (typeof module !== "undefined" && module.exports) module.exports = {toImage, toScreen, normBox, hitTest, resizeBox, keyClass, KEY_CLASS}`，顶层不碰 `document`）：
  - `toImage(pt, view) / toScreen(pt, view)`：`view = {scale, ox, oy}`（屏幕 = 图像 × scale + 偏移）
  - `normBox(b, w, h) -> b | null`：同 Task 7 后端的归一规则（前后端都做）
  - `hitTest(boxes, pt, tol) -> {i, part: "move"|"nw"|"ne"|"sw"|"se"|"n"|"s"|"w"|"e"} | null`（边角优先、最小的框优先）
  - `resizeBox(b, part, dx, dy) -> b`
  - `KEY_CLASS = {"1": 0, "2": 4, "3": 3, "4": 9, "5": 1, "6": 2, "7": 5, "8": 6, "9": 7, "0": 8}`（编号 = `[perception] classes` 顺序：player 0、name_tag 1、social_ring 2、self 3、player_unlit 4、typing 5、bench 6、bonfire 7、instrument 8、spirit 9；按 Global Constraints 的键位）；`keyClass(key) -> int | null`
  - 页面行为照 §5.2：过目 / 编辑两种模式（页面顶上一直显示「过目」/「编辑中」），列表筛选，按键表，滚轮缩放（以光标为中心）、按住空格拖动，H / 按住空格隐藏框（编辑模式里空格只用来拖画面），`dropped` 框淡虚线，回车通过后自动跳下一帧
- [ ] **Step 1: 写失败的测试** — node 跑：`toImage(toScreen(p))` 往返；缩放 2 倍偏移 (100, 50) 的换算；`normBox` 三个例子同 Task 7；`hitTest` 点在角上返回角、点在框内返回 move、两框重叠取小的；`resizeBox("se", 10, 20)`；`keyClass("2") === 4`、`keyClass("0") === 8`、`keyClass("x") === null`；`node --check frames.js`；静态页里没有 `confirm(` / `prompt(` / `alert(`（现有测试会扫到新文件，确认它覆盖 `frames.js`）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；用浏览器面板开管理面板（`console --port 19396 --no-browser` 起在 worktree，`PYTHONPATH=src`），用 Task 5 的测试数据造一个小收件箱，在整帧页过目 / 编辑 / 撤销各点一遍，截图
- [ ] **Step 5: 提交** — `feat(console): 标注页「整帧」：过目、编辑框、撤销；外形页「来自整理」筛选`

---

### Task 9: 停止时询问、侧栏提示

**Files:**
- Modify: `src/skydango/console/static/live.js`（停止按钮）、`common.js`（侧栏运行卡片下的整理 / 任务提示，侧栏的停止按钮）、`console.html`、`console.css`
- Test: `tests/test_console_page.py`

**Interfaces:**
- Consumes: Task 6 的 `/api/inbox/stop-info`、`/api/state` 的 `job` / `inbox`、`/api/jobs/start|stop`、`stop_run` 的 `curate`、叫醒的 `stop_job`
- Produces（`common.js` 导出里加纯函数 `inboxLine(state) -> {text, action: "open-labeling"|"start-inbox"|null} | null`、`stopQuestion(info) -> string | null`）：
  - `stopQuestion({hard: 85, pending: 2, eta_min: 4, ask: true})` = `"这次存了 85 张难例（还有 2 次运行没整理），现在整理吗？约 4 分钟"`；`hard = 0 && pending = 0` 或 `ask = false` → null（不问，直接停）
  - `inboxLine`：任务 running 且 job = inbox → `"整理素材：" + progress`；job = retrain → `"重训中：" + progress`；任务 failed → `"整理失败：" + tail 最后一行`；否则 `judge_left + glance + edit > 0` → `"N 张裁图、M 帧等你看 →"`（action open-labeling）；否则 `pending > 0` → `"K 次运行的素材没整理"`（action start-inbox）；都没有 → null
  - 停止：`stopQuestion` 非 null 时 `ask()` 三个按钮「整理」「下次再说」「取消」→ `post("api/run/stop", {curate: true|false})` / 不停；叫醒被 409（任务在跑）时 `ask("整理还没完（剩 …），先停下再叫醒？")`（job = retrain 加「训练停了要从头来」）→ 带 `stop_job: true` 重发
- [ ] **Step 1: 写失败的测试** — node 跑 `stopQuestion` 三种输入、`inboxLine` 五种状态的文字
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；浏览器面板里看侧栏三种提示（假造 `datasets/inbox` 的状态即可，不起团子）、停止对话框（起一个 `--dry-run --duration 60` 的团子要用模拟器——晚上真机再看，这里只看 `ask` 弹出：可以对假的 stop-info 打桩）
- [ ] **Step 5: 提交** — `feat(console): 停团子时问要不要整理素材，侧栏显示整理 / 重训进度和待看数量`

---

### Task 10: 重训和对比（`perception retrain`）

**Files:**
- Create: `src/skydango/vision/retrain.py`
- Modify: `src/skydango/cli.py`（`perception retrain` 子命令）、`src/skydango/vision/inbox.py`（`mark_trained(inbox, frames: int)` 写 `_stats.json` 的 `trained_at` / `trained_frames`）
- Test: `tests/test_retrain.py`（新）

**Interfaces:**
- Consumes: `attrs_train`（训练外形头的现有函数、`replay`、`replay_md`、`gt_fixes`）、`detect.make_detector`、`attrs.load_model`
- Produces:
  - `next_version(models: Path, tmp_yolo: Path) -> int`：`models/sky-yolo-v<N>.pt` 和 `tmp_yolo/sky-v<N>/` 里最大的 N + 1（现在 = 12）
  - `stash_caches(dataset: Path, to: Path) -> list[Path]`：把 `dataset/labels/*.cache`（含子目录）挪到 `to/`
  - `train_yolo(cfg: RetrainConfig, dataset: Path, out: Path, train=None) -> Path`：`train` 默认 = 调 ultralytics（`YOLO(cfg.base).train(data=dataset/"data.yaml", imgsz, epochs, batch, workers, project=out.parent, name=out.name, exist_ok=True, device=0, plots=True)`），返回 `out/weights/best.pt`；测试注入假的
  - `compare(old_yolo, new_yolo, old_attrs, new_attrs, dataset, cfg, inbox_frames: set[str]) -> dict`：四组 `replay`（旧 / 旧 + 旧头 / 新 / 新 + 新头）× 全部验证帧和"来自收件箱的验证帧"两份；ultralytics val 的各类 mAP50（`val=None` 时跳过，测试用）
  - `report_md(result: dict) -> str`：表格 + 固定两句提醒（"新旧用的是同一份答案"；某项差 ≤ 3 个点的格子后面标「（可能是波动）」）
  - `run_retrain(cfg: Config, dataset: Path, inbox: Path, out: Path, progress, train=None, val=None, attrs_train_fn=None) -> dict`：stash → 训 YOLO（复制成 `models/sky-yolo-v<N>.pt`）→ 训外形头（`models/attrs-<YYYYMMDD><字母>.npz`，字母从 a 起取第一个不存在的）→ `compare` → 写 `out/report.md`、`out/result.json`（`{"yolo": 新路径, "attrs": 新路径, "old_yolo", "old_attrs", "numbers": …}`）→ `mark_trained`；任何一步失败抛出、不写 `result.json` 的 `ok`、不改配置；每步 `progress("PROGRESS 训练 YOLO")` 之类
  - CLI：`perception retrain [--epochs N]` → `tmp/retrain/<时间>/`
- [ ] **Step 1: 写失败的测试** — `next_version`（`models/` 有 v7、v10，`tmp/yolo/sky-v11/` → 12）；`stash_caches`；`run_retrain` 用假 `train`（写一个空 `best.pt`）、假 `val`、假外形头训练函数、真的小数据集（3 帧验证集 + 假检测器打桩 `make_detector`）→ `models/` 里多了 v12 和 attrs npz、`report.md` 有四行和两句提醒、`_stats.json` 有 `trained_at`；假 `train` 抛异常 → 抛出、`result.json` 不存在
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**（把 `tmp/yolo/train_v10.py` 的参数搬进 `RetrainConfig` 默认值）
- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `feat(retrain): 一键重训 YOLO + 外形头，回放对比出报告（perception retrain）`

---

### Task 11: 面板上的重训、报告、换上 / 回退

**Files:**
- Modify: `src/skydango/console/server.py`、`src/skydango/console/frames.py`（或新 `retrain_view.py`，放得下就放 frames.py）、`static/frames.js`（整帧页顶上的计数、「重训」按钮、「最近一次报告」）、`console.html`
- Test: `tests/test_console_frames.py`

**Interfaces:**
- Consumes: Task 6 任务槽（job = retrain，命令 `perception retrain`）、Task 10 的 `result.json` / `report.md`、`console/settings` 写 `console.toml` 的现有函数、`markdown.js`
- Produces:
  - `GET /api/retrain/latest`：最近一次 `tmp/retrain/*/` 的 `{"dir", "report": md 文字, "result": result.json | null, "adopted": adopt.json | null}`
  - `POST /api/retrain/adopt {"what": "yolo"|"attrs"}`：把 `console.toml` 的 `perception.model` / `attrs.model` 改成 `result.json` 里的新路径；改之前的值（当前生效的，取自 `store` 合并后的配置）记进 `<dir>/adopt.json`（`{"yolo": {"old", "new", "t"}}`）；`POST /api/retrain/rollback {"what"}`：写回 `old`、删掉 `adopt.json` 里那一项；团子在跑时返回里带 `"note": "下次叫醒生效"`
  - 整帧页顶上：「上次训练以来通过：N 帧」、N ≥ `retrain_min` 时「重训」按钮高亮，不到也能点（先 `ask` 提醒数量少）；点了 `ask` 说明要占显卡很久、期间不能叫醒 → `/api/jobs/start {"job": "retrain"}`；「最近一次报告」展开用 `renderMarkdown` 渲染，下面「换上新 YOLO」「换上新外形头」和各自的「回退」
- [ ] **Step 1: 写失败的测试** — `latest` 读最新目录；`adopt` 写 `console.toml`、`adopt.json` 记旧值；`rollback` 写回；`result.json` 缺失时 `adopt` 409；`config.toml` 不被改（比对 sha256）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；浏览器面板里造一个假的 `tmp/retrain/<时间>/`（report.md + result.json）看报告、点换上 / 回退，看 `console.toml`
- [ ] **Step 5: 提交** — `feat(console): 整帧页一键重训、看对比报告、换上 / 回退模型`

---

### Task 12: 文档、全量测试、合并

**Files:**
- Modify: `CLAUDE.md`（「YOLO 感知层」下加「难例收件箱」一段：命令、目录、面板流程、spec / 计划路径、"还没在真机上跑过"；「运行目录」表 `hard/` 那行改成"下线时收进 datasets/inbox"；「常用命令」加 `perception inbox collect|process|status`、`perception retrain`）、`docs/progress/2026-09-28-yolo-training.md`（「第二层」末尾记 10-04 晚真机门槛 4 的结论和这条流水线）、`config.example.toml`（Task 1 已加，核对）
- [ ] **Step 1:** 全量测试放后台跑：`.venv\Scripts\python.exe -m pytest -q`，预期全过（`test_viewer.py::test_port_in_use_raises` 是已知偶发，失败就单独重跑它）
- [ ] **Step 2:** 用 requesting-code-review 自查整条分支
- [ ] **Step 3:** 提交文档 — `docs: 难例收件箱（CLAUDE.md、进度文档）`
- [ ] **Step 4:** 合并进 main 并推送（finishing-a-development-branch；本仓库规矩是合并进 main + push）
- [ ] **Step 5:** 真机验证（晚上，spec §11 五步）留给用户一起做；CLAUDE.md 里写明"还没在真机上跑过"
