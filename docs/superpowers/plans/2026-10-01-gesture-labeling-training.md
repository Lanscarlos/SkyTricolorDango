# 动作识别：Claude 初分 + 「标注」页 + 训练导出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让"挥手 / 鞠躬 / 欢呼 / 害羞"的动作识别从录像走到可用的 ONNX 模型：切片段（改）→ Claude 初分（新命令）→ 管理面板「标注」页核对 → DINOv2-small 冻住 + 时序头训练、导出、评估。

**Architecture:** 纯逻辑放 `vision/gesture.py`（切片段）、`vision/gesture_label.py`（拼图 / 提示词 / 解析，复用 `assist.Reviewer`）、`vision/gesture_train.py`（切分、特征、时序头、导出）、`console/labeling.py`（片段状态 / 挪文件夹 / 撤销）；
`cli.py` 和 `console/server.py` 只做编排和路由；页面是 `console/static/labeling.js` 一页。

**Tech Stack:** Python 3.13、numpy、OpenCV、PyTorch + transformers（`.pydeps`，DINOv2-small 已下载到 `.cache/huggingface`）、onnx / onnxruntime、Claude Code CLI（`assist.Reviewer`）、标准库 HTTP + 原生 JS、pytest。

**Spec:** `docs/superpowers/specs/2026-10-01-gesture-labeling-training-design.md`

## Global Constraints

- 类别（`[gesture] labels`，顺序 = 模型输出顺序）：`["none", "wave", "bow", "cheer", "shy"]`；`names`：`{"wave": "挥手", "bow": "鞠躬", "cheer": "欢呼", "shy": "害羞"}`
- 新配置：`[gesture] dataset = "datasets/gesture"`、`stride = 8`；`enabled` 默认 false、`frames` 16、`fps` 8、`size` 112、`min_prob` 0.9 不变
- 片段名：`<录像名>__<序号:04d>_track<轨迹>_t<开始秒:.2f>s`（`__` 分隔录像名）；数据目录下 `_` 开头的子目录不是类别（`_unlabeled`、`_discard`、`_features`、`_assist`）
- Claude 回答的 `label` 只能是 `wave|bow|cheer|shy|none|unsure`；结果文件 `claude.json` 放在片段文件夹里；**已确认（不在 `_unlabeled/`）的片段永远不动**
- 标注记录 `datasets/gesture/_labels.jsonl`，一行 `{"t", "clip", "from", "to"}`（`from` / `to` 是目录名：`_unlabeled` / 类别 / `_discard`）
- 训练：每帧 112×112 RGB ImageNet 归一化 → DINOv2-small → CLS + 各块平均 = 768 维；时序头 Conv1d(768→128,k3)→ReLU→Conv1d(128→128,k3)→ReLU→时间 max+mean 拼接→Linear(256→5)；按类别数量加权交叉熵、AdamW、按验证集宏平均 F1 早停
- 切分：按录像名，每类约 20% 录像进验证集、每类至少一段；某类只有一段录像 → 按开始秒数前 80% 训练、后 20% 验证，中间空 2 秒；写 `_split.json`
- 某类确认过的片段 < 20 拒绝训练；导出 ONNX 输入 `1×16×3×112×112`（RGB 0~1）、输出 `1×5`，**不覆盖** `models/gesture.onnx`（默认 `models/gesture-<YYYYMMDD>.onnx`）；报告 `tmp/gesture-train/<时间>/report.md`
- 面板接口在 `/api/gesture/` 下，沿用 `is_local_host` / `post_guard`；Claude 的理由在页面上只当纯文本
- 新代码的注释、日志、界面文字用中文，风格照周围代码；测试 `python -m pytest -q -p no:cacheprovider`（本机没有 `.venv`）；改完合并进 main 并推送

## Review Focus

1. **片段名里有中文 / 空格的录像名**（用户会起 `gesture-挥手-1` 这种名字）：`/api/gesture/frame` 收到百分号编码的中文片段名照样取到图 —— Task 3 `test_frame_with_chinese_clip_name`
2. **连按两次、两个标签页同时标**：第二次 `label` 时片段已经不在原处 → 返回 409 和一句话，不抛异常、不重复记 `_labels.jsonl` —— Task 3 `test_label_twice_is_conflict`
3. **用户手动在资源管理器里挪过文件夹后按撤销**：`undo` 找不到要挪回的片段 → 409 说清楚，`_labels.jsonl` 不变 —— Task 3 `test_undo_after_manual_move_is_refused`
4. **片段文件夹里帧数不够 16 张**（切的时候中断、用户删过图）：`gesture-label` 跳过并列出来，训练时也跳过 —— Task 2 `test_incomplete_clip_skipped`、Task 5 `test_incomplete_clip_not_in_dataset`
5. **导出的类别顺序和 `[gesture] labels` 对不上**：用户改过 `labels` 顺序后重训，ONNX 输出顺序跟着配置走、报告里写出顺序 —— Task 6 `test_export_follows_config_label_order`

---

### Task 1: 配置 + 切片段的三处改动

**Files:**
- Modify: `src/skydango/config.py`（`GestureConfig`）、`config.example.toml`（`[gesture]`）、`src/skydango/vision/gesture.py`（`eligible`、`extract_clips`）、`src/skydango/cli.py`（`_perception_clips`、`clips` 子命令参数）
- Test: `tests/test_gesture.py`

**Interfaces:**
- Produces:
  - `GestureConfig.labels/names` 新默认值、`dataset: str = "datasets/gesture"`、`stride: int = 8`
  - `near_center(box: Rect, width: int, ref_h: float, near: float, far: float) -> bool`（不远、框中心在画面中间一半）；`eligible(name, ...)` 改成 `bool(name) and near_center(...)`，行为不变
  - `clip_name(recording: str, n: int, track: int, start: float) -> str` → `f"{recording}__{n:04d}_track{track}_t{start:.2f}s"`；`recording_of(clip: str) -> str`（`__` 前面；没有 `__` 返回 `""`）
  - `extract_clips(frames, detector, out, cfg, conf, recording: str, ref_h: float | None = None, near: float = 0.8, far: float = 0.4) -> int`：只收 `near_center` 的 player；每条轨迹攒满 `frames` 张存一段，之后**留下后 `frames - stride` 张**接着攒（半重叠）；`ref_h` 为 None 时用这一帧的 `self` 框高，没有就 `0.2 × 帧高`
- CLI：`perception clips <录像目录> [-o 输出]`，`-o` 默认 `<[gesture] dataset>/_unlabeled`；录像名 = 录像目录名；`near` / `far` 取 `[perception]`

- [ ] **Step 1: 写失败的测试**（假检测器：每帧给定 player 框）

```python
def test_clip_name_and_recording():
    assert clip_name("gesture-wave-1", 12, 3, 24.0) == "gesture-wave-1__0012_track3_t24.00s"
    assert recording_of("gesture-wave-1__0012_track3_t24.00s") == "gesture-wave-1" and recording_of("0012_track3") == ""

def test_extract_clips_half_overlap(tmp_path):
    # 一个人一直在画面中间、近处，40 帧（8 fps）→ 16 帧一段、每 8 帧起一段：段起点 0、8、16 → 3 段
    n = extract_clips(frames_of(40, box=CENTER_NEAR), FakeDet(), tmp_path, GestureConfig(), 0.35, "gesture-wave-1", ref_h=300)
    assert n == 3
    names = sorted(p.name for p in tmp_path.iterdir())
    assert all(x.startswith("gesture-wave-1__") for x in names)
    assert all(len(list((tmp_path / x).glob("*.jpg"))) == 16 for x in names)

def test_extract_clips_skips_far_and_edge_people(tmp_path):
    n = extract_clips(frames_of(40, box=FAR_SMALL) + ..., FakeDet(), tmp_path, GestureConfig(), 0.35, "r", ref_h=300)
    assert n == 0  # 太远的、在画面最左边四分之一的都不切

def test_config_gesture_defaults():
    g = Config().gesture
    assert g.labels == ["none", "wave", "bow", "cheer", "shy"] and g.names["cheer"] == "欢呼" and g.names["shy"] == "害羞"
    assert (g.dataset, g.stride, g.enabled) == ("datasets/gesture", 8, False)
```

- [ ] **Step 2: 跑** `python -m pytest -q -p no:cacheprovider tests/test_gesture.py` → 新测试失败
- [ ] **Step 3: 实现**；`config.example.toml` 的 `[gesture]` 加 `dataset` / `stride` 并改 `labels` / `names`
- [ ] **Step 4: 跑** `tests/test_gesture.py tests/test_config.py tests/test_cli_phase3.py` → 通过（旧的 `eligible` 测试照过）
- [ ] **Step 5: 提交** `feat(gesture): 片段名带录像名、半重叠切、只切近处画面中间的人；第一版认挥手 / 鞠躬 / 欢呼 / 害羞`

---

### Task 2: Claude 初分（`vision/gesture_label.py` + `perception gesture-label`）

**Files:**
- Create: `src/skydango/vision/gesture_label.py`
- Modify: `src/skydango/cli.py`（`_perception_gesture_label` + 子命令）
- Test: `tests/test_gesture_label.py`

**Interfaces:**
- Consumes: Task 1 `recording_of`；`vision/assist.py` 的 `FrameInput`、`Protocol`、`Reviewer`、`AssistLimit`、`assist_command`、`assist_workdir`、`extract_json`；`brain.images.image_block`；`brain.claude.one_shot_message`
- Produces:
  - `GUESSES = ("wave", "bow", "cheer", "shy", "none", "unsure")`；`GESTURE_PROMPT_VERSION = 1`
  - `@dataclass(frozen=True) Guess`：`label: str`、`confidence: float`（夹到 0~1）、`reason: str`（截 60 字）
  - `contact_sheet(frames: list[np.ndarray]) -> np.ndarray`：4×4、每格 112×112（不是这个尺寸先缩放），左上角白底黑字帧号 0~15 → 448×448×3
  - `recording_hint(recording: str) -> str`：录像名（小写）里含 `wave`/`bow`/`cheer`/`shy` 之一 → "这段录像是好友在反复做<中文名>（中间有停顿）"；含 `none` → "这段录像里没有这四个动作"；都不含 → ""
  - `GESTURE_SYSTEM: str`、`build_gesture_message(frames: list[FrameInput], cfg) -> list[dict]`（每个片段：一行文字 "片段 <名字>：<提示>" + 拼图）、`parse_gesture_review(text, frames) -> dict[str, Guess]`（只收 `GUESSES` 里的 label；片段名对不上、label 不认识的那段不在结果里）
  - `GESTURE_PROTOCOL = Protocol(GESTURE_PROMPT_VERSION, GESTURE_SYSTEM, build_gesture_message, parse_gesture_review)`
  - `load_guess(clip_dir: Path) -> Guess | None`、`write_guess(clip_dir: Path, g: Guess, model: str) -> None`（`claude.json`：label / confidence / reason / model / prompt_version / t）
- 提示词要点（写进 `GESTURE_SYSTEM`）：看 4×4 拼图（16 帧、2 秒、左到右上到下）判断画面中间那个人在做什么；四个动作的样子 —— 挥手：一只手举起左右摆；鞠躬：上身前弯再起来；欢呼：双手往上举；害羞：捂脸 / 扭身；动作没做完、看不清、被挡 → `unsure`；只回 JSON `{"<片段名>": {"label", "confidence", "reason"}}`；**注明"描述待 10-01 晚用户核对动画后修改"的 TODO 不要写进提示词**，写在模块 docstring
- CLI `perception gesture-label [片段目录] [--recheck]`：默认 `<dataset>/_unlabeled`；每个子目录要正好 16 张图，不够的列出来跳过；已有 `claude.json` 且没 `--recheck` 的跳过；`FrameInput(stem=片段名, image=contact_sheet(...), candidates=[])`；
  `Reviewer(run, <dataset>/_assist, dataclasses.replace(cfg.assist, batch=8), source="gesture", protocol=GESTURE_PROTOCOL)`；结果写 `claude.json`；`AssistLimit` → 提示额度用完、重跑接着做；最后打印各类数量和用量

- [ ] **Step 1: 写失败的测试**

```python
def test_contact_sheet_size_and_order():
    frames = [np.full((112, 112, 3), i * 10, np.uint8) for i in range(16)]
    s = contact_sheet(frames)
    assert s.shape == (448, 448, 3) and s[60, 60, 0] == 0 and s[60 + 112 * 3, 60 + 112 * 3, 0] == 150

def test_recording_hint():
    assert "挥手" in recording_hint("gesture-wave-1") and "没有" in recording_hint("gesture-none-2") and recording_hint("rec") == ""

def test_parse_accepts_known_labels_and_unsure():
    frames = [FrameInput("a__0001_track1_t0.00s", SHEET, []), FrameInput("a__0002_track1_t1.00s", SHEET, [])]
    got = parse_gesture_review('{"a__0001_track1_t0.00s": {"label": "wave", "confidence": 1.4, "reason": "右手举起来摆"},'
                               ' "a__0002_track1_t1.00s": {"label": "dance", "confidence": 0.5}}', frames)
    assert got == {"a__0001_track1_t0.00s": Guess("wave", 1.0, "右手举起来摆")}

def test_message_has_hint_and_one_image_per_clip():
    msg = build_gesture_message([FrameInput("gesture-bow-1__0001_track1_t0.00s", SHEET, [])], AssistConfig())
    assert sum(b["type"] == "image" for b in msg) == 1 and any("鞠躬" in b.get("text", "") for b in msg)

def test_cli_skips_existing_guess_and_incomplete_clip(tmp_path, monkeypatch): ...  # 假 one_shot_message；已有 claude.json 的不送；--recheck 才送
def test_incomplete_clip_skipped(tmp_path, monkeypatch): ...  # 15 张图的片段不送、打印里列出
```

- [ ] **Step 2: 跑** `tests/test_gesture_label.py` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `tests/test_gesture_label.py tests/test_assist.py` → 通过
- [ ] **Step 5: 提交** `feat(gesture): perception gesture-label——Claude 看 4×4 拼图初分动作片段`

---

### Task 3: 标注的后端（`console/labeling.py` + 接口）

**Files:**
- Create: `src/skydango/console/labeling.py`
- Modify: `src/skydango/console/server.py`（GET / POST 路由）
- Test: `tests/test_console_labeling.py`

**Interfaces:**
- Consumes: Task 1 `recording_of`、`GestureConfig.labels/dataset`；Task 2 `load_guess`
- Produces（`console/labeling.py`，纯函数 / 小类，`root: Path` = 数据目录）：
  - `class GestureLabels(root: Path, labels: list[str])`
    - `state() -> dict`：`{"ok", "labels", "names", "counts": {目录名: 个数}, "clips": [{"clip", "recording", "where", "guess": {label, confidence, reason} | None}]}`，`where` ∈ `_unlabeled` / 类别 / `_discard`；数据目录不存在 → `{"ok": False, "text": "还没有片段：先跑 perception clips"}`
    - `frame(clip: str, i: int) -> bytes | None`：`clip` 必须**正好等于**数据目录某个合法子目录（`_unlabeled` / 类别 / `_discard`）下的文件夹名（遍历比对，不拼路径；含 `/` `\` `..` 一律 None），`0 <= i < 16`
    - `label(clip: str, to: str) -> tuple[int, dict]`：`to` ∈ labels ∪ {"discard"}（discard → `_discard`）；找到片段当前目录、`shutil.move` 到目标目录；目标已有同名 → 409；片段不在任何地方 → 409；成功追加 `_labels.jsonl` 并返回 200 和新状态的那一条
    - `undo() -> tuple[int, dict]`：取 `_labels.jsonl` 最后一条**没被撤销过**的，把片段从 `to` 挪回 `from`，追加一条 `{"t", "clip", "from": to, "to": from, "undo": true}`；找不到 → 409、不写记录
  - 路由：`GET /api/gesture/state`、`GET /api/gesture/frame?clip=&i=`（返回 `image/jpeg`，找不到 404）、`POST /api/gesture/label`、`POST /api/gesture/undo`；`GestureLabels` 按 `store._fallback().gesture` 现建（数据目录相对面板当前目录）

- [ ] **Step 1: 写失败的测试**（`tmp_path` 下造 `_unlabeled/<片段>/00.jpg…15.jpg` + `claude.json`）

```python
def test_state_lists_clips_with_guess(tmp_path): ...  # where="_unlabeled"、guess 来自 claude.json、counts
def test_label_moves_and_logs(tmp_path):
    g = GestureLabels(tmp_path, LABELS); code, item = g.label(CLIP, "wave")
    assert code == 200 and (tmp_path / "wave" / CLIP).is_dir() and not (tmp_path / "_unlabeled" / CLIP).exists()
    assert json.loads((tmp_path / "_labels.jsonl").read_text("utf-8").splitlines()[-1])["to"] == "wave"
def test_relabel_and_discard(tmp_path): ...  # wave → bow → discard，每次都在新目录
def test_undo_restores(tmp_path): ...
def test_label_twice_is_conflict(tmp_path): ...  # 第二次 label 同一个片段到同一类 → 409，_labels.jsonl 只有一行
def test_undo_after_manual_move_is_refused(tmp_path): ...  # 手动把片段挪走后 undo → 409，记录不变
def test_frame_rejects_traversal(tmp_path):
    g = GestureLabels(tmp_path, LABELS)
    assert g.frame("../secret", 0) is None and g.frame(CLIP, 16) is None and g.frame(CLIP, -1) is None and g.frame(CLIP, 0)[:2] == b"\xff\xd8"
def test_frame_with_chinese_clip_name(srv_with_gesture): ...  # 经 HTTP：片段名 "gesture-挥手-1__0001_track1_t0.00s" 百分号编码后照样 200
def test_api_requires_local_host_and_post_guard(srv_with_gesture): ...  # 同现有 /api/* 的校验
```

- [ ] **Step 2: 跑** `tests/test_console_labeling.py` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `tests/test_console_labeling.py tests/test_console_server.py` → 通过
- [ ] **Step 5: 提交** `feat(console): 标注接口——片段状态、取帧、确认 / 改类别 / 丢弃、撤销`

---

### Task 4: 「标注」页（前端）

**Files:**
- Create: `src/skydango/console/static/labeling.js`
- Modify: `src/skydango/console/static/console.html`（导航新组「数据」→「标注」、`<section id="page-labeling" class="page" hidden>`、`<script>`）、`common.js`（`PAGES`、`TITLES` 加 `labeling: "标注"`）、`console.css`
- Test: `tests/test_console_server.py`（页面里有 `page-labeling`、`labeling.js` 能取到）

**Interfaces:**
- Consumes: Task 3 的四个接口；`common.js` 的 `getJSON`、`post`、`el`、`$`、`Pages`
- Produces: `Pages.labeling = {show, hide}`；三栏：左列表（筛选：待确认〔默认，`unsure` 最前，其余按 confidence 升序〕/ Claude 猜某类 / 已确认某类 / 丢弃；顶上各类计数）、中播放（16 帧 3 倍放大、按 `fps` 8 循环；暂停、逐帧 `←` `→`（仅暂停时）、0.5 倍慢放；预加载后 3 段）、右（猜测、把握、理由用 `textContent`、录像名、五个类别 + 不要 + 撤销按钮）
- 键盘（焦点不在输入框时）：`Enter` 同意 Claude（`unsure` / 没猜时无效）、`1`~`5` = `wave, bow, cheer, shy, none`、`0` 不要、`z` 撤销、空格 暂停；选完跳下一段；`hide` 时停止播放、解绑键盘

- [ ] **Step 1: 写失败的测试**：`GET /` 页面里有 `id="page-labeling"`、`data-page="labeling"`、`console/static/labeling.js`；`GET /console/static/labeling.js` 200
- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `tests/test_console_server.py` → 通过；再用浏览器预览核对（`python -m skydango console --no-browser`，`datasets/gesture` 里放几段假片段）：列表、播放、按键确认、撤销都能用，截图存 `tmp/`
- [ ] **Step 5: 提交** `feat(console): 「标注」页——看动图、按键确认动作片段`

---

### Task 5: 训练数据：读片段、切分（`vision/gesture_train.py` 第一部分）

**Files:**
- Create: `src/skydango/vision/gesture_train.py`
- Test: `tests/test_gesture_train.py`

**Interfaces:**
- Consumes: Task 1 `recording_of`；`gesture.load_clip`
- Produces:
  - `@dataclass(frozen=True) Sample`：`clip: str`、`label: str`、`recording: str`、`start: float`（从片段名 `_t<秒>s` 解析）、`path: Path`
  - `list_samples(root: Path, labels: list[str], frames: int = 16) -> tuple[list[Sample], list[str]]`：只看 `labels` 里的类别目录；帧数不等于 `frames` 的进第二个返回值（跳过的片段名）
  - `MIN_PER_CLASS = 20`；`check_counts(samples, labels) -> list[str]`：不够的类别 → 提示（"欢呼只有 12 段，还差 8 段"）
  - `split(samples, labels, val_ratio=0.2, gap=2.0, seed=0) -> dict[str, list[str]]`：`{"train": [...], "val": [...], "warnings": [...]}`，规则见 Global Constraints；只有一段录像的类别按 `start` 切（前 80% 训练、后 20% 验证、中间 `gap` 秒内的不用）；某类进不了验证集 → warnings
  - `save_split(root, split) / load_split(root) -> dict | None`（`_split.json`）

- [ ] **Step 1: 写失败的测试**

```python
def test_list_samples_and_incomplete(tmp_path): ...
def test_incomplete_clip_not_in_dataset(tmp_path): ...  # 15 张图的片段在跳过列表里，不在 samples 里
def test_check_counts_message():
    assert check_counts(samples_with(cheer=12, others=30), LABELS) == ["欢呼只有 12 段，还差 8 段"]
def test_split_by_recording_keeps_recordings_whole(): ...  # 同一段录像的片段不会同时在 train / val
def test_split_every_class_has_val(): ...
def test_split_single_recording_by_time_with_gap():
    s = split(one_recording_samples(starts=range(0, 100)), ["none", "wave"], gap=2.0)
    # 前 80% 训练、后 20% 验证，切点两边 2 秒内的不在任何一边
```

- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `tests/test_gesture_train.py` → 通过
- [ ] **Step 5: 提交** `feat(gesture): 训练数据——读确认过的片段、按录像切分训练 / 验证集`

---

### Task 6: 训练、导出、报告（`gesture_train.py` 第二部分 + `perception gesture-train` + `gesture-eval` 只评验证集 + 文档）

**Files:**
- Modify: `src/skydango/vision/gesture_train.py`、`src/skydango/vision/gesture.py`（`evaluate(..., only: set[str] | None = None)`）、`src/skydango/cli.py`（`gesture-train` 子命令、`gesture-eval --all`）、`CLAUDE.md`
- Test: `tests/test_gesture_train.py`

**Interfaces:**
- Consumes: Task 5 全部；`gesture.OnnxGestureClassifier`、`gesture.evaluate`
- Produces:
  - `class Extractor(Protocol)`：`key: str`；`features(frames: torch.Tensor) -> torch.Tensor`（`T×3×S×S` RGB 0~1 → `T×D`）
  - `DinoExtractor(device: str)`：`facebook/dinov2-small`（`HF_HOME` 指到仓库 `.cache/huggingface`），内部做 ImageNet 归一化，输出 CLS + 各块平均 = 768；`key = "dinov2-small-112-v1"`
  - `TemporalHead(dim: int, classes: int)`：结构见 Global Constraints
  - `features_for(sample, extractor, cache_dir, flip: bool) -> np.ndarray`（缓存 `_features/<key>/<片段>[_flip].npy`）
  - `train(samples, split, labels, extractor, cache_dir, epochs=60, seed=0, device="cuda") -> tuple[TemporalHead, dict]`（dict：每轮训练损失、验证宏 F1、最好的那轮）
  - `class Exported(nn.Module)`：输入 `1×T×3×S×S` → `extractor` → `TemporalHead` → `1×C`
  - `export_onnx(extractor, head, path, frames=16, size=112) -> None`（opset 17）
  - `report_md(...) -> str`：每类数量、切分、最好的那轮、验证集每类精确率 / 召回率（`gesture.evaluate` 的结果）、认错的片段、**类别顺序**
- CLI `perception gesture-train [数据目录] [--epochs 60] [--out 路径] [--device cuda|cpu]`：`check_counts` 不过 → `SystemExit` 列出缺的；切分写 `_split.json`；训练 → 导出 → 拿 ONNX 和 PyTorch 在验证集前 3 段比输出（最大差 > 1e-3 记警告）→ `evaluate(only=验证集)` → 写报告；没 CUDA 退回 CPU 并提示
- `perception gesture-eval`：有 `_split.json` 时默认只评 `val`，`--all` 评全部

- [ ] **Step 1: 写失败的测试**（用假 `Extractor`：每帧取均值颜色 → 3 维，`key="fake"`；造两类颜色明显不同的片段各 25 段、几段录像）

```python
def test_train_learns_separable_fake_data(tmp_path):
    head, info = train(samples, split, ["none", "wave"], FakeExtractor(), tmp_path / "_features", epochs=30, device="cpu")
    assert info["best_f1"] > 0.9
def test_features_cached(tmp_path): ...  # 第二次 features_for 不调 extractor
def test_export_onnx_matches_torch(tmp_path):
    export_onnx(FakeExtractor(), head, tmp_path / "g.onnx", frames=16, size=112)
    # onnxruntime 和 PyTorch 同一段输入的输出最大差 < 1e-4；输出形状 (1, 2)
def test_export_follows_config_label_order(tmp_path): ...  # labels 顺序换了重训：OnnxGestureClassifier(labels=新顺序) 认对；报告里写出顺序
def test_evaluate_only_val(tmp_path): ...  # evaluate(only={...}) 只数 val 里的片段
def test_cli_refuses_when_class_too_small(tmp_path, monkeypatch): ...
```

- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现**；`CLAUDE.md`：代码结构表加 `gesture_label.py` / `gesture_train.py` / `console/labeling.py`，常用命令加 `perception gesture-label` / `gesture-train`，「YOLO 感知层」三期动作识别那条改成现状（四个动作、Claude 初分、「标注」页、训练命令、**还没有真数据和模型**），「管理面板」一节加「标注」页一句
- [ ] **Step 4: 跑** `tests/test_gesture_train.py tests/test_gesture.py` → 通过；全量 `python -m pytest -q -p no:cacheprovider` 一次
- [ ] **Step 5: 提交** `feat(gesture): perception gesture-train——DINOv2-small 冻住 + 时序头，导出 ONNX、验证集评估`
