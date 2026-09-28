# Claude 辅助标注 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `perception label <录像> --assist`：挑帧 → 检测器出人物候选框 → 隔离的 `claude -p --model sonnet` 核对 → 和弱标注合并写成 YOLO 数据集 + 预览 + 待核对清单。

**Architecture:** 逻辑都在新模块 `src/skydango/vision/assist.py`（纯函数 + 一个 `Reviewer` 类，真正调 Claude 的函数从外面注入）；`brain/claude.py` 加一个返回完整结果消息的 `one_shot_message`；`cli.py` 只加选项和一个编排函数。

**Tech Stack:** Python 3.13、numpy、opencv、ultralytics（延迟导入）、Claude Code CLI（stream-json）、pytest。

**Spec:** `docs/superpowers/specs/2026-09-28-assist-labeling-design.md`

## Global Constraints

- 回答 / 注释 / 用户可见文字用中文；注释密度和现有代码一致
- 不加 `--assist` 时 `perception label` 行为完全不变（现有测试全绿）
- 测试不连 Claude、不要模拟器、不要 GPU、不要 ultralytics（用到的测试 `pytest.importorskip("ultralytics")`）
- 默认值（`[assist]`）：`min_change = 30.0`、`max_gap = 20`、`proposal_models = ["models/yolo11n.pt", "models/yolo11x.pt"]`、`proposal_conf = 0.1`、`proposal_imgsz = 1280`、`batch = 5`、`jobs = 3`、`timeout = 300.0`、`model = "sonnet"`、`self_hint = "白色头发、橙色护目镜、橙粉色袍子、背蓝紫色圆背包、头顶没有名字标签"`
- 候选合并的 NMS IoU 固定 0.6；Claude 回答的类别集合：`self` `player` `player_unlit` `not_person` `duplicate`；`missing` 只能是前三个
- 令牌 / 配置目录沿用 `[brain]`（`claude_path`、`token_env`、`config_dir`），通过现有 `cli._brain_env(cfg)` 取
- 临时 / 调试输出放项目 `tmp/`；数据集输出在 `-o`（默认 `datasets/sky`）下的 `_assist/`、`_preview/`

## Review Focus

1. Claude 给的框坐标反了（x1 > x2）或超出画面（按缩小后的图报坐标）→ 规整、裁到画面内，裁完为空就丢，不崩 —— Task 3 `test_parse_normalizes_and_clips_boxes`
2. 回答夹着说明文字 / ```json 代码块 / 被截断 → 能取出 JSON 的照常用，取不出的整批按"没核对"，不崩 —— Task 3 `test_parse_tolerates_fences_and_prose`、`test_parse_garbage_returns_empty`
3. 某帧候选框为 0（检测器全漏，比如人挤在一起）→ 照样发给 Claude 让它补 `missing`，不要跳过 —— Task 5 `test_reviewer_sends_frames_without_candidates`
4. 额度用完中途停下后重跑同一条命令 → 已核对的帧命中缓存不再花额度；换了 `--model`（候选变了）或提示词版本变了 → 重新核对 —— Task 5 `test_reviewer_cache_hit_and_invalidation`
5. `--from-runs` 的难例帧名带运行目录名（含 `-`、`.`）→ 缓存文件名直接用帧名（已是 `[\w.-]+`），不挑帧（不是连续录像）—— Task 7 `test_assist_skips_picking_for_from_runs`

---

### Task 1: `[assist]` 配置 + 挑帧

**Files:**
- Modify: `src/skydango/config.py`（新增 `AssistConfig`，`Config.assist`）
- Modify: `config.example.toml`（加 `[assist]` 一节，每项一行注释）
- Create: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Produces: `AssistConfig`（字段见 Global Constraints）；`pick_frames(thumbs: Sequence[np.ndarray], min_change: float, max_gap: int) -> list[int]`（`thumbs` 是 `brain.images.thumb()` 的 64×36 灰度图；第 0 帧总留；和**上一张留下的**比平均绝对差 `> min_change` 或距上一张留下的 `>= max_gap` 帧就留）

- [ ] **Step 1: 写失败测试**

```python
def test_pick_frames_drops_similar_and_forces_gap():
    base = np.full((36, 64), 100, np.uint8)
    thumbs = [base] * 5 + [base + 50] + [base + 50] * 30
    assert pick_frames(thumbs, min_change=30, max_gap=20) == [0, 5, 25]

def test_assist_config_defaults():
    c = Config().assist
    assert (c.min_change, c.max_gap, c.batch, c.jobs, c.model) == (30.0, 20, 5, 3, "sonnet")
    assert c.proposal_models == ["models/yolo11n.pt", "models/yolo11x.pt"]
```

- [ ] **Step 2:** `python -m pytest -q tests/test_assist.py` → FAIL（ImportError）
- [ ] **Step 3:** 实现 `AssistConfig`、`pick_frames`；`config.example.toml` 加 `[assist]`（全部注释掉的默认值 + 说明）
- [ ] **Step 4:** `python -m pytest -q tests/test_assist.py tests/test_config.py` → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): [assist] 配置和挑帧"`

### Task 2: 候选框图 + 提示词

**Files:**
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Produces:
  - `PROMPT_VERSION: int = 1`
  - `draw_candidates(frame: np.ndarray, boxes: list[Rect]) -> np.ndarray`：拷贝一份，画白色细网格（每 100 px，边上写坐标数字）、编号框（2 px，编号从 1 开始、写在框**上方外侧**，靠顶时写在框下方外侧），不改原图
  - `build_message(frames: list[tuple[str, np.ndarray, list[Rect]]], self_hint: str) -> list[dict]`：内容块列表 = 1 个规则文字块（spec §4 全部规则 + 回答格式 + `self_hint`）+ 每帧一个文字块（`帧 <stem>：候选框 1=[x1,y1,x2,y2]；2=…`，没有候选写"没有候选框，只看有没有漏掉的人"）+ 该帧 `image_block(draw_candidates(...), 85)`
  - `ASSIST_SYSTEM: str`（一句：你是标注核对员，只输出 JSON）

- [ ] **Step 1: 写失败测试**

```python
def test_draw_candidates_keeps_original():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = draw_candidates(frame, [Rect(100, 200, 50, 120)])
    assert frame.max() == 0 and out.shape == frame.shape and out.max() > 0

def test_build_message_lists_frames_and_rules():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    msg = build_message([("a", frame, [Rect(10, 20, 30, 40)]), ("b", frame, [])], "蓝紫色圆背包")
    texts = [b["text"] for b in msg if b["type"] == "text"]
    assert "头顶有圆圈" in texts[0] and "蓝紫色圆背包" in texts[0] and "duplicate" in texts[0]
    assert "帧 a" in texts[1] and "1=[10,20,40,60]" in texts[1]
    assert "帧 b" in texts[2] and "没有候选框" in texts[2]
    assert [b["type"] for b in msg].count("image") == 2
```

- [ ] **Step 2:** 跑测试 → FAIL
- [ ] **Step 3:** 实现（规则文字照 spec §4 和 `tmp/assist2/INSTRUCTIONS.md` 的"类别""还要做的"两节，去掉读文件相关的话）
- [ ] **Step 4:** 跑测试 → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): 候选框图和提示词"`

### Task 3: 解析回答

**Files:**
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Produces:
  - `@dataclass class Verdict: cls: str; fixed: Rect | None; note: str`
  - `@dataclass class FrameReview: verdicts: dict[int, Verdict]; missing: list[tuple[str, Rect, str]]; unsure: str; problems: list[str]`
  - `parse_review(text: str, frames: dict[str, int], width: int, height: int) -> dict[str, FrameReview]`：`frames` 是 帧名 → 候选框个数。取文字里第一个 `{` 到最后一个 `}` 解析；解析失败返回 `{}`。只收 `frames` 里有的帧（多给的忽略、缺的不在结果里 = 没核对）。每个候选编号：缺了或类别不在集合里 → `Verdict("not_person", None, "")` 并在 `problems` 记一句（如 `"3 号没判"`、`"2 号类别 'npc' 不认识"`）。框（`fixed_box` / `missing.box`）：四个数、x/y 各自排序、裁到 `[0,width]×[0,height]`，宽或高 < 2 就丢（fixed 丢了等于 None）。`missing` 的类别不在前三个的丢掉

- [ ] **Step 1: 写失败测试**

```python
def test_parse_reads_verdicts_missing_unsure():
    text = '{"a": {"boxes": {"1": {"cls": "self", "note": "", "fixed_box": [0, 0, 50, 90]}, "2": {"cls": "duplicate", "note": "留 1"}},'
    text += ' "missing": [{"cls": "player_unlit", "box": [300, 300, 340, 400], "note": "黑影"}], "unsure": "看不清"}}'
    r = parse_review(text, {"a": 2}, 1920, 1080)["a"]
    assert r.verdicts[1].cls == "self" and r.verdicts[1].fixed == Rect(0, 0, 50, 90)
    assert r.verdicts[2].cls == "duplicate"
    assert r.missing == [("player_unlit", Rect(300, 300, 40, 100), "黑影")] and r.unsure == "看不清"

def test_parse_marks_missing_ids_and_bad_classes():
    r = parse_review('{"a": {"boxes": {"1": {"cls": "npc"}}, "missing": [], "unsure": ""}}', {"a": 2}, 1920, 1080)["a"]
    assert r.verdicts[1].cls == r.verdicts[2].cls == "not_person" and len(r.problems) == 2

def test_parse_normalizes_and_clips_boxes():
    text = '{"a": {"boxes": {}, "missing": [{"cls": "player", "box": [2000, 900, 1800, 1200]}, {"cls": "player", "box": [5000, 10, 6000, 20]}, {"cls": "pet", "box": [1, 1, 50, 50]}]}}'
    assert parse_review(text, {"a": 0}, 1920, 1080)["a"].missing == [("player", Rect(1800, 900, 120, 180), "")]

def test_parse_tolerates_fences_and_prose():
    text = '好的，结果如下：\n```json\n{"a": {"boxes": {"1": {"cls": "player"}}, "missing": [], "unsure": ""}}\n```\n以上。'
    assert parse_review(text, {"a": 1, "b": 1}, 1920, 1080).keys() == {"a"}

def test_parse_garbage_returns_empty():
    assert parse_review('{"a": {"boxes": ', {"a": 1}, 1920, 1080) == {}
```

- [ ] **Step 2:** 跑测试 → FAIL
- [ ] **Step 3:** 实现
- [ ] **Step 4:** 跑测试 → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): 解析 Claude 的核对结果"`

### Task 4: 合并成标注 + 预览 + 待核对清单

**Files:**
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Consumes: `Verdict`、`FrameReview`（Task 3）
- Produces:
  - `PEOPLE = ("player", "player_unlit", "self")`
  - `apply_review(candidates: list[Rect], review: FrameReview) -> list[tuple[str, Rect]]`：`PEOPLE` 类的候选（有 `fixed` 用 `fixed`）+ 全部 `missing`；`not_person` / `duplicate` 丢掉
  - `draw_review(frame: np.ndarray, weak: list[tuple[str, Rect]], candidates: list[Rect], review: FrameReview | None) -> np.ndarray`：弱标注照 `_perception_label` 预览的颜色；人物框 绿 `player` (0,220,0)、紫 `player_unlit` (255,0,200)、灰白 `self` (200,200,200) 4 px + 类别文字；去掉的候选红细框 1 px；`missing` 虚线；`review is None` 时左上角写"没核对"
  - `review_report(results: list[tuple[str, FrameReview | None]]) -> str`：Markdown，四节按 spec §5 顺序：没核对 / 有补框（附每个补框的类别和 note）/ `unsure` 或 `problems` 非空（附原话）/ 其余帧数量 + "抽查几张"；空的节写"（无）"

- [ ] **Step 1: 写失败测试**

```python
def _review(**kw):
    return FrameReview(verdicts=kw.get("v", {}), missing=kw.get("m", []), unsure=kw.get("u", ""), problems=kw.get("p", []))

def test_apply_review_keeps_people_uses_fixed_adds_missing():
    cands = [Rect(0, 0, 10, 10), Rect(20, 0, 10, 10), Rect(40, 0, 10, 10), Rect(60, 0, 10, 10)]
    r = _review(v={1: Verdict("self", Rect(0, 0, 12, 12), ""), 2: Verdict("duplicate", None, ""),
                   3: Verdict("not_person", None, ""), 4: Verdict("player_unlit", None, "")},
                m=[("player", Rect(90, 0, 10, 20), "")])
    assert apply_review(cands, r) == [("self", Rect(0, 0, 12, 12)), ("player_unlit", Rect(60, 0, 10, 10)), ("player", Rect(90, 0, 10, 20))]

def test_review_report_orders_sections():
    md = review_report([("f1", None), ("f2", _review(m=[("player", Rect(0, 0, 5, 5), "远处")])),
                        ("f3", _review(u="看不清")), ("f4", _review())])
    assert md.index("f1") < md.index("f2") < md.index("f3") and "远处" in md and "看不清" in md
    assert "其余 1 帧" in md

def test_draw_review_marks_unreviewed():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert draw_review(frame, [], [], None).max() > 0 and frame.max() == 0
```

- [ ] **Step 2:** 跑测试 → FAIL
- [ ] **Step 3:** 实现
- [ ] **Step 4:** 跑测试 → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): 合并标注、预览和待核对清单"`

### Task 5: `Reviewer`（分批、并发、缓存、重试、额度）+ `one_shot_message`

**Files:**
- Modify: `src/skydango/brain/claude.py`（加 `one_shot_message`，`one_shot` 改成调它）
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`、`tests/test_brain_claude.py`

**Interfaces:**
- Consumes: `build_message`、`PROMPT_VERSION`、`ASSIST_SYSTEM`（Task 2）；`parse_review`、`FrameReview`（Task 3）；`brain.claude.ClaudeError`
- Produces:
  - `brain.claude.one_shot_message(cmd, env, cwd, content, timeout) -> dict`：同 `one_shot`，但返回 `check_result` 通过后的完整 result 消息；`one_shot` = `check_result(one_shot_message(...))` 的文字（行为不变）
  - `assist_command(base: list[str], cfg: AssistConfig) -> list[str]`：照 `brain.eyes.eyes_command`，`--model cfg.model`、不加 `--effort`、`--tools ""`、`--system-prompt ASSIST_SYSTEM`
  - `@dataclass class FrameInput: stem: str; image: np.ndarray; candidates: list[Rect]`
  - `class AssistLimit(RuntimeError)`：额度用完
  - `class Reviewer.__init__(self, run: Callable[[list[dict]], dict], cache_dir: Path, cfg: AssistConfig, source: str)`：`run` 收内容块、返回 result 消息（`{"result": 文字, "usage": {...}}`），失败抛 `ClaudeError`；`source` 是候选来源（模型路径拼起来），写进缓存
  - `Reviewer.review(frames: list[FrameInput]) -> dict[str, FrameReview | None]`：先查缓存（`cache_dir/<stem>.json` 里 `prompt_version`、`source`、`candidates` 都相同才算命中）；没命中的按 `cfg.batch` 分批、`ThreadPoolExecutor(cfg.jobs)` 并发；每批 `ClaudeError`（非 limit）重试一次，再失败这批记 None；`limit=True` → 等已提交的批次结束、已成功的写好缓存后抛 `AssistLimit`；每批成功后每帧写缓存 `{"prompt_version", "source", "candidates": [[x1,y1,x2,y2]…], "review": <该帧原始 dict>}`；回答里缺的帧记 None、不写缓存
  - `Reviewer.usage: dict[str, int]`（累加 result 消息 `usage` 里的 `input_tokens`、`output_tokens`、`cache_read_input_tokens`）

- [ ] **Step 1: 写失败测试**（假的 `run` 记录每次收到的帧名，按帧名回一个 JSON）

```python
def _fake_run(calls, fail_times=0, limit=False):
    state = {"fails": fail_times}
    def run(content):
        stems = [b["text"].split()[1].rstrip("：") for b in content if b["type"] == "text" and b["text"].startswith("帧 ")]
        calls.append(stems)
        if limit:
            raise ClaudeError("额度用完", limit=True)
        if state["fails"] > 0:
            state["fails"] -= 1
            raise ClaudeError("超时")
        body = {s: {"boxes": {"1": {"cls": "player"}}, "missing": [], "unsure": ""} for s in stems}
        return {"result": json.dumps(body), "usage": {"input_tokens": 10, "output_tokens": 2}}
    return run

def _frames(n, cands=1):
    return [FrameInput(f"f{i}", np.zeros((1080, 1920, 3), np.uint8), [Rect(0, 0, 10, 10)] * cands) for i in range(n)]

def test_reviewer_batches(tmp_path):
    calls = []
    cfg = AssistConfig(batch=2, jobs=1)
    out = Reviewer(_fake_run(calls), tmp_path, cfg, "m").review(_frames(5))
    assert sorted(map(len, calls)) == [1, 2, 2] and all(out[f"f{i}"] is not None for i in range(5))
    assert out["f0"].verdicts[1].cls == "player"

def test_reviewer_cache_hit_and_invalidation(tmp_path):
    calls = []
    cfg = AssistConfig(batch=5, jobs=1)
    Reviewer(_fake_run(calls), tmp_path, cfg, "m").review(_frames(3))
    Reviewer(_fake_run(calls), tmp_path, cfg, "m").review(_frames(3))
    assert len(calls) == 1
    Reviewer(_fake_run(calls), tmp_path, cfg, "other").review(_frames(3))
    assert len(calls) == 2

def test_reviewer_retries_once_then_gives_up(tmp_path):
    calls = []
    out = Reviewer(_fake_run(calls, fail_times=1), tmp_path, AssistConfig(batch=5, jobs=1), "m").review(_frames(2))
    assert len(calls) == 2 and out["f0"] is not None
    calls2 = []
    out2 = Reviewer(_fake_run(calls2, fail_times=9), tmp_path / "b", AssistConfig(batch=5, jobs=1), "m").review(_frames(2))
    assert len(calls2) == 2 and out2 == {"f0": None, "f1": None}

def test_reviewer_limit_raises_and_keeps_done(tmp_path):
    with pytest.raises(AssistLimit):
        Reviewer(_fake_run([], limit=True), tmp_path, AssistConfig(batch=1, jobs=1), "m").review(_frames(2))

def test_reviewer_sends_frames_without_candidates(tmp_path):
    calls = []
    Reviewer(_fake_run(calls), tmp_path, AssistConfig(batch=5, jobs=1), "m").review(_frames(2, cands=0))
    assert calls == [["f0", "f1"]]

def test_reviewer_counts_usage(tmp_path):
    r = Reviewer(_fake_run([]), tmp_path, AssistConfig(batch=1, jobs=2), "m")
    r.review(_frames(3))
    assert r.usage["input_tokens"] == 30 and r.usage["output_tokens"] == 6
```

`tests/test_brain_claude.py` 加：用现有 `fake_claude.py` 跑 `one_shot_message`，断言返回的 dict 里 `result` 和 `one_shot` 返回的文字一致。

- [ ] **Step 2:** 跑 `python -m pytest -q tests/test_assist.py tests/test_brain_claude.py` → 新测试 FAIL
- [ ] **Step 3:** 实现 `one_shot_message`、`assist_command`、`FrameInput`、`AssistLimit`、`Reviewer`
- [ ] **Step 4:** 跑测试 → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): Reviewer 分批并发核对、缓存、重试和额度处理"`

### Task 6: 候选框（COCO person 合并 / 自训模型）

**Files:**
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Consumes: `vision.detect.Detection`、`vision.track.iou`
- Produces:
  - `merge_proposals(groups: list[list[Detection]], iou_thr: float = 0.6) -> list[Detection]`：所有组合并、按分数从高到低贪心去重（和已留下的 IoU > `iou_thr` 的丢）
  - `class CocoPeople.__init__(self, paths: list[str], conf: float, imgsz: int)`、`.detect(frame) -> list[Detection]`：每个模型 `predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False)`，转成 `Detection("player", Rect, score)`，再 `merge_proposals`；ultralytics 在构造时才导入
  - `people_candidates(dets: list[Detection]) -> list[Rect]`：给 `--model` 用 —— 只留 `PEOPLE` 类的框

- [ ] **Step 1: 写失败测试**

```python
def test_merge_proposals_dedups_across_models():
    a = [Detection("player", Rect(0, 0, 100, 200), 0.3)]
    b = [Detection("player", Rect(5, 5, 100, 200), 0.6), Detection("player", Rect(500, 0, 50, 100), 0.2)]
    out = merge_proposals([a, b])
    assert [d.score for d in out] == [0.6, 0.2]

def test_people_candidates_filters_classes():
    dets = [Detection("player", Rect(0, 0, 1, 1), 0.9), Detection("name_tag", Rect(0, 0, 1, 1), 0.9),
            Detection("self", Rect(1, 1, 1, 1), 0.9)]
    assert people_candidates(dets) == [Rect(0, 0, 1, 1), Rect(1, 1, 1, 1)]
```

- [ ] **Step 2:** 跑测试 → FAIL
- [ ] **Step 3:** 实现
- [ ] **Step 4:** 跑测试 → PASS
- [ ] **Step 5:** `git commit -m "feat(assist): 人物候选框"`

### Task 7: 命令行接线、文档、实测

**Files:**
- Modify: `src/skydango/cli.py`（`perception label` 加 `--assist`、`--all-frames`；`_perception_label` 开头 `if args.assist: return _perception_label_assist(cfg, args, items)`；新函数 `_perception_label_assist`）
- Modify: `CLAUDE.md`（常用命令加一行；代码结构表 `vision/assist.py`）
- Modify: `docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md` §13 第 4~5 步
- Test: `tests/test_assist.py`（CLI 层用 monkeypatch 假 `Reviewer.run` / 假检测器 / 假 OCR）

**Interfaces:**
- Consumes: Task 1~6 全部；现有 `_brain_env`、`_images`、`hard_images`、`weak_labels`、`_panel_open`、`split_of`、`yolo_line`、`data_yaml`、`brain.images.thumb`、`image_block`
- Produces: `_perception_label_assist(cfg: Config, args, items: list[tuple[Path, str]]) -> None`，顺序：
  1. `--assist --spin` → `SystemExit("--assist 不能和 --spin 一起用：转圈录像已经能自动补 self，二选一")`
  2. `_brain_env(cfg)` 取命令和环境（没令牌 / 没 claude 直接报错退出，这之前不跑检测）
  3. 挑帧：`--from-runs` 或 `--all-frames` 不挑；否则 `pick_frames([thumb(imread(p)) for p, _ in items], …)`；打印"挑了 N / M 帧"
  4. 每帧：弱标注（同现有逻辑）、候选（`--model` → `people_candidates(detector.detect(frame))`，`source = args.model`；否则 `CocoPeople(...)`，`source = ",".join(proposal_models)`）
  5. `Reviewer(lambda c: one_shot_message(assist_command(base, cfg.assist), env, Path("tmp/assist-claude"), c, cfg.assist.timeout), out / "_assist", cfg.assist, source).review(...)`；捕获 `AssistLimit` → 打印"订阅额度用完了：已核对的帧存在 <out>/_assist/，额度恢复后重跑同一条命令会接着做" 并 `SystemExit(1)`（不写数据集）
  6. 写数据集：标注 = 弱标注 + `apply_review`（没核对 = 只有弱标注）；预览 `draw_review` → `_preview/`；`review_report` → `_assist/review.md`；`data.yaml`
  7. 打印汇总：挑帧数、各类数量、补框数、没核对数、`Reviewer.usage`

- [ ] **Step 1: 写失败测试**：`test_assist_rejects_spin`（`--assist --spin` 抛 SystemExit）；`test_assist_skips_picking_for_from_runs`（假 Reviewer 收到全部难例帧）；`test_assist_writes_labels_preview_and_report`（2 张合成图、假检测器给 1 个框、假 run 判 `player` + 补 1 个 `missing` → `labels/*/<stem>.txt` 有 2 行类别 0、`_preview/<stem>.jpg` 和 `_assist/review.md` 存在）；`test_assist_limit_exits_without_writing_dataset`
- [ ] **Step 2:** 跑测试 → FAIL
- [ ] **Step 3:** 实现 CLI、改文档
- [ ] **Step 4:** `python -m pytest -q` → 全绿（除已知的 `test_viewer.py::test_port_in_use_raises`，另有任务处理）
- [ ] **Step 5: 实测**：`python -m skydango perception label tmp/record/20260928-200526 -o tmp/assist-run --assist`，期望挑出 74 帧、`review.md` 生成；抽 10 帧预览和 `tmp/assist2/*_review.jpg`（子代理结果）对照，类别基本一致；把预览总览和清单给用户看
- [ ] **Step 6:** `git commit -m "feat(assist): perception label --assist 命令行接线和文档"`
