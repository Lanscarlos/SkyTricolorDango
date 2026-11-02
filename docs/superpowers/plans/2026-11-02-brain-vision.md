# 大脑直接看图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 OpenAI 兼容的大脑（deepseek-flash）直接看图：工具返回的图不再代看，唤醒消息挑时机自动附一张画了编号框的截图，大脑对不上时写「核对：…」。

**Architecture:** 新文件 `brain/vision.py` 放纯计算（画编号图 + 图例、该不该附图、看不看得了图）；`ToolLoopBrain` 负责把图放进请求、历史里换成占位、接口拒收时退回代看；`Brain` 醒来时经钩子在身体线程取图；`cli` 接线；大脑时间线显示缩略图。`[brain.vision] enabled = false` 逐字照旧。

**Tech Stack:** Python 3.13、OpenCV（cv2）、openai SDK（OpenAI 兼容接口）、原生 JS（管理面板）、pytest。

**Spec:** `docs/superpowers/specs/2026-11-02-brain-vision-design.md`

## Global Constraints

- 用中文写注释、日志、提示词、文档；注释密度和命名照周围代码。
- `[brain.vision]`：`enabled = true`、`auto = true`、`min_gap = 20.0`、`size = [960, 540]`（`config.VisionConfig`，挂在 `BrainConfig.vision`）。
- `enabled = false`：`ToolLoopBrain` 的请求、历史、提示词、代看逐字照旧，`tests/data/toolloop_baseline.json` 不改、照样过。
- Claude 大脑（`BrainSession`）不收自动附图，`BrainSession.send` 签名不变；沙盒（`world.text_only`）不取图、不附图、用看不了图的提示词。
- 历史（`_past`）里不留图；图例文字保留。
- status 的「画面里」一行不改。
- 测试命令：`.venv\Scripts\python.exe -m pytest -q`（没有 `.venv` 的机器用 `python -m pytest -q`）；全量测试放后台跑。
- 不在主目录切分支：用 `using-git-worktrees` 开 worktree，改完合并进 main、推送，再按 CLAUDE.md 回主目录 `git pull --ff-only`。

## Review Focus

1. **一步里调了好几个看图工具**（`look` + `look_person`）：只接一条带图的 user 消息、按调用顺序放图；工具都报错时不接空的 user 消息。→ Task 3 测试 `test_two_image_tools_one_user_message`、`test_tool_error_no_image_message`。
2. **轮数到顶时最后一步调了看图工具**（`rounds >= max_steps + 2` 那次没执行）：不能留下没有对应 tool 消息的图。→ Task 3 测试 `test_image_not_added_for_unexecuted_calls`。
3. **画面黑着 / 身体还没截到图**（`last_frame is None`）：自动附图返回 None、这一轮照常只发文字，不报错。→ Task 2 测试 `test_pick_none_without_frame`、`test_pick_none_when_black`。
4. **接口拒收后的下一轮**：不再带图、工具走代看、唤醒消息开头只提醒一次「现在看不了图了」。→ Task 3 测试 `test_rejected_image_turns_vision_off`。
5. **overlay 里有没坐标意义的条目**（火焰候选 `ring`、名字标签 `name` / `tag`、`typing`、`rejected`）：不编号、不进图例。→ Task 1 测试 `test_skips_tags_bubbles_rings`。

---

### Task 1: 配置 + 编号图和图例

**Files:**
- Create: `src/skydango/brain/vision.py`
- Modify: `src/skydango/config.py`（`CompactConfig` 后面加 `VisionConfig`；`BrainConfig` 加 `vision` 字段）
- Modify: `src/skydango/vision/perception.py`（加 `overlay_fresh`）
- Test: `tests/test_brain_vision.py`

**Interfaces:**
- Produces:
  - `config.VisionConfig(enabled: bool = True, auto: bool = True, min_gap: float = 20.0, size: list[int] = [960, 540])`；`BrainConfig.vision: VisionConfig`
  - `PerceptionWatcher.overlay_fresh(now: float) -> bool`：`not self.paused and now - self._frame_at <= PEOPLE_STALE`
  - `brain/vision.py`：
    - `@dataclass Attachment: legend: str; block: dict; src: str; w: int; h: int; thumb: str`（`block` 是 `images.image_block` 那种 Claude 风格的图片块；`thumb` 是缩到 480 宽的 JPEG base64，给时间线；`src` 是 `"auto"` 或工具名）
    - `mark_frame(frame: np.ndarray, boxes: list[dict] | None, size: tuple[int, int], quality: int, src: str = "auto") -> Attachment`：`boxes` 是 `env.overlay(now)` 的条目，`None` = 不画框
    - 常量 `MARK_KINDS: dict[str, tuple[int, int, int]]`（BGR）：`friend` 绿、`maybe` / `unsure` / `dango_look` 黄、`stranger` / `unlit` / `player` 白、`self` 灰、`icon` / `request` 和 `OBJECT_NAMES` 里的物品青
    - `LEGEND_HEAD = "图上的编号（身体认的，可能错）："`、`NO_BOXES = "图上没画框（身体这会儿没认出东西）"`

- [ ] **Step 1: 写失败的测试** `tests/test_brain_vision.py`

```python
def test_numbers_left_to_right_and_legend():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    boxes = [
        {"x": 1278, "y": 331, "w": 124, "h": 157, "kind": "friend", "label": "懒洋洋大王"},
        {"x": 980, "y": 475, "w": 75, "h": 101, "kind": "self", "label": "团子"},
        {"x": 300, "y": 400, "w": 80, "h": 120, "kind": "maybe", "label": "像番茄炒蛋盖饭?"},
        {"x": 1500, "y": 500, "w": 80, "h": 120, "kind": "stranger", "label": "陌生人A"},
        {"x": 1300, "y": 250, "w": 60, "h": 60, "kind": "icon", "label": "牵手"},
    ]
    a = mark_frame(frame, boxes, (960, 540), 80)
    assert a.legend.splitlines() == [
        LEGEND_HEAD,
        "① 像番茄炒蛋盖饭（没看到名字）",
        "② 你自己",
        "③ 懒洋洋大王（好友，看到名字）",
        "④ 图标：牵手",
        "⑤ 陌生人A",
    ]
    assert (a.w, a.h) == (960, 540) and a.src == "auto"
    assert a.block["type"] == "image" and a.block["source"]["media_type"] == "image/jpeg"

def test_skips_tags_bubbles_rings():
    boxes = [{"x": 10, "y": 10, "w": 50, "h": 20, "kind": k, "label": "x"} for k in ("name", "tag", "typing", "ring", "rejected")]
    assert mark_frame(np.zeros((1080, 1920, 3), np.uint8), boxes, (960, 540), 80).legend == NO_BOXES

def test_no_boxes_means_plain_image():
    a = mark_frame(np.zeros((1080, 1920, 3), np.uint8), None, (960, 540), 80)
    assert a.legend == NO_BOXES

def test_draws_on_copy_with_kind_color():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    mark_frame(frame, [{"x": 100, "y": 300, "w": 200, "h": 300, "kind": "friend", "label": "小明"}], (1920, 1080), 95)
    assert not frame.any()  # 不改原图
    # 解码 block 回来，框的左边线附近是 MARK_KINDS["friend"] 的颜色（允许 JPEG 误差 40）
```

图例每行的写法：`friend` → `{label}（好友，看到名字）`；`maybe` / `unsure` → `{label 去掉末尾的 ?}（没看到名字）`；`self` / `dango_look` → `你自己` / `像你自己`；`unlit` → `陌生人（没点火，黑影）`；`stranger` → `{label}`；`player` → `没认出是谁的人`；`icon` / `request` → `图标：{label}`；物品 → `{label}`。编号用 ①~⑳（超过 20 个用 `(21)`），按框的 `x` 从左往右。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision.py -q`
Expected: FAIL（`ModuleNotFoundError: skydango.brain.vision`）

- [ ] **Step 3: 实现** `VisionConfig`、`BrainConfig.vision`、`overlay_fresh`、`mark_frame`

画法：原图坐标画在 `frame.copy()` 上（框线 3 px，框左上角外半径 20 的实心圆 + 黑色数字，cv2 `FONT_HERSHEY_SIMPLEX`），画完用 `images.fit` 缩到 `size`、`images.image_block` 编码；`thumb` 用 `images.encode_jpeg(fit(view, (480, 270)), 70)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/vision.py src/skydango/config.py src/skydango/vision/perception.py tests/test_brain_vision.py
git commit -m "feat(brain): 给大脑看的编号截图和图例"
```

---

### Task 2: 该不该附图、看不看得了图

**Files:**
- Modify: `src/skydango/brain/vision.py`
- Test: `tests/test_brain_vision.py`

**Interfaces:**
- Consumes: Task 1 的 `mark_frame`、`VisionConfig`
- Produces:
  - `TRIGGERS = frozenset({"chat", "arrive", "approach", "notice", "lull"})`
  - `should_attach(kinds: set[str], doubtful: bool, last_at: float, now: float, cfg: VisionConfig) -> bool`：`cfg.auto` 且 `now - last_at >= cfg.min_gap` 且（`kinds & TRIGGERS` 或 `doubtful`）
  - `doubtful(boxes: list[dict]) -> bool`：有 `kind` 为 `maybe` / `unsure` 的，或 `kind == "icon"` 且 `unknown` 为真
  - `class AutoVision`：`__init__(self, cfg: VisionConfig, quality: int)`；`last_at: float = -inf`；`pick(self, events: list, now: float, frame, env, black: bool) -> Attachment | None`：`frame is None` 或 `black` → None；`boxes = env.overlay(now) if hasattr(env, "overlay_fresh") and env.overlay_fresh(now) else None`；`should_attach(...)` 为真才 `mark_frame(..., src="auto")` 并刷新 `last_at`
  - `brain_sees(session, sees_model: bool, cfg: VisionConfig, text_only: bool) -> bool`：`session` 是 `BrainSession` → `sees_model`（旧行为）；否则 `cfg.enabled and not text_only and sees_model and not getattr(session, "vision_off", "")`

- [ ] **Step 1: 写失败的测试**（测试文件里自己定义 `ev(kind) -> Event(kind, "x", 0.0)`、`frame() -> np.zeros((1080, 1920, 3), np.uint8)`）

```python
def test_should_attach_triggers_and_gap():
    cfg = VisionConfig()
    assert should_attach({"chat"}, False, float("-inf"), 100.0, cfg)
    assert not should_attach({"stranger", "leave"}, False, float("-inf"), 100.0, cfg)   # 只有背景事件
    assert should_attach(set(), True, float("-inf"), 100.0, cfg)                         # 拿不准
    assert not should_attach({"chat"}, False, 90.0, 100.0, cfg)                           # min_gap 20
    assert not should_attach({"chat"}, False, float("-inf"), 100.0, VisionConfig(auto=False))

def test_doubtful():
    assert doubtful([{"kind": "unsure"}]) and doubtful([{"kind": "icon", "unknown": True}])
    assert not doubtful([{"kind": "friend"}, {"kind": "icon"}])

def test_pick_none_without_frame():  # Review Focus 3
    assert AutoVision(VisionConfig(), 80).pick([ev("chat")], 100.0, None, None, False) is None

def test_pick_none_when_black():
    assert AutoVision(VisionConfig(), 80).pick([ev("chat")], 100.0, frame(), None, True) is None

def test_pick_stale_overlay_gives_plain_image_and_sets_last_at():
    env = SimpleNamespace(overlay_fresh=lambda now: False, overlay=lambda now: pytest.fail("不该读"))
    auto = AutoVision(VisionConfig(), 80)
    a = auto.pick([ev("chat")], 100.0, frame(), env, False)
    assert a.legend == NO_BOXES and auto.last_at == 100.0
    assert auto.pick([ev("chat")], 110.0, frame(), env, False) is None   # 隔不到 20 秒

def test_brain_sees():
    tl = SimpleNamespace(vision_off="")
    assert brain_sees(tl, True, VisionConfig(), False)
    assert not brain_sees(tl, True, VisionConfig(enabled=False), False)
    assert not brain_sees(tl, True, VisionConfig(), True)            # 沙盒
    assert not brain_sees(tl, False, VisionConfig(), False)          # 模型没勾能看图
    assert not brain_sees(SimpleNamespace(vision_off="400"), True, VisionConfig(), False)
    # BrainSession：不管 enabled，照旧看模型（构造一个 BrainSession.__new__(BrainSession)）
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision.py -q`
Expected: FAIL（`ImportError: cannot import name 'should_attach'`）

- [ ] **Step 3: 实现** `TRIGGERS`、`should_attach`、`doubtful`、`AutoVision`、`brain_sees`（`BrainSession` 在函数里延迟导入）

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/vision.py tests/test_brain_vision.py
git commit -m "feat(brain): 挑时机给大脑附图、判断大脑看不看得了图"
```

---

### Task 3: ToolLoopBrain 收图、历史占位、拒收退回

**Files:**
- Modify: `src/skydango/brain/toolloop.py`
- Modify: `src/skydango/brain/sessions.py`（按能不能看图选提示词）
- Test: `tests/test_brain_toolloop_vision.py`

**Interfaces:**
- Consumes: Task 1 的 `Attachment`；`models.openai_compat.to_openai_content`
- Produces:
  - `ToolLoopBrain.send(self, text: str, image: Attachment | None = None) -> dict`；结果多 `images: list[dict]`（每张 `{"src", "w", "h"}`，没图不写这个键）
  - `ToolLoopBrain.vision_off: str`（空 = 没关；拒收后是原因，≤ 80 字）
  - 常量：`TOOL_IMAGE_NOTE = "截图在下一条消息里。"`、`IMAGE_HOLDER = "（这时附了一张截图）"`、`TOOL_IMAGE_HOLDER = "（看了截图）"`、`VISION_OFF_NOTE = "（现在看不了图了，看图的工具会请眼睛代看）"`、`SIGHTED_NOTE`（spec §5 原文 + `BLIND_NOTE` 后四条）
  - `make_session(..., text_only: bool = False)`：OpenAI 兼容时 `sighted = cfg.vision.enabled and not text_only and registry.sees(ref)`，系统提示词用 `SIGHTED_NOTE if sighted else BLIND_NOTE`（+ `ASIDE_NOTE`）

请求怎么组（spec §2.1 §4）：
- 唤醒消息有图：user 的 content = `to_openai_content([{"type":"text","text": text}, {"type":"text","text": image.legend}, image.block])`。
- 工具返回 `list[dict]`（ToolBox 在大脑能看图时原样返回内容块）：tool 消息 content = `TOOL_IMAGE_NOTE + "\n" + 文字块拼起来`；图片块攒起来。这一步所有 tool 消息之后，有图就追加**一条** user 消息：每张图前一段文字 `（{工具名} 的截图）`，然后图（`to_openai_content`）。
- 存历史：唤醒消息存成 `text + "\n" + image.legend + "\n" + IMAGE_HOLDER`（再照旧截 `HISTORY_TEXT`）；`_replay` 遇到 content 是 list 的 user 消息换成 `{"role": "user", "content": TOOL_IMAGE_HOLDER}`。
- 拒收：带图的请求抛错、`status_code == 400`、错误文字（小写）含 `image` / `vision` / `multimodal` 之一 → `vision_off = 原因`、WARNING 一行、把 `messages` 里所有图片片段换成文字 `（图发不过去）` 后**重发这一次**（不算轮数）；别的错照旧 `ModelError`。`vision_off` 之后第一次 `send` 在唤醒消息开头加 `VISION_OFF_NOTE + "\n"`（只加一次）。
- 压缩：`_after` 用的 `messages` 原样（带图），不改。

- [ ] **Step 1: 写失败的测试** `tests/test_brain_toolloop_vision.py`（沿用 `tests/test_brain_toolloop.py` 的 `client` / `msg` / `tool_call` / `toolbox` 夹具写法，记下每次 `create` 收到的 `messages`）

```python
def test_wake_image_in_request_and_placeholder_in_history():
    # send("醒来", image=att) → 第一次请求最后一条 user 的 content 是 [text, legend, image_url]
    # 第二轮请求里，第一轮的唤醒消息是字符串，含 legend 和 IMAGE_HOLDER，不含 "image_url"
    # result["images"] == [{"src": "auto", "w": 960, "h": 540}]

def test_tool_image_goes_to_next_user_message():
    # toolbox.run 返回 ([{"type":"text","text":"位置说明"}, image_block], False)
    # 第二次请求：tool 消息 content == TOOL_IMAGE_NOTE + "\n位置说明"；紧跟一条 user，content[0]["text"] == "（look 的截图）"、content[1]["type"] == "image_url"
    # 下一轮历史里那条 user 是 TOOL_IMAGE_HOLDER

def test_two_image_tools_one_user_message():   # Review Focus 1
    # 一步调 look + look_person，都返回图 → 两条 tool 之后只有一条 user，里面两张图，顺序 look 在前

def test_tool_error_no_image_message():        # Review Focus 1
    # toolbox.run 返回 ("没找到", True) → tool 后面直接是下一次 assistant，没有多出的 user

def test_image_not_added_for_unexecuted_calls():  # Review Focus 2
    # always_calls 风格的客户端一直调 look：最后一次没执行的调用后面不追加 user 图

def test_rejected_image_turns_vision_off():   # Review Focus 4
    # 第一次 create 抛 status_code=400、message "image_url is not supported" → 重发一次的 messages 里没有 image_url
    # brain.vision_off 非空；下一次 send 的唤醒消息以 VISION_OFF_NOTE 开头；再下一次不再加

def test_other_400_still_model_error():
    # 400 但文字不含 image/vision/multimodal → 抛 ModelError，vision_off 仍是空

def test_no_image_request_unchanged():
    # send(text) 不带图：请求和原来逐字一样（content 是字符串）

def test_make_session_picks_sighted_note(tmp_path):
    # 照 test_brain_toolloop.py::test_blind_note_in_system 的写法：
    # deepseek-flash（内置勾了能看图）→ system 含 SIGHTED_NOTE；VisionConfig(enabled=False) 或 text_only=True → 含 BLIND_NOTE
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_toolloop_vision.py -q`
Expected: FAIL（`send() got an unexpected keyword argument 'image'` 等）

- [ ] **Step 3: 实现** 上面的组请求、历史占位、拒收重发、`make_session` 选提示词

- [ ] **Step 4: 跑这个文件和旧的大脑循环测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_toolloop_vision.py tests/test_brain_toolloop.py tests/test_brain_toolloop_compact.py -q`
Expected: 全 PASS（`test_disabled_matches_baseline` 照样过）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/toolloop.py src/skydango/brain/sessions.py tests/test_brain_toolloop_vision.py
git commit -m "feat(brain): OpenAI 兼容大脑直接收图，历史里换成占位，拒收时退回代看"
```

---

### Task 4: 接线（look 画编号、醒来取图、sees、记录）

**Files:**
- Modify: `src/skydango/brain/body.py`（`look` 加 `marked` 参数）
- Modify: `src/skydango/brain/tools.py`（`ToolBox` 加 `marked` 参数；`look` 分支传给身体）
- Modify: `src/skydango/brain/loop.py`（`Brain.snapshot` 钩子；`_wake` 带图发；`_log` 记 `images` / `check`；「核对」INFO 日志）
- Modify: `src/skydango/cli.py`（`sees()` 改用 `brain_sees`；`make_session` 传 `text_only`；接 `AutoVision`；`_models_line` 带看不了图的原因）
- Test: `tests/test_brain_vision_wiring.py`

**Interfaces:**
- Consumes: Task 1 `mark_frame`、Task 2 `AutoVision` / `brain_sees`、Task 3 `send(text, image=)` / `vision_off`
- Produces:
  - `Body.look(self, marked: bool = False) -> list[dict]`：`marked` 且 env 有 `overlay_fresh` → 返回 `[{"type":"text","text": att.legend}, att.block]`（`mark_frame(frame, boxes, tuple(cfg.brain.vision.size), cfg.brain.jpeg_quality, src="look")`）；否则逐字照旧
  - `ToolBox(..., marked: Callable[[], bool] = lambda: False)`；`look(image=true)` 调 `b.call(lambda: b.look(marked=self.marked()))`
  - `Brain.snapshot: Callable[[list[Event]], Attachment | None] | None = None`；`_wake` 里 `image = self._snapshot(events)`（钩子出错记 WARNING、按 None），有图 `self.session.send(text, image=image)`，没图 `self.session.send(text)`
  - `parse_check(text: str) -> str`（放 `brain/vision.py`）：最后的文字里以「核对：」开头的行（去掉前缀，多行用「；」连起来），没有返回空
  - `brain.jsonl` 每轮多 `images`（取 `result["images"]`）、`check`（非空才写）；有「核对」时 `log.info("核对：%s", check)`

cli 接线（`_run_brain`）：
- `sees()` → `brain_sees(session, registry.sees(ModelRef(session.provider, session.model)), cfg.brain.vision, world.text_only)`（`brain_box` 为空时 False）
- `ToolBox(..., marked=lambda: cfg.brain.vision.enabled and sees())`
- `build = functools.partial(make_session, ..., text_only=world.text_only)`
- `cfg.brain.vision.enabled and not world.text_only` 时：`auto = AutoVision(cfg.brain.vision, cfg.brain.jpeg_quality)`；`brain.snapshot = lambda evs: body.call(lambda: auto.pick(evs, body.clock(), body.last_frame, env, body.blackout)) if isinstance(brain.session, ToolLoopBrain) and sees() else None`
- `_models_line`：`getattr(brain.session, "vision_off", "")` 非空时末尾加「；看不了图（{原因}）」

- [ ] **Step 1: 写失败的测试** `tests/test_brain_vision_wiring.py`

```python
def test_body_look_marked_returns_legend_then_image():
    # 用 tests/test_brain_body.py 的假身体夹具，env 给 overlay_fresh=True、overlay 一个 friend 框
    # body.look(marked=True) → [text(legend 以 LEGEND_HEAD 开头), image]；body.look() 和原来一样 [image, text(scene_note)]

def test_toolbox_look_passes_marked():
    # ToolBox(body, eyes, marked=lambda: True, sees=lambda: True) 调 look image=true → 身体收到 marked=True

def test_brain_wake_sends_image_from_snapshot():
    # 假 session 记下 send 的参数；brain.snapshot 返回一个 Attachment → send(text, image=att)
    # snapshot 返回 None → send(text) 只有一个位置参数（BrainSession 签名不变）
    # snapshot 抛异常 → 照样 send(text)，不中断这一轮

def test_brain_log_records_images_and_check(tmp_path):
    # result = {"result": "不说：x\n核对：①其实是懒洋洋大王", "images": [{"src": "auto", "w": 960, "h": 540}], ...}
    # brain.jsonl 最后一行 images == [...]，check == "①其实是懒洋洋大王"

def test_parse_check():
    assert parse_check("说了\n核对：A\n核对：B") == "A；B"
    assert parse_check("没有") == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision_wiring.py -q`
Expected: FAIL

- [ ] **Step 3: 实现** body / tools / loop / cli 的改动

- [ ] **Step 4: 跑这个文件和相关旧测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_vision_wiring.py tests/test_brain_body.py tests/test_brain_tools.py tests/test_brain_loop.py tests/test_cli_brain.py tests/test_brain_eyes.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py src/skydango/brain/tools.py src/skydango/brain/loop.py src/skydango/brain/vision.py src/skydango/cli.py tests/test_brain_vision_wiring.py
git commit -m "feat(brain): 醒来挑时机附编号截图，look 也画编号，记下附图和核对"
```

---

### Task 5: 大脑时间线显示附图 + 设置页开关

**Files:**
- Modify: `src/skydango/brain/trace.py`（`begin` 多一个可选 `image: dict | None`；`_steps` 认出带图的 user 消息）
- Modify: `src/skydango/brain/loop.py`（`self._trace("begin", reason, text, image_meta)`）
- Modify: `src/skydango/brain/toolloop.py`（工具图那条 user 消息也经 `on_message` 报给时间线）
- Modify: `src/skydango/console/static/brainlog.js`、`src/skydango/console/console.css`（颜色只用 `var(--…)`）
- Modify: `src/skydango/console/settings.py`（加 `brain.vision.enabled`、`brain.vision.auto` 两个 `Field`，放在 `brain.compact.*` 后面）
- Test: `tests/test_brain_trace.py`、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: Task 1 `Attachment.thumb` / `src`
- Produces:
  - 轮的字典多 `image: {"src": "auto", "thumb": "<base64>"} | None`；步骤多一种 `{"kind": "image", "src": "<工具名>", "thumb": "<base64>"}`
  - `BrainTrace` 只给最近 `MAX_THUMBS = 20` 张留 `thumb`，更早的把 `thumb` 置空（轮和步骤都算）
  - 「核对」步骤：最后文字里以「核对：」开头的行，`brainlog.js` 加 `check` 样式高亮
  - `ToolLoopBrain` 工具图：`on_message({"type": "user", "message": {"content": [{"type": "image_shot", "src": name, "thumb": att_thumb}]}})`，`_steps` 把 `image_shot` 转成 `image` 步骤（缩略图在 toolloop 里用 `images.encode_jpeg(fit(...,(480,270)),70)` 现做）

- [ ] **Step 1: 写失败的测试**

```python
def test_trace_begin_keeps_image_and_caps_thumbs():
    tr = BrainTrace()
    for i in range(25):
        tr.begin("events", "x", {"src": "auto", "thumb": f"t{i}"}); tr.finish({}, 1.0)
    turns = tr.since(0, 0)["turns"]
    assert sum(1 for t in turns if t["image"] and t["image"]["thumb"]) == 20
    assert turns[-1]["image"] == {"src": "auto", "thumb": "t24"}

def test_trace_image_shot_step():
    # feed 一条 image_shot → 当前轮 steps 里 {"kind": "image", "src": "look", "thumb": "..."}

def test_settings_has_vision_switches():
    # settings 字段清单里有 brain.vision.enabled / brain.vision.auto，类型 bool，分组 brain
```

前端（`brainlog.js`，没有 JS 单测的部分照现有页面手动看）：轮头在原因后面加「附图·自动」/「附图·look」；轮里第一行前放一个 `<details>`，`summary` 是「▸ 看到的图」，展开是 `<img src="data:image/jpeg;base64,…">`（宽度 100%、最大 480px）；`thumb` 为空时只留文字「（图太旧，没留）」；`copyText` 里写「[附图·自动]」不带 base64。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_trace.py -q`
Expected: FAIL

- [ ] **Step 3: 实现** trace / loop / toolloop / brainlog.js / console.css / settings.py

- [ ] **Step 4: 跑测试确认通过，再手动看页面**

Run: `.venv\Scripts\python.exe -m pytest tests/test_brain_trace.py tests/test_console_settings.py tests/test_brain_toolloop_vision.py -q`
Expected: PASS。然后起管理面板（`python -m skydango console`）开沙盒页，确认大脑控制台照常渲染、没有 JS 报错（沙盒不附图，只看不坏）。

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/trace.py src/skydango/brain/loop.py src/skydango/brain/toolloop.py src/skydango/console/static/brainlog.js src/skydango/console/console.css src/skydango/console/settings.py tests/
git commit -m "feat(console): 大脑控制台显示附图和核对，设置页加看图开关"
```

---

### Task 6: 文档、全量测试、合并

**Files:**
- Modify: `config.example.toml`（`[brain.compact]` 后面加 `[brain.vision]` 一节，四个键带注释）
- Modify: `CLAUDE.md`（「统管大脑」一节加一条「大脑直接看图」：spec / 计划路径、默认开、怎么触发、历史不留图、拒收退回、「核对」、`enabled = false` 逐字照旧、**还没在真机验证，spec §10 四步没走**；代码结构表 `brain/` 那一行加 `vision.py` 编号图 / 附图时机）

- [ ] **Step 1: 改文档**

- [ ] **Step 2: 全量测试（后台跑）**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全 PASS；有失败按 `systematic-debugging` 查，不跳过

- [ ] **Step 3: 提交、合并进 main、推送**

```bash
git add config.example.toml CLAUDE.md
git commit -m "docs: 大脑直接看图"
```

按 `finishing-a-development-branch` + CLAUDE.md：合并进 main、推送；确认 19390 / 19391 / 19392 没人听、主目录没有未提交的改动后，在主目录 `git pull --ff-only`，否则告诉用户之后再拉。
