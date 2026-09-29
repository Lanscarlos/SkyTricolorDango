# 物品识别 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO 感知层认出座位 / 篝火 / 乐器 / 先祖并告诉大脑；辅助标注加物品模式，给现有数据集补标。

**Architecture:** 类别追加在 `[perception] classes` 末尾；`vision/people.py` 加 `Thing`，`PerceptionWatcher.objects()` 给身体 / 眼睛 / 网页用；
`assist.Reviewer` 抽出 `Protocol`（提示词、解析、版本），新模块 `vision/objlabel.py` 是物品模式的协议和写回逻辑，cli 加 `perception label --objects`。

**Tech Stack:** Python 3.11+、numpy / OpenCV、pytest（合成画面 + 假检测器 + 假 Claude）

**Spec:** `docs/superpowers/specs/2026-09-29-object-recognition-design.md`

## Global Constraints

- 中文注释 / 日志 / 给大脑和 Claude 的文字；风格照所在文件
- 新类别**只追加**在 `classes` 末尾：`[..., "typing", "bench", "bonfire", "instrument", "spirit"]`，旧编号 0~5 不变
- `OBJECT_NAMES = {"bench": "座位", "bonfire": "篝火", "instrument": "乐器", "spirit": "先祖"}`（顺序即类别顺序）
- 用 v4（6 类）模型时一切照旧、不报错
- 物品模式写标注前必须备份 `labels/`；编号 0~5 的行除"人改先祖"外不动
- 跑测试：`python -m pytest -q`；每个任务一个提交，带 CLAUDE.md 要求的署名行；完成后合并进 main 并推送

## Review Focus

1. **标注文件里有空行 / 多余空格 / 坐标越界的行**（手工修过的数据集常见）→ 读得进来、原样保留，不崩 → Task 6 测试
2. **Claude 把不是人物的 P 编号（超出范围、写成 "P9"、写成数字）判成先祖** → 忽略并记进 problems，不改别的行 → Task 6 测试
3. **同一帧重跑 `--objects`**（第二次）→ 物品行不重复累加，人改先祖不会把已经是 spirit 的行再改 → Task 6 / Task 7 测试
4. **数据集里有图没标注文件（新图）或有标注没图** → 没标注的当空文件处理，没图的跳过 → Task 7 测试
5. **物品框和人框重叠（先祖站在人群里）** → 追踪器不把物品和人接成同一条轨迹，people() 不数进先祖 → Task 2 测试

---

### Task 1: 类别、`Thing` 和配置

**Files:**
- Modify: `src/skydango/config.py`（`PerceptionConfig.classes` 默认值追加 4 类；新增 `object_min_hits: int = 3`、`object_near: float = 0.85`、`object_far: float = 0.65`）
- Modify: `config.example.toml`（`[perception]` 里 classes 若有列出则追加；加三个新键和注释）
- Modify: `src/skydango/vision/people.py`
- Test: `tests/test_config.py`、`tests/test_perception.py`

**Interfaces:**
- Produces:
  ```python
  OBJECT_NAMES: dict[str, str]  # 见 Global Constraints
  @dataclass(frozen=True)
  class Thing:
      track_id: int
      kind: str      # OBJECT_NAMES 的键
      box: Rect
      side: str      # 同 Person：side_of(框中心 x, 画面宽)
      distance: str  # "近" / "中" / "远"
  def object_distance(bottom: float, height: int, near: float, far: float) -> str  # bottom ≥ height*near 近，≥ height*far 中，否则远
  def describe_things(things: list[Thing]) -> str  # "座位（左边·近）、先祖（前面·远）"；空 → ""
  ```

- [ ] **Step 1: 写失败的测试**
  - `test_config.py::test_perception_object_classes_appended`：`PerceptionConfig().classes[:6] == ["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]` 且 `[6:] == ["bench", "bonfire", "instrument", "spirit"]`；`object_min_hits == 3`、`object_near == 0.85`、`object_far == 0.65`；example 配置照样能加载
  - `test_perception.py::test_object_distance_and_describe_things`：`object_distance(1000, 1080, 0.85, 0.65) == "近"`、`(800, …) == "中"`、`(500, …) == "远"`；`describe_things([Thing(1, "bench", Rect(0,0,1,1), "左边", "近"), Thing(2, "spirit", Rect(0,0,1,1), "前面", "远")]) == "座位（左边·近）、先祖（前面·远）"`；`describe_things([]) == ""`
- [ ] **Step 2: 跑测试确认失败**：`python -m pytest -q tests/test_config.py tests/test_perception.py -k "object"` → FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**：`python -m pytest -q tests/test_config.py tests/test_perception.py` → PASS
- [ ] **Step 5: 提交** `feat(perception): 物品类别（座位 / 篝火 / 乐器 / 先祖）和 Thing`

---

### Task 2: `PerceptionWatcher.objects()` 和 overlay

**Files:**
- Modify: `src/skydango/vision/perception.py`（`objects()`；`overlay()` 物品分支）
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `Thing`、`OBJECT_NAMES`、`object_distance`、`side_of`（Task 1）
- Produces: `PerceptionWatcher.objects(now: float) -> list[Thing]`；`overlay()` 物品项 `{"kind": 类别名, "label": 中文名, ...}`

**行为：** `paused` → `[]`；只看 `last_tracks` 里 `cls in OBJECT_NAMES`、`now - t.last <= PEOPLE_STALE`、`t.hits >= cfg.object_min_hits` 的；
`distance = object_distance(t.box.y2, self._frame_h, cfg.object_near, cfg.object_far)`；排序同 `people()`（左 → 前 → 右，同一边按底边从下到上 = 近的在前）。

- [ ] **Step 1: 写失败的测试**（`bench(x, y, w, h)` = `Detection("bench", Rect(...), 0.9)` 之类的帮手）
  - `test_objects_need_hits_and_sort`：同一组框（左边底边 1000 的 bench、中间底边 500 的 spirit、右边底边 800 的 bonfire）连跑两帧 → `objects()` 为空；第三帧 → `[("bench","左边","近"), ("spirit","前面","远"), ("bonfire","右边","中")]`
  - `test_objects_empty_when_stale_or_paused`：三帧后 `objects(now + 2)` 为空；`hold("x")` 后为空
  - `test_spirit_among_people_not_counted_as_stranger`（Review Focus 5）：一个 spirit 框和一个 player 框重叠 50% 连跑 3 帧 → `people()` 只有那个 player、`objects()` 有 spirit，二者 track_id 不同
  - `test_overlay_has_object_labels`：overlay 里 bench 的一项 `kind == "bench"`、`label == "座位"`
  - `test_objects_empty_with_six_class_model`：检测器只出 player / name_tag → `objects()` 为 `[]`
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_perception.py -k "objects or spirit_among or overlay_has_object"` → FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_perception.py` → PASS
- [ ] **Step 5: 提交** `feat(perception): objects() 给出画面里的物品（方位、远近）`

---

### Task 3: 告诉大脑：状态、眼睛、提示词、网页

**Files:**
- Modify: `src/skydango/brain/body.py`（`status()`）
- Modify: `src/skydango/brain/images.py`（`scene_note`）
- Modify: `src/skydango/brain/prompt.py`（`BRAIN_RULES` 的「光遇常识」）
- Modify: `src/skydango/vision/viewer.py`（`describe_env`；页面 `COLORS` / `NAMES`）
- Test: `tests/test_brain_body.py`、`tests/test_brain_images.py`、`tests/test_brain_prompt.py`、`tests/test_viewer.py`

**Interfaces:**
- Consumes: `env.objects(now)`（Task 2，env 可能没有这个方法）、`describe_things`、`OBJECT_NAMES`

**文案（照抄）：**
- status：`"画面里的东西：" + describe_things(things)`，紧跟在"画面里：…"那段之后（没有"画面里"时照样按原顺序加在它该在的位置）；空列表不加
- scene_note：有物品时在人那段之后追加
  ```
  画面里认出的东西（坐标按这张图）：
  - 座位：(420, 610) 附近
  没列出的东西按你自己看到的说。
  ```
  坐标 = 框中心 × scale，四舍五入；物品顺序同 `objects()`
- 提示词「光遇常识」末尾加：`- 状态里的“画面里的东西”是认出来的座位、篝火、乐器、先祖（带方位和远近）；想过去可以用 move 小步走、边走边看；坐下、弹琴还不会，别答应。`
- 网页：`describe_env` 在 env 有 `objects` 时加 `out["附近的东西"] = describe_things(env.objects(now)) or "没有"`；
  `COLORS` 加 `bench:"#1d4ed8", bonfire:"#ea580c", instrument:"#f472b6", spirit:"#ffffff"`，`NAMES` 加 `bench:"座位", bonfire:"篝火", instrument:"乐器", spirit:"先祖"`

- [ ] **Step 1: 写失败的测试**
  - body：`FakeEnv` 子类带 `objects(now)` 返回两个 `Thing` → `"画面里的东西：座位（左边·近）、先祖（前面·远）" in b.status()`；返回 `[]` 或没有这个方法 → 不含"画面里的东西"
  - images：`scene_note` 的 env 带 `overlay` 和 `objects` → 输出含 `"画面里认出的东西（坐标按这张图）：\n- 座位：(205, 280) 附近"`（scale 0.5，框 Rect(400, 500, 20, 120)）和 `"没列出的东西按你自己看到的说。"`；没物品时两句都没有
  - prompt：`"画面里的东西" in static_prompt(ReplyConfig())` 且含 `"坐下、弹琴还不会"`
  - viewer：`describe_env` 对带 `objects` 的 env 有 `"附近的东西"`，值是 describe 文字；返回空时是 `"没有"`；页面 HTML 含 `bench:"#1d4ed8"` 和 `spirit:"先祖"`
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_brain_body.py tests/test_brain_images.py tests/test_brain_prompt.py tests/test_viewer.py -k "thing or object or 东西 or objects"` → FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_brain_body.py tests/test_brain_images.py tests/test_brain_prompt.py tests/test_viewer.py` → PASS
- [ ] **Step 5: 提交** `feat(brain): 状态、眼睛、提示词和网页带上画面里的物品`

---

### Task 4: `perception compare` 的物品统计

**Files:**
- Modify: `src/skydango/vision/compare.py`（`FrameResult.objects`、`compare_frames`、`summarize`、`report_md`）
- Test: `tests/test_compare.py`

**Interfaces:**
- Produces: `FrameResult.objects: dict[str, int] = field(default_factory=dict)`（这一帧 `yolo.last_tracks` 里每个物品类几个）；`summarize(...)["objects"] = {类别: {"frames": 出现的帧数, "avg": 出现时平均每帧几个（保留 2 位）}}`，只列出现过的类；
  `report_md` 加一节 `## 物品`：每类一行 `- 座位：出现在 12 帧，平均每帧 1.5 个`；一个都没有时写 `没有认出物品（模型里没有物品类别，或者录像里没有）`

- [ ] **Step 1: 写失败的测试**：构造 3 个 `FrameResult`（objects 分别 `{"bench": 2}`、`{"bench": 1, "spirit": 1}`、`{}`）→ `summarize` 的 `objects == {"bench": {"frames": 2, "avg": 1.5}, "spirit": {"frames": 1, "avg": 1.0}}`；`report_md` 含 `"## 物品"`、`"- 座位：出现在 2 帧，平均每帧 1.5 个"`；全空时含那句"没有认出物品"
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_compare.py -k object` → FAIL
- [ ] **Step 3: 实现**（`compare_frames` 里数 `yolo.last_tracks` 中 `t.cls in OBJECT_NAMES` 的）
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_compare.py` → PASS
- [ ] **Step 5: 提交** `feat(compare): 报告加物品统计`

---

### Task 5: `assist.Reviewer` 抽出 `Protocol`

**Files:**
- Modify: `src/skydango/vision/assist.py`
- Test: `tests/test_assist.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class FrameInput:
      stem: str
      image: np.ndarray
      candidates: list[Rect]
      people: list[Rect] = field(default_factory=list)   # 物品模式：已有人物框（P1、P2…）
      hints: list[str] = field(default_factory=list)     # 物品模式：候选框的类别猜测（和 candidates 一一对应）

  @dataclass(frozen=True)
  class Protocol:
      version: int
      system: str                                                          # claude -p 的 --system-prompt
      build: Callable[[list[FrameInput], AssistConfig], list[dict]]        # 一批帧 → 内容块
      parse: Callable[[str, list[FrameInput]], dict[str, Any]]             # 回答 → 帧名 → 核对结果（取不出的帧不在里面）

  PEOPLE_PROTOCOL: Protocol   # 现有行为：version=PROMPT_VERSION、system=ASSIST_SYSTEM、build/parse 包装现有函数
  Reviewer(run, cache_dir, cfg, source, protocol: Protocol = PEOPLE_PROTOCOL)
  def assist_command(base, cfg, system: str = ASSIST_SYSTEM) -> list[str]
  ```
- 缓存键：`{"prompt_version": protocol.version, "source": …, "candidates": …}`，`people` 非空时再加 `"people": [[x1,y1,x2,y2], …]`、`hints` 非空时加 `"hints"`；比对整个键。标人的旧缓存键不变（依然命中）

- [ ] **Step 1: 写失败的测试**
  - `test_reviewer_uses_protocol`：一个假协议（build 返回一个文本块、parse 返回 `{stem: "ok"}`）→ `review` 结果是 `"ok"`，run 收到的内容就是 build 的；缓存命中时不再调 run
  - `test_people_cache_key_unchanged`：用现有写法预先写好一个旧格式缓存文件（只有 prompt_version / source / candidates / review）→ 默认协议照样命中
  - `test_cache_key_includes_people`：同一帧 `people` 不同 → 缓存不命中
  - `test_assist_command_system_prompt`：`assist_command(base, cfg, system="X")` 里 `--system-prompt` 后面是 `X`
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_assist.py -k "protocol or cache_key or system_prompt"` → FAIL
- [ ] **Step 3: 实现**（`_cached` 用 `protocol.parse(json.dumps({stem: review}), [f])`；`_batch` 用 `protocol.build` / `protocol.parse`）
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_assist.py` → PASS（旧测试全过）
- [ ] **Step 5: 提交** `refactor(assist): Reviewer 的提示词和解析抽成 Protocol`

---

### Task 6: 物品模式的协议和写回（`vision/objlabel.py`）

**Files:**
- Create: `src/skydango/vision/objlabel.py`
- Test: `tests/test_objlabel.py`

**Interfaces:**
- Consumes: `FrameInput`、`Protocol`、`extract_json`、`_box`（Task 5 / 现有 assist）、`OBJECT_NAMES`（Task 1）、`weaklabel.yolo_line`
- Produces:
  ```python
  OBJECTS = tuple(OBJECT_NAMES)                  # ("bench", "bonfire", "instrument", "spirit")
  OBJECT_PROMPT_VERSION = 1
  OBJECT_SYSTEM: str
  OBJECT_RULES: str
  @dataclass class ObjectVerdict: cls: str; fixed: Rect | None; note: str        # cls ∈ OBJECTS ∪ {"not_object", "duplicate"}
  @dataclass class ObjectReview:
      verdicts: dict[int, ObjectVerdict]
      missing: list[tuple[str, Rect, str]]
      spirits: list[int]                           # 判成先祖的人物框编号（P 后面的数字，从 1 起）
      unsure: str
      problems: list[str] = field(default_factory=list)
  def build_object_message(frames: list[FrameInput], cfg: AssistConfig) -> list[dict]
  def parse_object_review(text: str, frames: list[FrameInput]) -> dict[str, ObjectReview]
  def apply_object_review(candidates: list[Rect], review: ObjectReview) -> list[tuple[str, Rect]]
  def people_in_labels(text: str, classes: list[str], width: int, height: int) -> list[Rect]   # player / player_unlit 行按文件顺序 → P1、P2…
  def rewrite_labels(text: str, classes: list[str], objects: list[tuple[str, Rect]], spirits: list[int], width: int, height: int) -> str
  def draw_objects_preview(frame, objects: list[tuple[str, Rect]], people: list[Rect], spirits: list[int], candidates: list[Rect], review: ObjectReview | None) -> np.ndarray
  def objects_report(results: list[tuple[str, ObjectReview | None]]) -> str
  OBJECTS_PROTOCOL = Protocol(OBJECT_PROMPT_VERSION, OBJECT_SYSTEM, build_object_message, parse_object_review)
  ```

**提示词（照抄）：**
```python
OBJECT_SYSTEM = "你是游戏截图的目标检测标注员。按用户给的规则逐帧标出物品、核对候选框，只输出一个 JSON 对象，不要别的文字。"
OBJECT_RULES = """下面是游戏《光·遇》(Sky) 的截图，每帧一张，要给目标检测补标"物品"。每张图上画了：
- 白色细网格：每 100 像素一条，边上的数字是原图像素坐标（原图 1920×1080）
- 灰色细框 P1、P2…：已经标好的人物框（玩家）
- 彩色框 1、2…：检测器给的物品候选框（可能没有）；每个框的精确坐标和检测器的猜测在图前面的文字里

物品只有四类：
- bench：座位。明显是给人坐的：长椅、石凳、秋千座、摆好的坐垫。台阶、石头、地面不算。
- bonfire：篝火。燃着的篝火堆，包括玩家放的篝火道具。蜡烛、烛火堆、灯笼不算。
- instrument：乐器。摆在场景里、没人拿着的乐器（钢琴、竖琴架……）。玩家手里拿着的不算。
- spirit：先祖。发光的先祖灵魂（站在原地、没被收集的），以及先祖回忆里半透明的人形。玩家、团子、宠物、跟着玩家的小光团不算。

要做的：
1. 每个候选框判一个：bench / bonfire / instrument / spirit / not_object（不是这四类）/ duplicate（和另一个候选是同一个东西）。类别对但框明显不准，给 fixed_box [x1,y1,x2,y2]。
2. 漏掉的物品（没有候选框框到的）：写进 missing，给类别和框 [x1,y1,x2,y2]（原图像素，框住整个物品，被挡住就框露出来的部分；只框先祖的身体，不含头顶光效）。框高不到 15 像素的太远，不用标。
3. 灰色人物框里其实是先祖的（先祖被当成了玩家）：把编号写进 spirits，比如 ["P2"]。只写确定的；是玩家的不要写。
拿不准的写进 unsure，一句话说明。

只输出一个 JSON 对象，每帧一项：
{"帧名": {"boxes": {"1": {"cls": "bench", "fixed_box": [x1,y1,x2,y2], "note": "…"}}, "missing": [{"cls": "spirit", "box": [x1,y1,x2,y2], "note": "…"}], "spirits": ["P2"], "unsure": ""}}
fixed_box、note 没有就不写；没有候选框时 boxes 写 {}。"""
```
每帧文字块：`帧 {stem}：人物 P1=[x1,y1,x2,y2]；P2=…；候选 1=[…](猜 bench)；2=…`（没有人物写"人物 无"，没有候选写"候选 无，只看有没有漏掉的物品"），后面跟画了网格 + 灰色 P 框 + 彩色编号候选框的图。

**解析和写回规则：**
- `spirits`：只收 `"P<n>"` 形式、`1 ≤ n ≤ len(people)` 的；别的（`"P9"` 超范围、`3`、`"x"`）进 `problems`（Review Focus 2）
- `missing` 只收 `cls in OBJECTS` 且框合法的
- 回答里缺某个候选编号 / 类别不认识 → `problems`，按 not_object 处理（同人物模式）
- `rewrite_labels`：逐行解析；空行丢掉；**编号不在 6~9 的行原样保留文本**（包括坐标越界的行，Review Focus 1），除非它是 `spirits` 里的第 n 个人物行 → 类别号改成 spirit、坐标文本不变；
  编号 6~9 的旧行全部丢掉，末尾追加这次的 `objects`（`yolo_line`）；解析不了的行原样保留。已经是 spirit 的行不算人物（不参与 P 编号，Review Focus 3）
- `objects_report`：标题 `# 待核对清单（物品模式）`；依次三节「人物框改成了先祖」「Claude 补了物品框」「没核对成」，每节列帧名 + 说明；再列 `unsure` / `problems`

- [ ] **Step 1: 写失败的测试**（`tests/test_objlabel.py`）
  - `test_build_message_lists_people_and_candidates`：两个 P 框、一个候选（hint bench）→ 文字块含 `"人物 P1=[…]；P2=[…]"`、`"候选 1=[…](猜 bench)"`；规则块是 `OBJECT_RULES`；图块一个
  - `test_parse_reads_boxes_missing_spirits`：正常回答 → verdicts、missing、`spirits == [2]`
  - `test_parse_bad_spirit_ids_go_to_problems`：`["P9", 3, "P1"]`（两个人物）→ `spirits == [1]`，problems 两条
  - `test_apply_object_review_keeps_objects_uses_fixed_adds_missing`
  - `test_rewrite_keeps_people_lines_and_replaces_objects`：输入含 player、name_tag、旧 bench 行、一个空行、一个坐标 1.2 的越界 player 行 → 输出里 0~5 类行文本不变（越界行也在）、旧 bench 行没了、新物品行在末尾、没有空行
  - `test_rewrite_turns_player_into_spirit`：spirits=[2] → 第二个 player / player_unlit 行类别号变成 9、坐标文本不变
  - `test_rewrite_twice_is_stable`（Review Focus 3）：同一结果写两次 == 写一次；已经是 spirit 的行不占 P 编号
  - `test_people_in_labels_order`：`people_in_labels` 按文件顺序返回 player / player_unlit，跳过 self / spirit
  - `test_objects_report_orders_sections`
  - `test_protocol_version_and_system`：`OBJECTS_PROTOCOL.version == OBJECT_PROMPT_VERSION`、`system == OBJECT_SYSTEM`
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_objlabel.py` → FAIL（模块不存在）
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_objlabel.py` → PASS
- [ ] **Step 5: 提交** `feat(assist): 物品模式的提示词、解析和写回`

---

### Task 7: `perception label --objects`

**Files:**
- Modify: `src/skydango/cli.py`（参数、`_perception_label` 分派、新函数 `_perception_label_objects`）
- Test: `tests/test_objlabel.py`

**Interfaces:**
- Consumes: `OBJECTS_PROTOCOL`、`OBJECTS`、`people_in_labels`、`apply_object_review`、`rewrite_labels`、`draw_objects_preview`、`objects_report`（Task 6）；`Reviewer`、`assist_command(..., system=OBJECT_SYSTEM)`（Task 5）
- Produces: `perception label <数据集目录> --objects [--model 模型]`

**行为：**
- 和 `--assist` / `--spin` / `--from-runs` 一起用 → `SystemExit("--objects 只对已经标好人的数据集跑，不能和 --assist / --spin / --from-runs 一起用")`
- `cfg.perception.classes` 里缺物品类 → `SystemExit`，提示配置里 classes 要带上 bench / bonfire / instrument / spirit
- 图：`<目录>/images/*/*.jpg`（按路径排序）；标注：`<目录>/labels/<同一 split>/<帧名>.txt`，不存在当空（Review Focus 4）
- 开始前：`labels/` 存在就整个复制到 `<目录>/_backup/labels-<YYYYmmdd-HHMMSS>/`，打印备份位置
- 候选：有 `--model` 时 `make_detector(...)` 的结果里 `cls in OBJECTS` 的框（hints = 类别），否则空
- `Reviewer(run, <目录>/_assist_objects, cfg.assist, source=args.model or "none", protocol=OBJECTS_PROTOCOL)`；每 `_ASSIST_CHUNK` 帧一段；额度用完同 `--assist` 的 SystemExit 提示
- 每帧：核对成了就 `rewrite_labels` 写回（新建目录）、预览写 `<目录>/_preview_objects/<帧名>.jpg`；没核对成的标注不动、清单里列出
- 结束：`<目录>/_assist/objects.md`、`data.yaml` 写 10 类；打印每类物品数、人改先祖几个、没核对几帧、用量

- [ ] **Step 1: 写失败的测试**（假环境照 `tests/test_assist.py::_cli_env`：合成 2 张图的数据集 `images/train/0000.jpg`、`0001.jpg`，`0000.txt` 有一个 player 行 + 一个 name_tag 行，`0001` 没有标注文件；假 Claude 回 `{"boxes": {}, "missing": [{"cls": "bench", "box": [100,700,400,900]}], "spirits": ["P1"], "unsure": ""}`）
  - `test_objects_mode_rewrites_labels_and_backs_up`：跑完 `0000.txt` 的 player 行变 spirit（类别 9）、name_tag 行原样、末尾多一个 bench 行（类别 6）；`0001.txt` 被创建、只有 bench 行（P1 不存在 → problems）；`_backup/labels-*/train/0000.txt` 是原文件；`_preview_objects/0000.jpg` 存在；`_assist/objects.md` 里"人物框改成了先祖"一节有 `0000`；`data.yaml` 有 `9: spirit`
  - `test_objects_mode_rerun_is_stable`：同一命令再跑一次（缓存命中）→ 标注文件内容和第一次一样
  - `test_objects_mode_rejects_other_modes`：`--objects --assist` → SystemExit
  - `test_objects_mode_needs_object_classes`：`cfg.perception.classes` 只有 6 类 → SystemExit
- [ ] **Step 2: 确认失败**：`python -m pytest -q tests/test_objlabel.py -k objects_mode` → FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**：`python -m pytest -q tests/test_objlabel.py tests/test_assist.py` → PASS
- [ ] **Step 5: 提交** `feat(cli): perception label --objects 给数据集补标物品`

---

### Task 8: 文档、全量测试、合并

**Files:**
- Modify: `CLAUDE.md`（代码结构表加 `vision/objlabel.py`；YOLO 一节加"物品识别"要点；常用命令加 `perception label <数据集> --objects`）
- Modify: `docs/progress/2026-09-28-yolo-training.md`（新流程、录数据建议、v5 上线门槛）
- Modify: `docs/superpowers/specs/2026-09-29-object-recognition-design.md`（状态改成代码已完成、待数据和训练）

- [ ] **Step 1: 写文档**
- [ ] **Step 2: 全量测试**：`python -m pytest -q --ignore=tests/test_ocr_real.py` → 全部 PASS
- [ ] **Step 3: 提交** `docs: 物品识别`
- [ ] **Step 4: 最终评审后合并进 main 并推送**（CLAUDE.md 规矩）
