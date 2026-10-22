# 身份板（外观为主、名字标签证实）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建纯计算的 `IdentityBoard`，按轨迹攒标签 / 外观 / 团子线索判"这个框是谁"，接进感知层（`[identity] enabled` 开关，关着逐字旧流程），外加拥挤录像的半自动标准答案和新旧对比评估。

**Architecture:** `vision/identity.py` 只吃纯数据 `FrameIn`、吐 `StepOut`（标签分配、每条轨迹的 `Verdict`、这一帧可学的样本）；`perception.py` 负责把一帧的检测 / 外观样本 / 平移翻成 `FrameIn`，再把 `StepOut` 写回原来的 `data` 字段，下游不改。评估工具 `vision/identity_eval.py` 录下每帧 `FrameIn`，扫门槛时只重放身份板。

**Tech Stack:** Python 3.13、numpy、OpenCV、pytest（合成证据 / 假检测器）；DINOv2-small ONNX（已有，`[appearance] dino`）。

**Spec:** `docs/superpowers/specs/2026-10-22-identity-board-design.md`

## Global Constraints

- `[identity] enabled = false`（代码默认）时感知层**逐字**是旧流程：老测试一条不改都要过
- 下游字段名和含义不变：`data["name"]`、`data["tagged"]`、`data["maybe"]`（好友像）、`data["unsure"] = (名字, 开始时间)`、`data["unsure_called"]`、`data["unsure_miss"]`、`data["dango"] = 框`、`data["stranger"]`、`data["sid"]`
- 身份板模块不 import cv2、不碰画面、不调模型；同一个 `FrameIn` 序列重放结果完全一样（不读墙钟、不用随机数）
- 配置默认值照 spec §2.3：`tag_hold 0.5`、`tag_stick 0.5`、`window 5`、`min_samples 3`、`friend_feature "dino"`、`drop 0.03`、权重 `pos 0.4 / look 0.4 / self 0.2 / pivot 0.2`、`dango_min 0.5`、`dango_switch 1.0`、`dango_learn 0.75`；`match` / `margin` / `unsure` / `unsure_wait` 读 `[appearance]`，`stranger_low` 默认等于 `[appearance] unsure`
- 喊一声不新加额度：只通过已有的 `PerceptionWatcher.unsure()` / `mark_unsure_called()` / `_call_tick` 那条路，身体（`Body._watch_call`）不改
- 临时文件、评估输出放 `tmp/`；测试用 `python -m pytest -q`（本机系统 Python 有 pytest；`tests/conftest.py` 会把 worktree 的 `src` 放最前）
- 用中文写注释、日志、提示，风格照 `perception.py`（短句、说清"为什么"）

## Review Focus

1. **标签在两条轨迹之间隔帧跳**（10-22 轨迹 6）：谁都不该在跳的那几帧变成陌生人，也不该学进底库 → Task 2 `test_flipping_tag_never_makes_a_stranger`
2. **好友底库还空着时**（刚上线、好友还没挂过标签）：没标签的点过火的人"对每个好友都低于门槛"是空真，会立刻判陌生人——应和旧流程一样等够 `stranger_after + STRANGER_GRACE` → Task 3 `test_empty_galleries_wait_like_before`
3. **团子分全部不计入**（没有记住的位置、底库不到 3 张、没 self、镜头没转）：不该随便挑一个人当团子 → Task 4 `test_no_dango_evidence_no_dango`
4. **身份板抛异常**：这一帧沿用上一帧结论、连续 10 帧出错自动关、回到旧流程不崩 → Task 7 `test_board_errors_fall_back_to_legacy`
5. **面板开关横移动画里**：团子位置会跟着画面横移 400 px，不该把团子换给别人 → Task 4 `test_panel_slide_keeps_dango`

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/skydango/vision/identity.py`（新） | 数据类型（`TagCand` `Scores` `TrackIn` `CallEnd` `FrameIn` `Verdict` `StepOut`）+ `IdentityBoard`，纯计算 |
| `src/skydango/config.py` | `IdentityConfig`，`Config.identity` |
| `src/skydango/vision/appearance.py` | `AppearanceBook.identity_scores(sample, feature) -> Scores` |
| `src/skydango/vision/perception.py` | 构造参数 `identity=`；`_identity_frame()` 拼 `FrameIn`；`_identity_apply()` 写回；`process()` 里按开关分流；出错自关 |
| `src/skydango/cli.py` | `_perception()` 按 `[identity]` 建身份板、启动日志；`perception identity-truth` / `identity-eval` 子命令 |
| `src/skydango/console/settings.py` | 设置页 `identity.enabled` |
| `src/skydango/vision/identity_eval.py`（新） | 草稿 / 拼图 / `--apply` 展开答案、按框对答案、指标、录 `FrameIn`、扫门槛、报告 |
| `tests/test_identity.py`（新） | 身份板单测 |
| `tests/test_perception_identity.py`（新） | 感知层集成 |
| `tests/test_identity_eval.py`（新） | 评估工具 |
| `CLAUDE.md`、`docs/progress/2026-10-22-identity.md`（新） | 文档 |

---

### Task 1: 配置和数据类型、带记性的标签分配

**Files:**
- Create: `src/skydango/vision/identity.py`
- Modify: `src/skydango/config.py`（`AppearanceConfig` 之后加 `IdentityConfig`；`Config` 里 `appearance` 后面加 `identity`）
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces:
  - `IdentityConfig`（dataclass）字段：`enabled: bool = False`、`tag_hold: float = 0.5`、`tag_stick: float = 0.5`、`window: int = 5`、`min_samples: int = 3`、`friend_feature: str = "dino"`、`drop: float = 0.03`、`stranger_low: float | None = None`（None = 用 `[appearance] unsure`）、`dango_w_pos: float = 0.4`、`dango_w_look: float = 0.4`、`dango_w_self: float = 0.2`、`dango_w_pivot: float = 0.2`、`dango_min: float = 0.5`、`dango_switch: float = 1.0`、`dango_learn: float = 0.75`
  - 常量：`FRIEND_TAG = "friend_tag"`、`FRIEND_LOOK = "friend_look"`、`DANGO = "dango"`、`UNSURE = "unsure"`、`STRANGER = "stranger"`、`PENDING = "pending"`
  - `@dataclass(frozen=True) TagCand(key: str, name: str | None, err: float)`：`key` = 读出来的名字，读不出时 `"#<标签轨迹 id>"`
  - `@dataclass(frozen=True) Scores(dino: dict[str, float], color: dict[str, float], me: float | None)`：每个好友底库按 DINOv2 / 颜色各自的最像一张（没分的好友不列）；`me` = 团子底库 DINOv2 最像一张。身份板按 `friend_feature` 挑：`dino` / `color` 用对应那份，`both` 两份都有时取平均、只有一份取那份
  - `@dataclass(frozen=True) TrackIn(id: int, box: Rect, first: float, unlit: bool = False, under_panel: bool = False, tags: tuple[TagCand, ...] = (), scores: Scores | None = None, self_score: float | None = None, with_bg: bool | None = None, relink: str | None = None, sid: str | None = None, called_at: float | None = None)`；`scores` 只在这一帧算出好样本时非 None
  - `@dataclass(frozen=True) CallEnd(at: float, shown: dict[int, str])`：窗口里哪条轨迹亮出了哪个名字
  - `@dataclass(frozen=True) FrameIn(now: float, width: int, height: int, panel: bool, sliding: bool, moved: bool, tracks: tuple[TrackIn, ...], call_end: CallEnd | None = None)`
  - `@dataclass Verdict(kind: str, name: str | None = None, score: float = 0.0, since: float = 0.0)`
  - `@dataclass StepOut(tags: dict[int, str], verdicts: dict[int, Verdict], learn: dict[int, tuple[str, str]])`：`tags` = 轨迹 id → 这一帧分到的标签 `key`；`learn` = 轨迹 id → `(kind, who)`，kind ∈ `"friend"` / `"me"` / `"stranger"`
  - `IdentityBoard(cfg: IdentityConfig, acfg: AppearanceConfig, *, stranger_after: float, grace: float, stranger_min_height: float, track_buffer: float, call_window: float)`，方法 `step(frame: FrameIn) -> StepOut`、属性 `dango_id: int | None`
- 这个 task 的 `step` 只做标签分配和 `FRIEND_TAG` / `PENDING`，其余 kind 在后面的 task 加

- [ ] **Step 1: 写失败的测试**（`tests/test_identity.py`；辅助函数 `board(**kw)` 建板子，`frame(now, *tracks, **kw)` 拼 `FrameIn`，`tr(id, x, tags=(), **kw)` 拼 `TrackIn`，框默认 `Rect(x, 400, 90, 220)`、`first=0.0`）

```python
def test_tag_goes_to_closest_and_becomes_friend_after_hold():
    b = board()
    for t in (0.0, 0.2, 0.4):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1100, tags=(TagCand("小明", "小明", 30.0),))))
        assert out.tags == {1: "小明"} and out.verdicts[1].kind == PENDING
    out = b.step(frame(0.5, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1100, tags=(TagCand("小明", "小明", 30.0),))))
    assert out.verdicts[1] == Verdict(FRIEND_TAG, "小明", out.verdicts[1].score, 0.5)

def test_sticky_tag_resists_a_slightly_better_neighbour():
    b = board()
    for t in (0.0, 0.3, 0.6):  # 轨迹 1 稳定挂着小明
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),))))
    out = b.step(frame(0.7, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),)), tr(2, 1050, tags=(TagCand("小明", "小明", 7.0),))))
    assert out.tags == {1: "小明"}  # 10 × tag_stick 0.5 = 5 < 7

def test_name_moves_only_after_tag_hold():
    b = board()
    for t in (0.0, 0.3, 0.6):
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 10.0),)), tr(2, 1300)))
    for t in (0.7, 0.9, 1.1):  # 标签只挨着轨迹 2 了
        out = b.step(frame(t, tr(1, 1000), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
        assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[1].name == "小明"
    out = b.step(frame(1.2, tr(1, 1000), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    assert out.verdicts[2].kind == FRIEND_TAG and out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "小明"

def test_unread_tag_marks_tagged_without_name():
    b = board()
    for t in (0.0, 0.3, 0.6):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("#77", None, 3.0),))))
    assert out.verdicts[1] == Verdict(FRIEND_TAG, None, out.verdicts[1].score, out.verdicts[1].since)

def test_identity_config_defaults():
    c = IdentityConfig()
    assert (c.enabled, c.tag_hold, c.window, c.friend_feature, c.dango_min) == (False, 0.5, 5, "dino", 0.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_identity.py -q`
Expected: FAIL（`ModuleNotFoundError: skydango.vision.identity`）

- [ ] **Step 3: 实现**
  - `IdentityConfig` 写进 `config.py`，`Config.identity: IdentityConfig = field(default_factory=IdentityConfig)`（`load_config` 按 dataclass 字段自动认 `[identity]`，不用改加载器——先确认 `load_config` 是按 `fields(Config)` 走的）
  - 分配：这一帧所有（轨迹, 候选）按误差升序贪心（一个 key 一条轨迹、一条轨迹一个 key）；上一帧把这个 key 分给了同一条轨迹的，误差乘 `tag_stick`
  - 每条轨迹记"当前名字 key 的这一段从什么时候开始挂、最后一次挂"（中间断开超过 `tag_hold` 算新的一段）；这一段持续满 `tag_hold` → 这条轨迹**稳定持有**这个 key
  - 一个 key 同时只有一个稳定持有者：新轨迹稳定后，原来的持有者失去它 → `Verdict(UNSURE, 名字, since=失去的时刻)`（key 是 `#…` 的读不出名字的，失去后回 `PENDING`）
  - 稳定持有者这一帧没分到标签（被挡）照旧 `FRIEND_TAG`
  - 本帧没出现的轨迹：超过 `track_buffer` 删掉它的状态

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_identity.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/identity.py src/skydango/config.py tests/test_identity.py
git commit -m "feat(identity): 身份板的数据类型和带记性的标签分配"
```

---

### Task 2: 外观判好友（像 / 拿不准）、一个名字一条轨迹、接回

**Files:**
- Modify: `src/skydango/vision/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: Task 1 全部类型
- Produces: `step` 新增 `FRIEND_LOOK` / `UNSURE`（外观）判定；`Verdict.score` = 判定所用的中位分

规则（spec §3.2~§3.3）：每条轨迹存最近 `window` 个样本按 `friend_feature` 挑出来的好友分；样本数 ≥ `min_samples` 时，对每个好友取中位数。
最高中位分 ≥ `match` 且领先第二名 ≥ `margin` → 候选 `FRIEND_LOOK`；已经是 `FRIEND_LOOK` 的只在 < `match - drop` 时掉下来；
同一名字多条候选 → 中位分最高的拿，被稳定持有（`FRIEND_TAG`）的名字不给；中位分在 `[unsure, match)` → `UNSURE`（name = 最像的）；
`relink` 有名字且（样本不够或对他的中位分 ≥ `unsure`）→ `FRIEND_LOOK`；稳定标签永远盖过外观。

- [ ] **Step 1: 写失败的测试**（辅助 `sc(me=None, **friends)` = `Scores(dino=friends, color=friends, me=me)`）

```python
def test_look_alike_after_min_samples():
    b = board()
    for i, t in enumerate((0.0, 0.1, 0.2)):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.93, 番茄=0.70))))
    assert out.verdicts[1].kind == FRIEND_LOOK and out.verdicts[1].name == "小明"

def test_one_outlier_sample_does_not_flip_look():
    b = board()
    for t, s in ((0.0, 0.93), (0.1, 0.93), (0.2, 0.93), (0.3, 0.40), (0.4, 0.93)):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=s, 番茄=0.7))))
        assert t < 0.2 or out.verdicts[1].kind == FRIEND_LOOK

def test_margin_too_small_is_unsure_not_look():
    b = board()
    for t in (0.0, 0.1, 0.2):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.90, 番茄=0.88))))
    assert out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "小明"

def test_name_given_to_one_track_only():
    b = board()
    for t in (0.0, 0.1, 0.2):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.95)), tr(2, 1400, scores=sc(小明=0.91))))
    assert out.verdicts[1].kind == FRIEND_LOOK and out.verdicts[2].kind != FRIEND_LOOK

def test_stable_tag_beats_look():
    b = board()
    for t in (0.0, 0.3, 0.6):
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("番茄", "番茄", 3.0),), scores=sc(小明=0.95, 番茄=0.5))))
    assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[1].name == "番茄"

def test_flipping_tag_never_makes_a_stranger():  # 10-22 轨迹 6
    b = board()
    for i in range(30):
        t = i * 0.1
        a, c = (3.0, 5.0) if i % 2 else (5.0, 3.0)  # 5 × tag_stick 0.5 = 2.5 < 3：黏在先拿到的那条上
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("饶总", "饶总", a),)), tr(2, 1080, tags=(TagCand("饶总", "饶总", c),))))
        assert all(v.kind != STRANGER for v in out.verdicts.values()) and out.learn == {}

def test_relink_counts_as_look_unless_appearance_disagrees():
    b = board()
    out = b.step(frame(0.0, tr(1, 1000, relink="小明")))
    assert out.verdicts[1].kind == FRIEND_LOOK
    for t in (0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 1000, relink="小明", scores=sc(小明=0.50, 番茄=0.6))))
    assert out.verdicts[1].kind != FRIEND_LOOK
```

- [ ] **Step 2: 跑测试确认失败**　Run: `python -m pytest tests/test_identity.py -q`　Expected: 新加的 FAIL
- [ ] **Step 3: 按上面的规则实现**（`test_flipping_tag_never_makes_a_stranger` 靠 `tag_stick` 让标签黏在先拿到的轨迹上，另一条一直 `PENDING`；`learn` 在 Task 5 才填，这里恒为空）
- [ ] **Step 4: 跑测试确认通过**　Run: `python -m pytest tests/test_identity.py -q`　Expected: PASS
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 外观判好友——中位分、滞回、一个名字一条轨迹、接回"`

---

### Task 3: 陌生人、拿不准喊一声、黑影、面板

**Files:**
- Modify: `src/skydango/vision/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: Task 1~2
- Produces: `step` 新增 `STRANGER`；`UNSURE` 的超时 / 喊一声收尾

规则（spec §4）：起算点 `base = max(first, 最后一次 under_panel 的时刻)`；`under_panel` 这一帧不判陌生人。
1. 黑影（`unlit`）：从没稳定持有过名字、`now - first ≥ stranger_after` → `STRANGER`
2. 从没稳定持有过名字、样本 ≥ `min_samples`、**有至少一个好友底库**、最近样本中位分对每个好友 < `stranger_low`、`me` 为 None 或 < `stranger_low` → `STRANGER`（不用等 `stranger_after`，但至少 `now - base ≥ stranger_after`）
3. 从没稳定持有过名字、`now - base ≥ stranger_after + grace` 还不是 `FRIEND_LOOK` / `UNSURE` → `STRANGER`（含"好友底库全空"和"一直没好样本"）；
   **例外**：最近 `stranger_after` 秒内这条轨迹头顶有过读出名字的标签候选（`TagCand.name` 不为 None，哪怕分给了别人）→ 不走这条，满 `stranger_after` 后进 `UNSURE`（name = 那个名字），等喊一声（人挤在一起、标签被旁边的人占着，10-22 轨迹 6）
4. 框高 < `stranger_min_height × height` → 永远不判陌生人（`PENDING`）
5. `UNSURE`：`call_end` 里 `shown` 给了这条轨迹它的名字 → 立刻 `FRIEND_TAG`（`since = now`，算稳定持有，Task 5 让这一帧可学）；`call_end.at == called_at` 却没给它 → `STRANGER`；
   `called_at` 为 None 且 `UNSURE` 已持续 `unsure_wait + call_window` → `STRANGER`；`called_at` 不为 None、窗口还没结束（没收到对应 `call_end`）→ 继续 `UNSURE`
6. 一旦 `STRANGER`，之后只有稳定标签 / 喊一声亮名字能改回来（外观再像也不改，跟旧流程一样靠名字纠错）

- [ ] **Step 1: 写失败的测试**

```python
def test_clearly_nobody_becomes_stranger():
    b = board()
    for t in (0.0, 0.5, 1.0):
        out = b.step(frame(t, tr(1, 1000, scores=sc(me=0.4, 小明=0.5, 番茄=0.6))))
    assert out.verdicts[1].kind == STRANGER

def test_empty_galleries_wait_like_before():
    b = board(stranger_after=1.0, grace=1.5)
    out = b.step(frame(2.0, tr(1, 1000, first=0.0, scores=sc())))
    assert out.verdicts[1].kind == PENDING
    out = b.step(frame(2.6, tr(1, 1000, first=0.0, scores=sc())))
    assert out.verdicts[1].kind == STRANGER

def test_once_tagged_never_stranger_by_looks_only():
    b = board()
    for t in (0.0, 0.3, 0.6):
        b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 5.0),)), tr(2, 1300)))
    for t in (0.7, 0.9, 1.1, 1.3):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.3)), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    out = b.step(frame(1.4, tr(1, 1000, scores=sc(小明=0.3)), tr(2, 1300, tags=(TagCand("小明", "小明", 4.0),))))
    assert out.verdicts[1].kind == UNSURE

def test_unsure_call_shown_confirms_and_not_shown_rejects():
    b = board()
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5)), tr(2, 1400, scores=sc(番茄=0.86, 小明=0.5))))
    out = b.step(frame(3.0, tr(1, 1000, called_at=2.5), tr(2, 1400, called_at=2.5), call_end=CallEnd(2.5, {1: "小明"})))
    assert out.verdicts[1].kind == FRIEND_TAG and out.verdicts[2].kind == STRANGER

def test_unsure_without_call_times_out():
    b = board(call_window=6.0)  # unsure_wait 2.0
    for t in (0.0, 0.1, 0.2):
        b.step(frame(t, tr(1, 1000, scores=sc(小明=0.86, 番茄=0.5))))
    assert b.step(frame(8.0, tr(1, 1000))).verdicts[1].kind == UNSURE
    assert b.step(frame(8.3, tr(1, 1000))).verdicts[1].kind == STRANGER

def test_tag_held_by_neighbour_means_unsure():  # 10-22 轨迹 6：标签一直被旁边的人占着
    b = board()
    for i in range(30):
        t = i * 0.1
        a, c = (3.0, 5.0) if i % 2 else (5.0, 3.0)
        out = b.step(frame(t, tr(1, 1000, tags=(TagCand("饶总", "饶总", a),)), tr(2, 1080, tags=(TagCand("饶总", "饶总", c),))))
    assert out.verdicts[1].kind == UNSURE and out.verdicts[1].name == "饶总"

def test_unlit_and_panel_and_small():
    b = board()
    out = b.step(frame(1.2, tr(1, 1000, unlit=True), tr(2, 300, under_panel=True), tr(3, 1500, box=Rect(1500, 400, 20, 40))))
    assert out.verdicts[1].kind == STRANGER and out.verdicts[2].kind == PENDING and out.verdicts[3].kind == PENDING
```

- [ ] **Step 2: 跑测试确认失败**　Run: `python -m pytest tests/test_identity.py -q`　Expected: 新加的 FAIL
- [ ] **Step 3: 按规则实现**（`board()` 默认 `stranger_after=1.0, grace=1.5, stranger_min_height=0.05, track_buffer=1.0, call_window=6.0`、`AppearanceConfig()` 默认门槛）
- [ ] **Step 4: 跑测试确认通过**　Expected: PASS
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 陌生人、拿不准喊一声收尾、黑影和面板"`

---

### Task 4: 团子分和"团子只有一个"

**Files:**
- Modify: `src/skydango/vision/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: Task 1~3
- Produces: `step` 新增 `DANGO`；`IdentityBoard.dango_id`；`IdentityBoard.dango_spot(panel: bool) -> Rect | None`（记住的位置，给 perception 的 `_self_areas` 以外的地方查看 / 调试用）

规则（spec §5.1），四项都在 0~1，不计入的项权重不参与归一：
- `pos`：记住的位置按 `panel` 分开存，每帧用现任团子的框更新（不过期）；`dist = |中心x 差| / 记住的框宽`，`pos = clip(1 - dist / 1.5, 0, 1)`；`frame.moved` 时 `pos = 0.5`；`frame.sliding` 或这个面板状态还没记过 → 不计入
- `look`：`scores.me` 不为 None 时 `clip((me - max(scores.dino.values(), default=0)) / LOOK_SPAN, 0, 1)`，`LOOK_SPAN = 0.15`；这一帧没样本沿用这条轨迹上一次的 `look`；从没有过 → 不计入
- `self`：`self_score` 不为 None → 它；None → 不计入
- `pivot`：`with_bg` False → 1、True → 0、None → 不计入
- 平滑：每条轨迹 `smooth = 0.7 × smooth + 0.3 × 本帧分`（第一次直接用本帧分）
- 候选：没有稳定持有名字、不是黑影、至少有一项计入、`smooth ≥ dango_min`；**只剩 `pos` 一项计入时不算**（光靠站在中间不够）
- 现任团子留着，除非：轨迹消失 / 稳定挂上了名字 / 它的 `smooth < dango_min` 连续 `dango_switch` 秒 / 另一个候选的 `smooth` 连续 `dango_switch` 秒比它高；`sliding` 时不换人
- 没有现任时取最高的候选；判成 `DANGO` 的轨迹不参加标签分配

- [ ] **Step 1: 写失败的测试**

```python
def test_dango_without_self_box_by_position_and_look():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))  # 记下位置
    for t in (0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 960, scores=sc(me=0.9, 小明=0.5)), tr(2, 1300, scores=sc(me=0.5, 小明=0.5))))
    assert out.verdicts[1].kind == DANGO and b.dango_id == 1

def test_no_dango_evidence_no_dango():
    b = board()
    out = b.step(frame(0.0, tr(1, 960), tr(2, 1300)))
    assert all(v.kind != DANGO for v in out.verdicts.values()) and b.dango_id is None

def test_self_box_jumping_onto_tagged_friend_does_not_steal():
    b = board()
    for t in (0.0, 0.1):
        b.step(frame(t, tr(1, 960, self_score=0.9), tr(2, 1040, tags=(TagCand("小明", "小明", 3.0),))))
    for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        out = b.step(frame(t, tr(1, 960), tr(2, 1040, self_score=0.9, tags=(TagCand("小明", "小明", 3.0),))))
    assert b.dango_id == 1 and out.verdicts[2].kind == FRIEND_TAG

def test_pivot_separates_dango_from_hugging_friend():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.9)))
    for t in (0.1, 0.2, 0.3, 0.4):
        out = b.step(frame(t, tr(1, 960, with_bg=False), tr(2, 990, with_bg=True)))
    assert b.dango_id == 1

def test_panel_slide_keeps_dango():
    b = board()
    for t in (0.0, 0.1):
        b.step(frame(t, tr(1, 960, self_score=0.9), tr(2, 1400)))
    for t in (0.2, 0.6, 1.0, 1.4, 1.8):  # 面板横移：团子滑到 1360，另一个人滑到原来团子的位置
        out = b.step(frame(t, tr(1, 1360), tr(2, 960), sliding=True))
    assert b.dango_id == 1 and out.verdicts[2].kind != DANGO

def test_challenger_needs_dango_switch_seconds():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=0.6), tr(2, 1300)))
    for t in (0.1, 0.5, 0.9):
        b.step(frame(t, tr(1, 960, self_score=0.6), tr(2, 1300, self_score=1.0)))
    assert b.dango_id == 1
    for t in (1.3, 1.7, 2.1):
        b.step(frame(t, tr(1, 960, self_score=0.6), tr(2, 1300, self_score=1.0)))
    assert b.dango_id == 2
```

- [ ] **Step 2: 跑测试确认失败**　Expected: 新加的 FAIL
- [ ] **Step 3: 按规则实现**
- [ ] **Step 4: 跑测试确认通过**　Expected: PASS（Task 1~3 的测试也全过）
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 团子分（位置 / 长相 / self / 不跟背景走）和换人滞回"`

---

### Task 5: 只从干净的帧学（`StepOut.learn`）

**Files:**
- Modify: `src/skydango/vision/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces: `StepOut.learn`，规则照 spec §6：
  - `("friend", 名字)`：`FRIEND_TAG` 有名字、这一段稳定持有满 `tag_hold`、**这一帧**分到了这个 key、`scores` 不为 None；喊一声亮名字确认的那一帧也算（`scores` 不为 None 时）
  - `("me", "")`：`DANGO`、`smooth ≥ dango_learn`、`scores` 不为 None
  - `("stranger", sid)`：`STRANGER`、`sid` 不为 None、`scores` 不为 None
  - 其余（`FRIEND_LOOK` / `UNSURE` / `PENDING` / 接回）不学

- [ ] **Step 1: 写失败的测试**

```python
def test_learn_only_stable_tag_frames_with_samples():
    b = board()
    s = sc(小明=0.9)
    outs = [b.step(frame(t, tr(1, 1000, tags=(TagCand("小明", "小明", 3.0),), scores=s))) for t in (0.0, 0.3, 0.6)]
    assert outs[0].learn == {} and outs[1].learn == {} and outs[2].learn == {1: ("friend", "小明")}
    out = b.step(frame(0.7, tr(1, 1000, scores=s)))  # 标签被挡：不学
    assert out.learn == {}

def test_learn_dango_and_stranger():
    b = board()
    b.step(frame(0.0, tr(1, 960, self_score=1.0)))
    out = b.step(frame(0.1, tr(1, 960, self_score=1.0, scores=sc(me=0.9))))
    assert out.learn == {1: ("me", "")}
    for t in (0.0, 0.5, 1.0):
        out = b.step(frame(t, tr(5, 1500, sid="陌生人A", scores=sc(me=0.2, 小明=0.3))))
    assert out.learn[5] == ("stranger", "陌生人A")

def test_look_alike_and_unsure_never_learn():
    b = board()
    for t in (0.0, 0.1, 0.2, 0.3):
        out = b.step(frame(t, tr(1, 1000, scores=sc(小明=0.95)), tr(2, 1300, scores=sc(番茄=0.85, 小明=0.5))))
    assert out.learn == {}
```

- [ ] **Step 2~4**: 跑失败 → 实现 → 跑通过（`python -m pytest tests/test_identity.py -q`）
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 底库只从干净的帧学"`

---

### Task 6: 感知层拼 `FrameIn`（外观分、团子线索、喊一声）

**Files:**
- Modify: `src/skydango/vision/appearance.py`（`AppearanceBook.identity_scores`）
- Modify: `src/skydango/vision/perception.py`（构造参数、`_identity_frame`）
- Test: `tests/test_perception_identity.py`（新）、`tests/test_appearance.py`

**Interfaces:**
- Consumes: Task 1~5 的类型和 `IdentityBoard`
- Produces:
  - `AppearanceBook.identity_scores(sample: Sample) -> Scores`：`dino` / `color` = 每个好友底库 `Gallery.best(sample.dino, "dino")` / `Gallery.best(sample.color, "color")`（None 的不列；`sample.dino` 为 None 时 `dino` 是空字典）；`me` = 团子底库 DINOv2 最像一张，团子底库样本 < 3 或 `sample.dino` 为 None → None。加锁同 `_scores`
  - `PerceptionWatcher.__init__(..., identity: IdentityBoard | None = None)`，存 `self.identity`
  - `PerceptionWatcher._identity_frame(players: list[Track], selfs: list[Track], tags: list[Track], fresh: dict[int, tuple[Sample, Rect]], now: float, width: int, height: int, panel_visible: bool) -> FrameIn`：
    - `tags`：对每条非黑影人物轨迹，所有 `_tag_err` 不为 None 的标签 → `TagCand(key, name, err)`（`key` = `tag.data["name"]` 或 `f"#{tag.id}"`）
    - `scores`：`fresh` 里有它 → `self.appearance.identity_scores(sample)`
    - `self_score`：`selfs` 里和它 `same_body` 的 `self` 轨迹这一帧（`last == now`）的最高 `score`
    - `with_bg`：`abs(self.last_shift[0]) ≥ PIVOT_SHIFT × 轨迹框宽`（`PIVOT_SHIFT = 0.25`）时，比较这一帧 `track.drift[0]` 的增量（上一帧值存 `data["drift_prev"]`）：`abs(增量 - shift) < abs(shift) / 2` → True，`abs(增量) < abs(shift) / 2` → False，否则 None；没转 → None
    - `relink`：`data["maybe"]` 且 `data["maybe_by"] == "relink"`（`_relink` 照旧写，身份板读）
    - `sid` = `data.get("sid")`、`called_at` = `data.get("unsure_called")`、`under_panel` = `_under_panel(...)`、`unlit` = `_unlit(...)`
    - `sliding` = `now < self._panel_slide_until`；`moved` = `now < self._quiet_until`（走路 / 转圈后的静默期）
    - `call_end`：`self._call` 这一帧刚结束（`_call_tick` 把 `ended` 置 True 的那一帧）→ `CallEnd(c.at, {轨迹 id: 名字 for 窗口里挂上名字的轨迹})`；为此 `_call_tick` 在窗口里把"轨迹 id → 名字"也记进 `c.tracks: dict[int, str]`（`CallResult` / 内部结果类型加这个字段，默认空）

- [ ] **Step 1: 写失败的测试**
  - `tests/test_appearance.py::test_identity_scores`：两个好友各学 3 张（颜色 / DINOv2 用 `_basis` 造正交向量），`dino` / `color` 两份分别等于对应特征的余弦；团子底库 2 张时 `me is None`、3 张时是 DINOv2 余弦
  - `tests/test_identity.py::test_friend_feature_picks_scores`：同一组样本 `dino` 像小明、`color` 像番茄，`friend_feature="dino"` 判小明、`"color"` 判番茄、`"both"` 按平均判
  - `tests/test_perception_identity.py::test_identity_frame_collects_evidence`：用 `test_perception_appearance.make()` 的写法建 watcher（多传 `identity=IdentityBoard(...)`），一帧里放一个带标签的 `player`、一个 `self` 框压在另一个 `player` 上；直接调 `_identity_frame(...)`，断言带标签那条的 `tags[0].name == XIAOMING`、压着 self 那条 `self_score == det.score`、`scores` 在 `every = 1` 时不为 None
  - `test_identity_frame_pivot`：给 watcher 设 `last_shift = (200.0, 0.0)`，一条轨迹 `drift` 增量 200（跟背景）、一条增量 0 → `with_bg` 分别 True / False

- [ ] **Step 2: 跑测试确认失败**　Run: `python -m pytest tests/test_appearance.py tests/test_perception_identity.py -q`　Expected: FAIL
- [ ] **Step 3: 实现**（只加方法和构造参数，`process()` 还不调用）
- [ ] **Step 4: 跑测试确认通过**　Run: `python -m pytest tests/test_appearance.py tests/test_perception_identity.py tests/test_perception*.py -q`　Expected: PASS（老测试不受影响）
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 感知层把一帧的证据拼成 FrameIn"`

---

### Task 7: `process()` 接上身份板、写回字段、出错自关

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception_identity.py`

**Interfaces:**
- Consumes: Task 6 的 `_identity_frame`、`IdentityBoard.step`
- Produces:
  - `PerceptionWatcher._identity_apply(out: StepOut, players: list[Track], tags: list[Track], fresh: dict, frame, now: float, tracks: list[Track]) -> tuple[dict[int, Track], list[Track]]`：返回（`tagged` = 轨迹 id → 标签 Track，给 `_call_tick` / 圆圈用；`dangos` = 判成团子的轨迹）
  - `PerceptionWatcher.identity_log: list[FrameIn] | None`（评估时设成 `[]` 就把每帧 `FrameIn` 追加进去；平时 None）
  - 常量 `IDENTITY_MAX_ERRORS = 10`

`process()` 分流（`self.identity is not None` 时）：
1. 跳过 `_mark_dango`；`dangos = []`、`players` = 全部 admitted、非 `_other_form` 的 player / unlit 轨迹
2. 原来 808 行的 `_assign_tags` 和 862~919 行那段（摘名字、`_appearance_identify`、判陌生人）不走；改成：`fresh = _appearance_features(...)` → `fi = _identity_frame(...)` →（`identity_log` 不为 None 时追加）→ `out = self.identity.step(fi)` → `_identity_apply(...)`
3. `_identity_apply` 写字段：`FRIEND_TAG` → `tagged = True`、有名字时 `name`、`tag_at = now`（这一帧分到标签时）；`FRIEND_LOOK` → `maybe = 名字`（`maybe_by` 是 relink 的保留）；`UNSURE` → `unsure = (名字, verdict.since)`、`unsure_first` 同；`DANGO` → `data["dango"] = box`、从 `players` 挪进 `dangos`；`STRANGER` → `stranger = True` 并调原来的 `_appearance_stranger(...)`（编号 / 又回来了照旧）；每帧先清掉上一帧这几个字段里和新结论不一致的（`name` 只在失去稳定持有时清，跟着 verdict 走）；`unsure_called` / `unsure_miss` 在 verdict 离开 `UNSURE` 时清掉
4. `out.learn` 里每一项调 `self._learn(kind, who, fresh[id], frame, now)`（`me` 用 `"me", ""`）
5. 和外观对不上：上一帧 `FRIEND_LOOK` 是 A、这一帧 `FRIEND_TAG` 是 B → `self._report(frame, now, f"按外观认成 {A}，名字标签是 {B}", tracks)`，原因写 `identity_conflict`（`_report_hard` 走 hardcases）
6. 每条轨迹 verdict 的 kind / name 变了记 DEBUG：`"轨迹 %d：%s %s → %s %s"`
7. `self_` 轨迹（cls `self`）：和某条非团子、`FRIEND_TAG` 的轨迹 `same_body` 的从 `selfs` 里去掉（不当团子上报、不学）
8. `step` 抛异常：`log.exception`，这一帧照上一帧的 `data` 不动、`_identity_errors += 1`；连续 `IDENTITY_MAX_ERRORS` 次 → `log.error("身份板连续出错，退回旧流程")`、`self.identity = None`、`self.identity_off = "连续出错"`（之后走旧流程）；成功一帧清零
9. `enabled` 关 / `identity is None`：所有改动都不生效（逐字旧流程）

- [ ] **Step 1: 写失败的测试**（`tests/test_perception_identity.py`，用 `make(identity=True)` 包装：`IdentityBoard(IdentityConfig(enabled=True), acfg, stranger_after=1.0, grace=1.5, stranger_min_height=0.0, track_buffer=..., call_window=6.0)`）

```python
def test_flipping_tag_between_crowded_friends_makes_no_stranger():
    w, det = make(identity=True)
    a, b = player(1000), player(1060)
    for i in range(40):  # 标签在 a、b 头上来回跳
        t = round(i * 0.1, 3)
        tg = tag(1000 if i % 2 else 1060, 110)
        run(w, det, [(a, PINK), (b, GREEN), (tg, None)], t, t)
    assert w.strangers(t) == 0

def test_dango_found_without_self_box():
    w, det = make(identity=True)
    me = player(915)
    run(w, det, [(me, WHITE), (Detection("self", me.box, 0.9), None)], 0.0, 0.3)
    t = run(w, det, [(me, WHITE), (player(1400), GREEN)], 0.4, 3.0)
    (d,) = [p for p in w.last_tracks if p.data.get("dango")]
    assert d.box.x == 915 and w.strangers(t) == 1

def test_legacy_path_unchanged_when_identity_off():
    w, det = make()
    assert w.identity is None  # make() 默认不挂身份板：test_perception_appearance 全部照旧

def test_board_errors_fall_back_to_legacy(monkeypatch):
    w, det = make(identity=True)
    monkeypatch.setattr(w.identity, "step", lambda fi: (_ for _ in ()).throw(RuntimeError("坏了")))
    run(w, det, [(player(1000), PINK)], 0.0, 1.2)
    assert w.identity is None and w.identity_off == "连续出错"

def test_identity_log_records_frames():
    w, det = make(identity=True)
    w.identity_log = []
    run(w, det, [(player(1000), PINK)], 0.0, 0.3)
    assert len(w.identity_log) == 4 and all(isinstance(f, FrameIn) for f in w.identity_log)
```

- [ ] **Step 2: 跑测试确认失败**　Run: `python -m pytest tests/test_perception_identity.py -q`　Expected: FAIL
- [ ] **Step 3: 实现上面 1~9**
- [ ] **Step 4: 跑全量测试**（放后台跑，跑完再看）　Run: `python -m pytest -q`　Expected: PASS，老测试零改动
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): 感知层接上身份板，关着逐字旧流程，出错自动退回"`

---

### Task 8: 命令行 / 管理面板接上开关

**Files:**
- Modify: `src/skydango/cli.py`（`_perception()` 和真机 `run` 建感知层的地方——都经 `_perception` / `_appearance_parts`，先 grep `PerceptionWatcher(` 找全）
- Modify: `src/skydango/console/settings.py`
- Test: `tests/test_cli_identity.py`（新）或并进已有的 cli 测试文件

**Interfaces:**
- Produces: `cli._identity_board(cfg: Config, dino) -> tuple[IdentityBoard | None, str]`：第二项是启动日志那一行

规则：`cfg.identity.enabled` 且 `cfg.perception.enabled` 且 `cfg.appearance.enabled` 且 `dino` 加载成功 → 建 `IdentityBoard(cfg.identity, cfg.appearance, stranger_after=cfg.perception.stranger_after, grace=STRANGER_GRACE, stranger_min_height=cfg.perception.stranger_min_height, track_buffer=cfg.perception.track_buffer, call_window=cfg.call.window)`，日志「身份板：外观为主、名字标签证实（好友特征 dino）」；
否则 None，日志说明原因（「身份板没开：要先开认装扮」/「…DINOv2 没加载」），`enabled = false` 时不打。
设置页加 `Field("identity.enabled", "身份板", "认人改成外观为主、名字标签证实（要配合认装扮和 DINOv2）；关掉 = 原来的认法", "bool", "features")`，放在 `appearance.enabled` 后面。

- [ ] **Step 1: 写失败的测试**：`_identity_board` 四种情况（全开 → 有板子、日志含「外观为主」；没开认装扮 → None + 「要先开认装扮」；dino None → None + 「DINOv2 没加载」；enabled false → None + 空串）；settings 清单里有 `identity.enabled`
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**，把板子传进 `PerceptionWatcher(identity=...)`
- [ ] **Step 4: 跑测试**　Run: `python -m pytest -q`（后台）　Expected: PASS
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): [identity] 开关接进 run 和管理面板设置页"`

---

### Task 9: 标准答案：`perception identity-truth`

**Files:**
- Create: `src/skydango/vision/identity_eval.py`
- Modify: `src/skydango/cli.py`（`psub.add_parser("identity-truth", …)`，接在 `track-eval` 旁边）
- Test: `tests/test_identity_eval.py`

**Interfaces:**
- Produces:
  - `draft_identities(tracks_seen: dict[int, TrackLog], tag_hold: float) -> dict[int, str]`：`TrackLog` = 一条轨迹在回放里的记录（`frames: list[(帧文件名, Rect)]`、`names: list[(时间, 名字)]` = 这一帧分到的读出名字的标签、`self_hits: int`）；身份 = 连续 ≥ `tag_hold` 秒挂着的名字里次数最多的；没有名字但 `self_hits` ≥ 帧数一半 → `"团子"`；否则 `"?"`
  - `write_draft(path: Path, ids: dict[int, str], logs: dict[int, TrackLog]) -> None`：TOML，每行 `t12 = "饶总"  # 出现 12.3~40.1 s，85 帧`
  - `write_sheet(path: Path, logs, frames_dir: Path, per_track: int = 4) -> None`：每条轨迹一格（编号 + 均匀挑 4 张裁图），OpenCV 拼图
  - `apply_draft(draft: Path, logs_path: Path, out: Path) -> int`：读改好的草稿（取值只能是好友名 / `团子` / `陌生人` / `不要` / `?`；别的值报错并给行号），按 `logs.json`（回放时存的 `TrackLog`）展开成 `truth.jsonl` 每行 `{"frame": 文件名, "box": [x, y, w, h], "who": 身份}`（`不要` / `?` 不写），返回写了几行
- CLI：`perception identity-truth <录像目录> [--model 模型] [--fps 6.5]` → `tmp/identity-truth/<录像名>/`（`truth-draft.toml`、`sheet.jpg`、`logs.json`），回放用 `_perception(cfg, args)` 建 watcher（`[identity]` 关，只要追踪和读名字）、`subsample` / `timed_files` / `_panel_open` 同 `track-eval`；
  `perception identity-truth <录像目录> --apply` → 读同目录的草稿和 `logs.json`，写 `<录像目录>/truth.jsonl`

- [ ] **Step 1: 写失败的测试**：`draft_identities` 三种（名字挂够 / 只闪一帧不算 / self 多的是团子）；`apply_draft` 来回（写草稿 → 改一行 → 展开 → 行数和身份对）；草稿里写 `"老登"`（不是好友也不是保留字）报错带行号——好友名单由调用方传入 `friends: list[str]`
- [ ] **Step 2: 跑测试确认失败**　Run: `python -m pytest tests/test_identity_eval.py -q`
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；再在一段旧录像上冒烟：`python -m skydango perception identity-truth tmp/record/walkaway-1002-1 --fps 6.5`（主目录的录像，worktree 里用绝对路径），打开 `sheet.jpg` 看一眼
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): identity-truth 半自动标准答案（草稿、拼图、--apply）"`

---

### Task 10: 新旧对比和扫门槛：`perception identity-eval`

**Files:**
- Modify: `src/skydango/vision/identity_eval.py`
- Modify: `src/skydango/cli.py`
- Test: `tests/test_identity_eval.py`

**Interfaces:**
- Consumes: Task 7 的 `identity_log`、Task 9 的 `truth.jsonl`
- Produces:
  - `label_of(track: Track) -> str`：一条轨迹在这一帧被判成什么：`data["dango"]` → `"团子"`；`name` → 名字；`maybe` → 名字；`unsure` → `"拿不准"`；`stranger` → `"陌生人"`；否则 `"未定"`
  - `score_frames(pred: dict[str, list[tuple[Rect, str]]], truth: dict[str, list[tuple[Rect, str]]], iou_min: float = 0.5) -> dict`：按帧、按 IoU 贪心配对，返回 `friend_as_stranger`（好友框被判陌生人的比例）、`dango_found`（团子框判成团子的比例）、`other_as_dango`（非团子框判成团子的比例）、`friend_right` / `friend_wrong`（判成正确 / 错误好友名的比例）、`unsure_rate`、`n`（各类框数）
  - `replay_board(frames: list[FrameIn], cfg: IdentityConfig, acfg: AppearanceConfig, **board_kw) -> dict[str, list[tuple[Rect, str]]]`：只用身份板重放，返回每帧的预测（需要 `FrameIn` 里带帧文件名：给 `FrameIn` 加可选字段 `tag: str = ""`，perception 不填，评估填）
  - `sweep(frames, truth, base_cfg, acfg, grid: dict[str, list]) -> list[dict]`：网格扫 `match` / `unsure` / `friend_feature`（`Scores` 里 DINOv2 / 颜色两份都有，换 `friend_feature` 只重放身份板）、`dango_*` 权重、`tag_hold`，返回按"三条目标都过、`unsure_rate` 最低"排序的组合
  - `report_md(base: dict, cur: dict, sweep: list[dict], meta: dict) -> str`
- CLI：`perception identity-eval <录像目录> [--model 模型] [--fps 6.5] [-o 目录]`：录像目录里没有 `truth.jsonl` → 报错提示先跑 `identity-truth`；回放两遍（旧流程 / 新流程，其余配置相同，新流程的 watcher 设 `identity_log = []`）；写 `tmp/identity-eval/<时间>/report.md`、`summary.json`、`frames.pkl`；喊一声离线没有身体：报告里 `unsure` 按"喊了都亮"和"喊了都不亮"两种情况各算一次（`replay_board` 在 `UNSURE` 满 `unsure_wait` 时注入 `CallEnd`）

- [ ] **Step 1: 写失败的测试**：`score_frames` 手写两帧答案（好友被判陌生人 1/2、团子认出 1/1、别人判团子 0）；`replay_board` 对同一份 `frames` 跑两次结果相同；`sweep` 在一个"`match` 低一点就能过"的合成例子上把那组排第一；`label_of` 各分支
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试确认通过**；冒烟：在 Task 9 冒烟用的录像上手写一份很小的 `truth.jsonl`（或用 `--apply` 的结果），跑 `identity-eval`，打开 `report.md` 看表格齐全
- [ ] **Step 5: 提交**　`git commit -am "feat(identity): identity-eval 新旧对比、按框对答案、只重放身份板扫门槛"`

---

### Task 11: 文档

**Files:**
- Modify: `CLAUDE.md`（「代码结构」表加 `vision/identity.py` / `identity_eval.py` 两行；新增「身份板（`[identity]`，要配合 `[perception]` + `[appearance]`）」一节：一段起因、规则要点（照 spec §3~§6 各一两行）、开关、评估命令、**代码默认关、还没在录像 / 真机上验证、数字是估的**；「常用命令」加 `identity-truth` / `identity-eval` 两行；「认装扮」一节开头加一句"开了 `[identity]` 时，maybe / unsure / 看着像团子由身份板判"）
- Create: `docs/progress/2026-10-22-identity.md`（10-22 晚的诊断数字、spec / 计划链接、用户要做的：录拥挤录像 → `identity-truth` → 改草稿 → `--apply` → `identity-eval` → 过线后 `console.toml` 打开 → 真机一晚，照 spec §11 四步列成清单）

- [ ] **Step 1: 写文档**
- [ ] **Step 2: 跑全量测试**（后台）　Run: `python -m pytest -q`　Expected: PASS
- [ ] **Step 3: 提交**　`git commit -am "docs: 身份板的用法、评估步骤和待验证清单"`
