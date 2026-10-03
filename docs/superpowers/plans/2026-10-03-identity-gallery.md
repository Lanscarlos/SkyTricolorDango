# 认人底库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 认装扮从"每人一个颜色平均特征"改成"每个身份一组多张样本（颜色 + DINOv2）"：DINOv2 判"看着像团子"，颜色分三档（像 / 可能是 / 不像），"可能是"自动按 Q 喊一声确认，启动时转一圈登记团子。

**Architecture:** 纯数据的 `vision/gallery.py`（样本、底库）挂进 `AppearanceBook` 的每个 `Profile`；感知层每次裁图出一个 `Sample`（两种特征），按"身份确定才学"的规矩进底库，没名字的轨迹按底库认；拿不准的轨迹通过 `env.unsure()` 交给身体现有的自动喊；登记走 `Body.enroll_self()` → `Camera.spin` → `env.sweep(..., enroll=True)`。

**Tech Stack:** Python 3.13、numpy、OpenCV、onnxruntime（DINOv2-small ONNX）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-03-identity-gallery-design.md`

## Global Constraints

- `[appearance] enabled = false` 时逐字照旧（默认仍然关）
- 默认值：`min_height` 0.13、`match` 0.88、`unsure` 0.83、`unsure_wait` 2.0、`dino` `"models/dinov2-small.onnx"`、`dango_match` 0.80、`gallery_max` 40、`enroll_max` 16；`margin` 0.05 等不变；`card_match` 留在配置里不报错、不再使用
- 常量：`DUP = 0.97`（gallery.py）、`DANGO_LOOK_HOLD = 3.0`（perception.py）、DINOv2 连续出错 10 次自关
- 好友只从**这一帧**挂着名字标签的样本学；maybe / unsure / 接回 / dango_look 都不学；不存负样本
- 名字标签永远说了算：挂上标签立刻清掉 maybe / unsure / dango_look
- 外观认出只算疑似：不发 arrive / return、不打招呼
- 自动喊的额度、间隔、拦截全部共用 `[call]` 现有的（`min_gap`、`auto_window`、`auto_quota`、`auto_again`、`_auto_call_blocked`）；dry-run 不按键
- 测试用 `python -m pytest -q`（没有 pytest 的机器用 `.venv\Scripts\python.exe -m pytest -q`）；全量测试放后台跑
- 代码注释、日志、status 文字用中文，风格同周围代码

## Review Focus

1. **团子身上的 `data["dango"]` 轨迹不在 `players` 里**（`process()` 第 682 行过滤掉了）：学团子样本要单独把它们传进去，否则团子底库只有 `self` 框的样本 —— Task 4 测试 `test_dango_marked_player_track_feeds_dango_gallery`
2. **好友这一次上线还没被标签认过**（底库空）：不能因为"没有候选"就把"可能是"卡住，也不能让 unsure 挡住判陌生人 —— Task 4 测试 `test_no_friend_gallery_goes_straight_to_stranger`
3. **喊完窗口里他只亮了一下、标签挂到了别的轨迹上**（两人挨着）：只按"他自己身上挂没挂上"判，别把别人的确认算到他头上 —— Task 5 测试 `test_call_outcome_tag_on_other_track_is_miss`
4. **转圈时团子样本全被聊天面板 / 别人挡住**：不能把挡住团子的人登记成团子 —— Task 7 测试 `test_enroll_skips_blocked_self_boxes`
5. **DINOv2 推理中途坏掉**：只关掉 DINOv2 这一路，颜色认人、名字标签照常 —— Task 3 测试 `test_dino_errors_disable_only_dino`

---

### Task 1: 配置项和 `Gallery`

**Files:**
- Create: `src/skydango/vision/gallery.py`
- Modify: `src/skydango/config.py`（`AppearanceConfig`）
- Test: `tests/test_gallery.py`、`tests/test_appearance.py::test_config_has_appearance_defaults`

**Interfaces:**
- Produces:
  - `@dataclass Sample(color: np.ndarray, dino: np.ndarray | None, t: float, h: float, pinned: bool = False)`（特征都是单位向量）
  - `class Gallery(max_size: int)`：`add(s: Sample) -> None`、`best(feat: np.ndarray | None, which: str) -> float | None`（`which` ∈ `"color"` / `"dino"`）、`__len__`、`pinned_count -> int`、`samples -> list[Sample]`（只读副本）
  - `DUP = 0.97`
  - `AppearanceConfig` 新字段：`unsure: float = 0.83`、`unsure_wait: float = 2.0`、`dino: str = "models/dinov2-small.onnx"`、`dango_match: float = 0.80`、`gallery_max: int = 40`、`enroll_max: int = 16`；改默认 `min_height = 0.13`、`match = 0.88`；`card_match` 注释改成"废弃，不再使用"

- [ ] **Step 1: 写失败的测试** `tests/test_gallery.py`（特征用手写单位向量，同 `test_appearance.py` 的 `_basis` / `near`）：
  - `test_add_dedups_and_refreshes_time`：加 `color=V`、t=0 再加 `near(V, 0.99)`、t=5 → `len == 1` 且那张 `t == 5`
  - `test_best_is_max_over_samples`：加 V_PINK、V_BLUE；`best(V_BLUE, "color") == pytest.approx(1.0)`；`best(near(V_PINK, 0.9), "color") == pytest.approx(0.9, abs=1e-3)`
  - `test_best_dino_none_when_missing`：样本 `dino=None` → `best(x, "dino") is None`；空底库 `best(x, "color") is None`；`best(None, "dino") is None`
  - `test_evicts_most_redundant_unpinned`：`max_size=3`，加 A、`near(A, 0.95)`、B、C → 剩 3 张，被挤的是 A 和 `near(A,0.95)` 里**旧的那张**（A）
  - `test_pinned_never_evicted`：`max_size=2`，A（pinned）、B、C → A 还在，`pinned_count == 1`
- [ ] **Step 2: 跑测试确认失败** `python -m pytest tests/test_gallery.py -q` → ImportError
- [ ] **Step 3: 实现 `gallery.py`**：去重只看颜色特征；挤人：在未钉住的样本里算每张和其余所有样本的最大余弦，最大的那张挤掉，相同时挤 `t` 小的；全都钉住时不挤（允许超出）
- [ ] **Step 4: 改配置默认值**，更新 `test_config_has_appearance_defaults` 断言为 `(False, "color", 0.88, 0.92, 0.05, 0.40)` 并加一行断言新字段 `(c.min_height, c.unsure, c.unsure_wait, c.dino, c.dango_match, c.gallery_max, c.enroll_max) == (0.13, 0.83, 2.0, "models/dinov2-small.onnx", 0.80, 40, 16)`
- [ ] **Step 5: 跑测试** `python -m pytest tests/test_gallery.py tests/test_appearance.py -q` → 全过
- [ ] **Step 6: 提交** `feat(appearance): 多张样本的底库 Gallery 和新配置项`

### Task 2: `AppearanceBook` 改用底库

**Files:**
- Modify: `src/skydango/vision/appearance.py`（`Profile`、`AppearanceBook`）
- Test: `tests/test_appearance.py`

**Interfaces:**
- Consumes: Task 1 的 `Sample`、`Gallery`
- Produces（签名变化，老调用方传颜色特征照样能用）：
  - `Profile.gallery: Gallery`
  - `learn(kind, who, feat, now, crop=None, *, dino=None, h=0.0, pinned=False) -> Profile`：`feat` 是颜色特征；照旧更新滑动平均 `feat`，另外 `gallery.add(Sample(feat, dino, now, h, pinned))`
  - `friend_scores(feat) -> list[tuple[str, float]]`：这次上线学到的好友，按颜色底库 `best` 从高到低
  - `assign_friends(cands: dict[int, np.ndarray], exclude: set[str]) -> dict[int, str]`：语义不变（≥ `match` 且领先 `margin`、一个名字一条），打分换成底库
  - `unsure_friends(cands: dict[int, np.ndarray], exclude: set[str]) -> dict[int, str]`：最高分落在 `[unsure, match)` 的（一个名字只给分数最高那条；排除已被 `assign_friends` 拿走的名字由调用方传 `exclude`）
  - `still_like(track_feat, name) -> bool`：他的底库 `best ≥ match`；不再看卡
  - `best_friend(feat) -> tuple[str | None, float]`：底库打分
  - `stranger_id(feat, now, exclude)`：打分换成各编号底库的 `best`；新编号时 `gallery.add` 第一张
  - `looks_like_dango(dino: np.ndarray | None) -> bool`：团子底库 `best(dino, "dino") ≥ dango_match` 且严格大于每个好友底库的 `best(dino, "dino")`（没有分的好友不算）；任一为空 / None → False
  - `load_cards` 照旧读描述和 `card_feats`（`card_state` / 换装还用），`_candidates` 不再加卡里的特征

- [ ] **Step 1: 改 / 加测试**：
  - 删 `test_card_only_friend_needs_stricter_threshold`，换成 `test_card_feature_never_assigns`：只 `load_cards` 了小明 → `assign_friends({1: V_PINK}, set()) == {}`、`still_like(V_PINK, "小明") is False`
  - `test_load_cards_takes_last_outfit` 去掉 `assign_friends` 那行，只留描述断言
  - `test_still_like_prefers_learned_over_card` 改名 `test_still_like_uses_gallery`：学过 V_PINK 和 V_BLUE 两张 → `still_like(V_BLUE)` 和 `still_like(V_PINK)` 都为真（多张样本，不再被平均冲掉）
  - `test_gallery_keeps_two_views_of_one_friend`：小明学 V_PINK、V_BLUE（两个角度）→ `assign_friends({1: V_BLUE}, set()) == {1: "小明"}`（平均特征时这会失败）
  - `test_unsure_band`：`match=0.88, unsure=0.83`，学 V_PINK → `unsure_friends({1: near(V_PINK, 0.85)}, set()) == {1: "小明"}`，`assign_friends` 同输入为 `{}`；`near(V_PINK, 0.80)` 两边都空
  - `test_looks_like_dango`：`learn("me", "", V_WHITE, 0, dino=V_PINK)`、`learn("friend", "小明", V_BLUE, 0, dino=V_BLUE)`；`dango_match=0.80` → `looks_like_dango(near(V_PINK, 0.9))` 真；`looks_like_dango(near(V_PINK, 0.75))` 假；`looks_like_dango(None)` 假；小明的 dino 也学一张 `near(V_PINK, 0.95)` 后 `looks_like_dango(near(V_PINK,0.9))` —— 比较两边 best，按实际值断言（团子更像才真）
  - `test_pinned_me_samples_survive`：`gallery_max=2`，`learn("me", …, pinned=True)` 一张、再普通学 3 张 → `me.gallery.pinned_count == 1`
- [ ] **Step 2: 跑测试确认失败** `python -m pytest tests/test_appearance.py -q`
- [ ] **Step 3: 实现**（`Profile` 加 `gallery: Gallery = field(...)`，`gallery_max` 从 cfg 传；陌生人 `stranger_id` 新建 `Profile` 时同样带底库）
- [ ] **Step 4: 跑测试** `python -m pytest tests/test_appearance.py tests/test_appearance_eval.py -q` → 全过
- [ ] **Step 5: 提交** `feat(appearance): 记忆簿按多张样本认人，关系卡旧特征不再拿来认人`

### Task 3: 一次裁图出两个特征

**Files:**
- Modify: `src/skydango/vision/appearance.py`
- Modify: `src/skydango/vision/perception.py`（构造参数、`_appearance_features`）
- Test: `tests/test_appearance.py`、`tests/test_perception_appearance.py`

**Interfaces:**
- Consumes: `attrs.crop(img, box, pad, size)`、`good_crop`、`OnnxEmbedder.embed`
- Produces:
  - `class DinoGuard(embedder, max_errors: int = 10)`：`embed(img) -> np.ndarray | None`（出错返回 None 并计数；连续 `max_errors` 次后 `enabled = False`、记 WARNING、以后一律 None）、`size` 属性转给 embedder
  - `make_sample(frame, box, others, blocked, cfg: AppearanceConfig, color_embedder, dino: DinoGuard | None, now: float) -> Sample | None`：`good_crop` 失败返回 None；颜色用 `good_crop` 的裁图；DINOv2 用 `attrs.crop(frame, box, 0.0, dino.size)`；`h = box.h / frame高`
  - `PerceptionWatcher(..., dino=None)`（`DinoGuard | None`）
  - `_appearance_features(...)` 返回 `dict[int, tuple[Sample, Rect]]`；照旧更新 `data["feat"]`（颜色 EMA）和 `data["samples"]`；新增 `data["sample"]`
  - 参加算特征的轨迹：`players`（不含黑影）+ `selfs` + **新参数 `dangos`**（`data["dango"]` 的人物轨迹，见 Review Focus 1）

- [ ] **Step 1: 写失败的测试**：
  - `test_make_sample_both_features`：假 dino（固定返回 `_basis(3)`）→ `Sample.dino` 是它、`Sample.color` 和 `ColorEmbedder().embed(good_crop(...))` 一致、`h` 对
  - `test_make_sample_without_dino`：`dino=None` → `Sample.dino is None`
  - `test_dino_errors_disable_only_dino`（Review Focus 5）：dino 每次抛异常，`max_errors=3` → 前 3 次返回 None、之后 `enabled is False`；`make_sample` 仍给出颜色特征
  - perception：`test_features_store_sample_on_track`（`make()` 加 `dino=DinoGuard(假)` 参数）→ 处理几帧后好友轨迹 `data["sample"].dino is not None`
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现** `DinoGuard`、`make_sample`；`_appearance_features` 改成调用 `make_sample`；`process()` 把 `data["dango"]` 的轨迹列表传进去（在第 682 行过滤 `players` 之前从 `tracks` 取）
- [ ] **Step 4: 跑** `python -m pytest tests/test_appearance.py tests/test_perception_appearance.py -q` → 全过（老测试的 `make()` 默认 `dino=None`，行为同颜色）
- [ ] **Step 5: 提交** `feat(appearance): 一次裁图出颜色和 DINOv2 两个特征`

### Task 4: 学和认（像团子 / 像 / 可能是）

**Files:**
- Modify: `src/skydango/vision/perception.py`（`_appearance_identify`、`_looks_checked`、`process()` 判陌生人循环、`_count_unnamed`、`unnamed`、`people()`、`nearest()`、viewer 的框、`_catalog_who`）
- Modify: `src/skydango/vision/people.py`（`Person.unsure`、`_describe_person`）
- Modify: `src/skydango/vision/static/stage.js`（`maybe` 旁边加 `unsure`、`dango_look` 两种颜色 / 虚线 / 图例名）
- Test: `tests/test_perception_appearance.py`、`tests/test_people.py`（没有就加进 `test_perception_appearance.py`）

**Interfaces:**
- Consumes: Task 2 的 `learn(..., dino=, h=)`、`assign_friends`、`unsure_friends`、`looks_like_dango`；Task 3 的 `data["sample"]`、fresh 字典
- Produces（轨迹 `data` 键，后面任务用）：
  - `data["dango_look"] = now`（最近一次判像团子的时间；`DANGO_LOOK_HOLD = 3.0` 秒内有效）
  - `data["unsure"] = (名字, 开始时间)`
  - `data["unsure_miss"] = True`（以后不再进 unsure）
  - `PerceptionWatcher._dangoish(t, now) -> bool`：`data["dango"]` 或 dango_look 在有效期内
  - `Person.unsure: bool = False`；`_describe_person` 对 `kind == "friend" and p.unsure` 输出 `可能是{name}（没看到名字，{where}）`

学的规矩（spec §3.1）：`self` 框和 `dangos` → `learn("me", "", …)`；`tagged` 里这一帧有名字 → `learn("friend", name, …)`；有 `sid` 的陌生人照旧；其余不学。

认的顺序（spec §3.2）：没名字、没 `tagged`、不是 `_dangoish`、不是黑影、不是 `unsure_miss` 的轨迹，好样本 ≥ `min_samples`，**拿这一次的 `data["sample"]` 比**（`assign_friends` / `unsure_friends` / `still_like` 传 `sample.color`，不再传滑动平均 `data["feat"]`；这一帧没算出新样本的轨迹不重新认，保持上一帧的结论）：
1. `looks_like_dango(sample.dino)` → `dango_look = now`，pop `maybe` / `unsure`
2. 否则 `assign_friends` → `maybe`；剩下的 `unsure_friends`（exclude 加上刚分出去的名字和 `shown`）→ 没有 unsure 的设 `unsure = (名字, now)`，已有同名 unsure 的保留开始时间；不再在 unsure 档的 pop
3. 挂上标签：pop `maybe`、`unsure`、`dango_look`（同现在 maybe 的写法）；`unsure` 的名字在别处亮着 → pop

后果：
- 判陌生人（`process()` 循环）：`_dangoish` 的轨迹跳过（不计 stranger）；`_looks_checked` 加条件：有 `unsure` 且 `now - 开始 < unsure_wait + call_window` 时返回 False；到时 pop `unsure` 并记 `unsure_miss`
- `_count_unnamed` / `unnamed()`：不算 `_dangoish` 和带 `unsure` 的
- `people()`：`_dangoish` 跳过；`unsure` → `Person(kind="friend", name=名字, sure=False, unsure=True)`
- viewer 框：`unsure` → kind `"unsure"`、label `可能是{名字}?`；dango_look → kind `"dango_look"`、label `像团子`
- `_catalog_who`：`_dangoish` 返回 `Who("团子", "self", True)`

- [ ] **Step 1: 写失败的测试**（`make(acfg_kw=...)` 里给 `match=0.88, unsure=0.83, min_samples=3, every=1`，dino 用按颜色查表的假 DinoGuard：粉 → `_basis(0)`、团子色 → `_basis(4)`）：
  - `test_learns_friend_only_on_tagged_frame`：前 1 秒挂标签、之后标签消失 2 秒 → 小明底库张数在标签消失后不再增加
  - `test_dango_marked_player_track_feeds_dango_gallery`（Review Focus 1）：只有 `self` 高分框 + 同位置的 player 框；之后只剩 player 框（`_mark_dango` 认的）→ 团子底库张数还在涨
  - `test_dango_look_blocks_maybe_and_stranger`：团子色的 player 框出现在画面别处（不在 self 位置）→ 3 秒后 `data["dango_look"]` 有、`data.get("stranger")` 假、`people()` 里没有他
  - `test_dango_look_expires`：判像团子后换成白色（陌生人色）→ `DANGO_LOOK_HOLD` 秒后可以判陌生人
  - `test_unsure_holds_stranger_then_gives_up`：小明学过粉色，来一个余弦落在 0.83~0.88 的颜色（把粉色和白色按比例混）的人 → `unsure_wait + call_window` 秒内不判陌生人、`people()` 里是 `unsure=True`；超时后 `unsure_miss` 为真、判成陌生人
  - `test_no_friend_gallery_goes_straight_to_stranger`（Review Focus 2）：这次上线没见过任何好友 → 白色的人到 `stranger_after` 就判陌生人，从没有 `unsure`
  - `test_tag_clears_unsure_and_dango_look`：unsure 的轨迹挂上小明标签 → `unsure` / `dango_look` 都没了，`name == 小明`
  - `test_describe_unsure_person`：`describe_people([Person(..., kind="friend", name="小明", sure=False, unsure=True, ...)])` 含 `可能是小明（没看到名字`
  - `test_unnamed_skips_dango_look_and_unsure`
- [ ] **Step 2: 跑确认失败** `python -m pytest tests/test_perception_appearance.py -q`
- [ ] **Step 3: 实现**（上面列的每处）
- [ ] **Step 4: 跑** `python -m pytest tests/test_perception_appearance.py tests/test_perception_dango.py tests/test_perception_catalog.py tests/test_perception.py -q` → 全过
- [ ] **Step 5: 提交** `feat(appearance): 看着像团子、像小明、可能是小明三档`

### Task 5: 感知层的"可能是"接上呼喊窗口

**Files:**
- Modify: `src/skydango/vision/perception.py`（`unsure()`、`mark_unsure_called()`、`_call_tick` 收尾）
- Test: `tests/test_perception_appearance.py`

**Interfaces:**
- Consumes: Task 4 的 `data["unsure"]`、`data["unsure_miss"]`
- Produces:
  - `unsure(now) -> list[tuple[int, str, float]]`：(轨迹 id, 名字, 开始时间)；只列 unsure 已持续 ≥ `unsure_wait`、没有 `unsure_called`、这个名字此刻没挂在任何轨迹（`name`）上、`self.labels` 里这个名字的标签不在 `PEOPLE_STALE` 内的；最近一帧的轨迹（同 `unnamed` 的取法），加锁读
  - `mark_unsure_called(ids: list[int], at: float) -> None`：给这些轨迹打 `data["unsure_called"] = at`
  - 窗口结束那一帧（`_call_tick` 里 `c.ended = True` 之后）：每条 `unsure_called` 落在这个窗口（`at` 相同）的轨迹：没挂上名字的 → pop `unsure`、`unsure_miss = True`、记 INFO "喊了一声，轨迹 N 头上没亮名字"；挂上了的已经由正常流程处理

- [ ] **Step 1: 写失败的测试**：
  - `test_unsure_listed_after_wait`：unsure 持续 1 秒时 `unsure(now) == []`，2 秒后列出 `[(tid, 小明, t0)]`
  - `test_unsure_not_listed_when_friend_named_elsewhere`：小明的标签挂在另一条轨迹上 → 不列
  - `test_call_outcome_tag_on_track_confirms`：`mark_unsure_called([tid], at)`、`called(at)`，窗口里他头上挂上小明标签 → `name == 小明`、没有 `unsure_miss`、小明底库多了一张
  - `test_call_outcome_no_tag_is_miss`：窗口里什么都没亮 → 窗口结束后 `unsure_miss`、之后 `unsure(now) == []`
  - `test_call_outcome_tag_on_other_track_is_miss`（Review Focus 3）：小明的标签在窗口里挂到旁边另一条轨迹上 → 他 `unsure_miss`、另一条是小明
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `python -m pytest tests/test_perception_appearance.py tests/test_brain_call_tool.py -q` → 全过
- [ ] **Step 5: 提交** `feat(appearance): 可能是小明的人交给呼喊窗口确认`

### Task 6: 身体自动喊加"拿不准"起因

**Files:**
- Modify: `src/skydango/brain/body.py`（`_watch_call`）
- Test: `tests/test_brain_call_body.py`

**Interfaces:**
- Consumes: Task 5 的 `env.unsure(now)`、`env.mark_unsure_called(ids, at)`；现有 `call_out(reason)`、`_auto_call_blocked`、`_call_times`、`_auto_found`
- Produces: `call_out("unsure")`（`last_call.reason == "unsure"`）

逻辑：`who`（刚走开的好友）为空时，取 `env.unsure(now)`（`hasattr(self.env, "unsure")` 为假时当空，同现在对 `unnamed` 的写法；`EnvWatcher` 不加这两个方法），去掉 `now - _auto_found[名字] <= auto_again` 的；有剩下的就过同一套间隔 / 额度 / 拦截检查（**不看** `env.unnamed`），先 `mark_unsure_called(ids, now)` 再 `call_out("unsure")`，`_call_times.append(now)`；被拦或 dry-run 照样已经打了 `unsure_called`；日志 "拿不准 X 是不是小明：自动喊一声"。喊完的收尾（`_collect_auto_call`、背景事件 `call`、`_auto_found`）和 auto 共用。

- [ ] **Step 1: 写失败的测试**（`CallEnv` 加 `unsure_list`、`marked`）：
  - `test_auto_call_for_unsure_person`：`env.unsure_list = [(5, "小明", 0.0)]`、没人走开 → 按了 Q、`reason == "unsure"`、`env.marked == [([5], at)]`
  - `test_unsure_call_shares_quota_and_gap`：先一次 auto，再在 `min_gap` 内有 unsure → 不按；过了 `min_gap` 再按；`auto_quota` 用完后不按
  - `test_unsure_call_skipped_when_busy`：输入框开着（`sender.opened`）→ 不按、但 `marked` 已记
  - `test_unsure_call_dry_run_logs_only`
  - `test_unsure_not_called_for_recently_found_friend`：`_auto_found["小明"] = now - 10` → 不按
- [ ] **Step 2: 跑确认失败** `python -m pytest tests/test_brain_call_body.py -q`
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `python -m pytest tests/test_brain_call_body.py tests/test_brain_call_again.py tests/test_brain_call_tool.py -q` → 全过
- [ ] **Step 5: 提交** `feat(call): 拿不准是不是好友时自动喊一声`

### Task 7: 启动时转一圈登记团子

**Files:**
- Modify: `src/skydango/vision/perception.py`（`sweep(frames, spin, enroll=False)`）
- Modify: `src/skydango/vision/sweep.py`（`SweepResult.enrolled: int = 0`）
- Modify: `src/skydango/brain/body.py`（`enroll_self()`、`run()` 进主循环前调一次、status 一行）
- Test: `tests/test_perception_appearance.py`、`tests/test_brain_appearance_body.py`

**Interfaces:**
- Consumes: Task 3 的 `make_sample`、Task 2 的 `learn(..., pinned=True)`；`find_self(...).per_frame`；`Camera.spin`
- Produces:
  - `PerceptionWatcher.sweep(frames, spin, enroll: bool = False) -> SweepResult`：`enroll` 且挂了记忆簿时，每帧团子框 = 这一帧分数最高的 `self` 检测，没有就用 `found.per_frame.get(fi)`；`make_sample`（others = 这一帧别的人物框，blocked = 空）；有样本的帧按时间均匀挑最多 `enroll_max` 张，`learn("me", "", s.color, now, dino=s.dino, h=s.h, pinned=True)`；`result.enrolled = 张数`；裁图存到 `self.enroll_dir`（构造参数 `enroll_dir: Path | None = None`，None 不存），文件名 `<序号>.jpg`
  - `Body.enroll_self() -> str`：返回 status 用的文字（`团子登记：N 张` / `团子登记：没转（原因）` / `团子登记：转了一圈没认出自己`），存进 `self.enroll_note`
  - 跳过条件（按顺序，原因原样写进文字）：`[appearance] enabled` 关、env 没有 `sweep`（不是感知层）、或感知层没挂 dino（直接返回空串，不写 status）；`dry-run`；`画面黑着`；`别的面板开着`；`没有视角控制`；`正在<技能>`
  - 转：同 `sweep_around` 的 live 分支（`_held("camera")`、`camera.spin(...)`、`env.sweep([(0.0, shot.before), *shot.frames], spin, enroll=True)`、finally 里 `_ref_thumb = None`、`_camera_moved("spin")`）
  - `run()`：`self.panel.start(...)` 之后、`while` 之前调 `enroll_self()`，异常只记日志
  - status：`self.enroll_note` 非空时 `parts.append(self.enroll_note)`（放在"你自己："那行后面）

- [ ] **Step 1: 写失败的测试**：
  - perception `test_sweep_enroll_pins_self_samples`：8 帧，每帧中间有团子色的 `self` 框 → `result.enrolled == 8`（`enroll_max=16`）、`me.gallery.pinned_count == 8`
  - `test_sweep_enroll_caps_and_spreads`：30 帧、`enroll_max=4` → 4 张，取到的帧号均匀（0、10、20、29 这类，断言首尾都在、间隔差不超过 1）
  - `test_sweep_enroll_falls_back_to_static_box`：没有 `self` 检测、只有一直在中间的 player 框 → 照样登记
  - `test_enroll_skips_blocked_self_boxes`（Review Focus 4）：团子框被另一个大框盖住超过 `max_overlap` → 那几帧不登记、`enrolled` 只算没被挡的
  - `test_sweep_without_enroll_learns_nothing`
  - body `test_enroll_self_spins_once_and_reports`（假 env 记 `enroll` 参数、返回 `enrolled=5`）→ status 含 `团子登记：5 张`、`cam.spins == 1`
  - `test_enroll_self_skipped_in_dry_run` → 不转，status 含 `团子登记：没转（dry-run）`
  - `test_enroll_self_zero_samples_warns` → `团子登记：转了一圈没认出自己`
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `python -m pytest tests/test_perception_appearance.py tests/test_brain_appearance_body.py tests/test_brain_body.py tests/test_body_camera_moved.py -q` → 全过
- [ ] **Step 5: 提交** `feat(appearance): 启动时转一圈登记团子`

### Task 8: cli 接线、和第二层共用 DINOv2、标定工具加底库模式

**Files:**
- Modify: `src/skydango/cli.py`（`_appearance_parts`、建 attrs 的地方、`_perception_appearance_eval`、`appearance-eval` 参数）
- Modify: `src/skydango/vision/appearance_eval.py`（`gallery_eval`、报告一节）
- Test: `tests/test_cli_appearance.py`、`tests/test_appearance_eval.py`

**Interfaces:**
- Consumes: Task 3 的 `DinoGuard`、`PerceptionWatcher(dino=..., enroll_dir=...)`
- Produces:
  - `_dino_embedder(cfg) -> OnnxEmbedder | None`：`[appearance] enabled` 且 `dino` 非空才建（device 用 `[appearance] device`）；文件不存在 / 加载失败 → WARNING `认装扮：DINOv2 加载失败（…），只用颜色`、返回 None。同一次运行只建一个：`[attrs]` 开着且 `Path(cfg.attrs.backbone).resolve() == Path(cfg.appearance.dino).resolve()` 时把它传给 `attrs.load_model(..., embedder=)`
  - `_appearance_parts` 多给 `dino=DinoGuard(emb)`（emb 为 None 时不给）和 `enroll_dir=run.path / "enroll"`（有运行目录时）
  - `appearance_eval.gallery_eval(sequence: list[tuple[float, str, Sample]], dango: list[tuple[float, Sample]], max_wrong=0.02) -> dict`：每个身份前一半时间挂标签的样本当底库、后一半当查询（颜色最像一张），开头 20 秒团子样本当团子底库（DINOv2）；返回 `{"match": …, "unsure": …, "dango_match": …, "recall": {...}, "bands": {"maybe": n, "unsure": n, "below": n}}`，门槛 = 负样本（团子 + 别的身份）错 < `max_wrong` 的最小两位小数；`unsure` = 错 < 0.07 的门槛
  - `perception appearance-eval <录像> --gallery`：收集时额外记每帧 `data["sample"]`（`Harvester.add` 加 `gallery=True` 时收 Sample 和团子样本），报告多一节"底库模式"
- [ ] **Step 1: 写失败的测试**：
  - `test_dino_embedder_shared_with_attrs`（monkeypatch `OnnxEmbedder` 计数）：两边路径相同 → 只建 1 个
  - `test_dino_embedder_missing_file_warns`：`dino = "models/没有.onnx"` → None、日志含"只用颜色"
  - `test_gallery_eval_suggests_thresholds`：合成两人 + 团子的样本序列（同身份余弦 0.95、跨身份 0.6）→ `match` 在 (0.6, 0.95] 之间、`bands` 三个数加起来等于查询数
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `python -m pytest tests/test_cli_appearance.py tests/test_appearance_eval.py tests/test_perception_attrs.py -q` → 全过
- [ ] **Step 5: 用真录像试一次**（GPU 机器上）：`python -m skydango perception appearance-eval tmp/record/walkaway-1002-1 --gallery`，报告里底库模式一节有三个建议值、和 spec §7 的数量级一致（match 0.85~0.92）
- [ ] **Step 6: 提交** `feat(appearance): cli 接上 DINOv2（和第二层共用）、appearance-eval 加底库模式`

### Task 9: 文档和收尾

**Files:**
- Modify: `CLAUDE.md`（「认装扮」一节：多张底库、三档、像团子、拿不准就喊、转圈登记、`card_match` 废弃、新配置项；「运行目录」表加 `enroll/`；「按 Q 喊一声」自动兜底加"拿不准"起因；常用命令 `appearance-eval --gallery`）
- Modify: `config.example.toml`（`[appearance]` 新项和新默认值，注释写来源"10-03 spike"）
- Modify: `docs/progress/2026-10-03-plan.md`（③ 结论 + 链接 spec / plan）

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 全量测试放后台跑** `python -m pytest -q` → 全过（失败就回到对应任务）
- [ ] **Step 3: 提交** `docs: 认人底库（CLAUDE.md、配置模板、进度）`
- [ ] **Step 4: 合并进 main 并推送**（本仓库规矩）
