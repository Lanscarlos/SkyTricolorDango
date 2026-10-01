# 认装扮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YOLO 人物框裁图算外观特征：名字标签没读到时把好友接回去（`maybe`）、这次上线认出回来的陌生人（「陌生人A」）、Haiku 把装扮写成文字（团子自己 / 好友 / 陌生人），好友的装扮记进关系卡，顺手攒认人模型的训练数据。

**Architecture:** 纯计算放 `vision/appearance.py`（裁图、颜色特征、外观记忆簿和规则），`PerceptionWatcher.process` 每帧调它判 `maybe` / 陌生人编号；
描述器 `vision/wardrobe.py` 是后台队列 + 一次性 `claude -p --model haiku`，结果回调进记忆簿；跨线程给身体的东西（老面孔回来、好友装扮）走 `pop_*` 队列，身体在自己线程里写关系卡、发事件。

**Tech Stack:** Python 3.13、numpy、OpenCV（HSV 直方图）、onnxruntime（可选特征模型）、Claude Code CLI（`one_shot`）、pytest。

**Spec:** `docs/superpowers/specs/2026-10-01-appearance-design.md`

## Global Constraints

- `[appearance] enabled = false`（默认）时：提示词、事件、status、`people()`、`overlay()` **逐字照旧**；不建记忆簿、不起描述器线程
- 外观要配合 `[perception] enabled`；只开 `[appearance]` 时记一条 WARNING、不认装扮（同 `[places]`）
- 名字标签永远说了算；`maybe` 不发 `arrive` / `return`，只刷新**还在身边**（`last_seen` 在 `perception.keep` 内）的好友
- 没点火的黑影（`player_unlit`）不算外观、不分陌生人编号
- 特征 `key` 不同（换了模型）的两份不比；默认颜色特征 `key = "color-v1"`
- 关系卡只在 `--live` 写盘；旧 `people.json` 没有 `outfits` 照常读；每人最多 `outfit_keep`（3）套，最近的在最后，特征存 3 位小数
- 描述提示词：只说颜色和样子，**不说季节名 / 物品名**；`desc` 截到 25 字；回 JSON `{"desc": …, "clear": …}`
- 描述额度：每小时最多 `describe_max`（20）次；额度用完 `quota_wait`（600 秒）后再试；同一身份 `clear: false` 后 `retry_after`（60 秒）再试、最多 3 次
- 攒数据：同一条轨迹每 `save_every`（2.0 秒）最多一张，一次运行最多 `save_max`（2000）张；路径 `runs/<…>/appearance/crops/<身份>/<时间毫秒>.jpg` + `appearance.jsonl`
- 配置默认值照 spec §9 的表；新代码的注释、日志、文字用中文，风格照周围代码
- 测试：`.venv\Scripts\python.exe -m pytest -q`（没有 `.venv` 就 `python -m pytest -q`）；改完合并进 main 并推送

## 对 spec 的一处修正（Task 12 同步改 spec）

spec §6 说"arrive 只在换了装、两套都有描述时多一句，描述没回来就放弃"。但换装要攒够样本（约 1~2 秒）再等 Haiku（几秒），arrive 时描述几乎不可能已经回来，这句话等于永远不出现。
改成：**好友换装的描述回来时，身体发一个背景事件 `outfit`**："小明换了装扮：上次是「…」，现在「…」"（`BACKGROUND`，随下一次醒来给大脑）；arrive 文字不变。

## Review Focus

1. **团子和好友挤在一起**：团子框和好友框重叠时裁图不是好样本 → 不学、不判；测试见 Task 2 `test_good_crop_rejects_overlap`、Task 4 `test_overlapping_self_is_not_learned_as_friend`
2. **两个好友撞衫**：小明、小红外观几乎一样时，没标签的轨迹两人都像 → 领先不到 `margin` 不给 `maybe`；Task 3 `test_no_maybe_when_two_friends_look_alike`
3. **感知暂停后恢复 / 镜头转回来**：轨迹断开重建，好友外观在记忆簿里还在 → 新轨迹攒够样本后接回 `maybe`，名字标签被挡期间不发 leave；Task 4 `test_friend_survives_tag_loss_via_appearance`
4. **描述器额度用完 / 子进程失败**：`ClaudeError(limit=True)` 后 10 分钟内不再调；普通失败按 `clear: false` 处理；感知线程不被阻塞；Task 6 `test_limit_pauses_describing`、`test_failure_counts_as_retry`
5. **旧 people.json、换了特征模型**：卡里的旧特征 `key` 不同时不比、但描述照样给大脑；Task 3 `test_card_feature_with_other_key_is_ignored`、Task 8 `test_old_people_json_without_outfits_loads`

---

### Task 1: 特征模型挪到公共模块 `vision/embed.py`

**Files:**
- Create: `src/skydango/vision/embed.py`
- Modify: `src/skydango/vision/places.py:57-112`（`_unit`、`OnnxEmbedder`、`NORMS` 改从 embed 导入；`make_embedder(cfg: PlacesConfig)` 留在 places）
- Test: `tests/test_embed.py`（新）；`tests/test_places.py` 原样要过

**Interfaces:**
- Produces: `embed.unit(v) -> np.ndarray`（原 `_unit`）、`embed.NORMS`、`embed.OnnxEmbedder(path: str, size: int = 224, norm: str = "imagenet", device: str = "cpu", what: str = "places.model")`、`embed.cosine(a, b) -> float`（单位向量点积，维度不同返回 0.0）
- places.py 里保留 `_unit = unit`、`OnnxEmbedder` 名字可导入（老测试 / cli 用）

- [ ] **Step 1: 写失败的测试** `tests/test_embed.py`

```python
def test_onnx_embedder_missing_file_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="appearance.model"):
        OnnxEmbedder(str(tmp_path / "x.onnx"), what="appearance.model")

def test_cosine_of_unit_vectors_and_dimension_mismatch():
    a, b = unit([1, 0, 0]), unit([1, 1, 0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(2 ** -0.5)
    assert cosine(a, unit([1, 0])) == 0.0
```

- [ ] **Step 2: 跑测试确认失败**：`.venv\Scripts\python.exe -m pytest tests/test_embed.py -q` → ImportError
- [ ] **Step 3: 实现**：把 `_unit`、`NORMS`、`OnnxEmbedder` 从 places.py 剪到 embed.py；错误提示里的 `places.model` 换成 `what`；places.py 从 embed 导入
- [ ] **Step 4: 跑** `pytest tests/test_embed.py tests/test_places.py -q` → 全过
- [ ] **Step 5: 提交** `refactor(vision): 特征模型挪到 embed.py，认装扮也要用`

---

### Task 2: 配置 + 裁图 + 颜色特征（`vision/appearance.py` 第一部分）

**Files:**
- Modify: `src/skydango/config.py`（新 `AppearanceConfig`，`Config.appearance`）、`config.example.toml`（加 `[appearance]` 段，照 spec §9 注释）
- Create: `src/skydango/vision/appearance.py`
- Test: `tests/test_appearance.py`（新）、`tests/test_config.py`（加一条）

**Interfaces:**
- Produces:
  - `AppearanceConfig`：spec §9 全部键 + `size: int = 224`、`norm: str = "imagenet"`、`device: str = "cpu"`、`quota_wait: float = 600.0`、`retry_after: float = 60.0`、`describe_timeout: float = 60.0`
  - `good_crop(frame, box: Rect, others: list[Rect], blocked: list[Rect], min_height: float, max_overlap: float) -> np.ndarray | None`：框高 < `min_height × 帧高` → None；和 `others`（别的人物 / 团子框）任一 IoU 或"被盖住的比例"> `max_overlap` → None；和 `blocked`（聊天面板）重叠面积占框 > `max_overlap` → None；否则返回框中间 60% 宽、整高的裁图
  - `describe_crop(frame, box: Rect) -> np.ndarray`：四周各扩 15%、夹到画面内（送去描述用）
  - `ColorEmbedder`：`key = "color-v1"`，`embed(img) -> np.ndarray`（单位向量）。头 = 上 30%、身体 = 下 70%；各算 HSV 直方图 `calcHist([hsv], [0, 1], mask, [16, 4], [0, 180, 0, 256])`，mask 去掉 V < 30 或 V > 245 的像素，各自 `unit` 后拼接再 `unit`
  - `make_embedder(cfg: AppearanceConfig)`：`"color"` → `ColorEmbedder()`；`.onnx` → `OnnxEmbedder(cfg.model, cfg.size, cfg.norm, cfg.device, what="appearance.model")`；其它 `ValueError`

- [ ] **Step 1: 写失败的测试**

```python
def person(color_head, color_body, h=300, w=120):  # 合成"人"：上 30% 一种颜色、下 70% 另一种
    ...
def test_color_embedder_separates_outfits():
    e = ColorEmbedder()
    pink, pink2, blue = e.embed(person(WHITE, PINK)), e.embed(person(WHITE, PINK, h=280)), e.embed(person(WHITE, BLUE))
    assert cosine(pink, pink2) > 0.95 and cosine(pink, blue) < 0.7

def test_color_embedder_head_matters():
    e = ColorEmbedder()
    assert cosine(e.embed(person(WHITE, PINK)), e.embed(person(BLACK_HAIR, PINK))) < 0.9

def test_same_color_different_brightness_is_close():  # 已知限制：亮度不同分不开
    e = ColorEmbedder()
    assert cosine(e.embed(person(WHITE, PINK)), e.embed(person(WHITE, PINK_DARKER))) > 0.85

def test_good_crop_rejects_small_and_overlap():
    f = frame_with_person(Rect(800, 400, 120, 300))
    assert good_crop(f, Rect(800, 400, 120, 300), [], [], 0.10, 0.2).shape[1] == 72  # 中间 60% 宽
    assert good_crop(f, Rect(800, 400, 30, 80), [], [], 0.10, 0.2) is None
    assert good_crop(f, Rect(800, 400, 120, 300), [Rect(830, 420, 120, 300)], [], 0.10, 0.2) is None
    assert good_crop(f, Rect(100, 400, 120, 300), [], [Rect(0, 0, 640, 900)], 0.10, 0.2) is None

def test_config_has_appearance_defaults():
    c = Config().appearance
    assert (c.enabled, c.model, c.match, c.card_match, c.margin, c.changed) == (False, "color", 0.85, 0.92, 0.05, 0.70)
```

- [ ] **Step 2: 跑** `pytest tests/test_appearance.py tests/test_config.py -q` → 失败
- [ ] **Step 3: 实现** `AppearanceConfig` 和上面四个函数 / 类
- [ ] **Step 4: 跑** 同上 → 通过
- [ ] **Step 5: 提交** `feat(appearance): 配置、裁图、颜色特征`

---

### Task 3: 外观记忆簿 `AppearanceBook`（`vision/appearance.py` 第二部分）

**Files:**
- Modify: `src/skydango/vision/appearance.py`
- Test: `tests/test_appearance.py`

**Interfaces:**
- Consumes: Task 2 的 `AppearanceConfig`、`embed.cosine`、`embed.unit`
- Produces（线程安全：内部一把锁；感知线程写、描述器回调写、身体线程读）：
  - `@dataclass Profile`：`feat: np.ndarray`、`n: int`、`updated: float`、`seen: float`、`desc: str = ""`、`desc_feat: np.ndarray | None = None`、`crops: deque`（`maxlen=5`，元素 `(框高, 描述用裁图)`）、`checked: bool = False`（好友：这次上线和关系卡比过没有）、`redescribed: int = 0`
  - `AppearanceBook(cfg: AppearanceConfig, key: str)`
  - `learn(kind: str, who: str, feat, now: float, crop: tuple[int, np.ndarray] | None = None) -> Profile`：kind ∈ `"friend" / "stranger" / "me"`；滑动平均 `feat = unit((1-ema)·old + ema·new)`，第一次直接用
  - `load_cards(outfits: dict[str, list[dict]]) -> None`：每个好友取最后一套；`key == self.key` 的特征进 `card_feats`，描述（不论 key）进 `card_desc`
  - `assign_friends(cands: dict[int, np.ndarray], exclude: set[str]) -> dict[int, str]`：候选 = `friends`（这次上线学到的，门槛 `match`）∪ `card_feats` 里没学到的（门槛 `card_match`），去掉 `exclude`；每条轨迹取最像的，要求 ≥ 门槛且比第二像的（任何候选）高 ≥ `margin`；同一个名字多条轨迹抢，只给相似度最高的那条
  - `still_like(track_feat, name) -> bool`：和该好友（学到的优先，否则卡里的）相似度 ≥ 对应门槛（`recheck` 用）
  - `stranger_id(feat, now) -> tuple[str, bool]`：最像的已有编号 ≥ `match` 且领先 ≥ `margin` → (编号, 是不是"回来了"：`now - seen > keep` 时为 True，`keep` 由构造参数 `keep: float` 传入，默认 5.0)；否则新编号 `陌生人A`、`B`…`Z`、`AA`、`AB`…，返回 (新编号, False)；都更新 `seen`
  - `forget(now) -> None`：删掉 `now - seen > stranger_forget` 的陌生人
  - `card_state(name) -> str`：`"new"`（卡里没有特征）/ `"same"`（相似度 ≥ `changed`）/ `"changed"`；只看 `key` 相同的卡特征
  - `drifted(kind, who) -> bool`：有 `desc_feat` 且 `cosine(feat, desc_feat) < changed`
  - `set_desc(kind, who, desc, feat) -> None`、`look(kind, who) -> str`（描述；好友没学到时退回 `card_desc`）
  - `best_crop(kind, who) -> np.ndarray | None`：`crops` 里框高最大的

- [ ] **Step 1: 写失败的测试**（特征直接用手写单位向量，不经裁图）

```python
def test_learn_averages_and_assigns_maybe():
    b = book(match=0.85, margin=0.05)
    b.learn("friend", "小明", PINK, 0.0)
    assert b.assign_friends({7: near(PINK)}, exclude=set()) == {7: "小明"}

def test_name_shown_elsewhere_is_excluded():
    b = book(); b.learn("friend", "小明", PINK, 0.0)
    assert b.assign_friends({7: PINK}, exclude={"小明"}) == {}

def test_two_tracks_one_name_only_best_wins():
    b = book(); b.learn("friend", "小明", PINK, 0.0)
    assert b.assign_friends({1: near(PINK, 0.9), 2: near(PINK, 0.97)}, set()) == {2: "小明"}

def test_no_maybe_when_two_friends_look_alike():
    b = book(); b.learn("friend", "小明", PINK, 0.0); b.learn("friend", "小红", near(PINK, 0.99), 0.0)
    assert b.assign_friends({1: PINK}, set()) == {}

def test_card_only_friend_needs_stricter_threshold():
    b = book(match=0.85, card_match=0.92); b.load_cards({"小明": [outfit(PINK, key="color-v1")]})
    assert b.assign_friends({1: near(PINK, 0.88)}, set()) == {}
    assert b.assign_friends({1: near(PINK, 0.95)}, set()) == {1: "小明"}

def test_card_feature_with_other_key_is_ignored():
    b = book(); b.load_cards({"小明": [outfit(PINK, key="other", desc="粉斗篷")]})
    assert b.assign_friends({1: PINK}, set()) == {} and b.card_state("小明") == "new"
    assert b.look("friend", "小明") == "粉斗篷"

def test_stranger_ids_reuse_new_back_and_forget():
    b = book(stranger_forget=1800, keep=5.0)
    assert b.stranger_id(WHITE, 0.0) == ("陌生人A", False)
    assert b.stranger_id(BLUE, 1.0) == ("陌生人B", False)
    assert b.stranger_id(near(WHITE), 2.0) == ("陌生人A", False)
    assert b.stranger_id(near(WHITE), 60.0) == ("陌生人A", True)  # 离开超过 keep 又认回来
    b.forget(60.0 + 1801); assert b.stranger_id(WHITE, 2000.0)[0] == "陌生人C"

def test_card_state_and_drift():
    b = book(changed=0.70); b.load_cards({"小明": [outfit(PINK)]})
    b.learn("friend", "小明", near(PINK, 0.95), 0.0); assert b.card_state("小明") == "same"
    b.learn("friend", "小红", BLUE, 0.0); assert b.card_state("小红") == "new"
    m = book(ema=1.0, changed=0.70)  # ema = 1：学一次就整个换掉
    m.learn("me", "", WHITE, 0.0); m.set_desc("me", "", "白斗篷", WHITE)
    assert not m.drifted("me", "")
    m.learn("me", "", BLUE, 1.0); assert m.drifted("me", "")
```

（`near(v, s)` 造一个和 v 余弦相似度为 s 的单位向量；`book(keep=5.0, **kw)` = `AppearanceBook(AppearanceConfig(**kw), "color-v1", keep=keep)`；`outfit(feat, key="color-v1", desc="")` 造一套关系卡装扮 dict）

- [ ] **Step 2: 跑** `pytest tests/test_appearance.py -q` → 失败
- [ ] **Step 3: 实现** `Profile`、`AppearanceBook`
- [ ] **Step 4: 跑** → 通过
- [ ] **Step 5: 提交** `feat(appearance): 外观记忆簿（maybe、陌生人编号、换装判定）`

---

### Task 4: 接进感知层（`maybe`、陌生人编号、老面孔回来）

**Files:**
- Modify: `src/skydango/vision/perception.py`（`__init__` 新参数、`process` 455-485 一段、`_watch_typing` / `_watch_approach` / `talkers` / `nearest` / `people` / `overlay`，新方法见下）
- Modify: `src/skydango/vision/people.py`（`Person` 加字段、`describe_people`）
- Modify: `src/skydango/vision/hardcases.py`（公开 `report`）
- Modify: `src/skydango/brain/images.py`（`scene_note` 认 `maybe`）
- Test: `tests/test_perception_appearance.py`（新；复用 `tests/test_perception.py` 的 `FakeDetector` / `FakeOcr` / `player` / `tag` / `Clock`，画面用彩色：在人物框位置涂颜色）

**Interfaces:**
- Consumes: Task 2/3（`good_crop`、`describe_crop`、`AppearanceBook`、`make_embedder`）
- Produces:
  - `PerceptionWatcher(..., appearance: AppearanceBook | None = None, embedder=None, appearance_cfg: AppearanceConfig | None = None, saver=None)`；`saver` 见 Task 5
  - 轨迹 `data`：`feat`（平滑后的单位向量）、`samples`（好样本数）、`maybe`（好友名）、`miss`（`maybe` 连续不像的次数）、`sid`（陌生人编号）
  - `pop_stranger_backs() -> list[str]`：取走"又回来了"的陌生人编号（身体发事件）
  - `my_look() -> str`、`looks(names: list[str]) -> dict[str, str]`：团子 / 好友当前的装扮描述（没有就不给）
  - `Person` 新字段（都有默认值，旧调用不变）：`sure: bool = True`、`sid: str | None = None`、`look: str = ""`
  - `describe_people`：`maybe` → `像小明（没看到名字，右边·远）`；有 `sid` 的陌生人 → `陌生人A（白斗篷，前面·中）`，没描述 → `陌生人A（前面·中）`；别的照旧
  - `overlay()`：`maybe` 轨迹 `kind = "maybe"`、`label = "像小明?"`；有 `sid` 的陌生人 `label = "陌生人A"`；每条多一个 `desc`（有描述才有）
  - `HardCaseCollector.report(frame, now, reason: str, detail: str, tracks: list[Track]) -> bool`（包 `_save`）

**`process` 里的顺序**（appearance 为 None 时整段跳过、行为不变）：
1. 现有的标签分配之后、陌生人判断之前，调 `self._appearance_features(frame, players, selfs, now, width, height, panel_visible)`：对 `player`（不含 `UNLIT`）和 `self` 轨迹，`track.hits % every == 0` 时裁好样本（`others` = 本帧其它人物和团子框、`blocked` = 面板开着时的 `log_roi`），每帧最多 `max_per_frame` 个（按 `data["feat_at"]` 最旧的先算），算特征、平滑进 `data["feat"]`、`samples += 1`，把 `(框高, describe_crop)` 交给记忆簿时用
2. 学：这一帧挂着有名字标签的轨迹 → `learn("friend", 名字, …)`；`self` → `learn("me", "", …)`
3. 没标签的 `player` 轨迹：已有 `maybe` 的，`still_like` 不成立就 `miss += 1`，到 `recheck` 摘掉；没有 `maybe`、`samples ≥ min_samples` 的放进候选 → `assign_friends(候选, exclude=shown ∪ 这一帧被标签占着的名字)` → 写 `maybe`
4. 原来的 `is_stranger` 加条件 `not player.data.get("maybe")`
5. `maybe` 的好友 `now - last_seen[name] <= cfg.keep` 时 `last_seen[name] = now`（不改 `labels`）
6. 判成陌生人、`samples ≥ min_samples`、还没 `sid` 的 → `stranger_id`；是"回来了"就进 `_stranger_backs`；有 `sid` 的继续 `learn("stranger", sid, …)`
7. 轨迹挂上标签而原来有 `maybe` / `sid`：摘掉；`maybe` 和标签名字不一样时 `hardcases.report(frame, now, "appearance", "按外观认成 X，名字标签是 Y", tracks)`
8. 每帧末尾 `appearance.forget(now)`
- `_watch_typing` / `talkers`：`friend = bool(name or tagged or maybe)`，`talkers` 的 `name = name or maybe`
- `_watch_approach`：`who = name or (maybe if maybe in self.nearby(now)) or (STRANGER if stranger)`
- `nearest()` 把 `maybe` 当名字；`people()`：`maybe` → `Person(kind="friend", name=maybe, sure=False)`；陌生人带 `sid` 和 `look`
- `scene_note`：`maybe` 框写成"像小明（没看到名字）：(x, y) 附近"，排在好友之后

- [ ] **Step 1: 写失败的测试**

```python
def test_friend_survives_tag_loss_via_appearance():
    # 小明（粉）带标签 1 秒 → 标签没了、轨迹断开重建（镜头转了）→ 攒够样本后是 maybe，不是陌生人，不走开
    ...
    assert [p.name for p in w.people(t) if not p.sure] == ["懒洋洋大王"]
    assert w.strangers(t) == 0 and w.nearby(t + 4.0) == ["懒洋洋大王"]

def test_maybe_does_not_bring_back_a_friend_who_left():
    # 小明走开超过 keep 后，没标签的粉色人回来：maybe 有，但 nearby 不含他
    assert w.nearby(t) == [] and w.people(t)[0].sure is False

def test_tag_overrides_maybe_and_reports_mismatch():
    # maybe=小明 的轨迹挂上 番茄炒蛋盖饭 的标签 → maybe 摘掉，hardcases.report 收到 "appearance"

def test_lit_stranger_gets_id_and_comes_back():
    # 白色陌生人 → 陌生人A；离开 6 秒（> keep）再出现 → pop_stranger_backs() == ["陌生人A"]

def test_overlapping_self_is_not_learned_as_friend():
    # 好友框和团子框 IoU > max_overlap：不 learn、samples 不涨

def test_unlit_gets_no_features():
    # player_unlit 轨迹没有 feat / sid

def test_appearance_off_is_unchanged():
    # appearance=None：people() 的 Person 都是 sure=True、sid=None；overlay 没有 desc 键

def test_describe_people_wording():
    assert describe_people([Person(1, "friend", "小红", R, "右边", "远", sure=False),
                            Person(2, "stranger", None, R, "前面", "中", sid="陌生人A", look="白斗篷")]) \
        == "像小红（没看到名字，右边·远）、陌生人A（白斗篷，前面·中）"
```

- [ ] **Step 2: 跑** `pytest tests/test_perception_appearance.py -q` → 失败
- [ ] **Step 3: 实现**（顺序见上）
- [ ] **Step 4: 跑** `pytest tests/test_perception_appearance.py tests/test_perception.py tests/test_brain_images.py tests/test_hardcases.py -q` → 全过
- [ ] **Step 5: 提交** `feat(perception): 按外观接回好友（maybe）、陌生人编号和老面孔`

---

### Task 5: 攒训练数据 `CropSaver`

**Files:**
- Modify: `src/skydango/vision/appearance.py`（加 `CropSaver`）、`src/skydango/vision/perception.py`（学完一张好样本后调 saver）
- Test: `tests/test_appearance.py`

**Interfaces:**
- Produces: `CropSaver(folder: Path, save_every: float, save_max: int, wall: Callable[[], float] = time.time)`；
  `offer(track_id: int, who: str, crop: np.ndarray, now: float, row: dict) -> bool`：同一 `track_id` 距上次 < `save_every` 不存；总数到 `save_max` 不存（只在刚满时记一次 INFO）；
  存 `folder/crops/<who>/<毫秒>.jpg`（`who` 里 Windows 不允许的字符换成 `_`），往 `folder/appearance.jsonl` 追加 `{"file": 相对路径, **row}`
- 感知层调用：`who` = 这一帧标签上的名字，否则 `f"t{track.id}"`；`row` = `{"t", "track", "tag", "maybe", "sid", "box": [x, y, w, h], "best": 最像的好友名, "score", "place": self.place}`；`self` 轨迹不存

- [ ] **Step 1: 写失败的测试**

```python
def test_crop_saver_rate_limits_per_track_and_caps(tmp_path):
    s = CropSaver(tmp_path, save_every=2.0, save_max=3)
    assert s.offer(1, "小明", CROP, 0.0, {"t": 0.0}) and not s.offer(1, "小明", CROP, 1.0, {})
    assert s.offer(2, "t2", CROP, 1.0, {}) and s.offer(1, "小明", CROP, 2.5, {}) and not s.offer(3, "t3", CROP, 9.0, {})
    assert len(list((tmp_path / "crops").rglob("*.jpg"))) == 3
    rows = [json.loads(l) for l in (tmp_path / "appearance.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["file"].startswith("crops/小明/")

def test_crop_saver_sanitizes_names(tmp_path):
    CropSaver(tmp_path, 2.0, 10).offer(1, 'a:b*?', CROP, 0.0, {})
    assert (tmp_path / "crops" / "a_b__").is_dir()
```

- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现** `CropSaver`；感知层在 Task 4 的第 2/6 步学完后 `saver.offer(...)`（saver 为 None 跳过，出错只记日志）
- [ ] **Step 4: 跑** `pytest tests/test_appearance.py tests/test_perception_appearance.py -q` → 通过
- [ ] **Step 5: 提交** `feat(appearance): 运行时攒认人模型的训练数据`

---

### Task 6: 描述器 `vision/wardrobe.py`

**Files:**
- Create: `src/skydango/vision/wardrobe.py`
- Test: `tests/test_wardrobe.py`（新）

**Interfaces:**
- Consumes: `brain.claude.ClaudeError`、`brain.images.image_block` / `fit`
- Produces:
  - 优先级常量 `ME = 0`、`FRIEND = 1`、`STRANGER = 2`
  - `WARDROBE_SYSTEM`、`WARDROBE_REQUEST`（中文；只说颜色和样子、不说季节名 / 物品名；看不清回 `clear: false`；只回 JSON）
  - `parse_outfit(text: str) -> tuple[str, bool] | None`：取文字里第一个 `{…}` 解析；`desc` 去空白、截 25 字；`clear` 缺省 True；`desc` 空 → `clear = False`；解析不了 → None
  - `wardrobe_command(base: list[str], model: str) -> list[str]`（同 `eyes_command` 的参数，`--model model`、`--system-prompt WARDROBE_SYSTEM`）
  - `Wardrobe(cfg: AppearanceConfig, describe: Callable[[list[dict]], str], on_done: Callable[[str, str, str, np.ndarray], None], clock=time.monotonic)`
    - `request(kind: str, who: str, priority: int, crop: np.ndarray, feat: np.ndarray, now: float) -> bool`：拒绝（返回 False）当：这个 `(kind, who)` 已在队列 / 正在描述；`retry_at` 没到；失败满 3 次（直到 `reset(kind, who)`）；否则入队（同一身份只留一个）
    - `reset(kind, who) -> None`：换装后清掉失败次数
    - `tick(now) -> bool`：额度窗内（最近 3600 秒）已调 `describe_max` 次、或 `now < limit_until` → 不做；否则按优先级（小的先）、同优先级先来先做取一个，调 `describe([image_block(fit(crop, (512, 512)), 85), {"type": "text", "text": WARDROBE_REQUEST}])`：
      成功且 clear → `on_done(kind, who, desc, feat)`；`clear: false` / 解析不了 / 普通 `ClaudeError` → 失败次数 +1、`retry_at = now + retry_after`；`ClaudeError(limit=True)` → `limit_until = now + quota_wait`、这一项放回队列
    - `run(stop: threading.Event) -> None`：每 0.5 秒 `tick`，异常只记日志
    - `calls: int`（描述过几次，status / 日志用）

- [ ] **Step 1: 写失败的测试**（`describe` 用假的：按调用依次返回字符串或抛异常）

```python
def test_parse_outfit():
    assert parse_outfit('好的 {"desc": "白色樱花发型、粉色长斗篷", "clear": true}') == ("白色樱花发型、粉色长斗篷", True)
    assert parse_outfit('{"desc": "", "clear": true}') == ("", False)
    assert parse_outfit("看不清") is None
    assert len(parse_outfit('{"desc": "' + "长" * 40 + '"}')[0]) == 25

def test_priority_and_dedup():
    w = wardrobe(['{"desc":"自己"}', '{"desc":"小明"}'])
    assert w.request("stranger", "陌生人A", STRANGER, C, F, 0) and w.request("me", "", ME, C, F, 0)
    assert not w.request("me", "", ME, C, F, 0)
    w.tick(0); assert done == [("me", "", "自己")]

def test_budget_per_hour():
    w = wardrobe(['{"desc":"x"}'] * 3, describe_max=2)
    for who in "abc": w.request("stranger", who, STRANGER, C, F, 0)
    w.tick(0); w.tick(1); assert not w.tick(2) and w.calls == 2
    assert w.tick(3601)

def test_unclear_retries_after_wait_then_gives_up():
    w = wardrobe(['{"desc":"","clear":false}'] * 3, retry_after=60)
    for t in (0, 60, 120):
        assert w.request("friend", "小明", FRIEND, C, F, t); w.tick(t)
    assert not w.request("friend", "小明", FRIEND, C, F, 999)
    w.reset("friend", "小明"); assert w.request("friend", "小明", FRIEND, C, F, 999)

def test_limit_pauses_describing():
    w = wardrobe([ClaudeError("额度", limit=True), '{"desc":"x"}'], quota_wait=600)
    w.request("me", "", ME, C, F, 0); w.tick(0)
    assert not w.tick(599) and w.tick(600) and done[-1][2] == "x"

def test_failure_counts_as_retry():
    w = wardrobe([ClaudeError("挂了")]); w.request("me", "", ME, C, F, 0); w.tick(0)
    assert not w.request("me", "", ME, C, F, 30)
```

- [ ] **Step 2: 跑** `pytest tests/test_wardrobe.py -q` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** → 通过
- [ ] **Step 5: 提交** `feat(appearance): 描述器（Haiku 把装扮写成一句话）`

---

### Task 7: 感知层触发描述、好友装扮队列

**Files:**
- Modify: `src/skydango/vision/perception.py`
- Test: `tests/test_perception_appearance.py`

**Interfaces:**
- Consumes: Task 3 `AppearanceBook`（`card_state`、`drifted`、`best_crop`、`set_desc`、`look`）、Task 6 `Wardrobe.request` / `reset` 和优先级常量
- Produces:
  - 属性 `wardrobe: Wardrobe | None = None`（cli 在建好大脑后挂上）；回调方法 `on_described(kind: str, who: str, desc: str, feat) -> None`（在描述器线程里调）
  - `@dataclass(frozen=True) OutfitNote`：`name: str`、`feat: list[float]`（3 位小数）、`key: str`、`state: str`（`"new" / "same" / "changed" / "described"`）、`desc: str = ""`
  - `pop_outfits() -> list[OutfitNote]`（身体取走）
- **触发规则**（每帧学完之后；`wardrobe` 为 None 或 `describe = false` 时只做判定、不 request）：
  - 团子：`me` 还没描述过且有好样本 → `request("me", "", ME, …)`；描述过且 `drifted("me", "")`（每次上线最多 `redescribe_max` 次）→ `reset` + `request`
  - 好友：`samples ≥ min_samples` 且这次上线还没 `checked` → `checked = True`，`state = card_state(name)`，推 `OutfitNote(name, feat, key, state)`；`"same"` → `set_desc("friend", name, card_desc, feat)`（直接用卡里的描述，不请求）；`"new"` / `"changed"` → request（框高 ≥ `describe_min_height × 帧高` 的样本才有，`best_crop` 为 None 时下一帧再试）
  - 好友上线中 `drifted`（`redescribed < redescribe_max`）→ 推 `OutfitNote(..., "changed")`、`redescribed += 1`、`reset` + request
  - 陌生人：有 `sid`、没描述过、`distance` 是 近 / 中、框高够 → `request("stranger", sid, STRANGER, …)`
  - `on_described`：`set_desc`；`kind == "friend"` 时推 `OutfitNote(who, feat, key, "described", desc)`

- [ ] **Step 1: 写失败的测试**（假 Wardrobe 只记 request）

```python
def test_me_described_once_then_on_drift():
def test_friend_without_card_is_described_and_noted_new():
    assert [n.state for n in w.pop_outfits()] == ["new"] and fake.requests == [("friend", "懒洋洋大王")]
def test_friend_same_outfit_uses_card_desc_without_request():
    # load_cards 有一套粉色、desc="粉色长斗篷" → state "same"、不 request、looks(["懒洋洋大王"]) == {"懒洋洋大王": "粉色长斗篷"}
def test_described_friend_pushes_note():
    w.on_described("friend", "懒洋洋大王", "粉色长斗篷", F)
    assert w.pop_outfits()[-1].state == "described" and w.looks(["懒洋洋大王"])["懒洋洋大王"] == "粉色长斗篷"
def test_far_stranger_not_described():
```

- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `pytest tests/test_perception_appearance.py -q` → 通过
- [ ] **Step 5: 提交** `feat(perception): 触发装扮描述、好友装扮交给身体`

---

### Task 8: 关系卡记装扮

**Files:**
- Modify: `src/skydango/inner/ledger.py`（`Card.outfits`、三个方法）
- Modify: `src/skydango/console/inner_view.py` + `src/skydango/console/static/console.html`（关系卡显示最近几套：描述 + 日期）
- Test: `tests/test_inner_ledger.py`、`tests/test_inner_store.py`、`tests/test_console_inner.py`

**Interfaces:**
- Produces:
  - `Card.outfits: list[dict] = field(default_factory=list)`，元素 `{"desc", "feat", "key", "first", "last"}`
  - `Ledger.all_outfits() -> dict[str, list[dict]]`（深拷贝；启动时给 `AppearanceBook.load_cards`）
  - `Ledger.wear(name: str, feat: list[float], key: str, changed: bool, now: float) -> None`：`match_friend` 归名；没卡就建卡；`changed` 或没有装扮 → 追加一套（`desc=""`、`first=last=今天`），超过 `outfit_keep` 删最旧；否则更新最后一套的 `last`、`feat`
  - `Ledger.describe_outfit(name: str, desc: str, now: float) -> str | None`：写进最后一套的 `desc`；最后一套是今天新加的、且前一套有描述时返回前一套的描述（给 `outfit` 事件），否则 None
  - `InnerConfig` 不动；`outfit_keep` 从 `AppearanceConfig` 传：`Ledger(..., outfit_keep: int = 3)`

- [ ] **Step 1: 写失败的测试**

```python
def test_wear_new_then_same_updates_last_day():
def test_wear_changed_appends_and_keeps_three():
def test_describe_outfit_returns_previous_desc_only_for_a_change():
    l.wear("小明", F1, "color-v1", True, D1); l.describe_outfit("小明", "白斗篷", D1)
    l.wear("小明", F2, "color-v1", True, D2)
    assert l.describe_outfit("小明", "粉斗篷", D2) == "白斗篷"
def test_old_people_json_without_outfits_loads(tmp_path):
    # 写一份没有 outfits 字段的 people.json → load_people 读成功、card.outfits == []
def test_outfits_written_only_when_live(tmp_path):
def test_inner_view_shows_outfits():
```

- [ ] **Step 2: 跑** `pytest tests/test_inner_ledger.py tests/test_inner_store.py tests/test_console_inner.py -q` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** → 通过
- [ ] **Step 5: 提交** `feat(inner): 关系卡记好友的装扮`

---

### Task 9: 身体和大脑看到的

**Files:**
- Modify: `src/skydango/brain/body.py`（`_watch_comings`、`status`、`look_person` / 新 `_locate_by_look`、`target_x`）
- Modify: `src/skydango/brain/events.py`（`BACKGROUND` 加 `stranger_back`、`outfit`；`Event.kind` 注释）
- Modify: `src/skydango/brain/prompt.py`（`APPEARANCE_RULES`、`brain_prompt(..., appearance: bool = False)`）
- Modify: `src/skydango/brain/tools.py`（`look_person` 描述加一句"也可以写「陌生人A」"）
- Test: `tests/test_brain_appearance_body.py`（新，用 `tests/test_brain_body.py` 的 `body()` 和 `FakeEnv` 扩一个 `FakeLookEnv`）、`tests/test_brain_prompt.py`

**Interfaces:**
- Consumes: Task 4 `pop_stranger_backs` / `my_look` / `looks` / `Person.sure|sid|look`；Task 7 `pop_outfits` / `OutfitNote`；Task 8 `Ledger.wear` / `describe_outfit`
- Produces:
  - `_watch_comings` 末尾（`env` 有这些方法才调）：
    - 每个 `pop_stranger_backs()` → `events.put("stranger_back", f"刚才那个{sid}（{look}）又回来了")`（没描述时去掉括号）
    - 每个 `pop_outfits()`：`new` / `same` / `changed` → `_ledger_call("wear", name, feat, key, state != "same", wall)`；`described` → `prev = _ledger_call("describe_outfit", name, desc, wall)`，`prev` 有值 → `events.put("outfit", f"{name}换了装扮：上次是「{prev}」，现在「{desc}」", who=name)`
  - `status()`：
    - "身边的好友："之前加一行 `你自己：{my_look}`（有才加）
    - 好友装扮：`ledger.status_line(near, wall, looks=looks)`；`Ledger.status_line` 加可选参数 `looks: dict[str, str] | None = None`，有描述时括号里放在最前：`小明（粉色长斗篷·今天一起 40 分钟·…）`；没 ledger 时 `小明（粉色长斗篷）`
    - "画面里："照用 `describe_people`（Task 4 已改）
  - `_locate` 只用 `sure` 的人；`look_person` 在 `_locate` 找不到时调 `_locate_by_look(name, now) -> tuple[Rect, str] | None`：先找 `sid == name` 的陌生人，再找 `sure=False` 且名字相同 / 相似的人；返回 (框, 附注)，附注是 `"（没看到名字，按外观认的）"` 或 `""`，加到结果文字后面
  - `target_x`：`sure` 的人排在前面
  - `APPEARANCE_RULES`（插在 `REMEMBER_ANCHOR` 前，`appearance` 为 True 时）：

```
- 状态里的装扮（“你自己”、好友名字后面、陌生人后面的颜色和样子）是看图猜的，可能不准；“像小明”是没看到名字、按外观认的，别当成一定是他。
- 提装扮像玩家那样随口提：夸、吐槽、问在哪换的；别报一长串。好友换了装扮可以说一句，别每次见面都念。
- “你自己”那行是你身上穿的，有人问你穿的什么就照着答。
```

- [ ] **Step 1: 写失败的测试**

```python
def test_stranger_back_is_background_event(clock):
def test_outfit_notes_go_to_ledger_and_change_event(clock):
    # pop_outfits 给 changed 再给 described → ledger 有两套、事件 "懒洋洋大王换了装扮：上次是「白斗篷」，现在「粉斗篷」"，kind 在 BACKGROUND
def test_status_shows_my_look_and_friend_look(clock):
    assert "你自己：白色樱花发型" in s and "懒洋洋大王（粉色长斗篷·" in s
def test_status_unchanged_without_appearance(clock):
def test_look_person_finds_stranger_by_id_and_maybe_friend(clock):
def test_prompt_appearance_rules_only_when_enabled():
    assert APPEARANCE_RULES not in brain_prompt(ReplyConfig(), None)
    assert APPEARANCE_RULES in brain_prompt(ReplyConfig(), None, appearance=True)
```

- [ ] **Step 2: 跑** `pytest tests/test_brain_appearance_body.py tests/test_brain_prompt.py -q` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `pytest tests/test_brain_appearance_body.py tests/test_brain_prompt.py tests/test_brain_body.py tests/test_brain_events.py tests/test_brain_inner_body.py tests/test_brain_peek.py tests/test_brain_track.py -q` → 全过
- [ ] **Step 5: 提交** `feat(brain): 状态、事件、look_person 用上装扮和陌生人编号`

---

### Task 10: 接线（cli、管理面板开关）

**Files:**
- Modify: `src/skydango/cli.py`（`_scene_watcher`、`_run_brain`、`_perception`）
- Modify: `src/skydango/console/settings.py`（两个 `Field`）
- Test: `tests/test_cli_brain.py`、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: 前面所有任务
- Produces:
  - `_scene_watcher`：`cfg.appearance.enabled` 且感知层开着 → 建 `embedder = make_embedder(cfg.appearance)`、`AppearanceBook(cfg.appearance, embedder.key, keep=cfg.perception.keep)`、有 `run` 且 `save` 时 `CropSaver(run.path / "appearance", …)`，传给 `PerceptionWatcher`；感知层没开时 WARNING `"[appearance] 要配合 [perception] 用，现在没打开，不认装扮"`；`_perception`（`perception detect` / `view --model`）同样挂记忆簿（不挂 saver、不挂描述器）
  - `_run_brain`：`env` 有 `appearance` 属性且不为 None 时：`env.appearance.load_cards(ledger.all_outfits())`（有 ledger 才做）；`describe` 开着且不是沙盒（`world.name != "sandbox"`）→ 建 `Wardrobe(cfg.appearance, describe=lambda c: one_shot(wardrobe_command(base, cfg.appearance.describe_model), claude_vars, work / "wardrobe", c, cfg.appearance.describe_timeout), on_done=env.on_described, clock=clock)`，`env.wardrobe = wardrobe`，起线程 `wardrobe.run(stop)`；`brain_prompt(..., appearance=env 有记忆簿)`；`Ledger` 建时传 `outfit_keep=cfg.appearance.outfit_keep`
  - 管理面板：`Field("appearance.enabled", "认装扮", "按外观接回没读到名字的好友、认回来的陌生人；要配合 YOLO 感知层", "bool", "features")`、`Field("appearance.describe", "描述装扮", "让 Haiku 把团子和身边人的装扮写成一句话（花额度）", "bool", "features")`

- [ ] **Step 1: 写失败的测试**

```python
def test_scene_watcher_attaches_appearance_book(monkeypatch, tmp_path):
def test_appearance_without_perception_warns(caplog):
def test_run_brain_loads_cards_and_starts_wardrobe(...):  # 照 test_cli_brain 里现有 _run_brain 测试的假 world / 假 claude 写法
def test_console_has_appearance_switches():
```

- [ ] **Step 2: 跑** `pytest tests/test_cli_brain.py tests/test_console_settings.py -q` → 失败
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑** `pytest -q`（全量）→ 全过
- [ ] **Step 5: 提交** `feat(cli): 接上认装扮和描述器，管理面板加开关`

---

### Task 11: 可视化网页

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`COLORS` / `NAMES` 加 `maybe`；`draw` 里 `maybe` 用虚线；鼠标悬停显示 `desc`；`describe_env` 加"团子穿着"）
- Test: `tests/test_viewer.py`（或现有 viewer 测试文件）

**Interfaces:**
- Consumes: Task 4 `overlay()` 的 `kind = "maybe"`、`desc`；`my_look()`
- Produces: `COLORS.maybe = "#86efac"`、`NAMES.maybe = "按外观认的好友"`；`ctx.setLineDash` 对 `maybe` 用 `[6,4]`；画布 `mousemove`：鼠标落在某个有 `desc` 的框里时在框下方画一行描述；`describe_env` 在 `my_look()` 有值时加 `"团子穿着": …`

- [ ] **Step 1: 写失败的测试**：`describe_env` 带 `my_look` 的假 env 返回里有"团子穿着"；页面 HTML 含 `maybe:"#86efac"`
- [ ] **Step 2: 跑** → 失败
- [ ] **Step 3: 实现**；`view` 打开一张合成截图用浏览器看一眼虚线框和悬停（预览工具，截图存 `tmp/`）
- [ ] **Step 4: 跑** viewer 测试 → 通过
- [ ] **Step 5: 提交** `feat(viewer): 按外观认的好友画浅绿虚线、悬停看装扮`

---

### Task 12: 离线标定 `perception appearance-eval` + 文档

**Files:**
- Create: `src/skydango/vision/appearance_eval.py`
- Modify: `src/skydango/cli.py`（子命令 `appearance-eval`，参数 `source`、`--model`、`--embed`、`-o/--output`）
- Modify: `CLAUDE.md`（新增「认装扮（`[appearance]`）」一节、代码结构表三行、常用命令一行）、`docs/superpowers/specs/2026-10-01-appearance-design.md`（§6 改成 `outfit` 事件，见上面「对 spec 的一处修正」）
- Test: `tests/test_appearance_eval.py`（新）

**Interfaces:**
- Produces（纯函数，不碰 YOLO）：
  - `pair_scores(tracks: dict[str, list[np.ndarray]]) -> tuple[list[float], list[float]]`：键是身份（好友名或 `t<轨迹>`），返回（同一身份两两相似度，不同身份两两相似度）；同一身份最多取 50 个样本、不同身份每对身份最多 200 对（固定随机种子）
  - `suggest(same: list[float], diff: list[float], max_wrong: float = 0.02) -> dict`：`match` = 不同人相似度的 `1 - max_wrong` 分位数（取不小于它的最小 0.01 刻度）；`changed` = 同一身份相似度的 5% 分位数；`margin` 固定 0.05；带上两边的样本数和分位数
  - `replay(sequence: list[tuple[float, str, np.ndarray]], cfg: AppearanceConfig, hide_every: float = 5.0) -> dict`：按时间重放（身份已知），每隔 `hide_every` 秒把一段标签藏掉，用 `AppearanceBook.assign_friends` 判，统计 `right` / `wrong` / `missed`
  - `report_md(summary: dict) -> str`
- cli：对录像跑 `_perception` 拿到带名字的轨迹（用 Task 4 的感知层，`saver=None`），收集 `track.data["feat"]` 序列，写 `tmp/appearance-eval/<时间>/report.md` + `summary.json`，打印建议值；`wrong / (right + wrong) > 2%` 时报告里写"建议往严了调"

- [ ] **Step 1: 写失败的测试**

```python
def test_suggest_threshold_from_distributions():
    s = suggest(same=[0.9] * 100, diff=[0.5] * 98 + [0.8, 0.95])
    assert s["match"] == pytest.approx(0.81) and s["margin"] == 0.05
def test_replay_counts_right_wrong_missed():
def test_report_mentions_tighten_when_wrong_rate_high():
```

- [ ] **Step 2: 跑** `pytest tests/test_appearance_eval.py -q` → 失败
- [ ] **Step 3: 实现**；写 CLAUDE.md 一节（照「内心层」那节的写法：设计 / 计划链接、做了什么、**还没在真机上跑过、数字都是估的**、spec「真机验证」五步、`enabled = false` 完全照旧）和 spec 修正
- [ ] **Step 4: 跑** `pytest -q`（全量）→ 全过
- [ ] **Step 5: 提交** `feat(perception): appearance-eval 离线标定；docs: 认装扮`
- [ ] **Step 6: 合并进 main 并推送**（按 CLAUDE.md 的规矩）
