# 开关聊天面板时不断轨迹 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 聊天面板开 / 关时（2 秒的横移动画、面板盖住左边三分之一），好友轨迹不再断、不再冒假的"走动 / 走近"。

**Architecture:** 三处改动都在感知层 `vision/perception.py`，加一个身体 → 感知层的通知：
A. 面板开着时只过滤面板区域里的名字标签 / 气泡 / 圆圈，人物框留着；面板后面没挂过标签的人先不判陌生人。
B. `_pan_step` 改用整张缩略图（不只上半），只在面板开着时遮掉面板，不再遮人物框。
C. 面板开关（感知层自己看到面板标志翻转，或者 `PanelManager` 按 C 时经身体通知）之后 `panel_settle` 秒内按"镜头在动"处理：速度清零、不攒走近 / 运动历史。

**Tech Stack:** Python 3.13、OpenCV（`cv2.phaseCorrelate`）、pytest（合成画面 + 假检测器，不需要模拟器）

**Spec:** `docs/progress/2026-10-03-plan.md`「① 认人：结果」5b 节（录像分析和 A / B / C 的由来）；录像在 `tmp/record/panel-toggle-1003-{1,2}`（不进 git），分析脚本 `tmp/panel_toggle/`

## Global Constraints

- 跑代码在 worktree 里：`PYTHONPATH=src`，测试 `python -m pytest -q`（这台机器没有 `.venv`，用系统 Python）；全量测试放后台跑
- 新开关放 `PerceptionConfig`，默认开：`panel_people: bool = True`、`panel_settle: float = 2.5`（0 = 关）；同步写进 `config.example.toml` 的 `[perception]`
- 新开关进 `PANEL_SWITCHES = ("panel_people", "panel_settle")`（perception.py，紧挨 `TRACKING_SWITCHES`），`trackeval.SWITCHES = TRACKING_SWITCHES + PANEL_SWITCHES`：track-eval 的基线 = 全关。**不要**把它们加进 `TRACKING_SWITCHES`（`_resume` 用它判断要不要 `_pan_recheck`）
- 两个开关都关 + `track_pan = False` 时行为逐字照旧（现有测试 `test_panel_area_and_bottom_bar_are_ignored`、`test_all_switches_off_is_unchanged` 不改）
- 面板区域就是现在 `_filter` 用的 `self.log_roi`（`vision.log_roi`，默认 `[0, 0, 0.335, 0.855]`）
- 注释、日志、提交说明用中文；提交末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. 面板开着时一个**新**出现在面板后面的人（没挂过标签）：不该过 `stranger_after` 就判陌生人（看不到他头顶的标签）；面板关了、标签还是没有，才照常判 → Task 1 的 `test_untagged_player_under_panel_waits_for_panel_to_close`
2. 面板后面的黑影（`player_unlit`）不需要标签，照常判陌生人 → Task 1 的 `test_unlit_under_panel_still_stranger`
3. 面板里的聊天行"- 名字"不能被当名字标签读（OCR 会读出好友名，挂到面板后面的陌生人头上）→ 名字标签在面板里照旧丢，Task 1 的 `test_people_under_open_panel_are_kept` 断言 `ocr.calls == 0`
4. 转镜头时团子在屏幕上不动、背景在动：B 不遮人物框后，平移估计不能被团子拉成 0 → Task 2 的 `test_pan_full_frame_ignores_static_center_person`
5. 面板开关后 `panel_settle` 内大脑 / 身体照常拿得到 `people()`（只是没有运动方向），不是整个暂停 → Task 3 的 `test_panel_flip_quiets_motion_only` 断言 `nearby()` 照旧

---

### Task 1: A — 面板后面的人留着

**Files:**
- Modify: `src/skydango/config.py`（`PerceptionConfig`，`edge_band` 那组下面加 `panel_people`）
- Modify: `src/skydango/vision/perception.py`（`_filter` ≈ 1451 行；陌生人判定 ≈ 796 行；`PANEL_SWITCHES` 常量 ≈ 98 行）
- Modify: `src/skydango/vision/trackeval.py:19`（`SWITCHES`）
- Modify: `config.example.toml`（`[perception]`，`motion_hold` 后面）
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces: `PerceptionConfig.panel_people: bool = True`；`perception.PANEL_SWITCHES: tuple[str, ...]`；`perception.PANEL_DROP = ("name_tag", "typing", "social_ring")`（面板开着时在面板里也要丢的类别）；`PerceptionWatcher._under_panel(box: Rect, width: int, height: int, panel_visible: bool) -> bool`

- [ ] **Step 1: 写失败的测试**（`tests/test_perception.py`，用文件里现成的 `FakeDetector` / `FakeOcr` / `watcher` / `player` / `tag`；面板区域 x < 643、y < 923）

```python
def test_people_under_open_panel_are_kept():
    det = FakeDetector()
    det.frames = [[player(200), tag(190, 110), ring(245)]]
    ocr = FakeOcr({110: "懒洋洋大王"})
    w = watcher(det, ocr)
    w.process(frame(), 0.0, panel_visible=True)
    assert [t.cls for t in w.last_tracks] == ["player"] and ocr.calls == 0  # 人留着，标签 / 圆圈照旧丢


def test_named_friend_survives_panel_opening():
    det, clock = FakeDetector(), Clock()
    w = watcher(det, FakeOcr({110: "懒洋洋大王"}), clock=clock)
    det.frames = [[player(200), tag(190, 110)]]
    for i in range(10):
        clock.t = i * 0.1
        w.process(frame(), clock.t, panel_visible=False)
    first = [t for t in w.last_tracks if t.cls == "player"][0].id
    det.frames = [[player(200), tag(190, 110)]]
    for i in range(10, 40):  # 面板开着 3 秒
        clock.t = i * 0.1
        w.process(frame(), clock.t, panel_visible=True)
    (p,) = [t for t in w.last_tracks if t.cls == "player"]
    assert p.id == first and p.data.get("name") == "懒洋洋大王" and not p.data.get("stranger")
    assert w.nearby(clock.t) == ["懒洋洋大王"]


def test_untagged_player_under_panel_waits_for_panel_to_close():
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    det.frames = [[player(200)]]
    for i in range(30):
        clock.t = i * 0.1
        w.process(frame(), clock.t, panel_visible=True)
    assert not w.last_tracks[0].data.get("stranger")
    clock.t = 3.0
    w.process(frame(), 3.0, panel_visible=False)  # 面板关了还是没标签：照常判
    assert w.last_tracks[0].data.get("stranger") is True


def test_unlit_under_panel_still_stranger():
    det, clock = FakeDetector(), Clock()
    w = watcher(det, clock=clock)
    det.frames = [[Detection("player_unlit", Rect(200, 400, 90, 220), 0.9)]]
    for i in range(15):
        clock.t = i * 0.1
        w.process(frame(), clock.t, panel_visible=True)
    assert w.last_tracks[0].data.get("stranger") is True


def test_panel_people_off_drops_people_under_panel():
    det = FakeDetector()
    det.frames = [[player(200)]]
    w = watcher(det, panel_people=False)
    w.process(frame(), 0.0, panel_visible=True)
    assert w.last_tracks == []
```

（`Detection` / `Rect` 若文件顶上没导入就补上；`Clock` 已在文件里。）

- [ ] **Step 2: 跑测试看它失败**

Run: `PYTHONPATH=src python -m pytest tests/test_perception.py -k "panel" -q`
Expected: 前四个 FAIL（`TypeError: ... unexpected keyword 'panel_people'` 或人物框被丢、`last_tracks` 为空），`test_panel_area_and_bottom_bar_are_ignored` 照旧 PASS

- [ ] **Step 3: 实现**

- `config.py`：`panel_people: bool = True  # 聊天面板开着时面板后面的人照样认（面板半透明，10-03 录像）；只丢面板里的名字标签 / 气泡 / 圆圈；false = 照旧整块丢`
- `perception.py`：`PANEL_DROP`、`PANEL_SWITCHES`；`_filter` 里面板那一条改成：中心在面板里且 `(not self.cfg.panel_people or det.cls in PANEL_DROP)` 才丢
- `_under_panel(box, width, height, panel_visible)`：`panel_visible and self.cfg.panel_people and _inside(_center(box), roi_rect(self.log_roi, width, height))`
- 点过火的人的 `is_stranger` 加一条 `and not self._under_panel(player.box, width, height, panel_visible)`（黑影那支不加）
- `trackeval.py`：`SWITCHES = TRACKING_SWITCHES + PANEL_SWITCHES`（`baseline()` 把 `panel_settle` 设成 `False` 即 0，Task 3 用）
- `config.example.toml`：`panel_people = true  # 聊天面板开着时面板后面的人照样认，只丢面板里的名字标签`

- [ ] **Step 4: 跑测试看它通过**

Run: `PYTHONPATH=src python -m pytest tests/test_perception.py tests/test_trackeval.py tests/test_perception_tracking.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py src/skydango/vision/perception.py src/skydango/vision/trackeval.py config.example.toml tests/test_perception.py
git commit -m "fix(perception): 聊天面板开着时面板后面的人照样认，没挂标签的先不判陌生人"
```

### Task 2: B — 平移估计用整张缩略图

**Files:**
- Modify: `src/skydango/vision/perception.py`（`_pan_step` ≈ 540 行和它在 `process` 里的调用）
- Test: `tests/test_perception_tracking.py`

**Interfaces:**
- Produces: `PerceptionWatcher._pan_step(frame: np.ndarray, panel_visible: bool) -> tuple[float, float] | None`（去掉 `dets` 参数；返回值和副作用 `self._pan` 不变）

- [ ] **Step 1: 写失败的测试**（用文件里的 `make` / `run` / `player` / `body` / `_texture`）

```python
def _sky_over_ground(seed=0):
    """上半是一片夜空（几乎没纹理），下半是地面：10-03 录像里开面板时只看上半估得偏小。"""
    img = _texture(seed)
    img[:540] = 40
    return img


def test_pan_full_frame_when_top_is_sky():
    base = _sky_over_ground()
    w, det = make(track_predict=False)
    run(w, det, [player(300)], 0.0, 0.3, img=base)
    run(w, det, [player(460)], 0.3, 0.4, img=np.roll(base, 160, axis=1))
    assert w.last_shift is not None and w.last_shift[0] == pytest.approx(160, abs=8)


def test_pan_full_frame_ignores_static_center_person():
    """转镜头：团子站在画面中间不动、背景在动，不能被团子拉成 0。"""
    base = _texture(3)
    me = _texture(4)[300:900, 800:1100]

    def img(shift):
        out = np.roll(base, shift, axis=1)
        out[300:900, 800:1100] = me
        return out

    w, det = make(track_predict=False)
    run(w, det, [player(1400)], 0.0, 0.3, img=img(0))
    run(w, det, [player(1480)], 0.3, 0.4, img=img(80))
    assert w.last_shift[0] == pytest.approx(80, abs=8)


def test_pan_skips_open_panel():
    rng = np.random.default_rng(9)
    base = _texture(6)

    def img(shift):
        out = np.roll(base, shift, axis=1)
        out[:, :640] = rng.integers(0, 255, (1080, 640, 3), dtype=np.uint8)  # 面板那块每帧乱变
        return out

    w, det = make(track_predict=False)
    for i, t in enumerate((0.0, 0.1, 0.2)):
        det.frames = [[player(1400)]]
        w.process(img(0), t, True)
    det.frames = [[player(1520)]]
    w.process(img(120), 0.3, True)
    assert w.last_shift[0] == pytest.approx(120, abs=8)
```

- [ ] **Step 2: 跑测试看它失败**

Run: `PYTHONPATH=src python -m pytest tests/test_perception_tracking.py -k "pan" -q`
Expected: `test_pan_full_frame_when_top_is_sky` FAIL（`last_shift is None`：上半纯色、std < 1）；另两个可能已经 PASS（它们是护栏，改完不能挂）

- [ ] **Step 3: 实现** `_pan_step(frame, panel_visible)`：1/8 灰度缩略图**整张**（去掉 `small[: small.shape[0] // 2]`），遮罩只在 `panel_visible` 时去掉左三分之一，不再遮人物 / 标签框；`process` 里改调用；docstring 写明为什么不遮人（10-03 录像：遮人物框的均值块是两帧里一样的静止方块，把相位相关往 0 拉，开面板时累计只估出 314~368 px，不遮 363~383，真值约 420）

- [ ] **Step 4: 跑测试看它通过**

Run: `PYTHONPATH=src python -m pytest tests/test_perception_tracking.py tests/test_perception.py tests/test_track_motion.py -q`
Expected: 全 PASS（`test_pan_feeds_tracker`、`test_motion_uses_pan_compensated_x`、`test_camera_turn_in_hold_*` 照旧）

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/perception.py tests/test_perception_tracking.py
git commit -m "fix(perception): 画面平移用整张缩略图、不遮人物框（只看上半时夜空没纹理，开面板估得偏小）"
```

### Task 3: C — 面板开关之后安静一会儿

**Files:**
- Modify: `src/skydango/config.py`（`panel_settle`，紧跟 `panel_people`）
- Modify: `src/skydango/vision/perception.py`（`camera_moved` ≈ 532 行；`process` 开头记面板标志翻转；`__init__` 加 `self._last_panel: bool | None = None`）
- Modify: `src/skydango/chat/panel.py`（`PanelManager.__init__` 加 `self.on_press`；`_press` ≈ 326 行）
- Modify: `src/skydango/brain/body.py`（`__init__` 里 `self.panel` 建好之后接 `on_press`）
- Modify: `config.example.toml`
- Test: `tests/test_perception_tracking.py`、`tests/test_panel.py`、`tests/test_body_camera_moved.py`

**Interfaces:**
- Consumes: Task 1 的 `PANEL_SWITCHES`（已含 `panel_settle`）
- Produces: `PerceptionConfig.panel_settle: float = 2.5`；`camera_moved(at, "panel")`：`panel_settle > 0` 时 `_quiet_until = max(_quiet_until, at + panel_settle)`、`tracker.calm(同一时刻)`；`PanelManager.on_press: Callable[[float], None] | None`（按 C 时以按键时刻 `self._pressed_at` 调用，出错只记日志）

- [ ] **Step 1: 写失败的测试**

`tests/test_perception_tracking.py`：

```python
def test_camera_moved_panel_quiets_for_panel_settle():
    w, det = make()
    t = run(w, det, walk, 0.0, 1.0)
    w.camera_moved(t, "panel")
    t = run(w, det, walk, t, t + 2.0)
    assert not body(w).data.get("hist") and body(w).vx == 0  # 2.5 秒还没过
    t = run(w, det, walk, t, t + 1.0)
    assert body(w).data.get("hist") and body(w).vx != 0


def test_panel_flip_quiets_motion_only():
    """站着的好友，面板一开画面右移 2 秒（黑帧估不出平移）：不报往右走，人照样在身边。"""
    for settle, moved in ((2.5, False), (0.0, True)):
        w, det = make(panel_settle=settle)
        t = run(w, det, [player(800), tag(800)], 0.0, 2.0)
        x = 800
        for i in range(20):
            x += 20
            det.frames = [[player(x), tag(x)]]
            w.process(frame(), round(t + i * 0.1, 3), True)
        end = round(t + 2.0, 3)
        assert w.nearby(end) == [NAME]
        assert (body(w).data.get("motion") == "往右走") is moved
```

（`walk` / `NAME` / `tag` 是文件里现成的；`run` 传的 `panel_visible` 恒为 False，上面直接调 `process`。）

`tests/test_panel.py`：

```python
def test_press_reports_to_on_press():
    m, device = manager([False, True])
    m.clock = lambda: 12.5
    seen = []
    m.on_press = seen.append
    m.ensure_open()
    assert seen == [12.5]


def test_on_press_error_does_not_break_press():
    m, device = manager([False, True])
    m.on_press = lambda at: 1 / 0
    assert m.ensure_open() is True and device.calls == [("hw_key", 46)]
```

`tests/test_body_camera_moved.py`：

```python
def test_panel_press_tells_perception_without_blocking_attention(clock):
    env = MovedEnv()
    b, _, _, _ = body(clock, live=True, panel_mode="auto", env=env)
    env.moved.clear()  # body() 里 panel.start 的动静不算
    b.panel.ensure_open()
    assert kinds(env) == ["panel"] and env.moved[0][0] == clock()
    assert b._camera_moved_at == float("-inf")  # 注意力不让开（同注意力自己的 nudge）
```

- [ ] **Step 2: 跑测试看它失败**

Run: `PYTHONPATH=src python -m pytest tests/test_perception_tracking.py tests/test_panel.py tests/test_body_camera_moved.py -k "panel" -q`
Expected: FAIL（`panel_settle` 不认识、`on_press` 没被调、`kinds(env) == []`）

- [ ] **Step 3: 实现**

- `config.py`：`panel_settle: float = 2.5  # 聊天面板开 / 关后画面横移的动画约 2 秒（10-03 录像）：这么久内速度清零、不攒走近 / 运动历史；0 = 关`
- `perception.camera_moved`：`kind == "panel"` 单独一支（`panel_settle > 0` 才动），其余照旧
- `process` 开头（`_filter` 之前）：`self._last_panel` 不是 None 且和这一帧 `panel_visible` 不同 → `self.camera_moved(now, "panel")`；然后记下这一帧的标志
- `PanelManager`：`self.on_press: Callable[[float], None] | None = None`；`_press` 按完键、记下 `_pressed_at` 后调用（`try/except Exception: log.exception(...)`）
- `Body.__init__`：`self.panel.on_press = lambda at: self._notify_camera("panel", at)`（用 `_notify_camera`，**不是** `_camera_moved`：不设 `_camera_moved_at`）
- `config.example.toml`：`panel_settle = 2.5  # 聊天面板开 / 关后这么多秒内不攒运动方向 / 走近（横移动画约 2 秒）；0 = 关`

- [ ] **Step 4: 跑测试看它通过**

Run: `PYTHONPATH=src python -m pytest tests/test_perception_tracking.py tests/test_panel.py tests/test_body_camera_moved.py tests/test_brain_body.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/config.py src/skydango/vision/perception.py src/skydango/chat/panel.py src/skydango/brain/body.py config.example.toml tests/test_perception_tracking.py tests/test_panel.py tests/test_body_camera_moved.py
git commit -m "fix(perception): 聊天面板开关后 2.5 秒不攒运动方向 / 走近（按 C 时身体先通知）"
```

### Task 4: 录像回放测试

**Files:**
- Create: `tests/panel_replay_extract.py`（照 `tests/light_replay_extract.py`：要本机录像和 YOLO，不在 pytest 里跑）
- Create: `tests/data/panel_replay.json`
- Create: `tests/test_panel_replay.py`

**Interfaces:**
- Consumes: Task 1~3 的开关；`trackeval.subsample`、`compare.timed_files`、`cli._panel_open`
- Produces: `panel_replay.json` = `{"1": {"frames": [[t, panel(0/1), [[cls, x, y, w, h, score], ...]], ...]}, "2": {...}}`：两段录像各按 14 帧 / 秒抽帧（`subsample`），检测器 `conf = 0.2`，`t` 保留 3 位小数

- [ ] **Step 1: 写抽取脚本并生成数据**

`panel_replay_extract.py`：读 `tmp/record/panel-toggle-1003-{1,2}`，`make_detector(cfg.perception.model, …, conf=0.2, …)` 每帧检测，`_panel_open(cfg, im)` 记面板标志，写 `tests/data/panel_replay.json`（`ensure_ascii=False`，一帧一行，控制在 200 KB 以内）。

Run（主目录下，读主目录 `config.toml` 和 `models/`）：`PYTHONPATH=<worktree>/src python <worktree>/tests/panel_replay_extract.py --out <worktree>/tests/data/panel_replay.json`
Expected: 打印两段各 420 / 450 帧

- [ ] **Step 2: 写回放测试**（`FakeDetector` 按帧给 `Detection`；OCR 一律读成 `懒洋洋大王`（这两段里只有他挂名字标签）；帧是黑图，平移估不出，只测 A 和 C）

```python
@pytest.mark.parametrize("on", [True, False])
def test_friend_track_survives_panel_opening(on):
    """第 2 段：19.50 s、27.10 s 面板标志翻成开着，好友站在面板后面（10-03 断在这两帧）。"""
    ids = replay("2", panel_people=on)  # {t: 这一帧挂着 懒洋洋大王 名字的人物轨迹 id 或 None}
    for a, b in ((19.4, 22.9), (27.0, 30.9)):
        same = at(ids, a) is not None and at(ids, a) == at(ids, b)
        assert same is on


@pytest.mark.parametrize("settle", [2.5, 0.0])
def test_standing_friend_not_walking_when_panel_toggles(settle):
    """第 1 段：好友一直站着，开关 10 次面板。"""
    motions = replay_motions("1", panel_settle=settle)  # 好友轨迹出现过的所有 motion
    assert ({"往左走", "往右走"} & motions == set()) is (settle > 0)
```

helper：`_watch(seg, **cfg)` 建 `PerceptionWatcher`（同 `test_perception_tracking.make`，`far_crops=0`），逐帧 `det.frames = [dets]; w.process(np.zeros((1080, 1920, 3), np.uint8), t, bool(panel))`，每帧回调一次；
`replay(seg, **cfg) -> dict[float, int | None]` 记下每帧 `data.get("name") == "懒洋洋大王"` 的 player 轨迹 id；`replay_motions(seg, **cfg) -> set[str]` 收集这些轨迹出现过的 `data.get("motion")`；`at(ids, t)` 取 ≤ t 的最近一帧。

- [ ] **Step 3: 跑测试**

Run: `PYTHONPATH=src python -m pytest tests/test_panel_replay.py -q`
Expected: 4 个 PASS（开关打开的两种对、关掉的两种复现 10-03 的问题）。若"关掉"那两种没复现，说明回放没对上录像，先查抽取数据再说，不改断言凑数

- [ ] **Step 4: 提交**

```bash
git add tests/panel_replay_extract.py tests/data/panel_replay.json tests/test_panel_replay.py
git commit -m "test(perception): 开关聊天面板的两段录像回放（面板后面的好友不断轨迹、站着的不报走动）"
```

### Task 5: 录像验收、文档、合并

**Files:**
- Modify: `docs/progress/2026-10-03-plan.md`（5b 末尾记结果）
- Modify: `CLAUDE.md`（「YOLO 感知层」里加一条：面板开着时面板后面的人照样认、平移用整帧、`panel_settle`）

- [ ] **Step 1: 全量测试（后台）**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: 全 PASS

- [ ] **Step 2: track-eval 对比**（主目录下、PYTHONPATH 指 worktree 的 src；输出到 `tmp/track-eval/1003p-<录像名>`）

- 两段新录像：`--fps 14` 和 `--fps 6.5`；和 `tmp/track-eval/panel-toggle-1003-*`（改之前）比：第 2 段 14 fps 当前配置的确认冤枉 4 → 应下降、断开 8 → 应下降；第 1 段运动方向里不再有"往左走 / 往右走"
- 之前 7 段（`play-1001-1`、`gesture-bow-1`、`gesture-wave-1`、`q-call-20260930`、`-b`、`-c` 用 `--fps 6.5`，`walkaway-1002-1` 用 `--fps 8`），和 `tmp/track-eval/1003b-*` 比：断开、假走开、确认冤枉、接回错**都不能变多**
- 某段变差且查明是 B 的锅（断开原因变成"画面平移"、或转镜头的段落轨迹接错）：`_pan_step` 改成"整帧 + 遮这一帧的人物框"再比一次（10-03 实测这样开面板时累计 314~368 px，比只看上半好得多），两种结果都记进进度文档

- [ ] **Step 3: 写文档**：进度文档 5b 末尾加「修了」一段（A / B / C 各一句 + track-eval 前后数字 + 还要真机看的：面板开着时站在左边的好友 status 还在、`run --view` 里面板后面的人有框）；CLAUDE.md 加一条

- [ ] **Step 4: 提交、合并、推送**

```bash
git add docs/progress/2026-10-03-plan.md CLAUDE.md
git commit -m "docs: 开关面板不断轨迹的修法和录像验收"
git fetch origin && git merge origin/main
PYTHONPATH=src python -m pytest -q
git push origin HEAD:main
```
