# 感知层二期（代码部分）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把二期设计里能在没有模型、没有真机的情况下写完的部分全部做完：`#spin` / `camera spin`、环绕扫描 + 转圈认团子、检测结果交给眼睛、认说话人（`typing`）、距离和走向。真机标定（§0）、补数据重训（`typing`）和 §6 验收留给 GPU 机器。

**Architecture:** 纯计算（方位角、8 方位、合并、认团子、距离分档）放新模块 `vision/sweep.py`，全部是纯函数；`PerceptionWatcher` 只负责跑检测 / OCR 把结果喂给它，另外在每帧 `process()` 里加 `typing` 关联、走向判断。镜头转圈在 `brain/camera.py`（`Camera.spin`），结果落盘在 `runlog.py`。身体 / Agent / 工具只通过 duck typing 用新接口（`hasattr(env, "sweep")` 等），`[perception] enabled = false` 时行为不变。

**Tech Stack:** Python 3.11+，numpy、opencv；测试 pytest（合成画面 + 假检测器 / 假 OCR / 假设备 / 假时钟）。

**Spec:** `docs/superpowers/specs/2026-09-28-perception-phase2-design.md`（总纲 `…-perception-yolo-architecture-v0.2.md` §14；`#spin` 的细节在 `2026-09-27-owner-commands-design.md` §6）

## Global Constraints

- `[perception] enabled = false` 时身体 / Agent / 眼睛 / 工具说明的行为和现在完全一样（新代码都用 `hasattr(env, ...)` 判断）
- YOLO 类别表只往后追加：`["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]`（类别名以模型元数据为准，旧模型不会出 `typing`）
- 新配置 `[spin]`：`seconds_per_turn = 2.0`、`fps = 15.0`、`max_turns = 2`、`min_interval = 10.0`、`hfov = 90.0`（**待 §0 标定**）、`merge_deg = 30.0`、`self_motion = 0.03`
- `[perception]` 新增：`typing_window = 8.0`、`near = 0.8`、`far = 0.4`（人物框高 ÷ 团子框高，**待标定**）、`self_height = 0.2`（没有团子框时假定团子框高占屏高的比例，**待标定**）、`approach_window = 1.5`、`approach_grow = 0.25`、`approach_cooldown = 60.0`、`approach_strangers = true`
- 用户已确认（写回设计 §7）：认说话人**只在恰好一个候选时加注**；`approach` 也报陌生人，但加开关 `approach_strangers`
- 身体**不自动转圈**：只在大脑调 `look_around`、主人发 `#spin` 时转；`#spin` 只加普通模式，大脑模式不加
- `#spin` 在 dry-run 下照样转、照样存图，只有确认回复不发；大脑的 `look_around` 在 dry-run 下不转（和现在的 `camera` 工具一致）
- 文件名不能带 `:`（Windows）：转圈存到 `spin/<HHMMSS>/`，帧文件 `000_0.07s.jpg`
- 眼睛不因 `approach` 自动看（`AUTO_LOOK_KINDS` 不加）
- 注释、日志、CLI 输出、给大脑 / 主人的文字用中文，风格照现有代码；测试命令 `python -m pytest -q`

## Review Focus

1. **帧太少的扫描**（dry-run 只有当前一帧、转圈中截图失败只剩几张）→ 不认团子（`find_self` 少于 5 帧返回空），方位只按画面位置算，不崩。测试在 Task 4（`find_self` 帧数不够）和 Task 6（dry-run 单帧）。
2. **牵着手转圈**：对方紧挨团子、在每一帧都在画面中间 → 按方位合并会把它摊到 8 个方向上。几乎不动的链上的名字标签单独记成"身边"，不参与方位合并。测试在 Task 5。
3. **方位在 0° / 360° 附近**：同一个陌生人一帧在 355°、一帧在 5° → 合成一个，方位取圆周平均（≈ 0°，"前"），不是 180°。测试在 Task 4。
4. **暂停前后的框高拼在一起**（转镜头、开好友树之后同一条轨迹的框高突变）→ 不报 `approach`：恢复时清掉每条轨迹的框高历史。测试在 Task 11。
5. **转到一半出错**（截图抛异常、Ctrl+C）→ 方向键松开、面板重开、感知暂停解除；`#spin` 回一句"没转成：…"，主循环不崩。测试在 Task 1（松键、重开）和 Task 2（Agent 回报错）。

---

### Task 1: `Camera.spin()` + `[spin]` 配置 + 存转圈结果

**Files:**
- Modify: `src/skydango/brain/camera.py`、`src/skydango/config.py`、`config.example.toml`、`src/skydango/runlog.py`
- Test: `tests/test_brain_camera.py`、`tests/test_runlog.py`、`tests/test_config.py`

**Interfaces:**
- Produces:
  - `SpinConfig`（dataclass，字段见 Global Constraints），`Config.spin: SpinConfig`
  - `SpinResult`（`brain/camera.py` 的 dataclass）：`before: np.ndarray`、`frames: list[tuple[float, np.ndarray]]`（按住后第几秒, 图）、`after: np.ndarray`、`seconds: float`（实际按住多久）、`panel_reopened: bool`（面板原来关着也算 True）、`blackout: bool`（有整屏黑的帧，`images.is_black`）
  - `Camera(device, step, panel_visible, panel_key, sleep=time.sleep, clock=time.monotonic)`；`Camera.spin(capture: Callable[[], np.ndarray], turns: int = 1, seconds_per_turn: float = 2.0, fps: float = 15.0) -> SpinResult`；`_ready()` 改成 `yield was_open`（现有调用不受影响）
  - `runlog.write_spin(folder: Path, result: SpinResult, turns: int) -> dict`（写 `before.jpg`、`after.jpg`、`{i:03d}_{t:.2f}s.jpg`、`summary.json`，返回 summary）；`RunDir.save_spin(result, turns, stamp: str) -> Path`（写到 `<run>/spin/<stamp>/`）
  - summary 的键：`turns`、`seconds`（保留 2 位）、`frames`、`fps`（帧数 ÷ 秒数，秒数为 0 时 0）、`panel_reopened`、`blackout`、`drift`（`difference(thumb(before), thumb(after))`，保留 3 位）

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_brain_camera.py —— 假时钟：sleep 推进时钟
def spin_cam(panel=True, capture=None):
    c, dev, state = cam(panel)
    clock = {"t": 0.0}
    c.clock = lambda: clock["t"]
    c.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    return c, dev, state

def test_spin_holds_right_and_captures_at_fps():
    c, dev, state = spin_cam()
    r = c.spin(lambda: scene(), turns=1, seconds_per_turn=1.0, fps=10)
    assert [round(t, 2) for t, _ in r.frames] == [round(0.1 * i, 2) for i in range(10)]
    assert dev.calls[0] == ("hw_key", 46) and dev.calls[-1] == ("hw_key", 46)
    assert dev.calls.count(("hw_down", 106)) == 1 and dev.calls.count(("hw_up", 106)) == 1
    assert r.seconds == pytest.approx(1.0) and r.panel_reopened and not r.blackout
    assert c.describe() == "原位"  # 整圈不改偏移

def test_spin_releases_key_and_reopens_panel_on_error():
    c, dev, state = spin_cam()
    n = {"i": 0}
    def capture():
        n["i"] += 1
        if n["i"] == 3:
            raise RuntimeError("截图失败")
        return scene()
    with pytest.raises(RuntimeError):
        c.spin(capture, seconds_per_turn=1.0, fps=10)
    assert ("hw_up", 106) in dev.calls and state["panel"] is True

def test_spin_turns_multiply_duration():
    c, _, _ = spin_cam(panel=False)
    r = c.spin(lambda: scene(), turns=2, seconds_per_turn=1.0, fps=5)
    assert len(r.frames) == 10

# tests/test_runlog.py
def test_save_spin_writes_frames_and_summary(tmp_path):
    ...  # SpinResult(before=黑图, frames=[(0.0, 图), (0.07, 图)], after=同 before, seconds=0.14, panel_reopened=False, blackout=True)
    folder = run.save_spin(result, 1, "101530")
    assert folder == run.path / "spin" / "101530"
    assert {p.name for p in folder.iterdir()} == {"before.jpg", "after.jpg", "000_0.00s.jpg", "001_0.07s.jpg", "summary.json"}
    s = json.loads((folder / "summary.json").read_text("utf-8"))
    assert s["frames"] == 2 and s["drift"] == 0.0 and s["panel_reopened"] is False and s["blackout"] is True

# tests/test_config.py
def test_spin_section(tmp_path):  # [spin] hfov = 75 → cfg.spin.hfov == 75，其余默认值 2.0 / 15.0 / 2 / 10.0 / 30.0 / 0.03
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_brain_camera.py tests/test_runlog.py tests/test_config.py -q`
Expected: FAIL（`spin` / `save_spin` / `cfg.spin` 不存在）

- [ ] **Step 3: 实现**

`spin()` 在 `with self._ready() as was_open:` 里：先 `before = capture()`，按下 right，按住期间循环截图（每张记 `clock() - start`，按 `1/fps` 补齐间隔，时间到 `turns × seconds_per_turn` 就停），`finally` 松键；`sleep(0.3)` 后截 `after`。出 `_ready` 后 `panel_reopened = not was_open or self.panel_visible()`。不改 `offset`。
`config.example.toml` 加 `[spin]` 一节（每项一行注释，`hfov` 注明"待 camera spin 标定"）。

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/camera.py src/skydango/config.py config.example.toml src/skydango/runlog.py tests/
git commit -m "feat(camera): spin() 按住方向键转一圈、按 fps 截图；[spin] 配置；转圈结果落盘"
```

### Task 2: `#spin` 主人命令（普通模式）

**Files:**
- Modify: `src/skydango/chat/commands.py`、`src/skydango/agent.py`
- Test: `tests/test_commands.py`、`tests/test_agent.py`

**Interfaces:**
- Consumes: `Camera.spin`、`SpinResult`、`RunDir.save_spin`（Task 1）
- Produces:
  - `CommandRouter(store, on_pause, status, spin: Callable[[int], str] | None = None, max_turns: int = 2)`；`_KNOWN` 加 `"spin"`
  - `Agent(..., camera=None)`；`Agent._spin(turns: int) -> str`（Task 6 会在这里接扫描结果）

文案（照主人命令设计 §6）：`"这次没开视角控制"`、`"格式不对，是 #spin 或 #spin 2"`、`"刚转过，等 {N} 秒"`（N 向上取整）、`"画面黑着，转不了"`、`"转完了，{秒:.1f} 秒 {张} 张"`、`"（中途画面黑了）"`、`"（聊天面板没打开，要手动按 C）"`、出错 `"没转成：{异常第一行}"`。整句超过 `reply.max_chars` 截断。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_commands.py
@pytest.mark.parametrize("text,turns", [("#spin", 1), ("#spin 2", 2), ("#spin 5", 2), ("#spin 0", 1)])
def test_spin_turns_are_clamped(text, turns):
    got = []
    r = CommandRouter(None, lambda p: None, lambda: "", spin=lambda n: got.append(n) or "转完了", max_turns=2)
    assert r.handle(text) == "转完了" and got == [turns]

def test_spin_bad_number():   # "#spin abc" → "格式不对，是 #spin 或 #spin 2"
def test_spin_without_camera():  # spin=None → "这次没开视角控制"

# tests/test_agent.py（FakeCamera.spin 返回 SpinResult，记下调用次数）
def test_owner_spin_turns_camera_and_confirms():
    # 主人发 "#spin" → camera.spin 被调一次；agent.sent[-1] 以 "转完了，" 开头；pending 为空；limiter 没记
def test_second_spin_within_min_interval_is_refused():  # 同一时刻第二次 → "刚转过，等 10 秒"，spin 没再调
def test_spin_in_dry_run_still_turns_but_does_not_send():  # dry_run：spin 调了、sender 没发
def test_spin_refused_on_black_screen():  # 截图全黑 → "画面黑着，转不了"
def test_spin_error_is_reported():  # FakeCamera.spin 抛 RuntimeError("截图失败") → "没转成：截图失败"，step() 不抛
def test_spin_saves_to_run_dir(tmp_path):  # 有 run 时 run.path/"spin" 下多一个目录
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_commands.py tests/test_agent.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

`Agent._spin`：限频（`self._last_spin`，初值 `-inf`）→ 截一张判黑屏 → 记下时间 → `with self._held("camera"):` 调 `camera.spin(self.device.screenshot, turns, spin.seconds_per_turn, spin.fps)` → 有 `run_dir` 就 `save_spin(result, turns, time.strftime("%H%M%S"))` → 拼回复。异常在 `_spin` 里接住（`log.exception` + 报错文案）。`Agent.__init__` 里 `spin=self._spin if camera is not None else None`。

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git commit -am "feat(commands): #spin 主人命令——转一圈、存截图、回转了几秒几张"
```

### Task 3: 命令行 `camera spin` + 普通模式建 `Camera`

**Files:**
- Modify: `src/skydango/cli.py`
- Test: `tests/test_cli_brain.py`（或新建 `tests/test_cli_camera.py`，照现有 CLI 测试的假设备写法）

**Interfaces:**
- Consumes: `Camera.spin`、`runlog.write_spin`（Task 1）；`Agent(camera=...)`（Task 2）
- Produces: `cmd_camera(cfg, args)`；子命令 `camera spin [--turns N] [--seconds S] [-o 目录]`（默认 `tmp/spin/<HHMMSS>`）；`_camera(cfg, dev, reader) -> Camera`（普通模式和大脑模式共用，替换 `_run_brain` 里现有的构造）

- [ ] **Step 1: 写失败的测试**

```python
def test_camera_spin_saves_frames(tmp_path, monkeypatch):
    # 假设备 + 假时钟；main(["camera", "spin", "--seconds", "0.2", "-o", str(tmp_path / "s")])
    # → tmp_path/"s"/"summary.json" 存在；stdout 里有 "张" 和 "summary.json" 的路径
def test_camera_spin_turns_clamped(...):  # --turns 5 → summary["turns"] == cfg.spin.max_turns
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_cli_camera.py -q`
Expected: FAIL（没有 `camera` 子命令）

- [ ] **Step 3: 实现**

`cmd_camera`：建设备、`_build_reader`、`_camera`，`--turns` 夹到 `[1, spin.max_turns]`，`--seconds` 覆盖 `seconds_per_turn`；转完 `write_spin`，打印"转了 X 秒 N 张（实际 F fps），面板重开：是/否，转前转后差异 D → 目录"。`_run_agent` 建 `_camera(...)` 传给 `Agent`。

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git commit -am "feat(cli): camera spin 标定命令；普通模式也建 Camera（#spin 用）"
```

### Task 4: `vision/sweep.py` —— 方位、合并、认团子、距离（纯函数）

**Files:**
- Create: `src/skydango/vision/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Produces:
  - `DIRECTIONS = ["前", "右前", "右", "右后", "后", "左后", "左", "左前"]`；`BESIDE = "身边"`；显示用 `SIDE_TEXT = {"身边": "身边", "前": "正前方", "右前": "右前方", "右": "右边", "右后": "右后方", "后": "正后方", "左后": "左后方", "左": "左边", "左前": "左前方"}`（显示顺序：身边在最前，然后按 `DIRECTIONS`）
  - `STRANGER_WHO = "陌生人"`、`UNLIT_WHO = "陌生人（没点火）"`、`UNKNOWN_WHO = "没认出名字的人"`
  - `bearing(t: float, cx: float, width: int, seconds_per_turn: float, hfov: float) -> float`（设计 §1.2 的公式，取模 360）
  - `direction(deg: float) -> str`（每个方向占 45°，"前" = 337.5°~22.5°）
  - `distance(box_h: float, ref_h: float, near: float, far: float) -> str`（`ratio = box_h / ref_h`：≥ near "近"，≥ far "中"，否则 "远"）
  - `@dataclass Sighting(degrees: float, who: str, frame: int, height: float | None = None, beside: bool = False)`
  - `SweepEntry`、`SweepResult` 同设计 §1.2（`direction` 可以是 `"身边"`）；`SweepResult.text() -> str`
  - `merge(sightings: list[Sighting], merge_deg: float, ref_h: float, near: float, far: float) -> list[SweepEntry]`
  - `@dataclass SelfFound(box: Rect | None, static: set[tuple[int, int]], per_frame: dict[int, Rect])`
  - `find_self(boxes: list[list[Rect]], width: int, self_motion: float, min_ratio: float = 0.8, link_iou: float = 0.3, min_frames: int = 5) -> SelfFound`

合并规则：`beside` 的好友 sighting 按名字合成一条 `direction = "身边"`；其他好友按名字合并，方位取**圆周平均**；`STRANGER_WHO` / `UNLIT_WHO` 一起按方位贪心聚类（和聚类圆周平均相差 < `merge_deg` 并进去），类别取出现帧数最多的；`UNKNOWN_WHO` 单独聚类。`frames` = 不同帧号的个数；`distance` 用各 sighting 里最大的 `height`，全是 None 时为 None。

`text()`：按显示顺序每个方向一段 `"{SIDE_TEXT}：{内容}"`，段之间 `"；"`；内容里好友名字（有距离时加 `"（近）"`）在前，然后 `"{n} 个陌生人"`（n 含没点火的，有没点火的加 `"（{u} 个没点火）"`），然后 `"{k} 个没认出名字的人"`，用 `"、"` 连。没有条目 → `"转了一圈，身边没看到别人"`。

认团子：按帧顺序把框接成链（和链上最后一个框 IoU ≥ `link_iou`，贪心取最大），每条链算出现帧占比、移动量（各帧框中心到平均中心的平均距离 ÷ `width`）、平均中心离画面中心的距离；占比 ≥ `min_ratio` 且移动量 < `self_motion` 的是"几乎不动的链"，全部框进 `static`。取离中心最近的一条；另有一条不动的链、两条平均中心相距 < 该链框宽中位数 → `box = None`（牵手分不开）。否则 `box` = 各帧框 x / y / w / h 的中位数，`per_frame` = 那条链每帧的框。帧数 < `min_frames` → `SelfFound(None, set(), {})`。

- [ ] **Step 1: 写失败的测试**

```python
def test_bearing_and_direction():
    assert bearing(0.0, 960, 1920, 2.0, 90) == 0.0
    assert bearing(0.5, 960, 1920, 2.0, 90) == 90.0
    assert bearing(0.0, 1920, 1920, 2.0, 90) == 45.0
    assert bearing(0.0, 0, 1920, 2.0, 90) == 315.0
    assert [direction(d) for d in (0, 22, 23, 90, 180, 200, 270, 337.4, 338, 359)] == \
        ["前", "前", "右前", "右", "后", "后", "左", "左前", "前", "前"]

def test_friend_sightings_merge_by_name():
    s = [Sighting(80, "懒洋洋大王", i, 200) for i in range(3)] + [Sighting(100, "懒洋洋大王", 3, 220)]
    [e] = merge(s, 30, ref_h=200, near=0.8, far=0.4)
    assert (e.who, e.direction, e.frames, e.distance) == ("懒洋洋大王", "右", 4, "近")
    assert e.degrees == pytest.approx(85, abs=1)

def test_strangers_apart_are_not_merged_and_wraparound_is_merged():
    far_apart = merge([Sighting(10, STRANGER_WHO, 0), Sighting(60, STRANGER_WHO, 1)], 30, 200, 0.8, 0.4)
    assert len(far_apart) == 2
    [e] = merge([Sighting(355, STRANGER_WHO, 0), Sighting(5, STRANGER_WHO, 1)], 30, 200, 0.8, 0.4)
    assert e.direction == "前" and (e.degrees < 3 or e.degrees > 357)

def test_cluster_takes_majority_class():
    s = [Sighting(180, UNLIT_WHO, i) for i in range(3)] + [Sighting(185, STRANGER_WHO, 3)]
    [e] = merge(s, 30, 200, 0.8, 0.4)
    assert e.who == UNLIT_WHO and e.frames == 4

def test_beside_friend_is_not_spread_over_directions():
    s = [Sighting(i * 45.0, "卡洛", i, 300, beside=True) for i in range(8)]
    [e] = merge(s, 30, 200, 0.8, 0.4)
    assert e.direction == "身边"

def test_result_text():
    r = SweepResult([SweepEntry("前", 0, "懒洋洋大王", 5, None), SweepEntry("右后", 140, STRANGER_WHO, 3, None),
                     SweepEntry("右后", 150, UNLIT_WHO, 4, None)], None, 30, 2.0)
    assert r.text() == "正前方：懒洋洋大王；右后方：2 个陌生人（1 个没点火）"
    assert SweepResult([], None, 30, 2.0).text() == "转了一圈，身边没看到别人"

def sweeping(n=20, width=1920):  # 中间一个不动的框 + 一个从右往左横扫的框
    return [[Rect(900, 500, 120, 240), Rect(1800 - i * 90, 450, 100, 220)] for i in range(n)]

def test_find_self_picks_the_still_box_in_the_middle():
    f = find_self(sweeping(), 1920, 0.03)
    assert f.box == Rect(900, 500, 120, 240) and len(f.per_frame) == 20 and all(j == 0 for _, j in f.static)

def test_find_self_gives_up_when_holding_hands():
    frames = [[Rect(900, 500, 120, 240), Rect(1000, 500, 110, 230)] for _ in range(20)]
    assert find_self(frames, 1920, 0.03).box is None

def test_find_self_needs_enough_frames():
    assert find_self(sweeping(3), 1920, 0.03) == SelfFound(None, set(), {})

def test_find_self_ignores_boxes_seen_in_few_frames():  # 中间不动的框只出现在 10/20 帧 → box None
def test_distance_bands():
    assert [distance(h, 200, 0.8, 0.4) for h in (160, 159, 80, 79)] == ["近", "中", "中", "远"]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_sweep.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `vision/sweep.py`**（照上面的接口和规则；圆周平均用 `atan2(Σsin, Σcos)`）

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest tests/test_sweep.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/sweep.py tests/test_sweep.py
git commit -m "feat(perception): 环绕扫描的纯计算——方位角、8 方位、合并、转圈认团子、距离分档"
```

### Task 5: `PerceptionWatcher.sweep()` + 运行时团子框

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/config.py`（`[perception]` 的 `near` / `far` / `self_height`）、`config.example.toml`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `vision/sweep.py` 全部（Task 4）；`SpinConfig`（Task 1）
- Produces:
  - `PerceptionWatcher.sweep(frames: list[tuple[float, np.ndarray]], spin: SpinConfig) -> SweepResult`
  - `PerceptionWatcher.self_box: Rect | None`（转圈认出的团子；`_filter` 优先用它排除团子，没有才用 `self_roi`）
  - `PerceptionWatcher._ref_height(frame_h: int) -> float`：`self_box.h` → 最近一帧 `self` 轨迹的框高 → `cfg.self_height × frame_h`（Task 9、11 也用）
  - `PerceptionWatcher._infer: threading.Lock`：`detector.detect` 和 `_ocr` 都在这把锁里调（sweep 和后台线程可能同时用检测器 / OCR）
  - 私有 `_match_name(text, friends) -> str | None`（从 `_read_name` 抽出来，两处共用）

每帧：检测 → `_filter(..., panel_visible=False, drop_self=False)`（新参数，sweep 不先排除团子）→ 按 `conf` 过滤。所有帧的 `player` / `player_unlit` / `self` 框交给 `find_self`；每帧的名字标签跑 `_ocr` + `_match_name`，认出 → 好友名 sighting，读不出或对不上 → `UNKNOWN_WHO`；标签下面的人物（`_tag_over`，Detection 也有 `.box`）属于 `static` → 这条 sighting `beside=True`；人物框在 `static` 里、类别是 `self`、头顶有标签的都跳过；剩下 `player_unlit` → `UNLIT_WHO`，`player` 框高 ≥ `stranger_min_height × 画面高` → `STRANGER_WHO`，更小的跳过。sighting 的 `height` = 人物框高（标签下没找到人物时 None）。`find_self` 认出团子就更新 `self.self_box` 并 `log.info`。**不碰** `tracker`、`last_seen`、`labels`、`requests`。

- [ ] **Step 1: 写失败的测试**

```python
def spin_frames(n=20):  # t = i * 0.1，seconds_per_turn = 2.0，hfov = 90
    return [(i * 0.1, frame()) for i in range(n)]

def test_sweep_reports_friend_direction_and_strangers():
    det = FakeDetector()
    # 每帧：中间不动的团子 player(900, 500)；懒洋洋大王的标签 + 人物在画面中心（t=0.5 时正对 → 右）只出现在第 5 帧；
    # 一个黑影 player_unlit 只出现在第 10 帧的画面中心（→ 后）
    ...
    r = w.sweep(spin_frames(), SpinConfig())
    assert {(e.who, e.direction) for e in r.entries} == {("懒洋洋大王", "右"), (UNLIT_WHO, "后")}
    assert r.self_box == Rect(900, 500, 90, 220) and w.self_box == r.self_box

def test_sweep_does_not_touch_runtime_state():  # sweep 之后 w.last_seen == {}、w.tracker.tracks == {}、w.requests == {}
def test_holding_partner_is_reported_beside():  # 团子旁边紧挨一个带"卡洛"标签、每帧都不动的人 → ("卡洛", "身边")，self_box None
def test_unknown_tag_is_not_a_stranger():  # 标签读出"路人甲"（不在好友名单）→ UNKNOWN_WHO，不是 STRANGER_WHO
def test_small_untagged_player_is_skipped():  # 框高 < 8% 屏高、没标签 → 不出条目
def test_self_box_from_sweep_excludes_self_at_runtime():
    # sweep 认出团子后，process() 一帧：同一位置的 player 连续 2 秒 → w.strangers(now) == 0
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_perception.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**（照上面的规则；`config.example.toml` 的 `[perception]` 加 `near` / `far` / `self_height`，注明待标定）

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git commit -am "feat(perception): sweep() 汇总一圈的检测；转圈认出的团子替代 self_roi"
```

### Task 6: `look_around` 走扫描；`#spin` / `camera spin` 带扫描结果

**Files:**
- Modify: `src/skydango/brain/body.py`、`src/skydango/brain/tools.py`、`src/skydango/brain/mcp_server.py`、`src/skydango/brain/prompt.py`、`src/skydango/agent.py`、`src/skydango/cli.py`
- Test: `tests/test_brain_tools.py`、`tests/test_brain_body.py`、`tests/test_brain_prompt.py`、`tests/test_agent.py`

**Interfaces:**
- Consumes: `PerceptionWatcher.sweep`（Task 5）、`Camera.spin`（Task 1）、`Agent._spin`（Task 2）
- Produces:
  - `Body.sweep_around() -> str`：黑屏 → `ToolError("画面黑着（在切场景），现在看不了")`；dry-run 或没 camera → `env.sweep([(0.0, self.fresh_frame())], cfg.spin)`，返回 `"dry-run：没真的转，只看了前面。" + text()`；否则 `with self._held("camera"):` 里 `camera.spin(...)` + `env.sweep([(0.0, before), *frames], cfg.spin)`，`finally` 置 `_ref_thumb = None`，`last_frame = after`
  - `ToolBox._exec("look_around")`：`hasattr(b.env, "sweep")` → `b.call(b.sweep_around, timeout=AROUND_TIMEOUT)`（不叫眼睛、没开眼睛也能用）；否则照旧
  - `tools.descriptions(sweep: bool) -> dict[str, str]`：`sweep=True` 时 `look_around` 换成 `"环顾四周：原地连续转一圈，身体认出每个方向有谁（好友名字、几个陌生人），返回文字，最后回到原来的朝向。几秒就好。"`；`build_server(toolbox)` 改用 `descriptions(hasattr(toolbox.body.env, "sweep"))`（`DESCRIPTIONS` 保留 = `descriptions(False)`）
  - `brain_prompt(reply_cfg, store, quick_around: bool = False)`：`quick_around` 时把"（要十几秒，别常用）"换成"（几秒就好）"
  - `Agent._spin`：`hasattr(env, "sweep")` 时在 `held("camera")` 里接着 `sweep`，回复 = 转完那句 + 备注 + `"；" + sweep.text()`，再按 `max_chars` 截断
  - `cmd_camera`：`[perception] enabled` 时用 `_scene_watcher(cfg, dev=dev, background=False)` 扫一遍，打印 `text()` 和耗时

- [ ] **Step 1: 写失败的测试**

```python
def test_look_around_uses_sweep_when_perception_is_on():  # FakeEnv 有 sweep → 返回 sweep 的文字；eyes.describe 没被调；camera.spin 调了一次
def test_look_around_falls_back_to_eyes_without_sweep():  # 现有行为不变
def test_look_around_sweep_in_dry_run_does_not_turn():  # dry-run：camera.spin 没调，sweep 收到 1 帧，文字以 "dry-run" 开头
def test_look_around_description_switches():
    assert "几秒就好" in descriptions(True)["look_around"] and "十几秒" in descriptions(False)["look_around"]
def test_brain_prompt_quick_around():  # quick_around=True 时没有 "要十几秒"
def test_spin_reply_includes_sweep_text():  # Agent + FakeEnv.sweep 返回 text "正前方：懒洋洋大王" → sent[-1] 里有它
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `python -m pytest tests/test_brain_tools.py tests/test_brain_body.py tests/test_brain_prompt.py tests/test_agent.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**（`_run_brain` 里 `brain_prompt(..., quick_around=hasattr(env, "sweep"))`）

- [ ] **Step 4: 跑测试，确认通过**

Run: `python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git commit -am "feat(brain): 打开感知层时 look_around 连续转一圈交给 YOLO 汇总，不叫眼睛；#spin 回复带扫描结果"
```

### Task 7: `perception label --spin`：转圈录像自动补 `self` 框

**Files:**
- Modify: `src/skydango/vision/weaklabel.py`、`src/skydango/cli.py`
- Test: `tests/test_perception.py`（weaklabel 部分）或 `tests/test_cli_camera.py`

**Interfaces:**
- Consumes: `find_self`（Task 4）
- Produces:
  - `weaklabel.with_self(boxes: list[tuple[str, Rect]], me: Rect) -> list[tuple[str, Rect]]`：去掉和 `me` IoU ≥ 0.5 的 `player` / `self` 框，加一条 `("self", me)`
  - `perception label <spin 目录> --spin --model 模型`：只收 `NNN_*.jpg`（按文件名排序，跳过 `before` / `after`）；先对全部帧跑检测（结果缓存，后面合并预标注时复用），`find_self(…, cfg.spin.self_motion)`，认出就给 `per_frame` 里的每帧 `with_self`；没 `--model` → `SystemExit("--spin 要配 --model：认团子要先用模型框出人")`；认不出 → 打印"认不出团子（……），不补 self"，照常出其余标注

- [ ] **Step 1: 写失败的测试**

```python
def test_with_self_replaces_overlapping_player():
    boxes = [("player", Rect(900, 500, 90, 220)), ("player", Rect(100, 500, 90, 220)), ("name_tag", Rect(900, 440, 90, 40))]
    out = with_self(boxes, Rect(902, 502, 90, 220))
    assert ("self", Rect(902, 502, 90, 220)) in out and ("player", Rect(900, 500, 90, 220)) not in out and len(out) == 3

def test_label_spin_requires_model(...):  # main(["perception", "label", dir, "--spin"]) → SystemExit，信息里有 "--model"
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest -q` Expected: FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(perception): label --spin 用转圈认出的团子给每帧补 self 框"
```

### Task 8: 检测结果交给眼睛（`scene_note`）

**Files:**
- Modify: `src/skydango/brain/images.py`、`src/skydango/brain/eyes.py`、`src/skydango/brain/body.py`、`src/skydango/cli.py`（`_run_brain` 和 `cmd_look` 里 `Eyes(...)` 的构造）
- Test: `tests/test_brain_images.py`、`tests/test_brain_eyes.py`、`tests/test_brain_body.py`

**Interfaces:**
- Produces:
  - `scene_note(env, now: float, scale: float) -> str`（用 `env.overlay(now)`；坐标 = 框中心 × scale，取整）
  - `Eyes(..., note: Callable[[float, float], str] | None = None)`：给了就用 `note(now, scale)`，不再用 `labels` + `label_note`
  - 判 YOLO 用 `hasattr(env, "strangers")`（EnvWatcher 也有 `overlay`，不能拿它判）
  - `look [图片]`（§2 验收要对 20 张截图各跑一次）：给了图片就描述它，不截屏；识别器改用 `_scene_watcher(cfg, background=False)`，
    YOLO 时同一帧 `process` 两次（`now = 0` 和 `now = perception.stranger_after`，好让点过火的陌生人也判出来）；
    描述之前先打印交给眼睛的那段位置说明，方便人工核对

`scene_note` 格式（顺序：好友、没点火、陌生人、团子，同类按 x 排）：

```
画面里认出的人（坐标按这张图）：
- 懒洋洋大王：(812, 340) 附近
- 陌生人（没点火，黑影）：(1105, 420) 附近
- 陌生人：(300, 390) 附近
- 团子（就是“你”自己）：(640, 470) 附近
没列出的人都叫“陌生人”。
```

好友取 `kind == "friend"` 的人物框；某个名字只有 `kind == "name"` 的标签框时写 `"- 名字：头顶名字在 (x, y)，人在名字下方"`（x 取标签中心，y 取标签上沿）。其他 kind（`player`、`tag`、`ring`、`request`、`typing`）不列。一个人都没有 → `"画面里没认出人（可能被挡住、离得远，或者没人）\n没列出的人都叫“陌生人”。"`

提示词：`EYES_SYSTEM` 的人名那一行换成 `"- 人名只用给出的名字；给出的陌生人、团子位置照用；没列出的人都叫“陌生人”。"`；`LOOK_REQUEST` 好友一项末尾加 `"；团子自己不用描述"`。

- [ ] **Step 1: 写失败的测试**

```python
def test_scene_note_lists_people_by_kind_and_scales():
    class Env:
        def overlay(self, now):
            return [{"x": 1580, "y": 560, "w": 100, "h": 140, "kind": "unlit", "label": "陌生人（没点火）"},
                    {"x": 1170, "y": 440, "w": 96, "h": 140, "kind": "friend", "label": "懒洋洋大王"},
                    {"x": 900, "y": 640, "w": 120, "h": 130, "kind": "self", "label": "团子"},
                    {"x": 400, "y": 520, "w": 100, "h": 130, "kind": "stranger", "label": "陌生人"},
                    {"x": 1170, "y": 380, "w": 96, "h": 40, "kind": "name", "label": "懒洋洋大王"},
                    {"x": 10, "y": 10, "w": 50, "h": 50, "kind": "typing", "label": "正在输入"}]
    assert scene_note(Env(), 0.0, 2 / 3).splitlines() == [
        "画面里认出的人（坐标按这张图）：", "- 懒洋洋大王：(812, 340) 附近", "- 陌生人（没点火，黑影）：(1087, 420) 附近",
        "- 陌生人：(300, 390) 附近", "- 团子（就是“你”自己）：(640, 470) 附近", "没列出的人都叫“陌生人”。"]

def test_scene_note_name_only_and_empty(): ...
def test_eyes_use_scene_note_when_given():  # Eyes(note=lambda now, s: "NOTE") → describe 收到的文字块以 "NOTE" 开头
def test_body_look_uses_scene_note_with_perception():  # FakeEnv 有 strangers + overlay → look 的文字块里有 "没列出的人都叫"
def test_body_look_keeps_label_note_without_perception():  # 现有测试不改即可
def test_look_describes_given_image(tmp_path, monkeypatch):  # main(["look", 图片]) 不截屏；打印的内容里有位置说明（假 one_shot）
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest tests/test_brain_images.py tests/test_brain_eyes.py tests/test_brain_body.py -q` Expected: FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(eyes): 眼睛拿 YOLO 认出的好友 / 陌生人 / 团子位置，不再自己编名字"
```

### Task 9: `typing` 气泡关联 + `speaker_hint`

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/config.py`（`classes` 追加 `"typing"`、`typing_window = 8.0`）、`config.example.toml`、`src/skydango/vision/viewer.py`
- Test: `tests/test_perception.py`、`tests/test_viewer.py`

**Interfaces:**
- Consumes: `_ref_height`（Task 5）、`distance`（Task 4）
- Produces:
  - `PerceptionWatcher.speaker_hint(now: float) -> str | None`
  - 私有 `_person_below(bubble: Rect, people: list) -> 同类元素 | None`：气泡水平中心落在人物框（左右各放宽 25%）内，气泡下沿在 `[人物上沿 − 2.5 × 人物高, 人物上沿 + 0.3 × 人物高]`，取竖直距离最近的
  - `overlay()` 里 `typing` 轨迹：`kind = "typing"`、`label = "正在输入"`；viewer `COLORS.typing = "#e879f9"`、`NAMES.typing = "正在输入"`

每帧：`typing` 轨迹先看团子（`self` 轨迹的框、`self_box`）→ 在团子上方就跳过；否则 `_person_below(气泡, players)` 找到人物 → `player.data["typing_at"] = now`，往 `self._typing` 追加 `(now, player.id, 是好友, 没点火, 框中心 x, 框高, 画面宽, 画面高)`（是好友 = 有 `name` 或 `tagged`），丢掉超过 `typing_window` 的。
`speaker_hint`：窗口内不是好友的记录按轨迹 id 去重，**恰好一个** → 用它最新一条：横向 `x < 宽/3` "左边"、`> 宽×2/3` "右边"、否则 "前面"；远近用 `distance(框高, _ref_height(画面高), near, far)`，"远" → "远处"、否则 "近处"；返回 `"（说话的可能是{横向}{远近}那个{没点火的陌生人|陌生人}）"`。零个或多个 → None。

- [ ] **Step 1: 写失败的测试**

```python
def bubble(x, y=180, w=80, h=60):
    return Detection("typing", Rect(x, y, w, h), 0.9)

def test_bubble_over_unlit_stranger_gives_hint():
    # 帧里：player_unlit 在 (1500, 400, 90, 220)，bubble(1505)；clock=10
    assert w.speaker_hint(12.0) == "（说话的可能是右边近处那个没点火的陌生人）"  # 220 ÷ (0.2 × 1080) ≈ 1.02 → 近
    assert w.speaker_hint(19.0) is None  # 过了 typing_window

def test_bubble_over_friend_or_self_gives_no_hint():  # 气泡在有标签的好友上方 / 在 self 框上方 → None
def test_two_strangers_typing_gives_no_hint():  # 两个没标签的陌生人各有气泡 → None
def test_overlay_marks_typing():  # overlay 里有 {"kind": "typing", "label": "正在输入"}
def test_default_classes_end_with_typing():
    assert PerceptionConfig().classes == ["player", "name_tag", "social_ring", "self", "player_unlit", "typing"]
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest tests/test_perception.py tests/test_viewer.py -q` Expected: FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(perception): typing 气泡关联到下方的人，给陌生人的消息猜说话人"
```

### Task 10: 消息加注说话人（大脑 + 普通模式）

**Files:**
- Modify: `src/skydango/chat/reader.py`、`src/skydango/brain/body.py`、`src/skydango/agent.py`
- Test: `tests/test_brain_body.py`、`tests/test_agent.py`

**Interfaces:**
- Consumes: `speaker_hint`（Task 9）
- Produces: `chat.reader.with_speaker_hint(env, messages: list[Message], now: float) -> list[Message]`：`env` 没有 `speaker_hint` 时原样返回；`speaker` 是 `""` 或 `"陌生人"` 的消息，hint 不为 None 时 `dataclasses.replace(m, text=m.text + hint)`；其他原样

调用点：`Body._heard` 开头（大脑事件、备用回复、聊天记录都带上）；`Agent.step` 摘掉主人命令之后、进 `pending` 之前。

- [ ] **Step 1: 写失败的测试**

```python
def test_stranger_message_gets_speaker_hint_in_event():
    # FakeEnv.speaker_hint 返回 "（说话的可能是左边近处那个陌生人）"；读到 Message("你好", ..., speaker="陌生人")
    # → chat 事件文字 == "聊天  陌生人：「你好（说话的可能是左边近处那个陌生人）」"
def test_friend_message_gets_no_hint():  # speaker="懒洋洋大王" → 文字不变
def test_agent_pending_message_carries_hint():  # Agent：pending[0].text 以提示结尾
def test_no_hint_without_perception():  # env 没有 speaker_hint → 不变
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest tests/test_brain_body.py tests/test_agent.py -q` Expected: FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(chat): 陌生人的消息后面加注说话的可能是画面上哪个人"
```

### Task 11: 距离和走向（感知层）

**Files:**
- Modify: `src/skydango/vision/perception.py`、`src/skydango/config.py`（`approach_window`、`approach_grow`、`approach_cooldown`、`approach_strangers`）、`config.example.toml`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `distance`（Task 4）、`_ref_height`（Task 5）
- Produces:
  - `approaching(hist: list[tuple[float, float, float]], width: int, grow: float) -> bool`（模块函数；`hist` = `(时间, 框高, 框中心 x)`）：至少 3 个样本；`k = max(1, len // 3)`，前 k 个和后 k 个的框高均值之比 ≥ `1 + grow`，且后 k 个的 `|x − 宽/2|` 均值 ≤ 前 k 个的，或 < `0.15 × 宽`
  - `PerceptionWatcher.pop_approaches() -> list[str]`：取走并清空（加锁），元素是好友名或 `STRANGER`
  - `PerceptionWatcher.nearest(now: float) -> tuple[str, str] | None`：最近一帧里框最高的、身份已知的人物（好友名 / `"陌生人"`，没点火也算陌生人），配 `distance(...)`

每帧（不在暂停中）：对每个非团子人物，往 `track.data["hist"]` 追加样本、丢掉超过 `approach_window` 的；`approaching(...)` 为真 → key = 好友名，或 `STRANGER`（`approach_strangers = false` 时跳过陌生人；身份未知的跳过）；同一 key `approach_cooldown` 秒内只记一次（陌生人共用一个 key）。`_resume()` 清掉所有轨迹的 `hist`。

- [ ] **Step 1: 写失败的测试**

```python
def test_approaching_rules():
    grow = [(i * 0.1, 100 + i * 5, 1400 - i * 20) for i in range(15)]
    assert approaching(grow, 1920, 0.25)
    jitter = [(i * 0.1, 100 * (1.1 if i % 2 else 0.9), 960) for i in range(15)]
    assert not approaching(jitter, 1920, 0.25)
    outward = [(i * 0.1, 100 + i * 5, 1300 + i * 30) for i in range(15)]
    assert not approaching(outward, 1920, 0.25)
    assert not approaching(grow[:2], 1920, 0.25)

def test_friend_walking_up_is_reported_once_per_cooldown():  # 带标签的好友框 1.5 s 内变大 → pop_approaches() == ["懒洋洋大王"]；再来一遍（60 s 内）→ []
def test_stranger_approach_follows_switch():  # 没标签的陌生人走过来 → ["陌生人"]；approach_strangers=False → []
def test_no_approach_across_a_pause():  # 暂停前框高 100、release 后框高 140 → []
def test_nearest_picks_the_tallest_known_person():  # 好友框高 300、陌生人 150、团子框 220 → ("懒洋洋大王", "近")
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest tests/test_perception.py -q` Expected: FAIL
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(perception): 按团子框高分远近，框持续变大且往中间走判定走过来"
```

### Task 12: 身体的 `approach` 事件和 `status` 最近的人

**Files:**
- Modify: `src/skydango/brain/body.py`、`src/skydango/brain/events.py`（`Event.kind` 注释加 `approach`）
- Test: `tests/test_brain_body.py`、`tests/test_brain_eyes.py`

**Interfaces:**
- Consumes: `pop_approaches`、`nearest`（Task 11）
- Produces: 事件 `approach`：好友 `"{名字} 朝你走过来了"`、陌生人 `"有个陌生人朝你走过来了"`；正牵着手的那个人跳过。`status` 在陌生人数后加 `"离你最近的：{谁}（{近|中|远}）"`（`nearest` 返回 None 时不加）

- [ ] **Step 1: 写失败的测试**

```python
def test_approach_events():  # FakeEnv.pop_approaches → ["懒洋洋大王", "陌生人"] → 两个 approach 事件，文字如上
def test_holding_partner_approach_is_dropped():  # holding = "卡洛"，pop_approaches → ["卡洛"] → 没有事件
def test_status_shows_nearest():  # FakeEnv.nearest → ("懒洋洋大王", "近") → status 里有 "离你最近的：懒洋洋大王（近）"
def test_eyes_do_not_auto_look_on_approach():
    assert "approach" not in AUTO_LOOK_KINDS
```

- [ ] **Step 2: 跑测试，确认失败** — Run: `python -m pytest tests/test_brain_body.py tests/test_brain_eyes.py -q` Expected: FAIL
- [ ] **Step 3: 实现**（在 `_watch_people` 里，陌生人判断之后）
- [ ] **Step 4: 跑测试，确认通过** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 5: 提交**

```bash
git commit -am "feat(brain): approach 事件（有人朝团子走过来）、status 里离得最近的人"
```

### Task 13: 文档

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-perception-phase2-design.md`（状态：代码部分已完成；§6 验收待 GPU 机器 / 真机）、`docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`（§11 `self` 那一项、§14 标"已实现（代码）"、§15 二期状态）、`docs/superpowers/specs/2026-09-27-owner-commands-design.md`（§6 状态：已实现）、`docs/game-ops.md` §2 方向键那一行（加"一圈多少秒 / 视野角用 `camera spin` 标定，**待测**"）、`CLAUDE.md`（代码结构加 `vision/sweep.py`；常用命令加 `camera spin`、`perception label --spin`；感知层一节加二期接口；`#spin`）

- [ ] **Step 1: 改文档**
- [ ] **Step 2: 跑全部测试** — Run: `python -m pytest -q` Expected: 全部 PASS
- [ ] **Step 3: 提交**

```bash
git commit -am "docs: 感知层二期代码部分完成；camera spin / #spin / label --spin 用法"
```

## GPU 机器 / 真机上剩下的（不在本方案里）

1. `camera spin` 标定 `seconds_per_turn`、`hfov`，回答总纲 §14.5，写进 game-ops §2
2. 录"好友 / 陌生人在打字、刚发出消息"的画面，补标 `typing`，重训；确认设计 §3"要在真机确认的"三条
3. 标定 `near` / `far` / `self_height`、`approach_grow`
4. 跑设计 §6 的验收，结果写回设计文档
