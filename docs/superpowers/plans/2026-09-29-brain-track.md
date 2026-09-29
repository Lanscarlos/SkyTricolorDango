# 技能 D：`track(name)` 盯着好友 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 大脑调 `track(名字, 秒)` 后，身体按主循环的节拍小步转镜头，把这个好友保持在画面中间附近，跟丢 / 到时间发 `task_failed` / `task_done`；`camera_reset` 改成闭环复位。

**Architecture:** 设备层加一个"在模拟器里 sleep 的精确短按"；`Camera` 加 `nudge()`（按秒记账）和"转之前存参照缩略图、复位时先粗转回去再小步比对"的闭环复位；
身体加 `target_x()`（先 YOLO 人物框，框不稳就用名字标签的 x）和 `track()` 工具入口；新文件 `brain/track.py` 放 `TrackSkill`（纯控制逻辑，不碰设备），挂在已有的 `SkillRunner` 上。
**和路线图的差别**：路线图里的 `Tracker.pan(dx)`（转镜头后把所有轨迹框平移）不做 —— D0 实测近处的人和远处的背景横移差好几倍，统一平移本身不对；
跟踪每圈按名字重新找目标，不依赖轨迹 id。增益也不做"按远近分档"的表，第一版用比例控制 + "转不动就停手"，真机验收后再看要不要分档。

**Tech Stack:** Python 3.13、现有 `brain/`（Body / SkillRunner / ToolBox / MCP）、`vision/people.py`、pytest（合成画面 + 假设备）

**Spec:** `docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md` 的「D. track」一节（含 2026-09-29 D0 结论和"D0 带出来的设计要求"）；
实测数据 `docs/game-ops.md` §2「转视角实测（D0）」。

## Global Constraints

- 技能在身体线程里跑，`tick()` 里**不许 sleep 超过 0.3 s**（身体还要读聊天）；按键时长最长 0.1 s，等画面停稳靠"记下上次按键时间、之后几圈不动"，不靠 sleep
- 方向：**右转（键码 106）画面里的东西往左移**；目标在中线右边 → 按右键
- 最短按键 `nudge_min = 0.02` s（实测一下就移 35~40 px），最长 `nudge_max = 0.1` s；死区 `deadband = 0.15`（× 半屏宽 960 ≈ 144 px，大于最小修正量）
- 按键停稳时间 `settle = 0.6` s（D0 里松手 0.3~0.6 s 后位置才稳定）
- 跟丢：`lost_after = 3.0` s 看不到目标 → failed"跟丢了（最后在{左边/前面/右边}）"
- 位置只认新鲜的：人物框来自 `env.people(now)`（已经只给 ≤ 1 s 的），名字标签要 `now - t <= max_age = 0.5` s
- 工具 `track(name, seconds=30)`：`seconds` 夹到 1~60；dry-run 拒绝（手动控制 live=True 照做）；没开感知层 / 没有镜头 / 开始时画面里找不到这个人都拒绝（技能不开始）
- 跟踪期间身体**不发** `arrive` / `leave` / `stranger` 事件、也不更新"身边有谁"的比较基准（转镜头时人进出画面是自己转的，不是人走了）；技能结束后下一圈照常比较
- `camera_reset` 必须闭环：只靠反向按同样时长会差到 90°（D0 实测）
- 开 / 关聊天面板画面横移约 130 px：技能开始后的第一个 `settle` 秒不按键（等借面板的动画和横移结束）

## Review Focus

1. **目标离团子很近**：转镜头时它在屏幕上横移有上限（D0：0.2 s 以后饱和在 ~84 px），可能永远进不了死区 —— 期望：连续 `stall_nudges = 3` 次同方向按键后误差没缩小 20 px 以上，就停手（note "他就在旁边，转不动了"），直到目标位置相对停手时变化超过死区再重新开始；不能一直转圈（Task 4 测）
2. **人物框时有时无**（v4 人挨着团子时只框住 67%）：期望用名字标签的 x 接上，不算跟丢（Task 3 测）
3. **跟踪中目标被别人挡住 1~2 秒**：期望不算跟丢（< `lost_after`），也不在看不到时乱按键（Task 4 测）
4. **参照图拍的时候画面里有人走动**：闭环复位的相似度会低 —— 期望比对只用画面上半部分的背景（缩略图取 y 0~45%、x 34%~100%，避开聊天面板和人），找不到明显高峰（最高相似度 < 0.5）就停在粗转的位置并在结果里说"没对准"（Task 2 测）
5. **跟踪中大脑 stop_task / 身体退出**：期望松开按键（本来就是一条命令按完即松，不会卡键）、不发事件、镜头不复原（交给 camera_reset）（Task 4 测）

---

### Task 1: 设备层的精确短按 `hw_key_hold`

**Files:**
- Modify: `src/skydango/device/base.py`（加默认实现）、`src/skydango/device/adb.py`（一条 shell 命令）
- Modify: `tests/conftest.py`（`FakeDevice` 记录）
- Test: `tests/test_adb.py`

**Interfaces:**
- Produces: `Device.hw_key_hold(code: int, seconds: float) -> None`；`FakeDevice.calls` 里记 `("hw_hold", code, seconds)`

- [ ] **Step 1: 写失败的测试** `test_hw_key_hold_sleeps_on_device`：用现有 `ShellRunner` 录命令，`dev.hw_key_hold(106, 0.05)` 只发**一条** shell 命令，内容依次含 `sendevent <dev> 1 106 0`、`sendevent <dev> 1 106 1`、`sleep 0.05`、`sendevent <dev> 1 106 0`（先补抬起，同 `hw_key`）
- [ ] **Step 2: 跑测试确认失败** `python -m pytest -q tests/test_adb.py -k hold` → AttributeError
- [ ] **Step 3: 实现**：`base.Device.hw_key_hold` 默认 = `hw_key_down` + `time.sleep` + `hw_key_up`（try/finally 保证抬起）；`AdbDevice.hw_key_hold` 覆盖成一条命令（`sleep` 在模拟器里，时长不受 adb 往返影响，D0 就是这么测的）；`FakeDevice.hw_key_hold` 追加 `("hw_hold", code, seconds)`
- [ ] **Step 4: 跑测试确认通过** `python -m pytest -q tests/test_adb.py`
- [ ] **Step 5: 提交** `feat(device): hw_key_hold 一条命令按住指定秒数`

### Task 2: `Camera.nudge` 和闭环 `reset`

**Files:**
- Modify: `src/skydango/brain/camera.py`
- Test: `tests/test_brain_camera.py`

**Interfaces:**
- Consumes: Task 1 `device.hw_key_hold`
- Produces:
  - `Camera.nudge(direction: str, seconds: float) -> float`：`direction ∈ {"left","right"}`，`seconds` 夹到 `[0.02, 0.1]`，返回实际按的秒数；**不**借面板、不按 BACK（调用方 = 技能，已经借了面板）；累计 `self.turn_seconds`（右正左负）
  - `Camera.remember() -> None`：镜头在原位（`offset` 全 0 且 `turn_seconds == 0`）时截一张图存成参照缩略图 `self.ref`；不在原位时不覆盖。`move()`、`nudge()` 开头都先调它
  - `Camera.reset() -> str`：先按原来的方式撤销步数偏移，再把 `turn_seconds` 反向按回去（每次最多 0.1 s，分几次），然后**有参照图时**闭环细调：左右各试 0.02 s 小步，哪边相似度升就往哪边走，直到两边都不升或走满 `REFINE_MAX = 30` 步；最高相似度 < 0.5 时结果里带"（没对准，可能差一点）"。结束清掉 `ref`、`turn_seconds = 0`
  - 模块函数 `thumb_similarity(a: np.ndarray, b: np.ndarray) -> float`：两张图各取 y 0~45%、x 34%~100% 缩成 160×90 灰度，返回相关系数（-1~1）

- [ ] **Step 1: 写失败的测试**
  - `test_nudge_holds_arrow_and_counts_seconds`：`cam.nudge("right", 0.05)` 后 `device.calls` 里有 `("hw_hold", 106, 0.05)`，`turn_seconds == 0.05`；`nudge("left", 0.5)` 夹成 0.1、`turn_seconds == -0.05`；`nudge("up", …)` 抛 `ValueError`
  - `test_nudge_does_not_touch_panel`：用 `fake_panel` 开着的面板，`nudge` 之后面板状态不变、没有 46 号键
  - `test_reset_undoes_nudges_then_refines_to_reference`：假设备的 `screenshot()` 按"累计转了多少秒"返回一张横向平移过的合成图（平移像素 = 累计秒数 × 800，模拟画面随镜头移动）；先 `nudge("right", 0.1)` 三次，再让假设备多漂 0.04 s（模拟反向按不准），`reset()` 之后累计漂移回到 |x| ≤ 0.02 s，结果不含"没对准"
  - `test_reset_without_reference_only_reverses`：没有参照图（构造后直接改 `turn_seconds = 0.2`）时 `reset()` 只反向按 0.2 s（两次 0.1），不做细调
  - `test_reset_reports_when_reference_never_matches`（Review Focus 4）：假设备每次截图都返回随机噪声图，`reset()` 结果含"没对准"，细调最多 `REFINE_MAX * 2` 次按键
  - `test_thumb_similarity_ignores_chat_panel_area`：两张图只在左边 0~34% 区域不同 → 相似度 > 0.99
- [ ] **Step 2: 跑测试确认失败** `python -m pytest -q tests/test_brain_camera.py`
- [ ] **Step 3: 实现**：细调时每按一步要等画面停：用 `self.sleep(0.4)`（`reset` 是大脑工具 / 退出收尾调的，不在技能 tick 里，可以等）；相似度用 `thumb_similarity`
- [ ] **Step 4: 跑测试确认通过**（含已有的 camera 测试：`move` / `reset` 旧行为不能变 —— 没有 nudge、没有参照图时 `reset` 和原来一样）
- [ ] **Step 5: 提交** `feat(camera): nudge 短按记秒数，reset 先粗转再按参照图闭环细调`

### Task 3: 身体找目标位置 `target_x`，跟踪期间不报人来人走

**Files:**
- Modify: `src/skydango/brain/body.py`、`src/skydango/config.py`（`TrackConfig`、`Config.track`）
- Test: `tests/test_brain_body.py`

**Interfaces:**
- Consumes: 现有 `env.people(now)`、`env.labels`（`{名字: (x, y, w, h, t)}`）、`similar()`（容忍 OCR 错一两个字，同 `_locate`）
- Produces:
  - `TrackConfig`（`[track]`）：`deadband=0.15`、`gain=0.1`（秒 / 单位偏差）、`nudge_min=0.02`、`nudge_max=0.1`、`settle=0.6`、`lost_after=3.0`、`max_age=0.5`、`stall_nudges=3`、`stall_px=20.0`、`max_seconds=60`
  - `Body.target_x(name: str, now: float) -> tuple[float, str] | None`：返回 `(中心 x 像素, 来源)`，来源 `"body"`（`people()` 里名字对得上的好友框中心）或 `"tag"`（`labels` 里 `now - t <= track.max_age` 的名字标签中心）；先精确匹配再 `similar(…, 0.75)`；都没有返回 `None`
  - `Body.frame_width -> int`（`last_frame` 的宽，没有时 1920）
  - 跟踪期间静默：`_watch_people` 开头，若 `self.skills.active` 有属性 `quiet_people = True`，直接 return（不发事件、不更新 `_nearby` / `_strangers` / 请求集合）

- [ ] **Step 1: 写失败的测试**
  - `test_target_x_prefers_body_box`：`FakeEnv.people_list` 有 `Person(…, name="懒洋洋大王", box=Rect(900, 400, 100, 300))`、`labels` 也有 → `(950.0, "body")`
  - `test_target_x_falls_back_to_fresh_tag`（Review Focus 2）：`people_list` 空、`labels["懒洋洋大王"] = (1200, 300, 160, 50, now - 0.2)` → `(1280.0, "tag")`；标签时间改成 `now - 0.8` → `None`
  - `test_target_x_tolerates_ocr_typo`：`labels` 里是"懒洋洋大玉" → 找"懒洋洋大王"能拿到
  - `test_no_people_events_while_quiet_skill_runs`：`b.skills.active` 设成一个带 `quiet_people = True` 的假技能，`env.near` 从 `["懒洋洋大王"]` 变成 `[]` → 这一圈没有 `leave`；清掉 `active` 后下一圈发 `leave`
- [ ] **Step 2: 跑测试确认失败** `python -m pytest -q tests/test_brain_body.py -k "target_x or quiet"`
- [ ] **Step 3: 实现**（`TrackConfig` 放 `SpinConfig` 后面，`Config` 加 `track: TrackConfig = field(default_factory=TrackConfig)`）
- [ ] **Step 4: 跑测试确认通过** `python -m pytest -q tests/test_brain_body.py tests/test_config.py`
- [ ] **Step 5: 提交** `feat(body): target_x 找好友位置（人物框优先、名字标签兜底），跟踪期间不报人来人走`

### Task 4: `TrackSkill`

**Files:**
- Create: `src/skydango/brain/track.py`
- Test: `tests/test_brain_track.py`

**Interfaces:**
- Consumes: Task 2 `camera.nudge(direction, seconds) -> float`；Task 3 `body.target_x(name, now)`、`body.frame_width`、`body.cfg.track`；`skills.SkillStep`；`people.side_of(cx, width)`
- Produces: `TrackSkill(name: str, seconds: float)`，属性 `name = "track"`、`goal = f"盯着{name}"`、`timeout = seconds + 5`、`needs_camera = True`、`quiet_people = True`；实现 `start / tick / stop`（见 `skills.Skill`）

控制逻辑（每次 tick）：
1. `now - 开始 < settle` 或 `now - 上次按键 < settle` → running，不按键（等面板动画 / 画面停稳）
2. `target_x` 为 None：记"最后一次看到"的时间和方位；超过 `lost_after` → failed"跟丢了（最后在{side}）"；否则 running"看不到{name}，等一下"
3. 偏差 `e = (cx - W/2) / (W/2)`；`|e| < deadband` → running"{name}在中间"，清掉卡住计数
4. 卡住判断（Review Focus 1）：同方向连续按了 `stall_nudges` 次、而 `|cx - W/2|` 比这一串开始时缩小不到 `stall_px` → 进入"停手"：记下停手时的 cx，running"{name}就在旁边，转不动了"；停手期间只有 |cx - 停手时 cx| > deadband × W/2 才解除
5. 否则按键：方向 `"right" if cx > W/2 else "left"`，时长 `clamp(gain × |e|, nudge_min, nudge_max)`，`camera.nudge(...)`，记上次按键时间
6. `now - 开始 >= seconds` → done"盯了 {N} 秒，{name}在{side}"（在 1 之前判断，到点就结束）

- [ ] **Step 1: 写失败的测试**（假 body：`target_x` 按脚本返回、`camera.nudge` 记录调用、`clock` fixture 控时间；`frame_width = 1920`；`cfg.track` 用默认值）
  - `test_waits_settle_before_first_nudge`：开始后 0.3 s 目标在 x=1700 → 没按键；0.7 s → 按了 `("right", 0.077…)`（`0.1 × (1700-960)/960`，允许 ±0.001）
  - `test_left_of_center_turns_left_and_clamps`：x=0 → `("left", 0.1)`；x=900（e≈-0.06，死区内）→ 不按
  - `test_no_second_nudge_until_settled`：按完 0.3 s 内即使还偏也不再按；0.6 s 后再按
  - `test_short_occlusion_is_not_lost`（Review Focus 3）：目标消失 2 s 再出现 → 一直 running，消失期间不按键
  - `test_lost_after_three_seconds_reports_last_side`：最后看到在 x=1800 → 消失 3.1 s 后 failed，note 含"跟丢了"和"右边"
  - `test_stall_stops_nudging_near_target`（Review Focus 1）：目标一直在 x=1400 不变（转了也移不动）→ 按满 3 次后停手，之后 5 s 内不再按键、note 含"转不动"；目标跳到 x=1700（差 > 144 px）→ 解除，又按键
  - `test_done_after_seconds`：`TrackSkill("懒洋洋大王", 5)`，目标一直在中间 → 5 s 时 done，note 含"盯了 5 秒"和"前面"
  - `test_stop_does_nothing_to_camera`（Review Focus 5）：`stop()` 之后没有新的 nudge / reset 调用
- [ ] **Step 2: 跑测试确认失败** `python -m pytest -q tests/test_brain_track.py`
- [ ] **Step 3: 实现** `brain/track.py`（只依赖 `skills.SkillStep`、`vision.people.side_of`；不导入设备 / YOLO）
- [ ] **Step 4: 跑测试确认通过** `python -m pytest -q tests/test_brain_track.py tests/test_brain_skills.py`
- [ ] **Step 5: 提交** `feat(skill): TrackSkill 小步转镜头把好友保持在画面中间`

### Task 5: `track` 工具接进身体、大脑和文档

**Files:**
- Modify: `src/skydango/brain/body.py`（`Body.track`）、`src/skydango/brain/tools.py`、`src/skydango/brain/mcp_server.py`、`src/skydango/brain/prompt.py`
- Modify: `config.example.toml`（`[track]` 一节）、`CLAUDE.md`（统管大脑一节：技能层现在有 `track`；工具清单加 `track`）、`docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md`（D 期标"代码已完成，待真机验收"）
- Test: `tests/test_brain_body.py`、`tests/test_brain_tools.py`、`tests/test_brain_prompt.py`、`tests/test_brain_mcp.py`

**Interfaces:**
- Consumes: Task 4 `TrackSkill`；Task 3 `target_x`；现有 `self.skills.start(body, skill) -> str`、`ToolError`、`self._dry(live)`
- Produces:
  - `Body.track(name: str, seconds: int = 30, live: bool = False) -> str`：dry-run → `ToolError("dry-run 不转镜头盯人")`；`self.camera is None` 或 env 没有 `people` → `ToolError`；`seconds` 夹到 `1 ~ track.max_seconds`；`target_x(name, now)` 为 None → `ToolError(f"画面里没看到 {name}（…认得出的名字…）；先 look_around 找找")`（复用 `_recognized`）；否则 `self.skills.start(self, TrackSkill(name, seconds))`
  - 工具 `track`：参数 `name`（字符串，必填）、`seconds`（整数，默认 30）；算"做了事"（加进 `ACTIONS`）；说明：`"转镜头一直盯着某个好友（让他留在画面中间），做完或跟丢了会用 task_done / task_failed 告诉你。离你很近的人本来就在画面里，不用盯。"`
  - 提示词「做事（stop_task）」一节加一行：track 是"开始做一件事"的工具之一；盯着的时候状态里"正在做：盯着…"

- [ ] **Step 1: 写失败的测试**
  - `test_track_starts_skill_when_target_visible`（live body，`people_list` 有这个好友）→ 返回"开始盯着懒洋洋大王了…"，`b.skills.active.name == "track"`
  - `test_track_refused_in_dry_run`、`test_track_refused_when_target_not_visible`（错误里有"没看到"和现在认得出的名字）、`test_track_clamps_seconds`（传 999 → `TrackSkill.timeout == 60 + 5`）
  - `test_tools_track_passes_args`（ToolBox `run("track", {"name": "小明", "seconds": 10})` 调到 `body.track("小明", 10)`；`"track" in ACTIONS`）
  - `test_mcp_lists_track`（照 `test_brain_mcp.py` 现有写法，工具列表里有 track）
  - `test_prompt_mentions_track`（`static_prompt` 里有 "track"）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**，并写 `config.example.toml` 的 `[track]`（每项一行注释，数字来自 D0 的写"2026-09-29 D0 实测"，`gain` 写"估的，真机验收后调"）
- [ ] **Step 4: 跑全量测试** `python -m pytest -q` → 全部通过
- [ ] **Step 5: 提交** `feat(brain): track 工具（盯着好友），文档和配置模板`

### Task 6: 真机验收（人工，不写代码）

前提：`[perception] enabled = true`、`capture = "own"`；live（`run --live --view`）；用网页「手动控制」或让卡洛在聊天里说 `#` 命令请团子盯他。

- [ ] 开阔地带，卡洛在**离团子 5 米以上**的地方绕团子慢走一圈，`track(卡洛, 30)`：10 次里 ≥ 8 次全程不跟丢（路线图的验收标准）
- [ ] 卡洛站在团子旁边（1~2 米）：不应一直转圈，状态显示"就在旁边，转不动了"
- [ ] 每次结束后 `camera_reset`：截图和开始前比，朝向差在十几像素以内（对照 `tmp/d0_restore.py` 的效果）
- [ ] 跟踪期间 `runs/<…>/agent.log` 里没有 `arrive` / `leave` / `stranger` 事件；结束后正常
- [ ] 结果和调过的 `gain` / `deadband` 写回 game-ops §2 和路线图 D 期，合并进 main 并推送
