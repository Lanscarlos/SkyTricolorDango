# 可视化网页：手动控制身体 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `run --brain --view` 时，网页上能手动让身体说话、做动作、转视角 / 环视、看人；手动的总是真执行，照样过身体护栏，做成后告诉大脑。

**Architecture:** 身体方法加 `live` 参数（放开"dry-run 不执行"）、`check_friend` 抽出按原图坐标的 `check_friend_at`、退出总是复原镜头；
新模块 `brain/manual.py` 的 `ManualControl` 校验参数、经 `Body.call` 执行、放 `manual` 事件；`Viewer` 加 `/control/options`、`POST /control`（本机 + 请求头 + Host 校验）
和页面控制栏；`cli._run_brain` 在本机模式下挂上。

**Tech Stack:** Python 3.13 标准库；页面是 `viewer.py` 内嵌原生 JS / CSS。

**Spec:** `docs/superpowers/specs/2026-09-28-viewer-manual-control-design.md`

## Global Constraints

- 大脑调用身体的行为**完全不变**（不传 `live` 时和现在一样）；现有测试一条不改就要全过
- 手动操作只放开"dry-run 不执行"；`clean_reply`（不能自称真人）、说话限速、牵手 `force`、画面黑着、看人的各项护栏全部保留
- 视角步数 1～`camera.MAX_STEPS`（4）；视角操作名 `left right up down zoom_in zoom_out`（`camera.KEYS`）
- 控制只在 `viewer.host` ∈ {`127.0.0.1`, `localhost`, `::1`} 时挂；`POST /control` 要 `Content-Type: application/json` + `X-Skydango: 1`；
  `/control*` 的 `Host` 必须是 `127.0.0.1:<端口>` / `localhost:<端口>` / `[::1]:<端口>`；请求体上限 4096 字节
- 状态码：参数不对 400、请求头 / Host 不对 403、超过 4 KB 413、没挂控制 404
- 页面数据一律 `textContent`；不引外部库；沿用 CSS 变量
- 在 worktree 里跑 `python -m skydango` / `python -c` 要加 `PYTHONPATH=src`（包是可编辑安装、指向主仓库）；pytest 由 conftest 处理
- 测试命令：`python -m pytest -q`

## Review Focus

1. **dry-run 下手动说的话必须进 `self_filter`**，否则读聊天把团子自己那句当成别人说的，又叫醒大脑去"回复自己" → Task 1 `test_live_say_in_dry_run_sends_and_remembers`
2. **dry-run 下手动转了镜头，退出要转回来**（现在 dry-run 退出跳过复原）→ Task 1 `test_shutdown_resets_camera_moved_by_hand_in_dry_run`
3. **别的网页用普通表单 POST（`text/plain`、不带自定义头）** 必须被拒 → Task 3 `test_control_rejects_cross_site_style_requests`
4. **画布被 CSS 缩放**：点画面换算原图坐标要按画布显示尺寸算，不是画布像素 → Task 4 `test_click_maps_to_frame_pixels`
5. **身体还没建好时页面已经打开**：控制栏要等到挂上后出现，而不是永远不出现 → Task 4 `test_page_retries_control_options`

---

### Task 1: 身体——`live` 参数、`check_friend_at`、退出总是复原镜头

**Files:**
- Modify: `src/skydango/brain/body.py`（`say`、`emote`、`camera_move`、`camera_reset`、`capture_around`、`sweep_around`、`check_friend`、`shutdown`）
- Test: `tests/test_brain_body.py`

**Interfaces:**
- Produces:
  - `Body.say(text, live=False)`、`Body.emote(name, force=False, live=False)`、`Body.camera_move(action, steps=1, live=False)`、
    `Body.camera_reset(live=False)`、`Body.capture_around(live=False)`、`Body.sweep_around(live=False)`
  - `Body.check_friend_at(sx: int, sy: int, live: bool = False) -> list[dict]`：原图像素坐标；返回和原来 `check_friend` 一样的块列表（图片块 + 文字块）
  - `Body.check_friend(x, y)`：外层不变（校验 look 新鲜度、换算坐标）→ 调 `check_friend_at`
  - 每个方法里的 dry-run 判断统一成 `dry = self.cfg.reply.dry_run and not live`

- [ ] **Step 1: 写失败的测试**（用现有 `body()`、`FakeEmotes`、`FakeCamera`、`friend_body()`）

```python
def test_live_say_in_dry_run_sends_and_remembers(clock):
    b, device, _, _ = body(clock)                      # dry-run
    assert b.say("晚安", live=True) == "已发送：晚安"
    assert ("text", "晚安") in device.calls
    assert b.self_filter.is_self("晚安", clock())       # 读聊天不会把团子自己这句当成别人说的
def test_live_say_still_filtered_and_rate_limited(clock):
    # dry-run + live：say("我是真人啊", live=True) 抛 ToolError(match="真人")；连说两句第二句抛 match="太快"
def test_live_emote_and_camera_in_dry_run(clock):
    # emote("鞠躬", live=True) == "做了「鞠躬」" 且 emotes.done == ["鞠躬"]；holding 时不 force 仍拒绝
    # camera_move("left", 2, live=True) → cam.moves == [("left", 2)]；camera_reset(live=True) 不以 "dry-run" 开头
    # blackout 时 camera_move(..., live=True) 仍抛 match="黑"
def test_shutdown_resets_camera_moved_by_hand_in_dry_run(clock):
    # FakeCamera 加一个 resets 计数；dry-run body，camera_move("left", 1, live=True) 后 shutdown() → resets == 1
def test_check_friend_at_uses_frame_pixels(clock):
    # friend_body(clock, live=False)；不 look 直接 check_friend_at(1200, 450, live=True) → checker.calls == [(1200, 450)]
    # 功能关了 → match="没开"；(100, 1050) → match="按钮栏"；holding → match="牵着"
def test_capture_and_sweep_around_live_in_dry_run(clock):
    # capture_around(live=True) 调了 camera.around（FakeCamera 加 around：记次数，返回 [frame]*4）；不 live 仍只截当前一张
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_body.py -q`
Expected: 新测试 FAIL（`unexpected keyword argument 'live'` / 没有 `check_friend_at`），老测试照过

- [ ] **Step 3: 实现**：按 Interfaces 改；`check_friend` 的"look 新鲜度 + 1280×720 → 原图换算"留在外层，其余搬进 `check_friend_at`；`shutdown` 去掉 `not self.cfg.reply.dry_run` 条件

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_body.py tests/test_brain_tools.py tests/test_brain_mcp.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/body.py tests/test_brain_body.py
git commit -m "feat(brain): 身体方法加 live（dry-run 下也真执行，给手动控制用）、check_friend_at 按原图坐标、退出总是复原镜头"
```

---

### Task 2: `ManualControl`

**Files:**
- Create: `src/skydango/brain/manual.py`
- Test: `tests/test_brain_manual.py`（`from test_brain_body import body, FakeEmotes, FakeCamera, friend_body` 复用夹具）

**Interfaces:**
- Consumes: Task 1 的身体方法（都带 `live=True` 调）；`brain.tools.AROUND_TIMEOUT`；`camera.KEYS`、`camera.MAX_STEPS`
- Produces:
  - `ManualControl(body, eyes=None, events=None)`
  - `options() -> dict`：`{"emotes": list[str], "camera": list[str], "max_steps": int, "friend_check": bool, "max_chars": int, "dry_run": bool}`
    （`emotes` = `body.emotes.available()` 或 `[]`；`camera` = `list(KEYS)` 或 `[]`（没有 `body.camera`）；`friend_check` = `body.friend_checker is not None and cfg.friend_check.enabled`）
  - `run(action: str, args: dict) -> dict`：`{"ok": bool, "text": str}`；参数不对抛 `ValueError`（中文原因）
  - 放事件：`events.put("manual", f"主人在面板上手动让团子{描述}：{结果}")`；描述按动作：`说了「…」`、`做了动作「…」`、`转了视角（左转 ×2）`、`把视角复位`、`环视了一圈`、`点了画面上 (x, y) 的人`
  - `check_friend` 的结果是块列表：只取文字块拼起来作为 `text`

- [ ] **Step 1: 写失败的测试**

```python
def test_say_runs_live_and_tells_brain(clock):
    # dry-run body；ManualControl(b, events=events).run("say", {"text": "晚安"})
    # 让 body.call 在本线程直接执行：先设 b._thread = threading.get_ident()（Body.call 在身体线程里调时直接执行 fn）
    # → {"ok": True, "text": "已发送：晚安"}；events.drain() 里有 kind == "manual" 且 "说了「晚安」" in text
def test_tool_errors_become_not_ok_and_no_event(clock):
    # say("我是真人") → ok False、"真人" in text；events 里没有 manual
def test_bad_arguments_raise_value_error(clock):
    # run("fly", {})、run("say", {})、run("say", {"text": 3})、run("camera", {"action": "left", "steps": 5})、
    # run("camera", {"action": "spin"})、run("check_friend", {"x": "a", "y": 1}) 都抛 ValueError
def test_emote_force_and_camera(clock):
    # emote {"name": "鞠躬", "force": True} 牵手时也做成；camera {"action": "left", "steps": 2} → cam.moves == [("left", 2)]
def test_look_around_uses_sweep_or_eyes(clock):
    # env 有 sweep → 走 sweep_around；否则 capture_around 后 eyes.describe_around(frames, now) 的返回作为 text；
    # eyes=None → text 以 "转完了" 开头
def test_check_friend_returns_text_only(clock):
    # friend_body；run("check_friend", {"x": 1200, "y": 450}) → ok True，text 含 "已经关上"，不含 base64
def test_options(clock):
    # 有 FakeEmotes / FakeCamera → emotes == ["鞠躬"]，camera == ["left","right","up","down","zoom_in","zoom_out"]，max_steps == 4，
    # friend_check False，max_chars == 40，dry_run True；没有 emotes / camera → 空列表
def test_unexpected_error_is_reported(clock):
    # 身体方法抛 RuntimeError → ok False、text 以 "出错了" 开头
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_brain_manual.py -q`
Expected: FAIL，`No module named 'skydango.brain.manual'`

- [ ] **Step 3: 实现 `src/skydango/brain/manual.py`**（模块 docstring 说明：给可视化网页手动试身体用、总是真执行、照样过护栏、做成后告诉大脑）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_brain_manual.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/brain/manual.py tests/test_brain_manual.py
git commit -m "feat(brain): ManualControl——手动让身体说话 / 做动作 / 转视角 / 环视 / 看人，做成后告诉大脑"
```

---

### Task 3: `/control` 接口和安全校验

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`Viewer.control = None`；`do_GET` 加 `/control/options`；新增 `do_POST`；Host 校验辅助函数）
- Test: `tests/test_viewer.py`

**Interfaces:**
- Consumes: Task 2 的 `options()`、`run(action, args)`（`ValueError` → 400）
- Produces: `Viewer.control`；`LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")`（Task 5 用来判断本机模式）；`MAX_BODY = 4096`

- [ ] **Step 1: 写失败的测试**（假控制对象 `FakeControl`：记下调用，`options()` 返回固定 dict，`run` 对 `action == "bad"` 抛 `ValueError("不认识")`）

```python
def post(url, body: bytes, headers: dict): ...   # urllib.request.Request(method="POST")；返回 (status, json 或 None)，HTTPError 也返回 status

def test_control_404_without_control(): ...          # GET control/options → 404；POST control → 404
def test_control_options_and_run():
    # GET control/options → 200 且等于 FakeControl.options()
    # POST {"action":"say","args":{"text":"晚安"}} + 两个头 → 200、{"ok": True, ...}，FakeControl 收到 ("say", {"text": "晚安"})
def test_control_rejects_cross_site_style_requests():
    # 缺 X-Skydango → 403；Content-Type text/plain → 403；Host: evil.example:端口 → 403（GET options 也 403）；FakeControl 没被调用
def test_control_bad_body():
    # 5000 字节 → 413；"不是json" → 400；"[1,2]" → 400；action == "bad" → 400 且返回 {"ok": False, "text": "不认识"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viewer.py -q -k control`
Expected: FAIL（没有 `do_POST`，501 / 404）

- [ ] **Step 3: 实现**：Host 校验 = 去掉端口后在 `LOCAL_HOSTS`（`[::1]` 去方括号）且端口等于服务实际端口；
POST 先查 Host → 头 → 长度（`Content-Length` 缺失或 > `MAX_BODY` → 413）→ JSON → `control.run`；返回体 `json.dumps(ensure_ascii=False).encode(errors="replace")`

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viewer.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add src/skydango/vision/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): /control 接口——只收本机、带自定义头的 JSON 请求（防跨站、防 DNS 重绑定）"
```

---

### Task 4: 页面上的「手动控制」栏

**Files:**
- Modify: `src/skydango/vision/viewer.py`（`PAGE`：HTML、CSS、`// ---- control ----` 之后的 JS）
- Test: `tests/test_viewer.py`
- 临时：`tmp/control_page_demo.py`（不提交；假 `ManualControl`，起在 8767）

**Interfaces:**
- Consumes: Task 3 的 `/control/options`、`POST /control`
- 页面行为：按 spec §3「页面」逐条实现。纯函数（放在行首，便于 node 测）：
  - `function toFrame(clientX, clientY, rect, width, height)` → `[x, y]` 原图像素（`Math.round((clientX - rect.left) * width / rect.width)`，y 同理）
  - `function controlLine(action, args, res)` → 结果记录的一行文字（不含时间）

- [ ] **Step 1: 写失败的测试**

```python
def test_page_has_control_section(): ...  # id="control"、id="ctl-say"、id="ctl-emote"、id="ctl-camera"、id="ctl-pick"、id="ctl-log"、"/control/options"、"X-Skydango"
def test_control_script_never_uses_innerhtml(): ...  # "// ---- control ----" 之后到 "// ---- brain ----" 之前不含 innerHTML
def test_click_maps_to_frame_pixels():
    # 用 _node 执行 toFrame(650, 400, {left:50, top:100, width:960, height:540}, 1920, 1080) → [1200, 600]
def test_page_retries_control_options():
    # control 段脚本里 404 分支有 setTimeout(…, 3000)（字符串断言）
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viewer.py -q -k "control or frame_pixels"`
Expected: 新测试 FAIL

- [ ] **Step 3: 实现页面**（`PAGE` 不是原始字符串：JS 里的 `\n` 写成 `\\n`；改完用 `node --check` 查语法）

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viewer.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 浏览器里核对**：`tmp/control_page_demo.py` 起 `Viewer(port=8767)`，挂假控制（`run` 记下调用、sleep 1 秒模拟身体、`say` 含"真人"时返回失败）和一个 BrainTrace；
内置浏览器打开，逐项核对：dry-run 黄色提示、说话（回车 / 按钮、字数）、动作下拉 + force、六个视角按钮 + 步数 + 复位 + 环视、
看人（选人十字 → 确认 → 坐标对）、灰掉的看人按钮提示、请求中全部置灰、结果记录红绿、`<b>` 原样显示、窄屏布局；
再测"先 404 后挂上"：脚本启动 5 秒后才设 `v.control`，页面应在几秒内出现控制栏。完了停掉脚本、确认没有残留 python

- [ ] **Step 6: 提交**

```bash
git add src/skydango/vision/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): 页面上的手动控制栏（说话、动作、视角、环视、看人，结果记录）"
```

---

### Task 5: `run --brain --view` 接线、文档、真机验证

**Files:**
- Modify: `src/skydango/cli.py`（`_run_brain`）、`CLAUDE.md`、spec
- Test: `tests/test_cli_brain.py`

**Interfaces:**
- Consumes: Task 2 `ManualControl`、Task 3 `Viewer.control` / `LOCAL_HOSTS`

- [ ] **Step 1: 写失败的测试**（照 `test_run_brain_with_viewer_records_turns`，`FakeViewer` 加 `control = None`）

```python
def test_run_brain_with_viewer_attaches_control(tmp_path, monkeypatch):
    # cfg.viewer.host 默认 127.0.0.1 → 跑完 isinstance(v.control, ManualControl)
def test_run_brain_on_lan_has_no_control(tmp_path, monkeypatch, caplog):
    # cfg.viewer.host = "0.0.0.0" → v.control is None，日志里有 "局域网模式下关掉了手动控制"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_cli_brain.py -q`
Expected: 新测试 FAIL

- [ ] **Step 3: 实现**：`_run_brain` 建好 `body`、`eyes`、`events` 后，`viewer is not None` 时按 `cfg.viewer.host in LOCAL_HOSTS` 挂 `ManualControl(body, eyes, events)` 或打 WARNING

- [ ] **Step 4: 跑全部测试**

Run: `python -m pytest -q`
Expected: 全部 PASS（基线 598 passed, 5 skipped + 新测试）

- [ ] **Step 5: 真机验证**（**先问用户现在方不方便**；按 spec §7；改之前截图 + `PYTHONPATH=src python -m skydango emotes wheel` 记下轮盘）

Run（在主仓库目录跑，用 worktree 的代码）：`PYTHONPATH=<worktree>/src python -m skydango run --brain --view --duration 300`
逐个试：说一句、做一个动作、左转两步再复位、环视一圈；看人按钮是灰的。每步截图确认游戏里真的发生、时间线上有 `manual` 事件叫醒大脑。
退出后截图核对镜头原位、`emotes wheel` 和原来一样；`Get-CimInstance Win32_Process` 确认没有残留 python。

- [ ] **Step 6: 文档**：CLAUDE.md「识别可视化」「统管大脑」两节各补一句；spec 状态改"已实现"，写上真机核对结果

- [ ] **Step 7: 提交**

```bash
git add src/skydango/cli.py tests/test_cli_brain.py CLAUDE.md docs/superpowers/specs/2026-09-28-viewer-manual-control-design.md
git commit -m "feat(brain): run --brain --view 时挂上手动控制（只在本机模式）；文档"
```
