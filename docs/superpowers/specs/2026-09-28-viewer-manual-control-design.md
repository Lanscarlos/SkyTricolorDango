# 可视化网页：手动控制身体 — 设计

日期：2026-09-28　状态：**设计已和用户确认，待实现**

在识别可视化网页（`vision/viewer.py`，设计见 `2026-09-28-viewer-design.md`、`2026-09-28-viewer-brain-trace-design.md`）上
加一栏「手动控制」：`run --brain --view` 时，在网页上直接让身体说话、做动作、转视角、看人。

## 目标

**手动试身体的功能。** 绕过大脑，点按钮直接让身体做一件事，看这些能力在游戏里到底好不好用；
大脑同时开着，能在大脑时间线上看到它对手动操作的反应。

成功的样子：大脑在跑，在面板上输入一句话点「说」，团子在游戏里真的说出来，面板显示「已发送：……」；
点「左转」镜头真的转了；做不成的（说得太快、牵着手……）显示原因；退出时镜头、轮盘恢复原样。

## 需求（已和用户确认）

1. **挂在 `run --brain --view` 上**，大脑开着也能用，和大脑共用同一个身体、排队执行。`view`（只看不动）和普通 Agent 模式不加。
2. 四类操作：**说话 say、动作 emote、视角 camera（含复位、环视一圈）、看人 check_friend**。
3. **手动的总是真执行**：dry-run 下大脑照样只打印，但面板上点的会真的在游戏里执行。
4. 手动操作照样过身体的护栏（`clean_reply` 不能自称真人、说话限速、牵着手做动作要确认、画面黑着不转……），只放开"dry-run 不执行"这一条。
5. 手动做成一件事要告诉大脑（事件），免得它看到自己没说过的话犯糊涂。
6. 只有本机能控制；局域网模式（`viewer.host = "0.0.0.0"`）不开控制；别的网页不能借浏览器偷偷发指令。

## 方案选择

**方案 A：网页加控制栏，经 `ManualControl` 交给身体（`Body.call`），身体方法多一个 `live` 参数。**

放弃的方案：
- **B. 走大脑的工具箱 `ToolBox.run`**：手动操作会占大脑那一轮"最多 6 次工具、2 句话"的名额，还混进大脑的用量统计。
- **C. 当成主人命令交给大脑去做**：那是"给大脑下指令"，不是这次要的；也没法保证它照做。

## 1. 身体（`brain/body.py`）

- `say`、`emote`、`camera_move`、`camera_reset`、`capture_around`、`sweep_around` 加参数 `live: bool = False`。
  现在的判断 `if self.cfg.reply.dry_run:` 改成"dry-run **且不是 live**"才只打印。大脑调用不传，行为完全不变。
- `check_friend` 拆两层：
  - 外层 `check_friend(x, y)`（大脑用）不变：坐标按 `look(image=true)` 那张图（`brain.image_size`，1280×720），要求 `max_look_age` 秒内看过。
  - 核心抽成 `check_friend_at(sx, sy, live=False)`：坐标是原始截图像素。护栏照旧——功能开关 `[friend_check] enabled`、
    画面黑着、牵着手、底部按钮栏（`env.roi` 之外）、聊天面板挡着、`min_interval`；截图照样存 `runs/<…>/friend-check/`。
    外层换算好坐标后调它。
  - 手动时返回给网页的是文字（原来的 note），图片不传给网页（截图已存盘，画面也会在网页上刷新）。
- **退出复原**：`shutdown()` 去掉"dry-run 不复原镜头"的条件，总是调 `camera.reset()`（它只撤销记下的偏移，没动过就什么都不做）。
  轮盘本来就总会 `restore()`。

## 2. 手动控制（新模块 `brain/manual.py`）

`ManualControl(body, eyes=None, events=None)`：

- `options() -> dict`：网页填按钮用
  ```json
  {"emotes": ["鞠躬", "害羞"], "camera": ["left", "right", "up", "down", "zoom_in", "zoom_out"], "max_steps": 4,
   "friend_check": false, "max_chars": 40, "dry_run": true}
  ```
  `emotes` 取 `body.emotes.available()`（没开动作为空列表）；`camera` 为空列表表示没有视角控制；`friend_check` = 有 checker 且 `enabled`。
- `run(action: str, args: dict) -> dict`：返回 `{"ok": bool, "text": str}`
  | action | args | 身体调用 |
  |---|---|---|
  | `say` | `text`（非空字符串） | `body.say(text, live=True)` |
  | `emote` | `name`、`force`（可选，默认 false） | `body.emote(name, force, live=True)` |
  | `camera` | `action` ∈ camera 的 6 个操作、`steps` ∈ 1～`MAX_STEPS`（4），默认 1 | `body.camera_move(action, steps, live=True)` |
  | `camera_reset` | 无 | `body.camera_reset(live=True)` |
  | `look_around` | 无 | 开了感知层（env 有 `sweep`）→ `body.sweep_around(live=True)`；否则 `body.capture_around(live=True)` 后交给眼睛 `eyes.describe_around`，没开眼睛就回"转完了，截了 N 张" |
  | `check_friend` | `x`、`y`（原图像素，整数） | `body.check_friend_at(x, y, live=True)` |
  - 参数不对（不认识的 action、缺字段、类型不对、步数越界）→ 抛 `ValueError`，网页回 400，不交给身体。
  - 身体调用一律经 `body.call(fn, timeout)`（`look_around` 用和工具箱一样的 `AROUND_TIMEOUT`），在身体线程里执行。
  - `ToolError` → `{"ok": false, "text": 原因}`；其他异常记 `log.exception`，回 `{"ok": false, "text": "出错了：…"}`。
- **告诉大脑**：成功后 `events.put("manual", "主人在面板上手动让团子" + 描述 + "：" + 结果)`，例如
  "主人在面板上手动让团子说了「晚安」：已发送：晚安"。会叫醒大脑一次。失败的不放。放事件出错只记 DEBUG。
- 每次操作记一行 INFO：`手动：say {"text": "晚安"} → 已发送：晚安`。

## 3. 网页

### 接口（`vision/viewer.py`）

- `Viewer.control = None`（`ManualControl`；只在 `run --brain --view`、且本机模式时挂上）。
- `GET /control/options` → `control.options()`；没挂上 → 404。
- `POST /control`，请求体 `{"action": "...", "args": {...}}` → `control.run(...)` 的结果（200）。
  请求会等身体做完才返回（几秒）；服务是多线程的，不影响 `/snapshot`、`/brain`。

### 安全

1. **只在本机**：`viewer.host` 不是 `127.0.0.1` / `localhost` / `::1` 时不挂控制，启动时 WARNING"局域网模式下关掉了手动控制"。
2. **防别的网页借浏览器发指令（CSRF）**：`POST /control` 必须 `Content-Type: application/json` 且带请求头 `X-Skydango: 1`，
   否则 403。跨站的这种请求浏览器会先发预检（OPTIONS），我们不响应它，发不过来。
3. **防 DNS 重绑定**：`/control`、`/control/options` 检查 `Host` 头，必须是 `127.0.0.1:<端口>`、`localhost:<端口>`、`[::1]:<端口>`，否则 403。
4. 请求体超过 4 KB → 413；不是合法 JSON / 不是对象 → 400。

### 页面：画面和大脑时间线之间加一栏「手动控制」

页面启动时请求一次 `/control/options`：404 就整栏不显示；之后每次操作完再刷新一次（动作列表会变）。

- 栏头：「手动控制」；`dry_run` 为真时显示黄色提示"手动操作会真的在游戏里执行（大脑是 dry-run 也一样）"。
- **说话**：输入框 + 「说」按钮，回车也发；旁边显示 `字数 / max_chars`。
- **动作**：下拉框（`emotes`）+ 「做」+ 勾选框「牵着手也做（会松手）」→ `force`。`emotes` 为空时整行置灰。
- **视角**：左转 / 右转 / 抬头 / 低头 / 拉近 / 拉远 六个按钮 + 步数输入框（1～`max_steps`，默认 1）+「复位」+「环视一圈」。`camera` 为空时置灰。
- **看人**：「在画面上选人」→ 下一次点画面换算成原图坐标（画布坐标 × 原图宽 / 画布宽），画面上画个十字，
  弹出确认「点 (x, y) 这个人？」后才发。`friend_check` 为假时按钮置灰，悬停提示 `[friend_check] enabled = false`。
- **结果记录**：按钮下面最近 10 条，`HH:MM:SS 说「晚安」→ 已发送：晚安`，成功绿、失败红；400 / 403 / 网络错误也记一条红的。
- 请求没回来时所有控制按钮置灰，一次只做一件事。
- 所有来自数据的文字用 `textContent`；样式沿用现有 CSS 变量；窄屏照常在画面下方。

## 4. 接线（`cli.py`）

`_viewer(cfg, brain=True)` 在打开浏览器之前挂 `BrainTrace`（已有）；`_run_brain` 建好 body / eyes / events 后，
本机模式下 `viewer.control = ManualControl(body, eyes, events)`。控制在身体准备好之前请求 `/control/options` 会拿到 404——
所以页面拿到 404 后**每 3 秒再试一次，直到拿到为止**（和大脑时间线不同：控制对象要等身体建好才有，比浏览器打开晚几秒是必然的）。

## 5. 边界情况

| 情况 | 处理 |
|---|---|
| 大脑正在用身体（比如在环视） | 手动命令在身体队列里排队，做完才轮到；超时照 `Body.call` 的提示 |
| 手动说话太快 | 身体限速拒绝，面板显示"说得太快了" |
| 说了"我是真人"之类 | `clean_reply` 拦下，面板显示原因 |
| 牵着手做动作 | 不勾「牵着手也做」→ 拒绝并说明；勾了 → 做（会松手） |
| 程序正在退出 | "身体已经停了" |
| dry-run 下手动转了视角 / 换了轮盘 | 退出时照样复原（§1） |
| 局域网模式 | 不挂控制，页面不显示控制栏 |

## 6. 测试（`python -m pytest -q`，不需要模拟器）

- `tests/test_brain_manual.py`（假设备 / 假身体夹具，照 `tests/test_brain_body.py`）：
  dry-run 下手动 `say` 真的发了、`self_filter` 记住了、事件队列多一条 `manual`，而大脑的 `say` 仍只打印；
  `clean_reply` 拦截、限速、牵手 + `force`、画面黑着不转；参数校验（不认识的 action、缺字段、步数越界）；
  `check_friend_at` 用原图坐标、功能没开报错；`options()` 内容；`look_around` 两条路。
- `tests/test_brain_body.py`：dry-run 下手动转过视角，`shutdown()` 把镜头转回来。
- `tests/test_viewer.py`：没挂控制 404；缺 `X-Skydango` / `Content-Type` 不对 / `Host` 不对 → 403；超过 4 KB → 413；坏 JSON → 400；
  正常请求返回结果；页面有控制栏元素、控制栏脚本不用 `innerHTML`。
- `tests/test_cli_brain.py`：`run --brain --view` 时 `viewer.control` 挂上；局域网模式不挂。

## 7. 真机验证（需要用户在场、游戏在前台）

改之前先截图、`emotes wheel` 读一遍轮盘记下原状态。`run --brain --view --duration 300`（dry-run），在面板上逐个试：
说一句、做一个动作、左转两步再复位、环视一圈；看人默认关，只确认按钮是灰的。每一步截图确认游戏里真的发生了，
时间线上看得到大脑被叫醒、有反应。退出后核对镜头回到原位、轮盘和原来一样。

## 8. 文档

CLAUDE.md「识别可视化」「统管大脑」两节各补一句；本文档状态改"已实现"。
