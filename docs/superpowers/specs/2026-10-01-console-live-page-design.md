# 管理面板「真机团子」页改版（照沙盒三栏）

状态：**代码已完成**（计划 `docs/superpowers/plans/2026-10-01-console-live-page.md`），待真机验证（§6 第 4 步）

2026-10-01。改的是 `python -m skydango console` 的 `#live` 页，外加 viewer 和身体的少量后端。上一版设计见 `2026-10-01-console-redesign-design.md` §4.2。

## 1. 为什么改

2026-10-01 晚真机 live 跑的时候用户截图：「这个页面也太丑了，能不能重新排版设计一下，可以参考沙盒那个」。毛病：
- 跑着时左边是 iframe 嵌了整个识别可视化网页（`vision/viewer.py` 的 PAGE），深色主题，和管理面板的浅色纸感对不上
- iframe 有固定高度，里面还要自己滚，游戏画面被裁掉一截，手动控制、大脑时间线都要在小框里翻
- 右边状态卡片 + 原始日志，下面空了一大块
- 看不到聊天记录：听到了什么、团子说了什么、心里想了什么，只能去大脑时间线里一轮一轮翻

成功的样子：跟沙盒页一个样子，一屏里同时看到团子现在怎么样、游戏画面、大脑在想什么、聊天记录。

## 2. 布局

整页是顶栏 + 三栏，占满一屏（`height: 100vh` 减顶栏），每栏自己滚。不再用 iframe。配色和字体都用管理面板的（`console.css` 的 `:root` 变量）。

### 2.1 顶栏

- **停着**：启动选项横排一行：模式（统管大脑 / 普通 Agent）· 真的发送（勾上后旁边红字「会在游戏里真的说话、做动作」）· 聊天时做动作 · 时长（秒，空 = 一直跑）·「叫醒团子」。
  排法和沙盒顶栏一样。预检问题放在顶栏下面的黄条里，和沙盒的 `sb-problems` 一样，每条带「去设置 →」。沙盒在跑时整行置灰，按钮写「沙盒在跑，先下线」（同现在）
- **跑着**（含启动中、收尾中）：状态 chip（运行中 / 正在启动 / 在收尾）· 真发还是只看 · 已运行多久 · 运行目录（小字，能选中复制）·「日志」按钮
- **横幅**：强杀、孤儿团子的横幅照旧放顶栏下面（`live.js` 现有逻辑）
- **日志抽屉**：点「日志」从右边滑出，宽约 560px，内容就是现在的日志 `<pre>`（`/api/logs` 增量拉取、ERROR / WARNING 上色、在底部才跟到底）。
  点按钮或按 Esc 收起。抽屉关着也照样拉日志，打开就是最新的。团子出错停下时自动打开

### 2.2 左栏「团子」

- **上半「现在」**：
  - 内心部分直接用 `Inner.renderNow`（数据 `GET api/inner`，团子在跑时已经合并了子进程的实时数据），和沙盒一样带「完整内心 →」（切到团子源、跳内心页）。
    每 5 秒刷新一次。内心层没开（`api/inner` 报错或者没数据）时这一块只写「内心层没开」
  - 下面接真机特有的几项，来自 `live/status` 的 `info`，2 秒一次：身边的好友、陌生人、互动请求、开着的面板、牵着手、正在做、刚说过、场合、聊天面板。
    排成紧凑的「标签：值」两列，不再是一张张大卡片；身边的好友是名字小标签
- **下半「手动控制」**（`livectl.js`，接口同 viewer：`live/control/options`、`POST live/control`）：
  - 一行一项：说话（输入框 + 字数 + 说）· 动作（下拉 + 做）· 视角（左转 右转 抬头 低头 拉近 拉远 + 步数 + 复位 + 环视一圈）· 看人（在画面上选）· 盯人（名字 + 在画面上选 + 秒 + 盯 + 停下）· 面板（读面板 关面板，没开 `[panels]` 时不显示）
  - 按钮可不可点、提示文字照搬 viewer 的 `ctlLock` 逻辑；正在做的时候整块锁住，写「正在做：…」
  - 结果记在下面一小段（最近 10 条，时间 + 做了什么 → 结果，成败上色）
  - 在画面上选人后的确认用 `ask()`（不用原生 `confirm`，内嵌浏览器里用不了）
  - 大脑是 dry-run 时顶上一行提示「手动操作会真的在游戏里执行」（同 viewer）
  - `control/options` 404（身体还没建好）时 3 秒后重试，期间整块写「身体还没准备好」

### 2.3 中栏

- **上面：游戏画面**。画在页面里的 `<canvas>`，宽度占满中栏，16:9 不裁，高度跟着宽度走（最高不超过中栏高度的 60%）。
  - 画面上方一行小工具：暂停 / 继续、隐藏框 / 显示框、存图（下载带框的 PNG）、帧率和分辨率（「3.0 帧/秒 · 1920×1080」，连不上时红字「连不上」）、「图例」按钮（点开浮层显示颜色图例，默认收着）
  - 鼠标悬停有 `desc` 的框：画一行装扮描述（同 viewer）
  - 手动控制的「看人」「盯人 · 在画面上选」进入选人模式后点画面（同 viewer：盯人按快照里的框认名字，看人画十字再 `ask()` 确认）
  - 数据：长轮询 `live/snapshot?after=<seq>`（同 viewer），只在这一页开着时拉；切走就停，回来接着拉
- **下面：大脑控制台**。`mountBrainConsole(容器, "live/brain")`，和沙盒同一个组件（`brainlog.js`）。普通 Agent 模式没有 `/brain`（404），这一块写「普通 Agent 没有大脑」

### 2.4 右栏「聊天记录」

和沙盒聊天记录同一个样式（渲染函数共用，见 §4）：别人说的在右、团子说的在左（樱花底）、动作是旁白、来去是分隔线、被拦下的话加删除线和原因、心里想的是「── 心里：… ──」。
在底部才跟到底，最多留 500 行。只读，**没有输入框**（让团子说话用左栏手动控制）。数据见 §3.2。

### 2.5 停着时

三栏照样摆着，每栏写一句占位：左「叫醒之后这里是团子现在的样子」、中「叫醒之后这里是游戏画面和大脑」、右「叫醒之后这里是聊天记录」。
刚停下（exited / crashed）时三栏保留最后看到的内容（聊天记录、大脑、最后一帧），直到下次叫醒才清空，方便停了再看一眼。

### 2.6 窄屏

宽度 < 1000px 时三栏变上下排（同沙盒的做法），只保证能用。

## 3. 数据和后端

### 3.1 画框代码共用：`vision/static/stage.js`

把 viewer PAGE 里内联的画框代码抽出来：颜色表 `COLORS`、名字表 `NAMES`、`drawStage(ctx, img, snapshot, opts)`（画图 + 框 + 标签 + 十字 + 悬停描述）、`nameAt(boxes, x, y)`、`toFrame(clientX, clientY, rect, w, h)`。
- 顶层不碰 `document`，`module.exports` 导出纯函数，node 里能测（同 `brainlog.js` 的写法）
- viewer 的 PAGE 改成 `<script src="stage.js">` 后调用它，行为不变：把 `stage.js` 加进 `viewer.STATIC`（和 `brain_trace.js` 一样）
- 管理面板已经经 `/static/<名字>` 提供 `viewer.STATIC` 里的文件（`static_asset`），`console.html` 直接 `<script src="static/stage.js">`，不用等子进程

### 3.2 真机聊天记录

- `sandbox/transcript.py` 的 `Transcript` 挪到 `brain/transcript.py`（沙盒改成从新位置引用，`sandbox/transcript.py` 不留转发）。它要一个有 `wall()` 的钟：沙盒用 `SimClock`，真机用身体的墙钟
- **身体加回调** `Body.on_line: Callable[[str, str, str, str], None] | None`（kind, text, who, why），默认 None。出错只记日志（同 `on_blocked`）。只在真机的 `_run_brain` 里接上；沙盒照旧在 world / control 里记，不接 `on_line`，免得重复。调用点：
  - **heard**：读到的每条新消息（身体处理新消息的那一圈，和发 `chat` / `owner_command` 事件同一处），who = 说话人（看不出写「（看不出是谁）」）
  - **said**：`say` 走到发送那一步之后。dry-run（没真的发）时 text 后面加「（dry-run，没真的发）」。手动控制让团子说的也记
  - **act**：`emote` 做成之后「（团子做了 X）」；反射动作「（团子下意识地 X）」；身体替大脑冒输入气泡「（团子头顶冒出输入气泡）」（同沙盒的文字）
  - **blocked**：`_blocked` 里，who = "团子"，why = 原因（`on_blocked` 本身不动，真机另外走 `on_line`）
  - **event**：心里想的：`mused` 记下时「── 心里：… ──」
- **来去这类事件**不在身体里一处处加：`EventQueue` 加 `tap(fn)`，每放一个事件调 `fn(kind, text, who)`（合并没合并都调）；
  `brain/transcript.py` 的纯函数 `event_line(kind, text, who)` 把好友来了 / 走开 / 回来、牵手 / 松手、整屏黑 / 恢复、陌生人又回来了变成一行
  （「── 小明 来到身边 ──」），别的事件返回 None 不记。真机组装时 `events.tap(...)` 接上
- **viewer 加** `viewer.chat`（`Transcript | None`）和 `GET /chat?after=<seq>&wait=<秒>`：有比 `after` 新的行立刻返回，否则最多等 `wait` 秒（上限 `viewer.WAIT` 2 秒：管理面板转发 `/live/*` 的超时只有 5 秒），返回 `{"v": 最新 seq, "lines": [...]}`。
  没挂 `chat` 时 404。Host 校验同别的接口
- 管理面板经 `live/chat` 拿，不用改转发代码（`/live/*` 的 GET 都转）。普通 Agent 模式不挂 `chat`，右栏写「普通 Agent 没有聊天记录」
- 只在内存里，最近 500 行；停了就没了，要复盘看 `runs/<…>/`

### 3.3 不动的

`live/status`、`live/control`、`live/brain`、`/api/logs`、`api/inner` 的后端不动。单独跑 `view` / `run --view` 时 viewer 的网页（深色、带手动控制和大脑时间线）只是改成引用 `stage.js`，样子和行为不变。

## 4. 文件

| 文件 | 改什么 |
|---|---|
| `vision/static/stage.js` | 新：画框（§3.1） |
| `vision/viewer.py` | `STATIC` 加 `stage.js`；PAGE 引用它、删掉内联的画框代码；加 `chat` 属性和 `/chat` |
| `brain/transcript.py` | 从 `sandbox/transcript.py` 挪过来 |
| `sandbox/*.py`、`console/*.py` | 改 import |
| `brain/body.py` | `on_line` 和各调用点 |
| `brain/events.py` | `EventQueue.tap` |
| `cli.py` | 真机 `_run_brain` 建 `Transcript`、接 `body.on_line`、挂 `viewer.chat` |
| `console/static/chatlog.js` | 新：聊天行渲染（`lineKind`、`chatLine`、`appendChat`），从 `sandbox.js` 抽出来，沙盒和真机共用 |
| `console/static/livectl.js` | 新：手动控制 |
| `console/static/live.js` | 重写：顶栏、三栏、画面长轮询、聊天长轮询、状态、日志抽屉 |
| `console/static/console.html` | `#page-live` 整段重写，加 `static/stage.js`、`chatlog.js`、`livectl.js` 的 `<script>` |
| `console/static/console.css` | live 页的样式；颜色只用 `:root` 变量（有测试查） |
| `console/static/sandbox.js` | 改用 `chatlog.js` |

`console/server.py` 只提供 `static/` 目录里真有的文件，新加的 js 放进去就能用。

## 5. 测试（`python -m pytest -q`）

- `Transcript` 挪位置后原有的测试照过（改 import）
- 身体：heard / said（真发、dry-run、手动）/ act（动作、反射、气泡）/ blocked / event / 心里，各调一次 `on_line`，参数对；`on_line` 抛异常不影响身体；`on_line` 为 None 时照旧
- viewer：`/chat` 有新行立刻返回、没有时等到超时返回空、`after` 比最新大时从头来、没挂 `chat` 时 404、Host 不对拒绝；PAGE 引用 `stage.js`、`/stage.js` 能取到
- `stage.js`（node）：`nameAt` 认名字标签下面的人、取最小的框；`toFrame` 换算坐标
- `chatlog.js`（node）：`lineKind` 各类型（原 `sandbox.js` 的测试挪过来）
- 管理面板页面：`#page-live` 里有三栏和顶栏、没有 iframe；整个 `static/` 不出现原生 `confirm(` / `prompt(` / `alert(`（已有测试）；颜色只在 `:root`（已有测试）
- 已有的 console / viewer / sandbox 测试都照过

## 6. 验证（改完、合并前）

1. `view --images <录像目录>` 回放，看 viewer 自己的网页画框和以前一样（抽 `stage.js` 没改坏）
2. 起管理面板 + 沙盒，沙盒页聊天记录照旧（抽 `chatlog.js` 没改坏）
3. 起管理面板，用 `fake_claude` 或真机 dry-run 跑一次真机团子：三栏都有内容、画面不裁、手动控制能用、日志抽屉能开关；停下后内容还在。截图给用户看
4. 真机 live 跑一次等用户方便时再做（不用 worktree 里的代码去碰正在跑的团子）

## 7. 不做

- 不改 viewer 网页的样子（只抽 `stage.js`）
- 聊天记录不落盘、不能从这里发言（让团子说话用手动控制）
- 不做深色主题
- 不做画面全屏 / 放大（以后要再说）
