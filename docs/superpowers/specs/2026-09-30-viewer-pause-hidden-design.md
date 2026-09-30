# 识别可视化：看不见时暂停取画面 — 设计

日期：2026-09-30　状态：待实现

## 背景

viewer（`vision/viewer.py`）只在浏览器来拉 `/snapshot` 时才缩图、压 JPEG（实测一帧约 4.5 ms：缩到 1280 宽 2.2 ms + JPEG 1.9 ms + base64/JSON 0.35 ms，
10 帧/秒约占一个核的 4~5%）。没人拉就不压，这一点已经做到了。

漏掉的是"开着网页但没在看"：
1. 浏览器标签页切到后台、窗口最小化 —— 页面脚本照样一直长轮询 `/snapshot`，后台每秒照压 10 张
2. 管理面板（`console`）的「实时画面」页用 iframe 嵌 viewer（`live/`），iframe **加载一次后一直留着**，切到总览 / 设置等别的页签时只是 `hidden`。
   iframe 里的 `document.visibilityState` 跟的是整个浏览器标签页，不看 iframe 自己隐没隐藏 —— 所以用户在管理面板别的页签时，viewer 也一直在拉

不考虑的方案（讨论过）：浏览器直接读进程内存（浏览器沙箱做不到）、传原始像素（每秒 26 MB，比压 JPEG 还费）、NVENC 推 H.264（本机场景省不了多少、框对齐和依赖成本高）。

## 目标

页面看不见时停止拉 `/snapshot`，服务端因此不再压 JPEG；重新看得见时立刻接上最新一帧。

## 设计

### 1. viewer 页面（`PAGE` 里的脚本）

暂停有三个独立的原因，任一成立就不拉：

| 原因 | 来源 | 状态栏文字 |
|---|---|---|
| 用户点了「暂停」 | 按钮（现有） | `已暂停` |
| 标签页在后台 | `document.hidden` + `visibilitychange` 事件 | `页面在后台，暂停取画面` |
| 嵌在管理面板里、不在「实时画面」页 | 父页面 `postMessage` | `不在实时画面页，暂停取画面` |

- 纯函数 `pullWhy(userPaused, docHidden, frameShown)`：都不成立返回 `""`（该拉），否则按上表**从上到下**返回第一个成立的原因文字
- 现有的 `paused` 变量保留给 `loop()` 用，改成由 `sync()` 统一算：`paused = !!pullWhy(...)`；暂停时状态栏显示原因（灰色，同"连不上"的样式），恢复时调 `loop()`（已有 `running` 防重入）
- 「暂停 / 继续」按钮只改 `userPaused`，按钮文字只跟 `userPaused` 走（后台自动暂停时按钮不变成"继续"）
- 恢复后下一次请求带着旧的 `after=seq`，服务端只留最新一帧，所以立刻拿到最新画面，不会补发积压
- 正在等的那一次长轮询不取消：回来后 `loop()` 里已有 `if(paused)break`，最多多压一帧
- 暂停期间画面停在最后一帧，框、存图照常可用
- 父页面消息：`window.addEventListener("message", …)`，只收 `e.source === window.parent && e.origin === location.origin`、
  `e.data.type === "skydango-view"` 的，取 `e.data.shown`（布尔）作为 `frameShown`；别的消息忽略。
  直接打开 viewer（不在 iframe 里）时 `frameShown` 一直是 `true`
- **不暂停**：大脑时间线（`/brain` 长轮询，纯 JSON、没有图，停了还可能漏看轮次）、手动控制

### 2. 管理面板（`console/static/console.html`）

- 新函数 `tellLive()`：iframe 有 `src` 时向 `$("live-frame").contentWindow` 发 `{type:"skydango-view", shown: 当前页签是 live}`，targetOrigin 用 `location.origin`
- 调用时机：`showTab()` 末尾；iframe 的 `load` 事件（iframe 是在 `render()` 里、团子进入运行状态时加载的，这时用户可能不在实时画面页 —— 加载完要马上告诉它）
- iframe 加载后到收到消息之前的一小段时间 viewer 按"看得见"照拉，可以接受；**viewer 默认拉**，这样收不到消息（旧面板、直接打开）也不会一直黑着

### 3. 服务端

不改。`Viewer.snapshot()` 本来就只在被请求时压图。

## 测试

- node 跑 `pullWhy` 的真值表（三个原因的组合、优先级）
- `PAGE` 里有 `visibilitychange` 监听、`message` 监听带 `e.source===window.parent` 和 `e.origin===location.origin` 检查
- `console.html` 里 `showTab` 和 iframe `load` 都调 `tellLive`、`postMessage` 带 `location.origin`；两份脚本 `node --check` 照旧通过
- 浏览器核对（离线，用 `view --images`）：模拟 `document.hidden`、`postMessage` 后网络请求里不再出现新的 `snapshot`，恢复后接上

## 真机验证（晚上）

1. `run --view` 开网页，切到别的标签页 10 秒再切回：画面马上接上；团子日志 / CPU 无异常
2. 管理面板启动团子，停在「总览」：浏览器开发者工具里 `live/snapshot` 请求停着；切到「实时画面」立刻开始拉
