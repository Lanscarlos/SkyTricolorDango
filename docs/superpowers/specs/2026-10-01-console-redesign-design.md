# 管理面板改版（左栏 + 三栏工作台 + 和果子纸感）

2026-10-01。改的是 `python -m skydango console` 的网页（`src/skydango/console/static/console.html`，1150 行单文件）和少量后端。
设计时的线框和效果图在 `.superpowers/brainstorm/`（不进 git）：`structure.html`（选了 A）、`visual-style.html`（选了 2）、`sandbox-page.html`（沙盒页效果图）。

## 1. 为什么改、给谁用

用户（卡洛）两台电脑：单位电脑没有模拟器、主要跑**大脑沙盒**调团子；个人电脑跑真机团子。面板的用途全都有：
在沙盒里冒充好友聊天、盯大脑时间线调提示词、录剧本回放做回归、在内心页修剪性格档案和看关系卡、填密钥改设置排错、真机跑着时挂着看状态。**沙盒是主场**，宽屏（≥ 1400px）。

今天实际用下来的毛病：
- 沙盒启动失败只显示「出错停下了（日志在「总览」）」，要切页翻日志才知道是缺 DeepSeek Key；而且这本该在启动前就查出来（大脑模式的预检漏查了大模型 Key）
- 「重置记忆」等用浏览器原生 `confirm()`，在 Claude 桌面版的内嵌浏览器里被直接取消，点了没反应
- 沙盒页从上往下堆：聊天、身边、场景、剧本、大脑时间线，边聊边看大脑要来回滚
- 「叫醒团子」在总览、沙盒在另一页，两者共用一个子进程槽，界面上看不出来
- 剧本回放的报告只给一个文件路径，得自己去翻文件

成功的样子：宽屏一屏内同时看到聊天、大脑、心里；谁在跑、为什么停了在左栏一眼看到；启动前就知道缺什么、点一下跳到那个设置；确认框在内嵌浏览器里能用；报告在页面里读。

## 2. 整体框架（骨架 A）

- 去掉顶部页签，改成**左栏（约 200px）+ 右侧工作区**。路由仍用 `location.hash`（`#sandbox` `#live` `#inner` `#scenarios` `#settings` `#device`），刷新停在原页；**默认进 `#sandbox`**。
  旧的 `#overview` 映射到 `#live`（书签不坏）
- 左栏从上往下：
  1. **标志**：三颗团子 + 「三彩团子」，团子颜色随运行状态变（沿用现在 `body[data-state]` 的做法：starting 呼吸、stopping 灰、crashed 红）
  2. **「正在跑」卡片**（团子 / 沙盒共用一个子进程槽，所以只有一张）：
     - 停着：「都没在跑」
     - 启动中 / 运行中：哪一个在跑。沙盒：沙盒时间 + 心情 + 精力；真机：真发 / 只看 + 身边几个人。按钮「停止」（沙盒叫「下线（写日记）」）
     - 停止中：「在收尾…」
     - 出错停下：砖红边框，直接写出错原因（`runner.status().error`，见 §6），「看日志」展开最近 20 行
     - 点卡片空白处跳到在跑的那一页
  3. **导航**，三组：运行（沙盒、真机团子）、复盘（内心、剧本和报告）、配置（设置、设备）。
     导航项右侧可带小标记：真机团子「没在跑 / 在跑」，剧本和报告「回放中」或报告数，设置有错时「!」
- **启动入口留在各自页面**（沙盒页选起始时间；真机页选模式 / 真发 / 动作 / 时长），左栏卡片只看状态和停止。一个在跑时，另一页的启动按钮置灰并写原因（「沙盒在跑，先下线」）
- **全局**：页面内对话框代替所有 `confirm()` / `prompt()` / `alert()`（「另存为剧本」现在用两次 `prompt()` 问名字和说明，同样在内嵌浏览器里失效）：
  `common.js` 的 `ask(text, {ok, danger, fields})` 返回 Promise——没有 `fields` 时确定 = `true`、取消 = `false`；有 `fields`（`[{name, label, placeholder, value}]`）时确定 = `{name: 值}`、取消 = `null`；操作结果用右下角提示条（`toast(text, kind)`，4 秒消失，出错的不自动消失）
- 宽度 < 1000px：左栏收成顶部横栏（标志 + 卡片缩成一行 + 导航横排可滚），三栏变上下排。只保证能用，不专门优化

## 3. 视觉（和果子纸感，只做浅色）

- 颜色全部是 `console.css` 里 `:root` 的变量，以后加深色只改变量：
  纸底 `--paper #f6f1e7`、左栏 `--side #efe7d8`、卡片 `--card #fffdf8`、线 `--line #e6dccb`、墨 `--ink #3b332b`、次要 `--muted #8f816c`；
  樱花 `--sakura #e58aab` / `--sakura-bg #f9dce7` / `--sakura-ink #7b2f4c`（团子说的话、选中项）；
  抹茶 `--matcha #7fae7d` / `--matcha-bg #e3efdc`（运行中、成功、启动按钮）；琥珀 `--amber #d39a3a`（警告）；砖红 `--brick #c4573f`（出错、停止、被拦）
- 卡片：`--card` 底 + 1px `--line` 边 + 12px 圆角 + 很淡的阴影；小标题 12px、字距 .12em、`--muted`
- 字体只用系统字体：`"Microsoft YaHei UI","Microsoft YaHei","PingFang SC",system-ui`；等宽 `"Cascadia Mono",Consolas`。**不加载外部字体**（面板要能离线，国内也连不上 Google Fonts）。数字 `tabular-nums`
- 文字对比度：正文 `--ink` 对 `--paper` / `--card` 都 ≥ 7:1；`--muted` 只用于小标题和辅助说明
- 这次**不做深色模式**

## 4. 页面

### 4.1 沙盒（`#sandbox`，主场）

效果图见 `sandbox-page.html`。
- **顶栏**
  - 运行中：沙盒时间牌（大号「11:28」+ 小字「10月1日 周四」）、快进（+10 分钟 / +1 小时 / 到明早 9 点 / 拨到 [输入] 设成）；右侧「录制中 · N 步」「另存为剧本…」「重置记忆」（运行中置灰）
  - 停着：启动区（接着上次 / 睡一晚到第二天 9 点 / 自定义 [HH:MM 或 YYYY-MM-DD HH:MM]）+「启动沙盒」+「重置记忆」；**预检问题列在启动按钮旁**，带 `setting` 的显示「去设置 →」，点了跳 `#settings` 对应项（§4.5）；有问题时启动按钮置灰
  - 回放中：顶栏下多一条「回放『<剧本>』第 i/n 步 ·【停止回放】」，冒充和世界操作全部置灰（沿用现在 `SB.replay` 的判断）
- **三栏**（`grid-template-columns: 1.15fr 1.1fr .85fr`，页面本身不滚、高度占满视口，各栏内部滚；新内容来时如果原本就在底部则自动跟到底）
  - **左：聊天记录 + 冒充**。团子说的靠右、樱花底；别人说的靠左、米色底；事件（来了 / 走了 / 快进 / 上线下线）是居中分隔线；反思是虚线框（"── 反思：… ──"）；动作旁白斜体靠右；被拦的话删除线 + 砖红小字原因。
    底部固定输入区：说话人下拉（好友 + 「陌生人…」→ 出现名字输入框）、输入框（回车发送）、发送。说话人选择记在 `localStorage`（try/catch 包住）
  - **中：大脑时间线**。标题行「sonnet / low · 已醒 N 轮 · 在线」+「只看做了事的」（默认勾）。渲染沿用 `vision/static/brain_trace.js`；面板里的纸感样式写在 `console.css`，用 `.console` 作用域覆盖 `brain_trace.css` 的颜色，不改 viewer
  - **右：现在 + 世界**。现在：心情（大字 + 一句原因）、精力（档位 + 条）、别扭、收着点、心愿，右上「完整内心 →」跳 `#inner` 并切到沙盒。
    「身边」：好友是带 × 的标签（× = 走了）、「谁来了」输入 + 来了、陌生人 −/数字/+、地名 + 改；「团子看到的」：场景多行输入 + 改场景、新鲜事输入 + 放
- 剧本列表、回放、报告**挪到「剧本和报告」页**；沙盒页只留录制状态和「另存为剧本…」（录制随沙盒启动自动开始）
- 停着时聊天记录保留上一次的，大脑栏显示「（沙盒跑起来之后这里是大脑时间线）」
- 接口全部沿用现有的 `/api/sandbox/*` 和 `/sandbox/*` 转发，不改

### 4.2 真机团子（`#live`，原「总览」+「实时画面」）

- 停着：启动区——模式（统管大脑 / 普通 Agent 调试用）、真的发送、聊天时做动作（config.toml 关了动作时只能关不能开，沿用 `emotes_allowed`）、运行时长、「叫醒团子」；预检问题同沙盒页（带「去设置 →」）
- 跑着：左约 60% 实时画面（iframe `live/`，16:9），右一列状态卡片：身边、心情 / 精力、正在做、刚说过、场合（沿用现有状态数据）；底部日志尾巴（可折叠，默认展开最近 30 行）
- 上次留下的团子 / 沙盒占着端口：页面顶部横幅 + 「让它退出」（沿用 `orphan` 和 `/api/orphan/stop`）
- 沙盒在跑时：启动区置灰，写「沙盒在跑，先下线」

### 4.3 内心（`#inner`）

- 内容和数据不变（`/api/inner?source=dango|sandbox`），只改排版：左列「现在」「精力曲线 + 心情色带」「反思记录」；右列「性格档案」「关系卡」「日子和日记」
- 标题旁两段式开关「团子 / 沙盒」（代替现在的两个按钮）
- 性格条目的「删」用页面内确认框
- 刷新逻辑不变（团子醒着每 5 秒，其它时候打开时读一次 + 「刷新」）

### 4.4 剧本和报告（`#scenarios`）

- 左：剧本列表（`/api/sandbox/scenarios`：名字、备注、步数），每条「回放」→ 页面内确认（"沙盒在跑会先下线（写日记）……每一步都真的调 Claude、花额度"，现有文案）。
  回放中显示进度（第 i/n 步、当前步说明）+「停止回放」；示例剧本的说明（`docs/sandbox-scenarios/` 换好友名再复制进 `sandbox/scenarios/`）在列表为空时显示
- 右：报告列表（§6 新接口，新的在前：剧本名、时间），点开在右侧阅读区渲染
- **markdown 渲染**：`scenarios.js` 里自写极简渲染器，只支持 `#`~`###` 标题、`-` / `1.` 列表、`>` 引用、``` 代码块、行内 `code`、`**粗体**`、空行分段；一律先转义 HTML 再套标签（报告里有聊天原话）
- 不做两份报告对比（以后再说）

### 4.5 设置（`#settings`）

- 设置项和分组不变（`settings.FIELDS`、`/api/settings`），只改样式
- 页面左侧加分组目录（粘性），点了跳到分组
- 每个设置项的容器 `id="set-<key 里的点换成横线>"`（如 `set-secret-llm`）；从「去设置 →」进来时 `#settings/secret.llm` → 滚到该项、高亮闪 2 秒

### 4.6 设备（`#device`）

只换样式，行为不变。

## 5. 文件拆分（方案甲：多个静态文件、不加构建）

| 文件 | 内容 |
|---|---|
| `console/static/console.html` | 骨架：左栏、六个 `<section>` 容器、确认框和提示条容器；按顺序引入 css / js |
| `console/static/console.css` | 主题变量、左栏、通用组件（按钮、输入、卡片、标签、分隔线、确认框、提示条）、各页样式、`brain_trace` 的纸感覆盖、窄屏 |
| `console/static/common.js` | `$` / `el`、`api(path, body?)`（POST 带 `X-Skydango: 1` + JSON）、`ask()`、`toast()`、路由（hash → 页面，`onShow` / `onHide` 钩子）、左栏卡片和导航标记的轮询、`problemList(el, problems)`（渲染预检问题 + 去设置链接） |
| `console/static/sandbox.js` `live.js` `inner.js` `scenarios.js` `settings.js` `device.js` | 各页逻辑，每个文件注册 `Pages.<名字> = {init, show, hide}`；页面之间只经 `common.js` 通信 |

- 页面里的请求仍用相对路径（面板放在 `/` 下）
- `brain_trace.js` / `brain_trace.css` 仍从 `/static/` 取（和 viewer 共用）
- 现有 JS 的逻辑（轮询、长轮询 `/sandbox/state`、内心曲线 SVG、设置保存 / 测试按钮、设备检测）**搬家 + 改 DOM**，不重写算法

## 6. 后端改动

1. **预检补查大模型 Key**（`console/preflight.py`）：`opts.brain` 时也检查 `store.secret("llm")`（`cfg.llm.provider == "echo"` 时不查）。
   原因：`cli._run_brain` 无条件 `make_llm(cfg.llm)` 做大脑离线时的备用回复，没有 Key 直接退出。问题文案：「大脑离线时的备用回复要大模型的 API Key：去设置页填」
2. **预检问题带跳转目标**：`preflight()` 返回 `list[dict]`，每项 `{"text": str, "setting": str | None}`；`setting` 是 `settings.FIELDS` 的 key（`secret.llm`、`secret.claude`），没有对应项的为 `None`。
   `ConsoleServer` 里拼的问题（端口被占、正在检测设备、secrets.toml 坏了、沙盒在跑……）同样改成 dict。`/api/state` 的 `problems` 和启动接口的拒绝回复跟着变
3. **沙盒预检提前给**：`/api/sandbox/info` 加 `problems`：`launch_sandbox` 起进程前做的那些检查（团子占着槽、正在回放、`preflight`、沙盒端口上有孤儿），抽成一个不起进程的 `sandbox_problems()` 两处共用；沙盒自己正在跑时为空列表（页面这时显示运行中的界面）
4. **出错原因**：`Runner.status()` 加 `error: str | None`——状态是 crashed 时，从日志环形缓冲里倒着找第一行以「错误:」开头的（去掉前缀 `错误: `），找不到就取最后一行非空日志（截 200 字）；其它状态为 `None`
5. **报告接口**（只读，校验 Host，同其它 `/api/*`）：
   - `GET /api/sandbox/reports` → `{"reports": [{"name": "放鸽子-20261001-1130.md", "scenario": "放鸽子", "time": <mtime 秒>, "size": <字节>}]}`，按 mtime 倒序，只列 `<sandbox dir>/reports/*.md`；目录不存在返回空列表
   - `GET /api/sandbox/reports/<name>` → `{"name", "text"}`；`name` 必须恰好是上面列表里的一个文件名（不含 `/` `\` `..`，以 `.md` 结尾，`resolve()` 后父目录就是 reports 目录），否则 404；单文件上限 2 MB，超了 413
   - `scenario` 从文件名解析：`replay.py` 写的是 `<剧本>-%Y%m%d-%H%M%S[-<序号>].md`（剧本名里可能有 `-`），用 `^(.*)-(\d{8}-\d{6})(?:-\d+)?\.md$` 取第 1 组；对不上的文件 `scenario` 为文件名去掉 `.md`
6. **面板静态文件**：`GET /console/static/<文件>`，只提供 `console/static/` 下扩展名为 `.css` / `.js` 的文件，文件名不得含 `/` `\` `..`；不校验 Host（同 `/static/`，内容和页面一样公开）。`pyproject.toml` package-data 加 `console/static/*.css`、`console/static/*.js`
7. **不改**：子进程、沙盒、viewer 的接口和行为；现有 `/api/*` 地址和参数（只加字段）；设置的读写；安全检查（Host、`post_guard`）

## 7. 测试

- **pytest**（假设备、不起真子进程，沿用 `tests/test_console_*.py` 的写法）：
  - 预检：大脑模式缺大模型 Key → 有问题且 `setting == "secret.llm"`；`provider = "echo"` 不报；缺 Claude 令牌 → `setting == "secret.claude"`；问题都是 dict
  - `/api/state`、`/api/sandbox/info` 的 `problems` 是 dict 列表；沙盒 info 带 `problems`
  - `Runner.status()["error"]`：crashed 时取到「错误: 没有找到 API Key…」那行；没有「错误:」行时取最后一行；运行中为 `None`
  - 报告：列出（倒序、只要 .md、目录不存在为空）、读取；`..`、斜杠、不存在的名字、非 .md → 404；超大 → 413；Host 不对 → 403
  - 静态：`/console/static/console.css` 200 且类型对；`.html`、`..`、不存在 → 404
  - 现有 `test_console_*` 全部通过（改了 `problems` 格式的断言跟着改）
- **页面**（内嵌浏览器，`fake_claude` 跑沙盒、不花额度；做法见 `tests/test_cli_sandbox.py`）：
  1. 缺 Key 时沙盒页预检列出问题、启动置灰；点「去设置 →」跳到并高亮 API Key 那项
  2. 启动沙盒 → 左栏卡片变「沙盒 运行中」带沙盒时间；冒充发言、来了 / 走了、快进；三栏同屏、各自滚动；大脑栏有轮次
  3. 「重置记忆」「下线」「删性格条目」的页面内确认框能用（取消不做、确定才做）
  4. 子进程出错停下 → 卡片上直接显示原因
  5. 剧本和报告页：列表、打开一份报告（造一个测试报告文件）
  6. 宽 1500px 和 900px 各截一张图
  - 真机团子页这台电脑没有模拟器：只核对停着的样子、预检；跑起来的样子等用户在个人电脑上看（写进验收清单）

## 8. 不做

深色模式；手机 / 平板专门适配；两份报告对比；改 viewer（`vision/viewer.py` 的网页）的样子；改任何子进程的接口。
