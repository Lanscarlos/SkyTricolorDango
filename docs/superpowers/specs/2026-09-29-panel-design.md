# 管理面板 — 设计

日期：2026-09-29　状态：**设计已和用户确认，待写实现计划**

把现在只能看的识别可视化网页（`vision/viewer.py`，设计见 `2026-09-28-viewer-design.md` 等）扩成一个**管理面板**：
先启动面板，在网页里填密钥、改常用配置、检测 MuMu 连接，再一键启动 / 停止团子；团子运行时在同一个页面里看实时识别、大脑思考和团子状态。

## 目标

**不用终端也能把团子跑起来、看清楚它在干什么。** 省掉 `setx` 设令牌、手改 `config.toml`、敲 `run --brain --live --view` 这些步骤。

成功的样子：`python -m skydango panel` → 浏览器打开面板 → 设置页填好 DeepSeek Key / Claude 令牌 → 设备页点「开始检测」全绿 →
总览页选「大脑、只看」点「启动」→ 状态变成「运行中」，卡片上显示身边有谁，实时页能看到画面和大脑时间线 → 点「停止」，
团子复原镜头、换回轮盘后退出，状态变成「已退出」。

## 需求（已和用户确认）

1. **第一版先给用户自己用，架构上给别人用留口子**：假设 Python 和依赖已经装好；不做打包、不做首次启动向导。
   「留口子」指：密钥存本机文件、不要求 `setx`；设置项有中文说明和校验；出错在面板上看得到，不用翻终端。
2. 面板能**启动 / 停止**团子（普通 Agent 或统管大脑），同一时间只有一个团子。
3. 面板能**改一份挑选过的常用配置**（清单见 §4），**不改用户手写的 `config.toml`**。
4. 面板能**填 LLM 的 Key 和 Claude 令牌**，并提供「测试」按钮。
5. 面板能**检测 MuMu 连接**（adb、设备在线、截图、输入法、游戏在前台、模拟键盘）。
6. 团子运行时能看**实时识别画面、大脑时间线、手动控制**（复用现有 viewer）和**团子状态**（身边好友、陌生人、互动请求、牵手、正在做什么、最近说的话）。

## 方案选择

**选定：面板是父进程，团子是子进程（`run --view`），实时画面由面板转发。**

- 退出收尾（复原镜头、换回轮盘、大脑写经过、结束 Claude Code 进程树）全挂在 `_run_agent` / `_run_brain` 的 `KeyboardInterrupt` / `finally` 上，
  子进程方式原样复用；团子崩了面板还在；面板保证只有一个子进程，避免 CLAUDE.md 里记的「两个 Agent 同时在线」。
- 密钥经子进程的环境变量传入，读 Key 的代码（`chat/llm.py` 的 `read_key`、`brain/claude.py`）不用改。

放弃的方案：
- **同一进程、团子跑在线程里**：`Agent.run` 没有外部叫停的口子，要改主循环；OCR / Claude Code / MCP 服务在同一进程里反复启停容易残留线程和端口；团子崩了面板一起没。
- **面板只负责启动，实时画面在新标签页开原来的 viewer**：两个页面、两个端口，团子停了实时页就断，不像一个面板。

## 1. 架构和模块

```
浏览器 ──► 面板进程（python -m skydango panel，127.0.0.1:8760）
             ├─ /                一个页面，4 个页签（static/panel.html）
             ├─ /api/settings    读写 panel.toml / secrets.toml、测试按钮
             ├─ /api/device      设备检测（团子没运行时才能用）
             ├─ /api/run         启动 / 停止 / 状态 / 日志尾巴
             └─ /live/*  ──转发──► 子进程 run --view（127.0.0.1:8761）
                                     / /snapshot /brain /control /control/options
                                     /status /shutdown（新增）
```

新增包 `src/skydango/panel/`，每个文件只管一件事：

| 文件 | 内容 |
|---|---|
| `server.py` | HTTP 服务和路由、转发 `/live/*` |
| `runner.py` | 管子进程：拼命令行、起、停、状态机、日志环形缓冲、找运行目录 |
| `settings.py` | 设置清单（每项的键、中文名、说明、类型、校验、分组）、读写 `panel.toml` / `secrets.toml`、来源标记、密钥打码、小 TOML 写出器 |
| `devicecheck.py` | 设备检测的各项检查，复用 `cmd_devices` 的逻辑 |
| `static/panel.html` | 面板页面（HTML + CSS + 原生 JS，无构建步骤） |

改动的现有文件：

| 文件 | 改动 |
|---|---|
| `config.py` | 新增 `PanelConfig`（§4）；`load_config` 在 `config.toml` 后叠加 `panel.toml`，记下被覆盖的键；读 `secrets.toml` 写进 `os.environ` |
| `vision/viewer.py` | 本机 Host / `X-Skydango` 头校验抽成公共函数（面板共用）；新增 `/shutdown`、`/status`；页面里 4 处 `fetch("/…")` 改成相对路径 |
| `cli.py` | 新增 `panel` 子命令；`run` 新增参数 `--viewer-port`、`--no-browser`、`--parent-pid`、`--no-brain`、`--dry-run` |
| `.gitignore` | 加 `panel.toml`、`secrets.toml` |

只用标准库，不加依赖。

## 2. 启停和生命周期

### 启动

- 命令行：`python -m skydango [--config …] run [--brain] [--live] [--no-emotes] [--duration N] --view --viewer-port <child_port> --no-browser --parent-pid <面板 pid>`
  （`sys.executable`，工作目录固定为仓库目录——`.pydeps`、缓存目录依赖这一点）。
- **启动选项说了算**：模式和真发 / 只看总是显式传（`--brain` 或 `--no-brain`，`--live` 或 `--dry-run`），`config.toml` 里的 `brain.enabled`、`reply.dry_run` 不会偷偷改掉面板上的选择。
  「做动作」只能关不能开：`emotes.enabled = false` 时总览上这一项置灰并注明「config.toml 里关掉了」。
- 子进程环境变量 = 面板的环境变量 + `secrets.toml` 的 `[env]` + `PYTHONIOENCODING=utf-8`。
- Windows 上加 `CREATE_NEW_PROCESS_GROUP`：面板终端按 Ctrl+C 时由面板决定先停谁。
- stdout / stderr 由后台线程读进环形缓冲（`panel.log_lines` 行）；从「本次运行的日志和截图: <路径>」这一行取运行目录。
- 由面板启动时，子进程的 viewer 强制 `host = 127.0.0.1`、端口 `child_port`、不开浏览器。

### 状态机

**未启动 → 启动中 → 运行中 → 正在退出 → 已退出（退出码）/ 崩溃**

- 子进程的 `child_port` 有响应（`GET /status` 200）才算「运行中」；60 秒还没响应仍显示「启动中」，日志照样滚动，方便看卡在哪。
- 子进程自己退出：退出码 0 →「已退出」，非 0 →「崩溃」；日志和运行目录保留到下一次启动。
- 「启动」只在「未启动 / 已退出 / 崩溃」时可用。

### 停止

1. 面板 `POST /shutdown` 给子进程；子进程调 `_thread.interrupt_main()`（可注入，测试用），走和 Ctrl+C 完全一样的收尾。
2. 页面显示「正在退出（复原镜头、换回轮盘…）」，最多等 `panel.stop_timeout` 秒（默认 60，live 大脑退出前要写记忆）。
3. 超时（或 `/shutdown` 发不过去）→ 按进程树强杀（Windows `taskkill /F /T /PID`，其他系统 `killpg`），页面醒目提示
   「强制结束了，轮盘可能没换回，请用 `python -m skydango emotes wheel` 检查」。

### 防残留

- 子进程 `--parent-pid` 看门狗：每 2 秒看一次面板进程还在不在，不在就 `interrupt_main()` 走正常退出。
- 面板启动时 `child_port` 已经有人响应（上次留下的）：页面提示并给「让它退出」按钮（发 `/shutdown`）。
- 面板自己退出（Ctrl+C）：先按「停止」流程停子进程，再退出。

### 实时画面转发

- `GET /live/`、`/live/snapshot`、`/live/brain`、`/live/control/options`、`/live/status` 原样转发到子进程（带 query），超时 5 秒（长轮询 2 秒 + 余量）。
- `POST /live/control`：面板先做自己的本机 Host + `X-Skydango` + JSON + 大小校验，再以 `Host: 127.0.0.1:<child_port>`、`X-Skydango: 1` 转发。
- 子进程不在「运行中」或连不上：返回 503 + `{"ok": false, "text": "团子没在运行"}`。

## 3. 子进程新增的接口（`vision/viewer.py`）

- `GET /status`：只返回状态 JSON，不带图片，给总览页每 2 秒拉一次。内容是最近一帧的 `info`（身边的好友、陌生人、互动请求……）
  加上 `run --brain` 时的牵手、正在做、最近说的话（取自 `update()` 的 `info`，大脑模式由身体填，普通模式由 Agent 填已有的「刚说过」）。
- `POST /shutdown`：本机 Host + `X-Skydango` 头校验，通过就调中断函数，返回 `{"ok": true}`。
- 这两个接口单独跑 `view` 时也在（`view` 收到 `/shutdown` 同样按 Ctrl+C 处理），不单独开关。

## 4. 设置和密钥

### 三个文件

| 文件 | 谁写 | 内容 |
|---|---|---|
| `config.toml` | 用户手写 | 面板只读不写 |
| `panel.toml` | 面板 | 和 `config.toml` 同结构，只含下面清单里的项，外加 `[panel]` |
| `secrets.toml` | 面板 | `[env]` 表，键是环境变量名（`DEEPSEEK_API_KEY = "sk-…"`、`SKYDANGO_CLAUDE_TOKEN = "…"`） |

两个新文件与 `config.toml` 同目录，都进 gitignore。页面注明「明文保存在本机 secrets.toml，别发给别人」。

### 加载顺序

默认值 → `config.toml` → `panel.toml`。**终端直接敲 `run` 等命令也读 `panel.toml` 和 `secrets.toml`**，面板里改的在哪儿启动都生效。
- 启动时打一行 INFO：「panel.toml 覆盖了 3 项：device.serial、llm.model、brain.enabled」。
- `secrets.toml` 的值**覆盖**已有的环境变量（否则面板里换了 Key 还在用旧的 `setx`），打一行 INFO「用 secrets.toml 里的 DEEPSEEK_API_KEY」（不打值）。

### 设置清单

**启动选项**（存在 `[panel]`，下次打开面板还是上次的选择；不影响终端 `run`）：模式（普通 Agent / 统管大脑）、真发 / 只看、聊天时做动作、运行时长（空 = 一直跑）

**连接**：`device.adb_path`、`device.serial`、`device.capture`（auto / mumu / adb）

**普通模式的大模型**：`llm.provider`（openai / anthropic / echo）、`llm.base_url`、`llm.model`、API Key（存到 `secrets[env][<llm.api_key_env>]`）

**大脑**：Claude 令牌（存到 `secrets[env][<brain.token_env>]`）、`brain.claude_path`、`brain.model`、`brain.eyes_model`

**功能开关**：`env.enabled`、`perception.enabled` + `perception.model`、`places.enabled`、`friend_check.enabled`

**身份**：`reply.disclosure_prefix`、主人昵称（一项，保存时同时写 `reply.owner_name` 和 `brain.owner_name`；两者在 config.toml 里不一致时显示两个值并提示）

不放：人设、好友（`memory/`，以后可以做「记忆」页签）、坐标和阈值（继续手改 toml）。

### `[panel]` 配置段（`PanelConfig`）

```toml
[panel]
port = 8760          # 面板端口；和 view 的 8765 错开，可以同时开
child_port = 8761    # 团子子进程的 viewer 端口
stop_timeout = 60.0  # 停止时最多等几秒，超了强杀
log_lines = 500      # 日志尾巴保留几行
# 上次的启动选项，面板写
brain = false
live = false
emotes = true
duration = 0.0
```

### 页面上的密钥

- 浏览器拿不到完整密钥：`GET /api/settings` 只返回「已设置（sk-…a1b2）」/「没设置」；输入框留空 = 不改；「清除」按钮删掉这一项。
- 每一项标来源：默认 / config.toml / 面板；「用回 config.toml 的值」= 从 `panel.toml` 删掉这一项。

### 校验和保存

- 类型：开关、整数、小数、文字、下拉、路径。类型不对 / 下拉值不在范围内 → 拒绝保存，返回哪一项、为什么。
- 路径不存在（adb、YOLO 模型、claude）→ 警告，不拦。
- 写入：先写同目录临时文件再 `os.replace`。标准库不能写 TOML，自己写一个只支持字符串（正确转义）、布尔、整数、小数、字符串列表、嵌套表的小写出器；
  测试要求写出去再用 `tomllib` 读回完全一致。
- 团子运行时保存：照常保存，提示「重启团子后生效」。

### 测试按钮（点了才测，会花一点额度）

- **测试大模型**：用页面上当前（可能还没保存的）值临时建一个 `make_llm`，发一句很短的请求，`max_retries = 0`、超时 20 秒，显示成功与否、耗时、回复前 50 字；`provider = echo` 时直接说「echo 不调模型」。
- **测试 Claude 令牌**：先检查 `claude_path` 找得到（`claude --version`），再用大脑的隔离配置目录和令牌跑一次 `claude -p --model haiku "只回复 ok"`，超时 60 秒；复用 `brain/claude.py` 起进程和清理环境变量的逻辑。

## 5. 页面

一个页面，顶部常驻状态栏：`● 运行中 · 大脑 · live · 已运行 12 分钟 ［停止］`。4 个页签：

### ① 总览（默认）

- 启动选项 + 大号启动 / 停止按钮；选了「真发」时旁边写「会在游戏里真的说话」。
- 启动前预检（§7）没过：按钮下直接显示原因和「去设置页」链接。
- 运行中的状态卡片（`/live/status`，每 2 秒）：身边好友（几个、都有谁）、陌生人、互动请求、牵着手没有、正在做什么、最近说的一句话。
- 日志尾巴：自动滚动（往上翻时暂停滚动），WARNING 黄、ERROR 红；运行目录路径。

### ② 实时

- iframe 嵌 `/live/`：面板转发的就是子进程自己的 viewer 页面。viewer 页面的 fetch 改成相对路径后，放在 `/`（单独 `view`）和 `/live/`（面板）都能用；
  画面、识别框、大脑时间线、手动控制都不重写。
- 团子没运行：显示「团子没在运行」和「去总览」按钮；团子启动后自动加载 iframe。

### ③ 设置

- 分组：启动选项之外的 §4 清单（连接、普通模式的大模型、大脑、功能开关、身份）。
- 每项：中文名、一句说明、来源标记；改过没保存的高亮；底部「保存」「放弃修改」；测试按钮放在各自的分组里。

### ④ 设备

「开始检测」逐项显示 ✓ / ✗ / ⚠ / 跳过，失败项附「怎么办」：

1. adb 路径存在且能运行（显示 `adb version` 第一行）
2. 设备在线：`serial` 不在 `adb devices` 的在线列表里 → ✗，列出实际看到的设备，点一下填进设置（未保存状态）
3. 截图：尺寸（不是 1920×1080 → ⚠）、截图方式（MuMu 原生 / adb screencap）、耗时 ms、缩略图
4. 当前输入法是 `device.ime_id`（否则 ⚠：输中文要 ADBKeyboard）
5. 游戏在前台：前台窗口包名含 `com.netease.sky`（否则 ⚠）
6. 模拟键盘设备找得到（否则 ⚠）

前面的项失败，依赖它的后续项标「跳过」。每一步有超时（`device.adb_timeout`）。团子运行时整页置灰：「团子运行中，设备归它用」，`/api/device` 返回 409。

### 视觉

沿用现有 viewer 的深色风格（CSS 变量同一套），实时页嵌进来不突兀；实现时用 `frontend-design` skill 过一遍排版。

## 6. 安全

面板能改密钥、能让团子在游戏里说话，比现有 viewer 更严：

- 面板**只监听 127.0.0.1**，第一版不给改。
- 所有 `/api/*`、`/live/*`（含 GET）校验 Host 是本机地址 + 面板端口（防 DNS 重绑定）；`/` 和静态页不校验。
- 所有 POST 还要：`X-Skydango: 1` 头、`Content-Type: application/json`、请求体 ≤ 64 KB（设置可能比手动控制大）。
- 子进程由面板启动时 viewer 强制本机；`/shutdown`、`/status` 同样校验。
- 日志里不打密钥的值。

## 7. 出错处理

- **启动前预检**（`POST /api/run/start` 先做，不过就不起子进程，返回原因）：
  - `load_config` 能加载（能提前发现「未知配置项」等）
  - 大脑模式：`brain.token_env` 对应的密钥有值（secrets 或环境变量）
  - 普通模式、`llm.provider != "echo"`：`llm.api_key_env` 对应的 Key 有值
  - 没有已在运行的子进程
- 子进程一启动就退出：「崩溃（退出码 N）」，日志尾巴里有报错。
- 转发失败：503 + 中文原因，页面显示，不卡住。
- 设备检测每步超时，一步卡住不拖住整页。
- 强杀：见 §2。

## 8. 测试

`python -m pytest -q`，不需要模拟器：

- `test_panel_settings.py`：TOML 写出器往返、加载顺序（默认 → config → panel）、被覆盖键的记录、来源标记、密钥打码 / 留空不改 / 清除、类型校验、主人昵称写两处、原子写入
- `test_panel_runner.py`：假子进程脚本（打日志、监听端口、响应 `/shutdown`、可选择不理 shutdown / 一启动就崩）测状态机各转换、超时强杀、日志环形缓冲、运行目录解析、只允许一个实例、命令行拼接、环境变量注入
- `test_panel_server.py`：路由、Host / 头 / Content-Type / 大小校验（403 / 413）、用假上游 HTTP 服务测转发（GET 带 query、POST 头改写）、没运行时 503、设备检测运行中 409
- `test_panel_devicecheck.py`：假 adb 输出，逐项 ✓ / ✗ / ⚠ / 跳过
- `test_viewer.py` 补：`/shutdown` 调注入的中断函数、`/status`、页面里没有以 `/` 开头的 fetch 路径
- `test_config.py` 补：`panel.toml` 叠加、`secrets.toml` 写进环境变量并覆盖旧值、`[panel]` 默认值
- 父进程看门狗：判断逻辑（给一个假的「进程在不在」函数）单测

**真机验证**（Windows + MuMu，用户跑；单元测试通过不等于游戏里能用）：
1. `panel` → 设备页检测全绿（或按提示修到全绿）
2. 设置页填 Key / 令牌，两个测试按钮都成功
3. 总览选「普通、只看」启动 → 运行中，状态卡片有数据，实时页有画面；停止 → 已退出
4. 大脑 + 只看：实时页有大脑时间线，手动控制能用；停止后用 `emotes wheel` 核对轮盘复原
5. 团子运行中直接关掉面板终端 → 几秒内子进程自己退出（`Get-CimInstance Win32_Process` 确认没有残留 python）
6. 让子进程不理 `/shutdown`（比如卡在 adb）时的强杀路径和提示（能复现再测）

## 9. 第一版不做

打包成 exe、首次启动向导、在面板里编辑人设 / 好友 / 记忆、局域网访问、同时跑多个团子、系统凭据管理器存密钥、翻看以前的运行记录、面板里改坐标和阈值。
这些都能在这个结构上以后加。
