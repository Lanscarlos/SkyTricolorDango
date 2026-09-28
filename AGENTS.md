# AGENTS.md — 给接手这个仓库的 Agent

SkyTricolorDango：在 MuMu 模拟器上"自己玩光遇"的 Agent。纯视觉：只截图 + 模拟输入，不读内存、不注入、不抓包。
当前能力：读聊天记录 → 大模型回复 → 在游戏里发言（可以带表情动作）；快捷动作轮盘的读取 / 编辑 / 做动作。

**先读 [docs/game-ops.md](docs/game-ops.md)**：光遇 + MuMu 的操作方式、界面坐标、实测踩过的坑都在里面。
不看它直接往游戏里发输入，很容易打错字、卡在界面里，或者覆盖掉用户的轮盘配置。

## Agent 规则

- 用中文回答。
- 每次任务改完，主动把功能分支合并进 main 并推送到远程。

## 环境

- Windows + MuMu 12，游戏包名 `com.netease.sky.vivo`（国服 vivo 渠道），横屏 1920×1080
- adb 用 MuMu 自带的：`D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe`，设备名 `emulator-5554`
  （`127.0.0.1:16384` 在这台机器上是 offline）
- Python 3.13；OCR 用 `rapidocr` 3.x（`rapidocr_onnxruntime` 不支持 3.13）
- 包装在用户目录，`skydango` 不在 PATH 上，用 `python -m skydango ...`
- `config.toml` 是本机配置（gitignore），模板是 `config.example.toml`
- 模拟器里装了 ADBKeyboard 并设为当前输入法（输中文用）；用户自己要打字时 `python -m skydango ime off`
- 在 Git Bash 里调 `adb shell` 带 `/dev/...` 路径时要 `export MSYS_NO_PATHCONV=1`，否则路径会被改写

## 代码结构

| 位置 | 内容 |
|---|---|
| `src/skydango/device/adb.py` | 截图（raw screencap）、tap/swipe、`hw_key*`（sendevent 模拟实体键盘）、ADBKeyboard 输入、`ime_shown()` |
| `src/skydango/device/mumu.py` | MuMu 原生截图（external_renderer_ipc.dll，约 9 ms/张），`AdbDevice.screenshot()` 优先用它 |
| `src/skydango/vision/chatlog.py` | 解析聊天记录面板（C）：分行、拆说话人、认自己的消息 / 被屏蔽的消息、前后帧对齐找新消息 |
| `src/skydango/vision/icons.py` | 动作图标的剪影匹配（多尺度 matchTemplate） |
| `src/skydango/vision/bubbles.py` | 旧方案：3D 画面里找头顶气泡（不推荐，见 game-ops） |
| `src/skydango/chat/` | 读消息（`reader.py`）、大模型回复（`responder.py`）、发送（`sender.py`）、去重、记忆（`memory.py`，见下） |
| `src/skydango/game/wheel.py` | 快捷动作轮盘：图标库、读取 / 编辑轮盘、按数字键做动作、扫描动作列表 |
| `src/skydango/game/emotes.py` | 聊天时做动作：可用动作 + 限速、轮盘上的直接按键、白名单动作换进 swap_slots、退出恢复 |
| `src/skydango/agent.py` | 主循环：读 → 攒一会儿 → 回复 → 限速 → 发送；默认 dry-run |
| `src/skydango/runlog.py` | 每次 `run` 的运行目录（见下） |
| `src/skydango/vision/env.py` | 识别环境：每隔几秒在后台 OCR 3D 画面，认好友头顶的名字（身边有谁）和地名，写进提示词 |
| `src/skydango/vision/detect.py` `track.py` `perception.py` `weaklabel.py` | YOLO 感知层（开发中，默认关）：检测器（ONNX / ultralytics）、IoU 追踪、`PerceptionWatcher`（接口同 env，多认陌生人）、弱标注，见下 |
| `src/skydango/vision/viewer.py` | 识别可视化网页（`view` / `run --view`）：标准库 HTTP 服务，画面 + 识别框 + 状态放在同一份快照里，框和中文标签由浏览器画 |
| `src/skydango/game/social.py` | 社交互动：好友头顶圆圈里出现牵手 / 拥抱 / 击掌图标时点圆圈接受（请求由 env 的后台扫描发现），图标模板在 `assets/social/` |
| `src/skydango/game/friendtree.py` | 点人物打开好友树面板、截图、关掉（大脑的 `check_friend`，默认关，未在真机验证） |
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`claude.py` 起进程 / 隔离、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角 |
| `src/skydango/config.py` | 所有可调参数和默认值（坐标都是 0~1 归一化，按 1920×1080 标定） |

## 记忆（`memory/`，不进 git）

| 文件 | 内容 | 谁写 |
|---|---|---|
| `profile.md` | 三彩团子的人设（优先于配置里的 `reply.persona`） | 用户 |
| `friends.md` | 好友：游戏昵称、本名、称呼、关系 | 用户 |
| `inbox.md` | 随手记：每轮回复后后台单独调一次模型，只挑值得记的新信息，下一句就能用上 | 自动 |
| `notes.md` | 长期记忆：每 `notes_every` 轮把聊天记录 + inbox 整理进来（合并去重、删过期） | 自动，用户可改 |
| `history.jsonl` | 逐轮聊天记录，重启读回最近 `history_turns` 轮 | 自动 |

每次回复前都重新读这些文件，改了不用重启。只有 `run --live` 读写记忆（dry-run 的回复没真的发出去）。

**改了回复规则后要注意 history**：模型会模仿读回来的旧回复，盖过新规则（实测加了"动不了、别答应跑图"之后，
历史里"行，我跟着你们跑"那几句让它照样答应；清掉历史后立刻生效）。规则大改时把 `history.jsonl` 挪到 `memory/archive/`，
要点已经在 notes.md 里，不会失忆。
`memory init` 用配置生成 profile / friends，`memory show` 查看，`memory update` 立刻整理。

## 运行目录（`runs/`，不进 git）

每次 `run` 建一个 `runs/<时间>-<dry|live>[-echo]/`，启动时会打印路径；只留最近 `run.keep` 次（默认 20）。
排查问题先翻这里，不用再手动重定向日志：

| 文件 | 内容 |
|---|---|
| `agent.log` | 全量日志（DEBUG；终端照旧 INFO，`-v` 才显示 DEBUG） |
| `rows.log` | 面板每次变化时每行的识别结果，`*` 标出被判成新消息的行 —— 查漏读 / 晚读 / 读错说话人 |
| `replies.jsonl` | 每轮：收到的消息、回复（`null` = 不回复或被过滤）、是否真的发出 |
| `frames/*.jpg` | 读到新消息时截的聊天面板（红框标新消息），`run.save_frames = false` 关掉 |
| `config.json` | 本次实际生效的配置（含 `--live` / `--echo` 覆盖） |
| `brain.jsonl` | 大脑每一轮：subtype、轮数、用量、total_cost_usd（订阅不按它收费，参考）、用了哪些工具、最后说了什么（只有 `--brain`）；`brain/` 下是 Claude Code 的工作目录（mcp.json、prompt.md） |

## 识别环境（`[env]`）

每 `env.interval` 秒在后台线程对 3D 画面做一次 OCR（单独的 OCR 实例、`env.threads` 个线程，不和读聊天抢 CPU），
面板开着时跳过被挡住的左边：
- 身边有谁：好友头顶的名字标签（实测置信度 1.00），拿 friends.md 的 `## 标题` 去模糊匹配；`env.keep` 秒内看到过就算在旁边
- 在哪：读到 `env.places` 里的地名就记下（进入新区域时的地名提示 —— **未在真机验证**）

结果写进提示词的“现在的环境”一节。回复规则里原来“你看不到画面、别编自己在哪”那条改成了“只知道环境里写到的”，
不改的话模型会无视环境信息。`python -m skydango env` 对当前画面识别一次，看它认出了什么。

扫描时顺带看好友名字下方的圆圈（`[social]`）：图标变成牵手 / 拥抱 / 击掌就记为请求，主循环里去点圆圈接受
（原地没反应补点、在动就等、消失就完成，见 game-ops §6）。好友的都接受，陌生人只接受点火（图标还没录到）；
输入框开着时不点；dry-run 只打印。`python -m skydango record` 连续截图，用来观察新的界面变化。

## YOLO 感知层（`[perception]`，开发中）

设计和 GPU 机器上的操作步骤见 `docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`（§13）。
- YOLO 做视觉第一道关卡：每帧检测 `player` / `player_unlit` / `name_tag` / `social_ring` / `self`，名字标签只裁小图跑 OCR 识别，身份跟着名字走
- 没点火的陌生人是黑影，单独一类 `player_unlit`，看到就是陌生人；点过火的陌生人外观和好友一样，靠名字标签分：有标签且对得上 friends.md 是好友，一直没标签、离得不远的是陌生人（身体发 `stranger` 事件）
- `enabled = true` 时替换 env 的定时整图 OCR，接口一样，身体 / 社交 / 眼睛不用改；关掉就退回原来的
- **还没有训练好的模型，所有阈值都没在真机验证**；`models/`、`datasets/` 不进 git

## 识别可视化（`[viewer]`，`view` / `run --view`）

设计见 `docs/superpowers/specs/2026-09-28-viewer-design.md`。浏览器打开 `http://127.0.0.1:8765/`：游戏画面上画出认出的好友名字（绿）、
没认出的名字标签（黄）、陌生人（橙）、没点火的黑影（紫）、团子（灰白）、互动圆圈（青）/ 请求（红粗框）、聊天记录面板（灰虚线）、
新读到的消息（粉，留 3 秒）；右边是身边有谁、陌生人、互动请求、检测耗时，`run --view` 时还有模式、待回复 / 刚说过（普通 Agent）
或牵着手、最近事件（大脑）。页面上能暂停、隐藏框、存图（浏览器下载带框的 PNG）。
- `view` **不往游戏里发任何输入**（不重开面板、不接请求、不说话），玩家自己玩、旁边开着看；`--images` 回放录像，每张图当场认完再显示；
  `--model` 临时用 YOLO 模型；`--port`、`--no-browser`
- `run --view` 行为和平时一样，身体 / Agent 每圈把这一帧交给网页（限 `viewer.fps`，没人看不压 JPEG），出错只记 DEBUG 日志
- 只监听 127.0.0.1（画面里有好友昵称和聊天）；`viewer.host = "0.0.0.0"` 手机也能看，但同一局域网的人都能看。网页不存盘

## 统管大脑（`[brain]`，`run --brain`）

设计见 `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体部分见 `2026-09-27-brain-design.md`）。
- 大脑 = 常驻的无界面 Claude Code（`claude -p` stream-json，订阅登录，`--model sonnet --effort low`）；身体的工具经本机 MCP 服务（`sky`）给它，
  `--tools ""` 关掉所有内置工具，只能调 look / look_at / look_around / status / chat_log / say / emote / set_request_policy / camera / camera_reset / check_friend
- **和用户自己的 Claude Code 隔离**：单独配置目录 `.brain-claude/` + `claude setup-token` 生成的令牌（用户环境变量 `SKYDANGO_CLAUDE_TOKEN`）。
  沿用用户登录会把用户的插件、钩子、技能一起加载进大脑（实测）。子进程里去掉 `ANTHROPIC_API_KEY`（有它时 `-p` 一定用它）
- 眼睛 = 一次性 `claude -p --model haiku`：有人来 / 走、画面大变（隔 ≥20 秒）或 3 分钟没看时，把身体最近一帧写成文字；大脑醒来的消息里只有文字，要原图才 `look(image=true)`
- `check_friend(x, y)`：点一下人物打开右侧好友树面板，截图交给大脑自己判断是不是好友，再按 ESC 关掉（用户确认）、恢复聊天面板（`game/friendtree.py`）。
  面板样子、好友和陌生人的面板怎么区分**未核对**，`[friend_check] enabled` 默认关；先用 `friend-check X Y` 手动核对，截图在 `runs/<…>/friend-check/`
- 身体线程独占设备；每轮最多 6 次工具、2 句话（`ToolBox` 计数）；`say` 照样过 `clean_reply`；牵着手时 `emote` 要 `force=true`；陌生人只能接点火
- 大脑一轮 120 秒没结果就结束进程、下次 `--resume` 接回；连续失败 120 秒或额度用完：聊天交给 `[llm]`（DeepSeek）备用回复，额度用完 10 分钟后再试
- 退出：身体先复原镜头、恢复轮盘（不等大脑）→ live 时让大脑写一份经过记进 `inbox.md` → 按进程树结束 Claude Code
- 前提：`pip install --user mcp`；运行一次 `claude setup-token` 并 `setx SKYDANGO_CLAUDE_TOKEN "<令牌>"`

## 常用命令

```bash
python -m skydango devices                # 连接 / 截图尺寸 / 当前输入法
python -m skydango shot [--grid]          # 截图到 tmp/shot.png
python -m skydango detect                 # 读一次聊天记录面板（先在游戏里按 C），标注图 tmp/detect.png
python -m skydango say "【AI】你好"        # 发一句（输入框没开会先按 Enter）
python -m skydango chat --emotes 鞠躬,害羞  # 终端里和人设聊天，假装轮盘上有这些动作
python -m skydango run [--echo] [--live] [--duration 秒] [--no-emotes]  # Agent；默认 dry-run，--echo 不调模型，--duration 到点自己退出，牵着手时 --no-emotes
python -m skydango run --brain [--live] [--duration 秒]  # 统管大脑（Claude Code，订阅）；先 claude setup-token、设 SKYDANGO_CLAUDE_TOKEN
python -m skydango look [--prompt 文件]    # 截一张图让眼睛（Haiku）描述（调眼睛的提示词、看它会不会编名字）
python -m skydango friend-check X Y       # 点一下人物打开好友树、截图、再关掉（核对面板和关法），图在 tmp/friend-check/
python -m skydango memory init|show|update # 记忆：生成人设 / 好友文件、查看、立刻整理
python -m skydango env                    # 对当前画面识别一次环境（身边有谁、在哪）
python -m skydango view [--images 目录] [--model 模型] [--port 端口] [--no-browser]  # 只看不动：网页上实时画识别框；--images 回放录像
python -m skydango run --view ...          # 跑 Agent / 大脑时顺便开可视化网页
python -m skydango perception bench [--model yolo11n.pt] [--images 目录]  # YOLO 测速（没训练前用官方模型看硬件）
python -m skydango perception detect [图片]  # 跑一遍 YOLO 感知，标注图 tmp/perception.png
python -m skydango perception label <录像目录> [--preview]  # 用现有识别器出弱标注 → datasets/sky（YOLO 格式）
python -m skydango emotes scan            # 截下动作列表所有图标 → emotes/scan/，总览图 _sheet.png
python -m skydango emotes wheel           # 读轮盘 8 格
python -m skydango emotes set 5 鞠躬       # 放动作进格子（3、8 默认锁定）
python -m skydango emotes do 鞠躬          # 做动作（不在轮盘上会先换上去）
python -m pytest -q                       # 单元测试（合成画面 + 假设备，不需要模拟器）
```

图标库：`emotes/<动作名>.png`，由 `emotes scan` 的结果改名而来（用户负责命名）；`_` 开头的文件被忽略。

## 工作约定

- **临时文件放项目下的 `tmp/`**（gitignore），不要放系统 Temp 目录 —— 用户要能直接翻看截图。
- 改了游戏里的东西（尤其是轮盘）要能还原：改之前先截图 / 读出原状态，测完恢复并核对。
- 往游戏里发输入前先确认状态：输入框开没开（`ime_shown()`）、聊天记录面板开没开、编辑界面有没有残留。
  每一步操作后截图看一眼，再决定下一步。
- **别用 `timeout` 包 `run`**：Git Bash 的 `timeout.exe` 在 Windows 上会派生子进程，停掉外层后 Python 变成孤儿继续跑
  （实测两个 Agent 同时在线，重复回复、CPU 被占满）。要限时用 `run --duration`；提前停就按进程树结束 python
  （`taskkill /F /T /PID <pid>`），停完用 `Get-CimInstance Win32_Process` 确认没有残留的 python / timeout。
- 默认 dry-run。`reply.disclosure_prefix` 默认是 `【AI】`；用户本机 `config.toml` 里去掉了（角色"三彩团子"扮演陪玩）。
- 身份底线：人设可以不主动提自己是 AI，但**不能声称自己是真人**，有人认真问时要承认。
  由 `responder.RULES` 的"身份"一节 + `clean_reply` 里的 `_CLAIMS_HUMAN` 硬过滤共同保证（实测只靠提示词，DeepSeek 三次里有两次会说"我是真人"）。
  光遇未成年玩家多，不要删掉这两处。
- 光遇用户协议禁止自动化工具，有封号风险 —— 这是用户知情的前提，不要在游戏里做破坏性 / 刷屏类操作。
- 新增游戏交互时，先在真实游戏里手动验证（截图确认），再写进代码和 game-ops 文档。
- 聊天时做动作会松开牵手；Agent 退出时会把 `emotes.swap_slots` 换回原样，改换轮盘的逻辑要保证这一点。
