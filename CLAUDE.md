# CLAUDE.md — 给接手这个仓库的 Agent

SkyTricolorDango：在 MuMu 模拟器上"自己玩光遇"的 Agent。纯视觉：只截图 + 模拟输入，不读内存、不注入、不抓包。
当前能力：读聊天记录 → 大模型回复 → 在游戏里发言（可以带表情动作）；快捷动作轮盘的读取 / 编辑 / 做动作。

**先读 [docs/game-ops.md](docs/game-ops.md)**：光遇 + MuMu 的操作方式、界面坐标、实测踩过的坑都在里面。
不看它直接往游戏里发输入，很容易打错字、卡在界面里，或者覆盖掉用户的轮盘配置。

## Agent 规则

- 用中文回答。
- 每次任务改完，主动把功能分支合并进 main 并推送到远程。

## Skill：动手之前先挑一个

`.claude/skills/` 里的 skill 跟着仓库走，本地和云端（claude.ai/code）都会自动加载——插件装在本地是没用的，云端每次从仓库重新克隆。

**任何任务开始前，先看有没有对得上的 skill；有就先调用它，再动手。** 包括「先问个澄清问题」「先翻一下代码」之前——skill 会告诉你该怎么翻。用之前宣告一句「用 X skill 来做 Y」，然后照着它走；发现不合适再放弃。

| Skill | 什么时候用 |
| --- | --- |
| `frontend-design` | 新做或重做网页界面（目前只有识别可视化 `vision/viewer.py`）：视觉方向、字体、排版，避免模板感 |
| `brainstorming` | 要做新功能／改行为，需求和设计还没定死 |
| `writing-plans` | 需求清楚了，多步骤改动，写代码之前先出方案 |
| `executing-plans` / `subagent-driven-development` | 按方案逐条实现（`writing-plans` 收尾时由用户选：Native 用前者，Subagent-driven 用后者） |
| `test-driven-development` | 实现功能或修 bug，写实现代码之前 |
| `systematic-debugging` | 遇到 bug、测试挂了、行为不符合预期 |
| `verification-before-completion` | 要说「做完了／修好了／过了」之前，先跑命令拿证据 |
| `requesting-code-review` / `receiving-code-review` | 提交或合并前自查；以及收到评审意见之后 |
| `finishing-a-development-branch` | 实现完成、测试通过，决定怎么合回去 |
| `dispatching-parallel-agents` | 有两件以上互不依赖的事可以并行 |
| `using-git-worktrees` | 需要跟当前工作区隔离的分支作业 |
| `writing-skills` | 新增或修改 skill 本身 |
| `diagnosing-superpowers` | 某次会话里 skill 用得不对（没触发、重复干活、没按方案走、太慢太贵），要查原因或给上游报 bug |
| `using-superpowers` | 上面这套规矩的总纲 |

`frontend-design` 是 Anthropic 官方原版（[anthropics/skills](https://github.com/anthropics/skills)，Apache 2.0）；其余是 [obra/superpowers](https://github.com/obra/superpowers) 官方 v6.4.2（MIT）。来源、版本和同步方法见 `.claude/skills/README.md`。

- **skill 管不到真机**：TDD、验证类 skill 说的「跑测试」在这里指 `python -m pytest -q`（合成画面 + 假设备）；往游戏里发输入的改动仍然要按下面「工作约定」先在真机上截图验证，单元测试通过不等于游戏里能用。
- **合并方式以本文件为准**：`finishing-a-development-branch` 会列几种收尾选项，本仓库的规矩是改完合并进 main 并推送（见上）。

## 环境

- Windows + MuMu 12，游戏包名 `com.netease.sky.vivo`（国服 vivo 渠道），横屏 1920×1080
- adb 用 MuMu 自带的：`D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe`，设备名 `emulator-5554`
  （`127.0.0.1:16384` 在这台机器上是 offline）
- Python 3.13；OCR 用 `rapidocr` 3.x（`rapidocr_onnxruntime` 不支持 3.13）
- 包装在用户目录，`skydango` 不在 PATH 上，用 `python -m skydango ...`
- **YOLO 训练环境不占 C 盘**：显卡 RTX 5070 Ti Laptop（12 GB）。torch 2.11+cu128、ultralytics 等装在仓库的 `.pydeps/`（gitignore），
  用户 site-packages 里的 `skydango-pydeps.pth` 把它加进 sys.path；**当前目录在仓库里时**，pip / ultralytics / torch / matplotlib / CUDA 的缓存都指到 `.cache/`
  （`src/skydango/cachedirs.py`，`python -m skydango.cachedirs` 重写 .pth）。所以训练、`pip install` 都要在仓库目录下跑；
  `yolo` 不在 PATH 上，用 `.pydeps\bin\yolo.exe`。ultralytics 设置（`.cache/ultralytics/Ultralytics/settings.json`）的 datasets / weights / runs 指到仓库的 `datasets/`、`models/`、`runs/`
- 往 `.pydeps/` 加包：`python -m pip install --no-deps --target .pydeps <包>`（`--target` 不看已装的包，不加 `--no-deps` 会把 numpy 等再装一份、还可能装上 CPU 版 torch；
  缺哪个依赖就显式列出来，装完 `python -m pip check`）。torch 要加 `--index-url https://download.pytorch.org/whl/cu128`
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
| `src/skydango/vision/detect.py` `track.py` `perception.py` `weaklabel.py` | YOLO 感知层（开发中，默认关）：检测器（ONNX / ultralytics）、IoU 追踪、`PerceptionWatcher`（接口同 env，多认陌生人；画面被挡时暂停计时）、弱标注，见下 |
| `src/skydango/vision/hardcases.py` `compare.py` `augment.py` | 感知层一期工具：运行时收集难例、离线对比 YOLO 和整图 OCR（`perception compare`）、训练集增强（`perception augment`） |
| `src/skydango/vision/assist.py` | Claude 辅助标注（`perception label --assist`）：挑帧、人物候选框、`claude -p` 核对（分批并发、缓存、额度用完可续跑）、合并成标注 + 预览 + 待核对清单 |
| `src/skydango/vision/sweep.py` | 感知层二期：环绕扫描的纯计算（方位角、8 方位、多帧合并、转圈认团子、远近分档） |
| `src/skydango/vision/places.py` `unknownnames.py` `gesture.py` | 感知层三期：认地图（参考截图匹配）、没认出的名字清单、别人对团子做的动作（研究性质：切片段、评估、ONNX 接口） |
| `src/skydango/vision/viewer.py` | 识别可视化网页（`view` / `run --view`）：标准库 HTTP 服务，画面 + 识别框 + 状态放在同一份快照里，框和中文标签由浏览器画 |
| `src/skydango/game/social.py` | 社交互动：好友头顶圆圈里出现牵手 / 拥抱 / 击掌图标时点圆圈接受（请求由 env 的后台扫描发现），图标模板在 `assets/social/` |
| `src/skydango/game/friendtree.py` | 点人物打开好友树面板、截图、关掉（大脑的 `check_friend`，默认关，未在真机验证） |
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`claude.py` 起进程 / 隔离、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角 |
| `src/skydango/config.py` | 所有可调参数和默认值（坐标都是 0~1 归一化，按 1920×1080 标定） |
| `.claude/skills/` | 随仓库走的 skill（本地和云端都自动加载），见上面「Skill」一节和该目录的 README |

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
| `hard/*.jpg`、`hard.jsonl` | YOLO 感知层可能认错的画面（难例）和原因、检测框（`[perception] hardcases`，每次最多 200 张）；要用的及时 `perception label runs --from-runs` 收进数据集 |
| `spin/<时间>/` | 主人 `#spin` 转一圈的截图：转前 / 转完 / 每帧（文件名带按住后第几秒）和 `summary.json` |
| `unknown_names/` | YOLO 感知层读得清楚、但不在 friends.md 里的名字（`names.jsonl` + 每个名字一张裁剪图）；`perception unknown-names` 汇总，**只列出，不自动写 friends.md** |
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
**分三期实现**（总表见总纲 §15）：一期 `…-perception-phase1-design.md`（替换现有识别、上线可用）、二期 `…-phase2-design.md`（环绕扫描、认说话人等）、三期 `…-phase3-design.md`（认地图、跟随等）。实现某一期先读总纲 + 那一期的文档。
- YOLO 做视觉第一道关卡：每帧检测 `player` / `player_unlit` / `name_tag` / `social_ring` / `self`，名字标签只裁小图跑 OCR 识别，身份跟着名字走
- 没点火的陌生人是黑影，单独一类 `player_unlit`，看到就是陌生人；点过火的陌生人外观和好友一样，靠名字标签分：有标签且对得上 friends.md 是好友，一直没标签、离得不远的是陌生人（身体发 `stranger` 事件）
- `enabled = true` 时替换 env 的定时整图 OCR，接口一样，身体 / 社交 / 眼睛不用改；关掉就退回原来的
- **画面被挡时暂停计时**：黑屏、转镜头、开好友树、换轮盘、接互动时身体 / Agent 调 `env.held(原因)`；玩家自己开全屏界面靠"集体消失"规则兜底（≥ 2 人同时不见 + 画面大变）。EnvWatcher 是空实现
- **难例**：运行时把可能认错的画面存进 `runs/<…>/hard/`（旁路整图 OCR 核对、低置信度、闪烁、黑影来回变），下一轮 `perception label --from-runs --model` 预标注后只需修正
- **二期（代码已完成，见 `…-phase2-design.md`）**：`look_around` 在打开感知层时改成连续转一圈（`Camera.spin`）交给 YOLO 汇总"哪个方向有谁"，不叫眼睛；
  转圈时一直在中间不动的人就是团子（`self_box` 代替 `self_roi`，`perception label --spin` 自动补 `self` 框）；眼睛拿 YOLO 认出的好友 / 陌生人 / 团子位置（`scene_note`）；
  新类别 `typing`（头顶气泡）→ 陌生人的消息加注"说话的可能是右边远处那个陌生人"（恰好一个候选才加）；远近按团子框高分档，新事件 `approach`（有人朝团子走过来，`approach_strangers` 可关）
- **三期（代码已完成，见 `…-phase3-design.md`）**：远处的小人（框高 < `far_height`）没挂上名字标签时在它头顶裁一块再检测一次（`far_crops`）；
  认地图（`[places]`，要配合 `[perception]`）：`places/<地名>/*.jpg` 图库（不进 git）+ 图像特征模型，每 30 秒 / 画面大变后认一次，写进提示词"看起来在：…"；
  读得清楚但不在好友名单里的名字记进 `runs/<…>/unknown_names/`；别人对团子挥手 / 鞠躬（`[gesture]`，研究性质、**没有模型、默认关**）→ 身体发 `gesture` 事件，大脑可以用 `emote` 回礼；
  跟随只做了第 1 步：提示词里说想跟谁走就请他牵手（视觉伺服 `follow` 要等 `move` 工具接好、标定）
- 普通模式主人命令 `#spin [圈数]`：转一圈、截图存 `runs/<…>/spin/<时间>/`，打开感知层时回复带扫描结果；大脑模式不加
- **2026-09-28 训到第四版 `models/sky-yolo-v4.pt`**（311 帧、3 张地图，Claude 辅助标注、用户没人工核对），进度、数据、标注规则和待办见 `docs/progress/2026-09-28-yolo-training.md`；
  **还没在 `run` 里打开过感知层，所有阈值都没在真机验证**（`[spin] seconds_per_turn` / `hfov`、`near` / `far` / `self_height`、`approach_grow`、`typing_window`）；`models/`、`datasets/` 不进 git

## 识别可视化（`[viewer]`，`view` / `run --view`）

设计见 `docs/superpowers/specs/2026-09-28-viewer-design.md`。浏览器打开 `http://127.0.0.1:8765/`：游戏画面上画出认出的好友名字（绿）、
没认出的名字标签（黄）、陌生人（橙）、没点火的黑影（紫）、团子（灰白）、互动圆圈（青）/ 请求（红粗框）、聊天记录面板（灰虚线）、
新读到的消息（粉，留 3 秒）；右边是身边有谁、陌生人、互动请求、检测耗时，`run --view` 时还有模式、待回复 / 刚说过（普通 Agent）
或牵着手、最近事件（大脑）。页面上能暂停、隐藏框、存图（浏览器下载带框的 PNG）。
- `view` **不往游戏里发任何输入**（不重开面板、不接请求、不说话），玩家自己玩、旁边开着看；`--images` 回放录像，每张图当场认完再显示；
  `--model` 临时用 YOLO 模型；`--port`、`--no-browser`
- `run --view` 行为和平时一样，身体 / Agent 每圈把这一帧交给网页（限 `viewer.fps`，没人看不压 JPEG），出错只记 DEBUG 日志
- 只监听 127.0.0.1（画面里有好友昵称和聊天）；`viewer.host = "0.0.0.0"` 手机也能看，但同一局域网的人都能看。网页不存盘
- `run --brain --view` 时画面下方多一栏**大脑时间线**（设计见 `docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md`）：最近 50 轮，
  每轮点开看叫醒原因、收到的消息、说的话 / 思考、工具调用和返回、耗时和 tokens；能只看做了事的轮次、复制一整轮。数据在 `brain/trace.py`（只在内存里），走 `/brain` 长轮询
- `run --brain --view` 时还有一栏**手动控制**（设计见 `docs/superpowers/specs/2026-09-28-viewer-manual-control-design.md`）：直接让身体说话、做动作、转视角 / 复位 / 环视、
  看人（点画面上的人），**总是真执行**（大脑 dry-run 也一样），照样过身体的护栏；做成后放 `manual` 事件告诉大脑。`brain/manual.py` + `POST /control`；
  只在本机模式挂（`viewer.host` 是局域网时关掉），请求要带 `X-Skydango` 头、`Host` 必须是本机（防别的网页 / DNS 重绑定）

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
- 调提示词时加 `--view`：网页上的大脑时间线能看到每一轮它收到了什么、调了什么工具、工具返回了什么（见「识别可视化」）；
  手动控制栏能绕过大脑直接试身体的工具（身体方法的 `live=True`），大脑会收到 `manual` 事件
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
python -m skydango look [图片] [--prompt 文件]  # 截一张图（或给一张截图）让眼睛（Haiku）描述，先打印交给它的位置说明
python -m skydango camera spin [--turns N] [--seconds S]  # 转一圈、边转边截图存 tmp/spin/<时间>/（标定一圈几秒、视野角）
python -m skydango friend-check X Y       # 点一下人物打开好友树、截图、再关掉（核对面板和关法），图在 tmp/friend-check/
python -m skydango memory init|show|update # 记忆：生成人设 / 好友文件、查看、立刻整理
python -m skydango env                    # 对当前画面识别一次环境（身边有谁、在哪）
python -m skydango view [--images 目录] [--model 模型] [--port 端口] [--no-browser]  # 只看不动：网页上实时画识别框；--images 回放录像
python -m skydango run --view ...          # 跑 Agent / 大脑时顺便开可视化网页
python -m skydango perception bench [--model yolo11n.pt] [--images 目录]  # YOLO 测速（没训练前用官方模型看硬件）
python -m skydango perception detect [图片]  # 跑一遍 YOLO 感知，标注图 tmp/perception.png
python -m skydango perception label <录像目录> [--preview] [--model 模型]  # 弱标注（+ 模型预标注）→ datasets/sky（YOLO 格式）
python -m skydango perception label <录像目录> --assist [--model 模型] [--all-frames]  # Claude 辅助标注：挑帧 + 人物框由 Sonnet 核对，清单在 datasets/sky/_assist/review.md（令牌同 [brain]）
python -m skydango perception label runs --from-runs --model 模型  # 把各次运行存下的难例收进数据集
python -m skydango perception label <spin 目录> --spin --model 模型  # 转圈录像：认出团子，每帧自动补 self 框
python -m skydango perception augment datasets/sky  # 训练集加运动模糊 / 压暗样本（只动 train）
python -m skydango perception compare <录像目录> [--model 模型] [--far-crops 0]  # 同一批录像对比 YOLO 和整图 OCR → tmp/compare/<时间>/report.md（含远处认出率）
python -m skydango perception unknown-names [--last 5]  # 最近几次运行里读到、但不在好友名单里的名字（只列出）
python -m skydango perception clips <录像目录>  # 动作识别的数据：按人物轨迹切 16 帧片段 → datasets/gesture/_unlabeled
python -m skydango perception gesture-eval datasets/gesture --model 模型  # 动作模型的精确率 / 召回率
python -m skydango places add <地名> [--image 图]  # 认地图：截当前画面（遮掉人和 UI）存进 places/<地名>/
python -m skydango places test [目录] | places bench --model A --model B  # 逐张认地图 / 图库上留一法比较特征模型
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
