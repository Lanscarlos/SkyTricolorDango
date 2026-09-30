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
| `src/skydango/chat/` | 读消息（`reader.py`）、大模型回复（`responder.py`）、发送（`sender.py`）、聊天面板开关（`panel.py`，见下）、去重、记忆（`memory.py`，见下） |
| `src/skydango/game/wheel.py` | 快捷动作轮盘：图标库、读取 / 编辑轮盘、按数字键做动作、扫描动作列表 |
| `src/skydango/game/emotes.py` | 聊天时做动作：可用动作 + 限速、轮盘上的直接按键、白名单动作换进 swap_slots、退出恢复 |
| `src/skydango/agent.py` | 主循环：读 → 攒一会儿 → 回复 → 限速 → 发送；默认 dry-run |
| `src/skydango/runlog.py` | 每次 `run` 的运行目录（见下） |
| `src/skydango/vision/env.py` | 识别环境：每隔几秒在后台 OCR 3D 画面，认好友头顶的名字（身边有谁）和地名，写进提示词 |
| `src/skydango/vision/envdiff.py` | 两轮聊天之间环境变了什么（谁来了 / 走了、陌生人数、到了哪），写进那一轮的用户消息、跟着历史走 |
| `src/skydango/vision/people.py` | `Person`（track_id、好友 / 陌生人 / 黑影、框、左 / 前 / 右、近 / 中 / 远）：感知层的 `people()` 给身体、技能、大脑用 |
| `src/skydango/vision/detect.py` `track.py` `perception.py` `weaklabel.py` | YOLO 感知层（开发中，默认关）：检测器（ONNX / ultralytics）、IoU 追踪、`PerceptionWatcher`（接口同 env，多认陌生人；画面被挡时暂停计时）、弱标注，见下 |
| `src/skydango/vision/hardcases.py` `compare.py` `augment.py` | 感知层一期工具：运行时收集难例、离线对比 YOLO 和整图 OCR（`perception compare`）、训练集增强（`perception augment`） |
| `src/skydango/vision/assist.py` | Claude 辅助标注（`perception label --assist`）：挑帧、人物候选框、`claude -p` 核对（分批并发、缓存、额度用完可续跑）、合并成标注 + 预览 + 待核对清单；`Protocol` 让物品模式复用 |
| `src/skydango/vision/objlabel.py` | 物品模式（`perception label <数据集> --objects`）：提示词、解析、写回（人物行不动、物品行整体替换、人改先祖）、预览、清单 |
| `src/skydango/vision/sweep.py` | 感知层二期：环绕扫描的纯计算（方位角、8 方位、多帧合并、转圈认团子、远近分档） |
| `src/skydango/vision/places.py` `unknownnames.py` `gesture.py` | 感知层三期：认地图（参考截图匹配）、没认出的名字清单、别人对团子做的动作（研究性质：切片段、评估、ONNX 接口） |
| `src/skydango/vision/viewer.py` | 识别可视化网页（`view` / `run --view`）：标准库 HTTP 服务，画面 + 识别框 + 状态放在同一份快照里，框和中文标签由浏览器画 |
| `src/skydango/vision/panels.py` `game/panels.py` `assets/panels/` | 面板识别：特征卡快看 + OCR 细读 + 通用兜底认出开着哪些面板（`vision`）；按卡片关面板、点按钮（`game`）；五张特征卡（见「面板识别」） |
| `src/skydango/game/social.py` | 社交互动：好友头顶圆圈里出现牵手 / 拥抱 / 击掌图标时点圆圈接受（请求由 env 的后台扫描发现），图标模板在 `assets/social/` |
| `src/skydango/game/friendtree.py` | 点人物打开好友树面板、截图、关掉（大脑的 `check_friend`，默认关，未在真机验证） |
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`claude.py` 起进程 / 隔离、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角、`skills.py` 技能层（见「统管大脑」）、`occasion.py` 场合（见「看场合主动开口」）、`reflex.py` 反射（见「身体反射」） |
| `src/skydango/inner/` | 内心层（见「内心层」）：`ledger.py` 关系卡和这次上线（纯数据、拼文字）、`store.py` 读写 `memory/inner/`、`days.py`「日子」一节、`backfill.py` 从 history 回填、`open_ledger` / `show_lines`；第 2 期 `energy.py` 精力、`mind.py` 心情 / 别扭 / 心愿、`effects.py` 倍数、`reflect.py` 反思、`finish_reflection`；第 3 期 `persona.py` 性格档案（口头禅 / 老梗 / 看法） |
| `src/skydango/console/` | 管理面板（`console`）：设置清单和 `console.toml` / `secrets.toml` 读写（`settings.py` `tomlfile.py`）、团子子进程起停（`runner.py`、子进程侧看门狗 `watchdog.py`）、启动预检 / 测试按钮 / 设备检测（`preflight.py` `probes.py` `devicecheck.py`）、HTTP 服务和转发（`server.py`）、页面 `static/console.html`（见「管理面板」） |
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
| `inner/` | 内心账本（见「内心层」）：`people.json` 好友关系卡、`days.jsonl` 每次上线一行、`current.json` 这一次（运行中）、`mind.json` 心情 / 别扭 / 心愿、`diary.md` 日记（一天一节）、`persona.json` 性格档案（口头禅 / 老梗 / 看法，可以直接删条目） | 自动，不用手改 |

每次回复前都重新读这些文件，改了不用重启。只有 `run --live` 读写记忆（dry-run 的回复没真的发出去）。

**改了回复规则后要注意 history**：模型会模仿读回来的旧回复，盖过新规则（实测加了"动不了、别答应跑图"之后，
历史里"行，我跟着你们跑"那几句让它照样答应；清掉历史后立刻生效）。规则大改时把 `history.jsonl` 挪到 `memory/archive/`，
要点已经在 notes.md 里，不会失忆。
`memory init` 用配置生成 profile / friends，`memory show` 查看，`memory update` 立刻整理。
随手记和整理（`NotesKeeper`）在大脑模式和 `memory update` 里走 Claude Code（`[brain] memory_model`，默认 sonnet，一次性 `claude -p`、令牌同大脑、不给工具，`brain/claude.py` 的 `ClaudeLlm`）；
只有 `run --no-brain` 的普通 Agent 还用 `[llm]`（DeepSeek）。

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

## 聊天面板（`[panel]`，`chat/panel.py`）

设计见 `docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md`。面板开着时方向键转视角、缩放、长按 Z、Q 都没反应，还挡住画面左边三分之一，
所以由 `PanelManager` 统一开关（一次 run 只有一个，cli 建好传给身体 / Agent、镜头、轮盘、互动、好友树、技能）：
- `mode = "always"`（**默认**）：一直开着，关久了自动重开 —— 和以前一样
- `mode = "auto"`：平时关着，每 30 秒按 C 看一眼；来了好友 / 有人走近 / 好友头顶冒出"正在输入"气泡 / 大脑调 `chat_log` 时提前看；
  读到新消息就一直开着，安静 45 秒再关；团子说话前先开面板。**还没在真机验收（spec §9），验收前别改默认**
- **新代码要关面板一律 `with panel.borrow("谁"):`**（点屏幕会顺带关面板的用 `close=False`），不许自己按 C；
  嵌套时最外层归还才恢复，闲着时归还不重开
- 聊天内容只从面板读；YOLO 的 `typing` 气泡只用来判断"该去看了"（这一类还没标注数据，气泡触发要等重训）

## 识别环境（`[env]`）

每 `env.interval` 秒在后台线程对 3D 画面做一次 OCR（单独的 OCR 实例、`env.threads` 个线程，不和读聊天抢 CPU），
面板开着时跳过被挡住的左边：
- 身边有谁：好友头顶的名字标签（实测置信度 1.00），拿 friends.md 的 `## 标题` 去模糊匹配；`env.keep` 秒内看到过就算在旁边
- 在哪：读到 `env.places` 里的地名就记下（进入新区域时的地名提示 —— **未在真机验证**）

结果写进提示词的“现在的环境”一节（每次现取、不进历史）；两轮之间的**变化**（“这之间：小明来了；陌生人 0→2 个”）另外写在那一轮用户消息的开头，
会跟着 history.jsonl 存下来（`vision/envdiff.py`，重启后第一轮不写）。回复规则里原来“你看不到画面、别编自己在哪”那条改成了“只知道环境里写到的”，
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
- **物品识别（代码已完成，见 `2026-09-29-object-recognition-design.md`；还没有数据和模型）**：类别末尾追加 `bench` 座位 / `bonfire` 篝火 / `instrument` 乐器 / `spirit` 先祖（编号 6~9，旧编号不变；用 v4 时自然为空）。
  `objects()` 给出方位和远近（按框底边，`object_near` / `object_far` **未标定**，连续 `object_min_hits` 帧才算）；状态里"画面里的东西：座位（左边·近）"、眼睛的位置说明、网页"附近的东西"都有；
  只认出来告诉大脑，**不会走过去坐下**（F 期）。先祖单独成类后不再算陌生人。数据：`perception label datasets/sky --objects` 给已标好人的数据集补标（Claude 判成先祖的人物框自动改、列进清单；重跑跳过做过的帧和你改过的帧，`--recheck` 才重核；增强图不核对，augment 放在最后）；
  头顶气泡 `typing` 也在物品模式里一起标（`objlabel.LABEL_CLASSES` = 四类物品 + typing；typing **不进** `OBJECT_NAMES`，`objects()` 不报气泡；提示词版本 3）；
  `--only <通配>`（fnmatch 按图片文件名，带不带 `.jpg` 都行，可多次 / 逗号分隔）只处理匹配的帧，别的帧不核对、不写回、不进清单；
  烛火、光之翼等收集品不做（刷资源）。上线门槛和操作顺序见训练进度文档
- 普通模式主人命令 `#spin [圈数]`：转一圈、截图存 `runs/<…>/spin/<时间>/`，打开感知层时回复带扫描结果；大脑模式不加
- **本机在用 `models/sky-yolo-v7.pt`**（2026-09-30，539 帧，能认头顶气泡 `typing`；v4 在 run 里真机用过、v5 / v6 没上线），进度、数据、标注规则、各版对比和待办见 `docs/progress/2026-09-28-yolo-training.md`；
  **标气泡别只靠 Claude**：它标不出好友名字下方叠着的文字气泡，要 OCR 兜底（做法见进度文档 09-30 一节）；
  **还没在 `run` 里打开过感知层，所有阈值都没在真机验证**（`[spin] seconds_per_turn` / `hfov`、`near` / `far` / `self_height`、`approach_grow`、`typing_window`）；`models/`、`datasets/` 不进 git

## 识别可视化（`[viewer]`，`view` / `run --view`）

设计见 `docs/superpowers/specs/2026-09-28-viewer-design.md`。浏览器打开 `http://127.0.0.1:19399/`：游戏画面上画出认出的好友名字（绿）、
没认出的名字标签（黄）、陌生人（橙）、没点火的黑影（紫）、团子（灰白）、互动圆圈（青）/ 请求（红粗框）、聊天记录面板（灰虚线）、
新读到的消息（粉，留 3 秒）；右边是身边有谁、陌生人、互动请求、检测耗时，`run --view` 时还有模式、待回复 / 刚说过（普通 Agent）
或牵着手、最近事件（大脑）。页面上能暂停、隐藏框、存图（浏览器下载带框的 PNG）。
- `view` **不往游戏里发任何输入**（不重开面板、不接请求、不说话），玩家自己玩、旁边开着看；`--images` 回放录像，每张图当场认完再显示；
  `--model` 临时用 YOLO 模型；`--port`、`--no-browser`
- `run --view` 行为和平时一样，身体 / Agent 每圈把这一帧交给网页（限 `viewer.fps`，没人看不压 JPEG），出错只记 DEBUG 日志
- 只监听 127.0.0.1（画面里有好友昵称和聊天）；`viewer.host = "0.0.0.0"` 手机也能看，但同一局域网的人都能看。网页不存盘
- `run --view`（大脑模式）时画面下方多一栏**大脑时间线**（设计见 `docs/superpowers/specs/2026-09-28-viewer-brain-trace-design.md`）：最近 50 轮，
  每轮点开看叫醒原因、收到的消息、说的话 / 思考、工具调用和返回、耗时和 tokens；能只看做了事的轮次、复制一整轮。数据在 `brain/trace.py`（只在内存里），走 `/brain` 长轮询
- `run --view`（大脑模式）时还有一栏**手动控制**（设计见 `docs/superpowers/specs/2026-09-28-viewer-manual-control-design.md`）：直接让身体说话、做动作、转视角 / 复位 / 环视、
  看人（点画面上的人）、盯人（输入名字或在画面上点好友，调 `track`）/ 停下（`stop_task`）、读面板 / 关面板，**总是真执行**（大脑 dry-run 也一样），照样过身体的护栏；做成后放 `manual` 事件告诉大脑。`brain/manual.py` + `POST /control`；
  只在本机模式挂（`viewer.host` 是局域网时关掉），请求要带 `X-Skydango` 头、`Host` 必须是本机（防别的网页 / DNS 重绑定）

## 面板识别（`[panels]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-29-panels-design.md`，面板表、录样本和核对步骤见 game-ops §7。
- **快看**：每个已知面板一张特征卡（`assets/panels/<名字>/card.toml` + 模板小图，特征 template / dark / builtin / text），身体每圈判断、去抖；
  整屏黑的帧不判。**细读**：OCR 面板区域 → 标题、正文、按钮（撤退类 / 卡片 allow / 要放行 / never）。
  **通用兜底**：画面大变 / 右上角 × / 每 5 秒 OCR 屏幕中部（聊天面板里的字不算），整行是按钮词 + 一段正文就是"不认识的面板"，
  **默认只报告、不拦操作**（`unknown_blocks = false`，头顶气泡、告示牌可能误认）
- 身体：面板开 / 关发 `panel` 事件，`status` 里有"开着的面板"，开着时 `env.held("panel")`；换轮盘、看好友树期间 `expect` 自己打开的面板。
  **遮挡护栏** `clear_view`：转镜头、做动作、点人、接互动、说话前，已核对且 `auto` 的面板顺手关掉，别的拒绝、交给大脑
- 大脑工具 `panel_read` / `panel_press` / `panel_close`（按之前会再读一次，面板换了就要求重读）；按钮规则：撤退类（整个按钮就是关闭 / 取消……）直接按，别的要卡洛在聊天里 `#允许 <按钮>`（60 秒、用一次作废），
  `[panels] never` 和卡片 `never`（花钱、删好友、退出、共享空间"加入"……）**放行了也不按**
- **五张卡（聊天记录面板、动作面板、轮盘编辑、好友树、共享空间邀请）都还没在真机核对、缺模板图**：未核对的只报告不自动关，
  录样本后用 `panels scan` / `cut` / `read` 核对再改 `verified = true`

## 管理面板（`[console]`，`console`）

设计见 `docs/superpowers/specs/2026-09-29-console-design.md`，计划 `docs/superpowers/plans/2026-09-29-console.md`。`python -m skydango console` → 浏览器开 `http://127.0.0.1:19390/`：
总览（启动选项 + 叫醒 / 停止 + 状态卡片 + 日志尾巴）、实时画面（iframe 嵌子进程的 viewer）、设置、设备检测。**还没在真机上用过**（spec §8 真机验证 1~6）。
- **三个文件**：`config.toml` 面板只读不写；面板改的设置写 `console.toml`、密钥按环境变量名写 `secrets.toml`（明文，都 gitignore，和 config.toml 同目录）。
  加载顺序 默认值 → config.toml → console.toml；`secrets.toml` **覆盖**已有环境变量。终端直接跑命令也读这两个文件（启动时日志里打「console.toml 覆盖了 N 项」）；
  `console` 自己不把密钥写进自己的环境变量（页面上「清除」之后子进程才不会继承旧 Key），只注入它起的子进程
- **父子进程**：面板起 `python -m skydango -c <config> run [--no-brain] --live|--dry-run [--no-emotes] [--duration N] --view --viewer-port 19391 --no-browser --parent-pid <面板>`；
  模式和真发 / 只看总是显式传，`config.toml` 的 `reply.dry_run` 不会改掉面板上的选择。同一时间只有一个子进程
- **停止**：面板 `POST /shutdown` 给子进程的 viewer → `interrupt_main()`，走和 Ctrl+C 一样的收尾；`[console] stop_timeout`（60 秒）还没退就按进程树强杀，页面提示检查轮盘。
  子进程每 2 秒看父进程还在不在（`--parent-pid`），面板没了就自己正常退出；`/shutdown` 和看门狗共用一个**只中断一次**的钩子（`watchdog.once`），第二次中断不会打断收尾；
  面板终端里等收尾时再按一次 Ctrl+C = 强杀。19391 已经有人响应（上次留下的团子）时提示并能让它退出，**这时不让启动新的**（免得两个团子同时在线）
- Windows 上子进程用 `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`：有自己的隐藏控制台，关掉面板终端窗口不会把团子直接结束（**未在真机验证**）。
  面板到子进程的本机请求都**不走代理**（开着 Clash 时 urllib 默认会把 127.0.0.1 也转走）；`console.toml` / `secrets.toml` 坏了面板照样能开，在设置页和预检里报错
- **安全**：只监听 127.0.0.1；`/api/*`、`/live/*` 都校验 Host（防 DNS 重绑定）；POST 要 `X-Skydango: 1` + JSON + ≤ 64 KB（和 viewer 共用 `is_local_host` / `post_guard`）。
  浏览器拿不到完整密钥（只显示「已设置（sk-…abcd）」）
- 团子运行时不能做设备检测（设备归身体线程独占）；运行中改设置照样保存，提示重启后生效
- viewer 为此多了 `/status`（只有状态、不带图）和 `/shutdown`，页面里的请求改成相对路径（放在 `/` 和 `/live/` 下都能用）；大脑模式的状态多了「正在做」「刚说过」

## 统管大脑（`[brain]`，`run` 默认）

设计见 `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体部分见 `2026-09-27-brain-design.md`）。
- **`run` 默认就是大脑模式**；`run --no-brain` 进旧的普通 Agent（`agent.py`），只留作调试（`--brain` 保留兼容，不用加）
- 大脑 = 常驻的无界面 Claude Code（`claude -p` stream-json，订阅登录，`--model sonnet --effort low`）；身体的工具经本机 MCP 服务（`sky`）给它，
  `--tools ""` 关掉所有内置工具，只能调 look / look_at / look_person / look_around / status / chat_log / recall / say / emote / set_request_policy / camera / camera_reset / move / check_friend / track / stop_task / panel_read / panel_press / panel_close
- **和用户自己的 Claude Code 隔离**：单独配置目录 `.brain-claude/` + `claude setup-token` 生成的令牌（用户环境变量 `SKYDANGO_CLAUDE_TOKEN`）。
  沿用用户登录会把用户的插件、钩子、技能一起加载进大脑（实测）。子进程里去掉 `ANTHROPIC_API_KEY`（有它时 `-p` 一定用它）
- 眼睛 = 一次性 `claude -p --model haiku`：有人来 / 走、画面大变（隔 ≥20 秒）或 3 分钟没看时，把身体最近一帧写成文字；大脑醒来的消息里只有文字，要原图才 `look(image=true)`
- `check_friend(x, y)`：点一下人物打开右侧好友树面板，截图交给大脑自己判断是不是好友，再按 ESC 关掉（用户确认）、恢复聊天面板（`game/friendtree.py`）。
  面板样子、好友和陌生人的面板怎么区分**未核对**，`[friend_check] enabled` 默认关；先用 `friend-check X Y` 手动核对，截图在 `runs/<…>/friend-check/`
- `recall(query, who, days)`：翻以前的聊天（`chat/recall.py`）：在 `history.jsonl` 里按关键词（中一个就算，中得多的排前）/ 人名 / 往前几天（默认 14，最多 90）找原话，
  最多 8 轮、每轮 200 字，附上 notes.md / inbox.md 里相关的行；只读、不调模型、不占身体线程，dry-run 也能查。提示词要求有人提起以前的事时先查再答，查不到就说记不清、别顺着编。
  history 只记团子开口的那几轮（那句话 + 之前听到的），团子没接话时别人聊的不在里面
- **重启接得上话**：启动时把 `history.jsonl` 最近 `[brain] history_turns`（默认 20，0 关）轮原话放进系统提示词末尾的「上次聊到哪」一节，
  标出最后一轮是多久前（格式同 recall）；放系统提示词而不是第一条消息，自动压缩时不会被总结掉。dry-run 也带（只读）
- `look_person(名字)`：按名字（容忍 OCR 错一两个字）找到这个人、裁出来给大脑看原图（"看我衣服好看吗"）；打开感知层用 YOLO 的框，
  否则按名字标签往下估一块（宽 3 倍、高 6 倍标签高，**未在真机核对**）；和 look 共用频率限制；找不到时告诉大脑现在认得出谁。
  **被团子挡住时换角度**（`[peek]`，`brain/peek.py` 纯决策 + `Body._peek`，**未在真机验证**）：好友只有名字标签、标签压在团子框上（团子框 = YOLO 的 `self`，只信画面水平中间附近的）→
  在 `look_person` 里闭环：每按一下（`Camera.nudge` 左右 / `zoom_once` 拉近拉远）等画面停稳、看 YOLO 最新结果再定下一下；贴太近先拉远、转不动拉远再转、露出来太小就拉近（拉近后又挡住 / 出画面退一步）；
  **看完镜头不复位**（结果和 status 的"镜头："告诉大脑，要不要 `camera_reset` 它自己定）；status 里有"被你挡住：小明"。dry-run、黑屏、track 在跑时不转，只说被挡住了
- **技能层**（`brain/skills.py`，计划见 `docs/superpowers/plans/2026-09-28-brain-skills-roadmap.md`）：大脑一轮几秒到十几秒，盯人 / 走过去 / 点火这种要每秒修正的事来不及做，
  交给身体按主循环的节拍闭环做。同时最多一个技能；黑屏、超时、出错都算失败；结束时发 `task_done` / `task_failed` 事件叫醒大脑，
  状态里有"正在做：…"；`stop_task` 叫停；退出时先停技能再复原镜头。approach / light_candle 见计划 E~G 期。
  状态里的"画面里：小明（左边·近）…"来自感知层 `people()`（暂停中或最近一帧超过 1 秒就是空的）
- **`track(名字, 秒)`**（技能 D，`brain/track.py`，计划 `docs/superpowers/plans/2026-09-29-brain-track.md`；**代码已完成，待真机验收**）：要打开感知层、有镜头、开始时画面里认得出这个人，dry-run 拒绝（手动控制照做）；秒数夹到 1~60。
  每圈按名字找目标（人物框优先，框不稳用 `max_age` 秒内的名字标签 x），偏出死区就 `Camera.nudge` 短按 0.02~0.1 s（`hw_key_hold`：一条 adb 命令、sleep 在模拟器里），按完等 `settle` 秒；
  离得太近转不动（同方向按 3 次误差没缩小 20 px）就停手；`lost_after` 秒看不到算跟丢；聊天记录面板开着时只等（不按键、不算转不动 / 跟丢）。
  跟踪期间身体不发 `arrive` / `leave` / `stranger`、丢掉攒着的 `approach` / `gesture`、不报画面大变，也不调 `env.held()`；互动请求、自动接受、牵手状态照常。
  跟踪中调 `camera` / `camera_reset` / `look_around` / `check_friend` 会先停下 track（结果里写"先停下了盯着…"）。数字在 `[track]`（`gain` 是估的）。
  **`camera_reset` 现在是闭环**：每次离开原位前只拍一张参照缩略图（画面上半、避开聊天面板）；nudge 按时长分别记净次数（转动和时长不成比例、左右对称，D0），
  复位时按同样的时长逐次反向重放，再左右 0.02 s 小步比相似度（最多 30 步），最高相似度 < 0.5 就退回粗转位置、结果里说"没对准"。
  身体走动过、黑屏过之后参照图作废（只粗转）。工具 / 手动控制的 `camera_reset` 等 60 秒（`RESET_TIMEOUT`）；退出时先恢复轮盘、再复位镜头，细调限 8 秒
- `move(direction, steps)`：W/A/S/D 小步走（`brain/locomotion.py`，每步按住 `[brain] move_step` 秒，**步长没在真机标定**）；一次 1~3 步、两次隔 `move_min_interval` 秒，
  牵着手要 `force=true`，走出去没有复位
- **主人命令窗口**：卡洛（`[brain] owner_name`，精确匹配）发 `#` 开头的消息后 `owner_window` 秒内放宽：`move` 一次最多 6 步、不用等间隔；
  `emote` 不管动作冷却（`emotes.min_interval`，换轮盘的冷却照旧）；`camera` 一次最多 8 步；牵着手 `move` 不用 `force`。
  真用到放宽时工具结果标"（主人命令模式）"。底线、身份规则不受影响
  聊天记录面板开着也能走（卡片 `allows` 里有 `move`），别的面板挡着照 `clear_view` 处理；dry-run 不按键
- **什么时候叫醒大脑**（`brain/events.py` 的 `BACKGROUND`）：聊天、主人命令、好友第一次来、有人走近、任务结束等攒 `chat.debounce` 秒就叫醒；
  背景事件（陌生人来去、好友走开 / `return` 回来、互动请求和已接受、牵手 / 松手、画面变化）不单独叫醒，攒够 `brain.background_wait`（20 秒）才兜底叫醒一次。
  同一个人攒着的"走开"和"回来"互相抵消，陌生人事件只留最新的。好友走开后 `brain.rejoin`（60 秒）内又出现发 `return`，不再打招呼。
  2026-09-29 真机 v4 感知层 dry-run：改之前 5 分钟醒 42 次，41 次是背景事件
- 身体线程独占设备；每轮最多 6 次工具、2 句话（`ToolBox` 计数）；`say` 照样过 `clean_reply`；做动作不会松开牵手（牵着手也照做，只有 `move` 要 `force=true`）；陌生人只能接点火
- 大脑一轮 120 秒没结果就结束进程、下次 `--resume` 接回；连续失败 120 秒或额度用完：聊天交给 `[llm]`（DeepSeek）备用回复，额度用完 10 分钟后再试
- 记忆整理（随手记 inbox.md、整理 notes.md）也走 Claude：`[brain] memory_model`（默认 sonnet），每次起一个一次性 `claude -p`（工作目录 `runs/<…>/brain/memory/`）；额度用完时这一笔跳过，notes 下次再整理
- 退出：身体先恢复轮盘、再复原镜头（不等大脑）→ live 时让大脑写一份经过记进 `inbox.md` → 按进程树结束 Claude Code
- 调提示词时加 `--view`：网页上的大脑时间线能看到每一轮它收到了什么、调了什么工具、工具返回了什么（见「识别可视化」）；
  手动控制栏能绕过大脑直接试身体的工具（身体方法的 `live=True`），大脑会收到 `manual` 事件
- 前提：`pip install --user mcp`；运行一次 `claude setup-token` 并 `setx SKYDANGO_CLAUDE_TOKEN "<令牌>"`

## 看场合主动开口（`[proactive]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-29-proactive-chat-design.md`，计划 `docs/superpowers/plans/2026-09-29-proactive-chat.md`。
远期目标是把团子做成有存在感的**游戏里的伙伴**（像 Neuro-sama，但不做主播）；这是第一步：看到了就有话说。**还没在真机上跑过，数字都是估的**（spec「真机验证」四步）。
- **眼睛挑新鲜事**：自动看时附上上一份描述（≤ `prev_max_age`），多要一项“新鲜事”（天黑了、好友换装 / 弹琴、篝火……），身体放 `notice` 事件叫醒大脑；
  认地图的地名从一个变成另一个也发（"看起来到了雨林"）。大脑自己 `look` 不发；`notice_min` 内、和上一条一样、没熟人、主动额度用完都不发（省额度）。好友在身边时眼睛 `auto_look_busy`（60 秒）看一次
- **场合**（`brain/occasion.py`，纯计算）：热闹 / 安静 / 没熟人 + 上次主动开口有没有人接 + 还能主动说几句，写进 `status`（网页、管理面板卡片也有）。
  **主动** = 大脑这一轮不是被聊天 / 主人命令叫醒的（`brain_busy()` 为假）；手动控制说的不算
- **护栏**（`body.say`，只管主动）：没熟人不说、10 分钟额度（热闹 4 / 安静 2）、两句间隔 60 秒、连续 3 句没人接就停到有好友说话；dry-run 也计数
- **提示词**：“别自言自语”换成「主动开口」一节（看场合、带自己的看法别播报、新鲜事不是任务、不说就在心里写“不说：原因”）。
  调的时候用 `run --view` 看大脑时间线
- **喜好**：`memory init` 的人设模板多了「喜好和看法」；已有的 `memory/profile.md` 要自己把这一节加进去。随手记会记下团子说过的评价，下次态度一致
- `enabled = false` 完全照旧（旧提示词、不发 `notice`、不拦）。history.jsonl 先不清（旧回合只是没主动，不冲突），太保守再按「记忆」一节挪走

## 身体反射（`[reflex]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-30-body-reflex-design.md`，计划 `docs/superpowers/plans/2026-09-30-body-reflex.md`。
"让团子活着"四个子项目的第一个（反射 → 关系账 → 情绪 → 性格语气，见 spec 开头）。大脑一轮要几秒到十几秒，常见时刻由身体按规则当场反应，大脑再补上说什么。
**还没在真机上跑过，数字都是估的**（spec「真机验证」四步）。
- **判断在跟团子说话**（`brain/reflex.py` 的 `addressed()`，先从严）：好友说的，且叫了 `[proactive] self_names`、或团子说完 `followup_window`（30 秒）内、或身边只有他一个好友；陌生人、`#` 命令不算
- **输入气泡**：身体读到这样的消息当圈就 `sender.open()`（先 `panel.before_speak()`），头顶冒"正在输入"；大脑 `say` 直接用这个框发。
  关框：大脑"开框之后才开始的那一轮"结束了没说话（`Brain.last_turn` → `body.brain_turn`）、开了 `bubble_max`（45 秒）、**任何按键 / 点屏幕之前**（`clear_view` 里统一关，`say` 除外；`panel_press` / `panel_close` 也关）、有互动请求要接（`clear_view("social")`）。
  只关自己开的框（`sender.opened`）；技能（track）在跑、大脑离线、dry-run 时不开；status 里"输入框：开着（身体替你开的…）"
- **动作反射**（只用轮盘上已有的动作，不换轮盘）：被叫到时开框前按 `addressed_chance` 做 `addressed` 清单里的一个；感知层的 `gesture` 按 `return_chance` 用 `return_map` 回礼（回了就只发 `reflex` 事件、不再叫大脑决定，没回照旧发 `gesture`；框开着就回完再开回来）；
  闲着 `idle_min`~`idle_max` 秒（随机，有人说话 / 团子说话 / 任何按键类工具都重新计时）做 `idle` 清单里的一个（框开着不做）。**三张清单默认空**，用户按自己的轮盘填
- **额度**：反射**不占**大脑的动作冷却（`EmotePlayer.perform(name, reflex=True)` 只更新 `last_any`）；自己 `quota_window` 里最多 `quota` 个；任何两个动作之间至少 `min_gap`（4 秒，大脑的 `emote` 也查）
- 做完放 `reflex` 背景事件（不单独叫醒大脑），status "刚才下意识：…"、网页"反射"；提示词在「说话」一节多一句"身体已经替你冒了输入气泡"。黑屏、技能在跑、有互动请求、面板挡着时不做；dry-run 走 `pretend`
- `enabled = false` 完全照旧（不开框、不做动作、`gesture` 照旧交给大脑、提示词不变、`emote` 不查 `min_gap`）；管理面板有 `reflex.enabled` / `reflex.bubble` 两个开关

## 内心层（`[inner]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-30-inner-phase1-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase1.md`。
"让团子活着"的内在部分，统一放在一个内心层里，分三期：**1 账本**（这一期：关系卡 + 日子，不调模型）→ 2 反思（心情、精力、心愿、日记，接上后果）→ 3 性格（口头禅、老梗、放开太乖的规则）。
代替了反射 spec 开头列的子项目 2~4。**还没在真机上跑过，数字都是估的**（spec「真机验证」五步）。
- **关系卡**（只给好友，名字 = friends.md 的 `## 标题`，OCR 错字按最像的好友归）：第一次见、上次在身边、见过几次（离上次超过 `visit_gap` 30 分钟再出现才算新的一次，名字标签闪一下不多算）、
  哪几天、一共 / 今天待了多久（单圈最多 `max_step` 秒）、说过几句 / 跟团子说的几句（反射的 `addressed()`）、他最后一句
- **日子**：每次上线一行（开始、结束、见到谁、听到 / 说了几句、下线前大脑写的经过）；live 时每 `save_every` 秒写 `current.json`，被强杀后下次启动补一行"意外断了"；退出时身体收尾后、大脑写经过前先落一次账（`checkpoint`），写经过时被强杀也算正常下线
- **身体记账**（`Body._ledger_call`，出错只记日志）：`_watch_people` 每圈 `present`（跟踪中也记）、`_heard` 每句 `heard`、`say` 记 `said`、每圈末尾 `save`
- **大脑看到**：`arrive` 事件"小明 来到身边（第 13 次见；上次 3 天前；今天第一次）。他上次最后说的（3 天前）：「…」"（`return` 不带）；
  status "身边的好友：小明（今天一起 40 分钟·认识 21 天·一起玩过 9 天）"；系统提示词「日子」（第几次上线、上次下线多久前、上次的经过、很久没见的好友）；「说话」一节加了按交情说话、别报数字
- **第一次启用**（没有 `people.json`）从 `history.jsonl` 回填：第一次 / 最后一次说话、说过几句、最后一句、哪几天、见过几次（≈ 出现过的回填上线段数）；上线按相邻两轮隔 `session_gap` 分段。
  `people.json` 坏了就改名 `.bad-<时间>`、空卡开始、**不回填**
- **只在 `--live` 写盘**；dry-run 内存里照记（status 是真的），回填、补意外结束都只在内存里。`memory show` 多打印关系卡和最近 10 次上线（只读）
- `enabled = false` 完全照旧；管理面板有 `inner.enabled` 开关

**第 2 期：反思**（设计见 `docs/superpowers/specs/2026-09-30-inner-phase2-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase2.md`；**还没在真机上跑过，数字都是估的**，spec「真机验证」四步）
- **精力**（`energy.py`，身体每圈现算，不存盘）：几点（白天满、深夜低）− 连着挂了多久（离上次下线不到 `rest_gap` 算没睡、接着累）+ 最近 10 分钟有好友跟团子说话 − 最近一小时热闹太久；四档 精神 / 还行 / 有点累 / 困
- **反思**（`reflect.py`，一次性 `claude -p`，`[inner] reflect_model` 默认 sonnet，后台线程，结果回身体线程套进 `Mind`）：有动静时每 `reflect_every`（20 分钟）、好友说够 6 句后安静 3 分钟、下线时各一次；
  材料是上次以来的聊天原话、来去、相关好友的关系卡和笔记、人设、精力。回 JSON：心情（开心 / 平常 / 低落 / 烦 + 一句带原因的话）、别扭、心愿增删；下线那次再加日记和要点。额度用完 10 分钟内不反思，失败沿用上一份
- **代码守的规矩**（`mind.py`）：别扭只冲一起玩过 ≥ `grudge_min_days`（3）天的好友、最长 `grudge_max`（2 小时，同一个人再给也不续期）、同时 1 条、消气后冷却同样长；别扭对象说了“难过 / 不舒服 / 想哭 / 认真的……”（`DISTRESS`）身体当场撤掉，不等反思；离上次反思超过 `rest_gap`（睡过一觉）心情回到平常；心愿最多 3 条、7 天过期，kind 只有 惦记（挂在好友名下）/ 想做 / 小心思；不认识的一律丢
- **后果**（`effects.py`，倍数相乘）：主动开口额度 开心 ×1.5、低落 / 烦 ×0.5、有点累 ×0.75、困 ×0.5（下限 ×0.25、至少 1 句）；烦时被叫到不做小动作、开心概率 ×1.5；困时闲着的小动作更勤、大脑心跳慢一档。
  **接话永远照常**；在跟谁闹别扭时，整批都是他说的话不冒输入气泡（故意晚点接），牵手等互动照接
- **大脑看到**：status "心里：有点闷（…）· 有点困（…）· 跟小明闹别扭（…，还有 40 分钟消气）· 惦记：…"；arrive 后面接"你惦记着：…"；「日子」带上一篇日记（代替"上次的经过"）；提示词多一段心情 / 别扭（认真问、说难过时立刻作废）/ 惦记 / 小心思的规矩；网页和管理面板卡片有"心情""精力"
- **下线**：`checkpoint` → 最终反思（材料是这次上线的全部聊天，最多 80 行；超时压到 `console.stop_timeout − 25` 秒）写日记（`diary.md`，每次下线一段，「日子」取最后一段、带日期）和要点（inbox.md、`days.jsonl` 的经过）→ `close`，**不再让大脑写经过**；`reflect = false` 照第 1 期
- 只在 live 写 `mind.json` / `diary.md` / inbox；dry-run 照样反思（`--view` 里看得到），不写盘；管理面板有 `inner.reflect` 开关

**第 3 期：性格**（设计见 `docs/superpowers/specs/2026-09-30-inner-phase3-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase3.md`；**还没在真机上跑过，数字都是估的**，spec「真机验证」五步）
- **性格档案**（`persona.py`，`memory/inner/persona.json`）：口头禅（`catchphrases_max` 5）、和某个好友的老梗（每人 `jokes_per_friend` 3、总共 `jokes_max` 20）、看法（一个话题一句，`opinions_max` 10）。
  反思（`persona` 开着时系统提示词接上 `PERSONA_SYSTEM`、材料带「你攒下的性格」）顺带给 `persona_add` / `persona_used`；**代码守的规矩**（`Persona.apply`）：截 30 字（话题 10 字）、带敏感词（胖瘦丑矮、长相身材脸、爸妈家里、成绩考试分数、几岁年纪、学校班）的丢、
  老梗只挂在一起玩过 ≥ `grudge_min_days` 天的好友名下（OCR 错字按最像的好友归）、收着点的人不记新老梗、几乎一样的不重复加、同话题换立场；超上限先删 `hits` 最少的、再删最久没用的；`fade_days`（14 天）没用就淡出（启动时和每次套完）
- **大脑看到**：系统提示词「你攒下的性格」（启动时算一次，放在「日子」之前，空的不写）；`arrive` 在“你惦记着”之后接“你们的老梗：…”（最多 2 条）；
  提示词多一段「脾气」（`TEMPER_RULES`：有立场、熟人互损但损事不损人、会拒绝会偷懒但卡洛的 `#` 命令和有人真难过时不偷懒、有执念、对“收着点”的人好好说话），“接对方的话往下聊”换成“接得住就接，不想接也可以换个话头或者吐槽一句”；底线各节一字不动
- **收着点**（身体兜底）：任何好友说了难过类的话（第 2 期的 `DISTRESS`）→ `soft_minutes`（30 分钟）内 status 写“对小明收着点（他刚说「…」）”，反思不给他记新老梗
- `memory init` 的人设模板多了「## 脾气」（毛病 / 执念 / 雷点）；已有的 `memory/profile.md` 要自己加。`memory show` 末尾打印性格档案（只读），`profile.md` 永远优先、程序不改
- 只在 live 写 `persona.json`；dry-run 只在内存里，坏文件不改名。`persona = false` = 第 2 期原样（提示词、反思提示词和材料逐字一样，不记收着点）；管理面板有 `inner.persona` 开关
- 还是太乖：旧的乖回复会把模型拉回去，按「记忆」一节把 `history.jsonl` 挪到 `memory/archive/` 再试

## 常用命令

```bash
python -m skydango devices                # 连接 / 截图尺寸 / 当前输入法
python -m skydango shot [--grid]          # 截图到 tmp/shot.png
python -m skydango detect                 # 读一次聊天记录面板（先在游戏里按 C），标注图 tmp/detect.png
python -m skydango say "【AI】你好"        # 发一句（输入框没开会先按 Enter）
python -m skydango chat --emotes 鞠躬,害羞  # 终端里和人设聊天，假装轮盘上有这些动作
python -m skydango console [--port 端口] [--no-browser]  # 管理面板：填密钥、改设置、检测设备、启动 / 停止团子、看实时画面
python -m skydango run [--live | --dry-run] [--duration 秒] [--no-emotes]  # 团子（默认接统管大脑、dry-run）；先 claude setup-token、设 SKYDANGO_CLAUDE_TOKEN；--duration 到点自己退出
python -m skydango run --no-brain [--echo] [--live]  # 调试用的普通 Agent（DeepSeek 回复）；--echo 不调模型
python -m skydango panels scan [图片或目录]    # 面板识别：每张卡开没开、每个特征的分数 + 通用兜底，标注图 tmp/panels/（不发输入）
python -m skydango panels read [图片]          # 细读开着的面板：标题、正文、按钮和类别
python -m skydango panels cut <截图> <卡片> <文件名> --roi x1,y1,x2,y2  # 裁模板图存进 assets/panels/<卡片>/
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
python -m skydango perception label datasets/sky --objects [--model 模型] [--only 通配]  # 物品模式：给已标好人的数据集补标座位 / 篝火 / 乐器 / 先祖和头顶气泡 typing（先备份 labels/），清单在 _assist/objects.md；--only 只做文件名匹配的帧
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
- 做动作**不会**松开牵手（用户实测；以前记的"会松手"是对方自己断开的），走动才会。Agent 退出时会把 `emotes.swap_slots` 换回原样，改换轮盘的逻辑要保证这一点。
