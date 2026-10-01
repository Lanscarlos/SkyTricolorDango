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
- **有的机器上系统 Python 没装 pytest / skydango**（2026-09-30 实测）：用仓库的 `.venv`（gitignore，skydango 开发模式安装 + pytest / pillow / mcp / uvicorn / openai），
  测试跑 `.venv\Scripts\python.exe -m pytest -q`；git worktree 里也用主目录的这个 `.venv`（`tests/conftest.py` 会把 worktree 的 `src` 放在最前面）
- **YOLO 训练环境不占 C 盘**：显卡 RTX 5070 Ti Laptop（12 GB）。torch 2.11+cu128、ultralytics 等装在仓库的 `.pydeps/`（gitignore），
  用户 site-packages 里的 `skydango-pydeps.pth` 把它加进 sys.path；**当前目录在仓库里时**，pip / ultralytics / torch / matplotlib / CUDA 的缓存都指到 `.cache/`
  （`src/skydango/cachedirs.py`，`python -m skydango.cachedirs` 重写 .pth）。所以训练、`pip install` 都要在仓库目录下跑；
  `yolo` 不在 PATH 上，用 `.pydeps\bin\yolo.exe`。ultralytics 设置（`.cache/ultralytics/Ultralytics/settings.json`）的 datasets / weights / runs 指到仓库的 `datasets/`、`models/`、`runs/`
- 往 `.pydeps/` 加包：`python -m pip install --no-deps --target .pydeps <包>`（`--target` 不看已装的包，不加 `--no-deps` 会把 numpy 等再装一份、还可能装上 CPU 版 torch；
  缺哪个依赖就显式列出来，装完 `python -m pip check`）。torch 要加 `--index-url https://download.pytorch.org/whl/cu128`
- `config.toml` 是本机配置（gitignore），模板是 `config.example.toml`；记忆在私有仓库，新电脑先按「记忆」一节克隆到 `private/`
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
| `src/skydango/vision/embed.py` `onnxrt.py` | 特征模型的公共部分：`OnnxEmbedder`、`unit` / `cosine`（认地图、认装扮共用）；`onnxrt.py` 按 device（cuda / dml / cpu）选 onnxruntime 后端、建会话（YOLO、特征模型、动作模型共用） |
| `src/skydango/vision/appearance.py` | 认装扮（见「认装扮」）：好样本裁图、内置颜色特征 `ColorEmbedder`、外观记忆簿 `AppearanceBook`（好友 / 陌生人编号 / 团子自己、关系卡旧特征、`assign_friends`、换装判定）、攒训练数据 `CropSaver` |
| `src/skydango/vision/wardrobe.py` | 装扮描述器：排队、优先级、每小时额度，一次性 `claude -p --model haiku` 把人物裁图写成一句话 |
| `src/skydango/vision/appearance_eval.py` | 认装扮的离线标定（`perception appearance-eval`）：收集轨迹特征、相似度分布、建议门槛、藏标签重放、报告 |
| `src/skydango/vision/viewer.py` | 识别可视化网页（`view` / `run --view`）：标准库 HTTP 服务，画面 + 识别框 + 状态放在同一份快照里，框和中文标签由浏览器画 |
| `src/skydango/vision/panels.py` `game/panels.py` `assets/panels/` | 面板识别：特征卡快看 + OCR 细读 + 通用兜底认出开着哪些面板（`vision`）；按卡片关面板、点按钮（`game`）；六张特征卡（见「面板识别」） |
| `src/skydango/game/social.py` | 社交互动：好友头顶圆圈里出现牵手 / 拥抱 / 击掌图标时点圆圈接受（请求由 env 的后台扫描发现），图标模板在 `assets/social/` |
| `src/skydango/vision/candle.py` | 火焰圆盘（没点火的黑影站到身边时他身上的深色圆 + 火焰）：只用来判断黑影在能点火的距离里，**绝不点**；`white_ring` 分孤儿圆圈是举蜡烛请求（有白圈）还是圆盘 |
| `src/skydango/game/friendtree.py` | 点人物打开好友树面板、截图、关掉（大脑的 `check_friend`，默认关，未在真机验证） |
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`claude.py` 起进程 / 隔离、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角、`locomotion.py` 小步走（`move`）、`skills.py` 技能层（见「统管大脑」）、`attention.py` 空闲注意力 / `peek.py` 换角度、`occasion.py` 场合（见「看场合主动开口」）、`reflex.py` 反射（见「身体反射」） |
| `src/skydango/inner/` | 内心层（见「内心层」）：`ledger.py` 关系卡和这次上线（纯数据、拼文字）、`store.py` 读写 `memory/inner/`、`days.py`「日子」一节、`backfill.py` 从 history 回填、`open_ledger` / `show_lines`；第 2 期 `energy.py` 精力、`mind.py` 心情 / 别扭 / 心愿、`effects.py` 倍数、`reflect.py` 反思、`finish_reflection`；第 3 期 `persona.py` 性格档案（口头禅 / 老梗 / 看法）；内心页 `log.py` 流水账（`MindLog` + 反思前后 `diff`）、`api.py` `/inner` 接口的解析；`lull.py` 冷场追踪（见「冷场时的心理活动」） |
| `src/skydango/console/` | 管理面板（`console`）：设置清单和 `console.toml` / `secrets.toml` 读写（`settings.py` `tomlfile.py`）、团子子进程起停（`runner.py`、子进程侧看门狗 `watchdog.py`）、启动预检 / 测试按钮 / 设备检测（`preflight.py` `probes.py` `devicecheck.py`）、HTTP 服务和转发（`server.py`）、内心页数据（`inner_view.py`：读 `memory/inner/`、在跑时合并实时、删性格条目）、沙盒（`sandbox_view.py` 重置记忆 / 起始时间下限、`scenario.py` 剧本格式、`replay.py` 录制回放报告）、报告读取（`reports.py`）、页面 `static/`（`console.html` + `console.css` + `common.js` / `markdown.js` + 每页一个 js，左侧栏 + 六页：沙盒（默认）/ 真机团子 / 内心 / 剧本和报告 / 设置 / 设备；见「管理面板」「大脑沙盒」） |
| `src/skydango/brain/backstage.py` | 幕后（见「幕后」）：拼「幕后」一节、取"卡洛上次以来改了你什么"（git 提交）、读写 `inner/backstage.json` 标记 |
| `src/skydango/brain/world.py` `src/skydango/sandbox/` | 大脑沙盒（见「大脑沙盒」）：`World` / `BrainParts`（`_run_brain` 拆出的"接世界的东西"）；`sandbox/` 模拟时钟 `clock.py`、沙盒世界 `world.py`、聊天记录 `transcript.py`、操作和状态 `control.py`、JSON 接口 `server.py` |
| `src/skydango/config.py` | 所有可调参数和默认值（坐标都是 0~1 归一化，按 1920×1080 标定） |
| `.claude/skills/` | 随仓库走的 skill（本地和云端都自动加载），见上面「Skill」一节和该目录的 README |

## 记忆（`memory/`，不进 git；两台电脑用私有仓库同步，见本节末尾）

| 文件 | 内容 | 谁写 |
|---|---|---|
| `profile.md` | 三彩团子的人设（优先于配置里的 `reply.persona`） | 用户 |
| `friends.md` | 好友：游戏昵称、本名、称呼、关系 | 用户 |
| `inbox.md` | 随手记：每轮回复后后台单独调一次模型，只挑值得记的新信息，下一句就能用上 | 自动 |
| `notes.md` | 长期记忆：每 `notes_every` 轮把聊天记录 + inbox 整理进来（合并去重、删过期） | 自动，用户可改 |
| `history.jsonl` | 逐轮聊天记录，重启读回最近 `history_turns` 轮 | 自动 |
| `inner/` | 内心账本（见「内心层」）：`people.json` 好友关系卡、`days.jsonl` 每次上线一行、`current.json` 这一次（运行中）、`mind.json` 心情 / 别扭 / 心愿、`diary.md` 日记（一天一节）、`persona.json` 性格档案（口头禅 / 老梗 / 看法，可以直接删条目，管理面板「内心」页也能删）、`mind_log.jsonl` 流水账（每次反思改了什么、每 5 分钟一条精力、网页上删的条目；启动时删 30 天前的）、`backstage.json` 幕后告诉过团子的最后一个提交（见「幕后」） | 自动，不用手改 |

每次回复前都重新读这些文件，改了不用重启。只有 `run --live` 读写记忆（dry-run 的回复没真的发出去）。

**改了回复规则后要注意 history**：模型会模仿读回来的旧回复，盖过新规则（实测加了"动不了、别答应跑图"之后，
历史里"行，我跟着你们跑"那几句让它照样答应；清掉历史后立刻生效）。规则大改时把 `history.jsonl` 挪到 `memory/archive/`，
要点已经在 notes.md 里，不会失忆。
`memory init` 用配置生成 profile / friends，`memory show` 查看，`memory update` 立刻整理。
随手记和整理（`NotesKeeper`）在大脑模式和 `memory update` 里走 Claude Code（`[brain] memory_model`，默认 sonnet，一次性 `claude -p`、令牌同大脑、不给工具，`brain/claude.py` 的 `ClaudeLlm`）；
只有 `run --no-brain` 的普通 Agent 还用 `[llm]`（DeepSeek）。

### 两台电脑共用记忆（私有仓库）

本仓库是公开的，记忆和沙盒数据放在**私有仓库** `git@github.com:Lanscarlos/SkyTricolorDango-Memory.git`（2026-10-01 建），
克隆到本仓库根目录的 `private/`（gitignore）：`private/memory/` 是上面那张表里的记忆，`private/sandbox/` 是沙盒目录（记忆、`clock.json`、剧本、报告）。
用户白天在单位电脑、晚上在个人电脑：真机 `run --live` 只在个人电脑跑，单位电脑主要跑大脑沙盒。

新电脑第一次：
```bash
git clone git@github.com:Lanscarlos/SkyTricolorDango-Memory.git private
```
然后在本机 `config.toml` 里加（不加就还是用 `memory/`、`sandbox/`，读不到这份记忆）：
```toml
[reply]
memory_dir = "private/memory"

[sandbox]
dir = "private/sandbox"
```
- **用之前** `git -C private pull`，**团子 / 沙盒下线后**在 `private/` 里提交并推送（`history.jsonl`、`inner/` 是一直追加的，两台电脑别同时跑，不然会冲突）
- `private/` 里的 `.gitattributes` 是 `* -text`：文件原样保存，不转换换行符
- 沙盒的记忆目录不能在记忆目录里面（`check_separate` 会拒绝），所以是 `private/` 下并列两个目录，而不是直接把私有仓库克隆成 `memory/`
- **私有仓库别改成公开**，内容也别往本仓库里贴：里面有好友昵称、本名和聊天原话（好友里可能有未成年人）

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
| `appearance/` | 认装扮攒的训练数据：`crops/<身份>/*.jpg`（好友名，或 `t<轨迹>`）+ `appearance.jsonl`（`[appearance] save`，每条轨迹 2 秒一张、每次最多 2000 张） |
| `brain.jsonl` | 大脑每一轮：subtype、轮数、用量、total_cost_usd（订阅不按它收费，参考）、用了哪些工具、最后说了什么（只有 `--brain`）；`brain/` 下是 Claude Code 的工作目录（mcp.json、prompt.md） |

## 聊天面板（`[panel]`，`chat/panel.py`）

设计见 `docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md`。面板开着时方向键转视角、缩放、长按 Z、Q 都没反应，还挡住画面左边三分之一，
所以由 `PanelManager` 统一开关（一次 run 只有一个，cli 建好传给身体 / Agent、镜头、轮盘、互动、好友树、技能）：
- `mode = "always"`：一直开着，关久了自动重开 —— 以前的做法（沙盒强制用它）
- `mode = "auto"`（**默认**，2026-09-30 真机验收后改的，结果见 game-ops §3「按需打开聊天面板」）：平时关着，每 30 秒按 C 看一眼；来了好友 / 有人走近 / 好友头顶冒出"正在输入"气泡 / 大脑调 `chat_log` 时提前看；
  读到新消息就一直开着，安静 45 秒再关；团子说话前先开面板。**老测试要常开的在辅助函数里写明 `mode = "always"`**
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
（原地没反应补点、在动就等、消失就完成，见 game-ops §6）。好友的都接受，陌生人只接受举蜡烛给团子点火（头顶没名字的圆圈要有白圈才算，深色火焰圆盘绝不点，见 game-ops §6）；
输入框开着时不点；dry-run 只打印。`python -m skydango record` 连续截图，用来观察新的界面变化。
没点火的黑影站到团子身边 `light_after` 秒（他身上出现深色圆盘 + 火焰，`vision/candle.py` 认；只有大脑模式找，普通 Agent 不管 light 请求），大脑模式下身体按 3 号键举蜡烛（**绝不点那个圆盘**：会跟着人走），
举满 `lit_min`（2 秒）后 YOLO 认成 `player` 满 `lit_frames` 帧、且他身上的圆盘消失超过 1 秒才算点亮（近处还黑着的人 YOLO 常直接认成 `player`；闪光时会冒出重复轨迹、原轨迹冻住——按最近处理的一帧算 `lit_stale` 秒没接上就换和它最后的框重叠的新 `player`，这条路要原来那个人和替身自己的圆盘都消失超过 1 秒、替身还要被扫满 1 秒；感知暂停 / 没在跑时一律不算）、过 `bow_delay` 鞠躬（顺带放下蜡烛）；`light_timeout` 秒没亮就按 3 放下、这个人不再点；鞠躬没做完不再举第二次、举之前先查 `reflex.min_gap`、身体替大脑开着输入框时不举；
放下前先关替大脑开的框（按数字键会关掉它）；举着时接受了别的互动（点圆圈会放下蜡烛）或黑过屏（切场景、状态不明）就不再按 3 放下。接受别人点火后也鞠躬。
大脑能用 `set_request_policy("stranger", "light", false)` 关掉；**未在真机验证**（spec `docs/superpowers/specs/2026-10-01-light-unlit-stranger-design.md` §9；录像 c / d 上的核对见 game-ops §6）

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
- **核显 / 没有 N 卡的机器**（`device = "dml"`，`vision/onnxrt.py`，**未在 Windows 核显上验证**）：在 GPU 机器上导出 `.onnx`（`.pydeps\bin\yolo.exe export model=models/sky-yolo-v7.pt format=onnx imgsz=960`）拷过去，
  `pip uninstall onnxruntime` 再 `pip install onnxruntime-directml`（两个包都叫 `onnxruntime` 模块，只能装一个；以后 `pip install -e ".[ocr]"` 会把 onnxruntime 装回来盖掉，要重装 directml；`pip check` 报 skydango 缺 onnxruntime 是正常的）。
  先 `perception bench --model models/sky-yolo.onnx --images <录像目录>` 看后端是不是 `DmlExecutionProvider`、每帧多少 ms，再按实测把 `fps` 降下来（估计 2~3）。
  DirectML 没装就退回 CPU 并警告；`.pt` 模型不支持 dml（torch 没有 DirectML，退回 CPU）。`[places]` / `[appearance]` 的 `device` 也能填 dml，动作模型跟着 `[perception] device`

## 认装扮（`[appearance]`，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-01-appearance-design.md`，计划 `docs/superpowers/plans/2026-10-01-appearance.md`。
名字标签挡住 / 太远读不到时靠外观把人认回来，认得出走开又回来的陌生人，说得出谁穿的什么。**还没在真机上跑过，数字都是估的**：
先用 `perception appearance-eval` 在录像上标定 `match` / `changed`（spec §10），再按 spec「真机验证」五步（`view` 看"像小明?"、陌生人走开再回来还是同一个编号、dry-run 的 status 和描述额度、`--live` 换装和关系卡、`runs/<…>/appearance/` 的图）。
- **特征**：YOLO 人物框（点过火的人和团子）每 `every` 帧裁一次好样本（框够高、不被别的框 / 聊天面板压住），默认内置颜色直方图（`model = "color"`，头 / 身体各一份色相饱和度），也能换 `.onnx`；平滑进轨迹 `data["feat"]`
- **好友**（`maybe`）：只从这一帧挂着名字标签的样本学；没标签的轨迹攒够 `min_samples` 个好样本后和这次见过的好友（`match`）/ 关系卡旧特征（`card_match`，更严）比，过门槛且领先第二像的 `margin` → "像小明"。
  名字标签永远说了算；标签挂在别处的不参加；一个名字一帧只给一条轨迹；连续 `recheck` 个样本不像就摘。**后果保守**：不算陌生人、好友还在身边时刷新在场（标签被挡不冒"走开了"），但不发 `arrive` / `return`、不打招呼；`people()` 里 `sure = False`
- **陌生人**：认装扮开着时好样本不够先不判陌生人，最多多等 1.5 秒（`STRANGER_GRACE`）；判成陌生人后编号"陌生人A / B…"（这次上线不复用），走开超过 `keep` 又被认回来发 `stranger_back` 背景事件（"刚才那个陌生人A（白斗篷）又回来了"）；
  同屏两个陌生人不会共用编号；`stranger_forget`（30 分钟）没见就忘
- **装扮描述**（`describe`，默认开）：描述器排队用 Haiku 把裁图写成一句话（团子自己优先、好友其次、近处陌生人最后；每小时 `describe_max` 次、同一人一次上线最多重新描述 `redescribe_max` 次，额度用完等 `quota_wait`）；
  status 多一行"你自己：…"，"身边的好友：小明（粉色长斗篷·…）"，"画面里：…、像小红（没看到名字，…）、陌生人A（白斗篷，…）"；`look_person` 也接受"陌生人A"；提示词多一段装扮的规矩（`enabled` 时才加）。沙盒里不描述
- **判不判换装**（`outfit_change`，**默认关**）：10-01 在 15 段录像上标定（`tmp/appearance-eval/`），颜色特征下同一身衣服常跌到 0.15~0.5，0.70 的门槛会把同一身判成换装、往关系卡里记假装扮；
  换 DINOv2-small（`models/dinov2-small.onnx`）也没稳住（对比在 `tmp/appearance-eval/compare-color-dinov2.md`）。**关着时**：不追加新的一套、不发 `outfit` 事件、上线中途不重新描述；
  好友 / 团子每次上线描述一次（每个好友每次上线多一次 Haiku），好友的描述**覆盖**关系卡最近那一套；描述回来前 status 不写他的装扮（不拿上次的旧描述说事）。换了更稳的特征模型再打开；`changed` 默认 0.40
- **关系卡**：`Card.outfits` 每个好友留最近 `outfit_keep`（3）套（描述、特征、特征模型 key、第一次 / 最后一次的日期），启动时载入记忆簿；`outfit_change` 开着时：第一次靠标签认出好友、和卡里最近一套比：< `changed` 算换了装、排描述，
  新描述回来后放 `outfit` 背景事件（"小明换了装扮：上次是「…」，现在「…」"，arrive 文字不变）；同一套直接用卡里的描述、不花额度。只在 live 写盘；管理面板「内心」页关系卡显示装扮
- **攒数据**（`save`，默认开）：好样本存进 `runs/<…>/appearance/`（见「运行目录」），以后训认人模型用（spec §8，这一期不做）
- **识别可视化**：按外观认的好友画浅绿虚线、标"像小明?"，陌生人标编号，鼠标悬停看装扮
- **已知限制**：颜色直方图不看亮度，白 / 灰 / 黑发色、同色深浅分不开；不同地图、白天晚上光照差得多，跨天靠关系卡认人弱（所以 `card_match` 更严）；撞衫（季节装扮、默认斗篷）会认错；身高没做；没点火的黑影没有外观
- `enabled = false` 照旧：系统提示词的规矩、事件、status、`people()` / 识别框逐字一样；只有 `look_person` 工具说明里总写着能传「陌生人A」（关着时找不到、照常报没找到，无害）；管理面板有 `appearance.enabled`、`appearance.describe` 两个开关
- 中途换装（`outfit_change` 开着时）：好友 / 团子的平均特征离上次描述（或上次换装稳下来时）低于 `changed` 就记一次换装（好友发 `changed`），平均特征挪稳到新那套之前不再判；不看描述有没有回来（描述关了、没挂描述器、描述器放弃了照样判），最多 `redescribe_max` 次

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
- **掉线弹框**（`disconnect`，靠文字"连接错误 / 网络连接失败"认）："重试""取消"都在卡的 `never` 里、没有关法：身体记一条 WARNING、
  告诉大脑"游戏掉线了，按钮都别按，等卡洛自己点重试"（2026-09-30 晚真机掉线过一次，当时"重试"被当成正文、大脑连按灰的"取消"）
- **六张卡（聊天记录面板、动作面板、轮盘编辑、好友树、共享空间邀请、掉线弹框）都还没在真机核对、缺模板图 / 截图**：未核对的只报告不自动关，
  录样本后用 `panels scan` / `cut` / `read` 核对再改 `verified = true`

## 管理面板（`[console]`，`console`）

设计见 `docs/superpowers/specs/2026-09-29-console-design.md`，计划 `docs/superpowers/plans/2026-09-29-console.md`。`python -m skydango console` → 浏览器开 `http://127.0.0.1:19390/`：
**2026-10-01 重做**（设计 `docs/superpowers/specs/2026-10-01-console-redesign-design.md`，计划 `docs/superpowers/plans/2026-10-01-console-redesign.md`）：左侧栏（运行卡片：状态、停止、`run.error`；各页入口）+ 六页，路由 `#sandbox`（沙盒，**默认页**）、`#live`（真机团子 = 原来的总览 + 实时画面：停着是启动表单 + 预检问题，跑着左边 iframe 嵌子进程 viewer、右边状态卡片 + 日志）、`#inner`（内心，见下）、`#scripts`（剧本和报告）、`#settings`（设置，带分组目录）、`#device`（设备）；旧的 `#overview` 已改名 `#live`。**还没在真机上用过**（spec §8 真机验证 1~6）。
- **页面拆分**：`static/console.html` + `console.css`（颜色只在 `:root` 定义，别处用 `var(--…)`，有测试查）+ `common.js`（`$` `el` `post` `ask` `toast` `problemList` `Pages` `go` `refresh` 等公共接口，node 能 require）+ `markdown.js` + 每页一个 js（`sandbox.js` `live.js` `inner.js` ……），经 `/console/static/<名字>.css|js` 提供，只给 `static/` 目录列表里真有的文件（Windows 设备名 con.js / nul.js 一律 404，不碰文件系统）。页面不引外部资源、请求一律相对路径。
- **原生 `confirm` / `prompt` / `alert` 全换成页内对话框和提示条**（`ask()` / `toast()`；Claude 桌面版内嵌浏览器里原生弹窗用不了），测试禁止再出现。
- **预检**：大脑模式下也查 LLM Key；每个问题带 `setting` 跳转目标，「去设置 →」跳到设置页并高亮那一行（樱花底闪一下）。
- **剧本和报告页**：报告在页内直接读（`console/reports.py`，`GET /api/sandbox/reports[/<name>]`），不用再去翻 `sandbox/reports/`。
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
- **「内心」页**（设计见 `docs/superpowers/specs/2026-09-30-inner-viewer-design.md`；**代码已完成，还没在真机上用过**）：现在（心情 / 精力 / 别扭 / 收着点 / 心愿）、精力曲线 + 心情色带（24 小时 / 7 天，圆点 = 一次反思，点了跳到记录）、
  反思记录（`changes` 逐行、没收下的折叠、下线那次标出）、性格档案（每条能「删」）、关系卡、最近 10 次上线和日记。
  `GET /api/inner` 读 `memory/inner/`（团子不在跑也能复盘），团子 `running` 时再取子进程 viewer 的 `/inner`（`Body.inner_snapshot`，经身体线程、3 秒超时），现在 / 性格以实时为准，取不到标"实时取不到"；
  `POST /api/inner/forget`：在跑转发 `/inner/forget`（`Body.forget`，live 才写 `persona.json`；dry-run 的团子只删内存里的，面板顺手把文件也改了），没在跑直接改 `persona.json`，启动 / 停止中拒绝；别处有团子在跑（孤儿端口有响应、或 `current.json` 在 3×`save_every` 内更新过 = 终端里的 `run --live`）也拒绝，免得它把删掉的写回去。
  团子醒着时每 5 秒刷新，别的时候打开时读一次 + 「刷新」。流水账只在开了反思时记（`[inner] reflect`），live 写盘、dry-run 只在内存里

## 统管大脑（`[brain]`，`run` 默认）

设计见 `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体部分见 `2026-09-27-brain-design.md`）。
- **`run` 默认就是大脑模式**；`run --no-brain` 进旧的普通 Agent（`agent.py`），只留作调试（`--brain` 保留兼容，不用加）
- 大脑 = 常驻的无界面 Claude Code（`claude -p` stream-json，订阅登录，`--model sonnet --effort low`）；身体的工具经本机 MCP 服务（`sky`）给它，
  `--tools ""` 关掉所有内置工具，只能调 look / look_at / look_person / look_around / status / chat_log / recall / say / emote / set_request_policy / camera / camera_reset / attention / move / check_friend / track / stop_task / panel_read / panel_press / panel_close
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
- **`track(名字, 秒)`**（技能 D，`brain/track.py`，计划 `docs/superpowers/plans/2026-09-29-brain-track.md`；**2026-09-30 真机验收通过**，实测见 game-ops §2）：要打开感知层、有镜头、开始时画面里认得出这个人，dry-run 拒绝（手动控制照做）；秒数夹到 1~60。
  每圈按名字找目标（人物框优先，框不稳用 `max_age` 秒内的名字标签 x），偏出死区就 `Camera.nudge` 短按 0.02~0.1 s（`hw_key_hold`：一条 adb 命令、sleep 在模拟器里），按完等 `settle` 秒；
  离得太近转不动（同方向按 3 次误差没缩小 20 px；越按越大是追不上、接着追；好友在画面外时光遇把名字贴在屏幕边上、x 卡住不变，所以在画面最边上也不算转不动）就停手；从画面边上出去的，看不到的这几秒里接着往那边转去找（在中间被挡住不转）；`lost_after` 秒看不到算跟丢；盯人单独的最长按键 `chase_max`（0.15 s，2026-09-30 真机后从 0.1 提上来，`gain` 也提到 0.15）；聊天记录面板开着时只等（不按键、不算转不动 / 跟丢）。
  跟踪期间身体不发 `arrive` / `leave` / `stranger`、丢掉攒着的 `approach` / `gesture`、不报画面大变，也不调 `env.held()`；互动请求、自动接受、牵手状态照常。
  跟踪中调 `camera` / `camera_reset` / `look_around` / `check_friend` 会先停下 track（结果里写"先停下了盯着…"）。数字在 `[track]`（`gain` 是估的）。
  **`camera_reset` 现在是闭环**：每次离开原位前只拍一张参照缩略图（画面上半、避开聊天面板）；nudge 按时长分别记净次数（转动和时长不成比例、左右对称，D0），
  复位时按同样的时长逐次反向重放，再左右 0.02 s 小步比相似度（最多 30 步），最高相似度 < 0.5 就退回粗转位置、结果里说"没对准"。
  身体走动过、黑屏过之后参照图作废（只粗转）。工具 / 手动控制的 `camera_reset` 等 60 秒（`RESET_TIMEOUT`）；退出时先恢复轮盘、再复位镜头，细调限 8 秒
- **空闲注意力（东张西望）**（`[attention]`，`brain/attention.py` 纯决策 + `Body._watch_attention`；spec `docs/superpowers/specs/2026-09-30-idle-attention-design.md`、计划 `docs/superpowers/plans/2026-09-30-idle-attention.md`；**未在真机验证，数字都是估的**）：
  **只在 `[panel] mode = "auto"`、聊天面板关着、身体闲着时动**（always 模式下完全不动）。身体做反射、大脑定模式：候选目标（好友 / 陌生人说话 = 感知层 `talkers()`、走近 = `recent_approaches()`、挥手鞠躬、站着的好友）按
  基础兴趣 ×（1 − 看腻）挑一个，`Camera.nudge` 小步把他拉向画面中间（中间带 40% 内不动，同 track 的方向和转不动判断）；看腻了 / 转不动就看别的，没什么可看就隔 8~20 s（× 心情精力 `Effects.wander` × 模式）随意往一边看一眼。**不回位**。
  冒气泡 / 有人走近时**先看一眼再开面板**（`PanelManager.hold_off`，最多 `look_first` 2 s）。不借面板、不走 `clear_view`、不重置反射的闲着计时、不套 `env.held()`；按键后 settle 内冒出的 approach 当成自己转出来的丢掉。
  输入框开着、技能在跑、有互动请求、在回聊天、黑屏、牵手、别的面板开着、刚做完动作、dry-run 时不按（dry-run 照样算，status 有"在看：…"）。
  大脑工具 `attention(mode, focus)`：随意 / 好奇（随意看更勤、陌生人说话更有意思）/ 专心（只看好友说话、冲团子来的、关注的人）/ 别动；focus = 更想看谁；**不算"做了事"**（不进 `ACTIONS`）
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
- 身体线程独占设备；每轮最多 6 次工具、2 句话（`ToolBox` 计数）；`say` 照样过 `clean_reply`；做动作不会松开牵手（牵着手也照做，只有 `move` 要 `force=true`）；陌生人只能接点火（他举蜡烛给团子点）和点亮（light：团子按 3 举蜡烛给没点火的黑影点）
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

## 冷场时的心理活动（`[lull]`，大脑模式）

设计见 `docs/superpowers/specs/2026-10-01-lull-musing-design.md`，计划 `docs/superpowers/plans/2026-10-01-lull-musing.md`。
起因：沙盒里好友安静下来后团子一直不说话，大脑只写“不说：等卡洛回”，心里太平静。**代码已完成，还没用真 Claude / 真机跑过，数字都是估的**（spec「用真 Claude 在沙盒验证」四步）。
- **两种冷场**（`inner/lull.py` 的 `LullTracker`，纯计算、墙钟）：① 好友在身边不说话了：最后一句（谁说的都算）过去 `stages[0]` 秒、这句之前 `talk_window` 秒内有好友说过话、他还在身边；
  ② 聊着聊着走了：好友走开时他 `leave_spoke` 秒内说过话、或团子 `leave_said` 秒内说过话。说话人按 `similar(…, 0.75)` 模糊匹配（OCR 错字），看不出是谁的不算好友
- **节点**：`stages`（60 / 180 / 360 秒）各放一次立刻叫醒的 `lull` 事件（不在 `BACKGROUND` 里，不受心跳退档影响）；② 走开先照旧发背景 `leave`，`leave_grace`（15 秒）还没回来才叫醒一次（名字标签闪一下不算，宽限内回来 `leave` / `return` 照旧互相抵消），之后用后两个节点；
  最后一个节点之后再过 `stages[0]` 秒冷场结束（“后来就一直安静着” / “一直没回来”），不再挂在 status 里；快进跳过几个节点只发最新的；黑屏时不推进；
  冷场中团子又说话不重新计时，只在文字里写“这之间你又说了…”；跟踪技能在跑时不判 ②（`_watch_comings` 本来就不跑）
- **结束**：任何好友开口（附在那条聊天事件后面：“（冷场了 2 分钟，你刚才在想：…）”）；② 的人回来（叫醒过的：原来的背景 `return` 换成 `lull` 事件“X 回来了（走开了 3 分钟，你刚才在想：…）”，超过 `rejoin` 的 `arrive` 也附上）；
  ① 的对象聊着聊着走了，想过的话转给 ②；冷了很久才走（不算聊着聊着）或镜头转开，① 结束、留下总结“后来他走开了”。冷场追踪和人来人走用同一份身边名单（`_lull_near`）
- **心里想的**：提示词「冷场的时候」要求被冷场叫醒时在最后的文字里写一行“心里：……”；`Brain.on_text` 每轮把最后的文字交给 `body.mused`（经 `body.call`），`parse_musing` 取最后一行、截 `musing_max` 字，
  只挂到大脑这一轮开始之前就有、已经叫醒过的冷场上（`brain.last_turn[0]`，冷场在这一轮里结束了就丢掉）；不加工具。status 多一项“冷场：懒洋洋大王 3 分钟没说话（最后是你说的「…」）· 在想：…（1 分钟时）”
- **留下的影响**：冷场结束（或下线前最终反思时还没结束）写成一行、说话人“（冷场）”，按冷场开始的时间插进反思材料（`_reflect_chat` / `_session_chat`）；冷到最后一个节点算一次 `reflector.stirred`；代码不直接改心情，由反思定
- **看得到的地方**：status、沙盒聊天记录旁白“── 心里：… ──”（`body.on_musing`）、内心页「现在」的“在想”（`inner_snapshot()["musing"]`，只有实时）、流水账 `musing` 行（`MindLog.musing`，live 才写盘，内心页反思记录里列出）。
  **不写** `history.jsonl`、随手记、关系卡
- **顺带修了**：被 `arrive` 之类叫醒的一轮里接上一句没回的好友聊天（`proactive.reply_window` 内、团子之后还没说过，`Body._pending_reply`）不再算主动开口；
  「主动开口」的“分寸”写清“没人接”只看状态里“上次主动开口”那一行（这一句改动 `enabled = false` 时也在）
- 不依赖内心层（`mind` 为 None 时照样叫醒、记“在想”，只是不进反思材料和流水账）。`enabled = false`：不认冷场、`leave` / `return` 照旧、提示词和 status 照旧；管理面板有 `lull.enabled` 开关

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
  反思（`persona` 开着时系统提示词接上 `PERSONA_SYSTEM`、材料带「你攒下的性格」）顺带给 `persona_add` / `persona_used`；**代码守的规矩**（`Persona.apply`）：截 30 字（话题 10 字）、带敏感词（胖瘦丑矮、长相身材脸、爸妈家里、成绩考试分数作业老师、几岁年纪年龄今年岁、学校班小学生初中高中、本名 QQ 微信、哭）/ 难过类的话（`DISTRESS`）/ 声称是真人的话（`claims_human`）的丢；口头禅和看法里**不许提好友名字**（人只进老梗）、
  老梗只挂在一起玩过 ≥ `grudge_min_days` 天的好友名下（OCR 错字按最像的好友归）、收着点的人不记新老梗、几乎一样的不重复加、同话题换立场；超上限只在旧条目里挤（这次新记的不挤）：先删 `hits` 最少的、再删最久没用的；换了立场算刚用过；`fade_days`（14 天）没用就淡出（启动时和每次套完）；启动时 `prepare` 按现在的规矩把旧条目再筛一遍、手写的 `since = 0` 当作现在记的
- **大脑看到**：系统提示词「你攒下的性格」（启动时算一次，放在「日子」之前，空的不写）；`arrive` 在“你惦记着”之后接“你们的老梗：…”（最多 2 条）；
  提示词多一段「脾气」（`TEMPER_RULES`：有立场、熟人互损但损事不损人、会拒绝会偷懒但卡洛的 `#` 命令和有人真难过时不偷懒、有执念、对“收着点”的人好好说话），“接对方的话往下聊”换成“接得住就接，不想接也可以吐槽一句或者敷衍两句”；底线各节一字不动
- **收着点**（身体兜底）：任何好友说了难过类的话（第 2 期的 `DISTRESS`）→ `soft_minutes`（30 分钟）内 status 写“对小明收着点（他刚说「…」）”、他来时 arrive 不带老梗，反思不给他记新老梗（下线那次反思看整次上线：这次上线里说过难过的都算）
- `memory init` 的人设模板多了「## 脾气」（毛病 / 执念 / 雷点）；已有的 `memory/profile.md` 要自己加。`memory show` 末尾打印性格档案（只读），`profile.md` 永远优先、程序不改
- 只在 live 写 `persona.json`；dry-run 只在内存里，坏文件不改名。`persona = false` = 第 2 期原样（提示词、反思提示词和材料逐字一样，不记收着点）；管理面板有 `inner.persona` 开关
- 还是太乖：旧的乖回复会把模型拉回去，按「记忆」一节把 `history.jsonl` 挪到 `memory/archive/` 再试

## 幕后（`[backstage]`，大脑模式）

设计见 `docs/superpowers/specs/2026-10-01-backstage-design.md`，计划 `docs/superpowers/plans/2026-10-01-backstage.md`。
像 Neuro-sama 和 Vedal：团子知道自己是 AI、卡洛（`[brain] owner_name`）做了她，能跟卡洛聊自己怎么运作。**默认关**（公开仓库），**还没用真 Claude 验证**（spec §7 五步，在沙盒里做）。
- **提示词**：「## 身份」整节换成「## 幕后」（`brain/backstage.py` 的 `section`）：事实（脑子 / 眼睛 / 反思的模型名取配置，身体、记忆文件、沙盒是什么）+ 三档：
  卡洛（名字一字不差）完全出戏、能聊深的、半开玩笑的存在主义；**知情好友**（friends.md 他那一节写「知道团子是 AI」，代码不解析、模型自己读）大方承认是 AI、开玩笑，不聊深；其他人照原来的「身份」规则。
  `owner_name` 为空时没有卡洛那一档。「底线」一节、`_CLAIMS_HUMAN` 硬过滤不变；陌生人看不到好友之间的聊天，所以不管旁人
- **`introspect(topic)`**：只读工具，topic = 精力（逐项加减，`inner/energy.py` 的 `energy_parts`）/ 反思（流水账最近 3 次 + 下次大概几分钟后）/ 性格（用过几次、几天后淡出）/ 日记（最近一篇，正文截 600 字）/ 眼睛（`Eyes.latest` 原文）；
  开关打开才注册（`ToolBox(backstage=True)`），不算"做了事"；对应的内心层没开就回"没开"
- **更新记录**：启动时取 `inner/backstage.json` 的 `seen..HEAD`（没有 / 不在历史里就按最近 `changelog_days` 天），`--no-merges`、只要 `feat` / `fix` / `perf`、scope 为 `console` / `viewer` 的不要，最多 `changelog_max` 条，提交标题原样接在「幕后」末尾由她自己转述；
  **只在 live 写标记**（dry-run 每次看到同一批），沙盒记在沙盒记忆目录；git 出错 / 超时 2 秒只是这一小节为空
- `enabled = false`：提示词、工具列表逐字照旧；管理面板有 `backstage.enabled` 开关

## 大脑沙盒（`[sandbox]`，`sandbox` / 管理面板「沙盒」页）

设计见 `docs/superpowers/specs/2026-09-30-brain-sandbox-design.md`，计划 `docs/superpowers/plans/2026-09-30-brain-sandbox.md`。
不开 MuMu，让**真的大脑 + 身体 + 内心层**接一个假世界：冒充好友说话、造来去、快进时间，看团子的反应和心里；调性格和提示词用。
**代码已完成，只用 fake_claude 在浏览器里核对过，还没用真 Claude 跑过**（spec「验证」四步，见下）。
- **拆出世界**（`brain/world.py`）：`cli._run_brain(cfg, run, world=None, …, on_ready, trace, *, no_emotes)` 只管组装；`World` 给设备、读聊天、身边、轮盘、说话、走路……和 `clock` / `wall`。
  真机是 `cli._game_world`（原样搬的，`world=None` 时在检查完令牌之后才建），沙盒是 `sandbox/world.py` 的 `sandbox_world`。身体、事件队列、眼睛、反思器、大脑循环、账本都用 world 的钟
- **沙盒世界**：模拟时钟 `SimClock`（只往前拨；起始时间不早于 `clock.json` / 沙盒 `days.jsonl` 最后一次下线）、中灰截图（不算黑屏）、队列读聊天、名单当身边、手写场景当眼睛（空 = 看不清，不调 Haiku）；
  说话 / 动作 / 走路 / 气泡都写进沙盒聊天记录（`sandbox/transcript.py`）；没有镜头、好友树、面板、互动请求（工具回"沙盒里没有这个"）；`look` / `look_person` 只给文字（`World.text_only`）
- **总是 live，但只写 `sandbox/memory/`**（`[sandbox] dir`，整个 `sandbox/` 不进 git）：真的 `memory/` 永远不碰（端到端测试比 sha256）；
  沙盒记忆目录和 `reply.memory_dir` 重合（相同或互相包含，比如 `dir = "."`）时，重置和启动都拒绝。`--duration` 按真实时间，快进不会提前下线。随手记、反思、日记都真的调 Claude，和真机一样花额度
- **子进程** `python -m skydango sandbox --port 19392 [--start resume|sleep|HH:MM|"YYYY-MM-DD HH:MM"]`：只给 JSON 接口（`sandbox/server.py`：`/status` `/state` 长轮询 `/op` `/brain` `/inner` `/inner/forget` `/shutdown`，本机 Host + `post_guard`）；
  操作经 `body.call` 在身体线程做（`sandbox/control.py`：冒充发言、来去、陌生人、地名、场景、新鲜事、快进、拨时间、立刻反思）；下线 = `/shutdown`，走最终反思 → 日记 → 合账本，再存 `clock.json`
- **管理面板**：子进程槽带 kind（团子 / 沙盒），**同一时间只能有一个**（共用令牌和 `.brain-claude/`），另一个在跑时拒绝并提示先停；`/sandbox/*` 转发、`/live/*` 只在团子时转。
  「沙盒」页：顶栏启动选项（接着上次 / 睡一晚 / 自定义）或沙盒时间牌 + 快进；三栏从左到右 团子（现在、身边、场景和新鲜事）/ 大脑控制台 / 聊天记录。
  聊天记录团子说的在左（樱花底）、冒充的人在右，动作旁白、事件分隔线、被拦的删除线 + 原因；
  大脑控制台（`console/static/brainlog.js`，和 viewer 的 `brain_trace.js` 同一个 `/brain` 数据）是终端排版的日志（底色同左右两栏、等宽字）：轮头（时间 · 原因 · 做了什么 · 耗时 · tokens）、收到的事件一行一条、状态 / 场景折叠、`▶` 工具调用、`↳` 返回，新的一轮在底部、自动跟到底；
  「重置记忆」只在停着时能点（用 `memory/` 覆盖沙盒记忆，`memory/archive/` 不复制）；「内心」页顶上能切 团子 / 沙盒
- **剧本**（`console/scenario.py` 格式、`console/replay.py` 录制和回放，`sandbox/scenarios/*.toml`）：沙盒启动就开始录，停止 / 再启动记成 offline / online，「另存为」写成剧本；
  回放：沙盒在跑先下线 → 按 `[start]` 重置 / 起 / 发身边 → 每步发出后等安静（`[sandbox] step_timeout` 180 秒，超时记下接着走）；回放中手动操作被拒、页面置灰；「停止回放」做完当前这步就停。
  报告 `sandbox/reports/<剧本>-<时间>.md`：每步之后的聊天记录、反思行、`expect`（只写进报告，不检查）、最后的心里、这次写的日记。示例剧本在 `docs/sandbox-scenarios/`（人名是占位的，换成自己的好友名再复制进 `sandbox/scenarios/`）
- **已知问题**（都偏小，没修）：
  - 身体正在执行 say 这类命令时点「下线」，中断会经 Future 传进 MCP 线程报一段未处理异常（`run` 原来就这样）
  - 大脑时间线每轮的开始时间是真实时间，大脑收到的消息里是沙盒时间
  - 好友一直在身边时快进超过 `visit_gap`，关系卡会多记一次"见过"
  - 回放时的操作不录；回放完接着手动玩的从 `memory = "keep"` 起录，存不成"回放 + 手动"一整个剧本
  - 额度用完时 idle 不看事件队列（大脑醒不来，事件会一直积着）：回放照样往下走、报告里那一步标"额度用完"，这段时间团子的反应其实没测到
- **要用真 Claude 验证**（spec「验证」，都没做）：
  1. 重置记忆、启动，冒充好友聊几句：聊天记录、"现在"、大脑时间线都在动
  2. 快进到 23:30：精力变困、主动开口变少（聊天记录里 blocked 或干脆不说）；额度用完时顶上写"额度用完，约 N 分钟后再试"
  3. 回放「放鸽子」（换成本机好友名、要一起玩过 3 天以上的老朋友）：报告里有别扭、说难过后撤掉；反思改了什么进聊天记录（"── 反思：… ──"）；「内心」页切沙盒看流水账和曲线
  4. 下线再"睡一晚"上线：「日子」里带上一篇日记；最后确认真的 `memory/` 没被动过

## 常用命令

```bash
python -m skydango devices                # 连接 / 截图尺寸 / 当前输入法
python -m skydango shot [--grid]          # 截图到 tmp/shot.png
python -m skydango detect                 # 读一次聊天记录面板（先在游戏里按 C），标注图 tmp/detect.png
python -m skydango say "【AI】你好"        # 发一句（输入框没开会先按 Enter）
python -m skydango chat --emotes 鞠躬,害羞  # 终端里和人设聊天，假装轮盘上有这些动作
python -m skydango console [--port 端口] [--no-browser]  # 管理面板：填密钥、改设置、检测设备、启动 / 停止团子、看实时画面；「沙盒」页不开模拟器调大脑
python -m skydango sandbox [--port 19392] [--start resume|sleep|HH:MM|"YYYY-MM-DD HH:MM"]  # 大脑沙盒子进程（一般由管理面板起；只有 JSON 接口），记忆只写 sandbox/memory/
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
python -m skydango perception appearance-eval <录像目录> [--model YOLO模型] [--embed color|模型.onnx]  # 认装扮离线标定：同一个人 / 不同人的相似度、建议的 match / changed、藏标签重放 → tmp/appearance-eval/<时间>/report.md
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
