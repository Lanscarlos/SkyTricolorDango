# CLAUDE.md — 给接手这个仓库的 Agent

SkyTricolorDango：在 MuMu 模拟器上"自己玩光遇"的 Agent。纯视觉：只截图 + 模拟输入，不读内存、不注入、不抓包。
当前能力：读聊天记录 → 大模型回复 → 在游戏里发言（可以带表情动作）；快捷动作轮盘的读取 / 编辑 / 做动作。

**先读 [docs/game-ops.md](docs/game-ops.md)**：光遇 + MuMu 的操作方式、界面坐标、实测踩过的坑都在里面。
不看它直接往游戏里发输入，很容易打错字、卡在界面里，或者覆盖掉用户的轮盘配置。

## Agent 规则

- 用中文回答。
- 每次任务改完，主动把功能分支合并进 main 并推送到远程。
- **没有用户许可，禁止在项目主目录（`D:\Lanscarlos\Develop\SkyTricolorDango`）切换到别的分支**（`git checkout` / `switch` 别的分支、`checkout -b` 都算）：
  几个会话常同时在这个仓库里干活，主目录还是用户和管理面板跑团子的地方（10-02、10-03 晚都有会话在主目录切了分支，别的会话的提交落错分支、团子差点跑上没写完的代码）。
  要开分支就用 `using-git-worktrees` 开自己的 worktree，改完合并进 main；确实要动主目录的分支，先问用户。也别在主目录留下没提交的改动就走。

## Skill：动手之前先挑一个

`.claude/skills/` 里的 skill 跟着仓库走，本地和云端（claude.ai/code）都会自动加载——插件装在本地是没用的，云端每次从仓库重新克隆。

**任何任务开始前，先看有没有对得上的 skill；有就先调用它，再动手。** 包括「先问个澄清问题」「先翻一下代码」之前——skill 会告诉你该怎么翻。用之前宣告一句「用 X skill 来做 Y」，然后照着它走；发现不合适再放弃。

| Skill | 什么时候用 |
| --- | --- |
| `frontend-design` | 新做或重做网页界面（管理面板 `console/static/`）：视觉方向、字体、排版，避免模板感 |
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
- **onnxruntime 用 GPU 版**（10-02）：`.pydeps` 里装的是 `onnxruntime-gpu==1.23.2`（CUDA 12，和 torch cu128 对上；1.30 要 CUDA 13，驱动不够），用户目录里的 CPU 版 `onnxruntime` 已卸掉。
  CUDA / cuDNN 的 DLL 借 torch 的 `lib` 目录（`vision/onnxrt.py` 建 CUDA 会话前自动 `preload_dlls`）。用户目录的 site-packages 在 `sys.path` 里排在 `.pydeps` 前面：
  **别再往用户目录装 `onnxruntime`**（`pip install -e ".[ocr]"` 会装回来、盖掉 GPU 版，装了就再 `pip uninstall onnxruntime`）；`pip check` 报 skydango 缺 onnxruntime、onnxruntime-gpu 缺 coloredlogs 都不影响
- 编译 `android/a11y/` 的无障碍客户端要 JDK（本机 `D:\Java\azul-18.0.2.1`）和 R8（含 d8）：`.pydeps/r8/r8-9.4.28.jar`（Google Maven `com.android.tools:r8`，10-04 下载、SHA1 核对过）；
  不用 Android SDK（隐藏 API 照 `android/a11y/stubs/` 编译）。编好的 jar 进 git，只改 Java 时才要重编
- `config.toml` 是本机配置（gitignore），模板是 `config.example.toml`；记忆在私有仓库，新电脑先按「记忆」一节克隆到 `private/`
- **模型按用处选**（10-05，见「模型供应商」）：默认全走 DeepSeek（`DEEPSEEK_API_KEY`；10-06 起识图也是，用能看图的 `deepseek-flash`）；Claude 令牌（`SKYDANGO_CLAUDE_TOKEN`）只有用到 Claude 的用处才要（默认只当备用：大脑 / 记忆 / 反思 / 眼睛 / 装扮描述 / 看图标注）
- 模拟器里装了 ADBKeyboard（团子输中文用），用户自己打字用 MuMu 里的搜狗输入法。**`run` 启动时自动切到 ADBKeyboard、停下时切回来**（`device/ime.py`，`[device] switch_ime`，默认开；
  切回启动时的那个，启动时已经是 ADBKeyboard（上次被强杀没切回）就找装了的搜狗，`[device] user_ime` 可以指定）；被强杀时切不回，在管理面板「设备」页「输入法」一栏手动切，
  或者 `python -m skydango ime off`（切回搜狗 / `user_ime`，都没有就 `ime reset`）。切换出错只记日志、团子照常跑（打不了中文）。10-04 真机上搜狗 ↔ ADBKeyboard 来回切正常；「启动时已经是 ADBKeyboard、按 id 里有没有 `sogou` 去找搜狗」这条路还没在真机触发过
- 在 Git Bash 里调 `adb shell` 带 `/dev/...` 路径时要 `export MSYS_NO_PATHCONV=1`，否则路径会被改写

## 代码结构

| 位置 | 内容 |
|---|---|
| `src/skydango/device/adb.py` | 截图（raw screencap）、tap/swipe、`hw_key*`（sendevent 模拟实体键盘）、ADBKeyboard 输入、`ime_shown()` |
| `src/skydango/device/mumu.py` | MuMu 原生截图（external_renderer_ipc.dll，约 9 ms/张），`AdbDevice.screenshot()` 优先用它 |
| `src/skydango/device/ime.py` | 输入法切换：`run` 启动时切到 ADBKeyboard、停下时切回（命令 `ime on` / `off` / `status`，见「环境」） |
| `src/skydango/device/a11y.py` `android/a11y/` `assets/a11y/` | 读游戏的无障碍节点（聊天行、头顶名字 / 气泡的文字 + 坐标，不用 OCR；见 game-ops §3「无障碍节点」）：`A11yReader` 推 jar、`app_process` 常驻 UiAutomation 客户端、`latest()` 取快照；jar 源码在 `android/a11y/`，`python android/a11y/build.py` 编译（要 javac + `.pydeps/r8/` 的 R8）。读聊天已接上（下面两行），**认人还没接** |
| `src/skydango/vision/a11yui.py` | 无障碍快照分类（纯计算）：`classify` → `UiView`（面板行、头顶名字标签、按列挂好说话人的气泡、输入栏、面板占位）；坐标按 1920×1080 量、按比例缩放，只认光遇包名的快照 |
| `src/skydango/chat/a11yreader.py` | 用无障碍节点读聊天（`[vision] source = "a11y"`，见「聊天面板」）：`A11yChatReader`（接口同 `ChatReader`；面板行精确对齐、面板关着读好友头顶气泡、两边读到同一句只报一次，`Message.source` = panel / bubble；`tags_in_view` / `typing` / `want_peek` 给面板管理器）、`FallbackReader`（读不到自动重启 3 次、再不行整次 run 退回 OCR，`reads_bubbles` / `describe`） |
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
| `src/skydango/vision/detect.py` `track.py` `perception.py` `weaklabel.py` | YOLO 感知层（开发中，默认关）：检测器（ONNX / ultralytics）、追踪（IoU + 低分框续命 + 速度预测 + 画面平移 `estimate_shift`）、`PerceptionWatcher`（接口同 env，多认陌生人；画面被挡时暂停计时；续命、失踪好友接回、运动方向）、弱标注，见下 |
| `src/skydango/vision/attrs.py` | 感知层第二层运行时（见「感知层第二层」）：`AttrModel`（冻住的 DINOv2 + `.npz` 线性头）、`PersonAttrs`（按轨迹裁图、投票、放行 / 撤下、点没点火）、`load_model`、`crop` |
| `src/skydango/vision/attrs_data.py` | 第二层的数据：`perception crops`（数据集 / 难例 / 图片目录 → `datasets/attrs/`）、`--writeback`（标注页确认过的写回 `datasets/sky`：补漏标的人、改点没点火、删不是人，增强图一起改）、`perception attrs-label`（Claude 初分，复用 `assist.Reviewer`） |
| `src/skydango/vision/attrs_train.py` | 第二层的训练和评估：`perception attrs-train`（提特征 + 线性头 → `models/attrs-<日期>.npz` + 报告）、`attrs-eval`、`bench --attrs` |
| `src/skydango/vision/trackeval.py` | 追踪和接回的离线评估（`perception track-eval`）：基线（升级开关全关）vs 当前配置，断开原因、假走开、确认冤枉、接回对错、运动方向 |
| `src/skydango/vision/hardcases.py` `compare.py` `augment.py` | 感知层一期工具：运行时收集难例、离线对比 YOLO 和整图 OCR（`perception compare`）、训练集增强（`perception augment`） |
| `src/skydango/vision/icons_map.py` `icon_eval.py` | 地图交互图标（见「认交互图标」）：`icons_map.py` 归属 `owner_of`（人 / 先祖 / 物件 / 地图）、叫法 `map_label`、多数表决、裁图、`classify_template` 模板匹配、DINOv2 最近邻 `IconGallery`、运行时 `MapIcons`、收件箱整图火焰预标 `flame_rings`；`icon_eval.py` 离线评估 `perception icon-eval` 和截模板 `icon-cut` |
| `src/skydango/vision/inbox.py` | 难例收件箱（见「难例收件箱」）：`collect`（`runs/<…>/hard` → `datasets/inbox/<运行>/raw`，记 `_index.jsonl`）、`process`（YOLO + 外形头先筛，一致的整帧进待过目、拿不准的裁图交外形页人工判，不调 Claude）、`status`、去重、整次运行分验证 / 训练、通过的帧写进 `datasets/sky` |
| `src/skydango/vision/retrain.py` | 一键重训和对比（`perception retrain`）：训 YOLO + 外形头 → 新旧模型在验证集上回放 + ultralytics val → `tmp/retrain/<时间>/report.md` + `result.json`，失败抛 `RetrainFailed`、不改配置 |
| `src/skydango/vision/assist.py` | Claude 辅助标注（`perception label --assist`）：挑帧、人物候选框、`claude -p` 核对（分批并发、缓存、额度用完可续跑）、合并成标注 + 预览 + 待核对清单；`Protocol` 让物品模式复用 |
| `src/skydango/vision/objlabel.py` | 物品模式（`perception label <数据集> --objects`）：提示词、解析、写回（人物行不动、物品行整体替换、人改先祖）、预览、清单 |
| `src/skydango/vision/sweep.py` | 感知层二期：环绕扫描的纯计算（方位角、8 方位、多帧合并、转圈认团子、远近分档） |
| `src/skydango/vision/places.py` `unknownnames.py` `gesture.py` | 感知层三期：认地图（参考截图匹配）、没认出的名字清单、别人对团子做的动作（研究性质：切片段、评估、ONNX 接口） |
| `src/skydango/vision/gesture_label.py` | 动作片段的 Claude 初分（`perception gesture-label`）：16 帧拼成 4×4 一张图、提示词、解析，结果写进片段目录的 `claude.json`（复用 `assist.Reviewer`） |
| `src/skydango/vision/gesture_train.py` | 动作模型训练（`perception gesture-train`）：读确认过的片段、按录像切训练 / 验证集（`_split.json`）、DINOv2-small 冻住提特征（缓存 `_features/`）+ 时序头、导出 ONNX、报告；torch 延迟导入 |
| `src/skydango/console/labeling.py` | 「标注」页的后端（`/api/gesture/*`）：片段状态、取帧、确认 / 改类别 / 丢弃 = 挪文件夹、撤销，记在 `_labels.jsonl` |
| `src/skydango/console/emotenames.py` | 「标注 → 动作名」的后端（`/api/emotes/*`）：按剪影把图标库 `emotes/*.png` 配给 `emotes/scan/` 的扫描图标、起名（复制）/ 改名 / 清除（挪进 `_removed/`）、名字检查、提示配置里引用旧名字的地方 |
| `src/skydango/vision/embed.py` `onnxrt.py` | 特征模型的公共部分：`OnnxEmbedder`、`unit` / `cosine`（认地图、认装扮共用）；`onnxrt.py` 按 device（cuda / dml / cpu）选 onnxruntime 后端、建会话（YOLO、特征模型、动作模型共用） |
| `src/skydango/vision/appearance.py` | 认装扮（见「认装扮」）：好样本裁图、内置颜色特征 `ColorEmbedder`、外观记忆簿 `AppearanceBook`（好友 / 陌生人编号 / 团子自己各一个底库、`assign_friends`、`looks_like_dango`、换装判定）、攒训练数据 `CropSaver` |
| `src/skydango/vision/gallery.py` | 认装扮的多样本底库（纯数据）：`Sample`（颜色 + DINOv2 两个特征）、`Gallery`（去重、上限、钉住、`best` 取最像一张） |
| `src/skydango/vision/catalog.py` | 装扮图鉴第 1 期（见「装扮图鉴」）：运行时把近处的人清楚的整身裁图存进 `catalog/inbox/`（门槛、每人留最好几张、写盘、索引）、离线工具的总览拼图 |
| `src/skydango/vision/wardrobe.py` | 装扮描述器：排队、优先级、每小时额度，用 `[models.wardrobe]` 把人物裁图写成一句话 |
| `src/skydango/vision/appearance_eval.py` | 认装扮的离线标定（`perception appearance-eval`）：收集轨迹特征、相似度分布、建议门槛、藏标签重放、报告 |
| `src/skydango/vision/halo.py` `halo_eval.py` | 按 Q 喊一声的呼唤光圈：头顶区域、按键后连拍认团子（`HaloWatch`，纯计算）；离线标定 `perception halo-eval` |
| `src/skydango/brain/calling.py` | 按 Q 喊一声：`CallResult`、给大脑 / 事件 / status 的文字、`wait_result`（在调用方线程里等呼喊窗口结束）、`call_available` |
| `src/skydango/vision/viewer.py` | 团子的 HTTP 接口（`run` 总是开，见「团子的接口」）：画面 + 识别框 + 状态放在同一份快照里、`/status`（带 `run` 节）、大脑、聊天、手动控制、内心、退出；自己没有网页，框和中文标签由管理面板的 `console/static/stage.js` 画 |
| `src/skydango/vision/panels.py` `game/panels.py` `assets/panels/` | 面板识别：特征卡快看 + OCR 细读 + 通用兜底认出开着哪些面板（`vision`）；按卡片关面板、点按钮（`game`）；六张特征卡（见「面板识别」） |
| `src/skydango/game/social.py` | 社交互动：好友头顶圆圈里出现牵手 / 拥抱 / 击掌图标时点圆圈接受（请求由 env 的后台扫描发现），图标模板在 `assets/social/` |
| `src/skydango/vision/lighting.py` | 点亮陌生人的纯计算（`FlameWatch`）：每处火焰一条线索、认出不动的假火焰（灯笼）、出请求的那条、举蜡烛后火焰怎么没的 → 点亮 / 走开 / 接着等（spec `2026-10-03-light-flame-vanish-design.md`；10-02 晚 6 次存图的回放在 `tests/test_light_replay.py`） |
| `src/skydango/vision/candle.py` | 火焰：`find_flames` / `find_flame` 在调用方给的范围（团子周围）里找火焰，`black()` 量人物框有多黑（判点亮用），`white_ring` 分孤儿圆圈是举蜡烛请求（有白圈）还是圆盘；火焰圆盘**绝不点** |
| `src/skydango/game/friendtree.py` | 点人物打开好友树面板、截图、关掉（大脑的 `check_friend`，默认关，未在真机验证） |
| `src/skydango/brain/` | 统管大脑：`body.py` 身体（事件、命令队列、工具的护栏）、`loop.py` 大脑循环、`session.py` 常驻 Claude Code、`toolloop.py` OpenAI 兼容的大脑（`ToolLoopBrain`，function-calling 循环）、`sessions.py` 按供应商建大脑会话（`make_session`）、`claude.py` 旧名字的转引（进程、命令都搬到了 `models/claude_code.py`）、`llm_tools.py` OpenAI 兼容大脑的工具 schema（唯一来源）、`trace.py` 大脑时间线（给管理面板）、`manual.py` 手动控制、`mcp_server.py` + `tools.py` 工具、`eyes.py` 眼睛、`camera.py` 视角、`locomotion.py` 小步走（`move`）、`skills.py` 技能层（见「统管大脑」）、`attention.py` 空闲注意力 / `search.py` 有意识地找 / `find.py` 找人技能 / `peek.py` 换角度、`occasion.py` 场合（见「看场合主动开口」）、`reflex.py` 反射（见「身体反射」） |
| `src/skydango/inner/` | 内心层（见「内心层」）：`ledger.py` 关系卡和这次上线（纯数据、拼文字）、`store.py` 读写 `memory/inner/`、`days.py`「日子」一节、`backfill.py` 从 history 回填、`open_ledger` / `show_lines`；第 2 期 `energy.py` 精力、`mind.py` 心情 / 别扭 / 心愿、`effects.py` 倍数、`reflect.py` 反思、`finish_reflection`；第 3 期 `persona.py` 性格档案（口头禅 / 老梗 / 看法）；内心页 `log.py` 流水账（`MindLog` + 反思前后 `diff`）、`api.py` `/inner` 接口的解析；`lull.py` 冷场追踪（见「冷场时的心理活动」） |
| `src/skydango/models/` | 模型供应商和按用处选模型（见「模型供应商」）：`config.py` 解析 `[providers.*]` / `[models.*]`、旧字段换算、校验；`claude_code.py`（`claude -p`）/ `openai_compat.py`（DeepSeek 等）两种接入方式；`errors.py` `ModelError`（`down` = limit / auth）；`gate.py` 按供应商的闸；`registry.py` `Registry` / `GatedCall`（主 → 备）。**代码只从这里拿模型** |
| `src/skydango/console/` | 管理面板（`console`）：设置清单和 `console.toml` / `secrets.toml` 读写（`settings.py` `tomlfile.py`）、「模型」页后端（`models_view.py`）、团子子进程起停（`runner.py`、子进程侧看门狗 `watchdog.py`）、启动预检 / 测试按钮 / 设备检测（`preflight.py` `probes.py` `devicecheck.py`）、HTTP 服务和转发（`server.py`）、接管终端起的团子时读它的 agent.log（`logtail.py`）、内心页数据（`inner_view.py`：读 `memory/inner/`、在跑时合并实时、删性格条目）、沙盒（`sandbox_view.py` 重置记忆 / 起始时间下限、`scenario.py` 剧本格式、`replay.py` 录制回放报告）、报告读取（`reports.py`）、整理 / 重训的任务槽（`jobs.py`）、整帧页后端（`frames.py`）、重训报告和换上 / 回退（`retrain_view.py`）、页面 `static/`（`console.html` + `console.css` + `common.js` / `markdown.js` + 共用的 `brainlog.js`（大脑控制台）/ `chatlog.js`（聊天行）+ 每页一个 js（真机页的手动控制另在 `livectl.js`、标注页的整帧页 `frames.js`、重训区 `retrain.js`），左侧栏 + 八页：沙盒（默认）/ 真机团子（照沙盒三栏：团子 / 画面 + 大脑 / 聊天记录，加日志抽屉）/ 内心 / 剧本和报告 / 标注 / 模型 / 设置 / 设备；见「管理面板」「大脑沙盒」） |
| `src/skydango/brain/backstage.py` | 幕后（见「幕后」）：拼「幕后」一节、取"卡洛上次以来改了你什么"（git 提交）、读写 `inner/backstage.json` 标记 |
| `src/skydango/brain/addressee.py` | 分清好友在跟谁说话（见「分清在跟谁说话」）：`Addressee.judge` 每句判一次（`Verdict`：跟你说 / 跟别人说 / 说给大家 / 拿不准）、`said`、`parse_aliases`（friends.md 的 `- 叫法：`）、`legacy_addressed`（`enabled = false` 时反射用的旧规则）；纯规则、不调模型 |
| `src/skydango/chat/addressee_eval.py` | 上面规则的离线评估：从 `agent.log` 取多人聊天、Claude 初标、规则重放、`review.md` 给人核对、`report.md` 三条门槛（`addressee label` / `eval`） |
| `src/skydango/brain/world.py` `src/skydango/sandbox/` | 大脑沙盒（见「大脑沙盒」）：`World` / `BrainParts`（`_run_brain` 拆出的"接世界的东西"）；聊天记录 `brain/transcript.py`（沙盒和真机共用，带长轮询 `wait_since` 和事件分隔线 `event_line`）；`sandbox/` 模拟时钟 `clock.py`、沙盒世界 `world.py`、操作和状态 `control.py`、JSON 接口 `server.py` |
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

**去 AI 味**（10-05，看了 09-27 ~ 10-03 的 443 轮真机 history）：句句「啦 / 呀」结尾、一轮两句里第二句复读第一句、贺卡式安慰、给好友讲身体怎么运作、每次都「卡洛来啦」。
- 提示词「说话」一节：一轮一般只说一句、语气有起伏、做不到的随口推掉别讲运作、别说贺卡话、别每次都打招呼；第一句 say 的工具结果附 `tools.SAY_ENOUGH` 提醒（还能说第二句时）
- 同一轮接着说的句子（上一句记下后 `FOLLOWUP_GAP` 20 秒内、中间没人说话）history 里记成「（没人接话，你接着上一句又说）」，不再冒充「主动开口」
- 私有仓库：`profile.md` 的说话习惯改成描述 + 真聊天例句（像她的 / 不像她的）；`tools/curate_history.py` 挪走旧的复读、招呼、点名的 AI 味回合（先备份进 `archive/`，默认只列出、`--apply` 才改），**在本机推完新记录、团子下线后跑**
`memory init` 用配置生成 profile / friends，`memory show` 查看，`memory update` 立刻整理。
随手记和整理（`NotesKeeper`）走 `[models.memory]`（默认 DeepSeek、备用 Claude sonnet，见「模型供应商」）；`run --no-brain` 的普通 Agent 也一样。

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
| `a11y.jsonl` | 无障碍读法的原始快照（每变一次 / 每秒心跳一行，30 分钟几 MB）：读聊天出问题时拿它回放（`A11yChatReader` 逐行喂 `parse_line`） |
| `replies.jsonl` | 每轮：收到的消息、回复（`null` = 不回复或被过滤）、是否真的发出 |
| `frames/*.jpg` | 读到新消息时截的聊天面板（红框标新消息），`run.save_frames = false` 关掉 |
| `config.json` | 本次实际生效的配置（含 `--live` / `--echo` 覆盖） |
| `hard/*.jpg`、`hard.jsonl` | YOLO 感知层可能认错的画面（难例）和原因、检测框（`[perception] hardcases`，每次最多 200 张）；下线时自动收进 `datasets/inbox`（`[inbox]`，runs/ 轮换不再丢；也能手动 `perception inbox collect`）；旧办法 `perception label runs --from-runs` 仍可用 |
| `look_person/` | `look_person` 裁给大脑的图（`<时间>-<名字>-crop.jpg`）和画了裁剪范围的整帧（`-frame.jpg`），事后核对它看到的是谁 |
| `spin/<时间>/` | 主人 `#spin` 转一圈的截图：转前 / 转完 / 每帧（文件名带按住后第几秒）和 `summary.json` |
| `light/<时间>/` | 每次点亮陌生人（`[social]` light）：出请求那一刻 + 举起后每 0.5 秒的截图（每次最多 30 张、每次 run 最多 50 次；所有火焰候选画青圈，他那团画粗）和 `summary.json`（线索、结局、举起时和最后的 `black()`、火焰最后看到的时间和位置、最后在不在边上 `away`） |
| `unknown_names/` | YOLO 感知层读得清楚、但不在 friends.md 里的名字（`names.jsonl` + 每个名字一张裁剪图）；`perception unknown-names` 汇总，**只列出，不自动写 friends.md** |
| `enroll/` | 启动时转一圈登记团子取到的团子裁图（`<序号>.jpg`），事后核对登记的是不是团子各个角度 |
| `icons/` | 认交互图标时认不出 / 低分的图标裁图 + `icons.jsonl`（`[icons] save`，每条轨迹 2 秒一张、每次最多 200 张）；难例原因 `icon_unknown` |
| `appearance/` | 认装扮攒的训练数据：`crops/<身份>/*.jpg`（好友名，或 `t<轨迹>`）+ `appearance.jsonl`（`[appearance] save`，每条轨迹 2 秒一张、每次最多 2000 张） |
| `brain.jsonl` | 大脑每一轮：subtype、轮数、用量、total_cost_usd（订阅不按它收费，参考）、用了哪些工具、最后说了什么（只有大脑模式，`--no-brain` 没有）；`brain/` 下是 Claude Code 的工作目录（mcp.json、prompt.md） |

## 聊天面板（`[panel]`，`chat/panel.py`）

设计见 `docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md`。面板开着时方向键转视角、缩放、长按 Z、Q 都没反应，还挡住画面左边三分之一，
所以由 `PanelManager` 统一开关（一次 run 只有一个，cli 建好传给身体 / Agent、镜头、轮盘、互动、好友树、技能）：
- `mode = "always"`：一直开着，关久了自动重开 —— 以前的做法（沙盒强制用它）
- `mode = "auto"`（**默认**，2026-09-30 真机验收后改的，结果见 game-ops §3「按需打开聊天面板」）：平时关着，每 30 秒按 C 看一眼；来了好友 / 有人走近 / 好友头顶冒出"正在输入"气泡 / 大脑调 `chat_log` 时提前看；
  读到新消息就一直开着，安静 45 秒再关；团子说话前先开面板。**老测试要常开的在辅助函数里写明 `mode = "always"`**
- **新代码要关面板一律 `with panel.borrow("谁"):`**（点屏幕会顺带关面板的用 `close=False`），不许自己按 C；
  嵌套时最外层归还才恢复，闲着时归还不重开
- **读聊天走无障碍节点**（`[vision] source = "a11y"`，默认；要 `[vision] mode = "log"`（10-04 起代码默认就是 log，之前是 bubble）；`"ocr"` = 原来截图 OCR 的读法，逐字照旧；设计 `docs/superpowers/specs/2026-10-04-a11y-chat-reader-design.md`，10-04 晚 spec §8 六步真机走通：画面里关着面板聊、画面外 15 秒内看一眼接住、抢连接退回 OCR、杀客户端自动重启、退出不留进程）：
  面板开着读面板行（精确对齐、没有错字），面板关着读好友头顶的气泡（说话人 = 名字标签，只有点点 = 在打字）；挂不上名字的原文气泡不报、叫面板看一眼（`want_peek`）。
  读不到自动重启（最多 3 次），3 次都失败或 30 秒没快照就整次 run 退回 OCR（status「读聊天：OCR（无障碍读不到：…）」）；同一时间只能有一个无障碍连接（`a11y` 命令和 `run` 同时开，后起的退回 OCR）
- **「聊着（面板关着）」**（`talking`，只在 `auto` + 无障碍读法时有）：新消息是从头顶气泡读到的、或者说话的人都在画面里（名字标签 `TAG_RECENT` 3 秒内看到过；面板开着时读到的是面板行）→ 不开面板、开着就关掉，
  每 `chat_peek`（15 秒）看一眼面板接住画面外的人，安静 `quiet_close` 秒回闲着；说话的人不在画面里才进聊天中、开着面板；团子说话前跟谁聊的人 3 秒内都看到过就不开面板，闲着时主动开口、3 秒内看到过好友的名字标签也不开（进聊着、跟画面里的好友聊），否则照旧开；中途退回 OCR 立刻改成聊天中、开面板。`always` 模式不受影响。
  10-04 晚真机修过的坑见 game-ops §3「无障碍节点」（关面板动画 3.5 秒内看到面板不算玩家开的、看一眼至少开 1 秒、面板行按位置排、刚出现时陆续加载的行不报）
- OCR 读法下聊天内容只从面板读；YOLO 的 `typing` 气泡只用来判断"该去看了"（v7 起能认 `typing`，09-30 晚真机触发过）。无障碍读法下身体不拿 YOLO 气泡叫面板，谁在打字直接写进 status（"在打字：小明"）

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
点亮陌生人（只有大脑模式找，普通 Agent 不管 light 请求）：团子框周围（左右 `light_area_x` = 1.4、往上 `light_area_up` = 0.5 倍框高）连续 `light_after`（1.5）秒冒着火焰（`vision/candle.py` 的 `find_flames`，每处火焰一条线索，`vision/lighting.py`；**不管 YOLO 认没认出这个人**；认出名字的好友标签下、聊天面板下、`bonfire` 框里的不算；原地待 4 秒、从没到过 `disk_sure` 的不动火焰（灯笼）不算；团子框丢了沿用最后一个，最多 3 秒，再久用这个面板状态下记住的高分团子框），身体按 3 号键举蜡烛（**绝不点那个圆盘**：会跟着人走）；
**主要看火焰怎么消失**（10-03，spec `2026-10-03-light-flame-vanish-design.md`）：举满 `lit_min`（1 秒）后他那团火焰连着 2 次扫描、`lit_vanish`（0.8 秒）没看到——在范围边上 / 变小了再没的算走了、放下不鞠躬；原地没的算点亮（下面没人、`black()` 量不准也算），只有下面的人量得到、还黑着时接着等；同一个人连续两次变亮可以提前判；
**10-04 按 10-03 晚真机复盘补的**：举起后一次都没看到他的火焰（举之前就没了）不判点亮，满 `lit_min + lit_vanish` 放下、不鞠躬；举着时分数 `LIT_WEAK`（0.6）~ `disk_min_score` 的弱火焰和好友名字标签下面的火焰作为 `extra` 只给他那团用（认到他之前只认离他 `LIT_JUMP` 以内的弱火焰，证明举起后还看到过他；认到之后只认标签下面的，弱的不续命）；认到他之后他那团只配离预测位置 `LIT_JUMP`（0.25 倍框高）以内的火焰（举起后第一次还按 `light_jump`：请求到举起常隔 1~2 秒）；"还黑着"按比他（同一个人）量到过第二黑的时候降没降够 `lit_drop` 判，最多等 `LIT_DARK_WAIT`（2.5 秒）就放下、不鞠躬；`summary.json` 多了 `seen` / `darkest`；10-03 晚 8 次存图加进了 `tests/test_light_replay.py`（14 次全对）；
点亮后过 `bow_delay`（0.5 秒）鞠躬（顺带放下蜡烛；不等挂着的互动请求，别的反射照旧等）；`light_timeout` 秒没结果放下；没点亮冷却 `light_cooldown`（60 秒），判走开的冷却里火焰连着冒满 3 秒（`GONE_RETRY`）可以再举一次（只一次；点亮了冷却作废）；举着蜡烛 / light 请求挂着还没举（最多 6 秒）/ 身边火焰刚冒出来时聊天面板先别动（`PanelManager.hold_still`，不看一眼、不关、不重开：开关面板画面横移 300~400 px 会跟丢火焰，10-04 19:51:45 因此误判走开）；
鞠躬没做完不再举第二次、举之前先查 `reflex.min_gap`、身体替大脑开着输入框时不举；放下前先关替大脑开的框（按数字键会关掉它）；举着时接受了别的互动（点圆圈会放下蜡烛）或黑过屏（切场景、状态不明）就不再按 3 放下。接受别人点火后也鞠躬。每次存图到 `runs/<…>/light/`。
大脑能用 `set_request_policy("stranger", "light", false)` 关掉；10-02 / 10-03 晚真机测过（10-03 晚举了 8 次：对 3、错 2、没依据 3），10-04 按复盘修了；**还没有真的"走开"样本**（spec `docs/superpowers/specs/2026-10-01-light-flame-around-self-design.md` §11；数字和录像核对见 game-ops §6）。篝火「点燃」图标也是火焰圆圈，要等 YOLO 学会 `bonfire` 才排除得掉（之前团子站篝火旁可能误举一次）

## YOLO 感知层（`[perception]`，开发中）

设计和 GPU 机器上的操作步骤见 `docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`（§13）。
**分三期实现**（总表见总纲 §15）：一期 `…-perception-phase1-design.md`（替换现有识别、上线可用）、二期 `…-phase2-design.md`（环绕扫描、认说话人等）、三期 `…-phase3-design.md`（认地图、跟随等）。实现某一期先读总纲 + 那一期的文档。
- YOLO 做视觉第一道关卡：每帧检测 `player` / `player_unlit` / `name_tag` / `social_ring` / `self`，名字标签只裁小图跑 OCR 识别，身份跟着名字走
- 没点火的陌生人是黑影，单独一类 `player_unlit`，也等 `stranger_after` 才算陌生人（10-03 起，好友偶尔被认成黑影一两帧），挂过名字标签的轨迹不算；点过火的陌生人外观和好友一样，靠名字标签分：有标签且对得上 friends.md 是好友，一直没标签、离得不远的是陌生人（身体发 `stranger` 事件）
- **同一个人两个框**（10-03，`merge_people`）：检测器按类别各自 NMS，同一个人常同时留下 `player` 和 `player_unlit`（IoU ≥ `MERGE_IOU` 0.7）：合成一个、留分数高的。
  真黑影也常是 `player` 分数更高（candle 录像 p0.61 / u0.32），所以不在这里定点没点火：被压掉的黑影框记在留下的轨迹上（`data["unlit_at"]`，点亮陌生人用）；`_people_boxes` 只用框，不受影响
- `enabled = true` 时替换 env 的定时整图 OCR，接口一样，身体 / 社交 / 眼睛不用改；关掉就退回原来的
- **画面被挡时暂停计时**：黑屏、转镜头、开好友树、换轮盘、接互动时身体 / Agent 调 `env.held(原因)`；玩家自己开全屏界面靠"集体消失"规则兜底（≥ 2 人同时不见 + 画面大变）。EnvWatcher 是空实现
- **团子自己**（10-03，`_mark_dango`）：YOLO 在团子身上常常只出 `player`（`self` 只有 0.3 左右或干脆没有），团子在屏幕上的位置随聊天面板开 / 关差约 400 px；
  按面板开关分别记住最近的高分 `self` 框（`DANGO_MEMORY` 30 秒），落在那里（`same_body`：IoU ≥ 0.45）的人物轨迹打 `data["dango"]`、跟着轨迹走：不判陌生人、不挂名字标签、不算没挂名字的人（自动喊一声）。
  这一帧有 `self` 框而它不在上面、或者没 `self` 框时横着离开记住的位置超过一个团子框宽就摘掉；贴在团子身前、框把团子包住的陌生人不算（IoU 低）。
  **`self` 框的位置连续性**（10-04，10-03 晚团子和番茄并排坐时 YOLO 偶尔把高分 `self` 打在番茄身上，团子被当场摘掉、记住的位置被覆盖，图鉴"陌生人"105 张里约 70 张是团子）：
  `self` 框要和团子连得上（`near_dango`：同一个身体或中心横着不到 0.75 倍框宽；参考 = 标着团子的轨迹、1 秒内的 `self` 轨迹、记住的位置），连不上的当成别人（`demote_selfs`：改成 `player`，和那人自己的 player 框重合就去掉），
  在那儿待满 `SELF_JUMP_HOLD`（1 秒，中间断开不超过 `SELF_JUMP_GAP` 0.6 秒）才信；`one_self` 先留够分又连得上的；低分 `self` 只在团子附近提升（`promote_weak_self(near=)`）；
  聊天面板开关后 `SELF_PANEL_SLIDE`（2.5 秒，按 C 时身体先通知）的横移动画里不查、也不按"离开记住的位置"摘；团子轨迹的连续性拿上一帧的框原样和按画面平移挪过的各比一次（团子是镜头支点，转镜头时不跟背景走）。
  7 段录像上判陌生人的轨迹：5 段逐条一样，gesture-bow-1 / walkaway 各少一条被误判的好友（轨迹没再断开），没有谁被标成团子
- **难例**：运行时把可能认错的画面存进 `runs/<…>/hard/`（旁路整图 OCR 核对、低置信度、闪烁、黑影来回变），下一轮 `perception label --from-runs --model` 预标注后只需修正
- **二期（代码已完成，见 `…-phase2-design.md`）**：`look_around` 在打开感知层时改成连续转一圈（`Camera.spin`）交给 YOLO 汇总"哪个方向有谁"，不叫眼睛；
  转圈时一直在中间不动的人就是团子（`self_box` 代替 `self_roi`，`perception label --spin` 自动补 `self` 框）；眼睛拿 YOLO 认出的好友 / 陌生人 / 团子位置（`scene_note`）；
  新类别 `typing`（头顶气泡）→ 陌生人的消息加注"说话的可能是右边远处那个陌生人"（恰好一个候选才加）；远近按团子框高分档，新事件 `approach`（有人朝团子走过来，`approach_strangers` 可关）
- **三期（代码已完成，见 `…-phase3-design.md`）**：远处的小人（框高 < `far_height`）没挂上名字标签时在它头顶裁一块再检测一次（`far_crops`）；
  认地图（`[places]`，要配合 `[perception]`）：`places/<地名>/*.jpg` 图库（不进 git）+ 图像特征模型，每 30 秒 / 画面大变后认一次，写进提示词"看起来在：…"；
  读得清楚但不在好友名单里的名字记进 `runs/<…>/unknown_names/`；别人对团子挥手 / 鞠躬 / 欢呼 / 害羞（`[gesture]` 四个动作 wave / bow / cheer / shy + none，研究性质、默认关）→ 身体发 `gesture` 事件，大脑可以用 `emote` 回礼。
  数据和训练（设计 `docs/superpowers/specs/2026-10-01-gesture-labeling-training-design.md`）：`perception clips` 切片段 →（`perception gesture-label` Claude 初分：10-02 实测认不出，`--blind` 不看录像名时挥手录像一段挥手都没判出来，跳过）管理面板「标注」页人工标 →
  `perception gesture-train`（DINOv2-small 冻住 + 时序头，导出 `models/gesture-<日期>.onnx`、不覆盖 `gesture.onnx`，报告 `tmp/gesture-train/<时间>/report.md`）→ `perception gesture-eval`（有 `_split.json` 时只评验证集）；
  **10-02 标了第一批（挥手 86 / 鞠躬 54 / 都不是 124，各只有一两段录像）、试训 3 类没达标：换了录像就认不出，缺多场景素材，欢呼 / 害羞还没录**（每类至少 20 段才训练；进度见 `docs/progress/2026-10-02-tonight.md` 末尾）；
  跟随只做了第 1 步：提示词里说想跟谁走就请他牵手（视觉伺服 `follow` 要等 `move` 工具接好、标定）
- **物品识别（代码已完成，见 `2026-09-29-object-recognition-design.md`；还没有数据和模型）**：类别末尾追加 `bench` 座位 / `bonfire` 篝火 / `instrument` 乐器 / `spirit` 先祖（编号 6~9，旧编号不变；用 v4 时自然为空）。
  `objects()` 给出方位和远近（按框底边，`object_near` / `object_far` **未标定**，连续 `object_min_hits` 帧才算）；状态里"画面里的东西：座位（左边·近）"、眼睛的位置说明、网页"附近的东西"都有；
  只认出来告诉大脑，**不会走过去坐下**（F 期）。先祖单独成类后不再算陌生人。数据：`perception label datasets/sky --objects` 给已标好人的数据集补标（Claude 判成先祖的人物框自动改、列进清单；重跑跳过做过的帧和你改过的帧，`--recheck` 才重核；增强图不核对，augment 放在最后）；
  头顶气泡 `typing` 也在物品模式里一起标（`objlabel.LABEL_CLASSES` = 四类物品 + typing；typing **不进** `OBJECT_NAMES`，`objects()` 不报气泡；提示词版本 3）；
  `--only <通配>`（fnmatch 按图片文件名，带不带 `.jpg` 都行，可多次 / 逗号分隔）只处理匹配的帧，别的帧不核对、不写回、不进清单；
  烛火、光之翼等收集品不做（刷资源）。上线门槛和操作顺序见训练进度文档
- 普通模式主人命令 `#spin [圈数]`：转一圈、截图存 `runs/<…>/spin/<时间>/`，打开感知层时回复带扫描结果；大脑模式不加
- **本机在用 `models/sky-yolo-v10.pt`**（2026-10-04，标注页核对过的人物写回 datasets/sky 后重训，回放召回 62% → 75%；之前是 v7（09-30，能认头顶气泡 `typing`），v8 / v9 是座位试训、没上线；v11 黑影涨了但精确率和先祖掉了，没上线）。感知层的阈值是照 v7 定的，v10 分数整体偏高，留意陌生人 / 走近事件会不会变多，进度、数据、标注规则、各版对比和待办见 `docs/progress/2026-09-28-yolo-training.md`；
  **标气泡别只靠 Claude**：它标不出好友名字下方叠着的文字气泡，要 OCR 兜底（做法见进度文档 09-30 一节）；
  感知层已在真机 `run` 里用过多次（好友走远、标签淡掉后轨迹断开被判成陌生人，下面「追踪和接回」修的就是它）；**下面这些阈值还没在真机标定**（`[spin] seconds_per_turn` / `hfov`、`near` / `far` / `self_height`、`approach_grow`、`typing_window`）；`models/`、`datasets/` 不进 git
- **追踪和接回**（spec `docs/superpowers/specs/2026-10-01-tracking-relink-motion-design.md`，计划 `docs/superpowers/plans/2026-10-01-tracking-relink-motion.md`；**代码已完成；`track-eval` 在 7 段录像上跑过（walkaway 那段接回对 2 错 0）、10-02 晚起真机上见过接回生效，阈值还没按结果调**，spec §8 真机验证四步没逐条走）：
  追踪器两段匹配（`low_conf ~ conf` 的低分框只续旧轨迹、不开新的；`track_low` 时检测器按 `low_conf` 出框）+ 速度预测 + 中心距离兜底（`track_center_gate`，永远排在 IoU 候选之后）+ 画面平移补偿（整张 1/8 灰度缩略图、面板开着时遮掉左三分之一，`cv2.phaseCorrelate`）；
  **续命**（`sticky_names`，按 Q 那期先实现了 `_keep_named`，见「按 Q 喊一声」）：挂过名字的轨迹没断就一直刷新 `last_seen`，断了才开始算 `keep`；
  **失踪好友接回**（`relink`）：挂着名字的轨迹被删 → 失踪记录，`keep` 秒内在预测位置附近冒出来的没名字的人接成 `maybe`（`maybe_by = "relink"`，"像小明"，后果同认装扮的 maybe：不判陌生人、刷新在场、不发 arrive / return），有歧义（一人对两条记录 / 两人一样近）不接，标签亮出来名字说了算，认装扮不拿外观否掉它；
  **运动方向**（`motion`）：`motion_hist` 存补偿过平移的中心 x，`motion_of` 出 走近 / 走远 / 往左走 / 往右走 / 站着（防抖 `motion_hold`），`Person.motion`、status"小明（左边·中，正在走远）"、网页悬停；只给数据，track / 注意力 / 冷场还没用它；
  身体动镜头调 `env.camera_moved(at, kind)`（turn / zoom / move / spin；缩放、走路、转圈后 `[track] settle` 秒内不攒走近 / 运动历史、速度清零；注意力自己的 nudge 和 `move` 只通知、不设 `_camera_moved_at`）；
  护栏（最终评审后加的）：低分框阶段只认 IoU、只靠低分框续着的轨迹最多续命 5 秒（`LOW_ONLY_MAX`）；跟着镜头一起动的人（配上的是不带平移的预测框）速度和运动方向不减背景平移（`Track.drift`）；
  接回的"像他"30 秒（6 个 `keep`）没被名字标签证实就摘、也不报走近；身体在 `held("camera")` 里转了镜头，恢复后第一帧拿暂停前的缩略图估一次平移（门槛 0.4），估不出就作废所有轨迹的位置、清掉失踪记录；
  开关全关逐字是原来的行为（`camera_moved` 的缩放 / 走路静默期不受开关管）。定阈值：`perception track-eval <录像目录> [--fps 6.5]` → `tmp/track-eval/<时间>/report.md`（基线 vs 当前配置；目标接回证实错 = 0、确认冤枉明显下降）
- **开关聊天面板不断轨迹**（10-03，计划 `docs/superpowers/plans/2026-10-03-panel-toggle-tracking.md`，录像分析见 `docs/progress/2026-10-03-plan.md` 5b；录像里面板开关是 2 秒的镜头横移动画、有视差）：
  A `panel_people`（默认开）：面板开着时只丢面板区域里的名字标签 / 气泡 / 圆圈（`PANEL_DROP`），人物框留着，面板后面没挂过标签的点过火的人先不判陌生人（`_under_panel`，黑影照常判；面板关了从关的那一刻起再等 `stranger_after`），面板里的 `self` 框当普通人（开面板时团子在右边，那是误检）、面板后面的人不做远处裁图；
  B 平移估计（`_pan_step`）用整张缩略图、只在面板开着时遮左三分之一，不再遮人物框（受 `track_pan` 管）；C `panel_settle`（2.5 秒，0 = 关）：面板标志翻转 / 按 C（`PanelManager.on_press` → `camera_moved(at, "panel")`）后这段时间速度清零、不攒走近 / 运动历史。
  `panel_people = false`、`panel_settle = 0`、`track_pan = false` 逐字照旧；**只在合成画面里测过，录像回放测试和 track-eval 前后对比还没跑**（要本机的 `tmp/record/panel-toggle-1003-*`，见 5b 末尾「还没做」）
- **感知层第二层（`[attrs]`，代码默认关；本机 10-04 起开着：`models/attrs-20261004d-mask.npz`，上线门槛 1~3 过了，门槛 4（真机效果）边开边验、还没定论）**：设计 `docs/superpowers/specs/2026-10-02-perception-attrs-design.md`，计划 `docs/superpowers/plans/2026-10-02-perception-attrs.md`。
  **裁图框外填灰**（10-04，`attrs.CROP_KEEP` 0.05，写进模型的 `<头>.keep`，旧模型没有 = 不遮挡）：瘦高的框补成正方形会把旁边的人裁进来，离线对比 5 折 F1 0.76 → 0.90（`tmp/attrs-mask/report-fixed.md`）；
  存盘的裁图不遮挡（标注页要看上下文），训练提特征时按框事后遮挡；整帧回放的标准答案按标注页确认过的修正（`attrs_train.gt_fixes`，只在内存里）。
  YOLO 框出人物后按轨迹裁图，交给冻住的 DINOv2-small + 线性头（`models/attrs-<日期>.npz`，Claude 只当标注老师、不进运行时）判**外形**：不是人 / 点过火 / 黑影 / 先祖 / 共享空间玩家 / 变身。
  用途：① 复核——YOLO 高分框被稳定判"不是人"就撤下，低分框（`low_conf ~ conf`）等复核说是人才放行（目标是降门槛捞漏检）；② 点没点火和 YOLO 类别投票、带滞回；
  ③ 先祖 / 共享空间不算陌生人，变身照常认人。`enabled = false` 逐字是原来的行为；`[perception] enabled = false` 时不生效；主干和 npz 对不上 / 连续出错自动关掉。
  数据：`perception crops` → `perception attrs-label`（花额度）→ 管理面板「标注」页的「外形」标签页确认 → `crops --writeback` → `perception attrs-train` → `attrs-eval`；
  管理面板画面上画灰色虚线（被撤）/ "复核"（靠复核放行），难例多 `attrs_reject` / `attrs_disagree` 两种原因。上线门槛四条和剩下要人做的步骤见 `docs/progress/2026-09-28-yolo-training.md`「第二层」
  DINOv2-small 在 5070 Ti 上一张裁图约 5 ms、4 张约 20 ms（10-02，onnxruntime-gpu 1.23.2）；CPU 上一张约 32 ms，没 GPU 时别开，或者 `max_crops = 1`、`every = 1.0`（`[perception] device = "cuda"` 而主干只在 CPU 上跑时启动会警告）
- **难例收件箱（`[inbox]` / `[retrain]`；代码已完成，**还没在真机上跑过**，spec §11 五步；真 ultralytics 训练 / val 只用假函数测过，第一次真跑时核对 epoch 回调和 `all_ap` 读法）**：设计 `docs/superpowers/specs/2026-10-04-hardcase-inbox-design.md`，计划 `docs/superpowers/plans/2026-10-04-hardcase-inbox.md`。把真机 `run` 存的难例变成训练数据：下线时 `runs/<…>/hard/` 收进 `datasets/inbox`（`[inbox] enabled`）→ 管理面板停团子时问要不要整理（`[inbox] ask`），侧栏也提示没整理的运行 → `perception inbox process` 用 YOLO + 外形头先筛（外形头和 YOLO 一致、把握 ≥ `agree` 的人物框自动确认，全是这样的帧直接进待过目；拿不准的裁图交外形页人工判（不调 Claude）；`dup_diff` / `dup_gap` 去重；整次运行按运行目录名 crc32 % `val_every` 分验证 / 训练）→ 标注页「外形」页的「来自整理」筛选判拿不准的裁图 +「整帧」页过目 / 编辑（按键）→ 通过的帧进 `datasets/sky` → 整帧页「重训」= `perception retrain`（YOLO + 外形头，`[retrain]` 起点权重 / imgsz / epochs / batch / workers；回放对比报告 `tmp/retrain/<时间>/report.md`，攒够 `retrain_min` 张核对过的才提示）→「换上」写 `console.toml`（绝不碰 `config.toml`），「回退」按 adopt.json 还原。整理 / 重训是面板的任务槽（`console/jobs.py`，进度写 `tmp/jobs/`），任务在跑时叫醒团子 / 沙盒会先问要不要停任务
- **核显 / 没有 N 卡的机器**（`device = "dml"`，`vision/onnxrt.py`，**未在 Windows 核显上验证**）：在 GPU 机器上导出 `.onnx`（`.pydeps\bin\yolo.exe export model=models/sky-yolo-v10.pt format=onnx imgsz=960`）拷过去，
  `pip uninstall onnxruntime` 再 `pip install onnxruntime-directml`（两个包都叫 `onnxruntime` 模块，只能装一个；以后 `pip install -e ".[ocr]"` 会把 onnxruntime 装回来盖掉，要重装 directml；`pip check` 报 skydango 缺 onnxruntime 是正常的）。
  先 `perception bench --model models/sky-yolo.onnx --images <录像目录>` 看后端是不是 `DmlExecutionProvider`、每帧多少 ms，再按实测把 `fps` 降下来（估计 2~3）。
  DirectML 没装就退回 CPU 并警告；`.pt` 模型不支持 dml（torch 没有 DirectML，退回 CPU）。`[places]` / `[appearance]` 的 `device` 也能填 dml，动作模型跟着 `[perception] device`

## 认交互图标（`[icons]`，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-05-icon-detection-design.md`，计划 `docs/superpowers/plans/2026-10-05-icon-detection.md`（路线图 ②b）。
目标：画面里所有可点的圆圈 / 图标（互动请求、地图上的交互点）YOLO 都框成 `social_ring`，再认出是什么，写进状态和画面——**只认、不点**。**代码默认关；代码完成，还没有地图图标的模板 / 参考图、标好的数据和新 YOLO，真机验证见 spec §8**。
- **三步**（`vision/icons_map.py`）：① 归属 `owner_of`：圈在谁头顶（`under_x` / `under_up`）→ 人 / 先祖 / 物件 / 地图；② 认种类：`classifier = "template"`（`classify_template`，模板按框缩放匹配）或 `"dino"`（DINOv2 最近邻 `IconGallery`，`dino_match` / `dino_margin`，参考图在 `assets/icons`）；
  ③ 每条轨迹多数表决（`vote`、`min_hits` 连续帧）后才进 `PerceptionWatcher.icons(now)`；归属是地图的叫法走 `map_label`。
  认种类按轨迹节流：同一个圈最多每 `every`（0.5 秒，票攒满后 2 倍）认一次、一帧最多 `max_per_frame`（4）个，没轮到的沿用上次的结果；底库的 DINOv2 连续出错跳闸后退回模板；
  没有模板（`[social]` 关着 / 模板目录空）也没有底库时不开（启动警告，`icons()` 为空）
- **防误点（不受 `[icons] enabled` 管，关着也生效）**：没名字的圆圈只有「下面是人」（owner = person）才可能成为陌生人请求；先祖 / 物件 / 地图上的圈永远不进 requests。
  第二层没放行的低分人物框不算人、YOLO 漏一帧人请求立刻撤——**偏安全，真机可能少接一点陌生人点火**
- **看得到的地方**（`enabled` 时）：status「画面里的图标：…」、眼睛的位置说明（scene_note）、系统提示词一句、管理面板画面 kind `icon`（认不出画虚线）；认不出 / 低分的图标裁图存 `runs/<…>/icons/`，难例原因 `icon_unknown`。管理面板有 `icons.enabled`
- **改名**：`PerceptionWatcher` / `EnvWatcher` 的圆圈模板分类器实例属性改叫 `ring_icons`（构造参数仍是 `icons=`）
- **离线**：`perception icon-eval` 在标好的数据上比两种分类器；`perception icon-cut` 截模板 / 参考图
- **收件箱配套**（`perception inbox add`）：`<录像目录> [--every N]` 导入录像；`datasets/sky --redo` 老帧回炉（`_redo-<时间>/redo.json`，沿用原 split，通过时允许覆盖、先备份到 `datasets/sky/_backup/redo-<时间>/`，增强图 `_blur` / `_dark` 的标注副本一起改，撤销还原）；
  整理时 `find_flames` 整图补 `social_ring` 框（`flame_ring` = 1.82，**只量到 1 个样本**，而且那些旧框是 100 px 定尺，等有火焰录像重量）。标注规则见 `docs/progress/2026-09-28-yolo-training.md`

## 认装扮（`[appearance]`，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-01-appearance-design.md`（第一版）和 `docs/superpowers/specs/2026-10-03-identity-gallery-design.md`（**10-03 改成多张底库**，计划 `docs/superpowers/plans/2026-10-03-identity-gallery.md`，依据是 10-03 的离线 spike）。
名字标签挡住 / 太远读不到时靠外观把人认回来，认得出走开又回来的陌生人，说得出谁穿的什么。**代码默认关；本机 10-04 晚在 `console.toml` 打开、第一次上真机（启动转圈 31 帧里 4 帧是好样本、登记 4 张，都是正面附近的角度）；门槛还是 10-03 spike 估的**：
先用 `perception appearance-eval <录像> --gallery` 重算 `match` / `unsure` / `dango_match`（`tmp/record/walkaway-1002-1` 等还没跑），再按 identity-gallery spec §9「真机验证」五步（转圈登记的 status 和 `enroll/` 图、好友走远标签淡掉后管理面板画面里一直是"像他"、拿不准时自动喊勤不勤、镜头转过 / 面板开关后团子不被当陌生人、用当晚录像重标门槛）。
- **特征**：YOLO 人物框（点过火的人和团子）每 `every` 帧裁一次好样本（框高 ≥ `min_height` 0.13、不被别的框 / 聊天面板压住），一次裁图出两个特征：颜色直方图（`model = "color"`，头 / 身体各一份色相饱和度，也能换 `.onnx`）和 DINOv2-small（`[appearance] dino`，整框补成正方形，空 / 缺文件 = 只用颜色）；
  颜色平滑进轨迹 `data["feat"]`（只给装扮描述 / 判换装），这一次的 `Sample` 放 `data["sample"]`。`[attrs] backbone` 是同一个文件时共用一个推理会话（建在第二层原来的 device 上，不把第二层降到 CPU），否则用 `[appearance] device`；DINOv2 连续出错 10 次自关，只剩颜色
- **底库**（`vision/gallery.py`）：每个身份（团子 / 好友 / 陌生人编号）存多张样本、认人取**最像的一张**（spike：好友认出率颜色 56% → 89%）；去重 = 颜色 ≥ `DUP`（0.97）且（没有 DINOv2 或 DINOv2 也 ≥ 0.97）只刷新那张的时间；超过 `gallery_max`（40）挤掉和别的最像的（一样像挤旧的）；钉住的（转圈登记）不去重、不挤；
  **只从身份确定的框学**：团子（`self` 框 / `_mark_dango` 标记）、**这一帧**挂着名字标签的好友、已有编号的陌生人；像小明 / 可能是 / 接回 / 看着像团子都不学、不存负样本。认团子用 DINOv2、认好友用颜色（同场景颜色好过 DINOv2，两个合起来也没更好）；**团子底库每次上线重建**（跨天不稳），关系卡旧特征（`card_match`，已废弃、留在配置里不报错）不再拿来认人
- **三档**（只认没挂名字、没团子标记的点过火的轨迹，好样本攒够 `min_samples` 才判；名字标签永远说了算，挂上立刻清掉下面各档）：
  ① **看着像团子**（`dango_look`）：团子底库 DINOv2 最像 ≥ `dango_match`（0.80）且比任何好友都像团子 → 效果同团子标记（不判陌生人、不算没挂名字的人、不自动喊、图鉴按团子算），`DANGO_LOOK_HOLD` 3 秒没再判像就失效；
  **团子只有一个**（10-04，`_one_dango`）：这一帧已经认出团子（`self` 框 / `_mark_dango` 标记）时别人都不判、已经挂着的当场摘掉；没认出时最多留一个团子底库分数最高的（一样像留原来那个）。
  起因是 10-04 晚团子旁边的人被判"像团子"：DINOv2 补成正方形的裁图把旁边的团子也裁进去了（挨着团子的人 16/20 过 0.80），裁图不遮挡和 `dango_match` 还没改；
  ② **像小明**（`maybe`，颜色最像一张 ≥ `match` 0.88 且领先第二像 `margin`）：不算陌生人、好友还在身边时刷新在场（标签被挡不冒"走开了"），但不发 `arrive` / `return`、不打招呼；`people()` 里 `sure = False`；一个名字一帧只给一条轨迹，连续 `recheck` 个样本不像就摘；
  ③ **可能是小明**（`unsure`，`unsure` 0.83 ≤ 分数 < `match`）：先不判陌生人，持续 `unsure_wait`（2 秒）后交给「按 Q 喊一声」确认（见该节的"拿不准"起因）。喊完他身上亮出标签 = 确认（这一帧学进底库）、标签亮在别人身上 / 什么都没亮 = 摘掉、什么都没亮时记 `unsure_miss`（这条轨迹不再进"可能是"、不再为它喊，但仍可能变成像团子 / 像小明）、照常判陌生人；
  一直没喊成（额度、被拦、dry-run、`[call] auto` 关着）挂满 `unsure_wait + call.window` 秒摘掉；他的呼喊还在途中时不到期（免得标签亮前先判了陌生人）；"可能是"从第一次进入算起，暂停（`held`）会把计时一起往后挪。
  **身体的路径（`track` 盯人目标、`look_person` 按外观、`find` / 注意力找人）不把"可能是"当好友**，只有标签说了算或像小明才行；管理面板画面 / status：像小明 = 浅绿虚线"像小明?"，可能是 = 浅绿点线"可能是小明?"（status"可能是小明（没看到名字）"），看着像团子 = 灰白虚线
- **启动转圈登记团子**（`Body.enroll_self()`，大脑第一轮之前）：`Camera.spin` 转一整圈（不套 `panel.borrow`：`spin` 自己关 / 重开聊天面板，身体在 `held("camera")` 里），每张截图取团子框（高分 `self` 框优先，否则 `sweep` 的 `self_box`），裁好样本均匀挑最多 `enroll_max`（16）张钉住进团子底库，裁图存 `runs/<…>/enroll/`，status"团子登记：N 张"。
  条件：感知层 + `[appearance] enabled` + DINOv2 加载成功；**不转**（WARNING + status"团子登记：没转（原因）"；转之前自己截一张现看画面黑不黑、开着哪些面板，因为这时主循环还没跑过）：dry-run、画面黑着、别的面板开着、截不到图、没有视角控制、正在跑技能；转了但一张都没取到记"转了一圈没认出自己"，都不重试
- **陌生人**：认装扮开着时好样本不够先不判陌生人，最多多等 1.5 秒（`STRANGER_GRACE`）；判成陌生人后编号"陌生人A / B…"（这次上线不复用），走开超过 `keep` 又被认回来发 `stranger_back` 背景事件（"刚才那个陌生人A（白斗篷）又回来了"）；
  同屏两个陌生人不会共用编号；`stranger_forget`（30 分钟）没见就忘
- **装扮描述**（`describe`，默认开）：描述器排队用 `[models.wardrobe]`（默认 `deepseek/deepseek-flash`，备 `claude/haiku`）把裁图写成一句话（团子自己优先、好友其次、近处陌生人最后；每小时 `describe_max` 次、同一人一次上线最多重新描述 `redescribe_max` 次，额度用完等 `quota_wait`）；
  status 多一行"你自己：…"，"身边的好友：小明（粉色长斗篷·…）"，"画面里：…、像小红（没看到名字，…）、陌生人A（白斗篷，…）"；`look_person` 也接受"陌生人A"；提示词多一段装扮的规矩（`enabled` 时才加）。沙盒里不描述
- **判不判换装**（`outfit_change`，**默认关**）：10-01 在 15 段录像上标定（`tmp/appearance-eval/`），颜色特征下同一身衣服常跌到 0.15~0.5，0.70 的门槛会把同一身判成换装、往关系卡里记假装扮；
  换 DINOv2-small（`models/dinov2-small.onnx`）也没稳住（对比在 `tmp/appearance-eval/compare-color-dinov2.md`）。**关着时**：不追加新的一套、不发 `outfit` 事件、上线中途不重新描述；
  好友 / 团子每次上线描述一次（每个好友每次上线多一次 Haiku），好友的描述**覆盖**关系卡最近那一套；描述回来前 status 不写他的装扮（不拿上次的旧描述说事）。换了更稳的特征模型再打开；`changed` 默认 0.40
- **关系卡**：`Card.outfits` 每个好友留最近 `outfit_keep`（3）套（描述、特征、特征模型 key、第一次 / 最后一次的日期），启动时载入记忆簿（只用来取描述 / 判换装，**不拿来认人**）；`outfit_change` 开着时：第一次靠标签认出好友、和卡里最近一套比：< `changed` 算换了装、排描述，
  新描述回来后放 `outfit` 背景事件（"小明换了装扮：上次是「…」，现在「…」"，arrive 文字不变）；同一套直接用卡里的描述、不花额度。只在 live 写盘；管理面板「内心」页关系卡显示装扮
- **攒数据**（`save`，默认开）：好样本存进 `runs/<…>/appearance/`（见「运行目录」），以后训认人模型用（只有一个好友的录像训不出来，这一期不做）
- **管理面板画面**：见上面三档的画法；陌生人标编号，鼠标悬停看装扮
- **已知限制**：颜色直方图不看亮度，白 / 灰 / 黑发色、同色深浅分不开；不同地图、白天晚上光照差得多（所以底库只认这次上线、不跨天）；撞衫（季节装扮、默认斗篷）会认错；好友中途变身（雪人）底库对不上；框高 < 0.13 的远处小人两种特征都认不出（spike 结论：瓶颈是朝向 / 场景，不是距离，所以不做远处底库）；身高没做；没点火的黑影没有外观

## 装扮图鉴（`[catalog]`，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-02-catalog-collect-design.md`（开头有五期路线图：收集 → 归类 + 部位 + 管理面板起名 → 游戏里主动问 → 用图鉴认 → 地图 / 先祖 / 物品），计划 `docs/superpowers/plans/2026-10-02-catalog-collect.md`。
目标是让团子认出好友身上发型 / 斗篷 / 面具的**俗称**：图鉴从团子自己的经历里长出来，认法是 DINOv2 特征检索 + Claude 对图确认（新加单品不用重训）。**这一期只做收集；10-02 起每次真机 run 都在收，门槛是估的**。已知问题：裁图偶尔把聊天面板裁进去（面板遮挡的门槛漏了，没修）；10-03 晚"陌生人"里约七成是团子，靠 self 连续性修了，修后还没复查。
- `run`（dry-run 和 live 都算）时感知层每帧把人物框交给 `CatalogCollector`：够大（`min_height`）、不贴边、没被别人 / 聊天面板挡（和认装扮共用 `clear_box`）、清楚（拉普拉斯方差 ≥ `sharp_min`）才收；
  好友（名字标签证实的）、团子、感知层判过陌生人（`data["stranger"]`）的点过火的人收，有名字标签但不在好友名单里的不收，黑影 / 先祖 / 共享空间 / 第二层没放行的不收；"像小明"的按陌生人存、索引记 `maybe`；陌生人框太黑（`dark_max`，YOLO 认错的黑影）不收，团子身上多出来的 player 框按团子算
- 每个身份每次运行留最好的 `per_who` 张、两两隔 `gap` 秒；陌生人轨迹断了写出、每 `flush_every` 秒和退出时全部写出；存进 `catalog/inbox/<日期>/<运行>/<身份>/<名次>.jpg` + 同目录 `index.jsonl`（`catalog/` 不进 git，只在本机；第 2 期起完名的精选图才进私有仓库）；**`catalog/` 不会自动清理**（不像 `runs/` 只留最近 20 次），一晚几 MB，要清自己删
- 和认装扮（`[appearance]`）完全分开，不带它的副作用；`enabled = false` 时感知层逐字照旧；管理面板有 `catalog.enabled`
- 定门槛：`catalog collect <录像目录>` → `tmp/catalog/<时间>/`（`sheet.jpg` 总览、`candidates.jsonl` 每个候选过没过哪条门槛）

## 团子的接口（`[viewer]`，`run` 总是开）

设计见 `docs/superpowers/specs/2026-10-04-console-attach-design.md`（**10-04 删了 viewer 网页和 `view` 命令**，计划 `docs/superpowers/plans/2026-10-04-console-attach.md`；面板起的团子 10-04 晚起天天在用，**终端起 → 面板接管 → 再起被拒 → 关面板接着跑这 spec §5 四步还没完整走过**）。
画面、大脑、聊天、手动控制都在管理面板「真机团子」页看，团子自己不出网页、不开浏览器。
- `run`（大脑模式和 `--no-brain` 都算）建好运行目录之后、碰设备之前，总在 `127.0.0.1:[viewer] port`（默认 **19391**）开接口：
  `/status` `/snapshot` `/brain` `/chat` `/control` `/control/options` `/inner` `/inner/forget` `/shutdown`（本机 Host 校验、POST 要 `X-Skydango` 头 + JSON、`/shutdown` = Ctrl+C 走正常收尾）。
  **端口被占就拒绝启动**（「19391 端口上已经有一个团子在跑…」，不建设备）：面板起的、终端起的，同一时间只有一个团子
- `/status` 带 `run` 节（pid、运行目录、live / brain / emotes / duration、启动时间、`console` = 是不是面板起的）：管理面板只认带它的响应是团子
- **管理面板接管终端起的团子**：槽空着时（面板启动、页面每秒拉 `/api/state`、叫醒前、起沙盒前、删性格条目前）探这个端口，探到就当"在跑（终端起的）"显示，
  画面、大脑、聊天、手动控制、内心页实时合并、停止都照常（停止先 `/shutdown`，`stop_timeout` 还在就按 pid 杀进程树）；连续 3 次探不通 = 退出了；
  日志抽屉读那次运行目录 `agent.log` 的 INFO 以上（拿不到终端输出）；**面板退出不停它**（不是面板的子进程）
- 身体 / Agent 每圈把这一帧交给接口（限 `viewer.fps`，没人看不压 JPEG），出错只记 DEBUG 日志；大脑时间线数据在 `brain/trace.py`（只在内存里），手动控制 `brain/manual.py` **总是真执行**（大脑 dry-run 也一样）、照样过身体的护栏，做成后放 `manual` 事件告诉大脑
- 只听本机（画面里有好友昵称和聊天）：`viewer.host`、`console.child_port` 已废弃（读到只警告），局域网 / 手机看画面的用法取消了
- `run --view` 的 `--view` 留成隐藏参数，给了只打一行提示；`run --no-browser` 也一样留成隐藏的空参数；`--viewer-port` 删了

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
**2026-10-01 重做**（设计 `docs/superpowers/specs/2026-10-01-console-redesign-design.md`，计划 `docs/superpowers/plans/2026-10-01-console-redesign.md`）：左侧栏（运行卡片：状态、停止、`run.error`；各页入口）+ 八页，路由 `#sandbox`（沙盒，**默认页**）、`#live`（真机团子，**2026-10-01 晚再改**，spec `docs/superpowers/specs/2026-10-01-console-live-page-design.md`：顶栏停着是启动选项、跑着是运行信息 + 「日志」抽屉；三栏 左 现在 + 身边和状态 + 手动控制（`livectl.js`）/ 中 画面（直接画在页面里，画框代码 `console/static/stage.js`）+ 大脑控制台（`brainlog.js`）/ 右 聊天记录（viewer `/chat`，身体 `on_line` + 事件队列 `tap` 记进 `brain/transcript.py`，只在内存里、最近 500 行）；停下后内容留着；**还没在真机上用过**）、`#inner`（内心，见下）、`#scenarios`（剧本和报告）、`#labeling`（标注）、`#models`（模型，见「模型供应商」）、`#settings`（设置，带分组目录）、`#device`（设备）；旧的 `#overview` 已改名 `#live`。10-02 晚起真机 run 都从这里起（spec §8 真机验证 1~6 没逐条走）。
- **页面拆分**：`static/console.html` + `console.css`（颜色只在 `:root` 定义，别处用 `var(--…)`，有测试查）+ `common.js`（`$` `el` `post` `ask` `toast` `problemList` `Pages` `go` `refresh` 等公共接口，node 能 require）+ `markdown.js` + 每页一个 js（`sandbox.js` `live.js` `inner.js` ……），经 `/console/static/<名字>.css|js` 提供，只给 `static/` 目录列表里真有的文件（Windows 设备名 con.js / nul.js 一律 404，不碰文件系统）。页面不引外部资源、请求一律相对路径。
- **原生 `confirm` / `prompt` / `alert` 全换成页内对话框和提示条**（`ask()` / `toast()`；Claude 桌面版内嵌浏览器里原生弹窗用不了），测试禁止再出现。
- **预检**：大脑模式下也查 LLM Key；每个问题带 `setting` 跳转目标，「去设置 →」跳到设置页并高亮那一行（樱花底闪一下）。
- **剧本和报告页**：报告在页内直接读（`console/reports.py`，`GET /api/sandbox/reports[/<name>]`），不用再去翻 `sandbox/reports/`。
- **「标注」页**（`#labeling`，左栏「数据」组；后端 `console/labeling.py`）：逐段看动作片段的动图，按键标（1~9 = 类别、0 = 丢弃、Z = 撤销；10-03 去掉了「回车 = 同意 Claude」：初分认不出动作，直接人工标），片段挪进 `datasets/gesture/<动作>/`，然后 `perception gesture-train`。同一页还有「外形」标签页（第二层，后端同 `console/labeling.py`）：逐张看人物裁图和 Claude 的初分，确认 / 改类别 / 丢弃，裁图挪进 `datasets/attrs/form/<类别>/`；10-04 起只出 不是人 / 点亮的人 / 黑影 / 先祖 四个按钮（共享空间、变身按点没点火标；先祖不够 20 张训练时并进不是人），筛选里照旧列全部类别。
  外形页的「导入未确认」筛选（10-04）：`datasets/sky` 直接导进 `form/` 的裁图没人看过、标签混了不少错的（`confirmed = false`，按 `_labels.jsonl` 里有没有人标过算，`attrs_data.hand_labels`）；在它现在的类别上按一下 = 原地确认（记一条 from == to、不挪文件），`attrs-train` 默认只用确认过的；其中「回放用」（`replay`）是数据集验证集帧里的框 = 整帧回放的标准答案，先过它们回放才准。
  第三个标签页「动作名」（`console/emotenames.py` + `static/emotenames.js`）：网格列出 `emotes scan` 扫下来的全部图标，起名 = 把 `scan/NNN.png` 复制成图标库的 `emotes/<名字>.png`，改名 / 清除（挪进 `emotes/_removed/`）；
  哪个扫描图标叫什么按剪影配（门槛 0.7、每张库图只配一个），不靠编号，重扫不错位。**只起名**，团子能做哪些动作照旧由轮盘 / 白名单决定；团子在跑也能起，下次启动生效；
  改名 / 清除时旧名字还被 `social.after_light` / `emotes.extra` / `gesture.names` / `reflex` 清单引用的只提示、不自动改。没有“重新扫描”按钮（要往游戏里按键，终端跑 `emotes scan`）
- **「整帧」标签页**（标注页第四个标签，`console/static/frames.js`，后端 `console/frames.py`，难例收件箱，见「YOLO 感知层」）：整理好的整帧过目 / 编辑、重训区（`retrain.js`）；侧栏有「素材没整理」一行（点了起整理），团子停下时若有难例会问「现在整理吗」；整理 / 重训期间起团子会先问要不要停任务
- **三个文件**：`config.toml` 面板只读不写；面板改的设置写 `console.toml`、密钥按环境变量名写 `secrets.toml`（明文，都 gitignore，和 config.toml 同目录）。
  加载顺序 默认值 → config.toml → console.toml；`secrets.toml` **覆盖**已有环境变量。终端直接跑命令也读这两个文件（启动时日志里打「console.toml 覆盖了 N 项」）；
  `console` 自己不把密钥写进自己的环境变量（页面上「清除」之后子进程才不会继承旧 Key），只注入它起的子进程
- **父子进程**：面板起 `python -m skydango -c <config> run [--no-brain] --live|--dry-run [--no-emotes] [--duration N] --parent-pid <面板>`（接口端口来自同一份配置的 `[viewer] port`）；
  模式和真发 / 只看总是显式传，`config.toml` 的 `reply.dry_run` 不会改掉面板上的选择。同一时间只有一个团子（面板起的或终端起的，见「团子的接口」）
- **停止**：面板 `POST /shutdown` 给团子的接口 → `interrupt_main()`，走和 Ctrl+C 一样的收尾；`[console] stop_timeout`（60 秒）还没退就按进程树强杀，页面提示检查轮盘。
  子进程每 2 秒看父进程还在不在（`--parent-pid`），面板没了就自己正常退出；`/shutdown` 和看门狗共用一个**只中断一次**的钩子（`watchdog.once`），第二次中断不会打断收尾；
  面板终端里等收尾时再按一次 Ctrl+C = 强杀。19391 上已经有团子（终端起的、上次留下的）就**接管**它、不起新的；端口被别的程序占着时叫醒会报出来
- Windows 上子进程用 `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`：有自己的隐藏控制台，关掉面板终端窗口不会把团子直接结束（**未在真机验证**）。
  面板到子进程的本机请求都**不走代理**（开着 Clash 时 urllib 默认会把 127.0.0.1 也转走）；`console.toml` / `secrets.toml` 坏了面板照样能开，在设置页和预检里报错
- **安全**：只监听 127.0.0.1；`/api/*`、`/live/*` 都校验 Host（防 DNS 重绑定）；POST 要 `X-Skydango: 1` + JSON + ≤ 64 KB（和 viewer 共用 `is_local_host` / `post_guard`）。
  浏览器拿不到完整密钥（只显示「已设置（sk-…abcd）」）
- 团子运行时不能做设备检测、不能切输入法（设备归身体线程独占）；运行中改设置照样保存，提示重启后生效
- 设备页「输入法」一栏（`POST /api/device/ime`：`{}` 读、`{"set": id}` 切，只能切 `ime list -a -s` 里有的）：列出装了的输入法、标出当前的，点「切到这个」
- 团子的接口有 `/status`（只有状态、不带图，带 `run` 节）和 `/shutdown`；大脑模式的状态多了「正在做」「刚说过」
- **「内心」页**（设计见 `docs/superpowers/specs/2026-09-30-inner-viewer-design.md`；真机和沙盒都在用，spec 真机验证没逐条走）：现在（心情 / 精力 / 别扭 / 收着点 / 心愿）、精力曲线 + 心情色带（24 小时 / 7 天，圆点 = 一次反思，点了跳到记录）、
  反思记录（`changes` 逐行、没收下的折叠、下线那次标出）、性格档案（每条能「删」）、关系卡、最近 10 次上线和日记。
  `GET /api/inner` 读 `memory/inner/`（团子不在跑也能复盘），团子 `running` 时再取团子接口的 `/inner`（`Body.inner_snapshot`，经身体线程、3 秒超时），现在 / 性格以实时为准，取不到标"实时取不到"；
  `POST /api/inner/forget`：在跑转发 `/inner/forget`（`Body.forget`，live 才写 `persona.json`；dry-run 的团子只删内存里的，面板顺手把文件也改了），没在跑直接改 `persona.json`，启动 / 停止中拒绝；终端起的团子会先被接管、照样转发给它；`current.json` 在 3×`save_every` 内更新过（别处有团子在跑、面板探不到）也拒绝，免得它把删掉的写回去。
  团子醒着时每 5 秒刷新，别的时候打开时读一次 + 「刷新」。流水账只在开了反思时记（`[inner] reflect`），live 写盘、dry-run 只在内存里

## 模型供应商（`[providers]` / `[models]`）

设计见 `docs/superpowers/specs/2026-10-05-model-providers-design.md`，计划 `docs/superpowers/plans/2026-10-05-model-providers.md`。**代码 10-05 做完，spec §6 的沙盒 / 真机验证四步还没走。**
- **供应商** `[providers.<id>]`：接入方式只有 `claude-code`（`claude -p`，`path` / `token_env` / `config_dir`）和 `openai`（OpenAI 兼容，`base_url` / `key_env` / `timeout` / `max_retries`）；`models` 列模型、`vision` 勾能看图的。
  两个文件都没写 `[providers]` 时用内置的 `claude` + `deepseek`（deepseek 的地址 / Key 名 / 模型取旧 `[llm]`）；写了就只用写了的（config.toml 和 console.toml 按 id 合并）
- **用处** `[models.<用处>]`（`main` / `backup` 写成「供应商id/模型名」，按第一个 `/` 拆；`temperature` / `max_tokens` 只在配置文件里写）：
  brain / memory / reflect 默认 `deepseek/deepseek-chat`、备 `claude/sonnet`；reply / text_label `deepseek/deepseek-chat`；eyes / wardrobe / image_label `deepseek/deepseek-flash`（V4.1 Flash，内置的 deepseek 供应商把它勾成能看图；10-06 起，之前是 Claude），备 `claude/haiku` / `claude/haiku` / `claude/sonnet`。
  **10-06 换识图默认之前、面板保存过「模型」页的**：console.toml 里整张写着旧的 deepseek 供应商（没有 `deepseek-flash`），要在模型页给 deepseek 加上它、勾能看图，再把三个看图的用处「恢复默认」
  看图的用处（eyes / wardrobe / image_label）只能选能看图的，选错了这一处停用、团子照常起；引用了不存在的供应商同样停用；模型不在列表里只警告
- **旧字段照样读**（启动时警告一行）：`[llm]`、`[brain] model / eyes_model / memory_model / claude_path / token_env / config_dir`、`[inner] reflect_model`、`[appearance] describe_model`、`[assist] model`，换算见 spec §1.3；同一用处 `[models.*]` 说了算。
  **本机 `config.toml` 只有 `[llm]`（DeepSeek）、没写 `brain.model`：换算后大脑就是 DeepSeek**（不用改配置）
- **管理面板「模型」页**（`#models`，`console/models_view.py` + `static/models.js`）：供应商卡片（地址 / 路径、Key 或令牌、模型和「能看图」、测试、删除；还在用的、config.toml 里定义的删不了）、
  「+ 添加供应商」（DeepSeek / ChatGPT / 通义 / Ollama / Claude Code 模板）、每个用处的主 / 备下拉和「恢复默认」；整页一个「保存」写 console.toml（只写和默认 + config.toml 不一样的用处），Key 写 secrets.toml。
  面板第一次保存会把当时生效的供应商（含旧 `[llm]` 换算出来的）整张写进 console.toml，之后改 config.toml 的 `[llm]` 不再影响 deepseek 那家。设置页的大模型一组和 Claude 令牌挪到这里了
- **预检**：大脑模式只在大脑的主和备都用不了时拦（「大脑没有能用的模型：…」，跳到模型页那一行）；普通模式查回复的主模型；别的用处有问题不拦（启动日志一行一个，用到时那一处停用或改走备用）
- 运行时见「统管大脑」（闸、代看）；`brain.jsonl` 每轮多 `provider` / `model`；离线命令（`memory update`、`look`、看图标注四个命令、`addressee label`）各走对应用处、各自一套闸
- 注意：大脑换了模型，`history.jsonl` 里 Claude 的旧回复会被 DeepSeek 模仿（想要的接续感）；要清照「记忆」一节挪进 `archive/`

## 统管大脑（`[brain]`，`run` 默认）

设计见 `docs/superpowers/specs/2026-09-27-brain-claude-code-design.md`（身体部分见 `2026-09-27-brain-design.md`）。
- **`run` 默认就是大脑模式**；`run --no-brain` 进旧的普通 Agent（`agent.py`），只留作调试（`--brain` 保留兼容，不用加）
- **大脑用哪个模型看 `[models.brain]`（10-05 起默认 `deepseek/deepseek-chat`，备用 `claude/sonnet`，见「模型供应商」）**：
  OpenAI 兼容的走 `ToolLoopBrain`（function-calling 循环、每轮重发、带最近 `[brain] history`（8）轮的唤醒消息和"你这一轮说了 / 做了什么"，看不了图）；
  claude-code 的是常驻的无界面 Claude Code（`claude -p` stream-json，订阅登录，`--effort low`），身体的工具经本机 MCP 服务（`sky`）给它，
  `--tools ""` 关掉所有内置工具；两种都只能调 look / look_at / look_person / look_around / status / chat_log / recall / say / emote / set_request_policy / camera / camera_reset / attention / move / check_friend / track / find / stop_task / panel_read / panel_press / panel_close（开了 `[call]` 还有 `call`，开了 `[backstage]` 还有 `introspect`）
- **和用户自己的 Claude Code 隔离**：单独配置目录 `.brain-claude/` + `claude setup-token` 生成的令牌（用户环境变量 `SKYDANGO_CLAUDE_TOKEN`）。
  沿用用户登录会把用户的插件、钩子、技能一起加载进大脑（实测）。子进程里去掉所有 `ANTHROPIC_*`（`API_KEY` 在时 `-p` 一定用它；10-03 晚漏了 `ANTHROPIC_BASE_URL`，令牌被发到别的地址、连着 179 次 401）和用户自己的 OAuth 令牌
- 眼睛 = `[models.eyes]`（默认 `deepseek/deepseek-flash`、备 `claude/haiku`，只能选能看图的模型）：有人来 / 走、画面大变（隔 ≥20 秒）或 3 分钟没看时，把身体最近一帧写成文字；大脑醒来的消息里只有文字，要原图才 `look(image=true)`
- `check_friend(x, y)`：点一下人物打开右侧好友树面板，截图交给大脑自己判断是不是好友，再按 ESC 关掉（用户确认）、恢复聊天面板（`game/friendtree.py`）。
  面板样子、好友和陌生人的面板怎么区分**未核对**，`[friend_check] enabled` 默认关；先用 `friend-check X Y` 手动核对，截图在 `runs/<…>/friend-check/`
- `recall(query, who, days)`：翻以前的聊天（`chat/recall.py`）：在 `history.jsonl` 里按关键词（中一个就算，中得多的排前）/ 人名 / 往前几天（默认 14，最多 90）找原话，
  最多 8 轮、每轮 200 字，附上 notes.md / inbox.md 里相关的行；只读、不调模型、不占身体线程，dry-run 也能查。提示词要求有人提起以前的事时先查再答，查不到就说记不清、别顺着编。
  history 只记团子开口的那几轮（那句话 + 之前听到的），团子没接话时别人聊的不在里面
- **重启接得上话**：启动时把 `history.jsonl` 最近 `[brain] history_turns`（默认 20，0 关）轮原话放进系统提示词末尾的「上次聊到哪」一节，
  标出最后一轮是多久前（格式同 recall）；放系统提示词而不是第一条消息，自动压缩时不会被总结掉。dry-run 也带（只读）
- `look_person(名字)`：按名字（容忍 OCR 错一两个字）找到这个人、裁出来给大脑看原图（"看我衣服好看吗"）；打开感知层用 YOLO 的框，
  否则按名字标签往下估一块（宽 3 倍、高 6 倍标签高，**未在真机核对**）；和 look 共用频率限制；找不到时告诉大脑现在认得出谁。
  **被团子挡住时换角度**（`[peek]`，`brain/peek.py` 纯决策 + `Body._peek`；10-02 晚真机试过没转、裁到了团子自己（已修），修后还没触发过，要好友站到团子正后方专门测）：好友只有名字标签、标签压在团子框上（团子框 = YOLO 的 `self`，只信画面水平中间附近的）→
  在 `look_person` 里闭环：每按一下（`Camera.nudge` 左右 / `zoom_once` 拉近拉远）等画面停稳、看 YOLO 最新结果再定下一下；贴太近先拉远、转不动拉远再转、露出来太小就拉近（拉近后又挡住 / 出画面退一步）；
  **看完镜头不复位**（结果和 status 的"镜头："告诉大脑，要不要 `camera_reset` 它自己定）；status 里有"被你挡住：小明"。dry-run、黑屏、track 在跑时不转，只说被挡住了
  和团子框是同一个身体（`same_body`）的人物框不算他的身体（10-02 晚 22:26：团子身上的框挂上了他的名字，一看就"露出来了"、裁了团子自己的背影）；
  每一下按什么、看到什么写 DEBUG（"换角度 第 N 下：…"）；裁给大脑的图和画了裁剪范围的整帧存进 `runs/<…>/look_person/`
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
- **空闲注意力（东张西望）**（`[attention]`，`brain/attention.py` 纯决策 + `brain/search.py` 怎么找 + `Body._watch_attention`；spec `docs/superpowers/specs/2026-09-30-idle-attention-design.md`、
  **10-03 改成有意识地找** `docs/superpowers/specs/2026-10-03-attention-search-design.md`、计划 `docs/superpowers/plans/2026-10-03-attention-search.md`；**未在真机验证，数字都是估的**）：
  **只在 `[panel] mode = "auto"`、聊天面板关着、身体闲着时动**（always 模式下完全不动；「聊着」（面板关着跟画面里的好友聊）时 `PanelManager.quiet()` 为假，也不动，不把镜头从聊天对象身上转走）。
  被动注意照旧：候选目标（好友 / 陌生人说话、走近、挥手鞠躬、站着的好友）按基础兴趣 ×（1 − 看腻）挑一个，`Camera.nudge` 小步拉向画面中间；冒气泡 / 有人走近时**先看一眼再开面板**（`PanelManager.hold_off`，最多 `look_first` 2 s）。
  **随意看删了，没有动机就不转**；没有被动目标时做"找"（`search.py`，分段转：一段 `seg_presses` 下、停 `dwell` 秒）：
  ① **找刚走开的好友**（身体按 `env.labels` 里他名字标签最后的位置发起）：从画面边上出去的先往那边转 `lost_segments` 段、再按 Q 看名字贴在哪边；在中间淡掉的先按 Q（**自动喊并进来了**，额度照 `[call]`；注意力接管时 `_watch_call` 不再自己判断），画面里还有没挂名字的人就不转；
  ② **环顾**：画面里 `scan_after` 秒没人、离上次 `scan_every` 秒（× 精力 `Effects.wander` × 模式），往 `Heading` 记的**最久没看过的方位**转 `scan_segments` 段，看到人就停；
  站着的好友不打断找，有人说话 / 走近打断，`resume_within` 秒没进展就不找了。结果：没找到 / 可能是他、环顾从"没人"变"有人"才发**背景事件 `search`**，其余只写 status（"在找：小明（他刚从左边走了…）""刚才往右后方看了看：附近没人"）。
  朝向估计（`press_deg` 待 `camera spin` 标定）在走路、黑屏、别人转镜头后清零。输入框开着、技能在跑、有互动请求、在回聊天、黑屏、牵手、别的面板开着、刚做完动作、dry-run 时不按（dry-run 照样算）。
  大脑工具 `attention(mode, focus)`：随意 / 好奇（环顾更勤、陌生人说话更有意思）/ 专心（只看好友说话、冲团子来的、关注的人，不环顾、照样找走开的好友）/ 别动；**不算"做了事"**。
  刚找完 / 刚被别人（`find`、`track`、`look_around`、`camera`、`look_person`）转过镜头也算环顾过了，隔 `scan_every` 才环顾，技能在跑时不发起环顾；没转没喊的找走开的好友不发事件、被打断的环顾 status 写“想往…看看，被打断了”。
  管理面板有 `attention.search` 开关（关掉 = 不找、不环顾，回到只做被动注意；`_watch_call` 照旧自己喊）
  旧配置里的 `wander_*` 四项加载时跳过并警告（`config.DEPRECATED`）
- **`find(名字, 秒)`**（技能，`brain/find.py`）：转镜头找某个好友——刚走开过按①找，没线索先按 Q、名字贴边就往那边转、没有就转一圈；找到 / "可能是他"发 `task_done`，没找到 `task_failed`；
  镜头不复位、要一直盯着由大脑接着调 `track`；dry-run 拒绝；不在好友名单里拒绝；画面里已经有他直接回"就在画面里"
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
- 大脑一轮 120 秒没结果就结束进程、下次 `--resume` 接回；连续失败 120 秒：聊天交给 `[models.reply]` 的纯文字备用回复。
  **主模型那家额度 / 余额用完或认证出错**（大脑自己撞上，或别的用处先撞上关了那家的闸）切到 `[models.brain] backup`（懒建，可以跨 Claude / OpenAI 兼容），不切回（额度恢复了要重启）；
  没有备用就按 `limit_retry` 退避。`[brain] fallback` / `force_fallback` / `fallback_max_tokens` 删了（读到只警告），`fallback_history` 改名 `history`
- **按供应商的闸**（`models/gate.py`，原 10-04 的 Claude 总闸推广）：一次运行一套，大脑、记忆、反思、眼睛、装扮描述、备用回复共用；关闸 = claude-code 的额度 / 认证错、OpenAI 兼容的 401 / 403 / 402 / 429 insufficient_quota（别的 429、超时、5xx 不关）。
  每个用处的 `GatedCall`：主那家闸开着走主，撞墙关闸、这一笔改走备；主和备都用不了抛 `ModelUnavailable`（眼睛 / 装扮描述停用、随手记跳过、反思沿用上一份）。
  下线那次反思主和备都压到 `_final_timeout`（管理面板 `stop_timeout` − 25 秒，备用不重试），免得被强杀丢日记；沙盒自己的场景描述不接闸；status 多一行「模型：大脑 deepseek/deepseek-chat」（切过备用写原因，停用的看图用处跟在后面）
- **代看**（大脑模型看不了图时，OpenAI 兼容的大脑一律算看不了）：`look(image=true)` / `look_at` / `look_person` / `check_friend` / `panel_read(image=true)` 照样能调，多一个可选参数 `question`；
  拿到的图交给眼睛（`Eyes.proxy`）写成文字，返回「（眼睛代看）……」；眼睛用不了返回「现在看不了图」；和眼睛共用 `look_min_interval`（刚看过就直接给眼睛的描述）
- `say` 拦重复（10-04）：10 分钟内说过几乎一样的话（`similar` 0.85）拦下、告诉大脑"你刚说过「…」"，Claude 大脑也一样；手动控制不管
- 记忆整理（随手记 inbox.md、整理 notes.md）走 `[models.memory]`（默认 DeepSeek，备用 Claude sonnet；claude-code 时一次性 `claude -p`，工作目录 `runs/<…>/brain/memory/`）
- 退出：身体先恢复轮盘、再复原镜头（不等大脑）→ 输入法切回 → 开了反思（`[inner] reflect`，默认）走最终反思写日记和要点（见「内心层」第 2 期），`reflect = false` 时 live 才让大脑写一份经过记进 `inbox.md` → 按进程树结束 Claude Code
- 调提示词时开管理面板「真机团子」页：大脑控制台能看到每一轮它收到了什么、调了什么工具、工具返回了什么（见「团子的接口」）；
  手动控制栏能绕过大脑直接试身体的工具（身体方法的 `live=True`），大脑会收到 `manual` 事件
- 前提：`pip install --user mcp`；大脑的主或备有一家能用（DeepSeek Key，或运行一次 `claude setup-token` 并 `setx SKYDANGO_CLAUDE_TOKEN "<令牌>"`），两家都不能用才拒绝启动

## 看场合主动开口（`[proactive]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-29-proactive-chat-design.md`，计划 `docs/superpowers/plans/2026-09-29-proactive-chat.md`。
远期目标是把团子做成有存在感的**游戏里的伙伴**（像 Neuro-sama，但不做主播）；这是第一步：看到了就有话说。**已在真机 live 里默认开着跑过（9-30 起），spec「真机验证」清单没逐条走，数字都是估的**（spec「真机验证」四步）。
- **眼睛挑新鲜事**：自动看时附上上一份描述（≤ `prev_max_age`），多要一项“新鲜事”（天黑了、好友换装 / 弹琴、篝火……），身体放 `notice` 事件叫醒大脑；
  认地图的地名从一个变成另一个也发（"看起来到了雨林"）。大脑自己 `look` 不发；`notice_min` 内、和上一条一样、没熟人、主动额度用完都不发（省额度）。好友在身边时眼睛 `auto_look_busy`（60 秒）看一次
- **场合**（`brain/occasion.py`，纯计算）：热闹 / 安静 / 没熟人 + 上次主动开口有没有人接 + 还能主动说几句，写进 `status`（网页、管理面板卡片也有）。
  **主动** = 大脑这一轮不是被聊天 / 主人命令叫醒的（`brain_busy()` 为假）；手动控制说的不算
- **护栏**（`body.say`，只管主动）：没熟人不说、10 分钟额度（热闹 4 / 安静 2）、两句间隔 60 秒（好友刚来 `greet_window` 60 秒内、团子还没开口时打招呼不受间隔限制，额度照旧）、连续 3 句没人接就停到有好友说话；dry-run 也计数
- **提示词**：“别自言自语”换成「主动开口」一节（看场合、带自己的看法别播报、新鲜事不是任务、不说就在心里写“不说：原因”）。
  调的时候在管理面板「真机团子」页看大脑控制台
- **喜好**：`memory init` 的人设模板多了「喜好和看法」；已有的 `memory/profile.md` 要自己把这一节加进去。随手记会记下团子说过的评价，下次态度一致
- `enabled = false` 完全照旧（旧提示词、不发 `notice`、不拦）。history.jsonl 先不清（旧回合只是没主动，不冲突），太保守再按「记忆」一节挪走

## 分清在跟谁说话（`[addressee]`，大脑模式）

设计见 `docs/superpowers/specs/2026-10-05-addressee-design.md`，计划 `docs/superpowers/plans/2026-10-05-addressee.md`，路线图 ③。起因：好友之间聊天团子句句都接（10-03 / 10-04 真机）。
**代码 10-05 做完，还没在真机验证；离线评估（`addressee label` / `eval`）也还没在真实数据上跑过，三条门槛过没过不知道，数字都是估的。旧日志（没有「跟谁说」行）上算出的门槛只作参考（团子当时句句都接，「在接你的话」窗口几乎一直开着，`report.md` 按规则理由分列）；正式门槛用合并后录下的新日志算。代码带着 `enabled` 默认开合并，不合适可在管理面板设置页关掉。**
- **每句好友聊天判一次**（`brain/addressee.py`，纯规则、不调模型；身体 `_heard` 里调）：`me` 跟你说 / `other` 跟别人说 / `all` 说给大家 / `unsure` 拿不准。
  规则从上往下命中即停：不是好友 → unsure；叫了 `[proactive] self_names` → me；叫了别的好友 → other；`group_words`、或身边 ≥ 2 个好友时开头是 `greet_words` → all；
  接团子的话（团子 `followup_window` 内说过、他之后头 `followup_lines` 句、中间没有跟别人的来往）→ me；`thread_window` 内还在跟别人一来一回（团子没插进来）→ other；身边只有他 → me；其余 unsure。
  卡洛的 `#` 命令、陌生人的聊天不进这套判断（陌生人一律 unsure）。判断出错记日志、按 unsure
- **叫法**：friends.md 每个好友那一节加一行 `- 叫法：卡洛、老登`（顿号 / 逗号 / 空格分开，加上游戏昵称）；≥ 3 个字的叫法允许错一个字（OCR），1~2 个字只认原样。没写就只认昵称，改了不用重启。已有的 friends.md 自己加
- **事件**：`me` / `all` / `unsure` → `chat`（立刻叫醒）；`other` → 主动开口额度还有放 `aside`（叫醒）、用完放 `aside_bg`（`BACKGROUND`，攒着）。正文末尾带标注「（跟别人说：在回小明）」。
  `[proactive] enabled = false` 时 `other` 一律 `aside_bg`。日志（INFO）一行 `跟谁说 …`，评估能原样回放
- **插话走主动开口的护栏**：`Brain.chat_turn` 只看 `chat` / `owner_command`，整批都是 aside 的那一轮 `say` 算主动开口（额度、`min_gap`、连着没人接就停、没熟人不说）；有一句 `me` / `all` / `unsure` 就算接话、不拦
- **别的地方跟着改**：反射（输入气泡、被叫到的小动作）、关系卡「跟团子说的几句」、`_cheered_at` 只认 `me`；status 「场合」后多一行「小明 和 阿花 在聊（1 分钟内 6 句）」；聊天记录（真机 / 沙盒共用）行尾带标注，沙盒的 heard 行改由身体写
- **提示词**（`brain/prompt.py`「说话」一节）：按标注写接不接的规矩（跟别人说默认不接、可以插一句；拿不准宁可不接；有人说先别回就照做；不接就心里写“不说：…”）；OpenAI 兼容大脑的 `ASIDE_NOTE` 再提醒一次
- **`enabled = false` 逐字照旧**（不判、没有 aside、提示词和事件正文不变、反射用 `legacy_addressed`）；管理面板设置页有 `addressee.enabled`
- **离线评估**（数据在 `datasets/addressee/<时间>/`，不进 git，里面是好友原话）：
  `python -m skydango addressee label <runs…> [--out 目录]` 取各次运行 `agent.log` 的聊天、挑多人聊天段落 + 约 20% 单人对照 → `lines.jsonl`，Claude 初标（花额度，额度用完可续跑）→ `claude.jsonl`，规则重放，写 `review.md`（不一致的全列、一致的抽 20%，在 `标：` 后填 跟团子 / 跟别人 / 大家 / 看不出；重跑保留已填的标）；
  `python -m skydango addressee eval <目录>` 读回人标 → `report.md`（混淆矩阵 + 判错的句子）；三条门槛：跟别人说被判 me ≤ 5%、跟团子被判 other ≤ 10%、unsure ≤ 40%；没有足够人工核对的标准答案、或没过线退出码 1。
  改了规则不用重标，eval 用当前规则重放。数据来源建议 10-03 21:02、23:04、23:25、10-04 21:27 那几次多人聊天
- 沙盒剧本 `docs/sandbox-scenarios/两个好友互相聊.toml`（占位名，换成 friends.md 里的好友名）；真机验收：两个以上好友在的一晚，抽查「跟谁说」行、数只有 aside 的那几轮团子开了几次口、问好友「怎么什么都接」没有；另外数一数「在接你的话」→ 团子回 → 又「在接你的话」的连锁（规则 4 会自己续命，`occasion.reply_state` 又把它当「有人接」，插话护栏不触发；要收紧可选：插话那一轮之后不开规则 4 窗口 / 规则 4 只给团子上一句回应的那个人 / `reply_state` 跳过判成 other 的句子）

## 身体反射（`[reflex]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-30-body-reflex-design.md`，计划 `docs/superpowers/plans/2026-09-30-body-reflex.md`。
"让团子活着"四个子项目的第一个（反射 → 关系账 → 情绪 → 性格语气，见 spec 开头）。大脑一轮要几秒到十几秒，常见时刻由身体按规则当场反应，大脑再补上说什么。
**已在真机 live 里默认开着跑过（9-30 起），spec「真机验证」清单没逐条走，数字都是估的**（spec「真机验证」四步）。
- **判断在跟团子说话**：10-05 起和「分清在跟谁说话」共用同一个判断，只对 `跟你说`（`me`）冒气泡 / 做小动作；`followup_window` 挪到了 `[addressee]`（`[reflex]` 里写着还生效、启动时警告）。`[addressee] enabled = false` 时才用原来的 `legacy_addressed()`（`brain/addressee.py`，原 `reflex.py` 的 `addressed()`，先从严）：好友说的，且叫了 `[proactive] self_names`、或团子说完 `followup_window`（30 秒）内、或身边只有他一个好友；陌生人、`#` 命令不算
- **输入气泡**：身体读到这样的消息当圈就 `sender.open()`（先 `panel.before_speak()`），头顶冒"正在输入"；大脑 `say` 直接用这个框发。
  关框：大脑"开框之后才开始的那一轮"结束了没说话（`Brain.last_turn` → `body.brain_turn`）、开了 `bubble_max`（45 秒）、**任何按键 / 点屏幕之前**（`clear_view` 里统一关，`say` 除外；`panel_press` / `panel_close` 也关）、有互动请求要接（`clear_view("social")`）。
  只关自己开的框（`sender.opened`）；技能（track）在跑、大脑离线、dry-run 时不开；status 里"输入框：开着（身体替你开的…）"
- **动作反射**（只用轮盘上已有的动作，不换轮盘）：被叫到时开框前按 `addressed_chance` 做 `addressed` 清单里的一个；感知层的 `gesture` 按 `return_chance` 用 `return_map` 回礼（回了就只发 `reflex` 事件、不再叫大脑决定，没回照旧发 `gesture`；框开着就回完再开回来）；
  闲着 `idle_min`~`idle_max` 秒（随机，有人说话 / 团子说话 / 任何按键类工具都重新计时）做 `idle` 清单里的一个（框开着不做）。**三张清单默认空**，用户按自己的轮盘填
- **额度**：反射**不占**大脑的动作冷却（`EmotePlayer.perform(name, reflex=True)` 只更新 `last_any`）；自己 `quota_window` 里最多 `quota` 个；任何两个动作之间至少 `min_gap`（4 秒，大脑的 `emote` 也查）
- 做完放 `reflex` 背景事件（不单独叫醒大脑），status "刚才下意识：…"、网页"反射"；提示词在「说话」一节多一句"身体已经替你冒了输入气泡"。黑屏、技能在跑、有互动请求、面板挡着时不做；dry-run 走 `pretend`
- `enabled = false` 完全照旧（不开框、不做动作、`gesture` 照旧交给大脑、提示词不变、`emote` 不查 `min_gap`）；管理面板有 `reflex.enabled` / `reflex.bubble` 两个开关

## 按 Q 喊一声（`[call]`，大脑模式，要配合 `[perception]`）

设计见 `docs/superpowers/specs/2026-10-01-q-call-design.md`，计划 `docs/superpowers/plans/2026-10-01-q-call.md`，操作见 game-ops「呼喊找好友」「呼唤特效」。
起因：好友稍远一点头顶的名字标签就淡掉（人还在画面里），只有按 Q 才亮出来约 5 秒；以前标签一淡就被判成陌生人、5 秒后"走开了"。**10-02 / 10-03 晚真机跑过：手动喊通过，自动兜底不勤了（10-03 晚自动喊 4 次、`auto_again` 挡了 2 次），但只认回 1 次；数字都是估的**（spec §6 真机验证六步没逐条走）。
- **轨迹续命**（`[perception] sticky_names`，平时就生效）：挂过名字标签的 `player` 轨迹只要这一帧还接得上就刷新"在身边"；轨迹断了（`track_buffer`）才开始算 `keep` 5 秒；只靠低分框续着的最多续 5 秒（`LOW_ONLY_MAX`，见「追踪和接回」）。按外观认的 `maybe`、黑影不续
- **贴边标签**（`[perception] edge_band = 0.06`）：名字标签贴在屏幕最左 / 最右、又没挂上人 = 好友在画面外：**不算在身边**（不刷新 `last_seen`、不挂圆圈），
  但 `labels` 照记（盯人 track 靠它往画面外转——和 spec §1.3 不同，见计划「偏离 spec」）；`edge_band = 0` 照旧
- **呼喊窗口**（`PerceptionWatcher.called(at)` / `call_result(at)`）：按键后 `[call] window`（6 秒）里挂上名字的人记方位 / 远近、贴边的记"在画面外"，窗口里 `far_crops` 不退避、块数 ×2；
  结束那一帧数还剩几个没挂名字的点过火的人（`unnamed`）；暂停时窗口跟着往后挪。EnvWatcher 是空实现
- **身体喊一声**（`Body.call_out(reason)`）：没开 / 没感知层 / 黑屏 / `min_gap`（20 秒，卡洛 `#` 命令窗口里大脑不受限）/ dry-run（不按，照样计间隔）→ 不按；
  `clear_view("call")` 关掉身体开的输入框，`ime_shown()` 还开着就拒绝（Q 会变成打字）；`panel.borrow("call")` 里拍基准、`hw_key(16)` 短按、连拍 `burst` 1 秒、`env.called(at)`
- **光圈认团子**（`[call] halo`，**默认关**）：连拍里按键（命令发出前记的时间）后 0~0.8 秒头顶区域比基准亮 `halo_rise` 以上、恰好一个人、在画面中间（`halo_center`）→ 写 `env.self_box`（YOLO 已经认出一个团子就不覆盖；和 self 框重叠的 player 框算同一个人）；
  别人也在喊 / 镜头刚动过 / 喊之前聊天面板开着（关面板时画面横移）/ 没有新鲜的人物框 / 黑屏就放弃，**不为确认再按**。认错 = 把一个好友当成团子过滤掉，10-03 跑了 `perception halo-eval tmp/record/q-call-20260930-c`，信号太弱定不出门槛（团子光圈只升 7~8、噪声 6.5），继续关着，要换算法
- **自动兜底**（`[call] auto`，`Body._watch_call`）：好友 `auto_after_leave`（30 秒）内走开、还没回来、这次走开没为他喊过，画面里有没挂名字的人，`auto_window` 1 分钟最多 `auto_quota` 3 次（10-02 晚从 10 分钟改的；还受 `min_gap` 20 秒限制）；喊一声认回来的好友 `auto_again`（5 分钟）内不再为他自动喊（10-03，他多半一直站在稍远处、标签淡了而已）；
  输入框开着、技能在跑、有互动请求、在举蜡烛、别的面板开着、刚做完动作、大脑在回聊天、黑屏都不喊；dry-run 只记日志。喊完不等，窗口结束后放**背景事件** `call`（"你下意识喊了一声：认出 小明（右边·远）…"），
  认回来的好友照常 `return` 抵消那条 `leave`。
  **第二种起因：拿不准**（identity-gallery spec §5）：认装扮里某个人是"可能是小明"（`env.unsure(now)`，持续 `unsure_wait` 秒、小明没在别处确认）时也自动喊一声，**共用**上面的 `min_gap` / `auto_window` / `auto_quota` / `auto_again` 和所有拦截；额度 / 间隔 / `auto_again` 挡住不算尝试（他留在名单里、下一圈还能喊，受"可能是"超时约束），拦截（输入框、技能……）/ dry-run / `call_out` 拒绝才打 `unsure_called`（喊成没喊成都打，免得反复排队）；喊完窗口结束时按标签有没有亮出来判（见「认装扮」）。注意力的找人接管"刚走开的好友"那一声时，这个起因照旧在 `_watch_call` 里喊。空闲注意力的找人开着时（`[attention] search`，模式随意 / 好奇 / 专心，面板 auto），好友走开后的这一声改由「找刚走开的好友」当一步去喊（额度同上），`_watch_call` 不再自己喊
- **大脑工具 `call()`**（`calling.call_available`：`[call] enabled` 且 env 是感知层才注册，在 `look_around` 后面，算"做了事"）：身体按完键就回来，MCP 线程里等窗口结束（最多 `window + 4` 秒），
  返回"喊了一声：认出 …；小红在画面外（左边）；还有 1 个没挂名字的人。光圈：…"；拒绝时返回原因。提示词「视角」一节加一句什么时候喊；status 多一行"上次喊：2 分钟前（认出小明）"；
  网页手动控制"喊一声（Q）"（总是真执行、照样过检查）；管理面板有 `call.enabled` / `call.auto`
- `enabled = false`：没有工具、不自动喊，提示词 / status / 工具列表逐字照旧（续命、贴边是 `[perception]` 的开关）

## 冷场时的心理活动（`[lull]`，大脑模式）

设计见 `docs/superpowers/specs/2026-10-01-lull-musing-design.md`，计划 `docs/superpowers/plans/2026-10-01-lull-musing.md`。
起因：沙盒里好友安静下来后团子一直不说话，大脑只写“不说：等卡洛回”，心里太平静。**已在真机 live 和沙盒里默认开着跑过（日志里见过冷场叫醒），spec「用真 Claude 在沙盒验证」四步没逐条走，数字都是估的**。
- **两种冷场**（`inner/lull.py` 的 `LullTracker`，纯计算、墙钟）：① 好友在身边不说话了：最后一句（谁说的都算）过去 `stages[0]` 秒、这句之前 `talk_window` 秒内有好友说过话、他还在身边；
  ② 聊着聊着走了：好友走开时他 `leave_spoke` 秒内说过话、或团子 `leave_said` 秒内说过话。说话人按 `similar(…, 0.75)` 模糊匹配（OCR 错字），看不出是谁的不算好友
- **节点**：`stages`（60 / 180 / 360 秒）各放一次立刻叫醒的 `lull` 事件（不在 `BACKGROUND` 里，不受心跳退档影响）；② 走开先照旧发背景 `leave`，`leave_grace`（15 秒）还没回来才叫醒一次（名字标签闪一下不算，宽限内回来 `leave` / `return` 照旧互相抵消），之后用后两个节点；
  最后一个节点之后再过 `stages[0]` 秒冷场结束（“后来就一直安静着” / “一直没回来”），不再挂在 status 里；快进跳过几个节点只发最新的；黑屏时不推进；
  冷场中团子又说话不重新计时，只在文字里写“这之间你又说了…”；跟踪技能在跑时不判 ②（`_watch_comings` 本来就不跑）
- **结束**：任何好友开口（附在那条聊天事件后面：“（冷场了 2 分钟，你刚才在想：…）”）；② 的人回来（叫醒过的：原来的背景 `return` 换成 `lull` 事件“X 回来了（走开了 3 分钟，你刚才在想：…）”，超过 `rejoin` 的 `arrive` 也附上）；
  ① 的对象聊着聊着走了，想过的话转给 ②；冷了很久才走（不算聊着聊着）或镜头转开，① 结束、留下总结“后来他走开了”。冷场追踪和人来人走用同一份身边名单（`_lull_near`）
- **心里想的**：提示词「冷场的时候」要求被冷场叫醒时在最后的文字里写一行“心里：……”；`Brain.on_text` 每轮把最后的文字交给 `body.mused`（经 `body.call`），`parse_musing` 取最后一行、截 `musing_max` 字，
  只挂到大脑这一轮开始之前就有、已经叫醒过的冷场上（`brain.last_turn[0]`，冷场在这一轮里结束了就丢掉）；不加工具。status 多一项“冷场：懒洋洋大王 3 分钟没说话（最后是你说的「…」）· 在想：…（1 分钟时）”。
  新的一次冷场还没想过时，叫醒的那句末尾附“之前冷场时你想过「…」「…」，这次换个想法。”（最近 2 句，`EARLIER`；沙盒里每次都想“他在忙，我等等”）
- **留下的影响**：冷场结束（或下线前最终反思时还没结束）写成一行、说话人“（冷场）”，按冷场开始的时间插进反思材料（`_reflect_chat` / `_session_chat`）；冷到最后一个节点算一次 `reflector.stirred`；代码不直接改心情，由反思定
- **看得到的地方**：status、沙盒聊天记录旁白“── 心里：… ──”（`body.on_musing`）、内心页「现在」的“在想”（`inner_snapshot()["musing"]`，只有实时）、流水账 `musing` 行（`MindLog.musing`，live 才写盘，内心页反思记录里列出）。
  **不写** `history.jsonl`、随手记、关系卡
- **顺带修了**：被 `arrive` 之类叫醒的一轮里接上一句没回的好友聊天（`proactive.reply_window` 内、团子之后还没说过，`Body._pending_reply`）不再算主动开口；
  「主动开口」的“分寸”写清“没人接”只看状态里“上次主动开口”那一行（这一句改动 `enabled = false` 时也在）
- 不依赖内心层（`mind` 为 None 时照样叫醒、记“在想”，只是不进反思材料和流水账）。`enabled = false`：不认冷场、`leave` / `return` 照旧、提示词和 status 照旧；管理面板有 `lull.enabled` 开关

## 内心层（`[inner]`，大脑模式）

设计见 `docs/superpowers/specs/2026-09-30-inner-phase1-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase1.md`。
"让团子活着"的内在部分，统一放在一个内心层里，分三期：**1 账本**（这一期：关系卡 + 日子，不调模型）→ 2 反思（心情、精力、心愿、日记，接上后果）→ 3 性格（口头禅、老梗、放开太乖的规则）。
代替了反射 spec 开头列的子项目 2~4。**已在真机 live 里默认开着跑过（9-30 起），spec「真机验证」清单没逐条走，数字都是估的**（spec「真机验证」五步）。
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

**第 2 期：反思**（设计见 `docs/superpowers/specs/2026-09-30-inner-phase2-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase2.md`；**已在真机 live 里默认开着跑过（9-30 起），spec「真机验证」清单没逐条走，数字都是估的**，spec「真机验证」四步）
- **精力**（`energy.py`，身体每圈现算，不存盘）：几点（白天满、深夜低）− 连着挂了多久（离上次下线不到 `rest_gap` 算没睡、接着累）+ 最近 10 分钟有好友跟团子说话 − 最近一小时热闹太久；四档 精神 / 还行 / 有点累 / 困
- **反思**（`reflect.py`，`[models.reflect]`（默认 DeepSeek、备用 Claude sonnet），后台线程，结果回身体线程套进 `Mind`）：有动静时每 `reflect_every`（20 分钟）、好友说够 6 句后安静 3 分钟、下线时各一次；
  材料是上次以来的聊天原话、来去、相关好友的关系卡、friends.md 里他们那一节（日记里的称呼、性别照它，`friend_sections`）和笔记、人设、精力。回 JSON：心情（开心 / 平常 / 低落 / 烦 + 一句带原因的话）、别扭、心愿增删；下线那次再加日记和要点。主那家撞墙改走备用（见「统管大脑」的闸），都不行沿用上一份
- **代码守的规矩**（`mind.py`）：别扭只冲一起玩过 ≥ `grudge_min_days`（3）天的好友、最长 `grudge_max`（2 小时，同一个人再给也不续期）、同时 1 条、消气后冷却同样长；别扭对象说了“难过 / 不舒服 / 想哭 / 认真的……”（`DISTRESS`）身体当场撤掉，不等反思；离上次反思超过 `rest_gap`（睡过一觉）心情回到平常；心愿最多 3 条、7 天过期，kind 只有 惦记（挂在好友名下）/ 想做 / 小心思；不认识的一律丢
- **后果**（`effects.py`，倍数相乘）：主动开口额度 开心 ×1.5、低落 / 烦 ×0.5、有点累 ×0.75、困 ×0.5（下限 ×0.25、至少 1 句）；烦时被叫到不做小动作、开心概率 ×1.5；困时闲着的小动作更勤、大脑心跳慢一档。
  **接话永远照常**；在跟谁闹别扭时，整批都是他说的话不冒输入气泡（故意晚点接），牵手等互动照接
- **大脑看到**：status "心里：有点闷（…）· 有点困（…）· 跟小明闹别扭（…，还有 40 分钟消气）· 惦记：…"；arrive 后面接"你惦记着：…"；「日子」带上一篇日记（代替"上次的经过"）；提示词多一段心情 / 别扭（认真问、说难过时立刻作废）/ 惦记 / 小心思的规矩；网页和管理面板卡片有"心情""精力"
- **下线**：`checkpoint` → 最终反思（材料是这次上线的全部聊天，最多 80 行；超时压到 `console.stop_timeout − 25` 秒）写日记（`diary.md`，每次下线一段，「日子」取最后一段、带日期）和要点（inbox.md、`days.jsonl` 的经过）→ `close`，**不再让大脑写经过**；`reflect = false` 照第 1 期
- 只在 live 写 `mind.json` / `diary.md` / inbox；dry-run 照样反思（管理面板里看得到），不写盘；管理面板有 `inner.reflect` 开关

**第 3 期：性格**（设计见 `docs/superpowers/specs/2026-09-30-inner-phase3-design.md`，计划 `docs/superpowers/plans/2026-09-30-inner-phase3.md`；**已在真机 live 里默认开着跑过（9-30 起），spec「真机验证」清单没逐条走，数字都是估的**，spec「真机验证」五步）
- **性格档案**（`persona.py`，`memory/inner/persona.json`）：口头禅（`catchphrases_max` 5）、和某个好友的老梗（每人 `jokes_per_friend` 3、总共 `jokes_max` 20）、看法（一个话题一句，`opinions_max` 10）。
  反思（`persona` 开着时系统提示词接上 `PERSONA_SYSTEM`、材料带「你攒下的性格」）顺带给 `persona_add` / `persona_used`；**代码守的规矩**（`Persona.apply`）：提示词要 30 字（话题 10 字），超一点整句收下，超过 40 字（话题 15 字）整条不记、不截半句、带敏感词（胖瘦丑矮、长相身材脸、爸妈家里、成绩考试分数作业老师、几岁年纪年龄今年岁、学校班小学生初中高中、本名 QQ 微信、哭）/ 难过类的话（`DISTRESS`）/ 声称是真人的话（`claims_human`）的丢；口头禅和看法里**不许提好友名字**（人只进老梗）、
  老梗只挂在一起玩过 ≥ `grudge_min_days` 天的好友名下（OCR 错字按最像的好友归）、收着点的人不记新老梗、几乎一样的不重复加、同话题换立场；超上限只在旧条目里挤（这次新记的不挤）：先删 `hits` 最少的、再删最久没用的；换了立场算刚用过；`fade_days`（14 天）没用就淡出（启动时和每次套完）；启动时 `prepare` 按现在的规矩把旧条目再筛一遍、手写的 `since = 0` 当作现在记的
- **大脑看到**：系统提示词「你攒下的性格」（启动时算一次，放在「日子」之前，空的不写）；`arrive` 在“你惦记着”之后接“你们的老梗：…”（最多 2 条）；
  提示词多一段「脾气」（`TEMPER_RULES`：有立场、熟人互损但损事不损人、会拒绝会偷懒但卡洛的 `#` 命令和有人真难过时不偷懒、有执念、对“收着点”的人好好说话），“接对方的话往下聊”换成“接得住就接，不想接也可以吐槽一句或者敷衍两句”；底线各节一字不动
- **收着点**（身体兜底）：任何好友说了难过类的话（第 2 期的 `DISTRESS`）→ `soft_minutes`（30 分钟）内 status 写“对小明收着点（他刚说「…」）”、他来时 arrive 不带老梗，反思不给他记新老梗（下线那次反思看整次上线：这次上线里说过难过的都算）
- `memory init` 的人设模板多了「## 脾气」（毛病 / 执念 / 雷点）；已有的 `memory/profile.md` 要自己加。`memory show` 末尾打印性格档案（只读），`profile.md` 永远优先、程序不改
- 只在 live 写 `persona.json`；dry-run 只在内存里，坏文件不改名。`persona = false` = 第 2 期原样（提示词、反思提示词和材料逐字一样，不记收着点）；管理面板有 `inner.persona` 开关
- 还是太乖：旧的乖回复会把模型拉回去，按「记忆」一节把 `history.jsonl` 挪到 `memory/archive/` 再试

## 幕后（`[backstage]`，大脑模式）

设计见 `docs/superpowers/specs/2026-10-01-backstage-design.md`，计划 `docs/superpowers/plans/2026-10-01-backstage.md`。
像 Neuro-sama 和 Vedal：团子知道自己是 AI、卡洛（`[brain] owner_name`）做了她，能跟卡洛聊自己怎么运作。**代码默认关**（公开仓库）；本机 `config.toml` 开着，10-01 真机试过、10-03 晚备用大脑也调过 `introspect`（spec §7 五步没逐条走）。
- **提示词**：「## 身份」整节换成「## 幕后」（`brain/backstage.py` 的 `section`）：事实（脑子 / 眼睛 / 反思的模型名取配置，身体、记忆文件、沙盒是什么）+ 三档：
  卡洛（名字一字不差）完全出戏、能聊深的、半开玩笑的存在主义；**知情好友**（friends.md 他那一节写「知道团子是 AI」，代码不解析、模型自己读）大方承认是 AI、开玩笑，不聊深；其他人照原来的「身份」规则。
  `owner_name` 为空时没有卡洛那一档。「底线」一节、`_CLAIMS_HUMAN` 硬过滤不变；陌生人看不到好友之间的聊天，所以不管旁人
- **`introspect(topic)`**：只读工具，topic = 精力（逐项加减，`inner/energy.py` 的 `energy_parts`）/ 反思（流水账最近 3 次 + 下次大概几分钟后）/ 性格（用过几次、几天后淡出）/ 日记（最近一篇，正文截 600 字）/ 眼睛（`Eyes.latest` 原文）/ 改动（不看标记、按最近 `changelog_days` 天取提交，`cli._recent_changes`：提示词里的更新记录只给没告诉过的，再问就靠这个查）；
  开关打开才注册（`ToolBox(backstage=True)`），不算"做了事"；对应的内心层没开就回"没开"
- **更新记录**：启动时取 `inner/backstage.json` 的 `seen..HEAD`（没有 / 不在历史里就按最近 `changelog_days` 天），`--no-merges`、只要 `feat` / `fix` / `perf`、scope 为 `console` / `viewer` 的不要，最多 `changelog_max` 条，提交标题原样接在「幕后」末尾由她自己转述；
  **只在 live 写标记**（dry-run 每次看到同一批），沙盒记在沙盒记忆目录；git 出错 / 超时 2 秒只是这一小节为空
- `enabled = false`：提示词、工具列表逐字照旧；管理面板有 `backstage.enabled` 开关

## 大脑沙盒（`[sandbox]`，`sandbox` / 管理面板「沙盒」页）

设计见 `docs/superpowers/specs/2026-09-30-brain-sandbox-design.md`，计划 `docs/superpowers/plans/2026-09-30-brain-sandbox.md`。
不开 MuMu，让**真的大脑 + 身体 + 内心层**接一个假世界：冒充好友说话、造来去、快进时间，看团子的反应和心里；调性格和提示词用。
**已用真 Claude 跑过，但 spec「验证」四步（见下）没逐条走**。
- **拆出世界**（`brain/world.py`）：`cli._run_brain(cfg, run, world=None, …, on_ready, trace, *, no_emotes)` 只管组装；`World` 给设备、读聊天、身边、轮盘、说话、走路……和 `clock` / `wall`。
  真机是 `cli._game_world`（原样搬的，`world=None` 时在检查完令牌之后才建），沙盒是 `sandbox/world.py` 的 `sandbox_world`。身体、事件队列、眼睛、反思器、大脑循环、账本都用 world 的钟
- **沙盒世界**：模拟时钟 `SimClock`（只往前拨；起始时间不早于 `clock.json` / 沙盒 `days.jsonl` 最后一次下线）、中灰截图（不算黑屏）、队列读聊天、名单当身边、手写场景当眼睛（空 = 看不清，不调 Haiku）；
  说话 / 动作 / 走路 / 气泡都写进沙盒聊天记录（`brain/transcript.py`，和真机共用）；没有镜头、好友树、面板、互动请求（工具回"沙盒里没有这个"）；`look` / `look_person` 只给文字（`World.text_only`）
- **总是 live，但只写 `sandbox/memory/`**（`[sandbox] dir`，整个 `sandbox/` 不进 git）：真的 `memory/` 永远不碰（端到端测试比 sha256）；
  沙盒记忆目录和 `reply.memory_dir` 重合（相同或互相包含，比如 `dir = "."`）时，重置和启动都拒绝。`--duration` 按真实时间，快进不会提前下线。随手记、反思、日记都真的调 Claude，和真机一样花额度
- **子进程** `python -m skydango sandbox --port 19392 [--start resume|sleep|HH:MM|"YYYY-MM-DD HH:MM"]`：只给 JSON 接口（`sandbox/server.py`：`/status` `/state` 长轮询 `/op` `/brain` `/inner` `/inner/forget` `/shutdown`，本机 Host + `post_guard`）；
  操作经 `body.call` 在身体线程做（`sandbox/control.py`：冒充发言、来去、陌生人、地名、场景、新鲜事、快进、拨时间、立刻反思）；下线 = `/shutdown`，走最终反思 → 日记 → 合账本，再存 `clock.json`
- **管理面板**：子进程槽带 kind（团子 / 沙盒），**同一时间只能有一个**（共用令牌和 `.brain-claude/`），另一个在跑时拒绝并提示先停；`/sandbox/*` 转发、`/live/*` 只在团子时转。
  「沙盒」页：顶栏启动选项（接着上次 / 睡一晚 / 自定义）或沙盒时间牌 + 快进；三栏从左到右 团子（现在、身边、场景和新鲜事）/ 大脑控制台 / 聊天记录。
  聊天记录团子说的在左（樱花底）、冒充的人在右，动作旁白、事件分隔线、被拦的删除线 + 原因；
  大脑控制台（`console/static/brainlog.js`，`/brain` 数据）是终端排版的日志（底色同左右两栏、等宽字）：轮头（时间 · 原因 · 做了什么 · 耗时 · tokens）、收到的事件一行一条、状态 / 场景折叠、`▶` 工具调用、`↳` 返回，新的一轮在底部、自动跟到底；
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
- **spec「验证」四步**（用真 Claude 跑过，但没逐条对着走）：
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
python -m skydango run [--live | --dry-run] [--duration 秒] [--no-emotes]  # 团子（默认接统管大脑、dry-run）；模型按 [models.*]（默认要 DEEPSEEK_API_KEY，用到 Claude 的才要 SKYDANGO_CLAUDE_TOKEN）；--duration 到点自己退出
python -m skydango run --no-brain [--echo] [--live]  # 调试用的普通 Agent（[models.reply] 回复）；--echo 回复不调模型
python -m skydango panels scan [图片或目录]    # 面板识别：每张卡开没开、每个特征的分数 + 通用兜底，标注图 tmp/panels/（不发输入）
python -m skydango panels read [图片]          # 细读开着的面板：标题、正文、按钮和类别
python -m skydango panels cut <截图> <卡片> <文件名> --roi x1,y1,x2,y2  # 裁模板图存进 assets/panels/<卡片>/
python -m skydango look [图片] [--prompt 文件]  # 截一张图（或给一张截图）让眼睛（[models.eyes]）描述，先打印交给它的位置说明
python -m skydango camera spin [--turns N] [--seconds S]  # 转一圈、边转边截图存 tmp/spin/<时间>/（标定一圈几秒、视野角）
python -m skydango friend-check X Y       # 点一下人物打开好友树、截图、再关掉（核对面板和关法），图在 tmp/friend-check/
python -m skydango memory init|show|update # 记忆：生成人设 / 好友文件、查看、立刻整理
python -m skydango env                    # 对当前画面识别一次环境（身边有谁、在哪）
python -m skydango a11y [--watch 秒] [--all] [--save 文件]  # 读游戏的无障碍节点：聊天行、头顶名字等 UI 文字 + 坐标（不发输入）；--save 存原始快照给回放测试（配 --watch）
python -m skydango ime on|off|status      # 输入法：切到 ADBKeyboard / 切回搜狗（或 [device] user_ime）/ 看当前
python -m skydango record [--seconds 60] [--fps 5] [-o 目录]  # 连续截图存 tmp/record/<时间>，观察界面变化、录训练素材
python -m skydango perception bench [--model yolo11n.pt] [--images 目录]  # YOLO 测速（没训练前用官方模型看硬件）
python -m skydango perception detect [图片]  # 跑一遍 YOLO 感知，标注图 tmp/perception.png
python -m skydango perception label <录像目录> [--preview] [--model 模型]  # 弱标注（+ 模型预标注）→ datasets/sky（YOLO 格式）
python -m skydango perception label <录像目录> --assist [--model 模型] [--all-frames]  # Claude 辅助标注：挑帧 + 人物框由 Sonnet 核对，清单在 datasets/sky/_assist/review.md（令牌同 [brain]）
python -m skydango perception label runs --from-runs --model 模型  # 把各次运行存下的难例收进数据集
python -m skydango perception inbox collect|process|status  # 难例收件箱：collect 把 runs/*/hard 收进 [inbox] dir；process 用 YOLO + 外形头先筛（模型缺了就报错退出）；status 看各阶段数量
python -m skydango perception inbox add <录像目录> [--every N] | datasets/sky --redo  # 收件箱：导入录像 / 老帧回炉（备份 + 撤销），整图火焰补圆圈框
python -m skydango perception icon-eval <截图目录或数据集> [--model …] [--labels 目录] [-o 输出]  # 地图交互图标：比模板和 DINOv2 两种分类器 → tmp/icon-eval/<时间>/
python -m skydango perception icon-cut <图片> <种类> --box x,y,w,h [--template]  # 裁图存进 [icons] refs（assets/icons）/<种类>/，--template 再存一份模板到 [social] icons_dir（assets/social）
python -m skydango perception retrain [--epochs N]  # 一键重训 YOLO + 外形头并回放对比 → tmp/retrain/<时间>/report.md（不改配置，换上去管理面板整帧页）
python -m skydango perception label <spin 目录> --spin --model 模型  # 转圈录像：认出团子，每帧自动补 self 框
python -m skydango perception label datasets/sky --objects [--model 模型] [--only 通配]  # 物品模式：给已标好人的数据集补标座位 / 篝火 / 乐器 / 先祖和头顶气泡 typing（先备份 labels/），清单在 _assist/objects.md；--only 只做文件名匹配的帧
python -m skydango perception augment datasets/sky  # 训练集加运动模糊 / 压暗样本（只动 train）
python -m skydango perception compare <录像目录> [--model 模型] [--far-crops 0]  # 同一批录像对比 YOLO 和整图 OCR → tmp/compare/<时间>/report.md（含远处认出率）
python -m skydango perception halo-eval <录像目录> [--model 模型]  # 呼唤光圈标定：每个人头顶的亮度曲线、建议的 [call] halo_rise → tmp/halo-eval/<时间>/report.md + curves.png
python -m skydango perception track-eval <录像目录> [--model 模型] [--fps 6.5]  # 追踪升级前后对比：断开和原因、假走开、确认冤枉、接回对错、运动方向 → tmp/track-eval/<时间>/report.md
python -m skydango perception crops <来源...> [--model 模型] [--conf 0.2] [--out datasets/attrs]  # 第二层的数据：数据集 / 难例目录（支持 runs/*/hard）/ 图片目录 → 人物裁图，已标注的按类别放，对不上的进 _unlabeled/
python -m skydango perception crops datasets/sky --writeback  # 标注页确认过的写回 datasets/sky 的 labels/：补漏标的人、改点没点火、删确认不是人的框（「不要」和没确认的不动），增强图的标注副本一起改；先备份到 <数据集>/_backup/
python -m skydango perception attrs-label [datasets/attrs] [--recheck]  # Claude 初分 _unlabeled 裁图（花额度），再去管理面板「标注」页「外形」确认
python -m skydango perception attrs-train [datasets/attrs] [--out 路径] [--force] [--device cuda|dml|cpu] [--all] [--no-mask]  # 训练外形头 → models/attrs-<日期>.npz + 报告（是 [attrs] model 时要加 --force）；默认只用标注页确认过的，--all 连 datasets/sky 导进来没确认的也用；--no-mask 按旧裁法（不遮挡框外）
python -m skydango perception attrs-eval datasets/sky --model models/attrs-<日期>.npz [--attrs-data datasets/attrs]  # 外形头在 YOLO 数据集验证集上的评估（答案按标注页确认过的修正）
python -m skydango perception bench --attrs [--model …]  # 测速时再测一遍加第二层后的 fps
python -m skydango perception unknown-names [--last 5]  # 最近几次运行里读到、但不在好友名单里的名字（只列出）
python -m skydango addressee label <runs目录...> [--out datasets/addressee/<时间>]  # 分清在跟谁说话的离线评估：取 agent.log 的多人聊天、Claude 初标（花额度）、规则重放，写 lines.jsonl / claude.jsonl / review.md；去 review.md 的 `标：` 后填人工答案
python -m skydango addressee eval <datasets/addressee/时间>  # 读回人标，出 report.md（混淆矩阵 + 三条门槛）；没过线或标准答案不够退出码 1
python -m skydango perception appearance-eval <录像目录> [--model YOLO模型] [--embed color|模型.onnx]  # 认装扮离线标定：同一个人 / 不同人的相似度、建议的 match / changed、藏标签重放 → tmp/appearance-eval/<时间>/report.md
python -m skydango perception appearance-eval <录像目录> --gallery  # 底库模式：前一半挂标签的当底库、后一半当查询，开头 20 秒的团子框当团子底库，给 match / unsure / dango_match 的建议值和三档人数分布（还没在录像上跑过，先跑 tmp/record/walkaway-1002-1）
python -m skydango catalog collect <录像目录> [--model 模型] [--fps 6.5]  # 装扮图鉴：录像上试跑收集 → tmp/catalog/<时间>/（sheet.jpg、candidates.jsonl），定 min_height / sharp_min
python -m skydango perception clips <录像目录> [--force]  # 动作识别的数据：按人物轨迹切 16 帧片段 → datasets/gesture/_unlabeled；这段录像切过就拒绝，--force 只切数据目录里哪儿都还没有的片段
python -m skydango perception gesture-label [片段目录] [--blind]  # Claude 初分动作片段（默认 datasets/gesture/_unlabeled），再去管理面板「标注」页确认；--blind 不给录像名提示、写 claude-blind.json（标注页优先显示），看名字时它常照名字判
python -m skydango perception gesture-train [数据目录] [--epochs 60] [--out 路径] [--device cuda|cpu]  # 训练动作模型 → models/gesture-<日期>.onnx + tmp/gesture-train/<时间>/report.md（旁边复制一份 _split.json）；--out 是 [gesture] model 时要加 --force
python -m skydango perception gesture-eval datasets/gesture --model 模型 [--all]  # 动作模型的精确率 / 召回率（有 _split.json 时只评验证集，模型比切分旧会提醒）
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
