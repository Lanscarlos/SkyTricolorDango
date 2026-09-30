# 大脑沙盒（管理面板「沙盒」页）— 设计

日期：2026-09-30　状态：**代码已完成，待真 Claude 验证**（实施计划 `docs/superpowers/plans/2026-09-30-brain-sandbox.md`；只用 fake_claude 在浏览器里核对过，「验证」四步还没用真 Claude 跑）

## 背景

内心层三期（账本、反思、性格）都**还没在真机上跑过**，而它们只关心对话：谁说了什么、谁来了走了、现在几点。
在 MuMu 上验证又慢又贵（要开游戏、要有好友在线、要熬夜才能看"深夜犯困"、要等 20 分钟才反思一次）。
沙盒让团子的**大脑 + 身体 + 内心层**原样跑起来，只把"接世界的那一圈"换成假的，在管理面板上冒充好友聊天、造来去、快进时间，调性格和提示词。

**测不到的**（仍要真机）：名字标签闪烁时 `arrive` / `visits` 乱不乱、读错说话人时别扭记到谁头上、动作真的做出来没有、真实节奏下的额度。沙盒的输入是干净的。

## 需求（已和用户确认）

1. **用途**：主要给用户自己调性格 / 提示词（看得透：心里、大脑时间线）；顺带能把一段对话录成剧本回放
2. **记忆**：沙盒有自己的一份（`sandbox/memory/`），能一键从 `memory/` 重置；真的 `memory/` 永远不被沙盒写
3. **时间**：模拟时钟，能快进（只往前），反思 / 别扭消气 / 额度 / 精力都跟着走
4. **画面**：不用截图、不调 Haiku；团子"看到"的是手写的场景文字，空着 = 看不清；新鲜事手动放
5. **剧本**：能录能放、出报告、人来看结果；格式留 `expect` 字段，这一版不检查
6. **界面**：子进程只给接口，页面是管理面板里原生的「沙盒」页；内心的完整视图复用「内心」页（加 团子 / 沙盒 切换）
7. **目录**：整个 `sandbox/` 放仓库根目录、不进 git（剧本里是好友真昵称）

## §1 三者关系

```text
浏览器 ──► 管理面板 console :19390（唯一入口）
             ├─ 页面：总览 │ 实时画面 │ 内心（团子 / 沙盒）│ 沙盒（新）│ 设置 │ 设备检测
             ├─ 子进程槽：同一时间只起一个 —— 团子 或 沙盒
             ├─ /live/*    → 团子  run --view --viewer-port 19391
             └─ /sandbox/* → 沙盒  sandbox --port 19392（新）
```

- 两个子进程里身体、大脑、内心层、提示词是**同一套代码**；区别只在外面接的世界、时钟和记忆目录
- 互斥：团子在跑时沙盒页提示"先停团子"，反之亦然（共用 `.brain-claude/` 和令牌，也省额度）
- 子进程不认识面板，只靠 `--parent-pid` 看门狗；停止走 `POST /shutdown` → `interrupt_main()`，和团子同一条收尾路径

## §2 把 `_run_brain` 拆出"世界"

`cli._run_brain` 现在把设备、OCR、YOLO、轮盘、镜头和身体 / 大脑 / 内心层搅在一起。沙盒**不抄一份**，而是：

- `_run_brain(cfg, world, run, ...)` 只管组装身体、大脑、内心层、MCP 工具、收尾
- `world` 提供"接世界的东西"：`device`、`reader`、`panel`、`scene`（env / perception）、`social`、`emotes`、`camera`、`sender`、`locomotion`、`friend_checker`、`panels`、`eyes` 的建法，以及 `clock`（间隔用）/ `wall`（墙上时间）和 `memory_dir`
- `GameWorld`：把现有的 `_device` / `_build_reader` / `_scene_watcher` / `_build_emotes` / `Camera` … 原样搬进去，**`run` 行为不变**（现有 `test_cli_brain` 全过为证）
- `SandboxWorld`（`src/skydango/sandbox/world.py`）：见下表

| 部件 | 沙盒版 |
|---|---|
| 时钟 `SimClock`（`sandbox/clock.py`） | `wall() = time.time() + offset`，`clock() = time.monotonic() + offset`；`skip(秒)` 加偏移；`set_time("23:30" 或 "2026-10-01 09:00")` = 往前跳到下一个这个时刻，**早于当前沙盒时间就拒绝**；`save()` / `load()` 存 `sandbox/clock.json`（上次停下时的沙盒墙上时间）。身体、事件队列、账本、反思器、流水账、大脑循环都用它（大脑拿自己的钟和事件队列的时间比，必须同一个钟）；一轮的超时在 `brain/claude.py` 里直接用真实时间，快进不会把正在想的一轮判超时 |
| 设备 | `SandboxDevice`：截图永远是一张 1920×1080 中灰图（不黑，免得 `is_black` 判黑屏），按键 / 点屏幕 / 输入法都什么也不做，`ime_shown()` 为假 |
| 聊天读取 | `SandboxReader`：线程安全队列；页面"冒充 X 说 Y"入队，身体下一圈 `read()` 取出成 `Message(text, speaker=X)`。团子自己的话不进队列（直接进聊天记录，见"说话"） |
| 身边 | `SandboxEnv`：接口同 `EnvWatcher`（`observe` 空转、`nearby(now)`、`requests` / `circles` 空、`held()` 空实现），好友名单、陌生人数、地名由页面设置；身体照常 diff 出 `arrive` / `leave` / `return` / `stranger` |
| 眼睛 | 真的 `Eyes`，`describe` 换成返回场景文字（空 = "看不清，眼前什么也看不出来"），不调 Haiku；自动看照常（不花钱，只刷新场景描述）；`look(image=true)`、`look_person` 只给文字（`World.text_only`） |
| 新鲜事 | 页面手动放：身体线程里走和眼睛发现新鲜事同一个入口（照样受 `notice_min`、没熟人、主动额度的限制；被拦下时页面显示原因） |
| 说话 | `SandboxSender`：`send` 把团子的话写进沙盒聊天记录（`clean_reply`、限速、主动护栏都在 `body.say` 里照常过）；`open()` 记一笔"（团子头顶冒出输入气泡）" |
| 动作 | `EmotePlayer` 照常建，但轮盘是假的：可用动作 = `emotes/` 图标库里的名字（没有图标库就用 `[sandbox] emotes` 列表）；做动作记"（团子做了 鞠躬）"；反射动作同样记 |
| 走路 | 成功、记一笔"（团子往前走了 2 步）" |
| 镜头 | 没有（`camera = None`）：`camera` / `camera_reset` / `track` / `look_around` 按身体原有的"没有镜头"处理 |
| 好友树 / 面板识别 / 互动请求 | 这一版不做：`friend_checker = None`、`panels = None`、`social` 空；工具返回"沙盒里没有" |

**模式**：沙盒总是 live（`reply.dry_run = False`）—— 记忆、内心层、随手记、notes 整理、反思、日记都真的写盘，但只写 `sandbox/memory/`（`cfg.reply.memory_dir` 在子进程里改成它）。
随手记 / 整理 / 反思都真的调 Claude，额度和真机一样消耗。运行目录照旧 `runs/<时间>-sandbox/`（agent.log、brain.jsonl、brain/ 工作目录），和团子一起按 `run.keep` 轮换。

`python -m skydango sandbox --port 19392 --no-browser --parent-pid <pid> [--start 接着 | 睡一晚 | "2026-10-01 09:00"]`：管理面板起它；终端也能直接跑（调试用，没有页面，只有接口）。

## §3 沙盒子进程的接口（`sandbox/server.py`）

只监听 127.0.0.1；GET / POST 都校验 Host（`is_local_host`），POST 要 `X-Skydango: 1` + JSON + ≤ 64 KB（`post_guard`），和 viewer 共用。

| 路由 | 内容 |
|---|---|
| `GET /state?after=N` | 长轮询（同 `/brain` 的写法，版本号 + Condition，最多等 25 秒）：沙盒时间、精力档位、身边（好友 / 陌生人数 / 地名）、场景、聊天记录（`after` 之后的新行）、`idle`、额度状态 |
| `POST /op` | `{"op": …, …}`，经 `body.call` 在身体线程执行，返回 `{"ok", "text"}`；op：`say`（`who`、`text`）、`come` / `leave`（`who`）、`strangers`（`n`）、`place`（`name`）、`scene`（`text`）、`notice`（`text`）、`skip`（`seconds`）、`time`（`at`）、`reflect`（立刻反思一次，已经在跑就返回"正在反思"） |
| `GET /brain?after=` | 大脑时间线，和 viewer 同一个处理函数（抽成共用） |
| `GET /inner`、`POST /inner/forget` | 和团子的 viewer 同一个处理函数（内心可视化 spec §2，抽成共用） |
| `GET /status`、`POST /shutdown` | 同 viewer；`/shutdown` 走 `watchdog.once` 的只中断一次钩子 |

**聊天记录**（内存，最近 500 行，每行 `{seq, t（沙盒墙上时间）, kind, who, text}`）：`kind` = `heard`（冒充的发言）/ `said`（团子说的）/ `act`（动作、走路、气泡）/ `event`（来了 / 走了 / 快进 / 改场景 / 新鲜事 / 反思结果一句话，取流水账的 `changes`）/ `blocked`（被护栏拦下的话，比如主动额度用完）。

**`idle`** = 事件队列空 + 大脑不在一轮里 + 反思不在跑 + 身体命令队列空，持续 ≥ 2 秒。给回放用。

**下线**：没有单独的 op —— 页面"下线（写日记）"就是 `POST /shutdown`：收尾时走现有的最终反思 → 日记 → `ledger.close`，然后 `SimClock.save()`。

## §4 管理面板这边

**runner**（`console/runner.py`）：子进程槽加 `kind`（`"dango"` / `"sandbox"`），端口按 kind 取（`child_port` 19391 / `[sandbox] port` 19392）；
`start` 时另一种在 `starting / running / stopping` 就拒绝；启动前两个端口都探一下，有上次留下的进程就提示并能让它退出（沿用现有的"8761 被占"逻辑）。

**转发**（`console/server.py`）：`/sandbox/*` → `127.0.0.1:19392`，GET 全转；POST 只转 `/sandbox/op`、`/sandbox/inner/forget`；沙盒没在跑返回 503。
**新接口**（`console/sandbox_view.py`）：

| 路由 | 内容 |
|---|---|
| `POST /api/sandbox/start` | `{"start": "resume" \| "sleep" \| "2026-10-01 09:00"}`，起子进程（`--start` 透传） |
| `POST /api/sandbox/stop` | 同团子的停止（下线写日记） |
| `POST /api/sandbox/reset` | 只能在沙盒停着时：删掉 `sandbox/memory/`、`sandbox/clock.json`，把 `memory/` 复制过去（`memory/archive/` 不复制）；`memory/` 不存在就建空目录 |
| `GET /api/sandbox/info` | 沙盒记忆里 `friends.md` 的 `## 标题`（给说话人下拉框、身边名单用）+ `[brain] owner_name` + 起始时间下限（`floor_text`） |
| `GET /api/sandbox/scenarios`、`POST /api/sandbox/save`、`POST /api/sandbox/replay`、`POST /api/sandbox/replay/stop`、`GET /api/sandbox/replay` | 剧本列表 / 另存为 / 回放 / 停止回放 / 回放进度（§6） |

**「内心」页**（按内心可视化 spec 实现）加 **团子 / 沙盒** 切换：`GET /api/inner?source=sandbox` 读 `sandbox/memory/inner/`，沙盒在跑时实时部分取 `/sandbox/inner`；删性格条目同理转发或直接改 `sandbox/memory/inner/persona.json`。
沙盒的流水账时间是沙盒时间，曲线自然按沙盒时间画；`skip` 前后各记一条 `energy`，曲线上看得出跳了一段。

## §5 「沙盒」页（`console.html`）

导航在「内心」后面加「沙盒」。布局（宽屏；手机竖排）：

```text
┌ 沙盒 ───────────────────────────────────────────────────────────────────────────┐
│ ● 运行中 · 大脑在想…   [下线（写日记）]   [重置记忆]                              │
│ 沙盒时间 9月30日 周三 23:30 · 精力：困   [+10分钟] [+1小时] [到明早 9 点] [设成 __:__]│
├────────────────────────────────────────────┬────────────────────────────────────┤
│ 聊天记录                                    │ 现在（同「内心」页第 1 块的画法）    │
│  ── 小明来了 ──                             │  心情 / 精力 / 别扭 / 收着点 / 心愿  │
│  小明：在吗                                 │  [看完整内心 →]                      │
│  团子：哼，还知道来啊                        ├────────────────────────────────────┤
│  （团子做了 背过身）                        │ 身边  [小明 ×] [+ 好友 ▾]           │
│  ── 快进 1 小时 ──                          │       陌生人 [-] 2 [+]  地名 [云野]  │
│  ── 反思：心情 低落→平常；别扭撤了：小明 ── ├────────────────────────────────────┤
│                                             │ 场景 [大家围着篝火坐着……        ]  │
│ [小明 ▾] [说点什么……             ] [发送]  │ 新鲜事 [天黑了            ] [放]    │
│                                             ├────────────────────────────────────┤
│                                             │ 剧本  ● 录制中 [新录制] [另存为…]    │
│                                             │       [放鸽子 ▾] [回放] [停止回放]   │
├────────────────────────────────────────────┴────────────────────────────────────┤
│ 大脑时间线（和实时画面页同一个组件）                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- **没在跑**时：顶上是启动选项（起始时间：接着上次 / 睡一晚到第二天 9 点 / 自定义）+ [启动]；团子在跑时启动按钮置灰、提示"先停团子"
- **起始时间的下限** = `clock.json` 和沙盒 `days.jsonl` 最后一次上线的结束时间取晚的（重置后 `days.jsonl` 来自真记忆，所以沙盒时间不会早于真实的最后一次上线）；早于下限就拒绝。第一次 / 重置后"接着上次"= 现在
- **只有一个停的按钮**「下线（写日记）」：和团子的停止一样走最终反思 → 日记 → 合账本；超时强杀照旧
- **说话人**下拉：沙盒 friends.md 里的好友 + "陌生人（自己填名字）" + 主人（`owner_name`，可以发 `#` 命令）
- **聊天记录**按 `kind` 分样式：`heard` 左、`said` 右、`act` / `event` 居中灰字、`blocked` 删除线 + 原因
- **身边**：加好友 = 来了，× = 走了；陌生人数、地名即改即生效
- **时钟**：快进按钮、"设成"输入框；拨不回去时提示原因
- **重置记忆**：只在停着时可按，确认框写明"会用 memory/ 覆盖沙盒记忆，沙盒里聊出的交情、日记、性格都会没"
- 大脑时间线：把 viewer 页面里画时间线的 JS / CSS 抽成共用的一段（viewer 和管理面板都引用），不写两份
- 刷新：`/sandbox/state` 长轮询，时间线走 `/sandbox/brain` 长轮询
- 视觉沿用管理面板现有的颜色 / 字体 / 卡片，实现时用 `frontend-design` skill 做排版细节

## §6 剧本（`console/scenario.py`，`sandbox/scenarios/*.toml`）

剧本 = 沙盒页操作的记录。录制和回放都在**管理面板进程**里做：录制 = 记下转发过的 `/op`、启动、停止；回放 = 按顺序再发一遍。
所以剧本能**跨多次上线**（下线 → 等子进程退出 → 按 `online` 再起一个）。

```toml
name = "放鸽子"
note = "说好来又不来，看会不会闹别扭；认真说难过后别扭要撤"

[start]
memory = "reset"          # reset = 先从 memory/ 重置；keep = 用沙盒现在的记忆
time = "20:00"            # 起始沙盒时间：HH:MM（下限之后最近的这个时刻）或 "YYYY-MM-DD HH:MM"（早于下限报错）
nearby = ["小明"]
strangers = 0
place = "云野"
scene = "云野，傍晚，篝火旁"

[[steps]]
who = "小明"
say = "我 9 点再来找你玩"
[[steps]]
leave = "小明"
[[steps]]
skip = "2h"               # 快进：30s / 10m / 2h；或 time = "23:30"
[[steps]]
come = "小明"
[[steps]]
reflect = true
[[steps]]
who = "小明"
say = "对不起……我今天真的很难过"
expect = "别扭当场撤掉"    # 这一版只读进来、写进报告，不检查
[[steps]]
offline = true            # 下线写日记
[[steps]]
online = "sleep"          # 再上线：sleep = 睡一晚到第二天 9 点；或 "HH:MM" / 完整时间
```

- 一步一个动作键：`say`（配 `who`）/ `come` / `leave` / `strangers` / `place` / `scene` / `notice` / `skip` / `time` / `reflect` / `offline` / `online`；外加可选 `expect`、`wait`（这一步后多等几秒真实时间，给主动开口留机会）
- **载入即校验**：不认识的键、一步里有两个动作、时间格式不对、`online` 前面没有 `offline` → 报错并指出第几步，不开始回放
- **录制**：从沙盒启动开始自动录，停止 / 再启动接着录（记成 `offline` / `online`），直到点"新录制"或重置记忆。"另存为"写 `[start]`（启动时的选项和身边状态）+ 所有步骤；启动时没重置记忆就写 `memory = "keep"` 并提示"这个剧本依赖沙盒当时的记忆"
- **回放**：沙盒在跑先停掉（下线），按 `[start]` 重置 / 起一个；每步发出后等 `idle`（§3），`[sandbox] step_timeout`（180 秒）还不安静就记"超时"、接着下一步；`offline` 等子进程退出（最多 `console.stop_timeout`）。回放过程就在沙盒页上实时显示，页面上的手动操作在回放时置灰
- **停止回放**：当前这一步做完后停，沙盒保持运行，可以接着手动玩
- **报告**（`sandbox/reports/<剧本>-<时间>.md`）：每一步做了什么 + 这一步之后的聊天记录新行（团子说的话、动作、被拦下的话）+ 那段时间流水账里的反思 `changes` + `expect`；末尾附最后的心里、这次写的日记。同一个剧本多跑几次对比着看
- 示例剧本放 `docs/sandbox-scenarios/`（进 git，人名是占位的 `小明` / `阿花`，用前换成自己的好友名再复制到 `sandbox/scenarios/`）：放鸽子、深夜犯困、第二天上线

## §7 配置（`[sandbox]`）

```toml
[sandbox]
dir = "sandbox"          # 沙盒目录（memory/、scenarios/、reports/、clock.json）
port = 19392             # 沙盒子进程的接口端口
step_timeout = 180       # 回放时每步最多等多久安静（秒，真实时间）
wake_hour = 9            # "睡一晚" / "到明早" 拨到几点
emotes = []              # 没有 emotes/ 图标库时假装轮盘上有这些动作
```

`.gitignore` 加 `sandbox/`。管理面板设置清单不加（都是很少改的）。

## §8 出错怎么办

| 情况 | 处理 |
|---|---|
| 没设令牌 / 没装 `mcp` | 启动预检和团子共用，沙盒页报出来 |
| Claude 额度用完 | 和真机一样：聊天交给 DeepSeek 备用回复、反思跳过；状态里写"额度用完，X 分钟后再试" |
| 子进程崩了 | 页面显示日志尾巴；下次启动账本补一行"意外断了"（正好测补记） |
| 19392 被上次留下的沙盒占着 | 提示并能让它退出，退出前不让启动 |
| 团子 / 沙盒同时启动 | 拒绝，提示先停另一个 |
| 时钟往回拨、运行中重置 | 拒绝并说原因 |
| `memory/` 不存在或是空的 | 重置后是空记忆，团子用 `reply.persona` |
| 剧本写错 | 载入时报错、指出第几步 |
| 回放中某步请求失败 / 超时 | 记进报告，接着下一步 |
| `clock.json` 坏了 | 当作没有（从现在开始），日志 WARNING |

## §9 测试（`python -m pytest -q`，不要 MuMu、不调真 Claude）

- `SimClock`：`skip`、`set_time` 只往前（今天已过就明天、完整时间早于当前拒绝）、`save` / `load`、坏文件
- `SandboxWorld` + 真 `Body`（假时钟）：冒充发言 → `chat` 事件；名单加 / 删 → `arrive` / `leave` + 关系卡；团子 `say` 进聊天记录、`blocked` 记原因；`look` 返回场景文字 / "看不清"；动作记一笔；新鲜事走 notice 护栏
- **拆出世界不改 `run`**：现有 `test_cli_brain` 全过；新增沙盒端到端（`tests/fake_claude.py` 当大脑）：说话进聊天记录、记忆只写 `sandbox/memory/`，**`memory/` 一个字节不变**（跑前跑后比哈希）
- 沙盒接口：各 op、`/state` 长轮询、`idle`、安全校验（非本机 Host、缺头、非 JSON、超 64 KB 拒绝）
- 管理面板：`/sandbox/*` 转发、团子 / 沙盒互斥、重置（停着才行、`memory/` 原样）、`/api/inner?source=sandbox`
- 剧本：载入校验（各种错）、存 / 读来回一致、回放引擎用假子进程接口测（顺序、等 idle、超时、`offline` → `online` 重起、停止回放）、报告内容
- 页面：导航有「沙盒」、各区域容器 id 在（同现有页面测试查字符串）

## §10 实施顺序

一份计划、两步，各自能用：
- **A 沙盒本体**：拆出世界（§2）、沙盒接口（§3）、runner / 转发 / 重置 / 「内心」页切换（§4）、「沙盒」页手动玩（§5）
- **B 剧本**：录制、回放、报告、示例剧本（§6）

## §11 这次不做

牵手 / 拥抱等互动请求、好友树、面板识别；用真截图当画面；剧本断言和模型打分；多个沙盒同时开；手机访问沙盒页。

## 验证（不需要 MuMu）

1. 管理面板开沙盒页、重置记忆、启动：冒充好友聊几句，大脑时间线、聊天记录、"现在"都在动
2. 快进到 23:30：精力变困、主动开口变少（聊天记录里 `blocked` 或者干脆不说）
3. 跑示例剧本"放鸽子"：报告里有别扭、说难过后撤掉；「内心」页切到沙盒能看到流水账和曲线
4. 下线再"睡一晚"上线：「日子」里带上一篇日记；`memory/` 没被动过

之后把结论写进 CLAUDE.md（新增「大脑沙盒」一节、代码结构表加 `sandbox/`）。
