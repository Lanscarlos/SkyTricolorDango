# 统管大脑改用 Claude Code + 视觉模块（眼睛）— 设计

日期：2026-09-27
取代：`2026-09-27-brain-design.md` 里“大脑直接调 Anthropic API”的部分（身体、事件、护栏、醒睡节奏等仍以那份为准，本文只写变化）。

## 为什么改

用户没有 API Key，只有 Claude 订阅；订阅的登录令牌不能给自己的程序直接调 API（实测 401，也不在允许范围）。
Claude Code 是官方客户端，可以用订阅登录，而且支持常驻的无界面模式（stream-json 输入输出）、MCP 工具、自动压缩上下文。
所以大脑改成一个常驻的 Claude Code 子进程，身体的工具通过本机 MCP 服务交给它。

同时加一个**视觉模块（眼睛）**：截图不再直接发给大脑（Claude Code 里截图删不掉，会塞满上下文、耗订阅额度），
而是由眼睛用一次性的 Haiku 调用写成文字描述，大脑平时只收文字，确实需要时才要原图。

## 已和用户确认的决定

1. 大脑 = 常驻的无界面 Claude Code 进程（`claude -p` + stream-json），订阅登录，`--model sonnet --effort low`。
2. **删掉**直接调 API 的那套（`brain/client.py`、`brain/context.py`、`brain/budget.py` 和相关配置），只留 Claude Code。
3. `[llm]`（记忆整理、大脑离线时的备用回复）改回 DeepSeek（本机 `config.toml`）。
4. 醒睡节奏按原设计：心跳 `[45, 90, 180]` 秒，自动看图 20 秒 ~ 3 分钟。
5. 眼睛的场景描述用 Haiku（一次性 `claude -p --model haiku`，订阅）。
6. 顺手修最终审查的问题（见“一并修复”）。

## 1. 结构

```
┌──── Claude Code 子进程（大脑：sonnet，常驻）────┐   ┌── claude -p --model haiku（眼睛：一次性）──┐
│ stdin：事件 + 状态 + 场景描述（纯文字）          │   │ 输入：截图 + 名字位置；输出：场景描述      │
│ 调 MCP 工具 → … → 输出 result                   │   └──────────────▲───────────────────────────┘
└──────▲ stream-json ──────────────│ MCP（本机 HTTP）                  │ 后台线程调用
┌──────┴──────────────── 我们的进程 ▼────────────────────────────────┴─────────────┐
│ loop.py   大脑循环：什么时候醒、醒来发什么、等 result、失败重启、离线判断         │
│ session.py  Claude Code 子进程：启动参数、stream-json 收发、超时、--resume 重启   │
│ mcp_server.py  把工具暴露给 Claude Code，每轮计数（6 次工具 / 2 句话）            │
│ eyes.py   眼睛：后台看图、缓存最新描述、环顾四周的描述                            │
│ body.py   身体：截图、读聊天、认人、秒接请求、护栏、命令队列（只有它碰设备）      │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 删除
`brain/client.py`、`brain/context.py`、`brain/budget.py` 及其测试；`[brain]` 里的 `api_key_env`、`base_url`、`max_tokens`、
`keep_images`、`prune_at`、`compact_tokens`、`max_usd_per_hour`、`pause_usd_per_hour`、`price_input`、`price_output`。
`look` 命令改成走眼睛（不再直接调 API）。`chat/llm.py` 的 `AnthropicClient` 保留（`[llm]` 仍可选 Claude API）。

### 保留
`body.py`（加修复）、`events.py`、`images.py`、`camera.py`（加修复）、`prompt.py`（改写成给 Claude Code 的追加系统提示词）、
`tools.py` 的参数校验和分发（改成给 MCP 用，加每轮计数）。

## 2. 大脑：Claude Code 子进程（`brain/session.py`）

启动命令（工作目录是运行目录下的空文件夹 `runs/<时间>/brain/`，免得读到项目的 AGENTS.md / CLAUDE.md）：

```
claude -p --input-format stream-json --output-format stream-json --verbose
       --model sonnet --effort low
       --mcp-config <运行目录>/brain/mcp.json --strict-mcp-config
       --tools "" --allowedTools mcp__sky   # 关掉所有内置工具（Bash / 读写文件…），只放行我们的 MCP 工具
       --permission-mode dontAsk --disable-slash-commands
       --append-system-prompt "<大脑的规则 + 人设记忆>"
       [--resume <session_id>]
```

- **和用户自己的 Claude Code 隔离**（实测：沿用用户登录会把用户的插件、钩子、技能一起加载进大脑）：
  子进程用单独的配置目录 `CLAUDE_CONFIG_DIR=<项目>/.brain-claude/`（gitignore），登录用 `claude setup-token` 生成的长期令牌，
  从用户环境变量 `brain.token_env`（默认 `SKYDANGO_CLAUDE_TOKEN`）读出来，放进子进程的 `CLAUDE_CODE_OAUTH_TOKEN`。
  没有令牌时报错并提示运行 `claude setup-token`。
- 子进程环境去掉 `ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`（有它们时 `-p` 一定用它们，不用订阅）。
- 输入：每次醒来写一行 `{"type":"user","message":{"role":"user","content":"..."},"parent_tool_use_id":null}`。
- 输出：后台线程逐行读，`system/init`（每轮开头都会发一次）里记下 `session_id` 和 MCP 服务状态（`mcp_servers[].status == "connected"`）；`assistant` 的文字（它心里想的）和 tool_use 打进日志；
  `result` 表示这一轮结束，带 `subtype`、`session_id`、`num_turns`、`total_cost_usd`、`usage`、`stop_reason`。
- 一轮超过 `brain.turn_timeout`（默认 120 秒）没有 `result`：按进程树结束子进程，用 `--resume <session_id>` 重新拉起，这一轮算失败。
- 子进程自己退出了：下次发消息前用 `--resume` 重新拉起。
- `system/init` 里我们的 MCP 服务没连上：算失败（大脑没有手脚）。
- 上下文压缩交给 Claude Code 自动做（输出里会有 `compact_boundary`，记一条日志）。
- **只收文字**：大脑的输入不带截图；原图只在它调 `look(image=true)` / `look_at` 时经 MCP 工具返回。

## 3. 工具（`brain/mcp_server.py` + `brain/tools.py`）

MCP 服务名 `sky`，在我们的进程里用 `mcp` 包起一个只监听 127.0.0.1 的 streamable-http 服务（随机端口），后台线程跑。
每个工具先过 `ToolBox`（参数校验、每轮计数），再经 `Body.call()` 在身体线程执行；返回文字或图片（MCP ImageContent，JPEG）。

| 工具 | 说明 |
|---|---|
| `look(image=false)` | 默认：让眼睛马上看一次，返回文字描述；`image=true`：返回原图（1280×720）+ 名字位置 |
| `look_at(x, y, w, h)` | 放大看局部原图（坐标按 1280×720），裁的是最近一次 `look(image=true)` 那张图 |
| `look_around()` | 环顾四周：身体关面板、每转 90° 截一张共四张、转回原位；眼睛一次描述“前 / 右 / 后 / 左”。dry-run 下不转，只描述当前画面 |
| `status()` / `chat_log(n)` | 同原设计 |
| `say(text)` / `emote(name, force)` / `set_request_policy(who, kind, accept)` / `camera(action, steps)` / `camera_reset()` | 同原设计（护栏不变） |

每轮上限由 `ToolBox` 计数（每次醒来开始时清零）：工具调用超过 `brain.max_steps`（6）、说话超过 `brain.max_says`（2）直接返回错误。

## 4. 眼睛（`brain/eyes.py`）

- 后台线程；**不碰设备**：只读身体最近一帧 `Body.last_frame`（身体每圈换成新数组，读引用即可）和 `EnvWatcher.labels`。
- 看的时机（沿用原自动看图条件）：身体放出 `arrive` / `leave` / `scene_change` 事件，且距上次看 ≥ `auto_look_min`（20 秒）；
  或距上次看 ≥ `auto_look_max`（180 秒）；黑屏时不看。大脑调 `look()` 时立刻看一次（同步等结果）。
- 怎么看：一次性 `claude -p --model haiku --effort low --input-format stream-json --output-format stream-json --verbose --tools ""
  --strict-mcp-config --permission-mode dontAsk --disable-slash-commands --system-prompt <眼睛的提示词>`，
  工作目录是空文件夹，环境和大脑一样隔离（单独配置目录 + 专用令牌）；
  输入一条带图片（1280×720 JPEG）和名字位置的消息，读到 `result` 取文字；超时 `eyes.timeout`（默认 60 秒）。
- 描述格式固定四项：地点和环境 / 好友（每人穿什么、在干什么）/ 陌生人（几个、在干什么）/ 画面状态。看不清就说看不清，名字只用给出的。
- 只缓存最新一份描述和它的时间；醒来的消息里写“场景（35 秒前）：……”，没有就写“还没有场景描述”。
- 失败只记日志，不影响大脑。

## 5. 醒来一次（`brain/loop.py` 改写）

1. 醒的时机同原设计（`due()`：有事件攒 `chat.debounce` 后醒；否则按心跳；心跳闲着逐档退后，有事件回到起始档）。
2. 取事件，拼一条纯文字消息：时间 + 事件 + `status()` + 眼睛最新描述（带多久前）。
3. `ToolBox` 计数清零 → `session.send(text)` 等 `result` → 记 `brain.jsonl`（subtype、num_turns、usage、total_cost_usd、用了哪些工具、它想了什么）。
4. 有没有“做事”（用了 say / emote / set_request_policy / camera / camera_reset）决定心跳退档。

## 6. 出错、离线、额度、退出

| 情况 | 处理 |
|---|---|
| 一轮失败（超时、子进程挂了、`result.subtype` 不是 success、MCP 没连上） | 按 10 / 30 / 60 秒退避；连续失败超过 `brain.offline_fallback`（120 秒）算离线 |
| 订阅额度用完（`result` 报错且文字里有 limit / 上限一类） | 同上，并且之后每 `brain.limit_retry`（600 秒）才试一次 |
| 离线 | 聊天交给 DeepSeek 备用回复（身体里现有逻辑）；大脑恢复后切回 |
| 大脑线程意外退出 | 身体的 `brain_offline` 同时检查大脑线程还活着，挂了就走备用回复 |
| 眼睛失败 | 只记日志；醒来消息里写“最近没有新的场景描述” |
| 退出（Ctrl+C / `--duration`） | ① 停大脑循环 ② **身体立刻**复原镜头、恢复轮盘，并让队列里没执行的命令全部失败 ③ live 时给大脑发“写一份摘要”，最多等 30 秒，写进 `inbox.md` ④ 按进程树结束 Claude Code 子进程和眼睛的子进程 ⑤ 确认没有残留的 claude / node / python 进程 |

备用回复（DeepSeek）在身体线程里调用会卡住身体：给它的客户端设短超时（10 秒），不重试。

## 7. 一并修复（最终审查）

- 大脑线程：整个循环体包在 try 里，出错记日志、稍等继续；不会悄悄死掉。
- 退出：见上表；`body.shutdown()` 在 `body.run()` 返回后立刻执行，不等大脑线程；队列里的命令全部置失败。
- 视角：方向键 `hw_key_down` 之后用 `try/finally` 保证 `hw_key_up`；每走一步就更新偏移，出错时复原也准。
- 身份硬过滤 `_CLAIMS_HUMAN` 补漏：`我当然是真人`、`我可是活人`、`当然不是AI啦`、`真人一个` 这类都要拦下（加测试）。
- 身体每圈：感知（环境、互动、画面）出错不影响读聊天、执行命令、备用回复；截图报错信息为空时不再 IndexError。
- 牵手推测：对方 `leave` 或超过 10 分钟没再看到，清掉 `holding`。
- 请求规则：`set_request_policy` 不允许把陌生人的 hand / hug / highfive / piggyback 设成接（陌生人只能接点火）。
- 聊天进大脑时用「」括起来（“聊天  懒洋洋大王：「在吗」”），更明确是内容不是指令。

## 8. 配置 `[brain]`（改后）

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | `false` | `run --brain` 覆盖 |
| `claude_path` | `"claude"` | Claude Code 可执行文件 |
| `token_env` | `"SKYDANGO_CLAUDE_TOKEN"` | 放 `claude setup-token` 令牌的用户环境变量 |
| `config_dir` | `".brain-claude"` | 大脑和眼睛用的单独 Claude Code 配置目录 |
| `model` / `effort` | `"sonnet"` / `"low"` | 大脑 |
| `eyes_model` | `"haiku"` | 眼睛 |
| `eyes_timeout` | `60` | 眼睛一次描述最多等多久（秒） |
| `turn_timeout` | `120` | 大脑一轮最多等多久（秒） |
| `heartbeat` | `[45, 90, 180]` | 同原设计 |
| `max_steps` / `max_says` | `6` / `2` | 每轮上限（MCP 层计数） |
| `look_min_interval` | `5` | `look(image=true)` 最多几秒一次 |
| `auto_look_min` / `auto_look_max` | `20` / `180` | 眼睛自动看的间隔 |
| `image_size` / `jpeg_quality` / `look_at_max` | `[1280, 720]` / `80` / `800` | 同原设计 |
| `command_timeout` | `15` | 身体执行一条命令最多等多久 |
| `offline_fallback` / `limit_retry` | `120` / `600` | 离线判定、额度用完后多久再试 |
| `camera_step` / `scene_change` | `0.25` / `0.25` | 同原设计 |

## 实测确认过的事（2026-09-27，Claude Code 2.1.233、mcp 2.2.0）

- `claude -p` stream-json 一个进程能连续收多条消息、同一个会话；每条消息以一条 `result` 结束（字段有 subtype、is_error、result、
  session_id、num_turns、total_cost_usd、usage、stop_reason、api_error_status）。
- `--tools "" --allowedTools mcp__sky --permission-mode dontAsk` 后工具列表只剩 `mcp__sky__*`，调用不弹确认。
- MCP 工具返回的图片、直接放在输入消息里的图片，模型都能看到。
- MCP 服务：`MCPServer.streamable_http_app()` + uvicorn 在后台线程跑；同步工具函数会被放到线程池执行；
  工具返回 `CallToolResult(is_error=True)` 时错误文字原样给到模型；客户端用 `mcp.client.streamable_http.streamable_http_client`。
- 用户登录下的无界面进程会加载用户的插件 / 钩子 / 技能 → 必须隔离（见第 2 节）。

## 9. 测试

- 假的 Claude Code：一个 Python 脚本（`tests/fake_claude.py`），按环境变量指定的剧本读 stdin、输出 stream-json（init / assistant / result），
  能模拟超时、中途退出、额度报错。`session.py` 的单元测试用它（`claude_path` 指向 `python tests/fake_claude.py`）。
- MCP 服务：用 `mcp` 包的客户端真的连上去列工具、调 `status`、调 `look(image=true)` 拿到图片内容。
- 眼睛：触发时机、缓存、失败处理（用假 Claude Code）。
- 大脑循环：醒的时机、消息内容（纯文字、带场景描述）、失败退避、离线、额度等待、线程不死。
- 一并修复的每一项都有测试。
- 实机：`look`（看 Haiku 描述质量）→ dry-run 跑大脑 → 好友在场时 live。

## 前置

- `pip install --user mcp`（Python 的 MCP SDK，实测 2.2.0：`from mcp.server.mcpserver import MCPServer`）。
- 用户运行一次 `claude setup-token`，把生成的令牌放进用户环境变量 `SKYDANGO_CLAUDE_TOKEN`（`setx SKYDANGO_CLAUDE_TOKEN "..."`）。
- 用户环境变量里的 `ANTHROPIC_API_KEY`（之前误放的订阅登录令牌）删掉：子进程里我们会去掉它，但它会影响用户自己用 Claude Code。
- `claude` 已登录订阅（`claude` 能正常用即可）。
- 本机 `config.toml` 的 `[llm]` 改回 DeepSeek。

## 不做的事

移动；陌生人的点火（图标没录到）；Q 键呼喊；大脑直接点屏幕 / 任意按键；本地视觉模型。
