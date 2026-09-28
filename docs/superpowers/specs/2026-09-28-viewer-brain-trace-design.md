# 可视化网页：大脑活动时间线 — 设计

日期：2026-09-28　状态：**设计已和用户确认，待实现**

在识别可视化网页（`vision/viewer.py`，设计见 `2026-09-28-viewer-design.md`）上加一栏「大脑」：
`run --brain --view` 时实时显示统管大脑（常驻 Claude Code）每一轮的来龙去脉。

## 目标

**调试 / 调提示词。** 大脑某一轮做了怪事（答应跑图、重复回答、该说话没说……），打开网页往上翻，
能看到它当时收到的原话、调了什么工具、工具返回了什么、最后说了什么，足以判断该改提示词的哪一句。

## 需求（已和用户确认）

1. **只管大脑模式**（`run --brain --view`）。普通 Agent 模式（DeepSeek 回复）不管；`view`（只看不动）没有大脑，不显示这一栏。
2. **只看实时**：只存在内存里，不写盘，不做"打开以前某次运行回看"。`agent.log`、`brain.jsonl` 照旧。
3. 每一轮要看到：为什么被叫醒、身体发给它的完整消息、它说的每段话、调的每个工具（参数 + 返回）、思考（有内容才有）、
   这一轮的结果（耗时、tokens、有没有出错 / 超时）。

## 方案选择

**方案 A：大脑那边加一个内存里的记录器（`BrainTrace`），网页用单独的 `/brain` 长轮询接口取，画面下方加一栏时间线。**

放弃的方案：
- **B. 塞进现有快照右栏的 `info`**：右栏太窄；快照每秒最多 10 次、连着图一起发，长文本反复传，也放不下展开细节。
- **C. 单独页面滚动显示日志**：只是把 agent.log 搬到网页，没有分轮、没有展开，调试效率和翻日志差不多。

## 1. 数据：`src/skydango/brain/trace.py`

`BrainTrace`：线程安全的内存记录（一把锁 + `threading.Condition`），最多留最近 `MAX_TURNS = 50` 轮。
只在 `run --brain --view` 时创建；不开网页就不存在，大脑那边零开销。

### 一轮（`Turn`）记什么

| 字段 | 内容 | 来源 |
|---|---|---|
| `id` | 自增序号 | `begin()` |
| `start` | 开始时间（墙钟，页面显示 `HH:MM:SS`） | `begin()` |
| `reason` | `events`（有新事件）/ `heartbeat`（定时醒来）/ `farewell`（退出前总结） | `Brain.wake` / `farewell` |
| `prompt` | 身体发出去的完整消息（事件 + 状态 + 眼睛的描述） | `Brain.wake` 里的 `text`；farewell 是 `SUMMARY_REQUEST` |
| `steps` | 按顺序的步骤，见下 | `session.on_message` |
| `end` | 结束时间；进行中为 `None` | `finish()` / `fail()` |
| `result` | subtype、num_turns、耗时、tokens（输入 / 输出 / 缓存读 / 缓存写 / 思考）、`total_cost_usd`（订阅不按它收费，只做参考） | `session.send` 返回的 result |
| `error` | 失败时的错误信息（超时、额度用完、MCP 没连上……） | `ClaudeError` |
| `updated` | 最后一次变化时的版本号（增量返回用） | 每次改动 |

### 步骤（`steps` 里每一项）

从 stream-json 消息里拆出来：

| `kind` | 来源 | 字段 |
|---|---|---|
| `thinking` | `assistant` 消息里 `type: "thinking"` 的块，**有非空内容才记** | `text` |
| `text` | `assistant` 消息里 `type: "text"` 的块（去掉空白后非空） | `text` |
| `tool` | `assistant` 消息里 `type: "tool_use"` 的块 | `id`、`name`（去掉 `mcp__sky__` 前缀）、`input` |
| `result` | `user` 消息里 `type: "tool_result"` 的块 | `tool_use_id`、`text`、`error`（`is_error`） |

- `tool_result` 的 `content` 可能是字符串，也可能是块列表：文字块拼起来，图片块记成 `[图片]`，不传原图。
- 单条文本超过 `MAX_CHARS = 20000` 字截断，末尾注明"（截断，原长 N 字）"。
- `system`、`result` 等其他类型的消息不进步骤（`result` 由 `finish()` 处理）。
- 思考内容：大脑用 `effort low`，实测 `thinking_tokens` 有时是 25，但 stream-json 里不一定带思考原文。
  有就显示，没有就只在结果里显示"思考 N tokens"。

### 轮外消息

正常情况下所有 `assistant` / `user` 消息都在某一轮的 `begin()` 和收尾之间。万一在两轮之间收到（比如上一轮超时后进程还在吐输出），
记进一个 `id = 0`、`reason = "outside"` 的"轮外"轮次，不丢；页面上显示为"轮外"。

### 接口

```python
trace = BrainTrace(state=callable)      # state() -> dict：栏头的总体状态，见 §3
trace.begin(reason, prompt)             # 开一轮；上一轮没收尾就先按"被打断"收尾
trace.feed(message: dict)               # session.on_message 每条都给它
trace.finish(result: dict, seconds)     # 这一轮成功
trace.fail(error: str, seconds)         # 这一轮失败（已收到的步骤保留，看得出卡在哪）
trace.since(after: int, timeout) -> dict  # 等到版本号 > after（最多 timeout 秒），返回增量
```

`since` 返回：

```json
{"boot": "3f9a1c2e", "version": 128, "oldest": 79, "state": {...}, "turns": [ /* updated > after 的轮次，按 id 升序 */ ]}
```

- `after` 比当前版本号大（程序重启过）：当 `after = 0` 处理，返回全部。
- 等到超时也照样返回 200，`turns` 为空、`state` 是最新的（栏头的"N 秒后重试"要靠它刷新）。
- 被挤出 50 轮上限的轮次，浏览器那边按 `id` 自己删（返回里带 `oldest`：现存最早的轮次 id）。

所有入口都包 `try/except`，出错只记 DEBUG 日志，**监控出问题不能影响大脑本身**。

## 2. 接线

- **`brain/loop.py`**：`Brain.__init__` 多一个可选参数 `trace=None`。
  - `wake`：拼好 `text` 后 `trace.begin(reason, text)`；`session.send` 成功 → `trace.finish(result, 耗时)`；
    抛 `ClaudeError` → `trace.fail(str(exc), 耗时)`。
  - `farewell` 同样三处（reason 为 `farewell`）。
  - `trace=None` 时行为和现在完全一样。
  - 另外给 trace 提供 `state()`：模型 / effort、轮数、连续失败次数、多少秒后重试、是否已转备用回复（`offline()`）。
- **`cli.py` `_run_brain`**：有 viewer 时建 `BrainTrace`，交给 `Brain(trace=...)`；
  `BrainSession(on_message=...)` 改成先 `log_brain_message(m)` 再 `trace.feed(m)`；`viewer.brain = trace`。
  没有 viewer 时一切照旧。
- **`vision/viewer.py`**：`Viewer` 加属性 `brain = None`；新增 `GET /brain?after=<版本号>`：
  - `brain` 是 `None` → 404（页面就不显示这一栏）
  - 否则 `brain.since(after, WAIT)` → 200 JSON
  - 和 `/snapshot` 完全分开：大脑文字再多也不拖慢画面。

## 3. 页面

画面下方加一栏「大脑」，和画面同宽（手机宽度时照样在画面下面）；右侧状态栏不变。
页面启动时请求一次 `/brain`，404 就整栏不显示、不再请求。

### 栏头：总体状态

> 大脑 · sonnet / low · 已醒 37 轮 · 在线

失败时换成"连续失败 2 次，30 秒后重试"（黄）或"已转备用回复（DeepSeek）"（红）；`/brain` 连不上时显示"连不上"，1 秒后重试。

栏头右边两个开关：
- **只看做了事的轮次**：藏掉心跳醒来、没调过工具也没说话的轮次（最常见的噪音）。默认关。
- 画面的「暂停」不影响大脑这一栏，一直更新。

### 轮次列表：最新的在最上面

每轮折叠时一行：

```
▶ 22:41:59  新消息   say×1 emote×1   3.2s   in 1.5k · out 88 · 思考 25
▶ 22:41:33  心跳     （什么都没做）   1.8s   in 1.4k · out 20
▶ 22:40:51  新消息   失败：超时（120s）                        ← 红色
▼ 22:42:10  新消息   进行中…                                   ← 自动展开
```

- 进行中的那一轮自动展开，步骤实时往下加；其余点一下展开 / 收起，数据刷新后保持用户的展开状态。
- 原因显示：`events` → 新消息 / 事件，`heartbeat` → 心跳，`farewell` → 退出前总结。

### 展开后：按时间顺序

1. **收到**：身体发来的整段消息，等宽字体原样显示
2. 每个步骤一块，用颜色和前缀区分：
   - 💭 思考（有内容才有）
   - 💬 说：它在文字里说的话
   - 🔧 调用 `say`：参数格式化成 JSON
   - ↩ 返回：工具结果；超过 10 行先折叠，点开看全文；`error` 为真的标黄（护栏拒绝 / 出错）
3. **结果**：subtype、耗时、各项 tokens、参考花费；失败时显示错误信息（红）
4. **复制这一轮**按钮：整轮按纯文本复制到剪贴板，方便贴给别人讨论

### 风格

沿用现有深色面板（`--bg` / `--panel` / `--line` 等变量），不引入外部库、字体。所有文字用 `textContent` 写入，不拼 HTML（聊天内容里可能有尖括号）。

## 4. 边界情况

| 情况 | 处理 |
|---|---|
| 一轮中途进程挂了 / 超时 | `fail()`：标失败、写错误信息，已收到的步骤保留 |
| `begin()` 时上一轮还没收尾（不该发生） | 上一轮按"被打断"收尾 |
| 工具返回是图片（`look(image=true)`） | 记 `[图片]` |
| 超长文本 | 截断到 2 万字并注明 |
| 超过 50 轮 | 丢最早的；返回里带 `oldest`，浏览器删掉更早的 |
| 程序重启、浏览器没刷新 | 返回里的 `boot`（每个进程随机）变了，或版本号变小 → 浏览器清空、从 0 重新拉，这次的响应丢掉（新进程的版本号可能已经超过浏览器记的旧值，只看版本号发现不了——实现时核对页面发现的）；`after` 比当前版本号大 → 当 0 处理 |
| 多个浏览器同时看 | 各自长轮询，`since` 只读，互不影响 |
| trace 自身出错 | 只记 DEBUG 日志，不影响大脑 |

## 5. 测试（`python -m pytest -q`，不需要模拟器）

- `tests/test_brain_trace.py`：
  - 喂模拟的 stream-json 消息（文本、思考、空思考、tool_use、字符串 / 块列表的 tool_result、带图片的、`is_error` 的），检查步骤拆分
  - 截断、50 轮上限和 `oldest`、增量返回（只给 `updated > after` 的轮次）、`after` 过大当 0 处理、超时返回空 `turns`
  - `fail()` 保留已有步骤；`begin()` 打断未收尾的一轮；轮外消息
- `tests/test_brain_loop.py`（已有文件里加）：假 session 走一遍成功 / 失败 / farewell，检查 trace 收到的调用；`trace=None` 行为不变
- `tests/test_viewer.py`（已有文件里加）：0 号端口起服务测 `/brain`（没有 trace → 404，有 → 200 JSON，超时 → 200 空 `turns`）；页面里有大脑这一栏的元素

## 6. 真机验证

`python -m skydango run --brain --view --duration 300`（dry-run，`say` 不会真的发出去）：
在浏览器里确认轮次、原因、收到的消息、工具调用和返回、结果都显示出来，进行中的一轮实时更新；截图存 `tmp/`。
顺带核对 stream-json 里实际有没有 `user` / `tool_result` 消息、有没有思考原文——如果 `tool_result` 不在 stream 里，
改成在 `ToolBox.run` 里记返回（它知道工具名、参数、结果、是否出错），接口不变。

## 7. 文档

- CLAUDE.md 的「识别可视化」一节补一句：`run --brain --view` 时画面下方有大脑时间线
- 本文档状态改成"已实现"
