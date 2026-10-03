# DeepSeek 备用大脑：Claude 额度不足时顶上（设计）

2026-10-03。起因：大脑 = 常驻的 Claude Code（订阅登录，`claude -p`），订阅额度用完（429）后现在只会退避 600 秒，聊天交给纯文字的「备用回复」（`chat/responder.py` 走 `[llm]` DeepSeek，**没工具**）——团子只能干巴巴回一句话，不能说话做事。用户要的是：**大脑还是默认 Claude，Claude 额度不足时 DeepSeek 顶上做「备用大脑」**（带工具、能做事），并且切过去就不再自动切回。

## 0. 已定的事

| 问题 | 结论 |
|---|---|
| 什么时候切 | Claude 订阅额度不足（`ClaudeError.limit == True`，即 429 / 额度字样）→ 立刻切，不等 `limit_retry` |
| 切不切回 | **不自动回**：切过去后一直用 DeepSeek 到本次 run 结束；DeepSeek 自己失败只退避重试，不回 Claude |
| 看得见吗 | **盲跑**：DeepSeek API（`deepseek-chat`）不支持看图；Claude 额度没了眼睛（haiku）也没额度。备用大脑只靠文字（状态、聊天记录、OCR/YOLO 认出的身边人名地名）+ 眼睛留下来的旧场景描述 |
| 留后路 | 视觉留一个 seam（`openai_tools(bind=…)` + 提示词一句"看不到画面"），以后接别的视觉模型时翻开关 + 加回工具，loop / 切换逻辑不动 |
| 做法 | 做法 A：`DeepSeekBrain` 用 function-calling 循环复用同一个 `ToolBox`，`send(text) -> result` 签名和 `BrainSession` 一致，`loop.py` 只换 `session` |
| 工具范围 | 全部工具去掉 5 个看图/点人的：`look` / `look_at` / `look_person` / `look_around` / `check_friend`；其余照旧（都是身体动作或文字，不依赖 Claude） |

没选的做法：做法 B（只给 say/emote/status 几个工具的单轮降级版，能力差一截）；做法 C（把大脑抽象成 `[brain] provider` 可插拔后端，一次做重构 + 兜底两件事，YAGNI 偏重）。都放弃。

## 1. DeepSeek 备用大脑（新模块 `brain/deepseek.py`）

一个类 `DeepSeekBrain`，签名和 [session.py](src/skydango/brain/session.py) 的 `BrainSession` 对齐：

```python
class DeepSeekBrain:
    def __init__(self, client, system, toolbox, tools, turn_timeout): ...
    def send(self, text: str) -> dict: ...   # 失败抛 ClaudeError(msg, limit=False)
```

`send()` 内部是 OpenAI function-calling 循环：

```
messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
deadline = now + turn_timeout
循环（最多 `[brain] max_steps`（默认 6）+ 2 次工具回合，或超 deadline；`max_steps` 由构造时传入）:
    resp = client.chat.completions.create(model, messages=messages, tools=tools, temperature, max_tokens=fallback_max_tokens)
    msg = resp.choices[0].message
    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
    没有 tool_calls → break，最后文字 = msg.content
    有 → 逐个执行：
        out, err = toolbox.run(tc.function.name, json.loads(tc.function.arguments))
        content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)  # 盲跑下都是 str，这只是兜底
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
return {"result": 最后文字, "subtype": "success"}
```

要点：

- **工具执行完全复用 `ToolBox.run`**（[tools.py:171](src/skydango/brain/tools.py#L171)）：参数校验、每轮 6 步 / 2 句限额、`acted` 统计、`body.call` 的身体线程排队都照旧——`Brain._wake` 每轮醒来前已经调过 `toolbox.begin_turn()`。
- **客户端**：直接用 `openai` SDK 按 `[llm]` 建（`base_url` / `model` / `api_key_env` 读 Key，`timeout=cfg.llm.timeout`、`max_retries=0`）；在 [chat/llm.py](src/skydango/chat/llm.py) 不动现有 `complete()` 契约，`deepseek.py` 里自己建 client（避免把 tools 塞进 `Responder` 走的那条简单路径）。
- **超时**：整个 `send()` 受 `[brain] turn_timeout` 限制（和 Claude 的 `until_result(turn_timeout)` 语义一致），超时抛 `ClaudeError("超时")`。
- **失败**：DeepSeek API 报错 / 超时抛 `ClaudeError(msg, limit=False)`——`Brain._failed` 已经会按 `BACKOFF` 退避，`limit=False` 保证不会触发"切回"逻辑。

## 2. 工具 schema 与盲跑 / 留后路（新模块 `brain/llm_tools.py`）

单一来源 `openai_tools(toolbox, *, blind=True) -> list[dict]`，输出 OpenAI 格式：

```python
{"type": "function", "function": {"name": "say", "description": DESCRIPTIONS["say"],
   "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
```

- **描述**复用 [tools.py](src/skydango/brain/tools.py) 的 `DESCRIPTIONS`（已经是单一来源）。
- **参数**：一张紧凑表（`name → 参数列表（名字、类型、默认值）`），和 [mcp_server.py](src/skydango/brain/mcp_server.py) 的 `@srv.tool` 签名、[tools.py](src/skydango/brain/tools.py) 的 `_bind` 三处保持同步——文件里加注释点名这三处，改参数时一起改。
- **盲跑排除集**：`BLIND_EXCLUDE = {"look", "look_at", "look_person", "look_around", "check_friend"}`。`blind=True` 时从 schema 里去掉这些工具；`panel_read` 也去掉 `image` 参数（只给文字版）。保留的工具全部不依赖 Claude（`status` / `chat_log` / `recall` / `say` / `emote` / `set_request_policy` / `camera` / `camera_reset` / `attention` / `move` / `track` / `find` / `stop_task` / `panel_read` / `panel_press` / `panel_close` / `call` / `introspect`，按现有开关 `sweep` / `backstage` / `call` 增删）。
- **留后路**：以后接上别的视觉模型 = 把 `blind` 翻 `False`（加回那 5 个工具）+ 换一个能看图的 client，`loop.py` 和切换逻辑一行不动。`blind` 语义只影响工具列表和提示词，不影响 `ToolBox` 本身。

## 3. 切换与错误处理（[loop.py](src/skydango/brain/loop.py)）

`Brain` 增加一个可选的 `fallback_session`（`DeepSeekBrain`，没有就退化成今天的纯文字备用）和一个 `on_fallback` 标志：

1. **切过去**：`_failed()` 里，`exc.limit == True` 且有 `fallback_session` 且还没切换 → 永久 `self.session = self.fallback_session`、`self.on_fallback = True`，并清掉失败态（`failing_since = None`、`backoff_until = -inf`、`limited = False`）——下一轮立刻用 DeepSeek，不等 `limit_retry`。日志一条"Claude 额度用完，切到 DeepSeek 备用大脑"。
2. **不再走纯文字备用**：`offline()` 在 `on_fallback` 时，只要 DeepSeek 还没连续失败（`failing_since` 为 None 或没过 `offline_fallback`）就返回 `False`——[body.py:491](src/skydango/brain/body.py#L491) 不再把聊天丢给纯文字 `Responder`，而是照常排进大脑事件，由 DeepSeek 大脑自己 `say` / `emote`。
3. **不切回**：DeepSeek 失败照 `_failed` 的 `BACKOFF` 退避重试，**不回 Claude**。DeepSeek 也连续失败过 `offline_fallback` → `offline()` 回到 `True`，此时兜底到纯文字备用回复（也走 `[llm]` DeepSeek；等于没救了就闭嘴）。
4. **附带**：退出前写"这次的经过"（`farewell`）在切换后也交给 DeepSeek 写，不丢；`trace_state()` 加 `on_fallback` 键，网页 / 控制台标"备用大脑（DeepSeek）"（区别于现在的"已转备用回复"）。

## 4. 提示词

复用 [prompt.py:224](src/skydango/brain/prompt.py#L224) `brain_prompt()` 产出的**整段**（在 [cli.py:2429](src/skydango/cli.py#L2429) 现算一次，Claude 和 DeepSeek 共用），DeepSeek 版再追加一小段 `FALLBACK_NOTE`：

> 你现在是备用大脑（DeepSeek）：Claude 额度不足，眼睛（看图的）也不可用，看不到画面。判断只靠消息里的状态、聊天记录、身边人名地名（身体 OCR/YOLO 认的）和旧场景描述；工具里没有看图 / 点人的，别硬调。

这段 system 直接进 DeepSeek 请求，不走 `--append-system-prompt-file`。眼睛（`Eyes`）本身**不改**：额度没了一时 `describe` 会失败、`summary()` 返回旧描述，`Brain.message` 照旧把它拼进醒来消息——DeepSeek 看到的是"旧场景描述"，这没问题。

## 5. 配置（`[brain]`，`config.py` 的 `BrainConfig`）

| 键 | 默认 | 说明 |
|---|---|---|
| `fallback` | `True` | Claude 额度用完是否切 DeepSeek 备用大脑。`[llm]` 没 Key 时自动退回现在的纯文字备用回复（不硬报错） |
| `fallback_max_tokens` | `4096` | 备用大脑回复长度（`[llm] max_tokens=200` 对大脑太短；DeepSeek 输出上限 8192） |
| `force_fallback` | `False` | 调试：**从启动就用 DeepSeek 大脑**，不等额度真用完就能验证这套循环（眼睛 / 记忆 / 反思仍走 Claude，因为它们不受 `force_fallback` 管） |

DeepSeek 的 `base_url` / `model` / `api_key_env` / `temperature` / `timeout` 都复用 `[llm]`，不另加键。

## 6. 测试

- 单测（假 client，注入 `chat` 返回脚本化 tool_calls）：循环——一次工具调用 / 多步连续 / 步数上限（模型硬调时 `ToolBox.run` 回"这一轮做的事够多了"）/ 工具结果非 str 兜底 / API 报错抛 `ClaudeError(limit=False)` / 超 `turn_timeout` 抛错。
- 单测（假 Claude `BrainSession`）：`ClaudeError(limit=True)` → 断言 `Brain` 换成 `fallback_session`、`offline() == False`、下一条消息走 DeepSeek；DeepSeek 报错 → 退避且**不切回**；DeepSeek 连续失败 → `offline()` 回 `True`。
- 回归：`python -m pytest -q` 全绿；`fallback = False`（或没 Key、没 `fallback_session`）时整条路径惰性、行为逐字照旧。
- 时间线：`trace_state()` 的 `on_fallback` 在网页 / 控制台有标识。

## 7. 验证

1. `python -m pytest -q` 全绿。
2. 沙盒：`[brain] force_fallback = true` 起沙盒，冒充好友说几句，看聊天记录里 DeepSeek 大脑会不会用 `say`/`emote` 回、大脑控制台标"备用大脑（DeepSeek）"。
3. 真机（个人电脑晚上）：`force_fallback = true` + `run --view --dry-run`，看时间线里 DeepSeek 的 tool_calls 和结果，确认不会去调被排除的视觉工具、身体动作工具正常。
4. 真实额度触发（可选，等真 429）：确认自动切、不切回。

## 还没做 / 已知限制

- DeepSeek 的工具调用 / agentic 能力比 Claude Code 弱，6 步 + 长提示词这套可能要多调提示词；`fallback_max_tokens`、温度复用 `[llm]` 都是先取的默认，真机再看。
- 盲跑下 `check_friend`（好友树截图）不可用，大脑只能用名字标签 / 状态判断"是不是好友"。
- 备用大脑没做 DeepSeek 自己的额度/并发处理：DeepSeek 报错就是退避重试，到头就闭嘴（纯文字备用回复也是 DeepSeek）。
