# Claude 总闸：额度 / 认证出错后一次性调用不再撞墙，记忆和反思改走 DeepSeek（设计）

2026-10-04。起因：10-03 晚真机复盘（`docs/progress/2026-10-03-plan.md`「10-03 晚真机复盘」→ DeepSeek 备用大脑一节）。
大脑之外还有好几处各自起一次性 `claude -p`：随手记 / 整理 notes（`chat/memory.py` 的 `NotesKeeper`，经 `brain/claude.py` 的 `ClaudeLlm`）、反思（`inner/reflect.py`）、
眼睛（`brain/eyes.py`）、装扮描述（`vision/wardrobe.py`）。它们互相不知道 Claude 已经不能用了：

- 那晚 `ANTHROPIC_BASE_URL` 泄漏（已修，0bff728）导致每次都 401：一次运行起了 31 个失败的 claude 进程，随手记失败 118 次、整理 notes 失败 61 次，**5 次上线的日记、要点全没写**
- 整理 notes 失败后不清"待整理"，下一轮接着试（`NotesKeeper.maybe_update`）
- 大脑循环（`brain/loop.py` 的 `_failed`）只在 `ClaudeError.limit`（429 / 额度字样）时切 DeepSeek 备用大脑；401 这类认证错误只退避，最后掉到纯文字备用回复
- 时间线（`brain/trace.py`）里 DeepSeek 那几轮没有它和工具调用一起回的中间文字、token 用量是 null、状态里的模型名还是 sonnet

## 0. 已定的事

| 问题 | 结论 |
|---|---|
| 哪些错误算"Claude 不能用了" | 额度（`limit`：429 / limit / 上限 / 额度，原有判断）和**认证**（新：401 / authentication / invalid api key / OAuth token 失效之类）；超时、进程起不来、别的错误不算 |
| 切回 Claude 吗 | **这次运行里不切回**（用户 10-04 定）：和大脑"切过去就不切回"一致；额度通常几小时才恢复，认证错本来就恢复不了。代价：额度恢复了要重启团子才回 Claude |
| 记忆 / 反思怎么办 | 改走 `[llm]`（DeepSeek）：`LlmClient.complete(system, messages, max_tokens)` 和 `ClaudeLlm` 同一个接口，一个包装就能切 |
| 眼睛 / 装扮描述怎么办 | DeepSeek 看不了图：闸关了就**跳过**（不起进程、不重试），眼睛留着旧描述 |
| 时间线 | 补 DeepSeek 的中间文字、usage，切过去后状态里的模型名显示 `[llm] model` |

没选的：每处各自加退避（还是各撞一次墙、401 照样撞满；也不会改走 DeepSeek）；额度错后定时试探 Claude（用户选了不切回）。

## 1. 总闸 `ClaudeGate`（`brain/claude.py`）

```python
class ClaudeGate:
    """这次运行里 Claude 还能不能用（大脑和一次性调用共用一个）。关了就不再开。线程安全（记忆、反思、眼睛各在自己的线程）。"""
    def ok(self) -> bool: ...
    def trip(self, kind: str, detail: str) -> None: ...   # kind = "limit" / "auth"；只第一次记 WARNING
    reason: str | None   # 关闸原因（"额度用完" / "认证失败：…"），状态 / 日志用
```

- `ClaudeError` 加 `auth: bool`（和 `limit` 并列）；`check_result` 认认证错：`api_error_status == 401`，或文字里有 `authentication` / `invalid api key` / `invalid x-api-key` / `oauth token`（不分大小写）
- `kind(exc)`：`"limit"` / `"auth"` / `None`；只有前两种关闸
- 一次运行一个闸：`cli._run_brain` 建好，交给大脑循环和下面每个包装；`memory update`、管理面板的测试按钮这些单独的一次性命令不用闸（照旧）

## 2. 大脑循环（`brain/loop.py`）

- `_failed`：`limit` **或 `auth`** → 关闸 + 切 DeepSeek 备用大脑（原来只认 `limit`）；没有备用大脑（没开 / 没 Key）时照旧退避，认证错按 `limit_retry` 退避（不是 10/30/60 秒那档：认证错重试也没用）
- 闸已经被别处（比如随手记）关上、大脑还在用 Claude：**下一轮开始前**就切过去，不用等大脑自己撞一次（`_wake` 开头看 `gate.ok()`）
- 切过去时日志写原因（"Claude 认证失败，切到 DeepSeek 备用大脑"）

## 3. 一次性调用

### 3.1 文字的：随手记、整理 notes、反思 → `GatedLlm`

```python
class GatedLlm:
    """先 Claude（ClaudeLlm），闸关了或这次撞上额度 / 认证错 → 关闸、这一笔当场改用 backup（[llm] DeepSeek）。backup 为 None：抛原来的错。"""
    def __init__(self, claude: ClaudeLlm, backup: LlmClient | None, gate: ClaudeGate): ...
    def complete(self, system, messages, max_tokens=None) -> str: ...
```

- 闸关着：直接用 backup，不起 claude 进程
- Claude 失败但不是额度 / 认证（超时之类）：照旧抛，由调用方按原来的方式处理（不换 DeepSeek，免得一次超时就改道）
- backup 用 `make_llm(cfg.llm)`，但 `max_tokens` 取足够长（反思要写日记：沿用 `[brain] fallback_max_tokens`）、`max_retries` 照 `[llm]`
- `[llm]` 没 Key / `openai` 没装：backup = None，闸关了以后这些调用照旧失败（和现在一样，只是不再起进程）
- 反思走 DeepSeek 时提示词、JSON 格式不变（`parse_reflection` 已经能合并多段 JSON）；`Reflector` 里"额度用完 10 分钟不反思"在闸关后不再触发（GatedLlm 已经改道）

### 3.2 整理 notes 失败别每轮重试（`chat/memory.py`）

- `NotesKeeper.update_now` 失败后记 `_retry_after = wall() + NOTES_RETRY`（600 秒），`maybe_update` 在这之前不试；成功清掉
- 随手记（`jot`）失败照旧跳过这一笔（本来就不重试）

### 3.3 看图的：眼睛、装扮描述 → 闸关了跳过

- `cli` 里给眼睛的 `describe`、给装扮描述器的 `describe` 外面包一层：闸关着 → 抛 `ClaudeError("Claude 不能用了，看不了图", limit=True)`，不起进程；撞上额度 / 认证错 → 关闸再抛
- 眼睛：已有的"这次没看成，等下一个时机"照旧，但闸关着时 `tick` 直接不看（`due()` 之前判断），日志不再每次 WARNING；`look` 工具在 DeepSeek 大脑里本来就没有
- 装扮描述器：已有的 `limit` → 等 `quota_wait` 照旧；闸关后等多久都没用，直接不排描述（`_limit_until = inf`）

## 4. 时间线（`brain/deepseek.py`、`brain/loop.py`）

- `DeepSeekBrain`：一次回复里 `content` 和 `tool_calls` 同时有时，先把 `content` 作为 `{"type": "assistant", "message": {"content": [{"type": "text", "text": …}]}}` 交给 `on_message`（时间线里显示成"说 / 想"那一行，同 Claude）
- `send()` 的结果带 `usage`：把每次请求的 `resp.usage.prompt_tokens` / `completion_tokens` 累加成 `{"input_tokens": …, "output_tokens": …}`（`trace.py` 和 `brain.jsonl` 认这两个键）；`num_turns` = 请求次数；拿不到 usage（假客户端、接口没回）就是 None
- `loop.trace_state()` 的 `model`：在用备用大脑时是 `[llm] model`（比如 `deepseek-chat`），否则照旧 `[brain] model`

## 5. 状态 / 日志里看得到的

- 关闸时一条 WARNING："Claude 不能用了（认证失败：…），这次运行里记忆、反思改走 DeepSeek，眼睛和装扮描述停用"
- 大脑时间线状态：`on_fallback` 已有；加 `claude_gate`（关闸原因，没关是 None）
- 之后的一次性调用改道 / 跳过只记 DEBUG（不刷屏）

## 6. 测试（都不碰真 Claude / DeepSeek）

- `check_result`：401、"Authentication Fails, Your api key: ****gwAA is invalid"（10-03 晚原文格式）→ `auth = True`；429 → `limit`；超时 → 都不是
- `ClaudeGate`：trip 一次后 `ok()` 一直 False；只记一次 WARNING；多线程同时 trip 不出错
- `GatedLlm`：闸开 → Claude；Claude 认证错 → 关闸、这一笔用 backup；闸关 → 不调 Claude（假 ClaudeLlm 计数为 0）直接 backup；Claude 超时 → 照旧抛、不关闸；backup None + 闸关 → 抛
- 大脑循环：认证错也切 DeepSeek；闸被别处关了 → 下一轮开始前就切；没有备用大脑时认证错按 `limit_retry` 退避
- `NotesKeeper`：整理失败后 600 秒内 `maybe_update` 不再调 llm
- 眼睛 / 装扮描述：闸关后 `tick` 不调 describe、描述器不排
- `DeepSeekBrain`：content + tool_calls → on_message 先给 text；usage 累加；loop 的 `model` 在备用大脑时是 `[llm] model`
- `cli._run_brain` 接线：同一个闸交给 loop、NotesKeeper、Reflector、眼睛、描述器（沙盒世界也一样）

## 7. 不做

- 额度恢复后自动切回 Claude（用户定了不切回）
- DeepSeek 看图（没有视觉模型）
- `memory update` 命令、管理面板测试按钮、`perception label --assist` 等单独的一次性命令接闸（它们自己会报错，人在旁边）
- 闸状态存盘（重启就重新试 Claude）

## 8. 真机 / 沙盒验证

1. 本机 `config.toml` 的 `force_fallback` 删掉，`SKYDANGO_CLAUDE_TOKEN` 临时改成错的，沙盒启动：日志一条"Claude 不能用了（认证失败…）"、大脑切 DeepSeek、随手记和反思照常写（看 `sandbox/memory/inbox.md`、内心页流水账）、没有成串的 claude 进程（任务管理器）
2. 下线：日记照写（走 DeepSeek）
3. 时间线：DeepSeek 那几轮有中间文字、tokens，状态模型名 `deepseek-chat`
4. 令牌改回来、重启：回到 Claude
