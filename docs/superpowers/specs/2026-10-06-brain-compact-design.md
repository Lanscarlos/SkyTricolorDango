# OpenAI 兼容大脑的历史：只往后接 + 超预算压成前情提要（`[brain.compact]`）

2026-10-06。只管 `brain/toolloop.py` 的 `ToolLoopBrain`（DeepSeek 等 OpenAI 兼容的大脑）；常驻 Claude Code 的 `BrainSession` 自己会压缩，不受影响。

## 起因（2026-10-06 沙盒实测）

`ToolLoopBrain` 每次请求重发：系统提示词（约 1.3 万 token，启动时拼好，之后不变）+ 最近 `[brain] history`（8）轮的唤醒消息和工具调用（`_replay`，`HISTORY_CHARS` 12000 字）+ 本轮唤醒消息；一轮通常 2 次以上请求。

1. 8 轮之前这次上线里聊过的事大脑就不知道了，只能靠它主动调 `recall` / `chat_log`
2. 窗口满了之后每轮丢最老的一轮，系统提示词之后的整段历史前缀都变了，DeepSeek 前缀缓存（命中 ¥0.02 / 百万、没命中 ¥1 / 百万）每轮对不上：41 轮大脑花 ¥0.22，八成是没命中的输入，平均每轮约 4.3k token 没命中
3. 随手记 inbox.md 只在启动时进系统提示词，这次上线中途记下的大脑看不到

## 目标和不做的

- 一次上线内历史**只往后接**（两次压缩之间每个请求都是上一个请求的前缀加东西），缓存一路命中
- 上一次请求的 `prompt_tokens` 到 `budget` 才压一次：老的部分（连同上一份前情提要）写成「这次上线到现在的前情提要」，最近 `keep_turns` 轮保留原话
- 中途新记的 inbox 接在下一轮唤醒消息末尾
- 压缩失败退回现在的滑动 8 轮
- 最终反思多拿一份最后的前情提要当材料（长的一晚日记也写得全）
- **不做**：前情提要不写进任何记忆文件、不进下次的「上次聊到哪」（下次接得上话由「上次聊到哪」/「日子」的日记 / 最终反思的要点管）；Claude 大脑不动；不改 `_replay` 的压法（每轮结束压成短的样子照旧，见 §1 的代价）

### 预算怎么定的（粗算，沙盒实测后再调）

命中缓存便宜但不白给，不压的话一晚上累计发出的 token 随轮数平方涨。按一轮历史约 2k token、一轮 2 次请求、压一次的开销约 ¥0.009（前情提要 + 保留的几轮第一次没命中约 5k，加输出约 1k）：

| 预算 | 两次压缩之间 | 每轮（命中 + 压缩分摊） |
|---|---|---|
| 3.2 万 | 约 7 轮 | 约 ¥0.0020 |
| **6.4 万（默认）** | 约 23 轮 | 约 ¥0.0020 |
| 10 万 | 约 41 轮 | 约 ¥0.0026 |
| 20 万 | 约 91 轮 | 约 ¥0.0044 |
| 不压（一晚约 300 轮） | — | 平均约 ¥0.012 |

现在约 ¥0.0054 / 轮。最省的点在 4~6 万，曲线很平；再往上主要是命中的钱涨，而且攒几百轮它自己说过的原话，模仿旧回复的毛病（CLAUDE.md「记忆」「去 AI 味」）更重。窗口大小不是限制。

## §1 请求里装什么

```
[system]    系统提示词 + BLIND_NOTE (+ ASIDE_NOTE)        ← 启动时一次，永远不变
[user]      前情提要（压过才有，见 §2.3 的格式）
[assistant] （知道了）                                    ← 占位，别让两条 user 挨着（deepseek-reasoner 不收）
[user] 唤醒消息 → [assistant / tool …] 这一轮做了什么      ← 最近几轮 + 之后接上的，_replay 压过的样子
…… 只往后接 ……
[user]      这一轮的唤醒消息（+ §3 新记的 inbox）
```

- 前情提要放在**历史开头**，不接在系统提示词后面：接在后面每压一次 1.3 万的系统提示词都要没命中一次（约 ¥0.013）
- 每一轮仍按 `_replay` 存成短的样子（唤醒消息截 `HISTORY_TEXT` 1500 字、工具返回截 200 字、id 换成 `h<seq>_<n>`），**存下就不再变**。代价：这一轮里发出去的是完整的样子，下一轮起是短的样子，所以每轮还有约 1~2k token 没命中（新唤醒消息 + 上一轮的短样子）；换来历史长得慢
- 开着时 `_past` 不再按 `history` 截（只靠压缩收），`_history_messages` 的 `HISTORY_CHARS` 也只在滑动模式（§4）用

## §2 压缩

### 2.1 什么时候

每轮 `send` 结束时，这一轮**最后一次请求**的 `prompt_tokens` ≥ `budget`，且：没有压缩在跑、不在失败冷却里（§4）、`_past` 比 `keep_turns` 多 → 在后台线程起一次压缩。同一时间最多一个。取不到 `usage`（`prompt_tokens` 没给）的那一轮不判。

### 2.2 怎么压

- 切点 `cut = len(_past) - keep_turns`：被压的是 `_past[:cut]` 和当前的前情提要；`_past[cut:]` 留原话
- **请求接着大脑自己缓存好的前缀发**（用户选的方案 A）：同一个 client、同一个模型、同一个 `tools`，`messages` = 触发那一轮最后一次请求的完整 `messages`（system + 前情提要 + 历史 + 这一轮的唤醒消息和它做的事，以它最后那条 assistant 结尾）+ 一条 user 压缩指令；`tool_choice = "none"`，`temperature` 用 0.3，`max_tokens` 用 `[models.brain]` 的
- 压缩指令（`COMPACT_REQUEST`）要点：
  - 把「前情提要」和 `[<第一条留原话的唤醒消息的时间戳>]` 之前的那些轮写成一份新的前情提要，那条之后的不用写（它们会原样留着）
  - 写：这段时间谁来过 / 走了（大概几点）、跟谁聊了什么（具体的事、名字、数字照原话）、你答应过什么还没做、谁说过难过 / 不舒服要收着点、聊出来的约定和老梗、你刚记下的事；不写：心情精力、状态里每次都有的东西、你调了什么工具
  - 用「你」称呼团子自己，按时间顺序，最多 `recap_max` 字；只写前情提要本身，不要开场白
- 结果：去掉首尾空白；空、带工具调用、超时、`ModelError` → 失败（§4）；超过 `recap_max × 1.5` 字截到 `recap_max` 字加「……」（宁可截也不丢）
- 超时用 `[brain] turn_timeout`

### 2.3 换上

压缩线程把结果交回（加锁的一个槽），**下一次 `send` 开头**换上：`recap = 新文字`、`_past = _past[cut:]`（压缩期间新接的几轮还在后面）、`compactions += 1`。换上那一轮的请求从前情提要起都没命中一次，之后又一路命中。

前情提要那条 user 消息的格式：

```
（这次上线到现在的前情提要，第 N 次整理，写到 21:03 为止；之后的原话在后面）
……正文……
```

## §3 中途新记的 inbox

- `ToolLoopBrain` 多一个 `inbox: Callable[[], str] | None`（cli 给 `store.inbox`，dry-run 也给：dry-run 不写 inbox，读到的不会变）
- 建会话时把当时 inbox 的非空行记进 `_seen`（系统提示词里已经有了）；每次 `send` 前读一次，**不在 `_seen` 里的新行**接在唤醒消息末尾、记进 `_seen`：

  ```
  你刚记下：
  - 小明下周考试
  ```

- 按行内容记，不按位置：整理 notes.md（`consume_inbox`）会挪走行
- 接上的文字就是这一轮唤醒消息的一部分，跟着进历史、压缩时自然进前情提要
- 读出错只记日志，这一轮不带
- 懒建的备用 `ToolLoopBrain`（主换成另一家 OpenAI 兼容）建的时候按当时的 inbox 记 `_seen`，启动后到切换之间记的那几行它看不到（备用本来就没有历史，可以接受）

## §4 失败了怎么办

- 压缩失败（闸关了 / 额度 / 余额 / 认证、超时、空结果）：日志 WARNING，进**滑动模式**：每次请求 = system + 前情提要（有的话）+ `_history_messages()`（最近 `[brain] history` 轮、`HISTORY_CHARS` 字，和现在一样）+ 本轮；`_past` 照样往后接（不丢，给下一次压缩用）
- 失败后 `retry` 秒内不再起压缩；冷却过了，下一轮结束时满足 §2.1 就再试，成功就回到只往后接
- 一直压不成、`_past` 长到上一次请求 `prompt_tokens` ≥ 2 × `budget`：也进滑动模式（防止压缩线程卡住时越拖越长）
- `ModelError` 不关闸：大脑自己下一次请求撞上了照旧由 `Brain._failed` 关闸、切备用
- 大脑切到 Claude 备用：前情提要不带过去（照旧）

## §5 用量

- `models/config.py` 的 `USES` 加一个 `Use("recap", "压缩", "大脑历史太长时写成前情提要（跟着大脑的模型，不单独选）", …)`，带一个新字段 `follows = "brain"`：
  - 解析配置时它的主 / 备永远等于 brain 的（`[models.recap]` 写了只警告、不生效）；预检、`requirements`、`log_summary`、status 的「模型」一行不单独报它
  - 管理面板「模型」页这一行只显示「跟着大脑」、没有下拉、不写进 console.toml
- `ToolLoopBrain` 多一个 `meter` 和 `backup: bool`（`make_session` 传 `registry.meter`；主还是备由 `_brain_sessions` 传），每次压缩 `meter.record("recap", provider, model, backup=…, usage=…, ok=…)`（usage 同大脑：input / output / cache_read_input_tokens）；用量卡片、`runs/usage.json` 里单独一行

## §6 看得到的地方

- 大脑时间线（`brain/trace.py`，管理面板大脑控制台）：压缩换上时多一条「── 压缩：第 N 次，压掉 18 轮 → 前情提要 1200 字，用时 8 秒 ──」，点开看全文；失败也记一条（原因、进了滑动模式）
- `agent.log`：起压缩、换上、失败各 INFO / WARNING 一行
- `runs/<…>/brain/recap.jsonl`：每次换上一行（时间、第几次、压掉几轮、字数、用时、usage、全文），事后核对它记得对不对；失败也记一行
- `brain.jsonl` 每轮多 `history_mode`（append / sliding）和 `recap`（第几次，0 = 还没压过）

## §7 最终反思带上前情提要

- `ToolLoopBrain.recap`（字符串，没压过是空）；cli 给身体 `body.recap_text = lambda: getattr(brain.session, "recap", "")`
- `Body.reflect_materials(final=True)` 把它交给 `reflect_materials_text`，不为空时在聊天原话前面加一节：

  ```
  ## 这次上线早些时候（前情提要，大脑自己写的）
  ……
  ```

- 只给下线那次（`final`），中途的反思不带（它们只看上次反思以来的）；正在跑的压缩不等，用最后一份换上的
- `[brain.compact] enabled = false` 或 Claude 大脑：没有这一节，材料逐字照旧

## §8 配置

```toml
[brain.compact]        # 只对 OpenAI 兼容的大脑生效
enabled = true         # false = 逐字照旧：滑动 8 轮、唤醒消息不带新记的 inbox、没有 recap 用量、最终反思材料不变
budget = 64000         # 上一次请求的 prompt_tokens 到这个数就压
keep_turns = 6         # 压完留几轮原话
recap_max = 1500       # 前情提要最多多少字
retry = 300            # 压缩失败后隔多久再试（秒）
```

- `BrainConfig` 下嵌一个 `CompactConfig`（`_merge` 支持嵌套 dataclass）
- 管理面板设置页加 `brain.compact.enabled` / `budget` / `keep_turns`
- 默认开着合并，不合适在设置页关掉

## §9 测试（假的 OpenAI 兼容客户端，记下每次请求的 messages / kwargs）

1. 两次压缩之间，每次请求的 `messages` 都以上一次请求的 `messages` 为前缀（跨轮：上一轮最后一次请求去掉这一轮的部分后，按 `_replay` 的样子比）
2. `prompt_tokens` 没到 `budget` 不压；到了只起一次；压缩请求的 `messages[:-1]` 等于触发那一轮最后一次请求的 `messages`、最后一条是压缩指令、`tool_choice == "none"`、`tools` 一样
3. 换上后：请求里第二条是前情提要、第三条是占位 assistant，后面是最近 `keep_turns` 轮 + 压缩期间接上的轮；再往后又只往后接
4. 第二次压缩的指令里带上一份前情提要（在被压的前缀里），结果替换旧的
5. 压缩失败（抛 `ModelError`、空结果、带工具调用）→ 滑动模式（请求和 `enabled = false` 时的历史一样，前面多前情提要）；`retry` 秒内不再试，过了再试、成功回到只往后接
6. 压缩一直不回来、`prompt_tokens` ≥ 2 × `budget` → 滑动
7. inbox：启动时就有的行不带；新加的行下一轮带、只带一次；被挪走再加回同一行不重复带
8. 用量：每次压缩 `meter.record("recap", …)` 一笔，`backup` 照传
9. `enabled = false`：请求逐字和现在一样（拿现在的代码录一份当基准）
10. 最终反思材料：有前情提要时多那一节，没有 / 关着时逐字照旧
11. `recap` 用处：`[models.recap]` 写了只警告、主备跟着 brain；模型页那一行没有下拉

## §10 沙盒验证（用真 DeepSeek）

1. 临时把 `budget` 调到 32000（30 轮里能触发），重置沙盒记忆、启动
2. 第 3 轮冒充好友说一件具体的事（「我下周三考物理」），之后正常聊 30 轮以上，中间让随手记记下点东西，看下一轮唤醒消息里有没有「你刚记下」
3. 第 20 轮之后问「我哪天考什么来着」：答对；看 `recap.jsonl` 里前情提要写没写这件事
4. 对比：同样的剧本 `enabled = false` 跑一遍，看用量卡片 / `runs/usage.json` 的大脑命中率和每轮花费（目标：没命中的输入明显少于现在的每轮约 4.3k）
5. 下线，看日记有没有写到前情提要里的早先的事
6. `budget` 恢复默认

## 风险

- DeepSeek 的前缀缓存怎么把 `tools`、`tool_choice` 算进前缀没有文档保证：`tool_choice = "none"` 如果改变了渲染，压缩请求会整段没命中（约 ¥0.06 一次，可接受）。沙盒第 4 步看 recap 那行的命中数，对不上就改成不传 `tool_choice`、只在指令里说别调工具，返回带工具调用按失败处理
- 前情提要写漏 / 写错：大脑会照着错的说。指令要求名字、数字照原话；`recall` 照旧能查原话
