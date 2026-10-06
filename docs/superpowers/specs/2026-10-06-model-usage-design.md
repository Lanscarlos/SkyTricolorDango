# 模型用量和额度（设计）

日期：2026-10-06　状态：设计已和用户确认（聊天里），待用户审 spec

## 0. 起因和目标

10-05 起模型按用处选（`[providers]` / `[models]`），DeepSeek 按量扣钱、Claude 走订阅额度，可管理面板上看不到用了多少：
只有大脑控制台每轮的 tokens、status 里一行「模型：」。用户想在面板上——特别是「真机团子」和「沙盒」页——看到每个模型的用量和额度。

用户确认过的决定：
- 显示四样：**每个用处的用量**（按「供应商/模型」分开）、**折算花了多少钱**、**DeepSeek 账户余额**、**闸的状态和原因**（哪家关了、现在走主还是备）。
- 范围：**这次运行**（实时）+ **今天累计**（落一个小账本，停着也能看）。不做按天 / 按月的历史表。
- 单价照 DeepSeek 价格页（https://api-docs.deepseek.com/zh-cn/quick_start/pricing，10-06 取）默认填好，分空闲 / 高峰。
- **顺带这次就改**：价格页上已经没有 `deepseek-chat`，原来默认用它的用处（brain / memory / reflect / reply / text_label）默认改成 `deepseek-flash`。

## 1. 默认模型换成 deepseek-flash

- `models/config.py` 的 `USES`：brain / memory / reflect / reply / text_label 的 `main` 从 `deepseek/deepseek-chat` 改成 `deepseek/deepseek-flash`（备用不变：前三个 `claude/sonnet`，后两个没有）。
- 内置 deepseek 供应商的模型表：`deepseek-flash`、`deepseek-v4-pro`、`deepseek-chat`、`deepseek-reasoner`（旧名留着：已有配置还能选），`vision` 仍只有 `deepseek-flash`。
  面板「+ 添加供应商」的 DeepSeek 模板同样改。
- `[llm] model` 的默认值改成 `deepseek-flash`（只在旧写法 `[llm]` 写了别的字段、没写 model 时有影响）。
- **不动**：config.toml / console.toml 里明确写了 `deepseek-chat` 的照旧用它；旧 `[llm] model = "deepseek-chat"` 照旧换算成 reply 的主模型。
- 面板保存过「模型」页的机器：console.toml 里整张写着 deepseek 供应商，没列 `deepseek-v4-pro`。新默认 `deepseek-flash` 在 10-06 那次改动后多半已经加上；
  没加的启动时照现有规则警告「模型不在列表里」、照样用。用处是否改成新默认看 console.toml 有没有写那个用处（写了的不变，要新默认点「恢复默认」）。
- 要在真机核对的：`deepseek-flash` 做大脑时 function calling 正常（`ToolLoopBrain`）。

## 2. 记账：`models/usage.py`

### 2.1 `UsageMeter`

一次进程一个，线程安全，挂在 `Registry.meter` 上（`None` = 不记：预检、面板测试按钮）。

- `record(use, ref, *, backup, usage, ok, at)`：一笔调用。`usage` 是后端返回的 dict（`input_tokens` / `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens`，缺的当 0）；
  `ok = False` 只加失败次数。同时算出这一笔的钱（§3）累加进去。
- 按 (用处, 供应商/模型, 主还是备) 分行累加：次数、失败次数、输入、输出、缓存读、缓存写、钱、钱是不是估的。
- 每一笔同时记进「这次运行」和「还没写进账本的增量」（§4）。
- `rate_limit(provider, info, at)`：Claude Code 的限额事件（§5.2）；`balance(provider, info | error, at)`：余额（§5.1）。
- `snapshot(now)` → `/usage` 的内容（§6.1）。

### 2.2 在哪里记

| 地方 | 怎么记 |
|---|---|
| `GatedCall` | `_send` 成功后 `record(ok=True)`，抛错（任何错）`record(ok=False)`；主 / 备照实际走的那家记 |
| 大脑 | `brain/loop.py` 记 `brain.jsonl` 的地方，按结果里的 `provider` / `model` 记一笔 `brain`（Claude Code 常驻会话和 `ToolLoopBrain` 都经这里）；失败的一轮记 `ok=False`；切过备用的记成备 |
| `ToolLoopBrain` | 顺带修：累加 `prompt_cache_hit_tokens` 进 `cache_read_input_tokens`（现在漏了，DeepSeek 大脑缓存命中算不出钱） |

面板「模型」页的测试按钮、预检用的 `Registry` 没有 meter，不记。

### 2.3 来源

`UsageMeter(source=…)`：`live`（`run`，含 `--no-brain` 和 dry-run——模型调用是真花钱的）、`sandbox`、`offline`（`memory update`、`look`、看图标注、`addressee label` 等离线命令，`_registry()` 默认）。

## 3. 折算花费

### 3.1 配置

供应商下多三个字段（只对 `openai` 接入方式有意义；`claude-code` 走订阅，**不算钱**）：

```toml
[providers.deepseek]
# 元 / 百万 token，空闲价：命中缓存的输入、没命中的输入、输出
prices = { "deepseek-flash" = [0.02, 1, 4], "deepseek-v4-pro" = [0.15, 4.5, 13.5] }
peak = 2.0                                  # 高峰价 = 空闲价 × peak（1 = 不分时段）
peak_hours = ["09:00-12:00", "14:00-18:00"] # 北京时间周一到周五
```

- 内置 deepseek 供应商默认：上面两个模型照价格页；`deepseek-chat` / `deepseek-reasoner` 价格页上没有，**按 Flash 的价算、标「估」**（内部记一个估价模型集合）。
- 时段按 `Asia/Shanghai` 算，周一到周五；**不认法定节假日**（节假日的工作日时段按高峰算，偏高），面板上钱数一律带「估」字样的提示（悬停说明）。
- 没填单价的模型：只记 token，钱记 `None`，面板显示「没填单价」。
- 「模型」页：每个模型一行多三个小数字框（命中 / 没命中 / 输出）+ 供应商的「高峰倍数」；保存照现有规则写 console.toml。时段只在配置文件里写。

### 3.2 怎么算

OpenAI 兼容返回的 `prompt_tokens` 包含命中缓存的部分：没命中 = `input_tokens − cache_read_input_tokens`（不小于 0）。
一笔钱 = (命中 × 命中价 + 没命中 × 没命中价 + 输出 × 输出价) / 1e6 × (高峰 ? peak : 1)，按这一笔**发生那一刻**判高峰。

## 4. 今天累计的账本：`runs/usage.json`

- `[usage] ledger = "runs/usage.json"`（`runs/` 根目录，不被每次运行的轮换删；不进 git，也不进私有仓库——两台电脑各记各的）、`keep_days = 7`、`save_every = 60`、`balance_every = 300`。
- 格式：`{"version": 1, "days": {"2026-10-06": {"live": {"deepseek/deepseek-flash": {"brain": {calls, fails, input, output, cache_read, cache_write, cost, est}}}}}}`。日期按本机时间、这一笔发生的那天。
- 写入：每 `save_every` 秒和进程退出时，把「还没写进账本的增量」**加进**文件（读出来 → 相加 → 写临时文件 → 原子替换），成功后清空增量。
  同时只有一个团子 / 沙盒，但离线命令可能同时跑：用 `usage.json.lock`（`O_EXCL` 建、最多等 2 秒、超过 30 秒的旧锁当残留删掉）串起来；拿不到锁这次不写、增量留着下次。
- 离线命令：`_registry()` 建的 meter 用 `atexit` 退出时写一次。
- 删掉 `keep_days` 天以前的日期。文件坏了改名 `.bad-<时间>`、从空的开始（WARNING 一行）。写盘出错只记日志，不影响团子。

## 5. 额度

### 5.1 DeepSeek 余额

- 对接入方式是 `openai`、`base_url` 的主机是 `api.deepseek.com` 的供应商，每 `balance_every` 秒查一次 `GET <base_url>/user/balance`（带 Key；后台线程，启动时先查一次），
  记 `is_available`、各币种 `total_balance` / `granted_balance` / `topped_up_balance` 和查的时间。别的 OpenAI 兼容家没有统一的余额接口，不查。
- 查失败（没 Key、超时、非 200）记 DEBUG，面板显示「查不到余额」+ 原因。
- 团子 / 沙盒在跑时由子进程查（`live` / `sandbox`），停着时管理面板自己查（`console/usage_view.py`，用 secrets.toml 的 Key，结果缓存 `balance_every` 秒）。

### 5.2 Claude 订阅额度

- Claude Code 的 stream-json 会出 `{"type": "rate_limit_event", "rate_limit_info": {...}}`：`status`（allowed / allowed_warning / rejected）、`resetsAt`（秒）、`rateLimitType`（five_hour / seven_day …）、
  可能有 `utilization`（0~1）和 `unifiedWindows`（five_hour / seven_day 各自的 utilization、resetsAt）。字段都按可选解析，认不出的忽略。
- `BrainSession._seen` 和 `StreamProcess.until_result` 的一次性调用（`ClaudeCodeBackend` 经 `Registry.backend` 拿到 meter）看到就交给 `meter.rate_limit(provider, info)`，按窗口留最新的一条。
- 只有在跑时才有（面板停着不知道）。

### 5.3 闸

直接读 `ProviderGates.closed()`：每家开着 / 关了 + 原因；每个用处现在实际走哪个（`Registry.current`）、是不是备用、是不是停用。

## 6. 接口和页面

### 6.1 `/usage`

团子的接口（`vision/viewer.py`）和沙盒的接口（`sandbox/server.py`）各加 `GET /usage`，返回 `meter.snapshot()`：

```json
{"ok": true, "source": "live", "started": 1759750000,
 "run":   {"rows": [{"use": "brain", "label": "大脑", "model": "deepseek/deepseek-flash", "backup": false,
                     "calls": 41, "fails": 0, "input": 512000, "output": 9100, "cache_read": 460000, "cache_write": 0,
                     "cost": 0.091, "est": true}], "cost": 0.12, "est": true},
 "today": {"rows": [...], "cost": 0.85, "est": true},
 "providers": [{"id": "deepseek", "kind": "openai", "closed": null,
                "balance": {"available": true, "items": [{"currency": "CNY", "total": 12.34, "granted": 0, "topped_up": 12.34}], "at": 1759750300, "error": null},
                "rate": null},
               {"id": "claude", "kind": "claude-code", "closed": "额度 / 余额用完",
                "balance": null, "rate": {"status": "rejected", "type": "five_hour", "utilization": 1.0, "resets_at": 1759760000,
                                          "windows": {"five_hour": {"utilization": 1.0, "resets_at": 1759760000}}, "at": 1759750100}}],
 "uses": [{"use": "brain", "label": "大脑", "current": "claude/sonnet", "backup": true, "disabled": null}]}
```

`today` = 账本里今天已经写盘的全部（所有来源、之前几次运行）+ 这个进程还没写盘的增量，按 (用处, 模型, 主 / 备) 合并，所以真机页的「今天」也包括白天沙盒、离线命令花的。
`rows` 按钱从多到少、再按次数排。

管理面板：`/live/usage`、`/sandbox/usage` 照现有转发（GET）；`GET /api/usage`：没在跑时读账本的今天 + 自己查的余额（`run` 为 `null`、`rate` 为 `null`），在跑时转发给子进程。

### 6.2 页面：`console/static/usage.js`

一个共用组件 `mountUsage(容器, 取数函数)`，每 5 秒刷新（页面不可见时不拉）：

- **每家一行**：名字 · 闸（开着 / 「关了：额度用完」红字）· 余额「¥12.34」（悬停：赠送 / 充值、几分钟前查的）或 Claude「5 小时窗口 62%，19:00 重置」（没比例只写状态）
- **每个用处一行**（这次运行）：用处 · 模型（走备用的标「备」，停用的灰）· 次数（失败数红字）· 输入 / 输出 / 缓存命中（k / M）· ¥
- 底部两行合计：这次 ¥0.12 · 今天 ¥0.85（「估」悬停说明：按价格页单价折算、不认节假日、`deepseek-chat` 按 Flash 价）
- 等宽数字，颜色只用 `console.css` 里 `:root` 的变量

放在：
- 「真机团子」页左栏「现在」下面一块可折叠的卡片「模型用量」；停着时显示今天和余额（`/api/usage`）
- 「沙盒」页左栏同样位置
- 「模型」页顶上一行：今天合计 + 余额

status 里原来的「模型：」一行不变。

## 7. 出错和边界

- 记账、算钱、余额、限额事件、写账本任何一步出错只记日志，绝不影响调用本身（包在 try 里）。
- 余额查询的线程是守护线程，退出时不等。
- 跨零点：一笔按发生那天记，这次运行的合计不分天。
- 没有 meter（预检、测试按钮）时一切照旧。

## 8. 测试

- `UsageMeter`：分行累加、主备分开、失败只加次数、算钱（命中 / 没命中拆分、高峰 / 空闲、估价模型、没单价）、snapshot 排序和合计
- `GatedCall` 主撞墙改走备时两笔各记各的；`ToolLoopBrain` 累加缓存命中
- 账本：增量合并、两个 meter 先后写不丢、锁被占不写、旧锁清掉、删 7 天前、坏文件改名
- 余额：假 HTTP 解析成功 / 非 200 / 超时；只查 `api.deepseek.com`
- 限额事件：合成 stream-json（有 / 没有 utilization、unifiedWindows、乱填的字段）
- 配置：`prices` / `peak` / `peak_hours` 解析和报错；新默认模型；旧 `[llm] model = "deepseek-chat"` 照旧
- 接口：viewer / 沙盒 `/usage`，面板 `/api/usage` 停着读账本
- `usage.js`：node 测渲染（关闸红字、备用标记、没单价、没余额）

## 9. 真机验证（做完后由用户走）

1. 管理面板起真机团子（DeepSeek 大脑），聊几句：「模型用量」里大脑、记忆的次数和 token 在涨，钱数是几分钱级别；余额有数
2. 跑完一晚，拿 DeepSeek 控制台当天的扣费和面板「今天」对一下（差多少、是不是 `deepseek-chat` 估价那部分）
3. 沙盒跑一会儿：沙盒页有用量；切回真机页「今天」包括沙盒那部分
4. 用到 Claude 的用处（看图备用、或把大脑临时切成 `claude/sonnet`）：出现 Claude 额度行；停着时卡片只剩今天和余额
