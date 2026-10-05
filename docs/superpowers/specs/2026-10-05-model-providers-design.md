# 模型供应商和按用处选模型（设计）

日期：2026-10-05　状态：设计已和用户逐节确认，待用户审 spec

## 0. 起因和目标

现在团子调模型的地方是写死的：大脑、眼睛、随手记 / 整理、反思、装扮描述、看图标注都走 Claude Code（`claude -p`，订阅令牌），
只有普通 Agent、备用回复、Claude 额度用完后的备用大脑走 `[llm]`（DeepSeek）。大脑 sonnet 常驻，热闹时一分钟四五轮（10-04 21:27 那次 32 分钟 139 轮），是订阅额度的大头。

目标（用户原话整理）：
1. **除了识图，其他全部默认换成 DeepSeek**；识图（眼睛、装扮描述、代看、看图标注）默认还是 Claude。
2. 管理面板有一个**维护模型供应商**的地方：现在支持 Claude 和 DeepSeek，以后要能加 ChatGPT 等。
3. **每个用到模型的地方都能选模型**（主模型 + 可选的备用模型）。

用户确认过的决定：
- 大脑换成看不了图的模型后，看图的工具**转给识图模型代看**（不是直接去掉）。
- **每个用处可配一个备用**；主模型那家额度 / 余额用完或认证错，这次运行里切备用。
- 接入方式这一期做两种：**Claude Code**（现在的 `claude -p`）和 **OpenAI 兼容**（DeepSeek / ChatGPT / 通义 / Kimi / Ollama）。不做 Anthropic API。
- 识图的用处**也能选 OpenAI 兼容的模型**：供应商的每个模型可以勾「能看图」，识图用处只能选勾了的。
- 配置写法用「供应商表 + 用处表」（方案 A；没选 B 各功能自带字段、C 预设档位）。

## 1. 配置

### 1.1 供应商 `[providers.<id>]`

```toml
[providers.claude]
kind = "claude-code"            # claude -p
path = "claude"                 # 原 [brain] claude_path
token_env = "SKYDANGO_CLAUDE_TOKEN"   # 原 [brain] token_env
config_dir = ".brain-claude"    # 原 [brain] config_dir
models = ["sonnet", "haiku", "opus"]
vision = ["sonnet", "haiku", "opus"]  # 能看图的（必须是 models 的子集）

[providers.deepseek]
kind = "openai"                 # OpenAI 兼容
base_url = "https://api.deepseek.com"
key_env = "DEEPSEEK_API_KEY"
timeout = 30.0                  # 原 [llm] timeout
max_retries = 2                 # 原 [llm] max_retries
models = ["deepseek-chat", "deepseek-reasoner"]
vision = []
```

- `id`：小写字母、数字、`-`、`_`，不能含 `/`。
- `kind` 只有 `claude-code` / `openai`；另有内部的 `echo`（`--echo` 用，面板不列）。
- **内置默认**：config.toml 和 console.toml 都没有任何 `[providers]` 时，用上面两家（deepseek 的地址 / Key 名 / 超时取旧 `[llm]`，旧 `[llm] model` 不在列表里就补进 models）。
  只要有一处写了 `[providers]`，就只用写了的（两个文件按 id 合并，console.toml 覆盖同 id 的字段）。
- 密钥照旧按环境变量名写 `secrets.toml`（`key_env` / `token_env`）。

### 1.2 用处 `[models.<用处>]`

```toml
[models.brain]
main = "deepseek/deepseek-chat"
backup = "claude/sonnet"        # 空 = 没有备用
# 只在配置文件里写（面板不调）：temperature、max_tokens，只对 openai 供应商生效
```

模型写成 `供应商id/模型名`，按第一个 `/` 拆（模型名里可以再有 `/`，比如 Ollama）。

| 用处 | 说明 | 主（默认） | 备（默认） | 看图 | temperature / max_tokens 默认 |
|---|---|---|---|---|---|
| `brain` | 大脑：每次醒来想、说、调工具 | deepseek/deepseek-chat | claude/sonnet | 否 | 0.8 / 4096（原 `fallback_max_tokens`） |
| `memory` | 随手记 inbox.md、整理 notes.md、`memory update` | deepseek/deepseek-chat | claude/sonnet | 否 | 0.8 / 4096 |
| `reflect` | 反思、日记、性格 | deepseek/deepseek-chat | claude/sonnet | 否 | 0.8 / 4096 |
| `reply` | 普通 Agent（`--no-brain`）、`chat` 命令、大脑离线时的纯文字回复 | deepseek/deepseek-chat | — | 否 | 0.8 / 200（原 `[llm]`） |
| `text_label` | 文字标注（`addressee label`） | deepseek/deepseek-chat | — | 否 | 0.8 / 4096 |
| `eyes` | 眼睛（截图写成文字）、给大脑代看 | claude/haiku | — | 是 | 0.8 / 2048 |
| `wardrobe` | 装扮描述 | claude/haiku | — | 是 | 0.8 / 1024 |
| `image_label` | 看图标注：`perception label --assist` / `--objects`、`attrs-label`、`gesture-label` | claude/sonnet | — | 是 | 0.8 / 8192 |

- 看图的用处主、备都必须是该供应商 `vision` 里的模型；不是就**启动报错**（这个用处在这次运行里停用并 WARNING，不让整个团子起不来），管理面板标红、预检报出来。
- 引用了不存在的供应商、或模型不在那家的 `models` 里：同上处理（模型不在列表里只警告、照样用：用户可能直接在配置里写了新型号）。
- 超时**不搬**：`[brain] turn_timeout`、`eyes_timeout`、`memory_timeout`、`[inner] reflect_timeout`、`[appearance] describe_timeout`、`[assist] timeout` 留在原处（它们是功能自己的节奏）。
- `[brain] effort` 留在原处，只对 claude-code 生效。

### 1.3 旧字段换算（读到就换、警告「已改到 [models] / [providers]」，不报错）

| 旧字段（只在配置文件里**显式写了**才换） | 换成 |
|---|---|
| `[llm] provider / base_url / model / api_key_env / timeout / max_retries` | 内置 deepseek 供应商的对应字段；`reply` 的主 = `deepseek/<model>`；`temperature` / `max_tokens` → `[models.reply]` |
| `[llm] provider = "echo"` | `reply` 的主 = 内部 echo |
| `[llm] provider = "anthropic"` | 不再支持：WARNING，`reply` 用默认 |
| `[brain] model` | `[models.brain] main = "claude/<值>"` |
| `[brain] eyes_model` | `[models.eyes] main = "claude/<值>"` |
| `[brain] memory_model` | `[models.memory] main = "claude/<值>"` |
| `[inner] reflect_model` | `[models.reflect] main = "claude/<值>"` |
| `[appearance] describe_model` | `[models.wardrobe] main = "claude/<值>"` |
| `[assist] model` | `[models.image_label] main = "claude/<值>"` |
| `[brain] claude_path / token_env / config_dir` | 内置 claude 供应商的 `path / token_env / config_dir` |
| `[brain] fallback / force_fallback / fallback_max_tokens / fallback_history` | 废弃：备用由 `[models.*] backup` 决定；`fallback_history` 改名 `[brain] history`（OpenAI 兼容大脑带几轮，默认 8） |

同一用处 `[models.*]` 和旧字段都写了：`[models.*]` 说了算。本机现在的 `config.toml` 只有 `[llm]`（DeepSeek）、没写 `brain.model`，换算后大脑按新默认走 DeepSeek。

## 2. 运行时

### 2.1 新包 `src/skydango/models/`

代码只从这里拿模型，别处不再自己拼 `claude -p` 命令或建 OpenAI 客户端。

| 文件 | 内容 |
|---|---|
| `config.py` | `ProviderConfig`、`UseConfig`、`USES`（上表：名字、说明、看图、默认主 / 备、默认参数）、旧字段换算、校验（返回问题清单给预检 / 面板） |
| `content.py` | 一份中立的消息内容：文字段 + JPEG 段；`to_claude()`（stream-json 的 text / image 块）、`to_openai()`（`image_url` 的 data URI） |
| `claude_code.py` | 搬 `brain/claude.py` 的 `StreamProcess` / `one_shot` / `check_result`、`ClaudeLlm`（文字）、看图的一次性调用（原 `eyes_command` / `wardrobe_command` / `assist_command` 合成一个带参数的命令构造）；`BrainSession` 留在 `brain/session.py`，命令从这里取 |
| `openai_compat.py` | 搬 `chat/llm.py` 的 `OpenAICompatClient`（文字 + 看图）、错误分类 |
| `gate.py` | `ProviderGates`：每家一个闸（原 `ClaudeGate` 推广） |
| `registry.py` | `Registry(cfg, gates, workdir)`：`text(use)`、`vision(use)`、`brain(...)`、`describe(use)`（给面板 / status 的一行字）；主 / 备自动切换 |

`chat/llm.py` 留 `LlmClient` 协议和 `EchoClient`，`make_llm` 改成 `Registry.text("reply")` 的薄包装（老测试的入口）；`brain/claude.py` 留兼容的 re-export（`ClaudeError` 等），新代码从 `models` 引。

### 2.2 闸和备用

- 一次运行一个 `ProviderGates`（cli 建好，大脑、记忆、反思、眼睛、装扮描述、代看共用；离线命令各自建一个）。
- **关闸**（这家这次运行里不再用）：
  - claude-code：现在 `check_result` 判的额度（limit）和认证错（auth）。
  - openai：HTTP 401 / 403（认证）、402（DeepSeek 余额不足）、429 且错误码 `insufficient_quota`（OpenAI 额度用完）。
- **不关闸**：429 限流（其余）、超时、连接错误、5xx、回答格式不对 —— 照各功能现在的重试 / 退避 / 沿用上一份。
- 文字 / 看图调用（`GatedCall`，原 `GatedLlm` / `gated_describe` 推广）：主那家闸开着走主；主出关闸类错误 → 关主的闸、这一笔改走备；主的闸关着直接走备；备也关了 / 没有备 → 抛 `ModelUnavailable`，调用方按现在"闸关了"的分支处理（眼睛 / 装扮描述停用、随手记跳过、反思沿用上一份）。
- 下线反思的超时预算（`_final_timeout`）照旧压到主和备上。
- 日志：关闸时 WARNING 一行「deepseek 余额用完（402），这次运行里 brain / memory / reflect 改用 claude/sonnet」。

### 2.3 大脑

- 会话按主模型那家的 kind 建：
  - claude-code：现在的 `BrainSession`（常驻 stream-json、MCP `sky`、`--resume`）。
  - openai：现在的 `DeepSeekBrain` 改名 `ToolLoopBrain`（`brain/toolloop.py`），function-calling 循环、每轮重发、带最近 `[brain] history` 轮。
- 备用同样按它的 kind 建，**懒建**（切过去时才起进程 / 建客户端）。主的闸关了（大脑自己的错、或别的用处先触发的）→ 切到备、不切回（同现在）。
- 大脑连续失败 `offline_fallback`（120 秒）→ 聊天交给 `reply` 用处的纯文字回复（同现在）。
- 提示词：`FALLBACK_NOTE` 去掉「你是备用大脑、Claude 额度不足」，改成中性的「你看不到画面……看图的工具会请眼睛代看，返回文字」+ 原来那几条（别重复、recall 查不到别编、说做动作就真调 emote）；只有大脑模型看不了图时加这段。`ASIDE_NOTE` 照旧。
- 「幕后」一节里的模型名（脑子 / 眼睛 / 反思）取 `Registry.describe`，切过备用后不改提示词（启动时定）。

### 2.4 代看（大脑模型看不了图时）

- 看不了图 = 大脑当前在用的模型不在那家的 `vision` 里（Claude 的 sonnet 能看，就照旧给原图）。
- 工具列表：`look`、`look_at`、`look_person`、`look_around`、`check_friend`、`panel_read` **都保留**，会返回图的这几个多一个可选参数 `question`（想看清什么，空 = 描述一下）。`llm_tools.BLIND_EXCLUDE` 删掉，`openai_tools(blind=)` 改成按"看不看得了图"加 `question`。
- `ToolBox` 拿到身体给的图时：大脑看得了图 → 照旧返回图；看不了 → 交给 `Registry.vision("eyes")`（提示词：大脑的 question + 现有的位置说明 `scene_note` / `label_note`，要求只说看得到的、别编名字），返回「（眼睛代看）……」文字。
  `look(image=true)` 代看的结果带位置说明里的坐标，大脑接着用 `look_at` / `check_friend` 时坐标照旧按 1280×720 给。
- 和眼睛共用 `look_min_interval` 频率限制；`look_person` 照旧存图到 `runs/<…>/look_person/`。
- `eyes` 用处停用（闸关、没配好）→ 返回「现在看不了图」。
- 沙盒（`text_only`）照旧只给文字，不代看。

### 2.5 各处改接

| 地方 | 现在 | 改成 |
|---|---|---|
| `cli._run_brain` 记忆 / 反思 / 眼睛 / 备用回复 | `ClaudeLlm` + `GatedLlm`、`one_shot(eyes_command)`、`make_llm(cfg.llm)` | `registry.text("memory")` / `text("reflect")` / `vision("eyes")` / `text("reply")` |
| `cli._wardrobe` | `one_shot(wardrobe_command)` + `gated_describe` | `registry.vision("wardrobe")` |
| `cli._fallback_brain` / `_gated_backup` | `[llm]` | 删掉，由 `registry.brain` 和 `GatedCall` 代替 |
| `cmd_memory update` | `ClaudeLlm(memory_model)` | `registry.text("memory")` |
| `cmd_chat`、`--no-brain` 的回复和记忆 | `make_llm(cfg.llm)` | `registry.text("reply")`、`registry.text("memory")` |
| `cmd_look` | `one_shot(eyes_command)` | `registry.vision("eyes")` |
| `perception label --assist` / `--objects`、`attrs-label`、`gesture-label` | `assist_command` + `one_shot_message`（要 usage） | `registry.vision("image_label")`（返回文字 + usage；`Reviewer` 的并发、缓存、续跑不变） |
| `addressee label` | `assist_command` | `registry.text("text_label")` |
| 管理面板测试按钮 `probes.test_llm` / `test_claude` | 分两种 | `probes.test_provider(provider, 未保存的值)` |

用到 Claude 的地方才检查 Claude 令牌和 `claude` 命令（`_brain_env` 改成 `registry` 按需检查）：全用 DeepSeek 时没有 Claude 令牌也能起团子（眼睛 / 装扮描述那几个用处停用并警告）。

## 3. 管理面板「模型」页（`#models`）

左栏「设置」上面加一页「模型」（`console/static/models.js`，后端 `console/models_view.py`，接口 `GET/POST /api/models`、`POST /api/models/test`）。

**供应商**（一家一张卡片）：
- id、接入方式；Claude Code：claude 路径、令牌（原设置页「Claude 令牌」挪过来）、配置目录；OpenAI 兼容：接口地址、Key（打码，同现在）、Key 的环境变量名。
- 模型列表：每行模型名 + 「能看图」勾选，可增删。
- 「测试」：用页面上还没保存的值，发一句「只回复 ok」；勾了能看图的再带一张 64×64 的小图问"什么颜色"。
- 「删除」：还有用处在用（主或备）就拒绝，列出是哪几处；在 config.toml 里定义的标「在 config.toml 里，面板不能删」。
- 「+ 添加供应商」：选接入方式；OpenAI 兼容给预填模板（DeepSeek / ChatGPT / 通义 / Ollama：地址和常见模型名，Key 自己填）。

**用处**（一行一个，顺序同 §1.2 的表）：说明、主下拉、备下拉（含「无」）、看图标记；下拉按「供应商 / 模型」列，看图的用处只列能看图的；每行标来源（默认 / config.toml / 面板）+「恢复默认」；有问题的行标红。

**保存**：整页一个按钮，写 console.toml 的 `[providers.*]` / `[models.*]`，Key 写 secrets.toml；团子 / 沙盒在跑时照样保存、提示下次启动生效。

**设置页**：删掉「大模型」组（`llm.*`、`secret.llm`）和 `secret.claude`、`brain.claude_path`、`brain.model`、`brain.eyes_model`、`brain.memory_model`，放一行「模型在『模型』页设置 →」；`proactive.auto_look_busy`、`appearance.describe` 说明里的 Haiku 改成「识图模型」。

**预检**：按用处查——每个用处的主模型那家有没有 Key / 令牌（claude-code 再查 `claude` 命令在不在）、看图用处选的模型能不能看图、供应商存不存在；问题带 `setting` 跳到「模型」页那一行。只有用到 Claude 才要 Claude 令牌。

页面守现有规矩：颜色只用 `var(--…)`、不用原生弹窗（`ask` / `toast`）、请求相对路径、`common.js` 的公共接口。

## 4. 记录和显示

- `brain.jsonl` 每轮多 `provider` / `model`；openai 那一路也记 usage（`prompt_tokens` / `completion_tokens` / `prompt_cache_hit_tokens`，有就记）。
- 大脑时间线（`brain/trace.py`）轮头带模型名；status 和管理面板运行卡片：「大脑：deepseek/deepseek-chat」，切过备用写「已切到 claude/sonnet（deepseek 余额用完）」。
- 启动日志每个用处一行：`模型：brain = deepseek/deepseek-chat（备 claude/sonnet）`。

## 5. 测试（`python -m pytest -q`，不调真模型）

- 配置：内置默认；`[providers]` 两个文件按 id 合并；旧字段换算（只写 `[llm]` 的本机写法、显式写了 `brain.model`、echo、anthropic）；看图用处选了不能看图的；引用不存在的供应商；`models/` 的 `id` 校验。
- `content.py`：同一份内容转 Claude 块 / OpenAI 消息。
- 两种 kind：文字 / 看图（假 OpenAI 客户端、假 `StreamProcess`）；错误分类（401 / 402 / 403 / 429 insufficient_quota 关闸，429 其他 / 超时不关）。
- `GatedCall`：主成功；主关闸类错误 → 切备、闸关；闸已关直接走备；都不行抛 `ModelUnavailable`。
- 大脑：主 openai → `ToolLoopBrain`；主 claude-code → `BrainSession`；主的闸关了切备（跨 kind）；代看工具把 question + 图交给 eyes、返回文字；看得了图的大脑照旧拿图；`openai_tools` 带 `question`。
- cli：`_run_brain` 全用 DeepSeek、没有 Claude 令牌时能起（眼睛停用并警告）；离线命令走对用处。
- 面板：`/api/models` 读写（Key 不回传、打码）、删除还在用的被拒、config.toml 里的不能删；预检按用处；静态页面的现有检查（颜色变量、原生弹窗）覆盖 `models.js`。
- 现有用到 `cfg.llm` / `brain.model` / `ClaudeGate` / `DeepSeekBrain` 的测试跟着改。

## 6. 真机 / 沙盒验证（上线后用户做）

1. 沙盒：默认配置（大脑 DeepSeek）重置记忆启动，冒充好友聊几句；让它 `look_person` 一个人（沙盒是 text_only，应回"沙盒里只给文字"）；快进触发一次反思，「内心」页有结果；`brain.jsonl` 的 provider 是 deepseek。
2. 管理面板「模型」页：两家都点「测试」通过；把大脑主改成 claude/sonnet 保存、重启沙盒，status 显示变了；改回来。
3. 真机 live：status「大脑：deepseek/deepseek-chat」；好友问"看我衣服好看吗"时大脑调 `look_person`、拿到「（眼睛代看）」文字再回；眼睛 / 装扮描述日志里还是 haiku；DeepSeek 后台余额扣得正常。
4. （可选）把 DeepSeek Key 填错一位启动：大脑和记忆 / 反思切到 claude/sonnet，status 写原因。

注意：大脑换了模型，`history.jsonl` 里 Claude 的旧回复会被 DeepSeek 模仿（这是想要的接续感）；要清照 CLAUDE.md「记忆」一节挪进 `archive/`。

## 7. 这一期不做

- Anthropic API 接入方式。
- 面板上调每个用处的 temperature / max_tokens（配置文件里写）。
- 费用统计页。
- 运行中热切换模型（改了要重启）。
- 按用处的「备用的备用」（只有一个备用）。

## 8. 文档

更新 `CLAUDE.md`：「统管大脑」（大脑默认 DeepSeek、代看、闸按供应商）、「管理面板」（模型页）、代码结构表（`models/` 包）、「环境」里令牌那句（只有用到 Claude 才要）；`config.example.toml` 加 `[providers]` / `[models]` 示例、旧字段标废弃。
