# Agent 表情动作能力 — 设计

日期：2026-09-27

## 目标

让聊天 Agent 在合适的时候做表情动作（鞠躬、害羞、欢呼……）：由大模型决定做不做、做哪个，
可以边说话边做，也可以只做动作不说话。

## 需求（已和用户确认）

1. **动作来源两种都要**：
   - 轮盘上现有的动作：直接按数字键 1~8，不改轮盘，几乎零延迟；
   - 白名单动作（`emotes.extra`）：不在轮盘上时换到指定格子（`emotes.swap_slots`）再做。
2. **模型决定**：每轮可以只说话 / 说话 + 动作 / 只做动作。提示词要求偶尔用，代码再加频率上限。
3. **牵手**：做动作会松开牵手（game-ops §4 实测）。靠用户手动关：配置 `emotes.enabled` + `run --no-emotes`。

## 方案选择

模型用**句首标签**表达动作：`[害羞]哪有啦`，只做动作就只输出 `[害羞]`。
一次模型调用，不加延迟，不动现有回复格式。
（放弃的方案：JSON 输出——要重写清洗逻辑、口语风格可能变僵；单独一次调用选动作——延迟翻倍。）

## 1. 动作从哪来、怎么做

- **轮盘内容以实测为准**：启动时开一次编辑界面读 8 个格子（`Wheel.refresh()`）。
  前提：轮盘上的动作和白名单动作在图标库里有命名好的图标（用户从 `tmp/emscan` 挑出来改名放进 `emotes/`，不用全命名）。
- **白名单动作只换进 `swap_slots`**，用户的常用布局不被打乱；启动时记下这些格子原来的动作，
  退出（含 Ctrl+C）时恢复并重新读一遍核对。
- **做动作的顺序**：输入框开着先按 BACK 关掉（否则数字键会打字）→ 做动作 → 发文字。
- **换动作的流程**：按 C 关聊天记录面板（面板开着时长按 Z 没反应）→ 编辑轮盘 → 按 C 重新打开面板，约 5~10 秒。
  期间的新消息留在面板里，重开后 `ChatReader` 和上一帧对比仍能读到。
- **限速**：两次动作间隔 ≥ `emotes.min_interval`（默认 20 s），两次换轮盘间隔 ≥ `emotes.swap_min_interval`（默认 120 s）。
- **dry-run**：只打印"将会做动作"，不按键、不开编辑界面（启动时读轮盘除外——它不改任何东西）。

**待真机确认**：聊天记录面板开着时按数字键能否做动作。若不能，`perform` 也要先关面板、做完再开。

## 2. 提示词、解析、历史

### 提示词

动作可用时，`build_system_prompt` 在规则后追加一节 `## 动作`：

- 列出本轮可用的动作名；
- 格式：句首 `[动作名]`，一轮最多一个；只做动作就只输出 `[动作名]`；
- 大多数时候不用；适合：见面 / 道别鞠躬、被夸害羞、开心欢呼、别人点名要你做某个动作；
- 只能用列表里的；别人要列表外的动作，用文字接话（"这个我还没学会"），不要编动作名。

### 本轮可用动作在调用模型前决定

`Responder` 接收回调 `available_emotes: Callable[[], list[str]]`（默认返回空列表），每轮调用：

| 状态 | 可用列表 |
|---|---|
| 动作关闭，或距上次动作不足 `min_interval` | 空 → 提示词里不出现"动作"一节 |
| 距上次换轮盘不足 `swap_min_interval` | 只有轮盘上现有的 |
| 其他 | 轮盘上现有的 + 白名单 |

这样模型选出的动作几乎都能执行，历史里记的就是真做了的。

### 解析

`clean_reply(raw, max_chars, emotes)` 改为返回 `Reply | None`，`Reply` 是 `dataclass(text: str | None, emote: str | None)`：

- 句首或句尾的 `[xx]` / `【xx】` 视为动作标签，最多取一个；
- 名字不在本轮列表里 → 去掉标签只留文字，记 warning；
- 去掉标签后文字为空且没有有效动作 → `None`（不回复）；
- `_CLAIMS_HUMAN` 照常检查文字部分；命中则**整轮丢弃**，动作也不做；
- `max_chars` 截断只作用于文字；`<skip>` 行为不变。

### 历史 / 记忆

assistant 内容存成 `[害羞]哪有啦` / `[害羞]`（内存历史和 `history.jsonl` 相同），让模型记得自己做过什么、少重复。
inbox / notes 整理原样处理，不改。

### Agent

- `Responder.reply` 返回 `Reply | None`；
- 有动作 → `EmotePlayer.perform`（dry-run 只打日志）；再有文字 → 加 `disclosure_prefix` 后照旧发送、记 `RateLimiter`、`SelfFilter`；
- 只做动作的一轮不占文字发送的限速名额，只记动作限速；
- `perform` 抛 `WheelError` → 记日志、跳过动作，文字照发；
- `Agent.sent` 记录不变（只记文字），另加 `Agent.emoted: list[str]` 便于测试。

## 3. 代码结构、配置、测试

### 新模块 `src/skydango/game/emotes.py` — `EmotePlayer`

依赖：`Device`、`Wheel`、`EmoteConfig`、面板状态回调 `panel_visible: Callable[[], bool]`、面板开关键码、`clock` / `sleep`。

| 方法 | 作用 |
|---|---|
| `start()` | 面板开着就按 C 关 → `wheel.refresh()` → 记下 `swap_slots` 原内容 → 按 C 重开面板 |
| `available(now) -> list[str]` | 第 2 节的规则 |
| `perform(name, now) -> int` | 输入框开着先 BACK；在轮盘上直接按键；否则关面板 → `wheel.ensure(name, candidates=swap_slots)` → 开面板 → 按键；更新两个限速时间 |
| `restore()` | `swap_slots` 里被换过的格子逐个换回原动作（关面板 → 编辑 → 开面板），再 `refresh()` 核对，不一致记 warning |

`wheel.py` 小改：`ensure(name, candidates=None)` / `_victim(candidates)` 可限定候选格子；
候选格子与 `locked_slots` 求差，为空时抛 `WheelError`。

### 配置 `[emotes]`（`EmoteConfig`）

| 项 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 图标库为空时自动视为关闭并提示 |
| `extra` | `[]` | 白名单：可以换上轮盘的动作名 |
| `swap_slots` | `[]` | 允许换的格子；为空 = 只用轮盘现有动作（最安全） |
| `min_interval` | `20.0` | 两次动作最小间隔（秒） |
| `swap_min_interval` | `120.0` | 两次换轮盘最小间隔（秒） |

面板开关键复用 `vision.log_open_key`。`extra` 里不在图标库的名字启动时 warning 并忽略。
`config.example.toml` 补 `[emotes]` 示例。

### CLI

- `run --no-emotes`：本次运行不做动作；
- `run` 在动作启用时构造 `EmotePlayer`、`start()`，在 `try/finally` 里 `restore()`（Ctrl+C 也恢复）；
  `start()` 失败 → 记 warning，本次运行不做动作，聊天照常。

### 测试（合成数据 + 假设备）

- `tests/test_responder.py`：标签解析（`[xx]` / `【xx】`、句首 / 句尾、只有动作、列表外的名字、声称真人时整轮丢弃）；
  可用列表为空 / 非空时提示词里有无"动作"一节；历史里存带标签的内容。
- `tests/test_agent.py`：先做动作再发文字；只做动作不占 `RateLimiter`；dry-run 不按键；`WheelError` 时文字照发。
- `tests/test_emotes.py`：按对数字键；输入框开着先 BACK；换动作前后按 C 关 / 开面板、只动 `swap_slots`；
  限速期内 `available` 的结果；`restore` 换回原动作。

### 文档

- `AGENTS.md`：代码结构表加 `game/emotes.py`，常用命令加 `run --no-emotes`；
- `docs/game-ops.md` §4：写入真机验证结果（面板开着时数字键是否生效等）。

### 真机验证（实现后）

1. 聊天记录面板开着时按数字键，看是否做动作；
2. `run`（dry-run）观察模型选动作的频率和时机；
3. 和好友 `run --live` 跑一段；退出后 `emotes wheel` 核对轮盘已恢复。

## 前置工作（用户）

从 `tmp/emscan` 挑出轮盘上的 6 个动作和想加进白名单的动作，复制到 `emotes/` 并改成动作名；
`config.toml` 里 `wheel.library_dir` 改回 `emotes`。

## 不做的事

- 不自动识别牵手状态；
- 不做运行中的热键开关；
- 不让模型一轮做多个动作，也不做动作序列。
