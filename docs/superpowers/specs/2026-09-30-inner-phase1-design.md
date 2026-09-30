# 内心层 第 1 期：关系卡 + 日子 — 设计

日期：2026-09-30　状态：**设计已和用户确认，待写实施计划**

## 背景：内心层

远期目标（用户）：团子像 Neuro-sama 那样有"活着"的感觉（游戏里的伙伴，不做主播）。身体反射（`2026-09-30-body-reflex-design.md`）管"外在"：马上有反应、会做小动作。
这份开始做"内在"。和用户对过，欠缺的是五样：

1. 没有内在状态：情绪、精力、兴致
2. 没有自己想要的东西（心愿、惦记的事）
3. 关系没有厚度（见过几次、上次什么时候、一起经历过什么）
4. 性格不够锋利（太乖、没有口头禅和老梗）
5. 没有"日子"的概念（第几次上线、隔了多久、今天做了什么）

用户选了**统一的内心层**（代替反射设计里列的子项目 2 关系账、3 情绪、4 性格语气，并把心愿和日子并进来）：

- **一处存放**：`memory/inner/`，关系卡、心情、心愿、日子、团子自己的梗都在这里
- **两种节奏**：身体按事件当场改（不调模型）；定期"反思"（一次性 `claude -p`，同 `NotesKeeper`）改慢变量
- **一个出口**：大脑醒来的消息 / `status` 里带上"心里"的东西（会变的不放系统提示词，免得缓存失效）
- **状态要有后果**：影响主动开口的额度、话长短、反射，不只是写进提示词

分三期，每期单独 spec → 计划 → 实现 → 真机验证：

| 期 | 内容 | 调模型 |
|---|---|---|
| **1（这份）** | 账本：关系卡（数字）+ 日子；给大脑看 | 不调 |
| 2 | 反思：心情（带原因的文字）、精力、心愿 / 惦记的事、日记；接上后果（主动开口额度、话长短、反射概率、心跳） | 反思调 |
| 3 | 性格：反思从聊天里沉淀口头禅、老梗、看法；放开提示词里太乖的规则 | 反思调 |

第 1 期只记账、只读现有数据，不往游戏里发任何新输入。

## 需求（已和用户确认）

1. 关系卡**只给好友**（friends.md 里的 `## 标题`）；陌生人不建卡
2. 账本是**机器文件**，用户不手改；`memory show` 能看。对某人的描述还是写在 friends.md / notes.md
3. 第一次启用时**从 history.jsonl 回填**能推出来的
4. 账**在身体主循环里直接记**（不订阅事件队列：事件会被合并、`leave` / `return` 会互相抵消，拿来记账会少算）
5. 只在 `--live` 时写盘；dry-run 只在内存里记（`status` 照样是真的），不写盘

大脑模式专用；`run --no-brain` 的普通 Agent 不接。

## §1 存什么（`memory/inner/`，不进 git）

### `people.json`：每个好友一张卡

键是 friends.md 的 `## 标题`。

| 字段 | 意思 | 怎么记 |
|---|---|---|
| `first_met` | 第一次见（或第一次聊天） | 第一次在场 / 说话时记；回填取 history 里最早的 |
| `last_seen` | 上次在身边 | 在场时每圈更新 |
| `visits` | 见过几次 | 离 `last_seen` 超过 `visit_gap`（1800 秒）再出现算新的一次。不用身体的 `brain.rejoin`（60 秒）：名字标签闪一下就会多算 |
| `days` | 在身边或说过话的日期列表（`"2026-09-30"`，本地日期） | 当天第一次在场 / 说话时追加 |
| `minutes` | 一共在身边待了多久（分钟，存浮点） | 在场时按这一圈和上一圈的间隔累加，单圈最多 `max_step`（5 秒），卡顿时不一下加一大截 |
| `today` / `today_minutes` / `today_visits` | 今天是哪天、今天待了多久、今天见了几次 | 日期变了清零 |
| `lines` | 他在聊天里说过几句 | 见 §2 名字匹配 |
| `to_me` | 其中跟团子说的几句 | 用反射的 `reflex.addressed()` 判断（叫了名字 / 团子刚说完 / 身边只有他一个好友） |
| `last_line` | 他最后说的一句 `{t, text}` | text 截到 40 字 |

文件顶上还有 `version`（1）和 `backfilled`（回填时间，没回填是 `null`）。时间都是 `time.time()`（墙上时间）。

### `days.jsonl`：每次上线一行

```json
{"start": 1759200000.0, "end": 1759207200.0, "live": true, "ended": "normal", "friends": ["小明", "阿花"],
 "heard": 42, "said": 17, "summary": "下线前大脑写的经过……"}
```

- `ended`：`normal`（正常退出）/ `crash`（意外结束，下次启动补的）/ `backfill`（回填推出来的）
- `friends`：这次在身边或说过话的好友（按第一次出现的顺序）；`heard`：好友和陌生人说的句数（"我"不算）；`said`：团子说的句数
- `summary`：`Brain.farewell()` 写的经过（现在只进 inbox.md，以后也存一份在这里）；没写出来是空字符串
- **只有 live 的上线才写进来**（dry-run 不写），所以"第几次上线"只数 live

### `current.json`：这一次上线（运行中）

和 `days.jsonl` 一行同样的字段，外加 `saved`（最后保存的时间）。live 时每 `save_every`（60 秒）连同 `people.json` 一起写；正常退出时追加进 `days.jsonl` 后删掉。
启动时发现它还在 → 上次是意外结束：`end = saved`、`ended = "crash"` 追加进 `days.jsonl`，再删掉。

### 写盘

- 只有身体线程写（`present` / `heard` / `said` 在身体线程调；`save` 在身体主循环里按时调；`close` 在 cli 的收尾里，身体线程已经停了）
- 读的地方：`status`、`arrive` 说明都在身体线程拼；「日子」在启动时拼；`memory show` 在另一个进程只读文件。`Ledger` 里仍放一把锁（网页 / 大脑线程以后要读），读的方法返回拷贝
- 原子写：先写 `*.tmp` 再 `replace`（同 notes.md）

## §2 怎么记（`inner/ledger.py`）

`Ledger(cfg: InnerConfig, directory, friends: Callable[[], list[str]], persist: bool, now: float)`。时间参数一律传墙上时间（身体用 `self.wall()`）。

**名字匹配**：新函数 `match_friend(name, friends) -> str | None`：和每个好友算 `similar()`（`chat/tracker.py`），取分数最高且 ≥ 0.75 的那个。
身边好友（`env.nearby()` 已经是 friends.md 的标题）和聊天说话人（OCR 出来的，可能错字）都过它，同一个人不会被拆成两张卡。空名字、`"我"` 返回 `None`。

**身体调的方法**

- `present(names, now) -> dict[str, str]`：每圈调（身体算完身边好友之后），返回"这一圈算作新见面的好友 → arrive 说明"。对每个在场的好友：
  - 没卡 → 建卡（`first_met = now`）
  - `now - last_seen > visit_gap`（或刚建卡）→ `visits += 1`、`today_visits += 1`，并按**更新前**的卡写一句 arrive 说明（§4 ②）
  - 和上一次 `present` 的间隔 `dt = min(now - prev, max_step)`（`dt < 0`，即墙上时间往回跳，按 0 算），加到 `minutes` / `today_minutes`
  - `last_seen = now`；当天日期不在 `days` 里就追加；这次上线的 `friends` 里没有就加进去
- `heard(speaker, text, to_me, now)`：身体读到新聊天时每句调。这次上线 `heard += 1`；说话人匹配到好友 → `lines += 1`、`to_me` 为真再 `to_me += 1`、`last_line` 更新、日期和 `friends` 同上
- `said(now)`：团子每说一句（`body.say` 真的说出去了，含 dry-run 的计数）→ 这次上线 `said += 1`
- `save(now)`：距上次保存 ≥ `save_every` 才写；`persist = False` 时什么都不写

**cli 调的方法**

- `close(summary, now)`：写 `people.json`、把这一次追加进 `days.jsonl`（`ended = "normal"`）、删掉 `current.json`。`persist = False` 时什么都不写

**跨日期**：`today_minutes` 在 `present` 里发现 `today` 不是今天就清零；`status_line` 读的时候也按今天判断（隔夜挂机不会显示昨天的时长）。

## §3 回填（`inner/backfill.py`）

`people.json` 不存在时启动做一次（dry-run 也做，只是不写盘，这次运行在内存里用）：

- 逐轮读 `history.jsonl`：每轮的 `user` 按行拆出"名字：内容"（`recall._heard` 用的同一种格式），说话人过 `match_friend`
  - 好友：`first_met` = 最早一轮的时间；`lines` 按句数加；`last_line` = 最后一句；`days` = 说过话的日期；`last_seen` = 最后一次说话的时间（近似）
  - `visits`、`minutes`、`to_me` 推不出来，从 0 开始
- 按相邻两轮间隔 > `session_gap`（7200 秒）分段，每段一行 `ended = "backfill"`、`start` / `end` 是段里第一 / 最后一轮的时间、`friends` / `heard` / `said` 按段里的数、`summary` 空，写进 `days.jsonl`（`days.jsonl` 已经有内容时不回填上线记录，只回填卡）
- `people.json` 里记 `backfilled` 时间；日志写"回填了 N 个好友、M 次上线"

## §4 给大脑看什么

**① 系统提示词：新一节「日子」**（`inner/days.py` 的 `days_prompt()`，`brain_prompt` 放在「上次聊到哪」之前；启动时算一次）

> ## 日子
> 今天 9月30日（周三）。这是你第 23 次上线，今天第 2 次；上次下线是 3 小时前（意外断了）。最近 7 天上线了 4 天。
> 上次见到了：小明、阿花。上次的经过：……
> 很久没见的好友：阿花（12 天前）、小红（20 天前）

- "第几次"数 `days.jsonl` 的行数 + 1（回填的也算）；"上次下线"是最后一行的 `end`，`ended = "crash"` 时注"意外断了"
- "很久没见"：`last_seen` 超过 `long_gap`（7 天）的好友，按多久没见从近到远，最多 3 个；没有就不写这一行
- 第一次上线（`days.jsonl` 空、也没回填出东西）：只写"今天 …。这是你第一次上线。"
- 结尾一句：“这些是给你心里有数的，别跟人报数字。”

**② `arrive` 事件**（`body.py` 放 `arrive` 的地方，`return` 不加）：`present` 返回的说明接在"来到身边"后面

> 小明 来到身边（第 13 次见；上次 3 天前；今天第一次）。他上次最后说的（3 天前）：「明天要考试」

- 第一次见（这张卡是这圈刚建的）："（第一次在身边见到）"
- 上次 > `long_gap`：加"，好久没见了"
- "今天第一次" / "今天第 N 次"：按 `today_visits`（加上这一次之后）
- "上次最后说的"：只在 `last_line` 是今天之前的时候附
- 说明由 `present` 在改 `last_seen` **之前**写好返回（不然"上次"永远是刚才）；身体放 `arrive` 时查 `notes.get(name, "")`。
  名字标签闪一下、身体按 `rejoin` 发的是 `return`，这时 `present` 也不会算新见面（没超过 `visit_gap`），两边一致；
  超过 `rejoin` 但没到 `visit_gap` 又出现时，身体发 `arrive` 但 `notes` 里没有这个人 → 不加说明（同一次见面里走开又回来）

**③ `status` 的"身边的好友"**：`ledger.status_line(names, now)` 代替原来的名字列表

> 身边的好友：小明（今天一起 40 分钟·认识 21 天·一起玩过 9 天）、阿花（今天刚来·认识 3 天）

- "今天一起 N 分钟"不到 1 分钟写"今天刚来"；"认识 N 天" = 今天 − `first_met` 的日期差，当天认识写"今天刚认识"；"一起玩过 N 天" = `len(days)`，等于 1 时不写
- 没开内心层 / 没这张卡：照旧只写名字

**④ 提示词规矩**（`brain/prompt.py`，内心层开着才加；放在「说话」一节里）

- 状态和事件里写了你和每个好友的交情（见过几次、上次什么时候、他上次说了什么）。按交情说话：熟的人随便点，刚认识的客气点；隔了很久见面可以表现出来（“好久不见”“你去哪了”）
- 别报数字（不说“我们见过 13 次”“你今天待了 40 分钟”）
- 他上次提过的事可以接着问（“考试怎么样了”），拿不准先 `recall`

**⑤ `memory show`**：原有输出后面加"关系卡"（每个好友一行，同 status_line 再加见过几次、上次什么时候、说过几句 / 跟你说过几句）和"最近 10 次上线"（开始时间、多长、见到谁、怎么结束的、经过的前 40 字）。

可视化网页、管理面板这一期不加（第 2 期有了心情一起加）。

## §5 接线（`cli.py`、`brain/body.py`）

- `cli` 大脑模式：`cfg.inner.enabled` 且有 `memory_dir` 时建 `Ledger`（`persist = not dry_run`）：先处理 `current.json`（意外结束）→ 没有 `people.json` 就回填 → 开始这一次。交给 `Body`（`body.ledger`，默认 `None`）和 `brain_prompt(..., days=days_prompt(...))`
- 身体：
  - 算身边好友那处（`body.py` 放 `arrive` 的函数）：`notes = self.ledger.present(near, wall)`；`arrive` 文字加 `notes.get(name, "")`
  - `_heard`：每句 `self.ledger.heard(m.speaker, m.text, to_me=self._addressed(m, now), wall)`（`_addressed` 反射已经有；反射关着也照样能调，它是纯判断）
  - `say`：真的说出去（或 dry-run 计数）后 `self.ledger.said(wall)`
  - 主循环每圈末尾 `self.ledger.save(wall)`
  - `status`："身边的好友"用 `self.ledger.status_line(near, wall)`
  - `ledger is None` 时这些全跳过；每处 `try/except` 只记日志（记账出错不能影响身体）
- 退出（`cli` 的 `finally`）：`farewell()` 改成返回写出的经过文字（没写出来是 `""`，现在返回 bool）→ `ledger.close(summary, wall)`。`farewell` 没跑（dry-run / 大脑断了）也要 `close`（dry-run 时 `close` 不写盘）

## §6 配置（新 `[inner]`，`config.py` 的 `InnerConfig`）

| 键 | 默认 | 作用 |
|---|---|---|
| `enabled` | `true` | 总开关；`false` = 完全照旧（不记账、提示词和 status 不变、不回填） |
| `visit_gap` | 1800 | 离上次在场超过这么多秒再出现，算新的一次见面 |
| `session_gap` | 7200 | 回填时相邻两轮隔这么多秒算两次上线 |
| `long_gap` | 7 | 超过这么多天没见算"好久没见" |
| `save_every` | 60 | live 时每隔多少秒写一次 `current.json` / `people.json` |
| `max_step` | 5 | 算在一起的时长时，单圈最多算几秒 |

`config.example.toml` 加这一节；管理面板设置清单（`console/settings.py`）加 `enabled`。**数字都是估的**。

## §7 出错怎么办

- 记账、写盘、拼文字出错：只记日志（`log.exception`），身体、大脑照常；文字拼不出就退回原来的样子（只写名字）
- `people.json` 读不了（坏 JSON / 版本不认识）：改名成 `people.json.bad-<时间>` 放一边，从空卡开始，**不回填**（免得错数据再算一遍），日志 WARNING
- `days.jsonl` 读不了的行跳过（同 `ChatMemory`）；`current.json` 读不了：改名成 `.bad-<时间>`，不补
- 墙上时间往回跳：`dt` 按 0 算；`last_seen` 不往回改
- friends.md 改了名字：旧名字的卡留着，不显示（不在好友名单里的卡 `status_line` / 「日子」都不用）

## §8 测试（`python -m pytest -q`，假时钟，不碰设备）

- `match_friend`：错一个字对到同一个人；两个好友都像时取最像的；空 / "我" / 不像的是 `None`
- `present`：30 分钟内再出现算同一次见面、超过算两次；名字闪一下（隔几秒）不多算；单圈时长上限；墙上时间往回跳；跨午夜 `today_minutes` 清零；返回的 arrive 说明（第一次见 / 今天第一次 / 今天第 N 次 / 好久没见 / 上次最后说的只附今天之前的）
- `heard`：好友说的记到卡上（`lines`、`to_me`、`last_line` 截 40 字）；陌生人、"我"、看不出是谁的不记卡，只算这次上线的 `heard`
- `status_line`：今天刚来 / 今天刚认识 / 一起玩过 1 天不写；没卡只写名字
- `days_prompt`：第一次上线；第 N 次、今天第 N 次、上次意外断了；很久没见最多 3 个、按近到远；没有很久没见的不写那行
- 回填：构造一份 history（几个好友、跨两天、中间隔 3 小时），检查 `first_met` / `last_line` / `lines` / `days` 和上线分段；`days.jsonl` 已有内容时只回填卡
- 写盘：`save_every` 之内不重复写；原子写；`close` 追加一行并删 `current.json`；启动时 `current.json` 在 → 补一行 `crash`；`people.json` 坏了 → 改名、空卡、不回填
- `persist = False`（dry-run）：内存里照记，磁盘上什么都没有
- 身体接线（现有的假设备）：`arrive` 事件带上交情；`return` 不带；`status` 的"身边的好友"换了；`say` 计数；`ledger = None` 时和以前一样
- `brain_prompt`：开着有「日子」一节和交情规矩，`enabled = false` 时和现在逐字一样
- `farewell` 返回经过文字（原来返回 bool 的调用方跟着改）

## 真机验证（单元测试过了不等于能用）

1. 第一次 `run --live`：看回填日志（"回填了 N 个好友、M 次上线"），`memory show` 的关系卡和上线记录对不对得上印象
2. 好友来了：`--view` 大脑时间线里 `arrive` 那条带不带交情、`status` 的"身边的好友"对不对；名字标签闪的时候 `visits` 有没有乱涨
3. 正常退出再启动：「日子」一节写"第 N 次、上次 X 前"、上次的经过
4. 强杀（`taskkill /F /T`）再启动：补上一行 `crash`，「日子」写"意外断了"
5. 跑一阵看大脑有没有报数字（"我们见过 13 次"），有就改提示词

之后把结论写进 CLAUDE.md（「记忆」一节加 `memory/inner/`）。
