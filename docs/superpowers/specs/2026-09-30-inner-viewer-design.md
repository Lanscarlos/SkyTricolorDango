# 内心层可视化（管理面板「内心」页）— 设计

日期：2026-09-30　状态：**代码已完成，待真机验证**（实施计划 `docs/superpowers/plans/2026-09-30-inner-viewer.md`；见文末「真机验证」三步）

## 背景

内心层三期（账本、反思、性格）都**还没在真机上跑过，数字都是估的**。调参时要能看到团子的内心：心情 / 精力怎么变、每次反思改了什么、日记、关系卡、性格档案，并且能删掉沉淀歪了的性格条目。
现在心情只存"现在"一份（`mind.json`），精力每圈现算，**没有历史**；所以要先加一份流水账，再做页面。

## 需求（已和用户确认）

1. **边跑边看 + 跑完复盘**：放在管理面板（`console`，团子不在跑也能开），加一个「内心」标签页
2. **能改的只有：删性格条目**（口头禅 / 老梗 / 看法）。别扭、心愿、心情、关系卡都只看不改
3. 做法 A：管理面板读 `memory/inner/` 的文件；团子在跑时经 `/live/inner` 取实时数据；删除在跑时交给团子的身体线程做，不在跑时直接改 `persona.json`；反思记进 `mind_log.jsonl`

## §1 流水账（`memory/inner/mind_log.jsonl`，`inner/log.py` 的 `MindLog`）

一行一条；live 时追加写盘，dry-run 只在内存里（最近 200 条）。

| kind | 什么时候 | 字段 |
|---|---|---|
| `reflect` | 每次反思（含下线那次） | `t`、`final`、`mood {level, text}`、`energy {level, score}`、`grudge {who, why, until} \| null`、`wants [text…]`、`changes [str…]`、`dropped [str…]` |
| `energy` | 每 `ENERGY_EVERY`（300 秒） | `t`、`level`、`score` |
| `forget` | 网页上删了性格条目 | `t`、`what`（`catchphrase` / `joke` / `opinion`；`kind` 固定是 `forget`）、`text`、`who` / `topic` |

**`changes`**：`diff(before, after) -> list[str]`（纯函数，对比反思前后的 `Mind` + `Persona` 快照，不靠模型）：
- `心情 平常→开心（有人来聊天）`（档位变了；只换了那句话写 `心情：…`）
- `新别扭：小明（放鸽子）`、`别扭撤了：小明`
- `新心愿：想看日落`、`心愿了结：…`
- `新口头禅：…`、`新老梗：小明——…`、`新看法：雨林——…`、`看法换了：雨林 丑→其实还行`、`淡出：…`（包括超上限被挤掉的）

**保留**：启动时 `trim(LOG_DAYS = 30)` 删掉 30 天前的行（live 时重写文件，原子写）。
`read(path, since) -> list[dict]`：管理面板读文件用（读不了的行跳过）。

## §2 接口

**团子**（`vision/viewer.py`，`run --view` 时 cli 接上）：
- `GET /inner` → `{"running": true, "at": 墙上时间, "mood", "energy", "grudge", "wants", "soft": [{who, text, until}], "persona": Persona.to_dict(), "log": 内存里最近的记录}`（`Body.inner_snapshot()`，经身体线程取，超时 3 秒）
- `POST /inner/forget`，body `{"kind": "catchphrase|joke|opinion", "text": …, "who": …, "topic": …}` → `Body.forget(...)`（身体线程里删、live 写 `persona.json`、记 `forget`）→ `{"ok": true}` 或 `{"ok": false, "error": "找不到这条（可能已经淡出了）"}`
- POST 规矩同手动控制：只本机（`is_local_host`）、`X-Skydango: 1`、JSON、≤ 64 KB（`post_guard`）

**管理面板**（`console/server.py` + 新 `console/inner_view.py`）：
- `GET /api/inner` → 读文件：关系卡（`card_line` 所需字段 + 是否在好友名单里）、最近 10 次上线（`recent_days`）、最近 10 篇日记（`last_diaries`）、`mind.json`、`persona.json`、最近 7 天的 `mind_log`；
  团子在跑（runner 状态 running）时再取 `/live/inner`（超时 3 秒），**心情 / 精力 / 别扭 / 心愿 / 收着点 / 性格以实时为准**，记录合并（按 `t` 去重）。返回里带 `source`：`"live"` / `"files"` / `"files_fallback"`（在跑但取不到实时）
- `POST /api/inner/forget`：在跑 → 转发 `/live/inner/forget`；没在跑 → `inner_view.forget_offline()` 直接改 `persona.json`（原子写）+ 记 `forget`；runner 是 `starting` / `stopping` → `{"ok": false, "error": "团子正在启动 / 停止，稍等再删"}`
- 都沿用现有的 Host 校验和 `post_guard`

## §3 页面（`console/static/console.html`）

导航在「实时画面」后面加「内心」。宽屏两栏、手机竖排；顶上一个来源标签：**实时** / **上次保存于 HH:MM** / **实时取不到，显示的是上次保存的**。

1. **现在**：心情（档位色点 + 那句话）、精力（档位 + 0~100 分数条）、别扭（对谁、为什么、还有多久）、收着点（对谁、还剩多久）、心愿（一排标签：惦记 / 想做 / 小心思）
2. **曲线**：精力分数折线 + 心情档位的背景色带（颜色同色点）；反思点画小圆点，点了下面的时间线跳到那一次；切换"最近 24 小时 / 7 天"；手写 SVG，不引外部库
3. **反思记录**：时间线（新的在上），每条：时间、心情变化、`changes` 逐行、`dropped` 灰色默认折叠；下线那次标"下线"；`forget` 也在里面；"只看有改动的"开关
4. **性格档案**：口头禅 / 看法 / 老梗（按人分组），每条"用过 N 次 · 上次 X 前" + "删"按钮（确认后 POST；启动 / 停止中按钮置灰、提示稍等）
5. **关系卡**：表格（名字、认识几天、一起玩过几天、见过几次、上次在身边、说过几句 / 跟你说过几句）；不在好友名单里的排最后、淡色
6. **日子和日记**：最近 10 次上线 + 最近 10 篇日记

刷新：在跑时每 5 秒取一次；没在跑时打开读一次 + "刷新"按钮。视觉沿用管理面板现有的颜色 / 字体 / 卡片，实现时用 `frontend-design` skill 做排版细节。

## §4 接线

- `Body`：`apply_reflection` 前后各拍 `Mind` / `Persona` 快照（`copy.deepcopy`）→ `MindLog.reflect(...)`；`_inner_tick` 每 `ENERGY_EVERY` 记一条 `energy`；`forget(kind, text, who, topic) -> str`（空字符串 = 删成功，否则是原因）；`inner_snapshot() -> dict`；都走 `_inner_call`
- `finish_reflection`：也记一条 `reflect`（`final=True`）
- cli：`reflector` 不为空时建 `MindLog`（`persist = not dry_run`，启动时 `trim`），交给 Body；`run --view` 时把 `body.inner_snapshot` / `body.forget`（经 `body.call`）挂到 viewer
- 没开反思（`reflect = false`）：`/inner` 里 `mood` / `energy` / `grudge` / `wants` 都是 `null`（`persona` 只看 `persona` 开关，开着照常给）、`log` 为空；页面这几块显示"（没开反思）"，关系卡 / 日子 / 日记照常（来自文件）

## §5 出错怎么办

- 文件不存在 / 读不了：那一块显示"（还没有）"，其余照常
- 在跑但 `/live/inner` 取不到：退回文件，`source = "files_fallback"`
- 删的条目不存在：返回原因，页面刷新
- `mind_log.jsonl` 写不了：只记日志，不影响团子

## §6 测试（`python -m pytest -q`，假设备、假子进程 `tests/fake_child.py` 的写法）

- `diff`：心情档位变 / 只换话、别扭新 / 撤、心愿增 / 了结、口头禅 / 老梗 / 看法新增、看法换立场、淡出 / 挤掉，各一行文字
- `MindLog`：内存上限 200、live 写盘、dry-run 不写、`trim` 删 30 天前、`read` 跳过坏行
- 身体：反思后多一条 `reflect`（`changes` 对）；每 300 秒一条 `energy`；`forget` 删掉 + 写盘（只 live）+ 记 `forget`；删不存在的返回原因
- viewer：`/inner` 结构；`/inner/forget` 缺头 / 非本机 / 非 JSON 拒绝
- 管理面板：没在跑只读文件、删除改文件；在跑合并实时、删除转发；实时取不到退回文件；启动 / 停止中拒绝删除
- 页面：导航有「内心」、六块区域的容器 id 都在（同现有页面测试查字符串）

## 真机验证

1. 边跑边开「内心」页：曲线、反思记录的 `changes` 对不对得上真发生的事
2. 网页上删一条口头禅：`persona.json` 里没了、团子之后不再用
3. 团子停掉后打开：复盘内容齐全，来源标签写"上次保存于…"

之后把结论写进 CLAUDE.md「管理面板」和「内心层」两节。
