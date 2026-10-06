# 好友在不在场：走出画面不算走开（设计）

2026-10-06。起因（用户）：现在好友走出画面约 5 秒就算"走开了"（`[perception] keep`），好友频繁进出画面时团子觉得他走了又回来，
在画面外待满 `[brain] rejoin`（60 秒）再回来就被当成新来的、又打一次招呼；冷场 ②"聊着聊着走了"、自动喊、找人也跟着误触发。
用户要的规则：**好友不在这个场景里满 15 秒才算离开**；只是走出画面（转个视角就能看到）、或者在远处没显示名字但按 Q 之后屏幕上能亮出他的名字标签，都不算离开。

游戏事实（用户 10-06 告知，**还没截图核对**，见 game-ops「呼喊找好友」）：好友在同一个场景、但在画面外时——
1. 离得近：名字标签自己贴在屏幕边上（已见过左右两边：中心 x 卡在 92~94 / 1822~1830）
2. 离得远：名字不显示，按 Q 后才贴到屏幕边上；贴在哪条边取决于团子视角和好友的相对位置，**上下左右四条边都可能**

## 0. 已定的事

| 问题 | 结论 |
|---|---|
| "在身边"拆成几层 | 三层：**身边**（画面里看得到）/ **附近**（贴边、或最近按 Q 喊到过）/ **找不到**（都没有，开始计时）；再往后才是**走开** |
| 什么时候算走开 | 找不到满 `leave_after`（15 秒）**且**喊过一声也没亮出他；一直喊不成最多等 `confirm_max`（60 秒）；不能喊（`[call]` 关、dry-run）满 15 秒就算 |
| 喊到一次管多久 | `recheck`（90 秒，可调，用户说 1~2 分钟都行）；过期了还看不到就重新算找不到、再喊。好友一直在远处玩 ≈ 每 90 秒喊一声 |
| 喊的额度 | 保留 `[call] min_gap`（20 秒）和 `auto_quota` / `auto_window`（1 分钟 3 声）作为兜底上限；`auto_after_leave` 和"走开"那种起因的 `auto_again` 不再管（被 `recheck` 取代）。一声同时确认所有找不到的好友 |
| 身边 ↔ 附近 | **不发事件、不叫醒大脑**，只在 status 里多一行"附近：…" |
| `nearby()` 的意思 | 改成"在场"（身边 + 附近）；另加 `in_view()` = 身边。用到 `nearby()` 的地方问的都是"他还在不在"，不用改 |
| 注意力转头看看 | 时机不变：好友离开画面满 `keep`（5 秒）时照旧往他出去的那边转着看看（以前这一刻就是"走开"） |
| 开关 | `[perception] presence`，默认开；`false` 逐字是原来的行为。感知层没开（`EnvWatcher`）不受影响 |

没选的做法：
- 只靠贴边标签、不自动喊（最安静，但远处的好友照样被判走开，没解决问题）
- 喊到一次就一直算在场、不再喊（喊得最少，但他真走了要很久才知道）

## 1. 在场判断（新模块 `vision/presence.py`，纯计算）

`Presence` 按好友名字记状态，不碰设备、不 sleep，时间由调用方给（感知层冻结过的 `now`）。

### 1.1 输入（感知层每帧调用）

| 方法 | 什么时候调 |
|---|---|
| `seen(name, now)` | 这一帧他在画面里：读到名字标签、或 `sticky_names` 续着的轨迹、按外观认的 `maybe` / 接回（和现在刷新 `last_seen` 的地方完全一样） |
| `edge(name, side, now)` | 这一帧他的名字标签贴在屏幕边上（§2.1），side = 左边 / 右边 / 上边 / 下边 |
| `called(at)` | 一次呼喊窗口开始（不管是谁按的：大脑 `call`、身体自动喊、找走开的好友那一步） |
| `call_done(at, found)` | 这次呼喊窗口结束，`found` = 窗口里亮出名字的好友（在画面里或贴边都算，`CallSeen.friends` 的键） |
| `shift(d, now)` | 暂停（`held`）结束：所有时间戳往后挪 `d`（同 `last_seen` 的做法，`perception.py` 恢复时那段） |

### 1.2 状态和规则

每个好友记：`last_view`（最后在画面里）、`last_edge`（最后贴边）、`confirmed`（最后一次喊到他）、`lost_at`（这次"找不到"的起点）、`tried`（`lost_at` 之后结束过一次没亮出他的呼喊）、`out_at`（最近一次离开画面的时间）。
只记这次运行里在场过的好友；走开后状态清掉，再出现从头来。

- **身边**：`now - last_view ≤ keep`
- **附近**：不在身边，且（`now - last_edge ≤ keep` 或 `now - confirmed ≤ recheck`）
- **找不到**：上面都不满足。起点 `lost_at` = 最后一条证据的时间——
  - 从身边 / 贴边掉下来的：`max(last_view, last_edge)`（所以"15 秒"从最后一次真的看到他算起）
  - 喊到过、`recheck` 过期掉下来的：`confirmed + recheck`（喊到的证据过期了，这时才算看不到）
- **走开**（`pop_left(now)` 返回、然后清掉状态），找不到且满足任意一条：
  1. `now - lost_at ≥ leave_after` 且 `tried`
  2. `now - lost_at ≥ confirm_max`（一直喊不成）
  3. 不能喊（构造时 `can_call = False`：`[call]` 关、`auto` 关、dry-run）且 `now - lost_at ≥ leave_after`
- 呼喊窗口结束：`found` 里有他 → `confirmed = at`（回到附近）；没有他、且窗口开始时间 ≥ `lost_at` → `tried = True`
- 任何时候重新 `seen` / `edge` / 喊到：回到身边 / 附近，`tried` 清掉

### 1.3 输出

| 方法 | 给谁 |
|---|---|
| `present(now)` | 在场的名字（身边 + 附近）→ 感知层 `nearby()` |
| `in_view(now)` | 身边的名字 → 感知层 `in_view()` |
| `around(now)` | 附近的人和怎么知道的：`(名字, "画面外·右边" / "远处", 喊到的时间或 None)` → status |
| `need_call(now)` | 找不到、`tried` 为假、还没走开的名字 → 身体决定要不要喊（§3.2） |
| `went_out(name)` | `out_at`：最近一次从身边掉到附近 / 找不到的时间 → 注意力、`find` |

## 2. 感知层（`vision/perception.py`）

### 2.1 贴边标签四条边

`_offscreen(tag, width)` 改成 `_offscreen(tag, width, height)`：标签中心在左右 `edge_band × 屏宽` 或上下 `edge_band × 屏高` 里、又没挂上人，返回 左边 / 右边 / 上边 / 下边。
其余照旧：贴边不刷新 `last_seen`、`labels` 照记、呼喊窗口里记 `Seen(side, None, on_screen=False)`；`presence` 开着时多调一次 `presence.edge(name, side, now)`。
呼喊结果的文字（`calling.py`）"在画面外（上边）"这类直接用新 side。盯人 `track` 只看 x，上下边的标签照样给出左右偏向，不改。

**风险**：YOLO 只在左右两边见过贴边标签，上下两边认不认得出没核对（真机验证第 5 步）；认不出要补标、重训。代码先四条边都处理。

### 2.2 `nearby()` / `in_view()`

- `presence` 开：`nearby(now)` = `presence.present(now)`，`in_view(now)` = `presence.in_view(now)`
- `presence` 关：`nearby()` 照旧（`last_seen` 在 `keep` 内），`in_view()` = `nearby()`
- `EnvWatcher` 加 `in_view = nearby`，沙盒世界同样（名单 = 身边）

`last_seen`、`keep`、`sticky_names`、接回、认装扮都不动；`seen(...)` 就挂在现在刷新 `last_seen[name] = now` 的几处旁边。
`_call_tick` 收尾那一帧调 `presence.call_done(at, set(c.friends))`；`called()` 里调 `presence.called(at)`；暂停恢复时调 `presence.shift(d, now)`。

## 3. 身体（`brain/body.py`）

### 3.1 人来人走

`_watch_comings` 不用改逻辑：`env.nearby()` 已经是"在场"，`leave` 只在真走开时发。只改 `leave` 的文字：
`presence` 开着时「小明 走开了（喊了一声也没看到，15 秒了）」/「小明 走开了（20 秒没看到，没喊成）」/ 不能喊时「小明 走开了（15 秒没看到）」——
感知层给 `left_note(name)` 拿原因，`EnvWatcher` 没有这个方法就用原来的文字。`return` / `arrive` 按 `rejoin` 分照旧。

### 3.2 自动喊一声（`_watch_call`）

- `presence` 开：起因 ①"好友刚走开、画面里有没挂名字的人"换成 **在场确认**：`who = env.need_call(now)`，**不要求**画面里有没挂名字的人；
  不看 `auto_after_leave`、`auto_again`、`_auto_called`。照旧查 `min_gap`、`auto_quota` / `auto_window`、`_auto_call_blocked`（输入框、技能、互动请求、举蜡烛、别的面板、刚做完动作、大脑在回聊天、黑屏）。
  日志「看不到 小明、阿花 了：喊一声看看还在不在」。这种喊**不发** `call` 背景事件（每 90 秒一次，太吵），结果只进 status 的"上次喊"
- 起因 ②"拿不准是不是小明"照旧（包括它自己的 `auto_again`）；两种起因同时有时一声算两样
- dry-run 不按键：`Presence` 构造时 `can_call = False`，满 `leave_after` 直接走开
- `presence` 关：逐字照旧

### 3.3 注意力找人、`find`

- 身体新记 `_out_at`：每圈比较 `env.in_view()` 前后，掉出去的记时间（`presence` 关时它和 `_left_at` 同一时刻）
- `_start_lost_search`：触发从 `_left_at` 换成 `_out_at`、"他不在 `in_view`"，时间窗口还是 30 秒（`auto_after_leave` 的值改成模块常量 `OUT_RECENT`，`presence` 关时照旧读配置）
- `_search_call`（找走开的好友那一步喊一声）：照旧受 `min_gap` / 额度限制；"刚喊回来过他"那条在 `presence` 开时改成"`recheck` 内喊到过他"
- `find`：`_left_at` + `auto_after_leave` 换成 `_out_at` + `OUT_RECENT`
- 冷场、关系卡、分清在跟谁说话、主动开口、大脑醒得勤不勤、两轮之间的 envdiff 都用 `nearby()`，自动变成按"在场"算，不改代码

### 3.4 status

"身边的好友：小明（…）"只列 `in_view()`（原来的写法和附注照旧），后面多一行（有人才写）：
`附近：阿花（画面外·右边）、老登（远处，1 分钟前喊到）`。管理面板 viewer `/status` 同样拆成"身边的好友" / "附近"两项。

## 4. 配置

`[perception]`（数字都是估的）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `presence` | `true` | 总开关；`false` 逐字照旧 |
| `leave_after` | 15.0 | 找不到满这么久、并且喊过一声也没有，才算走开 |
| `recheck` | 90.0 | 喊到一次管多久；过期还看不到就再喊（用户：1~2 分钟都行） |
| `confirm_max` | 60.0 | 一直喊不成最多等多久 |
| `keep` | 5.0（不变） | 现在只定"身边"这一层 |
| `edge_band` | 0.06（不变） | 四条边都看：左右按屏宽、上下按屏高 |

`[call] auto_after_leave` 在 `presence` 开时不用（启动日志 INFO 一行），`auto_again` 只管"拿不准"那种起因。
管理面板设置页加 `perception.presence`、`perception.leave_after`、`perception.recheck`。

## 5. 测试（`python -m pytest -q`）

- `tests/test_presence.py`（纯计算）：贴边 → 附近；看不到 → `need_call`；喊到 → 附近、管 `recheck` 秒；喊了没有 + 满 15 秒 → 走开；
  没喊过满 15 秒不走开；喊不成满 `confirm_max` → 走开；`can_call = False` 满 15 秒 → 走开；`recheck` 过期 → 重新 `need_call`、起点是过期那一刻；
  窗口开始早于 `lost_at` 的呼喊不算 `tried`；`shift` 后计时顺延；走开后再 `seen` 从头来
- 感知层：上下两边的贴边标签认成"上边 / 下边"、不刷新 `last_seen`；`nearby()` 含贴边 / 喊到的好友、`in_view()` 不含；`presence = false` 时 `nearby()` 照旧
- 身体：好友离开画面不发 `leave`；几个好友同时找不到只喊一声；喊不到满 15 秒发 `leave`、文字带原因；在场确认不发 `call` 事件；
  status 两行；注意力找人的触发时刻 = 离开画面满 5 秒；dry-run 不按键、满 15 秒走开
- `presence = false`：现有测试全部照旧通过

## 6. 真机验证（要好友配合；不往游戏里发新的输入，只是按 Q 的时机变了）

1. 好友走到镜头外、离得近（标签贴边），站 1 分钟：不发 `leave`，status"附近：小明（画面外·右边）"
2. 好友在同一个场景里走远到看不见：团子自动喊一声，status"远处，刚喊到"；之后约每 90 秒喊一声，好友那边觉得频率能不能接受
3. 好友离开场景或下线：约 15~20 秒发 `leave`
4. 走开后 60 秒内回来是 `return`，超过 60 秒是 `arrive`（同现在）
5. 录一段好友在团子上方 / 下方画面外的录像（`record`），核对 YOLO 认不认得出上下两边的贴边标签；认不出就补标、重训

## 7. 文档

CLAUDE.md「按 Q 喊一声」「YOLO 感知层」两节、game-ops「呼喊找好友」（加上用户告知的远近两种贴边情况）跟着改。
