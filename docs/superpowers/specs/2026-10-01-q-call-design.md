# 按 Q 喊一声：找好友 + 顺带认团子（设计）

2026-10-01。起因（用户告知，**还没截图核对**，见 game-ops「呼喊找好友」）：好友稍微离远一点，头顶就不显示名字标签——哪怕人还在屏幕里；
只有好友自己按 Q、或团子按 Q 时标签才亮出来，随时间淡化，约 5 秒。现在"身边有谁"全靠名字标签，于是：

- 框高 ≥ `stranger_min_height`（0.08 屏高）的好友标签一淡，1 秒后被判成**陌生人**，5 秒后（`[perception] keep`）被判成**走开**——
  轨迹上粘着的名字（`data["name"]`）不刷新 `last_seen`，只有读到标签（和认装扮的 `maybe`）才刷新
- 走开的误报一路传下去：`leave` / `return` 来回跳、眼睛自动看一眼（花额度）、冷场②"聊着聊着走了"、牵手状态被清、关系卡少算时长、超过 `rejoin` 回来又打一次招呼
- 远处小人（框高 < 0.08）不算好友也不算陌生人；三期 `far_crops` 在标签根本没画出来时也认不到；认装扮的好样本要框高 ≥ 0.10，远处小人学不到

另外按 Q 的呼唤光圈能用来认团子自己（game-ops「呼唤特效」，09-30 实测）：短按后头上冒亮圈，只有 0.25~0.45 s、最亮约 0.2 s；
**别人按 Q 的一模一样**，只能对时间——团子按下后 ~1 s 内谁头上冒圈。

## 0. 已定的事

| 问题 | 结论 |
|---|---|
| 谁来喊 | 大脑有 `call` 工具（社交上的事它定）＋ 身体在"刚走开的好友 + 画面里有没挂名字的人"时自动兜底喊一次（修识别上的洞） |
| 标签淡掉但轨迹没断 | 继续算在身边：挂过标签的轨迹只要不断就一直刷新"在身边"，断了才开始算 5 秒。平时就生效，和按 Q 无关 |
| 认团子做到哪 | 顺带做：每次为找人按 Q 时连拍 1 秒看光圈；恰好一个人冒圈且在画面中间附近才算；对不上就放弃，**不为确认再按** |
| 频率 | 保守：任意两次至少隔 20 秒；身体自动喊 10 分钟最多 3 次、同一个好友走开只为他喊一次；大脑只受 20 秒间隔限制（卡洛 `#` 命令窗口里不限） |
| 做法 | 身体同步按键 + 连拍（约 1 秒）→ 感知层开"呼喊窗口"被动收标签 → 大脑工具在 MCP 线程里等窗口结束取结果 |
| 长按 | 不用（大喊：蹲下跳起 + 罩住半个画面的大球，太显眼） |
| 好友自己按 Q | 不认他的光圈；他的标签亮出来照常挂到轨迹上、续命 |

没选的做法：做成技能层的技能（技能同时只能一个，track 在跑就不能喊；`tick` 不许 sleep 超过 0.3 s，身体循环只有 6~7 帧 / 秒，1 秒连拍塞不进去）；
交给感知层 `capture="own"` 的常驻线程认光圈（感知层得知道团子什么时候按键，按键和识别搅在一起，还要多维护一种模式）。

## 1. 感知层（`vision/perception.py`）

### 1.1 轨迹续命（`[perception] sticky_names = true`）

**挪到 `2026-10-01-tracking-relink-motion-design.md` 第 3 节实现**（配置名、语义不变；那份还管追踪器少断、断了在 `keep` 内按位置接回成"像小明"）。这份的计划里不再做这一条。

### 1.2 呼喊窗口

新接口（加锁，身体线程调）：

```python
def called(self, at: float, *, by_self: bool = True) -> None   # 开窗口 [at, at + window]
def call_result(self, at: float) -> CallSeen | None             # 窗口结束后才有；at 对不上 / 没结束返回 None
```

- 窗口里 `far_crops`：不退避（忽略 `FAR_RETRY` 的 5 秒退避），每帧最多块数 ×2——窗口里远处小人的标签真的画出来了
- 窗口里这一帧挂上名字的轨迹记进结果：`{名字: Seen(side, distance, on_screen=True)}`（方位 / 远近同 `people()` 的算法；同一个名字留最后一次）
- 贴边标签（1.3）记成 `Seen(side, None, on_screen=False)`
- 窗口结束那一帧再数一次：窗口里还剩几条没挂名字的点过火的人物轨迹（含框高 < 0.08 的远处小人）→ `unnamed`（"没挂名字" = 既没 `name` 也没 `maybe`，同追踪 spec §4.4）
- 暂停期间窗口跟着往后挪（`_resume` 里和 `last_seen` 一起 shift）
- 结果只留最近一次；`CallSeen` 是纯数据（`friends: dict[str, Seen]`、`unnamed: int`、`at`、`ended`）

`window` 来自 `[call] window`（6.0 秒：标签约 5 秒 + 余量）。

### 1.3 贴边标签（`[perception] edge_band = 0.06`）

好友在画面外时，游戏把他的名字标签贴在屏幕边上（track 里见过：x 卡在 1822~1830 / 92~94，约 0.95 / 0.048 屏宽）。

- 标签中心 x 在最左 / 最右 `edge_band` 屏宽里、`_assign_tags` 没给它挂上人物 → **画面外标签**
- 画面外标签**不刷新** `last_seen` / `labels`（不然一喊就冒出一串假的"来到身边"，5 秒后又一串"走开了"）；只在呼喊窗口里记进结果
- 这是对现有行为的改动（现在贴边标签照样算在身边）；`edge_band = 0` 关掉
- **要今晚真机核对**：贴边可能只在某些情况下出现（见 §6 第 1 步）

### 1.4 好友自己按 Q

不认他的光圈。他的标签亮出来 → 照常挂到轨迹上 → 1.1 续命。不需要窗口。

## 2. 身体喊一声（`Body.call_out`）+ 光圈（`vision/halo.py`）

### 2.1 `Body.call_out(reason: str, *, live: bool) -> CallResult`

身体线程里同步执行，约 1~1.3 秒。

**能不能喊**（不满足就不按，`CallResult.refused` 写原因）：

- `[call] enabled`、大脑模式；dry-run（`live` 为假）不按键，返回"dry-run：没喊"
- 没黑屏、没在环视（look_around / spin）
- 距上次喊 ≥ `min_gap`（20 秒）；卡洛 `#` 命令窗口里（`owner_window`）大脑的 `call` 不受限，工具结果标"（主人命令模式）"
- 身体自动喊（`reason == "auto"`）另有 2.3 的条件和额度

**顺序**：

1. `clear_view("call")`：关掉身体替大脑开的输入框，挡着的面板照现有规则；再查 `ime_shown()`，开着就拒绝（字母键会变成打字）
2. `with self.panel.borrow("call") as was_open:`（面板开着时按 Q 没反应）；`was_open` 时等 `track.settle` 秒，避开面板关掉那一下的画面横移
3. 拍一张基准图；从感知层取最近一帧的人物框（点过火的 `player` + YOLO 的 `self` 候选）
4. `device.hw_key(LINUX_KEY_Q)` 短按（键码 16，常量放 `device/base.py`，和 `LINUX_KEY_ENTER` 一起；沙盒桩照 `sandbox/world.py` 其他键）
5. 连拍 `burst`（1.0 秒），MuMu 原生截图约 9 ms 一张；不存整图，每张交给 `HaloWatch.add(frame, t)`，它只裁各人头顶区域算个数
6. `env.called(按键时刻)` 开窗口；归还面板（之后面板照 auto 模式的规则开关，不强求窗口里一直关着）
7. 记 `last_call = 按键时刻`；光圈结果按 2.2 处理

`CallResult`：`at`、`reason`、`refused`、`halo`（`"self"` / `"others"` / `"none"` / `"skipped"` / `"off"`）、`self_box`、之后由 `env.call_result(at)` 补上的 `seen: CallSeen | None`。

### 2.2 光圈怎么认（`vision/halo.py`，纯计算）

```python
def head_region(box: Rect, width: int, height: int) -> Rect
class HaloWatch:
    def __init__(self, base: np.ndarray, boxes: dict[int, Rect], pressed_at: float, cfg: CallConfig) -> None
    def add(self, frame: np.ndarray, t: float) -> None
    def result(self, center_x: float, width: int) -> tuple[str, int | None]   # (halo 状态, track_id)
```

- **区域**：以框顶为中心，宽约 2.5 倍框宽；纵向从框顶往上 0.5 倍框高到往下 0.3 倍框高（光圈约 1.3 → 3 倍头宽）；裁到画面内
- **数**：每帧该区域平均亮度（灰度或 HSV 的 V）减基准图同区域，再减整帧平均亮度的变化（抵消画面整体变亮变暗）
- **时间窗**：只取按键后 `0.1 ~ 0.8` 秒内的最大值；基准图里就亮着的（别人刚喊过、正在扩散）因为是减基准，自然不算
- **判定**：恰好一个人的最大值 ≥ `halo_rise`、其余都 < `halo_rise / 2`，且他的框中心离画面中线 ≤ `halo_center` × 屏宽 → `"self"`；
  两个以上过门槛 → `"others"`（别人也在喊）；没有 → `"none"`

**放弃 / 跳过**：`halo = false` → `"off"`；连拍中截图失败或黑屏、这 1 秒里镜头动过（`settle` 秒内 track / 注意力 nudge 过）→ `"skipped"`。**不为确认再按一次。**

**写回**：`"self"` 且（YOLO 最近一帧没有 `self`，或 `self` 候选不止一个）→ `env.self_box = 那个框`（和 `sweep()` 写的同一个字段，`_self_areas` / `_ref_height` / `Body._self_box` / `pick_self` 直接用上；转镜头时 `_forget_self` 照旧清掉）。
YOLO 已经稳稳认出团子时不覆盖。

**`halo` 默认关**：认错的后果重——感知层 `_filter` 会把 `self_box` 附近的人物框整个过滤掉，认错团子 = 把一个好友藏起来。先 `perception halo-eval` 标定 `halo_rise`、真机核对过再开。

### 2.3 身体自动兜底（`Body._watch_call(now)`）

每圈在 `_watch_people` 之后判断。**全部满足才喊**：

- `[call] auto`
- 有好友在 `auto_after_leave`（30 秒）内发过 `leave`，且这次走开还没为他自动喊过（`_left_at` 记的这次）
- 感知层这一帧有没挂名字的点过火的人物（陌生人，或框高 < 0.08、没归类的远处小人）；按位置接回的"像小明"（`maybe`）不算没挂名字——说明他可能只是走远了、标签淡了
- 自动喊 `auto_window`（600 秒）里少于 `auto_quota`（3）次；和大脑共用 `min_gap`
- **不喊**：输入框开着（不管谁开的）、有技能在跑、有互动请求、在举蜡烛点火（light）、聊天记录面板以外的面板开着、`reflex.min_gap` 内刚做过动作、大脑这一轮正在说话（`brain_busy()`）、黑屏、dry-run（dry-run 照样判断、记日志"会喊"，不按）

喊完不等：窗口结束后（`call_result` 有了）放**背景事件** `call`（加进 `events.py` 的 `BACKGROUND`，不单独叫醒大脑）：
"你下意识喊了一声：认出 小明（右边·远）；小红在画面外（左边）；还有 1 个没挂名字的人。"

认回来的好友：标签一亮 → 又在 `nearby` 里 → `_watch_comings` 发 `return`（`rejoin` 内）→ 事件队列里那条 `leave` 被抵消（现有 `CANCELS`）；
冷场②走 `returned()` 结束，宽限 15 秒内回来的不叫醒大脑。

出错只记日志，不发事件。

## 3. 大脑看到什么

### 3.1 工具 `call()`

- 不带参数；说明："短按 Q 喊一声：稍远的好友头顶会亮出名字约 5 秒，用来找人、看谁还在附近；每喊一次附近的好友都看得到，别常喊。"
- 执行：`body.call(lambda: b.call_out("brain", live=...))`（约 1 秒），然后 **MCP 线程**轮询 `body.call(lambda: env.call_result(at))` 等窗口结束（最多 `window + 4` 秒），身体照常读聊天
- 返回（例）："喊了一声：认出 小明（右边·远）、懒洋洋大王（左边·近）；小红在画面外（左边）；还有 1 个没挂名字的人。光圈：认出了你自己（中间）。"
  光圈是 `off` / `skipped` 时不写那一句；`others` 写"别人也在喊，没认出你自己"；没认出任何好友写"没看到谁的名字"
- 拒绝时返回原因（"20 秒前刚喊过""输入框开着"……）
- 登记：`tools.py` 的 `DESCRIPTIONS`（放在 `look_around` 后面）/ `_bind` / `ACTIONS`（算做了事）/ `SANDBOX_MISSING`（"沙盒里没有这个"）、`mcp_server.py` 装饰器；`[call] enabled = false` 或感知层没开（`[perception] enabled = false`，EnvWatcher 收不到窗口结果）时不注册、身体也不自动喊
- 提示词（`enabled` 时）在工具规则里加一句：刚还在聊的好友"走开了"但画面里还有人、有人问"你在哪 / 看得到我吗"、卡洛让你找人时，可以喊一声；别常喊

### 3.2 其他

- status 多一行（有过才写）："上次喊：2 分钟前（认出小明；你自己在中间）"
- 网页手动控制栏（`brain/manual.py` + viewer）加"喊一声"按钮：总是真执行、照样过 `call_out` 的检查，做完放 `manual` 事件
- 管理面板设置清单加 `call.enabled`、`call.auto`

## 4. 配置

`[call]`（新的 `CallConfig`，在 `Config` 里注册）：

| 项 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 关掉：没有 `call` 工具、不自动喊，提示词 / status / 工具列表逐字照旧 |
| `auto` | true | 身体自动兜底 |
| `min_gap` | 20.0 | 任意两次之间的秒数 |
| `auto_quota` / `auto_window` | 3 / 600.0 | 自动喊的额度 |
| `auto_after_leave` | 30.0 | 好友走开多少秒内还会为他自动喊 |
| `window` | 6.0 | 呼喊窗口秒数 |
| `burst` | 1.0 | 按键后连拍秒数 |
| `halo` | **false** | 光圈认团子 |
| `halo_rise` | 25.0（估的，halo-eval 标定） | 头顶区域平均灰度（0~255）比基准高出多少才算冒圈 |
| `halo_center` | 0.35 | 同 `peek.self_center`，只信画面中间附近 |

`[perception]` 加：`edge_band = 0.06`（`sticky_names` 见追踪 spec）。

## 5. 离线标定：`perception halo-eval <录像目录>`

在录像上逐帧跑 YOLO，对每条人物轨迹打印 / 画出头顶区域的亮度变化曲线（同 2.2 的算法，基准取前 0.3 秒的中位数），标出超过候选门槛的时间段；
报告写 `tmp/halo-eval/<时间>/report.md` + 曲线图。用 `tmp/record/q-call-20260930-c`（30 fps）核对：短按时那个人的峰值、旁边人的噪声、面板横移那几帧，给出建议的 `halo_rise`。

## 6. 测试

单元测试（`python -m pytest -q`，合成画面 + 假设备）：

- `halo.py`：合成"一个人头顶变亮"的序列能认出他；两人同时变亮 → `others`；基准里就亮着的不算；整帧一起变亮被抵消；时间窗外的峰不算；不在画面中间的不算 `self`
- 感知层（续命的测试在追踪 spec 里）：窗口里挂上名字的进 `call_result`；贴边标签不刷新 `nearby`、进结果的 `on_screen=False`；暂停时窗口跟着挪；`unnamed` 计数
- 身体：按键前先借面板、`ime_shown()` 为真时拒绝、按的是键码 16；各种"不能喊"；`min_gap`、主人命令窗口；自动喊的触发、额度、同一个好友一次；背景事件文字；dry-run 不按键；`halo` 写回 `self_box` 的条件
- 工具：`call` 的返回文字（各种光圈状态、拒绝）；`enabled = false` 时工具列表、提示词逐字不变；沙盒里回"没有这个"

**真机验证**（晚上做；数字都是估的）：

1. 先核对 game-ops「呼喊找好友」的四件待核对的事：标签多远消失（按距离还是屏幕大小）、按 Q 亮出来的标签 YOLO / OCR 读不读得到、贴边标签什么时候出现、09-27 那次"3 秒内没出现名字"的原因
2. `perception halo-eval tmp/record/q-call-20260930-c`，定 `halo_rise`
3. `run --view`：好友退开到标签消失，不再冒假的"走开 / 陌生人"
4. 手动控制栏"喊一声"、大脑 `call`：返回里认出了远处的好友；画面外的好友写在画面外
5. 好友走远：身体自动喊一次，`leave` 被抵消，大脑没去说"他走了"
6. 打开 `halo`：团子单独按 Q 时认出自己（`view` 里团子框对了），和好友同时按时放弃

## 7. 改哪些文件

| 文件 | 改动 |
|---|---|
| `src/skydango/config.py` | `CallConfig`；`PerceptionConfig.sticky_names` / `edge_band` |
| `src/skydango/device/base.py` | `LINUX_KEY_Q = 16` |
| `src/skydango/vision/halo.py`（新） | `head_region`、`HaloWatch` |
| `src/skydango/vision/perception.py` | `called` / `call_result`、贴边标签、窗口里的 `far_crops`（续命见追踪 spec） |
| `src/skydango/vision/env.py` | EnvWatcher 的空实现：`called` 什么都不做、`call_result` 返回 None（感知层没开时本来就不喊，只是让接口一致） |
| `src/skydango/brain/body.py` | `call_out`、`_watch_call`、status 一行、`CallResult` |
| `src/skydango/brain/events.py` | `call` 进 `BACKGROUND` |
| `src/skydango/brain/tools.py` / `mcp_server.py` | `call` 工具 |
| `src/skydango/brain/prompt.py` | `brain_prompt` 里工具规则一句（`enabled` 时，关掉逐字不变） |
| `src/skydango/brain/manual.py` + viewer 页面 | "喊一声"按钮 |
| `src/skydango/console/settings.py` | `call.enabled`、`call.auto` |
| `src/skydango/sandbox/world.py` | Q 键桩 |
| `src/skydango/cli.py` | `perception halo-eval` |
| `docs/game-ops.md`、`CLAUDE.md` | 真机核对后更新；CLAUDE.md 加「按 Q 喊一声」一节 |

## 8. 不做的

- 长按 Q（大喊）
- 为确认团子再按第二次 Q；单独为认团子按 Q
- 认好友按 Q 时他头上的光圈（用不着，标签亮了就行）
- 注意力（东张西望）里的"好友不见了"目标——自动兜底已经覆盖最常见的情况，以后再说
- 跟着标签去找画面外的好友（转镜头 / 走过去）：结果只告诉大脑"在画面外（左边）"，要不要 `camera` / `track` 它自己定
