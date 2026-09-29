# 面板识别 — 设计

日期：2026-09-29　状态：**已实现**（`vision/panels.py`、`game/panels.py`、`assets/panels/`、`brain/body.py` / `tools.py` / `mcp_server.py` / `prompt.py` / `manual.py`、`vision/viewer.py`、`cli.py`；计划 `docs/superpowers/plans/2026-09-29-panels.md`）；五张卡未核对，真机核对见 game-ops §7「面板」

让团子实时知道画面上开着哪些面板（聊天记录面板、动作面板、轮盘编辑、好友树、各种弹框……）：
身体操作前知道画面被挡了，不在面板上乱点；大脑能读出面板上的文字和按钮，自己决定怎么处理。

## 目标

1. **实时知道开着什么**：身体每一圈都知道现在有哪些面板开着，面板开 / 关时告诉大脑。
2. **操作前不被挡**：转镜头、做动作、点人、接互动、说话之前先确认画面干净；认识的面板顺手关掉，不认识的交给大脑。
3. **读懂面板、自己处理**：大脑能读出面板的标题、正文、按钮，决定按哪个；危险按钮有硬护栏。

成功的样子：大脑在跑，用户手动按 E 打开动作面板 → 大脑收到"开了：动作面板"；大脑想转镜头，身体先按 E 关掉（已核对的卡）再转；
误点共享空间圆圈弹出邀请框 → 大脑收到"出现面板：共享空间邀请……按钮：取消、加入"，`panel_press("取消")` 关掉，"加入"它按不了。

## 需求（已和用户确认）

1. **范围选 C**：感知 + 认识的面板身体自己收拾 + 不认识的弹框由大脑决定点哪个按钮。
2. **按钮安全边界选"白名单打底、主人临时放行"**：撤退类按钮直接按；其他按钮要卡洛在聊天里 `#允许 <按钮>` 放行；
   另有一份**放行了也不按**的硬底线（花钱、删好友、举报、退出之类，用户同意）。
3. **第一期能叫出名字的面板**：聊天记录面板、动作面板（E）、轮盘编辑界面、好友树、共享空间邀请弹框；其余由"通用兜底"认成"不认识的面板"。
4. **样本以后再录**：用户能录，但不是现在（开发在云端，没有模拟器）。所以第一期的五张特征卡都是**未核对**（`verified = false`），
   代码、测试全部用合成画面完成；录样本、核对卡片是上线后的第二步，只改数据不改代码。
5. **只接大脑模式**。并且（附带改动）**`run` 默认进大脑模式**，旧的普通 Agent 降级为调试用，`--no-brain` 才进。

## 方案选择

**选定：方案 1 —— 每个面板一份"特征卡"（模板 + 区域亮度 + 配置），加通用 OCR 兜底。**

- 特征卡判断每帧几毫秒，身体每一圈都能看，适合做实时护栏；每个面板录 3~5 张截图就能做（和 `assets/social/` 一个路子）。
- 认不出来的面板照样能 OCR 出文字、按钮交给大脑，满足"不认识的弹框由大脑决定"。
- 缺点：游戏改 UI 要重录模板。

放弃的方案：
- **加进 YOLO 感知层**（`panel_*` / `button` / `close` 类）：每种面板要标几十上百帧、重训，感知层本身默认关、没在真机跑过；
  留作面板多了以后的升级路线。
- **交给眼睛（Haiku）看图**：几秒一次、耗额度，做不了"操作前实时看一眼"；让模型给坐标点按钮不够准。

## 结构

| 位置 | 做什么 |
|---|---|
| `src/skydango/vision/panels.py` | 只看不动：加载特征卡、快看（每帧判断开着哪些面板）、细读（OCR 出标题 / 正文 / 按钮）、通用兜底 |
| `assets/panels/<名字>/card.toml` + 模板图 | 特征卡数据，进 git |
| `src/skydango/game/panels.py` | 动手：`PanelOps.close(面板)`、`PanelOps.press(按钮)`，每步截图确认 |
| `src/skydango/brain/body.py` | 每圈快看、面板事件、`status`、遮挡护栏 `clear_view`、`expect` |
| `src/skydango/brain/tools.py` / `mcp_server.py` / `prompt.py` | 三个新工具 `panel_read` / `panel_press` / `panel_close`，提示词补规则 |
| `src/skydango/brain/manual.py` + `vision/viewer.py` | 可视化上画面板框、按钮；手动控制加"读面板""关面板" |
| `src/skydango/config.py` | `[panels]` |
| `src/skydango/cli.py` | `panels` 子命令；`run` 默认大脑、`--no-brain` |

### 数据类型（`vision/panels.py`）

```python
@dataclass(frozen=True)
class Panel:
    name: str          # 卡片目录名，如 "emote_panel"；不认识的面板是 "unknown"
    label: str         # 给人 / 大脑看的名字，如 "动作面板"；不认识的是 "不认识的面板"
    box: Rect          # 面板占的区域（整张图坐标）
    verified: bool     # 卡片核对过没有；unknown 恒为 False
    layer: int         # 叠放顺序，越大越靠上（"最上面那个面板"按它取）

@dataclass(frozen=True)
class Button:
    text: str
    box: Rect
    kind: str          # "retreat"（撤退类，可直接按）/ "allow"（卡片登记过，可直接按）/ "other"（要放行）/ "never"（永远不按）

@dataclass(frozen=True)
class PanelReading:    # 细读结果
    panel: Panel
    title: str
    text: str          # 正文（除标题、按钮外的文字，按行拼起来）
    buttons: list[Button]
    t: float

@dataclass(frozen=True)
class PanelState:      # 快看结果
    panels: list[Panel]            # 按 layer 从上到下
    def top(self) -> Panel | None  # 最上面的一个（不含聊天记录面板）
    def blocking(self, action: str) -> list[Panel]  # 挡着这个操作的面板
```

## 特征卡

`assets/panels/<名字>/card.toml`，同目录放模板小图。坐标全是 0~1 归一化（按 1920×1080 标定），和 `config.py` 一致。

```toml
label = "动作面板"
verified = false             # 没核对：只报告"开着（未核对）"，身体不会自动去关
layer = 20
region = [0.69, 0.0, 0.94, 1.0]   # 面板占的地方：遮挡判断、细读 OCR、网页画框
confirm_frames = 2           # 连续几帧命中才算开 / 连续几帧不命中才算关（防闪烁）
allows = []                  # 这个面板开着时仍然可以做的操作（见"遮挡护栏"），默认都挡

[[features]]                 # 全部命中才算开着
kind = "template"
image = "pencil.png"
roi = [0.69, 0.30, 0.94, 0.45]
threshold = 0.8

[[features]]
kind = "dark"
roi = [0.70, 0.05, 0.93, 0.95]
max_value = 70               # 区域 V 通道中位数上限
max_std = 40                 # 区域 V 通道标准差上限（半透明底比 3D 场景均匀）

[close]
ways = ["key:18"]            # 依次试：key:<Linux 键码> / tap:<x>,<y> / esc / button:<撤退类按钮文字>
auto = true                  # 已核对时，身体操作前可以自己关

[buttons]
roi = [0.3, 0.6, 0.7, 0.8]   # 细读时在这块找按钮；不写 = 整个 region
allow = []                   # 这个面板上可以直接按的非撤退类按钮
never = []                   # 这个面板上永远不按的按钮（放行也不按）
title_roi = []               # 标题在哪；不写 = region 里最上面一行
```

- **加载**：`load_cards(dir)` 逐个目录读，缺字段、模板图不存在、kind 写错 → 抛带卡片名和字段名的错误，启动时就报出来。
  `_` 开头的目录忽略（和 `emotes/` 一致）。
- **特征种类**（第一期四种）：

| kind | 判断 | 速度 |
|---|---|---|
| `template` | 剪影模板匹配，复用 `vision/icons.py` 的多尺度匹配；`roi` 里最高分 ≥ `threshold` | 快 |
| `dark` | `roi` 的 V 通道中位数 ≤ `max_value` 且标准差 ≤ `max_std` | 快 |
| `builtin` | 调代码里注册的函数 `name = "chat_input"`；第一期只有它（调 `ChatReader.panel_visible`） | 快 |
| `text` | 细读的 OCR 结果里含 `any = ["共享空间", ...]` 中任一个 | 慢，只在细读时判断 |

  含 `text` 特征的卡：快看阶段只看其余特征（或通用兜底命中），先当"不认识的面板"；细读后 `text` 命中才改认成这张卡。

### 第一批五张卡（全部 `verified = false`）

| 目录 | label | 怎么认（初稿，录样本后核对） | 关法 | 备注 |
|---|---|---|---|---|
| `chat_log` | 聊天记录面板 | `builtin: chat_input` | `key:46`（C），`auto = false` | **不算遮挡**：所有操作都 `allows`；镜头类操作照旧由 `Camera._ready()` 自己关 / 开 |
| `emote_panel` | 动作面板 | 铅笔图标 `template` + 右侧 `dark` | `key:18`（E） | 位置按 game-ops §4（x 1330~1800） |
| `wheel_editor` | 轮盘编辑界面 | 右上 × `template`（1888, 32）+ 轮盘中心 ✓ `template`（643, 540） | `tap:0.983,0.03` | 长按 Z 的普通轮盘只在按住时出现，不做卡 |
| `friend_tree` | 好友树 | 右侧大块 `dark`（`friend_check.panel_left` 往右） | `esc` | 面板样子本来就未核对（game-ops §5） |
| `shared_invite` | 共享空间邀请 | 通用兜底 + `text: any = ["共享空间"]` | `button:取消` | `never = ["加入"]` |

模板图初稿：五张卡的模板图第一期**没有真实截图**。代码按"卡片里引用的模板图不存在 → 这个特征判为不命中，启动时打一行警告"处理
（这是 `verified = false` 卡的特例；`verified = true` 的卡缺图直接报错），这样没模板也能先上线，只是这些卡暂时认不出、落到通用兜底。
录样本后用 `panels cut` 裁出模板、调阈值、改 `verified = true`。

## 快看和细读

### 快看（身体线程，每圈）

`PanelWatcher.observe(frame, now) -> PanelState`：
1. 每张卡算快特征（`template` / `dark` / `builtin`），全部命中记一次"命中"；按 `confirm_frames` 做开 / 关的去抖。
2. 通用兜底的触发检查（见下），满足就把一帧交给后台细读线程（已有任务在跑就跳过）。
3. 后台细读的结果（不认识的面板 / 含 `text` 特征的卡）在下一次 `observe` 时并进 `PanelState`。

性能目标：五张卡 + 触发检查在 1920×1080 上 < 10 ms（模板匹配只在各自 `roi` 里做）。测试里用合成画面测时间上限，防止以后加卡变慢。

### 细读

`PanelWatcher.read(frame, panel) -> PanelReading`：对 `panel.box` 做一次 OCR（单独的 OCR 实例，线程数 `panels.ocr_threads`，和 env 一样不跟读聊天抢），然后：
- **按钮**：配了 `buttons.roi` 时，这块里 ≤ `button_max_chars`（默认 6）个字的文字块都算按钮；
  没配时在整个 `box` 里找，≤ 6 个字**并且**含按钮词（`button_words` / `retreat` / `never` / 卡片 `allow` / `never`）或者是 "×" 的才算
  （不然"共享空间"这种短标题会被当成按钮）；
  分类顺序：全局 `never` 或卡片 `never` 命中 → `never`；`retreat` 词表命中（或 × 模板）→ `retreat`；卡片 `allow` 命中 → `allow`；其余 `other`。
  词表匹配用"包含"（"取消" 匹配 "取消邀请"），`never` 优先级最高（"退出并删除" 是 never，不是 retreat）。
- **标题**：`title_roi` 里的文字；没配就取最上面一行。
- **正文**：其余文字按行拼。
- 缓存：同一面板、画面缩略图差异 < 阈值时复用上次结果（`panel_read` 连着调不重复 OCR）。

### 通用兜底（不认识的面板）

**什么时候去看**（任一即可，都满足冷却 `panels.unknown_cooldown` 默认 2 s）：
- 画面相对上一张"干净画面"（快看没发现任何非聊天面板时的缩略图）变化 ≥ `panels.change`（复用 `images.thumb` / `difference`）；
- 屏幕右上角区域出现 × 模板（`assets/panels/_common/close.png`，录样本前不存在 → 这条不触发）；
- 兜底每 `panels.scan_interval`（默认 5 s）看一次。

**怎么判断**：后台 OCR 屏幕中部 `panels.unknown_roi`（默认 `[0.15, 0.08, 0.85, 0.92]`，避开左边聊天面板和底部输入栏），
满足 **至少一个按钮词**（`retreat` / `never` / `panels.button_words` 词表：确定、取消、关闭、加入、同意、好的、知道了……）
**且** 正文 ≥ `panels.unknown_min_chars`（默认 6）个字 → "不认识的面板"，`box` 取这些文字的外接框再外扩 `panels.unknown_pad`。
只有名字标签、零星字（"2级"、"●"）的不算。

**怎么消失**：认成不认识的面板后，快看阶段对它的 `box` 缩略图做差异：变化大 → 再细读一次确认，读不到按钮词就算关了；
另外 `panels.unknown_ttl`（默认 30 s）没再确认过也算关（防止卡住一直挡着）。

**已知误判**：3D 场景里的石碑、告示牌文字可能凑出"按钮词 + 正文"。最坏的结果是大脑多读一次、`panel_close` 找不到能按的，
不会乱点（不认识的面板没有 `auto` 关法）。实测后再调门槛。

## 身体

### 面板事件和状态

- 每圈 `_sense` 里 `self.panels_state = self.panel_watcher.observe(frame, now)`。
- 开 / 关（去抖之后）放 `panel` 事件：`开了：动作面板（未核对）` / `关了：动作面板`。聊天记录面板的开关**不走这里**（沿用现在的 `_watch_panel`，"关了半分钟"那套），免得重复。
- 不认识的面板**细读完才发**：`出现不认识的面板：「<标题>」<正文前 40 字>，按钮：取消、加入`；含 `text` 特征的卡同理（"开了：共享空间邀请……"）。
- `status` 多一段 `开着的面板：动作面板（未核对）、不认识的面板「……」`；没有就不写。
- 面板开着（非聊天面板）期间调 `env.held("panel")`，关了释放 —— 画面被挡时感知层不误报"走开了"。

### 遮挡护栏 `clear_view(action)`

在这些操作的开头调用（真执行前；dry-run 也检查，返回的提示里说明"真执行时会先关掉 X"）：

| action | 调用处 |
|---|---|
| `camera` | `camera_move`、`camera_reset`、`capture_around`、`sweep_around` |
| `emote` | `emote` |
| `check_friend` | `check_friend_at` |
| `social` | 自动接互动（身体内部） |
| `say` | `say` |

逻辑：
1. `blocking(action)` = 开着的面板里，不在 `expect` 集合里、卡片 `allows` 不含这个 action 的。
2. 没有 → 通过。
3. 每个挡着的面板：`verified` 且 `close.auto` → `PanelOps.close`，关上了放事件 `顺手关掉了动作面板`；
   否则 / 关不上 → **不执行**：
   - 工具调用 → `ToolError("被「动作面板（未核对）」挡着：可以 panel_read 看看，或者 panel_close 关掉")`；关不上的额外放 `error` 事件。
   - 身体内部的自动行为（接互动）→ 这一次跳过，DEBUG 日志，下次再看。

### `expect`

身体自己要打开面板的操作包在 `with self.panel_watcher.expect("wheel_editor"):` 里（换轮盘 → `wheel_editor`；`check_friend` → `friend_tree`）：
期间这个面板不算遮挡、开 / 关不发事件；退出时如果它还开着，按正常规则发"开了"事件（说明没关上）。和现有的 `env.held` 并存。

## 动手（`game/panels.py`）

`PanelOps(device, watcher, cfg, sleep)`：
- `close(panel) -> bool`：按卡片 `close.ways` 依次试；每试一种等 `panels.ui_delay`（默认 0.8 s）截图快看，面板不在了就返回 True。
  点屏幕前复用 `social.touch_mode` 看摇杆圈，没有就先点一下唤醒（game-ops §1）。
  不认识的面板：撤退类按钮 → × 模板 → 都没有返回 False。**不按 BACK**（游戏里 BACK 会做什么未验证）。
- `press(button) -> PanelState`：点按钮框中心（同样先确认触屏模式），等 `ui_delay`，截图快看 + 细读，返回按完的状态；
  画面几乎没变时返回里注明"按了但画面没变"。
- 点屏幕会关掉聊天记录面板 → 不专门处理，交给现有的 `PanelKeeper.maybe_reopen`。

## 大脑的工具

| 工具 | 参数 | 做什么 |
|---|---|---|
| `panel_read` | `image: bool = false` | 返回每个开着的面板（不含聊天记录面板）：名字、是否核对、标题、正文、编号的按钮（`[1] 取消（可以按）`、`[2] 加入（不能按）`、`[3] 确定（要主人放行）`）；`image=true` 附面板区域原图（`look_at_max` 限制）。没有面板时说"没有开着的面板" |
| `panel_press` | `button: str`（编号或文字） | 按按钮。要求：15 秒内 `panel_read` 过、那个面板还开着（快看仍在）、过安全规则；返回按完的状态 |
| `panel_close` | — | 关最上面的面板（`PanelState.top()`），顺序：卡片关法 → 撤退类按钮 → × → 返回"找不到关的办法" |

- `panel_press`、`panel_close` 进 `ACTIONS`（算"做了事"）；dry-run 不真点（返回"dry-run：会点「取消」"）；手动控制 `live=True` 真执行。
- 照样占每轮 6 次工具的名额。

### 按钮安全规则

| 按钮 kind | 能不能按 |
|---|---|
| `retreat`（关闭 / 取消 / 返回 / 拒绝 / 稍后 / 知道了 / 以后再说 / ×） | 直接按 |
| `allow`（卡片登记） | 直接按 |
| `other` | 要主人放行 |
| `never`（全局 `panels.never` + 卡片 `never`） | **放行了也不按** |

**主人放行**：卡洛（`brain.owner_name`）在聊天里发 `#允许 <按钮文字>` → 身体记一条放行（文字、到期时间 `now + panels.permit_window`）。
`panel_press` 按 `other` 按钮时找文字相近（包含或相似度 ≥ 0.8）且没过期的放行，**用一次就作废**；
放行不绑定具体面板（一次一个按钮、60 秒内有效已经够窄）。`#允许` 照样走现有的 `owner_command` 事件，大脑能看到。

**提示词**补：面板挡着时先 `panel_read`；能按撤退类就关掉；想按要放行的按钮、卡洛在身边时可以在聊天里问他，不在就别按、关掉面板；`never` 的按钮别想办法绕。

## 配置 `[panels]`

```toml
[panels]
enabled = true              # 大脑模式默认开；false = 退回现在的行为（不看面板、不加工具、不做护栏）
cards_dir = "assets/panels"
ocr_threads = 2
scan_interval = 5.0         # 通用兜底：没有别的触发时，隔多久看一次屏幕中部
unknown_cooldown = 2.0
unknown_roi = [0.15, 0.08, 0.85, 0.92]
unknown_min_chars = 6
unknown_pad = 0.02
unknown_ttl = 30.0
change = 0.25               # 画面变化触发阈值（和 brain.scene_change 同一种度量，实测后调）
button_max_chars = 6
button_words = ["确定", "取消", "关闭", "加入", "同意", "好的", "知道了", "返回", "拒绝", "稍后", "以后再说", "确认"]
retreat = ["关闭", "取消", "返回", "拒绝", "稍后", "知道了", "以后再说"]
never = ["购买", "充值", "支付", "兑换", "删除", "屏蔽", "举报", "退出", "注销"]
permit_window = 60.0
read_ttl = 15.0             # panel_press 要求 panel_read 在这么久以内
ui_delay = 0.8
```

## 命令行

都不往游戏里发输入（`panels` 默认读当前截图，只截图不点）：

```bash
python -m skydango panels scan [图片或目录]     # 逐张快看：认出了哪些面板、每个特征命中没有 / 分数；标注图 tmp/panels/
python -m skydango panels read [图片]           # 细读一次：标题、正文、按钮和 kind；含通用兜底判定
python -m skydango panels cut <图片> <卡片> <文件名> --roi x1,y1,x2,y2   # 裁一块存成该卡片的模板图
```

录样本用现有的 `record`。docs/game-ops.md 新增的"面板"一节给出**录样本清单**：每个面板录"打开前 / 打开动画中 / 开着 / 关后"，
在 2~3 张不同地图各录一次（`dark` 特征要看背景亮暗）。

## 可视化（`run --view`）

- 画面上画面板框：已核对蓝色虚线、未核对黄色虚线、不认识的红色虚线，框上标 label；细读过的按钮画小框（`retreat` / `allow` 绿、`other` 灰、`never` 红）。
- 右栏加"开着的面板"。
- 手动控制栏加"读面板""关面板"两个按钮（走 `panel_read` / `panel_close` 的身体方法，`live=True`，做成放 `manual` 事件）。

## 附带改动：默认进大脑模式

- `run` 默认 = 现在的 `run --brain`；新增 `--no-brain` 进旧的普通 Agent；`--brain` 保留为无操作的兼容参数（帮助里写"已是默认"）。
- 帮助文字、CLAUDE.md 的常用命令和"统管大脑"一节：普通模式标注为**调试用**。
- 大脑前提（`SKYDANGO_CLAUDE_TOKEN`、`mcp`）缺失时：报清楚的错并提示 `--no-brain`，不静默退回普通模式。

## 测试（`python -m pytest -q`，合成画面 + 假设备 + 假 OCR）

- 特征卡：解析正常卡；缺字段 / kind 写错 / 已核对卡缺模板图 → 报错带卡片名；未核对卡缺模板图 → 特征不命中 + 警告。
- 快看：`template` / `dark` / `builtin` 各自命中与不命中；`confirm_frames` 去抖（闪一帧不算开 / 关）；五张卡 + 触发检查的耗时上限。
- 细读：按钮分类（`never` 优先于 `retreat`、包含匹配、卡片 `allow` / `never`）；标题 / 正文拆分；缓存命中不重复 OCR。
- 通用兜底：三种触发；"按钮词 + 正文"判定为面板、只有名字标签不算；消失（画面变 / ttl）。
- 身体：面板开 / 关事件各一次、不认识的细读后才发、聊天面板不重复发；`status` 文字；`env.held("panel")` 成对；
  `clear_view`：已核对 auto 关成功 / 关不上 / 未核对拒绝 / `allows` 放行 / `expect` 期间不挡 / 自动接互动跳过。
- 工具：`panel_read` 编号和 kind 文案；`panel_press` 的安全规则（直接按、要放行、放行过期、放行用一次作废、never 放行也拒、`panel_read` 过期要重读、面板已关）；
  `panel_close` 的尝试顺序；dry-run 不点；手动控制 `live=True` 真点。
- 附带改动：`run` 不带参数进大脑、`--no-brain` 进普通模式、`--brain` 兼容。

## 文档

- `docs/game-ops.md` 新增一节"面板"：已知面板表（位置、关法、核对状态）、录样本清单、核对步骤。
- `CLAUDE.md`：代码结构加 `vision/panels.py`、`game/panels.py`、`assets/panels/`；常用命令加 `panels`；新增"面板识别（`[panels]`）"一节；
  "统管大脑"里工具列表加三个面板工具；普通模式标注调试用。

## 上线顺序

1. **代码合进 main**：五张卡全是未核对 → 身体只报告、不自动关；遮挡护栏遇到未核对的面板拒绝、交给大脑；大脑能读文字、按撤退类按钮。
2. **用户录样本**（`record`）→ 用 `panels scan` 逐张核对、`panels cut` 裁模板、调阈值 → 改 `verified = true`；每核对一张，身体就多收拾一种面板。
   核对结论写回 game-ops"面板"一节。

## 不做（第一期）

- 普通 Agent 模式接面板（降级为调试用）。
- YOLO 认面板、Haiku 看图认面板。
- 面板里的滚动、翻页、输入文字；多步流程（比如自动走完一个设置向导）。
- 按 BACK 关面板（行为未验证）。
- 长按 Z 的普通轮盘（只在按住时出现，身体自己控制）。
