# 认交互图标：YOLO 统一框图标 → 认种类 → 判归属（设计，路线图 ②b）

2026-10-05。路线图（`2026-10-04-future-roadmap-design.md`）② 最小版拆成三份：②a 面板核对、**②b 认交互图标（本文）**、②c 火焰改用 YOLO 图标框。
②c 要等本文的模型；②a 和本文互不依赖。

## 0. 目标和已经定下的事

**目标**：YOLO 只负责把画面里所有交互图标框出来（一个类别）；框里是哪种图标由模板 / DINOv2 最近邻认；属于谁按几何关系判。
结果写进 status、眼睛的位置说明和管理面板画面。**只认不点**：团子的任何代码路径都不会去点地图图标。

**和用户定下的**（brainstorming，按问答顺序）：
1. 认出的地图图标**只写 status 和眼睛、画在管理面板上，不发事件、不叫醒大脑**（和现在「画面里的东西：座位」一样）
2. YOLO 类别：**`social_ring`（2 号）扩成通用图标**，编号不变、含义扩大；旧的 262 个框沿用。代价：现在「没名字的圈 = 陌生人」的假设要改（见 §3.3）
3. 标框**复用难例收件箱的整帧页**（`worktree-hardcase-inbox` 分支，spec `2026-10-04-hardcase-inbox-design.md`）：本文依赖它先合进 main
4. 第一批地图图标 10 种：坐下、蜡烛 / 灯点火、留影、音乐、共享空间、篝火点燃、先祖、乐器 / 秋千、门 / 机关、留言 / 告示
5. 认种类：**模板和 DINOv2 最近邻两种都接**，录完样本用同一批裁图离线比，按结果选（`[icons] classifier`）；分类头不训，只攒裁图

**现状**（10-05 查的）：
- YOLO 10 类（`config.py` `PerceptionConfig.classes`）；`datasets/sky` 816 帧（train 677 / val 139），`social_ring` 262 个框**全是 100×100 定尺框**：
  都是弱标注按名字标签下方（`weaklabel.weak_labels`）/ 人头顶（`ring_labels`）用模板生成的，**一个地图物件都没标过**（物品模式碰到的白圈对话图标、♪ 圆圈都删了）
- 圈的种类：`game/social.py` 的 `IconClassifier`（米白剪影 `cream()` + 多尺度模板，`assets/social/` 10 张），感知层 `_classify` 以圈心裁一块交给它
- 没名字的圈（`perception.py` 的 orphans）：认出 `candle` 且有白圈（`white_ring`）→ `Request("陌生人", "candle")`，**身体会去点**；没白圈 = 深色火焰圆盘、不算请求
- 火焰（点亮陌生人）不走 YOLO：`vision/candle.py` 的 `find_flames` 在团子周围按颜色 + 火焰模板找，`vision/lighting.py` 的 `FlameWatch` 判点亮 / 走开

**不做**：点地图图标（坐下归路线图 ④；其他以后要点时单独定）；火焰换成 YOLO（②c，`candle.py` / `lighting.py` / `_watch_flames` 这一期一行不动）；面板核对和面板上的图标按钮（②a）；
训图标分类头（只攒裁图）；烛火堆这类收集品（不刷资源，路线图「已定」）；共享空间「加入」永远不点。

## 1. 第一批图标

| 归属 | 图标 | 现在 |
|---|---|---|
| 人头顶 | ✦、牵手、拥抱、击掌、背背、举蜡烛的白圈火焰、眼睛（在看留影 / 听音乐）、共享空间、没点火的陌生人 | 有模板，路径不变 |
| 黑影头顶 | 深色火焰圆盘 | 走 `candle.py`；本文让 YOLO 也框它（为 ②c 攒数据），运行时照旧不算请求 |
| 地图物件 | 坐下、蜡烛 / 灯点火、留影、音乐、共享空间、篝火点燃、先祖、乐器 / 秋千、门 / 机关、留言 / 告示 | 没有 |

种类的名字（代码里的 kind）**看完真机再定**：同一个图标出现在好几处只算一种，意思按归属分（例如火焰图标下面是篝火 = 「篝火点燃」，什么都没有 = 「可以点的蜡烛 / 灯」，见 §3.2）。
名字和中文叫法放进 `game/social.py` 的 `KIND_NAMES`，新增 `MAP_KINDS`（地图图标的种类集合）。

## 2. 数据

### 2.1 真机先看清楚（晚上）

每种图标写进 `docs/game-ops.md` 新的一节「地图交互图标」：长什么样（截图存 `tmp/icons/`）、多远会冒出来、大小随不随距离变、用户手动点了会怎样、在哪几张地图见过。
另外看清：**手里举的蜡烛火苗和头顶的火焰圆盘是不是一回事、会不会同时出现**（决定 ②c 的标注规则，本文先记下来）。

同时 `python -m skydango record` 在常去的几张地图录像，每种图标尽量覆盖 3 张以上地图、远近、白天晚上；录像名带地图和图标（如 `icons-yunye-sit-1005`）。

### 2.2 模板和参考裁图

- **模板**：每种地图图标截一张，放进 `assets/social/<kind>.png`（同一种多张按 `-2`、`-3` 编号，`kind_of` 已支持），和人头顶的共用一个库
- **参考裁图**（DINOv2 最近邻用）：每种 3~5 张，`assets/icons/<kind>/*.jpg`，裁法同运行时（§3.2）；从不同地图、远近里挑
- 截法：`perception icon-cut <截图> <kind> --box x,y,w,h [--template]`（裁图同时写参考图；`--template` 再按 `cream()` 剪影存模板），或者从 `icon-eval` 的总览图里按编号挑

### 2.3 进收件箱

收件箱现在只收 `runs/*/hard/`，加两样：
- **`perception inbox add <录像目录> [--every N]`**：录像按收件箱的去重规则（`dup_diff` / `dup_gap`）抽帧，`--every` 先隔 N 张取一张（默认 1），
  当成一次「运行」放进 `datasets/inbox/<录像目录名>/`（`raw/` + `frames.json`，没有 `hard.jsonl`、`reason = "import"`），分边照收件箱（目录名 crc32）
- **`perception inbox add datasets/sky --redo`**：老帧回炉。每帧带上现有标注当预标注；**分边沿用原来的 train / val**（不按哈希重分，验证集不变）；
  通过时**允许覆盖** `datasets/sky` 里的同名帧（只有 `--redo` 进来的帧可以），覆盖前把原标注备份到 `datasets/sky/_backup/labels-<时间>/`（一次回炉一个备份目录）
- **难例**：运行时有不认识的地图图标的帧存一张进 `hard/`（原因 `icon_unknown`，受难例自己的上限），下线照常收进收件箱

### 2.4 预标

收件箱整理（`perception inbox process`）的预标注多一步：`find_flames` 扫**整张图**（不限团子周围，聊天面板区域照旧丢），分数 ≥ `disk_sure` 的火焰补成 `social_ring` 框，
框半径 = 火焰半高 × `[icons] flame_ring`（拿 2.1 的样本标定；和已有圆圈框 IoU ≥ 0.5 的不重复加）。YOLO 现有的 `social_ring` 照常预标。
地图图标第一批没有模型能预标，靠人在整帧页画；训出第一版后用新模型预标后面的录像。

### 2.5 整帧页改框

漏的补上、地图图标手画（类别 6「圆圈」）；老的 100×100 框顺手拉紧到贴着圆圈（不强求一次改完，回炉的帧改到哪算哪）。
**标注规则**（写进 `docs/progress/2026-09-28-yolo-training.md`）：所有可点的圆圈 / 图标都标 `social_ring`，框贴着圆圈外沿；举在手里的火苗按 2.1 看清后的结论标或不标。

### 2.6 重训

用收件箱的一键重训（新 YOLO 版本号按收件箱的规则取）；看报告里 `social_ring` 和其他类的数字（验收见 §6），由用户点「换上」。

## 3. 运行时

### 3.1 每帧流程

YOLO 照旧只出 `social_ring` 框，圆圈照旧跟踪（`PANEL_DROP` 照旧丢面板里的、`far_crops` 照旧补远处的）。每个圆圈轨迹：

1. **判归属**（§3.3）
2. 归属是好友 / 陌生人 / 黑影：**走现在的路径，一行不改**（`_classify` + 现有模板、`requests`、白圈 / 圆盘）
3. 归属是先祖 / 物件 / 地图：**认种类**（§3.2），投票定下来以后进 `icons()`（§3.4）
4. 分数低 / 不认识的：存裁图（§3.5）

### 3.2 认种类

- 裁图：以 YOLO 框为中心、补成正方形、边长 = 框长边 × 1.3，缩放到统一大小（模板用 112，DINOv2 用它的输入尺寸）
- `[icons] classifier = "template"`（默认）：`IconClassifier` 用整个库（人头顶 + 地图的模板），因为框给了尺寸，尺度只在 0.9~1.1 里找；门槛同现在（0.75，或 ≥ 0.65 且领先第二名 0.15）
- `classifier = "dino"`：冻住的 DINOv2-small（同第二层 / 认装扮那个文件，有共享会话就共用）提特征，和 `assets/icons/<kind>/` 每张参考图比余弦相似度，取最像的一张；
  ≥ `dino_match`（估 0.75）且领先第二种 `dino_margin`（估 0.05）才算认出。参考图启动时提一次特征缓存在内存里。DINOv2 加载失败 → WARNING、退回模板
- 都不够 → `unknown`（「不认识的图标」）
- **投票**：每条圆圈轨迹记最近 `vote`（5）次的结果，最多的那种（`unknown` 不参与，全是 `unknown` 才是 `unknown`）
- **意思按归属分**：同一个 kind 在不同归属下的叫法查一张表（`social.py` 的 `map_label(kind, owner)`），例如火焰 + 篝火 = 「篝火点燃」、火焰 + 地图 = 「可以点的蜡烛 / 灯」、三颗星 + 先祖 = 「先祖」

### 3.3 判归属（新函数 `owner_of(ring, tags, people, objects)`，纯计算）

按顺序，先命中的算：
1. 正上方有名字标签（现有 `_tag_above`）→ `friend`
2. 正下方有点过火的人 / 黑影的框（不算团子：`data["dango"]` / 看着像团子）→ `person`
3. 正下方有先祖框（YOLO `spirit` 或外形头认出的先祖）→ `spirit`
4. 正下方有篝火 / 座位 / 乐器框 → 那个物件（`bonfire` / `bench` / `instrument`）
5. 都没有 → `map`

「正下方」：圈心 x 落在框左右各放宽 `[icons] under_x`（估 0.25 倍框宽）以内，圈心在框上沿以上 0 ~ `under_up`（估 1.0 倍框高）、或在框里上半；
多个候选取最近的。数字拿 2.1 的样本标定（写一个 `icon-eval` 的归属总览，见 §5）。

### 3.4 防误点（关键）

现在「没名字的圈 + 火焰 + 白圈」就生成陌生人请求、身体去点。扩类后地图上的灯、篝火点燃图标也会被框出来、也没名字。
改成：**只有归属是 `person` 的圈才可能生成陌生人请求**；`spirit` / 物件 / `map` 的圈永远不进 `requests`、不进 `circles`。
陌生人离得远、人物框没出来时圈判成 `map`、错过一次点火请求——可以接受（比误点安全）。单元测试钉住。

### 3.5 报给谁：`PerceptionWatcher.icons(now)`

- 和 `objects()` 平行，返回 `Icon(track_id, kind, owner, label, box, side)`：连续 `min_hits`（3）帧看到、投票有结果的圆圈轨迹；`unknown` 也返回（label「不认识的图标」）
- `side`：左边 / 前面 / 右边，同 `objects()` 的规矩；**远近先不报**（2.1 看完图标大小随不随距离变再定）
- 排序同 `objects()`；暂停中（`held`）或最近一帧超过 1 秒返回空
- **攒裁图**：分数低于门槛 + 0.1 的、`unknown` 的，每条轨迹 2 秒最多一张，存 `runs/<…>/icons/<kind 或 unknown>/<时间>.jpg` + `icons.jsonl`（kind、两种分类器的结果和分数、归属、框），每次 run 最多 `save_max`（200）张

### 3.6 开关

`[icons]`（加进 `config.example.toml` 和管理面板设置清单）：

| 项 | 默认 | 说明 |
|---|---|---|
| `enabled` | false（验收过再改成 true，见 §6；本机验证时在 `console.toml` 打开） | false = 不判 3~5 的归属、不报地图图标、不攒裁图；人头顶的圈照旧 |
| `classifier` | `"template"` | `"template"` / `"dino"` |
| `dino_match` / `dino_margin` | 0.75 / 0.05 | 估的，`icon-eval` 后定 |
| `vote` / `min_hits` | 5 / 3 | |
| `under_x` / `under_up` | 0.25 / 1.0 | 估的 |
| `flame_ring` | 实现时量 | 只给预标用：从 `datasets/sky` 里认出 `candle` 的圆圈框上跑 `find_flames`，取「圆圈半径 / 火焰半高」的中位数写成默认值 |
| `save` / `save_max` | true / 200 | |

`enabled = false` 时行为逐字照旧，**除了 §3.4**：防误点的规则不受开关管（新 YOLO 换上后地图上的圈本来就会被框出来，关了开关也不能让它们变成请求）。
`[perception] enabled = false` 时整个不生效（旧的整图 OCR 只认好友名字下的圈，本来就碰不到地图图标）。

## 4. 给大脑和画面

- **status**：「画面里的东西」后面一行「画面里的图标：坐下（右边）、留影（左边）、不认识的图标 1 个」（`brain/body.py`，同物件那一行；空的不写）。好友 / 陌生人头顶的圈不进这一行
- **眼睛**：`brain/images.py` 的 `scene_note` 加「- 坐下图标：(x, y) 附近」
- **提示词**：讲物件的那句后面加「图标只说明那里能互动，你现在不会去点它，别答应去坐下 / 去点」（`[icons] enabled` 时才加）
- **不发事件、不叫醒**
- **管理面板画面**：overlay 加 kind `icon`（label = 叫法），`console/static/stage.js` 的 `COLORS` / `NAMES` 加一项（单独配色，`unknown` 用虚线）；人头顶的圈照旧 `ring` / `request`
- 沙盒没有画面，不涉及

## 5. 离线工具

- **`perception icon-eval <标注帧目录 | 录像目录> [--model 模型]`** → `tmp/icon-eval/<时间>/`：
  - 跑 YOLO → 判归属 → 两种分类器都跑（不看 `classifier`）
  - 每个「判成的种类」一张总览图（`template-<kind>.jpg`、`dino-<kind>.jpg`，裁图带编号），人眼数错了几个
  - 两种分类器意见不同的裁图单独拼一张；归属总览（每个圈画出判给了谁）一张
  - `report.md`：每种图标两种分类器各判出多少、`unknown` 占多少、分数分布，建议的 `dino_match`；给标注帧目录时再按 YOLO 标注算 `social_ring` 召回
- **`perception icon-cut`**：见 2.2

## 6. 验收

都达到才在 `config.py` 把 `[icons] enabled` 默认改成 true：
1. **框**：重训后，`social_ring` 在收件箱分出来的验证帧（没参与训练的运行）上召回 ≥ 80%；其他类的 mAP50 不比当时在用的模型（v10）低 3 个点以上
2. **种类**：每种地图图标在验证帧上认对 ≥ 80%（看 `icon-eval` 总览图数）；**认成别的种类**的接近 0（比认成「不认识」更糟）；按结果定 `classifier`
3. **不碰点火**：`tests/test_light_replay.py` 14 次回放结果不变
4. **真机一晚**：地图物件 / 先祖的圈一次都没被点过（`agent.log` 里查 `accept` / 「去点」）；status 报的图标和管理面板画面对得上

80% 是估的门槛，看第一版结果可以调（调了写进本节）。

## 7. 测试（`python -m pytest -q`，合成画面 + 假检测器 / 假特征模型）

- `owner_of`：五种归属、优先级（有名字标签又有人在下面 → friend）、团子不算、多个候选取最近
- **地图 / 先祖 / 物件的圈不进 `requests` 和 `circles`**（含「火焰 + 白圈 + 下面没人」）；`enabled = false` 时也一样
- 两种分类器：合成图标各自认对、分数不够 → `unknown`、DINOv2 加载失败退回模板
- 投票、`min_hits`、`icons()` 的排序和暂停时为空
- `map_label` 按归属换叫法
- 攒裁图：每条轨迹 2 秒一张、`save_max` 上限、`icons.jsonl` 字段
- status 行、`scene_note`、提示词那句只在 `enabled` 时有
- `inbox add`：录像导入（抽帧、去重、分边）、`--redo` 沿用原分边、只有回炉帧能覆盖、覆盖前备份
- 预标：整图 `find_flames` → 圆圈框、和已有框不重复
- `stage.js` 的 `icon` 配色（同现有的页面测试）

## 8. 真机验证（晚上）

1. 每种图标走过去看清楚、写 game-ops 的「地图交互图标」；同时录像
2. 截模板和参考图（`icon-cut`）
3. 录像 `inbox add` → 整理 → 整帧页改框（含回炉一批老帧）→ 重训 → 看报告、换上新 YOLO
4. `icon-eval` 跑验证帧：比两种分类器、标 `under_x` / `under_up`、定 `classifier` 和 `dino_match`
5. 开 `[icons]` 跑一晚：对着管理面板画面核对 status，查日志确认地图图标没被点过；记进进度文档

## 9. 文件

新增：`src/skydango/vision/icons_map.py`（`owner_of`、`map_label` 的数据、投票、DINOv2 最近邻 `IconGallery`、裁图、`Icon`）、`src/skydango/vision/icon_eval.py`（`icon-eval` / `icon-cut`）、`assets/icons/<kind>/`、`tests/test_icons_map.py`、`tests/test_icon_eval.py`；
改：`vision/perception.py`（orphans 改成按归属分流、`icons()`、overlay、攒裁图、难例 `icon_unknown`）、`game/social.py`（`KIND_NAMES`、`MAP_KINDS`）、`vision/inbox.py`（`add`、`--redo`、预标多一步）、
`cli.py`（`perception icon-eval` / `icon-cut` / `inbox add`）、`config.py` + `config.example.toml`（`[icons]`）、`console/settings.py`、`brain/body.py`（status）、`brain/images.py`（`scene_note`）、`brain/prompt.py`（一句）、
`console/static/stage.js`、`docs/game-ops.md`、`docs/progress/2026-09-28-yolo-training.md`（标注规则）、CLAUDE.md。
