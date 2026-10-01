# 感知层第二层：人物属性 + 复核（设计）

日期：2026-10-02。总纲见 `2026-09-28-perception-yolo-architecture-v0.2.md`，这一份在 YOLO 之后加一层"裁图细看"。

## 0. 起因和拆分

用户想让团子看得更细：姿态 / 在做什么、朝向、外形类别、拿着什么，而且**不只告诉大脑，也要驱动身体**（朝向接进 `addressed()`、好友坐下就过去坐、弹琴就停下来听）。
讨论中又补了一个目标：YOLO 会把不是人的东西框成玩家，裁图交给第二个模型复核，能不能更准。

拆成三个子项目，各自 spec → 计划 → 实现：

1. **（这一份）第二层骨架 + 外形 / 复核头**：人物框 → 裁图 → 共享特征 → 属性头 → 按轨迹投票 → 放行 / 撤下、点没点火、外形进 `Person`
2. 补齐其余属性：姿态、朝向、拿着什么（骨架定了以后每个属性 = 标裁图 + 训一个头）
3. 身体用属性（每样是单独的行为改动，要真机验证；"走过去坐下"依赖还没做的 move / approach 技能）

已有的"裁图细看"（名字 OCR、圆圈模板、`far_crops`、认装扮、动作片段、火焰）这一期**不搬进来**，照旧。

## 1. 摸底（10-02，v7 在 `datasets/sky` 验证集 139 帧上；脚本和拼图在 `tmp/check/verify/`）

| | 数 | 看图 |
|---|---|---|
| 和标注对不上的人物框（门槛 0.2） | 47（≥ 0.35 的 24） | 约一半是真误框：开花的树（0.72）、红餐椅（0.53）、茶壶、篝火、地图 / 设置图标、雪人摊位装饰；另一半是**人、标注漏了**（长椅上坐 / 躺的、暗处黑影，最高 0.87） |
| 点没点火认反 | 16 | 两个方向都有：点过火的黑斗篷认成黑影、逆光的人认成点过火 |
| 漏掉的人（门槛 0.35） | 79 / 256（31%） | 难例 `runs/*/hard/` 里 0.2~0.35 的低分人物框**多数是真人**：挤在一起的黑影、半挡住的、坐着的 |

用户见过的误框：先祖 / 雕像 / 壁画（难例里有白色发光人形轮廓）。

结论：复核能挡误框，**更大的收益是降低 YOLO 门槛、让复核去捞低分框**；点没点火也能一起判。验证集本身漏标不少，评估之前要先补标（§4）。

## 2. 已定的事

- 运行时：冻住的 DINOv2-small（`models/dinov2-small.onnx`，复用 `OnnxEmbedder`）+ 每个属性一个线性头（numpy，`.npz`）。Claude 只当标注老师（初分），不进运行时
- 第一个头叫**外形**（`form`）：`not_person` 不是人 / `lit` 点过火 / `unlit` 黑影 / `spirit` 先祖 / `shared` 共享空间玩家 / `morph` 变身（雪人、白鹿……）
- 放行：YOLO 高分框（≥ `[perception] conf`）照旧马上放行、复核稳定判"不是人"再撤；低分框（`[perception] low_conf` ~ `conf`）等复核说是人才放行；追踪器升级（10-02 已合并）里"低分框只续不开"对人物类放开：可以开**待复核**的新轨迹
- 点没点火**两边都看**：YOLO 类别和外形头投票，带滞回
- 已标注数据自动进训练集；对不上的框 Claude 初分 + 标注页确认；确认是人的**写回 `datasets/sky`**（先备份）
- `[attrs] enabled` 默认关，过了上线门槛（§7）再开

## 3. 运行时（`vision/attrs.py`）

### 3.1 模型

```python
class AttrModel:
    """DINOv2 主干 + 若干线性头。npz 里：heads = ["form", ...]；每个头 <名>.W (D×K)、<名>.b (K)、<名>.labels (K)；
    每个头还有 <名>.applies_to（用在哪些 YOLO 类别上，外形头 = ["player", "player_unlit"]）、<名>.pad（裁图四周放多少）；
    还有 backbone（主干文件名 + OnnxEmbedder.key 里尺寸 / 归一化部分）、trained（日期）。"""
    def __init__(self, path: str, device: str) -> None: ...
    def predict(self, crops: list[np.ndarray]) -> list[dict[str, np.ndarray]]: ...  # 每张裁图：头名 → softmax 概率
```

- `predict` 一批裁图：逐张 `OnnxEmbedder.embed`（主干要是支持 batch 维的导出就一次跑；不支持就逐张，§9 第一步测速后定）→ 拼成 N×D → 每个头 `softmax(F @ W + b)`
- 主干和 npz 记的对不上（文件名 / 尺寸 / 归一化）→ 抛错，见 §6
- 裁图：`gesture.person_crop` 的做法（框四周各放 `pad` 比例、补成正方形、缩到主干尺寸），训练和运行时共用同一个函数 `attrs.crop()`

- **不只给人用**：头带 `applies_to`，`predict` 按裁图来自哪个类别只算对应的头。这一期只有外形头；以后图标头（§12）只是往同一个 npz 里加一个 `applies_to = ["social_ring"]` 的头，数据 / 标注 / 训练工具都按头名分目录（`datasets/attrs/<头名>/<类别>/`），不用改骨架

### 3.2 `PersonAttrs`（`PerceptionWatcher` 持有，跑在感知层现有后台线程里）

每帧在 `tracker.update()` 之后、分 `selfs / players` 之前调用：

```python
def update(self, frame, tracks: list[Track], now: float, panel_visible: bool) -> None
```

1. **挑要裁的**：人物轨迹（`player` / `player_unlit`，`self` 不裁）里这一帧更新过的。优先级：没复核过的 → 离上次裁图超过 `every`（0.5 s）的，按框高从大到小 → 每帧最多 `max_crops`（4）张。被聊天面板挡住一半以上的不裁（同认装扮的 `panel` 判断）
2. 一批 `predict`，每条轨迹 `data["form_hist"]` 追加 `(now, 概率)`，只留最近 `votes`（5）次
3. `data["form"] = (类别, 平均概率)`：最近几次概率取平均后最大的那一类；`data["form_n"]` = 次数
4. 感知暂停（`paused`）时不裁、不更新

### 3.3 放行（决定下游看不看得到这条轨迹）

`process()` 里 `players = …` 那一行改成先过 `attrs.admit(track)`：

| 轨迹 | 放行 |
|---|---|
| 出现过 YOLO 分 ≥ `conf` 的检测（`data["strong"]`，第一次就记） | 放行；`form` 是 `not_person`、平均概率 ≥ `reject`（0.7）、`form_n` ≥ `reject_n`（3）→ 撤下，`data["rejected"] = True`，之后一直不放行（同一条轨迹不翻案） |
| 只有低分检测 | `form` 是人形五类（`lit` / `unlit` / `spirit` / `shared` / `morph` 任一）、概率 ≥ `accept`（0.6）、`form_n` ≥ 2 才放行 |
| 还没复核过的低分轨迹 | 不放行 |

- **接追踪器升级（10-02 已在 main）**：检测器已经按 `low_conf`（0.25）出框，`low_conf ~ conf` 的框作为 `low` 交给 `Tracker.update(…, low=)`，**只续旧轨迹、不开新轨迹**；只靠低分框续着的轨迹，续命（`sticky_names`）最多 `LOW_ONLY_MAX`（5 s）。这一期：
  - `Tracker` 加参数 `open_low: frozenset[str]`（默认空 = 现在的行为）；`[attrs] enabled` 时传 `{"player", "player_unlit"}`：这两类的低分框配不上任何轨迹时开一条**待复核**的新轨迹（`strong_last` 保持 `-inf`，`data["strong"] = False`），返回值里和被低分框续上的轨迹一样附在后面
  - 复核放行一次（§3.2 第 3 步结论是人形、概率 ≥ `accept`）就把 `track.strong_last = now`：复核等于一次高分确认，续命和"只靠低分框"的 5 秒上限照这个时间算；被撤下的轨迹不刷新
  - 门槛不另设：摸底是在 0.2 做的、现在 `low_conf` 是 0.25，`attrs-train` 报告里两档都回放一遍，再决定要不要把 `low_conf` 调到 0.2（它也影响难例收集和续轨迹）
  - `low`（给 `promote_weak_self`、`LIT_LOW`、难例用的那批）含义不变
- 被撤下 / 没放行的轨迹照样留在 `tracker` 里（不删，免得下一帧又建一条新的从头复核），只是不进 `players`、不进 `_people_boxes`
- `enabled = false` 时：`open_low` 为空、`admit` 恒真，和现在逐字一样

### 3.4 点没点火

每条放行的人物轨迹算一个"黑影分" `u ∈ [0, 1]`：

- YOLO 一侧：这条轨迹最近 `votes` 次检测里 `player_unlit` 的比例（跟踪器跨类匹配，`cross` 里本来就有这两类）
- 外形一侧：平均概率里 `unlit / (lit + unlit + shared + morph)`（先祖不参加）
- `u = (yolo_w × YOLO 侧 + (1 − yolo_w) × 外形侧)`，`yolo_w` 默认 0.5；外形还没复核过时只用 YOLO 侧

`data["unlit"]` 滞回：初值按 `u ≥ 0.5`；之后连续 `flip_votes`（3）帧 `u` 都在另一边（> 0.6 或 < 0.4）才翻。
**下游所有 `t.cls == UNLIT` / `!= UNLIT` 的判断改成读 `is_unlit(t)`**（`enabled = false` 时就是 `t.cls == UNLIT`）：陌生人计数、`_assign_tags`、`far_tags` 的人选、认装扮的人选。`sweep()`（转圈汇总）用的是逐帧检测、没有轨迹，照旧看 YOLO 类别。

**不改的**：点亮陌生人那条链路（`feat/light-flame`：团子周围找火焰、`black()` 量人物多黑、`lit_frames` 判点亮）照它自己的规则，这一期不接外形头 —— 它刚改完、还没上真机，两边一起动出问题分不清是谁。等它真机过了，再单独评估要不要让"点亮了"也参考外形头。

### 3.5 先祖、共享空间、变身

- `spirit` / `shared`：放行（画面里看得到），但**不算陌生人、不发 `stranger` / `approach` / 来去事件、不挂名字标签**；`people()` 里 `kind = "spirit"` / `"shared"`
- `morph`：当普通人处理（变身的是玩家，可能是好友，名字标签照挂），`people()` 里多带 `form = "morph"`
- YOLO 自己的 `spirit` 类（物品那一组，编号 9）照旧走 `objects()`；外形头判成 `spirit` 的人物轨迹也并进 `objects()` 的先祖，**同一位置两边都有时只报一个**（框 IoU ≥ 0.5 算同一个）
- status："画面里：小明（左边·近）、一个先祖（右边·远）、一个共享空间的人（前面·中）"

### 3.6 `Person` 多带的

`people.py` 的 `Person` 加 `form: str | None`（外形类别，没复核过是 `None`）和 `form_p: float`。以后姿态 / 朝向 / 拿着什么照这个样子加字段，子项目 2 再定。

## 4. 数据和标注

### 4.1 `perception crops <来源> [--model 模型] [--out datasets/attrs]`

来源三种（可以一次给几个）：

1. **已标注数据集**（`datasets/sky`）：人物类标注框直接裁，自带类别 —— `player` → `lit`、`player_unlit` → `unlit`、`spirit`（编号 9）→ `spirit`；`self` 不要。增强图（`perception augment` 出的）不要
2. **同一个数据集上模型跑出来、和标注对不上的框**：`--model` 在 `--conf`（默认 0.2）跑，`player` / `player_unlit` 预测框和任何人物标注 IoU < 0.4 的裁出来，**类别未知**
3. **难例**（`runs/*/hard/` 或任意录像目录）：模型在 `--conf` 跑出来的人物框，类别未知；同一张图里两框 IoU ≥ 0.5 只要一个

裁图存 `datasets/attrs/_unlabeled/<来源帧名>__<x>_<y>_<w>_<h>.jpg`，旁边一份 `_crops.jsonl`（来源图、框、模型分数、YOLO 类别、从哪来）；
已知类别的直接进 `datasets/attrs/form/<类别>/`。重跑跳过已经有的（按文件名）。

### 4.2 Claude 初分：`perception attrs-label [目录]`

仿照 `gesture-label`：未知裁图每 16 张拼一张 4×4（每格标编号），交给 `claude -p`（`assist.Reviewer`：分批并发、缓存、额度用完能续跑、令牌同 `[brain]`）。
提示词给六类的定义和例子（树、椅子、UI 图标、篝火是 `not_person`；先祖 = 修长偏透明发光；共享空间 = 矮小蓝色半透明；变身 = 雪人 / 白鹿等，是玩家），
回每格的类别 + 一句理由，写进 `_unlabeled/claude.json`。

### 4.3 标注页「外形」

管理面板「标注」页（`console/labeling.py`）加一个标签页：一次显示一张大裁图 + 它在原图里的位置缩略图 + Claude 的猜测和理由。
按键同动作那一页：回车 = 同意 Claude、1~6 = 类别、0 = 丢弃、Z = 撤销；确认 = 挪进 `datasets/attrs/form/<类别>/`，记在 `_labels.jsonl`。
`labeling.py` 现在是按"片段 = 文件夹"写的，这里抽出"条目 = 文件"的共用部分（挪文件、日志、撤销），两种标签页共用。

### 4.4 写回 `datasets/sky`：`perception crops --writeback`

确认成人形五类的、来源是 §4.1 第 2 种（数据集里对不上的框）的，按原框写回那一帧的标注：`lit` / `morph` / `shared` → `player`，`unlit` → `player_unlit`，`spirit` → `spirit`。
写之前把 `labels/` 备份到 `_backup/labels-<时间>`；同一帧已经有 IoU ≥ 0.5 的人物框就不写；打印写了几帧几个框。难例来源（第 3 种）不写回（不在数据集里）。

## 5. 训练：`perception attrs-train [datasets/attrs] [--out 路径]`

- **切分**：按录像（裁图来源帧名里的录像名）切训练 / 验证，写 `datasets/attrs/_split.json`（同 `gesture_train.split`）；`datasets/sky` 的验证帧来的裁图一律进验证
- **特征**：主干跑一遍、按 `OnnxEmbedder.key` 缓存进 `datasets/attrs/_features/`；训练时左右翻转的特征也缓存一份
- **头**：带类别权重的多类逻辑回归（numpy 写，几百维 × 几千张，几秒训完，不要 torch）；L2 系数在验证集上从几档里挑
- **样本不够**：某一类训练样本 < `min_per_class`（20）时，`shared` / `morph` 并进 `lit`（都是点过火的玩家，后果一样）；`spirit` 不够就从头里去掉（训练时丢掉这类样本），运行时先祖照旧只靠 YOLO 的 `spirit` 类。`not_person` / `lit` / `unlit` 任何一类不够就不出模型、报错。报告里写明哪类被并 / 被去掉
- **输出**：`models/attrs-<日期>.npz`（不覆盖 `[attrs] model` 指的那个，要覆盖加 `--force`）+ `tmp/attrs-train/<时间>/report.md`：
  - 每类精确率 / 召回率 / 混淆矩阵
  - **整帧回放**：`datasets/sky` 的验证帧上，模拟 §3.3 的放行（单帧版：每个框当成复核了 `reject_n` 次同样的结果），比较"只用 YOLO（门槛 `conf`）"和"YOLO（0.2 和 `low_conf` 两档）+ 复核"的人物精确率 / 召回率，以及点没点火认反几个
  - 建议的 `accept` / `reject`（在验证集上扫一遍，精确率不低于只用 YOLO 前提下召回最高的那组）

`perception attrs-eval <录像目录或数据集> --model 模型`：只做整帧回放和报告，不训练。

## 6. 出错和边界

- `[attrs] model` 不存在 / 打不开 / 主干对不上：启动时 WARNING 一次，`PersonAttrs` 不建，等于 `enabled = false`（也不开低分轨迹）
- 单帧 `predict` 出错：`log.exception`，这一帧不更新任何轨迹（放行状态沿用上一帧）；连续 `max_errors`（10）帧出错就关掉整个第二层、不再开低分轨迹，WARNING
- 低分轨迹在复核前一律不放行（宁可晚一点看到）
- 跟踪器里留着的被撤轨迹（§3.3）跟着跟踪器的 `buffer` 正常过期
- `[perception] enabled = false` 时 `[attrs]` 不生效

## 7. 上线门槛

`[attrs] enabled` 默认 `false`，全满足才建议打开（写进进度文档）：

1. 整帧回放：人物**精确率不低于**只用 YOLO；**召回率明显上去**（目标漏检 31% → 20% 以下 —— 估的，第一份报告出来后按实际改）
2. 点没点火认反的数比只用 YOLO 少
3. 5070 Ti 上 `perception bench` 加 `--attrs`：感知层帧率降不超过 10%
4. 真机三步（§9）

## 8. 看得到的地方

- **识别可视化**：被撤的框画灰色虚线、标"不是人 0.9"；靠复核放行的低分框标"复核"；先祖 / 共享空间换颜色（沿用物品先祖的颜色 / 浅蓝）；鼠标悬停看外形六类概率和黑影分 `u`
- **难例**多两种原因：`attrs_disagree`（放行的轨迹 YOLO 侧和外形侧对点没点火意见相反超过 2 秒）、`attrs_reject`（高分框被撤）
- `perception detect [图片]` 的标注图里同样画出

## 9. 实施顺序和验证

**前置**：`feat/light-flame` 合进 main 之后再动 `perception.py`（它正在改同一个文件的点火部分）。追踪器升级（`2026-10-01-tracking-relink-motion-design.md`）10-02 已合并，这一期在它上面接：`open_low` 和复核刷新 `strong_last`（§3.3）；失踪好友接回（`relink`）、续命（`sticky_names`）、运动方向不接 `rejected` 的轨迹和外形是 `spirit` / `shared` 的轨迹。

1. 测速：`dinov2-small.onnx` 在 cuda 上单张 / 批 4 张各多少 ms，导出的有没有 batch 维（没有就重导一份带动态 batch 的，放 `models/dinov2-small-b.onnx`）
2. 数据：`perception crops datasets/sky --model models/sky-yolo-v7.pt` + 难例 → `attrs-label` → 标注页确认 → `--writeback`
3. 训练 + 报告，按 §7 第 1~3 条看数
4. 真机（晚上）：
   - `view --images <录像>`（不发输入）看灰框 / 复核框对不对，特别是茶座、开花的树、篝火那几段
   - `run --dry-run` 10 分钟，`[attrs] enabled` 开 / 关各一次，比较 `stranger` / `approach` 事件数和 `hard/` 里的 `attrs_*`
   - 有黑影来时看点亮陌生人那条链路照常（这一期没改它，只确认没被放行规则打乱）

**测试**（`python -m pytest -q`，不要真模型）：假主干（固定向量）+ 手写线性头：
放行四种情况（高分放行 / 高分被撤后不翻案 / 低分复核后放行 / 低分未复核不放行）、`max_crops` 优先级、投票和 `form_n`、黑影分加权和滞回、
`is_unlit` 在 `enabled = false` 时等于 `cls == UNLIT`、主干对不上时退回、连续出错关掉、先祖 / 共享空间不算陌生人、`objects()` 不重复报先祖；
`crops` 的三种来源和 IoU 去重、`--writeback` 的备份和不重复写、训练的切分和样本不够时去掉类别、npz 读写往返。

## 10. 配置（`[attrs]`）

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | `false` | |
| `model` | `"models/attrs.npz"` | |
| `backbone` | `"models/dinov2-small.onnx"` | 必须和 npz 里记的一致 |
| `device` | 跟 `[perception] device` | |
| `every` | 0.5 | 每条轨迹多久裁一次（秒） |
| `max_crops` | 4 | 每帧最多裁几张 |
| `votes` | 5 | 每条轨迹留几次结果 |
| `accept` | 0.6 | 低分框放行的概率门槛 |
| `reject` / `reject_n` | 0.7 / 3 | 撤下高分框 |
| `yolo_w` | 0.5 | 点没点火里 YOLO 一侧的权重 |
| `flip_votes` | 3 | 黑影滞回要连续几帧 |
| `max_errors` | 10 | 连续出错几帧关掉 |

数字都是估的，`attrs-train` 报告会给 `accept` / `reject` 的建议值。管理面板加 `attrs.enabled` 开关。

## 11. 改哪些文件

- 新：`vision/attrs.py`（`AttrModel`、`crop`、`PersonAttrs`、`is_unlit`）、`vision/attrs_data.py`（`crops`、writeback、`attrs-label` 的提示词和解析）、`vision/attrs_train.py`（切分、特征缓存、逻辑回归、回放报告）
- 改：`vision/track.py`（`open_low`）、`vision/perception.py`（`open_low` / 复核刷新 `strong_last`、`update` / `admit`、`is_unlit` 替换、先祖 / 共享空间、`objects()` 去重）、`vision/people.py`（`form`）、`vision/hardcases.py`（两种原因）、`vision/viewer.py` + 静态页（画法）、
  `console/labeling.py` + 标注页 js（「外形」标签页）、`config.py`（`AttrsConfig`）、`cli.py`（`perception crops / attrs-label / attrs-train / attrs-eval`、`bench --attrs`）、`brain` 里拼 status 的地方（先祖 / 共享空间的说法）
- 文档：CLAUDE.md 代码结构表和「YOLO 感知层」一节、进度文档

## 12. 不做的

- 姿态 / 朝向 / 拿着什么（子项目 2）、身体用属性（子项目 3）
- 重训 YOLO 合并 `player` / `player_unlit`（等第二层稳了再说）
- 认装扮改用同一份 DINOv2 特征（颜色特征的问题另说，见「认装扮」）
- 把名字 OCR、圆圈模板、火焰这些已有的细看搬进来
- **图标头**（10-02 用户提的：YOLO 只框"图标 / 圆圈 / 头顶的东西"，DINOv2 头分是什么图标）：骨架留了口子（§3.1 `applies_to`），值得做的候选是 `typing` 和白色"对话"图标分不清（09-30 物品模式标错 80 个）、圆圈里模板没覆盖到的新图标；
  固定图案的小图标模板匹配往往已经够准，要不要换、换哪些，等外形头上线后单独评估
- 点亮陌生人链路接外形头（§3.4）
