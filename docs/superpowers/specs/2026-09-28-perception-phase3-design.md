# 感知层 · 三期：看得更远、更广 — 设计

日期：2026-09-28　状态：**代码部分已完成**（实施计划 `docs/superpowers/plans/2026-09-28-perception-phase3.md`）：
§1 / §2 / §5 的代码和测试都在，验收待 GPU 机器和真机；§3 只做了数据工具、离线评估和运行时接口（默认关），模型待录数据训练；
§4 只做了第 1 步（提示词），第 2 步待 `move` 工具。**§3 动作识别是研究性质，可能调整或取消**
总纲：`2026-09-28-perception-yolo-architecture-v0.2.md`（下称"总纲"）　前两期：`…-phase1-design.md`、`…-phase2-design.md`

> **给实现这一期的 agent**：先读总纲、一期 §4（暂停计时）、二期 §1（sweep、团子框）、§4（距离）。
> 这一期的各项**互相独立**，可以分给不同 agent 并行做；都依赖一期上线，§4 跟随还依赖大脑的 `move` 工具（见下）。
> 往游戏里发输入的改动要先在真机截图验证（AGENTS.md「工作约定」）。

## 目标

1. 远处的好友也认得出名字
2. 知道自己在哪张图（顺带补上"打开 YOLO 后没有地名"的缺口，总纲 §11）
3. 看得懂别人对团子做的动作（挥手、鞠躬……），能回礼
4. 能跟着好友走
5. 好友名单漏掉的人，能方便地补进 friends.md

## 范围

| 功能 | 依赖 | 性质 |
|---|---|---|
| §1 远处小目标二次检测 | 一期 | 优化 |
| §2 认地图 | 一期（用人物框遮掉人） | 新能力；补缺口 |
| §3 别人的动作识别 | 一期 + 二期的追踪 / 距离；大量新数据 | **研究性质** |
| §4 跟随 | 一期；大脑 `move` 工具（`2026-09-27-brain-move-design.md`，**尚未实现**） | 新能力；有风险 |
| §5 没认出的名字 → 补进 friends.md | 一期 | 小工具 |

## 1. 远处小目标二次检测

### 问题

imgsz 960 时，远处的名字标签只有十几像素高，YOLO 框不到、OCR 也读不出，远处的好友只能被判成"还没判定的人"。

### 方案：在人物头顶二次检测

- 触发：人物框高 < `far_height`（默认画面高的 8%，和一期 `stranger_min_height` 一致），且这条轨迹 1 s 内没有关联到名字标签
- 裁剪：人物框上方，宽 = 3 × 框宽、高 = 2.5 × 框高（名字标签在头顶上方），放大到 640 × 640 再跑一次检测，只要 `name_tag` / `social_ring`；
  框映射回原图坐标，交给平时的名字 OCR 流程
- 限额：每帧最多 `far_crops`（3）个裁剪，同一条轨迹每秒最多一次；GPU 上可以把几个裁剪拼成一批跑
- 对比方案：整张图 imgsz 提到 1280。二次检测只在需要时花时间，整图提尺寸每帧都花 —— 用 `perception bench` 和一期的 `perception compare` 两种都测，选划算的

### 实现（代码已完成）

- `PerceptionWatcher._far_tags`：候选 = `player`（黑影不用）、框高 < `far_height`、这一帧没挂上名字标签、离上次挂上标签和上次裁剪都 ≥ 1 s；
  每帧按"最久没裁过"取前 `far_crops` 个。裁剪（`far_region`，夹到画面内，太小不裁）直接交给检测器，它自己 letterbox 放大到 `imgsz`；
  找到的框映射回整图、和原图已有的同类框重叠（IoU ≥ 0.3）的丢掉，其余接进追踪器，走平时的名字 OCR / 圆圈流程
- 没做"几个裁剪拼成一批跑"：先测单张的耗时再说
- `perception compare` 报告多了"远处的人"一节（远处 player 认出好友名的比例）；`compare` / `bench` 加 `--far-crops N` 开关对比，`bench` 打印二次检测次数

### 验收

一期 `perception compare` 同一批录像：远处（框高 < 8%）好友的认出率提升 ≥ 30 个百分点；整帧 p95 耗时仍满足总纲 §9（≤ 66 ms）。

## 2. 认地图：参考截图匹配

### 问题

原来的 EnvWatcher 顺带读地名提示（未验证），YOLO 版没有；训练一个场景分类模型要大量数据，加新地图还要重训。

### 方案

- **图库**：`places/<地名>/*.jpg`（每个地方几张截图，不同角度 / 时间）。**不进 git**（截图里可能有好友），`.gitignore` 加 `places/`
- **加图**：`python -m skydango places add <地名>` 截当前画面存进图库（先遮掉人物和 UI，见下）
- **特征**：一个现成的小型图像特征模型（ONNX，候选：MobileCLIP-S0 图像编码器、DINOv2-small；用 `places bench` 在图库上比准确率和耗时再定），
  每张图算一个向量，图库的向量缓存到 `places/_index.npz`
- **遮挡处理**：算特征前把人物框、名字标签、圆圈（YOLO 结果）和固定 UI 区域（底部按钮栏、聊天面板）涂成画面平均色，免得"有没有人"影响匹配
- **判定**：和图库逐个算余弦相似度，最像的 ≥ `place_min`（默认 0.8，待标定）且比第二像的**别的地方**高 ≥ `place_margin`（0.05）才算认出
- **什么时候跑**：`scene_change` 事件后、以及每 `place_interval`（30 s）一次；暂停（一期 §4）期间不跑
- **输出**：`PerceptionWatcher.place` / `place_at`（和 EnvWatcher 同名字段，身体和提示词不用改），`describe()` 里加"- 看起来在：云野"
- **测试命令**：`places test [图片目录]` 对一批截图逐张打印认出的地点和相似度

### 实现（代码已完成）

- `vision/places.py`：`mask_scene`（遮挡）、`ThumbEmbedder`（内置基线，`model = "thumb"`）/ `OnnxEmbedder`（输入按 `size`、`norm` = imagenet / clip / none，
  输出取二维的那个，否则取 CLS token）、`PlaceLibrary`（图库 + 缓存）、`decide`（判定）、`PlaceRecognizer`
- `PerceptionWatcher` 每 `place_interval` 秒、或画面比上次认地图时大变（隔 ≥ 3 s）认一次；认不出不改原来的，`env.place_keep` 秒内写进提示词
- 命令：`places add <地名> [--image 图] [--model]`、`places test [目录]`、`places bench --model A --model B`（留一法，打印认对 / 认错 / 不说和耗时）
- `[places] enabled` 要配合 `[perception] enabled`；图库是空的时记警告、不认地图

### 验收

每个地方留 3 张不在图库里的截图做测试：认对 ≥ 90%，认错（说了一个错的地名）≤ 3%（宁可不说，不能说错）。

### 待确认

图库的截图由用户来截（`places add`），第一版覆盖哪些地方？建议先做常去的 5~8 个。

## 3. 别人的动作识别（研究性质）

### 目标

认出别人**对着团子**做的少数几个动作，发事件给大脑，让它决定要不要回礼（用现有的 `emote` 工具）。

### 第一版范围

只认 **2~3 个动作**（建议：挥手、鞠躬，具体由用户定）。认不准宁可不报。

### 方案

- **数据**：请好友配合，对着团子反复做这几个动作，用 `record --fps 8` 录；也录"站着 / 走路 / 做别的动作"作为负样本。
  按人物轨迹裁出 16 帧的片段（2 s），人工标动作
- **模型**：两个方向，先做 A，不行再试 B：
  - A. 每帧用小 CNN（或 YOLO 的主干网络）对人物裁剪提特征 → 时间上做池化 / 小 GRU → 分类
  - B. 用现成的人体姿态模型提关键点 → 关键点序列分类（光遇角色是卡通人形，姿态模型未必认得，要先验证）
- **触发**：只对距离"近 / 中"（二期 §4）、正对着团子（人物框在画面中间 1/2 内）的好友跑，每条轨迹每 2 s 判一次
- **事件** `gesture`："懒洋洋大王对你挥手" —— 每人每种动作 30 s 内只报一次；大脑自己决定回不回（提示词里说明可以用 `emote` 回礼，不强制）

### 实现（数据工具和运行时接口，默认关；还没有模型）

- 数据：`record --fps 8` → `perception clips <录像目录> [-o datasets/gesture/_unlabeled]` 按人物轨迹切成 16 帧的片段
  （`<序号>_track<轨迹>_t<开始秒>s/00.jpg…15.jpg`，人物框放宽 15%、缩放到 112×112）→ 人工把片段目录挪进 `datasets/gesture/<动作>/`
  （`none` 放站着、走路、别的动作的负样本）
- **模型约定**（训练脚本还没写，GPU 机器上按这个约定训练、导出 ONNX）：输入 `1×16×3×112×112`（RGB，0~1），
  输出 `1×len(labels)` 的分数（logits 或概率都行，推理时统一 softmax），顺序同 `[gesture] labels`（默认 `none, wave, bow`）
- 评估：`perception gesture-eval datasets/gesture --model …` 打印每个动作的精确率 / 召回率，标出是否达标
- 运行时：`[gesture] enabled` 时 `PerceptionWatcher` 对认出名字、近 / 中、在画面中间一半的好友攒片段，每 2 s 判一次；
  概率 ≥ `min_prob`（0.9）才报，同一人同一动作 30 s 一次；身体发 `gesture` 事件（牵着手的那个人不报），提示词说明可以用 `emote` 回礼

### 验收

精确率 ≥ 90%（报错动作会很尴尬），召回率 ≥ 60%；在录像上离线评估，再真机试 10 分钟。
**做不到精确率 90% 就不上线**，把结论写进本文。

## 4. 跟随

### 分两步

1. **靠牵手（现在就能用）**：牵着手时好友会带团子走（game-ops §6）。大脑想跟人走时，可以用 `say` 请对方牵手；接受牵手已经是身体的反射。
   这一步只改大脑提示词：说明"想跟着谁走，可以请他牵手"，**不加代码**
2. **视觉伺服**：没有牵手时，跟着某个好友走
   - 前提：大脑的 `move` 工具（`2026-09-27-brain-move-design.md`）已实现并在真机标定过步长
   - 新工具 `follow(name, seconds)`：身体每 0.5 s 看一次目标（按名字找轨迹）：
     目标偏左 / 偏右 → 转镜头（`camera`）让它回到中间；目标距离"远"（二期 §4）→ 往前走一小步；"近" → 停
   - 停止条件：到时间；目标 3 s 没看到；进了"近"；画面黑；主人发 `#pause`；**任何一步出错**
   - 限制：只跟好友；`seconds` 最多 30
   - 跟随期间**不用**一期 §4 的 `held(...)`（它会停掉检测，而跟随要一直看着目标）；
     改成身体在跟随期间不发 `arrive` / `leave` / `stranger` 事件（镜头和位置一直在变，这些事件没有意义），跟随结束后照常

> **第 2 步暂缓（2026-09-28）**：前提不成立 —— `brain/locomotion.py` 有了，但 `Body` / `ToolBox` 还没接 `move` 工具，
> 步长也没在真机标定。先实现 `2026-09-27-brain-move-design.md` 并标定，再做 `follow`。第 1 步（提示词）已改。

### 风险

- 地形：会走进水里、掉下悬崖、卡在墙角（game-ops §2 的 W 键实测：按住 1 s 就能走进水里）。第一版只在用户在旁边看着时用，`[follow] enabled` 默认关
- 光遇有飞行、滑翔，好友一飞就跟丢 → 跟丢就停，告诉大脑

### 验收

开阔地带跟着好友走 10 次（每次 20 s），不跟丢 ≥ 7 次，**0 次**掉水 / 掉崖。

## 5. 没认出的名字 → 补进 friends.md

### 问题

名字标签读出来了、但 friends.md 里没有（新加的好友、改了昵称），现在这些字就被丢掉了，这个人会一直被当成"还没判定的人"。

### 方案

- `PerceptionWatcher` 记下"读得很清楚（置信度 ≥ 0.9）但对不上 friends.md"的名字：出现次数、最后一次时间、一张裁剪图，
  存到 `runs/<这次>/unknown_names/`（每个名字一张图 + `names.jsonl`）
- 命令 `python -m skydango perception unknown-names [--runs runs/]`：汇总最近几次运行，按出现次数排序打印，附裁剪图路径
- **只列出，不自动写 friends.md**：用户看了决定要不要加（手动编辑 friends.md，或者在游戏里用 `#friend 昵称 备注`）
- 大脑用 `check_friend`（已有的工具）确认某人是好友时，可以在心里记下，但也**不自动写**文件

### 实现（代码已完成）

- `vision/unknownnames.py`：`UnknownNames`（`PerceptionWatcher` 读名字标签时，OCR 置信度 ≥ 0.9、对不上 friends.md 就记，
  同一条标签轨迹只记一次，和已记的名字相似（≥ 0.75）算同一个）；`collect` 汇总最近几次运行
- `python -m skydango perception unknown-names [--runs runs] [--last 5]`；运行结束时也会打印记了几个

### 验收

跑 30 分钟，列表里的名字人工核对：真的是没登记的好友名字 ≥ 80%（其余是 OCR 读错的）。

## 6. 要改的地方

| 功能 | 文件 |
|---|---|
| §1 | `vision/perception.py`（二次检测）、`config.py`（`far_height`、`far_crops`） |
| §2 | `vision/places.py`（新）、`vision/perception.py`（`place`）、`cli.py`（`places add / test / bench`）、`.gitignore`、`config.py`（`[places]`） |
| §3 | `vision/gesture.py`（新）、训练脚本、`brain/body.py`（`gesture` 事件）、`brain/prompt.py`（回礼说明） |
| §4 | 第 1 步：`brain/prompt.py`；第 2 步：`brain/follow.py`（新）、`brain/tools.py` + `mcp_server.py`（`follow`）、`config.py`（`[follow]`） |
| §5 | `vision/perception.py`、`runlog.py`、`cli.py`（`perception unknown-names`） |

每一项都要：单元测试（合成数据 / 假设备）、真机结果写进 `docs/game-ops.md`、完成后在总纲 §15 分期表里标"已完成"。

## 7. 待确认

实现时（2026-09-28）先按下面的默认做了，用户可以改：

1. §2 第一版图库覆盖哪些地方？→ 代码不依赖这个答案，用户自己 `places add`；建议先截常去的 5~8 个，每处 3~5 张
2. §3 第一版认哪 2~3 个动作？好友愿不愿意配合录数据？→ 默认挥手、鞠躬（`[gesture] labels` 可改）；
   录数据要好友配合，这一步用户来定，没有数据就一直关着
3. §4 第 2 步（视觉伺服）要不要做？→ 暂缓，前提（`move` 工具接好并标定）还没满足；先只用第 1 步（靠牵手）
