# 感知层 · 一期：替换现有识别，上线可用 — 设计

日期：2026-09-28　状态：**代码部分已完成（§3~§5 的工具和逻辑，见实施方案 `docs/superpowers/plans/2026-09-28-perception-phase1.md`），M0 / M1 / M3 待 GPU 机器和真机**
总纲：`2026-09-28-perception-yolo-architecture-v0.2.md`（下称"总纲"）　后续：二期 `…-phase2-design.md`、三期 `…-phase3-design.md`

> **给实现这一期的 agent**：先读总纲 §2~§6（前提、分层、类别、世界状态、接法）和 `docs/game-ops.md` §5~§6。
> 感知层骨架已经在代码里（`vision/detect.py` `track.py` `perception.py` `weaklabel.py`，`[perception]` 默认关）。
> 这一期是在骨架上把它做到"能打开用"。往游戏里发输入的改动要先在真机截图验证（AGENTS.md「工作约定」）。

## 目标

训练出第一版模型，让 `[perception] enabled = true` 能替换现在的定时整图 OCR，并且**比现在更好**：
认得出陌生人、发现互动请求更快、不乱报"来了 / 走了"。

**一期完成的标志**：总纲 §9 的验收指标全部达标，且本文 §7 的补充验收通过，用户在 `config.toml` 里打开 `[perception]` 日常使用。

## 范围

| 做 | 不做（放到后面几期） |
|---|---|
| M0 测速、M1 数据和训练、M2 离线对比、M3 接进身体 | 环绕扫描、转圈认团子（二期） |
| **画面被挡时冻结状态**（§4） | 眼睛拿检测结果、认说话人、距离走向（二期） |
| **数据增强**（§2.3） | 远处二次检测、认地图、动作识别、跟随（三期） |
| **难例收集 + 用模型预标注**（§5） | |
| 离线对比工具 `perception compare`（§3） | |

## 1. M0 测速

工具已就绪：`python -m skydango perception bench --model yolo11n.pt`（总纲 §13 第 2 步）。

- 在 GPU 机器上、**开着游戏**跑，imgsz 960 和 640 各测一次
- 记录：YOLO 检测 p50 / p95、截图耗时、整帧耗时、`perception bench` 打印的后端（必须是 CUDA / TensorRT，不能是 CPU）
- 同时肉眼看游戏有没有变卡
- 结果写进总纲 §13 "在这台开发机上验证过"那段下面（新起一段"GPU 机器实测"）

验收见总纲 §9 的 M0 行。不达标：先试 TensorRT（`format=engine`），再降 imgsz，再降 fps。

## 2. M1 数据与训练

### 2.1 采集清单

用 `python -m skydango record --seconds 120 --fps 2` 分段录，第一轮目标 **≥ 400 张**，按场景尽量均匀：

| 场景 | 至少 | 说明 |
|---|---|---|
| 好友在身边（有名字标签） | 80 | 远近都要有；多人站一起、互相遮挡 |
| 陌生人：没点火的黑影 | 60 | 远近都要有 |
| 陌生人：点过火的 | 40 | 和好友外观一样、没有标签 |
| 暗场景（暮土、禁阁、夜晚） | 60 | 这里的普通人和黑影最容易混 |
| 强光效 / 特效 | 30 | 做动作、放烟花、光之翼附近 |
| 转视角中的模糊帧 | 40 | 边录边按住方向键 |
| 圆圈各种状态 | 40 | ✦、牵手 / 拥抱 / 击掌 / 背背请求、牵手时圆圈消失 |
| 聊天面板开着 / 关着 | 各半 | 贯穿以上各类 |
| 没有人的空画面 | 30 | 负样本，要**确定真的没人** |

### 2.2 标注

工具已就绪：`perception label <录像目录> --preview` 自动出 `name_tag` / `social_ring`，
人工在 X-AnyLabeling 里补 `player`、`player_unlit`、`self` 并修错框（总纲 §13 第 4~5 步）。

- **每张图都要补全**；类别顺序不能改（总纲 §4）
- 验证集（`--val 0.15`）里上表每类场景都要有，不然 mAP 看不出暗场景的问题

### 2.3 数据增强（新增）

**原则：真实数据优先，增强补不足。** 转视角的模糊帧、暗场景先靠 §2.1 真录；增强只用来放大这些样本。

训练参数（ultralytics）：

| 参数 | 值 | 原因 |
|---|---|---|
| `hsv_v` | 0.5 | 亮度变化大（白天 / 暮土 / 夜晚） |
| `hsv_s` | 0.5 | 光效、滤镜 |
| `fliplr` | 0.5 | 场景左右对称无所谓；名字标签镜像后仍像一行字，可以接受 |
| `flipud` | 0 | 游戏画面不会倒过来 |
| `degrees` | 0 | 镜头不会歪 |
| `mosaic` | 1.0（最后 10 轮关掉：`close_mosaic=10`） | 默认值，多人拼图有利于学遮挡 |
| `scale` | 0.5 | 远近变化 |

**离线增强脚本**（新增命令 `perception augment <数据集目录>`）：只处理 `images/train`，生成带后缀的新图，标注原样复制（框不变）：
- **水平运动模糊**：随机抽 30% 的图，核长 9~25 px（模拟转视角）→ `<名字>_blur.jpg`
- **压暗**：随机抽 20% 的图，gamma 1.8~2.5 → `<名字>_dark.jpg`
- 不对 `images/val` 做，验证集保持真实分布
- 可重复运行（已有 `_blur` / `_dark` 的不再生成），`--seed` 固定随机数

**难负样本**：暗场景里的普通 `player` 和 `player_unlit` 要成对出现在训练集和验证集里；
训完用 `yolo val` 看两类之间的混淆矩阵，混淆 > 10% 就先补这两类的真实数据。

### 2.4 训练记录

每次训练在 `models/` 里放一个同名的 `.txt`，记下：日期、数据集张数（train / val）、imgsz、epochs、基础模型、mAP50（各类）。
模型命名 `sky-yolo-<日期>-<imgsz>.onnx`，`config.toml` 的 `perception.model` 指向当前用的那个。

## 3. M2 离线对比工具 `perception compare`（新增）

```
python -m skydango perception compare <录像目录> [--model M] [--interval 3] [-o tmp/compare/<时间>]
```

同一批录像（`record` 录的，文件名 `NNNN_TTT.TTs.jpg` 里带时间）上跑两套识别器：
- **现有方案**：`EnvWatcher`，按 `--interval`（默认 3 s，同 `env.interval`）模拟定时扫描 —— 按文件名里的时间，距上次扫描 ≥ interval 才扫
- **YOLO**：`PerceptionWatcher`，每张都跑（`background=False`）

输出目录：
- `report.md`：总表 + 逐项结论（见下）
- `diff/<帧>.jpg`：两边结论不一样的帧，左右拼图，各自画框（复用 `overlay()`）
- `summary.json`：机器可读的同样内容

报告内容（对应总纲 §9）：

| 项 | 怎么算 |
|---|---|
| 好友认出率 | 每帧两边各认出哪些好友。"现有方案"只在它扫描的帧上比。列出两边不一致的帧（→ `diff/`），由人判断谁对 |
| 发现互动请求的延迟 | 每个 (谁, 哪种) 请求，两边第一次报出的时间差 |
| `stranger` 事件 | YOLO 这边按身体的规则（陌生人数 0 → 有、有 → 0）数事件次数，列出时间点，人对照录像判断有没有误报 |
| `leave` 事件 | 两边各自按 `keep` 模拟出的"走开"事件，列出时间点 |
| 耗时 | 两边平均每帧 / 每次扫描耗时 |

不做自动的"真值"：录像里谁在、谁是好友，由人看 `diff/` 判断。mAP 用 `yolo val` 单独看。

## 4. 画面被挡时冻结状态（新增，**M3 之前必须完成**）

### 问题

YOLO 版把"走开"的判定从 30 s 缩到 `keep` = 5 s。下面这些情况画面里的人会短暂"消失"，5 s 一过就会连发"XX 走开了"，
恢复后又是"XX 来到身边" —— 每个事件都叫醒大脑和眼睛，误导它又花额度：

| 情况 | 持续 | 谁知道 |
|---|---|---|
| 牵手被带着走时整屏黑（game-ops §6） | 十几秒 | 身体：`_watch_screen` 已经算 `blackout` |
| 身体自己转镜头：`camera` / `camera_reset` / `look_around` / 二期的 `#spin` | 1~5 s | 身体（是它发的指令） |
| `check_friend` 打开好友树面板 | 2~3 s | 身体 |
| 换轮盘（`emotes` 进编辑界面，盖住半个画面） | 5~10 s | 身体 |
| 接互动请求：点圆圈后团子走过去 | 1~4 s | 身体（`SocialHandler.accept`） |
| 玩家自己打开地图、商店等全屏界面 | 不定 | 谁都不知道 → 靠"集体消失"规则（见下） |

### 方案：暂停计时

`PerceptionWatcher` 加"暂停"：暂停期间**不跑检测、不累计"没看到"的时间**；恢复时把暂停的时长补回去。

```python
def hold(self, reason: str) -> None: ...        # 开始暂停（可以叠加多个原因）
def release(self, reason: str) -> None: ...     # 去掉一个原因；全部去掉才恢复
@contextmanager
def held(self, reason: str): ...                # with env.held("camera"): ...
```

- 恢复时：暂停了 `d` 秒，就把 `last_seen` 里每个名字的时间、陌生人记录（`_strangers`）的时间都加 `d`，
  追踪器里轨迹的 `last` 也加 `d`（不然轨迹会因为 `track_buffer` 过期而断掉，恢复后名字要重读一遍）
- 暂停期间 `observe()` 直接返回；`capture = "own"` 的线程也跳过检测
- `nearby()` / `strangers()` 在暂停期间照常返回暂停前的结果
- `EnvWatcher` 加同名的空实现（`hold` / `release` / `held` 什么都不做）：接口一致，身体不用判断是哪种识别器
- 暂停超过 `hold_max`（默认 60 s）自动恢复并打一条警告日志，防止哪里忘了 `release` 导致永远不报"走开"

### 谁来调用

| 触发 | 位置 | 原因名 |
|---|---|---|
| 整屏黑 / 恢复 | `Body._watch_screen`：`blackout` 变 True 时 `hold`，变 False 时 `release` | `"blackout"` |
| 转镜头 | `Body.camera_move` / `camera_reset` / `capture_around` 外面包 `with self.env.held("camera")` | `"camera"` |
| 好友树 | `Body.check_friend` 包 `held("friend_tree")` | `"friend_tree"` |
| 换轮盘 | `EmotePlayer` 换轮盘的那段（`perform` 需要换格子时）：身体调用处包 `held("wheel")` | `"wheel"` |
| 接互动 | `Body._watch_people` 调 `social.handle` 时包 `held("social")` | `"social"` |
| 普通模式 | `Agent` 里对应的几处（接互动、做动作）同样包上 | 同上 |

### "集体消失"规则（兜底：玩家自己开了全屏界面）

一帧里**上一帧还看得到的 ≥ 2 个人物（或名字标签）全部不见了**，且这一帧和上一帧的缩略图差异 > `brain.scene_change`（0.25）：
自动 `hold("occlusion")`；之后任何一帧又检测到人物或名字标签就 `release("occlusion")`；
最多等 `occlusion_hold`（默认 30 s），过了就放弃（恢复计时，之后照常按 `keep` 判定走开）。

只有 1 个人消失不触发（一个人走出画面是正常的）。

### 配置

`[perception]` 新增：`hold_max = 60.0`、`occlusion_hold = 30.0`。

### 测试

- `hold` / `release` 叠加：两个原因都 release 才恢复；`held()` 出错也 release
- 暂停 20 s 后恢复：`nearby()` 里的人不因为这 20 s 走开；恢复后再过 `keep` 没看到才走开
- 暂停期间 `observe()` 不调检测器
- 轨迹不因暂停而断：恢复后同一个名字标签不重复 OCR
- 集体消失：2 人同时消失 + 画面大变 → 自动暂停；1 人消失 → 不暂停；人回来 → 恢复；超过 `occlusion_hold` → 恢复
- `hold_max` 超时自动恢复
- 身体：`blackout` 期间没有 `leave` 事件；`camera_move` 期间没有 `leave` 事件（用假 env 验证调用了 `held`）

## 5. 难例收集 + 用模型预标注（新增）

第一版模型肯定会漏检、误检。运行时自动把"可能认错了"的画面存下来，下一轮只补标这些，比盲目多录多标省力。

### 5.1 收集（新模块 `vision/hardcases.py`，`HardCaseCollector`）

`PerceptionWatcher` 处理完一帧后交给收集器判断，满足任一条件就存：

| 原因 | 判据 |
|---|---|
| `ocr_only` | **旁路核对**：每 `audit_interval`（默认 30 s）在后台线程对当前帧做一次整图 OCR（复用 `EnvWatcher` 的扫描逻辑），OCR 以 ≥ 0.9 置信度读到某个好友名字，而 YOLO 这一帧没认出这个好友 |
| `yolo_only` | 同一次核对里，YOLO 认出了某好友，OCR 没读到（优先级低，可能只是 OCR 漏了） |
| `low_conf` | 某条轨迹连续 ≥ 0.5 s 的置信度都在 0.25~`conf` 之间（检测器要以 0.25 出框，`PerceptionWatcher` 再按 `conf` 过滤；低于 `conf` 的框只给收集器看） |
| `flicker` | 同一位置 3 s 内人物框出现又消失 ≥ 3 次 |
| `unlit_vs_player` | 同一条轨迹在 `player` 和 `player_unlit` 之间来回变（需要追踪器按位置跨类别关联，用 IoU ≥ 0.5 判断是同一个人） |

限额：每次运行最多 `hardcase_max`（200）张；两次保存至少隔 5 s；和上一张存下的图缩略图差异 < 0.03 就不存（去重）。
暂停（§4）期间不收集。

存放：`runs/<这次>/hard/<时分秒>_<原因>.jpg`（Windows 文件名不能带冒号） + `runs/<这次>/hard.jsonl`（每行：文件名、原因、细节、这一帧的检测框）。

`runs/` 只保留最近 `run.keep` 次，**要用的难例要及时收进数据集**（见 5.2）。

### 5.2 预标注：`perception label` 扩展

```
python -m skydango perception label <目录> [--model M] [--from-runs]
```

- `--model M`：除了现有的弱标注（OCR 名字框、模板圆圈），再用当前模型的预测（置信度 ≥ 0.25）当初始标注；
  同类框 IoU > 0.5 时保留弱标注的（OCR 的名字框更准）
- `--from-runs`：`<目录>` 是 `runs/`，把每次运行的 `hard/*.jpg` 收集起来（文件名前加运行目录名，避免重名）
- 人工在 X-AnyLabeling 里只需要**修正**，不用从零画

### 5.3 配置

`[perception]` 新增：`hardcases = true`（打开感知层时默认收集）、`audit_interval = 30.0`、`hardcase_max = 200`、`low_conf = 0.25`。

### 5.4 测试

- 各触发条件用假检测器 / 假 OCR 构造，确认存图、原因正确
- 限额、间隔、去重
- 暂停期间不收集
- `label --model`：弱标注和模型预测合并规则；`--from-runs` 收集路径

## 6. M3 接进身体

代码已接好（总纲 §6、§13 第 8 步），这一步是**上线检查**：

1. `config.toml`：`[perception] enabled = true`、`model` 指向训好的模型、`capture = "own"`
2. dry-run：`run --brain --view --duration 600`，用可视化网页（`docs/superpowers/specs/2026-09-28-viewer-design.md`）边看边核对
3. 翻 `runs/<这次>/agent.log`，逐条核对 `arrive` / `leave` / `stranger` / `request` 事件和实际画面
4. 专门测 §4 的几种遮挡：牵手被带走、转镜头、开好友树、换轮盘、自己开地图 —— 都不应该出现误报的 `leave`
5. 出问题：`enabled = false` 回退；问题画面放进难例

## 7. 一期验收

总纲 §9 全部指标，加上：

| 项 | 目标 |
|---|---|
| 遮挡误报 | §6 第 4 步的每种遮挡各测 3 次，**0 次**误报 `leave` |
| 难例收集 | 跑 30 分钟，存下的难例 ≤ 200 张，人工抽查 20 张，≥ 半数确实是模型的错 |
| 回退 | `enabled = false` 后行为和一期之前完全一样（跑现有单元测试 + 真机 5 分钟） |

## 8. 要改的地方

| 文件 | 改动 |
|---|---|
| `vision/perception.py` | `hold` / `release` / `held`、恢复时补时间、集体消失规则；把低置信度框交给收集器 |
| `vision/env.py` | `hold` / `release` / `held` 空实现 |
| `vision/track.py` | `shift(d)`：所有轨迹的 `last` 加 d；跨类别（`player` ↔ `player_unlit`）按 IoU 关联的选项 |
| `vision/hardcases.py`（新） | `HardCaseCollector` |
| `vision/detect.py` | 检测器出框阈值和 `PerceptionWatcher.conf` 分开（出框用 `low_conf`） |
| `brain/body.py`、`agent.py` | §4 表里的各处 `held(...)` |
| `runlog.py` | `hard/` 目录和 `hard.jsonl` |
| `config.py`、`config.example.toml` | §4、§5 的新配置 |
| `cli.py` | `perception compare`、`perception augment`、`perception label --model / --from-runs` |
| `docs/game-ops.md` | 真机确认的结果（全屏界面、遮挡时长） |
| 总纲 | §13 补 GPU 实测；§11 更新已解决的缺口 |

## 9. 要在真机确认的

- 玩家自己打开地图 / 商店 / 设置时，画面怎么变（"集体消失"规则的前提）
- 换轮盘编辑界面挡住多少画面、持续多久
- 接互动请求后团子走过去的这几秒，名字标签是否一直可见（决定 `"social"` 暂停是否必要）

## 10. 已确认（用户，2026-09-28）

1. 难例收集在打开感知层时默认开（`hardcases = true`），会往 `runs/` 写图 —— **可以**
2. 旁路核对每 30 s 做一次整图 OCR（约 1 s CPU，后台线程）—— **可以，30 秒一次**

## 11. 实现备注（代码部分完成后补）

- 集体消失（`occlusion`）暂停期间检测照跑，等人重新出现；其他原因的暂停不跑检测
- 暂停期间 `nearby()` / `strangers()` 按暂停开始那一刻算；恢复时所有时间补上暂停时长，但不超过恢复时刻
- 追踪器始终开 `player` ↔ `player_unlit` 跨类别关联（IoU ≥ 0.5），同一个人换类别仍是同一条轨迹
- 难例的 `low_conf`、`unlit_vs_player` 每条轨迹只存一次；同一帧有多个原因时文件名用第一个，`hard.jsonl` 的 detail 里都写
- `perception compare` 只认 `record` 录的文件名（带时间）；一张都没有时报错退出
