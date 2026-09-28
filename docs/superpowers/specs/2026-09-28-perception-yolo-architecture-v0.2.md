# 感知层 · YOLO 第一道关卡 v0.2

> 取代 v0.1（`2026-09-28-perception-yolo-architecture-v0.1.md`）。v0.1 是按从零开始写的，
> 没对上仓库现状；这一版只写**在现有架构上加 YOLO**这一件事。
>
> **进度（2026-09-28）**：代码骨架已完成、单元测试覆盖，默认关闭（`[perception] enabled = false`）。
> 还没有训练好的模型，也没在真机 / GPU 上跑过 —— 测速（M0）不阻塞后面的开发，GPU 机器到手后跑 §13 的步骤。
>
> 修订记录：
> - 初稿：分层、类别、路线
> - 骨架实现后：陌生人外观（没点火是黑影 → `player_unlit`）、好友树确认（`check_friend`，ESC 关）写进来；
>   §3 / §5 / §6 改成和实际实现一致；新增 §9 验收标准、§11 已知缺口
> - 新增 §14 提案：转视角环绕扫描、转圈认团子（**待确认，未实现**）

## 1. 目标

在现有身体 / 大脑架构前面加一层 YOLO：以 15fps 在 GPU 上检测画面里的人物、名字标签和互动圆圈，
作为视觉的第一道关卡——快速、持续地知道"画面里有什么、在哪"，
只在有变化时才去调用贵的环节（OCR、模板匹配、眼睛 Haiku、大脑）。

要解决的两个现有短板：
1. **陌生人看不见**：`vision/env.py` 靠 OCR 读名字标签认人，陌生人没有标签，身体完全不知道身边有陌生人。
2. **环境扫描又慢又盲**：env 每 3 s 对 1280×960 做一次整图 OCR（0.7~1 s，CPU），不管画面变没变；
   互动请求发现得慢（2~4 s），名字标签被挡一下就要靠 30 s 的 `keep` 窗口兜底。

## 2. 已定的前提

| 项 | 结论 |
|---|---|
| 平台 | MuMu 模拟器 + ADB（不变）；截图用 `device/mumu.py` 原生截图（约 9 ms/张） |
| 硬件 | 50 系 N 卡 → YOLO 跑 GPU，15fps 没压力（待 M0 实测） |
| 大脑 | 不变：Claude Code + MCP 工具，DeepSeek 备用；**不改成 JSON 意图** |
| 聊天 | 不变：读聊天记录面板（C），**YOLO 不管聊天气泡**（头顶气泡实测不可靠，见 game-ops §3） |
| 身份 | 不变：不能声称自己是真人（AGENTS.md 底线） |
| 好友 / 陌生人外观 | 没点火的陌生人是黑色剪影；点火后外观和好友一样，只能靠头顶名字标签分，或点人物看好友树（用户告知，game-ops §5） |

## 3. 分层

```
MuMu 截图（capture = "body" 用身体主循环的帧；"own" 感知线程自己截，15fps）
  → YOLO 检测：player / player_unlit / name_tag / social_ring / self（GPU）
  → 过滤：聊天面板开着时面板里的框、底部按钮栏、团子自己
  → 追踪（IoU 贪心；够用就不上 ByteTrack）+ 名字关联
  → 世界状态（轨迹上挂名字投票、身份）
      ├→ 按需精识别：新 name_tag 轨迹 → 裁小图只跑 OCR 识别（不跑检测）
      │              social_ring → 现有剪影模板匹配认图标（一次约 1 ms，每帧都认）
      ├→ 反射：好友发起互动 → SocialHandler 直接接受（现有逻辑）
      └→ 事件：arrive / leave / stranger / request → body.events → 唤醒大脑 / 眼睛
```

原则：
- **大模型不看检测框**。15fps 的框只进世界状态，汇总成事件后才给大脑；大脑要细节时自己 `status` / `look`。
- **YOLO 只回答"是什么、在哪"，不回答"是谁"**。好友还是点过火的陌生人由名字标签 + friends.md 判断，不做成 YOLO 类别
  （点过火的陌生人外观和好友一样，标注会自相矛盾）。唯一的例外是没点火的黑影：外观上就能分，单独一类 `player_unlit`。
- 贵的识别由 YOLO 的结果触发，不再定时整图扫。
- 需要点屏幕的确认（好友树）不放在感知层，由大脑按需调 `check_friend`。

## 4. 检测类别

| 类别 | 说明 | 用途 |
|---|---|---|
| `player` | 其他玩家的角色（含披风），**外观可见的**（好友，或点过火的陌生人） | 身边有几个人、陌生人检测、以后的跟随 |
| `name_tag` | 头顶名字标签 | 裁剪 → OCR 认名字 → 关联到 player |
| `social_ring` | 名字下方的圆圈（✦ / 互动图标） | 裁剪 → 模板匹配认请求类型 |
| `self` | 团子自己 | 排除自己（也可以用 `self_roi` 固定区域，见 §11；转圈自动认团子见 §14） |
| `player_unlit` | 没点火的陌生人：黑色剪影、看不到外观 | 一看到就是陌生人，不用等标签判定 |

- **类别顺序一旦开始标注就不能改**（YOLO 标注文件里存的是序号）。以后加类别只往后追加；`player_unlit` 就是这样加在最后的。
- 以后再考虑：陌生人头顶的彩虹心形图标（含义未知）、点火相关 UI、场景地标。

## 5. 世界状态（`vision/perception.py`）

- **名字**：name_tag 轨迹刚出现就裁小图 OCR 一次，之后每 `ocr_retry`（1 s）再读，最多 `ocr_votes`（3）次，取出现最多的好友名；
  对不上 friends.md 的一直按 `ocr_retry` 重试（一张十几毫秒）。
  **身份跟着名字走，不跟着轨迹 id 走**：转视角轨迹断了，新轨迹一读出名字就接回同一个人，不需要单独的重关联逻辑。
- **身份**：
  - `player_unlit` → 陌生人，立刻，远近都算
  - `player` 轨迹上出现过名字标签 → 不是陌生人（标签之后被挡住也不改判）
  - `player` 出现超过 `stranger_after`（1 s）一直没标签，且框高 ≥ `stranger_min_height`（8% 屏高）→ 陌生人
    （太远的好友标签可能读不到，框小的不判，免得把远处的好友当成陌生人）
- **圆圈**：挂到正上方的名字标签（圆圈中心在标签上沿往下约 2.23 倍标签高度，game-ops §6 实测）。
  好友标签下没有圆圈 → 圆圈状态记 None（牵着手时圆圈会消失，身体靠它猜牵手状态）。
  上面没有名字标签的圆圈算陌生人的，发起的请求记在"陌生人"名下，按 `accept_strangers`（只接点火）处理。
- **防抖**：好友 `keep`（5 s）没看到才算走开；陌生人数取最近 `keep` 秒里单帧最多看到几个（轨迹会断，不数轨迹条数）。
- **name_tag ↔ player 关联**：标签水平中心落在人物框（左右各放宽 25%）内，标签下沿在人物上沿往上 1.5 个身高到身体上半截之间，取最近的。

## 6. 和现有代码的接法

- **接口同 `EnvWatcher`**：`observe()`、`nearby()`、`describe()`、`requests`、`labels`、`circles`、`last_seen`，
  多了 `strangers()` / `unlit()` / `keep`。`Body._watch_people`、`SocialHandler`、眼睛不用改。
- **身体新事件 `stranger`**：陌生人数从 0 变成有（"身边来了陌生人（N 个，其中 M 个还没点火）"）和变回 0 时各发一次；
  `status` 多一项陌生人数；眼睛在 `stranger` 事件后也会自动看一眼（同 arrive / leave，受 `auto_look_min` 限制）。
- **帧从哪来**：`capture = "body"` 用身体主循环的截图（约 0.15 s 一张，到不了 15fps，但最安全）；
  `"own"` 感知线程按 `fps` 自己截。MuMu 截图缓冲区共用，已加锁排队。
- **配置 `[perception]`**：`enabled`、`model`、`device`、`imgsz`、`conf`、`iou`、`fps`、`capture`、`keep`、`stranger_after`、
  `stranger_min_height`、`ocr_retry`、`ocr_votes`、`self_roi`、`classes`、`track_buffer`、`track_iou`。
  `enabled = false` 时退回现在的整图 OCR，出问题能一键回退。
- **没做**：眼睛改成"实体集合变化"触发（现在 arrive / leave / stranger 事件已经覆盖了主要情况，画面大变仍用缩略图差）。

## 7. 推理后端

- 训练：Ultralytics（YOLO11n / YOLO26n 起步，不够再上 s）。
- 部署：导出 ONNX → `onnxruntime-gpu`，或导出 TensorRT 引擎；`.pt` / `.engine` 直接走 ultralytics。
- **50 系（Blackwell）要新版 CUDA**：PyTorch 要装 cu128 及以上的版本，onnxruntime-gpu / TensorRT 也要选支持 sm_120 的版本，
  旧版装上会报"no kernel image"或者退回 CPU —— 装完先确认真的在 GPU 上跑（`perception bench` 会打印后端）。
- 模拟器渲染也用这张卡，测速时看一下游戏帧率有没有掉。
- **推理尺寸 960**：名字标签只有 40~50 px 高，640 时缩到 15 px 左右，偏小。训练和推理的 `imgsz` 要一致；导出时固定了尺寸的 ONNX 以模型为准。

## 8. 数据

- 采集：`python -m skydango record` 录画面，要覆盖：好友 / 陌生人（**没点火的黑影**、点过火的）、多人重叠、转视角的模糊帧、
  暗场景（暮土、禁阁）/ 光效强的场景、圆圈的各种状态（✦、各种请求、牵手时消失）、聊天面板开着 / 关着、镜头拉近 / 拉远。
- 弱标注起步：`perception label` 用现有识别器自动出框 —— 整图 OCR 认出的好友名字 → `name_tag`，模板认得出的圆圈 → `social_ring`；
  人工补 `player`、`player_unlit`、`self`，修错框。工具用 **X-AnyLabeling**（能直接导入 / 导出 YOLO 格式）。
- **每张图都要补全**：只标了一部分的图会教模型"这里没有人"。没人的空画面也留一些（负样本），但要确定真的没人。
- 起步量：几百张标好的图先训一版，看漏检集中在哪类场景再补。
- 数据放 `datasets/`、模型放 `models/`（都 gitignore）。录像里有好友的昵称，别外传。
- 每次训练记下用的数据集版本和 `imgsz`（比如模型文件名 `sky-yolo-<日期>-960.onnx`），出问题能对上。

## 9. 验收标准（初定，M0 / M2 实测后调整）

| 阶段 | 指标 | 目标 |
|---|---|---|
| M0 测速 | YOLO 检测 p95（imgsz 960，GPU） | ≤ 20 ms |
| | 截图 + 整个感知每帧 | ≤ 66 ms（15fps） |
| | 游戏帧率 | 肉眼看不出掉帧 |
| M2 离线对比 | 好友认出率（同一批录像，逐帧对比现有整图 OCR） | 不低于现有方案 |
| | 发现互动请求的延迟 | ≤ 1 s（现在 2~4 s） |
| | 把好友误判成陌生人 | 10 分钟录像里 ≤ 1 次 `stranger` 事件 |
| | 验证集 mAP50（`player` / `name_tag` / `social_ring`） | 参考，≥ 0.8 |

达不到就先补数据再训，不急着打开 `enabled`。

## 10. 迭代路线

1. **M0 测速**：拿现成的 COCO 预训练 nano 模型在本机 GPU + MuMu 开着的情况下跑 15fps。—— 工具已就绪（`perception bench`），等 GPU 机器。
2. **M1 数据**：录画面 + 弱标注 + 人工补标，训第一版。—— 工具已就绪（`record`、`perception label`）。
3. **M2 离线对比**：按 §9 在同一批录像上比 YOLO 和现有整图 OCR。—— **还缺工具**：`perception compare <录像目录>`，
   两套识别器逐帧跑同一批图（`record` 的文件名里带时间），输出每帧认出的好友 / 陌生人 / 请求和汇总。下一步做。
4. **M3 接进身体**：`[perception] enabled = true`，先 dry-run 看 `runs/` 里的事件。—— 代码已接好，等模型。
5. **M4 以后**：跟随（先用牵手，视觉伺服最后做）、场景识别、用 `check_friend` 的结果补 friends.md（要用户确认）。
   §14 的环绕扫描可以在 M1 之后就做（转圈认团子能给 M1 的数据自动标 `self`），见 §14.6。

## 11. 已知缺口 / 待定

- [x] mumu.py 原生截图的线程安全：截图缓冲区共用，已加锁；DLL 句柄跨线程调用是否稳定仍待真机确认
- [x] 陌生人外观：没点火是黑影 → `player_unlit`；点过火的只能靠名字标签，或大脑调 `check_friend` 点人物看好友树（ESC 关）
- [ ] **打开 YOLO 后没有地名了**：EnvWatcher 顺带读地名（`env.places`），PerceptionWatcher 不读。地名提示本来就没在真机验证过，
  先接受这个缺口；要补的话，在画面大变（`scene_change`）后对画面中上部做一次整图 OCR，或者加一个地名文字的 YOLO 类别
- [ ] `self` 用类别还是固定区域：建议**先标 `self` 类别**（团子在画面中间、外观固定，标起来很快），`self_roi` 留作兜底；M1 训完看效果再定。
  §14 的转圈认团子如果可行，`self` 框可以从转圈录像里自动标，不用人工标
- [ ] 陌生人发起点火时，头顶有没有圆圈、在什么位置（现在假设和好友一样，没录到）
- [ ] `stranger_min_height`（8%）、`stranger_after`（1 s）、关联规则的系数都是估的，M2 用录像调
- [ ] 好友树面板长什么样、好友和陌生人的面板怎么区分（`friend-check X Y` 截图核对后写进 game-ops 和大脑提示词）
- [ ] 代码里的 ESC 是模拟 MuMu 实体键盘（sendevent），和用户在电脑键盘上按 ESC 是否等效，待 `friend-check` 实测

## 12. 风险

- **识别稳定性**：暗场景、光效、半透明重叠 → 有针对性地补数据；M2 的离线对比兜底。
- **误判陌生人会吵醒大脑**：`stranger` 事件会叫醒大脑和眼睛（有额度成本）。事件只在人数从 0 变有 / 变回 0 时发，
  M2 用 §9 的误判指标把关；实在多就给 `stranger` 事件加冷却。
- **GPU 争用**：模拟器渲染和推理共用显卡，M0 要看游戏帧率；不行就降 `fps` / `imgsz` 或导出 TensorRT。
- **封号**：纯视觉不变，不读内存、不注入；检测频率高不会增加游戏里的操作，操作频率仍由身体的护栏限速。
  `check_friend` 会点屏幕，限 30 秒一次、默认关。
- **回退**：`[perception] enabled = false` 恢复现有实现。

## 13. 实现与使用

### 代码

| 位置 | 内容 |
|---|---|
| `vision/detect.py` | 检测器：`.onnx` → onnxruntime（自带 letterbox、YOLOv8/11 输出 + NMS、YOLO26 端到端输出）；`.pt` / `.engine` → ultralytics |
| `vision/track.py` | IoU 贪心追踪 |
| `vision/perception.py` | `PerceptionWatcher`：接口同 `EnvWatcher`，多了 `strangers()` / `unlit()`；名字标签只跑 OCR 识别（`RapidOcrEngine.read_line`） |
| `vision/weaklabel.py` | 弱标注：整图 OCR 的好友名字框 → `name_tag`，模板认得出的圆圈 → `social_ring` |
| `game/friendtree.py` | 点人物打开好友树、截图、ESC 关掉、恢复聊天面板（大脑的 `check_friend`，`[friend_check]` 默认关） |
| `brain/body.py` | 新事件 `stranger`，`status` 里多一项陌生人数；`check_friend` 的护栏（要新鲜的 look 图、不点面板 / 按钮栏、牵手时不点、限频） |
| `cli.py` | `perception bench / detect / label`、`friend-check`；`run` 在 `[perception] enabled` 时用 YOLO 感知层替换 env 扫描 |

在这台开发机（云端 CPU，无 GPU）上验证过：yolo11n / yolo26n 导出的 ONNX 经 `OnnxYoloDetector` 解码，
和 ultralytics 官方推理的框差几个像素（letterbox 补边方式不同）；1920×1080 输入、640 推理约 30~40 ms/帧（CPU）。
训练 → 导出 → 加载的整条链路也用两张合成图跑通过（自定义类别名能从 ONNX 元数据读回来）。

### GPU 机器上的步骤

1. **装环境**（50 系 = Blackwell，要 CUDA 12.8+ 的构建）
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -e ".[yolo]"
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   用 `.onnx` 模型推理时还要 GPU 版 onnxruntime：先 `pip uninstall onnxruntime`，再 `pip install onnxruntime-gpu`
   （两个同时装会冲突；rapidocr 用 GPU 版也能跑）。不想折腾就直接用 `.pt` / `.engine`，走 ultralytics。
2. **M0 测速**（不用等训练）：开着游戏跑
   `python -m skydango perception bench --model yolo11n.pt`，看 YOLO 检测、截图的耗时，同时看游戏有没有变卡。
3. **M1 采数据**：`python -m skydango record --seconds 120 --fps 2`，多录几段（见 §8 要覆盖的场景）。
4. **弱标注**：`python -m skydango perception label tmp/record/<时间> -o datasets/sky --preview`
   （好友名单外的名字也想标就加 `--all-text`，之后人工删错的）。
5. **人工补标**：X-AnyLabeling 打开 `datasets/sky/images/train`（和 val），导入 YOLO 标注，
   **每一张**都补上 `player`、`player_unlit`（没点火的黑影）和 `self`、修正错框。没补全的图会教模型"这里没有人"。
6. **训练 + 导出**：
   ```bash
   yolo detect train data=datasets/sky/data.yaml model=yolo11n.pt imgsz=960 epochs=100
   yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=960   # 或 format=engine（TensorRT）
   ```
   把导出的模型放到 `models/sky-yolo.onnx`（`models/`、`datasets/` 都不进 git）。
7. **看效果**：`python -m skydango perception detect`（当前画面，标注图 `tmp/perception.png`），
   对照画面检查名字、陌生人、圆圈认得对不对。
8. **M2 / M3**：按 §9 对比达标后，在 `config.toml` 里 `[perception] enabled = true`（要 15fps 再加 `capture = "own"`），
   先 dry-run 跑 `run --brain --duration 300`，翻 `runs/` 里的日志看 arrive / leave / stranger 事件准不准。

## 14. 提案：环绕扫描与转圈认团子（待确认，未实现）

### 14.1 想法

1. **环绕扫描**：大脑需要时，身体按住方向键让镜头连续转一圈，YOLO 在转的过程中每帧检测，
   汇总成"一圈里哪个方向有谁"，交给大脑。
2. **转圈认团子**：开阔地带转视角时镜头绕着团子转，**团子在屏幕上几乎不动，别的东西都横着扫过**。
   所以一圈里"一直在画面中间附近、位置几乎不动"的那个人物就是团子。

### 14.2 环绕扫描

- **替代现在 `look_around` 的做法**：现在是每转 90° 停一下截图，4 张图交给眼睛（Haiku）描述，要十几秒、花额度。
  YOLO 版连续转一圈（按实测 0.5 s ≈ 90°，一圈约 2 s），15fps 能截约 30 帧，只给大脑文字，不叫 Haiku。
- **方位怎么算**：每个检测的方位 = 开始转以来按住的时长折算的角度 + 它在画面里偏离中心多远 × 水平视野角 / 画面宽。
  只需要粗分成 8 个方向（前、右前、右、右后……），对精度要求不高。
- **同一个人在多帧里出现要合并**：好友按名字合并；陌生人按方位聚成一个（相差 < 30° 算同一个）。
- **结果示例**：`正前方：懒洋洋大王；右后方：2 个陌生人（1 个没点火）；左边：番茄炒蛋盖饭（名字看不清，按位置推测）`。
- **转完回原位**：转满一圈正好回到原来的朝向，镜头净偏移不变（沿用 `Camera.around` 的做法）。
- **打开 YOLO 时 `look_around` 用这个，关着时仍用眼睛**。大脑的工具名和用法不变。

### 14.3 为什么是"按需转一圈"，不是"一直转"

- **转视角要关聊天记录面板**（现在的 `Camera` 是先关面板再转；面板开着能不能转没测过）→ 一直转就读不到聊天
- **接互动要点头顶的圆圈**，画面一直在转，坐标对不上
- **画面一直大变**，会不停触发 `scene_change`，把眼睛和大脑叫醒
- 镜头只在自己的屏幕上转，别人看不到，不会显得奇怪 —— 所以按需转没有社交上的顾虑

**什么时候转**：大脑调 `look_around` 时；刚到新地方（`scene_change` 之后）身体自己转一圈（要不要自动转，待确认）；
有人喊团子但画面里看不到人时。平时镜头不动，YOLO 照常看前方。

### 14.4 转圈认团子

- **判据**：一圈里出现在 ≥ 80% 的帧、画面位置移动最小（按框的中心算，相对画面宽度）、离画面中心最近的那条 `player` 轨迹 = 团子。
- **用处**：
  1. **自动标 `self`**：录几段转圈，自动给团子打 `self` 框，训练数据里团子不用人工标（§11 的待定项）。
  2. **运行时重新定位团子**：镜头拉近拉远、被墙挡住自动推近时，团子的位置和大小会变，固定的 `self_roi` 会不准；
     转一圈就能把 `self_roi` 更新成团子现在的位置和大小。
- **需要注意**：
  - **牵手时**对方紧挨团子，转的时候也几乎不动 → 取离画面中心最近、移动最小的；两个分不开就不下结论
  - 不转的时候团子本来也基本在固定位置（第三人称跟随镜头）；转圈真正补上的是"和团子站得很近、叠在一起的人"怎么区分
  - 前提是 YOLO 会在团子身上出框：第一版模型还没有 `self` 类时，团子会被当成 `player` 框出来，正好拿来用

### 14.5 要先在真机确认的

- 按住方向键是不是**匀速**转、转一圈要多少秒（现在只知道 0.5 s ≈ 90°）
- **聊天记录面板开着时方向键有没有反应**：能转的话，扫描时就不用关面板，影响小很多
- 转动中的画面**有多模糊**，YOLO 还认不认得（训练数据里要放转圈时的帧）
- 画面的**水平视野角**大概多少度（算方位用；可以转 90° 看同一个东西在画面上移动了多少像素来反推）
- 开阔地带以外（靠墙、室内）镜头会被挡住推近，团子会变大、位置可能偏，§14.4 的判据还成不成立

**已确认做手动测试命令**（设计见主人命令设计 `2026-09-27-owner-commands-design.md` §6，待实现）：
- 电脑前：`python -m skydango camera spin [--turns N] [--seconds S]`，截图存 `tmp/spin/<时间>/`
- 不在电脑前：普通模式 `run` 跑着时，主人在游戏里发 `#spin` / `#spin 2`，截图存 `runs/<这次>/spin/<时间>/`，团子回"转完了，1.9 秒 28 张"
- 每帧文件名带"按住后第几秒"，加上转前 / 转完两张图 —— 用来标定一圈的时长、看模糊程度、看镜头能不能转回原位

### 14.6 放在路线的哪里

- **转圈认团子**：M1 训出第一版模型（有 `player`）之后就能做，而且能反过来帮 M1 补 `self` 标注 → 建议排在 M1 和 M2 之间
- **环绕扫描**：依赖模型认人的效果，建议 M3（接进身体）之后做
- `camera spin` / `#spin` 不依赖模型，随时可以先做，用来回答 §14.5

### 14.7 待确认

1. 环绕扫描是否只在大脑调 `look_around` 时转，还是刚到新地方身体也自动转一圈？
2. 转圈认团子的结果，除了自动标注，要不要在运行时自动更新 `self_roi`？
3. ~~`camera spin` 手动命令要不要先做？~~ → 做，外加普通模式的 `#spin`（见 §14.5）
