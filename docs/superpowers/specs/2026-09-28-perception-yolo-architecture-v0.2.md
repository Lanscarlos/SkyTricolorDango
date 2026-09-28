# 感知层 · YOLO 第一道关卡 v0.2

> 取代 v0.1（`2026-09-28-perception-yolo-architecture-v0.1.md`）。v0.1 是按从零开始写的，
> 没对上仓库现状；这一版只写**在现有架构上加 YOLO**这一件事。
>
> **进度（2026-09-28）**：代码骨架已完成、单元测试覆盖，默认关闭（`[perception] enabled = false`）。
> 还没有训练好的模型，也没在真机 / GPU 上跑过 —— 测速（M0）不阻塞后面的开发，GPU 机器到手后跑 §12 的步骤。

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
| 硬件 | 50 系 N 卡 → YOLO 跑 GPU，15fps 没压力 |
| 大脑 | 不变：Claude Code + MCP 工具，DeepSeek 备用；**不改成 JSON 意图** |
| 聊天 | 不变：读聊天记录面板（C），**YOLO 不管聊天气泡**（头顶气泡实测不可靠，见 game-ops §3） |
| 身份 | 不变：不能声称自己是真人（AGENTS.md 底线） |

## 3. 分层

```
MuMu 截图（感知线程，15fps）
  → YOLO 检测：player / name_tag / social_ring（GPU）
  → 追踪（ByteTrack）+ 名字关联
  → 世界状态（实体表）
      ├→ 按需精识别：新 name_tag 轨迹 → 裁小图 OCR 识别（只跑 rec，不跑 det）
      │              social_ring 裁剪变化 → 现有剪影模板匹配认图标
      ├→ 反射：好友发起互动 → SocialHandler 直接接受（现有逻辑）
      └→ 事件：arrive / leave / stranger_near / request / scene_change → body.events → 唤醒大脑 / 眼睛
```

原则：
- **大模型不看检测框**。15fps 的框只进世界状态，汇总成事件后才给大脑；大脑要细节时自己 `status` / `look`。
- **YOLO 只回答"是什么、在哪"，不回答"是谁"**。好友还是陌生人由名字标签 + friends.md 判断，不做成 YOLO 类别
  （同一套外观可能是好友也可能是陌生人，标注会自相矛盾）。
- 贵的识别由 YOLO 的结果触发，不再定时整图扫。

## 4. 检测类别（初版）

| 类别 | 说明 | 用途 |
|---|---|---|
| `player` | 其他玩家的角色（含披风） | 身边有几个人、陌生人检测、以后的跟随 |
| `name_tag` | 头顶名字标签 | 裁剪 → OCR 认名字 → 关联到 player |
| `social_ring` | 名字下方的圆圈（✦ / 互动图标） | 裁剪 → 模板匹配认请求类型 |
| `self` | 团子自己 | 排除自己；也可以先用屏幕中间固定区域代替，不标 |

以后再考虑：陌生人头顶的彩虹心形图标（含义未知）、点火相关 UI、场景地标。

## 5. 世界状态

实体表（每条轨迹一行）：

```
track_id, 屏幕框, 名字(可空), 身份(friend / stranger / unknown), 圆圈状态, 首次出现, 最后出现
```

- **身份**：轨迹上关联到能对上 friends.md 的名字 → friend；持续 N 帧（如 1 s）都没有标签 → stranger；之前 → unknown。
- **名字**：每条轨迹 OCR 一次，结果缓存；读不清就下一次标签变大 / 变清楚时再读，多帧投票。
  轨迹断了（转视角、被挡）用名字重新接回原实体。
- **防抖**：实体消失超过 `leave_after`（如 5 s，比现在的 30 s 短得多，因为 15fps 下不会被偶尔一帧遮挡骗到）才发 `leave`。
- **name_tag ↔ player 关联**：标签框在 player 框正上方、水平中心接近；圆圈在标签下方约 2.23 倍标签高度（game-ops §6 的实测值，可直接用来校验）。

## 6. 和现有代码的接法

- **`EnvWatcher` 的接口不变**：`nearby()`、`requests`、`circles`、`labels`、`describe()` 照旧对外提供，
  内部实现换成 YOLO + 世界状态。`Body._watch_people`、`SocialHandler` 基本不用改。新增陌生人时加一个事件（`stranger_near` / `stranger_leave`）。
- **单独的感知线程**：身体主循环约 0.15 s 一圈、还要读聊天和执行命令，15fps 放不进去。
  感知线程自己截图、跑 YOLO、更新世界状态；身体线程每圈只读状态快照。**需要确认**：mumu.py 的原生截图能不能两个线程同时调
  （不能就加锁，或者由感知线程截、身体线程取它的最新帧）。
- **眼睛触发**：`_watch_screen` 现在用缩略图像素差判断"画面大变"；可以改成（或加上）"实体集合变化"触发，更准，少叫 Haiku。
- **配置**：新增 `[perception]`：`enabled`、`model`、`fps`（默认 15）、`device`（`cuda` / `cpu`）、`conf`、`leave_after`、`stranger_after`。
  `enabled = false` 时退回现在的整图 OCR，出问题能一键回退。

## 7. 推理后端

- 训练：Ultralytics（YOLO11n / YOLO26n 起步，不够再上 s）。
- 部署：导出 ONNX → `onnxruntime-gpu`，或导出 TensorRT 引擎。
- **50 系（Blackwell）要新版 CUDA**：PyTorch 要装 cu128 及以上的版本，onnxruntime-gpu / TensorRT 也要选支持 sm_120 的版本，
  旧版装上会报"no kernel image"或者退回 CPU —— 装完先确认真的在 GPU 上跑。
- 模拟器渲染也用这张卡，测速时看一下游戏帧率有没有掉。

## 8. 数据

- 采集：`python -m skydango record` 录画面，要覆盖：好友 / 陌生人（灰的、点过火的）、多人重叠、转视角的模糊帧、
  暗场景 / 光效强的场景、圆圈的各种状态、聊天面板开着 / 关着。
- 弱标注起步：用现有识别器自动出框 —— env 的 OCR 框 → `name_tag`，social 的圆圈位置 → `social_ring`；
  人工只补 `player` 框、修错框（Label Studio / CVAT / X-AnyLabeling 均可）。
- 起步量：几百张标好的图先训一版，看漏检集中在哪类场景再补。
- 数据放 `datasets/`（gitignore），不进仓库。

## 9. 迭代路线

1. **M0 测速**：拿现成的 COCO 预训练 nano 模型在本机 GPU + MuMu 开着的情况下跑 15fps，确认耗时、GPU 占用、游戏不掉帧。
2. **M1 数据**：录画面 + 弱标注 + 人工修正，训第一版 `player` / `name_tag` / `social_ring`。
3. **M2 离线对比**：同一批录像上，YOLO + 裁剪 OCR 和现有整图 OCR 比：认出的好友、发现请求的延迟、误报。**达标才替换**。
4. **M3 接进身体**：感知线程 + 世界状态，`EnvWatcher` 换实现，保留 `enabled = false` 回退；加陌生人事件。
5. **M4 以后**：跟随（先用牵手，视觉伺服最后做）、场景识别。

## 10. 待定

- [x] mumu.py 原生截图的线程安全：截图缓冲区是共用的，已加锁（两个线程同时截会排队）；DLL 句柄跨线程调用是否稳定仍待真机确认
- [ ] 陌生人没点火时是不是灰色剪影、点过火后和好友外观是否一样（真机确认，决定 `stranger` 判断的规则）
- [ ] `self` 用类别还是固定区域

## 11. 风险

- **识别稳定性**：暗场景、光效、半透明重叠 → 有针对性地补数据；M2 的离线对比兜底。
- **封号**：纯视觉不变，不读内存、不注入；检测频率高不会增加游戏里的操作，操作频率仍由身体的护栏限速。
- **回退**：`[perception] enabled = false` 恢复现有实现。

## 12. 实现与使用

### 代码

| 位置 | 内容 |
|---|---|
| `vision/detect.py` | 检测器：`.onnx` → onnxruntime（自带 letterbox、YOLOv8/11 输出 + NMS、YOLO26 端到端输出）；`.pt` / `.engine` → ultralytics |
| `vision/track.py` | IoU 贪心追踪 |
| `vision/perception.py` | `PerceptionWatcher`：接口同 `EnvWatcher`，多了 `strangers()`；名字标签只跑 OCR 识别（`RapidOcrEngine.read_line`） |
| `vision/weaklabel.py` | 弱标注：整图 OCR 的好友名字框 → `name_tag`，模板认得出的圆圈 → `social_ring` |
| `brain/body.py` | 新事件 `stranger`（陌生人来了 / 都走了），`status` 里多一项陌生人数；眼睛在 `stranger` 事件后也会自动看一眼 |
| `cli.py` | `perception bench / detect / label`；`run` 在 `[perception] enabled` 时用 YOLO 感知层替换 env 扫描 |

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
   **每一张**都补上 `player` 和 `self`、修正错框。没补全的图会教模型"这里没有人"。
6. **训练 + 导出**：
   ```bash
   yolo detect train data=datasets/sky/data.yaml model=yolo11n.pt imgsz=960 epochs=100
   yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=960   # 或 format=engine（TensorRT）
   ```
   把导出的模型放到 `models/sky-yolo.onnx`（`models/`、`datasets/` 都不进 git）。
7. **看效果**：`python -m skydango perception detect`（当前画面，标注图 `tmp/perception.png`），
   对照画面检查名字、陌生人、圆圈认得对不对。
8. **M2 / M3**：确认没问题后在 `config.toml` 里 `[perception] enabled = true`（要 15fps 再加 `capture = "own"`），
   先 dry-run 跑 `run --brain --duration 300`，翻 `runs/` 里的日志看 arrive / leave / stranger 事件准不准。
