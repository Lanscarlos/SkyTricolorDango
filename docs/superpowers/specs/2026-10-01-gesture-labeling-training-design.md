# 动作识别：Claude 初分 + 管理面板「标注」页 + 训练导出（设计）

日期：2026-10-01。感知层三期 §3（`docs/superpowers/specs/2026-09-28-perception-phase3-design.md`）只做了数据工具（`perception clips`）、
离线评估（`perception gesture-eval`）和运行时接口（`OnnxGestureClassifier`，`[gesture]` 默认关），**没有模型、没有训练脚本、没有省力的标注办法**。
用户 10-01 brainstorming 定下：第一版认 **挥手、鞠躬、欢呼、害羞** 四个动作（加"都不是"共五类）；Claude 用命令行初分、**核对放进管理面板的「标注」页**；
模型用 **DINOv2-small 冻住 + 小时序头**，训练集 / 验证集**按录像切**。素材 10-01 晚录（清单 `docs/progress/2026-10-01-tonight.md`）。

## 1. 数据怎么流、放在哪

```
tmp/record/gesture-wave-1/ …（record --fps 8，一段只录一种动作）
   │  perception clips（改：片段名带录像名、半重叠、只切像运行时会判的人）
   ▼
datasets/gesture/_unlabeled/<录像名>__<序号>_track<轨迹>_t<开始秒>s/00.jpg … 15.jpg
   │  perception gesture-label（Claude 初分）→ 片段文件夹里多一个 claude.json
   ▼
管理面板「标注」页：看动图、按键确认 → 挪进 datasets/gesture/<动作>/<片段>/，不要的进 _discard/；每一步记 _labels.jsonl
   ▼
perception gesture-train → 特征缓存 _features/、切分 _split.json → models/gesture-<日期>.onnx + 报告
   ▼
perception gesture-eval（默认只评验证集）→ 精确率 ≥ 90%、召回率 ≥ 60% → 用户换上 models/gesture.onnx、打开 [gesture]
```

- 类别（`[gesture] labels`，顺序就是模型输出顺序）：`none, wave, bow, cheer, shy`；`names`：挥手 / 鞠躬 / 欢呼 / 害羞
- 数据目录 `[gesture] dataset`，默认 `datasets/gesture`（不进 git，同 `datasets/sky`）；`_` 开头的子目录不是类别

### 1.1 `perception clips` 的三处改动（`vision/gesture.py` `extract_clips`）

1. **片段名带录像名**：`<录像名>__<序号>_track<轨迹>_t<开始秒>s`（录像名 = 录像目录名，`__` 分隔）。按录像切分要用
2. **半重叠**：每条轨迹每 `stride` 帧（默认 8 = 1 秒）起一段新的 16 帧片段（现在是首尾相接、一个动作常被切成两半）；轨迹断了（同现在，`gap` 内没接上）重新攒
3. **只切像运行时会判的人**：框高够近 / 中（同 `eligible` 的远近判断，参照高按团子框 / `self_height` 估）、框中心在画面中间一半；不要求认出名字（录像里不跑 OCR）

## 2. Claude 初分（`perception gesture-label <片段目录>`，新 `vision/gesture_label.py`）

- **拼图**：16 帧按 4×4 排成一张图（每格 112×112，左上角标帧号 0~15，左到右、上到下是时间顺序）
- **一次 8 个片段**：每张拼图前一行文字标片段名；**录像名当提示**：录像名里有 `wave` / `bow` / `cheer` / `shy` / `none` 的，告诉 Claude "这段录像是好友在反复做 X（中间有停顿）"或"这段录像里没有这四个动作"——任务就成了"这一段是在做 X 还是在停顿"
- **提示词**：四个动作在光遇里的样子用文字写（挥手：一只手举起左右摆；鞠躬：上身前弯再起来；欢呼：双手往上举；害羞：捂脸 / 扭身）。**10-01 晚录完由用户看一眼动画、改描述**；提示词带版本号
- **回答**：JSON，每个片段 `{"label": "wave|bow|cheer|shy|none|unsure", "confidence": 0~1, "reason": "一句话"}`；`unsure` = 看不清、被挡、动作只露一半
- **存**：片段文件夹里 `claude.json`（加上模型名、提示词版本、时间）；解析不了 / 片段名对不上的那批不写、下次重跑
- **复用辅助标注框架**（`vision/assist.py` 的 `Reviewer` + 新的 `Protocol`，同物品模式）：分批并发、按批缓存、额度用完下次接着跑；默认 Sonnet（`[assist]` 的模型和令牌，同大脑）
- 已有 `claude.json` 的跳过，`--recheck` 才重猜；**已经确认过（不在 `_unlabeled/`）的片段永远不动**
- 量：今晚约 15 段 × 2 分钟 → 约 1500~2000 个片段 → 两百多次调用，每张拼图几百 token

## 3. 管理面板「标注」页

- 左栏导航新开一组「数据」，下面「标注」（`#labeling`）；新 `console/static/labeling.js` + `<section id="page-labeling">`，沿用改版后的三栏工作台
- **左栏**片段列表：筛选「待确认」（默认；`unsure` 最前，其余按 Claude 的 `confidence` 从低到高）/「Claude 猜的某一类」/「已确认的某一类」/「丢弃的」；顶上各类计数
- **中栏**播放：16 帧放大 3 倍，按 `[gesture] fps`（8）循环；暂停、逐帧、0.5 倍慢放；预加载后几段
- **右栏**：Claude 的猜测、把握、理由（**纯文本显示**，模型写的不当 HTML）、录像名、按钮
- **键盘**：`Enter` 同意 Claude；`1`~`5` 挥手 / 鞠躬 / 欢呼 / 害羞 / 都不是；`0` 不要；`Z` 撤销；空格 暂停。选完自动跳下一段；已确认的重新按键就改类别
- **接口**（`console/server.py`，新 `console/labeling.py` 放逻辑；沿用 `is_local_host` / `post_guard`）：
  - `GET /api/gesture/state` → 类别、各类计数、所有片段（片段名、录像名、状态 = 待确认 / 某一类 / 丢弃、Claude 的猜测）
  - `GET /api/gesture/frame?clip=<片段名>&i=<0~15>` → jpg。片段名必须是数据目录里**实际存在**的片段文件夹名（不拼路径、不认 `..` / 斜杠），帧号只能 0~15
  - `POST /api/gesture/label` `{clip, label}`（`label` ∈ 类别 ∪ `discard`）→ 挪文件夹，追加 `_labels.jsonl` 一行 `{t, clip, from, to}`；目标已存在同名就拒绝
  - `POST /api/gesture/undo` → 按 `_labels.jsonl` 最后一条挪回去并记一条撤销
- 和团子运行无关：团子在跑也能标，不占子进程；数据目录不存在时页面提示先跑 `perception clips`

## 4. 训练和导出（`perception gesture-train <数据目录>`，新 `vision/gesture_train.py`）

- **特征**：每帧 112×112（`[gesture] size`）RGB、ImageNet 归一化，过 DINOv2-small（`facebook/dinov2-small`，PyTorch，有 CUDA 用显卡），
  取 CLS + 各块特征平均拼成 768 维；一段 = 16×768。缓存 `_features/<片段>.npy`（键带模型名、尺寸、是否镜像）
- **增强**：每段再算一份左右镜像的特征
- **时序头**：Conv1d(768→128, k3) → ReLU → Conv1d(128→128, k3) → ReLU → 时间上 max + mean 拼接 → Linear(→5)；交叉熵按类别数量加权；AdamW；按验证集宏平均 F1 早停
- **切分**（`_split.json`）：按录像名，每一类挑约 20% 的录像进验证集、保证每类至少一段；某类只有一段录像 → 那段按开始秒数前 80% 训练、后 20% 验证，中间空 2 秒（重叠的片段不跨两边）
- **导出**：一个 ONNX 包含 归一化 + DINOv2 + 时序头：输入 `1×16×3×112×112`（RGB 0~1），输出 `1×5` 分数——**对上 `OnnxGestureClassifier` 的现有约定**，运行时不改；
  导出后拿 ONNX 和 PyTorch 在验证集几段上比输出，再跑一遍验证集评估
- **输出**：`models/gesture-<日期>.onnx`（**不覆盖** `models/gesture.onnx`）；报告 `tmp/gesture-train/<时间>/report.md`：每类数量、切分、训练曲线要点、每类精确率 / 召回率（按 `min_prob`）、认错的片段名
- `perception gesture-eval`：有 `_split.json` 时默认只评验证集（`--all` 评全部）
- **运行时开销**：每 2 秒对一个好友判一次；16 帧 112 在 CPU 上约一百多毫秒、显卡上不到 10 毫秒，不强求 onnxruntime-gpu
- 依赖：`transformers`、`huggingface_hub` 已在 `.pydeps`（10-01 装的，DINOv2 也已下载到 `.cache/huggingface`）

### 出错时

- 没 CUDA：退回 CPU 训练，提示会慢
- 某一类确认过的片段 < 20：拒绝训练，说清缺哪类、还差多少
- 某一类没有能进验证集的录像：警告（报告里也写）

## 5. 配置改动（`[gesture]`）

| 键 | 原来 | 改成 |
|---|---|---|
| `labels` | `none, wave, bow` | `none, wave, bow, cheer, shy` |
| `names` | 挥手、鞠躬 | 挥手、鞠躬、欢呼、害羞 |
| `dataset` | — | `"datasets/gesture"` |
| `stride` | — | `8`（切片段每几帧起一段） |

`enabled` 照旧默认关；`min_prob` 0.9、`cooldown` 30 秒不变。

## 6. 测试

单元测试（`python -m pytest -q`，不用真模型、不用真 Claude）：
- 切片段：片段名带录像名、半重叠（stride）、只切近处画面中间的人（假检测器）
- 拼图尺寸和帧号；Claude 回答解析（`unsure`、格式乱、片段名对不上）；跳过已猜过的 / 已确认的；录像名提示
- 标注接口：确认、改类别、丢弃、撤销；不存在的片段、`..` / 斜杠、帧号越界、目标已存在都拒绝；`_labels.jsonl` 记录
- 切分：按录像、只有一段录像时按时间切且中间空 2 秒、每类都有验证数据
- 训练流程：用假的小特征提取器代替 DINOv2，跑通 训练 → 导出 → ONNX 和 PyTorch 输出一致 → 评估，检查形状和标签顺序
- `gesture-eval` 有 `_split.json` 时只评验证集

真机 / 真数据（10-02 白天，用 10-01 晚的录像）：
1. `perception clips` 切完看数量、抽几段看是不是完整动作
2. `gesture-label` 跑完，网页上过一遍，记下 Claude 的同意率
3. `gesture-train`，看报告：验证集精确率 ≥ 90%、召回率 ≥ 60%
4. 达标后换上 `models/gesture.onnx`、`[gesture] enabled = true`，`view` 里看好友做动作时有没有 `gesture` 事件（晚上）

## 7. 改动范围

| 位置 | 改什么 |
|---|---|
| `vision/gesture.py` | `extract_clips` 三处改动；`evaluate` 支持只评验证集 |
| `vision/gesture_label.py`（新） | 拼图、提示词、解析、`Protocol`（复用 `assist.Reviewer`） |
| `vision/gesture_train.py`（新） | 特征提取 + 缓存、切分、时序头、训练、导出、报告 |
| `console/labeling.py`（新）、`console/server.py`、`console/static/labeling.js`（新）、`console.html`、`console.css` | 「标注」页和接口 |
| `config.py`、`config.example.toml` | `[gesture]` 的 `labels` / `names` / `dataset` / `stride` |
| `cli.py` | `perception gesture-label`、`perception gesture-train`；`clips` / `gesture-eval` 的新参数 |
| `CLAUDE.md` | 代码结构表、常用命令、YOLO 感知层一节里动作识别的说明 |
