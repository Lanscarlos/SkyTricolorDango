# Claude 辅助标注（`perception label --assist`）设计

日期：2026-09-28　状态：设计已和用户逐段确认，待实现

## 1. 为什么做

YOLO 感知层一期要先有训练数据。现在的弱标注（`perception label`）只能自动标 `name_tag` 和 `social_ring`，
`player` / `player_unlit` / `self` 要人工在 X-AnyLabeling 里**每张都画**，用户觉得太累。

2026-09-28 的试验（`tmp/assist/`，20 张）：
- 官方 COCO 模型（yolo11n + yolo11x 合并，person 类，阈值 0.1，推理尺寸 1280）能框出大部分亮着的小人，
  但漏黑影、误检宠物 / 先祖 / 特效，人挤在一起时重复框很多
- 把编号候选框 + 坐标网格画在图上交给 Sonnet 核对：分类约九成正确，会去重、补漏框（位置大致对、偏松）、放大只框了头的框；
  主要错误是团子和别人挨在一起时认错团子，以及"带圆圈的蓝色飞行人形"算不算玩家（已由用户定规则，见 §4）
- 成本：20 张约 37 万 token（子代理，含读图工具的往返）

目标：**人工从"每张画四五个框"变成"看一眼、改一成"**；拿不准的帧单独列出来优先看。

## 2. 命令

```bash
python -m skydango perception label <录像目录> --assist [--model 模型] [--all-frames]
```

在现有 `perception label` 上加选项；不加 `--assist` 时行为完全不变。

- 加 `--assist` 时，`--model` 的预测只当候选框交给 Claude 核对，不再像现在这样直接合并进标注（`merge_labels`）
- 可以和 `--from-runs`（难例）一起用；不能和 `--spin` 一起用（转圈录像已经能自动补 `self`，直接报错提示二选一）

## 3. 流程

### 3.1 挑帧（加 `--assist` 时默认开，`--all-frames` 关）

录像每秒 2 帧，相邻帧很像。按时间顺序走一遍，每帧缩成 64×36 灰度图，和**上一张留下的帧**比平均绝对差：
大于 `min_change` 就留；离上一张留下的已经 `max_gap` 帧（默认 20 帧 = 10 秒）也强制留，免得慢慢走近的过程被漏掉。

标定：这次录像（镜头一直在动，相邻帧差中位 17.5）`min_change = 30` → 240 张留 74 张。

只有挑中的帧跑 Claude、写进数据集。

### 3.2 候选框

- 没给 `--model`：`[assist] proposal_models`（默认 `models/yolo11n.pt`、`models/yolo11x.pt`）的 COCO person 类，
  阈值 `proposal_conf = 0.1`、推理尺寸 `proposal_imgsz = 1280`，合并后 NMS（IoU 0.6）去重
- 给了 `--model`（自己训的模型）：用它的 `player` / `player_unlit` / `self` 当候选（阈值 `perception.low_conf`）
- `name_tag` / `social_ring` 照旧走现有弱标注（OCR + 圆圈模板），不交给 Claude

### 3.3 Claude 核对

- **给它看的**：每帧一张标注图 —— 原图 + 坐标网格（每 100 px，数字是原图坐标）+ 编号候选框（框线细、编号在框外），
  文字里附每个框的精确坐标。不另发原图
- **怎么发**：每 `batch = 5` 帧一条消息，起一次 `claude -p --model sonnet`（`[assist] model`），`--tools ""` 关掉内置工具
  （图片直接放在消息里，编码复用 `brain/images.py`）；进程隔离沿用大脑的做法（`brain/claude.py` 的 `claude_env` /
  `resolve_claude` / `one_shot`，配置目录和 `SKYDANGO_CLAUDE_TOKEN` 同 `[brain]`）。默认 `jobs = 3` 个进程并发
- **回答格式**（只要 JSON）：
  ```json
  {"<帧名>": {"boxes": {"1": {"cls": "self", "note": "", "fixed_box": null}},
              "missing": [{"cls": "player", "box": [x1, y1, x2, y2], "note": ""}],
              "unsure": ""}}
  ```
  `cls` ∈ `self` / `player` / `player_unlit` / `not_person` / `duplicate`；`missing` 的 `cls` 只能是前三个

### 3.4 缓存和出错

- 每帧的候选框和核对结果存 `<数据集>/_assist/<帧名>.json`；重跑跳过已核对的帧（换了候选来源或提示词版本就重新核对：缓存里记 `prompt_version` 和候选来源）
- 一批 `timeout = 300` 秒没结果重试一次；还不行这几帧记为"没核对"（只写弱标注、列进清单）
- 回答解析不了 / 缺帧：那几帧按"没核对"处理；个别框编号缺了按 `not_person` 处理并在清单里提一句；类别写错的框同样处理
- 坐标越界的框裁到画面内，裁完为空的丢掉
- 订阅额度用完（`ClaudeError.limit`）：停下，已核对的保留，打印"额度恢复后重跑同一条命令会接着做"
- 没设 `SKYDANGO_CLAUDE_TOKEN` / 找不到 `claude`：开始前就报错退出，不跑检测

## 4. 提示词里的规则

- 头顶有圆圈（✦、动作小图标、火焰、眼睛等）的人形**一律是玩家**，哪怕通体发蓝光、半透明、在空中飞（用户 2026-09-28 确认）；
  **深色实心圆里画着飞人 + 小人图标的也算**：那是在其他共享空间里的玩家，游戏把他们显示成发蓝光的半透明人形，
  图标是"加入共享空间"的入口（样子和白色描边空心圈不同，子代理曾因此误判成先祖，用户确认是玩家）；
  没有圆圈、半透明发蓝光、摆固定姿势的是先祖（`not_person`）
- `player_unlit`：整个人纯黑的剪影；穿黑斗篷但身上有亮色花纹、脸是亮的算 `player`
- `not_person`：宠物 / 小动物、先祖、石像、灯笼、特效、UI
- `duplicate`：同一个人只留最贴合的框
- 每张图最多一个 `self`；团子长相来自 `[assist] self_hint`（默认"白色头发、橙色护目镜、橙粉色袍子、背蓝紫色圆背包、头顶没有名字标签"，换装后改这一项）；
  候选框同时框住团子和别人时不要整框判成 `self`，要拆开（`fixed_box` + `missing`）；没有背包的白发角色不是团子
- 漏掉的人：用网格估框，框住整个身体、不含头顶名字 / 圆圈；太远太小（身高 < 约 25 px）的不列；被 UI 挡住一部分但看得出是人的要列

提示词带版本号 `PROMPT_VERSION`，改规则时加一，缓存按它失效。

## 5. 输出

- **YOLO 标注**：弱标注（`name_tag`、`social_ring`）+ Claude 认可的人物框（有 `fixed_box` 用修正后的）+ Claude 补的框，
  写进 `<数据集>/labels/`，格式同现在；只有挑中的帧写进数据集
- **预览** `_preview/`：绿 `player`、紫 `player_unlit`、灰白 `self`、红细框 = 去掉的候选、虚线 = Claude 补的框；名字标签 / 圆圈照旧
- **待核对清单** `_assist/review.md`，按优先级：
  1. 核对失败的帧（人物框要全补）
  2. 有 Claude 补框的帧（偏松，要拉紧），附它的说明
  3. 它写了 `unsure` 的帧，附原话；以及回答里缺编号 / 类别写错的框
  4. 其余帧的数量，建议抽查几张
- **终端汇总**：挑了多少帧、每类标了多少、补了多少、失败多少、用量（input / output token，参考，订阅不按它计费）

## 6. 代码结构

| 位置 | 内容 |
|---|---|
| `vision/assist.py`（新） | `pick_frames`（挑帧）、`propose_people`（候选框，延迟导入 ultralytics）、`draw_candidates`（网格 + 编号）、`build_prompt`、`parse_review`（解析 + 校验）、`apply_review`（合并成标注框）、`review_report`（清单）、`Reviewer`（分批、并发、缓存、重试；真正调 Claude 的函数从外面传进来） |
| `brain/claude.py` | 复用 `one_shot` / `claude_env` / `resolve_claude`，不新写进程管理 |
| `config.py` | 新增 `AssistConfig`（`[assist]`）：`min_change`、`max_gap`、`proposal_models`、`proposal_conf`、`proposal_imgsz`、`batch`、`jobs`、`timeout`、`model`、`self_hint` |
| `cli.py` | `perception label` 加 `--assist`、`--all-frames`，只做接线（文件已 1400 多行，逻辑不往里放） |
| `config.example.toml` | 加 `[assist]` 一节 |

## 7. 测试

`python -m pytest -q`，不连 Claude、不要模拟器、不要 GPU：
- 挑帧：合成画面（变化小的连着几帧只留一张、超过 `max_gap` 强制留）
- 解析：带 ```json 代码块、前后有废话、缺帧、缺编号、类别写错、坐标越界 / 反了
- 合并：`duplicate` / `not_person` 丢掉、`fixed_box` 替换、`missing` 加入、弱标注保留
- `Reviewer`：假的调用函数 —— 分批、缓存命中跳过、超时重试一次后记失败、额度用完抛出并保留已完成的
- 清单：各优先级分组

实测：拿 `tmp/record/20260928-200526` 跑一遍，对照 `tmp/assist2/` 的子代理结果看是否一致；预览和清单给用户看。

## 8. 文档

CLAUDE.md 常用命令、感知层总纲 §13 第 4~5 步（弱标注 → 辅助标注 → 人工核对）。

## 9. 不做的（YAGNI）

- 不让 Claude 标 `name_tag` / `social_ring` / `typing`
- 不做 X-AnyLabeling 插件 / 自动打开
- 不做多轮对话式修正（一帧只问一次；要重判就删缓存重跑）
