# 2026-09-28 晚：YOLO 感知层第一次真训练

接手的人先看这份，再看 `docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md`（总纲）和
`2026-09-28-assist-labeling-design.md`（辅助标注）。

## 一句话

在本机 RTX 5070 Ti Laptop 上装好 GPU 环境，录了 4 段、标了 311 帧（3 张地图），训到第四版模型 `models/sky-yolo-v4.pt`，
认人 / 团子 / 黑影 / 共享空间玩家 / 先祖已经比较靠谱；**还没在 `run` 里打开感知层**。

## 环境（本机）

- torch 2.11.0+cu128、ultralytics 8.4.164（`pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu128`，再 `pip install --user -e ".[yolo]"`）
- 官方模型 `models/yolo11n.pt`、`models/yolo11x.pt`（辅助标注的候选框用）
- 测速（实时截图 1920×1080、推理 960、yolo11n）：截图 20 ms + 检测 22 ms + 整个感知 28 ms ≈ 48 ms/帧，15 fps 够用
- 训练脚本在 `tmp/yolo/train_v*.py`：`YOLO("models/yolo11n.pt").train(data="datasets/sky/data.yaml", imgsz=960, epochs=120, batch=16, workers=2, project=<绝对路径>, …)`
  - **坑**：`project` 写相对路径时 ultralytics 会存进 `runs/detect/<project>/`，而 `runs/` 是 Agent 运行目录（会被清理）→ 用绝对路径
  - 改了标注后删 `datasets/sky/labels/*.cache` 再训
- 一轮 120 epoch：164 帧约 10 分钟，311 帧约 15~20 分钟

## 数据集 `datasets/sky/`（不进 git）

| 来源 | 帧数 | 说明 |
|---|---|---|
| `tmp/record/20260928-200526`（石墙地图 + 洞穴） | 74 | 240 帧挑 74；子代理（Sonnet）核对，手动改过共享空间玩家 |
| `tmp/record/20260928-214948`（同上，好友发请求） | 88 | 360 帧挑 89，`--assist --model v1`，1 帧核对失败已挪到 `_unpicked/` |
| `tmp/record/20260928-220217`（黑斗篷、躺着的人） | 2 | 用户挑的；手动补了两个躺着的黑影和眼睛圆圈 |
| `tmp/record/20260928-224124`（花地图夜景，先祖多） | 147 | 360 帧挑 147，`--assist --model v3`，15 路并发；手动改了 3 处 |

- 训练 254 / 验证 57；没挑中的帧在 `_unpicked/`
- 待核对清单：`datasets/sky/_assist/review_<录像>.md`（第一批的在 `tmp/assist2/review.md`）；**用户没有人工核对过**（不会用 X-AnyLabeling）
- 花地图那批的标注估计有一两成错（先祖 / 玩家分不清、人挤在一起），其余两批约一成
- 补陌生人圆圈之前的标注备份：`tmp/labels_backup_v2/`

### 标注规则（用户确认，也写在辅助标注提示词里，版本 3）

- 头顶有圆圈的人形一律是人（不是 `not_person`）；是 `player` 还是 `player_unlit` **只看身体是不是纯黑**
- 发蓝光、半透明、**个子和玩家一样矮小**的是**其他共享空间里的玩家**（`player`），图标时有时无也算；头顶深色实心圆 + 飞人图标 = 共享空间入口，绝不能点
- **先祖**：身体修长（接近成年人比例）、偏透明，常摆固定姿势（花地图的红袍白帽）；头顶深色实心圆 + 三颗星不算玩家圆圈 → `not_person`
- 团子：白发、橙色护目镜、橙粉色袍子、蓝紫色圆背包、头顶没名字；每帧最多一个
- 变身成动物的玩家（白鹿，头顶有共享空间图标）**没标**：太少见、形状不像人

## 模型（`models/`，不进 git）

| 版本 | 数据 | 效果 / 问题 |
|---|---|---|
| v1 `sky-yolo-v1.pt` | 74 帧 | 认人、团子基本行；**一个圆圈都框不出**；黑斗篷玩家 → 黑影；躺着的人 → 团子 |
| v2 | 164 帧 | 黑斗篷、躺着的黑影对了；换到花地图：先祖 / 花草 → 陌生人，团子 → 陌生人 |
| v3 | 164 帧 + 陌生人圆圈（`ring_labels` 补了 65 个） | 陌生人头顶的 ✦ 圆圈能框出来 |
| **v4** `sky-yolo-v4.pt` | 311 帧（加花地图） | 花地图上先祖不再认成人、团子认对、模糊画面也认得出；验证集 mAP50 0.82（验证集变难了，和前几版不可比） |

和整图 OCR 对比（v1，166 帧）：认好友同帧结果一致，每帧 49 ms vs 每次扫描 875 ms。

看效果：`python -m skydango view --model models/sky-yolo-v4.pt`（只看不动）。

## 今晚合并进 main 的代码

- `fix(chatlog)`：聊天输入框变暗（面板开着一会儿后）也认得出 —— 之前弱标注把面板里的好友名当成头顶名字
- `feat(assist)`：`perception label --assist`（挑帧 → 候选框 → 隔离的 `claude -p` 核对 → 合并标注 + 预览 + 待核对清单）；
  审查后修了：子进程读到项目 CLAUDE.md（每批多 1 万 token）、JSON 提取不稳、全部原图压内存
- `fix(perception)`：一帧只留一个团子；一个名字标签只分给对得最正的那个人
- `feat(social)`：新图标模板 火焰（`candle`，**陌生人举着蜡烛走到团子旁边要点火**）、陌生人平时（`stranger`）、眼睛（`eye`，在看留影 / 听音乐）、共享空间（`shared`）；后三种只是状态（`PASSIVE`），松手判断也认它们
- `feat(label)`：`ring_labels` 按人物框在头顶附近用图标模板补 `social_ring`（陌生人的也标）

本机 `config.toml` 加了 `[assist] jobs = 15`、`batch = 3`（15 路并发，147 帧约 8 分钟）。

## 用量参考（订阅额度）

辅助标注每帧约 3k 输入 + 1.8k 输出 token（修掉 CLAUDE.md 加载之后）；89 帧一批约 27 万输入 / 16 万输出，147 帧约 48 万 / 26 万。

## 物品识别（v5 起，spec `docs/superpowers/specs/2026-09-29-object-recognition-design.md`）

类别追加了 `bench`（座位）/ `bonfire`（篝火）/ `instrument`（乐器）/ `spirit`（先祖），编号 6~9，旧的 0~5 不变。代码已完成，**还没有数据和模型**。

头顶气泡 `typing`（编号 5，一直在类别里但没有标注）**也在这一轮一起标**：物品模式的提示词里多了 typing（`..` / `...` 正在输入和发出去的一行字都算、只框气泡、团子头顶的也标），
写回时和物品行一样整体替换。typing 不是物品：感知层的 `objects()` 不会把它报成"画面里的东西"。
只想补某些帧（比如专门录的气泡片段）用 `--only`：`perception label datasets/sky --objects --only 'rec_0929_*'`
（fnmatch，按图片文件名匹配，带不带 `.jpg` 都行；可以给多次或逗号分隔；别的帧不核对、不写回、不进清单，`labels/` 照样整个备份）。

操作顺序：
1. 录像：专门去有长椅、篝火、钢琴、先祖的地方各录一段（每类至少 50 个框、2~3 张地图）；花地图（先祖多）已有录像
2. 新录像先照旧 `perception label <录像> --assist` 标人
3. 对整个数据集补物品：`perception label datasets/sky --objects`（第一轮没有物品模型，Claude 按网格直接框；v5 训出来后加 `--model models/sky-yolo-v5.pt` 当候选）。
   写回前自动备份 `labels/` 到 `_backup/`；人物行除"Claude 判成先祖的"外不动。重跑只核对没做过的帧（额度用完后接着做）；
   上次做过之后**你在 X-AnyLabeling 里改过的帧会跳过**（人工优先），要让 Claude 重新核对加 `--recheck`（已有的物品框当"已标"候选，Claude 没判的留着，删 / 改的列进清单）。
   增强图（`_blur` / `_dark`）不核对：**`perception augment` 放在补完物品之后**，已经增强过的先删掉增强图再重新 augment。311 帧约 100 万输入 / 55 万输出 token
4. 看 `datasets/sky/_assist/objects.md`（先看"人物框改成了先祖"），用 X-AnyLabeling 修正
5. 训练 v5（10 类）
6. **上线门槛**：同一批录像 v4、v5 各跑一次 `perception compare`，好友认出率、远处认出率不低于 v4（花地图上陌生人误报应减少）；
   训练输出里每类物品验证集 mAP50 ≥ 0.6；`view --model models/sky-yolo-v5.pt` 真机看一圈没有明显误认。都满足再把 `[perception] model` 换成 v5

## 没做完 / 待办（按建议顺序）

1. **用 v4 在 `run` 里试感知层**：`config.toml` 加 `[perception] enabled = true`、`model = "models/sky-yolo-v4.pt"`、`device = "cuda"`，
   先 dry-run 跑 5 分钟，看 arrive / leave / stranger 事件准不准（v1 时模拟出 leave 误报两次）；所有阈值都还没在真机标定
2. 用户边玩边用 `view --model models/sky-yolo-v4.pt` 找认错的场景，再录对应地图补数据（新地图认不准时就是缺数据）
3. **呼唤 + 小脑**（讨论结论，还没设计）：不训练小脑模型，继续"大脑（LLM）定方针、身体（规则）照方针立刻执行"。
   缺：Q 键呼唤（先真机实测按 Q 会发生什么）、"看不见好友"事件、心情 / 请求方针写进记忆（重启不丢）。要走 brainstorming
4. **聊天气泡**（待办）：`typing` 类还没有标注（物品模式已经能标，见上面「物品识别」）；设想平时关聊天栏，看到好友气泡或每 20~30 秒打开一次。前提要真机确认关面板时消息是否仍记录
5. 单独的任务（已开任务卡片）：大脑 / 眼睛的 `claude -p` 很可能也读到项目 CLAUDE.md（同辅助标注的问题）；`test_viewer.py::test_port_in_use_raises` 在 Windows 上偶发失败
6. 辅助标注审查留下的小问题（没修）：`boxes` 返回成列表时整帧当"不是人"并写进缓存；坐标 NaN 让命令崩；额度用完后排队批次仍起进程；
   帧名带 `.jpg` / "帧 " 前缀时整批丢；会话记录堆在 `.brain-claude/projects/`；提示词写死 1920×1080；`_perception_label` 里两个没用的导入
7. 未跟踪、没动的文件：`emotes/`（用户的图标库，没提交也没 gitignore，要问用户）、`.claude/launch.json`、`docs/superpowers/plans/2026-09-27-brain-move-owner.md`
