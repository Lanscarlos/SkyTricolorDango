# 难例收件箱：live 难例 → 外形头先筛 → 整帧核对 → 一键重训对比（设计）

2026-10-04。起因：10-04 晚真机验证 YOLO v10 + 外形头（`[attrs]`）后，用户问"每次 live 能不能自动把这些情况收起来、让外形头先判、再离线反过来训 YOLO"。

## 0. 目标和已经定下的事

**目标**：每次 live 存下的难例不再丢（`runs/` 只留最近 20 次，`hard/` 跟着被删），尽量少花人手变成 YOLO 能用的标注；攒够了一键重训、和旧模型对比，换不换由用户定。

**真机看到的、这条流水线要接住的情况**（10-04 晚 `runs/20261004-192711-live-brain`）：
- 外形头撤对了空椅子上的框、纠正了 YOLO 把黑影当点过火、放行了被挡一半的低分小人；
- 但 YOLO 把**篝火和坐在后面的懒洋洋大王框成一个框**，外形头判"不是人"0.72 撤掉，团子看不见他——框本身画错，外形头修不了，要人画框；
- 草地上的假框外形头只判了 0.69（门槛 0.7）没撤：门槛分不开，根子在 YOLO 的框。

**和用户定下的**（brainstorming，按问答顺序）：
1. 团子下线后自动整理；但下线常常仓促，**点停止时先问"现在整理吗"**，可以"下次再说"；不管选哪个，下线时先把 `hard/` 复制进不会被清理的收件箱
2. 框画错的帧（篝火 + 人、漏框）在**标注页里做整帧编辑**（拖框、删框、改类别）
3. **所有帧进训练集前都要人看过整张图**：全自动一致的帧也给人快速过目（回车一秒一张），有分歧的先逐张判裁图
4. 攒够了**一键训练 + 对比**（YOLO 和外形头一起），**换不换模型由用户点**
5. 数据放**单独的待核对区** `datasets/inbox/`，整帧通过才复制进 `datasets/sky`

**不做**：自动训练、自动换模型；外部标注工具；动作片段（gesture）进这条流水线；两台电脑同步收件箱（真机 live 只在个人电脑跑，`datasets/` 不进 git）。

## 1. 整体流程

```
run 下线 ──① 收（一两秒）──> datasets/inbox/<运行>/raw/      状态 collected
                                   │ ② 整理（面板任务 / 命令，几分钟，用显卡）
                                   ▼
               去重 → 预标注 → 人物框裁图 + 外形头 → 每个框分流 → 帧状态
                                   │
             ┌─────────────────────┴───────────────────┐
     有框等人判（crops）                       全部定了（glance）
     标注页「外形」逐张判裁图 ──判完──>       标注页「整帧」过目
                                     回车通过 / E 编辑 / 0 不要
                                   │ ③ 通过
                                   ▼
               datasets/sky/images|labels/<train|val>/   状态 done
                                   │ ④ 攒够 → 「重训」（面板任务）
                                   ▼
          新 YOLO + 新外形头 → 回放对比报告 → 用户点「换上」/「回退」
```

## 2. 数据

### 2.1 目录

```
datasets/inbox/
  _index.jsonl              每次运行一行：run、collected_at、processed_at、帧数、各状态计数（整理 / 过目时更新）
  _stats.json               上次整理每帧平均耗时（估时间用）、上次训练用到的帧数和时间
  <运行>/
    raw/<文件>.jpg          ① 从 runs/<运行>/hard/ 复制来的原图
    hard.jsonl              ① 原样复制（当初为什么存、当时的框）
    frames.json             ② 起：每帧一条，见 2.2
    labels/<帧>.txt         ② 预标注（YOLO 格式，类别编号同 datasets/sky）；编辑后改这里
```

帧名 = `<运行>_<原文件名去后缀>`（同 `weaklabel.hard_images`），进 `datasets/sky` 也用这个名字。

### 2.2 `frames.json`

```json
{"<帧名>": {
   "file": "raw/192007_attrs_disagree.jpg",
   "reason": "attrs_disagree",          // hard.jsonl 里的原因（过目时显示）
   "split": "train",                    // 按运行目录名哈希：crc32(run) % 5 == 0 → val，整次运行同一边
   "dup_of": null,                      // 去重：和哪一帧几乎一样（被去掉的帧只记这一项）
   "boxes": [{"cls": 4, "box": [x, y, w, h], "score": 0.56, "src": "yolo",
              "crop": "<裁图名>|null", "auto": "agree|drop_low|null"}],
   "editing": false,                     // 人按了 E、还没保存
   "error": null,                        // 整理这一帧出错的原因（跳过）
   "decision": null                      // 人在整帧页做的：{"what": "pass|discard", "edited": true|false, "t": …, "dataset": "train/<帧名>"}
}}
```

**帧状态不存，现算**（标注页每次读时算，避免外形页改了裁图还要回头改帧）：
- `decision` 是 pass → `done`；discard → `discarded`；
- 否则有裁图还在 `_unlabeled`（没人判）→ `crops`；
- 否则有裁图被判成「不要」（看不清，挪进了 `datasets/attrs/_discard/`）或人在整帧页按了 E 没保存（`frames.json` 记 `editing: true`）→ `edit`；
- 否则 → `glance`。

裁图判成什么，读 `datasets/attrs` 现有的记账（`attrs_data.hand_labels` + 裁图现在在哪个 `form/<类>/`）。

### 2.3 裁图

整理时人物框的裁图走现有的 `attrs_data._Writer`，写进 `datasets/attrs`，`_crops.jsonl` 那一行 `source = "inbox"`、`image` = 收件箱原图、`group` = 运行目录名。
- 自动一致的 → 直接放进 `form/<类>/`，`_labels.jsonl` 记一条 `by = "auto-agree"`（同 10-04 的做法，外形头训练默认就用它们）
- 其余 → `_unlabeled/`，`claude.json` 里写猜测（`model = "attrs-screen"`），理由栏「YOLO 判黑影 0.56，外形头判点过火 0.92」
- `writeback` 只处理 `source = "dataset"` 的行，`source = "inbox"` 的**不会被写回**（加测试钉住）；它们的结果在整帧通过时写进新帧的标注

## 3. ① 收（`perception inbox collect`；`run` 下线时自动）

- `cli._stop_scene` 停掉感知线程之后：`hard.saved > 0` 就把这次运行的 `hard/*.jpg`、`hard.jsonl` 复制到 `datasets/inbox/<运行>/`（已经在了跳过），`_index.jsonl` 记一行；`[inbox] collect = false` 或出错只记 WARNING，**不影响下线**
- 面板起的、终端起的、`--no-brain` 都走 `_stop_scene`，都会收；被强杀的收不到，由 ② 开头补
- `perception inbox collect [runs]`：手动补收所有还在的 `runs/*/hard/`（幂等）

## 4. ② 整理（`perception inbox process`）

先做一遍 collect（补被强杀的），然后按运行目录时间顺序处理所有还没整理完的运行；每处理完一帧就写 `frames.json`（中途被杀下次跳过做过的帧）。

1. **去重**：同一运行里按文件名时间排序，和前一张保留帧比 1/8 灰度缩略图的平均像素差 < `[inbox] dup_diff`（默认 6）且时间相隔 ≤ `dup_gap`（5 秒）→ 记 `dup_of`，后面不处理
2. **预标注**：复用 `_perception_label` 的单帧部分（YOLO 按 `low_conf` 出框、名字标签 OCR、圆圈图标），抽成一个函数给两边用；`--model` = 当前 `[perception] model`
3. **人物框**：`player` / `player_unlit` / `spirit` 按运行时的 `merge_people` 合并重复框，每个裁图交外形头（当前 `[attrs] model`，主干同运行时；裁图框外填灰同 `attrs.CROP_KEEP`）
4. **每个人物框分流**（`p` = 外形头最大类的把握，YOLO 类别和外形类别对应 player↔lit、player_unlit↔unlit、spirit↔spirit；shared / morph 并进 lit）：

| YOLO 分数 | 外形头 | 结果 |
|---|---|---|
| ≥ `conf` | 类别一致且 p ≥ `agree`（0.9） | 自动确认（`auto = agree`），标注用 YOLO 的框和类别 |
| ≥ `conf` | 判"不是人"（任何把握） | 给人判（篝火那种不能自动删） |
| ≥ `conf` | 其他（不一致、没把握） | 给人判 |
| `low_conf` ~ `conf` | 类别一致且 p ≥ `agree` | 自动确认，补成正式框 |
| `low_conf` ~ `conf` | 判"不是人"且 p ≥ `agree` | 自动丢（`auto = drop_low`）：本来 YOLO 就不报它；过目时用淡虚线画出来，人能看到丢了什么 |
| `low_conf` ~ `conf` | 其他 | 给人判 |

5. 别的类别（名字标签、圆圈、团子、气泡、物品）只用 YOLO ≥ `conf` 的预标注，不过外形头，靠人过目
6. 结束时更新 `_index.jsonl`、`_stats.json`（每帧平均耗时），打印「整理完 3 次运行：去重掉 40、自动 210 个框、给你判 18 张裁图，32 帧等过目」

`perception inbox status`：每次运行各状态多少帧、外形页还有多少张来自整理的裁图、上次训练以来通过了多少帧。

## 5. ③ 标注页

### 5.1「外形」页
- 筛选多一项「来自整理」（`source = inbox`）；猜测栏的理由照 2.3
- 其余不变：判完一帧的最后一张裁图，这一帧在整帧页自动变成 `glance`（状态现算，见 2.2）

### 5.2「整帧」页（新标签页，后端 `console/frames.py`，接口 `/api/frames/*`；前端 `static/frames.js`）

**过目模式**（默认）：
- 左边帧列表，筛选：运行 / 状态（待过目、要编辑、等判裁图、已通过、不要了）；顶上「待过目 23 · 要编辑 2 · 等判裁图 5 · 上次训练以来通过 58」
- 中间整张截图，所有类别的框：颜色按类别（同 `stage.js` 的配色），标签 = 类别 + 来源（自动 / 你判的 / YOLO 预标 / 你画的）；`drop_low` 的淡虚线；旁边一行「当初为什么存：外形头撤下」
- 键：回车 = 通过；E = 编辑；0 = 整帧不要；Z = 撤销上一步（撤销通过 = 把复制进 `datasets/sky` 的两个文件删掉、`decision` 清空）；← → 翻帧；H / 按住空格 = 隐藏所有框（找漏框的人）
- 通过 = 按最终框（自动的 + 外形页判过的：判成别的类就改类别、判"不是人"就去掉）写 `labels/<帧>.txt`，再把原图和标注复制进 `datasets/sky/images|labels/<split>/<帧名>.*`；目标已存在就拒绝（提示重名）

**编辑模式**：
- 空白处拖 = 画新框（当前类别）；点框 = 选中；拖框内 = 移动；拖边角 = 调大小；删除键 = 删选中的框
- 数字键选类别（作用于选中的框或下一个新框）：1 点过火的人 2 黑影 3 团子 4 先祖 5 名字标签 6 圆圈 7 气泡 8 座位 9 篝火 0 乐器；旁边一排可点的类别按钮；屏幕上一直标明"编辑中"（编辑里 0 是乐器，过目里 0 是不要）
- 滚轮以光标为中心放大，按住空格拖动画面
- 回车 = 保存并通过；Esc = 放弃修改回过目
- 保存时：新画 / 改过类别的人物框裁一张图记进 `datasets/attrs` 的 `form/<类>/`（`by = "frame-edit"`，算人确认过的）；这一帧不再走外形头

坐标换算、拖框 / 改大小、按键 → 类别这些纯计算放进 `frames.js` 里能被 node require 的函数（同 `common.js` 的做法），测试里直接测。

## 6. 面板：任务槽和提示

### 6.1 任务槽
- `console/server.py` 多一个 `Runner` 实例当任务槽（`kind = "job"`，`job` = `inbox` / `retrain`），子进程 `python -m skydango -c <config> perception inbox process` 或 `perception retrain`，日志进 `tmp/jobs/<时间>-<job>.log`
- **和团子 / 沙盒互斥**（都要显卡）：任务在跑时叫醒团子 → 页内对话框「整理还没完（剩 120 帧），先停下再叫醒？」（重训加一句"训练停了要从头来"），确认就停任务再叫醒；团子 / 沙盒在跑时不能开任务
- 停任务 = 按进程树结束（整理可续跑，训练不改任何配置）

### 6.2 整理的提示
- **点「停止」**（真机团子页、侧栏卡片的停止都算）：面板先查这次运行目录 `hard/` 有几张、收件箱里还有几次没整理 → `ask()`「这次存了 85 张难例（还有 2 次运行没整理），现在整理吗？约 4 分钟」：「整理」/「下次再说」/「取消」（取消 = 不停团子）；0 张且没有没整理的就不问
- 选「整理」：照常停团子，等它退出（`exited`）后开任务；团子被强杀也照样开（② 开头会补收）
- 接管的终端团子也问
- **侧栏运行卡片下**：任务中「整理素材：第 2/3 次运行 · 120/300 帧」（任务子进程每处理一帧往日志写一行进度，面板读最后一行）；完了「18 张裁图、32 帧等你看 →」（跳标注页）；有没整理的「3 次运行的素材没整理 [整理]」（面板启动时也显示）
- `[inbox] ask = false`：停止时不问（素材照样收进收件箱，侧栏照样提示）

## 7. ④ 一键重训和对比（`perception retrain`）

### 7.1 触发
- 标注页（整帧页顶上）「上次训练以来通过：58 帧」；≥ `[inbox] retrain_min`（50）时「重训」按钮亮起；不到也能点，先提示"只有 N 帧，可能看不出差别"
- 点了 → `ask()` 说明大概要多久、期间不能叫醒团子 → 开任务

### 7.2 做什么
1. **YOLO**：把 `tmp/yolo/train_v10.py` 的设置挪进代码（`[retrain]`：`base = "models/yolo11n.pt"`、`imgsz = 960`、`epochs = 120`、`batch = 16`、`workers = 2`），先把 `datasets/sky/labels/*.cache` 挪走（ultralytics 只按文件大小判缓存过期，v10 踩过），训练输出 `tmp/retrain/<时间>/yolo/`，最好的权重复制成 `models/sky-yolo-v<N>.pt`（N = `models/` 和 `tmp/yolo/sky-v*` 里最大的号 + 1，现在会是 v12，免得和没上线的 v11 混）
2. **外形头**：`attrs-train`（默认只用确认过的，含 auto-agree 和 frame-edit）→ `models/attrs-<日期><字母>.npz`
3. **对比**（`datasets/sky` 验证集、答案按 `gt_fixes` 修正，同 10-04 的回放）：
   - 整帧回放 `attrs_train.replay`：旧 YOLO、旧 YOLO + 旧外形头、新 YOLO、新 YOLO + 新外形头，各列精确率 / 召回 / 点没点火认反（门槛 `[perception] conf`、`low_conf`）
   - ultralytics val 各类 mAP50（旧 / 新）
   - 上面两项再单独算一遍"来自收件箱的验证帧"
   - 固定提醒：新旧用的是同一份答案；某项差 ≤ 3 个点时注明"可能只是单次训练的波动"（v11 那次）
4. 报告 `tmp/retrain/<时间>/report.md` + `result.json`（新模型路径、各项数字）；`_stats.json` 记这次训练用到的帧数和时间（"上次训练以来"从这里算）

### 7.3 换不换
- 标注页「重训」旁「最近一次报告」，页内渲染（同「剧本和报告」页用 `markdown.js`），底下「换上新 YOLO」「换上新外形头」两个按钮，可以只换一个
- 换 = 写 `console.toml` 的 `perception.model` / `attrs.model`（这两项加进设置清单；`config.toml` 面板仍然只读），旧值记进 `tmp/retrain/<时间>/adopt.json`；旁边「回退」把旧值写回
- 团子在跑时也能换，提示"下次叫醒生效"

## 8. 配置（`[inbox]`、`[retrain]`，加进 `config.example.toml` 和设置清单）

| 项 | 默认 | 说明 |
|---|---|---|
| `inbox.enabled` | true | false = 下线不收、面板不问、没有整帧页按钮（命令照样能跑） |
| `inbox.dir` | `datasets/inbox` | |
| `inbox.ask` | true | 点停止时问不问 |
| `inbox.agree` | 0.9 | 自动确认 / 自动丢低分框的把握 |
| `inbox.dup_diff` / `dup_gap` | 6 / 5 秒 | 去重 |
| `inbox.val_every` | 5 | 运行目录名 crc32 % 它 == 0 进验证集 |
| `inbox.retrain_min` | 50 | 「重训」按钮亮起的帧数 |
| `retrain.base` / `imgsz` / `epochs` / `batch` / `workers` | 见 7.2 | 同 v10 |

## 9. 出错

- 收：失败只记 WARNING，下线不受影响；② 开头补收
- 整理：单帧出错（读不了图、检测器异常）记进 `frames.json` 的 `error`、跳过，接着下一帧；外形头 / YOLO 加载失败整个任务失败，侧栏显示原因 + 日志最后几行
- 通过：`datasets/sky` 里已有同名文件 → 拒绝、提示；复制一半失败 → 删掉已复制的、`decision` 不写
- 重训：任何一步失败都不改配置；训出来的模型文件留着；侧栏显示失败的那一步和日志最后几行

## 10. 测试

单元测试（`python -m pytest -q`，假检测器 / 假特征模型 / 合成图）：
- 收：复制、幂等、`hard.saved = 0` 不收、出错不抛
- 去重、分边（同一运行同一边）、六种分流（表 4）、只处理没做过的帧（续跑）
- 帧状态现算：外形页挪了裁图后状态跟着变；「不要」→ edit
- 通过 / 撤销：复制和删除 `datasets/sky` 两个文件、最终标注（改类别、去掉不是人、`drop_low` 不写）、重名拒绝
- `writeback` 不碰 `source = inbox` 的裁图行
- `/api/frames/*`：Host / `X-Skydango` 校验同其他接口；编辑保存后人物框裁图进 `form/`、记 `by = frame-edit`
- 任务槽：任务在跑时不能叫醒 / 起沙盒（除非先停）；团子在跑时不能开任务
- 停止时的询问：有难例才问；选「整理」后团子退出才开任务
- 重训：版本号取法、缓存挪走、`adopt.json` 和回退（训练本身用一个假的 train 函数）
- `frames.js` 纯计算：缩放 / 平移下的坐标换算、拖出负宽高的框归一、按键 → 类别；页面不出现原生 `confirm` / `prompt` / `alert`（已有测试）

## 11. 真机验证（晚上）

1. 面板叫醒团子跑 15 分钟以上 → 点停止：弹出询问、张数对；选「整理」→ 团子正常收尾退出后侧栏出现整理进度，完了显示要看几张
2. `datasets/inbox/<运行>/` 里有 raw / hard.jsonl / frames.json / labels；`perception inbox status` 数字和侧栏一致
3. 标注页：外形页「来自整理」判完 → 整帧页出现这些帧；过目十几张（回车）；找一张篝火 / 漏框的帧编辑（删大框、画人和篝火）→ `datasets/sky` 多出这些帧、标注对
4. 「下次再说」一次：下次面板启动侧栏提示；跑满 20 次前收件箱里的不丢（看 `datasets/inbox/`）
5. 「重训」先把 `retrain.epochs` 临时改成 3 走通一遍（报告、换上、回退、团子叫醒后日志里模型路径对），再改回 120 正式训

## 12. 文件

新增：`src/skydango/vision/inbox.py`（收、去重、分边、分流、帧状态、通过 / 撤销，纯数据和文件，不碰面板）、`src/skydango/vision/retrain.py`（训练、对比、报告、版本号）、`src/skydango/console/frames.py`（整帧页后端、任务的进度读取）、`src/skydango/console/static/frames.js`；
改：`cli.py`（`_stop_scene` 收、`perception inbox` / `retrain` 子命令、预标注单帧函数抽出来）、`config.py` + `config.example.toml`、`console/server.py`（任务槽、停止询问的数据、`/api/frames/*`、`/api/inbox/*`、`/api/retrain/*`）、`console/settings.py`（新配置项 + 两个模型路径）、`console/static/console.html` / `live.js` / `labeling.js` / `common.js`（侧栏提示）、CLAUDE.md、进度文档。
