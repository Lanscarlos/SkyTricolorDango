# 认人底库：多张样本 + 双特征 + 团子转圈登记 + 拿不准就喊（设计）

日期：2026-10-03。依据：`tmp/spike-far-dino/result2.md`（三轮离线 spike，第 3 轮在末尾；明细 `result4.md`、`fuse4.md`）。
接在 10-03 已合并的 ① 认人修复之上（`_mark_dango`、`merge_people`、换角度、自动喊不重复），改造认装扮（`[appearance]`，spec `2026-10-01-appearance-design.md`）。

## 0. 目标和不做的事

用户的方向：YOLO 只负责把人框出来，身份交给后面；**好友只在这一帧挂着名字标签时才存特征**；团子启动时登记自己；换装 / 变身后重新登记。

现状：名字标签说了算、外观只给"像小明"（疑似）这个骨架已经有了；要换的是底库这一层——
现在每个人只有一个滑动平均特征、只用颜色直方图，还拿关系卡里跨天的旧特征认人。

成功的样子：
- 好友走到中远处、名字标签淡掉时，仍算他在身边、不被判成陌生人（"像小明"）
- 团子不被当成别人，镜头动过、聊天面板开关过之后也一样
- 拿不准的人，身体自动按 Q 喊一声确认，亮出标签就转成确认，并把这一帧的样子补进底库

不做：
- 外观认出来就当真（发"来到身边"、打招呼）——认错的代价是对着陌生人叫好友的名字
- 跨天靠外观认人（关系卡的旧特征不再拿来认人）
- 训练"认人头"（只有一个好友的录像，训不出来；本设计运行时攒的样本以后正好当训练数据）
- 远处底库（spike 第 2 轮：瓶颈是朝向 / 场景，不是距离）
- 装扮描述、判换装（`outfit_change`）的逻辑不动

## 1. 依据（spike 第 3 轮，21 段录像、一个好友）

| 问题 | 结果 |
|---|---|
| 底库平均成一个 vs 存多张取最像的 | 颜色认好友 56% → 89%，DINOv2 43% → 54%（错 < 2% 时，前后半切） |
| DINOv2 输入 | 整框补成正方形明显好过中间 60% 拉伸；patch 平均比 CLS 差 |
| 团子 vs 好友二选一 | DINOv2 分对 93% / 97%，颜色 82% / 94%；失败一次是开头 20 秒没拍到团子正面，一次是好友中途变身雪人 |
| 跨天复用团子底库 | AUC 0.38 ~ 0.99，不稳 → 每次上线重建 |
| 同场景认好友 | 颜色好过 DINOv2；两者合起来不比单用颜色好 |
| 框高 0.10 ~ 0.13 | 两种特征都认不出 → 好样本门槛提到 0.13 |

所以分工：**认"是不是团子"用 DINOv2，认"这次上线里是不是这个好友"用颜色**，底库存多张、取最像的一张。

## 2. 结构和组件

### 2.1 `vision/gallery.py`（新，纯数据，不碰设备）

- `Sample`：`color`（颜色特征，单位向量）、`dino`（DINOv2 特征，单位向量；没配 DINOv2 时为 None）、`t`（时间）、`h`（框高 / 画面高）、`pinned`（钉住不挤）
- `Gallery`（一个身份的一组样本）：
  - `add(sample)`：和已有某张**颜色特征**余弦 ≥ `DUP`（0.97）→ 不加新的，只把那张的 `t` 刷新成新的；否则加入。
    超过 `gallery_max`（40）时挤掉一张：只在没钉住的里挑，挑"和别的样本最像"（最近邻余弦最高）的那张，一样像时挤旧的
  - `best(feat, which)`：`which` = `"color"` / `"dino"`，返回和底库里最像那一张的余弦（取最大值）；底库空、或这一路没有特征返回 None
  - `__len__`、`pinned_count`
- 线程安全由 `AppearanceBook` 的锁负责，`Gallery` 本身不加锁

### 2.2 `AppearanceBook`（改）

- `Profile` 多一个 `gallery: Gallery`。**认人一律用底库**：`assign_friends`、`still_like`、`best_friend`、`stranger_id` 改成按 `gallery.best(…, "color")` 打分
- 新增 `looks_like_dango(sample) -> bool`：团子底库 `best(…, "dino") ≥ dango_match`，并且**比任何好友底库的 dino 分都高**；团子底库空 / 没有 dino 特征 → False
- 新增 `friend_score(sample) -> list[(名字, 分数)]`：按颜色分从高到低，给第 3 节分三档用
- `learn(kind, who, sample, now, crop, pinned=False)`：除了原来的滑动平均 `feat`（只给装扮描述 / 判换装用，逻辑不动），把样本加进底库
- 关系卡旧特征：`load_cards` 照旧读描述；`card_feats` / `card_match` **不再参与认人**（`_candidates` 只看这次上线学到的好友）。写关系卡（`OutfitNote` 带特征）照旧，换装逻辑不受影响

### 2.3 算特征（感知层 `_appearance_features` 改）

- 一次裁图出两个特征：颜色沿用 `good_crop`（中间 60% 宽）+ `ColorEmbedder`；DINOv2 用 `attrs.crop(frame, box, pad=0.0, size)`（补成正方形）+ `OnnxEmbedder`
- 模型：`[appearance] dino`（默认 `"models/dinov2-small.onnx"`；空 = 不用 DINOv2）。第二层开着且 `[attrs] backbone` 是同一个文件时共用同一个 `OnnxEmbedder`（一个推理会话），否则自己建一个；device 跟 `[appearance] device`
- 轨迹上：`data["feat"]` 照旧是颜色的滑动平均（`samples` 计数照旧）；新增 `data["sample"]` = 这一次的 `Sample`（给学和认用）
- `[appearance] model` 仍然可以是 `.onnx`（替换颜色那一路），行为同现在

### 2.4 配置（`AppearanceConfig`）

| 项 | 默认 | 说明 |
|---|---|---|
| `min_height` | 0.10 → **0.13** | 好样本框高 |
| `match` | 0.85 → **0.88** | 颜色、最像一张：≥ 它（且领先第二像 `margin`）= "像小明" |
| `unsure` | **0.83**（新） | `unsure ≤ 分数 < match` = "可能是小明"，去喊 |
| `unsure_wait` | **2.0**（新） | "可能是"持续这么久才喊（秒） |
| `dino` | **"models/dinov2-small.onnx"**（新） | 空 = 只用颜色、不判像团子 |
| `dango_match` | **0.80**（新） | DINOv2、最像一张：≥ 它且比任何好友都像团子 = "看着像团子" |
| `gallery_max` | **40**（新） | 每个身份最多几张 |
| `enroll_max` | **16**（新） | 转圈登记最多钉住几张 |
| `card_match` | 废弃 | 留着读旧配置不报错，不再使用 |

`margin`、`recheck`、`every`、`max_per_frame`、`max_overlap`、`min_samples` 不变。

## 3. 每帧怎么学、怎么认

顺序接在现有流程后：追踪 → `_mark_dango`（按位置认团子）→ 名字标签挂人 → 下面这些（`_appearance_identify` 改）。

### 3.1 学（只从身份确定的框学）

- **团子**：`self` 框，或 `_mark_dango` 打了标记（`data["dango"]`）的人物轨迹 → 团子底库
- **好友**：**这一帧**挂着名字标签（`tagged` 里有、且标签读出了名字）→ 他的底库。之前挂过、这一帧被挡住的**不学**
- **陌生人**：已经有编号（`data["sid"]`）→ 那个编号的底库（同现在）
- "像小明" / "可能是小明" / 按位置接回的 / "看着像团子"的：**都不学**

### 3.2 认（只认没挂名字、没有团子标记的点过火的轨迹）

1. **像不像团子**：`looks_like_dango(sample)` 为真 → `data["dango_look"] = now`，摘掉 maybe / unsure。
   效果同团子标记：不判陌生人、不挂"像小明"、不算没挂名字的人（`unnamed`、自动喊、`_count_unnamed`）、图鉴按团子算。
   `DANGO_LOOK_HOLD`（3 秒）内没再被判像团子就失效；挂上名字标签立刻失效（名字说了算）
2. **像哪个好友**（颜色，`friend_score`，好样本攒够 `min_samples` 后才判）：
   - 最高分 ≥ `match` 且领先第二像 ≥ `margin` → `data["maybe"]`（"像小明"），后果同现在：不判陌生人、刷新在场，不发 arrive / return、不打招呼；`people()` 里 `sure = False`
   - `unsure` ≤ 最高分 < `match` → `data["unsure"] = (名字, 开始时间)`（"可能是小明"），**先不判陌生人**，交给第 5 节
   - 最高分 < `unsure` → 照常走陌生人流程
3. **已经"像小明"的**：沿用现在的规矩——标签亮出来名字说了算、他的标签此刻在别处就摘、连续 `recheck` 次 < `match` 就摘
4. 一个名字一帧只给一条轨迹（同现在）；"可能是"也一样（同一个名字只给分数最高那条）

### 3.3 和判陌生人的关系

`_looks_checked`（能不能判陌生人）多两个条件：有 `dango_look` 不判；有 `unsure` 时等它结束（第 5 节，最多 `unsure_wait + call.window` 秒）。原来的 `STRANGER_GRACE` 照旧。

## 4. 启动时转一圈登记团子

- **时机**：身体启动、轮盘和面板准备好之后，**大脑第一轮之前**，`Body.enroll_self()` 一次。条件：感知层开着、`[appearance] enabled`、`dino` 加载成功
- **转**：`with panel.borrow("enroll"):` 里调现成的转圈（`Camera.spin` 一整圈，同 `look_around`），转完 `env.camera_moved("spin")`；一整圈回到原朝向，不额外复位。
  转圈截图照旧交给 `env.sweep(...)`（顺带更新"哪个方向有谁"和 `self_box`），不单独发事件
- **取团子框**：每张截图跑一遍检测；有高分 `self` 框用它，否则用这次 `sweep` 认出的 `self_box`（转圈时一直在画面中间不动的那个人，对每张截图都适用）。
  按 §2.3 的好样本条件裁图、算两种特征，按时间均匀挑最多 `enroll_max` 张，`learn("me", "", sample, pinned=True)`
- **跳过**（不转，只靠平时攒；记 WARNING，status 写"团子登记：没转（原因）"）：dry-run；黑屏 / 加载中；别的面板挡着（`panels.state.others()`）；技能在跑。
  **转了但一张团子样本都没取到**：同样记 WARNING、status 写"团子登记：转了一圈没认出自己"。都不重试
- 成功时 status 写"团子登记：N 张"；转圈截图里取到的团子裁图存进 `runs/<…>/enroll/`（事后核对）
- 之后团子的样本走 §3.1；钉住的不挤。团子换装后新样子靠平时的样本占满底库，旧的钉住样本不像新的团子，也不会让别人被误认成团子

## 5. 拿不准就喊

- **起因**：某条轨迹 `data["unsure"]` 持续 ≥ `unsure_wait`，小明此刻没在别处被确认（他的名字没挂在任何轨迹上、标签没在别处亮着），这条轨迹没为它喊过（`data["unsure_called"]`）
- **感知层给身体的接口**：`env.unsure(now) -> list[(轨迹 id, 名字, 开始时间)]`（只列满足上面条件的）；EnvWatcher 空实现返回 []
- **怎么喊**：`Body._watch_call` 多一种起因 `"unsure"`：没有"刚走开的好友"时再看 `env.unsure(now)`；额度、间隔、拦截条件**全部共用**现在的自动喊（`min_gap` 20 秒、`auto_window` 1 分钟 `auto_quota` 3 次、`auto_again` 5 分钟、`_auto_call_blocked` 的那些情况）；dry-run 只记日志、不按键。
  喊之前对列出的轨迹打 `data["unsure_called"] = 按键时间`（喊成没喊成都打，避免反复排队）
- **喊完怎么判**（感知层在呼喊窗口结束那一帧，`_call_tick` 收尾时处理带 `unsure_called` 的轨迹）：
  - 窗口里他身上挂上了小明的标签 → 什么都不用做：正常挂标签流程已经确认、摘掉 unsure，并按 §3.1 把这一帧学进小明的底库（闭环）
  - 小明的标签亮在别人身上 → 摘掉 unsure（同 maybe 的规则）
  - 窗口结束他身上什么都没亮 → 摘掉 unsure，记 `data["unsure_miss"] = True`（这条轨迹以后不再进"可能是"、不再为它喊），照常走陌生人流程。**不存负样本**
- **一直没喊成**（额度用完、被拦、dry-run、`[call] auto` 关着）：`unsure` 挂满 `unsure_wait + call.window` 秒就摘掉，记 `unsure_miss`，照常走陌生人流程
- **大脑看到的**：喊完照旧是背景事件 `call`（"你下意识喊了一声：认出 小明（右边·远）"）；status"画面里"写"可能是小明（没看到名字）"，区别于"像小明（没看到名字）"。不加新事件、不叫醒大脑
- 识别可视化：可能是 = 浅绿点线 + "可能是小明?"；看着像团子 = 灰白虚线

## 6. 出错和兼容

- DINOv2 模型缺失 / 加载失败：启动 WARNING，只用颜色，`looks_like_dango` 恒为 False、不转圈登记
- DINOv2 推理连续出错 10 次：关掉 DINOv2 这一路（同第二层的自关），记 WARNING
- 算特征 / 学 / 认任何一步出错：只记日志，当这一帧没算到；名字标签、`_mark_dango`、接回照常
- `[appearance] enabled = false`：逐字照旧（**默认仍然关**，真机验证完再开）
- `dino = ""`：颜色多张底库 + 拿不准就喊；不判像团子、不转圈登记
- `[call] enabled / auto` 关着：不喊，"可能是"到时就摘

## 7. 标定工具

`perception appearance-eval` 加 `--gallery`：按 spike 第 3 轮的切法（每段录像前一半挂标签的当好友底库、后一半当查询；开头 20 秒的团子框当团子底库；团子框当同场景负样本），
打印 `match` / `unsure` / `dango_match` 的建议值（错 < 2% 的门槛和对应认出率）和三档的人数分布，写进报告。不加 `--gallery` 时同现在。

默认值来源（框高 ≥ 0.13）：颜色最像一张 0.88 → 好友认出 87%、团子 + 陌生人被误认 1.5%；0.83 → 认出 94%、误认 6.7%（0.83 ~ 0.88 之间约 5% 的陌生人会被白喊一次）；
DINOv2 团子 0.80 → 团子认出 58%、好友被当团子 0.4%（只用开头 20 秒的团子底库，转圈登记后应更高）。

## 8. 测试（`pytest`，合成特征 / 假设备）

- `Gallery`：去重刷新时间、上限、挤最重复的、一样像挤旧的、钉住不挤、`best` 两路、空底库
- 学：只从这一帧挂标签 / 团子标记 / `self` 框学；maybe / unsure / 接回 / dango_look 不学
- 认：像团子挡住 maybe 和判陌生人、`DANGO_LOOK_HOLD` 失效、挂标签立刻失效；三档分数各走各的路；一个名字一帧一条（maybe 和 unsure 都是）；`card_feats` 不参与
- 拿不准就喊：`env.unsure` 的条件；`_watch_call` 起因 unsure、额度和拦截共用；喊完三种结局；没喊成超时摘掉；`unsure_miss` 不再进
- 启动转圈：跳过条件、取团子框的顺序（self 框优先、否则 sweep 的 self_box）、`enroll_max` 均匀挑、钉住
- 兼容：`enabled = false` 逐字照旧；`dino = ""` 时不判像团子、不转圈

## 9. 真机验证（晚上）

1. 启动看转圈登记：status"团子登记：N 张"，`runs/<…>/enroll/` 里的裁图是不是团子各个角度
2. 好友从近处走到中远处、标签淡掉：`view` 里一直是"像他"，不冒"走开了"、不判陌生人
3. 拿不准时自动喊：喊得勤不勤（日志"自动喊一声"），亮出标签后是不是转成确认、底库多了样本
4. 镜头转过、聊天面板开关过之后：团子框还被当团子（不出"陌生人"、不进图鉴的陌生人）
5. 用 `perception appearance-eval --gallery` 在当晚录像上重算门槛，必要时改默认值
