# 认装扮：靠外观接回好友、认老面孔陌生人、说得出穿的什么（设计）

日期：2026-10-01。用户 10-01 brainstorming 定下：**两个问题都要解决**（好友名字标签没读到被当成陌生人；这次上线里认出回来的陌生人），
外加**团子知道自己和身边的人穿的什么**；**好友的外观记进关系卡、陌生人只记这次上线**；走「本机外观特征 + Haiku 文字描述」两层（方案 1），
**顺手攒认人模型的训练数据**，以后换成自己训的认人模型（方案 3）；**身高第一版不做**。

## 1. 现状和问题

- 身份跟着追踪轨迹走（`vision/perception.py` `process`）：一条 `player` 轨迹上出现过名字标签就一直算好友（`data["tagged"]`），
  一直没标签、过了 `stranger_after`（1 秒）、框高过了 `stranger_min_height` 就是陌生人
- 轨迹 `track_buffer`（1 秒）没匹配上就删（`vision/track.py`）。转镜头、被挡住、走出画面再回来都会接成**新轨迹**；
  这时好友的名字标签恰好读不到（远、被 UI / 团子挡住），1 秒后就被判成陌生人 → 陌生人事件、陌生人数不对、说话口气不对
- 点过火的陌生人外观和好友一样，只能靠名字标签分（架构 v0.2 §2）；陌生人断了轨迹就是一个新的陌生人，认不出"刚才那个"
- 团子不知道自己穿的什么；好友问"我衣服好看吗"时要靠 `look_person` 现看

## 2. 做法选择

**选：本机算外观特征 + Haiku 写文字描述（两层）。** 认人每隔几帧在本机算，不花额度；文字描述每个"新外观"只请 Haiku 写一次、按特征缓存。

没选的：
- **全交给 Haiku**：几秒一次、花额度，接轨迹这种实时的事做不了
- **现在就训认人模型**：要先攒数据、标注、训练。留成升级路线（§8），接口一样，第一版顺手攒数据
- **在 YOLO 上加认人**：YOLO 学的是"所有人都是 `player`"，认人要学"这个人和那个人不一样"，目标相反、损失函数不同；
  ultralytics 不直接支持检测 + 认人一体（FairMOT / JDE 那类要自己改网络）；绑在一起以后每次为物品重训 YOLO 都会影响认人
- **身高**：画面里的框高主要由远近决定；能当比例尺的只有名字标签和团子自己，可点过火的陌生人正好没有名字标签，帮不上最想解决的问题。
  以后可以试"人物框高 ÷ 名字标签高"给好友关系卡加"个子高 / 矮"（要先用录像验证标签像素高是否随距离同比缩放），这一期不做

**原则：外观只负责"先别判陌生人"和"八成是谁"，名字标签和 `check_friend` 才是最终依据。**

## 3. 组件

### 3.1 `vision/appearance.py`（新，纯计算）

- `crop_person(frame, box, others) -> np.ndarray | None`：从人物框里裁身体，只取**中间 60% 宽**（少带背景）。
  不是好样本时返回 None：框高 < `min_height` 像素；和别的人物框、团子框、聊天面板区域重叠超过 `max_overlap`
- `ColorEmbedder`（`model = "color"`，默认）：裁图按高度分**头（上 30%）**和**身体 / 斗篷（下 70%）**两块，
  各算 HSV 直方图（色相 16 × 饱和度 4，亮度权重低：只用来去掉太暗 / 太亮的像素），各自归一化后拼起来再归一化。`key = "color-v1"`
- ONNX 特征模型：把 `places.OnnxEmbedder` 挪到公共位置（`vision/embed.py`），错误提示里的配置名改成参数传入（认地图照旧报 `places.model`），
  外观用 `[appearance] model = "<路径>.onnx"`。`make_embedder` 同理按 `"color"` / `.onnx` 选
- `AppearanceBook`（外观记忆簿，只在内存里）：
  - `friends`：名字 → `Profile`（平均特征、样本数、最后更新时间、描述、特征的 `key`）
  - `strangers`：`"陌生人A"` → `Profile`（多了最后看到时间）；字母用完接着 `AA`、`AB`…
  - `me`：团子自己的 `Profile`
  - 平均特征用滑动平均（`ema`），存的时候归一化
- 相似度：余弦。**特征 `key` 不同（换了模型）的两份不比**

### 3.2 接进 `PerceptionWatcher`

- `[appearance] enabled` 且挂了 `AppearanceBook` 时，`process` 每帧对 `player` 轨迹（不含 `player_unlit`：黑影没有外观）
  每隔 `every` 帧裁一次图、算特征，平滑后放 `track.data["feat"]`、好样本数放 `track.data["samples"]`，再按 §4 判断
- `self` 框同样算，并进 `book.me`
- 特征计算放在检测之后、`_watch_typing` 之前；单帧最多算 `max_per_frame` 个（人多时轮流算），超时不影响检测节拍
- 暂停恢复（`_resume`）时不清特征（外观跟镜头无关）

### 3.3 `vision/wardrobe.py`（新，描述器）

- 后台一个线程 + 队列；一次性 `claude -p --model <describe_model>`（默认 haiku），令牌、隔离、`--tools ""` 同眼睛（复用 `eyes_command` 的做法）
- **送什么图**：这个身份最近 `pick_from`（5）个好样本里框最大的那张，按框四周各扩 15%（带上头部），框高不到 `describe_min_height` 不送
- **提示词**：只说看到的颜色和样子（发型 / 头饰、面具、斗篷 / 衣服颜色和长短、手里拿的），**不说季节名 / 物品名**；
  回 JSON `{"desc": "白色樱花发型、粉色长斗篷、戴面具", "clear": true}`，`desc` 截到 25 字；背影、模糊、被挡回 `clear: false`
- `clear: false` / 解析失败：这个身份 `retry_after`（60 秒）后用新的好样本再试，最多 3 次
- **什么时候描述**（优先级从高到低，同一身份在队列里只留一个）：
  1. 团子自己：这次上线第一次有好样本；`book.me` 的特征和描述时的特征相似度 < `changed`（在衣柜换了装）
  2. 好友：这次上线第一次靠名字标签认出、攒够 `min_samples` 个好样本，且关系卡里没有装扮或 §4.4 判定换了装
  3. 陌生人：每个"陌生人X"第一次走到中 / 近距离
- **额度**：每小时最多 `describe_max`（20）次；额度用完（同 `ClaudeLlm` 的判断）就跳过，`quota_wait`（10 分钟）后再试
- dry-run 也描述（只看不发输入，花额度同眼睛），只是不写关系卡

## 4. 认人规则

### 4.1 学外观（只从有把握的样本学）

- 轨迹**这一帧**挂着好友名字标签（`tagged` 里有它、标签有名字）且裁图是好样本 → 并进 `book.friends[名字]`
- 判成陌生人、已经有编号的轨迹 → 并进 `book.strangers[编号]`
- `self` 框 → 并进 `book.me`

### 4.2 没挂名字标签的 `player` 轨迹

1. **先比好友**（`maybe`）：候选 = 这次上线见过的好友 + 关系卡里有装扮特征的好友，去掉：
   - 名字标签此刻清清楚楚挂在别处的（同 `shown`）
   - 这一帧已经分给别的轨迹的（同一帧两条都像小明，只给相似度高的那条）

   轨迹好样本数 ≥ `min_samples`（3），和最像的好友相似度 ≥ 门槛、且比第二像的高 ≥ `margin` → `data["maybe"] = 名字`、`stranger = False`。
   门槛：这次上线学到的特征用 `match`；只有关系卡里旧特征的用 `card_match`（更严：光照、地图不同时颜色会偏）
2. **比不上**：走原来的陌生人判断（`stranger_after`、`stranger_min_height`）。判成陌生人且好样本够了，再比 `book.strangers`：
   最像的 ≥ `match` 且领先 ≥ `margin` → 接上那个编号；否则起新编号。`data["sid"] = "陌生人A"`
3. **名字标签永远说了算**：轨迹后来挂上标签，`maybe` / `sid` 摘掉，按标签走；`maybe` 和标签名字对不上时记难例（`hard/`，原因 `appearance`）
4. `maybe` 一旦给了，在这条轨迹上保留，除非：标签证明不是他；或者连续 `recheck` 个新样本都低于门槛（摘掉，重新判断）
5. 陌生人编号 `stranger_forget`（30 分钟）没出现就从记忆簿删掉

### 4.3 `maybe` 的后果（保守）

- **算**：不算陌生人（不发陌生人事件、`strangers()` 不数他）；`people()` 里 `kind = "friend"`、`name = 名字`，`Person` 新加 `sure: bool = True`，`maybe` 的是 `False`
- **刷新在场**：好友此刻还"在身边"（`last_seen` 在 `keep` 内）时，`maybe` 轨迹刷新他的 `last_seen` → 名字标签被挡几秒不会冒出走开 / 回来
- **不算**：好友已经走开了（超过 `keep`），靠外观"回来"不刷新 `last_seen`、不发 `arrive` / `return`，等名字标签确认；
  只靠关系卡旧特征认出（这次上线还没见过他的名字标签）的同理，只显示"像小明"
- `track` 盯人：有带名字的框（`sure`）用它；没有才用 `maybe` 的框（在名字标签 x 之前）
- `nearest()`、`approach`：`maybe` 按那个好友算（`approach` 只在他还在身边时发，同上一条）

### 4.4 换装

- 这次上线第一次靠名字标签认出好友、攒够 `min_samples` 后，拿 `book.friends[名字]` 和关系卡最近一套装扮的特征比：
  相似度 < `changed` → 换了装（新的一套，排描述）；≥ `changed` → 同一套（更新最后见到的日期，不重新描述）。关系卡没有装扮 → 新的一套
- 上线中挂着标签换衣服：身份不受影响，平均特征慢慢跟上；和描述时的特征相似度 < `changed` 时重新描述一次（同一个人一次上线最多 `redescribe_max` 次，默认 3）
- 关系卡特征的 `key` 和现在的模型不同：不比，当作"没有装扮特征"（描述照样留着给大脑看）

### 4.5 已知限制

- 颜色直方图受光照影响大（白天 / 晚上、不同地图），**同一次上线内接轨迹较可靠，跨天靠关系卡认人较弱**，所以 `card_match` 更严；真正解决靠 §8
- 撞衫（热门季节装扮、默认斗篷）会认错：所以 `maybe` 不打招呼、不算回来；陌生人编号认错只影响"又回来了"那句背景话
- 没点火的黑影没有外观，不参与

## 5. 关系卡

- `inner/ledger.py` 的 `Card` 新加 `outfits: list[dict]`，每套 `{"desc": str, "feat": [float…], "key": str, "first": "YYYY-MM-DD", "last": "YYYY-MM-DD"}`，
  最近的在最后，最多 `outfit_keep`（3）套；特征存 3 位小数
- 旧的 `people.json` 没有 `outfits` 照常读；只在 `--live` 写盘（同账本其他字段）
- 只给好友（名字 = friends.md 的 `## 标题`，同关系卡）；陌生人不进关系卡
- 描述器给好友写完描述，经身体线程（`Body._ledger_call`）写回这套的 `desc`；描述没回来之前这套 `desc` 为空，大脑那边不写外观
- 管理面板「内心」页的关系卡显示最近几套装扮（描述 + 日期）

## 6. 大脑看到的

- **status**：
  - 新一行"你自己：白色樱花发型、粉色长斗篷"（`book.me` 有描述时）
  - "身边的好友：小明（粉色长斗篷·今天一起 40 分钟·…）"：装扮放在交情前面，没描述不写
  - "画面里：小明（左边·近）、像小红（没看到名字，右边·远）、陌生人A（白斗篷，前面·中）"
- **arrive**：只在 §4.4 判定换了装、且两套都有描述时多一句"他换了装扮：上次是「…」，现在「…」"；没换不写。
  描述还没回来时不等，这句话放弃（避免拖住 arrive）
- **背景事件** `stranger_back`："刚才那个陌生人A（白斗篷）又回来了"：编号离开超过 `keep` 后被认回来；不单独叫醒大脑（进 `BACKGROUND`）
- **`look_person`**：也接受"陌生人A"（按 `sid` 找框）；"像小明"的框在找不到带名字的框时也用，结果里写"（没看到名字，按外观认的）"
- **提示词**（「说话」一节，`enabled` 时才加）：
  - 装扮是看图猜的，可能不准；"像小明"是没看到名字、按外观认的，别当成一定是他
  - 提装扮像玩家那样随口提（夸、吐槽、问在哪换的），别报一长串
  - "你自己"那行是你身上穿的，有人问就照着答
- **识别可视化**：`maybe` 的框画浅绿虚线、标"像小明?"；陌生人标"陌生人A"；`overlay` 多一个 `desc` 字段，鼠标悬停显示

## 7. 攒训练数据

- `[appearance] save` 开着时（默认开），好样本存进 `runs/<…>/appearance/`：
  - `crops/<身份>/<时间毫秒>.jpg`：身份 = 名字标签确认过的好友名；其余 `t<轨迹id>`（只在同一次运行里代表同一个人）
  - `appearance.jsonl` 一行一张：时间、轨迹 id、身份、名字标签、`maybe`、`sid`、框、和最像的好友 / 陌生人的相似度、当时认出的地图
- 限量：同一条轨迹每 `save_every`（2 秒）最多一张；一次运行最多 `save_max`（2000）张，超了只记一次日志
- 第一版只存，不整理；整理成训练集（按好友名跨运行合并）的命令放到 §8

## 8. 升级路线（方案 3，这一期不做）

- 单独训一个认人特征模型：输入人物裁图，输出特征向量；小的现成主干网络微调（度量学习），导出 ONNX，`[appearance] model` 指过去就换上
- 训练环境同 YOLO（`.pydeps` 里的 torch、本机 RTX 5070 Ti）
- 数据：§7 攒的图。同一条轨迹 = 同一个人；好友名字标签串起不同天、不同地图
- YOLO 不动

## 9. 配置 `[appearance]`

| 键 | 默认 | 说明 |
|---|---|---|
| `enabled` | false | 要配合 `[perception] enabled` |
| `model` | "color" | `"color"` 或 `.onnx` 路径（`size` / `norm` / `device` 同 `[places]`） |
| `every` | 3 | 每条轨迹每几帧算一次特征 |
| `max_per_frame` | 4 | 一帧最多算几个 |
| `min_height` | 0.10 | 好样本的框高（相对截图高度） |
| `max_overlap` | 0.2 | 和别的框 / 面板重叠超过这个比例就不是好样本 |
| `ema` | 0.2 | 平均特征的更新速度 |
| `min_samples` | 3 | 判 `maybe` / 陌生人编号 / 换装前至少几个好样本 |
| `match` | 0.85 | 估的，以 §10 离线标定为准 |
| `card_match` | 0.92 | 只有关系卡旧特征时的门槛 |
| `margin` | 0.05 | 最像的要比第二像的高这么多 |
| `recheck` | 5 | `maybe` 连续这么多个新样本低于门槛就摘掉 |
| `changed` | 0.70 | 低于它算换了装 |
| `stranger_forget` | 1800 | 秒 |
| `describe` | true | 关掉就只认人、不调模型 |
| `describe_model` | "haiku" | |
| `describe_min_height` | 0.18 | 送去描述的框高 |
| `describe_max` | 20 | 每小时最多描述几次 |
| `redescribe_max` | 3 | 同一个人一次上线最多重新描述几次 |
| `outfit_keep` | 3 | 关系卡每人留几套 |
| `save` | true | 攒训练数据 |
| `save_every` | 2.0 | 秒 |
| `save_max` | 2000 | 一次运行最多存几张 |

管理面板加 `appearance.enabled`、`appearance.describe` 两个开关。

## 10. 测试和验证

### 离线标定（白天能做）

新命令 `python -m skydango perception appearance-eval <录像目录> [--model YOLO模型] [--embed color|模型.onnx]`：
1. 对录像跑 YOLO + 名字 OCR，得到带名字的轨迹
2. 统计"同一个人"（同名字 / 同轨迹）和"不同人"的相似度分布，按"接错率 ≤ 2%"给出建议的 `match` / `margin` / `changed`
3. **假装看不到名字标签**：把带名字的轨迹每隔一段藏掉标签，按 §4 重放，统计接回对了 / 接错 / 漏接（接错比漏接严重）

报告 `tmp/appearance-eval/<时间>/report.md`。默认值以它为准再改。

### 单元测试（`python -m pytest -q`）

- 外观特征：不同颜色的合成"人"能分开；同色不同亮度分不太开（写成已知限制的测试）；好样本判断（太小、重叠）
- 记忆簿 / 规则：标签优先；同一帧两条都像同一个人只给一条；标签挂在别处的不参加；`maybe` 只刷新还在身边的好友、不产生 arrive；
  `card_match` 比 `match` 严；`recheck` 摘掉；陌生人编号接上 / 新建 / 遗忘；换装判定；特征 `key` 不同不比
- 描述器（假 `claude`）：JSON 解析、`clear: false` 重试、额度、排队去重和优先级、额度用完等待
- 关系卡：旧 `people.json` 能读；只在 live 写；最多 3 套
- 身体 / 大脑文字：status 三处、arrive 换装、`stranger_back`、`look_person("陌生人A")`
- **`enabled = false` 时提示词、事件、status 逐字照旧**

### 真机验证（晚上；没做完之前只在 `view` 和 dry-run 用）

1. `view` 打开外观识别：好友走到远处 / 背后、名字标签消失后，框上是"像小明?"而不是"陌生人"
2. 点过火的陌生人走开再回来，还是同一个"陌生人X"
3. `run --view` dry-run：status 有"你自己"和好友的装扮；描述器调用次数在额度内
4. `--live`：好友换一套装扮再来，arrive 里有换装那句、关系卡里写了新的一套
5. `runs/<…>/appearance/` 里的图数量、身份对

## 11. 改动范围

| 位置 | 改什么 |
|---|---|
| `vision/appearance.py`（新） | 裁图、`ColorEmbedder`、`AppearanceBook`、§4 的规则（纯函数 / 类，不碰设备） |
| `vision/embed.py`（新） | `OnnxEmbedder` / `make_embedder` 从 `places.py` 挪过来（`places.py` 改成从这里导入） |
| `vision/wardrobe.py`（新） | 描述器：队列、提示词、解析、额度 |
| `vision/perception.py` | `process` 里算特征、判 `maybe` / `sid`、刷新在场；`people()` / `overlay()` / `nearest()`；攒数据；`appearance` 难例 |
| `vision/people.py` | `Person.sure`；`describe_people` 写"像小明（没看到名字…）"、"陌生人A" |
| `inner/ledger.py` `inner/store.py` | `Card.outfits` 读写、换装判定的入口 |
| `brain/body.py` `brain/events.py` `brain/tools.py` | status、arrive、`stranger_back`（`BACKGROUND`）、`look_person` 认"陌生人A" |
| `brain/prompt.py` | §6 那段（`APPEARANCE_RULES`，`enabled` 时才拼进去，同 `INNER_RULES` 的做法） |
| `vision/viewer.py` `console/` | 浅绿虚线框、悬停描述；内心页关系卡显示装扮；两个开关 |
| `config.py` `config.example.toml` | `[appearance]` |
| `cli.py` | `perception appearance-eval` |
| `CLAUDE.md` | 新增「认装扮（`[appearance]`）」一节、代码结构表 |
