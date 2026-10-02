# 装扮图鉴第 1 期：运行时收集（设计）

日期：2026-10-02。总纲见 `2026-09-28-perception-yolo-architecture-v0.2.md`；相关的认装扮见 `2026-10-01-appearance-design.md`，第二层见 `2026-10-02-perception-attrs-design.md`。

## 0. 起因和路线图

用户想让团子认出好友身上的发型、斗篷、面具……的**名字**（玩家之间约定俗成的叫法，非官方）。自己建图鉴太难，所以让图鉴**从团子自己的经历里长出来**：
见过 → 归类 → 问人 → 记住 → 认出来。像 Neuro-sama 那样有"会成长"的感觉：好友 / 卡洛带着团子到处认东西。

认法是**检索**，不是分类器：冻住的 DINOv2 把裁图变成向量，图鉴里存"向量 → 名字"，新加一件只是多存一个向量，**模型不用重训**；
认的时候取最像的前几件，再让 Claude 看图在候选里挑（选择题，不凭记忆猜俗称）。

分五期，各自 spec → 计划 → 实现：

1. **收集（这一份）**：运行时把近处的人清楚的整身裁图存进本机 `catalog/inbox/`；不发输入、不说话、不调模型
2. 归类 + 部位：DINOv2 局部特征（patch token）分部位（先无监督聚类，看能不能自然分出头发 / 脸和面具 / 斗篷 / 身体，分得开就给每堆起名；分不干净再在 patch 特征上训一个线性头），
   按部位聚成"没见过的单品"；管理面板「标注」页加「装扮」标签页给单品起名；起好名的图鉴（每件几张图 + 索引）进私有仓库 `private/catalog/`
3. 游戏里主动问：攒够次数的单品，在好友（尤其卡洛）身边时问叫什么（问好友自己身上的、或趁陌生人还在旁边问），从回答里学；卡洛说的 / 两个人说法一致的才算正式名字，别人说的记成"某某说叫…"；走主动开口的额度和护栏
4. 用图鉴认：检索前 5 + Claude 对图确认，没把握就说"像是……？"；结果按装扮缓存，只在换装 / 被问到时认
5. 以后：地图（进入新区域时 OCR 读到的地名当免费标签，接现有 `[places]` 图库）、先祖、物品（YOLO 只认粗类，具体名字交给图鉴）照同一个结构接进来

这一份只做第 1 期，但存储格式给后面几期留好接口（`kind` 字段）。部位**这一期不切**：整图在手，第 2 期看了真实数据再定怎么分。

## 1. 已定的事

- 收集器是**独立的新模块**（`vision/catalog.py` 的 `CatalogCollector`），自己的开关 `[catalog]`，和认装扮（`[appearance]`）完全分开：不需要打开认装扮，也不带进它的副作用（陌生人多等 1.5 秒、编号、Haiku 描述）
- **近处的人一直都收**，按质量把关、每个身份每次运行只留最好的几张；"刚点亮陌生人""东张西望正对着某人"这些时刻人本来就近、画面稳，自然包含在内，不往点亮链路和注意力模块里挂钩子
- 存**整个人的裁图**（框四周放宽、原分辨率）+ 元数据，不切部位
- dry-run 和 live **都收集**（只是截图存盘，同难例收集）
- 原始收集只放**本机** `catalog/`（gitignore）；第 2 期起完名的精选图和索引才进私有仓库

## 2. 存储

```
catalog/                              # gitignore
  inbox/                              # 第 1 期：原始收集
    2026-10-02/                       # 日期（本地时间，运行开始那天）
      20261002-213000-live-brain/     # 一次运行一个目录，和 runs/ 下的运行目录同名
        懒洋洋大王/1.jpg 2.jpg …
        团子/…
        陌生人-t37/…
        index.jsonl                   # 这次运行存下的每张一行；每次写盘整份重写（先写临时文件再替换），和目录里的图一一对应
```

`index.jsonl` 一行：

| 字段 | 说明 |
|---|---|
| `file` | 相对 `catalog/` 的路径 |
| `kind` | `"outfit"`（以后地图 / 先祖 / 物品用别的值） |
| `who` | 好友名（friends.md 的 `## 标题`）/ `"团子"` / `"陌生人-t<轨迹>"` |
| `who_kind` | `friend` / `self` / `stranger` |
| `sure` | 好友名是不是被名字标签证实（这一期好友一定是 true；"像小明"的按陌生人存） |
| `maybe` | 按外观 / 位置接回的"像谁"（没有就 null） |
| `run` | 运行目录名 |
| `t` | 截图时的感知层时间（感知层的 `now`，monotonic；离线工具里是录像时间） |
| `wall` | 截图时的 epoch 秒 |
| `box` | `[x, y, w, h]`，原画面坐标 |
| `height` | 框高 / 画面高 |
| `sharp` | 清晰度（§3.1 第 4 条） |
| `score` | 得分（§3.2） |
| `place` | 当时认地图认出的地方（感知层 `place`，没有就空串） |

文件名是缓冲里的名次 `1.jpg` ~ `<per_who>.jpg`（JPEG 质量 90）。目录名用认装扮的 `_folder_name` 规则（Windows 不能用的字符替换掉）。

- **好友**：轨迹上挂着名字标签证实过的名字（`data["tagged"]` 且有 `data["name"]`）
- **团子**：YOLO 的 `self` 轨迹（感知层认定的那一个）
- **陌生人**：点过火、没有被证实的名字的人物轨迹，按轨迹编号；只有"像小明"（`maybe`）的也按陌生人存，索引里记上 `maybe`
- **不收**：没点火的黑影（感知层 `_unlit`，全黑看不出装扮）；第二层判成先祖 / 共享空间的（`_other_form`）；第二层没放行 / 撤下的；YOLO 认成 player 的黑影靠黑度兜底（§3.1 第 5 条）；团子身上的 player 框（这一帧没 self 框，或和 self 框重叠 ≥ 0.3）按团子算、不当陌生人收（1 秒内见过的团子框都算）
- 陌生人只记轨迹编号，不记名字和聊天

## 3. 收集器（`vision/catalog.py`）

```python
class CatalogCollector:
    def __init__(self, cfg: CatalogConfig, root: Path, run_name: str, day: str) -> None: ...
    def update(self, frame, players: list[Track], selfs: list[Track], now: float, panel_rect: Rect | None,
               place: str, judge) -> None: ...   # judge：感知层给的回调，返回 (who, who_kind, sure, maybe) 或 None（不收）
    def dropped(self, tracks: Iterable[Track]) -> None: ...  # 追踪器删掉的轨迹：陌生人那份写出去
    def close(self) -> None: ...                 # 全部写出
```

身份判断（谁、收不收）由感知层通过 `judge` 给，收集器不碰 `is_unlit` / 第二层这些细节，方便单独测试。

### 3.1 门槛（全部满足才算可用）

每条轨迹最多每 `every` 秒看一次（`data` 之外自己记上次看的时间）。

1. **够大**：框高 ≥ `min_height` × 画面高
2. **完整**：框离画面四边都至少 `edge` 像素（人没被画面切掉）
3. **没被挡**：和别的人物 / 团子框不重叠、不压在聊天记录面板上 —— 把 `appearance.good_crop` 里的判断拆成 `appearance.clear_box(box, others, blocked, max_overlap) -> bool`，`good_crop` 改为调用它（行为不变），收集器也用它，`max_overlap` 用认装扮的同一个默认值 0.2
4. **清楚**：裁图转灰度、缩到高 256 像素，`cv2.Laplacian(..., CV_64F).var()` ≥ `sharp_min`
5. **不是黑影**（只对陌生人）：`candle.black(frame, box)` < `dark_max`（10-02 量过：黑影 0.61~0.93、点过火的人 ≤ 0.26；好友披黑斗篷有 0.66，所以好友不查）
6. **身份可收**：`judge` 不返回 None

裁图：框四周各放宽 `pad`（0.1）比例，夹到画面内，原分辨率。

### 3.2 挑图

- 每个身份一个缓冲：好友和团子按名字，陌生人按轨迹编号（好友轨迹断了再接上，还是同一个缓冲）
- 得分 `score = sharp × min(height / full_height, 1)`，`full_height` = 0.5（框高到画面一半就不再加分）
- 缓冲最多 `per_who` 张，两两拍摄时间至少隔 `gap` 秒：
  - 缓冲没满且和已有的都隔够 `gap` → 放进去
  - 和已有的某几张隔不够 `gap`：比这几张都好就把它们换成这一张（同一段时间留最好的），否则丢掉
  - 满了：比最差的那张好、且和其余的都隔够 `gap` → 替换最差的
- 缓冲里存的是裁图本身（内存里），写盘才压 JPEG

### 3.3 写盘

- 陌生人的轨迹被追踪器删掉（`tracker.dropped`）→ 写出他那份、清掉缓冲
- 每 `flush_every` 秒把所有缓冲写一遍（覆盖这次运行里同一个身份之前写的文件；缓冲变少时——一张新图替换掉两张挨得近的——删掉多出来的旧文件），然后重写这次运行的 `index.jsonl`
- 写出过的陌生人清掉缓冲后，索引里照样留着他那几行（索引按"已写出的全部"重写，不只是当前缓冲）
- `close()`（感知层 `stop()`）全部写出
- 写盘在感知层后台线程里做；任何 IO 错误只记一条 WARNING，不抛出、不影响感知
- 一次运行最多存 `max_per_run` 张（按缓冲里的 + 已写出并清掉缓冲的陌生人算）：满了以后任何会让张数增加的放入都不做（新身份开不了缓冲、没满的缓冲也不再加），已有的照常替换（不增加张数）
- 目录在第一次真的写图时才建（没收到任何图的运行不留空目录）

## 4. 接线

- `cli._scene_watcher`：`[catalog] enabled`、`[perception] enabled`、有运行目录（`run`，大脑模式和 `--no-brain` 都算）时建 `CatalogCollector(cfg.catalog, Path(cfg.catalog.dir), run.path.name, 运行开始那天)` 传给 `PerceptionWatcher(catalog=…)`；
  `view`、`perception detect`、沙盒没有运行目录或不走感知层，不建。`[perception]` 没开时不建、不提示（`[catalog]` 默认开，提示会让每个没开感知层的人每次都看到）
- `PerceptionWatcher.process()`：追踪完、分好 `selfs / players`、名字标签挂好、接回做完之后（陌生人判定那一段之后）调 `catalog.update(…)`；
  `tracker.dropped` 交给 `catalog.dropped(…)`；`stop()` 调 `catalog.close()`。感知暂停（`paused`）时不调 `update`
- 收集器只读轨迹，不改任何 `data`；`catalog = None` 时感知层行为逐字照旧
- `.gitignore` 加 `catalog/`
- 管理面板设置页加 `catalog.enabled`

## 5. 配置（`[catalog]`）

| 项 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 只截图存盘，没有对外行为；要配合 `[perception] enabled` |
| `dir` | `"catalog"` | 存储目录（相对当前目录） |
| `every` | 0.5 | 每条轨迹多久看一次（秒） |
| `min_height` | 0.25 | 框高至少占画面多少（估的，§6 定） |
| `edge` | 8 | 框离画面边缘至少几像素 |
| `pad` | 0.1 | 裁图四周放宽的比例 |
| `sharp_min` | 50.0 | 清晰度门槛（估的，§6 定） |
| `dark_max` | 0.5 | 陌生人框里很暗的像素占比到这个就不收（YOLO 认错的黑影） |
| `per_who` | 6 | 每个身份每次运行最多几张 |
| `gap` | 3.0 | 留下的图两两至少隔几秒 |
| `flush_every` | 300 | 多久写一次盘（秒） |
| `max_per_run` | 300 | 每次运行最多几张 |

`config.example.toml` 加这一节。

## 6. 离线工具：`python -m skydango catalog collect <录像目录> [--model 模型] [--fps 6.5] [-o 输出目录]`

- 用录像时间当时钟（同 `track-eval`：`compare.timed_files`、`trackeval.subsample`），在录像上跑感知层 + 收集器，输出到 `tmp/catalog/<时间>/`（目录结构和运行时一样，运行名用录像目录名）
- 另外写 `sheet.jpg`（所有存下的图按身份排成拼图，每张标清晰度、得分、框高）和 `candidates.jsonl`（**所有**看过的裁图的清晰度、框高、过没过每一条门槛，不只是存下的），用来定 `min_height` / `sharp_min`
- 白天就能在现有录像上用（`tmp/record/` 里的 play / q-call / gesture / candle 几段）

## 7. 测试（`python -m pytest -q`，合成画面 + 假轨迹）

- 门槛：太小、贴边、和别人重叠、压在面板上、模糊（合成图高斯模糊后）都不收；`judge` 返回 None 不收；`every` 内不重复看
- `clear_box` 拆出来后 `good_crop` 行为不变（现有认装扮测试原样通过）
- 挑图：最多 `per_who` 张；隔不够 `gap` 时只留好的那张；满了换最差的；好友换轨迹后进同一个缓冲
- 写盘：陌生人轨迹删掉时写出并清缓冲（索引里还在）；定时写盘覆盖同一身份的文件、索引整份重写且和目录里的图一一对应；`close()` 全部写出；`max_per_run` 满了不开新陌生人；写盘抛 `OSError` 时不抛出
- 目录结构、`index.jsonl` 字段；"像小明"的按陌生人存且记 `maybe`
- 感知层：`catalog = None` 时行为不变；暂停时不调 `update`；黑影 / 先祖 / 共享空间 / 没放行的 `judge` 返回 None；`stop()` 调 `close()`
- 离线工具：合成的小录像目录上跑出图、`sheet.jpg`、`candidates.jsonl`

## 8. 真机验证（晚上）

1. 白天先 `catalog collect` 跑现有录像，看 `sheet.jpg` / `candidates.jsonl` 定 `min_height`、`sharp_min`
2. 照常 `run --dry-run` 或 `--live` 玩一会儿（好友走近、站着、点亮陌生人），下线后看 `catalog/inbox/<日期>/<运行>/`：近处的好友和团子有没有存到、清不清楚、6 张是不是不同角度；
   有没有误收黑影 / 背景 / 半个人；`index.jsonl` 字段对不对
3. 看 `agent.log` 有没有写盘 WARNING；感知帧率和不开时比没有明显下降

## 9. 改哪些文件

| 文件 | 改动 |
|---|---|
| `src/skydango/vision/catalog.py`（新） | `CatalogCollector`、清晰度、得分、挑图、写盘 |
| `src/skydango/vision/appearance.py` | 拆出 `clear_box`，`good_crop` 调用它 |
| `src/skydango/vision/perception.py` | `catalog` 参数、`judge`、在 `process()` / `stop()` 里调用 |
| `src/skydango/config.py`、`config.example.toml` | `CatalogConfig` / `[catalog]` |
| `src/skydango/cli.py` | `_scene_watcher` 建收集器；`catalog collect` 子命令 |
| `src/skydango/console/settings.py` | `catalog.enabled` 开关 |
| `.gitignore` | `catalog/` |
| `CLAUDE.md` | 代码结构表、一节简短说明、常用命令 |
| `tests/test_catalog.py`（新）等 | §7 |
