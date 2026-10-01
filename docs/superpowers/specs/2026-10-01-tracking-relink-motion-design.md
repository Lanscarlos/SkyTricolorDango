# 追踪器升级 + 失踪好友接回 + 运动方向（设计）

2026-10-01。起因（用户在真机 `run` 里多次遇到）：好友走过来时认出来了，走远一点名字标签自动隐藏，YOLO 感知层就把他当成**陌生人**，
接着冒"陌生人朝团子走过来了"、"小明走开了"。

## 0. 现状和原因

感知层已经有前后帧关联（`vision/track.py` 的 IoU 贪心追踪），名字挂在轨迹上：轨迹出现过标签就记 `tagged`，之后标签淡掉也不判陌生人
（`perception.py` `process()` 里 `player.data["tagged"] = True`）。**只要轨迹不断，现在就不会误判**；问题出在轨迹断：

- **低分框不进追踪**：`conf` 0.35 以下的框直接丢掉（只给难例收集看）。人走远、框变小，分数往下掉，轨迹就断。
  10-01 21:50 那次 run 的 `hard.jsonl` 185 条里 151 条是 `low_conf`；同一次 `agent.log` 里"陌生人 朝团子走过来了"十几次
- **小框 IoU 太敏感**：几十像素高的框挪几个像素 IoU 就掉到 `track_iou`（0.3）以下；身体截图实际只有 6~7 帧 / 秒（`capture = "body"`），帧间位移更大
- **转镜头不告诉追踪器**：空闲注意力、track、peek 会 `Camera.nudge`，整幅画面平移，IoU 全部失配（注意力的 nudge 也不调 `env.held()`）
- 断了以后新轨迹没挂过标签 → `stranger_after`（1 秒）后判陌生人；名字的 `last_seen` 不再刷新 → `keep`（5 秒）后"走开"

另外按 Q 那份设计（`2026-10-01-q-call-design.md`）§1.1「轨迹续命」还没实现：挂过标签的轨迹在标签淡掉后不刷新 `last_seen`，轨迹不断也会被判走开。

## 1. 已定的事

| 问题 | 结论 |
|---|---|
| 做法 | 在现有 `Tracker` 上自己加（两段匹配 + 速度预测 + 中心距离兜底 + 画面平移补偿），不用 ultralytics 的 BoT-SORT（检测器主要走 ONNX、核显机器没有 torch、跨类别和 `data` 要另包一层） |
| 和按 Q 的续命怎么分 | 按 Q 那份 §1.1 `sticky_names` **挪到这份里实现**（配置名、语义不变）：续命管"轨迹没断时算在身边"，这份管"让轨迹少断、断了在 `keep` 内接回来" |
| 接回来算多确定 | 两档：追踪器自己在 1 秒内续上的（含低分框续上的）还是同一条轨迹，**确定**；轨迹被删后靠失踪记录按位置接回的，**"像小明"**（`maybe`，`sure = False`），标签再亮才转确定 |
| 运动方向给谁用 | 这一期只做感知层的数据（`Person.motion`）+ status 一行文字；track / 注意力 / 冷场②以后再接 |
| 怎么定阈值 | 新的离线工具 `perception track-eval`，基线和新配置在同一批录像上对比 |

## 2. 追踪器（`vision/track.py`）

### 2.1 接口

```python
class Tracker:
    def __init__(self, buffer=1.0, min_iou=0.3, cross=frozenset(), cross_iou=0.5,
                 *, center_gate: float = 0.0, predict: bool = False) -> None
    def update(self, dets, now, *, low=(), shift=None, prune=True) -> list[Track]
    dropped: list[Track]   # 最近一次 update 删掉的轨迹（每次 update 开头清空再填）
```

- `low`：这一帧置信度在 `low_conf ~ conf` 之间的框（`process()` 里本来就算好了，`last_low`）
- `shift`：这一帧相对上一帧的画面平移 `(dx, dy)` 像素，`None` = 不知道
- `prune = False`：不删过期轨迹（`_far_tags` 同一帧第二次调 `update` 时用）
- 新参数都不传 / `center_gate = 0`、`predict = False` 时**逐字是现在的行为**：`gesture.py`、`hardcases.py` 里另建的两个 Tracker 不改

### 2.2 轨迹多记的东西

`Track` 加 `vx, vy, vh`（像素 / 秒，指数平均，`α = 0.5`）和 `weak_hits`（被低分框续上的次数，track-eval 用）。
速度用"这一帧的框 − 上一次的框 − 这期间累计的画面平移"除以时间差算；时间差 > 0.5 秒或刚创建的轨迹不更新速度。
没匹配上的轨迹不滑行，预测时现算：

```
预测框 = 最后的框 + (vx, vy, vh) × min(now − last, 0.5) + 这期间累计的画面平移
```

`predict = False` 时预测框就是最后的框。

### 2.3 两段匹配

1. **高分框**和所有轨迹配对。每一对算两个预测框（带画面平移 / 不带——光遇镜头绕着团子转，近处的人和远处背景平移量不一样，两个都试，取 IoU 大的），
   - IoU ≥ `min_iou`（跨类别 ≥ `cross_iou`）→ 候选，分数 = IoU
   - 否则 `center_gate > 0` 时看中心距离：中心距离 ≤ `center_gate` × 预测框高、框高比在 0.67~1.5 之间 → 候选，分数 = `0.3 × (1 − 距离 / 门槛)`，**低于任何 IoU 候选**（IoU 候选至少 0.3）；跨类别不走这条
   - 贪心按分数从高到低配，同现在
2. 第 1 步剩下的**轨迹**再和**低分框**同样配一遍；配上的照常更新，`weak_hits += 1`。**低分框只续旧轨迹，不开新轨迹**，没配上的低分框丢掉
3. 第 1 步剩下的高分框开新轨迹，同现在

返回值顺序同现在（按 `dets` 顺序；低分框续上的轨迹附在后面）。`name_tag` 轨迹也走同样的规则（标签轨迹续得上，名字投票、"同一条标签只 OCR 几次"更管用）。

### 2.4 画面平移估计（`vision/track.py` 的 `estimate_shift`）

```python
def estimate_shift(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray | None) -> tuple[float, float] | None
```

- 感知层每帧存一张缩小到 1/8 的灰度图；取画面**上半部分**，聊天面板开着时去掉左边三分之一，人物框所在区域填成均值（不让人带偏）
- `cv2.phaseCorrelate`（加 Hanning 窗）；响应值 < `PAN_MIN_RESPONSE`（0.1，估的）或位移 > 缩略图宽的 1/3 → `None`
- 结果乘回 8 倍，交给 `update(shift=…)`，同时累加到感知层的 `self._pan`（累计平移，失踪记录和运动方向用）

### 2.5 镜头事件

新接口 `env.camera_moved(at: float, kind: str)`（`kind` = `"turn"` / `"zoom"` / `"move"` / `"spin"`），EnvWatcher 是空实现。
身体每次设 `_camera_moved_at` 的地方（工具 `camera`、track、peek、注意力 nudge、`move`、`camera_reset`、look_around）顺带调一次；`move` 工具走路时用 `"move"`。

- `"zoom"` / `"move"` / `"spin"`：`at` 之后 `[track] settle` 秒内的帧不更新速度、清掉轨迹的 `hist` 和 `motion_hist`（缩放、走路时框高突变，不能当成人在走近走远）；匹配照常
- `"turn"`：什么都不清，靠平移估计补偿

### 2.6 暂停

`Tracker.shift(d)` 照旧挪时间；暂停恢复（`_resume`）时所有轨迹速度清零、`self._pan` 的上一张缩略图作废（镜头可能动过）。

## 3. 轨迹续命（从按 Q 那份 §1.1 挪过来，`[perception] sticky_names`）

`process()` 里标签挂完、身份摘完之后（`_assign_tags` 那段之后）：

- 轨迹 `data["tagged"]` 为真、`data["name"]` 有值（`maybe` 不算）、这一帧被更新过（`now − t.last <= PEOPLE_STALE`）→ `last_seen[name] = now`
- 轨迹断了（tracker 删掉）才开始算 `keep` 那 5 秒
- 同名标签清楚地出现在别处时摘掉名字（现有逻辑）照旧；两人交叉走过、追踪器把身份换错时，名字会跟到另一个人身上，标签再亮时纠正
- 暂停（`held`）时不续（`_blocked()` 本来就跳过检测）
- `sticky_names = false`：逐字照旧

## 4. 失踪好友接回（`[perception] relink`）

### 4.1 失踪记录

每帧 `tracker.update` 之后看 `tracker.dropped`：被删的人物轨迹挂着确定的名字（`tagged` 且有 `name`）→ 记一条

```python
@dataclass
class Lost:
    name: str
    box: Rect          # 最后的框
    vx: float; vy: float; vh: float
    last: float        # 最后看到的时间（t.last，不是删掉的时间）
    pan: tuple[float, float]  # 当时的累计画面平移
```

同一个名字只留最新一条。有效期：`now − last <= keep`（和"还算在身边"同一个窗口，接回永远发生在走开之前，不会先冒"走开"再冒"回来"）。
暂停恢复时 `last` 跟着 `_resume` 往后挪。被 `maybe_by = "relink"` 的轨迹被删时也记（名字取 `maybe`，`last` 沿用这条轨迹最后看到的时间），可以再接。

### 4.2 接回条件

每帧在判陌生人之前，对每条有效的失踪记录找候选。**全部满足才接**：

- 候选：`player` 类（不含黑影）、不是团子、没 `tagged`、没 `name`、没 `maybe` 的轨迹，且 `first >= lost.last − 0.2`（失踪之后才冒出来的；已经被判成陌生人的也可以）
- 预测中心 = 最后的框中心 + `(vx, vy) × min(dt, 1.0)` + `(self._pan − lost.pan)`，`dt = now − lost.last`；带平移 / 不带平移两个都试，取近的
- 中心距离 ≤ `min(0.6 + 0.5 × dt, 2.5)` × 最后的框高；框高比 0.5~2
- 这个名字的标签此刻没有清楚地出现在别处（`shown` 里有它 → 失踪记录直接作废）
- **有歧义不接**：同一个候选对得上两条失踪记录，或者一条失踪记录有两个候选、近的那个距离不到远的 1/1.3

### 4.3 接上之后

- `data["maybe"] = 名字`、`data["maybe_by"] = "relink"`，删掉这条失踪记录；日志 INFO"轨迹 12 像是 小明（断了 2.3 秒，按位置接回）"
- 后果全部沿用现有 `maybe`（认装扮那一套保守后果）：不判陌生人；好友还在 `keep` 内时刷新 `last_seen`（`process()` 里 `if maybe and now - last_seen <= keep`）；
  不发 `arrive` / `return`；`people()` 里 `sure = False`、status "像小明（没看到名字，…）"；可视化浅绿虚线"像小明?"
- 标签再亮出来：照常挂 `name` / `tagged`，同时去掉 `maybe`、`maybe_by`（标签是别的名字也一样，名字永远说了算）
- 认装扮（`_appearance_identify`）对 `maybe_by = "relink"` 的轨迹**不做"连续不像就摘"**、也不改成别的好友（颜色特征不稳，不能否掉位置连续性）；`_learn` 照旧只从挂着标签的样本学
- 同名标签清楚地出现在别处 → 摘掉 `maybe`、`maybe_by`（和现有摘 `name` 同一条规则）

### 4.4 和按 Q 那份对齐

- `CallSeen.unnamed`、身体自动喊的"画面里有没挂名字的人"：只数既没 `name` 也没 `maybe` 的人
- 接回后不冒假的 `leave`，自动喊的触发本来就少了，这是想要的

## 5. 运动方向（`[perception] motion`）

### 5.1 怎么算

每条人物轨迹（含黑影）在 `data["motion_hist"]` 存 `(时间, 框高, 补偿后的中心 x)`，补偿后的 x = 屏幕 x − `self._pan[0]`。
现有 `hist`（给 `approaching()`）不动，"走近"事件逐字照旧。镜头 `"zoom"` / `"move"` / `"spin"` 后 `settle` 秒内、暂停恢复时清空。

纯函数 `motion_of(hist, now, cfg) -> str | None`，结论是 `"走近"` / `"走远"` / `"往左走"` / `"往右走"` / `"站着"`：

- 只看最近 `motion_window`（1.5 秒）；样本 < 3 个或覆盖时间 < 窗口的 60% → `None`
- 前 1/3 和后 1/3 的平均比：径向 `r = h1 / h0 − 1`，横向 `s = (x1 − x0) / ((h0 + h1) / 2)`
- `|r| / motion_grow` 和 `|s| / motion_side` 都 < 1 → 站着；否则取大的那个：`r > 0` 走近、`r < 0` 走远、`s > 0` 往右走、`s < 0` 往左走
- **防抖**：新结论连续 `motion_hold`（0.5 秒）才替换 `data["motion"]`；`None` 立刻生效

"往左 / 往右"是**团子画面里的方向**，不是地图方位；光遇镜头绕着团子转，补偿只是近似，所以宁可 `None` 不说错。

### 5.2 给出去

- `Person` 加 `motion: str | None = None`（默认值，旧代码不用改）；`people()` 填 `data.get("motion")`
- `_describe_person`：`motion` 是走近 / 走远 / 往左走 / 往右走时在位置后面加"，正在走远"之类（"小明（左边·中，正在走远）"）；站着、`None` 不写
- 识别可视化：框的悬停文字带上运动方向
- `motion = false`：`Person.motion` 一直 `None`，status 逐字照旧

## 6. 离线评估：`perception track-eval <录像目录> [--model 模型] [--fps 6.5]`

新文件 `vision/trackeval.py`，复用 `compare.py` 的读录像（`frame_time`）和"录像时间当时钟"：

- 同一批录像跑两遍：**基线**（第 7 节的新开关全关）和**当前配置**；`--fps` 按录像时间抽帧，模拟 run 时身体截图的帧率
- 每个好友（出现过标签的名字）统计：
  - 轨迹断了几次，**原因**：断前那条轨迹的预测位置附近 ① 有低分框（低分框）② 有高分框但 IoU / 距离没过线（位移）③ 那一帧平移估计 > 框宽（画面平移）④ 什么框都没有（漏检）
  - 假的"走开"几次（`nearby` 里掉出去、`keep` 内又出现）
  - **确认冤枉**几次：判成陌生人的轨迹后来挂上了这个好友的标签
  - 接回几次：后来标签证实对 / 证实错（挂上别的名字）/ 一直没证实
  - 运动方向时间线（文字，每条轨迹一行："小明 #12 12.3~15.0 走远 → 15.0~17.2 往右走"）
- 报告 `tmp/track-eval/<时间>/report.md`，基线和当前配置并排；调阈值到"接回证实错"为 0、"确认冤枉"明显下降

先用 `tmp/record/play-1001-1`、`q-call-20260930*`、`gesture-*` 这几段。录像里没有镜头事件，只靠平移估计。

## 7. 配置（`[perception]`）

| 项 | 默认 | 说明 |
|---|---|---|
| `sticky_names` | true | 第 3 节（从按 Q 那份挪过来） |
| `track_low` | true | 低分框续轨迹（2.3 第 2 步） |
| `track_predict` | true | 速度预测 + 中心距离兜底 |
| `track_center_gate` | 0.6 | 中心距离门槛（× 预测框高，估的） |
| `track_pan` | true | 画面平移估计和补偿（2.4） |
| `relink` | true | 失踪好友接回（第 4 节） |
| `motion` | true | 运动方向（第 5 节） |
| `motion_window` | 1.5 | 秒 |
| `motion_grow` | 0.15 | 框高变化比例门槛（估的） |
| `motion_side` | 0.6 | 横向位移门槛，单位是身高（估的） |
| `motion_hold` | 0.5 | 防抖秒数 |

新开关**全关时逐字等于现在的行为**（有测试保证）。数字都没标定，track-eval 定完再上真机。

## 8. 测试

单元测试（`python -m pytest -q`，合成框 / 合成画面）：

- **追踪器**：低分框只续不开；小框横移超过 IoU 门槛、在中心距离门槛内接上；整幅画面平移时（传 `shift`）轨迹不断；预测最多往前 0.5 秒；
  中心距离候选排在 IoU 候选之后；`dropped` 返回被删的轨迹；`prune = False` 不删；新参数全不传时和现在逐字一样（`tests/test_tracker.py` 现有用例原样通过）
- **平移估计**：合成平移能估出（误差 < 1 个缩略图像素 × 8）；纯色画面返回 `None`；面板区域被去掉
- **续命**：标签消失但轨迹不断 → 一直在 `nearby`；轨迹断了 `keep` 秒后才不在；`sticky_names = false` 照旧
- **接回**：门槛内能接、接上是 `maybe` / `sure = False`、不判陌生人、不冒 `leave`；有歧义不接；标签在别处时作废；超过 `keep` 不接；黑影不接；
  标签亮出来后转确定（同名 / 别名）；认装扮不摘它；接回的轨迹再断还能接；暂停期间不过期
- **运动方向**：五种结论；样本不够返回 `None`；防抖；`"zoom"` 镜头事件后清空；status 的写法；`motion = false` 照旧
- **身体**：各处动镜头都调 `env.camera_moved`（假 env 记录调用）
- **track-eval**：合成一段"好友走远、标签消失、框变小变低分"的序列，基线报断开和冤枉、新配置不报

**真机验证**（晚上做）：

1. `perception track-eval` 跑上面几段录像，定 `track_center_gate`、`motion_*`、`PAN_MIN_RESPONSE`
2. `run --view`：好友退到标签消失，框不变橙色"陌生人"，最多变浅绿虚线"像小明?"；不冒"走开了""陌生人走过来"
3. 空闲注意力转镜头时，框的编号不跳
4. status 里"正在走远 / 往左走"和画面对得上

## 9. 改哪些文件

| 文件 | 改动 |
|---|---|
| `src/skydango/vision/track.py` | `Tracker` 两段匹配、预测、中心距离、`dropped`、`prune`；`estimate_shift` |
| `src/skydango/vision/perception.py` | 平移估计和 `_pan`、`camera_moved`、续命、失踪记录和接回（`_appearance_identify` 跳过 `maybe_by = "relink"`）、`motion_hist` / `motion_of`、`people()` 填 `motion` |
| `src/skydango/vision/people.py` | `Person.motion`、`_describe_person` |
| `src/skydango/vision/env.py` | `camera_moved` 空实现 |
| `src/skydango/vision/viewer.py`（+ 页面） | 悬停文字带运动方向 |
| `src/skydango/vision/trackeval.py`（新）、`src/skydango/cli.py` | `perception track-eval` |
| `src/skydango/brain/body.py` | 动镜头处调 `env.camera_moved` |
| `src/skydango/config.py` | 第 7 节的配置 |
| `docs/superpowers/specs/2026-10-01-q-call-design.md` | §1.1 指向本文件；`unnamed` 口径 |
| `CLAUDE.md` | YOLO 感知层一节加"追踪和接回"一段（实现后） |

## 10. 不做的

- 用运动方向的地方（track 提前转、空闲注意力、冷场②、新的"走远了"事件）：以后再接
- 地图上的方位（需要知道团子朝向）
- 重新识别（ReID）模型：认装扮那条线在做，这里只靠位置连续性
- 黑影的接回（没点火的人没有名字可接）
