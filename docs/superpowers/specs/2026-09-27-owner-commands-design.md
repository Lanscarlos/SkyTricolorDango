# 主人命令（# 开头）— 设计

日期：2026-09-27

## 目标

卡洛作为主人，在游戏聊天里发一条以 `#` 开头的消息，团子把它当命令执行（改好友备注、记笔记、暂停/恢复自动回复、查状态），
而不是当成普通聊天转发给大模型。

## 需求（已和用户确认）

1. **只做普通 Agent 这条主循环**（`agent.py` + `chat/responder.py`），不碰实验中的统管大脑（`brain/`）。
2. **谁是主人靠配置识别**：新增 `reply.owner_name`，填游戏里显示的昵称（比如 `"懒洋洋大王"`），
   和聊天记录面板解析出的 `speaker` 做精确匹配。留空表示不启用这个功能。
   - 只有 `vision.mode = "log"` 时 `Message.speaker` 才有值（当前配置就是 log 模式）；其他模式下这个功能自动失效。
3. **第一版支持 4 个命令**：`#friend`、`#remember`、`#pause` / `#resume`、`#status`。
4. **命令永远回一句固定确认语**（不调模型），用来和平时聊天的语气区分开，让主人确定命令真的执行了。
5. **dry-run 下命令依然真的写文件**：`#friend` / `#remember` 是主人自己的本地操作，不算"团子回复了什么"，
   不受 `reply.dry_run` 影响；但确认话要不要真的发进游戏，还是照 `dry_run` 的老规矩（只打印不发送）。

## 方案选择

**命令在读取阶段拦截，走独立的确认回复，不进大模型的对话历史。**

新模块 `chat/commands.py` 提供 `CommandRouter`，`Agent.step()` 里从这一帧读到的新消息中，
把「speaker 是主人 且文本以 `#`/`＃` 开头」的行单独摘出来处理，不进入 `pending`（不占用 debounce），
处理结果（确认/报错文案）当作一次独立的发送，不占用 `RateLimiter`（那是给聊天回复防刷屏用的，
命令是主人主动触发，应该马上有反馈，不该被别人的聊天限速卡住）。

放弃的方案：
- **走 `Responder`，用大模型生成确认话**：更"自然"，但结构化命令不需要语言生成，多一次 API 调用、多一份延迟和不确定性
  （模型可能编造"记好了"以外的内容，或把命令语法本身当聊天接话）。
- **命令走终端 / 本地文件**：不是用户要的形式，主人想直接在游戏里用聊天发。

## 1. 命令语法

一行消息同时满足：
- `speaker == cfg.reply.owner_name`（`owner_name` 非空）
- 文本（去掉首尾空白）以 `#` 或 `＃` 开头

才判定为命令；否则一律当普通聊天，走原来的 `pending` → `Responder` 流程。

命令格式：`#<命令词> <参数...>`（命令词和参数之间用空白分隔，`#` 和命令词之间不留空格）。

| 命令 | 参数 | 作用 |
|---|---|---|
| `#friend <昵称> <备注内容>` | 昵称（不含空格）+ 备注（其余全部） | 在 `friends.md` 追加一条好友备注 |
| `#remember <内容>` | 其余全部 | 随手记一条到 `inbox.md`（复用 `MemoryStore.add_memos`） |
| `#pause` | 无 | 暂停自动回复别人的消息 |
| `#resume` | 无 | 恢复自动回复 |
| `#status` | 无 | 回报当前运行状态（只读，不改变任何状态） |

解析失败（命令词不认识、必填参数缺失）→ 回一句报错确认，同样不转发给大模型。

## 2. 执行位置与流程

`CommandRouter`（`chat/commands.py`）：

```python
class CommandRouter:
    def __init__(self, store: MemoryStore | None, on_pause: Callable[[bool], None], status: Callable[[], str]) -> None: ...
    def handle(self, text: str) -> str:  # 返回要回复的确认/报错文案
        ...
```

- `store`：`#friend` / `#remember` 用；`store is None`（`reply.memory_dir` 没配）时这两个命令回"没开记忆功能"。
- `on_pause`：`#pause`/`#resume` 调用它切换 `Agent` 的暂停开关。
- `status`：`#status` 调用它拿一行状态文本。

`Agent.step()` 改动：

1. `fresh = self.reader.read(frame, now)` 之后，若 `cfg.reply.owner_name` 非空，
   把 `m.speaker == owner_name and _is_command(m.text)` 的消息挑出来（保留原有顺序），从 `fresh` 里剔除。
2. 对每条命令消息，调用 `self.commands.handle(m.text)` 得到确认文案，包成 `Reply(text=confirm)`，
   走和普通回复相同的发送分支（`dry_run` 判断、`sender.send`、`self_filter.remember`、`disclosure_prefix`），
   但**不经过 `RateLimiter`**，也**不写入 `Responder` 的对话历史 / `MemoryStore.history`**（命令不是"聊天"）。
3. 剩下的 `fresh`（非命令消息）按老流程继续：加入 `pending`、等 debounce、交给 `Responder`。
   若 `Agent.paused` 为真，这些消息只记日志（`log.info` 一行"已暂停，忽略"），不进入 `pending`（相当于团子在偷懒，
   但命令本身在第 2 步已经处理过，不受这个开关影响）。

`Agent` 新增：
- `self.paused: bool = False`（运行时状态，不落盘，重启恢复 `False`）
- `self.commands = CommandRouter(store, on_pause=self._set_paused, status=self._status_line)`
  （`store` 从 `Responder` 已经持有的 `MemoryStore` 里取，或者 `Agent` 构造时单独传入，取决于实现时的接线方便程度）

## 3. 具体命令行为

### `#friend <昵称> <备注内容>`

- 参数不足两段（缺昵称或备注）→ `"格式不对，是 #friend 昵称 备注内容"`。
- 在 `friends.md` 里找 `## <昵称>` 这一节：
  - 存在：在该节最后一个非空行之后、下一个 `## ` 标题之前，追加一行 `- <备注内容>`。
  - 不存在：在文件末尾追加 `\n## <昵称>\n- <备注内容>\n`，新建一节。
  - 原子写：先写临时文件再 `replace`（照抄 `MemoryStore.write_notes` 的手法），避免写到一半崩了丢文件。
- `MemoryStore` 新增方法 `add_friend_note(name: str, note: str) -> None` 承担这部分逻辑（纯文本操作，
  单元测试直接测这个方法，不用起 Agent）。
- 成功确认：`"记好啦"`。

### `#remember <内容>`

- 内容为空 → `"格式不对，是 #remember 内容"`。
- 直接调用现有 `store.add_memos([内容])`（写 `inbox.md`，和聊天时"随手记"共用同一份文件、同一把锁）。
- 成功确认：`"记下了"`。

### `#pause` / `#resume`

- 不需要参数；多余参数忽略。
- `#pause`：`Agent.paused = True`，确认 `"先歇会儿，不自动理别人了"`。
- `#resume`：`Agent.paused = False`，确认 `"好，继续陪聊"`。
- 已经是目标状态也照常回确认（幂等，不报错）。

### `#status`

- 只读，拼一行状态，受 `cfg.reply.max_chars` 截断保护（和普通回复一样）：
  - 运行模式：`live` / `dry-run`
  - 暂停状态：`运行中` / `已暂停`
  - 待处理消息数：`len(self.pending)`
  - 发送限速余量：这一分钟还能发几条（`RateLimiter` 新增一个只读方法 `remaining(now)`，
    返回 `max_per_minute - len(self._sent within 60s)`）
- 格式示例：`live｜运行中｜待处理0｜限速8/8`。

### 未知命令

- 命令词不在上述 4 个之列 → `"没这个命令"`。

## 4. 边界情况

- `owner_name` 为空（默认）：功能整体不启用，`_is_command` 判断永远为假，不影响现有行为。
- `vision.mode` 不是 `"log"`：`speaker` 恒为空字符串，`owner_name` 再怎么配都匹配不上，功能自动失效
  （文档里注明，不额外处理）。
- 确认文案本身也会被面板重新读到：和普通回复一样调用 `self_filter.remember(text, now)`，避免下一帧把自己的确认话当成新消息。
- 命令消息本身（比如"#friend 懒洋洋大王 很会玩游戏"这行原文）不进 `Responder` 的历史、不写 `MemoryStore.history`，
  避免以后聊天时模型看到自己历史记录里夹着命令语法。
- 一帧里同时有多条命令消息（比如网络卡顿攒了几条）：按 `fresh` 里的原有顺序依次处理、依次发送确认，
  不做特殊合并（预期极少发生）。

## 5. 测试

- `tests/test_commands.py`（新增）：
  - `#friend` 追加到已有小节 / 新建小节，原子写不丢已有内容。
  - `#remember` 落到 `inbox.md`。
  - `#pause` / `#resume` 触发回调。
  - `#status` 输出格式（配合假的 `RateLimiter`/`pending` 状态）。
  - 缺参数、未知命令的报错文案。
- `tests/test_agent.py`（补充用例）：
  - 主人发 `#remember 测试` → 触发确认发送，不进入 `pending`、不占用限速。
  - 非主人发以 `#` 开头的文本 → 当普通聊天处理（走 `Responder`）。
  - 主人发 `#pause` 后，别人的消息不再触发大模型回复；再发 `#resume` 恢复。
  - `owner_name` 为空时，即使 speaker 匹配也不触发命令解析（行为等同功能关闭）。

## 6. 追加：`#spin` 转一圈（2026-09-28，已和用户确认，**已实现**，待真机验证）

用途：测试感知层设计 §14 的"环绕扫描 / 转圈认团子"。主人不在电脑前时，从自己的账号在游戏里发 `#spin`，
团子用普通模式 `run` 跑着，转一圈、把每帧截图存下来，回一句"转完了"。在电脑前测试用命令行 `camera spin`（见下）。

### 需求（已和用户确认）

1. **只加到普通模式**（`run`，`chat/commands.py`）。大脑模式（`run --brain`）不加：
   大脑模式的 `#` 消息按原设计只交给大脑、打开授权窗口，不解析成直接执行的指令，这条原则不为测试命令破例。
2. 另加命令行 `python -m skydango camera spin`，在电脑前直接测，不用经过聊天。
3. 格式、限频、回复按下面的提案。

### 命令格式

- `#spin`：转一圈
- `#spin <圈数>`：转 1~2 圈；小于 1 按 1，大于 2 按 2，不是数字 → `"格式不对，是 #spin 或 #spin 2"`
- 只认主人（`owner_name` 精确匹配），和其他命令一样

### 执行

1. **限频**：距上次 `#spin` 不到 `spin.min_interval`（10 s）→ 回 `"刚转过，等 N 秒"`，不转
2. **画面黑着**（切场景）→ 回 `"画面黑着，转不了"`
3. 转之前：输入框开着先按 BACK；**聊天记录面板开着就先按 C 关掉**（沿用 `Camera._ready`；面板开着时方向键有没有反应还没测过）
4. **按住 → 方向键**转 `圈数 × spin.seconds_per_turn` 秒（一圈的时长默认按实测 0.5 s ≈ 90° 估成 2.0 s，**待标定**），
   按住期间按 `spin.fps`（15）连续截图，每张记下"按住后第几秒"；时间到松开。出错 / Ctrl+C 也要松开（同 `Camera._press`）
5. 转之后：等 0.3 s 镜头停稳再截一张"转完"的图；再按 C 重开聊天记录面板
6. 转满整圈，镜头朝向理论上回到原位；实际偏多少，看"转之前 / 转完"两张图对比（标定 `seconds_per_turn` 用）

这几秒（约 3~5 s）主循环被占住、读不到聊天；面板重开后照常读，消息不会丢（换轮盘时也是这样）。

### 存下来的东西

`runs/<这次运行>/spin/<时:分:秒>/`：
- `before.jpg`、`after.jpg`：转之前 / 转完
- `000_0.07s.jpg`、`001_0.13s.jpg` ……：转动中的每帧，文件名带按住后的秒数
- `summary.json`：圈数、按住时长、实际截了几张、实际帧率、面板有没有重开成功、转前转后两张图的差异（0~1，越小越接近原位）

命令行 `camera spin` 存到 `tmp/spin/<时:分:秒>/`，内容一样，结束时打印小结。

### 回复

- 成功：`"转完了，<秒数> 秒 <张数> 张"`，比如 `"转完了，1.9 秒 28 张"`
- 打开 YOLO（`[perception] enabled`）以后，后面再加一句扫描结果（感知层设计 §14.2），比如 `"前面懒洋洋大王，右后2个陌生人"`；
  超过 `reply.max_chars` 就截断
- 面板没重开成功：成功回复后面加 `"（聊天面板没打开，要手动按 C）"`

### dry-run

和 `#friend` / `#remember` 一样，`#spin` 是主人的测试操作，**dry-run 下照样转、照样存截图**（镜头只在自己屏幕上转，别人看不到，没有副作用）；
只有确认回复照 dry-run 的老规矩只打印不发送。

### 要改的地方

- `chat/commands.py`：`_KNOWN` 加 `spin`；`CommandRouter` 加一个可选回调 `spin: Callable[[int], str] | None`，
  没传（比如没有视角控制）时回 `"这次没开视角控制"`
- 普通模式现在没有 `Camera`（只在大脑模式里建）：`_run_agent` 也建一个，交给 `Agent`
- `brain/camera.py`：`Camera` 加 `spin(turns, seconds_per_turn, fps, capture) -> 帧列表 + 时间`；和 `around()` 一样在 `_ready()` 里做，
  转完不改 `offset`（整圈）
- `config.py` 新增 `[spin]`：`seconds_per_turn = 2.0`、`fps = 15.0`、`max_turns = 2`、`min_interval = 10.0`
- `cli.py`：`camera spin [--turns N] [--seconds S]`（`--seconds` 临时覆盖一圈时长，标定用）
- `runlog.py`：`RunDir` 加存 spin 结果的方法

### 边界情况

- 牵着手时转视角：只动镜头、不动角色，不会松手，照常执行
- 转的时候有人发起互动请求：这几秒不处理，转完后请求还在就照常接（`social.max_age` 10 s 内）
- 转的时候画面整屏黑（切场景）：照样转完、照样存，回复里加 `"（中途画面黑了）"`
- `#spin` 本身和确认回复都不进聊天历史（同其他命令）

### 测试

- `tests/test_commands.py`：`#spin` / `#spin 2` / `#spin 5`（按 2）/ `#spin abc`（格式错）/ 没有回调时的回复
- `tests/test_brain_camera.py`：`spin()` 按住右键、按 fps 截图、时间到松开；中途出错也松开；面板开着时先关后开；转完 `offset` 不变
- `tests/test_agent.py`：主人发 `#spin` → 调回调、回确认、不进 `pending`；限频内第二次回"刚转过"；dry-run 下照样调回调但不发送
