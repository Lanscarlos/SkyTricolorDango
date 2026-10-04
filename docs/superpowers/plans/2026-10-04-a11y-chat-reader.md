# 读聊天改用无障碍节点 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 读聊天从 OCR 截图改成读光遇镜像出来的无障碍节点：面板开着读面板行，面板关着读好友头顶气泡；读不到时整次 run 退回 OCR。

**Architecture:** `vision/a11yui.py` 把一份快照（`device/a11y.py` 的 `Snapshot`）纯计算地分成面板行 / 名字标签 / 气泡 / 输入栏；`chat/a11yreader.py` 的 `A11yChatReader` 接口同 `ChatReader`，管面板行对齐、气泡跟踪、去重；`FallbackReader` 包住它和 OCR 的 `ChatReader`，负责重启和退回。`PanelManager` 加一个「聊着（面板关着）」状态。

**Tech Stack:** Python 3.13，pytest（合成快照 + 假读取器，不用模拟器）。

**Spec:** `docs/superpowers/specs/2026-10-04-a11y-chat-reader-design.md`（先读 §1 实测事实和 §4 的对照表）

## Global Constraints

- 在 worktree 里干活；`python -m skydango` 要 `PYTHONPATH=src`（否则跑的是主目录的代码）；测试 `python -m pytest -q`（全量放后台跑，约 7 分钟）
- 中文注释 / docstring / 日志，照周围代码的密度和写法
- **公开仓库：测试里不许出现真实好友昵称和聊天原话**，用"小明""小红""团子"这类假名字
- 对外接口不变：身体 / Agent / PanelManager / viewer 照旧调用 `read(frame, now)`、`panel_visible(frame)`、`panel_closed_since`、`settling`、`trace_path`、`log_rows`、`detect`
- `vision.source = "ocr"` 或 `vision.mode != "log"` 时逐字照旧（不起读取器）；`[panel] mode = "always"` 时面板行为逐字照旧
- 光遇包名 `com.netease.sky.vivo`；坐标按 1920×1080 量，代码里换算成比例（乘以实际宽高）
- 改完合并进 main 并推送：`git fetch && git rebase origin/main && git push origin HEAD:main`

## Review Focus

1. **当前窗口不是光遇**（系统弹框，比如今天按出的"确定要退出游戏吗？"、游戏重启中）：不报消息、面板算关着、不崩 —— Task 2 / Task 3 有测试
2. **面板正在开 / 关**（2 秒动画里，同一份快照里既有面板行又有气泡）：已经报过的那句不能再报一次 —— Task 4 `test_bubble_not_rereported_across_panel_toggle`
3. **friends.md 的标题和游戏里的昵称对不上**：他的原文气泡挂不上名字，要触发 `want_peek()` 而不是丢掉 —— Task 4 `test_unattached_text_bubble_wants_peek`
4. **读取器反复死掉**（被别的工具抢连接）：最多重启 3 次就切 OCR，不能一直重启；重启期间面板开没开要照样答得上来 —— Task 5 有测试
5. **重新登录 / 换场景把面板历史清了**：当新基准，不能把整屏历史当新消息回复 —— Task 3 `test_rebaseline_when_nothing_matches`

---

### Task 1: `a11y --watch --save` 录原始快照

**Files:**
- Modify: `src/skydango/device/a11y.py`（`A11yReader.__init__` 加 `on_line`，`_pump` 调它）
- Modify: `src/skydango/cli.py`（`cmd_a11y` 和 `a11y` 子命令加 `--save`）
- Test: `tests/test_a11y.py`

**Interfaces:**
- Produces: `A11yReader(..., on_line: Callable[[bytes], None] | None = None)`：每收到一份**解析成功**的快照，用原始那一行（去掉行尾换行）调一次；回调出错只记 DEBUG。`python -m skydango a11y --watch 秒 --save 文件`：把原始 JSON 行追加进文件（UTF-8，一行一份）

- [ ] **Step 1: 写失败的测试** `test_on_line_gets_raw_snapshot_lines`：照 `test_watch_keeps_latest_and_stops` 的 `FakeProc`，喂 `[LINE + b"\n", b"garbage\n"]`，`on_line=got.append`；断言 `got == [LINE.encode("utf-8")]`
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_a11y.py -k on_line`，预期 FAIL（`unexpected keyword argument 'on_line'`）
- [ ] **Step 3: 实现** `on_line` 和 `--save`（`--save` 只在 `--watch` 时有意义，没给 `--watch` 就报错退出）
- [ ] **Step 4: 跑** `python -m pytest -q tests/test_a11y.py`，预期全过
- [ ] **Step 5: 提交** `feat(device): a11y --watch --save 存原始快照（回放测试用）`

---

### Task 2: 快照分类 `vision/a11yui.py`

**Files:**
- Create: `src/skydango/vision/a11yui.py`
- Create: `tests/a11ysnap.py`（合成快照的小工具，Task 2~5 共用）
- Test: `tests/test_a11yui.py`

**Interfaces:**
- Consumes: `skydango.device.a11y.Node`、`Snapshot`、`split_speaker`
- Produces（全部 frozen dataclass）：
  - `PanelRow(speaker: str, text: str, is_self: bool, masked: bool, box: tuple[int,int,int,int], visible: bool)`，方法 `key() -> tuple[bool, str, str]` = `(is_self, speaker, text)`；自己的行 `speaker = ""`；被屏蔽（内容全是点点）时 `text = ""`、`masked = True`
  - `Tag(name: str, box: tuple[int,int,int,int])`
  - `Bubble(text: str, box: tuple[int,int,int,int], speaker: str | None, typing_only: bool)`：`speaker=None` = 挂不上名字标签；`typing_only` = 全是点点和空格
  - `UiView(in_game: bool, panel_open: bool, rows: tuple[PanelRow, ...], tags: tuple[Tag, ...], bubbles: tuple[Bubble, ...], input_text: str | None)`：`input_text=None` = 输入栏没开
  - `GAME_PACKAGE = "com.netease.sky.vivo"`
  - `classify(snap: Snapshot | None, width: int, height: int, friends: Collection[str]) -> UiView`：`snap` 为 None 或包名不是光遇 → `in_game=False`、其余全空、`panel_open=False`
  - `bubble_key(text: str) -> str`：去掉末尾 `[ .]*` 再 `strip()`（`"怎么样.. "` → `"怎么样"`）
  - `is_dots(text: str) -> bool`：去掉空白后非空且全是 `.` / `。` / `…`

**分类规则**（spec §2.1；`s = width / 1920`，`v = height / 1080`）：
- 输入栏：宽 > 0.9·width 且 y1 > 0.85·height
- 面板占位：文字 `聊天……`、x1 < 0.1·width、y1 > 0.9·height、宽 < 0.15·width
- 别人的面板行：文字里有 `" - "`、x1 ≤ 30·s、x2 ≤ 0.335·width（看不看得见都要）
- `panel_open` = 占位看得见，或者有看得见的"别人的面板行"
- 自己的面板行（只在 `panel_open` 时）：x1 > 30·s、0.335·width − 60·s ≤ x2 ≤ 0.335·width、没有 `" - "`
- 其余**看得见**的节点是 3D 画面里的候选：高 40·v~46·v、不是点点、文字精确等于 `friends` 里某个名字 → `Tag`（同一列 x 中心差 ≤ 25·s 有两个都符合时，最上面那个是标签）；其余是 `Bubble`，挂到 x 中心差 ≤ 25·s、y1 比它小（在它上方）的名字标签上，有几个就取竖直距离最近的
- 节点顺序保持快照里的顺序（面板行从上到下）

`tests/a11ysnap.py` 提供（坐标照 spec §1）：`node(text, box, visible=True)`、`snap(*nodes, pkg=GAME_PACKAGE)`、`row(text, speaker, y, visible=True)` → 框 `(21, y, 21 + 20*len, y + 38)`、`self_row(text, y)` → `(560, y, 618, y + 40)`、`placeholder(visible=True)` → `(77, 1021, 154, 1051)`、`input_bar(text="聊天……")` → `(14, 962, 1906, 1032)`、`tag(name, cx, y)` → 高 42、宽 156、`bubble(text, cx, y, lines=1)` → 高 `38 if lines == 1 else 34*lines + 4`

- [ ] **Step 1: 写失败的测试**（`friends = {"小明", "小红"}`，`classify(..., 1920, 1080, friends)`）
  - `test_panel_rows_self_rows_and_placeholder`：别人的行（含滚出去的 `visible=False`）+ 被屏蔽行 `"..... - 陌生人"` + 自己的行 + 占位 → `panel_open`；`[r.key() for r in rows] == [(False,"小明","你好"), (False,"陌生人",""), (True,"","嗯")]`；被屏蔽那行 `masked`
  - `test_panel_closed_tags_and_bubbles`：没有占位、没有面板行；`tag("小明", 1053, 279)` 下面 `bubble("怎么样", 1053, 437)`、`bubble("团子", 1053, 498)`；另一列陌生人 `bubble(".....", 1387, 279)` → `panel_open is False`、`tags == (Tag("小明", …),)`、两个气泡 `speaker == "小明"`、陌生人气泡 `speaker is None and typing_only`
  - `test_bubble_saying_friend_name_is_not_a_tag`：小明的气泡里说了"小红"（高 38）→ 不是 `Tag`，是小明的气泡
  - `test_invisible_bubble_ignored`：画面外 `visible=False` 的气泡不出现在 `bubbles`
  - `test_input_bar`：`input_bar("测试")` → `input_text == "测试"`，它不算面板占位也不算气泡；没有输入栏 → `None`
  - `test_not_in_game`：`snap(..., pkg="android")` 和 `None` → `in_game is False`、`rows == () and bubbles == ()`
  - `test_bubble_key_and_dots`：`bubble_key("怎么样.. ") == "怎么样"`、`bubble_key("怎么样") == "怎么样"`、`is_dots(" ..")`、`not is_dots("好吧...")`、`not is_dots("")`
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_a11yui.py`，预期 FAIL（模块不存在）
- [ ] **Step 3: 实现** `vision/a11yui.py`（规则见上）
- [ ] **Step 4: 跑** 同上，预期全过
- [ ] **Step 5: 提交** `feat(vision): 无障碍快照分类（面板行 / 名字标签 / 头顶气泡 / 输入栏）`

---

### Task 3: `A11yChatReader` 面板行部分

**Files:**
- Create: `src/skydango/chat/a11yreader.py`
- Modify: `src/skydango/chat/reader.py`（`Message` 加 `source: str = ""`，放在最后一个字段）
- Test: `tests/test_a11yreader.py`

**Interfaces:**
- Consumes: Task 2 的 `classify`、`PanelRow`、`UiView`；`chat.reader.Message`、`Detection`；`vision.chatlog.LogRow`；`vision.bubbles.Rect`；`chat.tracker.SelfFilter`
- Produces:
  - `A11yChatReader(source: Callable[[], Snapshot | None], friends: Callable[[], list[str]], chat: ChatConfig, ocr_cfg: OcrConfig, self_filter: SelfFilter)`
  - `read(frame, now: float) -> list[Message]`：`frame` 只用来取宽高（`frame.shape[:2]`），None 时按 1920×1080；面板行报出的 `Message(text, Rect(l, t, r - l, b - t), now, speaker, source="panel")`（`(l, t, r, b)` 是行框）
  - `panel_visible(frame=None) -> bool`：现取一份快照分类
  - 属性 `panel_closed_since: float | None`（`read()` 里维护：看不到面板时记下开始时间，看到了清成 None；刚建好时是 None）、`settling`（恒 False）、`reads_bubbles`（恒 True）、`trace_path: Path | None`
  - `log_rows(frame=None) -> list[LogRow]`（看得见的面板行，`LogRow(speaker, text, is_self, Rect)`）、`detect(frame=None) -> list[Detection]`（`[Detection(r.display(), r.box) for r in log_rows()]`）
  - `view(frame=None) -> UiView`：最近一次 `read()` 分好类的结果（Task 4、6 用）

**行为**（spec §3.1）：
- 比较键 `PanelRow.key()` 精确相等；对齐：在新一列 `cur` 里找下标 `j`，使 `cur[j-k] == prev[-1-k]` 连续成立的 `k` 最多（并列取最大的 `j`），`j` 之后是新行；`prev` 为空或一个都对不上 → `prev = cur`、不报（对不上时记 INFO「面板历史对不上，当新基准」）
- 面板没开 / 不在游戏里：不动 `prev`，返回 `[]`
- 报之前过滤：`is_self`、`masked`、`len(text.replace(" ", "")) < ocr_cfg.min_chars`、`chat.ignore_patterns`、`self_filter.is_self(text, now)`
- `trace_path` 设了时，面板行（看得见的）有变化就照 `ChatReader._trace` 的格式追加一笔（`*` 标新行）

- [ ] **Step 1: 写失败的测试**（假 source：`snaps = [...]; source = lambda: snaps[0]`，每步换 `snaps[0]`；`SelfFilter(30, 0.8)`；`ChatConfig()`、`OcrConfig()` 默认值）
  - `test_first_snapshot_is_baseline`：第一次 `read` 有三行历史 → `[]`
  - `test_new_rows_reported`：再来一份多一行 `"在吗 - 小明"` → 一条 `Message`，`speaker == "小明"`、`text == "在吗"`、`source == "panel"`
  - `test_same_line_twice_reported_twice`：`prev = [a, 哈哈]`，`cur = [a, 哈哈, 哈哈]` → 报一条"哈哈"；再来 `[a, 哈哈, 哈哈, 哈哈]` → 又报一条
  - `test_top_rows_scrolled_away`：`prev = [a, b, c]`、`cur = [b, c, d]` → 只报 `d`
  - `test_rebaseline_when_nothing_matches`：`prev = [a, b]`、`cur = [x, y, z]` → `[]`；再来 `[x, y, z, w]` → 只报 `w`
  - `test_rows_added_while_panel_closed`：开着 `[a]` → 关着（没有面板行和占位）→ 再开 `[a, b, c]` → 报 `b`、`c`
  - `test_self_masked_and_filtered_rows_not_reported`：新增自己的行、被屏蔽行、`self_filter.remember("我刚说的", now)` 后出现的 `"我刚说的 - 小明"` → 都不报
  - `test_panel_closed_since`：面板开 → `None`；关 → 等于关的那次 `now`；`panel_visible()` 跟着变
  - `test_not_in_game_reads_nothing`：包名不是光遇 → `[]`、`panel_closed_since` 不是 None、下一份正常快照照常对齐（不当新基准）
  - `test_log_rows_and_detect`：`detect()` 给出 `"小明：你好"` 这样的显示文字
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_a11yreader.py`，预期 FAIL
- [ ] **Step 3: 实现** 面板行部分 + `Message.source`
- [ ] **Step 4: 跑** `python -m pytest -q tests/test_a11yreader.py tests/test_chatlog.py tests/test_agent.py`，预期全过（老测试不受 `Message` 新字段影响）
- [ ] **Step 5: 提交** `feat(chat): A11yChatReader 读面板行（精确对齐，不用 OCR）`

---

### Task 4: `A11yChatReader` 头顶气泡、去重、打字、看一眼

**Files:**
- Modify: `src/skydango/chat/a11yreader.py`
- Test: `tests/test_a11yreader.py`、`tests/test_a11y_replay.py`（新，本机录像回放，文件不在就跳过）

**Interfaces:**
- Consumes: Task 2 的 `Bubble`、`bubble_key`、`is_dots`；Task 3 的 `A11yChatReader`
- Produces（`A11yChatReader` 新增）：
  - `tags_in_view() -> list[str]`：最近一次 `read()` 看到的名字标签
  - `typing() -> list[str]`：最近一次 `read()` 里头上有"只有点点的气泡"或"最新一条气泡末尾挂着点"的好友
  - `want_peek() -> str | None`：有挂不上名字的原文气泡时返回 `"bubble_text"`，取一次就清掉
  - 气泡报出的 `Message(text, Rect(l, t, r - l, b - t), now, speaker, source="bubble")`（气泡框）

**行为**（spec §3.2、§3.3）：
- 面板没开时：按说话人统计当前气泡的 `Counter(bubble_key(text))`（不算 `typing_only`）；某个键的次数比上一份多 → 新消息，`text` 用这个键**第一次出现时**的原文 `strip()`（记在 `{(说话人, 键): 原文}` 里，键从气泡上消失时删掉）；照样过 `min_chars`、`ignore_patterns`、`self_filter`
- 第一份快照、以及面板开着的每一份：只更新"见过"的计数，不报
- `speaker is None` 的气泡：`typing_only` 的不管；有原文的先过 `self_filter.is_self`，是自己的丢掉，否则记一次 `want_peek`（同一句只记一次）
- 气泡报出去的进 `reported`：`deque` 里放 `(speaker, text, now)`，留 180 秒；面板行报之前，`reported` 里有说话人、原文（都 `strip()`）相同的就删掉那一条、这一行不报
- `rows.log`：气泡报出的消息也追加一行 `气泡 小明：怎么样`

- [ ] **Step 1: 写失败的测试**（面板关着的快照：`tag("小明", 1053, 279)` + 若干 `bubble(...)`）
  - `test_bubble_new_message`：先 `[tag]`，再 `[tag, bubble("怎么样")]` → 一条，`speaker == "小明"`、`source == "bubble"`
  - `test_typing_animation_not_new`：`"怎么样"` → `"怎么样.. "` → `"怎么样..."` → `"怎么样"`：只在第一次报；`typing()` 在带点时是 `["小明"]`
  - `test_dots_only_bubble_is_typing`：`[tag, bubble(" ..")]` → 不报、`typing() == ["小明"]`
  - `test_same_bubble_twice`：先一个"哈哈"，再两个"哈哈" → 第二次报一条
  - `test_first_snapshot_bubbles_not_reported`：第一份就有气泡 → 不报
  - `test_unattached_text_bubble_wants_peek`：没有名字标签的列里 `bubble("在吗")` → 不报，`want_peek() == "bubble_text"`，再取 → `None`；同一句下一份不再记
  - `test_own_bubble_dropped`：`self_filter.remember("测试", now)`，没挂名字的 `"测试"` 气泡 → 不报、不 `want_peek`
  - `test_panel_open_ignores_bubbles`：面板开着时冒出新气泡 → 不报；面板关上后同一个气泡还在 → 也不报
  - `test_bubble_then_panel_row_deduped`：关着时气泡报了"怎么样"，再开面板对出新行 `"怎么样 - 小明"` → 不报；面板后来又多一行同样的 → 报（一条抵一条）
  - `test_bubble_not_rereported_across_panel_toggle`：同一份快照里面板行和气泡同时有"怎么样"（动画中）→ 面板开 / 关来回两次，总共只报一次
  - `test_tags_in_view`：`tags_in_view() == ["小明"]`
  - `tests/test_a11y_replay.py::test_replay_local_recording`：`tmp/a11y-live/raw.jsonl` 不在就 `pytest.skip`；在就逐行 `parse_line` 喂给 `A11yChatReader`（friends 用本机 `config.toml` 的记忆目录取，取不到就 skip），断言不抛异常、每条消息 `speaker` 非空、`source in ("panel", "bubble")`，并打印读到的消息（`-s` 时看）
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_a11yreader.py tests/test_a11y_replay.py`，预期新测试 FAIL
- [ ] **Step 3: 实现**（规则见上）
- [ ] **Step 4: 跑** 同上，预期全过（回放测试在没有录像时是 skipped）
- [ ] **Step 5: 提交** `feat(chat): 面板关着时读好友头顶气泡（认说话人、打字、和面板去重）`

---

### Task 5: `FallbackReader`、配置、接进 run / view

**Files:**
- Modify: `src/skydango/chat/a11yreader.py`（加 `FallbackReader`）
- Modify: `src/skydango/config.py`（`VisionConfig.source`、`PanelConfig.chat_peek`）、`config.example.toml`
- Modify: `src/skydango/cli.py`（`_chat_reader`、`_stop_reader`；`_game_world`、`_run_agent`、`cmd_view` 改用它并在收尾时停）
- Modify: `src/skydango/console/settings.py`（加 `vision.source`）
- Test: `tests/test_a11yreader.py`、`tests/test_cli_brain.py`（建 reader 的 monkeypatch 照着补）、`tests/test_config.py`、`tests/test_console_settings.py`

**Interfaces:**
- Consumes: `device.a11y.A11yReader`（`start/stop/latest(max_age)/alive/error`）、Task 3~4 的 `A11yChatReader`、OCR 的 `ChatReader`
- Produces:
  - `VisionConfig.source: str = "a11y"`（`"a11y"` / `"ocr"`）；`PanelConfig.chat_peek: float = 15.0`
  - `FallbackReader(client, a11y: A11yChatReader, ocr: ChatReader)`，常量 `RESTART_DELAYS = (2.0, 5.0, 10.0)`、`STALE = 30.0`、`MAX_AGE = 2.0`
  - `start() -> None`：`client.start()`，抛异常就 `_to_ocr(原因)`
  - `read(frame, now) -> list[Message]`、`panel_visible(frame) -> bool`、属性 `panel_closed_since`、`settling`、`reads_bubbles`（OCR 模式下 False）、`trace_path`（setter 同时设给两个 reader）；`tags_in_view()` / `typing()` / `want_peek()`（OCR 模式下 `[]` / `[]` / `None`）；`log_rows()` / `detect()` 转给当前的 reader
  - `using_ocr: bool`、`reason: str`、`describe() -> str`：`"无障碍"` 或 `"OCR（无障碍读不到：<原因>）"`
  - `stop() -> None`：`client.stop()`，出错只记日志
  - `cli._chat_reader(cfg, dev) -> tuple[reader, SelfFilter]`：`source == "a11y"` 且 `vision.mode == "log"` 时建 `A11yReader(cfg.device.adb_path, cfg.device.serial)` + `A11yChatReader(lambda: client.latest(max_age=MAX_AGE), _friend_names(cfg), …)` + `FallbackReader` 并 `start()`；否则就是 `_build_reader(cfg)`；`source` 不是这两个值 → `ValueError`
  - `cli._stop_reader(reader) -> None`：有 `stop` 就调、吞异常记日志
  - 管理面板 `Field("vision.source", "读聊天的方式", "a11y：读游戏的无障碍节点（准、面板关着也能读好友头顶气泡），读不到自动退回 OCR；ocr：截图识别（原来的做法）", "choice", "features", ("a11y", "ocr"))`

**行为**（spec §5）：
- 健康：`client.latest(max_age=MAX_AGE)` 拿得到就是正常，记下 `_good_at = now`；`read()` 照常交给 `a11y.read`
- 客户端死了（`not client.alive`）：记死的时间；到 `死的时间 + RESTART_DELAYS[重启次数]` 再 `client.start()`；重启次数用完、又死了 → `_to_ocr(client.error 或 "客户端反复退出")`；重启后收到快照 → 次数清零
- 客户端活着但 `now - _good_at > STALE` → `_to_ocr("30 秒没收到快照")`
- 不健康期间：`read()` 返回 `[]`；`panel_closed_since` 按 `ocr.panel_visible(frame)` 维护（`frame` 为 None 时保持原值）
- `_to_ocr(原因)`：WARNING「读聊天退回 OCR：<原因>」、`client.stop()`、之后一切交给 `ocr`（它从空基准开始）
- `_game_world` 的 `close()`、`_run_agent` 的 `finally`、`cmd_view`（实时，`dev` 不为 None）的 `finally` 都 `_stop_reader(reader)`；`cmd_detect`、`panels`、`camera spin`、`friend-check` 照旧用 `_build_reader`（它们要 OCR 或只要 `panel_visible(frame)` 看静态图）

- [ ] **Step 1: 写失败的测试**（假 client：`alive` / `error` / `latest()` 可设，`start()` 计数、可设成抛异常；假 OCR reader：`read` 返回固定列表、`panel_visible` 返回可设的值）
  - `test_start_failure_falls_back_to_ocr`：`start()` 抛 `A11yError("already registered")` → `using_ocr`、`"already registered" in describe()`、`read()` 返回 OCR 的结果
  - `test_restart_then_recover`：活着 → 死了（`now=10`）→ `now=11` 不重启 → `now=12` 重启（`start` 次数 2）→ 收到快照 → 再死、再按 2 秒重启（次数清零了）
  - `test_three_failed_restarts_switch_to_ocr`：死 → 重启 → 没快照又死 …… 第 3 次重启后又死 → `using_ocr`；之后不再 `start()`
  - `test_stale_switches_to_ocr`：`alive` 但 `latest()` 一直 None，`now` 从 0 到 31 → `using_ocr`、`"30 秒" in reason`
  - `test_unhealthy_uses_ocr_panel_visibility`：死着等重启时 `read(frame, now)` 返回 `[]`，`panel_closed_since` 跟着假 OCR 的 `panel_visible` 变
  - `test_trace_path_propagates`、`test_stop_calls_client_stop`、`test_ocr_mode_has_no_bubbles`（`reads_bubbles is False`、`want_peek() is None`）
  - cli：`test_chat_reader_ocr_source_is_plain_chatreader`（`vision.source = "ocr"` → 返回 `ChatReader`、没起客户端）；`test_chat_reader_rejects_unknown_source`
  - 配置：`config.example.toml` 写上 `source = "a11y"`（`[vision]`）和 `chat_peek = 15.0`（`[panel]`）并加注释；跑现有的"示例配置和默认值一致"类测试
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_a11yreader.py tests/test_config.py tests/test_cli_brain.py tests/test_console_settings.py`，预期新测试 FAIL
- [ ] **Step 3: 实现** `FallbackReader`、配置项、`_chat_reader` / `_stop_reader`、三处接线、设置清单
- [ ] **Step 4: 跑** 同上，预期全过；再后台跑全量 `python -m pytest -q`
- [ ] **Step 5: 真机冒烟**（模拟器开着时；不往游戏里发输入）：`PYTHONPATH=src python -m skydango view --no-browser`（Git Bash；PowerShell 里先 `$env:PYTHONPATH="src"`）跑 20 秒后 Ctrl+C，日志里没有"退回 OCR"；停下后 `adb shell "ps -A | grep skydango"` 没有输出
- [ ] **Step 6: 提交** `feat(chat): 读聊天默认走无障碍节点，读不到自动退回 OCR（vision.source）`

---

### Task 6: 面板「聊着（面板关着）」状态 + 身体不再拿 YOLO 气泡叫面板

**Files:**
- Modify: `src/skydango/chat/panel.py`
- Modify: `src/skydango/brain/body.py`（两处 `typing_seen` → `bubble_seen`；status 加"在打字"；status 里"读聊天：…"）
- Modify: `CLAUDE.md`（「聊天面板」一节、代码结构表、`[vision] source`）、`docs/game-ops.md` §3（a11y 已接进读聊天、`talking` 状态）
- Test: `tests/test_panel.py`（照现有面板测试的假 reader / 假设备写法）、`tests/test_brain_body.py`（typing 相关）

**Interfaces:**
- Consumes: reader 的 `reads_bubbles`、`tags_in_view()`、`want_peek()`、`typing()`（OCR 的 `ChatReader` 没有这些属性：一律 `getattr(self.reader, name, 默认)`，默认 False / `[]` / `None`）；`Message.source`；`PanelConfig.chat_peek`
- Produces: `PanelManager.state` 多一个值 `"talking"`；`describe()` 在 talking 时返回 `"聊着（面板关着，N 秒后看一眼）"`

**行为**（spec §4 对照表；只在 `self.auto` 且 `reads_bubbles` 为真时生效，否则逐字照旧）：
- `tick(now, fresh, visible, blackout)`：`fresh` 非空且全是 `source == "bubble"`：state 是 `chatting` 就照旧刷新活动时间；否则进 `talking`（`_set("talking", "从头顶气泡读到消息")`），`_talk_with |= {m.speaker for m in fresh}`、`_last_activity = now`，不按键；有 `source != "bubble"` 的照旧进 `chatting`
- 每圈（auto 且 `reads_bubbles`）先取 `want_peek()`，有原因就 `trigger(原因, now)`
- `trigger()` 在 `idle` 和 `talking` 都能记 `_pending`
- `_tick_talking(now, visible, blackout)`：`visible`（面板被打开了、不是自己看一眼开的）→ `chatting`；`not reads_bubbles`（中途退回 OCR）→ `chatting` 并 `ensure_open()`；`now - _last_activity >= quiet_close` → `idle`（清 `_talk_with`）；否则到了 `chat_peek`（`now - _last_read >= chat_peek`）或有 `_pending` 且过了 `peek_cooldown`、不黑屏 → `_open_peek`
- 看一眼从 `talking` 发起：结束时（没读到新消息、关面板）回到 `talking`，`quiet_close` 已过就回 `idle`；读到面板行的新消息照旧进 `chatting`（`_open_peek` 记下发起时的状态，`_tick_peek` 关面板时用它）
- `before_speak(now)`：state 是 `talking` 且 `_talk_with` ⊆ `set(reader.tags_in_view())` → 只刷新 `_last_activity`，不开面板；否则照旧
- `bubble_seen()` 在 `talking` 时什么都不做；`should_be_open()` 在 `talking` 时 False
- 身体：`reads_bubbles` 为真时两处 `typing_seen` 都不调 `panel.bubble_seen`；status 在面板那一项后面加 `在打字：小明`（`typing()` 非空时），以及 `读聊天：<describe()>`（reader 有 `describe` 时）

- [ ] **Step 1: 写失败的测试**（假 reader：`reads_bubbles`、`tags`、`peek` 可设；`fresh` 用 `Message(..., source="bubble")`）
  - `test_bubble_message_enters_talking_without_key`：idle 时 `tick(fresh=[小明·bubble])` → `state == "talking"`、设备没收到按键
  - `test_panel_message_still_enters_chatting`：`source="panel"` → `chatting`
  - `test_talking_peeks_every_chat_peek`：talking 里 `now` 推进到 `chat_peek` → 按键看一眼；看完没新消息 → 回 `talking`
  - `test_talking_goes_idle_after_quiet_close`
  - `test_before_speak_keeps_panel_closed_when_partner_in_view`：talking、`tags=["小明"]` → 不按键、仍 talking；`tags=[]` → 开面板、`chatting`
  - `test_want_peek_triggers_peek`：idle，`peek = "bubble_text"` → 下一圈按键看一眼
  - `test_fallback_to_ocr_while_talking_opens_panel`：talking 中 `reads_bubbles` 变 False → `chatting` 并按键开面板
  - `test_ocr_reader_unchanged`：用没有这些属性的 reader（比如现有测试的假 reader），`fresh` 任意 → 行为同改之前（`chatting`）
  - `test_always_mode_unchanged`：`mode = "always"` 时 bubble 消息也不进 talking
  - body：`test_typing_seen_ignored_when_reader_reads_bubbles`（照现有 `typing_seen` 测试改：reader `reads_bubbles=True` → `panel.bubble_seen` 没被调）
- [ ] **Step 2: 跑** `python -m pytest -q tests/test_panel.py tests/test_brain_body.py tests/test_brain_reflex.py tests/test_agent.py`，预期新测试 FAIL
- [ ] **Step 3: 实现** 面板状态机、身体两处、status；更新 CLAUDE.md 和 game-ops
- [ ] **Step 4: 跑** 同上全过；后台跑全量 `python -m pytest -q`，预期全过
- [ ] **Step 5: 提交** `feat(panel): 跟画面里的好友聊天时面板关着（talking），说话的人不在画面里才开`
- [ ] **Step 6: 合并推送**：`git fetch && git rebase origin/main`，再跑一次相关测试，`git push origin HEAD:main`；真机验证按 spec §8 晚上做，结果补进 game-ops 和 spec 状态行
