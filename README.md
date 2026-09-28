# SkyTricolorDango 🍡

一个“自己玩光遇”的 Agent 原型。第一个能力是**在游戏里自动陪聊**：看到别人说话，用大模型生成一句回复，打字发出去。

- 运行环境：**MuMu 模拟器**（adb 控制）+ 电脑上的 Python
- 纯视觉：只看画面、只模拟输入，**不读内存、不注入、不改游戏、不抓包**

> ⚠️ 光遇国际服与国服的用户协议都禁止脚本 / 自动化工具，使用有封号风险，建议用小号。
> 默认会在每句回复前加上 `【AI】`，让对方知道是 AI 在说话。光遇里未成年玩家很多，请保留这个标识。

## 工作原理

```
MuMu 画面 ──adb screencap──▶ 聊天记录面板（按 C 打开）──▶ OCR（RapidOCR）
                                                              │
                                        按行对齐找新消息 / 拆出说话人 / 跳过自己和被屏蔽的
                                                              │
MuMu 输入框 ◀──模拟实体键盘 Enter 打开 + ADBKeyboard 中文输入 + 发送── 限速 ◀── 大模型生成回复
```

| 模块 | 位置 | 说明 |
|---|---|---|
| 设备层 | `src/skydango/device/` | adb 截屏（raw 格式，比 PNG 快）、点击、`sendevent` 模拟实体键盘、ADBKeyboard 中文输入 |
| 视觉 | `src/skydango/vision/` | 解析聊天记录面板（`chatlog.py`）；备选：找头顶气泡（`bubbles.py`）。不需要训练模型 |
| 聊天 | `src/skydango/chat/` | 模糊去重、忽略自己的气泡、人设与安全规则、发送流程 |
| 主循环 | `src/skydango/agent.py` | 攒几秒合并连发消息，限速后回复；默认 dry-run |
| 统管大脑（可选） | `src/skydango/brain/` | `run --brain`：常驻的 Claude Code（订阅）当大脑，眼睛（Haiku）把画面写成文字，经本机 MCP 调身体的工具 |

## 快速开始

### 1. 准备 MuMu

1. MuMu 12 默认开启 adb，第一个实例地址是 `127.0.0.1:16384`（多开第 N 个是 `16384 + 32*N`，多开器里能看到）。
2. 下载 [ADBKeyboard](https://github.com/senzhk/ADBKeyBoard/releases)（`keyboardservice-debug.apk`），拖进 MuMu 安装。它用来输入中文，`adb shell input text` 不支持中文。
3. 建议把模拟器分辨率固定下来（比如 1920×1080 横屏），面板区域等参数都依赖分辨率。
4. 键位方案用「PC端操作方案」：它把键盘原样透传给游戏，游戏里 Enter 打开聊天框、C 打开聊天记录。
   注意 `adb shell input keyevent` 发出的虚拟按键游戏不认，所以代码用 `sendevent` 直接写 MuMu 的键盘设备（`/dev/input/event*`）。
5. MuMu 的 adb 设备名可能是 `emulator-5554` 而不是 `127.0.0.1:16384`，以 `adb devices` 为准。

### 2. 安装

需要 Python 3.11+。

```bash
pip install -e ".[ocr,openai]"        # 用 Claude 的话再加 anthropic
copy config.example.toml config.toml  # macOS / Linux 用 cp
```

`config.toml` 里把 `device.adb_path` 指向 MuMu 自带的 adb（`MuMu 安装目录\shell\adb.exe`）更稳。大模型默认用 OpenAI 兼容接口（示例是 DeepSeek），Key 放在环境变量里：

```bash
set DEEPSEEK_API_KEY=sk-...           # PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
```

### 3. 逐步联调

```bash
skydango devices                  # 能连上、能截图、看当前输入法
skydango ime on                   # 切到 ADBKeyboard（ime off 恢复默认输入法）

skydango shot --grid              # 截一张带 0~1 坐标网格的图，用来量 vision.log_roi 等区域
skydango detect                   # 对当前画面读一次聊天记录面板（先在游戏里按 C），输出标注图 tmp/detect.png
skydango detect 某张截图.png       # 也可以对存下来的截图调参

skydango view                     # 只看不动：浏览器里实时看画面 + 识别框（好友名字、陌生人、互动圆圈、新消息）
skydango view --images tmp/record/某次  # 回放 record 录的图，不用连模拟器

skydango say "测试一下"            # 只测发送流程：（输入框没开时）按 Enter → 输入 → 提交
skydango chat                     # 不开游戏，在终端里和人设对话，调提示词（--echo 不调模型）

skydango run                      # 启动 Agent，dry-run：只打印“将会发送”
skydango run --live               # 真的发送
skydango run --no-emotes          # 不做表情动作（牵着手时用：做动作会松开牵手）
skydango run --view               # 跑的同时开可视化网页（http://127.0.0.1:8765/）

skydango look                     # 截一张图让眼睛（Claude Haiku）描述（要先 claude setup-token、设 SKYDANGO_CLAUDE_TOKEN）
skydango run --brain              # 统管大脑，dry-run：只打印它想说想做的
```

## 需要你用真实截图调的地方

默认值是在 1920×1080 的 MuMu 上用真实画面标定的，换分辨率或界面有变化时再调：

- `vision.log_roi`：聊天记录面板的区域，底部要避开面板下面的输入框。`skydango detect` 的输出里每行应该是“说话人：内容”“[我] ……”或“陌生人：（被屏蔽）”。
- `vision.log_self_min_value`：文字框背景亮度高于它算自己发的（浅色气泡），别人的消息是深色底。
- `sender.*`：打开聊天框的按键（`open_chat_key`，28 = Enter）、提交方式（输入法发送动作 / 回车 / 点发送按钮）。

光遇不会显示没解锁聊天的陌生人说的话（面板和头顶气泡里都只有省略号），所以 Agent 只能和解锁了聊天的好友、或坐在长椅上的人对话。

`vision.mode = "bubble"` 是在 3D 画面里找头顶气泡的旧方案：气泡半透明，容易和白色角色、光效粘在一起，只作为备选。

## 开发

```bash
pip install -e ".[ocr,dev]"
pytest
```

测试用合成画面覆盖了聊天记录解析、前后帧对齐、气泡检测、去重、限速、发送流程；装了 OCR 时还会跑一次真实 RapidOCR 的端到端识别。

- [AGENTS.md](AGENTS.md)：代码结构、常用命令、工作约定（给接手的 Agent / 开发者）
- [docs/game-ops.md](docs/game-ops.md)：光遇 × MuMu 的实测操作手册——按键、界面坐标、聊天记录格式、动作轮盘编辑、踩过的坑

快捷动作轮盘（数字键 1~8 触发）：

```bash
skydango emotes scan              # 截下动作列表所有图标到 emotes/scan/，把想用的改名放到 emotes/（如 emotes/鞠躬.png）
skydango emotes wheel             # 读轮盘 8 格
skydango emotes set 5 鞠躬         # 换格子（3、8 默认锁定）
skydango emotes do 鞠躬            # 做动作，不在轮盘上会先换上去
```

聊天时做动作：`[emotes]` 配置。轮盘上（除锁定格外）的动作模型都能用；`extra` 里的动作会在需要时换进 `swap_slots`，退出时换回去。
轮盘上的动作要在图标库里有命名好的图标（启动时读一次轮盘靠它们认）。

## 路线图

- [x] v0.1 聊天闭环：截屏 → 气泡 → OCR → 大模型 → 发送
- [x] 读聊天记录面板：区分说话人、识别自己的消息和被屏蔽的消息
- [ ] 滚动聊天记录面板，启动时读取更早的上下文
- [x] MuMu 原生截图接口（约 9 ms/张，adb 约 400 ms）；面板文字没变时跳过 OCR
- [ ] 自动弹琴、固定 UI 操作（点蜡烛、收发爱心）
- [ ] 跑图：场景识别定位 + 路线回放
