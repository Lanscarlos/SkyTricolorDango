# SkyTricolorDango 🍡

一个“自己玩光遇”的 Agent 原型。第一个能力是**在游戏里自动陪聊**：看到别人说话，用大模型生成一句回复，打字发出去。

- 运行环境：**MuMu 模拟器**（adb 控制）+ 电脑上的 Python
- 纯视觉：只看画面、只模拟输入，**不读内存、不注入、不改游戏、不抓包**

> ⚠️ 光遇国际服与国服的用户协议都禁止脚本 / 自动化工具，使用有封号风险，建议用小号。
> 默认会在每句回复前加上 `【AI】`，让对方知道是 AI 在说话。光遇里未成年玩家很多，请保留这个标识。

## 工作原理

```
MuMu 画面 ──adb screencap──▶ 气泡检测（颜色阈值 + 形状）──▶ OCR（RapidOCR）
                                                              │
                                                    去重 / 过滤自己的话
                                                              │
MuMu 输入框 ◀──ADBKeyboard 中文输入 + 发送── 限速 ◀── 大模型生成回复
```

| 模块 | 位置 | 说明 |
|---|---|---|
| 设备层 | `src/skydango/device/` | adb 截屏（raw 格式，比 PNG 快）、点击、ADBKeyboard 中文输入 |
| 视觉 | `src/skydango/vision/` | 找浅色圆角气泡 → 裁剪 → OCR，不需要训练模型 |
| 聊天 | `src/skydango/chat/` | 模糊去重、忽略自己的气泡、人设与安全规则、发送流程 |
| 主循环 | `src/skydango/agent.py` | 攒几秒合并连发消息，限速后回复；默认 dry-run |

## 快速开始

### 1. 准备 MuMu

1. MuMu 12 默认开启 adb，第一个实例地址是 `127.0.0.1:16384`（多开第 N 个是 `16384 + 32*N`，多开器里能看到）。
2. 下载 [ADBKeyboard](https://github.com/senzhk/ADBKeyBoard/releases)（`keyboardservice-debug.apk`），拖进 MuMu 安装。它用来输入中文，`adb shell input text` 不支持中文。
3. 建议把模拟器分辨率固定下来（比如 1920×1080 横屏），按钮位置和气泡参数都依赖分辨率。

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

skydango shot --grid              # 截一张带 0~1 坐标网格的图 → 在 config 里填 sender.open_chat 等坐标
skydango detect                   # 对当前画面跑一次气泡检测 + OCR，输出标注图 detect.png
skydango detect 某张截图.png       # 也可以对存下来的截图调参

skydango say "测试一下"            # 只测发送流程：点开聊天框 → 输入 → 提交
skydango chat                     # 不开游戏，在终端里和人设对话，调提示词（--echo 不调模型）

skydango run                      # 启动 Agent，dry-run：只打印“将会发送”
skydango run --live               # 真的发送
```

## 需要你用真实截图调的地方

这些参数是按“浅色气泡 + 深色文字”的一般假设写的，还没有用光遇的真实画面校准过：

- `vision.bubble.*`：气泡颜色阈值、尺寸、填充率。用 `skydango detect` 看标注图，红框框住气泡、没框住云和雪地就对了。
- `vision.roi`：只在这块区域里找气泡，排除底部操作栏和顶部 UI。
- `sender.*`：打开聊天框的按钮位置、提交方式（输入法发送动作 / 回车 / 点发送按钮）。
- 气泡检测效果不好时，可以先把 `vision.mode` 改成 `"roi"`，直接对整块区域做 OCR。

## 开发

```bash
pip install -e ".[ocr,dev]"
pytest
```

测试用合成画面覆盖了气泡检测、去重、限速、发送流程；装了 OCR 时还会跑一次真实 RapidOCR 的端到端识别。

## 路线图

- [x] v0.1 聊天闭环：截屏 → 气泡 → OCR → 大模型 → 发送
- [ ] 用真实截图校准气泡参数，补充光遇界面样本
- [ ] 区分说话人（气泡位置 ↔ 角色位置）、识别自己的气泡
- [ ] MuMu 原生截图接口（比 adb 更快）
- [ ] 自动弹琴、固定 UI 操作（点蜡烛、收发爱心）
- [ ] 跑图：场景识别定位 + 路线回放
