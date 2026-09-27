"""配置加载。所有字段都有默认值，config.toml 只需要写想覆盖的部分。"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 坐标约定：配置里的坐标一律用 0~1 的归一化值（相对截图宽高），
# 这样换分辨率 / 换模拟器实例不用重新标定。


@dataclass
class DeviceConfig:
    adb_path: str = "adb"
    # MuMu 12 第一个实例默认 127.0.0.1:16384，多开实例依次 +32；旧版 MuMu 6 是 127.0.0.1:7555
    serial: str = "127.0.0.1:16384"
    auto_connect: bool = True
    adb_timeout: float = 10.0
    # ADBKeyboard 输入法，用来输入中文（adb shell input text 不支持中文）
    ime_id: str = "com.android.adbkeyboard/.AdbIME"
    # 模拟实体键盘用的输入设备，例如 "/dev/input/event4"；为空自动找带 KEY_ENTER 的设备
    key_device: str = ""
    # 截图方式："auto"（找得到 MuMu 的 external_renderer_ipc.dll 就用 MuMu 原生截图，约 9 ms；
    # 否则 adb screencap，约 400 ms）/ "mumu" / "adb"。MuMu 的安装目录从 adb_path 推出来
    capture: str = "auto"
    mumu_instance: int = 0  # MuMu 多开时的实例编号


@dataclass
class BubbleConfig:
    """聊天气泡检测参数：浅色、低饱和、近似圆角矩形、里面有深色文字。"""

    min_value: int = 215  # HSV 亮度下限（0~255）
    max_saturation: int = 45  # HSV 饱和度上限（0~255）
    ink_value: int = 140  # 低于此亮度的像素视为“文字墨迹”
    min_width: float = 0.03  # 以下尺寸均相对截图宽/高
    min_height: float = 0.025
    max_width: float = 0.6
    max_height: float = 0.25
    min_fill: float = 0.7  # 轮廓面积 / 外接矩形面积，圆角矩形约 0.85~0.95
    min_ink: float = 0.01  # 墨迹像素占比范围，过滤纯白的云、雪地
    max_ink: float = 0.5
    padding: int = 4  # 送去 OCR 前往外扩的像素


@dataclass
class VisionConfig:
    # 截屏间隔（秒）。MuMu 原生截图约 9 ms，面板没变时不跑 OCR，所以可以看得很勤；用 adb 截图时一轮约 0.8 s
    poll_interval: float = 0.15
    # "log"：读聊天记录面板（光遇按 C 打开，推荐）；"bubble"：找头顶气泡再 OCR；"roi"：直接对整块区域 OCR
    mode: str = "bubble"
    # 搜索区域 [x1, y1, x2, y2]（归一化），默认排除底部操作栏
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.85])
    bubble: BubbleConfig = field(default_factory=BubbleConfig)
    # 聊天记录面板所在区域 [x1, y1, x2, y2]（归一化），底部要避开面板下面的输入框
    log_roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.335, 0.855])
    # 打开面板的按键（Linux 键码，46 = C）；启动时面板没开会自动按一下，0 表示不自动打开
    log_open_key: int = 46
    # 文字框背景亮度中位数 ≥ 这个值算“自己的消息”（浅色气泡），别人的消息是深色底
    log_self_min_value: int = 150
    # 面板里的文字像素变化少于这么多（半分辨率下）就不重新 OCR；背景里飘过的小光点不算
    log_change_pixels: int = 40
    log_max_skip: float = 3.0  # 最多这么久不 OCR，到时间了强制识别一次
    # 只在看到面板底部的“聊天……”输入框（= 面板开着）时才读；关着超过几秒会自动按 log_open_key 重新打开
    log_require_panel: bool = True
    log_reopen_after: float = 5.0  # 面板关着这么久就按键重新打开；0 表示不自动打开
    log_reopen_cooldown: float = 30.0  # 两次自动打开之间至少隔这么久（按了没用时不要一直按）
    # 非空时，每次读到新消息就把标注后的截图存到这里，方便调参
    debug_dir: str = ""


@dataclass
class OcrConfig:
    engine: str = "rapidocr"
    min_score: float = 0.6
    min_chars: int = 1
    threads: int = 8  # OCR 用的 CPU 线程数；默认会用满所有核，实测 8 线程和用满一样快


@dataclass
class ChatConfig:
    dedupe_ttl: float = 20.0  # 同一句话在这么久内再出现不算新消息（气泡会停留好几秒）
    similarity: float = 0.8  # 模糊去重阈值，吸收 OCR 抖动
    debounce: float = 0.8  # 收到新消息后再等这么久，把连发的几句合并成一次回复
    self_window: float = 60.0  # 自己发出的话在这段时间内被读回来时忽略
    max_pending: int = 8
    ignore_patterns: list[str] = field(default_factory=list)  # 正则，命中的气泡直接忽略


DEFAULT_PERSONA = """你叫三彩团子，是光遇里一个很可爱的女孩子，玩了挺久了。
最喜欢樱花发型，特意把自己的光之子调成了小个子，平时爱穿可爱风的衣服，看到好看的斗篷和发型会走不动路。
性格软萌但不做作，爱开玩笑，偶尔吐槽；喜欢跑图、收集先祖、坐着看风景。聊天像和熟人说话，不端着。"""


@dataclass
class ReplyConfig:
    dry_run: bool = True  # 默认只打印不发送，确认效果后再用 --live
    # 让对方知道是 AI 在回复。光遇里未成年玩家不少，强烈建议保留
    disclosure_prefix: str = "【AI】"
    max_chars: int = 40
    min_interval: float = 3.0  # 两次发送之间至少间隔（秒）
    max_per_minute: int = 8  # 几个人一起聊时 4 句/分钟会让回复拖到 10 多秒
    history_turns: int = 40  # 带给模型的最近几轮对话
    persona: str = DEFAULT_PERSONA
    # 认识的人：游戏昵称 → 说明。更推荐写在记忆目录的 friends.md 里（`skydango memory init` 生成）
    friends: dict[str, str] = field(default_factory=dict)
    # 记忆目录（不进 git）：profile.md 人设、friends.md 好友、notes.md 长期记忆、inbox.md 随手记、history.jsonl 聊天记录。
    # 为空不启用；dry-run 时不读也不写（dry-run 的回复没真的发出去）
    memory_dir: str = "memory"
    notes_every: int = 15  # 每攒够这么多轮，把聊天记录和随手记整理进长期记忆；0 表示不整理（随手记照常）


@dataclass
class LlmConfig:
    # "openai"：任何 OpenAI 兼容接口；"anthropic"；"echo"：不调模型，原样回显，用于联调
    provider: str = "openai"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.8
    max_tokens: int = 200
    timeout: float = 30.0


@dataclass
class SenderConfig:
    # 打开聊天输入框要点的位置（归一化）；为空表示输入框已经打开 / 由别的方式打开
    open_chat: list[float] = field(default_factory=list)
    # 用实体键盘按键打开聊天框（Linux 键码，28 = Enter），优先于 open_chat；0 表示不用
    open_chat_key: int = 0
    open_delay: float = 1.0  # 打开输入框后最多等多久（输入框一出现就开始输入，实测约 0.07 s）
    # 提交方式："editor_action"（输入法回车动作）/ "enter"（回车键）/ "tap"（点发送按钮）
    submit: str = "editor_action"
    editor_action: int = 4  # EditorInfo.IME_ACTION_SEND；不灵的话试 6（DONE）或 2（GO）
    send_button: list[float] = field(default_factory=list)
    type_delay: float = 0.2
    after_delay: float = 0.3
    close_with_back: bool = False


@dataclass
class WheelConfig:
    """快捷动作轮盘：光遇里按 1~8 直接做对应格子的动作；长按 Z 打开轮盘时按 E 进入编辑。

    编辑界面：点右侧列表里的动作 → 点轮盘格子，就换好了（实时生效）；点右上角 × 关闭。
    格子编号从正上方开始顺时针 1~8。坐标都按 1920×1080 的编辑界面标定。
    """

    library_dir: str = "emotes"  # 图标库：每个动作一张图，文件名就是动作名（如 鞠躬.png）
    locked_slots: list[int] = field(default_factory=lambda: [3, 8])  # 不让程序覆盖的格子
    open_key: int = 44  # Z（Linux 键码）
    edit_key: int = 18  # E
    slot_keys: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])  # 数字键 1~8
    editor_center: list[float] = field(default_factory=lambda: [0.335, 0.5])
    # 关闭编辑界面点右上角的 ×：中间的 ✓ 只有改过东西之后才能关
    editor_close: list[float] = field(default_factory=lambda: [0.983, 0.03])
    editor_radius: float = 0.18  # 格子中心到轮盘中心的距离，相对截图高度
    slot_icon_size: float = 0.12  # 格子图标的裁剪边长，相对截图高度
    list_roi: list[float] = field(default_factory=lambda: [0.693, 0.037, 0.9375, 1.0])  # 右侧动作列表
    list_scroll_x: float = 0.815
    list_scroll_from: float = 0.83  # 往下翻：从这个高度拖到 list_scroll_to
    list_scroll_to: float = 0.42
    match_threshold: float = 0.8  # 在动作列表里找图标（和图标库同尺寸）
    slot_match_threshold: float = 0.72  # 认轮盘格子里的图标（更大，还可能带“2级”之类的字）
    ui_delay: float = 0.8  # 每次操作后等界面动画


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    reply: ReplyConfig = field(default_factory=ReplyConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    sender: SenderConfig = field(default_factory=SenderConfig)
    wheel: WheelConfig = field(default_factory=WheelConfig)


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> Any:
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"未知配置项: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"配置项 {path}{key} 应该是一个表")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None) -> Config:
    config = Config()
    if path is None:
        return config
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到配置文件: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    return _merge(config, data)
