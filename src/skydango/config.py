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
    log_reopen_cooldown: float = 10.0  # 两次自动打开之间至少隔这么久（切场景时界面会隐藏，按了没用；别一直按）
    # 已废弃，改用 [run]。还设着的话当作 run.dir 用
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
    # 等模型回复时先打开输入框：头顶显示“正在输入”，像真人在打字；模型不回复就关掉。dry-run 不做
    type_ahead: bool = True


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
class EmoteConfig:
    """聊天时做表情动作（game/emotes.py）。做动作会松开牵手：牵着手时用 `run --no-emotes`。"""

    enabled: bool = True  # 图标库是空的时候自动不做
    # 白名单：不在轮盘上时可以换上去的动作（图标库里的名字）
    extra: list[str] = field(default_factory=list)
    # 允许换的格子；为空就只用轮盘上现有的动作。启动时记下原来的动作，退出时换回去
    swap_slots: list[int] = field(default_factory=list)
    min_interval: float = 20.0  # 两次动作至少隔这么久（秒）
    swap_min_interval: float = 120.0  # 两次换轮盘至少隔这么久：每次换要关聊天面板 5~10 秒


@dataclass
class RunConfig:
    """每次 `run` 在 dir 下建一个目录（时间-模式），放日志、识别记录、回复记录、截图、配置快照。"""

    dir: str = "runs"
    keep: int = 20  # 只留最近这么多次运行，旧的自动删；0 表示不删
    save_frames: bool = True  # 读到新消息时截面板存 JPG
    jpeg_quality: int = 85


@dataclass
class EnvConfig:
    """识别环境：每隔几秒对 3D 画面做一次 OCR，看好友头顶的名字标签（身边有谁）和地名（在哪张图）。"""

    enabled: bool = True
    interval: float = 3.0  # 扫描间隔（秒）；整块画面 OCR 一次 0.7~1 s，在后台线程里做；也决定发现互动请求有多快
    keep: float = 30.0  # 这么久内看到过名字标签就算在旁边（标签会被挡住、会闪）
    place_keep: float = 600.0  # 地名提示记多久
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 0.89])  # 扫描区域，排除底部输入栏
    min_score: float = 0.8
    threads: int = 4  # 这个 OCR 单独用的线程数，别和读聊天抢 CPU
    # 已知地名：画面上读到其中之一就记下（进入新区域时的地名提示，未在真机验证）
    places: list[str] = field(
        default_factory=lambda: ["晨岛", "云野", "雨林", "霞谷", "暮土", "禁阁", "暴风眼", "伊甸之眼", "家园", "遇境", "圣岛"]
    )


@dataclass
class SocialConfig:
    """社交互动：好友头顶圆圈里的图标变成牵手 / 拥抱 / 击掌时，点圆圈接受（发现请求靠 [env] 的后台扫描）。"""

    enabled: bool = True
    icons_dir: str = "assets/social"  # 圆圈图标模板，文件名就是类型
    accept_friends: list[str] = field(default_factory=lambda: ["hand", "hug", "highfive", "piggyback"])
    accept_strangers: list[str] = field(default_factory=lambda: ["candle"])  # 陌生人只接受点火（图标还没录到）
    icon_offset: float = 2.23  # 圆圈中心在名字标签上沿往下 这么多倍标签高度（实测标签 44 px 高、圆圈在下方 98 px）
    max_age: float = 10.0  # 请求是多久之内看到的才处理（后台每 env.interval 秒扫一次）
    cooldown: float = 15.0  # 同一个人的同一种请求处理完后隔多久才再处理
    accept_timeout: float = 6.0  # 点了之后最多等多久（团子要走过去）
    panel_key: int = 46  # 点屏幕会关掉聊天记录面板，接受完按这个键（C）重新打开
    check_delay: float = 0.6  # 每次点完等多久再看
    remember: float = 120.0  # 接受之后多久内在提示词里提一句
    error_backoff: float = 10.0  # adb 出错（比如查输入框失败）后这么久不处理请求，免得每 0.15 s 起一个 adb 进程


@dataclass
class BrainConfig:
    """统管大脑（brain/）：常驻的 Claude 收事件、看画面，决定说什么、做什么。`run --brain` 打开。"""

    enabled: bool = False
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = ""
    timeout: float = 60.0
    effort: str = "low"  # low / medium / high；运行中途改会让缓存失效，所以整个运行期固定
    max_tokens: int = 4000  # 单次输出上限（含思考）
    heartbeat: list[float] = field(default_factory=lambda: [45.0, 90.0, 180.0])  # 没事件时隔多久醒一次，闲着就退到下一档
    max_steps: int = 6  # 每次醒来最多调几次工具
    max_says: int = 2  # 每次醒来最多说几句
    look_min_interval: float = 5.0  # look 最多几秒一次
    auto_look_min: float = 20.0  # 自动附图：有人来 / 走 / 画面大变时，距上次看图至少这么久
    auto_look_max: float = 180.0  # 超过这么久没看图，醒来时一定附一张
    image_size: list[int] = field(default_factory=lambda: [1280, 720])  # 给模型的截图尺寸（约 1.2k tokens）
    jpeg_quality: int = 80
    look_at_max: int = 800  # look_at 返回图的最长边
    keep_images: int = 2  # 对话记录里只留最近几张截图
    prune_at: int = 4  # 攒到这么多张再一次性删（改前面的消息会让缓存失效，少改几次）
    compact_tokens: int = 40000  # 一次请求的输入超过这么多 token，就让它写摘要、从摘要重开
    command_timeout: float = 15.0  # 身体执行一条命令最多等多久
    offline_fallback: float = 120.0  # 大脑连续失败这么久，聊天交给备用回复
    max_usd_per_hour: float = 3.0  # 最近一小时估算花费超过这个：只为聊天醒，停心跳和自动附图
    pause_usd_per_hour: float = 5.0  # 超过这个：暂停大脑，只留身体反射和备用回复
    price_input: float = 2.0  # 每百万 token 美元（Sonnet 5）；缓存读按 0.1 倍、写按 1.25 倍估
    price_output: float = 10.0
    camera_step: float = 0.25  # 转视角每步按住方向键的秒数（0.5 s 约 90°）
    scene_change: float = 0.25  # 缩略图平均差异（0~1）超过这个算画面大变
    move_step: float = 0.3  # 移动每步按住方向键的秒数（实测 0.3 s 几乎不动、1 s 幅度很大，先取偏小的默认值）
    move_min_interval: float = 3.0  # 两次 move 调用最小间隔（秒），避免连续走位
    owner_name: str = ""  # 卡洛的游戏昵称，精确匹配；留空 = 主人命令模式关闭
    owner_window: float = 30.0  # 收到一条 # 开头的命令后，放宽 move/emote/camera 限制多少秒


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
    emotes: EmoteConfig = field(default_factory=EmoteConfig)
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    social: SocialConfig = field(default_factory=SocialConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)


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
