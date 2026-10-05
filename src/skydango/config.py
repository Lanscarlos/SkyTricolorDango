"""配置加载。所有字段都有默认值，config.toml 只需要写想覆盖的部分。"""

from __future__ import annotations

import dataclasses
import logging
import os
import tomllib
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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
    # run 启动时切到 ADBKeyboard、停下时切回来（device/ime.py）；用户自己打字用搜狗，免得每次手动切
    switch_ime: bool = True
    # 停下时切回哪个输入法（比如 "com.sohu.inputmethod.sogou/.SogouIME"）；空 = 启动时的那个，
    # 启动时已经是 ADBKeyboard（上次被强杀没切回）就找装了的搜狗输入法
    user_ime: str = ""
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
    mode: str = "log"  # 10-04 从 bubble 改：无障碍读法（source）、聊天面板开关、主人命令都只在 log 下生效
    # log 模式下读聊天的方式（spec 2026-10-04-a11y-chat-reader）："a11y" = 读游戏的无障碍节点（准、面板关着也能读好友头顶气泡），
    # 读不到自动退回 OCR；"ocr" = 截图识别（原来的做法）
    source: str = "a11y"
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
class PanelConfig:
    """聊天记录面板什么时候开（chat/panel.py，设计见 docs/superpowers/specs/2026-09-29-chat-panel-on-demand-design.md）。"""

    # "always" = 一直开着（关久了重开）；"auto" = 平时关着，定时 / 有人来 / 冒气泡时看一眼，聊天中保持打开（2026-09-30 真机验收后默认）
    mode: str = "auto"
    idle_peek: float = 30.0  # 闲着时多久看一眼（秒）
    quiet_close: float = 45.0  # 聊天中安静多久关面板
    peek_cooldown: float = 5.0  # 两次看一眼至少隔多久
    bubble_wait: float = 15.0  # 看到气泡后最多开着等多久
    bubble_gone: float = 3.0  # 气泡消失后再等多久
    bubble_strangers: bool = False  # 陌生人的气泡也触发（没解锁聊天的陌生人在面板里只有省略号）
    open_timeout: float = 1.5  # 按键后最多等多久面板出现
    chat_peek: float = 15.0  # 「聊着」时多久看一眼面板（读无障碍节点时面板关着也读得到气泡）


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
    # 主人（比如卡洛）在游戏里的昵称，和聊天记录面板解析出的 speaker 精确匹配；留空 = 主人命令模式关闭。
    # 只有 vision.mode = "log" 时 speaker 才有值，其他模式下这个功能自动失效。见 chat/commands.py
    owner_name: str = ""


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
    max_retries: int = 2  # SDK 自己的重试次数；大脑离线时的备用回复用 0（在身体线程里跑，不能卡太久）


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
    # 动作列表最上面「最近使用」图标右下角的时钟角标；`emotes scan` 跳过带它的几行（全是后面动作的重复）
    recent_badge: str = "assets/emotes/recent_clock.png"
    match_threshold: float = 0.8  # 在动作列表里找图标（和图标库同尺寸）
    # 认轮盘格子里的图标（更大，还可能带“2级”之类的字）。10-02 实测：真动作 0.72~0.93（6 号格邪笑最低，当时叫欢呼）、空 / 锁定格子 0.47；
    # 原来是 0.72，邪笑在门槛上来回跳，有时读不出来
    slot_match_threshold: float = 0.6
    ui_delay: float = 0.8  # 每次操作后等界面动画


@dataclass
class EmoteConfig:
    """聊天时做表情动作（game/emotes.py）。做动作不会松开牵手（用户实测）。"""

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
    accept_friends: list[str] = field(default_factory=lambda: ["hand", "hug", "highfive", "piggyback", "candle"])  # 好友举蜡烛也回应（2026-09-30 用户）
    accept_strangers: list[str] = field(default_factory=lambda: ["candle", "light"])  # 陌生人：接受点火；light = 团子举蜡烛点亮没点火的陌生人（2026-10-01）
    icon_offset: float = 2.23  # 圆圈中心在名字标签上沿往下 这么多倍标签高度（实测标签 44 px 高、圆圈在下方 98 px）
    max_age: float = 10.0  # 请求是多久之内看到的才处理（后台每 env.interval 秒扫一次）
    cooldown: float = 15.0  # 同一个人的同一种请求处理完后隔多久才再处理
    accept_timeout: float = 6.0  # 点了之后最多等多久（团子要走过去）
    panel_key: int = 46  # 已不用（面板由 chat/panel.py 统一开关，键见 vision.log_open_key）；留着免得旧配置报错
    check_delay: float = 0.6  # 每次点完等多久再看
    remember: float = 120.0  # 接受之后多久内在提示词里提一句
    error_backoff: float = 10.0  # adb 出错（比如查输入框失败）后这么久不处理请求，免得每 0.15 s 起一个 adb 进程
    # 点亮没点火的陌生人（spec 2026-10-01-light-unlit-stranger，都**未在真机验证**）
    light_after: float = 1.5  # 团子身边的火焰连续看到这么久才举蜡烛（别点路过的；10-03 从 3 改成 10-02 晚本机试过的 1.5）
    light_timeout: float = 8.0  # 举着蜡烛最多等这么久看他亮起来
    lit_min: float = 1.0  # 举蜡烛后至少这么久才可能算点亮（10-02 晚最快的一次举起 0.6 秒火焰就没了）
    lit_vanish: float = 0.8  # 判点亮：火焰离最后看到这么久（还要连着 2 次扫描没看到）才算没了（spec 2026-10-03-light-flame-vanish §3）
    lit_edge: float = 0.3  # 火焰最后在离范围左右边不到这么多倍团子框高的地方没的：算走开，不算点亮
    light_area_x: float = 1.4  # 在团子周围找火焰：左右各多少倍团子框高（10-02 在 10-01 晚的帧上量：真火焰最远 1.27 / 0.45 倍框高，再放宽一成）
    light_area_up: float = 0.5  # 从团子框上沿往上多少倍框高（下到框底；10-02 在 10-01 晚的帧上量：真火焰最远 1.27 / 0.45 倍框高，再放宽一成）
    light_jump: float = 0.5  # 相邻两次火焰位置差多少倍团子框高以内算同一个人
    light_cooldown: float = 60.0  # 没点亮（走了 / 超时 / 被打断）之后多久不再举：认不出是谁，只能按时间
    lit_v: int = 50  # 判点亮：HSV 的 V 低于这个算"很暗"的像素
    lit_black: float = 0.35  # 判点亮：人物框里很暗的像素占比低于这个算变亮（火焰没了时还 ≥ 这个 = 还黑着，接着等）……
    lit_drop: float = 0.25  # ……而且比举蜡烛时降了这么多以上，才算点亮
    after_light: str = "鞠躬"  # 点亮别人 / 接受别人点火后做的动作（轮盘上要有；鞠躬顺带放下蜡烛），空 = 不做
    bow_delay: float = 0.5  # 看到他亮起来 / 接受点火后等这么久再鞠躬（判点亮在火焰没了约 0.8 秒后，闪光动画已经开始；10-03 从 2.5 改）
    candle_slot: int = 3  # 轮盘上"举蜡烛"在第几格：按一下举起、再按一下放下（用户 2026-10-01）
    disk_min_score: float = 0.68  # 火焰圆盘的模板匹配分
    disk_sure: float = 0.85  # 连着看到火焰的这一段里至少一帧匹配到这么高才出请求（10-01 晚真机：真圆盘常到 0.86~0.99，灯笼菱形最高 0.78）
    flame: str = "assets/candle/flame.png"  # 圆盘里的火焰模板


@dataclass
class PerceptionConfig:
    """YOLO 感知层：视觉的第一道关卡，持续检测人物 / 名字标签 / 互动圆圈，替代 [env] 的定时整图 OCR。

    打开后 [env] 的扫描不再跑（env.enabled 仍要为 true，它决定要不要识别环境）；关掉就退回原来的整图 OCR。
    设计见 docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md。以下阈值都**未在真机验证**。
    """

    enabled: bool = False
    model: str = "models/sky-yolo.onnx"  # .onnx 用 onnxruntime；.pt / .engine 用 ultralytics
    # cuda（N 卡，onnxruntime-gpu）/ dml（DirectML：核显、A 卡，onnxruntime-directml，只对 .onnx 有用）/ cpu；
    # 要的 GPU 后端没装就退回 CPU 并警告（vision/onnxrt.py）。动作模型也用这个
    device: str = "cuda"
    # 模型里读不到类别名时用。player_unlit = 没点火的陌生人（黑色剪影）；点过火的陌生人外观和好友一样，标 player
    classes: list[str] = field(default_factory=lambda: [  # 新类别只能追加在末尾：标注文件存的是编号
        "player", "name_tag", "social_ring", "self", "player_unlit", "typing", "bench", "bonfire", "instrument", "spirit",
    ])
    imgsz: int = 960  # 推理尺寸，要和训练时一致；名字标签只有 40~50 px 高，640 时缩到 15 px 左右，偏小
    conf: float = 0.35
    iou: float = 0.5  # NMS 阈值（端到端模型不用）
    fps: float = 15.0  # 最多每秒检测几帧
    # 帧从哪来："body" 用身体主循环的截图（约 0.15 s 一张，到不了 15fps）；
    # "own" 感知线程自己截图（能到 15fps，但 MuMu 原生截图能不能两个线程同时调还没验证）
    capture: str = "body"
    track_buffer: float = 1.0  # 轨迹这么久没匹配上就删（秒）
    track_iou: float = 0.3
    keep: float = 5.0  # 好友这么久没看到才算走开（15fps 下偶尔被挡一下骗不到它，比 env.keep 短）
    stranger_after: float = 1.0  # 人物这么久都没有名字标签才算陌生人
    stranger_min_height: float = 0.08  # 点过火的人物框至少这么高（相对截图高度）才判陌生人：太远的好友名字标签可能看不清
    ocr_retry: float = 1.0  # 名字标签还没认出是谁时，隔多久再 OCR 一次
    ocr_votes: int = 3  # 同一条标签轨迹最多 OCR 几次（取出现最多的名字）
    ocr_threads: int = 2
    self_roi: list[float] = field(default_factory=list)  # 团子自己所在区域 [x1, y1, x2, y2]（归一化）；空 = 靠 self 类别排除
    # 画面被挡时暂停计时（黑屏、转镜头、开好友树、换轮盘……见一期设计 §4）
    hold_max: float = 60.0  # 暂停超过这么久自动恢复（防止哪里忘了恢复，永远不报"走开了"）
    occlusion_hold: float = 30.0  # 多人同时消失 + 画面大变（玩家自己开了全屏界面）时最多暂停这么久
    # 难例收集（一期设计 §5）：可能认错的画面存到 runs/<这次>/hard/，下一轮只补标这些
    hardcases: bool = True
    audit_interval: float = 30.0  # 每隔多久在后台做一次整图 OCR 核对 YOLO 认出的好友（约 1 s CPU）；0 = 不核对
    hardcase_max: int = 200  # 每次运行最多存几张
    low_conf: float = 0.25  # 检测器按这个出框；conf 以下的框不进追踪，只给难例收集看
    # 二期：远近（人物框高 ÷ 团子框高；阈值待真机标定）
    near: float = 0.8  # 比值 ≥ 这个算"近"
    far: float = 0.4  # 比值 < 这个算"远"，中间是"中"
    self_height: float = 0.2  # 没有团子框时，假定团子框高占屏高的这么多（待标定）
    approach_window: float = 1.5  # 走过来：看最近这么久的框高变化
    approach_grow: float = 0.25  # 框高增大超过这个比例（且往画面中间走）算朝团子走过来（待标定）
    approach_cooldown: float = 60.0  # 同一个人这么久内只报一次（陌生人整体算一个）
    approach_strangers: bool = True  # 陌生人走过来也报（陌生人多的地方嫌吵就关掉）
    typing_window: float = 8.0  # 陌生人的消息：这么久内头顶冒过"正在输入"气泡的人才算可能的说话人（待标定）
    bubble_gap: float = 1.0  # 同一个人的气泡消失超过这么久再出现，算新的一句（空闲注意力：新的一句让团子不再看腻）
    # 三期 §1：远处小目标二次检测（名字标签太小，YOLO 框不到）
    far_height: float = 0.08  # 人物框高 < 截图高 × 这个、又没挂上名字标签时，在它头顶裁一块再检测一次
    far_crops: int = 3  # 每帧最多裁几块（同一条轨迹每秒最多一次）；0 = 关
    # 物品（座位 / 篝火 / 乐器 / 先祖，spec 2026-09-29-object-recognition）
    object_min_hits: int = 3  # 一条物品轨迹至少连续看到几帧才算（防一闪而过的误认）
    object_near: float = 0.85  # 远近按框底边：底边 ≥ 画面高 × 这个算近（待标定）
    object_far: float = 0.65  # ≥ 这个算中，再高算远（待标定）
    # 按 Q 喊一声（spec 2026-10-01-q-call §1）
    edge_band: float = 0.06  # 名字标签中心在最左 / 最右这么宽（屏宽比例）里、又没挂上人 = 好友在画面外（不算在身边）；0 = 关
    # 追踪和接回（spec 2026-10-01-tracking-relink-motion；数字都没标定，perception track-eval 定完再上真机）。
    # 下面的开关全关 = 原来的行为
    sticky_names: bool = True  # 挂过名字标签的轨迹只要不断就一直算在身边（好友稍远标签就淡掉）；false = 照旧只靠标签
    track_low: bool = True  # 低分框（low_conf ~ conf）续旧轨迹（只续不开）
    track_predict: bool = True  # 速度预测 + 中心距离兜底
    track_center_gate: float = 0.6  # 中心距离门槛（× 预测框高，估的）
    track_pan: bool = True  # 估计画面平移（转镜头）并补偿
    relink: bool = True  # 好友轨迹断了，keep 秒内在附近冒出来的没名字的人按位置接回成"像他"
    motion: bool = True  # 运动方向（走近 / 走远 / 往左走 / 往右走 / 站着）
    motion_window: float = 1.5  # 看最近这么久（秒）
    motion_grow: float = 0.15  # 框高变化比例超过这个算走近 / 走远（估的）
    motion_side: float = 0.6  # 横向位移超过这么多个身高算往左 / 往右（估的）
    motion_hold: float = 0.5  # 新结论连续这么久才换（防抖）
    panel_people: bool = True  # 聊天面板开着时面板后面的人照样认（面板半透明，10-03 录像）；只丢面板里的名字标签 / 气泡 / 圆圈；false = 照旧整块丢
    panel_settle: float = 2.5  # 聊天面板开 / 关后画面横移的动画约 2 秒（10-03 录像）：这么久内速度清零、不攒走近 / 运动历史；0 = 关


@dataclass
class AssistConfig:
    """Claude 辅助标注（`perception label --assist`）：检测器出人物候选框，隔离的 claude -p 核对，
    结果和弱标注合并写成 YOLO 数据集。设计见 docs/superpowers/specs/2026-09-28-assist-labeling-design.md。
    令牌和配置目录沿用 [brain]。"""

    min_change: float = 30.0  # 挑帧：64×36 灰度缩略图和上一张留下的帧平均差超过这么多才留（2026-09-28 录像：240 张留 74 张）
    max_gap: int = 20  # 离上一张留下的帧这么多帧也强制留（2 fps 录像 = 10 秒），免得慢慢走近的过程被漏掉
    proposal_models: list[str] = field(default_factory=lambda: ["models/yolo11n.pt", "models/yolo11x.pt"])  # 没给 --model 时用的 COCO 模型（person 类）
    proposal_conf: float = 0.1  # 候选框阈值：宁可多给，让 Claude 去掉
    proposal_imgsz: int = 1280
    batch: int = 5  # 每次 claude -p 核对几帧
    jobs: int = 3  # 同时跑几个 claude -p
    timeout: float = 300.0  # 一批这么久没结果就重试一次
    model: str = "sonnet"
    self_hint: str = "白色头发、橙色护目镜、橙粉色袍子、背蓝紫色圆背包、头顶没有名字标签"  # 团子长相，换装后改这里


@dataclass
class PlacesConfig:
    """认地图（感知层三期 §2）：和图库里每个地方的参考截图比特征，最像、又和别的地方拉开差距才算认出。"""

    enabled: bool = False
    dir: str = "places"  # 图库：places/<地名>/*.jpg（不进 git，截图里可能有好友）；places add <地名> 往里加
    model: str = "models/places.onnx"  # 图像特征模型（ONNX，候选 MobileCLIP-S0 / DINOv2-small，places bench 比了再定）；"thumb" = 内置缩略图基线
    size: int = 224  # 输入尺寸（模型固定了尺寸时以模型为准）
    norm: str = "imagenet"  # 输入归一化：imagenet / clip / none（MobileCLIP 用 none）
    device: str = "cpu"  # cuda / dml / cpu，同 [perception] device
    place_min: float = 0.8  # 最像的余弦相似度至少这么高（待标定）
    place_margin: float = 0.05  # 且比第二像的"别的地方"高这么多（宁可不说，不能说错）
    place_interval: float = 30.0  # 每隔这么久认一次；画面大变后也认一次


@dataclass
class AppearanceConfig:
    """认装扮：从 YOLO 人物框裁图算外观特征，名字标签看不到时靠外观把人认回来（要配合 [perception]）。数字多是估的，以离线标定为准。"""

    enabled: bool = False
    model: str = "color"  # "color" = 内置颜色直方图；或 .onnx 特征模型路径
    size: int = 224  # 同 [places]
    norm: str = "imagenet"
    device: str = "cpu"  # cuda / dml / cpu，同 [perception] device
    every: int = 3  # 每条轨迹每几帧算一次特征
    max_per_frame: int = 4  # 一帧最多算几个
    min_height: float = 0.13  # 好样本的框高（相对截图高度）
    max_overlap: float = 0.2  # 和别的框 / 面板重叠超过这个比例就不是好样本
    ema: float = 0.2  # 平均特征的更新速度
    min_samples: int = 3  # 判 maybe / 陌生人编号 / 换装前至少几个好样本
    match: float = 0.88  # 估的
    card_match: float = 0.92  # 废弃，不再使用（留着旧配置不报错）
    margin: float = 0.05  # 最像的要比第二像的高这么多
    recheck: int = 5  # maybe 连续这么多个新样本低于门槛就摘掉
    # 多样本底库（身份底库设计 2026-10-03）
    unsure: float = 0.83  # 底库最高相似度在 [unsure, match) 之间算"疑似"，等 unsure_wait 秒再定
    unsure_wait: float = 2.0  # 秒
    dino: str = "models/dinov2-small.onnx"  # 第二个特征（DINOv2）模型，没有这个文件就只用颜色
    dango_match: float = 0.80  # 认团子自己（底库）的门槛
    gallery_max: int = 40  # 每个身份底库最多留几张样本
    enroll_max: int = 16  # 启动转圈登记团子时，一次最多钉进底库几张
    # 判换装（outfit_change 开着时）：低于它算换了装。10-01 标定：颜色特征下同一身衣服常跌到 0.15~0.5，所以默认不判
    changed: float = 0.40
    # false：不判换装（不追加新的一套、不发换装事件、上线中途不重新描述），好友 / 团子每次上线描述一次、覆盖最近那一套
    outfit_change: bool = False
    stranger_forget: float = 1800.0  # 秒
    describe: bool = True  # 关掉就只认人、不调模型
    describe_model: str = "haiku"
    describe_min_height: float = 0.18  # 送去描述的框高
    describe_max: int = 20  # 每小时最多描述几次
    redescribe_max: int = 3  # 同一个人一次上线最多重新描述几次
    outfit_keep: int = 3  # 关系卡每人留几套
    quota_wait: float = 600.0  # 额度用完后隔多久再试
    retry_after: float = 60.0  # 描述不清楚时隔多久用新样本再试
    describe_timeout: float = 60.0
    save: bool = True  # 攒训练数据
    save_every: float = 2.0  # 秒
    save_max: int = 2000  # 一次运行最多存几张


@dataclass
class GestureConfig:
    """别人对团子做的动作（感知层三期 §3，研究性质）：好友对着团子挥手、鞠躬时发 gesture 事件，大脑决定回不回礼。

    还没有模型：先 `perception clips` 从录像切片段、人工分到 <动作>/ 目录，训练导出 ONNX 后用 `perception gesture-eval` 评估，
    精确率 ≥ 90%、召回率 ≥ 60% 才打开。"""

    enabled: bool = False
    model: str = "models/gesture.onnx"  # 输入 1×frames×3×size×size（RGB，0~1），输出每个标签的分数（logits 或概率）
    labels: list[str] = field(default_factory=lambda: ["none", "wave", "bow", "cheer", "shy"])  # 模型输出的顺序；none = 没做这几个动作
    names: dict[str, str] = field(default_factory=lambda: {"wave": "挥手", "bow": "鞠躬", "cheer": "欢呼", "shy": "害羞"})  # 事件里的中文名
    frames: int = 16  # 一段几帧（2 s）
    fps: float = 8.0  # 每秒取几帧
    stride: int = 8  # 切片段时每隔几帧起一段（16 帧一段、8 帧起一段 = 半重叠）
    dataset: str = "datasets/gesture"  # 动作数据目录：_unlabeled/ 是刚切的，<动作>/ 是分好类的
    size: int = 112  # 人物裁剪缩放到的边长
    interval: float = 2.0  # 同一个人隔这么久判一次
    min_prob: float = 0.9  # 概率到这个才报（报错动作很尴尬，宁可不报）
    cooldown: float = 30.0  # 同一个人同一种动作这么久内只报一次


@dataclass
class InboxConfig:
    """难例收件箱：live 运行时存下的难例 → 外形头先筛 → 整帧核对 → 一键重训对比。"""

    enabled: bool = True
    dir: str = "datasets/inbox"  # 收件箱目录（gitignore）
    ask: bool = True  # 管理面板停团子时问不问要不要整理（这次存了难例 / 还有没整理的运行时）
    agree: float = 0.9  # 外形头与 YOLO 一致的概率门槛
    dup_diff: float = 6.0  # 去重：缩略图平均像素差低于这个算重复
    dup_gap: float = 5.0  # 去重：和上一张保留帧相隔不超过这么久（秒，按文件名的时间）才可能算重复
    val_every: int = 5  # 按运行目录名 crc32 % val_every == 0 整次运行进验证集（约每 val_every 次运行一次）
    retrain_min: int = 50  # 攒够这么多张核对过的才提示重训


@dataclass
class RetrainConfig:
    """一键重训的参数。"""

    base: str = "models/yolo11n.pt"  # 起点权重
    imgsz: int = 960
    epochs: int = 120
    batch: int = 16
    workers: int = 2


@dataclass
class AttrsConfig:
    """感知层第二层：人物属性 + 复核（给 YOLO 的人物框裁图，过冻住的 DINOv2 主干 + npz 线性头，判形态等属性、复核低分框）。

    默认关：模型（`perception attrs-train` 训练）过了上线门槛再开。低分框的阈值复用 [perception] low_conf。"""

    enabled: bool = False
    model: str = "models/attrs.npz"  # 线性头文件（W / b / labels / applies_to / pad）
    backbone: str = "models/dinov2-small.onnx"  # 冻住的特征主干，文件名要和 npz 里记的一致
    device: str = ""  # 空 = 跟 [perception] device
    every: float = 0.5  # 同一条轨迹隔这么久判一次（秒）
    max_crops: int = 4  # 每一轮最多裁几个框
    votes: int = 5  # 属性投票看最近几次
    accept: float = 0.6  # 投票概率到这个才算认定
    reject: float = 0.7  # 复核判成"不是人"的概率门槛
    reject_n: int = 3  # 累计复核够这么多次、平均下来"不是人"的概率还 ≥ reject 才撤下（高分框）
    # 点没点火的黑影分里 YOLO 一侧的权重；YOLO 一侧 = 最近几帧里 player_unlit 的比例（不是 YOLO 置信度）。
    # 要 < 0.5 外形头才翻得动一直认成同一类的 YOLO：0.3 时外形头黑影占比连续 flip_votes 票 > 0.86 / < 0.14 才翻
    yolo_w: float = 0.3
    flip_votes: int = 3  # 形态要翻转，新形态至少连续几票
    max_errors: int = 10  # 推理连续出错这么多次就自己关掉


@dataclass
class CatalogConfig:
    """装扮图鉴第 1 期（spec 2026-10-02-catalog-collect）：运行时把近处的人清楚的整身裁图存进 catalog/inbox/。

    只截图存盘、不发输入，所以默认开；要配合 [perception]。门槛是估的，用 `catalog collect` 在录像上定。"""

    enabled: bool = True
    dir: str = "catalog"  # 存储目录（相对当前目录，gitignore）
    every: float = 0.5  # 每条轨迹多久看一次（秒）
    min_height: float = 0.25  # 框高至少占画面多少（估的）
    edge: int = 8  # 框离画面四边至少几像素（人没被画面切掉）
    pad: float = 0.1  # 裁图四周放宽的比例
    sharp_min: float = 50.0  # 清晰度门槛：裁图缩到高 256 后拉普拉斯方差（估的）
    dark_max: float = 0.5  # 陌生人的框有多黑（candle.black）到这个就不收：多半是 YOLO 认成 player 的黑影（只对陌生人，好友披黑斗篷照收）
    per_who: int = 6  # 每个身份每次运行最多存几张
    gap: float = 3.0  # 留下的图两两至少隔几秒
    flush_every: float = 300.0  # 多久写一次盘（秒）
    max_per_run: int = 300  # 每次运行最多存几张


@dataclass
class IconsConfig:
    """认地图交互图标（spec 2026-10-05-icon-detection）：YOLO 框出的图标认出种类、写进状态和画面（只认不点）。要配合 [perception]。"""

    enabled: bool = False
    classifier: str = "template"  # 认种类的办法：template 模板匹配 / dino DINOv2 特征检索
    dino: str = "models/dinov2-small.onnx"  # classifier = "dino" 时的特征模型
    refs: str = "assets/icons"  # 参考图目录
    dino_match: float = 0.75  # DINOv2 最像一张的相似度门槛
    dino_margin: float = 0.05  # 领先第二像的种类至少多少
    vote: int = 5  # 每条轨迹多数表决看最近几次
    min_hits: int = 3  # 连续几帧都在才算
    every: float = 0.5  # 同一个圈最多每几秒认一次（票攒满 vote 次以后放慢到 2 倍；0 = 每帧都认）
    max_per_frame: int = 4  # 一帧最多认几个圈（DINOv2 在 CPU 上一张约 30 ms；没轮到的沿用上次的结果；0 = 不限）
    under_x: float = 0.25  # 圈算在某个框头顶：横着放宽几倍框宽
    under_up: float = 1.0  # 圈算在某个框头顶：往上最多几倍框高
    crop_scale: float = 1.3  # 图标裁图放大几倍
    flame_ring: float = 1.82  # 收件箱整图火焰预标：圆圈框半径 = 火焰半高 × 这么多（10-05 在 datasets/sky 的 1 个火焰圆圈上量的中位数；数据集里只有这 1 个，有火焰录像后再量）
    save: bool = True  # 认不出的图标存盘
    save_max: int = 200  # 每次运行最多存几张

    def check(self) -> None:
        if self.classifier not in ("template", "dino"):
            raise ValueError(f"配置项 icons.classifier 只能是 template 或 dino，现在是 {self.classifier!r}")


@dataclass
class FriendCheckConfig:
    """大脑的 check_friend 工具：点一下人物打开右侧的好友树面板，截图给大脑看是不是好友，再关掉。

    关面板用 ESC（用户确认）；面板样子、好友和陌生人的面板怎么区分还没核对：先用 `python -m skydango friend-check X Y` 试，确认后再打开 enabled。
    """

    enabled: bool = False
    open_delay: float = 1.2  # 点完等多久再截图（面板弹出动画）
    close_delay: float = 0.8  # 每试一种关法后等多久再看
    # 关面板的办法，依次试：esc 实体键盘 ESC（用户确认能关）、back 安卓返回键、tap 点 close_tap
    close: list[str] = field(default_factory=lambda: ["esc"])
    close_tap: list[float] = field(default_factory=list)  # [x, y] 归一化；close 里有 "tap" 时点这里
    panel_left: float = 0.55  # 面板在画面右边：比较这条线右边的变化
    changed: float = 0.06  # 右侧缩略图平均差异超过这个算面板打开了（0~1）
    min_interval: float = 30.0  # 两次确认至少隔几秒（点屏幕会关聊天面板、打断别的）
    max_look_age: float = 15.0  # 坐标按上次 look(image=true) 的图给；那张图太旧（人走了）就不点


@dataclass
class PanelsConfig:
    """面板识别（vision/panels.py）：画面上开着哪些面板（特征卡 assets/panels/ + 通用 OCR 兜底），身体的遮挡护栏和大脑的面板工具用它。

    设计见 docs/superpowers/specs/2026-09-29-panels-design.md。只接大脑模式。
    """

    enabled: bool = True  # 大脑模式默认开；false = 退回原来的行为（不看面板、不加工具、不做护栏）
    cards_dir: str = "assets/panels"
    ocr_threads: int = 2  # 细读 / 通用兜底单独用的 OCR 线程数，别和读聊天抢 CPU
    scan_interval: float = 5.0  # 通用兜底：没有别的触发时，隔多久看一次屏幕中部
    unknown_cooldown: float = 2.0  # 两次通用兜底扫描至少隔这么久
    unknown_roi: list[float] = field(default_factory=lambda: [0.15, 0.08, 0.85, 0.92])  # 通用兜底 OCR 的区域（避开左边聊天面板、底部输入栏）
    unknown_min_chars: int = 6  # 除按钮外至少这么多字才算面板（名字标签、"2级"这类零星字不算）
    unknown_pad: float = 0.02  # 不认识的面板框：文字外接框往外扩（相对截图宽）
    unknown_ttl: float = 30.0  # 这么久没再确认过就当它关了（防止一直挡着）
    unknown_blocks: bool = False  # 不认识的面板挡不挡操作；没在真机核对误报率之前只报告、不拦（头顶气泡、告示牌可能误认）
    change: float = 0.25  # 画面变化触发阈值（和 brain.scene_change 同一种度量，实测后调）
    button_max_chars: int = 6  # 按钮最多几个字
    button_words: list[str] = field(
        default_factory=lambda: ["确定", "取消", "关闭", "加入", "同意", "好的", "知道了", "返回", "拒绝", "稍后", "以后再说", "确认", "重试"]
    )
    retreat: list[str] = field(default_factory=lambda: ["关闭", "取消", "返回", "拒绝", "稍后", "知道了", "以后再说"])  # 撤退类：大脑可以直接按
    never: list[str] = field(
        default_factory=lambda: ["购买", "充值", "支付", "兑换", "删除", "屏蔽", "举报", "退出", "注销"]
    )  # 永远不按，主人 #允许 也不按
    permit_window: float = 60.0  # 主人 #允许 之后多久内能按一次
    read_ttl: float = 15.0  # panel_press 要求 panel_read 在这么久以内
    ui_delay: float = 0.8  # 关面板 / 按按钮后等界面动画


@dataclass
class ViewerConfig:
    """团子的 HTTP 接口（管理面板「真机团子」页的画面、大脑、聊天、手动控制、停止都经它）：`run` 总是开、只听 127.0.0.1。
    见 docs/superpowers/specs/2026-10-04-console-attach-design.md。"""

    port: int = 19391  # 管理面板也按它找团子；端口被占时 run 拒绝启动（同一时间只有一个团子）
    fps: float = 10.0  # 最多每秒更新几帧
    width: int = 1280  # 发给浏览器的图缩到这么宽
    quality: int = 70  # JPEG 质量


@dataclass
class SpinConfig:
    """转一圈（`#spin` / `camera spin` / 打开感知层时的 look_around），见感知层二期设计 §0、§1。"""

    seconds_per_turn: float = 2.0  # 按住方向键转一圈要几秒（按 0.5 s ≈ 90° 估的，待 camera spin 标定）
    fps: float = 15.0  # 转的时候每秒截几张
    max_turns: int = 2  # #spin 最多转几圈
    min_interval: float = 10.0  # 两次 #spin 至少隔几秒
    hfov: float = 90.0  # 画面水平视野角（度），算方位用（待 camera spin 标定）
    merge_deg: float = 30.0  # 陌生人方位相差不到这么多度算同一个
    self_motion: float = 0.03  # 转一圈时框中心平均移动不到画面宽的这么多，才可能是团子


@dataclass
class TrackConfig:
    """技能 track（盯着好友）：小步转镜头把目标保持在画面中间。实测数据见 game-ops §2「转视角实测（D0）」。"""

    deadband: float = 0.15  # 目标中心离画面中线不到半屏宽的这么多就不转（0.15 × 960 ≈ 144 px，大于最小修正量）
    gain: float = 0.15  # 按键秒数 = gain × 偏差（半屏宽为 1）；2026-09-30 真机：0.1 追不上近处横着走的人
    nudge_min: float = 0.02  # 最短按键（秒）：2026-09-29 D0 实测一下就移 35~40 px
    nudge_max: float = 0.1  # 最长按键（秒）：技能在身体线程里跑，不能按太久（look_person 换角度、空闲注意力用）
    chase_max: float = 0.15  # 盯人（track）自己的最长按键：2026-09-30 真机 0.1 追不上；D0 近处的人 0.2 s 以后就饱和
    settle: float = 0.6  # 按完等画面停稳（秒）：D0 实测松手 0.3~0.6 s 后位置才稳定；开始后也先等这么久
    lost_after: float = 3.0  # 这么多秒看不到目标算跟丢
    max_age: float = 0.5  # 名字标签超过这么多秒没更新就不用
    stall_nudges: int = 3  # 同方向连续按这么多次……
    stall_px: float = 20.0  # ……误差缩小不到这么多像素 = 离得太近转不动了，停手
    max_seconds: int = 60  # 一次最多盯多久


@dataclass
class PeekConfig:
    """look_person 发现好友被团子挡住：边转边看（每一步只看 YOLO）让人露出来，太小就拉近。镜头不复位。
    按键、等画面稳定、转不动的判断沿用 [track]。数字都是估的，没在真机标定。"""

    enabled: bool = True
    self_center: float = 0.35  # YOLO 的团子框中心离屏幕中线不超过屏宽的这么多才采信（团子是镜头支点，一般在中间；2026-09-30 实测镜头没跟正时偏到 29%）
    too_close: float = 0.5  # 团子框高超过屏高的这么多 = 镜头贴太近，先拉远再转
    too_small: float = 0.25  # 露出来的好友框高不到屏高的这么多 = 看不清，拉近
    clear_overlap: float = 0.3  # 好友框和团子框水平重叠不到好友框宽的这么多 = 露出来了
    gain: float = 0.3  # 按键秒数 = gain × 还差多少（半屏宽算 1），夹到 [track] nudge_min~nudge_max
    max_zoom_out: int = 2
    max_zoom_in: int = 3
    max_presses: int = 10  # 转 + 缩放一共最多按几次
    max_seconds: float = 8.0  # 最多花多久
    tag_age: float = 1.0  # 名字标签超过这么多秒没更新就不算（判断挡住、每一步测量）
    poll: float = 0.15  # 等画面稳定后，每隔这么久看一次感知结果


@dataclass
class CallConfig:
    """按 Q 喊一声（spec docs/superpowers/specs/2026-10-01-q-call-design.md）：稍远的好友头顶亮出名字约 5 秒，
    大脑用 call 工具找人，身体在好友"走开"但画面里还有没挂名字的人时自动兜底喊一次；顺带看呼唤光圈认团子自己。
    数字都是估的，没在真机标定。"""

    enabled: bool = True  # false：没有 call 工具、不自动喊，提示词 / status / 工具列表逐字照旧（感知层没开时也一样）
    auto: bool = True  # 身体自动兜底
    min_gap: float = 20.0  # 任意两次之间至少隔几秒（卡洛 # 命令窗口里大脑的 call 不受限）
    auto_quota: int = 3  # 身体自动喊：auto_window 秒里最多几次
    auto_window: float = 60.0  # 10-02 晚用户：10 分钟 3 次太严，改成 1 分钟 3 次
    auto_after_leave: float = 30.0  # 好友走开多少秒内还会为他自动喊（同一次走开只喊一次）
    auto_again: float = 300.0  # 自动喊认回来（窗口里看到名字）的好友，这么久内不再为他自动喊（10-02 晚站在稍远处的好友 80 秒被喊了 3 声）
    window: float = 6.0  # 呼喊窗口：按键后这么久里亮出来的名字都算（标签约 5 秒 + 余量）
    burst: float = 1.0  # 按键后连拍几秒看光圈
    halo: bool = False  # 光圈认团子：认错会把一个好友当成团子过滤掉，先 perception halo-eval 标定、真机核对再开
    halo_rise: float = 25.0  # 头顶区域平均灰度（0~255）比按键前高出这么多才算冒圈（估的）
    halo_center: float = 0.35  # 冒圈的人框中心离画面中线不超过屏宽的这么多才算团子（同 peek.self_center）


@dataclass
class BrainConfig:
    """统管大脑（brain/）：常驻的 Claude Code（订阅）收事件、调身体的工具；眼睛（Haiku）把画面写成文字。`run --brain` 打开。

    和用户自己的 Claude Code 隔离：单独配置目录 config_dir + `claude setup-token` 生成的令牌（放在用户环境变量 token_env）。
    """

    enabled: bool = False
    claude_path: str = "claude"
    token_env: str = "SKYDANGO_CLAUDE_TOKEN"
    config_dir: str = ".brain-claude"  # 大脑和眼睛用的单独 Claude Code 配置目录（gitignore）
    model: str = "sonnet"
    effort: str = "low"  # 要快
    eyes_model: str = "haiku"  # 眼睛：把截图写成文字的模型（一次性 claude -p）
    eyes_timeout: float = 60.0  # 眼睛一次描述最多等多久（秒）
    memory_model: str = "sonnet"  # 记忆整理（随手记 inbox.md、整理 notes.md）用的模型（一次性 claude -p）
    memory_timeout: float = 120.0  # 记忆整理一次最多等多久（秒）
    history_turns: int = 20  # 启动时把 history.jsonl 最近几轮原话放进大脑的系统提示词（重启后接得上话），0 不带
    turn_timeout: float = 120.0  # 大脑一轮最多等多久（秒），超了结束进程、下次用 --resume 接回
    heartbeat: list[float] = field(default_factory=lambda: [45.0, 90.0, 180.0])  # 没事件时隔多久醒一次，闲着就退到下一档
    background_wait: float = 20.0  # 背景事件（陌生人来去、好友走开、已处理的互动请求…）自己不叫醒大脑，攒了这么久还没醒才叫醒一次
    rejoin: float = 60.0  # 好友走开后这么久内又出现算"回来了"（背景事件），不再当成来到身边叫醒大脑打招呼
    max_steps: int = 6  # 每次醒来最多调几次工具
    max_says: int = 2  # 每次醒来最多说几句
    look_min_interval: float = 5.0  # look 最多几秒一次（再调给缓存的描述）
    auto_look_min: float = 20.0  # 眼睛自动看：有人来 / 走 / 画面大变时，距上次看至少这么久
    auto_look_max: float = 180.0  # 超过这么久没看，眼睛一定看一次
    image_size: list[int] = field(default_factory=lambda: [1280, 720])
    jpeg_quality: int = 80
    look_at_max: int = 800  # look_at 返回图的最长边
    command_timeout: float = 15.0  # 身体执行一条命令最多等多久
    offline_fallback: float = 120.0  # 大脑连续失败这么久，聊天交给备用回复
    limit_retry: float = 600.0  # 订阅额度用完后多久再试
    camera_step: float = 0.25  # 转视角每步按住方向键的秒数（0.5 s 约 90°）
    scene_change: float = 0.25  # 缩略图平均差异（0~1）超过这个算画面大变
    move_step: float = 0.3  # 移动每步按住方向键的秒数（实测 0.3 s 几乎不动、1 s 幅度很大，先取偏小的默认值）
    move_min_interval: float = 3.0  # 两次 move 调用最小间隔（秒），避免连续走位
    owner_name: str = ""  # 卡洛的游戏昵称，精确匹配；留空 = 主人命令模式关闭
    owner_window: float = 30.0  # 收到一条 # 开头的命令后，放宽 move/emote/camera 限制多少秒
    # DeepSeek 备用大脑（spec 2026-10-03-deepseek-fallback-brain）：Claude 订阅额度用完时顶上
    fallback: bool = True  # 额度用完切 DeepSeek 备用大脑；[llm] 没 Key 时退回纯文字备用回复
    fallback_max_tokens: int = 4096  # 备用大脑回复长度（[llm] max_tokens=200 对大脑太短）
    force_fallback: bool = False  # 调试：从启动就用 DeepSeek 大脑，不等额度真用完
    fallback_history: int = 8  # 备用大脑带最近几轮（它每轮重发、不是常驻会话；不带就不记得刚说过什么，10-03 晚重复说）；0 = 不带


@dataclass
class ProactiveConfig:
    """看场合主动开口（docs/superpowers/specs/2026-09-29-proactive-chat-design.md）：眼睛挑新鲜事发 notice、场合给大脑看、主动开口的护栏。
    数字都是估的，没在真机验证；enabled = false 完全照旧。"""

    enabled: bool = True
    notice_min: float = 60.0  # 两个 notice 事件至少隔几秒
    prev_max_age: float = 600.0  # 上一份场景描述超过几秒就不拿来比（不要新鲜事）
    auto_look_busy: float = 60.0  # 好友在身边时眼睛最久几秒看一次（平时是 brain.auto_look_max）
    busy_window: float = 180.0  # 热闹：这么多秒里别人说了至少 busy_lines 句
    busy_lines: int = 4
    reply_window: float = 90.0  # 主动那句之后多少秒内有好友说话算有人接
    quota_window: float = 600.0  # 主动开口额度的时间窗口（秒）
    quota_busy: int = 4  # 热闹时窗口内最多主动说几句
    quota_quiet: int = 2  # 安静时窗口内最多主动说几句
    min_gap: float = 60.0  # 两句主动的话之间至少隔几秒
    greet_window: float = 60.0  # 好友刚来这么多秒内、团子还没开口：打招呼不受 min_gap 限制（额度照旧）
    cold_after: int = 3  # 连续几句主动的话没人接就暂停主动，等有好友说话
    self_names: list[str] = field(default_factory=lambda: ["团子", "三彩"])  # 聊天里出现这些字算“叫了你”


@dataclass
class LullConfig:
    """冷场时的心理活动（docs/superpowers/specs/2026-10-01-lull-musing-design.md）：好友在身边不说话了 / 聊着聊着走了，
    按节点叫醒大脑、记它心里想的。数字都是估的，没在真机验证；enabled = false 完全照旧。"""

    enabled: bool = True
    talk_window: float = 300.0  # 最后一句之前多少秒内有好友说过话，才算“刚才在聊”
    stages: list[float] = field(default_factory=lambda: [60.0, 180.0, 360.0])  # 冷场多少秒时各叫醒一次（走开的用后两个）
    leave_spoke: float = 120.0  # 好友走开时他多少秒内说过话，算聊着聊着走了
    leave_said: float = 60.0  # 好友走开时团子多少秒内说过话，算聊着聊着走了
    leave_grace: float = 15.0  # 聊着聊着走开后多少秒还没回来才叫醒（名字标签闪一下不算）
    musing_max: int = 60  # “心里：”一行最多几个字


@dataclass
class AttentionConfig:
    """空闲注意力（东张西望，spec 2026-09-30-idle-attention、2026-10-03-attention-search）：闲着时按兴趣小步转镜头看说话 / 走近 /
    对团子做事的人；没有这些时找刚走开的好友、一个人待着时往最久没看过的方向看一片，没有动机就不转。
    只在聊天面板 auto 模式、面板关着时动。数字都是估的，真机调。按键长短、settle、转不动沿用 [track]。"""

    enabled: bool = True
    talk_friend: float = 1.0  # 基础兴趣：好友在说话（头顶气泡）
    act_on_me: float = 0.9  # 有人对团子做动作（挥手 / 鞠躬）
    approach: float = 0.7  # 有人朝团子走过来
    talk_stranger: float = 0.4  # 陌生人在说话
    friend_present: float = 0.2  # 画面里站着的好友
    focus_interest: float = 1.0  # 大脑 attention(focus=…) 关注的人
    min_interest: float = 0.15  # 实际兴趣低于它不算目标
    bore_rate: float = 0.2  # 目标在中间带里时每秒涨的看腻（0~1）
    recover_rate: float = 0.05  # 不在中间带时每秒消的看腻
    stuck_bored: float = 0.5  # 转不动时加的看腻
    center_band: float = 0.4  # 画面中间多宽算"已经在看"（屏宽的比例）
    gain: float = 0.1  # 按键秒数 = gain × 偏差（半屏宽算 1），夹到 [track] nudge_min~nudge_max
    switch_margin: float = 0.2  # 新目标的实际兴趣要比当前高出这么多才换
    switch_hold: float = 2.0  # 刚换过目标这么多秒内不再换
    look_first: float = 2.0  # 冒气泡 / 有人走近时先看这么久再让聊天面板开
    # 有意识地找（spec 2026-10-03-attention-search）：找刚走开的好友、一个人时环顾一片；都没有就不转
    search: bool = True  # 关掉只剩被动注意（有人说话 / 走近 / 对团子做动作就转过去看）
    press_deg: float = 18.0  # 每下 [track] nudge_max 约转几度（按 [spin] seconds_per_turn 2 秒一圈估的，待 camera spin 标定）
    seg_presses: int = 3  # 分段转：一段按几下
    dwell: float = 1.5  # 一段转完停几秒（等 YOLO 出框、名字标签读出来）
    scan_after: float = 20.0  # 画面里没人多久才环顾
    scan_every: float = 60.0  # 两次环顾至少隔几秒（再乘心情精力、模式的倍数）
    scan_segments: list[int] = field(default_factory=lambda: [2, 3])  # 环顾一次转几段（随机取一个）
    scan_look: float = 3.0  # 环顾看到陌生人停下看几秒
    lost_segments: int = 2  # 找走开的好友往一边最多转几段（Q 之后看到贴边标签再给同样多）
    resume_within: float = 10.0  # 找到一半被挡住 / 打断，这么久内接着找，超过就不找了
    max_step: float = 1.0  # 每圈时间差上限（沙盒模拟时钟会一下跳几小时）


@dataclass
class ReflexConfig:
    """身体反射（见 docs/superpowers/specs/2026-09-30-body-reflex-design.md）：有人跟团子说话马上冒输入气泡、回礼、闲着做小动作。数字都是估的"""

    enabled: bool = True  # false = 完全照旧（不开框、不做反射动作、gesture 照旧交给大脑）
    bubble: bool = True  # 有人跟团子说话时身体马上打开输入框（头顶“正在输入”），大脑想好了用这个框发
    followup_window: float = 30.0  # 已挪到 [addressee]；没写 [addressee] followup_window 时用它（见 followup_window()）
    bubble_max: float = 45.0  # 替大脑开的框最长开多久（秒）
    addressed: list[str] = field(default_factory=list)  # 被叫到时开框前偶尔做的小动作（轮盘上的动作名）
    addressed_chance: float = 0.3
    return_map: dict[str, str] = field(default_factory=dict)  # 回礼：[gesture] labels 里的标签 → 动作名，比如 wave = "挥手"
    return_chance: float = 0.7
    idle: list[str] = field(default_factory=list)  # 闲着时的小动作
    idle_min: float = 180.0  # 闲着多久做一个（秒，在 idle_min~idle_max 之间随机）
    idle_max: float = 420.0
    quota_window: float = 600.0  # 反射动作额度的时间窗口（秒）
    quota: int = 4  # 窗口内反射最多做几个动作
    min_gap: float = 4.0  # 任何两个动作（反射或大脑）之间至少隔几秒，给动画留时间


@dataclass
class AddresseeConfig:
    """分清好友在跟谁说话（spec 2026-10-05-addressee-design）：给每句好友聊天标 跟你说 / 跟别人说 / 说给大家 / 拿不准。数字都是估的"""

    enabled: bool = True  # false = 逐字照旧：不判、事件不带标注、没有 aside、提示词不变、反射用原来的 addressed() 规则
    followup_window: float | None = None  # 团子说完多少秒内好友接话算“在跟团子说”；None = 用 [reflex] followup_window
    followup_lines: int = 2  # 接话窗口里最多算头几句
    thread_window: float = 20.0  # 两个好友之间“还在聊”的窗口（秒）
    group_words: list[str] = field(default_factory=lambda: ["大家", "你们", "各位"])
    greet_words: list[str] = field(  # 身边两个以上好友时，开头是这些算说给大家
        default_factory=lambda: ["晚上好", "早上好", "中午好", "来了", "我来啦", "拜拜", "晚安", "走了"]
    )


@dataclass
class BackstageConfig:
    """幕后（spec 2026-10-01-backstage）：团子知道自己是 AI、主人做了她，能跟主人聊她自己怎么运作。"""

    enabled: bool = False  # 公开仓库：默认不出戏
    changelog_max: int = 10  # 「更新记录」最多几条
    changelog_days: int = 7  # 没有标记（或标记不在历史里）时往前看几天


@dataclass
class InnerConfig:
    """内心层第 1 期（见 docs/superpowers/specs/2026-09-30-inner-phase1-design.md）：给好友记关系卡、每次上线记一行，
    写在 memory/inner/，只在 --live 时写盘。数字都是估的"""

    enabled: bool = True  # false = 完全照旧（不记账、不回填、提示词和 status 不变）
    visit_gap: float = 1800.0  # 离上次在场超过这么多秒再出现，算新的一次见面
    session_gap: float = 7200.0  # 回填 history.jsonl 时，相邻两轮隔这么多秒算两次上线
    long_gap: float = 7.0  # 超过这么多天没见算“好久没见”
    save_every: float = 60.0  # live 时每隔多少秒存一次 people.json / current.json
    max_step: float = 5.0  # 算在一起待了多久时，单圈最多算几秒（卡顿时不一下加一大截）
    # 第 2 期：反思（docs/superpowers/specs/2026-09-30-inner-phase2-design.md）
    reflect: bool = True  # 反思总开关（心情、精力、别扭、心愿、日记）；false = 第 1 期原样
    reflect_model: str = "sonnet"  # 反思用的模型（一次性 claude -p，令牌同大脑）
    reflect_every: float = 1200.0  # 有动静时多久反思一次（秒）
    reflect_after_quiet: float = 180.0  # 好友说够 reflect_min_lines 句后安静多久反思
    reflect_min_lines: int = 6
    reflect_timeout: float = 90.0  # 一次反思最多等多久（下线那次也是）
    rest_gap: float = 3600.0  # 两次上线隔多久算睡过一觉（不到就接着上次的累算）
    grudge_min_days: int = 3  # 一起玩过几天才可能闹别扭
    grudge_max: float = 7200.0  # 别扭最长多久（秒），到点自己消气
    wants_max: int = 3  # 心愿 / 惦记的事最多几条
    want_days: float = 7.0  # 心愿几天后过期
    diary_prompt: int = 1  # 「日子」带几篇日记
    # 第 3 期：性格（docs/superpowers/specs/2026-09-30-inner-phase3-design.md）
    persona: bool = True  # 性格总开关（口头禅、老梗、看法，提示词「脾气」，收着点）；false = 第 2 期原样
    fade_days: float = 14.0  # 多少天没用的条目淡出
    catchphrases_max: int = 5
    jokes_per_friend: int = 3
    jokes_max: int = 20
    opinions_max: int = 10
    soft_minutes: float = 30.0  # 好友说难过后多久内对他收着点


@dataclass
class ConsoleConfig:
    """管理面板（`console`，见 docs/superpowers/specs/2026-09-29-console-design.md）。启动选项由面板写进 console.toml，只影响面板启动的团子。"""

    port: int = 19390  # 面板端口（团子的接口端口是 [viewer] port）
    stop_timeout: float = 60.0  # 停止时最多等几秒（live 大脑退出前要写记忆），超了强杀
    log_lines: int = 500  # 日志尾巴保留几行
    # 上次的启动选项，面板写
    brain: bool = True  # 统管大脑（run 的默认）；false = 普通 Agent（调试用）
    live: bool = False
    emotes: bool = True
    duration: float = 0.0  # 0 = 一直跑


@dataclass
class SandboxConfig:
    """大脑沙盒（`sandbox`，见 docs/superpowers/specs/2026-09-30-brain-sandbox-design.md §7）。管理面板设置清单不加（很少改）。"""

    dir: str = "sandbox"  # 沙盒目录：memory/、scenarios/、reports/、clock.json（不进 git）
    port: int = 19392  # 沙盒子进程的接口端口
    step_timeout: float = 180.0  # 回放时每步最多等多久安静（秒，真实时间）
    wake_hour: int = 9  # "睡一晚" / "到明早" 拨到几点
    emotes: list[str] = field(default_factory=list)  # 没有 emotes/ 图标库时假装轮盘上有这些动作


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
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
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    places: PlacesConfig = field(default_factory=PlacesConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    assist: AssistConfig = field(default_factory=AssistConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    attrs: AttrsConfig = field(default_factory=AttrsConfig)
    inbox: InboxConfig = field(default_factory=InboxConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    icons: IconsConfig = field(default_factory=IconsConfig)
    friend_check: FriendCheckConfig = field(default_factory=FriendCheckConfig)
    panels: PanelsConfig = field(default_factory=PanelsConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    spin: SpinConfig = field(default_factory=SpinConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    peek: PeekConfig = field(default_factory=PeekConfig)
    call: CallConfig = field(default_factory=CallConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    lull: LullConfig = field(default_factory=LullConfig)
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
    addressee: AddresseeConfig = field(default_factory=AddresseeConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    inner: InnerConfig = field(default_factory=InnerConfig)
    backstage: BackstageConfig = field(default_factory=BackstageConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    providers: dict[str, dict] = field(default_factory=dict)  # 模型供应商：[providers.<id>]，原样的表（models/config.py 解析）
    models: dict[str, dict] = field(default_factory=dict)  # 每个用处选哪个模型：[models.<用处>]
    sources: dict[str, str] = field(default_factory=dict)  # 配置文件里写了的叶子键 → "config" / "console"


# 删掉的配置项：旧的 config.toml / console.toml 里还写着时跳过、提醒一句（别的未知键照旧报错）
DEPRECATED = frozenset({
    "attention.wander_min", "attention.wander_max", "attention.wander_presses", "attention.wander_same_side",  # 2026-10-03 随意看删了
    "viewer.host", "console.child_port",  # 2026-10-04 viewer 网页删了：接口只听本机、端口统一用 [viewer] port
})


# 挪了位置的配置项：旧位置照样生效（还是原来的字段），只警告
MOVED = {"reflex.followup_window": "addressee.followup_window"}


def followup_window(cfg: "Config") -> float:
    """团子说完多少秒内好友接话算在跟团子说：[addressee] 优先，没写用 [reflex] 的旧位置。"""
    value = cfg.addressee.followup_window
    return cfg.reflex.followup_window if value is None else value


def _merge(obj: Any, data: dict[str, Any], path: str = "") -> Any:
    known = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if f"{path}{key}" in MOVED:
            log.warning("配置项 %s%s 挪到了 %s，旧位置照样生效", path, key, MOVED[f"{path}{key}"])
        if key not in known:
            if f"{path}{key}" in DEPRECATED:
                log.warning("配置项 %s%s 已经不用了，可以删掉", path, key)
                continue
            raise ValueError(f"未知配置项: {path}{key}")
        current = getattr(obj, key)
        if not path and key in ("providers", "models"):
            if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
                one = "每家一个 [providers.<id>]" if key == "providers" else "每个用处一个 [models.<用处>]"
                raise ValueError(f"配置项 {key} 应该是一个表，{one}")
            for name, table in value.items():
                current.setdefault(name, {}).update(table)
        elif dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"配置项 {path}{key} 应该是一个表")
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)
    return obj


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _leaves(data: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            out += _leaves(value, f"{prefix}{key}.")
        else:
            out.append(f"{prefix}{key}")
    return out


def load_config(path: str | Path | None, overlay: str | Path | None = None) -> Config:
    """默认值 → path（config.toml）→ overlay（面板写的 console.toml）。"""
    config = Config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件: {p}")
        data = _read_toml(p)
        _merge(config, data)
        config.sources.update(dict.fromkeys(_leaves(data), "config"))
    if overlay is not None:
        o = Path(overlay)
        try:
            data = _read_toml(o)
            _merge(config, data)
        except (ValueError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"{o.name}：{exc}") from exc
        config.sources.update(dict.fromkeys(_leaves(data), "console"))
    config.icons.check()
    return config


def overlay_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """console.toml 里所有叶子键的点路径（不含面板自己的 [console]），启动时打日志用。"""
    out: list[str] = []
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out += overlay_keys(value, dotted + ".")
        elif not dotted.startswith("console."):
            out.append(dotted)
    return sorted(out)


def console_paths(config_path: Path) -> tuple[Path, Path]:
    """面板写的两个文件和 config.toml 放在一起：(console.toml, secrets.toml)。"""
    folder = Path(config_path).parent
    return folder / "console.toml", folder / "secrets.toml"


def read_secrets(path: Path) -> dict[str, str]:
    """secrets.toml 的 [env] 表：环境变量名 → 值。不存在返回 {}。"""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        env = _read_toml(path).get("env", {})
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path.name} 读不出来：{exc}") from exc
    if not isinstance(env, dict):
        raise ValueError(f"{path.name} 的 env 应该是一个表")
    for name, value in env.items():
        if not isinstance(value, str):
            raise ValueError(f"{path.name} 里的 {name} 应该是字符串")
    return dict(env)


def apply_secrets(path: Path, environ: MutableMapping[str, str]) -> list[str]:
    """secrets.toml 的值覆盖已有环境变量（否则面板里换了 Key 还在用旧的 setx）。返回写了哪些名字。"""
    secrets = read_secrets(path)
    environ.update(secrets)
    return list(secrets)


@dataclass
class Loaded:
    cfg: Config
    overridden: list[str]  # console.toml 覆盖了哪些配置项
    secrets: list[str]  # 从 secrets.toml 写进环境变量的名字（不含值）


def load_all(config_path: Path, environ: MutableMapping[str, str] = os.environ) -> Loaded:
    """命令行入口用：config.toml（可以没有）+ console.toml + secrets.toml。"""
    config_path = Path(config_path)
    overlay, secrets = console_paths(config_path)
    cfg = load_config(config_path if config_path.exists() else None, overlay if overlay.exists() else None)
    overridden = overlay_keys(_read_toml(overlay)) if overlay.exists() else []
    return Loaded(cfg, overridden, apply_secrets(secrets, environ))
