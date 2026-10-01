"""配置加载。所有字段都有默认值，config.toml 只需要写想覆盖的部分。"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from collections.abc import MutableMapping
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
    match_threshold: float = 0.8  # 在动作列表里找图标（和图标库同尺寸）
    slot_match_threshold: float = 0.72  # 认轮盘格子里的图标（更大，还可能带“2级”之类的字）
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
    light_after: float = 3.0  # 黑影身上的火焰圆盘连续看到这么久才举蜡烛（别点路过的）
    light_timeout: float = 8.0  # 举着蜡烛最多等这么久看他亮起来（YOLO player_unlit → player）
    lit_frames: int = 3  # YOLO 连续几帧认成 player 才算点亮
    lit_min: float = 2.0  # 举蜡烛后至少这么久才可能算点亮（YOLO 近处会把还黑着的人认成 player；未在真机验证）
    lit_iou: float = 0.3  # lit()：替身 player 和原轨迹最后的框至少重叠这么多才算同一个人（未在真机验证）
    lit_stale: float = 0.5  # lit()：原轨迹这么久没接上检测就当冻住了（闪光时出现重复轨迹、原轨迹框冻住；未在真机验证）
    after_light: str = "鞠躬"  # 点亮别人 / 接受别人点火后做的动作（轮盘上要有；鞠躬顺带放下蜡烛），空 = 不做
    bow_delay: float = 2.5  # 看到他亮起来 / 接受点火后等这么久再鞠躬（等闪光动画）
    candle_slot: int = 3  # 轮盘上"举蜡烛"在第几格：按一下举起、再按一下放下（用户 2026-10-01）
    disk_min_score: float = 0.68  # 火焰圆盘的模板匹配分
    disk_dark: float = 90.0  # 圆盘外环亮度均值低于这个才算深色圆盘（有白圈的是举蜡烛的请求）
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
    min_height: float = 0.10  # 好样本的框高（相对截图高度）
    max_overlap: float = 0.2  # 和别的框 / 面板重叠超过这个比例就不是好样本
    ema: float = 0.2  # 平均特征的更新速度
    min_samples: int = 3  # 判 maybe / 陌生人编号 / 换装前至少几个好样本
    match: float = 0.85  # 估的
    card_match: float = 0.92  # 只有关系卡旧特征时的门槛（更严）
    margin: float = 0.05  # 最像的要比第二像的高这么多
    recheck: int = 5  # maybe 连续这么多个新样本低于门槛就摘掉
    changed: float = 0.70  # 低于它算换了装
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
    labels: list[str] = field(default_factory=lambda: ["none", "wave", "bow"])  # 模型输出的顺序；none = 没做这几个动作
    names: dict[str, str] = field(default_factory=lambda: {"wave": "挥手", "bow": "鞠躬"})  # 事件里的中文名
    frames: int = 16  # 一段几帧（2 s）
    fps: float = 8.0  # 每秒取几帧
    size: int = 112  # 人物裁剪缩放到的边长
    interval: float = 2.0  # 同一个人隔这么久判一次
    min_prob: float = 0.9  # 概率到这个才报（报错动作很尴尬，宁可不报）
    cooldown: float = 30.0  # 同一个人同一种动作这么久内只报一次


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
    """识别过程可视化：本机网页实时显示画面 + 识别框（`run --view` 或 `python -m skydango view`）。"""

    host: str = "127.0.0.1"  # 只给本机看：画面里有好友昵称和聊天
    port: int = 19399
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
    cold_after: int = 3  # 连续几句主动的话没人接就暂停主动，等有好友说话
    self_names: list[str] = field(default_factory=lambda: ["团子", "三彩"])  # 聊天里出现这些字算“叫了你”


@dataclass
class AttentionConfig:
    """空闲注意力（东张西望，spec 2026-09-30-idle-attention）：闲着时按兴趣小步转镜头看说话 / 走近 / 对团子做事的人，
    没什么可看就随意看看；只在聊天面板 auto 模式、面板关着时动。数字都是估的，真机调。按键长短、settle、转不动沿用 [track]。"""

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
    wander_min: float = 8.0  # 没什么可看时，隔 wander_min~wander_max 秒（再乘心情精力 / 模式的倍数）随意看一眼
    wander_max: float = 20.0
    wander_presses: list[int] = field(default_factory=lambda: [2, 4])  # 随意看一次连按几下
    wander_same_side: int = 2  # 往同一边连着随意看最多几次
    max_step: float = 1.0  # 每圈时间差上限（沙盒模拟时钟会一下跳几小时）


@dataclass
class ReflexConfig:
    """身体反射（见 docs/superpowers/specs/2026-09-30-body-reflex-design.md）：有人跟团子说话马上冒输入气泡、回礼、闲着做小动作。数字都是估的"""

    enabled: bool = True  # false = 完全照旧（不开框、不做反射动作、gesture 照旧交给大脑）
    bubble: bool = True  # 有人跟团子说话时身体马上打开输入框（头顶“正在输入”），大脑想好了用这个框发
    followup_window: float = 30.0  # 团子说完多少秒内好友接话，算“在跟团子说”
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

    port: int = 19390  # 面板端口；和 view 的 19399 错开，可以同时开
    child_port: int = 19391  # 面板起的团子子进程的 viewer 端口
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
    friend_check: FriendCheckConfig = field(default_factory=FriendCheckConfig)
    panels: PanelsConfig = field(default_factory=PanelsConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    spin: SpinConfig = field(default_factory=SpinConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    peek: PeekConfig = field(default_factory=PeekConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    inner: InnerConfig = field(default_factory=InnerConfig)
    backstage: BackstageConfig = field(default_factory=BackstageConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


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


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: str | Path | None, overlay: str | Path | None = None) -> Config:
    """默认值 → path（config.toml）→ overlay（面板写的 console.toml）。"""
    config = Config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件: {p}")
        _merge(config, _read_toml(p))
    if overlay is not None:
        o = Path(overlay)
        try:
            _merge(config, _read_toml(o))
        except (ValueError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"{o.name}：{exc}") from exc
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
