"""大脑的工具：说明（给 MCP 服务用）+ 执行（参数逐个校验、每轮计数，交给身体在身体线程里做）。出错不抛异常，原因交给大脑。"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable

from .body import REQUEST_KINDS, ToolError
from .camera import KEYS

log = logging.getLogger(__name__)

DESCRIPTIONS = {  # 顺序固定：MCP 工具列表按这个顺序注册
    "look": "看现在的画面。默认让眼睛马上看一眼，返回文字描述；image=true 时返回原图（1280×720）和认出的名字位置，"
            "只在文字不够用、要自己看细节时才要原图。",
    "look_at": "放大看局部原图：坐标按 look(image=true) 那张 1280×720 的图给（左上角 x、y，宽 w，高 h），裁的就是那一张。",
    "look_around": "环顾四周：原地转一圈（每 90° 看一次），眼睛描述前 / 右 / 后 / 左各有什么，最后转回原来的朝向。要十几秒，别常用。",
    "status": "看身体现在的状态：面板和输入框开没开、身边有谁、是不是牵着手、镜头、能做的动作、互动规则、刚说过的话。",
    "chat_log": "看最近 n 条聊天记录（含你自己说的，标成“我”），n 1~50，默认 20。",
    "say": "在游戏里发一句话。一次一句，口语，短。",
    "emote": "做一个动作（只能用 status 里“能做的动作”列出的）。牵着手时会被拦下，确定要松手才传 force=true。",
    "set_request_policy": (
        "改互动请求的规则（本次运行有效）。who：好友昵称，或 \"*\" 表示所有好友、\"stranger\" 表示陌生人；"
        f"kind：{' / '.join(REQUEST_KINDS)}（hand 牵手、hug 拥抱、highfive 击掌、piggyback 背背、candle 点火、* 所有）；"
        "accept：接不接。陌生人只能接点火。"
    ),
    "camera": (
        f"转视角 / 缩放，action：{' / '.join(KEYS)}（left、right 左右转，每步约 45°；up 往上看、down 往下看；zoom_in 拉近、zoom_out 拉远）。"
        "steps 1~4，默认 1。转之前身体会关掉聊天记录面板，转完再打开。"
    ),
    "camera_reset": "把镜头转回原位（按之前转过的反着转回去）。",
    "check_friend": (
        "确认画面里某个人是不是你的好友：身体点一下这个人，右边会打开好友树面板，截图给你看，看完自动关掉。"
        "x、y 是这个人身上的一点，按 15 秒内 look(image=true) 那张 1280×720 的图给。"
        "只在名字标签认不出、又确实需要知道时用（点过火的陌生人和好友长得一样）；点屏幕会暂时关掉聊天记录面板，别常用。"
    ),
}
SWEEP_AROUND = "环顾四周：原地连续转一圈，身体认出每个方向有谁（好友名字、几个陌生人），返回文字，最后回到原来的朝向。几秒就好。"


def descriptions(sweep: bool) -> dict[str, str]:
    """工具说明；打开感知层（sweep=True）时 look_around 换成 YOLO 环绕扫描的说法。"""
    return {**DESCRIPTIONS, "look_around": SWEEP_AROUND} if sweep else dict(DESCRIPTIONS)


TOOL_NAMES = list(DESCRIPTIONS)
ACTIONS = {"say", "emote", "set_request_policy", "camera", "camera_reset", "check_friend"}  # 算“做了事”的工具（心跳退档用）
AROUND_TIMEOUT = 30.0  # 环顾一圈要关面板、转四次，比一般命令慢

_MISSING = object()


def _get(args: dict, key: str, default, kind: type, label: str):
    value = args.get(key, default)
    if value is _MISSING:
        raise ToolError(f"缺少参数 {key}")
    if (kind is int and isinstance(value, bool)) or not isinstance(value, kind):
        raise ToolError(f"参数 {key} 应该是{label}")
    return value


def _int(args: dict, key: str, default=_MISSING) -> int:
    return _get(args, key, default, int, "整数")


def _str(args: dict, key: str, default=_MISSING) -> str:
    return _get(args, key, default, str, "字符串")


def _bool(args: dict, key: str, default=_MISSING) -> bool:
    return _get(args, key, default, bool, " true / false")


class ToolBox:
    def __init__(self, body, eyes=None, max_steps: int = 6, max_says: int = 2) -> None:
        self.body = body
        self.eyes = eyes
        self.max_steps = max_steps
        self.max_says = max_says
        self._lock = threading.Lock()  # Claude Code 可能并行调工具：计数要加锁
        self.begin_turn()

    def begin_turn(self) -> None:
        """每次醒来开始时清零：这一轮最多 max_steps 次工具、max_says 句话。"""
        with self._lock:
            self.calls = 0
            self.says = 0
            self.acted = False
            self.used: list[str] = []

    def status(self) -> str:
        """醒来时拼消息用：不算这一轮的工具次数。"""
        try:
            return self.body.call(self.body.status)
        except ToolError as exc:
            return f"（状态读不到：{exc}）"

    def run(self, name: str, args: dict) -> tuple[str | list[dict], bool]:
        """执行一个工具，返回 (结果内容, 是不是出错)。"""
        with self._lock:
            self.calls += 1
            self.used.append(name)
            if self.calls > self.max_steps:
                return "这一轮做的事够多了，先停下，下次醒来再说", True
            if name == "say":
                if self.says >= self.max_says:
                    return "这一轮已经说得够多了，别刷屏", True
                self.says += 1  # 先占上名额：并行调 say 也不会超；没说成再还回去
        try:
            out = self._exec(name, args or {})
        except Exception as exc:
            if name == "say":
                with self._lock:
                    self.says -= 1
            if not isinstance(exc, ToolError):
                log.exception("工具 %s 出错", name)
                return f"出错了：{exc}", True
            return str(exc), True
        if name in ACTIONS:
            self.acted = True
        log.info("工具 %s %s → %s", name, json.dumps(args or {}, ensure_ascii=False), out if isinstance(out, str) else "[图片]")
        return out, False

    def _exec(self, name: str, a: dict):
        b = self.body
        if name == "look":
            image = _bool(a, "image", False)
            if image or self.eyes is None:
                return b.call(b.look)
            now = b.clock()
            if self.eyes.latest is not None and now - self.eyes.last_look < b.cfg.brain.look_min_interval:
                return self.eyes.summary(now)
            return self.eyes.describe_frame(b.call(b.fresh_frame), now)
        if name == "look_around":
            if hasattr(getattr(b, "env", None), "sweep"):  # 打开了感知层：转一圈交给 YOLO，不叫眼睛
                return b.call(b.sweep_around, timeout=AROUND_TIMEOUT)
            if self.eyes is None:
                raise ToolError("没开眼睛，看不了四周")
            frames = b.call(b.capture_around, timeout=AROUND_TIMEOUT)
            return self.eyes.describe_around(frames, b.clock())
        return b.call(self._bind(name, a))

    def _bind(self, name: str, a: dict) -> Callable[[], object]:
        b = self.body
        if name == "look_at":
            x, y, w, h = _int(a, "x"), _int(a, "y"), _int(a, "w"), _int(a, "h")
            return lambda: b.look_at(x, y, w, h)
        if name == "status":
            return b.status
        if name == "chat_log":
            n = _int(a, "n", 20)
            return lambda: b.chat_log(n)
        if name == "say":
            text = _str(a, "text")
            return lambda: b.say(text)
        if name == "emote":
            emote, force = _str(a, "name"), _bool(a, "force", False)
            return lambda: b.emote(emote, force)
        if name == "set_request_policy":
            who, kind, accept = _str(a, "who"), _str(a, "kind"), _bool(a, "accept")
            return lambda: b.set_policy(who, kind, accept)
        if name == "camera":
            action, steps = _str(a, "action"), _int(a, "steps", 1)
            return lambda: b.camera_move(action, steps)
        if name == "camera_reset":
            return b.camera_reset
        if name == "check_friend":
            x, y = _int(a, "x"), _int(a, "y")
            return lambda: b.check_friend(x, y)
        raise ToolError(f"没有这个工具：{name}")
