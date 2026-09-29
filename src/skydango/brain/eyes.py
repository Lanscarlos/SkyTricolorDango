"""眼睛：把截图写成文字描述（一次性的 Haiku），后台按时机自动看，缓存最新一份。

不碰设备：只读身体最近一帧（Body.last_frame）和 EnvWatcher 认出的名字位置。
大脑平时只收这份文字；要原图才调 look(image=true)。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable

import numpy as np

from ..config import BrainConfig, ProactiveConfig
from .images import fit, image_block, label_note

log = logging.getLogger(__name__)

AUTO_LOOK_KINDS = {"arrive", "leave", "stranger", "scene_change"}  # 这些事件发生后（隔够 auto_look_min）自动看一眼

EYES_SYSTEM = """你是一个《光遇》玩家的眼睛：看游戏截图，写成简短的中文文字，给另一个 AI 看。
- 只写看到的，不猜；看不清就说看不清。
- 人名只用给出的名字；给出的陌生人、团子位置照用；没列出的人都叫“陌生人”。
- 不客套，不给建议，不提问。"""

LOOK_REQUEST = """按这四项写，每项一两句：
地点和环境：像哪张图 / 什么地方、建筑、天气、白天还是晚上
好友：上面列出的每个人穿什么（斗篷、发型、面具、颜色）、在干什么；没有就写“没看到”；团子自己不用描述
陌生人：大概几个、在干什么
画面状态：有没有弹窗、黑屏、看不懂的图标；正常就写“正常”"""

# 看场合主动开口（spec 2026-09-29-proactive-chat §1）：自动看时附上上一份描述，多要一项“新鲜事”
NEWS_ITEM = """新鲜事：和上次比，有没有值得跟朋友提一句的变化？最多两条，一条一句；没有或拿不准就写“无”。
  值得提：天黑了 / 下雨了、到了新地方；好友换了斗篷 / 发型；好友在做有意思的事（弹琴、坐下、睡着冒 Z、放烟花、跳舞）；出现显眼的东西（篝火、先祖、冥龙、彩虹）
  不值得：镜头角度变了、人挪了几步、陌生人来来去去"""
LOOK_REQUEST_NEWS = LOOK_REQUEST.replace("按这四项写", "按这五项写") + "\n" + NEWS_ITEM
NEWS_MAX = 80
_PREFIX = r"^[\s\-•*#]*(?:\d+[.、)）][\s*]*)?"  # 列表符号、标题井号、编号（“5. ”）
_NEWS_HEAD = re.compile(_PREFIX + r"新鲜事[\s*]*(?:[（(][^）)]*[）)])?[\s*]*(?:[：:][\s*]*(.*)|$)")
_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.、)）])\s*")
_NEXT_ITEM = re.compile(_PREFIX + r"(?:地点和环境|好友|陌生人|画面状态)[\s*]*[：:]")  # 请求里的别的项：新鲜事到此为止
_NOTHING = re.compile(r"^(?:无|没有|暂无|none)(?:$|[，,。.；;（(、\s]|明显|变化|什么|新鲜)", re.IGNORECASE)
_NOTHING_WORDS = ("没什么变化", "没啥变化", "没有变化", "无变化", "没有明显", "无明显", "和上次差不多", "跟上次差不多", "和上次基本一样")


def parse_news(text: str) -> str:
    """眼睛描述里的“新鲜事”一项：多行条目用“；”连起来；无 / 没有 / 找不到都返回空。"""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        m = _NEWS_HEAD.match(raw.strip())
        if m is None:
            continue
        items = [m.group(1) or ""]
        # 后面的行算不算新鲜事：冒号后面已经有内容时只收列表条目；标题式（冒号后面空着）按第一条的格式，
        # 列表就只收列表、“名字：…”就只收带冒号的行 —— 格式不一样的（比如最后一句总结）到此为止
        bullets = bool(items[0].strip())
        for more in lines[i + 1 :]:
            if not more.strip():
                if any(p.strip() for p in items):
                    break
                continue  # 标题下面空一行再列条目
            if _NEXT_ITEM.match(more):
                break
            if len(items) == 1 and not items[0].strip():
                bullets = bool(_BULLET.match(more))
            elif bullets and not _BULLET.match(more) or not bullets and not re.search(r"[：:]", more):
                break
            items.append(_BULLET.sub("", more))
        parts = [p.strip().strip("*").strip().rstrip("。.").strip() for p in items]
        news = "；".join(p for p in parts if p)
        if not news or _NOTHING.match(news) or any(w in news for w in _NOTHING_WORDS):
            return ""
        return news[:NEWS_MAX]
    return ""


AROUND_REQUEST = """这是原地转一圈拍的四张图（前、右、后、左）。每个方向一两句：有什么地形 / 建筑、有没有人（认不出名字，只写几个人、穿什么、在干什么）。
最后一句总结现在在什么地方。"""


def eyes_command(base: list[str], cfg: BrainConfig) -> list[str]:
    return [
        *base, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--model", cfg.eyes_model, "--effort", "low", "--tools", "", "--strict-mcp-config",
        "--permission-mode", "dontAsk", "--disable-slash-commands", "--system-prompt", EYES_SYSTEM,
    ]


class Eyes:
    def __init__(
        self,
        cfg: BrainConfig,
        describe: Callable[[list[dict]], str],  # 内容块 → 文字（一次性 claude -p；测试里换成假的）
        frame: Callable[[], np.ndarray | None],  # 身体最近一帧
        labels: Callable[[], dict],  # 名字 → (x, y, w, h, 看到的时间)
        blackout: Callable[[], bool],
        label_keep: float = 7.0,  # 名字多久内看到过才算在这张图里
        clock: Callable[[], float] = time.monotonic,
        note: Callable[[float, float], str] | None = None,  # (now, 缩放) → 位置说明；YOLO 感知层时用 scene_note，代替 labels
        proactive: ProactiveConfig | None = None,  # 看场合主动开口；None 或 enabled = false 时照旧
        busy: Callable[[float], bool] = lambda now: False,  # 好友在不在身边：在就看得勤一点
        on_news: Callable[[str], None] = lambda text: None,  # 自动看时挑出了新鲜事（在眼睛线程里调）
    ) -> None:
        self.cfg = cfg
        self.describe = describe
        self.frame = frame
        self.labels = labels
        self.blackout = blackout
        self.label_keep = label_keep
        self.clock = clock
        self.note = note
        self.proactive = proactive if proactive is not None and proactive.enabled else None
        self.busy = busy
        self.on_news = on_news
        self.latest: tuple[str, float] | None = None
        self.last_look = float("-inf")
        self.look_request = LOOK_REQUEST
        self._poked = False
        self._lock = threading.Lock()  # 自动看和大脑要看可能撞上：同一时间只看一次

    def notice(self, kind: str) -> None:
        if kind in AUTO_LOOK_KINDS:
            self._poked = True

    def due(self, now: float) -> bool:
        if self.blackout():
            return False
        since = now - self.last_look
        longest = self._longest(now)
        return since >= longest or (self._poked and since >= self.cfg.auto_look_min)

    def _longest(self, now: float) -> float:
        """最久多久一定看一次：好友在身边时用 auto_look_busy。"""
        if self.proactive is not None and self.busy(now):
            return self.proactive.auto_look_busy
        return self.cfg.auto_look_max

    def tick(self, now: float) -> bool:
        """后台线程每秒调一次：到了时机就看一眼。返回这次有没有去看。"""
        if not self.due(now):
            return False
        frame = self.frame()
        if frame is None:
            return False
        asked = self._previous(now) is not None
        try:
            text = self.describe_frame(frame, now, news=True)
        except Exception as exc:
            log.warning("眼睛这次没看成：%s", exc)
            self.last_look = now  # 别每秒都重试，等下一个时机
            return True
        news = parse_news(text) if asked else ""
        if news:
            try:
                self.on_news(news)
            except Exception:
                log.exception("新鲜事没交出去")
        return True

    def _previous(self, now: float) -> str | None:
        """要新鲜事时拿来比的上一份描述（带多久前）；没开、没有、太旧都是 None。"""
        if self.proactive is None or self.latest is None:
            return None
        text, t = self.latest
        if now - t > self.proactive.prev_max_age:
            return None
        return f"上次（{now - t:.0f} 秒前）看到的：\n{text}\n\n"

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(1.0):
            try:
                self.tick(self.clock())
            except Exception:
                log.exception("眼睛出错")

    def describe_frame(self, frame: np.ndarray, now: float, news: bool = False) -> str:
        """news：自动看时对比上一份描述、多要一项新鲜事（大脑自己 look 不要）。"""
        view = fit(frame, tuple(self.cfg.image_size))
        scale = view.shape[1] / frame.shape[1]
        if self.note is not None:
            note = self.note(now, scale)
        else:
            recent = {n: v for n, v in self.labels().items() if now - v[4] <= self.label_keep}
            note = label_note(recent, scale)
        previous = self._previous(now) if news else None
        request = previous + LOOK_REQUEST_NEWS if previous is not None else self.look_request
        content = [image_block(view, self.cfg.jpeg_quality), {"type": "text", "text": note + "\n\n" + request}]
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def describe_around(self, frames: list[np.ndarray], now: float) -> str:
        content: list[dict] = []
        for side, frame in zip(("前", "右", "后", "左"), frames):
            content += [{"type": "text", "text": f"【{side}】"}, image_block(fit(frame, tuple(self.cfg.image_size)), self.cfg.jpeg_quality)]
        content.append({"type": "text", "text": AROUND_REQUEST if len(frames) > 1 else self.look_request})
        with self._lock:
            text = self.describe(content).strip()
        self._keep(text, now)
        return text

    def summary(self, now: float) -> str:
        if self.latest is None:
            return "场景：还没有场景描述"
        text, t = self.latest
        return f"场景（{now - t:.0f} 秒前看的）：\n{text}"

    def _keep(self, text: str, now: float) -> None:
        self.latest = (text, now)
        self.last_look = now
        self._poked = False
