"""反思（spec 2026-09-30-inner-phase2 §2）：隔一阵回头想想刚才发生的事，更新心情、别扭、心愿；下线前再写日记和要点。

模型只给建议（JSON），收不收由 mind.Mind.apply 定。一次性 claude -p（和随手记同一套 ClaudeLlm），跑在后台线程里，
结果放进队列，身体线程每圈取走再套进 Mind（Mind 不用锁）。什么时候跑由身体报的动静决定。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from ..brain.claude import ClaudeError
from ..chat.memory import format_date
from ..config import InnerConfig
from .mind import parse_reflection

log = logging.getLogger(__name__)

LIMIT_RETRY = 600.0  # 额度用完后多久不再反思（同 [brain] limit_retry 默认）
MAX_CHAT = 80  # 材料里最多带几行聊天

REFLECT_SYSTEM = """你是《光遇》里的三彩团子（人设见材料），现在回头想想刚才这段时间，更新自己的内心状态。材料里有：现在几点、精力、你现在心里的状态、这段时间的聊天原话（“我”是你自己说的）、谁来了谁走了、相关好友的关系卡和笔记。

按人设和发生的事来想：
- 心情（mood）：level 只能是 开心 / 平常 / 低落 / 烦 之一；text 是一句带原因的口语（不超过 40 字），比如“有点闷，小明说好来又没来”。别凭空大起大落，没什么事就保持原样。
- 闹别扭（grudge）：只在有由头时对熟悉的好友闹点小别扭（被放鸽子、被冷落、被开过分的玩笑），写 {"who": 好友名, "why": 一句原因}；对方已经道歉、认真哄过、或者说自己难过，就撤掉（写 null）；不变就写 "keep"。只是小脾气，不记仇。
- 心愿（wants_add / wants_done）：kind 只能是 惦记 / 想做 / 小心思。惦记 = 好友说过的、以后想问问的事（考试、生病、约好的事），who 写那个好友；想做 = 想和大家一起做、要别人配合的事（看日落、听谁弹琴、围着篝火坐一会儿）；小心思 = 游戏里的事或人设里已有喜好的延伸（想换个发型、最近很想去看星星）。不编上学、上班、生病这类现实生活里的事。已经了结的写进 wants_done（照原文）。没有新的就给空列表。
- 不记隐私：年龄、学校、住址、电话、微信 / QQ 一律不写（对方可能是未成年人）。

只输出一个 JSON 对象，不要解释：
{"mood": {"level": "...", "text": "..."}, "grudge": "keep", "wants_add": [{"kind": "...", "text": "...", "who": "..."}], "wants_done": ["..."], "diary": "", "memos": []}

材料最后说“这是今天下线前的最后一次”时，再写：diary = 一段第一人称日记（不超过 200 字，口语，写今天和谁玩了什么、心情怎么样）；memos = 给以后的自己的要点（不超过 5 条，每条不超过 30 字，写清楚是谁、具体日期，只记以后用得上的）。其他时候这两项留空。"""

# 第 3 期（spec 2026-09-30-inner-phase3 §3）：persona 开着时接在 REFLECT_SYSTEM 后面
PERSONA_SYSTEM = """顺便想想你自己的性格（材料里“你攒下的性格”是已经记下的）：
- 口头禅（catchphrases）：这段时间你自己说过、有人接或者笑了的说法，记下原话（不超过 30 字）。
- 老梗（jokes）：你和某个熟悉的好友之间反复提的事、外号，写 {"who": 好友名, "text": 一句（不超过 30 字）}。只给一起玩过好几天的熟人记，陌生人、刚认识的不记。
- 看法（opinions）：你表过态的话题，写 {"topic": 话题（不超过 10 字）, "stance": 你的立场（不超过 30 字）}；同一个话题改了主意就写新的立场。
- 这段时间你用了已经记下的口头禅 / 老梗 / 看法，而且有人接，把它的原文（口头禅和老梗写 text，看法写 topic）写进 persona_used。
- 损人的、拿外貌 / 身材 / 家里 / 成绩 / 年龄开玩笑的一律不记。没有就给空的，别硬凑。

在上面那个 JSON 对象里多加两项：
"persona_add": {"catchphrases": ["..."], "jokes": [{"who": "...", "text": "..."}], "opinions": [{"topic": "...", "stance": "..."}]}, "persona_used": ["..."]"""


def _block(title: str, lines: list[str]) -> str:
    return f"{title}\n" + ("\n".join(lines) if lines else "（没有）")


def materials(
    now: float,
    energy_note: str,
    mind_line: str,
    chat: list[tuple[float, str, str]],  # (墙上时间, 说话人, 内容)；团子自己是“我”
    comings: list[str],  # “小明 来到身边”这类
    cards: list[str],  # 相关好友的关系卡（card_line）
    notes: list[str],  # notes.md / inbox.md 里提到他们的行
    persona: str,
    final: bool,
    traits: str | None = None,  # 第 3 期：已经攒下的性格（Persona.section()）；None = 不写这一段，"" = 还没有
) -> str:
    """拼给反思的材料（一条 user 消息）。"""
    d = time.localtime(now)
    parts = [
        f"现在：{format_date(now)} {d.tm_hour:02d}:{d.tm_min:02d}",
        f"精力：{energy_note or '（不知道）'}",
        f"你现在心里：{mind_line or '平常'}",
        _block("人设：", [persona.strip()] if persona.strip() else []),
        _block("这段时间的聊天（“我”是你自己说的）：", [f"{who or '（看不出是谁）'}：{text}" for _, who, text in chat[-MAX_CHAT:]]),
        _block("这段时间谁来了谁走了：", comings),
    ]
    if traits is not None:
        body = [line for line in traits.splitlines() if line.strip() and not line.startswith("## ")]
        parts.append("你攒下的性格：\n" + ("\n".join(body) if body else "（还没有）"))
    parts += [
        _block("相关的好友：", cards),
        _block("笔记里提到他们的：", notes),
    ]
    if final:
        parts.append("这是今天下线前的最后一次：再写一段第一人称日记（不超过 200 字）和给以后的自己的要点（不超过 5 条）。")
    return "\n\n".join(parts)


class Reflector:
    def __init__(self, cfg: InnerConfig, llm, clock: Callable[[], float] = time.monotonic, threaded: bool = True,
                 system: str = REFLECT_SYSTEM) -> None:
        self.cfg = cfg
        self.system = system  # 第 3 期：persona 开着时是 REFLECT_SYSTEM + PERSONA_SYSTEM
        self.llm = llm  # complete(system, messages) -> str（brain.claude.ClaudeLlm）
        self.clock = clock
        self.threaded = threaded  # 测试里同步跑
        now = clock()
        self._last_run = now  # 上线后先攒一阵再反思
        self._stirred = False  # 上次反思以来有没有动静
        self._lines = 0  # 上次反思以来好友说了几句
        self._last_line = float("-inf")
        self._paused_until = float("-inf")
        self._running = False
        self._final = False  # 下线前的最终反思开始了：后台的结果一律丢掉
        self._results: queue.Queue[dict] = queue.Queue()

    @property
    def running(self) -> bool:
        return self._running

    # ---- 身体报动静 ----
    def heard(self, friend: bool, now: float) -> None:
        self._stirred = True
        if friend:
            self._lines += 1
            self._last_line = now

    def stirred(self, now: float) -> None:
        self._stirred = True

    # ---- 什么时候跑 ----
    def due(self, now: float) -> bool:
        if self._running or self._final or now < self._paused_until:
            return False
        if self._stirred and now - self._last_run >= self.cfg.reflect_every:
            return True
        return self._lines >= self.cfg.reflect_min_lines and now - self._last_line >= self.cfg.reflect_after_quiet

    def start(self, content: str, now: float) -> None:
        self._last_run = now
        self._stirred = False
        self._lines = 0
        self._running = True
        if self.threaded:
            threading.Thread(target=self._run, args=(content,), name="reflect", daemon=True).start()
        else:
            self._run(content)

    def poll(self) -> dict | None:
        """后台反思的结果（身体线程每圈取）；最终反思开始后永远是 None。"""
        if self._final:
            return None
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def final(self, content: str) -> dict | None:
        """下线前的最终反思：同步跑（最多 reflect_timeout，由 llm 的超时管），额度暂停中也试一次。"""
        self._final = True
        return self._ask(content)

    # ---- 内部 ----
    def _run(self, content: str) -> None:
        try:
            result = self._ask(content)
        finally:
            self._running = False
        if result is not None and not self._final:
            self._results.put(result)

    def _ask(self, content: str) -> dict | None:
        try:
            raw = self.llm.complete(self.system, [{"role": "user", "content": content}])
        except ClaudeError as exc:
            if exc.limit:
                self._paused_until = self.clock() + LIMIT_RETRY
                log.warning("反思：订阅额度用完了，%.0f 秒内不再反思", LIMIT_RETRY)
            else:
                log.warning("反思失败，沿用上一份：%s", exc)
            return None
        except Exception:
            log.exception("反思出错，沿用上一份")
            return None
        result = parse_reflection(raw)
        if result is None:
            log.warning("反思的回答不是 JSON，沿用上一份：%s", (raw or "")[:120])
        return result
