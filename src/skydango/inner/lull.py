"""冷场时的心理活动（spec 2026-10-01-lull-musing §1~§3）：认出冷场、按节点出叫醒大脑的文字、记团子心里想过的话。

纯计算、不碰设备：身体每圈把 墙钟、身边的好友、聊天（body.chat 的 (墙钟, 说话人, 内容)，团子是“我”）交进来，
拿回这一圈要叫醒大脑的 Cue；好友开口时调 heard() 结束冷场、拿回附在聊天后面的话。时间都用墙钟。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..chat.tracker import similar
from ..config import LullConfig

ME = "我"  # body.chat 里团子自己的说话人
QUOTE_MAX = 30  # 引用的话最多几个字
_MUSING = re.compile(r"^\s*心里\s*[：:]\s*(.+?)\s*$", re.M)

Chat = Sequence[tuple[float, str, str]]


@dataclass
class Lull:
    kind: str  # "silent" 在身边不说话了 / "left" 聊着聊着走了
    who: tuple[str, ...]  # 冷场对象（身边名单里的名字）；silent 按最后说话时间排，最后说话的在最后
    t0: float  # 墙钟：silent = 冷场前最后一句的时间；left = 走开的时间（黑屏时往后挪：不推进）
    last: tuple[str, str] | None  # 冷场前最后一句 (说话人, 内容)；团子是“我”
    stage: int = -1  # 已经叫醒到第几个节点（stages 的下标）
    musings: list[tuple[float, float, str]] = field(default_factory=list)  # (墙钟, 冷了几秒, 想法)
    said: list[str] = field(default_factory=list)  # 冷场中团子又说的
    ending: str = ""  # 怎么结束的（给反思材料的总结用）
    ended_at: float | None = None
    created: float = float("-inf")  # 墙钟：开始追踪的时间（“心里：”只挂到大脑这一轮开始之前就有的冷场上）
    announced: bool = True  # left 先等 leave_grace 秒才叫醒（名字标签闪一下不算走开）；还没叫醒的不进状态、不挂想法
    opening: str = ""  # left 叫醒时的那一句


@dataclass(frozen=True)
class Cue:
    text: str  # 给大脑看的一行（lull 事件）
    final: bool  # 这次冷场的最后一个节点


def parse_musing(text: str, limit: int) -> str:
    """大脑这一轮最后的文字里，最后一行“心里：……”的内容（截 limit 字）；没有返回空。"""
    found = _MUSING.findall(text or "")
    return found[-1].strip()[:limit] if found else ""


def _dur(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds)} 秒" if seconds < 60 else f"{int(seconds // 60)} 分钟"


def _q(text: str) -> str:
    return text if len(text) <= QUOTE_MAX else text[:QUOTE_MAX] + "…"


class LullTracker:
    def __init__(self, cfg: LullConfig, is_friend: Callable[[str], bool]) -> None:
        self.cfg = cfg
        self.is_friend = is_friend  # 说话人是不是好友（模糊匹配好友名单；“我”和看不出是谁的不是）
        self._lulls: list[Lull] = []
        self._finished: list[Lull] = []
        self._done_t0 = float("-inf")  # 上一次情况①冷场的 t0：同一句“最后一句”不再开始
        self._last_wall: float | None = None

    # ---- 每圈 ----
    def tick(self, wall: float, nearby: Sequence[str], chat: Chat, paused: bool = False) -> list[Cue]:
        prev, self._last_wall = self._last_wall, wall
        if paused:  # 黑屏（切场景）：不开始、不发节点，冷场时间不推进
            if prev is not None:
                for lull in self._lulls:
                    lull.t0 += max(0.0, wall - prev)
            return []
        silent = self._silent()
        if silent is None:
            self._start_silent(wall, nearby, chat)
        else:
            self._update_silent(silent, nearby, chat, wall)
        stages, cues = self.cfg.stages, []
        for lull in list(self._lulls):
            if not lull.announced:
                if wall - lull.t0 >= self.cfg.leave_grace:
                    lull.announced = True
                    cues.append(Cue(lull.opening, False))
                continue
            cue = self._advance(lull, wall)
            if cue is not None:
                cues.append(cue)
            elif lull.stage == len(stages) - 1 and wall - lull.t0 >= stages[-1] + stages[0]:
                # 最后一个节点之后再过一阵：冷场变成普通的安静（spec §1），留下总结
                self._end(lull, wall, "一直没回来" if lull.kind == "left" else "后来就一直安静着")
        return cues

    def _silent(self) -> Lull | None:
        return next((lull for lull in self._lulls if lull.kind == "silent"), None)

    def _start_silent(self, wall: float, nearby: Sequence[str], chat: Chat) -> None:
        if not chat:
            return
        t, who, text = chat[-1]
        if wall - t < self.cfg.stages[0] or t <= self._done_t0:
            return
        spoke: dict[str, float] = {}
        for ct, speaker, _ in chat:
            if t - self.cfg.talk_window <= ct <= t and self.is_friend(speaker):
                name = _match(speaker, nearby)
                if name is not None:
                    spoke[name] = max(spoke.get(name, ct), ct)
        if spoke:
            self._lulls.append(Lull("silent", tuple(sorted(spoke, key=spoke.get)), t, (who, text), created=wall))

    def _update_silent(self, lull: Lull, nearby: Sequence[str], chat: Chat, wall: float) -> None:
        still = tuple(n for n in lull.who if n in nearby)
        if not still:  # 对象都走了（冷了很久才走、或镜头转开）：结束，留下总结；聊着聊着走的已经在 left() 里交给情况②
            self._end(lull, wall, "后来他走开了")
            return
        lull.who = still
        lull.said = [text for t, who, text in chat if t > lull.t0 and who == ME]

    # ---- 情况②：聊着聊着走了 ----
    def left(self, name: str, wall: float, chat: Chat) -> bool:
        """好友走开时调：他刚说过话、或团子刚说过话，就开始一个“走开”的冷场（leave_grace 秒后才叫醒），返回 True。"""
        spoke = any(t >= wall - self.cfg.leave_spoke and similar(who, name, 0.75) for t, who, _ in chat)
        said = any(t >= wall - self.cfg.leave_said and who == ME for t, who, _ in chat)
        if not (spoke or said) or not chat:
            return False
        musings: list[tuple[float, float, str]] = []
        silent = self._silent()
        if silent is not None and name in silent.who:
            silent.who = tuple(n for n in silent.who if n != name)
            if not silent.who:  # 冷场对象只剩他：想过的话跟着转过来
                self._lulls.remove(silent)
                self._done_t0 = max(self._done_t0, silent.t0)
                musings = silent.musings
        self._lulls = [lull for lull in self._lulls if not (lull.kind == "left" and lull.who == (name,))]
        _, who, text = chat[-1]
        said_by = "你" if who == ME else "他" if similar(who, name, 0.75) else (who or "（看不出是谁）")
        opening = f"冷场  {name} 聊着聊着走开了。走之前最后是{said_by}说的「{_q(text)}」。"
        self._lulls.append(Lull("left", (name,), wall, (who, text), stage=0, musings=musings, created=wall,
                                announced=False, opening=opening))
        return True

    def returned(self, name: str, wall: float) -> str | None:
        """好友回来：有他“走开”的冷场就结束、返回附注；还没叫醒过（闪了一下）就悄悄结束、返回 None。"""
        lull = next((x for x in self._lulls if x.kind == "left" and x.who == (name,)), None)
        if lull is None:
            return None
        if not lull.announced:
            self._quiet_end(lull, wall, "很快就回来了")
            return None
        note = self._left_note(lull, wall)
        self._end(lull, wall, "后来他回来了")
        return note

    def _quiet_end(self, lull: Lull, wall: float, ending: str) -> None:
        """还没叫醒过的“走开”：没想过就当没发生；带着从情况①转来的想法时照样留下总结。"""
        if lull.musings:
            self._end(lull, wall, ending)
        else:
            self._lulls.remove(lull)

    def _left_note(self, lull: Lull, wall: float) -> str:
        thought = self._thought(lull)
        return f"（走开了 {_dur(wall - lull.t0)}" + (f"，你刚才在想：{thought}）" if thought else "）")

    def _advance(self, lull: Lull, wall: float) -> Cue | None:
        stages = self.cfg.stages
        passed = [i for i, s in enumerate(stages) if wall - lull.t0 >= s]
        if not passed or passed[-1] <= lull.stage:
            return None
        lull.stage = passed[-1]  # 一次跳过几个节点（快进）只发最新的
        return Cue(self._cue_text(lull, wall), lull.stage == len(stages) - 1)

    # ---- 文字 ----
    def _obj(self, lull: Lull) -> str:
        return lull.who[0] if len(lull.who) == 1 else f"大家（{'、'.join(lull.who)}）"

    def _cue_text(self, lull: Lull, wall: float) -> str:
        if lull.kind == "left":
            return f"冷场  {lull.who[0]} 走开 {_dur(wall - lull.t0)}了，还没回来。"
        speaker, text = lull.last or ("", "")
        if speaker == ME:
            tail = f"最后是你说的「{_q(text)}」，{'他没接' if len(lull.who) == 1 else '没人接'}。"
        else:
            tail = f"最后是{speaker or '（看不出是谁）'}说的「{_q(text)}」，你没接。"
        if lull.said:
            tail += f"这之间你又说了「{_q(lull.said[-1])}」。"
        return f"冷场  {self._obj(lull)} {_dur(wall - lull.t0)}没说话了。{tail}"

    def _thought(self, lull: Lull) -> str:
        return lull.musings[-1][2] if lull.musings else ""

    def status(self, wall: float) -> str:
        parts = []
        for lull in self._live():
            if lull.kind == "left":
                head = f"{lull.who[0]} 走开 {_dur(wall - lull.t0)}了 "
                head += f"· 在想：{self._thought(lull)}（{_dur(lull.musings[-1][1])}时）" if lull.musings else "· 在想：（还没想过）"
                parts.append(head)
                continue
            speaker, text = lull.last or ("", "")
            said = "你" if speaker == ME else (speaker or "（看不出是谁）")
            head = f"{self._obj(lull)} {_dur(wall - lull.t0)}没说话（最后是{said}说的「{_q(text)}」）"
            if lull.musings:
                _, cold, thought = lull.musings[-1]
                head += f"· 在想：{thought}（{_dur(cold)}时）"
            else:
                head += "· 在想：（还没想过）"
            parts.append(head)
        return "冷场：" + "；".join(parts) if parts else ""

    def summary(self, lull: Lull, wall: float) -> str:
        """给反思材料的一行。"""
        end = lull.ended_at if lull.ended_at is not None else wall
        speaker, _ = lull.last or ("", "")
        dur = _dur(end - lull.t0)
        if lull.kind == "left":
            out = f"{lull.who[0]}聊着聊着走开了 {dur}。"
        elif speaker == ME:
            out = f"{dur}，{self._obj(lull)}没接你的话。"
        else:
            out = f"{dur}，{self._obj(lull)}说完你没接，大家都没再说话。"
        if lull.musings:
            out += "你心里想过：" + " → ".join(m for _, _, m in lull.musings) + "。"
        return out + lull.ending

    # ---- 好友开口、想法 ----
    def heard(self, wall: float, speaker: str, text: str) -> str:
        """好友开口：结束冷场，返回附在这条聊天后面的话；没有冷场（或不是好友）返回空。"""
        if not self.is_friend(speaker):
            return ""
        note = ""
        silent = self._silent()
        if silent is not None:
            thought = self._thought(silent)
            note = f"（冷场了 {_dur(wall - silent.t0)}" + (f"，你刚才在想：{thought}）" if thought else "）")
            self._end(silent, wall, f"后来{speaker}说「{_q(text)}」")
        for lull in [x for x in self._lulls if x.kind == "left" and similar(speaker, x.who[0], 0.75)]:
            if not lull.announced:
                self._quiet_end(lull, wall, f"后来他说「{_q(text)}」")
                continue
            note = note or self._left_note(lull, wall)
            self._end(lull, wall, f"后来他说「{_q(text)}」")
        return note

    def _end(self, lull: Lull, wall: float, ending: str) -> None:
        self._lulls.remove(lull)
        lull.ending, lull.ended_at = ending, wall
        self._finished.append(lull)
        if lull.kind == "silent":
            self._done_t0 = max(self._done_t0, lull.t0)

    def muse(self, text: str, wall: float, since: float | None = None) -> bool:
        """把大脑心里想的挂到活着的冷场上；since（墙钟，大脑这一轮开始的时间）之后才开始的不挂。没有可挂的返回 False（这句丢掉）。"""
        targets = [lull for lull in self._live() if since is None or lull.created <= since]
        for lull in targets:
            lull.musings.append((wall, wall - lull.t0, text))
        return bool(targets)

    def _live(self) -> list[Lull]:
        """已经叫醒过大脑的冷场（还在宽限里的“走开”不算）。"""
        return [lull for lull in self._lulls if lull.announced]

    # ---- 取出 ----
    def active(self) -> list[Lull]:
        return list(self._lulls)

    def pop_finished(self) -> list[Lull]:
        out, self._finished = self._finished, []
        return out

    def flush(self, wall: float) -> list[Lull]:
        """下线前：活着的都当作结束取走。"""
        out = self._lulls
        for lull in out:
            lull.ending, lull.ended_at = "到下线还没结束", wall
        self._lulls = []
        return out

    def snapshot(self, wall: float) -> list[dict]:
        return [
            {
                "kind": lull.kind,
                "who": list(lull.who),
                "since": lull.t0,
                "last": list(lull.last) if lull.last else None,
                "musings": [{"t": t, "text": m} for t, _, m in lull.musings],
            }
            for lull in self._live()
        ]


def _match(speaker: str, names: Sequence[str]) -> str | None:
    """聊天里的说话人对应身边名单里的哪个名字（OCR 可能错一两个字）。"""
    return next((n for n in names if similar(speaker, n, 0.75)), None)
