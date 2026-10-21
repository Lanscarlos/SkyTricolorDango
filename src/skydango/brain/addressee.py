"""分清好友在跟谁说话：纯规则 + 一点对话状态，不调模型（spec docs/superpowers/specs/2026-10-05-addressee-design.md §3）。

身体每读到一句好友聊天就 `judge` 一次，拿回 `Verdict`（跟你说 / 跟别人说 / 说给大家 / 拿不准）；团子说话时调 `said`。
时间都用身体的 clock；模块只记最近 2 分钟的聊天和每句的判断。

团子说话时按它前面那批没回的好友聊天定接话窗口给谁（10-11 修规则 4 的连锁）：没有这批 = 主动开口，谁接都算；
这批里有跟你说 / 说给大家 / 拿不准的 = 在回这些人，只给他们开；整批都是跟别人说的 = 插话，谁都不开，旁边两个人的对话串接着算。
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..chat.tracker import similar
from ..config import AddresseeConfig
from ..inner.mind import sounds_upset
from .occasion import ME

LABEL_NAMES = {"me": "跟你说", "other": "跟别人说", "all": "说给大家", "unsure": "拿不准"}

KEEP = 120.0  # 对话状态只留最近这么多秒（团子上次开口之后、还留着的好友聊天就是「它在回的那批」）
CONTINUE = 20.0  # 上一句之后这么多秒内、中间没人说话又开口 = 接着上一句说（同 body.FOLLOWUP_GAP），沿用上一句的对象
TRAIL = "~～!！?？。.，,、… \t"  # 判句尾的招呼词前先去掉的标点和空白
CALLED = "叫了你"  # 规则 1 的理由；判断里也拿它认「叫了团子」
CROSSED = "（跨过你的话）"  # 跨过团子的话判出来的对话串，理由后面带上，离线评估能分开统计
FUZZY_MIN = 3  # 叫法至少这么多个字才允许错一个字（2 个字错一个就只剩一个字对得上）
VOCATIVE_NEXT = "你快在呢呀啊"  # 句首的名字后面紧跟这些 = 在喊他（不在身边也算叫了他）


@dataclass(frozen=True)
class Verdict:
    label: str  # me / other / all / unsure
    reason: str  # 一句人话，进事件和日志
    target: str = ""  # other 时在跟谁说（好友名单里的名字）

    def tag(self) -> str:
        name = LABEL_NAMES[self.label]
        return f"{name}：{self.reason}" if self.reason else name


def parse_aliases(friends_md: str, names: Sequence[str]) -> dict[str, list[str]]:
    """好友名 → [昵称, *叫法]。叫法来自 friends.md 里 `## 昵称` 那一节的 `- 叫法：甲、乙` 一行（全角 / 半角冒号都行），没有就只有昵称。"""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in (friends_md or "").splitlines():
        if line.startswith("## "):
            current = found.setdefault(line[3:].strip(), [])
        elif line.startswith("#"):
            current = None
        elif current is not None:
            m = re.match(r"^\s*[-*]\s*叫法\s*[：:]\s*(.*)$", line)
            if m:
                current.extend(a for a in re.split(r"[、，,；;\s]+", m.group(1)) if a)
    out: dict[str, list[str]] = {}
    for name in names:
        words = [name]
        for a in found.get(name, []):
            if a not in words:
                words.append(a)
        out[name] = words
    return out


def _ends_clause(text: str, word: str) -> bool:
    """句尾（去掉标点）是单独一个分句的这个词：「我先下了 晚安」「困了，拜拜~」算，「bug又来了」「他刚走了」不算。"""
    tail = text.rstrip(TRAIL)
    return tail.endswith(word) and (len(tail) == len(word) or tail[-len(word) - 1] in TRAIL)


def _to_group(text: str, word: str) -> bool:
    """「大家 / 你们」是在招呼一群人：开头就是，或者它所在的那个分句里没有单数的「你」（「你们」「你好」不算）。
    「你怎么知道大家是不是在跟你说话的」是在问一个人，「大家」只是话里提到（10-21 沙盒）；「你好呀大家」「谢谢大家，你也早点睡」照旧算。"""
    if word not in text:
        return False
    if text.lstrip(TRAIL).startswith(word):
        return True
    return any(word in c and "你" not in c.replace("你们", "").replace("你好", "") for c in re.split(f"[{re.escape(TRAIL)}]+", text))


def _calls_at_start(text: str, word: str) -> bool:
    """句首点名喊人（呼格）：「小红你在哪」「小红，快来」「小红在吗」。「小红今天又被骂了」是在说起他。"""
    head = text.lstrip(TRAIL)
    if not head.startswith(word):
        return False
    rest = head[len(word):]
    return not rest or rest[0] in TRAIL or rest[0] in VOCATIVE_NEXT


def _mentions(text: str, word: str) -> bool:
    """句子里提到了这个名字：原样包含；≥ 3 个字的名字允许同长度的一段只差一个字（OCR 错字）。"""
    if not word:
        return False
    if word in text:
        return True
    n = len(word)
    if n < FUZZY_MIN:
        return False
    return any(sum(a != b for a, b in zip(text[i : i + n], word)) <= 1 for i in range(len(text) - n + 1))


Line = tuple[float, str, str, str, bool]  # (时间, 谁, 判断, 对象, 叫了团子 / 说给大家)


def _alternating(lines: Sequence[Line], who: str) -> Verdict | None:
    """说话人（连着的算一个）最后是 他、X、他 交替：和 X 一来一回。"""
    seq: list[str] = []
    for e in lines:
        if not seq or seq[-1] != e[1]:
            seq.append(e[1])
    if not seq or seq[-1] != who:
        seq.append(who)
    if len(seq) >= 3 and seq[-1] == seq[-3] == who and seq[-2] != who:
        return Verdict("other", f"和{seq[-2]}一来一回", seq[-2])
    return None


class Addressee:
    def __init__(self, cfg: AddresseeConfig, self_names: Sequence[str], followup_window: float) -> None:
        self.cfg = cfg
        self.self_names = [n for n in self_names if n]
        self.followup_window = followup_window
        self._lines: deque[Line] = deque()  # (时间, 谁, 判断, 对象, 叫了团子 / 说给大家)；团子是 ME
        self._said_at: float | None = None
        # 接话窗口给谁：direct = 跟团子说过 / 说给大家的人，窗口照给；loose = 拿不准的人（None = 谁都算，主动开口），
        # 只在团子开口之后还没别的好友说话时给（两人不点名斗嘴时第二个人起就交给对话串）；两个都空 = 插话，谁都不给
        self._direct: frozenset[str] = frozenset()
        self._loose: frozenset[str] | None = None
        self._ref_at: float | None = None  # 对话串从这之后算：插话不挪它，旁边两个人的对话串不会被团子插一句打断

    def said(self, now: float) -> None:
        """团子说了一句：按它在回的那批聊天定接话窗口给谁。"""
        self._prune(now)
        batch = [e for e in self._lines if e[1] != ME and (self._said_at is None or e[0] > self._said_at)]
        if batch:
            self._direct = frozenset(e[1] for e in batch if e[2] in ("me", "all"))
            self._loose = frozenset(e[1] for e in batch if e[2] == "unsure") - self._direct
        elif self._said_at is None or now - self._said_at > CONTINUE:
            self._direct, self._loose = frozenset(), None
        # else：接着上一句又说（一轮两句），中间没人说话：沿用上一句的对象
        if not self._interjected():
            self._ref_at = now
        self._said_at = now
        self._lines.append((now, ME, "", "", False))

    def _prune(self, now: float) -> None:
        while self._lines and now - self._lines[0][0] > KEEP:
            self._lines.popleft()

    def _interjected(self) -> bool:
        """团子上一句是插话（在回的那批全是跟别人说的）。"""
        return not self._direct and self._loose is not None and not self._loose

    def _since_said(self, now: float) -> float | None:
        return None if self._said_at is None else now - self._said_at

    def _around(self, now: float, name: str, nearby: Sequence[str]) -> bool:
        """提到的这个好友能被叫到：在身边，或者 thread_window 内说过话（身边名单还没认出他）。
        不在的人只是被说起（「你还记得辣椒周六去干嘛了吗」，10-21 沙盒），不算跟他说。"""
        return any(similar(n, name, 0.75) for n in nearby) or any(
            e[1] == name and now - e[0] <= self.cfg.thread_window for e in self._lines)

    def judge(
        self, now: float, speaker: str, text: str, *, friends: dict[str, list[str]], nearby: Sequence[str]
    ) -> Verdict:
        """判一句并记进状态。"""
        self._prune(now)
        who = next((n for n in friends if speaker and speaker != ME and similar(speaker, n, 0.75)), "")
        if not who:
            return Verdict("unsure", "不是好友说的")
        v = self._rules(now, who, text, friends, nearby)
        self._lines.append((now, who, v.label, v.target, v.label == "all" or v.reason == CALLED))
        return v

    def _rules(self, now: float, who: str, text: str, friends: dict[str, list[str]], nearby: Sequence[str]) -> Verdict:
        cfg = self.cfg
        if any(n in text for n in self.self_names):
            return Verdict("me", CALLED)
        for name, words in friends.items():
            if name != who and any(_mentions(text, w) for w in words) and (
                    self._around(now, name, nearby) or any(_calls_at_start(text, w) for w in words)):
                return Verdict("other", f"叫了{name}", name)
        if any(_to_group(text, w) for w in cfg.group_words) or (
            len(nearby) >= 2 and any(text.startswith(w) or _ends_clause(text, w) for w in cfg.greet_words)
        ):
            return Verdict("all", "说给大家")

        upset = sounds_upset(text)  # 说难过 / 不舒服是说给在场所有人的，没点名别人就不归进对话串
        banter = self._banter(now, who, nearby)
        if banter is not None:
            return Verdict("all", f"说难过的话（刚才在跟{banter.target}说）") if upset else banter

        since = self._since_said(now)
        in_window = since is not None and since <= self.followup_window
        after = [e for e in self._lines if e[1] != ME and (self._said_at is None or e[0] > self._said_at)]
        loose = (self._loose is None or who in self._loose) and not any(e[1] != who for e in after)
        if in_window and (who in self._direct or loose):
            mine = sum(1 for e in after if e[1] == who)
            involved = any(e[2] == "other" and (e[1] == who or e[3] == who) for e in after)
            if mine < cfg.followup_lines and not involved:
                return Verdict("me", "在接你的话")

        context = [e for e in self._lines if e[1] != ME and (self._ref_at is None or e[0] > self._ref_at)]
        thread = self._thread(now, who, context, nearby)
        if thread is not None:
            return Verdict("all", f"说难过的话（刚才在跟{thread.target}说）") if upset else thread

        recent = [e for e in after if now - e[0] <= cfg.thread_window]
        if len(nearby) == 1 and similar(nearby[0], who, 0.75) and not any(e[1] != who for e in recent):
            return Verdict("me", "身边只有他")
        if upset:
            return Verdict("all", "说难过的话")
        if in_window and self._interjected():
            return Verdict("unsure", "可能在接你插的那句")
        return Verdict("unsure", "")

    def _banter(self, now: float, who: str, nearby: Sequence[str]) -> Verdict | None:
        """thread_window 内没人叫团子 / 说给大家时，跨过团子的话看对话串（还在跟 X 说 / 在回 X / 和 X 一来一回）：
        10-11 沙盒里团子插一句 / 回一句之后，两人互相回的话被当成接团子的话，团子一直跟着接。
        X 已经不在身边、或者 X 上一句是在跟团子说（三人轮流接团子的话）的不算。"""
        window = [e for e in self._lines if now - e[0] <= self.cfg.thread_window]
        recent = [e for e in window if e[1] != ME]
        if any(e[4] for e in recent):
            return None
        v = self._thread(now, who, recent, nearby)
        if v is None:
            return None
        last_x = next((e for e in reversed(recent) if e[1] == v.target), None)
        if last_x is not None and last_x[2] == "me":
            return None
        crossed = any(e[1] == ME for e in window)
        return Verdict(v.label, v.reason + CROSSED, v.target) if crossed else v

    def _thread(self, now: float, who: str, after: list[Line], nearby: Sequence[str]) -> Verdict | None:
        """还在跟别人一来一回（团子上次不是插话的开口之后、thread_window 内）：判成 other，没有就 None。
        那个人已经不在身边的不算（10-11 评审：只剩他一个人时被走开的人的旧话带偏）。"""
        v = self._thread_any(now, who, after)
        return v if v is not None and any(similar(n, v.target, 0.75) for n in nearby) else None

    def _thread_any(self, now: float, who: str, after: list[Line]) -> Verdict | None:
        recent = [e for e in after if now - e[0] <= self.cfg.thread_window]
        last_own = next((e for e in reversed(recent) if e[1] == who), None)
        if last_own is not None and last_own[2] == "other":
            return Verdict("other", f"还在跟{last_own[3]}说", last_own[3])
        reply = next((e for e in reversed(recent) if e[1] != who and e[2] == "other" and e[3] == who), None)
        if reply is not None:
            return Verdict("other", f"在回{reply[1]}", reply[1])
        return _alternating(recent, who)

    def thread_note(self, now: float) -> str:
        """最近 thread_window×3 秒里两个好友互相说话：「小明 和 阿花 在聊（1 分钟内 6 句）」，没有就空串。"""
        window = self.cfg.thread_window * 3
        self._prune(now)
        recent = [e for e in self._lines if e[1] != ME and e[2] == "other" and now - e[0] <= window]
        pairs: dict[frozenset[str], list[tuple[float, str, str]]] = {}
        for t, who, _, target, _ in recent:
            pairs.setdefault(frozenset((who, target)), []).append((t, who, target))
        best: tuple[int, list[tuple[float, str, str]]] | None = None
        for key, lines in pairs.items():
            if len(key) == 2 and {a for _, a, _ in lines} == set(key) and (best is None or len(lines) > best[0]):
                best = (len(lines), lines)
        if best is None:
            return ""
        first, second = best[1][0][1], best[1][0][2]
        span = f"{int(window // 60)} 分钟内" if window % 60 == 0 else f"{int(window)} 秒内"
        return f"{first} 和 {second} 在聊（{span} {best[0]} 句）"


def legacy_addressed(
    speaker: str,
    text: str,
    *,
    is_friend: Callable[[str], bool],
    self_names: Sequence[str],
    nearby: Sequence[str],
    since_said: float | None,
    followup_window: float,
    owner: str = "",
) -> bool:
    """好友这句是不是在跟团子说（`[addressee] enabled = false` 时用的老规则）：叫了名字、团子刚说完不久、或身边只有他一个好友。先从严。"""
    if not speaker or speaker == ME:
        return False
    if owner and speaker == owner and text.startswith("#"):  # 主人命令另有处理
        return False
    if not is_friend(speaker):
        return False
    if any(name in text for name in self_names):
        return True
    if since_said is not None and since_said <= followup_window:
        return True
    return len(nearby) == 1 and similar(nearby[0], speaker, 0.75)
