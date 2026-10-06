"""分清好友在跟谁说话：纯规则 + 一点对话状态，不调模型（spec docs/superpowers/specs/2026-10-05-addressee-design.md §3）。

身体每读到一句好友聊天就 `judge` 一次，拿回 `Verdict`（跟你说 / 跟别人说 / 说给大家 / 拿不准）；团子说话时调 `said`。
时间都用身体的 clock；模块只记最近 2 分钟的聊天和每句的判断。
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

KEEP = 120.0  # 对话状态只留最近这么多秒
FUZZY_MIN = 3  # 叫法至少这么多个字才允许错一个字（2 个字错一个就只剩一个字对得上）


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


class Addressee:
    def __init__(self, cfg: AddresseeConfig, self_names: Sequence[str], followup_window: float) -> None:
        self.cfg = cfg
        self.self_names = [n for n in self_names if n]
        self.followup_window = followup_window
        self._lines: deque[tuple[float, str, str, str]] = deque()  # (时间, 谁, 判断, 对象)；团子是 ME
        self._said_at: float | None = None

    def said(self, now: float) -> None:
        """团子说了一句。"""
        self._prune(now)
        self._said_at = now
        self._lines.append((now, ME, "", ""))

    def _prune(self, now: float) -> None:
        while self._lines and now - self._lines[0][0] > KEEP:
            self._lines.popleft()

    def _since_said(self, now: float) -> float | None:
        return None if self._said_at is None else now - self._said_at

    def judge(
        self, now: float, speaker: str, text: str, *, friends: dict[str, list[str]], nearby: Sequence[str]
    ) -> Verdict:
        """判一句并记进状态。"""
        self._prune(now)
        who = next((n for n in friends if speaker and speaker != ME and similar(speaker, n, 0.75)), "")
        if not who:
            return Verdict("unsure", "不是好友说的")
        v = self._rules(now, who, text, friends, nearby)
        self._lines.append((now, who, v.label, v.target))
        return v

    def _rules(self, now: float, who: str, text: str, friends: dict[str, list[str]], nearby: Sequence[str]) -> Verdict:
        cfg = self.cfg
        if any(n in text for n in self.self_names):
            return Verdict("me", "叫了你")
        for name, words in friends.items():
            if name != who and any(_mentions(text, w) for w in words):
                return Verdict("other", f"叫了{name}", name)
        if any(w in text for w in cfg.group_words) or (
            len(nearby) >= 2 and any(text.startswith(w) for w in cfg.greet_words)
        ):
            return Verdict("all", "说给大家")

        since = self._since_said(now)
        after = [e for e in self._lines if e[1] != ME and (self._said_at is None or e[0] > self._said_at)]
        if since is not None and since <= self.followup_window:
            mine = sum(1 for e in after if e[1] == who)
            involved = any(e[2] == "other" and (e[1] == who or e[3] == who) for e in after)
            if mine < cfg.followup_lines and not involved:
                return Verdict("me", "在接你的话")

        thread = self._thread(now, who, after)
        upset = sounds_upset(text)  # 说难过 / 不舒服是说给在场所有人的，没点名别人就不归进对话串
        if thread is not None:
            return Verdict("all", f"说难过的话（刚才在跟{thread.target}说）") if upset else thread

        recent = [e for e in after if now - e[0] <= cfg.thread_window]
        if len(nearby) == 1 and similar(nearby[0], who, 0.75) and not any(e[1] != who for e in recent):
            return Verdict("me", "身边只有他")
        return Verdict("all", "说难过的话") if upset else Verdict("unsure", "")

    def _thread(self, now: float, who: str, after: list[tuple[float, str, str, str]]) -> Verdict | None:
        """还在跟别人一来一回（团子说完之后、thread_window 内）：判成 other，没有就 None。"""
        recent = [e for e in after if now - e[0] <= self.cfg.thread_window]
        last_own = next((e for e in reversed(recent) if e[1] == who), None)
        if last_own is not None and last_own[2] == "other":
            return Verdict("other", f"还在跟{last_own[3]}说", last_own[3])
        reply = next((e for e in reversed(recent) if e[1] != who and e[2] == "other" and e[3] == who), None)
        if reply is not None:
            return Verdict("other", f"在回{reply[1]}", reply[1])
        seq: list[str] = []
        for e in recent:
            if not seq or seq[-1] != e[1]:
                seq.append(e[1])
        if not seq or seq[-1] != who:
            seq.append(who)
        if len(seq) >= 3 and seq[-1] == seq[-3] == who and seq[-2] != who:
            return Verdict("other", f"和{seq[-2]}一来一回", seq[-2])
        return None

    def thread_note(self, now: float) -> str:
        """最近 thread_window×3 秒里两个好友互相说话：「小明 和 阿花 在聊（1 分钟内 6 句）」，没有就空串。"""
        window = self.cfg.thread_window * 3
        self._prune(now)
        recent = [e for e in self._lines if e[1] != ME and e[2] == "other" and now - e[0] <= window]
        pairs: dict[frozenset[str], list[tuple[float, str, str]]] = {}
        for t, who, _, target in recent:
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
