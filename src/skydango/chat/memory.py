"""记忆：全部放在一个目录里（默认 memory/，不进 git），都是能直接打开编辑的文本文件。

- profile.md   自己的人设（用户写，程序不改）；有它就不用配置里的 reply.persona
- friends.md   好友：昵称、本名、怎么称呼、关系（用户写，程序不改）
- notes.md     长期记忆：每聊若干轮，让模型把新的聊天记录整理进去（用户也可以改、删）
- history.jsonl  逐轮的原始聊天记录，重启时读回最近几十轮
- notes_state.json  笔记整理到了哪一轮
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .entries import archive_line, check_rewrite, parse_inbox, parse_notes, render_notes, settle

if TYPE_CHECKING:
    from ..config import ReplyConfig

log = logging.getLogger(__name__)

GAP_NOTE_AFTER = 30 * 60  # 隔了这么久再聊，告诉模型过了多久，免得接着几小时前的话题说
SKIP = "<skip>"


@dataclass(frozen=True)
class Turn:
    t: float  # time.time()
    user: str  # 发给模型的那条 user 消息（format_incoming 的结果）
    reply: str  # 自己的回复；不回复时是 SKIP


_WEEKDAYS = "一二三四五六日"


def format_date(t: float) -> str:
    """给记忆用的日期：2026年9月27日（周日）。模型要靠它把“今天”“下周六”换成具体日期。"""
    d = time.localtime(t)
    return f"{d.tm_year}年{d.tm_mon}月{d.tm_mday}日（周{_WEEKDAYS[d.tm_wday]}）"


def format_gap(seconds: float) -> str:
    if seconds >= 86400:
        return f"{int(seconds // 86400)} 天"
    if seconds >= 3600:
        return f"{int(seconds // 3600)} 小时"
    return f"{int(seconds // 60)} 分钟"


class ChatMemory:
    """逐轮的聊天记录（jsonl，一行一轮）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def all(self) -> list[Turn]:
        if not self.path.is_file():
            return []
        turns: list[Turn] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                turns.append(Turn(float(d["t"]), str(d["user"]), str(d["reply"])))
            except (ValueError, KeyError, TypeError):
                log.warning("聊天记录里有一行读不了，跳过: %s", line[:80])
        return turns

    def load(self, max_turns: int) -> list[Turn]:
        return self.all()[-max_turns:] if max_turns > 0 else []

    def append(self, user: str, reply: str, t: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(t, 1), "user": user, "reply": reply}, ensure_ascii=False) + "\n")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


class MemoryStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.history = ChatMemory(self.dir / "history.jsonl")
        self._state = self.dir / "notes_state.json"
        self._inbox = self.dir / "inbox.md"
        self._inbox_lock = threading.Lock()  # 聊天线程随手记、后台整理线程清理，都要改 inbox

    def inbox(self) -> str:
        return _read(self._inbox)

    def add_memos(self, memos: list[str]) -> None:
        """随手记：回复时模型顺手记下的事，立刻写进 inbox.md，下一句回复就能用上。"""
        memos = [m.strip() for m in memos if m.strip()]
        if not memos:
            return
        with self._inbox_lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self._inbox.open("a", encoding="utf-8") as fh:
                fh.write("".join(f"- {m}\n" for m in memos))
        log.info("记住: %s", "；".join(memos))

    def consume_inbox(self, snapshot: str) -> None:
        """整理完成后，删掉已经整理进笔记的那部分（snapshot），整理期间新记的保留。"""
        with self._inbox_lock:
            current = _read(self._inbox)
            if not snapshot or not current.startswith(snapshot):  # 用户手动改过：不动它，下次整理再说
                return
            rest = current[len(snapshot) :].strip()
            self._inbox.write_text(rest + "\n" if rest else "", encoding="utf-8")

    def profile(self) -> str:
        return _read(self.dir / "profile.md")

    def friends(self) -> str:
        return _read(self.dir / "friends.md")

    def friend_names(self) -> list[str]:
        """friends.md 里每个 “## 标题” 就是一个好友的游戏昵称。"""
        return [line[3:].strip() for line in self.friends().splitlines() if line.startswith("## ") and line[3:].strip()]

    def add_friend_note(self, name: str, note: str) -> None:
        """`#friend` 命令用：在 friends.md 里给 `## <name>` 这一节追加一行备注；没有这一节就新建。"""
        path = self.dir / "friends.md"
        text = _read(path)
        lines = text.splitlines()
        heading = f"## {name}"
        start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
        if start is None:
            sep = "\n\n" if text else ""
            new_text = f"{text}{sep}{heading}\n- {note}\n"
        else:
            end = start + 1
            while end < len(lines) and not lines[end].startswith("## "):
                end += 1
            insert_at = end
            while insert_at > start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines.insert(insert_at, f"- {note}")
            new_text = "\n".join(lines) + "\n"
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("friends.md.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(path)  # 先写临时文件再改名，写到一半崩了也不会丢旧内容

    def notes(self) -> str:
        return _read(self.dir / "notes.md")

    def write_notes(self, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "notes.md.tmp"
        tmp.write_text(text.strip() + "\n", encoding="utf-8")
        tmp.replace(self.dir / "notes.md")  # 先写临时文件再改名，写到一半崩了也不会丢旧笔记

    def _state_dict(self) -> dict:
        try:
            d = json.loads(self._state.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return d if isinstance(d, dict) else {}

    def notes_until(self) -> float:
        try:
            return float(self._state_dict()["until"])
        except (KeyError, ValueError, TypeError):
            return float("-inf")

    def set_notes_until(self, t: float) -> None:
        self._merge_state(until=t)

    def _merge_state(self, **kv) -> None:
        """合并写 notes_state.json：不覆盖别的键。"""
        d = self._state_dict()
        d.update(kv)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._state.write_text(json.dumps(d), encoding="utf-8")

    def entries_since(self) -> float | None:
        """记忆条目化第一次整理的时间（有了就不再备份旧 notes）；没整理过是 None。"""
        try:
            return float(self._state_dict()["entries_since"])
        except (KeyError, ValueError, TypeError):
            return None

    def set_entries_since(self, t: float) -> None:
        self._merge_state(entries_since=t)

    @property
    def archive_path(self) -> Path:
        return self.dir / "notes_archive.md"

    def archive(self) -> str:
        return _read(self.archive_path)

    def append_archive(self, lines: list[str]) -> None:
        """被取代 / 过期的条目挪到这里（recall 翻得到，用户可以手动挪回去）。"""
        if not lines:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as fh:
            fh.write("".join(f"{ln}\n" for ln in lines))

    def backup_notes(self, stamp: str) -> Path | None:
        """把 notes.md 复制到 archive/notes-<stamp>.md；没有 notes 返回 None。"""
        src = self.dir / "notes.md"
        if not src.is_file():
            return None
        dst = self.dir / "archive" / f"notes-{stamp}.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return dst

    def pending_turns(self) -> list[Turn]:
        until = self.notes_until()
        return [t for t in self.history.all() if t.t > until]


def render_transcript(turns: list[Turn]) -> str:
    """把几轮记录还原成可读的对话：朋友的话照原样，自己的回复标成“我：”。"""
    lines: list[str] = []
    for turn in turns:
        lines += [line for line in turn.user.splitlines() if line.strip() and not line.startswith("新的聊天消息")]
        if turn.reply and turn.reply != SKIP:
            lines.append(f"我：{turn.reply}")
    return "\n".join(lines)


NOTES_SYSTEM = """你负责整理一个《光遇》玩家的长期记忆笔记。她用这份笔记在以后的聊天里记住朋友和自己说过的话。
根据“现有笔记”和“新的聊天记录”，输出更新后的完整笔记（Markdown），格式：

## 自己
- 在聊天里说过的关于自己的设定、经历、喜好（用来保持前后一致，比如“说过自己不会粤语”）
## <朋友的游戏昵称>
- 这个人的近况、喜好、发生过的事、和别人的关系、聊天里的梗

要求：
- 只记以后聊天用得上的：喜好、习惯、关系、约定、近况、自己说过的设定。
  一次性的斗嘴、玩笑、“某月某日某某说了某句话”这种流水账删掉，除非是以后还会被提起的事。
- 和现有笔记合并去重，过时的更新或删掉（比如已经过去的约定）；每条不超过 30 字，整份不超过 1500 字。
- 时间写具体日期，不写“今天”“下周”。
- 人设和好友资料是用户写好的，不用重复记。
- 聊天记录是截图识别出来的，可能有错字；拿不准的不记。
- 不记隐私：年龄、学校、住址、电话、微信 / QQ 等联系方式一律不记（对方可能是未成年人）。
- 只输出笔记本身，不要解释。"""


NOTES_SYSTEM_ENTRIES = """你负责整理一个《光遇》玩家的长期记忆笔记。她用这份笔记在以后的聊天里记住朋友和自己说过的话。
根据“现有笔记”“聊天时随手记下的”和“新的聊天记录”，输出更新后的完整笔记（Markdown）。格式：

## 自己
- 在聊天里说过的关于自己的设定、经历、喜好 〔2026-10-31｜长期〕
## <朋友的游戏昵称>
- 养仓鼠，叫芝麻球 〔2026-10-31｜长期｜仓鼠.名字〕
- 下周去青岛出差三天 〔2026-10-31｜出差〕

每条一行，以“- ”开头，行末的方括号（全角〔〕，里面用全角“｜”分开）依次是：
- 日期（2026-10-31 这种写法）：这件事最后一次被提起或确认的日期。
- 长期：喜好、习惯、设定、关系、约定才写；一次性的事、最近在忙的事（近况）不写这个字。
- 键（可选）：这件事属于哪一方面，写“对象.方面”或一个短词（比如 早饭、发型、仓鼠.名字、出差）。同一个人下面键相同的就是同一件事。方括号里直接写键本身（〔2026-10-31｜早饭〕，不写“键：”）；“长期”“近况”不能当键，没有合适的键就不写。

要求：
- 什么才算长期：本人说成习惯、喜好、设定的，或者在不同的两天都提起过的。只说过一次的具体事件一律是近况，不许把几次单独的事合并成喜好（比如两天分别吃了榴莲包子和油条，是两条早饭的近况，不是“爱吃榴莲包子、油条”）。
- 同一件事再次被提起：只把日期改成最新那天，不要重复写一条。
- 会变的事（名字、发型、计划、在忙什么、早饭吃了什么……）给一个键；新的说法和旧的同键时，只写新的，旧的不要留。
- 只记以后聊天用得上的。一次性的斗嘴、玩笑、“某月某日某某说了某句话”这种流水账删掉，除非是以后还会被提起的事。
- 每条不超过 30 字；长期的条目加起来不超过 1500 字，近况不限（太旧的会由程序自动收走）。
- 日期写具体日期，不写“今天”“下周”；日期是这件事发生 / 被提起的那天，不是今天整理的日子：正文里写着日期（“2026-10-26 头疼”“10月4日认错人”）的，方括号里就写那个日期。
- 正文里原有的日期照抄，不许改成别的日期。
- 现有笔记里没有方括号的老条目：正文里写着日期的，加上方括号写那个日期；看不出日期的照抄原样、不加方括号（程序当它长期、日期不详）。
- 随手记里“某年某月某日 的要点：”开头的，是她那天下线时给自己记的，归到“自己”下面，日期就是那天。
- 现有笔记里的条目不要随便删：只删确实过时、作废的，删了的程序会挪进旧笔记存档。
- 人设和好友资料是用户写好的，不用重复记。
- 聊天记录是截图识别出来的，可能有错字；拿不准的不记。
- 不记隐私：年龄、学校、住址、电话、微信 / QQ 等联系方式一律不记（对方可能是未成年人）。
- 只输出笔记本身，不要解释。"""


MEMO_SYSTEM_ENTRIES = """你在帮一个《光遇》玩家记笔记。下面有好友资料、已经记下的内容，以及刚发生的一轮聊天（“我：”后面是她自己说的）。
只挑出这轮聊天里以后用得上、而且还没记过的新信息，优先记长期成立的：喜好、习惯、关系、约定、近况（在忙什么、身体 / 心情），
以及“我”新说出口的关于自己的设定、喜好和评价（比如“我说雨林太湿，不喜欢待”；免得以后说法不一致）。
- 先记好友说的他自己的事（吃了什么、在忙什么、身体 / 心情、计划）；你自己的回话、接梗、吐槽、损人的话不记，除非说出了关于你自己的新设定。
- 一次性的斗嘴、玩笑、夸奖吐槽、“某某说了某句话”这种流水账不记；好友资料和已经记下的内容里有的不记。
- 只说过一次的具体事、这一轮里提的要求不写“长期”；本人明说是习惯、喜好、一直这样的才写。
- 每条一行，以“- ”开头，写成“- 谁：内容 〔日期｜长期｜键〕”：谁用游戏昵称或本名（自己说的用“我”）；内容不超过 30 字；同一件事只记一条。
- 方括号用全角〔〕，里面用全角“｜”分开：日期必填（2026-10-31 这种写法，就是这轮聊天发生的那天）；
  是习惯、喜好、设定、关系、约定才写“长期”，一次性的事和近况不写；键可选，会变的事（名字、发型、计划、在忙什么）写一个短词，直接写词本身（〔2026-10-31｜早饭〕，不写“键：”；“近况”不是键）。
- 看清是谁说的：“名字：”后面是那个人说的，“我：”后面是自己说的。
- “今天”“明天”“下周六”这类相对时间换成具体日期（比如“9月27日加班到九点”“约好10月3日跑暴风眼”）。
- 寒暄、玩笑、没信息量的话不记；已经记过的不重复记；拿不准的不记（聊天是截图识别的，可能有错字）。
- 不记年龄、学校、住址、电话、微信 / QQ 等（对方可能是未成年人）。
- 没有值得记的，只输出：无"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    return re.sub(r"\n?```\s*$", "", text).strip()


MEMO_SYSTEM = """你在帮一个《光遇》玩家记笔记。下面有好友资料、已经记下的内容，以及刚发生的一轮聊天（“我：”后面是她自己说的）。
只挑出这轮聊天里以后用得上、而且还没记过的新信息，优先记长期成立的：喜好、习惯、关系、约定、近况（在忙什么、身体 / 心情），
以及“我”新说出口的关于自己的设定、喜好和评价（比如“我说雨林太湿，不喜欢待”；免得以后说法不一致）。
- 一次性的斗嘴、玩笑、夸奖吐槽、“某某说了某句话”这种流水账不记；好友资料和已经记下的内容里有的不记。
- 每条一行，以“- ”开头，不超过 30 字，写清楚是谁（用游戏昵称或本名）。同一件事只记一条。
- 看清是谁说的：“名字：”后面是那个人说的，“我：”后面是自己说的。
- “今天”“明天”“下周六”这类相对时间换成具体日期（比如“9月27日加班到九点”“约好10月3日跑暴风眼”）。
- 寒暄、玩笑、没信息量的话不记；已经记过的不重复记；拿不准的不记（聊天是截图识别的，可能有错字）。
- 不记年龄、学校、住址、电话、微信 / QQ 等（对方可能是未成年人）。
- 没有值得记的，只输出：无"""


def _ask(llm, system: str, content: str, max_tokens: int) -> tuple[str, bool]:
    """（正文，是不是到 max_tokens 被截断了）。GatedCall 有 message()、带得出 truncated；只有 complete() 的（测试、旧客户端）当没截断。"""
    message = getattr(llm, "message", None)
    if callable(message):
        out = message(system, content, max_tokens=max_tokens)
        return out.get("result") or "", bool(out.get("truncated"))
    return llm.complete(system, [{"role": "user", "content": content}], max_tokens=max_tokens) or "", False


def parse_memos(raw: str) -> list[str]:
    memos = []
    for line in (raw or "").splitlines():
        text = line.strip().lstrip("-*•").strip()
        if text and text != "无":
            memos.append(text)
    return memos


NOTES_RETRY = 600.0  # 整理 notes 失败后至少隔这么久（秒）才自动再试，免得额度 / 认证出错时每轮都撞墙


class NotesKeeper:
    """管长期记忆，所有活都在一个后台线程里按顺序做，不耽误回复：

    - 随手记：每轮回复之后单独调一次模型，只挑出值得记的新信息写进 inbox.md，下一句回复就能用上
    - 整理：每攒够 every 轮，让模型把新的聊天记录和 inbox 归并进 notes.md
    """

    def __init__(
        self, llm, store: MemoryStore, persona: str, every: int = 15, background: bool = True, wall: Callable[[], float] = time.time,
        entries: ReplyConfig | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.wall = wall  # 整理时告诉模型"今天是"哪天：沙盒传模拟时钟
        # 记忆条目化（[reply] memory_entries）：None 或关着 = 逐字照旧
        self.entries = entries if entries is not None and entries.memory_entries else None
        self.persona = persona
        self.every = every
        self.background = background
        self.pending = len(store.pending_turns())
        self._lock = threading.Lock()
        self._update_queued = False
        self._retry_after = float("-inf")  # wall() 时间：失败后这之前 maybe_update 不再自动整理
        self._tasks: queue.Queue | None = None

    def _submit(self, task) -> None:
        if not self.background:
            task()
            return
        if self._tasks is None:
            self._tasks = queue.Queue()
            threading.Thread(target=self._worker, name="memory", daemon=True).start()
        self._tasks.put(task)

    def _worker(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except Exception:
                log.exception("记忆后台任务出错")

    def turn_added(self, turn: Turn) -> None:
        self.pending += 1
        self._submit(lambda: self.jot(turn))
        self.maybe_update()

    def jot(self, turn: Turn) -> list[str]:
        """随手记：从这一轮里挑出值得记的新信息，写进 inbox。"""
        known = "\n\n".join(p for p in (self.store.notes(), self.store.inbox()) if p)
        when = format_date(turn.t) + (f"，即 {datetime.fromtimestamp(turn.t).date().isoformat()}" if self.entries else "")
        content = (
            f"这轮聊天发生在 {when}\n\n"
            f"## 好友资料\n{self.store.friends() or '（无）'}\n\n"
            f"## 已经记下的\n{known or '（无）'}\n\n"
            f"## 刚发生的一轮聊天\n{render_transcript([turn])}"
        )
        try:
            if self.entries:  # 方括号多占几十个字，留宽一点免得半截
                raw, cut = _ask(self.llm, MEMO_SYSTEM_ENTRIES, content, 300)
            else:
                raw, cut = _ask(self.llm, MEMO_SYSTEM, content, 200)
        except Exception:
            log.exception("随手记失败，跳过这一轮")
            return []
        memos = parse_memos(raw)
        if cut and memos:  # 最后一行多半只写了半截
            log.warning("随手记被截断，丢掉最后半行：%s", memos[-1])
            memos = memos[:-1]
        self.store.add_memos(memos)
        return memos

    def maybe_update(self) -> None:
        if self.every <= 0 or self.pending < self.every:
            return
        if self.wall() < self._retry_after:
            return
        with self._lock:
            if self._update_queued:
                return
            self._update_queued = True
        self._submit(self._update)

    def update_now(self) -> bool:
        """整理所有还没整理的记录和随手记；返回有没有写入新笔记。"""
        turns = self.store.pending_turns()
        inbox = self.store.inbox()
        if not turns and not inbox:
            return False
        persona = self.store.profile() or self.persona
        old_notes = self.store.notes()
        today = format_date(self.wall()) + (f"，即 {datetime.fromtimestamp(self.wall()).date().isoformat()}" if self.entries else "")
        content = (
            f"今天是 {today}\n\n"
            f"## 人设（参考，不用记）\n{persona or '（无）'}\n\n"
            f"## 好友资料（参考，不用记）\n{self.store.friends() or '（无）'}\n\n"
            f"## 现有笔记\n{old_notes or '（空）'}\n\n"
            f"## 聊天时随手记下的（归到对应的人下面）\n{inbox or '（无）'}\n\n"
            f"## 新的聊天记录\n{render_transcript(turns) or '（无）'}"
        )
        # 整份重写：输出不够长会截在半截（10-26 截掉了后面三个好友那几节），条目版还多方括号，留足
        system, max_tokens = (NOTES_SYSTEM_ENTRIES, 8000) if self.entries else (NOTES_SYSTEM, 4000)
        try:
            raw, cut = _ask(self.llm, system, content, max_tokens)
        except Exception:
            self._retry_after = self.wall() + NOTES_RETRY
            log.exception("整理长期记忆失败，%d 分钟后再试", int(NOTES_RETRY // 60))
            return False
        if cut:
            self._retry_after = self.wall() + NOTES_RETRY
            log.warning("整理长期记忆的输出到 max_tokens=%d 被截断了，拒绝写盘、保留旧笔记（笔记太长了，删一些或调大上限）", max_tokens)
            return False
        notes = _strip_fences(raw)
        if not notes:
            self._retry_after = self.wall() + NOTES_RETRY
            log.warning("整理长期记忆时模型返回了空内容，保留旧笔记")
            return False
        if self.entries:
            notes = self._settle(old_notes, notes, inbox)
            if notes is None:
                self._retry_after = self.wall() + NOTES_RETRY
                return False
        self._retry_after = float("-inf")
        self.store.write_notes(notes)
        if turns:
            self.store.set_notes_until(turns[-1].t)
        self.store.consume_inbox(inbox)
        log.info("长期记忆已更新（整理了 %d 轮、%d 条随手记）", len(turns), len(inbox.splitlines()))
        return True

    def _settle(self, old_notes: str, raw: str, inbox: str = "") -> str | None:
        """条目化的整理结果：剥代码块、把关、取代 / 淡出（被挪走的进 archive）。返回要写盘的文本，拒绝返回 None。"""
        cleaned = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("```"))
        why = check_rewrite(old_notes, cleaned)
        new = parse_notes(cleaned)  # 开头那句“好的，以下是…”不是条目，解析时自然丢掉
        if why is None and not new and old_notes.strip():
            why = "没解析出任何条目"
        if why:
            log.warning("整理结果不像样，拒绝写盘（%s），保留旧笔记", why)
            return None
        now = self.wall()
        today = datetime.fromtimestamp(now).date()
        if self.store.entries_since() is None:  # 第一次按条目整理：先留一份旧 notes
            self.store.backup_notes(datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S"))
            self.store.set_entries_since(now)
        keep, gone = settle(parse_notes(old_notes), new, today, self.entries.recent_days, inbox=parse_inbox(inbox))
        self.store.append_archive([archive_line(e, today, why) for e, why in gone])
        if gone:
            log.info("记忆整理收走 %d 条：%s", len(gone), "；".join(f"{e.text}（{r}）" for e, r in gone[:5]))
        return render_notes(keep)

    def _update(self) -> None:
        try:
            if self.update_now():
                self.pending = len(self.store.pending_turns())
        finally:
            with self._lock:
                self._update_queued = False
