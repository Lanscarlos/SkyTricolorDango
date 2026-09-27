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
from dataclasses import dataclass
from pathlib import Path

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

    def notes(self) -> str:
        return _read(self.dir / "notes.md")

    def write_notes(self, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "notes.md.tmp"
        tmp.write_text(text.strip() + "\n", encoding="utf-8")
        tmp.replace(self.dir / "notes.md")  # 先写临时文件再改名，写到一半崩了也不会丢旧笔记

    def notes_until(self) -> float:
        try:
            return float(json.loads(self._state.read_text(encoding="utf-8"))["until"])
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return float("-inf")

    def set_notes_until(self, t: float) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._state.write_text(json.dumps({"until": t}), encoding="utf-8")

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


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    return re.sub(r"\n?```\s*$", "", text).strip()


MEMO_SYSTEM = """你在帮一个《光遇》玩家记笔记。下面有好友资料、已经记下的内容，以及刚发生的一轮聊天（“我：”后面是她自己说的）。
只挑出这轮聊天里以后用得上、而且还没记过的新信息，优先记长期成立的：喜好、习惯、关系、约定、近况（在忙什么、身体 / 心情），
以及“我”新说出口的关于自己的设定（免得以后说法不一致）。
- 一次性的斗嘴、玩笑、夸奖吐槽、“某某说了某句话”这种流水账不记；好友资料和已经记下的内容里有的不记。
- 每条一行，以“- ”开头，不超过 30 字，写清楚是谁（用游戏昵称或本名）。同一件事只记一条。
- 看清是谁说的：“名字：”后面是那个人说的，“我：”后面是自己说的。
- “今天”“明天”“下周六”这类相对时间换成具体日期（比如“9月27日加班到九点”“约好10月3日跑暴风眼”）。
- 寒暄、玩笑、没信息量的话不记；已经记过的不重复记；拿不准的不记（聊天是截图识别的，可能有错字）。
- 不记年龄、学校、住址、电话、微信 / QQ 等（对方可能是未成年人）。
- 没有值得记的，只输出：无"""


def parse_memos(raw: str) -> list[str]:
    memos = []
    for line in (raw or "").splitlines():
        text = line.strip().lstrip("-*•").strip()
        if text and text != "无":
            memos.append(text)
    return memos


class NotesKeeper:
    """管长期记忆，所有活都在一个后台线程里按顺序做，不耽误回复：

    - 随手记：每轮回复之后单独调一次模型，只挑出值得记的新信息写进 inbox.md，下一句回复就能用上
    - 整理：每攒够 every 轮，让模型把新的聊天记录和 inbox 归并进 notes.md
    """

    def __init__(self, llm, store: MemoryStore, persona: str, every: int = 15, background: bool = True) -> None:
        self.llm = llm
        self.store = store
        self.persona = persona
        self.every = every
        self.background = background
        self.pending = len(store.pending_turns())
        self._lock = threading.Lock()
        self._update_queued = False
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
        content = (
            f"这轮聊天发生在 {format_date(turn.t)}\n\n"
            f"## 好友资料\n{self.store.friends() or '（无）'}\n\n"
            f"## 已经记下的\n{known or '（无）'}\n\n"
            f"## 刚发生的一轮聊天\n{render_transcript([turn])}"
        )
        try:
            raw = self.llm.complete(MEMO_SYSTEM, [{"role": "user", "content": content}], max_tokens=200)
        except Exception:
            log.exception("随手记失败，跳过这一轮")
            return []
        memos = parse_memos(raw)
        self.store.add_memos(memos)
        return memos

    def maybe_update(self) -> None:
        if self.every <= 0 or self.pending < self.every:
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
        content = (
            f"今天是 {format_date(time.time())}\n\n"
            f"## 人设（参考，不用记）\n{persona or '（无）'}\n\n"
            f"## 好友资料（参考，不用记）\n{self.store.friends() or '（无）'}\n\n"
            f"## 现有笔记\n{self.store.notes() or '（空）'}\n\n"
            f"## 聊天时随手记下的（归到对应的人下面）\n{inbox or '（无）'}\n\n"
            f"## 新的聊天记录\n{render_transcript(turns) or '（无）'}"
        )
        try:
            raw = self.llm.complete(NOTES_SYSTEM, [{"role": "user", "content": content}], max_tokens=1500)
        except Exception:
            log.exception("整理长期记忆失败，下次再试")
            return False
        notes = _strip_fences(raw or "")
        if not notes:
            log.warning("整理长期记忆时模型返回了空内容，保留旧笔记")
            return False
        self.store.write_notes(notes)
        if turns:
            self.store.set_notes_until(turns[-1].t)
        self.store.consume_inbox(inbox)
        log.info("长期记忆已更新（整理了 %d 轮、%d 条随手记）", len(turns), len(inbox.splitlines()))
        return True

    def _update(self) -> None:
        try:
            if self.update_now():
                self.pending = len(self.store.pending_turns())
        finally:
            with self._lock:
                self._update_queued = False
