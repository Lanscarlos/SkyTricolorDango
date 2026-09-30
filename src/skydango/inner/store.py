"""memory/inner/ 的读写（spec §1、§7）：people.json、days.jsonl、current.json。

原子写（先写 .tmp 再 replace）；读不了的文件改名成 <名字>.bad-<时间> 放一边（只读时不改名）。
读的时候不建目录：dry-run 和 memory show 不能在磁盘上留下东西。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..chat.memory import format_date
from .ledger import Card, Session
from .mind import Mind

log = logging.getLogger(__name__)

VERSION = 1


class InnerStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.people_path = self.dir / "people.json"
        self.days_path = self.dir / "days.jsonl"
        self.current_path = self.dir / "current.json"
        self.mind_path = self.dir / "mind.json"
        self.diary_path = self.dir / "diary.md"

    # ---- people.json ----
    def people_exists(self) -> bool:
        return self.people_path.is_file()

    def load_people(self, quarantine: bool = True) -> tuple[dict[str, Card], float | None, bool]:
        """(卡, 回填时间, 读没读成)。文件不存在算读成了（空的）；读不了 → quarantine 时改名放一边。"""
        if not self.people_exists():
            return {}, None, True
        try:
            data = json.loads(self.people_path.read_text(encoding="utf-8"))
            if data.get("version") != VERSION:
                raise ValueError(f"不认识的版本：{data.get('version')!r}")
            cards = {str(name): Card.from_dict(d) for name, d in (data.get("people") or {}).items()}
            backfilled = data.get("backfilled")
            return cards, (float(backfilled) if backfilled is not None else None), True
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            if quarantine:
                bad = self._quarantine(self.people_path)
                log.warning("内心账本 people.json 读不了（%s），改名成 %s，从空卡开始", exc, bad.name)
            else:
                log.warning("内心账本 people.json 读不了：%s", exc)
            return {}, None, False

    def write_people(self, cards: dict[str, Card], backfilled: float | None) -> None:
        data = {"version": VERSION, "backfilled": backfilled, "people": {n: c.to_dict() for n, c in cards.items()}}
        self._write(self.people_path, json.dumps(data, ensure_ascii=False, indent=1))

    # ---- days.jsonl ----
    def days(self) -> list[Session]:
        if not self.days_path.is_file():
            return []
        out = []
        for line in self.days_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Session.from_dict(json.loads(line)))
            except (ValueError, TypeError, AttributeError):
                log.warning("days.jsonl 里有一行读不了，跳过：%s", line[:80])
        return out

    def append_day(self, s: Session) -> None:
        self.append_days([s])

    def append_days(self, sessions: list[Session]) -> None:
        if not sessions:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.days_path.open("a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in sessions))

    # ---- current.json ----
    def write_current(self, s: Session) -> None:
        self._write(self.current_path, json.dumps(s.to_dict(), ensure_ascii=False))

    def peek_current(self) -> Session | None:
        """只读：上次没正常结束的那一次（没有 / 读不了都是 None）。"""
        try:
            return Session.from_dict(json.loads(self.current_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, AttributeError, OSError):
            return None

    def take_current(self) -> Session | None:
        """读出来就删；读不了改名放一边。"""
        if not self.current_path.is_file():
            return None
        s = self.peek_current()
        if s is None:
            bad = self._quarantine(self.current_path)
            log.warning("内心账本 current.json 读不了，改名成 %s，不补这一次", bad.name)
            return None
        self.current_path.unlink(missing_ok=True)
        return s

    def drop_current(self) -> None:
        self.current_path.unlink(missing_ok=True)

    # ---- mind.json（第 2 期）----
    def load_mind(self, quarantine: bool = True) -> Mind:
        """读不了（坏 JSON / 版本不认识）→ quarantine 时改名放一边；都从“平常”开始。"""
        if not self.mind_path.is_file():
            return Mind()
        try:
            data = json.loads(self.mind_path.read_text(encoding="utf-8"))
            if data.get("version") != VERSION:
                raise ValueError(f"不认识的版本：{data.get('version')!r}")
            return Mind.from_dict(data)
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            if quarantine:
                bad = self._quarantine(self.mind_path)
                log.warning("内心 mind.json 读不了（%s），改名成 %s，从平常开始", exc, bad.name)
            else:
                log.warning("内心 mind.json 读不了：%s", exc)
            return Mind()

    def write_mind(self, mind: Mind) -> None:
        self._write(self.mind_path, json.dumps({"version": VERSION, **mind.to_dict()}, ensure_ascii=False, indent=1))

    # ---- diary.md（第 2 期）----
    def append_diary(self, text: str, now: float) -> None:
        """一天一节（## 日期），同一天再写就在这一节末尾空一行接着写。"""
        text = text.strip()
        if not text:
            return
        heading = f"## {format_date(now)}"
        existing = self.diary_path.read_text(encoding="utf-8") if self.diary_path.is_file() else ""
        headings = [line for line in existing.splitlines() if line.startswith("## ")]
        if headings and headings[-1] == heading:
            chunk = f"\n{text}\n"
        else:
            chunk = ("\n" if existing else "") + f"{heading}\n\n{text}\n"
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.diary_path.open("a", encoding="utf-8") as fh:
            fh.write(chunk)

    def last_diaries(self, n: int) -> list[str]:
        """最后 n 天的日记正文（旧到新，段落之间空一行）。"""
        if n <= 0 or not self.diary_path.is_file():
            return []
        days: list[list[str]] = []
        for line in self.diary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                days.append([])
            elif days:
                days[-1].append(line)
        return ["\n".join(lines).strip() for lines in days if "\n".join(lines).strip()][-n:]

    # ---- 内部 ----
    def _write(self, path: Path, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)  # 写到一半崩了也不会丢旧内容

    @staticmethod
    def _quarantine(path: Path) -> Path:
        bad = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%d-%H%M%S')}")
        path.replace(bad)
        return bad
