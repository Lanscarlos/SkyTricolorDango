"""memory/inner/ 的读写（spec §1、§7）：people.json、days.jsonl、current.json；第 2 期 mind.json、diary.md；第 3 期 persona.json。

原子写（先写 .tmp 再 replace）；读不了的文件改名成 <名字>.bad-<时间> 放一边（只读时不改名）。
读的时候不建目录：dry-run 和 memory show 不能在磁盘上留下东西。
"""

from __future__ import annotations

import json
import re
import logging
import time
from pathlib import Path

from ..chat.memory import format_date
from .ledger import Card, Session
from .mind import Mind
from .persona import Persona

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
        self.persona_path = self.dir / "persona.json"

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

    # ---- persona.json（第 3 期）----
    def load_persona(self, quarantine: bool = True) -> Persona:
        """读不了（坏 JSON / 版本不认识 / 格式不对）→ quarantine 时改名放一边；都从空档案开始。"""
        if not self.persona_path.is_file():
            return Persona()
        try:
            data = json.loads(self.persona_path.read_text(encoding="utf-8"))
            if data.get("version") != VERSION:
                raise ValueError(f"不认识的版本：{data.get('version')!r}")
            return Persona.from_dict(data)
        except (ValueError, TypeError, AttributeError, KeyError, OSError) as exc:
            if quarantine:
                bad = self._quarantine(self.persona_path)
                log.warning("性格档案 persona.json 读不了（%s），改名成 %s，从空的开始", exc, bad.name)
            else:
                log.warning("性格档案 persona.json 读不了：%s", exc)
            return Persona()

    def write_persona(self, persona: Persona) -> None:
        self._write(self.persona_path, json.dumps({"version": VERSION, **persona.to_dict()}, ensure_ascii=False, indent=1))

    # ---- diary.md（第 2 期）----
    def append_diary(self, text: str, now: float) -> None:
        """一天一节（## 日期），同一天再写就在这一节末尾空一行接着写。"""
        text = re.sub(r"\n\s*\n+", "\n", text.strip())  # 一次写一段：段内不留空行（空行用来分开每次下线）
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
        """最后 n 篇日记（每次下线一篇，旧到新），带日期、换行压成空格：“9月30日：今天……”。"""
        if n <= 0 or not self.diary_path.is_file():
            return []
        entries: list[str] = []
        label, para = "", []

        def flush():
            if para:
                entries.append(f"{label}：" + " ".join(para) if label else " ".join(para))
                para.clear()

        for line in self.diary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                m = re.search(r"(\d+)年(\d+)月(\d+)日", line)
                label = f"{int(m.group(2))}月{int(m.group(3))}日" if m else ""
            elif line.strip():
                para.append(line.strip())
            else:
                flush()
        flush()
        return entries[-n:]

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
