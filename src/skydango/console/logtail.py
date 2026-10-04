"""读终端起的团子的 agent.log 尾巴（管理面板接管时拿不到它的终端输出，spec 2026-10-04-console-attach §2）。

agent.log 是 DEBUG 全量（runlog.RunDir.attach_log 的格式 "%(asctime)s %(levelname)s %(name)s: %(message)s"），
这里只要 INFO 以上（和终端看到的一样）；一条记录的续行（traceback、多行消息）跟着它那一条走。
"""

from __future__ import annotations

import re
from pathlib import Path

HEAD = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} (DEBUG|INFO|WARNING|ERROR|CRITICAL) ")
SHOWN = {"INFO", "WARNING", "ERROR", "CRITICAL"}


class LogTail:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._keep = False  # 上一条记录要不要（它的续行跟着走）

    def read_new(self) -> list[str] | None:
        """从上次读到的地方往后读完整的行，返回要显示的；文件读不到返回 None。文件变短（换了文件）就从头读。"""
        try:
            with open(self.path, "rb") as f:
                size = f.seek(0, 2)
                if size < self._offset:
                    self._offset, self._keep = 0, False
                f.seek(self._offset)
                data = f.read(size - self._offset)
        except OSError:
            return None
        end = data.rfind(b"\n")
        if end < 0:  # 还没有一整行
            return []
        self._offset += end + 1
        out = []
        for raw in data[: end + 1].split(b"\n")[:-1]:
            line = raw.decode("utf-8", "replace").rstrip("\r")
            head = HEAD.match(line)
            if head:
                self._keep = head.group(1) in SHOWN
            if self._keep:
                out.append(line)
        return out
