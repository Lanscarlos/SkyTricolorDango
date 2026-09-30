"""管理面板「沙盒」页的数据（brain-sandbox spec §4）：重置沙盒记忆、沙盒里的好友名单、起始时间下限。

纯文件操作，不碰网络；ConsoleServer 调这些再包成接口。
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from ..chat.memory import MemoryStore
from ..sandbox.clock import floor_time, load_saved

log = logging.getLogger(__name__)

SKIP = ("archive",)  # memory/ 下不复制的目录（挪走的旧聊天记录）


def reset(sandbox_dir: Path, memory_dir: Path) -> None:
    """用 memory/ 覆盖沙盒记忆：删掉 sandbox/memory/ 和 clock.json，再把 memory/（除了 archive/）复制过去。
    memory/ 不存在就建一个空的沙盒记忆。真的 memory/ 只读。"""
    sandbox_dir, memory_dir = Path(sandbox_dir), Path(memory_dir)
    target = sandbox_dir / "memory"
    if target.exists():
        shutil.rmtree(target)
    (sandbox_dir / "clock.json").unlink(missing_ok=True)
    if memory_dir.is_dir():
        shutil.copytree(memory_dir, target, ignore=lambda d, names: [n for n in names if Path(d) == memory_dir and n in SKIP])
    else:
        target.mkdir(parents=True)
    log.info("沙盒记忆重置了：%s → %s", memory_dir, target)


def friends(sandbox_dir: Path) -> list[str]:
    """沙盒记忆里 friends.md 的 “## 标题”（说话人下拉框、身边名单用）。"""
    return MemoryStore(Path(sandbox_dir) / "memory").friend_names()


def _fmt(t: float) -> str:
    lt = time.localtime(t)
    return f"{lt.tm_mon}月{lt.tm_mday}日 {lt.tm_hour:02d}:{lt.tm_min:02d}"


def start_info(sandbox_dir: Path, now: float) -> dict:
    """起始时间下限：clock.json（上次停下时的沙盒时间）和 days.jsonl 最后一次上线的结束时间取晚的。

    floor_text 只在下限比现在晚时给（"最早只能从 9月30日 22:00 开始"）；早于现在时"接着上次"就是现在，不用提。"""
    floor = floor_time(Path(sandbox_dir))
    return {
        "floor": floor,
        "saved": load_saved(Path(sandbox_dir) / "clock.json"),
        "floor_text": _fmt(floor) if floor > now else "",
    }
