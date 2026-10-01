"""幕后（spec 2026-10-01-backstage）：团子知道自己是 AI、卡洛做了她。

这里拼「幕后」一节，取"卡洛上次以来改了你什么"（git 提交），读写标记 inner/backstage.json。
只在启动时算一次；git 出任何错都只是这一小节为空，不影响启动。
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

Runner = Callable[[list[str]], str]

GIT_TIMEOUT = 2.0
SCAN_MAX = 500  # 按天退回时最多往回翻几个提交
TITLE = re.compile(r"^(feat|fix|perf)(\(([^)]*)\))?!?:\s*")
HIDDEN_SCOPES = {"console", "viewer"}  # 她感觉不到的改动
DAY = 86400


def git_runner(repo: Path) -> Runner:
    def run(args: list[str]) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
            timeout=GIT_TIMEOUT, check=True,
        ).stdout

    return run


def repo_root() -> Path | None:
    """从 skydango 包目录往上找含 .git 的目录（worktree 里 .git 是文件，也算）。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def read_seen(path: Path) -> str | None:
    try:
        seen = json.loads(Path(path).read_text(encoding="utf-8")).get("seen")
    except (OSError, ValueError, AttributeError):
        return None
    return seen if isinstance(seen, str) and seen else None


def write_seen(path: Path, sha: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"seen": sha}), encoding="utf-8")
    tmp.replace(path)


def _when(t: float, now: float) -> str:
    days = (time.localtime(now).tm_yday - time.localtime(t).tm_yday) if time.localtime(now).tm_year == time.localtime(t).tm_year \
        else round((now - t) / DAY)
    return "今天" if days <= 0 else "昨天" if days == 1 else f"{days} 天前"


def _reachable(run: Runner, seen: str | None) -> bool:
    if not seen:
        return False
    try:
        run(["merge-base", "--is-ancestor", seen, "HEAD"])
    except subprocess.CalledProcessError:
        return False
    return True


def changelog(run: Runner, seen: str | None, now: float, max_n: int = 10, days: int = 7) -> tuple[list[str], str | None]:
    """返回 (给提示词的行, HEAD 哈希)：seen 之后（seen 不在历史里就按最近 days 天）的 feat / fix / perf，新到旧，最多 max_n 条。"""
    try:
        head = run(["rev-parse", "HEAD"]).strip()
        fmt = "--format=%H%x1f%ct%x1f%s"
        if _reachable(run, seen):
            out, since = run(["log", "--no-merges", fmt, f"{seen}..HEAD"]), None
        else:
            out, since = run(["log", "--no-merges", fmt, f"-n{SCAN_MAX}", "HEAD"]), now - days * DAY
        picked: list[str] = []
        for row in out.splitlines():
            parts = row.split("\x1f")
            if len(parts) != 3:
                continue
            _, ct, subject = parts
            t = float(ct)
            if since is not None and t < since:
                continue
            m = TITLE.match(subject)
            if m is None or (m.group(3) or "").strip() in HIDDEN_SCOPES:
                continue
            picked.append(f"- {_when(t, now)} {subject}")
    except Exception as exc:
        log.warning("取更新记录失败，这次不写：%s", exc)
        return [], None
    if len(picked) > max_n:
        picked = picked[:max_n] + [f"- 还有 {len(picked) - max_n} 条小改动"]
    return picked, head or None
