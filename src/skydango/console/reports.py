"""沙盒回放报告的列出和读取（只读；console-redesign spec Task 3）。

报告由 `Replayer` 写进 `<沙盒目录>/reports/<剧本>-<时间>[-N].md`。
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

MAX_REPORT = 2 * 1024 * 1024
_NAME = re.compile(r"(.*)-(\d{8}-\d{6})(?:-\d+)?\.md")


def list_reports(report_dir: Path) -> list[dict]:
    """目录里的 .md 文件，新的在前；没有目录就是空。"""
    rows = []
    try:
        for path in report_dir.glob("*.md"):
            try:
                if not path.is_file():
                    continue
                st = path.stat()
            except OSError:
                continue
            m = _NAME.fullmatch(path.name)
            rows.append({"name": path.name, "scenario": m.group(1) if m else path.name[:-3],
                         "time": st.st_mtime, "size": st.st_size})
    except OSError:
        return []
    rows.sort(key=lambda r: r["time"], reverse=True)
    return rows


def read_report(report_dir: Path, name: str) -> tuple[int, dict]:
    """名字必须恰好是列表里的某一项（挡掉所有路径穿越）。"""
    name = unquote(name)
    row = next((r for r in list_reports(report_dir) if r["name"] == name), None)
    if row is None:
        return 404, {"ok": False, "text": "没有这份报告"}
    if row["size"] > MAX_REPORT:
        return 413, {"ok": False, "text": "报告太大（超过 2 MB）"}
    try:
        text = (report_dir / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 404, {"ok": False, "text": "没有这份报告"}
    return 200, {"name": name, "text": text}
