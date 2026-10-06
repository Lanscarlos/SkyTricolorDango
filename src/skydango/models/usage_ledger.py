"""用量账本（spec 2026-10-06-model-usage §4）：runs/usage.json 按 日期 → 来源 → "用处|供应商/模型|main或backup" 累加。

每次 add 把一份增量加进文件（读 → 相加 → 临时文件 → 原子替换），用 usage.json.lock 和同时在跑的离线命令串起来；
只留最近 keep_days 天。文件坏了改名 .bad-<时间>、从空的开始。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from .usage import merge_rows

log = logging.getLogger(__name__)

LOCK_WAIT = 2.0  # 拿锁最多等几秒
LOCK_STALE = 30.0  # 超过这么久的锁当成上次崩掉留下的
VERSION = 1


def ledger_for(cfg) -> "Ledger | None":
    """[usage] ledger 的账本；环境变量 SKYDANGO_USAGE_LEDGER 盖过配置（测试设成空 = 不记账本）。"""
    path = os.environ.get("SKYDANGO_USAGE_LEDGER", cfg.usage.ledger)
    return Ledger(Path(path), cfg.usage.keep_days) if path else None


class Ledger:
    def __init__(self, path: Path, keep_days: int = 7) -> None:
        self.path = Path(path)
        self.keep_days = keep_days
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def read(self) -> dict:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": VERSION, "days": {}}
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get("days"), dict):
                raise ValueError("不是账本")
            return data
        except ValueError:
            bad = self.path.with_name(f"{self.path.name}.bad-{time.strftime('%Y%m%d-%H%M%S')}")
            try:
                self.path.rename(bad)
                log.warning("用量账本坏了，改名成 %s，从空的开始", bad.name)
            except OSError:
                log.warning("用量账本坏了，也改不了名：%s", self.path)
            return {"version": VERSION, "days": {}}

    def day_rows(self, day: str) -> dict:
        """这一天所有来源合并成 {key: 行}。读不出来是空的。"""
        try:
            sources = self.read()["days"].get(day) or {}
        except OSError:
            return {}
        rows: dict = {}
        for source_rows in sources.values():
            if isinstance(source_rows, dict):
                rows = merge_rows(rows, source_rows)
        return rows

    def _lock(self) -> bool:
        deadline = time.monotonic() + LOCK_WAIT
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > LOCK_STALE:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def add(self, delta: dict, today: str) -> bool:
        """把增量加进文件；拿不到锁 / 写失败返回 False（调用方留着增量下次再写）。"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self._lock():
                log.debug("用量账本被占着，这次不写")
                return False
        except OSError:
            log.debug("用量账本的锁建不了", exc_info=True)
            return False
        try:
            data = self.read()
            days = data["days"]
            for day, sources in delta.items():
                into = days.setdefault(day, {})
                for source, rows in sources.items():
                    into[source] = merge_rows(into.get(source) or {}, rows)
            floor = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=self.keep_days)).strftime("%Y-%m-%d")
            data["days"] = {d: v for d, v in sorted(days.items()) if d > floor}
            data["version"] = VERSION
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except (OSError, ValueError):
            log.warning("用量账本写不进去：%s", self.path, exc_info=True)
            return False
        finally:
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass
