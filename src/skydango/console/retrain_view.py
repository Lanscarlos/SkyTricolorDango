"""整帧页的重训区（spec 2026-10-04-hardcase-inbox-design §7.1 / §7.3）：最近一次 tmp/retrain/<时间>/ 的报告，
「换上」新模型（写 console.toml，绝不碰 config.toml）和「回退」。

adopt.json 每项记 `{"old", "new", "t", "had_override"}`：had_override = 换之前 console.toml 自己有没有这一项。
回退时有就写回 old，没有就 revert（让 config.toml 的值重新生效，不把它钉进 console.toml）。
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

from .settings import SettingsStore, _has
from .tomlfile import read

KEYS = {"yolo": "perception.model", "attrs": "attrs.model"}
_LOCK = threading.Lock()


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class RetrainView:
    def __init__(self, root: Path, store: SettingsStore) -> None:
        self.root, self.store = Path(root), store

    def _latest_dir(self) -> Path | None:
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []
        return dirs[-1] if dirs else None

    def latest(self) -> dict:
        d = self._latest_dir()
        if d is None:
            return {"ok": True, "dir": None, "report": "", "result": None, "adopted": None}
        try:
            report = (d / "report.md").read_text(encoding="utf-8")
        except OSError:
            report = ""
        return {"ok": True, "dir": d.name, "report": report, "result": _json(d / "result.json"), "adopted": _json(d / "adopt.json")}

    def _adopted(self, d: Path) -> dict:
        data = _json(d / "adopt.json")
        return data if isinstance(data, dict) else {}

    def _write(self, d: Path, data: dict) -> None:
        p = d / "adopt.json"
        if data:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        elif p.exists():
            p.unlink()

    def adopt(self, what: str, running: bool) -> tuple[int, dict]:
        if what not in KEYS:
            return 400, {"ok": False, "text": "what 要是 yolo 或 attrs"}
        d = self._latest_dir()
        result = _json(d / "result.json") if d else None
        if not isinstance(result, dict) or not result.get(what):
            return 409, {"ok": False, "text": "最近一次重训没有做完（没有 result.json），不能换上"}
        new = str(result[what])
        if not Path(new).is_file():
            return 409, {"ok": False, "text": f"新模型文件不存在：{new}"}
        key = KEYS[what]
        with _LOCK:
            try:
                cfg = self.store.effective()
                console = read(self.store.console_path)
            except ValueError as exc:
                return 409, {"ok": False, "text": str(exc)}
            old = cfg.perception.model if what == "yolo" else cfg.attrs.model
            saved = self.store.save({key: new})
            if not saved["ok"]:
                return 409, {"ok": False, "text": "；".join(saved["errors"].values())}
            data = self._adopted(d)
            prev = data.get(what)
            if prev and prev.get("new") == new:  # 重复点：别把"回退目标"盖成新模型自己
                return 200, self._ok(running, {"adopted": data})
            data[what] = {"old": old, "new": new, "t": dt.datetime.now().isoformat(timespec="seconds"),
                          "had_override": _has(console, key)}
            self._write(d, data)
        return 200, self._ok(running, {"adopted": data})

    def rollback(self, what: str, running: bool) -> tuple[int, dict]:
        if what not in KEYS:
            return 400, {"ok": False, "text": "what 要是 yolo 或 attrs"}
        d = self._latest_dir()
        data = self._adopted(d) if d else {}
        rec = data.get(what)
        if not rec:
            return 409, {"ok": False, "text": "这次重训的模型还没换上过，没有可回退的"}
        key = KEYS[what]
        with _LOCK:
            saved = self.store.save({key: rec["old"]}) if rec.get("had_override") else self.store.save({}, [key])
            if not saved["ok"]:
                return 409, {"ok": False, "text": "；".join(saved["errors"].values())}
            data.pop(what)
            self._write(d, data)
        return 200, self._ok(running, {"adopted": data or None})

    @staticmethod
    def _ok(running: bool, extra: dict) -> dict:
        out = {"ok": True, **extra}
        if running:
            out["note"] = "下次叫醒生效"
        return out
