"""标注页「动作名」的后端：给 `emotes scan` 扫下来的图标起名（设计见 docs/progress/2026-10-02-tonight.md「动作命名页」）。

- 扫描图标在 `<图标库>/scan/NNN.png`；图标库是 `<图标库>/<名字>.png`（`game/wheel.py` 的 `EmoteLibrary`，不读子目录和 `_` 开头的）
- 哪个扫描图标叫什么：按剪影把图标库的每张图配给最像的扫描图标（每张只配一个、分数 ≥ `MATCH`），不靠编号，重扫后不错位
- 起名 = 复制 `scan/NNN.png` → `<名字>.png`；改名 = 重命名库里那张；清除 = 挪进 `_removed/`
- 只起名：团子能做哪些动作照旧由轮盘 / 白名单决定；团子在跑时也能起，下次启动才生效
编号永远不拼进路径：先遍历 scan 目录比对文件名，找得到才用；名字先过 `check_name`。
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..imageio import imread
from ..vision.icons import best_match, silhouette, trim

SCAN, REMOVED = "scan", "_removed"
MATCH = 0.7  # 剪影相似度门槛（同一个图标 ≈ 0.9 以上，不同图标多在 0.5 以下）
MAX_LEN = 12
_SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]
_ID = re.compile(r"^\d{3,4}$")
_BAD_CHARS = set('<>:"/\\|?*')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_LOCK = threading.Lock()  # 每个请求现建一个 EmoteNames：找→改文件 整段串行
_CACHE: dict = {"key": None, "pairs": None}  # 文件没变就不重新配（163 × N 次模板匹配约零点几秒）


def check_name(raw) -> tuple[str, str | None]:
    """(去掉首尾空白的名字, 错误)；没错误时 错误 = None。"""
    if not isinstance(raw, str):
        return "", "名字要是文字"
    name = raw.strip()
    if not name:
        return name, "名字不能是空的"
    if len(name) > MAX_LEN:
        return name, f"名字最多 {MAX_LEN} 个字"
    if any(c in _BAD_CHARS or ord(c) < 32 for c in name):
        return name, '名字里不能有 < > : " / \\ | ? * 和控制字符'
    if name[0] in "_.":
        return name, "名字不能以 _ 或 . 开头（图标库会忽略 _ 开头的文件）"
    if name[-1] in ". ":
        return name, "名字不能以 . 或空格结尾（Windows 文件名不允许）"
    if name.split(".")[0].lower() in _RESERVED:
        return name, f"「{name}」是 Windows 保留的文件名，换一个"
    return name, None


def references(cfg: Config, name: str) -> list[str]:
    """配置里引用这个动作名的地方（改名 / 清除时提醒，不自动改）。"""
    out = []
    if cfg.social.after_light == name:
        out.append("social.after_light")
    if name in cfg.emotes.extra:
        out.append("emotes.extra")
    out += [f"gesture.names.{k}" for k, v in cfg.gesture.names.items() if v == name]
    if name in cfg.reflex.addressed:
        out.append("reflex.addressed")
    if name in cfg.reflex.idle:
        out.append("reflex.idle")
    out += [f"reflex.return_map.{k}" for k, v in cfg.reflex.return_map.items() if v == name]
    return out


def _mask(path: Path) -> np.ndarray | None:
    try:
        img = imread(path)
    except (OSError, RuntimeError):
        return None
    m = trim(silhouette(img))
    return m if m.any() else None


def _score(scan: np.ndarray, lib: np.ndarray) -> float:
    pad = 8
    big = cv2.copyMakeBorder(scan, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return best_match(big, lib, _SCALES).score


class EmoteNames:
    def __init__(self, library: str | Path, cfg: Config | None = None) -> None:
        self.lib, self.cfg = Path(library), cfg

    # ---- 文件 ----
    def _scan_files(self) -> dict[str, Path]:
        d = self.lib / SCAN
        if not d.is_dir():
            return {}
        return {p.stem: p for p in sorted(d.glob("*.png")) if _ID.match(p.stem) and p.is_file()}

    def _lib_files(self) -> dict[str, Path]:
        if not self.lib.is_dir():
            return {}
        return {p.stem: p for p in sorted(self.lib.glob("*.png")) if not p.stem.startswith("_") and p.is_file()}

    def _pairs(self, scans: dict[str, Path], libs: dict[str, Path]) -> dict[str, tuple[str, float]]:
        """扫描编号 → (库里的名字, 分数)。分数从高到低贪心配，每张只配一次。"""
        key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in [*scans.values(), *libs.values()])
        if _CACHE["key"] == key:
            return dict(_CACHE["pairs"])
        smask = {i: _mask(p) for i, p in scans.items()}
        cands = []
        for name, p in libs.items():
            lm = _mask(p)
            if lm is None:
                continue
            for i, sm in smask.items():
                if sm is None:
                    continue
                s = _score(sm, lm)
                if s >= MATCH:
                    cands.append((s, i, name))
        cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        pairs: dict[str, tuple[str, float]] = {}
        used: set[str] = set()
        for s, i, name in cands:
            if i in pairs or name in used:
                continue
            pairs[i] = (name, s)
            used.add(name)
        _CACHE["key"], _CACHE["pairs"] = key, dict(pairs)
        return pairs

    def _refs(self, name: str) -> list[str]:
        return references(self.cfg, name) if self.cfg is not None and name else []

    # ---- 接口 ----
    def state(self) -> dict:
        scans = self._scan_files()
        if not scans:
            return {"ok": False, "text": f"{self.lib / SCAN} 里还没有扫描图标：先在终端跑 python -m skydango emotes scan"}
        libs = self._lib_files()
        pairs = self._pairs(scans, libs)
        items = [{"id": i, "name": pairs[i][0] if i in pairs else None, "score": round(pairs[i][1], 3) if i in pairs else None}
                 for i in scans]
        matched = {n for n, _ in pairs.values()}
        return {"ok": True, "library": str(self.lib), "items": items, "total": len(items), "named": len(pairs),
                "orphans": [n for n in libs if n not in matched]}

    def icon(self, i: str) -> bytes | None:
        p = self._scan_files().get(i) if isinstance(i, str) else None
        if p is None:
            return None
        try:
            return p.read_bytes()
        except OSError:
            return None

    def name(self, i, raw) -> tuple[int, dict]:
        name, err = check_name(raw)
        if err:
            return 400, {"ok": False, "text": err}
        with _LOCK:
            scans = self._scan_files()
            src = scans.get(i) if isinstance(i, str) else None
            if src is None:
                return 404, {"ok": False, "text": "没有这个图标（刷新一下）"}
            libs = self._lib_files()
            old = self._pairs(scans, libs).get(i, (None, 0.0))[0]
            if old == name:
                return 200, {"ok": True, "id": i, "name": name, "old": old, "refs": []}
            if any(n.casefold() == name.casefold() and n != old for n in libs):
                return 409, {"ok": False, "text": f"图标库里已经有「{name}」了，重名了：同一个动作不同等级分开起名，比如「{name}2级」"}
            dst = self.lib / f"{name}.png"
            try:
                if old is not None:
                    tmp = self.lib / f"_renaming-{time.time_ns()}.png"  # 只改大小写时 Windows 上直接改名可能不生效：先挪开
                    libs[old].rename(tmp)
                    tmp.rename(dst)
                else:
                    shutil.copyfile(src, dst)
            except OSError as exc:
                return 409, {"ok": False, "text": f"改不了文件：{exc}"}
            return 200, {"ok": True, "id": i, "name": name, "old": old, "refs": self._refs(old) if old else []}

    def clear(self, i) -> tuple[int, dict]:
        with _LOCK:
            scans = self._scan_files()
            if not isinstance(i, str) or i not in scans:
                return 404, {"ok": False, "text": "没有这个图标（刷新一下）"}
            libs = self._lib_files()
            old = self._pairs(scans, libs).get(i, (None, 0.0))[0]
            if old is None:
                return 409, {"ok": False, "text": "这个图标还没有名字"}
            bin_dir = self.lib / REMOVED
            dst = bin_dir / f"{old}.png"
            if dst.exists():
                dst = bin_dir / f"{old}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000}.png"
            try:
                bin_dir.mkdir(exist_ok=True)
                libs[old].rename(dst)
            except OSError as exc:
                return 409, {"ok": False, "text": f"挪不动：{exc}"}
            return 200, {"ok": True, "id": i, "name": None, "old": old, "refs": self._refs(old)}
