"""面板自己的 TOML 读写：标准库 tomllib 只能读，写出器只支持面板用得到的类型（字符串、布尔、数字、字符串列表、表）。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def read(path: Path) -> dict:
    """不存在返回 {}；语法错抛 ValueError（带文件名，设置页原样显示）。"""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path.name} 读不出来：{exc}") from exc


def _string(text: str) -> str:
    # JSON 字符串的转义是 TOML basic string 的子集，只差 U+007F：JSON 不转义，TOML 不许原样出现
    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007f")


def _key(key: str) -> str:
    return key if _BARE.match(key) else _string(key)


def _value(value) -> str:
    # JSON 字符串的转义是 TOML basic string 的子集；bool 要先于 int 判断
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "[" + ", ".join(_string(v) for v in value) + "]"
    raise ValueError(f"TOML 写不了这种值：{value!r}")


def _table(data: dict, prefix: list[str], out: list[str]) -> None:
    scalars = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    tables = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if prefix and (scalars or not tables):
        out.append(f"[{'.'.join(_key(p) for p in prefix)}]")
    for k, v in scalars:
        out.append(f"{_key(k)} = {_value(v)}")
    if prefix and scalars:
        out.append("")
    for k, v in tables:
        _table(v, prefix + [k], out)


def dumps(data: dict) -> str:
    out: list[str] = []
    _table(data, [], out)
    return "\n".join(out).rstrip("\n") + "\n"


def write_atomic(path: Path, text: str) -> None:
    """同目录临时文件写完再替换：写到一半崩了也不会留下坏文件。"""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
