"""身份标准答案的人工标注；只处理离线草稿，不接触设备。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import tomllib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..vision.identity_eval import (
    check_label, draft_identities, frame_times, label_at, load_logs,
    parse_label, read_draft, render_draft,
)

_LOCK = threading.RLock()
_RESERVED = {'团子', '陌生人', '不要', '?'}


def _error(code, text):
    return code, {'ok': False, 'text': text}


def _replace_files(files: dict[Path, bytes | None]) -> None:
    """先准备所有临时文件，失败则恢复本次操作前的文件。"""
    old = {p: p.read_bytes() if p.exists() else None for p in files}
    staged, changed = {}, []
    try:
        for path, data in files.items():
            if data is not None:
                fd, name = tempfile.mkstemp(prefix='.identity-', dir=path.parent)
                staged[path] = Path(name)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(data)
        for path, data in files.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(staged[path], path)
            changed.append(path)
    except OSError:
        for path in reversed(changed):
            if old[path] is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(old[path])
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def _keep_comments(rendered: str, original: str) -> str:
    """保留开头说明和轨迹原注释；名字中的 # 不视为注释。"""
    header, comments = [], {}
    started = False
    for row in original.splitlines():
        match = re.match(r'\s*"?t(\d+)"?\s*=', row)
        if match:
            started = True
            for at, char in enumerate(row):
                if char != '#':
                    continue
                try:
                    tomllib.loads(row[:at])
                except tomllib.TOMLDecodeError:
                    continue
                comments[int(match[1])] = row[at:]
                break
            else:
                comments[int(match[1])] = ''
        elif not started:
            header.append(row)
    rows = header if header else rendered.splitlines()[:1]
    for row in rendered.splitlines()[1:]:
        id = int(re.match(r't(\d+)', row)[1])
        if id in comments:
            # 默认注释在 JSON 字符串之后，取最后的出现时间标记。
            row = row.rsplit('  # 出现 ', 1)[0]
            if comments[id]:
                row += '  ' + comments[id]
        rows.append(row)
    return '\n'.join(rows) + '\n'


class IdentityTruth:
    def __init__(self, root: Path, friends: Callable[[], list[str]], tag_hold: float) -> None:
        self.root, self.friends, self.tag_hold = Path(root), friends, tag_hold

    def _run(self, run):
        if not isinstance(run, str) or not re.fullmatch(r'[\w\-.]+', run) or run in ('.', '..'):
            raise FileNotFoundError('没有这段录像的草稿')
        path = self.root / run
        if (not path.is_dir() or path.resolve().parent != self.root.resolve()
                or not (path / 'truth-draft.toml').is_file() or not (path / 'logs.json').is_file()):
            raise FileNotFoundError('没有这段录像的草稿')
        return path

    def _friends(self):
        try:
            return list(dict.fromkeys(self.friends()))
        except (OSError, ValueError):
            return []

    def _read(self, run):
        path = self._run(run)
        draft = path / 'truth-draft.toml'
        mtime = draft.stat().st_mtime_ns
        labels, lines = read_draft(draft)
        logs = load_logs(path / 'logs.json')
        friends = self._friends()
        for id, segs in labels.items():
            if id not in logs:
                raise ValueError(f'第 {lines[id]} 行：没有轨迹 t{id}')
            check_label(segs, logs[id], set(friends) | _RESERVED, lines[id], f't{id}')
        reviewed = path / 'reviewed.json'
        seen = set(json.loads(reviewed.read_text(encoding='utf-8'))['tracks']) if reviewed.exists() else set()
        if draft.stat().st_mtime_ns != mtime:
            raise ValueError('草稿在读取时改过，刷新一下')
        return path, labels, logs, friends, seen, mtime

    def runs(self) -> dict:
        if not self.root.is_dir():
            return {'runs': []}
        names = []
        for path in self.root.iterdir():
            try:
                self._run(path.name)
            except (OSError, ValueError):
                continue
            names.append(path.name)
        return {'runs': sorted(names, reverse=True)}

    def state(self, run: str) -> tuple[int, dict]:
        with _LOCK:
            try:
                path, labels, logs, friends, seen, mtime = self._read(run)
                source = json.loads((path / 'logs.json').read_text(encoding='utf-8'))['source']
                guesses = draft_identities(logs, self.tag_hold)
                tracks, done, total = [], 0, 0
                for id, log in sorted(logs.items(), key=lambda row: (-len(row[1].frames), row[0])):
                    segs = labels.get(id, [(0.0, '?')])
                    reviewed = id in seen or segs != [(0.0, guesses[id])]
                    names = dict(log.names)
                    frames = [{'name': name, 't': at, 'box': [b.x, b.y, b.w, b.h], 'tag': names.get(at)}
                              for (name, b), at in zip(log.frames, frame_times(log))]
                    total += len(frames)
                    done += sum(reviewed and label_at(segs, f['t']) != '?' for f in frames)
                    tracks.append(dict(id=id, frames=frames, guess=guesses[id],
                                       segments=[list(s) for s in segs], reviewed=reviewed,
                                       two_names=len(set(names.values())) >= 2))
                return 200, dict(ok=True, run=run, mtime=mtime, source=source,
                                 source_missing=not Path(source).is_dir(), friends=friends,
                                 friends_note=None if friends else '读不到好友名单（friends.md），只能标团子 / 陌生人 / 不要 / ?',
                                 coverage=dict(done=done, total=total), tracks=tracks)
            except FileNotFoundError as exc:
                return _error(404, str(exc))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return _error(400, str(exc))

    def _image(self, run, frame, track=None):
        try:
            path = self._run(run)
            logs = load_logs(path / 'logs.json')
            candidates = logs.values() if track is None else [logs[track]]
            box = next((b for log in candidates for name, b in log.frames if name == frame), None)
            if box is None or not isinstance(frame, str) or Path(frame).name != frame or ':' in frame:
                return None
            source = Path(json.loads((path / 'logs.json').read_text(encoding='utf-8'))['source'])
            target = source / frame
            if target.resolve().parent != source.resolve():
                return None
            img = cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            if track is not None:
                x1, x2 = max(0, int(box.x - box.w * .15)), min(w, int(box.x2 + box.w * .15))
                y1, y2 = max(0, int(box.y - box.h * .15)), min(h, int(box.y2 + box.h * .15))
                if x2 <= x1 or y2 <= y1:
                    return None
                img = img[y1:y2, x1:x2]
                h, w = img.shape[:2]
                if h < 160:
                    img = cv2.resize(img, (max(1, round(w * 160 / h)), 160))
            elif w > 1280:
                img = cv2.resize(img, (1280, max(1, round(h * 1280 / w))))
            ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return encoded.tobytes() if ok else None
        except (OSError, ValueError, KeyError, TypeError, cv2.error):
            return None

    def crop(self, run: str, track: int, frame: str) -> bytes | None:
        return self._image(run, frame, track)

    def frame(self, run: str, frame: str) -> bytes | None:
        return self._image(run, frame)

    def _edit(self, body, undo=False):
        with _LOCK:
            try:
                path = self._run(body.get('run'))
                draft, reviewed = path / 'truth-draft.toml', path / 'reviewed.json'
                mtime = draft.stat().st_mtime_ns
                if body.get('mtime') != mtime:
                    return _error(409, '草稿在别处改过，刷新一下')
                bak, rbak = path / 'truth-draft.toml.bak', path / 'reviewed.json.bak'
                if undo:
                    if not bak.is_file() or not rbak.is_file():
                        return _error(400, '没有可以撤销的')
                    files = {draft: bak.read_bytes(), reviewed: rbak.read_bytes(), bak: None, rbak: None}
                else:
                    _, labels, logs, friends, seen, read_mtime = self._read(body.get('run'))
                    if read_mtime != mtime:
                        return _error(409, '草稿在别处改过，刷新一下')
                    id = body.get('track')
                    if type(id) is not int or id not in logs:
                        return _error(400, '没有这条轨迹')
                    segs = parse_label(body.get('segments'), 1)
                    check_label(segs, logs[id], set(friends) | _RESERVED, 1, f't{id}')
                    segs[0] = (0.0, segs[0][1])
                    # 归零后仍须递增，避免负时间切点造出不可读的草稿。
                    parse_label([list(s) for s in segs], 1)
                    labels[id] = segs
                    original = draft.read_text(encoding='utf-8')
                    text = _keep_comments(render_draft(labels, logs), original)
                    if b'\r\n' in draft.read_bytes():
                        text = text.replace('\n', '\r\n')
                    seen.add(id)
                    files = {bak: draft.read_bytes(), rbak: reviewed.read_bytes() if reviewed.exists() else b'{"tracks": []}',
                             reviewed: json.dumps({'tracks': sorted(seen)}, ensure_ascii=False).encode('utf-8'),
                             draft: text.encode('utf-8')}
                if draft.stat().st_mtime_ns != mtime:
                    return _error(409, '草稿在别处改过，刷新一下')
                _replace_files(files)
                return 200, {'ok': True, 'mtime': draft.stat().st_mtime_ns}
            except FileNotFoundError as exc:
                return _error(404, str(exc))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return _error(400, f'保存失败：{exc}')

    def label(self, body: dict) -> tuple[int, dict]:
        return self._edit(body)

    def undo(self, body: dict) -> tuple[int, dict]:
        return self._edit(body, undo=True)
