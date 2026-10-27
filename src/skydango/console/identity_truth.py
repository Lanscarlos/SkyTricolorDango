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
    check_label, draft_identities, draft_key, frame_times, label_at, load_logs,
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
        key = draft_key(row)
        if key:
            started = True
            for at, char in enumerate(row):
                if char != '#':
                    continue
                try:
                    tomllib.loads(row[:at])
                except tomllib.TOMLDecodeError:
                    continue
                comments[int(key[1:])] = row[at:]
                break
            else:
                comments[int(key[1:])] = ''
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
        # 名单暂时不可用时仍可看旧答案；本次新标签在 _edit 另按当前名单校验。
        readable_names = set(friends) if friends else {who for segs in labels.values() for _, who in segs}
        for id, segs in labels.items():
            if id not in logs:
                raise ValueError(f'第 {lines[id]} 行：没有轨迹 t{id}')
            check_label(segs, logs[id], readable_names | _RESERVED, lines[id], f't{id}')
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
                return 200, dict(ok=True, run=run, mtime=str(mtime), source=source,
                                 source_missing=not Path(source).is_dir(), friends=friends,
                                 friends_note=None if friends else '读不到好友名单（friends.md），只能标团子 / 陌生人 / 不要 / ?',
                                 coverage=dict(done=done, total=total), tracks=tracks)
            except FileNotFoundError as exc:
                return _error(404, str(exc))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return _error(400, str(exc))

    def _image(self, run, frame, track=None, dimensions=False):
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
            original_size = (w, h)
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
            if not ok:
                return None
            return (encoded.tobytes(), original_size) if dimensions else encoded.tobytes()
        except (OSError, ValueError, KeyError, TypeError, cv2.error):
            return None

    def crop(self, run: str, track: int, frame: str) -> bytes | None:
        return self._image(run, frame, track)

    def frame(self, run: str, frame: str) -> bytes | None:
        return self._image(run, frame)

    def frame_view(self, run: str, frame: str):
        """整帧及原始尺寸，供缩图上叠加标注框。"""
        return self._image(run, frame, dimensions=True)

    def _edit(self, body, undo=False):
        with _LOCK:
            try:
                path = self._run(body.get('run'))
                draft, reviewed = path / 'truth-draft.toml', path / 'reviewed.json'
                mtime = draft.stat().st_mtime_ns
                if body.get('mtime') != str(mtime):
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
                    tomllib.loads(text)  # 保留注释之后仍须是可读草稿，失败不得覆盖。
                    if b'\r\n' in draft.read_bytes():
                        text = text.replace('\n', '\r\n')
                    mark_reviewed, record_undo = body.get('reviewed', True), body.get('record_undo', True)
                    if type(mark_reviewed) is not bool or type(record_undo) is not bool:
                        return _error(400, '审核状态和撤销标记必须是布尔值')
                    if mark_reviewed:
                        seen.add(id)
                    else:
                        seen.discard(id)
                    # 前端逆向恢复历史时不生成“重做”备份，避免刷新后又恢复误操作。
                    files = {bak: draft.read_bytes() if record_undo else None,
                             rbak: (reviewed.read_bytes() if reviewed.exists() else b'{"tracks": []}') if record_undo else None,
                             reviewed: json.dumps({'tracks': sorted(seen)}, ensure_ascii=False).encode('utf-8'),
                             draft: text.encode('utf-8')}
                    previous = body.get('next_undo')
                    if not record_undo and previous is not None:
                        if not isinstance(previous, dict) or type(previous.get('track')) is not int or previous['track'] not in logs:
                            return _error(400, '撤销历史里没有这条轨迹')
                        prior_id = previous['track']
                        prior_segs = parse_label(previous.get('segments'), 1)
                        check_label(prior_segs, logs[prior_id], set(friends) | _RESERVED, 1, f't{prior_id}')
                        if type(previous.get('reviewed')) is not bool:
                            return _error(400, '撤销历史缺少审核状态')
                        prior_seen = seen | {prior_id} if previous['reviewed'] else seen - {prior_id}
                        prior_text = _keep_comments(render_draft(labels | {prior_id: prior_segs}, logs), original)
                        tomllib.loads(prior_text)
                        if b'\r\n' in draft.read_bytes():
                            prior_text = prior_text.replace('\n', '\r\n')
                        files[bak] = prior_text.encode('utf-8')
                        files[rbak] = json.dumps({'tracks': sorted(prior_seen)}).encode('utf-8')
                if draft.stat().st_mtime_ns != mtime:
                    return _error(409, '草稿在别处改过，刷新一下')
                _replace_files(files)
                return 200, {'ok': True, 'mtime': str(draft.stat().st_mtime_ns)}
            except FileNotFoundError as exc:
                return _error(404, str(exc))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return _error(400, f'保存失败：{exc}')

    def label(self, body: dict) -> tuple[int, dict]:
        return self._edit(body)

    def undo(self, body: dict) -> tuple[int, dict]:
        return self._edit(body, undo=True)
