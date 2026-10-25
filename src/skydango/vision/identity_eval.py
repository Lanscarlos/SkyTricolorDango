"""身份板的离线标准答案与评估。草稿须人工核对，不能把自动初判当真值。"""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..imageio import imread, imwrite
from .bubbles import Rect


@dataclass
class TrackLog:
    frames: list[tuple[str, Rect]] = field(default_factory=list)
    names: list[tuple[float, str]] = field(default_factory=list)
    self_hits: int = 0
    times: list[float] = field(default_factory=list)  # 含没挂标签的帧，才能判断名字是否连续


def draft_identities(tracks_seen: dict[int, TrackLog], tag_hold: float) -> dict[int, str]:
    identities = {}
    for id, log in sorted(tracks_seen.items()):
        named = dict(log.names)
        times = log.times or [t for t, _ in log.names]
        votes = Counter()
        run_name, start, count = None, 0.0, 0
        last = 0.0

        def finish():
            if run_name and last - start + 1e-9 >= tag_hold:
                votes[run_name] += count

        for at in times:
            name = named.get(at)
            if name != run_name:
                finish()
                run_name, start, count = name, at, 0
            count += 1
            last = at
        finish()
        if votes:
            identities[id] = min(votes, key=lambda n: (-votes[n], n))
        elif log.frames and log.self_hits * 2 >= len(log.frames):
            identities[id] = '团子'
        else:
            identities[id] = '?'
    return identities


def write_draft(path: Path, ids: dict[int, str], logs: dict[int, TrackLog]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# 初判不是标准答案：请对着 sheet.jpg 改成好友名 / 团子 / 陌生人 / 不要 / ?。']
    for id, who in sorted(ids.items()):
        log = logs[id]
        times = log.times or [t for t, _ in log.names] or [0.0]
        lines.append(f't{id} = {json.dumps(who, ensure_ascii=False)}  # 出现 {min(times):.1f}~{max(times):.1f} s，{len(log.frames)} 帧')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def save_logs(path: Path, logs: dict[int, TrackLog], **meta) -> None:
    rows = {str(id): {'frames': [[name, [b.x, b.y, b.w, b.h]] for name, b in log.frames],
                     'names': log.names, 'self_hits': log.self_hits, 'times': log.times}
            for id, log in sorted(logs.items())}
    Path(path).write_text(json.dumps({'tracks': rows, **meta}, ensure_ascii=False) + '\n', encoding='utf-8')


def load_logs(path: Path) -> dict[int, TrackLog]:
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['tracks']
    return {int(id): TrackLog([(name, Rect(*box)) for name, box in row['frames']],
                            [(t, name) for t, name in row['names']], row['self_hits'], row.get('times', []))
            for id, row in rows.items()}


def apply_draft(draft: Path, logs_path: Path, out: Path, *, friends: list[str]) -> int:
    text = Path(draft).read_text(encoding='utf-8')
    ids = tomllib.loads(text)
    logs = load_logs(logs_path)
    lines = {m.group(1): n for n, line in enumerate(text.splitlines(), 1)
             if (m := re.match(r'\s*"?(t\d+)"?\s*=', line))}
    allowed = set(friends) | {'团子', '陌生人', '不要', '?'}
    rows = []
    # 先把整份草稿验完，再写文件，错一行也不盖掉旧答案。
    for key, who in ids.items():
        line = lines.get(key, 1)
        if not re.fullmatch(r't\d+', key) or int(key[1:]) not in logs:
            raise ValueError(f'第 {line} 行：没有轨迹 {key}')
        if not isinstance(who, str) or who not in allowed:
            raise ValueError(f'第 {line} 行：{who!r} 不是好友名，也不是团子 / 陌生人 / 不要 / ?')
        if who in ('不要', '?'):
            continue
        rows.extend({'frame': name, 'box': [b.x, b.y, b.w, b.h], 'who': who}
                    for name, b in logs[int(key[1:])].frames)
    Path(out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    return len(rows)


def write_sheet(path: Path, logs: dict[int, TrackLog], frames_dir: Path, per_track: int = 4) -> None:
    import cv2
    import numpy as np

    per_track = max(1, per_track)
    cell_w, cell_h, columns = per_track * 96, 154, 3
    rows = max(1, (len(logs) + columns - 1) // columns)
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 235, np.uint8)
    for n, (id, log) in enumerate(sorted(logs.items())):
        x, y = n % columns * cell_w, n // columns * cell_h
        cv2.putText(sheet, f'#{id}  {len(log.frames)} frames', (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
        indices = np.linspace(0, len(log.frames) - 1, per_track).astype(int) if log.frames else []
        for k, index in enumerate(indices):
            name, box = log.frames[int(index)]
            img = imread(Path(frames_dir) / name)
            h, w = img.shape[:2]
            crop = img[max(0, box.y):min(h, box.y2), max(0, box.x):min(w, box.x2)]
            if not crop.size:
                continue
            ch, cw = crop.shape[:2]
            scale = min(88 / cw, 118 / ch)
            crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))))
            ch, cw = crop.shape[:2]
            xx, yy = x + k * 96 + (96 - cw) // 2, y + 30 + (118 - ch) // 2
            sheet[yy:yy + ch, xx:xx + cw] = crop
    imwrite(Path(path), sheet)


def collect_logs(items, watcher) -> dict[int, TrackLog]:
    from .perception import same_body

    seen: dict[int, TrackLog] = {}
    now = [0.0]
    watcher.clock = lambda: now[0]
    for at, name, image, panel in items:
        now[0] = at
        watcher.process(image, at, panel)
        if watcher._frame_at != at:
            continue
        tracks = watcher.last_tracks
        selfs = [t for t in tracks if t.cls == 'self']
        players = [t for t in tracks if t.cls in ('player', 'player_unlit')
                   and watcher._admitted(t) and not watcher._other_form(t)]
        people = players + [s for s in selfs if not any(same_body(s.box, p.box) for p in players)]
        tags = [t for t in tracks if t.cls == 'name_tag']
        assigned = watcher._assign_tags([p for p in players if not watcher._unlit(p)], tags)
        for p in people:
            log = seen.setdefault(p.id, TrackLog())
            log.frames.append((name, p.box))
            log.times.append(at)
            label = assigned.get(p.id)
            if label and label.data.get('name'):
                log.names.append((at, label.data['name']))
            cx = p.box.x + p.box.w / 2
            log.self_hits += int(image.shape[1] * 0.25 <= cx <= image.shape[1] * 0.75
                                 and any(same_body(p.box, s.box) for s in selfs))
    return seen
