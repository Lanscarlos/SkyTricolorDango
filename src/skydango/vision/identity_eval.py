"""身份板的离线标准答案与评估。草稿须人工核对，不能把自动初判当真值。"""

from __future__ import annotations

import json
import math
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

from ..imageio import imread, imwrite
from .bubbles import Rect
from .identity import DANGO, FRIEND_LOOK, FRIEND_TAG, STRANGER, UNSURE, CallEnd, IdentityBoard

RESERVED = {'团子', '陌生人', '拿不准', '未定', '?', '不要'}
Segments = list[tuple[float, str]]


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


def frame_times(log: TrackLog) -> list[float]:
    if len(log.times) == len(log.frames):
        return log.times
    times = []
    for name, _ in log.frames:
        match = re.search(r'_(\d+(?:\.\d+)?)s\.jpg$', name)
        if not match:
            raise ValueError(f'帧名里没有时间：{name}')
        times.append(float(match[1]))
    return times


def parse_label(value, line: int) -> Segments:
    if isinstance(value, str):
        return [(0.0, value)]
    if not isinstance(value, list) or not value:
        raise ValueError(f'第 {line} 行：分段应为非空的 [开始秒数, 名字] 数组')
    segs = []
    for item in value:
        if (not isinstance(item, list) or len(item) != 2
                or type(item[0]) not in (int, float) or not math.isfinite(item[0])
                or not isinstance(item[1], str)):
            raise ValueError(f'第 {line} 行：每段必须是 [数字, 字符串]，时间须有限')
        if segs and item[0] <= segs[-1][0]:
            raise ValueError(f'第 {line} 行：分段开始时间必须严格递增')
        segs.append((float(item[0]), item[1]))
    return segs


def check_label(segs: Segments, log: TrackLog, allowed: set[str], line: int, key: str) -> None:
    for _, who in segs:
        if who not in allowed:
            raise ValueError(f'第 {line} 行：{who!r} 不是好友名，也不是团子 / 陌生人 / 不要 / ?')
    times = frame_times(log)
    if times and segs[0][0] > times[0]:
        raise ValueError(f'第 {line} 行：{key} 第一段要从 0 开始')


def label_at(segs: Segments, at: float) -> str:
    return next((who for start, who in reversed(segs) if start <= at), segs[0][1])


def read_draft(path: Path) -> tuple[dict[int, Segments], dict[int, int]]:
    text = Path(path).read_text(encoding='utf-8')
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        line = getattr(exc, 'lineno', None) or text.count('\n') + 1
        raise ValueError(f'第 {line} 行：草稿 TOML 格式有误：{exc}') from exc
    lines = {m[1]: n for n, row in enumerate(text.splitlines(), 1)
             if (m := re.match(r'\s*"?(t\d+)"?\s*=', row))}
    labels, numbers = {}, {}
    for key, value in values.items():
        line = lines.get(key, 1)
        if not re.fullmatch(r't\d+', key):
            raise ValueError(f'第 {line} 行：没有轨迹 {key}')
        id = int(key[1:])
        if id in labels:
            raise ValueError(f'第 {line} 行：重复轨迹 {key}')
        labels[id], numbers[id] = parse_label(value, line), line
    return labels, numbers


def render_draft(labels: dict[int, Segments], logs: dict[int, TrackLog]) -> str:
    lines = ['# 初判不是标准答案：请对着 sheet.jpg 改成好友名 / 团子 / 陌生人 / 不要 / ?。']
    for id, segs in sorted(labels.items()):
        log = logs[id]
        times = log.times or [t for t, _ in log.names] or [0.0]
        quoted = lambda who: json.dumps(who, ensure_ascii=False)
        value = quoted(segs[0][1]) if len(segs) == 1 else '[' + ', '.join(
            f'[{start:g}, {quoted(who)}]' for start, who in segs) + ']'
        lines.append(f't{id} = {value}  # 出现 {min(times):.1f}~{max(times):.1f} s，{len(log.frames)} 帧')
    return '\n'.join(lines) + '\n'


def write_draft(path: Path, ids: dict[int, str], logs: dict[int, TrackLog]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_draft({id: [(0.0, who)] for id, who in ids.items()}, logs), encoding='utf-8')


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
    ids, lines = read_draft(draft)
    logs = load_logs(logs_path)
    allowed = set(friends) | {'团子', '陌生人', '不要', '?'}
    rows = []
    # 先把整份草稿验完，再写文件，错一行也不盖掉旧答案。
    for id, segs in ids.items():
        line = lines[id]
        if id not in logs:
            raise ValueError(f'第 {line} 行：没有轨迹 t{id}')
        log = logs[id]
        check_label(segs, log, allowed, line, f't{id}')
        for (name, b), at in zip(log.frames, frame_times(log)):
            who = label_at(segs, at)
            if who not in ('不要', '?'):
                rows.append({'frame': name, 'box': [b.x, b.y, b.w, b.h], 'who': who})
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


def label_of(track) -> str:
    d = track.data
    if d.get('dango') or (track.cls == 'self' and 'identity_verdict' not in d):
        return '团子'
    if d.get('name') or d.get('maybe'):
        return d.get('name') or d['maybe']
    if d.get('unsure'):
        return '拿不准'
    return '陌生人' if d.get('stranger') else '未定'


def score_frames(pred, truth, iou_min: float = 0.5) -> dict:
    from .track import iou

    n = dict(total=0, friend=0, dango=0, other=0, matched=0)
    counts = Counter()
    for name, raw in truth.items():
        actual = [(box, who) for box, who in raw if who not in ('?', '不要')]
        guesses = pred.get(name, [])
        pairs = sorted(((iou(a, b), i, j) for i, (a, _) in enumerate(actual)
                        for j, (b, _) in enumerate(guesses)), key=lambda p: (-p[0], p[1], p[2]))
        matched, used = {}, set()
        for overlap, i, j in pairs:
            if overlap >= iou_min and i not in matched and j not in used:
                matched[i] = guesses[j][1]
                used.add(j)
        n['matched'] += len(matched)
        for i, (_, who) in enumerate(actual):
            guess = matched.get(i, '未定')
            friend = who not in RESERVED
            n['total'] += 1
            n['friend'] += friend
            n['dango'] += who == '团子'
            n['other'] += who != '团子'
            counts['friend_as_stranger'] += friend and guess == '陌生人'
            counts['dango_found'] += who == '团子' and guess == '团子'
            counts['other_as_dango'] += who != '团子' and guess == '团子'
            counts['friend_right'] += friend and guess == who
            counts['friend_wrong'] += friend and guess not in RESERVED and guess != who
            counts['unsure_rate'] += guess == '拿不准'
    denominators = {'friend_as_stranger': 'friend', 'dango_found': 'dango', 'other_as_dango': 'other',
                    'friend_right': 'friend', 'friend_wrong': 'friend', 'unsure_rate': 'total'}
    return {key: counts[key] / n[den] if n[den] else 0.0 for key, den in denominators.items()} | {'n': n}


def _verdict_label(v):
    if v.kind == DANGO:
        return '团子'
    if v.kind in (FRIEND_TAG, FRIEND_LOOK) and v.name:
        return v.name
    if v.kind == UNSURE:
        return '拿不准'
    return '陌生人' if v.kind == STRANGER else '未定'


def replay_board(frames, cfg, acfg, *, call_mode=None, stats=None, call_gap=20.0,
                 auto_window=60.0, auto_quota=3, **board_kw):
    if call_mode not in (None, 'shown', 'dark'):
        raise ValueError('call_mode 要是 shown / dark / None')
    kw = dict(stranger_after=1.0, grace=1.5, stranger_min_height=0.05, track_buffer=1.0, call_window=6.0) | board_kw
    board = IdentityBoard(cfg, acfg, **kw)
    pred, previous, calls, strangers = {}, {}, [], set()
    for index, frame in enumerate(frames):
        live = {t.id for t in frame.tracks}
        unsure = {id: v for id, v in previous.items() if id in live and v.kind == UNSURE}
        recent = [at for at in calls if frame.now - at <= auto_window]
        ready = any(frame.now - v.since >= acfg.unsure_wait for v in unsure.values())
        if (call_mode and ready and not frame.call_end and len(recent) < auto_quota
                and (not calls or frame.now - calls[-1] >= call_gap)):
            shown, taken = {}, set()
            if call_mode == 'shown':
                for id, v in sorted(unsure.items(), key=lambda item: (-item[1].score, item[0])):
                    if v.name and v.name not in taken:
                        shown[id] = v.name
                        taken.add(v.name)
            frame = replace(frame, call_end=CallEnd(frame.now, shown), tracks=tuple(
                replace(t, called_at=frame.now) if t.id in unsure else t for t in frame.tracks))
            calls.append(frame.now)
        out = board.step(frame)
        previous = out.verdicts
        pred[frame.tag or str(index)] = [(t.box, _verdict_label(out.verdicts[t.id])) for t in frame.tracks]
        strangers.update(t.id for t in frame.tracks if out.verdicts[t.id].kind == STRANGER)
    if stats is not None:
        stats.update(calls=len(calls), stranger_tracks=len(strangers))
    return pred


def _passed(scores):
    n = scores['n']
    return (n['friend'] > 0 and n['dango'] > 0 and n['other'] > 0
            and scores['friend_as_stranger'] <= 0.02 and scores['dango_found'] >= 0.90
            and scores['other_as_dango'] <= 0.01)


def sweep(frames, truth, base_cfg, acfg, grid, **board_kw):
    rows = []
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        ap = {k: v for k, v in params.items() if k in ('match', 'unsure', 'margin')}
        ip = {k: v for k, v in params.items() if k not in ap}
        scores = score_frames(replay_board(frames, replace(base_cfg, **ip), replace(acfg, **ap), **board_kw), truth)
        rows.append({'params': params, 'metrics': scores, 'passed': _passed(scores)})
    return sorted(rows, key=lambda r: (not r['passed'], r['metrics']['unsure_rate'],
                                      r['metrics']['friend_as_stranger'], r['metrics']['friend_wrong']))


def report_md(base, cur, sweep, meta) -> str:
    metrics = [('friend_as_stranger', '好友被判陌生人', '≤ 2%'), ('dango_found', '团子认出', '≥ 90%'),
               ('other_as_dango', '别人被判团子', '≤ 1%'), ('friend_right', '好友认对', '记录'),
               ('friend_wrong', '好友认错', '记录'), ('unsure_rate', '拿不准', '记录')]
    lines = ['# 身份板：新旧对比', '', f'录像：{meta.get("source", "?")}；抽帧 {meta.get("fps", "?")} fps。',
             '这份报告只评估人工标准答案覆盖的框；自动草稿未经核对不能作为准确率验收。', '',
             '| 指标 | 旧流程 | 身份板 | 目标 |', '|---|---:|---:|---|']
    for key, title, goal in metrics:
        lines.append(f'| {title} | {base[key]:.2%} | {cur[key]:.2%} | {goal} |')
    lines += ['', f'框数：旧流程 {base["n"]}；身份板 {cur["n"]}。',
              f'陌生人编号：旧流程 {meta.get("baseline_ids", "未记录")}；身份板 {meta.get("current_ids", "未记录")}。',
              '', '## 喊一声（离线假设）', '',
              '以下是两种假设的结果，不是真机准确率；共用现有自动喊的间隔和额度。']
    for mode, title in [('shown', '喊了都亮'), ('dark', '喊了都不亮')]:
        row = meta.get('calls', {}).get(mode, {})
        lines.append(f'- {title}：{row.get("calls", 0)} 声；指标 {row.get("metrics", "未记录")}')
    lines += ['', '## 门槛建议', '',
              '只重放已录下的 FrameIn，外观底库打分固定；候选必须回到原录像复验，不会自动改配置。']
    eligible = [row for row in sweep if row['passed']]
    if not eligible:
        lines.append('没有同时满足三条目标的组合，或标准答案缺好友 / 团子；继续默认关闭。')
    else:
        lines += ['| 参数 | 拿不准占比 |', '|---|---:|']
        lines += [f'| {json.dumps(row["params"], ensure_ascii=False)} | {row["metrics"]["unsure_rate"]:.2%} |'
                  for row in eligible[:20]]
    return '\n'.join(lines) + '\n'


def read_truth(path: Path):
    truth = {}
    for n, text in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            box = Rect(*row['box'])
            if not isinstance(row['who'], str) or box.w <= 0 or box.h <= 0:
                raise ValueError('框或身份无效')
            truth.setdefault(row['frame'], []).append((box, row['who']))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'truth.jsonl 第 {n} 行：{exc}') from exc
    return truth


def collect_predictions(items, watcher):
    from .perception import same_body

    predictions, ids = {}, set()
    now = [0.0]
    watcher.clock = lambda: now[0]
    for at, name, image, panel in items:
        now[0] = at
        old_log_size = len(watcher.identity_log) if watcher.identity_log is not None else 0
        watcher.process(image, at, panel)
        if watcher.identity_log is not None and len(watcher.identity_log) > old_log_size:
            watcher.identity_log[-1] = replace(watcher.identity_log[-1], tag=name)
        if watcher._frame_at != at:
            predictions[name] = []
            continue
        players = [t for t in watcher.last_tracks if t.cls in ('player', 'player_unlit')
                   and watcher._admitted(t) and not watcher._other_form(t)]
        selfs = [t for t in watcher.last_tracks if t.cls == 'self'
                 and not any(same_body(t.box, p.box) for p in players)]
        predictions[name] = [(t.box, label_of(t)) for t in players + selfs]
        ids.update(t.data['sid'] for t in players if t.data.get('sid') and t.data.get('stranger'))
    return predictions, len(ids)
