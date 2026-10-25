import json
from pathlib import Path

import numpy as np
import pytest

from skydango.imageio import imwrite, imread
from skydango.vision.bubbles import Rect
from skydango.vision.identity_eval import (
    TrackLog, draft_identities, write_draft, write_sheet, apply_draft, save_logs,
)

R = Rect(10, 20, 30, 60)


def logs():
    return {
        1: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R), ('2_0.60s.jpg', R)],
                    [(0, '小明'), (0.3, '小明'), (0.6, '小明')], 0, [0, 0.3, 0.6]),
        2: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R), ('2_0.60s.jpg', R)],
                    [(0, '小明')], 0, [0, 0.3, 0.6]),
        3: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R)], [], 2, [0, 0.3]),
    }


def test_draft_identities_stable_flash_and_self():
    assert draft_identities(logs(), 0.5) == {1: '小明', 2: '?', 3: '团子'}


def test_missing_tag_frame_breaks_confirmation():
    log = logs()[1]
    log.names = [(0, '小明'), (0.6, '小明')]
    assert draft_identities({1: log}, 0.5) == {1: '?'}


def test_apply_draft_round_trip_and_ignored(tmp_path):
    rows = logs()
    draft, seen, out = (tmp_path / n for n in ('draft.toml', 'logs.json', 'truth.jsonl'))
    write_draft(draft, draft_identities(rows, 0.5), rows)
    save_logs(seen, rows)
    draft.write_text(draft.read_text(encoding='utf-8').replace('t2 = "?"', 't2 = "陌生人"'), encoding='utf-8')
    assert apply_draft(draft, seen, out, friends=['小明']) == 8
    truth = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert truth[0] == {'frame': '0_0.00s.jpg', 'box': [10, 20, 30, 60], 'who': '小明'}
    assert [r['who'] for r in truth].count('陌生人') == 3


def test_apply_invalid_friend_reports_line_without_overwriting(tmp_path):
    draft, seen, out = (tmp_path / n for n in ('draft.toml', 'logs.json', 'truth.jsonl'))
    save_logs(seen, logs())
    draft.write_text('# 注释\nt1 = "老登"\n', encoding='utf-8')
    out.write_text('保留', encoding='utf-8')
    with pytest.raises(ValueError, match='第 2 行'):
        apply_draft(draft, seen, out, friends=['小明'])
    assert out.read_text(encoding='utf-8') == '保留'


def test_sheet_creates_viewable_crops(tmp_path):
    rows = logs()
    for name, _ in rows[1].frames:
        imwrite(tmp_path / name, np.full((100, 100, 3), (30, 100, 200), np.uint8))
    write_sheet(tmp_path / 'sheet.jpg', rows, tmp_path)
    image = imread(tmp_path / 'sheet.jpg')
    assert image.shape[0] >= 100 and image.shape[1] >= 200
    assert (image[:, :, 2] > image[:, :, 0] + 50).any()


def test_truth_cli_generates_and_applies_with_real_watcher(tmp_path, monkeypatch):
    from skydango import cli
    from test_perception import FRIENDS, player, tag
    from test_perception_appearance import make, paint, PINK

    source = tmp_path / 'recording'
    source.mkdir()
    image = paint((player(1000), PINK))
    for n, t in enumerate((0, 0.3, 0.6)):
        imwrite(source / f'{n}_{t:.2f}s.jpg', image)
    w, det = make(appearance=False)
    det.frames = [[player(1000), tag(990, 110)]]
    monkeypatch.setattr(cli, '_perception', lambda cfg, args: (det, w))
    monkeypatch.setattr(cli, '_friend_names', lambda cfg: lambda: list(FRIENDS))
    monkeypatch.chdir(tmp_path)
    cli.main(['perception', 'identity-truth', str(source)])
    draft = tmp_path / 'tmp/identity-truth/recording/truth-draft.toml'
    assert FRIENDS[0] in draft.read_text(encoding='utf-8')
    cli.main(['perception', 'identity-truth', str(source), '--apply'])
    truth = [json.loads(line) for line in (source / 'truth.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(truth) == 3 and {r['who'] for r in truth} == {FRIENDS[0]}
