import json
import shutil

import cv2
import numpy as np
import pytest

from skydango.console.identity_truth import IdentityTruth
from skydango.vision.identity_eval import TrackLog, save_logs, write_draft, draft_identities, read_draft
from skydango.vision.bubbles import Rect


def make(tmp_path, root=None):
    root = root or tmp_path / 'identity'
    run = root / 'r1'
    run.mkdir(parents=True)
    source = tmp_path / '录像'
    source.mkdir(exist_ok=True)
    frames = [(f'{i}_{i:.2f}s.jpg', Rect(20, 10, 30, 60)) for i in range(4)]
    for name, _ in frames:
        (source / name).write_bytes(cv2.imencode('.jpg', np.full((200, 1600, 3), 80, np.uint8))[1].tobytes())
    logs = {1: TrackLog(frames, [(0, '小明'), (1, '小明'), (2, '小红'), (3, '小红')], times=[0, 1, 2, 3]),
            2: TrackLog(frames[:2], [], 2, [0, 1])}
    save_logs(run / 'logs.json', logs, source=str(source))
    write_draft(run / 'truth-draft.toml', draft_identities(logs, .5), logs)
    return IdentityTruth(root, lambda: ['小明', '小红'], .5), run, source


def label(api, segments, **extra):
    return api.label({'run': 'r1', 'track': 1, 'segments': segments,
                      'mtime': api.state('r1')[1]['mtime'], **extra})


def test_state_sorted_coverage_and_two_names(tmp_path):
    api, _, _ = make(tmp_path)
    assert api.runs() == {'runs': ['r1']}
    code, s = api.state('r1')
    assert code == 200 and s['coverage'] == {'done': 0, 'total': 6}
    assert [t['id'] for t in s['tracks']] == [1, 2]
    assert s['tracks'][0]['two_names'] and s['tracks'][0]['guess'] == '小明'
    assert s['tracks'][0]['frames'][2]['tag'] == '小红'


def test_label_splits_and_marks_reviewed(tmp_path):
    api, run, _ = make(tmp_path)
    assert label(api, [[0, '小明'], [2, '陌生人']])[0] == 200
    assert 't1 = [[0, "小明"], [2, "陌生人"]]' in (run / 'truth-draft.toml').read_text(encoding='utf-8')
    assert json.loads((run / 'reviewed.json').read_text()) == {'tracks': [1]}
    assert api.state('r1')[1]['coverage'] == {'done': 4, 'total': 6}


def test_adopt_guess_counts_as_reviewed(tmp_path):
    api, run, _ = make(tmp_path)
    before = (run / 'truth-draft.toml').read_bytes()
    assert label(api, [[0, '小明']])[0] == 200
    assert (run / 'truth-draft.toml').read_bytes() == before
    assert api.state('r1')[1]['coverage']['done'] == 4


def test_hand_edited_line_counts_as_reviewed(tmp_path):
    api, run, _ = make(tmp_path)
    path = run / 'truth-draft.toml'
    path.write_text(path.read_text(encoding='utf-8').replace('t1 = "小明"', 't1 = [[0, "陌生人"], [2, "?"]]'), encoding='utf-8')
    assert api.state('r1')[1]['coverage']['done'] == 2


def test_label_rejects_bad_name_keeps_draft(tmp_path):
    api, run, _ = make(tmp_path)
    before = (run / 'truth-draft.toml').read_bytes()
    assert label(api, [[0, '不是好友']])[0] == 400
    assert (run / 'truth-draft.toml').read_bytes() == before
    assert not (run / 'reviewed.json').exists() and not list(run.glob('*.bak'))


def test_label_stale_mtime_409(tmp_path):
    api, run, _ = make(tmp_path)
    mtime = api.state('r1')[1]['mtime']
    path = run / 'truth-draft.toml'
    path.write_text('t1 = "?"\n', encoding='utf-8')
    assert api.label({'run': 'r1', 'track': 1, 'segments': [[0, '小明']], 'mtime': mtime})[0] == 409
    assert path.read_text(encoding='utf-8') == 't1 = "?"\n'


def test_undo_restores_one_step(tmp_path):
    api, run, _ = make(tmp_path)
    assert label(api, [[0, '陌生人']])[0] == 200
    before = (run / 'truth-draft.toml').read_bytes()
    assert label(api, [[0, '不要']])[0] == 200
    assert api.undo({'run': 'r1', 'mtime': api.state('r1')[1]['mtime']})[0] == 200
    assert (run / 'truth-draft.toml').read_bytes() == before
    assert api.undo({'run': 'r1', 'mtime': api.state('r1')[1]['mtime']})[0] == 400


@pytest.mark.parametrize('friends', [lambda: [], lambda: (_ for _ in ()).throw(OSError('读不到'))])
def test_friends_missing_note(tmp_path, friends):
    api, run, _ = make(tmp_path)
    api.friends = friends
    (run / 'truth-draft.toml').write_text('t1 = "?"\nt2 = "团子"', encoding='utf-8')
    assert '读不到好友名单' in api.state('r1')[1]['friends_note']
    assert label(api, [[0, '小明']])[0] == 400


def test_bad_toml_reports_line(tmp_path):
    api, run, _ = make(tmp_path)
    (run / 'truth-draft.toml').write_text('t1 = [', encoding='utf-8')
    code, s = api.state('r1')
    assert code == 400 and '行' in s['text']


def test_paths_refused(tmp_path):
    api, _, _ = make(tmp_path)
    for run in ('..', 'a/b', '不存在', '.', 'CON'):
        assert api.state(run)[0] == 404
    assert api.crop('r1', 1, '../outside.jpg') is None
    assert api.crop('r1', 2, '3_3.00s.jpg') is None
    assert api.frame('r1', 'no.jpg') is None


def test_crop_and_frame_jpeg(tmp_path):
    api, _, source = make(tmp_path)
    crop = api.crop('r1', 1, '0_0.00s.jpg')
    assert crop.startswith(b'\xff\xd8')
    assert cv2.imdecode(np.frombuffer(crop, np.uint8), 1).shape[0] >= 160
    frame = api.frame('r1', '0_0.00s.jpg')
    assert cv2.imdecode(np.frombuffer(frame, np.uint8), 1).shape[1] == 1280
    shutil.rmtree(source)
    assert api.crop('r1', 1, '0_0.00s.jpg') is None
    assert api.state('r1')[1]['source_missing'] is True


def test_missing_label_and_preserve_comments(tmp_path):
    api, run, _ = make(tmp_path)
    path = run / 'truth-draft.toml'
    path.write_text('# 人工说明\nt1 = "小明"  # 人工备注\n', encoding='utf-8')
    assert api.state('r1')[1]['tracks'][1]['segments'] == [[0, '?']]
    assert label(api, [[0, '陌生人']])[0] == 200
    assert path.read_text(encoding='utf-8').startswith('# 人工说明\nt1 = "陌生人"  # 人工备注\n')


def test_write_failure_keeps_draft_and_reviewed(tmp_path, monkeypatch):
    from skydango.console import identity_truth
    api, run, _ = make(tmp_path)
    assert label(api, [[0, '小明']])[0] == 200
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    def fail(*args):
        raise OSError('写盘失败')
    monkeypatch.setattr(identity_truth.os, 'replace', fail)
    assert label(api, [[0, '陌生人']])[0] == 400
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == before
