import json
from pathlib import Path

import numpy as np
import pytest

from skydango.imageio import imwrite, imread
from skydango.vision.bubbles import Rect
from skydango.vision.identity_eval import (
    TrackLog, draft_identities, write_draft, write_sheet, apply_draft, save_logs,
    label_of, score_frames, replay_board, sweep, report_md,
)

R = Rect(10, 20, 30, 60)


@pytest.mark.parametrize('data,want', [({'dango': R}, '团子'), ({'name': '小明'}, '小明'),
    ({'maybe': '小明'}, '小明'), ({'unsure': ('小明', 0)}, '拿不准'), ({'stranger': True}, '陌生人'), ({}, '未定')])
def test_label_of(data, want):
    from skydango.vision.track import Track
    assert label_of(Track(1, 'player', R, 0.9, 0, 0, data=data)) == want


def test_legacy_self_is_labelled_as_dango():
    from skydango.vision.track import Track
    assert label_of(Track(1, 'self', R, 0.9, 0, 0)) == '团子'


def test_score_frames_hand_calculated_and_unmatched_denominator():
    b = Rect(100, 20, 30, 60)
    truth = {'a': [(R, '小明'), (b, '团子')], 'b': [(R, '小明'), (b, '陌生人')]}
    pred = {'a': [(R, '陌生人'), (b, '团子')], 'b': [(R, '小明'), (b, '陌生人')]}
    scores = score_frames(pred, truth)
    assert scores['friend_as_stranger'] == 0.5
    assert scores['dango_found'] == 1.0 and scores['other_as_dango'] == 0
    assert scores['friend_right'] == 0.5 and scores['friend_wrong'] == 0
    missing = score_frames({}, truth)
    assert missing['n']['total'] == 4 and missing['n']['matched'] == 0 and missing['dango_found'] == 0


def evidence_frames():
    from skydango.vision.identity import FrameIn, TrackIn, Scores
    return [FrameIn(t, 1920, 1080, False, False, False,
        (TrackIn(1, Rect(400, 400, 90, 220), 0, scores=Scores({'小明': 0.86}, {'小明': 0.86}, None)),
         TrackIn(2, Rect(960, 400, 90, 220), 0, self_score=1.0)), tag=f'f{t}') for t in (0, 1, 2, 3)]


def test_replay_is_deterministic_and_sweep_ranks_lower_match():
    from skydango.config import IdentityConfig, AppearanceConfig
    frames = evidence_frames()
    cfg, acfg = IdentityConfig(min_samples=1), AppearanceConfig(match=0.90)
    first = replay_board(frames, cfg, acfg)
    assert replay_board(frames, cfg, acfg) == first
    truth = {f.tag: [(t.box, who) for t, who in zip(f.tracks, ['小明', '团子'])] for f in frames}
    rows = sweep(frames, truth, cfg, acfg, {'match': [0.90, 0.85]})
    assert rows[0]['params'] == {'match': 0.85} and rows[0]['passed']


def test_replay_simulates_both_call_outcomes():
    from skydango.config import IdentityConfig, AppearanceConfig
    frames = evidence_frames()
    cfg, acfg = IdentityConfig(min_samples=1), AppearanceConfig()
    shown, dark = {}, {}
    a = replay_board(frames, cfg, acfg, call_mode='shown', stats=shown)
    b = replay_board(frames, cfg, acfg, call_mode='dark', stats=dark)
    assert a['f2'][0][1] == '小明' and b['f2'][0][1] == '陌生人'
    assert shown['calls'] == dark['calls'] == 1


def test_report_contains_comparison_calls_and_thresholds():
    scores = score_frames({'a': [(R, '小明')]}, {'a': [(R, '小明')]})
    text = report_md(scores, scores, [], {'source': 'demo', 'fps': 6.5, 'calls': {'shown': {'calls': 1}, 'dark': {'calls': 2}}})
    assert '好友被判陌生人' in text and '团子认出' in text and '喊了都亮' in text and '门槛' in text


def logs():
    return {
        1: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R), ('2_0.60s.jpg', R)],
                    [(0, '小明'), (0.3, '小明'), (0.6, '小明')], 0, [0, 0.3, 0.6]),
        2: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R), ('2_0.60s.jpg', R)],
                    [(0, '小明')], 0, [0, 0.3, 0.6]),
        3: TrackLog([('0_0.00s.jpg', R), ('1_0.30s.jpg', R)], [], 2, [0, 0.3]),
    }


@pytest.mark.parametrize('tail,want', [('陌生人', ['小明', '小明', '陌生人', '陌生人']),
                                     ('?', ['小明', '小明']), ('不要', ['小明', '小明'])])
def test_apply_segments_cut_on_frame(tmp_path, tail, want):
    seen, draft, out = (tmp_path / n for n in ('logs.json', 'draft.toml', 'truth.jsonl'))
    log = TrackLog([(f'{i}_{i:.2f}s.jpg', R) for i in range(4)], times=[0, 1, 2, 3])
    save_logs(seen, {1: log})
    draft.write_text(f't1 = [[0, "小明"], [2.0, "{tail}"]]\n', encoding='utf-8')
    assert apply_draft(draft, seen, out, friends=['小明']) == len(want)
    assert [json.loads(r)['who'] for r in out.read_text(encoding='utf-8').splitlines()] == want


@pytest.mark.parametrize('value', ['[[0, "小明"], [0, "陌生人"]]', '[[1.5, "小明"]]',
    '[["小明", 0]]', '[]', '[[true, "小明"]]', '[[nan, "小明"]]', '[[inf, "小明"]]'])
def test_apply_segment_errors_have_line_and_keep_old(tmp_path, value):
    seen, draft, out = (tmp_path / n for n in ('logs.json', 'draft.toml', 'truth.jsonl'))
    save_logs(seen, logs())
    draft.write_text('# 注释\nt1 = ' + value, encoding='utf-8')
    out.write_text('保留', encoding='utf-8')
    with pytest.raises(ValueError, match='第 2 行'):
        apply_draft(draft, seen, out, friends=['小明'])
    assert out.read_text(encoding='utf-8') == '保留'


def test_frame_times_fallback_to_filename():
    from skydango.vision.identity_eval import frame_times
    log = TrackLog(frames=[('0003_000.46s.jpg', R), ('0005_000.77s.jpg', R)])
    assert frame_times(log) == [0.46, 0.77]
    log.frames.append(('坏帧名.jpg', R))
    with pytest.raises(ValueError):
        frame_times(log)


def test_render_round_trip_odd_names(tmp_path):
    from skydango.vision.identity_eval import render_draft, read_draft
    labels = {1: [(0.0, '小"明\\'), (2.5, '陌生人')], 2: [(0.0, '团 子')]}
    path = tmp_path / 'draft.toml'
    path.write_text(render_draft(labels, logs()), encoding='utf-8')
    assert read_draft(path)[0] == labels
    assert 't1 = [[0, ' in path.read_text(encoding='utf-8')


def test_write_draft_output_unchanged(tmp_path):
    path = tmp_path / 'draft.toml'
    write_draft(path, draft_identities(logs(), 0.5), logs())
    assert path.read_text(encoding='utf-8') == (
        '# 初判不是标准答案：请对着 sheet.jpg 改成好友名 / 团子 / 陌生人 / 不要 / ?。\n'
        't1 = "小明"  # 出现 0.0~0.6 s，3 帧\n'
        't2 = "?"  # 出现 0.0~0.6 s，3 帧\n'
        't3 = "团子"  # 出现 0.0~0.3 s，2 帧\n')


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


def test_eval_cli_requires_truth_before_models(tmp_path, monkeypatch):
    from skydango import cli
    monkeypatch.setattr(cli, '_perception', lambda *args: pytest.fail('没有答案时不该加载模型'))
    with pytest.raises(SystemExit, match='先跑'):
        cli.main(['perception', 'identity-eval', str(tmp_path)])


def test_eval_cli_writes_report_summary_and_replay_frames(tmp_path, monkeypatch):
    import pickle
    from skydango import cli
    from test_perception import FRIENDS, player, tag
    from test_perception_appearance import paint, PINK
    from test_perception_identity import make

    source = tmp_path / 'recording'
    source.mkdir()
    truth = []
    for n, t in enumerate((0, 0.3, 0.6)):
        name = f'{n}_{t:.2f}s.jpg'
        imwrite(source / name, paint((player(1000), PINK)))
        b = player(1000).box
        truth.append({'frame': name, 'box': [b.x, b.y, b.w, b.h], 'who': FRIENDS[0]})
    (source / 'truth.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in truth), encoding='utf-8')

    def watcher(cfg, args):
        w, det = make(identity=cfg.identity.enabled)
        det.frames = [[player(1000), tag(990, 110)]]
        return det, w

    monkeypatch.setattr(cli, '_perception', watcher)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'report'
    cli.main(['perception', 'identity-eval', str(source), '-o', str(out)])
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['baseline']['friend_right'] == 1.0
    assert summary['current']['n']['friend'] == 3
    with (out / 'frames.pkl').open('rb') as fh:
        frames = pickle.load(fh)
    assert len(frames) == 3 and [f.tag for f in frames] == [r['frame'] for r in truth]
    assert '喊了都不亮' in (out / 'report.md').read_text(encoding='utf-8')
