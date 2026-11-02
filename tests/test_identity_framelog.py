import numpy as np

from skydango.vision.bubbles import Rect
from skydango.vision.identity import FrameIn, TrackIn
from skydango.vision.identity_log import FrameLog, load_frames


def fi(now, *ids):
    return FrameIn(now, 1920, 1080, False, False, False,
                   tuple(TrackIn(i, Rect(100 * i, 400, 90, 220), 0.0) for i in ids))


def img():
    return np.zeros((1080, 1920, 3), np.uint8)


def test_frames_round_trip(tmp_path):
    log = FrameLog(tmp_path / 'identity')
    for t in (0.0, 0.1, 0.2):
        log.record(fi(t, 1, 2), img())
    log.close()
    frames = load_frames(tmp_path / 'identity' / 'frames.pkl')
    assert [f.now for f in frames] == [0.0, 0.1, 0.2]
    assert [t.id for t in frames[0].tracks] == [1, 2]


def test_snapshot_every_second_tags_the_frame(tmp_path):
    log = FrameLog(tmp_path / 'identity', snap_every=1.0)
    tags = [log.record(fi(t, 1), img()).tag for t in (10.0, 10.4, 11.0, 11.5, 12.2)]
    log.close()
    assert tags == ['10.000', '', '11.000', '', '12.200']
    names = sorted(p.name for p in (tmp_path / 'identity' / 'frames').iterdir())
    assert names == ['10.000.jpg', '11.000.jpg', '12.200.jpg']
    # 存盘的帧带着同样的 tag，重放时对得上画面
    assert [f.tag for f in load_frames(tmp_path / 'identity' / 'frames.pkl')] == tags


def test_snapshots_stop_at_cap_but_frames_keep_going(tmp_path):
    log = FrameLog(tmp_path / 'identity', snap_every=1.0, snap_max=2)
    for t in range(5):
        log.record(fi(float(t), 1), img())
    log.close()
    assert len(list((tmp_path / 'identity' / 'frames').iterdir())) == 2
    assert len(load_frames(tmp_path / 'identity' / 'frames.pkl')) == 5


def test_truncated_tail_is_ignored(tmp_path):
    log = FrameLog(tmp_path / 'identity')
    log.record(fi(0.0, 1), img())
    log.record(fi(1.0, 1), img())
    log.close()
    path = tmp_path / 'identity' / 'frames.pkl'
    path.write_bytes(path.read_bytes()[:-5])  # 被强杀：最后一帧只写了一半
    assert [f.now for f in load_frames(path)] == [0.0]


class BrokenFile:
    closed = False

    def write(self, data):
        raise OSError('磁盘满了')

    def close(self):
        pass


def test_write_error_turns_logging_off(tmp_path):
    log = FrameLog(tmp_path / 'identity')
    log._fh.close()
    log._fh = BrokenFile()  # 模拟写盘出错
    out = log.record(fi(0.0, 1), img())
    assert out.now == 0.0 and log.off
    log.record(fi(1.0, 1), img())  # 不再抛


def test_record_after_close_is_skipped_quietly(tmp_path, caplog):
    log = FrameLog(tmp_path / 'identity')
    log.close()
    log.record(fi(0.0, 1), img())
    assert not log.off and not caplog.records
