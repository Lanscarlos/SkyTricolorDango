"""管理面板读终端起的团子的 agent.log（spec 2026-10-04-console-attach §2「日志抽屉」）。"""

from skydango.console.logtail import LogTail

T = "2026-10-04 21:30:00,123"


def write(path, text, mode="a"):
    with open(path, mode, encoding="utf-8", newline="") as f:
        f.write(text)


def test_logtail_keeps_info_and_above(tmp_path):
    p = tmp_path / "agent.log"
    write(p, f"{T} DEBUG skydango.x: 细节\n{T} INFO skydango.x: 读到了\n{T} WARNING skydango.y: 小心\n")
    assert LogTail(p).read_new() == [f"{T} INFO skydango.x: 读到了", f"{T} WARNING skydango.y: 小心"]


def test_logtail_keeps_continuation_lines_of_kept_records(tmp_path):  # Review Focus 2
    p = tmp_path / "agent.log"
    write(p, f"{T} ERROR skydango.x: 出错\nTraceback (most recent call last):\n  File \"a.py\"\n"
             f"{T} DEBUG skydango.x: 多行\n  第二行\n")
    assert LogTail(p).read_new() == [f"{T} ERROR skydango.x: 出错", "Traceback (most recent call last):", '  File "a.py"']


def test_logtail_reads_incrementally_and_waits_for_full_line(tmp_path):
    p = tmp_path / "agent.log"
    write(p, f"{T} INFO a: 一\n{T} INFO a: 二半")
    tail = LogTail(p)
    assert tail.read_new() == [f"{T} INFO a: 一"]
    assert tail.read_new() == []
    write(p, "截\n")
    assert tail.read_new() == [f"{T} INFO a: 二半截"]


def test_logtail_crlf_lines(tmp_path):
    p = tmp_path / "agent.log"
    write(p, f"{T} INFO a: 一\r\n")
    assert LogTail(p).read_new() == [f"{T} INFO a: 一"]


def test_logtail_restarts_when_file_shrinks(tmp_path):
    p = tmp_path / "agent.log"
    write(p, f"{T} INFO a: 很长的一行很长的一行\n")
    tail = LogTail(p)
    tail.read_new()
    write(p, f"{T} INFO a: 新\n", mode="w")
    assert tail.read_new() == [f"{T} INFO a: 新"]


def test_logtail_missing_file_is_none(tmp_path):
    assert LogTail(tmp_path / "nope.log").read_new() is None


def test_logtail_continuation_of_dropped_record_at_chunk_start_stays_dropped(tmp_path):
    p = tmp_path / "agent.log"
    write(p, f"{T} DEBUG a: 多行\n")
    tail = LogTail(p)
    assert tail.read_new() == []
    write(p, "  续行\n")
    assert tail.read_new() == []
