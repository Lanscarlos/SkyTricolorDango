from skydango import cli
from skydango.config import InboxConfig, load_config
from skydango.vision.inbox import collect, collect_all, frame_name, pending_runs, split_of


def test_split_of_whole_run_same_side():
    assert split_of("20261004-192711-live-brain", 5) in ("train", "val")
    vals = [r for r in (f"run{i}" for i in range(100)) if split_of(r, 5) == "val"]
    assert 10 <= len(vals) <= 30 and all(split_of(r, 5) == "val" for r in vals)


def test_frame_name():
    assert frame_name("r1", "192007_attrs_disagree.jpg") == "r1_192007_attrs_disagree"


def test_collect_copies_and_is_idempotent(tmp_path):
    run = tmp_path / "runs" / "20261004-192711-live-brain"
    (run / "hard").mkdir(parents=True)
    for n in ("192007_attrs_disagree.jpg", "192102_low_conf.jpg"):
        (run / "hard" / n).write_bytes(b"x")
    (run / "hard.jsonl").write_text('{"file": "192007_attrs_disagree.jpg"}\n', encoding="utf-8")
    inbox = tmp_path / "inbox"
    assert collect(run, inbox) == 2 and collect(run, inbox) == 0
    assert sorted(p.name for p in (inbox / run.name / "raw").iterdir()) == ["192007_attrs_disagree.jpg", "192102_low_conf.jpg"]
    assert (inbox / run.name / "hard.jsonl").is_file() and pending_runs(inbox) == [run.name]
    assert len((inbox / "_index.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_collect_nothing(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    assert collect(run, tmp_path / "inbox") == 0 and not (tmp_path / "inbox" / "r").exists()


def test_collect_all_and_pending(tmp_path):
    runs = tmp_path / "runs"
    for name, n in (("a", 1), ("b", 0), ("c", 2)):
        (runs / name / "hard").mkdir(parents=True)
        for i in range(n):
            (runs / name / "hard" / f"{i}.jpg").write_bytes(b"x")
    inbox = tmp_path / "inbox"
    assert collect_all(runs, inbox) == ["a", "c"]
    assert collect_all(runs, inbox) == []
    with (inbox / "_index.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"run": "a", "processed_at": "2026-10-04T20:00:00"}\n')
    assert pending_runs(inbox) == ["c"]


class _Env:
    unknown = None
    catalog = None

    def __init__(self, folder, saved=1):
        class Hard:
            pass

        self.hardcases = Hard()
        self.hardcases.saved = saved
        self.hardcases.folder = folder

    def stop(self):
        pass


def _fake_run(tmp_path):
    run = tmp_path / "runs" / "r1"
    (run / "hard").mkdir(parents=True)
    (run / "hard" / "1.jpg").write_bytes(b"x")
    return run


def test_stop_scene_collects(tmp_path, capsys):
    run = _fake_run(tmp_path)
    cfg = InboxConfig(dir=str(tmp_path / "inbox"))
    cli._stop_scene(_Env(run / "hard"), cfg)
    assert (tmp_path / "inbox" / "r1" / "raw" / "1.jpg").is_file()
    assert "datasets/inbox" in capsys.readouterr().out


def test_stop_scene_disabled_or_none_does_not_collect(tmp_path):
    run = _fake_run(tmp_path)
    cli._stop_scene(_Env(run / "hard"), InboxConfig(enabled=False, dir=str(tmp_path / "inbox")))
    cli._stop_scene(_Env(run / "hard"))
    assert not (tmp_path / "inbox").exists()


def test_stop_scene_never_raises(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("盘满了")

    monkeypatch.setattr("skydango.vision.inbox.collect", boom)
    cli._stop_scene(_Env(run / "hard"), InboxConfig(dir=str(tmp_path / "inbox")))


def test_inbox_defaults():
    cfg = load_config(None)
    assert cfg.inbox.enabled and cfg.inbox.ask and cfg.inbox.dir == "datasets/inbox"
    assert (cfg.inbox.agree, cfg.inbox.dup_diff, cfg.inbox.dup_gap, cfg.inbox.val_every, cfg.inbox.retrain_min) == (0.9, 6.0, 5.0, 5, 50)
    assert (cfg.retrain.base, cfg.retrain.imgsz, cfg.retrain.epochs, cfg.retrain.batch, cfg.retrain.workers) == ("models/yolo11n.pt", 960, 120, 16, 2)


def test_inbox_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[inbox]\nask = false\nretrain_min = 30\n[retrain]\nepochs = 3\n", encoding="utf-8")
    cfg = load_config(p)
    assert not cfg.inbox.ask and cfg.inbox.retrain_min == 30 and cfg.retrain.epochs == 3


def test_cli_inbox_collect(tmp_path, capsys):
    run = tmp_path / "runs" / "r1"
    (run / "hard").mkdir(parents=True)
    (run / "hard" / "1.jpg").write_bytes(b"x")
    (tmp_path / "config.toml").write_text(f'[inbox]\ndir = "{(tmp_path / "inbox").as_posix()}"\n', encoding="utf-8")
    cli.main(["-c", str(tmp_path / "config.toml"), "perception", "inbox", "collect", str(tmp_path / "runs")])
    assert "收了 1 次运行 / 共 1 张" in capsys.readouterr().out
    assert (tmp_path / "inbox" / "r1" / "raw" / "1.jpg").is_file()
