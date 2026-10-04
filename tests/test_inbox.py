from skydango.config import load_config


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
