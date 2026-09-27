import json
import logging

from conftest import FakeDevice, FakeOcr, scene
from skydango.agent import Agent
from skydango.chat.llm import EchoClient
from skydango.chat.reader import ChatReader
from skydango.chat.responder import Responder
from skydango.chat.sender import ChatSender
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.imageio import imread
from skydango.runlog import RunDir, prune
from skydango.vision.bubbles import Rect


def make_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    return cfg


def test_create_names_dir_and_snapshots_config(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.reply.dry_run = False
    run = RunDir.create(cfg, "live", now=0)
    assert run.path.parent == tmp_path / "runs"
    assert run.path.name.endswith("-live")
    snap = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
    assert snap["reply"]["dry_run"] is False
    # 同一秒再跑一次不会撞名
    again = RunDir.create(cfg, "live", now=0)
    assert again.path != run.path and again.path.exists()


def test_prune_keeps_newest_and_ignores_other_dirs(tmp_path):
    root = tmp_path / "runs"
    for name in ["20260101-000000-dry", "20260102-000000-live", "20260103-000000-dry", "keepme"]:
        (root / name).mkdir(parents=True)
    removed = prune(root, 2)
    assert [p.name for p in removed] == ["20260101-000000-dry"]
    assert sorted(p.name for p in root.iterdir()) == ["20260102-000000-live", "20260103-000000-dry", "keepme"]
    assert prune(root, 0) == []


def test_create_respects_keep(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.run.keep = 2
    root = tmp_path / "runs"
    for name in ["20200101-000000-dry", "20200102-000000-dry"]:
        (root / name).mkdir(parents=True)
    run = RunDir.create(cfg, "dry")
    assert sorted(p.name for p in root.iterdir()) == ["20200102-000000-dry", run.path.name]


def test_save_frame_crops_panel_as_jpg(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.vision.mode = "log"
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 0.5]
    run = RunDir.create(cfg, "dry")
    path = run.save_frame(scene(), [Rect(10, 10, 50, 20)])
    assert path is not None and path.suffix == ".jpg" and path.parent == run.frames
    assert imread(path).shape[:2] == (360, 640)  # 1280×720 的左上四分之一

    cfg.run.save_frames = False
    assert run.save_frame(scene(), []) is None


def test_attach_log_writes_debug_to_file_only(tmp_path):
    cfg = make_cfg(tmp_path)
    run = RunDir.create(cfg, "dry")
    root = logging.getLogger()
    console = logging.StreamHandler()
    root.addHandler(console)
    old_level = root.level
    root.setLevel(logging.INFO)
    try:
        run.attach_log()
        logging.getLogger("skydango.agent").debug("调试细节")
        assert console.level == logging.INFO  # 终端不会被 DEBUG 刷屏
    finally:
        run.close()
        root.removeHandler(console)
        root.setLevel(old_level)
        logging.getLogger("skydango").setLevel(logging.NOTSET)
    assert "调试细节" in (run.path / "agent.log").read_text(encoding="utf-8")


def test_agent_records_frames_rows_and_replies(tmp_path, clock):
    cfg = make_cfg(tmp_path)  # 默认 dry-run、bubble 模式
    run = RunDir.create(cfg, "dry")
    device = FakeDevice([scene([(400, 200, 300, 50)])])
    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    reader = ChatReader(FakeOcr(["你好"] * 5), cfg.vision, cfg.ocr, cfg.chat, self_filter)
    sender = ChatSender(device, cfg.sender, lambda: (1280, 720), sleep=lambda s: None)
    agent = Agent(
        cfg, device, reader, Responder(EchoClient(), cfg.reply), sender, self_filter,
        clock=clock, sleep=lambda s: None, run=run,
    )
    agent.step()
    clock.advance(2)
    assert agent.step() == "【AI】收到：你好"

    assert len(list(run.frames.glob("*.jpg"))) == 1
    lines = (run.path / "replies.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["messages"] == [{"speaker": "", "text": "你好"}]
    assert entry["reply"] == "【AI】收到：你好" and entry["sent"] is False


def test_reader_traces_panel_changes_to_rows_log(tmp_path):
    import numpy as np

    from skydango.vision.ocr import OcrLine

    class SeqOcr:
        def __init__(self, frames):
            self.frames = iter(frames)

        def recognize(self, img):
            return next(self.frames)

    def line(i, text):
        return OcrLine(text, 0.95, Rect(20, 15 + i * 54, 200, 30))

    cfg = make_cfg(tmp_path)
    cfg.vision.mode = "log"
    cfg.vision.log_require_panel = False  # 合成画面里没画面板底部的输入框
    cfg.vision.log_change_pixels = 0
    old = [line(0, "早上好 - 懒洋洋大王")]
    new = old + [line(1, "去霞谷吗 - 懒洋洋大王")]
    reader = ChatReader(SeqOcr([old, old, new, new]), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    reader.trace_path = tmp_path / "rows.log"
    frame = np.full((720, 1280, 3), 50, np.uint8)
    for t in range(4):
        reader.read(frame, float(t))
    text = reader.trace_path.read_text(encoding="utf-8")
    assert "t=0.0" in text and "t=2.0" in text  # 基线、冒出新行
    assert "t=1.0" not in text  # 没变化的那帧不记
    assert "* y=" in text and "去霞谷吗" in text
