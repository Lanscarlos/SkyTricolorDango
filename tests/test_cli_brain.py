from conftest import FakeDevice, scene
from test_brain_body import FakeReader

from skydango import cli
from skydango.brain import client as client_mod
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.runlog import RunDir


def test_run_brain_wires_body_and_brain(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, cfg):
            pass

        def create(self, system, messages, tools=None, max_tokens=None):
            calls.append(messages)
            return {"content": [{"type": "text", "text": "看看"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2}}

    cfg = Config()
    cfg.run.dir = str(tmp_path / "runs")
    cfg.llm.provider = "echo"
    cfg.env.enabled = False
    cfg.reply.memory_dir = ""
    monkeypatch.setattr(client_mod, "BrainClient", FakeClient)
    monkeypatch.setattr(cli, "_device", lambda cfg: FakeDevice([scene()]))
    monkeypatch.setattr(cli, "_build_reader", lambda cfg: (FakeReader(), SelfFilter(60, 0.8, "")))
    run = RunDir.create(cfg, "dry-brain")
    cli._run_brain(cfg, run, no_emotes=True, duration=1.5)
    assert calls, "上线后应该马上醒一次（先看一眼周围）"
    assert calls[0][0]["content"][1]["type"] == "image"
    assert (run.path / "brain.jsonl").exists()


def test_run_and_look_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda cfg, args: seen.update(brain=args.brain))
    monkeypatch.setattr(cli, "cmd_look", lambda cfg, args: seen.update(prompt=args.prompt))
    cli.main(["run", "--brain"])
    cli.main(["look", "--prompt", "q.txt"])
    assert seen == {"brain": True, "prompt": "q.txt"}
