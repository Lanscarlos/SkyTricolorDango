import os

import pytest

from skydango import cli


def fake_console(monkeypatch, events):
    class FakeServer:
        def __init__(self, *a, **k):
            self.orphan = False

        def start(self):
            events.append("start")
            return "http://127.0.0.1:19390/"

        def stop(self):
            events.append("server.stop")

    class FakeRunner:
        def __init__(self, *a, **k):
            pass

        def close(self):
            events.append("runner.close")
            if "impatient" in events:
                raise KeyboardInterrupt  # 等收尾时又按了一次 Ctrl+C

        def kill(self):
            events.append("runner.kill")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("skydango.console.server.ConsoleServer", FakeServer)
    monkeypatch.setattr("skydango.console.runner.Runner", FakeRunner)
    monkeypatch.setattr(cli.time, "sleep", interrupt)


def test_console_starts_server_and_stops_runner_on_ctrl_c(tmp_path, monkeypatch, capsys):
    events = []
    fake_console(monkeypatch, events)
    monkeypatch.chdir(tmp_path)
    cli.main(["console", "--no-browser"])
    assert events == ["start", "runner.close", "server.stop"]
    assert "管理面板：http://127.0.0.1:19390/" in capsys.readouterr().out


def test_console_does_not_load_secrets_into_its_own_environment(tmp_path, monkeypatch, caplog):
    # 面板每次现读 secrets.toml；要是启动时写进了自己的 os.environ，页面上「清除」之后子进程还会继承旧 Key
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / "secrets.toml").write_text('[env]\nDEEPSEEK_API_KEY = "sk-x"\n', encoding="utf-8")
    fake_console(monkeypatch, [])
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("INFO"):
        cli.main(["console", "--no-browser"])
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert "用 secrets.toml" not in caplog.text  # 没注入就别说注入了


def test_console_port_in_use_is_a_clear_error(tmp_path, monkeypatch):
    class Busy:
        def __init__(self, *a, **k):
            self.orphan = False

        def start(self):
            raise OSError("address in use")

    monkeypatch.setattr("skydango.console.server.ConsoleServer", Busy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="console --port"):
        cli.main(["console", "--no-browser", "--port", "19390"])


@pytest.mark.parametrize("name, text", [("console.toml", "[device\n"), ("console.toml", "[foo]\nbar = 1\n"), ("secrets.toml", "[env\n")])
def test_console_starts_with_broken_overlay(tmp_path, monkeypatch, caplog, name, text):  # 终审 Important 4
    (tmp_path / name).write_text(text, encoding="utf-8")
    events = []
    fake_console(monkeypatch, events)
    monkeypatch.chdir(tmp_path)
    cli.main(["console", "--no-browser"])
    assert events[0] == "start" and name in caplog.text


def test_broken_overlay_still_fails_other_commands(tmp_path, monkeypatch):
    (tmp_path / "console.toml").write_text("[device\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["devices"])


def test_second_ctrl_c_force_kills(tmp_path, monkeypatch):
    events = ["impatient"]
    fake_console(monkeypatch, events)
    monkeypatch.chdir(tmp_path)
    cli.main(["console", "--no-browser"])
    assert events[-3:] == ["runner.close", "runner.kill", "server.stop"]
