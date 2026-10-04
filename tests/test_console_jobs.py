"""面板任务槽 JobRunner + ConsoleServer 的任务接口（难例收件箱 Task 6）。"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from test_console_server import GOOD, FakeRunner, free_port, make_server, no_registry, request, upstream  # noqa: F401

from skydango.console.jobs import JobRunner


class FakeProc:
    def __init__(self, text: str, code: int = 0, hang: bool = False):
        self.stdout = io.BytesIO(text.encode("utf-8"))
        self.pid, self.code, self.hang, self.killed = 4321, code, hang, False

    def wait(self, timeout=None):
        t0 = time.monotonic()
        while self.hang and not self.killed and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        return -9 if self.killed else self.code

    def poll(self):
        return None if self.hang and not self.killed else self.wait()


def wait_state(jobs, *states):
    t0 = time.monotonic()
    while jobs.status()["state"] not in states:
        assert time.monotonic() - t0 < 5, jobs.status()
        time.sleep(0.01)
    return jobs.status()


def make_jobs(tmp_path, proc):
    def popen(cmd, **kw):
        popen.calls.append((cmd, kw))
        return proc

    popen.calls = []
    kills = []

    def kill(pid):
        kills.append(pid)
        proc.killed = True

    return JobRunner(tmp_path, kill=kill, popen=popen), popen, kills


def test_done_with_progress_and_tail(tmp_path):
    text = "开始\nPROGRESS 1/10 r0\nPROGRESS 3/10 r1\n整理完了\n"
    jobs, popen, _ = make_jobs(tmp_path, FakeProc(text))
    assert jobs.status()["state"] == "idle"
    jobs.start("inbox", ["py", "x"], {"A": "1"})
    st = wait_state(jobs, "done")
    assert st["job"] == "inbox" and st["exit_code"] == 0
    assert st["progress"] == "3/10 r1"
    assert st["tail"][-1] == "整理完了" and len(st["tail"]) == 4
    assert popen.calls[0][0] == ["py", "x"] and popen.calls[0][1]["cwd"] == tmp_path
    log = Path(st["log"])
    assert log.read_text(encoding="utf-8").splitlines()[-1] == "整理完了"
    assert "inbox" in log.name and log.parent == tmp_path / "tmp" / "jobs"


def test_failed_exit_code(tmp_path):
    jobs, _, _ = make_jobs(tmp_path, FakeProc("整理不了：没有模型\n", code=1))
    jobs.start("retrain", ["py"], {})
    st = wait_state(jobs, "failed")
    assert st["exit_code"] == 1 and st["tail"] == ["整理不了：没有模型"]


def test_tail_keeps_last_20(tmp_path):
    jobs, _, _ = make_jobs(tmp_path, FakeProc("".join(f"l{i}\n" for i in range(50))))
    jobs.start("inbox", ["py"], {})
    st = wait_state(jobs, "done")
    assert st["tail"] == [f"l{i}" for i in range(30, 50)]


def test_busy_start_raises_and_unknown_job(tmp_path):
    jobs, _, _ = make_jobs(tmp_path, FakeProc("", hang=True))
    jobs.start("inbox", ["py"], {})
    with pytest.raises(RuntimeError, match="还在跑"):
        jobs.start("retrain", ["py"], {})
    with pytest.raises(ValueError):
        make_jobs(tmp_path, FakeProc(""))[0].start("bogus", ["py"], {})
    jobs.stop()


def test_stop_kills_tree_and_marks_stopped(tmp_path):
    jobs, _, kills = make_jobs(tmp_path, FakeProc("PROGRESS 1/2 a\n", hang=True))
    jobs.start("inbox", ["py"], {})
    jobs.stop()
    st = wait_state(jobs, "stopped")
    assert kills == [4321] and st["state"] == "stopped"


# ---- server ----
class KindRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.kind, self.run_dir = "dango", None

    def status(self):
        return {**super().status(), "kind": self.kind, "run_dir": self.run_dir}


class FakeJobs:
    def __init__(self):
        self.state, self.started, self.stopped, self.job = "idle", [], 0, "inbox"

    def start(self, job, cmd, env):
        if self.state == "running":
            raise RuntimeError("整理还在跑")
        self.started.append((job, cmd, env))
        self.state = "running"

    def stop(self):
        self.stopped += 1
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "job": self.job, "progress": "", "tail": [], "exit_code": None, "log": None}


def make(tmp_path, upstream):
    inbox = tmp_path / "inbox"
    (tmp_path / "config.toml").write_text(f'[inbox]\ndir = "{inbox.as_posix()}"\n', encoding="utf-8")
    srv = make_server(tmp_path, upstream, secrets='[env]\nSKYDANGO_CLAUDE_TOKEN = "tok"\nDEEPSEEK_API_KEY = "k"\n', child_port=free_port())
    srv.runner = srv.fake_runner = KindRunner()
    srv.jobs = FakeJobs()
    srv.inbox = inbox
    srv.curate_poll = 0.01
    return srv


@pytest.fixture
def srv(tmp_path, upstream):
    s = make(tmp_path, upstream)
    yield s
    s.stop()


def post(srv, path, body):
    return request(srv.url + path, json.dumps(body).encode(), GOOD)


RUN = {"brain": True, "live": False, "emotes": True, "duration": 0}


def test_state_has_job_and_inbox_summary(srv):
    from skydango.vision import inbox as ib

    (srv.inbox / "r1").mkdir(parents=True)
    ib.save_frames(srv.inbox, "r1", {"a.jpg": {"boxes": []}, "b.jpg": {"boxes": []}})
    state = request(srv.url + "api/state")[1]
    assert state["job"]["state"] == "idle"
    summary = state["inbox"]
    assert set(summary) >= {"pending", "judge_left", "glance", "edit", "passed_since_train", "retrain_min"}
    assert summary["retrain_min"] == 50


def test_state_inbox_null_when_disabled(tmp_path, upstream):
    (tmp_path / "config.toml").write_text("[inbox]\nenabled = false\n", encoding="utf-8")
    s = make_server(tmp_path, upstream, child_port=free_port())
    s.jobs = FakeJobs()
    try:
        assert request(s.url + "api/state")[1]["inbox"] is None
    finally:
        s.stop()


def test_dango_start_refused_while_job_running(srv):
    srv.jobs.state = "running"
    status, res = post(srv, "api/run/start", RUN)
    assert status == 409 and res["job"]["state"] == "running" and "整理" in res["error"]
    assert srv.fake_runner.started == [] and srv.jobs.stopped == 0


def test_dango_start_with_stop_job_stops_first(srv):
    srv.jobs.state = "running"
    status, res = post(srv, "api/run/start", {**RUN, "stop_job": True})
    assert status == 200, res
    assert srv.jobs.stopped == 1 and len(srv.fake_runner.started) == 1


def test_jobs_start_refused_when_dango_busy(srv):
    srv.fake_runner.state = "running"
    status, res = post(srv, "api/jobs/start", {"job": "inbox"})
    assert status == 409 and srv.jobs.started == []


def test_jobs_start_and_stop(srv):
    status, res = post(srv, "api/jobs/start", {"job": "inbox"})
    assert status == 200 and res["ok"] is True
    job, cmd, env = srv.jobs.started[0]
    assert job == "inbox" and cmd[1:3] == ["-m", "skydango"] and cmd[-3:] == ["perception", "inbox", "process"]
    assert env["SKYDANGO_CLAUDE_TOKEN"] == "tok" and env["PYTHONIOENCODING"] == "utf-8"
    assert post(srv, "api/jobs/start", {"job": "inbox"})[0] == 409  # 已在跑
    assert post(srv, "api/jobs/start", {"job": "bogus"})[0] == 400
    assert post(srv, "api/jobs/stop", {})[0] == 200 and srv.jobs.stopped == 1


def test_stop_run_curate_starts_job_after_exit(srv):
    srv.fake_runner.state = "running"
    assert post(srv, "api/run/stop", {"curate": True}) == (200, {"ok": True})
    time.sleep(0.1)
    assert srv.jobs.started == []  # 团子还没退
    srv.fake_runner.state = "exited"
    t0 = time.monotonic()
    while not srv.jobs.started and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    assert srv.jobs.started and srv.jobs.started[0][1][-3:] == ["perception", "inbox", "process"]


def test_stop_run_without_curate_starts_nothing(srv):
    srv.fake_runner.state = "running"
    post(srv, "api/run/stop", {})
    srv.fake_runner.state = "exited"
    time.sleep(0.1)
    assert srv.jobs.started == []


def test_stop_info(srv):
    from skydango.vision import inbox as ib

    run_dir = srv.inbox.parent / "runs" / "r9"
    (run_dir / "hard").mkdir(parents=True)
    for i in range(3):
        (run_dir / "hard" / f"{i}.jpg").write_bytes(b"x")
    (run_dir / "hard" / "n.txt").write_text("x")
    srv.fake_runner.run_dir = str(run_dir)
    srv.inbox.mkdir()
    (srv.inbox / "_stats.json").write_text(json.dumps({"sec_per_frame": 2.0}), encoding="utf-8")
    res = request(srv.url + "api/inbox/stop-info")[1]
    assert res["hard"] == 3 and res["ask"] is True
    assert res["pending"] == len(ib.pending_runs(srv.inbox))
    assert res["eta_min"] == round(2.0 * 3 / 60, 1)
    assert request(srv.url + "api/inbox/stop-info", headers={"Host": "evil.com"})[0] == 403


def test_stop_info_default_seconds_per_frame(srv):
    run_dir = srv.inbox.parent / "runs" / "r9"
    (run_dir / "hard").mkdir(parents=True)
    for i in range(40):
        (run_dir / "hard" / f"{i}.jpg").write_bytes(b"x")
    srv.fake_runner.run_dir = str(run_dir)
    res = request(srv.url + "api/inbox/stop-info")[1]
    assert res["hard"] == 40 and res["eta_min"] == round(1.5 * 40 / 60, 1)


# ---- 第 1 轮评审修正 ----
def test_retrain_job_command(srv):
    assert post(srv, "api/jobs/start", {"job": "retrain"})[0] == 200
    assert srv.jobs.started[0][1][-2:] == ["perception", "retrain"]


def test_start_run_rechecks_job_under_lock(srv):
    srv.jobs.state = "running"
    srv._job_conflict = lambda body: None  # 锁外的检查已经过时：任务在检查之后才起
    status, res = post(srv, "api/run/start", RUN)
    assert status == 409 and "整理" in res["error"] and srv.fake_runner.started == []


def test_launch_sandbox_rechecks_job_under_lock(srv):
    srv.jobs.state = "running"
    code, res = srv.launch_sandbox("resume")
    assert code == 409 and "整理" in res["error"] and srv.fake_runner.started == []


class PidRunner(KindRunner):
    pid = None

    def status(self):
        return {**super().status(), "pid": self.pid}


def test_curate_gives_up_when_another_dango_ran(tmp_path, upstream):
    s = make(tmp_path, upstream)
    s.runner = s.fake_runner = PidRunner()
    try:
        s.fake_runner.state, s.fake_runner.pid = "running", 100
        post(s, "api/run/stop", {"curate": True})
        s.fake_runner.state, s.fake_runner.pid = "running", 200  # 期间另一个团子起来了
        time.sleep(0.1)
        s.fake_runner.state = "exited"
        time.sleep(0.1)
        assert s.jobs.started == []
        assert s._curate_after is False
    finally:
        s.stop()


def test_job_reader_error_marks_failed(tmp_path):
    class Boom:
        pid = 1

        class stdout:
            def __iter__(self):
                raise OSError("pipe")

        def __init__(self):
            self.stdout = self.stdout()

        def wait(self, timeout=None):
            return 0

    jobs, _, _ = make_jobs(tmp_path, Boom())
    jobs.start("inbox", ["py"], {})
    assert wait_state(jobs, "failed")["state"] == "failed"


def test_job_env_unbuffered_without_mutating(tmp_path):
    jobs, popen, _ = make_jobs(tmp_path, FakeProc(""))
    env = {"A": "1"}
    jobs.start("inbox", ["py"], env)
    assert popen.calls[0][1]["env"]["PYTHONUNBUFFERED"] == "1" and "PYTHONUNBUFFERED" not in env
    wait_state(jobs, "done")


def test_stop_info_ask_false_when_disabled(tmp_path, upstream):
    (tmp_path / "config.toml").write_text("[inbox]\nenabled = false\nask = true\n", encoding="utf-8")
    s = make_server(tmp_path, upstream, child_port=free_port())
    try:
        assert request(s.url + "api/inbox/stop-info")[1]["ask"] is False
    finally:
        s.stop()


def test_job_busy_names_retrain(srv):  # 终审 5：重训在跑时起团子，不再说「整理还没完」
    srv.jobs.state, srv.jobs.job = "running", "retrain"
    status, res = post(srv, "api/run/start", RUN)
    assert status == 409 and "重训还没完" in res["error"] and "整理" not in res["error"]
