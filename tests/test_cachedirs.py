import os

from skydango import cachedirs


def run_pth(root, cwd, monkeypatch, env=None):
    """在 cwd 下执行生成的 .pth 第二行（Python 启动时 site 就是这样 exec 它的），返回改过的环境变量。"""
    env = {} if env is None else env
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.chdir(cwd)
    exec(cachedirs.pth_text(root).splitlines()[1])
    return env


def test_first_line_adds_pydeps(tmp_path):
    assert cachedirs.pth_text(tmp_path).splitlines()[0] == str(tmp_path / ".pydeps")


def test_inside_repo_points_caches_into_repo(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    env = run_pth(tmp_path, tmp_path / "src", monkeypatch)
    assert env["PIP_CACHE_DIR"] == str(tmp_path / ".cache" / "pip")
    assert env["YOLO_CONFIG_DIR"] == str(tmp_path / ".cache" / "ultralytics")
    assert set(env) == set(cachedirs.DIRS)


def test_outside_repo_leaves_env_alone(tmp_path, monkeypatch):
    repo, other = tmp_path / "repo", tmp_path / "repo2"  # 前缀相同的别的目录也不算
    repo.mkdir()
    other.mkdir()
    assert run_pth(repo, other, monkeypatch) == {}


def test_does_not_override_existing_values(tmp_path, monkeypatch):
    env = run_pth(tmp_path, tmp_path, monkeypatch, {"TORCH_HOME": "E:/torch"})
    assert env["TORCH_HOME"] == "E:/torch"


def test_install_writes_user_site(tmp_path, monkeypatch):
    monkeypatch.setattr(cachedirs.site, "getusersitepackages", lambda: str(tmp_path / "site"))
    path = cachedirs.install(tmp_path / "repo")
    assert path == tmp_path / "site" / cachedirs.PTH_NAME
    assert path.read_text(encoding="utf-8") == cachedirs.pth_text(tmp_path / "repo")
