import os
import re
import subprocess
import time

from skydango.brain.backstage import changelog, git_runner, read_seen, write_seen

NOW = time.mktime((2026, 10, 1, 20, 0, 0, 0, 0, -1))
DAY = 86400


def _git(repo, *args, when=None):
    env = dict(os.environ)
    if when is not None:
        stamp = f"{int(when)} +0800"
        env.update(GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          env=env, check=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _commit(repo, msg, when=NOW - 3600):
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg, when=when)
    return _git(repo, "rev-parse", "HEAD")


def _titles(lines):
    return [re.sub(r"^- (今天|昨天|\d+ 天前) ", "", line) for line in lines]


def test_changelog_filters(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, "init")
    _commit(repo, "feat(brain): 新本事")
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, "docs: 写文档")
    _commit(repo, "style: 改颜色")
    _commit(repo, "feat(console): 面板")
    _commit(repo, "fix(viewer): 网页")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "fix: 修好了")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge branch 'side'", "side", when=NOW - 3500)
    head = _commit(repo, "perf: 快一点", when=NOW - 3400)
    lines, got_head = changelog(git_runner(repo), None, NOW)
    assert _titles(lines) == ["perf: 快一点", "fix: 修好了", "feat(brain): 新本事"]
    assert got_head == head


def test_changelog_since_seen(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, "feat: 一")
    seen = _commit(repo, "feat: 二")
    _commit(repo, "feat: 三")
    lines, _ = changelog(git_runner(repo), seen, NOW)
    assert _titles(lines) == ["feat: 三"]


def test_changelog_seen_is_head(tmp_path):
    repo = _repo(tmp_path)
    head = _commit(repo, "feat: 一")
    assert changelog(git_runner(repo), head, NOW) == ([], head)


def test_changelog_unknown_seen_falls_back_to_days(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, "feat: 很久以前", when=NOW - 8 * DAY)
    _commit(repo, "feat: 最近", when=NOW - 2 * DAY)
    lines, _ = changelog(git_runner(repo), "0" * 40, NOW)
    assert _titles(lines) == ["feat: 最近"]


def test_changelog_limit(tmp_path):
    repo = _repo(tmp_path)
    for i in range(13):
        _commit(repo, f"feat: 第{i}个", when=NOW - 3600 + i)
    lines, _ = changelog(git_runner(repo), None, NOW, max_n=10)
    assert len(lines) == 11 and lines[-1] == "- 还有 3 条小改动"
    assert _titles(lines[:1]) == ["feat: 第12个"]


def test_changelog_parses_bang_and_chinese(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, "feat(inner)!: 心情：更别扭（试试）")
    _commit(repo, "feat(console)!: 面板大改")
    lines, _ = changelog(git_runner(repo), None, NOW)
    assert _titles(lines) == ["feat(inner)!: 心情：更别扭（试试）"]


def test_changelog_when_labels(tmp_path):
    repo = _repo(tmp_path)
    _commit(repo, "feat: 三天前", when=NOW - 3 * DAY)
    _commit(repo, "feat: 昨天的", when=NOW - DAY)
    _commit(repo, "feat: 今天的", when=NOW - 60)
    lines, _ = changelog(git_runner(repo), None, NOW)
    assert lines == ["- 今天 feat: 今天的", "- 昨天 feat: 昨天的", "- 3 天前 feat: 三天前"]


def test_changelog_runner_failure():
    def boom(args):
        raise subprocess.TimeoutExpired("git", 2)

    assert changelog(boom, None, NOW) == ([], None)


def test_read_seen_bad_json(tmp_path):
    path = tmp_path / "backstage.json"
    assert read_seen(path) is None
    path.write_text("{坏", encoding="utf-8")
    assert read_seen(path) is None
    path.write_text('{"other": 1}', encoding="utf-8")
    assert read_seen(path) is None


def test_write_then_read_seen(tmp_path):
    path = tmp_path / "inner" / "backstage.json"
    write_seen(path, "abc123")
    assert read_seen(path) == "abc123"
    assert [p.name for p in path.parent.iterdir()] == ["backstage.json"]


from skydango.brain.backstage import section  # noqa: E402


def test_section_with_owner():
    s = section("卡洛", "deepseek/deepseek-chat", "claude/haiku", "deepseek/deepseek-chat", [])
    assert s.startswith("## 幕后") and "做出你的人是卡洛" in s and "大模型（deepseek/deepseek-chat）" in s and "（claude/haiku）" in s
    assert "知道团子是 AI" in s and "introspect" in s and "###" not in s
    assert "{" not in s


def test_section_owner_tier_points_to_changes():  # 告诉过一次就不再写进提示词：再问时要去查，别说“没人告诉我”
    assert "introspect(改动)" in section("卡洛", "sonnet", "haiku", "sonnet", [])
    assert "introspect(改动)" not in section("", "sonnet", "haiku", "sonnet", [])


def test_section_without_owner():
    s = section("", "sonnet", "haiku", "sonnet", ["- 昨天 feat: x"])
    assert "做出你的人是" not in s and "（聊天里名字一字不差）" not in s and "{owner}" not in s
    assert "上次以来你被改了什么" in s and "- 昨天 feat: x" in s
    assert "你问" not in s and "跟你聊幕后" not in s


def test_section_changelog_heading_with_owner():
    s = section("卡洛", "sonnet", "haiku", "sonnet", ["- 今天 feat: y"])
    assert "### 卡洛上次以来改了你什么" in s and s.endswith("- 今天 feat: y")


def test_section_keeps_braces_in_changelog():
    assert "- 今天 feat: {x}" in section("卡洛", "s", "h", "s", ["- 今天 feat: {x}"])


def test_section_says_how_to_admit_without_tripping_filter():
    """“我又不是真人”会被 clean_reply 的真人过滤拦下：提示词要让她直接说“我是 AI”。"""
    from skydango.chat.responder import clean_reply

    for owner in ("卡洛", ""):
        s = section(owner, "sonnet", "haiku", "sonnet", [])
        assert "直接说“我是 AI”" in s and "我又不是真人" in s
    assert clean_reply("我是AI啦", 30) and not clean_reply("我又不是真人", 30)


def test_section_cheeky_owner_line():  # 贱兮兮：卡洛是固定的损友
    from skydango.brain.backstage import OWNER_CHEEKY

    s = section("卡洛", "sonnet", "haiku", "sonnet", [], cheeky=True)
    line = OWNER_CHEEKY.replace("<owner>", "卡洛")
    assert line in s and "<owner>" not in s and "# 命令" in line
    assert s.index("introspect(改动)") < s.index(line) < s.index("知情的好友")
    assert section("卡洛", "sonnet", "haiku", "sonnet", []) == section("卡洛", "sonnet", "haiku", "sonnet", [], cheeky=False)
    assert section("", "sonnet", "haiku", "sonnet", [], cheeky=True) == section("", "sonnet", "haiku", "sonnet", [])
