from skydango.chat.commands import CommandRouter, is_command
from skydango.chat.memory import MemoryStore


def router(store, paused=None, status_text="live｜运行中｜待处理0｜限速8/8"):
    paused = paused if paused is not None else []

    return CommandRouter(store, on_pause=lambda p: paused.append(p), status=lambda: status_text), paused


# ---- is_command ----


def test_is_command_recognizes_hash_prefixes():
    assert is_command("#remember 测试")
    assert is_command("＃remember 测试")  # 全角
    assert is_command("  #status  ")  # 去掉首尾空白再判断
    assert not is_command("普通聊天")
    assert not is_command("")


# ---- #friend ----


def test_friend_appends_to_existing_section(tmp_path):
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 本名卡洛\n\n## 番茄炒蛋盖饭\n- 老登\n", encoding="utf-8")
    cmd, _ = router(MemoryStore(tmp_path))
    assert cmd.handle("#friend 懒洋洋大王 很会玩游戏") == "记好啦"
    text = (tmp_path / "friends.md").read_text(encoding="utf-8")
    assert "## 懒洋洋大王\n- 本名卡洛\n- 很会玩游戏\n\n## 番茄炒蛋盖饭\n- 老登\n" == text


def test_friend_creates_new_section_when_missing(tmp_path):
    (tmp_path / "friends.md").write_text("## 懒洋洋大王\n- 本名卡洛\n", encoding="utf-8")
    cmd, _ = router(MemoryStore(tmp_path))
    assert cmd.handle("#friend 新朋友 第一次见") == "记好啦"
    text = (tmp_path / "friends.md").read_text(encoding="utf-8")
    assert text == "## 懒洋洋大王\n- 本名卡洛\n\n## 新朋友\n- 第一次见\n"


def test_friend_creates_file_when_missing(tmp_path):
    cmd, _ = router(MemoryStore(tmp_path / "memory"))
    assert cmd.handle("#friend 新朋友 第一次见") == "记好啦"
    text = (tmp_path / "memory" / "friends.md").read_text(encoding="utf-8")
    assert text == "## 新朋友\n- 第一次见\n"


def test_friend_missing_args():
    cmd, _ = router(MemoryStore("unused"))
    assert cmd.handle("#friend 懒洋洋大王") == "格式不对，是 #friend 昵称 备注内容"
    assert cmd.handle("#friend") == "格式不对，是 #friend 昵称 备注内容"


def test_friend_without_memory_enabled():
    cmd, _ = router(None)
    assert cmd.handle("#friend 懒洋洋大王 很会玩游戏") == "没开记忆功能"


# ---- #remember ----


def test_remember_writes_inbox(tmp_path):
    cmd, _ = router(MemoryStore(tmp_path))
    assert cmd.handle("#remember 卡洛周三要加班") == "记下了"
    assert "卡洛周三要加班" in (tmp_path / "inbox.md").read_text(encoding="utf-8")


def test_remember_missing_content():
    cmd, _ = router(MemoryStore("unused"))
    assert cmd.handle("#remember") == "格式不对，是 #remember 内容"
    assert cmd.handle("#remember   ") == "格式不对，是 #remember 内容"


def test_remember_without_memory_enabled():
    cmd, _ = router(None)
    assert cmd.handle("#remember 卡洛周三要加班") == "没开记忆功能"


# ---- #pause / #resume ----


def test_pause_and_resume_call_back():
    cmd, paused = router(None)
    assert cmd.handle("#pause") == "先歇会儿，不自动理别人了"
    assert paused == [True]
    assert cmd.handle("#resume") == "好，继续陪聊"
    assert paused == [True, False]


# ---- #status ----


def test_status_returns_callback_text():
    cmd, _ = router(None, status_text="dry-run｜运行中｜待处理2｜限速8/8")
    assert cmd.handle("#status") == "dry-run｜运行中｜待处理2｜限速8/8"


# ---- 未知命令 ----


def test_unknown_command():
    cmd, _ = router(None)
    assert cmd.handle("#跳舞") == "没这个命令"
    assert cmd.handle("#") == "没这个命令"
