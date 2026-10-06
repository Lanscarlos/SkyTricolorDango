"""OpenAI 兼容大脑的历史压缩（spec 2026-10-06-brain-compact）：配置、compact.py 的纯函数和 InboxWatch。"""


def test_compact_config_defaults_and_toml(tmp_path):
    from skydango.config import CompactConfig, Config, load_config

    assert Config().brain.compact == CompactConfig(enabled=True, budget=64000, keep_turns=6, recap_max=1500, retry=300.0)
    p = tmp_path / "c.toml"
    p.write_text("[brain.compact]\nenabled = false\nbudget = 32000\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.brain.compact.enabled is False and cfg.brain.compact.budget == 32000 and cfg.brain.compact.keep_turns == 6
    assert cfg.sources["brain.compact.budget"] == "config"


def test_compact_settings_fields():
    from skydango.console.settings import KNOWN

    assert {"brain.compact.enabled", "brain.compact.budget", "brain.compact.keep_turns"} <= set(KNOWN)
    assert KNOWN["brain.compact.enabled"].kind == "bool" and KNOWN["brain.compact.budget"].kind == "int"
    assert KNOWN["brain.compact.keep_turns"].kind == "int"


# ---- brain/compact.py ----
STAMP = "[2026年10月6日（周二） 21:03:15] 事件：\n- 小明：在吗"


def test_wake_stamp_and_until():
    from skydango.brain.compact import until_of, wake_stamp

    assert wake_stamp(STAMP) == "2026年10月6日（周二） 21:03:15" and until_of(wake_stamp(STAMP)) == "21:03"
    assert wake_stamp("写一份经过") == "" and until_of("") == ""


def test_compact_request_mentions_cut_and_limit():
    from skydango.brain.compact import compact_request

    q = compact_request(STAMP, keep=6, recap_max=1500)
    assert "[2026年10月6日（周二） 21:03:15] 那条消息之前" in q and "1500" in q and "前情提要" in q
    assert "答应" in q and "难过" in q and "原话" in q        # spec §2.2 要写的几样
    assert "最后 6 轮之前" in compact_request("没有时间戳", keep=6, recap_max=1500)


def test_recap_message_format():
    from skydango.brain.compact import RECAP_ACK, recap_message

    assert recap_message("正文", 2, "21:03", 6) == "（这次上线到现在的前情提要，第 2 次整理，写到 21:03 为止；之后的原话在后面）\n正文"
    assert "写到最后 6 轮之前为止" in recap_message("正文", 1, "", 6)
    assert RECAP_ACK == "（知道了）"


def test_clean_recap():
    from skydango.brain.compact import clean_recap

    assert clean_recap("  好  ", 10) == "好" and clean_recap(None, 10) == ""
    assert clean_recap("一" * 15, 10) == "一" * 15          # 没超 1.5 倍：整句收下
    assert clean_recap("一" * 16, 10) == "一" * 10 + "……"


def test_inbox_watch():
    from skydango.brain.compact import InboxWatch, inbox_note

    text = ["- 旧的一行\n"]
    w = InboxWatch(lambda: text[0])
    assert w.fresh() == []
    text[0] = "- 旧的一行\n- 小明下周考试\n\n"
    assert w.fresh() == ["- 小明下周考试"] and w.fresh() == []
    text[0] = "- 小明下周考试\n"
    assert w.fresh() == []
    text[0] = "- 旧的一行\n- 小明下周考试\n"
    assert w.fresh() == []                                   # 整理挪走再加回：不重复
    assert inbox_note(["- 小明下周考试", "阿花生日"]) == "你刚记下：\n- 小明下周考试\n- 阿花生日"
    assert inbox_note([]) == ""


def test_inbox_watch_read_error():
    from skydango.brain.compact import InboxWatch

    def boom():
        raise OSError("坏了")

    assert InboxWatch(boom).fresh() == []
