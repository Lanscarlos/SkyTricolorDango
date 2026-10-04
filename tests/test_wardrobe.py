import numpy as np

from skydango.brain.claude import ClaudeError
from skydango.config import AppearanceConfig
from skydango.vision.wardrobe import ME, FRIEND, STRANGER, Wardrobe, parse_outfit, wardrobe_command

C = np.zeros((40, 20, 3), np.uint8)
F = np.zeros(8, np.float32)


def make(replies, **kw):
    done = []
    replies = list(replies)

    def describe(content):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    cfg = AppearanceConfig(**kw)
    w = Wardrobe(cfg, describe, lambda k, who, desc, feat: done.append((k, who, desc)))
    return w, done


def test_parse_outfit():
    assert parse_outfit('好的 {"desc": "白色樱花发型、粉色长斗篷", "clear": true}') == ("白色樱花发型、粉色长斗篷", True)
    assert parse_outfit('{"desc": "", "clear": true}') == ("", False)
    assert parse_outfit("看不清") is None
    assert len(parse_outfit('{"desc": "' + "长" * 40 + '"}')[0]) == 25
    assert parse_outfit('{"desc": " 红 斗篷 "}') == ("红斗篷", True)


def test_command():
    cmd = wardrobe_command(["claude"], "haiku")
    assert cmd[0] == "claude" and cmd[cmd.index("--model") + 1] == "haiku"
    assert "--system-prompt" in cmd


def test_priority_and_dedup():
    w, done = make(['{"desc":"自己"}', '{"desc":"小明"}'])
    assert w.request("stranger", "陌生人A", STRANGER, C, F, 0) and w.request("me", "", ME, C, F, 0)
    assert not w.request("me", "", ME, C, F, 0)
    w.tick(0)
    assert done == [("me", "", "自己")]


def test_budget_per_hour():
    w, done = make(['{"desc":"x"}'] * 3, describe_max=2)
    for who in "abc":
        w.request("stranger", who, STRANGER, C, F, 0)
    w.tick(0)
    w.tick(1)
    assert not w.tick(2) and w.calls == 2
    assert w.tick(3601)


def test_unclear_retries_after_wait_then_gives_up():
    w, done = make(['{"desc":"","clear":false}'] * 3, retry_after=60)
    for t in (0, 60, 120):
        assert w.request("friend", "小明", FRIEND, C, F, t)
        w.tick(t)
    assert not w.request("friend", "小明", FRIEND, C, F, 999)
    w.reset("friend", "小明")
    assert w.request("friend", "小明", FRIEND, C, F, 999)


def test_limit_pauses_describing():
    w, done = make([ClaudeError("额度", limit=True), '{"desc":"x"}'], quota_wait=600)
    w.request("me", "", ME, C, F, 0)
    w.tick(0)
    assert not w.tick(599) and w.tick(600) and done[-1][2] == "x"


def test_failure_counts_as_retry():
    w, done = make([ClaudeError("挂了")])
    w.request("me", "", ME, C, F, 0)
    w.tick(0)
    assert not w.request("me", "", ME, C, F, 30)


def test_wardrobe_stops_queueing_when_unavailable():  # Claude 总闸关了：不排、不描述
    calls = []
    w = Wardrobe(AppearanceConfig(), lambda content: calls.append(content) or '{"desc":"x"}', lambda *a: None,
                 available=lambda: False)
    assert not w.request("me", "", ME, C, F, 0)
    assert not w.tick(0) and calls == [] and w.calls == 0


def test_wardrobe_closed_after_queueing_keeps_quiet():  # 排进去之后才关闸：也不再描述
    calls = []
    is_open = [True]
    w = Wardrobe(AppearanceConfig(), lambda content: calls.append(content) or '{"desc":"x"}', lambda *a: None,
                 available=lambda: is_open[0])
    assert w.request("me", "", ME, C, F, 0)
    is_open[0] = False
    assert not w.tick(0) and calls == []
