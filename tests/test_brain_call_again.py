"""自动喊一声太勤（docs/progress/2026-10-03-plan.md ① 第 5 项）。

10-02 晚 22:02~22:04：好友一直站在稍远处没走，每次喊完标签亮 → 认回 → 三四十秒后轨迹断了、接回的"像他"30 秒没被标签证实摘掉
→ 又判走开 → 又喊（80 秒喊了 3 声）。喊一声认回来的好友，[call] auto_again 秒内不再为他自动喊。
"""

from test_brain_call_body import call_body, left, pressed

from skydango.vision.people import CallSeen, Seen


def found(b, env, clock, friends):
    env.results[b.last_call.at] = CallSeen(b.last_call.at, friends, unnamed=0, ended=True)
    b._watch_call(clock())


def test_friend_found_by_auto_call_is_not_called_for_again_soon(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    found(b, env, clock, {"小明": Seen("右边", "远")})
    clock.advance(40)
    left(b, clock)  # 40 秒后又"走开"了（标签淡掉、轨迹断了）
    b._watch_call(clock())
    assert pressed(device) == 1
    clock.advance(b.cfg.call.auto_again)
    left(b, clock)
    b._watch_call(clock())
    assert pressed(device) == 2  # 过了 auto_again：真走开了还是会喊


def test_friend_seen_offscreen_also_counts_as_found(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    found(b, env, clock, {"小明": Seen("左边", None, on_screen=False)})
    clock.advance(40)
    left(b, clock)
    b._watch_call(clock())
    assert pressed(device) == 1


def test_friend_not_found_can_be_called_for_again(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    found(b, env, clock, {})  # 没认出来
    clock.advance(40)
    left(b, clock)
    b._watch_call(clock())
    assert pressed(device) == 2


def test_other_friends_still_get_called_for(clock):
    b, device, env, _ = call_body(clock)
    left(b, clock)
    b._watch_call(clock())
    found(b, env, clock, {"小明": Seen("右边", "远")})
    clock.advance(40)
    left(b, clock, name="小红")
    b._watch_call(clock())
    assert pressed(device) == 2
