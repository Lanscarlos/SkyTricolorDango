"""身体接上心情、精力和反思（spec 2026-09-30-inner-phase2 §5 §6 §8）。"""

import time

from conftest import FakeLlm
from test_brain_body import FakeEnv, body, msg

from skydango.config import Config
from skydango.inner.effects import Effects
from skydango.inner.ledger import Card, Ledger
from skydango.inner.mind import Grudge, Mind, Mood, Want
from skydango.inner.reflect import Reflector
from skydango.inner.store import InnerStore

FRIENDS = ["懒洋洋大王", "阿花"]
WALL = time.mktime((2026, 9, 30, 15, 0, 0, 0, 0, -1))  # 下午 3 点：精神，不影响额度


def make(clock, tmp_path, live=False, reflect_reply="{}", **kw):
    env = FakeEnv()
    cfg = Config().inner
    led = Ledger(cfg, lambda: FRIENDS, WALL, store=InnerStore(tmp_path), persist=live,
                 cards={"懒洋洋大王": Card(first_met=0, days=["d1", "d2", "d3"])})
    reflector = Reflector(cfg, FakeLlm(reflect_reply), clock, threaded=False)
    b, dev, reader, events = body(clock, live=live, env=env, ledger=led, mind=Mind(), reflector=reflector,
                                  wall=lambda: WALL, **kw)
    b.friend_names = lambda: FRIENDS
    return b, env, reader, events


def test_status_has_heart_line(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    b.step()
    assert "心里：平常 · 精神" in b.status()


def test_arrive_want_note_first_meeting(clock, tmp_path):
    b, env, _, events = make(clock, tmp_path)
    b.mind.wants = [Want("惦记", "考试考得怎么样", "懒洋洋大王", 0, 9e12)]
    env.near = ["懒洋洋大王"]
    b.step()
    assert [e.text for e in events.drain() if e.kind == "arrive"] == [
        "懒洋洋大王 来到身边（第一次在身边见到）。你惦记着：考试考得怎么样"
    ]


def test_mood_scales_proactive_quota(clock, tmp_path):
    b, env, *_ = make(clock, tmp_path)
    env.near = ["懒洋洋大王"]
    b.step()
    assert b.occasion().left == 2  # 安静 2
    b.mind.mood = Mood("低落", "闷", 0)
    assert b.occasion().left == 1  # 安静 2 × 0.5
    assert b.effects() == Effects(quota=0.5, wander=1.5)


def test_grudge_target_no_bubble_others_yes(clock, tmp_path):  # Review Focus 4
    b, env, reader, _ = make(clock, tmp_path, live=True)
    b.mind.grudge = Grudge("懒洋洋大王", "放鸽子", 0, 9e12)
    env.near = ["懒洋洋大王", "阿花"]
    reader.batches = [[msg("团子在吗")]]
    b.step()
    assert b._bubble_at is None
    clock.advance(40)
    reader.batches = [[msg("团子在吗"), msg("团子！", speaker="阿花")]]
    b.step()
    assert b._bubble_at is not None


def test_bubble_opens_without_grudge(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path, live=True)
    env.near = ["懒洋洋大王"]
    reader.batches = [[msg("团子在吗")]]
    b.step()
    assert b._bubble_at is not None


def test_reflection_result_applied_on_body_thread(clock, tmp_path):
    b, env, reader, _ = make(clock, tmp_path, reflect_reply='{"mood": {"level": "开心", "text": "有人来聊天"}}')
    reader.batches = [[msg("嗨")]]
    b.step()
    clock.advance(1300)
    b.step()
    b.step()
    assert (b.mind.mood.level, b.mind.mood.text) == ("开心", "有人来聊天")
    content = b.reflector.llm.calls[0][1]
    assert "懒洋洋大王：嗨" in content and "精力：精神" in content


def test_reflection_written_only_when_live(clock, tmp_path):
    reply = '{"mood": {"level": "开心", "text": "好"}}'
    for live in (False, True):
        d = tmp_path / str(live)
        b, env, reader, _ = make(clock, d, live=live, reflect_reply=reply)
        reader.batches = [[msg("嗨")]]
        b.step()
        clock.advance(1300)
        b.step()
        b.step()
        assert (d / "mind.json").exists() is live


def test_show_info_has_mood_and_energy(clock, tmp_path):
    class Viewer:
        info = None

        def update(self, frame, now, **kw):
            Viewer.info = kw["info"]

    b, *_ = make(clock, tmp_path, viewer=Viewer())
    b.mind.mood = Mood("开心", "有人来聊天", 0)
    b.step()
    b.step()
    assert Viewer.info["心情"] == "有人来聊天" and Viewer.info["精力"] == "精神"


def test_sleepy_at_night(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    b.wall = lambda: time.mktime((2026, 10, 1, 3, 0, 0, 0, 0, -1))
    b.step()
    assert b.energy_now().level in ("有点累", "困") and b.effects().quota < 1


def test_errors_fall_back_to_neutral(clock, tmp_path):
    b, *_ = make(clock, tmp_path)
    b.step()

    def boom(*a):
        raise RuntimeError("坏了")

    b.mind.line = boom
    assert "心里：" not in b.status()
    b.mind = None
    assert b.effects() == Effects()


def test_no_mind_unchanged(clock):
    b, *_ = body(clock, env=FakeEnv())
    b.step()
    assert "心里" not in b.status() and b.effects() == Effects()


def test_distress_drops_grudge(clock, tmp_path):  # 终审 I2
    b, env, reader, _ = make(clock, tmp_path, live=True)
    b.mind.grudge = Grudge("懒洋洋大王", "放鸽子", 0, 9e12)
    env.near = ["懒洋洋大王"]
    reader.batches = [[msg("团子 我今天真的很难过")]]
    b.step()
    assert b.mind.grudge is None and b._bubble_at is not None
    assert (tmp_path / "mind.json").exists()
    assert "闹别扭" not in b.status()


def test_materials_say_how_long_ago(clock, tmp_path):  # 终审 I4
    b, *_ = make(clock, tmp_path)
    b.mind.updated = WALL - 3 * 3600
    b.step()
    assert "3 小时前想的" in b.reflect_materials(False)


def test_final_materials_cover_whole_session(clock, tmp_path):  # 终审 I5
    b, env, reader, _ = make(clock, tmp_path, reflect_reply='{"mood": {"level": "开心"}}')
    reader.batches = [[msg("早上的话")]]
    b.step()
    clock.advance(1300)
    b.step()  # 后台反思开始：这一段的材料清空
    b.step()
    assert "早上的话" not in b.reflect_materials(False)
    assert "早上的话" in b.reflect_materials(True)


def test_idle_scale_survives_reflex_done(clock):  # 终审 I6：困了闲着的小动作一直更勤，不只第一次
    from test_brain_reflex_body import rx

    b, *_ = rx(clock, idle=["伸懒腰"], idle_min=100, idle_max=100)
    b.effects = lambda: Effects(idle=0.5)
    b.reflexes.stir(clock(), scale=0.5)
    done = b.emotes.done
    clock.advance(51)
    b.step()
    assert done[-1] == ("伸懒腰", True)
    n = len(done)
    clock.advance(51)
    b.step()
    assert len(done) == n + 1  # 下一个也按 ×0.5 的间隔


def test_materials_have_friend_profiles(clock, tmp_path):  # 日记里把女生写成“他”
    b, env, reader, _ = make(clock, tmp_path)
    b.friends_text = lambda: "## 懒洋洋大王\n- 本名卡洛\n\n## 阿花\n- 女生\n"
    reader.batches = [[msg("在吗", "阿花")]]
    b.step()
    text = b.reflect_materials(False)
    assert "阿花：女生" in text and "本名卡洛" not in text  # 只带这段时间相关的好友


def test_final_materials_cover_friends_from_whole_session(clock, tmp_path):  # 日记看整次上线：早先说过话的好友也要带上
    b, env, reader, _ = make(clock, tmp_path, reflect_reply='{"mood": {"level": "开心"}}')
    b.friends_text = lambda: "## 阿花\n- 女生\n"
    reader.batches = [[msg("早上好", "阿花")]]
    b.step()
    clock.advance(1300)
    b.step()  # 后台反思开始：这一段的材料清空
    b.step()
    assert "阿花：女生" in b.reflect_materials(True)


def test_final_materials_recap(clock, tmp_path):   # spec 2026-10-06-brain-compact §7：只给下线那次
    b, env, reader, _ = make(clock, tmp_path)
    assert "前情提要" not in b.reflect_materials(True)          # 默认没有
    b.recap_text = lambda: "小明说下周三考物理"
    assert "## 这次上线早些时候（前情提要，大脑自己写的）\n小明说下周三考物理" in b.reflect_materials(True)
    assert "前情提要" not in b.reflect_materials(False)

    def boom():
        raise RuntimeError("坏了")

    b.recap_text = boom
    assert "前情提要" not in b.reflect_materials(True)           # 出错当空
