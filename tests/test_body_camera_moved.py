"""身体动镜头时告诉感知层（spec 2026-10-01-tracking-relink-motion §2.5）：env.camera_moved(at, kind)。"""

from types import SimpleNamespace

from test_brain_attention_body import AttnEnv, NudgeCam, friend_talking
from test_brain_body import FakeCamera, FakeEnv, FakeLocomotion, body


class MovedEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.moved = []

    def camera_moved(self, at, kind):
        self.moved.append((at, kind))

    def sweep(self, frames, spin):
        return SimpleNamespace(text=lambda: "转了一圈")


def kinds(env):
    return [k for _, k in env.moved]


def test_camera_tool_turn_and_zoom(clock):
    env = MovedEnv()
    b, _, _, _ = body(clock, live=True, camera=FakeCamera(), env=env)
    b.camera_move("left", 1)
    b.camera_move("zoom_in", 1)
    assert kinds(env) == ["turn", "zoom"] and env.moved[0][0] == clock()


def test_move_tool_is_move(clock):
    env = MovedEnv()
    b, _, _, _ = body(clock, live=True, locomotion=FakeLocomotion(), env=env)
    b.move("forward", 1)
    assert kinds(env) == ["move"]


def test_dry_run_move_does_not_report(clock):
    env = MovedEnv()
    b, _, _, _ = body(clock, live=False, locomotion=FakeLocomotion(), env=env)
    b.move("forward", 1)
    assert env.moved == []


def test_reset_and_around(clock):
    env = MovedEnv()
    b, _, _, _ = body(clock, live=True, camera=FakeCamera(), env=env)
    b.camera_reset()
    b.capture_around()
    b.sweep_around()
    assert kinds(env) == ["zoom", "spin", "spin"]


class MovedAttnEnv(AttnEnv):
    def __init__(self):
        super().__init__()
        self.moved = []

    def camera_moved(self, at, kind):
        self.moved.append((at, kind))


def test_attention_nudge_is_turn(clock):
    from test_brain_attention_body import body as _body
    import numpy as np

    env = MovedAttnEnv()
    b, *_ = _body(clock, live=True, env=env, camera=NudgeCam(), panel_mode="auto",
                  frames=[np.full((1080, 1920, 3), 90, np.uint8)])
    b.friend_names = lambda: ["小明", "小红"]
    friend_talking(env, clock())
    b.step()
    assert kinds(env) == ["turn"]


def test_env_without_camera_moved_is_fine(clock):
    b, _, _, _ = body(clock, live=True, camera=FakeCamera(), env=FakeEnv())
    b.camera_move("left", 1)  # 不报错
