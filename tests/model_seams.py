"""测试用：把 Registry 里的 Claude Code 换成假进程（spec 2026-10-05-model-providers 之前的测试世界）。

老测试的大脑 / 记忆 / 反思都跑在假 claude 进程上、回复走 echo；新默认（大脑 DeepSeek）要显式写进 cfg.models。
"""

OLD_WORLD = {
    "brain": {"main": "claude/sonnet", "backup": ""},
    "memory": {"main": "claude/sonnet", "backup": "echo/echo"},
    "reflect": {"main": "claude/sonnet", "backup": "echo/echo"},
    "reply": {"main": "echo/echo"},
}


def fake_claude(monkeypatch, base, env=None):
    """claude_base 直接给 (base, env)：不查令牌、不找 claude 命令。"""
    monkeypatch.setattr("skydango.models.claude_code.claude_base",
                        lambda provider, environ=None: (list(base), dict(env or {})))


def old_world(monkeypatch, cfg, base, env=None):
    fake_claude(monkeypatch, base, env)
    for use, table in OLD_WORLD.items():
        cfg.models[use] = dict(table)
    return cfg
