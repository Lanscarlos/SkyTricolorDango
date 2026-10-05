from skydango.brain.claude import ClaudeError
from skydango.models.errors import ModelError, down_kind
from skydango.models.gate import ProviderGates


def test_gate_per_provider():
    g = ProviderGates()
    assert g.ok("deepseek") and g.ok("claude")
    assert g.trip("deepseek", "limit", "402") is True
    assert g.trip("deepseek", "auth", "x") is False          # 已经关了，原因不改
    assert not g.ok("deepseek") and g.ok("claude")
    assert "用完" in g.reason("deepseek") and g.closed() == {"deepseek": g.reason("deepseek")}


def test_claude_error_is_model_error():
    e = ClaudeError("x", limit=True)
    assert isinstance(e, ModelError) and e.down == "limit" and e.limit and e.provider == "claude"
    assert down_kind(ClaudeError("x", auth=True)) == "auth"
    assert down_kind(RuntimeError("x")) is None and down_kind(ModelError("x")) is None
