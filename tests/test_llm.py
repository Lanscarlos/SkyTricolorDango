import pytest

from skydango.chat.llm import make_llm, read_key
from skydango.config import LlmConfig


def test_read_key_reports_missing_variable(monkeypatch):
    monkeypatch.delenv("SKYDANGO_NO_SUCH_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="SKYDANGO_NO_SUCH_KEY"):
        read_key("SKYDANGO_NO_SUCH_KEY")


def test_openai_with_timeout_returns_shorter_copy(monkeypatch):  # 终审 M1：下线反思改走 DeepSeek 时缩短超时、不重试
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    client = make_llm(LlmConfig(timeout=90.0, max_retries=2), api_key="sk-given")
    short = client.with_timeout(35.0, max_retries=0)
    assert short is not client and short.cfg is client.cfg
    assert (short._client.timeout, short._client.max_retries) == (35.0, 0)
    assert (client._client.timeout, client._client.max_retries) == (90.0, 2)  # 原来的不动
    assert client.with_timeout(20.0)._client.max_retries == 2  # 不给就沿用


def test_make_llm_uses_given_key(monkeypatch):
    # 管理面板「测试大模型」用页面上还没保存的 Key：不经过环境变量
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    client = make_llm(LlmConfig(), api_key="sk-given")
    assert client._client.api_key == "sk-given"
