from types import SimpleNamespace

import numpy as np
import openai
import pytest

try:  # openai 3.x 换成了 httpx2
    import httpx2 as httpx
except ImportError:
    import httpx

from skydango.brain.images import image_block
from skydango.chat import llm as chat_llm
from skydango.models.config import ProviderConfig
from skydango.models.errors import ModelError
from skydango.models.openai_compat import OpenAIBackend, classify

DS = ProviderConfig("deepseek", "openai", models=("deepseek-chat",), base_url="https://x", key_env="DEEPSEEK_API_KEY")


class FakeClient:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="好"))],
                               usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, prompt_cache_hit_tokens=3))


def _status_error(status, code=None):
    body = {"error": {"code": code, "message": "x"}} if code else None
    resp = httpx.Response(status, request=httpx.Request("POST", "http://x"))
    return openai.APIStatusError("boom", response=resp, body=body)


def test_text_message():
    c = FakeClient()
    out = OpenAIBackend(DS, "deepseek-chat", temperature=0.5, max_tokens=99, client=c).message(
        "S", "你好", history=[{"role": "user", "content": "前"}, {"role": "assistant", "content": "答"}])
    kw = c.calls[0]
    assert kw["messages"][0] == {"role": "system", "content": "S"}
    assert kw["messages"][-1] == {"role": "user", "content": "你好"} and len(kw["messages"]) == 4
    assert (kw["model"], kw["temperature"], kw["max_tokens"]) == ("deepseek-chat", 0.5, 99)
    assert out == {"result": "好", "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3},
                   "provider": "deepseek", "model": "deepseek-chat"}


def test_image_blocks_become_image_url():
    c = FakeClient()
    OpenAIBackend(DS, "deepseek-chat", temperature=0.8, max_tokens=10, client=c).message(
        "S", [image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "看"}])
    user = c.calls[0]["messages"][-1]["content"]
    assert user[0]["type"] == "image_url" and user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user[1] == {"type": "text", "text": "看"}


def test_classify():
    assert classify(_status_error(401)) == "auth" and classify(_status_error(403)) == "auth"
    assert classify(_status_error(402)) == "limit"
    assert classify(_status_error(429, "insufficient_quota")) == "limit"
    assert classify(_status_error(429, "rate_limit_exceeded")) is None
    assert classify(_status_error(500)) is None and classify(TimeoutError()) is None


def test_sdk_error_wrapped():
    with pytest.raises(ModelError) as e:
        OpenAIBackend(DS, "deepseek-chat", temperature=0.8, max_tokens=10, client=FakeClient(_status_error(402))).message("S", "x")
    assert e.value.down == "limit" and e.value.provider == "deepseek"


def test_missing_key_is_auth_down(monkeypatch):
    monkeypatch.setattr(chat_llm, "_user_env", lambda name: "")
    with pytest.raises(ModelError) as e:
        OpenAIBackend(DS, "deepseek-chat", temperature=0.8, max_tokens=10, environ={}).message("S", "x")
    assert e.value.down == "auth" and e.value.provider == "deepseek"


# 10-21 沙盒：deepseek-flash 默认先思考，max_tokens 小的用处（随手记 200、整理笔记 1500）被思考用光、正文空
DS_OFFICIAL = ProviderConfig("deepseek", "openai", models=("deepseek-flash",), base_url="https://api.deepseek.com",
                             key_env="DEEPSEEK_API_KEY")


def test_thinking_off_on_deepseek_sends_disabled():
    c = FakeClient()
    OpenAIBackend(DS_OFFICIAL, "deepseek-flash", temperature=0.8, max_tokens=200, thinking=False, client=c).message("S", "x")
    assert c.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_thinking_default_or_other_provider_sends_nothing():
    c = FakeClient()
    OpenAIBackend(DS_OFFICIAL, "deepseek-flash", temperature=0.8, max_tokens=200, client=c).message("S", "x")
    OpenAIBackend(DS, "deepseek-flash", temperature=0.8, max_tokens=200, thinking=False, client=c).message("S", "x")
    assert "extra_body" not in c.calls[0] and "extra_body" not in c.calls[1]


def test_empty_content_cut_by_length_warns(caplog):
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")],
                           usage=SimpleNamespace(prompt_tokens=5, completion_tokens=200, prompt_cache_hit_tokens=0))
    c = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    with caplog.at_level("WARNING"):
        out = OpenAIBackend(DS_OFFICIAL, "deepseek-flash", temperature=0.8, max_tokens=200, client=c).message("S", "x")
    assert out["result"] == ""
    assert any("max_tokens" in r.getMessage() and "deepseek-flash" in r.getMessage() for r in caplog.records)
