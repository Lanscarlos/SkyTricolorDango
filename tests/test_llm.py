import pytest

from skydango.chat.llm import read_key


def test_read_key_reports_missing_variable(monkeypatch):
    monkeypatch.delenv("SKYDANGO_NO_SUCH_KEY", raising=False)
    monkeypatch.setattr("skydango.chat.llm._user_env", lambda name: "")
    with pytest.raises(RuntimeError, match="SKYDANGO_NO_SUCH_KEY"):
        read_key("SKYDANGO_NO_SUCH_KEY")

