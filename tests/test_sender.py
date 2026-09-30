import numpy as np
from conftest import FakeDevice

from skydango.chat.sender import ChatSender
from skydango.config import SenderConfig


def test_sender_opened_flag():
    dev = FakeDevice([np.zeros((720, 1280, 3), np.uint8)])
    s = ChatSender(dev, SenderConfig(open_chat_key=28), lambda: (1280, 720), sleep=lambda s: None)
    assert not s.opened
    s.open()
    assert s.opened
    s.send("好")
    assert not s.opened
    s.open()
    s.cancel()
    assert not s.opened
