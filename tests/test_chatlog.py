import numpy as np

from skydango.chat.reader import ChatReader
from skydango.chat.tracker import SelfFilter
from skydango.config import Config
from skydango.vision.bubbles import Rect
from skydango.vision.chatlog import new_rows, parse_rows
from skydango.vision.ocr import OcrLine

PW, PH = 640, 900  # 面板区域大小
ROW_H, ROW_GAP = 40, 14


def row_y(i):
    return 10 + i * (ROW_H + ROW_GAP)


def panel(light_rows=()):
    """合成面板：深色背景，指定行画成浅色（自己的气泡）。"""
    img = np.full((PH, PW, 3), 50, np.uint8)
    for i in light_rows:
        img[row_y(i) : row_y(i) + ROW_H, 300:630] = (190, 210, 216)
    return img


def line(i, text, x=20, w=200):
    return OcrLine(text, 0.95, Rect(x, row_y(i) + 5, w, ROW_H - 10))


def test_parse_rows_speaker_self_and_masked():
    lines = [
        line(0, "-陌生人"),  # “........ - 陌生人”，省略号 OCR 读不出来
        line(1, "去霞谷吗"),
        line(1, "- 懒洋洋大王", x=230, w=150),  # 同一行被拆成两段
        line(2, "【AI】好呀", x=320, w=280),
        line(3, "…… - 陌生人"),
        line(4, "陌生人", x=60, w=100),  # 省略号和“-”都没读出来
    ]
    rows = parse_rows(panel(light_rows=[2]), lines, self_min_value=150)
    assert [(r.speaker, r.text, r.is_self) for r in rows] == [
        ("陌生人", "", False),
        ("懒洋洋大王", "去霞谷吗", False),
        ("", "【AI】好呀", True),
        ("陌生人", "", False),
        ("陌生人", "", False),
    ]
    assert rows[0].masked and rows[3].masked and rows[4].masked and not rows[1].masked


def test_new_rows_alignment():
    same = lambda a, b: a == b  # noqa: E731
    assert new_rows([], ["a", "b"], same) == []  # 刚开始：已有的都算历史
    assert new_rows(["a", "b", "c"], ["a", "b", "c"], same) == []
    assert new_rows(["a", "b", "c"], ["b", "c", "d", "e"], same) == [2, 3]  # 滚上去一行、来了两条
    assert new_rows(["m", "m", "m"], ["m", "m", "R"], same) == [2]  # 一堆相同的“陌生人”行
    assert new_rows(["a", "b", "c"], ["a", "b", "c", "a"], same) == [3]  # 有人重复说了一句
    assert new_rows(["a", "b"], ["x", "y"], same) == []  # 完全对不上：重新建立基线，不乱回


class SeqOcr:
    """每次 recognize 返回下一组预设的识别结果。"""

    def __init__(self, frames):
        self.frames = list(frames)

    def recognize(self, img):
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


def test_reader_log_mode_reports_only_new_messages_from_others():
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 1.0]
    history = [line(0, "-陌生人"), line(1, "早上好 - 懒洋洋大王")]
    frames = [
        history,
        history,
        # 滚上去一行，新增：自己的气泡 + 好友一句 + 陌生人（被屏蔽）
        [line(0, "早上好 - 懒洋洋大王"), line(1, "【AI】早呀", x=320), line(2, "去霞谷吗-懒洋洋大王"), line(3, "-陌生人")],
    ]
    reader = ChatReader(SeqOcr(frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8, "【AI】"))
    frame = np.full((PH, PW * 2, 3), 50, np.uint8)
    frame[row_y(1) : row_y(1) + ROW_H, 300:630] = (190, 210, 216)  # 第 1 行是自己的浅色气泡

    assert reader.read(frame, 0.0) == []  # 基线
    assert reader.read(frame, 1.0) == []  # 没变化
    fresh = reader.read(frame, 2.0)
    assert [(m.speaker, m.text) for m in fresh] == [("懒洋洋大王", "去霞谷吗")]


def test_format_incoming_includes_speaker():
    from skydango.chat.reader import Message
    from skydango.chat.responder import format_incoming

    text = format_incoming([Message("去霞谷吗", Rect(0, 0, 1, 1), 0.0, "懒洋洋大王"), Message("嗯", Rect(0, 0, 1, 1), 0.0)])
    assert "懒洋洋大王：「去霞谷吗」" in text and "\n「嗯」" in text


def _agent(ocr_frames, shown=False):
    from conftest import FakeDevice

    from skydango.agent import Agent

    cfg = Config()
    cfg.vision.mode = "log"
    device = FakeDevice([np.full((PH, PW * 2, 3), 50, np.uint8)])
    device.shown = shown
    reader = ChatReader(SeqOcr(ocr_frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    return Agent(cfg, device, reader, None, None, None, sleep=lambda s: None), device


def test_ensure_log_open_presses_c_only_when_panel_closed():
    agent, device = _agent([[]])
    agent.ensure_log_open()
    assert device.calls == [("hw_key", 46)]

    agent, device = _agent([[line(0, "-陌生人")]])  # 面板已经开着
    agent.ensure_log_open()
    assert device.calls == []

    agent, device = _agent([[]], shown=True)  # 正在打字：按 C 会变成输入字母
    agent.ensure_log_open()
    assert device.calls == []
