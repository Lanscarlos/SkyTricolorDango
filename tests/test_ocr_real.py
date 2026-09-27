"""端到端：合成一张带中文气泡的截图，走真实的气泡检测 + RapidOCR。没装 OCR 时跳过。"""

from pathlib import Path

import numpy as np
import pytest

rapid = pytest.importorskip("rapidocr_onnxruntime")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from skydango.chat.reader import ChatReader  # noqa: E402
from skydango.chat.tracker import SelfFilter  # noqa: E402
from skydango.config import Config  # noqa: E402
from skydango.vision.ocr import RapidOcrEngine  # noqa: E402

FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def font():
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, 26)
    pytest.skip("没有中文字体")


def test_bubble_ocr_end_to_end():
    img = Image.new("RGB", (1280, 720), (50, 70, 130))
    d = ImageDraw.Draw(img)
    f = font()
    d.rounded_rectangle((400, 200, 720, 250), 18, fill=(250, 250, 248))
    d.text((420, 208), "你好呀，一起去霞谷吗", font=f, fill=(60, 60, 60))
    d.rounded_rectangle((820, 420, 1020, 470), 18, fill=(250, 250, 248))
    d.text((840, 428), "我在等先祖", font=f, fill=(60, 60, 60))
    d.text((100, 650), "底部按钮文字", font=f, fill=(255, 255, 255))  # 不在气泡里，应忽略
    frame = np.array(img)[:, :, ::-1].copy()

    cfg = Config()
    reader = ChatReader(RapidOcrEngine(), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    texts = [m.text for m in reader.read(frame, 0.0)]
    assert texts == ["你好呀，一起去霞谷吗", "我在等先祖"]
    assert reader.read(frame, 1.0) == []
