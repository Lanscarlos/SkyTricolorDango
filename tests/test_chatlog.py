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
        line(2, "【AI】好呀", x=320, w=290),
        line(3, "…… - 陌生人"),
        line(4, "陌生人", x=60, w=100),  # 省略号和“-”都没读出来
        line(5, "懒洋洋大王", x=370, w=160),  # 面板关着时 3D 场景里的名字标签：不贴边，不算
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
    cfg.vision.log_require_panel = False  # 合成画面里没画面板底部的输入框
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 1.0]
    cfg.vision.log_change_pixels = 0  # 同一张画面配不同的 OCR 结果：每次都识别
    history = [line(0, "-陌生人"), line(1, "早上好 - 懒洋洋大王")]
    frames = [
        history,
        history,
        # 滚上去一行，新增：自己的气泡 + 好友一句 + 陌生人（被屏蔽）
        [line(0, "早上好 - 懒洋洋大王"), line(1, "【AI】早呀", x=320, w=290), line(2, "去霞谷吗-懒洋洋大王"), line(3, "-陌生人")],
    ]
    reader = ChatReader(SeqOcr(frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8, "【AI】"))
    frame = np.full((PH, PW * 2, 3), 50, np.uint8)
    frame[row_y(1) : row_y(1) + ROW_H, 300:630] = (190, 210, 216)  # 第 1 行是自己的浅色气泡

    assert reader.read(frame, 0.0) == []  # 基线
    assert reader.read(frame, 1.0) == []  # 没变化
    assert reader.read(frame, 2.0) == []  # 冒出新行：先等一帧确认
    fresh = reader.read(frame, 2.2)
    assert [(m.speaker, m.text) for m in fresh] == [("懒洋洋大王", "去霞谷吗")]


def test_format_incoming_includes_speaker():
    from skydango.chat.reader import Message
    from skydango.chat.responder import format_incoming

    text = format_incoming([Message("去霞谷吗", Rect(0, 0, 1, 1), 0.0, "懒洋洋大王"), Message("嗯", Rect(0, 0, 1, 1), 0.0)])
    assert "懒洋洋大王：「去霞谷吗」" in text and "\n「嗯」" in text


def screen_with_pill(top, bg=50):
    """1920×1080 截图：左侧面板底部画一个“聊天……”胶囊输入框（亮色描边、深色内部），上沿在 top。"""
    img = np.full((1080, 1920, 3), bg, np.uint8)
    img[top : top + 3, 15:625] = (230, 230, 230)
    img[top + 3 : top + 60, 15:625] = 45
    img[top + 57 : top + 60, 15:625] = (230, 230, 230)
    return img


def test_find_input_top_follows_panel_layout():
    from skydango.vision.chatlog import find_input_top

    assert find_input_top(screen_with_pill(1006)[:, :643]) == 1006  # 输入框关着：面板往下伸
    assert find_input_top(screen_with_pill(932)[:, :643]) == 932  # 输入框开着：整体上移
    assert find_input_top(np.full((1080, 643, 3), 50, np.uint8)) is None
    assert find_input_top(np.full((1080, 643, 3), 235, np.uint8)) is None  # 雪地之类的亮背景不算


def screen_with_dim_pill(top, bg=65):
    """面板开着、胶囊变暗的样子（2026-09-28 录像实测）：外圈 2 px 深色边，里面 2 px 灰线（V≈84），
    内部深色（V≈26），往下 60 px 还有一条灰线。上沿灰线的位置和白色描边时一样。"""
    img = np.full((1080, 1920, 3), bg, np.uint8)
    img[top - 2 : top + 64, 12:628] = 25
    img[top : top + 2, 15:625] = 84
    img[top + 60 : top + 61, 15:625] = 85
    img[top + 20 : top + 40, 80:150:9] = 110  # “聊天……”的灰字
    return img


def test_find_input_top_finds_dimmed_pill():
    """实测：面板开着一会儿后胶囊变成暗灰描边（V≈84，不到白色阈值 170），整段录像都判成“面板没开”，
    面板里的“内容 - 好友名”被弱标注当成了头顶的名字标签。"""
    from skydango.vision.chatlog import find_input_top

    assert find_input_top(screen_with_dim_pill(1006)[:, :643]) == 1006
    assert find_input_top(screen_with_dim_pill(1006, bg=120)[:, :643]) == 1006  # 背后场景亮一些
    fading = np.full((1080, 643, 3), 104, np.uint8)  # 从白色变暗的过渡帧（实测 V：外面 104、边 95、线 159、内部 88）
    fading[1004:1070, 12:628] = 95
    fading[1008:1064, 12:628] = 88
    fading[1006:1008, 15:625] = 159
    fading[1066:1067, 15:625] = 159
    assert find_input_top(fading) in (1006, 1007)
    no_bottom = screen_with_dim_pill(1006)
    no_bottom[1066:1067] = 25  # 只有一条灰线、没有下沿：不算（外面的背景也比阈值亮，不能当下沿）
    assert find_input_top(no_bottom[:, :643]) is None


def test_reader_reads_rows_down_to_the_input_box():
    """输入框关着时最新一行在 y≈955，在默认 log_roi（到 0.855）外面，也要读到。"""
    cfg = Config()
    cfg.vision.mode = "log"
    seen = []

    class SizeOcr:
        def recognize(self, img):
            seen.append(img.shape[:2])
            return []

    reader = ChatReader(SizeOcr(), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    reader.log_rows(screen_with_pill(1006))
    reader.log_rows(screen_with_pill(932))
    reader.log_rows(np.full((1080, 1920, 3), 50, np.uint8))  # 找不到输入框：用 log_roi
    assert [h for h, _ in seen] == [1006 - 4, 932 - 4, round(0.855 * 1080)]


def test_parse_rows_merges_wrapped_long_message():
    """长消息在同一个深色框里折成几行（实测：行与行几乎贴着，上一行写满），要拼回一条。"""
    img = panel(light_rows=[8])
    lines = [
        line(0, "-陌生人"),
        OcrLine("啊哈哈哈哈你在干什么，好搞笑啊我跟你说今", 0.99, Rect(13, 80, 566, 37)),
        OcrLine("天上班的时候有个超级好玩的瓜，想不想听听", 0.99, Rect(13, 115, 567, 36)),
        OcrLine("看？-懒洋洋大王", 0.96, Rect(11, 152, 228, 39)),
        OcrLine("-陌生人", 0.95, Rect(20, 222, 100, 40)),  # 下一条：隔了 30 px，不合并
        line(8, "【AI】好呀", x=320, w=290),
    ]
    rows = parse_rows(img, lines, self_min_value=150)
    assert [(r.speaker, r.text, r.is_self) for r in rows] == [
        ("陌生人", "", False),
        ("懒洋洋大王", "啊哈哈哈哈你在干什么，好搞笑啊我跟你说今天上班的时候有个超级好玩的瓜，想不想听听看？", False),
        ("陌生人", "", False),
        ("", "【AI】好呀", True),
    ]


def test_parse_rows_does_not_merge_short_rows_that_touch():
    """上一行没写满就不是折行，哪怕挨得很近。"""
    lines = [
        OcrLine("早 - 懒洋洋大王", 0.99, Rect(13, 80, 200, 37)),
        OcrLine("晚安 - 卡洛", 0.99, Rect(13, 119, 180, 37)),
    ]
    rows = parse_rows(panel(), lines, self_min_value=150)
    assert [(r.speaker, r.text) for r in rows] == [("懒洋洋大王", "早"), ("卡洛", "晚安")]


def test_reader_skips_ocr_when_panel_text_unchanged():
    """截图很便宜（MuMu 原生约 9 ms），OCR 贵（约 0.4 s）：面板文字没变就不识别。"""
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.vision.log_require_panel = False  # 合成画面里没画面板底部的输入框
    calls = []

    class CountOcr:
        def recognize(self, img):
            calls.append(1)
            return [line(0, "早 - 懒洋洋大王")]

    reader = ChatReader(CountOcr(), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    frame = np.full((1080, 1920, 3), 50, np.uint8)
    frame[100:130, 20:300] = 230  # 一行白字
    reader.read(frame, 0.0)
    reader.read(frame.copy(), 0.2)
    assert len(calls) == 1  # 没变：跳过

    noisy = frame.copy()
    noisy[500:504, 400:404] = 255  # 背景里飘过一个小光点：不算变化
    reader.read(noisy, 0.4)
    assert len(calls) == 1

    changed = frame.copy()
    changed[140:170, 20:200] = 230  # 底部多了一行
    reader.read(changed, 0.6)
    assert len(calls) == 2

    reader.read(changed, 0.6 + cfg.vision.log_max_skip + 0.1)  # 太久没识别：强制识别一次
    assert len(calls) == 3


def test_find_input_top_ignores_long_self_bubble():
    """实测漏读：自己的长回复是一条几乎横贯面板的浅色气泡，它的下沿也是“亮线、下面变暗”，
    落到面板下半部分时被当成了输入框，读取范围被截在气泡下沿，下面的新消息全读不到。"""
    from skydango.vision.chatlog import find_input_top

    for bubble_y in (700, 760, 850, 900):
        img = screen_with_pill(1006)
        img[bubble_y : bubble_y + 45, 75:630] = (200, 225, 232)  # 浅色气泡
        img[bubble_y + 15 : bubble_y + 30, 90:600:12] = 40  # 气泡里的深色字
        assert find_input_top(img[:, :643]) == 1006, bubble_y
    img = screen_with_pill(932)
    img[830:920, 75:630] = (200, 225, 232)  # 折成两行的长气泡，紧挨着打开状态的输入框
    assert find_input_top(img[:, :643]) == 932


def test_split_speaker_accepts_fullwidth_dash():
    """实测 OCR 会把“ - ”读成全角“－”或长破折号。"""
    lines = [line(0, "马上你就知道了－ 懒洋洋大王"), line(1, "你这个号是别人的了—番茄炒蛋盖饭")]
    rows = parse_rows(panel(), lines, self_min_value=150)
    assert [(r.speaker, r.text) for r in rows] == [("懒洋洋大王", "马上你就知道了"), ("番茄炒蛋盖饭", "你这个号是别人的了")]


def test_reader_ignores_glitch_frame_with_few_rows():
    """实测：界面滚动 / 重绘的瞬间截到一帧只读出一行“陌生人”，拿它当基准会把已经读过的几行又当成新消息。"""
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.vision.log_require_panel = False  # 合成画面里没画面板底部的输入框
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 1.0]
    cfg.vision.log_change_pixels = 0
    # 最后一条“陌生人”后面还有已经读过的行：坏帧的“陌生人”会对齐到它，后面的全被当成新的
    full = [line(0, "-陌生人"), line(1, "在哪 - 卡洛"), line(2, "-陌生人"), line(3, "啥意思 - 卡洛"), line(4, "号是别人的了 - 卡洛")]
    frames = [full, [line(10, "-陌生人")], full + [line(5, "你猜 - 卡洛")]]
    reader = ChatReader(SeqOcr(frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    frame = np.full((PH, PW * 2, 3), 50, np.uint8)
    assert reader.read(frame, 0.0) == []
    assert reader.read(frame, 0.2) == []  # 坏帧：忽略，不改基准
    assert reader.read(frame, 0.4) == []  # 新行先等一帧确认
    assert [m.text for m in reader.read(frame, 0.6)] == ["你猜"]


def test_reader_accepts_shrunk_panel_after_it_persists():
    """面板真的变短了（不是一闪而过的坏帧）：连续几帧都这样就接受为新基准，不回复旧消息。"""
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.vision.log_require_panel = False  # 合成画面里没画面板底部的输入框
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 1.0]
    cfg.vision.log_change_pixels = 0
    full = [line(i, f"第{i}句 - 卡洛") for i in range(6)]
    short = [line(0, "新的 - 卡洛")]
    frames = [full, short, short, short, short + [line(1, "再来 - 卡洛")]]
    reader = ChatReader(SeqOcr(frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    frame = np.full((PH, PW * 2, 3), 50, np.uint8)
    got = [m.text for t in range(6) for m in reader.read(frame, t * 0.2)]  # 最后多一帧确认新行
    assert got == ["再来"]


def _agent(frames, shown=False, clock=None):
    from conftest import FakeDevice

    from skydango.agent import Agent

    cfg = Config()
    cfg.vision.mode = "log"
    cfg.panel.mode = "always"  # 面板常开时的重开逻辑（2026-09-30 起默认 auto）
    device = FakeDevice(frames)
    device.shown = shown
    reader = ChatReader(SeqOcr([[]]), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    kwargs = {"clock": clock} if clock else {}
    return Agent(cfg, device, reader, None, None, None, sleep=lambda s: None, **kwargs), device


def test_ensure_log_open_uses_input_box_as_panel_marker():
    """实测：面板关着时，3D 场景里的名字标签被读成一行，以为面板开着，没按 C → 整轮读不到消息。
    现在看面板底部的“聊天……”输入框：它是面板的一部分，面板关了就没有。"""
    closed = np.full((1080, 1920, 3), 50, np.uint8)
    closed[400:440, 300:600] = 230  # 场景里的名字标签
    agent, device = _agent([closed, screen_with_pill(1006)])
    assert agent.ensure_log_open() is True
    assert device.calls == [("hw_key", 46)]

    agent, device = _agent([screen_with_pill(1006)])  # 面板已经开着
    assert agent.ensure_log_open() is True and device.calls == []

    agent, device = _agent([closed], shown=True)  # 正在打字：按 C 会变成输入字母
    assert agent.ensure_log_open() is False and device.calls == []


def test_reader_skips_ocr_while_panel_closed():
    """面板关着时不跑 OCR：既不会误读场景里的字，也不会因为 3D 画面一直在动而让 OCR 把 CPU 跑满。"""
    cfg = Config()
    cfg.vision.mode = "log"
    calls = []

    class CountOcr:
        def recognize(self, img):
            calls.append(1)
            return []

    reader = ChatReader(CountOcr(), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    closed = np.full((1080, 1920, 3), 50, np.uint8)
    for t in range(5):
        closed[400:440, 300:600] = 100 + 30 * t  # 场景在动
        assert reader.read(closed.copy(), float(t)) == []
    assert calls == [] and reader.panel_closed_since == 0.0
    reader.read(screen_with_pill(1006), 5.0)
    assert calls == [1] and reader.panel_closed_since is None


def test_agent_reopens_closed_panel_with_cooldown():
    t = [0.0]
    closed = np.full((1080, 1920, 3), 50, np.uint8)
    agent, device = _agent([closed], clock=lambda: t[0])
    for step in range(80):  # 每步 0.5 s，一共 40 s，面板一直关着
        t[0] = step * 0.5
        agent.step()
    presses = [c for c in device.calls if c == ("hw_key", 46)]
    assert len(presses) == 4  # 关了 5 s 后按一次；按了没用，之后每 10 s（冷却）再按一次：5、15、25、35 s


def test_new_row_is_confirmed_on_next_frame_to_get_full_speaker():
    """实测：新消息淡入时名字比内容晚出现，第一帧读成“?：嗯应该是正太”，交给模型的就没有说话人。
    新行第一次出现先不报，强制下一帧再识别一次，用完整的版本。"""
    cfg = Config()
    cfg.vision.mode = "log"
    cfg.vision.log_roi = [0.0, 0.0, 0.5, 1.0]
    cfg.vision.log_require_panel = False
    base = [line(0, "-陌生人"), line(1, "老登你喜欢什么发型 - 懒洋洋大王")]
    frames = [base, base + [line(2, "嗯应该是正太 -")], base + [line(2, "嗯应该是正太 - 懒洋洋大王")]]
    reader = ChatReader(SeqOcr(frames), cfg.vision, cfg.ocr, cfg.chat, SelfFilter(60, 0.8))
    before = np.full((PH, PW * 2, 3), 50, np.uint8)
    after = before.copy()
    for k in range(0, 30, 6):  # 底部冒出新的一行白字：2 px 粗的笔画，底色还是暗的
        after[row_y(2) + k : row_y(2) + k + 2, 20:300] = 230
    assert reader.read(before, 0.0) == []
    assert reader.read(after, 0.2) == []  # 刚冒出来：先不报
    fresh = reader.read(after, 0.4)  # 画面没变，但在等确认：照样再识别一次
    assert [(m.speaker, m.text) for m in fresh] == [("懒洋洋大王", "嗯应该是正太")]


def test_split_speaker_falls_back_to_middle_dot():
    rows = parse_rows(panel(), [line(0, "嗯应该是正太·懒洋洋大王"), line(1, "哈·哈 - 卡洛")], self_min_value=150)
    assert [(r.speaker, r.text) for r in rows] == [("懒洋洋大王", "嗯应该是正太"), ("卡洛", "哈·哈")]


def test_reader_settling_is_false_initially():
    from skydango.chat.reader import ChatReader
    from skydango.chat.tracker import SelfFilter
    from skydango.config import ChatConfig, OcrConfig, VisionConfig
    from conftest import FakeOcr

    reader = ChatReader(FakeOcr([]), VisionConfig(mode="log"), OcrConfig(), ChatConfig(), SelfFilter(3.0, 0.8, ""))
    assert reader.settling is False
    reader._confirming = True
    assert reader.settling is True
