"""头顶聊天气泡、系统提示、聊天面板的行别当成"没认出的名字"（docs/progress/2026-10-03-plan.md ① 第 6 项）。

样本来自 runs/*/unknown_names/names.jsonl（10-01 ~ 10-02 攒下的，几乎全是这些）。
"""

import numpy as np
import pytest

from skydango.vision.unknownnames import UnknownNames, looks_like_name

FRIENDS = ["懒洋洋大王", "番茄炒蛋盖饭"]


@pytest.mark.parametrize("text", [
    "看看你今天能懒洋洋大王",  # 气泡和名字读成了一行
    "团子你好 - 懒洋洋大王",  # 聊天面板的行
    "我去试试- 陌生人",
    "辣椒 点亮了您的星星！",  # 系统提示
    "下午6:39",
    "21:05",
    "团子你好冷漠，你都不动一",  # 带标点的气泡
    "能重复吗这个？",
    "怎么啦，谁欺负你啦",
    "回到遇境领取心火",
    "和陌生人一起坐在长凳上",
    "此好友收到的留影",
    "加速前平均延迟 12",
    "团子你知道我在哪里吗呀呀呀",  # 太长，不像名字
    "原来如洋洋大王",  # 和好友名有一大截一样：气泡和名字标签叠在一起读出来的
    "羊洋大王",
    "懒洋",  # 好友名的一截（标签读了一半）
    "懒洋洋懒洋洋",
    "团子你好",  # 带"你 / 我"、语气词结尾：头顶聊天气泡
    "我在吃东西",
    "于回来啦",
    "真的假的",
    "举报",  # 界面上的字
    "好友码",
    "确认退出",
    "连接错误",
    "取消特别关注",
    "重温浮空宗师的回忆",
])
def test_not_a_name(text):
    assert not looks_like_name(text, FRIENDS)


@pytest.mark.parametrize("text", ["小红帽", "Lanscarlos", "阿亚喵喵", "菌子哥", "老头"])
def test_looks_like_a_name(text):
    assert looks_like_name(text, FRIENDS)


def test_add_skips_bubbles(tmp_path):
    u = UnknownNames(tmp_path, lambda: list(FRIENDS), wall=lambda: 0.0)
    u.add("看看你今天能懒洋洋大王", np.zeros((10, 30, 3), np.uint8))
    u.add("辣椒 点亮了您的星星！", np.zeros((10, 30, 3), np.uint8))
    assert u.entries == {}
    u.add("小红帽", np.zeros((10, 30, 3), np.uint8))
    assert list(u.entries) == ["小红帽"]
