import asyncio

import numpy as np
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from test_brain_tools import FakeBody, FakeEyes

from skydango.brain.images import image_block
from skydango.brain.mcp_server import SkyServer, to_mcp
from skydango.brain.tools import TOOL_NAMES, ToolBox


class ImageBody(FakeBody):
    def look(self):
        return [image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "名字"}]


async def talk(url, calls):
    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            results = [await session.call_tool(name, args) for name, args in calls]
            return names, results


def test_to_mcp_converts_text_and_images():
    out = to_mcp([image_block(np.zeros((8, 8, 3), np.uint8)), {"type": "text", "text": "名字"}])
    assert out[1] == "名字" and type(out[0]).__name__ == "Image"
    assert to_mcp("描述") == ["描述"]


def test_tools_over_mcp():
    tb = ToolBox(ImageBody(), FakeEyes())
    server = SkyServer(tb)
    server.start()
    try:
        names, (status, look, emote, bad) = asyncio.run(talk(server.url, [
            ("status", {}), ("look", {"image": True}), ("emote", {"name": "鞠躬"}), ("say", {"text": "太快"}),
        ]))
    finally:
        server.stop()
    assert names == TOOL_NAMES
    assert not status.is_error and [c.text for c in status.content] == ["状态"]
    assert [c.type for c in look.content] == ["image", "text"]
    assert not emote.is_error and tb.body.calls[-1] == ("emote", "鞠躬")
    assert bad.is_error and bad.content[0].text == "说得太快了"


def test_mcp_lists_track():
    tb = ToolBox(FakeBody(), FakeEyes())
    server = SkyServer(tb)
    server.start()
    try:
        names, (track,) = asyncio.run(talk(server.url, [("track", {"name": "小明", "seconds": 10})]))
    finally:
        server.stop()
    assert "track" in names
    assert not track.is_error and tb.body.calls[-1] == ("track", "小明", 10)


def test_mcp_say_jab():
    from test_brain_tools import JabBody

    tb = ToolBox(JabBody(), FakeEyes(), jab=True)
    server = SkyServer(tb)
    server.start()
    try:
        _, (said,) = asyncio.run(talk(server.url, [("say", {"text": "你才废物", "jab": True})]))
    finally:
        server.stop()
    assert not said.is_error and tb.body.calls[-1] == ("say", "你才废物", True)
