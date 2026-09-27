"""命令行入口：skydango <命令> [-c config.toml]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, load_config
from .imageio import imread, imwrite

log = logging.getLogger("skydango")


def _device(cfg: Config):
    from .device.adb import AdbDevice

    dev = AdbDevice(
        cfg.device.serial, cfg.device.adb_path, cfg.device.adb_timeout, cfg.device.ime_id, cfg.device.key_device
    )
    if cfg.device.auto_connect:
        log.debug(dev.connect())
    return dev


def _screen_size_fn(device):
    cache: dict[str, tuple[int, int]] = {}

    def size() -> tuple[int, int]:
        if "wh" not in cache:
            h, w = device.screenshot().shape[:2]
            cache["wh"] = (w, h)
        return cache["wh"]

    return size


def _build_reader(cfg: Config):
    from .chat.reader import ChatReader
    from .chat.tracker import SelfFilter
    from .vision.ocr import make_ocr

    self_filter = SelfFilter(cfg.chat.self_window, cfg.chat.similarity, cfg.reply.disclosure_prefix)
    return ChatReader(make_ocr(cfg.ocr.engine), cfg.vision, cfg.ocr, cfg.chat, self_filter), self_filter


# ---- 命令 ----
def cmd_devices(cfg: Config, args) -> None:
    dev = _device(cfg)
    print("已连接设备:", ", ".join(dev.devices()) or "（无）")
    img = dev.screenshot()
    print(f"{cfg.device.serial} 截图尺寸: {img.shape[1]}x{img.shape[0]}")
    print("当前输入法:", dev.current_ime())


def cmd_shot(cfg: Config, args) -> None:
    from .vision.bubbles import draw_grid

    img = _device(cfg).screenshot()
    if args.grid:
        img = draw_grid(img)
    imwrite(args.output, img)
    print(f"已保存 {args.output}（{img.shape[1]}x{img.shape[0]}）")


def cmd_detect(cfg: Config, args) -> None:
    from .vision.bubbles import annotate

    img = imread(args.image) if args.image else _device(cfg).screenshot()
    reader, _ = _build_reader(cfg)
    detections = reader.detect(img)
    height, width = img.shape[:2]
    if not detections:
        print("没有识别到文字。可以试试调低 vision.bubble.min_value / 调高 max_saturation，或改用 mode = \"roi\"。")
    for i, det in enumerate(detections):
        b = det.box
        print(f"[{i}] ({b.x / width:.3f}, {b.y / height:.3f}) {b.w}x{b.h}  {det.text}")
    if args.output:
        imwrite(args.output, annotate(img, [d.box for d in detections]))
        print(f"标注图已保存到 {args.output}")


def cmd_ime(cfg: Config, args) -> None:
    dev = _device(cfg)
    if args.action == "on":
        dev.enable_adb_keyboard()
    elif args.action == "off":
        dev.reset_ime()
    print("当前输入法:", dev.current_ime())


def cmd_say(cfg: Config, args) -> None:
    from .chat.sender import ChatSender

    dev = _device(cfg)
    ChatSender(dev, cfg.sender, _screen_size_fn(dev)).send(args.text)


def cmd_chat(cfg: Config, args) -> None:
    """不开游戏，在终端里和人设对话，调提示词用。"""
    from .chat.llm import make_llm
    from .chat.reader import Message
    from .chat.responder import Responder
    from .vision.bubbles import Rect

    if args.echo:
        cfg.llm.provider = "echo"
    responder = Responder(make_llm(cfg.llm), cfg.reply)
    print("输入一句话模拟别人的聊天气泡（多句用 | 分隔），空行退出。")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        msgs = [Message(t.strip(), Rect(0, 0, 1, 1), 0.0) for t in line.split("|") if t.strip()]
        reply = responder.reply(msgs)
        print("（不回复）" if reply is None else cfg.reply.disclosure_prefix + reply)


def cmd_run(cfg: Config, args) -> None:
    from .agent import Agent
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    if args.live:
        cfg.reply.dry_run = False
    if args.echo:
        cfg.llm.provider = "echo"
    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    responder = Responder(make_llm(cfg.llm), cfg.reply)
    sender = ChatSender(dev, cfg.sender, _screen_size_fn(dev))
    agent = Agent(cfg, dev, reader, responder, sender, self_filter)
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n已停止")


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except AttributeError:
                pass

    parser = argparse.ArgumentParser(prog="skydango", description="光遇视觉 Agent（MuMu 模拟器）")
    parser.add_argument("-c", "--config", default="config.toml", help="配置文件路径（默认 config.toml，不存在则用默认值）")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="检查 adb 连接、截图尺寸、当前输入法").set_defaults(func=cmd_devices)

    p = sub.add_parser("shot", help="截一张图（--grid 叠加坐标网格，用来标定按钮位置）")
    p.add_argument("-o", "--output", default="shot.png")
    p.add_argument("--grid", action="store_true")
    p.set_defaults(func=cmd_shot)

    p = sub.add_parser("detect", help="对截图 / 当前画面做一次气泡检测和 OCR，用来调参")
    p.add_argument("image", nargs="?", help="图片路径；不填则实时截屏")
    p.add_argument("-o", "--output", default="detect.png", help="标注图输出路径")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("ime", help="切换 ADBKeyboard 输入法")
    p.add_argument("action", choices=["on", "off", "status"])
    p.set_defaults(func=cmd_ime)

    p = sub.add_parser("say", help="往游戏里发一句话，测试发送流程")
    p.add_argument("text")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("chat", help="在终端里和人设对话，调提示词")
    p.add_argument("--echo", action="store_true", help="不调模型，原样回显")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("run", help="启动 Agent（默认 dry-run）")
    p.add_argument("--live", action="store_true", help="真的发送消息")
    p.add_argument("--echo", action="store_true", help="不调模型，原样回显（联调用）")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg_path = Path(args.config)
    if not cfg_path.exists() and args.config != "config.toml":
        parser.error(f"找不到配置文件 {cfg_path}")
    try:
        cfg = load_config(cfg_path if cfg_path.exists() else None)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.func(cfg, args)
    except (RuntimeError, ImportError, ValueError, FileNotFoundError) as exc:
        if args.verbose:
            raise
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
