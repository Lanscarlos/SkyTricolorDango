"""命令行入口：skydango <命令> [-c config.toml]"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Config, load_config
from .imageio import imread, imwrite
from .runlog import RunDir

log = logging.getLogger("skydango")


def _capture(cfg: Config):
    mode = cfg.device.capture
    if mode == "adb":
        return None
    if mode not in ("auto", "mumu"):
        raise ValueError(f"不支持的 device.capture: {mode}")
    from .device.mumu import MumuCapture, MumuError, find_mumu

    found = find_mumu(cfg.device.adb_path)
    if found is None:
        if mode == "mumu":
            raise RuntimeError("device.capture = \"mumu\" 但没找到 MuMu 的截图接口，device.adb_path 要指向 MuMu 自带的 adb")
        return None
    try:
        return MumuCapture(*found, instance=cfg.device.mumu_instance)
    except MumuError:
        if mode == "mumu":
            raise
        log.warning("MuMu 截图接口加载失败，改用 adb screencap", exc_info=True)
        return None


def _device(cfg: Config):
    from .device.adb import AdbDevice

    dev = AdbDevice(
        cfg.device.serial,
        cfg.device.adb_path,
        cfg.device.adb_timeout,
        cfg.device.ime_id,
        cfg.device.key_device,
        capture=_capture(cfg),
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
    return ChatReader(make_ocr(cfg.ocr.engine, cfg.ocr.threads), cfg.vision, cfg.ocr, cfg.chat, self_filter), self_filter


# ---- 命令 ----
def cmd_devices(cfg: Config, args) -> None:
    dev = _device(cfg)
    print("已连接设备:", ", ".join(dev.devices()) or "（无）")
    import time

    t = time.perf_counter()
    img = dev.screenshot()
    ms = (time.perf_counter() - t) * 1000
    how = "MuMu 原生" if dev.capture is not None and not dev._capture_failed else "adb screencap"
    print(f"{cfg.device.serial} 截图尺寸: {img.shape[1]}x{img.shape[0]}（{how}，{ms:.0f} ms）")
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
    print("输入一句话模拟别人的聊天（“名字：内容”带上说话人，多句用 | 分隔），空行退出。")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        msgs = []
        for part in (t.strip() for t in line.split("|")):
            speaker, sep, text = part.partition("：")  # “名字：内容”模拟面板里带说话人的消息
            if part:
                msgs.append(Message(text.strip(), Rect(0, 0, 1, 1), 0.0, speaker.strip()) if sep else Message(part, Rect(0, 0, 1, 1), 0.0))
        reply = responder.reply(msgs)
        print("（不回复）" if reply is None else cfg.reply.disclosure_prefix + reply)


def _wheel(cfg: Config):
    from .game.wheel import EmoteLibrary, Wheel

    return Wheel(_device(cfg), cfg.wheel, EmoteLibrary(cfg.wheel.library_dir))


def cmd_emotes(cfg: Config, args) -> None:
    wheel = _wheel(cfg)
    if args.action == "scan":
        out = args.output or str(Path(cfg.wheel.library_dir) / "scan")
        paths = wheel.scan_list(out)
        print(f"截了 {len(paths)} 个图标到 {out}，总览图 {Path(out) / '_sheet.png'}")
        print(f"把想用的复制到 {cfg.wheel.library_dir}/ 并改名成动作名，例如 {cfg.wheel.library_dir}/鞠躬.png")
    elif args.action == "wheel":
        names = wheel.library.names
        print(f"图标库（{cfg.wheel.library_dir}）: {', '.join(names) or '（空）'}")
        for slot, (name, score) in wheel.refresh().items():
            lock = " [锁定]" if slot in cfg.wheel.locked_slots else ""
            print(f"  {slot}: {name or '?'}  ({score:.2f}){lock}")
    elif args.action == "set":
        wheel.assign(args.slot, args.name, force=args.force)
        print(f"格子 {args.slot} → {args.name}")
    elif args.action == "do":
        slot = wheel.perform(args.name)
        print(f"做了「{args.name}」（格子 {slot}）")


PROFILE_TEMPLATE = """{persona}

## 外貌
- （比如：樱花发型、小个子、爱穿什么）

## 说话习惯
- （口头禅、常用的语气词、不喜欢说的话）
"""

FRIENDS_TEMPLATE = """# 好友资料：每个人一节，标题写游戏里显示的昵称（要和聊天记录面板里的名字一致）
# 可以写：本名、怎么称呼、是什么关系、和其他朋友的关系、需要注意的事

{friends}"""


def cmd_memory(cfg: Config, args) -> None:
    from .chat.memory import MemoryStore, NotesKeeper

    if not cfg.reply.memory_dir:
        raise ValueError("reply.memory_dir 为空，没有启用记忆")
    store = MemoryStore(cfg.reply.memory_dir)
    if args.action == "init":
        store.dir.mkdir(parents=True, exist_ok=True)
        files = {
            "profile.md": PROFILE_TEMPLATE.format(persona=cfg.reply.persona.strip()),
            "friends.md": FRIENDS_TEMPLATE.format(
                friends="\n\n".join(f"## {name}\n- {note}" for name, note in cfg.reply.friends.items())
                or "## 好友昵称\n- 本名 / 怎么称呼 / 什么关系"
            ),
        }
        for name, content in files.items():
            path = store.dir / name
            if path.exists():
                print(f"已存在，跳过: {path}")
            else:
                path.write_text(content, encoding="utf-8")
                print(f"已生成: {path}")
        print(f"记忆目录: {store.dir.resolve()}")
        print("profile.md 是人设、friends.md 是好友，都可以直接改；notes.md / inbox.md 聊天时自动生成")
        if cfg.reply.friends:
            print("提示：friends.md 已包含配置里的 [reply.friends]，可以把 config.toml 里那一节删掉，免得重复")
    elif args.action == "show":
        print(f"记忆目录: {store.dir.resolve()}")
        for title, text in (
            ("人设 profile.md", store.profile()),
            ("好友 friends.md", store.friends()),
            ("长期记忆 notes.md", store.notes()),
            ("随手记 inbox.md", store.inbox()),
        ):
            print(f"\n===== {title} =====\n{text or '（空）'}")
        print(f"\n聊天记录 {len(store.history.all())} 轮，其中 {len(store.pending_turns())} 轮还没整理进长期记忆")
    elif args.action == "update":
        from .chat.llm import make_llm

        keeper = NotesKeeper(make_llm(cfg.llm), store, cfg.reply.persona, background=False)
        print("长期记忆已更新" if keeper.update_now() else "没有需要整理的内容（或整理失败，见日志）")


def _friend_names(cfg: Config):
    from .chat.memory import MemoryStore

    store = MemoryStore(cfg.reply.memory_dir) if cfg.reply.memory_dir else None

    def names() -> list[str]:  # 好友名单：friends.md 的 “## 标题” + 配置里的 reply.friends；每次现取
        found = store.friend_names() if store else []
        return list(dict.fromkeys([*found, *cfg.reply.friends]))

    return names


def _icon_classifier(cfg: Config):
    from .game.social import IconClassifier, load_icons

    icons = load_icons(cfg.social.icons_dir) if cfg.social.enabled else {}
    if cfg.social.enabled and not icons:
        log.warning("没找到互动图标模板（%s），不处理牵手等请求", cfg.social.icons_dir)
    return IconClassifier(icons) if icons else None


def _env_watcher(cfg: Config, background: bool = True, icons=None):
    from .vision.env import EnvWatcher
    from .vision.ocr import make_ocr

    return EnvWatcher(
        make_ocr(cfg.ocr.engine, cfg.env.threads), cfg.env, _friend_names(cfg), cfg.vision.log_roi, background,
        icons=icons, icon_offset=cfg.social.icon_offset,
    )


def cmd_env(cfg: Config, args) -> None:
    """对当前画面识别一次环境（身边有谁、在哪），用来验证。"""
    from .vision.chatlog import find_input_top
    from .vision.bubbles import roi_rect

    dev = _device(cfg)
    frame = dev.screenshot()
    height, width = frame.shape[:2]
    area = roi_rect(cfg.vision.log_roi, width, height)
    panel = find_input_top(frame[:, area.x : area.x2]) is not None
    env = _env_watcher(cfg, background=False, icons=_icon_classifier(cfg))
    env.cfg.min_score = 0.0  # 调试：全打印出来
    lines = env.ocr.recognize(frame)
    print(f"聊天记录面板: {'开着（左边不扫）' if panel else '关着'}；好友名单: {'、'.join(env.names()) or '（空）'}")
    for line in lines:
        print(f"  {line.score:.2f} ({line.box.x},{line.box.y})  {line.text}")
    env.observe(frame, 0.0, panel_visible=panel)
    print("\n写进提示词的环境：\n" + (env.describe(0.0) or "（什么都没认出来）"))
    for req in env.requests.values():
        print(f"互动请求：{req.name} → {req.kind}（圆圈在 {req.pos}）")


def cmd_record(cfg: Config, args) -> None:
    """连续截图存成 jpg，用来观察游戏里的界面变化（比如别人发起牵手时出现什么提示）。"""
    import cv2

    dev = _device(cfg)
    out = Path(args.output or f"tmp/record/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / args.fps
    start = time.monotonic()
    n = 0
    print(f"录制 {args.seconds:.0f} 秒、每秒 {args.fps} 张 → {out}（Ctrl+C 提前结束）")
    try:
        while time.monotonic() - start < args.seconds:
            t = time.monotonic() - start
            frame = dev.screenshot()
            cv2.imwrite(str(out / f"{n:04d}_{t:06.2f}s.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n += 1
            time.sleep(max(0.0, start + n * interval - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"录了 {n} 张")


def cmd_run(cfg: Config, args) -> None:
    if args.live:
        cfg.reply.dry_run = False
    if args.echo:
        cfg.llm.provider = "echo"
    if cfg.vision.debug_dir:
        log.warning("vision.debug_dir 已废弃，改用 [run] dir；这次先把它当 run.dir 用")
        cfg.run.dir = cfg.vision.debug_dir
    mode = ("dry" if cfg.reply.dry_run else "live") + ("-echo" if cfg.llm.provider == "echo" else "")
    run = RunDir.create(cfg, mode)
    run.attach_log()
    log.info("本次运行的日志和截图: %s", run.path.resolve())
    try:
        _run_agent(cfg, run)
    finally:
        run.close()


def _run_agent(cfg: Config, run: RunDir) -> None:
    from .agent import Agent
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    llm = make_llm(cfg.llm)
    store = notes = None
    if cfg.reply.memory_dir and not cfg.reply.dry_run:  # dry-run 的回复没真的发出去，不记
        from .chat.memory import MemoryStore, NotesKeeper

        store = MemoryStore(cfg.reply.memory_dir)
        if not store.profile():
            log.warning("还没有人设文件 %s/profile.md，先用配置里的 persona；可以运行 memory init 生成", store.dir)
        notes = NotesKeeper(llm, store, cfg.reply.persona, cfg.reply.notes_every)
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _env_watcher(cfg, icons=icons) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(
            dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel_visible=reader.panel_visible
        )

    def env_text() -> str:  # 现在的环境 + 刚接受的互动，每次回复前现取
        now = time.monotonic()
        return "\n".join(p for p in (env.describe(now), social.describe(now) if social else "") if p)

    responder = Responder(llm, cfg.reply, store=store, notes=notes, env=env_text if env else None)
    sender = ChatSender(dev, cfg.sender, _screen_size_fn(dev))
    agent = Agent(cfg, dev, reader, responder, sender, self_filter, run=run, env=env, social=social)
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n已停止")


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdin, sys.stdout, sys.stderr):
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
    p.add_argument("-o", "--output", default="tmp/shot.png")
    p.add_argument("--grid", action="store_true")
    p.set_defaults(func=cmd_shot)

    p = sub.add_parser("detect", help="对截图 / 当前画面做一次气泡检测和 OCR，用来调参")
    p.add_argument("image", nargs="?", help="图片路径；不填则实时截屏")
    p.add_argument("-o", "--output", default="tmp/detect.png", help="标注图输出路径")
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

    p = sub.add_parser("emotes", help="快捷动作轮盘：扫描图标库 / 查看 / 编辑 / 做动作")
    esub = p.add_subparsers(dest="action", required=True)
    q = esub.add_parser("scan", help="把动作列表里的所有图标截下来，用来建图标库")
    q.add_argument("-o", "--output", help="输出目录（默认 <library_dir>/scan）")
    esub.add_parser("wheel", help="打开编辑界面，读出轮盘 8 个格子现在是什么")
    q = esub.add_parser("set", help="把某个动作放进轮盘的某个格子")
    q.add_argument("slot", type=int)
    q.add_argument("name")
    q.add_argument("--force", action="store_true", help="允许覆盖锁定的格子")
    q = esub.add_parser("do", help="做一个动作（不在轮盘上就先换上去）")
    q.add_argument("name")
    p.set_defaults(func=cmd_emotes)

    p = sub.add_parser("memory", help="记忆：init 生成人设 / 好友文件，show 查看，update 立刻整理长期记忆")
    p.add_argument("action", choices=["init", "show", "update"])
    p.set_defaults(func=cmd_memory)

    sub.add_parser("env", help="对当前画面识别一次环境（身边有谁、在哪张图），用来验证").set_defaults(func=cmd_env)

    p = sub.add_parser("record", help="连续截图存到 tmp/record/，用来观察界面变化")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("-o", "--output", help="输出目录（默认 tmp/record/<时间>）")
    p.set_defaults(func=cmd_record)

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
