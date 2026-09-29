"""命令行入口：skydango <命令> [-c config.toml]"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .config import Config, Loaded, load_all, load_config
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
    names = [n.strip() for n in args.emotes.split(",") if n.strip()]  # 假装轮盘上有这些动作
    responder = Responder(make_llm(cfg.llm), cfg.reply, available_emotes=lambda: names)
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
        print("（不回复）" if reply is None else reply.render(cfg.reply.disclosure_prefix))


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

## 喜好和看法
- 地图：最喜欢云野，能坐着看云海；霞谷的滑道很爽；雨林湿漉漉的，不爱久待；暮土有冥龙，嘴上说不怕，其实会躲
- 装扮：樱花发型天下第一；喜欢浅色、粉色系的可爱斗篷，看到好看的会忍不住夸、问在哪换的
- 爱做的事：坐着看风景、听别人弹琴；对收集先祖有兴趣，但懒得跑，更愿意被牵着走
- 说话的态度：夸人是真心的，不敷衍；吐槽点到为止，不损人
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
        from .brain.claude import ClaudeLlm

        base, claude_vars = _claude_base(cfg)
        llm = ClaudeLlm(base, claude_vars, cfg.brain.memory_model, Path("tmp/memory-claude"), cfg.brain.memory_timeout)
        keeper = NotesKeeper(llm, store, cfg.reply.persona, background=False)
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


def _scene_watcher(cfg: Config, icons=None, dev=None, background: bool = True, run: RunDir | None = None):
    """[env] 打开时"身边有谁"由谁来认：[perception] 打开就用 YOLO 感知层，否则用原来的定时整图 OCR。

    有运行目录（run）且 perception.hardcases 打开时，顺带收集难例到 runs/<这次>/hard/。
    """
    if not cfg.perception.enabled:
        if cfg.places.enabled:
            log.warning("[places] 要配合 [perception] 用（YOLO 感知层里认地图），现在没打开，不认地图")
        return _env_watcher(cfg, background=background, icons=icons)
    from .vision.detect import make_detector
    from .vision.ocr import make_ocr
    from .vision.perception import PerceptionWatcher, detector_conf

    p = cfg.perception
    detector = make_detector(p.model, p.classes, p.imgsz, detector_conf(p), p.iou, p.device)
    log.info("YOLO 感知层：%s（%s），最多 %.0f fps，帧来自%s", p.model, "、".join(getattr(detector, "providers", [])),
             p.fps, "感知线程自己截图" if p.capture == "own" else "身体主循环")
    hardcases = None
    if p.hardcases and run is not None:
        from .vision.hardcases import HardCaseCollector

        audit = make_ocr(cfg.ocr.engine, cfg.env.threads) if p.audit_interval > 0 else None
        hardcases = HardCaseCollector(run.hard, p, cfg.env, cfg.vision.log_roi, _friend_names(cfg), ocr=audit)
    places = _place_recognizer(cfg) if cfg.places.enabled else None
    unknown = None
    if run is not None:
        from .vision.unknownnames import UnknownNames

        unknown = UnknownNames(run.path / "unknown_names", _friend_names(cfg))
    return PerceptionWatcher(
        detector, make_ocr(cfg.ocr.engine, p.ocr_threads), p, cfg.env, _friend_names(cfg), cfg.vision.log_roi,
        icons=icons, background=background, capture=dev.screenshot if dev is not None else None,
        scene_change=cfg.brain.scene_change, hardcases=hardcases, unknown=unknown,
        places=places, place_interval=cfg.places.place_interval,
        gestures=_gesture_classifier(cfg), gesture_cfg=cfg.gesture,
    )


def _gesture_classifier(cfg: Config):
    if not cfg.gesture.enabled:
        return None
    from .vision.gesture import OnnxGestureClassifier

    log.info("认动作（研究性质）：%s，只报 %s", cfg.gesture.model, "、".join(cfg.gesture.names.values()))
    return OnnxGestureClassifier(cfg.gesture.model, cfg.gesture.labels, cfg.perception.device)


def _place_recognizer(cfg: Config, required: bool = False):
    """认地图：读图库（places/<地名>/*.jpg）。图库是空的：required 时报错，否则记警告、不认地图。"""
    from .vision.places import PlaceLibrary, PlaceRecognizer, make_embedder

    library = PlaceLibrary(Path(cfg.places.dir), make_embedder(cfg.places))
    n = library.load()
    if not n:
        msg = f"认地图的图库是空的（{cfg.places.dir}/）：先在游戏里用 places add <地名> 截几张"
        if required:
            raise FileNotFoundError(msg)
        log.warning(msg)
        return None
    log.info("认地图：%d 个地方、%d 张参考截图（%s）", len(library.places()), n, cfg.places.model)
    return PlaceRecognizer(library, cfg.places, cfg.env.roi, [cfg.vision.log_roi])


def _scene_boxes(cfg: Config):
    """认地图前要遮掉的人物 / 名字标签 / 圆圈：返回 图 → 框 的函数。YOLO 模型文件在就建一次检测器（一批图共用），不在就只遮固定 UI。"""
    if not Path(cfg.perception.model).exists():
        return lambda img: []
    from .vision.detect import make_detector

    p = cfg.perception
    try:
        detector = make_detector(p.model, p.classes, p.imgsz, p.conf, p.iou, p.device)
    except Exception as exc:
        log.warning("遮人用的 YOLO 跑不了（%s），只遮固定 UI", exc)
        return lambda img: []

    def boxes(img) -> list:
        try:
            return [d.box for d in detector.detect(img)]
        except Exception as exc:
            log.warning("遮人用的 YOLO 出错（%s），这张只遮固定 UI", exc)
            return []

    return boxes


def cmd_places(cfg: Config, args) -> None:
    """认地图的图库：add 截当前画面（遮掉人和 UI）存进 places/<地名>/；test 逐张认；bench 留一法比较特征模型。"""
    from .vision.places import PlaceLibrary, PlaceRecognizer

    if getattr(args, "model", None) and args.action != "bench":
        cfg.places.model = args.model[0] if isinstance(args.model, list) else args.model
    if args.action == "add":
        img = imread(args.image) if args.image else _device(cfg).screenshot()
        boxes = _scene_boxes(cfg)(img)
        library = PlaceLibrary(Path(cfg.places.dir))  # 只存图，不算特征：还没选好特征模型也能先攒图库
        recognizer = PlaceRecognizer(library, cfg.places, cfg.env.roi, [cfg.vision.log_roi])
        path = library.add(args.name, recognizer.mask(img, boxes))
        note = f"遮掉了 {len(boxes)} 个人物 / 标签" if boxes else "只遮了底部按钮栏和聊天面板（没有 YOLO 模型，画面里的人没遮）"
        print(f"存进 {args.name}：{path}（{note}）；图库现在有 {len(library.entries)} 张：{'、'.join(library.places())}")
    elif args.action == "test":
        recognizer = _place_recognizer(cfg, required=True)
        files = _images(args.source) if args.source else []
        items = [(f.name, imread(f)) for f in files] or [("当前画面", _device(cfg).screenshot())]
        find_boxes = _scene_boxes(cfg)
        for name, img in items:
            m = recognizer.recognize(img, find_boxes(img))
            verdict = f"{m.name} {m.score:.2f}" if m.name else f"认不出（最像 {m.best} {m.score:.2f}）"
            print(f"{name}  → {verdict}  第二像的别的地方 {m.second:.2f}")
    elif args.action == "bench":
        _places_bench(cfg, args.model or [cfg.places.model])


def _places_bench(cfg: Config, models: list[str]) -> None:
    """留一法：图库里每张图拿去和其余的图比，看认对 / 认错 / 不说各多少、每张多久。"""
    import dataclasses

    from .vision.places import PlaceLibrary, decide, make_embedder

    for model in models:
        pc = dataclasses.replace(cfg.places, model=model)
        library = PlaceLibrary(Path(pc.dir), make_embedder(pc))
        n = library.load()
        if not n:
            raise FileNotFoundError(f"图库是空的（{pc.dir}/）：先用 places add <地名> 截几张")
        right = wrong = silent = 0
        started = time.perf_counter()
        for i, (name, path) in enumerate(library.entries):
            m = decide(library.scores(library.vecs[i], exclude=path), pc.place_min, pc.place_margin)
            if m.name is None:
                silent += 1
            elif m.name == name:
                right += 1
            else:
                wrong += 1
                print(f"  认错：{path} 是 {name}，认成了 {m.name}（{m.score:.2f}）")
        embed_ms = _embed_ms(library)
        print(f"{model}：{len(library.places())} 个地方 {n} 张，认对 {right}/{n}、认错 {wrong}、不说 {silent}"
              f"（目标：认对 ≥ 90%、认错 ≤ 3%）；算一张特征约 {embed_ms:.0f} ms；比对 {(time.perf_counter() - started) * 1000:.0f} ms")


def _embed_ms(library) -> float:
    img = imread(library.entries[0][1])
    started = time.perf_counter()
    for _ in range(3):
        library.embedder.embed(img)
    return (time.perf_counter() - started) * 1000 / 3


def _stop_scene(env) -> None:
    """退出时停掉感知线程；收集了难例就告诉用户在哪（runs/ 只留最近几次，要用的及时收进数据集）。"""
    if hasattr(env, "stop"):
        env.stop()
    hard = getattr(env, "hardcases", None)
    if hard is not None and hard.saved:
        print(f"难例：存了 {hard.saved} 张 → {hard.folder}（收进数据集：perception label runs --from-runs）")
    unknown = getattr(env, "unknown", None)
    if unknown is not None and unknown.entries:
        print(f"没认出的名字：{len(unknown.entries)} 个 → {unknown.folder}（汇总：perception unknown-names）")


def _viewer(cfg: Config, open_browser: bool = True, brain: bool = False, on_shutdown=None):
    """起可视化网页（后台线程），打印地址、打开浏览器。端口被占用时直接退出并提示换一个。
    brain：大脑模式，打开浏览器前先挂上大脑时间线（页面第一次请求 /brain 拿到 404 就不再请求了）。"""
    from .vision.viewer import Viewer

    viewer = Viewer(cfg.viewer)
    if brain:
        from .brain.trace import BrainTrace

        viewer.brain = BrainTrace()
    try:
        url = viewer.start()
    except OSError as exc:
        raise SystemExit(f"可视化网页起不来（{cfg.viewer.host}:{cfg.viewer.port}）：{exc}\n端口可能被占用了，"
                         "用 view --port 换一个，或改 config.toml 的 viewer.port") from exc
    if cfg.viewer.host in ("127.0.0.1", "localhost", "::1"):  # 局域网模式不收 /shutdown（和手动控制一样）
        if on_shutdown is None:
            import _thread

            from .console.watchdog import once

            on_shutdown = once(_thread.interrupt_main)
        viewer.on_shutdown = on_shutdown  # POST /shutdown（管理面板点停止）= Ctrl+C，走同样的收尾
    where = "只有本机能看" if cfg.viewer.host in ("127.0.0.1", "localhost", "::1") else "同一局域网的人都能看！"
    print(f"可视化：{url}（{where}）")
    if open_browser:
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception:
            log.debug("打不开浏览器", exc_info=True)
    return viewer


def _view_frames(files: list[Path], dev, sleep=time.sleep):
    """view 的帧：回放就按顺序读图，实时就一直截图（截图失败跳过这一圈，网页上保留上一帧）。产出 (帧, 来源)。"""
    if files:
        for i, path in enumerate(files, 1):
            try:
                frame = imread(path)
            except Exception:
                log.warning("读不了 %s，跳过", path)
                continue
            yield frame, f"回放 {path.name}（{i}/{len(files)}）"
        return
    while True:
        try:
            yield dev.screenshot(), "实时截图（只看，不操作游戏）"
        except Exception as exc:
            log.warning("截图失败：%s", (str(exc).splitlines() or [type(exc).__name__])[0])
            sleep(0.5)


def _view(cfg: Config, frames, reader, env, viewer, clock=time.monotonic, sleep=time.sleep) -> int:
    """view 的主循环：读聊天记录面板 + 认人 → 交给网页。只看，不往游戏里发任何输入。返回显示了几帧。"""
    from .brain.images import is_black
    from .vision.viewer import panel_box

    period = 1.0 / max(cfg.viewer.fps, 0.1)
    shown = 0
    for frame, source in frames:
        started = clock()
        fresh = []
        try:
            fresh = reader.read(frame, started)
        except Exception:
            log.exception("读聊天出错")
        panel = panel_box(cfg.vision, reader, frame)
        try:
            env.observe(frame, started, panel_visible=panel is not None)
        except Exception:
            log.exception("识别出错")
        info = {"画面": "黑着（切场景？）"} if is_black(frame) else {}
        shown += viewer.update(frame, started, env=env, panel=panel, messages=fresh, info=info, source=source)
        for m in fresh:
            log.info("读到: %s", f"{m.speaker}：{m.text}" if m.speaker else m.text)
        sleep(max(0.0, period - (clock() - started)))
    return shown


def cmd_view(cfg: Config, args) -> None:
    """只看不动：实时截图 → 读聊天记录面板 + 认人（整图 OCR 或 YOLO）→ 网页上画框。不往游戏里发任何输入。"""
    if args.model:
        cfg.perception.enabled, cfg.perception.model = True, args.model
    if args.port is not None:
        cfg.viewer.port = args.port
    files = _images(args.images) if args.images else []
    if files:  # 回放：每张图都当场认完再显示（不在后台线程、不按间隔跳过），结果可复现
        cfg.env.interval = 0.0
        cfg.perception.fps, cfg.perception.capture = 1000.0, "body"
    dev = None if files else _device(cfg)
    reader, _ = _build_reader(cfg)
    env = _scene_watcher(cfg, _icon_classifier(cfg), dev=dev, background=not files)
    viewer = _viewer(cfg, open_browser=not args.no_browser)
    print("只看，不操作游戏；Ctrl+C 结束" + (f"。回放 {len(files)} 张图" if files else ""))
    try:
        _view(cfg, _view_frames(files, dev), reader, env, viewer)
        if files:
            print("回放完了，网页停在最后一张；Ctrl+C 结束")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if hasattr(env, "stop"):
            env.stop()
        viewer.stop()


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


def _panel_open(cfg: Config, frame) -> bool:
    from .vision.bubbles import roi_rect
    from .vision.chatlog import find_input_top

    height, width = frame.shape[:2]
    area = roi_rect(cfg.vision.log_roi, width, height)
    return find_input_top(frame[:, area.x : area.x2]) is not None


def _images(path: str) -> list[Path]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    files = [p for p in files if not any(part.startswith("_") for part in p.relative_to(root).parts)] if root.is_dir() else files
    if not files:
        raise FileNotFoundError(f"{path} 里没有图片")
    return files


def _perception(cfg: Config, args, dev=None):
    from .vision.detect import make_detector
    from .vision.ocr import make_ocr
    from .vision.perception import PerceptionWatcher, detector_conf

    p = cfg.perception
    for key in ("model", "device", "imgsz"):
        if getattr(args, key, None):
            setattr(p, key, getattr(args, key))
    if getattr(args, "far_crops", None) is not None:
        p.far_crops = args.far_crops
    detector = make_detector(p.model, p.classes, p.imgsz, detector_conf(p), p.iou, p.device)
    icons = _icon_classifier(cfg)
    watcher = PerceptionWatcher(
        detector, make_ocr(cfg.ocr.engine, p.ocr_threads), p, cfg.env, _friend_names(cfg), cfg.vision.log_roi,
        icons=icons, background=False, capture=dev.screenshot if dev is not None else None,
    )
    return detector, watcher


def _stats(values: list[float]) -> str:
    import numpy as np

    arr = np.asarray(values)
    return f"平均 {arr.mean():6.1f}  中位 {np.percentile(arr, 50):6.1f}  p95 {np.percentile(arr, 95):6.1f}  最大 {arr.max():6.1f} ms"


def cmd_perception(cfg: Config, args) -> None:
    if args.action == "bench":
        _perception_bench(cfg, args)
    elif args.action == "detect":
        _perception_detect(cfg, args)
    elif args.action == "label":
        _perception_label(cfg, args)
    elif args.action == "augment":
        _perception_augment(args)
    elif args.action == "compare":
        _perception_compare(cfg, args)
    elif args.action == "unknown-names":
        _perception_unknown_names(cfg, args)
    elif args.action == "clips":
        _perception_clips(cfg, args)
    elif args.action == "gesture-eval":
        _perception_gesture_eval(cfg, args)


def _perception_clips(cfg: Config, args) -> None:
    """动作识别的数据（三期 §3）：record 录的画面 → 按人物轨迹切成 16 帧的片段，人工挪进 <数据目录>/<动作>/。"""
    from .vision.compare import timed_files
    from .vision.detect import make_detector
    from .vision.gesture import extract_clips

    p = cfg.perception
    if args.model:
        p.model = args.model
    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    detector = make_detector(p.model, p.classes, p.imgsz, p.conf, p.iou, p.device)
    out = Path(args.output)
    n = extract_clips(((t, imread(path)) for t, path in timed), detector, out, cfg.gesture, p.conf)
    print(f"切出 {n} 段（每段 {cfg.gesture.frames} 张、{cfg.gesture.fps:g} 张/秒）→ {out}")
    print(f"人工看一遍，把片段目录挪进 <数据目录>/<动作>/（{' / '.join(cfg.gesture.labels)}；none = 站着、走路、别的动作），"
          "再用 perception gesture-eval 评估")


def _perception_gesture_eval(cfg: Config, args) -> None:
    """在分好类的片段上评估动作模型：精确率 ≥ 90%、召回率 ≥ 60% 才打开 [gesture]。"""
    from .vision import gesture

    g = cfg.gesture
    if args.model:
        g.model = args.model
    clf = gesture.OnnxGestureClassifier(g.model, g.labels, cfg.perception.device)
    r = gesture.evaluate(Path(args.data), clf, g)
    pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"{r['clips']} 段，概率 ≥ {g.min_prob} 才算报了：")
    for label, v in r.items():
        if label in ("all", "clips"):
            continue
        print(f"  {label}（{g.names.get(label, label)}）：精确率 {pct(v['precision'])}、召回率 {pct(v['recall'])}"
              f"（对 {v['tp']}、错报 {v['fp']}、漏 {v['fn']}）")
    a = r["all"]
    ok = (a["precision"] or 0) >= 0.9 and (a["recall"] or 0) >= 0.6
    print(f"总的：精确率 {pct(a['precision'])}、召回率 {pct(a['recall'])} → "
          + ("达标（精确率 ≥ 90%、召回率 ≥ 60%），可以真机试 10 分钟" if ok else "没达标，先别打开 [gesture]"))


def _perception_unknown_names(cfg: Config, args) -> None:
    """最近几次运行里读得清楚、但不在好友名单里的名字。只列出来，要加好友自己改 friends.md（或游戏里 #friend）。"""
    from .vision.unknownnames import collect

    rows = collect(Path(args.runs), args.last, _friend_names(cfg)())
    if not rows:
        print(f"最近 {args.last} 次运行里没有没认出的名字")
        return
    print(f"最近 {args.last} 次运行里读到、但不在好友名单里的名字（按出现次数）：")
    for r in rows:
        print(f"{r['count']:>4} 次  {r['name']}  最后 {r['last']}  {r['image']}")
    print("是好友的话手动加进 memory/friends.md（## 昵称），或者在游戏里发 #friend 昵称 备注；OCR 读错的不用管")


def _perception_compare(cfg: Config, args) -> None:
    """同一批录像上对比现有的整图 OCR 和 YOLO 感知层（一期 M2），输出 report.md / summary.json / diff/。"""
    import json

    import cv2

    from .vision.compare import compare_frames, report_md, side_by_side, summarize, timed_files

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    cfg.env.interval = args.interval
    out = Path(args.output or f"tmp/compare/{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "diff").mkdir(parents=True, exist_ok=True)
    icons = _icon_classifier(cfg)
    env = _env_watcher(cfg, background=False, icons=icons)
    _, yolo = _perception(cfg, args)
    print(f"{len(timed)} 帧（{timed[0][0]:.1f}~{timed[-1][0]:.1f} s），现有方案每 {args.interval:g} 秒扫一次，YOLO 每帧都跑 → {out}")

    def frames():
        for n, (t, path) in enumerate(timed, 1):
            frame = imread(path)
            if n % 50 == 0:
                print(f"  {n}/{len(timed)}")
            yield t, path.name, frame, _panel_open(cfg, frame)

    def on_diff(result, frame):
        img = side_by_side(frame, env.overlay(result.t), yolo.overlay(result.t))
        imwrite(out / "diff" / f"{Path(result.file).stem}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

    results = compare_frames(frames(), env, yolo, on_diff)
    summary = summarize(results, cfg.env.keep, cfg.perception.keep)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_md(summary), encoding="utf-8")
    print(f"两边不一致 {len(summary['diff_frames'])} 帧；报告：{out / 'report.md'}")


def _perception_augment(args) -> None:
    """训练集加运动模糊 / 压暗的样本（只动 images/train，标注原样复制）。"""
    from .vision.augment import augment_dataset

    try:
        counts = augment_dataset(Path(args.dataset), seed=args.seed, blur=args.blur, dark=args.dark)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    print(f"新生成：运动模糊 {counts['blur']} 张、压暗 {counts['dark']} 张（已经有的跳过 {counts['skipped']} 张）→ {args.dataset}/images/train")
    print("验证集没动（保持真实分布）。训练前在 X-AnyLabeling 里抽查几张，框应该还对得上")


def _perception_bench(cfg: Config, args) -> None:
    """测速：检测一帧要多久、整个感知一帧要多久、能不能到 perception.fps。模型可以先用官方的 yolo11n.pt（COCO）凑合。"""
    dev = None if args.images else _device(cfg)
    detector, watcher = _perception(cfg, args, dev)
    files = _images(args.images) if args.images else []
    frames = [imread(f) for f in files[: args.n]]
    print(f"模型 {cfg.perception.model}，推理尺寸 {getattr(detector, 'imgsz', cfg.perception.imgsz)}，后端 {'、'.join(getattr(detector, 'providers', []))}")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"torch {torch.__version__}，GPU：{torch.cuda.get_device_name(0)}")
        else:
            print(f"torch {torch.__version__}，没有可用的 CUDA")
    except ImportError:
        pass
    grab, det, total = [], [], []
    panel = False
    for i in range(args.n + 5):  # 前 5 帧预热（加载 CUDA 内核），不计
        started = time.perf_counter()
        frame = frames[i % len(frames)] if frames else dev.screenshot()
        grabbed = time.perf_counter()
        if i == 0 and dev is not None:
            panel = _panel_open(cfg, frame)
        watcher.process(frame, time.monotonic(), panel)
        if i >= 5:
            d, t = watcher.timings[-1]
            grab.append((grabbed - started) * 1000)
            det.append(d)
            total.append(t)
    print(f"测了 {args.n} 帧（{'图片 ' + args.images if args.images else '实时截图'}，分辨率 {frame.shape[1]}×{frame.shape[0]}）")
    if dev is not None:
        print(f"  截图     {_stats(grab)}")
    print(f"  YOLO 检测 {_stats(det)}")
    print(f"  整个感知 {_stats(total)}（含追踪、名字 OCR、圆圈匹配）")
    print(f"  远处二次检测 {watcher.far_runs} 次（far_crops = {cfg.perception.far_crops}，--far-crops 0 关掉对比）")
    per_frame = (sum(grab) + sum(total)) / len(total)
    target = 1000 / cfg.perception.fps
    verdict = "够用" if per_frame <= target else "不够，降低 fps / imgsz 或换 GPU 后端"
    print(f"每帧合计约 {per_frame:.1f} ms，目标 {cfg.perception.fps:.0f} fps = {target:.1f} ms → {verdict}")
    counts: dict[str, int] = {}
    for d in watcher.last_dets:
        counts[d.cls] = counts.get(d.cls, 0) + 1
    print("最后一帧检测到：" + ("、".join(f"{k}×{v}" for k, v in counts.items()) or "（无）"))
    if dev is not None:
        print("提示：测速时看一下游戏画面有没有变卡（模拟器也在用这张显卡）")


def _perception_detect(cfg: Config, args) -> None:
    """对一张图 / 当前画面跑一遍感知，把框画出来。"""
    import cv2

    dev = None if args.image else _device(cfg)
    _, watcher = _perception(cfg, args)
    frame = imread(args.image) if args.image else dev.screenshot()
    panel = _panel_open(cfg, frame)
    now = max(cfg.perception.stranger_after, 0.1)
    watcher.process(frame, 0.0, panel)
    watcher.process(frame, now, panel)  # 同一帧再跑一遍：人物要持续 stranger_after 秒没有名字才判陌生人
    colors = {"player": (0, 200, 0), "player_unlit": (80, 80, 80), "name_tag": (0, 200, 255), "social_ring": (255, 120, 0), "self": (200, 200, 200)}
    out = frame.copy()
    for t in watcher.last_tracks:
        b = t.box
        label = f"{t.cls}#{t.id} {t.score:.2f}"
        if t.data.get("name"):
            label += f" {t.data['name']}"
        elif t.data.get("text"):
            label += f" ?{t.data['text']}"
        if t.data.get("stranger"):
            label += " stranger"
        cv2.rectangle(out, (b.x, b.y), (b.x2, b.y2), colors.get(t.cls, (0, 0, 255)), 2)
        cv2.putText(out, label.encode("ascii", "replace").decode(), (b.x, max(12, b.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(t.cls, (0, 0, 255)), 1)
        extra = f"  名字 OCR：{t.data.get('text', '')} → {t.data.get('name', '（没对上好友名单）')}" if t.cls == "name_tag" else ""
        print(f"{t.cls:<12} {t.score:.2f}  ({b.x},{b.y},{b.w},{b.h}){'  陌生人' if t.data.get('stranger') else ''}{extra}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    imwrite(args.output, out)
    print(f"聊天记录面板：{'开着' if panel else '关着'}；标注图：{args.output}")
    print("\n写进提示词的环境：\n" + (watcher.describe(now) or "（什么都没认出来）"))
    for req in watcher.requests.values():
        print(f"互动请求：{req.name} → {req.kind}（圆圈在 {req.pos}）")


def _weak_boxes(cfg: Config, args, frame, ocr, icons, names: list[str]) -> list:
    """一帧的弱标注：整图 OCR 读好友名字标签 + 圆圈模板；面板开着时跳过面板区域。"""
    from .vision.bubbles import roi_rect
    from .vision.weaklabel import weak_labels

    height, width = frame.shape[:2]
    skip = [roi_rect(cfg.vision.log_roi, width, height)] if _panel_open(cfg, frame) else []
    return weak_labels(
        frame, ocr.recognize(frame), names, icons, cfg.social.icon_offset, skip=skip,
        keep=roi_rect(cfg.env.roi, width, height), all_text=args.all_text, min_score=args.min_score,
    )


_ASSIST_CHUNK = 50  # --assist 每段多少帧：检测 → 核对 → 写盘，原图用完就放掉


def _perception_label_assist(cfg: Config, args, items: list) -> None:
    """--assist：挑帧 → 人物候选框 → claude -p 核对 → 和弱标注合并写成数据集 + 预览 + 待核对清单。"""
    import cv2

    from .brain import claude
    from .brain.images import thumb
    from .vision import assist
    from .vision.ocr import make_ocr
    from .vision.weaklabel import data_yaml, ring_labels, split_of, yolo_line

    base, env = _brain_env(cfg)  # 没令牌 / 没 claude 先报错，别白跑检测
    a = cfg.assist
    if not (args.from_runs or args.all_frames) and items:  # 难例不是连续录像，不挑
        picked = assist.pick_frames([thumb(imread(p)) for p, _ in items], a.min_change, a.max_gap)
        print(f"挑了 {len(picked)} / {len(items)} 帧（和上一张差不多的去掉；--all-frames 不挑）")
        items = [items[i] for i in picked]
    if args.model:
        from .vision.detect import make_detector

        p = cfg.perception
        detector = make_detector(args.model, p.classes, p.imgsz, p.low_conf, p.iou, p.device)
        propose, source = (lambda f: assist.people_candidates(detector.detect(f))), args.model
    else:
        coco = assist.CocoPeople(a.proposal_models, a.proposal_conf, a.proposal_imgsz)
        propose, source = (lambda f: [d.box for d in coco.detect(f)]), ",".join(a.proposal_models)
    ocr = make_ocr(cfg.ocr.engine, cfg.env.threads)
    icons = _icon_classifier(cfg)
    names = _friend_names(cfg)()
    out = Path(args.output)
    cmd = assist.assist_command(base, a)
    work = assist.assist_workdir()
    reviewer = assist.Reviewer(
        lambda content: claude.one_shot_message(cmd, env, work, content, a.timeout), out / "_assist", a, source
    )
    print(f"{len(items)} 帧交给 Claude（{a.model}）核对：每批 {a.batch} 帧、{a.jobs} 路并发")
    index = {c: i for i, c in enumerate(cfg.perception.classes)}
    counts: dict[str, int] = {}
    results: list = []  # (帧名, 核对结果)，写清单用
    for c0 in range(0, len(items), _ASSIST_CHUNK):  # 一段一段来：原图不一次全读进内存（1000 帧约 6 GB）
        frames, weak = [], {}
        for path, stem in items[c0 : c0 + _ASSIST_CHUNK]:
            frame = imread(path)
            weak[stem] = _weak_boxes(cfg, args, frame, ocr, icons, names)
            frames.append(assist.FrameInput(stem, frame, propose(frame)))
        try:
            reviews = reviewer.review(frames)
        except assist.AssistLimit:
            raise SystemExit(f"订阅额度用完了：已核对的帧存在 {out / '_assist'}，额度恢复后重跑同一条命令会接着做") from None
        for f in frames:
            height, width = f.image.shape[:2]
            review = reviews.get(f.stem)
            results.append((f.stem, review))
            boxes = weak[f.stem] + (assist.apply_review(f.candidates, review) if review else [])
            if icons is not None:  # 陌生人头顶的圆圈：弱标注只看好友名字下方
                boxes += [("social_ring", r) for r in ring_labels(f.image, boxes, icons)]
            split = split_of(f.stem, args.val)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            imwrite(out / "images" / split / f"{f.stem}.jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = [yolo_line(index[c], b, width, height) for c, b in boxes if c in index]
            (out / "labels" / split / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            for c, _ in boxes:
                counts[c] = counts.get(c, 0) + 1
            (out / "_preview").mkdir(parents=True, exist_ok=True)
            imwrite(out / "_preview" / f"{f.stem}.jpg", assist.draw_review(f.image, weak[f.stem], f.candidates, review),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"  {min(c0 + _ASSIST_CHUNK, len(items))}/{len(items)} 帧")
    (out / "_assist").mkdir(parents=True, exist_ok=True)
    report = out / "_assist" / "review.md"
    report.write_text(assist.review_report(results), encoding="utf-8")
    (out / "data.yaml").write_text(data_yaml(out, cfg.perception.classes), encoding="utf-8")
    failed = sum(1 for _, r in results if r is None)
    added = sum(len(r.missing) for _, r in results if r)
    print("标注：" + "、".join(f"{c}×{v}" for c, v in counts.items()) + f"；Claude 补框 {added} 个；没核对 {failed} 帧")
    u = reviewer.usage
    print(f"用量（参考，订阅不按它计费）：输入 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']}（含图片）、输出 {u['output_tokens']} token")
    print(f"下一步：先看 {report}，再用 X-AnyLabeling 打开 {out / 'images'} 修正")


def _perception_label(cfg: Config, args) -> None:
    """用现有识别器给录下来的画面出弱标注（名字标签 + 圆圈），写成 YOLO 数据集。"""
    import cv2

    from .vision.bubbles import Rect, roi_rect
    from .vision.ocr import make_ocr
    from .vision.weaklabel import data_yaml, hard_images, merge_labels, ring_labels, split_of, weak_labels, with_self, yolo_line

    classes = cfg.perception.classes
    index = {c: i for i, c in enumerate(classes)}
    root = Path(args.source)
    if args.assist and args.spin:
        raise SystemExit("--assist 不能和 --spin 一起用：转圈录像已经能自动补 self，二选一")
    if args.spin and not args.model:
        raise SystemExit("--spin 要配 --model：认团子要先用模型框出人")
    if args.spin:  # camera spin / #spin 录的目录：只要转动中的帧（NNN_*.jpg），按文件名排就是时间顺序
        items = [(p, f"{root.name}_{p.stem}") for p in sorted(root.glob("[0-9][0-9][0-9]_*.jpg"))]
    elif args.from_runs:  # runs/*/hard/*.jpg：运行时收集的难例，文件名前加运行目录名
        items = hard_images(root)
    else:
        items = []
        for path in _images(args.source):
            rel = path.relative_to(root) if root.is_dir() else Path(path.name)
            items.append((path, "_".join(rel.with_suffix("").parts)))
    if args.assist:
        return _perception_label_assist(cfg, args, items)
    detector = None
    if args.model:  # 用当前模型预标注：人工只需要修正，不用从零画
        from .vision.detect import make_detector

        p = cfg.perception
        detector = make_detector(args.model, p.classes, p.imgsz, p.low_conf, p.iou, p.device)
    predicted: dict[str, list] = {}  # 文件名 → 模型预测（--spin 先全部跑一遍认团子，后面合并预标注时复用）
    selves: dict[str, Rect] = {}
    if args.spin:
        from .vision.perception import people_boxes
        from .vision.sweep import find_self

        people = []
        for path, stem in items:
            predicted[stem] = detector.detect(imread(path))
            people.append([d.box for d in people_boxes([d for d in predicted[stem] if d.score >= cfg.perception.conf])])
        frame0 = imread(items[0][0]) if items else None
        found = find_self(people, frame0.shape[1] if frame0 is not None else 1920, cfg.spin.self_motion)
        if found.box is None:
            print("认不出团子（转圈录像里没有一直在中间不动的人，或者牵着手分不开），不补 self")
        else:
            selves = {items[i][1]: box for i, box in found.per_frame.items()}
            b = found.box
            print(f"认出团子：({b.x}, {b.y}) {b.w}×{b.h}，{len(selves)} 张补 self 框")
    out = Path(args.output)
    ocr = make_ocr(cfg.ocr.engine, cfg.env.threads)
    icons = _icon_classifier(cfg)
    names = _friend_names(cfg)()
    counts = {c: 0 for c in classes}
    print(f"{len(items)} 张图 → {out}；好友名单：{'、'.join(names) or '（空，只能配 --all-text）'}"
          + (f"；模型预标注：{args.model}" if detector is not None else ""))
    for n, (path, stem) in enumerate(items, 1):
        frame = imread(path)
        height, width = frame.shape[:2]
        boxes = _weak_boxes(cfg, args, frame, ocr, icons, names)
        weak = len(boxes)
        if detector is not None:
            boxes = merge_labels(boxes, predicted[stem] if stem in predicted else detector.detect(frame))
        if stem in selves:
            boxes = with_self(boxes, selves[stem])
        if detector is not None and icons is not None:  # 模型预标注出了人：顺带补他们头顶的圆圈
            boxes += [("social_ring", r) for r in ring_labels(frame, boxes, icons)]
        split = split_of(stem, args.val)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        imwrite(out / "images" / split / f"{stem}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [yolo_line(index[c], box, width, height) for c, box in boxes if c in index]
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for c, _ in boxes:
            counts[c] = counts.get(c, 0) + 1
        if args.preview:
            view = frame.copy()
            for i, (c, b) in enumerate(boxes):
                color = ((0, 200, 255) if c == "name_tag" else (255, 120, 0)) if i < weak else (180, 0, 255)  # 紫 = 模型预标注
                cv2.rectangle(view, (b.x, b.y), (b.x2, b.y2), color, 2)
            (out / "_preview").mkdir(parents=True, exist_ok=True)
            imwrite(out / "_preview" / f"{stem}.jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if n % 20 == 0:
            print(f"  {n}/{len(items)}")
    (out / "data.yaml").write_text(data_yaml(out, classes), encoding="utf-8")
    print("自动标出：" + "、".join(f"{c}×{v}" for c, v in counts.items()))
    print(f"数据集配置：{out / 'data.yaml'}")
    print("下一步：用 X-AnyLabeling 打开 images/ 导入 YOLO 标注，给**每一张**补上 player（其他玩家）、player_unlit（没点火的黑影）"
          "和 self（团子自己）框、"
          "修正错框 —— 没补全的图会教模型“这里没有人”，训出来会漏检")


def _panel(cfg: Config, dev, reader, mode: str | None = None):
    """聊天记录面板的开关（只有它按 C）。一次 run 只建一个，传给身体 / Agent、镜头、轮盘、互动、好友树。
    mode 覆盖 [panel] mode：单独的命令（camera spin、friend-check）用 "always"，做完恢复成原样。"""
    import dataclasses

    from .chat.panel import PanelManager

    panel_cfg = cfg.panel if mode is None else dataclasses.replace(cfg.panel, mode=mode)
    return PanelManager(cfg.vision, panel_cfg, dev, reader)


def _friend_checker(cfg: Config, dev, panel=None):
    from .game.friendtree import FriendChecker

    if panel is None:
        reader, _ = _build_reader(cfg)
        panel = _panel(cfg, dev, reader, mode="always")
    return FriendChecker(dev, cfg.friend_check, panel=panel)


def cmd_friend_check(cfg: Config, args) -> None:
    """手动试一次：点 (x, y) 打开好友树面板、截图、按配置的办法关掉。用来核对面板样子和关面板的办法。"""
    dev = _device(cfg)
    checker = _friend_checker(cfg, dev)
    result = checker.check(args.x, args.y)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for tag, img in (("before", result.before), ("opened", result.opened), ("after", result.after)):
        imwrite(out / f"{tag}.png", img)
    opened = checker.looks_open(result.changed)
    print(f"点了 ({args.x}, {args.y})；右侧变化 {result.changed:.3f}（阈值 {cfg.friend_check.changed}）→ "
          f"{'像是打开了面板' if opened else '没什么变化，可能没点中人'}")
    if opened:
        print("面板" + (f"用 {result.closed_by} 关上了" if result.closed else f"没关上（试了 {'、'.join(cfg.friend_check.close)}），请手动关掉"))
    print(f"截图：{out}/before.png、opened.png、after.png —— 看看 opened.png 里好友和陌生人的面板有什么区别")


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


def cmd_console(cfg: Config, args) -> None:
    """管理面板：本机网页里填密钥、改设置、检测设备、启动 / 停止团子、看实时画面。设计见 docs/superpowers/specs/2026-09-29-console-design.md。"""
    from .console import runner as runner_mod
    from .console import server as server_mod
    from .console.settings import SettingsStore

    config_path = Path(args.config)
    port = args.port or cfg.console.port
    runner = runner_mod.Runner(Path.cwd(), cfg.console.child_port, cfg.console.stop_timeout, cfg.console.log_lines)
    server = server_mod.ConsoleServer(config_path, SettingsStore(config_path), runner, port, cfg.console.child_port)
    try:
        url = server.start()
    except OSError as exc:
        raise SystemExit(f"管理面板起不来（127.0.0.1:{port}）：{exc}\n端口可能被占用了，用 console --port 换一个") from exc
    print(f"管理面板：{url}（只有本机能看）；Ctrl+C 结束（会先停掉团子）")
    if server.orphan:
        print(f"注意：{cfg.console.child_port} 端口上有上次留下的团子，面板上可以让它退出")
    if not args.no_browser:
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception:
            log.debug("打不开浏览器", exc_info=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n正在停止团子…（再按一次 Ctrl+C 强制结束，轮盘可能换不回）")
    try:
        runner.close()
    except KeyboardInterrupt:
        print("强制结束团子；用 python -m skydango emotes wheel 检查轮盘")
        runner.kill()
    finally:
        server.stop()


def cmd_run(cfg: Config, args) -> None:
    if args.live:
        cfg.reply.dry_run = False
    if args.dry_run:  # 管理面板总是显式传 --live / --dry-run，config.toml 的 dry_run 不偷偷改掉面板上的选择
        cfg.reply.dry_run = True
    if args.viewer_port is not None:  # 管理面板起的子进程：viewer 固定端口、只给本机
        cfg.viewer.port, cfg.viewer.host = args.viewer_port, "127.0.0.1"
    import _thread

    from .console import watchdog

    request_exit = watchdog.once(_thread.interrupt_main)  # 看门狗和 /shutdown 共用：只中断一次，不打断收尾
    if args.parent_pid is not None:
        watchdog.watch_parent(args.parent_pid, request_exit)
    if args.echo:
        cfg.llm.provider = "echo"
    cfg.brain.enabled = not args.no_brain  # 默认接大脑；普通 Agent 只留作调试
    if cfg.vision.debug_dir:
        log.warning("vision.debug_dir 已废弃，改用 [run] dir；这次先把它当 run.dir 用")
        cfg.run.dir = cfg.vision.debug_dir
    mode = ("dry" if cfg.reply.dry_run else "live") + ("-echo" if cfg.llm.provider == "echo" else "")
    mode += "-brain" if cfg.brain.enabled else ""
    run = RunDir.create(cfg, mode)
    run.attach_log()
    log.info("本次运行的日志和截图: %s", run.path.resolve())
    viewer = None
    try:
        if args.view:
            viewer = _viewer(cfg, open_browser=not args.no_browser, brain=cfg.brain.enabled, on_shutdown=request_exit)
        if cfg.brain.enabled:
            _run_brain(cfg, run, args.no_emotes, args.duration, viewer)
        else:
            _run_agent(cfg, run, args.no_emotes, args.duration, viewer)
    finally:
        if viewer is not None:
            viewer.stop()
        run.close()


def _build_emotes(cfg: Config, dev, panel, no_emotes: bool):
    """准备聊天时做动作：读一次轮盘。做不了（关掉了 / 图标库空 / 读轮盘失败）返回 None，聊天照常。"""
    if no_emotes or not cfg.emotes.enabled:
        log.info("这次不做动作")
        return None
    from .game.emotes import EmotePlayer
    from .game.wheel import EmoteLibrary, Wheel

    library = EmoteLibrary(cfg.wheel.library_dir)
    if not library.names:
        log.warning("图标库 %s 是空的，这次不做动作（先 emotes scan，把用得上的改名放进去）", cfg.wheel.library_dir)
        return None
    player = EmotePlayer(dev, Wheel(dev, cfg.wheel, library), cfg.emotes, panel)
    try:
        player.start()
    except Exception as exc:
        log.warning("读轮盘失败，这次不做动作: %s", exc)
        return None
    log.info("能做的动作: 轮盘上 %s；可以换上去的 %s", "、".join(player.on_wheel()) or "（无）", "、".join(player.extra) or "（无）")
    return player


def _run_agent(cfg: Config, run: RunDir, no_emotes: bool = False, duration: float = 0.0, viewer=None) -> None:
    from .agent import Agent
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender
    from .vision.envdiff import snapshot

    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    panel = _panel(cfg, dev, reader)
    llm = make_llm(cfg.llm)
    from .chat.memory import MemoryStore

    # #friend/#remember 是主人自己的本地操作，不算"团子回复了什么"：不受 dry_run 影响，配了就写
    command_store = MemoryStore(cfg.reply.memory_dir) if cfg.reply.memory_dir else None
    store = notes = None
    if cfg.reply.memory_dir and not cfg.reply.dry_run:  # dry-run 的聊天回复没真的发出去，不记
        from .chat.memory import NotesKeeper

        store = command_store
        if not store.profile():
            log.warning("还没有人设文件 %s/profile.md，先用配置里的 persona；可以运行 memory init 生成", store.dir)
        notes = NotesKeeper(llm, store, cfg.reply.persona, cfg.reply.notes_every)
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _scene_watcher(cfg, icons, dev, run=run) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel=panel)

    def env_text() -> str:  # 现在的环境 + 刚接受的互动，每次回复前现取
        now = time.monotonic()
        return "\n".join(p for p in (env.describe(now), social.describe(now) if social else "") if p)

    emotes = _build_emotes(cfg, dev, panel, no_emotes)
    responder = Responder(
        llm, cfg.reply, store=store, notes=notes, env=env_text if env else None,
        available_emotes=emotes.available if emotes else None,
        env_snapshot=(lambda: snapshot(env, time.monotonic())) if env else None,
    )
    sender = ChatSender(dev, cfg.sender, _screen_size_fn(dev))
    agent = Agent(
        cfg, dev, reader, responder, sender, self_filter, run=run, env=env, social=social, emotes=emotes, store=command_store,
        viewer=viewer, camera=_camera(cfg, dev, panel), panel=panel,
    )
    try:
        agent.run(duration)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        _stop_scene(env)
        if emotes is not None:
            try:
                emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")
        try:
            panel.shutdown()  # 按需模式：退出时把聊天面板恢复成开着
        except Exception:
            log.exception("聊天面板没恢复")


def _camera(cfg: Config, dev, panel):
    """视角控制：转的时候向面板管理者借面板（关掉），转完归还。普通模式（#spin）和大脑模式共用。"""
    from .brain.camera import Camera

    return Camera(dev, cfg.brain.camera_step, panel)


def cmd_camera(cfg: Config, args) -> None:
    """camera spin：在电脑前直接转一圈、存每帧截图，标定一圈几秒、视野角、画面多模糊（感知层二期 §0）。"""
    from .runlog import write_spin

    dev = _device(cfg)
    reader, _ = _build_reader(cfg)
    camera = _camera(cfg, dev, _panel(cfg, dev, reader, mode="always"))
    spin = cfg.spin
    turns = max(1, min(args.turns, spin.max_turns))
    seconds = args.seconds if args.seconds is not None else spin.seconds_per_turn
    out = Path(args.output) if args.output else Path("tmp/spin") / time.strftime("%H%M%S")
    print(f"转 {turns} 圈（一圈按住 {seconds:.2f} 秒，{spin.fps:.0f} fps）…")
    result = camera.spin(dev.screenshot, turns, seconds, spin.fps)
    s = write_spin(out, result, turns)
    print(f"转了 {s['seconds']} 秒 {s['frames']} 张（实际 {s['fps']} fps），面板重开：{'是' if s['panel_reopened'] else '否'}，"
          f"转前转后差异 {s['drift']}{'，中途画面黑了' if s['blackout'] else ''}")
    print(f"截图和 summary.json 在 {out}")
    if cfg.perception.enabled:  # 顺便看 YOLO 汇总得对不对、多久
        env = _scene_watcher(cfg, dev=dev, background=False)
        started = time.perf_counter()
        swept = env.sweep([(0.0, result.before), *result.frames], spin)
        print(f"扫描（{time.perf_counter() - started:.1f} 秒）：{swept.text()}")
        if swept.self_box is not None:
            b = swept.self_box
            print(f"认出团子：({b.x}, {b.y}) {b.w}×{b.h}")


def _claude_base(cfg: Config, hint: str = "") -> tuple[list[str], dict[str, str]]:
    """隔离的 Claude Code（单独配置目录 + claude setup-token 令牌）：命令 + 环境。大脑、眼睛、记忆整理共用。"""
    from .brain.claude import claude_env, resolve_claude
    from .chat.llm import read_key

    try:
        token = read_key(cfg.brain.token_env)
    except RuntimeError:
        raise RuntimeError(
            f"没有找到大脑用的 Claude 令牌：先运行 claude setup-token，再 setx {cfg.brain.token_env} \"<令牌>\"" + hint
        ) from None
    return resolve_claude(cfg.brain.claude_path), claude_env(token, cfg.brain.config_dir)


def _brain_env(cfg: Config) -> tuple[list[str], dict[str, str]]:
    """大脑和眼睛的 Claude Code：先检查 mcp，再 _claude_base。"""
    import importlib.util

    hint = "；不想接大脑可以用 --no-brain（调试用的普通模式）"
    if importlib.util.find_spec("mcp") is None:
        raise RuntimeError("大脑要用 mcp：先 pip install --user mcp" + hint)
    return _claude_base(cfg, hint)


def _run_brain(cfg: Config, run: RunDir, no_emotes: bool = False, duration: float = 0.0, viewer=None) -> None:
    """统管大脑：身体在当前线程跑（独占设备）；大脑（常驻 Claude Code）和眼睛各一个后台线程；工具经本机 MCP 服务。"""
    import dataclasses
    import threading

    from .brain.body import Body
    from .brain.claude import ClaudeLlm, one_shot
    from .brain.events import EventQueue
    from .brain.eyes import Eyes, eyes_command
    from .brain.images import scene_note
    from .brain.loop import Brain, log_brain_message
    from .brain.mcp_server import SkyServer
    from .brain.prompt import brain_prompt
    from .brain.session import BrainSession
    from .brain.tools import ToolBox
    from .brain.trace import BrainTrace
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    base, claude_vars = _brain_env(cfg)  # 先检查令牌和 claude 命令，缺了早点报错
    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    panel = _panel(cfg, dev, reader)
    store = notes = None
    if cfg.reply.memory_dir:  # dry-run 也读人设和记忆（看大脑的表现要用），但不写
        from .chat.memory import MemoryStore, NotesKeeper

        store = MemoryStore(cfg.reply.memory_dir)
        if not cfg.reply.dry_run:
            # 记忆整理也走 Claude（订阅）：随手记、整理 notes.md 各起一次性 claude -p，在记忆后台线程里跑
            memory_llm = ClaudeLlm(base, claude_vars, cfg.brain.memory_model, run.path / "brain" / "memory", cfg.brain.memory_timeout)
            notes = NotesKeeper(memory_llm, store, cfg.reply.persona, cfg.reply.notes_every)
    live_store = None if cfg.reply.dry_run else store
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _scene_watcher(cfg, icons, dev, run=run) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel=panel)
    emotes = _build_emotes(cfg, dev, panel, no_emotes)
    camera = _camera(cfg, dev, panel)
    friend_checker = _friend_checker(cfg, dev, panel) if cfg.friend_check.enabled else None
    panels, panel_ops = _panels(cfg, dev, reader)
    events = EventQueue()
    # 大脑离线时的备用回复：在身体线程里调，给短超时、不重试；不带记忆存储，免得和身体重复记聊天记录
    fallback = Responder(make_llm(dataclasses.replace(cfg.llm, timeout=10.0, max_retries=0)), cfg.reply)
    body = Body(
        cfg, dev, reader, ChatSender(dev, cfg.sender, _screen_size_fn(dev)), self_filter, events,
        env=env, social=social, emotes=emotes, camera=camera, friend_checker=friend_checker, fallback=fallback, store=live_store, notes=notes, run=run,
        viewer=viewer, panel=panel, panels=panels, panel_ops=panel_ops,
    )
    work = run.path / "brain"
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(eyes_command(base, cfg.brain), claude_vars, work / "eyes", content, cfg.brain.eyes_timeout),
        frame=lambda: body.last_frame,
        labels=lambda: dict(env.labels) if env else {},
        blackout=lambda: body.blackout,
        label_keep=cfg.env.interval * 2 + 1,
        note=(lambda now, s: scene_note(env, now, s)) if hasattr(env, "strangers") else None,
        proactive=cfg.proactive,
        busy=(lambda now: bool(env.nearby(now))) if env else (lambda now: False),
        on_news=body.news,
    )
    body.friend_names = _friend_names(cfg)
    events.subscribe(eyes.notice)
    toolbox = ToolBox(body, eyes, cfg.brain.max_steps, cfg.brain.max_says, memory=store)  # recall 只读，dry-run 也给
    server = SkyServer(toolbox)
    server.start()
    trace = None if viewer is None else (viewer.brain or BrainTrace())  # 网页上的大脑时间线（一般 _viewer 已经挂好）
    session = BrainSession(
        base, claude_vars, work / "session", server.url, brain_prompt(
            cfg.reply, store, quick_around=hasattr(env, "sweep"), panel_auto=cfg.panel.mode == "auto", history_turns=cfg.brain.history_turns,
            proactive=cfg.proactive.enabled,
        ),
        cfg.brain.model, cfg.brain.effort, cfg.brain.turn_timeout,
        on_message=trace.chain(log_brain_message) if trace is not None else log_brain_message,
    )
    brain = Brain(
        cfg.brain, cfg.chat, session, toolbox, events, nearby=env.nearby if env else (lambda now: []),
        eyes=eyes, run=run, store=live_store, trace=trace,
    )
    if viewer is not None:
        viewer.brain = trace
        from .brain.manual import ManualControl
        from .vision.viewer import LOCAL_HOSTS

        if cfg.viewer.host in LOCAL_HOSTS:  # 手动控制：网页上直接让身体说话 / 做动作 / 转视角，只给本机
            viewer.control = ManualControl(body, eyes, events)
        else:
            log.warning("可视化网页开在局域网（%s）：局域网模式下关掉了手动控制", cfg.viewer.host)
    stop = threading.Event()
    brain_thread = threading.Thread(target=brain.run, args=(stop,), name="brain", daemon=True)
    eyes_thread = threading.Thread(target=eyes.run, args=(stop,), name="eyes", daemon=True)
    body.brain_offline = lambda now: brain.offline(now) or not brain_thread.is_alive()
    body.brain_busy = lambda: brain.chat_turn
    brain_thread.start()
    eyes_thread.start()
    try:
        body.run(duration, stop)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        stop.set()
        _stop_scene(env)
        if panels is not None:
            panels.close()
        body.shutdown()  # 先复原镜头、恢复轮盘、让排队的命令失败：不等大脑
        brain_thread.join(timeout=5)
        if live_store is not None and not brain_thread.is_alive():
            try:
                brain.farewell()  # 把这次的经过记进 inbox.md
            except Exception:
                log.exception("退出前写经过失败")
        session.close()
        server.stop()


def _panels(cfg: Config, dev, reader):
    """面板识别：(PanelWatcher, PanelOps)；[panels] enabled = false 时 (None, None)。"""
    if not cfg.panels.enabled:
        return None, None
    from .game.panels import PanelOps
    from .vision.ocr import make_ocr
    from .vision.panels import PanelWatcher, load_cards

    watcher = PanelWatcher(
        cfg.panels, load_cards(cfg.panels.cards_dir), make_ocr(cfg.ocr.engine, cfg.panels.ocr_threads),
        {"chat_input": reader.panel_visible},
    )
    return watcher, PanelOps(dev, watcher, cfg.panels)


def _feature_note(feature, hit: bool, score: float) -> str:
    mark = "✓" if hit else "✗"
    if feature.kind == "template":
        if feature.template is None:
            return f"template {feature.image} 没有模板图 ✗"
        return f"template {feature.image} {score:.2f}{'≥' if hit else '<'}{feature.threshold:.2f} {mark}"
    if feature.kind == "dark":
        return f"dark 中位 {score:.0f}{'≤' if hit else '>'}{feature.max_value} {mark}"
    return f"{feature.kind} {feature.name} {mark}"


def cmd_panels(cfg: Config, args) -> None:
    """面板识别的核对工具：scan 逐张看每张卡的特征分数、read 细读、cut 裁模板。都不往游戏里发输入。"""
    import cv2

    from .vision.bubbles import roi_rect
    from .vision.panels import UNKNOWN, PanelWatcher, describe_reading, load_cards

    if args.action == "cut":
        folder = Path(cfg.panels.cards_dir) / args.card
        if not folder.is_dir():
            raise SystemExit(f"没有卡片 {args.card}（{folder} 不存在）")
        roi = [float(v) for v in args.roi.split(",")]
        if len(roi) != 4:
            raise SystemExit("--roi 要 4 个数：x1,y1,x2,y2（0~1 归一化）")
        img = imread(args.image)
        height, width = img.shape[:2]
        out = folder / args.name
        imwrite(out, roi_rect(roi, width, height).crop(img))
        print(f"存到 {out}；改 card.toml 引用它，再用 panels scan 看分数")
        return

    from .vision.ocr import make_ocr

    cards = load_cards(cfg.panels.cards_dir)
    ocr = make_ocr(cfg.ocr.engine, cfg.panels.ocr_threads)
    reader = _build_reader(cfg)[0]
    builtins = {"chat_input": reader.panel_visible}
    if args.source:
        shots = [(p.name, imread(p)) for p in _images(args.source)]
    else:
        shots = [("screenshot.png", _device(cfg).screenshot())]
    out_dir = Path("tmp") / "panels"
    for name, frame in shots:
        watcher = PanelWatcher(cfg.panels, cards, ocr, builtins, background=False)
        watcher.observe(frame, 0.0)  # 第一次 observe 一定会跑一次通用兜底
        found = [p for p in watcher.state.others() if p.name == UNKNOWN or watcher.cards[p.name].texts]
        checks = watcher.explain(frame)
        if args.action == "read":
            print(f"== {name}")
            height, width = frame.shape[:2]
            panels = [watcher.panel_for(c.card, width, height) for c in checks if c.open and c.card.name != "chat_log"]
            readings = [watcher.read(frame, p, 0.0) for p in panels] + [watcher.readings[p.name] for p in found]
            if not readings:
                print("  没有开着的面板")
            for r in readings:
                print(f"  {r.panel.describe()}：{describe_reading(r)}")
                for b in r.buttons:
                    print(f"    {b.text}  {b.kind}")
            continue
        print(f"== {name}")
        view = frame.copy()
        height, width = frame.shape[:2]
        for check in checks:
            card = check.card
            state = "开着" if check.open else ("没开（靠文字认，见通用兜底）" if not card.quick else "没开")
            notes = " | ".join(_feature_note(f, hit, score) for f, hit, score in check.hits)
            print(f"  {card.name} {card.label}{'' if card.verified else '（未核对）'}：{state}  {notes}".rstrip())
            r = roi_rect(list(card.region), width, height)
            cv2.rectangle(view, (r.x, r.y), (r.x2, r.y2), (0, 200, 0) if check.open else (128, 128, 128), 2)
        for p in found:
            print(f"  通用兜底：{p.describe()} {describe_reading(watcher.readings[p.name])}")
            cv2.rectangle(view, (p.box.x, p.box.y), (p.box.x2, p.box.y2), (0, 0, 255), 3)
        if not found:
            print("  通用兜底：没认出面板")
        imwrite(out_dir / (Path(name).stem + ".png"), view)
    watcher.close()


def cmd_look(cfg: Config, args) -> None:
    """截一张图让眼睛（Haiku）描述一遍：在真实画面上调眼睛的提示词，看它认得准不准、会不会编名字。"""
    from .brain.claude import one_shot
    from .brain.eyes import Eyes, eyes_command
    from .vision.bubbles import roi_rect
    from .vision.chatlog import find_input_top

    from .brain.images import label_note, scene_note

    base, claude_vars = _brain_env(cfg)
    frame = imread(args.image) if args.image else _device(cfg).screenshot()
    height, width = frame.shape[:2]
    area = roi_rect(cfg.vision.log_roi, width, height)
    panel = find_input_top(frame[:, area.x : area.x2]) is not None
    env = _scene_watcher(cfg, background=False)
    yolo = hasattr(env, "strangers")
    if yolo:  # 同一帧跑两次：隔 stranger_after 秒，点过火的陌生人才判得出来
        env.process(frame, 0.0, panel)
        env.process(frame, cfg.perception.stranger_after, panel)
    else:
        env.observe(frame, 0.0, panel_visible=panel)
    Path("tmp").mkdir(exist_ok=True)
    if not args.image:
        imwrite("tmp/look.jpg", frame)
    now = cfg.perception.stranger_after if yolo else 0.0
    note = (lambda t, s: scene_note(env, t, s)) if yolo else None
    scale = min(cfg.brain.image_size[0] / width, cfg.brain.image_size[1] / height, 1.0)
    print("交给眼睛的位置说明：\n" + (scene_note(env, now, scale) if yolo else label_note(dict(env.labels), scale)) + "\n")
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(
            eyes_command(base, cfg.brain), claude_vars, Path("tmp/look-claude"), content, cfg.brain.eyes_timeout
        ),
        frame=lambda: frame,
        labels=lambda: dict(env.labels),
        blackout=lambda: False,
        clock=lambda: now,
        note=note,
    )
    if args.prompt:
        eyes.look_request = Path(args.prompt).read_text(encoding="utf-8")
    started = time.perf_counter()
    text = eyes.describe_frame(frame, now)
    print(text)
    print(f"\n{cfg.brain.eyes_model}：{time.perf_counter() - started:.1f} 秒" + ("" if args.image else "；截图存在 tmp/look.jpg"))


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
    p.add_argument("--emotes", default="", help="假装轮盘上有这些动作（逗号分隔），看模型怎么用")
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

    p = sub.add_parser("look", help="截一张图让眼睛（Claude Haiku）描述一遍（调眼睛的提示词，看它认得准不准）")
    p.add_argument("image", nargs="?", help="描述这张图（不截屏）；不填就截当前画面")
    p.add_argument("--prompt", help="换一个问题：文本文件路径")
    p.set_defaults(func=lambda cfg, args: cmd_look(cfg, args))

    p = sub.add_parser("friend-check", help="点一下人物打开好友树面板、截图、再关掉（核对面板和关法；坐标按原图，如 1920×1080）")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)
    p.add_argument("-o", "--output", default="tmp/friend-check")
    p.set_defaults(func=cmd_friend_check)

    p = sub.add_parser("camera", help="视角：spin 转一圈、边转边截图（标定一圈几秒、视野角，存到 tmp/spin/<时间>）")
    csub = p.add_subparsers(dest="action", required=True)
    q = csub.add_parser("spin", help="按住方向键转一圈，按 spin.fps 截图，存转前 / 转完 / 每帧和 summary.json")
    q.add_argument("--turns", type=int, default=1, help="转几圈（最多 spin.max_turns）")
    q.add_argument("--seconds", type=float, help="一圈按住几秒（临时覆盖 spin.seconds_per_turn，标定用）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/spin/<时间>）")
    p.set_defaults(func=cmd_camera)

    p = sub.add_parser("record", help="连续截图存到 tmp/record/，用来观察界面变化")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("-o", "--output", help="输出目录（默认 tmp/record/<时间>）")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("perception", help="YOLO 感知层：测速 / 单帧检测 / 自动出弱标注")
    psub = p.add_subparsers(dest="action", required=True)
    for name, text in (("bench", "测速：检测和整个感知每帧多久、够不够 perception.fps"), ("detect", "对一张图 / 当前画面跑一遍感知，画框")):
        q = psub.add_parser(name, help=text)
        q.add_argument("--model", help="模型文件（默认 perception.model）；还没训练时可以先用 yolo11n.pt 测速")
        q.add_argument("--device", choices=["cuda", "cpu"])
        q.add_argument("--imgsz", type=int)
        if name == "bench":
            q.add_argument("--far-crops", type=int, help="远处二次检测每帧最多几块（覆盖 perception.far_crops，0 = 关）")
            q.add_argument("--images", help="用这个目录 / 这张图测（默认实时截图）")
            q.add_argument("-n", type=int, default=200, help="测多少帧")
        else:
            q.add_argument("image", nargs="?", help="图片路径；不填则实时截屏")
            q.add_argument("-o", "--output", default="tmp/perception.png")
    q = psub.add_parser("label", help="用现有识别器给录下来的画面自动标名字标签和圆圈，写成 YOLO 数据集")
    q.add_argument("source", help="图片目录（比如 record 录的 tmp/record/<时间>）；配 --from-runs 时是 runs/")
    q.add_argument("-o", "--output", default="datasets/sky")
    q.add_argument("--model", help="再用这个模型的预测（置信度 ≥ perception.low_conf）当初始标注，和弱标注重叠的留弱标注")
    q.add_argument("--from-runs", action="store_true", help="source 是 runs/：收集每次运行存下的难例（hard/*.jpg）")
    q.add_argument("--spin", action="store_true", help="source 是 camera spin / #spin 录的目录：转圈认出团子，每帧补 self 框（要配 --model）")
    q.add_argument("--val", type=float, default=0.15, help="验证集比例")
    q.add_argument("--all-text", action="store_true", help="画面里读到的字都当名字标签候选（不只好友名单里的），要人工删错的")
    q.add_argument("--min-score", type=float, default=0.9)
    q.add_argument("--preview", action="store_true", help="另存一份画了框的图到 <output>/_preview/，快速检查")
    q.add_argument("--assist", action="store_true", help="Claude 辅助标注：检测器出人物候选框，claude -p 核对后写进标注（令牌同 [brain]，见 [assist]）")
    q.add_argument("--all-frames", action="store_true", help="配 --assist：不挑帧，每一帧都核对（默认去掉和上一张差不多的帧）")
    q = psub.add_parser("compare", help="同一批录像上对比现有的整图 OCR 和 YOLO（认出率、请求延迟、陌生人 / 走开事件、耗时）")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="模型文件（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "cpu"])
    q.add_argument("--imgsz", type=int)
    q.add_argument("--interval", type=float, default=3.0, help="现有方案多久扫一次（同 env.interval）")
    q.add_argument("--far-crops", type=int, help="远处二次检测每帧最多几块（覆盖 perception.far_crops，0 = 关；开关各跑一次对比）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/compare/<时间>）")
    q = psub.add_parser("unknown-names", help="汇总最近几次运行里读到、但不在好友名单里的名字（只列出，不改 friends.md）")
    q.add_argument("--runs", default="runs", help="运行目录的上级（默认 runs/）")
    q.add_argument("--last", type=int, default=5, help="看最近几次运行")
    q = psub.add_parser("clips", help="动作识别的数据：录像按人物轨迹切成 16 帧的片段（人工再分到 <动作>/ 目录）")
    q.add_argument("source", help="record 录的目录（record --fps 8，文件名里带时间）")
    q.add_argument("-o", "--output", default="datasets/gesture/_unlabeled")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q = psub.add_parser("gesture-eval", help="在分好类的片段（<数据目录>/<动作>/<片段>/）上评估动作模型的精确率 / 召回率")
    q.add_argument("data", help="数据目录，比如 datasets/gesture")
    q.add_argument("--model", help="动作模型（默认 gesture.model）")
    q = psub.add_parser("augment", help="训练集加运动模糊（转视角）/ 压暗（暗场景）的样本，标注原样复制")
    q.add_argument("dataset", help="数据集目录（perception label 的输出，比如 datasets/sky）")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--blur", type=float, default=0.3, help="抽多少比例的图做运动模糊")
    q.add_argument("--dark", type=float, default=0.2, help="抽多少比例的图压暗")
    p.set_defaults(func=cmd_perception)

    p = sub.add_parser("places", help="认地图的图库：截图加进 places/<地名>/、逐张测试、比较特征模型")
    psub = p.add_subparsers(dest="action", required=True)
    q = psub.add_parser("add", help="截当前画面（遮掉人物和 UI）存进 places/<地名>/")
    q.add_argument("name", help="地名，比如 云野")
    q.add_argument("--image", help="用这张图，不截屏")
    q.add_argument("--model", help="特征模型（默认 places.model；\"thumb\" = 内置基线）")
    q = psub.add_parser("test", help="对一批截图（或当前画面）逐张认地图，打印地名和相似度")
    q.add_argument("source", nargs="?", help="图片目录 / 图片；不填则截当前画面")
    q.add_argument("--model", help="特征模型（默认 places.model）")
    q = psub.add_parser("bench", help="图库上留一法：每个模型认对 / 认错 / 不说多少、多快")
    q.add_argument("--model", action="append", help="特征模型，可以给多个（默认 places.model）")
    p.set_defaults(func=cmd_places)

    p = sub.add_parser("panels", help="面板识别的核对工具：逐张看特征分数、细读面板、裁模板（不往游戏里发输入）")
    psub = p.add_subparsers(dest="action", required=True)
    q = psub.add_parser("scan", help="逐张快看：每张卡开没开、每个特征的分数，外加通用兜底；标注图存 tmp/panels/")
    q.add_argument("source", nargs="?", help="图片目录 / 图片；不填则截当前画面")
    q = psub.add_parser("read", help="细读开着的面板：标题、正文、按钮和按钮类别")
    q.add_argument("source", nargs="?", help="图片目录 / 图片；不填则截当前画面")
    q = psub.add_parser("cut", help="从截图里裁一块，存成某张卡片的模板图")
    q.add_argument("image", help="截图（1920×1080）")
    q.add_argument("card", help="卡片目录名，比如 emote_panel")
    q.add_argument("name", help="模板文件名，比如 pencil.png")
    q.add_argument("--roi", required=True, help="x1,y1,x2,y2（0~1 归一化）")
    p.set_defaults(func=cmd_panels)

    p = sub.add_parser("view", help="只看不动：实时截图 → 认人 / 读聊天 → 网页上画识别框（不操作游戏）")
    p.add_argument("--images", help="回放这个目录 / 这张图（比如 record 录的），不用连模拟器")
    p.add_argument("--model", help="用这个 YOLO 模型（等于临时打开 [perception]）")
    p.add_argument("--port", type=int, help="网页端口（默认 viewer.port）")
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("console", help="管理面板：填密钥、改设置、检测设备、启动 / 停止团子、看实时画面")
    p.add_argument("--port", type=int, help="面板端口（默认 [console] port = 8760）")
    p.add_argument("--no-browser", action="store_true", help="不打开浏览器")
    p.set_defaults(func=cmd_console)

    p = sub.add_parser("run", help="启动团子（默认接统管大脑、dry-run；--no-brain 是调试用的普通模式）")
    live = p.add_mutually_exclusive_group()
    live.add_argument("--live", action="store_true", help="真的发送消息")
    live.add_argument("--dry-run", action="store_true", help="只打印不发送（覆盖 config.toml 的 reply.dry_run）")
    p.add_argument("--echo", action="store_true", help="不调模型，原样回显（联调用）")
    p.add_argument("--duration", type=float, default=0.0, help="跑多少秒后自动结束（默认一直跑）")
    p.add_argument("--no-emotes", action="store_true", help="这次不做动作（牵着手时用：做动作会松开牵手）")
    p.add_argument("--brain", action="store_true", help="接统管大脑（已是默认，保留兼容）")
    p.add_argument("--no-brain", action="store_true", help="不接大脑，用旧的普通 Agent（调试用）")
    p.add_argument("--view", action="store_true", help="开可视化网页：实时显示画面和识别框（地址见 [viewer]）")
    p.add_argument("--viewer-port", type=int, help="可视化网页用这个端口、只给本机看（管理面板用）")
    p.add_argument("--no-browser", action="store_true", help="开可视化网页时不打开浏览器")
    p.add_argument("--parent-pid", type=int, help="这个进程没了就自己退出（管理面板用）")
    p.set_defaults(func=lambda cfg, args: cmd_run(cfg, args))

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
        # 面板每次现读 secrets.toml、只注入它起的子进程：写进它自己的环境变量的话，页面上「清除」之后子进程还会继承旧 Key
        loaded = load_all(cfg_path, environ={} if args.func is cmd_console else os.environ)
    except ValueError as exc:
        if args.func is not cmd_console:
            parser.error(str(exc))
        # 面板要能打开才能在设置页上把坏掉的 console.toml / secrets.toml 指出来：先只用 config.toml
        log.warning("%s；面板先只用 config.toml，设置页上会显示这个错误", exc)
        try:
            loaded = Loaded(load_config(cfg_path if cfg_path.exists() else None), [], [])
        except ValueError as exc2:
            parser.error(str(exc2))
    cfg = loaded.cfg
    if loaded.overridden:
        log.info("console.toml 覆盖了 %d 项：%s", len(loaded.overridden), "、".join(loaded.overridden))
    if loaded.secrets and args.func is not cmd_console:
        log.info("用 secrets.toml 里的 %s", "、".join(loaded.secrets))
    try:
        args.func(cfg, args)
    except (RuntimeError, ImportError, ValueError, FileNotFoundError) as exc:
        if args.verbose:
            raise
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
