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


def _ime_switch(cfg: Config, dev):
    """run 启动时切到 ADBKeyboard（[device] switch_ime），返回的 ImeSwitch 在停下时 stop() 切回去。"""
    from .device.ime import ImeSwitch

    ime = ImeSwitch(dev, cfg.device.ime_id, cfg.device.user_ime)
    if cfg.device.switch_ime:
        ime.start()
    return ime


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
    from .device.ime import restore_target

    dev = _device(cfg)
    if args.action == "on":
        dev.enable_adb_keyboard()
    elif args.action == "off":  # 切回搜狗（或 device.user_ime）；找不到就恢复系统默认
        back = restore_target("", cfg.device.ime_id, dev.list_imes(), cfg.device.user_ime)
        if back:
            dev.set_ime(back)
        else:
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

## 脾气
- 毛病：嘴硬，爱装懒，被催就更不想动
- 执念：樱花发型天下第一，谁说不好看跟谁急
- 雷点：讨厌被说“像机器人”（打哈哈带过，别较真）
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
        if cfg.inner.enabled:
            from .inner import show_lines

            print("\n" + "\n".join(show_lines(cfg.inner, store.dir / "inner", _friend_names(cfg)(), time.time())))
            if cfg.inner.persona:
                from .inner.store import InnerStore

                print("\n" + "\n".join(InnerStore(store.dir / "inner").load_persona(quarantine=False).show_lines()))
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


def _appearance_parts(cfg: Config, run: RunDir | None = None) -> dict:
    """认装扮（spec 2026-10-01-appearance）：给 PerceptionWatcher 的关键字参数（记忆簿、特征模型、配置、存训练数据）。
    [appearance] 没开时是空的；没有运行目录（view、perception detect）或 save = false 时不存训练数据。描述器由 _run_brain 挂。"""
    a = cfg.appearance
    if not a.enabled:
        return {}
    from .vision.appearance import AppearanceBook, CropSaver, make_embedder

    embedder = make_embedder(a)
    saver = CropSaver(run.path / "appearance", a.save_every, a.save_max) if run is not None and a.save else None
    log.info("认装扮：特征 %s%s", embedder.key, f"，训练数据存进 {saver.folder}" if saver is not None else "")
    return {
        "appearance": AppearanceBook(a, embedder.key, keep=cfg.perception.keep),
        "embedder": embedder, "appearance_cfg": a, "saver": saver,
    }


def _catalog_collector(cfg: Config, run: RunDir | None):
    """装扮图鉴收集（spec 2026-10-02-catalog-collect §4）：[catalog] 开着、有运行目录（run 的 dry-run / live）才建；
    view、perception detect 没有运行目录，不收。[perception] 没开时 _scene_watcher 根本不走到这里。"""
    if not cfg.catalog.enabled or run is None:
        return None
    from .vision.catalog import CatalogCollector

    collector = CatalogCollector(cfg.catalog, Path(cfg.catalog.dir), run.path.name, time.strftime("%Y-%m-%d"))
    log.info("图鉴收集：近处的人清楚的裁图存进 %s", collector.folder)
    return collector


def _scene_watcher(cfg: Config, icons=None, dev=None, background: bool = True, run: RunDir | None = None, light: bool = False):
    """[env] 打开时"身边有谁"由谁来认：[perception] 打开就用 YOLO 感知层，否则用原来的定时整图 OCR。

    有运行目录（run）且 perception.hardcases 打开时，顺带收集难例到 runs/<这次>/hard/。
    light：大脑模式（只有它的身体会按 3 举蜡烛）才找黑影身上的火焰圆盘、出 light 请求。
    """
    if not cfg.perception.enabled:
        if cfg.places.enabled:
            log.warning("[places] 要配合 [perception] 用（YOLO 感知层里认地图），现在没打开，不认地图")
        if cfg.appearance.enabled:
            log.warning("[appearance] 要配合 [perception] 用，现在没打开，不认装扮")
        return _env_watcher(cfg, background=background, icons=icons)
    from .vision.detect import make_detector
    from .vision.ocr import make_ocr
    from .vision.perception import PerceptionWatcher, detector_conf

    p = cfg.perception
    want_light = light and cfg.social.enabled and "light" in cfg.social.accept_strangers
    attrs = _person_attrs(cfg)
    detector = make_detector(p.model, p.classes, p.imgsz, detector_conf(p, want_light, attrs=attrs is not None), p.iou, p.device)
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
    flame = None
    if want_light:
        from .vision.candle import load_flame

        flame = load_flame(cfg.social.flame)
    return PerceptionWatcher(
        detector, make_ocr(cfg.ocr.engine, p.ocr_threads), p, cfg.env, _friend_names(cfg), cfg.vision.log_roi,
        icons=icons, background=background, capture=dev.screenshot if dev is not None else None,
        scene_change=cfg.brain.scene_change, hardcases=hardcases, unknown=unknown,
        places=places, place_interval=cfg.places.place_interval,
        gestures=_gesture_classifier(cfg), gesture_cfg=cfg.gesture,
        social_cfg=cfg.social, flame=flame,
        light_dir=run.path / "light" if want_light and run is not None else None, **_appearance_parts(cfg, run),
        call_window=cfg.call.window, camera_settle=cfg.track.settle, attrs=attrs,
        catalog=_catalog_collector(cfg, run),
    )


def _person_attrs(cfg: Config):
    """感知层第二层（spec 2026-10-02-perception-attrs）：[attrs] 开着、模型读得进来才建 PersonAttrs；
    模型不存在 / 打不开 / 主干对不上时 load_model 已经警告过，返回 None（等于没开）。"""
    if not cfg.attrs.enabled:
        return None
    from .vision.attrs import PersonAttrs, load_model

    model = load_model(cfg.attrs, _attrs_device(cfg))
    if model is None:
        return None
    log.info("感知层第二层：%s（主干 %s）", cfg.attrs.model, cfg.attrs.backbone)
    _warn_attrs_on_cpu(cfg, model)
    return PersonAttrs(cfg.attrs, model)


def _warn_attrs_on_cpu(cfg: Config, model) -> None:
    """[perception] device = cuda，第二层的主干会话却只有 CPU 后端（onnxruntime 是 CPU 版）：警告一次。
    实测 CPU 上 DINOv2-small 一张裁图约 32 ms、4 张约 128 ms，会把感知层拖慢一大截。"""
    get = getattr(getattr(getattr(model, "embedder", None), "session", None), "get_providers", None)
    if cfg.perception.device != "cuda" or get is None:
        return
    if set(get()) <= {"CPUExecutionProvider"}:
        log.warning("感知层第二层的主干在 CPU 上跑（onnxruntime 是 CPU 版，一张裁图约 30 ms），会拖慢感知层："
                    "装 onnxruntime-gpu，或者把 [attrs] max_crops 改成 1、every 改成 1.0")


def _call_enabled(cfg: Config, env) -> bool:
    """按 Q 喊一声（spec 2026-10-01-q-call）：开着 [call] 且 env 是 YOLO 感知层才有 call 工具。"""
    from .brain.calling import call_available

    return call_available(cfg, env)


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

    from .vision.people import OBJECT_NAMES

    def boxes(img) -> list:
        try:  # 物品（长椅、钢琴……）是地标，不遮
            return [d.box for d in detector.detect(img) if d.cls not in OBJECT_NAMES]
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


def cmd_catalog(cfg: Config, args) -> None:
    """图鉴收集的离线工具（spec 2026-10-02-catalog-collect §6）：录像上跑感知层 + 收集器（录像时间当时钟），
    存下的图和运行时同样的结构，另写 candidates.jsonl（每个看过的候选过没过门槛）和 sheet.jpg，用来定 min_height / sharp_min。"""
    import json
    from collections import Counter

    from .vision.catalog import CatalogCollector, contact_sheet
    from .vision.compare import timed_files
    from .vision.trackeval import subsample

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    out = Path(args.output or f"tmp/catalog/{time.strftime('%Y%m%d-%H%M%S')}")
    _, watcher = _perception(cfg, args)
    now = [0.0]
    watcher.clock = lambda: now[0]
    collector = CatalogCollector(cfg.catalog, out, Path(args.source).name, "offline", wall=lambda: now[0], trace=True)
    watcher.catalog = collector
    kept = list(subsample(timed, args.fps))
    print(f"{len(timed)} 帧里按 {args.fps:g} 帧 / 秒抽了 {len(kept)} 帧 → {out}")
    try:
        for n, (t, path) in enumerate(kept, 1):
            now[0] = t
            frame = imread(path)
            watcher.process(frame, t, _panel_open(cfg, frame))
            if n % 100 == 0:
                print(f"  {n}/{len(kept)}")
    finally:  # Ctrl+C / 读图出错时，已经收到的照样写出
        watcher.stop()  # 收集器全部写出
        out.mkdir(parents=True, exist_ok=True)
        (out / "candidates.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in collector.candidates), encoding="utf-8")
    fails = Counter(r["fail"] or "收下" for r in collector.candidates)
    print(f"看了 {len(collector.candidates)} 个候选：" + "、".join(f"{k} {v}" for k, v in fails.most_common()))
    made = contact_sheet(collector.folder)
    if made is None:
        print("一张都没存下（看 candidates.jsonl 是哪条门槛挡住的）")
        return
    sheet, legend = made
    imwrite(out / "sheet.jpg", sheet)
    print(f"存了 {collector.saved} 张 → {collector.folder}；总览 {out / 'sheet.jpg'}")
    for i, line in enumerate(legend, 1):
        print(f"  第 {i} 行：{line}")


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
    catalog = getattr(env, "catalog", None)
    if catalog is not None and catalog.saved:
        print(f"图鉴收集：存了 {catalog.saved} 张 → {catalog.folder}")


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
    attrs = _person_attrs(cfg)
    detector = make_detector(p.model, p.classes, p.imgsz, detector_conf(p, attrs=attrs is not None), p.iou, p.device)
    icons = _icon_classifier(cfg)
    watcher = PerceptionWatcher(
        detector, make_ocr(cfg.ocr.engine, p.ocr_threads), p, cfg.env, _friend_names(cfg), cfg.vision.log_roi,
        icons=icons, background=False, capture=dev.screenshot if dev is not None else None,
        **_appearance_parts(cfg),  # 只挂记忆簿：不存训练数据、不描述
        attrs=attrs,
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
    elif args.action == "gesture-train":
        _perception_gesture_train(cfg, args)
    elif args.action == "gesture-label":
        _perception_gesture_label(cfg, args)
    elif args.action == "attrs-label":
        _perception_attrs_label(cfg, args)
    elif args.action == "crops":
        _perception_crops(cfg, args)
    elif args.action == "attrs-train":
        _perception_attrs_train(cfg, args)
    elif args.action == "attrs-eval":
        _perception_attrs_eval(cfg, args)
    elif args.action == "appearance-eval":
        _perception_appearance_eval(cfg, args)
    elif args.action == "halo-eval":
        _perception_halo_eval(cfg, args)
    elif args.action == "track-eval":
        _perception_track_eval(cfg, args)


def _existing_clips(dataset: Path, out: Path, recording: str) -> dict[Path, list[str]]:
    """数据目录每个子目录（_unlabeled、各类别、_discard……）和输出目录里，这段录像已经切出来的片段：{目录: [片段名]}。"""
    bases = sorted(d for d in dataset.iterdir() if d.is_dir()) if dataset.is_dir() else []
    if out.is_dir() and out.resolve() not in {b.resolve() for b in bases}:
        bases.append(out)
    found: dict[Path, list[str]] = {}
    for base in bases:
        names = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(f"{recording}__"))
        if names:
            found[base] = names
    return found


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
    out = Path(args.output) if args.output else Path(cfg.gesture.dataset) / "_unlabeled"
    recording = Path(args.source).resolve().name  # 片段名带录像名，后面标注 / 训练按它找回来源
    existing = _existing_clips(Path(cfg.gesture.dataset), out, recording)
    if existing and not args.force:
        where = "、".join(f"{base}（{len(names)} 段）" for base, names in existing.items())
        raise SystemExit(
            f"录像 {recording} 切过了：{where} 里已经有 {recording}__ 开头的片段。重切会在 _unlabeled 里再造一份已经标过的片段，"
            "训练时两份可能标得不一样。\n加 --force 只切数据目录里哪儿都还没有的片段（已有的名字一律跳过）；"
            "要是另一段录像碰巧同名，先把录像目录改个名再切")
    names = {n for group in existing.values() for n in group}
    detector = make_detector(p.model, p.classes, p.imgsz, p.conf, p.iou, p.device)
    n = extract_clips(((t, imread(path)) for t, path in timed), detector, out, cfg.gesture, p.conf, recording,
                      near=p.near, far=p.far, existing=names)
    print(f"切出 {n} 段（每段 {cfg.gesture.frames} 张、{cfg.gesture.fps:g} 张/秒）→ {out}"
          + (f"；数据目录里已有的 {len(names)} 段同名片段没重切" if names else ""))
    print(f"人工看一遍，把片段目录挪进 <数据目录>/<动作>/（{' / '.join(cfg.gesture.labels)}；none = 站着、走路、别的动作），"
          "再用 perception gesture-eval 评估")


def _perception_crops(cfg: Config, args) -> None:
    """第二层外形头的数据：从数据集 / 录像 / 难例目录裁人物图，--writeback 把标注页确认过的人物写回数据集，见 vision/attrs_data.py。"""
    import glob
    from datetime import datetime

    from .vision import attrs_data as ad

    sources: list[Path] = []
    for s in args.source:  # runs/*/hard 可能被 shell 展开了，也可能原样传进来
        hits = sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
        if not hits:
            raise SystemExit(f"{s} 没有匹配到任何目录")
        sources += [Path(h) for h in hits]
    for src in sources:
        if not src.is_dir():
            raise SystemExit(f"{src} 不是目录")
    out = Path(args.out)
    is_dataset = lambda d: (d / "images").is_dir() and (d / "labels").is_dir()  # noqa: E731
    if args.writeback:
        datasets = [d for d in sources if is_dataset(d)]
        if not datasets:
            raise SystemExit("--writeback 要给数据集目录（含 images/ 和 labels/）")
        for d in datasets:
            res = ad.writeback(d, out, datetime.now())
            print(f"{d}: 写回 {res['boxes']} 个框（{res['frames']} 帧）" + ("，原 labels/ 备份在 _backup/" if res["boxes"] else ""))
        return
    from .vision.detect import make_detector

    p = cfg.perception
    detector = make_detector(args.model or p.model, p.classes, p.imgsz, args.conf, p.iou, p.device)
    for src in sources:
        if is_dataset(src):
            counts = ad.crops_from_dataset(src, detector, out, args.conf)
        else:
            counts = ad.crops_from_images(src, detector, out, args.conf)
        print(f"{src}: " + "，".join(f"{k} {v}" for k, v in counts.items()))
    print(f"裁图在 {out}（_unlabeled/ 等 Claude 初分、标注页确认）")


def _attrs_replays(cfg: Config, frames: list, model, notes: list[str]):
    """在验证帧上回放 0.2 和 [perception] low_conf 两档：([回放结果], 建议阈值)。检测器建不起来 / 没有带标注的帧就返回空。"""
    from .vision import attrs_train as at
    from .vision.detect import make_detector

    p, a = cfg.perception, cfg.attrs
    if not frames:
        notes.append("没有可回放的验证帧（数据集的 images/val 里没有图），没做整帧回放")
        return [], None
    try:
        detector = make_detector(p.model, p.classes, p.imgsz, 0.2, p.iou, p.device)
    except Exception as exc:
        notes.append(f"YOLO 检测器加载失败（{exc}），没做整帧回放")
        return [], None
    replays, suggest = [], None
    for low in sorted({0.2, p.low_conf}):
        records = at.collect(frames, detector, model, low)
        if suggest is None or low == p.low_conf:
            suggest = at.sweep_thresholds(records, p.conf)
        replays.append({**at.simulate(records, p.conf, a.accept, a.reject, a.yolo_w), "conf_low": low, "conf": p.conf,
                        "accept": a.accept, "reject": a.reject, "reject_n": a.reject_n, "yolo_w": a.yolo_w})
    return replays, suggest


def _attrs_device(cfg: Config, args=None) -> str:
    return (getattr(args, "device", None) or cfg.attrs.device or cfg.perception.device) or "cpu"


def _perception_attrs_train(cfg: Config, args) -> None:
    """训练外形头（vision/attrs_train.py）：切分 → 合并类别 → DINOv2 特征（缓存）→ numpy 线性头 → 存 .npz → 报告 tmp/attrs-train/<时间>/report.md。"""
    import dataclasses
    import datetime as dt
    from pathlib import Path

    from .vision import attrs, attrs_train as at
    from .vision.embed import OnnxEmbedder

    a = cfg.attrs
    now = dt.datetime.now()
    out = Path(args.out) if args.out else at.default_out(Path("models"), now)
    at.check_out(out, a.model, args.force)  # 最终路径（含默认路径）都查
    data = Path(args.data) if args.data else Path("datasets/attrs")
    if not (data / "form").is_dir():
        raise SystemExit(f"{data}/form 不存在：先 perception crops 裁图、attrs-label 初分，再在管理面板「标注」页确认")
    device = _attrs_device(cfg, args)
    try:
        embedder = OnnxEmbedder(a.backbone, norm="imagenet", device=device, what="attrs.backbone")
    except Exception as exc:
        raise SystemExit(f"主干 {a.backbone} 加载失败：{exc}") from None
    try:
        res = at.run_training(data, embedder, data / "_features")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    ev = res["eval"]
    print(f"训练 {res['train_n']} 张（加镜像）、验证 {res['val_n']} 张，l2 = {res['l2']:g}，宏平均 F1 {ev['macro_f1']:.3f}")
    for n in res["notes"]:
        print(f"  {n}")
    attrs.save_model(out, {"form": res["head"]}, Path(a.backbone).name, f"{embedder.size}:imagenet", f"{now:%Y%m%d}")
    print(f"模型 → {out}")
    model = attrs.load_model(dataclasses.replace(a, model=str(out)), device, embedder)
    notes: list[str] = []
    replays, suggest = [], None
    if model is not None:
        frames = []
        for root in sorted({r["dataset"] for r in at.crop_rows(data).values() if r.get("source") == "dataset" and r.get("dataset")}):
            frames += at.frames_in(Path(root))
        replays, suggest = _attrs_replays(cfg, frames, model, notes)
    for r in replays:
        print(f"回放 conf_low = {r['conf_low']:g}：纯 YOLO 精确率 {r['baseline']['precision']:.0%}、召回率 {r['baseline']['recall']:.0%}；"
              f"加外形头 {r['second']['precision']:.0%} / {r['second']['recall']:.0%}")
    for n in notes:
        print(f"  {n}")
    folder = Path("tmp") / "attrs-train" / f"{now:%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(at.report_md(data=data, model=out, when=now, result=res, replays=replays,
                                                    suggest=suggest, notes=notes), encoding="utf-8")
    print(f"报告 → {folder / 'report.md'}" + (f"；建议 accept = {suggest[0]:g}、reject = {suggest[1]:g}" if suggest else ""))


def _perception_attrs_eval(cfg: Config, args) -> None:
    """只跑整帧回放：给数据集（images/val + labels/val）和外形头模型，写报告 tmp/attrs-train/<时间>/report.md。"""
    import datetime as dt
    from pathlib import Path

    from .vision import attrs, attrs_train as at

    cfg.attrs.model = args.model
    model = attrs.load_model(cfg.attrs, _attrs_device(cfg, args))
    if model is None:
        raise SystemExit(f"外形头模型 {args.model} 加载失败（原因见上面的警告）")
    notes: list[str] = []
    replays, suggest = _attrs_replays(cfg, at.frames_in(Path(args.dataset)), model, notes)
    now = dt.datetime.now()
    for r in replays:
        print(f"conf_low = {r['conf_low']:g}")
        print("\n".join(at.replay_md(r)))
    for n in notes:
        print(n)
    folder = Path("tmp") / "attrs-train" / f"{now:%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(at.report_md(data=Path(args.dataset), model=Path(args.model), when=now, result=None,
                                                    replays=replays, suggest=suggest, notes=notes), encoding="utf-8")
    print(f"报告 → {folder / 'report.md'}" + (f"；建议 accept = {suggest[0]:g}、reject = {suggest[1]:g}" if suggest else ""))


def _perception_attrs_label(cfg: Config, args) -> None:
    """第二层外形头的 Claude 初分：_unlabeled/ 里的裁图每 16 张拼成 4×4 一张图，结果写进 _unlabeled/claude.json，见 vision/attrs_data.py。"""
    import dataclasses

    from .brain import claude
    from .imageio import imread
    from .vision import assist, attrs_data as ad

    out = Path(args.source) if args.source else Path("datasets/attrs")
    unl = out / "_unlabeled"
    if not unl.is_dir():
        raise SystemExit(f"{unl} 不是目录：先 perception crops 裁图，或者把数据目录当参数传进来")
    existing = ad.load_form_guesses(unl)
    todo = ad.pending_crops(unl, existing, args.recheck)
    done = len(ad.pending_crops(unl, {}, True)) - len(todo)
    if not todo:
        print(f"没有要初分的裁图（{done} 张已有 {ad.FORM_GUESS_FILE}；要重做加 --recheck）")
        return
    base, env = _brain_env(cfg)  # 没令牌 / 没 claude 先报错
    a = dataclasses.replace(cfg.assist, batch=2)  # 每批 2 张拼图（32 张裁图）
    cmd = assist.assist_command(base, a, system=ad.FORM_PROTOCOL.system)
    work = assist.assist_workdir()
    cache = Path("tmp") / "attrs-label" / "cache"  # Reviewer 的缓存按拼图名存，这里用不上：续跑靠 claude.json
    cache.mkdir(parents=True, exist_ok=True)
    reviewer = assist.Reviewer(
        lambda content: claude.one_shot_message(cmd, env, work, content, a.timeout), cache, a,
        "attrs", protocol=ad.FORM_PROTOCOL,
    )
    print(f"{len(todo)} 张裁图交给 Claude（{a.model}）初分：每张拼图 {ad.FORM_PER_SHEET} 张、每批 {a.batch} 张拼图、{a.jobs} 路并发"
          + (f"；{done} 张已有 {ad.FORM_GUESS_FILE}，跳过" if done else ""))
    counts: dict[str, int] = {}
    failed = 0
    unreadable: list[str] = []
    step = ad.FORM_PER_SHEET * a.batch * max(1, a.jobs)  # 每轮并发跑完就写盘：额度用完 / 中断最多丢这一轮
    for c0 in range(0, len(todo), step):
        sheets: list[assist.FrameInput] = []
        for s0 in range(c0, min(c0 + step, len(todo)), ad.FORM_PER_SHEET):
            names, crops = [], []
            for n in todo[s0 : s0 + ad.FORM_PER_SHEET]:
                try:  # 裁图可能刚在标注页被挪走或者坏了：跳过这张
                    crops.append(imread(unl / n))
                    names.append(n)
                except (OSError, RuntimeError, ValueError):
                    unreadable.append(n)
            if names:
                sheets.append(ad.make_sheet_input(names, crops))
        batch_new: dict[str, ad.Guess] = {}
        try:
            for got in (reviewer.review(sheets) if sheets else {}).values():
                batch_new.update(got or {})
        except assist.AssistLimit:
            raise SystemExit(f"订阅额度用完了：已初分的裁图存好了，额度恢复后重跑同一条命令会接着做（{unl / ad.FORM_GUESS_FILE}）") from None
        asked = {n for s in sheets for n in s.hints}
        failed += len(asked - set(batch_new))
        batch_new = {n: g for n, g in batch_new.items() if (unl / n).is_file()}  # 初分期间被标注页挪走的不写
        ad.write_form_guesses(unl, batch_new, a.model)
        for g in batch_new.values():
            counts[g.label] = counts.get(g.label, 0) + 1
        print(f"  {min(c0 + step, len(todo))}/{len(todo)} 张")
    print("初分：" + ("、".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "没有") + f"；没初分成 {failed} 张")
    if unreadable:
        print(f"读不了 {len(unreadable)} 张（裁图没了或者坏了，跳过）")
    u = reviewer.usage
    print(f"用量（参考，订阅不按它计费）：输入 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']}（含图片）、输出 {u['output_tokens']} token")
    print("下一步：管理面板「标注」页确认外形，再 perception crops --writeback")


def _perception_gesture_label(cfg: Config, args) -> None:
    """动作片段的 Claude 初分：每段 16 帧拼成 4×4 一张图，结果写进片段目录的 claude.json（标注页读它），见 vision/gesture_label.py。"""
    import dataclasses

    from .brain import claude
    from .vision import assist, gesture_label as gl
    from .vision.gesture import SUFFIXES, load_clip

    dataset = Path(cfg.gesture.dataset)
    root = Path(args.source) if args.source else dataset / "_unlabeled"
    if not root.is_dir():
        raise SystemExit(f"{root} 不是目录：先 perception clips 切片段，或者把片段目录当参数传进来")
    clips = sorted(d for d in root.iterdir() if d.is_dir())
    blind = args.blind
    name = gl.BLIND_FILE if blind else gl.GUESS_FILE
    protocol = gl.GESTURE_BLIND_PROTOCOL if blind else gl.GESTURE_PROTOCOL
    key = gl.blind_key if blind else (lambda clip: clip)  # 给 Claude 看的片段名（盲分时连录像名也藏起来）
    incomplete, done, todo = [], 0, []
    for d in clips:
        if sum(1 for p in d.iterdir() if p.suffix.lower() in SUFFIXES) != 16:
            incomplete.append(d.name)
        elif gl.load_guess(d, name) is not None and not args.recheck:
            done += 1
        else:
            todo.append(d)
    if incomplete:
        print(f"跳过 {len(incomplete)} 段（不是正好 16 张图）：" + "、".join(incomplete))
    if not todo:
        print(f"没有要初分的片段（{done} 段已有 {name}；要重做加 --recheck）")
        return
    base, env = _brain_env(cfg)  # 没令牌 / 没 claude 先报错
    a = dataclasses.replace(cfg.assist, batch=8)
    cache = dataset / "_assist" / ("blind" if blind else "")  # 两种初分的缓存键一样，分开放
    if args.recheck:  # 重做：Reviewer 的缓存也清掉，不然直接命中
        for d in todo:
            (cache / f"{key(d.name)}.json").unlink(missing_ok=True)
    cmd = assist.assist_command(base, a, system=protocol.system)
    work = assist.assist_workdir()
    reviewer = assist.Reviewer(
        lambda content: claude.one_shot_message(cmd, env, work, content, a.timeout), cache, a,
        "gesture", protocol=protocol,
    )
    print(f"{len(todo)} 段交给 Claude（{a.model}）初分" + ("（不看录像名）" if blind else "")
          + f"：每批 {a.batch} 段、{a.jobs} 路并发" + (f"；{done} 段已有 {name}，跳过" if done else ""))
    same, both = 0, 0  # --blind：和看录像名的那次比
    counts: dict[str, int] = {}
    failed, gone = 0, 0
    unreadable: list[str] = []
    for c0 in range(0, len(todo), _ASSIST_CHUNK):
        chunk, frames = [], []
        for d in todo[c0 : c0 + _ASSIST_CHUNK]:
            try:  # 片段可能刚在标注页被挪走，或者有张图坏了：跳过这一段，别让整次初分停下
                frames.append(assist.FrameInput(key(d.name), gl.contact_sheet(load_clip(d)), []))
            except (OSError, RuntimeError, ValueError):
                unreadable.append(d.name)
                continue
            chunk.append(d)
        try:
            guesses = reviewer.review(frames) if frames else {}
        except assist.AssistLimit:
            raise SystemExit(f"订阅额度用完了：已初分的片段存好了，额度恢复后重跑同一条命令会接着做（缓存 {cache}）") from None
        for d in chunk:
            g = guesses.get(key(d.name))
            if g is None:
                failed += 1
                continue
            if not d.is_dir():  # Claude 看的这几分钟里，片段在标注页被标走了
                gone += 1
                continue
            try:
                gl.write_guess(d, g, a.model, name)
            except OSError:
                gone += 1
                continue
            counts[g.label] = counts.get(g.label, 0) + 1
            other = gl.load_guess(d) if blind else None
            if other is not None:
                both += 1
                same += other.label == g.label
        print(f"  {min(c0 + _ASSIST_CHUNK, len(todo))}/{len(todo)} 段")
    print("初分：" + ("、".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "没有") + f"；没初分成 {failed} 段")
    if both:
        print(f"和看录像名的那次一致 {same}/{both} 段")
    if gone:
        print(f"已被标走 {gone} 段（初分时已经在标注页挪走了，没写 claude.json）")
    if unreadable:
        print(f"读不了 {len(unreadable)} 段（片段目录没了或者有图坏了，跳过）：" + "、".join(unreadable))
    u = reviewer.usage
    print(f"用量（参考，订阅不按它计费）：输入 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']}（含图片）、输出 {u['output_tokens']} token")


def _perception_gesture_train(cfg: Config, args) -> None:
    """训练动作模型（vision/gesture_train.py）：确认过的片段 → 切分（_split.json）→ DINOv2-small 特征 + 时序头
    → 导出 ONNX（不覆盖 models/gesture.onnx）→ 和 PyTorch 比输出 → 验证集评估 → 报告 tmp/gesture-train/<时间>/report.md。"""
    import datetime as dt
    import shutil

    from .vision import gesture, gesture_train as gt

    g = cfg.gesture
    if args.out and Path(args.out).resolve() == Path(g.model).resolve() and not args.force:
        raise SystemExit(f"--out {args.out} 就是 [gesture] model 正在用的模型：没评估过的新模型别直接覆盖它。"
                         "换个路径训练、看完报告达标再复制过去；确实要覆盖就加 --force")
    data = Path(args.data) if args.data else Path(g.dataset)
    if not data.is_dir():
        raise SystemExit(f"{data} 不是目录：先 perception clips 切片段，再在管理面板「标注」页确认")
    samples, skipped = gt.list_samples(data, g.labels, g.frames)
    if skipped:
        print(f"跳过 {len(skipped)} 段（不是正好 {g.frames} 张图）：" + "、".join(skipped))
    missing = gt.check_counts(samples, g.labels, g.names)
    if missing:
        raise SystemExit(f"确认过的片段不够（每类至少 {gt.MIN_PER_CLASS} 段），先在管理面板「标注」页多标一些：\n  "
                         + "\n  ".join(missing))
    sp = gt.split(samples, g.labels)
    gt.save_split(data, sp)
    print(f"切分：训练 {len(sp['train'])} 段、验证 {len(sp['val'])} 段 → {data / gt.SPLIT_FILE}")
    for w in sp["warnings"]:
        print(f"  注意：{w}")
    notes: list[str] = []
    device = args.device
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("没有 CUDA：退回 CPU 训练（提特征会慢很多）")
            notes.append("没有 CUDA，在 CPU 上训练")
            device = "cpu"
    print(f"加载 DINOv2-small（{device}）…")
    extractor = gt.DinoExtractor(device)

    def progress(epoch: int, loss: float, f1: float) -> None:
        print(f"  第 {epoch} 轮：训练损失 {loss:.4f}、宏平均 F1 {f1:.3f}")

    print(f"提特征（缓存在 {data / '_features'}）、训练时序头，最多 {args.epochs} 轮：")
    head, info = gt.train(samples, sp, g.labels, extractor, data / "_features", epochs=args.epochs, device=device,
                          size=g.size, progress=progress)
    print(f"最好第 {info['best_epoch']} 轮：宏平均 F1 {info['best_f1']:.3f}")
    now = dt.datetime.now()
    out = Path(args.out) if args.out else gt.default_out(Path("models"), now)
    gt.export_onnx(extractor, head, out, frames=g.frames, size=g.size)
    print(f"导出 → {out}")
    by_name = {s.clip: s for s in samples}
    first = [by_name[c] for c in sp["val"][:3] if c in by_name]
    parity = None
    if first:
        parity = gt.onnx_diff(extractor, head, out, [gt.clip_array(gesture.load_clip(s.path), g.size) for s in first])
        if parity > 1e-3:
            notes.append(f"ONNX 和 PyTorch 的输出最大差 {parity:.2e}（> 1e-3），导出可能有问题")
            print(f"注意：{notes[-1]}")
    evaluation = None
    if sp["val"]:
        clf = gesture.OnnxGestureClassifier(str(out), g.labels, "cpu")
        evaluation = gesture.evaluate(data, clf, g, only=set(sp["val"]))
        print("验证集：")
        _print_gesture_eval(evaluation, g)
    folder = Path("tmp") / "gesture-train" / f"{now:%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data / gt.SPLIT_FILE, folder / gt.SPLIT_FILE)  # 下次训练会改写数据目录里的切分，这份留给这个模型对照
    report = folder / "report.md"
    report.write_text(gt.report_md(
        data=data, model=out, labels=g.labels, names=g.names, samples=samples, split=sp, info=info,
        evaluation=evaluation, min_prob=g.min_prob, skipped=skipped, parity=parity, notes=notes, when=now,
    ), encoding="utf-8")
    print(f"报告 → {report}（这次的切分也复制了一份在旁边）")
    print(f"类别顺序（= 模型输出顺序）：{', '.join(g.labels)}；达标后再复制成 {g.model}、打开 [gesture] enabled")


def _print_gesture_eval(r: dict, g) -> bool:
    """打印 gesture.evaluate 的结果，返回达没达标。"""
    pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"{r['clips']} 段，概率 ≥ {g.min_prob} 才算报了：")
    for label, v in r.items():
        if label in ("all", "clips", "wrong"):
            continue
        print(f"  {label}（{g.names.get(label, label)}）：精确率 {pct(v['precision'])}、召回率 {pct(v['recall'])}"
              f"（对 {v['tp']}、错报 {v['fp']}、漏 {v['fn']}）")
    a = r["all"]
    ok = (a["precision"] or 0) >= 0.9 and (a["recall"] or 0) >= 0.6
    print(f"总的：精确率 {pct(a['precision'])}、召回率 {pct(a['recall'])} → "
          + ("达标（精确率 ≥ 90%、召回率 ≥ 60%），可以真机试 10 分钟" if ok else "没达标，先别打开 [gesture]"))
    return ok


def _perception_gesture_eval(cfg: Config, args) -> None:
    """在分好类的片段上评估动作模型：精确率 ≥ 90%、召回率 ≥ 60% 才打开 [gesture]。
    数据目录里有 _split.json（gesture-train 写的）时默认只评验证集，--all 评全部。"""
    from .vision import gesture
    from .vision.gesture_train import SPLIT_FILE, load_split

    g = cfg.gesture
    if args.model:
        g.model = args.model
    data = Path(args.data)
    only = None
    split = None if args.all else load_split(data)
    if split is not None and split.get("val"):
        only = set(split["val"])
        print(f"只评验证集（{len(only)} 段，按 {data / SPLIT_FILE}；--all 评全部）")
        try:
            older = Path(g.model).stat().st_mtime < (data / SPLIT_FILE).stat().st_mtime
        except OSError:
            older = False
        if older:  # 切分是后来的训练重写的：这个模型训练时用的是另一份切分
            print("注意：这个模型比现在的切分旧，验证集里可能有它训练过的片段，分数会偏高"
                  "（它训练时的切分在 tmp/gesture-train/<时间>/_split.json）")
    elif split is not None:
        print(f"{data / SPLIT_FILE} 里没有验证集：评全部片段（包括训练过的，结果会偏好）")
    clf = gesture.OnnxGestureClassifier(g.model, g.labels, cfg.perception.device)
    _print_gesture_eval(gesture.evaluate(data, clf, g, only=only), g)


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


def _perception_track_eval(cfg: Config, args) -> None:
    """追踪和接回的离线评估（spec 2026-10-01-tracking-relink-motion §6）：同一批录像上基线（升级开关全关）和当前配置各跑一遍，
    按录像时间抽帧（--fps，模拟身体截图），写 report.md / summary.json。"""
    import dataclasses
    import json

    from .vision.compare import timed_files
    from .vision.trackeval import baseline, evaluate, report_md, subsample

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    out = Path(args.output or f"tmp/track-eval/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    _, current = _perception(cfg, args)  # 先建当前配置（--model 等覆盖写进 cfg.perception），基线照抄它再关开关
    # 基线也不接第二层（[attrs] 开着时当前配置接、基线不接，才比得出升级的效果）
    _, base = _perception(dataclasses.replace(cfg, perception=baseline(cfg.perception),
                                              attrs=dataclasses.replace(cfg.attrs, enabled=False)), args)
    kept = list(subsample(timed, args.fps))
    print(f"{len(timed)} 帧里按 {args.fps:g} 帧 / 秒抽了 {len(kept)} 帧（{kept[0][0]:.1f}~{kept[-1][0]:.1f} s），基线和当前配置各跑一遍 → {out}")

    def frames():
        for n, (t, path) in enumerate(kept, 1):
            frame = imread(path)
            if n % 100 == 0:
                print(f"  {n}/{len(kept)}")
            yield t, path.name, frame, _panel_open(cfg, frame)

    results = {}
    for key, watcher in (("baseline", base), ("current", current)):
        print("基线（开关全关）" if key == "baseline" else "当前配置")
        results[key] = evaluate(frames(), watcher)
    meta = {"source": str(args.source), "frames": len(kept), "fps": args.fps}
    (out / "summary.json").write_text(json.dumps(results | {"meta": meta}, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(report_md(results["baseline"], results["current"], meta), encoding="utf-8")
    print(f"报告：{out / 'report.md'}")


def _perception_appearance_eval(cfg: Config, args) -> None:
    """认装扮的离线标定（spec 2026-10-01-appearance §10）：录像上跑感知层（强制打开认装扮，不存训练数据、不描述），
    收集每条轨迹的外观特征 → 相似度分布、建议的 match / changed / margin、藏标签重放，写 report.md / summary.json。"""
    import dataclasses
    import json

    from .vision.appearance_eval import Harvester, pair_scores, replay, report_md, suggest, untagged
    from .vision.compare import timed_files

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    a = cfg.appearance
    a.enabled = True
    if args.embed:
        a.model = args.embed
    out = Path(args.output or f"tmp/appearance-eval/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    _, watcher = _perception(cfg, args)  # 只挂记忆簿：saver = None、没有描述器
    print(f"{len(timed)} 帧（{timed[0][0]:.1f}~{timed[-1][0]:.1f} s），特征 {watcher.embedder.key} → {out}")
    harvest = Harvester()
    for n, (t, path) in enumerate(timed, 1):
        frame = imread(path)
        watcher.process(frame, t, _panel_open(cfg, frame))
        harvest.add(watcher.tracker.tracks.values(), t)
        if n % 50 == 0:
            print(f"  {n}/{len(timed)}")
    ids = harvest.by_identity()
    same, diff = pair_scores(ids)
    s = suggest(same, diff)
    seq = harvest.sequence()
    summary = {
        "source": str(args.source), "frames": len(timed), "embed": watcher.embedder.key,
        "identities": {k: len(v) for k, v in ids.items()}, "suggest": s,
        "current": {"match": a.match, "changed": a.changed, "margin": a.margin},
        "replay": replay(seq, a),
    }
    if s["match"] is not None and s["changed"] is not None:
        summary["replay_suggested"] = replay(seq, dataclasses.replace(a, match=s["match"], changed=s["changed"], margin=s["margin"]))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_md(summary), encoding="utf-8")
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"{len(ids)} 个身份（{sum(not untagged(k) for k in ids)} 个带名字），"
          f"同一身份 {s['same_n']} 对、不同身份 {s['diff_n']} 对")
    print(f"建议：match {fmt(s['match'])}（现在 {a.match:g}）、changed {fmt(s['changed'])}（现在 {a.changed:g}）、margin {s['margin']:g}")
    r = summary["replay"]
    print(f"藏标签重放（现在的配置）：对 {r['right']}、错 {r['wrong']}、漏 {r['missed']}；报告：{out / 'report.md'}")


def _perception_halo_eval(cfg: Config, args) -> None:
    """呼唤光圈的离线标定（spec 2026-10-01-q-call §5）：录像上逐帧跑感知层，画每条人物 / 团子轨迹头顶的亮度变化曲线，
    标出超过门槛的时间段，给建议的 [call] halo_rise；写 report.md / curves.png / summary.json。"""
    import json

    from .vision.compare import timed_files
    from .vision.halo_eval import CANDIDATES, Curves, plot, report_md, segments, suggest

    try:
        timed, skipped = timed_files(_images(args.source))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    if skipped:
        print(f"跳过 {len(skipped)} 张文件名里没有时间的图（比如 {skipped[0].name}）")
    out = Path(args.output or f"tmp/halo-eval/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    cfg.appearance.enabled = False
    _, watcher = _perception(cfg, args)
    print(f"{len(timed)} 帧（{timed[0][0]:.1f}~{timed[-1][0]:.1f} s）→ {out}")
    curves = Curves()
    for n, (t, path) in enumerate(timed, 1):
        frame = imread(path)
        watcher.process(frame, t, _panel_open(cfg, frame))
        curves.add(frame, t, [tr for tr in watcher.last_tracks if tr.last == t])
        if n % 50 == 0:
            print(f"  {n}/{len(timed)}")
    rises = curves.rises()
    current = cfg.call.halo_rise
    summary = {
        "source": str(args.source), "frames": len(timed), "suggest": suggest(rises), "current": current,
        "tracks": {str(k): {"peak": max(v for _, v in pts), "segments": segments(pts, current)} for k, pts in rises.items()},
        "counts": {f"{c:g}": sum(len(segments(pts, c)) for pts in rises.values()) for c in CANDIDATES},
    }
    plot(rises, out / "curves.png")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_md(summary), encoding="utf-8")
    s = summary["suggest"]
    print(f"{len(rises)} 条轨迹；噪声 {s['noise']}、最高的峰 {s['peak']} → 建议 halo_rise {s['halo_rise']}（现在 {current:g}）")
    print(f"报告：{out / 'report.md'}")


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
    if getattr(args, "attrs", False):
        _bench_attrs(cfg, args, detector, frames, dev)
    if dev is not None:
        print("提示：测速时看一下游戏画面有没有变卡（模拟器也在用这张显卡）")


def _bench_attrs(cfg: Config, args, detector, frames: list, dev) -> None:
    """bench --attrs：同样的帧上，只检测 vs 检测 + 最多 [attrs] max_crops 张人物裁图过外形头，各多少 fps。模型加载不上就说明原因、跳过。"""
    from .vision import attrs, attrs_train as at

    model = attrs.load_model(cfg.attrs, _attrs_device(cfg, args))
    if model is None:
        print(f"--attrs：外形头模型 {cfg.attrs.model} 没加载上（原因见上面的警告），只测了上面这些")
        return
    plain, withattrs = at.bench_attrs(lambda i: frames[i % len(frames)] if frames else dev.screenshot(),
                                      detector, model, args.n, 5, cfg.attrs.max_crops)
    fps = lambda ms: 1000 / (sum(ms) / len(ms))  # noqa: E731
    print(f"外形头（每帧最多 {cfg.attrs.max_crops} 张裁图）：")
    print(f"  只检测           {_stats(plain)}  -> {fps(plain):.1f} fps")
    print(f"  检测 + 外形头    {_stats(withattrs)}  -> {fps(withattrs):.1f} fps")


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


def _print_skipped(skipped: list[str], reason: str) -> None:
    """perception label 收尾：没写进数据集的帧和原因（没有就不打印）。"""
    if skipped:
        print(f"跳过 {len(skipped)} 帧（{reason}，没写图和标注）：")
        for stem in skipped:
            print(f"  {stem}.jpg")


def _perception_label_assist(cfg: Config, args, items: list) -> None:
    """--assist：挑帧 → 人物候选框 → claude -p 核对 → 和弱标注合并写成数据集 + 预览 + 待核对清单。"""
    import cv2

    from .brain import claude
    from .brain.images import thumb
    from .vision import assist
    from .vision.ocr import make_ocr
    from .vision.weaklabel import CLASH_REASON, data_yaml, dataset_clash, ring_labels, split_of, write_sample, yolo_line

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
    skipped: list[str] = []  # 数据集里已有同名的另一张图：不核对、不写
    for c0 in range(0, len(items), _ASSIST_CHUNK):  # 一段一段来：原图不一次全读进内存（1000 帧约 6 GB）
        frames, weak = [], {}
        for path, stem in items[c0 : c0 + _ASSIST_CHUNK]:
            frame = imread(path)
            if dataset_clash(out, stem, frame) is not None:  # 交给 Claude 之前就查，不白花额度
                skipped.append(stem)
                continue
            weak[stem] = _weak_boxes(cfg, args, frame, ocr, icons, names)
            frames.append(assist.FrameInput(stem, frame, propose(frame)))
        try:
            reviews = reviewer.review(frames)
        except assist.AssistLimit:
            raise SystemExit(f"订阅额度用完了：已核对的帧存在 {out / '_assist'}，额度恢复后重跑同一条命令会接着做") from None
        for f in frames:
            height, width = f.image.shape[:2]
            review = reviews.get(f.stem)
            boxes = weak[f.stem] + (assist.apply_review(f.candidates, review) if review else [])
            if icons is not None:  # 陌生人头顶的圆圈：弱标注只看好友名字下方
                boxes += [("social_ring", r) for r in ring_labels(f.image, boxes, icons)]
            lines = [yolo_line(index[c], b, width, height) for c, b in boxes if c in index]
            if write_sample(out, split_of(f.stem, args.val), f.stem, f.image, lines) is not None:
                skipped.append(f.stem)
                continue
            results.append((f.stem, review))
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
    _print_skipped(skipped, CLASH_REASON)
    failed = sum(1 for _, r in results if r is None)
    added = sum(len(r.missing) for _, r in results if r)
    print("标注：" + "、".join(f"{c}×{v}" for c, v in counts.items()) + f"；Claude 补框 {added} 个；没核对 {failed} 帧")
    u = reviewer.usage
    print(f"用量（参考，订阅不按它计费）：输入 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']}（含图片）、输出 {u['output_tokens']} token")
    print(f"下一步：先看 {report}，再用 X-AnyLabeling 打开 {out / 'images'} 修正")


def _perception_label_objects(cfg: Config, args) -> None:
    """--objects：给已经标好人的数据集补标物品（座位 / 篝火 / 乐器 / 先祖）和头顶气泡 typing，见 vision/objlabel.py。"""
    import fnmatch
    import hashlib
    import json
    import shutil

    import cv2

    from .brain import claude
    from .vision import assist, objlabel
    from .vision.augment import SUFFIXES
    from .vision.track import iou
    from .vision.weaklabel import data_yaml

    if args.assist or args.spin or args.from_runs:
        raise SystemExit("--objects 只对已经标好人的数据集跑，不能和 --assist / --spin / --from-runs 一起用")
    classes = cfg.perception.classes
    lacking = [c for c in objlabel.LABEL_CLASSES if c not in classes]
    if lacking:
        raise SystemExit(f"[perception] classes 里缺这几类（{'、'.join(lacking)}）：要带上 typing 和 bench / bonfire / instrument / spirit（新类别加在末尾）")
    root = Path(args.source)
    everything = sorted(root.glob("images/*/*.jpg"))
    if not everything:
        raise SystemExit(f"{root / 'images'} 下面没有图：--objects 要对已经标好人的数据集跑（比如 datasets/sky）")
    images = [p for p in everything if not p.stem.endswith(SUFFIXES)]
    if len(images) < len(everything):  # 增强图是原图的副本：单独核对白花钱、标注还和原图对不上
        print(f"跳过 {len(everything) - len(images)} 张增强图（_blur / _dark）：补完物品后先删掉它们，再重新 perception augment")
    only = getattr(args, "only", None)
    if only:  # 只做这几帧：别的帧不核对、不写回、不进清单（labels/ 照样整个备份）
        patterns = [p.strip() for arg in only for p in arg.split(",") if p.strip()]
        images = [p for p in images if any(fnmatch.fnmatch(p.stem, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns)]
        if not images:
            raise SystemExit(f"--only {' '.join(only)} 一帧也没匹配上：按图片文件名匹配（不带扩展名的 0001 或带扩展名的 0001.jpg 都行，"
                             f"通配 * ? []），增强图（_blur / _dark）不算")
        print(f"--only：只处理 {len(images)} 帧")
    base, env = _brain_env(cfg)  # 没令牌 / 没 claude 先报错
    (root / "data.yaml").write_text(data_yaml(root, classes), encoding="utf-8")  # 先写：中途停下时标注里已经有 6~9 类
    labels = root / "labels"
    if labels.is_dir():  # 写回前整个备份：人物标注是人工修过的
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup, n = root / "_backup" / f"labels-{stamp}", 1
        while backup.exists():  # 同一秒里重跑
            n += 1
            backup = root / "_backup" / f"labels-{stamp}-{n}"
        shutil.copytree(labels, backup)
        print(f"标注先备份到 {backup}")
    detector = None
    if args.model:
        from .vision.detect import make_detector

        p = cfg.perception
        detector = make_detector(args.model, p.classes, p.imgsz, p.low_conf, p.iou, p.device)
    a = cfg.assist
    cmd = assist.assist_command(base, a, system=objlabel.OBJECT_SYSTEM)
    work = assist.assist_workdir()
    reviewer = assist.Reviewer(
        lambda content: claude.one_shot_message(cmd, env, work, content, a.timeout), root / "_assist_objects", a,
        args.model or "none", protocol=objlabel.OBJECTS_PROTOCOL,
    )
    # 每帧写回后记下标注的哈希：重跑时标注没变的帧跳过（清单照样列），被人改过的帧也跳过（人工优先），--recheck 才重新核对
    written = root / "_assist_objects" / "_written"
    digest = lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: E731
    results: list = []
    edited: list[str] = []
    todo = []
    for path in images:
        label = labels / path.parent.name / f"{path.stem}.txt"
        text = label.read_text(encoding="utf-8") if label.is_file() else ""  # 新图还没标注：当空的
        try:
            state = json.loads((written / f"{path.stem}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = None
        if state is not None and not args.recheck:
            if state.get("hash") == digest(text):
                results.append((path.stem, objlabel.review_from_state(state.get("review") or {})))
            else:
                edited.append(path.stem)
            continue
        todo.append(path)
    done = len(images) - len(todo) - len(edited)
    print(f"{len(todo)} 帧交给 Claude（{a.model}）补标物品和气泡：每批 {a.batch} 帧、{a.jobs} 路并发"
          + (f"；{done} 帧上次做过、{len(edited)} 帧你改过，跳过（要重新核对加 --recheck）" if done or edited else ""))
    counts: dict[str, int] = {}
    spirits = 0
    for c0 in range(0, len(todo), _ASSIST_CHUNK):
        frames, where = [], {}
        for path in todo[c0 : c0 + _ASSIST_CHUNK]:
            label = labels / path.parent.name / f"{path.stem}.txt"
            text = label.read_text(encoding="utf-8") if label.is_file() else ""
            frame = imread(path)
            height, width = frame.shape[:2]
            # 已有的物品 / 气泡（重跑 / 手工修过）也当候选：写回时这些行整体替换，不交给 Claude 再核对就丢了
            cands = objlabel.objects_in_labels(text, classes, width, height)
            hints = [objlabel.EXISTING + c for c, _ in cands]
            if detector is not None:
                model = [(d.cls, d.box) for d in detector.detect(frame)
                         if d.cls in objlabel.LABEL_CLASSES and all(iou(d.box, b) < 0.6 for _, b in cands)]
                cands += model
                hints += [f"猜 {c}" for c, _ in model]
            people = objlabel.people_in_labels(text, classes, width, height)
            frames.append(assist.FrameInput(path.stem, frame, [b for _, b in cands], people=people, hints=hints))
            where[path.stem] = (label, text)
        try:
            reviews = reviewer.review(frames)
        except assist.AssistLimit:
            raise SystemExit(f"订阅额度用完了：已核对的帧存在 {root / '_assist_objects'}，额度恢复后重跑同一条命令会接着做") from None
        for f in frames:
            height, width = f.image.shape[:2]
            label, text = where[f.stem]
            review = reviews.get(f.stem)
            results.append((f.stem, review))
            objects = objlabel.apply_object_review(f.candidates, review, f.hints) if review else []
            if review is not None:  # 没核对成的标注不动
                label.parent.mkdir(parents=True, exist_ok=True)
                new = objlabel.rewrite_labels(text, classes, objects, review.spirits, width, height)
                label.write_text(new, encoding="utf-8")
                written.mkdir(parents=True, exist_ok=True)
                (written / f"{f.stem}.json").write_text(json.dumps(
                    {"hash": digest(new), "review": objlabel.review_to_state(review)}, ensure_ascii=False), encoding="utf-8")
                spirits += len(review.spirits)
                for c, _ in objects:
                    counts[c] = counts.get(c, 0) + 1
            (root / "_preview_objects").mkdir(parents=True, exist_ok=True)
            preview = objlabel.draw_objects_preview(f.image, objects, f.people, review.spirits if review else [], f.candidates, review)
            imwrite(root / "_preview_objects" / f"{f.stem}.jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"  {min(c0 + _ASSIST_CHUNK, len(todo))}/{len(todo)} 帧")
    (root / "_assist").mkdir(parents=True, exist_ok=True)
    report = root / "_assist" / "objects.md"
    report.write_text(objlabel.objects_report(sorted(results, key=lambda r: r[0]), edited), encoding="utf-8")
    failed = sum(1 for _, r in results if r is None)
    print("物品 / 气泡：" + ("、".join(f"{c}×{v}" for c, v in counts.items()) or "没有") + f"；人物框改成先祖 {spirits} 个；没核对 {failed} 帧")
    u = reviewer.usage
    print(f"用量（参考，订阅不按它计费）：输入 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']}（含图片）、输出 {u['output_tokens']} token")
    print(f"下一步：先看 {report}，再用 X-AnyLabeling 打开 {root / 'images'} 修正")


def _perception_label(cfg: Config, args) -> None:
    """用现有识别器给录下来的画面出弱标注（名字标签 + 圆圈），写成 YOLO 数据集。"""
    import cv2

    from .vision.bubbles import Rect
    from .vision.ocr import make_ocr
    from .vision.weaklabel import (
        CLASH_REASON, data_yaml, dataset_clash, hard_images, label_items, merge_labels, ring_labels, split_of, with_self,
        write_sample, yolo_line,
    )

    if args.objects:
        return _perception_label_objects(cfg, args)
    if getattr(args, "only", None):  # 别处直接构造的 Namespace 可能没有这一项
        raise SystemExit("--only 只配 --objects 用")
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
    else:  # 一段录像 → "<录像>_<文件名>"，不然不同录像的同名帧会互相覆盖
        items = label_items(root, _images(args.source))
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
    skipped: list[str] = []
    for n, (path, stem) in enumerate(items, 1):
        frame = imread(path)
        if dataset_clash(out, stem, frame) is not None:  # 先查一遍，省得白跑 OCR
            skipped.append(stem)
            continue
        height, width = frame.shape[:2]
        boxes = _weak_boxes(cfg, args, frame, ocr, icons, names)
        weak = len(boxes)
        if detector is not None:
            boxes = merge_labels(boxes, predicted[stem] if stem in predicted else detector.detect(frame))
        if stem in selves:
            boxes = with_self(boxes, selves[stem])
        if detector is not None and icons is not None:  # 模型预标注出了人：顺带补他们头顶的圆圈
            boxes += [("social_ring", r) for r in ring_labels(frame, boxes, icons)]
        lines = [yolo_line(index[c], box, width, height) for c, box in boxes if c in index]
        if write_sample(out, split_of(stem, args.val), stem, frame, lines) is not None:
            skipped.append(stem)
            continue
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
    _print_skipped(skipped, CLASH_REASON)
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
            _run_brain(cfg, run, None, args.duration, viewer, no_emotes=args.no_emotes)  # 真机世界在里面检查完令牌后再建
        else:
            _run_agent(cfg, run, args.no_emotes, args.duration, viewer)
    finally:
        if viewer is not None:
            viewer.stop()
        run.close()


def cmd_sandbox(cfg: Config, args) -> None:
    """大脑沙盒（spec 2026-09-30-brain-sandbox）：不开 MuMu，真的大脑 + 身体 + 内心层接一个假世界，只给 JSON 接口（页面在管理面板里）。

    总是 live，但记忆只写 [sandbox] dir 下的 memory/（真的 memory/ 永远不碰）；模拟时钟从 --start 算出来。"""
    import _thread

    from .brain.trace import BrainTrace
    from .console import watchdog
    from .sandbox import clock as sandbox_clock
    from .sandbox.control import SandboxControl
    from .sandbox.server import SandboxServer
    from .brain.transcript import Transcript
    from .sandbox.world import Scene, sandbox_world

    sb = cfg.sandbox
    if isinstance(sb.wake_hour, bool) or not isinstance(sb.wake_hour, int) or not 0 <= sb.wake_hour <= 23:
        raise SystemExit(f"[sandbox] wake_hour 要是 0~23 的整数，现在是 {sb.wake_hour!r}")
    root = Path(sb.dir)
    memory = root / "memory"
    if cfg.reply.memory_dir:  # 沙盒只写 sandbox/memory/：和真的记忆目录重合就不起（否则会写进真记忆）
        from .console.sandbox_view import check_separate

        try:
            check_separate(root, Path(cfg.reply.memory_dir))
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    memory.mkdir(parents=True, exist_ok=True)  # 第一次：空记忆（管理面板「重置记忆」才从 memory/ 复制）
    cfg.reply.dry_run = False  # 沙盒总是 live：记忆、内心层都真的写，但只写 sandbox/memory/
    cfg.reply.memory_dir = str(memory)
    cfg.run.save_frames = False  # 灰图不用存
    cfg.brain.enabled = True
    floor, now = sandbox_clock.floor_time(root), time.time()
    try:
        start = sandbox_clock.resolve_start(args.start, floor, now, sb.wake_hour)
    except ValueError as exc:
        raise SystemExit(f"沙盒起始时间不对：{exc}") from None
    sim = sandbox_clock.SimClock(offset=start - now)
    request_exit = watchdog.once(_thread.interrupt_main)  # 看门狗和 /shutdown 共用：只中断一次，不打断收尾（写日记）
    if args.parent_pid is not None:
        watchdog.watch_parent(args.parent_pid, request_exit)
    port = args.port if args.port is not None else sb.port
    server = SandboxServer(port, request_exit)
    server.trace = BrainTrace()
    try:
        url = server.start()  # 先起接口：/status 立刻可用，管理面板据此判断子进程起来了
    except OSError as exc:
        raise SystemExit(f"沙盒接口起不来（127.0.0.1:{port}）：{exc}；端口可能被上次留下的沙盒占着") from exc
    run = RunDir.create(cfg, "sandbox")
    run.attach_log()
    log.info("本次运行的日志和截图: %s", run.path.resolve())
    log.info("沙盒接口：%s；沙盒时间从 %s 开始（记忆在 %s）", url, sandbox_clock._fmt(sim.wall()), memory.resolve())
    transcript = Transcript(sim)
    scene = Scene()
    world = sandbox_world(cfg, sim, transcript, scene)

    def ready(parts) -> None:
        body = parts.body
        server.inner = lambda: body.call(body.inner_snapshot, timeout=3)
        server.forget = lambda k, t, w, tp: body.call(lambda: body.forget(k, t, w, tp), timeout=3)
        server.control = SandboxControl(parts, world, sim, transcript, scene)

    try:
        _run_brain(cfg, run, world, getattr(args, "duration", 0.0) or 0.0, None, on_ready=ready, trace=server.trace)
    finally:
        try:
            sandbox_clock.save(root / "clock.json", sim.wall())
        except OSError:
            log.exception("沙盒时钟没存上")
        server.stop()
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
    ime = _ime_switch(cfg, dev)
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
        ime.stop()


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


def _fallback_brain(cfg: Config, toolbox, prompt: str):
    """DeepSeek 备用大脑（spec 2026-10-03-deepseek-fallback-brain §1）：额度用完时 cli 切过去用。

    没开 / 没 Key 返回 None（退化成现在的纯文字备用回复）。"""
    from .brain.deepseek import FALLBACK_NOTE, DeepSeekBrain, build_client
    from .brain.llm_tools import openai_tools

    if not cfg.brain.fallback:
        return None
    try:
        client = build_client(cfg.llm)
    except (RuntimeError, ImportError) as exc:
        log.warning("备用大脑没开（%s）", exc)
        return None
    return DeepSeekBrain(
        client, prompt + "\n\n" + FALLBACK_NOTE, toolbox, openai_tools(toolbox, blind=True),
        model=cfg.llm.model, temperature=cfg.llm.temperature, max_tokens=cfg.brain.fallback_max_tokens,
        max_steps=cfg.brain.max_steps, turn_timeout=cfg.brain.turn_timeout,
    )


def _game_world(cfg: Config, run: RunDir, no_emotes: bool = False):
    """真机世界（brain-sandbox spec §2）：接 MuMu 的设备、OCR 读聊天、身边识别、轮盘、镜头……原样搬自 _run_brain。

    仍按模块名调 _device / _build_reader / _panels……（测试的 monkeypatch 照样生效）。"""
    from .brain.locomotion import Locomotion
    from .brain.world import World
    from .chat.sender import ChatSender

    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    panel = _panel(cfg, dev, reader)
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _scene_watcher(cfg, icons, dev, run=run, light=True) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel=panel)
    emotes = _build_emotes(cfg, dev, panel, no_emotes)
    camera = _camera(cfg, dev, panel)
    friend_checker = _friend_checker(cfg, dev, panel) if cfg.friend_check.enabled else None
    panels, panel_ops = _panels(cfg, dev, reader)

    def close() -> None:
        _stop_scene(env)
        if panels is not None:
            panels.close()

    ime = _ime_switch(cfg, dev)  # 最后才切：前面建东西出错时不用切回
    return World(
        device=dev, reader=reader, self_filter=self_filter, panel=panel, env=env, social=social, emotes=emotes, camera=camera,
        locomotion=Locomotion(dev, cfg.brain.move_step), sender=ChatSender(dev, cfg.sender, _screen_size_fn(dev)),
        friend_checker=friend_checker, panels=panels, panel_ops=panel_ops, close=close, restore=ime.stop,
    )


def _run_brain(
    cfg: Config, run: RunDir, world=None, duration: float = 0.0, viewer=None, on_ready=None, trace=None, *, no_emotes: bool = False,
) -> None:
    """统管大脑：身体在当前线程跑（独占设备）；大脑（常驻 Claude Code）和眼睛各一个后台线程；工具经本机 MCP 服务。

    world：接世界的东西（brain.world.World）；None = 真机（_game_world，在检查完令牌之后才建、才连设备）。
    时间一律从 world 取（沙盒是模拟时钟）。on_ready(BrainParts)：身体建好、线程启动前调一次。
    trace：大脑时间线（沙盒没有 viewer 时由调用方传进来）。"""
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
    from .brain.world import BrainParts
    from .chat.llm import make_llm
    from .chat.responder import Responder

    base, claude_vars = _brain_env(cfg)  # 先检查令牌和 claude 命令，缺了早点报错
    if world is None:
        world = _game_world(cfg, run, no_emotes)
    wall, clock = world.wall, world.clock
    env = world.env
    try:
        store = notes = None
        if cfg.reply.memory_dir:  # dry-run 也读人设和记忆（看大脑的表现要用），但不写
            from .chat.memory import MemoryStore, NotesKeeper

            store = MemoryStore(cfg.reply.memory_dir)
            if not cfg.reply.dry_run:
                # 记忆整理也走 Claude（订阅）：随手记、整理 notes.md 各起一次性 claude -p，在记忆后台线程里跑
                memory_llm = ClaudeLlm(base, claude_vars, cfg.brain.memory_model, run.path / "brain" / "memory", cfg.brain.memory_timeout)
                notes = NotesKeeper(memory_llm, store, cfg.reply.persona, cfg.reply.notes_every, wall=wall)
        live_store = None if cfg.reply.dry_run else store
        ledger = _inner_ledger(cfg, store, wall())
        mind, reflector = _inner_mind(cfg, ledger, base, claude_vars, run, wall(), clock)
        persona = _inner_persona(cfg, ledger, wall())
        mind_log = _inner_log(ledger, reflector, wall())
        events = EventQueue(clock=clock)
        # 大脑离线时的备用回复：在身体线程里调，给短超时、不重试；不带记忆存储，免得和身体重复记聊天记录
        fallback = Responder(make_llm(dataclasses.replace(cfg.llm, timeout=10.0, max_retries=0)), cfg.reply)
        body = Body(
            cfg, world.device, world.reader, world.sender, world.self_filter, events,
            env=env, social=world.social, emotes=world.emotes, camera=world.camera, locomotion=world.locomotion,
            friend_checker=world.friend_checker, fallback=fallback, store=live_store, notes=notes, run=run, viewer=viewer,
            panel=world.panel, panels=world.panels, panel_ops=world.panel_ops,
            ledger=ledger, mind=mind, reflector=reflector, persona=persona, mind_log=mind_log, clock=clock, wall=wall,
        )
        work = run.path / "brain"
        if world.describe is not None:
            describe = world.describe
        else:
            def describe(content):
                return one_shot(eyes_command(base, cfg.brain), claude_vars, work / "eyes", content, cfg.brain.eyes_timeout)
        if world.text_only:  # 沙盒：没有画面，不给位置说明
            note = lambda now, s: ""  # noqa: E731
        else:
            note = (lambda now, s: scene_note(env, now, s)) if hasattr(env, "strangers") else None
        eyes = Eyes(
            cfg.brain,
            describe=describe,
            frame=lambda: body.last_frame,
            labels=lambda: dict(env.labels) if env else {},
            blackout=lambda: body.blackout,
            label_keep=cfg.env.interval * 2 + 1,
            clock=clock,
            note=note,
            proactive=cfg.proactive,
            busy=(lambda now: bool(env.nearby(now))) if env else (lambda now: False),
            on_news=body.news,
        )
        body.friend_names = _friend_names(cfg)
        if store is not None:  # 反思用的人设、笔记和好友名单（dry-run 也读）
            body.profile_text = lambda: store.profile() or cfg.reply.persona
            body.memory_notes = lambda: f"{store.notes()}\n{store.inbox()}"
            body.friends_text = store.friends
        body.recent_changes = _recent_changes(cfg)
        events.subscribe(eyes.notice)
        # recall 只读，dry-run 也给
        toolbox = ToolBox(body, eyes, cfg.brain.max_steps, cfg.brain.max_says, memory=store, text_only=world.text_only,
                          sandbox=world.name == "sandbox", backstage=cfg.backstage.enabled, call=_call_enabled(cfg, env))
        wardrobe = _wardrobe(cfg, env, ledger, world, base, claude_vars, work, clock)
        server = SkyServer(toolbox)
        server.start()
    except BaseException:
        world.close()
        world.restore()
        raise
    if trace is None and viewer is not None:
        trace = viewer.brain or BrainTrace()  # 网页上的大脑时间线（一般 _viewer 已经挂好）
    prompt = brain_prompt(
        cfg.reply, store, quick_around=hasattr(env, "sweep"), panel_auto=cfg.panel.mode == "auto", history_turns=cfg.brain.history_turns,
        now=wall(), proactive=cfg.proactive.enabled, bubble=cfg.reflex.enabled and cfg.reflex.bubble,
        days=_days_prompt(ledger, cfg, wall()), inner=ledger is not None, mind=reflector is not None,
        persona_text=_persona_prompt(persona), temper=ledger is not None and cfg.inner.persona,
        appearance=getattr(env, "appearance", None) is not None, backstage=_backstage_prompt(cfg, store),
        lull=cfg.lull.enabled, call=_call_enabled(cfg, env),
    )
    session = BrainSession(
        base, claude_vars, work / "session", server.url, prompt,
        cfg.brain.model, cfg.brain.effort, cfg.brain.turn_timeout,
        on_message=trace.chain(log_brain_message) if trace is not None else log_brain_message,
    )
    fallback_session = _fallback_brain(cfg, toolbox, prompt)
    brain = Brain(
        cfg.brain, cfg.chat, session, toolbox, events, nearby=env.nearby if env else (lambda now: []),
        eyes=eyes, clock=clock, wall=wall, run=run, store=live_store, trace=trace, slow=lambda: body.effects().slow,
        fallback_session=fallback_session,
    )
    if cfg.brain.force_fallback and fallback_session is not None:
        brain.session = fallback_session
        brain.on_fallback = True
    if viewer is not None:
        viewer.brain = trace
        from .brain.manual import ManualControl
        from .vision.viewer import LOCAL_HOSTS

        if cfg.viewer.host in LOCAL_HOSTS:  # 手动控制：网页上直接让身体说话 / 做动作 / 转视角，只给本机
            viewer.control = ManualControl(body, eyes, events)
            # 内心页（spec 2026-09-30-inner-viewer §2）：都在身体线程里做，等 3 秒
            viewer.inner = lambda: body.call(body.inner_snapshot, timeout=3)
            viewer.forget = lambda k, t, w, tp: body.call(lambda: body.forget(k, t, w, tp), timeout=3)
            if world.name != "sandbox":  # 真机聊天记录（spec 2026-10-01-console-live-page §3.2）：沙盒有自己的
                from types import SimpleNamespace

                from .brain.transcript import Transcript, event_line

                chat = Transcript(SimpleNamespace(wall=wall))
                body.on_line = lambda kind, text, who, why: chat.add(kind, text, who, why=why)

                def _event(kind: str, text: str, who: str) -> None:
                    line = event_line(kind, text, who)
                    if line is not None:
                        chat.add("event", line)

                events.tap(_event)
                viewer.chat = chat
        else:
            log.warning("可视化网页开在局域网（%s）：局域网模式下关掉了手动控制", cfg.viewer.host)
    stop = threading.Event()
    brain_thread = threading.Thread(target=brain.run, args=(stop,), name="brain", daemon=True)
    eyes_thread = threading.Thread(target=eyes.run, args=(stop,), name="eyes", daemon=True)
    wardrobe_thread = (
        threading.Thread(target=wardrobe.run, args=(stop,), name="wardrobe", daemon=True) if wardrobe is not None else None
    )
    body.brain_offline = lambda now: brain.offline(now) or not brain_thread.is_alive()
    body.brain_busy = lambda: brain.chat_turn
    body.brain_turn = lambda: brain.last_turn  # 替大脑开的输入框：开框之后的那一轮结束了没说话就关
    if body.lulls is not None:  # 冷场时大脑心里想的（“心里：”）记到身体那边
        def hand_over(text: str) -> None:
            # 不等身体：身体在自动喊（约 4 秒）时 call 会超时报 ERROR、丢掉这轮的“心里”；退出时身体已停，直接丢掉
            began = brain.last_turn[0]
            if not body.post(lambda: body.mused(text, began)):
                log.debug("身体停了，这轮的文字不交了")

        brain.on_text = hand_over
    if on_ready is not None:
        try:
            on_ready(BrainParts(
                body=body, eyes=eyes, events=events, brain=brain, trace=trace, reflector=reflector, ledger=ledger, store=store,
                mind_log=mind_log,
            ))
        except Exception:
            log.exception("on_ready 出错")
    brain_thread.start()
    eyes_thread.start()
    if wardrobe_thread is not None:
        wardrobe_thread.start()
    try:
        body.run(duration, stop)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        stop.set()
        world.close()  # 停 env、关面板识别
        body.shutdown()  # 先复原镜头、恢复轮盘、让排队的命令失败：不等大脑
        world.restore()  # 切回用户的输入法（身体收尾时可能还要关输入框，放在它后面）
        brain_thread.join(timeout=5)
        if wardrobe_thread is not None:
            _finish_wardrobe(wardrobe_thread, body)
        if ledger is not None:
            try:
                ledger.checkpoint(wall())  # 写经过可能很慢、被强杀：先落账，别被当成意外断了
            except Exception:
                log.exception("内心账本落账出错")
        summary = ""
        if reflector is not None:  # 下线前的最终反思：写日记和要点，代替大脑写经过（dry-run 照跑、不写盘）
            summary = _final_reflection(cfg, body, reflector, ledger, live_store, wall())
        elif live_store is not None and not brain_thread.is_alive():
            try:
                summary = brain.farewell()  # 把这次的经过记进 inbox.md
            except Exception:
                log.exception("退出前写经过失败")
        if ledger is not None:
            try:
                ledger.close(summary, wall())  # 这一次记进 days.jsonl（dry-run 不写）
            except Exception:
                log.exception("内心账本收尾出错")
        session.close()
        server.stop()


WARDROBE_JOIN = 5.0  # 下线时等描述器手上那一个描述回来最多几秒（不为它拖住下线）


def _finish_wardrobe(thread: threading.Thread, body) -> None:
    """下线：等描述器线程停下（最多 WARDROBE_JOIN 秒），再把这期间回来的装扮写进关系卡（账本收尾之前）。
    身体主循环已经停了，这里在主线程替它取一次；出错只记日志。"""
    thread.join(timeout=WARDROBE_JOIN)
    if thread.is_alive():
        log.info("装扮描述还没回来，不等了")
    try:
        body._watch_outfits()
    except Exception:
        log.exception("下线时记装扮出错")


def _wardrobe(cfg: Config, env, ledger, world, base, claude_vars, work, clock):
    """认装扮（spec 2026-10-01-appearance）：关系卡里的旧外观载入记忆簿；[appearance] describe 开着、不是沙盒时建描述器挂到感知层上。
    感知层没挂记忆簿时什么都不做、返回 None。描述器的钟和感知层的帧时间同一个（world.clock）。"""
    book = getattr(env, "appearance", None)
    if book is None:
        return None
    if ledger is not None:
        try:
            book.load_cards(ledger.all_outfits())
        except Exception:
            log.exception("关系卡里的装扮载入出错，这次从头认")
    a = cfg.appearance
    if not a.describe or world.name == "sandbox":
        return None
    from .brain.claude import one_shot
    from .vision.wardrobe import Wardrobe, wardrobe_command

    cmd = wardrobe_command(base, a.describe_model)
    wardrobe = Wardrobe(
        a, describe=lambda content: one_shot(cmd, claude_vars, work / "wardrobe", content, a.describe_timeout),
        on_done=env.on_described, clock=clock,
    )
    env.wardrobe = wardrobe
    log.info("描述装扮：%s，每小时最多 %d 次", a.describe_model, a.describe_max)
    return wardrobe


def _inner_ledger(cfg: Config, store, now: float | None = None):
    """内心账本（spec 2026-09-30-inner-phase1）：[inner] enabled 且有记忆目录才有；出错就不记（团子照常跑）。"""
    if not cfg.inner.enabled or store is None:
        return None
    from .inner import open_ledger

    try:
        return open_ledger(
            cfg.inner, store.dir / "inner", _friend_names(cfg), store.history.all, persist=not cfg.reply.dry_run,
            now=time.time() if now is None else now, outfit_keep=cfg.appearance.outfit_keep,
        )
    except Exception:
        log.exception("内心账本打不开，这次不记")
        return None


def _inner_mind(cfg: Config, ledger, base, claude_vars, run, now: float | None = None, clock=time.monotonic):
    """内心层第 2 期：(Mind, Reflector)；没有账本或 [inner] reflect = false 时 (None, None)。"""
    if ledger is None or not cfg.inner.reflect:
        return None, None
    from .brain.claude import ClaudeLlm
    from .inner.reflect import PERSONA_SYSTEM, REFLECT_SYSTEM, Reflector

    try:
        mind = ledger.store.load_mind(quarantine=ledger.persist)
        if mind.wake(time.time() if now is None else now, cfg.inner.rest_gap):
            log.info("睡过一觉：心情回到平常")
        llm = ClaudeLlm(base, claude_vars, cfg.inner.reflect_model, run.path / "brain" / "reflect", cfg.inner.reflect_timeout)
        system = REFLECT_SYSTEM + "\n\n" + PERSONA_SYSTEM if cfg.inner.persona else REFLECT_SYSTEM
        return mind, Reflector(cfg.inner, llm, clock=clock, system=system)
    except Exception:
        log.exception("反思打不开，这次不反思")
        return None, None


def _inner_persona(cfg: Config, ledger, now: float | None = None):
    """内心层第 3 期：性格档案；没有账本或 [inner] persona = false 时 None。读不了：live 改名放一边、都从空的开始。"""
    if ledger is None or not cfg.inner.persona:
        return None
    now = time.time() if now is None else now
    try:
        persona = ledger.store.load_persona(quarantine=ledger.persist)
        for why in persona.prepare(now, _friend_names(cfg)()):
            log.info("性格档案：%s", why)
        persona.fade(now, cfg.inner)
        return persona
    except Exception:
        log.exception("性格档案打不开，这次不用")
        return None


def _inner_log(ledger, reflector, now: float | None = None):
    """内心流水账（spec 2026-09-30-inner-viewer §1）：开了反思才记；live 写 mind_log.jsonl，启动时删掉 30 天前的行。"""
    if ledger is None or reflector is None:
        return None
    from .inner.log import MindLog

    try:
        mind_log = MindLog(ledger.store.dir / "mind_log.jsonl", persist=ledger.persist)
        mind_log.trim(time.time() if now is None else now)
        return mind_log
    except Exception:
        log.exception("内心流水账打不开，这次不记")
        return None


def _persona_prompt(persona) -> str:
    """系统提示词「你攒下的性格」；拼出错就不写这一节。"""
    if persona is None:
        return ""
    try:
        return persona.section()
    except Exception:
        log.exception("拼「你攒下的性格」一节出错")
        return ""


def _backstage_prompt(cfg: Config, store, now: float | None = None, run=None, repo=None) -> str:
    """「幕后」一节（spec 2026-10-01-backstage）：[backstage] enabled 才有；live 时记下看到了哪个提交。拼出错就不写这一节。

    now 默认真实时间：提交时间是真实时间，别用世界的钟（沙盒的 wall() 会超前）。"""
    if not cfg.backstage.enabled:
        return ""
    from .brain.backstage import changelog, git_runner, read_seen, repo_root, section, write_seen

    try:
        mark = store.dir / "inner" / "backstage.json" if store is not None else None
        if run is None:
            repo = repo if repo is not None else repo_root()
            run = git_runner(repo) if repo is not None else None
        lines, head = [], None
        if run is not None:
            lines, head = changelog(run, read_seen(mark) if mark is not None else None, time.time() if now is None else now,
                                    cfg.backstage.changelog_max, cfg.backstage.changelog_days)
        if head and mark is not None and not cfg.reply.dry_run:
            write_seen(mark, head)
        return section(cfg.brain.owner_name, cfg.brain.model, cfg.brain.eyes_model, cfg.inner.reflect_model, lines)
    except Exception:
        log.exception("拼「幕后」一节出错")
        return ""


def _recent_changes(cfg: Config, run=None, repo=None, now=time.time):
    """introspect(改动) 用（spec 2026-10-01-backstage）：不看标记、按最近 changelog_days 天取提交；幕后关着返回 None。
    提示词里的更新记录只给没告诉过的，告诉过一次再问就得靠这个查。时间用真实时间（同 _backstage_prompt）。"""
    if not cfg.backstage.enabled:
        return None
    from .brain.backstage import changelog, git_runner, repo_root

    if run is None:
        repo = repo if repo is not None else repo_root()
        run = git_runner(repo) if repo is not None else None

    def lines() -> list[str]:
        if run is None:
            return []
        return changelog(run, None, now(), cfg.backstage.changelog_max, cfg.backstage.changelog_days)[0]

    return lines


def _final_timeout(cfg: Config) -> float:
    """下线反思最多等多久：管理面板 stop_timeout 到了会强杀，留 25 秒给身体收尾和等大脑线程。"""
    return min(cfg.inner.reflect_timeout, max(10.0, cfg.console.stop_timeout - 25))


def _final_reflection(cfg: Config, body, reflector, ledger, live_store, now: float | None = None) -> str:
    from .inner import finish_reflection

    try:
        if hasattr(reflector.llm, "timeout"):
            reflector.llm.timeout = _final_timeout(cfg)
        result = reflector.final(body.reflect_materials(True))
        return finish_reflection(
            result, body.mind, ledger.store, live_store, ledger.cards, body._safe_friends(), ledger.persist,
            time.time() if now is None else now, cfg.inner,
            persona=body.persona, soft=body.soft_names_this_session(), mind_log=body.mind_log, energy=body._energy,
        )
    except Exception:
        log.exception("下线前的反思出错")
        return ""


def _days_prompt(ledger, cfg: Config | None = None, now: float | None = None) -> str:
    if ledger is None:
        return ""
    try:
        diaries = ledger.store.last_diaries(cfg.inner.diary_prompt) if cfg is not None and cfg.inner.reflect else None
        return ledger.days_prompt(time.time() if now is None else now, diaries=diaries)
    except Exception:
        log.exception("拼「日子」一节出错")
        return ""


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
        q.add_argument("--device", choices=["cuda", "dml", "cpu"])
        q.add_argument("--imgsz", type=int)
        if name == "bench":
            q.add_argument("--far-crops", type=int, help="远处二次检测每帧最多几块（覆盖 perception.far_crops，0 = 关）")
            q.add_argument("--images", help="用这个目录 / 这张图测（默认实时截图）")
            q.add_argument("-n", type=int, default=200, help="测多少帧")
            q.add_argument("--attrs", action="store_true", help="再测一遍加第二层外形头（每帧最多 [attrs] max_crops 张人物裁图）的 fps")
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
    q.add_argument("--objects", action="store_true",
                   help="物品模式：source 是已经标好人的数据集，Claude 补标座位 / 篝火 / 乐器 / 先祖和头顶气泡 typing"
                        "（--model 的这几类框当候选；写回前备份 labels/）")
    q.add_argument("--recheck", action="store_true", help="配 --objects：上次做过的、你改过的帧也重新核对（默认跳过）")
    q.add_argument("--only", action="append", metavar="通配",
                   help="配 --objects：只处理文件名匹配的帧（fnmatch，对不带扩展名的文件名或带扩展名的文件名都试，比如 0001 / 'rec_*' / '*.jpg'）；"
                        "可以给多次或用逗号分隔；别的帧不核对、不写回、不进清单。一帧都没匹配上就报错退出")
    q = psub.add_parser("compare", help="同一批录像上对比现有的整图 OCR 和 YOLO（认出率、请求延迟、陌生人 / 走开事件、耗时）")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="模型文件（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("--imgsz", type=int)
    q.add_argument("--interval", type=float, default=3.0, help="现有方案多久扫一次（同 env.interval）")
    q.add_argument("--far-crops", type=int, help="远处二次检测每帧最多几块（覆盖 perception.far_crops，0 = 关；开关各跑一次对比）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/compare/<时间>）")
    q = psub.add_parser("track-eval", help="同一批录像上对比追踪升级前后：轨迹断开和原因、假走开、确认冤枉、接回对错、运动方向")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="模型文件（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("--imgsz", type=int)
    q.add_argument("--fps", type=float, default=6.5, help="按录像时间每秒抽几帧（模拟 run 时身体截图，默认 6.5）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/track-eval/<时间>）")
    q = psub.add_parser("appearance-eval", help="认装扮的离线标定：录像上统计同一个人 / 不同人的外观相似度，给出建议的门槛，藏标签重放")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("--embed", help="外观特征：color 或 .onnx 路径（默认 appearance.model）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/appearance-eval/<时间>）")
    q = psub.add_parser("halo-eval", help="呼唤光圈标定：录像上画每个人头顶的亮度变化曲线，给建议的 [call] halo_rise")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/q-call-20260930-c）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/halo-eval/<时间>）")
    q = psub.add_parser("unknown-names", help="汇总最近几次运行里读到、但不在好友名单里的名字（只列出，不改 friends.md）")
    q.add_argument("--runs", default="runs", help="运行目录的上级（默认 runs/）")
    q.add_argument("--last", type=int, default=5, help="看最近几次运行")
    q = psub.add_parser("clips", help="动作识别的数据：录像按人物轨迹切成 16 帧的片段（人工再分到 <动作>/ 目录）")
    q.add_argument("source", help="record 录的目录（record --fps 8，文件名里带时间）")
    q.add_argument("-o", "--output", help="输出目录（默认 <[gesture] dataset>/_unlabeled）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--force", action="store_true",
                   help="这段录像切过也接着切：数据目录里哪儿都还没有的片段才写（已标过、挪走的不会再造一份）")
    q = psub.add_parser("crops", help="第二层外形头的数据：从数据集 / 录像 / 难例目录裁人物图到 datasets/attrs；--writeback 把标注页确认的人物写回数据集")
    q.add_argument("source", nargs="+", help="来源目录：含 images/ 和 labels/ 的是数据集，其余当图片目录（支持 runs/*/hard 这样的通配）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--conf", type=float, default=0.2, help="检测置信度下限")
    q.add_argument("--out", default="datasets/attrs", help="输出目录")
    q.add_argument("--writeback", action="store_true",
                   help="不裁图：把 form/<人形类>/ 里确认过的数据集人物框写回 labels/（先备份到 <数据集>/_backup/）")
    q = psub.add_parser("attrs-label", help="第二层外形头的 Claude 初分：_unlabeled/ 里的裁图每 16 张拼成 4×4 一张图，结果写进 _unlabeled/claude.json")
    q.add_argument("source", nargs="?", help="数据目录（默认 datasets/attrs）")
    q.add_argument("--recheck", action="store_true", help="已有 claude.json 的裁图也重新初分")
    q = psub.add_parser("attrs-train", help="训练外形头：form/ 里确认过的裁图 -> DINOv2 特征 + numpy 线性头 -> models/attrs-<日期>.npz，整帧回放评估、写报告")
    q.add_argument("data", nargs="?", help="数据目录（默认 datasets/attrs）")
    q.add_argument("--out", help="模型输出路径（默认 models/attrs-<日期>.npz；是 [attrs] model 时要加 --force）")
    q.add_argument("--force", action="store_true", help="允许 --out 直接覆盖 [attrs] model 正在用的模型")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"], help="主干提特征用的后端（默认 [attrs] device，空 = 跟 [perception]）")
    q = psub.add_parser("attrs-eval", help="外形头只做整帧回放：数据集 images/val 上比较纯 YOLO 和加外形头复核，写报告")
    q.add_argument("dataset", help="YOLO 数据集目录（含 images/val 和 labels/val）")
    q.add_argument("--model", required=True, help="外形头 .npz")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q = psub.add_parser("gesture-label", help="动作片段的 Claude 初分：每段 16 帧拼成 4×4 一张图，结果写进片段目录的 claude.json")
    q.add_argument("source", nargs="?", help="片段目录的上级（默认 <[gesture] dataset>/_unlabeled）")
    q.add_argument("--recheck", action="store_true", help="已有 claude.json 的片段也重新初分")
    q.add_argument("--blind", action="store_true",
                   help="不看录像名（不给录像提示），结果写进 claude-blind.json（和 claude.json 并存，标注页优先显示它），最后打印两次一致多少段")
    q = psub.add_parser("gesture-train", help="训练动作模型：DINOv2-small 冻住 + 时序头，导出 ONNX（不覆盖 gesture.onnx）、验证集评估、写报告")
    q.add_argument("data", nargs="?", help="数据目录（默认 [gesture] dataset）")
    q.add_argument("--epochs", type=int, default=60, help="最多训练几轮（验证集 F1 10 轮不涨就停）")
    q.add_argument("--out", help="ONNX 输出路径（默认 models/gesture-<日期>.onnx；是 [gesture] model 时要加 --force）")
    q.add_argument("--force", action="store_true", help="允许 --out 直接覆盖 [gesture] model 正在用的模型")
    q.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="训练用的设备（没 CUDA 自动退回 CPU）")
    q = psub.add_parser("gesture-eval", help="在分好类的片段（<数据目录>/<动作>/<片段>/）上评估动作模型的精确率 / 召回率")
    q.add_argument("data", help="数据目录，比如 datasets/gesture")
    q.add_argument("--model", help="动作模型（默认 gesture.model）")
    q.add_argument("--all", action="store_true", help="评全部片段（默认有 _split.json 时只评验证集）")
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

    p = sub.add_parser("catalog", help="装扮图鉴：在录像上试跑收集（定门槛用）")
    psub = p.add_subparsers(dest="action", required=True)
    q = psub.add_parser("collect", help="录像上跑感知层 + 图鉴收集器，输出存下的图、candidates.jsonl、sheet.jpg")
    q.add_argument("source", help="record 录的目录（文件名里带时间，比如 tmp/record/<时间>）")
    q.add_argument("--model", help="YOLO 模型（默认 perception.model）")
    q.add_argument("--device", choices=["cuda", "dml", "cpu"])
    q.add_argument("--imgsz", type=int)
    q.add_argument("--fps", type=float, default=6.5, help="按录像时间每秒抽几帧（模拟 run 时身体截图，默认 6.5）")
    q.add_argument("-o", "--output", help="输出目录（默认 tmp/catalog/<时间>）")
    p.set_defaults(func=cmd_catalog)

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
    p.add_argument("--port", type=int, help="面板端口（默认 [console] port = 19390）")
    p.add_argument("--no-browser", action="store_true", help="不打开浏览器")
    p.set_defaults(func=cmd_console)

    p = sub.add_parser("sandbox", help="大脑沙盒：不开模拟器，真的大脑 + 内心层接假世界，只给 JSON 接口（管理面板「沙盒」页用）")
    p.add_argument("--port", type=int, help="接口端口（默认 [sandbox] port = 19392）")
    p.add_argument("--no-browser", action="store_true", help="（沙盒没有自己的页面，只为和 run 一致）")
    p.add_argument("--parent-pid", type=int, help="这个进程没了就自己退出（管理面板用）")
    p.add_argument("--start", default="resume", help="沙盒时间从哪开始：resume 接着上次 / sleep 睡一晚 / HH:MM / \"YYYY-MM-DD HH:MM\"")
    p.add_argument("--duration", type=float, default=0.0, help="跑多少秒后自动下线（默认一直跑，调试用）")
    p.set_defaults(func=lambda cfg, args: cmd_sandbox(cfg, args))

    p = sub.add_parser("run", help="启动团子（默认接统管大脑、dry-run；--no-brain 是调试用的普通模式）")
    live = p.add_mutually_exclusive_group()
    live.add_argument("--live", action="store_true", help="真的发送消息")
    live.add_argument("--dry-run", action="store_true", help="只打印不发送（覆盖 config.toml 的 reply.dry_run）")
    p.add_argument("--echo", action="store_true", help="不调模型，原样回显（联调用）")
    p.add_argument("--duration", type=float, default=0.0, help="跑多少秒后自动结束（默认一直跑）")
    p.add_argument("--no-emotes", action="store_true", help="这次不做动作")
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
