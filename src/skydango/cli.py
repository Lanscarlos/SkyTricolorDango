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


def _scene_watcher(cfg: Config, icons=None, dev=None):
    """[env] 打开时"身边有谁"由谁来认：[perception] 打开就用 YOLO 感知层，否则用原来的定时整图 OCR。"""
    if not cfg.perception.enabled:
        return _env_watcher(cfg, icons=icons)
    from .vision.detect import make_detector
    from .vision.ocr import make_ocr
    from .vision.perception import PerceptionWatcher

    p = cfg.perception
    detector = make_detector(p.model, p.classes, p.imgsz, p.conf, p.iou, p.device)
    log.info("YOLO 感知层：%s（%s），最多 %.0f fps，帧来自%s", p.model, "、".join(getattr(detector, "providers", [])),
             p.fps, "感知线程自己截图" if p.capture == "own" else "身体主循环")
    return PerceptionWatcher(
        detector, make_ocr(cfg.ocr.engine, p.ocr_threads), p, cfg.env, _friend_names(cfg), cfg.vision.log_roi,
        icons=icons, capture=dev.screenshot if dev is not None else None,
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
    from .vision.perception import PerceptionWatcher

    p = cfg.perception
    for key in ("model", "device", "imgsz"):
        if getattr(args, key, None):
            setattr(p, key, getattr(args, key))
    detector = make_detector(p.model, p.classes, p.imgsz, p.conf, p.iou, p.device)
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


def _perception_label(cfg: Config, args) -> None:
    """用现有识别器给录下来的画面出弱标注（名字标签 + 圆圈），写成 YOLO 数据集。"""
    import cv2

    from .vision.bubbles import roi_rect
    from .vision.ocr import make_ocr
    from .vision.weaklabel import data_yaml, split_of, weak_labels, yolo_line

    classes = cfg.perception.classes
    index = {c: i for i, c in enumerate(classes)}
    files = _images(args.source)
    out = Path(args.output)
    ocr = make_ocr(cfg.ocr.engine, cfg.env.threads)
    icons = _icon_classifier(cfg)
    names = _friend_names(cfg)()
    counts = {c: 0 for c in classes}
    root = Path(args.source)
    print(f"{len(files)} 张图 → {out}；好友名单：{'、'.join(names) or '（空，只能配 --all-text）'}")
    for n, path in enumerate(files, 1):
        frame = imread(path)
        height, width = frame.shape[:2]
        skip = [roi_rect(cfg.vision.log_roi, width, height)] if _panel_open(cfg, frame) else []
        boxes = weak_labels(
            frame, ocr.recognize(frame), names, icons, cfg.social.icon_offset, skip=skip,
            keep=roi_rect(cfg.env.roi, width, height), all_text=args.all_text, min_score=args.min_score,
        )
        rel = path.relative_to(root) if root.is_dir() else Path(path.name)
        stem = "_".join(rel.with_suffix("").parts)
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
            for c, b in boxes:
                cv2.rectangle(view, (b.x, b.y), (b.x2, b.y2), (0, 200, 255) if c == "name_tag" else (255, 120, 0), 2)
            (out / "_preview").mkdir(parents=True, exist_ok=True)
            imwrite(out / "_preview" / f"{stem}.jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if n % 20 == 0:
            print(f"  {n}/{len(files)}")
    (out / "data.yaml").write_text(data_yaml(out, classes), encoding="utf-8")
    print("自动标出：" + "、".join(f"{c}×{v}" for c, v in counts.items()))
    print(f"数据集配置：{out / 'data.yaml'}")
    print("下一步：用 X-AnyLabeling 打开 images/ 导入 YOLO 标注，给**每一张**补上 player（其他玩家）、player_unlit（没点火的黑影）"
          "和 self（团子自己）框、"
          "修正错框 —— 没补全的图会教模型“这里没有人”，训出来会漏检")


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
    if args.brain:
        cfg.brain.enabled = True
    if cfg.vision.debug_dir:
        log.warning("vision.debug_dir 已废弃，改用 [run] dir；这次先把它当 run.dir 用")
        cfg.run.dir = cfg.vision.debug_dir
    mode = ("dry" if cfg.reply.dry_run else "live") + ("-echo" if cfg.llm.provider == "echo" else "")
    mode += "-brain" if cfg.brain.enabled else ""
    run = RunDir.create(cfg, mode)
    run.attach_log()
    log.info("本次运行的日志和截图: %s", run.path.resolve())
    try:
        if cfg.brain.enabled:
            _run_brain(cfg, run, args.no_emotes, args.duration)
        else:
            _run_agent(cfg, run, args.no_emotes, args.duration)
    finally:
        run.close()


def _build_emotes(cfg: Config, dev, reader, no_emotes: bool):
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
    if cfg.vision.mode == "log":
        panel_visible, panel_key = (lambda: reader.panel_visible(dev.screenshot())), cfg.vision.log_open_key
    else:
        panel_visible, panel_key = (lambda: False), 0
    player = EmotePlayer(dev, Wheel(dev, cfg.wheel, library), cfg.emotes, panel_visible, panel_key)
    try:
        player.start()
    except Exception as exc:
        log.warning("读轮盘失败，这次不做动作: %s", exc)
        return None
    log.info("能做的动作: 轮盘上 %s；可以换上去的 %s", "、".join(player.on_wheel()) or "（无）", "、".join(player.extra) or "（无）")
    return player


def _run_agent(cfg: Config, run: RunDir, no_emotes: bool = False, duration: float = 0.0) -> None:
    from .agent import Agent
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
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
    env = _scene_watcher(cfg, icons, dev) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(
            dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel_visible=reader.panel_visible
        )

    def env_text() -> str:  # 现在的环境 + 刚接受的互动，每次回复前现取
        now = time.monotonic()
        return "\n".join(p for p in (env.describe(now), social.describe(now) if social else "") if p)

    emotes = _build_emotes(cfg, dev, reader, no_emotes)
    responder = Responder(
        llm, cfg.reply, store=store, notes=notes, env=env_text if env else None,
        available_emotes=emotes.available if emotes else None,
    )
    sender = ChatSender(dev, cfg.sender, _screen_size_fn(dev))
    agent = Agent(
        cfg, dev, reader, responder, sender, self_filter, run=run, env=env, social=social, emotes=emotes, store=command_store
    )
    try:
        agent.run(duration)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if hasattr(env, "stop"):
            env.stop()
        if emotes is not None:
            try:
                emotes.restore()
            except Exception:
                log.exception("恢复轮盘失败，请用 emotes wheel 检查")


def _brain_env(cfg: Config) -> tuple[list[str], dict[str, str]]:
    """大脑和眼睛的 Claude Code：命令 + 隔离的环境（单独配置目录 + claude setup-token 令牌）。"""
    from .brain.claude import claude_env, resolve_claude
    from .chat.llm import read_key

    try:
        token = read_key(cfg.brain.token_env)
    except RuntimeError:
        raise RuntimeError(
            f"没有找到大脑用的 Claude 令牌：先运行 claude setup-token，再 setx {cfg.brain.token_env} \"<令牌>\""
        ) from None
    return resolve_claude(cfg.brain.claude_path), claude_env(token, cfg.brain.config_dir)


def _run_brain(cfg: Config, run: RunDir, no_emotes: bool = False, duration: float = 0.0) -> None:
    """统管大脑：身体在当前线程跑（独占设备）；大脑（常驻 Claude Code）和眼睛各一个后台线程；工具经本机 MCP 服务。"""
    import dataclasses
    import threading

    from .brain.body import Body
    from .brain.camera import Camera
    from .brain.claude import one_shot
    from .brain.events import EventQueue
    from .brain.eyes import Eyes, eyes_command
    from .brain.loop import Brain, log_brain_message
    from .brain.mcp_server import SkyServer
    from .brain.prompt import brain_prompt
    from .brain.session import BrainSession
    from .brain.tools import ToolBox
    from .chat.llm import make_llm
    from .chat.responder import Responder
    from .chat.sender import ChatSender

    base, claude_vars = _brain_env(cfg)  # 先检查令牌和 claude 命令，缺了早点报错
    dev = _device(cfg)
    reader, self_filter = _build_reader(cfg)
    reader.trace_path = run.rows_log
    store = notes = None
    if cfg.reply.memory_dir:  # dry-run 也读人设和记忆（看大脑的表现要用），但不写
        from .chat.memory import MemoryStore, NotesKeeper

        store = MemoryStore(cfg.reply.memory_dir)
        if not cfg.reply.dry_run:
            notes = NotesKeeper(make_llm(cfg.llm), store, cfg.reply.persona, cfg.reply.notes_every)
    live_store = None if cfg.reply.dry_run else store
    icons = _icon_classifier(cfg) if cfg.env.enabled else None
    env = _scene_watcher(cfg, icons, dev) if cfg.env.enabled else None
    social = None
    if env and icons:
        from .game.social import SocialHandler

        social = SocialHandler(
            dev, cfg.social, icons, _friend_names(cfg), dry_run=cfg.reply.dry_run, panel_visible=reader.panel_visible
        )
    emotes = _build_emotes(cfg, dev, reader, no_emotes)
    if cfg.vision.mode == "log":
        panel_visible, panel_key = (lambda: reader.panel_visible(dev.screenshot())), cfg.vision.log_open_key
    else:
        panel_visible, panel_key = (lambda: False), 0
    camera = Camera(dev, cfg.brain.camera_step, panel_visible, panel_key)
    events = EventQueue()
    # 大脑离线时的备用回复：在身体线程里调，给短超时、不重试；不带记忆存储，免得和身体重复记聊天记录
    fallback = Responder(make_llm(dataclasses.replace(cfg.llm, timeout=10.0, max_retries=0)), cfg.reply)
    body = Body(
        cfg, dev, reader, ChatSender(dev, cfg.sender, _screen_size_fn(dev)), self_filter, events,
        env=env, social=social, emotes=emotes, camera=camera, fallback=fallback, store=live_store, notes=notes, run=run,
    )
    work = run.path / "brain"
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(eyes_command(base, cfg.brain), claude_vars, work / "eyes", content, cfg.brain.eyes_timeout),
        frame=lambda: body.last_frame,
        labels=lambda: dict(env.labels) if env else {},
        blackout=lambda: body.blackout,
        label_keep=cfg.env.interval * 2 + 1,
    )
    events.subscribe(eyes.notice)
    toolbox = ToolBox(body, eyes, cfg.brain.max_steps, cfg.brain.max_says)
    server = SkyServer(toolbox)
    server.start()
    session = BrainSession(
        base, claude_vars, work / "session", server.url, brain_prompt(cfg.reply, store),
        cfg.brain.model, cfg.brain.effort, cfg.brain.turn_timeout, on_message=log_brain_message,
    )
    brain = Brain(
        cfg.brain, cfg.chat, session, toolbox, events, nearby=env.nearby if env else (lambda now: []),
        eyes=eyes, run=run, store=live_store,
    )
    stop = threading.Event()
    brain_thread = threading.Thread(target=brain.run, args=(stop,), name="brain", daemon=True)
    eyes_thread = threading.Thread(target=eyes.run, args=(stop,), name="eyes", daemon=True)
    body.brain_offline = lambda now: brain.offline(now) or not brain_thread.is_alive()
    brain_thread.start()
    eyes_thread.start()
    try:
        body.run(duration, stop)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        stop.set()
        if hasattr(env, "stop"):
            env.stop()
        body.shutdown()  # 先复原镜头、恢复轮盘、让排队的命令失败：不等大脑
        brain_thread.join(timeout=5)
        if live_store is not None and not brain_thread.is_alive():
            try:
                brain.farewell()  # 把这次的经过记进 inbox.md
            except Exception:
                log.exception("退出前写经过失败")
        session.close()
        server.stop()


def cmd_look(cfg: Config, args) -> None:
    """截一张图让眼睛（Haiku）描述一遍：在真实画面上调眼睛的提示词，看它认得准不准、会不会编名字。"""
    from .brain.claude import one_shot
    from .brain.eyes import Eyes, eyes_command
    from .vision.bubbles import roi_rect
    from .vision.chatlog import find_input_top

    base, claude_vars = _brain_env(cfg)
    dev = _device(cfg)
    frame = dev.screenshot()
    height, width = frame.shape[:2]
    area = roi_rect(cfg.vision.log_roi, width, height)
    panel = find_input_top(frame[:, area.x : area.x2]) is not None
    env = _env_watcher(cfg, background=False)
    env.observe(frame, 0.0, panel_visible=panel)
    Path("tmp").mkdir(exist_ok=True)
    imwrite("tmp/look.jpg", frame)
    eyes = Eyes(
        cfg.brain,
        describe=lambda content: one_shot(
            eyes_command(base, cfg.brain), claude_vars, Path("tmp/look-claude"), content, cfg.brain.eyes_timeout
        ),
        frame=lambda: frame,
        labels=lambda: dict(env.labels),
        blackout=lambda: False,
        clock=lambda: 0.0,
    )
    if args.prompt:
        eyes.look_request = Path(args.prompt).read_text(encoding="utf-8")
    started = time.perf_counter()
    text = eyes.describe_frame(frame, 0.0)
    print(text)
    print(f"\n{cfg.brain.eyes_model}：{time.perf_counter() - started:.1f} 秒；截图存在 tmp/look.jpg")


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
    p.add_argument("--prompt", help="换一个问题：文本文件路径")
    p.set_defaults(func=lambda cfg, args: cmd_look(cfg, args))

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
            q.add_argument("--images", help="用这个目录 / 这张图测（默认实时截图）")
            q.add_argument("-n", type=int, default=200, help="测多少帧")
        else:
            q.add_argument("image", nargs="?", help="图片路径；不填则实时截屏")
            q.add_argument("-o", "--output", default="tmp/perception.png")
    q = psub.add_parser("label", help="用现有识别器给录下来的画面自动标名字标签和圆圈，写成 YOLO 数据集")
    q.add_argument("source", help="图片目录（比如 record 录的 tmp/record/<时间>）")
    q.add_argument("-o", "--output", default="datasets/sky")
    q.add_argument("--val", type=float, default=0.15, help="验证集比例")
    q.add_argument("--all-text", action="store_true", help="画面里读到的字都当名字标签候选（不只好友名单里的），要人工删错的")
    q.add_argument("--min-score", type=float, default=0.9)
    q.add_argument("--preview", action="store_true", help="另存一份画了框的图到 <output>/_preview/，快速检查")
    p.set_defaults(func=cmd_perception)

    p = sub.add_parser("run", help="启动 Agent（默认 dry-run）")
    p.add_argument("--live", action="store_true", help="真的发送消息")
    p.add_argument("--echo", action="store_true", help="不调模型，原样回显（联调用）")
    p.add_argument("--duration", type=float, default=0.0, help="跑多少秒后自动结束（默认一直跑）")
    p.add_argument("--no-emotes", action="store_true", help="这次不做动作（牵着手时用：做动作会松开牵手）")
    p.add_argument("--brain", action="store_true", help="用统管大脑（Claude Code）指挥：看画面、决定说什么做什么")
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
