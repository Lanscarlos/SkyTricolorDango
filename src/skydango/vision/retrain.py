"""一键重训和对比（`perception retrain`，spec 2026-10-04-hardcase-inbox-design §7）。

挪走标注缓存 → 训 YOLO（`[retrain]`，同 tmp/yolo/train_v10.py）→ 复制成 models/sky-yolo-v<N>.pt → 训外形头
（只用标注页确认过的）→ models/attrs-<日期><字母>.npz → 旧 / 新模型在 datasets/sky 验证集上回放（答案按 `gt_fixes` 修正）
和 ultralytics val 的各类 mAP50，"全部验证帧"和"来自收件箱的验证帧"各一份 → report.md + result.json → `_stats.json` 记训练时间。

任何一步失败抛 `RetrainFailed`（带是哪一步），不写 result.json、不改配置；已经写出的模型文件留着。
换不换新模型由管理面板决定（spec §7.3），这里不碰配置。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from ..config import Config, RetrainConfig
from . import attrs, attrs_train as at
from . import inbox as ib

DATASET = Path("datasets/sky")
MODELS = Path("models")
TMP_YOLO = Path("tmp/yolo")
ATTRS_ROOT = Path("datasets/attrs")
WOBBLE = 0.03  # 新旧差这么多以内：可能只是单次训练的波动（v11 那次）
ROWS = (("old", "旧 YOLO"), ("old_head", "旧 YOLO + 旧外形头"), ("new", "新 YOLO"), ("new_head", "新 YOLO + 新外形头"))
_YOLO_FILE = re.compile(r"^sky-yolo-v(\d+)\.")
_YOLO_DIR = re.compile(r"^sky-v(\d+)$")


class RetrainFailed(RuntimeError):
    """重训某一步失败：step = 中文的步骤名，cause = 原来的异常。"""

    def __init__(self, step: str, cause: BaseException) -> None:
        super().__init__(f"{step}：{type(cause).__name__}: {cause}")
        self.step = step
        self.cause = cause


# ---------- 小工具 ----------

def next_version(models: Path, tmp_yolo: Path) -> int:
    """models/sky-yolo-v<N>.* 和 tmp_yolo/sky-v<N>/ 里最大的 N + 1（没训上线的 v11 也占号，免得混）。"""
    nums = [0]
    if Path(models).is_dir():
        nums += [int(m.group(1)) for p in Path(models).iterdir() if (m := _YOLO_FILE.match(p.name))]
    if Path(tmp_yolo).is_dir():
        nums += [int(m.group(1)) for p in Path(tmp_yolo).iterdir() if p.is_dir() and (m := _YOLO_DIR.match(p.name))]
    return max(nums) + 1


def next_attrs(models: Path, now: dt.datetime) -> Path:
    """models/attrs-<日期><字母>.npz，字母从 a 起取第一个没用过的（attrs-20261004d-mask.npz 这种带后缀的也算用过 d）。"""
    date = f"{now:%Y%m%d}"
    pat = re.compile(rf"^attrs-{date}([a-z])(?![a-z])")
    used = {m.group(1) for p in Path(models).iterdir() if (m := pat.match(p.name))} if Path(models).is_dir() else set()
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c not in used:
            return Path(models) / f"attrs-{date}{c}.npz"
    raise FileExistsError(f"{models} 里 attrs-{date}a ~ z 都用过了")


def stash_caches(dataset: Path, to: Path) -> list[Path]:
    """dataset/labels/ 下所有 .cache（含子目录）挪到 to/（保留相对路径）：ultralytics 只按文件大小判缓存过期，
    标注改了大小没变就读旧缓存（v10 踩过）。返回挪过去的路径。"""
    labels = Path(dataset) / "labels"
    moved: list[Path] = []
    for p in sorted(labels.rglob("*.cache")) if labels.is_dir() else []:
        dst = Path(to) / p.relative_to(labels)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dst)
        moved.append(dst)
    return moved


def _ultra_train(rc: RetrainConfig, data: Path, out: Path, progress: Callable[[str], None]) -> None:
    from ultralytics import YOLO

    model = YOLO(rc.base)
    model.add_callback("on_train_epoch_end", lambda tr: progress(f"PROGRESS 训练 YOLO {tr.epoch + 1}/{rc.epochs} 轮"))
    model.train(data=str(Path(data).resolve()), imgsz=rc.imgsz, epochs=rc.epochs, batch=rc.batch, workers=rc.workers,
                project=str(Path(out).parent.resolve()), name=Path(out).name, exist_ok=True, device=0, plots=True)


def train_yolo(cfg: RetrainConfig, dataset: Path, out: Path, train=None,
               progress: Callable[[str], None] = lambda s: None) -> Path:
    """训 YOLO（train 默认调 ultralytics：YOLO(base).train(data=dataset/data.yaml, …, project=out.parent, name=out.name)），
    返回 out/weights/best.pt。train(cfg, data_yaml, out, progress) 可以换成假的（测试）。"""
    (train or _ultra_train)(cfg, Path(dataset) / "data.yaml", Path(out), progress)
    best = Path(out) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"训练完没找到 {best}")
    return best


def ultra_val(rc: RetrainConfig) -> Callable[[Path, Path, Path], dict[str, float]]:
    """ultralytics val：val(权重, data.yaml, 输出目录) → {类别: mAP50, "all": 平均}（验证集里没有的类别不列）。"""
    def val(weights: Path, data: Path, out: Path) -> dict[str, float]:
        from ultralytics import YOLO

        r = YOLO(str(weights)).val(data=str(Path(data).resolve()), imgsz=rc.imgsz, batch=rc.batch, workers=rc.workers,
                                   split="val", plots=False, verbose=False, project=str(Path(out).parent.resolve()),
                                   name=Path(out).name, exist_ok=True)
        res = {str(r.names[c]): float(r.box.all_ap[i, 0]) for i, c in enumerate(r.box.ap_class_index)}
        res["all"] = float(r.box.map50)
        return res

    return val


def train_attrs(cfg: Config, embedder, attrs_root: Path, out: Path, now: dt.datetime) -> dict:
    """训外形头（同 `perception attrs-train` 的默认：只用标注页确认过的、框外填灰）→ 存 out。返回 `run_training` 的结果。"""
    res = at.run_training(attrs_root, embedder, Path(attrs_root) / "_features", confirmed_only=True, keep=attrs.CROP_KEEP)
    attrs.save_model(out, {"form": res["head"]}, Path(cfg.attrs.backbone).name, f"{embedder.size}:imagenet", f"{now:%Y%m%d}")
    return res


def _inbox_yaml(dataset: Path, images: list[Path], work: Path) -> Path:
    """只含收件箱验证帧的 data.yaml：图片清单写进 txt（train / val 都指它），别的行照抄数据集的 data.yaml。"""
    work.mkdir(parents=True, exist_ok=True)
    txt = work / "inbox_val.txt"
    txt.write_text("".join(f"{Path(p).resolve()}\n" for p in images), encoding="utf-8")
    lines = [ln for ln in (Path(dataset) / "data.yaml").read_text(encoding="utf-8").splitlines()
             if not re.match(r"^(train|val|test|path)\s*:", ln)]
    head = [f"path: {Path(dataset).resolve().as_posix()}", f"train: {txt.resolve().as_posix()}", f"val: {txt.resolve().as_posix()}"]
    out = work / "inbox_val.yaml"
    out.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    return out


# ---------- 对比 ----------

def compare(old_yolo: str | Path | None, new_yolo: str | Path, old_attrs, new_attrs, dataset: Path, cfg: Config,
            inbox_frames: set[str], *, attrs_root: Path | None = ATTRS_ROOT, val=None, work: Path | None = None,
            step: Callable[[str], None] = lambda s: None) -> dict:
    """旧 / 新模型在 dataset 验证集上比：整帧回放四行（旧 YOLO、旧 + 旧外形头、新 YOLO、新 + 新外形头）
    × "全部验证帧"和"来自收件箱的验证帧"（inbox_frames = 帧名，不带扩展名）两份；val 不是 None 时再加各类 mAP50。
    old_attrs / new_attrs 是加载好的外形头（None = 没有，那一行空着）；old_yolo 文件不存在时旧的两行都空着。
    返回 {all: 一份, inbox: 一份或 None（没有收件箱验证帧）, params, notes}；一份 = {frames, gt, rows: {old, old_head, new, new_head}, map50}。"""
    p, a = cfg.perception, cfg.attrs
    notes: list[str] = []
    frames = at.frames_in(dataset, "val")
    mine = [f for f in frames if f.stem in inbox_frames]
    other = [f for f in frames if f.stem not in inbox_frames]
    fixes = at.load_fixes(attrs_root, notes)
    det_conf = min(0.2, p.low_conf)
    parts: dict[str, dict | None] = {"all": {"rows": {}}, "inbox": {"rows": {}} if mine else None}
    if not mine:
        notes.append("验证集里没有来自收件箱的帧（收件箱通过的帧都分到了训练集），没单独算")
    have_old = old_yolo is not None and Path(old_yolo).is_file()
    if not have_old:
        notes.append(f"没有旧 YOLO（{old_yolo} 不存在），旧的两行空着")
    for side, yolo, head in (("old", old_yolo, old_attrs), ("new", new_yolo, new_attrs)):
        if side == "old" and not have_old:
            for part in parts.values():
                if part is not None:
                    part["rows"].update(old=None, old_head=None)
            continue
        step(f"回放对比（{'旧' if side == 'old' else '新'}模型）")
        detector = at.replay_detector(p, yolo, det_conf)
        rec_mine = at.collect(mine, detector, head, p.low_conf, fixes)
        rec_other = at.collect(other, detector, head, p.low_conf, fixes)
        for key, recs in (("all", rec_mine + rec_other), ("inbox", rec_mine)):
            part = parts[key]
            if part is None:
                continue
            r = at.replay_row(recs, p.low_conf, p.conf, a)
            part["rows"][side] = r["baseline"]
            part["rows"][f"{side}_head"] = r["second"] if head is not None else None
            part["frames"], part["gt"] = r["frames"], r["gt"]
    for key, part in parts.items():
        if part is None:
            continue
        part["map50"] = None
        if val is None:
            continue
        work = Path(work) if work is not None else Path(".")
        data = Path(dataset) / "data.yaml" if key == "all" else _inbox_yaml(dataset, mine, work)
        part["map50"] = {}
        for side, yolo in (("old", old_yolo), ("new", new_yolo)):
            if side == "old" and not have_old:
                part["map50"][side] = None
                continue
            step(f"mAP50（{'旧' if side == 'old' else '新'} YOLO，{'全部验证帧' if key == 'all' else '收件箱验证帧'}）")
            part["map50"][side] = val(Path(yolo), data, work / f"val-{side}-{key}")
    if val is None:
        notes.append("没算 mAP50")
    params = {"conf_low": p.low_conf, "conf": p.conf, "accept": a.accept, "reject": a.reject, "yolo_w": a.yolo_w}
    return {**parts, "params": params, "notes": notes}


# ---------- 报告 ----------

def _cell(v: float | None, old: float | None) -> str:
    """百分数；有旧值时带上差几个点，差 ≤ WOBBLE 标「（可能是波动）」（一模一样的只写 ±0）。"""
    if v is None:
        return "—"
    s = f"{v:.0%}"
    if old is None:
        return s
    if v == old:
        return s + "（±0）"
    d = round((v - old) * 100)
    return s + f"（{d:+d}）" + ("（可能是波动）" if abs(v - old) <= WOBBLE + 1e-9 else "")


def _part_md(title: str, part: dict | None, params: dict) -> list[str]:
    if part is None:
        return [f"## {title}", "", "没有来自收件箱的验证帧，这一节跳过。", ""]
    L = [f"## {title}（{part.get('frames', 0)} 帧、{part.get('gt', 0)} 个人物标注）", ""]
    if params:
        L += [f"整帧回放：conf_low = {params['conf_low']:g}、conf = {params['conf']:g}、accept = {params['accept']:g}、"
              f"reject = {params['reject']:g}、yolo_w = {params['yolo_w']:g}", ""]
    L += ["| | 对 | 错报 | 漏 | 精确率 | 召回率 | 点没点火认反 |", "|---|---|---|---|---|---|---|"]
    rows = part["rows"]
    for key, name in ROWS:
        s = rows.get(key)
        if s is None:
            why = "没有旧 YOLO" if key.startswith("old") and rows.get("old") is None else "没有旧外形头" if key == "old_head" else "—"
            L.append(f"| {name} | {why} | | | | | |")
            continue
        base = rows.get(key.replace("new", "old")) if key.startswith("new") else None
        L.append(f"| {name} | {s['tp']} | {s['fp']} | {s['fn']} | "
                 f"{_cell(s['precision'], base and base['precision'])} | {_cell(s['recall'], base and base['recall'])} | "
                 f"{s['mismatch']} / {s['pairs']} |")
    L.append("")
    m = part.get("map50")
    if m:
        old, new = m.get("old") or {}, m.get("new") or {}
        names = [n for n in dict.fromkeys([*old, *new]) if n != "all"] + ["all"]
        L += ["ultralytics val 各类 mAP50：", "", "| 类别 | 旧 | 新 |", "|---|---|---|"]
        for n in names:
            o, v = old.get(n), new.get(n)
            if o is None and v is None:
                continue
            L.append(f"| {'全部（平均）' if n == 'all' else n} | {_cell(o, None)} | {_cell(v, o)} |")
        L.append("")
    return L


def report_md(result: dict) -> str:
    """重训报告：新旧模型、整帧回放四行和 mAP50（全部验证帧 / 来自收件箱的验证帧），加两句固定提醒。"""
    n = result["numbers"]
    params = n.get("params") or {}
    when = result.get("finished")
    L = [f"# 重训报告{' ' + when if when else ''}", "",
         f"- 新 YOLO：{result['yolo']}（旧：{result.get('old_yolo')}）",
         f"- 新外形头：{result['attrs']}（旧：{result.get('old_attrs')}）"]
    if result.get("train"):
        t = result["train"]
        L.append(f"- YOLO 训练：起点 {t['base']}、imgsz {t['imgsz']}、{t['epochs']} 轮、batch {t['batch']}；"
                 f"收件箱通过的帧用到 {result.get('trained_frames', 0)} 帧")
    head = result.get("head") or {}
    if head.get("macro_f1") is not None:
        L.append(f"- 外形头：验证集宏平均 F1 {head['macro_f1']:.3f}（训练 {head.get('train_n')} 张、验证 {head.get('val_n')} 张）")
    L += ["",
          "> 提醒：新旧用的是同一份答案（datasets/sky 验证集的标注，按标注页确认过的外形修正）——答案本身错的地方新旧一起错，只能看相对高低。",
          f"> 提醒：差 ≤ {WOBBLE * 100:.0f} 个点的格子后面标了「（可能是波动）」：可能只是单次训练的波动（v11 那次就是），别只凭它决定换不换。",
          ""]
    L += _part_md("全部验证帧", n["all"], params)
    L += _part_md("来自收件箱的验证帧", n.get("inbox"), params)
    notes = list(n.get("notes") or []) + list(result.get("notes") or [])
    if notes:
        L += ["## 注意", ""] + [f"- {x}" for x in notes] + [""]
    return "\n".join(L)


# ---------- 一键重训 ----------

def _device(cfg: Config) -> str:
    return cfg.attrs.device or cfg.perception.device or "cpu"


def run_retrain(cfg: Config, dataset: Path, inbox: Path, out: Path, progress: Callable[[str], None], train=None, val=None,
                attrs_train_fn=None, *, embedder=None, models: Path = MODELS, tmp_yolo: Path = TMP_YOLO,
                attrs_root: Path = ATTRS_ROOT, now: dt.datetime | None = None) -> dict:
    """见模块说明。progress 收 "PROGRESS <步骤>" 这样的整行；val 默认 ultralytics val；attrs_train_fn(cfg, embedder, attrs_root, 输出, now)
    默认 `train_attrs`；embedder 默认 [attrs] backbone 的 OnnxEmbedder。失败抛 `RetrainFailed`。"""
    started = time.time()
    now = now or dt.datetime.now()
    current = ["检查"]

    def step(name: str) -> None:
        current[0] = name
        progress(f"PROGRESS {name}")

    try:
        return _run(cfg, Path(dataset), Path(inbox), Path(out), progress, step, train, val, attrs_train_fn, embedder,
                    Path(models), Path(tmp_yolo), Path(attrs_root), now, started)
    except Exception as exc:
        raise RetrainFailed(current[0], exc) from exc


def _run(cfg, dataset, inbox, out, progress, step, train, val, attrs_train_fn, embedder, models, tmp_yolo, attrs_root,
         now, started) -> dict:
    step("检查")
    if not (dataset / "data.yaml").is_file():
        raise FileNotFoundError(f"{dataset / 'data.yaml'} 不存在")
    if not at.frames_in(dataset, "val"):
        raise FileNotFoundError(f"{dataset / 'images' / 'val'} 里没有图，没法对比")
    if embedder is None and not Path(cfg.attrs.backbone).is_file():
        raise FileNotFoundError(f"外形头的 DINOv2 主干 {cfg.attrs.backbone} 不存在")
    if attrs_train_fn is None:  # 外形头的数据先查一遍，别等 YOLO 训了几个小时才发现不够
        samples = at.list_samples(attrs_root, confirmed_only=True)
        at.merge_labels({f: sum(1 for _, ff in samples if ff == f) for f in attrs.FORMS}, at.MIN_PER_CLASS)
    passed = ib.passed_frames(inbox, before=started)
    yolo_out = models / f"sky-yolo-v{next_version(models, tmp_yolo)}.pt"
    attrs_out = next_attrs(models, now)
    out.mkdir(parents=True, exist_ok=True)

    step("挪走标注缓存")
    stash_caches(dataset, out / "caches")

    step("训练 YOLO")
    best = train_yolo(cfg.retrain, dataset, out / "yolo", train, progress)
    if yolo_out.exists():
        raise FileExistsError(f"{yolo_out} 已经有了")
    yolo_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, yolo_out)

    step("训练外形头")
    if embedder is None:
        from .embed import OnnxEmbedder

        embedder = OnnxEmbedder(cfg.attrs.backbone, norm="imagenet", device=_device(cfg), what="attrs.backbone")
    head_res = (attrs_train_fn or train_attrs)(cfg, embedder, attrs_root, attrs_out, now) or {}
    if not attrs_out.is_file():
        raise FileNotFoundError(f"外形头训练完没有 {attrs_out}")

    step("加载外形头")
    new_head = attrs.load_model(dataclasses.replace(cfg.attrs, model=str(attrs_out)), _device(cfg), embedder)
    if new_head is None:
        raise RuntimeError(f"新外形头 {attrs_out} 加载不了（原因见上面的警告）")
    notes: list[str] = []
    old_head = None
    if Path(cfg.attrs.model).is_file():
        old_head = attrs.load_model(cfg.attrs, _device(cfg), embedder)
    if old_head is None:
        notes.append(f"没有旧外形头（{cfg.attrs.model} 不存在或加载不了），「旧 YOLO + 旧外形头」那行空着")

    inbox_val = {d["dataset"].split("/", 1)[1] for d in ib.passed_frames(inbox) if d["dataset"].startswith("val/")}
    numbers = compare(cfg.perception.model, yolo_out, old_head, new_head, dataset, cfg, inbox_val, attrs_root=attrs_root,
                      val=val if val is not None else ultra_val(cfg.retrain), work=out, step=step)

    step("写报告")
    ev = head_res.get("eval") or {}
    result = {
        "ok": True, "yolo": str(yolo_out), "attrs": str(attrs_out),
        "old_yolo": cfg.perception.model, "old_attrs": cfg.attrs.model,
        "started": dt.datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
        "finished": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_frames": len(passed), "train": dataclasses.asdict(cfg.retrain),
        "head": {"macro_f1": ev.get("macro_f1"), "train_n": head_res.get("train_n"), "val_n": head_res.get("val_n")},
        "notes": notes, "numbers": numbers, "report": str(out / "report.md"),
    }
    (out / "report.md").write_text(report_md(result), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    step("记下训练时间")
    ib.mark_trained(inbox, len(passed), at=started)
    return result
