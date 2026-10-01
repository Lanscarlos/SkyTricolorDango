"""认装扮的离线标定（spec 2026-10-01-appearance §10）：录像上统计同一个人 / 不同人的外观相似度，给出建议的门槛，
再假装看不到名字标签、按 §4 重放，数接回对了 / 接错 / 漏接。纯函数（不碰 YOLO）；跑录像的部分在 cli 的 `perception appearance-eval`。"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from ..config import AppearanceConfig
from .appearance import AppearanceBook
from .embed import cosine

SAME_SAMPLES = 50  # 同一身份最多取这么多个样本两两比
DIFF_PAIRS = 200  # 每对不同身份最多比这么多对
MARGIN = 0.05
QUANTILES = (1, 5, 10, 50, 90, 95, 98, 99)


class Harvester:
    """跟着感知层收集每条轨迹的外观特征（`data["feat"]`，好样本数涨了才收一次）。
    身份最后再定：轨迹上出现过的名字里最多的那个；一直没有名字的是 t<轨迹 id>（只在这段录像里代表同一个人）。"""

    def __init__(self) -> None:
        self._samples: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
        self._names: dict[int, Counter] = defaultdict(Counter)
        self._counts: dict[int, int] = {}

    def add(self, tracks: Iterable, now: float) -> None:
        for t in tracks:
            if t.cls != "player" or t.last != now:
                continue
            d = t.data
            if d.get("name"):
                self._names[t.id][d["name"]] += 1
            n = d.get("samples", 0)
            if d.get("feat") is None or n <= self._counts.get(t.id, 0):
                continue
            self._counts[t.id] = n
            self._samples[t.id].append((now, np.asarray(d["feat"], np.float32).copy()))

    def _identity(self, tid: int) -> str:
        names = self._names.get(tid)
        return names.most_common(1)[0][0] if names else f"t{tid}"

    def sequence(self) -> list[tuple[float, str, np.ndarray]]:
        out = [(t, self._identity(tid), f) for tid, items in self._samples.items() for t, f in items]
        return sorted(out, key=lambda x: x[0])

    def by_identity(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = defaultdict(list)
        for _, who, f in self.sequence():
            out[who].append(f)
        return dict(out)


def pair_scores(tracks: dict[str, list[np.ndarray]]) -> tuple[list[float], list[float]]:
    """（同一身份两两相似度，不同身份两两相似度）。同一身份最多取 50 个样本，不同身份每对最多 200 对；固定随机种子。"""
    rng = random.Random(0)
    names = sorted(tracks)
    pools = {}
    for n in names:
        items = list(tracks[n])
        pools[n] = rng.sample(items, SAME_SAMPLES) if len(items) > SAME_SAMPLES else items
    same = [cosine(a[i], a[j]) for a in pools.values() for i in range(len(a)) for j in range(i + 1, len(a))]
    diff = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            nx, ny = len(tracks[x]), len(tracks[y])
            picks = range(nx * ny)
            if nx * ny > DIFF_PAIRS:
                picks = rng.sample(picks, DIFF_PAIRS)  # range 上抽样，不用先列出全部配对
            diff += [cosine(tracks[x][k // ny], tracks[y][k % ny]) for k in picks]
    return same, diff


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {f"p{q}": round(float(np.percentile(values, q)), 4) for q in QUANTILES}


def suggest(same: list[float], diff: list[float], max_wrong: float = 0.02) -> dict:
    """建议值：match = 让不同人相似度 ≥ 它的比例严格小于 max_wrong 的最小 0.01 刻度；changed = 同一身份相似度的 5% 分位数；margin 固定 0.05。"""
    match = None
    if diff:
        arr = np.asarray(diff, np.float64)
        for k in range(0, 102):  # 到 1.01：不同人全是 1.0 时也有个答案（等于"没法分"）
            t = k / 100
            if float(np.mean(arr >= t - 1e-9)) < max_wrong:
                match = t
                break
    changed = round(float(np.percentile(same, 5)), 2) if same else None
    return {
        "match": match, "changed": changed, "margin": MARGIN, "max_wrong": max_wrong,
        "same_n": len(same), "diff_n": len(diff), "same_q": _quantiles(same), "diff_q": _quantiles(diff),
    }


def replay(sequence: list[tuple[float, str, np.ndarray]], cfg: AppearanceConfig, hide_every: float = 5.0) -> dict:
    """按时间重放（身份已知）：从第一个样本起每 hide_every 秒一段，露着名字标签（好友的样本学进记忆簿）和
    全藏掉交替、第一段露着；藏掉的段里同一时刻的样本一起交给 assign_friends 判（不看轨迹攒了几个好样本）：
    好友认对 = right，认成别人（含陌生人 t<id> 被认成好友）= wrong，好友没认出 = missed，陌生人没认成好友 = stranger_ok。"""
    book = AppearanceBook(cfg, "eval")
    right = wrong = missed = stranger_ok = 0
    if not sequence:
        return {"right": 0, "wrong": 0, "missed": 0, "stranger_ok": 0, "wrong_rate": None, "hide_every": hide_every}
    t0 = sequence[0][0]
    by_time: dict[float, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for t, who, f in sequence:
        by_time[t].append((who, f))
    for t in sorted(by_time):
        items = by_time[t]
        hidden = math.floor((t - t0) / hide_every) % 2 == 1
        if not hidden:
            for who, f in items:
                if not untagged(who):
                    book.learn("friend", who, f, t)
            continue
        got = book.assign_friends({i: f for i, (_, f) in enumerate(items)}, set())
        for i, (who, _) in enumerate(items):
            name = got.get(i)
            if untagged(who):
                if name is None:
                    stranger_ok += 1
                else:
                    wrong += 1
            elif name is None:
                missed += 1
            elif name == who:
                right += 1
            else:
                wrong += 1
    decided = right + wrong
    return {"right": right, "wrong": wrong, "missed": missed, "stranger_ok": stranger_ok,
            "wrong_rate": wrong / decided if decided else None, "hide_every": hide_every}


def untagged(who: str) -> bool:
    """t<轨迹 id> = 一直没挂上名字的轨迹。"""
    return who.startswith("t") and who[1:].isdigit()


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.2f}"


def _replay_lines(r: dict, max_wrong: float) -> list[str]:
    rate = r.get("wrong_rate")
    lines = [f"- 接回对了 {r['right']}、接错 {r['wrong']}、漏接 {r['missed']}；陌生人没认成好友 {r.get('stranger_ok', 0)}",
             f"- 接错率（接错 / (对 + 错)）：{_fmt(rate, pct=True)}"]
    if rate is not None and rate > max_wrong:
        lines.append(f"- **接错率超过 {max_wrong:.0%}，建议往严了调**（`match` / `margin` 调高）：接错比漏接严重")
    return lines


def report_md(summary: dict) -> str:
    s = summary.get("suggest", {})
    max_wrong = s.get("max_wrong", 0.02)
    cur = summary.get("current", {})
    lines = ["# 认装扮离线标定", ""]
    head = f"共 {summary.get('frames', 0)} 帧"
    if summary.get("source"):
        head += f"（{summary['source']}）"
    if summary.get("embed"):
        head += f"，特征 `{summary['embed']}`"
    lines += [head + "。", "",
              "身份：名字标签认出的好友按名字，其余 `t<轨迹>`（一直没挂上名字的轨迹，可能是陌生人、也可能是名字被挡住的好友）。", "",
              "| 身份 | 样本数 |", "|---|---|"]
    ids = summary.get("identities", {})
    lines += [f"| {n} | {c} |" for n, c in sorted(ids.items(), key=lambda x: -x[1])]
    if not ids:
        lines.append("| （没收到样本） | |")
    lines += ["", "## 相似度分布", "",
              f"同一身份 {s.get('same_n', 0)} 对，不同身份 {s.get('diff_n', 0)} 对。", "",
              "| 分位数 | 同一身份 | 不同身份 |", "|---|---|---|"]
    sq, dq = s.get("same_q", {}), s.get("diff_q", {})
    lines += [f"| {k} | {_fmt(sq.get(k))} | {_fmt(dq.get(k))} |" for k in (f"p{q}" for q in QUANTILES)]
    lines += ["", f"## 建议值（接错率 < {max_wrong:.0%}）", "",
              "| 键 | 现在 | 建议 | 怎么算的 |", "|---|---|---|---|",
              f"| `match` | {_fmt(cur.get('match'))} | {_fmt(s.get('match'))} | 不同人相似度 ≥ 它的比例 < {max_wrong:.0%} 的最小 0.01 刻度 |",
              f"| `changed` | {_fmt(cur.get('changed'))} | {_fmt(s.get('changed'))} | 同一身份相似度的 5% 分位数 |",
              f"| `margin` | {_fmt(cur.get('margin'))} | {_fmt(s.get('margin'))} | 固定 |", ""]
    if s.get("match") is not None and s["match"] > 1.0:
        lines += ["不同人的相似度高到分不开：这个特征模型在这段录像上认不了人。", ""]
    for key, title in (("replay", "现在的配置"), ("replay_suggested", "建议的配置")):
        r = summary.get(key)
        if not r:
            continue
        lines += [f"## 藏标签重放（{title}）", "",
                  f"每 {r.get('hide_every', 5.0):g} 秒交替：露着名字标签时学外观，藏掉时按外观认（同 §4 的 `assign_friends`）。", ""]
        lines += _replay_lines(r, max_wrong) + [""]
    lines += ["说明：特征是轨迹上平滑过的平均特征（同运行时），同一条轨迹前后样本很像，同一身份的分布会偏高；",
              "`t<轨迹>` 里混着名字被挡住的好友时，不同身份的分布会偏高（建议值偏严）。改默认值前对照录像看一眼。"]
    return "\n".join(lines) + "\n"
