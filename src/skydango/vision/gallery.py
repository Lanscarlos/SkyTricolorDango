"""一个身份的多张样本底库（认装扮：每人留一组外观样本，而不是一个平均特征）。

样本的特征都是单位向量，余弦 = 点积。去重：颜色余弦 ≥ DUP，并且（任一边没有 DINOv2，或 DINOv2 余弦也 ≥ DUP）才算重复——
同一身衣服不同角度颜色直方图几乎一样，团子靠 DINOv2 认，只看颜色会把转圈登记压成一张；新来的钉住样本（转圈登记）从不去重，直接加。
满了挤掉最冗余的（和别的样本颜色最像的）一张，钉住的样本（名字标签证实过的）永远不挤。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DUP = 0.97  # 颜色（有 DINOv2 时还要 DINOv2）余弦 ≥ 这个算重复：只刷新那张的时间


@dataclass
class Sample:
    color: np.ndarray  # 颜色特征（单位向量）
    dino: np.ndarray | None  # DINOv2 特征（单位向量），没开 / 算不出就是 None
    t: float  # 采集时间
    h: float  # 采样时的框高 / 画面高
    pinned: bool = False  # 钉住的不会被挤掉


class Gallery:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._samples: list[Sample] = []

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def pinned_count(self) -> int:
        return sum(1 for s in self._samples if s.pinned)

    @property
    def samples(self) -> list[Sample]:
        """只读副本（改列表不影响底库）。"""
        return list(self._samples)

    def add(self, s: Sample) -> None:
        """加一张样本：和已有的某张重复（颜色余弦 ≥ DUP，且任一边没 DINOv2 或 DINOv2 余弦也 ≥ DUP）就只刷新那张；
        钉住的新样本从不算重复；超过上限挤掉最冗余的未钉住样本。"""
        best, best_cos = None, -1.0
        if not s.pinned:
            for old in self._samples:
                c = float(np.dot(old.color, s.color)) if old.color.shape == s.color.shape else -1.0
                if c < DUP or c <= best_cos:
                    continue
                if old.dino is not None and s.dino is not None and (
                    old.dino.shape != s.dino.shape or float(np.dot(old.dino, s.dino)) < DUP
                ):
                    continue
                best, best_cos = old, c
        if best is not None:
            best.t = max(best.t, s.t)
            best.h = s.h
            if best.dino is None and s.dino is not None:
                best.dino = s.dino
            return
        self._samples.append(s)
        while len(self._samples) > self.max_size:
            if not self._evict():
                break  # 全都钉住：允许超出

    def best(self, feat: np.ndarray | None, which: str) -> float | None:
        """feat 和底库里每张样本（which = "color" / "dino"）余弦的最大值；没有可比的返回 None。"""
        if feat is None:
            return None
        top = None
        for s in self._samples:
            ref = s.color if which == "color" else s.dino
            if ref is None or ref.shape != feat.shape:
                continue
            c = float(np.dot(ref, feat))
            if top is None or c > top:
                top = c
        return top

    def _evict(self) -> bool:
        """挤掉未钉住样本里和其余样本最大余弦最大的一张（相同时挤 t 小的）。没有可挤的返回 False。"""
        cand = [(i, s) for i, s in enumerate(self._samples) if not s.pinned]
        if not cand:
            return False
        worst, worst_key = None, None
        for i, s in cand:
            m = max(
                (float(np.dot(s.color, o.color)) for j, o in enumerate(self._samples) if j != i and o.color.shape == s.color.shape),
                default=-1.0,
            )
            key = (-round(m, 6), s.t)  # 最冗余的排最前，相同时 t 小的在前
            if worst_key is None or key < worst_key:
                worst, worst_key = i, key
        del self._samples[worst]
        return True
