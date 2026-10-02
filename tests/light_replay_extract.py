"""把 10-02 晚点亮陌生人的存图抽成回放观测 → tests/data/light_replay.json（spec 2026-10-03-light-flame-vanish §5）。

要 runs/ 里的原图和 YOLO 模型（都不进 git），平时不跑；改了 find_flames / 人物框的取法再在 GPU 机器上重抽：
    python tests/light_replay_extract.py [--repo 主目录] [--model models/sky-yolo-v7.pt]
存图上画了标注：白框是当时的团子框（直接读回来当 me），找火焰前把白框的像素抹成灰（不然会混进米白剪影）；
青圈、灰框、紫 / 绿框不是米白色，不影响找火焰。不存图片本身（公开仓库，图里有好友昵称和聊天）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from skydango.config import PerceptionConfig, SocialConfig
from skydango.imageio import imread
from skydango.vision.bubbles import Rect
from skydango.vision.candle import black, find_flames, load_flame
from skydango.vision.detect import make_detector
from skydango.vision.lighting import flame_area
from skydango.vision.perception import LIT_LOW, UNLIT, people_boxes

CASES = {  # 存图目录 → 实际结果（看存图判的，见 spec §0）
    "20261002-214059-live-brain/light/214229": "lit",
    "20261002-214059-live-brain/light/214424": "lit",
    "20261002-214059-live-brain/light/214919": "timeout",
    "20261002-215223-live-brain/light/215454": "lit",
    "20261002-215223-live-brain/light/215823": "lit",
    "20261002-220018-live-brain/light/221839": "lit",
}
MIN_SCORE = 0.6  # 比 disk_min_score 低一点，回放时再按配置筛


def white_box(img: np.ndarray) -> tuple[Rect, np.ndarray]:
    b, g, r = (img[:, :, i].astype(int) for i in range(3))
    white = (b > 235) & (g > 235) & (r > 235)
    cols, rows = np.where(white.sum(0) >= 100)[0], np.where(white.sum(1) >= 100)[0]
    return Rect(int(cols.min()), int(rows.min()), int(cols.max() - cols.min()), int(rows.max() - rows.min())), white


def offset(name: str) -> float:
    """request-+00.0 / raised--00.2 / raised-+01.4 → 相对举起的秒数。出请求的时间没记在名字里，放在举起前 0.5 秒；
    "raised--00.0" 是举起前最后一次扫描（-0.0 四舍五入来的），放在 -0.1。"""
    m = re.match(r"(request|raised)-([+-])(\d+\.\d)", name)
    t = float(m.group(3)) * (-1 if m.group(2) == "-" else 1)
    if m.group(1) == "request":
        return min(t, 0.0) - 0.5
    return -0.1 if m.group(2) == "-" and t == 0 else t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="models/sky-yolo-v7.pt")
    args = ap.parse_args()
    repo = Path(args.repo)
    pc, sc = PerceptionConfig(), SocialConfig()
    det = make_detector(repo / args.model, pc.classes, pc.imgsz, LIT_LOW, pc.iou, pc.device)
    flame = load_flame()
    out = {}
    for d, expect in CASES.items():
        frames = []
        for p in sorted((repo / "runs" / d).glob("*.jpg")):
            img = imread(p)
            me, white = white_box(img)
            clean = img.copy()
            clean[white] = (90, 90, 90)
            area = flame_area(me, sc, img.shape[1], img.shape[0])
            people = []
            for x in people_boxes(det.detect(clean)):
                if x.cls == "self":
                    continue
                blk = black(clean, x.box, sc.lit_v, exclude=me)
                people.append([x.box.x, x.box.y, x.box.w, x.box.h, x.cls == UNLIT, round(x.score, 3),
                               None if blk is None else round(blk, 3)])
            frames.append({
                "file": p.name, "t": offset(p.name), "size": [img.shape[1], img.shape[0]], "me": [me.x, me.y, me.w, me.h],
                "flames": [[f.x, f.y, round(f.r, 1), round(f.score, 3)] for f in find_flames(clean, area, flame, MIN_SCORE)],
                "people": people,  # [x, y, w, h, 是不是 player_unlit, 分数, black()]
            })
        frames.sort(key=lambda f: f["t"])
        out[d.split("/")[-1]] = {"expect": expect, "frames": frames}
        print(d, expect, len(frames))
    path = Path(__file__).parent / "data" / "light_replay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", path)


if __name__ == "__main__":
    main()
