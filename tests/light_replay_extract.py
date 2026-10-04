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
from skydango.vision.lighting import flame_area, under_tag
from skydango.vision.perception import LIT_LOW, UNLIT, merge_people, people_boxes

CASES = {  # 存图目录 → 实际结果（看存图判的，见 spec §0；10-03 的见 docs/progress/2026-10-03-plan.md「10-03 晚真机复盘」）
    "20261002-214059-live-brain/light/214229": "lit",
    "20261002-214059-live-brain/light/214424": "lit",
    "20261002-214059-live-brain/light/214919": "timeout",
    "20261002-215223-live-brain/light/215454": "lit",
    "20261002-215223-live-brain/light/215823": "lit",
    "20261002-220018-live-brain/light/221839": "lit",
    # 10-03 晚：举起后只存了一两张图、看不出亮没亮的记 timeout（存图到头还没判 lit 就对：没依据别判点亮）
    "20261003-204518-live-brain/light/204557": "lit",  # +2.0 亮了；人挤，旧判定配对跳到别人的火焰、判成走开
    "20261003-204518-live-brain/light/204706": "timeout",  # 1.7 秒判 lit 是对的，但存图只到 +0.8：回放里只要求别判错
    "20261003-210211-live-brain/light/210350": "timeout",  # 举起前约 1 秒火焰就没了，之后没存到他
    "20261003-210211-live-brain/light/212739": "lit",  # 约 +4 秒亮了；深色衣服 black 0.58，旧判定当成还黑着多等
    "20261003-210211-live-brain/light/212754": "lit",
    "20261003-232541-live-brain/light/232605": "timeout",  # 举起前 0.9 秒就没看到火焰，+0.7 还是黑斗篷
    "20261003-232541-live-brain/light/233045": "timeout",  # 火焰在好友名字标签下面被排除、当成没了；+0.8 还是全黑
    "20261003-232541-live-brain/light/233058": "timeout",  # 举起前 0.9 秒就没看到火焰，+0.1 全黑
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
            dets, flagged = merge_people(det.detect(clean))  # 和感知层一样先合框：被压掉的黑影框也算"认成黑影"
            dup = {id(dets[i]) for i in flagged}
            # 名字标签下面的火焰感知层不交给 FlameWatch（好友的点火圆圈）：这里不跑 OCR，YOLO 的标签都当认出了名字
            tags = [x.box for x in dets if x.cls == "name_tag"]
            found = find_flames(clean, area, flame, MIN_SCORE)
            masked = [f for f in found if any(under_tag((f.x, f.y), t) for t in tags)]
            for x in people_boxes(dets):
                if x.cls == "self":
                    continue
                blk = black(clean, x.box, sc.lit_v, exclude=me)
                people.append([x.box.x, x.box.y, x.box.w, x.box.h, x.cls == UNLIT or id(x) in dup, round(x.score, 3),
                               None if blk is None else round(blk, 3)])
            frames.append({
                "file": p.name, "t": offset(p.name), "size": [img.shape[1], img.shape[0]], "me": [me.x, me.y, me.w, me.h],
                "flames": [[f.x, f.y, round(f.r, 1), round(f.score, 3)] for f in found if f not in masked],
                "masked": [[f.x, f.y, round(f.r, 1), round(f.score, 3)] for f in masked],  # 在名字标签下面、被排除的
                "people": people,  # [x, y, w, h, 是不是 player_unlit, 分数, black()]
            })
        frames.sort(key=lambda f: f["t"])
        sp = repo / "runs" / d / "summary.json"  # 10-02 早的几次还没有
        summary = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        # 运行时那条线索最后看到火焰的时间（相对举起）和位置：存图稀，举起前的火焰可能一张都没存下来
        out[d.split("/")[-1]] = {"expect": expect, "frames": frames,
                                 "flame_last": summary.get("flame_last"), "flame_pos": summary.get("flame_pos")}
        print(d, expect, len(frames))
    path = Path(__file__).parent / "data" / "light_replay.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", path)


if __name__ == "__main__":
    main()
