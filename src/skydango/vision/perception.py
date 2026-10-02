"""YOLO 感知层：持续检测人物 / 名字标签 / 互动圆圈，汇总成"身边有谁、谁在发起互动、有没有陌生人"。

对外接口和 vision.env.EnvWatcher 一样（observe / nearby / describe / requests / labels / circles / last_seen），
身体、社交、眼睛不用改；`[perception] enabled = false` 就退回 EnvWatcher 的定时整图 OCR。

每一帧：
1. YOLO 出框（player / player_unlit / name_tag / social_ring / self），去掉面板挡住的、底部输入栏的、团子自己
2. 追踪：框接成轨迹
3. 名字标签轨迹：新出现就裁小图只跑 OCR 识别（不跑检测），多读几次投票，对上 friends.md 就是那个好友
   —— 身份跟着名字走，不跟着轨迹 id 走：转视角轨迹断了，新轨迹一读出名字就接回同一个人
4. 圆圈：挂到正上方的名字标签上，用现有的剪影模板认图标（牵手 / 拥抱 / ……）；上面没有标签的圆圈算陌生人的
5. 人物：未点火的黑影（player_unlit）一定是陌生人；点过火的陌生人外观和好友一样，
   只能靠头顶有没有名字标签分 —— 一直没有名字标签、离得不太远 → 陌生人
   （还有个办法是点一下人物，右边会打开好友树面板，看得出是不是好友；要点屏幕，没做）

二期（docs/superpowers/specs/2026-09-28-perception-phase2-design.md）：
- `sweep()`：转一圈截下的帧单独汇总成"哪个方向有谁"（vision/sweep.py 算方位、合并），顺带认出团子（self_box，之后代替 self_roi）
- 远近：人物框高 ÷ 团子框高；框持续变大、往画面中间走 → 朝团子走过来（`pop_approaches`，身体发 approach 事件）
- `typing`（头顶"正在输入"气泡）：挂到正下方的人身上；陌生人的消息靠它猜是画面上哪个人说的（`speaker_hint`）

认装扮（docs/superpowers/specs/2026-10-01-appearance-design.md，挂了 AppearanceBook 才有）：
- 人物框裁图算外观特征，挂着名字标签时学进记忆簿；名字标签看不到的人按外观认成好友（`maybe`，"像小明"），不算陌生人
- 点过火的陌生人按外观编号（"陌生人A"），走开又回来时认得出（`pop_stranger_backs`）

以上规则和阈值都还**没在真机验证**（模型还没训练），见 docs/superpowers/specs/2026-09-28-perception-yolo-architecture-v0.2.md。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import cv2
import numpy as np

from ..brain.images import difference, thumb
from ..chat.tracker import normalize, similar
from ..config import AppearanceConfig, EnvConfig, GestureConfig, PerceptionConfig, SocialConfig, SpinConfig
from ..game.social import IDLE, KIND_NAMES, LIGHT, LIGHT_KEY, Request, is_request
from .appearance import describe_crop, good_crop
from .attrs import unlit_score
from .candle import black, find_flame, white_ring
from .bubbles import Rect, roi_rect
from .catalog import Who, stranger_key
from .detect import Detection, Detector
from .embed import cosine, unit
from .gesture import ClipBuffer, eligible, person_crop
from .ocr import OcrEngine, join_lines
from .people import OBJECT_NAMES, OTHERS, WHO, CallSeen, Person, Seen, Thing, object_distance, side_of
from .sweep import STRANGER_WHO, UNKNOWN_WHO, UNLIT_WHO, Sighting, SweepResult, bearing, distance, find_self, merge
from .track import PAN_MIN_RESPONSE, PAN_RECHECK_RESPONSE, Track, Tracker, estimate_shift, iou
from .wardrobe import FRIEND as FRIEND_PRIORITY
from .wardrobe import ME as ME_PRIORITY
from .wardrobe import STRANGER as STRANGER_PRIORITY

log = logging.getLogger(__name__)

UNLIT = "player_unlit"  # 没点火的陌生人：黑色剪影
STRANGER = "陌生人"  # 陌生人头顶没有名字：发起的请求用这个名字（社交规则里按 stranger 处理）
SELF_MEMORY = 1.0  # 图鉴收集：团子框这么久内出现过的位置上冒出来的"陌生人"按团子算（YOLO 有的帧不给团子出 self 框）
SELF_IOU = 0.3  # 和团子框重叠这么多就算团子（_mark_dango 用 same_body 更严，团子身上的 player 框常比 self 框大一圈）
DISK_EVERY = 0.3  # 团子周围最多隔这么久找一次火焰
DISK_GAP = 1.0  # 火焰断开不超过这么久算同一条线索（火焰会晃）
SAME_BODY_IOU = 0.45  # same_body：团子身上的 player 框和 self 框
DANGO_MEMORY = 30.0  # 团子框（按聊天面板开 / 关分别记）这么久内见过：YOLO 没出 self 时，落在那个位置上的人物框就是团子
SELF_HOLD = 3.0  # 团子框丢了（黑影贴着团子时 YOLO 常认不出 self）最多沿用最近的框这么久：镜头跟着团子，屏幕位置几乎不变
LIT_LOW = 0.2  # 点亮陌生人开着时检测器按这个出框：晚上黑影分数低（10-01 晚 0.27 / 0.28），找"火焰下面那个人"时也看低分框
DIAG_EVERY = 0.5  # 点亮中每隔这么久存一张图（spec 2026-10-01-light-flame-around-self §5）
DIAG_MAX = 30  # 一次最多存这么多张
DIAG_RUNS = 50  # 一次运行最多存这么多次
FRAME_STALE = 0.5  # lit()：最近一次扫描比 FRAME_STALE + DISK_EVERY 更旧（感知没在跑 / 团子框丢了）就不判
LIT_SAME_IOU = 0.3  # 判点亮：变亮的人物框要和点亮中最近一次看到的"黑"框重叠这么多（同一个人；别的亮人走过来不算）
LIT_SCANS = 2  # 判点亮：连续这么多次扫描（每 DISK_EVERY 一次）都看到他变亮才算
ICON_OFFSET = 2.23  # 圆圈中心在名字标签上沿往下这么多倍标签高度（game-ops §6 实测）
PLACE_GAP = 3.0  # 画面大变后认地图：离上次至少隔这么久（转镜头时画面一直在变）
FAR_RETRY = 1.0  # 远处二次检测：同一条轨迹最多隔这么久裁一次（还要比 track_buffer 短）；挂上名字标签后这么久内也不裁
FAR_MISSES = 3  # 远处二次检测连着这么多次没找到名字标签……
FAR_BACKOFF = 5.0  # ……之后这么久才再裁一次
FAR_DUP_IOU = 0.3  # 二次检测找到的框和原图里已有的同类框重叠这么多就算同一个
UNKNOWN_MIN_SCORE = 0.9  # 没认出的名字：OCR 至少这么有把握才记下（三期 §5）
REQUEST_HOLD = 1.5  # 陌生人的圆圈还在原地、只是这一帧没认出图标（火焰会晃）：请求再留这么久，免得一闪就没、身体来不及点
PEOPLE_STALE = 1.0  # people()：最近一帧比这更旧（被挡住、没跑检测）就不再报里面的人
OCCLUSION = "occlusion"  # 多人同时消失 + 画面大变：玩家自己开了全屏界面（地图、商店……），暂停计时但检测照跑
# 认装扮开着时，没标签的人要攒够 min_samples 个好样本、先和好友比过外观才判陌生人（默认 every=3 要 9 帧，
# 比 stranger_after 长，不等的话标签被挡住的好友会先冒一次 stranger 事件）；一直攒不到好样本的（太远、被挡）
# 最多再多等这么久照样判
STRANGER_GRACE = 1.5
PAN_SCALE = 8  # 估计画面平移用的缩略图缩小这么多倍
RELINK_EARLY = 0.2  # 接回的候选轨迹最早可以比失踪记录的"最后看到"早这么久冒出来（同一帧里先删后建）
RELINK_PREDICT = 1.0  # 接回时按速度最多往前推这么久
RELINK_AMBIGUOUS = 1.3  # 一条记录的两个候选，近的要比远的近这么多倍才不算歧义
RELINK_MAX_KEEPS = 6  # 接回的"像他"这么多个 keep 还没被名字标签证实就摘掉（接错了总得有个出口）
LOW_ONLY_MAX = 5.0  # 只靠低分框续着的轨迹最多续命这么久（低分框可能是石像之类认错的东西）
TRACKING_SWITCHES = ("sticky_names", "track_low", "track_predict", "track_pan", "relink", "motion")  # 全关 = 原来的追踪
DISAGREE_SECONDS = 2.0  # 放行的轨迹 YOLO 侧和外形侧对点没点火持续相反这么久，才算 attrs_disagree 难例
PAN_KINDS = ("zoom", "move", "spin")  # 这几种镜头事件之后 camera_settle 秒内框高会突变：不更新速度、清走近 / 运动历史


def detector_conf(cfg: PerceptionConfig, light: bool = False, *, attrs: bool = False) -> float:
    """检测器的出框阈值：收集难例、低分框续轨迹、第二层复核低分框（attrs）时要看到 low_conf ~ conf 之间的框，
    点亮陌生人要看到 LIT_LOW 以上的黑影；判定仍按 conf。"""
    conf = min(cfg.low_conf, cfg.conf) if cfg.hardcases or cfg.track_low or attrs else cfg.conf
    return min(conf, LIT_LOW) if light else conf


def approaching(hist: list[tuple[float, float, float]], width: int, grow: float) -> bool:
    """(时间, 框高, 框中心 x) 的序列：前 1/3 和后 1/3 比，框高变大 ≥ grow，且往画面中间走（或一直在中间）。"""
    if len(hist) < 3:
        return False
    k = max(1, len(hist) // 3)
    h0 = sum(h for _, h, _ in hist[:k]) / k
    h1 = sum(h for _, h, _ in hist[-k:]) / k
    if h0 <= 0 or h1 < h0 * (1 + grow):
        return False
    c0 = sum(abs(x - width / 2) for _, _, x in hist[:k]) / k
    c1 = sum(abs(x - width / 2) for _, _, x in hist[-k:]) / k
    return c1 <= c0 or c1 < 0.15 * width


def motion_of(hist, now: float, cfg: PerceptionConfig) -> str | None:
    """(时间, 框高, 补偿过画面平移的中心 x) 的序列 → 走近 / 走远 / 往左走 / 往右走 / 站着；拿不准返回 None。

    只看最近 motion_window 秒，前 1/3 和后 1/3 比：径向 r = 框高比 − 1，横向 s = x 位移 ÷ 平均框高（单位是身高）。
    "往左 / 往右"是团子画面里的方向，不是地图方位。"""
    window = cfg.motion_window
    recent = [x for x in hist if now - x[0] <= window]
    if len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.6 * window:
        return None
    k = max(1, len(recent) // 3)
    h0 = sum(h for _, h, _ in recent[:k]) / k
    h1 = sum(h for _, h, _ in recent[-k:]) / k
    x0 = sum(x for _, _, x in recent[:k]) / k
    x1 = sum(x for _, _, x in recent[-k:]) / k
    if h0 <= 0 or h1 <= 0:
        return None
    r = h1 / h0 - 1
    side = (x1 - x0) / ((h0 + h1) / 2)
    rr, ss = abs(r) / cfg.motion_grow, abs(side) / cfg.motion_side
    if rr < 1 and ss < 1:
        return "站着"
    if rr >= ss:
        return "走近" if r > 0 else "走远"
    return "往右走" if side > 0 else "往左走"


def settle_motion(data: dict, value: str | None, now: float, hold: float) -> str | None:
    """防抖：新结论连续 hold 秒才换进 data["motion"]；None（拿不准）立刻生效。返回现在的结论。"""
    if value is None:
        data["motion"] = None
        data.pop("motion_cand", None)
    elif value == data.get("motion"):
        data.pop("motion_cand", None)
    else:
        cand = data.get("motion_cand")
        if cand is None or cand[0] != value:
            data["motion_cand"] = (value, now)
        elif now - cand[1] >= hold:
            data["motion"] = value
            data.pop("motion_cand", None)
    return data.get("motion")


def far_region(p: Rect, width: int, height: int) -> Rect | None:
    """远处小人头顶要再检测一次的区域：宽 3 倍框宽、高 2.5 倍框高（上方 2 个身高到身体上半截），水平居中，夹到画面内。"""
    x1, y1 = round(p.x + p.w / 2 - 1.5 * p.w), round(p.y - 2 * p.h)
    x2, y2 = x1 + 3 * p.w, y1 + round(2.5 * p.h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return Rect(x1, y1, x2 - x1, y2 - y1)


def _near(a: tuple[int, int], b: tuple[int, int], dist: int = 80) -> bool:
    return abs(a[0] - b[0]) <= dist and abs(a[1] - b[1]) <= dist


def one_self(dets: list[Detection]) -> list[Detection]:
    """一帧只有一个团子：留分数最高的 self，其余改成 player（实测模型会把躺在地上的别人也认成团子）。"""
    selfs = sorted((d for d in dets if d.cls == "self"), key=lambda d: d.score, reverse=True)
    extra = {id(d) for d in selfs[1:]}
    return [Detection("player", d.box, d.score) if id(d) in extra else d for d in dets]


def promote_weak_self(dets: list[Detection], low: list[Detection]) -> list[Detection]:
    """没有像样的 self 时，和低分 self 框几乎重合（IoU ≥ 0.5）的那个 player 就是团子，改成 self。

    实测（2026-09-30，好友站在团子正后方、镜头贴得近）：团子被认成 player 0.74、self 只有 0.29，
    好友的名字标签挂到了团子身上，look_person 裁了团子的背影，也找不到团子框、不换角度。只有团子会拿到 self 分数。"""
    if any(d.cls == "self" for d in dets):
        return dets
    pairs = [(iou(d.box, w.box), i) for i, d in enumerate(dets) if d.cls == "player" for w in low if w.cls == "self"]
    overlap, i = max(pairs, default=(0.0, -1))
    if overlap < 0.5:
        return dets
    out = list(dets)
    out[i] = Detection("self", dets[i].box, dets[i].score)
    return out


def same_body(a: Rect, b: Rect) -> bool:
    """两个框是不是同一个人的身体（IoU ≥ SAME_BODY_IOU）。团子身上的 player 框带着斗篷常比 self 框宽一圈（10-02 晚 IoU 0.55 左右）；
    不看"小框在大框里"：贴在团子身前的陌生人框常把团子框整个包住（play-1001-1 IoU 0.26 / 0.33），他不是团子。"""
    return iou(a, b) >= SAME_BODY_IOU


def people_boxes(dets: list[Detection]) -> list[Detection]:
    """一帧里的人（player / player_unlit / self）。模型可能在团子身上同时出 self 和 player 框：player 那个不算另一个人。"""
    selfs = [d for d in dets if d.cls == "self"]
    return selfs + [d for d in dets if d.cls in ("player", UNLIT) and not any(iou(d.box, s.box) >= 0.5 for s in selfs)]


def _center(r: Rect) -> tuple[float, float]:
    return r.x + r.w / 2, r.y + r.h / 2


def _inside(point: tuple[float, float], rect: Rect) -> bool:
    return rect.x <= point[0] < rect.x2 and rect.y <= point[1] < rect.y2


@dataclass
class Lost:
    """断掉的好友轨迹（spec §4.1）：keep 秒内在附近冒出来的没名字的人按位置接回成"像他"。"""

    name: str
    box: Rect  # 最后的框
    vx: float
    vy: float
    vh: float
    last: float  # 最后看到的时间
    pan: tuple[float, float]  # 当时的累计画面平移


@dataclass(frozen=True)
class Talker:
    """头顶正冒着气泡的人（空闲注意力用）。"""

    track_id: int
    name: str | None  # 好友名；陌生人 None
    friend: bool
    x: float  # 人物框中心 x（整图像素）
    last: float  # 最近一次看到气泡
    start: float  # 这次气泡开始（气泡断开超过 bubble_gap 再出现算新的一句）


@dataclass(frozen=True)
class OutfitNote:
    """好友的一套装扮（身体取走写进关系卡）：new = 卡里没有可比的，same / changed = 和卡里最近一套比，
    changed 也用于上线中途换装；described = 描述回来了（desc 有字）。"""

    name: str
    feat: list[float]  # 平均特征，3 位小数
    key: str  # 特征模型的 key
    state: str
    desc: str = ""


class PerceptionWatcher:
    def __init__(
        self,
        detector: Detector,
        ocr: OcrEngine,
        cfg: PerceptionConfig,
        env_cfg: EnvConfig,
        names: Callable[[], list[str]],
        log_roi: list[float],
        icons=None,  # game.social.IconClassifier
        background: bool = True,
        capture: Callable[[], np.ndarray] | None = None,  # capture = "own" 时感知线程自己截图用
        clock: Callable[[], float] = time.monotonic,
        scene_change: float = 0.25,  # 集体消失时，画面差异超过这个才算开了全屏界面（同 brain.scene_change）
        hardcases=None,  # vision.hardcases.HardCaseCollector：可能认错的画面存下来
        unknown=None,  # vision.unknownnames.UnknownNames：读得清楚但不在好友名单里的名字（三期 §5）
        places=None,  # vision.places.PlaceRecognizer：认地图（三期 §2）
        place_interval: float = 30.0,  # 每隔这么久认一次地图（画面大变后也认一次）
        gestures=None,  # vision.gesture.GestureClassifier：别人对团子做的动作（三期 §3，研究性质）
        gesture_cfg: GestureConfig | None = None,
        social_cfg: SocialConfig | None = None,  # 点亮陌生人（spec 2026-10-01）：和 flame 都有才找火焰圆盘
        flame: np.ndarray | None = None,
        light_dir: Path | None = None,  # 点亮陌生人每次存图 / summary.json 的目录（None = 不存）
        appearance=None,  # vision.appearance.AppearanceBook：认装扮（None = 不认，行为照旧）
        embedder=None,  # 外观特征模型（appearance.make_embedder）
        appearance_cfg: AppearanceConfig | None = None,
        saver=None,  # 攒认人模型的训练数据（CropSaver，None = 不存）
        call_window: float = 6.0,  # 按 Q 喊一声的窗口秒数（[call] window）
        camera_settle: float = 0.6,  # 镜头缩放 / 走路 / 转圈后这么久画面才稳（同 [track] settle）
        attrs=None,  # vision.attrs.PersonAttrs：第二层（复核低分框、撤下误框、点没点火两边投票；None = 不接，行为照旧）
        catalog=None,  # vision.catalog.CatalogCollector：图鉴收集（None = 不收，行为照旧）
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.names = names
        self.log_roi = log_roi
        self.icons = icons
        self.background = background
        self.capture = capture
        self.clock = clock
        self.scene_change = scene_change
        self.hardcases = hardcases
        self.unknown = unknown
        self.places = places
        self.place_interval = place_interval
        self._place_run = float("-inf")
        self._place_thumb: np.ndarray | None = None
        self._place_misses = 0  # 连着几次认不出
        self.gestures = gestures
        self.gesture_cfg = gesture_cfg or GestureConfig()
        self.light_cfg = social_cfg
        self.flame = flame
        self._flame: dict | None = None  # 团子身边现在的火焰线索（spec 2026-10-01-light-flame-around-self §3；self._lock 保护）
        self.light_dir = light_dir
        self._diag: dict | None = None  # 当前这次点亮的存图：dir / count / next / requested
        self._diag_runs = 0
        self._flame_seq = 0
        self._flame_check = float("-inf")
        self._flame_log = float("-inf")  # DEBUG 日志每秒最多一行
        self._frame_at = float("-inf")  # 最近一帧跑过 tracker.update 的时间（trackeval 用它认出被挡住提前返回的帧）
        self._lighting: dict | None = None  # 举着蜡烛等结果（mark_tried ~ light_done）
        self._cooldown_until = float("-inf")  # 没点亮之后这之前不出请求
        self._scan: dict | None = None  # 最近一次找火焰：area / me / flame（存图用）
        self._people_boxes: list[tuple[Rect, float]] = []  # 这一帧的人物框（含 LIT_LOW 以上的低分框）
        self._gestures: deque[tuple[str, str]] = deque(maxlen=50)  # (好友名, 动作)，身体取走；没人取（普通 Agent）时只留最近的
        self._gesture_at: dict[tuple[str, str], float] = {}
        self.appearance = appearance
        self.embedder = embedder
        self.appearance_cfg = appearance_cfg or AppearanceConfig()
        self.saver = saver
        self.catalog = catalog
        self._self_seen: tuple[float, list[Rect]] | None = None  # 最近一次看到团子框的时间和框（图鉴收集认团子用）
        self.call_window = call_window
        self._call: CallSeen | None = None  # 最近一次呼喊窗口（只留一次）；self._lock 保护
        self._call_span = (float("-inf"), float("-inf"))  # 窗口 [开始, 结束]：暂停时跟着往后挪（at 不变，身体拿它取结果）
        self._stranger_backs: list[str] = []  # 走开又回来的陌生人编号，身体取走
        self.wardrobe = None  # vision.wardrobe.Wardrobe：描述装扮（cli 建好大脑后挂上；None = 只判换装、不描述）
        self._outfits: list[OutfitNote] = []  # 好友的装扮，身体取走
        self._describe_want: set[tuple[str, str]] = set()  # 要（重新）描述的好友 / 团子，描述回来才去掉
        self._describe_asked: dict[tuple[str, str], float] = {}  # 上次 request 成功的时间：retry_after 内不再问
        self.keep = cfg.keep  # 身体说"走开了"时用
        self.attrs = attrs
        self.tracker = Tracker(  # 同一个人可能两类来回变
            cfg.track_buffer, cfg.track_iou, cross=frozenset({"player", UNLIT}),
            center_gate=cfg.track_center_gate if cfg.track_predict else 0.0, predict=cfg.track_predict,
            # 第二层开着：人物的低分框配不上旧轨迹也开一条待复核的（复核放行前不进 players）
            open_low=frozenset({"player", UNLIT}) if self._attrs_on() else frozenset(),
        )
        self.camera_settle = camera_settle
        self._quiet_until = float("-inf")  # 镜头缩放 / 走路 / 转圈：这之前的帧不攒走近 / 运动历史
        self._pan = (0.0, 0.0)  # 累计的画面平移（整图像素；失踪记录、运动方向用）
        self._pan_thumb: np.ndarray | None = None  # 上一帧的平移缩略图（暂停恢复后作废）
        self.last_shift: tuple[float, float] | None = None  # 这一帧估出的画面平移（track-eval 用）
        self._lost: dict[str, Lost] = {}  # 名字 → 断掉的好友轨迹（relink）
        self._others: list[Track] = []  # 这一帧外形是先祖 / 共享空间的放行轨迹（不在 players 里，people() / objects() 用）
        self._camera_held = False  # 这次暂停里有身体转镜头（hold("camera")）
        self._pan_recheck = False  # 恢复后第一帧：拿暂停前的缩略图估一次平移，估不出就作废所有轨迹的位置
        self.requests: dict = {}  # 名字 → game.social.Request
        self.labels: dict[str, tuple[int, int, int, int, float]] = {}
        self.circles: dict[str, tuple[str | None, float]] = {}
        self.last_seen: dict[str, float] = {}
        self.place = ""  # 认地图认出的地方（三期 §2，没挂 places 时一直空）；和 EnvWatcher 同名
        self.place_at = float("-inf")
        self.last_dets: list[Detection] = []  # 最近一帧的检测（调试画框用）
        self.last_tracks: list[Track] = []
        self.last_low: list[Detection] = []  # 最近一帧置信度在 conf 以下的框（不进追踪，给难例收集看）
        self.timings: deque[tuple[float, float]] = deque(maxlen=300)  # (检测 ms, 整帧 ms)
        self.far_runs = 0  # 远处二次检测跑了几次（测速 / compare 用）
        self.self_box: Rect | None = None  # 转圈认出的团子（sweep）；有它就不用 self_roi，下次转圈前一直用
        self._me_last: tuple[Rect, float] | None = None  # 沿用的最近一次团子框：(box, time)，黑影贴着时认不出 self
        self._dango_mem: dict[bool, tuple[Rect, float]] = {}  # 聊天面板开着 / 关着 → 最近一次高分 self 框和时间
        self._frame_h = 1080  # 最近一帧的高度（算远近用）
        self._frame_w = 1920  # 最近一帧的宽度（算在画面哪边用）
        self._approaches: list[str] = []  # 朝团子走过来的人（好友名 / STRANGER），身体取走
        self._approach_at: dict[str, float] = {}
        self._approach_log: deque[tuple[str, int, float]] = deque()  # (谁, 轨迹 id, 时间)：空闲注意力用，不取走
        self._typing: deque[tuple] = deque()  # (时间, 轨迹 id, 是好友, 没点火, 框中心 x, 框高, 画面宽, 画面高)
        self._strangers: deque[tuple[float, int]] = deque()  # (时间, 这一帧有几个陌生人)
        self._panel_visible = False
        self._holds: Counter = Counter()  # 暂停的原因 → 次数（可以嵌套）
        self._held_since: float | None = None
        self._occluded_at = 0.0
        self._prev_count = 0  # 上一帧看到几个人物 + 名字标签
        self._prev_thumb: np.ndarray | None = None
        self._last_run = float("-inf")
        self._busy = False
        self._lock = threading.Lock()
        self._infer = threading.Lock()  # 检测器 / OCR：sweep（身体线程）和后台感知线程可能同时用
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 画面被挡时暂停计时（一期设计 §4） ----
    @property
    def paused(self) -> bool:
        return self._held_since is not None

    def hold(self, reason: str) -> None:
        """开始暂停：不跑检测、不累计"没看到"的时间。原因可以叠加，全部 release 才恢复。"""
        with self._lock:
            if self._held_since is None:
                self._held_since = self.clock()
                log.debug("感知暂停：%s", reason)
            self._holds[reason] += 1
            if reason == "camera":
                self._camera_held = True

    def release(self, reason: str) -> None:
        with self._lock:
            if self._holds[reason] <= 0:
                self._holds.pop(reason, None)
                return
            self._holds[reason] -= 1
            if self._holds[reason] <= 0:
                del self._holds[reason]
            if self._holds or self._held_since is None:
                return
            since, self._held_since = self._held_since, None
        self._resume(since, reason)

    @contextmanager
    def held(self, reason: str) -> Iterator[None]:
        self.hold(reason)
        try:
            yield
        finally:
            self.release(reason)

    def _release_all(self, why: str) -> None:
        with self._lock:
            if self._held_since is None:
                return
            since, self._held_since = self._held_since, None
            self._holds.clear()
        self._resume(since, why)

    def _resume(self, since: float, why: str) -> None:
        """暂停了 d 秒：把所有"最后看到"的时间往后挪 d，恢复后要再过 keep 秒没看到才算走开。"""
        now = self.clock()
        d = now - since
        log.debug("感知恢复（%s），暂停了 %.1f 秒", why, d)
        self._prev_count, self._prev_thumb = 0, None  # 暂停前那一帧不能拿来判"集体消失"（镜头可能已经转走了）
        if self._camera_held and any(getattr(self.cfg, k) for k in TRACKING_SWITCHES):
            # 暂停期间身体转了镜头：留着暂停前的缩略图，恢复后第一帧估一次平移，估不出再作废位置（process 里）
            self._pan_recheck = True
        else:
            self._pan_thumb = None  # 平移不能跨暂停估
        self._camera_held = False
        self.tracker.calm(now)  # 速度也不能跨暂停用
        for track in list(self.tracker.tracks.values()):  # 暂停前后的框高不能连起来判"走过来"（镜头可能动过）
            track.data.pop("hist", None)
            track.data.pop("motion_hist", None)
            track.data.pop("clip", None)  # 动作片段也不能跨暂停拼起来
        if d <= 0:
            return
        for name, t in list(self.last_seen.items()):
            self.last_seen[name] = min(t + d, now)
        for lost in list(self._lost.values()):
            lost.last = min(lost.last + d, now)
        for track in list(self.tracker.tracks.values()):
            if "relink_at" in track.data:
                track.data["relink_at"] += d
        for name, (x, y, w, h, t) in list(self.labels.items()):
            self.labels[name] = (x, y, w, h, min(t + d, now))
        for name, (kind, t) in list(self.circles.items()):
            self.circles[name] = (kind, min(t + d, now))
        with self._lock:
            self._strangers = deque((min(t + d, now), n, u) for t, n, u in self._strangers)
        self.tracker.shift(d)
        with self._lock:
            if self._call is not None and not self._call.ended:  # 呼喊窗口跟着往后挪：暂停期间看不到的不算
                self._call_span = (self._call_span[0] + d, self._call_span[1] + d)
        if self.appearance is not None:  # 认装扮的陌生人编号也一样：暂停的时间不算走开，别报"又回来了"
            self.appearance.shift(d, now)
        with self._lock:  # 火焰线索 / 点亮中的时间也跟着挪：暂停的时间不算"火焰消失"
            if self._flame is not None:
                self._flame["first"] += d
                self._flame["last"] += d
            if self._lighting is not None:
                for key in ("raised", "flame_last", "person_at", "scan_at"):
                    self._lighting[key] += d  # -inf 加 d 还是 -inf
            self._cooldown_until += d

    def _check_hold_max(self) -> None:
        since = self._held_since
        if since is not None and self.clock() - since > self.cfg.hold_max:
            log.warning("感知暂停超过 %.0f 秒（%s），自动恢复", self.cfg.hold_max, "、".join(self._holds) or "?")
            self._release_all("hold_max")

    def _blocked(self) -> bool:
        """暂停且要跳过检测（只因为集体消失而暂停时检测照跑，等人重新出现）。"""
        return self.paused and set(self._holds) != {OCCLUSION}

    def _occlusion(self, frame: np.ndarray, dets: list[Detection]) -> bool:
        """集体消失规则。返回 True = 这一帧什么都看不到、正在暂停，不更新状态。"""
        count = sum(d.cls in ("player", UNLIT, "name_tag") for d in dets)
        small = thumb(frame)
        prev_count, prev_thumb = self._prev_count, self._prev_thumb
        self._prev_count, self._prev_thumb = count, small
        if self._holds.get(OCCLUSION):
            if count:
                self.release(OCCLUSION)
                return False
            if self.clock() - self._occluded_at > self.cfg.occlusion_hold:
                log.info("人还是都看不到（%.0f 秒），不再等了，照常计时", self.cfg.occlusion_hold)
                self.release(OCCLUSION)
            return True
        if prev_count >= 2 and count == 0 and prev_thumb is not None and difference(prev_thumb, small) > self.scene_change:
            log.info("%d 个人同时不见了、画面大变（可能开了全屏界面），暂停计时", prev_count)
            self._occluded_at = self.clock()
            self.hold(OCCLUSION)
            return True
        return False

    def camera_moved(self, at: float, kind: str) -> None:
        """身体动了镜头（turn / zoom / move / spin）。转镜头靠平移估计补偿，什么都不清；
        缩放、走路、转圈时框高会突变：settle 秒内不更新速度、清走近 / 运动历史（不能当成人在走近走远）。"""
        if kind in PAN_KINDS:
            until = at + self.camera_settle
            self._quiet_until = max(self._quiet_until, until)
            self.tracker.calm(until)

    def _pan_step(self, frame: np.ndarray, dets: list[Detection], panel_visible: bool) -> tuple[float, float] | None:
        """这一帧相对上一帧的画面平移（整图像素）：1/8 灰度缩略图的上半部分，聊天面板开着去掉左边三分之一，
        人物 / 团子 / 名字标签那块不用（人自己会走）。估出来就累加进 self._pan。"""
        height, width = frame.shape[:2]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (max(8, width // PAN_SCALE), max(8, height // PAN_SCALE)), interpolation=cv2.INTER_AREA)
        small = small[: small.shape[0] // 2].astype(np.float32)
        prev, self._pan_thumb = self._pan_thumb, small
        if prev is None or prev.shape != small.shape:
            return None
        mask = np.ones(small.shape, bool)
        if panel_visible:
            mask[:, : small.shape[1] // 3] = False
        for d in dets:
            if d.cls in ("player", UNLIT, "self", "name_tag", "typing"):
                b = d.box
                mask[max(0, b.y // PAN_SCALE): (b.y2 + PAN_SCALE - 1) // PAN_SCALE,
                     max(0, b.x // PAN_SCALE): (b.x2 + PAN_SCALE - 1) // PAN_SCALE] = False
        shift = estimate_shift(prev, small, mask, PAN_RECHECK_RESPONSE if self._pan_recheck else PAN_MIN_RESPONSE)
        if shift is None:
            return None
        shift = (shift[0] * PAN_SCALE, shift[1] * PAN_SCALE)
        self._pan = (self._pan[0] + shift[0], self._pan[1] + shift[1])
        return shift

    def _frozen(self, now: float) -> float:
        """暂停期间按暂停开始那一刻算：给身体的结果停在暂停前。"""
        since = self._held_since
        return min(now, since) if since is not None else now

    # ---- 帧从哪来 ----
    def observe(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        """身体主循环每圈调一次。capture = "own" 时只记下面板开没开，帧由感知线程自己截。"""
        self._panel_visible = panel_visible
        self._check_hold_max()
        if self._blocked():
            return
        if self.cfg.capture == "own" and self.capture is not None:
            self._ensure_thread()
            return
        if now - self._last_run < 1.0 / max(self.cfg.fps, 0.1):
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._last_run = now
        if self.background:
            threading.Thread(target=self._guarded, args=(frame, now, panel_visible), name="perception", daemon=True).start()
        else:
            self._guarded(frame, now, panel_visible)

    def _guarded(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        try:
            self.process(frame, now, panel_visible)
        except Exception:
            log.exception("感知出错")
        finally:
            with self._lock:
                self._busy = False

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.fps, 0.1)
        while not self._stop.is_set():
            started = self.clock()
            self._check_hold_max()
            if self._blocked():
                self._stop.wait(period)
                continue
            try:
                self.process(self.capture(), started, self._panel_visible)
            except Exception:
                log.exception("感知出错")
                self._stop.wait(1.0)
            self._stop.wait(max(0.0, period - (self.clock() - started)))

    def stop(self) -> None:
        self._stop.set()
        if self.catalog is not None:
            try:
                self.catalog.close()
            except Exception:
                log.exception("图鉴收集收尾出错")

    # ---- 一帧 ----
    def process(self, frame: np.ndarray, now: float, panel_visible: bool) -> None:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        self._frame_h, self._frame_w = height, width
        dets = one_self(self._filter(self._detect(frame), width, height, panel_visible))
        detected = time.perf_counter()
        low_all = [d for d in dets if d.score < self.cfg.conf]
        attrs_on = self._attrs_on()
        # 检测器为点亮陌生人放低到 LIT_LOW 后，promote_weak_self / 难例 / last_low 仍只看原来阈值以上的低分框
        floor = detector_conf(self.cfg, attrs=attrs_on)
        low = [d for d in low_all if d.score >= floor]
        dets = promote_weak_self([d for d in dets if d.score >= self.cfg.conf], low)
        if self._occlusion(frame, dets):
            return
        shift = self._pan_step(frame, dets, panel_visible) if self.cfg.track_pan else None
        self.last_shift = shift
        if self._pan_recheck:
            self._pan_recheck = False
            if shift is None:
                # 转了镜头、又估不出转了多少：原来屏幕位置上的人已经不是原来那个了。轨迹全作废（不记失踪），按名字重新认
                log.debug("转过镜头、估不出平移：%d 条轨迹的位置作废", len(self.tracker.tracks))
                if self.catalog is not None:
                    try:
                        self.catalog.dropped(list(self.tracker.tracks.values()))
                    except Exception:
                        log.exception("图鉴收集出错")
                self.tracker.tracks.clear()
                self._lost.clear()
        tracks = self.tracker.update(dets, now, low=low if self.cfg.track_low or attrs_on else (), shift=shift)
        self._frame_at = now
        if self.attrs is not None:
            self._review(frame, tracks, len(dets), now, width, height, panel_visible)
        if self.cfg.relink:
            self._note_lost(now)
        selfs = [t for t in tracks if t.cls == "self"]
        self._mark_dango(tracks, selfs, now, panel_visible)
        players = [t for t in tracks if t.cls in ("player", UNLIT) and not t.data.get("dango") and self._admitted(t)]
        if self.attrs is not None:
            # 外形是先祖 / 共享空间的：看得到，但不算陌生人、不挂名字、不进接回 / 续命 / 走近 / 运动 / 动作 / 认装扮（spec §3.5）
            others = [t for t in players if self._other_form(t)]
            players = [t for t in players if not self._other_form(t)]
            for t in others:
                t.data.pop("stranger", None)  # 投成先祖之前可能被当成过陌生人
            self._others = others
        tags = [t for t in tracks if t.cls == "name_tag"]
        rings = [t for t in tracks if t.cls == "social_ring"]
        bubbles = [t for t in tracks if t.cls == "typing"]
        for p in players:
            p.data["pan_at"] = self._pan  # 这一刻的累计平移（轨迹断了以后失踪记录用）
            if now < self._quiet_until:
                for key in ("hist", "motion_hist", "motion_cand"):
                    p.data.pop(key, None)
                if p.data.get("motion") is not None:
                    p.data["motion"] = None
        extra = self._far_tags(frame, players, tags, now, width, height, panel_visible)
        if extra:
            tags += [t for t in extra if t.cls == "name_tag"]
            rings += [t for t in extra if t.cls == "social_ring"]
            tracks = tracks + extra

        friends = self.names()
        for tag in tags:
            self._read_name(frame, tag, now, friends)

        owner: dict[int, Track] = {}  # 名字标签轨迹 id → 它下面的圆圈
        orphans: list[Track] = []  # 上面没有名字标签的圆圈
        for ring in rings:
            tag = self._tag_above(ring, tags)
            if tag is not None and tag.id not in owner:
                owner[tag.id] = ring
            else:
                orphans.append(ring)

        tagged = self._assign_tags([p for p in players if not self._unlit(p)], tags)
        assigned = {t.id for t in tagged.values()}
        seen = []
        for tag in tags:
            name = tag.data.get("name")
            if not name:
                continue
            b = tag.box
            edge = self._offscreen(tag, width) if tag.id not in assigned else None
            if edge:
                # 名字贴在屏幕边上 = 好友在画面外（spec 2026-10-01-q-call §1.3）：不算在身边（不然一喊就冒出一串假的"来到身边"），
                # 位置照记（盯人 track 靠它往画面外转），呼喊窗口里记进结果
                self.labels[name] = (b.x, b.y, b.w, b.h, now)
                self._call_note(name, Seen(edge, None, on_screen=False), now)
                continue
            self.last_seen[name] = now
            self.labels[name] = (b.x, b.y, b.w, b.h, now)
            seen.append(name)
            ring = owner.get(tag.id)
            kind = self._classify(frame, ring) if ring is not None else None  # 牵着手时圆圈会消失 → None
            if ring is not None:
                ring.data["kind"] = kind
            self.circles[name] = (kind, now)
            if is_request(kind):
                if name not in self.requests or self.requests[name].kind != kind:
                    log.info("%s 发起了互动：%s", name, kind)
                self.requests[name] = Request(name, kind, self._ring_center(ring), now)
            else:
                self.requests.pop(name, None)

        stranger_req = None
        for ring in orphans:
            kind = ring.data["kind"] = self._classify(frame, ring)
            if kind == "candle" and self._disk_like(frame, ring):
                kind = ring.data["kind"] = None  # 深色圆盘里的火焰：团子能去点亮他，不是他要给团子点火
            if is_request(kind):
                stranger_req = Request(STRANGER, kind, self._ring_center(ring), now)
                break
        old = self.requests.get(STRANGER)
        if stranger_req is not None:
            if old is None or old.kind != stranger_req.kind:
                log.info("陌生人发起了互动：%s", stranger_req.kind)
            self.requests[STRANGER] = stranger_req
        elif old is not None and now - old.seen_at <= REQUEST_HOLD and any(
            ring.data.get("kind") is None and _near(self._ring_center(ring), old.pos) for ring in orphans
        ):
            pass  # 圆圈还在原地、这一帧没认出图标：先留着（seen_at 不更新，最多留 REQUEST_HOLD 秒）
        else:
            self.requests.pop(STRANGER, None)

        strangers = unlit = 0
        shown = {t.data["name"] for t in tags if t.data.get("name")}
        for player in players:
            name = player.data.get("name")
            if name in shown and player.id not in tagged:
                # 这个名字的标签此刻清清楚楚在别处：之前是挂错了（好友从他身后走过），摘掉。
                # 标签只是被挡住（这一帧没看到）时不摘
                log.debug("轨迹 %d 不是 %s（标签在别处），摘掉名字", player.id, name)
                for key in ("name", "tagged", "tag_at"):
                    player.data.pop(key, None)
            relinked = player.data.get("maybe_by") == "relink"
            if relinked and (self._unlit(player) or (player.data.get("maybe") in shown and player.id not in tagged)):
                player.data.pop("maybe", None)  # 接回的人变成黑影 / 他的标签清清楚楚在别处：不是他
                player.data.pop("maybe_by", None)
            tag = tagged.get(player.id)
            if not self._unlit(player) and tag is not None:
                player.data["tag_at"] = now
                if tag.data.get("name"):
                    player.data["name"] = tag.data["name"]
                    if relinked:  # 名字永远说了算（是不是他都一样）
                        player.data.pop("maybe", None)
                        player.data.pop("maybe_by", None)
                player.data["tagged"] = True  # 这条轨迹上出现过名字标签：不是陌生人（标签被挡一下不改判）
        if self.cfg.relink:
            self._relink(players, shown, now)
        fresh = {}
        if self.appearance is not None:
            fresh = self._appearance_features(frame, players, selfs, now, width, height, panel_visible)
            self._appearance_identify(frame, players, selfs, tagged, shown, fresh, now, tracks)
        for player in players:
            if self._unlit(player):  # 没点火的黑影：一定是陌生人，远近都算
                player.data["stranger"] = True
                strangers += 1
                unlit += 1
                continue
            maybe = player.data.get("maybe")
            is_stranger = (
                not player.data.get("tagged")
                and not maybe
                and now - player.first >= self.cfg.stranger_after
                and player.box.h >= self.cfg.stranger_min_height * height
                and self._looks_checked(player, now)
            )
            player.data["stranger"] = is_stranger
            strangers += is_stranger
            if maybe and now - self.last_seen.get(maybe, float("-inf")) <= self.cfg.keep:
                self.last_seen[maybe] = now  # 好友还在身边、只是名字标签被挡住：别冒出"走开了"（已经走开的不靠外观接回来）
            if is_stranger and self.appearance is not None:
                self._appearance_stranger(frame, player, players, fresh.get(player.id), now)
        if self.cfg.sticky_names:
            self._keep_named(players, now)
        self._call_tick(players, tags, tagged, assigned, now, width, height)
        if self.appearance is not None:
            self.appearance.forget(now)
            if self.saver is not None:
                self._save_samples(frame, players, tagged, fresh, now)
        # 认得出是谁的人（好友）不算"火焰下面那个人"：好友本来就是亮的，站在火焰旁边会被当成点亮了
        known = [t.box for t in players if t.data.get("name") or t.data.get("tagged") or t.data.get("maybe")]
        # 第二层没放行 / 撤下的人物框（树、石像……）也不算（spec 2026-10-02-perception-attrs §3.3）
        # 先祖 / 共享空间的人身上也不会有能点的火焰。第二层出错自己关掉后就不藏了（留下的待复核轨迹 admitted 一直是 False）
        hidden = ([t.box for t in tracks if t.cls in ("player", UNLIT) and not self._admitted(t)] + [t.box for t in self._others]
                  if self._attrs_on() else [])
        self._people_boxes = [
            (d.box, d.score) for d in people_boxes(dets + [x for x in low_all if x.score >= LIT_LOW])
            if d.cls != "self" and not any(iou(d.box, k) >= 0.5 for k in known + hidden)
        ]
        bonfires = [t for t in tracks if t.cls == "bonfire"]
        self._watch_flames(frame, tags, bonfires, now, width, height, panel_visible)
        self._watch_typing(bubbles, players, selfs, now, width, height)
        if not self.paused:
            if now >= self._quiet_until:  # 镜头缩放 / 走路后画面还没稳：框高变化不是人在走
                self._watch_approach(players, now, width)
                if self.cfg.motion:
                    self._watch_motion(players, now)
            if self.gestures is not None:
                self._watch_gestures(frame, players, now, width, height)
        with self._lock:  # 身体线程会同时读（strangers()）
            self._strangers.append((now, strangers, unlit))
            while self._strangers and now - self._strangers[0][0] > self.cfg.keep:
                self._strangers.popleft()

        self.last_dets, self.last_tracks, self.last_low = dets, tracks, low
        if self.places is not None and not self.paused:
            self._watch_place(frame, dets, now)
        if self.hardcases is not None and not self.paused:
            try:
                self.hardcases.check(frame, now, tracks, low, set(seen), panel_visible)
            except Exception:
                log.exception("收集难例出错")
        if self.catalog is not None:
            try:
                if self.tracker.dropped:
                    self.catalog.dropped(self.tracker.dropped)
                if not self.paused:
                    panel = roi_rect(self.log_roi, width, height) if panel_visible else None
                    if selfs:
                        self._self_seen = (now, [s.box for s in selfs])
                    self.catalog.update(frame, players, selfs, now, panel, self.place, self._catalog_who)
            except Exception:
                log.exception("图鉴收集出错")
        self.timings.append(((detected - started) * 1000, (time.perf_counter() - started) * 1000))
        if seen:
            log.debug("旁边看到: %s", "、".join(seen))

    # ---- 第二层（spec 2026-10-02-perception-attrs §3.3、§3.4） ----
    def _attrs_on(self) -> bool:
        return self.attrs is not None and self.attrs.enabled

    def _unlit(self, t: Track) -> bool:
        """人物轨迹是不是没点火的黑影：第二层开着时 YOLO 和外形两边投票，否则就看 YOLO 类别。"""
        return self.attrs.is_unlit(t) if self._attrs_on() else t.cls == UNLIT

    def _other_form(self, t: Track) -> str | None:
        """第二层开着、外形投票是先祖 / 共享空间的人物轨迹：返回外形（"spirit" / "shared"），否则 None。"""
        if not self._attrs_on():
            return None
        form = t.data.get("form")
        return form[0] if form and form[0] in OTHERS else None

    @staticmethod
    def _admitted(t: Track) -> bool:
        """第二层放没放行这条人物轨迹（没接第二层时不写这个键 = 都放行）。"""
        return t.data.get("admitted", True)

    def _catalog_who(self, t: Track) -> Who | None:
        """图鉴收集的身份（spec 2026-10-02-catalog-collect §2）：团子 / 名字标签证实的好友 / 感知层判过的陌生人（或"像小明"）；黑影不收。
        陌生人不比感知层判得早（data["stranger"] 在调收集器之前算好）；有标签但不在好友名单里的不收（unknown_names 记）。
        process() 交给收集器的 players 已经去掉了第二层没放行和先祖 / 共享空间的。"""
        if t.cls == "self":
            return Who("团子", "self", True)
        if self._unlit(t):
            return None
        d = t.data
        tagged = d.get("tagged")
        if tagged and d.get("name"):
            return Who(d["name"], "friend", True)
        if self._near_dango(t):
            return None  # 团子身上的 player 框（_mark_dango 没认出来的）：团子自己有一份，不当陌生人收
        if tagged:
            return None
        if d.get("stranger") or d.get("maybe"):
            return Who(stranger_key(t.id), "stranger", False, d.get("maybe"))
        return None

    def _near_dango(self, t: Track) -> bool:
        """这条人物轨迹是不是落在团子刚待过的地方（SELF_MEMORY 秒内的团子框、重叠 ≥ SELF_IOU）。"""
        if self._self_seen is None:
            return False
        at, boxes = self._self_seen
        return t.last - at <= SELF_MEMORY and any(iou(t.box, b) >= SELF_IOU for b in boxes)

    def _review(self, frame: np.ndarray, tracks: list[Track], strong: int, now: float,
                width: int, height: int, panel_visible: bool) -> None:
        """tracker.update 之后：前 strong 条是高分框接上 / 新开的（记 strong，第一次就记），交给第二层复核、投票，
        再给这一帧的每条人物轨迹写 admitted。感知暂停时不复核（放行沿用上一帧的投票）。推理连续出错关掉后不再开待复核的轨迹。"""
        for t in tracks[:strong]:
            t.data["strong"] = True
        if not self.paused:
            panel = roi_rect(self.log_roi, width, height) if panel_visible else None
            self.attrs.update(frame, tracks, now, panel)
        if not self.attrs.enabled and self.tracker.open_low:
            log.info("人物属性关掉了：不再开待复核的低分轨迹")  # 关掉的 WARNING attrs 自己记过
            self.tracker.open_low = frozenset()
        for t in tracks:
            if t.cls in ("player", UNLIT):
                t.data["admitted"] = self.attrs.admit(t)
        self._report_attrs(frame, tracks, now)

    def _report_attrs(self, frame: np.ndarray, tracks: list[Track], now: float) -> None:
        """第二层的难例（spec §8）：attrs_reject = 一条高分轨迹被撤下的那一帧；attrs_disagree = 放行的轨迹 YOLO 侧和外形侧
        对点没点火持续相反 DISAGREE_SECONDS 秒。每条轨迹各报一次；暂停时不报。"""
        if self.hardcases is None or self.paused:
            return
        for t in tracks:
            if t.cls not in ("player", UNLIT):
                continue
            d = t.data
            if d.get("rejected") and d.get("strong") and not d.get("reject_reported"):
                d["reject_reported"] = True
                self._report_hard(frame, now, "attrs_reject", f"高分框 {t.score:.2f} 被第二层撤下（不是人）", tracks)
            side = self._disagree(t) if d.get("admitted") else None
            if side is None:
                d.pop("disagree_since", None)
            elif d.setdefault("disagree_since", now) <= now - DISAGREE_SECONDS and not d.get("disagree_reported"):
                d["disagree_reported"] = True
                self._report_hard(frame, now, "attrs_disagree", side, tracks)

    def _disagree(self, t: Track) -> str | None:
        """YOLO 侧（最近帧里 player_unlit 的比例）和外形侧（unlit 占 lit / unlit / shared / morph 的比例）分在 0.5 两边时的说明。"""
        hist, mean = t.data.get("cls_hist"), t.data.get("form_mean")
        if not hist or not mean or t.data.get("form_n", 0) == 0:
            return None
        den = sum(mean.get(k, 0.0) for k in ("lit", "unlit", "shared", "morph"))
        if den <= 0:
            return None
        yolo, form = sum(1 for c in hist if c == UNLIT) / len(hist), mean.get("unlit", 0.0) / den
        if (yolo > 0.5) == (form > 0.5):
            return None
        return f"YOLO 侧黑影占 {yolo:.2f}，外形侧 {form:.2f}，对点没点火意见相反"

    def _report_hard(self, frame: np.ndarray, now: float, reason: str, detail: str, tracks: list[Track]) -> None:
        try:
            self.hardcases.report(frame, now, reason, detail, tracks)
        except Exception:
            log.exception("收集难例出错")

    def _keep_named(self, players: list[Track], now: float) -> None:
        """轨迹续命（spec 2026-10-01-q-call §1.1）：好友稍远一点头顶的名字标签就淡掉，人还在画面里。
        挂过名字标签的轨迹只要这一帧还接得上，就一直算在身边；轨迹断了（track_buffer）才开始算 keep。
        按外观认的（maybe）不算；交叉走过身份换错时，名字跟到另一个人身上，标签再亮时纠正（同名标签在别处就摘名字）。"""
        for p in players:
            name = p.data.get("name")
            if (not self._unlit(p) and name and p.data.get("tagged") and now - p.last <= PEOPLE_STALE
                    and now - p.strong_last <= LOW_ONLY_MAX):  # 只靠低分框续着的最多续 LOW_ONLY_MAX 秒（低分框可能认错）
                self.last_seen[name] = now

    # ---- 按 Q 喊一声（spec 2026-10-01-q-call §1.2、§1.3） ----
    def called(self, at: float, *, by_self: bool = True) -> None:
        """团子按了 Q：开呼喊窗口 [at, at + call_window]，被动收这段时间里亮出来的名字。新窗口顶掉旧的。"""
        with self._lock:
            self._call = CallSeen(at, {})
            self._call_span = (at, at + self.call_window)
        log.debug("呼喊窗口：%.1f ~ %.1f", at, at + self.call_window)

    def call_result(self, at: float) -> CallSeen | None:
        """窗口结束后才有；at 对不上（被新的一次顶掉）或还没结束返回 None。"""
        with self._lock:
            c = self._call
            if c is None or c.at != at or not c.ended:
                return None
            return replace(c, friends=dict(c.friends))

    def unnamed(self, now: float) -> int:
        """最近一帧里没挂名字的点过火的人（陌生人、没归类的远处小人；按外观认的好友不算）。"""
        tracks = [t for t in list(self.last_tracks) if now - t.last <= PEOPLE_STALE]
        return self._count_unnamed([
            t for t in tracks
            if t.cls in ("player", UNLIT) and self._admitted(t) and not self._other_form(t) and not t.data.get("dango")
        ])

    def _count_unnamed(self, players: list[Track]) -> int:
        return sum(not self._unlit(p) and not p.data.get("name") and not p.data.get("maybe") for p in players)

    def _offscreen(self, tag: Track, width: int) -> str | None:
        """名字标签中心在最左 / 最右 edge_band 里：好友在画面外，返回在哪边。"""
        band = self.cfg.edge_band * width
        if band <= 0:
            return None
        cx = tag.box.x + tag.box.w / 2
        return "左边" if cx < band else ("右边" if cx > width - band else None)

    def _calling(self, now: float) -> bool:
        c = self._call
        return c is not None and not c.ended and self._call_span[0] <= now <= self._call_span[1]

    def _call_note(self, name: str, seen: Seen, now: float) -> None:
        if self._calling(now):
            with self._lock:
                self._call.friends[name] = seen  # 同一个名字留最后一次

    def _call_tick(self, players: list[Track], tags: list[Track], tagged: dict[int, Track], assigned: set[int],
                   now: float, width: int, height: int) -> None:
        """窗口里：这一帧挂上名字的人（方位 / 远近同 people()）、没对上人的名字标签记进结果；窗口过了的第一帧收尾。"""
        c = self._call
        if c is None or c.ended or now < self._call_span[0]:
            return
        if now > self._call_span[1]:
            with self._lock:
                c.unnamed = self._count_unnamed(players)
                c.ended = True
            log.debug("呼喊窗口结束：%s；没挂名字 %d 个", "、".join(c.friends) or "没看到名字", c.unnamed)
            return
        for tag in tags:  # 只看到名字、下面没框到人（贴边的在上面已经记过）
            name = tag.data.get("name")
            if name and tag.id not in assigned and self._offscreen(tag, width) is None:
                self._call_note(name, Seen(side_of(tag.box.x + tag.box.w / 2, width), None), now)
        ref = self._ref_height(height)
        for p in players:
            tag = tagged.get(p.id)
            name = tag.data.get("name") if tag is not None else None
            if name and not self._unlit(p):
                where = side_of(p.box.x + p.box.w / 2, width)
                self._call_note(name, Seen(where, distance(p.box.h, ref, self.cfg.near, self.cfg.far)), now)

    def _far_tags(self, frame: np.ndarray, players: list[Track], tags: list[Track], now: float,
                  width: int, height: int, panel_visible: bool) -> list[Track]:
        """三期 §1：远处的小人没挂上名字标签 → 在它头顶裁一块再检测一次（只要 name_tag / social_ring），返回新接上的轨迹。

        裁剪直接交给检测器，它自己 letterbox 放大到 imgsz，等于"放大再检测"。"""
        if self.cfg.far_crops <= 0:
            return []
        limit = self.cfg.far_height * height
        # 远处的标签只在裁剪那一帧出现：下次裁剪要赶在标签轨迹被追踪器删掉（track_buffer）之前，
        # 不然每次都是新轨迹，名字投票、"同一条标签轨迹只记一次"都不起作用
        retry = min(FAR_RETRY, 0.8 * self.cfg.track_buffer)
        calling = self._calling(now)  # 呼喊窗口里远处小人的标签真的画出来了：不退避、多裁几块

        def due(p: Track) -> bool:
            # 连着几次都没找到标签（多半是点过火的陌生人）：放慢到 FAR_BACKOFF 秒一次，别一直占推理
            wait = FAR_BACKOFF if p.data.get("far_miss", 0) >= FAR_MISSES and not calling else retry
            return now - p.data.get("tag_at", float("-inf")) >= retry and now - p.data.get("far_at", float("-inf")) >= wait

        def tag_inside(area: Rect | None) -> bool:  # 原图里已经框到了标签（只是位置没挂上人）：裁了也只会找到它
            return area is not None and any(_inside(_center(t.box), area) for t in tags)

        todo = [
            p for p in players
            if not self._unlit(p) and p.box.h < limit and self._tag_over(p, tags) is None and due(p)
            and not tag_inside(far_region(p.box, width, height))
        ]
        todo.sort(key=lambda p: p.data.get("far_at", float("-inf")))
        found: list[Detection] = []
        for p in todo[: self.cfg.far_crops * (2 if calling else 1)]:
            p.data["far_at"] = now
            area = far_region(p.box, width, height)
            if area is None:
                continue
            self.far_runs += 1
            hits = 0
            for d in self._detect(area.crop(frame)):
                if d.cls not in ("name_tag", "social_ring") or d.score < self.cfg.conf:
                    continue
                box = Rect(d.box.x + area.x, d.box.y + area.y, d.box.w, d.box.h)
                found.append(Detection(d.cls, box, d.score))
                hits += d.cls == "name_tag"
            p.data["far_miss"] = 0 if hits else p.data.get("far_miss", 0) + 1
        found = self._filter(found, width, height, panel_visible, drop_self=False)
        seen = {  # 这一帧原图里已经有的框：二次检测又找到一遍的不要（不能多出一条轨迹、多读一次 OCR）
            cls: [t.box for t in self.tracker.tracks.values() if t.cls == cls and t.last == now]
            for cls in ("name_tag", "social_ring")
        }
        fresh: list[Detection] = []
        for d in found:
            same = seen[d.cls]
            if any(iou(d.box, b) >= FAR_DUP_IOU for b in same) or any(
                f.cls == d.cls and iou(d.box, f.box) >= FAR_DUP_IOU for f in fresh
            ):
                continue
            fresh.append(d)
        return self.tracker.update(fresh, now, prune=False) if fresh else []  # 同一帧第二次：不删、不冲掉 dropped

    # ---- 认装扮（设计 §3.2、§4） ----
    def _appearance_features(self, frame: np.ndarray, players: list[Track], selfs: list[Track], now: float,
                             width: int, height: int, panel_visible: bool) -> dict[int, tuple[np.ndarray, Rect]]:
        """点过火的人和团子：每 every 帧裁一次好样本、算特征，平滑进 data["feat"]、好样本数进 data["samples"]。
        一帧最多算 max_per_frame 个（最久没算的先算）。返回这一帧新算的 {轨迹 id: (这次的特征, 框)}，学进记忆簿时用。"""
        acfg = self.appearance_cfg
        every = max(1, acfg.every)
        due = [t for t in [p for p in players if not self._unlit(p)] + selfs if t.hits % every == 0]
        due.sort(key=lambda t: t.data.get("feat_at", float("-inf")))
        blocked = [roi_rect(self.log_roi, width, height)] if panel_visible else []
        out: dict[int, tuple[np.ndarray, Rect]] = {}
        for t in due:
            if len(out) >= acfg.max_per_frame:
                break
            t.data["feat_at"] = now
            others = [o.box for o in players + selfs if o.id != t.id]
            crop = good_crop(frame, t.box, others, blocked, acfg.min_height, acfg.max_overlap)
            if crop is None:
                continue
            try:
                feat = unit(self.embedder.embed(crop))
            except Exception:
                log.exception("算外观特征出错")
                continue
            old = t.data.get("feat")
            t.data["feat"] = feat if old is None else unit((1 - acfg.ema) * old + acfg.ema * feat)
            t.data["samples"] = t.data.get("samples", 0) + 1
            out[t.id] = (feat, t.box)
        return out

    def _save_samples(self, frame: np.ndarray, players: list[Track], tagged: dict[int, Track],
                      fresh: dict, now: float) -> None:
        """这一帧新算出好样本的人（没点火的、团子不要）：裁图连同元数据交给 saver 攒训练数据，出错只记日志。"""
        for p in players:
            if self._unlit(p) or p.id not in fresh:
                continue
            feat, box = fresh[p.id]
            try:
                tag = tagged.get(p.id)
                name = (tag.data.get("name") if tag is not None else None) or None
                best, score = self.appearance.best_friend(feat)
                row = {
                    "t": now, "track": p.id, "tag": name, "maybe": p.data.get("maybe"), "sid": p.data.get("sid"),
                    "box": [box.x, box.y, box.w, box.h], "best": best, "score": round(score, 4), "place": self.place,
                }
                self.saver.offer(p.id, name or f"t{p.id}", describe_crop(frame, box), now, row)
            except Exception:
                log.exception("存认装扮训练数据出错")

    def _learn(self, kind: str, who: str, sample: tuple[np.ndarray, Rect], frame: np.ndarray, now: float) -> None:
        feat, box = sample
        p = self.appearance.learn(kind, who, feat, now, crop=(box.h, describe_crop(frame, box).copy()))
        self._outfit_triggers(kind, who, p, box, frame.shape[0], now)

    # ---- 装扮描述（设计 §3.3、§4.4）：学到新的好样本时才判断，不是每帧 ----
    def _outfit_triggers(self, kind: str, who: str, p, box: Rect, frame_h: int, now: float) -> None:
        book, acfg = self.appearance, self.appearance_cfg
        key = (kind, who)
        change = acfg.outfit_change  # 关着（默认）：不判换装，好友 / 团子每次上线描述一次
        # 判中途换装不看 want（描述关了、没挂描述器、描述器放弃了，want 都清不掉）：一次换装只算一次靠 mark_changed
        if kind == "me":
            if change and p.desc and p.redescribed < acfg.redescribe_max and book.drifted("me", ""):
                p.redescribed += 1
                log.info("团子换了装扮，重新描述（这次上线第 %d 次）", p.redescribed)
                self._redescribe(key)
            if not p.desc or key in self._describe_want:
                self._ask(kind, who, ME_PRIORITY, p, frame_h, now)
        elif kind == "friend":
            if not p.checked:
                if p.n < acfg.min_samples:
                    return
                p.checked = True
                state = book.card_state(who)
                if not change and state == "changed":
                    state = "same"  # 不判换装：只更新关系卡最近那一套，描述回来后覆盖它
                log.info("%s 的装扮：%s", who, {"new": "第一次记", "same": "和上次一样", "changed": "换了"}[state])
                self._note(who, p.feat, state)
                card = book.card_desc.get(who, "")
                if change and state == "same" and card:
                    book.set_desc("friend", who, card, p.feat)  # 同一套：直接用卡里的描述，不再花额度
                    return
                if self._describing():
                    with self._lock:
                        self._describe_want.add(key)
            elif change and p.redescribed < acfg.redescribe_max and book.drifted("friend", who):
                p.redescribed += 1
                log.info("%s 中途换了装扮，重新描述（这次上线第 %d 次）", who, p.redescribed)
                self._note(who, p.feat, "changed")
                self._redescribe(key)
            if key in self._describe_want:
                self._ask(kind, who, FRIEND_PRIORITY, p, frame_h, now)
        elif not p.desc and distance(box.h, self._ref_height(frame_h), self.cfg.near, self.cfg.far) != "远":
            self._ask(kind, who, STRANGER_PRIORITY, p, frame_h, now)

    def _note(self, name: str, feat: np.ndarray, state: str, desc: str = "") -> None:
        note = OutfitNote(name, [round(float(x), 3) for x in feat], self.appearance.key, state, desc)
        with self._lock:
            self._outfits.append(note)

    def _describing(self) -> bool:
        """挂着描述器、也开着描述：只有这时才记"要描述谁"（want），否则记了也没人清。"""
        return self.wardrobe is not None and self.appearance_cfg.describe

    def _redescribe(self, key: tuple[str, str]) -> None:
        self.appearance.mark_changed(*key)  # 平均特征挪稳到新那套之前不再判换装
        if not self._describing():
            return
        with self._lock:
            self._describe_want.add(key)
            self._describe_asked.pop(key, None)
        self.wardrobe.reset(*key)

    def _ask(self, kind: str, who: str, priority: int, p, frame_h: int, now: float) -> None:
        """交给描述器：框够高的样本才送；retry_after 内问过就不再问（描述器自己也去重、限次）。"""
        if not self._describing():
            return
        key = (kind, who)
        if now - self._describe_asked.get(key, float("-inf")) < self.appearance_cfg.retry_after:
            return
        crop = self.appearance.best_crop(kind, who, min_height=self.appearance_cfg.describe_min_height * frame_h)
        if crop is None:
            return  # 框不够高：等后面的样本
        if self.wardrobe.request(kind, who, priority, crop, p.feat, now):
            log.debug("排队描述装扮：%s %s", kind, who)
            with self._lock:
                self._describe_asked[key] = now

    def on_described(self, kind: str, who: str, desc: str, feat) -> None:
        """描述器线程回调：记进记忆簿；好友的再推一条 described 给身体（写关系卡、发 outfit 事件）。"""
        book = self.appearance
        if book is None:
            return
        feat = unit(feat)
        key = (kind, who)
        p = book.me if kind == "me" else (book.friends if kind == "friend" else book.strangers).get(who)
        if self.appearance_cfg.outfit_change and p is not None and cosine(feat, p.feat) < self.appearance_cfg.changed:
            # 送去描述之后换了装（请求在描述器里排着时）：这是上一套的描述，不记、不推，want 留着，
            # 这一套等下一个样本再请求（不算一次重新描述）
            log.info("装扮描述：%s 已经换了装，丢掉上一套的描述「%s」", who or "团子", desc)
            with self._lock:
                self._describe_asked.pop(key, None)
            return
        book.set_desc(kind, who, desc, feat)
        log.info("装扮描述：%s：%s", who or "团子", desc)
        with self._lock:
            self._describe_want.discard(key)
            self._describe_asked.pop(key, None)
        if kind == "friend":
            self._note(who, feat, "described", desc)

    def pop_outfits(self) -> list[OutfitNote]:
        """取走好友的装扮（new / same / changed / described），身体写进关系卡。"""
        with self._lock:
            out, self._outfits = self._outfits, []
        return out

    def _appearance_identify(self, frame: np.ndarray, players: list[Track], selfs: list[Track],
                             tagged: dict[int, Track], shown: set[str], fresh: dict, now: float,
                             tracks: list[Track]) -> None:
        """学（挂着名字标签的好友、团子）→ 名字标签说了算（摘掉 maybe / 编号）→ 没标签的按外观认好友（maybe）。"""
        book, acfg = self.appearance, self.appearance_cfg
        for me in selfs:
            if me.id in fresh:
                self._learn("me", "", fresh[me.id], frame, now)
        cands: dict[int, np.ndarray] = {}
        for p in players:
            d = p.data
            if self._unlit(p):  # 变成黑影了：没有外观可比
                d.pop("maybe", None)
                d.pop("maybe_by", None)
                d.pop("miss", None)
                continue
            tag = tagged.get(p.id)
            if tag is not None:  # 名字标签永远说了算
                name = tag.data.get("name")
                maybe = d.pop("maybe", None)
                d.pop("maybe_by", None)
                d.pop("miss", None)
                d.pop("sid", None)
                if maybe and name and maybe != name:
                    log.info("轨迹 %d 按外观认成 %s，名字标签是 %s", p.id, maybe, name)
                    self._report(frame, now, f"按外观认成 {maybe}，名字标签是 {name}", tracks)
                if name and p.id in fresh:
                    self._learn("friend", name, fresh[p.id], frame, now)
                continue
            if d.get("tagged") or d.get("name"):
                continue  # 之前挂过标签、这一帧被挡住：身份照旧
            maybe = d.get("maybe")
            if maybe and d.get("maybe_by") == "relink":
                continue  # 按位置接回的：颜色特征不稳，不拿外观否掉位置连续性（标签在别处时 process 里已经摘了）
            if maybe:
                if maybe in shown:  # 他的名字标签此刻清清楚楚挂在别处：不是他
                    log.debug("轨迹 %d 不像 %s 了（标签在别处）", p.id, maybe)
                    d.pop("maybe", None)
                    d.pop("miss", None)
                elif p.id in fresh:
                    d["miss"] = 0 if book.still_like(d["feat"], maybe) else d.get("miss", 0) + 1
                    if d["miss"] >= acfg.recheck:
                        log.debug("轨迹 %d 连着 %d 次不像 %s，摘掉", p.id, d["miss"], maybe)
                        d.pop("maybe", None)
                        d.pop("miss", None)
                continue
            if d.get("samples", 0) >= acfg.min_samples:
                cands[p.id] = d["feat"]
        if not cands:
            return
        taken = shown | {v for p in players for k in ("name", "maybe") if (v := p.data.get(k))}
        byid = {p.id: p for p in players}
        for tid, name in book.assign_friends(cands, taken).items():
            d = byid[tid].data
            d["maybe"], d["miss"] = name, 0
            d.pop("sid", None)
            log.info("轨迹 %d 没看到名字，按外观像 %s", tid, name)

    # ---- 失踪好友接回（spec §4） ----
    def _note_lost(self, now: float) -> None:
        """这一帧被追踪器删掉的好友轨迹记成失踪记录（同名只留最新）；过了 keep 的作废。"""
        for t in self.tracker.dropped:
            # 被撤下的、先祖 / 共享空间的不接回
            if t.cls not in ("player", UNLIT) or self._unlit(t) or not self._admitted(t) or self._other_form(t):
                continue
            d = t.data
            name = d.get("name") if d.get("tagged") else (d.get("maybe") if d.get("maybe_by") == "relink" else None)
            if name:
                self._lost[name] = Lost(name, t.box, t.vx, t.vy, t.vh, t.last, d.get("pan_at", self._pan))
        for name in [n for n, lost in self._lost.items() if now - lost.last > self.cfg.keep]:
            del self._lost[name]

    def _relink(self, players: list[Track], shown: set[str], now: float) -> None:
        """没名字的新轨迹冒在失踪好友的预测位置附近：接成"像他"（maybe，maybe_by = relink）。有歧义不接。
        接上之后 RELINK_MAX_KEEPS 个 keep 还没被名字标签证实就摘掉。"""
        limit = RELINK_MAX_KEEPS * self.cfg.keep
        for p in players:
            d = p.data
            if d.get("maybe_by") == "relink" and now - d.get("relink_at", now) > limit:
                log.info("轨迹 %d 接回成 %s 之后 %.0f 秒都没看到名字，摘掉", p.id, d.get("maybe"), now - d["relink_at"])
                for key in ("maybe", "maybe_by", "relink_at"):
                    d.pop(key, None)
                d["relink_expired"] = True
        if not self._lost:
            return
        held = shown | {v for p in players for k in ("name", "maybe") if (v := p.data.get(k))}
        for name in [n for n in self._lost if n in held]:
            del self._lost[name]  # 他的标签此刻在别处 / 已经在别的轨迹上：这条记录没用了
        cands = [
            p for p in players
            if not self._unlit(p) and not any(p.data.get(k) for k in ("tagged", "name", "maybe"))
        ]
        pairs: list[tuple[float, str, Track]] = []
        for lost in self._lost.values():
            dt = now - lost.last
            if dt > self.cfg.keep:
                continue
            ahead = min(max(dt, 0.0), RELINK_PREDICT)
            cx, cy = lost.box.x + lost.box.w / 2 + lost.vx * ahead, lost.box.y + lost.box.h / 2 + lost.vy * ahead
            px, py = self._pan[0] - lost.pan[0], self._pan[1] - lost.pan[1]
            gate = min(0.6 + 0.5 * dt, 2.5) * lost.box.h
            for p in cands:
                if p.first < lost.last - RELINK_EARLY or not lost.box.h:
                    continue
                if not 0.5 <= p.box.h / lost.box.h <= 2.0:
                    continue
                qx, qy = p.box.x + p.box.w / 2, p.box.y + p.box.h / 2
                dist = min(float(np.hypot(qx - cx - ox, qy - cy - oy)) for ox, oy in ((px, py), (0.0, 0.0)))
                if dist <= gate:
                    pairs.append((dist, lost.name, p))
        per_cand = Counter(p.id for _, _, p in pairs)
        for name in {n for _, n, _ in pairs}:
            mine = sorted((dist, p.id, p) for dist, n, p in pairs if n == name)
            if len(mine) >= 2 and mine[0][0] * RELINK_AMBIGUOUS > mine[1][0]:
                log.debug("%s 断了，附近冒出两个人、分不清是哪个，不接", name)
                continue
            dist, _, p = mine[0]
            if per_cand[p.id] > 1:
                log.debug("轨迹 %d 对得上好几个断掉的好友，不接", p.id)
                continue
            lost = self._lost.pop(name)
            p.data["maybe"], p.data["maybe_by"], p.data["relink_at"] = name, "relink", now
            for key in ("sid", "miss"):
                p.data.pop(key, None)
            log.info("轨迹 %d 像是 %s（断了 %.1f 秒，按位置接回）", p.id, name, now - lost.last)

    def _looks_checked(self, player: Track, now: float) -> bool:
        """能不能判陌生人了：没开认装扮、已经攒够好样本和好友比过外观、或者多等了 STRANGER_GRACE 还攒不够。"""
        return (
            self.appearance is None
            or player.data.get("samples", 0) >= self.appearance_cfg.min_samples
            or now - player.first >= self.cfg.stranger_after + STRANGER_GRACE
        )

    def _appearance_stranger(self, frame: np.ndarray, player: Track, players: list[Track], sample, now: float) -> None:
        """判成陌生人的轨迹：好样本够了就编号（认回以前的、或新编号），之后接着学他的外观。"""
        d = player.data
        sid = d.get("sid")
        if sid is None:
            if d.get("samples", 0) < self.appearance_cfg.min_samples:
                return
            held = {v for o in players if o is not player and (v := o.data.get("sid"))}  # 此刻别的轨迹占着的编号
            sid, back = self.appearance.stranger_id(d["feat"], now, exclude=held)
            d["sid"] = sid
            log.debug("轨迹 %d 是 %s", player.id, sid)
            if back:
                log.info("%s 又回来了", sid)
                with self._lock:
                    self._stranger_backs.append(sid)
        elif sample is not None:
            self._learn("stranger", sid, sample, frame, now)

    def _report(self, frame: np.ndarray, now: float, detail: str, tracks: list[Track]) -> None:
        if self.hardcases is None or self.paused:
            return
        try:
            self.hardcases.report(frame, now, "appearance", detail, tracks)
        except Exception:
            log.exception("收集难例出错")

    def _watch_place(self, frame: np.ndarray, dets: list[Detection], now: float) -> None:
        """三期 §2：每 place_interval 秒、或画面比上次认地图时大变（隔 ≥ PLACE_GAP 秒）认一次。
        偶尔一次认不出先留着原来的；画面大变后认不出、或连着两次认不出就清掉。"""
        small = thumb(frame)
        since = now - self._place_run
        changed = self._place_thumb is not None and difference(self._place_thumb, small) > self.scene_change
        if since < self.place_interval and not (changed and since >= PLACE_GAP):
            return
        self._place_run, self._place_thumb = now, small
        try:
            # 只遮人物 / 名字标签 / 圆圈 / 气泡；长椅、钢琴这些物品是地标，不遮（v4 建的图库也没遮它们）
            match = self.places.recognize(frame, [d.box for d in dets if d.cls not in OBJECT_NAMES])
        except Exception:
            log.exception("认地图出错")
            return
        if match.name is None:
            log.debug("认不出在哪（最像 %s %.2f，第二像 %.2f）", match.best, match.score, match.second)
            self._place_misses += 1
            # 宁可不说也不说错：画面大变后认不出（可能到了图库里没有的地方）、或连着两次认不出，就不再说原来的地名
            if self.place and (changed or self._place_misses >= 2):
                log.info("认不出在哪了，不再说在%s", self.place)
                self.place, self.place_at = "", float("-inf")
            return
        self._place_misses = 0
        if match.name != self.place:
            log.info("看起来到了：%s（相似度 %.2f）", match.name, match.score)
        self.place, self.place_at = match.name, now

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        with self._infer:
            return self.detector.detect(frame)

    def _filter(
        self, dets: list[Detection], width: int, height: int, panel_visible: bool, drop_self: bool = True
    ) -> list[Detection]:
        area = roi_rect(self.env_cfg.roi, width, height)  # 底部输入栏不要
        panel = roi_rect(self.log_roi, width, height) if panel_visible else None  # 面板里的"- 名字"不是名字标签
        mine = self._self_areas(width, height) if drop_self else []
        out = []
        for det in dets:
            c = _center(det.box)
            if not _inside(c, area) or (panel is not None and _inside(c, panel)):
                continue
            if det.cls in ("player", UNLIT) and any(_inside(c, r) for r in mine):
                continue
            out.append(det)
        return out

    def _self_areas(self, width: int, height: int) -> list[Rect]:
        """团子所在的区域：转圈认出的团子框（四周各放宽 25%，镜头跟随会有点晃）+ 配置的 self_roi。"""
        areas = []
        if self.self_box is not None:
            b = self.self_box
            areas.append(Rect(round(b.x - b.w / 4), round(b.y - b.h / 4), round(b.w * 1.5), round(b.h * 1.5)))
        if self.cfg.self_roi:
            areas.append(roi_rect(self.cfg.self_roi, width, height))
        return areas

    def _mark_dango(self, tracks: list[Track], selfs: list[Track], now: float, panel_visible: bool) -> None:
        """给这一帧的人物轨迹打"是团子"的标记（data["dango"] = 这一帧的框），打了的不算别人（不判陌生人、不挂名字标签、不算没挂名字的人）。

        YOLO 在团子身上常常只出 player、self 分数很低或者干脆没有（10-02 晚真机：图鉴 50 张"陌生人"里约 26 张是团子）。
        镜头跟着团子，团子在屏幕上几乎不动，只是聊天面板开 / 关时整个画面横移约 400 px：按面板开关分别记住最近的高分 self 框，
        DANGO_MEMORY 秒内落在那里的人物框就是团子。标记跟着轨迹走（镜头拉近拉远时框慢慢变），直到这一帧有 self 框、它又不在 self 框上，
        或者横着走开 / 框和上一帧对不上（被别人接走了）。"""
        strong = [s for s in selfs if s.strong_last == now]
        if strong:
            self._dango_mem[panel_visible] = (max(strong, key=lambda s: s.score).box, now)
        mem = self._dango_mem.get(panel_visible)
        refs = [s.box for s in selfs] + ([mem[0]] if mem is not None and now - mem[1] <= DANGO_MEMORY else [])
        for t in tracks:
            if t.cls not in ("player", UNLIT):
                continue
            if any(same_body(t.box, b) for b in refs):
                if not t.data.get("dango"):
                    t.data.pop("stranger", None)
                t.data["dango"] = t.box
            elif t.data.get("dango") and (selfs or self._walked_off(t, mem, now) or not same_body(t.box, t.data["dango"])):
                # 框和上一帧对不上 = 轨迹被别人接走了（陌生人一下子挡到团子身前，框胀成两倍多宽）
                log.debug("轨迹 %d 不在团子框上了：不再当成团子", t.id)
                t.data.pop("dango")
            elif t.data.get("dango"):
                t.data["dango"] = t.box  # 没有 self 框时跟着轨迹走（镜头拉近拉远框慢慢变）

    @staticmethod
    def _walked_off(t: Track, mem: tuple[Rect, float] | None, now: float) -> bool:
        """没有 self 框时：轨迹中心横着离开记住的团子位置超过一个团子框宽（团子是镜头支点，拉近拉远时中心几乎不动）。"""
        if mem is None or now - mem[1] > DANGO_MEMORY:
            return False
        box = mem[0]
        return abs(t.box.x + t.box.w / 2 - box.x - box.w / 2) > box.w

    def _read_name(self, frame: np.ndarray, tag: Track, now: float, friends: list[str]) -> None:
        """名字标签轨迹：刚出现就读一次，之后隔 ocr_retry 秒再读，读够 ocr_votes 次（且认出是谁）就不再读。"""
        votes: Counter = tag.data.setdefault("votes", Counter())
        tries = tag.data.get("tries", 0)
        if tries and (now - tag.data.get("ocr_at", float("-inf")) < self.cfg.ocr_retry):
            return
        if tries >= self.cfg.ocr_votes and votes:
            return
        tag.data["tries"], tag.data["ocr_at"] = tries + 1, now
        crop = tag.box.pad(4, frame.shape[1], frame.shape[0]).crop(frame)
        text, score = self._ocr(crop)
        if len(normalize(text)) < 2:
            return
        tag.data["text"] = text
        name = self._match_name(text, friends)
        if name:
            votes[name] += 1
            tag.data["name"] = votes.most_common(1)[0][0]
        elif self.unknown is not None and score >= UNKNOWN_MIN_SCORE and not tag.data.get("unknown_logged"):
            tag.data["unknown_logged"] = True  # 同一条标签轨迹只记一次
            try:
                self.unknown.add(text, crop)
            except Exception:
                log.exception("记没认出的名字出错")

    @staticmethod
    def _match_name(text: str, friends: list[str]) -> str | None:
        return next((n for n in friends if similar(text, n, 0.75)), None)

    def _ocr(self, crop: np.ndarray) -> tuple[str, float]:
        """名字标签的字和置信度（整块识别时取各段里最低的）。"""
        with self._infer:
            read_line = getattr(self.ocr, "read_line", None)
            if read_line is not None:
                line = read_line(crop)
                return (line.text, line.score) if line is not None else ("", 0.0)
            lines = self.ocr.recognize(crop)
            return join_lines(lines), min((l.score for l in lines), default=0.0)

    def _ref_height(self, frame_h: int) -> float:
        """算远近的参照：团子的框高（转圈认出的 → 最近一帧的 self 框 → 按 self_height 估）。"""
        if self.self_box is not None:
            return float(self.self_box.h)
        selfs = [t.box.h for t in list(self.last_tracks) if t.cls == "self"]
        if selfs:
            return float(max(selfs))
        return self.cfg.self_height * frame_h

    # ---- 二期：环绕扫描（设计 §1） ----
    def sweep(self, frames: list[tuple[float, np.ndarray]], spin: SpinConfig) -> SweepResult:
        """转一圈截下的帧 [(按住后第几秒, 图)] → 每个方向有谁。独立的一次观察：不碰追踪器、last_seen、请求。"""
        friends = self.names()
        people: list[list[Detection]] = []
        tags: list[list[Detection]] = []
        width = height = 0
        for _, img in frames:
            height, width = img.shape[:2]
            dets = [d for d in self._filter(self._detect(img), width, height, False, drop_self=False) if d.score >= self.cfg.conf]
            people.append(people_boxes(dets))
            tags.append([d for d in dets if d.cls == "name_tag"])
        found = find_self([[d.box for d in frame] for frame in people], width, spin.self_motion)
        mine = self._self_areas(width, height)  # 帧太少认不出团子时，靠之前认出的团子框 / self_roi 排除
        sightings: list[Sighting] = []
        for fi, (t, img) in enumerate(frames):
            where = lambda box: bearing(t, box.x + box.w / 2, width, spin.seconds_per_turn, spin.hfov)  # noqa: E731
            frame_people = [(bi, d) for bi, d in enumerate(people[fi]) if d.cls != "self"]
            under: dict[int, tuple[int, Detection]] = {}  # 标签序号 → 它下面的人
            for bi, d in frame_people:
                tag = self._tag_over(d, tags[fi])
                if tag is not None:
                    ti = tags[fi].index(tag)
                    if ti not in under or abs(tag.box.y2 - d.box.y) < abs(tag.box.y2 - under[ti][1].box.y):
                        under[ti] = (bi, d)
            for ti, tag in enumerate(tags[fi]):
                text, _ = self._ocr(tag.box.pad(4, width, height).crop(img))
                name = self._match_name(text, friends) if len(normalize(text)) >= 2 else None
                bi, body = under.get(ti, (None, None))
                sightings.append(Sighting(
                    where(tag.box), name or UNKNOWN_WHO, fi, float(body.box.h) if body is not None else None,
                    beside=bi is not None and (fi, bi) in found.static,
                ))
            tagged = {bi for bi, _ in under.values()}
            for bi, d in frame_people:
                if bi in tagged or (fi, bi) in found.static or any(_inside(_center(d.box), r) for r in mine):
                    continue
                if d.cls == UNLIT:
                    sightings.append(Sighting(where(d.box), UNLIT_WHO, fi, float(d.box.h)))
                elif d.box.h >= self.cfg.stranger_min_height * height:
                    sightings.append(Sighting(where(d.box), STRANGER_WHO, fi, float(d.box.h)))
        if found.box is not None:
            self.self_box = found.box
            log.info("转圈认出了团子：%s", found.box)
        ref = self._ref_height(height or self._frame_h)
        entries = merge(sightings, spin.merge_deg, ref, self.cfg.near, self.cfg.far)
        return SweepResult(entries, found.box, len(frames), frames[-1][0] if frames else 0.0)

    @staticmethod
    def _tag_above(ring: Track, tags: list[Track]) -> Track | None:
        """圆圈正上方的名字标签：圆圈中心在标签上沿往下约 2.23 倍标签高度、水平居中。"""
        rx, ry = _center(ring.box)
        best, best_err = None, float("inf")
        for tag in tags:
            tx = tag.box.x + tag.box.w / 2
            dy = (ry - tag.box.y) / tag.box.h
            if not 1.2 <= dy <= 3.5 or abs(rx - tx) > max(tag.box.w / 2, ring.box.w):
                continue
            err = abs(dy - ICON_OFFSET) + abs(rx - tx) / tag.box.h
            if err < best_err:
                best, best_err = tag, err
        return best

    @staticmethod
    def _tag_err(player, tag) -> float | None:
        """名字标签在不在这个人头顶：水平上差不多对齐，标签在人物上沿附近（上方一个半身高到身体上半截之间）。
        在的话返回对不齐的程度（越小越正），不在返回 None。"""
        p = player.box
        tx = tag.box.x + tag.box.w / 2
        if not p.x - p.w * 0.25 <= tx <= p.x2 + p.w * 0.25:
            return None
        if not p.y - 1.5 * p.h <= tag.box.y2 <= p.y + 0.5 * p.h:
            return None
        return abs(tx - (p.x + p.w / 2)) + abs(tag.box.y2 - p.y)

    @classmethod
    def _tag_over(cls, player, tags: list) -> object | None:
        """这个人头顶对得最正的名字标签（不管有没有分给别人；远处重裁、转圈扫描用）。"""
        scored = [(err, i) for i, t in enumerate(tags) if (err := cls._tag_err(player, t)) is not None]
        return tags[min(scored)[1]] if scored else None

    def _assign_tags(self, players: list[Track], tags: list[Track]) -> dict[int, Track]:
        """人物轨迹 id → 头顶的名字标签。一个标签只给对得最正的那个人（实测团子和好友挤在一起时两个人都挂上了好友的名字）。"""
        pairs = sorted(
            (err, i, j) for i, p in enumerate(players) for j, t in enumerate(tags)
            if (err := self._tag_err(p, t)) is not None
        )
        out: dict[int, Track] = {}
        used: set[int] = set()
        for _, i, j in pairs:
            if players[i].id not in out and j not in used:
                out[players[i].id] = tags[j]
                used.add(j)
        return out

    @staticmethod
    def _person_below(bubble: Rect, people: list):
        """气泡正下方的人：气泡水平中心落在人物框（左右各放宽 25%）内，气泡下沿在人物上沿往上 2.5 个身高到身体上部之间。"""
        bx = bubble.x + bubble.w / 2
        best, best_gap = None, float("inf")
        for person in people:
            p = person.box
            if not p.x - p.w * 0.25 <= bx <= p.x2 + p.w * 0.25:
                continue
            if not p.y - 2.5 * p.h <= bubble.y2 <= p.y + 0.3 * p.h:
                continue
            gap = abs(p.y - bubble.y2)
            if gap < best_gap:
                best, best_gap = person, gap
        return best

    def _watch_typing(self, bubbles: list[Track], players: list[Track], selfs: list[Track], now: float,
                      width: int, height: int) -> None:
        """头顶的"正在输入"气泡挂到下面的人身上（团子自己的不算），记下来给 speaker_hint 用。"""
        me = [s.box for s in selfs] + ([self.self_box] if self.self_box is not None else [])
        for b in bubbles:
            if self._person_below(b.box, [Track(0, "self", r, 1.0, now, now) for r in me]) is not None:
                continue  # 团子自己在打字（输入框开着时头顶也有气泡）
            owner = self._person_below(b.box, players)
            if owner is None:
                continue
            if now - owner.data.get("typing_at", float("-inf")) > self.cfg.bubble_gap:
                owner.data["bubble_start"] = now  # 新的一句
            owner.data["typing_at"] = now
            friend = bool(owner.data.get("name") or owner.data.get("tagged") or owner.data.get("maybe"))
            self._typing.append((now, owner.id, friend, self._unlit(owner), owner.box.x + owner.box.w / 2, owner.box.h,
                                 width, height))
        while self._typing and now - self._typing[0][0] > self.cfg.typing_window:
            self._typing.popleft()

    def _watch_approach(self, players: list[Track], now: float, width: int) -> None:
        for p in players:
            hist: deque = p.data.setdefault("hist", deque())
            hist.append((now, float(p.box.h), p.box.x + p.box.w / 2))
            while hist and now - hist[0][0] > self.cfg.approach_window:
                hist.popleft()
            maybe = p.data.get("maybe") if p.data.get("maybe_by") != "relink" else None  # 按位置接回的不报走近（可能接错）
            who = (
                p.data.get("name")
                or (maybe if maybe and maybe in self.nearby(now) else None)  # 按外观认的：他还在身边才算（同 last_seen 的规矩）
                or (STRANGER if p.data.get("stranger") else None)
            )
            if who is None or (who == STRANGER and not self.cfg.approach_strangers):
                continue
            if now - self._approach_at.get(who, float("-inf")) < self.cfg.approach_cooldown:
                continue
            if approaching(list(hist), width, self.cfg.approach_grow):
                self._approach_at[who] = now
                log.info("%s 朝团子走过来了", who)
                with self._lock:
                    self._approaches.append(who)
                    self._approach_log.append((who, p.id, now))
                    while self._approach_log and now - self._approach_log[0][2] > 30.0:
                        self._approach_log.popleft()

    def _watch_motion(self, players: list[Track], now: float) -> None:
        """每条人物轨迹（含黑影）记 (时间, 框高, 补偿过画面平移的中心 x)，算运动方向（防抖后放进 data["motion"]）。"""
        cfg = self.cfg
        for p in players:
            hist: deque = p.data.setdefault("motion_hist", deque())
            hist.append((now, float(p.box.h), p.box.x + p.box.w / 2 - p.drift[0]))  # 减掉被画面平移带走的（跟着镜头动的人不减）
            while hist and now - hist[0][0] > cfg.motion_window:
                hist.popleft()
            settle_motion(p.data, motion_of(hist, now, cfg), now, cfg.motion_hold)

    def _watch_gestures(self, frame: np.ndarray, players: list[Track], now: float, width: int, height: int) -> None:
        """三期 §3：认出名字、近 / 中、在画面中间的好友，攒够一段（frames 张）后每 interval 秒判一次动作。"""
        cfg = self.gesture_cfg
        ref = self._ref_height(height)
        for p in players:
            name = p.data.get("name")
            if self._unlit(p) or not eligible(name, p.box, width, ref, self.cfg.near, self.cfg.far):
                p.data.pop("clip", None)  # 走开 / 走远了：之前攒的不能和回来后的拼在一起
                continue
            buf: ClipBuffer = p.data.setdefault("clip", ClipBuffer(cfg.frames, cfg.fps))
            buf.push(now, person_crop(frame, p.box, cfg.size))
            if not buf.ready() or now - p.data.get("gesture_at", float("-inf")) < cfg.interval:
                continue
            p.data["gesture_at"] = now
            try:
                with self._infer:
                    label, prob = self.gestures.classify(buf.clip())
            except Exception:
                log.exception("认动作出错")
                continue
            if label == "none" or prob < cfg.min_prob:
                continue
            key = (name, label)
            if now - self._gesture_at.get(key, float("-inf")) < cfg.cooldown:
                continue
            self._gesture_at[key] = now
            log.info("%s 对团子做了动作：%s（%.2f）", name, label, prob)
            with self._lock:
                self._gestures.append(key)

    def talkers(self, now: float) -> list[Talker]:
        """1 秒内头顶冒着气泡的人（不含团子自己），带位置和这次气泡开始的时间。"""
        out = []
        for t in list(self.last_tracks):
            at = t.data.get("typing_at")
            if at is None or now - at > 1.0:
                continue
            name = t.data.get("name") or t.data.get("maybe")
            friend = bool(name or t.data.get("tagged"))
            out.append(Talker(t.id, name, friend, t.box.x + t.box.w / 2, at, t.data.get("bubble_start", at)))
        return out

    def recent_approaches(self, now: float, within: float = 5.0) -> list[tuple[str, float, float]]:
        """within 秒内朝团子走过来的人：(谁, 现在的框中心 x, 时间)；谁同 pop_approaches（好友名 / STRANGER）。不取走。
        x 按那条轨迹现在的位置取（人会接着走、镜头会转）；轨迹 1 秒没更新（走没了）就不报。"""
        tracks = {t.id: t for t in list(self.last_tracks) if now - t.last <= 1.0}
        with self._lock:
            log_ = [a for a in self._approach_log if now - a[2] <= within]
        return [(who, tracks[tid].box.x + tracks[tid].box.w / 2, t) for who, tid, t in log_ if tid in tracks]

    def pop_gestures(self) -> list[tuple[str, str]]:
        """取走"谁对团子做了什么动作"，身体变成 gesture 事件。"""
        with self._lock:
            out = list(self._gestures)
            self._gestures.clear()
        return out

    def pop_approaches(self) -> list[str]:
        """取走"朝团子走过来"的人（好友名 / 陌生人），身体变成 approach 事件。"""
        with self._lock:
            out, self._approaches = self._approaches, []
        return out

    def pop_stranger_backs(self) -> list[str]:
        """取走"走开又回来了"的陌生人编号（"陌生人A"），身体变成 stranger_back 事件。"""
        with self._lock:
            out, self._stranger_backs = self._stranger_backs, []
        return out

    def my_look(self) -> str:
        """团子自己现在的装扮描述（没有就空）。"""
        return self.appearance.look("me", "") if self.appearance is not None else ""

    def looks(self, names: list[str]) -> dict[str, str]:
        """好友现在的装扮描述：{名字: 描述}，没描述的不给。"""
        if self.appearance is None:
            return {}
        return {n: desc for n in names if (desc := self.appearance.look("friend", n))}

    def nearest(self, now: float) -> tuple[str, str] | None:
        """最近一帧里离团子最近（框最高）、认得出是谁的人：(好友名 / "陌生人", 近 / 中 / 远)。"""
        known = []
        for t in list(self.last_tracks):
            # 每个键只读一次：感知线程随时可能 pop 掉 maybe / sid（别的线程在读）
            who = t.data.get("name") or t.data.get("maybe") or (STRANGER if t.data.get("stranger") else None)
            if t.cls in ("player", UNLIT) and self._admitted(t) and not self._other_form(t) and who:
                known.append((t.box.h, who))
        if not known:
            return None
        h, who = max(known)
        return who, distance(h, self._ref_height(self._frame_h), self.cfg.near, self.cfg.far)

    def people(self, now: float) -> list[Person]:
        """最近一帧里认得出是谁的人（好友 / 陌生人 / 没点火的黑影；第二层开着时还有先祖 / 共享空间的人），左到右、同一边的近的在前。

        暂停中、或者这一帧已经过时（被挡住 / 没跑检测）返回空：别让技能盯着暂停前的人。
        """
        if self.paused:
            return []
        ref = self._ref_height(self._frame_h)
        out = []
        others = {t.id for t in list(self._others)}  # process 认定的先祖 / 共享空间的人（和团子框重叠的不算）
        for t in list(self.last_tracks):
            if now - t.last > PEOPLE_STALE or t.cls not in ("player", UNLIT) or not self._admitted(t) or t.data.get("dango"):
                continue
            d = t.data
            # 每个键只读一次（不 get 完再 []）：身体线程在读的同时，感知线程可能刚好 pop 掉 maybe / sid
            name, maybe, stranger = d.get("name"), d.get("maybe"), d.get("stranger")
            motion = d.get("motion") if self.cfg.motion else None
            sure, sid, look = True, None, ""
            form, form_p = (d.get("form") or (None, 0.0)) if self._attrs_on() else (None, 0.0)
            dark = self._unlit(t)
            if form in OTHERS and t.id in others:  # 先祖 / 共享空间的人：不挂名字，运动方向没算过（不在 players 里）
                kind, name, motion = form, None, None
            elif dark and stranger:  # 和团子框重叠的黑影不算（process 里没给它记 stranger）
                kind = "unlit"
            elif not dark and name:
                kind = "friend"
            elif not dark and maybe:  # 没看到名字、按外观认的好友
                kind, name, sure = "friend", maybe, False
            elif not dark and stranger:
                kind, sid = "stranger", d.get("sid")
                if sid and self.appearance is not None:
                    look = self.appearance.look("stranger", sid)
            else:
                continue
            side = side_of(t.box.x + t.box.w / 2, self._frame_w)
            out.append(Person(t.id, kind, name if kind == "friend" else None, t.box, side,
                              distance(t.box.h, ref, self.cfg.near, self.cfg.far), sure=sure, sid=sid, look=look,
                              motion=motion, form=form, form_p=form_p))
        order = {"左边": 0, "前面": 1, "右边": 2}
        return sorted(out, key=lambda p: (order[p.side], -p.box.h))

    def typing_seen(self, now: float, within: float = 1.0, strangers: bool = False) -> bool:
        """最近 within 秒里有好友（strangers 时陌生人也算）头顶冒着"正在输入"气泡（团子自己的不算）：聊天面板该打开看了。"""
        return any(now - r[0] <= within and (r[2] or strangers) for r in list(self._typing))

    def speaker_hint(self, now: float) -> str | None:
        """陌生人的消息是谁说的：最近 typing_window 秒内头顶冒过气泡、又不是好友的人恰好一个 → 说出他在画面哪儿。"""
        recent = [r for r in list(self._typing) if now - r[0] <= self.cfg.typing_window and not r[2]]
        if len({r[1] for r in recent}) != 1:
            return None
        _, _, _, dark, cx, h, width, height = recent[-1]
        side = side_of(cx, width)
        far = distance(h, self._ref_height(height), self.cfg.near, self.cfg.far) == "远"
        who = "没点火的陌生人" if dark else "陌生人"
        return f"（说话的可能是{side}{'远处' if far else '近处'}那个{who}）"

    # ---- 点亮陌生人（spec 2026-10-01-light-unlit-stranger §3） ----
    def _disk_like(self, frame: np.ndarray, ring: Track) -> bool:
        """头顶没名字的圆圈里是火焰：没有白圈 = 深色火焰圆盘（团子能去点亮他，点了圆盘会跟着他走），不是他要给团子点火。
        只看白圈、不看亮度（半透明圆盘透出后面亮的东西时外环并不暗，见 candle.RING_WHITE），也不管点亮陌生人开没开。"""
        cx, cy = self._ring_center(ring)
        return not white_ring(frame, cx, cy, ring.box.w / 2)

    def _self_now(self, now: float) -> Rect | None:
        """团子现在在哪：最近这一帧有新检测的 self 轨迹里分数最高的。找不到时沿用最近的框最多 SELF_HOLD 秒。
        黑影贴着团子时 YOLO 常认不出 self；镜头跟着团子，屏幕位置几乎不变，短时间沿用最近的框是安全的。"""
        # 只用这一帧刚匹配上的 self 轨迹；别的时候都走下面的沿用
        fresh_selfs = [t for t in self.tracker.tracks.values() if t.cls == "self" and now - t.last < 1e-6]
        if fresh_selfs:
            best = max(fresh_selfs, key=lambda t: t.score)
            self._me_last = (best.box, now)
            return best.box
        # 没有这一帧的新检测，沿用最近的框（黑影贴着认不出时）
        if self._me_last and now - self._me_last[1] <= SELF_HOLD:
            return self._me_last[0]
        return None

    def _flame_area(self, me: Rect, width: int, height: int) -> Rect:
        cfg, cx = self.light_cfg, me.x + me.w / 2
        x1, x2 = round(cx - cfg.light_area_x * me.h), round(cx + cfg.light_area_x * me.h)
        y1 = round(me.y - cfg.light_area_up * me.h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, me.y2)
        return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _flame_excluded(self, pos: tuple[int, int], tags: list[Track], bonfires: list[Track], width: int, height: int,
                        panel_visible: bool) -> str | None:
        """火焰落在这些地方当没看到：认出了名字的标签下面（好友举蜡烛给团子点火的圆圈）、篝火上方（"点燃"图标也是火焰圆圈）、开着的聊天面板。"""
        for t in tags:
            b = t.box
            if t.data.get("name") and _inside(pos, Rect(round(b.x - 0.5 * b.w), b.y, 2 * b.w, round(4.5 * b.h))):
                return f"在 {t.data['name']} 的名字标签下面"
        for t in bonfires:  # 图标浮在柴堆上方：左右各放宽 0.5 倍框宽、往上放宽 1 倍框高、下到框底
            b = t.box
            if _inside(pos, Rect(round(b.x - 0.5 * b.w), b.y - b.h, 2 * b.w, 2 * b.h)):
                return "在篝火上"
        if panel_visible and _inside(pos, roi_rect(self.log_roi, width, height)):
            return "聊天面板挡着"
        return None

    def _person_at(self, pos: tuple[int, int], r: float) -> Rect | None:
        """火焰往下那一块（横向 ±2r、纵向 −2r ~ +8r：火焰在胸口或头顶）里重叠最多的人物框（含低分框）。"""
        area = Rect(round(pos[0] - 2 * r), round(pos[1] - 2 * r), round(4 * r), round(10 * r))
        best, most = None, 0
        for box, _ in self._people_boxes:
            w = min(area.x2, box.x2) - max(area.x, box.x)
            h = min(area.y2, box.y2) - max(area.y, box.y)
            if w > 0 and h > 0 and w * h > most:
                best, most = box, w * h
        return best

    def _watch_flames(self, frame: np.ndarray, tags: list[Track], bonfires: list[Track], now: float, width: int, height: int,
                      panel_visible: bool) -> None:
        """团子身边连续 light_after 秒冒着火焰（够近的黑影身上才有）→ 出 light 请求，不管 YOLO 认没认出这个人。"""
        cfg = self.light_cfg
        if cfg is None or self.flame is None or now - self._flame_check < DISK_EVERY - 1e-6:  # 1e-6：0.6 - 0.3 < 0.3
            return
        self._flame_check = now
        why, flame = "", None
        me = self._self_now(now)
        if me is None:
            why = "没有团子框"
            self._scan = None
        else:
            area = self._flame_area(me, width, height)
            flame = find_flame(frame, area, self.flame, cfg.disk_min_score)
            if flame is not None:
                skip = self._flame_excluded((flame.x, flame.y), tags, bonfires, width, height, panel_visible)
                if skip:
                    why, flame = f"火焰 {flame.score:.2f} {skip}", None
            self._scan = {"area": area, "me": me, "flame": flame}
        with self._lock:
            clue = self._flame
            if flame is not None:
                pos = (flame.x, flame.y)
                person = self._person_at(pos, flame.r)
                blk = black(frame, person, cfg.lit_v, exclude=me) if person is not None else None
                same = (clue is not None and now - clue["last"] <= DISK_GAP
                        and np.hypot(pos[0] - clue["pos"][0], pos[1] - clue["pos"][1]) <= cfg.light_jump * me.h)
                if not same:
                    self._flame_seq += 1
                    clue = self._flame = {"id": self._flame_seq, "first": now, "best": 0.0, "announced": False}
                clue.update(last=now, pos=pos, r=flame.r, score=flame.score, black=blk, person_box=person, best=max(clue["best"], flame.score))
            elif clue is not None and now - clue["last"] > DISK_GAP:
                clue = self._flame = None
            lighting, cooling = self._lighting, now < self._cooldown_until
            if lighting is not None:  # 举着蜡烛：先更新火焰位置，再在那里找人、看他有多黑
                # 同一条线索，或这次找到的火焰离原来的位置不远（火焰位置跳了成了新线索，其实火还在）
                near = (flame is not None
                        and np.hypot(flame.x - lighting["pos"][0], flame.y - lighting["pos"][1]) <= cfg.light_jump * me.h)
                if clue is not None and (clue["id"] == lighting["id"] or near):
                    lighting["flame_last"], lighting["pos"], lighting["r"] = clue["last"], clue["pos"], clue["r"]
                if me is not None:  # 团子框丢了就没真的找过，不更新（lit() 靠 scan_at 判扫描新不新）
                    lighting["scan_at"] = now
                    person = self._person_at(lighting["pos"], lighting["r"])
                    if person is None:
                        lighting["bright"] = 0
                    else:
                        blk = black(frame, person, cfg.lit_v, exclude=me)
                        lighting["person"], lighting["person_at"] = (person, blk), now
                        black0 = lighting["black0"]
                        if blk is not None and blk >= cfg.lit_black:
                            lighting["dark_box"] = person  # 他黑着：记下这时的框，之后变亮的得是他
                        dark = lighting["dark_box"]
                        # blk 为 None = 被团子挡住大半量不准：人还在，但不算变亮
                        bright = (blk is not None and blk < cfg.lit_black and (black0 is None or black0 - blk >= cfg.lit_drop)
                                  and dark is not None and iou(person, dark) >= LIT_SAME_IOU)
                        lighting["bright"] = lighting["bright"] + 1 if bright else 0
            ready = (clue is not None and lighting is None and not cooling and now - clue["last"] <= DISK_GAP
                     and now - clue["first"] >= cfg.light_after and clue["best"] >= cfg.disk_sure)
            first = ready and not clue["announced"]
            if ready:
                clue["announced"] = True
                self.requests[LIGHT_KEY] = Request(STRANGER, LIGHT, clue["pos"], now, track=clue["id"])
            else:
                self.requests.pop(LIGHT_KEY, None)
        if now - self._flame_log >= 1.0:
            self._flame_log = now
            if clue is not None:
                log.debug("火焰线索 %d：(%d, %d) %.2f，已 %.1f 秒，best %.2f%s%s", clue["id"], *clue["pos"], clue["score"],
                          now - clue["first"], clue["best"], "（冷却中）" if cooling else "", "（正在点亮）" if lighting else "")
            elif why:
                log.debug("没找火焰 / 不算：%s", why)
        if lighting is not None and self._diag is not None and self._scan is not None and now >= self._diag["next"]:
            self._save_light(frame, now, "raised")
        if first:
            log.info("身边有没点火的陌生人：火焰出现了 %.0f 秒（线索 %d）", now - clue["first"], clue["id"])
            self._on_request(frame, now)

    def _on_request(self, frame: np.ndarray, now: float) -> None:
        """第一次出请求：新建这次的存图目录、存一张（dry-run 不按键也存：看得出该不该举）。"""
        if self.light_dir is None or self._diag_runs >= DIAG_RUNS:
            self._diag = None
            return
        self._diag_runs += 1
        stamp = time.strftime("%H%M%S")
        path, k = self.light_dir / stamp, 1
        while path.exists():
            k += 1
            path = self.light_dir / f"{stamp}-{k}"
        self._diag = {"dir": path, "count": 0, "next": float("-inf"), "requested": now}
        self._save_light(frame, now, "request")

    def _save_light(self, frame: np.ndarray, now: float, label: str) -> None:
        """画上搜索范围（灰）、团子框（白）、火焰（青圈 + 分数）、判点亮找到的人（绿 = 不黑 / 紫 = 黑 + black 值）。"""
        diag = self._diag
        if diag is None or diag["count"] >= DIAG_MAX:
            return
        diag["count"] += 1
        diag["next"] = now + DIAG_EVERY  # 先前进：存图失败也不会每次扫描都重试
        try:
            import cv2

            from ..imageio import imwrite

            img = frame.copy()
            scan = self._scan or {}
            if scan.get("area") is not None:
                a = scan["area"]
                cv2.rectangle(img, (a.x, a.y), (a.x2, a.y2), (160, 160, 160), 2)
            if scan.get("me") is not None:
                m = scan["me"]
                cv2.rectangle(img, (m.x, m.y), (m.x2, m.y2), (255, 255, 255), 2)
            f = scan.get("flame")
            if f is not None:
                cv2.circle(img, (f.x, f.y), round(f.r * 1.5), (255, 255, 0), 2)
                cv2.putText(img, f"{f.score:.2f}", (f.x + round(f.r * 1.5), f.y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            with self._lock:
                person = self._lighting["person"] if self._lighting is not None else None
                raised = self._lighting["raised"] if self._lighting is not None else None
            if person is not None:
                box, blk = person
                color = (200, 60, 200) if blk is None or blk >= self.light_cfg.lit_black else (60, 220, 60)
                cv2.rectangle(img, (box.x, box.y), (box.x2, box.y2), color, 2)
                cv2.putText(img, "?" if blk is None else f"{blk:.2f}", (box.x, box.y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            offset = now - (raised if raised is not None else now)
            diag["dir"].mkdir(parents=True, exist_ok=True)
            imwrite(diag["dir"] / f"{label}-{offset:+05.1f}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            log.debug("存点亮图出错", exc_info=True)

    def mark_tried(self, clue_id: int) -> None:
        """身体举起蜡烛了：开始"点亮中"（期间不出新请求），记下这时火焰在哪、他有多黑。"""
        now, cfg = self.clock(), self.light_cfg
        with self._lock:
            clue = self._flame if self._flame is not None and self._flame["id"] == clue_id else None
            self._lighting = {
                "id": clue_id, "raised": now, "requested": now,
                "pos": clue["pos"] if clue else (0, 0), "r": clue["r"] if clue else 20.0,
                "black0": clue["black"] if clue else None,
                "dark_box": clue.get("person_box") if clue and cfg is not None and clue["black"] is not None and clue["black"] >= cfg.lit_black else None,
                "flame_last": clue["last"] if clue else float("-inf"),
                "person": None, "person_at": float("-inf"),
                "scan_at": float("-inf"), "bright": 0,  # 最近一次真的找过火焰的时间；连续几次扫描看到他变亮
            }
            if self._diag is not None:
                self._diag["next"] = float("-inf")  # 举起后第一帧就存
            self.requests.pop(LIGHT_KEY, None)

    def lit(self, clue_id: int, since: float) -> bool | None:
        """举蜡烛（since 时刻）之后：True 他亮了；False 还在等（火焰还在 / 还黑着 / 拿不准）；None 火焰没了、那里也没人（走了）。

        一律按最近一次真的找过火焰的时间（scan_at）判，不看 process 写的帧时间（两者之间身体线程调进来会错位）。
        拿不准一律 False：感知暂停中、不是现在点亮中的线索、scan_at 比 FRAME_STALE + DISK_EVERY 还旧（感知没在跑 / 团子框丢了）、
        举起不满 lit_min 秒、火焰还在（断开不满 DISK_GAP）、最近一次扫描没找到人（先等等）。
        火焰消失超过 DISK_GAP 后：DISK_GAP 内都没找到人 → None；否则连续 LIT_SCANS 次扫描都看到他
        （含低分框）black() 比举蜡烛时降了 lit_drop 以上且低于 lit_black（举蜡烛时下面没人就只看 lit_black）→ True，宁晚勿早。
        变亮的人物框还要和点亮中最近一次看到他黑的框重叠 LIT_SAME_IOU 以上（同一个人）：从没看到过他黑的样子永远不判 True
        （等身体超时放下；别的亮人站到火焰原位置不能算）。"""
        cfg = self.light_cfg
        if cfg is None or self.paused:
            return False
        with self._lock:
            L = self._lighting
            if L is None or L["id"] != clue_id:
                return False
            ref, flame_last, person_at, bright = L["scan_at"], L["flame_last"], L["person_at"], L["bright"]
        if self.clock() - ref > FRAME_STALE + DISK_EVERY or ref - since < cfg.lit_min:
            return False
        if ref - flame_last <= DISK_GAP:
            return False
        if ref - person_at > DISK_GAP:
            return None
        if person_at < ref:  # 最近一次找火焰没找到人：先等等
            return False
        return bright >= LIT_SCANS

    def light_done(self, result: str) -> None:
        """身体这次点亮结束了（lit / gone / timeout / interrupted / dry-run / exit / failed）：结束"点亮中"；
        没点亮就冷却 light_cooldown 秒（认不出是谁，只能按时间）。重复调用无害。"""
        cfg = self.light_cfg
        with self._lock:
            L, self._lighting = self._lighting, None
            self.requests.pop(LIGHT_KEY, None)
            if L is None:
                return
            self._flame = None  # 这条线索用过了：之后要重新连续看满 light_after 秒
            if result != "lit" and cfg is not None:
                self._cooldown_until = self.clock() + cfg.light_cooldown
        log.info("点亮陌生人结束：%s（线索 %d）", result, L["id"])
        self._light_finished(L, result)

    def _light_finished(self, lighting: dict, result: str) -> None:
        """写这次的 summary.json：线索、出请求 / 举起时间、结果、举起时和最后的 black()、火焰最后看到的时间（都相对举起，秒）。"""
        diag, self._diag = self._diag, None
        if diag is None:
            return
        import json

        raised = lighting["raised"]
        rel = lambda t: None if t == float("-inf") else round(t - raised, 2)  # noqa: E731
        summary = {
            "clue": lighting["id"], "result": result, "images": diag["count"],
            "requested": rel(diag["requested"]), "raised": 0.0, "done": rel(self.clock()),
            "black_raised": lighting["black0"],
            "black_end": round(lighting["person"][1], 3) if lighting["person"] is not None and lighting["person"][1] is not None else None,
            "flame_last": rel(lighting["flame_last"]),
        }
        try:
            diag["dir"].mkdir(parents=True, exist_ok=True)
            (diag["dir"] / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            log.debug("写点亮 summary 出错", exc_info=True)

    @staticmethod
    def _ring_center(ring: Track) -> tuple[int, int]:
        cx, cy = _center(ring.box)
        return int(cx), int(cy)

    def _classify(self, frame: np.ndarray, ring: Track) -> str | None:
        if self.icons is None:
            return None
        cx, cy = self._ring_center(ring)
        r = round(56 * frame.shape[0] / 1080)
        region = frame[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
        return self.icons.classify(region)[0]

    # ---- 可视化（vision/viewer.py） ----
    def overlay(self, now: float) -> list[dict]:
        """最近一帧认出了什么：每个框一条 {x, y, w, h, kind, label}（整张截图坐标）。"""
        out = []
        others = {t.id for t in list(self._others)}  # 同 people()：只有 process 认定的先祖 / 共享空间的人才画成那样
        for t in list(self.last_tracks):
            d = t.data
            person = t.cls in ("player", UNLIT)
            attrs_on = person and self._attrs_on()
            mean = d.get("form_mean") if attrs_on else None
            if person and not self._admitted(t):
                if not d.get("rejected"):
                    continue  # 还没复核完的低分框先不画
                p = (mean or {}).get("not_person", 0.0)
                b = t.box  # 被第二层撤下的框：灰虚线画出来，看它为什么被撤
                entry = {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "kind": "rejected", "label": f"不是人 {p:.1f}",
                         "score": round(t.score, 2)}
                self._form_entry(entry, t, mean)
                out.append(entry)
                continue
            dark = person and self._unlit(t)
            other = self._other_form(t) if person and t.id in others else None
            # 每个键只读一次（同 people()）；who = 装扮描述按谁查
            name, maybe, sid, who = d.get("name"), d.get("maybe"), d.get("sid"), None
            if other:  # 先祖 / 共享空间的人（画法见 spec §8，后面再加）
                kind, label = other, WHO[other]
            elif person and not dark and maybe and not name:
                kind, label, who = "maybe", f"像{maybe}?", maybe
            elif person and not dark:
                kind = "stranger" if d.get("stranger") else ("friend" if name else "player")
                label = name or ((sid or "陌生人") if kind == "stranger" else "")
                who = name if kind == "friend" else sid
            elif dark:
                kind, label = "unlit", "陌生人（没点火）"
            elif t.cls == "self":
                kind, label = "self", "团子"
            elif t.cls == "typing":
                kind, label = "typing", "正在输入"
            elif t.cls == "name_tag":
                kind = "name" if d.get("name") else "tag"
                label = d.get("name") or (f"?{d['text']}" if d.get("text") else "?")
            elif t.cls in OBJECT_NAMES:
                kind, label = t.cls, OBJECT_NAMES[t.cls]
            elif t.cls == "social_ring":
                ring = d.get("kind")
                kind = "request" if is_request(ring) else "ring"
                label = KIND_NAMES.get(ring, "✦" if ring == IDLE else "?")
            else:
                kind, label = t.cls, t.cls
            b = t.box
            entry = {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "kind": kind, "label": label, "score": round(t.score, 2)}
            if self.cfg.motion and t.cls in ("player", UNLIT) and (motion := d.get("motion")):
                entry["motion"] = motion
            if self.appearance is not None and (desc := self._desc(kind, who)):
                entry["desc"] = desc
            if attrs_on:
                if not d.get("strong"):
                    # 低分框，靠第二层复核才放行的；label 保持干净（scene_note、网页点人取名字都用它），"复核·"由网页画字时加
                    entry["reviewed"] = True
                self._form_entry(entry, t, mean)
            out.append(entry)
        with self._lock:
            clue = self._flame
            if clue is not None and now - clue["last"] <= DISK_GAP:
                (x, y), r = clue["pos"], clue["r"]
                out.append({"x": round(x - r), "y": round(y - r), "w": round(2 * r), "h": round(2 * r),
                            "kind": "ring", "label": f"火焰 {clue['score']:.2f}", "score": round(clue["score"], 2)})
        return out

    def _form_entry(self, entry: dict, t: Track, mean: dict | None) -> None:
        """第二层开着时给人物条目附上外形概率 form 和黑影分 u（悬停显示）；没有就不写。"""
        if mean:
            entry["form"] = {k: round(v, 2) for k, v in mean.items()}
        u = unlit_score(t, self.attrs.cfg.yolo_w)
        if u is not None:
            entry["u"] = round(u, 2)

    def _desc(self, kind: str, who: str | None) -> str:
        """框对应的人的装扮描述（没有就空）；who = overlay 已经读出来的好友名 / 陌生人编号（不再回去查 data）。"""
        if kind in ("friend", "maybe") and who:
            return self.appearance.look("friend", who)
        if kind == "stranger" and who:
            return self.appearance.look("stranger", who)
        if kind == "self":
            return self.appearance.look("me", "")
        return ""

    def objects(self, now: float) -> list[Thing]:
        """最近一帧里的物品（座位 / 篝火 / 乐器 / 先祖），左到右、同一边近的在前。

        连续看到 object_min_hits 帧才算；暂停中、或者这一帧已经过时返回空（同 people()）。
        """
        if self.paused:
            return []
        out = []
        seen = [t for t in list(self.last_tracks) if now - t.last <= PEOPLE_STALE and t.hits >= self.cfg.object_min_hits]
        found = [(t, t.cls) for t in seen if t.cls in OBJECT_NAMES]
        ids = {t.id for t in seen}
        # 外形头认出的先祖也算；和 YOLO 自己的先祖框重叠（IoU ≥ 0.5）的是同一个，只报 YOLO 那个（spec §3.5）
        found += [
            (t, "spirit") for t in list(self._others)
            if t.id in ids and self._other_form(t) == "spirit"
            and not any(kind == "spirit" and iou(o.box, t.box) >= 0.5 for o, kind in found)
        ]
        for t, kind in found:
            side = side_of(t.box.x + t.box.w / 2, self._frame_w)
            out.append(Thing(t.id, kind, t.box, side,
                             object_distance(t.box.y2, self._frame_h, self.cfg.object_near, self.cfg.object_far)))
        order = {"左边": 0, "前面": 1, "右边": 2}
        return sorted(out, key=lambda o: (order[o.side], -o.box.y2))

    # ---- 给身体 / 提示词用的（和 EnvWatcher 一样） ----
    def nearby(self, now: float) -> list[str]:
        now = self._frozen(now)
        return [n for n in self.names() if now - self.last_seen.get(n, float("-inf")) <= self.cfg.keep]

    def strangers(self, now: float) -> int:
        """最近 keep 秒里同一帧最多看到几个陌生人（轨迹会断，所以不数轨迹条数）。"""
        now = self._frozen(now)
        with self._lock:
            recent = list(self._strangers)
        return max((n for t, n, _ in recent if now - t <= self.cfg.keep), default=0)

    def unlit(self, now: float) -> int:
        """其中还没点火（黑影）的有几个。"""
        now = self._frozen(now)
        with self._lock:
            recent = list(self._strangers)
        return max((u for t, _, u in recent if now - t <= self.cfg.keep), default=0)

    def describe(self, now: float) -> str:
        parts = []
        if self.place and self._frozen(now) - self.place_at <= self.env_cfg.place_keep:
            parts.append(f"- 看起来在：{self.place}（按画面认的，可能不准）")
        people = self.nearby(now)
        if people:
            parts.append(f"- 你身边现在有：{'、'.join(people)}（画面上能看到他们头顶的名字）")
        n = self.strangers(now)
        if n:
            dark = self.unlit(now)
            note = f"，其中 {dark} 个还没点火（黑影，看不到外观）" if dark else ""
            parts.append(f"- 身边还有 {n} 个陌生人（头顶没有名字{note}）")
        if people or n:
            parts.append("- 别的人看不到不代表不在（可能被挡住或离得远）")
        return "\n".join(parts)
