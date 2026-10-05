"""「好友在跟谁说话」的离线评估（spec docs/superpowers/specs/2026-10-05-addressee-design.md §8）。

流程：`addressee label` 从旧运行的 agent.log 里读出聊天、挑句子、用当前规则重放、请 Claude 初标、写 review.md 给人核对；
`addressee eval` 每次用当前规则重放，对着标准答案（人标的，没标就取两边一致的）算三条门槛。
改了规则不用重标：标注按句子 id 存，规则每次现跑。数据放 `datasets/addressee/<时间>/`（不进 git，里面有好友的话）。
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..brain import claude
from ..brain.addressee import LABEL_NAMES, Addressee, Verdict
from ..config import AddresseeConfig
from .tracker import similar

ME_NAME = "我"  # 团子自己，同 brain.occasion.ME
HUMAN = ("跟团子", "跟别人", "大家", "看不出")
RULE_TO_HUMAN = {"me": "跟团子", "other": "跟别人", "all": "大家"}
GOLD = ("跟团子", "跟别人", "大家")
PROMPT_VERSION = 1

OTHER_AS_ME_MAX = 0.05  # 标准答案「跟别人」被判 me 的比例上限
ME_AS_OTHER_MAX = 0.10  # 标准答案「跟团子」被判 other 的比例上限
UNSURE_MAX = 0.40  # unsure 占全部的比例上限

MULTI_WINDOW = 180.0  # 这句前后多少秒内有另一个好友说话 = 多人段落
CONTROL_EVERY = 5  # 其余好友句子每几句取一句当对照
NEARBY_GUESS = 300.0  # 旧日志没有「身边」：按这么多秒内说过话的好友估
CONTEXT = 6  # 给 Claude / review 的上下文：前后各几句
AGREE_SAMPLE = 0.2  # 规则和 Claude 一致的句子抽多少进 review

STAMP = "%Y-%m-%d %H:%M:%S,%f"
_HEAD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) +(\w+) +(\S+): (.*)$")
_JUDGED = re.compile(r"^跟谁说 (.*?)「(.*)」→ (.*)$")

SYSTEM = (
    "你在帮忙标注《光遇》里的聊天记录。游戏里大家只能打字，没有语音；好友之间也会互相聊天，"
    "不是每一句都是冲着「我」说的。「我」是一个叫三彩团子的玩家（聊天里说话人写作「我」）。\n"
    "每一条任务给出一段聊天，用 » 标出要判断的那一句，请判断说话人这一句是在对谁说：\n"
    "- 跟团子：在跟「我」说话（叫了我、接我刚说的话、问我、身边只有他一个人在跟我聊）\n"
    "- 跟别人：在跟另一个好友说话（叫了别人的名字、在回别人的话、两个人你来我往）\n"
    "- 大家：说给在场所有人（打招呼、告别、对大家喊话）\n"
    "- 看不出：根据上下文实在判断不了\n"
    "只回一个 JSON 对象，键是每条任务方括号里的 id，值是上面四个词之一，不要任何别的文字。"
)


@dataclass
class Line:
    id: str  # f"{run}#{序号}"
    run: str
    t: float  # 墙钟秒
    who: str  # 说话人；团子自己 = "我"；读不出说话人 = ""
    text: str
    nearby: list[str] | None = None  # 当时身边的好友（没记录 = None）


def _parse_time(stamp: str) -> float:
    return datetime.strptime(stamp, STAMP).timestamp()


def read_log(path: Path, run: str) -> list[Line]:
    """从 agent.log 里读聊天：`读到: 名字：内容`、团子发出的话（`已发送: …` / `[dry-run] 将会发送: …`）、`跟谁说 …` 里的「身边」。"""
    out: list[Line] = []
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = _HEAD.match(raw.rstrip("\r\n"))
            if not m:
                continue
            stamp, level, logger, msg = m.groups()
            if level == "DEBUG":
                continue
            who = text = None
            if logger == "skydango.brain.body" and msg.startswith("读到: "):
                body = msg[len("读到: ") :]
                if "：" in body:
                    who, text = body.split("：", 1)
                else:
                    who, text = "", body
            elif logger == "skydango.chat.sender" and msg.startswith("已发送: "):
                who, text = ME_NAME, msg[len("已发送: ") :]
            elif logger == "skydango.brain.body" and msg.startswith("[dry-run] 将会发送: "):
                who, text = ME_NAME, msg[len("[dry-run] 将会发送: ") :]
            elif logger == "skydango.brain.body" and msg.startswith("跟谁说 "):
                j = _JUDGED.match(msg)
                if j and "· 身边：" in j.group(3):
                    near = j.group(3).rsplit("· 身边：", 1)[1].strip()
                    for prev in reversed(out):  # 挂到上一条同说话人同内容的
                        if prev.who == j.group(1) and prev.text == j.group(2):
                            if prev.nearby is None:
                                prev.nearby = [] if near == "没人" else [n for n in near.split("、") if n]
                            break
                continue
            if who is None:
                continue
            out.append(Line(f"{run}#{len(out)}", run, _parse_time(stamp), who, text))
    return out


def _friend_of(who: str, friends: Sequence[str]) -> str:
    return next((n for n in friends if who and who != ME_NAME and similar(who, n, 0.75)), "")


def pick(lines: Sequence[Line], friends: Sequence[str]) -> list[str]:
    """挑要标的句子：多人段落里好友说的全要；其余好友句子每 CONTROL_EVERY 句取 1 句当对照。"""
    spoken = [(l, _friend_of(l.who, friends)) for l in lines]
    spoken = [(l, f) for l, f in spoken if f]
    picked: list[str] = []
    control = 0
    for i, (l, f) in enumerate(spoken):
        multi = any(o is not l and g != f and abs(o.t - l.t) <= MULTI_WINDOW and o.run == l.run for o, g in spoken)
        if multi:
            picked.append(l.id)
        else:
            if control % CONTROL_EVERY == 0:
                picked.append(l.id)
            control += 1
    return picked


def replay(
    lines: Sequence[Line], cfg: AddresseeConfig, self_names: Sequence[str], followup: float,
    aliases: dict[str, list[str]],
) -> dict[str, Verdict]:
    """按时间顺序用当前规则重放（每次运行新建一个 Addressee），返回每个非团子句子的判断。"""
    friends = list(aliases)
    out: dict[str, Verdict] = {}
    runs: dict[str, list[Line]] = {}
    for l in lines:
        runs.setdefault(l.run, []).append(l)
    for rl in runs.values():
        rl = sorted(rl, key=lambda x: x.t)
        addr = Addressee(cfg, self_names, followup)
        spoke: dict[str, float] = {}  # 好友 → 最后说话时间
        for l in rl:
            if l.who == ME_NAME:
                addr.said(l.t)
                continue
            f = _friend_of(l.who, friends)
            if f:
                spoke[f] = l.t
            nearby = l.nearby if l.nearby is not None else [n for n, t in spoke.items() if l.t - t <= NEARBY_GUESS]
            out[l.id] = addr.judge(l.t, l.who, l.text, friends=aliases, nearby=nearby)
    return out


# ---- 上下文 / Claude 初标 ----

def _context_map(context: Sequence[Line]) -> dict[str, int]:
    return {l.id: i for i, l in enumerate(context)}


def context_text(context: Sequence[Line], idx: int, mark: bool = True, k: int = CONTEXT) -> str:
    """第 idx 句前后各 k 句（同一次运行里的），要判的那句行首标 »。"""
    target = context[idx]
    rows = []
    for j in range(max(0, idx - k), min(len(context), idx + k + 1)):
        l = context[j]
        if l.run != target.run:
            continue
        rows.append(f"{'»' if j == idx and mark else ' '} {l.who or '（读不出）'}：{l.text}")
    return "\n".join(rows)


def _load_cache(cache: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if cache.is_file():
        for raw in cache.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("v") == PROMPT_VERSION and row.get("label") in HUMAN:
                done[row["id"]] = row["label"]
    return done


def load_claude(cache: Path) -> dict[str, str]:
    return _load_cache(Path(cache))


def _parse_reply(text: str) -> dict[str, str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, str) and v in HUMAN} if isinstance(data, dict) else {}


def claude_labels(
    items: Sequence[Line], context: Sequence[Line], run_fn: Callable[[list], dict], cache: Path, batch: int = 40,
) -> dict[str, str]:
    """请 Claude 给每句标 HUMAN 里的一个；已缓存的跳过，每批写一次盘（额度用完时已做的都在）。run_fn(content) → 含 result 文字的 dict。"""
    cache = Path(cache)
    got = _load_cache(cache)
    index = _context_map(context)
    todo = [l for l in items if l.id not in got]
    for s in range(0, len(todo), batch):
        part = todo[s : s + batch]
        blocks = []
        for l in part:
            blocks.append(f"[{l.id}]\n{context_text(context, index[l.id])}")
        text = "判断下面每条任务里 » 那一句：\n\n" + "\n\n".join(blocks)
        try:
            reply = run_fn([{"type": "text", "text": text}])
        except claude.ClaudeError as exc:
            if getattr(exc, "limit", False):
                raise SystemExit(f"订阅额度用完了：已标的存在 {cache}，额度恢复后重跑同一条命令会接着做") from None
            raise
        labels = _parse_reply(reply.get("result", ""))
        new = [(l.id, labels[l.id]) for l in part if l.id in labels]
        if new:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("a", encoding="utf-8") as fh:
                for i, lab in new:
                    fh.write(json.dumps({"id": i, "label": lab, "v": PROMPT_VERSION}, ensure_ascii=False) + "\n")
            got.update(new)
    return {l.id: got[l.id] for l in items if l.id in got}


# ---- review.md ----

def _rule_human(v: Verdict | None) -> str:
    return RULE_TO_HUMAN.get(v.label, "拿不准") if v else "没判"


def read_review(path: Path) -> dict[str, str]:
    """`### <id>` 下面的 `标：<值>`；空 = 没核对；值不在 HUMAN 里抛 ValueError。"""
    out: dict[str, str] = {}
    cur = None
    p = Path(path)
    if not p.is_file():
        return out
    for raw in p.read_text(encoding="utf-8").splitlines():
        if raw.startswith("### "):
            cur = raw[4:].strip()
        elif cur and raw.startswith("标："):
            val = raw[2:].strip()
            if val:
                if val not in HUMAN:
                    raise ValueError(f"{cur}：标的值「{val}」不是 {'/'.join(HUMAN)} 之一")
                out[cur] = val
    return out


def write_review(
    path: Path, items: Sequence[Line], context: Sequence[Line], rule: dict[str, Verdict], claude_l: dict[str, str],
    seed: int = 0,
) -> int:
    """规则和 Claude 不一致的全写，一致的抽 20%；已填过的 `标：` 保留（即使这次没被选中）。返回写了几句。"""
    path = Path(path)
    filled = read_review(path)
    rng = random.Random(seed)
    index = _context_map(context)
    chosen: list[Line] = []
    for l in items:
        c = claude_l.get(l.id)
        if c is None:
            continue
        agree = _rule_human(rule.get(l.id)) == c
        if not agree or rng.random() < AGREE_SAMPLE or l.id in filled:
            chosen.append(l)
    body = ["# 谁在跟谁说：待核对", "", "在每句的 `标：` 后面填 " + " / ".join(HUMAN) + "（留空 = 没核对）。", ""]
    for l in chosen:
        v = rule.get(l.id)
        body.append(f"### {l.id}")
        body.append(context_text(context, index[l.id]).replace("\n", "\n> ").join(["> ", ""]))
        reason = f"（{v.reason}）" if v and v.reason else ""
        body.append(f"规则：{_rule_human(v)}{reason} · Claude：{claude_l.get(l.id, '没标')}")
        body.append(f"标：{filled.get(l.id, '')}")
        body.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")
    return len(chosen)


# ---- 评估 ----

@dataclass
class Report:
    confusion: dict[str, dict[str, int]]
    other_as_me: float
    me_as_other: float
    unsure_rate: float
    total: int
    gold_total: int
    unreviewed: int
    wrong: list[tuple[str, str, str, str]] = field(default_factory=list)  # (id, 标准答案, 规则判的, 上下文)

    @property
    def passed(self) -> bool:
        return (
            self.other_as_me <= OTHER_AS_ME_MAX and self.me_as_other <= ME_AS_OTHER_MAX and self.unsure_rate <= UNSURE_MAX
        )

    def markdown(self) -> str:
        def ok(b: bool) -> str:
            return "过线" if b else "不过线"

        preds = list(LABEL_NAMES)
        out = [
            "# 谁在跟谁说：离线评估", "",
            f"- 跟别人被判成跟你说：{self.other_as_me:.1%}（上限 {OTHER_AS_ME_MAX:.0%}）→ {ok(self.other_as_me <= OTHER_AS_ME_MAX)}",
            f"- 跟团子被判成跟别人说：{self.me_as_other:.1%}（上限 {ME_AS_OTHER_MAX:.0%}）→ {ok(self.me_as_other <= ME_AS_OTHER_MAX)}",
            f"- 拿不准占全部：{self.unsure_rate:.1%}（上限 {UNSURE_MAX:.0%}，共 {self.total} 句）→ {ok(self.unsure_rate <= UNSURE_MAX)}",
            f"- 有标准答案的 {self.gold_total} 句；规则改了以后新出现的不一致 {self.unreviewed} 句没核对",
            f"- 结论：{'通过' if self.passed else '没通过'}", "",
            "## 混淆矩阵（行 = 标准答案，列 = 规则判的）", "",
            "| 标准答案 | " + " | ".join(LABEL_NAMES[p] for p in preds) + " |",
            "| --- |" + " --- |" * len(preds),
        ]
        for g in GOLD:
            out.append(f"| {g} | " + " | ".join(str(self.confusion[g][p]) for p in preds) + " |")
        out += ["", "## 判错的句子", ""]
        if not self.wrong:
            out.append("没有。")
        for i, gold, pred, ctx in self.wrong:
            out += [f"### {i}", f"标准答案：{gold} · 规则：{pred}", "```", ctx, "```", ""]
        return "\n".join(out)


def evaluate(
    lines: Sequence[Line], picked: Sequence[str], rule: dict[str, Verdict], claude_l: dict[str, str], human: dict[str, str],
) -> Report:
    """标准答案 = 人标的，否则规则和 Claude 一致的那个；「看不出」和没核对的不一致不计。"""
    index = _context_map(lines)
    confusion = {g: {p: 0 for p in LABEL_NAMES} for g in GOLD}
    wrong: list[tuple[str, str, str, str]] = []
    total = unsure = unreviewed = 0
    for i in picked:
        v = rule.get(i)
        if v is None:
            continue
        total += 1
        unsure += v.label == "unsure"
        mine = _rule_human(v)
        gold = human.get(i)
        if not gold:
            c = claude_l.get(i)
            if c is None:
                continue
            if c == mine:
                gold = c
            else:
                unreviewed += 1
                continue
        if gold not in GOLD:
            continue
        confusion[gold][v.label] += 1
        if mine != gold:
            wrong.append((i, gold, mine if v.label != "unsure" else "拿不准", context_text(lines, index[i]) if i in index else ""))
    def rate(gold: str, pred: str) -> float:
        n = sum(confusion[gold].values())
        return confusion[gold][pred] / n if n else 0.0

    return Report(
        confusion, rate("跟别人", "me"), rate("跟团子", "other"), unsure / total if total else 0.0, total,
        sum(sum(r.values()) for r in confusion.values()), unreviewed, wrong,
    )


# ---- 数据集目录 ----

def save_lines(path: Path, lines: Sequence[Line], picked: Sequence[str]) -> None:
    want = set(picked)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for l in lines:
            fh.write(json.dumps({**l.__dict__, "picked": l.id in want}, ensure_ascii=False) + "\n")


def load_lines(path: Path) -> tuple[list[Line], list[str]]:
    lines, picked = [], []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.pop("picked", False):
            picked.append(d["id"])
        lines.append(Line(**d))
    return lines, picked
