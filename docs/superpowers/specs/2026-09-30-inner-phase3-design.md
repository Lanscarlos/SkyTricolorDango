# 内心层 第 3 期：性格（口头禅、老梗、看法，放开太乖的规则）— 设计

日期：2026-09-30　状态：**代码已完成（计划 `docs/superpowers/plans/2026-09-30-inner-phase3.md`），待真机验证**

## 背景

内心层总路线见 `2026-09-30-inner-phase1-design.md` 开头：1 账本 → 2 反思（`2026-09-30-inner-phase2-design.md`）→ **3 性格（这份）**。
前两期让团子记得人、有心情；这一期让它**像它自己**：说话有自己的口头禅、和熟人有老梗、对事情有立场，敢唱反调、会拒绝、会互损，而不是有求必应的助手。

## 需求（已和用户确认）

1. 放开四样"太乖"：**唱反调、有立场**；**熟人之间互损**；**会拒绝、会偷懒**；**有怪癖和执念**
2. 沉淀出来的口头禅 / 老梗 / 看法**自动生效**：每类有上限、久不用会淡出；`profile.md`（用户写的）永远优先、程序不改；`memory show` 能看，用户能直接删
3. 做法 A：改大脑提示词（四样行为的许可和分寸）+ 反思沉淀性格档案（`persona.json`）进系统提示词；互损只对熟人，对方难过就收着

底线不动：身份（不说自己是真人）、隐私、不约线下、不暧昧、不说教、对未成年人友善、不骂人。大脑模式专用。

## §1 性格档案（`memory/inner/persona.json`，只在 live 写盘，dry-run 只在内存里）

```json
{"version": 1,
 "catchphrases": [{"text": "害，懒得动", "since": 0, "last_used": 0, "hits": 2}],
 "jokes": [{"who": "小明", "text": "上次把团子带进冥龙嘴里", "since": 0, "last_used": 0, "hits": 1}],
 "opinions": [{"topic": "雨林", "stance": "湿漉漉的，谁爱去谁去", "since": 0, "last_used": 0, "hits": 0}]}
```

| 类 | 是什么 | 上限 |
|---|---|---|
| 口头禅 `catchphrases` | 团子自己说过、有人接 / 笑过的说法 | `catchphrases_max`（5） |
| 老梗 `jokes` | 团子和某个好友之间的梗，挂在那个人名下 | 每人 `jokes_per_friend`（3）、总共 `jokes_max`（20） |
| 看法 `opinions` | 一个话题一句立场 | `opinions_max`（10） |

**沉淀**（接在第 2 期反思 JSON 后面，`persona` 开着时反思多交两样）：
- `persona_add`：`{"catchphrases": ["…"], "jokes": [{"who": "…", "text": "…"}], "opinions": [{"topic": "…", "stance": "…"}]}`
- `persona_used`：这段时间团子用过、而且有人接的条目原文（口头禅 / 老梗的 `text`、看法的 `topic`）→ `hits += 1`、`last_used = now`

**规矩**（`Persona.apply`，代码守）：
- 每条 `text` / `stance` 截 30 字，`topic` 截 10 字；空的丢
- **敏感词**（`SENSITIVE`：胖、瘦、丑、矮、长相、身材、脸、爸、妈、家里、成绩、考试、分数、几岁、年纪、学校、班）出现在条目里 → 丢
- 老梗的 `who` 要 `match_friend` 对上好友，且关系卡 `len(days) ≥ grudge_min_days`（和别扭同一个"熟人"门槛），否则丢；`who` 在"收着点"（§2）里 → 丢
- 和已有的几乎一样（`similar ≥ 0.75`）不重复加（口头禅比 `text`、老梗比同一个人的 `text`、看法比 `topic`）；看法的 `topic` 相同 → 换成新的 `stance`（`since` 更新，日志记一笔）
- 超上限：先删 `hits` 最少的，再按 `last_used or since` 最旧的
- **淡出** `fade(now)`：`now − (last_used or since) > fade_days`（14 天）的删掉；启动时和每次 `apply` 后各做一次

**大脑看到**：
- 系统提示词「你攒下的性格」（`Persona.section()`，启动时算一次，放在「日子」之前；空的不写）：
  > ## 你攒下的性格（慢慢和大家玩出来的；人设里写的优先）
  > 口头禅：害，懒得动；樱花发型天下第一
  > 看法：雨林——湿漉漉的，谁爱去谁去；霞谷滑道——爽
  > 和小明的老梗：上次把团子带进冥龙嘴里
  > 用得自然，别每句都用；同一个梗一次上线最多用一两回。
- `arrive`：`joke_note(name)` 接在"你惦记着"之后："。你们的老梗：上次把团子带进冥龙嘴里"（最多 2 条，按 `hits` 多的先）

## §2 放开提示词、「脾气」、收着点

**提示词**（`brain/prompt.py`，`persona` 开着才改）：
- 「说话」一节那句"接对方的话往下聊；对方没问就别硬找话题。"换成"接得住就接，不想接也可以换个话头或者吐槽一句；对方没问就别硬找话题。"（主动开口开着时后半句照旧被换掉）
- 加一节 `TEMPER_RULES`（「脾气」，放在交情 / 心情规矩之后），四条：
  - 有立场：不同意就直说（“我觉得雨林就是丑”），别动不动“你说得对”；可以争两句、嘴硬，但不吵架、不人身攻击
  - 熟人之间可以互损（状态里写了交情、一起玩过好几天的）：损事不损人——路痴、手残、放鸽子可以；外貌、身材、家里、成绩、年龄不碰。对方不接梗、语气不对就收；刚认识的、陌生人不损
  - 会拒绝、会偷懒：不想做就说不（“不要，懒”），不想聊可以敷衍两句；卡洛的 # 命令、有人真着急或真难过时别偷懒
  - 有执念和怪癖：人设和「你攒下的性格」里的执念可以反复提、口头禅会用，偶尔说点出人意料的话；别每句都用
  - 状态里写着“收着点”的人：不损、不唱反调、不拒绝，好好说话
- 底线各节（身份、隐私、不约线下、未成年人）一字不动

**人设模板**：`cli.PROFILE_TEMPLATE` 在「喜好和看法」后面加「## 脾气」：
```
## 脾气
- 毛病：嘴硬，爱装懒，被催就更不想动
- 执念：樱花发型天下第一，谁说不好看跟谁急
- 雷点：讨厌被说“像机器人”（打哈哈带过，别较真）
```
已有的 `memory/profile.md` 由用户自己加（同第 1 期「喜好和看法」）。

**收着点**（`brain/body.py`，代码兜底）：`_heard` 里任何好友（`match_friend` 对上）的话 `sounds_upset`（第 2 期的 `DISTRESS`）→ `_soft_until[名字] = wall + soft_minutes × 60`；
status 在"心里"后面加"对小明收着点（他刚说「今天真的很难过」）"（多人用"、"连，原话截 20 字；过期不写）；反思套性格时把收着点的名单传给 `Persona.apply`。
`persona = false` 时不记、不写。

## §3 反思（`inner/reflect.py`）

- `Reflector(cfg, llm, clock, threaded, system=REFLECT_SYSTEM)`：系统提示词可传；`persona` 开着时 cli 传 `REFLECT_SYSTEM + "\n\n" + PERSONA_SYSTEM`
- `PERSONA_SYSTEM` 要点：顺便想想团子自己的性格——这段时间团子说过的、有人接 / 笑的说法可以记成口头禅；和某个熟人之间反复提的事、外号可以记成老梗；团子表过态的话题记成看法（话题 ≤ 10 字、立场 ≤ 30 字）；团子用了已有的条目而且有人接，写进 `persona_used`；损人的、拿外貌 / 家里 / 成绩 / 年龄开玩笑的不记；没有就给空的。输出在原来的 JSON 里多 `persona_add` / `persona_used` 两项
- 材料（`materials`）多一段"你攒下的性格"（`Persona.section()` 的正文，没有写"（还没有）"）；`persona` 关着不带
- 结果在身体线程里 `Persona.apply`（后台 / 最终反思都一样），live 写 `persona.json`

## §4 接线

- `cli`：`persona` 开着且有账本时 `persona = ledger.store.load_persona(quarantine=persist)` → `fade(now)`；交给 `Body(..., persona=…)`、`brain_prompt(..., persona_text=persona.section(), temper=True)`、`Reflector(system=…)`；
  最终反思 `finish_reflection(..., persona=…)` 一起套、live 写盘；`memory show` 末尾打印性格档案（`Persona.show_lines()`，只读）
- `Body`：`apply_reflection` 顺带 `persona.apply(...)` + 写盘；arrive 接 `joke_note`；收着点；`reflect_materials` 带性格段
- 都走 `_inner_call`（出错只记日志）

## §5 配置（`[inner]` 追加）

| 键 | 默认 | 作用 |
|---|---|---|
| `persona` | `true` | 性格总开关；`false` = 第 2 期原样（需要 `reflect = true` 才沉淀，`reflect = false` 时只有提示词和收着点） |
| `fade_days` | 14 | 多少天没用的条目淡出 |
| `catchphrases_max` | 5 | |
| `jokes_per_friend` | 3 | |
| `jokes_max` | 20 | |
| `opinions_max` | 10 | |
| `soft_minutes` | 30 | 好友说难过后多久内对他收着点 |

`config.example.toml` 加；管理面板加 `inner.persona`。**数字都是估的**。

## §6 出错怎么办

- `persona.json` 读不了：改名 `.bad-<时间>`（dry-run 不改名），从空的开始
- 反思给的条目不合规矩：逐条丢、日志 INFO
- 拼「你攒下的性格」出错：这一节不写；身体里新调用出错只记日志

## §7 测试（`python -m pytest -q`，假模型，不碰设备）

- `Persona.apply`：上限和淘汰顺序；淡出；敏感词丢；老梗冲不熟 / 陌生人丢、错字对上正名；收着点的人不记；同话题看法换立场；重复不加；`persona_used` 加 hits；字段截长；格式不对不崩
- `section()` / `joke_note()` 文字；空档案不写
- store：`persona.json` 往返、坏文件改名 / 只读不改名
- 反思：`persona` 开着系统提示词带 `PERSONA_SYSTEM`、材料带性格段；关着和第 2 期一样
- 身体：说难过 → status 有"收着点"，`soft_minutes` 后没了；arrive 带老梗；反思结果写 `persona.json`（只在 live）
- 提示词：开着有「脾气」「你攒下的性格」、"接得住就接"；关着和第 2 期逐字一样；底线各节不变
- cli：live 写、dry-run 不写；`memory show` 有性格档案；`PROFILE_TEMPLATE` 有「脾气」

## 真机验证

1. 聊一晚上，`memory show` 看沉淀了什么，像不像团子；有没有沉淀进奇怪 / 损人的东西
2. 故意夸雨林 / 说它喜欢的东西不好，看会不会唱反调、嘴硬
3. 熟人互损守不守边界；说一句"难过"，30 分钟内是不是收着
4. 让它做点事看会不会拒绝 / 偷懒；`#` 命令照做
5. 还是太乖：把 `history.jsonl` 挪到 `memory/archive/`（旧的乖回复会把模型拉回去，见 CLAUDE.md「记忆」一节）再试

之后把结论写进 CLAUDE.md「内心层」一节。
