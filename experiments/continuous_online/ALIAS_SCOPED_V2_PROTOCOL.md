# PBSLM 在线别名记忆评估协议 v2

## 研究范围

本实验比较固定 Qwen3-1.7B 骨干上的四种中文在线记忆与检索系统。它不训练 Qwen，也不修改 PBSLM 的网络结构或权重。因此结果用于评估 PBSLM 项目中的外部在线记忆方案，不作为 PBSLM 模型本体能力或多模态能力的证据。

每条事件流和每个组别都从空状态开始，按 fixture 中的事件顺序重放。系统输入只包含当前事件及此前已到达的公开观察。查询 oracle、目标地点、类别标签、未来事件和评分证据均不提供给系统。评分器在推理完成后读取 oracle。

## 固定数据

| Split | 文件 | 流 | 观察 | 查询 | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| 开发集 | `evaluation/fixtures/alias_retrieval_dev_v2.jsonl` | 4 | 23 | 12 | `9c26bc0bd30534e615e2807f3dfc10cde292606dcfd12d897bba0045d75053ea` |
| 正式留出集 | `evaluation/fixtures/alias_retrieval_holdout_v2.jsonl` | 12 | 158 | 84 | `a38b410d54740372241d451b06e04687b460ddc573634ddad821457c81966372` |

正式运行前必须再次验证两个文件的字节级 SHA-256 与表中一致。任何 fixture 字节变化都使该运行不属于本协议。开发集用于提示、状态规则和实现迭代。正式留出集不得用于提示选择、规则调参、实现修正或继续门槛调整。完成留出集正序及反序运行前，不查看其题面或模型输出。

## 被测系统

候选 `scoped` 是固定生成模型外的一层确定性状态管理。维护模型从待处理观察中抽取 `register`、`set`、`clear`、`bind`、`rebind` 和 `revoke` 操作。观察在下一个查询到达时，先由固定提示中的线索规则筛出需要维护的部分，再按原顺序组成一个批次传给维护模型。解析器要求事件 ID 属于该批次，并对操作参数执行来源文本字面校验。地点写入还须满足正向、已确认地点关系规则。无后续查询的尾部观察不触发维护调用，并在 ingress trace 中标为跳过。

状态分别记录规范物件、带来源范围的别名，以及三态地点事实。地点状态为 `located`、`no_location` 或 `missing`。明确确认地点后为 `located`，明确注销地点记录后为 `no_location`，新登记但无地点证据时为 `missing`。别名可按来源建立、重绑或撤销。查询时由确定性状态读出实体和地点，不调用答题模型或候选链接模型。无唯一实体或无可读地点时输出「不确定」。

`scoped` 的维护批次、event ID、原始观察文本、操作验证结果、状态拒绝记录和模型调用均保存在 trace 中。来源范围是别名事实的一部分，不能由地点事实或查询中的相似词替代。

## 对照组

| 组别 | 定义 |
| --- | --- |
| `history` | 将查询前全部已到达观察按原顺序交给答题模型。 |
| `lexical` | 对查询运行固定 BM25，最多取 8 条观察，选中内容恢复为原始时间顺序后交给答题模型。使用 v1 的 NFKC、大小写折叠、字母数字保留、相邻字符二元组、`k1=1.2`、`b=0.75` 和同分近期优先规则。 |
| `structured` | 原样使用 v1 的逐观察维护、结构化状态、候选实体链接和模型生成答题流程。 |
| `scoped` | 使用本协议定义的批量维护、来源范围别名、地点三态和确定性直接读出流程。 |

四组使用同一冻结 Qwen3-1.7B checkpoint、BF16、贪心非思考解码和本机推理适配器。v1 对照的维护上限为 112 tokens、链接上限为 32 tokens、答题上限为 24 tokens。`scoped` 维护上限为 192 tokens，且没有链接或答题生成调用。`lexical` 上限为 8 条。组别顺序正序为 `history`、`lexical`、`structured`、`scoped`，反序复跑为其逆序。所有配置以预测文件的 config、checkpoint 文件 SHA-256 和运行环境记录为准。

## 评分与预注册判据

主指标是 84 道查询的地点答案精确匹配。答案按评分器规则执行 NFKC、首尾空白清理和末尾句子标点清理，然后逐字比较。金标为「不确定」的查询只有输出「不确定」才算正确。实体未知、别名歧义、实体已知但地点未知和地点已知的表现分别报告，不合并成同一种错误。

预注册成功门槛须在正序和反序正式运行中分别满足。`scoped` 的总题正确数不低于 `history`、`lexical`、`structured` 三组中最高者；金标为「不确定」的题目中，正确弃答数不低于三组基线中的最高者；`scoped` 所有模型调用的输入与输出 token 总数严格低于 `history`。任一门槛未满足均如实报告为未达标，不更改系统、样本、阈值或评分定义，也不据此省略正式结果。

同时报告各组总准确率、已知与未知子集、fixture case 类别、逐题配对胜负、实体准确、实体状态准确、地点状态准确、地点读出准确、实体与地点同时准确、memory evidence 覆盖、操作接受与拒绝数量，以及 fallback 数量。配对结果作为描述性统计，不据 84 题得出统计显著或普遍泛化结论。

## 成本、延迟与可复算性

每个组别报告模型调用次数、输入 tokens、输出 tokens、输入输出 token 合计、分用途调用与 token 数、模型服务时间、stream 总耗时、host 时间和单流峰值紧凑 JSON 状态字节数。成本从原始生成 trace 复算，不能仅引用汇总数字。逻辑 JSON 字节不代表 Python 堆内存或生产环境常驻内存。

逐题 `query_seconds` 从该组处理当前查询前开始计时。`scoped` 包含该查询触发的批量维护与直接读出，其他组包含查询时的实体链接和答题调用。评分器要求逐题计时为有限非负数，且不短于 trace 中锚定到该 query event 的模型服务时间。按组及 split 汇总查询数、总时间、均值、中位数、最近秩 P90、最小值和最大值。报告的是本机回放测量值，不等同于包含事件等待时间的在线服务 SLA，也不作为成功门槛。

正序与反序运行必须使用相同 fixture、checkpoint、提示、状态实现及软件配置，只有组别运行顺序不同。评分器逐组比较 query answers、实体、状态、读出、证据 ID、提示、fallback、available observation 数、维护原文及模型原始调用和 ingress 操作。计时不要求逐位相同，但应分别呈现。若结果不一致，报告各不一致字段和调用流，不选择较好的一轮替代另一轮。

推理与评分记录必须保存 fixture SHA-256、checkpoint 每个文件的 SHA-256、runner/scorer/状态实现及其依赖源文件 SHA-256、协议和测试 SHA-256、Python、PyTorch、Transformers、CUDA/cuDNN、GPU 型号及确定性设置。正式运行前以冻结清单记录所有源文件和 checkpoint 哈希。dirty worktree 状态须保留在 provenance 中；不得把未提交源文件误报为 commit 内容。

## 已知语义差异

正式留出集有 5 道题的标注把「撤销其他所有者后仅剩一个来源别名」视为可解析，即使问题没有明确给出来源。当前 `scoped` 实现对来源范围别名要求查询明确给出匹配来源；没有来源时不会用唯一存续的 scoped binding 代替 global binding。这是预先发现的系统与金标策略差异，五题仍保留在主分数中，不作例外加分、剔除或阈值调整。按既有 case 标签和逐题状态报告其影响。不得据正式结果为这五题修改实现后再把修改版称作本轮冻结评估。

## 开发集冻结观察

最终开发运行使用上述开发 fixture 和固定 checkpoint。正序与反序各 12 道查询，评分器逐组比较的答案、实体、读出、证据、调用与 ingress 记录全部一致。离线评分结果可从两份原始预测文件精确复算。开发集 `scoped` 为 8/12，`history`、`lexical` 和 `structured` 各为 4/12。金标「不确定」共 3 题，`scoped` 正确 2 题，三个基线分别正确 1、1、0 题。

开发集金标实体状态分布为 resolved 10、ambiguous 1、unknown 1；地点状态分布为 located 9、no_location 1、missing 2。没有实体已解析但地点未知的样本，ambiguous 和 unknown 各只有 1 题，不能据此校准或概括这些状态的泛化表现。

开发集资源记录显示 `scoped` 使用 3,853 个输入与输出 tokens、8 次维护调用，`history` 使用 3,145 tokens、12 次答题调用。因此开发集尚未达到预注册的 token 成本门槛。主正序测得的单题平均查询时间为 `scoped` 0.676 秒、`history` 0.237 秒、`lexical` 0.210 秒、`structured` 0.142 秒。该本机开发测量仅作描述，不能替代正式留出集的门槛结果，也不得与早期探索运行混算。

## 冻结产物

正式评估使用 `experiments/continuous_online/alias_online_freeze_v2.json` 中列出的冻结清单。清单包含 runner、scorer、共享源文件、测试、两个 fixture、协议、最终开发运行与评分、固定 checkpoint 的逐文件清单、运行环境及哈希。正式留出题面在冻结记录完成后仍保持未查看。

## 冻结记录

开始正式留出推理前，将清单中的文件哈希与工作区和 checkpoint 核对。评分时同时核对两个原始预测文件中的来源哈希与运行环境。`scoring_dependencies_sha256` 单独记录评分器所用、未由推理 runner 捕获的共享评分模块。

| Artifact | SHA-256 |
| --- | --- |
| v2 runner 及共享源文件清单 | 见 `alias_online_freeze_v2.json` 的 `source_sha256` |
| v2 scorer、状态实现与测试清单 | 见 `source_sha256`、`scoring_dependencies_sha256` 和 `test_sha256` |
| v2 协议文件 | 见 `alias_online_freeze_v2.json` 的 `protocol_sha256` |
| 开发与正式留出 fixture | 见 `alias_online_freeze_v2.json` 的 `fixture_sha256` |
| Qwen3-1.7B checkpoint 文件集合 | 见 `alias_online_freeze_v2.json` 的 `checkpoint.checkpoint_files` |
| 最终开发预测与评分文件 | 见 `alias_online_freeze_v2.json` 的 `development_artifacts` |

## 复现命令

在仓库根目录、同一固定 checkpoint 下先执行正序、再执行反序，输出路径必须事先不存在。

```bash
python -m evaluation.alias_online_compare_v2 --model-dir /path/to/Qwen3-1.7B --fixtures evaluation/fixtures/alias_retrieval_holdout_v2.jsonl --expect-fixture-sha256 a38b410d54740372241d451b06e04687b460ddc573634ddad821457c81966372 --output experiments/continuous_online/alias_online_holdout_forward_v2.json
python -m evaluation.alias_online_compare_v2 --model-dir /path/to/Qwen3-1.7B --fixtures evaluation/fixtures/alias_retrieval_holdout_v2.jsonl --expect-fixture-sha256 a38b410d54740372241d451b06e04687b460ddc573634ddad821457c81966372 --reverse-arms --output experiments/continuous_online/alias_online_holdout_reverse_v2.json
python -m evaluation.score_alias_online_v2 --fixtures evaluation/fixtures/alias_retrieval_holdout_v2.jsonl --predictions experiments/continuous_online/alias_online_holdout_forward_v2.json --repeat experiments/continuous_online/alias_online_holdout_reverse_v2.json --output experiments/continuous_online/alias_online_holdout_scores_v2.json
```

正式留出运行仅在上述协议、冻结清单、开发观察和输出哈希全部登记后执行。留出正序预测写入 `experiments/continuous_online/alias_online_holdout_forward_v2.json`，反序预测写入 `experiments/continuous_online/alias_online_holdout_reverse_v2.json`，复算评分写入 `experiments/continuous_online/alias_online_holdout_scores_v2.json`。这些输出路径须预先不存在，结果作为新文件保存。
