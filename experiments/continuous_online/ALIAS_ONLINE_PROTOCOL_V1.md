# 中文别名与当前状态检索协议 v1

用固定 Qwen3-1.7B 检验从中文位置记录增量更新的结构化状态是否改善别名、改写、语义更新和主动弃答。与完整历史和字面检索做同题对照。输入按事件顺序回放，模型只收到已到达的公开内容，评分另读标注。每条流和每组重建状态。此实验不训练模型，也不使用上一阶段的 16 槽架构。

## 样本与开发决策

- [开发集](../../evaluation/fixtures/alias_retrieval_dev_v1.jsonl)有 3 条流、22 条观察、17 题（12 个已知地点，5 个未知）。SHA-256 为 `33842d51304331eaea0914c27705c4a81da3d442288985df8f5a86c432c18fbf`。
- [正式留出集](../../evaluation/fixtures/alias_retrieval_holdout_v1.jsonl)有 12 条流、76 条观察、72 题（49 个已知地点，23 个未知）。SHA-256 为 `3721f122c924fad30a81bdc500cb7816e67483865311b8e5983b5eab902a04d6`。两集的查询实体、正确地点和事件文本互不重叠。编写者逐题核对标注，同一编写者负责复核。正式集由另一代理构造，提示迭代只读开发集模型输出。
- 两集均覆盖 12 类，包括显式别名、未知别名、近似编号、语序改写、歧义、隐式指代所在的多事件上下文、位置更正、作废、否认、假设和保持。实体已知而地点未知与实体未知分开记分。正式集只有一个查询前缀超过八条观察，BM25 覆盖了全部 63 个有标注证据的查询；本实验主要比较在线表示和推理，字面检索在高干扰长流上的表现留待单独检验。
- 初次开发探针的完整历史、BM25、结构化和结构化加回退分别答对 6、6、8 和 9 题。开发期允许一次提示修订。补充了别名方向、否定、作废和近似编号示例，并明确让模型复制状态字段。修订后分别答对 5、5、9 和 7 题；结构组实体正确数从 10 增至 13，位置读出从 9 增至 10。冻结修订版以检验实体和读出改善能否迁移。正式结果不用于修改本版提示、样本或门槛。两轮开发原始记录一并保存。

## 对照和判据

| 组 | 答题输入 |
| --- | --- |
| `history` | 全部已到达的观察，按原顺序。 |
| `lexical` | 对当前问题用 BM25 选至多八条观察。NFKC、大小写折叠、保留字母数字、相邻字符二元组，k1=1.2、b=0.75，同分优先近期，按原顺序提供选中的观察。 |
| `structured` | 每条观察调用模型提取 set、alias 或 clear，字面校验后更新无界表。优先字面匹配，必要时调用模型选候选。答案模型接收识别实体及地点。 |
| `structured_fallback` | 同一结构状态和链接流程。没有地点读出时改用 BM25 证据作答。 |

四组解码和系统级答题说明相同，问题及上一条问题可见。比较时保持同一 checkpoint、BF16、贪心非思考解码，答题 24 token、维护 112 token、链接 32 token。先按表中顺序跑一次，再反转组顺序复跑同一留出集。

- 主指标为地点完全匹配，先做 NFKC、去空白和末尾句子标点。未知须输出「不确定」。同时报告已知与未知、类别、配对独胜、误拒答与未拒答。报告实体匹配、读出匹配、读出正确而生成错误，以及维护操作的验证、采纳与拒绝；字面校验只保证操作参数来自观察。
- 结构候选的继续门槛适用于两次正式运行。对完整历史和 BM25 各净增至少 4 道正确题（72 题的 5.6 个百分点），23 道未知题中正确弃答数不低于两基线的最高值，并且总模型输入和输出 token 之和不超过完整历史。分别判定直接结构和带回退结构，任一未达项均报告；这是本轮资源决策阈值，样本量不足以据此声称统计显著。纯读出正确数作为零答题生成的诊断，未注册为第五组。
- 计入每条观察的维护模型调用、可选链接调用，以及每个查询的回答调用。报告输入输出 token、实测模型服务时间和单流最大紧凑 JSON 载荷。序列化字节数是逻辑计量，证据 ID 用于诊断覆盖，脚本顺序耗时只用于本机对照。复跑核对回答、实体、读出、证据、提示、回退和原始模型调用，忽略计时。

## 复现

从仓库根目录运行。固定 checkpoint 文件哈希和所用软件版本由推理结果记录，权重需本地准备。

```bash
python -m evaluation.alias_online_compare --model-dir /path/to/Qwen3-1.7B --fixtures evaluation/fixtures/alias_retrieval_holdout_v1.jsonl --output experiments/continuous_online/alias_online_formal_v1.json
python -m evaluation.alias_online_compare --model-dir /path/to/Qwen3-1.7B --fixtures evaluation/fixtures/alias_retrieval_holdout_v1.jsonl --reverse-arms --output experiments/continuous_online/alias_online_repeat_v1.json
python -m evaluation.score_alias_online --fixtures evaluation/fixtures/alias_retrieval_holdout_v1.jsonl --predictions experiments/continuous_online/alias_online_formal_v1.json --repeat experiments/continuous_online/alias_online_repeat_v1.json --output experiments/continuous_online/alias_online_scores_v1.json
```
