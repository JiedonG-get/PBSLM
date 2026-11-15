# 最小持续适应实验：三 seed 初步结果

2026-11-15。这是用户授权继续研究后的**新诊断实验**，不是 Phase 1 重训，不覆盖历史结果，也不是自然语言 Online 系统验收。

## 问题与任务

在真实 PBSLMv2 masked 路径中，A-only 更新能否在每次反馈到达后学习新映射？C 和旧任务保持是否分别成立？现有 M 写入是否贡献额外收益？

合成任务只有 16 个 key、48 词表、上下文长度4。先在 domain=1 学会 key→label；在线流为 domain=2 中8个 key 的新映射。context 内明确给出domain，所以同一key在旧domain的原答案仍有意义，遗忘不是任务本身不可兼容。

- 模型：k=8, depth=1, d=32, heads=4, E=2, refine=1, Flow=on, M=on, independent masked routing。
- 输出：固定单位码本连续向量；本小词表实验使用精确 cosine CE（tau=.1），**与 Phase 1 sampled-negative InfoNCE 不同**，不混用历史指标。
- 基础学习：每seed 300步、batch16、AdamW lr=.002；A/C同时监督。三个seed的旧任务A/C均达到100%。
- 在线流：128个事件；先预测并计分，再揭示当前答案；只有此后才更新A或写M。各arm共享同一初始state和事件顺序。
- A更新：既有O1六组白名单，lr=.001，128个step。C、共享底座、E、code冻结。
- M写入：观察时E特征作key，当次反馈label向量作value；写lr=.05，decay=.995。仅是同步因果实验调用，不是生产级记忆事务/回滚实现。
- Replay：每次当前反馈样本+2个已观察过的domain1样本；与A-only更新次数相同，但**样本计算量是3倍**，不是等FLOPs结论。
- `shuffled_memory` arm名称沿用脚本，但实际是确定性的循环错label对照，不是统计意义的随机洗牌；解释时称错值记忆。

## 运行与成本

```bash
python -m evaluation.continual_associations --seeds 0 1 2 --base-steps 300 --events 128 --output experiments/continual_associations/three_seed.json
```

GPU：RTX 4060 Laptop 8GB；PyTorch 2.11.0+cu128。三seed总耗时154.88秒；每seed约51–52秒。

pilot.json为更早的单seed预实验（没有replay arm），保留不覆盖；three_seed.json为本文结果来源。

## 三 seed 均值

| arm | 在线预测正确率（更新前） | 结束后新domain A | 未改变key的旧domain A | 改变key的旧domain A |
|---|---:|---:|---:|---:|
| 不更新 | 3.65% | 4.17% | 100% | 100% |
| 只更新A | 75.00% | 100% | 87.50% | 54.17% |
| A+回放 | 73.96% | 95.83% | 100% | 100% |
| 只写M | 3.65% | 4.17% | 100% | 100% |
| A+M | 75.52% | 100% | 87.50% | 54.17% |
| 错值M | 3.65% | 4.17% | 100% | 100% |

所有18个arm/seed运行均通过：

- 相同旧输入下C输出bitwise不变；
- 受保护参数/buffer哈希不变（A白名单和M显式排除）；
- 模型每个arm从同一base state恢复。

## 可以得出的结论

1. 当前A更新闭环在这个小任务上能适应反馈；不是只靠测试张量shape。
2. C稳定与A旧能力保持是不同问题：A-only仍然遗忘，虽然C不变。
3. Replay在这个任务上显著改善旧映射保持，但付出了额外样本计算；seed2新映射为87.5%，并非所有seed完美。
4. M范数增长不代表有用记忆。只写M与不更新一致，A+M与A-only的差异很小，不支持宣布M有效。

## 必须保留的限制

- 任务极小、只有3个seed，不能外推自然语言、长上下文或复杂环境。
- 基础阶段M为零且不写入，因此E读写通路没有被训练成有用记忆检索器；M无收益可能来自该机制尚未被教会，不能据此否定E/M长期设计。
- 旧任务仅16个映射，在线流仅128事件；没有长期容量、抗污染、反复切换、多用户或实时并发。
- 没有“普通模型同预算”架构对照，不能说PBSLM优于Transformer/单路径。
- 没有完整事件逐条轨迹和训练checkpoint，只保存配置、种子、聚合结果与脚本；下一批实验需更完整过程证据。
- 这是明确的同步test-then-train预序贯学习，不是持续输入/打断/续连Runtime。
- 没有置信区间或显著性检验，均值只是描述统计。

## 下一步

先验证“训练时有记忆/无记忆”是否使M获得可测因果收益，并增加有界回放/等样本预算对照；保留负结果。之后再推进Runtime状态机。README应把这些结果与Phase1自然语言baseline分开。
