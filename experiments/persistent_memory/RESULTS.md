# 持续 M：干扰、记忆年龄与错误纠正

本实验接续episodic_memory，不再每episode清空M。仍是熟悉key/label词表的合成任务，不是自然语言长期记忆结论。

## 协议

每seed先用既有500-episode随机映射训练出记忆读取能力，然后在固定参数下评估持续流：

- 16个候选key、8个候选label；96个新事实事件，每16个新事实插入一次错误反馈，随后立即插入同key的正确反馈，共102个事件。
- 每事件先查询，记录反馈前预测；随后向model透露feedback再写M。隐藏truth只用于evaluator。
- key再次出现时可以改成新truth，旧关联与新关联冲突；按每key最新truth评估所有已出现key。
- 参数不更新；每事件检查固定输入下C bitwise不变；每arm前后检查非M参数hash相同。
- 各arm使用相同训练后权重、同一事件流、同一写lr=.05。

对照：

| arm | 实际机制 | 外部存储/额外成本 |
|---|---|---|
| zero | M恒零 | 无 |
| append | 现有外积直接累加 | 每事件1次write |
| decay_append | 先乘.95再外积累加 | 每事件1次write+衰减 |
| rebuild8 | 最新反馈有界LRU记录8条，清M后重建 | 外部键值ledger，每事件最多8次write |
| rebuild16 | 同上16条 | 外部ledger，每事件最多16次write |

rebuild不是矩阵M自身自动覆写或纠错，是显式外部管理对照。ledger也只知道已到达feedback，错误反馈同样先入ledger，直到corrective feedback到达。

```bash
python -m evaluation.persistent_memory --seeds 0 1 2 --events 96 --training-episodes 500 --output experiments/persistent_memory/three_seed.json
python -m unittest tests.test_persistent_memory_protocol -v
```

三seed原始事件轨迹、年龄桶、写入量、耗时与源码fingerprint均在three_seed.json；pilot.json为更早seed0/32事实预实验，保留不覆盖。

## 三seed均值

| arm | 沿流最新truth平均回忆率 | 最终最新truth回忆率 | 收到corrective反馈后立即读对 | write调用数 |
|---|---:|---:|---:|---:|
| zero | 13.84% | 12.50% | 5.56% | 0 |
| append | 44.19% | 29.17% | 38.89% | 102 |
| decay_append | 58.83% | 52.08% | 38.89% | 102 |
| rebuild8 | 62.15% | 45.83% | 100% | 773.3 |
| rebuild16 | 82.89% | 83.33% | 94.44% | 1362.3 |

平均流评估耗时约11.4/11.7/11.7/13.5/14.9秒/seed，不含约26秒基础训练。不是严格benchmark。

18次纠正事件样本很少，corrective指标不做统计显著性推断。年龄桶是跨事件重复查询，样本相关，不能当独立iid样本计算置信区间。

## 指标的精确定义

- average_latest_truth_accuracy：每事件feedback应用后，对当前所有已见key按evaluator最新truth计分，再平均；不是未观察新事实的预测能力。
- final_latest_truth_accuracy：结束时所有已见key的最新truth正确率。
- prior_truth_accuracy_on_revisits：**反馈前**预测与同key上次truth相比，测已有回忆保持；不能当作本次新truth预测率。
- age桶：从key最后一次事件到当前事件的距离；0–3、4–15、16+。key的truth已更新后以最新truth为准。
- immediate_correct_after_correction：在纠正feedback写入后立刻查询该key，是否为corrected truth；不是“已经采纳错误后成功撤销的比例”。18次事件中append从错到对7次；decay从错到对6次、另1次在纠正前已经正确；rebuild8从错到对18次；rebuild16从错到对17次。没有延迟或中间干扰事件，不能外推延迟纠错。

## 解释

- 短episode内几乎完美的M回忆，不能直接延伸为持续写入下的完美保持。累加会保留旧冲突，最终平均仅29.17%。
- 衰减改善沿流回忆和最终回忆，但也会降低老关联；它没有解决立即纠错（本三seed均值与append相同）。
- 外部latest-feedback记录重建更容易显式覆盖旧值，但使用额外存储与多次写入，不是等成本算法胜出。
- rebuild8保新较好而淘汰旧key，最终整体低于rebuild16；rebuild16仍非100%，与更大负载、矩阵干扰及训练/评估分布差异相容，但没有单因子归因。训练episode内key/label均唯一；持续流则有重复key、重复label、更新和不同年龄，不能只归因于support大小。
- C稳定与参数固定在所有arm均通过，但不能据此说A记忆保持成功。

## 限制

- 本流96事实+6纠正，是有干扰的短流，不能叫无限长期记忆。
- key/label是熟悉的小词表，不验证语义、分布外事实或多模态。
- 正确反馈由模拟环境给定，没有事实验证或confidence gate；错误反馈是人工可控注入。
- raw M没有版本日志，实验没有把rollback策略接入Runtime；rebuild仅外部对照。
- 不比较普通字典/attention检索同预算性能；外部ledger能够直接保存映射，本任务无需神经模型就可检索。
- 当前只有3seed，无显著性声明；实验源码/设备fingerprint已捕获，未保存训练权重快照。
- 逐事件JSON保存当前key的前后预测与整体accuracy，未保存每个历史key在每个事件的完整预测；可以重算整体均值与纠正指标，但不能仅从JSON独立重建全部年龄桶命中数。后续运行应保存更细轨迹。

## 研究意义

M可以被训练为可读关联存储，但持久状态管理、冲突纠正和容量是独立问题。后续应该优先设计显式来源、更新/覆盖协议与预算，而不是把范数增长或短期回忆分数当长期学习成功。
