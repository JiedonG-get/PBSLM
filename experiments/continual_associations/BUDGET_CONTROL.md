# 回放的等 batch / 等更新次数对照

原始实验见 RESULTS.md；本文件是新的控制实验，不覆盖旧结果。

## 对照

新增 `a_current3`：把当次已揭示反馈的样本重复3次组成batch。它与 `a_replay` 使用相同的128次optimizer更新与batch=3，区别是后者用当前样本+2条已观察过的旧样本。所有arm使用同一个基础state、同一在线事件顺序、相同学习率。

```bash
python -m evaluation.continual_associations --seeds 0 1 2 --base-steps 300 --events 128 --arms none a_only a_current3 a_replay --output experiments/continual_associations/batch_budget_control.json
```

本次保存逐事件key、更新前预测、随后揭示的目标与正确性，便于检查prequential统计；完整数据在batch_budget_control.json。

## 三seed均值

| 方案 | 更新样本数/step | prequential | 新domain最终A | 未改变key旧domainA | 改变key旧domainA |
|---|---:|---:|---:|---:|---:|
| 不更新 | 0 | 3.65% | 4.17% | 100% | 100% |
| A-only | 1 | 75.00% | 100% | 87.50% | 54.17% |
| 当前样本重复3次 | 3 | 75.00% | 100% | 87.50% | 54.17% |
| 当前+2条旧样本回放 | 3 | 73.96% | 95.83% | 100% | 100% |

C bitwise稳定、受保护state哈希不变：所有seed/arm均通过。基础学习旧task准确率均100%。

每arm平均耗时：none3.53秒，A-only7.78秒，current3 7.92秒，replay7.71秒。计时不是严格性能benchmark（单顺序、GPU同步方式、路由union不同）；不把微小时间差当作性能优劣。

## 解释与局限

在这个任务上，重复当前样本并未改善保持，而旧样本回放改善保持；排除了“仅batch数量变大”的解释，但没有单独隔离旧样本信息与当前梯度压力下降的作用。仍不是严格等FLOPs：不同内容会改变union、参与块数，FLOPs没有profile。

两个方案每次用mean loss：replay中当前反馈权重为1/3，另外2/3为旧样本；这既是防遗忘机制，也是优化目标变化，不能仅解释成“加缓存”。所有旧映射在domain1、新映射在domain2，两者在任务定义上可同时满足。

仍只有16个key、128事件、3seed。旧映射保持测的是训练过/回放过的16个例子，不能称作held-out旧任务泛化。没有自然语言/长期漂移、重复阶段切换、对抗污染或单路径预算baseline。不是PBSLM优于其他模型的证据。
