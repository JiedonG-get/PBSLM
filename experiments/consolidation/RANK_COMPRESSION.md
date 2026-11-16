# 分批低秩统计记忆：容量与迁移取舍

## 固定设计

```bash
python -m evaluation.rank_compression_control
python -m unittest tests.test_low_rank_ridge tests.test_cumulative_ridge -v
```

冻结trained E、同checkpoint/标签/soft teacher/每stage200次采样顺序；3批各4熟悉实体，2训练模板，共24个独特输入反复抽样600次。无神经训练、不改A/M、无Runtime接线。

每stage先合并已有sketch Gram/cross与当前stage全部约束，再eigh取top-rank基，投影cross；丢弃方向不可恢复。λ=.01，bias也属于增强特征，CPU float64。rank4/8/16/32/97在运行前固定，不按held选择。

所有rank共享输入与目标。测试/未来group特征提前缓存仅用于评分，没有进入basis/cross拟合；每stage压缩只看过去sketch和当前训练约束。

## 三seed最终三批均值

| rank | train | held | 持久tensor payload |
|---:|---:|---:|---:|
| 4 | 58.33% | 55.56% | 4,672bytes |
| 8 | 88.89% | 76.39% | 9,344bytes |
| 16 | 100% | 88.89% | 18,688bytes |
| 32 | 100% | 91.67% | 37,376bytes |
| 97（完整增强维度） | 100% | 91.67% | 113,296bytes |

原未压缩ridge Gram+cross+weight149,768bytes；sketch保存U/s/projected-cross，不保存完整weight，所以比较同时含表示/存储布局变化，不是单纯rank压缩比。高rank97的特征分解布局本身已有不同常数。

**秩32超过独特训练输入24条，所以保住全部效果并不意外，不是大容量压缩突破。** 秩16保住train但held略低，也再次说明训练映射保留与跨表面迁移不同。

rank97每stage输出数值与原累计ridge比较，max prediction difference<1e-6断言通过（实际最大约1.37e-12）；完整prediction/target、压缩损失和storage在rank_compression200.json。新增sketch单测与全仓库回归通过：190 tests OK（9.302秒CPU），all_tests_rank_compression.log。

## 内存/计算边界

- payload只计算持久U/s/cross tensor，另有diagnostics/Python对象和底座/码本未计入。
- 每stage临时重建97×97 dense Gram并eigh，输入/测试feature缓存也存在；不声称降低峰值内存或实时成本。
- 目前每批压缩一次，不是逐事件压缩；多次压缩可能让方向丢失，不能假定顺序不敏感。
- discarded_gram_trace/discarded_cross_norm相对“先前sketch＋当前批”，不是原始全历史误差总量，日志明确标记。
- 只保留按特征能量top方向可能丢掉低方差而与答案重要的方向，当前方法不是任务最优压缩。
- 无删除/替换；误信息一旦投影保留，不能精确撤销。须与可撤销work ledger分开，不直接取代临时记忆。

## 研究结论

小任务上存在可测容量—迁移取舍；有效统计可以较小，但尚未证明对更多独立样本/新实体/真实文本适用。不能把600次重复采样说成600条独立记忆容量。

下一步应提高独立输入数量和任务多样性，并比较等字节保留样本/向量检索/任务感知sketch，而非继续在24条约束上调rank获得漂亮结果。Online运行需要明确何时压缩、何时撤销与何时拒绝旧结论；此模块没有解决这些语义策略。
