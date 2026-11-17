# 写入键也有噪声：状态污染发生在查询之前

本批无训练、固定content_fast600 delta权重和value读出，比较压缩矩阵与容量8显式内容槽；同输入key/value，查询始终干净原key。没有历史fallback/Runtime/语言成绩。

```bash
python -m evaluation.write_noise_control
python -m evaluation.summarize_write_noise
```

## 预定流

3seed×16流，prime8个exact key/value，再512次hot0更新。每步都先抽value和16D扰动，写key=normalize(original hot key+sigma×eps)，sigma0/.05/.15。即使sigma0也抽eps，保证条件配对。noise仅作用于后续write，prime和query不受扰动。

两个geometry：random球面与两两cos.9。**相关组cos.9恰好等于槽匹配阈值.9，有限精度会使priming合并受边界影响；这是匹配边界压力，不是稳定隔开的身份协议。** 不将该组误合并数量当泛化估计。

## 最终热/冷准确率（三seed pooled）

| geometry/写sigma | fast热 | fast冷 | slots热 | slots冷 |
|---|---:|---:|---:|---:|
| random/0 | 100% | 80.95% | 100% | 100% |
| random/.05 | 100% | 8.33% | 93.75% | 97.32% |
| random/.15 | 100% | 1.79% | 16.67% | 0% |
| correlated/0 | 100% | 12.20% | 100% | 16.67% |
| correlated/.05 | 100% | 12.20% | 89.58% | 16.37% |
| correlated/.15 | 100% | 10.12% | 14.58% | 2.98% |

每格最终hot48、cold336query。初始random cold fast88.99%、slots100%；correlated初始fast22.02%、slots26.79%，已较差，不把最终低分全归因512次遗忘。未知query每流固定1个独立随机键重复probe，最终只有48query，不代表广泛拒答安全；完整误接受数据在summary。

## 操作诊断（不是模型得到的身份标签）

脚本给每个槽附一个外部owner标签，用于事后判断该槽最近是哪种语义writer写入，**不进入分配、匹配、更新或query**。这不是证明槽内部只表示某一身份，更不是语言来源认证。

总计每条件24576次hot写：

- random/.05：9次未匹配的同身份重复分配，9次淘汰其他owner；
- random/.15：21795次同身份重复分配/满槽淘汰，其中336次淘汰冷owner，最终冷记录全失；
- correlated prime阶段384次写中319次匹配到另一owner，发生在阈值边界；后续噪声/槽位原已丢失共同影响结果。

计数定义：wrong_merge=匹配成功但最近owner不同；duplicate_allocation=不匹配且已有相同owner槽；eviction=未匹配且满槽；cold_eviction=淘汰owner不同于当前writer。后者在hot阶段可解释冷owner淘汰，prime阶段仅是不同owner，不混用。

## 能支持的解释

压缩delta对扰动方向也写入残差，反复热写可能影响更多方向，即使clean query也会错。显式槽在轻噪声下相对稳健，但完全替换prototype后，同一身份的独立噪声版本会互不匹配，导致重复分配/挤出冷记录。

这是同模型/同输入的机制压力，不证明所有架构或所有噪声都失败；匹配阈值未校准、identity语义由benchmark指定。拒绝/延迟不确定写入、保持锚定key或利用可观察连续性是后续候选，不能立刻改阈值在test上追分。

## 验证与边界

全probe保存prediction/target，summary逐条重新计数hot/cold错误类型和unknown计数；参数hash不变、query只读state。counters由实际匹配与新recency选择计算，owner不驱动更新。

新增两个确定性反例单测：cos.95不同key会被阈值合并；同身份两次cos.85偏移可误淘汰冷槽。连同动态分配/有限recency/非法输入和原fast数学契约，共9项通过，write_noise_tests.log。

文本只读复核只审查给定设计，不读取或执行实现，不冒称独立数据审计。持久状态fast512B、slots832B，不等资源；仍包含eval日志/标签等外部内存，无省算力/真实输入吞吐结论。

下一阶段应明确身份可观察依据，并以独立验证流评估何时允许写入、何时暂缓；仅在query端加UNKNOWN无法撤销先前错误覆盖。仍未接原PBSLM E/生成语义或生产Runtime。
