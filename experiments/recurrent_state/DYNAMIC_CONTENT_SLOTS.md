# 动态内容槽：保存键证据的强算法参考

新增DynamicContentSlots，不给entity ID/预定slot；通过cosine>=.9匹配已存key，否则分配首空槽、满时淘汰最久未写槽。命中完整替换key/value，保留原神经value embedding与MLP解码器但全部冻结。分配/匹配为标准算法，不是学习型动态寻址的新成果。

```bash
python -m evaluation.dynamic_slot_control
python -m evaluation.summarize_dynamic_slots
```

## 相同动态键任务

沿用continuous_key_control的随机种子与流程：每episode8个新16D键，每step随机键覆写新8类值，query75%来自8key池（可含未写键）、25%全新随机key。模型只收到当前key/value和固定slot tensor，不访问历史事件/真值表。评估96step×64流、3seed，无训练。

写key始终精确，noise仅加在query；没有测试写入噪声导致错误合并/分裂的情况。Unknown样本是球面随机/尚未写pool键，不是前批cos.99近身份负例；不能泛称未知实体识别已解决。

## 三seed均值

| capacity/query noise | 已观察身份最新值正确率 | 未观察身份拒答率 |
|---|---:|---:|
| 4/0 | 52.77% | 100% |
| 4/.05 | 52.77% | 100% |
| 4/.15 | 12.92% | 100% |
| 8/0 | 100% | 100% |
| 8/.05 | 100% | 100% |
| 8/.15 | 24.17% | 100% |

容量4被淘汰identity仍按任务的known计分，未用评价truth补答案，因此体现容量不足的损失。强query噪声已知正确低与固定阈值拒答有关，未记录每step误拒答/错值分解，不将全部错误严格归因为淘汰或阈值。

相同流下原key-value bank参考容量8也能完成无/轻噪声任务；当前方法应看作这个强baseline的有限tensor-state实现，而非击败检索。冻结神经reader正确解码直接存储value embedding是本次实测，不从理论保证任何reader都能做到。

## 状态与成本

每slot26float32：key16、value embedding8、occupancy1、write-recency rank1。容量4=416B/流，8=832B/流；对比delta矩阵512B只是激活payload，不是同资源/算力。保留实际连续key rows，不能称无key证据/全息压缩记忆。

write-recency使用0..capacity-1有界相对排序，不累计无限float时钟。Query不刷新recency（不是LRU读取策略）。整个read/assignment有全slot比较、Python batch循环，未做性能profile，不宣称省计算。

共享原fast938参数与value编码先验、模型运算、Python/RNG/eval缓存不计入以上payload。权重hash前后不变；episode仍存loss列表与评价truth/完整随机key池，所以整个研究进程不是常数内存系统。

## 验证与边界

7项定向测试（dynamic3+fast4）通过，dynamic_slot_tests.log：动态分配/覆写后其他value不改、更新recency淘汰、未知拒答、query只读、长反复更新rank上限、非法输入不改state，加既有delta数学契约。

summary核对保存n/correct比率，未存逐step预测，因此不是逐query独立审计。运行后补充finite/nonzero/shape/value合同，不改变计算规则或回写原结果。

文本设计复核未调用工具/读取代码/执行模型，只审查给定策略；不冒称独立代码或预测审计。当前没有真实语言、PBSLM E、Runtime或fallback整合，没有学习slot分配、source/撤销或持久恢复API。

## 研究判断

明确保存key证据与隔离value覆盖可以解决这个有限随机身份任务，但容量、相似身份与固定阈值仍制约行为。下一步应把‘压缩纯状态’与‘有限显式key状态’作为两条有成本账本的候选，而不是将显式存储的高分改名成神经泛化。真实文本key分布仍需独立对齐/有效性实验。
