# 有界可替换/撤销读出约束：临时状态候选

新增models/reversible_ridge.py：显式caller slot→feature/target矩阵，有界update-recency容量；同slot替换，删除后从保留约束重建统计并求解。全部重建成功后原子交换，失败保留旧状态。

它是**额外外部原约束记忆**，不是把经验全压入A权重，也不声称自动抽取slot/鉴真。重建成本O(保留约束数)与一次求解，没有使用rank-one inverse/downdate。

## 机制测试

```bash
python -m unittest tests.test_reversible_ridge tests.test_cumulative_ridge -v
python -m evaluation.reversible_ridge_control
```

- 替换结果等于latest-only联合重建，remove等于剩余约束重建。
- solve故障不消费version/slot；输入tensor复制隔离；容量超限淘汰update最旧slot。
- 无事件ID幂等、source序号、snapshot/crash持久化或线程事务；这里只是数值/状态原子参考。
- 全仓库186 tests OK（9.146秒CPU），all_tests_reversible_ridge.log。

## 实际冻结E纠正实验

3seed复用12个熟悉实体和原始checkpoint，保留train模板2/3的两个E特征，初始每个表面只加一次真实code约束（不是前一批200重复采样/soft目标）。更正前4实体颜色为旧4色循环旋转，各实体必改值。

append加入新约束但保留旧；replace替掉该slot两条旧约束。相同特征、直接已知label、lambda.01，CPU float64；不经过M teacher、不训练底座。

| 策略 | 改值实体train | 改值实体held | 未改实体train | 未改实体held |
|---|---:|---:|---:|---:|
| append旧+新约束 | 54.17% | 54.17% | 100% | 89.58% |
| replace旧约束 | 100% | 83.33% | 100% | 89.58% |

3seed小样本描述均值，不作显著性；同实体人工held模板，不是开放语义泛化。均值相同不表示未改实体每seed/每预测都不变：例如seed1 unchanged held append100%、replace93.75%。联合解会随约束改变，不能保证局部更新绝无其他行为变化。

替换后权重与latest-only oracle最大差<1e-7（代码断言）；随后删除前4实体，权重与仅后8实体独立重建同样一致。oracle是数值参考，不是实际系统获得隐藏truth。

## 成本与边界

- ridge统计/weight原为149,768bytes；另存每slot两条96维feature+48维target float64，共2,304bytes/slot，12slot为27,648bytes，不含Python/求解临时区/底座。
- JSON的replacement_storage_after_remove是删除后8slot实测payload，不能作为纠正时12slot的全部内存。初次运行包装层继承了内部raw_examples_stored=0字段，名称不适用于外部槽位；旧JSON保持原样，审查后storage删除该字段并显式报告retained_feature_target_rows。
- 审查后每slot新增默认最多8行限制，加上slot capacity与固定feature/target维度限制保留tensor payload；当前实验每slot2行不受影响。Python/瞬态输入复制/求解workspace不在这个上界声明内。
- 无始终A-only或零存储声明；明确保留原约束，错误确认会影响预测。
- capacity满会丢slot及其约束，影响对应能力；没有长期知识/已内化权重自动撤销。
- full rebuild适合小研究原型，不提供实时deadline或大型性能保证。
- 没有生产Runtime整合，固定底座/M-off/zero，语义与表达控制未验证。

这个结果支持“有界可撤销工作记忆可以减少旧值冲突”的窄机制，不证明一般泛化或完整M→A内化。下一步仍需同损失SGD对照与可靠性/来源策略，再讨论向长期A合并。
