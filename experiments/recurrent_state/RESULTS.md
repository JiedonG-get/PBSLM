# 无原始历史访问的学习型递归工作状态：GRU基线

用户提出优先依赖模型自身状态、必要时再查历史。本批是独立基线，未修改PBSLM主干、不删除历史研究记录、不实现历史fallback。

## 实验

```bash
python -m evaluation.recurrent_state_control --seeds 0 1 2 --updates 600 --width 96 --output experiments/recurrent_state/pilot600.json
```

8个显式entity ID、8个value ID。每步随机写一个entity的新value，然后在已见entity中均匀随机查询最新value（第一步必查刚写入者；查询选择利用已见集合，但不利用value）。模型只接收当前写入(entity,value)、上一步96维状态h，以及query entity；不接收truth表、旧输入或查询答案。query不更新h。全部参数offline训练，600update×batch32×24步BPTT；eval固定参数、每长度64流。长384步仍是8槽位反复覆盖，不是384条独立事实容量。

这只是符号关联/覆盖任务，不是语言模型、全息架构、事实提取或持续理解。输出query采样限定已见entity，是benchmark接口辅助；不评估未知entity拒答。

## 三seed均值

| 流长度 | 仅h查询 | 查询时将h置零 |
|---:|---:|---:|
| 24（训练长度） | 65.32% | 11.20% |
| 96 | 56.71% | 12.94% |
| 384 | 54.68% | 12.75% |

reset组运行相同写入/查询随机流，每次query仅把其读入h设零，不清掉持续更新的原state，也不改权重。随机8类参考12.5%。不作显著性推断，不能把多个相关step当独立样本。

384步正常组按距最后写入年龄pooled：0步76.19%、1–3步64.56%、4–15步47.86%、16步以上31.72%。最近值也没可靠读对，不能把全部失败归因长期遗忘。

## 内存与计算

每流h=96float32=384bytes，只描述工作状态tensor；模型参数、单次运算中间量、batch、评估truth/last/metrics不包含。训练BPTT保留24步计算图，不是训练常数内存。

评估脚本为诊断在列表中积累各步loss tensor，无梯度；所以整个评估过程不是严格常数内存实现，只有模型查询接口无历史访问。truth和last表只用于选已见query和评分，不传模型。

训练＋评估每seed约20秒GPU，非Online推理延迟。模型54,824参数，不包含在384bytes状态数内。结果记录source/env、weights hash与参数不变检查，但没有保存训练权重文件或逐事件预测（仅aggregate/age桶），未来集成需新增可加载checkpoint而非伪造本次artifact。

## 结论

状态承载信息有证据，清状态后接近随机；但单GRU全局状态在本预算下不足以可靠维护8个动态关联，不能取消历史fallback。也不能据单一预算/任务否定所有recurrent/状态架构。

下一结构候选应比较同预算全局h、分槽/低秩状态与明确overwrite门控，并测试更长流、写入年龄、错误覆盖及自信错误。历史回退策略必须独立测精度/调用成本，不以未经校准置信度宣称省算力。

两项实现测试通过：固定state形状/read-only query、update/read梯度。基线不写生产M、不更新live权重、不改变已有检索原型。
