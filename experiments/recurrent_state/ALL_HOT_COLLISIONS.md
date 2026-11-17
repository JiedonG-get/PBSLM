# 全部热实体：稀疏写入不能替代无冲突寻址

本批固定slots600三个learned checkpoint，无训练。每seed遍历8个hotID，共24种场景；每种16条流、2048次随机热值覆盖。初始按0..7原soft规则写入8值，复制为soft/top1_mass两arm，读规则不变。

```bash
python -m evaluation.all_hot_gate_control --steps 2048 --batch 16 --output experiments/recurrent_state/all_hot2048.json
python -m evaluation.summarize_all_hot
```

不同hotID用seed+131000+hot*100的预定独立流，因此同case两arm严格配对，不是不同hot共享相同具体value序列；也不与上一hot0/8192实验当作同序列时长消融。

## 结果

| 指标 | soft | top1_mass |
|---|---:|---:|
| 初始冷entity均值 | 99.07% | 99.07% |
| 最终冷entity均值 | 45.76% | 98.14% |
| 最终热entity均值 | 99.48% | 98.18% |
| 与hot不共享top槽的冷entity最终 | 46.08% | 99.06% |
| 与hot共享top槽的冷entity最终 | 18.75% | 21.88% |

“非碰撞”指实体top-slot与当前hot不同，不能保证soft路由没有重叠。碰撞子集仅32个最终query，来自seed1中hot3/冷7与hot7/冷3，各16流；不是独立模型复现。其初始正确率96.875%，所以最终低不是仅初始没学会。

seed1 hot7自身最终soft87.5%、top1_mass56.25%；稀疏化可以损害依赖分布式路由的实体。总体98%不能遮住该失败。

## 判断

稀疏写有助保住不共享主要槽位的记录，但相同slot容量竞争仍存在，不能直接把top1干预替换成最终机制。下一结构必须同时约束写入隔离与寻址冲突，或允许明确多槽分配/受控fallback；不能靠增加阈值/只挑热ID成功解决。

## 验证与限制

原JSON保存每case每arm检查点0/1/32/128/512/2048的逐流8个prediction/target、perentity计数和top-slot映射。summary逐预测重算并核对hot/cold计数、同initial probes。独立只读审查重算全部记录，零不一致，两个碰撞case的32个最终query分母也核对一致。参数hash前后不变；加载原文件SHA记录，没有查询原始历史。5项加载/路由/写隔离定向测试通过（0.196秒），all_hot_contract_tests.log。本轮只补评估与报告，未改变模型，不重复训练或扩大全套测试。

固定已知8ID，静态寻址、软priming、原soft reader，test-time门控变化；非语言、非动态实体、无PBSLM Runtime整合、无计算节省证据。仍计算所有slot candidate。无新训练或下载。

下一匹配训练实验应预先规定一一寻址约束/碰撞惩罚、同state大小与训练流，保留不加约束及显式寻址基线；检查uniform与all-hot两类结果，不按test碰撞选择seed。
