# 下一项匹配训练：选择性写入与碰撞约束

这是预先设计，不是已经执行的成绩；不修改原slots600权重。

## 问题

测试时top1抑制旁路写可保住非碰撞冷记录，但无法解决共享top槽，并可损害依赖soft多槽编码的热ID。因此需要从训练开始匹配写规则，而非只在测试时硬化。

## 预定arms

- soft：复用原learned模型正常训练，作配对baseline。
- sparse：训练前向top1写入，反向straight-through soft gate；原soft query保留，避免同时修改读写两处。
- sparse+collision：同sparse，再加入固定系数的route矩阵行间重叠惩罚（有限8ID、8slot先验）。
- explicit：onehot ID寻址强辅助参考，不作学会动态分配的证据。

不选后验成功seed，统一0/1/2，600updates×32×24，任务生成与旧实验同流；同96float状态。系数事先记录，不用all-hot test优化。

## 必须单独报告

uniform24/96/384，固定all-hot2048的hot/cold与共享top槽子集、route top唯一数量、熵、每seed结果。训练难度与test-time门控变化必须分开。

ST梯度是代理，不是离散选择的真实导数；碰撞惩罚使用全部已知entity集合，属于明确结构/监督先验，不证明动态未知实体。参数和FLOPs另报；所有slot candidate仍计算，不宣称省算力。

## 决策规则

如果只有collision loss下降而新事实读出差，不接受；如果均匀流提高但hot流差，不接受长期保持结论；如果只对已知8ID有效，停留在机制层，后续必须接连续特征query/未知身份任务。

本阶段目标是减少固定状态干扰，并不承诺代替完整语言模型或完全删去历史。原始历史研究/运行日志不删除。
