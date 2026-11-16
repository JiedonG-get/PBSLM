# 共享答案类别与软M教师内化对照

## 与第一批相比改变什么

第一批旧4事实和新4事实连答案类别也互斥，遗忘可能被类别偏移放大。本批旧4实体/新4实体仍不重叠，但两组使用完全相同4颜色，只改变entity→color排列。**本批mapping与训练采样RNG也改为61000/62000/63000系，前批为51000/52000/53000系；所以跨文件变化不是只改答案重叠的严格配对消融，不能把改善全归因共享类别。当前文件内部六arm保持严格同序列。**

新增full46-vocab soft teacher：KL(teacher||student)，temperature=1作用在原cosine/.1 logits概率，不额外temperature搜索。teacher normal/zero/wrong三个M干预，全部先缓存detach概率。所有student都关闭M、数值M零。

## 固定设计

- 原persisted text checkpoint加载，teacher在旧事实学习前另复制且全部冻结；student旧阶段200step O1。
- 分支none/directCE/hard normal teacher/soft normal teacher/soft zero-M teacher/soft wrong-M teacher。
- 每学习分支200step、batch1、AdamW lr.002；相同seed下当前entity/template序列完全一致。
- 没有replay或新表面训练，不比较soft+replay；避免同时改变多个因素。
- 每个new entity两种训练query模板2/3，两种held4/5只测。每指标8query，seed0/1/2描述均值不作显著性。
- teacher normal/zero/wrong非M hash一致，forward前后M hash一致；完整概率、熵、top1、M hash均保存。
- student protected named params和M零断言；不是全部配置/buffer/C输出bitwise不变认证。

```bash
python -m evaluation.consolidation_soft_control --seeds 0 1 2 --base-steps 200 --steps 200 --output experiments/consolidation/shared_soft200.json
python -m evaluation.summarize_soft_consolidation
```

soft_control_tests.log包含14项协议测试（soft3、consolidation2、O1九项）：teacher不接收梯度、KL相同分布零、无效概率拒绝、student未按O1冻结拒绝，以及原M隔离/白名单测试。

## 三seed结果

| arm | new同表面 | new held | old同表面 | old held |
|---|---:|---:|---:|---:|
| none | 12.50% | 20.83% | 100% | 50% |
| direct/hard teacher | 100% | 45.83% | 58.33% | 29.17% |
| soft normal teacher | 100% | 45.83% | 54.17% | 33.33% |
| soft zero-M teacher | 8.33% | 12.50% | 16.67% | 12.50% |
| soft wrong-M teacher | 0% | 29.17% | 58.33% | 37.50% |

teacher正常M top1三seed均100%，zero-M均值8.33%，wrong-M均0%。正常teacher平均熵0.0451 nats，错M0.0443，zero-M1.4079；正常分布接近hard，不能据soft无held提升否定一切soft蒸馏。

shared_soft200_summary.json从每个保存prediction/target重新计数核对，包含每seed原数。没有用held选hyperparameter。old held最初仅50%，报告退化须以该实际基线为准。

这证明当前任务中memory teacher提供了有效监督，而非zero-M教师偏置；硬/软都能在M-free学生记住同表面。soft没有解决held泛化或稳定保持，当前O1与小模型表示可能是候选瓶颈，但尚不足以归因。与第一批跨文件不同RNG，不把old保持变化全归因答案类别共享。

## 结论的边界

hard teacher正确时与truthCE没有信息差，结果相同符合预期。soft若改善也只能说明当前目标/预算的经验信号，不证明更好架构或一般语言知识迁移。

共享答案仍不能解决少量样本/模板、checkpoint预训练范围窄的问题。held只代表同实体新精确模板，不是新实体/新关系/自然语言全面泛化。

teacher来自独立原始checkpoint，是稳定外部记忆teacher→A copy，不是运行中的同一个model自发整理。实验只用于筛机制，慢更新与Online Runtime参数版本切换尚未实现。

正常M比零M教师好支持memory contribution，但零M分布并非随机/均匀；错误M导致truth下降需同时检查是否跟随错误目标，不能把所有错误都归因内化。
