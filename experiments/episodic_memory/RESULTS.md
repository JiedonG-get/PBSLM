# Episodic M：随机映射的因果记忆读出

日期：2026-11-15。用户授权后的独立合成研究实验，原始 Phase1/O1 结果不覆盖。

## 为什么做这项实验

上一项 continual_associations 中 M 为零进行基础训练，基础训练不能教会“有用的记忆读出”。因此其中只写M无收益既是负结果，也留下训练分布缺口。本实验直接检验：**如果离线阶段学会从记忆读随机关联，评估时只写M、不改权重，是否能回忆新的关联？**

这不是完整Online Learning，而是关联记忆机制测试。

## 协议

- 模型：真实 PBSLMv2 masked independent routing；vocab48、d32、k8、depth1、heads4、E2、refine1、Flow开启、field_scale=0、固定码本。
- 每个训练episode重新随机抽4个不同key（16个候选）和4个不同label（8个候选），映射每轮改变。
- 先观察support的key与label，然后将E特征/label向量写入M；查询同一key。**查询的答案先前已在support公开**，这是合法的support/query回忆任务，不是未见信息推理。
- 离线训练500个episode，AdamW lr=.002，优化A查询分类目标；没有强制C学随机答案。
- 每episode先清空M；写入使用现有no_grad外积机制，不通过写入反向传播，E可以通过读出获得梯度。
- 评估全新episode随机种子，64 episodes，正常M/零M/循环错值M三组。同一episode的key/query不变。
- 评估期间不执行任何参数更新，检查非M参数哈希，且C输出在三种M干预下bitwise不变。
- 训练support数4，额外评估support数2和8。key与label词表仍是熟悉集合，只有映射新鲜；不是未见词泛化。
- 小词表精确cosine CE、tau=.1，与Phase1 sampled-negative InfoNCE不同。

运行：

```bash
python -m evaluation.episodic_memory --seeds 0 1 2 --episodes 500 --eval-episodes 64 --output experiments/episodic_memory/three_seed.json
python -m unittest tests.test_episodic_protocol -v
```

原始逐seed结果、训练曲线与参数哈希：three_seed.json。pilot.json为更早的单seed4-support试验，未覆盖。

## 结果：三seed均值

| support数 | 正常M正确率 | 零M正确率 | 错值M对真实答案正确率 | 错值M匹配写入错值 |
|---|---:|---:|---:|---:|
| 2 | 100% | 13.80% | 0% | 100% |
| 4 | 100% | 10.42% | 0.13% | 99.87% |
| 8 | 99.87% | 13.09% | 0.07% | 99.67% |

每seed、每support大小、每M干预均通过评估参数哈希不变与C bitwise不变。

总耗时130.94秒（训练每seed约25–26秒，余下为评估）；单卡RTX4060 Laptop 8GB、torch2.11.0+cu128、fp32。3个协议单测通过，0.140秒CPU。

## 可以说什么

- 现有E/M读写机制在小型随机关联回忆任务上能产生明确、可干预的收益，评估不需要参数更新。
- 正常M大幅好于零M；循环错值后输出几乎跟随错值，支持“读了具体关联”而非仅有范数变化。
- **错值结果同时证明污染风险**：当前机制没有事实校验能力，写错就可能稳定读错。
- 与上一实验对照提示：M是否被训练成可用读通路很重要。但两实验任务、目标不同，不能把差异归因于唯一因素。

## 不能说什么

- 这不是自然语言事实学习、主题泛化、长程记忆或无监督世界学习。
- M每episode清空；2/4/8支持集只是短期关联负载，不是长期容量/遗忘曲线。
- 训练有记忆而评估无记忆存在分布变化，零M组不是同预算独立训练的架构baseline。
- C在评估M干预下不变，不代表C在离线训练中不变：共享底座在离线阶段参与更新。
- 没有普通attention检索、字典或单路径模型的等预算对照，不能声称PBSLM的记忆更优。
- 没有保存训练checkpoint，只保存可重跑脚本、配置、曲线、哈希、结果；精确GPU重跑可能有数值非确定性。
- 先看support答案再写入是任务定义；它不能用于证明模型从没有反馈的预测中学到了真值。

## 下一步

将训练好的读通路用于持续多episode流：有界写入、旧关联保持、冲突纠正、正常/零/错值干预和键相关性。同步推进模型无关Runtime状态机，确认commit不等于记忆确认；不将Runtime运行等同于学习成功。
