# 中文模板事实回忆：三 seed 配对结果

完整任务边界见PROTOCOL.md。数据使用真实中文问句+BPE，但模型是46-token紧凑子词表、d48的新训练原型，不是通用中文语言模型。

## 运行与配对

```bash
python -m evaluation.run_text_fact_pair
python -m evaluation.summarize_text_facts
```

paired800/baseline.json与key_aligned.json：seed0/1/2，每seed800episode，support4，eval每split32episode×4target。两组同seed/同episode映射与模板抽样、相同原始初始化；区别是key_aligned增加0.2×实体身份跨模板对齐目标。

每个episode的4实体与4不同颜色随机绑定；支持问句和答案先公开写M，随后换问法查询；eval不更新参数。每个episode/干预下C bitwise不变，非M state hash不变均通过。

## 三seed均值

| 实体与问句 | baseline 正常M | key_aligned 正常M | key_aligned 零M | key_aligned 错值M真实答案 |
|---|---:|---:|---:|---:|
| 熟悉实体/支持原问句 | 25.00% | 100% | 13.54% | 0% |
| 熟悉实体/训练过查询模板 | 25.00% | 100% | 12.76% | 0% |
| 熟悉实体/未见精确模板 | 25.00% | 99.22% | 15.63% | 0% |
| 新实体/支持原问句 | 24.48% | 99.22% | 13.54% | 0% |
| 新实体/训练过查询模板 | 25.52% | 99.22% | 11.98% | 0% |
| 新实体/未见精确模板 | 23.96% | 62.76% | 12.76% | 11.98% |

熟悉实体+未见模板，key_aligned三个seed：98.44%、100%、99.22%。
新实体+未见模板：53.91%、69.53%、64.84%，明显较低且有seed差异。

每项每seed128查询，来自32个episode；episode内相关，3seed只作描述统计，不作显著性宣称。

## 诊断结果

baseline熟悉实体/训练查询平均每episode只输出1.14种答案（应有4种颜色）；未见模板约1.13种。正常M与错值M的真实答案率均约25%，与“只从支持颜色集合选一项、没有正确绑定实体”相容。

key_aligned熟悉实体/训练查询平均4种答案，未见模板约3.97种；新实体/未见模板仅2.52种。额外实体对齐有助于键区分，但难的组合仍存在绑定/迁移失败。

错值M使熟悉实体读出转向写入的错误关联而非真实答案，支持“具体关联依赖”；也说明没有事实校验，错误写入仍然危险。

## 解释必须受限

- 成功依赖额外实体身份监督；训练知道support/query哪两行是同一实体。不能称无监督语义学习。
- key_aligned增加E前向、反向和对比目标，只有episode预算匹配，不是等FLOPs优势。baseline总201.15秒，key_aligned240.75秒（含评估），约20%时间增加。
- held模板主要是前缀/措辞变化，不是丰富新语法。新实体只有4个，eval每轮均包含它们。
- 词表和padding长度预先包含eval-only表面形式；这是transductive输入库存准备，未泄漏随机答案，但不能声称开放词表泛化。
- support答案以label由API提供，不是从完整陈述句抽取事实。所有实体名的输入编码distinct单测已通过。
- 无M是同一已训练模型的干预，不是独立无记忆训练baseline。C没有监督这个随机任务，低C分数不是A/C竞争结论。
- 显式实体抽取+字典可以解此任务；没有同预算字典/Transformer baseline，不证明PBSLM架构更优。
- 没有训练权重文件，只保留可重跑源码、配置、hash、曲线和逐episode预测。早期pilot字段命名不严谨已在PROTOCOL说明，不回写历史JSON。

## 可复核证据

- paired800/summary.json：从全部逐episodeA预测重新计数，并核对原metrics正确数/n一致；包含每seed结果与预测unique统计。
- paired800/*.json：原始映射、模板索引、答案/预测ID、BPE原始词表映射、源码hash、环境、dirty状态。
- paired800/*.log：原始子进程输出；runner.json包含命令/退出码。
- pilot.json与key_align_pilot.json：最早单seed预实验，独立保留，不覆盖。
- all_tests.log：最终测试日志，包含有界记忆故障修复后的全仓库测试。

## 对最终Online目标的意义

从重复token到文本问句后，原关联学习出现了明确失败。显式键对齐能修复本小任务的一部分，并呈现新实体×新问法的泛化落差。下一步应把该读通路接入有界反馈纠正会话，保存可加载checkpoint，再测持续文本流、旧事实保持和真实释义变化。当前还不是自然语言持续学习系统验收。
