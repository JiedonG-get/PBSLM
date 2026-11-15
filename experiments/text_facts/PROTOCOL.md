# 中文模板事实回忆：协议与适用范围

## 问题

从纯token重复的关联任务走向文本表面变化：支持记录用一种问法，查询换问法，模型是否仍能把实体与当次随机颜色绑定？

这仍是受控模板关联回忆，不是自然语言事实抽取或开放域问答。支持答案以独立label由接口公开，并非从完整陈述句自动读出。

## 数据与分割

- 仓库BPE tokenizer编码16个设备名、6种问句、8种单token颜色。
- 模型使用46个实际涉及BPE token的紧凑ID词表和新初始化embedding，不是16k-vocab预训练语言模型。
- 词表与固定padding长度从全部表面形式（包括held/novel）预先构建，这是transductive输入库存预配置，不是答案映射泄漏；必须披露。
- 熟悉实体前12个参与训练，新实体后4个不参与优化episode；每个新实体eval episode都包含这4个名字。
- support模板0/1，训练query模板2/3，held模板4/5只用于eval。
- held模板主要增加前缀/措辞，属于窄模板迁移，不代表广泛新语法/语义泛化。
- 每episode从8颜色抽4个不同值随机绑定到4实体；映射每轮改变，不可单靠固定实体→颜色表背答案。
- 所有输入左侧用BPE id0填充到同长度，无截断；该人工padding也限制自然语言解释。

## 训练对照

- 原目标：仅A输出颜色分类，精确cosine CE/tau.1。
- key_aligned：原目标+0.2×实体身份对齐CE；每个E块的support/query均值key归一化，相同实体跨模板是正例，另外3实体为负例。
- 实体对齐是显式额外监督和额外E前向，不是免费架构收益/等FLOPs对照。
- key对齐不用颜色label，但训练知道support/query中哪两行对应同一实体。不能称为无监督语义学习。
- 两组各800episode，seed0/1/2，随机mapping/模板流相同。相同episode数不代表相同计算成本。

## 评估

每seed每split32episode×4target。分别报告 familiar/novel × same_surface/seen_query_template/held_template × normal/zero/wrong M。

- 正常M：将已观察support的E特征和颜色写入。
- 零M：相同权重/输入，清空M；不是独立训练的无记忆baseline。
- 错值M：label循环移位，key保持；检查真实答案与写入错值匹配率。
- eval无参数更新，hash所有非M state_dict；检查C在M干预下bitwise不变。
- 同episode预测unique数量用于检查是否忽略实体而输出同一颜色。
- 完整逐episode输入元数据、target/predicted IDs、原始文本BPE映射与分割库存保留于JSON。

`zero`组的written_match_rate实际是与真实label匹配，没有写入值，不能解释为记忆读出。旧pilot中的novel_entity_tokens_unseen_in_training字段只是输入库存差，含held模板token，不能推断每个embedding实际是否训练；paired800版本改为明确token_inventory_evidence。

## 参考命令

```bash
python -m evaluation.run_text_fact_pair
python -m evaluation.summarize_text_facts
python -m unittest tests.test_text_fact_protocol -v
```

runner要求新目录，防止覆盖；原pilot/key_align_pilot保留。实验运行时源码hash捕获，不向旧结果补造provenance。

## 验收标准

支持机制的窄证据需要：正常M好于零M，错值M显著改变关联读出，熟悉/新实体与seen/held分别报告，参数和C保持不变。一般语言能力、开放事实学习、长期保持与并发Runtime均不在这项任务的结论范围。

即使高分，简单已知实体抽取+字典lookup可以完成这个任务；本实验没有证明神经矩阵检索优于普通检索baseline。
