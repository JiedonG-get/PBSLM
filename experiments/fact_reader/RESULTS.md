# 事实候选提取器：从手工字段到陈述候选的第一步

新增独立监督GRU reader，从完整中文陈述预测16设备实体、8颜色、可提取/拒绝。**不是PBSLM主干的新能力或架构收益**；主干与reader参数独立。tokenizer使用仓库完整BPE ID，reader输出仍是固定实体/颜色分类，不是开放槽位发现。

## 协议与成本

- reader：embedding32、双向GRU hidden48，valid/entity/color三头；entity/color目标只在正例训练。
- 每seed600step、batch32、一半正例/一半负例；AdamW lr=.002。
- 正例4种训练句型、4种held句型；负例5种训练（提问/否定/猜测/未知/多事实）、6种held（新问法/不能确认/也许/或者/旧状态未知/假设）。
- 实体/颜色全都参与训练，held只指精确句型；不是新实体泛化。
- 接受策略预先固定：valid概率>=.9且实体/颜色top1-top2 margin均>=.2，未用held调阈值。概率未经校准。
- 每split512例（256正/256负），seed0/1/2。测试组合可能与训练同词汇/实体颜色配对，仅句型不同；seen组是容易的组合内测试。

```bash
python -m evaluation.train_fact_reader --seeds 0 1 2 --steps 600 --test-examples 512 --output-dir experiments/fact_reader/three_seed600
```

每seed约5秒GPU；模型文件本地忽略，results.json保存hash/参数数/环境/source fingerprint/全部测试文本与预测。

## 结果（3seed描述均值）

| split | 正例字段exact（不考虑拒绝） | 正例接受覆盖 | 已接受候选exact precision | 负例误接受 |
|---|---:|---:|---:|---:|
| seen模板 | 100% | 100% | 100% | 0% |
| held模板 | 82.03% | 31.64% | 81.95% | 3.78% |

held precision perseed：66.67%、80.22%、98.97%；对应错误接受28、18、1条。表中precision是三个seed的macro均值81.95%，不是合并225/272=82.72%的micro值。均值不代表稳定可靠，不能把最好的seed当整体能力。

exact precision定义：候选必须来自valid正例且实体与颜色都正确；负例即使字段恰好匹配提到的词也算错误。单测固定该定义。

## 为什么不能自动写M

未见否定/不确定形式仍可能被接受。低precision会把错误关联写入，并且前面M实验表明错误内容容易被忠实回忆。因此没有自动写入；reader只propose候选。

`runtime/statement_candidates.py` 把source文本/sha绑定候选：

1. propose记录source到Runtime输入，不写M；
2. reader不接受则不能直接确认，可显式reject；
3. 已接受仍需要调用方显式confirm；用预测实体产生canonical query捕获E key，再把预测颜色送入原有bounded反馈确认；
4. receipt记录source hash，protocol确认记录保存source文本；同candidate重复confirm不重复写；
5. 写入失败恢复临时协议/capture，不重入失败writer；候选可重试。

这只是提取建议，不是真实性验证。调用方确认错误候选依旧污染记忆，不能用“需要确认”掩盖reader误判。

## 实际演示（不训练）

```bash
python -m evaluation.statement_reader_demo
```

使用seed0已保存reader和已保存PBSLM文本模型。输入：甲红、乙蓝、甲不是绿、把甲更新为绿；演示的人工/脚本确认正句、拒绝否定句（**决策显式写在脚本里，不是自动鉴真**）。最终查询得到甲绿、乙蓝；参数哈希不变。完整来源文本、候选分数、人工决策、回执和答案见statement_demo.json。

## 限制

- 全为人工合成、单关系实体颜色任务；模板分类器可利用表面词；无真实语料/多关系/多语言。
- reader接受是未校准policy，不是真值判断；阈值高仍有假正例和coverage损失。
- reader本身是额外参数/监督/计算，不能把收益归到PBSLM并行块。
- canonical query由预测实体class映射，slot不是自由抽取；颜色class映射为已知label。
- statement候选历史暂时只在内存，未加入会话snapshot；原bounded会话确认记录/M可保存，但未确认reader候选不可恢复。
- source是调用方文本，不含鉴权/证据验证；没有多来源冲突裁决。旧候选可在新陈述之后被显式确认，采用latest-confirmed顺序，不承诺latest-arrived时间优先。
- 独立静态审查确认source SHA与原文本、指标计数一致；随后加固reader config/vocab、阈值/margin有限范围与运行logits有限性，当前权重/阈值未改变、不重训。
- seen演示结果不能替代held测试，也不是长期保持或在线参数学习。

下一步应优先收集更丰富、独立验证的否定/时态/不确定数据与校准集，明确人工确认工作量；在提取安全性足够之前，不自动写入。
