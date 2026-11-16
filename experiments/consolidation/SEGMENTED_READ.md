# 问题—记录角色分离：标准检索基线

## 为什么做

布局诊断显示原单实体表示在前置另一问句时会混淆目标，但输入本身也无目标指令。本批不让模型猜角色：当前query单独编码，每条record key独立编码，value为caller已观察的显式反馈code。

models/segmented_working_read.py实现标准cosine nearest/softmax attention，是外部工作记忆读接口，不是新的注意力算法或已完成A内化。底座/读出全部固定，无训练、无数值M、无Runtime接线。

## 设计

```bash
python -m evaluation.segmented_read_control
python -m unittest tests.test_segmented_working_read -v
```

seed0/1/2已有checkpoint。每support4/8/16各32episode；抽实体并重新随机赋8色，label可重复。record key使用support模板0/1；query为训练过query模板2/3或held4/5，每条record实体都查询。E-key与token均值使用相同输入/值、相同nearest/temperature.1 soft policy。

query与key矩阵只含原输入的冻结特征，不将entity ID/index或真答案交给read_segments。实体索引只在研究脚本查文本/特征、评分。value本身是显式feedback，接口没有从完整事实句抽取color。全部16名字已预配置，4名字未参与原episodic优化。

每个查询将所有record同样排列，另随机permutation keys+values，核对检索value一致。soft在数学上对记录排列不变；hard在score并列时argmax受顺序影响，新增tie单测明确这一限制。当前实际episode未触发改变value的tie；不能宣称所有输入hard检索顺序无关。

## support16/held 三seed均值

| feature/method | truth答案率 | top record选择率 | 错值组truth答案率 | 错值组跟随写入值 |
|---|---:|---:|---:|---:|
| E/nearest | 87.30% | 85.29% | 1.95% | 87.30% |
| E/soft | 87.70% | 85.29% | 1.95% | 87.43% |
| token mean/nearest | 51.56% | 45.25% | 6.38% | 51.56% |
| token mean/soft | 37.50% | 45.25% | 8.20% | 37.43% |

record accuracy用实际对应entity的记录index评分，不因label重复而算对；答案率可能高于record率，必须分开。soft的top_record仅由score argmax产生，与soft最终value不完全等价。

错值组每个color+1 mod8，同key保持，单记录错值必不同truth；检索错record仍可能偶然答对。此组显示系统会读出错误feedback，绝不验证真值。

## 对结构的意义

- 显式query/record分离避免了靠concat顺序猜目标，标准检索已是强基线，soft对nearest提升很小，没有新方法优势证据。
- E已经受实体跨模板对齐监督，token均值没有相同结构/额外计算；分数差不能全部归因角色分离。
- 当前任务没有同一record槽位重复改值、过期、时态或多来源冲突；相关策略不能靠attention自动解决。
- 不与前一concat实验直接作因果百分比差比较：value接口、随机映射/数据流均不同，当前没有同任务concat控制。
- 角色由caller明确分离，提取/确认/鉴真仍未自动完成；模型没有生成流畅自然语言。

## 证据与限制

segmented_read.json保存每episode完整entity/template/目标/写入值/prediction/top_record/permutation和所有support配置metric；模型hash/M零检查通过。soft/nearest数值与tie、非法输入单测独立。全仓库196 tests OK（9.221秒CPU），all_tests_segmented_read.log。运行后补充了hard并列顺序语义的docstring和单测，不修改历史JSON预测。

这是容量有限且有明确接口的检索组件，存储实际record特征和值，不能叫零额外记忆或知识已经进入A。没有同字节/时延基准、无学习型dependency/说听policy、无持续语义续连验收。

下一步应将有source/revision与可撤销记录的工作状态作为独立角色输入，模型只在语义目标可见的条件下检验迁移，再与普通检索比较；不继续用拼接两个含糊问句逼模型猜实验者目标。
