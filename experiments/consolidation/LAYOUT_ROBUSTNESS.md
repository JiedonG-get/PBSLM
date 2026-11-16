# 相同实体/标签的布局配对诊断

## 目的与限制

上批128随机pair同时增加容量/组合/长度/新名字。本批固定16个实体与每seed同一组标签，取消记忆预算限制，分别观察布局变化。只是冻结E表示/读出诊断，不是模型理解指令或一般语言能力。

四布局：original原问句；repeat同问句重复；prefix前放另一实体问句；suffix后放另一实体问句。额外实体=(target+1)%16，在每目标的train/held表面间固定，**不是所有目标共享一个固定实体**。

每实体train模板2/3、held4/5。prefix/suffix里两个句子都是真实问句；模型没有收到角色/目标指令。评分只按实验者指定的原实体label，不代表自然语言里第二/第一问必须被忽略。重复会同时改变长度/重数，不能当纯长度消融。

## 两次运行分别保留

- layout_robustness16.json：每layout重新拟合ridge/填nearest库，只说明各layout自身可被读出。
- layout_robustness16_transfer.json：同样保存refit，并增加original上拟合的两种读出直接测其他layout，无refit。此前JSON不改。

```bash
python -m evaluation.layout_robustness_control --output experiments/consolidation/layout_robustness16_transfer.json
```

无底座梯度/M写入；ridge新拟合是闭式统计回归，nearest保存全部32train rows，无容量限制。标签各layout完全同一映射。

## 三seed held均值

| layout | refit ridge | refit nearest | original ridge不refit | original nearest不refit |
|---|---:|---:|---:|---:|
| original | 80.21% | 84.38% | 80.21% | 84.38% |
| repeat | 82.29% | 83.33% | 81.25% | 86.46% |
| prefix | 83.33% | 81.25% | 11.46% | 12.50% |
| suffix | 73.96% | 86.46% | 70.83% | 79.17% |

不refit的prefix预测匹配额外前置实体label的比例：ridge78.13%、nearest81.25%（事后混淆诊断，不是预注册模型质量指标）。原标签可能与额外实体标签相同，因此该比例与target正确率不是互斥分解。

## 解释

- 重复加长本身没有必然导致失败；至少本小条件下，不能用“长度翻倍”单独解释上批下降。
- refit能够恢复prefix读出，不代表原始表示/原任务decoder能直接适应新上下文。额外实体=(target+1)%16是固定一一映射，refit可能只利用额外实体身份预测target标签；因此高refit分数不能证明模型忽略背景或正确定位目标。后续需改变distractor配对、让目标身份独立可见。
- prefix/suffix不对称与“前置实体主导原单实体读出”相容，但当前未隔离位置编码、causal attention、均值池化、训练输入分布等原因。
- 最重要的是目标未显式标记：把历史文本/已说前缀直接拼接，不能保证神经模型知道哪段是当前任务、哪段仅背景。不能期待工作记忆容量独自解决角色/相关性。
- 所有实体仅16个（12熟悉+4原优化未见），held是有限表面；3seed描述均值，不作显著性/通用语言结论。familiar/untrained分组只按target名字，不能当整个输入都熟悉或不熟悉：例如target11搭配distractor12，target15搭配distractor0。

## 验证

layout token identity测试确认不截断，prefix/suffix与预设精确顺序相符；feature gradient与reservoir契约共5项定向回归通过，layout_robustness_tests.log。全仓库193 tests OK（9.543秒CPU），all_tests_layout_robustness.log。

每layout/方法/split下all/familiar/untrained_names的完整prediction/target、feature相似度、storage、checkpoint hash保留。首次与第二次源码不同（新增no-refit路径），来源各自捕获，不伪造原运行包含新测量。

## 下一步结构判断

应在连续输入主线中给事件/说话状态与当前query明确接口和训练角色，比较标记/分段读出与无标记拼接；不能从此实验直接断定需特定架构。先让实验中“谁是目标”对模型可见，并保持预算/目标配对，再验证泛化及当前输入是否影响真实生成。

这是从小容量记忆转向连续上下文结构的诊断。没有音视频、自然续连或自动relevance安全结论。
