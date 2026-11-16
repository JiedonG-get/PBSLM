# 冻结提取器的独立表面校准：阈值不能保证自动写入安全

本批不训练、不更新PBSLM、不写M。使用three_seed600已保存reader；新增calibration和test句型均与原train/held精确模板不重叠，实体/颜色库存仍熟悉。

```bash
python -m evaluation.calibrate_fact_reader
python -m unittest tests.test_calibration_protocol -v
```

## 预定选择规则

在每seed 512条校准例（256正/256负）上网格：valid threshold .5/.7/.9/.95/.99/.995/.999/.9999，entity/color margin 0/.1/.2/.4/.6。只保留经验exact precision>=.99且接受至少20例的策略，选接受数最多者；若没有则reject-all（precision必须记None而非100%）。

随后冻结选择，在另512条独立测试表面上评估。没有用测试结果再调阈值；没有按结果挑seed。

这里是经验接受policy选择，不是概率校准或分布外安全保证。

## 实际结果

| seed | 选中threshold/margin | 校准已接受precision / n | 测试已接受precision / n | 测试负例误接受 |
|---|---|---|---|---:|
| 0 | .995 / .6 | 100% / 78 | 81.18% / 85 | 6.25% |
| 1 | .999 / .4 | 100% / 38 | 100% / 78 | 0% |
| 2 | .5 / .4 | 99.49% / 197 | 71.55% / 116 | 12.89% |

原policy .9/.2在同测试集的precision：52.67%、96.43%、90.43%。校准不保证每seed改善；seed2反而降低，反映句型分布差异和经验选择不稳定。

正例接受覆盖率（选中policy测试）：26.95%、30.47%、32.42%。即使seed1没有错误接受，也只有有限样本，不证明真实部署零风险。

## 结论

- 仅提高阈值或满足校准集99%precision，不足以安全自动写记忆。
- 不把reject-all记为完美模型，不只报告条件precision而隐藏coverage。
- reader有效分类是“单一明确实体/颜色断言”的任务定义，不等于现实真值验证。
- 未见否定、模态、时态句式仍可能被错误接受；默认保持外部确认。
- 这些模板由研究者人工编写，非真实语料、非预注册benchmark；3seed描述统计，无安全性/显著性保证。

完整网格、选择结果、全部calibration/test文本和预测分数在calibration_shift.json。脚本源码hash与运行环境有记录，旧three_seed600和held测试不覆盖。

后续优先：更多独立语义难例、单独验证集、风险-覆盖评估与确认负担；保留来源引用与可拒绝候选，而非静默写M。
