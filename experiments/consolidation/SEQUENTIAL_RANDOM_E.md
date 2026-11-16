# 三批内化与随机E消融

## 设计

```bash
python -m evaluation.sequential_feature_consolidation --seeds 0 1 2 --steps 200
python -m evaluation.summarize_sequential_features
```

同一原始checkpoint，12个熟悉实体分为3批各4，三批使用同4颜色但不同排列。stage0为truthCE，stage1/2为独立原始冻结teacher的normal M soft KL。各stage200step/batch1，无回放，AdamW每stage重建；三arm相同targets/采样/读出初始化。

- trained_e：已学习并显式跨问法对齐的冻结E。
- random_e：只将ebank/e_mix替换为新随机初始化；tok/code/所有其他state逐tensor核对不改。teacher不随机化。
- token_mean：同冻结embedding均值复制两份，名义同4656readout参数；有效rank/计算不等。

teacher三seed三批train query均truth100%，完整soft分布缓存。student M-off且零，底座冻结；输出只有readout学习。

**random_e消融删掉所有E学习，不只删对齐目标**；token embedding仍是训练过的，不能叫完全随机模型。尚无相同offline训练预算、不用key-align目标的checkpoint对照。

## 结果

| 特征 | 最终三批train均值 | 最终三批held均值 | 最早批train：刚学完→最终 | 最早批held：刚学完→最终 |
|---|---:|---:|---:|---:|
| trained E | 97.22% | **79.17%** | 100%→91.67% | 100%→70.83% |
| random E | 80.56% | 50.00% | 91.67%→75.00% | 66.67%→41.67% |
| token mean | 100% | 65.28% | 100%→100% | 70.83%→70.83% |

trained E最终held各seed54.17%/100%/83.33%；token mean79.17%/54.17%/62.5%。**seed0是token mean更好**。E均值优势不能称跨seed稳定获胜，也不能称抗遗忘已解决。

trained E最早批held从100降至70.83%，下降29.17百分点；token mean该项均值没有下降，但初始更低。应同时看起点和终点，不能只比较遗忘百分点或只比较最终分。

每group/surface每seed仅8query，3seed描述统计；held只为同实体模板4/5（少量前缀变化），不是新实体/关系或广泛语言泛化。表中对group和seed等权平均，不作显著性。

summary从全部保存prediction/target重新计数并核对aggregate，包含完整stagewise矩阵和perseed结果。旧实验不覆盖、不回写source。

## 结论与下一步

已有E学习在此预算下有用，随机化后更差；但不能分辨贡献来自对齐目标、E结构或训练共同变化。多批后仍遗忘，故不把候选直接替换主模型。

下一项更有针对性的结构对照：冻结稳定表示，比较梯度readout慢更新与受约束的闭式/递归读出，检验是否能减少顺序梯度干扰，同时记录存储与计算代价；另需独立新问法/新实体评估。不以增加回放掩盖结构问题，也不把关闭M的readout背映射当通用知识内化。

## 验证边界

random_e单测确认仅E相关tensor改变、同seed复现；其余feature/soft目标测试共7项通过（sequence_ablation_tests.log）。全仓库182 tests OK（9.081秒CPU），all_tests_sequential_random_e.log。protected_hash验证named frozen参数，不是C输出/全部buffer的bitwise证据。

无生产Runtime权重切换、可撤销事实管理或长时自然语义状态。Online持续运行仍是目标，内化实验为支线机制筛查。
