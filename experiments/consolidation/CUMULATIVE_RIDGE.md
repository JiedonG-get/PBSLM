# 累计充分统计量读出：保留约束不等于泛化

本批为隔离研究参考，不修改原A、M、checkpoint或Runtime。新增models/cumulative_ridge.py：存Gram/cross统计量，在显式stage边界求解岭回归，不是每token低延迟递归逆矩阵更新。

## 设计与可比性

```bash
python -m evaluation.cumulative_ridge_control
python -m evaluation.summarize_ridge_control
```

复用sequence_random_e200中全部实体/label/teacher概率、checkpoint与随机E seed；输入采样顺序/200step每stage相同，共600次约束（每批4实体×2模板，反复抽样）。底座M关闭且零，特征冻结。缓存特征不反复跑E，CPU float64累计统计与求解。此前梯度特征在可用CUDA上计算、本次在CPU计算，是同函数/权重而非特征tensor bitwise一致的对照。

目标为sum||XW-Y||²+.01||W||²（bias也regularize）。第一批Y=真实code向量，后两批Y=teacher概率对code的期望。预测归一化后选cosine最近码。

**与梯度读出不是只改更新算法的消融**：此前为cosine logits CE/KL，此处为code向量平方误差；额外累计历史统计，CPU float64，求解/计算/存储不同。有效目标/正则共同变化，不能把收益纯归因闭式更新。

## 三seed描述均值

| 特征 | 累计ridge最终train | 累计ridge最终held | 梯度参考最终held |
|---|---:|---:|---:|
| trained E | 100% | **91.67%** | 79.17% |
| fresh random E | 100% | 34.72% | 50.00% |
| token mean | 100% | 56.94% | 65.28% |

trained E最早批held100%→87.5%，仍有12.5百分点下降，不是无遗忘；random E最早held70.83%→33.33%，token75%→54.17%。所有train保住不意味着held不变，更不是语言泛化解决。

同一批约束逆序累计后的最大weight绝对差约1.1e-11/9.7e-11/5.2e-11，确认该有限样本/float64下求解对约束顺序基本不敏感。改变stage先后并在每stage测成绩未做；不据此声称所有数据流无顺序影响。

每指标8query/group，3seed，只作描述。held同熟悉实体两种模板，不是新实体/新关系。

## 存储账本

每候选D=96、输出48，加bias97：

- Gram：75,272bytes；cross：37,248bytes；weight：37,248bytes；合计149,768bytes（float64），不含底座/码本/Python/求解临时区。
- 梯度readout参数4,656个float32=18,624bytes，仅权重；其训练另有梯度/Adam状态，不能拿此数字当全部训练内存。
- ridge模块不保存原样本；但**本实验为了逆序验证额外缓存了600条x/y约束，并缓存全部train/test特征**。raw_examples_stored=0仅描述模块，不描述整条研究脚本总内存。
- 结果字段statistics_fit_seconds_excluding_feature_cache实际包含stage评估、逆序重建和hash检查，应读作“排除特征缓存的实验耗时”，并非纯fit；不当公平速度benchmark。旧JSON字段保留不回写。

## 验证

单测增量统计与联合正规方程解一致、逆序预测一致、非法更新不改变统计；底座state hash/M断言和原feature梯度隔离回归通过。全仓库184 tests OK（9.518秒CPU），all_tests_cumulative_ridge.log。逐query重新计算汇总核对准确率，原预测/teacher来源SHA/source环境保留。

## 结构判断

- 累计约束能缓解本任务的训练映射丢失，但已学习特征仍决定held表现；随机特征train100%而held低，是背住约束不等于迁移的直接例子。
- trained E+ridge是可继续验证的候选，不是原A内化实现或结构优势定论。
- 当前统计不可删除/替换，同slot事实纠正会让旧新约束冲突。需要可撤销/加权/来源筛选的工作状态，不能无限累计所有经验。
- 必须增加同平方误差SGD、存储匹配、独立新问法和简单检索对照，再决定主模型整合。
- 无持续语义续连或自动说听成绩；慢更新边界与权重版本切换仍未接入live Runtime。
