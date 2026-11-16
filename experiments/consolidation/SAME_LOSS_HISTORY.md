# 同平方误差目标：历史约束与求解器分离

## 预定对照

```bash
python -m evaluation.ridge_same_loss_control
python -m unittest tests.test_ridge_iterative tests.test_cumulative_ridge -v
```

本批仅用冻结trained E特征，复用上一三批任务的checkpoint/标签/teacher概率/每stage200次采样顺序；CPU float64、相同λ=.01、bias regularized、目标为sum平方误差到code向量（stage0真值code，stage1/2教师期望code）。原A/M不改，不调用Runtime。

1. cumulative_direct：保留三批全部Gram/cross，直接解正规方程。
2. cumulative_CG：同一累计正规方程，共轭梯度迭代，warm start上一批解；并非minibatch SGD。
3. current_only_direct：每stage丢旧统计，只解本批约束；这是刻意移除历史约束的控制，不是最佳参数保持算法。

区别一对一：前两者只有求解方式不同；1与3保留的目标约束集合不同（都为SSE+同λ，但data总数不同，所以相对正则强度也不同）。不能声称是纯λ匹配、同FLOPs或同保留内存实验。

## 三seed最终三批均值

| arm | train | held |
|---|---:|---:|
| cumulative_direct | 100% | 91.67% |
| cumulative_CG | 100% | 91.67% |
| current_only_direct | 50.00% | 48.61% |

累计两solver每条query预测均一致；CG各stage12–52次迭代、最大相对residual约9.87e-10；全stage最大权重绝对差约1.53e-6，均低于预定1e-5。接近病态系统时低residual不意味着权重机器精度相同，故同时报告差值/预测。

current_only最后一批train/held均100%，但前批明显退化。例如seed0最早批train/held0%/0%。这显示当前批可拟合不代表历史能力保持。

## 能说什么

- 在这组固定特征/同损失问题中，直接与足够收敛的迭代求解没有表现差异。
- 保留累计历史约束对当前观察到的保持收益重要，不应将其包装为闭式方法凭空获得的能力。
- 尚未做同损失小批量SGD、相同存储预算/压缩约束对照；CG全历史统计不能替代该项，也不能证明原O1更新策略无问题。
- 两solver不是新神经结构泛化证据；trained E预先显式对齐，held仍只有同实体两问法，小n、三seed仅描述。
- 累计约束也可保持错误/过时记录，因此仍需前批可替换/撤销工作状态，不应无限无筛选积累。

## 实现验证

CG与独立torch.linalg.solve的随机正定问题比较、零rhs、非法shape/symmetry/λ输入测试通过；实验每stage收敛和weight差检查、底座参数hash/M零检查，完整预测/目标/目标函数值/残差见same_loss_history200.json。全仓库188 tests OK（9.258秒CPU），all_tests_same_loss_history.log。

反例界限：current_only是regularized minimum-norm solution，不做从旧权重的小步信赖域或锚定；它退化不等于任何不存原数据的系统必然遗忘。

下一步回到临时状态—慢内化的职责分离：先明确哪些约束可靠且仍有效，保留多少、如何压缩与撤销，再做受控慢更新；不把统计记忆重新命名为“已经写入A”。
