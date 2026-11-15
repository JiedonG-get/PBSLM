# EXPERIMENT_PLAN — Phase 0 → Phase 2

## 1. 原则

1. **先修训练闭环，再做模型消融**；
2. 每个组件单独加入，同数据、同训练 token 数、同步数、同 held、同评估；
3. 另设参数量/FLOPs 匹配的预算对照，避免把「更多计算」误认为「机制有效」；
4. 不只看 held top-1；重点看低频 token、上下文敏感性、困难负样本；
5. 任何组件如果打崩学习，先单独修它，不继续加下一个。

## 2. Phase 0：训练闭环整理（已完成）

- checkpoint：model + optimizer + step + config + CPU/CUDA RNG + data/neg generator；
- resume 跳过 field 初始化；
- 拒绝不兼容 config；
- 固定码本 seed/hash/冻结；
- 评估补齐 top1/top5、频率分组、上下文交换、困难负样本；
- 代码：`training/train_v21.py`；报告：`experiments/phase0/PHASE0_REPORT.md`。

## 3. Phase 0.5：smoke + preflight（已完成）

- A1/A2 各 30 步 smoke；
- A1 resume 30→50 步；
- batch=8 preflight：30 步，记录显存/耗时/checkpoint；
- 报告：`experiments/phase05/PREFLIGHT_RESULTS.md`、`PHASE05_RESULTS.md`。

结论：流程通过，但 30–50 步不足以判断模型能力；smoke 中上下文变化为 0、低频为 0，
必须在正式训练中持续观察。

## 4. Phase 1：Baseline A（已完成）

严格定义：

- 单路径自回归、连续向量输出；
- 无 A/C、无 InfoFlow、无 E/M、无 PerturbField、无 Shock/Activity；
- A1 独立可训练码本；A2 固定随机单位码本；
- 同数据、同步数、同评估。

阶段预算：15,000 步 × batch 8 × AR 8 = 960k next-token 样本。

结果：`experiments/phase1/PHASE1_RESULTS.md`。
关键观察：

- 上下文敏感性出现（ctx_change ~0.55–0.60）；
- 高频组 top1 ~0.59，低频/稀有仍 0；
- 受控替换保留最后 token 几乎不降准确率 → 仍偏局部 bigram；
- 单 seed 下 A2 略好，不能宣布架构优劣。

## 5. Phase 2：模块消融（待批准）

| 组 | 增加内容 | 模块消融 | 预算对照 |
|---|---|---|---|
| B | A/C 双路径 | 同 A1/A2 预算 | 单路径加宽/加块匹配参数量 |
| C | InfoFlow | A/B + flow | 去掉 flow 但补同等参数 |
| D | E/M 记忆 | A/B/C + E/M | `M=0` / 正常写入 / 打乱写值 |
| E | PerturbField | 前述 + field | 无 field + 同参数噪声基线 |
| F | 动态 Shock/Activity | 前述 + 动态调度 | 固定活跃预算 |

每组只加一个机制；记录收益、计算开销、副作用，尤其是低频组与上下文敏感性。

## 6. 评估与决策规则

每次 eval 统一记录：

- held_top1 / held_top5；
- 频率分组 top1（高频/中频/低频/稀有）；
- `cos`（预测向量与目标码本平均余弦）；
- `ctx_change_rate`、`other_ctx_acc`、`ctrl_keep_last_acc/change`；
- `hard_neg_win`（固定困难负样本池 top-512 高频 token，K=16/target，固定 seed）。

**一个组件只有同时表现为以下三点才算有效：**

1. 低频组 top1 有提升；
2. 上下文敏感性有稳定变化；
3. 困难负样本区分率上升。

否则即使 held top1 提升，也要先怀疑是频率偏置/更多计算/评估噪声。

## 7. 当前不做

- 不放大模型（先不跑 K=30/depth=4/d=768 长训练）；
- 不进入 Online；
- 不在指标不可信时继续加组件。
