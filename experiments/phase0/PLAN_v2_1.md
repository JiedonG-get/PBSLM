# v2.1 实验计划（Phase 0 → Phase 2）

目标：先把训练/评估闭环做对，再用严格消融回答——
「多块 + 双路径 + 记忆 + 信息流动 + 连续输出，能否在同等预算下获得比单路径更好的上下文预测能力」。

不启动训练，直到 Phase 0/0.5 完成并确认。

## 0. 三条硬约束（采纳）

1. **消融分两轨**：
   - 模块消融：同数据、同训练 token 数、同步数、同 held、同指标，只加一个模块；
   - 预算对照：额外用参数量/FLOPs 尽量匹配的对照模型，证明提升不是来自更多计算。
2. **Phase 0 每项修复标记是否影响当前 PBSLMv2 训练路径**，不修无关旧代码。
3. **Baseline A 严格定义**：单路径、自回归、连续向量输出，无 A/C、无 InfoFlow、无 E/M、无 PerturbField、无 Shock/Activity 动态；A1 可训练码本 / A2 固定码本，同数据同预算同评估。

## 1. Phase 0 代码修复清单（附影响判断）

| # | 修复 | 影响 PBSLMv2 训练路径 | 验收 |
|---|---|---|---|
| 1 | resume 时跳过 field 重新初始化 | 是 | 加载 ckpt 后 field 参数不变 |
| 2 | ckpt 保存/恢复 model + optimizer + step + shock 状态 + field 配置 | 是 | 断点续训 loss 连续、step/优化器状态一致 |
| 3 | 默认优化器明确；bitsandbytes 改为可选（缺失时回退 AdamW） | 是 | 无 bnb 环境可启动 |
| 4 | shock 改为样本级状态；默认不参与损失；与 batch 严格对齐 | 是 | 记录样本来源；无 shock 基线可跑 |
| 5 | 评估补齐：top-1/top-5、A/C 分开、频率分组、平均余弦、困难负样本区分率、上下文交换敏感性 | 是 | 每次 eval 输出完整指标 |
| 6 | field 验收：非零 field_scale，记录 KL/σ/μ，做有场 vs 无场对照 | 是 | KL 不因 scale=0 假性归零 |
| 7 | `use_amp` 要么真正 autocast，要么删除 | 是（fp16 路径） | 无未使用变量 |
| 8 | `direct_only` 真正跳过 decoder 计算 | 条件（使用该 flag 时） | 前后计算量/结果一致 |
| 9 | `dense` 明确为「A/C 同块」实验开关，不混入正式配置 | 条件 | 配置文档明确 |
| 10 | `online_params_gate` 的 `self.role` | 否（v2 未调用） | 修但优先级低 |
| 11 | QwenIO `_soft` 改为 buffer/刷新策略 | 否（v2 不用 QwenIO） | 修但优先级低 |

## 2. Phase 0.5 训练闭环 smoke（不正式训练）

- 20–50 步 forward/backward/optimizer step，检查 loss 有限、梯度范数正常；
- 保存 → 新建进程 → resume，检查 model/optimizer/step/shock/field 一致，loss 曲线连续；
  同时校验 CPU RNG、**CUDA RNG (`torch.cuda.get_rng_state_all()`)**、数据/负样本 generator 状态恢复；CPU 环境需能安全跳过 CUDA 字段；
- 评估函数在固定窗口上可复现，top-1/top-5/频率分组/上下文交换指标均有输出；
- 显存/速度记录，确认 batch/AR 设置可跑。

## 3. Phase 1：Baseline A

统一设置：
- 单路径（一个 encoder + 一个 decoder，无 role_a/role_c 路由）
- AR teacher forcing，n_in=16，一次预测 1 个 token（`--ar --ar-m 8`）
- 连续向量输出，**主损失 InfoNCE（batch + 采样负例，不做全词表 softmax）**；
  CE 仅作数据集/容量 sanity check，不作为最终路线指标
- 无 flow、无 E/M、无 field、无 shock/activity 动态
- 数据：`bpe_full_u32.bin`（520M tokens），固定 held 15% 尾部
- 同训练 token 数/步数/评估窗口

| 实验 | 输出码本 | 目的 |
|---|---|---|
| A1 | **独立可训练码本（不与输入 embedding 共享）** | 模型与码本共同学习 |
| A2 | 固定随机单位码本 | 目标空间固定时的预测能力 |

评估（所有实验统一）：
1. held top-1 / top-5 最近邻解码；
2. 高频 / 中频 / 低频 token 分组 top-1；
3. 预测向量与目标平均余弦；
4. 上下文交换敏感性（换上下文后预测变化率 + 正确率变化）；
5. 独立困难负样本胜率：固定负样本池 = 训练集最高频 top-512 token；每个目标用固定 seed 抽 K=16 个（排除目标），只在这 K 个候选上算胜率；
6. 受控上下文替换：保留最后一个 token、替换其余 token 后的准确率与预测变化率；
7. **指标口径**：
   - `other_ctx_acc` = 随机另一条上下文 + **它自己的正确目标** 的 top-1 绝对准确率，**不是**配对上下文交换的准确率变化；
   - 上下文敏感性主要看 `ctx_change_rate`、`ctrl_keep_last_acc`、`ctrl_keep_last_change`；
6. train/held 对照，检查是否只是背词频。

**决策规则**：不能只看 held top-1。至少要同时满足「低频组有提升 / 上下文交换有稳定影响 / 困难负样本区分率上升」才算该组件有效。

## 4. Phase 2：B–F 模块消融（两轨）

| 组 | 增加内容 | 模块消融 | 预算对照 |
|---|---|---|---|
| B | A/C 双路径 | 同 A1/A2 预算 | 单路径加宽/加块匹配参数量 |
| C | InfoFlow | A/C + flow | 去掉 flow 但加同等参数 |
| D | E/M 记忆 | A/C + flow + E/M | 零记忆 / 打乱写值对照 |
| E | PerturbField | 上述 + field | 无场 + 同参数噪声基线 |
| F | 动态 Shock/Activity | 上述 + 动态调度 | 固定活跃预算 |

每组只加一个机制；记录收益、计算开销、副作用（尤其低频组与上下文敏感性）。

## 5. 当前不做

- 不放大模型（先不跑 K=30/depth=4/d=768 的正式长训练）；
- 不启动 Kaggle；
- 不在指标不可信时继续加组件。
