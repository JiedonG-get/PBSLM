# ARCHITECTURE — A/C、E/M、InfoFlow 与连续输出

本文件把 PBSLM 的关键信息流写成可实现/可验收的形式，并与代码位置对应。

## 1. 记号

- `d`：隐藏维度
- `k`：主块数量
- `depth`：每个块的层数
- `e_n`：E 块数量
- `act_x`：活跃比例（默认 0.2）
- `M_i ∈ R^{d×d}`：第 i 个 E 块的记忆矩阵
- `z_a / z_c`：A/C 路径输出的连续向量
- `E_v`：token v 的码本向量（单位归一化）

## 2. A / C 双路径

```text
输入上下文 x（T 个 token）
  │
  ├─ token embedding（单位归一化）
  │
  ├─ 角色打分 s_A, s_C（对 x 的均值）
  │     ├─ 选出 A 块集合 S_A
  │     └─ 选出 C 块集合 S_C（与 S_A 互斥）
  │
  ├─ 活跃块计算 Y = BlockBank(x; S_A ∪ S_C)
  ├─ delta = Y - x
  ├─ InfoFlow(delta) → settled, agree
  │
  ├─ side_cons_A = Σ_{i∈S_A} w_i · settled_i   （A 侧共识）
  ├─ side_cons_C = Σ_{i∈S_C} w_i · settled_i   （C 侧共识）
  │
  ├─ E 块：E_out, mem_read
  ├─ a_enc = LN(x + side_cons_A + 0.1·ea_proj(mem_read))
  ├─ c_enc = LN(x + side_cons_C)
  │
  ├─ dec_a(cross-attn) → h_a → z_a
  └─ dec_c(cross-attn) → h_c → z_c
```

要点：

- A/C 不是完全独立系统：共享 token embedding、BlockBank、InfoFlow、E 块；
- 差异在路由集合、记忆读取、侧共识和解码器；
- `agree` 表示块间一致性，不是认知正确率；所有块一致也可能一起错。

代码：`models/model_v2.py::encode`。

## 3. E 块与记忆 M

### 3.1 读取

```text
E_out = E + einsum(E, e_mix)              # E 块之间混合
R_i = E_i M_i                              # 每个 E 块读自己的记忆
mem_read = mean_i(R_i)
```

实现中对 `M_i` 先按行归一化并在 A 编码里乘 `0.1`，原因是：记忆矩阵范数会随写入增长，
未归一化时 A 编码会被一个巨大的常量记忆向量主导（曾导致 a_enc 跨上下文几乎相同）。

### 3.2 写入

```text
M_i ← M_i + η · k_i v_iᵀ
M_i ← decay · M_i
```

- `k_i`：E 块特征（归一化）
- `v_i`：写入 value（目标 token 的 codebook 向量）
- `η`：写入学习率（默认 0.05）
- `decay`：遗忘（默认 0.98）

风险：持续写入会累积噪声；尚无容量管理/冲突解决策略。Phase 2 必须做
「M 恒为零 / 正常写入 / 打乱写入 value」三组对照。

代码：`models/model_v2.py::e_forward`、`write_memory`、`decay_memory`。

## 4. InfoFlow

```text
delta_i ∈ R^{B×T×d}
  → proj_in 到 fdim=64
  → 2 步 FlowStep（块间注意力）
  → proj_out 回 d
  → settled_i, agree_i
```

FlowStep：

```text
h ← LN(h + o(softmax(q(h)k(h)^T/√f) v(h)))
```

实际使用残差：

```text
settled = delta + proj_out(flow(delta))
```

避免低维 flow 把块提议抹掉。

`agree_i`：块 i 的状态与平均状态的归一化距离（一致性，不是正确性）。

## 5. 连续向量输出

### 5.1 目标

不维护 `V×d` 的 softmax，而是预测 token 的码本向量：

```text
z = normalize(f_θ(x)) ∈ R^d
target = normalize(E_y) ∈ R^d
decode = argmax_v cos(z, E_v)
```

### 5.2 训练目标

- Phase 1 实际主目标：sampled-negative InfoNCE（1 正 + 采样负例，排除与正样本相同 token）；
  batch 内负例是**后续待评估**的设计，不在 Phase 1 实现中；
- 诊断时也用 CE（只用于检查数据和容量，不是最终路线）；
- 另加 embedding 多样性/方差正则，防止码本塌缩。

### 5.3 码本模式

| 模式 | 说明 |
|---|---|
| trainable（A1） | 独立 `nn.Embedding` 码本，与输入 embedding 不共享；参与优化 |
| fixed（A2） | 固定 seed 生成的随机单位向量；`requires_grad=False`；不进入优化器 |

### 5.4 已知难点

- 语义相近 token 的码本可能相近，最近邻解码容易混淆；
- 可训练码本会随模型移动，训练 loss 与 held 泛化可能脱节；
- 16k 词表的最近邻解码对「区分度」要求高于普通分类。

## 6. PerturbField

```text
[activity, shock, emotion(8)] → MLP → μ, logσ
z_pert = normalize(z_a + scale·(μ + σ⊙ε))
ε ~ N(0, I)
```

- 预训练阶段 `scale` 应为非零，否则 field 对输出无影响，只有 KL 项会把 σ 拉向先验，表现为「KL 假性归零」；
- 情绪向量默认平静值 `e0 = [0.55, 0.15, 0.05, 0.55, 0, 0, 0, 0]`；
- 后训练阶段再启用情绪评定与后训练目标。

## 7. 评估口径

- `held_top1 / held_top5`：整体解码；
- 频率分组 top1：高频 ≥1%、中频 0.1–1%、低频 0.01–0.1%、稀有 <0.01%；
- `ctx_change_rate`：换随机上下文后预测变化率；
- `ctrl_keep_last_acc/change`：保留最后一个 token、替换其余后的准确率/变化率；
- `other_ctx_acc`：随机另一条上下文 + 它自己的正确目标的准确率（不是配对交换准确率）；
- `hard_neg_win`：固定困难负样本集合上的胜率；
- 所有指标都不应单独作为结论依据。
