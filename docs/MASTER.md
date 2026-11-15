# MASTER — PBSLM 架构总览

本文件描述 PBSLM 的完整设计（v2 系列），并明确哪些部分已经在代码中实现、哪些仍是后续阶段目标。

## 1. 设计目标

PBSLM（Parallel Block State Language Model）要验证的核心假设：

> 用**多个独立计算块 + 双路径判断 + 持续记忆 + 块间信息流动 + 连续向量输出**，
> 在相近参数量/FLOPs 预算下，是否能比单路径 baseline 学到更好的上下文预测表示。

它不是为了「模拟大脑」而堆模块；每个模块都必须能在受控消融里证明自己的增益。

## 2. 组件分层

### 2.1 BlockBank（并行块池）

- 每个块有独立的 QKV、FFN、LayerNorm、位置编码参数；
- 只计算当前被选中的块；
- 目标是让参数容量与实际 FLOPs 解耦；
- 关键风险：路由坍缩到少数块时，大参数量只是「闲置容量」。
  因此需要监控每块的激活次数、梯度范数、A/C 选中比例。

代码：`models/model.py::BlockBank`。

### 2.2 A / C 双路径

- A 路径：角色路由 → 所选块 → 可读取 E 块记忆 `M`；
- C 路径：角色路由 → 所选块 → **不读取 M**；
- legacy `encode` 共享batch路由，A先选、C排除A；可选 `use_sample_masks=True` 使用逐样本独立A/C路由，允许物理块重叠，comm/Flow状态按两侧mask分别计算。
- 两条路径共享底层块池、输入 embedding、comm与InfoFlow参数；E/M读取只进入A。
- 共享权重与只读Y/delta不等于参数独立；状态隔离证据限于测试配置。

代码：`models/model_v2.py::PBSLMv2.encode`。

### 2.3 E 块与外部记忆 M

- `e_n` 个 E 块常驻参与前向（不受 20% 活跃比例限制）；
- 每个 E 块维护一个 `d×d` 关联记忆矩阵 `M_i`；
- 读取：`R_i = E_i M_i`（实现中对记忆做过归一化/缩放，避免常量记忆吞掉 A 编码）；
- 写入：外积 `M_i ← decay·M_i + η k_i v_iᵀ`；
- C 路径不直接读取 M。

代码：`models/model_v2.py::PBSLMv2.e_forward`、`write_memory`、`decay_memory`。

### 2.4 InfoFlow（信息流动）

- 活跃块输出 `delta_i` → 低维投影（`fdim=64`）→ 块间注意力（2 步）→ `proj_out` 回 `d`；
- 返回 settled 状态与一致性 `agree`（块状态与平均状态的归一化距离）；
- 加残差：`settled = delta + flow(delta)`，避免流量把块提议抹掉。

代码：`models/model.py::InfoFlow`，`models/model_v2.py::encode`。

### 2.5 连续向量输出

- 模型输出 `d` 维连续向量 `z`，单位归一化；
- 训练目标：对齐 token 码本 `E[y]`；使用 InfoNCE/对比目标，而不是全词表 softmax；
- 解码：`argmax cos(z, E[v])`；
- 码本支持两种模式：
  - 可训练（A1）；
  - 固定随机单位向量（A2，固定 seed，不参与优化器）。

代码：`models/model_v21.py::SinglePathContinuousLM`（baseline）、`models/model_v2.py`（完整版）。

### 2.6 PerturbField（扰动场）

- 输入三旋钮：活跃度、震撼值、情绪向量（8 维，平静默认值 `e0`）；
- 输出 `μ`、`logσ`，对 A 侧输出加高斯扰动：
  `z_pert = normalize(z_a + scale·(μ + σ⊙ε))`；
- 训练时有 KL 正则，预训练期 scale 不应为 0（否则 KL 会被拉向先验，形成「假性 KL 归零」）；
- 当前 Phase 1 baseline 不启用扰动场。

代码：`models/model_v2.py::PerturbField`。

### 2.7 Online / 半在线

设计目标：A 侧在线更新 + 写记忆；C / 主块 / Flow 冻结。冻结C不等于A没有灾难性遗忘，必须分别验证。
已有小型合成prequential适应、记忆干预和同步Runtime接线实验；尚未完成自然语言持续学习、并发实时系统或长期保持验收。见根README的最新证据入口与 `RESEARCH_EXECUTION.md`。

## 3. 前向数据流（完整 v2）

```text
token ids
  │
  ├─ token embedding E_in（单位归一化）
  │
  ├─ E blocks（e_n 个，常驻）
  │     ├─ E_out = E_in + mix(E_in)
  │     ├─ 每个 E_i 读 M_i → mem_read
  │     └─ A 编码加入 ea_proj(mem_read)（C 不读）
  │
  ├─ BlockBank（k 个块）
  │     ├─ A/C 角色打分 → 选活跃块（20% 预算）
  │     ├─ bank.block_forward → delta_i
  │     ├─ bank.comm → 块间通信（v2已调用；masked分支按A/C分别计算）
  │     └─ InfoFlow(delta) → settled + agree
  │
  ├─ a_enc = LN(E_in + side_cons_A + 0.1·ea_proj(mem_read))
  ├─ c_enc = LN(E_in + side_cons_C)
  │
  ├─ dec_a/dec_c（cross-attention）→ h_a / h_c
  ├─ z_a = normalize(out_a(h_a)); z_c = normalize(out_c(h_c))
  ├─ z_pert = normalize(z_a + field_scale·(μ + σ⊙ε))
  └─ decode: nearest token in codebook
```

## 4. 当前实现状态

| 组件 | v2（`model_v2.py`） | v2.1 baseline（`model_v21.py`） |
|---|---|---|
| BlockBank | 有（k/depth 可配） | 有（k/depth 可配） |
| A/C 双路径 | 有 | 无（单路径） |
| E 块 + M | 有 | 无 |
| InfoFlow | 有 | 无 |
| 连续输出 + InfoNCE | 有（完整版用 CE/InfoNCE 调试） | 有 |
| PerturbField | 有（可关） | 无 |
| Online | 小型合成适应/记忆与同步协议原型；完整系统未验证 | 无Online机制 |

Phase 1 只验证 baseline；完整 v2 的消融与模块收益属于 Phase 2。

## 5. 已知风险

1. 稀疏路由可能让大部分块得不到训练；
2. 记忆矩阵可能累积噪声，需要归一化/遗忘/容量管理；
3. `agree` 是「一致性」不是「正确性」；
4. 连续向量要同时表达语义方向与区分 16k token，难度高于 softmax；
5. 可训练码本可能与模型共同移动目标空间，导致训练 loss 更好看但泛化变差；
6. 非自回归多位置输出会稀释监督（已改用 AR teacher forcing 诊断）。
