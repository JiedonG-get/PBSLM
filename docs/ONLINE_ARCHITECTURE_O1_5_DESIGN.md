# ONLINE_ARCHITECTURE_O1_5_DESIGN — A/C 路由与跨块信息流解耦设计

状态：**O1.5 架构分析文档（补充 active 三集合、严格 mask、Flow 语义、参数隔离边界、
BlockBank beta 索引、batch union 语义、逐块独立性、M1/M2 严格等价、
P0-5 可区分 beta 测试、P0-6 统一 mask 布局、P0-7 非零更新、P0-8 count 定义、P0-9 Flow attention mask）。
只做设计，不修改模型代码，不运行训练/loss/backward/optimizer.step/smoke。**
基线：O1 提交 `ad99bc1`；O1 只证明了**输出头已拆分**、六组 A 参数白名单和基础 C 稳定性，
**不代表 A/C 计算路径已经完全隔离，更不代表 Online Learning 已实现。**

> O1.5 只实现**路由与计算状态隔离**；BlockBank、comm1/comm2、InfoFlow、E 块的**参数仍然共享且冻结**。
> 因此 O1.5 不能宣称 A/C 参数独立；未来若允许共享参数更新，必须重新验证 C 稳定性。

---

## 1. 当前 A/C 的完整调用链

`models/model_v2.py::forward` → `encode` → A/C 输出头。

```text
forward(ids, m, act, ...)
  encode(ids, act)
    x = token_emb(ids)                         # (B,T,D)
    kA, kC = quotas(act)                       # act 是 batch 共享标量
    hmean = x.mean(1).mean(0)                  # (D,)  ← batch 共享
    sa = role_a(hmean)                         # (k,)  ← batch 共享
    sc = role_c(hmean)                         # (k,)  ← batch 共享

    if dense:
        ia = ic = all blocks
    else:
        ia = topk(sa, kA)                      # A 先选
        taken[ia] = True
        ic = topk(sc.masked_fill(taken,-inf), kC)  # C 从排除 A 的块中选

    chosen = [ia, ic]
    active = sorted(set(ia + ic))              # batch 共享 active 集合

    Y = bank.block_forward(x, active)          # (S,B,T,D)
    delta = Y - x.unsqueeze(0)
    delta = bank.comm(delta, ones(S))          # A/C 共享同一通讯
    settled, agree, _ = flow(delta)            # A/C 共享同一 Flow

    side_cons(A) = aggregate(settled[A rows], weights(sa))
    side_cons(C) = aggregate(settled[C rows], weights(sc))

    E_feats, mem_read = e_forward(x)           # E/M 只进 A
    a_enc = ln_a(x + side_cons(A) + ea_proj(mem_read))
    c_enc = ln_c(x + side_cons(C))             # C 不读 M

  h_a = dec_a(...); h_c = dec_c(...)
  z_a = normalize(out_a(h_a) + ctx_a)
  z_c = normalize(out_c(h_c) + ctx_c)          # O1 已拆分输出头
  z_pert = field(z_a)                          # field 只作用 A
```

### 1.1 耦合点

| # | 耦合点 | 代码证据 | 影响 |
|---|---|---|---|
| C1 | `hmean` 先对 T 求均值、再对 B 求均值 | `hmean = x.mean(1).mean(0)` | 路由不是逐样本独立；同一样本在 batch 中与单独输入可能选不同块 |
| C2 | `role_a` 先选，C 再排除 A 已选块 | `_select()` | **更新 `role_a` 会改变 C 候选集合** |
| C3 | A/C 共享 active 集合 | `active = sorted(set(ia + ic))` | active 集合随 A 选块变化 |
| C4 | `bank.comm` 对全部 active 块算 `others` | `BlockBank.comm()` | C 行的 `others` 含 A 块 |
| C5 | `flow` 对全部 active 块求均值/一致性 | `InfoFlow.forward` | C 的 settled/agree 含 A 块 |
| C6 | `side_cons(C)` 使用共享 `settled` 的 C 行 | `encode()` | C 编码依赖 C4/C5 |
| C7 | `side_I`/`I_c` 使用共享 `agree` 的 C 行 | `encode()` | C 的 `I_c` 依赖 C4/C5 |
| C8 | `out_c` 已独立 | O1 修改 | 只解决输出头，不解决上游 C1–C7 |

结论：O1 只解决了 C8；C1–C7 仍存在。`role_a` 在 O1 中被冻结，只是避免触发 C2，
不是架构上消除了耦合。

当前代码行号（O1 提交 `ad99bc1`）：

| 位置 | 代码 |
|---|---|
| `models/model_v2.py:135` | `hmean = x.mean(1).mean(0)`（batch 共享） |
| `models/model_v2.py:136` | `sa, sc = self.role_a(hmean), self.role_c(hmean)` |
| `models/model_v2.py:106-111` | `_select()`：A 先选，C 用 `masked_fill` 排除 A |
| `models/model_v2.py:142-143` | `chosen` / `active` 全局共享 |
| `models/model_v2.py:146` | `bank.comm(delta, ...)` 对全部 active 块 |
| `models/model_v2.py:148` | `flow(delta)` 对全部 active 块 |
| `models/model_v2.py:156-172` | `side_cons`/`side_I`，C 读取共享 settled/agree 的 C 行 |
| `models/model_v2.py:211` | O1 已拆分的 `out_c` 输出头 |

### 1.2 O1 测试结果的正确解释

- `role_a` 扰动测试已观察到 A 选块和 C 输出同时变化；
- O1 六组白名单的 C 稳定性成立，是因为 `role_a` 被排除在更新集合外；
- `field` 稳定性测试使用 `sample_noise=False`，只验证确定性 field 路径；
  随机扰动路径仍未验证；
- batch=2 测试验证了批量前向，但没有验证每个样本有独立路由；
- 这些都不改变 O1 结论：**输出路径拆分 + 受控白名单成立，A/C 计算路径未完全隔离。**

---

## 2. 目标架构与数据流

### 2.1 目标

1. A 的路由选择变化不再改变 C 的 `chosen_c`/`active_c`、通讯路径和输出；
2. 路由逐样本独立，不因 batch 组成变化；
3. `active_a`、`active_c`、`active_union` 三种集合语义分开；
4. A/C 的 comm/Flow/side_cons 只使用各自 active 集合，中间计算无跨角色泄漏；
5. 共享 BlockBank/comm/Flow 参数冻结，只做计算状态隔离；
6. 保持 C 的稳定性和旧 checkpoint 可加载；
7. O1.5 仍不实现 Online Runtime，只做架构隔离。

### 2.2 三种 active 集合必须分开

O1.5 必须统一以下语义，不能再用一个全局 `active` 同时表达三件事：

| 名称 | 含义 | 是否受 A 路由变化影响 | 用途 |
|---|---|---|---|
| `chosen_a` / `active_a` | 每个样本、A 自己选中的块 | 可以变化 | A 的 comm/flow/side_cons |
| `chosen_c` / `active_c` | 每个样本、C 自己选中的块 | **必须不变** | C 的 comm/flow/side_cons |
| `active_union` | 每个样本 A/C 选块的并集 | 可以变化 | **仅用于共享 BlockBank 的计算调度** |

规则：

1. `active_union` 是计算调度集合，不是稳定性指标；
2. A 路由变化可以改变 `active_union`，但不能改变 `chosen_c` / `active_c`；
3. C 的验收必须检查 `active_c`，不能检查全局 `active_union`；
4. A/C 选到同一物理块时，该块可以同时出现在 `active_a` 和 `active_c` 中，
   但两侧的 comm/flow/side_cons 必须独立计算；
5. 模型输出字段建议为：
   `chosen_a`、`chosen_c`、`active_a`、`active_c`、`active_union`；
   旧 `active` 字段只能作为兼容别名，不能作为新验收依据。

### 2.3 目标数据流

```text
x = token_emb(ids)                              # (B,T,D)
h_b = x.mean(1)                                 # (B,D) ← 每个样本独立
sa = role_a(h_b)                                # (B,k)
sc = role_c(h_b)                                # (B,k)

ia_b = topk(sa, kA)                             # (B,kA) 每样本独立
ic_b = topk(sc, kC)                             # (B,kC) 每样本独立，不排除 A
# 允许 A/C 选到同一块；是否强制不重叠需要单独实验

active_a = ia_b                                 # 每样本集合
active_c = ic_b                                 # 每样本集合，独立于 A
active_union = union(active_a, active_c)        # 只用于 block_forward 调度

Y = bank.block_forward(x, active_union)         # 块输出本身按块独立
delta = Y - x

# A 与 C 分别计算通讯/Flow，互不进入对方的块集合与 mask
settled_a, agree_a = side_comm_flow(delta, active_a, role=A)
settled_c, agree_c = side_comm_flow(delta, active_c, role=C)
# settled_* 是每块 delta + proj_out(h_side) 的残差，不是 proj_out(mean) 共识
# 详见第 2.6 节的 Flow 输出语义

side_cons_a = aggregate(settled_a, weights(sa, active_a))
side_cons_c = aggregate(settled_c, weights(sc, active_c))

I_a = mean(agree_a over active_a)
I_c = mean(agree_c over active_c)

a_enc = ln_a(x + side_cons_a + ea_proj(mem_read))
c_enc = ln_c(x + side_cons_c)
z_a = out_a(dec_a(...)) + ctx_a
z_c = out_c(dec_c(...)) + ctx_c
```

关键点：

- C 的选块**不再排除 A**；C 不感知 A 的选择；
- `active_a`/`active_c`/`active_union` 三者语义分开；
- `bank.comm`/`flow` 分别只看 A 或 C 的块集合；
- C 的 `settled`/`agree`/`I_c` 不含 A 块；
- C 不读 M，E 块参数冻结。

### 2.4 计算形式：masked 还是 per-sample

逐样本选块意味着不同样本的 A/C 支持集不同，`bank.comm`/`flow` 不能直接用全局 active list。
备选实现：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| M1 | 对每个样本单独调用 `bank.comm`/`flow` | 语义最清楚，易验证 | batch 循环，吞吐差 |
| M2 | 全体块 + per-sample mask | 可批量，语义可控 | 需要给 `BlockBank.comm`/`InfoFlow` 增加 mask 接口 |
| M3 | 按支持集分组批量 | 性能折中 | 分组/去重复杂，调试难 |

**建议 O1.5 先用 M1 做正确性基准，再用 M2 做批量化**；两者结果必须在单样本/小 batch 上一致。

M1/M2 一致性比较必须逐样本、逐块覆盖：

- `settled_a` / `settled_c`；
- `agree_a` / `agree_c`；
- `I_c`；
- `side_cons_c`；
- 最终 `z_c`。

不能只比较最终 `z_c`；中间 `settled`/`agree`/`side_cons` 任一不一致都必须失败。

### 2.5 comm/Flow 的严格 mask 定义

不能只在最后的 `side_cons` 聚合时加 mask；`bank.comm` 与 `InfoFlow` 的中间计算也必须严格隔离。

#### 2.5.1 `BlockBank.comm`

对每侧 `side ∈ {A,C}`，**唯一标准布局**为 `(G,B)`：

```text
mask_side[g, b] = 1   iff global block g 属于样本 b 的 active_side
# 形状 (G,B)，G = len(active_union_global)
# 向 (G,B,T,D) / (G,B,T,f) 广播时使用 mask_side[:, :, None, None]
```

`others` 的定义必须改为：

```text
tot_side[b]        = Σ_g mask_side[g,b] · Y_side[g,b]
others_side[g,b]   = tot_side[b] - mask_side[g,b] · Y_side[g,b]
```

通信更新项和输出也必须受 mask 控制：

```text
r_side[g,b]         = comm2(gelu(comm1(others_side[g,b])))
update_side[g,b]    = mask_side[g,b] · beta_g · r_side[g,b]
out_side[g,b]       = mask_side[g,b] · (Y_side[g,b] + update_side[g,b])
# 广播：mask_side[:, :, None, None] 到 (G,B,T,D)
```

约束：

1. `others` 只包含**当前样本、当前角色**选中的块；
2. 无效块 `mask=0` 不参与 `tot`/`others`；
3. 无效块不能接收 comm 更新，`update_side` 和 `out_side` 都必须乘 `mask_side`；
4. 每次/每层输出后再次应用 mask，避免无效块状态残留或重新进入后续计算；
5. 同一物理块被 A/C 同时选中时，A 侧和 C 侧分别用自己的 `mask_side`、自己的 `Y_side` 调用 comm；
   共享的是冻结的 `comm1/comm2` 参数，不是中间 `others` 状态；
6. 两侧的输出不得写回同一个可变缓存，也不能用一侧的 `out_side` 覆盖另一侧的输入。

当前 `BlockBank.comm()`（`models/model.py:121`）没有 mask 参数，O1.5 需要新增可选 mask 接口；
旧调用不带 mask 时必须保持原语义。

#### P0-6 mask 广播约定

`mask_a`/`mask_c` 统一为 `(G,B)` bool：

| 计算 | 广播方式 |
|---|---|
| `Y` `(G,B,T,D)` | `mask_side[:, :, None, None]` |
| `h` `(G,B,T,f)` | `mask_side[:, :, None, None]` |
| `count_b` | `mask_side.sum(0)`，形状 `(B,)` |
| `tot` | `(mask_side[:, :, None, None] * h).sum(0)`，形状 `(B,T,f/D)` |
| `mean` | `tot / count_b.clamp_min(1)[:, None, None]`，形状 `(B,T,f/D)` |
| `scale` | 有效块 norm 的 masked 均值，形状 `(B,T)` |
| `agree` | 只用有效块的 `h`，分母 `count_b.clamp_min(1)` |
| Flow attention key | 对样本 b 使用 `mask_side[:, b]`，形状 `(G,)`，作用在 logits 的 key 维 |
| Flow query | 只对 `mask_side[:, b]` 为真的块产生输出 |
| 每层输出 | 重新乘 `mask_side[:, :, None, None]` |

所有 A/C comm、FlowStep、mean、agree、settled 计算必须使用同一 `(G,B)` 布局；
不允许一侧用 `(G,B)`、另一侧用 `(B,G)`。

#### P0-8 count 的数学定义与广播

```text
count_b = mask_side.sum(0)                 # (B,)
count_b = count_b.clamp_min(1)             # 防除零；正常路径必须 >=1
tot     = (mask_side[:, :, None, None] * h).sum(0)   # (B,T,f)
mean    = tot / count_b[:, None, None]     # (B,T,f)
scale_b = (mask_side[:, :, None] * h.norm(dim=-1)).sum(0) / count_b[:, None] + 1e-6
```

- `count_b` 是每个样本在当前角色下的有效块数；
- `mean` 必须是 `(B,T,f)`；`scale_b` 必须是 `(B,T)`；
- `agree` 只对有效块计算，分母同样是 `count_b`；
- `count_b == 0` 不能只用 clamp 掩盖语义错误：
  - 正常架构不变量是每个样本、每个角色至少 1 个有效块（`kA >= 1`、`kC >= 1`）；
  - 正常路径必须在 softmax/mean 前断言 `count_b >= 1`；
  - 若测试或异常路径出现 `count_b == 0`，应抛出受控错误（不得产生 NaN）；
  - `clamp_min(1)` 只是数值保护，不是“空集合合法”的声明。

#### 2.5.2 `InfoFlow` 与 `FlowStep`

对每侧有效块 `mask_side`：

1. **attention key/value**：无效块的 key/value 必须屏蔽；可在 attention logits 上对无效 key 加 `-inf`；
2. **attention query**：无效 query 不产生输出；计算后按 mask 清零，不能把无效块的状态留在 `h` 中；
3. **每一层 FlowStep**：mask 必须随层传递，每层输出后重新应用；不能让无效块在下一层重新进入通信；
4. **`mean` / `scale` / `agree`**：只对当前样本、当前角色的有效块计算；
   分母是有效块数，不是全部 block 数；
5. **`proj_out(mean)`**：只对有效块均值投影，得到每侧 consensus；
6. **同一物理块**：A/C 两侧分别调用 Flow，分别计算 mask/attention/mean/agree；
   即使输入 `delta` 相同，也不能共享中间 attention/mean 状态。

当前 `InfoFlow.forward()` 使用 `mean = h.mean(0)`、`scale = h.norm(dim=-1).mean(0)`，
没有 mask；O1.5 需要可选 mask/weights 接口，旧调用保持原语义。

#### 2.5.2.1 P0-9 Flow attention mask 的广播与全无效处理

`h/q/k/v` 形状 `(G,B,T,f)`；attention logits 形状 `(B,T,G,G)`，
其中最后一维是 **key block** 维。

1. **key mask**：对样本 b，`key_mask_b = mask_side[:, b]`，形状 `(G,)`；
   将其广播到 logits 最后一维：
   ```text
   logits[..., ~key_mask_b] = -inf
   ```
   无效 key 不得参与 softmax；
2. **query mask**：无效 query 可以计算，但输出必须清零；
   或只对有效 query 计算；无论哪种，`msg`/`h` 的无效行必须乘 query mask；
3. **每层 FlowStep**：每层输出后重新应用 `mask_side[:, :, None, None]`；
   下一层 attention 继续使用同一 key/query mask，无效块不能重新进入；
4. **全无效 key**：
   - 正常路径不变量：`count_b >= 1`（`kA>=1`, `kC>=1`），必须在 softmax 前断言；
   - 若 `count_b == 0`：不得对全 `-inf` logits 直接 softmax；
     必须走受控错误路径（抛 `ValueError`）或显式跳过该样本并置零，
     具体行为必须由测试固定，且不得产生 NaN；
   - O1.5 验收要求这是显式不变量 + 受控错误/跳过，不允许静默 NaN。
5. **每层输出**：无效 query 和无效 key 的状态都不得写回 `h`；
6. **测试**：构造 `count_b==0` 的 mask，断言受控错误或显式零输出，不能出现 NaN。

#### 2.5.3 mask 的测试要求

必须固定 `chosen_c`/`active_c`，再改变 A 侧输入。

- 固定 `active_c`，改变 A 侧 mask / A-only 块 / `active_union` 中仅供 A 使用的块：
  C 的 comm/flow 中间输出和最终输出不得变化；
- 构造 A/C 重叠块，固定 `active_c`，扰动 A 侧该块：C 侧输出不得变化；
- 对 C 自身的无效块：只验证它们不参与 C 计算
  （`others`/`mean`/`scale`/`agree`/输出中没有它们的贡献），
  **不要**通过关闭 `active_c` 内有效 mask 位来要求 C 整体输出不变；
- 每一层 FlowStep 后检查无效块状态是否被清零；
- `agree`/`mean`/`scale` 的有效块分母必须正确；
- `active_union` 可以变化，但 `active_c` 与 C 中间/最终输出必须不变。

### 2.6 Flow 输出语义

必须对照当前代码固定以下语义，避免重构时把 settled 与 consensus 混淆：

当前 `models/model.py::InfoFlow.forward`：

```text
h = proj_in(delta)                 # (S,B,T,fdim) 低维 per-block 状态
for layer in layers: h=layer(h)    # FlowStep 输出仍为 (S,B,T,fdim)
mean = h.mean(0)                   # (B,T,fdim) 块间均值
scale = h.norm(dim=-1).mean(0)     # (B,T)
agree = 1 - ((h-mean)/scale).clamp(0,2)
cons = proj_out(mean)              # (B,T,D) 共识向量
return h, agree, cons
```

当前 `models/model_v2.py::encode`：

```text
settled_low, agree, _ = self.flow(delta)          # 忽略 cons
settled = delta + self.flow.proj_out(settled_low)  # 每块 d 维残差
side_cons = aggregate(settled[selected_rows], weights)
```

因此 O1.5 必须区分：

| 名称 | 形状/含义 | 是否用于 side_cons |
|---|---|---|
| `h_side` / `settled_low` | FlowStep 后的低维 per-block 状态 `(S,B,T,fdim)` | 否（仅作为投影输入） |
| `proj_out(h_side)` | 低维状态投影回原维度 `(S,B,T,D)` | 与 `delta` 相加构成 settled |
| `settled_side` | `delta_side + proj_out(h_side)` 的 per-block 残差 | **是** |
| `consensus_side` | `proj_out(masked_mean(h_side))`，`(B,T,D)` | **不是**（当前 v2 忽略 cons） |

O1.5 的 `settled_a`/`settled_c` 必须对应 `settled_side`（per-block 残差），
不能改成 `consensus_side`；`consensus_side` 只能作为额外输出/后续用途。
无 mask、全集块时应与 legacy 路径数值一致（回归测试）。

### 2.7 BlockBank beta 索引与 batch 语义（P0-1/P0-2/P0-3）

#### 2.7.1 `BlockBank.comm` 的 beta 必须按物理 block ID 索引

当前 `models/model.py::BlockBank.comm` 使用：

```python
beta = torch.sigmoid(self.comm_beta[:Y.shape[0]])
```

如果 `Y` 的行对应 active block 索引 `[2,5,7]`，这会错误使用 `comm_beta[0:3]`。

修正方案：

- `comm(Y, g, idx=None, mask=None)` 增加 `idx`（物理 block ID）；
- `idx` 为空时默认 `torch.arange(Y.shape[0])`，只保留旧连续索引语义；
- beta 改为 `torch.sigmoid(self.comm_beta[idx])`；
- `g`、`beta`、`mask` 都按 `idx` / 当前角色有效块对齐；
- 旧路径若需要复现历史行为，必须显式使用 legacy 开关，不能假装新旧等价。

#### 2.7.1.1 P0-5：beta 测试必须验证实际 `comm()` 行为

默认初始化 `comm_beta = full(-2.0)` 所有值相同，不能区分正确索引和错误索引；
只比较 `sigmoid(comm_beta[idx])` 与手工索引表达式也没有验证 `comm()` 本身。

测试必须：

1. **主动设置互不相同的 beta**：
   ```text
   comm_beta = tensor([-2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, ...])
   ```
2. **主动设置 `comm2` 为非零权重/偏置**，不能只写“非零”而不验证；
3. 使用确定性输入 `Y`、`g=ones(3)`、`idx=[2,5,7]` 调用**实际 `comm()`**；
4. 按同一组 `Y`/`g`/`comm1/comm2` 手工计算参考输出：
   ```text
   tot = Y.sum(0)
   others = tot.unsqueeze(0) - Y
   r = comm2(gelu(comm1(others)))
   expected = Y + sigmoid(comm_beta[[2,5,7]]) * r
   ```
5. **断言 `r.abs().max() > 0`**，确保本次输入下更新项确实非零；
   如果为零，测试 setup 失败，不能继续比较；
6. 用 `torch.testing.assert_close(actual, expected)` 或 `torch.allclose` 比较浮点输出，
   不要求逐位完全相等；
7. 用连续切片构造错误参考：
   ```text
   wrong = Y + sigmoid(comm_beta[:3]) * r
   ```
   断言 `not torch.allclose(actual, wrong)`，确认错误路径确实不同；
8. 记录 `actual/expected` 与 `actual/wrong` 的最大差值。

测试不能只验证手工写出的索引表达式，必须验证实际函数行为。

> 注意：`comm2` 默认 `zero_()`，如果不同时设置非零权重，beta 无论取什么都无法影响输出，
> 该测试会失去区分能力。

#### 2.7.2 `active_union` 的 batch 语义

现有 `BlockBank.block_forward` 接收一维 `idx`，不能直接接收逐样本集合。O1.5 明确：

| 名称 | 形状/类型 | 用途 |
|---|---|---|
| `active_union_per_sample[b]` | 每个样本的块集合 | M1 逐样本计算 |
| `active_union_global` | 批次内所有样本支持集的并集，1D sorted list | M2 一次性 block_forward 调度 |
| `mask_a` | **唯一标准布局 `(G,B)`**，G=`len(active_union_global)` | A 侧有效块掩码 |
| `mask_c` | **唯一标准布局 `(G,B)`** | C 侧有效块掩码 |

规则：

1. `active_union_per_sample` 只用于 M1 的逐样本调用；
2. `active_union_global = sorted(∪_b active_union_per_sample[b])` 才可传给现有 `block_forward` 的一维 `idx`；
3. M2 对 global union 只调用一次 `block_forward`，再用 `mask_a/mask_c` 选择；
4. 不能把 `active_union_per_sample` 直接当作 `block_forward` 的 idx；
5. 每个样本的 `mask_side` 只标记该样本在该侧的合法物理块；
6. A-only 块即使出现在 global union，也不能进入 C 的 mask/mean/agree。

#### 2.7.3 BlockBank 逐块独立性测试（P0-3）

BlockBank 参数是共享的，但 `block_forward` 的**单个块计算**必须与 union 无关。
必须测试：

1. `block_forward(x, [j])` 与 `block_forward(x, union)` 中第 j 行一致；
2. 向 union 增加其他物理块，不改变已有块的输出；
3. 扰动 A-only 物理块的参数，C-only 物理块的 block_forward 输出不变；
4. A/C 重叠物理块：扰动该块参数时，两侧该块的输出都变化（参数共享的正常结果），
   但不能出现中间状态/缓存污染；
5. BlockBank 不保存跨调用可变状态；同一 block 在两次调用中输出一致。

必须区分：

- **参数共享**：同一物理块被 A/C 同时使用时共享权重，这是预期行为；
- **计算结果共享**：不能把 A 侧某次调用的中间结果缓存后给 C 侧使用；
  block_forward 必须是纯函数。

### 2.8 M1/M2 严格等价条件（P0-4）

M1 是参考实现，M2 是批量实现；不能只比较最终 `z_c`。

#### 2.8.1 M1（逐样本参考）

```text
for b in range(B):
    u_b = active_union_per_sample[b]
    Y_b = bank.block_forward(x[b:b+1], u_b)   # (S_b,1,T,D)
    delta_b = Y_b - x[b:b+1]
    # side-specific comm/flow for A/C with mask
    ...
```

#### 2.8.2 M2（global union + mask）

```text
G = active_union_global
Y_g = bank.block_forward(x, G)                # (G,B,T,D)
for b:
    gather rows G -> u_b; apply mask_a/mask_c
    # side comm/flow
```

#### 2.8.3 必须逐样本、逐块比较的中间量

| 层级 | 比较内容 |
|---|---|
| BlockBank | 每块 `block_forward` 输出（M1 的块行 vs M2 global union 对应行） |
| comm | `others`、`update`、`out`（有效块逐块；无效块必须为零） |
| FlowStep | 每层 FlowStep 的 `h` 输出（有效块逐层/逐块） |
| Flow | `settled_c`、`agree_c` |
| side | `I_c`、`side_cons_c` |
| 最终 | `z_c` |

要求：

- 逐样本、逐块、逐层比较；
- 不能只比较 `z_c`；
- 记录每层的 `max_abs_err`、`max_rel_err`；
- fp32 建议 `atol=1e-6, rtol=1e-5`；如果确定性 CPU 能 bitwise 相等，优先要求 `torch.equal`；
- 任一中间层超出容差，M2 失败，不能“最终输出接近就通过”；
- 测试需要可选的 trace/中间返回（或测试 hook），生产路径可以关闭 trace，但不能关闭 mask。

---

## 3. 逐样本路由的影响

### 3.1 `hmean` 改为 `h_b`

- `x.mean(1)` 仍是 T 维池化；去掉 `.mean(0)` 即得到 `(B,D)`；
- `role_a/role_c` 输出从 `(k,)` 变为 `(B,k)`；
- 路由得分、选择、`side_cons` 权重、`side_I` 都需要 batch 维；
- 单样本与 batch 中同一样本的 `h_b` 相同（除浮点求和顺序），路由应一致。

### 3.2 受影响模块

| 模块 | 影响 |
|---|---|
| `quota` | 当前 act 标量，仍共享；如需 per-sample activity，后续扩展 |
| `role_a/role_c` | 输入变 `(B,D)`，输出 `(B,k)` |
| `_select` | 改为 per-sample topk；C 改为独立 topk |
| `chosen/active` | 拆成 `chosen_a/active_a`、`chosen_c/active_c`、`active_union` |
| `bank.block_forward` | 块输出本身独立；需要支持按样本 union/mask |
| `bank.comm` | 需要 per-sample mask/权重，只在各侧集合内聚合 |
| `InfoFlow` | 需要 per-sample mask/权重，分别算 A/C |
| `side_cons` | 权重从 `(k,)` 变 `(B,k)`，聚合按样本 |
| `side_I` | 按样本的 side agree |
| `forward` 输出 | 输出 `chosen_a`、`chosen_c`、`active_a`、`active_c`、`active_union`；旧 `active` 仅兼容别名 |

---

## 4. A/C 隔离方案

### 4.1 必须做

1. **逐样本路由**；
2. **C 独立选块**（不排除 A）；
3. **A/C 分别 comm/flow**；
4. `side_cons`/`I` 只读各自 settled/agree；
5. `role_a` 在隔离测试通过前仍冻结。

### 4.2 可选方案

| 方案 | 说明 | 评价 |
|---|---|---|
| B1 允许 A/C 重叠 | 各自 top-k，独立集合 | 最简单、最彻底解耦；需观察互补性/开销 |
| B2 强制不重叠 | 用固定优先级/reservation 选 C | 若 A 先选则 C 仍依赖 A，不能作为 C 稳定目标；需改为 C 先选或独立分区 |
| B3 独立块池 | A/C 各用独立块库 | 参数量/结构变化大，不符合当前共享 BlockBank 目标 |

**建议 B1**：允许重叠；共享块参数冻结，两侧只在各自集合内通讯。重叠是否降低 A/C 互补性留待后续实验。

### 4.3 参数隔离与计算状态隔离的边界

O1.5 只解决**计算状态隔离**，不解决参数独立：

- `BlockBank`（包括 `block_forward`、`comm1/comm2/comm_beta`）参数共享且冻结；
- `InfoFlow`（`proj_in/FlowStep/proj_out`）参数共享且冻结；
- `ebank/e_mix/ea_proj` 参数共享且冻结；
- 允许 A/C 选到同一物理块，但 A/C 的 comm/flow mask、attention、mean/agree、side_cons 中间状态必须独立；
- 共享参数被更新时，之前的 C 稳定性结论全部失效，必须重新设计/测试；
- O1.5 的验收结论只能写：
  “**在共享参数冻结条件下，A/C 路由和计算状态已隔离**”，
  不能写“A/C 参数已独立”或“A/C 完全独立”。

---

## 5. O1 白名单是否需要调整

### 当前 O1（六组，`role_a` 冻结）

`dec_a`、A query、`refine_proj`、`ln_a`、`ctx_a`、`out_a`。

### O1.5 分阶段

| 阶段 | 白名单 | 理由 |
|---|---|---|
| O1.5-A（逐样本路由 + C 独立选块） | 仍六组，`role_a` 冻结 | 先验证逐样本路由/独立选块，不引入路由学习 |
| O1.5-B（A/C 分别 comm/flow） | 仍六组 | 结构性隔离；`role_a` 更新仍可能改变 A active 支持，先不放开 |
| O1.5-C（隔离通过后） | **七组：增加 `role_a`** | C 不再依赖 A 路由，`role_a` 成为 A-only 可更新参数；需新增验收 |

`role_a` 增加的兼容性影响：

- 参数量不变（`role_a` 已存在）；
- optimizer 多一组，旧 checkpoint 可加载（参数形状不变）；
- 行为改变：A 的 active 集合会随训练变化，吞吐/显存波动；
- 回滚：`freeze_for_online()` 重新冻结 `role_a`，或 feature flag 关闭路由学习；
- 风险：路由塌缩/震荡、A 计算量变化、与 C 的块重叠变化；需监控 active 数、重叠率、A/C 互补性。

---

## 6. 需要修改的文件与函数

### 6.1 模型

| 文件 | 函数/区域 | 改动 |
|---|---|---|
| `models/model_v2.py` | `quotas` | 可选 per-sample act；O1.5 至少保持 act 标量 |
| `models/model_v2.py` | `_select` | 改为 per-sample topk；C 改为独立 topk |
| `models/model_v2.py` | `encode` | 去 `hmean.mean(0)`；per-sample 路由；分离 A/C comm/flow/side_cons/side_I |
| `models/model_v2.py` | `forward` | 输出 `chosen_a`、`chosen_c`、`active_a`、`active_c`、`active_union`；旧 `active` 仅兼容别名 |
| `models/model_v2.py` | `online_param_groups` | O1.5-C 增加 `role_a`；之前保持六组 |
| `models/model.py` | `BlockBank.comm` | 增加物理 `idx`/可选 mask；beta 用 `comm_beta[idx]`；`others`/update/out 只含有效块 |
| `models/model.py` | `BlockBank.comm` trace | 测试可选输出 `others/update/out`；生产可关闭 trace |
| `models/model.py` | `InfoFlow.forward` | 可选 mask；每层 FlowStep 维持 mask；mean/scale/agree 只统计有效块 |
| `models/model.py` | `FlowStep.forward` | 保存/传递 query/key/value/输出 mask，无效块不重新进入 |
| `models/model.py` | `BlockBank.block_forward` | 可选 union/掩码；保持块参数不变 |

### 6.2 测试

- 新增 `tests/test_online_o1_5.py`：
  - 单样本 vs batch 路由一致性；
  - batch>1 样本独立性/顺序无关；
  - A 路由变化 → C 的 `chosen_c`/`active_c`/z_c/I_c/top-k 不变；`active_union` 允许变化；
  - C 路由独立；
  - 六组/七组白名单；
  - M 写入不影响 C；
  - 旧 checkpoint 可加载。
- 保留 `tests/test_online_o1.py` 作为 O1 历史基线，不删除/不改写结论；
- 旧路径（legacy routing/comm）保留回归测试。

---

## 7. 分阶段实施计划

| 阶段 | 内容 | 先决条件 | 验收 | 失败条件 |
|---|---|---|---|---|
| O1.5-0 | 本设计评审 | O1 | 文档通过 | — |
| O1.5-A | 逐样本路由 + C 独立选块 | O1.5-0 | 单样本/batch 路由一致；样本顺序无关；`chosen_c`/`active_c` 不因 A 路由变化；共享 comm/flow 下 `z_c` 仍可能变化，只记录剩余耦合 | 同一样本路由随 batch 变化；`chosen_c`/`active_c` 随 A 路由变化 |
| O1.5-B | A/C 分别 comm/flow + 严格 mask；P0-1 beta 修正、P0-2 union 语义、P0-3 逐块独立性、P0-4 严格等价、P0-5/P0-7 beta 可区分测试、P0-6/P0-8/P0-9 mask/count/全无效安全 | O1.5-A | 六组扰动后 C 的 `settled_c`/`agree_c`/`I_c`/`side_cons_c`/`z_c` 稳定；A 路由变化时同上；M1/M2 逐层等价；非连续 beta/union/逐块/空集合测试通过 | C 任一中间/最终指标变化；mask 泄漏；P0 测试失败；NaN/除零；性能超预算 |
| O1.5-C | `role_a` 加入白名单（第七组） | O1.5-B | 七组白名单；A 路由更新后 `active_c`/C 仍稳定；`active_a` 数可解释 | C 变化；路由塌缩；吞吐异常 |
| O1.5-D | 兼容/回滚/性能收口 | O1.5-B | 旧 checkpoint 加载；legacy 路径回归；性能记录 | 旧 ckpt 失败；无法回滚 |

**O1.5 不启动训练。** 所有测试只做 forward/参数赋值/param_groups 检查。

### 7.1 O1.5-A 与 O1.5-B 的验收边界

| 项目 | O1.5-A | O1.5-B |
|---|---|---|
| `chosen_c` / `active_c` 不随 A 路由变化 | **必须通过** | 必须通过 |
| 逐样本路由 / batch 独立性 | **必须通过** | 必须通过 |
| `settled_c` / `agree_c` 稳定 | 允许仍受共享 comm/Flow 影响，只记录，不判定通过 | **必须通过** |
| `I_c` / `side_cons_c` 稳定 | 允许仍受共享 comm/Flow 影响，只记录 | **必须通过** |
| 最终 `z_c` 稳定 | 允许变化，作为剩余耦合证据 | **必须通过** |
| 严格 mask / 中间状态测试 | 不做通过要求 | **必须通过** |

- O1.5-A 只证明**路由集合**解耦，不能宣称 C 输出稳定；
- O1.5-B 完成后才允许宣称“在共享参数冻结下，C 的中间状态和最终输出稳定”。

---

## 8. 可证伪的验收测试

### 8.1 单样本 vs batch

```text
route(sample_i in batch=1) == route(sample_i in batch=N)
outputs C(sample_i in batch=1) ≈ outputs C(sample_i in batch=N)
```

- 路由集合必须相同；
- M1/M2 必须逐样本、逐块比较 `settled_a/c`、`agree_a/c`、`I_c`、`side_cons_c`、`z_c`，
  不能只比较最终输出；
- 输出容差：fp32 `atol=1e-6, rtol=1e-5`，记录浮点求和差异；
- 任一不同即失败。

### 8.2 batch>1 样本独立

- 打乱 batch 顺序，样本路由/输出按样本对应不变；
- 只在某样本上下文中扰动 A，其他样本 C 输出不变；
- 不同样本可以有不同 A/C 选块。

### 8.3 A 路由变化 → C 稳定

- 固定输入，确定性扰动 `role_a`；
- 确认 A 的选块变化；
- 检查**每个样本**的 C `chosen_c`/`active_c`/`z_c`/`I_c`/top-k 和 protected hash 不变；
- **允许 `active_union` 变化**；不得用 `active_union` 作为 C 稳定性指标；
- 如果 C 不变，记录；如果 C 变化，失败并定位到 comm/flow/mask。

### 8.4 C 路由独立

- 扰动 A 路由/参数，C 的 `chosen_c`/`active_c` 不依赖 A；
- 允许 A/C 选到同一块，但 C 的 comm/flow 只使用 `active_c` 的 mask；
- 检查 `active_union` 变化时 C 的中间 comm/flow 输出仍不变。

### 8.5 参数更新

- 六组（之后七组）逐组确定性扰动；
- C 指标不变；
- optimizer param_groups 不含 C/shared；
- 不 step、不 backward。

### 8.6 M 写入

- M 写入前后 C `chosen_c`/`active_c`/`z_c`/`I_c`/top-k 不变；
- A 的 M 读出可变化；
- batch>1 每个样本都检查。

### 8.7 严格 mask / 中间状态测试

- 固定 `chosen_c`/`active_c`，改变 A 侧 mask、A-only 块或 `active_union` 中仅供 A 使用的块：
  C 的 comm/Flow 中间输出与最终输出不得变化；
- 固定 `active_c`，构造 A/C 重叠块并扰动 A 侧该块：C 侧结果不变；
- 对 C 自身无效块：验证其不参与 C 的 `others`/`mean`/`scale`/`agree`/输出；
  不能通过关闭 `active_c` 内有效 mask 位来要求 C 整体输出不变；
- 每层 FlowStep 后，无效块输出必须为零/被排除，不能在下一层复活；
- comm 的 `update_side`/`out_side` 必须乘 `mask_side`，无效块不能接收更新；
- `agree`/`mean`/`scale` 的有效块分母正确（P0-8）；
- `count_b == 0` 的空集合走受控错误/显式零路径，不能 softmax 全 `-inf` 产生 NaN（P0-9）；
- `settled_side` 必须是 per-block 残差，不得被 `consensus_side` 替代。

### 8.8 M1/M2 严格等价测试（P0-4）

逐样本、逐块、逐层比较：

- block output；
- comm `others`/`update`/`out`；
- 每层 FlowStep 的 `h`；
- `settled_c`；
- `agree_c`；
- `I_c`；
- `side_cons_c`；
- `z_c`。

记录每层 `max_abs_err`/`max_rel_err`；任一层超容差即失败，不能只看最终 `z_c`。

非连续索引测试（P0-5）：

- 主动设置互不相同的 `comm_beta`；
- 设置非零 `comm2` 权重/偏置；
- 断言本次输入计算的 `r.abs().max() > 0`；
- `idx=[2,5,7]` 调用实际 `comm()`；
- 与按物理索引手工计算的 `expected` 使用 `assert_close`/`allclose` 比较；
- 与连续切片错误路径 `wrong` 比较，确认二者不同；
- 对 `comm` 的 `others`/`update`/`out` 分别检查物理 block 对齐；
- 不能只比较手工索引表达式。

BlockBank 逐块独立性测试：

- 单块输出与 union 中对应行一致；
- 增加其他物理块不改变已有块输出；
- 扰动 A-only 块参数不影响 C-only 块输出；
- A/C 重叠块按参数共享预期变化，但不能有跨调用中间状态污染。

---

## 9. 参数量、计算开销、兼容性与回滚

| 改动 | 参数量 | 计算开销 | 旧 checkpoint | 回滚 |
|---|---|---|---|---|
| 逐样本路由 | 0 | 路由/聚合增加 batch 维；很小 | 权重形状不变，可加载；行为变化 | `routing_mode=legacy_batch` |
| C 独立选块 | 0 | topk 不变；支持集可能增大 | 可加载；C 行为变化 | feature flag |
| A/C 分别 comm/flow | 0 | comm/flow 可能重复计算；约 +1 次/侧；需 profile | 可加载；语义变化 | `comm_mode=shared` |
| masked 聚合 | 0 | mask/权重小；可能有 padding 浪费 | 可加载 | 保留旧路径 |
| `role_a` 第七组 | 0 | `active_a` 变化，吞吐波动 | 可加载；训练行为变化 | 冻结 `role_a` |

**旧 checkpoint 只是权重可加载，不代表输出兼容。** 必须提供 legacy 路径或 architecture version 字段；
O1.5 回归测试必须同时验证 legacy 行为未被无意改变。

---

## 10. 尚未解决的风险

1. 逐样本路由会改变 C 的现有基线输出，不只是“隔离”；需要重新建立 C 基线。
2. A/C 重叠块可能降低互补性，或增加重复通讯；需要后续实验。
3. 逐样本 mask 在 batch 中实现复杂，容易出现跨样本泄漏。
4. `field` 随机扰动路径、bf16/fp16、多 seed 尚未验证。
5. `role_a` 解冻后可能出现路由塌缩/震荡；需要监控 active 数与重叠率。
6. 旧 checkpoint 行为兼容需要 feature flag + 回归测试，不能只靠 strict load。
7. O1.5 仍不包含 Runtime、M 因果写入、打断/续连；不能宣称 Online Learning；
8. O1.5 只隔离计算状态，不隔离参数；共享参数更新会使 C 稳定性结论失效；
9. 如果 `active_union` 被误用作 C 稳定性指标，会掩盖 C 路由变化；测试必须使用 `active_c`。
10. `BlockBank.comm` 当前 `comm_beta[:S]` 与物理 block ID 不对应；修正会改变现有行为，必须保留 legacy 基线并做非连续索引测试。
11. 若把 `active_union_per_sample` 直接传给 `block_forward` 的一维 idx，会产生错误调度；必须使用 global union + mask。
12. M1/M2 若只比较最终 `z_c`，会漏掉 comm/Flow 中间层泄漏；P0-4 测试必须逐层失败。
13. 默认 `comm_beta` 全相同、`comm2` 初始为零，beta 测试如果不改这两处就失去区分能力（P0-5）。
14. mask 布局不统一会引入转置错误；P0-6 要求唯一 `(G,B)` 布局并固定广播规则。
15. 即使 beta 互异，若本次输入下 `r` 为零，P0-5 测试仍会失去区分能力；必须断言 `r.abs().max() > 0`。
16. `count_b == 0` 或全无效 key 时直接 softmax 会 NaN；P0-8/P0-9 要求显式 count、断言和受控路径。

---

## 11. 本轮边界

- 只提交本设计文档；
- 不修改模型代码、测试或训练脚本；
- 不运行训练/loss/backward/optimizer.step/smoke；
- 不把 O1 的 C 稳定性结论扩大为 A/C 完全隔离；
- 不把 O1.5 的“计算状态隔离”写成“参数独立/完全独立”；
- O1.5 仍未验证共享参数更新、随机 field、bf16、多 seed；
- 保留 O1 与 Phase 1 的历史结果/基线，不覆盖原始结果；
- O1.5 只解决计算路径隔离，不借机改变其他模型机制；
- 不进入 O2 或 Online Learning。

### 11.1 允许与禁止的验收结论

**允许写：**

> 在共享参数冻结条件下，O1.5 已实现 A/C 路由与计算状态隔离；
> C 的 `active_c` 和输出在 A 路由变化时保持稳定。

**禁止写：**

- “A/C 参数已独立”；
- “A/C 完全独立”；
- “共享参数可安全在线更新”；
- “Online Learning 已实现”；
- “`active_union` 不变就代表 C 稳定”。

O1.5 的 C 稳定性只覆盖 A/C 路由与计算状态；若未来允许 BlockBank/comm/Flow 的共享参数更新，
必须重新设计 C 稳定性验证，不能沿用 O1.5 结论。
