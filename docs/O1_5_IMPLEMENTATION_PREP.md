# O1.5 最小实现准备：代码现状核查与修改计划

状态：**准备阶段文档。只读代码核查 + 计划；未修改模型/测试/训练代码，未运行 forward/backward/smoke/训练。**
目标：为 P0-11/P0-12 的最小代码实现提供真实代码位置、问题清单、修改边界和测试计划。

---

## 1. 真实文件清单

| 文件 | 作用 |
|---|---|
| `models/model.py` | 真实 `BlockBank`、`FlowStep`、`InfoFlow`、v15 PBSLM |
| `models/model_v2.py` | v2 A/C 路径与 `flow/comm` 调用 |
| `models/model_v21.py` | Phase 1 baseline；不调用 `comm/flow` |
| `tests/` | 现有 O1 测试；目前没有 P0-11/P0-12 测试 |
| `experiments/legacy/code/run_online.py` | legacy v15 在线入口 |

---

## 2. 真实代码位置与当前形状

### 2.1 `models/model.py::BlockBank.comm`（121–127）

```python
def comm(self, Y, g):
    w = g[:, None, None, None]
    tot = (w * Y).sum(0)
    others = tot.unsqueeze(0) - w * Y
    r = self.comm2(F.gelu(self.comm1(others)))
    beta = torch.sigmoid(self.comm_beta[: Y.shape[0]])[:, None, None, None]
    return Y + beta * w * r
```

- 输入：`Y (S,B,T,D)`，`g (S,)`；
- 无 `idx`，无 mask；
- `comm_beta[:S]` 使用连续切片，而不是实际物理 block ID；
- 无效块/NaN 直接进入 `comm1`；
- `beta * w * r` 只乘，不能清除 NaN。

### 2.2 `models/model.py::BlockBank.block_forward`（92–119）

- 输入 `x (B,T,D)`，`idx` 为一维物理块 ID；
- 输出 `(S,B,T,D)`；
- 每个块计算独立，不保存跨调用状态；
- **不接受逐样本 idx**，所有样本使用同一组块。

### 2.3 `models/model.py::FlowStep`（174–188）

```python
def forward(self, h):                 # h: (S,B,T,f)
    q, k, v = self.q(h), self.k(h), self.v(h)
    att = softmax(einsum("ibtd,jbtd->btij", q, k) / sqrt(f), -1)
    msg = einsum("btij,jbtd->ibtd", att, v)
    return self.ln(h + self.o(msg))
```

- 无 mask 参数；
- `q/k/v` 直接接收全部 `h`；
- attention logits 是 `(B,T,S,S)`，最后一维是 key block；
- 无 key/query mask；
- 无效块 NaN 可在 Q/K/V、attention、V matmul、LayerNorm 中传播；
- 返回 `(S,B,T,f)`。

### 2.4 `models/model.py::InfoFlow`（191–209）

```python
def forward(self, delta):
    h = self.proj_in(delta)                    # (S,B,T,f)
    for layer in self.layers: h=layer(h)
    mean = h.mean(0)                           # (B,T,f)
    scale = h.norm(dim=-1).mean(0) + 1e-6      # (B,T)
    agree = 1 - ((h-mean.unsqueeze(0)).norm(dim=-1)/scale.unsqueeze(0)).clamp(0,2)
    cons = self.proj_out(mean)                 # (B,T,D)
    return h, agree, cons
```

- 无 mask；
- `proj_in(delta)` 在清理之前；
- `mean/scale/agree` 使用全部 S 块；
- `agree` 是逐块 `(S,B,T)`；
- `cons` 是 `proj_out(mean)`；
- `h` 是低维 per-block 状态。

### 2.5 `models/model_v2.py` 调用

- `e_forward()` 第 113–122 行：`ebank.comm(E_out, ones)`；
- `encode()` 第 132–173 行：
  - 第 146 行：`delta = self.bank.comm(delta, ones(len(active)))`；
  - 第 148–149 行：`settled_low, agree, _ = self.flow(delta)`，
    `settled = delta + self.flow.proj_out(settled_low)`；
  - `flow` 调用无 mask；
  - 当前 routing/active 为 batch 共享。

### 2.6 v15/legacy 调用

- `models/model.py::PBSLM.encode` 第 355–400 行：
  - 第 381 行 `bank.comm(delta, ones)`；
  - 第 382 行 `settled, agree, _ = self.flow(delta)`；
  - 第 391 行 `flow.proj_out((settled[rows]*w).sum(0))`。
- legacy `run_online.py` 使用 v15 PBSLM；
- **v15 的 settled 语义与 v2 不同**：v15 的 `settled` 是低维 `h`，再投影聚合；
  v2 是 `delta + proj_out(h)` 的 per-block 残差。实现时必须保留。

### 2.7 Phase 1 baseline

- `models/model_v21.py` 不调用 `BlockBank.comm` 和 `InfoFlow`；
- `training/train_v21.py` 不涉及这些 mask 路径；
- Phase 1 结果不应受可选参数影响。

---

## 3. 具体问题清单

| # | 问题 | 真实位置 |
|---|---|---|
| 1 | `FlowStep/InfoFlow/comm` 没有 mask 参数 | `model.py:121-127,174-209` |
| 2 | `comm_beta[:S]` 与物理 block ID 不对应 | `model.py:126` |
| 3 | `comm` 的 `others/update/out` 没有 NaN-safe where | `model.py:121-127` |
| 4 | `proj_in(delta)` 在清理前执行 | `model.py:202` |
| 5 | `proj_out` bias 后没有再次清理 | `model.py:208` |
| 6 | FlowStep Q/K/V 无 mask/query 清理 | `model.py:183-188` |
| 7 | attention 无 key mask、无全无效 key 保护 | `model.py:186` |
| 8 | `mean/scale/agree` 使用全部块，无 count/where | `model.py:205-207` |
| 9 | `block_forward` 只接受一维 idx，不能直接传逐样本集合 | `model.py:92-119` |
| 10 | v2/v15 调用都不传 idx/mask；legacy 行为必须保持 | `model_v2.py:146,148`; `model.py:381-382` |
| 11 | 没有有效块 NaN/Inf 的显式边界检查 | `model.py` Flow/comm |
| 12 | 没有 P0-11/P0-12 测试 | `tests/` |
| 13 | `block_forward` 输入 NaN 无法由 block mask 隔离；同一 x 会污染所有块 | `model.py:92-119` |
| 14 | mask 形状/时间维/idx 对齐契约未冻结 | 计划接口 |
| 15 | `comm` 新路径只提供 idx 或只提供 mask 的非法组合未定义 | 计划接口 |
| 16 | M1 等价测试尚未逐层列出全部比较对象 | 计划测试 |

---

## 3.1 待冻结的接口契约（先于代码实施）

### 3.1.1 NaN-safe 保护边界

block mask 只能清理**已经是 block-level 张量**的无效块状态，不能修复
`block_forward` 输入本身的 NaN：

- `block_forward` 的输入 `x` 对同一样本的所有块是共享的；
- 如果 `x` 含 NaN，LayerNorm/Q/K/V/Attention/FFN/残差会先产生 NaN Y，
  之后再 `where(mask, Y, 0)` 已经无法恢复；
- 因此把无效块 NaN 注入 `block_forward` 的输入，不在 block mask 的保护范围内。

冻结规则：

1. 进入 `block_forward` 前，必须检查参与有效计算的输入 x/token 是否有限；
   该检查是 token/input-level，不是 block-level mask；
2. block-level NaN 测试应在 `block_forward` **之后**向 `Y`/`delta` 的无效行注入 NaN，
   再验证 comm/Flow 的 `where` 能隔离；
3. 若输入 x 已经含 NaN，走受控错误路径，不得声称 block mask 已处理；
4. P0-11 的保护范围要写成：
   “无效 block 的 Y/delta 及其下游 comm/Flow 被隔离”，
   而不是“任意输入 NaN 被 mask 隔离”。

### 3.1.2 mask 形状契约

O1.5 冻结为：

```text
mask_side: (G,B) bool
idx:       (G,) int64 physical block ID
```

- mask 在 O1.5 中**不随时间 T 变化**；暂不支持 `(G,B,T)`；
- `mask_a`、`mask_c` 都使用 `(G,B)`；
- `idx[g]` 与 `mask_side[g, :]` 严格一一对应；
- `len(idx) == mask_side.shape[0] == Y.shape[0] == G`，必须断言；
- `active_union_global` 的顺序就是 `idx` 的顺序；A/C 共用同一 G；
- Flow key mask 由 `(G,B)` 转成 `(B,1,1,G)`；
- 若未来需要 per-time mask，必须新版本接口，不能悄悄改变形状。

### 3.1.3 `BlockBank.comm` 新旧路径契约

只允许两种组合：

| idx | mask | 行为 |
|---|---|---|
| `None` | `None` | legacy 路径，完全保持旧行为 |
| 非空 | 非空 | 新路径：`beta[idx]` + NaN-safe mask |

- `idx=None, mask!=None`：必须抛 `ValueError`，不得静默使用旧 beta 切片；
- `idx!=None, mask=None`：必须抛 `ValueError`，不得只修 beta 却漏掉清理；
- 新路径断言 `len(idx)==Y.shape[0]==mask.shape[0]`；
- 旧路径必须逐位回归，不能为了“统一实现”顺手重写。

新路径必须保留原始 `g` 门控关系，不能因为当前部分调用 `g=ones` 就删除 `w`：

```text
w        = g[:, None, None, None]              # (G,1,1,1)
mask4    = mask_side[:, :, None, None]         # (G,B,1,1)
valid_Y  = where(mask4, Y, 0)                  # 先清理
weighted_Y = w * valid_Y
tot      = weighted_Y.sum(0)                   # (B,T,D)
others   = where(mask4, tot.unsqueeze(0) - weighted_Y, 0)
r        = comm2(gelu(comm1(others)))
update   = where(mask4, beta * w * r, 0)       # beta 与 w 都不能丢
out      = where(mask4, valid_Y + update, 0)
```

- `tot` 必须用 `w * valid_Y`；
- `others` 必须减 `weighted_Y`；
- `update` 必须保留 `beta * w * r`；
- 不得因为 `g` 当前全为 1 就删除 `w`。

### 3.1.4 M1 严格等价比对清单

不能只比较最终输出。逐样本、逐块、逐层比较：

1. `BlockBank.comm` 输出；
2. `proj_in` 输出；
3. 每层 Q/K/V；
4. Attention logits；
5. Attention weights；
6. 每层 msg；
7. 每层 FlowStep 输出；
8. masked `mean`、`scale`、`agree`；
9. `proj_out` 输出；
10. 最终输出。

必须覆盖 batch 内不同有效块集合、非连续物理块 ID，并记录每层最大误差。

### 3.1.5 InfoFlow masked 统计公式

冻结以下逐样本公式（`g` 为物理 block 索引，`G=len(idx)`）：

```text
count_b = mask_side.sum(0)                         # (B,)
# 正常路径先断言 count_b >= 1；任一样本为 0 必须受控报错

mask4     = mask_side[:, :, None, None]             # (G,B,1,1)
valid_h   = where(mask4, h, 0)                      # (G,B,T,f) 先清理
tot       = valid_h.sum(0)                          # (B,T,f)
mean_b    = tot / count_b[:, None, None]            # (B,T,f)

valid_norm = where(mask_side[:, :, None], valid_h.norm(dim=-1), 0)
scale_b    = valid_norm.sum(0) / count_b[:, None] + 1e-6  # (B,T)

# 无效位置的 h 和 mean 都先置零，再计算 centered/norm
safe_mean = where(mask4, mean_b.unsqueeze(0), 0)     # (G,B,T,f)
centered  = valid_h - safe_mean                      # 无效位置严格为 0
centered_norm = centered.norm(dim=-1)
agree_raw = 1 - (centered_norm / scale_b.unsqueeze(0)).clamp(0, 2)
agree     = where(mask_side[:, :, None], agree_raw, 0)   # (G,B,T)
```

约束：

- `valid_h` 必须在任何减法/`norm`/统计之前构造，避免无效 NaN 先进入运算；
- `count_b` 必须逐样本计算，不能只对整个 batch 判断一次；
- `count_b == 0` 的正常路径必须受控报错，不得继续除零；
- `agree` 是逐块 `(G,B,T)`；无效位置 `valid_h`、`safe_mean`、`centered` 均为 0，
  norm 后才 where，无效块严格为 0；
- `I_side` 的聚合若需要，必须单独使用：
  ```text
  I_side[b] = where(mask_side[:, b], agree[:, b], 0).sum(0) / count_b[b]
  ```
- `scale_b` 的定义必须与 M1 参考实现完全一致（当前 Flow 的 `h.norm(dim=-1).mean(0)+1e-6` 语义）；
- FlowStep 的全无效 key 检查必须逐样本执行，不能只对整个 batch 做一次判断。

---

## 4. 最小修改计划（后续实施，不在本轮执行）

### 4.0 实施顺序

1. 先冻结 3.1 的 mask/NaN 接口契约；
2. 实现 `BlockBank.comm` 新路径与 legacy 回归；
3. 实现 `FlowStep` P0-12 维度链；
4. 实现 `InfoFlow` 的 proj/清理/统计；
5. 新增诊断测试（forward/backward/M1 逐层等价），不启动训练。

### 4.1 `models/model.py`

1. **`BlockBank.comm`** 增加可选 `idx=None, mask=None`：
   - 仅 `(idx=None, mask=None)` 走旧路径，保持 legacy 行为（包括旧 beta 切片）；
   - 仅 `(idx!=None, mask!=None)` 走新路径：`beta=sigmoid(comm_beta[idx])`，
     并按 3.1.3 的公式保留 `w` 门控：
     `valid_Y=where(mask4,Y,0)`、`weighted_Y=w*valid_Y`、`tot=weighted_Y.sum(0)`、
     `others=where(mask4,tot.unsqueeze(0)-weighted_Y,0)`、
     `update=where(mask4,beta*w*r,0)`、`out=where(mask4,valid_Y+update,0)`；
   - 只提供一个参数：抛 `ValueError`；
   - 断言 `len(idx)==mask.shape[0]==Y.shape[0]`；
   - 不修改参数、不新增参数。
2. **`FlowStep.forward(h, mask=None)`**：
   - `mask=None`：完全保持旧 einsum 路径；
   - `mask` 非空：按 P0-12 维度链，`h_in=where`、`q/k/v=linear(h_in)` 后 where、
     Q/K/V permute、`K^T`、logits、`(B,1,1,G)` key mask、softmax(-1)、
     V matmul 前清理、msg permute 回 `(G,B,T,f)`、query mask、每层输出 where；
   - 不做结构/参数改变。
3. **`InfoFlow.forward(delta, mask=None)`**：
   - `mask=None`：旧路径；
   - `mask` 非空：`delta_in=where`、`proj_in`、where、FlowStep(mask)、
     按 3.1.5 计算 masked count/mean/scale/agree、`proj_out` 前后清理、`settled` 输出；
   - 保留 `return h, agree, cons` 的旧返回形状/语义。
4. 不修改 `BlockBank.block_forward` 签名；M2 在调用侧使用 global union。

### 4.2 `models/model_v2.py`（只作为后续计划，不在最小 patch 首步修改）

- 在 O1.5-A/B 阶段再传 `idx/mask`；
- 保留 `settled = delta + proj_out(h)` 的 v2 语义；
- 不改变 `out_a/out_c` 或参数白名单。

### 4.3 明确不修改

- 不增加/删除模型参数；
- 不改变 checkpoint 字段；
- 不改变训练目标/优化器/训练脚本；
- 不改变 O1/Phase 1 历史结果；
- v15/legacy 的 `mask=None` 路径必须逐位回归。

---

## 5. Checkpoint 与 legacy 影响

- 可选参数不进入 `state_dict`，不改变 checkpoint 格式；
- 旧 v15/v2 checkpoint 应能 strict load；
- 仅当调用方显式传 `idx/mask` 才启用新路径；
- v15/legacy/Phase 1 默认调用不传 mask，必须与旧输出一致；
- O1 的 `out_a/out_c` 和 Phase 1 基线不触碰。

---

## 6. 测试计划

新增 `tests/test_online_o1_5_nan_attention.py`（后续实施）：

1. batch=1；
2. batch>1；
3. G>1；
4. 每个样本不同有效块集合；
5. 非连续物理块索引；
6. 无效 key 不参与 Attention；
7. 无效 query 输出严格为零；
8. 全无效 key / count_b==0 逐样本受控报错，不得只做 batch 级判断；
9. **block_forward 之后**向无效块 `Y/delta` 注入 NaN，验证 comm/Flow 隔离；
10. 输入 x/token 注入 NaN，在 `block_forward` 前受控报错（不能声称由 block mask 处理）；
11. 有效块 `Y/delta` 注入 NaN，在指定边界抛出受控错误；
12. forward 输出 `isfinite`；
13. diagnostic backward：有效块梯度、模型参数梯度有限；
14. 批量实现与逐样本 M1 参考逐层 `allclose`，比较 3.1.4 的全部 10 个对象。

每条测试必须记录：

- 张量形状；
- `max_abs_err` / 容差；
- `isfinite` 布尔；
- 错误触发位置；
- 不调用 optimizer.step / 不训练。

---

## 7. 本轮边界

- 只提交本准备文档；
- 不修改模型代码、测试代码、训练脚本；
- 不运行 forward/backward/smoke/loss/训练；
- 不启动 O1.5-A/B/C；
- 不改变 O1/Phase 1 历史基线；
- 不宣布 P0-11、P0-12 或 O1.5-0 通过。
