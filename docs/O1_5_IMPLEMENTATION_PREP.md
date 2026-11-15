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

---

## 4. 最小修改计划（后续实施，不在本轮执行）

### 4.1 `models/model.py`

1. **`BlockBank.comm`** 增加可选 `idx=None, mask=None`：
   - `idx is None and mask is None`：完全走旧路径，保持 legacy 行为（包括旧 beta 切片）；
   - 传入 `idx`：`beta = sigmoid(comm_beta[idx])`；
   - 传入 `mask`：`valid_Y=where(mask,Y,0)`、`others=where(mask, tot.unsqueeze(0)-valid_Y,0)`、
     `update=where(mask,beta*r,0)`、`out=where(mask,valid_Y+update,0)`；
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
     masked count/mean/scale/agree、`proj_out` 前后清理、`settled` 输出；
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
8. 全无效 key 逐样本受控报错；
9. 无效块输入注入 NaN；
10. 有效块输入注入 NaN，在指定边界抛出受控错误；
11. forward 输出 `isfinite`；
12. diagnostic backward：有效块梯度、模型参数梯度有限；
13. 批量实现与逐样本 M1 参考逐层 `allclose`。

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
