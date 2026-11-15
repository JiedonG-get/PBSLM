# ONLINE_ARCHITECTURE_O1_5_DESIGN — A/C 路由与跨块信息流解耦设计

状态：**O1.5 架构分析文档。只做设计，不修改模型代码，不运行训练/loss/backward/optimizer.step/smoke。**
基线：O1 提交 `ad99bc1`；O1 只证明了**输出头已拆分**、六组 A 参数白名单和基础 C 稳定性，
**不代表 A/C 计算路径已经完全隔离，更不代表 Online Learning 已实现。**

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

1. A 的路由选择变化不再改变 C 的 active blocks、通讯路径和输出；
2. 路由逐样本独立，不因 batch 组成变化；
3. A/C 的 active set、`bank.comm`、`flow`、`side_cons` 逻辑隔离；
4. 保持 C 的稳定性和旧 checkpoint 可加载；
5. O1.5 仍不实现 Online Runtime，只做架构隔离。

### 2.2 目标数据流

```text
x = token_emb(ids)                              # (B,T,D)
h_b = x.mean(1)                                 # (B,D) ← 每个样本独立
sa = role_a(h_b)                                # (B,k)
sc = role_c(h_b)                                # (B,k)

ia_b = topk(sa, kA)                             # (B,kA) 每样本独立
ic_b = topk(sc, kC)                             # (B,kC) 每样本独立，不排除 A
# 允许 A/C 选到同一块；是否强制不重叠需要单独实验

Y = bank.block_forward(x, union(ia_b, ic_b))    # 块输出本身按块独立
delta = Y - x

# A 与 C 分别计算通讯/Flow，互不进入对方的块集合与 mask
settled_a, agree_a = side_comm_flow(delta, ia_b, role=A)
settled_c, agree_c = side_comm_flow(delta, ic_b, role=C)

side_cons_a = aggregate(settled_a, weights(sa, ia_b))
side_cons_c = aggregate(settled_c, weights(sc, ic_b))

I_a = mean(agree_a[selected A rows])
I_c = mean(agree_c[selected C rows])

a_enc = ln_a(x + side_cons_a + ea_proj(mem_read))
c_enc = ln_c(x + side_cons_c)
z_a = out_a(dec_a(...)) + ctx_a
z_c = out_c(dec_c(...)) + ctx_c
```

关键点：

- C 的选块**不再排除 A**；C 不感知 A 的选择；
- active 集合是逐样本的，不再是全局 `list`；
- `bank.comm`/`flow` 分别只看 A 或 C 的块集合；
- C 的 `settled`/`agree`/`I_c` 不含 A 块；
- C 不读 M，E 块参数冻结。

### 2.3 计算形式：masked 还是 per-sample

逐样本选块意味着不同样本的 A/C 支持集不同，`bank.comm`/`flow` 不能直接用全局 active list。
备选实现：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| M1 | 对每个样本单独调用 `bank.comm`/`flow` | 语义最清楚，易验证 | batch 循环，吞吐差 |
| M2 | 全体块 + per-sample mask | 可批量，语义可控 | 需要给 `BlockBank.comm`/`InfoFlow` 增加 mask 接口 |
| M3 | 按支持集分组批量 | 性能折中 | 分组/去重复杂，调试难 |

**建议 O1.5 先用 M1 做正确性基准，再用 M2 做批量化**；两者结果必须在单样本/小 batch 上一致。

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
| `chosen/active` | 从两个 list 变为 batch 级张量/掩码 |
| `bank.block_forward` | 块输出本身独立；需要支持按样本 union/mask |
| `bank.comm` | 需要 per-sample mask/权重，只在各侧集合内聚合 |
| `InfoFlow` | 需要 per-sample mask/权重，分别算 A/C |
| `side_cons` | 权重从 `(k,)` 变 `(B,k)`，聚合按样本 |
| `side_I` | 按样本的 side agree |
| `forward` 输出 | `chosen/active` 语义变化；测试需更新 |

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
| `models/model_v2.py` | `forward` | `chosen/active` 输出语义变更；兼容旧输出字段 |
| `models/model_v2.py` | `online_param_groups` | O1.5-C 增加 `role_a`；之前保持六组 |
| `models/model.py` | `BlockBank.comm` | 可选 per-sample mask/权重；支持 A/C 独立集合 |
| `models/model.py` | `InfoFlow.forward` | 可选 mask/权重；A/C 分别调用 |
| `models/model.py` | `BlockBank.block_forward` | 可选 union/掩码；保持块参数不变 |

### 6.2 测试

- 新增 `tests/test_online_o1_5.py`：
  - 单样本 vs batch 路由一致性；
  - batch>1 样本独立性/顺序无关；
  - A 路由变化 → C 选块/active/z_c/I_c/top-k 不变；
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
| O1.5-A | 逐样本路由 + C 独立选块 | O1.5-0 | 单样本/batch 路由一致；样本顺序无关；C 不因 A 选块变化（共享 comm 下允许失败，用来暴露剩余耦合） | 同一样本路由随 batch 变化；或测试无法覆盖 |
| O1.5-B | A/C 分别 comm/flow + masked 聚合 | O1.5-A | 六组扰动 C 稳定；A 路由变化 C 稳定；单 sample/batch>1 | C 任一指标变化；mask 泄漏；性能超预算 |
| O1.5-C | `role_a` 加入白名单（第七组） | O1.5-B | 七组白名单；A 路由更新后 C 仍稳定；A active 数可解释 | C 变化；路由塌缩；吞吐异常 |
| O1.5-D | 兼容/回滚/性能收口 | O1.5-B | 旧 checkpoint 加载；legacy 路径回归；性能记录 | 旧 ckpt 失败；无法回滚 |

**O1.5 不启动训练。** 所有测试只做 forward/参数赋值/param_groups 检查。

---

## 8. 可证伪的验收测试

### 8.1 单样本 vs batch

```text
route(sample_i in batch=1) == route(sample_i in batch=N)
outputs C(sample_i in batch=1) ≈ outputs C(sample_i in batch=N)
```

- 路由集合必须相同；
- 输出容差：fp32 `atol=1e-6, rtol=1e-5`，记录浮点求和差异；
- 任一不同即失败。

### 8.2 batch>1 样本独立

- 打乱 batch 顺序，样本路由/输出按样本对应不变；
- 只在某样本上下文中扰动 A，其他样本 C 输出不变；
- 不同样本可以有不同 A/C 选块。

### 8.3 A 路由变化 → C 稳定

- 固定输入，确定性扰动 `role_a`；
- 确认 A 的选块变化；
- 检查**每个样本**的 C chosen/active/`z_c`/`I_c`/top-k 和 protected hash 不变；
- 如果 C 不变，记录；如果 C 变化，失败并定位到 comm/flow/mask。

### 8.4 C 路由独立

- 扰动 A 路由/参数，C 的选块不依赖 A；
- 允许 A/C 选到同一块，但 C 的通讯集合只含 C 选块。

### 8.5 参数更新

- 六组（之后七组）逐组确定性扰动；
- C 指标不变；
- optimizer param_groups 不含 C/shared；
- 不 step、不 backward。

### 8.6 M 写入

- M 写入前后 C chosen/active/`z_c`/`I_c`/top-k 不变；
- A 的 M 读出可变化；
- batch>1 每个样本都检查。

---

## 9. 参数量、计算开销、兼容性与回滚

| 改动 | 参数量 | 计算开销 | 旧 checkpoint | 回滚 |
|---|---|---|---|---|
| 逐样本路由 | 0 | 路由/聚合增加 batch 维；很小 | 权重形状不变，可加载；行为变化 | `routing_mode=legacy_batch` |
| C 独立选块 | 0 | topk 不变；支持集可能增大 | 可加载；C 行为变化 | feature flag |
| A/C 分别 comm/flow | 0 | comm/flow 可能重复计算；约 +1 次/侧；需 profile | 可加载；语义变化 | `comm_mode=shared` |
| masked 聚合 | 0 | mask/权重小；可能有 padding 浪费 | 可加载 | 保留旧路径 |
| `role_a` 第七组 | 0 | A active 集合变化，吞吐波动 | 可加载；训练行为变化 | 冻结 `role_a` |

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
7. O1.5 仍不包含 Runtime、M 因果写入、打断/续连；不能宣称 Online Learning。

---

## 11. 本轮边界

- 只提交本设计文档；
- 不修改模型代码、测试或训练脚本；
- 不运行训练/loss/backward/optimizer.step/smoke；
- 不把 O1 的 C 稳定性结论扩大为 A/C 完全隔离；
- 不进入 O2 或 Online Learning。
