# ONLINE_ARCHITECTURE_DESIGN — PBSLM Online Architecture Design v1

状态：**设计文档，基于只读代码审查。没有修改代码，没有运行前向/smoke/训练。**
适用范围：PBSLM v2（`models/model_v2.py`）与 v15/legacy 在线雏形（`models/model.py`、`experiments/legacy/code/run_online.py`）。

---

## 1. 设计目标与边界

PBSLM 要验证的不是「一个离线 LM 的准确率」，而是：

> 稳定知识基础（C / 主块 / Flow）、持续变化的学习状态（A）、外部记忆（M）
> 能否在同一个实时运行系统中协作，并支持持续输入、选择性输出、打断与续连。

本设计只做架构与接口定义；**不把 Runtime 可运行等同于 Online Learning 成功**。
两条验证线分开：

- **架构验证线**：A/C 隔离、codebook 对照、A/C 互补性、InfoFlow、E/M；
- **运行时验证线**：连续输入、提交、静默、打断/续连、状态保存恢复。

---

## 2. 现状分类（已实现 / 部分实现 / 仅有设计 / 完全缺失）

| 能力 | 状态 | 证据 |
|---|---|---|
| A/C 双路径 forward | **已实现** | `models/model_v2.py::encode/forward` |
| C 不读 M | **已实现（forward 层）** | `c_enc` 不接 `mem_read` |
| A/C 输出层独立 | **缺失**：`z_a`/`z_c` 共用 `out_a` | `models/model_v2.py:205` |
| 在线参数白名单/冻结 | **v2 缺失**；v15 部分实现且有 bug | `models/model_v2.py` 无 `online_params`；`models/model.py:510` `online_params_gate` 引用不存在的 `self.role` |
| E/M 读写/衰减 | **已实现** | `models/model_v2.py:107-124, 226-240` |
| M 因果写入/回滚 | **完全缺失** | 当前 `write_memory` 为 `no_grad`、外部传 values，训练用目标 `y` 写；`training/train_v2.py:311` |
| InfoFlow | **已实现** | `models/model.py::InfoFlow` |
| PerturbField | **已实现（可关）** | `models/model_v2.py:183-223`；`field_scale=0` 时不影响 z，但 KL 仍可训练 |
| 动态 activity | **v2 缺失**；v15 legacy `online_stream` 有雏形 | `run_online.py:415-505` |
| 输出仲裁/静默 | **v2 缺失**；v15 有 `g/arb` 训练 logits | `models/model.py:433-451` |
| 运行时状态/commit/interrupt/resume | **完全缺失** | v15 `state` 仅 `act`，forward 返回 `"state": {}`；legacy 报告有 interrupt 实验但当前代码无 runtime hook |
| 在线 A 更新 | **v2 缺失**；v15 legacy 有 chunk 级更新 | `run_online.py::online_stream`，不是实时 runtime |
| C 稳定性自动测试 | **缺失** | 无测试文件 |
| Online 验收测试 | **仅有设计** | `docs/ONLINE.md` 的验收段落 |

---

## 3. A/C 输出投影拆分

### 3.1 当前问题

`models/model_v2.py`：

```python
self.out_a = nn.Linear(d, d)
...
z_a = F.normalize(self.out_a(h_a) + ctx_a, dim=-1)
z_c = F.normalize(self.out_a(h_c) + ctx_c, dim=-1)   # C 使用同一个投影
```

因此在线更新 A 的 `out_a` 会同时改变 C 的输出，C 不再是稳定基线。

### 3.2 目标设计

新增独立的 `out_c`，且迁移时保持 C 的当前输出不变：

```text
out_a: A 专属，可在线更新
out_c: C 专属，初始化时复制旧 out_a 的权重/偏置，之后冻结
```

迁移步骤：

1. 保存旧 `out_a.state_dict()`；
2. 创建 `out_c`，加载旧 `out_a` 权重；
3. `z_a = normalize(out_a(h_a) + ctx_a)`；
4. `z_c = normalize(out_c(h_c) + ctx_c)`；
5. `out_c` 永久 `requires_grad=False`，不进入任何优化器。

### 3.3 其他共享参数的影响边界

| 共享部分 | A 在线更新时 | C 稳定性 |
|---|---|---|
| token embedding `tok` | **冻结** | C 输入不变 |
| BlockBank `bank` | **冻结** | C 的底层块不变 |
| InfoFlow `flow` | **冻结** | C 的 side_cons/一致性不变 |
| E 块 `ebank/e_mix/ea_proj` | **冻结**（只写 M） | C 不读 E/M，C 输出不受影响 |
| 输出码本 `code` | **冻结** | 解码空间不变 |
| `out_a` / `out_c` | A 可更新，C 冻结 | 拆分后 C 稳定 |
| `field` | 冻结 | C 不经过 field（field 只作用 z_a） |
| `mem` 缓冲区 | 只写，不训练 | C 不读 M |
| `bank.comm1/comm2/comm_beta` | **不得加入在线优化器** | v2 body 不调用 bank.comm，属于未使用参数 |
| `a_pred` | 仅辅助 loss，不在 forward | 默认排除在线更新集合 |

结论：**必须先把共享 `out_a` 拆开，否则 C 稳定性无法成立。**

---

## 4. Online 参数白名单（按实际 forward 使用核对）

图例：`F` = 参与 forward；`L` = 仅参与训练 loss；`U` = v2 中未使用/无效。

| 参数组 | 来源 | 使用 | Online 建议 |
|---|---|---|---|
| `role_a.weight` | `role_a` | F | **允许更新**（A 路由） |
| `dec_a.*` | `dec_a` | F | **允许更新**（A 解码器） |
| `q_emb`, `q_pos` | A 查询 | F | **允许更新** |
| `refine_proj.*` | A refine | F | **允许更新** |
| `ln_a.*` | A 编码归一化 | F | **允许更新** |
| `ctx_a.*` | A 直接读出 | F | **允许更新** |
| `out_a.*` | A 输出 | F | **允许更新（拆分后）** |
| `a_pred.*` | 辅助预测 | L（不在 forward） | **默认冻结**；除非单独启用辅助目标 |
| `role_c`, `dec_c`, `q_emb_c/q_pos_c`, `refine_proj_c`, `ln_c`, `ctx_c`, `out_c` | C 专属 | F | **冻结** |
| `tok.weight` | 共享输入 | F | **冻结** |
| `bank.*`（qkv/out/ff/ln/amp/phase） | 共享 BlockBank | F | **冻结** |
| `bank.comm1/comm2/comm_beta` | BlockBank comm | **U（v2 encode 不调用 bank.comm）** | **不得列入更新集合** |
| `flow.*` | InfoFlow | F | **冻结** |
| `ebank.*`（含 ebank.comm） | E 块 | F | **冻结**（在线只写 M，不改 E 参数） |
| `e_mix`, `ea_proj` | E 混合/A 记忆读出 | F | **冻结**（E 策略后续单独评审） |
| `mem[i]` | 记忆缓冲区 | F（读），写由 no_grad | **只写，不训练**，不进入优化器 |
| `code` | 冻结码本 buffer | F（目标/解码） | **冻结** |
| `field.*` | 扰动场 | F（`field_scale>0` 才影响 z；KL 仍用） | **冻结**；后训练阶段再启用 |
| `emo_calm` | buffer | F | 冻结 |

注意：
- `a_pred` 不在 forward，不能当作有效在线更新参数；
- `bank.comm*` 在 v2 是未使用参数，不能因为 `requires_grad=True` 就列入白名单；
- 参数白名单必须由 **forward 调用图 + requires_grad + optimizer 实际 param_groups** 三方校验，不能只看配置。

---

## 5. C 稳定性自动测试

### 5.1 不变量

对同一批输入、同一 seed、`model.eval()`：

1. 更新 A 专属参数前后，**C 的输出向量、top-k 解码、I_c 完全一致**；
2. 所有共享参数（tok/bank/flow/code/ebank/e_mix/ea_proj/out_c）的 state hash 不变；
3. C 路径不读 M：M 改变后 C 输出不变（已有设计，但需测试）；
4. C 不进入任何 optimizer 的 param_groups。

### 5.2 测试步骤

```text
1. 构造/加载 v2.1 或 v2 模型，freeze 按白名单；
2. 固定 eval 输入 batch（多个上下文、多个 step）；
3. 记录 C 输出 z_c、top1/top5、I_c 的全量 hash；
4. 记录所有 shared/C 参数的 state hash；
5. 对 A 参数做一次模拟 optimizer.step()（随机或小目标）；
6. 重新前向同一输入；
7. 断言 C 输出全等/在容差内、shared/C hash 全等、optimizer param_groups 不含 C/shared。
```

容差：优先要求 bitwise equal；GPU 非确定性算子下允许 `torch.allclose(atol=1e-6, rtol=1e-5)`，但必须记录为何需要容差。

### 5.3 测试矩阵

- dtype：fp32 / bf16（分别）；
- batch>1；n_in 固定 16；
- 有/无 field（field 关闭时 C 不经过 field，仍要测）；
- M 全零 / M 已写入两种状态；
- 更新前后 + 打断/恢复一次；
- 不同 seed 下非确定性检查（记录容差）。

---

## 6. M 写入因果流程

### 6.1 五类信息必须分开

| 来源 | 在线时是否可用 | 是否允许写 M |
|---|---|---|
| 观察输入 `x_t` | 当时可用 | 只可作为 key，不能作为未来 value |
| 模型预测 `pred_t` | 当时可用 | 可作为候选 value，需 commit/gate |
| 已提交输出 `commit_t` | commit 后可用 | **主写入来源** |
| 用户反馈 `feedback_t` | 反馈到达后可用 | **优先写入**，可修正/补偿 |
| 外部验证 `verified_t` | 外部返回后可用 | 最高优先级 |

禁止：
- 在线用 teacher-forcing 的未来 token `y_t` 写 M；
- 在输出尚未 commit 前用未确认预测写 M；
- 用 batch 平均 value 覆盖样本级因果；
- 用 `I_A`/`I_C` 直接当置信度/正确率来 gate 写入。

### 6.2 候选写入生命周期

```text
observe(x_t)
  → advance()
  → propose()          # candidate output, no memory write
  → commit(n)          # 已提交输出；生成 candidate write
  → propose_write(id, key, value, provenance, confidence)
  → confirm_write(id)  # 或 reject_write(id)
  → memory_version += 1
```

错误/打断：
- `interrupt()`：未 commit 的候选写入全部 reject；已 commit 的写入保持；
- 用户纠正：旧的候选写入可 reject；必要时按反馈写补偿 value；
- 外部验证失败：rollback 到写入前版本，或写入反向补偿。

### 6.3 版本与回滚

- 每次确认写入记录 `(memory_version, step, provenance, key_hash, value_hash)`；
- 定期/关键事件 snapshot M；
- `rollback(version)` 恢复 M 到指定版本；
- 写入门控：
  - `provenance ∈ {committed_output, feedback, verified}`；
  - `confidence` 只作为策略输入，不作为真值；
  - 失败预测/低置信候选默认不写或低权重；
- 容量：衰减 + 重要性淘汰 + 上限，具体策略需独立实验。

当前 `write_memory()` 是 batch 平均、无版本、无队列、无 provenance，**不能直接用作在线写入**。

---

## 7. Runtime 状态机与接口

### 7.1 状态

```text
IDLE
 → OBSERVING        # 接收 token/音频等流
 → READY            # 当前输入已形成可用状态
 → PROPOSED         # 已产生候选输出
 → COMMITTING       # 提交中
 → SILENT           # 策略选择不输出
 → INTERRUPTED      # 新输入/用户打断
 → RESUMED          # 恢复后重新进入 READY
```

### 7.2 接口

| 接口 | 输入 | 输出 | 状态变化 |
|---|---|---|---|
| `observe(tokens, meta)` | 新输入、时间戳 | `events` | 追加 input buffer；可能触发 INTERRUPTED |
| `advance()` | — | A/C/I/activity/shock | 计算当前状态；不产生输出、不写 M |
| `propose()` | — | candidate output + policy inputs | READY→PROPOSED |
| `commit(n)` | n ≤ commit_cap | committed tokens/events | PROPOSED→COMMITTING→READY；确认候选 M 写入 |
| `interrupt()` | reason | 中断事件 | 清空未 commit 输出/候选写入；进入 INTERRUPTED |
| `resume()` | restore point | 恢复事件 | INTERRUPTED→RESUMED→READY |
| `snapshot()` | — | runtime state | 序列化 context/state/pending queues/version |
| `restore(snapshot)` | snapshot | — | 恢复状态；不自动产生输出 |

### 7.3 Runtime 状态至少包含

```text
conversation_id, step
token_buffer / context position
A/C recent state (encodings or cache)
active blocks / activity
shock / emotion (current values)
pending_output / pending_write_queue
interrupted/reason
last_committed_position
memory_version
RNG state
```

v15 的 `state` 只有 `act`，`forward` 返回 `"state": {}`，**不满足**上述要求。

---

## 8. 输出仲裁：agreement / confidence / policy 分离

三者不得混用：

| 概念 | 定义 | 不能用来做什么 |
|---|---|---|
| **Agreement** | A/C 的块间一致性（I_A/I_C 的相对关系） | 不能当正确率/置信度 |
| **Confidence** | 模型对候选输出的校准不确定性（如 top1–top2 margin、熵、校准温度） | 不能直接代替 agreement |
| **Policy** | 是否输出、输出哪一个、静默、commit 多少 | 不能把单一 I 阈值当策略全部 |

建议策略：

```text
inputs: agreement_A, agreement_C, calibrated_confidence, commit_cap, interrupt_flag
actions: SILENT | EXECUTE_A | WAIT | COMMIT(n)
```

- `SILENT`：不输出，但可保留状态/记忆候选；
- `EXECUTE_A`：仅当 A 的 agreement 不低于 C，且 confidence 超过校准阈值；
- 外部校准：用 held 集估计 confidence→正确率曲线；报告 precision/coverage/silence rate；
- 不使用 `I_A > I_C` 直接宣布“A 正确”；
- commit cap 控制打断粒度。

---

## 9. Online Learning：A 参数更新 vs M 写入

| 维度 | A 参数更新 | M 写入 |
|---|---|---|
| 载体 | optimizer 参数 | 非参数 buffer |
| 触发 | 完成 chunk / 延迟反馈 | commit/反馈/验证事件 |
| 学习率 | 小 lr，A 专属 | 写入 lr / gate |
| 回滚 | 参数 snapshot | memory version |
| 评估 | 新任务 + 旧任务 held | M 对照（零/正常/打乱） |
| 风险 | 灾难性遗忘 | 错误污染/容量 |

A 更新的最小约束：

1. 只更新白名单 A 参数；
2. shared/C/E 参数冻结；
3. 用**已观察 token 的延迟预测损失**或明确反馈，不用未来 token 泄漏；
4. replay buffer（历史 committed 样本）+ 梯度裁剪；
5. 每次更新前后在固定旧任务 held 集评估；超过阈值则回滚；
6. 与 M 写入分开记录，不能把 M 写入当作参数学习。

---

## 10. 分阶段实施计划

| 阶段 | 内容 | 依赖 | 验收 |
|---|---|---|---|
| O0 | 本设计 + 代码核对 | 无 | 本文档 |
| O1 | 拆分 `out_a/out_c`；冻结白名单；C 不变量测试 | O0 | C hash/输出不变、优化器不含 C/shared |
| O2 | Runtime state 骨架（observe/advance/propose/commit/interrupt/resume/snapshot） | O1 | 状态机单测、断点恢复 |
| O3 | M 候选写入队列 + provenance + version/rollback | O2 | 未来 token 泄漏测试、rollback 测试 |
| O4 | 输出仲裁/静默 + commit cap | O2 | precision/coverage/silence、打断测试 |
| O5 | A 在线更新最小闭环（冻结 shared/C） | O1/O2/O3 | 旧任务不退化、A 改进可测 |
| O6 | E/M 在线策略、动态 activity、PerturbField | O5 | 独立对照，不能与 O5 混淆 |

**O0–O4 完成前，不启动在线训练，也不进入 Phase 2。**

---

## 11. 当前代码无法支持的设计项

1. A/C 独立输出层：v2 共用 `out_a`；
2. Online 参数白名单：v2 无分组/冻结 API；
3. C 稳定性自动测试：不存在；
4. Runtime state machine：不存在；
5. commit/interrupt/resume/snapshot：不存在；
6. 输出静默/仲裁策略：不存在；
7. 因果 M 写入队列/版本/回滚：不存在；
8. 样本级 M 写入：当前 batch 平均且无 provenance；
9. 动态 activity：v2 固定 act；
10. 实时 A 在线更新：v2 无；legacy 只有 chunk 级离线模拟；
11. 历史 interrupt/commit 实验：legacy 报告有记录，但当前仓库没有对应 runtime 代码，无法复核；
12. `a_pred`、`field`、`bank.comm*` 等参数的实际 forward 使用边界需在 O1 用测试固化。

## 12. 本次边界

- 未修改任何代码或现有文档；
- 未运行前向、smoke 或训练；
- 未宣称 Runtime 或 Online Learning 已实现；
- 本文档是设计交付，下一步应先评审 O1 的参数边界与 C 稳定性测试。
