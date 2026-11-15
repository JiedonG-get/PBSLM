# O0.1 设计修订记录

状态：**只读代码核查 + 设计文档修订；未修改代码，未运行前向/smoke/训练。**
基线：设计文档 `docs/ONLINE_ARCHITECTURE_DESIGN.md`（原提交 `d225a39`）。

## 1. 必须修正：bank.comm 调用图

之前的文档把 `bank.comm1/comm2/comm_beta` 标为 v2 未使用参数，这是错误的。

代码证据：

- `models/model_v2.py:140`：`delta = self.bank.comm(delta, ...)`；
- `models/model.py::BlockBank.comm()` 使用 `comm1`、`comm2`、`comm_beta`；
- `models/model_v2.py:109`：`self.ebank.comm(E_out, ...)` 也使用 comm。

修订：

- `bank.comm*` 标记为 **forward 使用的共享参数**；
- 在线冻结，但**不得标为 unused**，也不得进入 A-only 更新集合；
- O1 的优化器/白名单测试必须从实际 forward 调用图出发，而不是从命名推断。

## 2. A/C 路由与跨块通信耦合

代码证据：

- `models/model_v2.py::_select()` 先选 A 的块，再在排除 A 已选块后选 C；
- `active = sorted(set(ia + ic))`；
- 所有 active 块一起经过 `bank.comm()`（`:140`）和 `flow()`；
- A/C 最后只分别聚合自己的 `side_cons`。

结论：

- 冻结 `bank`/`flow` 权重**不等于** C 的计算路径不变；
- 更新 `role_a` 会改变 A 选块，从而改变 C 的候选集合、active 集合、`bank.comm` 的 `others`
  和 `flow` 的跨块交互，进而可能改变 C 的 `c_enc`、`z_c`、`I_c`；
- 仅拆分 `out_a/out_c` 不足以证明 C 稳定。

O1 采用 **方案 B**：

- `role_a` 进入 forward，但 `requires_grad=False`，不进入在线 optimizer；
- O1 只更新：`dec_a`、A query、`refine_proj`、`ln_a`、`ctx_a`、拆分后的 `out_a`；
- A/C 独立路由/独立跨块交互（方案 A）留到 O1.5 设计，不在 O1 混做。

## 3. C 稳定性测试修订

逐个 A 参数组测试，而不是一次笼统 optimizer step：

- `dec_a`、A query、`refine_proj`、`ln_a`、`ctx_a`、`out_a`：C 必须不变；
- **`role_a`：预期 C 会变化**；该测试必须存在，用来证明路由耦合没有被测试跳过；
  在 O1 方案 B 中 `role_a` 不进入优化器；
- 每次测试记录：C 选块集合、`z_c`、top-k、`I_c`、shared/C 参数 hash；
- `direct_only=False` 必须覆盖真正的 `out_a/out_c` 路径；
- `direct_only=True` 单独测试 `ctx_a/ctx_c` 路径，不能替代输出层测试；
- 测试矩阵：fp32/bf16、batch>1、field 开关、M 零/已写入、打断/恢复前后。

## 4. M 写入语义修订

- `commit(n)`：只表示输出已提交给外部，**不改变 M**；
- `propose_write()`：产生候选写入，不改变 M；
- `confirm_write()`：才改变 M 和 `memory_version`；
- `reject_write()`：丢弃候选；
- `interrupt()`：只能拒绝 **尚未 confirm** 的候选写入；已 confirm 的写入必须通过显式
  rollback/纠错/补偿处理；
- 因果规则按“信息在写入时刻是否已经可用”定义：
  - 已观察输入可以作为 value，只要写入时该输入已到达；
  - 禁止使用写入时刻尚未观察到的未来 token；
  - 预测只有在 commit 后才可作为主写入来源；用户反馈/外部验证按到达时间处理。

## 5. 输出仲裁修订

- `agreement_A >= agreement_C` **不再**作为 EXECUTE_A 的硬条件；
- agreement 与 confidence 分开，两者都作为 policy 的输入特征；
- confidence 必须在 held 集校准（temperature/isotonic/Platt），并报告：
  precision、coverage、silence rate、错误 commit 率；
- 至少比较三种策略：A-only、C-only、agreement+confidence；
- `I_A/I_C` 不能直接当作正确率/置信度。

## 6. O1 最终参数白名单

**允许更新：**

- `dec_a.*`
- `q_emb`, `q_pos`
- `refine_proj.*`
- `ln_a.*`
- `ctx_a.*`
- `out_a.*`（拆分后的 A 输出）

**必须冻结：**

- `role_a`（O1 方案 B）
- C 专属：`role_c`、`dec_c`、`q_emb_c/q_pos_c`、`refine_proj_c`、`ln_c`、`ctx_c`、`out_c`
- 共享：`tok`、`bank.*`（含 `bank.comm*`）、`flow.*`、`ebank.*`（含 `ebank.comm`）、
  `e_mix`、`ea_proj`
- 码本 `code`、`field.*`、`emo_calm`
- `mem` 只写不训练
- `a_pred` 只参与辅助 loss，不在 forward；O1 不列入更新集合

## 7. O1 验收条件

1. 优化器 param_groups 只包含上述 6 个 A 组；
2. 对每个允许组分别更新后，C 的选块、`z_c`、top-k、`I_c` 与 shared hash 不变；
3. 对 `role_a` 更新，测试应记录 C 变化；O1 不把 `role_a` 放入 optimizer；
4. `direct_only=False` 的输出层测试通过；`direct_only=True` 单独通过；
5. 所有测试只做前向/参数 hash，不训练、不改模型结构。

## 8. 未解决/后续

- O1.5：A/C 独立路由与独立跨块交互方案；
- O2：Runtime 状态机、commit/interrupt/resume、非法转移；
- O3：M 候选队列/版本/rollback 的具体实现；
- O4：输出 policy 的 held 校准实验；
- O5：A 在线更新与旧任务保持；
- O6：E/M 在线策略、动态 activity、PerturbField。

本轮没有修改代码，没有运行任何实验。
