# O1.5 接手后的工程证据报告

日期：2026-11-15。起点远程 `9984267`。用户已授权自主推进研究、及时上传；本报告记录第一批工程证据，不把测试当作 Online Learning 成功。

## 本次实际完成

- 新增独立的逐样本紧凑 M1：每样本 union 长度不同，按物理块 ID 对齐 block 轴及 attention 的 query/key 轴。参考 comm/Flow/统计使用显式方程，不调用 production masked comm/Flow。
- 比较 Y/delta、comm others/update/out、proj_in、每层 Q/K/V/logits/attention/message/output、masked mean/scale/agree、settled、side、I、a_enc/c_enc、z_a/z_c。
- 确定性固定 C 改 A、固定 A 改 C；global union 允许改变，固定侧按物理 ID 比较。包含另一 batch 样本保持不变。
- 完全/部分/无重叠 actual forward；分别验证角色 tensor storage 与 mask。
- 8 组分支注入：comm/Flow × A无效C有效、C无效A有效、当前样本双无效但其他样本有效。注入使用分支局部副本；如果污染的是共享且对另一侧有效的 Y，则另一侧应报错，不能误称可屏蔽。
- 分别以非均匀固定 A/C probe 调用 autograd.grad，记录逐参数 None/isfinite/max_abs_grad。每次测试确认 state_dict 未改变、param.grad 未累积。
- 增加 masked 最外层 proj_out、settled、side/I、编码与最终 z 的有限性边界。修正 no-flow 时无效 agree 为零。legacy 数学路径不改。
- 修正 O1 top-k 测试：调用实际配置的 decode_nearest，而不是不论配置都用固定随机 code。

## 运行命令与证据

```bash
python -m evaluation.run_o1_5_evidence
python -m evaluation.check_checkpoint_compat --output experiments/o1_5/checkpoint_evidence.json
```

环境：Python 3.13.5、PyTorch 2.11.0+cu128；本轮测试 CPU fp32。硬件检测为 RTX 4060 Laptop 8GB，但未使用 GPU 训练。

联合 **60 tests OK**；实际原始 stdout/stderr 在 `gpt_verification/unittest.log`，环境与真实退出码在 `gpt_verification/environment.json`。没有 optimizer.step、训练或数据集学习实验。

独立 M1 容差固定为 atol=1e-6、rtol=1e-5。最大绝对差：scale 9.5367e-7；z_a 1.1921e-7；z_c 1.7881e-7。每层完整误差见原始日志。

## 梯度边界

- tok/bank/comm/flow 两侧均可有非零梯度，这是共享参数的预期结果，不是状态泄漏。
- A/C 专属 decoder/query/ctx/out/ln 与路由 score 梯度各自独立；E/e_mix/ea_proj 在当前 forward 中是 A-only，不应概称 C 的共享梯度来源。
- a_pred 是 loss-only，本轮 z probe 对其为 None；field_scale=0 且 probe=z_a/z_c，field 梯度为 None。
- kA=1 时选中 score 的 softmax 恒为1、hard topk索引不可导，role_a 不应被假定能从该路径有效学习。测试 fixture 用 kA=kC=2 检查路由权重的非退化梯度。
- probe 梯度不是实际训练质量、抗遗忘或学习收益。

## Checkpoint 证据

两个 .pt 均在本地仓库目录，被 Git 忽略；本轮只上传哈希、args、构造配置和结果，不上传 checkpoint。

- direct_ce: raw strict 缺 out_c.weight/bias；现有 out_c迁移后可strict load。dense=True、use_flow=False、direct_only=True，不能支持 sparse+Flow 历史兼容结论。
- nomem_100: raw strict 缺 out_c与ctx_a/ctx_c；迁移后仍缺ctx_a/ctx_c。未补造权重以让其通过。
- 修改时间不是经过证明的生成时间。args不完整，e_depth等构造默认值在证据中标注为推断。

## 已知限制

- 新独立 M1 只有一个小模型 fp32 CPU seed；复用 production BlockBank、CrossLayer 和 E 通路，不是这些模块的完全独立数值实现。
- 原先51项测试继续保留；旧primitive M1误差打印可能受 matching -inf相减影响，不能替代本轮独立M1日志。本轮新M1分别检查inf模式与有限值误差。
- 新finite边界只在use_sample_masks=True启用；未验证GPU/mixed precision/随机field。
- 未做新训练、Online收益、旧任务遗忘、Runtime、因果M写入、容量与污染实验。
- 状态隔离是有限配置的数值证据，不是“完全独立”或研究假设成立。

下一步：按 RESEARCH_EXECUTION 的受控预算建立最小持续适应任务，分离参数更新、记忆写入、旧能力保持与无更新对照，不再以继续堆文档代替实验。
