# Phase 1.1.1 结果：修正 last-token / 后缀消融，仅重评估

状态：**评估补丁完成；未训练、未进入 Phase 2。**
本文件修正 Phase 1.1 中 `last_token_only` 使用「重复最后一个 token」造成的结论偏差。

## 1. 本轮改动

1. 新增 **`last_token_only_masked`**：保留最后一个 token，前 n_in−1 个位置替换为 `<unk>`；
2. 保留旧 **`last_token_repeated`**（把最后一个 token 复制 n_in 次）作为对照，不再作为主结论；
3. 新增 **`drop_last1` / `drop_last2` / `drop_last4`**，只屏蔽末尾 1/2/4 个 token；
4. 核实 tokenizer：vocab=16,384，**id 0 = `<unk>`，`<unk>` id = 0**；
5. 校验 codebook 对照初始化：两配置 **state_dict 数值完全一致**，仅 `requires_grad` 不同；
6. 只对现有 A1/A2 checkpoint 重新评估，不训练。

## 2. 成本

- 10,000 windows × AR8 = 80,000 targets；
- 原始 + 8 个变体 = 9 次前向；
- A1 **101.1s**，A2 **101.1s**，无参数更新。

## 3. Context Generalization（与 Phase 1.1 相同的大 held-out 集）

| 指标 | A1 | A2 |
|---|---:|---:|
| top1 | 0.0712 | 0.0845 |
| top5 | 0.1506 | 0.1454 |
| high n / top1 / top5 | 9,651 / 0.5130 / 0.9341 | 9,651 / 0.5683 / 0.9356 |
| mid n / top1 / top5 | 10,774 / 0.0630 / 0.2472 | 10,774 / 0.1186 / 0.2330 |
| low n / top1 / top5 | 32,154 / 0.0015 / 0.0102 | 32,154 / 0.0000 / 0.0030 |
| rare n / top1 / top5 | 27,421 / 0.0006 / 0.0015 | 27,421 / 0.0000 / 0.0000 |

## 4. Context Sensitivity（固定 target，自然上下文替换）

| 指标 | A1 | A2 |
|---|---:|---:|
| 替换后 top1（原 target） | 0.0332 | 0.0319 |
| top1 utility（原 − 替换） | 0.0379 | 0.0526 |
| 替换后 top5 | 0.0799 | 0.0794 |
| 预测变化率 | 0.5835 | 0.6470 |
| cos shift | 0.3910 | 0.3515 |

## 5. Context Utility（修正后）

定义：`utility_top1 = 原始 top1 − 变体 top1`，正值表示原上下文更有用；
JSON 同时保留 `top1_delta_vs_orig = 变体 − 原始`（符号相反，避免误读）。

| 变体 | A1 top1 | A1 utility | A2 top1 | A2 utility |
|---|---:|---:|---:|---:|
| last_token_only_masked | 0.0439 | **+0.0272** | 0.0594 | **+0.0251** |
| last_token_repeated（旧） | 0.0388 | +0.0324 | 0.0328 | +0.0518 |
| drop_prefix（前 8→unk） | 0.0690 | +0.0022 | 0.0839 | +0.0006 |
| drop_last1 | 0.0447 | **+0.0265** | 0.0472 | **+0.0374** |
| drop_last2 | 0.0448 | +0.0264 | 0.0466 | +0.0379 |
| drop_last4 | 0.0447 | +0.0265 | 0.0449 | +0.0396 |
| prefix_shuffle | 0.0713 | −0.0001 | 0.0851 | −0.0005 |

## 6. 修正后的结论

1. **旧 `last_token_repeated` 的 utility 被高估**：
   - A1 3.24pp → masked 版 **2.72pp**；
   - A2 5.18pp → masked 版 **2.51pp**。
   因此不能再说「完整上下文比最后一个 token 好 3.2 / 5.2pp」；正确的主结论是 **+2.7 / +2.5pp**。

2. **保留最后一个 token、屏蔽前缀**（last_token_only_masked）与**保留前缀、只屏蔽最后一个 token**（drop_last1）都会掉点：
   - A1：两变体 utility 约 2.7pp；
   - A2：masked 2.5pp，drop_last1 3.7pp；
   说明最后一个 token 和前缀内容都对预测有贡献，且 A2 对最后一个 token 的依赖更强。

3. **drop_last1 ≈ drop_last2 ≈ drop_last4**：
   - 屏蔽 1 个 token 已造成主要损失；
   - 再屏蔽 2/4 个 token 只增加很小损失；
   - 因此近邻后缀的收益主要集中在**最后一个 token**，而不是更长的 4-token 后缀。

4. **drop_prefix / prefix_shuffle 几乎无影响**：
   - 删前 8 个 token 或打乱前 15 个 token，top1 变化 <0.3pp；
   - 这只能说明「**前缀顺序信息**在当前设置下没有明显效用」，不能推出「模型完全不使用前缀内容」；
   - 结合 drop_last1 的掉点，前缀内容仍可能通过最后几个位置参与预测。

5. A1/A2 仍有码本初始化和可训练参数量混杂，不能据此判断固定码本优于可训练码本。

## 7. Codebook 对照初始化校验

`configs/ctrl_codebook_trainable.json` vs `ctrl_codebook_frozen.json`：

- 在相同 seed 下独立构造，state_dict **逐参数数值完全一致**；
- state hash 相同：`aabd26588df480b7f9c590e9d3e44c747f3f826594f9c09bed1da0b210ae783e`；
- 总参数：**21,160,328 / 21,160,328**；
- 可训练参数：21,094,400 / 16,900,096（差异 = 码本）；
- 未训练，仅初始化校验。

## 8. 产物

- 脚本：`evaluation/phase1_1_eval.py`（9 个前向变体）
- 初始化检查：`evaluation/check_control_init.py`
- 结果：`experiments/phase1_1/metrics/eval_a1_p111.json`、`eval_a2_p111.json`、`control_init_check.json`
- 旧 Phase 1.1 结果保留；其 `last_token_only` 结论以本文件为准。
