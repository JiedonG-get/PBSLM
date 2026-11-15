# EVAL_PROTOCOL — Phase 1.1 评估定义

Phase 1.1 只做评估，不训练。目标是区分三个容易混淆的概念：

| 概念 | 问题 | 指标 |
|---|---|---|
| **Context Sensitivity** | 换上下文后，模型对**同一个目标**的预测是否变化？ | `pred_change_rate`、`cos_shift`、`accuracy`/`top1_delta` |
| **Context Utility** | 原上下文是否比消融后的上下文**更有用**？ | 原 top1 − 消融 top1 |
| **Context Generalization** | 在未见过的自然 held-out 上下文上，预测是否仍然正确？ | 固定 held 集的 top1/top5、频率分组 n/top1/top5 |

> 旧的 `ctx_change_rate` 同时改变 context 与 target，不能单独作为上下文敏感性/利用能力的证据。
> Phase 1.1 用固定 target、只改 context 的评估替代它作为主证据。

## 1. 固定 held-out 集

- 数据：`data/bpe_full_u32.bin` 的固定 15% 尾部（与 Phase 1 相同）；
- 起点：`np.linspace(0, len(held)-n_in-ar_m-2, windows)`，`windows=10,000`，位置固定；
- AR=8，每个窗口 8 个 next-token target → 80,000 个 target；
- 所有模型/实验使用同一批起点、同一频率阈值，保证可直接比较；
- 训练中的 64-window 快速 eval 保留，仅用于监控，不用于结论。

## 2. Context Generalization

- 原始自然上下文；
- 输出 overall top1/top5；
- 频率分组：高频 ≥1%，中频 0.1–1%，低频 0.01–0.1%，稀有 <0.01%（以训练集频率为准）；
- 每组报告 **n、top1、top5**。

## 3. Context Sensitivity（固定 target，改变 context）

对每个原始样本 `(x, y)`：

- 用 held 集中的**另一条自然上下文** `x'` 替换 `x`，但 **target 保持 y 不变**；
- 指标：
  - `pred_change_rate_vs_orig`：argmax 是否变化；
  - `cos_shift_vs_orig`：`1 - cos(z_x, z_x')`；
  - `top1` / `top5`：在替换上下文下 target y 的准确率；
  - `top1_delta_vs_orig`：替换上下文准确率 − 原上下文准确率。

这回答「换上下文后预测是否变化」，但**不直接回答变化后是否更正确**；正确性看 utility/generalization。

## 4. Context Utility（固定 target，自然/受控消融）

每个变体都保留原 target y：

| 变体 | 定义 |
|---|---|
| `last_token_only_masked` | 保留最后一个 token，前 `n_in-1` 位置替换为 `<unk>`(id=0) |
| `last_token_repeated` | 旧对照：把最后一个 token 复制 n_in 次（人工重复输入，不作为主结论） |
| `drop_prefix` | 前 `n_in//2` 个 token 替换为 `<unk>` |
| `drop_last1` / `drop_last2` / `drop_last4` | 只屏蔽最后 1/2/4 个 token |
| `prefix_shuffle` | 打乱前 `n_in-1` 个 token，最后一个 token 保持不变 |

指标：

- 各变体 top1/top5；
- JSON 中同时给出：
  - `top1_delta_vs_orig = 变体 − 原始`；
  - `utility_top1 = 原始 − 变体`（正 utility = 原上下文更有用）；
- 正值表示原上下文比该消融更有用。

注意：`<unk>` 是删除标记，不是自然 token；这仍是受控实验，不能完全代表自然语言中的缺失。

## 5. 输出

`evaluation/phase1_1_eval.py` 输出 JSON：

```json
{
  "context_generalization": {"top1":..., "top5":..., "freq_bucket": {...}},
  "context_sensitivity": {"matched_natural_context": {...}},
  "context_utility": {"last_token_only": {...}, "drop_prefix": {...}, ...},
  "definitions": {...},
  "cost": {"seconds":..., "forward_variants": 6}
}
```

结果写入 `experiments/phase1_1/metrics/eval_<tag>.json`。

## 6. 成本

- 10,000 windows × AR8 = 80,000 target；
- 原始 + 8 变体 = 9 次前向（Phase 1.1.1 修正后）；
- 不训练、不更新参数；batch=512，GPU 上预计数分钟。


## 7. Phase 1.1.1 修正

Phase 1.1 的 `last_token_only` 使用「重复最后一个 token」，会制造人工重复输入；Phase 1.1.1 增加
`last_token_only_masked`（保留最后一个 token、其余位置 `<unk>`），并新增 `drop_last1/2/4`。
因此 **last-token utility 以 `PHASE1_1_1_RESULTS.md` 为准**：A1 +2.72pp、A2 +2.51pp，而不是旧版的 +3.2/+5.2pp。
