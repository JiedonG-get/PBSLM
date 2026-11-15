# Phase 1.1 结果：固定 held-out 评估 + 上下文敏感性/效用/泛化

> **注意：本文件是 Phase 1.1 初版；其中 `last_token_only` 使用“重复最后一个 token”的旧实现，
> 其 utility 数字已被 `PHASE1_1_1_RESULTS.md` 修正（masked 版本 A1 +2.72pp、A2 +2.51pp）。
> 本文件保留作历史记录。**

状态：Phase 1.1 评估已完成；**未训练、未改 Phase 1 原始 logs/metrics、未进入 Phase 2。**

## 1. 目的与协议

Phase 1.1 修正 Phase 1 的评估口径：

- **Context Generalization**：固定 held-out 自然上下文上的 top1/top5，按频率分组；
- **Context Sensitivity**：固定 target、只改 context，看预测是否变化；
- **Context Utility**：固定 target、做自然/受控消融，看原上下文是否更有用。

完整定义见 `docs/EVAL_PROTOCOL.md`。Phase 1 的 `ctx_change_rate` 不再作为上下文敏感性/利用能力的主证据。

## 2. 评估设置与成本

- checkpoint：`ckpt_formal_a1.pt` / `ckpt_formal_a2.pt`（step 15,000；不进入 git）；
- 固定 held-out：**10,000 windows × AR8 = 80,000 targets**，起点固定，与训练时 64-window 快速 eval 分离；
- 6 次前向：原始 + 5 个变体（matched natural / last-token-only / drop prefix / drop suffix / prefix shuffle）；
- 实测成本：A1 **71.4 s**，A2 **70.0 s**（单卡 eval，无训练/无参数更新）。

## 3. Context Generalization（10k windows，固定 held）

| 指标 | A1（可训练码本） | A2（固定码本） |
|---|---:|---:|
| overall top1 | 0.0712 | **0.0845** |
| overall top5 | **0.1506** | 0.1454 |
| high 组 n / top1 / top5 | 9,651 / 0.5130 / 0.9341 | 9,651 / **0.5683** / 0.9356 |
| mid 组 n / top1 / top5 | 10,774 / 0.0630 / 0.2472 | 10,774 / **0.1186** / 0.2330 |
| low 组 n / top1 / top5 | 32,154 / 0.0015 / 0.0102 | 32,154 / 0.0000 / 0.0030 |
| rare 组 n / top1 / top5 | 27,421 / 0.0006 / 0.0015 | 27,421 / 0.0000 / 0.0000 |

结论：A2 在 overall top1、high/mid 组更好；A1 top5 略高。**低频/稀有组仍接近 0**，频率偏置问题在 80k 固定集上依然存在。

## 4. Context Sensitivity（固定 target，自然上下文替换）

“matched natural context” = 用 held 中另一条自然上下文替换原上下文，**target 保持原 y**。

| 指标 | A1 | A2 |
|---|---:|---:|
| 替换后 top1 | 0.0332 | 0.0319 |
| top1 − 原 top1 | −0.0379 | −0.0526 |
| 替换后 top5 | 0.0799 | 0.0794 |
| top5 − 原 top5 | −0.0707 | −0.0661 |
| 预测变化率 | 0.5835 | **0.6470** |
| 输出 cos shift | 0.3910 | 0.3515 |

解读：

- 换上下文后预测变化率 58–65%，cos shift 0.35–0.39：**输出确实对上下文敏感**；
- 同时同一 target 的准确率显著下降（−3.8 / −5.3 pp）：上下文不是随便换都行；
- 但这仍不能单独证明“长上下文理解”，要看下面的 utility 消融。

## 5. Context Utility（固定 target，消融原上下文）

`positive utility = 原 top1 − 变体 top1`；正值表示原上下文比该消融更有用。

### A1

| 变体 | 变体 top1 | utility（原−变体） |
|---|---:|---:|
| last-token-only | 0.0388 | **+0.0324** |
| drop_prefix（前 8 token→unk） | 0.0690 | +0.0022 |
| drop_suffix（后 4 token→unk） | 0.0447 | **+0.0265** |
| prefix_shuffle | 0.0713 | −0.0001 |

### A2

| 变体 | 变体 top1 | utility（原−变体） |
|---|---:|---:|
| last-token-only | 0.0328 | **+0.0517** |
| drop_prefix（前 8 token→unk） | 0.0839 | +0.0006 |
| drop_suffix（后 4 token→unk） | 0.0449 | **+0.0396** |
| prefix_shuffle | 0.0851 | −0.0006 |

关键结论：

1. **full context 比 last-token-only 更有用**（A1 +3.2pp，A2 +5.2pp）；
2. **删掉后缀（靠近 target 的 token）明显掉点**（+2.65 / +3.96 pp utility）；
3. **删掉前缀 / 打乱前缀几乎不影响**（<0.3pp）——当前模型对 16-token 上下文的利用主要是**局部后缀/近邻 token**，而不是长前缀；
4. 因此 Phase 1 的“上下文敏感性”应细化为：**局部上下文有用，长距离前缀利用尚未证明。**

## 6. 对 Phase 1 结论的修正

- Phase 1 的 `ctx_change_rate` 同时改变 context 和 target，且受控替换是 OOD 随机 token；
  新评估改为固定 target 后，才得到上面的 sensitivity/utility 分离结论。
- A1/A2 的差异仍受 **码本初始化不同 + 可训练参数量不同（差 4.19M）** 混杂；
  Phase 1 原始结果保留，不把 A2 的略好解释为“固定码本机制更好”。

## 7. 同初始化 codebook 对照（设计，未训练）

为区分「码本可训练性」和「参数/初始化混杂」，新增：

- 配置：`configs/ctrl_codebook_trainable.json` / `ctrl_codebook_frozen.json`；
- 两者 **总参数相同 21,160,328**，codebook 使用相同 seed=1234 的单位向量初始化；
- trainable 版码本可训练（trainable params 21.09M），frozen 版码本冻结（trainable params 16.90M）；
- 不增加 adapter，不为了参数量匹配随意加参数；报告需同时给出 total 与 trainable 参数量。
- 说明：`experiments/phase1_1/CODEBOOK_CONTROL.md`；
- **尚未启动训练**，等 Phase 1.1 审查后再决定。

## 8. 产物

- 脚本：`evaluation/phase1_1_eval.py`
- 配置：`configs/eval_phase1_1.json`
- 指标定义：`docs/EVAL_PROTOCOL.md`
- 结果 JSON：`experiments/phase1_1/metrics/eval_a1.json`、`eval_a2.json`
- 未修改 Phase 1 的 logs/metrics；Phase 1 原始结果仍在 `experiments/phase1/`。

## 9. 被评估 checkpoint 的 sha256

```text
46d566ab85f56cdfd7148a287fb17f1e9981599bc82159c4e4f6b65a870878f6  ckpt_formal_a1.pt
9f9826c924cbb57f07a5ccd32de813a61cc8692f75f4e399f46b5005a89372c9  ckpt_formal_a2.pt
```

checkpoint 不进入 git；评估 JSON 在 `experiments/phase1_1/metrics/`。
