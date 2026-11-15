# Phase 1 结果：Baseline A1 / A2 正式训练（15,000 步）

状态：**Phase 1 已完成；未进入 Phase 2。**
两组均从头训练，未从 preflight checkpoint resume。

## 1. 配置与预算

| 项 | A1 | A2 |
|---|---|---|
| config | `configs/formal_a1.json` | `configs/formal_a2.json` |
| codebook | **trainable（独立码本）** | **fixed（seed=1234）** |
| data | `bpe_full_u32.bin`，520M tokens，hold=0.15 | 同 |
| 训练 | 15,000 步 × batch8 × AR8 = 960k next-token 样本 | 同 |
| loss | **sampled-negative InfoNCE** τ=0.1, neg_n=128, exclude_positive（1 正 + 128 采样负例；非 batch negatives） | 同 |
| optimizer | AdamW lr=1e-3, grad_clip=1.0 | 同 |
| eval | 每 1,000 步，windows=64，hard_neg_pool=512/k=16/seed=4321 | 同 |
| trainable params | 21.09M | 16.90M |

## 2. 运行时间 / 显存 / 稳定性

| 项 | A1 | A2 |
|---|---|---|
| 最终日志 `sec` | 629.8s（~10.5 min） | 590.6s（~9.8 min） |
| 峰值 allocated / reserved | 0.677 / 0.732 GiB | 0.633 / 0.709 GiB |
| NaN / Inf / OOM | 无 | 无 |
| metrics 行数 | **15,000** | **15,000** |
| EVAL 次数 | 15 | 15 |
| 日志 | `experiments/phase1/logs/phase1_a1.log` | `experiments/phase1/logs/phase1_a2.log` |
| metrics | `experiments/phase1/metrics/metrics_formal_a1.jsonl` | `experiments/phase1/metrics/metrics_formal_a2.jsonl` |
| checkpoint | `experiments/phase1/checkpoints/ckpt_formal_a1.pt` | `experiments/phase1/checkpoints/ckpt_formal_a2.pt` |

metrics 文件在启动前确认不存在，运行后正好 15,000 行；两组输出路径独立，未覆盖 smoke/preflight。

## 3. Checkpoint 检查

| 检查 | A1 | A2 |
|---|---|---|
| step | 15,000 | 15,000 |
| codebook mode | trainable | fixed |
| stored hash == 当前码本 hash | True | True |
| optimizer state entries | 20 | **19**（码本不参与优化） |
| Adam step | 15,000 | 15,000 |
| `rng_cuda` 已保存 | 是 | 是 |
| 固定码本可复现 | — | **与同 seed 新模型一致，requires_grad=False** |
| sha256 | `46d566ab85f56cdfd7148a287fb17f1e9981599bc82159c4e4f6b65a870878f6` | `9f9826c924cbb57f07a5ccd32de813a61cc8692f75f4e399f46b5005a89372c9` |

## 4. 评估轨迹（selected）

评估记录按零索引 step 999/1999/... 对应实际 1,000/2,000/... 步。

| 组 | 实际步 | held_top1 | held_top5 | hard_neg_win | ctx_change | other_ctx_acc | ctrl_keep_last_acc | high | mid | low | rare |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 1,000 | 0.0664 | 0.1367 | 0.3630 | 0.2988 | 0.0566 | 0.0586 | 0.4928 | 0.0 | 0.0 | 0.0 |
| A1 | 5,000 | 0.0879 | 0.1543 | 0.4220 | 0.5488 | 0.0781 | 0.0625 | 0.6087 | 0.0469 | 0.0 | 0.0 |
| A1 | 10,000 | 0.0859 | 0.1543 | 0.4847 | 0.4922 | 0.0723 | 0.0840 | 0.6087 | 0.0312 | 0.0 | 0.0 |
| A1 | 15,000 | **0.0820** | **0.1602** | **0.5286** | **0.6094** | 0.0781 | 0.0801 | 0.5942 | 0.0156 | 0.0 | 0.0 |
| A2 | 1,000 | 0.0898 | 0.1445 | 0.5590 | 0.2383 | 0.0859 | 0.0781 | 0.6087 | 0.0625 | 0.0 | 0.0 |
| A2 | 5,000 | 0.0957 | 0.1543 | 0.5691 | 0.4629 | 0.0996 | 0.0781 | 0.6232 | 0.0938 | 0.0 | 0.0 |
| A2 | 10,000 | 0.0957 | 0.1582 | 0.5790 | 0.6055 | 0.0840 | 0.0938 | 0.6232 | 0.0938 | 0.0 | 0.0 |
| A2 | 15,000 | **0.0938** | **0.1562** | **0.5737** | **0.6680** | 0.0938 | 0.0918 | 0.5942 | 0.1094 | 0.0 | 0.0 |

最后 5 次 eval 平均：

| 组 | top1 | top5 | hard_neg_win | ctx_change | other_ctx_acc | ctrl_keep_last_acc | high | mid | low | rare |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.0824 | 0.1578 | 0.5191 | 0.5488 | 0.0793 | 0.0832 | 0.5826 | 0.0312 | 0.0 | 0.0 |
| A2 | 0.0942 | 0.1570 | 0.5738 | 0.5985 | 0.0926 | 0.0895 | 0.5855 | 0.1219 | 0.0 | 0.0 |

## 5. 训练 loss

| 组 | 首步 | 末步 | 最后 100 步均值 |
|---|---|---|---|
| A1 | 4.9467 | 2.6246 | **3.0423** |
| A2 | 5.2392 | 4.4584 | **4.3188** |

A1 训练 loss 明显更低，但 held top1 反而略低、mid 频率组更差。单次实验下不能排除**可训练码本与模型共同移动目标空间**造成的「训练 loss 更好看、泛化没有更好」；需要后续更多 seed / 对照才能下结论。

## 6. 观察与分析

1. **预测变化率上升**：`ctx_change_rate` 从 1k 步约 0.24–0.30 升到 15k 步约 0.61–0.67。
   但该实现同时改变了 context 与 target，且受控替换使用 OOD 随机 token，因此**不能**把该数值单独解释为上下文敏感性/上下文利用能力；Phase 1.1 会补固定 target 反事实评估。
2. **但模型仍强烈偏高频**：high 组 top-1 ~0.59，mid 组 A1 ~0.02、A2 ~0.11，low/rare 始终 0。整体 top1 8–9% 仍很低。
3. **受控上下文替换**：A1/A2 的 `ctrl_keep_last_acc` 与总 top1 接近（约 0.08–0.09），说明「保留最后一个 token、替换其余」几乎不降低准确率；模型当前仍主要依赖最后 token / 局部 bigram，而不是完整上下文。
4. **hard_neg_win**：A1 从 0.363 升到 0.529，A2 约 0.56–0.57，仍接近随机（0.5），困难负样本区分能力弱。
5. **A1 vs A2**：单次 15k 步、单 seed 下，A2 在部分指标上略好，A1 训练 loss 更低；但 A1/A2 同时存在
   **码本初始化不同**（A1 随机 trainable，A2 固定 seed 单位向量）和**可训练参数量不同**（21.09M vs 16.90M，
   差 4.19M = 码本）两个混杂。**不足以宣布哪个架构更好**，需要同初始化 trainable/frozen 对照。
6. Phase 1 目标已达成：Baseline A1/A2 正式训练和评估完成；没有进入 Phase 2。

## 6.5 Phase 1.1 补充评估

Phase 1.1 使用固定 10k windows（80k targets）重新评估，并把 Context Sensitivity / Utility / Generalization 分开：
`experiments/phase1_1/PHASE1_1_RESULTS.md`。本文件中的 Phase 1 数值保持不变。

## 7. Phase 1 配置归档

- 训练时配置快照：`experiments/phase1/configs/formal_a1_training.json`、`formal_a2_training.json`（从 commit `fb8caf9` 导出）；
- 当前 `configs/formal_*.json` 只把路径改为仓库相对路径，超参数未变。

## 8. 复现命令

```bash
cd E:/PycharmProject/experiments/online_v12
C:/Python313/python.exe -m training.train_v21 --config configs/formal_a1.json > experiments/phase1/logs/phase1_a1.log 2>&1
C:/Python313/python.exe -m training.train_v21 --config configs/formal_a2.json > experiments/phase1/logs/phase1_a2.log 2>&1
```
