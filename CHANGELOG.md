# CHANGELOG

## 2026-11-15

### Repository
- 建立 GitHub 仓库 `PBSLM`（public）。
- 重组目录：`models/`、`training/`、`docs/`、`experiments/phase0|phase05|phase1|legacy/`、`scripts/`。
- checkpoints 与 `.bin` 数据不进入 git；日志/metrics/报告/tokenizer 保留。

### Phase 0 — 训练闭环整理
- checkpoint 保存 model/optimizer/step/config/CPU/CUDA RNG/generator/codebook hash；
- resume 跳过 field 初始化；
- 不兼容 config 拒绝恢复；
- 固定码本 seed/hash/冻结；
- 评估补齐 top1/top5、频率分组、上下文交换、困难负样本。

### Phase 0.5 — smoke / preflight
- A1/A2 30 步 smoke；A1 resume 30→50；
- batch=8 preflight：无 OOM/NaN，A1 峰值 ~0.68 GiB allocated、稳态 ~0.039 s/step；
- 报告：`experiments/phase05/`。

### Phase 1 — Baseline A1/A2
- A1（可训练码本）、A2（固定码本）各 15,000 步，batch=8，AR teacher forcing；
- held_top1：A1 0.0824 / A2 0.0942（最后 5 次 eval 平均）；
- 低频/稀有 top1 仍为 0；`hard_neg_win` 接近随机；
- 未进入 Phase 2；报告：`experiments/phase1/PHASE1_RESULTS.md`。

## 2026-11-14

- 完成 v2 原型：A/C、E/M、InfoFlow、连续输出、PerturbField；
- 自训 BPE 16k，完整 zhwiki tokenize 为 520M tokens；
- v2-lite 隔离实验（密集块/无 flow/无记忆/simple decoder）；
- 发现非自回归多位置输出稀释监督，改用 AR teacher forcing；
- v2-lite AR 15k 步 held decode 10.94%（诊断阶段）；
- 修复 field 劫持、输出范数爆炸、码本未持久化等问题。

更早的 v12–v15 记录见 `experiments/legacy/README_legacy.md` 与 `experiments/legacy/REPORT_legacy.md`。
