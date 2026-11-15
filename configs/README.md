# Configs

| 配置 | 阶段 | 说明 |
|---|---|---|
| `smoke_a1/a2.json` | Phase 0.5 | 30 步 smoke，可训练/固定码本 |
| `preflight_a1/a2.json` | Phase 0.5 | batch=8 preflight |
| `baseline_a1/a2.json` | Phase 1 | Baseline A 基础配置 |
| `formal_a1/a2.json` | Phase 1 | 正式 15,000 步配置 |

所有路径都相对于仓库根目录；`data/bpe_full_u32.bin` 不纳入仓库（见根 README 的数据准备）。
