# REPO_LAYOUT — 目录与 artifact 策略

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 架构、Online、实验计划、历史决策 |
| `models/` | `model.py`（基础组件/v15）、`model_v2.py`（完整 v2）、`model_v21.py`（Baseline A） |
| `training/` | `train_v2.py`、`train_v21.py` |
| `configs/` | baseline/formal/preflight/smoke JSON 配置（相对仓库根路径） |
| `scripts/data/` | BPE、tokenize、语料下载/解析 |
| `scripts/analysis/` | 历史分析/评估脚本 |
| `experiments/phase0/` | 训练闭环整理报告、v2.1 计划 |
| `experiments/phase05/` | smoke / preflight 日志、metrics、报告 |
| `experiments/phase1/` | 正式 A1/A2 日志、metrics、结果、对照 CSV |
| `experiments/legacy/` | v12–v15 历史结果、日志、metrics、旧训练代码 |
| `tokenizer/` | 自训 BPE 16k tokenizer 与统计 |

## Artifact 策略

**不上传：**

- 所有 checkpoints：`*.pt`、`*.pth`、`*.ckpt`、`ckpt_*`
- 语料二进制：`*.bin`
- `data/`、`backup_v2_orig/`、缓存、临时文件

**上传：**

- 代码、配置、文档、报告
- 训练日志和逐步 metrics（体积小，便于复查）
- tokenizer JSON（约 1.3MB）
- 小体积结果 JSON / CSV

checkpoints 和完整语料的保留在本地/Kaggle/Release，不进入 git。
