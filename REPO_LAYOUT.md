# PBSLM Repository Layout

本仓库用于管理 PBSLM 实验项目：代码、配置、日志、metrics 与实验报告。

## 代码入口

- `model.py`：基础组件（BlockBank / CrossLayer / InfoFlow / v15 PBSLM）
- `model_v2.py`：完整 v2 模型（A/C、E/M、InfoFlow、连续输出、扰动场）
- `model_v21.py`：v2.1 Baseline A 单路径模型（A1 可训练码本 / A2 固定码本）
- `train_v2.py`：v2 实验训练脚本
- `train_v21.py`：v2.1 配置驱动训练脚本（InfoNCE、checkpoint/resume、完整评估）
- `train_bpe.py` / `tokenize_corpus.py`：自训 BPE 与语料 tokenize

## 配置与数据

- `configs/`：v2.1 的 baseline / formal / preflight / smoke 配置
- `tokenizer/`：BPE 16k tokenizer 与统计（小文件，纳入仓库）
- `data/`、`*.bin`：大语料/二进制数据**不纳入仓库**

## 文档与结果

- `README.md`：长期实验记录
- `PLAN_v2_1.md`、`PHASE0_REPORT.md`、`PREFLIGHT_RESULTS.md`、`PHASE1_RESULTS.md`：阶段计划与结果
- `PHASE1_METRICS_COMPARE.csv`：A1/A2 评估对照
- `phase1_a1.log` / `phase1_a2.log`：正式训练日志
- `metrics_formal_a1.jsonl` / `metrics_formal_a2.jsonl`：正式逐步 metrics
- `results_*.json` / `v*_summary.json`：历史实验汇总

## Artifact 策略

- **checkpoints（`*.pt`）不上传**；
- `.bin` 数据不上传；
- 日志/metrics/报告/小 CSV 上传；
- 大文件保留在本地/Kaggle/Release（如需要）。

## 当前阶段

- Phase 1 Baseline A1/A2 已完成（15,000 步）；
- 结果见 `PHASE1_RESULTS.md`；
- 未进入 Phase 2。
