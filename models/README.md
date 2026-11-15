# Models

- `model.py`：基础组件 `BlockBank` / `CrossLayer` / `InfoFlow`，以及 v15 `PBSLM`。
- `model_v2.py`：完整 v2 `PBSLMv2`（A/C、E/M、InfoFlow、连续输出、PerturbField）。
- `model_v21.py`：v2.1 Baseline A 单路径连续向量模型（A1 可训练码本 / A2 固定码本）。

注意：`model_v21.py` 是 Phase 1 实际训练模型；完整 v2 仍在实验阶段。
