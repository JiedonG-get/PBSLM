# Phase 0 代码整理报告（v2.1）

状态：**代码已整理，未启动 Phase 0.5，未启动任何训练。**
原始文件已备份到 `backup_v2_orig/`（model.py / model_v2.py / train_v2.py），原文件本身未被覆盖。

## 1. 修改文件清单

| 文件 | 状态 | 作用 |
|---|---|---|
| `model_v21.py` | 新增 | Baseline A 单路径连续向量模型：可训练码本 / 固定随机码本；无 A/C、无 flow、无 E/M、无 field、无 shock/activity |
| `train_v21.py` | 新增 | v2.1 配置驱动训练脚本：InfoNCE、checkpoint/resume、评估指标、随机数状态 |
| `configs/baseline_a1.json` | 新增 | Baseline A1：可训练码本 |
| `configs/baseline_a2.json` | 新增 | Baseline A2：固定随机单位码本（seed=1234） |
| `configs/smoke_a1.json` / `smoke_a2.json` | 新增 | Phase 0.5 专用：30 步、eval_every=10、save_every=10 |
| `PLAN_v2_1.md` | 新增 | 执行计划与验收标准 |
| `backup_v2_orig/` | 备份 | 原始三文件快照，可回滚 |

## 2. 每项修改的原因与位置

### A. 保留原始版本
- `backup_v2_orig/model.py`、`train_v2.py`、`model_v2.py`：原始文件未改，v2.1 用独立新文件。

### B. 先修正确性（checkpoint / resume / 初始化）

| 修复 | 位置 | 原因 |
|---|---|---|
| resume 跳过 field 初始化 | `train_v21.py` main：`if start_step == 0 and hasattr(model, "field")` | 原 `train_v2.py` 在 resume 后仍把 field 最后一层重置，覆盖已训练参数 |
| checkpoint 保存 model + optimizer + step + config + torch/numpy RNG + data/neg generator 状态 + codebook hash/mode | `save_checkpoint()` / resume 分支 | 原版只保存 model state，resume 不恢复优化器/步数/随机状态，无法复现 |
| 固定码本 hash 校验 | `codebook_hash()`、resume 校验 | A2 的固定码本必须训练期间不变；如果 resume 时 hash 不一致直接拒绝 |
| 默认优化器去 bitsandbytes 依赖 | `build_optimizer()`（AdamW/Adam/SGD） | 原 `train_v2.py` 默认 `adam8bit`，没有 bitsandbytes 就无法启动 |
| 数据窗口 generator 独立并保存状态 | `ar_batch()`、`save_checkpoint(data_gen)` | 原版 resume 后数据采样不可复现 |

### C. Baseline A 定义固定进配置

- A1/A2 配置除 `model.codebook` 外完全一致：数据、seed、batch、步数、loss（InfoNCE）、负样本数、评估窗口全部相同。
- InfoNCE 位置：`info_nce_loss()`；负样本显式排除与正样本相同 token：`negl.masked_fill(nid == y[:, None], ...)`。
- A2 固定码本：`model_v21.py` 内用 `torch.Generator().manual_seed(codebook_seed)` 生成单位向量并 `requires_grad_(False)`；优化器只收集 `requires_grad=True` 参数。

### D. 评估指标（解决「只看 top-1 / 背词频」）

`evaluate()` 输出：
- `held_top1`、`held_top5`
- `cos`：预测向量与目标码本平均余弦
- `hard_neg_win`：目标之外最相似 token 的竞争中，模型是否把目标排前面
- `ctx_change_rate` / `other_ctx_acc` / `ctrl_keep_last_acc` / `ctrl_keep_last_change`：
  换上下文后的预测变化率、随机另一条上下文配**它自己的正确目标**的准确率、保留最后 token 的受控替换准确率/变化率
- `freq_bucket`：高频（≥1%）、中频（0.1–1%）、低频（0.01–0.1%）、稀有（<0.01%）分组 top-1

## 3. v2.1 相对原版的差异

- **只实现 Baseline A 单路径**；A/C、InfoFlow、E/M、PerturbField、Shock/Activity 动态均未加入（Phase 2 再做）。
- 模型结构：输入 token embedding → `BlockBank`（k 个块全活跃）→ LayerNorm → (最后 token + 均值) 读出 → 线性输出 → 单位向量；目标码本 A1 可训练 / A2 固定随机。
- 训练：AR teacher forcing（每位置真实前文），InfoNCE（batch + 采样负例，无全词表 softmax）。
- 不改变原实验记录；原 `train_v2.py`/`model_v2.py` 仍保留可复现旧实验。

## 4. 配置文件与 checkpoint 格式

### 配置（JSON）
见 `configs/baseline_a1.json`：`seed / data / model / loss / optim / training / eval / logging / checkpoint`。
A2 只有 `model.codebook = "fixed"`、`codebook_seed = 1234`，输出路径不同，其余相同。

### Checkpoint（v2.1）
```python
{
  "format": "v2.1",
  "step": int,
  "config": dict,                 # 完整实验配置
  "model": state_dict,
  "optimizer": state_dict,
  "rng_torch": ByteTensor,
  "rng_cuda": list/None,            # torch.cuda.get_rng_state_all()
  "rng_np": tuple,
  "neg_gen": Tensor,              # 负样本采样 generator 状态
  "data_gen": Tensor,             # 数据窗口 generator 状态
  "codebook_hash": str,
  "codebook_mode": "trainable"|"fixed",
}
```
- resume 会校验 `format`、`codebook_mode`，固定码本还会校验 hash。
- field 初始化只在 `start_step == 0` 执行；resume 跳过。

## 5. 尚未解决的问题（明确列出）

1. **Phase 0.5 未执行**：按你的要求，代码整理完先报告，未做 smoke、未启动训练。
2. **dual 模型（A/C + flow + E/M + field）尚未迁入 v2.1**： Phase 2 再做；当前 v2.1 只有 baseline 单路径。
3. **shock 状态管理**：baseline 不含 shock；dual 版需要样本级 shock 状态和 checkpoint 字段，Phase 2 实现。
4. **旧 `model.py` 的两个问题未修**：`online_params_gate` 的 `self.role`、QwenIO `_soft` 非 buffer——已确认不影响 PBSLMv2/v2.1 baseline，标记为低优先级。
5. **`use_amp` / `direct_only` / `dense`** 只存在于旧脚本；v2.1 baseline 不再使用这些变量，后续 dual 迁入时统一处理。
6. 未做正式训练，因此没有新的 held/train 指标。

## 6. Phase 0.5 执行命令（待你确认后才运行）

```bash
cd E:/PycharmProject/experiments/online_v12

# A1 smoke：30 步 + 每 10 步评估/保存
C:/Python313/python.exe train_v21.py --config configs/smoke_a1.json

# A1 resume：从 30 步续到 50 步（验证 checkpoint/optimizer/step/RNG）
C:/Python313/python.exe train_v21.py --config configs/smoke_a1.json \
  --resume ckpt_smoke_a1.pt --steps 50

# A2 smoke（固定码本，验证 hash 不变、码本不更新）
C:/Python313/python.exe train_v21.py --config configs/smoke_a2.json
```

预期输出：每步 `STEP {...}`、每 10 步 `EVAL {...}`（top1/top5/频率分组/上下文交换/困难负样本）、checkpoint 保存；resume 后 step 从 30 开始、loss/指标连续，固定码本 hash 一致。

**在再次确认前，不执行 Phase 0.5，也不启动任何正式训练。**

---

## 7. 第二轮审查后的修改（6 项）

本节对应你提出的 6 条，只改代码，未执行 smoke / 未启动训练。

### (1) hard_neg_win 改为独立困难负样本指标

- **修改位置**：`train_v21.py::evaluate()`
- **修改前**：从模型全词表相似度里取「最相似的非目标 token」，再判断目标是否排它前面；这等价于 held Top-1。
- **修改后**：固定负样本来源为「训练集最高频 top-512 token 池」；对每个目标 token，用固定 `seed + token_id` 从池中抽 `K=16` 个负样本（排除目标本身），只在这 K 个候选上计算目标胜率。
- **指标**：`hard_neg_win`（目标相似度 > 每个固定负样本相似度的比例）；`hard_neg_source` 字符串随评估结果一起输出，明确来源。
- **验收**：指标计算不再依赖模型全词表 argmax；改变模型排序时该指标可独立变化；config 中 `hard_neg_pool/k/seed` 固定。

### (2) 上下文交换评估修正

- **修改位置**：`train_v21.py::evaluate()`
- **修改前**：随机新上下文的预测与「原上下文的目标 y」比较，目标不匹配；旧版指标口径错误（旧指标已删除）。
- **修改后**：
  - `other_ctx_acc`：随机新上下文 + **它自己的正确目标** 计算准确率；
  - `ctx_change_rate`：保留原预测变化率；
  - **受控上下文替换**：保留最后一个 token、其余位置替换为随机 token，计算 `ctrl_keep_last_acc` 与 `ctrl_keep_last_change`。
- **验收**：`other_ctx_acc` 目标来自新上下文自身；`ctrl_keep_last_*` 与原始目标配对；输出中不再包含旧指标名。

### (3) 排除 BlockBank 未使用的 comm 参数

- **修改位置**：`model_v21.py::__init__`、`train_v21.py::trainable_parameters()`
- **修改前**：BlockBank 复用了旧模块，`comm1/comm2/comm_beta` 从未在前向调用，但仍是 `requires_grad=True`，会被算进参数量和优化器。
- **修改后**：`model_v21` 中冻结 `bank.comm1/comm2/comm_beta`；`train_v21` 统一用 `trainable_parameters()` 计算参数量、优化器参数和梯度裁剪。
- **验收（静态检查已通过）**：comm 参数全部 `requires_grad=False`；trainable count 比 total count 少 comm 参数量；优化器只含实际参与前向的参数。

### (4) resume 配置兼容性校验

- **修改位置**：`train_v21.py::validate_resume_config()`、resume 分支
- **修改前**：只校验 format / codebook mode / fixed hash，模型结构、数据、loss、负样本设置不一致也会被恢复。
- **修改后**：`flatten_cfg` 深度比较 checkpoint 里保存的完整 config 与当前 config；不一致直接 `ValueError` 拒绝。
  - 允许变化：`training.steps`、`logging.metrics_jsonl/log_every/eval_every`、`checkpoint.save/save_every`、`device`；
  - 其余（model/data/loss/optim/eval/seed 等）必须一致。
- **验收（静态检查已通过）**：`training.steps` 改变可通过；`model.d` 改变会被拒绝并列出差异。

### (5) checkpoint 保存时重新计算 codebook hash

- **修改位置**：`train_v21.py::save_checkpoint()`
- **修改前**：hash 在训练开始时计算一次并一直写入后续 checkpoint。
- **修改后**：每次保存时调用 `codebook_hash(model)` 重新计算；固定码本 hash 应保持不变，A1 可训练码本 hash 会随训练变化。
- **验收**：A2 每次保存 hash 相同；A1 hash 可变化；resume 时固定码本仍用最新 hash 校验，不一致拒绝。

### (6) A1 独立可训练码本

- **修改位置**：`model_v21.py`（A1 明确注释 + 独立 `code_table`）、`PLAN_v2_1.md`
- **说明**：A1 的 `code_table` 是独立 `nn.Embedding`，与输入 `tok` 不共享参数；A2 的 `code_table` 固定随机单位向量、`requires_grad=False`。
- **验收（静态检查已通过）**：`code_table is not tok`；A2 码本无梯度；A1 码本参与优化器与 trainable count。

### 静态验证结果（未运行前向/训练）

- A1 码本独立于输入 embedding：True
- A2 码本 trainable：False
- comm 参数全部冻结：True
- trainable/total 参数统计差异 = comm 参数量
- resume 允许字段可通过；`model.d` 不一致被拒绝

**仍未执行 Phase 0.5；等待你审核后再运行。**


---

## 8. 最后两点修订（CUDA RNG + 指标口径）

只改代码/文档，未运行前向、smoke 或训练。

### (1) CUDA RNG checkpoint

- **修改位置**：`train_v21.py::save_checkpoint()` / resume 分支
- **修改前**：只保存 `torch.get_rng_state()`（CPU RNG）；`evaluate()` 里 `torch.randint(..., device=device)` 会消耗 CUDA RNG，resume 后无法完全复现。
- **修改后**：
  - checkpoint 新增 `rng_cuda = torch.cuda.get_rng_state_all()`（CUDA 不可用时为 `None`）；
  - resume 时若 `rng_cuda` 存在且 CUDA 可用，调用 `torch.cuda.set_rng_state_all()` 恢复；CPU 环境下安全跳过。
- **验收**：同一 checkpoint 在 GPU resume 后，eval 的随机上下文/随机 token 序列可复现；CPU 环境下不会因缺少 CUDA 报错。

### (2) `other_ctx_acc` 口径澄清

- **修改位置**：`train_v21.py::evaluate()` 文档字符串与注释、本报告、`PLAN_v2_1.md`
- **明确含义**：`other_ctx_acc` 是**随机另一条上下文 + 它自己的正确目标**的 top-1 绝对准确率；它不是配对上下文交换后的准确率变化。
- **上下文敏感性指标**：
  - `ctx_change_rate`：原上下文 vs 随机上下文的预测变化率；
  - `ctrl_keep_last_acc`：保留最后一个 token、替换其余后的准确率；
  - `ctrl_keep_last_change`：上述受控替换导致的预测变化率。
- **验收**：代码注释、报告和 PLAN 均按此口径描述，不再把 `other_ctx_acc` 当作配对交换准确率。
