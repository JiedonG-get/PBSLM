# Phase 0.5 smoke 结果（A1 / A1-resume / A2）

执行时间：2026-11-15 02:45–02:50
状态：**只运行 30/50 步 smoke，未启动正式训练。**

## 1. 执行命令与结果

```bash
cd E:/PycharmProject/experiments/online_v12

# A1：30 步
C:/Python313/python.exe train_v21.py --config configs/smoke_a1.json
# A1 resume：30 -> 50 步
C:/Python313/python.exe train_v21.py --config configs/smoke_a1.json \
  --resume ckpt_smoke_a1.pt --steps 50
# A2：30 步（固定码本）
C:/Python313/python.exe train_v21.py --config configs/smoke_a2.json
```

三次均 `exit 0`；日志：
- `phase05_a1.log`
- `phase05_a1_resume.log`
- `phase05_a2.log`
- 指标：`metrics_smoke_a1.jsonl`（30+20 行）、`metrics_smoke_a2.jsonl`（30 行）

## 2. Checkpoint 验证结果

| 检查 | A1（可训练码本） | A2（固定码本） |
|---|---|---|
| checkpoint step | 50（resume 后） | 30 |
| format | v2.1 | v2.1 |
| codebook mode | trainable | fixed |
| stored hash == 当前码本 hash | **True** | **True** |
| optimizer state entries | 20 | **19**（少一个：码本不参与优化） |
| optimizer 内 Adam step | **50**（resume 恢复） | 30 |
| `rng_cuda` 已保存 | 是 | 是 |
| data/neg generator 状态 | 已保存 | 已保存 |
| 固定码本可复现 | — | **与同 seed 新构造模型完全一致，requires_grad=False** |
| A1 码本参与训练 | **是**（trainable list 包含 code_table） | — |

resume 证据：
- `phase05_a1_resume.log`：`resumed step=30 from ckpt_smoke_a1.pt`；
- 续训第 1 步为 `STEP {"step": 30, "loss": 3.9899, ...}`；原 30 步最后为 step29 loss 4.079，优化器 Adam step 恢复为 50；
- checkpoint 里保存的 config `training.steps=30`（CLI `--steps 50` 只覆盖本次运行，允许字段）。

**负向测试（配置不兼容拒绝）**：
- 用临时 config 把 `model.d` 从 256 改成 512，再 resume A1 checkpoint；
- 程序在训练前抛出 `ValueError: resume 配置不兼容，拒绝恢复；model.d: ckpt=256 config=512`，exit 1；
- 日志：`phase05_bad_resume.log`。

## 3. 评估指标（Phase 0.5 仅验证流程，不代表模型质量）

评估每 10 步一次，`windows=64`。A1 的 30 步 + resume 20 步：

| run | eval# | held_top1 | held_top5 | cos | hard_neg_win | ctx_change | other_ctx_acc | ctrl_keep_last_acc | ctrl_keep_last_change |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 1 | 0.0566 | 0.1035 | 0.0044 | 0.5385 | 0.0 | 0.0664 | 0.0566 | 0.0 |
| A1 | 2 | 0.0312 | 0.1309 | −0.0479 | 0.5277 | 0.0 | 0.0195 | 0.0312 | 0.0 |
| A1 | 3 | 0.0312 | 0.1172 | −0.0812 | 0.5199 | 0.0 | 0.0195 | 0.0312 | 0.0 |
| A1-resume | 4 | 0.0566 | 0.1074 | −0.0836 | 0.4874 | 0.0 | 0.0664 | 0.0566 | 0.0 |
| A1-resume | 5 | 0.0566 | 0.0996 | −0.1061 | 0.4700 | 0.0 | 0.0664 | 0.0566 | 0.0 |

A2 的 30 步：

| run | eval# | held_top1 | held_top5 | cos | hard_neg_win | ctx_change | other_ctx_acc | ctrl_keep_last_acc | ctrl_keep_last_change |
|---|---|---|---|---|---|---|---|---|---|
| A2 | 1 | 0.0566 | 0.0859 | 0.0455 | 0.5389 | 0.0 | 0.0664 | 0.0566 | 0.0 |
| A2 | 2 | 0.0332 | 0.1152 | 0.0502 | 0.5476 | 0.3164 | 0.0215 | 0.0332 | 0.3184 |
| A2 | 3 | 0.0566 | 0.1152 | 0.0606 | 0.5414 | 0.0 | 0.0664 | 0.0566 | 0.0 |

训练 loss（首/末）：
- A1：4.9404 → 4.0790（30 步）；resume：3.9899 → 4.7045（30→50 步）
- A2：5.1852 → 4.2757（30 步）

频率分组 top-1（A1/A2 均）：高频（≥1%）0.23–0.42，中频/低频/稀有 **0.0**。这是 30 步 smoke 的预期早期状态，也再次说明后续不能只看总 top-1。

## 4. 观察 / 遗留

- **流程通过**：前向、反向、优化器 step、checkpoint 保存、resume、固定码本冻结、配置校验、CUDA RNG 保存均正常。
- **`ctx_change_rate=0` 只说明“当前评估扰动下预测 token 未变化”，不能证明输出向量本身是常量**；向量可能变化但 argmax 未变。准确结论：30–50 步内模型尚未表现出稳定、可观测的上下文敏感性；A2 只有一次 eval 出现变化（0.3164），属于早期噪声。
- 出现 NumPy memmap 只读 warning（`torch.from_numpy`），不影响数值；**已在 `train_v21.py` 中用 `data[...].copy()` 修复**（smoke 日志是修复前产生的，未重跑）。
- **参数规模**：A1 trainable 21.09M，A2 trainable 16.90M（差一个可训练码本，符合设计）。
- **可复现性缺口**：当前只验证了 resume 能继续训练；尚未做“从头连续 50 步 vs 30 步+resume 20 步”的逐步参数/loss 对照。这是后续正式训练前的可选加强项，不是本次 smoke 的阻断问题。
- Phase 0.5 不用于判断模型能力；下一步应按 PLAN 跑 Baseline A1/A2 的正式预算（同数据/同步数/同评估），再看低频组、上下文敏感性和困难负样本是否随训练上升。

## 5. checkpoint 证据补充

smoke checkpoint 本体在本地，sha256：

- `ckpt_smoke_a1.pt`：`0e32123345ba3c6463d210a5bf3a3ec72e511f94bd4521413ee0f88e6424a348`
- `ckpt_smoke_a2.pt`：`d40b214f270c4c09006b179a30ff06dae425351203ae36a351ebc980aab7fded`

报告中的 optimizer/RNG/码本结论来自对这些 checkpoint 的反序列化检查（见第 2 节），而不只是运行时日志。
