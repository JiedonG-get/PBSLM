# Baseline A1/A2 正式训练配置（待审，尚未启动）

## 1. 配置路径与公平性

- A1：`configs/formal_a1.json`（`model.codebook = "trainable"`）
- A2：`configs/formal_a2.json`（`model.codebook = "fixed"`，seed=1234）

深度 diff 后，两份配置**只有三处不同**：

| 字段 | A1 | A2 | 说明 |
|---|---|---|---|
| `model.codebook` | trainable | fixed | 实验变量 |
| `logging.metrics_jsonl` | metrics_formal_a1.jsonl | metrics_formal_a2.jsonl | 输出隔离 |
| `checkpoint.save` | ckpt_formal_a1.pt | ckpt_formal_a2.pt | 输出隔离 |

其余完全相同：seed=1234、数据 `bpe_full_u32.bin`、hold=0.15、n_in=16、ar_m=8、
steps=15000、batch=8、loss=InfoNCE tau=0.1 neg_n=128 exclude_positive=true、
AdamW lr=1e-3、eval windows=64、hard_neg_pool=512/k=16/seed=4321。

## 2. 预算与输出

- 训练预算：15,000 步 × batch 8 × AR 8 位置 = 960,000 个 next-token 样本；
- checkpoint：每 1,000 步保存，路径独立于 smoke；
- metrics：每步写 `metrics_formal_{a1,a2}.jsonl`，每 1,000 步 eval；
- 日志建议：`phase1_a1.log` / `phase1_a2.log`（正式运行时创建）。

## 3. 参数规模差异（预期）

- A1 trainable ≈ 21.09M（输入 embedding + 独立可训练码本 + BlockBank）；
- A2 trainable ≈ 16.90M（输入 embedding + 固定码本 + BlockBank，码本不进入优化器）。

这正是 A1/A2 的实验变量，不是配置不公平。

## 4. 尚未做的验证

- 未做「从头连续 50 步 vs 30 步+resume 20 步」的逐步参数/loss 对照；
- 未启动正式训练。

## 5. 启动命令（待批准后才执行）

```bash
cd E:/PycharmProject/experiments/online_v12
C:/Python313/python.exe python -m training.train_v21 --config configs/formal_a1.json > phase1_a1.log 2>&1
C:/Python313/python.exe python -m training.train_v21 --config configs/formal_a2.json > phase1_a2.log 2>&1
```

---

## 6. Preflight（batch=8，A1/A2 各 30 步）已通过

- 独立临时配置：`configs/preflight_a1.json` / `preflight_a2.json`；
- 独立输出：`metrics_preflight_a1.jsonl`、`metrics_preflight_a2.jsonl`、`ckpt_preflight_a1.pt`、`ckpt_preflight_a2.pt`；
- 无 NaN/Inf/OOM；checkpoint 正常保存并通过反序列化校验；
- GPU 峰值：A1 0.677 GiB allocated / 0.732 GiB reserved；A2 0.633 / 0.709 GiB；
- 稳态 step 耗时：A1 0.0385s、A2 0.0363s（首步 ~0.6s warmup）；
- metrics 文件：运行前不存在，运行后各 **30 行**，append 未被污染；
- smoke/正式文件未被覆盖；`ckpt_formal_*` 仍不存在。

正式 15,000 步配置未改动，等待批准后启动；正式 metrics 路径当前为空，启动前会再次确认。
