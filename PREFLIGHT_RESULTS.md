# Preflight 结果（batch=8，A1/A2 各 30 步）

状态：**通过。未启动 15,000 步正式训练，未进入 Phase 2。**
是否覆盖旧文件：**否**。preflight 使用独立 config/metrics/checkpoint 路径，smoke 与正式文件未改动。

## 1. 执行命令

```bash
cd E:/PycharmProject/experiments/online_v12

# A1 preflight
C:/Python313/python.exe train_v21.py --config configs/preflight_a1.json > preflight_a1.log 2>&1

# A2 preflight
C:/Python313/python.exe train_v21.py --config configs/preflight_a2.json > preflight_a2.log 2>&1
```

配置来源：`configs/formal_a1.json` / `formal_a2.json` 复制为临时 `configs/preflight_a*.json`，
只改 steps=30、batch=8、eval_every=10、save_every=10、输出路径；模型/数据/loss/优化器/eval 字段与正式一致。

独立输出：
- `metrics_preflight_a1.jsonl` / `metrics_preflight_a2.jsonl`
- `ckpt_preflight_a1.pt` / `ckpt_preflight_a2.pt`
- `preflight_a1.log` / `preflight_a2.log`

## 2. 稳定性

| 检查 | A1 | A2 |
|---|---|---|
| exit code | 0 | 0 |
| NaN / Inf | 无 | 无 |
| OOM / CUDA error | 无 | 无 |
| checkpoint 保存 | step 10/20/30，最终 step30 | 同左 |
| metrics 行数 | **30**（运行前文件不存在） | **30**（运行前文件不存在） |

metrics 采用 append 模式：本次两个文件都是**新建**的，行数正好 30。
正式训练前 `metrics_formal_a1.jsonl` / `metrics_formal_a2.jsonl` 当前也不存在；正式启动前会再次确认清空/不存在。

## 3. GPU 峰值显存（torch.cuda.max_memory）

| 运行 | allocated | reserved | 备注 |
|---|---|---|---|
| A1 preflight | **0.677 GiB** | **0.732 GiB** | batch8×AR8、fp32、21.09M trainable |
| A2 preflight | **0.633 GiB** | **0.709 GiB** | batch8×AR8、fp32、16.90M trainable |

这是 PyTorch 进程内峰值；不含桌面/驱动占用。

## 4. Step 耗时

| 运行 | 步数 | 首步 | 稳态中位数 | 稳态均值（排除首步） | 日志累计 `sec`（末步） |
|---|---|---|---|---|---|
| A1 preflight | 30 | ~0.60s（warmup） | **0.0385s** | ~0.0387s | 4.0s |
| A2 preflight | 30 | ~0.60s（warmup） | **0.0363s** | ~0.0364s | 3.9s |

按稳态 0.038s/step 估算：15,000 步约 **9.5 分钟**训练 + eval/checkpoint 开销（每卡单进程）。
实际会略高，正式训练前不以此作为硬承诺。

## 5. Checkpoint 检查

| 检查 | A1 | A2 |
|---|---|---|
| step | 30 | 30 |
| mode | trainable | fixed |
| stored hash == 当前码本 hash | True | True |
| optimizer state entries | 20 | **19**（码本不参与优化） |
| Adam step | 30 | 30 |
| `rng_cuda` 已保存 | 是 | 是 |
| data/neg generator 状态 | 是 | 是 |
| 固定码本可复现 | — | **与同 seed 新模型完全一致，requires_grad=False** |
| trainable params | 21.09M | 16.90M |

## 6. 旧文件隔离检查

- `ckpt_smoke_a1.pt` / `ckpt_smoke_a2.pt` 时间戳仍为 smoke 运行时间，未被覆盖；
- `ckpt_formal_a1.pt` / `ckpt_formal_a2.pt` 当前不存在（正式训练未启动）；
- `metrics_smoke_*` 未被 preflight 写入。

## 7. 结论

preflight 通过：batch=8 下无 OOM/NaN，checkpoint 正常保存并可通过反序列化校验，显存/耗时满足本地正式训练条件。

仍待批准：是否启动 `configs/formal_a1.json` / `formal_a2.json` 的 15,000 步正式训练。
