# O1 Test Notes

## Environment

- Python: 3.13.5
- PyTorch: 2.11.0+cu128
- Tests run on CPU (no GPU required)
- `torch.randn_like(..., generator=g)` was verified valid in this environment:
  ```text
  randn_like_generator_ok [1.5409960746765137, -0.293428897857666]
  ```
  If an older PyTorch raises `TypeError`, replace with:
  `torch.randn(p.shape, generator=g, dtype=p.dtype, device=p.device)`.

## Command

Run from repo root:

```bash
python -m unittest tests.test_online_o1 -v
```

Full output: `tests/o1_test_output.txt`.

## Covered

| 条件 | 状态 |
|---|---|
| fp32 | 已覆盖 |
| batch>1（batch=2） | 已覆盖 |
| n_in=16 | 已覆盖 |
| `direct_only=False` 完整 out_a/out_c 路径 | 已覆盖 |
| `direct_only=True` ctx_a/ctx_c 路径 | 已覆盖 |
| `dense=False`, `use_flow=True` | 已覆盖 |
| 六组 A 参数逐组扰动 | 已覆盖 |
| C 的 `chosen`/`active`/`z_c`/`I_c`/top-k（完整 batch×m） | 已覆盖 |
| 受保护参数 hash | 已覆盖 |
| optimizer param_groups（只构造，不 step） | 已覆盖 |
| M 全零 | 已覆盖 |
| M 已写入 | 已覆盖；包含 chosen/active/z_c/I_c/完整 batch top-k |
| field 关闭 | 已覆盖 |
| field 开启（field_scale=0.1） | 已覆盖；六组 A 参数逐组扰动 |
| 旧 checkpoint 缺少 out_c 的兼容加载 | 已覆盖（合成 state） |
| `role_a` 耦合诊断 | 已覆盖；A 选块变化已确认，C 变化已观察 |
| 前向基线 / 无 NaN | 已覆盖 |

## Not Covered

以下条件**尚未验证**，不能写成已通过：

- bf16 / fp16；
- 多 seed 的 C 稳定性（当前固定 seed=0）；
- GPU 与 CPU 结果一致性；
- 其他 n_in / m / k / depth 配置；
- M 衰减、容量、冲突策略；
- shock / emotion 的运行时驱动；
- 真实旧 checkpoint（本测试用合成 state；测试集中无 v2 checkpoint）；
- 真实 optimizer.step / backward / 训练 loss（O1 明确禁止，也不应覆盖）。

## Scope Boundary

O1 只验证：

- A/C 输出路径拆分；
- O1 六组 A 参数白名单；
- 基础 C 稳定性（逐组扰动 + top-k + protected hash）；
- `role_a` 路由耦合诊断。

**O1 不代表 Online Learning 已实现。** A/C 路由和跨块交互的结构性耦合属于 O1.5；
Runtime 状态骨架属于 O2。
