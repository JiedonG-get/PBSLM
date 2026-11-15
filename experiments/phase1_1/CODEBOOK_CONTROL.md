# 同初始化 Codebook 对照（设计，未训练）

目的：把「码本是否可训练」从 Phase 1 的混杂中单独拆出来。

## 对照设计

| 项 | trainable | frozen |
|---|---|---|
| config | `configs/ctrl_codebook_trainable.json` | `configs/ctrl_codebook_frozen.json` |
| codebook | trainable | fixed |
| codebook_init | fixed_seed | fixed_seed |
| codebook_seed | 1234 | 1234 |
| 总参数量 | 21,160,328 | 21,160,328 |
| 可训练参数量 | 21,094,400 | 16,900,096 |
| 差异 | 码本 4,194,304 参数可训练 | 码本冻结 |

- 两份配置的 model 总参数完全相同；
- 初始码本都用同一 seed=1234 的单位向量（trainable 版不会再被 normal 初始化覆盖）；
- 不添加 adapter；**不为了「参数量匹配」人为增加额外参数**；
- 报告时必须同时给出 total params 与 trainable params，不能只报其一。

## 与 Phase 1 A1/A2 的区别

Phase 1 的 A1/A2 同时存在：

1. 码本初始化不同（A1 normal 初始化，A2 fixed seed 单位向量）；
2. 可训练参数量不同（差 4.19M）；
3. 码本是否进入优化器不同。

因此 Phase 1 的 A1 vs A2 不能单独归因于「固定码本更好/更差」。
本对照只解决第 2、3 项；第 1 项通过相同 `codebook_init=fixed_seed` 消除。

## 状态

- 配置已就绪；**尚未训练**；
- 未进入 Phase 2；等 Phase 1.1 审查后再决定是否运行该对照。
