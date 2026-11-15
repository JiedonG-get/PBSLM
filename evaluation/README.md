# Evaluation

- `phase1_1_eval.py`：Phase 1.1/1.1.1 固定 held-out 评估（top1/top5、频率分组、上下文敏感性/效用、自然消融）。
- 当前 9 次前向：原始 + 8 个变体，包括 `last_token_only_masked`、`last_token_repeated`、`drop_last1/2/4`。
- 注意：`eval_a1.json`/`eval_a2.json` 是 Phase 1.1 旧结果；修正后的结果见 `eval_a1_p111.json`/`eval_a2_p111.json`。
- 配置：`configs/eval_phase1_1.json`。
- 指标定义：`docs/EVAL_PROTOCOL.md`。

运行示例（需要本地 checkpoint，仓库不包含 `.pt`）：

```bash
python evaluation/phase1_1_eval.py --checkpoint ckpt_formal_a1.pt --tag a1_p111
python evaluation/phase1_1_eval.py --checkpoint ckpt_formal_a2.pt --tag a2_p111
```

默认 10,000 windows × AR8 = 80,000 个 target；6 次前向（原始 + 5 个变体），不训练。
