# DECISIONS — 历史技术决策与理由

按时间顺序记录关键决策，避免以后重复踩坑。日期以实验推进阶段标注，不追求精确到天。

## 2026-11 早期：Qwen IO → 自训 tokenizer/head

- 早期使用冻结 Qwen3-1.7B 的 embedding + lm_head 作为输入/输出接口；
- 后来决定**不再依赖现成 LLM 的 tokenizer/head**：自训 BPE、自训连续输出头；
- 原因：大词表 softmax 不符合 PBSLM 的连续状态输出设计，也受限于显存；
- 结果：zhwiki 自训 BPE 16k → 520M tokens；`tokenizer/` 保留 tokenizer。

## 2026-11：输出不是概率分布

- PBSLM 输出「A 的流动结果或静默」，而不是下一个 token 的概率分布；
- 采用连续向量 + 码本最近邻解码；
- 训练目标从 cosine regression → sampled-negative InfoNCE（Phase 1 实际：1 正 + 采样负例），避免全词表 softmax；
  batch 内负例尚未实现，不能按「已实现」记录。
- 教训：纯 cosine + 随机码本在 batch 平均下有「常数输出」稳定点，模型会背词频。

## 2026-11：课程学习放弃，保留 GradNorm

- 课程学习显式放弃（用户决定）；
- 多 loss 使用 GradNorm 自适应权重（v2 早期）；
- 曾出现 GradNorm 把 LM 权重压到 0.017、KL 主导的崩溃；之后固定部分权重/限制范围。

## 2026-11：v2 完整模型的多项修正

- **field 不能劫持输出**：μ 单位化、场缩放 0.1；预训练期 `field_scale=0` 时 KL 会假性归零，
  因此 field 必须在非零 scale 下单独验收；
- **输出归一化**：cos 目标下 z 范数会爆炸，必须 normalize；
- **记忆读归一化**：未归一化时 A 编码被巨大常量记忆向量吞掉；
- **decoder cross-attn 残差**：q0 固定查询会淹没上下文信号，需要放大上下文残差；
- **码本持久化**：`code` 曾是 non-persistent buffer，加载 ckpt 解码全错，改为 persistent；
- **自训码本 vs 固定码本**：作为 A1/A2 对照，Phase 1 固定码本略好，但单 seed 不下结论。

## 2026-11：非 AR 多位置输出问题

- 早期 `m_out=8` 非自回归输出，位置 1..7 看不到前文，评估被稀释；
- 改为 AR teacher forcing（每个位置用真实前文）后，单 token held decode 才能超过 10%；
- 教训：评估必须和训练监督方式一致。

## 2026-11：训练稳定性

- Adafactor 在 4D 参数 + 半精度下出现 NaN，改用 AdamW / 8-bit Adam；
- PyTorch memmap 只读 warning 用 `data[...].copy()` 修复；
- checkpoint 必须保存 optimizer/step/RNG/generator/codebook hash，否则 resume 不可复现。

## 当前未决问题

1. 完整 v2 的 A/C、Flow、E/M、Field 尚未在严格消融中证明有增益；
2. 低频/稀有 token top1 仍为 0，频率偏置严重；
3. 受控上下文替换显示模型仍主要依赖最后 token；
4. 可训练码本与泛化的关系需要更多 seed；
5. Online / 半在线尚未实现。

## 原则

- 不把「held top1 稍好」当成机制有效的证据；
- 每个组件都要有模块消融 + 预算对照；
- 先修闭环，再做复杂结构；
- 如果一个模块只增加复杂度、没有可测收益，就删掉。
