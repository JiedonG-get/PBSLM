# PBSLM — Parallel Block State Language Model

**一个面向「并行块 + 双路径判断 + 持续记忆 + 信息流动」的实验性语言模型原型。**

> 当前状态：Phase 1 Baseline A1/A2 已完成；后续已补充 masked A/C 工程验证与小型合成持续适应、记忆干预实验。
> v2.1 仍是单路径语言模型 baseline；v2 的可选 masked 路径已有局部证据，**不代表完整 Online Learning 系统或通用语言能力已验证**。
> 最新证据入口：[研究执行边界](docs/RESEARCH_EXECUTION.md)、[O1.5验证](experiments/o1_5/GPT_VERIFICATION_REPORT.md)、[持续适应](experiments/continual_associations/RESULTS.md)、[随机映射记忆](experiments/episodic_memory/RESULTS.md)。

---

## 1. 我们在研究什么

我们想回答的核心问题不是「像不像大脑」，而是：

> **在相近的计算预算下，一个具有多个独立计算块、A/C 双路径判断、持续记忆 M、
> 块间信息流动和连续向量输出的模型，能否获得比普通单路径模型更好的上下文预测能力？**

围绕这个目标，PBSLM 的设计包含五条主线：

1. **并行 BlockBank**
   多个独立权重的块组成块池，每次只激活一部分。目标是让「参数容量」和「实际计算量」解耦。
2. **A / C 双路径**
   A 路径可以读取记忆 M，C 路径不读 M；两条路径对同一上下文做预测，用一致性作为仲裁信号。
3. **E 块 + 外部记忆 M**
   E 块常驻参与处理，并维护每个 E 块的 `d×d` 关联记忆矩阵；A 侧读取，C 侧隔离。
4. **InfoFlow**
   活跃块之间做低维信息交换，得到 settled 状态和块间一致性 `agree`。
5. **连续向量输出 + PerturbField**
   不直接维护词表 softmax，而是预测 token 对应的连续向量，再用最近邻码本解码；
   输出随机性由可学习扰动场 `μ/σ` 控制，输入是活跃度、震撼值、情绪旋钮。

此外，项目的长期核心目标是 **Online / 半在线学习**：将稳定知识参照 C、可适应路径 A 与经历记忆 M 分离，使模型持续接收信息、适应反馈，同时保留旧能力，并最终支持提交、打断和续连。更好的上下文预测是基础验证，不是这一目标的全部。当前合成实验已观察到 A 能适应但会遗忘、回放改善保持，以及训练后的 M 能读出新关联；尚未完成自然语言持续学习或实时运行系统（见上述最新证据与 `docs/ONLINE.md` 的历史设计）。

---

## 2. 当前进展（Phase 1）

Phase 1 只跑了最小 baseline：

- 单路径、自回归 teacher forcing；
- 连续向量输出 + **sampled-negative InfoNCE**（Phase 1 实际：1 个正样本 + 采样负例；不是 batch 内负例，也不做全词表 softmax）；
- **A1：独立可训练码本；A2：固定随机单位码本**
- 数据：520M tokens（自训 BPE 16k，zhwiki）；15,000 步 × batch 8。

| 指标（最后 5 次 eval 平均） | A1（可训练码本） | A2（固定码本） |
|---|---|---|
| held_top1 | 0.0824 | **0.0942** |
| held_top5 | 0.1578 | 0.1570 |
| hard_neg_win | 0.5191 | **0.5738** |
| ctx_change_rate | 0.5488 | **0.5985** |
| 高频组 top1 | 0.583 | 0.586 |
| 中频组 top1 | 0.031 | **0.122** |
| 低频/稀有组 top1 | 0.0 | 0.0 |

**诚实的结论：**

- 15k 步后**预测变化率**上升（`ctx_change_rate` 从 ~0.29 升到 ~0.6）；
  但该指标同时改变了目标与上下文，且受控替换是 OOD 随机 token，**不能单独证明上下文敏感性或上下文利用能力**（Phase 1.1 修正）；
- 但模型仍强烈依赖最高频 token / 最后一个局部 token：低频与稀有 token top1 仍为 0，
  `hard_neg_win` 接近随机；
- 单次实验下 A2 在部分指标上略好于 A1，但 A1/A2 同时存在**码本初始化不同 + 可训练参数量不同（差 4.19M）**的混杂，不能据此宣布架构优劣；
- 下一步不是放大模型，而是按 `docs/EXPERIMENT_PLAN.md` 做受控消融与更严格的评估。

完整数据见 [`experiments/phase1/PHASE1_RESULTS.md`](experiments/phase1/PHASE1_RESULTS.md)。

Phase 1.1 用固定 10k windows（80k targets）重新评估，并区分 Context Sensitivity / Utility / Generalization：
[`experiments/phase1_1/PHASE1_1_1_RESULTS.md`](experiments/phase1_1/PHASE1_1_1_RESULTS.md)（Phase 1.1.1 修正版）。
关键发现：masked last-token-only 相比完整上下文少 2.7 / 2.5 pp；删前缀/打乱前缀几乎不影响，
说明当前上下文利用主要是**局部后缀**，长前缀利用尚未证明。

---

## 3. 仓库结构

```text
PBSLM/
├── README.md                     # 本文件：目标、现状、快速开始
├── CHANGELOG.md                  # 变更记录
├── docs/
│   ├── MASTER.md                 # 完整架构设计
│   ├── ARCHITECTURE.md           # A/C/E/M 信息流与公式
│   ├── ONLINE.md                 # Online / 半在线设计
│   ├── EXPERIMENT_PLAN.md        # Phase 0 → Phase 2 实验路线
│   ├── DECISIONS.md              # 历史技术决策与理由
│   └── REPO_LAYOUT.md            # 目录与 artifact 策略
├── models/
│   ├── model.py                  # BlockBank / CrossLayer / InfoFlow / v15 PBSLM
│   ├── model_v2.py               # 完整 v2：A/C + E/M + Flow + 连续输出 + PerturbField
│   ├── model_v21.py              # v2.1 Baseline A：单路径连续向量（A1/A2）
│   └── README.md
├── training/
│   ├── train_v2.py               # v2 实验训练
│   ├── train_v21.py              # v2.1 配置驱动训练（InfoNCE / checkpoint / eval）
│   └── README.md
├── configs/                      # baseline / formal / preflight / smoke 配置
├── scripts/
│   ├── data/                     # BPE、tokenize、语料下载/解析
│   └── analysis/                 # 历史分析与评估脚本
├── evaluation/                   # Phase 1.1 评估脚本
├── experiments/
│   ├── phase0/                   # Phase 0 代码整理与计划
│   ├── phase05/                  # smoke / preflight
│   ├── phase1/                   # 正式 A1/A2
│   ├── phase1_1/                 # 固定 held-out 评估 / 上下文敏感性/效用
│   └── legacy/                   # v12–v15 历史结果、日志、metrics、旧代码
└── tokenizer/                    # 自训 BPE 16k tokenizer 与统计
```

**不上传**：所有 `*.pt` checkpoint、`.bin` 语料二进制、缓存与本地备份。
详细策略见 `docs/REPO_LAYOUT.md`。

---

## 4. 快速开始

### 依赖

```bash
pip install torch numpy tokenizers
```

### 数据准备

仓库不包含语料二进制。需要准备：

1. 原始文本（例如 zhwiki 清洗后的 `zhwiki_full.txt`）；
2. 用 `scripts/data/train_bpe.py` 在语料上训 BPE；
3. 用 `scripts/data/tokenize_corpus.py` 生成 `data/bpe_full_u32.bin`（uint32 tokens）。

```bash
python scripts/data/train_bpe.py --text <corpus.txt> --out tokenizer/bpe16k_tokenizer.json
python scripts/data/tokenize_corpus.py --text <corpus.txt> --out-prefix data/bpe_full \
  --tokenizer tokenizer/bpe16k_tokenizer.json
```

### Baseline A1（可训练码本）

```bash
python -m training.train_v21 --config configs/formal_a1.json
```

### Baseline A2（固定码本）

```bash
python -m training.train_v21 --config configs/formal_a2.json
```

Phase 0.5 smoke：

```bash
python -m training.train_v21 --config configs/smoke_a1.json
python -m training.train_v21 --config configs/smoke_a2.json
```

配置文件里的路径相对于仓库根目录；checkpoint 会写到 `experiments/phase*/checkpoints/`。
训练脚本在每次 eval 时输出 top1/top5、频率分组、`ctx_change_rate`、`ctrl_keep_last_*`、
`hard_neg_win` 等指标；逐步 metrics 写入 `experiments/phase*/metrics/*.jsonl`。

---

## 5. 文档

- [docs/MASTER.md](docs/MASTER.md)：架构全貌与当前实现边界
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)：A/C、E/M、InfoFlow、连续输出、扰动场
- [docs/ONLINE.md](docs/ONLINE.md)：Online 设计理念与冻结/更新策略
- [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)：Phase 0/0.5/1 结果与 Phase 2 消融计划
- [docs/DECISIONS.md](docs/DECISIONS.md)：为什么做这些技术选择，以及踩过的坑
- [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md)：Phase 1.1 评估指标定义
- [CHANGELOG.md](CHANGELOG.md)：按时间线的变更记录

---

## 6. 研究边界

- 这是**研究原型**，不是可用的通用语言模型；
- 当前 Phase 1 只是单路径 continuous-vector baseline；
- A/C、E/M、InfoFlow、PerturbField 的收益尚未被严格证明；
- 我们不会用 held top-1 单一指标下结论，重点看低频 token、上下文敏感性、困难负样本和 预算对照。

如果后续实验支持这些机制，它们才值得被保留；否则就删掉复杂度。
