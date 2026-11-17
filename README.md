# PBSLM — Parallel Block State Language Model

**以持续 Online 运行为首要目标的模型与系统研究原型。**

我们希望模型在输入不断到达时持续维护理解状态，能够选择说或继续听，在新信息使当前表达过时后及时打断，并保留已说内容、按新信息自然续接。长期面向音视频等连续环境，但当前只验证了有限文本与符号任务。

**Online 首先指持续运行，不等同于 Online Learning。** 临时工作状态、经历记忆 M 与慢速 A 内化服务于这个目标，不替代它。研究允许修改乃至舍弃原有结构，具体机制是否保留由对照实验决定。

## 当前状态（2026-11-18）

最新阶段已从固定 ID 推进到连续动态键：标准差值快速状态改善了随机键绑定，但相似键、写入噪声和文本迁移仍会失败；显式内容槽与保守写入也有容量和新信息接收代价。候选已接入结构化事件的异步运行器，但没有整合成原 PBSLM 的语言理解能力。见[动态键阶段报告](docs/DYNAMIC_KEY_STAGE_REPORT.md)。

已经有可运行的异步调度骨架、真实 PBSLM 小模型推理接线，以及内部状态记忆的独立结构候选。**尚未形成一个可靠持续理解、流畅对话的完整模型，也没有证明 PBSLM 整体优于普通模型加检索。**

| 方向 | 已有证据 | 尚未证明 |
|---|---|---|
| 持续运行 | 输入与推理/输出交叠、过期输出隔离、打断与重规划、静默输入处理、背压与任务清理 | 语义级自然续连、模型自主说听、硬实时、音视频理解 |
| 内部工作状态 | 无原始历史查询的递归/分槽候选；选择性写入减少长期干扰 | 动态未知实体、可靠寻址、容量超限行为、语言状态泛化 |
| M→A 内化 | 关闭学生 M 后可学入教师给出的具体关联 | 稳定旧能力、广泛换表达/组合迁移、可靠选择与撤销内化 |
| 历史回退 | 已保存预测与实际观察字典的离线回放，记录拒答、命中与维护成本 | live 集成、可靠触发、总计算或延迟节省 |

这些证据来自不同实验副本。**运行器、符号状态模型、读出候选与原 PBSLM 主模型尚未统一整合**，不能把各自的成功拼成完整系统已完成的结论。

### 最重要的发现与失败

- 全部新输入都硬打断会造成输出饥饿；全部当背景又会产生过时断言。目前只有调用方声明或保守规则决定相关性，没有模型自主相关性保证。见[策略对照](experiments/continuous_online/ROUTING_COMPARISON.md)。
- 原 M→A 更新能学具体答案，但会遗忘，换问法表现有限；软教师并未自动解决。见[内化实验](experiments/consolidation/RESULTS.md)与[软教师对照](experiments/consolidation/SOFT_SHARED_CONTROL.md)。
- 已训练 E 特征在部分小任务中有用，但扩大到 128 个随机组合键后，原高分没有直接扩展。见[多批内化](experiments/consolidation/SEQUENTIAL_RANDOM_E.md)与[容量负结果](experiments/consolidation/PAIR_CAPACITY.md)。
- 同为 96 个状态浮点数，学习型分槽在 384 步均匀符号流中约 97%；但连续 8,192 次更新单一热实体后，冷记录均值降至约 18.2%。短流高分不是长期保持保证。见[分槽对照](experiments/recurrent_state/SLOTTED_CONTROL.md)与[长流失败](experiments/recurrent_state/SKEW_RETENTION.md)。
- 匹配稀疏写入训练改善了偏斜流保持，但寻址碰撞仍失败。状态大小相同不等于参数、计算或接口辅助完全相同。见[匹配训练与残余失败](experiments/recurrent_state/SPARSE_MATCHED_TRAINING.md)。
- 有足够显式槽位时，普通最新值字典是很强的基线；回退实验的高分不能解释为神经模型已经解决记忆。字典仍维护全部写入，尚无整体省算力证据。见[字典基线](experiments/retrieval_baseline/RESULTS.md)与[回退回放](experiments/recurrent_state/FALLBACK_REPLAY.md)。

阶段结论集中见[内部状态优先报告](docs/INTERNAL_STATE_STAGE_REPORT.md)。具体数值以各实验的配置、来源哈希、原始结果和限制说明为准；不同任务的百分比不能直接横向比较。

## 快速体验持续运行原型

从仓库根目录安装依赖：

```bash
pip install torch numpy tokenizers
```

无需 checkpoint 的规则参考后端：

```bash
python -m runtime.online_cli --backend reference
```

逐行输入示例：

```text
/listen 甲号设备的颜色是红。
甲号设备的颜色是什么？
把甲号设备的颜色更新为绿。
/interrupt
/resume
/quit
```

`/listen` 接收并处理但不要求说话；`/interrupt` 阻止后续旧输出提交；`/resume` 按当前输入与已提交前缀重规划。**这是逐行终端，不是实时语音系统。** `/wait` 是测试/管道用屏障，普通交互无需使用。

| 后端 | 实际内容 | 依赖 |
|---|---|---|
| `reference` | 精确句法、符号状态、规则表达 | 不需要模型文件 |
| `reader` | 固定小 GRU 提取、符号状态、规则表达 | 本地 reader 权重 |
| `pbslm-query` | 真实固定 PBSLM 推理，单 token 颜色回答 | 本地文本模型权重 |

后两者需要本地研究 checkpoint，仓库不包含权重。**所有后端都不是通用聊天机器人**；规则表达不能当作神经生成能力，reader 也已知会误接受某些否定/不确定陈述。完整命令与限制见[终端说明](runtime/ONLINE_CLI.md)。

## 原始架构与当前候选

原始 PBSLMv2 包括：

- **BlockBank**：独立权重的并行块池，探索激活计算与参数容量的关系。
- **A/C 双路径**：A 可读取 M，C 不读取 M。在线实验中冻结的 C 是稳定参照；C 不变不等于 A 不遗忘，一致性也不等于正确性。
- **E 与 M**：E 产生键表示，每个 E 对应关联矩阵 M。现有累加写入会干扰，覆盖/撤销需明确机制。
- **InfoFlow**：活跃块间的信息交换与一致性信号。
- **连续向量输出与 PerturbField**：向量预测、码本解码和扰动控制。其总体收益尚未被严格证明。

v2.1 是单路径语言 baseline；v2 的 `use_sample_masks=True` 是可选独立逐样本 A/C 路径，默认仍保留 legacy 行为。实现、历史设计和未来设想不能混为一谈。见[架构总览](docs/MASTER.md)、[masked 路径验证](experiments/o1_5/GPT_VERIFICATION_REPORT.md)与[历史 Online 设计](docs/ONLINE.md)。

后来新增的冻结特征读出、累计/可撤销约束、低秩统计、全局/分槽状态等，均为**独立研究候选**，没有统一替换生产模型。它们不都属于原 A/M 参数，也不都属于参数内知识。见[结构假设](docs/ARCHITECTURE_HYPOTHESES.md)与[工作记忆角色设计](docs/WORKING_MEMORY_ROLE_DESIGN.md)。

## 实验入口与证据

| 问题 | 报告 |
|---|---|
| 语言基础与上下文利用 | [Phase 1](experiments/phase1/PHASE1_RESULTS.md)、[固定 held-out 重评](experiments/phase1_1/PHASE1_1_1_RESULTS.md) |
| A 适应、遗忘与回放 | [持续适应](experiments/continual_associations/RESULTS.md) |
| M 读出与持续干扰 | [随机映射记忆](experiments/episodic_memory/RESULTS.md)、[持久记忆](experiments/persistent_memory/RESULTS.md) |
| 中文表面迁移与错误抽取 | [文本事实](experiments/text_facts/RESULTS.md)、[新问法](experiments/text_surfaces/RESULTS.md)、[事实候选](experiments/fact_reader/RESULTS.md)、[校准失败](experiments/fact_reader/CALIBRATION.md) |
| 持续运行与状态处理 | [异步运行](experiments/continuous_online/RESULTS.md)、[独立输入处理](experiments/continuous_online/CONTINUOUS_STATE.md)、[过载](experiments/continuous_online/OVERLOAD.md) |
| 特征、约束与泛化 | [冻结 E 读出](experiments/consolidation/FEATURE_READOUT.md)、[同损失历史对照](experiments/consolidation/SAME_LOSS_HISTORY.md)、[压缩](experiments/consolidation/RANK_COMPRESSION.md)、[布局迁移](experiments/consolidation/LAYOUT_ROBUSTNESS.md) |
| 内部状态候选与失败 | [递归全局状态](experiments/recurrent_state/RESULTS.md)、[分槽状态](experiments/recurrent_state/SLOTTED_CONTROL.md)、[8192 次热写衰减](experiments/recurrent_state/SKEW_RETENTION.md)、[写门控与碰撞](experiments/recurrent_state/SPARSE_MATCHED_TRAINING.md)、[随机连续键与拒答回退](experiments/recurrent_state/CONTINUOUS_KEYS.md)、[离线字典回退回放](experiments/recurrent_state/FALLBACK_REPLAY.md) |

运行完整实现回归：

```bash
python -m unittest discover -s tests -v
```

测试验证实现契约，不代表模型能力。全仓库测试日志与后续定向测试分开保存；不把不同提交的测试数量当作同一次运行结果。大多数机制实验可很小，但语言能力、规模迁移和长期交互仍需要更高层次验证。

## 语言 baseline 与历史训练

Phase 1 A1/A2 是单路径、自回归 teacher forcing、sampled-negative InfoNCE 训练，使用自训 BPE 16k 与 zhwiki 语料。固定 held-out 评估显示低频/稀有词薄弱，长前缀利用尚未证明。因此当前 checkpoint 的语言质量不能支撑通用对话或电影理解验收。

原实验指标和混杂说明保留在上述 Phase 1 报告，不用后续小任务成绩替代。历史训练入口：

```bash
python -m training.train_v21 --config configs/formal_a1.json
python -m training.train_v21 --config configs/formal_a2.json
```

仓库不含训练语料二进制；准备数据与配置前先阅读[历史实验计划](docs/EXPERIMENT_PLAN.md)和[数据/产物策略](docs/REPO_LAYOUT.md)。**训练 baseline 不是启动持续交互入口所必需的步骤。** 不要覆盖既有 tokenizer 或历史结果来复现实验；新运行使用独立输出目录。

## 仓库导航

```text
models/       原 PBSLM 与独立结构候选
runtime/      同步会话、异步持续运行、终端入口与状态策略
training/     历史与现有训练入口
evaluation/   受控实验、回放、评分与来源记录
experiments/  配置、原始结果、日志和逐实验限制
docs/         架构、阶段报告、研究设计与验收边界
tests/        模型隔离、状态与运行协议回归
tokenizer/    自训 BPE tokenizer
```

不上传 `*.pt` 权重、语料 `.bin` 和缓存；公开配置、源码、结果、权重哈希与环境记录。部分运行只保存了汇总，部分有逐事件预测；可复核范围以对应报告为准。历史文档可能描述旧优先级或未来设计，当前主线以[Online 目标澄清](docs/CONTINUOUS_ONLINE_PRIORITY.md)为准。

## 下一阶段

当前最重要的工作不是继续在已知 8 ID 上追满分，而是：

1. 动态/未知身份的可靠寻址、选择性写入、有效性与撤销，明确容量不足时的拒答或回退。
2. 将工作状态候选接入主模型与持续运行器，避免用规则状态机代替神经理解。
3. 使用有基本语言能力的模型和独立连续事件集，分别评价状态正确、语义续连、误/漏打断与表达进展。
4. 实际计量状态优先方案的维护、检索、延迟与总资源成本，验证是否真的省计算。
5. 可靠经验经可审计、版本化的慢更新进入 A，并检验旧能力与错误撤销。

**项目仍处于研究阶段。** 没有实时电影理解、通用语言持续学习或架构优越性的结论。正结果决定哪些假设值得继续，负结果决定哪里必须改；若简单检索或状态方案更可靠，就保留简单方案。
