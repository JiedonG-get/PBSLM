# PBSLM 阶段研究报告：稳定参照、持续适应、记忆与运行协议

本报告汇总用户授权继续研究后完成的可复核工作。目标是把原来停留在审查与设计中的边界推进到实验，不保证所有架构假设成立。**本阶段完成了工程验证、最小持续学习实验和同步Runtime原型；完整自然语言Online系统尚未完成。**

## 核心结论

1. 在共享参数冻结、小型fp32 CPU配置下，masked A/C 的逐样本路由和计算状态有数值隔离证据；共享权重、共享只读block表征仍存在，不能称参数独立。
2. A-only参数更新能在合成反馈流上学习新映射，但A会遗忘旧映射；C不变不能替代A保持性。
3. 小任务回放改善训练过旧映射保持；控制batch大小和更新次数后仍观察到差异，但没有严格等FLOPs和objective-weight单因素对照。
4. 经离线随机映射任务训练，M能在不更新评估参数时回忆新映射；错值干预会让输出跟随错值，证明具体关联依赖，也证明污染风险。
5. 持续写入时矩阵M出现冲突与干扰；衰减缓解但不解决纠错。外部ledger重建改善结果，代价是额外存储和大量写入，不能算M的内生能力。
6. 真实模型已接通同步observe/predict/commit/interrupt/feedback/confirm-M/snapshot/restore路径，commit与记忆确认分离。它不是实时并发系统，也没有现场参数训练。
7. 同初始化码本短程配对支持frozen的早期趋势，不能取代完整15k步多seed正式对照。

## 证据索引

| 研究项 | 代码/数据 | 结果与边界 |
|---|---|---|
| O1.5 reference/NaN/梯度 | tests/test_masked_e2e_evidence.py、experiments/o1_5/gpt_verification | experiments/o1_5/GPT_VERIFICATION_REPORT.md |
| 本地checkpoint | evaluation/check_checkpoint_compat.py、experiments/o1_5/checkpoint_evidence.json | direct_ce迁移后可加载；nomem旧架构缺ctx不能strict加载 |
| A持续适应 | evaluation/continual_associations.py、experiments/continual_associations | RESULTS.md、BUDGET_CONTROL.md |
| Episodic M | evaluation/episodic_memory.py、experiments/episodic_memory | RESULTS.md：正常/零/错值干预 |
| Persistent M | evaluation/persistent_memory.py、experiments/persistent_memory | RESULTS.md：冲突、年龄、纠正、外部ledger成本 |
| 自然语言码本 | evaluation/run_codebook_pair.py、experiments/phase1_2/pilot1500 | RESULTS.md：1seed/1500steps/80k held |
| Runtime协议与模型接线 | runtime/session.py、runtime/model_session.py、experiments/runtime_integration | RESULTS.md：同步事务与恢复，不是并发 |

所有新实验独立目录；原Phase1 logs/metrics/checkpoints不覆盖；新checkpoint与大语料继续本地忽略，上传hash/配置/指标。

## 主要实验结果（描述统计）

### A参数学习与旧映射保持

三seed、每seed基础300步+128在线事件；真实masked PBSLMv2、d32、固定码本、合成词表48。每次先预测，再接收反馈，最后更新。新domain与旧domain由输入prefix区分。

| arm | 在线更新前正确率 | 最终新映射A | 原domain同key旧映射A |
|---|---:|---:|---:|
| 不更新 | 3.65% | 4.17% | 100% |
| A-only | 75.00% | 100% | 54.17% |
| 当前样本重复3次 | 75.00% | 100% | 54.17% |
| 当前+2旧样本回放 | 73.96% | 95.83% | 100% |

旧映射就是基础训练过/回放过的16条，不能称held-out旧任务泛化。C同输入bitwise稳定、受保护state hash不变均通过。回放降低当前样本在mean loss中的权重，未完全分离内容与压力效应。

### M回忆与污染

三seed，各500个新随机mapping离线episode；eval全新mapping，不更新参数，仅显式写M。

| 每episode关联数 | 正常M回忆 | 零M回忆 | 错值M真实答案 | 错值M跟随写入错值 |
|---|---:|---:|---:|---:|
| 4 | 100% | 10.42% | 0.13% | 99.87% |
| 8 | 99.87% | 13.09% | 0.07% | 99.67% |

查询key先前已在support出现并公开答案，是关联回忆任务；不能证明未反馈事实推理。M每episode清零，熟悉key/label词表。

### 持续M的局限

三个seed，96新事实+6个纠正=102事件，反复修改16个key的最新关联，参数固定。

| arm | 最终最新truth回忆率 | 纠正后立即读对 | write调用数/seed均值 |
|---|---:|---:|---:|
| append | 29.17% | 38.89% | 102 |
| decay append | 52.08% | 38.89% | 102 |
| external ledger8 rebuild | 45.83% | 100% | 773 |
| external ledger16 rebuild | 83.33% | 94.44% | 1362 |

这是本阶段的重要负结果：短期回忆有效，不等于长期持久学习成功。最新反馈ledger本身存储了答案，重建组不是同成本神经记忆优势证据。

### 同初始化码本pilot

相同初始化state SHA256、数据/负采样seed、1500steps。总参数相同，可训练差额为码本处理本身。固定80k-target评估：trainable Top1 5.36%、frozen 7.44%；low/rare Top1仍0。只1seed，不能宣布最终固定码本优越。

## 工程验证结论

- 独立紧凑M1对照不是简单B切片：按物理block ID对齐不同union的行与attention双轴。
- M1 final z最大绝对误差约A1.19e-7/C1.79e-7；约定fp32 atol1e-6/rtol1e-5通过。
- 非均匀A/C probe分别autograd，避免LayerNorm输出求和的退化测试。共享tok/bank/comm/flow两侧梯度存在；E读分支仅A有梯度。
- kA=1下hard topk不可导、singleton softmax不提供score学习信号；不能假定router已经有效学习。
- checkpoint证据区分原始strict与out_c迁移strict；本地文件不冒充GitHub artifact，mtime不冒充生成时间。
- Runtime指纹包含权重与推理配置；M写入失败本地恢复，幂等确认/冲突lr拒绝。无生产级崩溃一致性/鉴权/并发。
- 最新全仓库90 tests OK，CPU4.299秒，原始日志experiments/persistent_memory/all_tests.log。测试通过仅限实现契约，不是语言质量/在线学习验收。

## 计算预算与可复核性

硬件RTX4060 Laptop 8GB，Python3.13.5、torch2.11.0+cu128。主要学习实验为分钟级小模型；码本配对训练约129秒、held消融评估约212秒。各JSON保留具体耗时/seed/config，后续实验加source hashes、Git dirty状态、CUDA环境；较早pilot不能追溯补造当时provenance。

没有云端算力、外部服务或模型发布；文件正常Git提交/推送，摆脱早期API代理造成的双SHA历史，旧local史保留archive branch。

## 对最终目标的回答

**模型能否把稳定知识基础、学习状态、外部记忆分离并协作？**

目前在小型受控任务上获得部分支持：C稳定时A可以适应；训练过的读机制可以使用新M关联；同步协议可以控制候选、提交与反馈写入。然而A遗忘和M冲突明确存在，系统没有解决长期保持、可信纠错或真正实时运行。

**PBSLM是否比普通模型更好？**

当前没有足够证据。没有同预算Transformer/单路径/字典检索baseline，不能从合成满分推导架构优势。

**能否一边看电影、一边理解、一边参与交流？**

尚不能。没有视觉/音频输入链、多模态训练、实时延迟指标、并发取消或长期学习验证。本阶段建立的是可以继续实验的基础，不是成品。

## 后续优先级（需独立实验）

1. 持续记忆覆盖/纠错：有来源的记录、可撤销贡献、容量预算；与直接字典/attention检索同预算比较。
2. 自然语言学习与保持：固定历史held集、时间切分新信息、predict-before-feedback、多次任务切换；A泛化与C稳定分别测量。
3. 同初始化码本完整预算多seed、A/C/InfoFlow/E/M消融；先保证低频和上下文质量，再谈架构优势。
4. 真实Runtime：token级输出、置信度校准/静默policy、并发输入与取消、持久化/恢复及多用户隔离。
5. 更长流、更多seed、统计不确定性和严格计算/存储预算。

研究假设可能被否定；负结果同样应保留。不能把本阶段的小任务支持提升为通用Online能力声明。
