# 第三阶段：持续 Online 运行优先的原型边界

用户澄清了原始目标：首先是持续运行、打断续连、持续输入并选择性输出，不是首先Online Learning。前两阶段学习/记忆证据保留，但本阶段固定权重，以运行闭环为主线。

## 已实现并验证

1. 持续输入revision与输出提交fence：旧generation即使忽略cancel也不能提交新revision后的旧单位。
2. 输出进行中新input可接收；已提交prefix保留，新request收到latest events与prefix；resume不捏造重复输入。
3. should_respond=False持续接收；独立ingest worker让静默输入也处理，取消说话不取消已接受输入。
4. 专属串行推理executor：取消future不导致模型并行调用；native计算不抢占，后续请求可能等待旧计算。
5. 真实PBSLM单token和有界逐tokenrollout，模型/M指纹不变，无过期token提交。当前小模型不具备流畅生成能力。
6. 混合上下文传输：历史输入、渲染prefix和最新输入token按预算实际进入模型step0；无语义使用证据。
7. 生成器显式清理、close超时重试、任务上限与明确defer、输入queue背压、worker故障settle、失败future上限、状态copy。
8. reference与固定GRU的64事件×3seed连续符号状态任务，分开记录state/提取/时序/取消指标。
9. 终端入口支持边收输入边显示输出、listen/interrupt/resume/quit，便于体验机制；后端能力明确标注。

## 证据索引

- experiments/continuous_online/RESULTS.md：async loop、真实固定模型时序、生命周期审计。
- tests/test_inference_worker_scheduling.py：thread Event屏障，不靠重跑计时挑重叠。
- experiments/continuous_online/neural_token_rollout.json：真实10步神经输出与两次输入打断。
- experiments/continuous_online/CONTEXT_TRANSPORT.md：actual step0 token与预算、角色布局未训练限制。
- experiments/continuous_online/CONTINUOUS_STATE.md：静默处理与说话任务分离，reference/reader场景。
- experiments/continuous_online/STATE_BENCHMARK.md：基础/两种hard表面多seed，state与调度单独打分。
- runtime/ONLINE_CLI.md：不依赖checkpoint的reference入口和本地真实模型入口。

## 真正完成的是什么

一个可运行、可取消、版本隔离、有独立输入处理、有真实小模型推理接线的文本事件原型。它接近的是用户所说的Online运行机制，而不是证明语言模型持续理解已经解决。

## 尚未完成的能力

- **语义续连**：模型需要真正根据已说/新观察重组表达；当前prefix接线与规则intro不能证明。
- **自主说/听**：当前caller显式should_respond或CLI命令决定，未有学习型时机policy。
- **自然语言与世界状态**：小颜色模型/reader/符号state不等于开放理解，reader已有误接受证据。
- **持续感知**：无视频/音频/电影流、同步时钟、视觉记忆或TTS。
- **硬实时与资源**：native推理不可抢占，journal/output历史非总有界，无端到端SLO或高频overload验证。
- **生产可靠性**：未统一持久snapshot、跨进程/崩溃恢复、线程安全服务或多用户。
- **模型训练质量**：现有语言baseline弱，rollout质量不能做自然对话验收。

## 下一阶段方向

先定义小而有辨别力的连续语义任务/验收集，使用足够可用的固定语言后端，对照普通生成loop，再判断PBSLM状态机制的必要性。需要模型与训练投入时应单独预算，不能把协议层成功包装成神经架构优势。

本阶段没有下载新模型或扩大训练。当前全仓库156 tests OK（8.219秒CPU；experiments/continuous_online/all_tests_online_cli.log）。系统部分提供了可复核基础，但距离“看电影时实时理解并自然交流”仍很远；后续工作应沿这条主线，而非回到用小型在线学习分数替代目标。
