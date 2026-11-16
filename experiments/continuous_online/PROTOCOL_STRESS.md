# 连续运行压力测试与输出饥饿限制

```bash
python -m evaluation.continuous_protocol_stress
```

固定脚本协作backend，3seed×1000操作，随机交错输入/只听/interrupt/resume；每操作只让事件loop推进一轮，结束后投递最终回答请求。无神经模型、训练、记忆写入或速度benchmark。

结果protocol_stress1000.json：

- 3seed均0 stale commit，close后0生成任务、0pending。
- 最近输入history最多32；unfinished generation最多2（配置上限3）。
- 每seed最后revision1001完成；分别710/714/698次generation启动，绝大部分被取消。
- 每seed只提交最终3个单位，前面持续变化阶段无输出。

## 关键负结果

安全fence和资源边界通过，不等于流畅性通过。当前策略每个新输入都使当前输出generation过期；生产者足够快时，系统可以无限重规划而不说话。latest-only自动defer避免丢回答意图，但不解决无限新revision下的输出饥饿。

因此不能依据“最终请求完成”宣称无限持续输入下系统有进展，也不能把此结果称为硬实时或持续理解。

## 后续方向

- 将观察更新与必须撤销当前表达的事件分开；只有明确影响当前回答的更新才硬中断。
- 需要可验证relevance/urgency策略，不可简单忽略新事实保证输出量。
- 研究并行持续观察状态与表达快照的版本关系，防止软更新后输出过时断言。
- 真实媒体场景须指定最小响应进度、最大过时窗口和可撤销边界；当前尚无这些语义保证。

已有符号worker处理每条已接受输入，但真实PBSLM接线尚没有持续语义状态，也没有判断输入是否影响当前表达的学习型policy。先定义基准，再决定策略，不能随意加debounce掩盖问题。
