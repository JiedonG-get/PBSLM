# Runtime 协议原型：提交、打断、恢复与记忆候选

这是**模型无关、同步、单进程的状态机原型**。没有模型推理、线程/异步流、网络、优化器或数值M。它只验证事件契约，不是Online Learning实现。

## 可运行范围

```python
from runtime import RuntimeSession
s = RuntimeSession('demo')
s.observe('用户问题', event_time=1)
s.advance()
p = s.propose('第一句。第二句。')
s.commit(p, expected_offset=0, count=4)  # 字符粒度，不是模型token
s.interrupt('新信息')
s.observe('补充条件', event_time=2)
s.resume()
s.advance()
# 旧候选剩余部分不能继续提交；必须重新组织输出
restored = RuntimeSession.restore(s.snapshot())
```

- observe追加输入并增加revision；失效未提交候选。
- commit使用proposal_id、expected_offset、count生成幂等回执；完全相同的重试返回旧回执，不重复输出。
- interrupt丢弃未提交候选、拒绝其未确认记忆候选；已提交字符和已确认日志保留。
- resume回到observing，必须advance再propose。
- propose_write只产生JSON候选；confirm_write需要显式来源与可用时间，commit不等于确认正确。
- observed_input可以作为候选value；模型预测/已提交输出不能作为自身正确性的确认来源。
- external_verified只是调用方声称已外部验证，本原型不能鉴权或验证事实。
- snapshot/restore是本地协议状态恢复，不回放外部副作用，不恢复模型cache/RNG/权重/数值M。

## 验证

```bash
python -m unittest tests.test_runtime_session -v
```

15项测试通过，CPU 0.003秒（本次Lead复跑）。原始日志：`experiments/o1_5/gpt_verification/runtime_protocol.log`。
覆盖部分提交打断、输入revision、幂等提交、非法转移、未来信息、显式确认/拒绝、JSON恢复与损坏快照。

## 尚未实现

- 模型adapter、token级生成、置信度校准/静默policy；
- 实时流、取消正在执行的GPU计算、线程安全/锁、进程崩溃持久化；
- 数值M的事务应用、版本rollback与冲突补偿；
- 输入缓存容量限制、长会话压缩、候选TTL；
- 线上参数更新/经验回放/模型snapshot；
- 多用户隔离、外部确认鉴权。

下一步接线时必须保持predict-before-feedback和commit/confirm分离，不把状态机可运行作为学习成功的证据。
