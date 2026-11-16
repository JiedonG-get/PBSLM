# 持续输入状态与说话任务分离

主线是Online运行，不是Online Learning。本批没有训练、参数更新或数值M写入。

## 新的运行结构

ContinuousRunner.observe可调用backend.submit_input(revision,text)，该同步hook仅入有界队列，不能推理/等待。队列满则observe在修改revision/history前拒绝，调用方可重试。

runtime/continuous_state.py独立worker逐条处理已接受输入，即使should_respond=False也继续处理。每条输入产生按revision保存的状态快照。说话任务等待对应输入处理完成，但通过shield等待，取消说话不会取消输入future。

两种提取器：

- ExactReferenceExtractor：已知句法的确定性基线；
- LearnedReaderExtractor：此前已保存的固定GRU（不是PBSLM主干），不可靠性已在reader报告披露。

状态是最新已接受entity/color断言的符号记录。没有真值校验，不写PBSLM M；输出文字由明确规则renderer组织，不是神经自然语言生成。

## 场景验证

```bash
python -m evaluation.continuous_semantic_scenario
```

semantic_state_scenario.json保存参考parser和learned reader两条运行：

1. 静默接收甲红并处理，不输出。
2. 查询甲颜色，生成intro后新纠正甲绿到达。
3. 旧红色断言不提交，已提交prefix保留，新输出使用绿。
4. 静默接收乙蓝，输入状态也更新。
5. learned reader权重hash不变，未训练。

该场景用显式输出delay创造可观察窗口，调度错过窗口则断言失败，不靠最终拼字符串伪造取消。它是一个seen形式的规则/reader运行场景，不是连续电影理解、PBSLM语言续接或held提取安全证明。

## 生命周期与边界验证

独立审查发现并修复：

- processed结果曾直接暴露历史dict：现返回copy；journal记录deepcopy。
- extractor失败结果曾不参与快照淘汰：现成功/失败统一限制completed states/futures。
- aclose取消后不能重试：现共享shield drain任务，重试等待同一关闭过程，executor在finally清理。
- worker终止后处理待决future为明确error、排空queue并拒绝新输入，避免永久等待。

单测6项覆盖静默处理、取消输出不取消理解、队列背压不改history、state copy/失败快照上限、关闭取消后重新等待、未知/不确定输入不更新exact基线state。

## 关键限制

- 没有验证一般语义。state与renderer是可解释基线，reader是独立小分类器。
- 没有模型自主决定开口，should_respond由调用方给出；rule abstain仅处理无单一已知实体。
- 词汇/实体/关系很窄；learned reader可能把提问/否定误当断言，不能自动用于可信记忆。
- 输入queue/state有界，但journal与committed输出在现限制内累积；不是无限时长系统总内存保证。
- native线程不能被硬抢占；挂死extractor会拖延drain/shutdown，无硬实时承诺。
- backend状态没有持久snapshot协议；此前M会话snapshot与这条持续state尚未统一。
- 规则renderer利用已提交prefix长度生成续接intro，不是真正语义级去重/自我修正。

下一步应先用确定的连续语义评估分离接收时序与状态正确性，再决定让PBSLM承担哪些持续状态任务；不能仅凭有状态字典就声称持续神经理解。
