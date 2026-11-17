# 学习型快速状态的真实异步接线

本批固定content_fast600已保存delta权重；无训练、无外部键值bank/双基参考，不修改原PBSLM主模型。首次把这个连续键状态候选接入已有ContinuousRunner的独立输入worker。

```bash
python -m evaluation.fast_state_online_demo
python -m unittest tests.test_fast_state_online -v
```

## 输入与模型实际看到的内容

runtime/fast_state_online.py接收严格JSON：

- write：op/key/value，key为有限非零16维向量，value整数0..7；
- query：op/key，不允许target或额外字段。

模型输入为归一化当前key、write时观察value、128float快速状态。模型不读runner历史/journal，不接收query目标、实体index、seen mask或truth。状态是一个16×8tensor，由单线程串行worker拥有。query只读state。

submit同步验证格式后才入有界队列，失败不改runner revision；已接受事件由worker处理，取消说话不取消这些写入/查询。写入先计算candidate并检查有限性，成功后替换state；数值故障不替换矩阵。

每revision缓存预测/概率/state hash结果，而不是所有历史state快照，默认保留16条结果；旧revision显式不可访问。输入queue上限16，active worker另占一条。模型h不包含原始事件；**runner已有历史和audit journal仍保存输入文本，且不保证总日志内存有界**，不得混淆无模型history访问与整个系统不存日志。

## 可复核场景

每seed重新随机生成8个key/value，静默prime。查询key0后先提交固定intro并停在Event屏障；随后静默纠正key0为原值+1mod8，等待worker处理；再query同键、释放屏障，输出最新结果。

- 三seed均通过：旧query的JSON答案未提交，已提交intro保留，new query hash等于纠正后的state hash。
- 原始模型state_dict hash前后不变；state activations预期改变。
- 每个JSON答案来自真实冻结模型query，不用truth填值。
- 纠正前seed0/1/2分别正确/误UNKNOWN/正确；纠正后本次3个query均正确。
- 额外各一个冷key、一个独立unknown query均正确；样本极少，不用来宣称泛化准确率。

完整input与output journal、state hash、预测/target及服务耗时在content_fast600/online_adapter.json。模型accuracy不是协议pass条件，seed1错误完整保留。

intro和JSON是协议renderer，不是自然语言生成；屏障人为创造可重复打断窗口。这里没有正在运行GPU kernel抢占、长语言语义续连、自主说听决策或影片理解。输入顺序来自测试脚本，不能外推真实传感器延迟。

## 复用生命周期与故障测试

后台输入沿用continuous_state的队列、shield processed future、worker终止settle、取消关闭后可重复等待drain、executor finally清理。native推理若挂死仍不能硬抢占；没有硬实时/崩溃持久化。

新增4项adapter契约测试：

1. 静默写入/query取消/纠正/新query版本一致。
2. 非法事件不消费revision，completed结果数有界。
3. 写入故障保持旧矩阵，worker可继续query。
4. target额外字段、NaN、零向量拒绝。

全仓库230 tests OK（9.284秒CPU），content_fast600/online_adapter_all_tests.log。文本只读设计审查没有读取或执行实现，不冒称独立代码/预测审计。

## 仍未完成

- 尚未接原PBSLM E编码、A/C、M→A或模型权重版本切换；不能因为复用了runtime便称主模型整合完成。
- key/value由caller明确提供，无自然文本实体/事实抽取、来源鉴真、冲突/撤销语义。
- 模型相似键/unknown错误已在前批实测，此adapter不修复它们。
- should_respond由caller给出；write默认只处理，query才有结果，模型未自主选择发言时机。
- 无fallback、无总算力或延迟优势；记录observe服务时间仅是本次协程入口到返回，不是外部到达时延。
- 仅CPU、单stream、事件顺序处理、边界API，没有生产服务或统一snapshot。

下一步应让原PBSLM E把实际文本映射为这些key，但必须明确如何对齐/压缩维度及query/record角色，并保存新旧模型对照，不能把随机key模型直接宣称语言记忆有效。
