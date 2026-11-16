# 源序号参考策略：晚到事件不倒退同源状态

本批无训练/神经推理，无M写入。source_order模块是独立参考策略，尚未接到ContinuousRunner文本API，更没有真实video/audio时钟。

```bash
python -m evaluation.source_order_demo
python -m unittest tests.test_source_order tests.test_continuous_overload -v
```

## 语义

- key=(source,slot)，sequence必须非负整数。
- 更高sequence应用；较低sequence记录late_ignored；同sequence同值为duplicate_sequence，不更新。
- 同event_id同payload幂等；同ID异payload、同source/slot/sequence异值明确冲突。
- source之间sequence不比较；不同source对同slot值不同，返回cross_source_conflict和无resolved_value。
- 一致只代表来源声称相同，绝不等于真值验证。
- state和event history都有正容量；满时明确拒绝，不能偷偷淘汰水位后又接受旧序号。

## 结果

30个同源更新，seed0/1/2分别打乱到达顺序，每次更新后水位均等于至今最高源序号；最终都为29。忽略旧事件分别25/28/25，原始所有到达/回执/state在source_order.json。

5项定向测试通过（source_order4+overload1），覆盖迟到、幂等、深拷贝、冲突、来源不可比较和限额不丢水位。source_order_regression.log保留输出。

## 不解决什么

- 来源/序号由调用方声明，无鉴权、防重放持久层或时钟同步。
- event_id按当前实例全局唯一；来源命名碰撞由调用方负责。
- 容量耗尽需外部checkpoint/rotation设计，当前模块没有snapshot迁移API；不能无损无限运行。
- 序号重置/摄像头重启需新source epoch，当前未实现epoch协商。
- 同slot“更晚”不一定“更可信”；未知/撤销/条件事实、多来源裁决尚未定义。
- 与实际输入queue、rule belief backend、神经状态尚未统一；这是明确接口与测试基线，不是模型连续理解。

下一阶段应把source epoch与有界历史持久化规范接入接收API，避免迟到旧事实和恢复后丢水位；不能只增加一个时间字段便称时序已解决。
