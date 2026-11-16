# 持续输入过载：有界接收与明确重试

## 固定场景

```bash
python -m evaluation.continuous_overload
python -m unittest tests.test_continuous_overload tests.test_continuous_state tests.test_inference_worker_scheduling -v
```

本批不训练、不调用神经模型。threading.Event固定阻塞exact parser，验证接收契约而非吞吐/延迟。

1. 输入1已进入worker，固定阻塞不完成。
2. 再连续投递6条事实/纠正，queue_limit=4。
3. 前4条入队，后2条明确QueueFull；被拒绝时runner revision/history与backend future数量均未改变。
4. 调用interrupt但不取消输入worker，释放gate，排空已接受5条。
5. producer按原顺序重试2条，最终7条按接收revision顺序处理。
6. 最终甲紫、乙白；全部输入should_respond=False，无输出。

原始admission列表、rejection、retry、处理顺序、最终state与双层journal：overload_admission.json。

## 已验证边界

- queue仅保留4条等待项，active worker另占1条；不是总系统内存上限。
- accepted输入在该正常恢复场景不静默丢失；reject明确且不修改接收状态。
- 打断说话不取消输入处理；producer负责重试未接受的输入。
- 定向10项测试（overload1、state7、worker2）通过；日志overload_regression.log。没有为本次小改动重复全套训练/实验。

## 尚未解决

- 不能声称无限持续吞吐，native挂死仍可能拖住关闭。
- 重试后revision反映接收顺序，不是事件发生的真实时间；晚到旧事实覆盖新事实的问题尚未制定策略。
- 视频/音频不能简单无限重试；需要单独定义采样、合并、丢帧和时钟同步语义。
- 输出任务defer与输入QueueFull是不同背压：前者输入已接受但暂不说，后者输入未接受。
- 未模拟网络断线、进程崩溃、传感器时间或模型语义错误。

这项证据补的是持续运行的过载可控性，不是语言理解能力。下一步应明确输入源时间与迟到事件策略，并检验输出不会引用已经过时的世界状态。
