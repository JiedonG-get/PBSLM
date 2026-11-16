# 持续Online评估：将时序、提取与状态分开评分

本批固定reader权重、无训练/无数值M。运行框架ContinuousRunner+独立ingest worker，状态是符号latest accepted assertion，输出为规则renderer；不是PBSLM神经语义续接。

## 流与对照

```bash
python -m evaluation.continuous_state_benchmark
python -m evaluation.continuous_state_benchmark --hard-negatives --output experiments/continuous_online/state_benchmark64_hard.json
```

每seed64事实/纠正/否定/不确定输入交错；8已知实体、8颜色；事实truth只用于评分，不传extractor。每8事件插入已知实体问题，等intro提交后投递下一输入，创造输出中接收新信息的机会。其余多数输入显式should_respond=False，仍必须处理。

对照：精确已知句法parser（baseline）与固定GRU reader（此前已训练，非PBSLM主干）。两者使用相同输入/renderer/状态规则。

基础负例为“不是…”和“不能确认…”。hard变体仅替换为“别把…当成…”与“传闻…但尚未证实”，不是完整否定/时态benchmark。三seed输入实体颜色顺序随机，精确模板固定。

## 实际结果

基础流和hard变体均：

- 每seed64输入，其中32负例；两种extractor均0个负例被接受。
- mean_known_fact_accuracy、exact_state_match_rate均100%。
- 每seed7个输出中输入机会，旧规划断言后续提交0；stale revision提交0。
- reader参数hash未变。

学习型基础流各seedP95输入处理时间约2.25、1.65、1.84毫秒；hard流约2.20、1.78、2.17毫秒。这里是observe协程入口到processed返回，本机CPU单次运行，不含传感器/网络到达延迟，不承诺硬实时。

原始逐事件truth/actual state/extraction、Runtime与ingest journal在state_benchmark64.json和state_benchmark64_hard.json。

## 结果不能扩大

- 这里的负例都正确拒绝，不代表reader跨分布安全；此前独立模板校准/测试明确出现误接受，仍然有效。
- 各事件等待处理完成再下一事件；这验证输出与输入交叠、静默处理与状态一致，不是高频队列/持续过载压力测试。
- .02秒输出delay是显式renderer延迟，让中断窗口可观察，不是神经生成延迟。
- should_respond由caller脚本决定，没有学习自主开口/静默。
- rule renderer使用提取状态，不是PBSLM回答；精确parser知道句法，不证明一般理解。
- 状态管理记录最新接受陈述，未鉴真，也没有错误确认回滚/多来源冲突。
- truth和状态是小型实体颜色映射，非真实电影/语音/多模态事件。

## 故障边界

新增worker-cancel测试：已接受的当前和排队future都受控settle，后续输入拒绝，关闭报告worker取消并清理executor。结合已有close取消后重试、失败snapshot限额和state copy测试。

调度正确与语义正确必须独立验收。下一阶段真正需要的是丰富连续语义任务和可用语言生成后端，而不是把这个符号状态机的满分称为PBSLM持续理解成功。
