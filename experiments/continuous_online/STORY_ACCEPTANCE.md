# 连续短故事开发验收契约（尚无模型成绩）

数据：evaluation/fixtures/online_stories_dev.jsonl，单个JSON对象、4case×4event（非逐行事件）；格式pbslm.online_story_cases.v1。16个事件覆盖计划变更、物品转交、同源广播纠错、条件尚未发生与行动已发生。它是公开、人工编写的development fixtures，不是独立held-out benchmark。

## 标签与模型输入隔离

online_story_contract.input_view(case,index)只返回case_id、question、截至当前id/text事件。不传target_after/relation/interrupt/reason，不传未来事件。此函数是未来backend接线边界，不构成任何外部后端实际没读标签的认证。

每事件参考target_after与interrupt由研究者按封闭故事假设给出。例如未证实传闻不覆盖直接交接事实；真实人类对话中可能需要澄清/不确定表达，不能当成普遍语用标准。

## 独立预测格式

```json
[
  {"case_id":"plan-change","event_id":"p1","answer":"车站","interrupt":true}
]
```

只为说明格式，不是模型输出。运行：

```bash
python -m evaluation.online_story_contract --predictions external_predictions.json --output new_score.json
```

评分包含提交覆盖率、全部事件exact answer accuracy、漏必要打断、误打断；missing预测按错误/漏必要打断处理。相同key重复、未知event、非bool打断标记明确拒绝。

当前answer严格匹配标签，不做语义等价判断；它不是通用语言judge。调度时序、表达进展、自我修正与已说内容关系需要额外runtime trace评分，本接口不声称已覆盖。

## 已执行验证

3项scorer/input-contract单测与6项现有routing/background回归，共9项通过，story_contract_regression.log：

- backend view不含标签/未来；
- 合成oracle预测只验证评分器一致性，不作为模型100%成绩；
- 空预测覆盖0/准确0，漏打断正常计数；
- 重复key报错。

无模型推理、无下载、无训练、无新的神经能力结果。当前compact PBSLM fact模型不具备此词表/任务能力，不能把规则脚本包装成故事理解。

## 下一阶段

- 收集更广人工事件集并独立复核标签，把dev/validation/test分离，避免边看成绩边改题。
- 用足够可用的固定语言backend产生独立预测，保留输入、原始输出、token/time trace、模型hash与decode配置。
- 同流比较hard-all、caller依赖、学习型策略；精度/覆盖、漏打断/误打断、输出进展和时延分开。
- 如果现有PBSLM语言质量不足，应明确单独预算训练或语言backend，不扩大任务后再假装输入接线等于理解。
