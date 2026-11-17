# 状态优先的保守拒答基线：碰撞风险与覆盖率

运行：`python -m evaluation.collision_guard_control`。只分析既有sparse_train600的all-hot最终预测，无训练、无模型调用、无历史查询。策略固定：某entity的argmax slot不是唯一，则该entity所有query标记defer；不根据答案或test错误选阈值。

## Pooled结果（每arm3072个query，含hot和cold）

| 模型 | 状态直接回答覆盖 | 已回答准确率 | 待回退/拒答 | 找出的原错误比例 |
|---|---:|---:|---:|---:|
| soft | 91.67% | 53.62% | 8.33% | 10.55% |
| sparse | 41.67% | 100% | 58.33% | 100% |
| sparse+collision | 75% | 100% | 25% | 100% |

结果是对3seed×8hot×16流×8entity最终probe的pooled计数，不是冷entity-only，也不是uniform测试。某冲突entity即使另一冲突entity本次不热，也会保守defer，所以与此前‘当前hot冲突子集’分母不同。

## 含义

- 稀疏写模型的已观测错误可由已知ID静态碰撞标记筛出，但覆盖损失很大；不能只展示接受项100%。
- 碰撞先验改善coverage，不过还有25%请求需要外部处理；没有证明这些请求能被实际历史正确回答。
- soft的旁路干扰在非碰撞entity中也发生，这种结构标记不足以筛出风险。
- 该policy仅查8ID route元数据，非学习型置信度/校准/动态身份判断，也不验证事实真实性。

## 证据与边界

collision_guard.json保留source SHA、每seed每arm完整计数与pooled值。答案仅用于事后计分，选择只看top_slots；三项测试验证决定不随prediction/target变化、无碰撞也可能全错、不允许batch不匹配被zip静默截断。另复跑3项sparse模型契约，共6项通过，collision_guard_tests.log。运行后新增batch形状guard不改变当前结果。

已回答项在有限test无错误不代表部署零风险；多个probe共享训练模型，不是独立样本。没有均匀流逐query日志可供同样分析，未做未知query、噪声、路由漂移或跨任务检验。

本批没有实际fallback/历史存储，没有使用target当‘查历史后的答案’，也不报告虚构fallback后100%。没有推理延迟/算力或总内存节省证据。后续必须真的构建并计费fallback，再在独立验证集选择更完整的触发策略。

当前目标仍未完成：内部状态优先已有小符号机制证据，动态身份/文本理解/表达语义续连与可靠回退尚未连接。
