# 状态优先＋实际观察记录回退：离线反事实回放

本批不训练、不调用神经模型。使用sparse_train600已保存的最终神经预测，并按原all_hot随机生成器重建真实观察写入，构建容量4/8的最新值update-recency字典。不是live集成，也不将fallback结果重新写回神经h。

```bash
python -m evaluation.working_state_fallback_replay
python -m unittest tests.test_fallback_replay tests.test_collision_guard -v
```

回退触发仅看known-ID top-slot是否碰撞，不看target/答案正确性。字典答案只来自观察流；保存target只用于验证重建序列一致和计分。每seed/容量/arm每case先prime8，再2048次热写。字典对所有输入更新，query只在guard触发时访问。

## Pooled最终探针（每arm/容量3072query，含热冷）

| 模型/字典容量 | 全query正确率 | 回答覆盖 | 回退查询比例 | 回退未命中占全部query |
|---|---:|---:|---:|---:|
| soft/4 | 53.84% | 96.35% | 8.33% | 3.65% |
| soft/8 | 57.49% | 100% | 8.33% | 0% |
| sparse/4 | 70.31% | 70.31% | 58.33% | 29.69% |
| sparse/8 | 100% | 100% | 58.33% | 0% |
| sparse+collision/4 | 83.33% | 83.33% | 25% | 16.67% |
| sparse+collision/8 | 100% | 100% | 25% | 0% |

只有容量8完全保存已知8个entity的最新值，字典本身即可解任务；这里的100%不是神经模型泛化胜利。容量4丢弃旧entity，miss明确None/拒答，计入全部分母，不用评分truth补答案。soft非碰撞错误没被guard触发，因此即使字典容量8也仍错。

## 成本必须说明

每seed每arm/容量的8个hot case×16流×(8prime+2048write)=263168次字典更新。没有只因查询少就跳过写入，也没有去掉神经state更新。

容量4/8每流保存4/8个entity/value整数对，若假设每个int64，numeric payload为64/128bytes；这只是选定数值布局假设，不是信息论或任务必需下界（8类ID和值可位压缩）。JSON旧字段numeric_payload_lower_bound_bytes_per_stream名称过强，保留不回写，按此处假设解释。Python字典实际对象/索引开销更大，不能拿这个数当实测内存。另有96float状态、模型参数、probe日志。

没有测wall-time、token/算力、能耗或端到端延迟。不声称“减少75%总计算”；只减少这条policy在最终探针上发起的字典query比例。全状态原始history不存在，字典保存最新值而非完整事件历史。

## 研究结论

在有限known-ID场景，状态优先和外部最新值回退可结合，但可靠性仍依赖结构风险标记与足够记录容量。此guard无法捕获所有模型类型的错误，测试无错也不构成部署安全性。

下一阶段需要真正live后端、独立验证触发policy、未知/动态身份和文本编码，并比较直接字典/状态/混合方案的实测成本。不能仅用容易字典基线给弱状态兜底后宣布已解决记忆。

6项定向测试通过，fallback_tests.log（包括新增stream数量/形状guard，不改变已保存结果）；原始逐case答案/保留ID/查询命中与source SHA保留fallback_replay.json。无未知历史权限、无删除现有研究数据。
