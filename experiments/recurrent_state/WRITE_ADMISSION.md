# 保守写入接收：保护旧值必须与新信息接收一起计分

本批固定原fast value reader，比较原动态槽（匹配时移动key，满槽淘汰）与保守槽（匹配时锚定原key，满槽未匹配则defer）。同时改变两个政策，不是单因素消融；无训练，无语言/主模型/Runtime整合。

```bash
python -m evaluation.write_admission_control
python -m unittest tests.test_conservative_content_slots tests.test_dynamic_content_slots -v
```

3seed×16流，8随机球面键精确priming，512热0随机值更新，写sigma0/.05/.15。相同输入随机流与前write_noise随机组配对，干净原key查询；truth按最新**已送达**事件计分，即使策略拒绝也更新truth，避免将保持过时值算对。

保守模型返回accepted/deferred；deferred不入队、不保存、不自动重试。外部caller需决定重试或放弃，不能称无损。满槽后另送真正第9个随机身份，检验保护旧记录是否以阻止新身份为代价。

状态同为8×26float=832B，但算法policy/输入接收不同；不声明省算力。固定阈值.9未在此任务校准。只测随机键，刻意不复用上批cos.9等于阈值的边界条件。

## 三seed pooled结果

| 写噪声 | 原策略hot/cold | 保守策略hot/cold | 保守热写接收率 |
|---|---|---|---:|
| 0 | 100% / 100% | 100% / 100% | 100% |
| .05 | 93.75% / 97.32% | 100% / 100% | 100% |
| .15 | 16.67% / 0% | 27.08% / 100% | 25.25% |

每条件hot48query、cold336query，接收计24576次热写。另送第9身份，跨9run共144条：原策略全接受且当场读对；保守策略全拒绝，当场读对0。新身份全部拒绝是容量政策的直接结果，不表示识别了无效信息。表中高噪声hot低分不能只归拒绝：不同写入接收时刻、label偶然相同、reader输出均可影响，未作独立因果分解。

主代理逐保存prediction/target重新核对全部hot/coldcorrect计数，零不一致；6项定向测试通过（0.074秒初次，最终原始输出见content_fast600/write_admission_tests.log）。阶段收尾独立只读审查重算全部9run计数与第9身份144个probe，均一致；审查没有执行模型、测试或读取权重。文档与实验不支持任何语言能力或生产可靠性主张。

原始hot/cold预测、target、实际接收/拒绝总数、最后写入接收标记、9th身份结果保留write_admission512.json。拒绝不代表知道输入真伪，同样可能拒掉有效纠正；锚定key也可能不适应身份表征真实演变。

测试验证满槽未匹配的state完全不变、匹配后保留anchor并更新value、原动态槽覆盖/recency/错误输入等。候选是参考policy，尚无source/版本/撤销/持久审计或自动置信度。
