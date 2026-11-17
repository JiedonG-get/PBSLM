# 动态键阶段报告：从固定ID到内容状态，尚未统一可靠身份

## 主线与阶段范围

目标仍是PBSLM持续Online运行：持续输入、维护当前有效理解状态、选择表达与自然打断续连。内部状态优先是机制方向；本阶段没有开展CNN/逆傅立叶支线，没有删除历史数据，没有宣布一般语言或完整目标完成。

本阶段完成动态连续键任务、固定模型评估、标准参考与一个真实异步状态适配器。多数模块独立于原PBSLM A/C主干，不可把独立实验的各项成功相加当完整系统成绩。

## 已经确定的机制信号

| 问题 | 当前证据 | 不能推出 |
|---|---|---|
| 固定ID高分能否迁移到新连续键 | 128维globalGRU在原600update任务96步known约5.2%，UNKNOWN偏置严重 | 不能否定所有递归状态；单模型/有限预算 |
| 内容绑定/替换是否有用 | 16×8 delta状态96步known约86.5%、unknown判对87.1%；additive known37.5% | 标准delta-rule，不是新框架或语言能力 |
| 键相似是否重要 | 正交键隔离；.9相关键cold崩溃；explicit dual满秩参考可解 | 不能称相关键任务数学无解；dual额外存记录/索引 |
| 精确读出能否保证未知安全 | dual在相关键的随机UNKNOWN大量误接受；nearest阈值强噪声known大量拒答 | 不同readout分布/身份可观察线索须分开 |
| 能否接持续运行器 | fast-state真实CPU worker静默写/查询/纠正、旧输出fence，权重固定 | structured键/值，不是原PBSLM语言续连 |
| E能否直接作fast-key | 固定投影零适配known下降，identity投影改善熟悉名字 | unknown名字仍弱，不是全语言整合 |
| 额外span统计能否拒答 | held known覆盖崩溃至约1%，满秩后gate不起作用 | 低FA靠拒所有输入不是安全成果 |
| 保留有限显式键如何 | capacity8精确随机key参考可解，capacity4淘汰丢失，写噪声可重复分配挤出cold | standard算法memory，不是神经压缩状态胜利 |
| 保守写入是否足够 | anchor/noeviction保护cold，但强噪声热最新值仅27.1%，第9新身份全拒 | old保持必须与新信息获取共同评价 |

各百分比来自不同明确任务/样本，不能直接跨行比较优劣。

## 实验索引

- [连续键global基线](../experiments/recurrent_state/continuous_keys600/RESULTS.md)
- [delta/additive快速状态](../experiments/recurrent_state/CONTENT_FAST_STATE.md)
- [键几何与hot干扰](../experiments/recurrent_state/FAST_GEOMETRY.md)
- [显式dual几何参考](../experiments/recurrent_state/DUAL_GEOMETRY.md)
- [噪声与近未知身份边界](../experiments/recurrent_state/QUERY_BOUNDARY.md)
- [真实异步适配器](../experiments/recurrent_state/FAST_STATE_ONLINE.md)
- [E零适配迁移](../experiments/recurrent_state/TEXT_KEY_TRANSFER.md)
- [identity键对齐](../experiments/recurrent_state/TEXT_KEY_ALIGNMENT.md)
- [span拒答负结果](../experiments/recurrent_state/TEXT_SPAN_GUARD.md)
- [显式动态内容槽](../experiments/recurrent_state/DYNAMIC_CONTENT_SLOTS.md)
- [写噪声导致状态污染](../experiments/recurrent_state/WRITE_NOISE.md)
- [保守写入接收取舍](../experiments/recurrent_state/WRITE_ADMISSION.md)

## 接入主模型的判断

### 可保留作为下一整合候选

- 角色分离：query、观察value、已提交表达不混用；不能将模型说过的话自动当事实。
- 内容关联delta状态作为低存储参考，明确其地址干扰与未知检测不足。
- 有限显式key槽作为强基线与可控记录，不重新命名为A内化。
- 输入worker与表达任务分离，版本fence保证过时输出不提交；模型答案仍单独验收。

### 暂不进入可信回答路径

- 未经新分布校准的UNKNOWN分类头或cos.9阈值；
- held覆盖崩塌的span guard；
- 不带来源/身份连续线索的破坏性覆写和无限保守拒写；
- 随机投影接线即视为语义对齐；
- 使用评分truth或caller索引的参考算法包装成自主记忆。

这些是候选筛查结论，不是某模块以后永远不能改进。

## 下一阶段最小可辨别任务（不再泛化堆小实验）

统一一项端到端事件基准，至少包含：

1. 每episode新身份、明确query/观察角色、可观测的身份连续性线索。不要让相同向量同时可被任意解释为同ID噪声/新ID而不提供区别。
2. 正常新增、同ID修正、写入有噪声、容量耗尽和来源冲突；少量真实文本表面但标签/训练/test隔离。
3. 相同事件流下比较直接检索、固定矩阵状态、混合状态；保留全部接收/拒绝/淘汰/检索调用账本。
4. 同时评分latest正确性、旧记录保持、未知误接受、新信息接收、误/漏打断与表达取得进展；不得只报告accepted100%。
5. 接原PBSLM E/A前，明确状态维度/训练损失/身份接口及额外参数/维护成本。语言底座质量不足时应单独预算语言训练或固定后端验证，不用规则renderer代替。

## 审查与可复核范围

大多数本阶段独立复核为文本数学/设计审查，不执行代码、不读取原artifact；报告明确标注，不等同于独立预测数据审计。阶段最后对保守写入进行了严格限定文件的独立只读审查，重算9run热冷计数与第9身份144个probe均一致，没有运行模型/测试。主代理对保存prediction的实验重算计数；只保存aggregate的运行不能追溯伪造逐query证据。

一次只读设计子任务曾越权修改、训练与提交，引发源码接口不一致；恢复后的600update源码SHA与原capture一致，strict权重加载通过。额外800update研究保留在独立文件/报告，不并为独立复现或混入600update基线。后续实现采用主代理独占写入，文本审查不调用工具，避免共享文件并发修改。

当前最终回归241 tests OK（9.241秒CPU），日志[stage_final_regression.log](../experiments/recurrent_state/content_fast600/stage_final_regression.log)。这是实现契约，不是理解能力分数。

## 目标状态

12轮自动推进上限仅表示本次预算结束。动态键、未知/覆盖/干扰机制任务和真实状态adapter已有证据；固定ID依赖被部分移除，但可靠身份/拒答/语言泛化/算力优势/原主模型整合/持续语义续连尚未完成。目标保持未完成，不以阶段收口标记整体成功，也不把困难当作外部阻塞。
