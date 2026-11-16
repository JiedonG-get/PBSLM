# 持续Online运行：阶段验收与剩余能力缺口

当前主线以用户明确的持续Online运行为首要目标，固定权重，不优先Online Learning。本文件补充第三阶段报告后续的过载、时序与相关性进展。

## 本次阶段完成的工程闭环

- 持续输入/可取消输出，generation fence防过期提交。
- 真实PBSLM串行推理与有界token rollout，固定权重/M；小模型输出不验收为流畅语言。
- 已提交prefix/history/latest tokens有审计接线，但混合角色布局未训练，不能证明语义理解。
- 独立输入worker在静默/打断说话期间持续处理，queue容量拒绝与producer重试明确。
- generation容量满后latest-only自动恢复，后续只听/interrupt会清掉pending。
- background接口允许显式无关输入不取消表达，保持active/pending原回答目标；hard更新仍fence。
- 保守entity dependency策略和同流hard-all/background-all对照，明确表达进展与过时断言必须同时评价。
- 源序号策略独立验证同源乱序不倒退，未接真实感知源。
- CLI可体验reference/reader/真实小模型不同后端；能力标签明确。

## 证据与负结果

- 千次operation×3seed：安全边界通过，但hard-all持续重规划导致输出饥饿。
- 固定100event路由对照：hard-all输入阶段0输出、0过时；dependency32输出、0过时；background-all33输出、44过时。全部脚本与oracle依赖，非神经相关性能力。
- 普通精确slot字典高于矩阵M；尚无PBSLM架构更优证据。
- 学习型事实提取在未知句式上不可靠，校准不能保证安全，不能自动写持久M。
- 最近全仓库170 tests OK（8.923秒CPU），experiments/continuous_online/all_tests_routing_comparison.log。测试数反映实现契约，不反映智能水平。

## 不可宣布实现的目标

- 模型级持续理解：输入更新不是只入队，而要影响真正语义状态；现在只有小任务/符号state/单问句模型证据。
- 自然表达续连：真正根据已说与新观察修正长语言输出，避免重复/矛盾；当前renderer与短rollout不能证明。
- 自动relevance/说听policy：仍caller指定，依赖oracle/精确规则，不具备人物关系、事件因果与隐含相关性判断。
- 电影音视频感知、同步、实时TTS/GPU抢占、硬实时SLO、崩溃恢复与无限长期内存控制。

## 下一阶段应先固定的验收

1. 人工标注连续短故事输入，问题/关键纠正/无关细节在输出中到达；给每事件相关性和truth变化标签。
2. 同时评分输入处理延迟、表达取得进展、漏打断/误打断、当前断言正确性、已说内容保留及修正；不能一个总分混淆。
3. 给定可用固定语言后端，先比较普通生成loop与本Runtime；语言质量与调度收益分开。
4. 若PBSLM checkpoint质量不足，应独立安排受控语言训练预算/已有权重评估，不能靠协议层文字模拟替代主模型。
5. 先文本后audio/video；所有新模型下载、训练扩大和外部依赖单独记配置/资源，避免不可复核演示。

当前目标的“最小异步闭环和真实接线”已有证据，但“流畅语义续连、持续实时理解”仍未完成。研究目标保持活动，不把阶段工程完成标作最终能力完成。
