# 连续事件上下文接线：模型实际收到哪些信息

本批固定权重、不训练、不写M。目的仅是补齐旧backend“只看最新问句”的接线缺口，不宣称语义连续理解。

## 明确的预算与布局

runtime/event_context.py接收GenerationRequest(revision, inputs, committed_prefix)。使用既有BPE与checkpoint紧凑映射，模型输入为：

```text
[左pad][截取的历史输入][截取的已提交输出][最新输入suffix]
```

默认历史上限3token、prefix上限3token，至少2token给最新输入。若剩余预算可用则归还最新输入。截取和每部分span/source token数均记录；未知BPE/UNK明确报错，不静默替换。zero-budget历史/prefix不编码，避免无用段阻塞最新输入；正budget下当前实现仍检查全部候选历史与渲染prefix，已丢弃的旧段OOV也可能拒绝该次生成，这一可用性限制尚未解决。

这没有新增训练过的role token或embedding。span是审计元数据，模型仅看到上述token序列，不能据此说它懂得哪部分是观察/哪部分是自己说过的话。

## 实际神经接线

runtime/pbslm_context_backend.py沿用串行executor，每个generation先打包上下文，step0实际输入写入metrics。后续真实model(m=1)预测一个token并滚动加入固定窗口。新输入revision重新打包，不恢复旧suffix。

```bash
python -m evaluation.context_transport_demo
python -m unittest tests.test_event_context -v
```

context_transport.json包含3次输入、两次首token后新输入、所有packing spans/truncation和实际neural输入。

运行断言通过：

- 每次step0实际输入与packer报告完全一致；
- 续接request包含prior_input/committed_output/latest_input三类span；
- 旧revision不再提交；
- 模型配置/权重/非M buffer与M指纹均不变。

4项纯token预算单测：精确identity、最新suffix保留、unknown拒绝、空历史left-padding。全仓库146 tests OK，6.630秒CPU；原始all_tests_context_transport.log。

## 不能从中推出的结论

- 小模型原本训练输入是设备颜色问句，未训练混合事件/输出布局。本批输出质量未评价；即使token进入也不等于语义使用。
- 12token左右的短窗口只能保留局部片段，截掉实体前缀可能损害任务表现。
- 历史和渲染prefix被重新BPE编码；不承诺复原同一generated token分段，尤其标点/特殊输出可能OOV；模型生成id0或fallback文本可导致下一轮prefix编码失败，只能明确报错或由调用方选择prefix_tokens=0，不算流畅续连。
- spans只描述generation的step0；后续rollout移窗后不能沿用原role spans。metrics改为step0_pack_applied并保留初始generation revision。resume的generation revision可能大于最后input revision，两者分开记录。
- 固定窗口rolling不是状态空间连续理解、KV cache无损续接或长程语义状态。
- should_respond仍由调用方明确给出，不是模型自主选择开口。
- 没有音频/视频、硬实时延迟、GPU抢占或通用生成质量证据。

下一步应采用可验证的连续语义任务：信息逐段到达并改变结论，检查新结论、已说内容关系与静默policy；先把调度正确与模型理解正确分别评分，不能再用传输成功代替能力。
