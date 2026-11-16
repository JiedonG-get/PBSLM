# 持久化中文事实 Runtime 流结果

日期：2026-11-16。使用已保存的三份严格可加载文本模型（models800），将受控中文问句接入真实PBSLMv2→同步Runtime→显式颜色反馈→有界/累加M。

## 协议

- 8个熟悉设备名、8个单BPE颜色；每个事件输入一个限定模板问句；slot由调用方显式提供 `entity:i/color`。
- 每个事件先预测、提交或第一次事件部分提交后打断，再反馈；反馈确认后写M；模型权重不更新。
- 96个事实事件，其中每12个插入错误反馈后立即给正确纠正，共8个纠正、104次事件；所有3种arm同一seed事件流。
- 处理一半事件时snapshot/restore到另一模型实例，并对同一后续问句检查预测完全一致。
- fixed C输入在每次写M后bitwise不变；非M参数hash不变；stale commit被拒绝。
- `append`：现有矩阵累加；`bounded4/8`：外部latest-confirmed slot ledger覆盖并重建最多4/8条。重建是外部管理策略，不是M自动学会覆盖。

运行命令：

```bash
python -m evaluation.text_runtime_stream --seeds 0 1 2 --events 96 --output experiments/text_runtime/stream96.json
```

## 三 seed均值

| arm | 沿流最新truth平均回忆率 | 最终回忆率 | 纠正后立即读对 | 反馈应用write调用数（不含restore） |
|---|---:|---:|---:|---:|
| append | 41.25% | 45.83% | 58.33% | 104 |
| bounded4 | 59.63% | 50.00% | 100.00% | 406 |
| bounded8 | **96.48%** | **95.83%** | 91.67% | 768.3 |

表中的write次数仅为反馈确认应用，不包含中途restore重建。按本次保留槽位计，restore额外增加bounded4每seed4次、bounded8每seed8次；包含restore的总次数均值分别为104、410、776.3。原JSON字段 `numerical_write_calls` 的旧名称过宽，文件保留不覆盖。

第一次“部分提交”仅提交JSON token-ID文本的首个字符（通常是 `[`）；随后resume处理同一观察，没有并发新内容。它验证过期候选/字符提交协议，不是正在生成自然语言token时的实时打断。

三seed均通过：

- 旧候选被打断后再次提交受控拒绝；
- 中途snapshot/restore后同一模型问句预测一致；
- 参数不变；
- C bitwise稳定；
- 反馈stage不会提前写M；
- write确认幂等。

结果JSON保存每事件：实体、问题、反馈、反馈前预测、反馈后当前槽位查询、所有已见实体探针、保留槽位和memory version。

## 解释

- 8槽位外部覆盖策略在该负载下明显优于无限累加，但计算代价高：每次更新需要重建保留槽位；它不是矩阵M内生的覆盖机制。
- 4槽位能较快纠正，但会淘汰较旧实体，因此最终整体低于8槽位。
- 累加方案保留旧贡献，遇到同一实体反复改值会干扰，纠正率和最终保持明显下降。
- 这是一个比随机关联更接近目标的真实PBSLM文本调用链，但仍是受限模板和显式slot/颜色反馈；它不做开放文本事实抽取。

## 主要限制

1. 只有8个实体、8个颜色和限定模板；模型输入是compact BPE子词，不是通用16k语言模型推理。
2. slot由调用方提供；模型没有从自然文本自动识别实体或验证事实。
3. feedback由模拟环境直接给出；没有来源鉴权、置信度门控或事实验证。
4. 三seed无显著性检验；写入调用数不同，不能宣称bounded8同等预算更优。
5. 运行同步、单样本、CPU/GPU可取消和并发安全尚未实现。
6. checkpoint是推理专用文件，权重不上传；只保存hash、配置、事件轨迹和provenance。
7. 纠正只验证立即查询，没有延迟、冲突多方反馈和长期污染曲线；反馈前使用模板0，反馈后探针为模板4/5，不能把前后差异纯归因于数值纠错。表中只报告反馈后的正确率，不是撤销已采纳错误的比例。
8. 恢复检查来自代码中的fresh-model加载、快照相等与同问题预测相等断言；原JSON只保存通过布尔值，不能仅靠JSON独立重建恢复前后数值M。源码fingerprint覆盖所列文件，不代表所有依赖都已指纹化。

## 后续证据口径修复（不覆盖原实验）

`evaluation/audit_text_runtime_accounting.py` 只读取既有stream96.json，输出accounting_audit.json，核对逐事件probe汇总并派生恢复写入成本，不执行模型。

`accounting_v2_pilot12.json` 是后续单seed、12事实+1纠正的短接线检查：脚本分别记录feedback_application_write_calls、restore_rebuild_write_calls和总数，保存fresh零M哈希、期望/恢复M哈希与两份同问题预测。三种arm均通过恢复哈希相等、参数/C不变；该新pilot不替代原三seed结果。对应全仓库115 tests OK，原始输出all_tests_after_accounting.log。

本结果支持“受控文本模板上的显式反馈记忆接线可运行”，不支持“PBSLM已经实现实时自然语言Online Learning”。
