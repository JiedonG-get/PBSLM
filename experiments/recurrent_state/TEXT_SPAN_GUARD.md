# 键子空间拒答候选：低误接受不能靠拒掉所有查询取得

本批冻结E、已训练identity投影及fastdelta，不训练模型。新增固定大小KeySpanState保存已观察key张成空间的正交补P，不保存raw key列表。

```bash
python -m evaluation.text_span_guard
python -m evaluation.summarize_span_guard
```

## 几何状态与额外成本

P初始I16。每观察单位k，r=Pk，若||r||²>1e-12且rank<16，则P←P-rrᵀ/||r||²，随后对称化。query novelty=||Pq||。它检测‘是否偏离已观察span’，**不是是否见过相同身份**；span里的新组合仍可能通过。

P=16×16float64=2048bytes/流，加原fast状态512B，总2560B tensor payload；不计投影、E/fast权重、Python/rank、求解中间量。没有总体省内存/算力声明。当前无遗忘/撤销span方向，达到16rank后P≈0，检测能力饱和。

## 预先固定校准

每seed32校准episode、support8从前12熟悉名字抽，record模板0/1，known query模板2/3。阈值为全部known residual的higher95th percentile，约.0890/.0666/.0752；known校准保留95.3125%。不使用颜色/unknown标签选阈值，未按held调参。

这仍是同熟悉实体的几何校准，不是独立新身份校准。前12名字与投影训练一致、具体采样流新；query held4/5和另4名字不进入校准。

## held测试结果（三seed pooled）

每seed support8/16各16episode，query全部16名字、held模板4/5；随机颜色独立于key。guard在novelty超阈值时强制UNKNOWN，否则保留fast输出。

| support/方法 | known值正确率 | known回答覆盖 | unknown误接受 |
|---|---:|---:|---:|
| 8/原fast | 75.26% | 86.46% | 24.22% |
| 8/span guard | **1.04%** | **1.04%** | 0% |
| 16/原fast | 66.02% | 89.19% | 无unknown样本 |
| 16/span guard | 66.02% | 89.19% | 无unknown样本 |

support8中每seed128known/128unknown，support16每seed256known。guard只剩4/384known答对，380known误UNKNOWN；不能将unknown0误接受写成可靠性成功。

support16所有episode rank16，此时P近零，gate不起额外作用；没有unknown样本，所以不能从该组测得unknown识别率，理论饱和限制来自投影结构及单测。

本批流seed+232000，不与旧投影文件79.95%直接作政策百分比差；此文件内state/guard严格共享预测和输入，差异只来自明确gate。

## 解释

- template2/3校准接近训练对齐范围，而held模板4/5表示仍漂移，几何residual不稳定；known阈值无法保证跨问法coverage。
- ‘同span’与‘同身份’不同；unknown可能位于span，query改问法也可能偏离span。任何低FA都必须伴随known覆盖/值准确率报告。
- P额外增加存储但未解决身份语义，不能用‘全息/子空间’名称替代实测。
- 未知query仍在有限16name库存中，未开放词表、新关系/真实自然语义；颜色反馈直接给出。
- 不支持动态删除/纠正key身份、窗口老化或overflow策略，不作为live可信gate。

## 验证

summary从保存predictions重新计数所有指标，核对每条gate决定=novelty>阈值。记录校准输入/残差、threshold、test输入/预测/target、span rank和模型/投影来源。固定模型hash未变；模型查询不接收target。

5项定向测试（span2、projector1、projection2）通过，span_guard_tests.log。测试明确演示未见span线性组合也会novelty0、满rank时任何query都不新颖；对照显式Gram projector验证数学。文本数学复核未执行代码/读取artifact，不当作独立数据审计。

下一阶段不继续提高阈值追test覆盖，而应定义身份可观测协议/独立真实问法验证，再决定学习新颖性或保留显式key证据。当前候选不接入Runtime，也不修改原PBSLM主模型。
