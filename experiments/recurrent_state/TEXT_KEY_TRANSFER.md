# PBSLM E文本键直接迁移：固定投影不是充分对齐

本批固定两套既有模型：文本compact PBSLM的E编码器，及随机球面键训练的16×8 delta状态。无训练/投影搜索，不改原PBSLM A/C/M或Runtime。只是‘PBSLM派生E特征用作键’的接线诊断，不是完整语言记忆整合。

```bash
python -m evaluation.text_key_transfer
python -m evaluation.summarize_text_key_transfer
```

## 预定设计

- 文本E输出2×48维键，拼接后归一化为96维。
- 固定seed+211000随机QR得到96×16投影（列正交），然后再次归一化。不根据数据/答案拟合，不挑成功seed。
- 三seed，support4/8/12/16，各16episode；每次从16设备抽唯一支持实体、随机8色（可重复），support用模板0/1。
- 每次query全部16实体，same_surface中已写entity使用其实际support模板，未写entity模板0；held模板4/5。
- 反馈直接提供0..7颜色类；未写实体应输出8 UNKNOWN。实体整数index仅供脚本选择输入/评分，faststate仅看投影键/值。
- 参考：原96维/投影16维最近邻，各有always输出与固定cosine.9阈值；阈值未在文本数据校准。
- 预训练只优化过前12名字，另外4名字虽未优化但已预配词表；不是开放新词/无限身份。

## 三seed pooled held结果

| support/方法 | 已知值正确率 | 未写实体误接受率 |
|---|---:|---:|
| 8/fast16 | 55.73% | 27.08% |
| 8/nearest96 always | 89.84% | 100% |
| 8/nearest16 always | 85.42% | 100% |
| 8/nearest96 cos.9 | 34.64% | 0% |
| 8/nearest16 cos.9 | 34.38% | 0.52% |
| 16/fast16 | 38.54% | 无未写query |
| 16/nearest96 always | 85.29% | 无未写query |
| 16/nearest16 always | 80.99% | 无未写query |

support8每group384query、support16已知768query；query同episode相关，无显著性宣称。always方法不能拒答，所以未知100%误接受不是隐藏失败。固定阈值大量拒掉已知query，不能只展示其低误接受。

同值碰巧正确会抬高value率。另计nearest的真实record选择：support8 raw88.02%、projected83.07%；support16 raw83.98%、projected78.78%。faststate没有显式record index，不编造record准确率。

## 判断与限制

投影后检索有一定损失，但faststate进一步下降，与跨模型键分布/地址干扰/读出分布共同相关。不能将全部差值归因投影或某一网络组件：nearest与faststate本就有不同存储/读取归纳偏置。

E键受实体跨模板对齐监督，非一般文本encoder；faststate只在随机球面任务训练。该结果是零适配迁移，不是对任务适配训练后模型的上界。相同旧实体/表面库存，外部明确query与反馈值，不能声称语言事实抽取。

support16超过faststate训练时8livekeys，同时名字/投影几何有变；不得单因归因为容量。只随机选了预定的一份projection，没有更多projection重复或调参；3seed同时改变原E、fast权重和投影，不能单独估计projection方差。

## 资源账本

- faststate每流128float32=512B；其938参数=3752B模型参数（共享流），不含E编码器/中间量。
- 固定投影96×16float32=6144B（共享流），不是零成本。
- 8条nearest96键float32=3072B+8个int64 label64B；16维键为512B+64B；全部为tensor payload，不计Python/index/E/码本/实验缓存。
- 没测FLOPs/时间/峰值内存，不因512B状态便宣称总体省算力。无实际fallback。

## 证据

text_key_transfer.json保存全部episode的支持entity/template/value、query模板、target/prediction、nearest record/cosine；模型hash前后不变，源checkpoint/file/projection SHA记录。summary重新计数全部预测；初版value summary保留，record_summary额外记录实体检索，未覆盖原预测。

6项定向测试通过（projection固定/列正交、缺unknown分母与错误分区、faststate原数学契约），text_key_transfer_tests.log。文本只读设计审查未执行模型或独立读取数据，不能称独立实测审计。

## 下一步

先建立独立训练/验证/测试的键对齐：明确记录和query角色、动态身份以及无值泄漏的寻址目标，同时保留未经适配参考。不继续随机换projection寻找好分；模型学习到真实角色/键表示后，才讨论将状态更新并入持续Runtime并测语义续连。
