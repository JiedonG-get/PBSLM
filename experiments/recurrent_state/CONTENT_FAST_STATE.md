# 连续键快速关联状态：预定更新对照

新goal主线仍是持续Online。此候选独立于PBSLM主干，既不删除旧历史记录，也不自动接入真实Runtime。

## 结构和任务

每个流持有16×8浮点矩阵S（128float=512bytes），不是原始key/value列表。写入连续归一化键k与value embedding v：

- delta：S ← S + sigmoid(b) × outer(k, v − kᵀS)
- additive：S ← S + sigmoid(b) × outer(k, v)

query q仅读归一化qᵀS，再经8→48→9类MLP输出8个值或UNKNOWN。这里的S是独立快速状态，不是原PBSLM M参数，也不是慢速A内化。标准delta-rule/fast-weight关联记忆思想，不主张新算法。

value embedding初始化onehot、b初始化2，两arm同初始化/参数形状/事件流/训练预算。这个结构内置key×value绑定操作，与通用GRU存在强归纳偏置差异；相同state数不代表同计算/参数/信息辅助。

沿用continuous_key_control.py原600update、batch32、24step任务：每episode全新8个16D随机键，每步随机键更新值，query75%抽pool（可含尚未写键）、25%全新absent；两种未知均UNKNOWN。模型只有当前写key/value和递归state，query只有key/state，没拿到entity index、truth、seen mask或原历史。

评估24/96步，noise0/.05、query-reset，相同CPU generator种子和draw顺序。读权重固定；状态仍随观察更新。train24步BPTT不是常数内存训练；研究eval保存loss列表/计数，也不代表整个进程常数内存。

```bash
python -m evaluation.content_fast_control --updates 600 --seeds 0 1 2 --output-dir experiments/recurrent_state/content_fast600
python -m evaluation.summarize_content_fast
```

## 三seed结果（描述均值）

| 方案 | 24步已知值 | 24步未知判对 | 96步已知值 | 96步未知判对 |
|---|---:|---:|---:|---:|
| delta状态 | 88.43% | 93.30% | 86.46% | 87.05% |
| additive状态 | 58.77% | 91.68% | 37.48% | 64.89% |
| 原global GRU参考 | 1.42% | 98.04% | 5.17% | 87.83% |
| 显式键值bank参考 | 100% | 100% | 100% | 100% |

两candidate都是938参数、128float状态；原GRU85193参数，同state不等同结构/先验/计算公平。delta的96步known各seed86.67/86.60/86.11%；noise.05下known85.63%、unknown86.75%。两candidate query-reset known0/unknown100%，说明readout依赖state，但零state时偏向拒答。未给出逐错误类别，不能区分known错值与误拒答，也不宣称校准安全。

最终delta门值约.864–.867，additive约.935，说明训练后权重/门幅度不同。梯度通过residual read进入前状态，未detach residual；不是另一种只调整前向的规则。

summary重新核对全部保存计数和原GRU评估分母一致；它不是逐事件预测审计。三seed重复step相关，不作显著性/泛化保证。数学只读复核确认同键read被拉向新值；其他query的变化正比q与写key的内积，所以仍有交叉干扰，不承诺长期冷保持。

## 判据与限制

known与unknown分开计数，不能用拒答多掩盖值回忆失败。原bank参考存储实际key/latestvalue/seen、固定cosine.9，是强标准存储；不按当前query答案选记录。它在现任务可解决，不能将神经候选成绩当架构优越。

delta与additive会训练出不同beta/embedding/读出，因此成绩差不是严格同权重单操作干预；本批比较完整训练方案。要更强归因需冻结同权重cross-rule评估及固定beta控制。

未知拒答来自训练分类头，未经校准；未见key仍来自同随机球面分布，不是新语言/新概念。键相似与状态冲突、近邻难负例、容量超过8、长偏斜写入、真实文本E键/语义续连尚未测试。

模型checkpoint本地ignored，公开source/model SHA和汇总；无逐event预测，summary只核对保存计数和各arm查询分母，不伪称逐次输出可重算。只做定向实现测试，不重复全部历史训练。
