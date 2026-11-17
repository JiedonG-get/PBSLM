# 随机连续键：内部状态＋UNKNOWN拒答触发字典回退

## 目标

验证“优先使用内部状态，显式拒答后再回退”是否在状态无法可靠持有随机key时有用，同时记录查询回退率。此测试不假装省算力：dictionary仍处理每次观察。

## 任务/训练

- 每episode新生成8个均匀随机、归一化16D key，各绑定随机排列的8值之一；已知key的身份和值每轮重新抽样。
- 写入8条后，继续写入/查询交错；25%未知key，25%事件带同一已知key的新值纠正。unknown标签为第9类。
- GRU state宽96；模型输入仅连续key/value和state，查询没有entity ID/seen bit/raw history。
- 每update batch32，episode8初始write +16 query；800updates，每个episode重新抽key和值。不在一个episode内做纠正训练。
- 评估24/96总事件、key noise0/.05；eval包含纠正事件（与训练分布不同），state/参数固定。

```bash
python -m evaluation.continuous_key_state_control --seeds 0 1 2 --updates 800 --validation-batches 16 --output experiments/recurrent_state/continuous_keys_final800.json
python -m evaluation.summarize_continuous_keys_final
```

字典阈值在独立校准key几何验证集的固定网格上，按balanced accuracy选择0.94，不看测试episode label；这是finite synthetic validation，不代表部署校准。

## 3seed均值，96事件

| key noise | state已知准确 | state未知误接受 | state拒答触发fallback比例 | 混合总准确率 | exact key dict |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.41% | 22.59% | 77.60% | 83.13% | 100% |
| .05 | 6.02% | 18.44% | 81.96% | 86.47% | 100% |

Reset状态known准确率0%，unknown准确率100%；它学到/偏向UNKNOWN决策。state-known极低说明它并未形成可靠连续key关联。

24事件混合总准确率94.14%（噪声0）、96.88%（噪声.05），但fallback分别调用91.67%、95.83%。短episode高总分主要由字典承担，不是内部状态成功。

独立字典100%是因为它保留本episode的全部精确16D key/value，对unknown在该高维分布下阈值也完全分开；是很强的检索oracle baseline，不代表自然文本历史检索。

## 含义

- 当前GRU没有利用递归state稳定存储随机key关联；错误显著。
- abstain→fallback可以在此synthetic分布修正多数known错误，但调用率约78–82%，没有可声称的计算节省。
- fallback不由“state对/错”直接知道，而由UNKNOWN类别触发；state对未知key误自信时仍可能不查字典。
- 写入更新/纠正目前只在eval流出现，训练期没有纠正；不将此结果解释为纠正能力。
- 精确字典全收录8key，memory外部提供强先验；没有字节/FLOPs/物理时延匹配。

## 源码/工件说明

最初连续键控制器的错误接口版本尝试已失败，未产出模型指标；后修订训练及评估使用当前模型的`observe/query`接口，并写入独立continuous_keys_final800.json。旧continuous_keys_correction800.json是较早artifact、缺少state-abstain-fallback新字段，不用于本报告汇总。最终summary只读新文件，防止混源。

模型权重没有作为production checkpoint上传；JSON记录800update权重SHA、阈值、seed、每配置聚合与source fingerprint。脚本未保存逐query键值/预测，因此不能外部逐例重算，保存的是aggregate。未提供真实自然文本/语言embedding。

## 决策

不因混合总准确率高就启用历史回退或删状态日志。内部known精度极低、拒答率极高，不值得以“省计算”部署。下一步若继续，应先验证可学习query/key对齐或地址分配，再测confidence—coverage，并让fallback独立retrieval真正计入成本；语言/未知关系仍需单独benchmark。
