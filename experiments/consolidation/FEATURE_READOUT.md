# 结构候选1：冻结语义键到A的隔离读出

本批首次增加独立模型结构models/feature_readout.py，原PBSLMv2/Runtime/checkpoint格式不改。它替换实验中的A输出，不是生产残差adapter，也不是完整语言模型方案。

## 假设

已有O1在M-off时主要调整decoder/context，可能难以利用此前明确跨问法对齐的E键。若固定E特征可由小读出学会新映射并换问法使用，说明表示访问路径值得研究；不等于已证明架构更优。

## 对照

- e_key：冻结原始checkpoint全部参数；token_emb→E，token均值后按E归一化并拼接（2×48=96维）。
- token_mean：同冻结embedding的token均值归一化，复制两份成96维。
- 两者同Linear(96,48)、输出归一化并对固定code余弦分类，各4656可训练参数；同seed初始化权重相同。
- 注意复制token_mean不增加有效rank，所以相同参数个数不等于有效特征维度/信息或相同FLOPs。E路径额外冻结E计算，且E此前受显式键对齐监督；未做未对齐/随机E消融。
- M全过程off且零；只训练readout，梯度不流入底座。E前向内部可能仍计算零M读取，但返回的read未使用；“M-free”指没有M信息参与预测，不代表完全没有矩阵运算。
- old4/new4相同颜色集合、标签与shared_soft200完全相同；教师概率原样复用并校验checkpoint SHA和非M state hash。
- old阶段truthCE200step，new阶段soft normal M teacher200step；sample/template、步数、lr与此前O1相同。
- 梯度隔离、M guard、同初始化和参数数单测通过；全仓库181 tests OK（8.726秒CPU），all_tests_feature_readout.log保留原始输出。
- e_key/token_mean各自学习old事实，不是相同old模型状态分叉；O1原old训练起点/参数量不同，仅作任务参考，不宣称严格参数/FLOPs匹配。

```bash
python -m evaluation.feature_readout_control --seeds 0 1 2 --base-steps 200 --steps 200
python -m evaluation.summarize_feature_readout
```

## 三seed结果

| 路径 | old同表面之前→之后 | old held之前→之后 | new同表面之后 | new held之后 |
|---|---:|---:|---:|---:|
| e_key readout | 100%→100% | 100%→83.33% | 100% | **91.67%** |
| token_mean readout | 100%→100% | 75%→50% | 100% | 54.17% |
| 原O1 soft参考 | 100%→54.17% | 50%→33.33% | 100% | 45.83% |

e_key new held每seed75%/100%/100%；token_mean37.5%/50%/75%。每项仅8query，三seed描述均值不做显著性。summary逐prediction/target核对原aggregate一致，完整curve/预测/hash在feature_readout200.json。

## 支持的信号

- 当前任务中，访问已对齐E表示的小读出比该token均值对照更易迁移到同实体held模板。
- M输出目标能在M-free小读出中学习；不要求反向修改整个预训练网络。
- 旧held仍从100%降至83.33%，并非无遗忘方案；不回放只在这8实体/固定预算下训练表面保住，不能扩成长期保持保证。

## 不能推出

- E已显式对齐、所有实体名熟悉、held模板主要前缀变化，不是未知语义/新关系泛化或自然语言理解。
- 这是teacher-label监督训练，M没有自发决定何时内化，误M仍可能固化。
- readout绕过大部分A路径，不能把收益全部归因模块位置；额外随机/未对齐E、同有效rank、更多参数与任务对照还缺。
- 无live参数切换、自动权重版本管理或持续事件语义续接；仍是离线隔离copy实验。
- 小模型不能代表语言预训练充分的模型。规模升级前需多批内化、原episodic读取能力与独立新问法验证。

## 下一步结构判据

保持这个候选独立，先用多批事实和未知表面验证收益/遗忘，再比较冻结对齐E、随机E、原O1和简单向量检索。若信号不保持，不把此结构并入主模型；如果保持，再研究可靠性筛选、可撤销工作状态与慢adapter版本切换。
