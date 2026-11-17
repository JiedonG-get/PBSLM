# 观察键的双基参考：相似地址可以分开，但代价不是免费

本批不训练。固定content_fast600 delta权重与读出MLP，比较局部delta写入与显式保留key/value的双基参考。它不是新算法、无历史压缩优势、未接PBSLM或Runtime。

```bash
python -m evaluation.dual_geometry_control
python -m evaluation.summarize_dual_geometry
```

## 数学接口

已观察8条单位key组成K(8×16)，value embedding组成V(8×8)。满行秩且Gram最小特征值>=1e-8时：

- G=KKᵀ；D=KᵀG⁻¹（16×8），所以KD=I。
- M=DV；已存key读出KM=V。
- caller明确给出第i条记录被覆盖，M+=D[:,i] outer (newV-oldV_i)，更新V_i。

代码用solve(G,K).T求D，不显式求逆。这个baseline已知道全部primed记录的key/value和更新index，是强辅助；不能称自主寻址。所有key先逐条观察完才构建baseline，没有拿未来key。之后读出仍使用原模型训练过的MLP；向量恢复不逻辑保证分类正确，因此两者分别检查。

## 配对流

每seed32流，三geometry=正交/随机球面/.9cosine相关；同seed各condition使用相同基础随机矩阵和值随机序列。两种priming：

- independent：各key随机value；
- initially_same：8key最初value都等于各流第一个随机value；之后hot0照常随机改值，通常会与冷value冲突。

512次热写，固定checkpoints，原delta先后单遍priming；dual在全部priming完成后联合重建，初始计算/信息使用方式不同。不可称只改变一个低级算子的公平同资源对照。

## 三seed pooled冷值结果

| 几何/初始值 | delta初始 | delta512后 | dual初始/最终 |
|---|---:|---:|---:|
| 正交/独立 | 100% | 100% | 100% / 100% |
| 正交/相同 | 100% | 100% | 100% / 100% |
| 随机/独立 | 88.24% | 82.74% | 100% / 100% |
| 随机/相同 | 74.26% | 70.83% | 100% / 100% |
| .9相关/独立 | 24.40% | 10.71% | 100% / 100% |
| .9相关/相同 | 100% | 15.63% | 100% / 100% |

delta最终hot均100%；dual全部hot/cold为100%。每cold格672query，多个checkpoints相关，只作描述。新seed偏移与512长度不同，不与前2048结果当同流时长消融。

相似键同value在priming时易于读出，但出现热键改值后cold失败；支持当前局部更新难以分开重叠地址的假设。不说明所有同value构造都更好：随机键同值布局反而降低delta基线，需要保留读出分布/幅度变化等解释。

## 验证、成本与边界

- dual全probe读embedding最大绝对误差约1.4e-14，预测同时通过原learned decoder；summary逐query核对热冷correct/误unknown/错value计数。
- correlated Gram特征值.1和7.3，条件数73，满秩可解；不能把原delta失败描述为线性记忆普遍不可能。
- 每stream保存float64 K1024B、V512B、D1024B、M1024B，共3584B；delta状态512B。共同decoder/模型参数、临时solve空间和Python另计。可以优化冗余布局，但本次实际就是这些缓存，没有省内存/算力声明。
- dual拒绝重复/秩不足key，不支持新增超过维度/删记录/未知身份自动定位，也没有未知查询拒答测试。
- 近似key噪声会被逆Gram放大；本次只查询精确原key，100%不等于噪声/泛化可靠性。
- 记录identity由caller给出，value直接观察反馈，无语言抽取或真值认证。
- 正交/高相关是离开训练随机球面的几何干预，当前hot0和固定priming仍有限。
- 权重strict load/文件SHA与前后state_dict hash保留，无梯度/历史事件回看；但dual显式保留key/value表，不能说无外部记录。

7项定向测试通过（dual2、geometry1、fast4），dual_tests.log保留实际输出；包含dual在相关键覆盖后与联合重建一致、重复键拒绝、原fast-state数学契约。文本只读数学复核未执行代码或重算artifact，不当作独立实测审计；其中维度表达以此处与代码为准。

## 下一步

检验近键未知与输入噪声，比较精确键表、双基、局部delta在实际开销下的拒答/回忆；再考虑学得的key分离或条件更新。不能为了双基高分就接受caller索引或全键存储作为最终神经结构。研究主线仍是持续Online里的有效状态，不是保存更多参考表后宣布内化完成。
