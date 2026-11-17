# 只训练键投影：熟悉身份改善，未知身份仍失败

本批冻结原PBSLM文本E和content_fast600 delta网络，仅训练96×16投影1536参数。初始矩阵与随机投影对照完全相同。无颜色监督、无主干更新、无Runtime整合。

```bash
python -m evaluation.train_text_key_alignment --updates 600 --output-dir experiments/recurrent_state/text_align600
python -m evaluation.summarize_text_alignment
```

## 预定监督与划分

训练实体前12个，support模板0/1、query2/3；每步12实体配对，随机选问法。600step、AdamW.002、梯度clip1。loss为双向identityCE(cosine/.1)+.1×entity Gram相对identity的均方误差。

所有训练输入仅来自训练entity/template特征；held4/5与另外4名字不进入loss或优化器。E特征本身来自旧checkpoint，前12名字已在原episodic任务训练过，后4只做过词表/输入库存预配置；不能称完全未见语言实体。

该identity配对是额外明确监督，模型知道同一实体跨问法对应哪行；不是无监督语义获得，不含颜色值标签。主干/fast模型参数前后hash检查不变，投影file/hash另存。

每seed同时评估random初始与aligned最终投影，使用相同episode/entity/value/template流。support8/16各16episode，每episode随机颜色允许重复；query全部16实体held模板4/5，未写entity targetUNKNOWN。没有在test挑checkpoint/步数/投影seed/阈值。

## 三seed pooled support8 held

| 方法/指标 | 随机投影 | identity对齐 |
|---|---:|---:|
| fast全部已知值正确 | 60.94% | 79.95% |
| fast熟悉12名字已知值正确 | 67.56% | 89.97% |
| fast另4名字已知值正确 | 37.65% | 44.71% |
| fast全部未写entity误接受 | 28.91% | 27.08% |
| fast另4名字未写误接受 | 56.07% | 57.94% |
| nearest always已知值正确 | 85.68% | 90.63% |
| nearest真实record正确 | 84.38% | 89.84% |
| nearest cos.9已知值正确 | 36.20% | 38.80% |
| nearest cos.9未知误接受 | 0.52% | 0% |

NNalways不能拒答，其未知误接受固定100%，不能只展示已知高分。fixed阈值未校准，也大量拒绝known。record正确率从保存的nearest_entity对已写entity计数；颜色值重复可抬高答案率，必须区分。

support16已知值fast40.76%→66.67%，其中familiar48.26%→80.21%，new18.23%→26.04%；此组没有unknown query，FA为None，不误写成零风险。

以上只有3seed、每group少量实体、16episode，同episode/实体强相关，非显著性或一般语言能力声明。

## 可以支持什么

在当前有限模板任务，identity监督投影能使familiar键更适合既有fast读写。不是底座语言能力突破，投影拟合有限身份/模板可能是主要收益。另4名字泛化弱、误接受没有改善，不能取消fallback或自动信任记录。

与上一text_key_transfer.json不可直接当同流百分比改善：本批仅support8/16，旧批从support4开始消耗随机数，episode具体映射不同；本文件内部random/aligned严格配对。summary验证全部输入/目标配对并重算每预测指标。

## 资源与边界

- 投影1536float32=6144B；fast128float状态512B/流和938共享参数；E底座、运算、检索缓存不计入，无整体省算力结论。
- 初始投影QR列正交；训练后无正交约束，不承诺同几何/独立维数。
- 训练Gram损失是known身份排斥先验，强先验可能损害其他相似身份；未做CE-only/Gram-only消融，不归因某一项。
- 只写一次观察，不测长时覆盖/撤销；fast仍受非正交键干扰，未知分类头仍未经文本分布校准。
- 没有真正自然文本事实抽取/动态实体创建、source可信度或语言生成；query/record角色由接口提供。
- local投影权重存储但无单独loader/reload预测验证，不能冒称生产checkpoint。

## 验证

原始全部train曲线、输入模板/值/预测/目标/nearest选择均保留results.json；summary逐条核对，颜色仅在评估写入、从不参与投影训练。

7项定向测试（projector1、projection2、fast4）通过，tests.log；测试参数/形状、对齐梯度、投影固定、原state数学契约。文本只读设计审查没有读取文件或执行预测，不能称独立实测审计。

下一阶段应先做独立身份/问法的寻址对齐和拒答数据，而不是扩大已知12实体训练步数追高分；检验有无身份可观测区别，再将支持记录有效性/来源接入持续Online快照。
