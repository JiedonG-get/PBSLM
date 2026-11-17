# 固定快速状态：键几何与热写干扰

沿用content_fast600原权重，不训练、不使用原始历史。delta/additive分别保持各自训练权重与读取MLP，不能视为同权重仅切换一个算子的因果实验。

```bash
python -m evaluation.fast_state_geometry
python -m evaluation.summarize_fast_geometry
```

## 预设场景

每seed32流，8个16维单位键，升序各写一次值后只更新hot0，检查点0/1/8/32/128/512/2048。对同seed各rule/geometry共用原随机矩阵和value随机序列：

- random：球面独立随机键，训练分布相近；
- orthogonal：原矩阵QR生成的正交键；
- correlated：sqrt(.9)×共同正交方向+sqrt(.1)×各自正交方向，任意不同键cos≈.9。

orthogonal/correlated改变训练键几何；固定hot0/priming不是完整真实流覆盖。独立球面unknown每流只采一个，重复probe，不是与correlated键相似的困难负例。

## 三seed pooled最终探针

| rule/geometry | 初始cold | 最终cold | 最终hot | unknown拒答 |
|---|---:|---:|---:|---:|
| delta/orthogonal | 100% | 100% | 100% | 83.33% |
| delta/random | 89.58% | 83.78% | 100% | 86.46% |
| delta/correlated | 22.77% | 12.65% | 100% | 98.96% |
| additive/orthogonal | 100% | 100% | 10.42% | 30.21% |
| additive/random | 83.78% | 13.24% | 9.38% | 33.33% |
| additive/correlated | 35.57% | 13.10% | 10.42% | 36.46% |

每格hot/unknown96query、cold672query；同流各checkpoint重复相关，三seed只是描述，没有显著性/长期安全保证。

delta/random最终cold错误109项中93为误UNKNOWN、16为错值；delta/correlated最终cold587项错误全部是错值，占全部cold query87.35%，没有误UNKNOWN。相关键下unknown拒答高，不能掩盖已知查询的错误断言；未知probe本来是独立随机向量，较容易区分。

## 数学与归因

单位key的delta同键read更新为(1-beta)*old_read+beta*v，因此能替换重复写旧值；对其他q的影响正比(q·k)，非正交仍有干扰。正交键条件验证此机制，不证明语言键可免费变成正交。

correlated组初始就低（22.77%），不能将最终低值全部归因2048步遗忘；表示条件、顺序写干扰和未见分布共同存在。此构造8键Gram=(.1)I+(.9)11ᵀ，特征值.1与7.3、条件数73；键仍线性独立，不能宣称任何线性记忆都数学上无法解。当前delta规则并未维护逆Gram去除键相关性，所测失败不是普遍不可辨识证明。Additive正交cold不被hot干扰但hot旧值累加失败，说明保持未改记录与更新最新值是不同要求。

候选没有学习key encoder，直接采用输入坐标与点积；真实E/text特征可能高度相似，必须面对该限制。需要条件化寻址/分离、额外状态或检索，不应仅调读出阈值声称解决。

## 证据与边界

研究checkpoint weights_only/有限tensor/配置与strict shape加载，记录文件SHA和权重前后hash。全部probe保存预测/target，summary逐条重新计数已知错误分区和未知误接受。query克隆state确认只读；无参数更新、无历史回退、无语言能力或速度收益声明。

geometry单测独立验证Gram（orth I、correlated .9offdiag）、confusion分区；另4项fast-state数学/梯度测试（包括独立手算cross-key影响），共5项通过，geometry_tests.log。没有重复跑其他历史实验。文本数学复核未执行代码/核算原始artifact，不冒称独立实测数据审计；summary计数检查由主代理执行。

下一步应先在相同权重下测多次写入/不同键角度和容量变化，明确何种输入键需要拒答/检索；学习key映射或接PBSLM E时须单独训练/验证，不能将随机球面结果外推连续语义理解。
