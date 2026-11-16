# 128组合键容量诊断：小任务优势没有直接扩展

## 任务必须明确

本批不训练神经模型，无M teacher/Online Runtime。冻结现有E，组合键由**两条独立左pad的原颜色问句直接拼接**构成，任意无序设备对被赋一个随机颜色label（不是回答两个问题，也没有自然语言关系语义）。因此这是抽象关联容量/OOD特征诊断，不是中文理解benchmark。

16设备中4个名字未参与原episodic优化；unordered pair含重复，从136个候选抽128，为每个key采用a<=b的唯一规范顺序。没有测试反向输入，不能假定E表示顺序不变（E内部有位置偏置和因果mask）。原输入长度翻倍，无截断；组合/长度/部分名字分布变化同时存在。不能把失败归因架构唯一缺陷。

每pair训练模板2/3，held4/5，共256独特train输入、256held；128随机pair labels分布严格每色16key。随机打乱后8stage×16key，每表面只加一次，区别于此前重复600采样。每stage只用已到达train rows，不拟合未来/held。

```bash
python -m evaluation.pair_capacity_control
```

## 相同持久tensor payload预算

三档sketch rank8/16/32对应9,344/18,688/37,376bytes。reservoir nearest存96维float64 feature+int64 label，每行776bytes；分别可存12/24/48行，实际9,312/18,624/37,248bytes，均不超预算。

这是feature+label/统计tensor payload对照，不是Python/RNG/诊断/缓存/底座/峰值内存全匹配。码本/提取器共享。sketch dense eigensolve有额外瞬态计算；nearest有查询成本，未做时延公平profile。

另有不受这些预算限制的完整ridge和保存256train行nearest，以区分容量与特征迁移。完整nearest是看过的同feature查询训练样本，train100%不代表泛化。

## 三seed最终均值

| 方法 | train | held |
|---|---:|---:|
| 完整ridge | 65.89% | 22.01% |
| 保存全部train行nearest | 100% | 47.66% |
| rank8统计 | 24.09% | 23.18% |
| 同预算12行nearest | 18.23% | 17.32% |
| rank16统计 | 31.25% | 26.82% |
| 同预算24行nearest | 27.73% | 23.44% |
| rank32统计 | 46.88% | 24.48% |
| 同预算48行nearest | 39.19% | 31.38% |

随机8类参考为12.5%，但无统计显著性/广泛语言推断。每seed是不同checkpoint/pair顺序/映射，同seed各方法完全同流；nearest reservoir3档使用相同seed但容量不等，不能当配对同子集。

保存结果的事后分层（按全部seed pooled probe，不是上表seed均值）：仅含原熟悉12名字的pair，完整nearest held61.76%、完整ridge23.98%；包含4个未参与原优化名字的pair，分别28.53%、19.33%。这表明新名字明显影响检索，但熟悉名字组合也未恢复原单实体高分；它只是分层描述，非独立配对归因。

## 结果对研究的约束

- 压缩不是稳定胜出：小预算统计略好，更大预算nearest held反而更好。
- 完整ridge train不足说明线性code回归也有表示/函数容量限制，不能把所有失败归因丢掉历史。
- 完整nearest能背住train却held不到一半，显示既有E不能稳健映射这种组合表面；也可能受未见组合/长输入/新名字共同影响。
- 这不是‘模型完全没希望’，而是**此前24输入/单实体的高分不能外推**。
- 随机pair label没有可推断规律，未观察过的pair不应被要求推理正确；本held是同已见pair换表面，不是无反馈新pair预测。

## 验证与证据

每pair表面编码distinct断言（不证明feature唯一或可分）、预算检查、底座hash不变/M-off零；reservoir单测覆盖完整保存自检索、同seed确定、payload和非法输入不改state。全仓库192 tests OK（9.153秒CPU），all_tests_pair_capacity.log。原始所有pair/label、各stage全部prediction/target、method storage和checkpoint来源见pair_capacity128.json。运行后源码docstring更正了未证明的顺序不变表述，不改变运算或回写历史JSON。

全评估feature一次性缓存在研究进程中；缓存不是任一方法的记忆预算，不用于fit/检索库，不能将整个进程称等字节。没有迁移到语言模型的成绩。

## 决策

停止围绕原小颜色模型堆更多参数搜索。下一阶段需：

1. 使用分布内、具有可学习关系规律的任务，与随机绑定容量分开；
2. 逐项控制组合、长度、名字新颖性，不能一次全变后单因归因；
3. 确立真实文本/基本语言能力的下一层验证与训练预算；
4. 工作记忆保留简单检索强基线；只有神经路径带来实际迁移益处才整合。
