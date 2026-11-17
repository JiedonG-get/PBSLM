# 同96float：全局状态与学习型分槽状态

本批为独立符号机制实验，原PBSLM模型/Runtime未改。Offline训练、评估参数固定；查询只能看h+entity ID，没有原始事件历史访问。没有历史fallback或全息记忆结论。

## 设计

```bash
python -m evaluation.slotted_state_control --seeds 0 1 2 --updates 600 --output-dir experiments/recurrent_state/slots600
python -m evaluation.summarize_slotted_state
```

- global：96维GRU全局状态，54,824参数。
- learned：8×12维slot，共96float；entity embedding与可学习slot key的softmax分布用于写入混合/查询读出。路由只由有限entity ID决定，是静态学到的ID→slot映射，不根据state/内容动态分配。共享GRUCell预测每slot candidate，通过路由加权更新；4,976参数。
- explicit：相同8×12状态，使用输入entity ID one-hot路由，强辅助基线；4,976名义参数但其中keys不参与前向、无梯度，因此有效参数路径并不相同。

每架构600update×batch32×24步，AdamW lr.002，完全相同独立CPU generator输入/值/query流。架构初始化函数不同，不声称同初始化或同FLOPs。global本批复跑结果与旧pilot600逐项相同。

随机entity/value写入，查询已见entity最新value；8个已知槽位/8类，所有ID训练可见。测试24/96/384步，每长度64流。同seed每模型和reset-control使用相同事件流。

## 三seed均值

| 架构 | 24步 | 96步 | 384步 | 384步query屏蔽h |
|---|---:|---:|---:|---:|
| global | 65.32% | 56.71% | 54.68% | 12.75% |
| learned slots | 98.16% | 97.23% | **96.96%** | 12.94% |
| explicit slots | 100% | 100% | 100% | 12.91% |

learned384每seed100%/90.87%/100%，不是所有seed满分。对距最近写入16步以上，global31.72%、learned93.57%、explicit100%（pooled年龄桶）。所有模型评估前后state_dict hash不变。

learned top-slot分配distinct数为8/7/8；seed1两个entity主要使用相同slot，错误与寻址冲突相容，但未做路由干预因果实验，不能断言所有错误都源于该碰撞。

## 支持的判断

固定激活大小下，分离存储/选择性写入比当前globalGRU更适合该随机覆盖任务；全局混合状态的低分不是固定状态方案必然失败。学习型路由能在这个已知ID任务里学近似分槽，而无需查询原始历史。

## 不能扩大

- 模型见过全部8个entityID，学习型寻址可成为有限ID查表，不代表新实体、语言泛化、动态slot分配或电影理解。
- explicit把寻址问题交给caller，不能当神经模型自动理解实体的成绩。
- 状态都是96float32=384bytes/流；参数、候选slot计算、临时激活、RNG与训练开销不同。learned训练+评估约32秒/seed，global20秒，不能据参数更少就声称省算力。
- 384步仍是8槽位反复覆盖，不是384独立事实容量；query只选已见ID，无未知拒答/置信度校准/撤销任务。
- query-reset只屏蔽当前读入h，观察状态仍正常更新；BPTT训练保留24步激活，评估stream函数仍保存逐步loss标量，不是整个进程常数内存。
- explicit仍计算全部slot candidate再onehot选择，不是最省计算实现；未做更新稀疏化/延迟profile。
- 原始逐事件预测未保存，只有aggregate/age桶，不可独立重算每次预测；summary核对计数/分母一致。

## 产物与验证

本次训练保存local ignored .pt，results记录路径与state_dict hash，但尚未实现独立checkpoint loader/严格恢复测试；它们是研究训练产物，不承诺生产可恢复。未来不可用重训文件冒充本次权重。

四项模型测试通过：explicit未写slot bitwise不变、query不改h、learned routing/GRU有梯度、固定payload与global基线契约。全仓库200 tests OK（10.617秒CPU），slots600/all_tests.log。summary包含各seed、参数、时间与全部年龄桶。

## 下一步

先冻结本次权重，用长达数千步/偏斜写频率/反复热entity干扰稀有entity的流测试泄漏；soft routing每次微量写其他slot，短测试满分不保证长期不衰减。再测试未知ID/符号改名与明确动态分配，不依赖额外离散oracle掩盖泛化。只有这些机制通过，才把文本编码/E特征接入并研究历史fallback触发/成本；不立刻整合主模型。
