# 匹配稀疏写入训练与固定碰撞先验

按SPARSE_TRAINING_PLAN预定0/1/2、600update×batch32×24steps，AdamW lr.002。同96float激活、相同模型参数形状/初始化seed、相同CPU generator事件流。读规则始终soft；只改写入，第三arm加固定.1碰撞先验，未用test调参。

```bash
python -m evaluation.train_sparse_state --updates 600 --output-dir experiments/recurrent_state/sparse_train600
python -m evaluation.summarize_sparse_training
```

## 模型与先验

- soft：原learned slotted模型。
- sparse：train forward top1 unit gate、backward用soft路由直通代理；eval hard gate。直通不是离散选择真实梯度。
- sparse_collision：同sparse，加全部8个已知ID路由分布两两dot overlap均值×.1，鼓励少碰撞。

这项先验依赖已知8ID/8slot，不是动态新实体分配；仍计算全部slot candidate，不证明省算力。soft reader仍可读取多个slot，top-slot相同不是全部表示相同。

## 三seed均值

| arm | uniform384 | all-hot热实体 | all-hot冷实体 | 各seed distinct top slots |
|---|---:|---:|---:|---|
| soft | 96.96% | 99.48% | 45.76% | 8/7/8 |
| sparse | 99.63% | 100% | 92.63% | 7/5/5 |
| sparse+collision | 99.91% | 100% | 96.88% | 8/6/7 |

all-hot仍固定每case2048次只写单一hotID，16流×8hot×3seed；每casepriming改用该arm自身训练写规则，不同于上批test-time干预从共同soft priming出发。当前arm不同weights，只有训练数据/初始化/预算配对。

sparse冷保持perseed96.875/90.960/90.067%；加prior100/93.862/96.763%。三seed小样本描述，不作显著性。

## 未解决的失败

碰撞先验改善但没有强制一一寻址：seed1/2仍有top-slot碰撞。碰撞冷记录pooled：sparse28/224=12.5%，prior12/96=12.5%；非碰撞分别2462/2464≈99.92%、2592/2592=100%。两组碰撞样本集合不同，不能把12.5%相同解释为严格配对无变化。

均匀短流的接近100%依旧不能证明长期罕见记录可靠。有限ID软路由可用次级槽位区分信息，短流高分掩盖反复覆盖冲突。

## 边界与证据

- 全部已知符号ID、8值；不查询原始历史/真值表；query只h+entity，truth仅评分/已见query选择。
- 离线训练，eval no_grad/model.eval，参数hash不变；state activations更新。没有PBSLM主干/语言/Runtime集成。
- all-hot保存initial/final逐流预测，summary重新计数；uniform保存aggregate/age，不保存逐step输出。soft三seed最终权重hash与原slots600 learned模型完全一致，确认基线复现。
- 稀疏前向不改未选槽位、代理梯度、collision finite与已有state契约测试通过；全仓库205 tests OK（9.633秒CPU），sparse_train600/all_tests.log。
- 训练产物local ignored，当前研究v1没有独立loader/恢复预测验证；文件不上传，不冒充生产可加载接口。summary的prior_final实际是最后记录训练步在optimizer.step之前计算的prior，不是最终路由重新测量值。
- 时间包括训练+eval和同步开销，非推理延迟；参数/持久状态同形状不等于FLOPs相同。
- 预定explicit基线已有slots600结果，本次没有重训explicit或在这里重测其all-hot，不把旧hot0成绩当完整配对比较。

## 下一阶段判断

选择性写入是有效机制方向，但此已知ID任务还未解决碰撞，更不能减少真实语言fallback。后续需要明确：用强制唯一映射会增加任务先验，扩slot会增加容量，动态分配会增加状态；都必须单独比较与记账。停止在8ID短流上追满分，将可靠性/动态身份与可撤销工作状态作为下一验收。
