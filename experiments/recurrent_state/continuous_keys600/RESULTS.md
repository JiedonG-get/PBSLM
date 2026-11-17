# 连续键工作状态：去掉固定ID后的首个负结果

## 新目标

持续Online主线不变：内部工作状态优先、必要时再查记录；不研究CNN/逆傅立叶支线。本批只筛查固定实体ID之外的寻址难度，不修改PBSLM主模型，也不删除原历史数据。

## 预定任务与预算

```bash
python -m evaluation.continuous_key_control --updates 600 --seeds 0 1 2 --output-dir experiments/recurrent_state/continuous_keys600
python -m unittest tests.test_continuous_key_state tests.test_recurrent_working_state -v
```

每episode新采样8个16维归一化随机键，随机写入最新8类值。模型observe仅得到当前key/value及旧h，query仅h/key，没有实体整数index/已见mask/旧历史。训练32流×24步×600updates，AdamW.002，global GRU128float状态。评估参数固定，64流，长度24/96，query噪声0/.05，query-reset对照与正常组随机序列配对。

查询75%从全部8键均匀抽样（包含尚未写过的键），25%新生成缺席键。未写键和缺席键都应回答UNKNOWN第9类；因此模型不能只靠已见query选择辅助。噪声也加入每条query，标准差.05为固定预设，阈值不按测试调整。

## 基线与代价

外部key-value bank保存实际已观察键、最新值和seen标记，以cosine>=.9作为已知判定。实现预分配全部8随机键但seen mask禁止读取未写条目；未写键内容不应计成模型知识。这个baseline是强显式存储，不是神经学习。

128float h=512bytes/流。bank数值布局为8×16float32键＋8int64值＋8bool=584bytes，仅tensor payload；模型参数/运算/Python/评价缓存均不计入，不能宣称全资源公平或省算力。训练BPTT存激活，评价保存各步loss/统计，所以整个研究脚本不是常数内存。无live Runtime接线。

## 首批结果

96步无噪声已知值回忆perseed：2.97%、7.81%、4.72%（均值5.17%）；unknown判对率92.05%、83.41%、88.02%（均值87.83%）。24步已知1.42%、unknown98.04%；96步query-reset已知0%、unknown100%。银行参考同条件均100%。没有保存错误预测类别混淆矩阵，因此不能仅从known错误严格区分错值与误报unknown。

参考实现复用评价者truth矩阵作为最新已观察值的存储，选择只由query/key相似度与seen mask决定；未将当前query target作为检索答案。它是向量化的观察存储基线，不是完全独立实现的第二套事实账本。

模型主要倾向拒答，不能把unknown高分与known低分平均成‘成功记忆’。这里known准确率对照8个有效值并不等于随机9类整体基线；分开报告，全部数在results.json。

这不证明连续键状态方案不可行：仅测试一个globalGRU、有限600更新预算，未训练分槽连续内容寻址或调整loss平衡。它证明旧8ID高分不能外推，动态key→值绑定需要新机制/训练，不能依赖静态ID查表。

## 产物与验证

研究权重local ignored，文件SHA及state_dict SHA已保存。运行后发现模型源码被替换为不兼容接口，随后恢复为实际训练版本；恢复源码SHA09d701c1a3a9ff59e2fc30b805cf7aed52049207013be6102b1c1bfe491517a7与训练capture完全一致，3份权重在恢复代码上原始strict加载全部匹配。没有重训或回写结果。

三项定向测试通过：无ID接口、query不改h、update/read梯度及旧基线契约。query-reset屏蔽当前read h，不改变观察更新轨迹。无逐事件预测，仅known/unknown汇总与参数固定hash，不伪称逐次神经输出可重算。

## 下一步

先比较动态连续key的显式/学习型分槽与global，控制状态容量、写入匹配与未知拒答。显式键存储本身已足够解任务，应保留为强基线。随后才考虑文本E编码、噪声和未知身份泛化；不能在随机向量任务成功后跳称自然语言持续理解。
