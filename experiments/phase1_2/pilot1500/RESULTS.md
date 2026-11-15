# Phase 1.2 同初始化码本：1,500-step 配对预实验

这是新的短程诊断，不替代Phase1 15,000-step原始A1/A2，也不是完整码本结论。

## 对照与预算

以已准备的ctrl_codebook_trainable/frozen配置为来源，保持结构、输入初始化、单位码本初始化、数据顺序、负样本seed、优化器/学习率、训练步数一致。

唯一实验处理：输出码本是否requires_grad。

- seed=1234，两模型初始state_dict逐tensor数值完全相同；SHA256 `aabd26588df480b7f9c590e9d3e44c747f3f826594f9c09bed1da0b210ae783e`。
- 总参数均21,160,328；可训练参数21,094,400 vs16,900,096。差额是处理本身，不新增adapter凑参数量。
- 1,500步、batch8、AR8、n_in16、d256、k8、depth2；AdamW lr=.001；sampled-negative InfoNCE 128负例/tau.1。
- 使用未修改的training.train_v21入口。训练用已有zhwiki BPE数据，15%尾部held；训练数据和checkpoint SHA256见artifacts.json。
- 训练时512-target快速eval只做监控。主报告用Phase1.1.1固定10,000 windows×AR8=80,000 targets、相同消融与频率桶。

## 可重跑命令

```bash
python -m evaluation.run_codebook_pair --steps 1500 --output-dir experiments/phase1_2/pilot1500
python -m evaluation.eval_codebook_pilot
```

run_codebook_pair要求全新目录，防止append旧metrics或覆盖研究；eval脚本使用该pilot固定路径。现有目录不能原地重跑，要指定新目录并相应调整eval_config/命令。checkpoint为本地忽略文件；GitHub只上传配置、指标、日志和hash。

## 固定 held 结果

| 指标 | trainable | frozen |
|---|---:|---:|
| Top1 | 5.3625% | 7.4375% |
| Top5 | 12.9462% | 13.5188% |
| high n=9,651 Top1 | 44.4514% | 55.7455% |
| mid n=10,774 Top1 | 0% | 5.2905% |
| low n=32,154 Top1 | 0% | 0% |
| rare n=27,421 Top1 | 0% | 0% |

完整Top5/上下文消融/敏感性见eval_trainable.json与eval_frozen.json。

成本：训练trainable67.89秒、frozen61.54秒；9变体评估104.9秒、106.7秒。单RTX4060 Laptop GPU；一组配对、一个seed，顺序运行，不能把耗时差直接作为性能结论。

## 解释

这个同初始化短程对照中frozen码本Top1/Top5较高；初始化差异这一混杂已消除。它仍不能证明固定码本在收敛后更好：只跑1,500步、单seed、无显著性检验，且可训练参数数不同。

低频/稀有Top1仍为零；较高整体分数主要来自频繁token，不能据此说语言能力已解决。

后续若用于选择正式码本，需要完整预算、多seed配对与性能/频率分层报告。本实验仅支持优先保留frozen作为小型机制实验的稳定目标空间；不是最终架构选择。

原始Phase1 logs/metrics没有覆盖。本预实验独立放在phase1_2/pilot1500；没有把O1.5合成记忆任务与自然语言baseline混为一谈。
