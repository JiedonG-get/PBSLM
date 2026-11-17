# 固定权重长流：热实体更新污染冷记录

本批使用slots600实际保存权重，严格加载而非重训。无梯度、无参数更新、无原始历史查询/回退；不是PBSLM语言能力实验。

```bash
python -m evaluation.working_state_skew --batch 32 --output experiments/recurrent_state/skew8192.json
python -m evaluation.summarize_state_skew
```

## 固定流

每seed32流，先按entity0..7各写一个随机值；随后只更新entity0，值随机8类，其余7个entity不再写入。所有架构完全同priming值和热写序列；检查点0/1/8/32/128/512/2048/8192查询全部8entity，查询不改h。truth仅评分，不作为query输入。

hot固定0、priming顺序固定升序；因此这是一个预定压力场景，不是覆盖所有热ID/写入分布的普遍保证。与训练24步随机均匀写流存在显著分布/长度变化。

## 冷entity正确率：三seed均值

| 热写次数 | global | learned slots | explicit slots |
|---:|---:|---:|---:|
| 0 | 58.33% | 98.36% | 100% |
| 128 | 32.89% | 98.07% | 100% |
| 512 | 23.21% | 82.74% | 100% |
| 2048 | 16.96% | 50.00% | 100% |
| 8192 | 13.10% | **18.15%** | 100% |

learned8192每seed19.20%/24.55%/10.71%，热entity均100%；global热entity均82.29%。短均匀384步的高分不能推广到稀有记录长期保持。step0 global已较低，比较时必须保留基线起点。

## 诊断与限制

soft路由所有权重正，重复写热entity会更新其他slots。日志保存非hot最高权重slot之外的state漂移；这和冷信息衰减相容。但该漂移指标不代表所有那些slots只存冷实体，也未作固定同模型门控干预，不能将全部错误直接因果归到soft泄漏。

explicit onehot写入不改变其他slots，当前场景冷entity保持100%；但它依赖caller ID精确寻址，不是模型自动提取/分配新语言实体的证据，也不证明全任务无遗忘。

## 证据

- working_state_checkpoint.py支持旧v1研究格式的canonical默认维度；weights_only加载、有限tensor、严格keys/shape、固定配置与eval冻结检查。不支持任意未来结构。
- 每run记录文件SHA、训练seed/updates和参数前后hash；独立审查核对九个现有文件SHA与记录一致，参数hash与原slots600训练结果一致。不承诺跨设备bitwise重跑。
- 所有检查点保留每流8个预测/target ID、hot/cold计数、state漂移、最终reset对照。
- summary逐预测重新计数并核对aggregate，perseed/平均曲线均保留。检查点非独立样本，无显著性或安全性承诺。
- state_reset在最终只屏蔽query输入，不重写原state。模型工作状态无原始历史访问，评估者保存truth/probe日志。
- 本次没有计推理服务时间，不宣称整体省算力。状态96float不是全部模型/进程内存。
- 两项checkpoint严格恢复/坏配置测试与已有state测试通过；全仓库202 tests OK（10.292秒CPU），all_tests_skew.log。

## 研究决策

不能据短测试高分删除历史fallback。下一轮应固定同一learned模型，比较soft/hard/稀疏写门控（不先改读路由），检查冷保持与热正确性，明确测试时分布改动；再安排匹配训练的稀疏门控候选。长期持有还需动态实体寻址、冲突/未知query和语言接线验证。

该负结果不否定“内部状态优先”，它将问题缩小到选择性写入、干扰隔离与寻址可靠性，禁止把有限ID满分叫全息/通用记忆。
