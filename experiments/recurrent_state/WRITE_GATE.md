# 同模型写门控干预：抑制旁路写入保留冷记录

本批固定slots600 learned模型，不训练、不改读路由/query网络、不查历史。对上一8192热写失败做同模型干预，而不是与不同训练的oracle架构直接归因。

```bash
python -m evaluation.write_gate_control
python -m evaluation.summarize_write_gate
```

## 预定三arm

初始8实体按0..7各写一次，使用原soft写规则；将同一primed h复制三份。随后热entity0被随机值覆盖8192次，相同seed与所有值共享。

- soft：原softmax gate更新所有槽。
- top1_mass：只保留最高权重slot及该原权重，其余gate置零。
- top1_unit：只更新最高权重slot且gate=1。

candidate GRU计算、原soft查询路由和读出网络不变。top1选择来自学到的route，不是直接用entity ID当slot；但route本身仍是已知ID的固定查表式关联。干预只针对priming后的热写，不能当作从头硬路由训练的模型。

## 8192后结果：三seed

| 热写gate | 冷实体平均 | 每seed冷实体 | 热实体平均 | 非hot-top-slot漂移 |
|---|---:|---|---:|---:|
| soft | 18.15% | 19.20/24.55/10.71% | 100% | 明显非零 |
| top1_mass | 98.36% | 100/95.09/100% | 100% | 0 |
| top1_unit | 98.36% | 100/95.09/100% | 100% | 0 |

稀疏gate将初始冷记录成绩保住，但seed1初始95.09%本来不满分，不称修复所有错误。soft完整曲线/逐流预测与先前skew8192 learned arm完全一致，summary逐行核对。

## 支持的因果范围

只抑制非top槽位写入而保留top原幅度，就明显改善该热ID压力场景；不需要把top门值提高到1才有效。这比不同模型对照更支持当前soft旁路更新对冷衰减有重要贡献。

仍不可说全部错误唯一来自泄漏：原soft读路由会混合多个slot，priming也soft，seed1路由碰撞存在，测试分布与训练不同。非hot-top区域不是被证明只存冷信息的区域。这里hot0在三个seed都没有冷entity共享其top槽，其他热ID未必如此。

## 必须保留的限制

- 仅一个预定热ID和升序priming，未知实体、动态分配、轮换热ID、多实体并行关系均未测。
- top1是test-time分布改动，未测均匀任务原性能是否损失，不能直接替代生产模型。
- 依旧计算所有slot candidate，只改变写入门，不证明省算力；需真正稀疏执行/延迟profile。
- 无自然文本编码/自动实体抽取，无PBSLM主模型集成、历史fallback或语义续连。
- 只有检查点完整prediction/target，非每个中间写事件都有probe；均值无显著性推断。
- 384bytes只是每流h payload，不含参数/评估truth/日志；参数前后hash一致。

独立实现单测保证soft逐tensor等于原observe、稀疏gate其他槽不变、非法mode拒绝；严格checkpoint及固定状态测试继续通过。全仓库203 tests OK（10.701秒CPU），all_tests_write_gate.log。权重artifact沿用已有文件与SHA，不重训。

下一步：对所有热ID做同权重筛查，按实际top-slot碰撞分层；再固定训练预算训练匹配稀疏写模型，并比较已知/未知/偏斜流，不能凭单一热ID成功宣布长期记忆解决。
