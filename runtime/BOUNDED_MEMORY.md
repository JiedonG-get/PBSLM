# 有界反馈记忆与 Runtime 接线

实现：`runtime/bounded_memory.py`、`runtime/bounded_session.py`。

## 范围

这是外部latest-confirmed-feedback ledger，对调用方给出的语义槽位（如设备ID/颜色）执行替换、淘汰与重建；不是矩阵M自身学会覆盖，不是模型自动实体抽取，不是事实真实性验证。

- `BoundedFeedbackMemory`：容量正整数；update-recency LRU或插入顺序FIFO；保存detach的观察时E特征与已确认label。
- 更新同slot时旧贡献移除，重建当前保留记录；每次最多capacity次write_memory，成本O(capacity)。
- 删除与失败回滚；event_id幂等，payload冲突报错；audit回执复制给调用方。
- snapshot/restore重建M并验证数值一致，再原子安装；权重/配置指纹必须匹配。
- ledger有界，但audit/idempotency历史暂时无界；不能称整个系统内存有界。
- 新controller要求zero M；恢复已有M需使用snapshot；不静默抹去已有记忆。

## 因果Runtime接线

`BoundedCausalModelSession`单独继承原适配器，原append-only控制器/快照保留不改：

1. observe/advance捕获E keys并预测；
2. stage_feedback(..., slot=...)在反馈到达后生成候选，绑定slot到protocol证据；
3. confirm_memory才应用ledger替换与数值M重建；
4. Runtime确认失败时连同ledger、M、protocol恢复，候选可重试；
5. 重复candidate不重复应用，固定lr冲突报错；
6. snapshot包含base和controller，两者的M与确认历史必须一致；restore失败恢复原M。

slot由调用方明确提供。错误反馈仍可被确认；本层没有事实验证、自动冲突裁决或自然语言语义匹配。

## 验证

```bash
python -m unittest tests.test_bounded_memory tests.test_bounded_session -v
```

9+5项测试通过：替换等于独立最新值重建、LRU/FIFO淘汰、删除、重复/冲突event、写入NaN/异常回滚、配置/外部M变更guard、数值快照重建、Runtime确认失败跨历史回滚、恢复后继续纠正、拒绝候选不写M。

这些测试只验证协议与数值作用，不证明模型一定读对保留内容。持续干扰实验已证明M在更大负载下仍会出错，见 `experiments/persistent_memory/RESULTS.md`。

## 独立审计后的加固

- 失败回滚直接恢复旧M张量与控制器逻辑状态，不重入可能已损坏的write_memory；覆盖写成功而确认失败、rollback writer也故障的组合。
- restore先纯逻辑重放记录验证slot、顺序、payload和回执，防止数值M相同但slot标签/顺序被破坏的快照被接受。
- 有界会话核对base/ledger回执与确认payload一致；非默认lr使用配置默认值，冲突显式报错。
- 独立ledger restore允许allclose重建，但当前会话要求与base快照数值hash精确一致；因此**会话restore只承诺可重建为bitwise一致的本地执行环境，不承诺跨设备/不同数值内核迁移**。允许容差的跨设备恢复需要后续独立规范。

## 未完成

- 多进程/线程锁、GPU取消、崩溃一致性、外部存储持久化；
- audit/候选/观察历史的TTL与压缩；
- 模型自动抽取slot、引用/证据鉴权；
- 等预算检索baseline与真实文本流上的纠正收益；
- 数值M历史任意版本rollback（本次仅失败原子回滚与snapshot恢复）；
- 低延迟：当前频繁hash权重只适合研究。
