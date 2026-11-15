# 真实模型与 Runtime 的最小因果记忆接线

本批是同步token-ID机制验证。不是自然语言交互、并发GPU取消或完整Online Learning。

## 实现

runtime/model_session.py提供CausalModelSession，把真实eval模式PBSLMv2与RuntimeSession接起来：

1. observe(token IDs)记录输入时间/revision；advance调用真实model并捕获**观察时**E特征。
2. propose/commit提交JSON形式token ID文本；字符offset幂等，不是token级流式生成。commit不写M。
3. stage_feedback收到显式label后记录新事件；只生成候选，不写M。反馈必须不早于原观察。
4. confirm_memory才调用既有write_memory；重复candidate不重复应用；lr冲突重试报错。
5. 写入或确认失败时恢复之前M与协议状态；这是同步本地事务，不是崩溃一致性/跨进程事务。
6. snapshot保存协议、待确认E特征、数值M、应用回执；restore先验证再复制M，不重放外部输出/写入。
7. 模型权重/非M buffer、推理配置（含子模块res_scale/LayerNorm eps等）组成版本指纹；公开操作拒绝模型或外部M突变。
8. 回执记录lr、feature SHA256、M前后哈希与observation_id；该日志是审计记录，尚不是完整数值重放日志。

没有改训练脚本、O1六组白名单或模型state_dict结构。会话适配器只推理/显式M写入，无optimizer。

## 因果演示

```bash
python -m evaluation.runtime_memory_demo --seed 0 --output experiments/runtime_integration/demo.json
python -m evaluation.runtime_memory_demo --seed 1 --output experiments/runtime_integration/demo_seed1.json
```

每次演示先显式进行500episode的小型随机关联离线训练（同episodic_memory任务），随后清空M，再启动运行会话。

- 4次key观察→预测（在反馈前）→提交或打断→反馈到达→候选→确认M；
- 首次只提交1字符后打断，旧候选后续提交被拒绝；
- 每次确认后重复确认，验证只应用一次；
- JSON snapshot+restore后查询4条关联；
- seed0/seed1回忆均4/4，现场参数/非M buffer哈希不变；
- 4个确认版本；stale commit均受控拒绝。

这只是2个演示，不是统计实验。support答案在写入前已公开，因此是关联回忆，不是“未给答案模型就学会真值”。

### 证据版本说明

- demo.json是首次控制器版本的原始结果，保留未覆盖；它没有运行时源码哈希，也没有后补成“当时记录”。
- demo_seed1.json捕获了当时源码hash/软件环境/dirty状态；之后仅补充会话子模块配置指纹，更改的源码与记录hash可能不同。最终代码通过当前全套单测，但不能把早期demo称为当前源码逐字重跑。
- 参数hash只证明现场没有参数更新，不证明记忆内容真实；反馈provenance仍由调用方提供。

## 单测与独立复核

新增10项真实模型会话测试：commit不写M、反馈因果/幂等、故障恢复、待确认/已确认snapshot、打断恢复、外部M修改拒绝、错误模型/数值M恢复失败不破坏原M、相同权重不同配置拒绝、stale model下commit拒绝、冲突lr/损坏capture拒绝。

另有15项模型无关Runtime协议测试。

```bash
python -m unittest discover -s tests -v
```

最终 **88 tests OK，3.072秒CPU**；原始输出all_tests.log。包含原有O1/primitive/端到端参考与新协议测试。

独立静态审计发现并修复：版本只hash权重而忽略配置、部分公开操作无guard、确认回执缺lr与特征hash。没有把审计发现掩盖为“本来就已通过”。

## 未完成与风险

- 同步单进程、单样本会话；不保证线程安全、真正持续输入、GPU可取消或输出延迟。
- 每次_guard计算模型hash成本很高，只适合研究原型；生产版本需可信不可变version机制。
- 历史观察、候选和事件无容量上限；没有TTL/压缩。
- snapshot是可信本地数据，不是恶意快照安全边界；也没有鉴权、外部真值验证。
- JSON数值M/特征存储较大；未保存模型权重、CUDA RNG或KV cache。
- 没有在线参数更新与推理并发；适配器要求固定模型版本，参数更新应开新version/session。
- 没有数值M历史rollback、错值纠正策略、长期冲突/遗忘实验。
- 已确认的反馈可能仍是错误信息；当前确认只是程序协议，不是事实判定。

下一步：把保持性/记忆污染/有界容量与真实语言任务联系起来，继续报告负结果和预算；不能把本批演示称为完整系统完成。
