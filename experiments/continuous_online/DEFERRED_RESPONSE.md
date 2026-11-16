# 输出任务背压后的自动续调度

先前runner达到max_generation_tasks后只记录response_deferred，必须外部resume。现在只保存一个pending generation revision；旧任务done回调在容量释放时用最新inputs和已提交prefix重新调度。

## 契约

- 最新回答意图覆盖旧pending，不无限排队。
- input仍进入独立处理队列，输出pending不是丢弃输入。
- should_respond=False、interrupt、close清除pending；不会恢复过期表达。
- done callback没有await，检查revision与调度在单事件loop中完成。
- 生成错误不会无限自动重试；只有此前因容量被defer的最新请求会自动运行。
- 原生推理不能抢占，卡死后端仍可能永远占容量；自动恢复不承诺硬实时。
- CLI /wait同时识别active和pending，但仍有10秒等待上限。

## 验证

新增3项Event屏障测试：非协作旧generation占容量期间到达middle/newest，释放后仅newest运行；静默新输入与显式interrupt分别清掉pending。无sleep作为正确性依据。

定向17项测试通过：deferred3、runner10、worker2、CLI2，输出deferred_response_regression.log。修复不训练、不调用模型，不证明自主说话policy或自然语义续连。

这项新行为取代早期RESULTS.md所述“defer后需手动resume”；历史结果JSON保留，不补造重跑数据。
