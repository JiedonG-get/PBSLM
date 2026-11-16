# 持续 Online 运行：第一批异步证据（固定权重）

用户澄清：PBSLM最原始目标是持续在线运行，Online Learning是支线。本批无训练、无参数更新、无M写入。

## 实现

- runtime/continuous.py：异步输入revision、候选输出单位、提交fence、取消、重规划续接、等待/说的显式policy、close与错误日志。
- 每次observe先原子增加revision，不等待推理；新generation request收到recent inputs和已提交prefix。
- 旧任务收到cancel；无论是否协作取消，每个输出unit在提交前检查revision，防止旧任务忽略cancel后泄漏。
- runtime/pbslm_backend.py：真实固定PBSLMv2推理通过专属单线程executor串行运行，不堵塞asyncio接收loop。取消future不抢占已运行推理，后续推理可能等待。
- 不新增参数，不改变checkpoint、O1白名单或模型forward。

## 验证分层

### 确定性场景（协议，不是神经理解）

第一条信息生成“已收到：”并等待，补充信息到达取消旧后缀，新的request保留prefix并收到全部最新输入，输出“根据补充信息，重新组织回答。”。

这证明输入与输出任务可交叠、prefix保留与stale fence。文字本身由脚本提供，不证明模型会自然语言续接。

### 真实模型（实际神经推理，不是语言生成）

```bash
python -m evaluation.continuous_online_demo --output experiments/continuous_online/fixed_state_demo.json
```

加载本地已保存text模型，无训练/写M。先等真实首个回答，再短间隔输入12条中文颜色问句，部分should_respond=False，最后resume。

- 首次demo.json：4次输入到达发生在真实推理线程的start/end区间内；最长observe调用约43.9微秒。
- fixed_state_demo.json：1次实际计算区间内输入重叠；最长observe约40.5微秒；权重/非M状态+M hash未变。
- 所有提交的generation_revision与当时输入revision一致，无过期提交。
- 时序是本机单次观测，线程调度不同导致重叠数不同；不是硬实时延迟SLO或基准结论。
- 日志包含input到达、generation start/cancel/commit与线程inference时间；perf_counter是本进程计时，不是UTC。

**真实模型能力限制**：backend收到recent event context和已提交prefix，但当前小模型只编码最新问句；只生成一个颜色token答案；.02秒是显式输出pacing，不是神经token解码速度。没有电影理解、持续语义状态或真正长输出续接证据。

## 测试

新增8项确定性asyncio测试，使用Event/Queue屏障而不是sleep决定pass/fail：

- 输出进行中新输入到达、prefix/latest context传入；
- 无响应policy仍持续接收输入；
- 多次interrupt/resume；
- backend忽略一次CancelledError时旧unit被fence拒绝；
- close后不再提交；
- backend错误明确记录；
- history与输出预算边界。

全仓库138 tests OK，6.665秒CPU，原始all_tests.log。

## 独立审计后的生命周期修复

- 显式aclose生成器，避免stale/budget退出后资源等到GC才回收。
- close超时后可再次排空，demo即使close异常也会请求executor shutdown；运行native推理仍不能硬抢占，shutdown可能等待。
- resume不再重复append最新输入。
- 未结束generation task设置上限4；达到上限仍接收输入、记录response_deferred，不再创建任务。没有自动重调度，需调用方resume，报告为背压限制而非无损服务。
- 新增生成器finally与close超时重试测试。全套140 tests通过，日志all_tests_after_audit.log。
- lifecycle_fixed_demo.json保留最新实际运行：0次落在推理区间的输入重叠，max observe服务时间约36.1微秒，no_stale_commits与模型/M不变通过。先前1次/4次观测均保留，不挑选运行；真实计时依赖调度，零重叠不证明持续理解，非零也不证明理解。

## 与最终目标的差距

- 现在是持续输入/输出调度与旧输出抑制，不是持续理解能力已经证明。
- should_respond由调用方明确给出，模型未学习什么时候说/静默。
- resume增加generation revision并重规划，但不重复追加输入，不挤掉真实观察历史；不是模型KV cache无损恢复或原token续算。
- 无运行中GPU kernel抢占、无实时audio/video输入、无线程安全/多进程/crash persistence。
- journal和committed历史无界，recent输入有界；不能称整个运行系统内存有界。
- 字符/字符串unit由backend定义；协议测试的流式字符串不冒充神经AR生成。

下一步应验证更长的真实生成、连续输入下的语义状态与响应policy；优先固定权重，把Online运行本身作为主线，而不是继续用学习/记忆任务代替它。
