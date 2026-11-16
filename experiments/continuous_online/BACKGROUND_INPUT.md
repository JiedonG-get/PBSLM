# 持续观察不必全部硬打断：显式背景输入接口

## 问题与选择

此前1000次持续输入压力测试发现：每条输入都使generation过期，导致几乎不表达。现保留默认observe的保守硬打断，新增observe_background(text,relevance_assertion=...)。

这是**caller声明与当前表达无关**的策略接口，不是模型自主相关性理解。没有用debounce隐瞒新事实；若caller判断错误，旧语义仍可能输出，不能把调度安全当事实安全。

## 新契约

- background输入仍增加observation revision、记录history并提交独立ingest队列；不增加输出失效边界，不取消当前generation。
- 当前generation处理原snapshot，不自动看到背景输入。下一次新生成/resume使用最新history。
- hard observe、interrupt、close仍使当前generation失效，旧task即使无视cancel也不能提交。
- pending输出意图在背景输入后保留；有空位时按原回答请求的输入快照启动，避免将background当成待回答问题。背景不自动融入active/pending回答；下一次hard observe或resume才用完整最新history重规划。没有无限待输出队列。
- output journal新增output_fence_revision与latest_observation_revision。background模式下generation_revision可以低于latest observation，合法条件是等于output fence；旧的“每次提交generation==最新observation”只适用于无background的历史实验。
- 默认CLI尚未暴露background，避免未经说明改变原行为。调用方必须用非空relevance_assertion说明原因；该字符串不是鉴权或正确性证明。

## 测试

```bash
python -m unittest tests.test_background_input tests.test_deferred_response tests.test_continuous_runner tests.test_continuous_state -v
```

24项定向测试通过，原始background_input_regression.log。新增4项：

1. Event/Queue屏障下100条背景输入持续接收，原generation三个单位继续完成；101条输入全部提交backend hook。
2. 背景后hard correction使旧输出失效，cancellation-resistant后端也不泄漏。
3. background通过独立worker更新已知符号state，无需触发说话。
4. 容量满时defer的问题不会被随后background替换，容量释放后回答原请求。独立审查发现并修复该目标覆盖问题。

无训练、模型推理质量或吞吐benchmark。这只证明策略接口能允许进展，不证明自动判断哪些电影画面/声音与当前表达无关。

## 后续必要工作

- 学习型或保守可解释relevance判断需要独立误判评估；不能自动将未知事件当背景。
- 在回答依赖实体/关系上建立dependency标记，相关更新硬打断、无关更新可延后；当前接口尚未自动推导依赖。
- 给输出标注语义快照和过时窗口，与真实语言后端一起验证自我修正；现在generation版本仅调度含义。
- 视频持续采样、语音打断/TTS、模型持续状态都未完成。
