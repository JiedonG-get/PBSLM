# 持续文本对照：开发契约 v0

保留既有公开开发样例 `evaluation/fixtures/online_stories_dev.jsonl`（Git blob `4c7228642a19d83764a3f7cdc0c9196974a5e4e8`），不把熟悉样例当测试集。另建未用于提示词或政策调节的测试事件集；在运行对照前固定文本、注释规范和事件/输出时序，并发布其来源和哈希。公开样例只验收接线和评分。

## 同流对照

同一固定权重语言后端、解码配置和外部输入时序；对照为重读已到达历史、显式状态/检索、候选有界状态。各臂只能看到当时已到达的输入，以及自己的已提交表达。记录所用提示和状态载荷、外部证据读取与维护次数、输出单位时间和算力/内存测量范围。若候选用额外身份或事实字段，对照必须拿到同等辅助，否则单独标为有辅助诊断。应包含持续无关输入、表达中途的关键纠正，以及过时断言曾经提交和未曾提交两类情形。受控屏障验证协议另计，不作为实测时延。

先做语言后端预检：CPU 或单 GPU、本地固定版本、不训练，先核验权重及运行依赖，再做短问答与更正烟测。当前检查中 Qwen2.5-1.5B 的缓存快照只有 tokenizer 文件，Python 未装 transformers/llama_cpp；未验证其他位置的完整权重。本批不下载、不产生模型成绩。预检应报告无法理解任务的失败，不能改用规则答题再当语言成绩。

## 评分边界

`evaluation/online_story_trace.py` 对可信 harness 记录的输入与提交顺序评分；时间相同时按记录顺序。断言标签由独立标注方提供，不采用模型自报标签。先运行此脚本检查按到达时点评分的契约，另保留原逐事件预测评分。错误断言计数包含从未为真的答案；没有断言的片段单列，不默认正确。首次目标与每次目标变化分别记录首次正确断言延迟，未答与被后续目标取代分别报告。

当前评分器不能判定自然语言等价、是否明确撤回此前错误、正确前缀是否保留、是否发生物理取消，也不测输入排队延迟。下一版须额外记录标注的撤回/保留关系及可信运行轨迹，分别评分误/漏打断、语义修正与表达进展。合成单测仅验证计数，不是模型表现。

## Qwen3 固定后端与快照对照 v1

本地 Qwen3-1.7B BF16 可以通过 Transformers 离线加载。使用 checkpoint 的聊天模板、关闭 thinking、贪心解码。早期 smoke 的调度后端把消息传成了 tuple，模板丢掉了文本，日志只有 3 个输入 token。修复后的[问答预检](../experiments/continuous_online/qwen_direct_preflight_v1.json)记录算术和计划更正答对、交接答错。[调度预检](../experiments/continuous_online/qwen_direct_scheduler_v1.json)记录更正在旧工作线程推理期间到达，以及仅提交更正后的答案。线程内已开始的推理自然结束，提交单位为一整条回答。

快照对照使用同一后端、提问模板和解码预算。每臂只收到同一条流中已经到达的观察，提问由测试夹具标明。各臂的证据如下。

| 臂 | 回答时提供的证据 | 每次观察的维护 |
|---|---|---|
| `history` | 全部已到达观察 | 保存原文 |
| `recent` | 最近 3 条观察 | 长度为 3 的队列 |
| `retrieval` | 对问题按字符二元组交集取前 3 条，同分优先较新，再按原始时序呈现 | 保存原文并缓存二元组 |
| `bounded_summary` | 最多 128 个字符的递归笔记 | 同一 Qwen 用旧笔记和当前观察生成笔记，每次最多 96 个生成 token |

提问预算为 24 个生成 token。递归笔记是一个使用额外推理的文本状态候选。其推理调用、传入消息、状态和截断均有日志，可和直接检索的读写成本一起检查。标签由离线评分器读取，使用严格归一化匹配，单列拒答和空回答。准确率只反映这些合成短答样例。计划输入时戳用于记录事件顺序，本脚本没有按墙钟回放，也没有测语义续接或自主开口。

六条合成流由独立于运行结果的编写者定稿，共 66 个事件、24 个查询。首轮运行前确认 fixture SHA-256 为 `2db90a8d31d2165131ef14a8a59fffea2bdb44a86810d6fd456c6c786f1928fe`。另保留两条已公开的开发流。评估不使用任何真实对话数据。

```bash
python -m evaluation.qwen_local_preflight --model-dir /path/to/Qwen3-1.7B --output tmp/preflight.json
python -m evaluation.qwen_scheduler_smoke --model-dir /path/to/Qwen3-1.7B --output tmp/scheduler.json
python -m evaluation.continuous_memory_compare --model-dir /path/to/Qwen3-1.7B --fixtures evaluation/fixtures/continuous_memory_eval_v1.jsonl --output tmp/predictions.json
python -m evaluation.score_continuous_memory --fixtures evaluation/fixtures/continuous_memory_eval_v1.jsonl --predictions tmp/predictions.json --output tmp/scores.json
```

本次使用 PyTorch 2.11.0+cu128、Transformers 4.51.0 和单 GPU。评估命令需要可用的本地模型文件与 CUDA 环境，脚本采用 `local_files_only=True`。预检在有错答时保存结果并返回非零状态。
