# 已提交回答的纠正协议 v1

在固定 Qwen3-1.7B 和 `ContinuousRunner` 下，检验已提交回答之后的撤回与更正，并比较模型生成路线和零修订推理调用的模板路线。场景仅包含规则前端识别的中文样件位置。初始回答和每次修订均以整句为提交单位，所有输入安排在上一轮生成结束之后。

## 固定条件

- 两个开发案例验证直改与从未知到首次确认。正式集为 10 个不同实体的 44 个事件，每组评估 10 次初答、12 次撤回并更正、1 次首次确认和 21 个应保持沉默的事件。正式场景包括四个单次更正、重复投递、无关和同值观察、提示性传闻、从未知到确认、值反转及连续更正。两组样本的实体互不重叠，均采用同一规则语法。[固定开发与正式样本](../../evaluation/fixtures/postcommit_repair_contract_v1.py)
- 各组使用同一个规则前端和无界精确检索。收到相同事件 ID 与相同文本的重复投递时跳过写入，重复 ID 含不同文本时显式报错。同值、未确认和无关输入不改变当前查询的读出。读出改变后才安排修订。评分标签只读取于离线评分环节。
- 每组的初答提示、模型、单工作线程和解码条件相同。模型使用本地 BF16 权重、贪心解码和非思考模式，每次最多生成 48 token。固定代码和权重进行一次正式运行，再按相反组顺序完整复跑。

## 四组对照

| 组 | 已提交后的修订方式 |
| --- | --- |
| `direct_prefix` | 现行基线。向模型提供上一次对话的整个已提交前缀及当前检索证据，要求自然地明确撤回。 |
| `structured_prompt` | 给模型已发地点和当前确认地点两个字段，明确要求输出撤回或首次确认的固定句式。模型候选直接交给运行器。 |
| `guarded_commit` | 使用同一结构化提示。已发地点从实际提交的最近一条回答识别，交付时重新检查证据和已提交条数。候选须匹配本次转换的规范文本，否则由固定模板替代。 |
| `deterministic_template` | 使用相同的已发值提取、证据和提交校验，直接输出固定模板。初答仍由 Qwen 生成。 |

校验和模板只能安全处理单独的地点名、明确的「不确定」或本实验固定修订句式。无法识别旧回答、已发值与当前值一致或缺乏确认值时，校验组和模板组均保持沉默。若这些条件出现在正式样本的必修订事件中，计入失败并单列原因。轮次等待验证的是提交后顺序处理，过时快照拒绝另有可控单元测试覆盖。

## 判据与复现

- 每组报告初答正确数、12 次严格撤回正确数、首次确认结果、21 次静默结果、日志验证数和生成错误数。撤回须给出规范句式及正确的新旧地点。评分允许 Unicode NFKC 等价及末尾句号差异；句式之外的语义撤回可结合原文复核。首次确认呼应预设未知状态，接受单独的新地点或规范的首次确认句式。
- 通过门槛为 12/12 次严格撤回、1/1 次首次确认、21/21 次静默、10/10 个提交历史一致的会话、0 个生成错误。整个样本作为评分分母。初答和先前已发值的识别准确率单独报告。初答或读出问题导致未生成修订仍计入失败。
- 报告校验接受、模板兜底和无法处理的次数，以及修订模型调用数、输入与输出 token、工作线程服务时间及处理时间。两种校验路线共享语义判据。若模型路线与固定模板的严格结果相同，且没有额外的实验可验证收益，则模板为这类有限格式的实施候选。正式复跑对照提示、候选、路由、交付、提交及日志，忽略已知计时字段。[实验驱动](../../evaluation/postcommit_repair_contract.py) [离线评分与复跑检查](../../evaluation/score_postcommit_repair_contract.py)

```bash
python -m evaluation.postcommit_repair_contract --model-dir /path/to/Qwen3-1.7B --split formal --output experiments/continuous_online/postcommit_contract_formal_v1.json
python -m evaluation.postcommit_repair_contract --model-dir /path/to/Qwen3-1.7B --split formal --reverse-arms --output experiments/continuous_online/postcommit_contract_repeat_v1.json
python -m evaluation.score_postcommit_repair_contract --input experiments/continuous_online/postcommit_contract_formal_v1.json --replay experiments/continuous_online/postcommit_contract_repeat_v1.json --output experiments/continuous_online/postcommit_contract_scores_v1.json
```
