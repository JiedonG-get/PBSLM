# 冻结模型的新问句表面探针

本批不训练、不扩词表、不修改checkpoint。使用models800的3个已保存key-aligned模型，测试12种原训练/eval库存之外的人工问法。support仍使用原模板0/1和已公开颜色反馈。

```bash
python -m evaluation.text_surface_probe --seeds 0 1 2 --episodes 16 --output experiments/text_surfaces/frozen_probe.json
python -m evaluation.summarize_surface_probe
```

## 接口契约

新CompactTextCodec允许任意字符串进入BPE，但只接受原checkpoint紧凑词表内的token且长度不溢出；未知BPE ID、tokenizer真正unk、空输入或长度超限明确拒绝。不静默unk映射、不截断、不将拒绝算作正确。

接口“能编码”不表示问题属于本任务、更不表示模型有置信度。原ControlledTextSession的精确库存限制未改，新codec仅用于本次冻结权重探针。

## 三seed累计结果

每个split=3seed×16episode×4实体×12问法=2304次尝试；1920次能编码，覆盖83.33%。两个问法含原词表外子词，拒绝原因均为oov_bpe_ids。

| split/M | 可接收问题正确率 | 正确数/全部尝试 |
|---|---:|---:|
| 熟悉实体/正常M | 98.18% | 81.81% |
| 熟悉实体/零M | 14.74% | 12.28% |
| 熟悉实体/错值M（真实答案） | 0.36% | 0.30% |
| 新实体/正常M | 83.59% | 69.66% |
| 新实体/零M | 13.75% | 11.46% |
| 新实体/错值M（真实答案） | 5.42% | 4.51% |

完整每模板结果summary.json；逐问题文本、BPE编码、target/prediction和拒绝详情frozen_probe.json。评估非M state哈希未变。

## 可以得出的窄结论

- 已训练模型不只支持原96个精确表面字符串：在新人工问法且BPE可表示的子集上，正常M仍有较高回忆率。
- 语序/前缀变化不必重新训练就能部分迁移，但新实体组合仍低于熟悉实体。
- 词表契约限制了覆盖，不能只展示条件正确率；这不是完整自然语言能力。
- 错值干预使真实答案正确率下降，支持模型使用M关联，同时保留错误反馈风险。

## 限制

- 只有12个人工问法，都是同一个设备颜色任务，词汇/实体库存极小；不是真实开放式释义benchmark。
- 这些问法是在前期结果后人工拟定，非预注册测试，可能存在设计者选择偏差；没有挑选seed/删除失败问法。
- eval查询实体先前support已经公开正确颜色，是关联回忆，不是未给反馈的事实推理。
- compact词表在原训练时已预配置包含原eval库存；本探针只接受其中token。没有新增token、句法训练或模型学习。
- 同episode的12问法高度相关；这里是micro汇总，无独立iid显著性或语言泛化声明。
- 结果不与前一批held模板分数直接比较优劣：问法集合、episode seed不同。
- 无正常dictionary/Transformer同预算baseline，也没有多轮状态管理/并发运行。

下一步：将覆盖检测与明确拒绝作为文本adapter契约，再设计非模板的事实陈述与实体抽取训练；不能通过偷偷扩词表或改输入来制造高分。
