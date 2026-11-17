# 查询扰动与身份边界：精确恢复不等于可靠拒答

本批无训练，固定content_fast600 delta权重；同一组primed键和值随后128次hot0覆盖，然后仅改变query。两个几何条件：训练式随机球面键、两两cos.9相关键。3seed×32流×8query=每格768个查询。

```bash
python -m evaluation.fast_query_boundary
python -m evaluation.summarize_query_boundary
```

方法：原delta；保留观察键/值/dual/matrix的强双基参考（依旧通过原delta读出MLP）；存原键和值的cos>=.9最近邻。后两者增加原记录和结构先验，非等资源；未知判断未重新校准，不按test选阈值。

## 已知query正确率（原键加独立Gaussian扰动后归一化）

| geometry/noise | delta | dual | nearest .9 |
|---|---:|---:|---:|
| random/0 | 84.51% | 100% | 100% |
| random/.05 | 82.68% | 100% | 100% |
| random/.15 | 73.83% | 94.40% | 24.48% |
| correlated/0 | 22.27% | 100% | 100% |
| correlated/.05 | 22.27% | 99.22% | 99.87% |
| correlated/.15 | 22.27% | 69.40% | 24.61% |

## 未写入query误接受率（不是正确率，越低越好）

| geometry/unknown类型 | delta | dual | nearest .9 |
|---|---:|---:|---:|
| random/独立球面 | 8.72% | 41.80% | 0% |
| correlated/独立球面 | 0.91% | 90.23% | 0% |
| random/近键cos.95 | 83.59% | 99.61% | 100% |
| correlated/近键cos.95 | 100% | 96.88% | 100% |
| random/近键cos.99 | 85.94% | 100% | 100% |
| correlated/近键cos.99 | 100% | 100% | 100% |

## 必须区分的解释

- noisy-known被实验者指定‘仍是原身份’，near-unknown被指定‘新身份’；模型看到的都只是附近向量，没有身份/来源标记。分布可能重叠，不能要求相似度规则不借额外线索完美区分，也不能把误接受全归因网络失败。
- near-unknown不是实际语言同义/不同义样本；这里只是预定向量角度规则。近键通常不与其他存键完全相同，仍可能更接近另一键；结果记录max cosine及nearest record，不能按构造anchor当真实最近邻。
- cos.9 nearest拒随机未知良好，但强噪声known会低于门槛而被拒；这里没有更广阈值优化/独立验证校准，不能宣布最佳精度—覆盖。
- dual放大离已观察key span的读出不确定性，且原MLP未训练过dual分布；高误接受同时受几何/分布失配影响。不能把dual向量恢复好等同可信分类，也不能将分类问题全归因求逆数值错误。
- delta/correlated已知query很差，却对随机未知拒答好，不是安全记忆；nearunknown也失败。known wrong-value与false-unknown计数分别保存。
- 仅已知8key与hot0，训练/测试身份来自同随机键分布；无文本/动态扩容/多来源真实性/Online表达成绩。

## 验证与代价

query不改state、eval前后权重hash不变；原输入每个写入和三种方法共享，dual/bank只使用已观察键和值。target只计分，不选检索记录。所有预测、target、近键相似度/记录index均保存query_boundary.json，summary重新计数全部confusion，与aggregate核对一致。

几何/错误分区+dual覆盖契约共4项定向测试通过，query_boundary_tests.log。没有重新训练，不重复其他历史实验。文本只读复核不替代代码/数据独立执行审计。

没有时延/能耗/总体内存测量，参考方法显式保留key/value，不声称省算力。下一步应在独立验证数据上比较身份线索/明确key去噪协议，或构建有语义标签的输入；不能对无可观察身份区别的向量任务无限调置信度阈值。
