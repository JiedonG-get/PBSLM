# 保守实体依赖路由基线

runtime/dependency_router.py给background接口增加一个明确可解释的参考策略。无训练、无神经推理、无M写入，不是学习型自主表达决策。

## 策略

- caller先声明当前speech依赖的非空实体集合（0..15）。
- 只有ExactReferenceExtractor明确接受的已知正向entity/color更新，且entity不在依赖集合，才走background。
- 相关entity、未知/模糊/否定/问题、未声明依赖均hard observe。
- hard observe后清掉依赖，下一回答必须重新声明，不能自动沿用旧依赖。
- 事件只在runner接受后计入routing journal；queue满不伪造已接受记录。
- parser与依赖不是可信真值：漏声明人物/实体关系会误把相关事件当背景。

## 验证

```bash
python -m unittest tests.test_dependency_router tests.test_background_input tests.test_deferred_response -v
```

9项定向测试通过，dependency_routing_regression.log。Event/Queue屏障验证：

- 甲查询正在生成，20条乙正向更新不取消输出，旧generation可继续取得进展。
- 甲改绿进入hard通道，旧generation后续不再提交，新generation接着保留prefix输出。
- 否定/问题/也许/陌生文本均hard；没有dependencies也hard。

## 研究边界

这只提供可证伪的relevance策略baseline，不证明模型判断语义相关。每个回答如何提取dependencies、跨实体关系、时态/假设、多模态事件都未解决。当前默认CLI没有启用该router，不偷偷改变既有交互语义。

依赖仅由caller设置/清理，当前无自动绑定generation生命周期；调用方需在新任务/回答结束时clear，不能长期沿用。背景输入更新独立state但active/pending回答仍使用原snapshot，语义时效性仍需另测。

后续应比较hard-all、明确依赖规则、学习型相关性三种policy在同一固定事件集上的输出进展和漏中断率，避免只优化输出量而放行过时判断。
