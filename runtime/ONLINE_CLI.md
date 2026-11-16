# 持续 Online 终端原型

首要目的：体验输入/输出重叠、静默处理、打断与按新信息重规划；固定权重，不训练。

从仓库根目录：

```bash
python -m runtime.online_cli --backend reference
```

示例输入（每行按回车）：

```text
/listen 甲号设备的颜色是红。
甲号设备的颜色是什么？
把甲号设备的颜色更新为绿。
甲号设备的颜色是什么？
/interrupt
/resume
/quit
```

输出尚未完成时可继续输入。`/listen TEXT`接收且处理但不要求输出；`/interrupt`停止当前未提交输出；`/resume`按最新输入/已提交prefix重规划，不重复追加历史。已说的内容不会撤回，只阻止后续旧单位提交。

`/wait`是管道/测试用屏障，等待当前输入处理或生成完成；普通交互不需要它。不是模型自主“决定等一等”。

## 后端必须区分

| backend | 实际能力 | 不代表 |
|---|---|---|
| reference | 精确已知中文句法→符号state→规则renderer | 神经模型理解/生成 |
| reader | 固定小GRU提取→符号state→规则renderer | PBSLM主干语言能力/事实验证 |
| pbslm-query | 本地PBSLM checkpoint真实推理，一次一个颜色token | 长期state/流畅长文本续接 |

```bash
python -m runtime.online_cli --backend reader --reader experiments/fact_reader/three_seed600/seed0.pt
python -m runtime.online_cli --backend pbslm-query --checkpoint experiments/text_runtime/models800/seed0.pt
```

后两者需要本地ignored模型，GitHub不包含权重；reference不需要checkpoint。reader已知会误接受未见否定/时态句式，只应用于不可信ephemeral state，不写数值M。

可选 `--trace path.json`保存运行事件与backend轨迹，必须新路径；`--delay .15`是显式展示pacing，不是模型速度benchmark。

## 生命周期与限制

- stdin阅读与展示任务分离，观察输入不等待推理；executor串行，不会并发调用模型。
- 输出commit受revision fence保护，生成任务可取消，但native线程/GPU计算不可硬抢占。
- native推理挂死时shutdown仍可能等待；无硬实时或崩溃安全承诺。
- terminal是逐行输入，不是语音流；stdout展示单位不是TTS。
- should_respond是命令/caller显式指定，未学习自主说话policy。
- journal/committed输出并非无限有界；退出应使用/quit，Ctrl-C等异常流程尚未全面平台验证。
- 本入口方便体验机制，不是成品UI，不替代自动化时序测试。
