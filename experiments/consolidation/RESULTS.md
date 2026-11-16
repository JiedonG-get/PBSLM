# M→A 内化：第一项机制筛查

本轮是用户重新授权模型结构研究后的首项实验。主线仍为持续Online运行；这里的慢速内化是工作记忆/长期能力支线，不是在live Runtime中并发修改权重。

## 问题

M能读对当次新事实时，能否把读出迁入A，并在M关闭/清零后保持？是否能换问法使用？是否破坏旧能力？

## 配置

使用现有models800/seed0..2推理checkpoint，46-token compact BPE、d48、8block，已训练的文本关联模型，不是通用语言模型。每seed建立4个固定旧entity→color、再学习不重叠的4个新entity→color；旧/新实体都属于此前episodic训练的熟悉实体。当前8颜色随机排列后分为旧4/新4，**旧新答案类别也不重叠**，因此新阶段目标会强化类别偏移/压制旧标签；不能把遗忘全部归因于实体记忆容量或架构。后续须增加共享答案类别的对照。

学生：M始终关闭且零，200step旧事实建立；从同一旧事实student复制五arm：none、truthCE、memory-teacher硬标签CE、teacher+old回放、wrong-memory-teacher。每学习arm200 optimizer step，lr.002，O1六组白名单；shared/E/tok/code/route/C固定。refine=1部分白名单可能无实际梯度，不强求所有组改变。

教师：另从原始checkpoint拷贝（早于旧事实建立），所有参数冻结。support模板0分别写入正常/旋转错值M；query模板2/3预测top1缓存。student只看到缓存label，不能读教师M。teacher自身读错应单独记录，不能归因学生内化。

学生训练模板2/3，测试同表面2/3和held4/5；held是有限人工模板的窄迁移，不是语言泛化benchmark。support与teacher query分割不等于student没见过entity名字。

五arm当前sample/template序列相同；replay独立RNG，当前样本序列不被回放采样改变。replay每step额外一个旧例，即两倍examples，不是同FLOPs/目标加权对照。

## 首个seed结果（pilot200.json）

- 教师正常M：新事实query100%；错M：truth0%，跟随旋转值。
- 学新前：old同表面100%、held25%；new两者0%。
- truth/teacher：new同表面100%、held25%；old两者0%。
- teacher+replay：old同表面100%、held37.5%；new同表面100%、held12.5%。
- wrong teacher：new同表面truth0%，old同表面0%；held偶然37.5%不证明正确内化。

这支持“现有O1能把硬标签行为存入A”，同时暴露遗忘与held低分。教师正确时硬label与truthCE相同，两者同分不构成teacher内化算法优越证据。

## 三seed原预算复核（three_seed200.json）

| 分支 | new同表面 | new held | old同表面 | old held |
|---|---:|---:|---:|---:|
| 不学习 | 0% | 0% | 100% | 50% |
| truthCE | 95.83% | 33.33% | 4.17% | 4.17% |
| memory teacher硬标签 | 95.83% | 33.33% | 4.17% | 4.17% |
| teacher+旧回放 | 100% | 33.33% | 100% | 54.17% |
| wrong teacher | 0% | 20.83% | 8.33% | 8.33% |

正常teacher三seed均100%truth，错M teacher均0%truth。每项每seed仅8query，3seed描述均值不作显著性判断。回放改善旧事实保持，但当前预算没有改善new held均值；旧held微升并不构成统计泛化收益。错误teacher在同表面对其错误label的匹配率为100%/75%/100%（均值91.67%），truth0%，说明错误临时记忆可能固化进A。

## 实现验证与来源

isolation_tests.log包括学生use_mem=True或M非零时拒绝训练，以及一步O1更新仍保留protected参数、student M零和现有O1契约测试，共11 tests OK（1.603秒CPU）。实验每armassert frozen parameter hash一致和M零。

原pilot/three_seed进程启动时已有student.use_mem=False和终态M-zero断言；后续在train_steps入口新增提前拒绝memory访问的guard，不改训练数学。结果source hash记录各运行启动时源码，不回写旧JSON。

没有训练resume checkpoint、优化器/RNG快照或长期内化收益声明。此处protected_hash只验证非梯度named parameters，不是所有buffer/config或C输出bitwise不变的独立证据。teacher来自旧事实拟合前的稳定checkpoint，是外部稳定记忆teacher→学生的隔离实验，不是live同模型自主转移。旧基线held低，因此不能声称“损坏了原先高泛化能力”；只能报告观察到的旧同表面保持与held实际数值。

## 下一步

三seed复核后加入zero-memory teacher和soft logits KL，区分记忆提供的信号与既有teacher偏置；再测多批序列保持、原episodic读出能力。若只有同表面背诵且遗忘明显，下一结构候选应围绕稳定表示、可选择内化和隔离慢更新提出假设，不能仅扩大M或无限回放。
