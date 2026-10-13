# 中文长期记忆质量评测

采用作者发布的[CLongEval](https://github.com/zexuanqiu/CLongEval)中的LCvMem小集，固定源码commit、Hugging Face数据revision和文件SHA256。源码位于`~/AI_Stack/frameworks/CLongEval`，原始数据位于`~/AI_Stack/datasets/CLongEval`；数据不复制到业务Git仓库。上游代码为MIT许可，许可证保留在外部checkout；数据仓库没有单独的dataset card/license字段，项目记录作者公开下载地址及校验值。

```bash
python3 tools/lcvmem_prepare.py
python3 tools/project.py bench --suite memory-quality --split development --limit 20 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-v1" \
  --output benchmarks/results/lcvmem-development-baseline.json
```

评分依赖单独安装在`~/AI_Stack/environments/memory-eval`，版本及归档哈希见`metrics.lock`，不改变助手运行环境。使用原始`metrics.py`的`qa_f1_zh_score`及其中文归一化；错误样例按零分计入分母，不采用上游脚本跳过部分API错误的逻辑。

冻结分区见`benchmarks/corpora/lcvmem-v1.json`：358题、70组对话，分为82题开发集、200题保留集及76题备用集。同一完整对话的所有问题保持在同一分区，并检查跨组不存在相同日期、角色及至少32字符的重复消息。先前检查过的首个示例固定留在开发集。保留集尚未用于调参或运行；没有审计模型预训练是否见过公开基准。

准备阶段仅将原始带日期、角色的消息作为评测授权保存的历史事件，通过固定Mem0 provider及真实BGE批量写入；不根据问题或参考答案选择、改写或总结来源。准备进程退出后，新进程通过受管理Qdrant读取持久数据，使用实际`AssistantSession`、llama.cpp、原有top5和512-token预算回答原始问题。每题独立owner及会话/笔记/提醒数据库，评测回答不会成为后续题目的输入。此流程评估检索和回答，不证明普通对话自动记忆提取已经实现。

参考答案仅在离线评分阶段读取。Git报告保留题目ID、词级F1、归一化完全匹配、错误/副作用计数和来源指纹；完整问答、检索内容和轨迹放在运行目录`answers/`，不随业务仓库发布。`reference_verbatim_*`只是参考原文是否出现的诊断信号，不是语义召回率或正确性裁判。时间记录是观测值，不是受控性能对照。

## 开发对照

20个开发问题的[基线](../../benchmarks/results/lcvmem-development-baseline.json)词级F1为12.92，归一化完全匹配为0；20题均完成，没有意外记忆/笔记/提醒写入或跨主体证据。12题的检索文本包含参考答案原文，但仅3题将相关文本纳入实际提示。

原`memory_message`重复注入正文及完全相同的source excerpt。现在只去除完全相同的摘录副本，保留source ID；原文与正文不同时仍完整保留。相同数据、检索顺序、模型及预算下，[对照结果](../../benchmarks/results/lcvmem-development-comparison.json)F1为16.18，含参考原文的注入样例为6题，仍无意外写入。词级F1会惩罚额外措辞，不等于语义正确率；实际检查也观察到日期相关信息未召回、把助手回应当成用户原话等问题，当前长对话回答质量仍有明显缺口。

开发对照可复用已完成、无副作用的准备数据，但使用新会话和输出目录：

```bash
python3 tools/project.py bench --suite memory-quality --split development --limit 20 \
  --seed-from "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-v1" \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-compact-v1" \
  --output benchmarks/results/lcvmem-development-compact.json
```

复用前逐题检查来源状态未改变，且仅允许开发分区。一次回答运行中改变应用代码会导致运行失败。已开始但中断且没有完整结果的题目记录为错误，不悄悄重复计分。评分故障可单独运行`tools/lcvmem_score.py`读取已有结果，无需重新推理。

后续先在开发集完善时间范围、对话来源与相关上下文检索，再冻结版本运行200题保留集。完整JT-10还须覆盖保存/更正/遗忘、无事实、取消、来源、重启和索引恢复；公开语言基准、词级F1以及框架回归均不能替代该验收。
