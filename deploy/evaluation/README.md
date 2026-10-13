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

上述历史基线与去重结果对应`45735c8`及此前的应用版本。准备配置`body-only-v1`保留用于复现旧数据格式；新版读取器会使用日期过滤，旧数据没有这些字段，因此不能期待用新版代码重跑得到相同基线分数。当前应用的日期/角色评测使用下述显式配置及报告中的代码指纹。

开发对照可复用已完成、无副作用的准备数据，但使用新会话和输出目录：

```bash
python3 tools/project.py bench --suite memory-quality --split development --limit 20 \
  --seed-from "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-v1" \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-compact-v1" \
  --output benchmarks/results/lcvmem-development-compact.json
```

复用前逐题检查来源状态未改变，且仅允许开发分区。一次回答运行中改变应用代码会导致运行失败。已开始但中断且没有完整结果的题目记录为错误，不悄悄重复计分。评分故障可单独运行`tools/lcvmem_score.py`读取已有结果，无需重新推理。

## 发言日期和说话者

`calendar-speaker-v1`准备配置将原始消息已有的发言日期、用户/助手角色写入Mem0元数据，问题和参考答案不参与该过程。普通对话中的显式保存、更正也会记录本轮接收时间对应的本地日期和用户角色。旧记录不会凭空补上日期；更正时的新来源未提供日期或角色，会清除旧来源的这些标签。

对于“2023年4月30日我分享了什么？”这类明确日期的历史对话问题，应用在Qdrant检索前同时施加主体、命名空间、发言日及可识别的说话者条件。未写年份时只筛选月日，证据保留原年份。历史会话记录也按本地日期范围读取，遵守原有保留期；模型提示标明用户原话与助手回应。日期解析放在线程中，避免冷启动阻塞语音事件循环。

该能力目前只处理单个明确的发言日期。相对日期、日期范围、事件发生日期及事实历史版本尚未实现；它们仍使用普通检索。没有来源日期的旧记忆不会进入按发言日期过滤的结果。标签和提示也不能保证模型正确理解来源，开发题中仍出现相关原文缺失、把描述误认成所问诗句的错误。

首次准备此配置时使用新的运行目录：

```bash
python3 tools/project.py bench --suite memory-quality --split development --limit 20 \
  --source-metadata calendar-speaker-v1 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-source-v1" \
  --output benchmarks/results/lcvmem-development-source.json
```

后续仅可从相同准备配置复用数据，仍使用新回答目录和新会话。20题的[日期/角色结果](../../benchmarks/results/lcvmem-development-source-final.json)词级F1为24.73，归一化完全匹配为0；含参考原文的检索结果为14题，实际注入为10题，没有执行错误、意外写入或跨主体证据。这次同时改变来源元数据、检索条件及提示表示，属于开发功能对照，不是单因素消融，也不能直接对比官方全上下文榜单。

`tools/memory_sources_console_smoke.py`另外通过公开`interact`入口、真实BGE/llama/Mem0/Qdrant验证保存标签、进程重启后按日期回忆、不同说话者、日期缺失与主体隔离。该脚本使用独立临时档案和合成内容，不计入独立语言质量验收。

## 本轮来源引用与提示预算

开发错例中，关于诗句的正确原文已排进检索第5项，却因512-token预算未能注入。现在普通回忆用本轮短引用表示记忆和来源，完整memory ID、版本、source ID及原文仍随`evidence_references`返回；会话依赖继续写入真实memory ID。不同的原始措辞、日期和说话者仍进入提示。更正/遗忘轮保留完整ID和版本，工具参数及存储主键不采用短引用。

同一准备数据、模型、top5与512-token预算下，20题[新基线](../../benchmarks/results/lcvmem-development-pre-refs.json)与[短引用对照](../../benchmarks/results/lcvmem-development-refs.json)的来源状态和检索顺序20/20相同。平均注入量由2.65增至4.7条，含参考原文的注入题数由10增至14；诗句错例现在取到了被挤掉的原文。词级F1本次为21.17→27.13，均无执行错误、意外写入或跨主体证据。

另修复“我跟你/我与你/我同你分享”把宾语误判为说话者的问题；同一问题同时包含双方发言时不施加单一说话者条件。[合并结果](../../benchmarks/results/lcvmem-development-refs-speaker.json)F1为27.05，改正1题的角色过滤，另外19题检索顺序不变。全部短引用均能映射回实际ID、版本和来源；校验与对照见[比较报告](../../benchmarks/results/lcvmem-reference-comparison.json)。198项框架回归及公开`interact`保存/回忆/更正/遗忘测试通过。

相同旧格式的这次重跑得分21.17，早前为24.73：20题检索、注入ID和输入token数相同，8题回答文本不同，差异原因尚未确定。官方词级评分还使用首行，额外措辞或列表格式会影响分数。因此当前数据只支持开发诊断，不把单次F1变化当成稳定语义正确率。

完整82题开发集使用新目录运行，原始内容继续留在AI_Stack，200题保留集不参与：

```bash
python3 tools/project.py bench --suite memory-quality --split development \
  --source-metadata calendar-speaker-v1 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-all-refs-v1" \
  --output benchmarks/results/lcvmem-development-all-refs.json
```

本轮[82题结果](../../benchmarks/results/lcvmem-development-all-refs.json)已完成：词级F1为26.33，归一化完全匹配1/82；含参考原文的检索题数58，实际注入57，平均注入4.80条。没有执行错误、意外写入或跨主体证据。它扩大了开发覆盖，仍不代表语义正确率达到JT-10要求；200题保留集未运行。

## 已确认的长条目缺口

`tools/long_memory_probe.py`保存8份合成档案，每份1058字符、3104字节，均通过当前工具参数校验并经真实MemoryAdapter/Mem0/BGE/Qdrant持久化，精确读回也完整。它们仅在尾部项目名和交付箱编号不同，但BGE输入被截为512 tokens，8份档案得到相同向量，语义top5仅命中5/8目标。完整证据需要1553个模型token，在512预算下全部被跳过。[失败基线](../../benchmarks/results/long-memory-boundary-baseline.json)验证的是存储/索引/装配边界，没有调用LLM回答，也不宣称公开自然语言保存已验证。

隔离候选复用[Qdrant原生多向量](https://qdrant.tech/documentation/manage-data/vectors/#multivectors)：同一Mem0记忆ID保留完整正文，BGE使用原分词器的原文偏移切片，多个向量由Qdrant MaxSim检索；选中片段保留原文位置。未新增reranker模型。8份同内容档案各5个向量，目标首位命中8/8，所需证据降为223 tokens，同ID更新和整点删除通过；Mem0实体索引也需要相同的矩阵结构，已显式验证已加载实体关联的检索与清除。

FP32多段SDK写入约0.75秒，超出目标500毫秒，因此用[Sentence Transformers的ARM64动态量化工具](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#quantizing-onnx-models)导出独立候选。该int8模型约24MB，原模型约95MB；相同8份档案仍首位命中8/8，SDK写入观测0.304～0.330秒，查询0.018～0.022秒。时间不含正式适配器journal等开销，也不是容量/整机验收。原模型和应用配置未更换，汇总及原始证据见[可行性报告](../../benchmarks/results/long-memory-feasibility-summary.json)。

```bash
.venv/bin/python tools/long_memory_probe.py
.venv/bin/python tools/quantize_bge_probe.py
.venv/bin/python tools/multivector_memory_probe.py \
  --embedding-path "$HOME/AI_Stack/models/bge-small-zh-v1.5-onnx-arm64-int8" \
  --output benchmarks/results/multivector-memory-int8-feasibility.json
```

首个多向量探针误用了SDK已忽略的`limit`参数，结果保留为`multivector-memory-initial.json`并标记INCOMPLETE；正式候选结果已改用`top_k=5`并检查返回上限。该隔离阶段没有资格声明生产生命周期或JT通过。

随后已接入正式适配器、双collection绑定、原文片段、重建迁移和备份恢复，新档案默认采用锁定的int8窗口配置，旧绑定按原profile读取直到显式升级。动态int8跨记录batch曾造成恢复重算不一致，现固定为单条记忆独立编码，原严格向量校验保持不变。[真实公开入口验证](../../benchmarks/results/window-legacy-backup-integration.json)通过长原文保存/跨会话回忆、更正、原ID与主体保持、旧单向量备份恢复及遗忘不复活；[迁移故障验证](../../benchmarks/results/window-reindex-recovery.json)覆盖三个阶段各10次SIGKILL。

新格式在检索结果中保留全文，仅将有原文位置与哈希的窗口送入提示。评分现检查实际可见的正文窗口；参考答案只出现在未展示部分时，不再计为已注入。新profile的82题开发评测使用独立目录`lcvmem-development-windows-v1`，结果另存`lcvmem-development-windows.json`，不混用旧索引数据，也不使用200题保留集。更广质量、真实10k完整路径时延和全部JT仍未完成。

后续先在开发集完善时间范围、对话来源与相关上下文检索，再冻结版本运行200题保留集。完整JT-10还须覆盖保存/更正/遗忘、无事实、取消、来源、重启和索引恢复；公开语言基准、词级F1以及框架回归均不能替代该验收。
