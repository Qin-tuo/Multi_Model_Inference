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

新格式在检索结果中保留全文，仅将有原文位置与哈希的窗口送入提示。评分现检查实际可见的正文窗口；参考答案只出现在未展示部分时，不再计为已注入。新profile的[82题开发评测](../../benchmarks/results/lcvmem-development-windows.json)已完成，使用独立目录`lcvmem-development-windows-v1`，没有混用旧索引数据或使用200题保留集。

| 开发集配置 | 词级F1 | 归一化完全匹配 | 检索包含参考原文 | 实际提示包含参考原文 | 执行错误/意外写入/跨主体证据 |
|---|---:|---:|---:|---:|---|
| 原FP32单向量，82题 | 26.33 | 1/82 | 58 | 57 | 0/0/0 |
| int8原文窗口，82题 | 26.69 | 1/82 | 58 | 56 | 0/0/0 |

本轮没有证明整体回答质量明显提升。新配置解决了已复现的长条目截断与装配问题，但当前词级F1与实际可见参考原文数量表明，广泛检索和回答质量仍有缺口。归一化完全匹配也不是人工语义判定；这些开发指标均不能替代JT-10。

## 当前格式的正式适配器容量测量

`tools/managed_memory_capacity.py`使用真实Mem0Adapter、受认证Qdrant及当前多向量绑定。准备阶段复用冻结的合成编号资料，10%的主人条目加入长前缀，核心要求位于末尾；正式计时调用完整查询/写入适配器，包含绑定检查、BGE、原文窗口及写入journal/history。

```bash
.venv/bin/python tools/managed_memory_capacity.py --phase prepare --tier 10000 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/managed-window-capacity-10k-v1" \
  --output benchmarks/results/managed-window-capacity-10k.json

# 准备完成后，停止其他评测/导入，再单独计时：
.venv/bin/python tools/managed_memory_capacity.py --phase measure --tier 10000 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/managed-window-capacity-10k-v1" \
  --output benchmarks/results/managed-window-capacity-10k.json
```

100次查询按长短目标各50次分层，50次完整写入按长短各25次分层；写入持久性在计时后另行核对。首次准备可续跑，已有测量结果不可覆盖。[本配置10k计时结果](../../benchmarks/results/managed-window-capacity-10k.json)查询P95为46.97ms，完整写入P95为342.79ms；长写入单列P95为349.82ms，短写入为51.07ms。查询/写入无执行错误、无主体泄漏，50次写入均确认持久化，所观测六个模型/视觉服务的PID和重启计数未变化。该脚本记录后端驻留状态，没有前台语音负载或逐帧视觉测量，不能独自完成JT-11共存验收。

这份报告的`PASS`仅针对声明的读写时延和持久性检查：**编号top5召回仅9/100，其中长目标0/50、短目标9/50**。这是大量相似编号资料的合成压力观测，不是独立语言质量，却明确暴露了检索缺口；不能用时延通过掩盖它。[只读诊断](../../benchmarks/results/numbered-window-recall-diagnostic.json)核对100条目标原文均完整，长目标0/50进入原生向量前60候选，短目标38/50进入候选。当前实际方法为`mem0_dense`，可选BM25未启用。开发诊断中将资料主题也写入问题，长/短top5分别为30/50、41/50；这不是正式修复，也不能要求用户用额外信息绕过编号查询。后续须基于成熟检索组件验证关键词与语义候选合并，并维持主体、来源、更正和遗忘约束。

### 原生关键词与融合检索候选

现有Qdrant1.19.2包含[服务器本地BM25](https://qdrant.tech/documentation/inference/inference-bm25/)；其[多语言分词选项](https://qdrant.tech/documentation/search/text-search/full-text-search/)可以与禁用词干化、空停用词集组合。已检查本机固定版本的`src/common/inference/local_model.rs`与`bm25_inference.rs`，并实际向认证的回环服务提交文本，未安装FastEmbed或调用外部推理服务。

`tools/qdrant_lexical_probe.py`把冻结的10000条主人和100条访客种子资料复制到有归属标记的临时原生collection，保留现有int8多向量不重新编码，再由服务器生成BM25稀疏向量。两个候选分支及融合结果均施加主人/命名空间条件。完整原文和编号问题不改变；分别运行BM25、原生RRF、原生DBSF。

[可行性结果](../../benchmarks/results/qdrant-lexical-numbered-feasibility.json)三种方式的编号top5均为100/100，长/短各50/50，无主体泄漏，临时collection已清理。这证明该合成缺口可由现有开源组件的词项候选补足，该隔离结果本身不验证正式Mem0检索或更正/遗忘/恢复生命周期；数据准备曾与该探针重叠，没有时延结论，也不是独立语义质量或JT通过。

集成前还须处理上游边界：当前固定Mem0的`_search_vector_store`只把语义前60结果送入最终评分，关键词结果只贡献分数；仅启用可选BM25不能保证补回语义候选之外的长条目。现已在单个Mem0 provider实例接入原生DBSF候选集合，稠密与BM25写入复用同一本机认证客户端，schema3绑定冻结词项配置；新增、更正、迁移和恢复均按规范正文生成BM25。实际保存/召回/旧备份恢复已通过[集成验证](../../benchmarks/results/hybrid-memory-integration.json)，217项框架测试通过；默认配置暂未切换，完整时延、重复故障与开发质量仍在验证。显式升级见[部署说明](../memory/README.md#本地关键词与语义融合候选)。

正式Mem0适配器的[万条候选计时](../../benchmarks/results/managed-hybrid-capacity-10k.json)已完成，运行时其他重负载评测均已结束。与原窗口方案使用相同冻结准备配方、查询编号、写入长度和embedding产物，所观测六个模型/视觉后端的PID与重启计数一致；核对见[方案对照](../../benchmarks/results/hybrid-memory-capacity-comparison.json)。

| 正式适配器配置 | 编号top5 | 长/短目标命中 | 查询P95 | 完整写入P95 |
|---|---:|---|---:|---:|
| 原窗口dense | 9/100 | 0/50、9/50 | 46.97ms | 342.79ms |
| 本地BM25＋DBSF候选 | 100/100 | 50/50、50/50 | 62.29ms | 355.66ms |

候选长写入单列P95为359.30ms，50次写入全部持久确认，没有查询/写入错误或主体泄漏。这证明该编号压力集的改进已进入真实Mem0路径，并保持本轮200/500ms目标。每种配置仅一轮，包含各自真实的绑定和索引校验开销，没有前台语音负载；这不是独立语义正确率、单因素优化收益或完整JT-11通过。[候选迁移故障测试](../../benchmarks/results/hybrid-reindex-recovery.json)三个阶段各10次真实SIGKILL共30次全部通过。82题候选开发评测与预算对照已完成，结果见下；保留集继续不用。

### 语音、视觉与记忆同时工作

`tools/memory_voice_load.py`复用同一个真实`VoiceAssistant`与记忆适配器，在实际个人命名空间放入10000条主人记录和100条访客干扰记录，其中10%的主人记录为长资料，第1号记录替换为固定的茉莉花茶偏好。测试目录带专用标记，不使用日常用户档案。

```bash
.venv/bin/python tools/managed_memory_capacity.py --phase prepare --tier 10000 --personal \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/memory-voice-load-10k-v1" \
  --output benchmarks/results/memory-voice-load-10k.json

source /opt/ros/jazzy/setup.bash
source "$HOME/AI_Stack/workspaces/vision/install/setup.bash"
ROS_DOMAIN_ID=42 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv/bin/python tools/memory_voice_load.py \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/memory-voice-load-10k-v1" \
  --output benchmarks/results/memory-voice-load-10k.json
```

准备完成且其他导入/评测停止后再计时。固定语音文件轮流询问偏好、时间、实时画面和算术，至少20轮；同一适配器每两秒提交两次语义查询和一次明确事实写入，至少100次查询、50次写入。查询与写入都走个人命名空间，不能用空的语音档案绕开万条数据。周期采样检测/姿态的最新结果年龄和缺失比例，并记录20ms事件循环定时器的延误。音频回读在计时结束后执行。

首轮[并发结果](../../benchmarks/results/memory-voice-load-10k.json)已完成20轮语音、106次查询与53次完整写入，所有记忆操作均与前台语音区间重叠，应用代码指纹保持不变。

| 指标 | 本轮结果 | 本轮检查 |
|---|---|---|
| 记忆查询P95 | 126.80ms；0错误、0主体泄漏 | 通过200ms目标 |
| 完整写入P95 | 459.14ms；53次全部持久确认 | 通过500ms目标 |
| 检测/姿态有效频率 | 两者均约11.64Hz | 通过10Hz目标 |
| 检测/姿态最新年龄P95 | 173.94/178.13ms，无缺失采样 | 通过200ms目标 |
| 偏好问答/时间工具首段PCM P95 | 3.091/3.098秒，各5轮 | 未通过3秒目标 |
| 单图问答首段PCM P95 | 5.545秒，5轮 | 未通过5秒目标 |
| 算术首段PCM P95 | 2.773秒，5轮 | 通过3秒文件模式目标 |
| 事件循环20ms探针 | 延误P95 5.88ms、最大154.90ms，错过50个周期 | 诊断数据，尚未定位原因 |

因此报告保留为`FAIL`。20轮均输出完成，5次偏好问答的文字与实际WAV回读均包含所存茉莉花茶；六个被观测模型/视觉服务的PID和重启计数保持不变。包含启动与音频回读在内的资源监测最低可用内存约2.44GiB；计时区间与回读分开。各问题类型仅5轮，首轮样本也进入分位统计，不能视作稳定分布或最终验收。

它使用真实模型和相机、合成输入音频及WAV输出，不测声学停播，也没有画面内容的地面真值。提醒/手势负载、独立质量、B0–B3、正式JT及8小时验收仍须单独完成；两份容量准备配置也不是同数据单因素对照。正式应用仍使用原来的dense检索，本轮没有启用上述BM25候选。

并发脚本现接受`--profile bge-int8-windows-bm25-v1`，须用相同profile和`--personal`准备新的独立目录；已有测量目录不可重跑覆盖。可加`--trace-stages`启用诊断：用Pipecat原生observer记录识别/模型/TTS阶段帧和框架原生指标，并在同一实例上记录记忆检索/写入/版本读取、上下文装配、图像选择、原生token计数及等待输出回执的墙钟区间。各worker的相对时钟映射到单调时钟，单列observer投递延迟，不用回调到达时间冒充实际帧时间。追踪不保存原始音视频、提示或转写正文，最多20000个事件，超限显式计数。

`--debug-loop`另行启用asyncio慢回调诊断（阈值20ms），输出到运行日志。两个诊断选项默认关闭；插桩开销尚未完成对照，诊断结果不能直接与无插桩时延分布混作优化收益。已用实际Pipecat帧和时钟验证映射、首音频帧选择、正文不进入结构化追踪，以及包装器保持返回值/异常和可恢复性；实际语音阶段追踪尚待运行，155ms最大延误的原因仍未确定。

后续先在开发集完善时间范围、对话来源与相关上下文检索，再冻结版本运行200题保留集。完整JT-10还须覆盖保存/更正/遗忘、无事实、取消、来源、重启和索引恢复；公开语言基准、词级F1以及框架回归均不能替代该验收。

候选开发质量对照使用新索引目录及明确的配置，不复用旧向量，也不接触保留集：

```bash
python3 tools/project.py bench --suite memory-quality --split development \
  --source-metadata calendar-speaker-v1 --memory-profile bge-int8-windows-bm25-v1 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-hybrid-v1" \
  --output benchmarks/results/lcvmem-development-hybrid.json
```

此命令的准备配置、实际绑定及报告均记录profile；复用或续跑的索引不匹配时拒绝继续。候选容量使用`managed_memory_capacity.py --profile bge-int8-windows-bm25-v1`在新的专用目录准备和计时；计时必须等待其他重负载评测结束。

## 候选开发质量与证据预算

候选的完整82题已完成，原结果和后续开发对照均保留。下表每行都有0执行错误、0意外写入、0跨主体证据，归一化完全匹配均为1/82；200题保留集和76题备用集未运行。

| 配置 | 词级F1 | 检索包含参考原文 | 实际提示包含参考原文 |
|---|---:|---:|---:|
| 原窗口dense，5条/512tokens | 26.69 | 58 | 56 |
| [DBSF候选，5条/512tokens](../../benchmarks/results/lcvmem-development-hybrid.json) | 25.97 | 57 | 57 |
| [直接回答，5条/512tokens](../../benchmarks/results/lcvmem-development-hybrid-direct.json) | 26.96 | 57 | 57 |
| [取8条但装配仍截5条的失败对照](../../benchmarks/results/lcvmem-development-hybrid-top8.json) | 26.96 | 65 | 57 |
| [修正装配后，8条/512tokens](../../benchmarks/results/lcvmem-development-hybrid-top8-packed.json) | 26.95 | 65 | 57 |
| [8条/768tokens](../../benchmarks/results/lcvmem-development-hybrid-top8-768.json) | 29.58 | 65 | 65 |

第一份[候选对照](../../benchmarks/results/lcvmem-hybrid-comparison.json)有3题失去、4题新增可见参考原文，整体F1没有提高。错例同时包含真正的漏召回、作品类型混淆，以及正确诗句出现在第二行导致官方首行评分为0。通用提示现要求短事实直接用简短完整句回答，引用原话时不加标题；[同证据对照](../../benchmarks/results/lcvmem-hybrid-direct-comparison.json)82题检索/注入顺序一致，F1只小幅变化。全文包含参考字符串的题数由51变50，不能把格式得分上升当作语义质量提升。

随后发现两处配置未生效：持续会话固定使用5条/512tokens，装配函数又独立截取前5条。现在文本入口、`/new`、语音入口和评测均传递配置；装配遍历全部已限定候选，使用真实token计数决定是否加入，保留未注入原因与真实ID/来源。检索API的硬上限为20条；日常默认仍为5条、512tokens。证据预算必须至少64tokens，并为模型输出留出空间；完整提示仍由原生计数再次检查。实际Mem0/Qdrant回归验证8条检索、预算约束和主人/访客隔离，220项框架测试通过。

[未完全修正时的对照](../../benchmarks/results/lcvmem-hybrid-top8-comparison.json)确认82题的实际证据、提示token数和回答完全相同，说明仅取回更多条目没有生效。修正装配后，512预算下依然没有新增包含参考原文的提示；[预算对照](../../benchmarks/results/lcvmem-hybrid-top8-768-comparison.json)保持8条检索顺序82/82一致，只将预算改为768，在2K模型上下文中新增8题可见参考原文、无丢失，F1为29.58。全文出现参考字符串的题数为55；这只是词级/可见证据诊断，尚不是人工语义正确率或JT-10通过。

复现预算候选使用已核验的开发种子及新的回答目录：

```bash
python3 tools/project.py bench --suite memory-quality --split development \
  --source-metadata calendar-speaker-v1 --memory-profile bge-int8-windows-bm25-v1 \
  --memory-top-k 8 --memory-token-limit 768 \
  --seed-from "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-development-hybrid-v1" \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/lcvmem-hybrid-8x768-new-run" \
  --output benchmarks/results/lcvmem-hybrid-8x768-new-run.json
```

每次续跑核对检索参数；非法预算在创建评测目录前拒绝。`tools/lcvmem_compare.py`只用于完成后的开发集分析，参考答案不进入运行期。默认存储配置和5条/512tokens暂未切换；8条/768tokens的后续语音共存及缓存边界见下。

## 当前候选的语音共存基线

[20轮未插桩结果](../../benchmarks/results/memory-voice-hybrid-10k.json)使用DBSF、直接回答提示、5条/512tokens和独立万条个人档案，完成106次查询与53次写入。查询/写入P95为161.22/462.00ms，全部写入持久确认，0执行错误或主体泄漏。检测/姿态约11.49/11.42Hz，最新结果年龄P95约175/182ms。

偏好/时间/算术的首段文件PCM P95分别为2.902/2.859/2.701秒，图像为5.202秒，因此整体仍为`FAIL`。每类只有5轮，图像内容没有地面真值，也没有物理可听回执。该配置同时改变检索与答复提示，不能把与旧窗口方案的差异当成单因素收益。事件循环最大延误约145ms，原因仍待实际阶段追踪确认。

## 提示缓存与后续阶段追踪

恢复故障测试期间，实际可用内存曾降到约145MiB。固定llama.cpp版本默认每个服务允许8192MiB主机提示缓存，而启动入口未显式覆盖。原生日志记录VLM缓存88份提示、3324.574MiB，ASR缓存49份、431.170MiB。[原始观察](../../benchmarks/results/llama-prompt-cache-baseline.json)保留进程、日志、空闲槽核验和服务干预信息；Jetson的进程RSS与smaps口径不同，没有将它们当成独占物理内存相加，也不把全部压力归因于缓存。

当前启动器显式传入`--cache-ram`：VLM256MiB、ASR64MiB，配置位于`runtime.yaml`的`llama.prompt_cache_ram_mib`与`speech.asr_prompt_cache_ram_mib`。这个16GB部署档案拒绝负数/无限缓存及超过1024MiB的单服务设置，0表示关闭该缓存。模型、量化、2K上下文不变；整体进程/设备内存仍需单独测量。

停止空闲模型服务后，可用内存恢复到约9.2GiB，其中也包含模型本身释放的内存；ASR依赖关系同时停止了speech，随后三者已按新参数恢复。[20次文本＋20次语音缓存检查](../../benchmarks/results/llama-prompt-cache-bounded.json)观察到实际淘汰，缓存最高246.232/62.580MiB，校验标签及原语音问题均正确返回，没有工作进程重启；不是8h或独立质量验收。[公开语音保存/回忆](../../benchmarks/results/cache-bounded-voice-smoke.json)通过，222项框架回归通过。恢复故障测试跳过对话生成，40次最终均通过，但报告明确记录了服务干预，不据此声称全模型共存通过。

8条/768tokens的新测试先准备未测量的10000条主人＋100条访客档案，再用原生Qdrant快照和SQLite备份复制一份。[复制验证](../../benchmarks/results/voice-capacity-seed-clone.json)核对完整点校验和、规范合成正文/来源、纯种子操作及SQLite逻辑状态，为副本建立新的绑定。`tools/clone_capacity_fixture.py`仅接受未使用的合成容量档案，已测量档案会在创建目标前拒绝；它不是日常数据复制入口。备份和原始波形留在AI_Stack，测量中仍执行真实BGE/适配器。

| 8条/768tokens、缓存256/64MiB | 查询/写入P95 | 短问答各类最高P95 | 图像P95 | 结论 |
|---|---|---|---|---|
| [无插桩，20轮](../../benchmarks/results/memory-voice-hybrid-8x768.json) | 153.42/468.51ms | 3.271秒 | 4.895秒 | 短问答超3秒，FAIL |
| [阶段追踪，20轮](../../benchmarks/results/memory-voice-hybrid-8x768-trace.json) | 184.08/481.47ms | 3.304秒 | 4.273秒 | 诊断运行，短问答仍超标 |

两轮的记忆操作、文件输出、主人/访客隔离与视觉频率/年龄检查通过；每类只有5轮，现场画面、时间回答和缓存热度有所不同，因此没有把差值当作插桩开销或单因素收益。当前仍只验证文件PCM，没有物理可听起点或画面内容地面真值。

[阶段分析](../../benchmarks/results/voice-stage-diagnostic.json)得到1150个结构化事件、0超限丢弃。慢算术请求输入1339tokens，记忆证据707tokens，原生LLM首token等待约1.67秒，说明完整字面算式的长期记忆检索值得单独做选择性实验。图像轮还多次重算提示以移除历史，最终清空证据来满足2K预算；真实图像计数保护继续保留。主线程记忆generation读取约10–17ms，与最大136ms事件循环延误不重叠，最大延误原因仍未确定。墙钟区间包含等待、可能彼此嵌套，不作CUDA kernel或独占CPU时间解读。

## 完整算式的选择性记忆实验

新增的`memory.skip_literal_arithmetic`默认关闭。开启时，只对完整的数字/中文数字算式省略长期向量检索；模型仍实际回答，不计算或写死结果。当前会话历史及其记忆引用继续传递，个人问题、单位、变量、指代、日期、电话号码和额外指令走原路径；维护操作、图像请求也不启用此规则。跳过原因记录为`literal_arithmetic`。版本读取仍检查真实绑定与owner，只移到线程中执行，未降低存储检查。225项框架测试通过，原82题中文记忆开发问题均未命中该规则。

为避免优化后输入节奏随回复速度改变，开发对照采用[固定协议](../../benchmarks/manifests/literal-arithmetic-routing-v1.json)：同一封存种子快照分别恢复新档案，20个相同语音输入每10秒释放；两轮都先重启模型服务、保持8条/768tokens及256/64MiB缓存。记录释放延迟，并把它加入文件首PCM时延；实际文件帧仍按既有方式播放，不能视为物理声学端点。背景记忆按2秒周期持续至前台结束，因此末尾操作数量可以不同。

[基线](../../benchmarks/results/voice-fixed-arrival-baseline.json)20轮完成：算术P95为3.165秒，回忆4.262秒、时间3.209秒，图像4.923秒，整体FAIL。首次[开启规则的运行](../../benchmarks/results/voice-fixed-arrival-selective.json)在第15轮图像请求发生`CAMERA_EPOCH_CHANGED`，前三个算术回合虽较快，但不能形成完整20轮结论；[不完整比较](../../benchmarks/results/voice-routing-incomplete-comparison.json)不发布收益百分比。相机保护没有放宽，旧帧回答被拒绝，原失败保留。

随后使用另一份全新档案[被动观察GC复跑](../../benchmarks/results/voice-fixed-arrival-selective-v2.json)，20轮均完成，算术P95约1.941秒，但回忆4.348秒、时间3.130秒，整体仍FAIL；图像4.826秒。由于新增GC观测，与无观测基线不作严格单因素收益结论，比较工具检查观测模式并保留[诊断比较](../../benchmarks/results/voice-routing-gc-diagnostic-comparison.json)。

[GC诊断](../../benchmarks/results/voice-gc-pause-diagnostic.json)记录328次回收、0丢弃；一次generation2回收耗时580.1ms，与最大573.5ms事件循环延误重叠，所在时段为第15轮图像请求。相机输入再次重置epoch，但该轮没有形成问答失败。这是明确的时间相关证据，尚未完整隔离因果；回收策略保持原样，没有禁用GC或改阈值。启动期静态对象管理、关闭后的释放和长时内存曲线需要进一步验证。

`clone_capacity_fixture.py --from-snapshot`只重用此前完整回执绑定的不可变`seed-backup`，核对其manifest哈希并再次验证点/SQL/合成内容；不会复制已测工作目录新增的事实。`memory_voice_load.py --skip-literal-arithmetic/--no-skip-literal-arithmetic --turn-interval 10`选择开发条件；`--observe-gc`只记录回收时间。已测目录和失败结果不可覆盖。

## 启动期对象冻结实验

针对已观测的generation2停顿，增加独立文件测试进程`tools/heap_voice_worker.py`，比较启动时`collect`与`collect`后`freeze`。[冻结协议](../../benchmarks/manifests/startup-heap-v1.json)要求相同种子、20个固定到达输入、8条/768tokens、算式路由开启及相同缓存上限；两轮自动GC都保持开启、阈值不变。普通应用入口不启用这项策略。

本机CPython3.12.3启动即有375个永久代对象；`gc.unfreeze()`后为0，但完整回收后再次为375。[上游实现](https://github.com/python/cpython/blob/v3.12.3/Modules/gcmodule.c#L397)在`update_refs`中将不适用普通引用计数循环检测的immortal对象移入永久代。因此实验不再把“永久代必须为0”当作恢复条件，[前置条件记录](../../benchmarks/results/startup-heap-precondition.json)保留该修正。

专用工作进程在导入应用前记录启动基线；`StartupHeapPolicy`只在此基线上取得独占实验权。启动收集发生在语音输入开放前，冻结后新建对象仍参与自动回收；结束后解冻、回收，并核对永久代基线、GC开启状态和阈值均恢复。外来冻结状态或并发实验会被拒绝。实际弱引用/循环对象测试验证新对象可回收、冻结旧循环延迟至关闭后释放，227项框架回归通过。

启动、运行期和关闭回收区间分别记录，不能把启动开销藏入“运行期优化”。即使短测中没有完整回收，也可能只是把停顿推到测试窗口之外。该策略还没有成为生产默认，8h内存趋势、反复关闭/重开和真实音频采集生命周期仍待验证。

[短测对照](../../benchmarks/results/voice-heap-comparison.json)现已完成，两份原始结果均保留：

| 同一输入/种子，20轮 | 只在启动时回收 | 启动回收后冻结 |
|---|---:|---:|
| 启动处理 | 541ms | 536ms |
| 运行期间最大GC | 5.07ms | 10.30ms |
| 运行期间第二代GC次数 | 0 | 0 |
| 最大事件循环延误 | 167ms | 373ms |
| 回忆首段文件PCM P95 | 4.310秒 | 4.274秒 |
| 关闭处理 | 552ms | 641ms |
| GC恢复 | 原永久代基线375、开启、原阈值 | 原永久代基线375、开启、原阈值 |

[collect原始报告](../../benchmarks/results/voice-heap-collect.json)与[freeze原始报告](../../benchmarks/results/voice-heap-freeze.json)都完成20轮、无执行错误，仍因短问答时延超标记为FAIL。冻结轮启动时约54.6万个对象进入永久代，关闭后按约定恢复；测试证明生命周期机制可运行，但没有证明冻结改善整体延迟。两轮均未遇到运行期间第二代GC，无法据此判断长时间运行是否仍会停顿，故不启用为默认策略。首次回忆约4.3秒的问题将单独做冷启动阶段追踪。

## 冷启动瓶颈与原生微批量筛选

[冷启动20轮报告](../../benchmarks/results/voice-cold-stage.json)使用新的同种子档案、8条/768token记忆、固定10秒输入间隔及真实ASR/LLM/TTS，开启阶段和GC观察。20轮全部输出，记忆查询/写入P95为149.86/459.16ms；回忆、时间、图像、算术首文件PCM的各类P95为4.350/3.285/5.202/1.999秒。短问答与图像延迟均未达标，整体FAIL。

[阶段汇总](../../benchmarks/results/voice-cold-stage-diagnostic.json)包含逐轮时序及首轮提供方日志。首轮回忆输入1146tokens，ASR在VAD结束后693ms给出转写，准备上下文367ms，LLM原生TTFB2166ms、完整处理2705ms；其中提供方日志记录提示计算1928ms。TTS开始到首个原生音频帧约102ms。首轮图像TTFB约2960ms，后续约2002–2076ms；图像上下文准备545–725ms包含多次原生图文token计数。这些墙钟区间可嵌套，不能当成独占CPU或CUDA kernel时间；首PCM仍是文件输出。

同一固定版本新增编译`llama-bench`目标，未改上游源码。`tools/llama_prefill_probe.py`确认受管理VLM空闲后暂时停止它，运行原生基准并在`finally`中恢复原服务；其他后端继续驻留，相机/检测/姿态持续运行。程序保留原生JSON、日志、模型/二进制SHA、实际命令、45个重复样本、视觉与内存观察。筛选不包含图像编码、聊天模板、采样、主动ASR/TTS或记忆负载。

| 输入token数 | 微批量256，平均ms | 微批量512，平均ms | 微批量1024，平均ms |
|---|---:|---:|---:|
| 512 | 840.45 | 817.55 | 823.13 |
| 1024 | 1641.22 | 1632.79 | 1657.22 |
| 1536 | 2471.05 | 2457.42 | 2478.92 |

每组5次，逻辑批量2048、4线程、GPU offload99及默认原生warmup，权重相同。[原始报告](../../benchmarks/results/llama-prefill-screening.json)没有显示调整微批量能改善提示处理，因此保留当前512。测试期间检测/姿态为11.86/11.89Hz，最新结果年龄P95为174/189ms，最低可用内存约5.61GiB；该内存数不包含生产VLM图像编码器，不能代表整机完整负载。

结束时姿态的单次就绪探测返回`no_fresh_sample`，报告整体保留FAIL，没有据此重启姿态或覆盖失败。随后[独立复查](../../benchmarks/results/llama-prefill-readiness-recheck.json)七个服务全部READY。除计划中的VLM替换外，其余后端PID保持一致，配置文件没有变化。这是参数筛选，未达到B0–B3、每配置5轮至少100事件或JT门槛。下一候选是原生提示复用式推测解码，须先验证真实文字/工具/图像输出、草稿接受量和耗时，再考虑接入应用；默认推理解码策略未改变。
