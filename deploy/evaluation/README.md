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

## 原生提示匹配推测解码候选

`tools/llama_speculative_probe.py`使用固定版本原生`llama-server`，分别冷启动默认解码和`ngram-simple`，匹配长度2、草稿长度4、最少命中1。它复用提示中已有token生成草稿，并由同一个目标模型验证，不加载第二个模型，也不在应用中实现解码器。其他后端继续驻留，两个临时VLM退出后恢复受管理服务。全部请求关闭提示缓存，权重、图像投影、聊天模板、温度0及输出上限相同。

[固定合成输入](../../benchmarks/corpora/llama-speculative-v1.json)共有14题，覆盖来源事实、更正、角色区分、原文引用、算术、3个纯色图像、2种工具JSON；每种模式重复3次。预设答案只进入评估器，工具调用仅解析而不执行。[原始结果](../../benchmarks/results/llama-speculative-screening.json)的42对请求、输出正文、工具名称/参数及结束原因全部一致，84个预设答案检查通过，没有提示缓存命中。原生slot明确报告候选已启用，逐请求timings与原生Prometheus计数都记录草稿432tokens、接受240tokens。

| 请求类别 | 默认总响应P95 | ngram-simple总响应P95 | 每种模式样本数 |
|---|---:|---:|---:|
| 来源事实 | 850ms | 627ms | 9 |
| 原文引用 | 2546ms | 914ms | 9 |
| 工具JSON | 1945ms | 1283ms | 6 |
| 算术 | 446ms | 458ms | 9 |
| 纯色图像 | 2404ms | 2418ms | 9 |

这些是完整原生HTTP响应耗时，不是语音首PCM；算术和图像没有显示收益。结束快照的speech cgroup多出PID663421，使原报告的全进程集合不变检查失败，整体保留FAIL。语音服务原来的两个PID及重启计数不变，之后额外PID已消失，但未捕获其用途，不将其直接认定为某种探测子进程。[诊断汇总](../../benchmarks/results/llama-speculative-diagnostic.json)保留这个限制；本轮没有连续视觉频率测量，也不能代表独立质量、真实工具执行或JT验收。

启动器已提供默认关闭的`llama.speculative`配置。省略该字段或设置`{"type":"none"}`保持原路径；实验候选为`{"type":"ngram-simple","match_tokens":2,"draft_tokens":4,"min_hits":1}`。参数只转发给原生VLM服务，不用于ASR；匹配长度限定1–32，草稿长度和最少命中限定1–16，并拒绝布尔值、未知字段和模式。完整语音对照使用`tools/speculative_voice_pair.py`，新建同种子10k记忆库、冷启动VLM/ASR/TTS，每组20轮固定间隔输入；结束时恢复原配置和服务。候选是否进入默认配置，仍取决于完整负载、质量、取消/恢复和长期稳定性验证。

[完整对照执行记录](../../benchmarks/results/speculative-voice-pair.json)已完成：每组20轮、196次真实记忆查询及98次持久写入，启动回收与被动GC观察相同，没有执行错误、主体泄漏或丢失写入。[比较核验](../../benchmarks/results/speculative-voice-comparison.json)确认输入波形/顺序、初始数量、封存快照、8条/768token检索及10秒间隔相同；按唯一不同的推测解码配置重建的应用指纹与两份报告完全匹配，原生slot分别确认关闭/开启。

| 完整文件语音P95 | 默认 | ngram-simple |
|---|---:|---:|
| 记忆问答 | 4.377秒 | 4.098秒 |
| 时间问答 | 3.224秒 | 2.297秒 |
| 看图问答 | 4.772秒 | 4.840秒 |
| 算术问答 | 2.003秒 | 1.746秒 |
| 记忆查询 | 161.03ms | 155.19ms |
| 记忆写入 | 496.19ms | 514.23ms |

[默认报告](../../benchmarks/results/speculative-voice-pair-baseline.json)因短问答超过3秒而FAIL；[候选报告](../../benchmarks/results/speculative-voice-pair-ngram_simple.json)同时未满足短问答和写入500ms门槛，保留FAIL。候选时间/算术P95在本轮分别下降约28.8%/12.8%，并未解决冷启动回忆，也没有改善看图。两组检测/姿态约11.5–11.7Hz，年龄P95约174/190ms；事件循环最大延误约150/282ms。这里仅一组配对、每类5个样本，真实时间和相机画面可能变化，不能把百分比当作统计显著性或正式单因素验收。

测试后核对原始配置字节完全恢复，受管理VLM的原生slot回到`speculative=false`，七个后端READY。两项启动器缓存回归及默认/候选参数校验通过；现有227项框架报告没有被这组性能实验覆盖或冒充重新执行。下一步需解释写入尾延迟并改善记忆回答质量；候选仍待更广质量、取消/恢复和长期资源验证，不晋升为默认方案。

## 长记录写入阶段与CPU线程候选

`tools/voice_stage_trace.py`新增只在诊断时启用的Mem0、向量编码、Qdrant插入、history和journal阶段计时，并记录插桩源码SHA。原有锁、编码批量、写入顺序与持久性校验保留。[20轮阶段报告](../../benchmarks/results/memory-write-stage.json)完成194次查询、97次写入，3972个事件且无丢弃；整体写入P95为479ms，但49条长记录自身P95为544ms，短记录P95为158ms。

[按同一写入线程归因的汇总](../../benchmarks/results/memory-write-stage-diagnostic.json)排除了等锁期间另一线程的查询编码：每次写入实际只调用一次编码。长记录平均440ms，其中编码约322ms；短记录平均120ms，其中编码约16ms。最慢写入约636ms，包含331ms编码和197ms原生Qdrant插入；另两次慢写入在第一次绑定校验前有约101/102ms区间，与并发查询重叠。这个前置区间还包含参数校验与调度，不等于精确的锁等待测量；嵌套阶段不能直接相加，也不能据此把上一轮514ms尾延迟归因于推测解码。

`tools/embedding_threads_probe.py`在独立子进程中使用实际Mem0 HuggingFace provider、相同ARM64 int8模型与逐记录窗口编码，测试1/2/4个ORT内部线程；每种策略各20条长记录、20条短记录和20个强制重新编码的查询。

| ORT线程 | 长记录编码P95 | 短记录编码P95 | 查询编码P95 |
|---|---:|---:|---:|
| 1 | 559.25ms | 12.80ms | 20.15ms |
| 2 | 289.15ms | 9.68ms | 14.97ms |
| 4 | 165.14ms | 8.00ms | 10.48ms |

[比较报告](../../benchmarks/results/embedding-thread-comparison.json)确认输入、模型、脚本和驻留后端相同，1/4线程的60组float32向量分别与2线程逐元素一致，最大差为0。这是编码器筛选，未包含完整写入、前台语音或连续视觉频率测量；不据此宣称完整负载提速或所有输入的数值等价。

同时修复`memory.embedding_threads`原先未传入初始化代码的问题：`open_memory`现在将它交给Mem0适配器的真实ORT SessionOptions，支持整数1–4；未提供该参数的旧调用回退到2。非法值会在创建持久目录之前拒绝。完整227项框架回归通过。`memory_voice_load.py`记录实际ORT线程数/provider并与配置核对；已有成对语音执行器新增`--factor embedding_threads`，用于相同冷启动、记忆快照和20轮输入下比较2/4线程，再恢复原配置。

[完整2/4线程执行记录](../../benchmarks/results/embedding-voice-pair.json)已完成，两组实际ORT分别为intra2/4、inter1、CPUExecutionProvider。比较脚本按线程配置重建应用指纹，确认其余应用内容、输入波形与顺序、8条/768token检索、10秒间隔和封存种子一致。

| 完整负载P95 | 2线程 | 4线程 |
|---|---:|---:|
| 记忆查询 | 158.66ms | 140.42ms |
| 记忆写入 | 484.26ms | 357.08ms |
| 其中长记录写入 | 492.10ms | 382.89ms |
| 记忆问答首文件PCM | 4.232秒 | 4.373秒 |
| 时间问答首文件PCM | 3.247秒 | 3.153秒 |
| 看图问答首文件PCM | 4.867秒 | 4.790秒 |
| 算术问答首文件PCM | 2.016秒 | 1.966秒 |

[默认组](../../benchmarks/results/embedding-voice-pair-baseline.json)194次查询/97次写入，[4线程组](../../benchmarks/results/embedding-voice-pair-threads4.json)196次查询/98次写入；后台持续到语音结束，末尾数量可不同，每组均包含49次长记录写入。两组无执行错误/主体泄漏/丢失写入，检测/姿态频率保持约11.4–11.6Hz，年龄P95低于185ms。候选最长写入仍有506ms，不能声称每次写入都小于500ms；两个总报告仍因短问答3秒门槛失败。单次配对中写入P95下降约26.3%，不替代5轮/100事件、显著性或8h验收。

[比较报告](../../benchmarks/results/embedding-voice-comparison.json)记录原配置及服务已在实验结束时恢复、七个后端READY；随后基于同向量与完整负载结果，将`configs/runtime.yaml`中的应用默认改为4线程。使用当前默认窗口记忆档案的[公开语音入口复测](../../benchmarks/results/embedding-threads4-public-voice.json)完成真实ASR/记忆写入/再次回忆/LLM/TTS到文件，事实实际持久化、输出完整。推测解码继续关闭，长期稳定性与所有JT仍待验证。

## 词级低分与来源级错误复核

[四个开发案例复核](../../benchmarks/results/lcvmem-reader-error-diagnostic.json)选择词级F1为0且旧报告标记“参考答案可见”的全部4题，仅作诊断。两题的核心回答与参考答案是同义表达；一题的目标工具已在第一条注入来源中，模型仍回答了邻近的泛泛建议；另一题的正确用户来源排在第8条，因768token预算被丢弃，旧的“可见”统计来自另一日期、另一首作品的归一化子串重合。

因此，29.585词级F1不是29.585%语义准确率，“归一化参考答案子串出现”也不能证明正确来源已进入提示。官方评分和旧报告保持原样；上述人工式复核由助手完成，未经过独立人员裁定，不构成新的准确率或验收结果。原始语料与答案仍保留在仓库外，200题保留集和76题备用集未检查。后续需减少证据格式开销并验证来源保留，同时独立评估模型读取证据的能力。

## 单来源证据格式候选

`memory.evidence_format`新增默认`nested_v1`与实验`flat_v2`。候选对只有一个来源、且来源字段可完整表示的普通回忆条目，将date/speaker/evidence/span直接放到事实条目中，共用该事实的显示ref；完整memory ID、版本、原始source ID、未截断来源原文及窗口映射仍保存在同一个外部引用表。原文和日期/角色没有改写，多来源或扩展字段保留原嵌套格式；更正、遗忘使用的完整ID格式也保留。文本、新会话、语音及质量测试入口均传递配置。

[82题装配复放](../../benchmarks/results/evidence-format-packing.json)使用原开发集检索快照和真实llama.cpp tokenizer，没有重新检索或生成答案。每个新条目可还原为完整旧表示，来源映射完全一致。18题增加了一个来源，没有题目丢失原先保留的来源；总注入条目626→644，总证据tokens为57870→53042，均保持768上限。此前音乐问题从696tokens/7条变为691tokens/8条，实际目标日期的用户来源重新进入提示。这只证明装配与数据保留，不证明模型正确回答。

新增四项契约测试覆盖原文/角色/来源映射、窗口双重偏移、多来源/扩展字段/维护ID及非法格式，完整231项框架回归通过。[冻结协议](../../benchmarks/manifests/evidence-format-v1.json)要求同一代码、模型、4线程编码、8条检索、768tokens和原生评分，对全部82题分别运行nested/flat两种格式。使用新会话与已核验未变化的原存储，参考答案只用于事后评分；200题保留集及76题备用集不进入实验。当前默认仍为nested_v1，是否采用候选以完整质量结果为依据。

[本轮nested基线](../../benchmarks/results/lcvmem-development-evidence-nested.json)与[flat候选](../../benchmarks/results/lcvmem-development-evidence-flat.json)均完成82题，执行错误、意外写入、主体泄漏均为0，应用指纹与冻结协议一致。[比较](../../benchmarks/results/lcvmem-evidence-format-comparison.json)确认82题检索顺序相同，18题的可见来源增加，官方词级F1为28.9745→30.5796，逐题14项升、10项降、58项相同。归一化参考答案在完整回复中的出现次数54→58；这些仍不是语义正确率。

来源级检查确认此前音乐问题现在得到目标日期用户原话，模型回答正确作品和作曲家；已有首条证据的器材问题仍然漏读。另一个植物问题在来源相同的情况下，从具体困难退化为泛泛描述。[决策记录](../../benchmarks/results/evidence-format-decision.json)因此保留候选，未切换默认。此次[nested重复检查](../../benchmarks/results/lcvmem-nested-repeat-diagnostic.json)也发现与此前版本相比，82题检索、可见来源和提示token数一致，仍有6题回复不同、F1为29.5846→28.9745；两次运行还隔着资源配置/缓存历史等变化，不能据此声称确定的随机误差分布。单次小幅分数提升不足以证明稳定质量收益。

为进一步区分装配问题和阅读模型能力，已锁定[Qwen3.5-4B Q4_K_M候选及F16图像投影](../../models/candidates/qwen3.5-4b.json)的仓库revision、文件大小和SHA256，准备单独评估。候选使用官方文档提供的`enable_thinking=false`模式，继续走本地llama.cpp/Pipecat接口；[Qwen官方模型说明](https://huggingface.co/Qwen/Qwen3.5-4B#instruct-or-non-thinking-mode)、[量化产物来源](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)。产物准备不证明本机兼容、质量或速度，当前主模型manifest和默认配置未切换。

## 4B候选的加载与接口验证

[权重与投影文件](../../benchmarks/results/qwen35-candidate-artifacts.json)已按固定revision、字节数和SHA256校验。`tools/project.py`启动器支持显式模型条目、独立产物目录和布尔`enable_thinking`模板参数，原有默认入口保持原模型。`tools/reader_candidate_probe.py`先确认受管理VLM空闲，再由独立原生进程加载候选；检查该PID确实拥有模型端口，并在退出路径恢复3B受管理服务。原生模板只记录SHA，测试脚本快照放在各自AI_Stack产物目录并核对源码SHA。

[首次原生检查](../../benchmarks/results/qwen35-native-probe.json)成功加载33/33层到GPU，仍使用2048上下文；14个固定文字/工具/纯色图像请求均完成，原生token预检与实际usage逐项一致，工具名/参数可解析，未观察到reasoning输出。13个请求符合严格预期；长引用多输出一对引号，因此整体保留FAIL，不改写判定来获得通过。模型进程与主manifest/配置分离，结束后七个受管理后端READY。

该短检查最低MemAvailable为4806303744字节（约4.48GiB），包含其他驻留后端，但没有前台语音/记忆并发或连续视觉速率测量。原生工具JSON请求约3.9秒，不能据此认为候选更快；原生RSS、CUDA统一内存和文件映射也不能直接相加解释整机占用。

质量入口新增显式`--model-manifest`，在推理前核对真实`/props`模型路径与别名，将所选manifest纳入读者指纹并写入评分报告，防止模型身份错记。[冻结的读者对照协议](../../benchmarks/manifests/reader-candidate-v1.json)使用同一AssistantSession、4线程BGE、hybrid检索、8条/768tokens、flat_v2、2K上下文和128输出上限，比较既有3B结果与4B的完整82题开发集。温度0用于本轮受控比较，不声称复现上游推荐采样；更换分词器和原生模板可能改变实际提示token数与可见来源，必须在结果中检查。200题保留集和76题备用集不参与此次比较。

[4B完整82题结果](../../benchmarks/results/lcvmem-development-qwen35-flat.json)及[执行记录](../../benchmarks/results/qwen35-quality-probe.json)已完成：词级F1为31.3466，3B/flat基线为30.5796；没有意外写入或跨主体证据，1个错误保留在分母中。[比较](../../benchmarks/results/lcvmem-reader-model-comparison.json)显示77题可见证据相同，分词器/模板变化使5题装配不同，不能称为完全相同提示。归一化目标在完整回答中的出现次数仍为58/82。4B修正了植物问题，但器材问题虽引用了正确来源，仍否认给出过具体推荐。

逐题观测墙钟P50/P95为3B的2.755/5.909秒、4B的4.766/9.532秒，含会话初始化/关闭及来源检查，不是声学首PCM或独占GPU耗时。[决策](../../benchmarks/results/qwen35-reader-decision.json)因此保留3B默认，4B未选用。首轮82题质量记录及此前长引用加引号的失败均不改写。

## 记忆主题误判与查询续轮修复

4B的错误暴露了应用路由问题：一个研究记忆心理学的问题，因为同时出现“记忆”和“什么”，被强制路由到`list_memories`；重复列举的结果最终触发2K上下文上限保护。3B在同一问题上也有一次多余列举。这里修复意图判定，不放宽原生token预算。

`memory_listing_request`区分明确的存储清单请求与记忆理论/具体历史事实问题；后者继续语义检索，列表工具不进入其schema，执行端也会拒绝模型越过这个边界。明确清单和更正/遗忘前置查找仍可读取真实存储。明确查询成功后，通过Pipecat原生context移除本轮工具，防止重复列举结果继续累积；维护操作需要的后续写工具不受影响。

[3B专项回归](../../benchmarks/results/memory-topic-regression.json)和[4B专项回归](../../benchmarks/results/memory-topic-qwen35-regression.json)使用实际BGE、相同已保存开发来源、真实AssistantSession及各自模型，均正确回答目标技巧且不列举记忆；随后明确要求清单时，恰好调用一次、正常结束，来源存储未变化。4B外层[执行记录](../../benchmarks/results/qwen35-routing-probe.json)仍因原生长引用格式不符而保留FAIL，子流程回归PASS不覆盖它。两次运行均已结束，原3B服务恢复。

新增测试覆盖普通主题不能执行清单、明确清单/维护可读真实SDK存储，以及查询成功后原生context禁用工具。测试最初将空工具误断言为`[]`，按Pipecat实际`NOT_GIVEN`契约修正后，完整234项框架回归通过。另改进质量记录：独立捕获初次语义检索，避免`list_memories`更新候选后覆盖该快照；失败时也保留实际注入证据和token记录。历史报告不据此重写，专项回归不产生新的完整82题分数。

第一次[100条工具开发集复跑](../../benchmarks/results/tool-quality-memory-routing.json)得到99/100、无意外写入或重放新增。虽然满足该开发脚本的95%阈值，T082仍存在退步：笔记工具返回了正确数据，最终却只回复“明白了”。使用原合成笔记时间的[真实模型复现](../../benchmarks/results/tool-notes-guard-reproduction.json)和[输出过滤记录](../../benchmarks/results/tool-notes-guard-diagnostic.json)确认，模型以“我已经保存了以下笔记”开头，旧防虚假写入规则将整个列表丢弃。

现在仅在本轮明确清单查询已成功时，把这类已有状态的标题改为“当前已保存的笔记/记忆”或“当前已设置的提醒”，原列表数据保留。没有真实成功写入时，“已删除”等新操作确认仍被拒绝。另将查询完成条件限定为本轮主操作；当取消或更正先查找条目时，后续维护工具仍留在原生context中。新增回归覆盖列表正文保留、未执行删除的拒绝，以及含“列出”字样的取消请求不会过早清空后续工具。第二次全量工具与框架回归正在运行。

第二次[100条复跑](../../benchmarks/results/tool-quality-read-guard.json)仍为99/100：过滤器已保留正文，但T082中模型省略了工具已返回的笔记标题。最终对明确笔记清单采用实际工具数据呈现标题和正文，并保留“当前/最近三条”等显示范围；询问日期/时间时显示真实创建时间。这仅格式化已成功读取的数据，不新增存储、检索或工具执行机制，也不向后续模型上下文追加一份重复列表。未请求/未成功的写入确认继续受原规则限制。

更新后的236项框架回归通过，覆盖工具输出漏标题、误说删除时仍返回真实只读结果，以及无成功读取时不能伪造存储状态。最终100条实际模型/SDK工具回归仍在运行；之前两份99/100报告保留，不拼接成一次新的100/100。

[最终全量工具复跑](../../benchmarks/results/tool-quality-authoritative-inventory.json)现已完成，100/100通过、意外持久化变更0、重放额外变更0。T082实际输出包含工具返回的“会议”标题和相应正文。完整236项框架回归也通过。这是同一冻结合成开发语料的独立新运行；不是独立人类语言测试集、ASR/声学测试或JT-04验收。

## 当前阶段的中文图文模型筛选

2026-10-14起，长期记忆专项按用户要求延期；以下图文筛选继续服务于当前语音/视觉/工具阶段。

[数据清单](../../benchmarks/manifests/mmbench-cn-v1.json)固定MMBench中文V1.1开发源，按解码后图片哈希隔离开发、保留、备用各100题。开发组包含372个上游原有选项轮换；只使用开发组，不推理保留/备用组。数据、图片和原始回答保存在AI_Stack；Git登记哈希和统计。参考答案与运行输入分开，真实AssistantSession/Pipecat接收固定640×480图像、相同提问、空的独立用户记忆、2048上下文及128输出上限。

评分复用[锁定的VLMEvalKit组件](mmbench.lock.json)。仅加载校验过的匹配及循环评分函数，避免执行上游包初始化中的TLS设置；没有云端判卷或无法解析时的随机猜测。运行错误、状态异常及未解析答案均计错。

| 开发筛选 | Qwen2.5-VL-3B | Qwen3.5-4B |
|---|---:|---:|
| 标准选项顺序正确 | 70/100 | 75/100 |
| 全部提供的轮换都正确 | 55/100 | 58/100 |
| 全部372次回答正确 | 251/372 | 272/372 |
| 未解析/执行错误/意外写入/图像引用残留 | 0/0/0/0 | 0/0/0/0 |
| 每次请求墙钟中位数 | 2.397秒 | 2.108秒 |
| 每次请求墙钟P95 | 2.627秒 | 2.489秒 |

[逐题比较及证据](../../benchmarks/results/mmbench-reader-comparison.json)显示标准顺序13题改善、8题退步；两种模型都未达到预先登记的80%开发筛选目标。墙钟包括独立会话启动/清理，输出通常只有一个字母，不是语音首播时延。候选改变整个模型、tokenizer、模板和projector；单轮数据不能证明统计稳定收益或参数量因果关系。候选控制进程额外驻留的资源开销也保留在报告范围中。

4B的图像/token/状态审计通过；14项原生兼容检查仍有一项失败：原样复述时多加引号。父报告因此保留FAIL，图文评分报告单独记录完整有效结果。候选退出后原3B恢复，七个受管理服务就绪。当前不切换默认模型，不将公开选择题分数视作自然图文问答或JT-07通过。

已有完整结果可用以下命令重新审计、评分及比较；输出使用新文件名：

```bash
.venv/bin/python tools/mmbench_audit.py \
  --run-report benchmarks/results/mmbench-cn-qwen35-inference.json \
  --output /tmp/mmbench-qwen35-audit-recheck.json
/home/nvidia/AI_Stack/environments/evaluation/bin/python tools/mmbench_eval.py score \
  --run-report benchmarks/results/mmbench-cn-qwen35-inference.json \
  --output /tmp/mmbench-qwen35-score-recheck.json
.venv/bin/python tools/mmbench_compare.py --output /tmp/mmbench-comparison-recheck.json
```
