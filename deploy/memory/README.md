# 本机 Qdrant 服务

公开文字和语音入口默认使用 Mem0 2.2.1 + 本机Qdrant服务。原Qdrant local仍用于兼容SDK测试和历史基准。10k嵌入式基线的查询/写入P95为350/906ms，主要耗时在过滤和查询。这里评估同一SDK连接本机原生Qdrant服务，不更换BGE或事实事务边界。

## 长条目索引与已有档案升级

新建档案默认采用`bge-int8-windows-v1`：同一BGE-small-zh-v1.5的ARM64 int8 ONNX，256个BGE token窗口、32个重叠token，以及Qdrant原生MaxSim。每条记忆仍只有一个ID、完整正文和版本，多个向量存在同一个点中；实体索引使用相同向量结构。模型目录、完整artifact指纹及窗口参数锁定在`models/manifest.json`，通过`prepare --component memory`准备并核验。

语义检索选中与问题最相关的原文窗口，保留Unicode位置和正文哈希；提示不会改写原文，完整正文及来源仍保留在Mem0和本轮来源映射中。更正/遗忘继续操作规范记忆ID。超出预算而未展示的条目记录原因，不把本轮没取到证据解释为用户从未保存。

已存在的schema1档案继续按原FP32单向量配置读取，升级需要退出该档案的应用进程后执行：

```bash
python3 tools/project.py prepare --component memory
python3 tools/project.py reindex-memory --data-dir "$HOME/AI_Stack/runtime-data"

# 仅撤销尚未发布的新索引：
python3 tools/project.py reindex-memory --data-dir "$HOME/AI_Stack/runtime-data" --abort
```

旧的Qdrant local档案先按下文`migrate-memory`迁移到受管理服务，再执行`reindex-memory`。升级保持原事实ID、正文、版本、来源、SQLite控制库及数据沿革；先合并遗忘屏障，再重建主索引和有效实体关联。未完成remember/correct须先用原请求修复。新索引逐点校验后才发布，随后清理经过归属核验的旧原生collection。

`memory/reindex-transition.json`阻止中断期间的应用访问。重复相同命令续跑；完成后再次执行只检查当前绑定，不回放旧索引。发布前可撤销，发布后须续跑。不要删除或手工改写标记绕过检查。恢复得到的旧档案同样可以升级，其原有lineage保持不变。

动态int8量化受同批其他文本影响，因此窗口按单条记忆编码，写入、更新、迁移和恢复使用同一分组方式；没有放宽原向量一致性校验。[失败基线](../../benchmarks/results/window-restore-batch-baseline.json)和修复后的[正式适配器集成](../../benchmarks/results/window-memory-integration.json)均保留。公开入口已验证迁移后长资料回忆、更正、遗忘和备份恢复；[旧格式备份验证](../../benchmarks/results/window-legacy-backup-integration.json)还覆盖新档案直接保存长原文后跨会话回忆。

[迁移故障报告](../../benchmarks/results/window-reindex-recovery.json)在批次完成、校验完成和发布后三个阶段各执行10次真实SIGKILL，共30次，全部可续跑，且完成后重试不会复活已遗忘内容。这些合成资料验证不替代独立语义质量、10k完整负载时延、磁盘/断电及全部JT验收；新配置的82题开发评测另行运行。

### 本地关键词与语义融合候选

`bge-int8-windows-bm25-v1`已接入正式Mem0适配器，当前须显式选择，默认仍为上面的窗口配置。它复用相同BGE与原文窗口，由本机Qdrant1.19.2生成BM25稀疏向量，使用多语言分词、禁用词干化、空停用词集，再由原生DBSF合并语义和词项候选。两个分支与最终查询均保留主体、命名空间、来源日期/角色等过滤条件。实体关联继续由Mem0原有语义索引维护。

固定Mem0的关键词加分路径只对已有语义候选评分，因此本候选在单个provider实例上替换索引/查询方法，保留Mem0的ID、正文、历史、实体关系和应用journal；没有修改全局SDK factory。原生DBSF得分用同一比例缩放至Mem0允许的范围，保留融合排序，避免SDK将大于1的分数截成并列；不会再重复添加可选FastEmbed关键词分数。API复用原认证客户端，文本仅送到本机Qdrant。

schema3绑定记录检索配置SHA256，主collection还校验BM25/IDF配置。新增、更正、迁移和恢复均按规范正文生成词项索引；恢复不会直接沿用快照中的BM25向量。原BGE余弦一致性检查保留，写入后核对正文、稠密向量及稀疏结构。已有schema1/2绑定继续用原配置，升级命令为：

```bash
python3 tools/project.py reindex-memory --data-dir /实际应用数据目录 \
  --profile bge-int8-windows-bm25-v1
```

支持先前已完成窗口升级的档案再次升级；未发布的新升级可撤销并恢复上一份完成标记，已发布的升级继续续跑。工具拒绝降级索引。不要在正在使用该数据目录的应用旁执行升级；退出应用后重复相同命令即可续跑。

[实际模型集成](../../benchmarks/results/hybrid-memory-integration.json)已通过8条长资料的ID/原文/片段召回、公开保存及跨会话回忆、更正、旧备份恢复、主体隔离、遗忘及实体清理。实际服务回归另验证语义前60之外的词项候选、过滤条件、连续升级撤销/续跑、旧备份不复活，以及同配置恢复时重建损坏的词项向量。完整故障重复、10k正式时延、开发质量对照与独立JT验收继续进行，暂未更改默认配置。

`prepare.py`校验官方ARM64 musl归档SHA256、固定Qdrant 1.19.2及源码commit。程序和源码分别位于`~/AI_Stack/installed/qdrant-1.19.2/`、`~/AI_Stack/frameworks/qdrant/`，生产入口使用项目的systemd用户单元`mmassistant-qdrant.service`，未修改系统包。

```bash
python3 deploy/memory/prepare.py
.venv/bin/python tools/qdrant_capacity_compare.py --tier 10000 \
  --run-dir "$HOME/AI_Stack/runtime-data/benchmarks/qdrant-new-comparison" \
  --baseline-run "$HOME/AI_Stack/runtime-data/benchmarks/memory-capacity-10k-v1" \
  --baseline-report benchmarks/results/memory-capacity-10k-baseline.json \
  --output benchmarks/results/qdrant-new-comparison.json
```

评估进程只监听指定127.0.0.1端口，关闭gRPC、集群、CORS和telemetry，检查监听socket确实归该进程所有。配置使用该构建支持的YAML后缀，环境变量再次限定回环地址；缺失/错误配置导致额外监听时拒绝导入数据。未知目录和占用端口不会被接管。进程由评估命令创建并在结束时关闭，测试数据均为合成数据。

原生服务配置2个API worker、2个查询线程和1个优化CPU预算。HNSW `m=0`保留精确扫描；为`user_id/namespace/key/memory_type/operation_key/derived_from`建立原生keyword索引。Mem0补充其标准主体索引。点的ID、向量、payload从关闭的10k基线经Qdrant公开API导入，history和journal经SQLite backup复制；排除旧的50条计时写入及其回执，确保新的计时写入实际执行。导入器只允许本项目的合成容量目录，不是通用用户备份恢复功能。

[10k对照](../../benchmarks/results/qdrant-native-10k-comparison.json)查询/完整写入P95为47.5/56.6ms；100个查询的目标命中布尔结果与基线一致，仍为86/100，不能据此宣称独立记忆质量通过。[原始容量数据](../../benchmarks/results/qdrant-native-10k-comparison-capacity.json)保留逐次结果、索引配置、额外服务RSS、数据导入校验和后台状态。

[恢复检查](../../benchmarks/results/qdrant-recovery-smoke.json)使用真实BGE、Mem0、SQLite和本机Qdrant，保存、更正、遗忘确认后及向量提交后/history失败四种阶段各10次SIGKILL，共40次。验证已确认状态、主体隔离、派生摘要清除、历史清除、删除后不复活及部分提交幂等修复。history失败为明确标记的回调故障注入；这不代替整机断电或完整备份恢复。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv/bin/python tools/qdrant_recovery_smoke.py
```

[50k档](../../benchmarks/results/qdrant-native-50k-comparison.json)已完成：查询/写入P95=61.6/59.2ms，编号记录命中94/100，无查询错误、主体泄漏或未确认写入。它使用真实BGE及SDK批量准备，精确扫描，全部计时操作经过完整Mem0适配器。它仍是隔离容量基准；认证/绑定后的公开入口另做功能验证，不把这些组合为完整JT验收。

## 受管理的应用入口

```bash
python3 tools/project.py up --components qdrant,llama
python3 tools/project.py status --components qdrant
python3 tools/project.py interact --no-vision
```

`runtime.py`生成运行目录`qdrant/`内的deployment、私有认证文件及YAML配置；凭据不进入仓库、命令行或服务环境。SDK使用显式`trust_env=False`客户端，并同步检查锁定的服务器版本。服务启动前校验官方程序SHA256，只监听回环地址，关闭gRPC、集群与telemetry；配置有本地修改时不会覆盖。

`app/memory_store.py`通过Qdrant原生collection metadata核对服务实例、数据目录、embedding指纹及向量维度，使用数据目录及collection两个写入锁。首次配置可从`PREPARING`恢复，确认SDK初始化与删除屏障恢复后标记`READY`；已READY但collection丢失时拒绝创建空替代库。每次应用存储操作核对绑定。Mem0继续拥有事实、向量、history，应用journal继续承接幂等与删除屏障。

旧的嵌入式目录被明确拒绝自动切换；当前主运行目录此前没有用户数据库，部署时未迁移或覆盖用户内容。显式迁移入口见下文。[一致备份及恢复入口](../data/README.md)已加入，恢复合并当前删除/版本/会话/提醒控制状态并切换目录；最新控制库须保留，未完成写入须先修复。整盘丢失、配额和故障的完整矩阵仍未验收。

显式遗忘还通过`patches/mem0/purge_entities.py`清理Mem0实体索引：按主体及记忆ID分页查询，移除已删除事实的关联；没有剩余关联的实体一并删除，其他主体及仍有效的关联保留。上游在实体索引尚未加载时跳过清理，并抑制清理异常，因此适配器在重启后也直接使用Qdrant公开API检查实体集合。清理失败时删除屏障保持待处理，重启后重试成功才确认完成。真实SDK、Qdrant及SQLite回归覆盖重启清理、共享关联保留、主体隔离和故障恢复；测试embedding为明确标记的fixture，不证明完整备份恢复或磁盘级数据擦除。

更正也复用严格实体清理：新事实写入后，旧实体关联清理成功才确认操作完成；若失败，操作保留PENDING，通过同一request ID重试修复。回归覆盖SDK重启后尚未加载实体索引、清理故障及更正重放，避免旧实体给新事实提供错误的召回加分。

## 旧库原地迁移

```bash
# 退出使用该数据目录的 interact/voice；Qdrant后台服务保持运行。
python3 tools/project.py migrate-memory --data-dir /实际应用数据目录
# 仅撤销尚未完成的迁移：
python3 tools/project.py migrate-memory --data-dir /实际应用数据目录 --abort
```

参数是包含`memory/`和可选`records.sqlite`的应用数据目录，与`interact --data-dir`一致。源必须是本项目Mem0 2.2.1的`memory/qdrant`、`history.sqlite`和`operations.sqlite`，使用当前已验证的512维BGE ONNX模型；目标是配置中的受管理本机服务。更换embedding、搬到另一台机器或恢复任意备份不属于此命令的范围。

工具取得adapter、records及目标collection写入锁，通过Qdrant公开scroll/upsert/retrieve接口迁移事实及实体；SQLite history/journal留在原位置。保留事实ID、payload、版本和来源，逐批用真实BGE复算并检查保留向量的余弦一致性（≥0.99999），写入后读取核对payload、向量及总条数。未知collection、数据归属冲突、源内容改变或目标metadata不符会拒绝发布。

已有删除屏障和records中先于Mem0遗忘提交的删除意图会合并；派生事实一并排除，相关history/messages清除，实体关联裁剪，操作回执继续保留。源指纹覆盖事实ID/payload、SQLite逻辑内容和records删除记录，避免续跑时混入变更。此过程保留已有授权的删除意图，即使撤销迁移也不会撤销遗忘。

`store-transition.json`先记录迁移状态，检查完成后依次发布`store.json=READY`及`COMPLETED`。任一中间状态都阻止应用打开目录，包括绑定文件缺失的情况；重复原命令续跑同一collection。完成后重复命令只检查绑定，不再导入旧数据，因此不会覆盖新更正或复活迁移后已遗忘的事实。

`--abort`仅清除metadata完全匹配的未发布目标，保留原库及删除屏障；撤销自身也可中断续跑。完成后的迁移拒绝撤销。旧`memory/qdrant`目录会保留为停用副本；后续遗忘不会物理擦除该副本，也不得用它绕过当前删除日志恢复。完整备份/恢复和旧副本保留策略仍待交付。

[真实模型迁移报告](../../benchmarks/results/memory-migration-smoke.json)覆盖公开CLI、ID/版本/来源保持、语义召回、主体隔离、回执重放和迁移后遗忘/重启；输入是隔离的合成资料。[迁移专项测试](../../tests/test_memory_migration.py)另外覆盖分页、四个提交阶段中断、续跑、源变更、模型不匹配和撤销。它们不能替代独立记忆质量、整机断电或完整备份恢复验收。

[真实进程恢复报告](../../benchmarks/results/memory-migration-recovery-smoke.json)在标记写入、批次提交、核对完成、READY发布后分别执行一次真实SIGKILL，共4次；全部通过公开CLI续跑，确认事实只保留一份，旧源中的已遗忘事实被删除屏障排除。该检查使用真实BGE/Mem0/SQLite及认证Qdrant，但每阶段仅一次，尚不满足完整故障矩阵的每类10次要求。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/memory_migration_smoke.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/memory_migration_recovery_smoke.py
```
