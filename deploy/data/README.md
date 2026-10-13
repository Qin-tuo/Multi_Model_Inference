# 应用数据备份与恢复进度

现已提供一致备份、校验及恢复命令，复用Qdrant原生快照、SQLite backup和APScheduler恢复。恢复须提供当前控制数据库，在应用启动前合并删除、更正、会话及提醒状态；旧目录随后停止接受应用访问。

```bash
# 先退出使用该应用数据目录的 interact/voice/notifications，Qdrant服务继续运行。
python3 tools/project.py backup-data \
  --data-dir "$HOME/AI_Stack/runtime-data" \
  --output "$HOME/AI_Stack/backups/assistant-20261013-01"

python3 tools/project.py verify-backup \
  --backup "$HOME/AI_Stack/backups/assistant-20261013-01"
```

`--data-dir`须与实际`interact --data-dir`一致，包含已绑定的`memory/`。如使用默认入口，可传`$HOME/.local/share/multi-model-assistant`；它在本机指向AI_Stack运行数据目录。备份目录须与应用数据目录分离，每份完成的备份不可覆盖。未完成备份可用完全相同的命令续跑；续跑重新采集当时的一致状态，保留backup ID并更新采集时间。

备份包含：

| 文件 | 内容 |
|---|---|
| `memory.snapshot`、可选`entities.snapshot` | Qdrant原生collection快照，包括事实、向量、metadata及实体关联 |
| `memory/operations.sqlite` | 幂等回执、主体generation、删除屏障和取消来源 |
| `memory/history.sqlite` | Mem0历史及SDK消息 |
| 可选`records.sqlite` | 会话、笔记、记忆引用和删除记录 |
| 可选`reminders.sqlite` | APScheduler job store、提醒、通知回执及已有主动反馈策略 |
| `store-binding.json`、`manifest.json` | 目录/实例/embedding指纹、版本、文件校验、逻辑数据哈希和完成状态 |

工具获取memory、records、reminders及collection的排他锁。数据库不存在时记录缺席，不创建空数据库冒充已有数据；memory的journal和history必须存在。SQLite backup会包含已提交但仍在WAL中的内容，文件权限为0600。快照经认证的回环HTTP下载，校验服务返回的长度和SHA256；不复制模型、engine、原始音视频或Qdrant认证文件。

数据复制前后核对SQLite逻辑内容、原生collection逐点内容及数据清单。只有文件校验和源一致性核对全部通过，manifest才标记`COMPLETE`。`verify-backup`检查固定文件清单、SHA256、SQLite完整性及逻辑内容；不会把不完整目录视为可用备份。备份使用过程中保持不可变，应用运行数据的保留策略不等于自动删除旧备份。

Qdrant 1.19.2拒绝通过file URI读取其snapshot目录之外的文件，原始[失败证据](../../benchmarks/results/snapshot-roundtrip-baseline.json)已保留。恢复使用官方SDK公开的snapshot upload接口，只上传到不存在的新collection；不放宽服务路径限制，不覆盖原库。恢复续跑会核对已上传collection的身份及内容，不因观察超时直接重建或覆盖。

已收到名称的服务端临时快照写入备份回执，正常结束及续跑只清理这些已确认属于本次备份的文件。若进程在服务器生成快照、客户端收到其名称之前终止，可能留下无法确认归属的服务端临时快照；当前不自动删除未登记文件，也未完成备份配额/保留策略验收。

## 恢复与目录切换

```bash
python3 tools/project.py restore-data \
  --backup "$HOME/AI_Stack/backups/assistant-20261013-01" \
  --current-data-dir "$HOME/AI_Stack/runtime-data" \
  --data-dir "$HOME/AI_Stack/assistant-restored"

python3 tools/project.py interact --no-vision \
  --data-dir "$HOME/AI_Stack/assistant-restored" --user owner

# 仅撤销尚未启用的新目录，保留源数据库和备份：
python3 tools/project.py restore-data --abort \
  --data-dir "$HOME/AI_Stack/assistant-restored"
```

三个目录须彼此分离。`--current-data-dir`必须是当前活动档案，保有完整的最新SQLite控制数据库，并与备份属于同一数据沿革。当前活动档案的embedding指纹须匹配其锁定配置。旧索引配置的备份会先在私有临时collection核验，再依据当前删除/版本控制重建为当前配置；不会将旧格式向量直接混入新索引。当前控制库缺失、损坏或早于备份时明确失败，不能只凭一份旧备份推断后来发生的遗忘和播报。存在未完成的remember/correct时，当前实现要求先用原请求重试并确认持久结果，再启动恢复；该部分恢复矩阵仍待扩展。

恢复以当前journal/history、会话/笔记及提醒数据库为控制依据，保留备份之后的笔记、清除状态、幂等回执、取消来源及通知结果。事实从原生快照恢复，按当前journal核对主体、版本和操作ID；当前向量库中较新的更正或新增事实会合并进目标。每批向量经过当前BGE复算校验。同一已确认版本的当前向量缺失或损坏时可采用备份中的有效向量；若较新的确认版本在当前库和备份中都不存在，则拒绝发布。

多向量档案的主索引和实体索引均校验MaxSim结构及profile，备份必须包含其实体快照。旧单向量备份跨配置恢复时，已遗忘的记录和无效实体关联在重新嵌入前排除，当前较新版本从当前有效库补入。临时快照集合记录归属并在完成或撤销时清理，未核验的上传结果须先续跑核验。真实旧备份→当前多向量、更正版本保留及遗忘不复活见[集成报告](../../benchmarks/results/window-legacy-backup-integration.json)；临时快照阶段续跑和撤销另有框架回归，完整跨配置故障矩阵仍待扩展。

显式启用的`bge-int8-windows-bm25-v1`候选使用schema3检索绑定。恢复到该配置时，不论备份是旧格式还是同格式，均从通过当前版本/删除校验的规范正文重新生成BM25；不会恢复已损坏或过时的词项向量。BGE稠密向量仍须通过原有严格重算核验。[候选实际集成](../../benchmarks/results/hybrid-memory-integration.json)已验证旧备份、更正与遗忘；损坏BM25的同格式恢复另有实际Qdrant回归。完整候选故障重复和整机验收尚未完成。

跨配置恢复的[首轮故障测试](../../benchmarks/results/window-restore-fault-first.json)在旧快照暂存、快照核验完成、合并完成和旧源退休四个阶段各实际SIGKILL一次，全部续跑成功。每次均核对当前版本、新事实、主人/访客隔离、遗忘及派生内容/历史清除、提醒不重复；另验证实际索引丢失后的恢复。本轮使用`--skip-conversation`，没有生成恢复后的模型回答，也没有物理播报。 随后的[40次扩展验证](../../benchmarks/results/window-restore-fault-recovery.json)已完成，四个阶段各10次全部通过，保持相同数据校验与明确的测试范围。

```bash
.venv/bin/python tools/data_restore_smoke.py --legacy-backup --skip-conversation \
  --repeats 10 --output benchmarks/results/window-restore-fault-recovery.json
```

删除屏障和会话删除记录共同排除旧事实及派生摘要，并清除history/messages、实体关联及相关会话正文。实体索引缺少事实版本来源，因此恢复会失效更正过的事实关联；事实ID、正文和向量保留，其他有效实体关联继续使用。SDK正常更正路径也补充了严格实体清理，失败保留待重试状态。

提醒采用当前的取消、已输出和结果不确定状态。复制的APScheduler jobs先清空，应用启动时由现有ReminderService根据有效提醒重建任务，避免反序列化旧任务或写回旧路径。先前处于playing且没有完成回执的通知按既有规则变为uncertain，不自动重复播报。复制过程不会启动scheduler。

`data-transition.json`记录两边状态：新目录未完成时禁止访问，源目录冻结后也禁止启动；检查通过后先将源目录标记RETIRED，再允许新目录ACTIVE。记忆、会话和独立提醒入口均在取得自身写锁后检查该状态。中断时重复完全相同的恢复命令；已启用的新目录重复执行不会回放旧快照。未完成恢复可用`--abort`撤销，已启用的新目录不可回退。撤销保留未启用目标目录的诊断文件，不覆盖源控制数据。

恢复不会移动模型或Qdrant服务目录，旧数据库和备份也会保留；日常命令须指定新`--data-dir`。不要删除或手工修改状态标记来绕过目录检查。后端服务可能仍位于原运行根目录，不能把整个原根目录当作废弃数据删除。

当前验证范围是本机已注册服务、保留最新控制库情况下的恢复与索引重建。整盘丢失且没有最新删除/通知状态、磁盘级擦除、配额/保留策略及完整J6/JT故障验收仍未完成。

[真实模型备份报告](../../benchmarks/results/data-backup-smoke.json)使用实际BGE/Mem0/受管理Qdrant及四份SQLite数据库，通过公开CLI完成备份与校验，核对原生快照逐点哈希及SQLite逻辑哈希。数据库复制、快照下载和最终发布前各10次真实SIGKILL，共30次，均能拒绝半成品并续跑完成。源码及数据为隔离的合成测试资料，没有启动还原后的交互应用。

[旧备份状态基线](../../benchmarks/results/old-backup-controls-baseline.json)在备份后遗忘一个事实、取消一个提醒，再检查旧快照，确认其仍包含旧事实和active提醒。该原始失败保留作为对照；恢复入口通过当前控制库、删除清理及目录切换处理这个问题。

[真实恢复报告](../../benchmarks/results/data-restore-smoke.json)使用实际BGE/Mem0/Qdrant/SQLite/APScheduler和公开backup/restore/interact命令。验证了备份后的更正与新增、遗忘及派生内容清除、历史/会话清除、访客隔离、最新笔记保留、提醒取消/完成/uncertain状态、新路径任务及旧入口禁用。真实文字问答读取更正后的偏好，实际删除原collection后也能从新备份重建索引。冻结源、合并完成、退休旧源三个阶段各10次SIGKILL，共30次，均续跑成功且无重复提醒。提醒完成回执为明确标记的fixture，不是扬声器播放证据。

[恢复专项回归](../../tests/test_data_restore.py)另外覆盖新版本丢失拒绝、损坏/缺失向量修复、通知状态回退拒绝、六个切换检查点、来源变化、不同沿革、撤销及连续多次目录切换。完整框架回归为165项通过；这不改变整机JT仍待验收的状态。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/data_backup_smoke.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/data_restore_smoke.py
```
