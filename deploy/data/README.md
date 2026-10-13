# 应用数据备份与恢复进度

现已提供一致备份及校验命令，复用Qdrant原生快照和Python SQLite backup接口。完整恢复仍需合并当前删除、会话清除和提醒回执，并完成应用目录切换；当前不提供把旧快照直接发布为活动应用的命令。

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

Qdrant 1.19.2拒绝通过file URI读取其snapshot目录之外的文件，原始[失败证据](../../benchmarks/results/snapshot-roundtrip-baseline.json)已保留。还原检查使用官方SDK公开的snapshot upload接口，只上传到不存在的新collection；不放宽服务路径限制，不覆盖原库。此接口用于验证快照可还原，不代表已经完成产品恢复流程。

已收到名称的服务端临时快照写入备份回执，正常结束及续跑只清理这些已确认属于本次备份的文件。若进程在服务器生成快照、客户端收到其名称之前终止，可能留下无法确认归属的服务端临时快照；当前不自动删除未登记文件，也未完成备份配额/保留策略验收。

完整恢复的必要规则包括：先验证当前删除记录和主体归属；删除旧事实、派生摘要、历史和关联会话；核对备份后更正/新增的版本，无法恢复时明确失败；保留已完成、已取消和输出结果不确定的提醒状态；完成新目录绑定并阻止旧目录再次运行后，才能允许新实例开始交互。这些步骤仍在实施，不能仅凭快照上传成功宣称恢复完成。

[真实模型备份报告](../../benchmarks/results/data-backup-smoke.json)使用实际BGE/Mem0/受管理Qdrant及四份SQLite数据库，通过公开CLI完成备份与校验，核对原生快照逐点哈希及SQLite逻辑哈希。数据库复制、快照下载和最终发布前各10次真实SIGKILL，共30次，均能拒绝半成品并续跑完成。源码及数据为隔离的合成测试资料，没有启动还原后的交互应用。

[旧备份状态基线](../../benchmarks/results/old-backup-controls-baseline.json)在备份后遗忘一个事实、取消一个提醒，再检查旧快照，确认其仍包含旧事实和active提醒。这是完整恢复尚未完成的明确失败项，后续必须用当前删除和通知状态覆盖。备份成功及传输校验不能代替这一产品约束。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python tools/data_backup_smoke.py
```
