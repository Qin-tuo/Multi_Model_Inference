# 本机 Qdrant 服务

公开文字和语音入口默认使用 Mem0 2.2.1 + 本机Qdrant服务。原Qdrant local仍用于兼容SDK测试和历史基准。10k嵌入式基线的查询/写入P95为350/906ms，主要耗时在过滤和查询。这里评估同一SDK连接本机原生Qdrant服务，不更换BGE或事实事务边界。

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

旧的嵌入式目录被明确拒绝自动切换；当前主运行目录此前没有用户数据库，因此本次只建立服务及新库入口。既有数据迁移、完整备份/删除日志合并、权限和故障的完整矩阵仍待实现。`StoreBinding`拒绝目录复制或模型指纹变化不等于已经具备恢复工具。
