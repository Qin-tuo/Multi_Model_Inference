# 历史部署说明（Reachy 切换前）

以下命令仅适用于旧源码快照，不用于当前分支。旧源码保留在 `/home/nvidia/AI_Stack/archives/legacy-runtime-972de50`。

# 本机后端生命周期

本项目使用systemd用户级单元管理进程、重启和cgroup清理。`deploy/services.py`只生成配置、调用systemctl和检查现成HTTP/ROS接口；模型、相机和语音服务仍使用其上游入口。

```bash
python3 tools/project.py install-services
python3 tools/project.py up --components camera
python3 tools/project.py status --components camera
python3 tools/project.py logs --component camera --lines 30
python3 tools/project.py down --components camera
```

可选组件为`qdrant,llama,asr,speech,camera,vision,pose`，逗号分隔或`all`。`speech`需要`asr`，`vision/pose`需要`camera`；启动按依赖顺序等待就绪。`up`检查实际HTTP状态或新的ROS消息，失败时停止本次新启动的服务，保留调用前已有服务。`down`同时停止所选依赖的消费者，核对主PID和整个cgroup均已退出。

单元名为`mmassistant-*.service`，生成文件位于运行数据目录的`systemd/`，通过链接注册到`~/.config/systemd/user/`。安装时先验证临时文件，再替换本项目拥有且未被本地修改的配置；同名外部单元和用户修改不会被覆盖。当前没有设置开机自动启用。直接前台运行的`serve`进程由对应终端管理；受管理组件发现已有非本单元的同类进程或端口会拒绝接管。

重启策略为失败后等待5秒，120秒窗口最多3次启动。停止使用`KillMode=control-group`，先发SIGINT，20秒未退出则由systemd终止余下进程。主进程退出并不被当成整个后台进程组已清理。[systemd服务语义](https://github.com/systemd/systemd/blob/v255/man/systemd.service.xml)、[cgroup关闭语义](https://github.com/systemd/systemd/blob/v255/man/systemd.kill.xml)。

模型入口持有共享文件租约，重型ORT构建持有同一文件的独占租约。租约跨exec保留，实际进程退出后释放；另检查旧式构建/模型进程。这样限制安装期编译与模型驻留重叠，不能据此宣称运行时多模型容量已经验收。

后端日志位于`runtime-data/logs/backends/`，可通过`project.py logs`读取。它们使用Python标准库RotatingFileHandler，5MiB轮转阈值、4份备份、最多64KiB输入日志块；systemd继续负责实际监督，日志转发器不自行重启子进程。该设置不替代实验原始trace的单独配额。

`backend_status=READY`只表示所选后端进程和接口已就绪。报告始终另列`assistant_acceptance=NOT_VERIFIED`；模型质量、实际GPU节点位置、物理语音和JT-01～JT-13需各自证据。GPU版ORT已完成构建，包括Qdrant在内的七个后端同时就绪的记录见[全后端报告](../../benchmarks/results/full-backend-readiness.json)。持续视觉期间的5次文件语音问答已通过短时共存初测，完整负载、8小时稳定性和物理语音仍待验证。

验证见[生命周期报告](../../benchmarks/results/service-lifecycle-smoke.json)：真实相机生命周期、带标记的就绪故障注入、拒绝SIGINT的父子进程强制清理、真实systemd重启限流。此检查不是8小时或完整故障矩阵验收。

Qdrant使用6333回环端口，验证凭据访问成功、匿名访问拒绝以及监听进程归属。内部认证文件保存在运行目录`qdrant/`，不写入unit或日志。数据目录绑定与遗留数据约束见[记忆服务](../memory/README.md)。
