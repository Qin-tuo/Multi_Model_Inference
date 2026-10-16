# Reachy 软件服务部署

当前部署使用 systemd 用户服务。Reachy 应用负责会话、相机、感知和语音数据流；llama.cpp 与 OpenVoiceStream 负责推理；systemd 负责启动顺序、重启和进程组清理。

| 组件 | 实际入口与职责 | 默认端口 |
|---|---|---|
| `llama` | 原生 llama-server，Qwen3.5-4B 图文理解 | 8088 |
| `asr` | 原生 llama-server，Qwen3-ASR-0.6B | 8089 |
| `speech` | OpenVoiceStream，ASR 适配与 Matcha/Vocos TTS | 8621 |
| `app` | Reachy Web、单一相机采集、人体/脸/手、VAD、整轮控制、笔记/提醒 | 8090 |

当前软件配置采用语音文件输入/输出，不要求接入麦克风或扬声器。相机使用已连接的 D435i。物理录播、AEC、双讲与实际停播继续待验。

## 操作

在当前 Reachy 源码目录执行；使用独立环境，不使用旧 `.venv`：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services check
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services up
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services status
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services logs --components app --lines 50
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services down
```

`up --components app` 按依赖启动全部四个组件；`down --components app` 只停止应用，三个模型服务保持驻留。停止 ASR 会先停止其语音服务与应用消费者。当前不设置开机自启。

`up` 加载 `config/settings.yaml` 和平台 overlay，或显式 `--config PATH`，把完整有效配置冻结为 `~/AI_Stack/runtime-data/reachy-headless/systemd/settings.json`。所有服务使用同一份冻结配置。运行时更换配置或源码入口会被拒绝，应先 `down` 再 `up`。`status/down/logs/wait` 默认读取已安装的冻结配置；`up` 默认重新读取源码配置。

新单元为 `reachy-headless-{llama,asr,speech,app}.service`，生成文件位于上述 systemd 目录，以链接注册到 `~/.config/systemd/user`。旧 `mmassistant-*` 不由此命令接管；启动前必须明确停止旧项目持有的端口和相机。占用冲突会报告并退出，不终止外部进程。

## 运行约束与状态口径

- 安装时先用本机 `systemd-analyze verify` 检查单元，拒绝覆盖同名外部单元或已被人工修改的托管单元。部署写操作使用文件锁串行执行。
- 就绪同时检查单元归属、实际监听进程、模型别名、上下文/槽位、语音能力及应用有效配置。应用还须有新鲜相机帧和已启用感知/语音模块；PID 存在不等于就绪。
- 依赖等待使用 `ExecStartPre`；失败重启等待 5 秒，120 秒内最多启动 3 次。停止向整个 cgroup 发 SIGINT，30 秒未退出由 systemd 强制收尾，随后核验无残留进程。
- 启动保留旧部署的 2 GiB 可用内存下限。这只是启动准入条件，不能代替峰值资源测量或共存容量验收。未修改推理线程、并发、量化、上下文或缓存参数。
- 模型入口持有共享构建租约，避免与持独占租约的重型 ORT 构建重叠。运行时不自动下载模型。
- 模型清单、感知/VAD 锁及 `deploy/runtime-artifacts.lock.json` 校验现有权重、原生程序/库、TTS 工件和感知源码。OVS 源码要求固定提交且无未记录改动。全量校验会读取数 GiB 文件，请等待命令完成。
- 日志位于 `~/AI_Stack/runtime-data/reachy-headless/logs`，每组件按 5 MiB 轮转并保留 4 份备份。实验 trace 和测试输出有各自目录。

`READY` 仅表示当前软件配置就绪；真实手势质量、物理声音、B0 性能和长稳分别需要证据。离线环境变量与本地工件校验不等于已经完成断网启动或干净机器重部署验收。

## 切换与回退

旧源码固定在 `/home/nvidia/AI_Stack/archives/legacy-runtime-972de50`，独立环境、模型和用户数据保留。演练脚本先核对旧七服务归属和就绪，切换到新四服务，测试文字/画面/文件语音及完整应用循环，最后恢复旧七服务并检查新鲜 RGB 与对齐深度：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_deployment.py \
  --legacy-root /home/nvidia/AI_Stack/archives/legacy-runtime-972de50 \
  --output /home/nvidia/AI_Stack/runtime-data/reachy-deployment-new-run --cycles 20
```

输出目录必须是新目录。该脚本会临时中断本项目服务；结束时保留旧系统运行。演练记录不能替代后续正式切换状态。

正式回退顺序：当前入口执行 `main.py services down`，再到旧快照执行 `python3 tools/project.py up --components all`。旧管理器会将旧单元入口指向快照，避免回退依赖被更新的主目录。恢复后必须检查旧服务状态和 `scripts/probe_legacy_camera.py` 的新鲜 RGB/深度，不能只看进程。

[旧部署文档](README-legacy.md)只解释历史方案。当前准备工件的可运行性与干净环境重建是两个验收范围；完整重部署仍在 R6。
