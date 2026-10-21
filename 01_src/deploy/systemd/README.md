# 四服务部署与状态口径

`app` 管理相机、视觉、会话、语音数据流及 Web；`llama` 提供 4B 接口，`asr` 承载 Qwen3-ASR，`speech` 提供 OpenVoiceStream 语音 API。默认端口分别为 8090、8088、8089、8621。

```bash
./run.sh services check
./run.sh services up
./run.sh services status
./run.sh services logs --components app --lines 80
./run.sh services down
```

四种入口统一读取 `02_config/settings.orin-nx.yaml`，显式配置与环境覆盖优先。`up` 读取源码配置并保存完整生效配置到 `05_runtime/runtime-data/edge-assistant/systemd/settings.json`；`status/down/logs/wait` 默认使用已安装配置。更新配置前先停止服务。

`up --components app` 会启动它的后端依赖；`down --components app` 只关闭应用。停止提供者时管理器先停止依赖它的消费者。每个单元都检查源码目录、冻结配置和进程归属；不会接管外部服务或占用端口的陌生进程。

## 所有权与恢复

服务名为 `edge-assistant-{llama,asr,speech,app}.service`。生成单元位于项目运行数据目录，并链接到用户 systemd。操作使用文件锁串行化；模型服务持共享构建租约，重型构建/验证持独占租约。

`READY` 同时检查原生监听进程、模型/语音能力、有效配置和新鲜相机状态。`STOPPED` 要求受管理进程组为空，正常停止与超时后被 systemd 回收分别记录。`B2-EXIT-01` 的原应用退出超时仍开放；工具保存停止/恢复快照，不用一次成功退出证明偶发问题已修复。

日志在 `05_runtime/runtime-data/edge-assistant/logs/`。应用状态、动作数据库、图库和语音文件有明确所有者；运行包不携带这些可变业务数据。更换项目目录前先停止原实例，按管理器的归属检查重新注册。

当前保持文件语音配置，不设置开机自启。物理声音、真人准确率、长稳和性能目标分别验收。运行包独立恢复与源码交付状态见 [执行清单](../../../benchmarks/manifests/consolidation-progress.json)。
