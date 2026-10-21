# 运行操作

## 运行依赖

本机为 aarch64、Python 3.12、CUDA 13.2、TensorRT 10.16.2.10。应用和语音环境分别位于 `05_runtime/environments/edge-assistant`、`speech`，依赖已安装。CUDA、TensorRT 和设备驱动由主机提供。

`05_runtime` 只保留：

- `builds/llama/bin`：当前 llama-server 与必需动态库；软链接和它们的真实目标都属于运行依赖。
- `environments`：两个运行环境。
- `frameworks`：OpenVoiceStream 服务文件、视觉预后处理辅助代码和 TensorRT 缓冲区辅助代码。
- `models`：四个 GGUF 文件、十个视觉引擎、语音模型/词典与 VAD。
- `runtime-data`：笔记/提醒数据库、身份图库、语音输出、服务配置、运行锁和日志。

没有旧 AI_Stack、备份包、恢复副本或开发工具目录。普通 Git 克隆不含 `05_runtime` 大文件。

## 启动与停止

```bash
./run.sh config
./run.sh services check
./run.sh services up
./run.sh services status
./run.sh services logs --components app --lines 80
./run.sh services down
```

配置优先级：显式 `--config`、`EDGE_ASSISTANT_CONFIG`、默认 `02_config/settings.orin-nx.yaml`，基础字段在 `02_config/settings.yaml`。四服务就绪显示 `READY`；停止后确认状态为 `STOPPED`。

服务归属和进程组由 systemd 管理。曾出现过间歇性退出超时；若再次发生，检查 `services status` 与相应服务日志，确认进程是否退出。不要把一次启动就绪当作停止检查通过。

## 浏览器与文件语音

板端地址为 `http://127.0.0.1:8090/`。在远程电脑运行：

```bash
ssh -L 8090:127.0.0.1:8090 nvidia@<板端IP>
```

然后打开同一地址。主页面支持对话、相机和音频文件；`/debug` 显示实时帧率、帧龄、模型调用和板端资源。历史基线表、观测导出与离线分析流程不属于当前部署版本。

当前默认采用文件音频输入/输出。ReSpeaker Lite 实体录播、AEC 与双讲尚未验收，不能据文件链路推定这些能力已通过。

命令行可检查文字或图像推理：

```bash
./run.sh ask "你好" --json-output
./run.sh ask "描述这张图片" --image /path/to/image.jpg --json-output
```

完整对话、工具、取消和语音输出通过 Web 应用完成。
