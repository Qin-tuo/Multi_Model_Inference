# 当前语音部署

使用固定 OpenVoiceStream `18a459761647a641926f04d7d36c17035c11f8b7`，通过上游后端注册接口接入 Qwen3-ASR-0.6B 的本地 llama.cpp 服务。TTS 使用本机构建的 Matcha 三个 BF16 estimator 和 FP16 Vocos TensorRT 引擎。主模型另由 8088 端口的 llama.cpp 运行。

| 环节 | 所有者 | 当前格式/行为 |
|---|---|---|
| 输入与 VAD | Reachy 应用、Silero v6 | 16 kHz 单声道 PCM16；当前默认文件输入 |
| ASR | llama.cpp 8089，经 OVS 8621 接口调用 | 整句识别；不声明流式 partial |
| TTS | OVS Matcha/Vocos | 16 kHz 单声道 PCM16 |
| 会话、取消和输出 | Reachy TurnController | 单轮所有权；文件回复或后续实物播放 |

统一启动命令见 [systemd 部署](../systemd/README.md)。`01_src/deploy/speech/serve.py` 从冻结的有效配置生成 OVS profile，不再读取旧 `configs/runtime.yaml` 或导入旧 `tools.project`。ASR 术语提示在 `backends.asr_context`，当前沿用已有值，尚未做性能或质量调优。

运行工件位于 `项目/05_runtime/models/matcha-host`，对应文件哈希见 [runtime-artifacts.lock.json](../runtime-artifacts.lock.json)。OVS 与引擎来源见 [source-lock.json](source-lock.json)，Python 依赖见 [requirements.lock](requirements.lock)。上游 TRT 10.3 的序列化引擎不能作为本机 TRT 10.16 的可运行产物；当前使用本机已有、校验过的构建。

[历史移植和构建记录](../../../docs/archive/README-speech-legacy.md)保留最初的平台差异、失败输入与适配原因。旧构建配方依赖旧工具，应通过固定回退源码读取；当前管理入口只验证、运行准备好的工件，不隐式构建、安装或下载。

中文文件闭环和受控 PCM 打断已取得真实模型证据，见 [语音打断报告](../../../benchmarks/results/reachy-r3-voice-summary.json)。实物麦克风/扬声器、声学 AEC、双讲和实际停播按用户决定后置，不能用文件模式代替其验收。

清理后的当前目录保留 OVS 启动、ASR 注册、CUDA/Piper 兼容层和源版本锁。`source-lock.json` 中的 `host_build_recipe` 是构建发生时的历史路径；旧配方依赖 `tools.project`，应从固定旧运行快照执行。原文件及旧构建报告可按 [归档索引](../../../docs/archive/repository-cleanup-20261019.json) 恢复。
