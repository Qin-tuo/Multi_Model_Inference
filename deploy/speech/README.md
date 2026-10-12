# 首版语音服务的准备与兼容性核验

固定 **Qwen3-ASR 0.6B + Matcha zh/en TTS + Vocos**，使用 OpenVoiceStream 的独立语音服务。这里的准备脚本没有启动语音推理，J0.5 尚未通过。

## 已核验的上游

| 项目 | 固定值 |
|---|---|
| OVS 源码 | `18a459761647a641926f04d7d36c17035c11f8b7` |
| Jetson engine 子模块 | `e1d8d507300e329642937b8461ce03bbd2bd3a16` |
| 语音 compose | `deploy/docker-compose.edgellm-v091-voice.yml` |
| 语音 profile | `jetson-edgellm-v091-matcha` |
| compose 服务集合 | 只有 `speech` |
| ASR | `jetson.trt_edge_llm`，thinker INT4 B1 与 audio encoder |
| TTS | `jetson.matcha_trt`，CPU ONNX encoder、三个 BF16 estimator step engine、FP16 Vocos |
| 播报格式 | 16 kHz、单声道、PCM16；实际 capabilities 需启动后复核 |

出处：[固定版 compose](https://github.com/Seeed-Solution/openvoicestream/blob/18a459761647a641926f04d7d36c17035c11f8b7/deploy/docker-compose.edgellm-v091-voice.yml)、[固定版 profile](https://github.com/Seeed-Solution/openvoicestream/blob/18a459761647a641926f04d7d36c17035c11f8b7/configs/profiles/jetson-edgellm-v091-matcha.json)。模型仓库 revision、payload SHA256 与尺寸保存在 [source-lock.json](source-lock.json)，这些是上游产物的锁，并不证明产物适用于本机。

ASR 的内部实现使用 Edge-LLM fork；这是 OVS 的语音后端依赖。对话和图文主引擎仍采用 llama.cpp。独立 speech compose 不包含额外对话 LLM 或机器人 agent。

## 当前兼容性缺口

上游 v0.9.1 发布锁的目标是 **JP6.2 / L4T36.4.3 / CUDA12.6 / TensorRT10.3 / ORT1.23.2**。本机读到 **L4T39.2.1 / CUDA13.2.2 / TensorRT10.16.2 / Python3.12.3**。上游 compose 还挂载 `/usr/lib/python3.10/dist-packages/tensorrt`，本机没有该路径。

engine 子模块的 `engine-overlay/validate-manifest.py` 和 `verify-release-target.py` 明确拒绝其它目标元组。直接调用一键构建也不能完成本机移植；需要保留模型选择与补丁来源，生成本机目标 manifest、构建脚本和新产物锁。不可通过伪造版本环境变量让旧 engine 冒充本机产物。

当前 OVS installer 默认已经转向 v0.11 **CustomVoice candidate**。本项目必须明确选取固定 v0.9.1 Matcha 子集。发布锁记录 `voxedge=0.0.6a1`，当前固定版 Dockerfile 默认却是 `0.0.15a3`；本机移植时须依据实际验证的 backend API 固定 wheel 与哈希，不能同时宣称两个版本均为同一发布锁。

出处：[发布锁](https://github.com/Seeed-Solution/openvoicestream/blob/18a459761647a641926f04d7d36c17035c11f8b7/deploy/artifacts/v091-release-lock.json)、[引擎重建说明](https://github.com/suharvest/jetson-voice-engine/blob/e1d8d507300e329642937b8461ce03bbd2bd3a16/docs/reproduce-from-zero.md)。

## 可执行的准备命令

从项目根目录运行。源码保存在用户缓存，保留项目既有 references。

```bash
python3 deploy/speech/prepare.py --fetch --engine-source \
  --output "$HOME/.cache/multi-model-assistant/evidence/speech-source-inspection.json"
```

此命令只获取固定源码、初始化 Jetson engine 子模块并核验 profile/模型锁/服务集合。重复执行会复用源码，不下载模型或容器。输出 `needs_host_port` 代表已证实的兼容性缺口，不是推理失败或安装成功。

## 本机移植的执行顺序

1. 将固定 engine-overlay 的 manifest、目标验证和 CMake 配置复制到独立的本机构建层，记录每个改动；保留原版发布锁和 7 + 35 补丁来源。
2. 建立 Python3.12 或经过实测的隔离容器运行环境，固定 TensorRT Python binding、ORT 和 voxedge。容器中的运行库必须与 engine 构建所用 CUDA/TensorRT 实际匹配。
3. 下载固定 revision 的 Qwen3-ASR 与 Matcha 原始模型/ONNX 输入，验证哈希。旧发布 payload 可提供输入来源，旧序列化 TensorRT engine 不视为新平台兼容证据。
4. 在当前 Orin 上构建 ASR worker、plugin、audio encoder 和 thinker；编译任务数先限制为 2，避免与其它模型构建/加载争抢内存。
5. 使用 `models/matcha/split_matcha_trt.py` 和 `build_matcha_estimator_trt.sh` 对三个 split estimator 逐一构建；按 `build_matcha_engines.sh` 的 Vocos 段构建声码器。记录使用的 trtexec 参数、实际精度和峰值内存。
6. 建立本机 `speech-only` profile、镜像 digest 和 engine 哈希锁。仅启用 Qwen3-ASR 与 Matcha，不启用上游 v2v agent 会话控制。
7. 先验证文件 ASR、中文 TTS WAV/PCM，再验证麦克风、扬声器、AEC 和真实打断。音频设备由 Pipecat 音频 transport 唯一持有，语音容器无需 `/dev/snd`。

上游的 `build.sh --apply-only` 只负责 materialize 补丁源码，不是 engine 构建或推理验收。Matcha build 脚本需要真实 ONNX 产物，不能只修改目录名即视为可执行。

## 接口与验收证据

启动本机移植版后，先保存 `GET /readyz`、`/v1/models`、`/v1/capabilities` 的原始响应。文件测试使用 `POST /v1/audio/transcriptions` 和 `/v1/audio/speech`；ASR streaming 使用源码存在的 `WS /asr/stream`。如果使用私有 `/tts/stream`，其前 4 字节为 uint32 little-endian 采样率，后续为 PCM16，不能直接作为裸 PCM 播放。

J0.5 通过需要真实文件/录音中文识别、中文音频播放、能力与格式核验、未运行额外 LLM、共存峰值记录。本轮只能证明源码选择与兼容性差异。板载 APE capture 节点存在不能证明外部麦克风或音频连线可用；真实录播另行验证。
