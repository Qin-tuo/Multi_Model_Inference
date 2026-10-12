# 首版语音服务的准备与兼容性核验

固定 **Qwen3-ASR 0.6B + Matcha zh/en TTS + Vocos**，使用 OpenVoiceStream 的独立语音服务。已增加本机部署路径并通过文件推理与Pipecat语音管线验证。真机录播/AEC和完整质量验收尚未完成，J0.5保持未完成。

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

该上游发布组合的ASR使用Edge-LLM fork。当前本机路径保持Qwen3-ASR模型，改由已验证的llama.cpp CUDA音频后端执行，经OVS后端注册接口接入；对话VLM仍是独立的llama.cpp实例。Matcha采用本机构建的TensorRT引擎。

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

## 当前本机路径与复现

旧ASR发布包仅含TRT10.3序列化engine，不能在当前平台证明兼容；Matcha包包含原始ONNX。当前路径复用llama.cpp执行同一Qwen3-ASR-0.6B，使用上游Matcha拆图脚本和本机trtexec构建3个BF16 estimator及FP16 Vocos。ASR权重/revision/SHA256见模型manifest；TTS构建记录见`benchmarks/results/matcha-build.json`。此为平台兼容性适配，不是Edge-LLM性能对比。

```bash
python3 -m venv --system-site-packages .venv-speech
.venv-speech/bin/python -m pip install -r deploy/speech/requirements.lock
python3 deploy/speech/prepare.py --fetch --engine-source
python3 tools/project.py prepare --component speech
```

系统需提供与构建一致的TensorRT Python3.12 binding及CUDA13.2；主`.venv`提供ONNX拆图与回归依赖。旧engine留在独立输入缓存，运行目录只接受本机新建并校验SHA256的engine。

分别在三个终端启动本地服务：

```bash
python3 tools/project.py serve --component asr
python3 tools/project.py serve --component speech
python3 tools/project.py serve --component llama
```

对应端口为8089、8621、8088，均仅监听回环地址。OVS保留其原生路由、模型能力、并发门控和生命周期；本项目只注册ASR适配器、提供部署profile，并桥接CUDA13/Piper的新旧导入接口。GPU watchdog的RK debugfs权限异常通过实际Jetson标记处理；不替换其模型校验。

```bash
python3 tools/project.py smoke --component speech
.venv/bin/python -m app.voice_file \
  --input "$HOME/.local/share/multi-model-assistant/fixtures/roundtrip/request-0.wav" \
  --output-wav "$HOME/.local/share/multi-model-assistant/fixtures/voice-response.wav" \
  --report benchmarks/results/voice-file-smoke.json
```

当前ASR公开能力为整句文件识别，`/v1/capabilities`明确报告partial streaming为false。Pipecat使用真实Silero VAD分句与REST STT；TTS使用16kHz单声道PCM，不能套用OpenAI专用voice名称/24kHz假设。

术语提示配置位于`configs/runtime.yaml`的`asr_context`。保留无提示失败报告`speech-roundtrip-no-context.json`；同一音频加提示后通过，并保留TensorFlow反例。波形哈希记录于报告。三条合成用例与单次语音问答不能代替独立语音质量集、真人/噪声/AEC测试。

实物选型为ReSpeaker Lite双麦；用户暂时无法连接麦克风/扬声器，物理停播、回声与实际听觉体验仍待验证。

## 持续会话与本地音频依赖

`python3 tools/project.py voice` 已将原生Pipecat语音输入接到 `AssistantSession`，复用记忆、图像预算和工具。语音回答与APScheduler通知使用同一个原生输出队列和有序完成标记；HTTP合成结束不作为成功播放依据。真实文件模式的六轮验证见[报告](../../benchmarks/results/voice-assistant-smoke.json)。文件输出使用明确的`fixture`通道，通知状态为`fixture_delivered`，不会填写`played_utc`。

麦克风/扬声器入口使用Pipecat的 `LocalAudioInputTransport` / `LocalAudioOutputTransport`。其可选依赖安装在主应用环境：

```bash
sudo apt-get install --no-install-recommends portaudio19-dev
.venv/bin/python -m pip install -r deploy/speech/requirements-local-audio.lock
```

PortAudio排空使用 `stop_stream()` / `start_stream()`，仍需真实设备验证；软件AEC尚未接入。设备模式必须显式提供 `--input-device` 和 `--output-device`，不自动选取板载声卡代替ReSpeaker。

文件模式接受一份或多份有效WAV，处理完后退出；到期提醒需要进程保持运行，或以后由再次启动的消费者处理。OVS的流式WAV响应带未知长度头，不能直接用作有限时长输入文件；测试生成器请求PCM，再写出完整WAV头，原失败记录保留在 `voice-assistant-streamed-wav-baseline.json`。

当前Qwen模板对明确的工具轮使用llama.cpp原生assistant prefill；相同请求的五组交替对照中，原格式均用满128token而未调用，预填格式均正确调用。此前六轮文件初测的提醒首段PCM由14.1秒降至4.23秒；引入原文字段保真后最新实测4.61秒且无需修复；完整低延迟与工具质量验收仍待完成，原请求和ISO时间戳播报基线均保留。

## ASR分段完整性与取消

`app/adapters/asr.py` 继承固定版 `OpenAISTTService`，保留上游WAV分段和单个转写任务。活动请求之外最多保留一个待转写段；连续PCM缓存上限992000字节，编码段上限1008044字节，单个输入frame最多6400字节，聚合转写最多8192个UTF-8字节。每段从入队起12秒期限（包含排队等待），关闭SDK自动重试；原生会话结束watchdog设为14秒，先由ASR超时明确失败。统计记录PCM与接受的分段载荷，不将它们当作整个进程的峰值内存测量。

原生结束策略增加“最新分段已被聚合器消费”的条件，保留暂停后补充的完整发言；迟到转写在服务返回及实际聚合消费处都检查代次；异步完成事件还需与当前已消费分段的完整文本和序号相符，同一输入只准入一次。显式取消/输入失败通过原生frame队列重置当前发言，强制结束信号带控制标签，不纳入声学端点或时延测量；队列溢出、输入超限、空识别和请求超时都使整轮失败，不执行截断的指令。关闭语音worker使用原生取消，丢弃尚未执行的识别任务。

复现真实识别的延迟回调测试：

```bash
.venv/bin/python tools/voice_asr_interruption_probe.py
```

报告使用合成语音、真实Silero/ASR/LLM/BGE/Mem0/TTS，并在真实识别返回后延迟回调。它验证“请记住”与后续事实一起提交，以及取消的偏好不进入记忆；不宣称GPU请求已在计算中被抢占、真实麦克风停播时延或完整ASR质量已验收。
