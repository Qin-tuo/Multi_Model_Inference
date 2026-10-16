# Reachy 当前操作入口与迁移记录

2026-10-16。R1/R2已接通无机身网页、共享相机、人体/脸/手、4B问答、中文文件语音与VAD；当前补齐软件可靠性、部署与基线交接。用户已明确将实物麦克风/扬声器后置，当前B0-SW和视觉优先的交接不等待外设。

## 目录与来源

- 当前正式源码与运行入口：`/home/nvidia/Multi_Model_Inference`，`main` 已采用 Reachy 派生实现。
- 旧系统回退源码：`/home/nvidia/AI_Stack/archives/legacy-runtime-972de50`，默认旧七服务已停止。
- 保留的迁移工作树：`/home/nvidia/AI_Stack/projects/reachy-headless`，分支 `migration/reachy-headless`；后续以主目录为准。
- 新Python环境：`/home/nvidia/AI_Stack/environments/reachy-headless`，不继承旧环境的site-packages。
- 只读上游：`/home/nvidia/AI_Stack/research/reachy-mini-jetson-assistant-2673202`。
- 旧系统快照：tag `legacy-before-reachy-v3-20261015`；可验证bundle和配置在`/home/nvidia/AI_Stack/archives/reachy-r0-20261015`。
- [导入清单](../deploy/reachy-import.json)记录46个原样导入文件的SHA256。提交`2084ded`保留裁剪前原样导入；随后差异可直接比较。旧源码、工具、测试保留于旧标签，未删除运行目录或数据。

## 已可运行的入口

```bash
cd /home/nvidia/Multi_Model_Inference
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py config
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '只回答：连接成功' --json-output
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '图中的主要交通工具是什么？' --image /home/nvidia/AI_Stack/runtime-data/benchmarks/vision-input-diagnostic-v1/decoded.png --json-output
```

这些 CLI 诊断命令请在网页没有处理请求时使用。CLI ask是单次文字/图片诊断接口，不执行工具；网页支持中文语音文件、当前会话及本地笔记/提醒。物理语音验收仍待外设接入。图像例子使用历史固定测试图片，不是实时摄像头。JSON中的单次耗时用于诊断，不是冻结的性能基线。

网页默认启用本机 ASR/TTS、Silero 和 D435i，以文件模式处理语音，不占用声卡：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services up
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services status
```

浏览器访问 `http://127.0.0.1:8090`，输入中文问题或上传 16kHz 单声道 PCM16 WAV，可下载真实合成回复。默认文件输出不记作扬声器播放。

需要前台调试时，先执行 `main.py services down --components app`，保留模型服务；再运行 `run_web_vision_chat.py`。调试进程退出后用 `main.py services up --components app` 恢复。自动化调试应使用 `try/finally` 恢复服务并验证新鲜相机状态。`--no-camera` 和 `--no-audio` 仅用于显式缩小前台诊断范围，不代表完整配置就绪。不要同时启动托管与前台两个相机采集者。

另一个终端可以复现实际HTTP/WebSocket链路：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_web.py --text '只回答：连接成功'
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_web.py --camera --text '用一句话描述当前画面'
```

`run_vision_chat.py`复用同一入口与生命周期，不复制主控；`run_voice_chat.py`复用同一网页/会话生命周期，默认关闭相机。一次只接纳一个请求，多余请求返回409；网页停止本轮及原生后端释放确认已接入，具体边界见下方R3记录。

## 配置与模型

沿用上游typed dataclass和overlay组织。默认加载`config/settings.yaml`及`config/settings.orin-nx-headless.yaml`；显式`--config`或`REACHY_ASSISTANT_CONFIG`可选择另一个完整平台overlay。未知字段、错误类型、缺失文件、机器人/RAG字段和远端模型地址均明确失败。输出的effective config是当前参数依据。

上游YAML中的`vad.use_silero`未被原dataclass消费，已删除这个无效配置项；Silero v6已在R2实际接入并验证。

模型使用已有`assistant-vlm`本机服务。主模型由既有模型锁固定为Qwen3.5-4B Q4_K_M。客户端每次生成前使用同一请求参数调用原生多模态token计数端点，保留128输出和16 token余量；不按字符估算。显式模型名称不匹配时不自动替换；HTTP失败、异常结束和空输出不能伪装成功。

## 资源所有权

| 资源 | 当前所有者 | 新应用策略 |
|---|---|---|
| llama.cpp 8088 | `reachy-headless-llama` | 原生模型由 systemd 托管，应用只持有请求 |
| ASR 8089 / 语音8621 | `reachy-headless-asr/speech` | 使用已有模型和 OVS，不重复启动模型副本 |
| D435i | `reachy-headless-app` 中的 Camera | 单一采集，共享带 epoch/序号的有界帧缓冲 |
| 旧 ROS 视觉/姿态/Qdrant | 已停止的旧七服务 | 保留旧源码、模型与数据用于回退，不随新配置启动 |
| 麦克风/扬声器 | 实物后置 | 不生成采集或播放成功回执 |
| CLI 客户端 | 每次诊断请求 | 关闭自身 HTTP 流；使用前确保应用无正在处理的请求 |
| Web端口8090 | `reachy-headless-app` | 控制器拥有当前轮、取消及输出，systemd 拥有进程生命周期 |

所有新安装仅进入独立环境。不更换BSP、CUDA或TRT，不运行上游清理其他相机/音频进程的函数。依赖锁为`requirements.lock`，覆盖当前应用、CPU VAD及开发测试；ASR/TTS/LLM复用本机现有GPU服务，不把它们的Python依赖装入新应用环境。

## 阶段证据与未完成项

以下 R1～R3 是各阶段当时的测试记录；其中 `mmassistant-*` 指旧部署。当前启动、停止和回退以 [部署说明](../deploy/systemd/README.md) 为准。涉及旧服务名的历史故障脚本应在相应隔离配置中复现，不能直接套用到当前默认服务。

[阶段汇总](../benchmarks/results/reachy-r1-summary.json)、[真实网页文字](../benchmarks/results/reachy-r1-web-text.json)、[真实D435i/网页/VLM](../benchmarks/results/reachy-r1-camera-web.json)、[真人脸检测](../benchmarks/results/reachy-r1-live-face.json)、[旧RGB/深度恢复](../benchmarks/results/reachy-r1-legacy-restore.json)。

43项R1适用测试通过。Silero模型集成测试属于尚未接入的R2，本次命令明确排除该文件，不宣称全部模型或完整应用已验收。测试命令见阶段汇总。

YuNet固定版本和哈希见`models/perception.json`，权重与许可证存放AI_Stack，加载时验证哈希，运行时不联网下载。后端为OpenCV CPU；没有声称该小模型已使用GPU。缺失工件可显式运行`python scripts/prepare_face.py`准备。

相机使用本机已验证的D435i RGB节点（当前解析为/dev/video4），不使用上游默认元数据节点/dev/video1。帧带epoch、序号和主机收帧单调时间；不是硬件曝光时间。失效/过期帧不能继续交给模型。网页预览使用与人脸结果相同的原帧，避免把旧框套在新画面上。

真人脸验证连续取得10个不同帧的正例；这证明当前环境的检测连通，不是识别精度统计或身份验证。浏览器静态页面用本机Firefox实际渲染，HTTP/WebSocket验证使用真实模型；原始相机图片与截图保存在AI_Stack，没有提交Git。

当前尚待完成：手势真实负例/遮挡/多人矩阵、整机资源压力与完整故障矩阵、物理音频及B0基线。当前会话及本地工具已完成声明范围内的软件验证；真人手势已取得正例，新增证据见下文。

当前回退须先停止新四服务，再从固定旧源码启动旧七服务，并核验新鲜 RGB/对齐深度。详细步骤见部署说明；不通过 `git reset --hard` 回退在线运行目录，不覆盖用户数据。源码快照不等于在线数据库备份。


## R2中文语音接入

[语音阶段汇总](../benchmarks/results/reachy-r2-summary.json)与[实际文件链路](../benchmarks/results/reachy-r2-speech-file.json)记录真实模型结果；[设备缺失](../benchmarks/results/reachy-r2-missing-device.json)证明缺麦克风时明确失败，不作为实物验收。

沿用STT、TTS和pipeline接口。STT经本机OVS调用Qwen3-ASR，TTS使用Matcha/Vocos；原Whisper/Kokoro推理路径和额外TTS子进程已移除。Silero v6从固定faster-whisper 1.2.1发行包提取，校验PyPI wheel和模型SHA256，用CPU ORT执行；不安装Whisper/CTranslate2来加载这个小模型。

输入文件先做格式/长度检查，再经同一个Silero分段器、ASR、4B和有界TTS队列。中文按标点和字符上限分句，支持短中文。文件时间为相对偏移，有限文件的结束使用补零，不冒充真实声学端点。话筒块带输入epoch和主机接收时刻，静音/恢复时清空旧epoch。

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_speech.py \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r2-input.wav \
  --output /home/nvidia/AI_Stack/runtime-data/reachy-r2-speech/reply.wav
```

示例输入是本机TTS生成的中文测试语音；不是真人麦克风录音。复现脚本将结果与哈希写入输出旁的JSON。

合成队列最多2个文本片段；麦克风队列最多64个32ms音频块，溢出明确失败。单句PCM和整轮PCM有时长上限。默认仅保留最近10个本应用生成的回复WAV；写完并关闭WAV后才发布下载链接。文件保留回归发现过mtime并列导致新文件被删，已修复为始终保留刚生成的文件。

文件回执`file_written`与设备回执`device_completed`分开。设备回执表示播放器正常退出，不代表已经完成声学验收。当前默认input_mode/output_mode均为file。要进入设备模式需显式配置ReSpeaker Lite及输出设备；实际录播、AEC和双讲仍待硬件接入。

移除了结束所有录音进程、关闭桌面PulseAudio及改写其配置的逻辑。只管理自身录音/播放进程和AEC模块，AEC名称按实例隔离，选择输出不修改系统默认扬声器。音频错误会结束本轮并返回失败；R3已补充软件取消及后端释放观察，物理停播仍未验收。


## R2人体、手掌与21点手部

[感知验证汇总](../benchmarks/results/reachy-r2-perception-summary.json)记录实际后端和证据范围。人体使用已有YOLOv8n FP32/TF32 TensorRT引擎；脸使用YuNet；手掌和21点手部使用OpenCV Zoo原始ONNX和上游预后处理。手部当前为OpenCV CPU，未进行INT8转换或性能调优。

```bash
cd /home/nvidia/AI_Stack/projects/reachy-headless
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/setup_native_perception.py
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/prepare_perception.py --check-only
```

setup只在本应用环境链接系统匹配版本的TensorRT Python包，避免引入整个系统site-packages。prepare校验已保留的人体ONNX和本机engine；可显式下载锁定的手部权重与参考源码、应用已审查的坐标修正。若本机人体工件丢失，需要从保留工件恢复或按独立构建流程重新验证，不会静默换用CPU。

`app/perception.py`以一个工作线程处理最新帧，无待处理图像队列；人体、脸和手使用同一份原始BGR图。结果带frame_id、epoch、收帧/完成时刻和实际provider。网页图片与框线共同来自该帧；VLM选用同一帧并接收对应的简短辅助观测，观测明确允许漏检，不能冒充身份或完备真值。

运行应用后可以记录真实发布结果：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_perception.py \
  --seconds 10 --output /home/nvidia/AI_Stack/runtime-data/perception-observation.json
```

默认最多10个人体候选、2只手。手部每只返回21个点；z是模型估计的相对腕部量，不是D435i测量深度。模型没有逐点置信度，因此逐点confidence明确为空，保留整只手的置信度。

源码审查并复现了Palm后处理的NMS格式问题：上游将xyxy框送入要求xywh的接口，两个不相交的框可能被误合并。[最小补丁](../patches/perception/palm-nms-xywh.patch)和[回归](../benchmarks/results/reachy-r2-palm-nms.json)保留原版1框、修正版2框的证据。

原带标注缩略拼图没有得到可靠手部正例，这一结果仍保留。改用官方MediaPipe教程的干净图片后，得到1个人体、1张脸和2只手的完整21点结果。这些公开图验证模型/坐标接入，不能代替真人手势或质量基准。真机组合测试处理42个不同帧，无推理失败；该次未获得人体/手部正例，仍需补充真人验证。

TensorRT加载旧engine提示设备兼容性警告。[审查记录](../benchmarks/results/reachy-r2-engine-warning.json)显示构建记录与当前Orin的计算能力、SM数一致，engine早于本次启动；NVIDIA记录过同版本重启后的同类问题。本机根因未被完全证明，警告保持可见，系统版本保持固定，长期稳定性仍待验证。参考[官方问题记录](https://forums.developer.nvidia.com/t/trt-engine-plan-file-across-devices-warning-after-reboot-on-jetson-orin-nano/376848)。

人体预后处理适配来自Ultralytics公开示例，保留AGPL-3.0声明；TensorRT运行辅助代码来自NVIDIA样例，保留Apache-2.0声明。对应原文、固定提交及许可证见`deploy/perception-sources.lock.json`、`licenses/`和`THIRD-PARTY-NOTICES.md`。


## R3整轮取消的软件验证

[阶段汇总](../benchmarks/results/reachy-r3-cancellation-summary.json)、[原生后端观察](../benchmarks/results/reachy-r3-native-cancellation.json)、[真实Web链路](../benchmarks/results/reachy-r3-web-cancellation.json)和[浏览器按钮](../benchmarks/results/reachy-r3-browser-cancellation.json)记录本次范围。74项适用测试通过，包含真实本地阻塞HTTP连接和自有播放器替身进程，不使用扬声器假回执。

`TurnController`仍是唯一轮次所有者。停止请求携带当前`turn_id`；当前网页优先走已有WebSocket的`cancel`消息，`POST /api/cancel`仍可调用。迟到的旧轮次取消不能中断下一轮。取消标记先阻断新输出和后续模型调用，再中断自身HTTP连接、停止自身播放器。HTTPX的连接trace让应用在响应头到达之前也能中断阻塞读取；只shutdown该操作持有的socket，连接最终关闭由HTTPX负责。

WAV先写到暂存文件，成功完成后由控制器在与取消共用的锁内发布；取消不能发布半轮音频。之前已完成且仍在保留范围内的文件继续可用。WebSocket排队和实际发送时都检查该轮取消标记，迟到token/转写不会重新出现。文件回执与设备进程回执仍分开。

本地调用返回与后端释放是两个状态。这里的释放指本轮执行占用结束，模型权重继续常驻，不表示整机GPU空闲或内存归零。LLM检查原生`/slots`；ASR同时检查OVS占用与8089原生槽；TTS检查OVS持有到执行线程退出的占用。源码依据及哈希见汇总。释放未确认时保持409拒绝新轮，5秒后显示`release_unconfirmed`并继续探测恢复，不将超时当作成功。关闭应用时仍无法确认释放会明确报错，不会替外部systemd服务杀模型进程。

本次三个原生模型都观察到“本地已取消、后端仍忙”，随后空闲，下一请求成功。完整Web测试分别在ASR、LLM、TTS忙时取消：无迟到输出、取消音频404、下一轮正常生成WAV。Firefox实测还人为延迟取消HTTP回执1秒，验证旧回执不会覆盖较新的空闲状态。

复现时保持这些共享后端没有其他请求；先启动无相机文件模式，再运行：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_cancellation.py \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r2-input.wav \
  --output /home/nvidia/AI_Stack/runtime-data/reachy-r3-native-cancellation.json
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_turn_cancel.py \
  --base-url http://127.0.0.1:8090 \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r2-input.wav \
  --output /home/nvidia/AI_Stack/runtime-data/reachy-r3-web-cancellation.json
```

可选浏览器回归使用已安装Firefox与geckodriver，启动`geckodriver --host 127.0.0.1 --port 4445`后运行`python scripts/probe_browser_cancel.py --base-url http://127.0.0.1:8090 --output /home/nvidia/AI_Stack/runtime-data/reachy-r3-browser-cancel.json`；完成后退出自己启动的驱动进程。

这些单次时长属于功能诊断，不是B0或性能承诺。ASR当前已提交的识别和Matcha当前合成步骤仍可能运行至安全结束；不宣称CUDA kernel可被即时抢占。实物播放缓冲何时真正静音、AEC/双讲、手势完整质量、工具取消边界、模型崩溃/重启矩阵及长期持续运行尚未通过验收。


## R3挥手与静止举掌

[手势汇总](../benchmarks/results/reachy-r3-gesture-summary.json)、[真人测试](../benchmarks/results/reachy-r3-live-gestures.json)、[轨迹回放](../benchmarks/results/reachy-r3-gesture-replay.json)和[RGB/深度恢复](../benchmarks/results/reachy-r3-gesture-camera-restore.json)分别说明验证范围。当前88项适用测试通过；F06/F07仍为部分正例通过，未标记完整验收完成。

张掌判定复用[OpenCV Zoo固定版本demo](https://huggingface.co/opencv/handpose_estimation_mediapipe/blob/4b2a0b446e5cf2f11fb6b2c7251091c035d2c1f7/demo.py)的五指角度/距腕规则；原文、哈希和Apache-2.0许可证已锁定。这里只提取姿态规则，避免导入demo时解析命令行和联网下载。角度计算补充有限值、零向量及反余弦数值范围保护。

`app/gestures.py`接收感知线程发布的同帧手部结果，不再采集图像，也不创建模型线程。匿名短轨迹根据位置、手掌尺度和可靠手性作相互最近匹配；交叉歧义、遮挡和来源变化重建证据，不承诺跨人的身份保持。轨迹最多4条，每条历史最多32个样本。重复帧不增长证据，时间倒退、过期结果和相机epoch变化不能延续动作。

`GestureConfig`保留所有有效参数。当前开发起点为：静止保持1秒且至少5个不同帧；挥手在2秒窗口出现至少2次明确反向；移动幅度按手掌宽度归一化；手部完整入框且掌宽至少20像素。阈值尚未经过完整质量评估。一次举手触发后锁存，收手/连续非张掌后重新积累，避免挥手结束后仍举着手而立刻取消自己的问候。

事件带匿名来源、首末帧、证据时间和有效期。`TurnController`仍是唯一接纳/取消入口：挥手空闲时走现有LLM→TTS路径，忙碌时忽略且不排队；全局问候间隔5秒，重复/过期事件无效。设备模式下话筒静音同时抑制自动问候，配置`gestures.greetings_enabled: false`也可关闭问候；有效举掌停止仍可使用。停止绑定当时的turn_id，早于新轮次入场的画面不能取消它。

现场测试由用户在D435i前实际操作，150秒采样取得738个不同感知帧，其中人体正例732帧、人脸498帧、手部586帧；这是模型观察计数，不是标注准确率。记录到两次挥手问候、两次空闲状态下的举掌（无任务可取消），以及一次取消正在生成的计数回答。已确认一轮真实问候生成“你好！”的WAV，举掌取消后本轮占用释放。最初手腕/手指出界的帧不参与动作判断；界面现已提示完整入框。原服务恢复后再次收到新鲜RGB和对齐深度。

第一次手动协调的长回答在用户举掌前已完成，保留这一限制；成功的停止测试由检测到新张掌时发起计数请求，让用户持续举掌触发取消。观察窗口结束后应用按计划退出，因此现场未继续取得下一轮回复；独立回放用同一批实际手部关键点复现了全部5个事件及相同起止帧，再将标记为回放的事件送入实际4B/TTS，验证问候、取消、下一轮回复与既有音频保留。

在正常相机应用运行时，可从另一个终端记录（原始关键点默认只留本地，不提交Git）：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_gestures.py \
  --seconds 60 --output /home/nvidia/AI_Stack/runtime-data/gesture-observation.json
```

`--prime-stop-test`会在新的静止张掌候选出现时显式提交计数请求，用于协调真人停止测试；这不是生产入口的自动行为。仅观察时不加该参数。测试脚本不保存相机像素。

本机已记录的关键点可复现规则和实际后端控制：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_gesture_replay.py \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r3-live-gestures/observation.json \
  --output /home/nvidia/AI_Stack/runtime-data/reachy-r3-gesture-replay.json --exercise-models
```

回放绕过相机和视觉推理，不能增加真人样本数量或证明检测准确率。真实负例、遮挡、多人、不同距离/光照，以及实际扬声器何时静音仍未验收；当前未进行性能调参或采集B0。


## R3当前会话上下文

[阶段汇总](../benchmarks/results/reachy-r3-session-summary.json)、[真实文字多轮](../benchmarks/results/reachy-r3-session-web.json)、[真实语音文件多轮](../benchmarks/results/reachy-r3-speech-session.json)及[浏览器清空竞态](../benchmarks/results/reachy-r3-session-browser.json)记录本次验证。96项适用测试通过。

沿用Reachy的消息/`few_shot`接口，由已有`TurnController`维护当前会话；没有新增记忆agent。默认`session.enabled: true`、`session.max_turns: 4`，最多保留4轮成功完成的文字。失败、取消、语音合成失败的轮次不记为完整回答；自动挥手问候不占用用户上下文。达到模型输出上限时，历史文字附带截断标记。

每次请求仍调用llama.cpp原生输入token计数。若历史加当前请求、输出预留和16 token余量超过2048，则逐个移除最老的完整轮次，并重新计数；工具轮次包含其调用和回执消息，不留下孤立tool消息。不会切断当前输入或删除当前图片来伪装接纳。仅当前请求仍过大时，明确返回`CONTEXT_BUDGET_EXCEEDED`。原生计数、保留/移除轮数随结果记录；兼容字段`history_pairs_*`计数完整会话轮次。

视觉历史只保留文字，并注明该轮原图已经丢弃；下一轮不会重发旧图片。成功记录与输出发布、取消使用同一个控制器锁。语音文件经过VAD/ASR后也进入相同上下文路径，本次真实语音测试可接续“最喜欢的颜色是蓝色”，清空后返回“不知道”。

网页提供“清空会话”；接口为`POST /api/session/clear`，请求需带当前`session_id`。活动轮次或后端释放尚未结束时返回409；旧会话的迟到清空请求也返回409。清空只重置当前上下文和最近显示结果，既有输出WAV按原保留策略管理。当前上下文缓冲只保存在应用进程内，应用重启后开始新会话；模型日志、验证报告和显式笔记/提醒各自管理，长期记忆仍另阶段。

会话状态由控制器在提交/清空时有序发布，避免资源统计线程的旧快照覆盖新会话。浏览器验证故意延迟清空HTTP回执2秒，在它返回前开始新一轮；旧回执没有擦除新消息。

在隔离、初始为空的应用实例运行以下脚本；脚本会测试清空，不要用于保留中的实际对话：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_session.py \
  --base-url http://127.0.0.1:8090 --output /home/nvidia/AI_Stack/runtime-data/session-text.json
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_speech_session.py \
  --base-url http://127.0.0.1:8090 --output /home/nvidia/AI_Stack/runtime-data/session-speech.json
```

第二个脚本要求另一个初始为空且启用文件语音的实例，输入由本机TTS生成；不代表真人录音。浏览器回归复用已安装的Firefox/geckodriver，运行`python scripts/probe_browser_session.py --base-url http://127.0.0.1:8090 --output /home/nvidia/AI_Stack/runtime-data/session-browser.json`。

代码代号测试保留原始输出“青鹭 63B”；评分仅忽略空白，未把格式完全一致当作已通过。当前会话通过不代表长期记忆、工具、物理语音或完整项目验收通过。

[工具协议探测](../benchmarks/results/reachy-r3-tool-protocol-readiness.json)确认当前4B模板实际支持并选择了`create_note`，预检与实际输入token均为323；没有执行函数或写入笔记。工具协议探测之后已接入SQLAlchemy/SQLite与APScheduler，验证见下节。需要持续保持的边界包括：明确有限工具白名单和参数、重复操作幂等、取消前未开始的操作不执行、已开始事务收尾并保留回执、已提交内容跨取消/重启保留、到时通知与真实输出回执。


## R3持久笔记与提醒

[工具汇总](../benchmarks/results/reachy-r3-actions-summary.json)、[真实集成](../benchmarks/results/reachy-r3-actions-integration.json)、[提交后取消](../benchmarks/results/reachy-r3-actions-cancel.json)、[浏览器回执](../benchmarks/results/reachy-r3-actions-browser.json)及[语音会话并存](../benchmarks/results/reachy-r3-actions-speech-context.json)记录软件证据。提醒输出目前为WAV文件，不能据此声称已完成喇叭、AEC或实际停播验收。

当前平台配置默认启用`actions.enabled: true`；`--no-tools`或配置false可关闭工具和调度器。`main.py ask`仍是诊断接口；完整工具从Web/语音应用入口使用。数据放在`~/AI_Stack/runtime-data/reachy-actions/actions.sqlite`，由单个应用进程持有文件锁，另一个实例需指定独立数据库或关闭工具。新库不打开、迁移或覆盖旧项目数据库。

| 责任 | 实现 |
|---|---|
| 参数和允许的工具 | `app/actions/schema.py`，严格Pydantic参数、固定五个工具 |
| 数据与操作回执 | `app/actions/store.py`，SQLAlchemy 2.1.4＋SQLite WAL/FULL事务 |
| 时间与重启调度 | APScheduler 3.11.3，持久DateTrigger、单个执行线程；回调只标记到期 |
| 接纳、取消和模型调用 | 原有`TurnController`，每轮最多两次串行工具调用 |
| 到时通知 | 复用同一轮次与SpeechOutput，直接输出提醒原文，不再次调用LLM改写 |

五个工具是创建笔记、查询最近三条笔记预览、创建提醒、查询最近五条提醒、按编号取消提醒。笔记标题最多80字符、正文500字符；提醒文字160字符。时间使用含时区的ISO时间，或整数`delay_seconds`；相对时间从该请求登记时刻起算，执行时已过期的期限明确拒绝。当前不宣称已经验证复杂日历理解、所有查询表达或完整笔记产品功能。

默认最多100个待处理提醒、每类1000条业务记录和10000条请求账本；满额拒绝新写入，不删除已有记录。普通“请记住我喜欢蓝色”仍只进入当前会话；真实语音测试中笔记数保持0。长期记忆/RAG没有重新启用。

工具参数在执行前验证。控制器先在与取消共用的锁中确认调用可以开始，然后执行事务。取消会阻止下一次未开始的调用；已开始的事务完成后仍发布真实回执。笔记/提醒写入与该操作回执处于同一事务中，注入回执写入失败时两者一起回滚。真实模型测试分别在笔记和提醒提交后取消当前回答，记录仍保留，取消轮次没有可下载的半成品语音。

HTTP文字请求可携带32位十六进制`request_id`，音频入口通过同名查询参数传递。前端每次新操作生成一个ID；传输重试必须沿用原ID和原输入。同ID同输入返回已提交回执，不重新调用模型或增加副作用；已完成图像请求的回执重放也不要求摄像头当前可用。同ID不同输入失败。新的ID表示新的操作意图。请求账本与会话清空相互独立；取消/失败请求中的已提交操作也可重放回执，但不会自动继续其剩余步骤。

提醒先保存，再由APScheduler安排执行；调度暂时失败会明确显示，周期对账负责恢复。应用停止期间到时的提醒，重启后在未过期范围内进入待通知状态；已输出的提醒不会因再次重启重复输出。通知只有在播放器成功启动且即将写入，或WAV即将原子发布时，才记录输出开始。输出开始后未确认就重启，状态为`uncertain`，不能伪装已送达或自动重复。用户停止通知后保存`stopped`状态与原业务记录，不立即重播。取消已经输出的提醒返回`already_delivered`，保持原输出回执。

初次集成曾发现实际错误：[模型未查询便编造提醒](../benchmarks/results/reachy-r3-actions-initial-failure.json)。当时历史只保留了普通问答。现已保留完整原生工具调用/回执消息组，并明确查询必须本轮读取数据库；原场景复测通过。失败记录保留，不能据此推断所有模型幻觉已经消除。业务状态始终以执行回执和数据库为依据。

复现完整工具链需独立测试配置，将`actions.database`与`audio.output_directory`指向测试目录，保持input/output为file，然后运行（脚本自行管理8093测试应用，勿与现有实例重叠）：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_actions.py \
  --config /path/to/isolated-test-settings.yaml \
  --output /home/nvidia/AI_Stack/runtime-data/actions-integration.json
```

该脚本创建测试笔记和提醒，重启应用、观察到期通知、验证查询/取消及工具关闭。`scripts/probe_action_cancel.py`针对已启动的隔离应用，在收到真实提交回执后立即取消当前轮次。脚本不操作旧业务数据库。纯会话预算探测若需要复现工具接入前的配置，启动时加`--no-tools`；工具定义本身也占用2K上下文预算。

该次工具集成114项适用测试通过；软件证据包含事务/幂等、容量拒绝、调度恢复、原生模型选择、实际TTS文件和浏览器显示。物理播放、完整设备/模型故障矩阵、代表性长时间运行及B0仍待后续阶段；不进行自动性能调优。

## R3网页传输与读取边界

[汇总](../benchmarks/results/reachy-r3-web-summary.json)记录124项自动化测试，以及实际TCP、Firefox和本机4B的适用验证。[修复前复现](../benchmarks/results/reachy-r3-web-bounds-before.json)保留两处实际问题：事件循环暂停时，10000条画面消息留下10000个待执行回调；六个并发握手突破原四连接限制。修复后相同画面输入保留一个回调和最新一帧，握手先占位，最多四个客户端。

`app/web_limits.py`只负责传输缓冲、原生读取线程所有权与请求体边界；轮次仍由`TurnController`控制。初始容量集中在`WebConfig`，有效配置通过`/api/state`展示：

| 对象 | 默认上限与行为 |
|---|---|
| 线程到事件循环的消息 | 64条、4MiB、单消息2MiB，只有一个待执行排空回调；画面和监控保留最新值 |
| 每个网页客户端 | 16条、4MiB；可靠事件超限显式断开，重连读取当前状态 |
| 网页连接 | 握手之前预留，最多4个；失败/断开释放占位 |
| 请求体 | JSON 32KiB，取消/清空4KiB，音频沿用时长对应字节上限；解析JSON之前限制 |
| 慢上传 | 普通请求体读者4个、控制2个，整段读取最长10秒 |
| 状态与扬声器控制 | 最多2个原生线程任务，控制写入单独限制为同时1个；调用者2秒超时后仍保留真实任务占位 |
| WebSocket | 接收4KiB、协议接收队列4；单次发送最长2秒；取消可沿已有连接发送 |
| 浏览器记录 | 按turn_id关联答案和回执，最多100个消息节点；不作为长期聊天记录 |

初始化状态含事件游标，避免较旧回调覆盖已读取状态；未结束轮次的文本/回执单独保留。最终结果携带已提交回执，重连不会将旧轮答案拼进新轮。状态读取在限定线程中执行，取消接收不等待该读取；超时不意味着底层线程已退出。

[TCP测试](../benchmarks/results/reachy-r3-web-transport.json)使用独立Uvicorn、本地真实网络、合成消息和可控阻塞，未启动模型。20次连接/断开后FD和线程回到测试前水平；PSS增加约0.70MiB，不能凭此宣称长期无泄漏。两个状态任务超时仍占用两槽，多余读取被拒绝，已有连接可取消；四个慢请求体不占用HTTP取消的读者额度。慢网页被超时清理，测试服务器关闭后只剩主线程。

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_web_transport.py \
  --output /home/nvidia/AI_Stack/runtime-data/web-transport.json
```

该故障脚本将读取/发送超时缩短为0.2秒、请求体超时0.5秒，以可重复触发边界，不将结果当成生产性能。真实Firefox另行验证了本机模型的WebSocket取消、延迟清空回执和处理中重连。测试用文件语音、关闭摄像头和工具；正式完整配置仍需R4重验。

字节预算约束应用保留的缓冲，不等于整机内存上限。Uvicorn还设有16连接/任务并发限制，任意连接洪泛仍可能拒绝新HTTP请求。原生线程无法强制终止，关闭时若未释放会报错；实物扬声器路由与关闭并发、音频慢消费者、完整模型/设备故障及长时运行仍需后续验证。R3.resources保持进行中，B0尚未采集。

## R3模型退出恢复与调用总时限

[恢复汇总](../benchmarks/results/reachy-r3-recovery-summary.json)和[最终真实模型记录](../benchmarks/results/reachy-r3-recovery-native.json)覆盖四个场景：LLM推理中退出、ASR原生进程退出、TTS合成服务退出，以及笔记已提交后LLM退出。验证使用独立数据库和文件语音，未接触相机或实物声卡。

测试先确认模型正在执行，再核验监听进程属于指定systemd服务的子进程及cgroup，使用pidfd向该进程发送SIGKILL。现有systemd的`Restart=on-failure`启动新进程；应用本轮明确失败，未确认释放时返回409，观察到恢复后的原生槽/语音占用空闲才放行下一轮。四个场景的下一轮均成功生成语音文件，失败轮没有发布WAV、遗留partial或写入会话历史。

已提交笔记在模型退出后保持存在，同一request_id重试返回原回执，没有再次调用工具或新增笔记。恢复时长约11～18秒，仅为本次故障诊断记录，不是B0或性能承诺。四个场景在总时限修改前后各验证一次，未代替最终每类故障十次的要求。

首次组信号注入曾发生管理命令兼容问题：[原始失败记录](../benchmarks/results/reachy-r3-recovery-injection-failure.json)。命令报告辅助进程错误，但服务实际退出并重启，因此该次按测试未完整完成记录，不冒充四场景通过。后续改用核验归属的单进程句柄。

同时复现并修复了调用时限漏洞：HTTPX的读取超时约束单次等待，持续返回小块数据会延长整次请求。[旧实测](../benchmarks/results/reachy-r3-recovery-deadline-before.json)在80ms读取超时下仍持续约413ms；[修复后](../benchmarks/results/reachy-r3-recovery-deadline-after.json)在160ms总预算处中断该连接并回收计时线程。

现有`llm.timeout`、`stt.timeout`、`tts.timeout`同时作为各自一次推理调用的总预算：LLM包括该次原生token计数、历史裁剪及生成；ASR为一个语音片段；TTS为一个文本分句。它不是整轮多次工具调用或全部语音分句的总时长。`RequestIO`每次操作最多持有一个计时线程，退出时取消并join，之后才允许新操作。取消优先保留取消语义；超时不会被解释成后端已空闲，也不意味着GPU内核被即时抢占。ASR/TTS下一次成功后清除旧错误。

该次132项自动化测试通过，包括六个真实本地socket场景：三个adapter分别在响应头阻塞和持续小块返回时超时，随后同一adapter恢复正常请求。启动/健康读取时限及相机恢复的后续证据见下节；语音打断、实物音频与长时运行仍需继续核验。

复现真实故障时，先以独立配置启动`--no-camera`应用，使用文件音频、空测试数据库和独立输出目录。脚本只允许已核验的固定本机端点和三个服务；测试会使这些共享后端短暂不可用，不应与其他任务同时运行：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_service_recovery.py \
  --inject-managed-service-crashes \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r2-input.wav \
  --output /home/nvidia/AI_Stack/runtime-data/recovery-native.json
```

测试配置的数据库目录须含`reachy-r3-recovery-`，防止误用日常数据。脚本等待应用就绪，记录全部故障和恢复，在退出前确保受影响服务重新就绪；最后还需关闭自己启动的测试应用。当前[退出核验](../benchmarks/results/reachy-r3-recovery-cleanup.json)确认8093关闭、七个原服务active、主模型/ASR槽和语音计数为空闲。默认服务入口仍在旧工程，迁移切换尚未完成。

## R3健康探测与摄像头恢复

[本次汇总](../benchmarks/results/reachy-r3-health-camera-summary.json)包含144项自动化测试、真实控制接口、D435i驱动断开/重绑定和摄像头重复启停。完整项目仍未验收，B0尚未采集。

健康探测的[修复前](../benchmarks/results/reachy-r3-health-before.json)及[修复后](../benchmarks/results/reachy-r3-health-after.json)记录两个问题：原健康检查仅凭HTTP 200判定主模型正常，即使返回的是另一模型名；原槽查询也会被持续小块返回拖长。现由`request_io.control_json`统一处理启动、模型身份、模板能力、语音能力和槽状态GET：独立连接、总时限、最多64KiB保留JSON，超过边界明确失败。ASR的语音能力和原生槽分别观察，不能只凭代理空闲就放行。

主模型健康检查校验配置中的模型别名；部署工件身份仍由后端清单与启动配置保证。这不是对任意相同别名服务的权重证明。真实LLM/ASR/TTS控制接口以及中文识别/PCM合成已[复核](../benchmarks/results/reachy-r3-health-native.json)，错误探测不会被当作资源释放。

相机的[原始故障](../benchmarks/results/reachy-r3-camera-recovery-before.json)来自实际D435i RGB UVC接口断开：原线程退出，驱动重新绑定后仍没有画面。现在`Camera`保留一个采集线程，由它依次打开、读取、释放和重试；重连没有新增采集者。失败立即清空缓存、更换来源epoch并设置失效标记，感知结果与待发送旧画面均不能继续使用；新帧成功到达才清除错误并恢复ready。Web状态新增`camera`字段，包含阶段、帧龄、重试次数和缓存占用，失效时同时隐藏旧的人脸裁剪。

`vision.reconnect_seconds`默认1秒。`vision.v4l_select_timeout`默认2秒，在首次采集前设置OpenCV的进程级参数；[OpenCV 4.11源码](https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_v4l.cpp#L926-L983)表明这个参数会被缓存，因此修改它需要重启应用进程。它限定的是select等待，不能保证所有驱动ioctl都能在两秒内退出。读取未释放时，关闭仍明确报告`close_unconfirmed`，不会并发release正在读取的句柄。

[最终真实应用验证](../benchmarks/results/reachy-r3-camera-recovery-final.json)结果：

- 断流期间图像请求返回400，文字问答正常；旧图像、人体/脸/手结果失效。
- 重绑定后同一个采集线程恢复，新epoch贯穿小模型、Web画面和实际VLM图像请求。
- 接收到新epoch后没有再收到旧epoch画面；测试记录仅保存元数据，不保存摄像头图片、脸/手关键点或视觉问答正文。
- 应用退出后，另做一次摄像头预热及20次真实启停。每次关闭后FD均为49、线程均为19，与预热后的同进程状态一致；PSS从472462336到472517632字节。该进程仍包含已加载库和诊断工具，这不是“整机只剩这些资源”的结论，也不是长时无泄漏证明。
- 最后恢复原相机服务，核验三张不同时间戳的新鲜RGB和对齐深度，而不只查看服务active。

这两次成功故障注入是RGB驱动接口的解绑/重绑，不能冒充物理拔线、USB集线器掉电或任意驱动卡死验证。20次是摄像头模块启停，完整应用/模型/录播的重复生命周期仍归R4。本次sysfs枚举速度为5000Mb/s，保留为本次环境信息，不作为USB吞吐测量。

独占相机时，可执行以下有明确副作用的诊断。脚本只解绑配置中按序列号核验的D435i RGB接口，需已有免密码sudo；自动停止本项目旧相机服务、运行新应用、恢复驱动与服务，并用独立系统Python/ROS检查原RGB和深度。ROS只用于旧系统交接核验，不进入新应用运行依赖。

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_camera_recovery.py \
  --detach-configured-camera \
  --output /home/nvidia/AI_Stack/runtime-data/camera-recovery.json
```

模型/工具和网页压力的前期证据仍保留各自范围；软件语音打断的后续证据见下节。实物AEC/录播按用户要求后置；完整手势质量、软件压力/重复生命周期及8小时共存仍待完成。R3.resources和R3.recovery继续保持进行中。

## R3软件语音打断与当前交接范围

用户于2026-10-16确认：实物音频暂时跳过，接手后先优化视觉。当前目标是软件框架、D435i视觉实机、B0-SW与代码/数据流导读；物理录播、声学AEC/双讲、可听首声和实际停播标记DEFERRED。完成软件交接后，助手等待用户实操结果，不自动展开连续性能优化。具体门槛见[计划v3.1](../02-jetson-multimodel-inference.md)。

[语音打断汇总](../benchmarks/results/reachy-r3-voice-summary.json)、[原行为复现](../benchmarks/results/reachy-r3-voice-before.json)、[真实模型回放](../benchmarks/results/reachy-r3-voice-native.json)及[Firefox验证](../benchmarks/results/reachy-r3-voice-browser.json)记录软件范围。原来忙碌时的新麦克风片段直接被拒绝，没有触发取消；现在同一`TurnController`负责语音预留和轮次切换。

1. 采集读者持续产生带epoch、序号和主机单调时间的32ms PCM帧；静音、切换或故障使旧输入失效。
2. 原Silero分段器确认新讲话后，控制器预留一条语音并取消当时观察到的旧轮次。默认累计160ms有效语音触发提前确认，短的有效话语可在端点确认；这是开发参数，不是声学质量结论。
3. 分段器继续收完整句话。完成后仍只保留一条待处理语音，等待旧原生后端释放，再进入ASR、4B和TTS。资源等待中的语音不会提前启动另一个模型请求。
4. 更晚的新话语可替换尚未执行的语音；静音、源中断、明确停止/举掌或30秒等待到期会丢弃它。旧轮次的迟到取消不影响新话语。
5. 超长话语不截断执行，丢弃后等待安静再接纳下一句。输入停滞或不连续会终止本次直播输入、回收自身采集进程，并显示检查设备后重启应用；文字/文件请求仍可继续。

`audio.voice_interruptions`控制是否启用打断，`audio.pending_voice_seconds`约束等待，`audio.input_stale_seconds`约束进入VAD前的帧龄；输入队列仍为64块。`vad.confirm_speech_ms`控制提前确认，VAD最长时长按完整32ms帧限制，包含回看和尾部静音。

设备播放没有有效软件AEC时使用半双工降级：处理旧轮期间不解释麦克风内容，恢复前等待安静片段，避免把自身播报尾音或用户讲话后半句当成完整指令。该策略的状态机已经验证，实际声学效果留到后续硬件阶段。文件输出没有设备回声，可用于当前软件验证。

本次四个场景通过真实Silero、Qwen ASR、4B与Matcha/Vocos：新语音打断并完成答复、停止待处理语音、静音丢弃、PCM生产者卡住。录音命令和设备发现被明确替换为按实时节奏输出既有测试WAV的子进程，因此不能声称已使用实体麦克风。Firefox按钮实际发送待处理语音ID，输入故障后仍能发送文字。最后旧轮没有发布WAV，新轮文件可下载，采集/会话/输出/计时线程均释放；155项自动化测试通过。

使用新的独立测试目录复现软件链路；测试会调用当前原生模型，保持它们没有其他请求。可选`--webdriver http://127.0.0.1:4445`连接自己启动的geckodriver：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/probe_voice_interruptions.py \
  --input /home/nvidia/AI_Stack/runtime-data/reachy-r2-input.wav \
  --control /home/nvidia/AI_Stack/runtime-data/voice-replay-check/source.json \
  --output /home/nvidia/AI_Stack/runtime-data/voice-replay-check/result.json
```

软件R3.turn按新范围完成；完整声学、人的误触发率和双讲仍为后置项。接下来完成剩余手势质量、软件压力/完整生命周期、部署切换和B0-SW，再交给用户从视觉开始分析与优化。


## R4：托管部署、完整应用启停与回退（2026-10-16）

新入口只托管四个服务：llama.cpp 图文、llama.cpp ASR、OpenVoiceStream 和 Reachy 应用。应用内唯一相机采集为人体/脸/手与网页提供共享帧；不再依赖旧 `tools.project` 启动模型。配置被冻结后由 systemd 单元统一使用；模型、线程、并发、上下文和缓存参数与旧配置相同。命令和管理边界见 [部署说明](../deploy/systemd/README.md)。

[部署报告](../benchmarks/results/reachy-r4-deployment-summary.json)记录 20 次连续完整应用启停，随后在状态观察修复后再验 1 次；三个原生模型服务保持驻留。每次停止均检查 cgroup 无残留，启动核对模型身份、实际监听者、新鲜相机及有效配置。文字、画面问答与合成中文文件语音闭环成功；强制终止真实应用进程后，由 systemd 自动重启，下一次问答成功。166 项自动化测试通过。

两次真实回退都从固定旧源码 `AI_Stack/archives/legacy-runtime-972de50` 恢复旧七服务，并取得新鲜 RGB 和对齐深度。旧模型、数据和环境保留。此证据证明准备工件上的软件部署与回退，不代表完成断网冷启动、干净机器重建、B0 或 8 小时长稳。

真实手势负例、遮挡与多手样本按用户“稍后再测”继续待验；不将缺少样本记作通过。


## R4：真实软件共存与离线检查（2026-10-16）

[共存报告](../benchmarks/results/reachy-r4-coexistence-summary.json)中的有效复测覆盖持续 D435i/人体/脸/手、20 次模型忙时取消、20 次文字或画面问答、20 次中文文件语音与 80 次并发拒绝；第 5 个网页查看者收到 403。取消输出不可下载，下一轮正常，最终网页连接与待发送事件归零，仅保留 10 个回复 WAV，没有暂存文件，应用正常退出并恢复默认服务。176 个周期样本中的相机/感知均就绪，可用内存最低约 4.35 GiB。该数值属于本次有限压力观察，不作为 B0 性能结果。

第一次尝试的 20 轮交互均成功，最终进程列表断言失败，记录保留。后续观察确认 OVS 会周期性创建 `tegrastats` watchdog 子进程，模型服务本身没有重启；复测改为核对实际监听 PID、创建时间、supervisor 和重启计数。测量工具本身需要被验证。

短窗口中应用 PSS 约增加 11.8 MiB，FD 从 56 到 57，冷却窗口为 57–58；所有应用进程正常退出。这里没有把小幅增长直接判为泄漏，也没有宣称长时间无泄漏，原始曲线供基线及后续用户分析使用。

[离线报告](../benchmarks/results/reachy-r4-offline-summary.json)记录四服务在只有 `lo` 的独立网络空间中启动，向非回环地址连接得到内核 `ENETUNREACH`；实际模型进程 UID 为 1000。文字、摄像头和文件语音交互通过，四服务退出码均为 0，无残留，主机网络拓扑不变，原托管服务恢复 READY。此检查不代表从零构建、操作系统重启或 8 小时运行。

## B0-SW 输入与协议冻结

[输入清单](../benchmarks/manifests/reachy-b0-inputs-v1.json)锁定 3 份图像和 20 份合成中文语音，原文件位于 `~/AI_Stack/datasets/reachy-b0-sw-v1`，不上传音频/图像。[采集协议](../benchmarks/manifests/reachy-b0-sw-v1.json)固定视觉单模块、固定到达率回放、实机周期采样、单体语音/图文、共存、取消和过载的测量口径；正式数据尚未采集。

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/prepare_baseline_inputs.py
```

存在清单时该命令只验证文件哈希并复用，不重新生成语音。首次准备需要本机 TTS 服务；不得用之后重新合成、波形不同的文件悄悄替换冻结输入。合成语音的参考文字是 TTS 输入提示，尚未经独立转写；后续 CER 只报告合成闭环一致性，不冒充真人噪声条件下的 ASR 准确率。

下一步先采集视觉各阶段与帧龄，再采集单模块/共存及软件端到端指标，形成完整 B0-SW、架构导读和复现命令。性能优化仍留给用户接手后逐步完成。


## B0-SW：视觉部分已采集（V1/V2，2026-10-16）

[机器可读汇总](../benchmarks/results/reachy-b0-vision-summary.json)、[PNG 图表](../benchmarks/results/reachy-b0-vision.png)和 [SVG 图表](../benchmarks/results/reachy-b0-vision.svg)记录当前结果。应用推理代码、模型精度、并发、线程与频率参数保持原值，新增代码位于 `benchmarks/`，仅用于测量。正式 B0 尚未整体完成。

V1 使用三个固定输入，每模块、每输入各 20 组原始调用/插桩调用交替对照，共 180 组，输出数值全部匹配。输入解码和结果 JSON 转换不计入调用耗时；第一次切换输入的调用及预热另存。下表仅展示固定 640×480 人物图片的原始调用：

| 模块 | P50 | P95 | 口径 |
|---|---:|---:|---|
| YOLOv8n 人体 | 55.32 ms | 59.15 ms | 预处理＋同步 TensorRT 调用＋后处理 |
| YuNet 人脸 | 10.62 ms | 11.88 ms | OpenCV 原生检测及 Python 边界处理 |
| 掌检＋手关键点 | 37.82 ms | 46.98 ms | 同帧掌检与实际命中的手部 ROI；不是单个关键点模型 |

人体模块的分段 P50 约为预处理 1.66 ms、同步原生调用 11.14 ms、后处理 41.98 ms。这是后续用户定位工作的候选方向；未改写算法或提高配置频率。分段分位数不能简单相加代替总耗时分位数。TensorRT 调用含拷贝、执行与同步，不能称为纯 GPU kernel 时间；YuNet 原生调用含其内部预后处理。手关键点按实际 ROI 统计，空白输入没有该调用，不能填成“推理耗时 0”。

V2 每条件记录 60 秒、600 个独立周期样本，回放输入按 10 Hz 的绝对时钟到达，延误时跳过已错过的输入，不随推理完成速度放慢。回放按同一图像循环边界对齐；原生 Camera 的 JPEG 环和 Perception worker 继续使用，预热至少覆盖完整缓存。4B 使用一个持续请求客户端，并在正式窗口前完成至少五次请求；最后一个未完成请求在窗口末按计划取消并确认后端空闲。

| 条件 | 感知完成率 | 最新结果帧龄 P95 | 无结果比例 | 采样最低可用内存 |
|---|---:|---:|---:|---:|
| 固定回放：仅视觉 | 5.00 帧/s | 380.54 ms | 0.0% | 11.06 GiB |
| 固定回放：模型驻留空闲 | 5.00 帧/s | 363.72 ms | 0.0% | 5.40 GiB |
| 固定回放：持续 4B 请求 | 5.00 帧/s | 375.02 ms | 0.0% | 5.04 GiB |
| D435i 实机：模型驻留空闲 | 5.00 帧/s | 342.43 ms | 0.0% | 5.04 GiB |

帧龄从主机读到图像开始计算，不是传感器曝光或屏幕呈现时延。“无结果”指没有新鲜、已完成的结果；无人/无手的有效检测结果仍算可用。实机窗口未保存原始相机画面，也不与固定回放直接计算加速比。

这些是每条件一次的有序短窗口，动态时钟、热状态和调度历史仍有影响。4B 条件没有在本次窗口造成无结果样本或明显吞吐下降；不能据此宣称所有负载都不会争用。1 Hz 的 GPU load 快照可能漏掉短 TensorRT 工作，采样为零不表示模型未在 GPU 执行；原始 tegrastats 保留全进程温度/时钟，含预热，不声称已精确映射到每个阶段。PSS、RSS、系统可用内存也不能相加当作 GPU 分配量。

复现会临时停止或切换本项目服务，使用新输出目录，完成后恢复原配置；测量期间不并行构建、下载或运行其它测试：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python benchmarks/vision_modules.py \
  --output /home/nvidia/AI_Stack/runtime-data/benchmarks/vision-v1-new
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python benchmarks/vision_stream.py \
  --output /home/nvidia/AI_Stack/runtime-data/benchmarks/vision-v2-new
```

原始运行目录为 `reachy-b0-v1-20261016-a` 和 `reachy-b0-v2-20261016-a`，均在 `AI_Stack/runtime-data/benchmarks/`；保存了源文件快照、配置、逐事件记录、资源 CSV、原始日志和哈希索引。171 项自动化测试通过，四个默认服务已恢复。冻结协议中的状态字段记录冻结时刻；当前完成范围以任务清单和最新数据汇总为准。下一步采集 S1/L1/T1/E1，并形成完整 B0 与用户导读。
