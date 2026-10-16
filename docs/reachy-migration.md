# Reachy隔离迁移：当前操作入口与边界

2026-10-15。R1现已接通无机身网页、共享相机、人脸感知和4B问答；R2已接入中文文件语音、VAD与分句合成；人体/手部也已接入共享帧链路；真人手部正例、手势与完整取消继续推进，物理音频仍待接线。

## 目录与来源

- 运行旧系统：`/home/nvidia/Multi_Model_Inference`，仍在`main`。
- 新派生工程：`/home/nvidia/AI_Stack/projects/reachy-headless`，分支`migration/reachy-headless`。
- 新Python环境：`/home/nvidia/AI_Stack/environments/reachy-headless`，不继承旧环境的site-packages。
- 只读上游：`/home/nvidia/AI_Stack/research/reachy-mini-jetson-assistant-2673202`。
- 旧系统快照：tag `legacy-before-reachy-v3-20261015`；可验证bundle和配置在`/home/nvidia/AI_Stack/archives/reachy-r0-20261015`。
- [导入清单](../deploy/reachy-import.json)记录46个原样导入文件的SHA256。提交`2084ded`保留裁剪前原样导入；随后差异可直接比较。旧源码、工具、测试保留于旧标签，未删除运行目录或数据。

## 已可运行的入口

```bash
cd /home/nvidia/AI_Stack/projects/reachy-headless
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py config
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '只回答：连接成功' --json-output
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py ask '图中的主要交通工具是什么？' --image /home/nvidia/AI_Stack/runtime-data/benchmarks/vision-input-diagnostic-v1/decoded.png --json-output
```

CLI ask是单次文字/图片接口；网页已支持中文语音文件输入和回复WAV下载。多轮会话、工具及物理语音验收仍待后续阶段。图像例子使用历史固定测试图片，不是实时摄像头。JSON中的单次耗时用于诊断，不是冻结的性能基线。

网页默认启用本机ASR/TTS和Silero，以文件模式处理语音，不占用声卡：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/prepare_face.py --check-only
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/prepare_speech.py --check-only
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python run_web_vision_chat.py --no-camera
```

浏览器访问`http://127.0.0.1:8090`，输入中文问题。`--no-camera`关闭摄像头。上传16kHz、单声道PCM16 WAV，可下载真实合成回复。`--no-audio`显式关闭语音后端，只测试文字/视觉。默认文件模式不会把生成文件记为扬声器播放。

要验证真实相机，在旧相机服务停用期间运行新入口，退出后恢复原服务；只管理自己项目的相机服务：

```bash
systemctl --user stop mmassistant-camera.service
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python run_web_vision_chat.py
systemctl --user start mmassistant-camera.service
```

无论新应用是否正常退出，都要恢复原先启用的相机服务。自动化测试必须使用`try/finally`或shell退出trap，并在恢复后检查新鲜RGB和对齐深度。实测一次恢复时出现过深度启动警告；后续受控重启后两路均恢复，故服务PID存在不能作为恢复完成证据。

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
| llama.cpp 8088 | 旧systemd `mmassistant-llama` | 只连接，不重复启动或终止 |
| ASR 8089 / 语音8621 | 现有systemd后端 | R2已通过接口复用，不另起第二份模型进程 |
| D435i | 旧systemd `mmassistant-camera` | R1真机测试已受控切换并恢复；同一时刻只允许一个采集者 |
| 原视觉/姿态/Qdrant | 旧项目服务 | 当前保持原状；新应用不导入ROS/Pipecat/记忆 |
| 麦克风/扬声器 | 当前未接入指定实物 | 不生成采集或播放成功回执 |
| 新Python客户端 | 每次CLI请求 | 关闭自身HTTP流，释放后检查模型槽空闲 |
| Web端口8090 | 新应用可手动启动 | 只管理本项目实例，不自动切换默认服务 |

所有新安装仅进入独立环境。不更换BSP、CUDA或TRT，不运行上游清理其他相机/音频进程的函数。依赖锁为`requirements.lock`，覆盖当前应用、CPU VAD及开发测试；ASR/TTS/LLM复用本机现有GPU服务，不把它们的Python依赖装入新应用环境。

## 验证与未完成项

[阶段汇总](../benchmarks/results/reachy-r1-summary.json)、[真实网页文字](../benchmarks/results/reachy-r1-web-text.json)、[真实D435i/网页/VLM](../benchmarks/results/reachy-r1-camera-web.json)、[真人脸检测](../benchmarks/results/reachy-r1-live-face.json)、[旧RGB/深度恢复](../benchmarks/results/reachy-r1-legacy-restore.json)。

43项R1适用测试通过。Silero模型集成测试属于尚未接入的R2，本次命令明确排除该文件，不宣称全部模型或完整应用已验收。测试命令见阶段汇总。

YuNet固定版本和哈希见`models/perception.json`，权重与许可证存放AI_Stack，加载时验证哈希，运行时不联网下载。后端为OpenCV CPU；没有声称该小模型已使用GPU。缺失工件可显式运行`python scripts/prepare_face.py`准备。

相机使用本机已验证的D435i RGB节点（当前解析为/dev/video4），不使用上游默认元数据节点/dev/video1。帧带epoch、序号和主机收帧单调时间；不是硬件曝光时间。失效/过期帧不能继续交给模型。网页预览使用与人脸结果相同的原帧，避免把旧框套在新画面上。

真人脸验证连续取得10个不同帧的正例；这证明当前环境的检测连通，不是识别精度统计或身份验证。浏览器静态页面用本机Firefox实际渲染，HTTP/WebSocket验证使用真实模型；原始相机图片与截图保存在AI_Stack，没有提交Git。

当前尚待完成：真人手部正例及挥手/静止举掌、当前会话与有限工具、完整故障矩阵、物理音频及B0基线。R2感知和R3软件取消的新增证据见下文。

回退时退出新应用即可继续旧系统。默认服务未切换；如将来测试暂停旧相机，必须记录并恢复原服务状态。不要在运行目录执行`git reset --hard`来替代受控回退，不覆盖原用户数据。本次快照保存代码/配置，不宣称已备份在线数据库。


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

`TurnController`仍是唯一轮次所有者。网页停止使用`POST /api/cancel`并携带当前`turn_id`；迟到的旧轮次取消不能中断下一轮。取消标记先阻断新输出和后续模型调用，再中断自身HTTP连接、停止自身播放器。HTTPX的连接trace让应用在响应头到达之前也能中断阻塞读取；只shutdown该操作持有的socket，连接最终关闭由HTTPX负责。

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

这些单次时长属于功能诊断，不是B0或性能承诺。ASR当前已提交的识别和Matcha当前合成步骤仍可能运行至安全结束；不宣称CUDA kernel可被即时抢占。实物播放缓冲何时真正静音、AEC/双讲、手势触发、工具取消边界、模型崩溃/重启矩阵及长期持续运行尚未通过验收。
