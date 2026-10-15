# Reachy隔离迁移：当前操作入口与边界

2026-10-15。R1现已接通无机身网页、共享相机、人脸感知和4B问答；R2已接入中文文件语音、VAD与分句合成；人体/手部和完整整轮取消继续推进，物理音频仍待接线。

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

`run_vision_chat.py`复用同一入口与生命周期，不复制主控；`run_voice_chat.py`复用同一网页/会话生命周期，默认关闭相机。一次只接纳一个请求，多余请求返回409；当前尚不承诺语音、手势及生成中的即时取消。

## 配置与模型

沿用上游typed dataclass和overlay组织。默认加载`config/settings.yaml`及`config/settings.orin-nx-headless.yaml`；显式`--config`或`REACHY_ASSISTANT_CONFIG`可选择另一个完整平台overlay。未知字段、错误类型、缺失文件、机器人/RAG字段和远端模型地址均明确失败。输出的effective config是当前参数依据。

上游YAML中的`vad.use_silero`未被原dataclass消费，已删除这个无效配置项；真正的VAD接入与验证在后续阶段进行。

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

尚待完成：人体/手部感知、挥手/静止举掌、当前会话与有限工具、统一取消和后端释放、完整故障矩阵、物理音频及B0基线。

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

移除了结束所有录音进程、关闭桌面PulseAudio及改写其配置的逻辑。只管理自身录音/播放进程和AEC模块，AEC名称按实例隔离，选择输出不修改系统默认扬声器。音频错误会结束本轮并返回失败；整轮即时取消和GPU释放仍是R3任务。
