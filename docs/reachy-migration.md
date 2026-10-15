# Reachy隔离迁移：当前操作入口与边界

2026-10-15。R1现已接通无机身网页、共享相机、人脸感知和4B问答；完整中文音频、人体/手部及整轮取消仍待R2/R3。

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

这是单次请求接口，尚不提供多轮会话、工具或物理语音。图像例子使用历史固定测试图片，不是实时摄像头。JSON中的单次耗时用于诊断，不是冻结的性能基线。

网页入口现已可运行，纯软件模式不占用相机或声卡：

```bash
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python scripts/prepare_face.py --check-only
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python run_web_vision_chat.py --no-camera
```

浏览器访问`http://127.0.0.1:8090`，输入中文问题。`--no-camera`明确关闭摄像头；界面显示音频未启用，不伪造录播结果。

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

`run_vision_chat.py`复用同一入口与生命周期，不复制主控；`run_voice_chat.py`在音频适配完成前明确退出。一次只接纳一个请求，多余请求返回409；当前尚不承诺语音、手势及生成中的即时取消。

## 配置与模型

沿用上游typed dataclass和overlay组织。默认加载`config/settings.yaml`及`config/settings.orin-nx-headless.yaml`；显式`--config`或`REACHY_ASSISTANT_CONFIG`可选择另一个完整平台overlay。未知字段、错误类型、缺失文件、机器人/RAG字段和远端模型地址均明确失败。输出的effective config是当前参数依据。

上游YAML中的`vad.use_silero`未被原dataclass消费，已删除这个无效配置项；真正的VAD接入与验证在后续阶段进行。

模型使用已有`assistant-vlm`本机服务。主模型由既有模型锁固定为Qwen3.5-4B Q4_K_M。客户端每次生成前使用同一请求参数调用原生多模态token计数端点，保留128输出和16 token余量；不按字符估算。显式模型名称不匹配时不自动替换；HTTP失败、异常结束和空输出不能伪装成功。

## 资源所有权

| 资源 | 当前所有者 | 新应用策略 |
|---|---|---|
| llama.cpp 8088 | 旧systemd `mmassistant-llama` | 只连接，不重复启动或终止 |
| ASR 8089 / 语音8621 | 旧systemd `mmassistant-asr` / `mmassistant-speech` | R2再通过接口适配，当前网页和CLI不调用 |
| D435i | 旧systemd `mmassistant-camera` | R1真机测试已受控切换并恢复；同一时刻只允许一个采集者 |
| 原视觉/姿态/Qdrant | 旧项目服务 | 当前保持原状；新应用不导入ROS/Pipecat/记忆 |
| 麦克风/扬声器 | 当前未接入指定实物 | 不生成采集或播放成功回执 |
| 新Python客户端 | 每次CLI请求 | 关闭自身HTTP流，释放后检查模型槽空闲 |
| Web端口8090 | 新应用可手动启动 | 只管理本项目实例，不自动切换默认服务 |

所有新安装仅进入独立环境。不更换BSP、CUDA或TRT，不运行上游清理其他相机/音频进程的函数。依赖锁为`requirements.lock`，当前仅覆盖R1核心及开发测试；不包含尚未验证的GPU Python或语音模型依赖。

## 验证与未完成项

[阶段汇总](../benchmarks/results/reachy-r1-summary.json)、[真实网页文字](../benchmarks/results/reachy-r1-web-text.json)、[真实D435i/网页/VLM](../benchmarks/results/reachy-r1-camera-web.json)、[真人脸检测](../benchmarks/results/reachy-r1-live-face.json)、[旧RGB/深度恢复](../benchmarks/results/reachy-r1-legacy-restore.json)。

43项R1适用测试通过。Silero模型集成测试属于尚未接入的R2，本次命令明确排除该文件，不宣称全部模型或完整应用已验收。测试命令见阶段汇总。

YuNet固定版本和哈希见`models/perception.json`，权重与许可证存放AI_Stack，加载时验证哈希，运行时不联网下载。后端为OpenCV CPU；没有声称该小模型已使用GPU。缺失工件可显式运行`python scripts/prepare_face.py`准备。

相机使用本机已验证的D435i RGB节点（当前解析为/dev/video4），不使用上游默认元数据节点/dev/video1。帧带epoch、序号和主机收帧单调时间；不是硬件曝光时间。失效/过期帧不能继续交给模型。网页预览使用与人脸结果相同的原帧，避免把旧框套在新画面上。

真人脸验证连续取得10个不同帧的正例；这证明当前环境的检测连通，不是识别精度统计或身份验证。浏览器静态页面用本机Firefox实际渲染，HTTP/WebSocket验证使用真实模型；原始相机图片与截图保存在AI_Stack，没有提交Git。

尚待完成：中文ASR/TTS/VAD适配、人体/手部感知、挥手/静止举掌、当前会话与有限工具、统一取消和后端释放、完整故障矩阵、物理音频及B0基线。音频旧fallback中的全局进程清理必须在R2启用前移除。

回退时退出新应用即可继续旧系统。默认服务未切换；如将来测试暂停旧相机，必须记录并恢复原服务状态。不要在运行目录执行`git reset --hard`来替代受控回退，不覆盖原用户数据。本次快照保存代码/配置，不宣称已备份在线数据库。
