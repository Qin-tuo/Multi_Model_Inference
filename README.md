# Jetson 本地多模态助手

在 Jetson Orin NX 16GB 上构建具有中文语音、持续视觉、图文理解、长期记忆、打断、工具和提醒能力的本地助手。整体基底采用 Pipecat，记忆采用 Mem0，提醒采用 APScheduler；业务代码负责模型接入、策略和性能测量。实施任务与验收见 [项目计划](02-jetson-multimodel-inference.md)。

## 当前进度

项目处于 J0 依赖核验与框架接入阶段。框架/插件测试、真实GPU文本/图文推理、BGE转换回归、10项语义召回和3项跨应用进程记忆问答已通过。D435i双路640×480采集实测接近30Hz。真实Silero→ASR→LLM→TTS文件问答已跑通；时间查询、记忆写入及提醒创建的真实模型工具测试已通过，并增加未实际执行写入时的回复校验。完整模型质量、物理语音、ROS视觉检测和整机验收仍待完成。逐项进度见 [progress.json](benchmarks/manifests/progress.json)，测试范围见 [测试报告](benchmarks/results/framework-tests.json)。

| 功能 | 组件 | 当前状态 |
|---|---|---|
| 主框架 | Pipecat 1.12.0 原生 PipelineWorker/WorkerRunner | 持续文本会话、记忆/笔记工具、进程重启召回与逻辑取消通过初测；实时设备和语音工具整合继续推进 |
| 文本/图文模型 | llama.cpp CUDA + Qwen2.5-VL-3B Q4_K_M | 37/37层GPU offload及CUDA图像编码；两项问答sanity通过 |
| 长期记忆 | Mem0 2.2.1 + 本地Qdrant + SQLite历史/journal | BGE CPU ONNX、重启召回、更正/遗忘问答通过；完整语义质量集待验证 |
| 提醒 | APScheduler 3.11.3 + SQLAlchemy | 到期文字通知、静音、过期、重启去重通过；Pipecat音频完成协议通过夹具测试，实机播报待验证 |
| 语音 | OVS + llama.cpp Qwen3-ASR 0.6B + TensorRT Matcha/Vocos | 本机引擎与文件语音问答通过；真人/噪声/AEC待验证 |
| 视觉 | yolo_ros/TensorRT + RealSense/ROS 2 | 对齐RGB-D采集、检测/姿态TensorRT FP32单图对齐通过；ROS五个包编译通过，完整GPU ORT与ROS运行验证仍待完成；FP16失败样例保留 |

Qwen3B作为首轮功能与容量基线。模型身份和产物放在 [manifest](models/manifest.json)，框架适配通过配置读取；更换模型后重新验证质量、上下文预算和共存容量。Edge-LLM留到后续性能对比。

## 使用现有入口

本机外部框架、模型、构建产物与 Python 环境统一放在 **`/home/nvidia/AI_Stack/`**。其中 `frameworks/llama.cpp/` 是 llama.cpp 源码，`builds/llama/bin/llama-server` 是服务程序。目录与版本入口见 [依赖目录说明](deploy/dependencies-layout.md)；本机总目录内还有 `README.md` 和完整 Python 依赖清单 `dependencies.json`。原缓存路径及项目 `.venv*` 保留为兼容链接。

### 持续文本交互

先运行本地模型服务，再在另一终端开启会话：

当前GPU版ORT仍在构建，语言/语音模型服务暂时关闭。发生过编译与驻留模型叠加引起的OOM，启动入口现会在重型ORT构建存活时明确拒绝启动模型；待构建结束后再使用下面的命令。

```bash
python3 tools/project.py serve --component llama
# 另一个终端：
python3 tools/project.py interact --user owner
```

可直接输入“请记住，我喜欢茉莉花茶。”“我喜欢喝什么茶？”“创建笔记，标题是采购，内容是周末买绿茶和杯子。”“请查询我的笔记。”或“五分钟后提醒我休息。”。`/new` 开启新会话，`/stop` 停止当前回答，`/memories`、`/notes`、`/reminders`、`/history` 查询实际记录，`/quit` 退出。`--user guest` 使用独立档案；`--session-id ID` 恢复指定档案已输出的会话上下文；`--input-file PATH` 顺序执行最多64KiB的文本脚本。

此入口复用一个持续运行的 Pipecat worker 和原生工具 continuation。模型请求及工具回传请求均经 llama.cpp 实际模板与 tokenizer 检查；优先移除旧轮次，当前问题过长则明确失败。默认会话文本保留7天，仅成功输出的完整文本进入已完成历史；更正/遗忘会清除相关来源和派生会话记录，并失效当前旧上下文。笔记和偏好分别从各自存储读取，购物笔记不作为偏好证据。用户明确操作最多修复重试一次，未实际成功时不会口头确认。

当前取消验证覆盖原生 interruption、HTTP生成请求停止、服务slot空闲及下一轮恢复。控制台按完整回答输出，尚未实现逐token显示；此验证不证明物理停播、AEC或GPU资源释放。到期提醒已在控制台按“提醒：内容”显示，等待当前回答结束再输出。完整语音助手验收仍未完成。

提醒静音使用 `/mute-reminders on`、`/mute-reminders off`，设置按档案持久保存。未开始输出的失败最多重试3次，默认间隔5秒；提醒到期后默认有效1小时，可在`configs/runtime.yaml`配置。已开始输出但未确认的提醒保留为`uncertain`，重启后不自动重播，可查询后取消或重新设置。文字显示与音频播放分开记录。

模型服务不可用时，可独立运行文字提醒消费者：

```bash
python3 tools/project.py notifications --user owner
```

同一数据目录只运行一个消费者；`interact`已经包含通知消费。独立模式不需要加载语言模型。跨进程静音、过期、取消、主人/访客隔离及完成后不重播见[文字通知报告](benchmarks/results/reminder-delivery-smoke.json)。

音频通知接入采用 Pipecat 原生 `TTSSpeakFrame` 和有序输出标记，记录实际输出尝试，等待输出设备排空后才确认；合成结束、普通停止讲话事件或部分音频写出都不能作为成功依据。LocalAudio适配提供PortAudio排空接口，但麦克风/扬声器未连接，当前只通过标明为fixture的队列与故障测试，尚未完成完整语音会话接入和声学验收。[PortAudio完成语义](https://portaudio.com/docs/v19-doxydocs/portaudio_8h.html)

实际测试见[持续会话报告](benchmarks/results/session-smoke.json)、[独立进程与取消报告](benchmarks/results/application-smoke.json)。这些是合成开发样例；模型曾漏调用工具或把购物内容误当偏好，失败样例与校验修正已保留，不能以这几条通过替代完整工具质量集。

### 依赖与单组件检查

Python依赖使用独立虚拟环境；锁文件记录当前ARM64/Python3.12验证环境。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
python3 tools/project.py doctor
python3 tools/project.py test --suite core
```

`doctor`为只读探测，有未验证外设或缺失依赖时返回非零，并将本机原始报告写入用户数据目录。测试真实调用SDK与数据库；用于embedding/LLM的测试夹具明确标记，不替代模型质量验收。

准备 llama.cpp（首次需要网络、CUDA开发环境和足够磁盘）：

```bash
python3 tools/project.py prepare --component llama --jobs 2
python3 tools/project.py smoke --component llama
```

准备命令固定源码commit、模型revision与SHA256，下载可续传；本机构建与权重分别存放在 `~/AI_Stack/builds/`、`~/AI_Stack/models/`，脚本原缓存路径通过兼容链接访问。smoke临时启动真实llama-server，执行中文文本和合成图像问题，结束后关闭；它只证明加载和推理路径，不等于JT验收。

Pipecat真实文本入口需要本地llama-server已监听 `127.0.0.1:8088`、别名`assistant-vlm`：

```bash
python3 tools/project.py serve --component llama
```

在另一终端运行：

```bash
.venv/bin/python -m app.pipecat_bot --text-smoke \
  --prompt '你好，请用中文回答' \
  --output benchmarks/results/pipecat-text-smoke.json
```

准备并验证本地BGE检索：

```bash
python3 tools/project.py prepare --component memory
python3 tools/project.py smoke --component memory
.venv/bin/python tools/assistant_memory_smoke.py
```

最后一条命令需要llama服务已启动，使用独立临时库及合成偏好，验证保存/更正/遗忘后的真实回答；不读写用户记忆。实际查询指定 `--memory-dir` 与手选 `--user`，记忆证据按本地llama tokenizer限制在512token内。BGE运行使用CPU ONNX，PyTorch CPU仅用于转换和参考回归；锁文件固定官方ARM64 CPU wheel，避免额外安装CUDA运行库。

真实模型工具调用测试使用独立临时库：`.venv/bin/python tools/tool_smoke.py`。当前 llama.cpp 使用其上游 Qwen2.5 工具模板，原 Qwen2.5-VL 内嵌模板未提供工具能力。工具按当前明确请求筛选，参数经校验后交给 Mem0/APScheduler；未收到实际写入成功结果时，Pipecat 输出插件会阻止口头成功确认。三条测试不代表完整工具质量或语音打断验收。

音频设备确定为ReSpeaker Lite双麦，用户当前无法连接麦克风/扬声器，真机录播/AEC/打断列为待验证。D435i最新链路为5Gbps；早期USB2枚举仅保留为排查记录。

视觉数据接入已完成真实D435i→ROS→有界RGB-D缓存→Pipecat图像请求帧的初测。缓存使用2秒窗口，128MiB总预算中为传输与转换预留16MiB；仍被消费者引用的帧继续计入缓存占用。按提问时刻选择150ms以内的画面，停止采集后拒绝旧画面，驱动重启会更换数据代次。21次图像请求及一次驱动重启检查见[视觉接入报告](benchmarks/results/vision-bridge-smoke.json)。本项未包含检测推理或VLM图像问答，不代表JT-03已通过。

从项目根目录验证上述接入（脚本会启动、重启并关闭其拥有的相机进程，不保存原始图像）：

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python tools/vision_bridge_probe.py --launch-camera
```

语音与视觉准备细节分别见 [语音部署说明](deploy/speech/README.md)、[视觉部署说明](deploy/vision/README.md)。语音服务与文件管线使用部署说明中的独立入口。`interact` 已实现文本入口；完整 `up/down/bench/acceptance` 命令仍待对应里程碑接入，当前不可作为已实现接口使用。

## 数据和上游

本机运行数据位于 `~/AI_Stack/runtime-data/`，原路径 `~/.local/share/multi-model-assistant/` 是兼容入口，通过 `ASSISTANT_DATA_DIR` 覆盖；缓存根目录通过 `ASSISTANT_CACHE_DIR` 覆盖。仓库提交源码、配置、模型清单和脱敏验证报告。现有 `references/` 保留。

上游版本见 [依赖锁](deploy/upstreams.lock.json)。Mem0的局部历史清除补丁位于 `patches/mem0/`，用于落实显式遗忘；SDK的Qdrant/SQLite操作不是跨存储原子事务，适配器通过持久操作状态与删除屏障恢复，仍需整机故障验收。原始自建运行时/记忆原型已归档到本机缓存，不参与当前产品。
