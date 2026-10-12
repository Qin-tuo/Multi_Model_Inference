# Jetson 本地多模态助手

在 Jetson Orin NX 16GB 上构建具有中文语音、持续视觉、图文理解、长期记忆、打断、工具和提醒能力的本地助手。整体基底采用 Pipecat，记忆采用 Mem0，提醒采用 APScheduler；业务代码负责模型接入、策略和性能测量。实施任务与验收见 [项目计划](02-jetson-multimodel-inference.md)。

## 当前进度

项目处于 J0 依赖核验与框架接入阶段。26项框架/插件测试、真实GPU文本/图文推理、BGE转换回归、10项语义召回和3项跨应用进程记忆问答已通过。D435i双路640×480采集实测接近30Hz。完整模型质量、语音、视觉检测和整机验收仍待完成。逐项进度见 [progress.json](benchmarks/manifests/progress.json)，测试范围见 [测试报告](benchmarks/results/framework-tests.json)。

| 功能 | 组件 | 当前状态 |
|---|---|---|
| 主框架 | Pipecat 1.12.0 原生 PipelineWorker/WorkerRunner | 真实llama文本与记忆问答通过；音频链路待接入 |
| 文本/图文模型 | llama.cpp CUDA + Qwen2.5-VL-3B Q4_K_M | 37/37层GPU offload及CUDA图像编码；两项问答sanity通过 |
| 长期记忆 | Mem0 2.2.1 + 本地Qdrant + SQLite历史/journal | BGE CPU ONNX、重启召回、更正/遗忘问答通过；完整语义质量集待验证 |
| 提醒 | APScheduler 3.11.3 + SQLAlchemy | 持久调度、取消、去重、待播通知测试通过；播报消费待接入 |
| 语音 | OVS Qwen3-ASR 0.6B + Matcha/Vocos | 上游旧engine与当前系统不匹配，需要移植和重建 |
| 视觉 | yolo_ros/TensorRT + RealSense/ROS 2 | ROS/RGB-D采集通过；检测/姿态GPU模型尚未部署 |

Qwen3B作为首轮功能与容量基线。模型身份和产物放在 [manifest](models/manifest.json)，框架适配通过配置读取；更换模型后重新验证质量、上下文预算和共存容量。Edge-LLM留到后续性能对比。

## 使用现有入口

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

准备命令固定源码commit、模型revision与SHA256，下载可续传；构建与权重缓存在 `~/.cache/multi-model-assistant/`。smoke临时启动真实llama-server，执行中文文本和合成图像问题，结束后关闭；它只证明加载和推理路径，不等于JT验收。

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

音频设备确定为ReSpeaker Lite双麦，用户当前无法连接麦克风/扬声器，真机录播/AEC/打断列为待验证。D435i最新链路为5Gbps；早期USB2枚举仅保留为排查记录。

语音与视觉准备细节分别见 [语音部署说明](deploy/speech/README.md)、[视觉部署说明](deploy/vision/README.md)。完整 `up/down/interact/bench/acceptance` 命令将在对应里程碑接入，当前不可作为已实现接口使用。

## 数据和上游

运行数据默认位于 `~/.local/share/multi-model-assistant/`，通过 `ASSISTANT_DATA_DIR` 覆盖；缓存根目录通过 `ASSISTANT_CACHE_DIR` 覆盖。仓库提交源码、配置、模型清单和脱敏验证报告。现有 `references/` 保留。

上游版本见 [依赖锁](deploy/upstreams.lock.json)。Mem0的局部历史清除补丁位于 `patches/mem0/`，用于落实显式遗忘；SDK的Qdrant/SQLite操作不是跨存储原子事务，适配器通过持久操作状态与删除屏障恢复，仍需整机故障验收。原始自建运行时/记忆原型已归档到本机缓存，不参与当前产品。
