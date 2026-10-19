# 本机 AI 开源框架与依赖总目录

当前 Reachy 应用的主要依赖为 `environments/reachy-headless`、llama.cpp、OpenVoiceStream、OpenCV、TensorRT、Silero、SQLAlchemy 和 APScheduler。当前服务管理见 [Reachy 部署](systemd/README.md)。下表保留整台机器已经整理过的依赖目录；Pipecat、ROS 视觉、Mem0/Qdrant 等属于历史或后置方案，不随新四服务启动。旧工具操作须在 `/home/nvidia/AI_Stack/archives/legacy-runtime-972de50` 执行。

统一入口：`/home/nvidia/AI_Stack/`。业务项目仍在 `/home/nvidia/Multi_Model_Inference/`。框架、模型和运行环境集中放在此处，项目仓库保留业务接入代码、版本锁与验证报告。

| 目录 | 内容 |
|---|---|
| `frameworks/` | llama.cpp、Pipecat、OpenVoiceStream、voice-engine、yolo_ros 的 Git 源码入口；Mem0、APScheduler、Qdrant client、Ultralytics、PyTorch 等已安装包的源码入口；ONNX Runtime 源码及 RealSense ROS 安装包入口 |
| `builds/` | llama.cpp、ONNX Runtime 等编译目录；`llama/bin/llama-server` 是当前推理服务程序，`llama/bin/llama-bench` 是已编译的同版本原生性能工具 |
| `03_models/` | Qwen 图文/语音模型、BGE、YOLO、Matcha/Vocos 及本机构建的 TensorRT 引擎 |
| `environments/reachy-headless/` | 当前 Reachy 派生应用、OpenCV/感知、Silero、SQLAlchemy/APScheduler 和验证工具 |
| `environments/assistant/` | Pipecat、Mem0、APScheduler、BGE、PyAudio 等主应用 Python 环境 |
| `environments/speech/` | OpenVoiceStream/VoxEdge/Piper 等语音 Python 环境 |
| `environments/vision/` | Ultralytics、PyTorch CPU、ONNX 等模型导出/验证环境 |
| `environments/memory-eval/` | CLongEval官方评分函数所需的jieba/rouge/jsonlines等隔离环境，不改变助手运行包 |
| `datasets/CLongEval/` | 固定revision和SHA256的外部中文记忆数据，原始语料不进入业务仓库 |
| `environments/ort-build/` | ORT后续构建和上游lintrunner/clang-format检查的隔离开发环境 |
| `installed/` | 自行编译或官方发布的依赖程序；`onnxruntime-1.28.0-cuda13-sm87/` 包含已编译的 CUDA/TensorRT provider 库 |
| `workspaces/` | ROS视觉工作区及colcon构建/安装目录 |
| `sdk/` | 本机 CUDA 构建所需的 cuDNN/TensorRT 头文件和库链接布局，以及 CUDA 13.2 CCCL 兼容头文件覆盖层 |
| `system/` | 系统 CUDA、TensorRT、cuDNN、ROS 2 Jazzy、PortAudio 的访问入口，实际文件由系统包管理器维护 |
| `runtime-data/` | 本项目运行配置、测试输出与运行报告；后续持久记忆等用户状态也放在运行数据根目录下 |
| `logs/` | 构建、安装、服务与验证日志 |
| `reports/` | 依赖准备、安装过程报告；`system-packages.tsv`记录整台机器的系统安装包版本，范围包括项目之外的系统组件 |
| `fixtures/` | 本机测试输入与模型原始输出，不提交 GitHub |
| `downloads/` | 下载缓存 |
| `archives/` | 已退出当前方案的原型归档 |
| `cache/` | 旧缓存路径的兼容层与零散下载文件；内部目录链接到上述正式分类 |
| `projects/` | 相关业务项目的目录入口 |
| `references/` | 原项目参考资料的链接入口，保留原有 Git 子模块结构 |
| `dependencies.json` | 框架实际路径与 Git commit、各 Python 环境全部已安装发行包的名称/版本/位置、系统主要依赖版本和参考仓库获取状态 |

`frameworks/` 中，有些目录是 Git checkout，有些是指向已安装 Python 包的链接。包入口展示当前实际使用的实现代码；Git 源码及模型的锁定版本分别以项目 `01_src/deploy/reachy-upstream.lock.json`、`01_src/deploy/runtime-artifacts.lock.json`、`01_src/deploy/speech/source-lock.json` 和 `03_models/manifest.json` 为准。运行入口使用各自环境的 `bin/python`；语音环境使用系统 TensorRT Python 包。

常用框架的直接入口：

| 框架 | 本机目录 | 形式 |
|---|---|---|
| CLongEval | `frameworks/CLongEval/` | 固定Git源码，复用官方评分；原始数据在`datasets/CLongEval/` |
| llama.cpp | `frameworks/llama.cpp/` | Git 源码 |
| Pipecat | `frameworks/pipecat/` | Git 源码；实际运行包在 `environments/assistant/` |
| Qdrant server | `frameworks/qdrant/`、`installed/qdrant-1.19.2/` | 固定源码及校验过的官方ARM64程序；本机长期记忆服务 |
| Mem0 / APScheduler / Qdrant client | `frameworks/mem0/`、`frameworks/apscheduler/`、`frameworks/qdrant-client/` | 指向实际使用的已安装包 |
| OpenVoiceStream / voice-engine | `frameworks/openvoicestream/`、`frameworks/voice-engine/` | Git 源码及其子模块 |
| yolo_ros / ONNX Runtime | `frameworks/yolo_ros/`、`frameworks/onnxruntime/` | Git 源码 |
| Hugging Face C++ | `frameworks/huggingface-hub-cpp/` | 指向 ROS vendor 下载的 Git 源码 |
| Ultralytics | `frameworks/ultralytics/` | 指向视觉环境已安装包 |
| PyAudio / PortAudio | `frameworks/pyaudio/`、`system/portaudio` | Python 本地音频绑定及系统库入口 |
| JioNLP / jiojio | `frameworks/jionlp/`、`frameworks/jiojio/` | 已安装中文时间语义解析库及其依赖 |
| CUDA / TensorRT / ROS | `system/cuda/`、`system/tensorrt-sdk/`、`system/ros2-jazzy/` | 系统安装的访问入口 |

原 `references/repo/` 的九个学习参考子模块当前尚未初始化；清单明确标记 `checkout_available=false`。这些参考仓库不参与助手运行，目录入口不代表源码已下载。系统依赖仍由包管理器维护，`system/` 为统一访问入口。

为保持既有 CMake 配置、虚拟环境脚本及服务命令可用，以下旧路径保留为兼容链接：

```text
~/.cache/multi-model-assistant       -> ~/AI_Stack/cache
~/.local/share/multi-model-assistant -> ~/AI_Stack/runtime-data
项目/.venv                          -> ~/AI_Stack/environments/assistant
项目/.venv-speech                   -> ~/AI_Stack/environments/speech
项目/.venv-vision                   -> ~/AI_Stack/environments/vision
```

通过 systemd 启动或复用当前 llama.cpp 服务：

```bash
cd /home/nvidia/Multi_Model_Inference
/home/nvidia/AI_Stack/environments/reachy-headless/bin/python main.py services up --components llama
```

历史整理脚本位于旧源码快照的 `tools/organize_dependencies.py`，重复运行不会覆盖已有目录。迁移时应先短暂暂停使用这些目录的构建/服务进程，完成迁移和路径检查后恢复；不要直接删除兼容链接，否则现有构建缓存和虚拟环境脚本可能失效。模型、框架源码、编译产物和环境不会上传业务仓库；GitHub 保存可以重建它们的脚本、锁文件和本文档。

2026-10-19 清理说明：主仓库旧方案部署脚本及报告已进入 [清理前归档](../../08_docs/archive/repository-cleanup-20261019.json)。上述全机依赖目录保持保留。三个项目 `.venv*` 兼容链接仍被已安装环境中 pip/uvicorn 等脚本的 shebang 引用，删除会使这些脚本无法启动；当前应用入口继续使用 `environments/reachy-headless/bin/python`。
