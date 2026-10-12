# 本机 AI 开源框架与依赖总目录

统一入口：`/home/nvidia/AI_Stack/`。业务项目仍在 `/home/nvidia/Multi_Model_Inference/`。框架、模型和运行环境集中放在此处，项目仓库保留业务接入代码、版本锁与验证报告。

| 目录 | 内容 |
|---|---|
| `frameworks/` | llama.cpp、Pipecat、OpenVoiceStream、voice-engine、yolo_ros 的 Git 源码入口；Mem0、APScheduler、Qdrant client、Ultralytics、PyTorch 等已安装包的源码入口；ONNX Runtime 源码及 RealSense ROS 安装包入口 |
| `builds/` | llama.cpp、ONNX Runtime 等编译目录；`llama/bin/llama-server` 是当前推理服务程序 |
| `models/` | Qwen 图文/语音模型、BGE、YOLO、Matcha/Vocos 及本机构建的 TensorRT 引擎 |
| `environments/assistant/` | Pipecat、Mem0、APScheduler、BGE 等主应用 Python 环境 |
| `environments/speech/` | OpenVoiceStream/VoxEdge/Piper 等语音 Python 环境 |
| `environments/vision/` | Ultralytics、PyTorch CPU、ONNX 等模型导出/验证环境 |
| `environments/ort-build/` | ORT后续构建和上游lintrunner/clang-format检查的隔离开发环境 |
| `installed/` | 自行编译的依赖安装前缀；正在构建的 ORT 完成后写入这里 |
| `workspaces/` | ROS视觉工作区及colcon构建/安装目录 |
| `sdk/` | 本机 CUDA 构建所需的 cuDNN/TensorRT 头文件和库链接布局 |
| `system/` | 系统 CUDA、TensorRT、cuDNN、ROS 2 Jazzy 的访问入口，实际文件由系统包管理器维护 |
| `runtime-data/` | 本项目运行配置、测试输出与运行报告；后续持久记忆等用户状态也放在运行数据根目录下 |
| `logs/` | 构建、安装、服务与验证日志 |
| `reports/` | 依赖准备、安装过程报告；`system-packages.tsv`记录整台机器的系统安装包版本，范围包括项目之外的系统组件 |
| `fixtures/` | 本机测试输入与模型原始输出，不提交 GitHub |
| `downloads/` | 下载缓存 |
| `archives/` | 已退出当前方案的原型归档 |
| `cache/` | 旧缓存路径的兼容层与零散下载文件；内部目录链接到上述正式分类 |
| `projects/` | 相关业务项目的目录入口 |
| `dependencies.json` | 各 Python 环境内全部已安装发行包的名称、版本与位置，以及系统主要依赖版本 |

`frameworks/` 中，有些目录是 Git checkout，有些是指向已安装 Python 包的链接。包入口展示当前实际使用的实现代码；Git 源码及模型的锁定版本分别以项目 `deploy/upstreams.lock.json`、`models/manifest.json` 为准。运行入口使用各自环境的 `bin/python`；语音环境使用系统 TensorRT Python 包。

为保持既有 CMake 配置、虚拟环境脚本及服务命令可用，以下旧路径保留为兼容链接：

```text
~/.cache/multi-model-assistant       -> ~/AI_Stack/cache
~/.local/share/multi-model-assistant -> ~/AI_Stack/runtime-data
项目/.venv                          -> ~/AI_Stack/environments/assistant
项目/.venv-speech                   -> ~/AI_Stack/environments/speech
项目/.venv-vision                   -> ~/AI_Stack/environments/vision
```

启动当前 llama.cpp 服务：

```bash
cd /home/nvidia/Multi_Model_Inference
python3 tools/project.py serve --component llama
```

整理脚本是项目中的 `tools/organize_dependencies.py`，重复运行不会覆盖已有目录。迁移时应先短暂暂停使用这些目录的构建/服务进程，完成迁移和路径检查后恢复；不要直接删除兼容链接，否则现有构建缓存和虚拟环境脚本可能失效。模型、框架源码、编译产物和环境不会上传业务仓库；GitHub 保存可以重建它们的脚本、锁文件和本文档。
