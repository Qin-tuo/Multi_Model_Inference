# 项目内运行依赖

当前服务使用仓库内 `05_runtime/`。原 `/home/nvidia/AI_Stack` 保留为回退来源；当前启动路径、模型读取和数据库写入不再依赖它。独立性通过私有 mount/network namespace 验证：隐藏原 AI_Stack，网络只保留 lo，真实启动四个后端并完成文字、相机、语音接口及完整文件语音对话。主机目录和网络保持原状。

## 目录

| 路径（相对 05_runtime） | 内容 |
|---|---|
| `frameworks/llama.cpp/` | 固定上游源码与 Git 版本信息 |
| `frameworks/openvoicestream/` | 固定 OVS 源码及 Jetson voice engine 子模块 |
| `frameworks/opencv-zoo/`、`frameworks/tensorrt-samples-16ed0636/` | 当前视觉模型实现和原生推理辅助代码 |
| `builds/llama/bin/` | llama-server、llama-bench、llama-cli 与同版本共享库 |
| `environments/reachy-headless/`、`environments/speech/` | 已准备的 Python 环境，脚本路径已迁移 |
| `models/` | 4B/ASR GGUF、视觉 ONNX/TRT、Silero、Matcha/Vocos 工件 |
| `datasets/` | 当前 B0 固定图像和合成语音输入 |
| `baselines/b0/effective-config.json` | 原冻结配置的原字节副本 |
| `runtime-data/reachy-headless/` | 当前冻结运行配置、systemd 文件与服务日志 |
| `runtime-data/reachy-actions/`、`runtime-data/reachy-audio/` | 工具数据库和回复音频 |
| `validation/` | 本机验证过程及诊断记录 |
| `bundle.manifest.json` | 工件逐文件 SHA256 和符号链接清单 |
| `exports/` | 用于复制到其他兼容环境的运行依赖包 |

模型与原生文件仍按原始锁校验。`03_models` 和冻结 B0 JSON 保留原字节，历史绝对路径在读取时由 `project_paths.artifact_path` 转为项目内路径；缺失文件会失败，不回退读取旧 AI_Stack。B0 配置校验先核对原配置副本的哈希，再比较路径归一化后的全部参数。

## 启动

在项目根目录执行 `./run.sh services up`。不需要激活旧 `.venv`；开发时可使用 `source ./05_runtime/environments/reachy-headless/bin/activate`。原生程序通过启动器设置本地 `LD_LIBRARY_PATH`，不修改已锁定的 ELF 文件或模型引擎字节。

## 导入、校验和打包

从这台机器原有、版本匹配的 AI_Stack 导入（目标 05_runtime 必须不存在）：

```bash
python3 scripts/import_runtime.py --source /home/nvidia/AI_Stack
```

导入脚本只复制静态依赖和基线输入，不自动接管服务或复制正在使用的数据库。本次迁移在停止并确认原服务退出后复制运行数据，保留原件，再重新注册项目内单元；后续若重复迁移已有状态，应同样先停服务并核验注册单元归属。

```bash
python3 scripts/runtime_bundle.py inventory
python3 scripts/runtime_bundle.py verify
python3 scripts/runtime_bundle.py pack --output 05_runtime/exports/runtime-20261019.tar
```

运行包默认排除数据库、音频输出、缓存、日志和验证记录。它补充 Git 仓库中的代码，不包含 Jetson 操作系统。打包命令输出归档 SHA256；本次包的索引见 [项目运行依赖锁](project-runtime.lock.json)。

## 从运行包恢复

在相同架构、兼容系统环境的项目副本内，保持目标 `05_runtime` 不存在，使用已核对的 SHA256：

```bash
python3 scripts/runtime_bundle.py unpack --archive /path/to/runtime-20261019.tar --sha256 <包的SHA256>
./run.sh services check
./run.sh services up
```

恢复器验证包和文件摘要，重建已声明的系统 Python/TensorRT 链接，并调整环境脚本中的项目路径。源码和模型哈希保持可查。服务注册仍执行归属检查；同一台机器更换项目位置前应先停止并卸载旧注册，不覆盖其他实例。

## 主机依赖与边界

当前工件针对这块 Orin NX：aarch64、Python 3.12、Jetson Linux 39.2.x、CUDA 13.2、TensorRT 10.16.2.10。系统 Python、驱动、CUDA、TensorRT、音频/视频设备库继续由系统维护；语音环境沿用已有 system-site-packages 设置，依赖系统 NumPy、SciPy、PyYAML 等包。平台观测与系统链接见项目运行依赖锁。

本次验证证明在当前兼容主机上可以离开 AI_Stack 和外网运行；尚未证明跨 JetPack/跨硬件的直接兼容，TensorRT 引擎在不兼容环境需要重建。原源码及外部环境仍保留；迁移前项目版本为 `before-project-runtime-20261019`。
