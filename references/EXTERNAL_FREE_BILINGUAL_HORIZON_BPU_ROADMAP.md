# 外部免费双语地瓜 BPU / Horizon RDK 学习路线

> 路线定位：面向汽车、机器人和视觉端侧的 Horizon / D-Robotics BPU 专线。
> 推荐顺序：先完成 NVIDIA 主路线的通用基础，再把本路线作为第一条国产 BPU 主攻路线，最后进入瑞芯微 RKNN 路线。
> 资料范围：真实存在的公开课程、官方文档、官方 GitHub 项目和公开示例；不把当前仓库内部笔记当作外部参考资料。
> 核验日期：2026-08-21。Horizon 的工具链、BSP、模型格式和 API 与硬件型号强绑定，命令执行前必须按目标板卡版本重新核对。

## 0. 先给结论

你的目标不是只会调用一个 BPU 接口，而是能完成：

<code>PyTorch/ONNX -> 模型检查 -> PTQ/QAT -> OpenExplorer 编译 -> .bin/.hbm -> BPU Runtime -> C++/Python 应用 -> 摄像头/视频/ROS2 -> 性能与精度验收</code>

这条路线和 NVIDIA 路线的差异：

- NVIDIA 的性能问题常落到 CUDA kernel、TensorRT tactic、GPU memory 和 Nsight。
- BPU 的性能问题更多落到模型图是否完整编译、量化误差、图像格式、内存对齐、CPU/BPU/DSP 分工、BPU 核心调度和 BSP 版本。
- BPU 通常不是让你自由编写类似 CUDA 的通用 kernel。更重要的是模型结构、算子支持、量化调优、Runtime 集成、媒体链路和端到端 profiling。

### 0.1 推荐学习顺序

| 顺序 | 主线 | 目标 |
|---|---|---|
| 0 | 通用底座 | Linux、现代 C++、CMake、Python、ONNX、量化、Benchmark |
| 1 | NVIDIA 主路线 | CUDA、TensorRT、Jetson、DeepStream，建立 GPU 推理和 profiling 基准 |
| 2 | 地瓜 BPU 主路线 | OpenExplorer、PTQ/QAT、BPU Runtime、RDK 视觉应用、TROS |
| 3 | 瑞芯微 NPU 路线 | RKNN、RKNPU2、RGA、MPP、GStreamer，完成低成本端侧产品化能力 |
| 4 | 跨平台工程 | 同一模型和摄像头场景在 Orin、RDK、RK 上对齐精度、延迟、功耗和稳定性 |

### 0.2 最终验收

完成后至少应能独立交付：

1. 一个从 ONNX 到 BPU 固定点模型的转换和精度报告。
2. 一个 Python 快速验证和 C++ 板端 Runtime 版本。
3. 一个摄像头或视频输入的实时检测、分割或姿态应用。
4. 一份包含 BPU 延迟、端到端延迟、CPU、内存、温度、帧率和掉帧的报告。
5. 一份能解释模型慢、精度掉、板端崩、视频延迟高的排障记录。

## 1. 目标硬件与路线选择

### 1.1 先学哪一块板

建议把 **RDK X5** 作为学习和公开资料复现平台，把 **RDK S100** 作为机器人/VLA 和更高阶 BPU 平台，把 J6 和 S600 作为汽车量产方向的体系化阅读对象。

| 平台 | 学习价值 | 模型/软件关注点 | 路线位置 |
|---|---|---|---|
| RDK X3 | 低成本理解 BPU、摄像头和基础 demo；历史资料较多 | X 系列旧工具链、基础 DNN Runtime | 兼容性阅读，不做主线 |
| RDK X5 | 10 TOPS、Linux、ROS2/TROS、公开 Model Zoo、Python/C++ 示例丰富 | RDK OS、rdk_x5 分支、hbm_runtime、.bin | 第一块实战板 |
| RDK Ultra | J5/Bayes 体系和机器人视觉验证 | J5 工具链、历史 OpenExplorer 文档 | 理解旧 J5 生态 |
| RDK S100/S100P | 80/128 TOPS，适合机器人、VLM/VLA 和复杂感知 | S 系列 Runtime、.hbm、多加速器协同 | 第二阶段升级 |
| RDK S600 | 多 Nash 核心、最高约 560 TOPS，面向更高算力端侧 | S600 BSP、模型编译、异构调度和量产约束 | 高阶方向 |
| Horizon J6 系列 | 面向智能驾驶和新一代异构平台 | OpenExplorer J6、UCP、Linux/QNX、BPU/DSP/GDC/ISP | 汽车方向高级分支 |

硬件信息优先看 [D-Robotics Developer Community](https://developer.d-robotics.cc/)、[RDK Suite 官方文档](https://d-robotics.github.io/rdk_doc/en/) 和 [RDK X5 产品页](https://developer.d-robotics.cc/en/rdkx5)。不要用旧教程中的 TOPS、系统镜像或文件名推断当前板卡行为。

### 1.2 为什么先 X5，再 S100/J6

- X5 的公开 rdk_model_zoo 和 RDK 文档能快速建立从模型到应用的闭环。
- X5 的模型格式、分支和系统版本相对容易核对，适合先学会工程方法。
- S100 的算力和模型规模更适合具身智能，但工具、BSP、模型格式和板卡资料更依赖具体版本。
- J6、QNX 和量产车载开发涉及供应商授权、BSP、功能安全和客户工程包，不应作为零基础第一站。

### 1.3 公开路线和量产路线的边界

公开 RDK 路线可以让你掌握 BPU 部署方法，但不等于 OEM 量产车载软件栈。量产汽车还会增加：

- 传感器和 ISP 标定、同步、时间戳和多摄像头数据链路；
- QNX/Linux BSP、启动链、设备树、驱动和安全启动；
- AUTOSAR、ISO 26262、ASPICE、OTA 和故障降级；
- BEV、3D 检测、Occupancy、跟踪、规划控制和确定性延迟；
- 供应商 NDA 下的编译器、仿真器、硬件文档和性能模型。

本路线训练可迁移的工程能力，不把公开 demo 包装成量产经验。

## 2. BPU 软件栈全景

| 层次 | 关键字 | 需要掌握的能力 |
|---|---|---|
| 训练/导出 | PyTorch、ONNX、Ultralytics、OpenMMLab | 固定输入、动态维度、算子导出、输出契约 |
| 图检查 | Netron、ONNX checker、shape inference | 发现不支持算子、隐式 cast、动态 shape、布局问题 |
| 量化 | PTQ、QAT、calibration、int8、int16、fp16、mixed precision | 校准集、层误差、敏感层和异常值分析 |
| 编译 | OpenExplorer、HBDK、graph optimize | 把浮点图变成目标 BPU 模型，阅读编译日志和分段 |
| 模型格式 | X 系列 .bin、S 系列 .hbm | 识别模型和硬件、Runtime 的兼容边界 |
| Runtime | hbm_runtime、Hobot DNN、版本对应的 C/C++ API | 加载、输入输出、异步/批处理、BPU 核心选择 |
| 图像前处理 | BGR、RGB、NV12、resize、letterbox、stride、alignment | 保持训练、转换和板端输入契约一致 |
| 媒体 | V4L2、GStreamer、ISP、codec、DMA-BUF | 摄像头、解码、显示、编码和零拷贝链路 |
| 机器人 | ROS2、TROS、topic、message、executor、sensor fusion | 接入机器人感知和控制系统 |
| 性能 | BPU latency、pipeline latency、FPS、CPU、DDR、温度 | 分离模型、排队、拷贝和显示耗时 |
| 系统 | Debian/Ubuntu、systemd、device tree、BSP、交叉编译 | 部署、启动、自启动和版本复现 |

### 2.1 名词对照

| 名词 | 含义 | NVIDIA 类比 |
|---|---|---|
| BPU | Horizon Brain Processing Unit | GPU/NPU 类专用推理加速器 |
| OpenExplorer/OE | Horizon 算法工具链和开发环境 | CUDA + TensorRT + 部分 SDK |
| PTQ | 训练后量化 | TensorRT 校准的同类问题 |
| QAT | 量化感知训练 | fake quantization 训练 |
| Runtime | 板端模型加载和执行接口 | TensorRT Runtime/Execution Context |
| UCP | Horizon Unified Computing Platform | 异构硬件统一编程抽象 |
| TROS | 面向 RDK 的机器人中间件体系 | ROS2 + 厂商硬件组件 |

## 3. 免费课程与官方资料

这条路线的视频以中文为主，因为官方完整 OpenExplorer 课程主要在 Bilibili。英文部分用官方英文手册、README、API 和 sample 补齐。代码和版本以官方文档为准。

| 编号 | 课程/资料 | 语言与性质 | 用法 |
|---|---|---|---|
| H-C1 | [玩转地平线工具链竟如此简单](https://www.bilibili.com/video/BV1Xh411P73Z/) | 中文，D-Robotics 官方，50 集完整课程 | 主课，从第 1 集看到板端和高级示例 |
| H-C2 | [地平线编译及编程实践](https://www.bilibili.com/video/BV1pJ4m1u7F3/) | 中文，公开技术专场 | 补编译器、推理加速和编程实践 |
| H-C3 | [具身智能强化营](https://www.bilibili.com/video/BV15ptXzxEPh/) | 中文，D-Robotics 策划的公开系列 | 机器人、ROS、系统组成和仿真 |
| H-C4 | [智能控制与 Sim-to-Real](https://www.bilibili.com/video/BV1rVt1zEETh/) | 中文，公开完整单元 | 补机器人控制和感知到动作的系统视角 |
| H-C5 | [RDK X5 开发环境极速配置](https://www.bilibili.com/video/BV1yzCJBzEYv/) | 中文，D-Robotics 官方短课 | 第一次上板和环境配置 |
| H-E1 | [OpenExplorer English User Manual](https://doc.oe.horizon.auto/en/index.html) | English，官方 | 当前 J6 工具链和术语主参考 |
| H-E2 | [Toolchain Overview](https://doc.oe.horizon.auto/en/guide/oe_overview/preface/toolchain_overview.html) | English，官方 | 量化、Runtime、仿真和部署 |
| H-E3 | [RDK Model Zoo README](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | English + 简体中文，官方 | 直接跟练模型、Python/C++ sample |
| H-E4 | [D-Robotics RDK Documentation](https://d-robotics.github.io/rdk_doc/en/) | English，官方 | 系统、摄像头、Runtime、ROS 和开发板 |
| H-E5 | [Horizon OE Skills](https://github.com/HorizonRobotics/OE-Skills) | English + 中文说明，公开 GitHub | 高阶工具链操作参考，不能替代人工验证 |

### 3.1 H-C1 的观看顺序

1. 初识：平台概述、图像格式、数据排布、量化。
2. 入门：Docker/PTQ/QAT 环境、板端环境、快速转换和推理。
3. 进阶：配置文件、性能和精度评测、调优、debug。
4. 板端：batch、多模型、优先级、resizer、benchmark 和全流程。
5. 高阶：参考算法、编译和实际应用。

旧课程以 J5 为主要背景。流程和排障思路仍然有价值，但不能直接复制旧版工具名、模型下载地址和安装命令到 J6 或 X5。

## 4. 前置基础

### 4.1 必须会

- Linux shell、SSH、文件权限、进程、线程、信号、共享内存和网络；
- C++17、RAII、智能指针、STL、并发、异常、CMake、gdb；
- Python 虚拟环境、NumPy、OpenCV、PyTorch 和 ONNX 导出；
- Git 分支、tag、submodule、patch 和可复现实验记录；
- 卷积、归一化、softmax、检测头、NMS 和量化基本概念；
- latency、throughput、FPS、p50/p95、warmup、peak memory 和 accuracy parity。

这些基础可复用 [NVIDIA 主路线](EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md) 的 0 到 4 节，不必重复从头学。

### 4.2 上板前最低检查

命令只是示例，字段随 RDK OS 版本变化：

~~~bash
cat /etc/version
rdkos_info
uname -a
uname -m
python3 --version
ls /opt/hobot
~~~

同时记录板卡型号、内存、系统镜像、BSP、BPU 核心、Runtime、工具链、摄像头格式、供电、散热、温度、模型 hash、校准集 hash 和启动命令。

## 5. 阶段路线总表

| 阶段 | 时间 | 关键字 | 主课程 | 验收产物 |
|---|---:|---|---|---|
| H0 | 1 周 | Linux、C++、CMake、ONNX、OpenCV | NVIDIA 基础 + H-C5 | 登录板端并编译 C++ demo |
| H1 | 1 周 | BPU、OE、模型格式、输入布局 | H-C1 前 5 集 + H-E1 | BPU 全链路图和格式说明 |
| H2 | 2 周 | PTQ、calibration、int8、编译 | H-C1 PTQ 单元 + H-E2 | MobileNet/YOLO PTQ 和精度报告 |
| H3 | 1-2 周 | QAT、fake quant、敏感层、混合精度 | H-C1 QAT 单元 + H-E2 | PTQ/QAT 对照实验 |
| H4 | 2 周 | Python/C++ Runtime、输入输出、异步 | H-C1 板端单元 + H-E3 | 图片推理 Python/C++ 双版本 |
| H5 | 2 周 | NV12、图像处理、stride、DMA、缓存 | H-C1 图像单元 + H-E4 | 摄像头到 BPU 的预处理基准 |
| H6 | 2-3 周 | V4L2、GStreamer、ROS2、TROS | H-C3 + H-C4 + H-E4 | 实时视觉 ROS 节点 |
| H7 | 2-3 周 | 多模型、BPU core、优先级、队列 | H-C1 进阶单元 + H-C2 | 多模型调度和性能报告 |
| H8 | 3-4 周 | UCP、J6、QNX、汽车感知 | H-E1/H-E2 + H-C2 | J6/UCP 架构笔记 |

每个阶段都执行“看课 -> 复现 -> 改代码 -> 测量 -> 写报告”，只看视频不算完成。

## 6. H0：通用底座和开发环境

### 目标

- 在 x86 主机和 ARM64 板端区分 host toolchain 与 target runtime；
- 用 CMake 编译 C++，用 gdb 定位内存、线程和 ABI 问题；
- 用 Python 导出固定输入 ONNX；
- 阅读一个 RDK Model Zoo sample 的目录和启动脚本。

### 关键词

aarch64、cross compilation、sysroot、CMake toolchain file、ABI、shared library、rpath、LD_LIBRARY_PATH、SSH、systemd。

### 实验

1. 主机编译一个 C++ OpenCV 图像处理程序。
2. 板端运行同一程序并确认动态库和架构。
3. 用同一张图片生成 ONNX 输出，保存输入、输出和 hash。
4. 给程序加入耗时、帧计数、错误码和版本打印。

### 通过标准

- 能解释为什么 x86 模型转换工具不能直接当板端 Runtime；
- 能从 file、ldd、readelf、dmesg 和日志定位一次库或架构错误；
- 能在干净 shell 中重现一次实验。

## 7. H1：理解 BPU 和 OpenExplorer

### 学习目标

1. 浮点模型不等于板端执行模型。
2. 编译器只会把受支持的图和数据格式高效映射到 BPU。
3. 未支持算子、动态 shape、布局转换和 CPU fallback 会改变端到端性能。
4. 模型、Runtime、驱动和 BSP 必须来自兼容版本组合。

### 必须读的英文资料

- [OpenExplorer Toolchain Overview](https://doc.oe.horizon.auto/en/guide/oe_overview/preface/toolchain_overview.html)
- [OpenExplorer Documentation Overview](https://doc.oe.horizon.auto/en/guide/doc_introduction.html)
- [Horizon Toolchain Portal](https://oe.horizon.auto/)
- [RDK Suite Overview](https://d-robotics.github.io/rdk_doc/en/)

### 实验

画出：

<code>camera/file -> decode/ISP -> color convert -> resize/letterbox -> BPU model -> postprocess -> display/ROS topic</code>

为每个节点标注 CPU、BPU、ISP/PYM、DSP、外部内存和 DMA-BUF 的执行位置。

### 通过标准

- 能解释 .bin 和 .hbm 的平台边界；
- 能从 Model Zoo README 找到模型、输入格式、后处理和运行入口；
- 能指出 pipeline 中至少三个 CPU 瓶颈。

## 8. H2：PTQ、模型转换与编译

### 8.1 推荐模型顺序

- 分类：MobileNetV2 或 ResNet18；
- 检测：YOLOv5s 或 YOLO11n；
- 分割：小型语义分割模型；
- 机器人视觉：检测加姿态，不要一开始从 VLA 开始。

### 8.2 推荐流程

1. PyTorch/ONNX 浮点基线；
2. host CPU 或 GPU 参考输出；
3. OpenExplorer PTQ 配置；
4. 校准数据集；
5. 编译模型；
6. x86 simulation 或 host validation；
7. 板端 Runtime；
8. 逐层或逐输出精度比对。

### 8.3 必须理解的参数

input_type_rt、input_layout_rt、mean/std、RGB/BGR/NV12、输入尺寸、静态 shape、batch、calibration dataset、per-tensor/per-channel、混合精度、敏感层、目标芯片和后处理位置。

不同 OE 版本字段名称会变。不要把旧视频中的 YAML 或 Python 配置直接复制到新版本，先打开当前版本 sample 配置和 schema。

| 练习 | 目的 | 必须记录 |
|---|---|---|
| FP32 vs PTQ INT8 | 认识精度损失 | top-1/mAP、每类召回、失败图 |
| 不同 calibration 集 | 认识数据代表性 | 数据来源、数量、场景分布 |
| RGB vs NV12 | 认识输入契约 | 颜色转换和输出差异 |
| 静态尺寸 vs 多尺寸 | 认识编译约束 | 编译时间、模型大小、延迟、内存 |
| BPU-only vs fallback | 认识图分段 | 编译日志、CPU 时间、端到端 FPS |

### 通过标准

- 能判断精度下降来自校准集、预处理、后处理还是算子支持；
- 能输出浮点基线、量化参数、模型 hash 和板端结果；
- 能解释单看 BPU latency 为什么不等于应用 FPS。

## 9. H3：QAT 和精度调优

PTQ 跑通且精度确实不满足需求后再学 QAT。QAT 不应成为逃避模型结构和输入数据问题的第一选择。

### 关键词

fake quantization、calibration、observer、scale、zero point、saturation、outlier、mixed precision、sensitive layer、accuracy debug。

### 实验路径

1. 选出 PTQ 误差最大的 20 个样本。
2. 做浮点模型、仿真量化模型和板端量化模型三方对照。
3. 用逐层 dump 或 accuracy debug 找到误差扩散点。
4. 先改校准数据和预处理，再尝试敏感层高精度。
5. 最后加入 QAT，并记录训练超参数和 checkpoint。

### 通过标准

- 能画出浮点输出、fake quant 输出和编译模型输出的误差链；
- 能说明 QAT 解决什么问题、代价是什么；
- 保留一个可回退 PTQ 模型和固定评测集。

## 10. H4：Python 和 C++ Runtime

### 10.1 Python 的边界

Python 适合快速检查输入输出、验证预处理后处理、做精度回归和建立第一版 benchmark。正式项目至少要有 C++ 版本，明确线程、队列、缓冲区生命周期和错误恢复。

### 10.2 C++ Runtime 生命周期

1. 读取模型并校验文件大小和 hash。
2. 初始化 Runtime 和目标设备。
3. 查询输入输出 tensor 属性。
4. 准备正确布局、dtype、stride 和内存。
5. 提交推理并处理同步或异步返回。
6. 读取输出并执行后处理。
7. 释放输出、模型和上下文。
8. 在异常和退出路径释放资源。

不同 RDK 代际 API 名称不同，优先从 [RDK Model Zoo C++ samples](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples) 和当前板端头文件查接口。

### 10.3 最小 C++ 项目

~~~text
horizon-bpu-demo/
  CMakeLists.txt
  include/
  src/
    model_runner.cpp
    preprocess.cpp
    postprocess.cpp
    main.cpp
  configs/
  scripts/
  tests/
  reports/
~~~

最低功能：单图、摄像头/视频、模型和输入参数、warmup、平均延迟、p50/p95、FPS、错误码、Runtime 版本和结果图。

### 通过标准

- Python 和 C++ 输出在容许误差内一致；
- 运行 30 分钟不崩溃、不持续泄漏；
- 能区分模型、前处理、后处理和显示时间；
- 错误模型或错误输入能给出明确错误。

## 11. H5：图像格式、内存和硬件预处理

这是视觉端侧最容易被低估的阶段。许多“BPU 很快但系统很慢”的问题来自：

- 摄像头输出 NV12，模型却要求 RGB；
- resize 在 CPU 上逐帧执行；
- stride 和实际 width 不一致；
- 多次 BGR/NV12 拷贝；
- DMA-BUF 或 cache 没有正确同步；
- 后处理把大 tensor 拷回 CPU；
- 显示或编码阻塞推理线程。

### 关键词

NV12、YUV420、BGR/RGB、letterbox、stride、alignment、DMA-BUF、ION/CMA、cache sync、zero-copy、V4L2、ISP、PYM。

### 资料

- [RDK X5 Python detection sample](https://github.com/D-Robotics/rdk_x_doc/blob/main/docs/03_Basic_Application/03_pydev_demo_sample/RDK_X5/02_detection_sample.md)
- [RDK BPU C++ sample](https://d-robotics.github.io/rdk_doc/Basic_Application/cdev_demo_sample/bpu/)
- [RDK X5 Model Zoo Guide](https://d-robotics.github.io/rdk_doc/en/Algorithm_Application/model_zoo/rdk_x5_guide/)
- [RDK Suite documentation](https://d-robotics.github.io/rdk_doc/en/)

### 实验

分别测量：

1. 文件读取 + CPU resize + BPU；
2. 摄像头 + CPU color convert + BPU；
3. 摄像头 + 硬件预处理 + BPU；
4. 单 buffer、双 buffer、环形 buffer；
5. 结果完全回 CPU 和结果尽量留在设备侧的差异。

报告必须把模型 latency 和端到端 frame latency 分开。

### 通过标准

- 能画出每次内存拷贝和格式转换的位置；
- 能解释 buffer 的 owner、生命周期和同步点；
- 能在不改变模型精度的情况下减少至少一个不必要拷贝；
- 能用固定输入和固定帧率复现延迟变化。

## 12. H6：摄像头、视频和机器人应用

### 12.1 视觉应用路线

先做单路，再做多路：

1. 图片检测；
2. 视频文件检测；
3. USB 摄像头；
4. MIPI 摄像头；
5. 检测 + tracker；
6. 检测 + 编码/推流；
7. 多路摄像头或多模型。

### 12.2 ROS2/TROS 路线

掌握以下概念：

- node、topic、service、action；
- message timestamp、frame_id、QoS；
- callback group、executor 和线程；
- image_transport、camera info、同步；
- 感知节点与控制节点之间的延迟预算；
- 感知失败时的降级和安全状态。

用 [具身智能强化营](https://www.bilibili.com/video/BV15ptXzxEPh/) 建立机器人系统全貌，再回到 RDK 官方 sample 实现硬件侧。

### 12.3 第一个机器人项目

**项目：RDK X5 实时目标检测驱动行为反馈**

- 输入：USB/MIPI 摄像头；
- BPU：YOLO 小模型；
- 输出：ROS2/TROS topic；
- 控制：检测到目标后驱动 GPIO、舵机或仿真动作；
- 约束：感知到动作的 p95 延迟、丢帧率、断相机恢复；
- 证据：视频、源码、launch/config、性能报告和故障记录。

不要把控制闭环和大模型对话混在第一个项目里，先证明视觉链路的确定性。

## 13. H7：多模型、调度和性能优化

### 13.1 先测隔离，再测并发

至少准备三个模型：

- 高频低延迟检测；
- 中频分类或姿态；
- 低频 OCR 或语义模型。

每个模型先单独测：

- warmup 后平均 latency；
- p50/p95/p99；
- BPU 使用率；
- CPU 使用率；
- DDR/内存；
- 温度和降频；
- 结果精度。

然后再测：

- 同一 BPU core 串行；
- 多 core 分配；
- 不同优先级；
- 固定队列和有界队列；
- 丢帧策略；
- 模型 batch；
- 传感器输入和推理线程解耦。

### 13.2 课程对应

H-C1 中的 Batch 模型推理、多模型批量推理、多模型优先级调度、benchmark 评测及后处理，是本阶段的主视频单元。每看完一个单元，都要在当前 RDK 版本上重做，并记录与 J5 课程结果不同的地方。

### 13.3 通过标准

- 有一张模型隔离/并发矩阵；
- 能解释排队延迟和 BPU 执行延迟；
- 有背压、超时、丢帧和降级策略；
- 连续运行至少 1 小时并保存资源曲线；
- 模型加载失败、摄像头断开或内存不足时能恢复或明确退出。

## 14. H8：UCP、J6 和汽车方向

### 14.1 UCP 学什么

[UCP Overview](https://doc.oe.horizon.auto/3.8.1/guide/ucp/ucp_overview.html) 介绍了面向异构硬件的统一接口。学习重点不是记 API，而是理解：

- BPU、DSP、GDC、STITCH、JPU、VPU、PYRAMID、ISP 等硬件能力如何抽象；
- Linux/QNX 和交叉编译边界；
- 哪些工作适合模型编译器，哪些工作适合异构 Runtime；
- 资源、队列、内存和同步如何跨硬件单元传递；
- 同一应用在主机仿真、开发板和量产 BSP 中如何保持契约。

### 14.2 汽车视觉关键字

ADAS、DMS/OMS、surround view、multi-camera、BEV、3D detection、occupancy、sensor fusion、ISP、camera synchronization、QNX、functional safety、ISO 26262、ASPICE、ASIL、OTA、watchdog。

### 14.3 汽车方向项目

公开板卡上不追求复刻量产车，而做一个可验证的缩小版：

- 两路视频输入；
- 一个检测模型和一个车道/分割模型；
- 固定时间戳和同步策略；
- BPU/CPU/图像硬件分工；
- 结果融合和故障降级；
- 记录每一路 p95、端到端延迟、丢帧和温度；
- 模拟一路摄像头断开并验证系统行为。

### 14.4 通过标准

- 能画出从传感器到决策的时序图；
- 能写出模型、媒体、调度和安全边界；
- 能说明公开 RDK 经验与量产 J6/QNX 经验的差距；
- 能读懂一份新 BSP 的版本矩阵，而不是只会运行 demo。

## 15. 项目阶梯

### P0：单图分类

- 模型：MobileNetV2；
- 目标：理解模型格式、输入布局、Runtime 生命周期；
- 产物：Python/C++、精度对比、单次和批量 latency。

### P1：YOLO 检测

- 模型：YOLOv5s 或 YOLO11n；
- 目标：PTQ、NV12、后处理、NMS；
- 产物：mAP/召回、失败样本、实时 FPS。

### P2：摄像头实时检测

- 目标：V4L2/摄像头、双缓冲、预处理和显示；
- 产物：端到端 latency 分解、断流恢复、30 分钟稳定性。

### P3：检测 + 跟踪 + 推流

- 目标：BPU 推理、CPU 后处理、硬件编码和网络输出；
- 产物：GStreamer/ROS pipeline、带宽、延迟和丢帧报告。

### P4：多模型机器人感知

- 目标：检测、姿态、深度或 OCR 多模型并发；
- 产物：优先级、背压、超时和降级策略。

### P5：S100/VLA 感知节点

- 目标：把视觉模型转换、Runtime 和 ROS2/TROS 连接起来；
- 产物：模型输入输出契约、动作触发、实时性和失败安全报告。

### P6：跨平台同模型对比

- 目标：同一 ONNX 模型在 Orin、RDK 和 Rockchip 上保持数据集、预处理和指标一致；
- 产物：平台矩阵和选择建议，不用 TOPS 直接替代实测。

## 16. 性能与精度验收模板

每次实验至少保存以下字段：

~~~text
board_model:
soc:
ram:
os_version:
bsp_version:
toolchain_version:
runtime_version:
model_name:
model_sha256:
input_shape:
input_format:
precision:
calibration_dataset:
preprocess_ms:
bpu_ms:
postprocess_ms:
queue_ms:
end_to_end_ms:
fps:
p50_ms:
p95_ms:
cpu_percent:
bpu_utilization:
memory_mb:
temperature_c:
power_mode:
accuracy_metric:
failure_count:
notes:
~~~

### 必须分开的指标

- **BPU latency**：Runtime 报告的模型执行时间；
- **pipeline latency**：从输入帧到输出结果；
- **queue latency**：排队和调度等待；
- **goodput**：满足时限和精度约束的有效帧率；
- **accuracy parity**：浮点参考与板端结果的差异；
- **soak stability**：长时间运行中的崩溃、泄漏、降频和丢帧。

## 17. 常见故障排查顺序

### 模型无法编译

1. 检查目标芯片和 OE 版本；
2. 检查 ONNX opset、动态 shape 和输入输出数量；
3. 检查不支持算子和 graph 分段；
4. 将复杂后处理移到 CPU 做最小验证；
5. 用官方 sample 的相近模型做对照；
6. 固定模型、配置和日志后再提交论坛问题。

### 板端推理失败

1. 检查模型格式是否属于当前芯片；
2. 检查 Runtime、driver、BSP 版本组合；
3. 检查 uname -m 和动态库架构；
4. 检查 tensor dtype、layout、stride 和 buffer 大小；
5. 检查 BPU 设备权限和系统日志；
6. 用官方 Model Zoo 模型验证硬件，再换自己的模型。

### 精度突然很差

1. 先在板端保存实际输入；
2. 与 host 使用完全相同的颜色转换、resize 和 normalization；
3. 检查 letterbox、坐标反变换和 NMS；
4. 检查 calibration 集是否覆盖真实场景；
5. 检查模型输出顺序和量化 scale；
6. 最后再考虑 QAT。

### FPS 不高

1. 先把显示、编码、网络发送去掉；
2. 分开测预处理、BPU、后处理和排队；
3. 检查 CPU 是否逐帧转换大图；
4. 检查是否发生多次内存拷贝；
5. 检查 BPU core、优先级和 batch；
6. 固定功耗、温度和输入帧率后比较。

## 18. 学完后的能力与岗位

### 初级可交付能力

- 能在 RDK X5 上使用官方模型和自己的 ONNX 模型；
- 能完成 PTQ、转换、板端推理和基本精度验证；
- 能读懂 Python/C++ BPU sample；
- 能写一个 C++ 图片或视频推理程序。

### 中级端侧部署能力

- 能处理不支持算子、量化精度、输入格式和模型分段；
- 能把摄像头、预处理、BPU、后处理、显示/推流串成稳定 pipeline；
- 能用 ROS2/TROS 发布感知结果；
- 能测量并优化端到端延迟、内存和多模型调度；
- 能维护模型、BSP、Runtime 和工具链版本矩阵。

### 高级系统能力

- 能设计 BPU/CPU/DSP/媒体硬件的分工；
- 能比较 X5、S100、S600、J6 的软件边界；
- 能理解 UCP 和 Linux/QNX 交叉编译路径；
- 能把同一个视觉系统迁移到 NVIDIA、Horizon 和 Rockchip；
- 能为汽车或机器人项目写可审计的性能、精度和故障验收报告。

中文岗位关键词：BPU部署工程师、地平线算法工程师、边缘AI部署工程师、端侧推理优化工程师、机器人视觉工程师、RDK应用开发工程师、车载AI软件工程师、模型量化工程师、异构计算工程师。

英文岗位关键词：Edge AI Engineer、BPU Inference Engineer、Model Deployment Engineer、Embedded Vision Engineer、Robotics Perception Engineer、AI Compiler Engineer、Heterogeneous Computing Engineer、Automotive AI Software Engineer。

这条公开路线不能单独证明 OEM 量产经验、功能安全签字资格或 J6 商业项目权限。求职时要用可复测项目和原始数据证明能力。

## 19. 官方资源索引

### D-Robotics / Horizon

- [D-Robotics Developer Community](https://developer.d-robotics.cc/)
- [D-Robotics GitHub Organization](https://github.com/D-Robotics)
- [RDK Suite Documentation](https://d-robotics.github.io/rdk_doc/en/)
- [RDK X5 Product Page](https://developer.d-robotics.cc/en/rdkx5)
- [RDK Model Zoo, X5 branch](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5)
- [RDK Model Zoo Guide](https://d-robotics.github.io/rdk_doc/en/Algorithm_Application/model_zoo/rdk_x5_guide/)
- [RDK X5 Python detection sample](https://github.com/D-Robotics/rdk_x_doc/blob/main/docs/03_Basic_Application/03_pydev_demo_sample/RDK_X5/02_detection_sample.md)
- [D-Robotics RDK device skills](https://github.com/D-Robotics/rdk-device-skills)
- [D-Robotics device knowledge](https://github.com/D-Robotics/device-knowledge)

### OpenExplorer

- [OpenExplorer Portal](https://oe.horizon.auto/)
- [OpenExplorer Chinese Manual](https://doc.oe.horizon.auto/)
- [OpenExplorer English Manual](https://doc.oe.horizon.auto/en/index.html)
- [Toolchain Overview, English](https://doc.oe.horizon.auto/en/guide/oe_overview/preface/toolchain_overview.html)
- [Environment Deployment, English](https://doc.oe.horizon.auto/en/guide/env_install.html)
- [UCP Overview](https://doc.oe.horizon.auto/3.8.1/guide/ucp/ucp_overview.html)
- [Horizon OE Skills](https://github.com/HorizonRobotics/OE-Skills)

### 视频

- [D-Robotics OpenExplorer 50 集课程](https://www.bilibili.com/video/BV1Xh411P73Z/)
- [地平线编译及编程实践](https://www.bilibili.com/video/BV1pJ4m1u7F3/)
- [具身智能强化营：机器人学基础](https://www.bilibili.com/video/BV15ptXzxEPh/)
- [具身智能强化营：智能控制与 Sim-to-Real](https://www.bilibili.com/video/BV1rVt1zEETh/)
- [RDK X5 开发环境极速配置](https://www.bilibili.com/video/BV1yzCJBzEYv/)

## 20. 执行清单

### H0-H2

- [ ] 完成 NVIDIA 主路线的 C++/Linux/ONNX 基础
- [ ] 看完 H-C1 的前 13 集并整理术语表
- [ ] 在 RDK X5 或兼容环境跑通官方 Model Zoo
- [ ] 完成 MobileNet PTQ
- [ ] 保存模型 hash、校准集和精度报告

### H3-H5

- [ ] 完成一次 PTQ/QAT 对照
- [ ] 写出 Python/C++ 双 Runtime
- [ ] 完成 NV12/RGB/BGR 对照实验
- [ ] 测量预处理、BPU、后处理和端到端延迟
- [ ] 解决至少一个不必要拷贝或 CPU 瓶颈

### H6-H8

- [ ] 完成摄像头实时检测
- [ ] 完成 ROS2/TROS 感知节点
- [ ] 完成多模型优先级和背压
- [ ] 在 S100 或公开资料上理解 .hbm 和更高阶 Runtime
- [ ] 阅读 UCP/J6 文档并写一份汽车视觉架构设计

### 最终作品

- [ ] 一个公开 Git 仓库
- [ ] 一键运行脚本和版本清单
- [ ] C++ 端侧应用
- [ ] 精度、延迟、功耗、温度和稳定性报告
- [ ] 至少一个故障注入和恢复案例

---

**执行原则：先用官方 Model Zoo 跑通，再换自己的模型；先把 PTQ 和输入格式搞对，再学 QAT；先做单路实时视觉，再做多模型和机器人闭环；先建立 X5 的可复现实验，再进入 S100/J6/UCP。**
