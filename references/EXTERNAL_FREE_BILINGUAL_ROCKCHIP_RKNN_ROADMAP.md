# 外部免费双语瑞芯微 RKNN / RKNPU2 / RKNN3 学习路线（双并行路线）

> 路线定位：这是面向汽车、机器人和视觉端侧的 Rockchip NPU 专线。
> 推荐顺序：在 NVIDIA 主路线和地瓜 BPU 路线之后学习，先以 RK3588 Linux 本体 NPU 为主平台，再扩展到 RK3576、RK3568/RV1126B 和具体产品 BSP。
> 分支关系：本文统一包含 RKNN2/RKNPU2 本体 NPU 主线和 RKNN3/RK1828 协处理器并行线；RKNN3 内容从第 21 节开始。
> 资料范围：真实存在的公开课程、官方 GitHub 项目、官方或板卡厂商公开文档和公开示例；不依赖付费课程。
> 核验日期：2026-08-21。RKNN 工具、NPU driver、板卡 BSP 和系统镜像必须成套匹配，文档中的版本和命令不能跨版本盲抄。

## 0. 先给结论

瑞芯微端侧 AI 的完整学习对象不是只有 RKNN：

<code>PyTorch/ONNX -> RKNN-Toolkit2 -> quantization/build -> .rknn -> RKNN Runtime/RKNPU2 -> RGA preprocess -> MPP/V4L2 decode -> GStreamer/应用 -> benchmark</code>

你要形成的能力是：

- 把模型从 ONNX 转成目标芯片可运行的 RKNN；
- 用模拟器、板端 Python Lite2 和 C/C++ Runtime 做三方验证；
- 理解 INT8、混合精度、校准数据和不支持算子；
- 用 RGA 做 resize、色彩转换和布局处理；
- 用 MPP/V4L2/GStreamer 完成摄像头、解码、编码和推流；
- 把 NPU latency、预处理、后处理、排队和媒体延迟分开测量；
- 面对不同板卡厂商的 kernel、BSP、driver 和库版本差异。

### 0.1 本文与 RKNN3 并行路线的边界

| 路线 | 硬件 | 工具链 | 运行时 | 重点 |
|---|---|---|---|---|
| 本文 RKNN2 | RK3588 本体 NPU 等 | RKNN-Toolkit2 | Lite2、RKNPU2、librknnrt.so | 视觉模型、INT8、RGA/MPP、多路视频 |
| 并行 RKNN3 | RK3588 + RK1828 | RKNN3-Toolkit | RKNN3 Runtime、librknn3_api.so、librknn3_api_rkcp.so | 协处理器通信、LLM/VLM、session、多模型资产 |

两条路线可以共享 ONNX、C++、RGA、MPP、V4L2、GStreamer 和 benchmark 方法，但不能混用 Toolkit、Runtime、driver 假设、模型转换脚本和性能数据。先用本文 RKNN2 主线完成本体 NPU 视觉基线，再进入本文第 21 节的 RKNN3 / RK1828 并行线。

### 0.2 瑞芯微与 NVIDIA、地瓜的区别

| 维度 | NVIDIA | 地瓜/Horizon | Rockchip |
|---|---|---|---|
| 主要加速器 | CUDA GPU、Tensor Core、DLA | BPU 和异构硬件 | NPU、CPU、GPU、VPU、RGA |
| 模型工具链 | ONNX/TensorRT | OpenExplorer、PTQ/QAT、Runtime | RKNN-Toolkit2、RKNPU2 |
| 模型文件 | TensorRT engine | X 系列 .bin、S 系列 .hbm | .rknn |
| 视觉媒体 | CUDA/GStreamer/DeepStream | RDK media/TROS | MPP、RGA、V4L2、GStreamer |
| 优化重点 | kernel、tactic、显存和并发 | 编译覆盖率、量化、BPU 调度和格式 | 算子支持、量化、内存拷贝、NPU/媒体协同 |
| 工程风险 | 版本和 ARM64 容器 | 版本、BSP、供应商工具链 | 板卡 BSP 碎片化、驱动和库版本不一致 |

## 1. 目标平台选择

### 1.1 先学 RK3588

RK3588 适合学习的原因：

- NPU、CPU、GPU、视频编解码、RGA 和丰富 I/O 都在同一 SoC；
- 官方 RKNN Model Zoo 和 RKNPU2 sample 覆盖面较大；
- 开发板和 Linux 镜像选择多，容易做多媒体和视觉实验；
- 能同时练习模型转换和端到端系统优化。

| 平台 | 适合场景 | 本路线位置 |
|---|---|---|
| RK3588/RK3588S | 机器人视觉、边缘盒子、多路视频、VLM/小型多模态实验 | 第一主平台 |
| RK3576 | 新一代端侧视觉和中等算力产品 | RK3588 后迁移 |
| RK3566/RK3568 | 成本敏感设备、工业视觉、基础检测 | 理解兼容性和资源约束 |
| RK3562 | 低功耗产品和轻量视觉 | 作为裁剪路线 |
| RV1103/RV1106/RV1126B | IPC、摄像头和专用视觉设备 | 做产品方向分支 |
| RK2118 或其他新平台 | 新 SDK 适配 | 以当前 SDK 支持矩阵为准 |

官方 [RKNN-Toolkit2 README](https://github.com/airockchip/rknn-toolkit2) 当前列出 RK3588、RK3576、RK3566/RK3568、RK3562、RV1103/RV1106、RV1126B 等平台。具体支持要同时核对工具包、runtime 和 driver。

### 1.2 选哪一块开发板

优先选择能提供以下资料的 RK3588 板卡：

- 可刷写的 Linux 镜像和完整 BSP；
- 与系统匹配的 RKNPU driver、librknnrt.so、librga.so 和 MPP；
- 官方或板卡厂商公开的交叉编译工具链；
- CSI 摄像头、USB、HDMI、网络和硬件编解码接口；
- 能访问 kernel log、debugfs、设备节点和温度信息。

不要只按 NPU TOPS 选板。对视觉端侧来说，摄像头驱动、DDR、RGA、VPU、散热和 SDK 可获得性往往比峰值算力更决定交付速度。

## 2. RKNN 软件栈全景

| 层次 | 关键字 | 需要掌握的能力 |
|---|---|---|
| 训练/导出 | PyTorch、TensorFlow、ONNX、Ultralytics | 固定 shape、opset、输出节点和预处理契约 |
| PC 工具 | RKNN-Toolkit2 | load、config、build、export、模拟推理、精度和性能分析 |
| 量化 | INT8、INT16、FP16、混合精度、dataset | 校准集、量化误差、敏感层和异常值 |
| 模型文件 | .rknn | 目标平台、工具版本、输入输出属性和模型 hash |
| Python 板端 | RKNN-Toolkit-Lite2 | 快速部署和验证；只做板端推理，不做转换 |
| C/C++ 板端 | RKNN Runtime、RKNPU2、librknnrt.so | 模型生命周期、tensor、异步、零拷贝和多核 |
| NPU driver | rknpu kernel driver | 用户态 runtime 与 NPU 硬件交互 |
| 图像处理 | RGA、librga | resize、crop、rotate、色彩转换、blend |
| 编解码 | MPP、VPU、V4L2 | H.264/H.265/AV视频解码编码和 buffer 传递 |
| 媒体框架 | GStreamer、FFmpeg、DRM/KMS | 摄像头、解码、推理、显示、推流 |
| 应用 | OpenCV、C++、ROS2、Docker/systemd | 端侧服务、机器人节点和自启动 |
| 性能 | rknn perf、perf、trace、top、温度、NPU load | 分离 NPU、CPU、DDR、拷贝和排队 |

### 2.1 三个接口的边界

| 接口 | 运行位置 | 用途 |
|---|---|---|
| RKNN-Toolkit2 | PC 或工具链环境 | 模型转换、量化、模拟推理、评估 |
| RKNN-Toolkit-Lite2 | 目标板 | Python 快速部署和推理 |
| RKNN Runtime/RKNPU2 | 目标板 | C/C++ 产品应用和性能控制 |

官方 README 明确指出，RKNN-Toolkit2 不兼容旧的 RKNN-Toolkit。旧平台和新平台的仓库、模型格式和 API 不要混用。

## 3. 免费课程和资料总表

### 3.1 中文完整视频

| 编号 | 课程/视频 | 性质 | 用法 |
|---|---|---|---|
| R-C1 | [RKNPU2 从入门到实践，基于 RK3588/RK3568](https://www.bilibili.com/video/BV1Kj411D78q/) | 中文，迅为公开系列，含 Toolkit2、量化、Lite2、C API、零拷贝和项目 | 主课程，完整看完章节 |
| R-C2 | [RK3588 RKNPU2 开发和使用](https://www.bilibili.com/video/BV1eB4y1Z7gV/) | 中文，5 部分完整短系列 | 用于快速建立 RKNPU2 全貌 |
| R-C3 | [RKNN 模型转换与评估系列](https://www.bilibili.com/video/BV15w4m1Y7QL/) | 中文，5 部分，环境、转换、Lite2、性能和精度 | 专门补模型转换 |
| R-C4 | [瑞芯微 RK 系列端侧 AI 开发](https://www.bilibili.com/video/BV1QLR7BqE3d/) | 中文，公开长系列，覆盖架构、RKNN、MobileNet、YOLO、分割 | 作为完整项目参考，不要求一次看完全部嵌入式章节 |
| R-C5 | [RK 公开课：新一代 NPU 的介绍及使用](https://www.bilibili.com/video/BV1K44y1g7o5/) | 中文，瑞芯微公开课 | 补 NPU 架构和官方术语 |
| R-C6 | [RK3588 上 RGA 的简单使用](https://www.bilibili.com/video/BV1uCcfekEK9/) | 中文，RGA 实战单元 | 放在媒体和预处理阶段 |

R-C1 的公开视频可以免费观看；其配套资料、开发板支持或网盘内容可能有厂商限制。主路线只依赖官方公开仓库和文档，不把购买开发板或加入售后群作为前置条件。

### 3.2 English 官方资料

| 编号 | 资料 | 重点 |
|---|---|---|
| R-E1 | [RKNN-Toolkit2 README](https://github.com/airockchip/rknn-toolkit2) | 软件栈、支持平台、版本和下载入口 |
| R-E2 | [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | 官方模型、C/Python 示例和完整转换链 |
| R-E3 | [RKNPU2 source and examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2) | C/C++ Runtime、API、zero-copy 和 demo |
| R-E4 | [RKNPU2 API header](https://github.com/airockchip/rknn_model_zoo/blob/main/3rdparty/rknpu2/include/rknn_api.h) | 查询、输入输出、内存和多核接口 |
| R-E5 | [RGA Developer Guide](https://github.com/airockchip/librga/blob/main/docs/Rockchip_Developer_Guide_RGA_EN.md) | 图像格式、硬件限制和 IM2D API |
| R-E6 | [Rockchip MPP Developer Guide](https://github.com/rockchip-linux/mpp/blob/develop/doc/Rockchip_Developer_Guide_MPP_EN.md) | 硬件编解码、MPI、buffer 和中间件 |
| R-E7 | [Firefly RKNN English Guide](https://github.com/Firefly-docs/ai-docs/blob/master/en/Dev%20Toolchain/RKNN/RKNN2/rknn2.md) | 公开板卡环境、RKNPU2 和 Lite2 说明 |
| R-E8 | [OpenMMLab Rockchip backend guide](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/rockchip.md) | 从部署框架角度理解 RKNN backend |
| R-E9 | [Rockchip Linux GitHub organization](https://github.com/rockchip-linux) | kernel、MPP、rkbin 等系统源码 |

英文完整视频目前不如中文公开系列稳定，因此用 R-C1/R-C2 做跟练，用 R-E1 到 R-E9 校正术语、API 和版本。这个组合比只看翻译视频更适合后续读源码和处理 issue。

## 4. 前置基础

### 4.1 从 NVIDIA 和 Horizon 路线复用

不需要重新学习：

- Linux、C++、CMake、Git、gdb、perf；
- ONNX 导出、模型输入输出契约和基础量化；
- latency、throughput、p50/p95、warmup、精度回归；
- 摄像头、ROS2、GStreamer 的基本概念。

需要新增的 Rockchip 关键词：

RKNN-Toolkit2、RKNPU2、RKNN Runtime、RKNN Lite2、librknnrt、rknpu driver、RGA、librga、MPP、V4L2、DMA-BUF、NV12、DRM/KMS、BSP、Buildroot、Yocto、rkbin。

### 4.2 板端版本清单

在每块板上保存：

~~~bash
cat /etc/os-release
uname -a
uname -m
cat /proc/device-tree/compatible
ldconfig -p | grep -E 'rknn|rga|rockchip|mpp'
ls -l /dev | grep -E 'rknpu|rga|video'
~~~

另外记录 SoC、板卡、kernel、rootfs、BSP、交叉编译器、RKNPU driver、librknnrt、RKNN Toolkit2、RGA、MPP、NPU 核心、CPU governor、温度、模型 hash、输入格式和摄像头参数。

### 4.3 版本配对规则

1. 先确定板卡 BSP 和 kernel。
2. 从板卡 BSP 中确认 NPU driver 和 runtime。
3. 再选择与之兼容的 RKNN-Toolkit2。
4. 用同一版本 Model Zoo sample 验证。
5. 只有官方 sample 稳定后才换自己的模型。

不要把网上不同板卡的 librknnrt.so、rknpu driver 和 .rknn 文件随意拼接。

## 5. 阶段路线总表

| 阶段 | 时间 | 关键字 | 主视频 | 验收产物 |
|---|---:|---|---|---|
| R0 | 1 周 | Linux、C++、ONNX、BSP | R-C2 + 既有基础 | 能登录板端并编译 sample |
| R1 | 1 周 | NPU、RKNN 软件栈、版本 | R-C5 + R-E1 | 版本矩阵和架构图 |
| R2 | 2 周 | load、config、build、export、.rknn | R-C1 + R-C3 | MobileNet/YOLO 模型 |
| R3 | 2 周 | INT8、calibration、精度分析 | R-C1 + R-E2 | 浮点/量化精度报告 |
| R4 | 2 周 | Lite2、RKNPU2 C API、tensor | R-C1 + R-C2 | Python/C++ 双 Runtime |
| R5 | 2 周 | zero-copy、RGA、NV12、stride | R-C1 + R-C6 + R-E5 | 硬件预处理基准 |
| R6 | 2-3 周 | MPP、V4L2、GStreamer、DMA-BUF | R-C6 + R-E6 | 摄像头到 NPU pipeline |
| R7 | 2-3 周 | 多路视频、多模型、NPU core、profiling | R-C4 + R-E4 | 并发和稳定性报告 |
| R8 | 3 周 | 自定义算子、模型改图、BSP 适配 | R-C4 + R-E3/R-E8 | 非原生模型适配记录 |
| R9 | 2-3 周 | 机器人、车载视觉、跨平台 | R-C4 + ROS2 基础 | 可迁移的端侧视觉系统 |

每阶段必须有代码、原始数据、版本清单和复盘记录。

完成 R0-R4 的 RKNN2 基础，或已经具备等价的 ONNX、量化、C++ Runtime 和 benchmark 能力后，即可进入本文第 21 节的 RKNN3 / RK1828 + RK3588 并行路线。RKNN2 的 R5-R9 媒体、并发、算子和车载项目可以与 RKNN3 并行推进；共享方法可以复用，但 Runtime、模型资产和性能结论必须重新建立。

## 6. R0-R1：平台和 NPU 基础

### 学习目标

- 分清 PC 转换环境、板端 Lite2 和 C++ Runtime；
- 理解 NPU driver、runtime、模型文件和 BSP 的关系；
- 能从官方 README 找到支持平台和对应示例；
- 能读懂一个 RKNPU2 sample 的编译和运行路径。

### 必须画出的架构

~~~text
training/export
      |
      v
RKNN-Toolkit2 on host
      |
      v
model.rknn
      |
      +--> RKNN Lite2 Python on board
      |
      +--> RKNN Runtime/RKNPU2 C++ on board
                    |
                    v
              rknpu kernel driver -> NPU
~~~

### 实验

1. 用官方 Model Zoo 运行一个分类模型。
2. 用同一模型分别跑 Python 和 C++。
3. 打印 API、driver 和 SoC 版本。
4. 故意替换一个不匹配的模型或库，记录错误表现，再恢复。

### 通过标准

- 能解释为什么 .rknn 不是通用 ONNX；
- 能从 file、ldd、readelf、dmesg 和版本打印中定位一次 ABI/库错误；
- 能在 README 中找到当前平台的 branch 和运行脚本。

## 7. R2：RKNN 模型转换

### 7.1 标准工作流

1. 训练框架导出 ONNX。
2. 用 Netron 检查输入、输出、shape、opset 和算子。
3. 创建 RKNN 对象并配置 mean/std、目标平台和优化选项。
4. 加载 ONNX。
5. 用 calibration dataset 做 build。
6. 导出 .rknn。
7. 先在模拟器或 host API 验证。
8. 再复制到板端运行。

典型 API 名称包括 load_onnx、config、build、export_rknn、init_runtime、inference；实际签名以当前 Toolkit2 API 文档和 example 为准。

### 7.2 第一批模型

| 模型 | 为什么先学 |
|---|---|
| MobileNetV2 | 输入输出简单，适合验证转换和量化 |
| ResNet18 | 适合理解分类输出和性能评估 |
| YOLOv5s | 适合理解检测头、后处理和实时性 |
| YOLOv8/YOLO11n | 适合练新模型导出、算子支持和输出解析 |
| LPRNet 或小型 OCR | 适合视觉产品和多输入输出 |

### 7.3 必须记录的转换信息

- ONNX opset 和导出 commit；
- target_platform；
- 输入 shape、layout、dtype 和颜色格式；
- mean/std 和量化方式；
- calibration dataset 文件列表和 hash；
- build 日志、warning、未支持算子；
- .rknn 文件 hash 和大小；
- host、Lite2、Runtime 三方输出。

### 通过标准

- 能把一个自己的 ONNX 模型转成目标平台 .rknn；
- 能解释转换失败是模型图问题还是环境版本问题；
- 能在同一输入上对比 PyTorch、ONNX Runtime、RKNN host 和板端输出。

## 8. R3：量化、精度和算子支持

### 8.1 PTQ 精度调试顺序

1. 先确认 FP32 PyTorch 和 ONNX 输出一致。
2. 再确认 RKNN host 浮点输出一致。
3. 检查输入颜色、resize、letterbox、mean/std。
4. 检查 calibration dataset 的代表性。
5. 分析每层或每个输出的误差。
6. 尝试混合精度或保留敏感层。
7. 必要时改写模型图、拆分后处理或走 CPU。
8. 最后再考虑 QAT。

### 8.2 常见精度问题

- BGR/RGB 顺序反了；
- 训练时使用 letterbox，板端使用直接 resize；
- 训练和转换使用的 mean/std 不同；
- INT8 校准集过小或不覆盖夜间、逆光、小目标；
- NCHW/NHWC 或 tensor stride 错误；
- YOLO 输出解码、坐标缩放和 NMS 不一致；
- 量化后某些算子发生饱和或精度回退。

### 8.3 算子支持的工程处理

当模型包含不支持算子时，按以下顺序处理：

1. 查当前 Toolkit2 的 operator support 和 issue；
2. 用等价算子替换；
3. 在导出阶段折叠常量、删除冗余节点；
4. 把后处理移到 CPU；
5. 使用官方支持的 custom op 或 GPU fallback 能力；
6. 评估是否换一个更适合端侧的模型。

不要为了让模型能跑而忽略实际 CPU fallback 时间。

### 通过标准

- 有 FP32、FP16/混合精度、INT8 的精度和性能矩阵；
- 有至少 20 个失败样本的可视化；
- 能说明每个精度变化来自模型、输入还是 Runtime；
- 有一个可回退的非量化或高精度版本。

## 9. R4：Lite2 和 RKNPU2 C++ Runtime

### 9.1 Python Lite2

Lite2 适合快速验证板端模型、调试输入输出和后处理、做精度回归以及写早期 demo。Lite2 在板端只负责推理，不能替代 PC 端 Toolkit2 的模型转换。

### 9.2 C++ API 生命周期

应熟悉以下 API 家族，具体参数以当前 rknn_api.h 为准：

- rknn_init、rknn_destroy；
- rknn_query；
- rknn_inputs_set、rknn_run；
- rknn_outputs_get、rknn_outputs_release；
- rknn_create_mem、rknn_set_io_mem；
- rknn_set_core_mask；
- SDK/driver version、tensor attr、memory size、performance query。

### 9.3 产品级 Runner

最小 C++ runner 应具备：

- 从文件或内存加载模型；
- 查询并校验输入输出属性；
- 支持预分配和复用 buffer；
- 支持 warmup、循环推理和异常退出；
- 记录 NPU core、Runtime、driver 和模型 hash；
- 输出模型耗时、预处理、后处理、队列和端到端耗时；
- 有 CMake、单元测试和可重复启动脚本。

### 通过标准

- Python 与 C++ 输出在容许误差内一致；
- 单模型连续运行 30 分钟无崩溃和明显泄漏；
- 能切换不同输入尺寸或明确拒绝不支持的尺寸；
- 能通过 API 查询并记录 SDK/driver 版本。

## 10. R5：RGA、NV12、零拷贝和硬件预处理

### 10.1 为什么 RGA 是必学

很多 RK3588 项目模型本身很快，但端到端 FPS 很低，原因不是 NPU，而是：

- 摄像头 NV12 转 BGR/RGB 在 CPU 上完成；
- resize、crop、rotate 每帧调用 OpenCV；
- stride、对齐和实际分辨率处理错误；
- 在 NPU 前后反复分配和拷贝大 buffer；
- DMA-BUF fd、虚拟地址和 cache 没有正确同步；
- 后处理或显示线程阻塞了采集线程。

RGA 是独立的 2D 硬件加速器，常用于 resize、rotate、crop、bitblt、色彩转换和 alpha blend。先读 [RGA Developer Guide](https://github.com/airockchip/librga/blob/main/docs/Rockchip_Developer_Guide_RGA_EN.md)，再看 [RGA 实战视频](https://www.bilibili.com/video/BV1uCcfekEK9/)。

### 10.2 必须掌握的概念

NV12、YUV420、RGB、BGR、DRM fourcc、stride、alignment、DMA-BUF fd、importbuffer、wrapbuffer、cache sync、RGA 版本、驱动版本和 buffer 生命周期。

### 10.3 实验矩阵

| 方案 | 预处理位置 | 需要测什么 |
|---|---|---|
| A | OpenCV CPU | 基线耗时和 CPU 占用 |
| B | RGA 同步 API | RGA latency、格式限制和错误 |
| C | RGA + DMA-BUF | 拷贝次数、缓存同步和端到端延迟 |
| D | 采集 buffer 直接接 Runtime | 是否真的减少拷贝，精度是否一致 |
| E | 多路输入共享 buffer pool | 内存峰值、竞争和掉帧 |

### 通过标准

- 能解释每个 buffer 的 producer、consumer 和释放者；
- 能列出摄像头、RGA、NPU、显示之间的格式和 stride；
- 能证明至少一次 CPU 预处理被硬件替换后，端到端指标确实改善；
- 发生格式或权限错误时能从 RGA 日志、dmesg 和驱动版本定位原因。

## 11. R6：MPP、V4L2、GStreamer 和视频链路

### 11.1 MPP 学习目标

MPP 是 Rockchip 的媒体处理平台接口，连接硬件编解码器和上层应用/中间件。重点不是背 API，而是理解：

- decoder/encoder 的 packet、frame、buffer 生命周期；
- H.264/H.265 输入到 NV12/YUV frame 的路径；
- stride、crop、color format 和 buffer ownership；
- MPP 与 GStreamer、FFmpeg、V4L2 的边界；
- 解码、推理、显示、编码各自的线程和队列。

主资料：

- [Rockchip MPP Developer Guide](https://github.com/rockchip-linux/mpp/blob/develop/doc/Rockchip_Developer_Guide_MPP_EN.md)
- [rockchip-linux/mpp](https://github.com/rockchip-linux/mpp)
- [Rockchip GStreamer guide from Firefly](https://github.com/Firefly-docs/software-docs/blob/master/en/Multimedia/GStreamer/GStreamer.md)

如果 GitHub 上游仓库、分支或插件路径发生变化，优先使用目标板 BSP 自带的 MPP 和 GStreamer plugin。不要在产品项目中临时下载一份不同 commit 的库。

### 11.2 推荐 pipeline

#### 文件/网络视频

<code>H264/H265 stream -> MPP/GStreamer decode -> NV12 DMA-BUF -> RGA resize/color -> RKNN -> postprocess -> overlay/encode/RTSP</code>

#### 摄像头

<code>V4L2/CSI sensor -> ISP -> NV12 frame -> RGA -> RKNN -> tracker -> display or MPP encode</code>

### 11.3 实验顺序

1. MPP 单独解码并保存一帧 NV12。
2. V4L2 单独采集并打印 pixel format、width、height、stride。
3. RGA 将采集帧转为模型输入。
4. RKNN 单模型推理。
5. 接入 GStreamer 或编码器。
6. 加入队列、丢帧和恢复策略。
7. 扩展到两路或四路视频。

### 通过标准

- 能不用 OpenCV 作为主链路完成一条硬件解码到 NPU 的 pipeline；
- 能解释 decode latency、preprocess latency、NPU latency、postprocess latency 和 encode latency；
- 能在摄像头断开、网络抖动和解码错误时恢复或安全退出；
- 能用固定输入和固定功耗做多路吞吐测试。

## 12. R7：多模型、多路视频和性能优化

### 12.1 测试分层

先完成以下隔离测试：

1. 单张图片、单模型；
2. 单路视频、单模型；
3. 单路视频、检测加跟踪；
4. 两路视频、同一模型；
5. 多路视频、多个模型；
6. 推理加显示；
7. 推理加编码和网络发送。

每一步记录：

- NPU model latency；
- RGA latency；
- decode/encode latency；
- queue wait；
- end-to-end frame latency；
- FPS、p50、p95、p99；
- CPU、DDR、NPU load；
- 温度、频率、内存峰值；
- 丢帧、错误和恢复次数。

### 12.2 NPU 多核

RK3588 等平台支持多 NPU core 组合模式。学习以下 API 和概念：

- rknn_set_core_mask；
- RKNN_NPU_CORE_0、CORE_1、CORE_2、CORE_ALL 等模式；
- 多 context、多线程和多进程；
- batch 与多路实时输入的区别；
- 单模型 latency 和系统 goodput 的区别。

不要仅通过把 core 数设为最大来判断优化成功。必须比较功耗、温度、内存、队列等待和实时性。

### 12.3 调度项目

设计三个优先级：

- 高优先级：驾驶/机器人安全相关检测；
- 中优先级：跟踪、姿态或深度；
- 低优先级：OCR、分类、日志截图。

加入：

- 有界队列；
- 超时；
- 丢旧帧；
- 推理失败重试；
- 资源不足降级；
- 摄像头断流恢复；
- systemd 自动重启和健康检查。

### 通过标准

- 有单模型、单路、多路和多模型矩阵；
- 能解释 NPU 饱和、CPU 饱和、DDR 瓶颈和媒体瓶颈；
- 连续运行至少 1 小时，结果和资源曲线可复现；
- 能在负载超过预算时给出确定的降级行为。

## 13. R8：不支持算子、模型改图和框架适配

### 13.1 模型改图顺序

1. 固定输入 shape 和 batch；
2. 折叠常量、删除训练节点；
3. 将动态控制流改成静态图；
4. 用 RKNN 支持的算子替换等价表达；
5. 将 NMS、decode 或后处理移出主图；
6. 切分 NPU 子图和 CPU 子图；
7. 评估自定义算子和 GPU fallback 的成本；
8. 与换一个端侧友好模型做对比。

### 13.2 自定义算子不是第一选择

自定义算子需要同时处理：

- PC 转换器是否认识该算子；
- 板端 runtime 是否能加载；
- NPU、GPU 或 CPU 的执行位置；
- 输入输出 tensor 的 layout、dtype 和内存；
- 版本升级和多板卡兼容；
- 精度和性能回归。

只有当算子对业务不可替代且图改写无法接受时才进入这个分支。

### 13.3 框架适配

阅读 [OpenMMLab Rockchip backend guide](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/rockchip.md)，理解部署框架如何把：

- target_platform；
- mean/std；
- input_size；
- quantization dataset；
- backend config；
- model export；

组织成可复用的部署流程。重点是理解抽象，不要把框架默认配置当作所有板卡都正确。

### 通过标准

- 能把一个包含不支持算子的模型改成可部署版本；
- 有改图前后的精度、模型大小和端到端性能对比；
- 能说明 CPU fallback 的真实成本；
- 能维护一个最小 reproducible issue。

## 14. R9：机器人、车载视觉与产品化

### 14.1 机器人项目

**项目：RK3588 摄像头感知节点**

- 输入：USB 或 CSI 摄像头；
- 媒体：V4L2/MPP/GStreamer；
- 预处理：RGA；
- 推理：RKNN C++ Runtime；
- 输出：ROS2 topic；
- 可选：目标跟踪、深度、GPIO 或运动控制；
- 约束：p95 感知延迟、帧率、断流恢复和内存峰值。

### 14.2 汽车视觉项目

公开 RK3588 板卡不能直接等同于车规平台，但可做缩小版验证：

- 两路或四路视频；
- 车辆/行人/车道检测；
- 时间戳、同步和固定队列；
- 硬件解码、RGA、NPU、编码；
- 传感器异常、模型超时和资源超预算降级；
- 生成完整的性能、精度和稳定性报告。

汽车岗位还要补：

ADAS、DMS/OMS、surround view、BEV、3D detection、sensor fusion、ISP、camera synchronization、QNX、AUTOSAR、ISO 26262、ASPICE、ASIL、secure boot、OTA、watchdog。

### 14.3 产品化清单

- 交叉编译和 sysroot 固定；
- 模型、runtime、driver、kernel、rootfs 版本锁定；
- systemd 服务、自启动、健康检查和日志轮转；
- 摄像头和网络断开恢复；
- 温度和降频监测；
- 模型加载失败和内存不足处理；
- OTA 回滚；
- 许可证和第三方依赖清单；
- 一键部署和离线复现。

## 15. 项目阶梯

### P0：MobileNet 分类

- 目标：转换、量化、Lite2、C++ Runtime；
- 产物：PyTorch/ONNX/RKNN/板端输出对比。

### P1：YOLO 检测

- 目标：检测头、后处理、INT8、NMS；
- 产物：mAP、失败样本、单图和视频 FPS。

### P2：RGA + RKNN

- 目标：NV12、resize、色彩转换、stride、buffer；
- 产物：CPU 预处理和 RGA 预处理对比。

### P3：MPP + RKNN 视频 pipeline

- 目标：硬件解码、NPU 推理、显示/编码；
- 产物：端到端 latency、带宽、掉帧和恢复报告。

### P4：多路机器人视觉

- 目标：多摄像头、多模型、NPU core、ROS2；
- 产物：调度矩阵、优先级和降级策略。

### P5：非原生模型适配

- 目标：改图、算子替换、CPU fallback 或自定义算子；
- 产物：可复现转换脚本和精度/性能报告。

### P6：跨平台模型对比

- 目标：同一 ONNX 和同一数据集在 Orin、RDK、RK3588 上对比；
- 产物：平台选择矩阵，不用 TOPS 代替实测。

### P7：RKNN3 协处理器分支

- 目标：进入本文第 21 节的 RKNN3 / RK1828 + RK3588 并行路线；
- 产物：RK3588 主控媒体、RK1828 推理、通信、LLM/VLM 和恢复报告。

## 16. 性能与精度验收模板

~~~text
board_model:
soc:
ram:
os_version:
kernel_version:
bsp_version:
rknn_toolkit_version:
rknn_runtime_version:
rknpu_driver_version:
rga_version:
mpp_version:
model_name:
model_sha256:
target_platform:
input_shape:
input_format:
precision:
calibration_dataset:
decode_ms:
preprocess_ms:
npu_ms:
postprocess_ms:
encode_ms:
queue_ms:
end_to_end_ms:
fps:
p50_ms:
p95_ms:
p99_ms:
cpu_percent:
ddr_usage:
npu_utilization:
memory_mb:
temperature_c:
power_mode:
accuracy_metric:
drop_count:
recovery_count:
notes:
~~~

### 必须分开的结论

- NPU latency 不等于端到端 latency；
- 单帧 FPS 不等于多路 goodput；
- NPU utilization 高不一定代表系统最优；
- RGA latency 低不代表 DMA-BUF/cache 正确；
- 模型精度正确不代表摄像头真实输入契约正确；
- 一次跑通不代表长时间稳定。

## 17. 常见故障排查顺序

### Toolkit2 环境错误

1. 检查 Python、Linux 架构和 Toolkit2 版本；
2. 确认没有混用旧 RKNN-Toolkit；
3. 检查目标平台和官方支持矩阵；
4. 用官方 Model Zoo sample 验证环境；
5. 再加载自己的 ONNX。

### .rknn 无法在板端运行

1. 检查 target_platform；
2. 检查 .rknn、librknnrt、driver 和 kernel；
3. 查询 SDK/driver version；
4. 检查输入 tensor 属性、layout、dtype 和 stride；
5. 用同版本官方模型对照；
6. 保存 dmesg、Runtime 日志和模型 hash。

### 精度低

1. 固定并保存真实板端输入；
2. 对齐 RGB/BGR、NV12、resize、letterbox、mean/std；
3. 对齐输出解码和 NMS；
4. 增大并重新设计 calibration dataset；
5. 做逐层或输出误差分析；
6. 再考虑混合精度、QAT 或改图。

### RGA 报错

1. 查询 RGA API 和 driver 版本；
2. 检查 source/destination format、尺寸、stride 和对齐；
3. 检查 DMA-BUF fd 和权限；
4. 检查 cache sync；
5. 用官方 librga sample 最小复现；
6. 查 dmesg 中的 RGA driver 错误。

### 视频 pipeline 慢

1. 关闭显示和网络发送；
2. 单测 MPP decode；
3. 单测 RGA；
4. 单测 RKNN；
5. 测量每个 queue；
6. 检查是否发生 NV12/BGR 往返拷贝；
7. 固定 governor、温度和输入帧率后再比较。

## 18. 学完后的能力与岗位

### 初级可交付能力

- 能在 RK3588 上完成 ONNX 到 RKNN；
- 能使用 Lite2 和 C++ Runtime 跑通分类或检测；
- 能解释基本量化、输入格式和后处理问题；
- 能读懂官方 Model Zoo 和 RKNPU2 sample。

### 中级端侧部署能力

- 能处理算子不支持、模型改图、混合精度和 CPU fallback；
- 能用 RGA、MPP、V4L2、GStreamer 做视觉 pipeline；
- 能完成摄像头、推理、显示、编码和 ROS2 集成；
- 能做多模型、多路视频和 NPU core 调度；
- 能用原始数据解释瓶颈，而不是只报一个 FPS。

### 高级平台能力

- 能适配不同板卡 BSP、kernel、driver、runtime 和交叉编译环境；
- 能设计 DMA-BUF、buffer pool、零拷贝和故障恢复；
- 能把同一感知系统迁移到 NVIDIA、Horizon 和 Rockchip；
- 能为汽车/机器人项目写版本、性能、精度、温度和稳定性验收方案；
- 能向 RKNPU2、RGA、MPP 或部署框架提交高质量 issue/patch。

中文岗位关键词：瑞芯微 NPU 工程师、RKNN 部署工程师、端侧 AI 工程师、嵌入式视觉工程师、机器人感知工程师、AI 推理优化工程师、C++ 多媒体工程师、BSP/SDK 适配工程师、车载视觉软件工程师。

英文岗位关键词：Rockchip NPU Engineer、RKNN Deployment Engineer、Edge AI Engineer、Embedded Vision Engineer、Robotics Perception Engineer、Inference Optimization Engineer、Multimedia C++ Engineer、BSP/SDK Integration Engineer、Automotive Vision Software Engineer。

公开 RK3588 项目不能直接证明车规量产经验。量产岗位还需要安全、质量、BSP、传感器和长期交付证据。

## 19. 官方代码和文档索引

### RKNN / RKNPU

- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [RKNPU2 in RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2)
- [RKNPU2 API header in Model Zoo](https://github.com/airockchip/rknn_model_zoo/blob/main/3rdparty/rknpu2/include/rknn_api.h)
- [RKNN LLM optional branch](https://github.com/airockchip/rknn-llm)
- [OpenMMLab Rockchip backend](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/rockchip.md)

### RKNN3 / RK1828

- 本文第 21 节：RKNN3 / RK1828 + RK3588 并行路线
- [airockchip/rknn3-toolkit](https://github.com/airockchip/rknn3-toolkit)
- [RKNN3 Toolkit 中文 README](https://github.com/airockchip/rknn3-toolkit/blob/main/README_CN.md)
- [airockchip/rknn3-model-zoo](https://github.com/airockchip/rknn3-model-zoo)
- [Rockchip RKNN3 发布说明](https://www.rock-chips.com/a/cn/news/rockchip/2026/0309/2163.html)

### RGA / 媒体 / 系统

- [airockchip/librga](https://github.com/airockchip/librga)
- [RGA English Developer Guide](https://github.com/airockchip/librga/blob/main/docs/Rockchip_Developer_Guide_RGA_EN.md)
- [rockchip-linux/mpp](https://github.com/rockchip-linux/mpp)
- [MPP English Developer Guide](https://github.com/rockchip-linux/mpp/blob/develop/doc/Rockchip_Developer_Guide_MPP_EN.md)
- [rockchip-linux/kernel](https://github.com/rockchip-linux/kernel)
- [rockchip-linux/rkbin](https://github.com/rockchip-linux/rkbin)
- [rockchip-linux GitHub Organization](https://github.com/rockchip-linux)
- [Firefly RKNN docs](https://github.com/Firefly-docs/ai-docs/tree/master/en/Dev%20Toolchain/RKNN)
- [Firefly GStreamer docs](https://github.com/Firefly-docs/software-docs/blob/master/en/Multimedia/GStreamer/GStreamer.md)

### 免费视频

- [迅为 RKNPU2 从入门到实践](https://www.bilibili.com/video/BV1Kj411D78q/)
- [合众恒跃 RK3588 RKNPU2 五部分系列](https://www.bilibili.com/video/BV1eB4y1Z7gV/)
- [RKNN 模型转换与评估五部分系列](https://www.bilibili.com/video/BV15w4m1Y7QL/)
- [瑞芯微 RK 系列端侧 AI 长系列](https://www.bilibili.com/video/BV1QLR7BqE3d/)
- [瑞芯微公开课：新一代 NPU](https://www.bilibili.com/video/BV1K44y1g7o5/)
- [RK3588 RGA 实战](https://www.bilibili.com/video/BV1uCcfekEK9/)

## 20. 执行清单

### R0-R3

- [ ] 完成 NVIDIA 和 Horizon 路线中的 Linux/C++/ONNX/量化基础
- [ ] 看完 R-C1 的 Toolkit2、转换、评估和量化章节
- [ ] 用官方 Model Zoo 跑通一个分类模型
- [ ] 完成 MobileNet 或 ResNet 的 .rknn 转换
- [ ] 保存 PyTorch、ONNX、RKNN host 和板端输出

### R4-R6

- [ ] 写出 Lite2 和 C++ Runtime 双版本
- [ ] 使用 rknn_query 记录 tensor 和 SDK/driver 版本
- [ ] 完成 CPU resize 与 RGA resize 对比
- [ ] 完成 MPP/V4L2/GStreamer 到 RKNN 的单路 pipeline
- [ ] 分开记录 decode、RGA、NPU、后处理和 encode

### R7-R9

- [ ] 完成两路以上视频和多模型并发
- [ ] 完成 NPU core、优先级、队列和丢帧策略
- [ ] 适配一个包含不支持算子的模型
- [ ] 完成机器人 ROS2 感知节点或车载视觉缩小版
- [ ] 做 Orin、RDK、RK3588 的同模型对比

### 最终作品

- [ ] 一个公开 Git 仓库
- [ ] 固定版本和一键部署脚本
- [ ] C++ Runtime、RGA/MPP pipeline 和故障恢复
- [ ] 精度、延迟、功耗、温度、内存和稳定性报告
- [ ] 至少一个可复现的模型适配或性能优化案例
- [ ] 进入 RKNN3 并行路线完成 RK3588 + RK1828 协处理器项目

---

**执行原则：先用官方 Model Zoo 跑通，再换自己的模型；先把 .rknn、Runtime、driver、BSP 配对，再讨论性能；先测 CPU/RGA/MPP/NPU 各段，再做零拷贝和多路并发；先把 RK3588 Linux 做扎实，再迁移到其他 Rockchip 平台。**

## 21. RKNN3 / RK1828 + RK3588 并行路线

> 本节为瑞芯微主路线中的 RKNN3 协处理器分支。RKNN2/RKNPU2 与 RKNN3/RK1828 共享 Linux、C/C++、ONNX、媒体和 benchmark 基础，但 Toolkit、Runtime、模型资产和性能数据分别管理。
> 推荐在 RKNN2 的共享基础达到 R0-R4 后切入，也可以在具备等价基础后直接开始；RKNN2 的 R5-R9 与本分支可以并行推进。

### 21.1 先给结论

RKNN3 不是 RKNN2 的简单升级，也不是把 RKNN-Toolkit2 换一个包名。官方 RKNN3 SDK 由三部分组成：

| 层次 | 官方组件 | 位置 | 主要职责 |
|---|---|---|---|
| PC 工具 | RKNN3-Toolkit | x86/Windows/Linux 主机 | 模型转换、PC 推理、性能评估和部署准备 |
| 板端 Runtime | RKNN3 Runtime | RK3588 主控侧的 Linux/Android 应用 | C/C++ session、输入输出、通信和推理调度 |
| 示例仓库 | RKNN3 Model Zoo | PC + 板端 | CNN、LLM、VLM、ASR、TTS、Embedding 和 OCR 的完整参考实现 |

你的主工作流应当理解为：

<code>PyTorch/HuggingFace -> ONNX -> RKNN3-Toolkit -> RKNN3 model/weight -> RK3588 host app -> PCIe/USB/board transport -> RK1828 coprocessor -> C/C++ result</code>

在视觉产品中还要补上：

<code>camera/decoder -> RK3588 V4L2/MPP -> RGA/DMA-BUF -> RKNN3 Runtime -> postprocess -> ROS2/vehicle application</code>

这条路线最终要证明的不是“跑过一个 demo”，而是你能：

- 判断一个模型应该放在 RK3588 本体、RK1828 协处理器还是 CPU 上；
- 把 ONNX 模型和对应的配置、tokenizer、embedding、weight 等配套文件完整导出；
- 用 RKNN3 Runtime 的 C/C++ 接口建立可观测、可恢复的 session；
- 分离主控媒体处理、数据传输、协处理器推理和后处理的耗时；
- 解决 PCIe/USB、内存、队列、并发、温度和模型版本问题；
- 把 CNN 视觉链路进一步扩展到 LLM/VLM、语音和机器人应用。

### 21.2 RKNN3 与 RKNN2 的边界

这是整条并行路线最重要的判断。

| 维度 | RKNN2 / RKNPU2 主线 | RKNN3 / RK1828 并行线 |
|---|---|---|
| 典型硬件 | RK3588、RK3576、RK356x、RV11xx 等 SoC 本体 | RK3588/RK3588S 主控 + RK1820/RK1828 协处理器 |
| PC 工具 | RKNN-Toolkit2 | RKNN3-Toolkit |
| 板端接口 | RKNN Lite2、RKNPU2、librknnrt.so | RKNN3 Runtime、librknn3_api.so、librknn3_api_rkcp.so |
| 计算位置 | 主 SoC 内置 NPU | 主控负责系统和媒体，RK1828 负责协处理器推理 |
| 连接 | SoC 内部 NPU 设备路径 | PCIe、USB 或板卡实际提供的通信方式 |
| 典型模型 | CNN、检测、分割、姿态和轻量视觉 | CNN、LLM、VLM、ASR、TTS、Embedding、OCR |
| 模型产物 | RKNN2 工具链生成的 .rknn | RKNN3 工具链生成的 RKNN 模型及配套权重文件 |
| 典型优化 | 算子支持、INT8、RGA、零拷贝和多路 NPU | session、传输、权重/KV cache、并发和主控/协处理器流水线 |

官方 RKNN3 Toolkit README 明确说明 RKNN3-Toolkit 与旧 RKNN-Toolkit、RKNN-Toolkit2 不兼容。实际项目中必须把以下内容作为一个版本闭环：

<code>board BSP + kernel driver + RKNN3 Runtime + RKNN3 Toolkit + Model Zoo commit + model files</code>

以下内容不能跨线直接复制：

- RKNN2 的转换脚本不能默认用于 RKNN3；
- RKNN2 的 RKNPU2 C API 不能替代 RKNN3 Runtime；
- RKNN2 的 .rknn 文件不能因为扩展名相同就判定可以运行；
- RKNN2 的 core mask、异步接口和性能结论不能直接套到 RK1828；
- RKNN2 视频里的命令只能学习概念，不能替代当前 RKNN3 sample。

### 21.3 硬件拓扑：RK3588 + RK1828

#### 21.3.1 主控和协处理器如何分工

以 RK3588 + RK1828 为第一学习平台时，可以先按下面的系统职责建模：

| 部件 | 适合承担的工作 |
|---|---|
| RK3588 CPU | 应用控制、传感器管理、后处理、ROS2、日志、网络和故障恢复 |
| RK3588 VPU/MPP | H.264/H.265/音视频解码和编码 |
| RK3588 RGA | resize、crop、色彩转换、旋转、stride 和 buffer 处理 |
| RK3588 内存系统 | 摄像头、解码器、RGA、应用之间的 buffer 管理 |
| RK1828 | RKNN3 CNN/LLM/VLM/语音等模型推理 |
| 主控系统 | session 生命周期、任务队列、超时、重试、版本检查和服务化 |

这不是固定的产品分工。小模型、低延迟控制和简单后处理可能仍适合放在 RK3588 本体；RK1828 的价值在于把较重的 AI 负载从主控 CPU/NPU 路径中拆出来。最终放置位置必须用实测 latency、带宽、内存和功耗决定。

#### 21.3.2 连接方式要以板卡实现为准

官方资料描述 RK1820/RK1828 与主控平台可以通过 PCIe/USB 等高速方式交互，但不同开发板会改变：

- 枚举方式和设备节点；
- 驱动安装方式；
- 是否支持热插拔、休眠和唤醒；
- host 端 Runtime 的通信后端；
- 数据传输的内存拷贝和带宽；
- Linux 与 Android 的启动流程。

因此不要把某块板的 <code>lspci</code>、<code>lsusb</code>、设备节点或服务名写成所有 RK1828 板卡的通用规则。先阅读目标板的硬件手册和 Model Zoo demo README，再固定启动脚本。

#### 21.3.3 目标命名陷阱

官方 Toolkit README 的“支持平台”列表主要列出 RK1820、RK1828 和 RK3572，并把 RK3588、RK3576 等本体 NPU 平台引导到 RKNN2。这个表描述的是模型/加速器目标，不是否定 RK3588 + RK1828 组合。

在 RK3588 + RK1828 组合中：

- RK3588 是 host SoC，负责 Linux/Android 应用、媒体和 RKNN3 host-side demo；
- RK1828 是 coprocessor，负责 RKNN3 AI 执行；
- Model Zoo 的平台表和 build target 用来选择 host/coprocessor 组合；
- RK3588 本体 NPU 的模型仍然属于 RKNN2，不要因为 host 名称是 rk3588 就切换到 RKNN2 模型。

因此构建时要同时看 <code>target platform</code>、host SoC、coprocessor 和 demo 的 Runtime 库，不能只看一个芯片名称。

#### 21.3.4 适合的硬件学习顺序

| 顺序 | 平台 | 用途 |
|---|---|---|
| R3-H1 | RK3588 + RK1828，Linux | 本路线主平台，先跑官方 CNN 和 Qwen 示例 |
| R3-H2 | RK3588 + RK1828，摄像头/解码输入 | 建立视觉媒体到协处理器的流水线 |
| R3-H3 | RK3588 + RK1828，多 session/多路 | 做吞吐、队列、内存和稳定性优化 |
| R3-H4 | RK3588 + RK1828，Android | 了解移动/终端系统差异 |
| R3-H5 | RK3576 + RK1828 | 做 host SoC 迁移和兼容性报告 |

RK3576 只是后续迁移平台，不应替代你当前要练的 RK3588 + RK1828 主线。

### 21.4 关键字地图

#### 21.4.1 模型和工具链

RKNN3-Toolkit、RKNN3 Runtime、RKNN3 Model Zoo、ONNX、PyTorch、HuggingFace、ModelScope、Python 3.10/3.12、模型配置、tokenizer、embedding、weight-separated、GRQ、AWQ、KV cache、LoRA、session pause/resume。

#### 21.4.2 主控和通信

RK3588 host、RK1828 coprocessor、RKCP、PCIe、USB、Ethernet transport、设备枚举、服务端、session、通信后端、超时、重连、休眠/唤醒、模型加密、组件版本校验。

#### 21.4.3 视觉媒体

V4L2、MPP、RGA、DMA-BUF、NV12、RGB、stride、cache sync、buffer pool、GStreamer、零拷贝、硬件解码、硬件编码、帧丢弃、时间戳、ROS2 image pipeline。

#### 21.4.4 LLM/VLM 和语音

Qwen2.5、Qwen3、Qwen2.5-VL、Qwen3-VL、FastVLM、Gemma-4、SmolVLM、GLM-Edge、Qwen3-ASR、Qwen3-TTS、VITS、Whisper、SenseVoice、Embedding、Reranker、OCR。

#### 21.4.5 性能和产品化

TTFT、TPOT、decode TPS、prefill、KV cache、模型加载、权重加载、host-to-coprocessor transfer、queue depth、multi-session、CPU affinity、内存峰值、温度、功耗、错误率、systemd、health check、watchdog、灰度升级。

### 21.5 真实课程和资料策略

#### 21.5.1 先说清楚视频现状

截至本路线核验日期，没有找到一套稳定、完整、从 RKNN3 Toolkit 到 RK1828 + RK3588 C++ 产品部署的公开专题视频课程。网上较完整的瑞芯微视频主要是 RKNN2/RKNPU2，短视频则多为板卡演示。

因此本路线采用下面的真实组合：

1. 用 RKNN2 完整视频学习共享概念：ONNX、量化、Runtime、RGA、MPP、C++ 和端到端 benchmark。
2. 用官方 RKNN3 Model Zoo 作为真正的 RKNN3 实操主课程：它包含导出、转换、构建、推理示例和模型目录。
3. 用官方 RKNN3 Toolkit README 校正组件边界、平台支持和不兼容关系。
4. 用板卡厂商文档完成 RK1828 驱动、枚举和板端环境 bring-up。
5. 用开源项目完成 LLM/VLM 或语音的高级专项验证。

这意味着：视频负责建立方法，官方仓库负责给出当前 API 和命令，板卡文档负责硬件差异，项目负责证明结果。不能把 RKNN2 视频标题改成 RKNN3 课程。

#### 21.5.2 中文视频：共享基础，不复制 API

| 编号 | 视频 | 适合学习的节点 | 使用边界 |
|---|---|---|---|
| R3-C1 | [迅为 RKNPU2 从入门到实践](https://www.bilibili.com/video/BV1Kj411D78q/) | RKNN 转换、INT8、Lite2、C++、零拷贝和项目结构 | 只迁移概念，命令和 API 回到 RKNN3 Model Zoo |
| R3-C2 | [RK3588 RKNPU2 五部分系列](https://www.bilibili.com/video/BV1eB4y1Z7gV/) | RK3588 硬件、RKNPU2 全流程和排障 | 作为快速总览，不作为 RK1828 Runtime 教程 |
| R3-C3 | [RKNN 模型转换与评估系列](https://www.bilibili.com/video/BV15w4m1Y7QL/) | ONNX、量化数据集、精度对齐和性能评估 | 转换 API 需要替换为 RKNN3 版本 |
| R3-C4 | [瑞芯微 RK 系列端侧 AI 长系列](https://www.bilibili.com/video/BV1QLR7BqE3d/) | YOLO、MobileNet、部署工程和板端调试 | 用于补视觉项目结构 |
| R3-C5 | [RK3588 RGA 实战](https://www.bilibili.com/video/BV1uCcfekEK9/) | RGA、图像格式、resize 和预处理 | 与 RKNN3 的媒体前处理直接共享 |

每个视频节点都要配套阅读官方示例。视频里出现 <code>rknn_init</code>、<code>librknnrt.so</code>、RKNN2 core mask 等内容时，只把它当作 NPU Runtime 的概念示例，不要原样移植到 RKNN3。

#### 21.5.3 官方中文/英文主资料

| 编号 | 资料 | 语言 | 作用 |
|---|---|---|---|
| R3-D1 | [RKNN3 Toolkit 中文 README](https://github.com/airockchip/rknn3-toolkit/blob/main/README_CN.md) | 中文 | 组件、支持平台、模型类别、Python 版本和不兼容说明 |
| R3-D2 | [RKNN3 Toolkit English README](https://github.com/airockchip/rknn3-toolkit/blob/main/README.md) | English | 英文术语、版本变化和 issue 入口 |
| R3-D3 | [RKNN3 Model Zoo English README](https://github.com/airockchip/rknn3-model-zoo/blob/main/README.md) | English | 完整导出、转换、构建、部署流程 |
| R3-D4 | [RKNN3 Model Zoo 中文 README](https://github.com/airockchip/rknn3-model-zoo/blob/main/README_CN.md) | 中文 | 中文模型目录和使用说明 |
| R3-D5 | [Rockchip RKNN3 发布说明](https://www.rock-chips.com/a/cn/news/rockchip/2026/0309/2163.html) | 中文 | RK1820/RK1828 产品定位和官方发布背景 |
| R3-D6 | [DFRobot RK1828 getting started](https://wiki.dfrobot.com/dfr1263/docs/24734) | 中文/English | 板卡驱动、rknn-smi、PCIe 连接和设备检查示例 |
| R3-D7 | [Forlinx RK1820/RK1828 guide](https://docs.forlinx.net/ai-accelerator/rk1820_rk1828/RK1820_RK1828_AI_Accelerator_Development_Guide.html) | English | RK3588 + RK182X 板卡构建和 C++ 部署示例 |

R3-D6 和 R3-D7 是第三方板卡资料，适合学习 bring-up，不能替代 RKNN3 官方支持矩阵；R3-D7 可能受站点访问策略影响，打不开时以官方 Model Zoo 和 R3-D6 为准。

#### 21.5.4 完整项目型资料

| 资料 | 级别 | 用法 |
|---|---|---|
| [airockchip/rknn3-model-zoo](https://github.com/airockchip/rknn3-model-zoo) | 核心 | 按 examples 目录做主线，优先 CNN，再 Qwen，再 VLM |
| [rkvoice-stream RK1828 Qwen3-TTS](https://github.com/suharvest/rkvoice-stream/blob/main/docs/rk1828-qwen3-tts.md) | 高级专项 | 学习 RK1828 语音部署、文件组织和实际问题 |
| [rkvoice-stream Gemma4/RK1828](https://github.com/suharvest/rkvoice-stream/blob/main/docs/rk1828-gemma4.md) | 高级专项 | 学习多模态模型和第三方集成，先核对当前 SDK |

第三方项目可帮助理解真实工程，但其板卡、模型版本和运行时假设必须逐项复核。

### 21.6 前置能力

#### 21.6.1 必须已经具备

- Linux shell、SSH、动态库、进程线程和 systemd；
- C++17、CMake、RAII、线程安全和基本 gdb；
- Python 虚拟环境、NumPy、OpenCV、PyTorch 和 ONNX；
- 模型输入输出、固定 shape、opset、量化和精度回归；
- RGA、MPP、V4L2、GStreamer 和 DMA-BUF 的基本概念；
- benchmark 中的 warmup、p50/p95/p99、吞吐、内存和温度记录。

#### 21.6.2 RKNN3 新增能力

- 主控/协处理器系统拓扑；
- PCIe、USB 或板卡实际 transport 的设备枚举；
- C/C++ session 生命周期；
- 模型、weight、tokenizer、embed 和 config 的配套管理；
- LLM 的 prefill、decode、TTFT、TPOT 和 KV cache；
- 多 session、暂停/恢复、超时和重连；
- host 媒体 buffer 到 RK1828 Runtime 的数据路径。

#### 21.6.3 版本记录模板

~~~text
board:
host_soc: RK3588
coprocessor: RK1828
os:
kernel:
bsp:
transport: PCIe / USB / Ethernet / board-specific
driver:
rknn3_toolkit:
rknn3_runtime:
model_zoo_commit:
python:
compiler:
model_commit:
model_sha256:
model_assets: rknn / weight / tokenizer / embed / config
input_format:
power_mode:
temperature:
benchmark_command:
git_commit:
~~~

### 21.7 阶段路线总表

| 阶段 | 建议时间 | 关键字 | 主资料 | 验收产物 |
|---|---:|---|---|---|
| R3-0 | 1 周 | Linux、C++、ONNX、RKNN2 共享基础 | R3-C1 + R3-D3 | 能读懂并修改一个官方 demo |
| R3-1 | 1 周 | RK1828、RK3588、PCIe/USB、设备枚举 | R3-D5 + R3-D6 | 硬件拓扑图和版本矩阵 |
| R3-2 | 2 周 | Toolkit、ONNX、build、model assets | R3-D1 + R3-D3 | 一个 CNN 模型完整转换 |
| R3-3 | 2 周 | Runtime、C/C++、session、输入输出 | R3-D3 + R3-D7 | C++ 单模型推理程序 |
| R3-4 | 2 周 | CNN、量化、精度、后处理 | R3-C3 + Model Zoo | MobileNet/YOLO 报告 |
| R3-5 | 2-3 周 | Qwen、LLM、weight、tokenizer、KV cache | Model Zoo Qwen examples | 端侧 LLM 交互 demo |
| R3-6 | 2-3 周 | VLM、视觉输入、媒体管线 | Model Zoo FastVLM/Qwen-VL | 图像问答或视频抽帧 demo |
| R3-7 | 2 周 | MPP、RGA、V4L2、DMA-BUF | R3-C5 + MPP/RGA docs | 摄像头/视频到推理 pipeline |
| R3-8 | 2-3 周 | 多 session、传输、队列、性能 | Runtime examples + 自测 | TTFT/TPOT/FPS/带宽报告 |
| R3-9 | 2 周 | systemd、watchdog、恢复、Android | Model Zoo build scripts | 可部署服务和故障演练 |
| R3-10 | 2-4 周 | 机器人/汽车场景、跨平台 | 统一项目模板 | RKNN2/RKNN3/Horizon 对比 |

不要把 R3-5 的 LLM demo 当作 R3-8 的性能优化完成。能生成文本只说明功能链路通了，不能说明多路视觉和产品化完成。

### 21.8 分阶段学习与实践

#### R3-0：共享基础复用

目标是快速复用 NVIDIA 和 RKNN2 主线已经学过的知识：

- 阅读 R3-C1 的 ONNX、量化、Runtime、内存和 benchmark 部分；
- 画出 RKNN2 与 RKNN3 的组件边界；
- 用 C++ 写一个输入、推理、输出和耗时统计的最小框架；
- 用同一个 ONNX 模型保存 host 参考结果；
- 建立模型、代码、板端输出和 benchmark 的目录规范。

验收：

- 能解释 RKNN2 的 RKNPU2 路径为什么不能直接替换 RKNN3；
- 能在没有板卡时完成 ONNX 图检查和输出契约；
- 能把所有实验写成可重复命令。

#### R3-1：RK1828 + RK3588 bring-up

先不急着跑大模型，先完成硬件和通信确认：

1. 确认板卡型号、RK3588 变体、RK1828 模块、Linux/Android 和 BSP；
2. 查看 kernel log、设备枚举、驱动、Runtime 和 sample 版本；
3. 按板卡文档运行设备状态工具，例如 <code>rknn-smi</code>；
4. 运行 Model Zoo 提供的最小 CNN 或 session test；
5. 记录 RK1828 是否能被发现、模型是否能加载、推理是否有输出；
6. 断开或重启协处理器，验证应用的错误路径。

不要仅凭“程序启动成功”判断设备可用。至少要保存：

- 枚举日志；
- Runtime 版本；
- 模型加载日志；
- 首次推理输出；
- 设备温度和内存；
- transport 类型和板卡连接方式。

#### R3-2：RKNN3 Toolkit 和模型转换

按照 Model Zoo 的真实流程练习：

<code>HuggingFace/PyTorch export -> ONNX -> RKNN3 conversion -> model assets -> board build</code>

优先顺序：

1. MobileNet 或 ResNet；
2. YOLO 检测模型；
3. 一个带自定义后处理的视觉模型；
4. Qwen2.5-0.5B 或官方当前可用的轻量 LLM；
5. FastVLM 或 Qwen-VL。

每个模型都保存：

- 原始权重和 commit；
- ONNX 文件与 opset；
- RKNN3 conversion config；
- 量化设置和 calibration 数据；
- RKNN、weight、tokenizer、embed、config 等配套文件；
- host 参考输出；
- 板端输出和误差；
- 转换日志与警告。

Model Zoo 的 LLM 示例会生成 ONNX、配置、tokenizer、embed 等文件，部分示例还会采用权重分离方式生成 RKNN 和 weight 文件。不要只上传一个模型文件。

#### R3-3：C/C++ Runtime

先读官方 demo 的目录和构建脚本，再写自己的 wrapper。wrapper 至少应包含：

- 设备和组件版本检查；
- session 创建、加载和释放；
- 模型资产路径校验；
- 输入格式和 shape 校验；
- warmup 和正式推理分离；
- 单次推理耗时和阶段耗时；
- 错误码、超时和重试策略；
- 退出时资源回收；
- 可选的多 session 管理。

官方 Model Zoo 对 RK3588/RK3576 的 demo 使用 <code>librknn3_api.so</code> 和 <code>librknn3_api_rkcp.so</code>。库名、目录和加载方式以当前 demo 为准，不要从网上复制旧的 <code>librknnrt.so</code>。

#### R3-4：CNN 视觉基线

先用 CNN 把通信和 Runtime 变成可测量基线：

- MobileNet：分类和最小链路；
- YOLO：输入预处理、输出解析和 NMS；
- ResNet：固定 shape 与吞吐；
- DINOv3/SigLIP 等 Model Zoo 当前支持模型：理解 embedding 或视觉特征接口。

报告至少包含：

| 指标 | 说明 |
|---|---|
| model load | 模型和权重加载时间 |
| first inference | 首次推理时间 |
| warm inference | 预热后的 p50/p95/p99 |
| transport | host 到 RK1828 的数据传输 |
| preprocess | CPU/RGA 时间 |
| postprocess | NMS、解码或 embedding 后处理 |
| end-to-end | 从输入到应用结果 |
| memory | RK3588 和 RK1828 两侧峰值 |
| temperature | 稳态温度和降频情况 |

#### R3-5：LLM 路线

以官方 Model Zoo 的 Qwen2.5 或 Qwen3 示例为主：

1. 导出结构和权重；
2. 生成或准备量化配置；
3. 转换 RKNN 与 weight；
4. 准备 tokenizer、embed 和 config；
5. 在 RK3588 host 上构建；
6. 在 RK1828 上启动 C++ demo；
7. 记录首 token、生成速度、上下文长度和内存；
8. 测试短输入、长输入、重复请求和异常中断。

必须区分：

- model load time；
- prefill time；
- TTFT；
- decode time；
- TPOT；
- tokens per second；
- tokenizer 和 detokenizer CPU 时间；
- host/协处理器传输时间；
- KV cache 内存。

不要只记录一个“对话速度”。对交互式机器人，TTFT 和稳定的 TPOT 通常比一次性总耗时更有解释力。

#### R3-6：VLM、ASR 和 TTS

按难度选择一个专项：

| 专项 | 推荐入口 | 适合场景 |
|---|---|---|
| VLM | FastVLM、Qwen2.5-VL、Qwen3-VL | 摄像头问答、场景描述和异常解释 |
| ASR | Qwen3-ASR、SenseVoice、Whisper | 语音指令和机器人交互 |
| TTS | Qwen3-TTS、VITS | 机器人反馈和车载语音 |
| Embedding/Reranker | Qwen3-Embedding、Qwen3-Reranker | 本地检索和任务匹配 |
| OCR | PaddleOCR-VL | 仪表、标签和车牌类视觉文字 |

第一版不要同时做 VLM、ASR、TTS。先完成一个模型的导出、板端 Runtime、性能和错误恢复，再加入第二个模型。

#### R3-7：RK3588 媒体到 RK1828 推理

视觉端侧的核心不是单独的模型调用，而是：

<code>camera/stream -> V4L2 or MPP -> RGA -> buffer pool -> RKNN3 Runtime -> postprocess -> ROS2/vehicle app</code>

分阶段实现：

1. 文件输入 + CPU preprocess + RKNN3；
2. 文件输入 + RGA preprocess + RKNN3；
3. MPP/V4L2 decode + CPU copy + RKNN3；
4. MPP/V4L2 decode + DMA-BUF/RGA + RKNN3；
5. 多路输入、有界队列和丢帧策略；
6. 后处理与下一帧预处理并行。

每一阶段都保存相同输入下的精度和延迟，避免“换了 pipeline 后 FPS 变高但结果变差”。

#### R3-8：通信、并发和性能

将一次请求拆成：

<code>enqueue -> input preparation -> host/transport transfer -> coprocessor execution -> output transfer -> postprocess -> publish</code>

实验矩阵至少包括：

- 单 session 与多 session；
- 单路与多路；
- 短输入与长输入；
- CPU copy 与 DMA-BUF；
- 同步与异步；
- warm cache 与冷启动；
- 低温稳态与高温稳态；
- 正常设备与重启/断连恢复。

重点观察：

- transport 是否成为瓶颈；
- queue depth 是否造成尾延迟；
- RK3588 CPU 是否被 tokenizer、后处理或内存拷贝打满；
- RK1828 是否空闲等待输入；
- 多 session 是否增加内存或互相干扰；
- 模型加载和权重加载是否重复发生；
- 设备休眠/唤醒后是否需要重新初始化。

#### R3-9：产品化

至少完成一个长期运行服务：

- systemd 自启动；
- 模型和 Runtime 版本检查；
- 健康检查和心跳；
- 超时、重试和有限次数重启；
- 摄像头断流恢复；
- 协处理器断连后的降级策略；
- 日志分级和指标导出；
- 模型灰度替换和 hash 校验；
- 运行 12-24 小时的稳定性测试。

这里的“产品化”是工程练习，不等同于车规认证或量产交付。

### 21.9 官方 Model Zoo 作为完整实操课程

由于 RKNN3 专题视频稀缺，建议把官方 Model Zoo 当作一门按代码推进的课程。

#### 21.9.1 课程顺序

| 课次 | Model Zoo 目录/主题 | 输出 |
|---|---|---|
| L1 | MobileNet/ResNet | 第一个 CNN 模型和 C++ 推理 |
| L2 | YOLO | 检测后处理、输入格式和精度 |
| L3 | Qwen2.5 | LLM 资产、权重和 tokenizer |
| L4 | Qwen3 | KV cache、session 和长上下文实验 |
| L5 | FastVLM | 视觉编码器与 LLM 组合 |
| L6 | Qwen3-ASR/TTS | 音频输入输出和多模型应用 |
| L7 | build-linux/build-android | Linux/Android 构建差异 |
| L8 | 多 session/版本功能 | 并发、休眠唤醒和版本兼容 |

#### 21.9.2 官方构建命令的学习方式

Model Zoo 当前示例中可以看到类似下面的 Linux 构建形式：

~~~bash
./build-linux.sh -t rk3588 -a aarch64 -b Release -d Qwen2_5
~~~

这只是示例，不是所有 demo 的固定命令。执行前应检查：

- demo 名称是否存在；
- target 是否是 <code>rk3588</code>；
- host/board 架构；
- 交叉编译器；
- 模型资产路径；
- 当前 demo 是否要求 Android 或特定设备服务。

#### 21.9.3 每个 demo 的阅读顺序

1. README 和支持平台；
2. Python export 脚本；
3. ONNX 和 config 产物；
4. RKNN conversion 脚本；
5. C++ main 和 Runtime wrapper；
6. CMake/build 脚本；
7. lib 目录和模型资产；
8. benchmark 输出；
9. 失败处理和日志。

### 21.10 项目阶梯

#### P3-0：设备健康检查

目标：证明 RK3588 能发现 RK1828，并且 Runtime、驱动和通信链路一致。

功能：

- 输出 host/coprocessor 信息；
- 记录 transport 和设备状态；
- 加载最小 CNN；
- 做一次推理；
- 设备断连后输出明确错误；
- 保存版本和日志。

验收：换一次冷启动、重启一次服务、重复运行 100 次，结果和错误都可解释。

#### P3-1：RKNN3 CNN 基线

目标：用 MobileNet 或 YOLO 完成：

<code>ONNX -> RKNN3 model -> C++ Runtime -> result</code>

验收：

- host 参考与板端输出在容差内；
- load、transfer、inference、postprocess 分段计时；
- 提供一键构建和运行脚本；
- 记录完整版本矩阵。

#### P3-2：RK3588 视频到 RK1828

目标：主控完成视频输入和预处理，RK1828 完成推理。

功能：

- 文件或摄像头输入；
- MPP/V4L2 解码；
- RGA resize/色彩转换；
- buffer pool；
- RKNN3 C++ Runtime；
- NMS 和结果可视化；
- FPS、端到端 latency 和丢帧统计。

验收：至少支持单路稳定运行，并能说明 CPU copy、RGA、transport 和推理各自占比。

#### P3-3：端侧 LLM 服务

目标：基于 Qwen2.5 或官方当前可用模型完成本地交互服务。

功能：

- 请求队列；
- tokenizer；
- prefill/decode；
- TTFT/TPOT；
- 最大上下文和超时；
- 取消请求；
- 内存上限；
- 模型热启动；
- systemd 服务。

验收：短请求、长请求、并发请求和异常取消都不会导致进程失控。

#### P3-4：VLM 视觉问答

目标：把 RK3588 摄像头/视频帧和 RK1828 VLM 组合起来。

功能：

- 帧采样和时间戳；
- 图像预处理；
- 视觉 encoder；
- LLM prompt；
- 结果发布到 ROS2 或 HTTP；
- 低频语义任务不阻塞高频检测任务。

验收：检测和问答有独立队列、独立频率和超时策略。

#### P3-5：机器人感知协处理器

目标：形成机器人可用的小型异构系统：

- RK3588 负责摄像头、解码、RGA、ROS2 和控制；
- RK1828 负责一个高负载视觉/语言模型；
- CPU 负责后处理、状态机和安全降级；
- 断连时切换到轻量模型或仅保留基础感知；
- 记录每个时间戳和数据版本。

验收：连续运行、模拟断连、摄像头断流、模型超时和温度升高时，系统行为可预测。

#### P3-6：RKNN2 与 RKNN3 对比

同一视觉模型分别跑：

- RK3588 本体 NPU 的 RKNN2/RKNPU2；
- RK3588 + RK1828 的 RKNN3；
- Horizon RDK；
- NVIDIA Orin。

对比：

- 模型转换难度；
- 算子覆盖；
- 端到端 latency；
- NPU/协处理器利用率；
- 主控 CPU 占用；
- 内存；
- 多路扩展；
- 功耗和温度；
- 生态资料；
- 版本和交付风险。

不要用单一 TOPS 数字替代这张表。

### 21.11 Benchmark 模板

#### 21.11.1 CNN/视觉

~~~text
model:
input_shape:
input_format:
dataset:
host_soc:
coprocessor:
transport:
toolkit/runtime/driver:
model_sha256:
preprocess_cpu_ms:
preprocess_rga_ms:
transfer_in_ms:
coprocessor_inference_ms:
transfer_out_ms:
postprocess_ms:
end_to_end_p50_ms:
end_to_end_p95_ms:
fps:
streams:
cpu_percent:
memory_host_mb:
memory_coprocessor_mb:
temperature:
power:
drop_rate:
error_rate:
~~~

#### 21.11.2 LLM/VLM

~~~text
model:
quantization:
context_length:
prompt_tokens:
generated_tokens:
model_load_s:
prefill_ms:
ttft_ms:
tpot_ms:
decode_tokens_per_second:
tokenizer_cpu_ms:
transport_ms:
kv_cache_memory_mb:
peak_memory_mb:
concurrency:
temperature:
power:
error_rate:
~~~

#### 21.11.3 结果解释

每次报告都要回答：

1. 瓶颈在 RK3588 CPU、媒体、内存、transport 还是 RK1828；
2. 增加并发后是吞吐提高还是尾延迟恶化；
3. 单模型快是否掩盖了预处理和传输开销；
4. 量化是否改变了精度或输出稳定性；
5. 长时间运行是否出现降频、内存增长或设备异常；
6. 换成 RKNN2 后哪些结论仍然成立，哪些必须重测。

### 21.12 常见故障排查

| 症状 | 优先检查 |
|---|---|
| 找不到 RK1828 | 板卡供电、模块、PCIe/USB 枚举、kernel log、驱动和设备服务 |
| Runtime 初始化失败 | Toolkit/Runtime/driver/BSP 配对、组件版本校验和目标平台 |
| 模型无法加载 | RKNN2/RKNN3 混用、模型资产缺失、target 错误、权限和路径 |
| CNN 能跑、LLM 失败 | weight、tokenizer、embed、config、内存和上下文长度 |
| 首次很慢 | 模型加载、权重加载、缓存建立和 session 初始化 |
| 单路正常、多路超时 | queue depth、transport 带宽、CPU 后处理和多 session 内存 |
| 画面颜色错误 | NV12/RGB/BGR、stride、plane、RGA color conversion |
| FPS 高但应用慢 | decode、copy、RGA、排队、后处理或显示阻塞 |
| 长时间降速 | 温度、功耗模式、CPU governor、内存压力和设备休眠 |
| 断连后不能恢复 | Runtime 资源释放、服务重启、设备重枚举和模型重新加载 |

排障顺序：

1. 复现最小官方 demo；
2. 保存完整版本和日志；
3. 只替换一个变量；
4. 验证模型资产和输入格式；
5. 再加入自己的媒体和并发层；
6. 把最终修复写成脚本或测试。

### 21.13 完成这条并行路线后的能力

#### 技术栈

| 层次 | 能力 |
|---|---|
| 底座 | Linux、C/C++、CMake、交叉编译、动态库、线程、DMA-BUF、V4L2 |
| 模型 | PyTorch、ONNX、模型导出、量化、配置和精度回归 |
| RKNN3 | RKNN3-Toolkit、RKNN3 Runtime、Model Zoo、CNN/LLM/VLM |
| 主控媒体 | RK3588、MPP、RGA、GStreamer、摄像头和视频编解码 |
| 异构系统 | RK3588 host、RK1828 coprocessor、PCIe/USB、session、队列 |
| LLM | prefill、decode、TTFT、TPOT、KV cache、tokenizer、weight |
| 产品化 | systemd、watchdog、health check、恢复、版本和性能验收 |
| 应用 | ROS2 机器人感知、视觉问答、语音交互和多路视频 |

#### 可面对的岗位

中文关键词：RKNN3 部署工程师、Rockchip NPU 工程师、AI 协处理器工程师、端侧 LLM 部署工程师、嵌入式视觉工程师、机器人感知工程师、C++ 多媒体工程师、异构计算工程师、BSP/SDK 适配工程师、车载视觉软件工程师。

英文关键词：RKNN3 Deployment Engineer、Rockchip NPU Engineer、AI Coprocessor Engineer、Edge LLM Inference Engineer、Embedded Vision Engineer、Robotics Perception Engineer、C++ Multimedia Engineer、Heterogeneous Computing Engineer、BSP/SDK Integration Engineer、Automotive Vision Software Engineer。

这条路线能证明 RK1828 + RK3588 的工程能力，但不能单独证明车规认证、功能安全、ASPICE 或量产交付经验。相关岗位还要补 QNX、ISO 26262、AUTOSAR、传感器标定、实时系统和供应链协作。

### 21.14 官方资源索引

#### RKNN3 核心

- [RKNN3 Toolkit GitHub](https://github.com/airockchip/rknn3-toolkit)
- [RKNN3 Toolkit 中文 README](https://github.com/airockchip/rknn3-toolkit/blob/main/README_CN.md)
- [RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)
- [RKNN3 Model Zoo English README](https://github.com/airockchip/rknn3-model-zoo/blob/main/README.md)
- [RKNN3 Model Zoo 中文 README](https://github.com/airockchip/rknn3-model-zoo/blob/main/README_CN.md)
- [Rockchip RKNN3 发布说明](https://www.rock-chips.com/a/cn/news/rockchip/2026/0309/2163.html)

#### RK3588 主控媒体

- RKNN2 / RKNPU2 主线文档
- [librga](https://github.com/airockchip/librga)
- [RGA English Developer Guide](https://github.com/airockchip/librga/blob/main/docs/Rockchip_Developer_Guide_RGA_EN.md)
- [Rockchip MPP](https://github.com/rockchip-linux/mpp)
- [MPP English Developer Guide](https://github.com/rockchip-linux/mpp/blob/develop/doc/Rockchip_Developer_Guide_MPP_EN.md)
- [Rockchip Linux kernel](https://github.com/rockchip-linux/kernel)

#### 板卡和高级项目

- [DFRobot RK1828 getting started](https://wiki.dfrobot.com/dfr1263/docs/24734)
- [Forlinx RK1820/RK1828 guide](https://docs.forlinx.net/ai-accelerator/rk1820_rk1828/RK1820_RK1828_AI_Accelerator_Development_Guide.html)
- [rkvoice-stream RK1828 Qwen3-TTS](https://github.com/suharvest/rkvoice-stream/blob/main/docs/rk1828-qwen3-tts.md)
- [rkvoice-stream RK1828 Gemma4](https://github.com/suharvest/rkvoice-stream/blob/main/docs/rk1828-gemma4.md)

### 21.15 最终验收清单

#### 硬件和版本

- [ ] 能画出 RK3588 + RK1828 的主控/协处理器拓扑
- [ ] 能说明目标板实际使用的 PCIe/USB/其他 transport
- [ ] 保存 BSP、kernel、driver、Toolkit、Runtime 和 Model Zoo commit
- [ ] 能用官方工具或 sample 证明 RK1828 已枚举

#### 模型和 Runtime

- [ ] 完成 ONNX -> RKNN3 model -> C++ Runtime
- [ ] 完成 MobileNet/YOLO 精度和延迟报告
- [ ] 保存 model、weight、tokenizer、embed、config 等资产
- [ ] 能识别 RKNN2/RKNN3 混用错误
- [ ] 完成至少一个 Qwen/LLM 模型
- [ ] 完成至少一个 VLM、ASR 或 TTS 专项

#### 媒体和系统

- [ ] 完成 RK3588 MPP/V4L2/RGA 到 RK1828 的视觉 pipeline
- [ ] 分离 preprocess、transfer、inference、postprocess 和 publish
- [ ] 完成单路稳定运行、多路并发和丢帧策略
- [ ] 完成超时、断连、设备重启和模型重新加载
- [ ] 完成 systemd、health check 和 watchdog

#### 跨平台

- [ ] RKNN2 本体 NPU 与 RKNN3 协处理器都有可复现实验
- [ ] 与 NVIDIA Orin 和 Horizon RDK 使用统一输入和指标
- [ ] 给出 latency、吞吐、内存、温度、功耗和交付风险矩阵
- [ ] 发布一个包含脚本、版本、日志和 benchmark 的项目仓库
