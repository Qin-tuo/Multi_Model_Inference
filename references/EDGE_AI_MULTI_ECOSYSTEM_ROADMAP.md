# 汽车 / 机器人 / 视觉端侧 AI Infra 多生态总路线

> 目标画像：端侧模型部署、推理优化和异构硬件适配，优先面向 BPU、边缘 NPU、机器人视觉和汽车视觉。
> 学习顺序：NVIDIA 主路线 -> 地瓜 Horizon BPU -> 瑞芯微 Rockchip；进入瑞芯微后，RKNN2/RKNPU2 与 RKNN3/RK1828 两条路线并行。
> 资料原则：每条路线都有公开的中英文资料、官方代码、阶段项目和验收标准；RKNN3 专题视频稀缺，因此以官方 Model Zoo 的完整代码流程配合共享 RKNN2 课程。
> 核验日期：2026-08-21。版本敏感内容以各生态当前官方兼容矩阵和目标板卡 BSP 为准。

## 0. 结论

你不需要把三套生态完全重复学习，也不需要把 Rockchip 的两条分支重复学习全部底座。正确的结构是：

1. 用 NVIDIA 路线建立 GPU、CUDA、TensorRT、profiling 和推理系统底座。
2. 用地瓜路线第一次深入国产 BPU，重点学习 OpenExplorer、PTQ/QAT、BPU Runtime、RDK 和机器人应用。
3. 用瑞芯微 RKNN2/RKNPU2 路线补齐低成本 Linux NPU、RGA、MPP、V4L2/GStreamer 和 BSP 碎片化适配。
4. 并行进入 RKNN3/RK1828 路线，学习 RK3588 主控与 AI 协处理器之间的 Runtime、通信、LLM/VLM 和多 session。
5. 最后用同一个模型、同一组输入和同一个视觉场景做跨平台对比。

最终形成的不是三个孤立的 SDK 技能，而是一个可迁移的能力模型：

<code>模型导出 -> 图检查 -> 量化 -> 编译/转换 -> Runtime -> 预处理/媒体 -> 调度 -> 性能/精度/稳定性验收</code>

## 1. 四份主文档

| 顺序 | 文档 | 生态 | 核心硬件/软件 | 最终作品 |
|---|---|---|---|---|
| 1 | [NVIDIA 主路线](EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md) | NVIDIA | CUDA、Nsight、TensorRT、Jetson、DeepStream、Triton、Edge-LLM | Orin 上的模型和多模型推理系统 |
| 2 | [地瓜 BPU 路线](EXTERNAL_FREE_BILINGUAL_HORIZON_BPU_ROADMAP.md) | Horizon / D-Robotics | OpenExplorer、PTQ/QAT、BPU Runtime、RDK、TROS、UCP | RDK 上的实时视觉和机器人感知节点 |
| 3 | [瑞芯微 RKNN2 主线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md) | Rockchip native NPU | RKNN-Toolkit2、RKNPU2、RGA、MPP、V4L2、GStreamer | RK3588 本体 NPU 上的完整硬件视频到推理 pipeline |
| 3A | [瑞芯微 RKNN3 并行线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md) | RK3588 + RK1828 | RKNN3-Toolkit、RKNN3 Runtime、RKNN3 Model Zoo、PCIe/USB、LLM/VLM | RK3588 主控到 RK1828 协处理器的异构推理系统 |

### 1.1 文档使用方式

- NVIDIA 文档是 GPU 和系统推理的主路线，不因为学习国产生态而替换。
- Horizon 文档是第一条国产 BPU 专线，优先适合你的机器人、视觉和汽车边缘目标。
- Rockchip 有两条不同路线：RKNN2 练本体 NPU 和视觉媒体，RKNN3 练 RK1828 协处理器和端侧大模型。
- 四份路线文档中的 Linux、C++、ONNX、量化、benchmark 和故障排查只学一次，后续直接迁移；工具、Runtime 和模型必须分支记录。

## 2. 为什么按这个顺序

### 2.1 先 NVIDIA

NVIDIA 路线的优势是公开资料、工具和性能分析体系完整。CUDA、Nsight、TensorRT 和 Jetson 能让你建立：

- GPU 并行执行和内存层次的底层直觉；
- 模型图、算子、精度和 engine 的关系；
- 端到端 latency、吞吐、功耗和内存的测量方法；
- 多模型并发、服务化和可观测性；
- 阅读其他 NPU 编译器和 Runtime 时所需的系统抽象。

这不代表 NVIDIA 一定是你的最终产品平台，而是先建立一套可验证的基准语言。

### 2.2 再地瓜

地瓜路线更贴近你的 BPU、机器人和智能驾驶目标：

- RDK 和 TROS 直接面向机器人开发者；
- OpenExplorer 覆盖量化、编译、仿真、部署和验证；
- X5 的公开 Model Zoo 适合先做可复现实验；
- S100/S600/J6 可以逐步延伸到更高算力和汽车异构平台；
- UCP 让你接触 BPU、DSP、ISP、VPU 等多硬件协同。

官方资料入口包括 [D-Robotics RDK 文档](https://d-robotics.github.io/rdk_doc/en/)、[RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) 和 [OpenExplorer 英文手册](https://doc.oe.horizon.auto/en/index.html)。

### 2.3 后瑞芯微

瑞芯微路线的价值不只是 RKNN：

- RK3588 把 NPU、RGA、VPU、CPU、GPU 和丰富 I/O 放在一个可获得的 Linux 平台上；
- RKNPU2 的 C/C++ API、RGA 的图像处理和 MPP 的视频链路非常适合练端到端工程；
- 板卡和 BSP 选择多，能训练你处理真实产品的版本碎片化；
- 机器人和工业视觉项目常需要低成本、多路视频和快速量产适配。

官方入口包括 [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)、[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)、[librga](https://github.com/airockchip/librga) 和 [MPP](https://github.com/rockchip-linux/mpp)。

#### Rockchip 内部两条并行线

进入瑞芯微后不要强行串成一条 SDK 路线：

| 分支 | 主硬件 | 先解决的问题 | 适合的应用 |
|---|---|---|---|
| RKNN2/RKNPU2 | RK3588 本体 NPU | 模型转换、INT8、RGA/MPP、视频 pipeline、多路视觉 | 检测、分割、姿态、IPC、机器人视觉 |
| RKNN3/RK1828 | RK3588 + RK1828 | 协处理器通信、Runtime session、模型资产、LLM/VLM、并发 | 端侧大模型、多模态、语音和异构视觉 |

两条分支共享主控 Linux 和媒体基础，但 RKNN3 的 Toolkit、Runtime、库、示例和模型转换产物必须单独学习。先用 RKNN2 完成视觉基础，再将相同的输入契约和 benchmark 方法迁移到 RKNN3。

这里的“并行”指两条独立的技术分支，不是 SDK 依赖关系。具备共享 Linux/C++/ONNX 基础后，可以直接开始 RKNN3；若按最低风险执行，先完成 RKNN2 的 R0-R4 视觉基线，再进入 RKNN3 的 bring-up 和 Model Zoo。

## 3. 共享底座和生态特有技能

### 3.1 只学一次的共享底座

| 模块 | 关键字 | 最低验收 |
|---|---|---|
| Linux | shell、SSH、权限、进程、线程、systemd、日志 | 能在 ARM64 板端排查进程、库和服务 |
| C/C++ | C++17、RAII、STL、并发、CMake、gdb | 能写可测试的 C++ Runtime wrapper |
| Python | venv、NumPy、OpenCV、PyTorch、脚本工程 | 能导出 ONNX 和自动化评测 |
| 模型 | ONNX、Netron、opset、shape、layout、输出契约 | 能定位一次导出或 shape 错误 |
| 量化 | PTQ、QAT、calibration、INT8、FP16、混合精度 | 能解释精度下降并保留回退版本 |
| 性能 | warmup、p50/p95/p99、吞吐、goodput、峰值内存 | 能把模型和端到端耗时分开 |
| 视觉 | BGR/RGB/NV12、resize、letterbox、NMS、tracker | 能让 host 和板端结果一致 |
| 工程 | Git、版本清单、hash、CI、报告和故障复现 | 别人能按文档重跑你的实验 |
| 机器人 | ROS2 node/topic/QoS/executor、时间戳 | 能把感知结果发布给应用 |
| 媒体 | V4L2、GStreamer、DMA-BUF、队列 | 能画出摄像头到推理的 buffer 流 |

### 3.2 NVIDIA 特有

CUDA C++、thread/block/grid、warp、shared memory、coalescing、Tensor Core、CUDA Graph、Nsight Systems、Nsight Compute、Compute Sanitizer、TensorRT builder、plugin、DeepStream、Triton、Edge-LLM。

### 3.3 Horizon 特有

OpenExplorer、HBDK、PTQ/QAT、hbm_runtime、Hobot DNN、.bin/.hbm、BPU core、RDK OS、TROS、UCP、BPU/DSP/GDC/ISP/PYM、J6、QNX。

### 3.4 Rockchip 特有

RKNN-Toolkit2、RKNN Lite2、RKNPU2、librknnrt、rknpu driver、.rknn、RGA/librga、MPP、V4L2、DRM/KMS、NV12、DMA-BUF、rkbin、Buildroot、Yocto。

RKNN3-Toolkit、RKNN3 Runtime、RKNN3 Model Zoo、RK1820/RK1828、RK3588 host、RKCP、librknn3_api.so、librknn3_api_rkcp.so、PCIe、USB、session、weight、tokenizer、embedding、KV cache、TTFT、TPOT、LLM/VLM。

## 4. 总时间规划

以下按每周 15 到 20 小时、已有一定 Python/AI 基础估算。全职学习可以压缩，边工作边学不要压缩验收。

| 阶段 | 时间 | 路线 | 主要结果 |
|---|---:|---|---|
| A0 | 2-3 周 | 共享底座 | Linux/C++/CMake/ONNX/Benchmark |
| A1 | 12-16 周 | NVIDIA | CUDA、TensorRT、Jetson、DeepStream、Serving |
| A2 | 10-14 周 | Horizon | OpenExplorer、BPU、RDK、ROS/TROS |
| A3 | 6-8 周 | Rockchip RKNN2 | RKNN-Toolkit2、RKNPU2、RGA、MPP、GStreamer |
| A3A | 6-8 周 | Rockchip RKNN3 | RKNN3-Toolkit、RKNN3 Runtime、RK1828、LLM/VLM、通信 |
| A4 | 6-8 周 | 跨平台 | 同模型、同输入、同场景对比 |
| 合计 | 10-15 个月 | 端侧 AI Infra | 可求职的作品集和迁移能力 |

A3A 可以在 A3 的共享基础完成后并行展开，不要求把 A3 的所有高级算子和 BSP 适配全部做完。

不要用“看完视频”作为阶段完成条件。每个阶段都要有代码、原始数据、版本和失败记录。

## 5. 阶段门槛

### Gate A：通用和 NVIDIA

必须达到：

- 能写一个正确的 CUDA kernel，并用 Nsight 找到瓶颈；
- 能从 ONNX 构建 TensorRT engine；
- 能在 Orin 上完成单模型 FP16/INT8 基线；
- 能解释预处理、模型、后处理、队列和显示各段耗时；
- 有一个可复现的 CUDA 或 TensorRT 项目。

没有达到 Gate A，不建议直接钻进多套国产 SDK 的版本细节，因为你会缺少性能和正确性判断基准。

### Gate B：Horizon

必须达到：

- 能用官方 Model Zoo 或自己的 ONNX 完成 OpenExplorer/BPU 模型；
- 能做 PTQ，并知道什么时候需要 QAT；
- 有 Python 和 C++ Runtime；
- 能处理 NV12、stride、buffer 和摄像头输入；
- 能完成 RDK 单路实时视觉；
- 有 BPU latency 与端到端 latency 分解报告。

### Gate C1：Rockchip RKNN2

必须达到：

- 能完成 ONNX -> .rknn -> Lite2/C++ Runtime；
- 能解释 RKNN Toolkit2、Runtime、driver、BSP 的版本关系；
- 能用 RGA 做硬件预处理；
- 能用 MPP/V4L2/GStreamer 接入视频；
- 能完成至少两路或多模型实验；
- 能记录 NPU、CPU、DDR、RGA、解码和编码指标。

### Gate C2：Rockchip RKNN3

必须达到：

- 能画出 RK3588 + RK1828 的主控/协处理器拓扑；
- 能用官方 RKNN3 Model Zoo 完成 ONNX -> RKNN3 Runtime；
- 能解释 RKNN3-Toolkit、Runtime、模型资产和板卡通信方式的关系；
- 能完成一个 CNN 和一个 LLM/VLM 的 C++ demo；
- 能分离 transfer、session、推理、后处理和端到端指标；
- 能完成至少一次多 session、超时或协处理器断连恢复实验。

### Gate D：跨平台

必须达到：

- 同一模型在 Orin、RDK、RK3588 本体 NPU 和 RK3588 + RK1828 上输出可比较；
- 预处理、数据集、精度指标和 warmup 方式一致；
- 不用 TOPS 直接替代真实 latency/goodput；
- 能根据成本、功耗、生态、媒体能力和交付风险给出平台选择建议。

## 6. 统一项目作品集

### Project 1：统一模型基线

选择一个轻量检测模型，建立：

- PyTorch checkpoint；
- ONNX 导出；
- 固定测试集；
- 统一预处理；
- 浮点参考输出；
- 各目标平台和 Runtime 分支的模型文件；
- 精度、延迟、内存和功耗表。

### Project 2：端侧视觉 pipeline

输入同一段视频，分别实现：

- Orin：GStreamer/DeepStream/TensorRT；
- RDK：RDK media/TROS/BPU；
- RK3588 RKNN2：MPP/RGA/RKNN/GStreamer；
- RK3588 + RK1828 RKNN3：主控媒体/RGA/RKNN3 Runtime/协处理器。

每个平台都记录：

- decode；
- preprocess；
- inference；
- postprocess；
- tracker；
- encode/display；
- queue；
- end-to-end latency。

### Project 3：多模型调度

使用高频检测、中频姿态/跟踪、低频 OCR 或语义模型，设计：

- 优先级；
- 有界队列；
- 超时；
- 丢帧；
- 内存预算；
- 温度和降频策略；
- 摄像头断流恢复；
- 服务健康检查。

### Project 4：汽车/机器人缩小版

做一个两路视觉感知系统：

- 两路相机或视频；
- 检测加分割/车道；
- 时间戳和同步；
- 结果融合；
- 设备异常和模型超时降级；
- ROS2/TROS 输出；
- 完整稳定性报告。

这能把模型部署能力提升到系统工程能力，但不能宣称拥有车规量产资质。

### Project 5：RKNN3 协处理器异构系统

基于 [RKNN3 并行路线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md) 第 21 节做一个 RK3588 + RK1828 项目：

- RK3588 负责摄像头、解码、RGA、ROS2、队列和故障恢复；
- RK1828 负责 CNN、LLM 或 VLM 推理；
- 记录主控到协处理器的传输、session、推理和后处理；
- 实现超时、断连、模型资产校验和降级；
- 用相同输入与 RKNN2 本体 NPU 做对照。

验收重点是系统分工和可观测性，不是只展示一个大模型对话窗口。

## 7. 课程执行协议

### 每个关键节点固定使用 1-2 个完整视频

| 节点 | 主视频 | 第二资料/视频 |
|---|---|---|
| CUDA/Kernels | NVIDIA Modern CUDA C++ | CUDA Developer Tools 或 GPU MODE |
| TensorRT | NVIDIA 官方中文 TensorRT 系列 | NVIDIA TensorRT workshop |
| Horizon/OE | D-Robotics OpenExplorer 50 集 | 地平线编译及编程实践 |
| Horizon/机器人 | 具身智能强化营 | 智能控制与 Sim-to-Real |
| RKNN | 迅为 RKNPU2 完整系列 | RK3588 五部分 RKNPU2 系列 |
| RKNN 量化 | RKNN 转换与评估 5 部分 | 迅为量化和精度章节 |
| RKNN3 / RK1828 | 官方 RKNN3 Model Zoo examples | RKNPU2 完整系列作为共享基础，不能复制 RKNN2 API |
| RGA/媒体 | RGA 实战视频 | MPP/GStreamer 官方英文文档 |

视频的作用是建立全貌和跟练路径，官方文档的作用是确认 API、版本和限制，项目的作用是证明能力。三者不能互相替代。

### 每节课的输出

看完一个关键节点后必须留下：

1. 一页术语和架构笔记；
2. 一个能运行的最小代码；
3. 一次主动修改；
4. 一组原始 benchmark；
5. 一个失败或排障案例；
6. 一条版本和 commit 记录。

## 8. 版本治理

### 8.1 每条路线都要有版本锁

~~~text
board:
soc:
os:
kernel:
bsp:
driver:
toolkit:
runtime:
model:
model_sha256:
compiler_flags:
input_format:
calibration_dataset:
power_mode:
temperature:
benchmark_command:
git_commit:
~~~

### 8.2 遇到版本冲突的优先级

1. 目标硬件厂商当前兼容矩阵；
2. 目标板 BSP 自带的 driver、runtime 和 sample；
3. 官方 GitHub release/tag 和 README；
4. 官方英文/中文手册；
5. 板卡厂商公开文档；
6. 社区视频和博客。

旧视频可以教方法，不能替代当前版本验证。特别注意：

- NVIDIA 的 CUDA/TensorRT/DeepStream 与 JetPack；
- Horizon 的 OE、RDK OS、BPU 模型格式和 J5/J6 代际；
- Rockchip 的 RKNN-Toolkit2 与旧 RKNN-Toolkit、RKNPU driver、librknnrt、RGA、MPP。

## 9. 你最终具备的技术栈

### 底层

Linux、C/C++、CMake、交叉编译、动态库、线程、同步、内存、DMA-BUF、V4L2、GStreamer、systemd、gdb、perf。

### 模型和编译

PyTorch、ONNX、Netron、PTQ、QAT、INT8、FP16、混合精度、算子支持、图改写、模型编译器、精度回归。

### NVIDIA

CUDA C++、Nsight、TensorRT、Jetson、DeepStream、Triton、TensorRT-Edge-LLM。

### Horizon

OpenExplorer、BPU Runtime、RDK X5/S100、TROS、UCP、J6/QNX 体系认知。

### Rockchip

RKNN-Toolkit2、RKNN Lite2、RKNPU2、RGA、MPP、RGA/GStreamer pipeline、RK3588 BSP。

### Rockchip RKNN3

RKNN3-Toolkit、RKNN3 Runtime、RKNN3 Model Zoo、RK3588 host、RK1828 coprocessor、PCIe/USB、session、LLM/VLM、KV cache、TTFT、TPOT。

### 工程能力

Benchmark、profiling、功耗/温度、稳定性、故障恢复、模型版本、服务监控、跨平台迁移和技术报告。

## 10. 对应岗位

### 适合投递的岗位

中文：端侧 AI 部署工程师、推理优化工程师、模型量化工程师、CUDA 开发工程师、BPU 部署工程师、RKNN/RKNN3 部署工程师、AI 协处理器工程师、机器人视觉工程师、嵌入式视觉工程师、车载 AI 软件工程师、异构计算工程师、BSP/SDK 适配工程师。

英文：Edge AI Engineer、Inference Optimization Engineer、Model Deployment Engineer、CUDA Software Engineer、BPU Inference Engineer、Rockchip NPU/Coprocessor Engineer、Robotics Perception Engineer、Embedded Vision Engineer、Automotive AI Software Engineer、Heterogeneous Computing Engineer、BSP/SDK Integration Engineer。

### 需要额外补充才能竞争的岗位

- 车规量产：ISO 26262、ASPICE、AUTOSAR、QNX、功能安全、安全启动、OTA；
- 机器人系统：ROS2 深入、标定、SLAM、sensor fusion、控制和实时系统；
- 编译器/算子：LLVM、MLIR、IR、lowering、CUTLASS、GPU/NPU 微架构；
- 大规模 Serving：Kubernetes、Ray、NCCL、RDMA、vLLM/SGLang、分布式调度；
- 算法研究：训练、数据、论文、模型结构和评测体系。

## 11. 当前最短路径

### 现在

继续 [NVIDIA 主路线](EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md)，从 CUDA 编程、Nsight、TensorRT 和 Orin 单模型基线开始。

### 完成 NVIDIA Gate A 后

进入 [地瓜 BPU 路线](EXTERNAL_FREE_BILINGUAL_HORIZON_BPU_ROADMAP.md)：

1. 先看 D-Robotics OpenExplorer 50 集课程；
2. 用官方 RDK Model Zoo 跑通 X5 示例；
3. 完成 MobileNet/YOLO 的 PTQ；
4. 写 Python/C++ Runtime；
5. 接摄像头和 ROS2/TROS；
6. 再读 S100、J6、UCP。

### 完成 Horizon Gate B 后

进入瑞芯微的两条并行路线，建议先完成 RKNN2 视觉基线，再进入 RKNN3 协处理器分支：

1. [RKNN2 / RKNPU2 主线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)：先用 RK3588 和官方 Model Zoo，完成 ONNX 到 .rknn、Lite2/C++、RGA、MPP/V4L2/GStreamer 和多路视觉；
2. [RKNN3 / RK1828 并行线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)：进入主文档第 21 节，用 RK3588 + RK1828 完成 RKNN3 CNN、LLM/VLM、C++ Runtime、通信、session 和恢复；
3. 用相同模型、输入和 benchmark 模板比较 RKNN2 本体 NPU 与 RKNN3 协处理器。

## 12. 官方总入口

### NVIDIA

- [NVIDIA 主路线文档](EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md)
- [CUDA Platform](https://developer.nvidia.com/cuda)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [Jetson Documentation](https://docs.nvidia.com/jetson/index.html)
- [DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)

### Horizon / D-Robotics

- [地瓜 BPU 路线文档](EXTERNAL_FREE_BILINGUAL_HORIZON_BPU_ROADMAP.md)
- [D-Robotics Developer Community](https://developer.d-robotics.cc/)
- [RDK Documentation](https://d-robotics.github.io/rdk_doc/en/)
- [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5)
- [OpenExplorer English Manual](https://doc.oe.horizon.auto/en/index.html)
- [OpenExplorer Portal](https://oe.horizon.auto/)

### Rockchip

- [瑞芯微 RKNN 路线文档](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)
- [瑞芯微 RKNN3 / RK1828 并行路线（主文档第 21 节）](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [RKNN3 Toolkit](https://github.com/airockchip/rknn3-toolkit)
- [RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)
- [RKNPU2 examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2)
- [librga](https://github.com/airockchip/librga)
- [MPP](https://github.com/rockchip-linux/mpp)

## 13. 总验收清单

### 基础

- [ ] Linux/C++/CMake/ONNX 基础可独立完成
- [ ] 能读懂并修改官方 sample
- [ ] 有统一 benchmark 和版本记录模板

### NVIDIA

- [ ] CUDA kernel 正确性和性能项目
- [ ] TensorRT FP32/FP16/INT8 项目
- [ ] Jetson 摄像头或 DeepStream 项目
- [ ] 多模型或 Edge-LLM 项目

### Horizon

- [ ] OpenExplorer PTQ 项目
- [ ] Python/C++ BPU Runtime
- [ ] RDK 摄像头实时视觉
- [ ] ROS2/TROS 感知节点
- [ ] 多模型或 UCP/J6 架构报告

### Rockchip

- [ ] ONNX -> RKNN -> Lite2/C++
- [ ] INT8 精度和算子适配
- [ ] RGA 预处理
- [ ] MPP/V4L2/GStreamer 视频 pipeline
- [ ] 多路、多模型和故障恢复

### Rockchip RKNN3

- [ ] RK3588 + RK1828 拓扑、通信和版本矩阵
- [ ] ONNX -> RKNN3 model -> C++ Runtime
- [ ] CNN 与 LLM/VLM 模型资产完整部署
- [ ] transfer、session、推理、后处理分段 benchmark
- [ ] 多 session、超时、断连和恢复
- [ ] RK3588 主控媒体到 RK1828 协处理器 pipeline

### 跨平台

- [ ] 同模型、同数据集、同输入契约
- [ ] Orin/RDK/RK3588 性能矩阵
- [ ] 成本、功耗、生态和交付风险分析
- [ ] 可打开、可编译、可复测的公开作品集

---

**最终目标不是记住多套 API，而是能在新的边缘芯片上迅速建立模型、Runtime、媒体、调度和验收闭环。**
