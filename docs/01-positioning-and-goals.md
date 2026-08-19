# 方向定位与目标

## 结论先行

对于有嵌入式软件背景、准备以 Jetson Orin NX 16GB 为主平台的人，最匹配的 AI Infra 切入点不是“纯 LLM Serving”，而是：

> **边缘异构模型推理系统与性能优化**

“异构模型”指 CV、NLP/Embedding、LLM、VLM 等模型类型不同；“异构系统”还包括 CPU、GPU、DLA、视频编解码器、统一内存和 I/O 流水线。重点是端到端部署与性能工程，不只是调用推理 API。

## 为什么适合嵌入式软件工程师

已有能力与目标方向之间有较短的迁移路径：

| 嵌入式经验 | 在推理系统中的对应能力 |
| --- | --- |
| Linux/C/C++ | TensorRT C++ runtime、内存生命周期、服务进程 |
| 交叉架构与板端部署 | x86_64 主机与 aarch64 目标边界、容器和系统包兼容性 |
| 驱动/设备/I/O 意识 | 摄像头、V4L2/GStreamer、零拷贝和 DMA buffer |
| 实时性与状态机 | SLA、优先级、超时、背压、故障恢复 |
| 日志、诊断和资源约束 | profiling、telemetry、OOM/温控/磁盘问题 |
| 多线程和 IPC | 请求队列、worker、execution context pool、服务接口 |

需要补上的核心不是重新学习一套“AI 魔法”，而是模型结构、GPU 执行模型、TensorRT 工具链和实验方法。

## 与相邻方向的边界

### 不等于纯 LLM Serving

纯 LLM Serving 常以数据中心 GPU、多卡、长上下文、极高 token 吞吐为中心，重点包括分布式并行、KV cache 分层、prefill/decode 解耦和大规模多租户。

本方向会学习其中的 KV cache、连续批处理、TTFT/TPOT 和准入控制，但 Orin NX 的主问题是 16GB 统一内存中多个不同模型与多媒体流水线如何共存。CV 的 deadline 和掉帧通常比后台 LLM 吞吐优先。

### 不等于纯模型训练或算法

需要理解模型输入输出、Attention、卷积、归一化和量化误差，但目标不是提出新网络或大规模训练。训练知识服务于导出、校准和精度回归。

### 不等于 Linux BSP/驱动开发

Linux kernel、设备树、驱动和摄像头 bring-up 是底座能力；本仓库重点在用户态推理 runtime、调度与性能。只有证据表明瓶颈位于驱动/I/O 时才下沉到 BSP。

### 不等于服务器集群平台

Kubernetes、Ray、NCCL、多节点路由和弹性扩缩容是集群平台能力。单 Orin 项目先做好进程、线程、容器和模型级调度；集群概念用来理解接口，不作为前四个月主线。

### 不等于专职 CUDA Kernel 工程师

你需要能读 profile、理解 kernel、写简单融合 kernel 和 TensorRT Plugin，但不必一开始达到手写 GEMM/FlashAttention 或维护 CUTLASS 后端的深度。详见 [CUDA Kernel、算子与 Plugin](04-cuda-kernel-and-operator-optimization.md)。

## 参考业务场景

用一个固定场景约束后续学习，避免每项技术都变成孤立 Demo：

```text
实时摄像头
  -> 解码/预处理
  -> CV 检测或分割（硬 SLA）
  -> 结构化结果

文本/传感器事件
  -> Embedding/分类/重排（突发请求）

关键帧 + 文本
  -> LLM/VLM（后台、可降级）
  -> 摘要、解释或决策建议
```

三类负载在一块 Orin NX 16GB 上共享 CPU、GPU 与内存带宽：

- CV 保持稳定帧率和 p99 延迟；
- NLP 请求在可接受 p95 内完成；
- LLM/VLM 在剩余预算内提供可接受 TTFT/TPOT；
- 过载时优先降低后台模型频率、上下文或并发，不能让实时 CV 无限制掉帧；
- 系统必须避免 OOM，并记录功耗、温度和降频。

## 项目要回答的工程问题

### 模型层

- 哪些模型能稳定导出 ONNX？
- 哪些算子由 TensorRT 原生支持，哪些需要改图或 Plugin？
- FP16、INT8、INT4 对内存、延迟和任务质量的影响是什么？
- 动态输入范围应如何约束，避免 tactic 和内存失控？

### Runtime 层

- 每个 engine 需要几个 execution context？
- 如何复用 buffer、stream 和预处理结果？
- TensorRT、DeepStream、TensorRT Edge-LLM、Triton 各自负责什么？
- 直接 TensorRT C++ 与 Triton 托管的成本和收益是什么？

### 调度层

- 请求怎样排队、超时、取消和降级？
- 模型之间怎样分配并发、内存预算和优先级？
- dynamic batching 何时提高吞吐，何时破坏实时延迟？
- LLM 的 KV cache 上限和 CV 的硬 SLA 如何同时满足？

### 系统层

- 数据搬运、CPU 前后处理、GPU kernel、I/O，谁才是瓶颈？
- 并发时性能下降来自 SM、内存带宽、显存/统一内存还是 CPU？
- 热稳态性能与刚启动时有多大差别？
- 某模型崩溃或超时后能否恢复而不拖垮其他链路？

## 能力目标

### P0：必须形成作品证据

- 在 Orin 上可复现地安装和盘点 JetPack 推理环境；
- 完成 ONNX -> TensorRT engine -> C++/Python 推理；
- 使用 `trtexec` 建立单模型 latency/throughput/memory 基线；
- 使用 Nsight Systems 和 `tegrastats` 定位端到端瓶颈；
- 理解并使用 streams、events、异步 copy、CUDA Graphs；
- 实现一个融合预处理 CUDA kernel 或 TensorRT Plugin；
- 建立两个以上模型的并发、退化和过载实验；
- 输出带原始数据、版本和精度对照的优化报告。

### P1：形成系统完整性

- DeepStream/GStreamer 实时视频 pipeline；
- TensorRT Edge-LLM 的量化 LLM/VLM；
- Triton 模型仓库、动态批处理和 Perf Analyzer；
- 调度器的优先级、背压、准入控制和内存预算；
- 1-2 小时以上热稳态与故障恢复测试；
- 容器镜像和版本 manifest 可复现。

### P2：按岗位再深入

- vLLM/SGLang/TensorRT-LLM 调度源码；
- CUTLASS/CuTe/Triton language；
- PTX/SASS 与微架构级优化；
- 多 GPU 并行、NCCL、prefill/decode 解耦；
- 编译器 IR、MLIR/TVM 和代码生成。

## 成功标准

完成这个方向，不以“安装了多少框架”为标准，而看能否提供以下证据：

1. **可复现**：另一台同版本 Orin 能按说明重建环境和运行实验。
2. **正确**：优化前后有任务质量或数值误差对照。
3. **可解释**：每项优化有 profile 证据，而不是只报告总时间变化。
4. **可承载**：并发负载满足明确 SLA，过载行为可预测。
5. **可稳定**：热稳态、内存压力和故障恢复有测试结果。
6. **可维护**：版本、配置、模型和结果之间能追踪。

## 作品集表达

最终项目简历表述应接近：

> 在 Jetson Orin NX 16GB 上构建 CV + NLP + LLM/VLM 异构推理系统；基于 TensorRT/DeepStream/TensorRT Edge-LLM 完成模型部署与量化，通过 Nsight 定位数据搬运和 kernel 瓶颈，设计优先级、背压和内存预算，在给定功耗模式下保障实时 CV p99 SLA，并用并发扫描、热稳态和故障恢复实验量化优化收益。

括号中的具体模型、延迟和收益只能使用真实 Orin 实测结果回填。

## 当前阶段输出

本知识库先产出：技术栈地图、环境安装、CUDA 学习深度、并发架构、四到五个月路线、项目验收矩阵和开源资料清单。代码实现从单模型基线开始，避免一次引入过多变量。

