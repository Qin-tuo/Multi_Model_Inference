# CUDA Kernel、模型算子与 TensorRT Plugin

## 一句话回答

这里说的 **kernel 通常是 CUDA kernel**，即在 NVIDIA GPU 上并行执行的函数；Linux kernel 是管理整个操作系统的内核。它们同名，但职责、运行位置、开发方式都不同。

对于你的方向，CUDA 编程需要学，而且要学到能 profile、能解释并发、能写一个真实融合 kernel、能开发 TensorRT Plugin 的程度。前期不要求手写 GEMM/FlashAttention 或深入 PTX/SASS。

## 五个容易混淆的概念

| 名称 | 是什么 | 在哪里运行/存在 | 例子 |
| --- | --- | --- | --- |
| Linux kernel | 操作系统核心，管理进程、虚拟内存、驱动、文件系统、网络 | CPU 的特权态 | scheduler、V4L2 driver、memory reclaim |
| CUDA kernel | 由大量 GPU threads 并行执行的函数 | NVIDIA GPU | resize、normalize、NMS、GEMM 的底层实现 |
| 模型算子（operator/op） | 计算图中的语义节点 | PyTorch/ONNX/编译器 IR | Conv、MatMul、LayerNorm、Attention |
| TensorRT layer | TensorRT network 中的层表示 | build-time network / optimized engine | `IConvolutionLayer`、`IAttention` |
| TensorRT Plugin | TensorRT 不原生支持或需要自定义实现的 layer 扩展 | build + runtime；内部通常启动 CUDA kernel | 自定义 deformable op、融合预处理 |

关系可以是：

```text
ONNX LayerNorm operator
  -> TensorRT 识别并融合
  -> engine 中选择一个或多个 CUDA kernels

自定义 ONNX operator
  -> TensorRT 无原生实现
  -> TensorRT Plugin V3
  -> plugin enqueue() 启动自写 CUDA kernel
```

一个模型 operator 不一定只对应一个 CUDA kernel；多个 operator 也可能被 TensorRT 融合成一个 kernel。Nsight 看到的 kernel 名称不能简单按 ONNX 节点一一对应。

## Linux kernel 与 CUDA kernel 的详细区别

| 维度 | Linux kernel | CUDA kernel |
| --- | --- | --- |
| 目标 | 管理系统与硬件资源 | 并行完成某段数值/数据处理 |
| 执行处理器 | CPU | GPU SM 上的 CUDA cores/Tensor Cores 相关执行单元 |
| 权限 | kernel mode | GPU device code，由 Host runtime/driver 调度 |
| 调用方式 | syscall、interrupt、driver interface | `kernel<<<grid, block, stream>>>(...)` 或库/runtime 间接启动 |
| 并发单位 | process/thread/interrupt | grid/block/warp/thread |
| 内存视角 | virtual memory、page、VFS、DMA mapping | global/shared/local/constant/register 等层次 |
| 常用调试 | ftrace、perf、eBPF、dmesg、crash | Nsight Systems、Nsight Compute、compute-sanitizer |
| 典型工作 | driver、scheduler、network stack | preprocess、reduction、attention、plugin compute |

二者会在 GPU driver、内存分配、DMA buffer 和同步处相遇。例如应用向 CUDA runtime 提交 GPU 工作，用户态 driver 与 Linux 内核驱动协作完成资源和命令管理，但实际数值计算由 CUDA kernel 在 GPU 上执行。

## OpenAI Triton 又是什么

OpenAI Triton 是用 Python 风格 DSL 编写 GPU kernel 的编译工具；NVIDIA Triton Inference Server 是模型服务系统。两者不要混淆：

| Triton 名称 | 解决问题 | 当前优先级 |
| --- | --- | --- |
| NVIDIA Triton Inference Server | 模型管理、请求、batching、metrics | P1，服务与并发实验 |
| OpenAI Triton language | 编写/编译 GPU kernel | P2，已有 CUDA 基础后学习 |

对 Orin 项目，先学 CUDA C++ 和 TensorRT Plugin，能更直接理解 JetPack 工具链与 C++ runtime。之后用 Triton language 对比表达和自动调优。

## CUDA 执行模型最低知识

### Host 与 Device

- Host code 在 Arm CPU 上运行；
- device code 在 GPU 上运行；
- Host 负责准备数据、选择 stream、启动 kernel、处理结果；
- kernel launch 通常是异步的，错误可能在之后的同步点才暴露。

### Grid、Block、Warp、Thread

- 一个 kernel launch 形成 grid；
- grid 包含多个 blocks；
- block 包含 threads，并可共享 shared memory；
- GPU 以 warp 为重要执行/调度单位；
- 分支发散、访存不合并、寄存器过高都可能降低效率。

不需要先背所有硬件参数，但要会用设备查询和 Nsight 解释 launch configuration 是否合理。

### 内存层次

| 层次 | 特征 | 常见问题 |
| --- | --- | --- |
| Register | 每线程、最快、容量有限 | 使用过多导致 occupancy/溢出问题 |
| Shared memory | 每 block、可协作复用 | bank conflict、容量限制、同步错误 |
| L1/L2 cache | 硬件缓存 | 数据布局与复用差导致 miss |
| Global/DRAM | 容量大、延迟和带宽关键 | 非合并访问、多次读写中间 tensor |
| Constant/read-only | 特定访问模式有优势 | 不适合大规模可变数据 |

在 Orin 统一内存系统中，“CPU/GPU 共享物理内存”并没有消除访问模式和 DRAM 带宽成本。模型并发时，多个 engine 和视频流水线仍会争抢 DRAM。

## 你需要达到的 CUDA 学习深度

### 阶段 1：能看懂执行时间线

掌握：

- CUDA API 与 kernel launch；
- 同步/异步的差别；
- default stream 与显式 stream；
- `cudaMemcpyAsync` 的前提；
- event 计时与 CPU wall-clock 的差别；
- warm-up、lazy initialization 和同步点。

验收：能解释一次 TensorRT inference 的 CPU enqueue、memcpy、kernel、同步和响应链路。

### 阶段 2：能写正确的小 kernel

练习：vector add 只用于语法；真正作品使用图像 resize + normalize + HWC->CHW 或一个模型后处理。

掌握：

- indexing 和边界；
- grid/block 选择；
- coalesced memory access；
- 基本 shared memory；
- CUDA error checking；
- CPU reference 与数值误差测试；
- compute-sanitizer。

验收：不仅“结果看起来对”，还要对随机形状、边界尺寸和误差阈值自动检查。

### 阶段 3：能构建异步流水线

掌握：

- 每个模型/请求的 stream；
- events 建立依赖，不用全局同步；
- buffer pool 和固定生命周期；
- double/triple buffering；
- CUDA Graphs 降低重复 launch 的 CPU overhead；
- 多 stream 并不保证并行，受 SM/带宽/engine 资源限制。

验收：用 Nsight Systems 证明消除了不必要的空洞或全局同步，并报告并发退化。

### 阶段 4：能写 TensorRT Plugin V3

掌握：

- plugin creator/registry；
- build-time shape/type negotiation；
- serialization/version；
- workspace 和 alignment；
- `enqueue()` 使用 TensorRT 提供的 CUDA stream；
- dynamic shape；
- plugin namespace/version 与 engine compatibility；
- 单元正确性和 engine serialize/deserialize 测试。

验收：一个真实 unsupported/fusion 场景通过 Plugin 进入 TensorRT，而不是写一个无业务意义的加法层。

### 阶段 5：能用 Nsight Compute 优化热点

掌握：

- kernel duration 和 launch count；
- memory throughput 与访问效率；
- achieved occupancy 及其限制因素；
- warp stall 原因；
- roofline 的 compute-bound/memory-bound 判断；
- 优化前后同一输入、同一频率/功耗模式对照。

验收：指出瓶颈证据，提出一个假设，只改变一个关键变量，并验证正确性与性能。

### 暂缓内容

- 手写高性能 GEMM；
- FlashAttention 全实现；
- CUTLASS/CuTe 深度模板开发；
- PTX/SASS 指令级调优；
- 自研编译器/codegen；
- 为每个模型 operator 都写 custom kernel。

这些是 kernel/library/编译器岗位的进阶，不是多模型推理项目的起点。

## Streams、Contexts 与并发

### CUDA stream

stream 是有序 GPU 工作队列。同一 stream 中操作有顺序，不同 stream 的工作在资源允许时可能重叠。

### TensorRT execution context

engine 是优化后的模型；execution context 保存一次执行所需的动态状态。跨请求并发通常需要：

- 每个并发执行使用独立 context；
- context 对应合法 optimization profile；
- 每个 context 的 tensor address、shape 和 stream 生命周期正确；
- 评估多个 context 带来的 activation/device memory 增长。

不要让多个线程在未确认线程安全语义时共享一个 context。

### TensorRT auxiliary streams

TensorRT 可在单次 inference 内使用 auxiliary streams，也可通过多个 context/stream 做 cross-inference concurrency。更多 stream 会增加同步和 activation memory；builder 在空闲 GPU 上选的 tactic 在并发争用下未必最佳，因此必须在目标并发条件下压测。

### CUDA Graphs

适合形状和执行序列稳定、重复 launch overhead 明显的 workload。它减少 CPU launch 开销，不会自动减少 kernel 计算量，也不会解决内存带宽饱和。

## 正确的优化顺序

```text
定义 SLA 和正确性
  -> 建立可重复 baseline
  -> Nsight Systems 定位端到端瓶颈
  -> 消除多余 copy / sync / queue delay
  -> 调整 TensorRT shape profile / precision / tactic / context
  -> 使用成熟库和 TensorRT fusion
  -> Nsight Compute 分析确认的热点 kernel
  -> 仅对缺口写 Plugin / custom kernel
  -> 重测正确性、p95/p99、内存、功耗和热稳态
```

手写 kernel 通常排在后面，因为前处理串行、shape 范围过宽、队列策略或同步错误可能比单 kernel 慢得更多。

## 第一项推荐 CUDA 作品：融合图像预处理

### Baseline

CPU/OpenCV 或多个 GPU 操作完成：

1. resize；
2. color conversion；
3. normalize；
4. HWC -> CHW；
5. cast FP16/FP32。

### 自定义 kernel

将可安全融合的步骤合并，直接写入 TensorRT input buffer。

### 正确性

- 多种分辨率、奇数宽高、边界像素；
- 与 CPU reference 比较 max/mean error；
- FP16/FP32 分别设阈值；
- compute-sanitizer 无越界/race；
- 检查 color/channel/order。

### 性能

- 单次 latency 与 p95；
- DRAM read/write；
- kernel launch count；
- CPU 占用；
- CV 端到端 p99；
- 与其他模型并发时的退化。

如果融合 kernel 自身更快但增加了 pipeline 同步，最终端到端可能没有收益，所以必须测完整链路。

## 第一项推荐 Plugin 作品

选择模型中一个真实问题：

- ONNX parser 不支持的 custom op；
- 可融合但框架未融合的 preprocess/postprocess；
- 现有实现产生多次小 kernel launch；
- dynamic shape 下的特定算子路径。

Plugin 项目必须包含：

1. CPU 或 PyTorch reference；
2. 独立 CUDA kernel test；
3. TensorRT network/plugin test；
4. engine serialize/deserialize；
5. 多 shape/profile；
6. 并发 context；
7. 性能与数值误差报告；
8. plugin version/namespace 记录。

## Nsight 工具分工

| 工具 | 回答的问题 | 不该用来做什么 |
| --- | --- | --- |
| Nsight Systems | 时间花在哪、CPU/GPU 是否重叠、哪里同步/排队 | 直接判断单 kernel 每个 stall 的根因 |
| Nsight Compute | 某个 CUDA kernel 的访存、占用、warp stall | 长时间全应用无差别采集 |
| `trtexec` | TensorRT engine baseline、shape/precision/concurrency 参数 | 代替完整业务端到端测试 |
| `tegrastats` | 系统内存、频率、功耗/温度趋势 | 精确分解某模型的 GPU kernel 时间 |
| `perf`/eBPF | CPU、syscall、scheduler、I/O | CUDA kernel 内部指标 |
| compute-sanitizer | memory/race/API correctness | 证明性能最优 |

## 常见反模式

- kernel launch 后每次都 `cudaDeviceSynchronize()`；
- 用 pageable host buffer 却期望 copy 与 compute 重叠；
- 只用 CUDA event 测 GPU 段，却声称是 API 端到端延迟；
- 只看 occupancy，忽略实际 kernel duration 和带宽；
- 因为 kernel 很短就认为不值得优化，却忽略每帧启动数百次；
- 手写实现只测固定 shape，生产输入发生动态变化；
- Plugin 忽略 TensorRT 提供的 stream，破坏并发；
- 并发模型共享 buffer/context 导致隐蔽数据竞争；
- 在未锁定功耗/温度条件时比较微秒级差异。

## 学完的判断标准

你不需要靠背 API 证明“会 CUDA”。能完成下面任务即可进入项目主线：

- 从 Nsight Systems 解释一次完整 inference；
- 区分 CPU queue、CUDA API、memcpy、CUDA kernel 和同步时间；
- 写一个正确且有实际输入的融合 kernel；
- 用 Nsight Compute 给出访存或执行瓶颈证据；
- 将一个真实 custom op 封装为 TensorRT Plugin V3；
- 在单模型和并发模型下报告前后结果；
- 知道何时应继续用 TensorRT/库，而不是手写算子。

这就是边缘推理性能工程师所需的 CUDA 合格线。

