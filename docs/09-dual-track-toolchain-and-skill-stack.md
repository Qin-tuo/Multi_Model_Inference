# 双主线推理工程工具链与技能栈

核验日期：2026-08-20。

本文是当前路线的统一入口，目标同时覆盖两类岗位：

1. **边缘推理性能工程**：让 CV、NLP/Embedding、LLM/VLM 在 Orin NX 16GB 上可部署、可并存、可测量、可稳定运行；
2. **推理框架研发**：理解并改进 runtime、scheduler、memory pool、KV Cache、batching、CUDA kernel 和 TensorRT Plugin。

两条路线不是二选一。它们共享模型、C++、CUDA、TensorRT、Benchmark 和 Profiling 基础，但最终作品的证据不同：边缘岗位看目标板卡上的端到端约束，框架岗位看通用机制、源码理解和可复用优化。

## 1. 你的起点与目标

### 已有优势

- 嵌入式软件工程经验，具备 C/C++、Linux、设备和工程调试基础；
- 已在 `RK_LLM`、`S100_VLA`、`RK_S100_MFSysetem` 中积累模型部署、接口、日志和多设备验证经验；
- 主验证平台明确为 NVIDIA Jetson Orin NX 16GB，RK/S100 保留独立验证边界。

### 需要补齐的能力

- CUDA 执行模型、内存层次、streams/events/graphs 和异步拷贝；
- ONNX 到 TensorRT 的 engine、context、dynamic shape、Plugin 和量化；
- Nsight Systems/Compute、`tegrastats` 和可重复 Benchmark；
- 多模型调度、队列、背压、准入控制、显存预算和故障恢复；
- vLLM/SGLang/TensorRT-LLM 的 scheduler、KV Cache、continuous batching 源码思路。

### 两类岗位的区别

| 维度 | 边缘推理性能 | 推理框架研发 |
| --- | --- | --- |
| 优化对象 | 一块受限设备上的完整链路 | 可复用的 runtime、调度和算子机制 |
| 主要问题 | 延迟、吞吐、功耗、温度、内存、稳定性 | scheduler、allocator、KV Cache、kernel、API 生命周期 |
| 主要平台 | Orin、RK、S100 等板卡 | NVIDIA GPU，必要时覆盖边缘平台 |
| 关键工具 | TensorRT、DeepStream、Nsight Systems、`tegrastats` | CUDA、Nsight Compute、Plugin、vLLM/SGLang/TRT-LLM 源码 |
| 作品证据 | Orin 实测、多模型并发、过载和 soak 报告 | runtime/Plugin/kernel 实现、源码分析和前后 benchmark |

建议把一项 Orin 多模型项目拆成两份证据：一份证明系统在板端有效，另一份证明其中的调度、内存或 CUDA 组件具有框架研发深度。

## 2. 先理解完整生命周期

模型推理优化不是单独优化一个模型文件，而是以下闭环：

```text
模型来源/训练 checkpoint
        |
        v
导出 ONNX / 图检查 / 输入契约
        |
        v
TensorRT engine / Edge-LLM 制品 / 版本 manifest
        |
        v
runtime + context + buffer + scheduler
        |
        v
请求队列 / batching / 多模型准入 / 服务 API
        |
        v
Orin 目标执行：CPU、GPU、DLA、视频引擎、统一内存
        |
        v
正确性 + Nsight + telemetry + latency/throughput/quality
        |
        +--> 定位瓶颈、修改实现、回归、稳定性和发布证据
```

### Host 与 Orin 的边界

| 工作 | Host：Ubuntu x86_64 | Orin：Jetson aarch64 |
| --- | --- | --- |
| 模型准备 | PyTorch、导出脚本、ONNX 检查、数据整理 | 只放目标运行所需的模型/engine |
| 编译 | CMake、交叉编译或容器构建、报告分析 | 原生 ARM64 编译、必要的 target build |
| 推理 | 可做工具验证，不代替板端结论 | 所有 Orin 性能、功耗、温度和稳定性结论 |
| Profiling | Nsight GUI、报告比较、归档 | `nsys`/`ncu` CLI、采集和目标进程 |
| 依赖 | x86_64 工具和缓存 | JetPack/L4T、CUDA、TensorRT、arm64 容器 |

Host 的 x86_64 结果只能作为开发辅助；简历中的 Orin 性能数字必须来自 Orin，并记录 JetPack/L4T、功耗模式、模型、精度、输入和原始结果。

## 3. 技能深度标准

不要用“安装过工具”代表掌握。所有技能按下面四级记录：

| 深度 | 能力含义 | 证据例子 |
| --- | --- | --- |
| **Use 使用** | 按文档跑通已知流程 | 能用 `trtexec` 构建一个 FP16 engine |
| **Diagnose 诊断** | 能根据日志、trace、指标定位问题 | 解释 GPU 空洞来自同步还是 H2D 拷贝 |
| **Optimize 优化** | 在正确性约束下做前后对照 | Plugin 或调度修改带来可重复收益 |
| **Design/Contribute 设计/贡献** | 能形成可复用机制或 upstream-quality patch | 设计 context pool、提交修复并附测试 |

第一个求职里程碑的要求：大多数 P0 能达到 Diagnose；C++、TensorRT、Profiling、CUDA 基础和并发至少达到 Optimize；只选择一两个组件尝试 Design/Contribute。

## 4. 共享基础工具链

这些能力同时服务两个岗位，是前 4-5 个月的主干。

| 能力 | 工具/概念 | 目标深度 | 练习与作品证据 |
| --- | --- | --- | --- |
| Linux/C++ 工程 | C++17、RAII、线程、锁、条件变量、进程、动态库、错误处理 | Optimize | 写异步 inference worker、context pool 和可取消任务；附单元测试与崩溃日志 |
| 构建与依赖 | GCC/Clang、CMake、Ninja、`compile_commands.json`、Git、可选 ccache | Diagnose | 一条干净命令重建 runner/Plugin；记录 compiler、flags、依赖 revision |
| 调试与系统诊断 | GDB、core dump、`ldd`、`readelf`、`strace`、`perf` | Diagnose | 对一次崩溃、动态库冲突或 CPU 等待形成诊断报告 |
| C++ 正确性 | GoogleTest、ASan、UBSan、TSan（按代码适用） | Optimize | 为 queue、buffer lifetime、并发取消和边界输入编写回归 |
| Python 自动化 | venv、pip、PyTorch、pytest、JSON/YAML/CSV | Diagnose | 自动导出模型、生成输入、运行矩阵、汇总报告；不污染系统 Python |
| 模型基础 | Conv、GEMM、Attention、Norm、tokenizer、prefill/decode、KV Cache | Diagnose | 给 CV、Embedding、LLM 各画执行和内存图，写出瓶颈假设 |
| 图与输入契约 | ONNX、opset、dynamic shape、shape inference、Netron、ONNX Runtime | Optimize | 导出并校验 ONNX；记录 unsupported op、shape range 和 reference output |
| TensorRT 基础 | builder、tactic、engine、execution context、optimization profile、`trtexec` | Optimize | FP32/FP16 engine 可重建；比较 layer、GPU compute、端到端延迟和内存 |
| Benchmark 方法 | warm-up、固定输入、p50/p95/p99、吞吐、goodput、重复、热稳态 | Optimize | 场景 manifest、原始 CSV/JSON、分析脚本和结论；不只截平均值 |
| 可观测性 | 结构化日志、request ID、阶段计时、NVTX、`tegrastats` | Diagnose | 将 queue/preprocess/enqueue/postprocess 与设备资源放进同一时间轴 |

### 共享工具的使用顺序

```text
模型/输入契约
  -> ONNX 检查
  -> TensorRT 单模型基线
  -> tegrastats + 端到端指标
  -> Nsight Systems 时间线
  -> 定位关键层或 kernel
  -> Nsight Compute / Plugin / scheduler 优化
```

## 5. 边缘推理性能主线

### 必须掌握的能力

| 能力 | 重点工具 | 实际任务 | 求职证据 |
| --- | --- | --- | --- |
| JetPack 基线 | SDK Manager、L4T、CUDA、TensorRT、cuDNN/cuBLAS | 记录版本、功耗模式、存储、散热、设备节点 | 可复现 version manifest 和 smoke checklist |
| CV runtime | TensorRT C++/Python、`trtexec` | 一个检测/分割模型的 FP16、dynamic shape、预处理和后处理 | Orin 端到端 latency/throughput/quality 表 |
| 实时视频 | DeepStream、GStreamer、`nvinfer`、buffer/caps/metadata | 解码、batch mux、推理、后处理、丢帧策略 | FPS、p99、drop、温度、内存、功耗和 30 分钟 soak |
| CUDA 实用编程 | kernel、stream、event、pinned/device memory、异步 copy、CUDA Graph | 实现 resize + normalize + layout conversion 融合 kernel | CPU/GPU 对照、正确性、Nsight 时间线和停止优化理由 |
| 量化 | FP16、INT8、INT4、校准、质量阈值 | 对同一模型建立精度、内存、延迟矩阵 | 原始数据、校准配置、质量回归和可接受边界 |
| 端侧 LLM/VLM | TensorRT Edge-LLM、prefill/decode、TTFT、TPOT、tokens/s | 选择在 Orin 支持矩阵内的小模型，扫描 context、并发和 KV 预算 | model/engine hash、KV 容量、OOM 边界、质量和热稳态报告 |
| 多模型资源 | queue、priority、deadline、backpressure、admission control | CV 硬 SLA + NLP/LLM 后台负载，过载时 reject/degrade | 隔离/并发/饱和曲线和降级记录 |
| 服务交付 | HTTP/gRPC、健康检查、超时、取消、结构化错误 | 把 runner 变成可压测本地服务 | API 契约、错误分类、请求指标和恢复行为 |
| ARM64 容器 | Docker/NGC、NVIDIA Container Runtime、bind mount、digest | 固化模型依赖并保持 `linux/arm64` 与 JetPack 匹配 | image digest、启动命令、清洁环境复现记录 |

Triton 在 Orin 上作为成熟 serving 对照使用，必须先核验 JetPack/DeepStream 对应组合；不能把普通 dGPU release 的 backend 兼容性直接推导到 Orin。

### 边缘岗位最低作品包

```text
Orin 单 CV baseline
  + tegrastats/Nsight Systems telemetry
  + 一个真实 CUDA kernel 或 TensorRT Plugin
  + CV + NLP/Embedding + 小型 LLM/VLM 的 bounded scheduler
  + p95/p99、内存、功耗、温度、过载和 soak 报告
  + clean-room 复现指南
```

## 6. 推理框架研发主线

### 必须掌握的能力

| 能力 | 重点概念/代码 | 实际任务 | 求职证据 |
| --- | --- | --- | --- |
| Runtime 生命周期 | model load、engine、context、buffer、stream、同步/取消 | 设计 context/buffer pool，处理异常和释放顺序 | C++ 组件、测试和生命周期图 |
| CUDA 执行 | grid/block、memory hierarchy、stream/event/graph、launch overhead | 写一个带 CPU reference 的融合 kernel | 随机/边界正确性、compute-sanitizer、前后 profile |
| TensorRT Plugin | Plugin V3、shape/type、workspace、serialization、enqueue | 将一个 unsupported 或低效算子接入 engine | Plugin 源码、engine rebuild、并发和序列化回归 |
| GPU 性能 | Nsight Systems + Nsight Compute、occupancy、bandwidth、warp stall | Systems 找热点，Compute 定点分析 | `.nsys-rep`/`.ncu-rep` 与瓶颈解释 |
| Scheduler | request lifecycle、队列、priority、deadline、fairness、cancel | 实现小型 bounded scheduler 并对比 FIFO | 状态图、饱和扫描、p95/goodput 和过载行为 |
| Batching | static、dynamic、continuous/in-flight batching、padding/bucketing | 扫 batch size、queue delay、到达率，记录收益和代价 | batch 参数矩阵与退化边界 |
| LLM memory | KV Cache、block/page、prefix reuse、context 生命周期 | 估算并验证 KV 容量，限制 max tokens/concurrency | KV 预算、回收策略、OOM/质量回归 |
| 框架源码 | vLLM/SGLang/TRT-LLM scheduler、worker、cache、benchmark | architecture -> request -> scheduler -> cache -> worker 阅读闭环 | 源码笔记必须对应一个小实验，不只摘抄架构图 |
| Kernel library | Triton language、CUTLASS/CuTe、GEMM/Attention tile/layout | 在 Nsight 证明原生 kernel 是瓶颈后再选一个局部实验 | 参数、SM 目标、正确性和 benchmark；不盲目移植服务器样例 |
| 工程贡献 | issue 复现、最小 patch、单元/benchmark、文档 | 给官方样例或工具提交文档、脚本或小修复 | PR、复现仓库或可审查的 patch |

### 框架岗位最低作品包

```text
一个可测试的 C++ inference runtime/worker
  + 一个 CUDA 融合 kernel 或 TensorRT Plugin
  + 一个 bounded scheduler / context pool
  + KV Cache、batching 或 allocator 的源码分析与实验
  + Nsight 前后对照和 correctness regression
  + 至少一个开源项目级别的复现、文档修复或小 patch
```

不需要从零重写 vLLM 或 TensorRT-LLM。关键是能从源码中提取机制，做出可运行的缩小版，并用实验说明取舍。

## 7. Nsight：Host 看界面，Orin 做采集

### 工作模型

```text
Host x86_64
  nsys-ui / ncu UI / report archive
          │ SSH remote launch 或复制 report
          ▼
Orin aarch64
  nsys/ncu target CLI -> 目标程序 -> CPU/CUDA/GPU trace
```

Nsight Systems 负责系统时间线，Nsight Compute 负责热点 kernel。两者都不能替代 `tegrastats`：Nsight 解释事件和瓶颈，`tegrastats` 补充 Orin 的频率、温度、功耗和内存状态。

### GUI 远程采集

**Host：**

```bash
ssh <orin-user>@<orin-ip> 'uname -m; nsys --version; nsys status -e'
nsys-ui
```

在 GUI 中添加 Tegra/Jetson SSH 连接，目标程序路径、工作目录、环境变量都填写 Orin 上的路径。Host 负责启动控制和查看报告，采集过程发生在 Orin。Host 和 target 的 Nsight/JetPack 版本应保持兼容。

### CLI 采集与复制

**Orin：**

```bash
mkdir -p ~/profiles
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=~/profiles/multi_model \
  --force-overwrite=true \
  ./your_application
```

**Host：**

```bash
scp <orin-user>@<orin-ip>:~/profiles/multi_model.nsys-rep .
nsys-ui multi_model.nsys-rep
```

如果程序在容器中，确认容器包含目标架构和 profiling CLI，或在 Orin 主机上按官方容器说明收集；不要从 x86 容器复制 profiler 二进制到 ARM64 容器。

### 分析顺序

1. Systems 看 CPU 等待、队列、H2D/D2H、stream overlap、kernel 空洞和请求阶段；
2. TensorRT profile 看具体 layer/tactic 与 engine/context 行为；
3. Compute 只对已经确认的热点 kernel 收集 occupancy、memory throughput、cache 和 warp stall；
4. 修改一个变量，重复 correctness、latency、throughput、功耗和热稳态测试。

### 最小 Orin 采集检查

**Orin：**

```bash
nsys --version
nsys status -e
ncu --version
tegrastats --interval 1000
```

权限、驱动、target package 或报告版本不匹配时先修环境；空报告不能作为“kernel 没问题”的证据。

## 8. 20 周双主线路线

每一阶段使用同一个版本 manifest、模型 manifest、场景 manifest 和原始结果目录，同时交付两类成果。

| 时间 | 共享基础 | 边缘交付 | 框架交付 | 退出条件 |
| --- | --- | --- | --- | --- |
| 第 1 周 | JetPack、架构、CUDA/TensorRT smoke、实验规范 | Orin baseline/version manifest | CMake/C++ runner 能重建 | 重启后环境可用、证据目录固定 |
| 第 2-4 周 | PyTorch -> ONNX -> TensorRT、FP32/FP16、Benchmark | 单 CV engine 的端到端基线 | engine/context/buffer 生命周期图和 C++ runner | 正确性、p95、内存和命令可复现 |
| 第 5-7 周 | NVTX、Nsight Systems、CUDA streams/events/memory | 预处理、enqueue、后处理时间线 | 一个安全的融合 kernel + Compute Sanitizer | 能解释瓶颈，优化不破坏结果 |
| 第 8-10 周 | Plugin、DeepStream/GStreamer、测试 | 实时视频、drop/latest-frame、GPU/DLA 对照 | Plugin V3 serialization/shape/enqueue 回归 | 30 分钟运行无无界增长 |
| 第 11-14 周 | 量化、Edge-LLM、TTFT/TPOT、KV 预算 | 小型 LLM/VLM 的精度/内存/热稳态矩阵 | KV Cache/batching 源码对照 + 小实验 | OOM 边界和质量阈值明确 |
| 第 15-18 周 | 服务 API、队列、压测、指标 | CV + NLP + LLM 有界并发、降级和 soak | scheduler/context pool、FIFO vs priority 对照 | 过载行为可控、结果可复现 |
| 第 19-20 周 | clean-room、故障注入、文档和简历 | Orin 项目验收包 | patch/源码报告/框架岗位版本简历 | 每个数字可追溯到原始结果 |

前 20 周不把 Kubernetes、多节点并行、完整 CUTLASS kernel、PTX/SASS 或分布式 KV Cache 作为主线。它们在共享基础和一个可验证的单机系统之后按岗位要求补充。

## 9. JD 关键词与证据

| JD 关键词 | 必须能解释 | 项目证据 |
| --- | --- | --- |
| 性能分析/优化 | 如何从 trace 到瓶颈假设，再到可重复收益 | Nsight report、前后对照、停止优化理由 |
| C++/runtime | 生命周期、线程、同步、错误和资源释放 | worker/context pool、测试、崩溃恢复 |
| CUDA/operator | 访存、并发、launch、正确性和 kernel 边界 | 融合 kernel 或 Plugin、Compute Sanitizer、benchmark |
| TensorRT/ONNX | shape、engine、context、tactic、unsupported op | ONNX export、engine build、quality/latency 表 |
| batching/concurrency | queue delay、batch size、priority、backpressure、admission | 饱和扫描、p95/p99、reject/degrade 记录 |
| KV Cache/LLM | prefill/decode、TTFT/TPOT、context 和 cache 容量 | KV 预算、max tokens/concurrency、质量回归 |
| serving/高可用 | API、健康检查、超时、取消、故障边界 | 服务契约、错误分类、soak/fault report |
| 测试设计 | correctness、性能回归、热稳态、清洁复现 | manifest、原始数据、脚本和复现指南 |

### 不应作为主证据的内容

- 只列出安装过的框架或工具，没有实验和诊断；
- 只报告平均延迟，没有输入条件、p95/p99、warm-up 和重复次数；
- 只测模型 enqueue，不测预处理、排队、后处理和拷贝；
- 只说 INT4/INT8 更快，不做质量、内存和 OOM 边界回归；
- 只在服务器 GPU 上得到数字，却把它写成 Orin 结果；
- 一开始就读所有框架源码或手写复杂 kernel，却没有一个稳定 baseline。

## 10. 仓库阅读顺序

本地仓库分类见 [third_party/README.md](../third_party/README.md)。建议顺序：

1. `cuda-samples`、`TensorRT`：建立 CUDA/TensorRT 编译和运行基线；
2. `DeepStream`、`deepstream_reference_apps`：实时 CV 与历史多路参考；
3. `TensorRT-Edge-LLM`、`Model-Optimizer`：Orin LLM/VLM、量化和 KV 相关路径；
4. `triton-server`、`triton-tutorials`、`perf_analyzer`、`model_analyzer`：Serving 和压测对照；
5. `vllm`、`TensorRT-LLM`、`sglang`：scheduler、KV Cache、batching 源码对照；
6. `cutlass`、`CUDALibrarySamples`：在 Nsight 证明 kernel 是主要瓶颈后再深入。

主线仓库的 `main` 不等于 Orin 稳定 release。实验必须固定与 JetPack/L4T 匹配的版本；服务器框架源码主要用于机制学习，不能直接推导 Jetson 支持。

## 11. 何时算达到可投递水平

两类岗位都投递时，最低证据组合是：

- 一份可清洁重建的 Orin 单模型和多模型 benchmark；
- 一份带 Nsight/`tegrastats` 的端到端瓶颈报告；
- 一个经过 correctness、sanitizer 和 benchmark 的 CUDA kernel 或 TensorRT Plugin；
- 一个带 bounded queue、priority、backpressure 和 admission control 的 scheduler；
- 一份 Edge-LLM 的精度、KV、TTFT/TPOT 和 OOM 边界报告；
- 一份 vLLM/SGLang/TensorRT-LLM 源码机制与小实验对照；
- 至少一个可审查的文档、复现脚本或小型开源 patch。

这套证据可以分别包装为“Orin 多模型推理性能优化项目”和“推理 runtime/scheduler/CUDA 优化项目”，而不是把所有技术名词堆在一份简历里。
