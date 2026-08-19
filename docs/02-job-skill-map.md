# 岗位 JD 与技术栈映射

核验日期：2026-08-19。

## 样本岗位

本分析不是根据某一个“大模型部署”标题猜测，而是对照三类真实岗位：

1. [NVIDIA - AI Computing Software Development Engineer, LLM Inference](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Computing-Software-Development-Engineer--LLM-Inference_JR2019592)：强调 TensorRT-LLM/Edge-LLM、多平台推理软件、C/C++、性能分析和测试设计。
2. [百度 - 推理性能优化工程师](https://talent.baidu.com/jobs/detail/SOCIAL/14c8d37a-3288-4280-8fbf-71a878b010eb)：强调多模态 LLM/Diffusion、vLLM/SGLang/TensorRT-LLM、GPU 利用率、服务化、高并发和高可用。
3. [昆仑芯 - 模型加速与部署工程师](https://kunlunxin.zhiye.com/xiangqing?jobId=151141586)：同时覆盖 CV、Diffusion、LLM，明确要求 TensorRT/ONNX Runtime、CUDA Kernel、Plugin、量化、Nsight、Serving 和 batching。

这些 JD 共同说明：推理优化岗位不是“会调用 vLLM”即可，而是模型、runtime、kernel、服务与系统性能的交叉工程。

## 三类岗位不是同一个深度

| 岗位族 | 核心产出 | 高频技术 | 与你的匹配度 |
| --- | --- | --- | --- |
| 边缘部署与性能 | 单机/板端模型落地、端到端延迟、资源和稳定性 | C++、TensorRT、DeepStream、CUDA、Linux、容器 | **最高，当前主线** |
| 推理框架研发 | runtime、调度器、算子、内存管理和框架功能 | C++/Python、CUDA、PyTorch、vLLM/SGLang/TRT-LLM | 中高，可由边缘项目逐步深入 |
| 集群推理平台 | 多 GPU/多节点服务、资源调度、SLO、容量和高可用 | Kubernetes、NCCL、Ray/Dynamo、可观测性、存储/网络 | 中低，先理解概念，后续按岗位扩展 |

Orin 项目能直接证明第一类能力，也能覆盖第二类的缩小版问题：execution context、KV cache、队列、batching、内存预算和 CUDA kernel。它无法单独证明多机并行和大规模平台经验。

## 技术栈全景

### P0：必须掌握并产生作品证据

| 能力 | 需要达到的深度 | 为什么重要 | 学习任务 | 作品证据 |
| --- | --- | --- | --- | --- |
| Linux + C++17 | RAII、线程/同步、进程、动态库、CMake、日志、错误处理 | runtime 和 DeepStream 的主路径是 C/C++ | 写异步 inference worker 和 context pool | 可编译程序、测试、崩溃恢复记录 |
| Python | 模型导出、数据准备、校准、自动化分析 | 连接 PyTorch/ONNX 与部署工具 | 导出 ONNX、生成数据集和报告 | 固定依赖、脚本输入输出清晰 |
| 模型基础 | 理解 Conv、GEMM、Attention、Norm、tokenization、KV cache | 才能解释 profile 和量化误差 | 为 CV、Embedding、LLM 各画一张执行/内存图 | 模型 manifest 与瓶颈假设 |
| ONNX | opset、dynamic shape、shape inference、图检查 | TensorRT 常见交换格式 | 导出、simplify/检查、定位 unsupported op | 可重复生成并校验 ONNX |
| TensorRT | builder/runtime、optimization profile、tactic、engine/context、Plugin V3 | Orin 通用模型主推理引擎 | `trtexec` 基线 + C++ enqueueV3 | latency/throughput/memory/精度报告 |
| CUDA 基础 | kernel、thread hierarchy、memory hierarchy、streams/events/graphs、异步 copy | 判断并发、同步和访存瓶颈 | CUDA samples + 融合预处理 kernel | Nsight 前后对照和正确性测试 |
| Profiling | Nsight Systems 看时间线，Nsight Compute 看 kernel，`tegrastats` 看系统 | 优化必须有证据 | 建立 CPU/GPU/内存/功耗统一时间轴 | trace、关键截图、瓶颈解释 |
| 量化 | FP16/INT8/INT4、校准、weight-only、任务质量回归 | 16GB 内存下收益直接 | 对同一模型做精度/内存/延迟矩阵 | 原始数据和可接受阈值 |
| 并发调度 | 队列、优先级、deadline、backpressure、admission control | 多模型共存的核心 | CV 硬 SLA + 后台 LLM 调度 | 饱和扫描和过载行为 |
| Benchmark | warm-up、percentile、吞吐/goodput、控制变量、重复性 | 避免“跑通即优化” | 隔离/并发/热稳态实验矩阵 | CSV/JSON 原始结果和报告 |

### P1：完成项目系统性

| 能力 | 目标深度 | Orin 项目中的位置 |
| --- | --- | --- |
| DeepStream/GStreamer | 会搭 pipeline、理解 buffer/caps/plugin、配置 `nvinfer` | 实时 CV 和视频 I/O 主路径 |
| TensorRT Edge-LLM | 会转换、构建、量化、运行和测 TTFT/TPOT | LLM/VLM 端侧主路径 |
| Triton Inference Server | model repository、backend、instance、dynamic/sequence batcher、metrics | 对照成熟 serving 层，验证调度策略 |
| Docker/NGC | 架构匹配、GPU runtime、bind mount、digest、镜像分层 | 隔离 Python/模型依赖，固化实验 |
| 可观测性 | 指标、结构化日志、trace ID、错误分类 | 关联各模型请求与设备状态 |
| API/IPC | HTTP/gRPC 或轻量本地 IPC，取消/超时/健康检查 | 将 Demo 变成可压测服务 |
| 测试工程 | 单元、集成、回归、soak、故障注入 | 对应 NVIDIA JD 的 test design |

### P2：根据目标岗位深入

| 方向 | 学什么 | 何时投入 |
| --- | --- | --- |
| vLLM/SGLang 源码 | scheduler、paged KV cache、continuous batching、prefix cache | Orin 调度器完成后，对比服务器设计 |
| TensorRT-LLM 主线 | executor、inflight batching、KV cache manager、backend | 有 x86 NVIDIA GPU 或明确岗位需求时 |
| CUTLASS/CuTe/Triton language | GEMM/Attention 模板、tile、layout、自动调优 | Nsight 已证明原生/Plugin kernel 是主要瓶颈时 |
| PTX/SASS | 指令、寄存器、pipeline、反汇编 | 申请 kernel/library 岗且已有 CUDA 项目时 |
| 编译器 | graph IR、MLIR/TVM、fusion/codegen | 申请 TensorRT compiler/AI compiler 岗时 |
| 集群 | tensor/pipeline parallel、NCCL、Kubernetes、Dynamo/Ray | 转向数据中心或集群平台岗位时 |

## 你到底要不要学 CUDA

答案是：**要，但先达到“能够测量、解释和实现关键扩展”的工程深度，不必先成为专职算子工程师。**

岗位需求可以分成三个层次：

1. **部署工程师最低线**：看懂 CUDA 时间线，正确使用 stream/event/异步内存，理解 GPU 资源争用。
2. **推理性能工程师合格线**：能写、验证、profile 一个实际 kernel，能将不支持/低效算子实现为 TensorRT Plugin。
3. **Kernel/库工程师线**：精通 GEMM/Attention、CUTLASS/CuTe、PTX/SASS、微架构性能模型。

你的前 4-5 个月目标应达到第 2 层。第 3 层回报很高，但不是进入推理系统方向的前置条件。

## 多模型场景需要的模型知识

### CV

- resize/normalize/layout conversion、NMS 等前后处理；
- CNN/ViT 的输入形状和 batch 行为；
- 视频解码、颜色格式、GStreamer buffer；
- DLA 可支持层、GPU fallback 和端到端拷贝成本。

### NLP/Embedding

- tokenizer 和动态序列长度；
- padding/bucketing 对 batch 效率的影响；
- cosine/top-k 等后处理；
- 短请求对 launch/queue overhead 的敏感性。

### LLM/VLM

- prefill 与 decode 的资源特征；
- KV cache 容量与上下文/并发关系；
- continuous batching、prefix/KV reuse；
- weight-only INT4/INT8 与 KV 精度；
- TTFT、TPOT、tokens/s 和 goodput；
- VLM 图像编码阶段与语言解码阶段的资源变化。

### Diffusion/音频/其他模型

不是第一阶段主线，但架构要允许加入。Diffusion 是多步迭代、延迟较长；音频和时序模型可能要求连续状态。它们不能都套用无状态 CV 的 dynamic batching。

## 从 JD 到项目任务的映射

| JD 关键词 | 项目内必须做的事 |
| --- | --- |
| 性能分析/优化 | 保存 Nsight trace，解释瓶颈，做可复现前后对照 |
| C/C++ 与软件设计 | C++ runtime/worker、资源生命周期、错误和测试设计 |
| CUDA/访存优化 | 融合预处理 kernel 或 Plugin，分析带宽/占用/同步 |
| TensorRT/ONNX | dynamic shape、engine、量化、unsupported op 处理 |
| 多模型/高并发 | 多类负载、饱和扫描、p95/p99 和 goodput |
| Request Batching | 对不同模型分别验证 batch delay/size，不盲目统一 |
| 模型服务化 | 明确 API、超时、取消、健康检查、指标和版本 |
| 显存/内存优化 | 统一内存预算、engine/context/KV/buffer 分项记录 |
| 高可用 | 进程异常、模型失败、队列过载、热稳态恢复 |
| 开源贡献 | 给官方样例补 Jetson 复现、修文档/脚本或提交小修复 |

## 岗位差距评估方式

每月用证据而不是“学过”打分：

| 等级 | 判断标准 |
| --- | --- |
| 0 未接触 | 不能解释概念 |
| 1 能使用 | 跟随文档跑通，不能独立排错 |
| 2 能诊断 | 能用日志/profile 定位问题并解释 |
| 3 能优化 | 有正确性约束下的前后实验 |
| 4 能设计 | 能做权衡、故障处理和可维护实现 |
| 5 能贡献 | 能修框架问题、提交 upstream 或指导他人 |

求职边缘推理/性能岗位时，P0 项应普遍达到 2-3，TensorRT、profiling、C++ 和并发项目至少有两个达到 4。无需所有方向都达到 5。

## 不要形成的“伪技术栈”

- 列出十个框架但没有一个 profile；
- 只报告平均延迟，不报告 p95/p99 和输入条件；
- 只测模型推理，不测预处理、排队和后处理；
- 只说 INT4 更快，不做任务质量回归；
- 只测单模型峰值，不测并发退化与 OOM；
- 把 Docker、Triton 或 DeepStream 当作自动优化器；
- 把服务器 GPU 文档中的精度和吞吐结论直接套到 Orin。

## 对学习优先级的最终建议

```text
C++/Linux 工程能力
  -> TensorRT + ONNX + benchmark
  -> Nsight + CUDA 实用编程
  -> DeepStream 实时 CV
  -> Edge-LLM + KV cache/量化
  -> 多模型调度/服务/稳定性
  -> vLLM/TRT-LLM/CUTLASS 按岗位深入
```

这条路线既能承接嵌入式经验，也能逐步靠近 JD 中的推理框架、CUDA 和 Serving 要求。

