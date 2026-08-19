# GitHub 项目清单

核验日期：2026-08-19。

## 使用规则

1. 先看 `release/tag` 和支持矩阵，再 clone；
2. 在实验 manifest 中记录 commit SHA，不用浮动 `main/master` 复现性能；
3. 检查 architecture、JetPack/L4T、CUDA、TensorRT 和 Python；
4. `README 能运行` 不等于 `Orin 官方支持`；
5. 许可证、模型条款、第三方 submodule 都要独立检查；
6. 不同时 clone/编译全部项目，只为当前 milestone 引入一个。

优先级含义：

- **P0 实操**：直接服务于 Orin 单模型、多模型和 CUDA 基线；
- **P1 工具**：量化、容器、监控、压测和专项参考；
- **P2 源码研读**：理解框架/算子设计，默认不作为 Orin 主 runtime。

## P0：Orin 实操主线

### NVIDIA/cuda-samples

- 地址：[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- 用途：学习 CUDA API、执行模型和环境 smoke test。
- 先做：`deviceQuery`、`bandwidthTest`、stream/event/graph 相关小样例；使用与 CUDA 13.2 匹配的 tag/branch。
- 阅读重点：error handling、grid/block、async API；不要按目录从头到尾刷题。
- Orin 价值：验证 GPU/toolchain，并给自写 preprocess kernel 提供最小工程模板。
- 限制：sample 的 microbenchmark 不是业务端到端性能，也不代表 TensorRT 最优实现。

### NVIDIA/TensorRT

- 地址：[NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
- 用途：TensorRT 开源组件、parser、plugins、samples 与 `trtexec` 源码。
- 先做：查看与你设备 TensorRT 10.16 对应的 release；跑 `trtexec`，再读 ONNX sample 和 Plugin V3 示例。
- 阅读重点：engine/context、dynamic shape、buffer、enqueue、plugin serialization。
- Orin 价值：CV/NLP/通用 ONNX 模型的主 runtime。
- 限制：GitHub `main` 可能对应 TensorRT 11.x 等较新 API，不能直接拿来解释 JetPack 里的 10.16；以已安装 headers/docs 为准。

### NVIDIA/DeepStream

- 地址：[NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream)
- 用途：DeepStream 9.x 单仓库、release assets、samples、工具和参考应用。
- 先做：固定 `v9.1.0` release，阅读 Jetson 安装、sample configs 和 `nvinfer` pipeline。
- 阅读重点：GStreamer pipeline、buffer/caps、batch mux、metadata、source lifecycle、Triton integration。
- Orin 价值：实时视频、多路 CV 和硬件编解码主路径。
- 限制：Jetson 容器主要用于部署；DeepStream 9.1 Python `pyds` 已弃用，优先 C/C++ 或 `pyservicemaker`。

### NVIDIA-AI-IOT/deepstream_reference_apps（历史）

- 地址：[NVIDIA-AI-IOT/deepstream_reference_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)
- 用途：查找旧版 parallel inference 和历史参考应用。
- 重点：`deepstream_parallel_inference_app` 可帮助理解多分支 pipeline 思路。
- Orin 价值：作为架构和迁移参考，不作为 DeepStream 9.1 新项目基线。
- 限制：DeepStream 9.1 官方 release notes 将它列为不再更新、后续迁入 `NVIDIA/DeepStream` 的旧仓库。GitHub API 在核验日尚未设置 archived flag，但应以产品迁移说明为准。

### NVIDIA/TensorRT-Edge-LLM

- 地址：[NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- 用途：Jetson/DRIVE/DGX Spark 上的轻量 C++ LLM/VLM runtime。
- 先做：支持矩阵 -> 安装 -> supported models -> text quick start；固定 release/tag/commit。
- 阅读重点：`tensorrt_edgellm` 导出/量化、`cpp` runtime、engine builders、examples、KV/context reuse。
- Orin 价值：JetPack 7.2 Orin 的 LLM/VLM 首选实操项目。
- 限制：Orin 只走当前矩阵允许的 FP16/INT8/INT4；功能和模型支持是 model-specific；构建和模型制品占用大量磁盘/内存。

### triton-inference-server/server

- 地址：[triton-inference-server/server](https://github.com/triton-inference-server/server)
- 用途：NVIDIA Triton Inference Server 核心发行和文档。
- 先做：architecture、model repository/config、model execution、batcher、metrics、rate limiter。
- 阅读重点：dynamic/sequence batcher、instance group、queue policy、backend 生命周期、health/metrics。
- Orin 价值：Serving 能力的成熟对照；DeepStream 9.1 对应的 Jetson Triton 26.04 组合可用于实验。
- 限制：主仓库同时面向数据中心和 edge；backend/container 必须核对 Jetson 发行内容，不能直接使用通用 dGPU `latest`。

### triton-inference-server/tutorials

- 地址：[triton-inference-server/tutorials](https://github.com/triton-inference-server/tutorials)
- 用途：模型部署、dynamic batching、concurrent model execution 等概念教程。
- 先做：Conceptual Guide Part 1/2，使用一个 TensorRT engine 建最小 model repository。
- Orin 价值：快速理解 Triton 配置与性能实验闭环。
- 限制：教程命令可能使用 x86 dGPU container；在 Orin 替换为已核验的 multiarch/Jetson 组合。

## P1：优化、压测与平台工具

### NVIDIA/Model-Optimizer

- 地址：[NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- 用途：PTQ/QAT、低比特量化、剪枝、蒸馏和模型优化。
- 先做：只读与你选定模型/Edge-LLM export 相关的 quantization example；优先通过 Edge-LLM `.[tools]` 引入匹配依赖。
- 阅读重点：校准数据、quant config、export、quality evaluation。
- Orin 价值：INT8/INT4 checkpoint/ONNX 生成与质量回归。
- 限制：当前通用安装支持列表中的 aarch64 是 SBSA 表述，不能由此推导所有 Jetson 组合；部分加速 quantization kernel 要求较新 compute capability/格式，不适用于 Orin。

### NVIDIA/CUDALibrarySamples

- 地址：[NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)
- 用途：cuBLAS、cuDNN、cuFFT、cuSPARSE 等 CUDA libraries 示例。
- 先做：在自写 GEMM/卷积前阅读相应 library sample，理解 descriptor、workspace 和 stream。
- Orin 价值：学会优先复用成熟库，而不是从零写算子。
- 限制：不少样例只证明 API，不包含完整模型数据链路和并发设计。

### triton-inference-server/perf_analyzer

- 地址：[triton-inference-server/perf_analyzer](https://github.com/triton-inference-server/perf_analyzer)
- 用途：对 Triton/OpenAI-compatible 等服务生成负载，测 throughput/latency。
- 先做：`docs/cli.md` 的 concurrency mode、request-rate mode、percentile、warm-up、input data 和 stability。
- 阅读重点：client/server latency breakdown、queue/compute、measurement windows、Poisson arrival。
- Orin 价值：固定模型做并发/到达率扫描，为 scheduler/Triton 对照提供负载。
- 限制：压测 client 最好与 server 资源隔离；同机运行会争 CPU/内存。仓库已提示 `genai-perf` 逐步迁向 AIPerf，使用前核验当前工具路线。

### triton-inference-server/model_analyzer

- 地址：[triton-inference-server/model_analyzer](https://github.com/triton-inference-server/model_analyzer)
- 用途：扫描 model config、instance、batch 等部署组合。
- 先做：了解 config schema 和如何调用 Perf Analyzer；先手工建立小搜索空间。
- Orin 价值：验证 instance/batch 配置，而非凭经验选值。
- 限制：自动搜索本身消耗时间和资源；Jetson telemetry/组合支持需实测，不一定覆盖自研 Edge-LLM runtime。

### dusty-nv/jetson-containers

- 地址：[dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)
- 用途：构建/运行 Jetson 与 JetPack-L4T 适配的 ML containers。
- 先做：阅读 package list、build/run、数据/cache mount 和当前 JetPack compatibility。
- Orin 价值：快速隔离 PyTorch、LLM、VLM 和工具依赖。
- 限制：社区项目，不是 JetPack 的兼容性权威；先用官方原生/multiarch image 建基线，再将它用于加速实验。

### rbonghi/jetson_stats

- 地址：[rbonghi/jetson_stats](https://github.com/rbonghi/jetson_stats)
- 用途：`jtop` 交互监控与 Python API。
- 先做：在官方 `tegrastats` 基线后安装，比较显示项和版本识别。
- Orin 价值：开发期快速查看功耗、频率、内存和进程。
- 限制：社区工具，不能替代原始 `tegrastats`、Nsight 或应用指标；安装会改变 Python/system 状态，应记录版本。

### NVIDIA-AI-IOT/Lidar_AI_Solution

- 地址：[NVIDIA-AI-IOT/Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)
- 用途：LiDAR/camera 模型、BEVFusion 以及 cuPCL、SparseConv、YUV2RGB、cuOSD 等 GPU 组件。
- 先做：选择性阅读 CUDA preprocess/postprocess、buffer 和融合设计，不要直接构建整个方案。
- Orin 价值：参考复杂 CV/多传感器场景中自定义 CUDA 与 TensorRT 的结合。
- 限制：场景复杂、依赖重，不适合作为前两个月入门项目；版本支持要逐项核对。

## P2：推理框架与 Kernel 源码研读

### vllm-project/vllm

- 地址：[vllm-project/vllm](https://github.com/vllm-project/vllm)
- 用途：高吞吐 LLM inference/serving，PagedAttention、continuous batching、prefix caching 和 scheduler。
- 阅读顺序：架构文档 -> request lifecycle -> scheduler -> KV cache manager/block pool -> worker/model runner -> metrics/benchmarks。
- Orin 价值：为自研 admission/KV/queue 策略提供成熟设计参照。
- 限制：官方已有 ARM64 from-source 路径，但示例面向 Grace-Hopper/Grace-Blackwell；这不等于 Jetson Orin 的发行级支持。不要把它作为本项目 P0 runtime。

### NVIDIA/TensorRT-LLM

- 地址：[NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- 用途：数据中心 NVIDIA GPU 的 LLM runtime、executor、KV cache、inflight batching、kernels 和 Triton backend。
- 阅读顺序：architecture/executor -> runtime request -> KV cache manager -> scheduler -> benchmark -> Triton backend。
- Orin 价值：理解 NVIDIA LLM inference 的服务器侧设计，并与 Edge-LLM 比较。
- 限制：主线 support matrix 虽含 Linux aarch64/Ampere，但未把 Jetson Orin 明确列为持续测试的主线产品；旧 `v0.12.0-jetson` 对应 JetPack 6.1，是历史参考，不用于 JetPack 7.2 新项目。

### NVIDIA/cutlass

- 地址：[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- 用途：高性能 GEMM/卷积/Attention 模板与 CuTe DSL。
- 阅读顺序：quickstart -> GEMM hierarchy -> layouts -> profiler -> 选一个与 Orin SM 匹配的 example。
- Orin 价值：理解 Tensor Core tile、数据布局和高性能 kernel 结构；为高级 Plugin 优化准备。
- 限制：编译体量和模板复杂度高；必须核对目标 SM/CUDA 支持，不能因为服务器样例快就推断 Orin 收益。

### sgl-project/sglang

- 地址：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- 用途：LLM/VLM serving、scheduler、prefix/radix cache、structured generation。
- 阅读顺序：server architecture -> scheduler/request lifecycle -> cache -> benchmark。
- Orin 价值：对比 vLLM 的调度和 prefix reuse 思路。
- 限制：主要面向服务器 GPU 与分布式 serving；只做源码/概念对照，不列入 Orin 安装清单。

## 社区课程与知识地图

### lyy-ai/ai_infra

- 地址：[lyy-ai/ai_infra](https://github.com/lyy-ai/ai_infra)
- 定位：中文 AI Infra 课程总目录，覆盖 CUDA/HPC、AI 编译器、Runtime、LLM Serving、分布式训练、量化、多平台和工程体系。
- 推荐阅读：`01` 全景 -> `02` CUDA -> `04` Runtime -> `05` LLM Serving -> `07` 量化 -> `08` 多平台；本项目暂不优先 `06` 分布式训练。
- 与本路线结合：用它补概念和练习视角，再回到 NVIDIA 官方文档与本仓库的 Orin milestone 做验证。
- 注意：仓库核验时创建时间较新、提交数较少，README 声称部分目录含可运行代码，但统一运行环境写死为作者的 `/data/liyangyang/qwen35_env/bin/python`；不能直接复用。
- License：GitHub API 在核验日未检测到 license。阅读学习通常无碍，但在复制、修改或分发代码前必须由作者补充许可或获得授权。
- 证据等级：社区学习资料，不作为 JetPack/框架兼容性和性能结论的权威来源。

## 建议 clone 顺序

不要一次 clone 全部：

```text
第 1 月: cuda-samples + TensorRT
第 2 月: DeepStream + 选定 CUDA library sample
第 3 月: TensorRT-Edge-LLM + Model-Optimizer（由工具链带入）
第 4 月: Triton server/tutorials + perf_analyzer
之后: vLLM / TensorRT-LLM / CUTLASS / SGLang 按岗位阅读
全程: lyy-ai/ai_infra 作为课程索引，不作为运行依赖
```

## 固定 revision 模板

每次引入仓库时记录：

```text
repository: https://github.com/<owner>/<repo>
revision: <git commit SHA>
release/tag: <tag or none>
checked_at: 2026-08-19
platform: Jetson Orin NX 16GB / Ubuntu aarch64
jetpack_l4t: <exact>
purpose: <what this revision is used for>
local_changes: <patch list or none>
license_review: <result>
```

