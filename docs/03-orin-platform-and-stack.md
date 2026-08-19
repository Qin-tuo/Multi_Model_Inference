# Orin NX 平台与 NVIDIA 推理栈

核验日期：2026-08-19。版本信息来自 NVIDIA 官方发布页、产品数据手册和各框架支持矩阵；实际设备以 [环境安装与盘点](08-environment-setup.md) 的命令输出为准。

## 目标基线

- 目标模块：Jetson Orin NX 16GB（P3767-0000）。
- SDK Manager 截图选择：JetPack 7.2.1、Direct Flash、Ubuntu 24.04 x86_64 主机。
- NVIDIA JetPack 7.2 官方发布信息：Jetson Linux 39.2、CUDA 13.2.1、TensorRT 10.16.2。
- DeepStream 9.1 Jetson 包：基于 JetPack 7.2 GA / L4T 39.2，支持 Jetson Orin。
- TensorRT Edge-LLM：Jetson Orin + JetPack 7.2/CUDA 13.2 为官方支持组合。

`7.2.1` 的精确 L4T patch、内核和组件版本必须在刷机后记录，不能只根据 SDK Manager 的产品标签推断。

## 硬件约束决定软件设计

根据 [Jetson Orin NX 系列数据手册](https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/Jetson-Orin-NX-Series-Modules-Datasheet_DS-10712-001_v1.7.pdf)：Orin NX 16GB 集成 16GB 128-bit LPDDR5、Ampere GPU、Arm CPU、NVDLA 和多媒体硬件。

### 统一内存不是“无限零拷贝”

CPU 和 GPU 共享同一物理内存资源，减少了独立显存系统中的一类 PCIe 拷贝，但仍存在：

- 页面和 buffer 生命周期；
- cache/coherency 与访问模式成本；
- CPU、GPU、DLA、视频引擎共同争抢 DRAM 带宽；
- TensorRT engine weights、activation、workspace、context；
- LLM weights 与 KV cache；
- 视频 surface、预处理 buffer 和应用内存；
- Linux 文件缓存及其他进程。

因此“还有 2GB free”不是可靠容量规划。必须按组件记录峰值，并在并发与热稳态下观察 OOM/回收行为。

### 各计算单元的角色

| 单元 | 适合工作 | 注意事项 |
| --- | --- | --- |
| Arm CPU | I/O、tokenizer、调度、业务逻辑、轻量后处理 | CPU 满载也会导致 GPU 饥饿和队列延迟 |
| Ampere GPU | 通用 TensorRT、CUDA kernel、LLM/VLM | 多模型共享 SM、L2、DRAM 和 launch 通道 |
| NVDLA | 支持范围内的 CNN 层/engine | 算子与格式有限，GPU fallback 可能增加同步/拷贝 |
| 视频引擎 | H.264/H.265 等编解码 | 能减少 CPU/GPU 压力，但 pipeline buffer 仍占内存 |
| 摄像头/ISP 路径 | CSI/V4L2/图像输入 | carrier board、驱动和格式会影响零拷贝路径 |

### 功耗和温度是性能条件

所有 benchmark 必须记录 `nvpmodel` 模式、频率策略、风扇/散热和温度。短测试的峰值不能代表热稳态性能。`jetson_clocks` 仅用于受控实验，不能替代产品功耗设计。

## 软件栈分层

```text
模型与业务
  CV / Embedding / NLP / LLM / VLM
                  |
服务与调度        |  自研 scheduler / Triton Inference Server
                  |
领域流水线        |  DeepStream + GStreamer / TensorRT Edge-LLM
                  |
推理 Runtime      |  TensorRT / cuDNN / cuBLAS
                  |
并行计算与工具    |  CUDA / Nsight Systems / Nsight Compute
                  |
平台              |  JetPack / Jetson Linux / 驱动 / 容器 Runtime
                  |
硬件              |  CPU + GPU + DLA + Video + Unified Memory
```

每层解决的问题不同。Triton 不是 TensorRT 的替代品，DeepStream 也不是 CUDA 的替代品。

## 各组件的职责和学习优先级

### JetPack / Jetson Linux

提供 BSP、Linux kernel、驱动、CUDA、TensorRT、cuDNN、多媒体 API、工具和容器支持。它是兼容性根节点。

实践原则：

- 先用目标 JetPack 的系统包建立干净基线；
- 不用通用 Ubuntu CUDA 仓库覆盖 JetPack 组件；
- 固定 L4T/JetPack 后再固定 DeepStream、容器和编译产物；
- 升级 JetPack 视为一次平台迁移，重新构建 engine 和回归测试。

### CUDA

提供 GPU 编程模型、runtime、streams/events/graphs、memory API 和基础库。CUDA 的主要价值不是“所有算子都手写”，而是理解 TensorRT 和并发 workload 实际怎样使用 GPU。

主线：CUDA samples -> 异步流水线 -> Nsight -> 一个融合 kernel -> TensorRT Plugin。

### cuDNN / cuBLAS 等库

提供卷积、矩阵乘等高度优化实现。通常优先让 TensorRT/库选择成熟 kernel，手写实现只有在 profile 证明缺口时才合理。

### TensorRT

Orin 上 CV、Embedding、分类和通用 ONNX 模型的核心推理引擎：

- 解析/构建网络；
- 选择精度和 tactic；
- 融合层；
- 管理 optimization profile 和 execution context；
- 通过 Plugin V3 扩展自定义层；
- 用 `enqueueV3()` 异步提交到 CUDA stream。

`trtexec` 是第一基线工具，不是最终应用。需要分别测 GPU compute、host latency、端到端 latency 和数据传输。

### GStreamer / DeepStream

GStreamer 负责媒体 pipeline；DeepStream 在其上提供 NVIDIA buffer、推理、跟踪、消息等插件。它适合多路视频实时 CV，而不是通用 LLM serving。

DeepStream 9.1 的重要变化：

- 支持 Jetson Orin + JetPack 7.2；
- 主仓库迁到 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream)；
- 旧 `NVIDIA-AI-IOT/deepstream_reference_apps` 已归档；
- Python `pyds` bindings 已弃用，官方建议迁向 `pyservicemaker`；
- Jetson 的 Triton 组合对应 Triton 26.04；
- Jetson DeepStream 容器主要面向部署，原生 target 或官方说明的构建路径用于开发。

### TensorRT Edge-LLM

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 是本项目在 Orin 上学习 LLM/VLM 的首选 runtime：

- 轻量 C++ runtime；
- 在目标设备构建 TensorRT engine；
- Jetson Orin + JetPack 7.2 为官方支持；
- Orin 支持 FP16、INT8、INT4；不支持该矩阵中的 FP8/FP4 engine；
- 支持情况必须按具体模型列表核对；
- 模型/ONNX/engine 可能需要约 20-50GB 磁盘空间，需提前规划 NVMe。

它可以用于研究 weight quantization、KV cache、prefix/context reuse、TTFT/TPOT 和 C++ serving integration。

### Triton Inference Server

这里的 Triton 指 **NVIDIA Triton Inference Server**，不是 OpenAI Triton kernel language。

Triton 提供：

- model repository 和生命周期管理；
- HTTP/gRPC/C API；
- TensorRT、ONNX Runtime、Python 等 backend；
- per-model dynamic batcher、sequence batcher、instance group；
- 并发模型执行、队列策略、优先级和 metrics；
- Perf Analyzer / Model Analyzer 等配套工具。

在 Orin 上的定位：

- 作为成熟 serving 对照和部分实际托管方案；
- 优先使用与 DeepStream 9.1/JetPack 7.2 匹配的 Jetson Triton 发行组合；
- 先验证模型与 backend 是否在该 Jetson 构建中可用；
- 16GB 下 server/backend/model instance 本身有内存开销，不能默认比直接 TensorRT C++ 更优；
- dynamic batching 和多 instance 必须逐模型压测，不能同时盲开。

### Nsight Systems / Nsight Compute

- Nsight Systems：端到端时间线，观察 CPU threads、CUDA API、streams、memcpy、kernel、同步和 NVTX。
- Nsight Compute：单个 CUDA kernel 的吞吐、访存、占用、warp stall 等。

先 Systems 后 Compute：先确认真正耗时的 kernel，再对少数热点做深入采样。

## Orin 部署路径与源码学习路径

| 框架 | Orin NX 实践定位 | 主要学习内容 | 当前限制/注意 |
| --- | --- | --- | --- |
| TensorRT | **P0 实际部署** | engine/context/profile/plugin | 绑定 JetPack TensorRT 版本 |
| DeepStream 9.1 | **P0/P1 实际部署** | 视频 pipeline、多流、buffer | 版本需匹配 JP 7.2；Python API 正迁移 |
| TensorRT Edge-LLM | **P1 实际部署** | LLM/VLM、量化、KV/runtime | 模型/精度支持按矩阵；16GB 容量严格 |
| Triton 26.04 Jetson 组合 | **P1 对照/可选部署** | batcher、instance、metrics | 内存和 backend 支持需实测 |
| vLLM | **P2 源码/服务器对照** | PagedAttention、continuous batching、scheduler | 官方 ARM64 构建示例面向 Grace 系统，不等同 Jetson 发行支持 |
| TensorRT-LLM 主线 | **P2 源码/服务器对照** | executor、KV cache、inflight batching | 支持矩阵含 Linux aarch64/Ampere，但 Jetson 主线不是明确持续测试目标 |
| TensorRT-LLM `v0.12.0-jetson` | 历史参考 | JetPack 6.1 Jetson 路径 | 旧分支，不作为 JP 7.2 新项目主线 |
| SGLang | **P2 源码/服务器对照** | prefix cache、scheduler、structured generation | 数据中心 serving 假设较多 |
| Triton language | **P2 kernel 学习** | Python DSL 写 GPU kernel | 与 Triton Server 是两个项目；不是当前 P0 |

## 模型类型到推荐路径

| 模型/工作负载 | 第一选择 | 第二选择/对照 |
| --- | --- | --- |
| 单图 CV/Embedding ONNX | TensorRT C++/Python | Triton TensorRT backend |
| 多路实时视频 | DeepStream `nvinfer` | DeepStream Triton `nvinferserver` |
| 小型 NLP encoder | TensorRT | Triton dynamic batching |
| LLM/VLM | TensorRT Edge-LLM | 旧 TRT-LLM Jetson 仅作历史参考 |
| 多框架模型仓库 | Triton（先确认 Jetson backend） | 自研轻量 supervisor/scheduler |
| 极致低开销嵌入式集成 | 直接 C++ runtime | Triton 用于行为/性能对照 |

## DLA 是否应该使用

DLA 可用于给 GPU 腾出部分 CNN 负载，但不是自动收益：

1. 检查模型层和数据格式是否支持；
2. 分别构建 GPU-only 与 DLA engine；
3. 检查 GPU fallback；
4. 测端到端 latency、CPU、内存带宽和功耗；
5. 再测与 LLM 并发时是否改善 CV SLA。

如果 fallback 和数据转换成本大于释放的 GPU 资源，DLA 方案可能更差。

## Engine 与模型制品规则

- ONNX 是可移植中间表示，但仍受 opset/算子和动态形状约束；
- TensorRT engine 与 GPU 架构、TensorRT 版本、builder 配置高度相关；
- engine 与校准 cache 要记录来源 model hash；
- 不把旧 TensorRT engine 直接带到新的 JetPack；
- 不仅保存 engine，还要保存可重建脚本、shape profile、精度和构建日志。

## 推荐的学习环境分层

```text
Host Ubuntu 24.04 x86_64
  SDK Manager / 下载缓存 / Git / 报告查看
             |
             | USB flash + Ethernet/SSH
             v
Orin native baseline
  JetPack + CUDA + TensorRT + DeepStream + profiling target
             |
             +-> native C++/CUDA/TensorRT exercises
             |
             +-> JetPack-matched arm64 containers
                   model tools / optional Triton / isolated Python deps
```

具体步骤见 [学习环境安装](08-environment-setup.md)。

## 本项目的框架选择结论

1. **TensorRT 是主干**：覆盖 CV/NLP/通用 ONNX，也是 CUDA/Plugin 的落地点。
2. **DeepStream 是实时视频主干**：解决媒体流水线，不承担 LLM 调度。
3. **TensorRT Edge-LLM 是 Orin 上的 LLM/VLM 主干**。
4. **Triton Server 是 serving 能力对照和可选托管层**，要用实测决定是否保留。
5. **vLLM、SGLang、TensorRT-LLM 主线是设计参考**，不强行作为 Orin 生产 runtime。
6. **自研部分只做平台缺少的策略**：跨模型优先级、SLA、内存预算、背压和降级，而不是重写 TensorRT。

## 继续核验的官方入口

所有链接、版本依据和重新核验方法统一维护在 [官方资料索引](../references/official-resources.md)。

