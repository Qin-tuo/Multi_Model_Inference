# 官方资料索引

核验日期：2026-08-19。

外部网页会变化。开始实现、升级 JetPack 或重新构建 engine 前，重新检查 release/tag、页面更新时间和 support matrix。

## JetPack、SDK Manager 与 Jetson Linux

### JetPack 7.2 下载与发布信息

- [JetPack SDK Downloads and Notes](https://developer.nvidia.com/embedded/jetpack/downloads)
- 用途：确认 JetPack 7.2、Jetson Linux 39.2、CUDA 13.2.1、TensorRT 10.16.2 和支持硬件。
- 何时重查：每次 SDK Manager 显示新 patch 或准备升级。

### JetPack Archive

- [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive)
- 用途：确认历史 JetPack 与 L4T 映射，避免把 JetPack 6/7 的说明混用。

### SDK Manager

- [System Requirements](https://docs.nvidia.com/sdk-manager/system-requirements/index.html)
- [SDK Manager Documentation](https://docs.nvidia.com/sdk-manager/)
- 用途：Host OS、内存/磁盘最低要求、支持 target、安装/repair/log 流程。
- 注意：SDK Manager 自身支持某 Host OS，不自动代表每个 SDK package 都支持；继续查 compatibility matrix。

### Jetson Linux r39.2 / r39.2.1

- [Jetson Linux r39.2 Developer Guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/)
- [r39.2.1 Quick Start](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/IN/QuickStart.html)
- [Flashing Support](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/FlashingSupport.html)
- 用途：Orin 支持、Force Recovery、USB ID、rootfs/flash、BSP 和平台功能。
- 注意：第三方 carrier board 的 recovery/flash 参数仍以板卡厂商资料为准。

### Orin NX 硬件

- [Jetson Orin NX Series Modules Data Sheet](https://developer.download.nvidia.com/assets/embedded/secure/jetson/orin_nx/docs/Jetson-Orin-NX-Series-Modules-Datasheet_DS-10712-001_v1.7.pdf)
- 用途：CPU/GPU、16GB 128-bit LPDDR5、DLA、多媒体、接口和电气/热设计约束。
- 注意：峰值规格不是业务 inference 性能。

### Power/telemetry/validation

- [Tegrastats Utility](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [Jetson Test Plan and Validation](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/TestPlanValidation.html)
- 用途：解释内存、CPU/GPU/EMC、温度/功耗输出，验证 power mode 和冷/热启动。

## CUDA 与 Profiling

### CUDA Programming Guide

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA 13.2 PDF](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
- 用途：kernel、thread hierarchy、memory hierarchy、streams、events、graphs 和异步模型。
- 推荐章节：Programming Model、Hardware Implementation、Asynchronous Execution、CUDA Graphs、Performance Guidelines。

### CUDA Best Practices

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 用途：测量、访存、并行、数值和优化顺序。

### Nsight Systems

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)
- [User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- 用途：Host/target 区分、Tegra Embedded Platforms Edition、CPU/CUDA 时间线、NVTX。
- 注意：Jetson 使用与 JetPack 匹配的 target package；最新 Workstation package 不自动匹配 Tegra target。

### Nsight Compute

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- 用途：热点 CUDA kernel 的 memory、occupancy、warp stall 和 roofline 分析。
- 顺序：先用 Systems 找热点，再用 Compute 深入。

### CUDA 正确性检查

- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- 用途：检查 CUDA kernel 的越界、race、初始化和同步问题；先保证 correctness，再讨论性能。

## TensorRT

### 文档入口

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- 用途：安装、API、runtime、performance、troubleshooting。
- 注意：`latest` 在核验日已是 TensorRT 11.2.1，而 JetPack 7.2 带 TensorRT 10.16.2；涉及 API/plugin 时切换到匹配版本或使用设备 headers。

### 核心专题

- [Performance Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)：`trtexec`、warm-up、latency/throughput、dynamic shapes。
- [Optimizing TensorRT Performance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)：cross/within inference streams、contention、CUDA Graphs。
- [Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)：optimization profiles 和 runtime shapes。
- [Adding Custom Layers with C++ Plugins](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-cpp.html)：Plugin V3、shape 和 enqueue。
- [How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)：build/runtime memory、engine/context。

## DeepStream 9.1

- [DeepStream 9.1 Release Notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html)
- [DeepStream Installation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html)
- [DeepStream Docker Containers](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html)
- [DeepStream GitHub Monorepo FAQ](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Monorepo_FAQ.html)
- [DeepStream Migration Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Migration_guide.html)
- [DeepStream 9.1 API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/)
- 用途：JetPack 7.2/L4T 39.2 兼容、Orin 支持、arm64 package/container、Triton 26.04、Python API 迁移和旧仓库归档信息。
- 注意：安装包资产在 `NVIDIA/DeepStream` GitHub release，容器在 NGC `deepstream` repository。

## TensorRT Edge-LLM

- [Documentation Home](https://nvidia.github.io/TensorRT-Edge-LLM/)
- [Official Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)
- [Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/supported-models.html)
- [Installation](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)
- [Runtime Design](https://nvidia.github.io/TensorRT-Edge-LLM/developer_guide/software-design/llm-inference-runtime.html)
- 用途：Jetson Orin + JetPack/CUDA/TensorRT、模型/精度、Host export、target C++ build、KV/context runtime。
- 核验结论：JetPack 7.2/CUDA 13.2 的 Jetson Orin 是官方组合；Orin runtime precision 为 FP16/INT8/INT4，不包含 FP8/FP4。
- 注意：support matrix、supported models 和安装页都可能随 commit 更新，三者必须用同一 release/revision 阅读。

## Triton Inference Server

- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)
- [Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [Optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html)
- [Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html)
- [Release 26.04 Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html)
- 用途：model repository、dynamic/sequence batching、concurrent model execution、instance、metrics 和 release components。
- Jetson 入口：优先跟随 DeepStream 9.1 的 `9.1-triton-multiarch` 组合与其文档，不从通用 dGPU release notes 推断所有 Jetson backend。

### Triton 压测

- [Perf Analyzer repository](https://github.com/triton-inference-server/perf_analyzer)
- [Perf Analyzer CLI](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/cli.md)
- [Measurement and Metrics](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/measurements_metrics.md)
- [Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
- 用途：concurrency/request-rate、percentile、warm-up、measurement stability、model config search。

## Model Optimizer

- [Model Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/)
- [Linux Installation](https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html)
- 用途：PTQ/QAT/INT8/INT4、校准和 export。
- 注意：通用 Linux 页的 aarch64 支持写为 SBSA；Jetson 路径优先使用 TensorRT Edge-LLM 文档规定的 `.[tools]` 依赖组合。

## vLLM、TensorRT-LLM 与源码对照

### vLLM

- [GPU Installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [Documentation](https://docs.vllm.ai/en/latest/)
- 用途：理解 NVIDIA GPU 要求、ARM64 source build、serving 和 scheduler/KV 功能。
- 注意：ARM64 GPU 构建示例指向 GH200/GB 系统，不等同 Jetson Orin release support。

### TensorRT-LLM

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- 用途：服务器侧 executor、KV cache、inflight batching、benchmark 和 Triton backend。
- 注意：不要仅凭 Linux aarch64/Ampere 支持推断当前主线已在 Jetson Orin/JetPack 7.2 持续验证。

## 岗位 JD 原始来源

- [NVIDIA - AI Computing Software Development Engineer, LLM Inference](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Computing-Software-Development-Engineer--LLM-Inference_JR2019592)
- [百度 - 推理性能优化工程师](https://talent.baidu.com/jobs/detail/SOCIAL/14c8d37a-3288-4280-8fbf-71a878b010eb)
- [昆仑芯 - 模型加速与部署工程师](https://kunlunxin.zhiye.com/xiangqing?jobId=151141586)
- 用途：校准 C++/CUDA/框架/量化/Serving/测试的学习深度；岗位内容和有效状态可能变化，投递前重新打开原页面。

## 重新核验清单

升级或开始新实验时逐项回答：

```text
[ ] JetPack 与 L4T 精确映射是否变化？
[ ] Target module/carrier/storage 是否仍在支持列表？
[ ] CUDA/TensorRT/DeepStream 版本是否与当前 image/tag 匹配？
[ ] Edge-LLM 的 platform/precision/model 三个矩阵是否都支持？
[ ] GitHub 仓库是否归档、迁移或更改默认 API？
[ ] 容器 manifest 是否包含 linux/arm64，digest 是否记录？
[ ] 文档 latest 是否已经超出 JetPack 自带 major version？
[ ] 性能结论是否来自当前 target，而不是官方服务器数字？
```
