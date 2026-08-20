# 4-5 个月学习路线

## 路线目标

20 周后应能独立完成并解释：

- Orin NX 16GB 的可复现学习环境；
- CV/NLP 模型从 PyTorch/ONNX 到 TensorRT；
- `trtexec`、Nsight Systems/Compute、`tegrastats` 性能证据；
- 一个实际 CUDA 融合 kernel 和一个 TensorRT Plugin；
- DeepStream 实时视频 pipeline；
- TensorRT Edge-LLM 的小型量化 LLM/VLM；
- CV + NLP + LLM/VLM 的有界并发、准入和降级；
- 正确性、p95/p99、内存、功耗、热稳态和故障测试报告。

这不是“20 周精通所有 AI Infra”。目标是形成边缘异构推理系统的完整闭环，并建立继续深入 vLLM、TensorRT-LLM、CUDA kernel 或 Serving 的基础。

## 双主线交付方式

本路线同时服务两个主攻岗位：边缘推理性能和推理框架研发。每个阶段都保留同一个 Orin 项目 baseline，但分别产出两类证据：

| 轨道 | 关注点 | 典型交付 |
| --- | --- | --- |
| 边缘推理性能 | Orin 端到端延迟、吞吐、功耗、温度、内存、实时性和故障恢复 | TensorRT/DeepStream/Edge-LLM 多模型系统、Nsight/`tegrastats` 报告、soak 和过载记录 |
| 推理框架研发 | runtime 生命周期、CUDA、Plugin、scheduler、batching、KV Cache 和内存管理 | C++ worker/context pool、融合 kernel 或 Plugin、scheduler 原型、源码机制实验 |

前 4-5 个月的共享基础是 C++/Linux、ONNX、TensorRT、Benchmark 和 Nsight Systems；之后再按两个轨道分别增加 DeepStream/Edge-LLM 与 CUDA/Plugin/框架源码深度。不要把两个轨道拆成两套互不相干的项目，也不要为了“覆盖全面”提前引入 Kubernetes、多节点和分布式训练。统一工具链见[双主线工具链与技能栈](09-dual-track-toolchain-and-skill-stack.md)。

## 时间使用原则

- 70%：在目标硬件写代码、跑实验、排错；
- 20%：阅读官方文档和对应源码；
- 10%：整理原始结果和技术报告；
- 每周只引入一个主要变量；
- 每个优化先定义正确性和测量方式；
- 每阶段都保留可运行 baseline，避免只能在最终大系统里调试。

如果每周可投入时间较少，延长日历时间，不删掉验证环节。

## 阶段 0：环境与证据规范（第 1 周）

### 输入

- Jetson Orin NX 16GB + carrier board + NVMe/存储；
- Ubuntu 24.04 x86_64 Host；
- SDK Manager 中 JetPack 7.2.1 目标；
- [学习环境安装](08-environment-setup.md)。

### 练习

1. 完成 Direct Flash 和 target components；
2. 记录 L4T/JetPack/kernel/CUDA/TensorRT/cuDNN/DeepStream；
3. 记录 carrier、存储、散热、power mode；
4. 跑 CUDA `deviceQuery`、`tegrastats`、TensorRT help/smoke；
5. 建立实验目录、命名、Git revision 和证据标签。

### 输出

- exact version manifest；
- 环境安装日志与故障记录；
- 一份 clean-room smoke checklist；
- Orin 磁盘/模型/cache 目录规划。

### 退出条件

- Host/Orin 架构和命令边界明确；
- SDK Manager 无未处理失败项；
- CUDA 和 TensorRT 最小验证成功；
- `tegrastats`、功耗模式和温度可采集；
- 重启后环境仍可用；
- 没有用 x86 结果替代 Orin 实测。

### 暂缓

Triton、vLLM、多个 Python 环境、多模型、量化。

## 阶段 1：单模型 TensorRT 基线（第 2-4 周）

### 输入

- 一个小型分类模型用于工具链基线；
- 一个后续可用于视频的检测/分割 ONNX；
- 固定测试输入集与 reference outputs。

### 练习

1. 从 PyTorch 导出 ONNX，固定 opset、shape 和 preprocessing；
2. 用 ONNX checker/Runtime 做正确性对照；
3. `trtexec` 构建 FP32/FP16 engine；
4. 研究 `min/opt/maxShapes` 和 dynamic shape；
5. 写最小 TensorRT Python runner，再写 C++ runner；
6. 分解 preprocess、enqueue/GPU、postprocess、end-to-end；
7. 做 warm-up、10 次以上重复试验，报告 percentile。

### 输出

- model manifest（来源、revision、license、输入输出）；
- ONNX export script/log；
- engine build command/log；
- Python/C++ 一致性结果；
- FP32/FP16 latency/throughput/memory/quality 表。

### 退出条件

- engine 可从脚本重建，不能只保留二进制；
- 结果与 reference 在明确误差内；
- dynamic profile 不接受无界/不必要范围；
- 能解释 `trtexec` 输出中 GPU compute 与 host/end-to-end 差别；
- 三次独立运行的主要指标波动在预先设定阈值内。

### 暂缓

自定义 kernel、INT8、Triton、DeepStream、多模型。

## 阶段 2：Profiling 与 CUDA 实用编程（第 5-7 周）

### 输入

- 阶段 1 的 C++ runner；
- 固定图像输入和 CPU preprocess baseline；
- Nsight Systems/Compute 与 NVTX。

### 练习

1. 用 NVTX 标记 queue/preprocess/enqueue/postprocess；
2. 用 Nsight Systems 定位 CPU idle、同步、copy 和 launch overhead；
3. 学 streams/events、pinned/managed/device memory 的实际语义；
4. 实现 resize + normalize + HWC->CHW 融合 CUDA kernel；
5. 建 CPU reference 和随机/边界形状测试；
6. 用 compute-sanitizer 检查；
7. 用 Nsight Compute 分析访存、occupancy、warp stalls；
8. 评估 CUDA Graphs 对稳定 shape workload 的作用。

### 输出

- `.nsys-rep` 与关键时间线说明；
- kernel 正确性测试；
- Nsight Compute 前后报告；
- CPU/GPU preprocess 与端到端对照；
- 一篇“为什么快/为什么没快”的分析。

### 退出条件

- 能区分 Linux kernel、CUDA kernel、operator 和 Plugin；
- kernel 无越界/race，数值误差有阈值；
- 优化收益来自 profile 证据，不只来自单次 wall-clock；
- 即使无收益，也能解释瓶颈转移或同步成本；
- 能说明为什么下一步应继续优化或停止。

### 暂缓

PTX/SASS、手写 GEMM、CUTLASS 深入、FlashAttention 实现。

## 阶段 3：TensorRT Plugin 与实时 CV（第 8-10 周）

### 输入

- 检测/分割模型；
- 摄像头或固定视频源；
- 已验证的融合 kernel 或一个真实 unsupported op。

### 练习

1. 实现 TensorRT Plugin V3；
2. 覆盖 shape/type/serialization/context 并发测试；
3. 建 GStreamer pipeline，理解 caps/buffer/latency；
4. 使用 DeepStream 9.1 `nvinfer` 跑单路视频；
5. 测 decode/preprocess/infer/postprocess/display/message 各段；
6. 比较直接 TensorRT C++ 与 DeepStream；
7. 评估 DLA（仅当模型支持）并检查 GPU fallback；
8. 设计 latest-frame/drop 策略。

### 输出

- Plugin 源码、测试和 engine rebuild 说明；
- DeepStream 配置与 pipeline 图；
- 视频输入、FPS、p99、drop、内存和功耗报告；
- GPU-only/DLA 对照（若执行）。

### 退出条件

- 连续运行 30 分钟无资源持续增长；
- CV 端到端 timestamp 定义清晰；
- dropped/stale frame 可计数；
- Plugin engine 可序列化/反序列化并通过回归；
- DeepStream 与直接 runtime 的成本/收益有同模型对照。

### 暂缓

多路视频、大量 source、Triton 托管、LLM 并发。

## 阶段 4：量化与 LLM/VLM Runtime（第 11-14 周）

### 输入

- TensorRT Edge-LLM 当前支持矩阵；
- 能装入 16GB 并留出系统余量的小型受支持模型；
- 固定 prompt/任务集和输出质量评估方法。

### 练习

1. 区分 prefill/decode 与 TTFT/TPOT；
2. 理解 KV cache 容量与 context/concurrency 的关系；
3. 固定 Edge-LLM revision，安装 export/quantization 与 C++ runtime；
4. 对 FP16/INT8/INT4 中实际支持的路径做对照；
5. 扫描 input/output length、并发与 KV cache 预算；
6. 测 cold/warm start、内存峰值和 tokens/s；
7. 若模型支持，评估 context/KV reuse；
8. 用相同数据做任务质量回归。

### 输出

- checkpoint/ONNX/engine hashes；
- precision、memory、TTFT、TPOT、tokens/s、quality 表；
- KV cache 容量估算与实测；
- OOM 边界和安全 headroom；
- 一篇 Edge-LLM 与 vLLM/TRT-LLM 概念对照笔记。

### 退出条件

- 模型/精度确实在 Orin 支持矩阵内；
- INT4/INT8 不只报告速度，还报告质量；
- 不发生不可控 OOM，最大 context/tokens 有硬限制；
- TTFT、TPOT 和端到端 latency 分开；
- 能解释 continuous batching、KV cache 和普通 dynamic batching 的区别。

### 暂缓

大于板卡合理容量的大模型、多 GPU、Tensor Parallel、FP8/FP4、主线 TRT-LLM 强行移植。

## 阶段 5：多模型并发与 Serving（第 15-18 周）

### 输入

- 稳定 CV baseline；
- 小型 NLP/Embedding TensorRT engine；
- 稳定且受预算约束的 Edge-LLM 模型；
- 每个模型的 isolated profile。

### 练习

1. 为三类 workload 建独立 bounded queue；
2. TensorRT context/stream/buffer pool；
3. CV deadline/latest-frame；
4. NLP batch size/queue delay sweep；
5. LLM max concurrency/max tokens/KV admission；
6. 设计 priority、backpressure、admission control、cancel；
7. 隔离与并发退化对照；
8. 使用 Triton 26.04 Jetson 组合托管一个 TensorRT 模型；
9. 用 Perf Analyzer 做 concurrency/request-rate 扫描；
10. 比较直接 runtime、Triton 与 supervisor 组合。

### 输出

- scheduler policy 与状态图；
- 模型/内存/context 配置；
- 固定负载、饱和扫描、批处理扫描原始结果；
- CV p99、NLP p95、LLM TTFT/TPOT 和 system metrics 联合报告；
- 过载 reject/degrade/cancel 记录。

### 退出条件

- 队列和内存都有硬上限；
- CV SLA 在目标组合负载中满足；
- 过载时低优先级 workload 先降级，而不是全部超时；
- 模型隔离与并发退化可重复；
- Triton 是否保留有实测依据；
- 并发数、到达率、输入长度和 batch 参数全部记录。

### 暂缓

Kubernetes、多节点、复杂公平算法、自动扩缩容、分布式 KV cache。

## 阶段 6：稳定性、复现与作品化（第 19-20 周）

### 输入

- 完整多模型系统；
- 自动化 benchmark scenarios；
- 故障和环境矩阵。

### 练习

1. 2 小时以上 thermal soak；
2. memory pressure 与饱和负载；
3. worker crash、模型 load 失败、source disconnect；
4. cold boot 与 clean-room rebuild；
5. 三次以上重复实验；
6. 检查 license、安全、日志轮转、磁盘满；
7. 整理架构、决策、失败尝试和前后结果；
8. 找一个官方项目提交文档修复/复现/小 patch。

### 输出

- 一键/分步复现指南；
- version/model/scenario manifest；
- 原始数据 + 分析 notebook/script + 报告；
- stability/fault report；
- 简历项目描述和架构图；
- 可能的 upstream contribution。

### 退出条件

- clean install 能按文档重建；
- soak 无 OOM、无无界队列/内存增长；
- 失败组件恢复时间和影响范围可量化；
- 所有性能数字能追踪到原始结果；
- 文档明确哪些是 Orin 实测、哪些只是资料结论；
- 不使用“生产级”描述尚未验证的安全/运维能力。

## 每周固定节奏

| 时段 | 活动 |
| --- | --- |
| 开始 | 设定一个可证伪假设和 acceptance |
| 中段 | 实现/实验，保存完整日志和版本 |
| 后段 | 正确性 + profile + 重复测试 |
| 结束 | 写结论、失败原因、下一周唯一主变量 |

每周至少保留一个能重跑的命令，不接受只留截图。

## RK/S100 经验的加速用法

可以复用：

- 模型输入输出契约；
- 导出/转换的检查清单；
- 测试数据、日志和 benchmark schema；
- 服务接口、错误码和部署脚本结构；
- 多设备系统中的消息和故障边界。

必须重新验证：

- 算子支持、精度和量化格式；
- memory allocation/zero-copy；
- runtime 并发语义；
- 性能、功耗和温度；
- model binary/engine 兼容性。

建议先把三个旧项目中可复用的“方法”整理成 checklist，不搬运 vendor-specific binaries 和数字。

## 学习资料的读取顺序

1. JetPack/Jetson Linux 与目标版本 release notes；
2. TensorRT quick start、`trtexec`、dynamic shapes、performance；
3. CUDA Programming Guide 中 kernel/thread/memory/async；
4. Nsight Systems/Compute quick start；
5. DeepStream 9.1 samples 和 `nvinfer`；
6. TensorRT Edge-LLM installation/support/runtime；
7. Triton model config/batcher/perf_analyzer；
8. 最后阅读 vLLM/SGLang/TRT-LLM scheduler/KV 源码。

具体入口见 [GitHub 项目清单](../references/github-projects.md) 与 [官方资料索引](../references/official-resources.md)。

## 路线调整规则

- 若单模型 baseline 不稳定：不进入多模型；
- 若 profile 显示 CPU/I/O 瓶颈：先优化数据链路，不急着写 kernel；
- 若模型不在 Edge-LLM 支持矩阵：换模型，不用数周强行移植；
- 若 16GB 无合理 headroom：缩模型/context/精度，而非靠 OOM 重启；
- 若岗位更偏 kernel：在阶段 3 后增加 CUTLASS/CuTe；
- 若岗位更偏 Serving：在阶段 5 后转到 x86 GPU 做 vLLM/SGLang 对照；
- 若岗位更偏 CV：扩展 DeepStream 多路与 DLA，不必强求 LLM 深度。

## 20 周后继续深入

### 推理框架方向

读 vLLM/SGLang/TRT-LLM 的 scheduler、KV manager、executor，并在服务器 NVIDIA GPU 重现实验。

### CUDA/算子方向

学习 CUTLASS/CuTe/Triton language，选择 GEMM/Attention/Norm 一项做系统优化，不铺开所有算子。

### 边缘平台方向

扩展 DeepStream、多摄像头、DLA、零拷贝、systemd/OTA、安全启动与产品化监控。

### 集群 AI Infra 方向

补多 GPU/NCCL、Kubernetes、Dynamo/Ray、KV cache 分层与容量治理。此时 Orin 项目提供端到端性能工程基础，但仍需独立的集群项目证据。
