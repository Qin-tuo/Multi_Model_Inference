# Orin NX Multi-Model Inference

面向 NVIDIA Jetson Orin NX 16GB 的多模型推理、并发调度与性能优化知识库。

这里的目标不是只把一个 LLM 跑起来，而是在一块资源受限的边缘设备上，让实时 CV、NLP/Embedding、LLM/VLM 等不同负载可测量、可并存、可降级、可复现。职业方向可概括为：

> **边缘异构模型推理系统与性能优化**

## 当前状态

| 项目项 | 状态 |
| --- | --- |
| 技术调研与路线设计 | 已整理 |
| JetPack/框架兼容性 | 已按 2026-08-19 的官方资料核验 |
| 学习环境安装说明 | 已整理，需在目标设备上执行并回填版本清单 |
| Orin NX 单模型基线 | 尚未实测 |
| 多模型并发系统 | 尚未实现 |
| 生产级结论 | 无；必须经过目标硬件实测、稳定性和故障测试后才能形成 |

本仓库现阶段是**知识库和项目蓝图**，不是一个已经可运行的推理服务，也不包含虚构的 Orin 性能数据。

## 证据标签

阅读文档时按下面三类判断结论强度：

- **资料结论**：来自官方文档、官方仓库、岗位 JD 或明确标注的工程推导。
- **主机验证**：已经在 Ubuntu x86_64 主机完成，但不代表 Jetson aarch64 目标结果。
- **Orin 实测**：必须记录板卡、功耗模式、JetPack/L4T、模型、精度、输入和原始数据。目前尚无此类结果。

如果一条性能结论没有实验清单和原始结果，它只能是待验证假设。

## 文档导航

建议第一次按顺序阅读：

1. [方向定位与目标](docs/01-positioning-and-goals.md)
2. [岗位 JD 与技术栈映射](docs/02-job-skill-map.md)
3. [Orin NX 平台与 NVIDIA 推理栈](docs/03-orin-platform-and-stack.md)
4. [学习环境安装](docs/08-environment-setup.md)
5. [CUDA Kernel、算子与 Plugin](docs/04-cuda-kernel-and-operator-optimization.md)
6. [多模型并发架构](docs/05-multi-model-concurrency-architecture.md)
7. [4-5 个月学习路线](docs/06-learning-roadmap.md)
8. [项目蓝图与验收标准](docs/07-project-blueprint-and-acceptance.md)
9. [GitHub 项目清单](references/github-projects.md)
10. [官方资料索引](references/official-resources.md)

按问题快速阅读：

| 你当前的问题 | 先看 |
| --- | --- |
| 我是否应该学 CUDA | `04`，再看 `02` |
| Linux kernel 和 CUDA kernel 有什么区别 | `04` |
| Triton 是什么、是否适合 Orin | `03` 和 `05` |
| 多个 CV/NLP/LLM 如何并存 | `05` 和 `07` |
| SDK Manager 后还要装什么 | `08` |
| vLLM、TensorRT-LLM、Edge-LLM 如何选择 | `03` |
| 先读哪些开源项目 | `references/github-projects.md` |

## 范围

本仓库覆盖：

- Jetson Orin NX 16GB 的 CPU/GPU/DLA/视频引擎与统一内存约束；
- ONNX 到 TensorRT 的构建、动态形状、量化和执行上下文；
- DeepStream/GStreamer 的实时视频推理流水线；
- TensorRT Edge-LLM 的 LLM/VLM 端侧部署；
- Triton Inference Server 的模型管理、动态批处理和并发实验；
- CUDA streams/events/graphs、基础 kernel、TensorRT Plugin 和 profiling；
- 多模型队列、优先级、背压、准入控制、内存预算和可观测性；
- 延迟、吞吐、goodput、精度、功耗、温度和稳定性的实验设计。

## 非目标

当前阶段不追求：

- 从零手写完整推理框架；
- 把服务器集群技术照搬到 16GB 边缘板卡；
- 一开始就深入 PTX/SASS、CUTLASS 内核模板或编译器后端；
- 同时适配所有 Jetson、RK 和 S100 平台；
- 仅凭一次成功运行就声称“生产级”；
- 用峰值 TOPS 代替真实端到端性能测试。

## 与现有三个项目的关系

| 现有项目 | 可迁移经验 | 不能直接迁移的结论 |
| --- | --- | --- |
| `RK_LLM` | 模型导出、板端部署流程、接口和日志组织 | RK NPU 的模型格式、算子支持、性能数据 |
| `S100_VLA` | VLA/VLM 场景拆解、输入输出契约、验证思路 | S100 工具链与加速器行为 |
| `RK_S100_MFSysetem` | 多设备集成、消息链路、故障边界 | 单机 Orin 内部 GPU 争用和统一内存结论 |

三个项目继续在 RK 和 S100 上独立验证；本仓库只吸收可复用的方法论，并在 NVIDIA 平台重新建立证据。

## 最短实践路径

第一阶段不要直接做“三模型并发”：

1. 固化 JetPack/L4T 与功耗模式，保存版本清单。
2. 选择一个小型 CV ONNX，使用 `trtexec` 建立 FP32/FP16 单模型基线。
3. 加入 `tegrastats`、端到端延迟和结果正确性采集。
4. 用 Nsight Systems 找 CPU 等待、同步和数据搬运问题。
5. 加入第二个小模型，测隔离性能与并发退化。
6. 最后加入量化 LLM/VLM，设计优先级、背压和内存预算。

这一顺序能让每次增加复杂度时都有可对照的基线。

## 版本原则

- JetPack 提供的 CUDA、TensorRT、cuDNN 等系统包是 Orin 原生基线，不随意用桌面版包覆盖。
- TensorRT engine 不是通用模型文件；应在目标软件栈和目标 GPU 上构建并记录版本。
- 容器必须匹配 `linux/arm64`/aarch64 与 JetPack/L4T，不能直接使用 x86_64 镜像。
- 所有外部文档的兼容性结论都带核验日期；真正开工前仍要复核目标 tag/release。

## 外部资料使用原则

- 产品兼容性、安装和 API 语义优先采用 [NVIDIA 官方资料](references/official-resources.md)。
- [GitHub 项目清单](references/github-projects.md) 区分可直接实操、辅助工具和源码研读，不把活跃 `main` 当作稳定 release。
- 社区课程如 [lyy-ai/ai_infra](https://github.com/lyy-ai/ai_infra) 用于建立知识地图；其代码、环境、license 和平台结论需独立检查，不能覆盖官方支持矩阵。

## 后续代码目录触发条件

当第一个里程碑“单 CV 模型 + telemetry + benchmark harness”确定后，再添加：

```text
configs/       # 模型、功耗模式和实验场景
src/           # 运行时与采集代码
tests/         # 结果正确性和调度测试
benchmarks/    # 负载、原始结果和分析脚本
containers/    # 固定 digest 的 aarch64 环境
models/        # 仅保存 manifest，不提交大模型权重
```

现在不预建空目录，避免让仓库看起来已经存在未完成的实现。
