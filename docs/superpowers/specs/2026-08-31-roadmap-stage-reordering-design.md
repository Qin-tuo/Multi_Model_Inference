# AI Infra 路线图阶段重构设计

## 目标

将路线图从“按课程和技术类别排列”重构为“按知识依赖和能力闭环排列”。主线面向通用 GPU 推理与 Serving，LLM 推理必修；DeepStream、深度剪枝、Triton Language、驱动和编译器均按进入条件放入支线。

路线图只规定阶段顺序、学习内容、实践交付和完成条件，不规定周数或课时。

## 学习原则

1. 先建立最小但完整的知识体系，再在实际部署中深化，而不是只学习下一阶段立即调用的零散知识。
2. 一门课程可以按连续知识块拆分，但不能抽取互不相连的孤立片段。
3. 每个概念阶段都必须在后续工程阶段再次调用，形成“系统学习 -> 阶段项目 -> 间隔复习”的闭环。
4. 主线只保留通用 GPU 推理、模型部署、Serving 和 LLM 推理所需内容；方向性内容通过支线承载。
5. 参数量、FLOPs 或模型文件变小不能直接视为加速；量化、剪枝和稀疏化必须在目标硬件与实际 runtime 上测量。

## 新主线依赖

工程与工具基础 -> CUDA 与 GPU 性能 -> 深度学习基础原理 -> 模型压缩原理与设计 -> 模型结构、转换与推理引擎 -> 容器化与推理服务 -> Transformer 与 LLM 概念 -> LLM 推理与 Edge-LLM

## 八阶段设计

### 阶段一：工程与工具基础

学习内容：

- Python、NumPy、PyTorch tensor/module/inference；
- C++17、CMake、RAII、STL 和调试；
- Linux 用户态、Shell、进程、权限和 Git；
- Docker 入门：image、container、volume、端口、环境变量、基础 Dockerfile 和 ARM64 镜像识别。

完成条件：能独立构建、运行、调试并容器化一个 PyTorch 推理小程序。现有 `model-tools`、CPU GEMM 和 TCP 小服务继续保留。

### 阶段二：CUDA 与 GPU 性能

学习内容：

- grid/block/thread、Kernel 和内存层次；
- streams、events、pinned memory 和异步执行；
- transpose、reduction、GEMM 等并行模式；
- coalescing、bank conflict、occupancy、divergence 和 arithmetic intensity；
- Compute Sanitizer、Nsight Systems、Nsight Compute、PTX/SASS。

CUDA Developer Tools 只使用 NVIDIA 官方英文 playlist；不保留不可用的第三方中文镜像。

完成条件：能写出正确 Kernel，并用工具证据解释和优化瓶颈。项目 A 和项目 B 归属本阶段。

### 阶段三：深度学习基础原理

主课为 Deep Learning Specialization：

- Course 1 学习完整概念链；
- Course 2 只取初始化、正则化和优化算法原理，跳过调参实战；
- 参数、激活、损失、前向/反向、计算图、batch；
- 训练、验证、导出和推理的区别。

Course 3 跳过；Course 4 迁入视觉支线；Course 5 降为阶段七的查漏补缺，不作为主线必修。

增加一个有明确来源的定向补充：只查 [PyTorch `torch.nn` 官方参考](https://docs.pytorch.org/docs/stable/nn.html) 中的 `Linear`、`Conv2d`、`BatchNorm2d`、`Embedding`，并区分 weight、activation、gradient tensor 的角色。它不是第二门主课，也不提前替代阶段五的 D2L。

完成条件：能解释一个神经网络如何训练，以及部署阶段保留和移除了什么；必须提交书面计算图和训练 -> 验证 -> 导出 -> 推理的数据、状态、操作变化说明。PyTorch MLP 代码追踪只作可选验证。

### 阶段四：模型压缩原理与设计

学习内容：

- PTQ、QAT、calibration、scale/zero-point；
- per-tensor/per-channel、W8A8、W4A16；
- structured/unstructured pruning、稀疏化和剪枝后微调的原理与边界；
- 知识蒸馏；
- 参数量、FLOPs、内存带宽、算子支持、Engine tactic 与实际硬件收益之间的关系。

本阶段只负责通用量化、剪枝、稀疏化、蒸馏原理和验证设计；模型结构专题归阶段五，LLM 专属量化与缓存细节归阶段八。旧 Jetson Nano、旧 TAO 和旧安装命令不进入主线。

阶段五执行 FP32/FP16/INT8 量化 baseline；剪枝训练、2:4 稀疏和对应硬件实验只在模型压缩实战支线执行。

完成条件：能根据模型、硬件和精度要求提出量化或剪枝方案，并说明其风险和验证方法。

### 阶段五：模型结构、转换与推理引擎

D2L 作为连续的精选主课，不再标为“完整系列”。正式来源为 [官方课程页](https://courses.d2l.ai/zh-v2/)、B站官方 [“跟李沐学AI”系列](https://space.bilibili.com/1567748478/lists/358497?type=series)（mid `1567748478`，series `358497`，API 核验去除预告和课程安排后为 74 节官方课程）和 [中文教材](https://zh-v2.d2l.ai/)。学习内容：

- Module、参数、checkpoint 和模型结构读取；
- MLP、CNN、BatchNorm、ResNet；
- RNN/Seq2Seq、Attention、Transformer；
- tensor shape 和完整 forward 数据流；
- 使用现有 [《深度学习模型部署与剪枝优化实战》](https://www.bilibili.com/video/BV1Sw411y7Hs/) 的 P109-P121 轻量化 / MobileNet 单元作精选补充，学习 depthwise convolution、MobileNet 与轻量结构；
- PyTorch -> ONNX、ONNX Runtime 数值对齐；
- TensorRT Builder/Runtime、FP32/FP16/INT8、dynamic shape；
- plugin、Polygraphy 和 profiling；
- 异步预处理、数据搬运和 zero-copy。

D2L 的优化算法、完整训练、Kaggle、检测/分割、GAN、推荐系统和分布式训练不进入主线。

本阶段重新调用阶段四的量化设计，在 TensorRT 上完成 FP32/FP16/INT8 数值、精度、Engine 构建日志、实际 tactic、延迟、吞吐和内存 baseline；不要求执行剪枝或 2:4 稀疏。

GPU 预处理的必修实现是先按 [CV-CUDA Installation Guide](https://cvcuda.github.io/CV-CUDA/installation.html) 验证 Jetson Orin 支持或 source build，再交付 CV-CUDA pipeline。DALI 只作概念和条件比较：按 [DALI Support Matrix](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/support_matrix.html) 核对 Orin 的 CUDA 12.6 source-build-only 边界；只有在 JetPack 7.2.1 / CUDA 13.2.1 上实际验证源码构建和运行后才加入比较，不能假设 wheel 可用。

完成条件：能把陌生 PyTorch 模型转换为可复现的 TensorRT Engine，并交付精度、延迟、吞吐和内存报告。项目 C 归属本阶段。

### 阶段六：容器化与推理服务

学习内容：

- Docker multi-stage、Compose、network、healthcheck、GPU runtime 和 ARM64 镜像；
- HTTP 请求/响应、状态码、数据格式、超时和健康检查；
- gRPC `.proto`、stub、unary RPC、deadline 和错误码；
- Triton Inference Server 的 model repository 和 backend；
- dynamic batching、instance group、ensemble、metrics 和 Perf Analyzer；
- Triton 官方 HTTP client 和 unary gRPC client。

不要求自行实现完整 gRPC server。server/client/bidirectional streaming 和完整中文 gRPC 课程改为可选；只有明确的顺序、状态或长连接需求时才深入。

完成条件：能把 TensorRT 模型部署为可监控、可压测、可重复启动的服务，并解释 TensorRT runtime 与 Triton Server 的层级差异。项目 E 归属本阶段。

### 阶段七：Transformer 与 LLM 概念

主课固定为 CS224n Spring 2024 B 站合集：

- L7 Attention / LLM Introduction；
- L8 Self-Attention and Transformers；
- L9 Pretraining；
- L10 Post-training；
- L11 Natural Language Generation；
- 补充 tokenizer、embedding、encoder/decoder；
- BERT 与 GPT、pretraining 与 post-training 的区别。

D2L 的 Attention/Transformer 内容在此闭卷复习，不重新观看整课。若 RNN、Seq2Seq 和 Attention 前置不足，再回补 CS224n L5-L6。DLS Course 5 仅作为查漏补缺。

完成条件：能完整画出 Transformer 和生成过程，并解释训练资产如何成为可推理模型。

### 阶段八：LLM 推理与 Edge-LLM

学习内容：

- prefill/decode、KV cache；
- continuous batching、PagedAttention；
- TTFT、TPOT、tokens/s、吞吐、并发和排队；
- INT4、AWQ、GPTQ、SmoothQuant、LLM.int8()、量化格式与 runtime 兼容性；
- TensorRT-Edge-LLM、llama.cpp、MLC-LLM、ExecuTorch 等运行时；
- Orin 上的运行时选型、服务封装和性能测量。

CS336 只在此阶段使用模型结构、推理、评测和系统相关内容；Lecture 6 的 Triton/XLA 继续归入算子支线。

本阶段明确拥有 KV cache、AWQ、GPTQ、SmoothQuant 和 LLM.int8() 的方法差异、格式检查、内存预算与 runtime 支持矩阵验证。

完成条件：能在 Orin 上部署小模型服务，并给出运行时、量化、内存、功耗和性能选型依据。项目 F 归属本阶段。

## 可选支线

### 视觉流媒体支线

进入条件：完成阶段五。

内容：Jetson AI Fundamentals、摄像头与视频输入、GStreamer、DeepStream、NVMM/zero-copy、单路到多路、tracker、metadata、功耗与性能测量。DLS Course 4 和《深度学习模型部署与剪枝优化实战》的 GStreamer/DeepStream 单元放入此支线。项目 D 改为可选视觉项目。

### 模型压缩实战支线

进入条件：完成阶段五项目 C，并保留其未剪枝 TensorRT FP32/FP16/INT8 baseline。

内容：重访并扩展项目 C baseline，在 channel/filter pruning 与 Ampere 2:4 structured sparsity 中二选一；完成对应训练/微调、ONNX 导出、TensorRT tactic 检查，以及剪枝前后的精度、延迟、吞吐和内存对比。

### Triton Language、CUTLASS 和 CuTe 支线

进入条件：完成阶段二。Triton Language 是 Kernel DSL，与阶段六的 Triton Inference Server 明确分开。

### Linux Driver/BSP 支线

进入条件：完成阶段二。除非目标岗位明确要求，否则不进入主线。

### TVM/MLIR 支线

进入条件：先完成 Triton Language 或等价算子项目；它不是 TensorRT 前置课程。

## 文档级联调整

实现时必须同步更新以下位置，不能只移动八个阶段正文：

- 顶部目录、总览说明、主线依赖关系和八阶段地图；
- 各阶段目标、课程、大纲、方法、完成条件、项目和版本边界；
- 可选分支及其进入条件；
- 项目阶梯中项目 D 的可选属性，以及项目 E/F 的阶段归属；
- TensorRT/Triton/vLLM 概念澄清中的推荐顺序；
- 将“六周 CUDA 冲刺”改为不含时间估算的“CUDA 专项执行清单”，原“第 1–6 周”改为顺序步骤；
- 将“24 周执行节奏”改为不含周数的“阶段执行与复习原则”；
- 最终毕业项目：DeepStream 从必需组件改为视觉支线扩展；
- 一页式执行清单：按新主线排序，将 DeepStream 移入支线，并保留阶段一 CPU GEMM、Shell/TCP、阶段二 MNIST CUDA、阶段三书面计算图、阶段五 CV-CUDA baseline 等实际交付；
- 平台验证：CV-CUDA 是验证后必做路径，DALI 是 support-matrix 与源码验证通过后的条件比较；
- 能力定位和关键链接索引中的阶段引用；
- 全文所有“阶段二”到“阶段八”的旧交叉引用。

## 验证

完成文档重构后执行以下检查：

1. 搜索所有阶段编号和章节引用，逐项确认与新顺序一致。
2. 确认 DLS、D2L 和 CS224n 分别以阶段三、阶段五和阶段七为主落点；D2L 使用官方课程页、“跟李沐学AI”系列和中文教材，不使用无效 BVID。
3. 确认 DeepStream 不再出现在主线阶段地图或主线执行清单中。
4. 确认 Docker 在阶段一出现基础内容，在阶段六出现 Serving 内容。
5. 确认 Triton Server 位于阶段六，Triton Language 只位于支线。
6. 确认文档没有周数表、“24 周”、“六周 CUDA”或“第 N 周”式阶段安排残留。
7. 运行 Markdown 空白检查并审阅最终 diff，确保没有改变版本基线和无关资源。
8. 确认阶段四只含通用压缩原理/设计，MobileNet 在阶段五补充，LLM 专属量化在阶段八，剪枝执行在模型压缩实战支线。
9. 确认 CV-CUDA/DALI 平台门槛、项目 C baseline、执行清单与最终路线图一致。
