# 外部免费双语 AI Infra / CUDA / Jetson 学习路线

> 路线定位：这是本知识库的 **NVIDIA 生态主路线**。完成后再进入 [多生态总路线](EDGE_AI_MULTI_ECOSYSTEM_ROADMAP.md) 中的 [地瓜 BPU 路线](EXTERNAL_FREE_BILINGUAL_HORIZON_BPU_ROADMAP.md)，再学习 [瑞芯微双并行路线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)。
>
> 面向：Jetson Orin NX 16GB、CUDA/C++、TensorRT、DeepStream、Triton、边缘 LLM Serving
> 目标：从底层计算模型一路走到可测量、可部署、可服务化的 AI 推理系统
> 资料范围：真实存在的公开课程、官方文档、公开 GitHub 项目；不把当前文件夹中的内部笔记当作外部参考资料
> 版本核验日期：2026-08-21。版本敏感的命令和容器必须在执行前再次核对官方兼容矩阵。

---

## 0. 文档定位、目标画像与边界

### 0.1 路线定位与结论

你的主线不应该是把所有技术都学一遍，而应该是：

**Python + PyTorch + C/C++ + Linux 基础 → CUDA C++ / Kernel → 正确性与 Nsight 性能分析 → ONNX / TensorRT → Jetson / DeepStream → Triton / 服务化 → TensorRT-Edge-LLM / LLM Serving**

同时保留两条分支：

- **Linux 驱动 / BSP 分支**（分支 A）：适合想做 Jetson 底层、内核、设备树、驱动、系统启动和平台适配的人。
- **GPU 编译器 / Kernel DSL 分支**（分支 B/C）：CUDA 熟练后学习 Triton、CUTLASS/CuTe、TVM、MLIR，适合做算子、编译器和性能工程。

你现在先学 CUDA 是正确的起点。Linux 驱动很重要，但对于 AI 推理性能和部署这个目标，它不是第一门课；先具备读懂硬件行为、写 Kernel、做 Profile 和解释性能数据的能力，后续再进入驱动会更有效。

### 0.2 目标画像

这份路线按以下目标定制：

- 不是泛泛学习机器学习，而是进入 GPU 加速、AI 推理、边缘部署、Serving 和系统性能。
- 需要能自己写 CUDA Kernel，并能用工具解释为什么快或慢。
- 需要从模型导出一直做到 TensorRT Engine、视频流水线、服务 API 和监控。
- 需要中英文资料：英文资料作为权威主线，中文资料用于降低入门门槛和复习。
- 不购买课程；核心学习不依赖付费课程或付费云 GPU。
- 以本机 Orin NX 为真实部署目标，必要时把不适合 ARM64/Jetson 的实验放到 x86 GPU 或 Colab 上完成。

### 0.3 本机基线（硬件与版本）

根据本机环境检查结果：

- Jetson Orin NX 16GB 开发套件配置
- Ubuntu 24.04.4 LTS
- Linux kernel 6.8.12-tegra
- Jetson Linux / L4T R39.2.1
- JetPack 7.2.1
- CUDA 13.2.1、TensorRT 10.16.2、DeepStream 9.1（以本机实际安装包为准）
- GPU Compute Capability 8.7，属于 Orin 的 Ampere 架构（sm_87）
- 8 核 Cortex-A78AE
- 16GB unified memory，CPU 和 GPU 共享内存
- 1TB Samsung 990 PRO NVMe
- 当前功耗模式 25W

硬件与版本总入口：

- [JetPack SDK Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Documentation](https://docs.nvidia.com/jetson/index.html)
- [CUDA GPUs and Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [Jetson AI Lab Tutorials](https://www.jetson-ai-lab.com/tutorials/)

### 0.4 由硬件带来的学习边界

- Orin 是 SM 8.7；先围绕 Ampere 的内存访问、共享内存、Warp、Tensor Core、FP16/INT8 学习。
- Orin 是 ARM64 和 unified memory 平台；很多 x86_64 容器、Python wheel、Triton 示例不能直接照搬。
- 16GB 不是独立显存；模型、解码缓冲区、TensorRT Workspace、系统和服务会共同占用内存。
- Jetson 设备上的实际性能受功耗、温度、时钟和内存带宽影响，不能只看桌面 GPU 的教程结果。
- 最新的通用文档可能已经超过本机 JetPack 版本；安装和调试时优先使用 JetPack 自带的 CUDA、TensorRT、DeepStream 和对应版本文档。

### 0.5 资料使用规则：标签、不付费边界、视频优先、学习比例

**标签**

- **EN / Official**：英文官方资料，版本和 API 以此为准。
- **EN / Open**：英文公开课程或社区项目，适合跟练。
- **中文 / Official**：NVIDIA 中文站、中文文档或官方中文课程入口。
- **中文 / Supplement**：中文翻译、字幕或辅助讲解；代码和版本必须回到官方仓库核对。
- **Project**：必须实际运行、修改和测量的公开项目。

**不付费的边界**

主路线只使用公开免费内容。NVIDIA DLI 的课程目录中同时存在免费短课和付费课，不能把能看到课程页面误认为课程实验永久免费。云端 GPU 额度也可能变化，因此不把付费云作为前置条件。

中文资料有时是旧版本或第三方转载。遇到版本冲突时按以下优先级：

1. 本机 JetPack / L4T 版本对应的 NVIDIA 官方文档。
2. NVIDIA 官方 GitHub 仓库的 release、branch 和 README。
3. 官方、大学或原作者发布的完整课程视频。
4. 中文完整课程、翻译或字幕。

**建议学习比例：30% 看完整课程，55% 写代码和改项目，15% 查文档并写实验记录。**

**视频优先，不从文档第一页开始**

每一个阶段都按同一套方式执行：

1. 从该阶段给出的 1-2 门完整课程中选择 **一门主课**。
2. 每看完一个可编码章节，当天复现代码；不要等整门课看完才动手。
3. 完成该阶段唯一指定的阶段项目，并达到停止条件。
4. 只有遇到 API、版本、安装或指标定义问题时，才定点查官方文档。

第二门课是补课，不是重复刷课。主课听懂且项目通过，就直接进入下一阶段；只有主课缺少某个主题，才去第二门课找对应章节。

这里所说的“完整课程”指有连续教学结构、明确起点和终点的课程、系列课或完整 workshop。宣传片、零散大会演讲和只演示一次命令的视频，不计为主课。中文镜像若只是同一门英文课的翻译，会明确标成“同课中文镜像”，不会伪装成第二门独立课程。

官方文档在这份路线中只承担三种职责：

- 查本机版本对应的安装和兼容矩阵。
- 把旧视频中的 API 映射到当前 API。
- 核对 profiler 指标、配置字段和边界条件。

不要求通读 CUDA、TensorRT、DeepStream 或 Triton 的整本手册。

---

## 1. 总览：学习路径与阶段地图

你当前可以直接从 **阶段三（CUDA C++）** 开始，但阶段二（Python、PyTorch 与 C++ 基础）是必修底座。阶段二里的 Linux 用户态可以按经验补课；当你在指针、引用、编译链接、虚拟环境、NumPy、tensor、Shell、进程或权限上卡住时，先回补对应章节。**阶段一（模型概念底座）是可选前置**：只在你想先建立“模型 / Transformer 是什么”的直觉时进入，目标是能看懂后面要部署的模型结构，不要求写训练项目；已有模型基础的人可跳过，直接从阶段二/三开始。

主线依赖关系：

**Python / PyTorch / C++ / Linux → CUDA 编程 → GPU 架构与 Kernel 优化 → Sanitizer / Nsight / PTX-SASS → ONNX / TensorRT → Jetson / DeepStream → Docker / gRPC → Triton Server → LLM Inference → Orin Edge LLM**

高级分支（见第 10 章）：

**Linux 驱动（分支 A）**，或 **Triton Language → CUTLASS / CuTe（分支 B） → TVM / MLIR（分支 C）**

八阶段地图一览：

| 阶段 | 原节点 | 属性 | 唯一阶段项目 |
|---|---|---|---|
| 阶段一 · 模型概念底座 | -1 | 可选前置，概念层 | 不写训练项目；可选 PyTorch 复现 Transformer 前向 |
| 阶段二 · 基础工具 | 0 + 1 | 基础必修 | model-tools + C++17/CMake CPU GEMM + Shell/TCP 服务 |
| 阶段三 · CUDA 与 GPU 性能 | 2 + 3 + 4 | 必修 | vector add/SAXPY/transpose/reduction/tiled GEMM + Sanitizer/Nsight 报告 |
| 阶段四 · 量化理论基础 | 4.5 | 概念必修 | 一个 INT8 小模型 calibration 对比 |
| 阶段五 · 模型转换与预处理 | 5 + 5.5 | 部署必修 | PyTorch→ONNX→TensorRT + DALI GPU pipeline |
| 阶段六 · Jetson 与视频推理 | 6A + 6B | 视觉方向必修 | 单路→4 路 DeepStream pipeline |
| 阶段七 · 容器与服务化 | 7A + 7B + 8 | Serving 必修 | TensorRT backend Triton 服务 + gRPC/HTTP |
| 阶段八 · LLM 推理与服务 | 9 + 9.5 + 10 | 本机目标必修 | Orin 上 Edge-LLM 小模型服务 + 选型表 |

---

## 2. 阶段一 · 模型概念底座（原节点 -1，可选前置）

### 目标 / 为什么在这个阶段

这个节点不是主线必需，而是给“想先搞懂要部署的模型长什么样”的人准备的。它和主线目标是两层：主线是部署 / 推理 / infra，本节点是模型理解。因此原则是**看概念、不写训练项目**，避免把时间花在调参、分布式训练、DeepSpeed 上。

### 课程与资料

- **李沐 动手学深度学习 D2L v2**（中文 / Open，含 PyTorch 代码，完整系列）：[B站 BV1Z5411n7RB](https://www.bilibili.com/video/BV1Z5411n7RB/)；教材见 [d2l.ai](https://d2l.ai/)。
- **Deep Learning Specialization**（EN / Andrew Ng，概念向，跳过调参实战）：[deeplearning.ai](https://www.deeplearning.ai/courses/deep-learning-specialization/)。
- **Stanford CS224n**（EN / University，只取 **L10–L13、L15**：Transformer / 自注意力、BERT / 预训练、上下文表示、NLG；其余讲次跳过）：[web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/)。

##### 当阶段课程大纲（★ = 本阶段必修）

- **李沐 动手学深度学习 D2L v2（B站）** 大纲：
  - ★ 线性回归与 softmax 回归（建立“模型 = 可学习参数 + 前向计算”直觉）
  - ★ 多层感知机 MLP（层、激活、前向/反向的概念）
  - ★ 卷积神经网络 CNN：从 LeNet 到 ResNet 的结构概念
  - ★ 循环神经网络 RNN / LSTM（序列建模用途）
  - ★ 注意力机制与 Transformer（Q/K/V 与 attention 的 shape 流）
  - 模型选择 / 过拟合 / 优化算法（可选，略读）
  - 目标检测 / 语义分割 / GAN / 分布式训练（偏离主线，跳过）
- **吴恩达 Deep Learning Specialization** 大纲（只取概念章）：
  - ★ Course 1 神经网络与深度学习（NN/CNN 是什么）
  - ★ Course 4 卷积神经网络
  - ★ Course 5 序列模型中的 RNN/注意力概念
  - Course 2/3 调参与 ML 项目实战（跳过）
- **CS224n** 大纲（只取 **L10–L13、L15**，其余跳过）：
  - ★ L10 注意力机制 Attention、Self-Attention
  - ★ L11 Transformer 架构、BERT 与预训练
  - ★ L12 Finetuning、Prompting、NLG
  - ★ L13 上下文表示（ELMo / BERT / T5）
  - ★ L15 自然语言生成与预训练（衔接阶段八的 LLM 原理）
  - L1–L9、L14、L16–L19（词向量 / 句法分析 / 共指 / 解析等 NLP 任务）：全部跳过

### 学习方法与停止条件

- **怎么选课**：主课用李沐 D2L v2（有 PyTorch 代码，与阶段二的 PyTorch 线重合度最高）；吴恩达 Deep Learning Specialization 只取“神经网络 / CNN / RNN 是什么”的概念章，跳过调参与结构化 ML 项目实战；CS224n 只取 **L10–L13、L15**，其余词向量 / 句法分析 / 共指 / QA 等 NLP 任务讲次全部跳过。
- **明确不学**：吴恩达 Machine Learning Specialization（经典监督 / 无监督）偏离最高、回报最低，可大幅压缩或跳过；不把 fast.ai、LangChain / RAG / Agents 短课算进本节点。
- **看完即做**：只做一个轻量练习——用 PyTorch 复现一个 Transformer 前向，能写出 Q/K/V 的 shape 变化和 attention 输出的 shape。
- **停止条件**：能口头解释 tensor / 层 / 前向传播、CNN 与 RNN 的用途差异、attention 里 Q/K/V 各自的作用；知道“训练出来的模型”和“部署时跑的引擎”不是同一回事。达到后即可进入阶段二或直接到阶段三。
- **CS336 不在这里学**：从头训 LLM 属于阶段八，按主线节奏走，不要提前挪到概念底座。

### 阶段项目

不写训练项目；可选：用 PyTorch 复现一个 Transformer 前向并解释 Q/K/V 与 attention 的 shape 流。

### 版本 / 边界注意

本阶段纯概念，无版本绑定；但 CS224n 只看 L10–L13、L15，不要顺着完整课程刷到训练实战。

---

## 3. 阶段二 · 基础工具（原节点 0 Python/PyTorch/C++ + 节点 1 Linux 用户态）

### 目标 / 为什么在这个阶段

Python 在这条路线中不是“会一点脚本”即可，而是模型导出、量化校准、数据集处理、benchmark、日志分析、ONNX 检查和服务客户端的主要胶水语言。PyTorch 则是连接训练侧模型资产与部署侧 ONNX/TensorRT 的必修桥梁。C++ 是写 Kernel、服务与高性能组件的底座。Linux 用户态是开发、排障与后续驱动分支的前提。阶段二必须同时完成 Python、PyTorch、C++ 与 Linux 四条基础线。

### 课程与资料

**Python：**

- 英文主课：[Python Full Course for free](https://www.youtube.com/watch?v=ix9cRaBkVe0)（freeCodeCamp，完整零基础课程）
- 中文补课：[黑马程序员全套 Python 基础教程](https://www.bilibili.com/video/BV1o4411M71o/)
- 官方资料：[Python 官方教程](https://docs.python.org/3/tutorial/)、[venv 官方文档](https://docs.python.org/3/library/venv.html)、[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)、[NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

**PyTorch：**

- 英文主课：[PyTorch for Deep Learning & Machine Learning - Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA)（freeCodeCamp）
- 中文补课：[PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/)（小土堆，已完结系列；配套代码见 [GitHub](https://github.com/xiaotudui/pytorch-tutorial)）
- 官方校正：[Introduction to PyTorch on YouTube](https://docs.pytorch.org/tutorials/beginner/introyt/)、[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)、[PyTorch ONNX Tutorials](https://docs.pytorch.org/tutorials/beginner/onnx/index.html)

**C++：**

- 英文主课：[C++ Programming Course: Beginner to Advanced](https://www.youtube.com/watch?v=8jLOx1hD3_o)（freeCodeCamp）
- 中文补课：[黑马程序员 C++ 零基础到项目开发](https://www.bilibili.com/video/BV1ZH4y137ws/)
- 辅助资料：[LearnCpp](https://www.learncpp.com/)

**Linux 用户态：**

- 英文主课：[Introduction to Linux: Full Course](https://www.youtube.com/watch?v=sWbUDq4S6Y8)（freeCodeCamp，约 6 小时）
- 中文补课：[尚硅谷 Linux 应用层开发](https://www.bilibili.com/video/BV1DJ4m1M77z/)（文件 IO、进程线程、Socket、epoll 完整系列）
- 辅助资料：[MIT Missing Semester](https://missing.csail.mit.edu/)、[Pro Git](https://git-scm.com/book/en/v2)、[OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)、[LFS101 Introduction to Linux](https://training.linuxfoundation.org/training/introduction-to-linux/)

##### 当阶段课程大纲（★ = 本阶段必修）

- **Python 主课（freeCodeCamp / 黑马）** 大纲：
  - ★ 语法、条件/循环、函数、作用域
  - ★ 字符串、列表/字典/集合/元组
  - ★ 类与 dataclass、异常、模块/包
  - ★ 文件、路径、JSON/CSV
  - ★ 虚拟环境 venv、pip、requirements、版本锁定
  - ★ 命令行：argparse、环境变量、logging、subprocess
  - ★ NumPy：dtype/shape/stride/broadcast/向量化/矩阵乘
  - ★ OpenCV/PIL 图像读写、resize、批量遍历
  - Web/爬虫/数据分析（跳过）
- **PyTorch 主课（freeCodeCamp / 小土堆）** 大纲：
  - ★ tensor 的 shape/dtype/device/layout/contiguous
  - ★ 索引与 reshape/permute、nn.Module/forward
  - ★ eval()、no_grad()/inference_mode()
  - ★ state_dict、checkpoint 加载、CPU/CUDA 转移
  - ★ Dataset/DataLoader
  - ★ ONNX 导出与 dynamic shape、数值对齐
  - 完整训练大型模型（略，本节点只做推理/导出）
- **C++ 主课（freeCodeCamp / 黑马）** 大纲：
  - ★ pointer、reference、stack/heap
  - ★ RAII、class、template、STL、lambda
  - ★ CMake、gdb、编译与链接
  - 复杂模板元编程/全栈（跳过）
- **freeCodeCamp Introduction to Linux（约 6 小时）** 大纲：
  - ★ Linux 文件系统、Shell 与导航
  - ★ 文件/目录操作、权限（chmod/chown）、用户与组
  - ★ 管道、重定向、进程与信号、日志
  - ★ 包管理、SSH、基础网络
  - ★ `gdb` / `strace` 排查入门
  - ★ Shell 脚本与 cron（写监控脚本）
  - 运维认证/Linux 发行版内部（跳过）
- **尚硅谷 Linux 应用层开发** 大纲：
  - ★ 文件 IO（open/read/write、缓冲）
  - ★ 进程与线程（fork、pthread）
  - ★ Socket 网络编程
  - ★ epoll 多路复用
  - 内核/驱动部分（留到分支 A）

### 学习方法与停止条件

- **Python 必修范围**：语法/函数/作用域 → 文件/JSON/模块/包 → venv/pip/requirements → pathlib/argparse/logging/subprocess/glob/hashlib/time → NumPy dtype/shape/stride/broadcast/向量化/矩阵乘 → OpenCV/PIL 读写/resize/批量 → PyTorch tensor/Dataset/DataLoader/eval/no_grad/checkpoint/ONNX 导出 → pytest/断言/随机种子/类型标注。明确不把 Django/Flask 全栈、爬虫、复杂异步 Web、数据分析可视化、分布式训练、DeepSpeed 作为前置。
- **PyTorch 停止条件**：能加载公开 checkpoint 在固定输入上稳定得结果；能解释一次 shape/dtype/device/layout 错误；能用 `eval()` 和 `inference_mode()` 完成推理并正确处理 warmup 与 CUDA 同步；能把 PyTorch 模型导出 ONNX 并与 PyTorch 输出做容差比较；能说明训练、模型导出、TensorRT 构建和板端 Runtime 各自处于哪一层。
- **C++ 停止条件**：能解释值/引用/指针、对象生命周期、编译与链接，并能定位一次段错误。只抓 pointer/reference/stack-heap/RAII/class/template/STL/lambda/CMake/gdb；不要两门都刷。
- **Linux 停止条件**：能使用权限、管道、重定向、进程信号、日志、SSH、`gdb`、`strace`；知道 syscall、用户态和内核态的边界。CentOS 视频中的 `yum`、旧网络服务命令不复制到本机 Ubuntu 24.04；只迁移稳定概念。
- **阶段二基础停止条件**：能从零创建隔离环境并在另一台机器按文档复现；能读写常见模型/图像/JSON 文件；能解释 NumPy 的 shape/dtype/stride 与 PyTorch tensor 的关系；能把一次模型导出、校验和 benchmark 写成命令行工具；能定位依赖版本、路径、dtype 或 shape 错误。

### 阶段项目

在阶段二建立一个独立的 `model-tools` 目录，至少包含：

- `venv` 创建、依赖文件和一键运行说明；
- 用 `argparse` 扫描图片/视频数据集，生成带 hash 的 manifest；
- 用 NumPy 实现 CPU GEMM，并与 PyTorch 输出和耗时对齐；
- 读取图像并统一完成 RGB/BGR、resize、dtype 和 batch 处理；
- 导出一个简单 PyTorch 模型到 ONNX，并用 ONNX Runtime 做结果校验；
- 用 `subprocess` 调用外部 benchmark，输出 JSON/CSV；
- 用 logging 记录版本、输入 shape、随机种子和失败样本；
- 至少 5 个 pytest，覆盖空目录、错误路径、shape 和数值容差。

同时完成 C++17 + CMake CPU 矩阵乘法，随机输入校验并输出耗时；以及 Shell 监控脚本（采集 CPU、内存、温度、进程状态）+ 一个可并发处理请求的 C/C++ TCP 小服务。

### 版本 / 边界注意

本阶段主要是语言与用户态，无强版本绑定；但后续所有容器/Python wheel 都要面向 ARM64/JetPack 7.2.1 验证，不要在本阶段养成只看 x86 教程的习惯。

---

## 4. 阶段三 · CUDA 与 GPU 性能（原节点 2 CUDA C++ + 节点 3 架构与优化 + 节点 4 工具/Sanitizer/Nsight/PTX）

### 目标 / 为什么在这个阶段

CUDA 是整条路线的性能底座。阶段三把“写 Kernel → 解释性能 → 证明正确性”三件事串起来：先写对并跑通（节点 2），再用架构与优化知识解释快/慢（节点 3），最后用 Sanitizer/Nsight/PTX/SASS 形成“证据 → 假设 → 修改 → 复测”闭环（节点 4）。不把视频看完当作完成，必须以可复现的 Kernel 和性能报告为验收。

### 课程与资料

**节点 2 CUDA C++：**

- 主课：[NVIDIA Fundamentals of Accelerated Computing with Modern CUDA C++](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)（EN / Official，完整系列）；同课中文镜像 [BV1QvSKB4EMr](https://www.bilibili.com/video/BV1QvSKB4EMr/)
- 补课：[CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)（EN，freeCodeCamp，约 12 小时），代码见 [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)
- 官方实验：[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)、[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Samples](https://github.com/NVIDIA/cuda-samples)、[CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)

**节点 3 GPU 架构与 Kernel 优化：**

- 主课：[Stanford CS149 Parallel Computing 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)（EN / University，完整课程；最新讲次也可看 [CS149 fall25](https://gfxcourses.stanford.edu/cs149/fall25/)）
- 中文补课：[大规模并行处理器编程实战](https://www.bilibili.com/video/BV1gz421o7uH/)（PMPP 7 讲完整翻译课）
- 文档：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

**节点 4 正确性与性能工具：**

- 主课：[NVIDIA CUDA Developer Tools Tutorials](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR)（EN / Official，7 讲）；同课中文镜像 [BV14RU6BmE5u](https://www.bilibili.com/video/BV14RU6BmE5u/)
- 进阶：[GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)（EN / Open，[GPU Mode GitHub](https://github.com/gpu-mode)）
- 文档：[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)、[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)、[CUDA Toolkit 中文入口](https://developer.nvidia.cn/cuda-toolkit)

##### 当阶段课程大纲（★ = 本阶段必修）

- **NVIDIA Modern CUDA C++（官方 playlist）** 大纲：
  - ★ 环境搭建与第一个 Kernel（write/compile/run GPU code）
  - ★ CUDA Made Easy：用并行算法/标准库算法加速（执行空间、内存空间、kernel fusion）
  - ★ 线程层次 grid/block/thread 与自定义 Kernel（SIMT 模型、shared memory、cooperative algorithms）
  - ★ unified/managed memory 与内存迁移优化
  - ★ CUDA Streams 异步与并发、CUDA Events
  - ★ 用 Nsight Systems 做可视化 profiling
- **freeCodeCamp CUDA Course（约 12 小时）** 大纲：
  - ★ C/C++ review、setup、first kernels
  - ★ global/shared/register memory、error check
  - ★ streams、events、matrix multiplication
  - ★ PyTorch extension / Triton 章（仅了解，深入留到分支 B）
- **Stanford CS149 Parallel Computing 2023** 大纲：
  - ★ 并行计算导论与 work distribution
  - ★ GPU 体系结构（SM、warp、内存层次）
  - ★ CUDA 编程模型
  - ★ 并行模式：transpose / reduction / scan
  - ★ 内存与局部性（coalescing、bank conflict）
  - ★ 调度、同步与性能模型（occupancy、divergence、arithmetic intensity）
  - SIMD/向量化、FPGA 等非 GPU 章（可选）
- **大规模并行处理器编程实战 PMPP（B站 7 讲）** 大纲：
  - ★ 并行硬件与 CUDA 线程层次
  - ★ 内存层次与访问优化
  - ★ transpose / reduction / GEMM 实战
  - ★ 性能分析与案例
- **NVIDIA CUDA Developer Tools Tutorials（7 讲）** 大纲（固定顺序：Sanitizer → nsys → ncu → cuobjdump）：
  - ★ Nsight Tools 生态总览
  - ★ Intro to Nsight Systems（timeline、CPU/GPU 活动、内存拷贝）
  - ★ Intro to Nsight Compute（kernel 指标、guided analysis）
  - ★ Compute Sanitizer（memcheck / racecheck，越界与 race 检测）
  - ★ cuda-gdb 调试
  - ★ PTX/SASS 与 Binary Utilities（cuobjdump / nvdisasm）
  - ★ 进阶 profiling 与 case study
- **GPU MODE Lectures** 大纲：
  - ★ 与 Sanitizer/Nsight/SASS/算子性能相关的讲次
  - Tensor Core、FlashAttention、算子实现（按需）
  - 其它高级系统讲次（留到分支 B/C）

### 学习方法与停止条件

- **节点 2 看完即做**：vector add、SAXPY、二维矩阵加法；每个都有 CPU reference、边界输入、CUDA error check 和事件计时。停止条件：能不看答案写出索引，解释 grid/block/thread、异步 launch、global/shared/register/unified memory。
- **节点 3 看完即做**：naive/tiled transpose、naive/tree reduction、naive/tiled GEMM，并与 cuBLAS 比较。停止条件：能用 coalescing、bank conflict、occupancy、divergence、arithmetic intensity 解释性能，而不是只说“GPU 更快”。
- **节点 4 固定工具顺序**：`compute-sanitizer` → `nsys` → `ncu` → `cuobjdump` / `nvdisasm`。看完即做：故意制造越界或 race 并修复；导出系统 timeline；对唯一热点 Kernel 采集 memory、SOL、occupancy、warp stall；反汇编前后两个版本。停止条件：交付一份“证据 → 假设 → 修改 → 复测”的性能报告，且 Sanitizer 干净。
- **Orin 边界**：围绕 Ampere `sm_87` 实验；Hopper/Blackwell 的 TMA、WGMMA 示例只看概念，不作为本机验收。

### 阶段项目

vector add、SAXPY、2D add 和 CPU reference（节点 2）；transpose、reduction、tiled GEMM、cuBLAS 对比（节点 3）；Sanitizer + `nsys` + `ncu` + PTX/SASS 优化报告（节点 4）。维护一个自己的 CUDA 实验仓库，每个例子有 CPU reference、正确性测试、数据规模和性能记录。

### 版本 / 边界注意

本机 CUDA 13.2.1；编译确认目标架构为 Orin 的 sm_87。Nsight 的 target/host 分工：Jetson 采集，x86 host 分析通常更方便；采集前固定功耗模式、输入规模、频率和温度条件。不照抄 H100 的 sm_90、TMA、WGMMA 示例。

---

## 5. 阶段四 · 量化理论基础（原节点 4.5：MIT 6.5940 EfficientML 精译版）

### 目标 / 为什么在这个阶段

这个节点只补“为什么量化、怎么量化”的理论，不要求写训练代码；它支撑阶段五的 INT8、阶段八的 INT4/AWQ，避免只会抄量化命令而不懂取舍。先于阶段五/阶段八学习。

### 课程与资料

- 主课：[MIT 6.5940 EfficientML 精译版（B 站）](https://www.bilibili.com/video/BV1c8wNe1ErX)（中文 / MIT，免费完整课程，含中英字幕；原版见 [efficientml.ai](https://efficientml.ai/)）
- 落地文档：[NVIDIA TensorRT INT8/PTQ 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- 论文：[AWQ](https://arxiv.org/abs/2306.00978)、[GPTQ](https://arxiv.org/abs/2210.17323)、[SmoothQuant](https://arxiv.org/abs/2211.03602)、[LLM.int8()](https://arxiv.org/abs/2208.07339)
- 工具文档：[Hugging Face Optimum 量化文档](https://huggingface.co/docs/optimum)

##### 当阶段课程大纲（★ = 本阶段必修）

- **MIT 6.5940 EfficientML 精译版（B站）** 大纲：
  - ★ Lecture 1 课程导论与高效 AI 计算动机
  - 深度学习基础（★ 快速回顾）
  - ★ Lecture 量化 I：PTQ、线性/k-means 量化、per-tensor vs per-channel、calibration
  - ★ Lecture 量化 II：QAT、高级 PTQ、outlier 与 clipping
  - 剪枝 / NAS / 知识蒸馏（可选，概念了解）
  - ★ 高效 LLM：KV cache、W4/W8、GPTQ/AWQ/SmoothQuant/LLM.int8()
  - 端侧部署（★ 与本路线 Edge 目标相关）

### 学习方法与停止条件

- **必修关键字**：PTQ vs QAT、calibration（min-max / entropy / percentile）、per-tensor vs per-channel、weight-only（W4 / W8）vs weight-activation（W8A8）、scale/zero-point、outlier 与 clipping、GPTQ（基于 Hessian）vs AWQ（保护显著权重）vs SmoothQuant（迁移激活 outlier 到权重）。
- **看完即做**：不写训练；只做一个轻量实验——用 TensorRT 或 PyTorch 对一个 INT8 小模型做 calibration，对比“有校准 / 无校准”的精度 delta，并解释 per-channel 为何通常优于 per-tensor。
- **停止条件**：能解释 INT8 精度损失的来源、为何 LLM 需要 W4/AWQ 而非简单 W8A8、calibration 数据量与分布对结果的影响；知道阶段八里 FP16→INT4 的每一步在解决什么问题。

### 阶段项目

不写训练项目；可选：用 TensorRT/PyTorch 把一个 INT8 小模型做 calibration 并解释校准对精度的影响；对比 per-tensor 与 per-channel。

### 版本 / 边界注意

本阶段纯理论，但 calibration 工具链以本机 TensorRT 10.16.2 为准；论文只按需读摘要与动机。

---

## 6. 阶段五 · 模型转换与预处理（原节点 5 ONNX/TensorRT + 节点 5.5 DALI/CV-CUDA）

### 目标 / 为什么在这个阶段

把训练侧模型资产真正变成可在 Orin 上跑的 Engine，并解决最容易成为吞吐瓶颈却被忽略的预处理。阶段五 = ONNX/TensorRT 转换（节点 5）+ GPU 预处理（节点 5.5），二者共同决定端到端可部署吞吐。

### 课程与资料

**节点 5 ONNX 与 TensorRT：**

- 主课：[NVIDIA TensorRT 教程 4 部分](https://www.bilibili.com/video/BV15Y4y1W73E)（中文 / NVIDIA Official，完整系列）
- 补课：[Inference Optimization with NVIDIA TensorRT](https://www.youtube.com/watch?v=UnIuMXGylfY)（EN / NCSA，完整 workshop）
- 文档：[TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[TensorRT 10.x Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/quick-start-guide.html)、[TensorRT GitHub](https://github.com/NVIDIA/TensorRT)、[Torch-TensorRT](https://github.com/pytorch/TensorRT)
- 中文入口：[TensorRT 中文入门](https://developer.nvidia.cn/tensorrt-getting-started)、[NVIDIA 中文超级训练](https://www.nvidia.cn/developer/online-training/super-training/)（只看概念，命令以英文最新文档为准）

**节点 5.5 GPU 预处理与 DALI：**

- 主课文档：[NVIDIA DALI 文档与 Tutorials](https://docs.nvidia.com/deeplearning/dali/)、[DALI GitHub](https://github.com/NVIDIA/DALI)
- 对比库：[CV-CUDA](https://github.com/CVCUDA/CV-CUDA)（推理专用预处理 GPU 加速）
- 主课视频（需免费注册 NVIDIA Developer 后观看）：[GTC 2020 “Fast Data Pre-Processing with NVIDIA Data Loading Library (DALI)”](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21139)
- 补充免费视频（约 53 分钟，无需登录）：[NVIDIA DALI Data Loading Library 实战讲解](https://www.youtube.com/watch?v=PTWER9HIVHM)

##### 当阶段课程大纲（★ = 本阶段必修）

- **NVIDIA TensorRT 教程 4 部分（B站）** 大纲：
  - ★ builder / runtime / parser 工作流
  - ★ 精度：FP32 / FP16 / INT8 与 calibration
  - ★ dynamic shape 与 optimization profile
  - ★ plugin 与 Polygraphy
  - ★ 多流与性能分析（以当前 TensorRT 10.x 文档为准）
- **NCSA Inference Optimization with TensorRT** 大纲：
  - ★ 环境准备与 ONNX 导入
  - ★ builder 配置、精度校准
  - ★ 性能分析与部署
- **主课视频：GTC 2020 DALI talk** 大纲：
  - ★ 为什么数据预处理成为 GPU 吞吐瓶颈（CPU 空泡）
  - ★ DALI 是什么：GPU 加速的数据加载与增强库
  - ★ DALI Pipeline 与 Operator 模型
  - ★ 数据读取器：COCO / LMDB / TFRecord / Webdataset / NumPy
  - ★ 解码与增强算子：nvJPEG、resize、normalize、color、augmentation gallery
  - ★ 与 PyTorch / TensorFlow DataLoader 集成（iterator）
  - ★ 异步 pipeline、prefetch 与 DALI↔TensorRT 直连
  - ★ 性能分析：NVTX + Nsight Systems 看预处理空泡、多 GPU 数据并行
  - ★ 案例：ResNet50 with DALI、推理预处理加速
- **补充免费视频：DALI 实战讲解（YouTube）** 大纲（decode→resize→normalize pipeline 演示）：★ 全部章节本阶段必修。
- **辅助文档（非视频）：** NVIDIA DALI 官方文档与 Tutorials（https://docs.nvidia.com/deeplearning/dali/）、[DALI GitHub](https://github.com/NVIDIA/DALI)；CV-CUDA（https://github.com/CVCUDA/CV-CUDA）用于推理专用预处理（crop/normalize/color 的 GPU 算子），★ 作为对比与扩展阅读。

### 学习方法与停止条件

- **节点 5 看完即做**：PyTorch 导出 ONNX、ONNX Runtime 对齐、`trtexec` 构建 Engine、FP32/FP16/INT8、dynamic shape、warmup 后 benchmark。停止条件：能解释 Engine 为什么与目标 GPU/版本绑定，能报告 accuracy delta、p50/p95、throughput 和峰值内存。
- **节点 5.5 必修关键字**：decode 卸载、pinned memory、异步 pipeline、zero-copy、host↔device 拷贝、batch 拼接、DALI ↔ TensorRT 直连、CPU 预处理导致的 GPU 空泡。看完即做：用 DALI 搭一个 decode + resize + normalize 的 GPU pipeline，喂给 TensorRT/PyTorch；与 OpenCV/CPU 版本在同一输入上对比端到端吞吐，并用 `nsys` 看预处理阶段是否还占 GPU 空泡。停止条件：能指出当前 pipeline 的预处理是否瓶颈、是否应上 DALI/CV-CUDA、pinned memory 与异步带来的提升；知道 DeepStream（阶段六）的 NVMM 零拷贝正是同一思想的硬件实现。

### 阶段项目

PyTorch → ONNX → TensorRT，FP32/FP16/INT8 对比（节点 5）；用 DALI 搭 decode + resize + normalize pipeline，与 OpenCV/CPU 对比吞吐，测量 pinned memory、异步与 zero-copy 对预处理瓶颈的影响（节点 5.5）。

### 版本 / 边界注意

**强版本警告**：公开视频主要基于 TensorRT 8.x，本机是 TensorRT 10.16.2。视频只学概念；实现必须以 [TensorRT 10.x Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/quick-start-guide.html) 和当前 samples 为准，不照抄旧 bindings 或旧 plugin API。DeepStream 9.1 与本机 JetPack 7.2.1 对应；旧视频安装命令不可直接复制。

---

## 7. 阶段六 · Jetson 与视频推理（原节点 6A Jetson + 节点 6B GStreamer/DeepStream）

### 目标 / 为什么在这个阶段

把前面构建的 Engine 落到真实 Orin 设备上，并处理摄像头/视频流。阶段六 = Jetson AI 基础（节点 6A）+ GStreamer/DeepStream 视频流水线（节点 6B），目标是从单路扩展到多路并测量真实 FPS/延迟/功耗。

### 课程与资料

**节点 6A Jetson AI 基础：**

- 主课：[NVIDIA Jetson AI Fundamentals](https://www.youtube.com/playlist?list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8)（EN / Official，完整系列）；同课中文镜像 [BV1EGSmBWErR](https://www.bilibili.com/video/BV1EGSmBWErR/)
- 中文补课：[NVIDIA Jetson 边缘 AI 快速上手系列](https://www.bilibili.com/video/BV1yEzBYQEMt/)（Seeed Studio，完整系列）
- 项目起点：[dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)

**节点 6B GStreamer 与 DeepStream：**

- 主课：[Create Vision AI Applications With DeepStream](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81879/?playlistId=gtc26-computer-vision-and-video-analytics)（EN / NVIDIA，2026，2 小时完整 workshop）
- 中文补课：[深度学习模型部署与剪枝优化实战](https://www.bilibili.com/video/BV1Sw411y7Hs/)（学习其中 GStreamer / DeepStream 单元）
- 文档：[DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)、[DeepStream GitHub](https://github.com/NVIDIA/DeepStream)、[DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)、[DeepStream 中文开发入口](https://developer.nvidia.cn/deepstream-sdk)、[Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/)

##### 当阶段课程大纲（★ = 本阶段必修）

- **NVIDIA Jetson AI Fundamentals（官方 playlist）** 大纲：
  - ★ Hello AI World 与 jetson-inference 安装（忽略 Nano/JetPack4 命令）
  - ★ 图像分类 classification
  - ★ 目标检测 detection
  - ★ 语义分割 segmentation
  - ★ 摄像头与实时推理、训练与迁移学习（PyTorch）
  - ★ `tegrastats` 与各阶段耗时拆解（模型加载/预处理/推理/后处理/捕获/显示）
- **Seeed Jetson 边缘 AI 快速上手系列（B站）** 大纲：
  - ★ Jetson 上手、摄像头与边缘 AI 工作流
  - 进阶项目（按需）
- **Create Vision AI Applications With DeepStream（2026 GTC workshop）** 大纲：
  - ★ GStreamer 基础与 pipeline 概念（caps、buffer、metadata）
  - ★ DeepStream 插件：nvinfer / tracker / tiler / metadata
  - ★ 文件源与 RTSP/摄像头源、单路→多路
  - ★ 性能与功耗测量（FPS、延迟、温度、内存）
- **深度学习模型部署与剪枝优化实战（B站）** 大纲：
  - ★ GStreamer / DeepStream 相关单元
  - ★ 模型剪枝优化基础
  - 其它通用部署内容（按需）

### 学习方法与停止条件

- **节点 6A 看完即做**：从 jetson-inference 依次跑 classification、detection、segmentation，再接 USB/CSI 摄像头。停止条件：能区分模型加载、预处理、TensorRT 推理、后处理、捕获和显示耗时，并记录 `tegrastats`。版本注意：课程中 Nano/JetPack 4 的安装命令全部忽略；本机只用 JetPack 7.2.1 对应路径。
- **节点 6B 免费边界**：NVIDIA On-Demand 视频可免费观看；页面若要求登录，只注册免费 NVIDIA 账号，不购买 DLI 实验。真正的验收在本机 Orin 上完成。看完即做：文件单路 → 摄像头单路 → 4 路文件/RTSP；加入 decode、mux、`nvinfer`、tracker、tiler、metadata 和 sink。停止条件：能画出 pipeline，解释 caps、buffer、metadata、batch 和 zero-copy，并报告每路 FPS、端到端延迟、温度、功耗和内存。
- **强版本警告**：本机 DeepStream 9.1；旧课的安装命令、插件字段和 Python binding 不能直接复制，代码从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 当前 release 开始。

### 阶段项目

`jetson-inference` 摄像头分类、检测和分割（节点 6A）；单路到 4 路 decode / infer / tracker / metadata pipeline（节点 6B）。

### 版本 / 边界注意

课程中 Nano/JetPack 4 命令忽略；本机 JetPack 7.2.1、DeepStream 9.1。DeepStream 的 NVMM 零拷贝与阶段五 DALI 的 zero-copy 思想一致。

---

## 8. 阶段七 · 容器与服务化（原节点 7A Docker + 7B gRPC + 节点 8 Triton）

### 目标 / 为什么在这个阶段

把推理程序变成可部署、可测量的服务。阶段七 = Docker 容器（节点 7A）+ HTTP/gRPC（节点 7B）+ Triton Inference Server（节点 8）。重点是服务层：健康检查、动态批处理、ensemble、metrics，而不是替代 TensorRT 的 Kernel/runtime。

### 课程与资料

**节点 7A Docker：**

- 英文主课：[Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)（freeCodeCamp，约 2 小时 10 分）
- 中文补课：[尚硅谷 Docker 与微服务实战 2024](https://www.bilibili.com/video/BV1Zn4y1X7AZ/)

**节点 7B HTTP / gRPC：**

- 英文主课：[Getting Started With gRPC: Hands-On Codelab](https://www.youtube.com/watch?v=kAuK6VcAR10)（CNCF，约 75 分钟）
- 中文补课：[手把手 gRPC 基础教程](https://www.bilibili.com/video/BV1QT411H7ds/)（14 讲完整系列）

**节点 8 Triton Inference Server：**

- 主课：[NVIDIA Triton 从入门到精通](https://www.bilibili.com/video/BV1KS4y1v7zd/)（中文 / NVIDIA Official，20 讲完整系列）
- 补课：[Getting Started with NVIDIA Triton](https://www.youtube.com/watch?v=NQDtfSi5QF4)（EN / NVIDIA，完整入门专题）
- 文档：[Triton Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)、[Triton Tutorials](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/README.html)、[Triton Server](https://github.com/triton-inference-server/server)、[Triton Tutorials Repo](https://github.com/triton-inference-server/tutorials)、[NVIDIA 中文 TensorRT/Triton 课程入口](https://www.nvidia.cn/developer/online-training/super-training/)

##### 当阶段课程大纲（★ = 本阶段必修）

- **Docker（freeCodeCamp / 尚硅谷）** 大纲：
  - ★ 镜像与容器概念、Dockerfile（multi-stage）
  - ★ volume、network、healthcheck
  - ★ Docker Compose 服务编排
  - ★ 确认基础镜像支持 `linux/arm64`（Jetson 部署）
- **gRPC（CNCF codelab / 手把手 gRPC 14 讲）** 大纲：
  - ★ `.proto` 定义与 RPC 语义
  - ★ unary RPC
  - ★ server streaming / client streaming / bidirectional streaming
  - ★ 错误处理、deadline/timeout、health check
- **NVIDIA Triton 从入门到精通（B站 20 讲）** 大纲：
  - ★ 概念与 model repository
  - ★ backend：TensorRT / ONNX / PyTorch
  - ★ HTTP / gRPC client
  - ★ dynamic batching、instance group
  - ★ ensemble、metrics
  - ★ Perf Analyzer 与并发/延迟曲线（架构概念可学；配置字段以当前 Triton 文档为准）
- **Getting Started with NVIDIA Triton（官方专题）** 大纲：
  - ★ 安装、模型仓库、基础 client
  - ★ 最新版官方术语入口

### 学习方法与停止条件

- **Docker**：二选一完整课。完成 multi-stage Dockerfile、volume、network、healthcheck、Compose，并确认基础镜像支持 `linux/arm64`。
- **gRPC**：先 CNCF codelab，中文 14 讲补 unary、server/client streaming 和双向流。语言不同不影响理解 `.proto` 和 RPC 语义。
- **Triton 看完即做**：model repository、TensorRT backend、HTTP/gRPC client、dynamic batching、instance group、ensemble、metrics、Perf Analyzer。停止条件：画出 concurrency/batch 与 p50/p95/throughput 曲线，并能说明 Triton 解决的是服务层问题，不是替代 TensorRT Kernel/runtime。
- **阶段七停止条件**：能解释镜像与容器、host/container 文件和端口、HTTP 与 gRPC、deadline、错误码和流式请求；并把 TensorRT demo 包成有 `/health` 的容器服务，写一个 gRPC predict 接口。
- **版本注意**：中文 Triton 课基于 2022 年版本，架构概念仍可学；容器 tag、backend 支持和配置字段查当前 [Triton tutorials repo](https://github.com/triton-inference-server/tutorials) 与 Jetson platform matrix。2026 官方页面也可能使用 Dynamo-Triton 名称。

### 阶段项目

为 TensorRT 推理程序制作 ARM64 镜像和 Compose 服务（节点 7A）；`.proto` + unary/stream client/server + health check（节点 7B）；TensorRT backend、dynamic batching、ensemble、Perf Analyzer（节点 8）。

### 版本 / 边界注意

确认基础镜像支持 `linux/arm64`。Triton 的 backend/platform 组合不是全部在 Jetson 可用，尤其不要默认 Python backend 的 GPU 能力与 x86 服务器一致；以 [官方 backend platform matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md) 为准。

---

## 9. 阶段八 · LLM 推理与服务（原节点 9 CS336 + 节点 9.5 运行时选型 + 节点 10 Orin Edge LLM）

### 目标 / 为什么在这个阶段

从视觉推理走到 LLM 推理与边缘服务。阶段八 = LLM 原理（节点 9 CS336）+ 边缘 LLM 运行时选型对比（节点 9.5）+ 本机 Orin Edge LLM 落地（节点 10）。核心是不把选型绑死在一家运行时，并在 Orin 支持矩阵内做可重复 benchmark。

### 课程与资料

**节点 9 LLM Inference / Serving 原理：**

- 主课：[Stanford CS336 2026 Video Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)（EN / University，完整课程；重点第 6、10 讲）；作业与讲义见 [CS336 官方课程页](https://cs336.stanford.edu/)；[CS336 Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM)、[CS336 Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)
- 中文补课：[大模型推理技术研究](https://www.bilibili.com/video/BV1k2L9zyEt7/)（KV cache、vLLM、SGLang 等 9 讲完整系列）

**节点 9.5 边缘 LLM 运行时选型对比：**

- 入口：[Jetson AI Lab 边缘 LLM 教程](https://www.jetson-ai-lab.com/)（含 llama.cpp、MLC-LLM、Edge-LLM 对比指南；中文实操见 [GitHub 教程](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)）
- 运行时：[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)（[文档](https://nvidia.github.io/TensorRT-Edge-LLM/)、[Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)）、[llama.cpp](https://github.com/ggerganov/llama.cpp)（GGUF）、[MLC-LLM](https://github.com/mlc-ai/mlc-llm)（TVM 系）、[ExecuTorch](https://github.com/pytorch/executorch)（PyTorch Edge）
- 对比视频：[Run GenAI Locally on NVIDIA Jetson](https://www.youtube.com/watch?v=czteUSONG-c)（Ollama / vLLM / llama.cpp 选型与快速上手）
- 文档参考：[vLLM Docs](https://docs.vllm.ai/en/latest/)、[TensorRT-LLM Docs](https://nvidia.github.io/TensorRT-LLM/)

**节点 10 Orin Edge LLM：**

- 主课：[Make It Think: NVIDIA Jetson AI Lab](https://youtube.com/playlist?list=PLZrTAEPLeXfo)（EN / NVIDIA，2026，3 场完整系列）
- 补课：[Getting Started with Edge AI on NVIDIA Jetson](https://www.youtube.com/watch?v=t2Ecuu2FdC8)（EN / NVIDIA，完整直播课）
- 落地文档：[TensorRT-Edge-LLM 官方安装与教程](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)

##### 当阶段课程大纲（★ = 本阶段必修）

- **Stanford CS336 2026** 大纲：
  - ★ 大模型概览、Tokenizer
  - ★ 模型架构与训练数据
  - ★ Lecture 6：Kernels / Triton / XLA（需要写算子时看）
  - ★ Lecture 10：Inference（prefill/decode、KV cache、continuous batching）
  - 评测与系统（★ 关注推理相关）
- **大模型推理技术研究（B站 9 讲）** 大纲：
  - ★ KV cache、PagedAttention
  - ★ continuous batching、vLLM、SGLang
  - ★ 量化（INT4/AWQ/GPTQ）、speculative decoding
  - ★ RadixAttention 与上下文管理
- **主课视频：Run GenAI Locally on NVIDIA Jetson** 大纲：
  - ★ 为何在 Jetson 上本地跑 LLM/VLM（延迟、隐私、带宽）
  - ★ Ollama 快速实验
  - ★ vLLM 追求最佳吞吐（LLM/VLM）
  - ★ llama.cpp 轻量运行与 GGUF
  - ★ 运行时选型对比思路（Orin/Thor 适用场景）
- **辅助文档（非视频，均本阶段必修）：** Jetson AI Lab 边缘 LLM 教程、[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[MLC-LLM](https://github.com/mlc-ai/mlc-llm)、[ExecuTorch](https://github.com/pytorch/executorch) ★ 用于横向对比与选型表。
- **Make It Think: NVIDIA Jetson AI Lab（2026 三场系列）** 大纲：
  - ★ Edge LLM / VLM 在 Jetson 上的工作流
  - ★ 小模型（Qwen3 等）FP16 基线 → INT4/AWQ
  - ★ 简单 API 服务与 TTFT/TPOT/tokens/s/峰值 unified memory 测量
  - ★ 功耗、温度、上下文长度与并发报告
- **Getting Started with Edge AI on NVIDIA Jetson（官方直播课）** 大纲：
  - ★ LLM/VLM/Foundation Models 在机器人/边缘的落地
  - ★ 快速原型与部署思路

### 学习方法与停止条件

- **节点 9 看完即做**：用同一小模型和固定 prompt，对 batch、并发、上下文长度、量化方式做可复现 benchmark。停止条件：能严格区分 TTFT、TPOT、端到端延迟、tokens/s、单请求延迟和系统吞吐，并解释 prefill/decode 的瓶颈差异。框架关系：vLLM、SGLang、TensorRT-LLM/Edge-LLM 是 LLM runtime/engine 路线；Triton 是可选的通用服务层，二者不是同一个层级。
- **节点 9.5 必修关键字**：GGUF vs safetensors、weight-only 量化格式、runtime 对 AWQ/GPTQ 的支持、CUDA vs CPU offload、KV cache 管理、OpenAI-compatible API、易用性 vs 峰值性能 vs 内存占用。看完即做：选同一小模型（如 Qwen3 0.6B）在至少两个运行时（建议 TensorRT-Edge-LLM + llama.cpp）上跑，固定 prompt 与上下文，测量 TTFT、TPOT、tokens/s、峰值 unified memory 与量化格式。停止条件：交付一张选型表，标明每个运行时在 Orin 上的适用场景（快速验证 / 最高吞吐 / 最低内存 / 跨平台），并说明为何节点 10 仍优先 TensorRT-Edge-LLM 但不排斥其他。
- **节点 10 看完即做**：从支持矩阵内的小模型开始，先 FP16 基线，再 INT4/AWQ；提供简单 API，测 TTFT、TPOT、tokens/s、峰值 unified memory 和上下文长度。停止条件：服务可重复启动，连续请求无 OOM，报告中包含功耗、温度、量化、上下文、并发和版本。资料现实：截至 2026-08-21，还没有一门公开、完整且与当前 TensorRT-Edge-LLM release 同步的官方专项视频课；这里用最新 Jetson 完整视频系列建立工作流，代码只跟 [TensorRT-Edge-LLM 官方安装与教程](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)，不拿旧 TensorRT-LLM 服务器课程冒充 Orin 课程。

### 阶段项目

测 TTFT、TPOT、tokens/s、并发和 KV cache 占用（节点 9）；同一 Qwen3 0.6B 在 ≥2 个运行时跑，对比 TTFT/TPOT/tokens/s/峰值内存/量化格式/易用性，形成选型表（节点 9.5）；TensorRT-Edge-LLM 支持矩阵内的小模型 FP16 / INT4 服务（节点 10）。

### 版本 / 边界注意

本机 Orin NX 16GB、JetPack 7.2.1；节点 10 代码以 TensorRT-Edge-LLM 当前 release 为准。先选支持矩阵内小模型（Qwen3 0.6B/1.7B/4B），不盲目移植 14B 以上；不把 server GPU 上的 vLLM/TRT-LLM 命令直接复制到 Jetson。

---

## 10. 可选分支（原分支 A Linux 驱动 + 分支 B Triton DSL/CUTLASS/CuTe + 分支 C TVM/MLIR）

### 分支 A：Linux 驱动 / BSP

**进入时间**：CUDA/Nsight 主线完成后；除非目标岗位就是 BSP/驱动，否则每周并行 2 小时即可。主线位置：Linux 用户态 → module → 字符设备 → ioctl/poll/mmap → DMA/中断 → device tree → Jetson BSP → 调试与性能。

**课程与资料：**

- 英文主课：[Linux Device Drivers Development](https://www.youtube.com/watch?v=iSiyDHobXHA)（freeCodeCamp，约 5 小时）
- 中文主课：[韦东山：嵌入式 Linux 驱动开发基础](https://www.bilibili.com/video/BV14f4y1Q7ti/)（50 讲、约 17 小时）
- 文档：[LFD103 Kernel Development](https://training.linuxfoundation.org/training/a-beginners-guide-to-linux-kernel-development-lfd103/)、[Bootlin Training Materials](https://bootlin.com/docs/)、[Bootlin Kernel Training](https://bootlin.com/training/kernel/)、[Kernel Driver API](https://www.kernel.org/doc/html/latest/driver-api/index.html)、[Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)、[Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)、[TinyLab 内核文档中文版](https://tinylab-1.gitbook.io/linux-doc/zh-cn)

**当阶段课程大纲（★ = 本阶段必修）**

- **Linux Device Drivers Development（freeCodeCamp，约 5 小时）** 大纲：
  - ★ 内核模块 module 编写与加载
  - ★ syscall 与 /proc 接口
  - ★ 字符设备、ioctl
  - ★ GPIO、中断、工作队列（概念）
- **韦东山：嵌入式 Linux 驱动开发基础（B站 50 讲）** 大纲：
  - ★ 字符设备、ioctl、poll、mmap
  - ★ GPIO、中断、设备树匹配
  - ★ 不要在 Orin 启动链上直接试验

**学习方法与停止条件**：freeCodeCamp 5 小时快速建立 module、syscall、`/proc`；韦东山中文课补 GPIO、字符设备、中断、工作队列和 `mmap`。看完即做：在 VM、QEMU 或可恢复开发板完成 hello module、字符设备、ioctl、poll 和 mmap。停止条件：能从 device tree 匹配到 driver/probe，能用 `dmesg`、ftrace/perf 定位一次问题。不要先改 Orin 启动链或 NVIDIA GPU 驱动。项目：编译并加载 hello module；读懂一个字符设备驱动；能解释设备树、module、用户态/内核态。

### 分支 B：Triton Language、CUTLASS 和 CuTe

**进入时间**：只有阶段三（节点 3、4）已通过，才进入算子分支。这里的 Triton 是 Kernel DSL，不是 Triton Inference Server。

**课程与资料：**

- 主课：[Triton 从入门到大师](https://www.bilibili.com/video/BV1fMyWBgERM/)（中文，10 讲完整课）
- 专题：[Stanford CS336 2026 Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)（EN / University）
- 代码：[triton_docs_tutorials](https://github.com/evintunador/triton_docs_tutorials)
- 高级系列：[GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)（CUTLASS/CuTe 讲次）
- 官方文档：[Triton Language](https://triton-lang.org/main/)、[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)、[Triton GitHub](https://github.com/triton-lang/triton)、[CuTe tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)、[CUTLASS](https://github.com/NVIDIA/cutlass)

**当阶段课程大纲（★ = 本阶段必修）**

- **Triton 从入门到大师（B站 10 讲）** 大纲：
  - ★ Triton 编程模型与 program instance
  - ★ vector add → fused softmax → matmul → layer norm
  - ★ autotune 与 tile
  - ★ 与 CUDA 版本做正确性/性能/可读性对比
- **CS336 Lecture 6: Kernels, Triton, XLA** 大纲：
  - ★ kernel 实现视角下的 Triton / XLA
  - ★ 写算子时配套观看
- **分支 B2 — GPU MODE Lectures（CUTLASS / CuTe 讲次）** 大纲：
  - ★ CUDA / Triton / CUTLASS / CuTe 相关讲次
  - ★ GEMM tile / layout / MMA 与 CUTLASS profiler
  - ★ CuTe GEMM 实战理解 layout / tile / MMA
- **CuTe tutorial（官方文档）** 大纲：
  - ★ CuTe 快速上手与 layout 概念
  - ★ 一个 CuTe GEMM 项目

**学习方法与停止条件**：Triton 课后项目：vector add → fused softmax → matmul → layer norm；与 CUDA 版本做 correctness、性能和可读性对比。CUTLASS/CuTe 以 GPU MODE 为完整课程载体，再进入官方 CuTe tutorial 做项目。停止条件：能解释 program instance、tile、layout、MMA、autotune 的边界，并知道何时选 CUDA、Triton 或 CUTLASS。Triton 适合在 x86 NVIDIA GPU 或公开 notebook 环境学习；不要默认当前 Triton、PyTorch 和 ARM64 Jetson 组合可原生安装。

### 分支 C：TVM 和 MLIR

**进入时间**：Triton Kernel 项目完成后再学；它不是 TensorRT 前置课。

**课程与资料：**

- 主课：[MLC 机器学习编译](https://www.bilibili.com/video/BV15v4y1g7EU)（陈天奇课程，10 讲完整系列）；[中英文课程页](https://mlc.ai/summer22-zh/schedule)
- 专题：[LLVM MLIR Tutorial](https://www.youtube.com/watch?v=Y4SvqTtOIDk)（EN / LLVM Official，完整 workshop）
- 文档：[Apache TVM Documentation](https://tvm.apache.org/docs/)、[TVM End-to-End Optimization Tutorial](https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html)、[MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)

**当阶段课程大纲（★ = 本阶段必修）**

- **MLC 机器学习编译（陈天奇课程，B站 10 讲）** 大纲：
  - ★ TVM/MLC 概览与安装
  - ★ TensorIR schedule 与自动优化
  - ★ 编译到多后端（含 Orin/边缘）
  - ★ 配套 [中英文课程页](https://mlc.ai/summer22-zh/schedule)
- **LLVM MLIR Tutorial（官方 workshop）** 大纲：
  - ★ MLIR Toy dialect：AST → dialect → passes → lowering → LLVM
  - ★ dialect / pass / lowering 概念

**学习方法与停止条件**：完整学 MLC 10 讲及配套课程页，再看 LLVM MLIR workshop。看完即做：完成 TensorIR schedule/自动优化 notebook；再走一遍 MLIR Toy 的 AST → dialect → passes → lowering → LLVM。停止条件：能解释 graph optimization、tensor program、IR、dialect、pass、lowering、runtime 的层级关系，并能把一个优化落到可运行代码。不要在 CUDA 基础没掌握时从 MLIR 开始。

---

## 11. 六周 CUDA 冲刺（原 section 4，完整逐周计划）

你提供的两个 CUDA 视频可以保留，但它们的定位不同：

- [NVIDIA Modern CUDA C++ Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)：作为主课程视频，配合 NVIDIA 的公开 notebook 和 GitHub 实验。
- [freeCodeCamp CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)：作为长线补充，覆盖 CUDA API、矩阵乘法、Triton 和 PyTorch extension。
- [freeCodeCamp 课程代码仓库](https://github.com/Infatoshi/cuda-course)：不要只看视频，按章节运行并改代码。

结论是：**这两个视频足够启动，但不够形成完整能力**。缺口在官方编程模型、Best Practices、Sanitizer、Nsight、架构分析、TensorRT 和 Jetson 项目。下面的六周计划把缺口补齐。

六周内不要从文档通读开始。每周固定采用：**对应视频章节 → 当天复现 → 改一个参数或实现 → 出现具体问题后查文档 → 写一页结果**。

### 第 1 周：能编译、能运行、能解释线程层次

关键字：nvcc、host code、device code、kernel launch、grid、block、thread、global memory、device synchronization、error checking。

资料顺序：

1. Modern CUDA C++ Playlist 的 introduction、execution spaces 和第一个 Kernel。
2. 同步完成 [NVIDIA CUDA C++ Tutorial](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp) 对应 notebook。
3. freeCodeCamp 课程中 setup、C/C++ review、first kernels 部分。
4. 只在索引或 launch 行为不清楚时，查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 的 programming model 对应小节。

必须完成：vector add / SAXPY；CPU reference implementation；cudaGetLastError 和同步错误检查；记录编译命令、GPU 型号、数据规模、CPU/GPU 时间。

过关标准：能画出 grid → block → thread 的索引关系；能说明 kernel launch 是异步的，以及何时需要同步；能解释 unified memory 不是免费高速显存。

### 第 2 周：内存访问和常见基础 Kernel

关键字：coalesced access、stride、shared memory、bank conflict、register、occupancy、warp divergence、transpose、reduction。

必须完成：naive transpose 与 tiled transpose；naive reduction 与 shared-memory reduction；至少一次故意制造的非合并访问，并用数据说明代价。

视频与实验顺序：

1. [大规模并行处理器编程实战](https://www.bilibili.com/video/BV1gz421o7uH/) 第 4-6 讲：GPU 架构、memory tiling、performance considerations。
2. [Stanford CS149 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp) 的 GPU Architecture/CUDA 和 data-parallel thinking 相关讲次。
3. 从 [CUDA Samples](https://github.com/NVIDIA/cuda-samples) 选择 transpose、reduction 或相近 sample 改写。
4. 指标或规则不清楚时，再查 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。

过关标准：不用“GPU 更快”解释结果，而是能从访存、并行度、同步和算术强度解释结果；能看懂一个 sample 的 launch configuration 和 memory layout。

### 第 3 周：Streams、Events 和 CPU/GPU 重叠

关键字：cudaStream、cudaEvent、pinned host memory、pageable memory、overlap、double buffering、CUDA Graph、asynchronous copy。

必须完成：把大数组分块，使用两个 stream 尝试重叠 H2D、kernel、D2H；用 CUDA events 计时，不用端到端墙钟时间替代 kernel 时间；对比同步版本和异步版本。

视频与实验顺序：

1. freeCodeCamp CUDA 课程的 streams、events、pinned memory 相关章节。
2. CUDA Samples 中的 streams、events、graph 相关示例。
3. [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) 中选择一个异步库调用示例。
4. 只为核对异步语义和限制，查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 对应小节。

过关标准：能用 timeline 证明是否真的发生了重叠；能区分 API 调用耗时、kernel 执行耗时和端到端延迟。

### 第 4 周：矩阵乘法、cuBLAS 和真实小项目

关键字：tiled GEMM、arithmetic intensity、cuBLAS、cuBLASLt、FP32、FP16、TF32、WMMA、Tensor Core。

必须完成：naive GEMM；shared-memory tiled GEMM；cuBLAS GEMM；比较准确率、吞吐和工作量，不把库调用当成黑盒结束。

主项目：[MNIST CUDA](https://github.com/Infatoshi/mnist-cuda)；配套课程：[CUDA Course](https://github.com/Infatoshi/cuda-course)。

视频先看 freeCodeCamp 课程中的 matrix multiplication、cuBLAS、mixed precision 和 MNIST 项目章节；PMPP 第 5-6 讲用于补 tiling 与性能模型。

建议按以下阶段复现：1. PyTorch baseline；2. NumPy / C baseline；3. naive CUDA；4. cuBLAS；5. streams、fused kernel、TF32 或 FP16；6. 自己做一次 GEMM 或预处理 Kernel 的优化。

过关标准：有一张表记录每个版本的准确率、p50/p95 延迟、吞吐和峰值内存；能说明为什么自写 Kernel 不一定比 cuBLAS 快。

### 第 5 周：正确性、系统级 Profile、Kernel Profile

工具顺序必须固定为：

1. **Compute Sanitizer**：先保证没有 memory error、race、未初始化访问和同步错误。
2. **Nsight Systems**：看整个程序和流水线，定位 CPU/GPU 空洞、拷贝瓶颈、同步点。
3. **Nsight Compute**：只对热点 Kernel 做深入分析，查看 occupancy、memory throughput、warp stall 和指令指标。
4. **cuobjdump / nvdisasm / PTX**：最后用于验证编译结果和理解指令，不要一开始就盯着汇编猜性能。

视频与查询顺序：

1. 完整看 [NVIDIA CUDA Developer Tools Tutorials](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR)，或使用[同课中文镜像](https://www.bilibili.com/video/BV14RU6BmE5u/)。
2. 立刻对第 4 周项目执行 Sanitizer、`nsys` 和 `ncu`，不要先背指标。
3. 采集遇到疑问后，再查 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)。
4. 最后查 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) 和 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html) 核对反汇编结果。

Jetson 使用方式：Jetson 是 target；Nsight Systems / Nsight Compute 的图形界面通常放在 x86 Linux host。target 端可以先使用 CLI 采集，再把报告传到 host 分析。采集前固定功耗模式、输入规模、频率和温度条件。

必须交付：一份优化前/后报告；一张 Nsight Systems timeline；一张 Nsight Compute 热点指标截图或导出结果；一次 Compute Sanitizer 结果；一段由 profile 数据支持的优化结论。

### 第 6 周：CUDA 图、协作组、Tensor Core 与 PTX/SASS

关键字：cooperative groups、CUDA Graph、warp shuffle、WMMA、Tensor Core、PTX、SASS、sm_87、fatbin。

视频与实验顺序：

1. 从 [GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL) 选择 profiling、Tensor Core、SASS 和高性能 GEMM 相关讲次。
2. 回看 PMPP 的 GPU architecture、performance 和 Nsight 讲次。
3. 从 [CUDA Samples](https://github.com/NVIDIA/cuda-samples) 选择 cooperative groups、shuffle、CUDA Graph 或 WMMA 示例改写。
4. 只为核对语义和指令，查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html) 和 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)。

必须完成：用 cooperative groups 或 warp primitive 改写一个 reduction/scan 小实验；用 cuobjdump 或 nvdisasm 检查一个 Kernel 的目标架构和关键指令；在 Orin 上确认 sm_87 相关编译配置，不照抄 H100 的 sm_90、TMA、WGMMA 示例。

---

## 12. 真实公开项目阶梯（原 section 5，保留项目 A–F）

### 项目 A：CUDA 基础 Kernel 仓库

项目来源：[NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)、[NVIDIA CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)。

你要自己维护一个小仓库，至少包含：vector add / SAXPY、transpose、reduction、tiled GEMM、stream overlap、sanitizer 和 benchmark 脚本。验收：每个例子有 CPU reference、正确性测试、数据规模和性能记录。

### 项目 B：MNIST CUDA 逐级优化

项目来源：[Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)、[Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)。

验收：完成 PyTorch → C/CUDA → cuBLAS → streams/fusion → FP16 或 TF32 的至少四个版本，并解释每次变化。

### 项目 C：TensorRT 推理部署

项目来源：[NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)、[TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)、[TensorRT Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)。

任务：1. 训练或选择一个小模型；2. 导出 ONNX 并验证输出；3. 用 trtexec 或 Python/C++ API 构建 Engine；4. 比较 FP32、FP16、INT8；5. 加入 dynamic shape；6. 对不支持的算子尝试 plugin 或改图。

验收指标：accuracy delta、engine build 是否可复现、warmup 后 p50/p95 latency、throughput、峰值 unified memory、输入 shape 和 batch 的影响。

### 项目 D：Jetson 多路视频推理

项目来源：[NVIDIA DeepStream](https://github.com/NVIDIA/DeepStream)、[DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)、[DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)。

任务：先跑单路文件或 USB 摄像头；再扩展为 4 路文件/RTSP 输入；加入 decode、nvinfer、tracker、tiler、OSD 和 metadata；对比 OpenCV/Python 串行实现与 DeepStream pipeline；用 tegrastats 和 Nsight Systems 观察 CPU、GPU、内存、解码和推理。

验收：在固定功耗和输入条件下，报告每路 FPS、端到端延迟、GPU 利用率、内存、温度和降频情况。注意：旧的 [deepstream_reference_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps) 仓库已经停止更新；当前新项目优先从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 主仓库和对应 release 开始。

### 项目 E：Triton 模型服务

项目来源：[Triton Inference Server](https://github.com/triton-inference-server/server)、[Triton Tutorials](https://github.com/triton-inference-server/tutorials)、[Triton Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)、[Backend Platform Support Matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md)。

任务：建立 model repository；用 TensorRT backend 提供一个图像分类或检测模型；配置 dynamic batching 和多个 model instance；写 HTTP 和 gRPC client；加入 health、metrics、错误处理和超时；用 Perf Analyzer 测量 concurrency、latency 和 throughput；再做一个 ensemble，把预处理、推理、后处理串起来。

验收：画出并发数、batch、p50/p95 延迟和吞吐之间的曲线，并说明哪一个配置适合 Orin 的内存和功耗约束。

### 项目 F：Orin 上的 Edge LLM

项目来源：[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)、[TensorRT-Edge-LLM Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)、[Jetson AI Lab Edge-LLM Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)。

建议 capstone：选择支持矩阵中的 Qwen3 小模型，优先从 0.6B、1.7B 或 4B 级别开始；先做 FP16 可运行基线，再尝试 INT4/AWQ；用一个简单 HTTP API 包装推理；测量 TTFT、TPOT、tokens/s、峰值内存、上下文长度和连续请求行为。

不要一开始做：14B 以上模型的盲目移植；只看 tokens/s、不测首 token 延迟；把 server GPU 上的 vLLM/TRT-LLM 命令直接复制到 Jetson。

---

## 13. 概念澄清：TensorRT / Triton / vLLM / TensorRT-LLM 的关系（原 section 6）

### 13.1 TensorRT 是运行时和优化器

典型流程：PyTorch / TensorFlow → ONNX → TensorRT Builder → hardware-specific Engine → TensorRT Runtime。

TensorRT 解决的是模型图优化、Kernel 选择、精度、Engine 构建和执行。它本身不是完整的多租户 HTTP 服务。

### 13.2 Triton 是通用推理服务层

Triton 提供：model repository、多种 backend、HTTP/gRPC、dynamic batching、concurrent model execution、ensemble、health 和 metrics。

因此，**LLM serving 可以使用 Triton，但 LLM serving 不等于 Triton**。Triton 是一个通用 serving layer；模型运行时可能是 TensorRT、PyTorch、ONNX Runtime、Python backend 或其他 backend。

### 13.3 vLLM、TensorRT-LLM、Edge-LLM 的定位

- **vLLM**：偏通用服务器 GPU 的 LLM runtime/serving，重点是 KV cache、continuous batching 和高吞吐。
- **TensorRT-LLM**：偏 NVIDIA 服务器 GPU 的高性能 LLM runtime，通常需要匹配的 GPU、CUDA、容器和版本。
- **TensorRT-Edge-LLM**：面向 Jetson/边缘设备的路线；对当前 Orin NX 更有现实意义。
- **Triton**：可以承载多种模型和 backend，是服务编排与推理 API 层；是否适合某个 Jetson backend 必须查平台矩阵。

推荐顺序：1. TensorRT 单模型运行；2. Triton + TensorRT backend；3. 在服务器 GPU 上理解 vLLM/TensorRT-LLM 的 serving 概念；4. 在 Orin 上使用 TensorRT-Edge-LLM 的支持矩阵内路径。

Jetson 平台特别注意：Triton 的 backend/platform 组合不是全部可用，尤其不要默认 Python backend 的 GPU 能力与 x86 服务器一致；以 [官方 backend platform matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md) 为准。

---

## 14. 24 周执行节奏（原 section 9）

假设每周投入 10-12 小时；每周只有 5-6 小时时，把日历时间大约翻倍。

如果已经能独立完成 C++17/CMake 小程序，能创建可复现的 Python `venv`、使用 NumPy 和 PyTorch 处理 tensor，并熟悉 Shell、进程、权限、SSH 和 `gdb`，才可以跳过第 1-2 周；否则先完成阶段二的 Python + PyTorch `model-tools` 和 C++ 项目。遇到基础问题时再回看阶段二对应章节。

| 周数 | 主线 | 交付物 |
|---:|---|---|
| 1-2 | Python、PyTorch、C++、Linux、Git、CMake、venv、NumPy、ONNX、gdb | `model-tools` + CPU 矩阵乘法和 Git 记录 |
| 3 | CUDA 编程模型与第一个 Kernel | vector add/SAXPY |
| 4 | 内存层次、transpose、reduction | 两个正确且可 benchmark 的 Kernel |
| 5 | streams、events、pinned memory | overlap 实验和时间线 |
| 6 | tiled GEMM、cuBLAS、FP16/TF32 概念 | GEMM 对比表 |
| 7 | MNIST CUDA 项目 | baseline 与 naive CUDA |
| 8 | Sanitizer、Nsight Systems、Nsight Compute | 优化前后报告 |
| 9-10 | ONNX、TensorRT、trtexec | 第一个可复现 Engine |
| 11 | dynamic shape、FP16、INT8 | 精度/延迟/内存比较 |
| 12 | plugin、Polygraphy、Torch-TensorRT | 一个不支持算子的处理记录 |
| 13-14 | GStreamer、Jetson 视频输入、DeepStream 基础 | 单路视频推理 |
| 15-16 | 多路、tracker、metadata、tegrastats | 4 路视频报告 |
| 17 | HTTP/gRPC、Docker 基础 | 一个健康检查和推理 API |
| 18 | Triton model repository 和 TensorRT backend | 单模型 Triton 服务 |
| 19 | batching、ensemble、metrics、Perf Analyzer | 并发/延迟曲线 |
| 20-21 | Edge-LLM、量化、KV cache、服务指标 | Orin 上的小模型基线 |
| 22 | INT4/AWQ 和 API 包装 | LLM 服务 demo |
| 23 | 端到端整合和故障处理 | 可重复启动的项目 |
| 24 | 性能回归、文档、演示 | 完整技术报告和 benchmark |

---

## 15. 每个阶段统一的验收标准（原 section 10）

不要只以视频看完作为完成条件。每个项目都必须回答以下问题：

**正确性**：是否有 CPU 或框架 reference？随机输入、边界输入和不同 batch 是否通过？Compute Sanitizer 是否干净？模型导出前后精度差异是多少？

**性能**：测量的是 kernel、runtime 还是端到端？是否排除了首次加载和 warmup？是否报告 p50、p95，而不是只报平均值？优化前后是否有 Nsight 或 benchmark 证据？

**Edge 条件**：功耗模式是什么？温度和是否降频？unified memory 峰值是多少？冷启动和热运行结果是否不同？输入分辨率、batch、并发是否固定？

**Serving**：健康检查、超时、错误和重启如何处理？并发增加时延迟如何变化？dynamic batching 是否真的提高吞吐？是否有 Prometheus 或等价指标？

**LLM**：使用的模型和量化格式是否在目标平台支持矩阵内？是否分别测量 TTFT、TPOT、tokens/s？峰值内存和最大上下文是多少？多请求时是否出现 OOM、抖动或严重排队？

---

## 16. Orin NX 上的版本和操作纪律（原 section 11）

1. 先确认本机 JetPack、L4T、CUDA、TensorRT、DeepStream 版本，再执行任何教程命令。
2. 目标端优先使用 JetPack 提供的软件包；不要把 x86 CUDA 仓库或服务器 GPU 容器混装到 Jetson。
3. 编译 CUDA 时确认目标架构为 Orin 的 sm_87，或使用本机工具链推荐的架构参数。
4. 不要为了跟旧视频一致而随意降级 JetPack；先把旧 API 映射到当前文档。
5. DeepStream、Triton、CUDA 和 TensorRT 的容器必须匹配 JetPack/L4T；先看 release notes。
6. Nsight 的 target/host 分工要明确：Jetson 采集，x86 host 分析通常更方便。
7. 任何性能结论都要记录功耗、温度、输入 shape、batch、并发和软件版本。
8. 对 LLM，先选择支持矩阵内的小模型和精度；不要先从最大模型开始。
9. 版本敏感的 [DeepStream Release Notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html)、[TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[Edge-LLM Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html) 必须在每次部署前复核。

---

## 17. 中文资料的正确用法（原 section 12）

### 推荐作为主线的中文完整视频

- CUDA 性能：[大规模并行处理器编程实战 7 讲](https://www.bilibili.com/video/BV1gz421o7uH/)
- Nsight：[CUDA Developer Tools 同课中文镜像](https://www.bilibili.com/video/BV14RU6BmE5u/)
- TensorRT：[NVIDIA TensorRT 官方教程 4 部分](https://www.bilibili.com/video/BV15Y4y1W73E)
- Triton Server：[NVIDIA Triton 从入门到精通 20 讲](https://www.bilibili.com/video/BV1KS4y1v7zd/)
- LLM Serving：[大模型推理技术研究 9 讲](https://www.bilibili.com/video/BV1k2L9zyEt7/)
- Linux 驱动：[韦东山嵌入式 Linux 驱动开发基础 50 讲](https://www.bilibili.com/video/BV14f4y1Q7ti/)
- Triton Kernel：[Triton 从入门到大师 10 讲](https://www.bilibili.com/video/BV1fMyWBgERM/)
- AI 编译器：[MLC 机器学习编译 10 讲](https://www.bilibili.com/video/BV15v4y1g7EU)

做项目时再使用中文查询入口：[NVIDIA 中文 CUDA 平台](https://developer.nvidia.cn/cuda)、[NVIDIA 中文 TensorRT 入门](https://developer.nvidia.cn/tensorrt-getting-started)、[Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)、[Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)。

### 中文视频和字幕

中文视频适合建立概念、跟随界面和降低第一次阅读成本；但 CUDA、TensorRT、DeepStream 的版本变化很快。中文镜像可以看，代码必须回到官方仓库：

- [NVIDIA Modern CUDA C++ Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)
- [中文辅助镜像：Modern CUDA C++](https://www.bilibili.com/video/BV1QvSKB4EMr/)

Bilibili 上由 NVIDIA英伟达、原作者或大学发布的课程可以作为主课；第三方翻译镜像的版权、字幕质量和更新状态不由上游项目保证，因此只作为语言辅助，不作为唯一依据。

---

## 18. 明确排除的内容（原 section 13）

以下内容不放入免费主路线：

- NVIDIA DLI 中标价 30/90 美元的完整实践课程。
- 付费 Bootlin 讲师课程；只使用 Bootlin 公开 slides、labs 和源码。
- 依赖付费云 GPU 才能完成的实验。
- 仅针对 H100/Hopper 的高级课程作为 CUDA 入门。
- 旧版 DeepStream/Triton 教程中的固定安装命令。
- 未说明来源、版本和许可证的课程搬运或代码集合。

例如，H100 专用课程 [H100-Course](https://github.com/cudacourseh100/H100-Course) 可以在以后有 Hopper 机器时选修，但不适合作为 Orin SM87 的当前主线。

---

## 19. 最终毕业项目（原 section 14）

### Orin Edge AI Inference Platform

最终项目建议做成一个可以公开展示的仓库，包含：

- 一个 TensorRT FP16/INT8 图像模型。
- 一个 DeepStream 多路视频 pipeline。
- 一个 Triton TensorRT backend 服务，或在 Jetson 支持范围内的等价服务。
- 一个 TensorRT-Edge-LLM 小模型服务。
- HTTP/gRPC API、健康检查和 Prometheus 指标。
- Nsight Systems/Compute 报告。
- tegrastats 采样、功耗/温度记录和内存预算。
- 一键启动、固定版本、固定输入和可重复 benchmark。

最终报告至少回答：1. 哪些计算由现成库完成，哪些 Kernel 是自己写的？2. 哪个热点经过了 Profile，优化前后改变了什么？3. TensorRT 的精度、延迟和内存折中是什么？4. DeepStream 相比普通 OpenCV pipeline 的收益在哪里？5. Triton 在这里解决了什么，哪些问题它没有解决？6. Orin 上的 LLM 与服务器 GPU 上的 vLLM/TensorRT-LLM 有哪些限制差异？7. 在 25W、16GB unified memory 下，系统的实际上限是什么？

---

## 20. 一页式执行清单（原 section 15）

### 现在

- [ ] 安装并确认 Python `venv`、NumPy、PyTorch/ONNX 工具链，完成阶段二的 `model-tools`。
- [ ] 安装并确认 CUDA 编译链，运行 deviceQuery。
- [ ] 开始 NVIDIA Modern CUDA C++ Playlist。
- [ ] 跟完一个视频章节就复现并修改对应代码。
- [ ] 只在索引、API 或异步语义不清楚时，定点查询 CUDA Programming Guide。
- [ ] 建立自己的 CUDA 实验仓库。
- [ ] 完成 vector add、transpose、reduction。

### CUDA 基础过关后

- [ ] 完成 MNIST CUDA 的至少四级实现。
- [ ] 用 Compute Sanitizer 检查。
- [ ] 用 Nsight Systems 找流水线瓶颈。
- [ ] 用 Nsight Compute 分析一个热点 Kernel。
- [ ] 写出一份优化前后报告。

### 进入部署

- [ ] ONNX → TensorRT Engine。
- [ ] FP32/FP16/INT8 对比。
- [ ] 单路 → 4 路 DeepStream。
- [ ] Triton TensorRT backend + metrics。
- [ ] 支持矩阵内的 Edge-LLM 小模型。

### 再进入高级分支

- [ ] Linux module/driver/device tree。
- [ ] Triton DSL、CUTLASS/CuTe。
- [ ] TVM、MLIR。
- [ ] 更复杂的 LLM scheduler、KV cache 和多模型资源管理。

---

## 21. 学完整套后：能力水平、技术栈与岗位（原 section 16）

### 21.1 先给准确定位

如果你只是把视频看完，结果是“知道很多名词”，还不能据此判断达到岗位要求。如果你把每个阶段的项目、benchmark、性能报告和 Orin 毕业项目都真正完成，比较准确的定位是：

> **具备端到端 GPU 推理与边缘 AI 系统能力的初级到初中级工程师。**

你最终掌握的技术栈：

| 层次 | 主要技术 | 你应该能独立完成的事情 |
|---|---|---|
| 系统与工程基础 | C++17/20、Python、Bash、Linux、Git、CMake、gdb、strace、perf、Docker、ARM64 | 编译、调试、打包、排查进程/内存/权限/网络问题，维护可复现项目 |
| GPU 编程 | CUDA C++、nvcc、grid/block/thread、global/shared/register/unified memory、warp、streams、events、CUDA Graph | 写 vector/reduction/transpose/GEMM 等 Kernel，处理边界和异步执行，建立 CPU reference |
| GPU 性能工程 | coalescing、bank conflict、occupancy、divergence、arithmetic intensity、cuBLAS/cuBLASLt、FP16/INT8、Tensor Core | 通过数据而不是猜测优化 Kernel，并解释优化前后差异 |
| GPU 工具链 | Compute Sanitizer、Nsight Systems、Nsight Compute、cuobjdump、nvdisasm、PTX/SASS | 从错误、timeline、热点指标和指令结果形成“假设 → 修改 → 复测”闭环 |
| 模型推理优化 | PyTorch、ONNX、ONNX Runtime、TensorRT、`trtexec`、dynamic shape、calibration、plugin、Polygraphy | 完成 PyTorch → ONNX → Engine，核对精度并比较 FP32/FP16/INT8 的延迟和内存 |
| 边缘视觉 | Jetson Orin、JetPack/L4T、V4L2、GStreamer、DeepStream、tracker、metadata、`tegrastats` | 从单路扩展到多路视频，测量 FPS、端到端延迟、功耗、温度、内存和降频 |
| Serving | HTTP/gRPC、Docker Compose、Triton Server、TensorRT backend、model repository、dynamic batching、ensemble、Perf Analyzer、metrics | 把模型包装成有健康检查、超时、指标和并发测试的服务 |
| LLM 推理 | Transformer 推理、prefill/decode、KV cache、TTFT、TPOT、tokens/s、continuous batching、INT4/AWQ、TensorRT-Edge-LLM | 在 Orin 支持矩阵内选择小模型，做量化、上下文、并发和 unified memory 的可重复测试 |
| 高级分支 | Linux module/driver/device tree，Triton Language，CUTLASS/CuTe，TVM，MLIR | 在选择的一个分支上继续深入；不应把所有分支都宣称为同等熟练 |

### 21.2 最匹配的岗位

**第一梯队（完成核心主线后直接匹配）**：1. 推理优化工程师 / Inference Optimization Engineer（CUDA、TensorRT、ONNX、量化、Kernel、Nsight）；2. 边缘 AI 部署工程师 / Edge AI Deployment Engineer（Jetson、ARM64、JetPack、TensorRT、GStreamer、DeepStream）；3. TensorRT / GPU 性能工程师 / CUDA Software Engineer（CUDA C++、cuBLAS、shared memory、Tensor Core、Nsight）；4. Jetson / DeepStream / 计算机视觉部署工程师（多路视频、GStreamer、RTSP、tracker、metadata）。

**第二梯队（完成 Serving 与 LLM 后匹配）**：5. AI 推理平台工程师 / ML Inference Platform Engineer（Triton、gRPC、Docker、model repository、batching、metrics）；6. LLM 推理工程师 / LLM Inference Engineer（KV cache、TTFT/TPOT、量化、continuous batching、vLLM、SGLang、TensorRT-LLM、Triton）；7. 嵌入式 AI 软件工程师 / Embedded AI Systems Engineer（现代 C++、Linux、ARM64、Jetson、传感器、GStreamer）。

**高级分支岗位**：8. Triton Kernel / GPU Kernel Engineer（需完成分支 B，并增加 FlashAttention、算子融合、Triton/CUTLASS 源码和开源贡献）；9. ML Compiler Engineer（需完成分支 C，并补编译原理、LLVM/C++、IR 设计、pass 和 lowering）；10. Linux Driver / BSP Engineer（需完成分支 A，并补 C、设备树、DMA、中断、交叉编译、启动链和具体外设）。

### 21.3 求职关键词与证明材料

中文：`CUDA开发工程师`、`GPU性能优化工程师`、`推理优化工程师`、`TensorRT部署工程师`、`模型部署工程师`、`边缘AI工程师`、`Jetson开发工程师`、`DeepStream开发工程师`、`AI推理平台工程师`、`LLM推理优化工程师`、`嵌入式AI工程师`、`AI系统工程师`。

英文：`CUDA Software Engineer`、`GPU Performance Engineer`、`Inference Optimization Engineer`、`ML Systems Engineer`、`Edge AI Engineer`、`Jetson/Embedded AI Engineer`、`DeepStream Engineer`、`TensorRT Engineer`、`ML Inference/Serving Engineer`、`LLM Inference Engineer`、`Kernel Engineer`、`Compiler Engineer`。

不要把简历写成技术名词清单。至少准备四个可打开、可复测的证据：1. CUDA 性能项目（naive → tiled → library）；2. TensorRT 项目（ONNX、Engine、精度差异、FP32/FP16/INT8 benchmark）；3. Jetson/DeepStream 项目（多路视频、固定功耗报告）；4. Serving/LLM 项目（Triton 或 Edge-LLM 服务、并发曲线、TTFT/TPOT、OOM 边界）。

可参考岗位：[Shield AI Edge Systems](https://jobs.lever.co/shieldai/7e80ee36-5e20-4a4b-ba65-87d43db66cd6)、[昆仑芯模型加速与部署](https://kunlunxin.zhiye.com/xiangqing?jobId=151141586)、[百度异构计算](https://talent.baidu.com/jobs/detail/GRADUATE/15a59bf3-83f9-4c35-8d5e-bce6c50c59cc)、[Apple Model Inference](https://jobs.apple.com/en-in/details/200671782-0836/machine-learning-engineer-model-inference?team=MLAI)。

---

## 22. 关键链接索引（原 section 17）

### NVIDIA / CUDA

- [CUDA Platform](https://developer.nvidia.com/cuda)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)

### TensorRT / Jetson

- [JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Docs](https://docs.nvidia.com/jetson/index.html)
- [TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [TensorRT GitHub](https://github.com/NVIDIA/TensorRT)
- [DeepStream Docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream GitHub](https://github.com/NVIDIA/DeepStream)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/)
- [TensorRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/)

### Serving / Compiler

- [Triton Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Triton Server](https://github.com/triton-inference-server/server)
- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Language](https://triton-lang.org/main/)
- [Apache TVM](https://tvm.apache.org/docs/)
- [MLIR Tutorials](https://mlir.llvm.org/docs/Tutorials/)

### 基础 / Linux

- [MIT Missing Semester](https://missing.csail.mit.edu/)
- [LearnCpp](https://www.learncpp.com/)
- [Pro Git](https://git-scm.com/book/en/v2)
- [OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [Linux Foundation LFS101](https://training.linuxfoundation.org/training/introduction-to-linux/)
- [Linux Foundation LFD103](https://training.linuxfoundation.org/training/a-beginners-guide-to-linux-kernel-development-lfd103/)
- [Bootlin Docs](https://bootlin.com/docs/)

---

**执行原则：先把 CUDA Kernel 写对并测出来，再学 TensorRT；先把单模型部署稳定，再学 Triton；先把 Orin 上的小模型跑通，再讨论大模型 Serving。**
