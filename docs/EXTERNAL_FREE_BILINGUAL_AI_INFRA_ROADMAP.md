# 端侧AI Infra转岗路线图

**本机基线**：Jetson Orin NX 16GB（sm_87，unified memory）· JetPack 7.2.1 / L4T R39.2.1 · CUDA 13.2.1 · TensorRT 10.16.2 · DeepStream 9.1 · Ubuntu 24.04。核验日期 2026-08-21；版本敏感命令执行前必须复核官方兼容矩阵。

**目录**：1 总览 · 2 个人收藏区 · 3 工程与工具基础 · 4 CUDA 与 GPU 性能 · 5 深度学习基础原理 · 6 模型压缩与高效网络基础 · 7 模型结构、转换与推理引擎 · 8 容器化与推理服务 · 9 Transformer 与 LLM 概念 · 10 LLM 推理与 Edge-LLM · 11 可选支线 · 12 CUDA 专项执行清单 · 13 项目阶梯 · 14 概念澄清 · 15 阶段执行与复习原则 · 16 验收标准 · 17 版本纪律 · 18 中文资料用法 · 19 排除项 · 20 毕业项目 · 21 执行清单 · 22 能力与岗位 · 23 链接索引

## 1. 总览：学习路径与阶段地图

阶段一是工程底座，阶段二建立 CUDA 与 GPU 性能能力；阶段三到阶段五依次建立深度学习原理、模型压缩和模型部署闭环；阶段六学习通用推理服务；阶段七紧邻阶段八补齐 Transformer/LLM 概念。主线面向通用 GPU 推理与 Serving，LLM 推理必修；视觉流媒体、深度剪枝、算子 DSL、驱动和编译器按进入条件走支线。

主线依赖关系：

**Python / PyTorch / C++ / Linux / Docker 基础 → CUDA 与 GPU 性能 → 深度学习基础原理 → 模型压缩与高效网络基础 → 模型结构 / ONNX / TensorRT / GPU 预处理 → Docker Serving / HTTP / gRPC / Triton Server → Transformer 与 LLM 概念 → LLM Inference → Orin Edge-LLM**

可选支线（见第 11 章）：

**视觉流媒体（分支 A）**、**模型压缩实战（分支 B）**、**Triton Language → CUTLASS / CuTe（分支 C）**、**Linux 驱动 / BSP（分支 D）**、**TVM / MLIR（分支 E）**。

八阶段地图一览：

| 阶段 | 属性 | 核心交付 |
|---|---|---|
| 阶段一 · 工程与工具基础 | 基础必修 | model-tools + C++17/CMake CPU GEMM + 基础开发镜像 |
| 阶段二 · CUDA 与 GPU 性能 | 性能必修 | CUDA Kernel 仓库 + MNIST CUDA + Sanitizer/Nsight 报告 |
| 阶段三 · 深度学习基础原理 | 原理必修 | 神经网络训练、导出与推理流程说明 |
| 阶段四 · 模型压缩与高效网络基础 | 优化必修 | 量化/剪枝方案与硬件验证设计 |
| 阶段五 · 模型结构、转换与推理引擎 | 部署必修 | PyTorch→ONNX→TensorRT + GPU 预处理 pipeline |
| 阶段六 · 容器化与推理服务 | Serving 必修 | TensorRT backend Triton 服务 + HTTP/unary gRPC client |
| 阶段七 · Transformer 与 LLM 概念 | LLM 前置必修 | Transformer 与预训练/后训练知识图 |
| 阶段八 · LLM 推理与 Edge-LLM | 本机目标必修 | Orin Edge-LLM 服务 + 运行时选型表 |

---

## 2. 个人收藏 / 自定义资料区

> 这里存放你自己筛选、认为有用的视频与文档。路线不依赖此区，按需添加、自行维护。

### 基础补充

- [深入浅出数据结构（mycodeschool，42 讲已完结）](https://www.bilibili.com/video/BV1Fv4y1f7T1/)：链表/栈/队列/树/图的 C 实现（先修：C 指针）
- [4小时彻底掌握C指针（mycodeschool，已完结）](https://www.bilibili.com/video/BV1bo4y1Z7xf/)：指针与内存逐概念精讲

### 领域公开课（路线外补充，仅名校课程录像）

**GPU / 并行 / 系统**

- [斯坦福 CS149 并行计算 2023（中英字幕）](https://www.bilibili.com/video/BV1du17YfE5G/)：并行硬件、GPU 体系结构、性能优化（阶段二主课的 B 站版）
- [伯克利 CS267 并行计算应用 2022（中英字幕）](https://www.bilibili.com/video/BV1PS421978D/)：roofline、共享/分布式内存、MPI、GPU 编程
- [伯克利 CS162 操作系统](https://www.bilibili.com/video/BV1ab4y1b7BU/)：进程、虚存、并发、文件系统
- [斯坦福 CS107 计算机组织与系统 2016（中英字幕）](https://www.bilibili.com/video/BV1Nr421c7YB/)：C 与汇编、内存布局

**ML 系统 / LLM Infra**

- [CMU 10-414/714 机器学习系统](https://www.bilibili.com/video/BV1Rg4y137jH/)：从零实现 autograd、张量算子与编译
- [UCSD CSE 234 机器学习数据系统（2025，中英）](https://www.bilibili.com/video/BV1YSw4zDEF5/)：GPU/CUDA、算子编译、量化、推理服务（vLLM/PagedAttention/分离式架构）
- MIT 6.5940 EfficientML：已收进阶段四主线（B站 [BV1c8wNe1ErX](https://www.bilibili.com/video/BV1c8wNe1ErX)），不重复列出

**ML 原理 / 视觉 / RL**

- [台大李宏毅《生成式人工智能导论》2024](https://www.bilibili.com/video/BV16RkNYfEh4/)：LLM 与生成式 AI 导论（中文）
- [斯坦福 CS231n 视觉识别（2016）](https://www.bilibili.com/video/BV1Gx411Y75r/)：CNN 与计算机视觉
- [北邮 鲁鹏《计算机视觉与深度学习》](https://www.bilibili.com/video/BV1V54y1B7K3/)：计算机视觉系统课
- [西湖大学《强化学习的数学原理》](https://www.bilibili.com/video/BV1sd4y167NS/)：强化学习的数学基础
- [伯克利 CS285 深度强化学习 2026（中英字幕）](https://www.bilibili.com/video/BV1ryKG6pE7e/)：深度强化学习

**系统 / 分布式**

- [CMU 15-213 CSAPP《深入理解计算机系统》（2015）](https://www.bilibili.com/video/BV1iW411d7hd/)：信息的表示、链接、虚存、并发、系统 IO
- [MIT 6.824 分布式系统 2020（中英字幕）](https://www.bilibili.com/video/BV16M4m1m7YP/)：MapReduce、GFS、Raft、分布式一致性
- [斯坦福 CS144 计算机网络（中英字幕）](https://www.bilibili.com/video/BV1qotgeXE8D/)：TCP/IP、拥塞控制、路由

**体系结构 / 编译器**

- [CMU 15-418 并行计算机架构与编程（sp18，中英）](https://www.bilibili.com/video/BV18b421J7cA/)：多核/SIMD/GPU 体系结构与并行编程
- [北京大学《计算机组成》陆俊林](https://www.bilibili.com/video/BV1VE411o7nx/)：计算机组成（中文）
- [MIT 6.828 操作系统工程（2014，无字幕）](https://www.bilibili.com/video/BV1px411E7ST/)：xv6 实战操作系统内核
- [斯坦福 CS143 编译原理（中文字幕）](https://www.bilibili.com/video/BV1NE411376V/)：词法/语法分析、中间表示、代码生成

**AI Infra 学术研讨**

- [北大未名超算队 × LCPU：AI Infra Seminars——Inference & LLM Serving](https://www.bilibili.com/video/BV1gY4d6GEwR/)：推理系统与 LLM Serving 专题讲座

<!-- 后续在此区继续追加你认可的链接 -->

---

## 3. 阶段一 · 工程与工具基础（Python / PyTorch / C++ / Linux / Docker）

### 目标

Python = 部署侧胶水语言（导出、校准、benchmark、日志、客户端）；PyTorch = 训练资产 → ONNX/TensorRT 的桥梁；C++ = Kernel 与服务底座；Linux 用户态 = 开发排障前提；Docker = 环境封装与可复现工具。五条线都必须完成。

### 课程与资料

**Python：**

- 英文主课：[Python Full Course for free](https://www.youtube.com/watch?v=ix9cRaBkVe0)（freeCodeCamp，完整零基础课程）
- 中文速成（可选）：[3小时超快速入门 Python（动画教学，2026 新版）](https://www.bilibili.com/video/BV1Jgf6YvE8e/)（林粒粒呀，零基础动画课）
- 官方资料：[Python 官方教程](https://docs.python.org/3/tutorial/)、[venv 官方文档](https://docs.python.org/3/library/venv.html)、[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)、[NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

**PyTorch：**

- 主课（B站，中英字幕）：[mrdbourke《Learn PyTorch for Deep Learning in a Day》](https://www.bilibili.com/video/BV1kx4y1r79d/)（93 节，约 25.5 小时；配套代码见 [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)）
- 专题（只学前两节：1.1 课程先导、2.1 全连接神经网络整体结构）：[Pytorch框架与经典卷积神经网络与实战](https://www.bilibili.com/video/BV1e34y1M7wR/)（炮哥带你学）
- 原理补充：[PyTorch深度学习实践 完结合集（刘二大人）](https://www.bilibili.com/video/BV1Y7411d7Ys/)（B站，全部 13 讲）
- 官方校正：[Introduction to PyTorch on YouTube](https://docs.pytorch.org/tutorials/beginner/introyt/)、[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)、[PyTorch ONNX Tutorials](https://docs.pytorch.org/tutorials/beginner/onnx/index.html)

**C++：**

- 英文主课：[C++ Programming Course: Beginner to Advanced](https://www.youtube.com/watch?v=8jLOx1hD3_o)（freeCodeCamp）
- 辅助资料：[LearnCpp](https://www.learncpp.com/)

**Linux 用户态：**

- 基础层主课：[MIT Missing Semester（B站，中英字幕，IAP 2026）](https://www.bilibili.com/video/BV1CkArz1E4o/)（9 节约一周；Shell、管道、权限、脚本、SSH、Git、调试入门；英文原版见 [missing.csail.mit.edu](https://missing.csail.mit.edu/)）
- 系统层主课：[尚硅谷 Linux 应用层开发](https://www.bilibili.com/video/BV1DJ4m1M77z/)（文件 IO、进程线程、Socket、epoll 完整系列）
- 辅助资料：[Pro Git](https://git-scm.com/book/en/v2)、[OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)、[LFS101 Introduction to Linux](https://training.linuxfoundation.org/training/introduction-to-linux/)

**Docker 入门：**

- 英文主课：[Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)（freeCodeCamp，约 2 小时 10 分）
- 中文补课：[尚硅谷 Docker 与微服务实战 2024](https://www.bilibili.com/video/BV1Zn4y1X7AZ/)

##### 当阶段课程大纲（★ = 本阶段必修）

- **Python 主课（freeCodeCamp）** 大纲：
  - ★ 语法、条件/循环、函数、作用域
  - ★ 字符串、列表/字典/集合/元组
  - ★ 类与 dataclass、异常、模块/包
  - ★ 文件、路径、JSON/CSV
  - ★ 虚拟环境 venv、pip、requirements、版本锁定
  - ★ 命令行：argparse、环境变量、logging、subprocess
  - ★ NumPy：dtype/shape/stride/broadcast/向量化/矩阵乘
  - ★ OpenCV/PIL 图像读写、resize、批量遍历
  - ★ pytest、断言、随机种子、可复现 benchmark 输出
  - Web/爬虫/数据分析（跳过）
- **PyTorch（mrdbourke B站版 / 刘二）** 大纲：
  - ★ tensor 的 shape/dtype/device/layout/contiguous
  - ★ 索引与 reshape/permute、nn.Module/forward
  - ★ eval()、no_grad()/inference_mode()
  - ★ state_dict、checkpoint 加载、CPU/CUDA 转移
  - ★ Dataset/DataLoader
  - ★ ONNX 导出与 dynamic shape、数值对齐
  - 完整训练大型模型（略，本阶段只做推理/导出）
- **C++ 主课（freeCodeCamp）** 大纲：
  - ★ pointer、reference、stack/heap
  - ★ RAII、class、template、STL、lambda
  - ★ CMake、gdb、编译与链接
  - 复杂模板元编程/全栈（跳过）
- **MIT Missing Semester（基础层）** 大纲：
  - ★ Shell 与管道、权限
  - ★ Shell 脚本（写监控脚本）
  - ★ 命令行环境：SSH、任务控制
  - ★ Git 与调试入门（`gdb` / `strace` / profiling）
- **尚硅谷 Linux 应用层开发（系统层）** 大纲：
  - ★ 文件 IO（open/read/write、缓冲）
  - ★ 进程与线程（fork、pthread）
  - ★ Socket 网络编程
  - ★ epoll 多路复用
  - 内核/驱动部分（留到分支 A）
- **Docker 入门** 大纲：
  - ★ image、container 与 registry 的边界
  - ★ `docker pull/run/exec/logs/inspect`
  - ★ volume、端口、环境变量和基础 Dockerfile
  - ★ `linux/arm64` 与 Jetson 基础镜像识别
  - Compose、network、healthcheck、multi-stage 和 GPU runtime 留到阶段六

### 学习方法与停止条件

- **明确不学**：Django/Flask 全栈、爬虫、异步 Web、数据分析可视化、分布式训练、DeepSpeed。必修范围见上方大纲。
- **PyTorch 停止条件**：能加载公开 checkpoint 在固定输入上稳定得结果；能解释一次 shape/dtype/device/layout 错误；能用 `eval()` 和 `inference_mode()` 完成推理并正确处理 warmup 与 CUDA 同步；能把 PyTorch 模型导出 ONNX 并与 PyTorch 输出做容差比较；能说明训练、模型导出、TensorRT 构建和板端 Runtime 各自处于哪一层。
- **C++ 停止条件**：能解释值/引用/指针、对象生命周期、编译与链接，并能定位一次段错误。只抓 pointer/reference/stack-heap/RAII/class/template/STL/lambda/CMake/gdb。
- **Linux 停止条件**：能使用权限、管道、重定向、进程信号、日志、SSH、`gdb`、`strace`；知道 syscall、用户态和内核态的边界。CentOS 视频中的 `yum`、旧网络服务命令不复制到本机 Ubuntu 24.04；只迁移稳定概念。
- **Docker 停止条件**：能使用适配 `linux/arm64` 的基础镜像，把现有 PyTorch/ONNX CLI 推理工具封装为一个可复现的基础开发镜像，并在容器中运行。
- **阶段一基础停止条件**：能从零创建隔离环境并在另一台机器按文档复现；能读写常见模型/图像/JSON 文件；能解释 NumPy 的 shape/dtype/stride 与 PyTorch tensor 的关系；能把一次模型导出、校验和 benchmark 写成命令行工具；能定位依赖版本、路径、dtype 或 shape 错误。

### 阶段项目

在阶段一建立一个独立的 `model-tools` 目录，至少包含：

- `venv` 创建、依赖文件和一键运行说明；
- 用 `argparse` 扫描图片/视频数据集，生成带 hash 的 manifest；
- 用 NumPy 实现 CPU GEMM，并与 PyTorch 输出和耗时对齐；
- 读取图像并统一完成 RGB/BGR、resize、dtype 和 batch 处理；
- 导出一个简单 PyTorch 模型到 ONNX，并用 ONNX Runtime 做结果校验；
- 用 `subprocess` 调用外部 benchmark，输出 JSON/CSV；
- 用 logging 记录版本、输入 shape、随机种子和失败样本；
- 至少 5 个 pytest，覆盖空目录、错误路径、shape 和数值容差；
- 用基础 Dockerfile 将现有 PyTorch/ONNX CLI 推理工具打包为一个 ARM64-aware 开发镜像。

同时完成 C++17 + CMake CPU 矩阵乘法，随机输入校验并输出耗时；以及 Shell 监控脚本（采集 CPU、内存、温度、进程状态）+ 一个可并发处理请求的 C/C++ TCP 小服务。

### 面试向补充（独立于主线项目，求职前集中攻）

- **C++ 高频**
  - 智能指针：`unique_ptr` / `shared_ptr` / `weak_ptr`、引用计数原理、循环引用
  - 虚函数与 vtable：多态实现、纯虚函数、为什么基类析构要虚
  - 内存管理：new/delete vs malloc/free、内存泄漏定位、RAII
  - 移动语义：右值引用、`std::move` / `std::forward`、拷贝构造/赋值、Rule of Three/Five
  - STL 底层：vector 扩容、map（红黑树）vs unordered_map（哈希）、迭代器失效
  - 关键字：`const` / `static` / `inline`
  - 多线程：`thread` / `mutex` / `condition_variable` / `atomic`、死锁四条件
  - 编译链接：四阶段、静态库 vs 动态库、`extern "C"`
  - 追问级：lambda 引用捕获悬垂、虚继承与对象内存布局（概念级）
- **Python 高频**
  - GIL 与多线程 / 多进程 / 协程选型
  - 装饰器、生成器与迭代器、上下文管理器（`with`）
  - 深浅拷贝、可变 vs 不可变、函数传参是引用传递
  - 内存管理：引用计数 + 分代 GC、循环引用处理
  - 魔法方法与鸭子类型、闭包与 LEGB、`*args` / `**kwargs`
  - 追问级：`is` vs `==`、小整数缓存与字符串驻留、asyncio 事件循环（概念级）
- **PyTorch 高频（推理/部署向）**
  - autograd 动态图：计算图何时释放
  - `no_grad` / `eval` / `inference_mode` 三者区别
  - `view` / `reshape` / `contiguous` 与广播规则
  - `state_dict` 保存加载与 device 迁移
  - DataLoader：`num_workers` / `pin_memory`
  - AMP 混合精度：`autocast` / `GradScaler`
  - 常见内存泄漏：持有计算图引用、中间 tensor 累积
  - DDP 原理（进程级并行、梯度 all-reduce）；`torch.compile` 是什么
- **Linux 高频**
  - 进程 / 线程 / 协程区别与适用场景
  - 进程间通信：管道、共享内存、消息队列、socket、信号
  - I/O 多路复用：select / poll / epoll、LT vs ET（ET 为何必须配非阻塞 fd）
  - 虚拟内存、页表、fork 写时复制（COW）
  - 用户态 / 内核态与系统调用
  - 零拷贝：`mmap` / `sendfile`（与 GPU 部署的 zero-copy 呼应）
  - 文件描述符与 `ulimit`
  - 排查链：CPU 100% / 内存泄漏怎么用 `top` / `ps` / `perf` / `valgrind`
- **手写题（面试常考）**
  - C++：简化版 `shared_ptr`、线程安全单例、实现 String 类（Rule of Five）
  - Python：带参数装饰器、生成器实现生产者-消费者
  - PyTorch：完整训练循环、Multi-Head Attention 前向、自定义 `autograd.Function`

### 版本 / 边界注意

无强版本绑定；后续容器 / wheel 都要面向 ARM64/JetPack 7.2.1 验证。

---

## 4. 阶段二 · CUDA 与 GPU 性能

### 目标

CUDA 是整条路线的性能底座：写对 Kernel → 用架构知识解释快慢 → 用工具形成“证据 → 假设 → 修改 → 复测”闭环。验收是可复现的 Kernel 和性能报告，不是看完视频。

### 课程与资料

**CUDA C++ 入门：**

- 主课：[NVIDIA Fundamentals of Accelerated Computing with Modern CUDA C++](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)（EN / Official，完整系列）；同课中文镜像 [BV1QvSKB4EMr](https://www.bilibili.com/video/BV1QvSKB4EMr/)
- 补课：[CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)（EN，freeCodeCamp，约 12 小时），代码见 [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)
- 官方实验：[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)、[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Samples](https://github.com/NVIDIA/cuda-samples)、[CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)

**GPU 架构与 Kernel 优化：**

- 主课：[Stanford CS149 Parallel Computing 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)（EN / University，完整课程；最新讲次也可看 [CS149 fall25](https://gfxcourses.stanford.edu/cs149/fall25/)）
- 中文补课：[大规模并行处理器编程实战](https://www.bilibili.com/video/BV1gz421o7uH/)（PMPP 7 讲完整翻译课）
- 文档：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

**正确性与性能工具：**

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

- **CUDA 入门 · 看完即做**：vector add、SAXPY、二维矩阵加法；每个都有 CPU reference、边界输入、CUDA error check 和事件计时。停止条件：能不看答案写出索引，解释 grid/block/thread、异步 launch、global/shared/register/unified memory。
- **架构优化 · 看完即做**：naive/tiled transpose、naive/tree reduction、naive/tiled GEMM，并与 cuBLAS 比较。停止条件：能用 coalescing、bank conflict、occupancy、divergence、arithmetic intensity 解释性能，而不是只说“GPU 更快”。
- **工具链 · 固定顺序**：`compute-sanitizer` → `nsys` → `ncu` → `cuobjdump` / `nvdisasm`。看完即做：故意制造越界或 race 并修复；导出系统 timeline；对唯一热点 Kernel 采集 memory、SOL、occupancy、warp stall；反汇编前后两个版本。停止条件：交付一份“证据 → 假设 → 修改 → 复测”的性能报告，且 Sanitizer 干净。
- **Orin 边界**：围绕 Ampere `sm_87` 实验；Hopper/Blackwell 的 TMA、WGMMA 示例只看概念，不作为本机验收。

### 阶段项目

项目 A（CUDA 基础 Kernel 仓库）+ 项目 B（MNIST CUDA 逐级优化），见第 13 章。

### 版本 / 边界注意

本机 CUDA 13.2.1，编译目标 `sm_87`。Nsight：Jetson 采集、x86 host 分析；采集前固定功耗模式、输入规模、频率和温度。

---

## 5. 阶段三 · 深度学习基础原理

### 目标

建立训练模型、导出资产和部署推理之间的完整概念链，为阶段四的压缩方法和阶段五的模型结构/转换做准备。本阶段理解训练，但不开展大型训练或调参项目。

### 课程与资料

- 主课：**Deep Learning Specialization**（EN / Andrew Ng，保持课程连续性，只取训练与推理概念）：[deeplearning.ai](https://www.deeplearning.ai/courses/deep-learning-specialization/)。

##### 当阶段课程大纲（★ = 本阶段必修）

- **Deep Learning Specialization**：
  - ★ Course 1：参数、层、激活、损失、前向/反向传播、计算图和向量化
  - ★ Course 2：初始化、正则化和优化算法原理
  - ★ batch、训练/验证/导出/推理的区别
  - Course 2 调参实战和大型编程作业（跳过）
  - Course 3（跳过）
  - Course 4（移到视觉流媒体支线）
  - Course 5（仅作为阶段七查漏补缺）

### 学习方法与停止条件

- **课程连续性**：按 Course 1 → Course 2 的概念顺序学习，不把训练流程拆成零散术语。必修概念演练：能以一个小型 MLP 为例，在纸上或口头追踪 forward → loss → backward，解释计算图、梯度流、参数更新，以及导出后保留的前向计算。
- **停止条件**：能区分训练、验证、导出和推理的边界；能解释神经网络如何通过前向传播、损失、反向传播和参数更新完成训练，并指出优化器状态、梯度、反向传播、参数更新等训练专属状态或操作为何不出现在推理中。

### 阶段项目

无必做阶段项目；可选：用 PyTorch 实现小型 MLP 的 forward/loss/backward，或提交一份书面的计算图与梯度流追踪记录。

### 版本 / 边界注意

纯概念，无版本绑定。

---

## 6. 阶段四 · 模型压缩与高效网络基础

### 目标

本阶段建立完整的模型压缩概念与硬件约束。真实 ONNX/TensorRT 精度和速度验证在阶段五完成；更深入的剪枝训练与实验归入分支 B。参数量或 FLOPs 更小不等于已经证明加速。

### 课程与资料

- 主课：[MIT 6.5940 EfficientML 精译版（B 站）](https://www.bilibili.com/video/BV1c8wNe1ErX)（中文 / MIT，免费完整课程，含中英字幕；原版见 [efficientml.ai](https://efficientml.ai/)）
- 落地文档：[NVIDIA TensorRT INT8/PTQ 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- 论文：[AWQ](https://arxiv.org/abs/2306.00978)、[GPTQ](https://arxiv.org/abs/2210.17323)、[SmoothQuant](https://arxiv.org/abs/2211.03602)、[LLM.int8()](https://arxiv.org/abs/2208.07339)
- 工具文档：[Hugging Face Optimum 量化文档](https://huggingface.co/docs/optimum)
- 剪枝选学：[深度学习模型部署与剪枝优化实战](https://www.bilibili.com/video/BV1Sw411y7Hs/)

##### 当阶段课程大纲（★ = 本阶段必修）

- **MIT 6.5940 EfficientML、论文与配套文档** 大纲：
  - ★ PTQ、QAT、calibration、scale/zero-point
  - ★ per-tensor/per-channel、W8A8、W4A16
  - ★ structured/unstructured pruning、稀疏化和剪枝后微调
  - ★ 知识蒸馏
  - ★ depthwise convolution、MobileNet 与轻量结构
  - ★ EfficientML 高效 LLM：KV cache、W4/W8、GPTQ/AWQ/SmoothQuant/LLM.int8()
- 《深度学习模型部署与剪枝优化实战》：稀疏化、L1 正则、通道筛选、参数迁移、剪枝后微调
- 旧 Jetson Nano、旧 TAO 和旧安装命令（跳过）

### 学习方法与停止条件

- **量化关键字**：calibration 的 min-max / entropy / percentile 方法、outlier 与 clipping，以及 GPTQ（基于 Hessian）、AWQ（保护显著权重）和 SmoothQuant（迁移激活 outlier 到权重）的差异。
- **硬件约束**：同时考虑算子支持、实际精度格式、内存带宽、结构化稀疏支持和 Engine tactic；参数量或 FLOPs 只能描述理论规模，不能单独证明目标硬件上的加速。
- **阶段边界**：本阶段完成量化、剪枝、稀疏化、蒸馏和轻量结构的方案设计；阶段五再通过 ONNX/TensorRT 验证精度与速度，更深的剪枝训练和实验留到分支 B。
- **停止条件**：能针对给定模型、硬件和精度目标提出量化/剪枝方案，说明风险，并定义验证方法。

### 阶段项目

编写一份压缩验证方案：定义 accuracy delta、TensorRT Engine 构建日志与实际 tactics、延迟、吞吐和内存检查；由阶段五的项目 C 执行并记录结果。

### 版本 / 边界注意

量化与 calibration 工具链以本机 TensorRT 10.16.2 为准；旧课程中的命令不得直接复制，必须按当前官方文档和本机版本校正。论文按需读摘要。

---

## 7. 阶段五 · 模型结构、转换与推理引擎

### 目标

先读懂陌生 PyTorch 模型的 Module、参数、checkpoint、tensor shape 和完整 forward 数据流，再把模型结构阅读、ONNX/TensorRT 转换与 GPU 预处理串成一个可复现的部署闭环。

### 课程与资料

**模型结构阅读：**

- **李沐 动手学深度学习 D2L v2**（中文 / Open，含 PyTorch 代码）：[B站 BV1Z5411n7RB](https://www.bilibili.com/video/BV1Z5411n7RB/)；教材见 [d2l.ai](https://d2l.ai/)。本阶段按下面大纲连续学习一个精选区块，不重看完整系列。

**ONNX 与 TensorRT：**

- 主课：[NVIDIA TensorRT 教程 4 部分](https://www.bilibili.com/video/BV15Y4y1W73E)（中文 / NVIDIA Official，完整系列）
- 补课：[Inference Optimization with NVIDIA TensorRT](https://www.youtube.com/watch?v=UnIuMXGylfY)（EN / NCSA，完整 workshop）
- 文档：[TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[TensorRT 10.x Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/quick-start-guide.html)、[TensorRT GitHub](https://github.com/NVIDIA/TensorRT)、[Torch-TensorRT](https://github.com/pytorch/TensorRT)
- 中文入口：[TensorRT 中文入门](https://developer.nvidia.cn/tensorrt-getting-started)、[NVIDIA 中文超级训练](https://www.nvidia.cn/developer/online-training/super-training/)（只看概念，命令以英文最新文档为准）

**GPU 预处理与 DALI：**

- 主课文档：[NVIDIA DALI 文档与 Tutorials](https://docs.nvidia.com/deeplearning/dali/)、[DALI GitHub](https://github.com/NVIDIA/DALI)
- 对比库：[CV-CUDA](https://github.com/CVCUDA/CV-CUDA)（推理专用预处理 GPU 加速）
- 主课视频（需免费注册 NVIDIA Developer 后观看）：[GTC 2020 “Fast Data Pre-Processing with NVIDIA Data Loading Library (DALI)”](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21139)
- 补充免费视频（约 53 分钟，无需登录）：[NVIDIA DALI Data Loading Library 实战讲解](https://www.youtube.com/watch?v=PTWER9HIVHM)

#### 当阶段课程大纲（★ = 本阶段必修）

- **李沐 动手学深度学习 D2L v2（连续精选区块，不学完整系列）** 大纲：
  - ★ Module、参数、checkpoint 和模型结构读取
  - ★ MLP、CNN、BatchNorm、ResNet
  - ★ RNN/Seq2Seq、Attention、Transformer
  - ★ tensor shape 和完整 forward 数据流
  - 优化算法、完整训练、Kaggle、检测/分割、GAN、推荐系统和分布式训练（跳过）
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
- **辅助文档（非视频）：** 使用上方 DALI 官方文档、Tutorials 与 GitHub；CV-CUDA 作为推理专用预处理的对比与扩展阅读。

### 学习方法与停止条件

- **模型结构阅读 · 连续学习**：沿着 Module → 参数/checkpoint → tensor shape → 完整 forward 数据流阅读一个陌生模型，先解释每个阶段的输入输出和推理所需资产，再开始导出；只学上述 D2L 精选区块，不把它扩成完整训练课。
- **ONNX/TensorRT · 看完即做**：PyTorch 导出 ONNX、ONNX Runtime 对齐、`trtexec` 构建 Engine、FP32/FP16/INT8、dynamic shape、warmup 后 benchmark。停止条件：能解释 Engine 为什么与目标 GPU/版本绑定，能报告 accuracy delta、p50/p95、throughput 和峰值内存。
- **阶段四压缩方案验证 · 必修**：逐项执行阶段四提出的压缩验证方案，检查 PyTorch/ONNX 数值对齐、TensorRT 构建日志与实际 tactics、accuracy delta、延迟、吞吐和内存，并对比 FP32/FP16/INT8；适用时还要验证 structured sparsity。参数量或 FLOPs 降低只能说明理论规模变化，不能单独作为验收依据。
- **GPU 预处理 · 必修关键字**：decode 卸载、pinned memory、异步 pipeline、zero-copy、host↔device 拷贝、batch 拼接、DALI ↔ TensorRT 直连、CPU 预处理导致的 GPU 空泡。看完即做：用 DALI 搭一个 decode + resize + normalize 的 GPU pipeline，喂给 TensorRT/PyTorch；与 OpenCV/CPU 版本在同一输入上对比端到端吞吐，并用 `nsys` 看预处理阶段是否还占 GPU 空泡。停止条件：能指出当前 pipeline 的预处理是否瓶颈、是否应上 DALI/CV-CUDA、pinned memory 与异步带来的提升；知道 DeepStream（阶段五完成后进入的视觉分支）的 NVMM 零拷贝正是同一思想的硬件实现。
- **阶段五停止条件**：能把一个陌生 PyTorch 模型转换为可复现构建的 TensorRT Engine，报告 accuracy、latency、throughput、memory 和预处理瓶颈，并给出输入、版本与 benchmark 条件。
- **面试向**：INT8 对称 vs 非对称量化、per-channel 为什么更准、W4A16 是什么。

### 阶段项目

项目 C（TensorRT 推理部署，见第 13 章）+ 一个 DALI GPU 预处理 pipeline 与 OpenCV/CPU 的吞吐对比；同时执行阶段四压缩方案的精度、tactics、性能与内存验证。

### 版本 / 边界注意

**强版本警告**：视频基于 TensorRT 8.x，本机 10.16.2；概念照学，实现以 [TensorRT 10.x Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/quick-start-guide.html) 与当前 samples 为准。

---

## 8. 阶段六 · 容器化与推理服务（Docker / HTTP / gRPC / Triton Server）

### 目标

在阶段一 Docker 入门基础上补齐容器化部署、HTTP/unary gRPC 客户端语义和 Triton Server 服务层，形成 TensorRT Engine → Triton backend → 可观测客户端 benchmark 的完整服务链。Triton 解决 batching、ensemble、metrics 等服务层问题，不替代 TensorRT runtime/Kernel。

### 课程与资料

**Docker：**

- 阶段一已完成 Docker 初学内容；这里仅按需回看英文课：[Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)（freeCodeCamp，约 2 小时 10 分）
- 中文补课：[尚硅谷 Docker 与微服务实战 2024](https://www.bilibili.com/video/BV1Zn4y1X7AZ/)

**HTTP / gRPC：**

- 英文主课：[Getting Started With gRPC: Hands-On Codelab](https://www.youtube.com/watch?v=kAuK6VcAR10)（CNCF，约 75 分钟）
- 中文选修：[手把手 gRPC 基础教程](https://www.bilibili.com/video/BV1QT411H7ds/)（14 讲完整系列；需要自建完整 server 或学习流式 RPC 时再看）

**Triton Inference Server：**

- 主课：[NVIDIA Triton 从入门到精通](https://www.bilibili.com/video/BV1KS4y1v7zd/)（中文 / NVIDIA Official，20 讲完整系列）
- 补课：[Getting Started with NVIDIA Triton](https://www.youtube.com/watch?v=NQDtfSi5QF4)（EN / NVIDIA，完整入门专题）
- 文档：[Triton Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)、[Triton Tutorials](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/README.html)、[Triton Server](https://github.com/triton-inference-server/server)、[Triton Tutorials Repo](https://github.com/triton-inference-server/tutorials)、[NVIDIA 中文 TensorRT/Triton 课程入口](https://www.nvidia.cn/developer/online-training/super-training/)

#### 当阶段课程大纲（★ = 本阶段必修）

- **Docker 进阶（阶段一已学入门）** 大纲：
  - ★ multi-stage Dockerfile 与 Docker Compose
  - ★ network、healthcheck 与 GPU runtime
  - ★ 构建并确认支持 `linux/arm64` 的镜像（Jetson 部署）
- **HTTP 最小集** 大纲：
  - ★ request/response、status 与 JSON/binary tensor
  - ★ timeout 与 health check
- **gRPC 最小集（CNCF codelab）** 大纲：
  - ★ `.proto`、生成 stub 与 RPC 语义
  - ★ unary RPC
  - ★ deadline/timeout 与 error codes
  - 自己实现完整 gRPC server、server/client/bidirectional streaming，以及中文 14 讲完整课程（选修）
- **NVIDIA Triton 从入门到精通（B站 20 讲）** 大纲：
  - ★ 概念与 model repository
  - ★ TensorRT backend
  - ★ 官方 Triton HTTP client 与 unary gRPC client
  - ★ dynamic batching、instance group
  - ★ ensemble、metrics
  - ★ Perf Analyzer 与并发/延迟曲线（架构概念可学；配置字段以当前 Triton 文档为准）
- **Getting Started with NVIDIA Triton（官方专题）** 大纲：
  - ★ 安装、模型仓库、基础 client
  - ★ 最新版官方术语入口

### 学习方法与停止条件

- **Docker**：回看阶段一，不把镜像/容器初学内容当作新基础；本阶段完成 multi-stage Dockerfile、Compose、network、healthcheck、GPU runtime，并确认基础镜像支持 `linux/arm64`。
- **HTTP / gRPC 最小实践**：分别用官方 Triton HTTP client 和 unary gRPC client 发送推理请求；覆盖 JSON/binary tensor、status、health check、timeout/deadline 与 error codes。无需实现自定义 gRPC predict server，也无需掌握三种 streaming RPC。
- **Triton 看完即做**：model repository、TensorRT backend、dynamic batching、instance group、ensemble、metrics、Perf Analyzer。停止条件：画出 concurrency/batch 与 p50/p95/throughput 曲线，并能说明 Triton 解决的是服务层问题，不是替代 TensorRT runtime/Kernel。
- **阶段六停止条件**：能解释容器内外文件与端口映射、HTTP 与 unary gRPC 的差异、deadline/timeout 与错误处理，以及 Triton 的 backend、调度、batching、ensemble 和 metrics 行为。
- **版本注意**：中文 Triton 课基于 2022 年版本，架构概念仍可学；容器 tag、backend 支持和配置字段查当前 [Triton tutorials repo](https://github.com/triton-inference-server/tutorials) 与 Jetson platform matrix。2026 官方页面也可能使用 Dynamo-Triton 名称。
- **面试向**：TCP 三次握手/四次挥手与 TIME_WAIT；gRPC 为什么快（HTTP/2 多路复用 + protobuf）；Docker 隔离原理（namespace/cgroup）。

### 阶段项目

项目 E（Triton 模型服务，见第 13 章）：把 TensorRT backend 服务打包进适配 ARM64/GPU runtime 的容器，用官方 Triton HTTP client 和 unary gRPC client 完成请求，提供 health、metrics 与 timeout/deadline 行为，并测量 concurrency、batching、latency 和 throughput。

### 版本 / 边界注意

确认基础镜像支持 `linux/arm64`。Triton 的 backend/platform 组合不是全部在 Jetson 可用，尤其不要默认 Python backend 的 GPU 能力与 x86 服务器一致；以 [官方 backend platform matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md) 为准。

---

## 9. 阶段七 · Transformer 与 LLM 概念

### 目标

集中建立 Transformer、预训练、后训练与自回归生成的概念链，理解训练资产如何成为可部署的推理模型；本阶段不做训练项目。

### 课程与资料

- 主课：**Stanford CS224n Spring 2024**（EN / University，固定使用此 B 站合集）：[B站 BV163Jc6pENx](https://www.bilibili.com/video/BV163Jc6pENx/)。只连续学习 L7–L11；其余讲次和教程按下面边界处理。
- 闭卷回顾：阶段五的 [D2L Attention / Transformer](https://d2l.ai/) 内容，只画图和复述，不重看完整课程。
- 查漏补缺：[Deep Learning Specialization Course 5](https://www.deeplearning.ai/courses/deep-learning-specialization/) 只作为序列模型基础薄弱时的补课。

#### 当阶段课程大纲（★ = 本阶段必修）

- **Stanford CS224n Spring 2024** 大纲：
  - ★ L7 Attention / LLM Introduction
  - ★ L8 Self-Attention and Transformers
  - ★ L9 Pretraining
  - ★ L10 Post-training
  - ★ L11 Natural Language Generation
  - ★ tokenizer、embedding、encoder/decoder 补充
  - ★ BERT vs GPT、pretraining vs post-training
  - L5–L6：仅在 RNN/Seq2Seq/Attention 前置薄弱时补看
  - 其它 lectures/tutorials（跳过或选修）；benchmarking lecture 可选
- **D2L Attention / Transformer**：闭卷画出 Q/K/V、multi-head attention、残差与 encoder/decoder 数据流；卡住时只查对应教材，不重看完整课程。
- **Deep Learning Specialization Course 5**：仅查漏补缺，不作为第二门主课。

### 学习方法与停止条件

- **概念输出**：画出 Transformer 架构和 tokenizer → embedding → attention/MLP → logits → decoding 的生成流，逐层标注 tensor shape、训练资产与推理状态。
- **对比输出**：用同一张表解释 BERT 与 GPT、encoder 与 decoder、pretraining 与 post-training 的目标和推理方式差异。
- **阶段七停止条件**：能解释训练得到的 tokenizer、权重、配置和精度格式如何组成推理模型，并完整说明 Transformer 架构与自回归生成流；不以训练或调参项目作为验收。

### 阶段项目

无训练项目；提交 Transformer 与生成流图，以及 BERT/GPT、pretraining/post-training 对比表。

### 版本 / 边界注意

纯概念，无版本绑定；仅固定使用上述 Stanford CS224n Spring 2024 B 站合集，避免讲次错位。

---

## 10. 阶段八 · LLM 推理与 Edge-LLM

### 目标

在阶段七概念基础上学习 LLM 推理系统（CS336）→ 横向选择运行时（不绑一家）→ 在 Orin 部署小模型服务，并在支持矩阵内完成可复现的内存、功耗与性能测量。

### 课程与资料

**LLM 推理原理：**

- 主课：[Stanford CS336 2026 Video Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)（EN / University，按模型结构、推理、评测与系统主题选学，重点第 10 讲）；作业与讲义见 [CS336 官方课程页](https://cs336.stanford.edu/)；[CS336 Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM)
- 中文补课：[大模型推理技术研究](https://www.bilibili.com/video/BV1k2L9zyEt7/)（KV cache、vLLM、SGLang 等 9 讲完整系列）

**边缘 LLM 运行时选型：**

- 入口：[Jetson AI Lab 边缘 LLM 教程](https://www.jetson-ai-lab.com/)（含 llama.cpp、MLC-LLM、Edge-LLM 对比指南；中文实操见 [GitHub 教程](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)）
- 运行时：[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)（[文档](https://nvidia.github.io/TensorRT-Edge-LLM/)、[Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)）、[llama.cpp](https://github.com/ggerganov/llama.cpp)（GGUF）、[MLC-LLM](https://github.com/mlc-ai/mlc-llm)（TVM 系）、[ExecuTorch](https://github.com/pytorch/executorch)（PyTorch Edge）
- 对比视频：[Run GenAI Locally on NVIDIA Jetson](https://www.youtube.com/watch?v=czteUSONG-c)（Ollama / vLLM / llama.cpp 选型与快速上手）
- 文档参考：[vLLM Docs](https://docs.vllm.ai/en/latest/)、[TensorRT-LLM Docs](https://nvidia.github.io/TensorRT-LLM/)

**Orin Edge LLM：**

- 主课：[Make It Think: NVIDIA Jetson AI Lab](https://youtube.com/playlist?list=PLZrTAEPLeXfo)（EN / NVIDIA，2026，3 场完整系列）
- 补课：[Getting Started with Edge AI on NVIDIA Jetson](https://www.youtube.com/watch?v=t2Ecuu2FdC8)（EN / NVIDIA，完整直播课）
- 落地文档：[TensorRT-Edge-LLM 官方安装与教程](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)

#### 当阶段课程大纲（★ = 本阶段必修）

- **Stanford CS336 2026** 大纲：
  - ★ 模型结构与推理所需训练资产（承接阶段七，不重复学习 Transformer 入门）
  - ★ Lecture 10：Inference（prefill/decode、KV cache、continuous batching）
  - ★ 推理 evaluation、benchmarking 与 systems 内容
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

- **LLM 推理 · 看完即做**：用同一小模型和固定 prompt，对 batch、concurrency、queueing、context length 和量化方式做可复现 benchmark。停止条件：能严格区分 TTFT、TPOT、端到端 latency、tokens/s、单请求延迟和系统 throughput，并解释 prefill/decode 的瓶颈差异，以及 KV cache、continuous batching、PagedAttention 如何改变内存与调度。框架关系：vLLM、SGLang、TensorRT-LLM/Edge-LLM 是 LLM runtime/engine 路线；Triton 是可选的通用服务层，二者不是同一个层级。
- **量化与内存预算**：对比 FP16 与 INT4/AWQ/GPTQ，按权重、KV cache、临时 buffer 和 context length 估算 unified memory；把精度变化、OOM 边界和可用上下文写入报告。
- **运行时选型 · 必修关键字**：GGUF vs safetensors、weight-only 量化格式、runtime 对 AWQ/GPTQ 的支持、CUDA vs CPU offload、KV cache 管理、OpenAI-compatible API、易用性 vs 峰值性能 vs 内存占用。看完即做：在 TensorRT-Edge-LLM、llama.cpp、MLC-LLM、ExecuTorch 中按模型格式与支持矩阵选择，并让同一小模型（如 Qwen3 0.6B）至少在两个运行时（建议 TensorRT-Edge-LLM + llama.cpp）上跑；固定 prompt 与上下文，测量 TTFT、TPOT、tokens/s、峰值 unified memory 与量化格式。停止条件：交付一张选型表，标明每个运行时在 Orin 上的适用场景（快速验证 / 最高吞吐 / 最低内存 / 跨平台），并说明为何落地仍优先 TensorRT-Edge-LLM 但不排斥其他。
- **Edge LLM · 看完即做**：从支持矩阵内的小模型开始，先 FP16 基线，再 INT4/AWQ；在 Orin 上提供可重复启动的简单 API 服务，测 TTFT、TPOT、tokens/s、throughput、concurrency/queueing、峰值 unified memory、功耗、温度和 context length。停止条件：连续请求无 OOM，报告包含 runtime、service、模型、量化、内存预算、功耗、性能、并发和版本选择。资料现实：截至 2026-08-21，还没有一门公开、完整且与当前 TensorRT-Edge-LLM release 同步的官方专项视频课；这里用最新 Jetson 完整视频系列建立工作流，代码只跟 [TensorRT-Edge-LLM 官方安装与教程](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)，不拿旧 TensorRT-LLM 服务器课程冒充 Orin 课程。
- **阶段八停止条件**：能在 Orin 部署一个小模型服务，并根据支持矩阵与测量结果说明 runtime、quantization、memory、power 和 performance 选择。
- **面试向**：KV cache 为什么省算力；prefill 与 decode 的瓶颈差异；continuous batching 解决什么问题。

### 阶段项目

项目 F（Orin Edge LLM），见第 13 章；运行时选型表（≥2 家对比）作为前置交付。

### 版本 / 边界注意

本机 Orin NX 16GB、JetPack 7.2.1；Edge-LLM 代码以 TensorRT-Edge-LLM 当前 release 为准。先选支持矩阵内小模型（Qwen3 0.6B/1.7B/4B），不盲目移植 14B 以上；不把 server GPU 上的 vLLM/TRT-LLM 命令直接复制到 Jetson。CS336 Lecture 6（Kernels / Triton / XLA）不在本阶段课程大纲内，专属分支 C。

---

## 11. 可选支线（视觉 / 压缩 / 算子 / 驱动 / 编译器）

### 分支 A：视觉流媒体（Jetson / GStreamer / DeepStream）

**进入条件**：完成阶段五后再进入。本支线是可选视觉方向，主线不要求完成；Deep Learning Specialization Course 4 只作为本支线的可选先修或 CNN 概念补课，不是主线必修。

#### 课程与资料

**Jetson AI 基础：**

- 主课：[NVIDIA Jetson AI Fundamentals](https://www.youtube.com/playlist?list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8)（EN / Official，完整系列）；同课中文镜像 [BV1EGSmBWErR](https://www.bilibili.com/video/BV1EGSmBWErR/)
- 中文补课：[NVIDIA Jetson 边缘 AI 快速上手系列](https://www.bilibili.com/video/BV1yEzBYQEMt/)（Seeed Studio，完整系列）
- 项目起点：[dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
- 可选先修 / 补课：[Deep Learning Specialization Course 4: Convolutional Neural Networks](https://www.deeplearning.ai/courses/deep-learning-specialization/)（只在 CNN、classification、detection、segmentation 概念不牢时学习）

**GStreamer 与 DeepStream：**

- 主课：[Create Vision AI Applications With DeepStream](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81879/?playlistId=gtc26-computer-vision-and-video-analytics)（EN / NVIDIA，2026，2 小时完整 workshop）
- 中文补课：[深度学习模型部署与剪枝优化实战](https://www.bilibili.com/video/BV1Sw411y7Hs/)（只选 GStreamer / DeepStream 单元）
- 文档：[DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)、[DeepStream GitHub](https://github.com/NVIDIA/DeepStream)、[DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)、[DeepStream 中文开发入口](https://developer.nvidia.cn/deepstream-sdk)、[Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/)

#### 分支课程大纲（★ = 本支线必修）

- **Deep Learning Specialization Course 4（可选先修 / 补课）** 大纲：
  - CNN、卷积 / 池化、经典视觉网络的基本概念
  - classification、detection、segmentation 的任务差异
- **NVIDIA Jetson AI Fundamentals（官方 playlist）** 大纲：
  - ★ Hello AI World 与 jetson-inference 安装（忽略 Nano/JetPack 4 命令）
  - ★ 图像分类 classification
  - ★ 目标检测 detection
  - ★ 语义分割 segmentation
  - ★ 摄像头与实时推理、训练与迁移学习（PyTorch）
  - ★ `tegrastats` 与各阶段耗时拆解（模型加载 / 预处理 / 推理 / 后处理 / 捕获 / 显示）
- **Seeed Jetson 边缘 AI 快速上手系列（B站）** 大纲：
  - ★ Jetson 上手、摄像头与边缘 AI 工作流
  - 进阶项目（按需）
- **Create Vision AI Applications With DeepStream（2026 GTC workshop）** 大纲：
  - ★ GStreamer 基础与 pipeline 概念（caps、buffer、metadata）
  - ★ DeepStream 插件：`nvinfer` / tracker / tiler / metadata
  - ★ 文件、RTSP 与摄像头输入，单路 pipeline → 多路 pipeline
  - ★ 性能与功耗测量（FPS、延迟、温度、内存）
- **深度学习模型部署与剪枝优化实战（B站）** 大纲：
  - ★ GStreamer / DeepStream 相关单元
  - 其它通用部署与剪枝内容（转到分支 B 或按需）

#### 学习方法与停止条件

- **Jetson 基础 · 看完即做**：用 `jetson-inference` 依次跑 classification、detection、segmentation，再接 USB/CSI 摄像头。停止条件：能区分模型加载 / 预处理 / 推理 / 后处理 / 捕获 / 显示耗时，并记录 `tegrastats`。
- **DeepStream · 免费边界**：On-Demand 视频免费观看；要登录只注册免费账号，不买 DLI。看完即做：文件单路 → 摄像头单路 → 4 路文件/RTSP；加入 decode、mux、`nvinfer`、tracker、tiler、metadata、sink。停止条件：能画出 pipeline，解释 caps / buffer / metadata / batch / zero-copy，并报告每路 FPS、端到端延迟、温度、功耗和内存。

#### 项目与版本边界

- **可选视觉项目**：项目 D（Jetson 多路视频推理），见第 13 章；入门先用 `jetson-inference` 跑通三类任务。
- **版本警告**：课程中的 Jetson Nano / JetPack 4 命令全部忽略；本机边界是 JetPack 7.2.1、DeepStream 9.1，旧课安装命令、插件字段和 Python binding 不可复制，从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 当前 release 开始。
- **与阶段五的关系**：DeepStream 的 NVMM 零拷贝与阶段五 DALI 的 zero-copy 是同一类减少数据搬运的思想；这里把它落到视频 decode、batch 和推理 pipeline。

### 分支 B：模型压缩实战

**进入条件**：完成阶段五，并先建立可复现的未剪枝 TensorRT baseline；固定模型、数据集、输入 shape、精度、功耗模式、warmup 和测量方法，记录 baseline 的 accuracy、latency、throughput 与 memory。

#### 课程与资料

- 主课选段：[深度学习模型部署与剪枝优化实战](https://www.bilibili.com/video/BV1Sw411y7Hs/)（只学更深入的剪枝实现单元）
- 官方文档：[TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[TensorRT 10.x Structured Sparsity](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html)

#### 分支课程大纲（★ = 本支线必修）

- **路径一：channel / filter pruning（改变模型结构）**
  - ★ 通道 / filter 筛选、参数迁移与 post-pruning finetuning；这种剪枝会改变网络通道数或层结构
  - ★ 导出新结构的 ONNX，完成 PyTorch → ONNX Runtime 数值对齐，再按普通 TensorRT 流程构建和 benchmark
  - ★ 不把通道 / filter pruning 当作 2:4 权重稀疏，也不要求或暗示它会触发 sparse tactic
- **路径二：Ampere 2:4 structured sparsity（保持可稀疏加速的权重模式）**
  - ★ sparse training 与稀疏度调度，理解 L1 正则的作用与局限，并让权重满足 TensorRT 要求的 2:4 pattern
  - ★ post-pruning finetuning 后导出 ONNX，完成 PyTorch → ONNX Runtime 数值对齐
  - ★ 运行 `polygraphy inspect sparsity model.onnx`，验证 ONNX 权重确实满足 2:4 sparsity pattern
  - ★ 使用 TensorRT 支持的 FP16 或 INT8 precision，并通过 `trtexec --sparsity=enable` 或等价 TensorRT builder flag 启用 sparse tactics
  - ★ 保存 verbose build log，明确区分哪些 layer 具备 sparse tactic 资格，以及 builder 最终实际选择了哪些 sparse tactics
- **两条路径共同验收**：★ 在相同目标硬件与测量条件下对比剪枝前后的 accuracy、latency、throughput、memory

#### 学习方法与停止条件

从未剪枝 TensorRT baseline 开始，只改变一个剪枝变量，并明确选择 channel / filter pruning 或 Ampere 2:4 structured sparsity 路径。前者按结构变化 → finetuning → ONNX export → 普通 TensorRT benchmark 验证；后者按 2:4 sparse training → finetuning → ONNX export → sparsity inspection → sparse-enabled TensorRT build 与 benchmark 验证。参数量或 FLOPs 更小只说明理论规模变化，不是实际加速证明。

停止条件：两条路径都必须展示目标硬件上的端到端 accuracy、latency、throughput 和 memory 影响。channel / filter pruning 以普通 TensorRT benchmark 为准，不以 sparse tactic 为验收条件；2:4 路径还必须用 verbose build log 证明 eligible layer 是否实际选用了预期 sparse tactic。若剪枝后端到端没有收益，或 2:4 路径未选 sparse tactic，也要明确记录 no-benefit 结果、构建日志和原因分析，不能只汇报参数量或 FLOPs。

### 分支 C：Triton Language、CUTLASS 和 CuTe

**进入条件**：完成阶段二后可进入算子分支。这里的 Triton 是 Kernel DSL，不是阶段六的 Triton Inference Server；CS336 Lecture 6 专属本支线，不放入主线课程。

#### 课程与资料

- 主课：[Triton 从入门到大师](https://www.bilibili.com/video/BV1fMyWBgERM/)（中文，10 讲完整课）
- 专题：[Stanford CS336 2026 Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)（EN / University）
- 代码：[triton_docs_tutorials](https://github.com/evintunador/triton_docs_tutorials)
- 高级系列：[GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)（CUTLASS/CuTe 讲次）
- 官方文档：[Triton Language](https://triton-lang.org/main/)、[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)、[Triton GitHub](https://github.com/triton-lang/triton)、[CuTe tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)、[CUTLASS](https://github.com/NVIDIA/cutlass)

#### 分支课程大纲（★ = 本支线必修）

- **Triton 从入门到大师（B站 10 讲）** 大纲：
  - ★ Triton 编程模型与 program instance
  - ★ vector add → fused softmax → matmul → layer norm
  - ★ autotune 与 tile
  - ★ 与 CUDA 版本做正确性 / 性能 / 可读性对比
- **CS336 Lecture 6: Kernels, Triton, XLA** 大纲：
  - ★ kernel 实现视角下的 Triton / XLA
  - ★ 写算子时配套观看
- **GPU MODE Lectures（CUTLASS / CuTe 讲次）** 大纲：
  - ★ CUDA / Triton / CUTLASS / CuTe 相关讲次
  - ★ GEMM tile / layout / MMA 与 CUTLASS profiler
  - ★ CuTe GEMM 实战理解 layout / tile / MMA
- **CuTe tutorial（官方文档）** 大纲：
  - ★ CuTe 快速上手与 layout 概念
  - ★ 一个 CuTe GEMM 项目

#### 学习方法与停止条件

Triton 课后项目：vector add → fused softmax → matmul → layer norm；与 CUDA 版本做 correctness、性能和可读性对比。CUTLASS/CuTe 以 GPU MODE 为完整课程载体，再进入官方 CuTe tutorial 做项目。停止条件：能解释 program instance、tile、layout、MMA、autotune 的边界，并知道何时选 CUDA、Triton 或 CUTLASS。Triton 适合在 x86 NVIDIA GPU 或公开 notebook 环境学习；不要默认当前 Triton、PyTorch 和 ARM64 Jetson 组合可原生安装。

### 分支 D：Linux 驱动 / BSP

**进入条件**：完成阶段二后再进入。仅面向驱动 / BSP 岗位选择本支线；除非岗位明确要求，否则它不是主线。支线内顺序：Linux 用户态 → module → 字符设备 → ioctl/poll/mmap → DMA/中断 → device tree → Jetson BSP → 调试与性能。

#### 课程与资料

- 英文主课：[Linux Device Drivers Development](https://www.youtube.com/watch?v=iSiyDHobXHA)（freeCodeCamp，约 5 小时）
- 中文主课：[韦东山：嵌入式 Linux 驱动开发基础](https://www.bilibili.com/video/BV14f4y1Q7ti/)（50 讲、约 17 小时）
- 文档：[LFD103 Kernel Development](https://training.linuxfoundation.org/training/a-beginners-guide-to-linux-kernel-development-lfd103/)、[Bootlin Training Materials](https://bootlin.com/docs/)、[Bootlin Kernel Training](https://bootlin.com/training/kernel/)、[Kernel Driver API](https://www.kernel.org/doc/html/latest/driver-api/index.html)、[Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)、[Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)、[TinyLab 内核文档中文版](https://tinylab-1.gitbook.io/linux-doc/zh-cn)

#### 分支课程大纲（★ = 本支线必修）

- **Linux Device Drivers Development（freeCodeCamp，约 5 小时）** 大纲：
  - ★ 内核模块 module 编写与加载
  - ★ syscall 与 /proc 接口
  - ★ 字符设备、ioctl
  - ★ GPIO、中断、工作队列（概念）
- **韦东山：嵌入式 Linux 驱动开发基础（B站 50 讲）** 大纲：
  - ★ 字符设备、ioctl、poll、mmap
  - ★ GPIO、中断、设备树匹配
  - ★ 不要在 Orin 启动链上直接试验

#### 学习方法、项目与安全边界

freeCodeCamp 5 小时快速建立 module、syscall、`/proc`；韦东山中文课补 GPIO、字符设备、中断、工作队列和 `mmap`。看完即做：在 VM、QEMU 或可恢复开发板完成 hello module、字符设备、ioctl、poll 和 mmap。停止条件：能从 device tree 匹配到 driver/probe，能用 `dmesg`、ftrace/perf 定位一次问题。不要先改 Orin 启动链或 NVIDIA GPU 驱动。项目：编译并加载 hello module；读懂一个字符设备驱动；能解释设备树、module、用户态/内核态。

### 分支 E：TVM 和 MLIR

**进入条件**：完成 Triton Language 或同等 Kernel 项目后再学；它不是 TensorRT 前置课。

#### 课程与资料

- 主课：[MLC 机器学习编译](https://www.bilibili.com/video/BV15v4y1g7EU)（陈天奇课程，10 讲完整系列）；[中英文课程页](https://mlc.ai/summer22-zh/schedule)
- 专题：[LLVM MLIR Tutorial](https://www.youtube.com/watch?v=Y4SvqTtOIDk)（EN / LLVM Official，完整 workshop）
- 文档：[Apache TVM Documentation](https://tvm.apache.org/docs/)、[TVM End-to-End Optimization Tutorial](https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html)、[MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)

#### 分支课程大纲（★ = 本支线必修）

- **MLC 机器学习编译（陈天奇课程，B站 10 讲）** 大纲：
  - ★ TVM/MLC 概览与安装
  - ★ TensorIR schedule 与自动优化
  - ★ 编译到多后端（含 Orin/边缘）
  - ★ 配套 [中英文课程页](https://mlc.ai/summer22-zh/schedule)
- **LLVM MLIR Tutorial（官方 workshop）** 大纲：
  - ★ MLIR Toy dialect：AST → dialect → passes → lowering → LLVM
  - ★ dialect / pass / lowering 概念

#### 学习方法与停止条件

完整学 MLC 10 讲及配套课程页，再看 LLVM MLIR workshop。看完即做：完成 TensorIR schedule/自动优化 notebook；再走一遍 MLIR Toy 的 AST → dialect → passes → lowering → LLVM。停止条件：能解释 graph optimization、tensor program、IR、dialect、pass、lowering、runtime 的层级关系，并能把一个优化落到可运行代码。不要在 CUDA 基础没掌握时从 MLIR 开始。

---

## 12. 六周 CUDA 冲刺

起点视频：[NVIDIA Modern CUDA C++ Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)（主课）+ [freeCodeCamp CUDA Course](https://www.youtube.com/watch?v=86FAWCzIe_4)（补充），代码仓库 [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)。

**两个视频足够启动，不够形成完整能力**；缺口（官方编程模型、Best Practices、Sanitizer/Nsight、架构分析、TensorRT、Jetson 项目）由下面六周补齐。每周节奏：**视频章节 → 当天复现 → 改一处 → 查文档 → 写一页结果**。

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

## 13. 真实公开项目阶梯

### 项目 A：CUDA 基础 Kernel 仓库（阶段二主线）

项目来源：[NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)、[NVIDIA CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)。

你要自己维护一个小仓库，至少包含：vector add / SAXPY、transpose、reduction、tiled GEMM、stream overlap、sanitizer 和 benchmark 脚本。验收：每个例子有 CPU reference、正确性测试、数据规模和性能记录。

### 项目 B：MNIST CUDA 逐级优化（阶段二主线）

项目来源：[Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)、[Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)。

验收：完成 PyTorch → C/CUDA → cuBLAS → streams/fusion → FP16 或 TF32 的至少四个版本，并解释每次变化。

### 项目 C：TensorRT 推理部署（阶段五主线）

项目来源：[NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)、[TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)、[TensorRT Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)。

主线归属：阶段五；本项目负责执行阶段四设计的量化、剪枝与稀疏化硬件验证方案。

任务：1. 训练或选择一个小模型；2. 导出 ONNX 并验证输出；3. 用 trtexec 或 Python/C++ API 构建 Engine；4. 比较 FP32、FP16、INT8；5. 加入 dynamic shape；6. 对不支持的算子尝试 plugin 或改图。

验收指标：PyTorch/ONNX/TensorRT 数值对齐与 accuracy delta、真实 TensorRT Engine 构建日志和实际 tactics、engine build 是否可复现、warmup 后 p50/p95 latency、throughput、峰值 unified memory、输入 shape 和 batch 的影响，并完整比较 FP32、FP16、INT8。若测试 2:4 structured sparsity，必须沿用分支 B 的证据标准：用 verbose build log 区分具备 sparse tactic 资格（eligible）的 layer 与 builder 最终实际选中的 sparse tactic（selected）；任意剪枝、参数量或 FLOPs 降低都不能单独证明加速。

### 项目 D：Jetson 多路视频推理（分支 A 可选视觉项目，非主线）

项目来源：[NVIDIA DeepStream](https://github.com/NVIDIA/DeepStream)、[DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)、[DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)。

项目归属：分支 A 的可选视觉项目，不属于主线；完成阶段五后按方向进入。

任务：先跑单路文件或 USB 摄像头；再扩展为 4 路文件/RTSP 输入；加入 decode、nvinfer、tracker、tiler、OSD 和 metadata；对比 OpenCV/Python 串行实现与 DeepStream pipeline；用 tegrastats 和 Nsight Systems 观察 CPU、GPU、内存、解码和推理。

验收：在固定功耗和输入条件下，报告每路 FPS、端到端延迟、GPU 利用率、内存、温度和降频情况。注意：旧的 [deepstream_reference_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps) 仓库已经停止更新；当前新项目优先从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 主仓库和对应 release 开始。

### 项目 E：Triton 模型服务（阶段六主线）

项目来源：[Triton Inference Server](https://github.com/triton-inference-server/server)、[Triton Tutorials](https://github.com/triton-inference-server/tutorials)、[Triton Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)、[Backend Platform Support Matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md)。

任务：建立 model repository；用 TensorRT backend 提供一个图像分类或检测模型；配置 dynamic batching 和多个 model instance；使用官方 Triton HTTP client 和 unary gRPC client；加入 health、metrics、错误处理和超时；用 Perf Analyzer 测量 concurrency、latency 和 throughput；再做一个 ensemble，把预处理、推理、后处理串起来。不要求自建 gRPC server，也不要求实现 streaming RPC。

验收：画出并发数、batch、p50/p95 延迟和吞吐之间的曲线，并说明哪一个配置适合 Orin 的内存和功耗约束。

### 项目 F：Orin 上的 Edge LLM（阶段八主线）

项目来源：[TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)、[TensorRT-Edge-LLM Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)、[Jetson AI Lab Edge-LLM Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)。

主线归属：阶段八；阶段七的 Transformer 与 LLM 概念是本项目的前置必修。

建议 capstone：选择支持矩阵中的 Qwen3 小模型，优先从 0.6B、1.7B 或 4B 级别开始；先做 FP16 可运行基线，再尝试 INT4/AWQ；用一个简单 HTTP API 包装推理；测量 TTFT、TPOT、tokens/s、峰值内存、上下文长度和连续请求行为。

不要一开始做：14B 以上模型的盲目移植；只看 tokens/s、不测首 token 延迟；把 server GPU 上的 vLLM/TRT-LLM 命令直接复制到 Jetson。

---

## 14. 概念澄清：TensorRT / Triton / vLLM / TensorRT-LLM

### 14.1 TensorRT 是运行时和优化器

典型流程：PyTorch / TensorFlow → ONNX → TensorRT Builder → hardware-specific Engine → TensorRT Runtime。

TensorRT 解决的是模型图优化、Kernel 选择、精度、Engine 构建和执行。它本身不是完整的多租户 HTTP 服务。

### 14.2 Triton 是通用推理服务层

Triton 提供：model repository、多种 backend、HTTP/gRPC、dynamic batching、concurrent model execution、ensemble、health 和 metrics。

本节的 Triton 指 Triton Inference Server，不是用于编写 GPU Kernel 的 Triton Language；后者不在本节展开。

因此，**LLM serving 可以使用 Triton，但 LLM serving 不等于 Triton**。Triton 是一个通用 serving layer；模型运行时可能是 TensorRT、PyTorch、ONNX Runtime、Python backend 或其他 backend。

### 14.3 vLLM、TensorRT-LLM、Edge-LLM 的定位

- **vLLM**：偏通用服务器 GPU 的 LLM runtime/serving，重点是 KV cache、continuous batching 和高吞吐。
- **TensorRT-LLM**：偏 NVIDIA 服务器 GPU 的高性能 LLM runtime，通常需要匹配的 GPU、CUDA、容器和版本。
- **TensorRT-Edge-LLM**：面向 Jetson/边缘设备的路线；对当前 Orin NX 更有现实意义。
- **Triton**：可以承载多种模型和 backend，是服务编排与推理 API 层；是否适合某个 Jetson backend 必须查平台矩阵。

推荐顺序：1. TensorRT 单模型运行；2. Docker 化并使用 Triton + TensorRT backend；3. 补齐 Transformer/LLM 概念；4. 理解 vLLM/TensorRT-LLM 等服务器运行时；5. 在 Orin 上使用 TensorRT-Edge-LLM 支持矩阵内路径。GStreamer/DeepStream 属于阶段五后的视觉支线，不是 Triton 或 LLM 的前置。

Jetson 平台特别注意：Triton 的 backend/platform 组合不是全部可用，尤其不要默认 Python backend 的 GPU 能力与 x86 服务器一致；以 [官方 backend platform matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md) 为准。

---

## 15. 24 周执行节奏

假设每周投入 10-12 小时；每周只有 5-6 小时时，把日历时间大约翻倍。

如果已经能独立完成 C++17/CMake 小程序，能创建可复现的 Python `venv`、使用 NumPy 和 PyTorch 处理 tensor，并熟悉 Shell、进程、权限、SSH 和 `gdb`，才可以跳过第 1-2 周；否则先完成阶段一的 Python + PyTorch `model-tools` 和 C++ 项目。遇到基础问题时再回看阶段一对应章节。

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

## 16. 每个阶段统一的验收标准

不要只以视频看完作为完成条件。每个项目都必须回答以下问题：

**正确性**：是否有 CPU 或框架 reference？随机输入、边界输入和不同 batch 是否通过？Compute Sanitizer 是否干净？模型导出前后精度差异是多少？

**性能**：测量的是 kernel、runtime 还是端到端？是否排除了首次加载和 warmup？是否报告 p50、p95，而不是只报平均值？优化前后是否有 Nsight 或 benchmark 证据？

**Edge 条件**：功耗模式是什么？温度和是否降频？unified memory 峰值是多少？冷启动和热运行结果是否不同？输入分辨率、batch、并发是否固定？

**Serving**：健康检查、超时、错误和重启如何处理？并发增加时延迟如何变化？dynamic batching 是否真的提高吞吐？是否有 Prometheus 或等价指标？

**LLM**：使用的模型和量化格式是否在目标平台支持矩阵内？是否分别测量 TTFT、TPOT、tokens/s？峰值内存和最大上下文是多少？多请求时是否出现 OOM、抖动或严重排队？

---

## 17. Orin NX 上的版本和操作纪律

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

## 18. 中文资料的正确用法

- 中文视频用于建立概念、跟练界面、降低入门成本；版本与命令一律以英文官方文档为准。
- NVIDIA / 原作者 / 大学官方发布的中文课可作为主课；第三方翻译镜像只作语言辅助，代码必须回到官方仓库核对。
- 各阶段的中文课程与查询入口已在对应阶段卡列出，此处不再重复。
---

## 19. 明确排除的内容

以下内容不放入免费主路线：

- NVIDIA DLI 中标价 30/90 美元的完整实践课程。
- 付费 Bootlin 讲师课程；只使用 Bootlin 公开 slides、labs 和源码。
- 依赖付费云 GPU 才能完成的实验。
- 仅针对 H100/Hopper 的高级课程作为 CUDA 入门。
- 旧版 DeepStream/Triton 教程中的固定安装命令。
- 未说明来源、版本和许可证的课程搬运或代码集合。

例如，H100 专用课程 [H100-Course](https://github.com/cudacourseh100/H100-Course) 可以在以后有 Hopper 机器时选修，但不适合作为 Orin SM87 的当前主线。

---

## 20. 最终毕业项目

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

## 21. 一页式执行清单

### 现在

- [ ] 安装并确认 Python `venv`、NumPy、PyTorch/ONNX 工具链，完成阶段一的 `model-tools`。
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

## 22. 学完之后：能力定位与岗位

完成全部阶段项目与毕业项目后，定位是**具备端到端 GPU 推理与边缘 AI 系统能力的初级到初中级工程师**。
---

## 23. 关键链接索引

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
