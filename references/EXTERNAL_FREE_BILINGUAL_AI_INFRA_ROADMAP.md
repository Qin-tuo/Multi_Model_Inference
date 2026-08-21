# 外部免费双语 AI Infra / CUDA / Jetson 学习路线

> 路线定位：这是本知识库的 **NVIDIA 生态主路线**。完成后再进入 [多生态总路线](EDGE_AI_MULTI_ECOSYSTEM_ROADMAP.md) 中的 [地瓜 BPU 路线](EXTERNAL_FREE_BILINGUAL_HORIZON_BPU_ROADMAP.md)，再学习 [瑞芯微双并行路线](EXTERNAL_FREE_BILINGUAL_ROCKCHIP_RKNN_ROADMAP.md)。
>
> 面向：Jetson Orin NX 16GB、CUDA/C++、TensorRT、DeepStream、Triton、边缘 LLM Serving
> 目标：从底层计算模型一路走到可测量、可部署、可服务化的 AI 推理系统
> 资料范围：真实存在的公开课程、官方文档、公开 GitHub 项目；不把当前文件夹中的内部笔记当作外部参考资料
> 版本核验日期：2026-08-21。版本敏感的命令和容器必须在执行前再次核对官方兼容矩阵。

## 0. 先给结论

你的主线不应该是把所有技术都学一遍，而应该是：

**Python + PyTorch + C/C++ + Linux 基础 → CUDA C++ / Kernel → 正确性与 Nsight 性能分析 → ONNX / TensorRT → Jetson / DeepStream → Triton / 服务化 → TensorRT-Edge-LLM / LLM Serving**

同时保留两条分支：

- **Linux 驱动 / BSP 分支**：适合想做 Jetson 底层、内核、设备树、驱动、系统启动和平台适配的人。
- **GPU 编译器 / Kernel DSL 分支**：CUDA 熟练后学习 Triton、CUTLASS/CuTe、TVM、MLIR，适合做算子、编译器和性能工程。

你现在先学 CUDA 是正确的起点。Linux 驱动很重要，但对于 AI 推理性能和部署这个目标，它不是第一门课；先具备读懂硬件行为、写 Kernel、做 Profile 和解释性能数据的能力，后续再进入驱动会更有效。

## 1. 对目标和硬件的理解

### 1.1 你的目标画像

这份路线按以下目标定制：

- 不是泛泛学习机器学习，而是进入 GPU 加速、AI 推理、边缘部署、Serving 和系统性能。
- 需要能自己写 CUDA Kernel，并能用工具解释为什么快或慢。
- 需要从模型导出一直做到 TensorRT Engine、视频流水线、服务 API 和监控。
- 需要中英文资料：英文资料作为权威主线，中文资料用于降低入门门槛和复习。
- 不购买课程；核心学习不依赖付费课程或付费云 GPU。
- 以本机 Orin NX 为真实部署目标，必要时把不适合 ARM64/Jetson 的实验放到 x86 GPU 或 Colab 上完成。

### 1.2 本机基线

根据本机环境检查结果：

- Jetson Orin NX 16GB 开发套件配置
- Ubuntu 24.04.4 LTS
- Linux kernel 6.8.12-tegra
- Jetson Linux / L4T R39.2.1
- JetPack 7.2.1
- CUDA 13.2.1、TensorRT 10.16.2、DeepStream 9.1（以本机实际安装包为准）
- GPU Compute Capability 8.7，属于 Orin 的 Ampere 架构
- 8 核 Cortex-A78AE
- 16GB unified memory，CPU 和 GPU 共享内存
- 1TB Samsung 990 PRO NVMe
- 当前功耗模式 25W

硬件与版本总入口：

- [JetPack SDK Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Documentation](https://docs.nvidia.com/jetson/index.html)
- [CUDA GPUs and Compute Capability](https://developer.nvidia.com/cuda/gpus)
- [Jetson AI Lab Tutorials](https://www.jetson-ai-lab.com/tutorials/)

### 1.3 由硬件带来的学习边界

- Orin 是 SM 8.7；先围绕 Ampere 的内存访问、共享内存、Warp、Tensor Core、FP16/INT8 学习。
- Orin 是 ARM64 和 unified memory 平台；很多 x86_64 容器、Python wheel、Triton 示例不能直接照搬。
- 16GB 不是独立显存；模型、解码缓冲区、TensorRT Workspace、系统和服务会共同占用内存。
- Jetson 设备上的实际性能受功耗、温度、时钟和内存带宽影响，不能只看桌面 GPU 的教程结果。
- 最新的通用文档可能已经超过本机 JetPack 版本；安装和调试时优先使用 JetPack 自带的 CUDA、TensorRT、DeepStream 和对应版本文档。

## 2. 资料使用规则

### 2.1 标签

- **EN / Official**：英文官方资料，版本和 API 以此为准。
- **EN / Open**：英文公开课程或社区项目，适合跟练。
- **中文 / Official**：NVIDIA 中文站、中文文档或官方中文课程入口。
- **中文 / Supplement**：中文翻译、字幕或辅助讲解；代码和版本必须回到官方仓库核对。
- **Project**：必须实际运行、修改和测量的公开项目。

### 2.2 不付费的边界

主路线只使用公开免费内容。NVIDIA DLI 的课程目录中同时存在免费短课和付费课，不能把能看到课程页面误认为课程实验永久免费。云端 GPU 额度也可能变化，因此不把付费云作为前置条件。

中文资料有时是旧版本或第三方转载。遇到版本冲突时按以下优先级：

1. 本机 JetPack / L4T 版本对应的 NVIDIA 官方文档。
2. NVIDIA 官方 GitHub 仓库的 release、branch 和 README。
3. 官方、大学或原作者发布的完整课程视频。
4. 中文完整课程、翻译或字幕。

建议学习比例：**30% 看完整课程，55% 写代码和改项目，15% 查文档并写实验记录**。

### 2.3 视频优先，不从文档第一页开始

每一个节点都按同一套方式执行：

1. 从本路线给出的 1-2 门完整课程中选择 **一门主课**。
2. 每看完一个可编码章节，当天复现代码；不要等整门课看完才动手。
3. 完成该节点唯一指定的阶段项目，并达到停止条件。
4. 只有遇到 API、版本、安装或指标定义问题时，才定点查官方文档。

第二门课是补课，不是重复刷课。主课听懂且项目通过，就直接进入下一节点；只有主课缺少某个主题，才去第二门课找对应章节。

这里所说的“完整课程”指有连续教学结构、明确起点和终点的课程、系列课或完整 workshop。宣传片、零散大会演讲和只演示一次命令的视频，不计为主课。中文镜像若只是同一门英文课的翻译，会明确标成“同课中文镜像”，不会伪装成第二门独立课程。

官方文档在这份路线中只承担三种职责：

- 查本机版本对应的安装和兼容矩阵。
- 把旧视频中的 API 映射到当前 API。
- 核对 profiler 指标、配置字段和边界条件。

不要求通读 CUDA、TensorRT、DeepStream 或 Triton 的整本手册。

## 3. 视频课程驱动的总路线

你当前可以直接从 **节点 2：CUDA C++** 开始，但节点 0 的 Python、PyTorch 与 C++ 基础是必修底座。节点 1 的 Linux 用户态可以按经验补课；当你在指针、引用、编译链接、虚拟环境、NumPy、tensor、Shell、进程或权限上卡住时，先回补对应章节。

主线依赖关系：

**Python / PyTorch / C++ / Linux → CUDA 编程 → GPU 架构与 Kernel 优化 → Sanitizer / Nsight / PTX-SASS → ONNX / TensorRT → Jetson / DeepStream → Docker / gRPC → Triton Server → LLM Inference → Orin Edge LLM**

高级分支：

**Linux 驱动**，或 **Triton Language → CUTLASS / CuTe → TVM / MLIR**

### 3.1 每个关键节点的 1-2 门完整视频课

表中的“课程 1”默认是主课。“课程 2”用于补缺，不要求两门从头到尾重复观看。

| 节点 | 属性 | 完整视频课程 1 | 第二门完整课 / 专题补课 | 唯一阶段项目 |
|---|---|---|---|---|
| 0. C++、Python 与 PyTorch 基础 | 基础必修 | C++：[C++ Programming Course: Beginner to Advanced](https://www.youtube.com/watch?v=8jLOx1hD3_o)；PyTorch：[PyTorch for Deep Learning & Machine Learning - Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA)（EN / freeCodeCamp） | C++：[黑马程序员 C++ 零基础到项目开发](https://www.bilibili.com/video/BV1ZH4y137ws/)；Python：[Python Full Course for free](https://www.youtube.com/watch?v=ix9cRaBkVe0) 与[黑马程序员全套 Python 基础教程](https://www.bilibili.com/video/BV1o4411M71o/)；PyTorch：[PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/)（中文，已完结系列） | C++17 + CMake CPU GEMM；Python + PyTorch 模型导出、校验和 benchmark 工具链 |
| 1. Linux 用户态 | 必修 | [Introduction to Linux: Full Course](https://www.youtube.com/watch?v=sWbUDq4S6Y8)（EN，freeCodeCamp，约 6 小时） | [尚硅谷 Linux 应用层开发](https://www.bilibili.com/video/BV1DJ4m1M77z/)（中文，文件 IO、进程线程、Socket、epoll 完整系列） | Shell 监控脚本 + C/C++ TCP 服务 |
| 2. CUDA C++ 编程 | 必修，当前起点 | [NVIDIA Fundamentals of Accelerated Computing with Modern CUDA C++](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)（EN / Official，完整系列） | [CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)（EN，freeCodeCamp，约 12 小时） | vector add、SAXPY、2D add 和 CPU reference |
| 3. GPU 架构与 Kernel 优化 | 必修 | [Stanford CS149 Parallel Computing 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)（EN / University，完整课程） | [大规模并行处理器编程实战](https://www.bilibili.com/video/BV1gz421o7uH/)（中文，PMPP 7 讲完整翻译课） | transpose、reduction、tiled GEMM、cuBLAS 对比 |
| 4. CUDA 正确性与性能工具 | 必修 | [NVIDIA CUDA Developer Tools Tutorials](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR)（EN / Official，7 讲完整系列） | [GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)（EN / Open，高级完整系列） | Sanitizer + `nsys` + `ncu` + PTX/SASS 优化报告 |
| 5. ONNX 与 TensorRT | 必修 | [NVIDIA TensorRT 教程 4 部分](https://www.bilibili.com/video/BV15Y4y1W73E)（中文 / NVIDIA Official，完整系列） | [Inference Optimization with NVIDIA TensorRT](https://www.youtube.com/watch?v=UnIuMXGylfY)（EN / NCSA，完整 workshop） | PyTorch → ONNX → TensorRT，FP32/FP16/INT8 对比 |
| 6A. Jetson AI 基础 | 必修 | [NVIDIA Jetson AI Fundamentals](https://www.youtube.com/playlist?list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8)（EN / Official，完整系列） | [NVIDIA Jetson 边缘 AI 快速上手系列](https://www.bilibili.com/video/BV1yEzBYQEMt/)（中文 / Seeed Studio，完整系列） | `jetson-inference` 摄像头分类、检测和分割 |
| 6B. GStreamer 与 DeepStream | 视觉方向必修 | [Create Vision AI Applications With DeepStream](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81879/?playlistId=gtc26-computer-vision-and-video-analytics)（EN / NVIDIA，2026，2 小时完整 workshop） | [深度学习模型部署与剪枝优化实战](https://www.bilibili.com/video/BV1Sw411y7Hs/)（中文，完整部署课；学习其中 GStreamer / DeepStream 单元） | 单路到 4 路 decode / infer / tracker / metadata pipeline |
| 7A. Docker 容器 | Serving 必修 | [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)（EN，freeCodeCamp，约 2 小时 10 分） | [尚硅谷 Docker 与微服务实战 2024](https://www.bilibili.com/video/BV1Zn4y1X7AZ/)（中文，完整系列） | 为 TensorRT 推理程序制作 ARM64 镜像和 Compose 服务 |
| 7B. HTTP / gRPC | Serving 必修 | [Getting Started With gRPC: Hands-On Codelab](https://www.youtube.com/watch?v=kAuK6VcAR10)（EN / CNCF，约 75 分钟完整 workshop） | [手把手 gRPC 基础教程](https://www.bilibili.com/video/BV1QT411H7ds/)（中文，14 讲完整系列） | `.proto` + unary/stream client/server + health check |
| 8. Triton Inference Server | 通用 Serving 必修 | [NVIDIA Triton 从入门到精通](https://www.bilibili.com/video/BV1KS4y1v7zd/)（中文 / NVIDIA Official，20 讲完整系列） | [Getting Started with NVIDIA Triton](https://www.youtube.com/watch?v=NQDtfSi5QF4)（EN / NVIDIA，完整入门专题） | TensorRT backend、dynamic batching、ensemble、Perf Analyzer |
| 9. LLM Inference / Serving 原理 | LLM 必修 | [Stanford CS336 2026 Video Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)（EN / University，完整课程；重点第 6、10 讲） | [大模型推理技术研究](https://www.bilibili.com/video/BV1k2L9zyEt7/)（中文，KV cache、vLLM、SGLang 等 9 讲完整系列） | 测 TTFT、TPOT、tokens/s、并发和 KV cache 占用 |
| 10. Orin Edge LLM | 本机目标必修 | [Make It Think: NVIDIA Jetson AI Lab](https://youtube.com/playlist?list=PLZrTAEPLeXfo)（EN / NVIDIA，2026，3 场完整系列） | [Getting Started with Edge AI on NVIDIA Jetson](https://www.youtube.com/watch?v=t2Ecuu2FdC8)（EN / NVIDIA，完整直播课） | TensorRT-Edge-LLM 支持矩阵内的小模型 FP16 / INT4 服务 |
| A. Linux Kernel / Driver | 可选分支 | [Linux Device Drivers Development](https://www.youtube.com/watch?v=iSiyDHobXHA)（EN，freeCodeCamp，约 5 小时） | [韦东山：嵌入式 Linux 驱动开发基础](https://www.bilibili.com/video/BV14f4y1Q7ti/)（中文，50 讲、约 17 小时） | module + 字符设备 + ioctl/poll/mmap |
| B1. Triton Kernel Language | 可选分支 | [Triton 从入门到大师](https://www.bilibili.com/video/BV1fMyWBgERM/)（中文，10 讲完整课） | [Stanford CS336 2026 Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)（EN / University，完整专题课） | Triton vector add、softmax、matmul 与 CUDA 对比 |
| B2. CUTLASS / CuTe | B1 之后选修 | [GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)（EN，完整高级系列；重点 CUTLASS/CuTe 讲次） | 暂无同等质量、从零到完整且紧跟当前 CuTe 的免费官方视频课 | 用 CUTLASS profiler 和一个 CuTe GEMM 理解 layout / tile / MMA |
| C. TVM / MLIR 编译器 | 最后选修 | [MLC 机器学习编译](https://www.bilibili.com/video/BV15v4y1g7EU)（中文，陈天奇课程，10 讲完整系列） | [LLVM MLIR Tutorial](https://www.youtube.com/watch?v=Y4SvqTtOIDk)（EN / LLVM Official，完整 workshop） | TensorIR schedule + MLIR Toy dialect / lowering |

中文入口补充：NVIDIA Modern CUDA C++ 的[同课中文镜像](https://www.bilibili.com/video/BV1QvSKB4EMr/)，CUDA Developer Tools 的[同课中文镜像](https://www.bilibili.com/video/BV14RU6BmE5u/)，Jetson AI Fundamentals 的[同课中文镜像](https://www.bilibili.com/video/BV1EGSmBWErR/)。这些镜像用于中文字幕，不重复计算为第二门独立课程。

### 3.2 分节点观看方法与停止条件

#### 节点 0：C++、Python 与 PyTorch 基础，Python/PyTorch 必修

Python 在这条路线中不是“会一点脚本”即可，而是模型导出、量化校准、数据集处理、benchmark、日志分析、ONNX 检查和服务客户端的主要胶水语言。PyTorch 则是连接训练侧模型资产与部署侧 ONNX/TensorRT 的必修桥梁。节点 0 必须同时完成 C++、Python 和 PyTorch 三条基础线。

##### Python + PyTorch 必修范围

按下面顺序学习，不需要把 Python Web、爬虫或数据分析全套学完：

1. 语法、条件/循环、字符串、列表/字典/集合/元组、函数和作用域；
2. 异常、文件、路径、JSON/CSV、模块、包、类和 dataclass；
3. 虚拟环境、pip、requirements、版本锁定、命令行入口和环境变量；
4. pathlib、argparse、logging、subprocess、glob、hashlib 和 time；
5. NumPy 的 dtype、shape、stride、broadcast、向量化、矩阵乘法和序列化；
6. OpenCV/PIL 的图像读写、颜色格式、resize、批量遍历和结果保存；
7. PyTorch tensor、Dataset/DataLoader、eval/no_grad、checkpoint 和 ONNX 导出；
8. pytest、断言、随机种子、类型标注和可复现 benchmark 输出。

节点 0 不要求完整训练大型模型；明确不把以下内容作为前置：Django/Flask 全栈、爬虫、复杂异步 Web、数据分析可视化、分布式训练、DeepSpeed 和 Python 底层解释器。

##### Python 课程与官方资料

主课二选一，不能只看不写：

- 英文：[Python Full Course for free](https://www.youtube.com/watch?v=ix9cRaBkVe0)，freeCodeCamp，完整零基础课程；
- 中文：[黑马程序员全套 Python 基础教程](https://www.bilibili.com/video/BV1o4411M71o/)，覆盖环境、语法、函数、文件、面向对象、异常、模块和项目。

课程遇到版本或 API 问题时，以这些资料为准：

- [Python 官方教程](https://docs.python.org/3/tutorial/)；
- [venv 官方文档](https://docs.python.org/3/library/venv.html)；
- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)；
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)；
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)；
- [PyTorch ONNX Tutorials](https://docs.pytorch.org/tutorials/beginner/onnx/index.html)。

##### PyTorch 完整课程与部署向停止条件

PyTorch 在这里不要求先学成训练算法工程师，先掌握模型加载、推理、导出和验证。按下面顺序学习：

- 英文主课：[PyTorch for Deep Learning & Machine Learning - Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA)（freeCodeCamp）；
- 中文补课：[PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/)（小土堆，已完结系列，配套代码见[GitHub](https://github.com/xiaotudui/pytorch-tutorial)）；
- 官方视频校正：[Introduction to PyTorch on YouTube](https://docs.pytorch.org/tutorials/beginner/introyt/)。

必修关键字：tensor 的 shape/dtype/device/layout/contiguous、索引与 reshape/permute、`nn.Module`/`forward`、`eval()`、`no_grad()`/`inference_mode()`、`state_dict`、Dataset/DataLoader、checkpoint 加载、CPU/CUDA 转移、warmup/同步计时、ONNX 导出、dynamic shape 和数值对齐。

PyTorch 停止条件：

- 能加载一个公开 checkpoint，在固定输入上稳定得到结果；
- 能解释一次 shape、dtype、device 或 layout 错误；
- 能用 `eval()` 和 `inference_mode()` 完成推理，并正确处理 warmup 与 CUDA 同步；
- 能把 PyTorch 模型导出 ONNX，并与 PyTorch 输出做容差比较；
- 能说明训练、模型导出、TensorRT 构建和板端 Runtime 各自处于哪一层。

##### Python + PyTorch 必修项目：model-tools

在节点 0 建立一个独立的 <code>model-tools</code> 目录，至少包含：

- <code>venv</code> 创建、依赖文件和一键运行说明；
- 用 <code>argparse</code> 扫描图片/视频数据集，生成带 hash 的 manifest；
- 用 NumPy 实现 CPU GEMM，并与 PyTorch 输出和耗时对齐；
- 读取图像并统一完成 RGB/BGR、resize、dtype 和 batch 处理；
- 导出一个简单 PyTorch 模型到 ONNX，并用 ONNX Runtime 做结果校验；
- 用 <code>subprocess</code> 调用外部 benchmark，输出 JSON/CSV；
- 用 logging 记录版本、输入 shape、随机种子和失败样本；
- 至少 5 个 pytest，覆盖空目录、错误路径、shape 和数值容差。

Python + PyTorch 基础停止条件：

- 能从零创建隔离环境，并在另一台机器按文档复现；
- 能读写常见模型/图像/JSON 文件，不依赖手工改路径；
- 能解释 NumPy 的 shape、dtype、stride 与 PyTorch tensor 的关系；
- 能把一次模型导出、校验和 benchmark 写成命令行工具；
- 能定位一次依赖版本、路径、dtype 或 shape 错误。

达到以上条件后，Python + PyTorch 基础节点完成；后续只在实际模型、CUDA、TensorRT 和 Serving 项目中继续深化。

##### C++ 分支

- **怎么选课**：零基础选中文黑马；已有 Python/Java 基础且能听英文，选 freeCodeCamp。不要两门都刷。
- **只抓这些关键字**：pointer、reference、stack/heap、RAII、class、template、STL、lambda、CMake、gdb。
- **看完即做**：用 C++17 写 CPU 矩阵乘法，CMake 构建，随机输入校验，输出耗时。
- **停止条件**：能解释值/引用/指针、对象生命周期、编译与链接，并能定位一次段错误。达到后立刻进入 CUDA。

#### 节点 1：Linux 用户态，目标是能开发和排障

- **怎么选课**：先用 freeCodeCamp 6 小时建立全貌；中文课只补文件 IO、进程线程、Socket 和 epoll，不学成运维认证路线。
- **看完即做**：写一个采集 CPU、内存、温度、进程状态的 Shell 脚本，再写一个可并发处理请求的 C/C++ TCP 小服务。
- **停止条件**：能使用权限、管道、重定向、进程信号、日志、SSH、`gdb`、`strace`；知道 syscall、用户态和内核态的边界。
- **版本注意**：CentOS 视频中的 `yum`、旧网络服务命令不复制到本机 Ubuntu 24.04；只迁移稳定概念。

#### 节点 2：CUDA C++，你的当前起点

- **主课**：先完整跟 NVIDIA Modern CUDA C++，对应实验在 [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)。
- **补课**：freeCodeCamp 12 小时课程只补 C/C++ review、kernel、streams、PyTorch extension 和课程项目；代码在 [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)。
- **看完即做**：vector add、SAXPY、二维矩阵加法；每个都有 CPU reference、边界输入、CUDA error check 和事件计时。
- **停止条件**：能不看答案写出索引，解释 grid/block/thread、异步 launch、global/shared/register/unified memory。

#### 节点 3：GPU 架构和 Kernel 优化

- **怎么选课**：先看中文 PMPP 7 讲并同步敲代码；再从 CS149 补并行硬件、局部性、调度、同步和性能模型。CS149 不要求一次看完所有非 GPU 章节。
- **看完即做**：naive/tiled transpose、naive/tree reduction、naive/tiled GEMM，并与 cuBLAS 比较。
- **停止条件**：能用 coalescing、bank conflict、occupancy、divergence、arithmetic intensity 解释性能，而不是只说“GPU 更快”。
- **Orin 边界**：围绕 Ampere `sm_87` 实验；Hopper/Blackwell 的 TMA、WGMMA 示例只看概念，不作为本机验收。

#### 节点 4：Compute Sanitizer、Nsight、PTX 和 SASS

- **主课顺序**：完整看完 NVIDIA 7 讲工具课，然后立刻对节点 3 的最慢 Kernel 操作一次。GPU MODE 留到需要读更深的 SASS、Tensor Core 或算子案例时再看。
- **固定工具顺序**：`compute-sanitizer` → `nsys` → `ncu` → `cuobjdump` / `nvdisasm`。
- **看完即做**：故意制造越界或 race 并修复；导出系统 timeline；对唯一热点 Kernel 采集 memory、SOL、occupancy、warp stall；反汇编前后两个版本。
- **停止条件**：交付一份“证据 → 假设 → 修改 → 复测”的性能报告，且 Sanitizer 干净。

#### 节点 5：ONNX 和 TensorRT

- **怎么选课**：先看 NVIDIA 中文 4 部分建立 builder/runtime/parser/precision 全貌；再用 NCSA workshop 跟一遍完整模型转换。
- **看完即做**：PyTorch 导出 ONNX、ONNX Runtime 对齐、`trtexec` 构建 Engine、FP32/FP16/INT8、dynamic shape、warmup 后 benchmark。
- **停止条件**：能解释 Engine 为什么与目标 GPU/版本绑定，能报告 accuracy delta、p50/p95、throughput 和峰值内存。
- **强版本警告**：公开视频主要基于 TensorRT 8.x，本机是 TensorRT 10.16.2。视频只学概念；实现必须以 [TensorRT 10.x Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/quick-start-guide.html) 和当前 samples 为准，不照抄旧 bindings 或旧 plugin API。

#### 节点 6A：Jetson AI 基础

- **怎么选课**：官方 Jetson AI Fundamentals 作为主课；中文 Seeed 课程用于熟悉 Jetson、摄像头和边缘 AI 工作流。
- **看完即做**：从 [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference) 依次跑 classification、detection、segmentation，再接 USB/CSI 摄像头。
- **停止条件**：能区分模型加载、预处理、TensorRT 推理、后处理、捕获和显示耗时，并记录 `tegrastats`。
- **版本注意**：课程中 Nano/JetPack 4 的安装命令全部忽略；本机只用 JetPack 7.2.1 对应路径。

#### 节点 6B：GStreamer 和 DeepStream

- **怎么选课**：2026 NVIDIA workshop 是当前主课；中文旧课只用于 GStreamer pipeline、RTP/RTSP、DeepStream plugin 的概念和代码结构。
- **免费边界**：NVIDIA On-Demand 视频可免费观看；页面若要求登录，只注册免费 NVIDIA 账号，不购买 DLI 实验。真正的验收在本机 Orin 上完成。
- **看完即做**：文件单路 → 摄像头单路 → 4 路文件/RTSP；加入 decode、mux、`nvinfer`、tracker、tiler、metadata 和 sink。
- **停止条件**：能画出 pipeline，解释 caps、buffer、metadata、batch 和 zero-copy，并报告每路 FPS、端到端延迟、温度、功耗和内存。
- **强版本警告**：本机 DeepStream 9.1；旧课的安装命令、插件字段和 Python binding 不能直接复制，代码从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 当前 release 开始。

#### 节点 7A 与 7B：容器和服务协议

- **Docker**：二选一完整课。完成 multi-stage Dockerfile、volume、network、healthcheck、Compose，并确认基础镜像支持 `linux/arm64`。
- **gRPC**：先 CNCF codelab，中文 14 讲补 unary、server/client streaming 和双向流。语言不同不影响理解 `.proto` 和 RPC 语义。
- **看完即做**：把 TensorRT demo 包成有 `/health` 的容器服务，并写一个 gRPC predict 接口。
- **停止条件**：能解释镜像与容器、host/container 文件和端口、HTTP 与 gRPC、deadline、错误码和流式请求。

#### 节点 8：Triton Inference Server

- **主课**：NVIDIA 中文 20 讲是完整主课；英文 Getting Started 用于建立最新版官方术语入口。
- **看完即做**：model repository、TensorRT backend、HTTP/gRPC client、dynamic batching、instance group、ensemble、metrics、Perf Analyzer。
- **停止条件**：画出 concurrency/batch 与 p50/p95/throughput 曲线，并能说明 Triton 解决的是服务层问题，不是替代 TensorRT Kernel/runtime。
- **版本注意**：中文课基于 2022 年版本，架构概念仍可学；容器 tag、backend 支持和配置字段查当前 [Triton tutorials repo](https://github.com/triton-inference-server/tutorials) 与 Jetson platform matrix。2026 官方页面也可能使用 Dynamo-Triton 名称。

#### 节点 9：LLM Inference 和 Serving 原理

- **怎么选课**：中文 9 讲先建立 KV cache、continuous batching、PagedAttention、量化、speculative decoding、RadixAttention 的连接；再看 [CS336 Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM)，需要写算子时补第 6 讲。作业和讲义从 [CS336 官方课程页](https://cs336.stanford.edu/)查询。
- **看完即做**：用同一小模型和固定 prompt，对 batch、并发、上下文长度、量化方式做可复现 benchmark。
- **停止条件**：能严格区分 TTFT、TPOT、端到端延迟、tokens/s、单请求延迟和系统吞吐，并解释 prefill/decode 的瓶颈差异。
- **框架关系**：vLLM、SGLang、TensorRT-LLM/Edge-LLM 是 LLM runtime/engine 路线；Triton 是可选的通用服务层，二者不是同一个层级。

#### 节点 10：Orin 上的 Edge LLM

- **视频主线**：完整看 NVIDIA 2026 三场 Jetson AI Lab 系列；第二场官方直播课用于补 LLM、VLM 和 Jetson 工作流。
- **看完即做**：从支持矩阵内的小模型开始，先 FP16 基线，再 INT4/AWQ；提供简单 API，测 TTFT、TPOT、tokens/s、峰值 unified memory 和上下文长度。
- **停止条件**：服务可重复启动，连续请求无 OOM，报告中包含功耗、温度、量化、上下文、并发和版本。
- **资料现实**：截至 2026-08-21，还没有一门公开、完整且与当前 TensorRT-Edge-LLM release 同步的官方专项视频课。因此这里用最新 Jetson 完整视频系列建立工作流，代码只跟 [TensorRT-Edge-LLM 官方安装与教程](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html)。不拿旧 TensorRT-LLM 服务器课程冒充 Orin 课程。

#### 分支 A：Linux 驱动

- **进入时间**：CUDA/Nsight 主线完成后；除非目标岗位就是 BSP/驱动，否则每周并行 2 小时即可。
- **怎么选课**：freeCodeCamp 5 小时快速建立 module、syscall、`/proc`；韦东山中文课补 GPIO、字符设备、中断、工作队列和 `mmap`。
- **看完即做**：在 VM、QEMU 或可恢复开发板完成 hello module、字符设备、ioctl、poll 和 mmap。
- **停止条件**：能从 device tree 匹配到 driver/probe，能用 `dmesg`、ftrace/perf 定位一次问题。不要先改 Orin 启动链或 NVIDIA GPU 驱动。

#### 分支 B：Triton Language、CUTLASS 和 CuTe

- **进入时间**：只有节点 3、4 已通过，才进入算子分支。这里的 Triton 是 Kernel DSL，不是 Triton Inference Server。
- **Triton 课后项目**：vector add → fused softmax → matmul → layer norm；与 CUDA 版本做 correctness、性能和可读性对比。课程代码在 [triton_docs_tutorials](https://github.com/evintunador/triton_docs_tutorials)。
- **CUTLASS/CuTe 现实**：当前完整免费公开视频更接近高级 lecture series，而不是稳定的零基础课，所以以 GPU MODE 为完整课程载体；完成其 CUDA/Triton/CUTLASS/CuTe 相关讲次后，再进入官方 [CuTe tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html) 做项目。
- **停止条件**：能解释 program instance、tile、layout、MMA、autotune 的边界，并知道何时选 CUDA、Triton 或 CUTLASS。

#### 分支 C：TVM 和 MLIR

- **进入时间**：Triton Kernel 项目完成后再学；它不是 TensorRT 前置课。
- **怎么选课**：先完整学陈天奇 MLC 10 讲及配套 [中英文课程页](https://mlc.ai/summer22-zh/schedule)，再看 LLVM MLIR workshop。
- **看完即做**：完成 TensorIR schedule/自动优化 notebook；再走一遍 MLIR Toy 的 AST → dialect → passes → lowering → LLVM。
- **停止条件**：能解释 graph optimization、tensor program、IR、dialect、pass、lowering、runtime 的层级关系，并能把一个优化落到可运行代码。

### 3.3 节点文档与项目总览

下面的资料表只在看课和做项目时定点查询，不再作为起步阅读顺序。

| 阶段 | 建议时间 | 关键字 | 英文免费主资料 | 中文免费辅助 | 阶段项目与验收 |
|---|---:|---|---|---|---|
| 0. 基础工具 | 1-2 周 | Python、PyTorch、C++、Linux shell、Git、CMake、venv、pip、NumPy、ONNX、gdb、进程、线程、内存 | [Python 官方教程](https://docs.python.org/3/tutorial/)、[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)、[NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)、[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)、[LearnCpp](https://www.learncpp.com/)、[MIT Missing Semester](https://missing.csail.mit.edu/)、[Pro Git](https://git-scm.com/book/en/v2)、[OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/) | [Python Full Course for free](https://www.youtube.com/watch?v=ix9cRaBkVe0)、[黑马程序员全套 Python 基础教程](https://www.bilibili.com/video/BV1o4411M71o/)、[NVIDIA 中文在线培训入口](https://www.nvidia.cn/developer/online-training/) | 完成 `model-tools`：venv、数据 manifest、NumPy/PyTorch GEMM、PyTorch→ONNX 校验、JSON/CSV benchmark；同时完成 C++17/CMake CPU 矩阵乘法 |
| 1. Linux 与内核 | 并行 2-4 周 | syscall、进程、调度、内存、module、driver、device tree、BSP、交叉编译 | [LFS101 Introduction to Linux](https://training.linuxfoundation.org/training/introduction-to-linux/)、[LFD103 Kernel Development](https://training.linuxfoundation.org/training/a-beginners-guide-to-linux-kernel-development-lfd103/)、[Bootlin Training Materials](https://bootlin.com/docs/)、[Kernel Driver API](https://www.kernel.org/doc/html/latest/driver-api/index.html) | [Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)、[Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)、[TinyLab 内核文档中文版](https://tinylab-1.gitbook.io/linux-doc/zh-cn) | 编译并加载 hello module；读懂一个字符设备驱动；能解释设备树、module、用户态/内核态 |
| 2. CUDA 入门 | 1-2 周 | host/device、thread/block/grid、nvcc、global/shared/register/constant/unified memory | [NVIDIA CUDA C++ Tutorial](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)、[CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)、[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | [NVIDIA 中文 CUDA 平台](https://developer.nvidia.cn/cuda)、[NVIDIA CUDA 中文在线培训](https://www.nvidia.cn/training/online/) | vector add、SAXPY、二维矩阵加法；CPU/GPU 结果一致并记录耗时 |
| 3. CUDA 性能与库 | 2-3 周 | coalescing、shared memory、bank conflict、occupancy、warp、reduction、transpose、streams、events、pinned memory、cuBLAS | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[CUDA Samples](https://github.com/NVIDIA/cuda-samples)、[CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)、[Stanford CS149](https://gfxcourses.stanford.edu/cs149/fall25/) | [NVIDIA CUDA 中文开发者入口](https://developer.nvidia.cn/cuda)、[NVIDIA 中文在线课程](https://www.nvidia.cn/developer/online-training/) | transpose + reduction + tiled GEMM；用 profile 数据解释优化前后差异 |
| 4. CUDA 调试与底层 | 1-2 周 | Compute Sanitizer、Nsight Systems、Nsight Compute、PTX、SASS、cuobjdump、nvdisasm、WMMA | [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)、[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html) | [NVIDIA CUDA Toolkit 中文入口](https://developer.nvidia.cn/cuda-toolkit) | 找出一个真实热点；修复一次越界或 race；提交一份含 timeline、kernel metrics、SASS 对比的报告 |
| 5. ONNX 与 TensorRT | 3-4 周 | ONNX、trtexec、engine、builder、runtime、dynamic shape、FP16、INT8、calibration、plugin、Polygraphy | [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)、[TensorRT GitHub](https://github.com/NVIDIA/TensorRT)、[Torch-TensorRT](https://github.com/pytorch/TensorRT) | [TensorRT 中文入门](https://developer.nvidia.cn/tensorrt-getting-started)、[NVIDIA 中文超级训练](https://www.nvidia.cn/developer/online-training/super-training/)（只看概念，命令以英文最新文档为准） | 一个模型完成 PyTorch → ONNX → TensorRT；比较 FP32/FP16/INT8 的精度、延迟、内存 |
| 6. Jetson 与视频推理 | 3-4 周 | Jetson、V4L2、GStreamer、DeepStream、nvinfer、tracker、metadata、VPI、tegrastats | [DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)、[DeepStream GitHub](https://github.com/NVIDIA/DeepStream)、[Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)、[Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/) | [DeepStream 中文开发入口](https://developer.nvidia.cn/deepstream-sdk)、NVIDIA 中文视频资料 | 1 路摄像头扩展到 4 路；完成 decode/infer/tracker/tiler/metadata；测量 FPS、延迟、温度和功耗 |
| 7. 通用 Serving | 2-3 周 | HTTP/gRPC、模型仓库、backend、dynamic batching、ensemble、health、Prometheus、Perf Analyzer | [Triton Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)、[Triton Tutorials](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/README.html)、[Triton Server](https://github.com/triton-inference-server/server)、[Triton Tutorials Repo](https://github.com/triton-inference-server/tutorials) | [NVIDIA 中文 TensorRT/Triton 课程入口](https://www.nvidia.cn/developer/online-training/super-training/)（版本较旧） | TensorRT backend + model repository + dynamic batching + gRPC/HTTP + metrics；画并发/延迟曲线 |
| 8. LLM Edge Serving | 3-4 周 | tokenizer、KV cache、TTFT、TPOT、tokens/s、INT4、AWQ、continuous batching、OpenAI-compatible API | [TensorRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/)、[Edge-LLM GitHub](https://github.com/NVIDIA/TensorRT-Edge-LLM)、[vLLM Docs](https://docs.vllm.ai/en/latest/)、[TensorRT-LLM Docs](https://nvidia.github.io/TensorRT-LLM/) | [Jetson AI Lab Edge-LLM 教程](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)、[NVIDIA 中文 TensorRT 入口](https://developer.nvidia.cn/tensorrt-getting-started) | 在 Orin 上跑一个支持矩阵内的 Qwen3 小模型；记录 TTFT、TPOT、吞吐、峰值内存和上下文长度 |
| 9. 编译器与 Kernel DSL | 后续 4-8 周 | Triton、CUTLASS/CuTe、TVM、MLIR、fusion、lowering、autotune | [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)、[Triton GitHub](https://github.com/triton-lang/triton)、[TVM Docs](https://tvm.apache.org/docs/)、[MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)、[GPU Mode](https://github.com/gpu-mode) | 中文资料作为术语辅助，不作为版本依据 | 用 Triton 写 vector add、softmax、matmul；把一个算子与 CUDA 版本做性能和可读性对比 |

## 4. 现在开始：CUDA 六周冲刺

你提供的两个 CUDA 视频可以保留，但它们的定位不同：

- [NVIDIA Modern CUDA C++ Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)：作为主课程视频，配合 NVIDIA 的公开 notebook 和 GitHub 实验。
- [freeCodeCamp CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4)：作为长线补充，覆盖 CUDA API、矩阵乘法、Triton 和 PyTorch extension。
- [freeCodeCamp 课程代码仓库](https://github.com/Infatoshi/cuda-course)：不要只看视频，按章节运行并改代码。

结论是：**这两个视频足够启动，但不够形成完整能力**。缺口在官方编程模型、Best Practices、Sanitizer、Nsight、架构分析、TensorRT 和 Jetson 项目。下面的六周计划把缺口补齐。

六周内不要从文档通读开始。每周固定采用：**对应视频章节 → 当天复现 → 改一个参数或实现 → 出现具体问题后查文档 → 写一页结果**。

### 第 1 周：能编译、能运行、能解释线程层次

关键字：

nvcc、host code、device code、kernel launch、grid、block、thread、global memory、device synchronization、error checking。

资料顺序：

1. Modern CUDA C++ Playlist 的 introduction、execution spaces 和第一个 Kernel。
2. 同步完成 NVIDIA CUDA C++ Tutorial 对应 notebook。
3. freeCodeCamp 课程中 setup、C/C++ review、first kernels 部分。
4. 只在索引或 launch 行为不清楚时，查 CUDA Programming Guide 的 programming model 对应小节。

必须完成：

- vector add / SAXPY。
- CPU reference implementation。
- cudaGetLastError 和同步错误检查。
- 记录编译命令、GPU 型号、数据规模、CPU/GPU 时间。

过关标准：

- 能画出 grid → block → thread 的索引关系。
- 能说明 kernel launch 是异步的，以及何时需要同步。
- 能解释 unified memory 不是免费高速显存。

### 第 2 周：内存访问和常见基础 Kernel

关键字：

coalesced access、stride、shared memory、bank conflict、register、occupancy、warp divergence、transpose、reduction。

必须完成：

- naive transpose 与 tiled transpose。
- naive reduction 与 shared-memory reduction。
- 至少一次故意制造的非合并访问，并用数据说明代价。

视频与实验顺序：

1. [大规模并行处理器编程实战](https://www.bilibili.com/video/BV1gz421o7uH/)第 4-6 讲：GPU 架构、memory tiling、performance considerations。
2. [Stanford CS149 2023](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)的 GPU Architecture/CUDA 和 data-parallel thinking 相关讲次。
3. 从 [CUDA Samples](https://github.com/NVIDIA/cuda-samples) 选择 transpose、reduction 或相近 sample 改写。
4. 指标或规则不清楚时，再查 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。

过关标准：

- 不用“GPU 更快”解释结果，而是能从访存、并行度、同步和算术强度解释结果。
- 能看懂一个 sample 的 launch configuration 和 memory layout。

### 第 3 周：Streams、Events 和 CPU/GPU 重叠

关键字：

cudaStream、cudaEvent、pinned host memory、pageable memory、overlap、double buffering、CUDA Graph、asynchronous copy。

必须完成：

- 把大数组分块，使用两个 stream 尝试重叠 H2D、kernel、D2H。
- 用 CUDA events 计时，不用端到端墙钟时间替代 kernel 时间。
- 对比同步版本和异步版本。

视频与实验顺序：

1. freeCodeCamp CUDA 课程的 streams、events、pinned memory 相关章节。
2. CUDA Samples 中的 streams、events、graph 相关示例。
3. [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) 中选择一个异步库调用示例。
4. 只为核对异步语义和限制，查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)对应小节。

过关标准：

- 能用 timeline 证明是否真的发生了重叠。
- 能区分 API 调用耗时、kernel 执行耗时和端到端延迟。

### 第 4 周：矩阵乘法、cuBLAS 和真实小项目

关键字：

tiled GEMM、arithmetic intensity、cuBLAS、cuBLASLt、FP32、FP16、TF32、WMMA、Tensor Core。

必须完成：

- naive GEMM。
- shared-memory tiled GEMM。
- cuBLAS GEMM。
- 比较准确率、吞吐和工作量，不把库调用当成黑盒结束。

主项目：

- [MNIST CUDA](https://github.com/Infatoshi/mnist-cuda)
- 配套课程：[CUDA Course](https://github.com/Infatoshi/cuda-course)

视频先看 freeCodeCamp 课程中的 matrix multiplication、cuBLAS、mixed precision 和 MNIST 项目章节；PMPP 第 5-6 讲用于补 tiling 与性能模型。

建议按以下阶段复现：

1. PyTorch baseline。
2. NumPy / C baseline。
3. naive CUDA。
4. cuBLAS。
5. streams、fused kernel、TF32 或 FP16。
6. 自己做一次 GEMM 或预处理 Kernel 的优化。

过关标准：

- 有一张表记录每个版本的准确率、p50/p95 延迟、吞吐和峰值内存。
- 能说明为什么自写 Kernel 不一定比 cuBLAS 快。

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
4. 最后查 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) 和 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)核对反汇编结果。

Jetson 使用方式：

- Jetson 是 target；Nsight Systems / Nsight Compute 的图形界面通常放在 x86 Linux host。
- target 端可以先使用 CLI 采集，再把报告传到 host 分析。
- 采集前固定功耗模式、输入规模、频率和温度条件。

必须交付：

- 一份优化前/后报告。
- 一张 Nsight Systems timeline。
- 一张 Nsight Compute 热点指标截图或导出结果。
- 一次 Compute Sanitizer 结果。
- 一段由 profile 数据支持的优化结论。

### 第 6 周：CUDA 图、协作组、Tensor Core 与 PTX/SASS

关键字：

cooperative groups、CUDA Graph、warp shuffle、WMMA、Tensor Core、PTX、SASS、sm_87、fatbin。

视频与实验顺序：

1. 从 [GPU MODE Lectures](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)选择 profiling、Tensor Core、SASS 和高性能 GEMM 相关讲次。
2. 回看 PMPP 的 GPU architecture、performance 和 Nsight 讲次。
3. 从 [CUDA Samples](https://github.com/NVIDIA/cuda-samples)选择 cooperative groups、shuffle、CUDA Graph 或 WMMA 示例改写。
4. 只为核对语义和指令，查 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)和 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)。

必须完成：

- 用 cooperative groups 或 warp primitive 改写一个 reduction/scan 小实验。
- 用 cuobjdump 或 nvdisasm 检查一个 Kernel 的目标架构和关键指令。
- 在 Orin 上确认 sm_87 相关编译配置，不照抄 H100 的 sm_90、TMA、WGMMA 示例。

## 5. 真实公开项目阶梯

### 项目 A：CUDA 基础 Kernel 仓库

项目来源：

- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)

你要自己维护一个小仓库，至少包含：

- vector add / SAXPY
- transpose
- reduction
- tiled GEMM
- stream overlap
- sanitizer 和 benchmark 脚本

验收：每个例子有 CPU reference、正确性测试、数据规模和性能记录。

### 项目 B：MNIST CUDA 逐级优化

项目来源：

- [Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)
- [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)

验收：完成 PyTorch → C/CUDA → cuBLAS → streams/fusion → FP16 或 TF32 的至少四个版本，并解释每次变化。

### 项目 C：TensorRT 推理部署

项目来源：

- [NVIDIA TensorRT](https://github.com/NVIDIA/TensorRT)
- [TensorRT Quick Start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [TensorRT Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)

任务：

1. 训练或选择一个小模型。
2. 导出 ONNX 并验证输出。
3. 用 trtexec 或 Python/C++ API 构建 Engine。
4. 比较 FP32、FP16、INT8。
5. 加入 dynamic shape。
6. 对不支持的算子尝试 plugin 或改图。

验收指标：

- accuracy delta
- engine build 是否可复现
- warmup 后 p50/p95 latency
- throughput
- 峰值 unified memory
- 输入 shape 和 batch 的影响

### 项目 D：Jetson 多路视频推理

项目来源：

- [NVIDIA DeepStream](https://github.com/NVIDIA/DeepStream)
- [DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)
- [DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

任务：

- 先跑单路文件或 USB 摄像头。
- 再扩展为 4 路文件/RTSP 输入。
- 加入 decode、nvinfer、tracker、tiler、OSD 和 metadata。
- 对比 OpenCV/Python 串行实现与 DeepStream pipeline。
- 用 tegrastats 和 Nsight Systems 观察 CPU、GPU、内存、解码和推理。

验收：在固定功耗和输入条件下，报告每路 FPS、端到端延迟、GPU 利用率、内存、温度和降频情况。

注意：旧的 [deepstream_reference_apps](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps) 仓库已经停止更新；当前新项目优先从 [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) 主仓库和对应 release 开始。

### 项目 E：Triton 模型服务

项目来源：

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)
- [Triton Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
- [Backend Platform Support Matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md)

任务：

- 建立 model repository。
- 用 TensorRT backend 提供一个图像分类或检测模型。
- 配置 dynamic batching 和多个 model instance。
- 写 HTTP 和 gRPC client。
- 加入 health、metrics、错误处理和超时。
- 用 Perf Analyzer 测量 concurrency、latency 和 throughput。
- 再做一个 ensemble，把预处理、推理、后处理串起来。

验收：画出并发数、batch、p50/p95 延迟和吞吐之间的曲线，并说明哪一个配置适合 Orin 的内存和功耗约束。

### 项目 F：Orin 上的 Edge LLM

项目来源：

- [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- [TensorRT-Edge-LLM Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)
- [Jetson AI Lab Edge-LLM Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/model-optimization/tensorrt-edge-llm.mdx)

建议 capstone：

- 选择支持矩阵中的 Qwen3 小模型，优先从 0.6B、1.7B 或 4B 级别开始。
- 先做 FP16 可运行基线，再尝试 INT4/AWQ。
- 用一个简单 HTTP API 包装推理。
- 测量 TTFT、TPOT、tokens/s、峰值内存、上下文长度和连续请求行为。

不要一开始做：

- 14B 以上模型的盲目移植。
- 只看 tokens/s、不测首 token 延迟。
- 把 server GPU 上的 vLLM/TRT-LLM 命令直接复制到 Jetson。

## 6. TensorRT、Triton、vLLM、TensorRT-LLM 的关系

### 6.1 TensorRT 是运行时和优化器

典型流程：

PyTorch / TensorFlow → ONNX → TensorRT Builder → hardware-specific Engine → TensorRT Runtime

TensorRT 解决的是模型图优化、Kernel 选择、精度、Engine 构建和执行。它本身不是完整的多租户 HTTP 服务。

### 6.2 Triton 是通用推理服务层

Triton 提供：

- model repository
- 多种 backend
- HTTP/gRPC
- dynamic batching
- concurrent model execution
- ensemble
- health 和 metrics

因此，**LLM serving 可以使用 Triton，但 LLM serving 不等于 Triton**。Triton 是一个通用 serving layer；模型运行时可能是 TensorRT、PyTorch、ONNX Runtime、Python backend 或其他 backend。

### 6.3 vLLM、TensorRT-LLM、Edge-LLM 的定位

- **vLLM**：偏通用服务器 GPU 的 LLM runtime/serving，重点是 KV cache、continuous batching 和高吞吐。
- **TensorRT-LLM**：偏 NVIDIA 服务器 GPU 的高性能 LLM runtime，通常需要匹配的 GPU、CUDA、容器和版本。
- **TensorRT-Edge-LLM**：面向 Jetson/边缘设备的路线；对当前 Orin NX 更有现实意义。
- **Triton**：可以承载多种模型和 backend，是服务编排与推理 API 层；是否适合某个 Jetson backend 必须查平台矩阵。

推荐顺序：

1. TensorRT 单模型运行。
2. Triton + TensorRT backend。
3. 在服务器 GPU 上理解 vLLM/TensorRT-LLM 的 serving 概念。
4. 在 Orin 上使用 TensorRT-Edge-LLM 的支持矩阵内路径。

Jetson 平台特别注意：Triton 的 backend/platform 组合不是全部可用，尤其不要默认 Python backend 的 GPU 能力与 x86 服务器一致；以 [官方 backend platform matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md) 为准。

## 7. Linux 驱动分支：什么时候学、学到哪里

### 7.1 主线中的位置

如果目标是 AI inference/performance engineer，Linux 驱动不应阻塞 CUDA 主线。每周额外投入约 2 小时即可：

Linux 用户态 → module → 字符设备 → ioctl/poll/mmap → DMA/中断 → device tree → Jetson BSP → 调试与性能

如果目标明确是 BSP、内核或驱动工程师，再把它升级为主线。

### 7.2 免费资料

先从下面两门完整视频中选一门主课：

- [Linux Device Drivers Development Course for Beginners](https://www.youtube.com/watch?v=iSiyDHobXHA)：英文，约 5 小时，快速覆盖环境、内核/用户态、module、syscall 和 `/proc`。
- [韦东山：嵌入式 Linux 驱动开发基础知识](https://www.bilibili.com/video/BV14f4y1Q7ti/)：中文，50 讲、约 17 小时，覆盖字符设备、GPIO、中断、工作队列和 `mmap`。

做实验时再查询：

- [LFD103: A Beginner's Guide to Linux Kernel Development](https://training.linuxfoundation.org/training/a-beginners-guide-to-linux-kernel-development-lfd103/)
- [Bootlin Public Training Materials](https://bootlin.com/docs/)
- [Bootlin Kernel Training](https://bootlin.com/training/kernel/)
- [Linux Kernel Driver API](https://www.kernel.org/doc/html/latest/driver-api/index.html)
- [Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)
- [Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)
- [TinyLab Linux 文档中文版](https://tinylab-1.gitbook.io/linux-doc/zh-cn)

### 7.3 驱动分支项目

1. 在 x86 虚拟机或开发板上写可加载 hello module。
2. 写一个字符设备，支持 open/read/write/ioctl。
3. 用 poll 或 epoll 做阻塞/非阻塞读。
4. 阅读一个真实 Jetson 外设驱动和对应 device tree。
5. 学会看 dmesg、ftrace、trace-cmd、perf 和 sysfs。

不要为了练习驱动直接修改 Orin 的启动链或替换 NVIDIA 核心 GPU 驱动。先使用可恢复的虚拟机、独立模块和备份的 Jetson 系统。

## 8. GPU 编译器与 Kernel DSL 分支

这条线放在 CUDA 和 Nsight 之后：

CUDA C++ → 性能模型 → Triton → CUTLASS/CuTe → TVM → MLIR

### Triton

- 完整中文视频主课：[Triton 从入门到大师](https://www.bilibili.com/video/BV1fMyWBgERM/)
- 英文完整专题：[Stanford CS336 2026 Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)
- 课程代码：[evintunador/triton_docs_tutorials](https://github.com/evintunador/triton_docs_tutorials)
- [Triton Language](https://triton-lang.org/main/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton GitHub](https://github.com/triton-lang/triton)

学习顺序：

1. vector add
2. fused softmax
3. matmul
4. layer norm
5. fused attention

Triton 课程适合在 x86 NVIDIA GPU 或公开 notebook 环境学习。不要默认当前 Triton、PyTorch 和 ARM64 Jetson 组合可以原生安装；在 Orin 上最终仍要回到 CUDA/TensorRT/Edge-LLM 的实际支持范围。

### CUTLASS、TVM、MLIR

- [GPU MODE 完整高级系列](https://www.youtube.com/playlist?list=PLjG_zIhhamWJRAuxYNBI0QvVE0dmwNQLL)：用于 CUTLASS、CuTe、Tensor Core、SASS 和真实算子案例。
- [MLC 机器学习编译 10 讲](https://www.bilibili.com/video/BV15v4y1g7EU)：中文主课，配套中英文笔记和 notebook。
- [LLVM MLIR Tutorial](https://www.youtube.com/watch?v=Y4SvqTtOIDk)：英文官方完整 workshop。
- [CUTLASS](https://github.com/NVIDIA/cutlass)：学习模板化 GEMM、Tile、Tensor Core 和 CuTe。
- [Apache TVM Documentation](https://tvm.apache.org/docs/)：学习模型编译、算子调优和部署。
- [TVM End-to-End Optimization Tutorial](https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html)：从模型到优化。
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)：学习 AST、dialect、lowering 和 LLVM。

不要在 CUDA 基础没掌握时从 MLIR 开始；那会把计算、编译器和框架问题同时混在一起。

## 9. 24 周执行节奏

假设每周投入 10-12 小时；每周只有 5-6 小时时，把日历时间大约翻倍。

如果已经能独立完成 C++17/CMake 小程序，能创建可复现的 Python `venv`、使用 NumPy 和 PyTorch 处理 tensor，并熟悉 Shell、进程、权限、SSH 和 `gdb`，才可以跳过第 1-2 周；否则先完成节点 0 的 Python + PyTorch `model-tools` 和 C++ 项目。遇到基础问题时再回看节点 0-1 的对应章节。

| 周数 | 主线 | 交付物 |
|---:|---|---|
| 1-2 | Python、PyTorch、C++、Linux、Git、CMake、venv、NumPy、ONNX、gdb | `model-tools`（manifest、PyTorch→ONNX 校验、JSON/CSV benchmark）+ CPU 矩阵乘法和 Git 记录 |
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

## 10. 每个阶段统一的验收标准

不要只以视频看完作为完成条件。每个项目都必须回答以下问题：

### 正确性

- 是否有 CPU 或框架 reference？
- 随机输入、边界输入和不同 batch 是否通过？
- Compute Sanitizer 是否干净？
- 模型导出前后精度差异是多少？

### 性能

- 测量的是 kernel、runtime 还是端到端？
- 是否排除了首次加载和 warmup？
- 是否报告 p50、p95，而不是只报平均值？
- 优化前后是否有 Nsight 或 benchmark 证据？

### Edge 条件

- 功耗模式是什么？
- 温度和是否降频？
- unified memory 峰值是多少？
- 冷启动和热运行结果是否不同？
- 输入分辨率、batch、并发是否固定？

### Serving

- 健康检查、超时、错误和重启如何处理？
- 并发增加时延迟如何变化？
- dynamic batching 是否真的提高吞吐？
- 是否有 Prometheus 或等价指标？

### LLM

- 使用的模型和量化格式是否在目标平台支持矩阵内？
- 是否分别测量 TTFT、TPOT、tokens/s？
- 峰值内存和最大上下文是多少？
- 多请求时是否出现 OOM、抖动或严重排队？

## 11. Orin NX 上的版本和操作纪律

1. 先确认本机 JetPack、L4T、CUDA、TensorRT、DeepStream 版本，再执行任何教程命令。
2. 目标端优先使用 JetPack 提供的软件包；不要把 x86 CUDA 仓库或服务器 GPU 容器混装到 Jetson。
3. 编译 CUDA 时确认目标架构为 Orin 的 sm_87，或使用本机工具链推荐的架构参数。
4. 不要为了跟旧视频一致而随意降级 JetPack；先把旧 API 映射到当前文档。
5. DeepStream、Triton、CUDA 和 TensorRT 的容器必须匹配 JetPack/L4T；先看 release notes。
6. Nsight 的 target/host 分工要明确：Jetson 采集，x86 host 分析通常更方便。
7. 任何性能结论都要记录功耗、温度、输入 shape、batch、并发和软件版本。
8. 对 LLM，先选择支持矩阵内的小模型和精度；不要先从最大模型开始。
9. 版本敏感的 [DeepStream Release Notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html)、[TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)、[Edge-LLM Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html) 必须在每次部署前复核。

## 12. 中文资料的正确用法

### 推荐作为主线的中文完整视频

- CUDA 性能：[大规模并行处理器编程实战 7 讲](https://www.bilibili.com/video/BV1gz421o7uH/)
- Nsight：[CUDA Developer Tools 同课中文镜像](https://www.bilibili.com/video/BV14RU6BmE5u/)
- TensorRT：[NVIDIA TensorRT 官方教程 4 部分](https://www.bilibili.com/video/BV15Y4y1W73E)
- Triton Server：[NVIDIA Triton 从入门到精通 20 讲](https://www.bilibili.com/video/BV1KS4y1v7zd/)
- LLM Serving：[大模型推理技术研究 9 讲](https://www.bilibili.com/video/BV1k2L9zyEt7/)
- Linux 驱动：[韦东山嵌入式 Linux 驱动开发基础 50 讲](https://www.bilibili.com/video/BV14f4y1Q7ti/)
- Triton Kernel：[Triton 从入门到大师 10 讲](https://www.bilibili.com/video/BV1fMyWBgERM/)
- AI 编译器：[MLC 机器学习编译 10 讲](https://www.bilibili.com/video/BV15v4y1g7EU)

做项目时再使用中文查询入口：

- [NVIDIA 中文 CUDA 平台](https://developer.nvidia.cn/cuda)
- [NVIDIA 中文 TensorRT 入门](https://developer.nvidia.cn/tensorrt-getting-started)
- [Linux 内核文档中文版](https://docs.linuxkernel.org.cn/)
- [Linux Kernel Labs 中文翻译](https://github.com/linux-kernel-labs-zh/docs-linux-kernel-labs-zh-cn)

### 中文视频和字幕

中文视频适合建立概念、跟随界面和降低第一次阅读成本；但 CUDA、TensorRT、DeepStream 的版本变化很快。中文镜像可以看，代码必须回到官方仓库：

- [NVIDIA Modern CUDA C++ Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)
- [中文辅助镜像：Modern CUDA C++](https://www.bilibili.com/video/BV1QvSKB4EMr/)

Bilibili 上由 NVIDIA英伟达、原作者或大学发布的课程可以作为主课；第三方翻译镜像的版权、字幕质量和更新状态不由上游项目保证，因此只作为语言辅助，不作为唯一依据。

## 13. 明确排除的内容

以下内容不放入免费主路线：

- NVIDIA DLI 中标价 30/90 美元的完整实践课程。
- 付费 Bootlin 讲师课程；只使用 Bootlin 公开 slides、labs 和源码。
- 依赖付费云 GPU 才能完成的实验。
- 仅针对 H100/Hopper 的高级课程作为 CUDA 入门。
- 旧版 DeepStream/Triton 教程中的固定安装命令。
- 未说明来源、版本和许可证的课程搬运或代码集合。

例如，H100 专用课程 [H100-Course](https://github.com/cudacourseh100/H100-Course) 可以在以后有 Hopper 机器时选修，但不适合作为 Orin SM87 的当前主线。

## 14. 最终毕业项目

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

最终报告至少回答：

1. 哪些计算由现成库完成，哪些 Kernel 是自己写的？
2. 哪个热点经过了 Profile，优化前后改变了什么？
3. TensorRT 的精度、延迟和内存折中是什么？
4. DeepStream 相比普通 OpenCV pipeline 的收益在哪里？
5. Triton 在这里解决了什么，哪些问题它没有解决？
6. Orin 上的 LLM 与服务器 GPU 上的 vLLM/TensorRT-LLM 有哪些限制差异？
7. 在 25W、16GB unified memory 下，系统的实际上限是什么？

## 15. 一页式执行清单

### 现在

- [ ] 安装并确认 Python `venv`、NumPy、PyTorch/ONNX 工具链，完成节点 0 的 `model-tools`。
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

## 16. 学完整套后：能力水平、技术栈与岗位

### 16.1 先给准确定位

如果你只是把视频看完，结果是“知道很多名词”，还不能据此判断达到岗位要求。

如果你把每个节点的项目、benchmark、性能报告和 Orin 毕业项目都真正完成，那么比较准确的定位是：

> **具备端到端 GPU 推理与边缘 AI 系统能力的初级到初中级工程师。**

更具体地说，你的优势不是某一门 API 背得多，而是可以把一条链路从底层做到应用：

**写 CUDA Kernel → 用 Sanitizer/Nsight 找证据 → 用 TensorRT 做 Engine 和精度/性能权衡 → 用 Jetson/DeepStream 跑真实视频 → 用 Docker/gRPC/Triton 提供服务 → 在 Orin 上对 LLM 做资源受限的推理 benchmark。**

这相当于一个高度聚焦的 AI Systems / Inference Engineer 方向能力，不等同于“资深 CUDA 专家”、模型研究员或大规模 GPU 集群架构师。最终职级还取决于你此前的 C++/Linux/软件工程经验、学历和真实生产经历。

### 16.2 完成程度对应的水平

| 完成状态 | 能力判断 | 可以尝试的岗位层级 |
|---|---|---|
| 只看完课程 | 能解释概念、复述流程，但缺少独立排障和可验证结果 | 不建议仅凭此投递专业 CUDA/推理岗位 |
| 完成节点项目和验收 | 能独立写小 Kernel、导出模型、构建 Engine、部署服务并做基本性能分析 | 实习、校招、初级 GPU/推理/边缘 AI 岗位；有软件经验时可投初级到中级 |
| 完成 Orin 毕业项目，并有公开代码、报告、可重复 benchmark | 能跨层定位瓶颈，解释精度、延迟、吞吐、内存、功耗和温度的取舍 | 初级到初中级推理优化、TensorRT、Jetson/DeepStream、AI Serving 岗位 |
| 再积累真实生产系统、线上故障和多 GPU/多机经验 | 能承担系统设计、容量规划、SLO、灰度和回滚 | 中级 ML Systems / Inference Platform；距离高级岗位仍需按岗位补齐规模化经验 |

### 16.3 你最终掌握的技术栈

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

### 16.4 最匹配的岗位

#### 第一梯队：完成核心主线后直接匹配

1. **推理优化工程师 / Inference Optimization Engineer**
   - 关键词：CUDA、TensorRT、ONNX、量化、Kernel、Nsight、延迟、吞吐。
   - 你能负责模型转换、精度对齐、Kernel/Profile 和端到端 benchmark。

2. **边缘 AI 部署工程师 / Edge AI Deployment Engineer**
   - 关键词：Jetson、ARM64、JetPack、TensorRT、GStreamer、DeepStream、摄像头。
   - 你能把模型和视频 pipeline 部署到真实设备，并处理功耗、温度和内存限制。

3. **TensorRT / GPU 性能工程师 / CUDA Software Engineer**
   - 关键词：CUDA C++、cuBLAS、shared memory、Tensor Core、Nsight、custom operator。
   - 适合初级或偏应用优化岗位；要投核心算子团队，还需继续补 C++ 模板、算法、架构和开源贡献。

4. **Jetson / DeepStream / 计算机视觉部署工程师**
   - 关键词：多路视频、GStreamer、RTSP、tracker、metadata、V4L2、实时性。
   - 这是本机硬件最容易形成差异化作品集的方向。

#### 第二梯队：完成 Serving 与 LLM 后匹配

5. **AI 推理平台工程师 / ML Inference Platform Engineer**
   - 关键词：Triton、gRPC、Docker、model repository、batching、metrics、Perf Analyzer。
   - 可以投单机或边缘 Serving 岗；涉及 Kubernetes、多集群、GPU 调度的岗位还需要补云原生和分布式系统。

6. **LLM 推理工程师 / LLM Inference Engineer**
   - 关键词：KV cache、TTFT/TPOT、量化、continuous batching、vLLM、SGLang、TensorRT-LLM、Triton。
   - 你的优势会是低层性能和边缘部署；服务器级岗位还要补多卡并行、NCCL、分布式推理、Kubernetes 和线上容量规划。

7. **嵌入式 AI 软件工程师 / Embedded AI Systems Engineer**
   - 关键词：现代 C++、Linux、ARM64、Jetson、传感器、GStreamer、部署和诊断。
   - 完成分支 A 后，可以进一步投驱动相邻、BSP 相邻和机器人/自动驾驶边缘软件岗位。

#### 高级分支岗位

8. **Triton Kernel / GPU Kernel Engineer**：需要完成 B1/B2，并增加 FlashAttention、算子融合、Triton/CUTLASS 源码和开源贡献。
9. **ML Compiler Engineer**：需要完成 C，并补编译原理、LLVM/C++、IR 设计、pass 和 lowering。
10. **Linux Driver / BSP Engineer**：需要完成分支 A，并补 C、设备树、DMA、中断、交叉编译、启动链和具体外设。

### 16.5 建议搜索的岗位关键词

中文：`CUDA开发工程师`、`GPU性能优化工程师`、`推理优化工程师`、`TensorRT部署工程师`、`模型部署工程师`、`边缘AI工程师`、`Jetson开发工程师`、`DeepStream开发工程师`、`AI推理平台工程师`、`LLM推理优化工程师`、`嵌入式AI工程师`、`AI系统工程师`。

英文：`CUDA Software Engineer`、`GPU Performance Engineer`、`Inference Optimization Engineer`、`ML Systems Engineer`、`Edge AI Engineer`、`Jetson/Embedded AI Engineer`、`DeepStream Engineer`、`TensorRT Engineer`、`ML Inference/Serving Engineer`、`LLM Inference Engineer`、`Kernel Engineer`、`Compiler Engineer`。

### 16.6 这套路线暂时不能让你胜任的岗位

- **大规模分布式训练/推理平台高级岗位**：还缺 Kubernetes、GPU 调度、NCCL、RDMA/InfiniBand、Ray、TP/PP/EP、故障恢复和多集群运维。
- **核心 CUDA 编译器或 GPU 架构岗位**：还缺编译器实现、微架构研究、数学建模、硬件验证或长期开源贡献。
- **纯算法研究员/模型训练岗位**：这条路线重点是系统和推理，不覆盖数据、训练、论文和模型创新。
- **资深 Linux 驱动/BSP 岗位**：除非把驱动分支提升为主线，否则只能算驱动相邻能力。
- **高级/Staff LLM 推理岗位**：通常还要求多年生产经验、分布式系统和对 vLLM/SGLang/TensorRT-LLM 的源码级贡献。

### 16.7 真实求职时最重要的证明材料

不要把简历写成技术名词清单。至少准备四个可打开、可复测的证据：

1. **CUDA 性能项目**：naive → tiled → library 的代码、正确性测试、Nsight 报告和数据表。
2. **TensorRT 项目**：ONNX、Engine 构建脚本、精度差异、FP32/FP16/INT8 benchmark。
3. **Jetson/DeepStream 项目**：多路视频、固定功耗、FPS/延迟/温度/内存报告和一键启动说明。
4. **Serving/LLM 项目**：Triton 或 Edge-LLM 服务、并发曲线、TTFT/TPOT、OOM 边界和版本清单。

当前公开岗位的关键词与这条路线高度重合，但岗位规模差异很大：边缘系统岗位常同时要求现代 C++、Linux、Jetson/ARM64、GStreamer 和硬件相邻排障；模型加速岗位常要求 TensorRT、ONNX Runtime、CUDA Kernel、Nsight 和量化；大规模推理岗位则会进一步要求 vLLM/SGLang、Kubernetes、分布式系统和多 GPU 通信。可参考 [Shield AI Edge Systems 岗位](https://jobs.lever.co/shieldai/7e80ee36-5e20-4a4b-ba65-87d43db66cd6)、[昆仑芯模型加速与部署岗位](https://kunlunxin.zhiye.com/xiangqing?jobId=151141586)、[百度异构计算岗位](https://talent.baidu.com/jobs/detail/GRADUATE/15a59bf3-83f9-4c35-8d5e-bce6c50c59cc)和 [Apple Model Inference 岗位](https://jobs.apple.com/en-in/details/200671782-0836/machine-learning-engineer-model-inference?team=MLAI)。

## 17. 关键链接索引

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
