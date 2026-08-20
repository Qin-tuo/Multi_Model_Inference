# 免费 CUDA / Jetson 学习课程

面向 NVIDIA Jetson Orin NX 16GB、JetPack 7.2.1 / Jetson Linux R39.2.1 的免费学习资源整理。

## 快速选择

| 目标 | 首选资源 | 平台 |
| --- | --- | --- |
| 中文系统学习 CUDA | [GPU 并行计算与 CUDA 编程](https://www.bilibili.com/video/BV15Y4y1F7tE/) | B 站 |
| 英文一套学完 | [CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4) | YouTube / freeCodeCamp |
| 官方教材与练习 | [Fundamentals of Accelerated Computing with Modern CUDA C++](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp) | NVIDIA GitHub |
| Jetson 实战 | [Jetson AI Lab Tutorials](https://www.jetson-ai-lab.com/tutorials) | NVIDIA 官方 |

## 一、中文 CUDA 课程

### 1. GPU 并行计算与 CUDA 编程

[B 站课程链接](https://www.bilibili.com/video/BV15Y4y1F7tE/)

共 47 集，适合从零开始，主要内容包括：

- CUDA 环境、GPU 原理和 CUDA 编程模型
- Grid、Block、Thread 和 Kernel
- Global、Shared、Constant、Texture Memory
- CPU/GPU 数据传输与同步
- 归约、扫描、排序和直方图
- CUDA Stream、多 GPU、MPI-CUDA

这是中文资源里最适合作为主线课程的一套。建议按播放列表顺序学习，不要只挑安装视频看。

### 2. 异构并行编程 - CUDA，UIUC ECE408

[B 站课程链接](https://www.bilibili.com/video/BV11U4y1n7C8/)

UIUC ECE408/CS483 大学课程的中文搬运版，共 47 个视频。内容更偏理论、并行算法和性能优化，适合完成基础课程后继续学习：

- GPU 与 CPU 架构差异
- CUDA 内存模型和线程调度
- 矩阵乘法、卷积、归约等并行算法
- 内存带宽、访存模式和性能优化
- CUDA 工具和实验

课程主页和实验链接可以从视频简介进入。UIUC ECE408 也被 NVIDIA 列入大学 CUDA 课程资源列表：[NVIDIA Existing University Courses](https://developer.nvidia.com/educators/existing-courses)。

### 3. NVIDIA CUDA C++ 中文课程版本

[B 站课程链接](https://www.bilibili.com/video/BV1QvSKB4EMr/)

这是 NVIDIA 现代 CUDA C++ 课程的中文视频版本，课程简介提供了官方 GitHub 练习资源：[NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)。

建议把它作为中文讲解补充，代码和练习以官方 GitHub 仓库为准。

## 二、YouTube 免费课程

### 1. freeCodeCamp：CUDA Programming Course

[YouTube 视频](https://www.youtube.com/watch?v=86FAWCzIe_4)

接近 12 小时，适合英语阅读能力一般但希望完整学习的人。课程覆盖：

- C/C++ 基础复习
- GPU 架构和深度学习生态
- 第一个 CUDA Kernel
- CUDA API 和内存管理
- 矩阵乘法优化
- Triton 和 PyTorch CUDA 扩展
- MNIST 多层感知机实战

推荐重点观看 0:47:03 之后的 GPU、Kernel 和 CUDA API 部分；C/C++ 已经熟悉时可以加速播放。

### 2. NVIDIA Accelerated Computing Hub CUDA C++ 播放列表

[YouTube 播放列表](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr)

配套仓库：[CUDA C++ Tutorial](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/cuda-cpp)

内容包含视频、讲义、Notebook 和练习，主题包括：

- Execution Spaces
- Serial 与 Parallel 算法
- CUDA Memory Spaces
- Asynchrony 与 CUDA Streams
- Nsight 和 NVTX
- Kernel、同步、Histogram
- Shared Memory 和 Cooperative Groups

官方 Notebook 可以在本地运行，也可以使用 Google Colab 或 NVIDIA Brev。没有可用 NVIDIA GPU 时，可以先用 Colab 练习，具体 GPU 配额以账号当前情况为准。

## 三、NVIDIA 官方课程和配套资料

### 1. NVIDIA Accelerated Computing 中文学习路径

[NVIDIA Accelerated Computing 学习路径](https://www.nvidia.cn/training/learning-path/accelerated-computing/)

页面中包含 CUDA 入门、CUDA C++ 和 CUDA Python 等课程。免费内容以页面当前标注为准；部分中文进阶课程可能需要付费或单独注册。建议先看免费的 CUDA 入门内容，再决定是否参加进阶课程。

### 2. DLI：CUDA C++ 编程实战

[DLI CUDA C++ 课程](https://learn.nvidia.com/courses/course-detail?course_id=course-v1%3ADLI+S-AC-04+V2-ZH)

偏实操的 CUDA C++ 课程，适合已经掌握 Thread、Block、Grid 和基础内存管理之后学习。通常需要 NVIDIA 学习账号；课程是否包含云端实验、证书或其他限制，以页面当前说明为准。

### 3. DLI：CUDA Python

[DLI CUDA Python 课程](https://learn.nvidia.com/courses/course-detail?course_id=course-v1%3ADLI+S-AC-10+V1-ZH)

适合希望使用 Python、Numba 或 CUDA Python 进行 GPU 加速的人。建议先学 CUDA C++ 基础，再学习这一门，否则容易只会调用接口而不了解线程和内存模型。

### 4. CUDA 13.2 Programming Guide

[CUDA 13.2 Programming Guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/index.html)

这是与你当前主机 CUDA 13.2 版本对应的官方编程指南，适合查阅：

- CUDA 编程模型和执行层次
- Kernel 启动和线程组织
- 内存空间、同步和异步执行
- CUDA Runtime API
- Streams、Events 和并发执行

不建议一开始从头通读。先看视频和完成小练习，遇到 API 或概念再查对应章节。

### 5. CUDA Samples

[NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

官方示例代码，适合边学边运行。建议优先查看：

- `0_Introduction/vectorAdd`
- `0_Introduction/simpleOccupancy`
- `6_Advanced/simpleMultiCopy`
- `6_Advanced/convolutionTexture`
- `6_Advanced/streamOrderedAllocation`

Samples 的构建方式可能随 CUDA 版本变化，优先使用当前 CUDA 13.2 的仓库版本，不要照搬旧课程里的安装命令。

## 四、Jetson 专门资源

### 1. Jetson AI Lab Tutorials

[Jetson AI Lab 教程主页](https://www.jetson-ai-lab.com/tutorials)

适合 CUDA 基础完成后学习 Jetson 应用开发，包含：

- Jetson 入门和开发环境配置
- SSH、VS Code/Cursor 远程开发
- NVMe SSD、Docker 和内存优化
- LLM/VLM 在 Jetson 上部署
- TensorRT 推理和模型优化
- 摄像头、视觉模型和边缘 AI 应用

### 2. NVIDIA Jetson 软件入门

[Get Started with NVIDIA Jetson](https://developer.nvidia.com/embedded/learn/getting-started-jetson)

这是官方 Jetson 文档入口，适合查询设备、JetPack、开发工具、论坛和项目资料。对于第三方 C1902 载板，应优先参考载板厂商文档，并以实际支持的 JetPack 版本为准。

## 五、推荐学习顺序

### 阶段 1：CUDA 基础

1. B 站《GPU 并行计算与 CUDA 编程》前 10 到 15 集。
2. 学会 Thread、Block、Grid、Kernel 和基本内存拷贝。
3. 完成 `vectorAdd` 或 SAXPY。

### 阶段 2：内存和性能

1. 学习 Global Memory、Shared Memory 和 Memory Coalescing。
2. 实现矩阵加法、矩阵转置和基础矩阵乘法。
3. 学习同步、归约、Stream 和异步拷贝。

### 阶段 3：工具和优化

1. 使用 NVIDIA Hub Notebook 完成练习。
2. 使用 Nsight Systems 查看 CPU、数据传输和 Kernel 时间线。
3. 使用 Nsight Compute 分析访存、Occupancy 和 Kernel 性能。

### 阶段 4：Jetson 部署

1. 完成 Jetson AI Lab 的入门和 Docker/NVMe 配置。
2. 把 `vectorAdd`、矩阵乘法等程序移植到 Orin NX。
3. 再学习 TensorRT、摄像头、目标检测和 LLM/VLM 部署。

## 六、与你当前环境相关的注意事项

- 主机已经有 CUDA 13.2 和 `nvcc`，可以先学习和编译 CUDA C++。
- Jetson 端如果还没有安装 Jetson SDK Components，就可能没有 `nvcc`、头文件和开发库；此时可以先用 Google Colab，或等待 SDK Manager 完成目标端组件安装。
- 学习 CUDA 基础不需要等待 cuDNN 或 TensorRT 下载完成。cuDNN/TensorRT 主要在深度学习算子和推理部署阶段使用。
- 很多 B 站、UIUC 课程使用 CUDA 10/11，核心概念仍然适用，但安装命令可能过时。不要为了匹配视频而降级当前 CUDA 13.2。
- 不要在 Jetson 上随意添加 x86_64 CUDA 软件源。Jetson 的 CUDA、cuDNN、TensorRT 应使用与 JetPack 版本匹配的 ARM64 软件包或容器。
- 课程中的 `nvprof`、旧版 `cuda-memcheck` 等工具可能已经被 Nsight Systems、Nsight Compute 或 Compute Sanitizer 替代。

## 七、建议完成的第一个练习

按以下顺序完成即可验证学习效果：

1. CPU 实现向量加法。
2. CUDA Kernel 实现向量加法。
3. 比较 CPU 和 GPU 结果，加入错误检查。
4. 使用 CUDA Event 测量 Kernel 时间。
5. 改写为二维线程布局，实现矩阵加法。
6. 使用 Shared Memory 实现矩阵转置。
7. 使用 Nsight Systems 查看数据传输和 Kernel 执行时间。

完成这些练习后，再进入 TensorRT 和 Jetson 视觉/大模型项目，会更容易理解性能差异和问题来源。
