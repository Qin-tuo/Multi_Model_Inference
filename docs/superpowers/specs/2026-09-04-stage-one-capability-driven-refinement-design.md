# 阶段一能力驱动精细化设计

## 背景

当前路线图的阶段一已经列出 Python、NumPy、PyTorch、C++、Linux 和
Docker 的能力目标、课程、项目与停止条件，但“课程目录”和“阶段能力”没有
严格分开。结果是：

- 部分视频被描述为覆盖了实际没有讲授的工程专题；
- 多门入门课存在大量重复，却没有告诉学习者哪些章节需要跳过；
- NumPy、ONNX、CMake、GDB、现代 C++ 和 Docker 平台边界等必修能力缺少
  独立、连续的学习来源；
- 阶段一项目同时包含模型工具、CPU GEMM、Shell 监控和并发 TCP 服务，
  但课程先后顺序不足以支撑这些交付；
- “看完主课”容易被误认为完成阶段，而实际验收要求远高于课程覆盖。

本设计将阶段一从“按课程罗列”改为“按能力依赖推进”。课程只作为学习素材，
不再承担它没有覆盖的能力。主课中的无关部分明确跳过，真实缺口通过专题资料和
小实验补齐。

## 目标

1. 核验每门现有课程的真实来源、范围、时长和版本敏感性。
2. 明确课程之间的前置、交叉、重复、延后和替代关系。
3. 为每项阶段一能力指定主课选段、专题补课、最小实验和停止条件。
4. 让已完成的内容可以通过验收直接跳过，不要求重新刷课。
5. 保持阶段一聚焦工程底座，不提前吞并 CUDA、TensorRT、Serving 或驱动支线。
6. 让阶段一交付能够自然进入阶段二 CUDA，并为阶段五模型部署和阶段六 Serving
   保留清晰接口。

## 非目标

- 不把阶段一扩展成完整的数据科学、深度学习训练、操作系统或网络课程。
- 不要求顺序看完任何一门长课。
- 不在阶段一教授 TensorRT、Triton Server、Docker Compose、GPU 容器运行时或
  Kubernetes。
- 不把 Pandas、Matplotlib、Seaborn 或通用数据清洗设为阶段一毕业门槛。
- 不把 C++ 模板元编程、复杂原子内存序、Linux 内核/驱动或完整网络协议栈设为
  阶段一门槛。
- 不修改阶段二到阶段八的主体课程设计；只同步必要的阶段一引用和交接边界。

## 核心决策

### 能力优先于课程

阶段一按“能力目标 -> 课程选段 -> 专题补缺 -> 实验 -> 证据验收”组织。
课程名称下的目录只陈述该课程真实覆盖的内容；跨课程能力另设能力矩阵，不再把
路线图目标写成某门视频的课程大纲。

### 统一标签

阶段一使用下列标签：

| 标签 | 含义 |
|---|---|
| 主线必修 | 所有人都要掌握，并通过阶段项目验收 |
| 专题必修 | 主课没有完整覆盖，必须使用指定专题补齐 |
| 条件回补 | 只有触发条件出现时才学习，不顺序通刷 |
| 延后阶段 | 有价值，但由后续阶段负责 |
| 明确跳过 | 与当前目标无关或已经过时 |
| 已完成可验收 | 已经学习过；通过对应实验后无需重看 |

“官方校正”不是第二门课程。它负责校正当前 API、安装命令、平台支持和语义，
实际执行一律以当前官方文档为准。

### 精确记录来源

课程条目必须记录：原作者或机构、镜像/翻译性质、核验日期、选段名称。B 站
分 P 可能发生变化，因此同时记录 P 编号和章节标题；若两者不一致，以标题和
官方源内容为准。

### 课程完成不等于阶段完成

所有模块均以可运行产物和可复查证据结束。视频观看进度只进入学习日志，不作为
阶段门槛。

## 能力依赖与先后顺序

阶段一包含两条可以交错推进、最后汇合的链路：

```text
工程与模型链：
Linux Shell / Git
    -> Python 基础
    -> Python 工程化与 uv
    -> NumPy 数组、内存与图像布局
    -> PyTorch Tensor / Module / 推理 / DataLoader
    -> ONNX 导出与 ONNX Runtime 对齐
    -> model-tools
    -> Docker 单容器 CPU/headless 封装

系统与运行时链：
C++ 基础
    -> 生命周期 / RAII / STL / 智能指针
    -> 编译链接 / CMake / GDB / Sanitizer
    -> C++ CPU GEMM
    -> Linux fd / 进程 / 信号 / 线程
    -> socket 基础
    -> 并发 TCP 强化实验（按岗位选择）

汇合：
model-tools + C++ GEMM + Shell telemetry + Docker 基线
    -> 阶段一验收
    -> 阶段二 CUDA 与 GPU 性能
```

约束如下：

- NumPy 先于 PyTorch Tensor 是推荐顺序；已经先学 PyTorch Tensor 的学习者只需
  定点回补 NumPy，不从头重学。
- POSIX 系统编程放在 C/C++ 指针、生命周期和编译链接之后。
- Docker 放在已有可运行 CLI 之后，避免只练孤立的 `hello-world`。
- ONNX 先做简单导出和数值对齐；TensorRT engine、复杂动态图和优化 profile
  留到阶段五。
- Pandas/Matplotlib 在第一次产生真实 benchmark 数据后即时学习，不阻塞
  阶段一核心链路。

## 现有课程覆盖审计与定位

核验日期为 2026-09-04。

| 现有资源 | 真实定位 | 阶段一处理 |
|---|---|---|
| [Bro Code `Python Full Course for Free 2024` B站 AI 翻译版](https://www.bilibili.com/video/BV1wr5T6bEN4/) | 77 P、约 12 小时的通用 Python 入门；不是 freeCodeCamp | 修正来源；不通刷，只作 Python 缺口选段 |
| [林粒粒 3 小时 Python](https://www.bilibili.com/video/BV1Jgf6YvE8e/) | 语法、类、文件、异常和 `unittest` 入门 | 当前学习者已完成；作为 Python 基础，不再重复通刷 Bro Code |
| [尚硅谷 Python 数据分析课](https://www.bilibili.com/video/BV1D9GLzyEL6/) | NumPy、Pandas、Matplotlib 和通用数据分析 | NumPy P6-P21 定向使用；Pandas/Matplotlib 延迟触发 |
| [mrdbourke 一天学会 PyTorch B站版](https://www.bilibili.com/video/BV1kx4y1r79d/) | 93 P、约 25.5 小时，只覆盖官方课程 00-04 | 只选 Tensor、Module、推理、保存加载和 DataLoader 相关块 |
| [炮哥 PyTorch/CNN 课](https://www.bilibili.com/video/BV1e34y1M7wR/) | FC/CNN、经典网络和训练实践 | 从阶段一主课降为条件回补；模型结构内容延后阶段五 |
| [刘二 PyTorch 实践 13 讲](https://www.bilibili.com/video/BV1Y7411d7Ys/) | 回归、反传、DataLoader、CNN、RNN | 只保留 DataLoader 中文回补；训练原理延后阶段三 |
| [freeCodeCamp 31 小时 C++](https://www.youtube.com/watch?v=8jLOx1hD3_o) | C++20 语法、指针、引用、函数、模板、类和 OOP | 只承担语言基础；C++20 Concepts 和深 OOP 按当前目标跳过 |
| [MIT Missing Semester 2026](https://missing.csail.mit.edu/2026/) | Shell、环境、调试、Git、打包和代码质量 | Shell/Git/调试核心必修；当前学习者已完成，改做实验验收 |
| [尚硅谷 Linux 应用层开发](https://www.bilibili.com/video/BV1DJ4m1M77z/) | 130 P、约 25 小时的 C/POSIX 系统编程 | 按 fd、进程、线程、socket 专题选段，不通刷 |
| [freeCodeCamp/KodeKloud Docker 2019](https://www.youtube.com/watch?v=fqMOX6JJhGo) | Docker 基础到 Swarm/Kubernetes 的旧版概览 | 降为旧版概念参考，不作为当前命令规范 |
| [尚硅谷 Docker 3 小时速通 2024](https://www.bilibili.com/video/BV1Zn4y1X7AZ/) | 25 P，含容器、存储、网络、Compose、Dockerfile 和中间件案例 | 中文选段；Compose 延后，业务案例跳过 |

## 模块一：Shell、Git 与开发环境

### 目标

能够在 Ubuntu 主机和远程 ARM64 设备上导航、搜索、组合命令、管理任务、检查
身份权限、阅读日志并保留可复现的命令记录。

### 课程与取舍

- 主线必修：Missing Semester 2026 的 Shell、Command-line Environment、
  Debugging and Profiling、Version Control、Packaging and Shipping Code、Code
  Quality。
- 条件回补：Development Environment and Tools，仅在编辑器、SSH 或构建工具链
  不熟时回看。
- 明确跳过：Agentic Coding、Beyond the Code 不作为 Linux 基础门槛。
- 已完成可验收：当前学习日志已记录 9 讲完成，不要求重看。

### 专题必修

- 引号、变量展开、glob、管道、标准输入/输出/错误、退出码、`&&`/`||`；
- `set -euo pipefail` 的作用和边界，ShellCheck；
- `rg`、`find`、`jq`、`ps`、`pgrep`、`ss`、`lsof`、`journalctl`、`strace`；
- SSH、任务控制和长任务会话；
- Git 的 status、diff、log、branch、commit 和可恢复撤销。

### 最小实验与停止条件

- 写一个 Shell telemetry 脚本，连续采集 60 次 CPU、内存、目标进程状态和可用
  温度节点，输出 JSONL 或 CSV；
- 目标进程不存在、温度节点缺失和权限不足时输出可诊断错误；
- ShellCheck 通过；路径和变量正确引用；
- 能用 `pgrep`、`ss`、`journalctl`、`strace` 分别定位一次人为构造的问题。

## 模块二：Python 基础与工程化

### 前置与去重

Python 基础课二选一，不双刷。当前学习者已经完成林粒粒 3 小时课程，因此
Bro Code 只回补下列选段：

- P21 `lists, sets, and tuples`；
- P34-P35 `args & kwargs`、`iterables`；
- P39-P41 `modules`、`scope resolution`、`if __name__ == "__main__"`；
- P59-P61 `file detection`、`writing files`、`reading files`。

小游戏、银行/老虎机/加密/吊死鬼、闹钟、API 天气应用和 P66-P77 PyQt GUI
全部跳过。多线程仅在需要理解 Python 客户端并发时条件回补，不作为阶段一
Python 门槛。

### 专题必修

主课未覆盖的内容按项目顺序补齐：

1. `pathlib`、JSON/CSV、`hashlib` 和文件 manifest；
2. `dataclass`、类型标注、模块/包与 `pyproject.toml`；
3. `uv init/add/sync/run/tree/lock`、平台约束和依赖复现；
4. `argparse`、环境变量、约定退出码；
5. `logging` 和结构化日志；
6. `subprocess.run` 的参数列表、退出码、timeout、stdout/stderr，禁止默认
   `shell=True`；
7. `time.perf_counter`、随机种子和 benchmark 元数据；
8. pytest fixture、参数化、异常和数值断言。

官方 Python、PyPA、uv 和 pytest 文档作为这些专题的执行规范，不额外增加一门
综合 Python 视频课。

### 最小实验与停止条件

- 实现 `model-tools manifest`：扫描图片/视频，输出排序稳定、含 hash 的 manifest；
- 正常目录、空目录、错误路径和不可读文件具有稳定退出码；
- `uv sync --locked` 后测试通过，依赖树无冲突；
- 日志包含工具版本、输入路径、参数、随机种子和失败样本；
- 至少 5 个 pytest 覆盖正常、边界和错误路径。

## 模块三：NumPy、图像数组与数值验证

### 课程与取舍

- 主线必修：尚硅谷数据分析课 P6-P21 NumPy，重点 P7-P18，完成 P20-P21
  练习。
- 条件回补：如果能直接完成本模块实验，可用官方 NumPy fundamentals 查漏，
  不要求完整看课。
- 延后阶段：Pandas、Matplotlib 和房价数据分析项目不随 NumPy 顺序通刷。

### 专题必修

视频宽泛章节不足以替代以下专题：

- `shape`、`axis`、`dtype`、casting 和溢出；
- 基础/高级索引，view 与 copy，所有权与别名；
- reshape、transpose、stack/concatenate、broadcast；
- `strides`、C-contiguous、`ascontiguousarray`；
- `asarray`、`frombuffer` 及可能的零拷贝/复制边界；
- `matmul`、非方阵和 batched GEMM；
- `default_rng` 与固定随机种子；
- `allclose`、`assert_allclose`、`rtol`、`atol`；
- `.npy`/`.npz` 与原始二进制数组的适用边界。

### 最小实验与停止条件

- 将 OpenCV `uint8` HWC/BGR 图片转换为 contiguous `float32` NCHW/RGB batch；
- 对每一步断言 shape、dtype、stride、数值范围和是否共享内存；
- 实现 NumPy CPU GEMM，覆盖非方阵、batch 和边界 shape；
- 与 PyTorch CPU reference 使用显式容差对齐；
- 能解释广播何时不复制、何时会产生大临时数组。

## 模块四：PyTorch Tensor、模型资产与推理

### 课程选段

mrdbourke B站版按连续能力块选学：

- P15-P28：Tensor 创建、属性、运算、矩阵乘、reshape/permute、索引、NumPy
  互操作、随机性、GPU 与 device-agnostic code；
- P33-P36：`nn.Module`、模型内部、预测与推理入口；
- P42：保存和加载模型；P37-P41 仅在无法理解 checkpoint 如何产生时回补；
- P64-P65：mini-batch 与 DataLoader；
- P77、P80-P84：自定义图片数据到 Tensor、Dataset 和 DataLoader。

下列内容不在阶段一顺序通刷：

- P44-P58 分类训练与调参延后阶段三；
- CNN 结构与训练延后阶段五；
- 数据增强、过拟合和完整训练项目按后续项目需要回补；
- 炮哥课程的经典 CNN 主体延后阶段五；
- 刘二课程只保留第 8 讲作为 DataLoader 中文回补，其余延后阶段三/五。

### 专题必修

- Tensor view、storage、stride、storage offset、contiguous 与 memory format；
- `torch.layout`、维度语义和 NCHW/NHWC 的区别；
- `model.eval()`、`no_grad()`、`inference_mode()` 的正交关系；
- `state_dict`、`weights_only=True`、`map_location`、strict load 和跨设备加载；
- Dataset/DataLoader 的 batch、shuffle、`num_workers`、`pin_memory` 和
  `non_blocking`，性能结论必须实测；
- CUDA 计时的 warmup、异步执行和同步边界，只建立正确性意识，深入分析留阶段二。

### 最小实验与停止条件

- `permute` 后证明 Tensor 非 contiguous 且共享 storage，调用 `contiguous()` 后
  再 `view`；
- 使用含 Dropout/BatchNorm 的小模型证明 `eval()` 与 inference mode 解决的是
  不同问题；
- state dict 从 CPU 保存并在 CPU/CUDA 加载，strict load 无 missing/unexpected
  keys，同输入输出通过 `torch.testing.assert_close`；
- 自定义 Dataset/DataLoader 的样本总数、batch shape/dtype 正确；比较
  `num_workers=0/2` 和 pinning，只记录结果，不预设收益；
- benchmark 至少分离首次加载和 warmup，并输出 p50/p95、设备和软件版本。

## 模块五：ONNX 与 ONNX Runtime 最小闭环

### 定位

ONNX 不由现有三门 PyTorch 视频负责，必须成为独立专题。阶段一只建立“可导出、
可检查、可执行、可对齐”的最小闭环；陌生模型、复杂动态 shape、不支持算子、
图修改和 TensorRT profile 留阶段五。

### 专题必修

1. ONNX graph、node、tensor、initializer、input/output、opset 和 shape 概念；
2. 当前 PyTorch `torch.onnx.export` 推荐路径，实际参数以安装版本文档为准；
3. `onnx.checker`、模型输入输出契约和 Netron 阅读；
4. ONNX Runtime provider 显式选择与 PyTorch/ORT 输出对齐；
5. 简单 dynamic batch；动态空间维、控制流和自定义算子延后阶段五。

### 最小实验与停止条件

- 导出一个小型公开模型，记录 PyTorch、ONNX、opset/IR、provider 和平台版本；
- `onnx.checker.check_model(..., full_check=True)` 通过；
- 固定 shape 和至少两个未参与导出的合法 batch 均能运行；
- 所有输出使用明确 `rtol/atol` 对齐并记录最大绝对/相对误差；
- 错误 dtype、rank 或 shape 有负例测试和可诊断错误。

## 模块六：现代 C++17、CMake 与调试

### 课程与取舍

freeCodeCamp 31 小时视频只承担语言基础。按个人诊断结果选择：

- 基础薄弱时学习变量、控制流、数组、指针、引用、函数和字符串；
- 主线选学指针/引用、函数参数与返回、lambda、函数模板、类、构造/析构和
  多文件编译；
- C++20 Concepts、深继承层次和完整多态章节不作为阶段一主线；
- 所有阶段项目统一以 `-std=c++17` 构建。

### 专题必修

- 对象生命周期、所有权、RAII、Rule of Zero/Five 和移动语义；
- `unique_ptr` 优先，理解 `shared_ptr/weak_ptr` 的使用边界；
- `vector`、`array`、`string`、`string_view`、map/unordered_map 的基本选型；
- 迭代器、`algorithm`、`chrono`、`filesystem`；
- const correctness、异常与状态码的边界；
- 编译、汇编、链接，静态/动态库，符号、ABI 与 C ABI；
- target-based CMake：library/executable、PUBLIC/PRIVATE/INTERFACE、
  `cxx_std_17`、Debug/Release、CTest、`find_package` 和 Presets；
- GDB 的 breakpoint、run、next/step、print、bt 和 core；
- ASan/UBSan；并发强化实验再使用 TSan。

LearnCpp、CMake 官方教程、GDB 和 Sanitizer 官方文档作为专题来源，不要求再看
一门完整 C++ 长课。

### 最小实验与停止条件

- CPU GEMM 拆成 library、test、benchmark 三个 CMake target；
- fresh configure/build 的 Debug 和 Release 均成功；
- CTest 覆盖空/最小输入、非方阵和固定随机输入，误差阈值明确；
- ASan/UBSan 零报告；
- warmup 后重复计时，输出编译器、flags、shape、median/p95 JSON；
- 人为加入越界并用 GDB backtrace 定位，修复后增加回归测试；
- 人为制造一次缺符号或缺 `.so`，使用 `nm -C`、`readelf` 或 `ldd` 给出证据。

## 模块七：Linux 用户态与 POSIX 系统编程

### 前置

先完成 C/C++ 指针、生命周期、编译链接和基础调试，再进入本模块。Missing
Semester 负责工具使用，尚硅谷课程负责 POSIX API，二者不是重复课程。

### 核心选段

- P9-P14：GCC 编译流程与链接，和 C++/CMake 模块互相验证；
- P29-P33：`open`、系统调用和文件描述符；
- P35-P40：fork、exec、wait、进程树和孤儿进程；
- P42、P44、P54：errno、匿名管道和信号；
- P55-P56、P62-P63、P68-P69：线程、竞态、mutex 和 condition variable；
- P83-P84、P87-P88、P91-P93：虚拟内存、fork/exec、系统调用与进程/线程边界，
  只学稳定概念。

### 系统/框架岗位强化选段

- P76-P77：线程池；
- P98-P117：TCP、socket 和多连接；
- P128-P129：I/O 多路复用。

Windows/VS、Makefile 细枝末节、`system()`、异步线程取消、spinlock、完整 SysV
IPC、信号量全套、UDP、链路层和手写 daemon 不作为阶段一主线。

### 专题必修

课程 API 示例还需用 man-pages 补齐：partial read/write、EOF、`EINTR`、
`EAGAIN`、nonblocking、CLOEXEC、TCP 无消息边界、长度前缀、backpressure、
SIGPIPE、SIGTERM 和 epoll LT/ET。

### 停止条件

核心门槛：

- 能解释 fd、进程、线程、虚拟内存、系统调用与库函数边界；
- 能用 `gdb`、`strace`、`lsof`、`ss` 和 `/proc` 定位一次文件、进程或 socket
  问题；
- Shell telemetry 脚本通过异常场景测试。

系统/框架岗位强化门槛：

- 实现长度前缀 TCP 协议和有界工作队列；
- 覆盖拆包/粘包、partial I/O、断连、超长输入和背压；
- 并发压测无错误、无 fd 增长，SIGTERM 可停止接收并清理连接；
- ASan/UBSan/TSan 零报告。

并发 TCP 服务不再阻塞只走端侧部署主线的学习者，但仍是推理框架/C++ runtime
岗位的阶段一强化证据。

## 模块八：Docker 单容器与架构边界

### 课程与取舍

- 中文主选：尚硅谷 `Docker教程，3小时速通，docker部署到实战！` 的 P2、
  P6-P8、P10、P12-P14、P16、P20-P21；
- 2019 freeCodeCamp/KodeKloud 课程只作概念或英文回补，不执行其中旧安装、
  legacy link、旧 Compose 或 Python 2 示例；
- 中文课中的云服务器购买、Redis 主从、WordPress、Java 中间件和批量中间件
  部署跳过；`docker commit` 只认识，不作为可复现构建方式；
- Compose、服务网络/发现、healthcheck、multi-stage、Buildx、多平台发布、GPU
  runtime 和资源限制延后阶段六。

### 专题必修

- image、container、registry、tag、digest 和 layer 的边界；
- `pull/run/ps/exec/logs/inspect/stop/rm/build`；
- env、bind mount、named volume 和基础端口映射；
- Dockerfile 的 FROM/WORKDIR/COPY/RUN/ENV/USER/ENTRYPOINT/CMD；
- exec form、非 root USER、UID/GID 与挂载权限；
- build context、`.dockerignore`、cache 顺序和固定基础镜像 digest；
- OCI `linux/amd64`/`linux/arm64` 与 Jetson GPU 兼容是不同层级；
- 容器不封装宿主 kernel、BSP、firmware 或内核态 GPU driver。

### 阶段一与阶段六边界

阶段一只封装 CPU/headless `model-tools`。Host 必须完成 amd64 单容器复现，并
检查所选基础镜像的 arm64 manifest。若有 Orin，则增加原生 arm64 CPU smoke
test；QEMU 结果只能标记为仿真，不能证明 Jetson GPU 可用。

阶段六再核验 JetPack/L4T、NVIDIA Container Toolkit、GPU device/library
注入、NGC/Triton tag、TensorRT engine 和服务运行时。

### 停止条件

- fresh clone 后按记录命令构建成功，基础镜像和 Python 依赖精确固定；
- 容器以非 root UID 运行；模型/输入只读挂载，输出单独挂载；
- `docker run --rm --network none` 仍可执行固定 CPU/ORT 推理；
- 宿主和容器生成相同 manifest hash，PyTorch/ORT 输出满足明确容差；
- JSON 记录 git SHA、image digest、架构、Python/依赖版本和输入/模型 hash；
- 在声明支持的第二个平台复现；arm64/QEMU/Orin 证据类型明确区分。

## 按需性能报告补课

Pandas 和 Matplotlib 不作为阶段一前置课程。第一次产生真实 benchmark CSV/JSON
后再触发：

- Pandas：数据分析课 P34-P38、P43、P49；按需补 P45-P46；
- Matplotlib：P54、P55、P57-P59、P68；
- 必补 API：`read_csv/read_json`、筛选、`groupby/agg`、`quantile`、concat、
  `plot/scatter/hist/boxplot/subplots`、轴单位、legend、log scale 和 `savefig`。

触发实验为：读取 benchmark 数据，按 precision/batch/concurrency 分组，生成
p50/p95/p99、throughput、memory、power、temperature 汇总表和曲线。它最迟在
阶段二性能报告前完成，但不要求先看完整数据分析课。

## 项目结构调整

阶段一交付拆成相互独立但可汇合的四项：

1. `model-tools`：manifest、preprocess、export、validate、benchmark/report；
2. `cpp-gemm`：C++17/CMake library、tests、benchmark 和调试证据；
3. `telemetry`：Shell 采集工具及异常场景测试；
4. `container`：只封装 `model-tools`，不把 C++ GEMM、监控或 TCP 服务强行装入
   同一镜像。

并发 TCP 服务改为系统/框架岗位强化交付，不再是所有方向进入阶段二的硬门槛。
它仍保留在一页式清单的岗位强化区。

`model-tools` 至少提供以下子命令或等价入口：

```text
manifest   -> 扫描输入并产生稳定 hash 清单
preprocess -> 统一颜色、shape、dtype、layout 和 batch
export     -> PyTorch 模型导出 ONNX
validate   -> PyTorch 与 ORT 数值对齐
benchmark  -> 正确 warmup/计时并输出原始 JSONL/CSV
report     -> 生成最小汇总；复杂画图按需补课后加入
```

## 阶段一总门槛

进入阶段二前必须满足：

- Python/uv 项目可从 clean environment 重建，测试和依赖检查通过；
- NumPy 图像布局、GEMM 和 PyTorch reference 通过显式容差；
- PyTorch 模型能正确进入推理状态、保存/加载并处理固定 Dataset/DataLoader；
- ONNX checker、ORT 执行和多 batch 数值对齐通过；
- C++ GEMM 可 fresh build、CTest 通过、Sanitizer 干净并有 benchmark 原始数据；
- Shell telemetry 能采样并处理缺失资源；
- CPU/headless Docker 镜像能够在声明平台复现；
- 报告明确区分 Host、QEMU 和 Orin 证据，不声称未实测的 GPU/ARM64 结果。

课程已看但上述证据未完成，不能毕业；未完整看课但证据完整，可以跳过对应
基础章节。

## 版本和平台纪律

- “概念无强版本绑定”与“项目必须锁版本”分开陈述；
- Python、PyTorch、ONNX exporter、ORT 和 Docker 命令以执行日官方文档为准；
- 项目记录 Python/OS/架构、直接与传递依赖、模型 hash、opset、provider；
- 不把 x86_64 venv、wheel 或镜像直接复制到 arm64；
- `linux/arm64` 只证明 CPU 架构，不证明 Jetson CUDA/TensorRT/摄像头支持；
- Jetson GPU 容器和 TensorRT engine 仍遵循阶段六与阶段五的目标平台门槛。

## 路线图修改范围

实施时修改：

1. 阶段一的目标、课程、课程覆盖矩阵、能力模块、选段、补课、停止条件和项目；
2. 阶段一的版本/平台边界；
3. 一页式执行清单中的阶段一基础与岗位强化项；
4. 明确排除项中与阶段一课程取舍冲突的表述；
5. 关键链接索引中的新增官方专题和修正后的课程名称；
6. 必要的阶段二、五、六交接引用，但不重写这些阶段主体。

不修改学习日志中的既有完成记录，不回退路线图当前未提交改动。

## 专题资料基线

实施路线图时优先引用以下免费原始资料；视频负责建立连续认识，官方文档负责
当前语义和可执行规范。

### Python、NumPy 与报告

- [Python `dataclasses`](https://docs.python.org/3/library/dataclasses.html)
- [Python `argparse`](https://docs.python.org/3/library/argparse.html)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Python `subprocess`](https://docs.python.org/3/library/subprocess.html)
- [Python Packaging User Guide](https://packaging.python.org/)
- [uv projects](https://docs.astral.sh/uv/guides/projects/)
- [pytest Get Started](https://docs.pytest.org/en/stable/getting-started.html)
- [NumPy fundamentals](https://numpy.org/doc/stable/user/basics)
- [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy `ndarray.strides`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
- [NumPy `assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html)
- [Pandas GroupBy](https://pandas.pydata.org/docs/reference/groupby.html)
- [Matplotlib Quick start](https://matplotlib.org/stable/users/explain/quick_start.html)

### PyTorch 与 ONNX

- [mrdbourke 官方课程仓库](https://github.com/mrdbourke/pytorch-deep-learning)
- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [PyTorch ONNX tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [当前 `torch.onnx` API](https://docs.pytorch.org/docs/stable/onnx.html)

### C++、构建与 Linux

- [LearnCpp](https://www.learncpp.com/)
- [C++ Core Guidelines：Resource management](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-resource)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [GDB manual](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)
- [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- [Linux `read(2)`](https://man7.org/linux/man-pages/man2/read.2.html)
- [Linux `epoll(7)`](https://man7.org/linux/man-pages/man7/epoll.7.html)
- [Linux dynamic linker `ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html)
- [OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/)

### Docker 与平台边界

- [Docker Get Started](https://docs.docker.com/get-started/)
- [Dockerfile best practices](https://docs.docker.com/build/building/best-practices/)
- [Docker Build checks](https://docs.docker.com/reference/build-checks/)
- [Docker multi-platform builds](https://docs.docker.com/build/building/multi-platform/)
- [NVIDIA Container Toolkit architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

## 验证

实施完成后执行：

1. 搜索 `freeCodeCamp`，确认 Bro Code Python 课程不再错误归属；
2. 搜索阶段一所有“主课大纲”表述，确认课程真实目录与能力目标已经分离；
3. 确认 NumPy 和 ONNX 均有独立必修来源、实验和停止条件；
4. 确认 CMake、GDB、Sanitizer、RAII/STL/智能指针不再被宣称由 31 小时视频
   完整覆盖；
5. 确认 Missing Semester 与 POSIX Linux 课程的职责和顺序不同；
6. 确认 PyTorch 分类/CNN 训练没有同时在阶段一、三、五重复设为必修；
7. 确认 Pandas/Matplotlib 是数据产生后的条件补课；
8. 确认 Docker Compose、multi-stage、Buildx 和 GPU runtime 仍归阶段六；
9. 确认 `linux/arm64` 和 Jetson GPU 兼容没有混为一谈；
10. 确认并发 TCP 服务只在系统/框架岗位强化区作为门槛；
11. 核查所有新增链接可访问，B站选段同时记录 P 编号和标题；
12. 运行 `git diff --check`，审阅最终 diff，确保未修改无关内容或用户现有改动。
