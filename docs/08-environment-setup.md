# Jetson Orin NX 学习环境安装

核验日期：2026-08-19。

目标组合：Ubuntu 24.04 x86_64 Host + Jetson Orin NX 16GB + SDK Manager 中的 JetPack 7.2.1 Direct Flash。官方 JetPack 7.2 发布基线为 Jetson Linux 39.2、CUDA 13.2.1、TensorRT 10.16.2；实际 patch 版本必须刷机后盘点。

## 先读：主机和目标机不要混淆

本文命令均用以下标签：

- **Host**：运行 SDK Manager 的 Ubuntu x86_64 开发机。
- **Orin**：被刷机的 Jetson Orin NX，架构应为 aarch64/arm64。
- **Container**：运行在 Orin 上、与 JetPack 匹配的 arm64 容器。

不能把 Host 的 x86_64 `.deb`、wheel、TensorRT engine 或 Docker image 直接复制到 Orin 运行。文件名中的 `amd64`/`x86_64` 与 `arm64`/`aarch64` 必须匹配执行机器。

## 刷机风险

Direct Flash 会重写目标存储，可能清除原有系统和数据。开始前：

1. 确认 SDK Manager 显示的 target、module、carrier board 和目标存储正确。
2. 备份 Orin 上需要保留的数据、SSH key、模型、容器 volume 和配置。
3. 拔掉不相关的可移动磁盘，降低选错目标的风险。
4. 使用稳定电源和可传数据的 USB 线，不要用只有充电功能的线。
5. 刷机期间不要断电、拔线或让 Host 休眠。

如果使用第三方 carrier board，Force Recovery 引脚、供电和 flash 配置可能不同，先按板卡厂商手册确认；不要只套用 NVIDIA 参考载板步骤。

## 1. 硬件与容量准备

### 必需项

- Jetson Orin NX 16GB module + 兼容 carrier board；
- 符合载板要求的稳定电源；
- 数据能力正常的 USB cable，连接 Host 与 recovery USB 口；
- Host 有稳定网络；
- 首次启动使用显示器/键鼠，或准备串口/官方支持的 headless 设置；
- Orin 安装完系统后最好有有线网络，便于 SDK components 安装与 SSH；
- 主动散热，后续 benchmark 必须保证热条件稳定。

### 空间规划

SDK Manager 官方最低值是每个完整 Jetson SDK 版本：Host 27GB + target 16GB 可用空间。这个值只够安装，不适合模型开发。

建议单独规划：

| 数据 | 位置 | 规划原则 |
| --- | --- | --- |
| SDK Manager 下载/解包缓存 | Host | 至少保留一个完整版本和日志 |
| Git/源码/编译缓存 | Host/Orin | 与模型权重分开 |
| ONNX + TensorRT Edge-LLM engines | Orin NVMe | 官方提示单项目约需 20-50GB |
| Hugging Face/模型缓存 | Host 或 Orin NVMe | 用固定目录和 manifest，不散落在 home |
| 容器 image/layer | Orin NVMe | 定期盘点，不在 benchmark 前临时清理 |
| 原始 benchmark/trace | Host | trace 可能很大，实验后归档 |

如果根文件系统写入 NVMe，先确认 SDK Manager 的目标存储和载板启动支持。官方 Quick Start 要求 Orin NX/Nano 的 SD/USB rootfs 介质至少 64GB；模型项目更适合更大的 NVMe。

### SDK 安装目录注意事项

截图中的 `SDKs install folder` 位于 `/media/barry/.../JetsonPack`。这个目录是 **Host 下载、解包和刷机工作目录**，不是 Orin 安装后的运行目录。

外置盘建议：

- 使用本地 Linux 文件系统（优先 ext4）；
- 支持符号链接、Unix 权限、可执行位和足够 inode；
- 保证挂载点在重启后稳定；
- 路径尽量简单，避免网络盘、FUSE、exFAT/NTFS 权限语义导致脚本失败；
- 用 `df -Th` 确认文件系统和剩余空间。

**Host：**

```bash
uname -m
lsb_release -a
free -h
df -Th /media/barry/Qin专属备用盘/JetsonPack
```

期望 Host 架构是 `x86_64`，系统是受该 JetPack/SDK Manager 组合支持的 Ubuntu 版本，路径空间充足。

## 2. Host 基础检查

SDK Manager 自身支持 Ubuntu Desktop 24.04，但具体 JetPack 仍以 SDK Manager compatibility matrix 为准。

**Host：**

```bash
uname -a
uname -m
lsb_release -ds
lsusb
ip -brief address
df -h
```

检查：

- Host 不在虚拟机中，或已确认 USB passthrough 在 recovery 期间稳定；
- 防火墙/代理没有阻断 NVIDIA 下载；
- 当前用户能使用 `sudo`；
- 没有另一个 SDK Manager 或 flash 进程占用 target；
- 安装目录可写，系统盘和下载盘都有余量。

## 3. 进入 Force Recovery

Recovery 模式由 carrier board 决定。NVIDIA Orin Nano Developer Kit carrier board 常用 REC 与 GND jumper：断电，短接 REC/GND，连接 Host USB 后重新上电；第三方载板按其手册操作。

进入 recovery 后只在 Host 检查 USB ID。

**Host：**

```bash
lsusb | grep -i '0955:'
```

Orin NX 16GB P3767-0000 的官方 recovery USB product ID 是 `0955:7323`。看到 NVIDIA 设备只表示 recovery 枚举成功，不表示 target 存储和载板配置一定正确。

看不到设备时依次检查：

1. USB 口是否为载板 recovery/device 口；
2. USB 线是否支持数据；
3. 是否按正确顺序进入 Force Recovery；
4. `dmesg --follow` 是否出现反复断连；
5. 关闭 USB hub，直接连接 Host；
6. 根据载板手册核对 jumper/button；
7. 换线、换 Host USB 口后重试。

## 4. SDK Manager 选择与刷机

截图中的选择可以作为目标基线：

- Product Category：Jetson；
- Target Hardware：Jetson Orin NX modules；
- Target OS：JetPack 7.2.1；
- Host Machine：Ubuntu 24.04 x86_64；
- SDK：JetPack 7.2.1；
- Flash：Direct Flash；
- Additional SDK：按需选择 DeepStream 9.1；Video Codec 若 SDK Manager 明确显示对该 target 不可用则不要强选；Holoscan 不属于本项目 P0。

### SDK Manager 的两个安装阶段

1. **Flash Jetson Linux**：写入 bootloader/BSP/rootfs，需要 Force Recovery，具有破坏性。
2. **Install Jetson Runtime Components**：target 首次启动、创建用户并联网后，通过 SSH/包管理安装 CUDA、TensorRT 等组件。

因此刷完 BSP 后 SDK Manager 可能要求：

- 完成 Ubuntu first boot 和用户创建；
- Orin 与 Host 网络可达；
- 输入 Orin 用户名、密码和 IP；
- 继续 target component installation。

不要在进度刚到 “Flash completed” 时就认为 CUDA/TensorRT 已全部安装。

### 下载阶段可暂停，Flash 阶段不要中断

截图中的 `Pause for a bit` 适合在纯下载阶段暂停。进入 flash、partition、bootloader 或 target package transaction 后不应主动暂停或断电。

### 保存日志

刷机结束后保存：

- SDK Manager summary；
- terminal 日志；
- 下载 manifest；
- target hardware/存储选择；
- 成功和失败 component 列表。

这些信息是后续定位“系统能启动但 CUDA/DeepStream 缺失”的第一证据。

## 5. 首次启动与网络

在 Orin 完成 Ubuntu first boot，创建普通用户并设置 SSH。

**Orin：**

```bash
hostnamectl
uname -m
ip -brief address
df -hT
```

期望 `uname -m` 为 `aarch64`。先让 SDK Manager 完成 target components，再进行大规模 `apt upgrade` 或第三方包安装，避免改变基线。

**Host：**

```bash
ssh <orin-user>@<orin-ip> 'uname -m; hostname; uptime'
```

## 6. 刷机后只读版本盘点

下面命令不会安装或删除软件。可逐条执行，缺少可选组件时允许 `command not found`，但必须记录。

### OS、L4T、kernel、架构

**Orin：**

```bash
uname -a
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 2>/dev/null || true
```

### CUDA

**Orin：**

```bash
command -v nvcc || true
nvcc --version 2>/dev/null || true
ls -ld /usr/local/cuda* 2>/dev/null || true
dpkg-query -W 'cuda-*' 2>/dev/null | head -n 30
```

Jetson 通常没有桌面 dGPU 环境中完整可用的 `nvidia-smi`；不要以 `nvidia-smi` 缺失判断 GPU 不可用。

### TensorRT/cuDNN

**Orin：**

```bash
dpkg-query -W 'tensorrt*' 'libnvinfer*' 'libcudnn*' 2>/dev/null
ldconfig -p | grep -E 'libnvinfer|libcudnn' | head
command -v trtexec || find /usr/src/tensorrt -type f -name trtexec 2>/dev/null
python3 -c 'import tensorrt as trt; print(trt.__version__)' 2>/dev/null || true
```

Python import 失败不必然意味着 C++ runtime 缺失；先查 `.deb` 和动态库，再查 Python binding 的版本/解释器。

### DeepStream/GStreamer

仅在 SDK Manager 选择了 DeepStream 或手动安装后执行。

**Orin：**

```bash
deepstream-app --version-all 2>/dev/null || true
gst-launch-1.0 --version
gst-inspect-1.0 nvinfer 2>/dev/null | head -n 20
ls -ld /opt/nvidia/deepstream/deepstream-* 2>/dev/null || true
```

### Docker/NVIDIA Container Runtime

**Orin：**

```bash
docker --version 2>/dev/null || true
dpkg-query -W 'nvidia-container*' 2>/dev/null || true
docker info 2>/dev/null | grep -E 'Architecture|Runtimes|Default Runtime' || true
```

### 设备、功耗和 telemetry

**Orin：**

```bash
ls /dev/nvhost* 2>/dev/null | head
nvpmodel -q --verbose 2>/dev/null || true
tegrastats --interval 1000
```

`tegrastats` 会持续运行，用 `Ctrl+C` 停止。它是 Orin 系统资源/温度观察的基础，但不能替代 Nsight 的 GPU 时间线和 kernel 指标。

## 7. 环境轨道 A：原生 JetPack 基线

原生环境负责验证硬件和 SDK 本身。不要先堆叠多个 Python 环境或第三方容器。

### 系统组件缺失时

如果 SDK Manager 的 target component 安装失败，优先在 SDK Manager 中 retry/repair。只有官方文档确认 apt 源已正确配置时，才考虑安装 JetPack meta package。

**Orin（按需，不要在完整安装已存在时重复执行）：**

```bash
apt-cache policy nvidia-jetpack
```

先检查候选版本是否与 L4T 匹配，再决定是否安装；不要复制其他 JetPack 版本的 apt source。

### CUDA smoke test

先定位 JetPack 自带 samples 或使用 [cuda-samples](https://github.com/NVIDIA/cuda-samples) 的固定 tag/commit。第一批只跑：

- `deviceQuery`：设备和 compute capability；
- `bandwidthTest`：作为环境 sanity check，不等于业务带宽；
- `simpleStreams`/async API 样例：确认 stream 概念。

保存源码 revision、编译器输出和运行结果。

### TensorRT smoke test

准备一个小型、已知可用的 ONNX 后：

**Orin：**

```bash
trtexec --help
trtexec --onnx=<model.onnx> --saveEngine=<model-fp16.engine> --fp16
trtexec --loadEngine=<model-fp16.engine> --warmUp=1000 --duration=10
```

如果 `trtexec` 不在 `PATH`，使用盘点阶段找到的 JetPack 路径。第一次实验记录完整输出，不只截取 Throughput 一行。

### DeepStream 9.1

如果 SDK Manager 已安装 DeepStream，不再重复装。否则官方提供四种路径：SDK Manager、Jetson tar、Jetson arm64 `.deb`、容器。

原生 `.deb` 路径必须从 [NVIDIA/DeepStream v9.1 release](https://github.com/NVIDIA/DeepStream/releases/tag/v9.1.0) 获取 `arm64` 文件：

**Orin（仅在确认未安装且版本匹配时）：**

```bash
sudo apt-get install ./deepstream-9.1_9.1.0-1_arm64.deb
deepstream-app --version-all
```

不要在 Jetson 上安装 `_amd64.deb`。DeepStream 9.1 的 Python `pyds` 已弃用，新项目优先 C/C++ 或 `pyservicemaker` 路径。

## 8. 环境轨道 B：Orin arm64 容器

容器用于隔离模型依赖和固化实验，不替代 Host BSP/driver/nvidia-container runtime。

### 架构和兼容性规则

- image manifest 必须包含 `linux/arm64`；
- Jetson image/tag 必须与 JetPack/L4T/DeepStream release 匹配；
- `multiarch` 不表示同一个二进制跨架构运行，而是 registry 按平台拉取不同 image；
- 不使用通用 `latest` 做可复现实验；
- 记录 image digest，而不只记录 tag；
- models、results、cache 用明确 bind mount，删除容器不能丢实验数据。

### DeepStream 容器 smoke test

DeepStream 9.1 官方提供：

```bash
docker pull nvcr.io/nvidia/deepstream:9.1-samples-multiarch
docker pull nvcr.io/nvidia/deepstream:9.1-triton-multiarch
```

**Orin：**

```bash
docker image inspect nvcr.io/nvidia/deepstream:9.1-samples-multiarch \
  --format '{{.Architecture}} {{index .RepoDigests 0}}'
```

期望架构为 `arm64`。Jetson DeepStream 9.1 容器官方定位主要是部署；开发应用优先在 Orin 原生环境构建，再将二进制加入镜像，或按 NVIDIA 的 x86 cross-build 说明构建。

### Triton 的推荐起点

Orin 上先使用 DeepStream 9.1 已验证的 `9.1-triton-multiarch` 组合，而不是直接拉取面向数据中心 dGPU 的 `tritonserver:latest`。

先检查容器内：

```bash
uname -m
tritonserver --version
ls /opt/tritonserver/backends
```

确认目标 backend 存在后才创建 model repository。不同 backend 可能受 Jetson build 限制；启动成功不代表模型能加载。

## 9. TensorRT Edge-LLM 环境

### 路径拆分

- checkpoint quantize/export 工具：可在带 NVIDIA GPU 的 x86 Host 容器中运行，也可按官方说明在兼容环境运行；
- C++ runtime 和 TensorRT engine build：JetPack 7.2 Orin 路径在 **Orin 设备上**构建；
- 不把 x86 构建产物当作 Orin runtime；
- 每次使用固定 release tag/commit，不从浮动 `main` 产出长期基线。

### Orin C++ runtime 构建

下面来自当前官方 JetPack 7.2 Orin 安装路径。先在仓库 release 页面选定 tag/commit 并记录，再构建。

**Orin：**

```bash
sudo apt update
sudo apt install -y cmake build-essential git

nvcc --version
dpkg -l | grep tensorrt

git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout <verified-tag-or-commit>
git submodule update --init --recursive

mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=13.2 \
  -DENABLE_CUTE_DSL=ALL
make -j$(nproc)

./examples/llm/llm_build --help
./examples/llm/llm_inference --help
```

如果 16GB 设备编译时内存紧张，先降低并行度，如 `make -j2`，不要通过无限 swap 掩盖内存规划问题。`<verified-tag-or-commit>` 必须替换为实际值，不能原样执行。

### Export/quantization Python 环境

不要将 Edge-LLM tools 安装进系统 Python。每个固定 checkout 建自己的 venv：

**Host 或官方支持的转换环境：**

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout <same-verified-tag-or-commit>
git submodule update --init --recursive

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
tensorrt-edgellm-export --help
```

需要量化、LoRA merge 或 tokenizer 工具时才安装：

```bash
python -m pip install -e '.[tools]'
tensorrt-edgellm-quantize --help
```

`tools` extra 会引入 NVIDIA Model Optimizer 等依赖；不要同时在同一 venv 混入其他 release branch。实验性 server extra 不是第一阶段必需。

### Orin 精度边界

当前官方矩阵只将 FP16、INT8、INT4 列为 Jetson Orin runtime 精度路径，不要选择标为 FP8/FP4/NVFP4 的 checkpoint。模型是否支持还要再次查 supported models 页面。

## 10. Profiling 工具

### Nsight Systems

优先用于完整应用：

**Orin：**

```bash
nsys --version 2>/dev/null || true
```

可以在 Orin CLI 采集 `.nsys-rep`，复制到 Host GUI 查看；也可以按当前 Nsight Systems/JetPack 文档配置远程 target。采集时加入 NVTX 标记，区分排队、预处理、enqueue、后处理和响应。

### Nsight Compute

只对 Systems 已确认的少量热点 kernel 使用：

**Orin：**

```bash
ncu --version 2>/dev/null || true
```

若 target CLI 未安装，使用 JetPack 匹配的 Host/target 工具组合。profile 权限、驱动和工具版本不匹配时先修环境，不要根据空报告优化。

### 社区 telemetry 工具

- [jetson_stats](https://github.com/rbonghi/jetson_stats) / `jtop`：交互查看方便，但属于社区工具；
- [jetson-containers](https://github.com/dusty-nv/jetson-containers)：快速试验和构建 Jetson 容器很有价值，但 tag/support 要与当前 JetPack 核验；
- 官方 `tegrastats` 与原始 benchmark 仍是基础证据。

不要在基线盘点完成前安装社区工具，以免它们修改 Python 或系统依赖。

## 11. Version manifest

每次正式实验创建一个不可变版本清单，至少包括：

```yaml
device:
  module: Jetson Orin NX 16GB
  carrier_board: <model-and-revision>
  storage: <model-capacity-filesystem>
  cooling: <fan-heatsink-ambient>
platform:
  jetpack: <exact>
  l4t: <exact>
  ubuntu: <exact>
  kernel: <exact>
  power_mode: <nvpmodel-output>
runtime:
  cuda: <exact>
  tensorrt: <exact>
  cudnn: <exact>
  deepstream: <exact-or-not-installed>
  container_runtime: <exact>
artifacts:
  container_image: <repo@sha256:digest>
  source_revision: <git-sha>
  model_id: <name-and-revision>
  model_sha256: <sha256>
  onnx_sha256: <sha256>
  engine_sha256: <sha256>
experiment:
  scenario_revision: <git-sha>
  start_time: <iso-8601>
```

`version manifest` 不应手工凭记忆填写。后续实现阶段要提供采集脚本，并将完整命令输出保存为 artifact。

## 12. Clean-room smoke test 顺序

不要同时安装所有框架后才测试。按下面顺序，每步失败就停：

1. OS 可启动、SSH 稳定、磁盘和网络正常；
2. L4T/JetPack/架构盘点；
3. `tegrastats`、功耗模式和温度可读；
4. CUDA `deviceQuery`；
5. TensorRT `trtexec` 小模型 FP16；
6. DeepStream sample pipeline；
7. arm64 container + GPU runtime；
8. Edge-LLM C++ runtime help/small supported model；
9. Triton container health + 单 TensorRT backend model；
10. 才进入多模型并发。

每步保存成功条件和日志，这样出错时只回退一层。

## 13. 常见故障树

| 症状 | 首先检查 | 不要先做 |
| --- | --- | --- |
| Host 看不到 recovery | 线、USB 口、REC 顺序、`lsusb 0955:7323` | 重装 CUDA |
| BSP 成功但 CUDA 缺失 | SDK Manager target component 状态、SSH/网络、apt source | 随便添加桌面 CUDA repo |
| `Exec format error` | `uname -m`、包/image architecture | 重试同一个 x86 二进制 |
| container 无 GPU/设备 | BSP、nvidia-container 包、runtime、镜像是否 Jetson | 在容器里安装任意驱动 |
| TensorRT Python import 失败 | Python version、binding package、`libnvinfer` | 覆盖系统 TensorRT |
| engine deserialize 失败 | TensorRT/JetPack/GPU/Plugin 版本和 engine 来源 | 忽略并复用旧 engine |
| DeepStream plugin 缺失 | 9.1 安装状态、GStreamer registry、arm64 包 | 混装旧 DeepStream |
| Edge-LLM CMake 找不到 TRT | `/usr` 中 headers/libs、JetPack components | 下载 x86 TensorRT tar |
| 编译被 OOM kill | `dmesg`、并行 job、内存和 swap | 反复 `make -j$(nproc)` |
| 磁盘突然耗尽 | Docker layers、HF cache、ONNX/engine、SDK cache | 删除未知系统目录 |
| 并发后延迟暴涨 | queue、CPU、DRAM、GPU streams、热降频 | 立即手写 kernel |
| 长时间性能下降 | 温度、频率、功耗模式、ambient | 只看启动后 10 秒数据 |

## 14. 环境完成标准

环境“安装完成”必须同时满足：

- Host 与 Orin 架构边界清晰；
- SDK Manager 没有未处理的 failed component；
- exact version manifest 已生成；
- CUDA、TensorRT 和 telemetry smoke test 有日志；
- DeepStream/Edge-LLM/Triton 只对选定组件声明完成；
- 容器 image 有 arm64 验证和 digest；
- 模型/ONNX/engine/cache 路径和空间有规划；
- 没有把 x86 主机结果标成 Orin 实测。

完成这些检查后，才开始 [4-5 个月学习路线](06-learning-roadmap.md) 的单模型基线阶段。

