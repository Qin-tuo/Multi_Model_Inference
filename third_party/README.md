# Third-Party Source Index

这里保存知识库中引用的 GitHub 项目本地快照。实际 clone 位于
`third_party/repos/`，该目录被主仓库忽略，避免嵌套 Git 仓库进入提交历史。

## 目录

```text
third_party/
├── README.md
├── manifest.tsv
└── repos/                    # ignored by the parent repository
    ├── p0-hands-on/          # Orin 实操主线
    ├── p1-tooling/           # 量化、压测、容器和监控工具
    ├── p2-source-study/      # Serving/Kernel 源码研读
    └── community/            # 社区课程与知识地图
```

`manifest.tsv` 记录分类、本地目录、GitHub 仓库、默认分支、实际 HEAD、下载方式、
submodule 状态和用途。性能实验仍需在具体实验 manifest 中固定 tag、commit、
JetPack/L4T 和本地 patch。

本次快照于 2026-08-19 完成：19/19 个仓库通过 Git 对象完整性检查，所有
`origin` 均为官方 GitHub 地址，checkout 总计约 2.6 GB。精确 revision 和字节数见
`manifest.tsv`。

## P0 Hands-On

| 本地目录 | 上游 | 用途 |
| --- | --- | --- |
| `cuda-samples` | `NVIDIA/cuda-samples` | CUDA 环境、streams/events/graphs |
| `TensorRT` | `NVIDIA/TensorRT` | TensorRT samples、parser、Plugin |
| `DeepStream` | `NVIDIA/DeepStream` | DeepStream 9.x 主仓库 |
| `deepstream_reference_apps` | `NVIDIA-AI-IOT/deepstream_reference_apps` | 历史 parallel inference 参考 |
| `TensorRT-Edge-LLM` | `NVIDIA/TensorRT-Edge-LLM` | Orin LLM/VLM runtime |
| `triton-server` | `triton-inference-server/server` | Triton Server 核心和文档 |
| `triton-tutorials` | `triton-inference-server/tutorials` | Triton 入门与并发教程 |

## P1 Tooling

| 本地目录 | 上游 | 用途 |
| --- | --- | --- |
| `Model-Optimizer` | `NVIDIA/Model-Optimizer` | 量化与模型优化 |
| `CUDALibrarySamples` | `NVIDIA/CUDALibrarySamples` | CUDA libraries 示例 |
| `perf_analyzer` | `triton-inference-server/perf_analyzer` | 服务压测 |
| `model_analyzer` | `triton-inference-server/model_analyzer` | Triton 配置搜索 |
| `jetson-containers` | `dusty-nv/jetson-containers` | Jetson 容器构建与运行 |
| `jetson_stats` | `rbonghi/jetson_stats` | Jetson 开发期监控 |
| `Lidar_AI_Solution` | `NVIDIA-AI-IOT/Lidar_AI_Solution` | 复杂 CV/CUDA 参考 |

## P2 Source Study

| 本地目录 | 上游 | 用途 |
| --- | --- | --- |
| `vllm` | `vllm-project/vllm` | scheduler、PagedAttention、KV Cache |
| `TensorRT-LLM` | `NVIDIA/TensorRT-LLM` | executor、inflight batching、KV manager |
| `cutlass` | `NVIDIA/cutlass` | GEMM/CuTe/高性能 kernel |
| `sglang` | `sgl-project/sglang` | Serving、scheduler、prefix cache |

## Community

| 本地目录 | 上游 | 用途 |
| --- | --- | --- |
| `lyy-ai_infra` | `lyy-ai/ai_infra` | 中文 AI Infra 课程地图 |

## 下载策略

初次整理采用：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-tags <transfer-url> <local-directory>
git -C <local-directory> remote set-url origin <official-github-url>
```

这会下载当前 checkout 所需的源码，省略完整 Git 历史。由于官方直连当时速度过低，
传输阶段使用了加速端；完成后逐仓恢复官方 `origin`，执行
`git fsck --connectivity-only`，并将本地 HEAD 与官方分支 HEAD 核对。加速端不作为
版本依据。

Git LFS 大文件和 submodules 不会统一展开，避免一次下载与当前学习阶段无关的模型、
数据和第三方源码。包含 submodules 的仓库及数量已经记录在 `manifest.tsv`。

## 使用某个项目之前

```bash
cd third_party/repos/<category>/<repository>
git status --short
git rev-parse HEAD
git submodule status
```

如果该项目的官方构建文档明确需要 submodules，再执行：

```bash
git submodule update --init --recursive --depth 1
```

如果需要完整历史：

```bash
git fetch --unshallow
```

不要对全部 19 个仓库统一执行这两个命令。

## 更新规则

更新前先保存实验 revision。不要直接对所有仓库执行无约束的 `git pull`。推荐逐个：

```bash
git fetch origin <release-tag-or-branch> --depth 1
git checkout --detach <verified-commit>
```

更新 `manifest.tsv` 后重新执行对应项目的构建、正确性和性能回归。

## 重要边界

- `TensorRT`/`DeepStream` 的源码 `main` 可能比 JetPack 自带版本新，使用时切到匹配 release。
- `deepstream_reference_apps` 仅作历史参考，新实现使用 `NVIDIA/DeepStream`。
- `vLLM`、主线 `TensorRT-LLM`、`SGLang` 默认用于服务器设计研读，不是 Orin P0 runtime。
- `lyy-ai/ai_infra` 是社区课程，核验时未声明 license；复制或分发代码前需确认授权。
- 下载成功只表示 Git 对象和 checkout 完成，不表示依赖、submodules、模型或容器已安装。
