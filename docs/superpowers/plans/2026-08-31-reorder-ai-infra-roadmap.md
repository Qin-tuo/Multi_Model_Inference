# Dependency-Ordered AI Infra Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the AI Infra roadmap into eight dependency-driven main stages, move directional topics into prerequisite-gated branches, and remove calendar-based schedules.

**Architecture:** Keep the roadmap as one canonical Markdown document, but update it in independently verifiable logical sections. Preserve valid resource links and version boundaries while changing course ownership, stage cross-references, project requirements, and execution guidance to match the approved design.

**Tech Stack:** Markdown, ripgrep, Git

---

### Task 1: Rewrite the directory, overview, dependency chain, and stage map

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:5-30`
- Reference: `docs/superpowers/specs/2026-08-31-roadmap-stage-reordering-design.md:1`

- [ ] **Step 1: Capture the stale overview as the failing documentation check**

Run:

```bash
rg -n "阶段二 · 模型概念底座|阶段六 · Jetson 与视频推理|阶段七 · 容器与服务化|15 24 周节奏" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: all four stale labels match.

- [ ] **Step 2: Replace the directory and introductory dependency text**

Use this directory sequence:

```markdown
**目录**：1 总览 · 2 个人收藏区 · 3 工程与工具基础 · 4 CUDA 与 GPU 性能 · 5 深度学习基础原理 · 6 模型压缩与高效网络 · 7 模型结构、转换与推理引擎 · 8 容器化与推理服务 · 9 Transformer 与 LLM 概念 · 10 LLM 推理与 Edge-LLM · 11 可选支线 · 12 CUDA 专项执行清单 · 13 项目阶梯 · 14 概念澄清 · 15 阶段执行与复习原则 · 16 验收标准 · 17 版本纪律 · 18 中文资料用法 · 19 排除项 · 20 毕业项目 · 21 执行清单 · 22 能力与岗位 · 23 链接索引
```

Replace the old “阶段二可选前置” paragraph with text that states:

```markdown
阶段一是工程底座，阶段二建立 CUDA 与 GPU 性能能力；阶段三到阶段五依次建立深度学习原理、模型压缩和模型部署闭环；阶段六学习通用推理服务；阶段七紧邻阶段八补齐 Transformer/LLM 概念。主线面向通用 GPU 推理与 Serving，LLM 推理必修；视觉流媒体、深度剪枝、算子 DSL、驱动和编译器按进入条件走支线。
```

Use this exact main dependency chain:

```markdown
**Python / PyTorch / C++ / Linux / Docker 基础 → CUDA 与 GPU 性能 → 深度学习基础原理 → 模型压缩与高效网络 → 模型结构 / ONNX / TensorRT / GPU 预处理 → Docker Serving / HTTP / gRPC / Triton Server → Transformer 与 LLM 概念 → LLM Inference → Orin Edge-LLM**
```

Use this exact branch summary:

```markdown
**视觉流媒体（分支 A）**、**模型压缩实战（分支 B）**、**Triton Language → CUTLASS / CuTe（分支 C）**、**Linux 驱动 / BSP（分支 D）**、**TVM / MLIR（分支 E）**。
```

- [ ] **Step 3: Replace the eight-stage map**

Use this exact table:

```markdown
| 阶段 | 属性 | 核心交付 |
|---|---|---|
| 阶段一 · 工程与工具基础 | 基础必修 | model-tools + C++17/CMake CPU GEMM + 基础开发镜像 |
| 阶段二 · CUDA 与 GPU 性能 | 性能必修 | CUDA Kernel 仓库 + MNIST CUDA + Sanitizer/Nsight 报告 |
| 阶段三 · 深度学习基础原理 | 原理必修 | 神经网络训练、导出与推理流程说明 |
| 阶段四 · 模型压缩与高效网络 | 优化必修 | 量化/剪枝方案与硬件验证设计 |
| 阶段五 · 模型结构、转换与推理引擎 | 部署必修 | PyTorch→ONNX→TensorRT + GPU 预处理 pipeline |
| 阶段六 · 容器化与推理服务 | Serving 必修 | TensorRT backend Triton 服务 + HTTP/unary gRPC client |
| 阶段七 · Transformer 与 LLM 概念 | LLM 前置必修 | Transformer 与预训练/后训练知识图 |
| 阶段八 · LLM 推理与 Edge-LLM | 本机目标必修 | Orin Edge-LLM 服务 + 运行时选型表 |
```

- [ ] **Step 4: Verify the overview**

Run:

```bash
sed -n '1,40p' docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
rg -n "阶段一 · 工程与工具基础|阶段二 · CUDA 与 GPU 性能|阶段七 · Transformer 与 LLM 概念|分支 E" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: the directory, dependency chain, five branches, and all eight new rows agree.

- [ ] **Step 5: Commit the overview**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: reorder AI infra roadmap overview"
```

### Task 2: Rebuild stages one through four

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:87-400`
- Reference: `docs/superpowers/specs/2026-08-31-roadmap-stage-reordering-design.md:22-84`

- [ ] **Step 1: Add Docker basics to stage one**

Rename the heading to:

```markdown
## 3. 阶段一 · 工程与工具基础（Python / PyTorch / C++ / Linux / Docker）
```

Move the existing Docker course links into a new `**Docker 入门：**` resource block. Its required outline is:

```markdown
- ★ image、container 与 registry 的边界
- ★ `docker pull/run/exec/logs/inspect`
- ★ volume、端口、环境变量和基础 Dockerfile
- ★ `linux/arm64` 与 Jetson 基础镜像识别
- Compose、network、healthcheck、multi-stage 和 GPU runtime 留到阶段六
```

Add a stage project requirement that packages the existing PyTorch/ONNX command-line inference tool in a basic ARM64-aware development image. Do not remove the existing Python, PyTorch, C++, Linux, `model-tools`, CPU GEMM, Shell, or TCP service requirements.

- [ ] **Step 2: Replace the old stage-two block with the current CUDA block**

The new heading must be:

```markdown
## 4. 阶段二 · CUDA 与 GPU 性能
```

Move the complete CUDA resources, outlines, learning method, projects A/B, and version boundary from the old stage-three block. Change internal references from “阶段三” to “阶段二”, including the CS149 description in the personal collection. Preserve Modern CUDA C++, freeCodeCamp CUDA, CS149, PMPP, Sanitizer/Nsight, GPU MODE, `sm_87`, and all existing Kernel completion criteria.

- [ ] **Step 3: Replace the old stage-three block with DLS fundamentals**

The new heading and scope are:

```markdown
## 5. 阶段三 · 深度学习基础原理

### 目标

建立训练模型、导出资产和部署推理之间的完整概念链，为阶段四的压缩方法和阶段五的模型结构/转换做准备。本阶段理解训练，但不开展大型训练或调参项目。
```

Keep only Deep Learning Specialization as the main resource. The required outline is:

```markdown
- **Deep Learning Specialization**：
  - ★ Course 1：参数、层、激活、损失、前向/反向传播、计算图和向量化
  - ★ Course 2：初始化、正则化和优化算法原理
  - ★ batch、训练/验证/导出/推理的区别
  - Course 2 调参实战和大型编程作业（跳过）
  - Course 3（跳过）
  - Course 4（移到视觉流媒体支线）
  - Course 5（仅作为阶段七查漏补缺）
```

Completion criterion: explain how a neural network trains and which training-only state or operations are absent from inference. No stage project is required; a small MLP forward/backward trace is optional.

- [ ] **Step 4: Expand stage four from quantization into model efficiency**

Use this heading:

```markdown
## 6. 阶段四 · 模型压缩与高效网络基础
```

Keep MIT 6.5940 EfficientML, TensorRT quantization docs, AWQ/GPTQ/SmoothQuant/LLM.int8(), and Optimum. Add the existing Bilibili course `BV1Sw411y7Hs` as a selected pruning resource. The required concept sequence is:

```markdown
- ★ PTQ、QAT、calibration、scale/zero-point
- ★ per-tensor/per-channel、W8A8、W4A16
- ★ structured/unstructured pruning、稀疏化和剪枝后微调
- ★ 知识蒸馏
- ★ depthwise convolution、MobileNet 与轻量结构
- ★ EfficientML 高效 LLM：KV cache、W4/W8、GPTQ/AWQ/SmoothQuant/LLM.int8()
- 《深度学习模型部署与剪枝优化实战》：稀疏化、L1 正则、通道筛选、参数迁移、剪枝后微调
- 旧 Jetson Nano、旧 TAO 和旧安装命令（跳过）
```

State explicitly that real ONNX/TensorRT precision and speed validation occurs in stage five and deeper pruning training belongs to branch B. Replace the current calibration-only project with a compression proposal that defines accuracy, Engine tactic, latency, throughput, and memory checks; the actual experiment is consumed by project C.

- [ ] **Step 5: Verify stage ownership and commit**

Run:

```bash
rg -n "^## [3-6]\. 阶段|Deep Learning Specialization|D2L|CS224n|BV1Sw411y7Hs|项目 A|项目 B" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: stage two owns CUDA/projects A-B, stage three owns DLS, stage four owns compression; D2L and CS224n do not appear in stages three or four.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: rebuild foundational and compression stages"
```

### Task 3: Rebuild stages five through eight

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:401-658`
- Reference: `docs/superpowers/specs/2026-08-31-roadmap-stage-reordering-design.md:86-145`

- [ ] **Step 1: Add the D2L model-structure block to stage five**

Rename stage five to:

```markdown
## 7. 阶段五 · 模型结构、转换与推理引擎
```

Before the existing ONNX/TensorRT and DALI/CV-CUDA resources, add D2L as a continuous selected block:

```markdown
- ★ Module、参数、checkpoint 和模型结构读取
- ★ MLP、CNN、BatchNorm、ResNet
- ★ RNN/Seq2Seq、Attention、Transformer
- ★ tensor shape 和完整 forward 数据流
- 优化算法、完整训练、Kaggle、检测/分割、GAN、推荐系统和分布式训练（跳过）
```

Preserve the current ONNX/TensorRT and GPU preprocessing material. Add a required “compression validation” subsection that checks stage-four proposals through ONNX numerical alignment, TensorRT build logs/tactics, accuracy delta, latency, throughput, and memory. Project C must include FP32/FP16/INT8 and, when applicable, structured-sparsity comparisons.

- [ ] **Step 2: Replace stage six with containerized serving**

Use this heading:

```markdown
## 8. 阶段六 · 容器化与推理服务（Docker / HTTP / gRPC / Triton Server）
```

Move the old stage-seven Docker/gRPC/Triton resources here, but split the learning scope as follows:

```markdown
- Docker：multi-stage、Compose、network、healthcheck、GPU runtime、ARM64 镜像
- HTTP：请求/响应、状态码、JSON/二进制 tensor、超时、健康检查
- gRPC：`.proto`、stub、unary RPC、deadline、错误码
- Triton Server：model repository、TensorRT backend、dynamic batching、instance group、ensemble、metrics、Perf Analyzer
- Client：Triton 官方 HTTP client 和 unary gRPC client
- 自行实现完整 gRPC server、server/client/bidirectional streaming、中文 14 讲完整课（可选）
```

Remove the custom gRPC predict-server requirement. Project E must package the Triton TensorRT backend service and exercise both official clients.

- [ ] **Step 3: Replace stage seven with the LLM concept bridge**

Use this heading:

```markdown
## 9. 阶段七 · Transformer 与 LLM 概念
```

Use the pinned Bilibili collection `BV163Jc6pENx` and this exact required range:

```markdown
- ★ L7 Attention / LLM Introduction
- ★ L8 Self-Attention and Transformers
- ★ L9 Pretraining
- ★ L10 Post-training
- ★ L11 Natural Language Generation
- ★ tokenizer、embedding、encoder/decoder 补充
- ★ BERT vs GPT、pretraining vs post-training
- D2L Attention/Transformer：闭卷复习，不重看整课
- CS224n L5-L6：仅在 RNN/Seq2Seq/Attention 前置不足时回补
- DLS Course 5：仅作查漏补缺
```

No training project is required. The output is a Transformer/generation data-flow diagram and a comparison of BERT/GPT and pretraining/post-training.

- [ ] **Step 4: Tighten stage eight around inference**

Rename the heading to:

```markdown
## 10. 阶段八 · LLM 推理与 Edge-LLM
```

Preserve current runtime selection, Edge-LLM resources, metrics, project F, and platform constraints. Keep CS336 model structure, inference, evaluation, and system material here, but move Lecture 6 Kernels/Triton/XLA exclusively to branch C. Ensure the stage explicitly covers prefill/decode, KV cache, continuous batching, PagedAttention, TTFT, TPOT, INT4/AWQ/GPTQ, context/memory budgeting, and runtime selection.

- [ ] **Step 5: Verify stage ownership and commit**

Run:

```bash
rg -n "^## (7|8|9|10)\. 阶段|D2L|CS224n Spring 2024|BV163Jc6pENx|Triton Inference Server|Lecture 6" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: D2L is rooted in stage five, Triton Server in stage six, CS224n in stage seven, and CS336 inference in stage eight; Lecture 6 is not required in stage eight.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: rebuild deployment serving and LLM stages"
```

### Task 4: Rebuild the optional branches

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:659-741`
- Reference: `docs/superpowers/specs/2026-08-31-roadmap-stage-reordering-design.md:147-170`

- [ ] **Step 1: Replace the branch heading and ordering**

Use this branch order:

```markdown
## 11. 可选支线（视觉 / 压缩 / 算子 / 驱动 / 编译器）

### 分支 A：视觉流媒体（Jetson / GStreamer / DeepStream）
### 分支 B：模型压缩实战
### 分支 C：Triton Language、CUTLASS 和 CuTe
### 分支 D：Linux 驱动 / BSP
### 分支 E：TVM 和 MLIR
```

- [ ] **Step 2: Move the complete visual material into branch A**

Move the old stage-six Jetson, GStreamer, DeepStream resources, outlines, method, project D reference, and version warning into branch A. Set its entry condition to completion of stage five. Move DLS Course 4 here. Preserve the current NVIDIA/Seeed/DeepStream links, NVMM/zero-copy explanation, single-to-multistream workflow, and `tegrastats` measurements. Mark project D optional.

- [ ] **Step 3: Add branch B for pruning implementation**

Use this content:

```markdown
**进入条件**：完成阶段五，并已建立可复现的未剪枝 TensorRT baseline。

**内容**：稀疏训练、结构化剪枝、剪枝后微调、ONNX 导出、`polygraphy inspect sparsity`、TensorRT build log/sparse tactic 检查，以及剪枝前后的精度、延迟、吞吐和内存对比。

**停止条件**：不能只报告参数量或 FLOPs；必须证明目标硬件上的实际 tactic 和端到端收益，并记录无收益的结果。
```

Route the deeper selected pruning units from `BV1Sw411y7Hs` here.

- [ ] **Step 4: Relabel and update the existing technical branches**

Move current Triton Language/CUTLASS/CuTe to branch C and change its entry condition from stage three to stage two. Move Linux Driver/BSP to branch D and change its entry condition to stage two without a weekly time recommendation. Move TVM/MLIR to branch E and require a completed Triton Language or equivalent Kernel project.

- [ ] **Step 5: Verify branch separation and commit**

Run:

```bash
rg -n "^### 分支 [A-E]|进入条件|DeepStream|模型压缩实战|Triton Language|Linux 驱动|TVM 和 MLIR" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: five branches appear in A-E order; DeepStream only belongs to A, pruning implementation to B, Triton Language to C, driver to D, and compiler to E.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: move directional topics into roadmap branches"
```

### Task 5: Update the project ladder and concept clarification

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:842-915`

- [ ] **Step 1: Update project ownership and requirements**

Keep projects A-F identifiers stable. Apply these ownership rules:

```markdown
- 项目 A/B：阶段二 CUDA 主线
- 项目 C：阶段五主线；加入阶段四量化/稀疏方案的 TensorRT 实测
- 项目 D：分支 A 可选视觉项目
- 项目 E：阶段六主线；使用 Triton 官方 HTTP 和 unary gRPC client，不自建 gRPC server
- 项目 F：阶段八主线；阶段七是概念前置
```

Project C acceptance must mention accuracy delta, actual Engine tactics, latency, throughput, memory, and shape/batch. Project D text remains technically intact but is explicitly optional. Project E retains model repository, backend, batching, ensemble, metrics, timeouts, and Perf Analyzer.

- [ ] **Step 2: Update the TensorRT/Triton/LLM recommended sequence**

Use this exact sequence in section 14:

```markdown
推荐顺序：1. TensorRT 单模型运行；2. Docker 化并使用 Triton + TensorRT backend；3. 补齐 Transformer/LLM 概念；4. 理解 vLLM/TensorRT-LLM 等服务器运行时；5. 在 Orin 上使用 TensorRT-Edge-LLM 支持矩阵内路径。GStreamer/DeepStream 属于阶段五后的视觉支线，不是 Triton 或 LLM 的前置。
```

Keep the existing runtime-layer distinctions and Jetson backend support warning.

- [ ] **Step 3: Verify and commit**

Run:

```bash
rg -n "项目 [A-F]|可选视觉项目|推荐顺序|自建 gRPC|官方 HTTP|unary gRPC" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: all six projects have unambiguous ownership and the sequence matches stages five through eight.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: align roadmap projects with new stages"
```

### Task 6: Remove calendar scheduling from the roadmap

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:742-841`
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:917-946`

- [ ] **Step 1: Convert the CUDA sprint into an ordered checklist**

Rename section 12 to:

```markdown
## 12. CUDA 专项执行清单
```

Rename “第 1 周” through “第 6 周” to “步骤一” through “步骤六”. Preserve the current technical progression and commands: compile/run/indexing; memory/coalescing; streams/overlap; GEMM/cuBLAS; Sanitizer/Nsight; CUDA Graph/cooperative groups/Tensor Core/PTX-SASS. Remove duration promises and weekly wording.

- [ ] **Step 2: Replace the 24-week table with stage execution principles**

Use this complete replacement:

```markdown
## 15. 阶段执行与复习原则

路线按完成条件推进，不绑定周数或课时。已有能力只能通过对应交付物和停止条件跳过，不能以“看过课程”代替验收。

### 每阶段执行闭环

1. **系统学习**：按连续知识块完成主课，建立术语、因果关系和边界。
2. **阶段项目**：立即用代码、实验、图或报告调用所学内容。
3. **间隔复习**：进入后续阶段时闭卷解释前置知识，不清楚时定点回补，不重看整门课。
4. **证据验收**：以正确性、性能、版本、功耗、内存和可复现记录决定是否进入下一阶段。

### 主线与支线纪律

- 主线一次只推进一个阶段；支线必须满足进入条件。
- 支线不替代主线交付物，也不因课程有趣而提前插入。
- DeepStream、深度剪枝、Triton Language、驱动和编译器均按目标岗位选择。

### 回补规则

- 工程工具问题回补阶段一。
- Kernel 正确性或性能解释问题回补阶段二。
- 模型训练与推理边界问题回补阶段三。
- 量化、剪枝和稀疏化问题回补阶段四。
- 模型结构、shape、导出或 Engine 问题回补阶段五。
- API、容器、并发、batching 或监控问题回补阶段六。
- Transformer、预训练或生成问题回补阶段七。
- KV cache、LLM runtime、量化格式或服务指标问题回补阶段八。
```

- [ ] **Step 3: Verify calendar language is gone and commit**

Run:

```bash
if rg -n "24 周|六周 CUDA|第 [0-9]+ 周|每周并行" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md; then exit 1; fi
rg -n "CUDA 专项执行清单|步骤一|步骤六|阶段执行与复习原则|每阶段执行闭环" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: the negative search has no output; both replacement sections and ordered steps match.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: replace roadmap calendars with stage gates"
```

### Task 7: Align the capstone, one-page checklist, capability statement, and indexes

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:963-1108`

- [ ] **Step 1: Make DeepStream optional in the capstone**

The main capstone must require:

```markdown
- TensorRT FP16/INT8 image model and reproducible Engine build
- Triton TensorRT backend service with HTTP/unary gRPC clients, health and metrics
- TensorRT-Edge-LLM small-model service
- Nsight and tegrastats evidence, memory/power/version records, reproducible startup and benchmark
```

Move the multistream DeepStream pipeline to an explicitly labeled optional visual extension. Make the corresponding final-report question conditional on selecting branch A.

- [ ] **Step 2: Replace the one-page checklist with stage-ordered groups**

Use these group headings and required checklist topics:

```markdown
### 工程与 CUDA 基础
- stage-one toolchain, Docker basics, CUDA Kernels, Sanitizer/Nsight

### 模型原理与压缩
- DLS fundamentals, quantization, pruning/sparsity design

### 模型部署
- D2L model structures, ONNX alignment, TensorRT FP32/FP16/INT8, GPU preprocessing

### 推理服务
- Docker Serving, HTTP/unary gRPC clients, Triton backend/batching/metrics

### LLM 推理
- CS224n L7-L11, KV cache/prefill/decode, runtime comparison, Edge-LLM service

### 按岗位选择支线
- visual streaming, pruning implementation, Triton Language/CUTLASS/CuTe, driver/BSP, TVM/MLIR
```

Do not leave DeepStream in a mainline checklist group.

- [ ] **Step 3: Update late-document cross-references and indexes**

Keep the Orin version discipline and all current official links. Update stage names/numbers in capability text and link descriptions. In the key-link index, label DeepStream as visual-branch material and keep Triton Server separate from Triton Language. Do not modify the hardware/software baseline at the top of the document.

- [ ] **Step 4: Verify and commit**

Run:

```bash
rg -n "最终毕业项目|可选视觉扩展|工程与 CUDA 基础|模型原理与压缩|推理服务|LLM 推理|按岗位选择支线" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: the capstone and checklist distinguish mainline requirements from optional branches.

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: align capstone and checklist with roadmap stages"
```

### Task 8: Run a full-document consistency audit

**Files:**
- Modify if needed: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md`
- Verify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md`

- [ ] **Step 1: Verify the exact eight-stage order**

Run:

```bash
rg -n "^## (3|4|5|6|7|8|9|10)\. 阶段" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected, in order: engineering/tools; CUDA; DL fundamentals; compression; model structure/conversion/engine; containerized serving; Transformer/LLM concepts; LLM inference/Edge-LLM.

- [ ] **Step 2: Verify course and technology ownership**

Run:

```bash
rg -n "Deep Learning Specialization|D2L|CS224n Spring 2024|Triton Inference Server|Triton Language|DeepStream" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: DLS main content is stage three, D2L main content stage five, CS224n stage seven, Triton Server stage six, Triton Language branch C, and DeepStream branch A/project D/optional capstone only.

- [ ] **Step 3: Reject stale stage and calendar labels**

Run:

```bash
if rg -n "阶段二 · 模型概念底座|阶段三 · CUDA|阶段六 · Jetson|阶段七 · 容器|24 周|六周 CUDA|第 [0-9]+ 周|DeepStream.*必修" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md; then exit 1; fi
```

Expected: exit status 0 with no output.

- [ ] **Step 4: Inspect all remaining stage references and Markdown health**

Run:

```bash
rg -n "阶段[一二三四五六七八]|分支 [A-E]" docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git diff --check HEAD~7 HEAD -- docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git status --short
```

Expected: every cross-reference matches the approved ownership; no whitespace errors; unrelated pre-existing user changes remain untouched.

- [ ] **Step 5: Commit any final consistency corrections**

Only if step 4 required edits:

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: fix roadmap cross-reference consistency"
```
