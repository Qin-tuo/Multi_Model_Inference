# Orin NX Multi-Model Inference Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese technical knowledge base and project blueprint for multi-model inference optimization on Jetson Orin NX 16GB.

**Architecture:** Keep the repository documentation-first. Separate stable engineering guidance from time-sensitive compatibility and repository references, and label all unmeasured performance statements so later Orin experiments can replace assumptions with evidence.

**Tech Stack:** Markdown, NVIDIA JetPack 7.2, CUDA, TensorRT, DeepStream, TensorRT Edge-LLM, Triton Inference Server, Nsight Systems, Nsight Compute, Git.

---

## File Map

- `README.md`: repository purpose, scope, reading order, status, and shortest recommended path.
- `docs/01-positioning-and-goals.md`: role positioning, project boundary, and success definition.
- `docs/02-job-skill-map.md`: inference job families, recurring requirements, gaps, and portfolio evidence.
- `docs/03-orin-platform-and-stack.md`: Orin constraints and responsibility boundaries across the NVIDIA stack.
- `docs/04-cuda-kernel-and-operator-optimization.md`: Linux kernel/CUDA kernel/operator/plugin distinctions and required CUDA depth.
- `docs/05-multi-model-concurrency-architecture.md`: target architecture, scheduling, isolation, overload control, and measurements.
- `docs/06-learning-roadmap.md`: four-to-five-month staged learning and validation path.
- `docs/07-project-blueprint-and-acceptance.md`: flagship project increments, experiment matrix, artifacts, and acceptance gates.
- `docs/08-environment-setup.md`: host, flashing, native, container, tooling, and environment verification guide.
- `references/github-projects.md`: curated open-source repositories with priority and reading guidance.
- `references/official-resources.md`: authoritative product, compatibility, API, profiling, and deployment references.

## Task 1: Repository Entry And Positioning

**Files:**
- Create: `README.md`
- Create: `docs/01-positioning-and-goals.md`

- [x] **Step 1: Write the repository entry page**

Include these concrete sections: target reader, current status, scope, non-goals, document map, recommended reading paths, relationship to `RK_LLM`, `S100_VLA`, and `RK_S100_MFSysetem`, and evidence labels (`资料结论`, `主机验证`, `Orin 实测`). State that this repository does not yet contain runnable inference code or Orin benchmark results.

- [x] **Step 2: Write the positioning document**

Define the target direction as `边缘异构模型推理系统与性能优化`. Contrast it with pure LLM serving, pure model training, Linux BSP work, and cluster platform engineering. Define the flagship workload as real-time CV plus bursty NLP/embedding plus a background LLM or VLM sharing one Orin NX 16GB.

- [x] **Step 3: Verify navigation and scope language**

Run:

```bash
test -s README.md
test -s docs/01-positioning-and-goals.md
rg -n "资料结论|主机验证|Orin 实测|边缘异构模型推理系统与性能优化" README.md docs/01-positioning-and-goals.md
```

Expected: both files are non-empty and every evidence/positioning label is found.

- [x] **Step 4: Commit the entry and positioning documents**

```bash
git add README.md docs/01-positioning-and-goals.md
git commit -m "docs: define Orin inference project scope"
```

## Task 2: Job Skill Map And Orin Platform Stack

**Files:**
- Create: `docs/02-job-skill-map.md`
- Create: `docs/03-orin-platform-and-stack.md`

- [x] **Step 1: Write the job skill map**

Organize requirements into three role families: edge inference deployment/performance, inference framework development, and cluster inference platform. For each recurring skill, record expected depth, why it matters, a learnable task, portfolio evidence, and priority (`P0`, `P1`, `P2`). Cover C++/Python, Linux performance analysis, CUDA, TensorRT, ONNX, quantization, profiling, model serving, scheduling, observability, containers, and LLM-specific KV cache/batching.

- [x] **Step 2: Write the Orin and software-stack map**

Explain unified-memory and memory-bandwidth constraints, CPU/GPU/DLA/video-engine roles, and the responsibility of JetPack, CUDA, cuDNN/cuBLAS, TensorRT, DeepStream/GStreamer, TensorRT Edge-LLM, and Triton Inference Server. Explicitly distinguish an Orin deployment path from vLLM and mainline TensorRT-LLM source-study paths. Mark version and support statements with `核验日期：2026-08-19` and link them through the references documents added later.

- [x] **Step 3: Check required stack coverage**

Run:

```bash
for term in CUDA TensorRT DeepStream GStreamer "TensorRT Edge-LLM" Triton vLLM "TensorRT-LLM"; do rg -q "$term" docs/02-job-skill-map.md docs/03-orin-platform-and-stack.md || exit 1; done
```

Expected: exit status 0.

- [x] **Step 4: Commit the job and platform documents**

```bash
git add docs/02-job-skill-map.md docs/03-orin-platform-and-stack.md
git commit -m "docs: map inference skills and Orin stack"
```

## Task 3: Reproducible Learning Environment

**Files:**
- Create: `docs/08-environment-setup.md`
- Modify: `README.md`
- Modify: `docs/03-orin-platform-and-stack.md`

- [x] **Step 1: Document the physical and host prerequisites**

Start from the current target baseline shown in SDK Manager: Ubuntu 24.04 x86_64 host, Jetson Orin NX 16GB, and JetPack 7.2.1 direct flash. List the required data-capable USB cable, stable power, recovery-mode procedure, display/network options, NVMe/storage capacity planning, host download/cache capacity, and backup warning. Mark the SDK Manager install folder as a host-side cache rather than the Jetson runtime filesystem.

- [x] **Step 2: Document SDK Manager and post-flash verification**

Separate every command and action by execution location using `Host` and `Orin` labels. Cover SDK Manager selection, recovery detection, direct flash stages, first boot, package completion, and version inventory. Include read-only verification commands for architecture, Ubuntu/L4T release, CUDA, TensorRT, DeepStream/GStreamer, Docker/container runtime, device nodes, storage, and tegrastats. Explain which checks are optional when a component was not selected.

- [x] **Step 3: Define native and container learning tracks**

Make the native track the hardware/API baseline for CUDA samples, TensorRT, DeepStream, Nsight, and TensorRT Edge-LLM. Make the container track the isolation/reproducibility path for model dependencies and serving experiments. Explain aarch64 image requirements, JetPack/L4T compatibility, GPU runtime access, bind mounts, model/cache placement, Python virtual environments, and why host x86_64 wheels/images cannot be reused directly on Orin.

- [x] **Step 4: Add component-specific setup boundaries**

For CUDA, TensorRT, DeepStream, TensorRT Edge-LLM, Triton Inference Server, Model Optimizer, telemetry, and profiling tools, provide purpose, recommended install path, verification command, and failure symptoms. Use official commands only after checking current documentation; label community tools such as `jetson-containers` and `jetson_stats` as optional rather than platform prerequisites.

- [x] **Step 5: Add reproducibility and troubleshooting rules**

Define a version manifest containing JetPack/L4T, Ubuntu, kernel, CUDA, TensorRT, cuDNN, DeepStream, Python, container runtime, image digest, Git revisions, model hashes, and power mode. Include a clean-room smoke-test order and troubleshooting branches for recovery-device detection, package completion, architecture mismatch, missing GPU access, Python binding mismatch, disk exhaustion, OOM, and thermal throttling.

- [x] **Step 6: Validate environment-guide coverage**

Run:

```bash
for term in "Host" "Orin" "JetPack 7.2.1" "recovery" "aarch64" "TensorRT Edge-LLM" "tegrastats" "version manifest"; do rg -qi "$term" docs/08-environment-setup.md || exit 1; done
rg -q "08-environment-setup.md" README.md docs/03-orin-platform-and-stack.md
```

Expected: exit status 0 and the environment guide is linked from both navigation documents.

- [x] **Step 7: Commit the environment guide**

```bash
git add README.md docs/03-orin-platform-and-stack.md docs/08-environment-setup.md
git commit -m "docs: add Orin learning environment setup"
```

## Task 4: CUDA Depth And Multi-Model Concurrency

**Files:**
- Create: `docs/04-cuda-kernel-and-operator-optimization.md`
- Create: `docs/05-multi-model-concurrency-architecture.md`

- [x] **Step 1: Write the CUDA and operator guide**

Separate Linux kernel, CUDA kernel, graph operator, TensorRT layer, and TensorRT plugin. Define a practical learning ladder: profiling, memory transfer/lifetime, streams/events, CUDA Graphs, launch configuration, memory hierarchy, one fused preprocessing kernel, and one plugin. Give an optimization decision order that starts with measurement and avoids premature PTX/SASS or broad custom-operator work.

- [x] **Step 2: Write the concurrency architecture**

Cover request concurrency, model concurrency, pipeline concurrency, and heterogeneous concurrency. Define model/context pools, queues, priorities, deadlines, dynamic batching, backpressure, admission control, memory budgets, warm-up, failure recovery, and telemetry. Include a Mermaid architecture diagram and a lifecycle sequence for overload handling.

- [x] **Step 3: Define workload-specific metrics**

Record CV FPS/p99/drop rate, NLP throughput/p95, LLM TTFT/TPOT/tokens per second, isolated-versus-concurrent degradation, goodput, memory, power, temperature, and throttling. Explain that concurrency count alone is not a performance result.

- [x] **Step 4: Validate terminology and metrics**

Run:

```bash
for term in "Linux kernel" "CUDA kernel" "TensorRT Plugin" "TTFT" "TPOT" "p99" "admission control"; do rg -qi "$term" docs/04-cuda-kernel-and-operator-optimization.md docs/05-multi-model-concurrency-architecture.md || exit 1; done
```

Expected: exit status 0.

- [x] **Step 5: Commit optimization and concurrency documents**

```bash
git add docs/04-cuda-kernel-and-operator-optimization.md docs/05-multi-model-concurrency-architecture.md
git commit -m "docs: define CUDA and concurrency strategy"
```

## Task 5: Learning Roadmap And Flagship Project Acceptance

**Files:**
- Create: `docs/06-learning-roadmap.md`
- Create: `docs/07-project-blueprint-and-acceptance.md`

- [x] **Step 1: Write the four-to-five-month roadmap**

Split the path into reproducible baseline, TensorRT and profiling, CUDA/plugin practice, quantization and LLM runtime, multi-model scheduler, and stability/evidence phases. Every phase must state inputs, exercises, outputs, exit criteria, and activities that can be deferred. Include an accelerated path for reusing RK and S100 validation experience without mixing hardware-specific claims.

- [x] **Step 2: Write the project blueprint**

Define incremental milestones: single CV baseline, telemetry harness, second-model coexistence, background LLM/VLM, scheduler/overload control, and soak test. Specify repository artifacts that will be added when implementation begins: version manifest, model manifest, benchmark scenarios, raw results, analysis report, launch configuration, and reproducibility guide.

- [x] **Step 3: Define an experiment and acceptance matrix**

Include isolated baselines, concurrent fixed-load tests, saturation sweep, batch-size sweep, quantization quality/performance comparison, cold/warm start, memory-pressure test, thermal soak, process failure recovery, and repeatability. Acceptance must use measured before/after criteria and cannot label a demo as production-ready.

- [x] **Step 4: Verify roadmap completeness**

Run:

```bash
for term in "退出条件" "量化" "并发" "稳定性" "复现"; do rg -q "$term" docs/06-learning-roadmap.md docs/07-project-blueprint-and-acceptance.md || exit 1; done
```

Expected: exit status 0.

- [x] **Step 5: Commit the roadmap and project blueprint**

```bash
git add docs/06-learning-roadmap.md docs/07-project-blueprint-and-acceptance.md
git commit -m "docs: add roadmap and project acceptance plan"
```

## Task 6: Curated External Resources

**Files:**
- Create: `references/github-projects.md`
- Create: `references/official-resources.md`
- Modify: `README.md`
- Modify: `docs/03-orin-platform-and-stack.md`

- [x] **Step 1: Verify time-sensitive primary sources**

Use official NVIDIA documentation or project-owned repositories to verify JetPack/Orin support, TensorRT Edge-LLM support, DeepStream/Triton integration, CUDA profiling tools, and serving benchmark tools. Record `核验日期：2026-08-19` near compatibility statements and avoid inferring support from an unpinned default branch.

- [x] **Step 2: Write the GitHub guide**

Curate repositories into `P0 hands-on`, `P1 tooling`, and `P2 source study`. At minimum cover `NVIDIA/cuda-samples`, `NVIDIA/TensorRT`, `NVIDIA-AI-IOT/deepstream_reference_apps`, `NVIDIA/TensorRT-Edge-LLM`, `triton-inference-server/server`, `triton-inference-server/tutorials`, `NVIDIA/Model-Optimizer`, `triton-inference-server/perf_analyzer`, `triton-inference-server/model_analyzer`, `dusty-nv/jetson-containers`, `rbonghi/jetson_stats`, `vllm-project/vllm`, `NVIDIA/TensorRT-LLM`, `NVIDIA/cutlass`, and `sgl-project/sglang`. For each, state purpose, what to read or run, Orin relevance, and limitations.

- [x] **Step 3: Write the official-resource index**

Group links by platform/release, runtime, serving, profiling, and optimization. Add a short reason for every link and identify primary sources. Link this index from the README and platform document.

- [x] **Step 4: Check resource coverage and links**

Run:

```bash
for repo in cuda-samples TensorRT deepstream_reference_apps TensorRT-Edge-LLM perf_analyzer jetson-containers vllm cutlass; do rg -q "$repo" references/github-projects.md || exit 1; done
rg -n "https:" references/github-projects.md references/official-resources.md
```

Expected: every required repository is found and both reference files contain HTTPS links.

- [x] **Step 5: Commit the curated resources**

```bash
git add README.md docs/03-orin-platform-and-stack.md references/github-projects.md references/official-resources.md
git commit -m "docs: curate NVIDIA inference resources"
```

## Task 7: Whole-Repository Consistency And Handoff

**Files:**
- Modify: `README.md`
- Modify: any document with a verified inconsistency

- [x] **Step 1: Check file structure and internal links**

Run:

```bash
find README.md docs references -type f -name '*.md' -print | sort
for file in docs/01-positioning-and-goals.md docs/02-job-skill-map.md docs/03-orin-platform-and-stack.md docs/04-cuda-kernel-and-operator-optimization.md docs/05-multi-model-concurrency-architecture.md docs/06-learning-roadmap.md docs/07-project-blueprint-and-acceptance.md docs/08-environment-setup.md references/github-projects.md references/official-resources.md; do test -s "$file" || exit 1; done
```

Expected: the full designed structure is listed and every required file is non-empty.

- [x] **Step 2: Scan for unfinished markers and unsupported claims**

Run:

```bash
rg -n --glob '!docs/superpowers/**' "T[O]DO|T[B]D|REPLACE_ME|占位文本|保证提升|生产级已完成" README.md docs references && exit 1 || true
rg -n --glob '!docs/superpowers/**' "Orin 实测" README.md docs references
```

Expected: the first scan reports no unfinished or unjustified completion language; every `Orin 实测` occurrence clearly describes future evidence or an absent result.

- [x] **Step 3: Review the spec line by line**

Compare the completed documents against `docs/superpowers/specs/2026-08-19-orin-nx-multi-model-inference-knowledge-base-design.md`. Confirm coverage of role mapping, platform stack, environment setup, kernel terminology, concurrency, roadmap, project acceptance, references, and evidence rules.

- [x] **Step 4: Review the final diff and commit consistency fixes**

```bash
git diff --check
git status --short
git add README.md docs references
git commit -m "docs: finalize Orin inference knowledge base"
```

Expected: `git diff --check` exits 0. If there are no consistency edits, omit the final commit and record that the previous commits already contain the verified result.
