# Stage One Capability-Driven Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Stage 1 section of the AI Infra roadmap so course coverage, prerequisite order, topic supplements, skips, project boundaries, and measurable acceptance criteria accurately guide the learner.

**Architecture:** Keep one canonical roadmap file, but organize Stage 1 by capability modules rather than by undifferentiated course lists. Each module states prerequisites, selected course sections, missing topics, trigger-based supplements, a minimal experiment, and a stop condition; later-stage handoffs remain explicit. Preserve all unrelated dirty worktree changes.

**Tech Stack:** Markdown, shell tools (`rg`, `git`, `git diff --check`), official Python/PyTorch/NumPy/CMake/Linux/Docker documentation links.

---

### Task 1: Record and verify the implementation baseline

**Files:**
- Read: `docs/superpowers/specs/2026-09-04-stage-one-capability-driven-refinement-design.md`
- Read: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md`
- Read: `docs/learning-log.md`
- Modify: `docs/superpowers/plans/2026-09-04-stage-one-capability-driven-refinement.md` only if a discovered conflict requires an explicit plan correction

- [ ] **Step 1: Capture the current worktree state**

Run:

```bash
git status --short --branch
git diff -- docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md docs/learning-log.md
```

Expected: existing user changes are listed and are not reverted.

- [ ] **Step 2: Confirm the approved design is the source of truth**

Run:

```bash
rg -n "能力依赖与先后顺序|模块一|模块八|阶段一总门槛|路线图修改范围" \
  docs/superpowers/specs/2026-09-04-stage-one-capability-driven-refinement-design.md
```

Expected: all design sections are present before implementation starts.

- [ ] **Step 3: Commit the implementation plan**

```bash
git add docs/superpowers/plans/2026-09-04-stage-one-capability-driven-refinement.md
git commit -m "docs: plan stage one roadmap refinement"
```

### Task 2: Correct the Stage 1 course inventory and add the dependency framework

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:87-122`

- [ ] **Step 1: Replace the generic Stage 1 opening with the capability-driven contract**

Retain the Stage 1 heading, but state that it is an engineering foundation and that the canonical unit is `能力目标 -> 课程选段 -> 专题补缺 -> 实验 -> 证据验收`.

- [ ] **Step 2: Correct course names and source attribution**

Make these exact corrections:

```text
BV1wr5T6bEN4 -> Bro Code - Python Full Course for free 2024 (B站 AI 翻译版), not freeCodeCamp
BV1D9GLzyEL6 -> 尚硅谷 Python数据分析教程 (NumPy/Pandas/Matplotlib)
BV1Zn4y1X7AZ -> 尚硅谷 Docker教程，3小时速通，docker部署到实战！
```

For every listed resource, add a short role label: `主线必修`, `专题必修`, `条件回补`, `延后阶段`, `明确跳过`, or `已完成可验收`.

- [ ] **Step 3: Add the Stage 1 dependency diagram and ordering rules**

Add the two chains from the approved design:

```text
Linux Shell/Git -> Python 工程化/uv -> NumPy -> PyTorch -> ONNX/ORT -> model-tools -> Docker CPU/headless
C++ 基础 -> RAII/STL -> CMake/GDB/Sanitizer -> CPU GEMM -> POSIX fd/process/thread/socket -> TCP 强化
```

State that the chains may be studied in parallel but Docker follows a working CLI, and that prior verified work can skip repeated lessons.

- [ ] **Step 4: Verify the inventory correction**

Run:

```bash
rg -n "freeCodeCamp|Bro Code|BV1wr5T6bEN4|尚硅谷 Docker教程|能力目标" \
  docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: Bro Code is not attributed to freeCodeCamp, and the Stage 1 framework is visible.

- [ ] **Step 5: Commit the inventory/framework change**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: clarify stage one course roles and order"
```

### Task 3: Rewrite Python, NumPy, PyTorch, and ONNX as ordered modules

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:123-190`
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md` at the Stage 1 course and reference sections as needed to keep links consistent

- [ ] **Step 1: Replace the course-shaped syllabus with module-specific coverage**

Create four labeled modules in this order:

1. Shell/Git handoff to Python engineering (the existing Linux details can remain in Task 5).
2. Python basics and engineering: use completed 林粒粒 material; select Bro Code P21, P34-P35, P39-P41, P59-P61 only; add official `pathlib`, `dataclass`, `argparse`, logging, subprocess, uv, pytest, timing, and reproducibility references.
3. NumPy/image arrays: use 尚硅谷 P6-P21, but explicitly supplement stride, contiguous, view/copy, HWC/BGR -> NCHW/RGB, `asarray/frombuffer`, GEMM, and tolerance checks.
4. PyTorch/ONNX: use mrdbourke P15-P28, P33-P36, P42, P64-P65, P77/P80-P84; move training/CNN material to later stages; add official Tensor Views, saving/loading, DataLoader, benchmark, ONNX and ORT references.

- [ ] **Step 2: State overlaps and skips beside each module**

Use explicit labels and reasons. Do not require the learner to repeat the completed Python course or all three PyTorch courses. Mark `GradScaler`, DDP, `torch.compile`, complete training loops, advanced CNN/RNN, and ONNX dynamic control flow as later/conditional material.

- [ ] **Step 3: Add measurable module stop conditions**

Include these exact evidence requirements:

```text
Python: stable manifest CLI, exit codes, structured logging, >=5 pytest tests
NumPy: contiguous float32 NCHW/RGB conversion, non-square/batched GEMM, PyTorch reference tolerance
PyTorch: eval vs inference_mode experiment, state_dict weights_only/map_location load, DataLoader measurement
ONNX: checker pass, fixed plus two unseen legal batch shapes, explicit provider and rtol/atol comparison
```

- [ ] **Step 4: Add the on-demand Pandas/Matplotlib mini-lab boundary**

State that Pandas and Matplotlib are triggered after the first benchmark CSV/JSON exists. Select the relevant chapters from `BV1D9GLzyEL6` and require a report containing p50/p95/p99, throughput, memory, power, and temperature; do not make the full data-analysis course a Stage 1 prerequisite.

- [ ] **Step 5: Verify the model-chain structure**

Run:

```bash
rg -n "模块二|模块三|模块四|模块五|ONNX|Pandas|Matplotlib|weights_only|dynamic_shapes|p50/p95" \
  docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: Python -> NumPy -> PyTorch -> ONNX ordering is explicit, and Pandas/Matplotlib are trigger-based.

- [ ] **Step 6: Commit the model-chain change**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: refine stage one python and model modules"
```

### Task 4: Rewrite C++ and Linux as prerequisite-aware modules

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:107-190`
- Modify: the Stage 1 interview/advanced supplement block immediately following the modules

- [ ] **Step 1: Define the C++ order and accurate course boundary**

State that the freeCodeCamp video supplies C++ language basics only, and order the required topics as:

```text
syntax -> pointers/references -> lifetime/RAII -> STL/smart pointers -> compile/link -> CMake/CTest -> GDB/Sanitizers -> CPU GEMM
```

Use LearnCpp, CMake, GDB, C++ Core Guidelines, and sanitizer documentation for missing topics.

- [ ] **Step 2: Define the Linux/POSIX order and accurate course boundary**

State that Missing Semester teaches command-line/debugging workflow and the 尚硅谷 130-P course teaches C/POSIX APIs. Select fd/syscall, process, signal, thread/synchronization, TCP/socket, and epoll sections; defer or skip Windows/VS, SysV IPC bulk, UDP/link-layer, async cancellation, and daemon implementation.

- [ ] **Step 3: Move practical concurrency before interview-only theory**

Keep smart-pointer implementation, vtable layout, complex memory ordering, and similar interview questions in the interview block, but make RAII, ownership, basic `std::thread`, mutex, condition variable, fd handling, partial I/O, and graceful SIGTERM part of the practical foundation.

- [ ] **Step 4: Add independent acceptance criteria**

Require:

```text
cpp-gemm: Debug/Release fresh build, CTest, ASan/UBSan, fixed-seed tolerance, median/p95 JSON
telemetry: 60 samples, missing-resource handling, ShellCheck
TCP (framework track): length-prefix protocol, bounded queue, partial I/O, disconnect/backpressure, TSan
```

Mark TCP as a systems/framework-track strengthening deliverable rather than a universal gate to Stage 2.

- [ ] **Step 5: Verify C++/Linux sequencing and labels**

Run:

```bash
rg -n "RAII|CMake|CTest|Sanitizer|模块六|模块七|系统/框架岗位强化|并发 TCP" \
  docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: CMake/GDB/Sanitizer are no longer claimed as covered by the freeCodeCamp video, and TCP is labeled as strengthening.

- [ ] **Step 6: Commit the systems-module change**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: order stage one cxx and linux foundations"
```

### Task 5: Rewrite Docker boundaries and synchronize project/checklist handoffs

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:118-190`
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:492-556`
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:953-1001`
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md:1034-1185`

- [ ] **Step 1: Define Stage 1 Docker as CPU/headless only**

Select the useful sections from the 2024 Chinese Docker course and use the 2019 English course only for concepts. Add `.dockerignore`, build context, tag/digest, layer/cache, exec-form, non-root UID/GID, bind mount/volume permissions, and network-isolation topics as supplements.

- [ ] **Step 2: Preserve the Stage 6 boundary**

Keep Compose, multi-stage production builds, Buildx/multi-platform publishing, healthcheck, resource limits, NVIDIA Container Toolkit/CDI, GPU injection, NGC/Triton images, and service-network behavior in Stage 6. State that `linux/arm64` proves CPU architecture only, not Jetson CUDA/TensorRT/camera compatibility.

- [ ] **Step 3: Split the Stage 1 project deliverables**

Update the roadmap to describe four independent artifacts:

```text
model-tools | cpp-gemm | telemetry | container
```

Do not force C++ GEMM, telemetry, or TCP into the model-tools image. Keep model/input read-only mounts and outputs on a separate writable mount.

- [ ] **Step 4: Synchronize project ladder and one-page checklist**

Update project C/E/F handoff wording only where needed, label project D as optional visual branch, and move the Stage 1 core/strengthening checklist to match the new module order. Preserve all later-stage content not required for the handoff.

- [ ] **Step 5: Verify Docker and handoff boundaries**

Run:

```bash
rg -n "CPU/headless|linux/arm64|Jetson GPU|Compose|Buildx|NVIDIA Container Toolkit|模块八|model-tools|cpp-gemm|telemetry|container" \
  docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Expected: Stage 1 does not require GPU runtime or Compose, and Stage 6 still owns those topics.

- [ ] **Step 6: Commit the Docker/handoff change**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: clarify stage one docker and project handoffs"
```

### Task 6: Synchronize links, cross-references, and exclusion statements

**Files:**
- Modify: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md` wherever Stage 1, the one-page checklist, links index, exclusion list, or cross-stage handoff references the old coverage

- [ ] **Step 1: Add the official supplement links used by the modules**

Include the official Python, PyPA, uv, pytest, NumPy, PyTorch Tensor Views/Saving/DataLoader/benchmark, ONNX, LearnCpp, CMake, GDB, sanitizer, Linux man-page, Docker, and NVIDIA Container Toolkit links from the approved design.

- [ ] **Step 2: Remove contradictory language**

Replace any statement that says the listed video fully covers `argparse`, `logging`, `subprocess`, `pytest`, CMake, GDB, RAII/STL, ONNX, or ARM64 Jetson GPU support. Keep the distinction between concepts that are version-stable and project dependencies that must be locked.

- [ ] **Step 3: Check cross-stage ownership**

Confirm that:

```text
Stage 1: Python/NumPy/PyTorch/ONNX minimum, C++/Linux basics, CPU Docker
Stage 2: CUDA and GPU profiling
Stage 3: training/deep-learning principles
Stage 5: full model structure/TensorRT/GPU preprocessing
Stage 6: Compose/GPU container/HTTP/gRPC/Triton
Stage 7-8: Transformer/LLM and Edge-LLM
```

- [ ] **Step 4: Commit synchronized references**

```bash
git add docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
git commit -m "docs: synchronize stage one roadmap references"
```

### Task 7: Full documentation verification and review

**Files:**
- Read: `docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md`
- Read: `docs/superpowers/specs/2026-09-04-stage-one-capability-driven-refinement-design.md`
- Read: all commits created by Tasks 2-6

- [ ] **Step 1: Run structural searches**

```bash
rg -n "Bro Code|freeCodeCamp.*Python|主课.*大纲|NumPy|ONNX|CMake|GDB|RAII|Pandas|Matplotlib|CPU/headless|Compose|Buildx|Jetson GPU" \
  docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md
```

Review every match for correct ownership and wording.

- [ ] **Step 2: Check links and local anchors mechanically**

```bash
python - <<'PY'
from pathlib import Path
import re

p = Path("docs/EXTERNAL_FREE_BILINGUAL_AI_INFRA_ROADMAP.md")
s = p.read_text()
assert "Bro Code" in s
assert "ONNX" in s and "CMake" in s and "CPU/headless" in s
assert "Compose" in s and "GPU runtime" in s
assert "Pandas" in s and "Matplotlib" in s
print("stage-one content assertions passed")
PY
git diff --check origin/main...HEAD
```

Expected: assertions pass and Git reports no whitespace errors. Do not claim every external URL is reachable unless each was actually checked.

- [ ] **Step 3: Review the final diff without touching unrelated work**

```bash
git diff --stat origin/main...HEAD
git status --short --branch
git log --oneline --decorate -8
```

Expected: only the intended roadmap/spec/plan commits are included in the implementation branch; pre-existing user changes remain visible and untouched.

- [ ] **Step 4: Request a final code/documentation review**

Use the `requesting-code-review` workflow against the complete roadmap diff. Resolve any findings before completion.

