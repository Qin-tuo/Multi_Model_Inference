# Dual-Track Inference Toolchain And Skill Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one unified skill-stack guide for simultaneous edge inference performance and inference framework job applications, then connect it to the existing knowledge base.

**Architecture:** Create one lifecycle-oriented guide as the canonical matrix. Keep the existing platform, CUDA, concurrency, roadmap, and environment documents focused; update only their navigation and dual-track statements. Validate the result with deterministic Markdown, link, and content-coverage checks.

**Tech Stack:** Markdown, repository-relative links, `rg`, `awk`, `git diff --check`.

---

### Task 1: Write the unified dual-track guide

**Files:** Create `docs/09-dual-track-toolchain-and-skill-stack.md`.

- [ ] Add the target profile: embedded C/C++ and Linux experience, systematic CUDA/TensorRT start, and two primary role targets.
- [ ] Explain the role split: edge work optimizes end-to-end behavior on one constrained device; framework work optimizes reusable runtime, scheduler, memory, and kernel behavior.
- [ ] Add a shared matrix with capability, tools, required depth, Host/Orin location, exercise, and evidence columns.
- [ ] Cover Linux/C++ build/debug, CMake/Ninja, Python automation, model fundamentals, ONNX, TensorRT, benchmark design, telemetry, and Nsight Systems.
- [ ] Cover the full lifecycle toolchain: GCC/Clang, GDB/core dumps, `perf`, `strace`, sanitizers, PyTorch, ONNX Runtime, Netron, Polygraphy, `trtexec`, JetPack, CUDA, cuDNN/cuBLAS, TensorRT, DeepStream/GStreamer, Edge-LLM, Triton, NVTX, Nsight Compute, Compute Sanitizer, `tegrastats`/`jtop`, HTTP/gRPC, Perf Analyzer, Docker/NGC, ARM64 checks, manifests, CI, vLLM, SGLang, TensorRT-LLM, TensorRT Plugin, Triton language, CUTLASS/CuTe, and deferred PTX/SASS.
- [ ] Define the depth ladder `Use -> Diagnose -> Optimize -> Design/Contribute`; require diagnosis for most P0 skills and optimization for C++, TensorRT, profiling, CUDA basics, concurrency, and one selected Plugin/runtime component.
- [ ] Add the Host/Orin model and these concrete workflows: `ssh <user>@<orin-ip> 'nsys --version; nsys status -e'`, target-side `nsys profile --trace=cuda,nvtx,osrt`, `scp` of `.nsys-rep`, and Host GUI analysis. State that Host-only results are not Orin evidence.
- [ ] Add a 20-week map with one edge deliverable and one framework deliverable per phase, beginning with a TensorRT baseline and deferring Kubernetes, multi-node, PTX/SASS, and full CUTLASS work.
- [ ] Add a JD evidence matrix and anti-patterns: installation, average latency, or one successful run are not competence evidence.

### Task 2: Connect navigation and role mapping

**Files:** Modify `README.md` and `docs/02-job-skill-map.md`.

- [ ] Add the new guide after the job skill map in README navigation and add a quick-lookup row for preparing both role families.
- [ ] Mark edge inference performance and inference framework development as both primary targets in the job map, link the new guide, and leave cluster platform work as later context.

### Task 3: Connect roadmap and environment

**Files:** Modify `docs/06-learning-roadmap.md` and `docs/08-environment-setup.md`.

- [ ] Add a short roadmap section stating that every phase has an edge output and a framework output from the same Orin project.
- [ ] Add a focused Nsight Host/Orin section to the environment guide, label every command location, explain GUI versus target collection, and link the new guide.

### Task 4: Connect official references

**Files:** Modify `references/official-resources.md`.

- [ ] Ensure official links cover CUDA Programming Guide, Nsight Systems installation/user guide, Nsight Compute, TensorRT benchmarking, TensorRT Plugin documentation, JetPack/Orin documentation, and serving benchmark references.
- [ ] Mark community utilities as optional and do not imply unsupported Orin compatibility.

### Task 5: Verify and commit

**Files:** Verify `README.md`, `docs/02-job-skill-map.md`, `docs/06-learning-roadmap.md`, `docs/08-environment-setup.md`, `docs/09-dual-track-toolchain-and-skill-stack.md`, and `references/official-resources.md`.

- [ ] Run `test -s` for all six files and `rg -qi` for `Nsight Systems`, `Nsight Compute`, `Host`, `Orin`, `TensorRT Plugin`, `KV Cache`, `backpressure`, `CMake`, and `Compute Sanitizer` in the new guide; expect exit code 0.
- [ ] Run `git diff --check`; expect no whitespace errors.
- [ ] Review all new relative links and official URLs, stage only the six intended files, run `git diff --cached --check`, and commit with `docs: add dual-track inference skill stack`.
- [ ] Rerun the required-file check and `git status --short`; expect a clean parent worktree.
