# Orin NX Multi-Model Inference Knowledge Base Design

## Context

This repository collects the current research and future implementation plan for
multi-model inference on an NVIDIA Jetson Orin NX 16GB. The target workload is
broader than LLM serving: real-time CV models, language encoders, LLMs, and VLMs
may coexist and compete for CPU, GPU, unified memory, video buffers, and memory
bandwidth.

The repository is independent from `RK_LLM`, `S100_VLA`, and
`RK_S100_MFSysetem`. Those projects retain their own hardware validation and
integration boundaries.

## Objective

Create a Chinese-language technical knowledge base that:

- maps current inference-related job requirements to concrete skills;
- explains the Orin NX hardware and software stack;
- distinguishes Linux kernel work from CUDA kernels and model operators;
- documents multi-model concurrency, scheduling, isolation, and measurement;
- provides a reproducible host-to-Orin learning environment installation guide;
- curates official documentation and useful open-source repositories;
- defines a four-to-five-month learning roadmap and a future flagship project;
- can later grow into an implementation repository without premature empty
  source directories.

## Initial Repository Structure

```text
Orin_NX_Multi_Model_Inference/
├── README.md
├── docs/
│   ├── 01-positioning-and-goals.md
│   ├── 02-job-skill-map.md
│   ├── 03-orin-platform-and-stack.md
│   ├── 04-cuda-kernel-and-operator-optimization.md
│   ├── 05-multi-model-concurrency-architecture.md
│   ├── 06-learning-roadmap.md
│   ├── 07-project-blueprint-and-acceptance.md
│   ├── 08-environment-setup.md
│   └── superpowers/specs/
│       └── 2026-08-19-orin-nx-multi-model-inference-knowledge-base-design.md
└── references/
    ├── github-projects.md
    └── official-resources.md
```

`src/`, `tests/`, `configs/`, `benchmarks/`, and container files are deferred
until the first implementation milestone is selected. This avoids implying
that a runnable inference system already exists.

## Document Responsibilities

### README

Provide the repository purpose, audience, scope boundary, reading order, and a
compact status table. It must state that current content is research and design,
not Orin NX performance evidence.

### Positioning and Job Skill Map

Explain why edge heterogeneous inference is a better description than pure LLM
serving. Separate three role families: edge deployment and performance,
inference framework development, and cluster inference platform engineering.
Map recurring JD requirements to study topics and portfolio evidence.

### Platform and Technical Stack

Describe the Orin NX 16GB unified-memory constraint and the roles of JetPack,
CUDA, TensorRT, DeepStream, TensorRT Edge-LLM, and Triton Inference Server.
Clearly distinguish deployment targets from source-reading references such as
vLLM and mainline TensorRT-LLM.

### CUDA Kernel and Operator Optimization

Differentiate the Linux kernel, CUDA kernels, graph operators, and TensorRT
plugins. Define the required learning depth: profiling, streams, events, graphs,
memory hierarchy, a fused preprocessing kernel, and one TensorRT plugin before
advanced CUTLASS, PTX, or SASS work.

### Multi-Model Concurrency Architecture

Cover request, model, pipeline, and heterogeneous concurrency. Describe shared
memory-bandwidth contention, model/context pools, queueing, priorities,
deadlines, batching, backpressure, admission control, and failure handling.
Use a real-time CV workload plus bursty NLP and background LLM/VLM workload as
the reference scenario.

### Roadmap and Project Blueprint

Provide a staged four-to-five-month path from reproducible baselines through
TensorRT/CUDA, quantization, multi-model scheduling, serving, profiling, and
stability testing. The project blueprint must define measurable acceptance
criteria rather than claim production readiness from a successful demo.

### Learning Environment Setup

Provide two explicit setup paths: a native JetPack baseline and an isolated
container-based workspace. Cover the Ubuntu x86_64 host, SDK Manager/direct
flash, Orin post-flash verification, storage planning, version pinning, Python
environment boundaries, container runtime, profiling tools, optional serving
components, and health checks. Commands must be labeled by execution location
(`Host` or `Orin`) and must not assume that x86_64 packages or containers run on
the aarch64 target.

### References

Separate official documentation from GitHub projects. Every external resource
must include its purpose, recommended reading area, platform role, and a note
when it is intended only for source study. Time-sensitive compatibility claims
must include a verification date.

## Reference Architecture

```text
camera/video -> DeepStream/GStreamer -> TensorRT CV engines --+
                                                             |
text requests -> TensorRT NLP engines -----------------------+-> scheduler
                                                             |      |
LLM/VLM requests -> TensorRT Edge-LLM -----------------------+      v
                                                        metrics and evidence
```

The future implementation will compare direct TensorRT C++, DeepStream,
Triton-managed models, and a lightweight project-owned scheduler. Triton is a
serving and scheduling layer, not a replacement for TensorRT model execution.

## Evidence and Quality Rules

- Distinguish documentation, mock results, host results, and Orin hardware
  evidence.
- Pin JetPack, CUDA, TensorRT, DeepStream, model, and repository revisions for
  accepted benchmarks.
- Record TTFT, TPOT, throughput, p50/p95/p99 latency, dropped frames, memory,
  power, temperature, and thermal throttling where applicable.
- Compare isolated model performance with concurrent performance.
- Include accuracy or task-quality regression for every quantization result.
- Do not claim an optimization without a reproducible before/after experiment.
- Prefer official NVIDIA sources for compatibility and API claims.

## Future Implementation Trigger

Implementation directories are added only after selecting the first bounded
milestone. The recommended first milestone is a reproducible Orin baseline with
one TensorRT CV engine, device telemetry, and a benchmark harness. Multi-model
scheduling follows only after the single-model baseline is stable and measured.
