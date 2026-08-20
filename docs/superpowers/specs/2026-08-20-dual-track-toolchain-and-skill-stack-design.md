# Dual-Track Inference Toolchain And Skill Stack Design

## Context

The target is to apply for both edge inference performance roles and inference
framework development roles. The primary hardware is Jetson Orin NX 16GB;
RK and S100 remain independent validation platforms. The user already has an
embedded software background with usable C/C++ and Linux experience, but needs a
structured start in CUDA, TensorRT, profiling, model execution, and framework
internals.

The current knowledge base covers these topics, but the requirements are spread
across the job map, platform guide, CUDA guide, concurrency guide, roadmap, and
environment guide. It does not yet provide one lifecycle-oriented toolchain
matrix that distinguishes shared foundations from the two job tracks.

## Goals

- Provide one Markdown entry point for the complete inference engineering
  toolchain and skill stack.
- Treat edge inference performance and inference framework development as two
  primary tracks with a shared foundation.
- Classify each capability by learning depth: use, diagnose, optimize, and
  design/contribute.
- Mark where a tool runs: Host, Orin target, or both.
- Connect each capability to a concrete experiment, artifact, and job evidence.
- Keep the 4-5 month plan realistic for a learner starting CUDA/TensorRT from
  fundamentals.
- Preserve the existing documents as focused references instead of duplicating
  their full content.

## Non-goals

- Do not make Kubernetes, multi-node inference, or distributed training a
  primary requirement for the first 4-5 months.
- Do not claim that a server-oriented framework is a supported Orin runtime
  without a target-specific verification.
- Do not add a new implementation service, benchmark harness, or empty source
  tree as part of this documentation change.
- Do not replace the existing Orin platform, CUDA, concurrency, environment, or
  project acceptance guides.

## Recommended Approach

Create `docs/09-dual-track-toolchain-and-skill-stack.md` as the single entry
point. Update existing navigation and priority statements only where needed:

- `README.md`: add the new guide to the document navigation and quick lookup.
- `docs/02-job-skill-map.md`: explicitly mark both target role families as
  primary and link to the dual-track matrix.
- `docs/06-learning-roadmap.md`: express the roadmap as shared foundations plus
  two parallel evidence tracks.
- `docs/08-environment-setup.md`: add the Host/Orin profiling workflow and
  verification commands at the environment boundary.
- `references/official-resources.md`: add or connect primary documentation for
  the tools introduced by the matrix.

The new guide will not repeat full tutorials. Each row will state what to learn,
why it matters, where it runs, the required depth, a practical exercise, and the
artifact that proves competence.

## Document Structure

### 1. Target profile and role split

Explain the user's starting point and the difference between:

- edge inference performance: end-to-end behavior on a constrained device;
- inference framework development: runtime, scheduling, memory, kernel, and
  framework behavior.

Show their shared 60% foundation and the distinct evidence expected by each
role.

### 2. Lifecycle-oriented toolchain matrix

Organize tools by the work they support:

1. Linux/C++ build and debugging;
2. model export and graph validation;
3. JetPack/CUDA/TensorRT runtime;
4. CUDA correctness and GPU profiling;
5. real-time CV and media pipelines;
6. LLM/VLM runtime, quantization, and KV cache;
7. serving, load generation, and concurrency;
8. containers, reproducibility, observability, and CI;
9. framework source study and advanced kernel work.

The first six are primary for Orin. Cluster tools are included as later context,
not as first-phase prerequisites.

### 3. Depth ladder

Every capability uses the same ladder:

- **Use**: follow documentation and run a known workflow;
- **Diagnose**: inspect logs, traces, metrics, and failure symptoms;
- **Optimize**: make a controlled change and prove a correct improvement;
- **Design/contribute**: make a reusable component, architectural choice, or
  upstream-quality fix.

The required first milestone is depth 2 for most P0 skills, depth 3 for
TensorRT, profiling, C++, CUDA basics, and concurrency, and depth 4 only for
selected project components.

### 4. Host/Orin execution model

Clearly label operations performed on the x86 Host versus the Orin target:

```text
Host: export, compile, Nsight GUI, report analysis, source and artifact storage
Orin: target runtime, ARM64 dependencies, model execution, CLI collection,
      power/thermal and end-to-end measurements
```

Explain Nsight Systems/Compute remote SSH collection and the local CLI plus
report-copy workflow. State that Host results are not Orin performance evidence.

### 5. Shared foundation and two tracks

The shared foundation covers Linux/C++, CMake, model fundamentals, ONNX,
TensorRT basics, benchmark design, telemetry, and Nsight Systems.

The edge track emphasizes TensorRT/DeepStream, target resource budgets, thermal
behavior, multi-model admission and degradation, and deployment reliability.

The framework track emphasizes CUDA memory/streams/graphs, TensorRT Plugin,
runtime lifecycle, scheduler/allocator/KV cache concepts, vLLM/SGLang/
TensorRT-LLM source reading, and selected kernel implementation.

The guide must explicitly state that the tracks share one flagship Orin project
but require different evidence sections and résumé narratives.

### 6. 20-week execution map

Retain the existing phases, but annotate each phase with two deliverables:

- an edge deliverable such as a target baseline, DeepStream pipeline, or
  multi-model resource policy;
- a framework deliverable such as a profiled CUDA component, Plugin, scheduler
  prototype, or source-reading note backed by a small experiment.

The plan must defer PTX/SASS, full CUTLASS kernel work, Kubernetes, and multi-node
serving until the shared foundation and selected P0 experiments are complete.

### 7. JD evidence matrix

Map recurring JD terms to repository artifacts, for example:

- performance analysis -> Nsight reports and before/after explanation;
- C++/runtime -> tested worker, context pool, or lifecycle component;
- CUDA/operator optimization -> correct kernel or TensorRT Plugin;
- batching/concurrency -> controlled saturation and percentile report;
- production discipline -> version manifest, failure handling, soak test, and
  reproducibility guide.

## Verification

Before considering the documentation complete:

- every new tool has a stated purpose, depth, execution location, and evidence;
- the README links to the new guide;
- both role tracks are visible in the job map and roadmap;
- Host/Orin labels are present in the profiling section;
- all referenced official links resolve to primary documentation;
- no new section claims unmeasured Orin performance;
- Markdown formatting passes `git diff --check`;
- the parent repository remains clean after the documentation commit.
