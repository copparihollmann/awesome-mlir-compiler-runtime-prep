# awesome-mlir-compiler-runtime-prep

> A curated, audited, interview-focused list for **MLIR / LLVM / compiler / runtime / Triton / GPU codegen** preparation.

This list is **not** just “awesome MLIR”. That name would be too narrow for the links here.

- IR design and compiler architecture
- MLIR dialects, rewrites, lowering, bufferization, codegen
- runtime concerns: dispatch, memory, dynamic shapes, scheduling
- GPU and Triton layout reasoning
- LLVM / SSA / backend / toolchain fundamentals
- low-precision deployment tradeoffs

So the repo name stays broad on purpose:

**`awesome-mlir-compiler-runtime-prep`**

## How to use this list

### Priority legend

- **P0 — Core**: highest ROI for most MLIR/compiler/runtime interviews
- **P1 — Role-dependent**: very strong if the role mentions runtime, GPU, Triton, LLVM backend, or performance
- **P2 — Stretch / adjacent**: useful for standing out, but not first-pass material

### Coverage tags

- **MLIR Core** — dialects, passes, conversion, IR structure
- **Runtime** — dispatch, host/device split, memory, execution model
- **GPU/Triton** — layout, kernels, SIMT/SIMD reasoning
- **LLVM/Backend** — SSA, LLVM IR, assembly, triples, linker/toolchain
- **Quantization** — FP8/FP4/PTQ deployment context
- **Adjacent** — helpful background, but not central interview prep
- **Perf/Arch** — caches, locality, SIMD, microarchitecture, profiling, bottleneck analysis
- **Kernel Design** — tiling, GEMM hierarchy, packing, dataflow, instruction scheduling

### Suggested order

1. Do **P0 / MLIR Core** first.
2. Add **P0 / Runtime** next.
3. Choose one specialization:
   - **GPU/Triton** for accelerator and kernel roles
   - **LLVM/Backend** for lower-level compiler roles
   - **Quantization** if deployment and datatype support matters
4. If you are targeting performance-heavy roles, add **Perf/Arch** and **Kernel Design** before diving too deep into P2.
5. Use **P2** only after the above is solid.

---

## Fast study paths

### 5–7 day crash course

1. Chip Huyen — big picture
2. Stephen Diehl — MLIR introduction
3. Jeremy Kun MLIR tutorial repo
4. Lei — CodeGen Dialects, Linalg, Vector
5. Stephen Diehl — Memory in MLIR
6. Lei — Single-node ML Runtime Foundation
7. IREE — Data Tiling Walkthrough, Microkernels
8. Miguel Young — Why SSA?, LLVM IR, Target Triples

### 2–3 week interview track

Finish all P0 items, then choose one branch:

- **Runtime-heavy**: IREE posts, ApX course, Atomicless Concurrency
- **GPU-heavy**: Triton layout series, Gluon, tiny-gpu-compiler, SIMD article
- **Backend-heavy**: Assembly, linker scripts, pointers, target triples

### What to skip on a first pass

These are good resources, but **not** first-pass interview essentials:

- Neural Networks / Transformers explainers
- Typechecker Zoo
- Formatters
- Best C++ Library
- Curta
- Matrix post
- tuple / SAM closures unless the role is very language-implementation-heavy

---

## 1) Core mental model: what ML compilers and runtimes are doing

These are the best starting points for forming the right mental map.

- **[P0] [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)**  
  **Coverage:** MLIR Core, Runtime  
  **Why it matters:** Best big-picture intro in the set. Use this before learning dialect names.

- **[P0] [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Clear overview of LLVM, MLIR, modern compiler pipelines, and how MLIR fits as a multi-level IR.

- **[P0] [MLIR CodeGen Dialects for Machine Learning Compilers](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** One of the strongest “map of the territory” articles. Explains the dialect hierarchy from model-level representations down to kernel codegen.

- **[P1] [A Meticulous Guide to Advances in Deep Learning Efficiency over the Years](https://alexzhang13.github.io/blog/2024/efficient-dl/)**  
  **Coverage:** Adjacent, GPU/Triton, Quantization  
  **Why it matters:** Excellent macroscopic context for how kernels, compilers, libraries, hardware, and deployment evolved together. Great context, but broader than interview-core.

- **[P1] [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)**  
  **Coverage:** Runtime, GPU/Triton, Adjacent  
  **Why it matters:** Strong systems/performance context: rooflines, TPUs, sharded matmuls, inference, serving, profiling, and JAX.

---

## 2) Hands-on MLIR onboarding

These are the best practical starting points if you want to build intuition by doing.

- **[P0] [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial#mlir-tutorial)**  
  **Coverage:** MLIR Core  
  **Why it matters:** The best step-by-step beginner repo in this list. Covers setup, lowerings, passes, TableGen, dialect definition, traits, verifiers, canonicalization, rewrites, dialect conversion, LLVM lowering, and PDLL.

- **[P0] [MLIR Beginner-Friendly Tutorial: Part 1](https://www.youtube.com/watch?v=Uno_XhtkT5E)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Beginner-friendly video counterpart to the written material; good first watch before diving into custom dialects.

- **[P1] [MLIR YouTube playlist](https://www.youtube.com/watch?v=tW8FVmQBYPk&list=PLlKj-4rp1Gz1Vt3onHU_SLjDwn51UsXrG)**  
  **Coverage:** MLIR Core  
  **Why it matters:** More advanced than the beginner video. Good reinforcement after you know the basics; includes setup, codegen, JIT, and API-oriented walkthroughs.

- **[P1] [tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** Excellent educational project showing a full MLIR-based path from a C-like GPU kernel language to binary instructions and a cycle-accurate simulator. Great for people who learn best from end-to-end systems.

---

## 3) Core MLIR mechanics: dialects, transformations, and lowering

This is the heart of the list.

### Must-read core

- **[P0] [MLIR Linalg Dialect and Patterns](https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Essential for structured ops, tiling, fusion, distribution, vectorization, and lowering to loops.

- **[P0] [MLIR Vector Dialect and Patterns](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** Essential for understanding the vector level in progressive lowering: static tiles, vector ops, register-level reasoning, and lowering choices such as `vector.contract` strategies.

- **[P0] [Memory in MLIR](https://www.stephendiehl.com/posts/mlir_memory/)**  
  **Coverage:** MLIR Core, Runtime  
  **Why it matters:** High interview ROI. Explains tensors vs memrefs vs low-level memory forms — one of the most common places candidates are shaky.

### Strong follow-ups

- **[P1] [Affine Dialect and OpenMP](https://www.stephendiehl.com/posts/mlir_affine/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Valuable for loop nests, affine maps, polyhedral-style reasoning, parallelization, and dependency analysis.

- **[P1] [Linear Algebra in MLIR](https://www.stephendiehl.com/posts/mlir_linear_algebra/)**  
  **Coverage:** MLIR Core  
  **Why it matters:** Good second source for Linalg, especially `linalg.generic`, iterator types, indexing maps, reduction structure, and named ops.

- **[P1] [Specializing Python with E-graphs](https://www.stephendiehl.com/posts/mlir_egraphs/)**  
  **Coverage:** MLIR Core, Adjacent  
  **Why it matters:** Strong for rewrite systems and equality saturation. Not mandatory for most interviews, but very useful if the role mentions transformations or superoptimization-like reasoning.

- **[P1] [GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)**  
  **Coverage:** MLIR Core, GPU/Triton  
  **Why it matters:** End-to-end toy compiler path that reaches PTX. Best used after basic MLIR familiarity.

### Adjacent, not first-pass core

- **[P2] [Neural Networks](https://www.stephendiehl.com/posts/mlir_neural_networks/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Helpful for people who want model-side intuition, but this is not directly compiler prep.

- **[P2] [Transformers](https://www.stephendiehl.com/posts/mlir_transformers/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Good model context, not compiler-core.

- **[P2] [Typechecker Zoo](https://www.stephendiehl.com/posts/typechecker_zoo/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** Great language-implementation material, but it should not be ranked alongside core MLIR runtime prep.

- **[P2] [Stephen Diehl compiler tag page](https://www.stephendiehl.com/tags/compilers/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Useful as an index page, not as a primary resource itself.

---

## 4) Runtime systems and IREE-style deployment

This is the part many candidates under-prepare.

- **[P0] [Single-node ML Runtime Foundation](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)**  
  **Coverage:** Runtime  
  **Why it matters:** Best runtime bridge in the list. Covers the single-node runtime problem, host/device separation, resource management, dispatching work to accelerators, and uses IREE as a concrete example.

- **[P0] [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)**  
  **Coverage:** Runtime, MLIR Core  
  **Why it matters:** A very concrete real-world compiler walkthrough. Explains IREE’s encoding dialect, data tiling, fusion path, and how encodings are attached to matmul and propagated through the pipeline.

- **[P0] [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)**  
  **Coverage:** Runtime, MLIR Core, LLVM/Backend  
  **Why it matters:** One of the best articles here for connecting compiler lowering to runtime execution. Explains rewriting `linalg.matmul` to `linalg.mmt4d`, then to a ukernel op, then LLVM IR, bitcode linking, and target-specific inner loops.

- **[P1] [ML Compiler & Runtime Optimization Techniques (ApX)](https://apxml.com/courses/compiler-runtime-optimization-ml)**  
  **Coverage:** Runtime, MLIR Core, Quantization  
  **Why it matters:** Broad guided course. Good as a structured syllabus, but not as canonical as the P0 reads above.

### Interview questions this section helps with

- Where should compilation stop and the runtime take over?
- How do host and device responsibilities split?
- What happens to layouts, encodings, and kernel selection after lowering?
- How do dynamic shapes, resource management, and scheduling affect performance?

---

## 5) GPU, Triton, layouts, and explicit performance control

This branch is extremely valuable for GPU compiler and accelerator roles.

### Best GPU/Triton sequence

1. Triton Linear Layout: Concept  
2. Triton Linear Layout: Examples  
3. Triton Bespoke Layouts  
4. Gluon: Explicit Performance  
5. tiny-gpu-compiler  
6. GPU Compilation with MLIR  
7. Designing a SIMD Algorithm from Scratch

### Core GPU/Triton resources

- **[P1] [Triton Linear Layout: Concept](https://www.lei.chat/posts/triton-linear-layout-concept/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Best conceptual introduction to Triton linear layout. Focuses on GPU hierarchy, access patterns, and why a unified layout representation helps codegen.

- **[P1] [Triton Linear Layout: Examples](https://www.lei.chat/posts/triton-linear-layout-examples/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Turns the conceptual article into concrete layout constructions and operations. Best read immediately after the Concept post.

- **[P1] [Triton Bespoke Layouts](https://www.lei.chat/posts/triton-bespoke-layouts/)**  
  **Coverage:** GPU/Triton  
  **Why it matters:** Useful for understanding blocked/shared/MMA-style layouts and how warp/thread arrangements are expressed.

- **[P1] [Gluon: Explicit Performance](https://www.lei.chat/posts/gluon-explicit-performance/)**  
  **Coverage:** GPU/Triton, Runtime  
  **Why it matters:** Best for reasoning about portability vs performance and why domain-specific systems sometimes expose explicit controls instead of hiding them behind generic abstractions.

### Useful companions

- **[P1] [GPU Compilation with MLIR](https://www.stephendiehl.com/posts/mlir_gpu/)**  
  **Coverage:** GPU/Triton, MLIR Core  
  **Why it matters:** Toy pipeline to PTX; useful for “how would you build a small GPU compiler?” conversations.

- **[P1] [tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/)**  
  **Coverage:** GPU/Triton, MLIR Core  
  **Why it matters:** Especially strong for intuition-building.

---

## 6) LLVM, SSA, backend, and toolchain fundamentals

This is where strong candidates separate themselves from “I only know passes” candidates.

### Highest ROI backend reads

- **[P0] [Why SSA?](https://mcyoung.xyz/2025/10/21/ssa-1/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Outstanding explanation of SSA as a compiler architecture, including phi-style reasoning and CFG structure.

- **[P0] [A Gentle Introduction to LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Best practical introduction to reading LLVM IR without drowning in metadata.

- **[P1] [What the Hell Is a Target Triple?](https://mcyoung.xyz/2025/04/14/target-triples/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Very useful for backend and deployment interviews. Helps decode architecture / vendor / OS / ABI naming and what those strings really mean.

### Strong follow-ups

- **[P1] [Understanding Assembly, Part I: RISC-V](https://mcyoung.xyz/2021/11/29/assembly-1/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Excellent for control flow, branches, loops, and what IR eventually becomes.

- **[P1] [Designing a SIMD Algorithm from Scratch](https://mcyoung.xyz/2023/11/27/simd-base64/)**  
  **Coverage:** LLVM/Backend, GPU/Triton  
  **Why it matters:** Superb performance-thinking resource. Helps with vectorization, lane-wise reasoning, and “thinking like hardware”.

- **[P1] [The Taxonomy of Pointers](https://mcyoung.xyz/2021/05/24/ptr-taxonomy/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Good for low-level semantics, ownership, pointer representations, and memory model intuition.

- **[P2] [Everything You Never Wanted To Know About Linker Script](https://mcyoung.xyz/2021/06/01/linker-script/)**  
  **Coverage:** LLVM/Backend  
  **Why it matters:** Great if the role touches embedded, binaries, loaders, or deployment. Too deep for a first pass.

- **[P2] [Atomicless Concurrency](https://mcyoung.xyz/2023/03/29/rseq-checkout/)**  
  **Coverage:** Runtime, LLVM/Backend  
  **Why it matters:** Strong runtime-systems depth; good differentiator for systems-heavy roles.

### Adjacent / specialized backend reads

- **[P2] [What is a Matrix? A Miserable Pile of Coefficients!](https://mcyoung.xyz/2023/09/29/what-is-a-matrix/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Nice math intuition, but not interview-core.

- **[P2] [Single Abstract Method Traits](https://mcyoung.xyz/2023/05/11/sam-closures/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** More language-design and closure-lowering flavor than MLIR/runtime prep.

- **[P2] [std::tuple the Hard Way](https://mcyoung.xyz/2022/07/13/tuples-the-hard-way/)**  
  **Coverage:** Adjacent, LLVM/Backend  
  **Why it matters:** Useful for advanced C++ metaprogramming fluency, not core interview prep.

- **[P2] [The Art of Formatting Code](https://mcyoung.xyz/2025/03/11/formatters/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Interesting tooling/front-end article, not central to MLIR compiler/runtime prep.

- **[P2] [The Best C++ Library](https://mcyoung.xyz/2025/07/14/best/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Good engineering/craftsmanship essay, but should not be treated as interview-core.

- **[P2] [3Hz Computer, Hold the Transistors](https://mcyoung.xyz/2022/07/24/curta/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Fun and insightful, but far from core prep.

---

## 7) Quantization and low precision

Important for deployment-aware roles, but not part of the absolute minimum MLIR/compiler core.

- **[P1] [Optimizing LLMs for Performance and Accuracy with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)**  
  **Coverage:** Quantization  
  **Why it matters:** Practical PTQ overview. Good for calibration-aware discussions and modern schemes such as SmoothQuant, AWQ, AutoQuantize, and deployment-oriented tradeoffs.

- **[P1] [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)**  
  **Coverage:** Quantization  
  **Why it matters:** Best overview here for FP8 formats and where they fit in modern training stacks.

- **[P2] [Improving Quantized FP4 Weight Quality via Logit Distillation](https://dropbox.github.io/fp4_blogpost/)**  
  **Coverage:** Quantization  
  **Why it matters:** More advanced and more specialized. Strong if you want to discuss FP4 recovery methods and post-hoc correction, but not required for most interviews.

### Best use of this section

Read this section after the compiler/runtime fundamentals, not before.

---

## 8) Role-based reading tracks

### A. MLIR compiler engineer

- Chip Huyen intro
- Stephen Diehl intro
- Jeremy Kun tutorial repo
- Lei: CodeGen Dialects
- Lei: Linalg
- Lei: Vector
- Stephen Diehl: Memory in MLIR
- IREE Data Tiling
- Why SSA?
- LLVM IR

### B. MLIR + runtime / deployment role

- Lei: Single-node Runtime Foundation
- IREE Data Tiling
- IREE Microkernels
- Stephen Diehl: Memory in MLIR
- ApX course
- Target Triples
- Atomicless Concurrency

### C. GPU / Triton / kernel codegen role

- Lei: Vector
- Triton Linear Layout: Concept
- Triton Linear Layout: Examples
- Triton Bespoke Layouts
- Gluon: Explicit Performance
- tiny-gpu-compiler
- GPU Compilation with MLIR
- SIMD article

### D. LLVM / backend-heavy role

- Why SSA?
- LLVM IR
- Target Triples
- Assembly
- Pointers
- Linker Script
- IREE Microkernels

### E. Low-precision deployment-aware role

- NVIDIA PTQ post
- NVIDIA FP8 post
- Dropbox FP4 post
- IREE Microkernels
- Triton / GPU layout resources

---

## 9) Minimal shortlist

If you only have time for a very small subset, read these first:

- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial#mlir-tutorial)
- [MLIR Beginner-Friendly Tutorial: Part 1](https://www.youtube.com/watch?v=Uno_XhtkT5E)
- [MLIR CodeGen Dialects for Machine Learning Compilers](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)
- [MLIR Linalg Dialect and Patterns](https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/)
- [MLIR Vector Dialect and Patterns](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)
- [Memory in MLIR](https://www.stephendiehl.com/posts/mlir_memory/)
- [Single-node ML Runtime Foundation](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)
- [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)
- [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)
- [Why SSA?](https://mcyoung.xyz/2025/10/21/ssa-1/)
- [A Gentle Introduction to LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)
- [What the Hell Is a Target Triple?](https://mcyoung.xyz/2025/04/14/target-triples/)


---

## 10) Performance engineering, microarchitecture, memory hierarchy, and kernel design

This section is for the part of compiler/runtime work that is easiest to under-study: **how caches, pipelines, SIMD, memory layout, tiling, and instruction scheduling actually shape software and compiler design**.

For many MLIR/compiler interviews these reads are more **role-dependent** than the MLIR/IREE core, but for performance-heavy roles they are some of the highest-value resources in the whole list.

### Best first-pass sequence for this section

1. Drepper / LWN — memory hierarchy foundation  
2. Johnny’s Software Lab — cache locality from the programmer’s point of view  
3. Algorithmica — CPU Caches, SIMD, Computer Architecture, Pipelining, Profiling  
4. Easyperf — Vectorization, Performance Vocabulary, Better Code Locality, Microarchitectural Benchmarking  
5. CUTLASS Efficient GEMM — hierarchical tiling on GPUs  
6. IREE Data Tiling + IREE Microkernels — compiler/runtime bridge back into MLIR/IREE  
7. Travis Downs / GPUOpen / NVIDIA CUDA compile-time post — specialized follow-ups

### A. Foundational memory and cache mental models

- **[P0] [What every programmer should know about memory, Part 1](https://lwn.net/Articles/250967/)**  
  **Coverage:** Perf/Arch, Runtime  
  **Why it matters:** The classic memory-hierarchy primer. Best first read for caches, latency gaps, and why layout and access order dominate real performance.

- **[P0] [Make your programs run faster by better using the data cache](https://johnnysswlab.com/make-your-programs-run-faster-by-better-using-the-data-cache/)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** One of the best software-facing cache articles. Strong on spatial/temporal locality, cache lines, and why arrays, layout, and access patterns shape performance.

- **[P1] [Why do CPUs have multiple cache levels?](https://fgiesen.wordpress.com/2016/08/07/why-do-cpus-have-multiple-cache-levels/)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Great intuition builder for why cache hierarchies look the way they do, and why software should care about L1/L2/L3 separately instead of thinking about “cache” as one thing.

- **[P1] [Preface to the CPU performance optimization guide](https://gpuopen.com/learn/cpu-performance-guide/cpu-performance-guide-preface/)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Good vendor-backed on-ramp to practical CPU performance work, especially if you want a broad guide before getting lost in narrower posts.

### B. Algorithmica / Algorithms for Modern Hardware

These are best read as a **cluster**, not as isolated posts. The P0 items below are the best entry points; the rest are strong follow-ups depending on your interests.

- **[P1] [Complexity Models](https://en.algorithmica.org/hpc/complexity/)**  
  **Coverage:** Perf/Arch, Adjacent  
  **Why it matters:** Good bridge from asymptotic complexity to machine-aware cost models. Helpful when you want to reason about why the same `O(n^3)` algorithm can differ wildly in real runtime.

- **[P0] [Computer Architecture](https://en.algorithmica.org/hpc/architecture/)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Strong overview of the machine model you actually need for performance work: superscalar execution, pipelines, throughput, and hardware limits.

- **[P0] [Instruction-Level Parallelism](https://en.algorithmica.org/hpc/pipelining/)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Great for understanding dependency chains, overlap, and why instruction scheduling matters for compilers and hand-written kernels.

- **[P1] [Compilation](https://en.algorithmica.org/hpc/compilation/)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Useful companion for understanding what the compiler can and cannot do for you before you reach for intrinsics or assembly.

- **[P0] [Profiling](https://en.algorithmica.org/hpc/profiling/)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** High ROI. Good profiling habits are what separate serious optimization work from random loop rewriting.

- **[P1] [Arithmetic](https://en.algorithmica.org/hpc/arithmetic/)**  
  **Coverage:** Perf/Arch, Adjacent  
  **Why it matters:** Strong on floating-point behavior and arithmetic-level caveats that show up once you start optimizing aggressively.

- **[P2] [Number Theory](https://en.algorithmica.org/hpc/number-theory/)**  
  **Coverage:** Adjacent  
  **Why it matters:** Interesting case-study area for performance thinking, but far from first-pass compiler/runtime prep.

- **[P1] [External Memory](https://en.algorithmica.org/hpc/external-memory/)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** Good for stepping beyond caches into IO-aware and memory-hierarchy-aware algorithm design.

- **[P0] [RAM & CPU Caches](https://en.algorithmica.org/hpc/cpu-cache/)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** One of the strongest resources here for working sets, locality, and why loop order and layout change performance so dramatically.

- **[P0] [SIMD Parallelism](https://en.algorithmica.org/hpc/simd/)**  
  **Coverage:** Perf/Arch, LLVM/Backend, Kernel Design  
  **Why it matters:** Essential for vectorization-aware compiler and kernel work. Best read alongside Easyperf’s vectorization series and assembly inspection.

- **[P1] [Algorithms Case Studies](https://en.algorithmica.org/hpc/algorithms/)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** Good way to see how memory hierarchy, layout, and low-level effects influence algorithm design rather than just implementation.

- **[P1] [Data Structures Case Studies](https://en.algorithmica.org/hpc/data-structures/)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** Useful for understanding locality-friendly versus locality-hostile data structure choices and why they matter in real systems code.

### C. Easyperf / practical microarchitecture and measurement

Easyperf is one of the best “measure first, reason second, then tune” resources on the internet. The sequence below is the most useful order for this list.

- **[P0] [Vectorization part1. Intro.](https://easyperf.net/blog/2017/10/24/Vectorization_part1)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Great first stop for thinking about SIMD from the optimizer’s and programmer’s point of view.

- **[P1] [Code alignment issues.](https://easyperf.net/blog/2018/01/18/Code_alignment_issues)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Good example of how front-end effects and code layout matter even when the algorithm stays the same.

- **[P1] [Code alignment options in llvm.](https://easyperf.net/blog/2018/01/25/Code_alignment_options_in_llvm)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Nice compiler-facing follow-up to code alignment issues; useful if you want to connect performance symptoms back to compiler flags and backend behavior.

- **[P1] [Microbenchmarking fused instruction.](https://easyperf.net/blog/2018/02/04/Micro-ops-fusion)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Good for learning that “one instruction” is not the right level of abstraction for modern CPUs.

- **[P1] [Store forwarding by example.](https://easyperf.net/blog/2018/03/09/Store-forwarding)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Excellent concrete example of a microarchitectural effect that matters a lot in tight loops and hand-tuned kernels.

- **[P1] [Understanding CPU port contention.](https://easyperf.net/blog/2018/03/21/port-contention)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Very useful once you start reading assembly and trying to understand why throughput does not match naive instruction counts.

- **[P0] [Tools for microarchitectural benchmarking.](https://easyperf.net/blog/2018/04/03/Tools-for-microarchitectural-benchmarking)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** High ROI. Shows the tool stack you need to do serious low-level investigation instead of guesswork.

- **[P1] [What optimizations you can expect from CPU?](https://easyperf.net/blog/2018/04/22/What-optimizations-you-can-expect-from-CPU)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Useful reset on what hardware gives you “for free” and what still needs software/compiler help.

- **[P0] [Improving performance by better code locality.](https://easyperf.net/blog/2018/07/09/Improving-performance-by-better-code-locality)**  
  **Coverage:** Perf/Arch, Kernel Design  
  **Why it matters:** One of the best short pieces on why locality is a software design concern, not just a hardware curiosity.

- **[P0] [Performance analysis vocabulary.](https://easyperf.net/blog/2018/09/04/Performance-Analysis-Vocabulary)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Great glossary for speaking precisely about bottlenecks, front-end vs back-end issues, latency, throughput, and analysis methodology.

- **[P1] [Performance analysis of multithreaded applications.](https://easyperf.net/blog/2019/10/05/Performance-Analysis-Of-MT-apps)**  
  **Coverage:** Perf/Arch, Runtime  
  **Why it matters:** Strong follow-up once you start thinking about thread-level scaling, synchronization, and real runtime behavior.

- **[P1] [How to find expensive locks in multithreaded application.](https://easyperf.net/blog/2019/10/12/MT-Perf-Analysis-part2)**  
  **Coverage:** Perf/Arch, Runtime  
  **Why it matters:** Useful for lock contention, runtime bottlenecks, and practical profiling of multithreaded programs.

- **[P1] [Data-Driven tuning. Specialize switch with one hot case.](https://easyperf.net/blog/2019/11/22/data-driven-tuning-specialize-switch)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Nice case study in profile-aware specialization and branch-shaping for the hot path.

- **[P1] [Data-Driven tuning. Specialize indirect call.](https://easyperf.net/blog/2019/11/27/data-driven-tuning-specialize-indirect-call)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Good companion to the switch post; useful when reasoning about devirtualization-like effects and hot-path specialization.

- **[P1] [HW and SW rules of thumb.](https://easyperf.net/blog/2020/04/01/HW-SW-rules-of-thumb)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Short, useful set of heuristics to calibrate performance intuition.

- **[P0] [Memory Profiling Part 1. Introduction](https://easyperf.net/blog/2024/02/12/Memory-Profiling-Part1)**  
  **Coverage:** Perf/Arch, Runtime  
  **Why it matters:** Very strong recent entry point for memory profiling specifically, which is exactly where many compiler/runtime candidates remain weak.

- **[P1] [Thread Count Scaling Part 1. Introduction](https://easyperf.net/blog/2024/05/10/Thread-Count-Scaling-Part1)**  
  **Coverage:** Perf/Arch, Runtime  
  **Why it matters:** Good modern systems/perf follow-up once you start thinking about scaling behavior instead of single-core speed only.

- **[P0] [Performance Analysis and Tuning on Modern CPUs, 2nd edition](https://github.com/dendibakh/perf-book/releases/tag/2.0_release)**  
  **Coverage:** Perf/Arch, LLVM/Backend  
  **Why it matters:** Best long-form companion to the blog. Use it as a structured reference after the core posts above.

### D. GPU tiling, kernel hierarchy, and compiler-facing GEMM design

- **[P0] [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)**  
  **Coverage:** GPU/Triton, Kernel Design  
  **Why it matters:** One of the best resources for understanding hierarchical tiling on GPUs: threadblock tiles, warp tiles, thread tiles, and how memory hierarchy shapes kernel structure.

- **[P1] [Optimizing Compile Times for CUDA C++](https://developer.nvidia.com/blog/optimizing-compile-times-for-cuda-c/)**  
  **Coverage:** GPU/Triton, LLVM/Backend  
  **Why it matters:** Good compiler-engineering complement to the kernel-design material. Useful for thinking about build-time cost, template-heavy code, and toolchain behavior in CUDA-heavy stacks.

### E. Narrow but very useful follow-up

- **[P1] [Hardware Store Elimination](https://travisdowns.github.io/blog/2020/05/13/intel-zero-opt.html#l1-and-l2)**  
  **Coverage:** Perf/Arch  
  **Why it matters:** Strong example of how to reason from benchmark behavior to cache-level and microarchitectural explanations. Best read after you already know the basics.

### How this section connects back to the core compiler/runtime list

Use this section to answer the questions the MLIR/IREE posts naturally lead to but do not fully teach on their own:

- Why does a particular tile size help?  
- Why does one loop order vectorize and another one not?  
- Why does packing help even when the FLOP count stays the same?  
- Why does a compiler-generated loop still underperform a hand-tuned ukernel?  
- Why do cache lines, port pressure, and dependency chains change the best lowering?

---

## 11) Additional role-based reading track

### F. Performance / microarchitecture / kernel-engineering role

- Drepper / LWN memory article
- Johnny’s Software Lab cache article
- Algorithmica: Computer Architecture
- Algorithmica: RAM & CPU Caches
- Algorithmica: SIMD Parallelism
- Algorithmica: Instruction-Level Parallelism
- Easyperf: Vectorization part1
- Easyperf: Performance Analysis Vocabulary
- Easyperf: Better Code Locality
- Easyperf: Tools for Microarchitectural Benchmarking
- Easyperf: Memory Profiling Part 1
- CUTLASS: Efficient GEMM in CUDA
- IREE Data Tiling
- IREE Microkernels
- Easyperf: Port Contention / Store Forwarding
- Travis Downs: Hardware Store Elimination

---

## 12) Performance-heavy shortlist

If you already know the MLIR/IREE basics and want the smallest high-signal set for **performance-oriented compiler / runtime / kernel work**, read these next:

- [What every programmer should know about memory, Part 1](https://lwn.net/Articles/250967/)
- [Make your programs run faster by better using the data cache](https://johnnysswlab.com/make-your-programs-run-faster-by-better-using-the-data-cache/)
- [RAM & CPU Caches](https://en.algorithmica.org/hpc/cpu-cache/)
- [SIMD Parallelism](https://en.algorithmica.org/hpc/simd/)
- [Instruction-Level Parallelism](https://en.algorithmica.org/hpc/pipelining/)
- [Vectorization part1. Intro.](https://easyperf.net/blog/2017/10/24/Vectorization_part1)
- [Tools for microarchitectural benchmarking.](https://easyperf.net/blog/2018/04/03/Tools-for-microarchitectural-benchmarking)
- [Improving performance by better code locality.](https://easyperf.net/blog/2018/07/09/Improving-performance-by-better-code-locality)
- [Memory Profiling Part 1. Introduction](https://easyperf.net/blog/2024/02/12/Memory-Profiling-Part1)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)
- [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)


---

## 13) Supplemental systems, threads, OS, register allocation, and GPU register-pressure resources

These are **good additions**, but they are mostly **supporting material** rather than core MLIR/IREE/compiler-runtime prep. They fit best once the main compiler, runtime, and performance sections are already in place.

### A. Threads, operating systems, and systems background

- **[P2] [Learning About Threads: An Essential Guide for Developers](https://hackernoon.com/learning-about-threads-an-essential-guide-for-developers)**  
  **Coverage:** Runtime, Adjacent  
  **Why it matters:** Broad thread-level background. Useful if you want a lightweight refresher before deeper runtime, synchronization, or scaling material.

- **[P2] [Operating System Tutorial](https://www.geeksforgeeks.org/operating-systems/operating-systems/)**  
  **Coverage:** Runtime, Adjacent  
  **Why it matters:** Broad survey page covering processes, scheduling, synchronization, memory management, paging, virtual memory, and multithreading. Best used as a refresher or gap-filler, not as a primary source.

### B. Register allocation, graph coloring, and register-pressure-aware compiler work

- **[P1] [Controlling Virtual Register Pressure in LLVM Middle-End](https://llvm.org/devmtg/2014-10/Slides/Baev-Controlling_VRP.pdf)**  
  **Coverage:** LLVM/Backend, Perf/Arch  
  **Why it matters:** Strong LLVM-specific talk on how middle-end optimizations such as LICM and GVN can increase virtual register pressure and hurt downstream performance. Very useful once you start connecting optimization profitability to backend constraints.

- **[P2] [A Guide to Graph Colouring](https://rhydlewis.eu/gcol/)**  
  **Coverage:** LLVM/Backend, Adjacent  
  **Why it matters:** Good graph-coloring background if you want to strengthen the theory behind interference graphs and register allocation. More useful as supporting theory than as interview-first reading.

- **[P1] [Coloring Code: How Compilers Use Graph Theory](https://www.youtube.com/watch?v=K3mi2m7ccDQ)**  
  **Coverage:** LLVM/Backend, Adjacent  
  **Why it matters:** Nice visual companion for graph coloring and register-allocation intuition. Best paired with SSA / LLVM IR material and the LLVM register-pressure talk above.

### C. GPU register pressure and occupancy

- **[P1] [What is register pressure?](https://modal.com/gpu-glossary/perf/register-pressure)**  
  **Coverage:** GPU/Triton, Perf/Arch  
  **Why it matters:** Concise modern explanation of GPU register pressure, how virtual registers differ from physical register files, and why register-heavy kernels can reduce occupancy and hurt latency hiding.

### D. Supplemental compiler and graph-coloring video resources

- **[P2] [Graph Coloring](https://www.youtube.com/watch?v=xTbFiXAwjck)**  
  **Coverage:** LLVM/Backend, Adjacent  
  **Why it matters:** Useful companion if you want a quick graph-theory refresher before diving into interference graphs and register-allocation material.

- **[P1] [Register Allocation using Graph Coloring](https://www.youtube.com/watch?v=eWp_-XCwN1A)**  
  **Coverage:** LLVM/Backend, Adjacent  
  **Why it matters:** Good follow-up once you understand basic graph coloring and want to connect it directly to register-allocation strategy.

### Notes on overlap

- The Travis Downs article **[Hardware Store Elimination](https://travisdowns.github.io/blog/2020/05/13/intel-zero-opt.html#l1-and-l2)** is already included above in the performance section, so it does **not** need a duplicate entry.

### Best use of this supplemental section

Use these when you want to round out the edges of the main list:

- threads and OS background for runtime intuition
- graph coloring for register-allocation intuition
- LLVM register-pressure material for optimization-profitability reasoning
- GPU register-pressure material for occupancy-aware kernel design

---

## 14) Final minimal shortlists

If you want the list to stay comprehensive but still have a clean place to start, use one of these two shortlists at the very end.

### A. Minimal compiler / runtime shortlist

- [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
- [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/)
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial#mlir-tutorial)
- [MLIR Beginner-Friendly Tutorial: Part 1](https://www.youtube.com/watch?v=Uno_XhtkT5E)
- [MLIR CodeGen Dialects for Machine Learning Compilers](https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/)
- [MLIR Linalg Dialect and Patterns](https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/)
- [MLIR Vector Dialect and Patterns](https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/)
- [Memory in MLIR](https://www.stephendiehl.com/posts/mlir_memory/)
- [Single-node ML Runtime Foundation](https://www.lei.chat/posts/single-node-ml-runtime-foundation/)
- [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)
- [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)
- [Why SSA?](https://mcyoung.xyz/2025/10/21/ssa-1/)
- [A Gentle Introduction to LLVM IR](https://mcyoung.xyz/2023/08/01/llvm-ir/)
- [What the Hell Is a Target Triple?](https://mcyoung.xyz/2025/04/14/target-triples/)

### B. Minimal performance / kernel shortlist

- [What every programmer should know about memory, Part 1](https://lwn.net/Articles/250967/)
- [Make your programs run faster by better using the data cache](https://johnnysswlab.com/make-your-programs-run-faster-by-better-using-the-data-cache/)
- [Computer Architecture](https://en.algorithmica.org/hpc/architecture/)
- [RAM & CPU Caches](https://en.algorithmica.org/hpc/cpu-cache/)
- [SIMD Parallelism](https://en.algorithmica.org/hpc/simd/)
- [Instruction-Level Parallelism](https://en.algorithmica.org/hpc/pipelining/)
- [Vectorization part1. Intro.](https://easyperf.net/blog/2017/10/24/Vectorization_part1)
- [Tools for microarchitectural benchmarking.](https://easyperf.net/blog/2018/04/03/Tools-for-microarchitectural-benchmarking)
- [Improving performance by better code locality.](https://easyperf.net/blog/2018/07/09/Improving-performance-by-better-code-locality)
- [Memory Profiling Part 1. Introduction](https://easyperf.net/blog/2024/02/12/Memory-Profiling-Part1)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [IREE Data-Tiling Walkthrough](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/data-tiling-walkthrough.md?plain=1)
- [IREE Microkernels](https://github.com/iree-org/iree/blob/main/docs/website/docs/community/blog/posts/microkernels.md)
