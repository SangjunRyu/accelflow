# AccelFlow — NPU Runtime & Kernel Optimization Portfolio Project

## 0. Project Purpose

This project is a hands-on portfolio project for preparing for **BOS Semiconductors ML Systems Runtime Engineer / AI SW Engineer** roles.

The goal is **not** to claim performance on BOS Eagle-N hardware without access to the chip.

Instead, this project will:

- Use **WSL2 + Ubuntu 24.04**
- Use **Tenstorrent TTSim** to run a virtual Tenstorrent NPU device without physical accelerator hardware
- Use **TT-Metalium / TT-NN** to study and implement NPU-style runtime, kernel, data movement, tiling, buffering, synchronization, and operator execution
- Compare selected operators against equivalent **CUDA implementations on the local NVIDIA GPU**
- Build a small AI workload and trace its execution from high-level model/operator down to lower-level runtime/kernel concepts
- Document optimization reasoning, correctness validation, architecture constraints, and simulator limitations

The intended portfolio message is:

> I can transfer GPU/CUDA optimization concepts such as data locality, tiling, synchronization, pipelining, and memory movement into an NPU runtime/kernel programming environment, analyze execution bottlenecks, and reason about end-to-end AI accelerator software stacks.

---

# 1. Target Roles

## BOS Semiconductors — ML Systems Runtime Engineer

Relevant responsibilities to cover through this project:

- Runtime dispatch path analysis
- Prefetch / synchronization
- Data movement optimization
- Memory utilization
- Pipelining
- Tiling
- Device abstraction
- AI workload latency analysis
- Multi-device / scheduler concepts
- Kernel and operator implementation
- Linux-based system software workflow

## BOS Semiconductors — AI SW Engineer

Relevant responsibilities to cover:

- AI model porting
- Custom operators / kernels
- Runtime engine optimization
- Compiler / graph runtime understanding
- PyTorch model workflow
- Quantization-ready design
- Profiling and execution analysis
- CNN / Transformer operator understanding

---

# 2. Current Development Environment

## Host

- Windows
- WSL2 enabled
- Ubuntu 24.04 installed
- NVIDIA GeForce MX150 available on Windows for CUDA experiments
- CUDA development environment already used for:
  - global memory coalescing
  - shared memory
  - bank conflicts
  - matrix transpose
  - matrix multiplication
  - reductions
  - warp-level operations
  - CUDA streams
  - pinned memory
  - image filtering / histogram examples

## NPU Environment

Primary target:

- WSL2
- Ubuntu 24.04
- x86_64
- TTSim
- TT-Metalium
- TT-NN

Start with a **virtual Wormhole device** unless current upstream compatibility requires a different target.

Do not require physical Tenstorrent silicon for the initial project.

---

# 3. Important Constraint

## TTSim is not a real Eagle-N performance benchmark

TTSim is used for:

- functional validation
- kernel development
- runtime programming experiments
- data movement logic
- synchronization logic
- tiling experiments
- TT-Metal / TT-NN programming model exploration

Do **not** interpret simulator wall-clock runtime as actual BOS Eagle-N hardware latency.

Forbidden portfolio claims:

- "Eagle-N inference latency is X ms"
- "This optimization makes Eagle-N 2x faster"
- "TTSim proves NPU throughput is X TOPS"
- "Simulator runtime equals silicon runtime"

Allowed claims:

- "The kernel produced numerically correct output."
- "The implementation reduced redundant DRAM tile loads."
- "The optimized version reused L1/Circular Buffer data."
- "Synchronization points were reduced."
- "Execution was refactored into a pipelined reader/compute/writer structure."
- "CUDA latency was measured on actual NVIDIA hardware."
- "TTSim was used for NPU programming-model and functional validation."

---

# 4. High-Level Architecture

```text
                          AccelFlow
                              |
            +-----------------+-----------------+
            |                                   |
        CUDA Track                         NPU Track
            |                                   |
      NVIDIA MX150                         TT-Metalium
            |                                   |
   Real GPU execution                         TT-NN
            |                                   |
   Actual timing/profiling                    TTSim
            |                                   |
            |                         Virtual Tenstorrent NPU
            |                                   |
            +-----------------+-----------------+
                              |
                      Common Operators
                              |
       +----------+-----------+-----------+----------+
       |          |           |           |          |
   Vector Add   MatMul     Reduction   RMSNorm     Conv
```

The same operator should be implemented or analyzed through multiple layers where practical.

---

# 5. Core Technical Question

The project should answer:

> How does an AI accelerator software stack move data, schedule work, synchronize computation, and execute operators efficiently?

Sub-questions:

1. Where does the data live?
2. When is data moved?
3. Who initiates the movement?
4. How is work divided into tiles?
5. How are compute and data movement overlapped?
6. Where are synchronization points required?
7. How much intermediate state is stored?
8. What happens when an operator is decomposed across multiple cores?
9. What is different between CUDA's programming model and TT-Metal's programming model?
10. Which optimizations are architecture-specific and which are general accelerator principles?

---

# 6. Project Scope

## Phase 0 — Environment Verification

Goal:

Confirm that the local WSL2 Ubuntu 24.04 environment can build and execute a minimal TTSim + TT-Metalium example.

Tasks:

- Verify OS and architecture
- Verify available RAM / disk
- Install build dependencies
- Clone TTSim
- Clone TT-Metal
- Build required components
- Configure TTSim environment
- Run minimal programming example
- Record all setup steps in reproducible scripts

Deliverables:

```text
scripts/
  check_env.sh
  install_deps.sh
  setup_ttsim.sh
  build_tt_metal.sh
  verify_ttsim.sh
```

Also create:

```text
docs/environment.md
```

The file should record:

- Ubuntu version
- kernel version
- compiler version
- CMake version
- Python version
- TTSim commit/tag
- TT-Metal commit/tag
- known issues
- fixes applied

---

# 7. Phase 1 — Minimal NPU Kernel Execution

## Project 1: Integer / Vector Add

Start from the smallest possible working example.

Goals:

- Understand host-side program setup
- Understand device creation
- Understand command submission
- Understand buffer allocation
- Understand reader / compute / writer separation
- Validate device output against CPU result

Suggested progression:

### V0
Minimal official example

### V1
Custom vector-add input

```text
A + B -> C
```

### V2
Explicit buffer movement

```text
Host
  |
  v
Device DRAM
  |
  v
Reader Kernel
  |
  v
Circular Buffer / L1
  |
  v
Compute Kernel
  |
  v
Writer Kernel
  |
  v
Device DRAM
  |
  v
Host
```

Metrics / observations:

- buffer size
- tile size
- number of tiles
- synchronization points
- input/output correctness
- host/device command flow

Do not use simulator wall-clock time as NPU performance.

---

# 8. Phase 2 — Matrix Multiplication

Matrix multiplication is the first major optimization project.

## Goal

Rebuild concepts already learned in CUDA using the NPU programming model.

CUDA knowledge to transfer:

```text
Global memory
    ->
Shared memory
    ->
Registers / compute
    ->
Global memory
```

NPU-oriented view:

```text
DRAM
   ->
Data Movement / Reader
   ->
L1 / Circular Buffer
   ->
Compute Engine
   ->
L1 / Circular Buffer
   ->
Data Movement / Writer
   ->
DRAM
```

## Versions

### V0 — Reference

CPU / PyTorch reference.

### V1 — Naive implementation

Prioritize correctness.

### V2 — Tiled implementation

Introduce:

- tile decomposition
- L1 / Circular Buffer usage
- local data reuse

### V3 — Reduced redundant memory movement

Investigate whether the same A/B tiles are loaded repeatedly.

### V4 — Double buffering / pipelining

Conceptual target:

```text
time ---------------------------------------------------->

Load:      T0   T1   T2   T3
Compute:        T0   T1   T2   T3
Write:               T0   T1   T2   T3
```

Questions to document:

- Which operations can overlap?
- Which synchronization points remain unavoidable?
- What buffer capacity is required?
- How does tile size affect the execution structure?
- How is this different from CUDA streams or shared-memory tiling?

---

# 9. Phase 3 — Reduction

Implement a reduction operator such as:

```text
sum
mean
max
```

This phase exists because reduction forces reasoning about:

- partial results
- synchronization
- multiple cores
- intermediate buffers
- hierarchical reduction
- memory traffic

Compare with previous CUDA reduction exercises.

Document:

```text
CUDA:
thread -> warp -> block -> global

NPU:
tile -> local core result -> inter-core / output aggregation
```

Exact implementation will depend on supported TT-Metal APIs.

---

# 10. Phase 4 — RMSNorm Custom Operator

RMSNorm is the main custom operator candidate.

Formula:

```text
rms(x) = sqrt(mean(x^2) + eps)
y = x / rms(x) * weight
```

Internal stages:

```text
Input
  |
square
  |
reduction
  |
mean
  |
rsqrt
  |
multiply
  |
weight multiply
  |
Output
```

Why RMSNorm:

- small enough to understand completely
- combines reduction + elementwise operations
- representative of modern Transformer workloads
- requires intermediate data handling
- useful for runtime / kernel optimization study

## Implementations

1. PyTorch reference
2. CPU reference if useful
3. CUDA baseline
4. CUDA optimized version
5. TT-NN baseline
6. Custom TT-Metal / TT-NN operator if feasible

## Validation

For random inputs:

```text
max_abs_error
mean_abs_error
relative_error
```

Define tolerance explicitly based on datatype.

---

# 11. Phase 5 — Vision Workload

After low-level operators work, move to a small vision workload.

Preferred order:

1. Conv block
2. ResNet BasicBlock
3. MobileNet depthwise-separable block
4. Small CNN
5. Optional larger model analysis

## Recommended small CNN

Input:

```text
3 x 32 x 32
```

Example:

```text
Conv 3 -> 16
ReLU
Pool

Conv 16 -> 32
ReLU
Pool

Global Average Pool
Linear 32 -> 10
```

Possible dataset:

- CIFAR-10
- synthetic tensors for execution-only testing

Training quality is not the main goal.

The main goal is to understand operator execution.

---

# 12. Phase 6 — ResNet / MobileNet Block Analysis

Do not initially insist on running a full ResNet or YOLO end-to-end in TTSim.

Instead isolate one representative block.

## ResNet BasicBlock

```text
          +------------------------+
          |                        |
Input -> Conv -> Norm -> ReLU -> Conv -> Norm
          |                        |
          +---------- Add <--------+
                         |
                       ReLU
```

Analyze:

- intermediate tensor size
- convolution layout
- residual input lifetime
- data movement
- buffer reuse
- fusion opportunities
- layout conversion

## MobileNet Block

```text
Input
  |
Depthwise Conv
  |
Activation
  |
1x1 Pointwise Conv
  |
Output
```

Analyze:

- compute vs memory intensity
- arithmetic intensity
- tensor reuse
- layout
- tile shapes
- local memory demand

---

# 13. Phase 7 — Tiny Transformer Block

Optional but strongly recommended after RMSNorm / MatMul.

Example configuration:

```text
sequence_length = 128
hidden_size     = 256
num_heads       = 4
mlp_hidden      = 1024
batch_size      = 1
```

Block:

```text
Input
  |
RMSNorm
  |
QKV projection
  |
Attention
  |
Output projection
  |
Residual
  |
RMSNorm
  |
MLP
  |
Residual
```

Do not attempt to train a large LLM.

Use this to study:

- MatMul-heavy execution
- Softmax
- RMSNorm
- intermediate tensors
- operator scheduling
- memory-bound vs compute-bound behavior
- possible operator fusion

---

# 14. CUDA vs NPU Comparison Track

The project should explicitly compare accelerator programming concepts.

Example table to maintain:

| CUDA Concept | NPU / TT-Metal Concept | Notes |
|---|---|---|
| Global Memory | DRAM | Off-core larger memory |
| Shared Memory | L1 / Circular Buffer | Local data reuse |
| CUDA Thread Block | Tensix / compute core workload | Not one-to-one |
| Warp | Architecture-specific execution unit | Do not force exact equivalence |
| CUDA Stream | Command / runtime execution pipeline | Conceptually related, not identical |
| cudaMemcpyAsync | Data-movement kernels / NoC transfer | Different programming abstraction |
| Coalescing | Efficient tiled/sharded layout and transfers | Architecture-specific |
| __syncthreads | Kernel / buffer synchronization | Different scope and semantics |
| Shared-memory tiling | L1 / Circular Buffer tiling | Similar optimization principle |

Important:

Do not claim exact one-to-one equivalence where the architectures differ.

---

# 15. Benchmark Strategy

## CUDA

CUDA runs on real MX150 hardware.

Allowed real measurements:

- CUDA Event timing
- kernel latency
- host-to-device transfer
- device-to-host transfer
- end-to-end latency
- effective bandwidth where meaningful
- CPU vs CUDA comparison

Record:

```text
warmup count
measurement count
mean
median
min
max
standard deviation
```

## TTSim

Do not report simulator time as real NPU hardware performance.

Track structural metrics instead:

- tile count
- buffer count
- buffer size
- DRAM read/write pattern
- duplicated loads
- synchronization point count
- number of kernels
- number of device cores
- layout changes
- intermediate tensor size
- numerical correctness

If a simulator exposes useful architectural counters, record them separately and describe exactly what they mean.

---

# 16. Repository Structure

Initial proposed layout:

```text
accelflow/
|
+-- README.md
+-- PROJECT.md
+-- LICENSE
+-- .gitignore
|
+-- docs/
|   +-- environment.md
|   +-- architecture.md
|   +-- cuda-vs-npu.md
|   +-- measurement-policy.md
|   +-- troubleshooting.md
|   +-- portfolio-notes.md
|
+-- scripts/
|   +-- check_env.sh
|   +-- install_deps.sh
|   +-- setup_ttsim.sh
|   +-- build_tt_metal.sh
|   +-- verify_ttsim.sh
|
+-- cuda/
|   +-- vector_add/
|   +-- matmul/
|   +-- reduction/
|   +-- rmsnorm/
|   +-- benchmark/
|
+-- npu/
|   +-- vector_add/
|   +-- data_movement/
|   +-- matmul/
|   +-- reduction/
|   +-- rmsnorm/
|   +-- experiments/
|
+-- models/
|   +-- mini_cnn/
|   +-- resnet_block/
|   +-- mobilenet_block/
|   +-- tiny_transformer/
|
+-- validation/
|   +-- reference/
|   +-- numerical/
|
+-- benchmark/
|   +-- cuda/
|   +-- ttsim/
|   +-- results/
|
+-- tools/
|   +-- plotting/
|   +-- report/
|
+-- tests/
|
+-- third_party/
|   +-- README.md
|
+-- results/
    +-- figures/
    +-- tables/
    +-- logs/
```

Do not copy large external repositories directly into this repository unless necessary.

Prefer:

- git submodule
- documented external clone
- environment variable pointing to external TT-Metal checkout

---

# 17. Coding Standards

## C++

Target:

- modern C++
- C++17 or C++20 depending on upstream requirement
- RAII where possible
- avoid unnecessary macros
- explicit ownership
- error checking

## CUDA

Always:

- check CUDA API errors
- check kernel launch errors
- separate warm-up from measurement
- avoid including allocation in kernel-only timing unless explicitly measuring end-to-end execution

## Python

Use for:

- PyTorch references
- tensor generation
- numerical validation
- result parsing
- graphs / reports

## Shell

Scripts should:

- use `set -euo pipefail`
- print meaningful error messages
- be idempotent when practical
- avoid destructive changes
- detect missing dependencies

---

# 18. Reproducibility

Every experiment should record:

```text
date
git commit
OS
compiler
Python version
CUDA version
GPU
TTSim version
TT-Metal version
operator
tensor shape
dtype
parameters
result
```

Recommended format:

```text
results/<experiment>/<timestamp>/metadata.json
```

Example:

```json
{
  "backend": "cuda",
  "operator": "rmsnorm",
  "shape": [32, 1024],
  "dtype": "fp32",
  "warmup": 20,
  "iterations": 100
}
```

---

# 19. Testing Strategy

Minimum tests:

## Correctness

- deterministic small tensor
- random tensor
- boundary shape
- non-multiple-of-tile shape where supported
- zero input
- negative values
- large magnitude values if numerically safe

## Regression

Each optimized version must produce equivalent output to its reference within tolerance.

Suggested hierarchy:

```text
PyTorch reference
        |
        +-- CUDA implementation
        |
        +-- TT-NN implementation
        |
        +-- custom NPU implementation
```

---

# 20. Documentation Strategy

For every major optimization, document this structure:

## Problem

What is inefficient?

Example:

```text
The K-loop reloads the same tile from DRAM for each output tile.
```

## Evidence

What code or execution structure demonstrates it?

## Hypothesis

Why is it inefficient?

## Change

What was modified?

## Expected effect

Example:

```text
Reduce redundant DRAM reads by retaining the tile in local memory.
```

## Validation

- numerical correctness
- structural metric difference
- CUDA measurement if applicable

## Limitations

State what could not be measured without real NPU hardware.

---

# 21. Portfolio Deliverables

The final project should produce:

## 1. GitHub Repository

Readable by an interviewer without local setup.

## 2. Architecture Diagram

Show:

```text
Model
  ->
Framework
  ->
Runtime / Operator
  ->
Command / Buffer
  ->
Data Movement
  ->
Compute Kernel
  ->
Output
```

## 3. Optimization Case Studies

At least three.

Recommended:

1. MatMul tiling and data reuse
2. MatMul or operator pipelining / double buffering
3. RMSNorm custom operator

Optional:

4. ResNet / MobileNet block
5. Tiny Transformer execution analysis

## 4. CUDA vs NPU Comparison

Explain what optimization concepts transfer across accelerators.

## 5. Performance / Validation Report

Clearly separate:

```text
Real GPU measurement
vs
NPU simulator functional analysis
```

---

# 22. Suggested Portfolio Story

A strong final summary may look like:

> I initially studied CUDA through memory coalescing, shared-memory tiling, warp-level reduction, synchronization, streams, and matrix operations. I then reproduced similar accelerator optimization problems using TT-Metalium and TTSim without access to physical NPU hardware. I implemented and analyzed data-movement kernels, tiled matrix multiplication, buffering and synchronization, and a custom RMSNorm-style operator, while validating numerical correctness against PyTorch. Real latency measurements were performed only on actual CUDA hardware, while TTSim results were used strictly for functional and execution-structure analysis.

This story is more important than simply having many operators.

---

# 23. First Development Milestone

The first milestone is complete when all of the following are true:

```text
[ ] WSL2 Ubuntu 24.04 verified
[ ] system dependency versions recorded
[ ] TTSim installed or built
[ ] TT-Metalium installed or built
[ ] virtual Wormhole device selected
[ ] minimal official TT-Metal example executed
[ ] output validated
[ ] setup scripts committed
[ ] environment.md completed
[ ] known limitations documented
```

Do not proceed to complex operators until this milestone is reproducible.

---

# 24. Second Development Milestone

```text
[ ] vector add works on TTSim
[ ] buffer/data movement flow documented
[ ] CPU reference validates output
[ ] first custom kernel is committed
[ ] architecture diagram created
```

---

# 25. Third Development Milestone

```text
[ ] MatMul baseline implemented
[ ] tiled version implemented
[ ] local data reuse explained
[ ] pipelined or double-buffered version attempted
[ ] correctness regression tests pass
[ ] CUDA MatMul comparison available
```

---

# 26. Fourth Development Milestone

```text
[ ] RMSNorm PyTorch reference
[ ] RMSNorm CUDA baseline
[ ] RMSNorm NPU / TT-NN baseline
[ ] custom operator path investigated
[ ] numerical error report
[ ] optimization report
```

---

# 27. Stretch Goals

Only after the core project is stable:

- TT-MLIR graph inspection
- StableHLO / TTIR / TTNN IR tracing
- custom compiler pass exploration
- INT8 quantization experiment
- operator fusion analysis
- multi-core scheduling experiment
- multi-device concepts
- actual remote Tenstorrent hardware validation
- ResNet18 / MobileNet end-to-end execution where practical
- YOLO operator/block analysis
- profiling visualization dashboard

---

# 28. What Not To Do Initially

Avoid these at project start:

- training a large LLM
- running a full YOLO model before basic kernels work
- writing a complete NPU driver
- implementing a compiler from scratch
- attempting MLIR passes before understanding TT-Metal execution
- claiming TTSim performance equals silicon
- spending weeks reproducing library-quality GEMM
- optimizing without a correctness reference

Priority:

```text
Correctness
    ->
Execution understanding
    ->
Measurement
    ->
Optimization
    ->
Model integration
```

---

# 29. Instructions for Codex — Initial Repository Setup

Codex should perform the following initial work.

## Step 1

Create the repository skeleton from the structure described above.

Do not download or vendor TT-Metal yet unless necessary.

## Step 2

Create:

```text
scripts/check_env.sh
```

It should print:

- OS
- WSL detection
- CPU architecture
- available CPU cores
- RAM
- disk
- Python
- GCC/G++
- Clang
- CMake
- Ninja
- Git
- Docker availability
- NVIDIA/CUDA visibility if any

It should not modify the system.

## Step 3

Create:

```text
scripts/install_deps.sh
```

Target Ubuntu 24.04.

Before installing anything:

- clearly print planned packages
- use apt packages where practical
- avoid replacing system Python
- use a virtual environment for Python tooling
- do not install GPU/NPU kernel drivers because TTSim does not require physical Tenstorrent hardware

## Step 4

Create:

```text
scripts/setup_ttsim.sh
```

Requirements:

- use official Tenstorrent TTSim repository/release
- allow chip target selection through an environment variable
- default to Wormhole
- prefer pinned version/tag if a compatible version can be identified
- store external source under a clearly defined external directory such as:

```text
$HOME/dev/tenstorrent/
```

Do not put the entire upstream repository inside this project.

## Step 5

Create:

```text
scripts/setup_tt_metal.sh
```

Requirements:

- clone official TT-Metal repo with required submodules
- record commit hash
- avoid assuming physical `/dev/tenstorrent` devices
- configure for simulator usage based on current official TTSim documentation
- do not guess environment variables; verify against upstream documentation/source

## Step 6

Create:

```text
scripts/verify_ttsim.sh
```

It should:

1. verify TTSim library exists
2. verify TT-Metal build exists
3. set simulator environment
4. execute the smallest supported TT-Metal programming example
5. return non-zero exit code on failure
6. write a log under:

```text
results/setup/
```

## Step 7

Create:

```text
docs/environment.md
docs/measurement-policy.md
docs/troubleshooting.md
```

Include clear placeholder sections for detected environment information.

## Step 8

Create a root `README.md` containing:

- project purpose
- architecture overview
- quick start
- current milestone
- simulator limitation warning
- roadmap

---

# 30. Codex Safety / Accuracy Rules

Codex must follow these rules while setting up the project:

1. Do not invent TT-Metal or TTSim commands.
2. Check upstream documentation before writing simulator-specific environment variables.
3. Prefer current official Tenstorrent documentation and source.
4. Do not install Tenstorrent kernel drivers unless physical hardware actually requires them.
5. Do not modify Windows-side CUDA configuration.
6. Do not remove existing CUDA / NVIDIA installations.
7. Do not overwrite existing WSL configuration without an explicit reason.
8. Do not treat simulator timing as hardware performance.
9. Pin external versions when reproducibility benefits from it.
10. Keep setup scripts readable enough for a portfolio reviewer.

---

# 31. Suggested Codex First Prompt

Use this after giving Codex this PROJECT.md file:

```text
Read PROJECT.md completely before changing anything.

We are starting only Phase 0: environment verification and TTSim/TT-Metal bootstrap.

Environment:
- Windows host
- WSL2
- Ubuntu 24.04
- x86_64
- no physical Tenstorrent accelerator
- local NVIDIA MX150 exists, but it is unrelated to TTSim setup

Tasks:
1. Create the repository skeleton described in PROJECT.md.
2. Implement scripts/check_env.sh first.
3. Run it and save the environment report.
4. Inspect current official Tenstorrent TTSim and TT-Metal documentation before writing install commands.
5. Create an installation plan for TTSim + TT-Metal targeting a virtual Wormhole device.
6. Do not install packages or modify the system until after showing the plan and identifying any compatibility risks.
7. Never assume simulator wall-clock time represents real NPU performance.

Keep each change small and commit-ready.
```

---

# 32. Immediate Next Action

Start only here:

```bash
cd ~
mkdir -p ~/dev/accelflow
cd ~/dev/accelflow
```

Then provide this file to Codex and let Codex begin with:

```text
Phase 0
  ->
environment inspection
  ->
version compatibility check
  ->
reproducible setup
  ->
minimal TTSim example
```

Do not start model work before the basic simulator path is verified.

---

# 33. Official References

Use official sources as the primary technical references:

- TTSim:
  - https://github.com/tenstorrent/ttsim

- TT-Metal / TT-NN:
  - https://github.com/tenstorrent/tt-metal

- TT-Metal installation:
  - https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md

- TT-MLIR:
  - https://github.com/tenstorrent/tt-mlir

- Tenstorrent documentation:
  - https://docs.tenstorrent.com/

When upstream instructions differ from this project document, treat the current official upstream documentation as authoritative and update this document accordingly.

---

# 34. Success Criteria

This project is successful if an interviewer can see evidence that the developer can:

- understand an accelerator execution model
- reason about memory hierarchy
- implement kernels
- reason about tiling
- reason about synchronization
- reason about data movement
- build asynchronous/pipelined execution
- validate numerical correctness
- compare GPU and NPU programming models
- understand runtime/operator/model boundaries
- state measurement limitations correctly
- communicate optimization work clearly

The purpose is not to prove ownership of NPU hardware.

The purpose is to prove **accelerator systems engineering ability**.
