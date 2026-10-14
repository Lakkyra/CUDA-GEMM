# Product Requirement Document (PRD) 1: High-Performance CUDA GEMM Engine

## 1. Document Overview & Objective

* **Project Name:** High-Performance CUDA General Matrix Multiply (GEMM) Engine
* **Objective:** Design, implement, and benchmark an optimized dense FP32 matrix multiplication kernel ($C = \alpha \cdot AB + \beta \cdot C$) in CUDA C++, progressively scaling from a naive baseline to an architecture-aware kernel approaching `cuBLAS` performance.


* **Primary Target Role:** NVIDIA Compute Developer Technology (DevTech) / Compute Architecture.



---

## 2. Target Scope & Hardware Specifications

* **Target Architecture:** NVIDIA Ampere / Ada Lovelace / Hopper (SM 8.0, SM 8.9, SM 9.0).
* **Matrix Dimensions ($M \times N \times K$):** Powers of two ($512$ to $8192$), arbitrary non-power-of-two dimensions with boundary guards, and tall/skinny configurations.
* **Precision:** IEEE 754 Single-Precision Floating-Point (FP32).

---

## 3. Technical Requirements & Engineering Milestones

### Phase 1: Baseline & Memory Coalescing

* **Kernel 0 (Naive Implementation):** One thread computes one element of matrix $C$. Direct global memory loads for every multiply-accumulate operation.
* **Kernel 1 (Global Memory Coalescing):** Restructure thread mapping so adjacent threads in a warp access consecutive addresses in global memory, maximizing 128-byte cache line transactions.

### Phase 2: Shared Memory Exploitation & Tiling

* **Kernel 2 (2D Shared Memory Tiling):**
* Partition matrix into square tiles (e.g., $32 \times 32$).
* Cooperatively stage tiles from high-latency global memory into low-latency shared memory (`__shared__`).
* Enforce block synchronization barriers (`__syncthreads()`) between phases.


* **Kernel 3 (Shared Memory Bank Conflict Elimination):**
* Pad shared memory arrays or swizzle access indices to eliminate 32-way shared memory bank conflicts on 32-bit word accesses.



### Phase 3: Register Level Optimization & Instruction-Level Parallelism (ILP)

* **Kernel 4 (Register Tiling / 2D Thread Tiling):**
* Each thread computes a sub-tile of matrix $C$ (e.g., $8 \times 8$ elements per thread) cached directly in local registers.
* Reduce shared memory read operations by a factor of 8 relative to compute operations.


* **Kernel 5 (Vectorized Loads & Double Buffering):**
* Use 128-bit vectorized loads (`float4` / `LDG.E.128`) for global and shared memory data movement.
* Implement double buffering (software pipelining) to overlap global-to-shared memory loads with execution on current data.



---

## 4. Verification & Performance Criteria

| Metric | Target / Benchmark |
| --- | --- |
| **Numerical Accuracy** | Maximum absolute error $\vert{}C_{calc} - C_{ref}\vert{} \le 10^{-5}$ compared against host CPU reference and `cuBLAS`. |
| **Performance Target** | Achieve $\ge 80\%$ of `cuBLAS` FP32 performance on square matrices ($N \ge 4096$). |
| **Warp Execution Efficiency** | $\ge 90\%$ (Nsight Compute). |
| **Memory Throughput** | Near-saturation of target GPU DRAM sustained bandwidth during global memory loading phases. |

---

## 5. Profiling & Telemetry Configuration

* **Profiling Tool:** NVIDIA Nsight Compute (`ncu` CLI).


* **Key Profiling Metrics Captured:**
* `dram__bytes_read.sum`, `dram__bytes_write.sum` (Global memory traffic)
* `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (L1 cache transactions)
* `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` (Memory latency stalls)
* `smsp__warp_issue_stalled_barrier_per_warp_active.pct` (Thread synchronization overhead)



---

## 6. Deliverables & Repository Structure

```text
cuda-gemm-engine/
├── CMakeLists.txt
├── benchmarks/
│   ├── run_benchmark.py
│   └── plot_roofline.py
├── include/
│   ├── gemm_kernels.cuh
│   └── utils.cuh
├── profile/
│   ├── ncu_reports/
│   └── speedup_curves.png
└── src/
    ├── kernels/
    │   ├── 00_naive.cu
    │   ├── 01_coalesced.cu
    │   ├── 02_smem_tiled.cu
    │   ├── 03_bank_resolved.cu
    │   ├── 04_reg_tiled.cu
    │   └── 05_vectorized_pipelined.cu
    └── main.cu

```

---

---

