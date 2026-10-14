# High-Performance CUDA GEMM Engine

An optimized dense single-precision floating-point General Matrix Multiplication (GEMM, $C = \alpha \cdot AB + \beta \cdot C$) engine implemented in CUDA C++. Designed and engineered according to NVIDIA Compute Developer Technology (DevTech) and Compute Architecture standards.

The engine scales from an uncoalesced naive baseline to an architecture-aware kernel approaching `cuBLAS` performance on modern NVIDIA architectures (Ampere, Ada Lovelace, Hopper).

---

## 1. Architectural Evolution & Optimization Phases

```
+-----------------------------------------------------------------------------------------+
| Phase 1: Baseline & Memory Coalescing                                                  |
|   ├── Kernel 0: Naive uncoalesced global memory accesses (O(1) arithmetic intensity)     |
|   └── Kernel 1: Warp-coalesced memory mapping (128-byte cache line transactions)         |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
| Phase 2: Shared Memory Exploitation & Tiling                                            |
|   ├── Kernel 2: 2D Shared Memory Tiling (32x32 tiles, __syncthreads() staging barriers)   |
|   └── Kernel 3: Shared Memory Bank Conflict Elimination (+1 padding, 32-way conflict-free)|
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
| Phase 3: Register Level Optimization & Instruction-Level Parallelism (ILP)              |
|   ├── Kernel 4: 2D Register Tiling (128x128 macro-tile, 8x8 micro-tile per thread)      |
|   └── Kernel 5: Vectorized 128-bit loads (float4) + Double Buffering (Software Pipeline) |
+-----------------------------------------------------------------------------------------+
                                           │
                                           ▼
+-----------------------------------------------------------------------------------------+
| Verification & Target Benchmark: NVIDIA cuBLAS (cublasSgemm) Reference                  |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```text
cuda-gemm-engine/
├── CMakeLists.txt              # Production CMake build specification
├── README.md                   # System documentation & performance analysis
├── benchmarks/
│   ├── run_benchmark.py        # Automated benchmarking driver (M=N=K sweeps)
│   └── plot_roofline.py        # Roofline model and speedup curve generator
├── include/
│   ├── gemm_kernels.cuh        # Kernel prototypes and metadata catalog
│   └── utils.cuh               # Error checking, GpuTimer, CPU GEMM, numerical verification
├── profile/
│   ├── ncu_profiler.py         # Nsight Compute (ncu) profiling automated harness
│   ├── README.md               # Telemetry and metrics analysis guide
│   ├── ncu_reports/            # NCU output report directory
│   └── speedup_curves.png      # Generated Roofline and performance plots
└── src/
    ├── kernels/
    │   ├── 00_naive.cu                 # Phase 1: Uncoalesced baseline
    │   ├── 01_coalesced.cu             # Phase 1: Coalesced memory mapping
    │   ├── 02_smem_tiled.cu            # Phase 2: 2D Shared memory tiling
    │   ├── 03_bank_resolved.cu         # Phase 2: Bank conflict elimination
    │   ├── 04_reg_tiled.cu             # Phase 3: Register micro-kernel (8x8)
    │   └── 05_vectorized_pipelined.cu  # Phase 3: float4 vectorization & double buffering
    └── main.cu                         # CLI driver & benchmark execution engine
```

---

## 3. Hardware Requirements & Supported Architectures

* **Precision:** IEEE 754 Single-Precision Floating-Point (FP32).
* **Dimensions ($M \times N \times K$):** Powers of two (512 to 8192), arbitrary non-powers of two with boundary padding, and tall/skinny configurations.
* **Target Architectures:**
  * NVIDIA Ampere: `sm_80` (A100), `sm_86` (RTX 30-series)
  * NVIDIA Ada Lovelace: `sm_89` (RTX 40-series, L40)
  * NVIDIA Hopper: `sm_90` (H100)

---

## 4. Building the Project

### Prerequisites
* NVIDIA CUDA Toolkit 11.8+ or 12.x
* CMake 3.20+
* C++17 compatible host compiler (MSVC 2019/2022 on Windows, GCC 9+ or Clang 11+ on Linux)
* Python 3.9+ with `numpy` and `matplotlib` (for benchmarking & plotting)

### Build Commands
```bash
# Configure build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile all kernels and benchmark executable
cmake --build build --config Release -j
```

---

## 5. Usage & CLI Driver

```bash
# Run verification across all kernels on 1024x1024 matrices against CPU reference
./build/gemm_benchmark --verify --dim 1024 1024 1024

# Benchmark all kernels on a 4096 square matrix with 10 iterations
./build/gemm_benchmark --benchmark --size 4096 --iterations 10

# Benchmark a specific kernel (e.g., Kernel 5: Vectorized Pipelined)
./build/gemm_benchmark --kernel 5 --size 4096

# Run full automated benchmark suite with Python
python benchmarks/run_benchmark.py --executable ./build/gemm_benchmark

# Generate Roofline model and speedup comparison charts
python benchmarks/plot_roofline.py
```

---

## 6. Profiling with NVIDIA Nsight Compute (`ncu`)

Run profiling using the integrated Python harness:
```bash
python profile/ncu_profiler.py --executable ./build/gemm_benchmark --size 2048
```

Key profiling metrics captured:
1. `dram__bytes_read.sum`, `dram__bytes_write.sum`: Global DRAM memory traffic.
2. `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: L1 cache sector transactions.
3. `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`: Latency stalls due to memory transactions.
4. `smsp__warp_issue_stalled_barrier_per_warp_active.pct`: Overhead from `__syncthreads()` barrier waits.
