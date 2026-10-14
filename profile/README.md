# Profiling & Telemetry Configuration

This directory contains automated profiling harnesses and analysis guides for **NVIDIA Nsight Compute (`ncu`)** per Section 5 of the PRD.

---

## 1. Mandatory PRD Metrics & Architectural Interpretation

| Metric | Full NCU Hardware Event Name | Hardware Architecture Meaning | Optimization Target |
| :--- | :--- | :--- | :--- |
| **Global DRAM Reads** | `dram__bytes_read.sum` | Cumulative physical bytes transferred from Device DRAM over the memory controller bus. | Must drop by $\approx 32\times$ moving from Kernel 1 (coalesced) to Kernel 2 (shared memory tiled). |
| **Global DRAM Writes** | `dram__bytes_write.sum` | Cumulative physical bytes committed to Device DRAM. | Should match theoretical matrix $C$ size ($M \times N \times 4$ bytes). |
| **L1 / Texture Sector Requests** | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 32-byte memory sectors requested through the Load/Store Units (LSU). | Coalesced loads (Kernel 1) reduce sector counts by $4\times$ to $32\times$ over Kernel 0. |
| **Long Scoreboard Stalls** | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | Percentage of active warp cycles stalled waiting for DRAM global memory latency. | Double buffering & software pipelining (Kernel 5) hides this behind arithmetic execution. |
| **Barrier Stalls** | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | Percentage of active cycles stalled at `__syncthreads()` barrier instructions. | Optimized tile sizing and decoupled stages minimize warp barrier divergence. |

---

## 2. Running Automated NCU Profiling

Run NCU on a compiled binary for any kernel (e.g. matrix size $N=2048$):

```bash
# Profile all kernels and export both .ncu-rep and CSV telemetry:
python profile/ncu_profiler.py --executable ./build/Release/gemm_benchmark.exe --size 2048 --kernel all

# Profile only Kernel 5 (Vectorized & Pipelined):
python profile/ncu_profiler.py --executable ./build/Release/gemm_benchmark.exe --size 2048 --kernel 5
```

Direct CLI invocation:
```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct \
    --export profile/ncu_reports/k5_n2048 \
    ./build/Release/gemm_benchmark.exe --size 2048 --kernel 5 --iterations 1 --warmup 1 --no-verify
```

---

## 3. Speedup Curves and Roofline Model

Generate the 3-panel publication visualization:
```bash
python benchmarks/plot_roofline.py
```
This produces `profile/speedup_curves.png`.
