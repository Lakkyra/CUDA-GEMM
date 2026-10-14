#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <functional>

// Forward declarations of host launcher routines for all GEMM implementations

/**
 * Phase 1: Baseline & Memory Coalescing
 */

// Kernel 0: Naive baseline where each thread computes 1 element of C.
// Thread mapping: threadIdx.x -> row (m), threadIdx.y -> col (n).
// Causes uncoalesced memory reads from matrix B across threads in a warp.
void run_gemm_naive(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

// Kernel 1: Memory Coalescing.
// Restructures thread mapping so adjacent warp threads (threadIdx.x) access
// consecutive columns (consecutive memory addresses in row-major layout).
void run_gemm_coalesced(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

/**
 * Phase 2: Shared Memory Exploitation & Tiling
 */

// Kernel 2: 2D Shared Memory Tiling.
// Stages 32x32 tiles of matrix A and B into on-chip shared memory (__shared__),
// amortizing global memory accesses across the thread block.
void run_gemm_smem_tiled(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

// Kernel 3: Shared Memory Bank Conflict Elimination.
// Uses padding (+1 float) in shared memory arrays to eliminate 32-way bank conflicts
// during column/row traversals within 32-bank hardware shared memory.
void run_gemm_bank_resolved(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

/**
 * Phase 3: Register Level Optimization & Instruction-Level Parallelism (ILP)
 */

// Kernel 4: Register Tiling / 2D Thread Micro-Kernel.
// Macro-tile (128x128), BK=8. Each thread computes an 8x8 sub-tile of C
// entirely resident in local register files, reducing shared memory access
// pressure by up to 8x.
void run_gemm_reg_tiled(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

// Kernel 5: Vectorized Loads (128-bit float4) and Double Buffering.
// Uses LDG.E.128 vector memory instructions and software pipelining (ping-pong
// shared memory staging) to overlap global-to-shared data movement with FMA compute.
void run_gemm_vectorized_pipelined(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream = 0
);

/**
 * cuBLAS Baseline
 */

// cuBLAS reference GEMM wrapper for FP32 row-major C = alpha*A*B + beta*C
void run_gemm_cublas(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C
);

// Kernel description structure
struct KernelMeta {
    int id;
    std::string tag;
    std::string name;
    std::string description;
    std::string optimization_focus;
};

inline const std::vector<KernelMeta>& get_kernel_catalog() {
    static const std::vector<KernelMeta> catalog = {
        {0, "00_naive", "Kernel 0: Naive Baseline",
         "1 thread / element, uncoalesced global reads, direct MAC operations.",
         "Baseline arithmetic intensity O(1)"},
        {1, "01_coalesced", "Kernel 1: Global Coalesced",
         "Warp lanes aligned to row-major column stride, 128B coalesced burst loads.",
         "Memory Coalescing & L1 Warp Broadcast"},
        {2, "02_smem_tiled", "Kernel 2: 2D Shared Memory Tiled",
         "32x32 on-chip shared memory tiles with __syncthreads() staging barriers.",
         "Data Reuse & Latency Reduction"},
        {3, "03_bank_resolved", "Kernel 3: Bank Conflict Resolved",
         "Padded shared memory allocations (32x33) removing 32-way bank stalls.",
         "Shared Memory Bank Conflict Elimination"},
        {4, "04_reg_tiled", "Kernel 4: 2D Register Tiled (8x8)",
         "Macro-tile 128x128, micro-tile 8x8 per thread in registers; 8x SMEM bandwidth reduction.",
         "Register File Caching & Instruction-Level Parallelism (ILP)"},
        {5, "05_vectorized_pipelined", "Kernel 5: Vectorized & Double Buffered",
         "128-bit float4 vectorized loads + ping-pong software pipelined memory prefetching.",
         "Asynchronous Latency Hiding & Peak Throughput"},
        {6, "cublas", "NVIDIA cuBLAS (Reference)",
         "Production vendor GEMM library utilizing hand-crafted assembly / architecture kernels.",
         "NVIDIA Vendor Target Benchmark"}
    };
    return catalog;
}
