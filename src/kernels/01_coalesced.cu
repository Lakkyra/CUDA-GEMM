#include "gemm_kernels.cuh"
#include "utils.cuh"

/**
 * Kernel 1: Global Memory Coalescing
 *
 * Characteristics & Optimization Focus:
 * - Restructures the thread coordinate mapping to align with row-major memory layouts:
 *     threadIdx.x maps to column index (n) -> fast-varying dimension.
 *     threadIdx.y maps to row index (m)    -> slow-varying dimension.
 * - Coalescing Mechanism:
 *     In row-major matrix B of shape [K, N], consecutive columns (col, col+1, ...) are stored
 *     contiguously in physical DRAM.
 *     Because a warp consists of 32 threads with consecutive threadIdx.x (lanes 0..31),
 *     accessing B[k * N + col] requests 32 * 4 bytes = 128 contiguous bytes in parallel.
 *     This perfectly saturates an entire 128-byte L1/L2 cache line in a single memory transaction.
 * - Broadcast Mechanism:
 *     For matrix A of shape [M, K], all 32 threads sharing the same threadIdx.y access the
 *     same element A[row * K + k]. Hardware L1 warp broadcast delivers this value to all 32 lanes
 *     simultaneously without redundant memory requests.
 * - Global write-back to C[row * N + col] is likewise 100% coalesced.
 */
__global__ void gemm_coalesced_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Restructured mapping: threadIdx.x traverses columns (consecutive memory)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row * K + k] is broadcasted across the warp row
            // B[k * N + col] is coalesced across adjacent warp lanes (128-byte transaction)
            accum += A[row * K + k] * B[k * N + col];
        }

        // Coalesced write-back
        if (beta != 0.0f) {
            C[row * N + col] = alpha * accum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * accum;
        }
    }
}

void run_gemm_coalesced(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    // 32x32 = 1024 threads per block
    dim3 blockDim(32, 32);
    // gridDim.x covers columns (N), gridDim.y covers rows (M)
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    gemm_coalesced_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
