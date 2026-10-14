#include "gemm_kernels.cuh"
#include "utils.cuh"

/**
 * Kernel 0: Naive Implementation
 *
 * Characteristics:
 * - 1 thread computes exactly 1 element of matrix C: C[row, col].
 * - Every Multiply-Accumulate (MAC) operation executes 2 global memory loads (A[row, k] and B[k, col]).
 * - Thread mapping:
 *     threadIdx.x maps to the row dimension (m).
 *     threadIdx.y maps to the col dimension (n).
 * - Performance Bottleneck (Uncoalesced Global Memory Access):
 *     In CUDA warp execution, adjacent threads (lane 0 to lane 31) share the same threadIdx.y
 *     and vary by threadIdx.x.
 *     Thus, when loading A[row * K + k], adjacent warp lanes access addresses separated by
 *     a stride of K elements (K * 4 bytes). If K >= 32, this triggers 32 separate 32-byte
 *     memory sector transactions per warp load rather than 1 coalesced 128-byte transaction!
 * - Global memory bandwidth is heavily underutilized; arithmetic intensity is O(1).
 */
__global__ void gemm_naive_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Deliberate naive mapping: threadIdx.x traverses rows
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for non-power-of-two or arbitrary matrix dimensions
    if (row < M && col < N) {
        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // Direct uncoalesced global memory reads in the inner accumulation loop
            accum += A[row * K + k] * B[k * N + col];
        }

        // Write out with alpha and beta scaling
        if (beta != 0.0f) {
            C[row * N + col] = alpha * accum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * accum;
        }
    }
}

void run_gemm_naive(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    // 32x32 = 1024 threads per block (maximum supported per block on modern architectures)
    dim3 blockDim(32, 32);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    gemm_naive_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
